import asyncio
import hashlib
import json
import re
import time
from datetime import datetime, timezone
from uuid import uuid4

import redis.asyncio as aioredis
import structlog

from ..validation import ResponseValidator, ValidationResult, attempt_recovery
from ..validation.recovery_engine import check_screenshot_completeness
from ..agent.context import build_context, WEB_FEWSHOT
from ..db.models import AuditLog
from ..db.session import async_session
from ..memory.manager import MemoryManager
from ..memory.types import MemoryQuery, MemoryType
from ..memory.learning import detect_feedback, store_example, get_positive_examples, format_learned_examples
from ..memory.knowledge_graph import extract_from_conversation as kg_extract
from ..agent.self_model import record_message_processed as sm_record_message
from ..memory.temporal import extract_from_text as temporal_extract
from ..agent.epistemic import record_outcome as epistemic_record
from ..utils.redaction import redact
from ..models.manager import ModelManager
from ..models.types import Message, ModelRequest
from ..skills.parser import parse_skill_calls, strip_skill_calls
from ..skills.registry import SkillRegistry
from ..skills.executor import SkillExecutor
from ..skills.types import SkillResult
from ..skills.auto_detect import detect_skills
from ..decision_layer import Strategy, decide_execution_strategy, is_scheduling_request
from ..intent.action_classifier import classify_action_intent, ActionIntent as _ActionIntent
from ..intent.execution_planner import generate_plan, format_plan_for_prompt, update_plan_from_action_history, ExecutionPlan as _ExecutionPlan
from ..intent.plan_executor import PlanExecutor as _PlanExecutor, PlanStepExecution as _PlanStepExecution
from ..skills.openclaw.clawhub_client import get_client as get_clawhub_client
from ..skills.openclaw.loader import load_installed_skills, get_skills_dir, check_requirements
from .bus import EventBus
from .types import EventType

LAST_ACTIVE_KEY = "agent:last_active"

logger = structlog.get_logger()

import os

from ..config import settings as _settings

def _max_skill_rounds() -> int:
    return 12 if _settings.sovereign_mode else 8

# Hard cap on meaningful correction injections per request. Above this, return
# the best response so far instead of continuing to grind correction prompts.
MAX_MEANINGFUL_CORRECTIONS = 3


# ──────────────────────────────────────────────────────────────────────────
# Central enforcement policy — single source of truth.
# All intent gating, schedule honesty, side-effect text scrubbing, and
# placeholder validation live in src.policy. handlers.py only holds the
# stateful pieces (recent-action tracker per chat, request budget) and thin
# wrappers that bind the policy functions to that state.
# ──────────────────────────────────────────────────────────────────────────

from ..policy import (
    INTENT_GATE_PATTERNS as _INTENT_GATE_PATTERNS,
    REFERENCE_PHRASE_RE as _REFERENCE_PHRASE_RE,
    EMAIL_ADDR_RE as _EMAIL_ADDR_RE,
    SIDE_EFFECT_SKILLS as _SIDE_EFFECT_SKILLS,
    SKILL_SAFE_ACTIONS as _SKILL_SAFE_ACTIONS,
    SIDE_EFFECT_ANNOUNCEMENT_PATTERNS as _SIDE_EFFECT_ANNOUNCEMENT_PATTERNS,
    TIME_CLAIM_RE as _TIME_CLAIM_RE,
    is_placeholder_subject as _is_placeholder_subject,
    is_placeholder_body as _is_placeholder_body,
    user_message_provides_content as _user_message_provides_content,
    intent_gate_check as _policy_intent_gate_check,
    filter_inferred_side_effects as _policy_filter_inferred_side_effects,
    enforce_schedule_honesty as _policy_enforce_schedule_honesty,
    enforce_side_effect_text_gate as _policy_enforce_side_effect_text_gate,
    extract_fixed_time_unhonored as _extract_fixed_time_unhonored,
    has_real_task_create as _has_real_task_create,
    apply_final_response_policy as _apply_final_response_policy,
    new_trace as _new_decision_trace,
    record_trace as _record_decision_trace,
)

# ── Recent-action tracker (state) ─────────────────────────────────────────
# Records the last EXPLICIT side-effect action per chat. Lets follow-ups like
# "haz lo mismo" / "send it again" pass when the previous turn had an explicit
# action of the same type. TTL keeps the window short (5 min) so older context
# never authorizes new side-effects.
import threading as _intent_threading
_LAST_EXPLICIT_ACTION: dict = {}        # chat_id -> {skill, action, recipient, ts}
_LAST_EXPLICIT_ACTION_LOCK = _intent_threading.Lock()
_LAST_EXPLICIT_TTL_S = 300              # 5 minutes


# ── Recent-action storage: Redis-backed with L1 memory cache ─────────────
# Memory dict above keeps zero-latency reads in single-replica deployments.
# Redis adds multi-replica safety: a follow-up "haz lo mismo" sent to a
# different replica still resolves the previous turn's explicit action.
# Reads check memory first; write paths update both. Failures on the Redis
# side never block the request — the in-memory cache is the source of truth
# for the local replica.

_RECENT_ACTION_KEY_PREFIX = "recent_action:"

def _redis_sync_client():
    """Sync redis client, lazily — None if unavailable."""
    try:
        import redis  # type: ignore
        from ..config import settings as _s
        return redis.from_url(_s.redis_url, decode_responses=True, socket_connect_timeout=1)
    except Exception:
        return None


def _record_explicit_action(chat_id: str, skill_name: str, action: str, recipient: str = "") -> None:
    """Stamp an explicit side-effect action so a short-context follow-up can
    refer to it within the next few minutes. Writes BOTH memory and Redis;
    Redis errors are swallowed (memory is the local source of truth)."""
    if not chat_id:
        return
    import time as _t_rec
    payload = {
        "skill": skill_name, "action": action,
        "recipient": recipient, "ts": _t_rec.time(),
    }
    with _LAST_EXPLICIT_ACTION_LOCK:
        _LAST_EXPLICIT_ACTION[chat_id] = payload
    # Mirror to Redis (fail-safe, multi-replica visibility)
    try:
        _r = _redis_sync_client()
        if _r is not None:
            import json as _json
            try:
                _r.set(
                    f"{_RECENT_ACTION_KEY_PREFIX}{chat_id}",
                    _json.dumps(payload),
                    ex=_LAST_EXPLICIT_TTL_S,
                )
            finally:
                try:
                    _r.close()
                except Exception:
                    pass
    except Exception:
        pass


def _get_recent_explicit_action(chat_id: str, skill_name: str) -> dict:
    """Return the last explicit action for this chat+skill if still within
    TTL. Reads memory first; falls back to Redis on miss (multi-replica)."""
    if not chat_id:
        return {}
    import time as _t_get
    # 1) Memory L1
    with _LAST_EXPLICIT_ACTION_LOCK:
        s = _LAST_EXPLICIT_ACTION.get(chat_id)
        if s and s.get("skill") == skill_name and (_t_get.time() - s.get("ts", 0)) <= _LAST_EXPLICIT_TTL_S:
            return dict(s)
    # 2) Redis fallback (different replica may have written it)
    try:
        _r = _redis_sync_client()
        if _r is None:
            return {}
        try:
            raw = _r.get(f"{_RECENT_ACTION_KEY_PREFIX}{chat_id}")
        finally:
            try: _r.close()
            except Exception: pass
        if not raw:
            return {}
        import json as _json
        s = _json.loads(raw)
        if s.get("skill") != skill_name:
            return {}
        if _t_get.time() - s.get("ts", 0) > _LAST_EXPLICIT_TTL_S:
            return {}
        # Warm L1 cache for next read
        with _LAST_EXPLICIT_ACTION_LOCK:
            _LAST_EXPLICIT_ACTION[chat_id] = s
        return dict(s)
    except Exception:
        return {}


# ── Thin wrappers binding policy functions to local recent-action tracker ───

def _has_email_recipient(text: str) -> bool:
    if not text:
        return False
    return bool(_EMAIL_ADDR_RE.search(text))


def _intent_gate_check(skill_call, user_text, ctx_messages=None, chat_id=""):
    """Wrapper: passes the local recent-action resolver into the policy gate."""
    return _policy_intent_gate_check(
        skill_call,
        user_text,
        ctx_messages=ctx_messages,
        chat_id=chat_id,
        recent_action_resolver=_get_recent_explicit_action,
    )


def _filter_inferred_side_effects(skill_calls, user_text, ctx_messages=None, chat_id=""):
    """Wrapper: passes the local recent-action resolver and stamper into policy."""
    return _policy_filter_inferred_side_effects(
        skill_calls,
        user_text,
        ctx_messages=ctx_messages,
        chat_id=chat_id,
        recent_action_resolver=_get_recent_explicit_action,
        record_explicit_action=_record_explicit_action,
    )


def _enforce_schedule_honesty(response_text, skill_results, user_lang="en"):
    """Wrapper around policy.enforce_schedule_honesty (drops the trace dict
    for callers that don't track it). New paths should call
    apply_final_response_policy() instead."""
    text, _trace = _policy_enforce_schedule_honesty(response_text, skill_results, user_lang)
    return text


def _enforce_side_effect_text_gate(response_text, user_text, skill_results, chat_id="", user_lang="en"):
    """Wrapper around policy.enforce_side_effect_text_gate (drops trace).
    New paths should call apply_final_response_policy() instead."""
    text, _trace = _policy_enforce_side_effect_text_gate(
        response_text,
        user_text,
        skill_results,
        chat_id=chat_id,
        user_lang=user_lang,
        recent_action_resolver=_get_recent_explicit_action,
    )
    return text




# ──────────────────────────────────────────────────────────────────────────
# Request-scoped skill round budget
# Bounds total LLM rounds across all loops triggered by a single user request.
# The for-loop cap (MAX_SKILL_ROUNDS) protects each loop; this guards the
# *aggregate* (chat handler + nested goal/agent triggers) so a screenshot
# follow-up cannot grind 47 rounds across cascaded executions.
# ──────────────────────────────────────────────────────────────────────────

import threading as _budget_threading

_REQUEST_BUDGET: dict = {}                  # execution_id -> {"used", "cap", "tier", "stopped"}
_REQUEST_BUDGET_LOCK = _budget_threading.Lock()

# Per-tier total round budget. Per-loop MAX_SKILL_ROUNDS still applies inside
# each loop; this cap is for the SUM across loops within one user request.
_REQUEST_BUDGET_LIMITS = {
    "simple":  10,   # short msg, single skill (e.g. "captura de google.com")
    "normal":  20,   # default
    "complex": 36,   # multi-step, agent creation, recurring task with test run
}

# Markers that classify a request as complex (needs higher round budget).
_COMPLEX_MARKERS_RE = re.compile(
    r"\b("
    r"agente|agent|"
    r"daily|diari[oa]|todos\s+los\s+d[ií]as|every\s+day|cada\s+d[ií]a|cada\s+hora|"
    r"recurrente|recurring|programar|schedule|"
    r"informe|report|reporte|"
    r"monitor|monitorea|monitorear|"
    r"haz\s+una\s+prueba|test\s+run|ejecuci[oó]n\s+de\s+prueba"
    r")\b",
    re.IGNORECASE,
)


def _classify_request_tier(text: str) -> str:
    """Return 'simple' / 'normal' / 'complex' based on the user's message."""
    if not text:
        return "normal"
    if _COMPLEX_MARKERS_RE.search(text):
        return "complex"
    if text.count("http") >= 2:
        return "complex"
    if len(text) > 240:
        return "complex"
    if len(text) <= 60:
        return "simple"
    return "normal"


def request_budget_init(execution_id: str, text: str) -> dict:
    """Initialize a fresh budget for this request. Returns the state dict."""
    if not execution_id:
        return {}
    tier = _classify_request_tier(text)
    cap = _REQUEST_BUDGET_LIMITS.get(tier, _REQUEST_BUDGET_LIMITS["normal"])
    state = {"used": 0, "cap": cap, "tier": tier, "stopped": False}
    with _REQUEST_BUDGET_LOCK:
        _REQUEST_BUDGET[execution_id] = state
    logger.info(
        "request_budget.init",
        execution_id=execution_id[:12],
        tier=tier,
        cap=cap,
        text_len=len(text or ""),
    )
    return state


def request_budget_consume(execution_id: str, n: int = 1) -> tuple[bool, dict]:
    """Increment used count by n. Returns (allowed, snapshot).
    allowed=False once the cap has been exceeded; subsequent calls keep returning False.
    """
    if not execution_id:
        return True, {}
    with _REQUEST_BUDGET_LOCK:
        state = _REQUEST_BUDGET.get(execution_id)
        if not state:
            return True, {}
        state["used"] += n
        allowed = state["used"] <= state["cap"]
        if not allowed and not state["stopped"]:
            state["stopped"] = True
            logger.warning(
                "request_budget.exhausted",
                execution_id=execution_id[:12],
                tier=state["tier"],
                used=state["used"],
                cap=state["cap"],
            )
        return allowed, dict(state)


def request_budget_status(execution_id: str) -> dict:
    """Read-only snapshot of current budget state."""
    if not execution_id:
        return {}
    with _REQUEST_BUDGET_LOCK:
        s = _REQUEST_BUDGET.get(execution_id)
        return dict(s) if s else {}


def request_budget_release(execution_id: str) -> None:
    """Free the budget slot at the end of the request lifecycle."""
    if not execution_id:
        return
    with _REQUEST_BUDGET_LOCK:
        _REQUEST_BUDGET.pop(execution_id, None)

MAX_CHAT_SECONDS = 240  # hard wall-clock cap shared by all execution paths


def _response_fingerprint(text: str) -> str:
    """Short hash of a response for diminishing-returns detection.

    Trims whitespace and hashes the first 400 chars — enough to detect that
    the LLM produced effectively the same answer twice in a row.
    """
    if not text:
        return ""
    import hashlib
    snippet = " ".join(text.split())[:400]
    return hashlib.sha1(snippet.encode("utf-8", errors="ignore")).hexdigest()[:16]


def _correction_should_skip(ctx, correction_kind: str, response_text: str) -> tuple[bool, str]:
    """Decide whether to skip a planned correction injection.

    Returns (skip: bool, reason: str).

    Reasons to skip:
    - correction_count already at MAX_MEANINGFUL_CORRECTIONS
    - same correction_kind already fired ≥2 times this turn
    - response_text fingerprint matches the previous round (no progress)
    """
    if getattr(ctx, "correction_count", 0) >= MAX_MEANINGFUL_CORRECTIONS:
        return True, "correction_cap_reached"

    sigs = getattr(ctx, "correction_signatures", None) or []
    if sigs.count(correction_kind) >= 2:
        return True, f"repeat_correction:{correction_kind}"

    fp = _response_fingerprint(response_text)
    hashes = getattr(ctx, "last_response_hashes", None) or []
    if fp and hashes and hashes[-1] == fp:
        return True, "response_unchanged"

    return False, ""


def _correction_record(ctx, correction_kind: str, response_text: str) -> None:
    """Record that a correction is about to fire — increments counters and history."""
    try:
        ctx.correction_count = getattr(ctx, "correction_count", 0) + 1
        sigs = getattr(ctx, "correction_signatures", None)
        if sigs is None:
            ctx.correction_signatures = []
            sigs = ctx.correction_signatures
        sigs.append(correction_kind)
        if len(sigs) > 6:
            del sigs[:-6]
        fp = _response_fingerprint(response_text)
        hashes = getattr(ctx, "last_response_hashes", None)
        if hashes is None:
            ctx.last_response_hashes = []
            hashes = ctx.last_response_hashes
        if fp:
            hashes.append(fp)
            if len(hashes) > 3:
                del hashes[:-3]
    except Exception:
        pass
REDIS_APIKEYS_HASH = "apikeys"

_DATA_BLOCK_RE = re.compile(r"\[DATA\]:?\s*.*?(?:\[/DATA\]|(?=\[DATA\])|\Z)", re.DOTALL)

# Strip raw /data/... internal paths from responses (Telegram doesn't render markdown)
_INTERNAL_PATH_RE = re.compile(r"(?:https?://)?data/(?:chat-uploads|shared|screenshots?)/\S+|/data/(?:chat-uploads|shared|screenshots?)/\S+")
# Strip markdown image tags that reference internal paths or leftover [file saved] markers
# e.g. ![Caption]([file saved]), ![Caption](/data/screenshots/x.png), ![Caption](https://data/...)
_MD_IMG_GARBAGE_RE = re.compile(
    r"!\[[^\]]*\]\(<?\[file saved\]>?\)"                       # ![x]([file saved])
    r"|!\[[^\]]*\]\(<[^>]*data/screenshots?[^>]*>\)"           # ![x](<data/screenshots/...>)
    r"|!\[[^\]]*\]\((?:https?://)?data/[^)]*\)"               # ![x](/data/...) or ![x](https://data/...)
    r"|!\[[^\]]*\]\(/data/[^)]*\)"                             # ![x](/data/...)
    r"|!\[[^\]]*\]\([^)]*\[file saved\][^)]*\)",               # ![x](...[file saved]...)
    re.IGNORECASE,
)


def _strip_internal_paths(text: str) -> str:
    """Remove raw internal file paths and garbage markdown image tags from responses."""
    text = _INTERNAL_PATH_RE.sub("[file saved]", text)
    text = _MD_IMG_GARBAGE_RE.sub("", text)
    # Clean up any leftover standalone [file saved] markers
    text = re.sub(r"\s*\[file saved\]\s*", " ", text)
    return text.strip()


# Internal system prompt markers that must never reach Telegram users
_PROMPT_LEAK_RE = re.compile(
    r"\[TAREA PROGRAMADA:[^\]]*\][^\n]*"  # scheduled task header (same line only)
    r"|\bEJECUTA AHORA\b.*"              # execution directive
    r"|\[AGENT_IDENTITY\].*"             # identity block headers
    r"|\[KEY_DIRECTIVES\].*"             # directive block headers
    r"|\[STATE_EPISTEMIC\b[^\]]*\].*"   # epistemic state headers
    r"|\[REGLAS APRENDIDAS[^\]]*\].*"   # behavioral rules headers
    r"|\[SIMULACI[OÓ]N ANTICIPATORIA\][^\n]*",  # anticipatory simulation block
    re.IGNORECASE | re.MULTILINE,
)

# Markdown cleanup for Telegram — strip formatting artifacts
# Pattern order matters: longer patterns first
_MD_BOLD_RE = re.compile(r"\*\*(.+?)\*\*", re.DOTALL)
_MD_ITALIC_STAR_RE = re.compile(r"\*(.+?)\*", re.DOTALL)
_MD_ITALIC_UNDER_RE = re.compile(r"_(.+?)_", re.DOTALL)
_MD_HEADER_RE = re.compile(r"^#{1,4}\s+", re.MULTILINE)
_MD_SEPARATOR_RE = re.compile(r"^[-_*]{3,}\s*$", re.MULTILINE)
_MD_CODE_INLINE_RE = re.compile(r"`([^`]+)`")
_EXCESS_NEWLINES_RE = re.compile(r"\n{3,}")
# Execution summary blocks: "• *skill_name*: ..." (may span multiple lines) and footer lines
_EXEC_SUMMARY_BLOCK_RE = re.compile(
    r"^[•\-]\s*\*?\w[\w_]*\*?:.*?(?=\n[•\-]\s|\n\n|\Z)",  # bullet+skill block until next bullet or blank line
    re.MULTILINE | re.DOTALL,
)
_EXEC_SUMMARY_FOOTER_RE = re.compile(
    r"_?\d+\s+pasos?\s+ejecutados?_?\.?"   # "2 pasos ejecutados" / "_2 pasos ejecutados_"
    r"|_?\d+\s+steps?\s+executed_?\.?",     # English variant
    re.IGNORECASE,
)


_HTTP_HEADER_RE = re.compile(
    r"^(?:Content-Type|Content-Length|Cache-Control|Accept|Authorization|"
    r"X-[A-Za-z-]+|HTTP/\d|Transfer-Encoding|Connection|Server|Date)\s*:.*$",
    re.MULTILINE | re.IGNORECASE,
)


def _clean_telegram_output(text: str) -> str:
    """Strip markdown artifacts and prompt leakage from Telegram responses."""
    # 0. Strip leaked HTTP headers (Content-Type etc.)
    text = _HTTP_HEADER_RE.sub("", text)
    # 0b. Hard failsafe: strip any remaining raw <skill>...</skill> tags that
    #     strip_skill_calls() may have missed (e.g. backtick-wrapped or multiline body)
    text = re.sub(r"<skill>.*?</skill>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<parallel>.*?</parallel>", "", text, flags=re.DOTALL | re.IGNORECASE)
    # 1. Remove leaked system prompt fragments
    text = _PROMPT_LEAK_RE.sub("", text)
    # 2. Strip execution summary blocks (• *skill*: multi-line output / N pasos ejecutados)
    text = _EXEC_SUMMARY_BLOCK_RE.sub("", text)
    text = _EXEC_SUMMARY_FOOTER_RE.sub("", text)
    # 3. Strip markdown bold/italic (keep content) — bold first, then star-italic, then underscore-italic
    text = _MD_BOLD_RE.sub(r"\1", text)
    text = _MD_ITALIC_STAR_RE.sub(r"\1", text)
    text = _MD_ITALIC_UNDER_RE.sub(r"\1", text)
    # 4. Strip header prefixes (## Title → Title)
    text = _MD_HEADER_RE.sub("", text)
    # 5. Remove separator lines (--- / ___ / ***)
    text = _MD_SEPARATOR_RE.sub("", text)
    # 6. Unwrap inline code backticks
    text = _MD_CODE_INLINE_RE.sub(r"\1", text)
    # 7. Collapse excessive blank lines
    text = _EXCESS_NEWLINES_RE.sub("\n\n", text)
    return text.strip()


# ── Recent-output dedup + cognitive-note surfacing helpers ───────────────────
import hashlib as _hashlib_handlers

_RECENT_RESPONSE_KEY_PREFIX = "chat:last_response_hash:"
_RECENT_RESPONSE_TTL = 60   # seconds — only block exact repeats within 60s window
_COGNITIVE_NOTE_KEY_PREFIX = "chat:cognitive_note:"
_COGNITIVE_NOTE_TTL = 30    # seconds — note relevant only for the current turn


def _response_hash(text: str) -> str:
    """Stable short hash of a normalized response body for dedup comparison."""
    if not text:
        return ""
    norm = re.sub(r"\s+", " ", text.strip().lower())[:2000]
    return _hashlib_handlers.sha256(norm.encode("utf-8")).hexdigest()[:16]


async def _is_duplicate_response(redis_url: str, chat_id: str, text: str) -> bool:
    """Return True iff this exact response was already sent to this chat
    within the recent TTL window.  Lightweight — single Redis GET+SETEX."""
    if not redis_url or not chat_id or not text:
        return False
    h = _response_hash(text)
    if not h:
        return False
    try:
        import redis.asyncio as _aio
        r = _aio.from_url(redis_url, decode_responses=True)
        try:
            key = f"{_RECENT_RESPONSE_KEY_PREFIX}{chat_id}"
            prev = await r.get(key)
            await r.setex(key, _RECENT_RESPONSE_TTL, h)
            return prev == h
        finally:
            await r.aclose()
    except Exception:
        return False


async def _consume_cognitive_note(redis_url: str, chat_id: str) -> str:
    """Pop a brief user-facing note left by the cognitive layer this turn.

    Empty string when no note is pending.  Note is one short phrase like
    'I avoided retrying because the previous attempt failed.'
    """
    if not redis_url or not chat_id:
        return ""
    try:
        import redis.asyncio as _aio
        r = _aio.from_url(redis_url, decode_responses=True)
        try:
            key = f"{_COGNITIVE_NOTE_KEY_PREFIX}{chat_id}"
            note = await r.get(key)
            if note:
                await r.delete(key)
            return note or ""
        finally:
            await r.aclose()
    except Exception:
        return ""

# Detect complex multi-step task directives that should go straight to the goal engine
# (avoids double LLM call: chat response + goal planning)
_COMPLEX_DIRECTIVE_RE = re.compile(
    r"(?:"
    r"(?:objetivo|goal|tarea|instrucciones?)\s*[:\-]\s*\n"  # labeled sections
    r"|cada\s+\d+\s+hora"                                   # recurring: "cada 1 hora"
    r"|cada\s+hora\b"
    r"|automáticamente\s+cada"
    r"|automatically\s+every"
    r"|ejecuta\s+(?:esta\s+)?tarea\s+autom"                 # "ejecuta esta tarea automáticamente"
    r"|system[ao]\s+aut[oó]nomo"                            # "sistema autónomo"
    r"|(?:^\s*\d+[\.\)]\s+.{20,}\n){3}"                    # 3+ numbered list items
    r")",
    re.IGNORECASE | re.MULTILINE,
)

# ── Language system ──────────────────────────────────────────────────────────
# Lightweight, deterministic, no-LLM language detection + persistence.
#
# Architecture:
#   Fast layer  : Redis  key user:lang:<chat_id>   (TTL 30 days)
#   Durable layer: DB    table user_preferences.language
#
# On read  : Redis hit → return. Redis miss → load from DB, warm Redis.
# On write : Redis immediately + DB async (fire-and-forget).
# Lock rule: short/ambiguous inputs never switch the language.

_LANG_SIGNALS: dict[str, re.Pattern] = {
    # Portuguese BEFORE Spanish — share "como", "sim/sí" variants; check PT-specific tokens first
    "pt": re.compile(
        r"\b(olá|obrigado|obrigada|não\b|você|voce\b|tudo\s+bem|bom\s+dia"
        r"|boa\s+tarde|boa\s+noite|então|entao\b|aqui\b|isso\b|esse\b|essa\b)\b",
        re.IGNORECASE,
    ),
    "es": re.compile(
        r"\b(hola|gracias|qué\b|cómo\b|por\s+favor|buenos|días|tardes|noches"
        r"|sí\b|necesito|quiero|puedes|puede\b|dime\b|tengo\b|hacer\b|buscar"
        r"|enviar|cuánto|cuándo|dónde|ayuda\b|ahora\b|también|después|mañana"
        r"|hoy\b|ayer\b|precio\b|correo\b"
        # Short-Spanish boost: tokens that have no English collision and
        # commonly appear in 2-3 word Telegram messages without diacritics.
        # "hora", "tareas", "agentes" only match Spanish forms (English uses
        # "hour"/"agent"/"agents" — different stems).
        # "que/como/cuando/cuanto/donde" (unaccented) overlap with PT but
        # PT detection runs first and has stronger tokens (tudo bem, obrigado,
        # você, não), so PT-only text always wins over a bare "que".
        r"|que\b|como\b|cuando\b|cuanto\b|donde\b"
        r"|hora\b|tareas?\b|recordatorios?\b|agentes?\b|sub-?agentes?\b"
        # Short-ES content tokens — high-signal words that appear in 2-3
        # word commands and have no English collision. Without these,
        # "captura http://..." defaulted to EN because no other token in
        # the message hit the ES list.
        r"|captura\w*\b|imagen\b|im[aá]gen(?:es)?\b|portada\b|pantallazo\b"
        r"|enviar\b|env[ií]a\w*\b|mandar\b|revis(?:a|ar|ame|alo)\b"
        r"|env[íi]a(?:me|lo|la)?\b|mand[aá](?:me|lo|la)?\b"
        r"|mu[ée]strame\b|muestrame\b|dame\b"
        r"|haz\b|hazlo\b|dale\b|listo\b|prueba\b|ejec[úu]ta(?:lo|la)?\b"
        r"|borra(?:lo|la)?\b|elimina(?:r)?\b|olv[íi]dalo\b|recuerda\b)\b",
        re.IGNORECASE,
    ),
    "en": re.compile(
        r"\b(hello|hi\b|hey\b|thanks|thank\s+you|what\b|how\b|please|good\s+morning"
        r"|good\s+evening|yes\b|help\b|need\b|want\b|can\s+you|tell\s+me|search"
        r"|find\b|send\b|check\b|today|tomorrow|yesterday|price\b|email\b|when\b"
        r"|where\b|current"
        # Short-EN tokens — common interrogatives & auxiliaries seen in short
        # questions ("who are you?", "is it ready?", "do you know?").
        r"|who\b|why\b|time\b|date\b|now\b|is\b|are\b|do\b|does\b|can\b)\b",
        re.IGNORECASE,
    ),
    "fr": re.compile(
        r"\b(bonjour|merci|comment\b|s'il\s+vous\s+plaît|salut\b"
        r"|oui\b|non\b|aide\b|besoin\b|veux\b|chercher|aujourd'hui)\b",
        re.IGNORECASE,
    ),
}

# Language names for system prompt injection
_LANG_NAMES: dict[str, str] = {
    "en": "English",
    "es": "Spanish",
    "pt": "Portuguese",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "zh": "Chinese",
    "ja": "Japanese",
}

_LANG_REDIS_PREFIX = "user:lang:"
_LANG_REDIS_TTL = 30 * 86400  # 30 days

# Inputs that are too short or ambiguous to trigger a language switch.
# These include single-word acks, emojis, digits, and pure punctuation.
_LANG_LOCK_SHORT_RE = re.compile(
    r"^[\s\W\d]*$"                    # digits, punctuation, whitespace only
    r"|^[\U0001F300-\U0001F9FF\u2600-\u27BF\uFE00-\uFE0F\s]+$",  # emoji-only (\U for >U+FFFF)
    re.UNICODE,
)
_LANG_LOCK_AMBIGUOUS = frozenset({
    "ok", "okay", "okey", "si", "sí", "no", "yes", "yeah", "yep", "nope",
    "lol", "hm", "hmm", "ah", "oh", "wow", "k", "np", "ty", "thx",
    "👍", "👎", "🙏", "✅", "❌", "👌", "🤝",
})

# ── Low-intent guard ─────────────────────────────────────────────────────
# Tokens / inputs that carry no actionable intent on their own. When such
# a message arrives without prior context, the LLM MUST NOT produce
# arbitrary content (e.g. "ok" → fabricated weather report). Instead the
# handler returns a clarification fast-path and never invokes the LLM.
_LOW_INTENT_TOKENS = frozenset({
    "ok", "okay", "okey", "k",
    "sí", "si", "yes", "yeah", "yep", "yup", "no", "nope", "nada",
    "dale", "listo", "vale", "bueno", "bien", "claro", "perfecto", "exacto",
    "hm", "hmm", "ah", "oh", "wow", "ya", "uy", "uf", "lol",
    "np", "ty", "thx", "thanks", "gracias",
    # Greetings (hola/hello/hey/hi/ping) intentionally NOT here — they
    # have a dedicated trivial_fast_path that produces a friendly
    # response, which is correct behavior, not a hallucination.
})
_LOW_INTENT_EMOJI_RE = re.compile(
    r"^[\s\W\d\U0001F300-\U0001FAFF☀-➿︀-️]+$",
    re.UNICODE,
)


# Phrases that REQUIRE prior context — sending them on a fresh chat is a
# request to act on something that was never said. These were observed
# to cause LLM hallucinations ("haz lo mismo" → fabricated weather report
# from training-data noise).
_CONTEXT_REQUIRED_PHRASES_RE = re.compile(
    r"^\s*(?:"
    r"haz\s+lo\s+mismo"
    r"|hazlo\s+(?:de\s+nuevo|otra\s+vez)"
    r"|lo\s+mismo(?:\s+(?:de\s+antes|otra\s+vez))?"
    r"|do\s+the\s+same"
    r"|do\s+it\s+again"
    r"|same\s+(?:thing|as\s+before)"
    r"|again"
    r"|otra\s+vez"
    r"|de\s+nuevo"
    r")\s*[?!.]?\s*$",
    re.IGNORECASE,
)


def _is_low_intent(text: str) -> bool:
    """True if the message is too short / ambiguous to act on without
    prior context. Caller is responsible for checking that no
    last_exchange anchor exists before short-circuiting."""
    t = (text or "").strip()
    if not t:
        return True
    # Already-anchored retry confirms — skip (they have context).
    if t.startswith("[RETRY OF PREVIOUS:"):
        return False
    low = t.lower()
    # Single ambiguous token (with optional trailing punctuation)
    if low.rstrip("?!.,;") in _LOW_INTENT_TOKENS:
        return True
    # Emoji-only / digits-only / punctuation-only
    if _LOW_INTENT_EMOJI_RE.match(t):
        return True
    # Context-required phrases ("haz lo mismo", "do the same", etc.)
    # without an anchored prior — the LLM cannot infer "the same as what"
    # from training data, so any concrete answer is a hallucination.
    if _CONTEXT_REQUIRED_PHRASES_RE.match(t):
        return True
    # ≤2 tokens AND every token is in the ambiguous set
    tokens = [tok.rstrip("?!.,;") for tok in low.split()]
    if 1 <= len(tokens) <= 2 and all(tok in _LOW_INTENT_TOKENS for tok in tokens):
        return True
    return False


# Minimum word count for a language SWITCH (initial detection still uses 1-signal rule)
_LANG_SWITCH_MIN_WORDS = 4


# ── Static translation dictionary ─────────────────────────────────────────────
# Deterministic translations for common system messages.
# Use t(key, lang) instead of LLM for short confirmations, errors, status lines.

_SYS_PHRASES: dict[str, dict[str, str]] = {
    # Confirmations
    "task_created":         {"en": "Task created successfully.", "es": "Tarea creada correctamente.", "pt": "Tarefa criada com sucesso.", "fr": "Tâche créée avec succès."},
    "task_deleted":         {"en": "Task deleted.", "es": "Tarea eliminada.", "pt": "Tarefa eliminada.", "fr": "Tâche supprimée."},
    "task_paused":          {"en": "Task paused.", "es": "Tarea pausada.", "pt": "Tarefa pausada.", "fr": "Tâche mise en pause."},
    "task_resumed":         {"en": "Task resumed.", "es": "Tarea reanudada.", "pt": "Tarefa retomada.", "fr": "Tâche reprise."},
    "reminder_created":     {"en": "Reminder created.", "es": "Recordatorio creado.", "pt": "Lembrete criado.", "fr": "Rappel créé."},
    "reminder_deleted":     {"en": "Reminder deleted.", "es": "Recordatorio eliminado.", "pt": "Lembrete eliminado.", "fr": "Rappel supprimé."},
    "email_sent":           {"en": "Email sent.", "es": "Correo enviado.", "pt": "E-mail enviado.", "fr": "E-mail envoyé."},
    "email_failed":         {"en": "Failed to send email.", "es": "No se pudo enviar el correo.", "pt": "Falha ao enviar e-mail.", "fr": "Échec de l'envoi de l'e-mail."},
    "screenshot_taken":     {"en": "Screenshot captured.", "es": "Captura de pantalla realizada.", "pt": "Captura de tela realizada.", "fr": "Capture d'écran effectuée."},
    # Errors
    "error_generic":        {"en": "An error occurred.", "es": "Ocurrió un error.", "pt": "Ocorreu um erro.", "fr": "Une erreur s'est produite."},
    "error_not_found":      {"en": "Not found.", "es": "No encontrado.", "pt": "Não encontrado.", "fr": "Introuvable."},
    "error_timeout":        {"en": "Request timed out.", "es": "La solicitud tardó demasiado.", "pt": "A solicitação expirou.", "fr": "La demande a expiré."},
    "error_no_url":         {"en": "I need a valid URL to proceed.", "es": "Necesito una URL válida para continuar.", "pt": "Preciso de uma URL válida para prosseguir.", "fr": "J'ai besoin d'une URL valide pour continuer."},
    # Status
    "processing":           {"en": "Processing…", "es": "Procesando…", "pt": "Processando…", "fr": "Traitement en cours…"},
    "done":                 {"en": "Done.", "es": "Listo.", "pt": "Pronto.", "fr": "Terminé."},
    "no_results":           {"en": "No results found.", "es": "No se encontraron resultados.", "pt": "Nenhum resultado encontrado.", "fr": "Aucun résultat trouvé."},
    "not_configured":       {"en": "Not configured.", "es": "No configurado.", "pt": "Não configurado.", "fr": "Non configuré."},
    "duplicate_task":       {"en": "A similar task already exists: '{name}'. Use /task edit to modify it.", "es": "Ya existe una tarea similar: '{name}'. Usa /task edit para modificarla.", "pt": "Já existe uma tarefa semelhante: '{name}'. Use /task edit para modificá-la.", "fr": "Une tâche similaire existe déjà: '{name}'. Utilisez /task edit pour la modifier."},
}


def t(key: str, lang: str = "en", **kwargs: str) -> str:
    """Return a translated system phrase for the given language.

    Falls back to English if the key or language is not found.
    Supports simple {name}-style format substitutions via kwargs.
    """
    phrases = _SYS_PHRASES.get(key, {})
    text = phrases.get(lang) or phrases.get("en") or key
    if kwargs:
        try:
            text = text.format(**kwargs)
        except (KeyError, ValueError):
            pass
    return text


# Single-word greetings that are valid first-detection triggers on their own.
# These are unambiguous, language-specific, and commonly used as openers.
_LANG_GREETING_WHITELIST: frozenset[str] = frozenset({
    "hola", "hello", "bonjour", "olá", "ola",
})


def _detect_language(text: str) -> tuple[str | None, int]:
    """Keyword-based language detection. Returns (ISO 639-1 | None, signal_count).

    signal_count is the number of distinct pattern matches for the winning language.
    Returns (None, 0) if no language is detected.
    """
    if not text or len(text.strip()) < 2:
        return None, 0
    best_lang: str | None = None
    best_count: int = 0
    for lang, pattern in _LANG_SIGNALS.items():
        matches = pattern.findall(text)
        if matches and len(matches) > best_count:
            best_lang = lang
            best_count = len(matches)
    return best_lang, best_count


def _is_valid_first_detection(lang: str, text: str, signal_count: int) -> tuple[bool, str]:
    """Gate for first-time language assignment (no prior language stored).

    Returns (ok, reason). ok=True means the detection is reliable enough to persist.

    Rules — accept if ANY of:
    1. Whitelisted greeting (hola, hello, bonjour, olá)
    2. ≥2 signal matches detected
    3. Input is ≥3 words

    Reject if:
    - Ambiguous single token (ok, si, yes, no, emojis …)
    - Emoji-only or punctuation-only input
    """
    stripped = text.strip().lower()

    # Hard reject: ambiguous token
    if stripped in _LANG_LOCK_AMBIGUOUS:
        return False, "ambiguous_token"

    # Hard reject: emoji-only / punctuation-only
    if _LANG_LOCK_SHORT_RE.match(stripped):
        return False, "emoji_or_punctuation_only"

    # Whitelist: unambiguous greeting
    if stripped in _LANG_GREETING_WHITELIST:
        return True, "greeting_whitelist"

    # Accept: enough signals
    if signal_count >= 2:
        return True, "sufficient_signals"

    # Accept: sentence long enough to be meaningful
    if len(stripped.split()) >= 3:
        return True, "sentence_length"

    # Accept: short EN message with ≥1 English token. Spanish gets the same
    # symmetry treatment so short ES queries ("qué hora?") also pass.
    # Limit applies only when the language is es/en (non-trivial detection).
    if signal_count >= 1 and lang in ("en", "es") and len(stripped.split()) <= 5:
        return True, "short_message_token_match"

    return False, "insufficient_confidence"


def _should_switch_language(current_lang: str, detected_lang: str, text: str, signal_count: int) -> tuple[bool, str]:
    """Language lock: only switch on unambiguous, full-sentence input.

    Returns (should_switch, block_reason).

    Rules:
    - Same language → no switch
    - Ambiguous single token → no switch
    - Emoji-only or punctuation-only → no switch
    - Fewer than _LANG_SWITCH_MIN_WORDS words → no switch
    - Fewer than 2 signal matches in the target language → no switch
    """
    if detected_lang == current_lang:
        return False, "same_language"

    stripped = text.strip().lower()

    if stripped in _LANG_LOCK_AMBIGUOUS:
        return False, "ambiguous_token"

    if _LANG_LOCK_SHORT_RE.match(stripped):
        return False, "emoji_or_punctuation_only"

    word_count = len(stripped.split())
    if word_count < _LANG_SWITCH_MIN_WORDS:
        return False, f"too_short({word_count}_words)"

    if signal_count < 2:
        return False, f"weak_signal({signal_count})"

    return True, ""


async def _get_user_lang(redis_url: str | None, chat_id: str, db_session=None) -> str:
    """Return stored language for chat_id.

    Read order: Redis → DB → default 'en'.
    If Redis misses but DB has a value, warms Redis automatically.
    """
    # 1. Fast path: Redis
    if redis_url:
        try:
            import redis.asyncio as _aioredis
            _r = _aioredis.from_url(redis_url, decode_responses=True, socket_connect_timeout=1)
            try:
                val = await _r.get(f"{_LANG_REDIS_PREFIX}{chat_id}")
                if val:
                    return val
            finally:
                await _r.aclose()
        except Exception:
            pass

    # 2. Durable fallback: DB
    try:
        from ..db.models import UserPreference as _UP
        from ..db.session import async_session as _asession
        async with _asession() as _s:
            row = await _s.get(_UP, chat_id)
            if row and row.language:
                # Warm Redis for next call
                if redis_url:
                    asyncio.ensure_future(_redis_set_lang(redis_url, chat_id, row.language))
                return row.language
    except Exception:
        pass

    return "en"


async def _redis_set_lang(redis_url: str, chat_id: str, lang: str) -> None:
    """Write language to Redis only (internal helper)."""
    try:
        import redis.asyncio as _aioredis
        _r = _aioredis.from_url(redis_url, decode_responses=True, socket_connect_timeout=1)
        try:
            await _r.set(f"{_LANG_REDIS_PREFIX}{chat_id}", lang, ex=_LANG_REDIS_TTL)
        finally:
            await _r.aclose()
    except Exception:
        pass


async def _db_set_lang(chat_id: str, lang: str) -> None:
    """Upsert language to DB (internal helper, fire-and-forget safe)."""
    try:
        from datetime import datetime, timezone as _tz
        from sqlalchemy.dialects.postgresql import insert as _pg_insert
        from ..db.models import UserPreference as _UP
        from ..db.session import async_session as _asession
        async with _asession() as _s:
            stmt = _pg_insert(_UP).values(
                chat_id=chat_id,
                language=lang,
                updated_at=datetime.now(_tz.utc),
            ).on_conflict_do_update(
                index_elements=["chat_id"],
                set_={"language": lang, "updated_at": datetime.now(_tz.utc)},
            )
            await _s.execute(stmt)
            await _s.commit()
    except Exception:
        pass


async def _set_user_lang(redis_url: str | None, chat_id: str, lang: str) -> None:
    """Dual-layer persist: Redis immediately + DB async."""
    if redis_url:
        await _redis_set_lang(redis_url, chat_id, lang)
    asyncio.ensure_future(_db_set_lang(chat_id, lang))


# ── Tool Router cross-turn anti-retry memory ─────────────────────────────────
_TOOL_ERROR_REDIS_PREFIX = "tool_router:last_error:"
_TOOL_ERROR_TTL = 120  # seconds — short-lived, only covers consecutive turns


# ── Language failed-detection recovery ───────────────────────────────────────
_LANG_FAILED_REDIS_PREFIX = "lang:failed_detection:"
_LANG_FAILED_TTL = 1800        # 30 min hard Redis TTL
_LANG_FAILED_ACTIVE_WINDOW = 300  # 5 min logical active window — ignored after this


async def _set_lang_failed(redis_url: str | None, chat_id: str) -> None:
    """Mark that first-time language detection was rejected for this chat_id.

    Stores the Unix timestamp so _get_lang_failed can apply a 5-minute activity
    window — flag is ignored after inactivity regardless of Redis TTL.
    """
    if not redis_url:
        return
    try:
        import redis.asyncio as _aioredis
        import time as _time
        _r = _aioredis.from_url(redis_url, decode_responses=True, socket_connect_timeout=1)
        try:
            await _r.set(
                f"{_LANG_FAILED_REDIS_PREFIX}{chat_id}",
                str(int(_time.time())),
                ex=_LANG_FAILED_TTL,
            )
        finally:
            await _r.aclose()
    except Exception:
        pass


async def _clear_lang_failed(redis_url: str | None, chat_id: str) -> None:
    """Clear the failed-detection flag after a successful language assignment."""
    if not redis_url:
        return
    try:
        import redis.asyncio as _aioredis
        _r = _aioredis.from_url(redis_url, decode_responses=True, socket_connect_timeout=1)
        try:
            await _r.delete(f"{_LANG_FAILED_REDIS_PREFIX}{chat_id}")
        finally:
            await _r.aclose()
    except Exception:
        pass


async def _get_lang_failed(redis_url: str | None, chat_id: str) -> bool:
    """Return True if there is an *active* failed-detection flag for this chat_id.

    The flag is considered active only if it was set within the last 5 minutes.
    If older, it is logically ignored (key not deleted — TTL handles cleanup).
    This prevents stale flags from affecting interactions after inactivity.
    """
    if not redis_url:
        return False
    try:
        import redis.asyncio as _aioredis
        import time as _time
        _r = _aioredis.from_url(redis_url, decode_responses=True, socket_connect_timeout=1)
        try:
            val = await _r.get(f"{_LANG_FAILED_REDIS_PREFIX}{chat_id}")
        finally:
            await _r.aclose()
        if not val:
            return False
        try:
            age = _time.time() - float(val)
            return age <= _LANG_FAILED_ACTIVE_WINDOW
        except (ValueError, TypeError):
            # Unexpected stored value (legacy "1") — treat as stale, ignore
            return False
    except Exception:
        return False


async def _set_tool_error_memory(redis_url: str, chat_id: str, error: dict) -> None:
    """Persist last tool error for chat_id with short TTL."""
    try:
        import redis.asyncio as _aioredis
        _r = _aioredis.from_url(redis_url, decode_responses=True, socket_connect_timeout=1)
        try:
            await _r.set(
                f"{_TOOL_ERROR_REDIS_PREFIX}{chat_id}",
                json.dumps(error),
                ex=_TOOL_ERROR_TTL,
            )
        finally:
            await _r.aclose()
    except Exception:
        pass


async def _get_tool_error_memory(redis_url: str | None, chat_id: str) -> dict | None:
    """Return last tool error for chat_id if still within TTL, else None."""
    if not redis_url:
        return None
    try:
        import redis.asyncio as _aioredis
        _r = _aioredis.from_url(redis_url, decode_responses=True, socket_connect_timeout=1)
        try:
            raw = await _r.get(f"{_TOOL_ERROR_REDIS_PREFIX}{chat_id}")
            return json.loads(raw) if raw else None
        finally:
            await _r.aclose()
    except Exception:
        return None


def _build_lang_directive(lang: str) -> str:
    """Return system prompt block enforcing output language.

    Strengthened to address Bug #6: LLM was producing mixed-language output
    like "Son las 3:53 del Wednesday 29 de April de 2026" (Spanish syntax,
    English month). Now includes explicit examples + zero-tolerance phrasing.
    """
    lang_name = _LANG_NAMES.get(lang, lang.upper())
    base = (
        f"\n\n[LANGUAGE DIRECTIVE — STRICT]\n"
        f"The user communicates in {lang_name}. EVERY word of your response "
        f"to the user MUST be in {lang_name}. This is a HARD requirement, "
        f"NOT a preference.\n"
        f"This applies to: dates, days of week, month names, numerical "
        f"descriptions, time-of-day phrasing, and ALL natural language.\n"
        f"NEVER mix languages. NEVER produce text like 'Son las 3:53 del "
        f"Wednesday' (mixing Spanish syntax with English weekday) or 'It is "
        f"miércoles' — both are violations.\n"
        f"Internal skill calls, XML tags, JSON, and system markers remain "
        f"in English (these are not user-facing).\n"
    )
    # Specific anti-mix examples per language pair
    if lang == "en":
        base += (
            "Examples for English users:\n"
            "  GOOD: 'It is 3:53 PM on Wednesday, April 29, 2026.'\n"
            "  BAD : 'Son las 3:53 del Wednesday 29 de April' (mixed)\n"
            "  BAD : 'It is miércoles April 29' (mixed)\n"
        )
    elif lang == "es":
        base += (
            "Ejemplos para usuarios en español:\n"
            "  BUENO: 'Son las 3:53 PM del miércoles 29 de abril de 2026.'\n"
            "  MAL  : 'It is 3:53 del Wednesday 29 de April' (mezclado)\n"
            "  MAL  : 'Son las 3 PM on Wednesday' (mezclado)\n"
        )
    return base + "[/LANGUAGE DIRECTIVE]"


# ── Trivial message fast-path ────────────────────────────────────────────────
# Greetings / short social messages that need no tools or memory context.
# These get a lightweight LLM call with a minimal system prompt — no timeout risk.
_TRIVIAL_RE = re.compile(
    r"^(?:"
    r"hola|hi|hey|hello|buenos?\s+d[ií]as?|buenas?\s+tardes?|buenas?\s+noches?"
    r"|qu[eé]\s+tal|c[oó]mo\s+est[aá]s?|how\s+are\s+you|what'?s?\s+up"
    r"|gracias|thank(?:s|\s+you)|de\s+nada"
    r"|test|ping|prueba"
    r")[\s\!\.\?]*$",
    re.IGNORECASE,
)

# Task-SETUP requests: should bypass the goal engine and go through LLM directly
# so the agent can execute the workflow immediately AND create the recurring task.
# Pattern is intentionally broad — fixed regex previously missed common phrasings
# like "envíes un informe" / "que cada N horas revises..." → silent failure.
_TASK_SETUP_RE = re.compile(
    # Recurring interval markers — "cada N horas/días/minutos/semanas"
    r"\bcada\s+(?:\d+\s+)?(?:hora|d[íi]a|minuto|semana|mes|hr|h)\w*"
    # "todos los días" / "todas las semanas" — daily/weekly recurrence
    r"|\btodos?\s+los\s+(?:d[íi]as?|lunes|martes|mi[ée]rcoles|jueves|viernes|s[áa]bados?|domingos?)"
    r"|\btodas?\s+las\s+(?:semanas?|noches?|tardes?|ma[ñn]anas?)"
    r"|\bdiariamente\b|\bsemanalmente\b|\bmensualmente\b"
    r"|\bdaily\b|\bweekly\b|\bmonthly\b|\bevery\s+\d|\beach\s+(?:hour|day|week)"
    # Verb stems that frequently anchor recurring jobs — allow optional
    # determiner between verb and object ("envíes un informe", "manda el reporte")
    r"|env[íi]a?\w*\s+(?:un|el|los|las|me|nos|tu|mi)?\s*(?:informe|reporte|resumen|noticias?|update)"
    r"|mand[aá]\w*\s+(?:un|el|los|las|me|nos)?\s*(?:informe|reporte|resumen)"
    r"|monitor\w*\s+(?:el|la|los|las|el\s+precio|la\s+actividad|la\s+web|la\s+p[áa]gina)"
    r"|revis\w*\s+(?:el\s+precio|los\s+precios|las\s+noticias)"
    # "quiero que cada N" / "que me envíes" — common ES recurring phrasing
    r"|quiero\s+que\s+cada\b"
    r"|que\s+me\s+(?:env[íi]es|mandes|avises|notifiques|recuerdes)"
    # English equivalents
    r"|i\s+want\s+(?:you\s+)?to\s+(?:check|monitor|send)"
    r"|send\s+me\s+(?:a|the)\s+(?:report|summary|update)\s+(?:every|daily|weekly)"
    r"|ejecutar?\s+cada\s+\d"
    r"|ejecutar?\s+cada\s+hora",
    re.IGNORECASE,
)

# ── Tool Router Sanity Check ─────────────────────────────────────────────────
# Matches: https?://, www., or bare domain-like tokens (e.g. "example.com/path")
_ROUTER_URL_RE = re.compile(
    r"(?:https?://|www\.)[^\s\"'<>()\[\]]+|[a-zA-Z0-9][-a-zA-Z0-9]*\.[a-zA-Z]{2,}(?:/[^\s\"'<>()\[\]]*)?",
    re.IGNORECASE,
)
# Characters to strip from URL tails (common punctuation leak)
_URL_TAIL_STRIP = re.compile(r"[.,;:!?\)\]\">]+$")


def normalize_url(raw: str) -> str:
    """Normalize a raw URL token to a usable https:// URL.

    - Strips trailing commas, periods, parentheses, quotes
    - Adds https:// if the URL starts with www. or is a bare domain
    - Returns the cleaned URL string
    """
    url = _URL_TAIL_STRIP.sub("", raw.strip())
    if not url:
        return url
    lower = url.lower()
    if lower.startswith("https://") or lower.startswith("http://"):
        return url
    if lower.startswith("www."):
        return "https://" + url
    # Bare domain (e.g. "example.com/path") — add https://
    return "https://" + url


def _extract_urls(text: str) -> list[str]:
    """Extract and normalize all URLs from free text."""
    raw_matches = _ROUTER_URL_RE.findall(text or "")
    seen: set[str] = set()
    result = []
    for raw in raw_matches:
        url = normalize_url(raw)
        if url and url not in seen:
            seen.add(url)
            result.append(url)
    return result


def sanitize_tool_calls(
    skill_calls: list,
    user_text: str,
) -> tuple[list, list[dict]]:
    """Pre-execution tool router sanity check.

    Returns (sanitized_calls, router_errors).

    Rule 1: web_search + URL in user input → override to browser(navigate).
    Rule 2: browser + no URL in args and no URL in user input → drop call,
            emit structured error so the LLM can inform the user.
    Rule 3: All mismatches logged as agent.tool_mismatch.
    """
    if not skill_calls:
        return skill_calls, []

    sanitized: list = []
    errors: list[dict] = []
    user_urls = _extract_urls(user_text)

    for call in skill_calls:
        name = getattr(call, "skill_name", "") or ""
        args = getattr(call, "arguments", {}) or {}

        # Rule 1: web_search + URL in user input → override to browser navigate
        if name == "web_search" and user_urls:
            target_url = user_urls[0]
            logger.warning(
                "agent.tool_mismatch",
                requested_tool="web_search",
                correction="browser(navigate)",
                url=target_url,
                input=user_text[:120],
                reason="URL detected in user input — use browser instead of web_search",
            )
            from ..skills.types import SkillCall as _SC
            sanitized.append(_SC(
                skill_name="browser",
                arguments={"action": "navigate", "url": target_url, "session": "s1"},
            ))
            continue

        # Rule 2: browser + no URL in arguments and no URL in user input → reject
        if name == "browser":
            call_url = normalize_url(args.get("url", "") or "")
            action = args.get("action", "") or ""
            # Only enforce for navigate/capture/empty where a URL is required
            if action in ("navigate", "capture", "") and not call_url and not user_urls:
                logger.warning(
                    "agent.tool_mismatch",
                    requested_tool="browser",
                    correction="rejected",
                    input=user_text[:120],
                    reason="browser called with no URL in args or user input",
                )
                errors.append({
                    "error": "missing_url",
                    "tool": "browser",
                    "action": action or "navigate",
                    "message": "I need a valid URL to perform this action.",
                    "hint": "ask_user_for_url",
                })
                continue

            # Normalize URL in args if present
            if call_url and call_url != (args.get("url", "") or ""):
                args = {**args, "url": call_url}
                try:
                    call.arguments = args
                except Exception:
                    pass

        sanitized.append(call)

    return sanitized, errors


# Detect "already deleted / not found" responses — should NOT trigger retry loops
_ALREADY_GONE_RE = re.compile(
    r"\b(not found|no encontrado|ya (?:fue )?eliminad|already deleted|already removed|"
    r"does not exist|no existe|404|was already|ya estaba|item not found|goal not found|"
    r"agent not found|task not found|no se encontr[oó])\b",
    re.IGNORECASE,
)

# Detect user intent to see a plan WITHOUT any execution
_PLANNING_MODE_RE = re.compile(
    r"\b(?:"
    # Spanish — explicit no-execute requests
    r"no\s+ejecutes?|no\s+(?:lo\s+)?(?:crees?|hagas?|implementes?|pongas?\s+en\s+marcha|lances?|configures?)|"
    r"no\s+hagas?\s+nada\s+(?:todav[ií]a|aún|por\s+ahora|a[uú]n)|"
    r"sin\s+ejecutar|sin\s+crear|sin\s+hacer\s+nada|"
    r"solo\s+planificaci[oó]n|s[oó]lo\s+planificaci[oó]n|solo\s+an[aá]lisis|s[oó]lo\s+an[aá]lisis|"
    r"antes\s+de\s+(?:ejecutar|crear|hacer|implementar|proceder|configurar)|"
    # Spanish — explain-first requests
    r"solo\s+expl[ií]c[ae](?:me)?|s[oó]lo\s+expl[ií]c[ae](?:me)?|"
    r"solo\s+(?:descr[ií]be(?:me)?|mu[eé]stra(?:me)?|dime\s+c[oó]mo|analiza)|"
    r"s[oó]lo\s+(?:descr[ií]be(?:me)?|mu[eé]stra(?:me)?|dime\s+c[oó]mo|analiza)|"
    r"quiero\s+(?:ver|entender|saber|revisar)\s+(?:c[oó]mo\s+(?:lo\s+)?har[ií]as?|el\s+plan|la\s+estructura|la\s+arquitectura|el\s+enfoque)|"
    r"expl[ií]ca(?:me)?\s+(?:c[oó]mo\s+(?:lo\s+)?har[ií]as?|el\s+plan|primero|qu[eé]\s+har[ií]as?)|"
    r"c[oó]mo\s+(?:lo\s+)?har[ií]as?\b|qu[eé]\s+har[ií]as?\s+(?:t[uú]|para|si)\b|"
    r"primero\s+(?:expl[ií]ca(?:me)?|mu[eé]stra(?:me)?|dime\s+(?:el\s+plan|c[oó]mo))|"
    r"mu[eé]stra(?:me)?\s+el\s+plan|describe\s+(?:el\s+plan|la\s+arquitectura|c[oó]mo)|"
    r"cu[eé]ntame\s+c[oó]mo\s+(?:lo\s+)?har[ií]as?|"
    # English
    r"don['\u2019]?t\s+execute|do\s+not\s+execute|don['\u2019]?t\s+(?:create|run|start|do)\s+(?:anything|it\s+yet)|"
    r"only\s+explain|just\s+explain|just\s+(?:show|tell|describe|outline)\s+(?:me\s+)?(?:the\s+plan|how|it)|"
    r"show\s+me\s+(?:the\s+)?plan|describe\s+(?:the\s+)?(?:plan|approach|architecture)|"
    r"without\s+executing|before\s+(?:executing|running|doing\s+anything)|"
    r"how\s+would\s+you\s+(?:do|handle|approach|build|create)\b|"
    r"what\s+would\s+you\s+do\b"
    r")\b",
    re.IGNORECASE,
)

# Detect when LLM produces only planning text for a complex task (no skills executed at all)
_PLANNING_ONLY_RE = re.compile(
    r"\b(voy a (?:crear|proceder|iniciar|comenzar|configurar|establecer|implementar|desarrollar|ejecutar)|"
    r"vamos a (?:proceder|comenzar|iniciar|crear|implementar)|"
    r"comenzar[eé] (?:por|con|a)|iniciar[eé]|proceder[eé]|implementar[eé]|"
    r"primer(?:o|amente)|paso\s+\d|step\s+\d|primero\s+(?:voy|necesito|debo)|"
    r"I(?:'ll| will)\s+(?:now\s+)?(?:create|start|begin|proceed|set up|implement|first)|"
    r"(?:let me|I'll)\s+(?:outline|plan|describe|walk you through))\b",
    re.IGNORECASE,
)

# Detect when LLM verbally promises to retry but forgot to include a skill call
_RETRY_PROMISE_RE = re.compile(
    r"\b(voy a intentar|intentar[eé]|voy a buscar|buscar[eé]|d[eé]jame\s+(?:buscar|intentar|verificar|revisar|ver)|"
    r"buscar[eé]\s+(?:de\s+nuevo|nuevamente|otra\s+vez)|probar[eé]|lo\s+intento\s+(?:de\s+nuevo|nuevamente)|"
    r"intentando\s+(?:de\s+nuevo|otra\s+vez)|trying\s+again|will\s+try|let\s+me\s+(?:search|check|try|look)|"
    r"voy\s+a\s+(?:intentar|probar|buscar|verificar|revisar|proceder|tomar|acceder|navegar|abrir|capturar)|"
    r"voy\s+a\s+(?:intentar|probar|buscar|verificar|revisar)\s+(?:de\s+nuevo|nuevamente|otra\s+vez|otra\s+forma|diferente)|"
    r"permít(?:e|a)me|procedo\s+a|procediendo|te\s+(?:lo\s+)?muestro|ahora\s+(?:tomo|navego|abro|capturo|busco)|"
    r"I(?:'ll| will)\s+(?:try|proceed|search|check|capture|navigate|open|take)|"
    r"let\s+me\s+(?:try|proceed|access|take|open|get|fetch)|"
    r"a\s+continuaci[oó]n\s+(?:tomo|navego|busco|capturo)|"
    r"voy\s+a\s+proceder|per\s+me|tome\s+(?:una\s+)?captura)\b",
    re.IGNORECASE,
)

# ── Action Commitment Architecture — helper functions ─────────────────────────
# These build the system-prompt blocks and enforcement prompts for Phases 2–4.

def _build_action_commitment_block(intent: "_ActionIntent", attempts: int = 0) -> str:
    """Build the [ACTION COMMITMENT] block injected into the system prompt."""
    target_line = f"\n  Target: {intent.action_target}" if intent.action_target else ""
    skill_line = f"\n  Required skill: {intent.primary_skill}" if intent.primary_skill else ""

    composition_hint = ""
    if intent.primary_skill == "browser":
        if intent.action_type == "browser_package_check":
            _code = intent.tracking_code or "TRACKING_CODE_FROM_USER_MESSAGE"
            _site = intent.action_target or ""
            if _site and not _site.startswith("http"):
                _site = "https://" + _site
            if _site:
                composition_hint = (
                    "\n\nFor package/shipment tracking, use the dedicated track action:"
                    f'\n  <skill>browser(action="track", tracking_number="{_code}", url="{_site}", session="track1")</skill>'
                    "\n  This handles navigate → find input → type → submit → capture automatically."
                    "\n  If browser fails, fallback: python_exec with requests to the tracking API."
                )
            else:
                composition_hint = (
                    "\n\nFor package/shipment tracking: first run web_search to find the carrier's official "
                    "tracking page (use the tracking number prefix/suffix to identify the carrier — e.g. CN suffix → "
                    "China Post / EMS, US prefixes → USPS). Then use browser(action=\"track\", tracking_number=..., "
                    "url=<resolved URL>, session=\"track1\"). Do NOT default to any specific aggregator."
                )
        elif intent.action_type == "browser_form_workflow":
            tgt = intent.action_target or "TARGET_URL"
            if tgt and not tgt.startswith("http"):
                tgt = "https://" + tgt
            _goal = intent.workflow_objective or "complete the requested web action"
            composition_hint = (
                f"\n\nGoal: {_goal}"
                "\n\nFor web form / booking / registration workflows, follow this pattern:"
                f'\n  1. <skill>browser(action="navigate", url="{tgt}", session="wf1")</skill>'
                '\n  2. <skill>browser(action="capture", session="wf1")</skill>  — see the page first'
                '\n  3. For each input field: <skill>browser(action="type", selector="CSS_OR_XPATH", text="VALUE", session="wf1")</skill>'
                '\n  4. For dropdowns/selects: <skill>browser(action="select", selector="SELECT_SELECTOR", value="OPTION", session="wf1")</skill>'
                '\n  5. For submit/continue: <skill>browser(action="click", selector="BUTTON_SELECTOR", session="wf1")</skill>'
                '\n  6. After each significant action: <skill>browser(action="capture", session="wf1")</skill>  — verify state'
                "\n\nSelectors — try in order: CSS (most reliable), then XPath if CSS fails."
                "\nAfter success, call skill_manager to save this workflow as a reusable skill."
            )
        elif intent.action_type == "browser_web_workflow":
            tgt = intent.action_target or "TARGET_URL"
            if tgt and not tgt.startswith("http"):
                tgt = "https://" + tgt
            _goal = intent.workflow_objective or "complete the requested web task"
            composition_hint = (
                f"\n\nGoal: {_goal}"
                "\n\nFor general web workflows, decompose into browser primitives:"
                f'\n  1. <skill>browser(action="navigate", url="{tgt}", session="wf1")</skill>'
                '\n  2. <skill>browser(action="capture", session="wf1")</skill>  — inspect the page'
                "\n  3. Based on what you see, use: type / click / scroll / find / select as needed"
                '\n  4. After completing each step: <skill>browser(action="capture", session="wf1")</skill>'
                "\n\nBrowser primitives available: navigate, capture, type, click, scroll, find,"
                "\n  select, wait, execute_js, download_file, get_text."
                "\nIf the standard approach fails, try: python_exec with requests/BeautifulSoup,"
                "\nor web_search to find the API endpoint directly."
                "\nAfter success, ALWAYS call skill_manager(action='create', ...) to persist"
                "\nthe workflow as a reusable Python skill so it can be reused automatically."
            )
        else:
            tgt = intent.action_target or "TARGET_URL"
            composition_hint = (
                "\n\nFor web navigation, start with:"
                f'\n  1. <skill>browser(action="navigate", url="{tgt}", session="s1")</skill>'
                '\n  2. <skill>browser(action="capture", session="s1")</skill>'
                "\n  Then interact further if needed (type, click, scroll)."
            )

    escalation_note = ""
    if attempts >= 2:
        escalation_note = (
            f"\n\n⚠ ESCALATION: This is attempt {attempts + 1}. "
            "Previous direct attempts failed. You MUST try a different approach "
            "(e.g. python_exec with requests, fetch_url, or a different selector strategy)."
        )

    # Task lock — prohibit domain drift during committed tasks
    task_lock = ""
    if intent.action_type == "browser_package_check":
        task_lock = (
            "\n\nTASK LOCK — DOMAIN RESTRICTION:\n"
            "  This is an active package tracking task. While executing:\n"
            "  ✗ Do NOT discuss crypto prices, weather, news, or unrelated topics.\n"
            "  ✗ Do NOT provide general information about the tracking site.\n"
            "  ✓ Report ONLY what the browser returned about this specific package.\n"
            "  ✓ Include [TRACK_STATUS: FOUND/NOT_FOUND/PARTIAL/FAILED] in your response.\n"
            "  ✓ If FOUND: quote the actual status text from the page.\n"
            "  ✓ If NOT_FOUND or FAILED: state the exact reason (timeout/blocked/no-input)."
        )
    elif intent.action_type in ("browser_form_workflow", "browser_web_workflow"):
        _obj_desc = getattr(intent.objective_spec, "done_description", "") if hasattr(intent, "objective_spec") else ""
        _partial_signals = getattr(intent.objective_spec, "partial_only_signals", []) if hasattr(intent, "objective_spec") else []
        _partial_str = " / ".join(_partial_signals[:4]) if _partial_signals else "page loaded"
        task_lock = (
            f"\n\nOBJECTIVE DEFINITION — what 'done' means:\n"
            f"  ✓ DONE = {_obj_desc or 'objective confirmed in browser output'}\n"
            f"  ✗ NOT DONE = {_partial_str} (these are partial signals — keep executing)\n"
            "\nTASK LOCK — ANTI-PREMATURE COMPLETION:\n"
            "  Do NOT stop after navigating to the page.\n"
            "  Do NOT stop after opening the form.\n"
            "  Do NOT stop after clicking a button without seeing confirmation.\n"
            "  Keep executing until you see the OBJECTIVE MET (above).\n"
            "  If stuck: try a different selector, scroll, or alternative URL.\n"
            "  Report ONLY verified outcomes — no speculation."
        )
    elif intent.action_type == "browser_navigation" and intent.action_target:
        task_lock = (
            f"\n\nTASK LOCK — stay focused on: {intent.action_target}\n"
            "  Report ONLY what the browser retrieved. No speculation or off-topic content."
        )

    return (
        "\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "[ACTION COMMITMENT — EXECUTION REQUIRED]\n"
        f"The user explicitly requested an ACTION, not an explanation.{target_line}{skill_line}\n\n"
        "ABSOLUTE RULES:\n"
        "1. A text-only response is NOT acceptable here. You MUST emit at least one <skill> tag.\n"
        "2. Do NOT explain how the user could do it themselves.\n"
        "3. Do NOT say 'you can visit', 'I recommend visiting', or 'go to the website'.\n"
        "4. ATTEMPT the action using your skills. Execute — do not narrate.\n"
        "5. If one skill fails, try an alternative (python_exec, fetch_url, web_search).\n"
        f"{composition_hint}{task_lock}{escalation_note}\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )


def _build_enforcement_prompt(intent: "_ActionIntent") -> str:
    """Build the hard enforcement correction injected after a text-only response."""
    target = f" on {intent.action_target}" if intent.action_target else ""
    skill = intent.primary_skill or "browser"

    example = ""
    if skill == "browser":
        tgt = intent.action_target or "URL_FROM_USER_MESSAGE"
        if intent.action_type == "browser_package_check":
            _code2 = intent.tracking_code or "TRACKING_CODE_FROM_USER_MESSAGE"
            if tgt and tgt != "URL_FROM_USER_MESSAGE":
                _site2 = tgt if tgt.startswith("http") else "https://" + tgt
                example = (
                    f'\nExample: <skill>browser(action="track", tracking_number="{_code2}", url="{_site2}", session="track1")</skill>'
                )
            else:
                example = (
                    f'\nFirst find the carrier: <skill>web_search(query="tracking {_code2} carrier official site")</skill>'
                    f'\nThen track on the resolved URL: <skill>browser(action="track", tracking_number="{_code2}", url="<resolved>", session="track1")</skill>'
                )
        else:
            example = (
                f'\nImmediate example: <skill>browser(action="navigate", url="{tgt}", session="s1")</skill>'
                '\nthen: <skill>browser(action="capture", session="s1")</skill>'
            )
    elif skill == "gmail":
        example = '\nExample: <skill>gmail(action="send", to="RECIPIENT", subject="SUBJECT", body="BODY")</skill>'

    return (
        f"You did not execute the required action{target}. "
        "A text-only response is NOT acceptable for this request.\n"
        f"YOU MUST use the skill: {skill}\n"
        f"{example}\n"
        "Execute NOW without asking for confirmation, without explaining, without questioning. "
        "Just include the skill tag directly in your response."
    )


def _extract_failure_diagnostic(skill_results: list) -> str:
    """Extract a concrete technical failure reason from skill result outputs.

    Returns a short human-readable diagnostic string (or empty string if nothing useful found).
    """
    if not skill_results:
        return ""
    for r in skill_results:
        if r.success:
            continue
        err = (r.error or "").lower()
        out = (r.output or "").lower()
        combined = err + " " + out
        if "timeout" in combined or "timed out" in combined:
            return "Page load timed out (site took too long or blocked access)."
        if "anti-bot" in combined or "captcha" in combined or "403" in combined or "blocked" in combined:
            return "The site blocked automated access (anti-bot protection or CAPTCHA)."
        if "input" in combined and ("not found" in combined or "no element" in combined or "selector" in combined):
            return "Required input field not found on the page."
        if "navigation" in combined and ("failed" in combined or "error" in combined):
            return "Navigation failed — URL unreachable or DNS did not resolve."
        if "connection" in combined or "refused" in combined or "unreachable" in combined:
            return "Could not connect to the server (connection refused or host unreachable)."
        if err:
            # Return first line of error, capped at 120 chars
            first_line = (r.error or "").split("\n")[0].strip()
            return first_line[:120] if first_line else ""
    return ""


# ── System-Controlled Execution Engine ──────────────────────────────────────

# ── Step Enforcement Layer ────────────────────────────────────────────────────

def _verify_required_steps(
    action_type: str,
    action_history: list,
) -> tuple[bool, str]:
    """Verify mandatory browser interaction steps were executed in the correct order.

    This is SYSTEM authority — overrides any LLM claim of success.
    Checks actual recorded actions, not LLM output text.

    Returns: (steps_valid, reason_code)
    """
    if not action_history:
        return False, "no_action_history"

    actions = [h.get("action", "") for h in action_history]
    successful_actions = [h.get("action", "") for h in action_history if h.get("success")]
    all_output = " ".join(h.get("output", "") for h in action_history)

    # ── Package tracking ─────────────────────────────────────────────────────
    if action_type == "browser_package_check":
        has_compound_track = "track" in actions  # _do_track() compound action
        has_navigate = "navigate" in successful_actions
        has_type = "type" in successful_actions
        has_click = any(a in ("click", "submit") for a in successful_actions)
        has_result = "[TRACK_STATUS:" in all_output

        if has_compound_track:
            # action="track" internally handles navigate+type+click+wait
            # Only require that it produced a TRACK_STATUS label
            if not has_result:
                return False, "no_result_after_submit"
            return True, "compound_track_complete"

        # Manual step-by-step tracking
        if not has_navigate:
            return False, "missing_navigate"
        if not has_type:
            return False, "missing_type_action"
        if not has_click:
            return False, "missing_submit_action"
        if not has_result:
            # Check for DOM change as fallback evidence
            if _detect_dom_change_from_history(action_history):
                return True, "dom_change_detected"
            return False, "no_result_after_submit"
        return True, "all_steps_complete"

    # ── Form workflows ────────────────────────────────────────────────────────
    if action_type == "browser_form_workflow":
        has_navigate = "navigate" in successful_actions
        has_interaction = any(a in ("type", "select", "fill", "check") for a in successful_actions)
        has_click = any(a in ("click", "submit") for a in successful_actions)

        if not has_navigate:
            return False, "missing_navigate"
        if not has_interaction and not has_click:
            return False, "missing_form_interaction"
        if not has_click:
            return False, "missing_submit_action"
        return True, "form_steps_present"

    # ── Web workflows ─────────────────────────────────────────────────────────
    if action_type == "browser_web_workflow":
        has_navigate = "navigate" in successful_actions
        if not has_navigate:
            return False, "missing_navigate"
        # Require at least one interaction beyond pure navigate+capture
        has_interaction = any(
            a in ("type", "execute_js", "find", "scroll", "click", "select")
            for a in successful_actions
        )
        # OR substantial content was returned (scraping via navigate alone can work)
        has_content = len(all_output.strip()) > 300
        if not has_interaction and not has_content:
            return False, "missing_data_extraction"
        return True, "workflow_steps_present"

    # All other types — no mandatory sequence enforced
    return True, "no_requirements"


def _detect_dom_change_from_history(action_history: list) -> bool:
    """Detect if page content changed meaningfully after a submit/click action.

    Compares content volume before and after the first click/submit in history.
    Returns True if post-click content is ≥15% larger than pre-click content.
    """
    pre_content = ""
    post_content = ""
    found_submit = False

    for entry in action_history:
        action = entry.get("action", "")
        output = entry.get("output", "")

        if not found_submit:
            if action in ("navigate", "capture"):
                pre_content = output  # last pre-submit capture
            elif action in ("click", "submit", "track"):
                found_submit = True
        else:
            if action in ("capture", "extract") and output:
                post_content = output
                break  # use first post-submit capture

    if not found_submit or not post_content:
        return False

    pre_len = len(pre_content.strip())
    post_len = len(post_content.strip())

    if pre_len == 0:
        return post_len > 100
    return post_len > pre_len * 1.15


def _build_missing_step_prompt(
    intent: "_ActionIntent",
    reason: str,
    action_history: list,
) -> str:
    """Build a targeted correction prompt when mandatory steps are missing.

    SYSTEM authority: overrides LLM response entirely.
    Tells the LLM exactly which step was missed and what to do next.
    """
    action_type = intent.action_type
    target = intent.action_target or ""
    if target and not target.startswith("http"):
        target = "https://" + target

    executed = [h.get("action", "") for h in action_history if h.get("success")]
    executed_str = " → ".join(executed) if executed else "(none)"

    base = (
        f"[STEP ENFORCEMENT — EXECUTION INCOMPLETE]\n"
        f"Steps executed so far: {executed_str}\n\n"
    )

    if reason == "missing_type_action":
        code = intent.tracking_code or "the tracking code"
        return (
            base
            + f"PROBLEM: You navigated to the site but did NOT type the tracking code.\n"
            f"REQUIRED NEXT STEP: Type '{code}' into the search/tracking input field.\n"
            f'Use: <skill>browser(action="type", selector="input[type=text], input[name*=track], '
            f'#trackInput, input[placeholder*=track]", text="{code}", session="track1")</skill>\n'
            "Then click the submit/Track button. DO NOT respond until tracking results are visible."
        )

    if reason == "missing_submit_action":
        code = intent.tracking_code or ""
        code_line = f"Tracking code: {code}\n" if code else ""
        return (
            base
            + f"PROBLEM: You typed the input but did NOT click the submit/Track button.\n"
            f"{code_line}"
            "REQUIRED NEXT STEP: Click the submit or Track button NOW.\n"
            'Use: <skill>browser(action="click", selector="button[type=submit], '
            'button.track-btn, input[type=submit], button:contains(Track)", session="track1")</skill>\n'
            "After clicking, capture the page and verify results appeared.\n"
            "DO NOT stop until the tracking result section is visible."
        )

    if reason == "no_result_after_submit":
        return (
            base
            + "PROBLEM: You submitted the form but NO result appeared — page content did not change.\n"
            "REQUIRED: The result section must be visible before this task is complete.\n"
            "NEXT STEPS:\n"
            '1. <skill>browser(action="capture", session="track1")</skill> — check current state\n'
            '2. If still no result: try scrolling down or waiting: <skill>browser(action="scroll", '
            'direction="down", session="track1")</skill>\n'
            f'3. If site is broken: search for an alternative carrier page via '
            f'<skill>web_search(query="tracking {intent.tracking_code or "CODE"} alternative carrier")</skill> '
            f'and retry on the resolved URL.\n'
            "DO NOT respond until actual tracking data is visible."
        )

    if reason == "missing_form_interaction":
        return (
            base
            + f"PROBLEM: You navigated to {target} but did NOT interact with the form.\n"
            "REQUIRED: You must fill in the form fields before submitting.\n"
            "NEXT STEPS:\n"
            '1. <skill>browser(action="capture", session="wf1")</skill> — see the form\n'
            "2. For each field: browser(action=\"type\", selector=\"...\", text=\"...\", session=\"wf1\")\n"
            "3. Then click submit\n"
            "DO NOT respond until the form is submitted and confirmation is visible."
        )

    if reason == "missing_data_extraction":
        return (
            base
            + f"PROBLEM: You navigated to {target} but did not extract any data.\n"
            "REQUIRED: The goal is to extract real data from the page, not just load it.\n"
            "NEXT STEPS:\n"
            '1. <skill>browser(action="capture", session="wf1")</skill> — inspect the page\n'
            '2. Use browser(action="find", selector="...", session="wf1") to locate data elements\n'
            '3. Or: browser(action="execute_js", script="return document.body.innerText", '
            'session="wf1") to extract all text\n'
            "DO NOT respond until actual data (prices, list items, table rows) is extracted."
        )

    # Generic fallback
    return (
        base
        + f"PROBLEM: Required execution step missing — {reason}.\n"
        f"Target: {target}\n"
        "You MUST complete ALL required steps:\n"
        f"{'1. Navigate  2. Type/fill inputs  3. Click submit  4. Verify result visible' if action_type == 'browser_package_check' else '1. Navigate  2. Interact with page  3. Verify outcome visible'}\n"
        "Continue executing NOW. DO NOT respond until the objective is confirmed complete."
    )


# ── End Step Enforcement Layer ────────────────────────────────────────────────

def _count_browser_interaction_steps(accumulated_results: list) -> int:
    """Count meaningful browser interaction steps beyond simple navigate+capture.

    Uses three signals:
    1. Explicit action keywords in output text (typed, clicked, selected)
    2. Output content that isn't just nav/screenshot → implies interaction
    3. Fallback: ≥3 browser calls means at least 1 interaction happened
    """
    _INTERACTION_KEYWORDS_RE = re.compile(
        r"\b(?:typed?|clicked?|selected?|scrolled?|executed?\s+js|found\s+element|"
        r"filled?\s+(?:in|out)?|submitted?|action[=:]\s*(?:type|click|select|scroll|execute_js|find))\b",
        re.IGNORECASE,
    )
    _NAV_ONLY_RE = re.compile(
        r"^(?:navigated?\s+to|screenshot\s+(?:captured?|saved?|at|taken)|"
        r"page\s+loaded?|took\s+screenshot)",
        re.IGNORECASE,
    )
    browser_ok = [r for r in accumulated_results if r.skill_name == "browser" and r.success]
    count = 0
    for r in browser_ok:
        out = (r.output or "").strip()
        first_line = out.split("\n")[0].strip()
        if _INTERACTION_KEYWORDS_RE.search(out):
            count += 1
        elif out and not _NAV_ONLY_RE.match(first_line):
            # Output has content not starting with pure navigation signal
            if len(out) > 50:  # more than a brief nav confirmation
                count += 1

    # Fallback: ≥3 successful browser calls → at least 1 interaction implied
    if count == 0 and len(browser_ok) >= 3:
        count = len(browser_ok) - 2  # subtract navigate + final capture

    return max(count, 0)


def _verify_objective_achieved(
    intent: "_ActionIntent",
    accumulated_results: list,
) -> tuple[bool, str]:
    """Verify whether the REAL OBJECTIVE was achieved — not just whether steps ran.

    Returns: (objective_achieved, evidence_found)
    The system is the final authority — LLM text is irrelevant here.
    """
    from src.intent.action_classifier import ObjectiveSpec
    spec = intent.objective_spec
    if not spec or not spec.confirmation_patterns:
        return True, "no_spec"  # No spec = unconstrained, allow through

    # Collect all browser output text
    all_output = "\n".join(
        (r.output or "")
        for r in accumulated_results
        if r.skill_name == "browser" and r.success
    )

    if not all_output.strip():
        return False, "no_output"

    # Check confirmation patterns — at least one must match
    for pattern_str in spec.confirmation_patterns:
        try:
            if re.search(pattern_str, all_output, re.IGNORECASE | re.DOTALL):
                return True, f"pattern_matched:{pattern_str[:40]}"
        except re.error:
            pass

    # If doesn't require explicit confirmation, check for substantial content
    if not spec.requires_explicit_confirmation:
        _lines = [l.strip() for l in all_output.split("\n") if l.strip()]
        if len(_lines) >= 5 and len(all_output) >= 200:
            return True, "substantial_content"

    return False, "no_confirmation_evidence"


def _build_execution_audit(accumulated_results: list) -> str:
    """Build a step-by-step audit of what was attempted — for rich failure reporting."""
    steps = []
    for i, r in enumerate(accumulated_results):
        status = "✓" if r.success else "✗"
        skill = r.skill_name or "unknown"
        detail = ""
        if r.success:
            out = (r.output or "").split("\n")[0].strip()[:80]
            detail = f" → {out}" if out else ""
        else:
            err = (r.error or "").split("\n")[0].strip()[:80]
            detail = f" → ERROR: {err}" if err else ""
        steps.append(f"  Step {i+1}: {status} {skill}{detail}")
    return "\n".join(steps) if steps else "  (no steps executed)"


def _check_action_terminal_state(
    intent: "_ActionIntent",
    accumulated_results: list,
) -> tuple[bool, bool, str]:
    """Detect definitive terminal state from actual skill outputs only.

    SYSTEM IS THE FINAL AUTHORITY — not the LLM.
    Validates at the OBJECTIVE level, not the step level.

    Returns: (is_terminal, success, reason_code)
    """
    if not intent.action_commitment:
        return False, False, ""

    # ── Package tracking ─────────────────────────────────────────────────────
    if intent.action_type == "browser_package_check":
        for r in accumulated_results:
            if r.skill_name == "browser" and r.output:
                out = r.output
                if "[TRACK_STATUS: FOUND]" in out:
                    return True, True, "track_found"
                if "[TRACK_STATUS: PARTIAL]" in out:
                    # PARTIAL is only a real success if there's actual result content.
                    # Require "Tracking results for" section to be present with content > 100 chars.
                    _results_section = ""
                    if "Tracking results for" in out:
                        _results_section = out.split("Tracking results for", 1)[1][:500]
                    if len(_results_section.strip()) > 100:
                        return True, True, "track_partial"
                    # PARTIAL with no real content = treat as NOT_FOUND (keep looping)
                    return True, False, "track_partial_no_content"
                if "[TRACK_STATUS: NOT_FOUND]" in out:
                    return True, False, "track_not_found"
                if "[TRACK_STATUS: FAILED]" in out:
                    return True, False, "track_failed"
        return False, False, ""  # No TRACK_STATUS yet — keep looping

    # ── Simple navigation ────────────────────────────────────────────────────
    if intent.action_type == "browser_navigation":
        _nav_ok = any(r.skill_name == "browser" and r.success for r in accumulated_results)
        _shot_ok = any(
            r.skill_name == "browser" and r.success
            and "screenshot" in (r.output or "").lower()
            for r in accumulated_results
        )
        if _nav_ok and _shot_ok:
            return True, True, "nav_captured"
        return False, False, ""

    # ── Form workflows and web data extraction ────────────────────────────────
    if intent.action_type in ("browser_form_workflow", "browser_web_workflow"):
        _any_browser_ok = any(r.skill_name == "browser" and r.success for r in accumulated_results)
        _browser_results = [r for r in accumulated_results if r.skill_name == "browser"]
        _browser_count = len(_browser_results)
        _browser_fail_count = sum(1 for r in _browser_results if not r.success)

        # OBJECTIVE VALIDATION — did we actually achieve the goal?
        _obj_achieved, _evidence = _verify_objective_achieved(intent, accumulated_results)

        if _any_browser_ok and _obj_achieved:
            return True, True, f"objective_confirmed:{_evidence}"

        # Anti-premature: have we been trying long enough to declare failure?
        # ≥4 browser calls all failing = terminal failure
        if _browser_count >= 4 and _browser_fail_count >= 3:
            return True, False, "workflow_exhausted"

        # All non-first browser calls failed (we navigated but interaction fails)
        if _browser_count >= 3 and _browser_fail_count == _browser_count:
            return True, False, "workflow_all_steps_failed"

        # skill_manager already created a skill and browser succeeded = good enough
        _skill_created = any(r.skill_name == "skill_manager" and r.success for r in accumulated_results)
        if _skill_created and _any_browser_ok:
            return True, True, "workflow_skill_persisted"

        return False, False, ""  # keep looping — objective not yet confirmed

    # ── Generic: any successful skill = terminal ─────────────────────────────
    if any(r.success for r in accumulated_results):
        return True, True, "skill_succeeded"
    return False, False, ""


def _build_constrained_final_prompt(
    intent: "_ActionIntent",
    accumulated_results: list,
    success: bool,
    reason: str,
) -> str:
    """Build a tightly-constrained final formatting prompt.

    The LLM may only format the VERIFIED data from skill outputs.
    It cannot invent outcome, drift to other topics, or claim partial execution as success.
    """
    if not success:
        diagnostic = _extract_failure_diagnostic(accumulated_results)
        failure_detail = diagnostic or "No se pudo completar la acción solicitada."
        audit = _build_execution_audit(accumulated_results)
        _total_attempts = len(accumulated_results)
        _obj_desc = getattr(intent.objective_spec, "done_description", "") if hasattr(intent, "objective_spec") else ""
        _obj_line = f"Objective that was NOT achieved: {_obj_desc}\n" if _obj_desc else ""
        return (
            "━━ [TERMINAL: FAILURE — OBJECTIVE NOT ACHIEVED] ━━\n"
            f"{_obj_line}"
            f"Verified failure reason: {failure_detail}\n"
            f"Steps attempted ({_total_attempts} total):\n{audit}\n\n"
            "REQUIRED — report this failure with full transparency:\n"
            "1. State EXACTLY what was attempted (steps above)\n"
            "2. State the EXACT point where it failed\n"
            "3. State why (timeout / blocked / element not found / no confirmation received)\n"
            "4. Do NOT say 'I couldn't' without explaining WHY\n"
            "5. Do NOT invent partial success or fabricate results\n"
            "6. Do NOT mention unrelated topics (weather, crypto, etc.)\n"
            "7. Ask once if they want you to try again with a different approach\n"
            "Respond in the user's language. Maximum 4 sentences."
        )

    if intent.action_type == "browser_package_check":
        code = intent.tracking_code or "the package"
        _track_output = ""
        _status_line = ""
        for r in accumulated_results:
            if r.skill_name == "browser" and r.output and "TRACK_STATUS" in r.output:
                _track_output = r.output
                for line in r.output.split("\n"):
                    if "[TRACK_STATUS:" in line:
                        _status_line = line.strip()
                break
        _results_section = ""
        if "Tracking results for" in _track_output:
            _results_section = _track_output.split("Tracking results for", 1)[1][:600].strip()
        elif _track_output:
            _results_section = _track_output[-600:].strip()

        return (
            "━━ [TERMINAL: SUCCESS — PACKAGE TRACKING COMPLETE] ━━\n"
            f"Tracking code: {code}\n"
            f"Verified result from browser execution:\n{_results_section}\n"
            f"{_status_line}\n\n"
            "REQUIRED — report ONLY the verified data above:\n"
            "1. State the package status using ONLY facts from the result above\n"
            "2. Do NOT claim to know location/carrier if not shown\n"
            "3. Do NOT add weather, prices, or any unrelated information\n"
            "4. Mention the screenshot is attached if one was captured\n"
            "5. Use the user's language. Maximum 5 sentences."
        )

    if intent.action_type in ("browser_form_workflow", "browser_web_workflow"):
        _goal = intent.workflow_objective or "the requested web task"
        _success_outputs = [
            (r.skill_name, (r.output or "")[:500])
            for r in accumulated_results if r.success
        ]
        _data_block = "\n".join(f"[{n}]: {o}" for n, o in _success_outputs[:4])
        _skill_saved = any(r.skill_name == "skill_manager" and r.success for r in accumulated_results)
        _skill_note = "\n5. A reusable skill was created and saved automatically." if _skill_saved else ""
        return (
            "━━ [TERMINAL: SUCCESS — WEB WORKFLOW COMPLETE] ━━\n"
            f"Goal: {_goal}\n"
            f"Verified results from browser execution:\n{_data_block}\n\n"
            "REQUIRED — report ONLY the verified data above:\n"
            "1. Confirm what was accomplished (what action was completed)\n"
            "2. Quote key data from the results (confirmation #, status, extracted text)\n"
            "3. Mention any screenshot captured as visual confirmation\n"
            "4. Do NOT add unrelated information or speculation\n"
            f"{_skill_note}"
            "\nRespond in the user's language. Maximum 5 sentences."
        )

    _success_outputs = [
        (r.skill_name, (r.output or "")[:400])
        for r in accumulated_results if r.success
    ]
    _data_block = "\n".join(f"[{n}]: {o}" for n, o in _success_outputs[:3])
    return (
        "━━ [TERMINAL: SUCCESS] ━━\n"
        f"Verified skill results:\n{_data_block}\n\n"
        "REQUIRED — summarize using ONLY the verified data above:\n"
        "1. Confirm the completed action and its result\n"
        "2. Use ONLY facts from the results — no invention\n"
        "3. Do NOT mention unrelated topics\n"
        "Respond in the user's language. Maximum 4 sentences."
    )


# Domain drift detector — fires during committed action tasks and browser-context turns.
# Patterns are intentionally broad to catch any response that drifts into unrelated topics
# (weather, crypto prices, etc.) when the user is mid-task on a browser action.
_DRIFT_DURING_ACTION_RE = re.compile(
    r"(?:"
    # Weather — Spanish
    r"\bel\s+clima\s+(?:en|de|hoy|actual)\b|"
    r"\bclima\s+(?:actual|de\s+hoy|en\s+\w+)\b|"
    r"\btemperatura\s+(?:es|está|actual|en|de)\s*(?:de\s+)?\d|"
    r"\b\d+\s*°\s*C\b.{0,60}(?:nublado|soleado|lluvia|viento|cielo)|"
    r"\b(?:soleado|nublado|lluvia|tormenta|niebla)\b.{0,40}\b\d+\s*°|"
    r"\bbúsqueda\s+en\s+Google\s+para\s+[\"']?weather\b|"
    # Weather — English
    r"\bthe\s+weather\s+(?:in|is|for|today|currently)\b|"
    r"\btemperature\s+(?:is|in|of|currently)\b.{0,20}\d+\s*°|"
    r"\b(?:cloudy|sunny|rainy|foggy|stormy)\b.{0,40}\b\d+\s*°|"
    r"\bcurrently\s+in\s+\w+.{0,30}(?:temperature|°C|°F)\b|"
    # Crypto prices (unchanged — already working)
    r"\bBTC\s*[:=]\s*\$?[\d,]+|"
    r"\bETH\s*[:=]\s*\$?[\d,]+|"
    r"\bprecio\s+(?:actual\s+de\s+|de\s+|actual\s+)(?:btc|eth|bitcoin|ethereum)\s+(?:es|está)\b|"
    r"\bbitcoin\s+(?:está|is)\s+(?:at\s+\$|en\s+\$|\$)[\d,]+|"
    r"\b(?:btc|bitcoin|eth|ethereum)\s+(?:está\s+en|is\s+at)\s+\$?[\d,]+"
    r")",
    re.IGNORECASE,
)

# ── Phase 7: Execution–Response Consistency ──────────────────────────────────
# Maps action_type → keywords that MUST appear in a successful response.
# If a response claims success but lacks these markers, it is flagged as drifted.
_RESPONSE_CONSISTENCY_REQUIRED: dict[str, list[str]] = {
    "browser_package_check": ["paquete", "package", "rastreo", "tracking", "estado", "status",
                               "tránsito", "transit", "entregado", "delivered"],
    "browser_form_workflow":  ["completado", "completed", "enviado", "sent", "formulario",
                               "form", "confirmación", "confirmation", "resultado", "result"],
    "browser_web_workflow":   ["resultado", "result", "encontré", "found", "página", "page",
                               "información", "information", "datos", "data"],
    "browser_navigation":     ["página", "page", "sitio", "site", "navegué", "navigated",
                               "abrí", "opened", "cargó", "loaded"],
}

# Internal truth: stable system capability descriptor (Section 7 — minimal truth enforcement)
_SYSTEM_IDENTITY_FACTS = (
    "WASP autonomous agent. Capabilities: web browsing, package tracking, "
    "form submission, file operations, search, reminders, scheduling. "
    "Cannot browse without browser skill. Cannot send email without gmail skill."
)

def _check_response_consistency(
    response: str,
    action_type: str,
    terminal_success: bool,
) -> tuple[bool, str]:
    """Check that a success response references execution-relevant keywords.

    Returns (consistent: bool, reason: str).
    Only applies to terminal-success responses.
    """
    if not terminal_success or not response:
        return True, "ok"
    required = _RESPONSE_CONSISTENCY_REQUIRED.get(action_type, [])
    if not required:
        return True, "ok"
    r_lower = response.lower()
    if not any(kw in r_lower for kw in required):
        return False, f"missing_action_keywords for {action_type}"
    return True, "ok"


def _skill_persist_quality_gate(
    intent: "_ActionIntent",
    accumulated_results: list,
    evidence: str,
) -> tuple[bool, str]:
    """Quality gate — decide whether a workflow is worth persisting as a reusable skill.

    Phase 7+8: CONTROLLED skill creation with quality validation.
    Returns (should_persist, reason).
    """
    # Gate 1: Objective must be confirmed (not just steps attempted)
    if "no_confirmation_evidence" in evidence or "no_output" in evidence:
        return False, "gate_fail:no_confirmation_evidence"

    # Gate 2: Must have had real interaction (not just navigate+screenshot)
    _browser_results = [r for r in accumulated_results if r.skill_name == "browser" and r.success]
    if len(_browser_results) < 2:
        return False, "gate_fail:too_few_steps"

    # Gate 3: Workflow must be non-trivial (at least some interaction steps)
    spec = getattr(intent, "objective_spec", None)
    _min_steps = getattr(spec, "min_interaction_steps", 1) if spec else 1
    _interaction_count = _count_browser_interaction_steps(accumulated_results)
    if _interaction_count < _min_steps:
        return False, f"gate_fail:insufficient_interactions({_interaction_count}<{_min_steps})"

    # Gate 4: URL must be stable (not a session token or redirect URL)
    _target = intent.action_target or ""
    if re.search(r"(?:token|session|auth|oauth|redirect|callback|code=|state=)", _target, re.IGNORECASE):
        return False, "gate_fail:session_url_not_deterministic"

    # Gate 5: Workflow should be for an explicit user-defined goal
    _goal = intent.workflow_objective or ""
    if not _goal or len(_goal) < 10:
        return False, "gate_fail:no_clear_objective"

    return True, "gate_pass"


def _build_workflow_persist_prompt(
    intent: "_ActionIntent",
    accumulated_results: list,
    evidence: str = "",
) -> str:
    """Build a prompt that instructs the LLM to persist a successful novel workflow as a skill.

    Only called after quality gate passes (Phase 7+8 — controlled skill creation).
    """
    import re as _re
    _target = intent.action_target or ""
    _goal = intent.workflow_objective or "the web task"

    # Extract the browser call sequence from skill outputs for reference
    _steps: list[str] = []
    for r in accumulated_results:
        if r.skill_name == "browser" and r.success:
            _first_line = (r.output or "").split("\n")[0].strip()[:80]
            if _first_line:
                _steps.append(_first_line)

    _steps_block = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(_steps[:8]))

    # Derive a slug from the target domain
    _domain_m = _re.search(r"(?:https?://)?(?:www\.)?([a-z0-9\-]+)\.", _target, _re.IGNORECASE)
    _domain = _domain_m.group(1).lower().replace("-", "_") if _domain_m else "web"
    _verb_m = _re.search(r"\b(track|book|fill|register|scrape|extract|submit|buy|check)\b", _goal, _re.IGNORECASE)
    _verb = _verb_m.group(1).lower() if _verb_m else "workflow"
    _slug = f"{_verb}_{_domain}"[:40]

    # Build params based on interaction type
    _params = "url" if intent.action_type == "browser_web_workflow" else "url,input_data"

    return (
        f"[WORKFLOW PERSISTENCE — QUALITY GATE PASSED]\n"
        f"Workflow '{_goal}' on {_target} completed with confirmed outcome (evidence: {evidence}).\n"
        f"Execution trace ({len(_steps)} steps):\n{_steps_block}\n\n"
        "Quality criteria met: confirmed outcome, sufficient interaction steps, deterministic URL.\n"
        "Save this as a reusable skill NOW:\n"
        f'<skill>skill_manager(action="create", name="{_slug}", '
        f'description="Automates: {_goal[:80]}", '
        f'params="{_params}", '
        f'code="# Auto-synthesized from verified execution\\n'
        f'# Confirmed outcome: {evidence[:60]}\\n'
        f'# Target: {_target}\\n'
        f'url = kwargs.get(\\"url\\", \\"{_target}\\")\\n'
        f'nav = browser(action=\\"navigate\\", url=url, session=\\"wf1\\")\\n'
        f'# Add interaction steps based on workflow: type, click, select, verify\\n'
        f'shot = browser(action=\\"capture\\", session=\\"wf1\\")\\n'
        f'return SkillResult(skill_name=self.definition().name, success=bool(shot), output=shot)")'
        "</skill>\n\n"
        "After skill_manager succeeds: tell the user the workflow was saved as a reusable skill."
    )


# ── End System-Controlled Execution Engine ────────────────────────────────────

# ── Phase 2 helpers ───────────────────────────────────────────────────────────
#
# Improvement 4: early blocking-state detection
# Improvement 5: corpus normalization for spec evidence matching

# HARD blocks: states that cannot be resolved by any automated retry.
# These force an immediate loop continuation with a strong alternative-approach hint.
_HARD_BLOCKING_PATTERNS: "list[tuple]" = [
    (re.compile(r"\b(?:captcha|recaptcha|i.?m not a robot|solve the puzzle)\b", re.I), "captcha"),
    (re.compile(r"\baccess\s+denied\b|\b403\s*(?:Forbidden|Error)?\b|\bForbidden\b", re.I), "access_denied"),
]

# SOFT blocks: states that may be transient or workable — log hint but allow loop to continue.
# The LLM sees the failure in results_text and naturally adapts; we reinforce with a hint.
_SOFT_BLOCKING_PATTERNS: "list[tuple]" = [
    (re.compile(r"\b(?:log.?in|sign.?in)\s+(?:required|needed|to\s+continue|first)\b", re.I), "login_required"),
    (re.compile(r"\bauthentication\s+(?:required|failed|needed|error)\b", re.I), "auth_required"),
    (re.compile(r"\b(?:rate.?limit(?:ed)?|too\s+many\s+requests)\b|\b429\b", re.I), "rate_limited"),
    (re.compile(r"\bpaywall\b|\bsubscri(?:be|ption)\s+(?:required|to\s+access)\b", re.I), "paywall"),
]

_BLOCKING_STATE_HINTS: "dict[str, str]" = {
    "captcha": (
        "The site requires CAPTCHA — automated browser is blocked. "
        "Try: web_search to find the information, or http_request to a public API."
    ),
    "login_required": (
        "The site requires login. "
        "Try: web_search with site: prefix, or find a public version of this data."
    ),
    "auth_required": (
        "Authentication required — no stored credentials available. "
        "Try: web_search for publicly available data, or an alternative source."
    ),
    "access_denied": (
        "Access denied (403 Forbidden) — site is blocking automated access. "
        "Try: different URL, web_search, or python_exec with requests library."
    ),
    "rate_limited": (
        "Rate limited (429) — too many requests. Stop retrying the same URL. "
        "Try: alternative source, or web_search instead."
    ),
    "paywall": (
        "Content is behind a paywall. "
        "Try: web_search for a public summary or alternative source."
    ),
}


def _detect_blocking_state(results_text: str) -> "tuple[bool, str, str]":
    """Detect execution states that cannot be resolved by retrying the same approach.

    Returns (blocked: bool, hint: str, block_type: str).
    blocked=True    → a blocking pattern matched in skill output
    hint            → actionable alternative-approach message for the LLM
    block_type      → "hard" (captcha/403) or "soft" (auth/rate-limit/paywall) or ""

    HARD: force immediate alternative approach (caller should continue the loop).
    SOFT: advisory only — log hint, allow normal loop to continue.

    High-confidence only — patterns specific enough to avoid false positives.
    Fail-open: returns (False, "", "") on any exception.
    """
    try:
        for _pattern, _reason in _HARD_BLOCKING_PATTERNS:
            if _pattern.search(results_text):
                _hint = _BLOCKING_STATE_HINTS.get(_reason, "Try a different approach.")
                return True, f"[BLOCKED: {_reason}] {_hint}", "hard"
        for _pattern, _reason in _SOFT_BLOCKING_PATTERNS:
            if _pattern.search(results_text):
                _hint = _BLOCKING_STATE_HINTS.get(_reason, "Try a different approach.")
                return True, f"[SOFT-BLOCK: {_reason}] {_hint}", "soft"
    except Exception:
        pass
    return False, "", ""


def _normalize_for_spec(text: str) -> str:
    """Normalize text for ObjectiveSpec evidence matching.

    Strips HTML tags, collapses whitespace, lowercases — improves evidence
    detection without changing the validation logic itself.
    Fail-open: returns original text on any error.
    """
    try:
        _clean = re.sub(r"<[^>]{0,500}>", " ", text)  # strip HTML/XML tags
        _clean = re.sub(r"\s+", " ", _clean)           # collapse whitespace
        return _clean.lower().strip()
    except Exception:
        return text

# ── End Phase 2 helpers ────────────────────────────────────────────────────────

# ── End Action Commitment helpers ──────────────────────────────────────────────

# Model switch patterns — auto-detect model change requests
# Must be specific enough to avoid false positives with "usa shell", "cambia el color", etc.
_MODEL_SWITCH_PATTERNS = [
    # "cambia/cambies tu/el/mi cerebro/modelo/llm a X"
    re.compile(r"\b(?:cambia(?:te|s)?|cambiar)\s+(?:(?:tu|el|la|mi)\s+)?(?:cerebro|modelo|llm|model|brain)\s+(?:a|por)\s+(.+)", re.IGNORECASE),
    # "necesito que cambies/cambia tu/el modelo a X" (subjunctive cambies included)
    re.compile(r"\b(?:necesito|quiero|please|need)\s+(?:(?:que|you)\s+)?(?:cambi(?:a(?:te|r|s)?|e[s]?)|change|switch)\s+(?:(?:tu|el|la|mi|your|my)\s+)?(?:cerebro|modelo|llm|model|brain)\s+(?:a|por|to)\s+(.+)", re.IGNORECASE),
    # "cambia de modelo/cerebro a X" (the "de" variant)
    re.compile(r"\b(?:cambia(?:te|s)?|cambiar)\s+de\s+(?:modelo|cerebro|llm|model|brain)?\s*(?:a|por)\s+(.+)", re.IGNORECASE),
    # "cambia de X a Y" where Y is a known model family
    re.compile(r"\b(?:cambia(?:te|s)?|cambiar)\s+de\s+\S+(?:\s+\S+)?\s+(?:a|por)\s+((?:gpt|claude|gemini|llama|qwen|mistral|deepseek|grok|phi|sonnet|haiku|opus)[\w\s.:-]*)", re.IGNORECASE),
    # "cambia a X" where X looks like a model name (contains known model keywords)
    re.compile(r"\b(?:cambia(?:te|s)?|cambiar|switch)\s+(?:a|to)\s+((?:gpt|claude|gemini|llama|qwen|mistral|deepseek|grok|phi|sonnet|haiku|opus|moonshot|kimi)[\w\s.:-]*)", re.IGNORECASE),
    # "usa/pon/activa modelo X" (must have "modelo" keyword)
    re.compile(r"\b(?:usa(?:r)?|pon(?:er|me)?|activa(?:r)?|use)\s+(?:el\s+)?(?:modelo|model|llm)\s+(.+)", re.IGNORECASE),
    # "usa gpt/claude/etc" (known model families directly after usa)
    re.compile(r"\b(?:usa(?:r)?|pon(?:me)?|activa(?:r)?|use)\s+((?:gpt|claude|gemini|llama|qwen|mistral|deepseek|grok|phi|sonnet|haiku|opus|moonshot|kimi)[\w\s.:-]*)", re.IGNORECASE),
    # "quiero usar X" / "want to use X"
    re.compile(r"\b(?:quiero|want)\s+(?:que\s+)?(?:usar?|use)\s+((?:gpt|claude|gemini|llama|qwen|mistral|deepseek|grok|phi|sonnet|haiku|opus|moonshot|kimi)[\w\s.:-]*)", re.IGNORECASE),
    # "switch to X" (English)
    re.compile(r"\bswitch\s+to\s+(.+)", re.IGNORECASE),
    # "change model to X" (English)
    re.compile(r"\bchange\s+(?:(?:my|your|the)\s+)?(?:model|brain|llm)\s+to\s+(.+)", re.IGNORECASE),
    # "ponme gpt 4o" - direct model name after ponme
    re.compile(r"\bpon(?:me|er)?\s+((?:gpt|claude|gemini|llama|qwen|mistral|deepseek|grok|phi|sonnet|haiku|opus|moonshot|kimi)[\w\s.:-]*)", re.IGNORECASE),
]

# Canonical model name aliases for fuzzy matching
_MODEL_ALIASES = {
    "gpt-4o-mini": ["gpt 4o mini", "gpt4o mini", "gpt-4o-mini", "gpt4omini", "4o mini", "4o-mini",
                    "gpt 40 mini", "gpt40 mini", "gpt40mini", "40 mini", "4o mini"],
    "gpt-4o": ["gpt 4o", "gpt4o", "gpt-4o", "4o", "gpt 40", "gpt40"],
    "gpt-4": ["gpt 4", "gpt-4", "gpt4"],
    "gpt-3.5-turbo": ["gpt 3.5", "gpt-3.5", "gpt3.5", "3.5 turbo"],
    "claude-3-5-sonnet-20241022": ["claude sonnet", "claude 3.5 sonnet", "sonnet"],
    "claude-3-5-haiku-20241022": ["claude haiku", "claude 3.5 haiku", "haiku"],
    "claude-3-opus-20240229": ["claude opus", "claude 3 opus", "opus"],
    "gemini-2.0-flash": ["gemini flash", "gemini 2 flash", "gemini-flash"],
    "gemini-1.5-pro": ["gemini pro", "gemini 1.5 pro"],
    "grok-2": ["grok 2", "grok2", "grok-2"],
    "grok-beta": ["grok beta", "grok-beta"],
    "moonshot-v1-128k": ["moonshot 128k", "moonshot-128k", "kimi 128k", "kimi largo"],
    "moonshot-v1-32k": ["moonshot 32k", "moonshot-32k", "kimi 32k", "kimi"],
    "moonshot-v1-8k": ["moonshot 8k", "moonshot-8k", "kimi 8k"],
}

def _match_model_name(user_input: str, available_models: list[str]) -> str | None:
    """Fuzzy match a user's model request to an available model name."""
    clean = user_input.lower().strip().rstrip("?!.")
    # Direct match
    if clean in available_models:
        return clean
    # Substring match (e.g. "qwen" matches "qwen2.5:1.5b")
    for model in available_models:
        if clean in model.lower() or model.lower() in clean:
            return model
    # Alias match
    for canonical, aliases in _MODEL_ALIASES.items():
        if any(alias in clean for alias in aliases):
            # Check if canonical is available
            if canonical in available_models:
                return canonical
            # Check if any available model starts with the canonical prefix
            for model in available_models:
                if model.startswith(canonical.split("-")[0]):
                    return model
    return None

def _normalize_accents(text: str) -> str:
    """Strip diacritics for accent-insensitive matching (e.g. á→a, é→e)."""
    import unicodedata
    return "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )


def _detect_model_switch(text: str) -> str | None:
    """Detect if user wants to switch model. Returns extracted model name or None."""
    # Normalize accents so 'cámbiate' matches 'cambia' pattern
    text_norm = _normalize_accents(text)
    for pattern in _MODEL_SWITCH_PATTERNS:
        m = pattern.search(text_norm)
        if m:
            candidate = m.group(1).strip().rstrip("?!.")
            # Filter out false positives (too short, or common non-model words)
            if len(candidate) < 2 or candidate.lower() in {"eso", "esto", "algo", "otro", "tema", "modo"}:
                continue
            return candidate
    return None


# API key patterns for auto-detection in chat messages
_API_KEY_PATTERNS = [
    ("anthropic", re.compile(r"(sk-ant-[A-Za-z0-9_-]{20,})")),
    ("openai", re.compile(r"(sk-[A-Za-z0-9_-]{20,})")),
    ("google", re.compile(r"(AIza[A-Za-z0-9_-]{20,})")),
    ("xai", re.compile(r"(xai-[A-Za-z0-9_-]{20,})")),
]

PROVIDER_LABELS = {
    "openai": "OpenAI",
    "openai-codex": "OpenAI Codex OAuth",
    "anthropic": "Anthropic",
    "google": "Google Gemini",
    "xai": "xAI (Grok)",
}


def _detect_api_key(text: str) -> tuple[str, str] | None:
    """Detect an API key in text. Returns (provider, key) or None."""
    for provider, pattern in _API_KEY_PATTERNS:
        match = pattern.search(text)
        if match:
            return provider, match.group(1)
    return None


# Agent creation patterns — intercept before LLM so the skill is always called
# Patterns are intentionally strict to avoid false positives on complaint messages
# (e.g. "el nuevo agente no funciona" should NOT create an agent)
_AGENT_CREATE_PATTERNS = [
    # Explicit create verb + "un/una" + agent/bot — requires all three parts
    re.compile(r"\b(?:crea(?:r|me|lo)?|lanza(?:r)?)\s+(?:un|una)\s+(?:agente|bot|sub.agente|asistente\s+autónomo)\b", re.IGNORECASE),
    re.compile(r"\b(?:create|spawn|make|launch)\s+(?:a|an)\s+(?:agent|bot|sub.?agent|autonomous\s+assistant)\b", re.IGNORECASE),
    # "quiero/necesito un agente que..." — must have "que" to indicate purpose
    re.compile(r"\b(?:quiero|necesito)\s+(?:un|una)\s+(?:agente|bot)\s+que\b", re.IGNORECASE),
    # "start an agent" with explicit start intent
    re.compile(r"\bstart\s+(?:a\s+)?new\s+(?:agent|bot)\b", re.IGNORECASE),
]

# Phrases that veto agent creation even if a pattern matched (complaint/question phrases)
_AGENT_CREATE_VETO_PATTERNS = [
    re.compile(r"\bno\s+(?:funciona|está\s+funcionando|esta\s+funcionando|sirve|corre)\b", re.IGNORECASE),
    re.compile(r"\b(?:el|la|este|esta|ese|esa)\s+(?:agente|bot)\b.{0,30}\b(?:no|falla|error|roto|broken|fail)\b", re.IGNORECASE),
    re.compile(r"\b(?:problema|error|fallo|bug|issue)\b.{0,30}\b(?:agente|bot|agent)\b", re.IGNORECASE),
    re.compile(r"\bqué?\s+pasó\b", re.IGNORECASE),
    re.compile(r"\bno\s+(?:funciona|functioned|worked|running)\b", re.IGNORECASE),
    # Questions about inability: "por que no puedes crear", "why can't you create"
    re.compile(r"\bpor\s+qu[eé]\s+no\b", re.IGNORECASE),
    re.compile(r"\bwhy\s+(?:can'?t|cannot|couldn'?t|don'?t|didn'?t)\b", re.IGNORECASE),
    re.compile(r"\bno\s+puedes?\b", re.IGNORECASE),
    re.compile(r"\bno\s+pud(?:e|iste|o|imos)\b", re.IGNORECASE),
    # Conditional phrasing — user is offering a suggestion, not commanding
    re.compile(r"\bsi\s+(?:lo\s+)?consideras\b", re.IGNORECASE),
    re.compile(r"\bsi\s+(?:lo\s+)?crees?\s+(?:necesario|conveniente)\b", re.IGNORECASE),
    re.compile(r"\bif\s+(?:you\s+)?(?:think|consider|deem|feel)\s+it\b", re.IGNORECASE),
]

_AGENT_NAME_PATTERNS = [
    # Multi-word agent names up to 5 tokens. NON-greedy `{0,4}?` so the
    # shortest valid match wins — required because the end-of-string `$`
    # in the lookahead would otherwise allow the longest possible match
    # ("Mi Agente que monitoree algo" instead of "Mi Agente"). Stop words
    # are clause connectors only (no content words).
    re.compile(
        r'\b(?:llamado|llamada|ll[áa]malo|ll[áa]mala)\s+["\']([\w\s-]{1,40})["\']'
        r'|\b(?:llamado|llamada|ll[áa]malo|ll[áa]mala)\s+'
        r'([\w-]+(?:\s+[\w-]+){0,4}?)(?=\s+(?:para|que|y|con|de|en)\b|[,.!?]|$)',
        re.IGNORECASE,
    ),
    re.compile(
        r'\b(?:que\s+se\s+llame|named?)\s+["\']([\w\s-]{1,40})["\']'
        r'|\b(?:que\s+se\s+llame|named?)\s+'
        r'([\w-]+(?:\s+[\w-]+){0,4}?)(?=\s+(?:para|that|to|y|with|on|in|for)\b|[,.!?]|$)',
        re.IGNORECASE,
    ),
    re.compile(
        r'\bse\s+llame\s+["\']([\w\s-]{1,40})["\']'
        r'|\bse\s+llame\s+'
        r'([\w-]+(?:\s+[\w-]+){0,4}?)(?=\s+(?:para|que|y|con|de|en)\b|[,.!?]|$)',
        re.IGNORECASE,
    ),
]

_AGENT_STOP_WORDS = {
    "que", "me", "te", "se", "el", "la", "los", "las", "un", "una", "de", "del",
    "en", "por", "para", "con", "y", "o", "a", "e", "al", "los", "las",
    "the", "an", "is", "are", "to", "of", "and", "or", "my",
    "cada", "todos", "todo", "mis", "su", "sus", "este", "esta", "como",
    "cual", "crea", "crear", "quiero", "necesito", "lanza", "agente", "bot",
}


def _extract_agent_params(text: str) -> dict:
    """Heuristically extract agent name/description/identity_prompt from user text."""
    # Try to extract an explicit name from "llamado X" / "named X" patterns
    name: str | None = None
    for p in _AGENT_NAME_PATTERNS:
        m = p.search(text)
        if m:
            # Each pattern has 2 alternation groups (quoted | unquoted) —
            # take whichever matched.
            raw = m.group(1) or m.group(2) or ""
            name = raw.strip().rstrip(" ,.")
            if name:
                break

    # Extract the purpose — everything after "agente/bot que ..."
    purpose = text.strip()
    _purpose_re = re.compile(
        r'\b(?:agente|bot|sub.agente|agent|assistant)\s+(?:que\s+|para\s+|to\s+)?(.{5,})',
        re.IGNORECASE,
    )
    m = _purpose_re.search(text)
    if m:
        purpose = m.group(1).strip()

    # Generate a CamelCase name from key content words if none was found
    if not name:
        words = re.sub(r"[^\w\s]", "", purpose).split()
        key = [w.capitalize() for w in words if w.lower() not in _AGENT_STOP_WORDS and len(w) > 3][:3]
        name = "".join(key)[:24] if key else "MyAgent"

    # Normalize name: no spaces, max 30 chars
    name = re.sub(r"\s+", "", name.title())[:30]

    return {
        "name": name,
        "description": purpose[:200],
        "identity_prompt": purpose[:500],
        "autonomy_mode": "full",
    }


_AGENT_RUN_NOW_RE = re.compile(
    r"\b(?:"
    r"ejecuta(?:lo|la|r)?\s+(?:ahora|ya|now)|"
    r"(?:run|execute|trigger)\s+(?:it\s+)?now|"
    r"(?:corre|lanza|inicia|arranca)\s+(?:el\s+)?agent[e]?|"
    r"(?:run|start|launch)\s+(?:the\s+)?agent|"
    r"hazlo\s+(?:ahora|ya)|"
    r"(?:ejecuta|lanza)\s+(?:una\s+)?(?:prueba|test|ciclo)|"
    r"prueba(?:lo|la)\s+(?:ahora|ya)|"
    r"probar?\s+(?:el\s+)?agent[e]?|"
    r"test\s+(?:the\s+)?agent|"
    r"ejecuta\s+lo\s+que\s+te\s+ped[íi]|"
    r"haz\s+lo\s+que\s+te\s+dij[ei]|"
    r"haz\s+lo\s+que\s+ped[íi]|"
    r"ejecuta\s+(?:la\s+)?(?:tarea|proceso|monitoreo|tarea\s+programada)|"
    r"corre\s+lo\s+que\s+configuramos|"
    r"ejecuta\s+el\s+(?:proceso|script|programa|monitoreo)"
    r")\b",
    re.IGNORECASE,
)


def _detect_agent_run_now(text: str) -> bool:
    """Return True if the message is asking to run/trigger an agent immediately."""
    return bool(_AGENT_RUN_NOW_RE.search(text))


def _detect_agent_create(text: str) -> dict | None:
    """Return agent creation params dict if the text is an agent creation request, else None.

    Returns None if any veto pattern matches (e.g. complaint about existing agent).
    """
    # Check veto patterns first — complaint phrases override create intent
    for veto in _AGENT_CREATE_VETO_PATTERNS:
        if veto.search(text):
            return None
    for pattern in _AGENT_CREATE_PATTERNS:
        if pattern.search(text):
            return _extract_agent_params(text)
    return None


# Agent listing patterns
_AGENT_LIST_PATTERNS = [
    re.compile(r"\b(?:lista(?:r)?|muestra(?:me)?|ver|show|list)\s+(?:mis\s+|los\s+|all\s+|my\s+)?(?:agentes?|bots?|agents?)\b", re.IGNORECASE),
    re.compile(r"\b(?:que|qué|cuáles?|cuales?)\s+(?:agentes?|bots?)\s+(?:tengo|hay|existen|están?|estan?)\b", re.IGNORECASE),
    re.compile(r"\b(?:what|which)\s+(?:agents?|bots?)\s+(?:do i have|are there|exist)\b", re.IGNORECASE),
    re.compile(r"\bagentes?\s+(?:activos?|disponibles?|creados?)\b", re.IGNORECASE),
]


def _detect_agent_list(text: str) -> bool:
    """Return True if the text requests listing agents."""
    for pattern in _AGENT_LIST_PATTERNS:
        if pattern.search(text):
            return True
    return False


# Agent delete-all patterns — "elimina todos los agentes", "borra todos", "delete all agents"
_AGENT_DELETE_ALL_PATTERNS = [
    # Spanish: "elimina todos los agentes" / "borra los agentes" / "elimina mis agentes"
    re.compile(r"\b(?:elimin[ae](?:r)?|borra(?:r)?|destruy[ae](?:r)?|suprim[ae](?:r)?)\b.{0,20}?\b(?:agentes?|bots?)\b", re.IGNORECASE),
    # English: "delete all agents" / "remove all bots" / "wipe agents"
    re.compile(r"\b(?:delete|remove|destroy|wipe|clear)\s+(?:all\s+(?:the\s+)?)?(?:agents?|bots?)\b", re.IGNORECASE),
    # "todos los agentes" with delete verb anywhere nearby
    re.compile(r"\btodos?(?:\s+(?:mis|los|my))?\s+(?:agentes?|bots?|agents?)\b", re.IGNORECASE),
]

# Agent delete-single patterns — "elimina el agente X", "borra BitcoinMonitor"
_AGENT_DELETE_ONE_PATTERNS = [
    re.compile(r"\b(?:elimin[ae](?:r)?|borra(?:r)?|suprim[ae](?:r)?|remov[ae](?:r)?|borr[ao])\s+(?:el\s+|la\s+|al\s+)?(?:agente|bot|agent)?\s+(\w[\w\s-]{0,29}?)(?=\s*$|,|\.|\?|!)", re.IGNORECASE),
    re.compile(r"\b(?:delete|remove)\s+(?:the\s+)?(?:agent|bot)?\s+(\w[\w\s-]{0,29}?)(?=\s*$|,|\.|\?|!)", re.IGNORECASE),
]


def _detect_agent_delete_all(text: str) -> bool:
    """Return True if the text requests deleting ALL agents."""
    for pattern in _AGENT_DELETE_ALL_PATTERNS:
        if pattern.search(text):
            return True
    return False


_RELATIVE_CLAUSE_WORDS = {
    "que", "el", "la", "los", "las", "cual", "cuál", "the", "which", "that",
    "este", "esta", "ese", "esa", "uno", "una", "hay", "existe", "creado",
    "activo", "activa", "running", "existing", "there",
}

def _detect_agent_delete_one(text: str) -> str | None:
    """Return the agent name/id if the text requests deleting a specific agent, else None."""
    for pattern in _AGENT_DELETE_ONE_PATTERNS:
        m = pattern.search(text)
        if m:
            name = m.group(1).strip().rstrip(" .,!?")
            first_word = name.split()[0].lower() if name.split() else ""
            # Exclude generic words, relative pronouns, and descriptive phrases
            if name.lower() in {"agente", "bot", "agent", "todos", "all", "los", "the"}:
                return None
            if first_word in _RELATIVE_CLAUSE_WORDS:
                return None  # "que esta creado", "el que existe", etc.
            return name
    return None


# ── Task list auto-detect ─────────────────────────────────────────────────────
_TASK_LIST_PATTERNS = [
    re.compile(r"\b(?:que|qué|cuales?|cuáles?|cuantas?|cuántas?|muestra(?:me)?|lista(?:r)?|ver|dame)\s+(?:mis\s+)?(?:tareas?|tasks?)\b", re.IGNORECASE),
    re.compile(r"\btareas?\s+(?:programadas?|activas?|tienes?|tengo|hay|pendientes?)\b", re.IGNORECASE),
    re.compile(r"\b(?:cada\s+cu[aá]nto|qu[eé]\s+intervalo|cu[aá]ndo\s+se\s+ejecuta[n]?)\b.{0,40}(?:tareas?|tasks?)\b", re.IGNORECASE),
    re.compile(r"\btareas?\b.{0,40}\b(?:cada\s+cu[aá]nto|intervalo|ejecuta[n]?)\b", re.IGNORECASE),
    re.compile(r"\blist\s+(?:my\s+)?(?:scheduled\s+)?tasks?\b", re.IGNORECASE),
]

_REMINDER_LIST_PATTERNS = [
    re.compile(r"\b(?:que|qué|cuales?|cuáles?|muestra(?:me)?|lista(?:r)?|ver|dame)\s+(?:mis\s+)?recordatorios?\b", re.IGNORECASE),
    re.compile(r"\brecordatorios?\s+(?:activos?|pendientes?|tienes?|tengo|hay)\b", re.IGNORECASE),
    re.compile(r"\blist\s+(?:my\s+)?reminders?\b", re.IGNORECASE),
]

_REMINDER_DELETE_PATTERNS = [
    re.compile(r"\b(?:elimin[ae](?:r)?|borra(?:r)?|cancela(?:r)?|quita(?:r)?|suprim[ae](?:r)?|delete|remove|cancel)\b.{0,15}?\brecordatorios?\b", re.IGNORECASE),
    re.compile(r"\bdelete\b.{0,10}?\breminder[s]?\b", re.IGNORECASE),
    re.compile(r"\bcancel\b.{0,10}?\breminder[s]?\b", re.IGNORECASE),
]

# Task delete patterns — "elimina la tarea X", "borra monitor_btc_eth", "delete task X"
_TASK_DELETE_PATTERNS = [
    re.compile(r"\b(?:elimin[ae](?:r)?|borra(?:r)?|cancela(?:r)?|suprim[ae](?:r)?|delete|remove)\b\s+(?:la\s+|el\s+|la\s+tarea\s+|the\s+task\s+)?(\S[\w\-_]{1,60})$", re.IGNORECASE),
]


def _detect_task_delete(text: str) -> str | None:
    """Return task name to delete, or None if not a task-delete command."""
    text = text.strip()
    for pattern in _TASK_DELETE_PATTERNS:
        m = pattern.search(text)
        if m:
            name = m.group(1).strip(" .,!?")
            # Skip if the extracted name looks like a reminder/agent keyword
            if re.match(r"^(?:todo|todos|all|agentes?|bots?|recordatorios?|goals?|tareas?|la|el|the)$", name, re.IGNORECASE):
                return None
            return name
    return None


def _detect_reminder_delete(text: str) -> str | None:
    """Return 'all' if deleting all reminders, or extracted keyword for single delete.

    Keyword extraction: strip the delete verb + 'recordatorio/reminder' words,
    leaving the content word(s) that identify which reminder to remove.
    Falls back to 'all' only if explicitly requested or no keyword extracted.
    """
    _all_re = re.compile(r"\btodos?\s+(?:los?\s+)?recordatorios?\b|\ball\s+reminder[s]?\b", re.IGNORECASE)
    _STOP_WORDS = {
        "el", "la", "los", "las", "de", "del", "un", "una", "al", "mi", "su",
        "the", "my", "of", "a", "an",
        "recordatorio", "recordatorios", "reminder", "reminders",
    }
    _VERB_RE = re.compile(
        r"\b(?:elimin[ae](?:r)?|borra(?:r)?|cancela(?:r)?|quita(?:r)?|suprim[ae](?:r)?|delete|remove|cancel)\b",
        re.IGNORECASE,
    )
    for pattern in _REMINDER_DELETE_PATTERNS:
        if pattern.search(text):
            if _all_re.search(text):
                return "all"
            # Extract keyword: remove verb + 'recordatorio/reminder' words, keep content words
            _clean = _VERB_RE.sub("", text.strip())
            _clean = re.sub(r"\brecordatorios?\b|\breminders?\b", "", _clean, flags=re.IGNORECASE)
            _clean = re.sub(r"[.,!?]", "", _clean).strip()
            _words = [w for w in _clean.split() if w.lower() not in _STOP_WORDS and len(w) > 1]
            if _words:
                return " ".join(_words[:4])  # max 4 words for the keyword
            return "all"  # fallback only if no keyword extractable
    return None


def _detect_task_list(text: str) -> bool:
    """Return True if the text is asking to list scheduled tasks."""
    for pattern in _TASK_LIST_PATTERNS:
        if pattern.search(text):
            return True
    return False


def _detect_reminder_list(text: str) -> bool:
    """Return True if the text is asking to list reminders."""
    for pattern in _REMINDER_LIST_PATTERNS:
        if pattern.search(text):
            return True
    return False


# ── Task UPDATE auto-detect ───────────────────────────────────────────────
# Catches "actualiza la tarea X a cada N" / "cambia la tarea X" / "modifica X".
# Without this, an "update" intent is misinterpreted as a CREATE → duplicate
# tasks pile up. Returns (task_name, new_interval) or (None, None).
_TASK_UPDATE_RE = re.compile(
    r"\b(?:actualiza|cambia|modifica|edit[aá]|update)\s+(?:la\s+|the\s+)?(?:tarea|task)\s+"
    r"(?P<name>.+?)"
    r"\s+(?:a|to|por|para)\s+"
    r"(?:cada\s+)?(?P<interval>\d+\s*(?:hora|d[íi]a|minuto|semana|hr|h|hour|day|min|week)\w*)",
    re.IGNORECASE,
)


def _detect_task_update(text: str) -> tuple[str | None, str | None]:
    """If text expresses an update intent for a named task, return
    (task_name, new_interval). Otherwise (None, None)."""
    m = _TASK_UPDATE_RE.search(text)
    if not m:
        return None, None
    name = (m.group("name") or "").strip().rstrip(",.")
    interval = (m.group("interval") or "").strip()
    if not name or not interval:
        return None, None
    # Normalize interval: "6 horas" → "6h" / "every 6h" handled by task_manager
    return name, interval


# ── Gmail auto-detect ─────────────────────────────────────────────────────────
# Gmail App Password format: 4 groups of 4 lowercase letters (e.g. "aaaa bbbb cccc dddd")
_GMAIL_APP_PASSWORD_RE = re.compile(
    r"\b([a-z]{4})\s+([a-z]{4})\s+([a-z]{4})\s+([a-z]{4})\b"
)
_GMAIL_EMAIL_RE = re.compile(r"\b([a-zA-Z0-9._%+\-]+@gmail\.com)\b")

_GMAIL_INBOX_PATTERNS = [
    re.compile(r"\b(?:que|qué|cuales?|cuáles?|muestra(?:me)?|dame|ver|lista(?:r)?|revisa(?:r)?|cheque(?:a)?r?)\s+(?:los?\s+|mis?\s+)?(?:correos?|emails?|mails?|mensajes?)\b", re.IGNORECASE),
    re.compile(r"\b(?:correos?|emails?|mails?)\s+(?:que\s+)?(?:tienes?|hay|tengo|nuevos?|sin\s+leer|recientes?)\b", re.IGNORECASE),
    re.compile(r"\b(?:revisa(?:r)?|abre?|check)\s+(?:el\s+)?(?:inbox|correo|email|mail)\b", re.IGNORECASE),
    re.compile(r"\b(?:que|qué)\s+(?:hay\s+en|tienes?\s+en|hay\s+de\s+nuevo\s+en)\s+(?:el\s+)?(?:correo|email|inbox)\b", re.IGNORECASE),
    re.compile(r"\bcheck\s+(?:my\s+)?(?:inbox|email|mail|gmail)\b", re.IGNORECASE),
]


def _detect_gmail_configure(text: str) -> tuple[str, str] | None:
    """If text contains a Gmail address + app password, return (address, password) else None."""
    email_m = _GMAIL_EMAIL_RE.search(text)
    pw_m = _GMAIL_APP_PASSWORD_RE.search(text)
    if email_m and pw_m:
        address = email_m.group(1)
        password = f"{pw_m.group(1)} {pw_m.group(2)} {pw_m.group(3)} {pw_m.group(4)}"
        return address, password
    return None


def _detect_gmail_inbox(text: str) -> bool:
    """Return True if user is asking to see their emails."""
    for pattern in _GMAIL_INBOX_PATTERNS:
        if pattern.search(text):
            return True
    return False


# ── Behavioral correction detection ──────────────────────────────────────────
_CORRECTION_PATTERNS = [
    re.compile(r"\b(?:estas?|est[aá]s?)\s+(?:alucinando|inventando|mintiendo|equivocad[oa])\b", re.IGNORECASE),
    re.compile(r"\b(?:eso\s+(?:est[aá]\s+mal|no\s+es\s+correcto|no\s+es\s+verdad|es\s+(?:incorrecto|falso|mentira)))\b", re.IGNORECASE),
    re.compile(r"\bpor\s+qu[eé]\s+(?:dices?|dijiste)\s+que\s+no\s+puedes?\b", re.IGNORECASE),
    re.compile(r"\bcomo\s+(?:que|q)\s+no\s+puedes?\b", re.IGNORECASE),
    re.compile(r"\bsi\s+(?:me\s+)?(?:acabas?\s+de\s+decir|dijiste)\s+que\s+(?:pod[ií]as?|s[ií])\b", re.IGNORECASE),
    re.compile(r"\bte\s+(?:equivocaste|equivocas)\b", re.IGNORECASE),
    re.compile(r"\b(?:eso|ese|esa)\s+(?:no\s+est[aá]|no\s+existe|nunca\s+(?:exist[ií]a?|hubo))\b", re.IGNORECASE),
    re.compile(r"\b(?:yo\s+veo|veo\s+m[aá]s|hay\s+m[aá]s|tienes?\s+m[aá]s)\b.{0,30}\bcorreos?\b", re.IGNORECASE),
    re.compile(r"\b(?:no\s+es\s+(?:eso|as[ií]|cierto)|eso\s+no\s+es\s+real)\b", re.IGNORECASE),
    re.compile(r"\b(?:you(?:'re|\s+are)\s+(?:hallucinating|making\s+(?:it|things)\s+up|wrong|incorrect))\b", re.IGNORECASE),
    re.compile(r"\b(?:that(?:'s|\s+is)\s+(?:wrong|incorrect|false|not\s+right|not\s+true))\b", re.IGNORECASE),
    re.compile(r"\bwhy\s+(?:are\s+you\s+saying|did\s+you\s+say)\s+you\s+can't\b", re.IGNORECASE),
]


def _detect_correction(text: str) -> bool:
    """Return True if the user message looks like a correction of the agent's previous response."""
    for pattern in _CORRECTION_PATTERNS:
        if pattern.search(text):
            return True
    return False


# ── Lyrics auto-detect ────────────────────────────────────────────────────────
_LYRICS_PATTERNS = [
    re.compile(r"\b(?:dame|muestra(?:me)?|pon(?:me)?|quiero|necesito|d[ée]jame\s+ver|show\s+me|give\s+me|get\s+me)\s+(?:la\s+)?(?:letra|lyrics|words|canción|song)\b", re.IGNORECASE),
    re.compile(r"\b(?:letra|lyrics)\s+(?:de|of|del?)\s+.+", re.IGNORECASE),
    re.compile(r"\b(?:cantarla|sing\s+along|karaoke)\b", re.IGNORECASE),
    re.compile(r"\b(?:la\s+letra|the\s+lyrics)\s+(?:de|of|completa?|full|entera?)\b", re.IGNORECASE),
]

def _detect_lyrics_request(text: str) -> str | None:
    """If text is a lyrics request, return the best search query, else None."""
    for pattern in _LYRICS_PATTERNS:
        if pattern.search(text):
            return f"{text.strip()} lyrics letra"
    return None


# ── YouTube link auto-detect ───────────────────────────────────────────────────
_YOUTUBE_PATTERNS = [
    re.compile(r"\b(?:dame|muestra(?:me)?|pon(?:me)?|quiero|busca|encuentra|give\s+me|show\s+me|find)\s+(?:el\s+)?(?:link|enlace|url|video)\s+(?:de\s+)?(?:youtube|yt)\b", re.IGNORECASE),
    re.compile(r"\b(?:link|enlace|url|video)\s+(?:de\s+)?(?:youtube|yt)\s+(?:de|of|para|for)\s+.+", re.IGNORECASE),
    re.compile(r"\b(?:ver(?:lo)?|watch|ver\s+en|watch\s+on)\s+(?:youtube|yt)\b", re.IGNORECASE),
    re.compile(r"\b(?:el\s+)?(?:link|video|url)\s+(?:de|para|of|for)\s+(?:youtube|yt|la\s+cancion|the\s+song)\b", re.IGNORECASE),
    re.compile(r"\byoutube\b.{0,30}(?:link|video|url|ver|watch|cancion|song|artista|artist|banda|band)\b", re.IGNORECASE),
    re.compile(r"\b(?:link|video|url|ver|watch).{0,30}\byoutube\b", re.IGNORECASE),
]

def _detect_youtube_request(text: str) -> str | None:
    """If text is a YouTube link request, return the best search query, else None."""
    for pattern in _YOUTUBE_PATTERNS:
        if pattern.search(text):
            return f"{text.strip()} youtube"
    return None


async def _resolve_agent_wakeup(text: str, agent_orchestrator) -> str:
    """Resolve [AGENT_WAKEUP: agent_id=...] signal into an intent-driven execution prompt.

    The agent loads its stored intent and reconstructs the execution workflow from
    it — the task no longer needs to carry the full instruction text.

    Falls back to the raw wakeup text if the agent or intent cannot be loaded,
    ensuring no regression for partially-configured agents.
    """
    import re as _re
    m = _re.match(r'\[AGENT_WAKEUP:\s*agent_id=([^\]\n]+)', text)
    if not m:
        return text

    agent_id = m.group(1).strip()
    try:
        agent = await agent_orchestrator.get_agent(agent_id)
    except Exception:
        agent = None

    if not agent:
        return text  # Agent not found — fall through with raw signal

    from datetime import datetime, timezone as _tz
    try:
        from ..config import get_tz as _get_tz
        _local_now = datetime.now(_tz.utc).astimezone(_get_tz())
        now_str = _local_now.strftime("%Y-%m-%d %H:%M") + f" ({_get_tz().key})"
    except Exception:
        now_str = datetime.now(_tz.utc).strftime("%Y-%m-%d %H:%M UTC")

    intent = agent.intent
    if intent and (intent.description or intent.execution_strategy):
        # Full intent-driven prompt — agent owns the behavior
        constraints_lines = ""
        if intent.constraints:
            constraints_lines = "\n".join(
                f"• {k}: {v}" for k, v in intent.constraints.items()
            )
            constraints_lines = f"\nConstraints:\n{constraints_lines}"

        # Build response format hint from constraints if defined
        _resp_format = ""
        if intent.constraints.get("telegram_format") == "short_summary_only":
            _resp_format = (
                "\n\nIMPORTANT: After completing all tasks, your final text response must be:\n"
                "Resumen:\n"
                "Bitcoin (BTC) - $[actual BTC price] - [actual BTC change%] 24h\n"
                "Ethereum (ETH) - $[actual ETH price] - [actual ETH change%] 24h\n"
                "[one sentence market mood]"
            )

        prompt = (
            f"[TAREA PROGRAMADA: {agent.name}]\n"
            f"You are {agent.name}.\n\n"
            f"MISSION: {intent.description}\n\n"
            f"EXECUTION APPROACH:\n{intent.execution_strategy or 'Execute your mission using available skills.'}"
            f"{constraints_lines}\n\n"
            f"Triggered at {now_str}. Execute your full mission cycle now. "
            f"Do NOT create new tasks or agents — just run the workflow."
            f"{_resp_format}"
        )
    elif agent.identity_prompt:
        # Fallback: use identity_prompt as the intent (backward-compatible)
        prompt = (
            f"[TAREA PROGRAMADA: {agent.name}]\n"
            f"You are {agent.name}.\n\n"
            f"{agent.identity_prompt}\n\n"
            f"Triggered at {now_str}. Execute your full mission cycle now. "
            f"Do NOT create new tasks or agents — just run the workflow."
        )
    else:
        return text  # No intent, no identity_prompt — fall through

    return prompt


def _strip_data_blocks(text: str) -> str:
    """Remove leaked [DATA] blocks from LLM response."""
    cleaned = _DATA_BLOCK_RE.sub("", text)
    # Collapse excessive blank lines
    while "\n\n\n" in cleaned:
        cleaned = cleaned.replace("\n\n\n", "\n\n")
    return cleaned.strip()


def _extract_from_json_response(text: str) -> str:
    """If the LLM responded with raw JSON instead of human text, extract the relevant value.

    Handles:
    - clawd-crawlee: {"status":"SUCCESS","type":"GENERIC","data":"$38,795.16"}
    - Coinbase: {"data":{"base":"BTC","currency":"USD","amount":"38795.16"}}
    - Binance:  {"symbol":"BTCUSDT","price":"38795.16"}
    - markdown-wrapped JSON: ```json\n{...}\n```
    """
    import json as _json
    import re as _re

    stripped = text.strip()

    # Strip markdown code block if present
    md_match = _re.match(r"^```(?:json)?\s*\n([\s\S]*?)\n```$", stripped)
    if md_match:
        stripped = md_match.group(1).strip()

    # Only attempt extraction if the entire response looks like JSON
    if not (stripped.startswith("{") and stripped.endswith("}")):
        return text

    try:
        data = _json.loads(stripped)
    except _json.JSONDecodeError:
        return text

    # clawd-crawlee format: {"status":"SUCCESS","type":"GENERIC","data":"..."}
    if isinstance(data, dict) and data.get("status") == "SUCCESS" and "data" in data:
        val = data["data"]
        if isinstance(val, str):
            return val
        if isinstance(val, dict):
            # Try to find price-like field
            for k in ("amount", "price", "value", "last", "close"):
                if k in val:
                    return str(val[k])

    # Coinbase spot: {"data":{"base":"ETH","currency":"USD","amount":"1934.65"}}
    if isinstance(data, dict) and isinstance(data.get("data"), dict):
        inner = data["data"]
        if "amount" in inner and "base" in inner:
            try:
                price = float(inner["amount"])
                return f"{inner['base']}: ${price:,.2f} {inner.get('currency','USD')}"
            except (ValueError, TypeError):
                pass

    # Binance ticker: {"symbol":"ETHUSDT","price":"1934.65"}
    if isinstance(data, dict) and "price" in data and "symbol" in data:
        try:
            price = float(data["price"])
            symbol = data["symbol"].replace("USDT", "").replace("USD", "")
            return f"{symbol}: ${price:,.2f} USDT"
        except (ValueError, TypeError):
            pass

    # Generic: if single key with a numeric/price string value, return it
    if isinstance(data, dict) and len(data) == 1:
        val = next(iter(data.values()))
        if isinstance(val, (str, int, float)):
            return str(val)

    # Could not extract — return original text unchanged
    return text


async def _do_persist_trace(**kwargs) -> None:
    """Fire-and-forget wrapper for tracer.persist_trace."""
    try:
        from ..observability.tracer import persist_trace
        await persist_trace(**kwargs)
    except Exception:
        pass



# ─────────────────────────────────────────────────────────────────────────────
# _LoopContext — shared state container for _run_llm_loop
# Passed by reference; the method mutates fields in-place so callers can
# read accumulated state (trace_spans, action_history, etc.) after returning.
# ─────────────────────────────────────────────────────────────────────────────
import dataclasses as _dc


@_dc.dataclass
class _LoopContext:
    """All mutable state for one unified LLM round loop execution.

    Passed to _run_llm_loop() which mutates the mutable fields in-place.
    The caller reads accumulated state from the same object after the call.
    """
    # ── Core I/O ──────────────────────────────────────────────────────────────
    messages: list                           # mutable — loop appends rounds
    text: str                                # original user message
    user_id: str
    chat_id: str
    execution_id: str
    image_path: "str | None" = None         # vision — first round only

    # ── Control flags ─────────────────────────────────────────────────────────
    planning_mode: bool = False
    is_scheduled_trigger: bool = False      # suppresses intent completeness engine
    wants_screenshot: bool = False          # enables screenshot-continue hint
    start_time: float = _dc.field(
        default_factory=lambda: __import__("time").monotonic()
    )

    # ── Skills & artifacts (mutated by loop) ──────────────────────────────────
    auto_results: list = _dc.field(default_factory=list)
    exec_artifacts: dict = _dc.field(default_factory=lambda: {"screenshots": []})
    render_report_outputs: dict = _dc.field(default_factory=dict)

    # ── Action intent ─────────────────────────────────────────────────────────
    action_intent: "object | None" = None   # _ActionIntent
    exec_plan: "object | None" = None       # _ExecutionPlan | None

    # ── Progress callback (sync or async) ─────────────────────────────────────
    progress_callback: "object | None" = None

    # ── Accumulated outputs (readable after _run_llm_loop returns) ────────────
    action_history: list = _dc.field(default_factory=list)
    action_all_results: list = _dc.field(default_factory=list)
    action_terminal_detected: bool = False
    action_terminal_success: bool = False
    action_terminal_reason: str = ""
    action_final_prompt_override: str = ""  # consumed once per terminal detection
    action_enforcement_done: bool = False
    intent_retry_done: bool = False
    skill_round_count: int = 0
    trace_spans: list = _dc.field(default_factory=list)
    browser_timed_out: bool = False
    # All skill results executed in the LLM loop (regardless of action-commitment).
    # Used by post-response guards (e.g. _enforce_schedule_honesty) that need to
    # know what really happened this turn — not just action-flow results.
    loop_skill_results: list = _dc.field(default_factory=list)
    # Section 4 — block streak control. If the intent gate blocks the same kind
    # of side-effect repeatedly, stop further LLM rounds for this request.
    intent_block_streak: int = 0
    intent_block_streak_exceeded: bool = False
    # Set when the gate blocks gmail.send for missing_recipient. Post-loop, the
    # response builder substitutes an explicit ask if the LLM did not produce a
    # clear question on its own.
    intent_missing_recipient_seen: bool = False
    # Set when the gate blocks gmail.send for missing_content. Post-loop, the
    # response is substituted with "¿Qué quieres que envíe en el correo?".
    intent_missing_content_seen: bool = False
    media_paths: list = _dc.field(default_factory=list)  # screenshots + media paths
    invalid_photos: set = _dc.field(default_factory=set)  # CAPTURE_VALID:false paths — display only
    last_results_text: str = ""             # results_text from last loop iteration
    active_domain_lock: object = None      # Priority 2: DomainLock persisted across rounds
    # Phase 4: Execution Memory
    pattern_hint: str = ""                 # formatted guidance from matched ExecutionPattern
    reused_pattern_id: str = ""            # ID of pattern injected this turn (for failure tracking)

    # v2.5: Adaptive execution health state (optional — None when not wired)
    health_state: "object | None" = None   # HealthState from runtime.health_state

    # Tool Router: tracks dropped tool calls within this request
    tool_error_flag: bool = False          # True after sanitize_tool_calls drops all calls
    last_tool_error: dict = _dc.field(default_factory=dict)  # {type, tool} of most recent drop

    # ── Correction-loop diminishing-returns guard ─────────────────────────────
    # Counts how many times a meaningful correction prompt was injected this turn.
    # Hard-capped at MAX_MEANINGFUL_CORRECTIONS so a stuck LLM cannot grind through
    # all MAX_SKILL_ROUNDS in correction-only mode.
    correction_count: int = 0
    last_response_hashes: list = _dc.field(default_factory=list)  # last 3 response hashes
    correction_signatures: list = _dc.field(default_factory=list)  # last 3 correction kinds


def verify_execution(action: str, result) -> tuple[bool, str]:
    """Action-aware execution verification.

    Returns (success: bool, reason: str).
    reason is empty string on success, human-readable on failure.

    Actions:
        "email"          — Gmail send confirmed (success flag + confirmation token in output)
        "screenshot"     — File exists on disk and is non-empty
        "browser_capture"— At least one screenshot file produced
        "task_create"    — Task ID present in result output (Redis write confirmed)
        <default>        — result.success == True

    Usage:
        ok, reason = verify_execution("email", result)
        if not ok:
            logger.error("agent.execution_mismatch", action="email", result=..., reason=reason)
    """
    if result is None:
        return False, "result is None"
    if not getattr(result, "success", False):
        _err = getattr(result, "error", "") or ""
        return False, f"skill returned success=False: {_err[:80]}"

    output = getattr(result, "output", "") or ""
    import os as _os, re as _re

    if action == "email":
        # Gmail must have an explicit confirmation token in output
        _confirm_kw = ("sent", "enviado", "message_id", "delivered", "250", "success")
        if not any(kw in output.lower() for kw in _confirm_kw):
            return False, f"email output contains no confirmation token (output={output[:80]})"
        return True, ""

    if action == "screenshot":
        # Single screenshot: path must exist and be > 0 bytes
        paths = _re.findall(r"/data/screenshots/\S+\.png", output)
        if not paths:
            return False, "no screenshot path found in result output"
        path = paths[0]
        if not _os.path.isfile(path):
            return False, f"screenshot file not found on disk: {path}"
        size = _os.path.getsize(path)
        if size == 0:
            return False, f"screenshot file exists but is empty: {path}"
        return True, ""

    if action == "browser_capture":
        # Multiple captures: at least one valid file
        paths = _re.findall(r"/data/screenshots/\S+\.png", output)
        existing = [p for p in paths if _os.path.isfile(p) and _os.path.getsize(p) > 0]
        if not existing:
            return False, f"browser_capture produced no valid screenshot files (found paths={paths})"
        return True, ""

    if action == "task_create":
        # Task was persisted: output should contain a task_id or confirmation
        if not any(kw in output.lower() for kw in ("task_id", "creada", "created", "✓", "saved")):
            return False, f"task_create output has no confirmation (output={output[:80]})"
        return True, ""

    # Default: success flag is sufficient
    return True, ""


def _build_help_text(lang: str = "en") -> str:
    """Build the full /help command output with box-drawing table format.

    Command names always remain in English. Descriptions and section titles
    are translated to the user's language. Currently supports en / es.
    Falls back to English for unknown languages.
    """
    _is_es = lang == "es"

    # ── Translations ──────────────────────────────────────────────────────
    _T = {
        "cmd_col":    "Command"     if not _is_es else "Comando",
        "desc_col":   "Description" if not _is_es else "Descripción",
        "basics":     "Basics"      if not _is_es else "Básicos",
        "model":      "Model"       if not _is_es else "Modelo",
        "memory":     "Memory"      if not _is_es else "Memoria",
        "scheduler":  "Scheduler & Tasks"       if not _is_es else "Scheduler y Tareas",
        "skills":     "Skills"      if not _is_es else "Skills",
        "apis":       "APIs & Integrations"     if not _is_es else "APIs e Integraciones",
        "monitoring": "Web Monitoring"          if not _is_es else "Monitoreo Web",
        "identity":   "Identity"    if not _is_es else "Identidad",
        "openclaw":   "OpenClaw (External Skills)" if not _is_es else "OpenClaw (Skills externos)",
        "infra":      "Infrastructure"          if not _is_es else "Infraestructura",
    }

    def _table(title: str, rows: list[tuple[str, str]]) -> str:
        if not rows:
            return ""
        w_cmd = max(len(r[0]) for r in rows)
        w_desc = max(len(r[1]) for r in rows)
        w_cmd = max(w_cmd, len(_T["cmd_col"]))
        w_desc = max(w_desc, len(_T["desc_col"]))
        top    = f"  ┌─{'─' * w_cmd}─┬─{'─' * w_desc}─┐"
        header = f"  │ {_T['cmd_col']:<{w_cmd}} │ {_T['desc_col']:<{w_desc}} │"
        sep    = f"  ├─{'─' * w_cmd}─┼─{'─' * w_desc}─┤"
        bottom = f"  └─{'─' * w_cmd}─┴─{'─' * w_desc}─┘"
        lines = [f"\n  {title}\n", top, header]
        for cmd, desc in rows:
            lines.append(sep)
            lines.append(f"  │ {cmd:<{w_cmd}} │ {desc:<{w_desc}} │")
        lines.append(bottom)
        return "\n".join(lines)

    # ── Per-language row data ──────────────────────────────────────────────
    if _is_es:
        _d = {
            "ping":          "Health check — responde pong",
            "status":        "Estado completo (LLM, skills, scheduler, memoria)",
            "help":          "Muestra todos los comandos disponibles",
            "introspect":    "Reporte detallado de salud y rendimiento",
            "model_status":  "Info del modelo activo",
            "model_list":    "Modelos instalados (Ollama)",
            "model_avail":   "Modelos descargables",
            "model_switch":  "Cambiar modelo activo",
            "model_default": "Establecer modelo por defecto persistente",
            "model_dl":      "Descargar un modelo",
            "model_del":     "Eliminar un modelo",
            "mem_stats":     "Estadísticas de memoria (por tipo)",
            "mem_recent":    "Últimas n memorias episódicas",
            "mem_search":    "Buscar en memorias",
            "snap_create":   "Crear snapshot cognitivo",
            "snap_list":     "Listar snapshots",
            "sched_list":    "Todos los jobs programados",
            "sched_trig":    "Ejecutar un job ahora",
            "sched_pause":   "Pausar un job",
            "sched_resume":  "Reanudar un job",
            "task_list":     "Listar tareas personalizadas",
            "task_create":   "Crear tarea recurrente",
            "task_edit":     "Ver / editar tarea",
            "task_trig":     "Ejecutar tarea ahora",
            "task_pause":    "Pausar tarea",
            "task_resume":   "Reanudar tarea",
            "task_delete":   "Eliminar tarea",
            "skills_list":   "Lista todos los skills (con estado ON/OFF)",
            "skill_enable":  "Activar un skill",
            "skill_disable": "Desactivar un skill",
            "api_list":      "Providers configurados",
            "api_set":       "Configurar API key",
            "api_remove":    "Eliminar API key",
            "api_test":      "Probar conexión",
            "mon_list":      "Monitores activos",
            "mon_remove":    "Eliminar monitor",
            "id_show":       "Ver identity prompt actual",
            "id_set":        "Actualizar identity prompt",
            "id_reset":      "Resetear al default",
            "id_rollback":   "Restaurar versión anterior",
            "id_versions":   "Listar versiones guardadas",
            "oc_search":     "Buscar en ClawHub",
            "oc_install":    "Instalar skill",
            "oc_list":       "Skills instalados",
            "oc_remove":     "Eliminar skill",
            "br_status":     "Estado del contenedor",
            "br_restart":    "Reiniciar contenedor",
            "br_logs":       "Logs recientes del contenedor",
            "task_arg":      "<nombre> cada <intervalo>: <instrucción>",
        }
    else:
        _d = {
            "ping":          "Health check — responds pong",
            "status":        "Full system snapshot (LLM, skills, scheduler, memory)",
            "help":          "Show all available commands",
            "introspect":    "Detailed health and performance report",
            "model_status":  "Active model info",
            "model_list":    "Installed models (Ollama)",
            "model_avail":   "Downloadable models",
            "model_switch":  "Switch active model",
            "model_default": "Set persistent default model",
            "model_dl":      "Download a model",
            "model_del":     "Delete a model",
            "mem_stats":     "Memory statistics by type",
            "mem_recent":    "Last n episodic memories",
            "mem_search":    "Search memories",
            "snap_create":   "Save a cognitive snapshot",
            "snap_list":     "List snapshots",
            "sched_list":    "All scheduled jobs",
            "sched_trig":    "Run a job now",
            "sched_pause":   "Pause a job",
            "sched_resume":  "Resume a job",
            "task_list":     "List custom tasks",
            "task_create":   "Create a recurring task",
            "task_edit":     "View / edit a task",
            "task_trig":     "Run a task now",
            "task_pause":    "Pause a task",
            "task_resume":   "Resume a task",
            "task_delete":   "Delete a task",
            "skills_list":   "List all skills with ON/OFF status",
            "skill_enable":  "Enable a skill",
            "skill_disable": "Disable a skill",
            "api_list":      "Configured providers",
            "api_set":       "Set an API key",
            "api_remove":    "Remove an API key",
            "api_test":      "Test provider connection",
            "mon_list":      "Active monitors",
            "mon_remove":    "Remove a monitor",
            "id_show":       "View current identity prompt",
            "id_set":        "Update identity prompt",
            "id_reset":      "Reset to default",
            "id_rollback":   "Restore a previous version",
            "id_versions":   "List saved versions",
            "oc_search":     "Search ClawHub",
            "oc_install":    "Install a skill",
            "oc_list":       "Installed skills",
            "oc_remove":     "Remove a skill",
            "br_status":     "Container status",
            "br_restart":    "Restart container",
            "br_logs":       "Recent container logs",
            "task_arg":      "<name> every <interval>: <instruction>",
        }

    parts = [
        _table(_T["basics"], [
            ("/ping",       _d["ping"]),
            ("/status",     _d["status"]),
            ("/help",       _d["help"]),
            ("/introspect", _d["introspect"]),
        ]),
        _table(_T["model"], [
            ("/model status",            _d["model_status"]),
            ("/model list",              _d["model_list"]),
            ("/model available",         _d["model_avail"]),
            ("/model switch <name>",     _d["model_switch"]),
            ("/model default <name>",    _d["model_default"]),
            ("/model download <name>",   _d["model_dl"]),
            ("/model delete <name>",     _d["model_del"]),
        ]),
        _table(_T["memory"], [
            ("/memory stats",            _d["mem_stats"]),
            ("/memory recent [n]",       _d["mem_recent"]),
            ("/memory search <text>",    _d["mem_search"]),
            ("/snapshot create <label>", _d["snap_create"]),
            ("/snapshot list",           _d["snap_list"]),
        ]),
        _table(_T["scheduler"], [
            ("/schedule list",                               _d["sched_list"]),
            ("/schedule trigger <job>",                      _d["sched_trig"]),
            ("/schedule pause <job>",                        _d["sched_pause"]),
            ("/schedule resume <job>",                       _d["sched_resume"]),
            ("/task list",                                   _d["task_list"]),
            (f"/task create {_d['task_arg']}",               _d["task_create"]),
            ("/task edit <name>",                            _d["task_edit"]),
            ("/task trigger <name>",                         _d["task_trig"]),
            ("/task pause <name>",                           _d["task_pause"]),
            ("/task resume <name>",                          _d["task_resume"]),
            ("/task delete <name>",                          _d["task_delete"]),
        ]),
        _table(_T["skills"], [
            ("/skills",                  _d["skills_list"]),
            ("/skill enable <name>",     _d["skill_enable"]),
            ("/skill disable <name>",    _d["skill_disable"]),
        ]),
        _table(_T["apis"], [
            ("/api list",                  _d["api_list"]),
            ("/api set <provider> <key>",  _d["api_set"]),
            ("/api remove <provider>",     _d["api_remove"]),
            ("/api test <provider>",       _d["api_test"]),
        ]),
        _table(_T["monitoring"], [
            ("/monitor list",            _d["mon_list"]),
            ("/monitor remove <url|id>", _d["mon_remove"]),
        ]),
        _table(_T["identity"], [
            ("/identity show",                 _d["id_show"]),
            ("/identity set <text>",           _d["id_set"]),
            ("/identity reset",                _d["id_reset"]),
            ("/identity rollback <timestamp>", _d["id_rollback"]),
            ("/identity versions",             _d["id_versions"]),
        ]),
        _table(_T["openclaw"], [
            ("/openclaw search <query>", _d["oc_search"]),
            ("/openclaw install <slug>", _d["oc_install"]),
            ("/openclaw list",           _d["oc_list"]),
            ("/openclaw remove <slug>",  _d["oc_remove"]),
        ]),
        _table(_T["infra"], [
            ("/broker status <container>",  _d["br_status"]),
            ("/broker restart <container>", _d["br_restart"]),
            ("/broker logs <container>",    _d["br_logs"]),
        ]),
    ]
    return "\n".join(parts).strip()


class EventHandler:
    def __init__(
        self,
        bus: EventBus,
        stream_outgoing: str,
        memory: MemoryManager,
        model_manager: ModelManager,
        skill_registry: SkillRegistry | None = None,
        skill_executor: SkillExecutor | None = None,
        scheduler=None,
        introspector=None,
        broker_client=None,
        identity_manager=None,
        redis_url: str = "",
        goal_orchestrator=None,
        governor=None,
        agent_orchestrator=None,
    ):
        self.bus = bus
        self.stream_outgoing = stream_outgoing
        self.memory = memory
        self.model_manager = model_manager
        self.skill_registry = skill_registry
        self.skill_executor = skill_executor
        self.scheduler = scheduler
        self.introspector = introspector
        self.broker_client = broker_client
        self.identity_manager = identity_manager
        self.redis_url = redis_url
        self.goal_orchestrator = goal_orchestrator
        self.governor = governor
        self._agent_orchestrator = agent_orchestrator
        self._current_chat_id = ""
        # Communication Intelligence Layer — lazy-initialized on first use
        self._formatter = None
        # Communication Intelligence Layer — lazy-initialized
        # Learning loop: track last (user_input, skill_calls_text) per chat for feedback.
        # In-memory dict is the L1 cache; Redis persists across process restarts so
        # behavioral correction detection ("estás alucinando") survives a redeploy.
        # See _set_last_exchange / _get_last_exchange below.
        self._last_exchange: dict[str, tuple[str, str]] = {}
        # Track last browser URL per chat for "continúa hacia abajo" context
        self._last_browser_url: dict[str, str] = {}
        # Per-chat asyncio lock: ensures messages for the same chat are processed sequentially
        self._chat_locks: dict[str, asyncio.Lock] = {}

    async def _set_last_exchange(self, chat_id: str, user_input: str, skills_blob: str) -> None:
        """Cache last (user_input, skills_blob) for chat in memory + Redis.

        Redis persistence is what makes behavioral correction detection
        survive a process restart: without it, _last_exchange is empty
        right after redeploy and corrections like "estás alucinando" miss
        their referent.
        """
        chat_id = str(chat_id)
        self._last_exchange[chat_id] = (user_input, skills_blob)
        if not self.redis_url:
            return
        try:
            import json as _json
            r = aioredis.from_url(self.redis_url, decode_responses=True)
            try:
                await r.setex(
                    f"last_exchange:{chat_id}",
                    24 * 3600,  # keep 24h — covers any plausible restart window
                    _json.dumps({"input": user_input, "skills": skills_blob}),
                )
            finally:
                await r.aclose()
        except Exception:
            pass  # never block on persistence

    async def _get_last_exchange(self, chat_id: str) -> tuple[str, str] | None:
        """Read last exchange from memory; on miss, hydrate from Redis."""
        chat_id = str(chat_id)
        if chat_id in self._last_exchange:
            return self._last_exchange[chat_id]
        if not self.redis_url:
            return None
        try:
            import json as _json
            r = aioredis.from_url(self.redis_url, decode_responses=True)
            try:
                raw = await r.get(f"last_exchange:{chat_id}")
            finally:
                await r.aclose()
            if not raw:
                return None
            data = _json.loads(raw)
            entry = (data.get("input", ""), data.get("skills", ""))
            self._last_exchange[chat_id] = entry  # warm L1
            return entry
        except Exception:
            return None

    def _get_formatter(self):
        """Return the ResponseFormatter, lazily initialized."""
        if self._formatter is None:
            from ..communication.formatter import ResponseFormatter
            self._formatter = ResponseFormatter(self.model_manager)
        return self._formatter

    async def _run_boot_sequence(self, chat_id: str, first_text: str = "") -> str | None:
        """Return boot intro message if agent:is_fresh is set; None otherwise.

        Runs exactly once after a panic reset.
        Boot failsafe: is_fresh is only cleared AFTER the full response is built.
        If anything fails, is_fresh stays set and the sequence retries on the next message.
        Never raises. No LLM calls — pure response.

        first_text: the user's first message — used only to choose intro language.
        """
        if not self.redis_url:
            return None
        try:
            import redis.asyncio as aioredis
            _r = aioredis.from_url(self.redis_url, decode_responses=True, socket_connect_timeout=1)
            try:
                is_fresh = await _r.get("agent:is_fresh")
                if not is_fresh:
                    return None

                # ── Step 1: INTRO — detect language from the first message ──
                _boot_lang = "en"
                try:
                    if first_text:
                        _det, _sigc = _detect_language(first_text)
                        if _det == "es" and _sigc >= 1:
                            _boot_lang = "es"
                except Exception:
                    pass
                _is_es = _boot_lang == "es"
                lines: list[str] = (
                    [
                        "Hola, soy WASP. He sido reinicializado desde cero.",
                        "Toda la memoria cognitiva fue limpiada — estoy partiendo desde un estado limpio.",
                        "",
                    ] if _is_es else [
                        "Hello, I am WASP. I have been fully reinitialized from scratch.",
                        "All cognitive memory has been cleared — starting from a clean state.",
                        "",
                    ]
                )

                # ── Step 2: SYSTEM CHECK ───────────────────────────────────
                checks: list[tuple[str, str, bool]] = []

                # Telegram — implicitly connected (we received this message)
                checks.append(("Telegram", "connected", True))

                # Model liveness ping
                model_name = self.model_manager.active_model or ""
                provider   = self.model_manager.active_provider or ""
                model_ok   = bool(model_name)
                model_live = False
                if model_ok:
                    try:
                        _ping = await asyncio.wait_for(
                            self.model_manager.generate(ModelRequest(
                                messages=[Message(role="user", content="hi")],
                                max_tokens=1,
                            )),
                            timeout=8.0,
                        )
                        model_live = bool(getattr(_ping, "content", None))
                    except Exception:
                        model_live = False
                if model_ok and model_live:
                    checks.append(("Model", f"{model_name} ({provider}) — live", True))
                elif model_ok:
                    checks.append(("Model", f"{model_name} — unreachable", False))
                else:
                    checks.append(("Model", "not configured", False))

                # Gmail
                gmail_ok = False
                gmail_address = ""
                try:
                    _gcreds = await _r.hgetall("gmail:credentials")
                    gmail_address = _gcreds.get("address", "")
                    gmail_ok = bool(gmail_address)
                except Exception:
                    pass
                checks.append(("Email", gmail_address if gmail_ok else "not configured", gmail_ok))

                # Scheduler
                sched_ok = bool(self.scheduler)
                checks.append(("Scheduler", "running" if sched_ok else "not available", sched_ok))

                # Skills
                skill_count = 0
                try:
                    if self.skill_registry and hasattr(self.skill_registry, "_skills"):
                        skill_count = len(self.skill_registry._skills)
                except Exception:
                    pass
                checks.append(("Skills", f"{skill_count} loaded", skill_count > 0))

                # ── Step 3: STATUS OUTPUT ──────────────────────────────────
                lines.append("Diagnóstico del sistema:" if _is_es else "System Diagnostic:")
                all_ok = True
                for name, detail, ok in checks:
                    icon = "✅" if ok else "❌"
                    lines.append(f"  {icon} {name}: {detail}")
                    if not ok:
                        all_ok = False

                lines.append("")

                # ── Step 4: CONFIGURATION FLOW ─────────────────────────────
                if not (model_ok and model_live):
                    if _is_es:
                        lines.append("⚠️  No hay un modelo activo.")
                        lines.append("  Usa /api set anthropic <key> para configurar una API key.")
                        lines.append("  Proveedores disponibles: anthropic, openai, openai-codex, google, xai")
                    else:
                        lines.append("⚠️  No model is active.")
                        lines.append("  Use /api set anthropic <key> to configure an API key.")
                        lines.append("  Available providers: anthropic, openai, openai-codex, google, xai")
                    lines.append("")
                if not gmail_ok:
                    if _is_es:
                        lines.append("ℹ️  Email no configurado.")
                        lines.append("  Para enviar correos, comparte tu dirección de Gmail y app password cuando quieras.")
                    else:
                        lines.append("ℹ️  Email not configured.")
                        lines.append("  To send emails, share your Gmail address and app password when ready.")
                    lines.append("")

                # ── Step 5: CAPABILITIES ───────────────────────────────────
                if _is_es:
                    lines.append("Qué puedo hacer:")
                    lines.append("  Buscar en la web, navegar páginas, tomar capturas")
                    lines.append("  Enviar y leer correos, crear recordatorios y tareas recurrentes")
                    lines.append("  Ejecutar código Python, manejar archivos, monitorear precios")
                    lines.append("  Analizar imágenes, transcribir audio, rastrear envíos")
                else:
                    lines.append("What I can do:")
                    lines.append("  Search the web, browse pages, take screenshots")
                    lines.append("  Send and read emails, create reminders and recurring tasks")
                    lines.append("  Run Python code, manage files, monitor prices")
                    lines.append("  Analyze images, transcribe audio, track packages")
                lines.append("")

                # ── Step 6: READY STATE ────────────────────────────────────
                if all_ok:
                    lines.append(
                        "Todo operativo. Envíame tareas, automatizaciones o preguntas." if _is_es
                        else "Everything is operational. Send me tasks, automations, or questions."
                    )
                else:
                    lines.append(
                        "Listo para asistir. Configura los componentes faltantes cuando quieras." if _is_es
                        else "Ready to assist. Configure any missing components whenever you like."
                    )

                response = "\n".join(lines)

                # ── Step 7: SAFE FLAG CLEAR ────────────────────────────────
                # Only clear after successful response construction.
                # If anything above raised, is_fresh stays set and retries on next message.
                await _r.delete("agent:is_fresh")

                return response

            finally:
                await _r.aclose()

        except Exception:
            logger.exception("handlers.boot_sequence_error")
            # Boot failsafe: do NOT clear is_fresh — retry on next message
            return None

    async def _writeback_task_outcome(
        self,
        *,
        task_id: str,
        task_name: str,
        response_text: str,
        trace_spans: list[dict] | None,
        outcome: str,  # "completed" | "timeout" | "exception"
    ) -> None:
        """Phase 7: idempotent authoritative outcome writeback for scheduled tasks.

        Looks up by task_id first (UUID), falls back to name (legacy/agent path).
        Marks success only if outcome=='completed' AND the response is not a
        failsafe AND any required side-effect skills succeeded. Increments
        failure_count on timeout/exception/failsafe.
        """
        if not self.redis_url or not task_id:
            return
        try:
            from ..scheduler.custom_tasks import (
                get_task as _get_task,
                get_task_by_name as _get_by_name,
                save_task as _save_task,
            )
            _r = aioredis.from_url(self.redis_url, decode_responses=True)
            try:
                _task = await _get_task(_r, task_id)
                if _task is None and task_name:
                    _task = await _get_by_name(_r, task_name)
                if not _task:
                    logger.warning(
                        "custom_task.outcome_writeback_no_task",
                        task_id=task_id, task_name=task_name, outcome=outcome,
                    )
                    return

                # Decide success.
                if outcome == "timeout":
                    _success = False
                    _result_text = f"TIMEOUT after {180}s — no real outcome produced"
                elif outcome == "exception":
                    _success = False
                    _result_text = f"EXCEPTION — handler crashed before completion"
                else:
                    _failsafe_phrases = (
                        "No pude completar", "Intentare nuevamente", "Lo siento",
                        "No pude generar una respuesta", "tardó demasiado",
                    )
                    _is_failsafe = any(p in response_text for p in _failsafe_phrases)
                    spans = trace_spans or []
                    _gmail_ok = any(
                        s.get("skill", "").lower() == "gmail" and s.get("success")
                        for s in spans
                    )
                    _any_skill_ok = any(s.get("success") for s in spans)
                    # If task body asked for email, require gmail success.
                    # Otherwise any skill success counts.
                    _instr = _task.get("instruction", "").lower()
                    _needs_email = (
                        "email" in _instr or "correo" in _instr
                        or "gmail" in _instr or "@" in _instr
                    )
                    if _needs_email:
                        _success = _gmail_ok and not _is_failsafe
                    else:
                        _success = _any_skill_ok and not _is_failsafe
                    _result_text = ("OK" if _success else
                                    (response_text or "")[:120] or "no_response")

                _task["last_success"] = bool(_success)
                _task["last_result"] = _result_text
                if not _success:
                    _task["failure_count"] = int(_task.get("failure_count", 0) or 0) + 1
                else:
                    # Reset on a real success — circuit breaker forgives past failures.
                    _task["failure_count"] = 0

                # B4 fix: circuit breaker — auto-disable after 5 consecutive failures.
                _CB_THRESHOLD = 5
                _EARLY_WARN_THRESHOLD = 3
                _cb_tripped = False
                _early_warn_due = False
                _failure_count = int(_task.get("failure_count", 0) or 0)
                _last_warn_count = int(_task.get("last_warned_at_failure", 0) or 0)
                if (
                    not _success
                    and _failure_count >= _CB_THRESHOLD
                    and _task.get("enabled", True)
                ):
                    _task["enabled"] = False
                    _task["last_result"] = (
                        f"CIRCUIT_BREAKER_TRIPPED — disabled after "
                        f"{_task['failure_count']} consecutive failures. "
                        f"{_result_text[:80]}"
                    )
                    _cb_tripped = True
                # Phase 4: send a single early warning at >=3 failures so the
                # operator notices BEFORE the circuit breaker disables.
                # Dedup via last_warned_at_failure: only warn when the
                # current failure_count is higher than the last one we
                # warned about (so each new streak gets one notification).
                elif (
                    not _success
                    and _failure_count >= _EARLY_WARN_THRESHOLD
                    and _failure_count < _CB_THRESHOLD
                    and _failure_count > _last_warn_count
                    and _task.get("enabled", True)
                ):
                    _early_warn_due = True
                    _task["last_warned_at_failure"] = _failure_count
                if _success:
                    # Reset the warn marker so the next streak gets one warning.
                    _task["last_warned_at_failure"] = 0

                await _save_task(_r, _task)
                logger.info(
                    "custom_task.outcome_recorded",
                    task_id=_task.get("task_id", task_id),
                    name=_task.get("name", ""),
                    outcome=outcome,
                    success=_success,
                    failures=_task.get("failure_count", 0),
                    circuit_breaker_tripped=_cb_tripped,
                )
                # Phase 3 metrics
                try:
                    from ..observability.truth_metrics import bump as _tm_bump
                    await _tm_bump(self.redis_url, f"task_outcome_{outcome}")
                    if _cb_tripped:
                        await _tm_bump(self.redis_url, "task_circuit_breaker_tripped")
                    if _early_warn_due:
                        await _tm_bump(self.redis_url, "task_failing_warning_sent")
                except Exception:
                    pass

                # Notify the user when circuit breaker trips.
                if _cb_tripped:
                    try:
                        from ..communication.translator import apick as _apick_cb
                        _chat_id = str(_task.get("chat_id", "") or "")
                        _user_lang_cb = "en"
                        if _chat_id and self.redis_url:
                            try:
                                _user_lang_cb = (await _r.get(f"user:lang:{_chat_id}")) or "en"
                            except Exception:
                                _user_lang_cb = "en"
                        _cb_msg = await _apick_cb(
                            "task_circuit_breaker", _user_lang_cb,
                            _task.get("task_id", task_id),
                            self.model_manager, self.redis_url,
                            name=_task.get("name", "?"),
                            failures=int(_task.get("failure_count", 0) or 0),
                        )
                        await self.bus.publish(self.stream_outgoing, {
                            "event_type": EventType.TELEGRAM_RESPONSE,
                            "correlation_id": str(uuid4()),
                            "chat_id": _chat_id,
                            "text": _cb_msg,
                        })
                        logger.warning(
                            "custom_task.circuit_breaker_tripped",
                            task_id=_task.get("task_id", task_id),
                            name=_task.get("name", ""),
                            failures=_task.get("failure_count", 0),
                            chat_id=_chat_id,
                        )
                    except Exception as _cb_err:
                        logger.warning(
                            "custom_task.cb_notify_failed",
                            error=str(_cb_err)[:120],
                        )

                # Phase 4: early warning at ≥3 consecutive failures.
                if _early_warn_due:
                    try:
                        from ..communication.translator import apick as _apick_warn
                        _chat_id_w = str(_task.get("chat_id", "") or "")
                        _user_lang_w = "en"
                        if _chat_id_w and self.redis_url:
                            try:
                                _user_lang_w = (await _r.get(f"user:lang:{_chat_id_w}")) or "en"
                            except Exception:
                                _user_lang_w = "en"
                        _reason_short = (_result_text or "")[:80] or "skill failed"
                        _warn_msg = await _apick_warn(
                            "task_failing_warning", _user_lang_w,
                            _task.get("task_id", task_id) + ":warn",
                            self.model_manager, self.redis_url,
                            name=_task.get("name", "?"),
                            failures=_failure_count,
                            reason=_reason_short,
                            threshold=str(_CB_THRESHOLD),
                        )
                        await self.bus.publish(self.stream_outgoing, {
                            "event_type": EventType.TELEGRAM_RESPONSE,
                            "correlation_id": str(uuid4()),
                            "chat_id": _chat_id_w,
                            "text": _warn_msg,
                        })
                        logger.warning(
                            "custom_task.early_warning_sent",
                            task_id=_task.get("task_id", task_id),
                            name=_task.get("name", ""),
                            failures=_failure_count,
                            chat_id=_chat_id_w,
                        )
                    except Exception as _warn_err:
                        logger.warning(
                            "custom_task.early_warn_failed",
                            error=str(_warn_err)[:120],
                        )
            finally:
                await _r.aclose()
        except Exception as _oe:
            logger.warning(
                "custom_task.outcome_writeback_error",
                error=str(_oe)[:120], outcome=outcome,
            )

    async def _safe_publish_response(
        self,
        *,
        text: str,
        chat_id: str,
        correlation_id: str = "",
        skill_results=None,
        user_text: str = "",
        user_lang: str = "en",
        outer_trace=None,
        reason: str = "",
        extra_payload: dict | None = None,
        action_intent=None,
    ) -> str:
        """Single exit point for ALL TELEGRAM_RESPONSE publishes.

        Runs:
          • Honesty Layer (response_binding.apply_honesty_layer) — replaces
            or strips content that mentions topics not grounded in this
            turn's successful skill executions.
          • apply_final_response_policy — schedule honesty + side-effect
            gate + action announcer + factual grounding + markdown sanitize
            + language consistency.

        For paths that don't have skill context (boot, timeout, error
        fallback), pass skill_results=[] — both layers are no-ops.

        Returns the cleaned text actually published. Stamps the trace if
        outer_trace is provided.
        """
        cleaned = text or ""

        # ── Honesty Layer (NEW) ────────────────────────────────────────────────
        # Runs BEFORE the legacy response_guard so subsequent guards work on
        # already-trimmed text. Catches the May 2026 dialog failure mode:
        # response talks about a topic (screenshot/email/...) whose skill
        # never executed successfully in THIS turn.
        _before_honesty = cleaned
        # Phase 5/8: fetch the per-turn snapshot of user-declared stable
        # attributes — the honesty layer uses this as the truth source for
        # response claims about cat names, favourite colours, emails, etc.
        _user_attrs_snapshot: dict[str, str] | None = None
        if chat_id:
            try:
                from ..memory.user_attributes import list_attributes as _ua_list
                from ..db.session import async_session as _ua_async_session
                async with _ua_async_session() as _ua_sess:
                    _user_attrs_snapshot = await _ua_list(_ua_sess, str(chat_id))
            except Exception:
                _user_attrs_snapshot = None
        try:
            from .response_binding import apply_honesty_layer as _honesty
            cleaned, _hon_trace = _honesty(
                cleaned,
                skill_results=skill_results or [],
                user_text=user_text or "",
                user_lang=user_lang or "en",
                action_intent=action_intent,
                user_attributes=_user_attrs_snapshot,
            )
            if outer_trace is not None and _hon_trace.get("status") != "passthrough":
                try:
                    outer_trace.attach_response_guard({"honesty_layer": _hon_trace})
                except Exception:
                    pass
            # Phase 3: per-status truth metrics
            try:
                from ..observability.truth_metrics import bump as _t_bump
                _hl_status = _hon_trace.get("status", "")
                if _hl_status and _hl_status != "passthrough":
                    await _t_bump(self.redis_url, "honesty_layer_applied")
                    _v2_field = {
                        "v2_attribute_truth_override": "honesty_layer_v2_attribute_override",
                        "v2_capability_override": "honesty_layer_v2_capability_override",
                        "v2_replaced_ungrounded_data": "honesty_layer_v2_ungrounded_data",
                        "v2_stripped_ungrounded_data": "honesty_layer_v2_ungrounded_data",
                        "v2_memory_fabrication_stripped": "honesty_layer_v2_memory_fabrication",
                        "replaced": "honesty_layer_replaced_canonical",
                        "stripped": "honesty_layer_stripped",
                    }.get(_hl_status)
                    if _v2_field:
                        await _t_bump(self.redis_url, _v2_field)
            except Exception:
                pass
            # i18n: honesty layer always returns canonical English when it
            # REPLACES content. Translate to user_lang here. Stripped /
            # passthrough results stay as-is (already in user_lang from LLM).
            if (
                _hon_trace.get("__canonical_en")
                and (user_lang or "").lower() not in ("", "en", "en-us", "en-gb")
            ):
                try:
                    from ..communication.translator import translate as _i18n_translate
                    cleaned = await _i18n_translate(
                        cleaned, user_lang, self.model_manager, self.redis_url,
                    )
                except Exception as _tr_err:
                    logger.debug("safe_publish.translate_failed", error=str(_tr_err)[:120])
        except Exception as _hl_err:
            logger.debug("honesty_layer.skip", error=str(_hl_err)[:120])
        _after_honesty = cleaned

        try:
            cleaned, _gtrace = _apply_final_response_policy(
                cleaned,
                user_text=user_text or "",
                skill_results=skill_results or [],
                user_lang=user_lang or "en",
                chat_id=str(chat_id or ""),
                recent_action_resolver=_get_recent_explicit_action,
            )
            if outer_trace is not None:
                try:
                    outer_trace.attach_response_guard(_gtrace)
                except Exception:
                    pass
            # Phase 4/8 debug: log if anything stripped to empty
            if _before_honesty.strip() and not cleaned.strip():
                logger.warning(
                    "safe_publish.policy_stripped_to_empty",
                    chat_id=str(chat_id or ""),
                    reason=reason,
                    before_honesty=_before_honesty[:160],
                    after_honesty=_after_honesty[:160],
                    final=cleaned[:160],
                    policy_trace=_gtrace,
                )
        except Exception:
            # Never fail a publish because of policy — fall back to original.
            cleaned = text or ""

        # Empty/whitespace-only fallback — Telegram drops blank messages
        # silently which is the worst UX (user sees nothing, doesn't know
        # the request even ran). Always send something meaningful.
        if not (cleaned or "").strip():
            # Phase 4/8: log the original text so we can diagnose what
            # stripped it to empty.
            logger.warning(
                "safe_publish.empty_replaced_with_fallback",
                chat_id=str(chat_id or ""),
                reason=reason,
                original_text_preview=(text or "")[:200],
            )
            from ..communication.translator import apick as _i18n_apick
            cleaned = await _i18n_apick(
                "empty_fallback",
                user_lang or "en",
                (correlation_id or text or "")[:60],
                self.model_manager,
                self.redis_url,
            )

        # Phase 1 closure — i18n FINAL GUARD.
        # Last line of defence against LLM-mediated language leaks (e.g. an
        # English refusal sent to a Spanish chat after multi-step replans).
        # Lightweight heuristic detection; only triggers translation when
        # the response language clearly differs from user_lang.
        try:
            from ..communication.translator import final_lang_guard as _fg
            cleaned, _was_guarded = await _fg(
                cleaned, user_lang or "en",
                self.model_manager, self.redis_url,
            )
            if _was_guarded and outer_trace is not None:
                try:
                    outer_trace.attach_response_guard({"i18n_final_guard": True})
                except Exception:
                    pass
        except Exception as _fg_err:
            logger.debug("i18n.final_guard_skip", error=str(_fg_err)[:120])

        payload = {
            "event_type": EventType.TELEGRAM_RESPONSE,
            "correlation_id": correlation_id or "",
            "chat_id": str(chat_id or ""),
            "text": cleaned,
        }
        if extra_payload:
            payload.update(extra_payload)
        try:
            await self.bus.publish(self.stream_outgoing, payload)
            if reason:
                logger.info("telegram_response_published", reason=reason, chat_id=str(chat_id))
        except Exception:
            logger.exception("safe_publish.failed")
        return cleaned

    async def handle(self, msg_id: str, data: dict):
        event_type = data.get("event_type", "")
        chat_id = str(data.get("chat_id", ""))
        start = time.monotonic()

        logger.info(
            "event.received",
            event_type=event_type,
            correlation_id=data.get("correlation_id", ""),
            chat_id=chat_id,
        )

        # Per-chat sequential lock: if another message for this chat is being processed,
        # scheduled tasks wait silently; user messages wait and then execute.
        if chat_id and event_type in (EventType.TELEGRAM_MESSAGE, EventType.TELEGRAM_COMMAND):
            if chat_id not in self._chat_locks:
                self._chat_locks[chat_id] = asyncio.Lock()
            lock = self._chat_locks[chat_id]
            is_scheduled = data.get("user_id") == "scheduled_task"
            if lock.locked() and is_scheduled:
                # Scheduled task arrived while user is being served — skip this cycle,
                # it will fire again on the next interval
                logger.info("chat_lock.scheduled_task_skipped", chat_id=chat_id)
                return
        else:
            lock = None

        # Decision trace — created at the call site so a try/finally below
        # can record it on ANY exit path (early return inside the handler,
        # timeout, exception, normal completion).
        _outer_trace = None
        if event_type == EventType.TELEGRAM_MESSAGE:
            _outer_trace = _new_decision_trace(
                path=("scheduled_task" if data.get("user_id") == "scheduled_task" else "telegram"),
                chat_id=str(data.get("chat_id", "")),
                user_text=data.get("text", ""),
                request_tier=_classify_request_tier(data.get("text", "")),
            )

        async def _execute():
            nonlocal start
            response_text = ""
            _chat_id = str(data.get("chat_id", ""))
            _corr_id  = str(data.get("correlation_id", ""))
            try:
                if event_type == EventType.TELEGRAM_MESSAGE:
                    # Scheduled report tasks need more time: 3 browser captures (~38s)
                    # + LLM rounds + render_report + gmail + validation = up to ~120s.
                    # Browser actions (package tracking, form_submit) need up to ~120s:
                    # Chrome startup ~14s + navigation ~30s + form + results ~40s.
                    # Scheduled triggers get 180s (3 captures + report + gmail).
                    # Regular messages: 150s (was 90s — too short for browser flows).
                    _is_sched = data.get("user_id") == "scheduled_task"
                    _msg_timeout = 180.0 if _is_sched else 150.0
                    try:
                        response_text = await asyncio.wait_for(
                            self._handle_telegram_message(data, _outer_trace=_outer_trace),
                            timeout=_msg_timeout,
                        )
                    finally:
                        # Record the trace on ANY exit (return / timeout /
                        # exception / fast-path early return). Idempotent —
                        # internal callers also flip _recorded so we never
                        # double-write.
                        if _outer_trace is not None and not getattr(_outer_trace, "_recorded", False):
                            _outer_trace._recorded = True
                            try:
                                asyncio.ensure_future(_record_decision_trace(self.redis_url, _outer_trace))
                            except Exception:
                                pass
                elif event_type == EventType.TELEGRAM_COMMAND:
                    response_text = await self._handle_telegram_command(data)
                else:
                    logger.warning("event.unhandled", event_type=event_type)
                    return
            except asyncio.TimeoutError:
                # Phase 4: LLM/handler deadlock — always deliver a response
                _elapsed = int(time.monotonic() - start)
                _input_preview = (data.get("text") or "")[:120]
                # Classify task type for observability
                _task_type = "scheduled" if data.get("user_id") == "scheduled_task" else "interactive"
                logger.error(
                    "agent.timeout",
                    chat_id=_chat_id,
                    event_type=event_type,
                    elapsed_s=_elapsed,
                    input=_input_preview,
                    context="unexpected_timeout",
                    task_type=_task_type,
                )
                from ..communication.translator import apick as _pp_to
                _to_lang_pre = await _get_user_lang(self.redis_url, _chat_id) if _chat_id else "en"
                _timeout_msg = await _pp_to(
                    "outer_timeout", _to_lang_pre, _corr_id or "",
                    self.model_manager, self.redis_url,
                )
                try:
                    _to_lang = _to_lang_pre
                    response_text = await self._safe_publish_response(
                        text=_timeout_msg,
                        chat_id=_chat_id,
                        correlation_id=_corr_id,
                        skill_results=[],
                        user_text=data.get("text", ""),
                        user_lang=_to_lang,
                        outer_trace=_outer_trace,
                        reason="outer_timeout_fallback",
                    )
                except Exception:
                    response_text = _timeout_msg
                # Phase 7: timeout MUST update the scheduled task's outcome.
                if data.get("user_id") == "scheduled_task":
                    _t_match = re.search(
                        r"\[TAREA PROGRAMADA:\s*([^\]]+?)(?:\s*\|\s*id=([0-9a-f-]+))?\s*\]",
                        data.get("text", "") or "",
                    )
                    if _t_match:
                        _t_name = _t_match.group(1).strip()
                        _t_id = (_t_match.group(2) or "").strip() or _t_name
                        try:
                            await self._writeback_task_outcome(
                                task_id=_t_id, task_name=_t_name,
                                response_text=_timeout_msg, trace_spans=[],
                                outcome="timeout",
                            )
                        except Exception:
                            pass
            except Exception as e:
                # Phase 3/5: any unhandled exception must still produce a Telegram message
                logger.exception("event.handler_error")
                from ..communication.translator import apick as _pp_ex
                _ex_lang_pre = await _get_user_lang(self.redis_url, _chat_id) if _chat_id else "en"
                _error_msg = await _pp_ex(
                    "outer_exception", _ex_lang_pre, _corr_id or "",
                    self.model_manager, self.redis_url,
                )
                try:
                    _ex_lang = _ex_lang_pre
                    response_text = await self._safe_publish_response(
                        text=_error_msg,
                        chat_id=_chat_id,
                        correlation_id=_corr_id,
                        skill_results=[],
                        user_text=data.get("text", ""),
                        user_lang=_ex_lang,
                        outer_trace=_outer_trace,
                        reason="outer_exception_fallback",
                    )
                except Exception:
                    response_text = _error_msg
                # Phase 7: exception MUST also update the scheduled task's outcome.
                if data.get("user_id") == "scheduled_task":
                    _t_match = re.search(
                        r"\[TAREA PROGRAMADA:\s*([^\]]+?)(?:\s*\|\s*id=([0-9a-f-]+))?\s*\]",
                        data.get("text", "") or "",
                    )
                    if _t_match:
                        _t_name = _t_match.group(1).strip()
                        _t_id = (_t_match.group(2) or "").strip() or _t_name
                        try:
                            await self._writeback_task_outcome(
                                task_id=_t_id, task_name=_t_name,
                                response_text=_error_msg, trace_spans=[],
                                outcome="exception",
                            )
                        except Exception:
                            pass
            return response_text

        if lock:
            async with lock:
                response_text = await _execute()
        else:
            response_text = await _execute()

        if response_text is None:
            return

        latency_ms = int((time.monotonic() - start) * 1000)

        # Unpack cognitive trace if handler returned a tuple
        cognitive_trace = {}
        if isinstance(response_text, tuple):
            response_text, cognitive_trace = response_text

        # Audit log — redact secrets before persisting
        async with async_session() as session:
            audit = AuditLog(
                id=str(uuid4()),
                event_type=event_type,
                source="telegram",
                action=event_type,
                input_summary=redact(data.get("text", "")[:200]),
                output_summary=redact(response_text[:200]),
                user_id=str(data.get("user_id", "")),
                chat_id=str(data.get("chat_id", "")),
                latency_ms=latency_ms,
                metadata_json=cognitive_trace,
            )
            session.add(audit)
            await session.commit()

    async def _handle_telegram_message(self, data: dict, _outer_trace=None) -> str:
        text = data.get("text", "")
        chat_id = data.get("chat_id", "")
        user_id = data.get("user_id", "")
        correlation_id = data.get("correlation_id", "")
        # Unique execution ID for this request — propagated to all skill calls
        # Enables idempotency (e.g. gmail send dedup) and full lifecycle tracking
        _request_execution_id = str(uuid4())
        # Initialize request-scoped round budget (caps total LLM rounds across
        # cascaded loops in this single user request).
        request_budget_init(_request_execution_id, text)
        # Decision trace — created by the caller (handle._execute) and passed
        # in so the try/finally there records it on any exit path. We mutate
        # this object throughout the request; the caller persists at the end.
        _decision_trace = _outer_trace if _outer_trace is not None else _new_decision_trace(
            path=("scheduled_task" if str(user_id) == "scheduled_task" else "telegram"),
            chat_id=str(chat_id),
            user_text=text,
            request_tier=_classify_request_tier(text),
        )

        # ── Boot sequence: fires exactly once after a panic reset ─────────────
        _boot_msg = await self._run_boot_sequence(str(chat_id), first_text=text)
        if _boot_msg:
            await self.bus.publish(self.stream_outgoing, {
                "event_type": EventType.TELEGRAM_RESPONSE,
                "correlation_id": correlation_id,
                "chat_id": str(chat_id),
                "text": _boot_msg,
            })
            return _boot_msg

        # ── Language detection (non-blocking, fire-and-forget update) ─────────
        # Skip for scheduled tasks — they have no real user language
        _is_scheduled_lang = str(user_id) == "scheduled_task"
        _user_lang = await _get_user_lang(self.redis_url, str(chat_id))
        _lang_is_new_user = (_user_lang == "en")  # True when no prior language stored
        if not _is_scheduled_lang:
            _detected, _sig_count = _detect_language(text)
            if _detected:
                if _detected != _user_lang:
                    if _lang_is_new_user:
                        # Check if a previous first-detection was rejected for this user
                        _had_failed = await _get_lang_failed(self.redis_url, str(chat_id))

                        if _had_failed and _sig_count >= 2:
                            # Recovery path: prior rejection + now strong signal → force accept
                            _user_lang = _detected
                            asyncio.ensure_future(_set_user_lang(self.redis_url, str(chat_id), _detected))
                            asyncio.ensure_future(_clear_lang_failed(self.redis_url, str(chat_id)))
                            logger.info(
                                "lang.recovered_detection",
                                detected=_detected,
                                signal_count=_sig_count,
                                text_sample=text[:100],
                                chat_id=str(chat_id),
                            )
                        else:
                            # Normal path: apply stricter reliability gate
                            _ok, _reason = _is_valid_first_detection(_detected, text, _sig_count)
                            if _ok:
                                _user_lang = _detected
                                asyncio.ensure_future(_set_user_lang(self.redis_url, str(chat_id), _detected))
                                # Clear any stale failed flag on success
                                if _had_failed:
                                    asyncio.ensure_future(_clear_lang_failed(self.redis_url, str(chat_id)))
                                logger.info(
                                    "lang.detected",
                                    lang=_detected,
                                    signal_count=_sig_count,
                                    reason=_reason,
                                    chat_id=str(chat_id),
                                )
                            else:
                                # Mark that detection was attempted but rejected
                                asyncio.ensure_future(_set_lang_failed(self.redis_url, str(chat_id)))
                                logger.info(
                                    "lang.first_detection_rejected",
                                    detected=_detected,
                                    reason=_reason,
                                    text_sample=text[:100],
                                    chat_id=str(chat_id),
                                )
                    else:
                        # Subsequent switch: apply language lock
                        _switch_ok, _block_reason = _should_switch_language(
                            _user_lang, _detected, text, _sig_count
                        )
                        if _switch_ok:
                            _prev_lang = _user_lang
                            _user_lang = _detected
                            asyncio.ensure_future(_set_user_lang(self.redis_url, str(chat_id), _detected))
                            logger.info(
                                "lang.switched",
                                **{"from": _prev_lang, "to": _detected},
                                signal_count=_sig_count,
                                text_sample=text[:100],
                                chat_id=str(chat_id),
                            )
                        else:
                            logger.info(
                                "lang.lock_held",
                                current=_user_lang,
                                detected=_detected,
                                reason=_block_reason,
                                text_sample=text[:100],
                                chat_id=str(chat_id),
                            )

        # Parse media paths from metadata
        # NOTE: bus.py auto-JSON-decodes all fields, so metadata may already be a dict
        try:
            raw_meta = data.get("metadata", {})
            if isinstance(raw_meta, dict):
                metadata = raw_meta
            else:
                metadata = json.loads(raw_meta) if raw_meta else {}
        except Exception:
            metadata = {}
        image_path: str | None = metadata.get("image_path")
        audio_path: str | None = metadata.get("audio_path")
        video_path: str | None = metadata.get("video_path")

        # ── Trivial message fast-path ───────────────────────────────────────
        # Greetings and short social messages skip full context building.
        # Media messages always fall through to the full pipeline.
        if _TRIVIAL_RE.match(text.strip()) and not image_path and not audio_path and not video_path:
            try:
                _lang_name = _LANG_NAMES.get(_user_lang, "English")
                _trivial_sys = (
                    f"You are WASP, an autonomous agent. "
                    f"Respond naturally and briefly to greetings and short social messages. "
                    f"No markdown. No lists. Respond exclusively in {_lang_name}."
                )
                _trivial_req = ModelRequest(messages=[
                    Message(role="system", content=_trivial_sys),
                    Message(role="user", content=text),
                ])
                _trivial_model_resp = await asyncio.wait_for(
                    self.model_manager.generate(_trivial_req),
                    timeout=15.0,
                )
                _resp = (getattr(_trivial_model_resp, "content", None) or "").strip()
                if _resp:
                    # Route through policy — language consistency + scrub of
                    # any unverified action claim (the LLM is small + cheap
                    # and sometimes invents actions).
                    _resp = await self._safe_publish_response(
                        text=_resp,
                        chat_id=str(chat_id),
                        correlation_id=correlation_id,
                        skill_results=[],
                        user_text=text,
                        user_lang=_user_lang,
                        outer_trace=_decision_trace,
                        reason="trivial_fast_path",
                    )
                    return _resp
            except Exception as _te:
                logger.warning("handlers.trivial_fast_path_failed", error=str(_te)[:80])
                # Fall through to full pipeline on any failure

        # Handle video — extract first frame for vision analysis
        if video_path and not image_path:
            try:
                import subprocess, tempfile
                frame_dir = "/data/shared/uploads"
                os.makedirs(frame_dir, exist_ok=True)
                frame_file = os.path.join(frame_dir, f"frame_{int(__import__('time').time())}.jpg")
                result = subprocess.run(
                    ["ffmpeg", "-y", "-i", video_path, "-vf", "select=eq(n\\,0)",
                     "-frames:v", "1", "-q:v", "2", frame_file],
                    capture_output=True, timeout=30,
                )
                if result.returncode == 0 and os.path.getsize(frame_file) > 0:
                    image_path = frame_file
                    text = f"[VIDEO: {os.path.basename(video_path)}]\n\n{text}" if text else f"[VIDEO: {os.path.basename(video_path)}] Describe what you see in this video frame and answer the user's question."
                    logger.info("handlers.video_frame_extracted", frame=frame_file)
                else:
                    text = f"[Video received: {os.path.basename(video_path)}]\n\n{text}" if text else "[Video received — could not analyze visually]"
            except Exception as _ve:
                logger.warning("handlers.video_frame_failed", error=str(_ve))
                text = f"[Video recibido]\n\n{text}" if text else "[Video recibido]"

        # ── Audio transcription pipeline ──────────────────────────────────────────
        # Guarantee: EVERY audio path sends exactly one TELEGRAM_RESPONSE.
        # _audio_dispatched tracks whether we've already sent a final response.
        # The outer try/except is the absolute safety net — no exception can escape
        # without a response being sent.
        if audio_path:
            _audio_length = 0
            _audio_dispatched = False

            try:
                # Phase 4 — entry log
                try:
                    _audio_length = os.path.getsize(audio_path)
                except OSError:
                    _audio_length = 0
                logger.info(
                    "handlers.audio_handler_entered",
                    audio_bytes=_audio_length,
                    path=audio_path,
                )

                # Phase 5 — fail fast on missing / empty / corrupt file
                if _audio_length == 0:
                    _reason = "empty_file" if audio_path else "file_missing"
                    logger.warning("handlers.audio_failure", failure_type=_reason, audio_bytes=0)
                    await self._safe_publish_response(
                        text="El archivo de audio está vacío o no se pudo leer. Intenta enviarlo nuevamente.",
                        chat_id=str(chat_id),
                        correlation_id=correlation_id,
                        skill_results=[],
                        user_text=text,
                        user_lang=_user_lang,
                        outer_trace=_decision_trace,
                        reason=f"audio_{_reason}",
                    )
                    _audio_dispatched = True
                    logger.info("handlers.audio_handler_exited", outcome=_reason)
                    return ""

                if not self.model_manager.supports_audio():
                    await self._safe_publish_response(
                        text=(
                            "El modelo actual no soporta transcripción de audio. "
                            "Configura una API key de OpenAI con /api set openai <key> para habilitar esta función."
                        ),
                        chat_id=str(chat_id),
                        correlation_id=correlation_id,
                        skill_results=[],
                        user_text=text,
                        user_lang=_user_lang,
                        outer_trace=_decision_trace,
                        reason="audio_no_support",
                    )
                    _audio_dispatched = True
                    logger.info("handlers.audio_handler_exited", outcome="no_audio_support")
                    return ""

                # Phase 2 — immediate acknowledgement (user sees feedback before we block)
                try:
                    await self.bus.publish(self.stream_outgoing, {
                        "event_type": EventType.TELEGRAM_PROGRESS,
                        "correlation_id": correlation_id,
                        "chat_id": str(chat_id),
                        "text": "Procesando tu audio…",
                    })
                except Exception:
                    pass  # progress is best-effort; never block on this

                # Phase 1 — non-blocking thread so event loop is never held
                # Phase 3 — hard 12s failsafe via wait_for
                _audio_failure_type: str | None = None
                transcription = ""
                _t0 = time.monotonic()
                logger.info("handlers.transcription_started", audio_bytes=_audio_length)

                try:
                    transcription = await asyncio.wait_for(
                        asyncio.to_thread(
                            self.model_manager.transcribe_audio_sync, audio_path
                        ),
                        timeout=12.0,
                    )
                    _latency_ms = int((time.monotonic() - _t0) * 1000)
                    logger.info(
                        "handlers.transcription_finished",
                        latency_ms=_latency_ms,
                        chars=len(transcription),
                        audio_bytes=_audio_length,
                    )
                except asyncio.TimeoutError:
                    _audio_failure_type = "timeout"
                    logger.warning(
                        "handlers.transcription_timeout",
                        failure_type="timeout",
                        audio_bytes=_audio_length,
                        elapsed_ms=int((time.monotonic() - _t0) * 1000),
                    )
                except Exception as _exc:
                    _audio_failure_type = "error"
                    logger.warning(
                        "handlers.audio_failure",
                        failure_type="error",
                        error=str(_exc)[:120],
                        audio_bytes=_audio_length,
                    )

                # Low-confidence: non-empty but suspiciously short for a large file
                if transcription and not _audio_failure_type:
                    if len(transcription.strip()) < 10 and _audio_length > 20_000:
                        _audio_failure_type = "low_confidence"
                        logger.warning(
                            "handlers.audio_failure",
                            failure_type="low_confidence",
                            chars=len(transcription.strip()),
                            audio_bytes=_audio_length,
                        )
                        transcription = ""
                elif not transcription and not _audio_failure_type:
                    _audio_failure_type = "empty"
                    logger.warning(
                        "handlers.audio_failure",
                        failure_type="empty",
                        audio_bytes=_audio_length,
                    )

                # Phase 1 — success path: inject into text, fall through to LLM
                if transcription:
                    # Use transcription directly as user text — no prefix so LLM treats it as normal speech
                    text = f"{transcription}\n\n{text}".strip() if text else transcription
                    logger.info("handlers.audio_handler_exited", outcome="success_to_llm", chars=len(text))
                    # Falls through to normal LLM processing below
                else:
                    # Phase 1 — all failure paths: respond immediately, return
                    if _audio_failure_type == "timeout":
                        _audio_fail_msg = (
                            "No pude procesar el audio a tiempo. "
                            "Intenta con un mensaje más corto, o envía tu consulta por texto."
                        )
                    elif _audio_failure_type == "low_confidence":
                        _audio_fail_msg = (
                            "No pude procesar correctamente el audio con el modelo actual. "
                            "Puede ser una limitación técnica. "
                            "Intenta con un audio más claro y corto, o envía tu mensaje por texto."
                        )
                    else:
                        _audio_fail_msg = (
                            "No pude entender el audio. "
                            "Intenta con un mensaje más corto y claro, o envíalo por texto."
                        )
                    _audio_fail_msg = await self._safe_publish_response(
                        text=_audio_fail_msg,
                        chat_id=str(chat_id),
                        correlation_id=correlation_id,
                        skill_results=[],
                        user_text=text,
                        user_lang=_user_lang,
                        outer_trace=_decision_trace,
                        reason=f"audio_failure_{_audio_failure_type}",
                    )
                    _audio_dispatched = True
                    logger.info("handlers.audio_handler_exited", outcome=_audio_failure_type)
                    return _audio_fail_msg

                # Phase 2 — global safety: if text still empty after success path (shouldn't happen)
                if not text:
                    _safety_msg = "No pude procesar el audio correctamente. Intenta enviarlo nuevamente."
                    _safety_msg = await self._safe_publish_response(
                        text=_safety_msg,
                        chat_id=str(chat_id),
                        correlation_id=correlation_id,
                        skill_results=[],
                        user_text=text,
                        user_lang=_user_lang,
                        outer_trace=_decision_trace,
                        reason="audio_safety_empty_text",
                    )
                    _audio_dispatched = True
                    logger.warning("handlers.audio_handler_exited", outcome="safety_fallback_empty_text")
                    return _safety_msg

            except Exception as _audio_exc:
                # Phase 3 — absolute safety net: any unhandled exception in audio block
                logger.error(
                    "handlers.audio_handler_exception",
                    error=str(_audio_exc)[:120],
                    audio_bytes=_audio_length,
                )
                if not _audio_dispatched:
                    _exc_msg = "No pude procesar el audio correctamente. Intenta enviarlo nuevamente."
                    try:
                        _exc_msg = await self._safe_publish_response(
                            text=_exc_msg,
                            chat_id=str(chat_id),
                            correlation_id=correlation_id,
                            skill_results=[],
                            user_text=text,
                            user_lang=_user_lang,
                            outer_trace=_decision_trace,
                            reason="audio_exception_fallback",
                        )
                    except Exception:
                        pass  # best effort — can't do more if bus itself is down
                    logger.info("handlers.audio_handler_exited", outcome="exception_fallback")
                    return _exc_msg

        # Learning loop: detect feedback on the previous exchange.
        # Reads via _get_last_exchange so behavioral state survives a
        # process restart (Redis-backed L2 cache).
        feedback = detect_feedback(text)
        if feedback:
            _le = await self._get_last_exchange(chat_id)
            if _le:
                prev_input, prev_skills = _le
                if prev_skills:
                    asyncio.ensure_future(
                        store_example(prev_input, prev_skills, feedback, str(chat_id))
                    )
                    logger.info("learning.feedback_detected", outcome=feedback, chat_id=chat_id)

        # ── Retry-confirm anchor (Bug #8) ─────────────────────────────────
        # Short retry-confirms ("sí", "yes", "ok", "dale") in isolation are
        # ambiguous — the LLM previously could drift to unrelated actions
        # (asking about deleting tasks). When the previous turn was a real
        # request, anchor the current message to it as an explicit retry.
        _retry_confirm_re = re.compile(
            r"^\s*(?:s[íi]|yes|ok|okay|okey|dale|listo|"
            r"int[ée]ntalo\s+de\s+nuevo|try\s+again|otra\s+vez|de\s+nuevo|"
            # "haz lo mismo" / "do the same" / etc. — anchor to prior turn
            # when context exists; low-intent guard catches them otherwise.
            r"haz\s+lo\s+mismo|hazlo\s+(?:de\s+nuevo|otra\s+vez)|"
            r"lo\s+mismo(?:\s+(?:de\s+antes|otra\s+vez))?|"
            r"do\s+the\s+same|do\s+it\s+again|same\s+(?:thing|as\s+before)|again"
            r")\s*[?!.]?\s*$",
            re.IGNORECASE,
        )
        if _retry_confirm_re.match((text or "").strip()):
            _le_rc = await self._get_last_exchange(chat_id)
            if _le_rc:
                _prev_input_rc, _ = _le_rc
                if _prev_input_rc and len(_prev_input_rc.strip()) > 3:
                    # Rewrite text to make the retry explicit. Keeps the gate
                    # patterns aligned with the original intent.
                    text = f"[RETRY OF PREVIOUS: {_prev_input_rc[:200]}] " + text
                    logger.info("retry_confirm.anchored",
                                chat_id=str(chat_id), prev=_prev_input_rc[:60])

        # ── Low-intent guard ─────────────────────────────────────────────
        # If the message is still ultra-short / ambiguous AFTER retry-confirm
        # anchoring (i.e. no prior context to anchor to), short-circuit with
        # a clarification. Without this, the LLM occasionally fabricates
        # arbitrary content (e.g. "ok" alone produced a weather report from
        # training-data noise). This is a CODE-LEVEL gate, not a prompt
        # heuristic — the LLM is never invoked for these inputs.
        if _is_low_intent(text) and not _is_scheduled_lang:
            _li_msg = (
                "¿En qué quieres que te ayude? Te puedo: revisar precios, "
                "programar una tarea, leer tu correo, o capturar un sitio web."
                if (_user_lang or "en") == "es"
                else "What do you need help with? I can: check prices, "
                     "schedule a task, read your email, or capture a site."
            )
            logger.info(
                "low_intent.clarification",
                chat_id=str(chat_id), text_preview=text[:60],
            )
            return await self._safe_publish_response(
                text=_li_msg,
                chat_id=str(chat_id),
                correlation_id=correlation_id,
                skill_results=[],
                user_text=text,
                user_lang=_user_lang,
                outer_trace=_decision_trace,
                reason="low_intent_guard",
            )

        # Behavioral learning: detect corrections and queue for LLM analysis.
        # _get_last_exchange falls back to Redis so correction detection
        # works even right after a redeploy (in-memory cache empty).
        _le_corr = await self._get_last_exchange(chat_id) if _detect_correction(text) else None
        if _le_corr:
            prev_input, _ = _le_corr
            # Retrieve the actual previous agent response from episodic memory
            try:
                from ..memory.behavioral import queue_correction as _queue_correction
                from ..memory.manager import MemoryQuery, MemoryType
                async with async_session() as _bc_session:
                    # Filter by chat_id to avoid cross-chat contamination
                    _recent = await self.memory.retrieve(
                        _bc_session,
                        MemoryQuery(memory_type=MemoryType.EPISODIC, limit=10),
                    )
                    _prev_response = ""
                    for _r in _recent or []:
                        if str(_r.content.get("chat_id", "")) == str(chat_id):
                            _prev_response = _r.content.get("agent_response", "")
                            break
                asyncio.ensure_future(
                    _queue_correction(
                        user_request=prev_input or "",
                        agent_response=_prev_response,
                        user_correction=text,
                        chat_id=str(chat_id),
                    )
                )
                logger.info("behavioral.correction_detected_and_queued", chat_id=chat_id)
            except Exception as _bce:
                logger.warning("behavioral.correction_queue_failed", error=str(_bce))

        # Check vision capability before processing images
        if image_path and not self.model_manager.supports_vision():
            response_text = (
                "El modelo actual no soporta análisis de imágenes.\n"
                "Cambia a un modelo con visión:\n"
                "• GPT-4o: /api set openai <key>\n"
                "• Claude: /api set anthropic <key>\n"
                "• Gemini: /api set google <key>\n"
                "• LLaVA local: /model download llava:7b"
            )
            return await self._safe_publish_response(
                text=response_text,
                chat_id=str(chat_id),
                correlation_id=correlation_id,
                skill_results=[],
                user_text=text,
                user_lang=_user_lang,
                outer_trace=_decision_trace,
                reason="vision_unsupported",
            )

        # Annotate user message with image attachment so the model knows it IS receiving a vision input
        if image_path and self.model_manager.supports_vision():
            img_note = f"[IMAGEN ADJUNTA: {os.path.basename(image_path)}]"
            text = f"{img_note}\n\n{text}".strip() if text else img_note

        # Auto-detect API keys in message — configure and skip LLM
        detected = _detect_api_key(text)
        if detected:
            provider, api_key = detected
            response_text = await self._configure_api_key(provider, api_key)
            return await self._safe_publish_response(
                text=response_text,
                chat_id=str(chat_id),
                correlation_id=correlation_id,
                skill_results=[],
                user_text=text,
                user_lang=_user_lang,
                outer_trace=_decision_trace,
                reason="api_key_configured",
            )

        # Auto-detect model info/switch requests — bypass LLM entirely
        # Model listing: "que modelos tienes?", "que modelos hay?"
        _MODEL_LIST_RE = re.compile(
            r"\b(?:que|qué|cuáles?|cuales?|cuantos?|cuántos?)\s+(?:modelos?|llms?|cerebros?)\s+(?:tienes?|hay|están?|estan?|puedo|disponibles?|instalados?)\b",
            re.IGNORECASE,
        )
        _MODEL_STATUS_RE = re.compile(
            r"\b(?:que|qué|cual|cuál)\s+(?:modelo|llm|cerebro)\s+(?:usas?|tienes?|estás?\s+usando|estas?\s+usando|es)\b",
            re.IGNORECASE,
        )
        if _MODEL_LIST_RE.search(text):
            response_text = await self._handle_model_command(["list"])
            return await self._safe_publish_response(
                text=response_text,
                chat_id=str(chat_id),
                correlation_id=correlation_id,
                skill_results=[],
                user_text=text,
                user_lang=_user_lang,
                outer_trace=_decision_trace,
                reason="model_list",
            )
        if _MODEL_STATUS_RE.search(text):
            response_text = await self._handle_model_command(["status"])
            return await self._safe_publish_response(
                text=response_text,
                chat_id=str(chat_id),
                correlation_id=correlation_id,
                skill_results=[],
                user_text=text,
                user_lang=_user_lang,
                outer_trace=_decision_trace,
                reason="model_status",
            )

        model_request = _detect_model_switch(text)
        if model_request:
            # Get all available models from all providers
            status = self.model_manager.get_status()
            all_models = []
            for pname, pinfo in status["providers"].items():
                all_models.extend(pinfo.get("models", []))

            matched = _match_model_name(model_request, all_models)
            if matched:
                result = await self.model_manager.switch_model(matched)
                new_provider = self.model_manager.active_provider
                response_text = f"✅ Modelo cambiado a **{matched}** ({new_provider}). Listo para ayudarte."
            else:
                # Model not installed — try to suggest or offer to download
                canonical = None
                clean = model_request.lower().strip()
                for canon, aliases in _MODEL_ALIASES.items():
                    if any(alias in clean for alias in aliases) or clean in canon:
                        canonical = canon
                        break
                if canonical:
                    response_text = (
                        f"El modelo {canonical} no está instalado.\n"
                        f"Modelos disponibles: {', '.join(all_models) if all_models else 'ninguno'}\n\n"
                        f"Para configurar un proveedor: /api set <provider> <key>\n"
                        f"Para descargar un modelo local: /model download <name>"
                    )
                else:
                    response_text = (
                        f"No encontré un modelo que coincida con '{model_request}'.\n"
                        f"Modelos disponibles: {', '.join(all_models) if all_models else 'ninguno'}\n\n"
                        f"Usa /model available para ver opciones o /api set para configurar proveedores."
                    )
            return await self._safe_publish_response(
                text=response_text,
                chat_id=str(chat_id),
                correlation_id=correlation_id,
                skill_results=[],
                user_text=text,
                user_lang=_user_lang,
                outer_trace=_decision_trace,
                reason="model_switch",
            )

        # ── AGENT WAKEUP ────────────────────────────────────────────────────
        # When a task is linked to a sub-agent, the scheduler emits a minimal
        # [AGENT_WAKEUP: agent_id=...] signal instead of the full instruction.
        # The system loads the agent, reconstructs its intent, and builds the
        # execution prompt so the agent — not the task — owns the behavior.
        if text.startswith("[AGENT_WAKEUP:") and self._agent_orchestrator:
            text = await _resolve_agent_wakeup(text, self._agent_orchestrator)
            # If resolution fails (agent not found), text is returned unchanged
            # and falls through to the LLM with the raw wakeup signal.

        # Long messages (> 400 chars) are complex instructions — always send to LLM.
        # Short messages get auto-detection for quick commands (list tasks, list agents, etc.)
        _short_msg = len(text) <= 400

        # Auto-detect "run now" — ejecutalo ahora / run it now
        # Prefer task_manager trigger (non-blocking) when tasks exist; fall back to agent_manager.
        # Skip for scheduled task triggers — "EJECUTA AHORA" is an internal directive, not user intent.
        _is_scheduled_trigger = text.startswith("[TAREA PROGRAMADA:")
        if _detect_agent_run_now(text) and self.skill_executor and not _is_scheduled_trigger:
            from ..skills.types import SkillCall as _SC
            from ..scheduler.custom_tasks import list_tasks as _list_tasks, get_task_by_name as _get_task_by_name
            # Check if any tasks exist — if so, trigger the first one via task_manager
            try:
                _r_tmp = aioredis.from_url(self.redis_url, decode_responses=True)
                _existing_tasks = await _list_tasks(_r_tmp)
                # If user named a specific task ("ejecuta la tarea X"), look
                # it up by name and pass that explicit name to trigger so
                # task_manager doesn't silently pick the first task. Without
                # this, "ejecuta la tarea SinEsteNombre" used to trigger
                # whatever task happened to be first in the hash.
                _named_task = None
                _named_re = re.compile(
                    r"\b(?:ejecuta|corre|lanza|trigger|run)(?:r|me)?\s+"
                    r"(?:la\s+|the\s+)?(?:tarea|task)\s+(.+?)(?:\s*[?!.]|$)",
                    re.IGNORECASE,
                )
                _nm_match = _named_re.search(text)
                if _nm_match:
                    _named_candidate = _nm_match.group(1).strip().rstrip(".,!?")
                    if _named_candidate and len(_named_candidate) >= 3:
                        _named_task = await _get_task_by_name(_r_tmp, _named_candidate)
                await _r_tmp.aclose()
            except Exception:
                _existing_tasks = []
                _named_task = None
                _named_candidate = ""
            if _named_task:
                _run_call = _SC(
                    skill_name="task_manager",
                    arguments={"action": "trigger", "name": _named_task["name"]},
                    raw_text=f"[auto-detected: trigger named task '{_named_task['name']}']",
                )
            elif _nm_match and _existing_tasks:
                # User named a task but it was not found / ambiguous —
                # do NOT silently trigger the first task. Honest 404.
                _err_text = (
                    f"No encontré una tarea llamada '{_named_candidate}'. "
                    f"Tareas disponibles: {', '.join(t['name'] for t in _existing_tasks[:5])}."
                )
                return await self._safe_publish_response(
                    text=_err_text,
                    chat_id=str(chat_id),
                    correlation_id=correlation_id,
                    skill_results=[],
                    user_text=text,
                    user_lang=_user_lang,
                    outer_trace=_decision_trace,
                    reason="run_now_named_not_found",
                )
            elif _existing_tasks:
                _task_name = _existing_tasks[0].get("name", "")
                _run_call = _SC(
                    skill_name="task_manager",
                    arguments={"action": "trigger", "name": _task_name},
                    raw_text=f"[auto-detected: trigger task '{_task_name}']",
                )
            else:
                _run_call = _SC(
                    skill_name="agent_manager",
                    arguments={"action": "run_now"},
                    raw_text="[auto-detected: run agent now]",
                )
            _run_results = await self.skill_executor.execute_batch(
                [_run_call], user_id=str(user_id), chat_id=str(chat_id),
                execution_id=_request_execution_id,
            )
            _rr = _run_results[0] if _run_results else None
            response_text = _rr.output if (_rr and _rr.success) else (_rr.output or "No hay tareas ni agentes para ejecutar." if _rr else "No hay tareas ni agentes para ejecutar.")
            return await self._safe_publish_response(
                text=response_text,
                chat_id=str(chat_id),
                correlation_id=correlation_id,
                skill_results=_run_results or [],
                user_text=text,
                user_lang=_user_lang,
                outer_trace=_decision_trace,
                reason="run_now",
            )

        # Auto-detect agent creation requests — bypass LLM, call agent_manager directly
        _agent_params = _detect_agent_create(text) if _short_msg else None
        if _agent_params and self.skill_executor:
            logger.info("agent.auto_create_detected", name=_agent_params.get("name"))
            from ..skills.types import SkillCall as _SC
            _agent_call = _SC(
                skill_name="agent_manager",
                arguments={"action": "create", **_agent_params},
                raw_text=f"[auto-detected: create agent '{_agent_params['name']}']",
            )
            _agent_results = await self.skill_executor.execute_batch(
                [_agent_call], user_id=str(user_id), chat_id=str(chat_id),
                execution_id=_request_execution_id,
            )
            _ar = _agent_results[0] if _agent_results else None
            response_text = (
                _ar.output if (_ar and _ar.success)
                else f"No pude crear el agente: {(_ar.output or _ar.error or 'error desconocido') if _ar else 'error desconocido'}"
            )
            return await self._safe_publish_response(
                text=response_text,
                chat_id=str(chat_id),
                correlation_id=correlation_id,
                skill_results=_agent_results or [],
                user_text=text,
                user_lang=_user_lang,
                outer_trace=_decision_trace,
                reason="agent_create_auto",
            )

        # Auto-detect agent list requests
        if _short_msg and _detect_agent_list(text) and self.skill_executor:
            from ..skills.types import SkillCall as _SC
            _list_call = _SC(
                skill_name="agent_manager",
                arguments={"action": "list"},
                raw_text="[auto-detected: list agents]",
            )
            _list_results = await self.skill_executor.execute_batch(
                [_list_call], user_id=str(user_id), chat_id=str(chat_id),
                execution_id=_request_execution_id,
            )
            _lr = _list_results[0] if _list_results else None
            response_text = _lr.output if (_lr and _lr.success) else "No pude listar los agentes."
            return await self._safe_publish_response(
                text=response_text,
                chat_id=str(chat_id),
                correlation_id=correlation_id,
                skill_results=_list_results or [],
                user_text=text,
                user_lang=_user_lang,
                outer_trace=_decision_trace,
                reason="agent_list",
            )

        # Auto-detect delete a specific agent by name (check BEFORE delete-all)
        _del_name = _detect_agent_delete_one(text) if _short_msg else None
        if _del_name and self.skill_executor:
            from ..skills.types import SkillCall as _SC
            _del_call = _SC(
                skill_name="agent_manager",
                arguments={"action": "delete", "agent_id": _del_name},
                raw_text=f"[auto-detected: delete agent '{_del_name}']",
            )
            _del_results = await self.skill_executor.execute_batch(
                [_del_call], user_id=str(user_id), chat_id=str(chat_id),
                execution_id=_request_execution_id,
            )
            _dr = _del_results[0] if _del_results else None
            response_text = _dr.output if (_dr and _dr.success) else f"No pude eliminar el agente '{_del_name}'."
            return await self._safe_publish_response(
                text=response_text,
                chat_id=str(chat_id),
                correlation_id=correlation_id,
                skill_results=_del_results or [],
                user_text=text,
                user_lang=_user_lang,
                outer_trace=_decision_trace,
                reason="agent_delete_one",
            )

        # Auto-detect delete ALL agents
        if _short_msg and _detect_agent_delete_all(text) and self.skill_executor:
            from ..skills.types import SkillCall as _SC
            # If text also mentions tasks/goals → wipe_all (delete everything atomically)
            _mentions_tasks_or_goals = bool(re.search(r"\b(?:tareas?|tasks?|goals?)\b", text, re.IGNORECASE))
            _del_action = "wipe_all" if _mentions_tasks_or_goals else "delete_all"
            _del_call = _SC(
                skill_name="agent_manager",
                arguments={"action": _del_action},
                raw_text=f"[auto-detected: {_del_action}]",
            )
            _del_results = await self.skill_executor.execute_batch(
                [_del_call], user_id=str(user_id), chat_id=str(chat_id),
                execution_id=_request_execution_id,
            )
            _dr = _del_results[0] if _del_results else None
            response_text = _dr.output if (_dr and _dr.success) else "No pude eliminar los agentes."
            return await self._safe_publish_response(
                text=response_text,
                chat_id=str(chat_id),
                correlation_id=correlation_id,
                skill_results=_del_results or [],
                user_text=text,
                user_lang=_user_lang,
                outer_trace=_decision_trace,
                reason="agent_delete_all",
            )

        # Auto-detect task list queries — bypass LLM, always call task_manager directly
        # Skip for internal scheduled task triggers (they start with [TAREA PROGRAMADA:])
        if _short_msg and _detect_task_list(text) and not text.startswith("[TAREA PROGRAMADA:") and self.skill_executor:
            from ..skills.types import SkillCall as _SC
            _tl_call = _SC(
                skill_name="task_manager",
                arguments={"action": "list"},
                raw_text="[auto-detected: list tasks]",
            )
            _tl_results = await self.skill_executor.execute_batch(
                [_tl_call], user_id=str(user_id), chat_id=str(chat_id),
                execution_id=_request_execution_id,
            )
            _tlr = _tl_results[0] if _tl_results else None
            response_text = _tlr.output if (_tlr and _tlr.success) else "No pude listar las tareas."
            return await self._safe_publish_response(
                text=response_text,
                chat_id=str(chat_id),
                correlation_id=correlation_id,
                skill_results=_tl_results or [],
                user_text=text,
                user_lang=_user_lang,
                outer_trace=_decision_trace,
                reason="task_list",
            )

        # Auto-detect reminder list queries — bypass LLM, call list_reminders directly
        if _short_msg and _detect_reminder_list(text) and self.skill_executor:
            from ..skills.types import SkillCall as _SC
            _rl_call = _SC(
                skill_name="list_reminders",
                arguments={},
                raw_text="[auto-detected: list reminders]",
            )
            _rl_results = await self.skill_executor.execute_batch(
                [_rl_call], user_id=str(user_id), chat_id=str(chat_id),
                execution_id=_request_execution_id,
            )
            _rlr = _rl_results[0] if _rl_results else None
            response_text = _rlr.output if (_rlr and _rlr.success) else "No tengo recordatorios activos."
            return await self._safe_publish_response(
                text=response_text,
                chat_id=str(chat_id),
                correlation_id=correlation_id,
                skill_results=_rl_results or [],
                user_text=text,
                user_lang=_user_lang,
                outer_trace=_decision_trace,
                reason="reminder_list",
            )

        # Auto-detect task UPDATE (must check BEFORE create / delete so
        # "actualiza la tarea X a cada 6h" doesn't fall through to creation
        # of a duplicate). Routes directly to task_manager(action="update").
        if self.skill_executor:
            _upd_name, _upd_interval = _detect_task_update(text)
            if _upd_name and _upd_interval:
                from ..skills.types import SkillCall as _SC
                _upd_call = _SC(
                    skill_name="task_manager",
                    arguments={
                        "action": "update",
                        "name": _upd_name,
                        "interval": _upd_interval,
                    },
                    raw_text=f"[auto-detected: update task '{_upd_name}' to {_upd_interval}]",
                )
                _upd_results = await self.skill_executor.execute_batch(
                    [_upd_call], user_id=str(user_id), chat_id=str(chat_id),
                    execution_id=_request_execution_id,
                )
                _ur = _upd_results[0] if _upd_results else None
                if _ur and _ur.success:
                    response_text = _ur.output
                else:
                    response_text = (
                        f"No pude actualizar la tarea '{_upd_name}': "
                        f"{(_ur.error if _ur else 'error desconocido')[:120]}"
                    )
                return await self._safe_publish_response(
                    text=response_text,
                    chat_id=str(chat_id),
                    correlation_id=correlation_id,
                    skill_results=_upd_results or [],
                    user_text=text,
                    user_lang=_user_lang,
                    outer_trace=_decision_trace,
                    reason="task_update",
                )

        # Auto-detect single task delete by name — bypass LLM to avoid skill confusion
        _del_task_name = _detect_task_delete(text) if (_short_msg and self.skill_executor) else None
        if _del_task_name:
            from ..skills.types import SkillCall as _SC
            _dt_call = _SC(
                skill_name="task_manager",
                arguments={"action": "delete", "name": _del_task_name},
                raw_text=f"[auto-detected: delete task '{_del_task_name}']",
            )
            _dt_results = await self.skill_executor.execute_batch(
                [_dt_call], user_id=str(user_id), chat_id=str(chat_id),
                execution_id=_request_execution_id,
            )
            _dtr = _dt_results[0] if _dt_results else None
            response_text = _dtr.output if (_dtr and _dtr.success) else (
                _dtr.error if (_dtr and _dtr.error) else f"No se encontró la tarea '{_del_task_name}'."
            )
            return await self._safe_publish_response(
                text=response_text,
                chat_id=str(chat_id),
                correlation_id=correlation_id,
                skill_results=_dt_results or [],
                user_text=text,
                user_lang=_user_lang,
                outer_trace=_decision_trace,
                reason="task_delete_one",
            )

        # Auto-detect reminder delete requests — bypass LLM, call delete_reminder directly
        _del_reminder_kw = _detect_reminder_delete(text) if (_short_msg and self.skill_executor) else None
        if _del_reminder_kw:
            from ..skills.types import SkillCall as _SC
            _dr_call = _SC(
                skill_name="delete_reminder",
                arguments={"keyword": _del_reminder_kw},
                raw_text="[auto-detected: delete reminder]",
            )
            _dr_results = await self.skill_executor.execute_batch(
                [_dr_call], user_id=str(user_id), chat_id=str(chat_id),
                execution_id=_request_execution_id,
            )
            _drr = _dr_results[0] if _dr_results else None
            response_text = _drr.output if (_drr and _drr.success) else "No pude eliminar el recordatorio."
            return await self._safe_publish_response(
                text=response_text,
                chat_id=str(chat_id),
                correlation_id=correlation_id,
                skill_results=_dr_results or [],
                user_text=text,
                user_lang=_user_lang,
                outer_trace=_decision_trace,
                reason="reminder_delete",
            )

        # Auto-detect Gmail credentials — configure immediately (no length gate: credentials are always short)
        _gmail_creds = _detect_gmail_configure(text) if self.skill_executor else None
        if _gmail_creds:
            from ..skills.types import SkillCall as _SC
            _gm_addr, _gm_pw = _gmail_creds
            _gm_call = _SC(
                skill_name="gmail",
                arguments={"action": "configure", "address": _gm_addr, "password": _gm_pw},
                raw_text="[auto-detected: gmail configure]",
            )
            _gm_results = await self.skill_executor.execute_batch(
                [_gm_call], user_id=str(user_id), chat_id=str(chat_id),
                execution_id=_request_execution_id,
            )
            _gmr = _gm_results[0] if _gm_results else None
            if _gmr and _gmr.success:
                response_text = f"✅ Gmail conectado correctamente para {_gm_addr}. Ya puedo leer, enviar y buscar tus correos."
            else:
                err = (_gmr.error if _gmr else "") or "error desconocido"
                response_text = f"❌ Could not connect to Gmail: {err}. Verify the email address and app password."
            return await self._safe_publish_response(
                text=response_text,
                chat_id=str(chat_id),
                correlation_id=correlation_id,
                skill_results=_gm_results or [],
                user_text=text,
                user_lang=_user_lang,
                outer_trace=_decision_trace,
                reason="gmail_configure",
            )

        # Auto-detect Gmail inbox queries — call gmail(action="inbox") directly
        if _short_msg and _detect_gmail_inbox(text) and self.skill_executor:
            from ..skills.types import SkillCall as _SC
            _gmi_call = _SC(
                skill_name="gmail",
                arguments={"action": "inbox", "count": "15"},
                raw_text="[auto-detected: gmail inbox]",
            )
            _gmi_results = await self.skill_executor.execute_batch(
                [_gmi_call], user_id=str(user_id), chat_id=str(chat_id),
                execution_id=_request_execution_id,
            )
            _gmir = _gmi_results[0] if _gmi_results else None
            if _gmir and _gmir.success:
                response_text = _gmir.output
            elif _gmir and "no configurado" in (_gmir.error or "").lower():
                response_text = "Gmail is not configured yet. Share your Gmail address and app password so I can connect."
            else:
                response_text = _gmir.error if (_gmir and _gmir.error) else "No pude acceder al correo."
            return await self._safe_publish_response(
                text=response_text,
                chat_id=str(chat_id),
                correlation_id=correlation_id,
                skill_results=_gmi_results or [],
                user_text=text,
                user_lang=_user_lang,
                outer_trace=_decision_trace,
                reason="gmail_inbox",
            )

        # Auto-detect lyrics requests — bypass LLM, search directly
        _lyrics_query = _detect_lyrics_request(text)
        if _lyrics_query and self.skill_executor:
            from ..skills.types import SkillCall as _SC
            _s_call = _SC(skill_name="web_search", arguments={"query": _lyrics_query, "max_results": "5"}, raw_text="[auto: lyrics search]")
            _s_results = await self.skill_executor.execute_batch([_s_call], user_id=str(user_id), chat_id=str(chat_id), execution_id=_request_execution_id)
            _sr = _s_results[0] if _s_results else None
            if _sr and _sr.success and _sr.output:
                # Let LLM format the raw search result into a clean response with the lyrics
                # Don't skip LLM here — inject search results as pre-context and let it format nicely
                text = f"{text}\n\n[Resultados de búsqueda de letra]:\n{_sr.output[:3000]}"

        # Auto-detect YouTube link requests — bypass LLM, search directly
        _yt_query = _detect_youtube_request(text) if not _lyrics_query else None
        if _yt_query and self.skill_executor:
            from ..skills.types import SkillCall as _SC
            _s_call = _SC(skill_name="web_search", arguments={"query": _yt_query, "max_results": "5"}, raw_text="[auto: youtube search]")
            _s_results = await self.skill_executor.execute_batch([_s_call], user_id=str(user_id), chat_id=str(chat_id), execution_id=_request_execution_id)
            _sr = _s_results[0] if _s_results else None
            if _sr and _sr.success and _sr.output:
                text = f"{text}\n\n[Resultados de búsqueda YouTube]:\n{_sr.output[:2000]}"

        # Pre-LLM fast-path: complex multi-step directives go straight to goal engine
        # This avoids a double LLM call (chat response + goal planning) for long task specs
        # NOTE: skip for scheduled task messages and task-SETUP requests (monitoring, reports).
        # Task-setup requests must go through the LLM so it can execute immediately + create task.
        if (
            len(text) > 500
            and self.goal_orchestrator is not None
            and _COMPLEX_DIRECTIVE_RE.search(text)
            and not text.startswith("[TAREA PROGRAMADA:")
            and not _TASK_SETUP_RE.search(text)
        ):
            try:
                goal = await self.goal_orchestrator.create_goal(
                    objective=text,
                    chat_id=str(chat_id),
                    user_id=str(user_id),
                    priority=8,
                    source="telegram",
                )
                logger.info("handler.complex_directive_fast_path", goal_id=goal.id, chat_id=str(chat_id))
                # Immediate user-visible acknowledgment — without this the
                # user sees nothing until the goal completes (or never if it
                # fails). Goal completion / failure notifications are sent
                # by goal_orchestrator separately.
                _ack_text = (
                    "✅ Procesando tu solicitud — te aviso cuando termine."
                    if (_user_lang or "en") == "es"
                    else "✅ Processing your request — I will notify you when done."
                )
                response_text = await self._safe_publish_response(
                    text=_ack_text,
                    chat_id=str(chat_id),
                    correlation_id=correlation_id,
                    skill_results=[],
                    user_text=text,
                    user_lang=_user_lang,
                    outer_trace=_decision_trace,
                    reason="complex_directive_ack",
                )
                # Store episodic memory so follow-up messages have context
                try:
                    async with async_session() as _fpm_session:
                        await self.memory.store_episodic(
                            _fpm_session,
                            event_type="telegram.message",
                            user_input=text,
                            agent_response=response_text,
                            user_id=str(user_id),
                            chat_id=str(chat_id),
                        )
                except Exception:
                    pass
                return response_text
            except Exception as _gex:
                # Fall through to Decision Layer / LLM — never block on fast-path failure
                logger.warning("handler.complex_directive_goal_failed", error=str(_gex))

        # Check if any model is available
        if not self.model_manager.active_model:
            response_text = (
                "No model is currently active.\n"
                "Use /model download <name> to install a model, "
                "or /model available to see options."
            )
            return await self._safe_publish_response(
                text=response_text,
                chat_id=str(chat_id),
                correlation_id=correlation_id,
                skill_results=[],
                user_text=text,
                user_lang=_user_lang,
                outer_trace=_decision_trace,
                reason="no_active_model",
            )

        # Build context from memory (with skill catalog)
        active_model = self.model_manager.active_model
        skill_catalog = ""
        if self.skill_registry:
            skill_catalog = self.skill_registry.format_for_prompt()

        # Build Telegram live-progress callback — publishes telegram.progress events
        # so the bridge can send/edit a status message in real-time
        _progress_lines: list[str] = []

        async def _tg_progress(ev: dict):
            ev_type = ev.get("type", "")
            if ev_type == "thinking":
                r = ev.get("round", 1)
                if isinstance(r, int):
                    _is_es = _user_lang == "es"
                    if r <= 1:
                        line = "Procesando tu solicitud…" if _is_es else "Processing your request…"
                    elif r == 2:
                        line = "Analizando los resultados…" if _is_es else "Analyzing results…"
                    else:
                        line = (f"Completando la tarea (paso {r})…" if _is_es
                                else f"Completing task (step {r})…")
                else:
                    # Caller passed a descriptive string — use directly
                    line = str(r)
            elif ev_type == "skills_planned":
                skills = ev.get("skills", [])
                line = "  📋 " + ", ".join(skills)
            elif ev_type == "skill_done":
                sk = ev.get("skill", "?")
                ms = ev.get("ms", 0)
                ok = ev.get("success", True)
                line = f"  {'✓' if ok else '✗'} {sk} ({ms}ms)"
            else:
                return
            _progress_lines.append(line)
            status_text = "\n".join(_progress_lines[-6:])  # last 6 lines
            try:
                await self.bus.publish(self.stream_outgoing, {
                    "event_type": EventType.TELEGRAM_PROGRESS,
                    "correlation_id": correlation_id,
                    "chat_id": str(chat_id),
                    "text": status_text,
                })
            except Exception:
                pass  # Never block on progress failures

        # Auto-detect and pre-execute skills based on user message
        pre_results_text = ""
        auto_results = []  # always defined; may be populated by auto-detect below
        _action_intent: _ActionIntent = _ActionIntent()  # default: no commitment
        _action_attempts: int = 0
        if self.skill_executor:
            # If user is asking for browser/screenshot/page action but no URL is in the message,
            # inject the last known URL for this chat so the LLM doesn't invent example.com
            _last_url_tg = self._last_browser_url.get(str(chat_id), "")
            _BROWSER_INTENT_RE = re.compile(
                r"\b(?:captur[ae]|screenshot|pantallazo|pantallaz|screenshoot|"
                r"continu[aá]|sigue|m[aá]s\s+(?:abajo|capturas?)|"
                r"siguiente\s+captura|otra\s+captura|ahora\s+(?:abajo|m[aá]s)|"
                r"sigue\s+(?:bajando|scrolleando|desplaz)|"
                r"saca(?:me)?\s+(?:una\s+)?(?:captura|imagen|foto|pantallazo)|"
                r"toma(?:me)?\s+(?:una\s+)?(?:captura|imagen|foto|pantallazo)|"
                r"haz(?:me)?\s+(?:una\s+)?(?:captura|imagen|pantallazo)|"
                r"la\s+(?:p[aá]gina|noticia|sitio|url|misma|este\s+sitio)|"
                r"este\s+(?:sitio|art[ií]culo|p[aá]gina|link|enlace)|"
                r"la\s+misma\s+(?:p[aá]gina|url|direcci[oó]n)|"
                r"env[ií]a(?:me)?\s+(?:la|las|los|esas?|estos?)\s+(?:captura|imagen|foto|screenshot))\b",
                re.IGNORECASE,
            )
            _has_url_in_text = bool(re.search(r"https?://\S+", text))
            # Bare-domain match (e.g. "lasegunda.com", "wikipedia.org") — counts
            # as a URL signal so "captura de wikipedia.org" still works.
            _has_bare_domain = bool(re.search(
                r"\b(?:[a-z0-9-]+\.)+(?:com|net|org|io|co|cl|es|ai|app|dev|me|mx|ar|us|uk|gov|edu)\b",
                text, re.IGNORECASE,
            ))
            _has_any_url = _has_url_in_text or _has_bare_domain
            if _last_url_tg and _BROWSER_INTENT_RE.search(text) and not _has_any_url:
                text = f"{text}\n[URL activa: {_last_url_tg}]"
            # Require a URL for screenshot/browser intent. If user wants browser
            # action but no URL is anywhere (current msg, bare domain, last url),
            # ask for the URL instead of guessing or defaulting to Google.
            elif _BROWSER_INTENT_RE.search(text) and not _has_any_url and not _last_url_tg:
                from ..communication.translator import apick as _pp_url
                # Detect malformed URL token in text — different phrasing
                # helps the user see what went wrong vs "no URL given".
                _malformed_token = bool(re.search(
                    r"\b(?:[a-z]{1,5}t{1,3}p{0,2}s?|h+)://[^\s,]+",
                    text or "", re.IGNORECASE,
                )) or bool(re.search(
                    r"\b[a-z0-9][\w\-]+\.[a-z]{1}\b",  # 1-letter TLD looks malformed
                    text or "", re.IGNORECASE,
                ))
                _ask_url_msg = await _pp_url(
                    "url_malformed" if _malformed_token else "url_required",
                    _user_lang or "en",
                    text[:60],
                    self.model_manager, self.redis_url,
                )
                logger.info(
                    "screenshot.url_required",
                    chat_id=str(chat_id),
                    text_preview=text[:80],
                )
                return await self._safe_publish_response(
                    text=_ask_url_msg,
                    chat_id=str(chat_id),
                    correlation_id=correlation_id,
                    skill_results=[],
                    user_text=text,
                    user_lang=_user_lang,
                    outer_trace=_decision_trace,
                    reason="screenshot_url_required",
                )

            # Phase 1+5+8 (UX): SSRF intent fast-path — when the user explicitly
            # types a URL pointing at an internal/loopback/RFC-1918 address,
            # answer immediately in their language. Going through the LLM here
            # tends to produce English refusals because the safety reasoning
            # defaults to English. Direct phrase keeps tone + language right.
            _SSRF_TARGET_FAST_RE = re.compile(
                r"(?:https?|file|gopher)://"
                r"(?:localhost|127\.0\.0\.1|169\.254\.\d+\.\d+"
                r"|10\.\d{1,3}\.\d{1,3}\.\d{1,3}"
                r"|192\.168\.\d{1,3}\.\d{1,3}"
                r"|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}"
                r"|0\.0\.0\.0|::1|metadata\.google\.internal)"
                r"\S*",
                re.IGNORECASE,
            )
            _ssrf_match = _SSRF_TARGET_FAST_RE.search(text or "")
            if _ssrf_match and (
                _BROWSER_INTENT_RE.search(text) or "GET " in (text or "").upper()[:6]
            ):
                from ..communication.translator import apick as _pp_ssrf
                _target_str = _ssrf_match.group(0)[:80]
                _ssrf_msg = await _pp_ssrf(
                    "internal_address_refused",
                    _user_lang or "en",
                    _target_str,
                    self.model_manager, self.redis_url,
                    target=_target_str,
                )
                logger.warning(
                    "ssrf.fast_path_refused",
                    chat_id=str(chat_id),
                    target=_target_str[:80],
                )
                return await self._safe_publish_response(
                    text=_ssrf_msg,
                    chat_id=str(chat_id),
                    correlation_id=correlation_id,
                    skill_results=[],
                    user_text=text,
                    user_lang=_user_lang,
                    outer_trace=_decision_trace,
                    reason="ssrf_fast_path_refused",
                )

            # B5 fix: email-intent fast-path. When user clearly wants to send
            # an email but no recipient is in the text → ask for the recipient
            # via apick (translated). Avoids the LLM hitting an empty-fallback.
            _EMAIL_INTENT_RE = re.compile(
                r"\b(?:m[aá]nda(?:le)?\s+(?:un\s+)?(?:correo|email|mail)|"
                r"env[ií]a(?:le)?\s+(?:un\s+)?(?:correo|email|mail)|"
                r"send\s+(?:an?\s+|the\s+)?(?:email|mail|message)|"
                r"shoot\s+(?:an?\s+|the\s+)?email|"
                r"(?:write|escribe).*(?:correo|email))\b",
                re.IGNORECASE,
            )
            _EMAIL_ADDR_RE = re.compile(r"\b[\w.+\-]+@[\w\-]+\.[\w.\-]+\b")
            if (
                not _is_scheduled_trigger
                and _EMAIL_INTENT_RE.search(text or "")
                and not _EMAIL_ADDR_RE.search(text or "")
            ):
                from ..communication.translator import apick as _apick_em
                _ask_recip = await _apick_em(
                    "email_recipient_required",
                    _user_lang or "en",
                    (correlation_id or "") + ":em",
                    self.model_manager, self.redis_url,
                )
                logger.info(
                    "email.recipient_required_fast_path",
                    chat_id=str(chat_id),
                    text_preview=(text or "")[:80],
                )
                return await self._safe_publish_response(
                    text=_ask_recip,
                    chat_id=str(chat_id),
                    correlation_id=correlation_id,
                    skill_results=[],
                    user_text=text,
                    user_lang=_user_lang,
                    outer_trace=_decision_trace,
                    reason="email_recipient_required",
                )

            # Skip auto-detect for: (a) scheduled task triggers — full instruction drives LLM
            # directly; (b) scheduling requests — decision layer must handle these, not
            # auto-detect, to prevent web_search from firing on long scheduling messages.
            # Planning mode: user explicitly asked for explanation only — no execution
            _planning_mode = (
                not _is_scheduled_trigger
                and bool(_PLANNING_MODE_RE.search(text))
            )
            if _planning_mode:
                logger.info("planning_mode.detected", chat_id=str(chat_id), text_preview=text[:80])

            # ── Action Intent Classification (Phase 1) ────────────────────────
            _action_intent: _ActionIntent = classify_action_intent(
                text, planning_mode=_planning_mode
            )
            if _action_intent.action_commitment:
                logger.info(
                    "action_intent.committed",
                    action_type=_action_intent.action_type,
                    target=(_action_intent.action_target or "")[:80],
                    confidence=round(_action_intent.confidence, 2),
                    is_retry=_action_intent.is_retry_signal,
                )
                # B1 fix: when the user provides an explicit URL+browser intent,
                # save its domain as last_confirmed_domain. Subsequent follow-up
                # turns ("haz lo mismo / sácale más capturas") will be locked
                # to this domain by pre_execution_check.
                if (
                    _action_intent.action_target
                    and _action_intent.action_target.lower().startswith(("http://", "https://"))
                    and _action_intent.action_type.startswith(("browser_", "screenshot_"))
                    and self.redis_url and chat_id
                ):
                    try:
                        from urllib.parse import urlparse as _urlp_lcd
                        _dom_lcd = (_urlp_lcd(_action_intent.action_target).hostname or "").lower()
                        if _dom_lcd.startswith("www."):
                            _dom_lcd = _dom_lcd[4:]
                        if _dom_lcd:
                            _r_set = aioredis.from_url(self.redis_url, decode_responses=True)
                            try:
                                await _r_set.set(
                                    f"last_confirmed_domain:{chat_id}",
                                    _dom_lcd,
                                    ex=3600,  # 1h
                                )
                                logger.info(
                                    "last_confirmed_domain.set",
                                    chat_id=str(chat_id),
                                    domain=_dom_lcd,
                                )
                            finally:
                                await _r_set.aclose()
                    except Exception:
                        pass
            # ── End Action Intent Classification ──────────────────────────────

            # ── Execution Plan Generation (System Execution Planner) ──────────
            # Generates a deterministic step sequence before any LLM calls.
            # Known workflows (package tracking, forms) get concrete plans.
            # The LLM follows the plan; system validates outcomes.
            _exec_plan: _ExecutionPlan | None = generate_plan(_action_intent)
            _plan_executor: _PlanExecutor | None = (
                _PlanExecutor(_exec_plan, self.skill_executor)
                if (_exec_plan is not None and self.skill_executor)
                else None
            )
            if _exec_plan is not None:
                logger.info(
                    "execution_planner.plan_generated",
                    plan_id=_exec_plan.plan_id,
                    action_type=_exec_plan.action_type,
                    steps=len(_exec_plan.steps),
                    confidence=round(_exec_plan.confidence, 2),
                    required_steps=_exec_plan.required_step_ids,
                )
            # ── End Execution Plan Generation ────────────────────────────────

            _pre_is_scheduling = (not _is_scheduled_trigger) and is_scheduling_request(text)
            auto_calls = [] if (_is_scheduled_trigger or _pre_is_scheduling or _planning_mode) else detect_skills(text)
            if _pre_is_scheduling and auto_calls:
                # Safety: discard any auto-detected skills for scheduling requests
                auto_calls = []
            # ── Intent boundary gate (auto-detect path) ───────────────────
            # Same gate as the LLM loop: side-effecting skills (gmail/agent/task)
            # without explicit intent in the user's current message are dropped
            # before pre-execution. Catches cases where auto_detect hallucinates
            # a recipient (e.g. from few-shot examples in the system prompt).
            if auto_calls:
                _auto_kept, _auto_dropped = _filter_inferred_side_effects(
                    auto_calls, text, ctx_messages=None, chat_id=str(chat_id),
                )
                if _auto_dropped:
                    for _ad_sc, _ad_reason, _ad_intent in _auto_dropped:
                        logger.warning(
                            "intent_gate.blocked",
                            skill=_ad_sc.skill_name,
                            action=str((_ad_sc.arguments or {}).get("action", "")
                                       if isinstance(_ad_sc.arguments, dict) else ""),
                            reason=_ad_reason,
                            intent=_ad_intent,
                            path="auto_detect",
                            text_preview=(text or "")[:80],
                        )
                    auto_calls = _auto_kept
            if auto_calls:
                logger.info(
                    "skills.auto_detected",
                    count=len(auto_calls),
                    skills=[c.skill_name for c in auto_calls],
                )
                await _tg_progress({
                    "type": "skills_planned",
                    "skills": [c.skill_name for c in auto_calls],
                    "round": 0,
                })

                # ── C2 Fix: Pre-execution guard for auto-detect path ──────────
                # ALL externally-interacting auto-detect calls MUST pass the same
                # guard as the main LLM loop.  The original fix was browser-only;
                # CRIT-2 extends it to web_search / http_request / fetch_url so
                # that lyrics / YouTube / URL-fetch auto-detect routes are also
                # protected by domain lock, hard-block, and action validation.
                _EXTERNAL_GUARD_SKILLS = frozenset({
                    "browser", "web_search", "http_request", "fetch_url",
                })
                _auto_guard_ok = True
                _external_auto_calls = [
                    c for c in auto_calls if c.skill_name in _EXTERNAL_GUARD_SKILLS
                ]
                if _external_auto_calls:
                    # Multi-URL exemption: when ALL external calls have URLs
                    # that appear verbatim in the user's text, the user
                    # explicitly authorized them — skip the domain-drift
                    # guard, which would otherwise block multi-URL screenshot
                    # requests like "captura X y Y" (different domains by
                    # definition). Single-URL auto-detect still runs the
                    # guard normally.
                    _all_urls_explicit = (
                        len(_external_auto_calls) >= 2
                        and all(
                            (((c.arguments or {}).get("url") or "")
                                .replace("https://", "")
                                .replace("http://", "")
                                .replace("www.", "")
                                .split("/")[0]
                                .lower()
                                in text.lower())
                            for c in _external_auto_calls
                        )
                    )
                    if _all_urls_explicit:
                        logger.info(
                            "auto_detect.multi_url_exempt",
                            path="telegram",
                            count=len(_external_auto_calls),
                            urls=[((c.arguments or {}).get("url") or "")[:60] for c in _external_auto_calls],
                        )
                    try:
                        from .control_layer import (
                            pre_execution_check as _prex_fn,
                            infer_intent_from_text as _infer_fn,
                        )
                        _auto_intent = _action_intent
                        if not getattr(_auto_intent, "action_commitment", False):
                            _auto_intent = _infer_fn(text, _external_auto_calls)
                        # B1 fix: pass last_confirmed_domain.
                        _auto_last_dom = ""
                        if self.redis_url and chat_id:
                            try:
                                _r_a = aioredis.from_url(self.redis_url, decode_responses=True)
                                try:
                                    _auto_last_dom = (await _r_a.get(
                                        f"last_confirmed_domain:{chat_id}"
                                    )) or ""
                                finally:
                                    await _r_a.aclose()
                            except Exception:
                                _auto_last_dom = ""
                        _auto_prex = _prex_fn(
                            _external_auto_calls,
                            _auto_intent,
                            user_text=text,
                            domain_lock_override=None,
                            last_confirmed_domain=_auto_last_dom,
                        )
                        if _auto_prex.blocked and not _all_urls_explicit:
                            _first_args = (_external_auto_calls[0].arguments or {})
                            logger.warning(
                                "auto_detect.guard_blocked",
                                path="telegram",
                                reason=_auto_prex.violation_type,
                                url=_first_args.get("url", "")[:80],
                                query=_first_args.get("query", "")[:80],
                            )
                            # Discard blocked auto_detect results — fall through to LLM loop.
                            # The LLM has domain context from action_intent and will generate
                            # the correct tool call (e.g. browser instead of web_search).
                            # Never expose raw internal block messages to the user.
                            auto_calls = []
                            _external_auto_calls = []
                    except Exception as _auto_guard_err:
                        logger.warning(
                            "auto_detect.guard_error",
                            path="telegram",
                            error=str(_auto_guard_err)[:80],
                        )
                        # Fail-closed: guard exception blocks execution
                        _auto_guard_ok = False
                        _block_msg = "No se pudo verificar la seguridad de la operación solicitada."
                        return await self._safe_publish_response(
                            text=_block_msg,
                            chat_id=str(chat_id),
                            correlation_id=correlation_id,
                            skill_results=[],
                            user_text=text,
                            user_lang=_user_lang,
                            outer_trace=_decision_trace,
                            reason="auto_guard_fail_closed",
                        )
                # ── End C2 Pre-execution guard ────────────────────────────────

                if not auto_calls:
                    # Guard cleared auto_calls (domain lock block) — skip execution,
                    # fall through to LLM loop with empty auto_results.
                    auto_results = []
                else:
                    auto_results = await self.skill_executor.execute_batch(
                        auto_calls, user_id=str(user_id), chat_id=str(chat_id),
                        execution_id=_request_execution_id,
                    )
                for _ar in auto_results:
                    await _tg_progress({
                        "type": "skill_done",
                        "skill": _ar.skill_name,
                        "success": _ar.success,
                        "ms": _ar.execution_ms,
                    })

                # For reminders, monitors, and listings: respond directly, skip LLM
                _DIRECT_SKILLS = (
                    "create_reminder", "create_monitor",
                    "list_monitors", "list_reminders",
                )
                _ac0 = auto_calls[0] if auto_calls else None
                # wipe_all: respond directly, skip LLM (action already complete)
                if (
                    _ac0 is not None
                    and _ac0.skill_name == "agent_manager"
                    and _ac0.arguments.get("action") == "wipe_all"
                ):
                    _wr = auto_results[0]
                    _wipe_resp = _wr.output if _wr.success else f"Error al limpiar: {_wr.error}"
                    return await self._safe_publish_response(
                        text=_wipe_resp,
                        chat_id=str(chat_id),
                        correlation_id=correlation_id,
                        skill_results=auto_results or [],
                        user_text=text,
                        user_lang=_user_lang,
                        outer_trace=_decision_trace,
                        reason="agent_wipe_all",
                    )
                if _ac0 is not None and _ac0.skill_name in _DIRECT_SKILLS:
                    r = auto_results[0]
                    response_text = r.output if r.success else f"Error: {r.error}"
                    return await self._safe_publish_response(
                        text=response_text,
                        chat_id=str(chat_id),
                        correlation_id=correlation_id,
                        skill_results=auto_results or [],
                        user_text=text,
                        user_lang=_user_lang,
                        outer_trace=_decision_trace,
                        reason="direct_skill_response",
                    )

                # ── Multi-URL aggregator ─────────────────────────────────
                # When auto-detect produced 2+ browser captures (one per
                # URL the user named), build a deterministic per-URL
                # outcome list and ship it directly. Each URL must appear
                # in the response — without this, the LLM occasionally
                # summarized only the first URL it saw.
                _browser_calls = [c for c in auto_calls if c.skill_name == "browser"]
                if len(_browser_calls) >= 2:
                    _shot_re_multi = re.compile(r"(/data/screenshots/screenshot_\d+\.png)")
                    _is_es = (_user_lang or "en") == "es"
                    _summaries: list[str] = []
                    _photo_jobs: list[tuple[str, str]] = []  # (path, caption)
                    for _bc, _br in zip(auto_calls, auto_results):
                        if _bc.skill_name != "browser":
                            continue
                        _u = (_bc.arguments or {}).get("url", "") or "(sin URL)"
                        if not getattr(_br, "success", False):
                            _err_first = (getattr(_br, "error", "") or "").splitlines()[0][:80]
                            _line = (
                                f"• {_u} → ❌ {_err_first or 'falló'}"
                                if _is_es
                                else f"• {_u} → ❌ {_err_first or 'failed'}"
                            )
                            _summaries.append(_line)
                            continue
                        _out = getattr(_br, "output", "") or ""
                        # Error string in output (e.g. SSRF block returns
                        # "Error: URL blocked...") — treat as failure even
                        # though success=True. The success flag refers to
                        # the skill not crashing, not to the URL working.
                        if _out.lstrip().lower().startswith("error:"):
                            _err_line = _out.splitlines()[0][:120]
                            _summaries.append(
                                f"• {_u} → ❌ {_err_line}"
                                if _is_es
                                else f"• {_u} → ❌ {_err_line}"
                            )
                        elif "[CAPTURE_VALID: false]" in _out:
                            _summaries.append(
                                f"• {_u} → 🚫 bloqueado (login/captcha)"
                                if _is_es
                                else f"• {_u} → 🚫 blocked (login/captcha)"
                            )
                        elif "[CAPTURE_VALID: true]" in _out:
                            _summaries.append(
                                f"• {_u} → ✅ captura enviada"
                                if _is_es
                                else f"• {_u} → ✅ screenshot sent"
                            )
                        else:
                            _summaries.append(
                                f"• {_u} → ✅ navegado"
                                if _is_es
                                else f"• {_u} → ✅ navigated"
                            )
                        # Collect photos for sending (best-effort)
                        for _sm in _shot_re_multi.finditer(_out):
                            _path = _sm.group(1)
                            if os.path.exists(_path) and "[CAPTURE_VALID: false]" not in _out:
                                _photo_jobs.append((_path, _u))

                    # Ship photos first, then the aggregated text
                    for _path, _cap in _photo_jobs[:10]:
                        try:
                            await self.bus.publish(self.stream_outgoing, {
                                "event_type": EventType.TELEGRAM_RESPONSE,
                                "correlation_id": str(uuid4()),
                                "chat_id": str(chat_id),
                                "photo_path": _path,
                                "text": _cap,
                            })
                        except Exception:
                            pass
                    _header = "Resultados:" if _is_es else "Results:"
                    _agg_text = _header + "\n" + "\n".join(_summaries)
                    return await self._safe_publish_response(
                        text=_agg_text,
                        chat_id=str(chat_id),
                        correlation_id=correlation_id,
                        skill_results=auto_results or [],
                        user_text=text,
                        user_lang=_user_lang,
                        outer_trace=_decision_trace,
                        reason="multi_url_aggregation",
                    )

                # For browser auto-detect: analyze results and decide response strategy
                if _ac0 is not None and _ac0.skill_name == "browser":
                    nav_result = next((r for r in auto_results if r.success and "Navigated to" in (r.output or "")), None)
                    # Detect ALL screenshot paths (handles both capture and scroll_capture)
                    _shot_re_ad = re.compile(r"(/data/screenshots/screenshot_\d+\.png)")
                    _auto_shot_path = None
                    _all_auto_shot_paths: list[str] = []
                    for _ar in auto_results:
                        if _ar.success:
                            for _sm in _shot_re_ad.finditer(_ar.output or ""):
                                if os.path.exists(_sm.group(1)) and _sm.group(1) not in _all_auto_shot_paths:
                                    _all_auto_shot_paths.append(_sm.group(1))
                    if _all_auto_shot_paths:
                        _auto_shot_path = _all_auto_shot_paths[0]
                    has_screenshot = bool(_all_auto_shot_paths)
                    any_error = next((r for r in auto_results if not r.success), None)
                    # Remember the browser URL for "continúa hacia abajo" follow-ups
                    _req_url = _ac0.arguments.get("url", "")
                    if _req_url:
                        self._last_browser_url[str(chat_id)] = _req_url

                    # Detect anti-bot failure or empty content
                    browser_blocked = False
                    browser_empty = False
                    if nav_result:
                        output_lower = nav_result.output.lower()
                        if "anti-bot" in output_lower or "data:," in nav_result.output or "could not load" in output_lower:
                            browser_blocked = True
                        else:
                            # Check if page has meaningful content after "---" separator
                            parts = nav_result.output.split("---", 1)
                            body_text = parts[1].strip() if len(parts) > 1 else ""
                            if len(body_text) < 50:
                                browser_empty = True
                                logger.info("browser.empty_content", url=_ac0.arguments.get("url", ""), body_len=len(body_text))

                    # If browser was blocked or empty, fall back to fetch_url
                    # EXCEPTION: if the request was specifically a screenshot (capture/scroll_capture),
                    # do NOT fall back to text content — return a clear error instead.
                    _was_capture = _ac0.arguments.get("action") in ("capture", "scroll_capture")
                    # For scroll_capture: nav_result is None (different output format) but screenshots may still exist
                    _is_scroll_capture = _ac0.arguments.get("action") == "scroll_capture"
                    if browser_blocked or browser_empty or (any_error and not nav_result and not _is_scroll_capture):
                        fallback_url = _ac0.arguments.get("url", "")
                        if _was_capture:
                            # Screenshot failed — tell user directly with actual error, never lie about anti-bot
                            _actual_err = any_error.error if any_error else ""
                            # Show only first line of error (hide Selenium stacktraces)
                            _err_first_line = (_actual_err or "").splitlines()[0][:100] if _actual_err else ""
                            if "anti-bot" in _actual_err.lower() or "data:," in _actual_err:
                                _reason = "El sitio tiene protección anti-bot que bloquea navegadores automáticos."
                            elif "ERR_CONNECTION_RESET" in _actual_err or "ERR_NAME_NOT_RESOLVED" in _actual_err:
                                _reason = "No se pudo conectar al sitio (error de red)."
                            elif "timed out" in _actual_err.lower() or "timeout" in _actual_err.lower():
                                _reason = "El sitio tardó demasiado en responder."
                            elif _err_first_line:
                                _reason = f"Error: {_err_first_line}"
                            else:
                                _reason = "El sitio no respondió."
                            # Suggest digital edition for lasegunda.com failures
                            from datetime import date as _date
                            _alt_hint = ""
                            if "lasegunda.com" in fallback_url and "digital" not in fallback_url:
                                _today = _date.today()
                                _digital_url = f"https://digital.lasegunda.com/{_today.strftime('%Y/%m/%d')}/A"
                                _alt_hint = f"\n💡 La edición digital sí funciona: {_digital_url}"
                            response_text = (
                                f"No pude tomar la captura de {fallback_url}.\n"
                                f"{_reason}{_alt_hint}"
                            )
                            return await self._safe_publish_response(
                                text=response_text,
                                chat_id=str(chat_id),
                                correlation_id=correlation_id,
                                skill_results=auto_results or [],
                                user_text=text,
                                user_lang=_user_lang,
                                outer_trace=_decision_trace,
                                reason="screenshot_capture_failed",
                            )
                        elif fallback_url and self.skill_executor:
                            logger.info("browser.fallback_to_fetch", url=fallback_url, reason="blocked" if browser_blocked else "empty" if browser_empty else "error")
                            from ..skills.types import SkillCall as SC
                            fetch_call = SC(
                                skill_name="fetch_url",
                                arguments={"url": fallback_url, "max_chars": "6000"},
                                raw_text=f"[fallback: fetch {fallback_url}]",
                            )
                            fetch_results = await self.skill_executor.execute_batch(
                                [fetch_call], user_id=str(user_id), chat_id=str(chat_id),
                                execution_id=_request_execution_id,
                            )
                            if fetch_results and fetch_results[0].success:
                                pre_results_text = fetch_results[0].output
                                # Fall through to LLM processing below
                            else:
                                error_msg = fetch_results[0].error if fetch_results else "fetch failed"
                                response_text = f"No pude acceder a {fallback_url}: {error_msg}"
                                return await self._safe_publish_response(
                                    text=response_text,
                                    chat_id=str(chat_id),
                                    correlation_id=correlation_id,
                                    skill_results=fetch_results or [],
                                    user_text=text,
                                    user_lang=_user_lang,
                                    outer_trace=_decision_trace,
                                    reason="browser_fetch_failed",
                                )
                        else:
                            response_text = f"Error opening the page: {any_error.error if any_error else 'anti-bot protection'}"
                            return await self._safe_publish_response(
                                text=response_text,
                                chat_id=str(chat_id),
                                correlation_id=correlation_id,
                                skill_results=auto_results or [],
                                user_text=text,
                                user_lang=_user_lang,
                                outer_trace=_decision_trace,
                                reason="page_open_error",
                            )

                    # Browser succeeded with content
                    elif has_screenshot and _auto_shot_path:
                        # Screenshot(s) request — apply CAPTURE_VALID gate before sending
                        url = ""
                        title = ""
                        if nav_result:
                            for line in nav_result.output.splitlines():
                                if line.startswith("Navigated to:"):
                                    url = line.split(":", 1)[1].strip()
                                elif line.startswith("Title:"):
                                    title = line.split(":", 1)[1].strip()
                        else:
                            # For scroll_capture, nav info is in the result output
                            for _ar in auto_results:
                                if _ar.success:
                                    for line in (_ar.output or "").splitlines():
                                        if line.startswith("Page:"):
                                            url = line.split(":", 1)[1].strip()

                        # Determine which shots are invalid (CAPTURE_VALID: false)
                        _ad_invalid_shots: set = set()
                        for _ar in auto_results:
                            if _ar.success and "[CAPTURE_VALID: false]" in (_ar.output or ""):
                                for _sm in _shot_re_ad.finditer(_ar.output or ""):
                                    _ad_invalid_shots.add(_sm.group(1))

                        _ad_invalid_warning_sent = False
                        # Filter to existing + valid photos so media-group counts are correct
                        _valid_paths_ad = [
                            p for p in _all_auto_shot_paths
                            if os.path.exists(p) and p not in _ad_invalid_shots
                        ]
                        _invalid_paths_ad = [
                            p for p in _all_auto_shot_paths
                            if os.path.exists(p) and p in _ad_invalid_shots
                        ]
                        total_valid = len(_valid_paths_ad)
                        # Telegram album id (only used when 2+ valid photos)
                        _mg_id_ad = str(uuid4()) if total_valid >= 2 else ""
                        for _idx, _shot_p in enumerate(_valid_paths_ad):
                            if total_valid > 1:
                                cap = f"Captura {_idx+1}/{total_valid} — {title or url}"
                            else:
                                cap = f"{title}\n{url}".strip() if title else url
                            payload = {
                                "event_type": EventType.TELEGRAM_RESPONSE,
                                "correlation_id": correlation_id,
                                "chat_id": str(chat_id),
                                "photo_path": _shot_p,
                                "text": cap,
                            }
                            if _mg_id_ad:
                                payload["media_group_id"] = _mg_id_ad
                                payload["media_group_index"] = str(_idx)
                                payload["media_group_total"] = str(total_valid)
                            await self.bus.publish(self.stream_outgoing, payload)
                        # Send AT MOST one invalid-capture warning + screenshot.
                        # Both texts go through apick → translated to user_lang.
                        if _invalid_paths_ad:
                            from ..communication.translator import apick as _apick_inv
                            _warn_text = await _apick_inv(
                                "invalid_capture_warning", _user_lang or "en",
                                (correlation_id or "") + ":invwarn",
                                self.model_manager, self.redis_url,
                            )
                            await self._safe_publish_response(
                                text=_warn_text,
                                chat_id=str(chat_id),
                                correlation_id=correlation_id,
                                skill_results=[],
                                user_text=text,
                                user_lang=_user_lang,
                                outer_trace=_decision_trace,
                                reason="invalid_capture_warning",
                            )
                            _ad_invalid_warning_sent = True
                            _cap_text = await _apick_inv(
                                "invalid_capture_caption", _user_lang or "en",
                                (correlation_id or "") + ":invcap",
                                self.model_manager, self.redis_url,
                            )
                            await self.bus.publish(self.stream_outgoing, {
                                "event_type": EventType.TELEGRAM_RESPONSE,
                                "correlation_id": correlation_id,
                                "chat_id": str(chat_id),
                                "photo_path": _invalid_paths_ad[0],
                                "text": _cap_text,
                            })
                        total = total_valid + (1 if _invalid_paths_ad else 0)
                        return f"Captura{'s' if total > 1 else ''} enviada{'s' if total > 1 else ''}: {url}"

                    else:
                        # Browser succeeded with content — feed through LLM
                        # so the agent can analyze and answer the user's actual question
                        if nav_result:
                            pre_results_text = nav_result.output
                            # Fall through to LLM processing below
                        else:
                            pre_results_text = ""

                # Only build pre_results_text from auto_results if browser path didn't set it
                if not pre_results_text:
                    lines = []
                    for result in auto_results:
                        if result.success:
                            lines.append(f"{result.output}")
                        else:
                            lines.append(f"[{result.skill_name} error: {result.error}]")
                    pre_results_text = "\n".join(lines)

        # ── Active Flow Context Lock + Hard Context Reset ─────────────────────────
        # If there is a recently-failed workflow for this chat, treat the current
        # user message as a follow-up/recovery instruction for THAT flow.
        # Prevents domain drift (e.g. crypto failure → weather hallucination).
        # EXTENSION: Hard Context Reset fires when the user clearly changes intent
        # (different domain, no recovery signals) — clears all flow state and injects
        # a [CONTEXT RESET] block so the LLM treats the new message as a fresh start.
        _active_flow: dict | None = None
        _context_reset_block: str = ""  # injected into system prompt on intent switch
        _context_reset_info: dict | None = None  # {"old_domain": str, "new_domain": str}
        if self.redis_url and not _is_scheduled_trigger:
            try:
                from .flow_state import (
                    load_active_flow as _load_flow,
                    clear_active_flow as _clear_flow,
                    is_explicit_domain_switch as _is_domain_switch,
                )
                from .context_reset import (
                    is_intent_switch as _is_intent_switch,
                    detect_message_domain as _detect_msg_domain,
                    perform_hard_reset as _hard_reset,
                    build_context_reset_block as _build_reset_block,
                )
                _active_flow = await _load_flow(self.redis_url, str(chat_id))
                if _active_flow:
                    _old_domain = _active_flow.get("domain", "unknown")
                    if _is_domain_switch(text, _active_flow):
                        # Explicit cancel ("olvida eso") + unrelated domain → clear
                        await _clear_flow(self.redis_url, str(chat_id))
                        _active_flow = None
                        logger.info("active_flow.cleared_by_domain_switch",
                                    chat_id=str(chat_id))
                    elif _is_intent_switch(text, _active_flow):
                        # Hard intent switch: clear flow AND inject reset block
                        _new_domain = _detect_msg_domain(text) or "general"
                        await _hard_reset(self.redis_url, str(chat_id),
                                          _old_domain, _new_domain)
                        _context_reset_block = _build_reset_block(_old_domain, _new_domain)
                        _context_reset_info = {"old_domain": _old_domain, "new_domain": _new_domain}
                        _active_flow = None
                        logger.info("context_reset.intent_switch_detected",
                                    chat_id=str(chat_id),
                                    old_domain=_old_domain,
                                    new_domain=_new_domain)
                    else:
                        logger.info("active_flow.locked",
                                    chat_id=str(chat_id),
                                    domain=_active_flow.get("domain"),
                                    flow_type=_active_flow.get("flow_type"))
            except Exception:
                _active_flow = None

        # ── Decision Layer ────────────────────────────────────────────────────────
        # Classify the request BEFORE sending to LLM or planner.
        # Only activates when no auto-detected skills intercepted the request.
        # Routing:
        #   SCHEDULED_TASK → call task_manager(action="create") directly (no LLM)
        #   SUB_AGENT      → call agent_manager(action="create") directly (no LLM)
        #   SCRIPT         → route to GoalOrchestrator (same as GOAL)
        #   GOAL           → route directly to GoalOrchestrator, skip LLM round-trip
        #   DIRECT_RESPONSE → proceed normally (no change)
        _dl_strategy = Strategy.DIRECT_RESPONSE
        # Decision Layer runs whenever there are no auto-detected skills.
        # NOTE: goal_orchestrator is NOT required here — SCHEDULED_TASK and
        # SUB_AGENT can be handled by skill_executor alone.  GOAL/SCRIPT fall
        # through to the LLM if goal_orchestrator is unavailable.
        if not auto_calls and self.skill_executor and not _planning_mode:
            try:
                _dl_strategy = decide_execution_strategy(text)
                logger.info(
                    "decision_layer.classified",
                    strategy=_dl_strategy.value,
                    text_len=len(text),
                    text_preview=text[:80],
                )

                if _dl_strategy == Strategy.SCHEDULED_TASK:
                    if self.governor:
                        _gov_ok, _gov_msg = await self.governor.check_allow("create_task", user_id=str(user_id))
                        if not _gov_ok:
                            return await self._safe_publish_response(
                                text=_gov_msg,
                                chat_id=str(chat_id),
                                correlation_id=correlation_id,
                                skill_results=[],
                                user_text=text,
                                user_lang=_user_lang,
                                outer_trace=_decision_trace,
                                reason="governor_block_create_task",
                            )
                    # Immediate acknowledgment — user sees response right away
                    await _tg_progress({"type": "thinking", "round": "Preparando tarea programada…"})
                    from ..decision_layer import extract_task_params
                    from ..skills.types import SkillCall as _SC
                    _tp = extract_task_params(text)
                    # FIX 10 — Verbatim instruction: override LLM-summarized
                    # instruction with the user's original message text. The
                    # task name from the decision layer is kept (already short).
                    _tp["instruction"] = text
                    _task_call = _SC(
                        skill_name="task_manager",
                        arguments={"action": "create", **_tp},
                        raw_text=f"[decision_layer: schedule '{_tp['name']}']",
                    )
                    _task_results = await self.skill_executor.execute_batch(
                        [_task_call], user_id=str(user_id), chat_id=str(chat_id),
                        execution_id=_request_execution_id,
                    )
                    _tr = _task_results[0] if _task_results else None

                    # FIX 5 — Test-run trigger: if the user explicitly asked for
                    # a test execution, immediately trigger the task once.
                    _test_run_re = re.compile(
                        r"\b(?:haz\s+una\s+(?:ejecuci[oó]n\s+de\s+)?prueba|"
                        r"ejecu(?:c|t)i?(?:[óo]n)?\s+de\s+prueba|"
                        r"prueba\s+ahora|ejec[úu]ta(?:lo|la)?\s+(?:ahora|una\s+vez)|"
                        r"test\s+run|run\s+(?:it|this)\s+(?:once|now)|"
                        r"trigger\s+(?:it|this)\s+now)\b",
                        re.IGNORECASE,
                    )
                    _trigger_output = ""
                    if _tr and _tr.success and _test_run_re.search(text):
                        try:
                            _trigger_call = _SC(
                                skill_name="task_manager",
                                arguments={"action": "trigger", "name": _tp.get("name", "")},
                                raw_text=f"[decision_layer: test-run '{_tp.get('name','')}']",
                            )
                            _trigger_results = await self.skill_executor.execute_batch(
                                [_trigger_call], user_id=str(user_id), chat_id=str(chat_id),
                                execution_id=_request_execution_id,
                            )
                            _ttr = _trigger_results[0] if _trigger_results else None
                            if _ttr and _ttr.success:
                                _trigger_output = (_ttr.output or "")[:300]
                                logger.info(
                                    "test_run.triggered",
                                    task=_tp.get("name", ""),
                                    chat_id=str(chat_id),
                                )
                            else:
                                _trigger_output = (
                                    f"(No se pudo correr la ejecución de prueba: "
                                    f"{(_ttr.error or 'error desconocido')[:120]})"
                                ) if _ttr else "(No se pudo correr la ejecución de prueba)"
                        except Exception as _trig_exc:
                            logger.warning("test_run.trigger_failed", error=str(_trig_exc)[:120])
                            _trigger_output = "(Falló la ejecución de prueba — el task quedó programado igual.)"

                    # Communication layer: natural language confirmation
                    response_text = await self._get_formatter().format_task_created(
                        user_request=text,
                        task_params=_tp,
                        success=bool(_tr and _tr.success),
                        raw_output=(_tr.output or _tr.error or "") if _tr else "",
                        user_lang=_user_lang or "en",
                    )
                    if _trigger_output:
                        response_text = response_text.rstrip() + f"\n\nEjecución de prueba: {_trigger_output}"
                    # Route through central policy: schedule honesty disclaimer
                    # attaches if user named a clock time, action announcer
                    # appends "Acciones: Task scheduled: X" from real result.
                    response_text = await self._safe_publish_response(
                        text=response_text,
                        chat_id=str(chat_id),
                        correlation_id=correlation_id,
                        skill_results=[_tr] if _tr else [],
                        user_text=text,
                        user_lang=_user_lang,
                        outer_trace=_decision_trace,
                        reason="fast_path_task_create",
                    )
                    return response_text

                elif _dl_strategy == Strategy.SUB_AGENT:
                    # FIX 6 — Block sub-agent creation unless the user EXPLICITLY
                    # said "agente" / "agent" / "sub-agent" / "dedicated agent".
                    # Falls through to LLM if not — typically the LLM correctly
                    # uses task_manager only.
                    _agent_intent_re = _INTENT_GATE_PATTERNS.get("agent_manager")
                    if _agent_intent_re and not _agent_intent_re.search(text):
                        logger.warning(
                            "intent_gate.blocked",
                            skill="agent_manager", action="create",
                            reason="no_explicit_intent_in_user_message",
                            intent="inferred_blocked",
                            path="decision_layer_sub_agent",
                            text_preview=(text or "")[:80],
                        )
                        # Don't return; fall through to the LLM loop below.
                        _dl_strategy = Strategy.DIRECT_RESPONSE
                if _dl_strategy == Strategy.SUB_AGENT:
                    if self.governor:
                        _gov_ok, _gov_msg = await self.governor.check_allow("create_agent", user_id=str(user_id))
                        if not _gov_ok:
                            return await self._safe_publish_response(
                                text=_gov_msg,
                                chat_id=str(chat_id),
                                correlation_id=correlation_id,
                                skill_results=[],
                                user_text=text,
                                user_lang=_user_lang,
                                outer_trace=_decision_trace,
                                reason="governor_block_create_agent",
                            )
                    await _tg_progress({"type": "thinking", "round": "Iniciando nuevo agente…"})
                    from ..decision_layer import extract_agent_params
                    from ..skills.types import SkillCall as _SC
                    _ap = extract_agent_params(text)
                    _agent_call = _SC(
                        skill_name="agent_manager",
                        arguments={"action": "create", **_ap},
                        raw_text=f"[decision_layer: create agent '{_ap['name']}']",
                    )
                    _agent_results = await self.skill_executor.execute_batch(
                        [_agent_call], user_id=str(user_id), chat_id=str(chat_id),
                        execution_id=_request_execution_id,
                    )
                    _ar = _agent_results[0] if _agent_results else None
                    # Communication layer: natural language confirmation
                    response_text = await self._get_formatter().format_agent_created(
                        user_request=text,
                        agent_params=_ap,
                        success=bool(_ar and _ar.success),
                        raw_output=(_ar.output or _ar.error or "") if _ar else "",
                    )
                    return await self._safe_publish_response(
                        text=response_text,
                        chat_id=str(chat_id),
                        correlation_id=correlation_id,
                        skill_results=_agent_results or [],
                        user_text=text,
                        user_lang=_user_lang,
                        outer_trace=_decision_trace,
                        reason="fast_path_agent_create",
                    )

                elif _dl_strategy in (Strategy.GOAL, Strategy.SCRIPT):
                    if not self.goal_orchestrator:
                        # No orchestrator available — fall through to LLM
                        _dl_strategy = Strategy.DIRECT_RESPONSE
                    else:
                        if self.governor:
                            _gov_ok, _gov_msg = await self.governor.check_allow("create_goal", user_id=str(user_id))
                            if not _gov_ok:
                                return await self._safe_publish_response(
                                    text=_gov_msg,
                                    chat_id=str(chat_id),
                                    correlation_id=correlation_id,
                                    skill_results=[],
                                    user_text=text,
                                    user_lang=_user_lang,
                                    outer_trace=_decision_trace,
                                    reason="governor_block_create_goal",
                                )
                        # Route directly to GoalOrchestrator — no LLM round-trip needed.
                        # HIGH-3 Fix: await creation so we know whether it succeeded.
                        # create_goal() only writes metadata (fast) — the execution
                        # itself happens asynchronously in the background.
                        try:
                            await self.goal_orchestrator.create_goal(
                                objective=text,
                                chat_id=str(chat_id),
                                user_id=str(user_id),
                                priority=8,
                                source="telegram",
                            )
                            if self.governor:
                                await self.governor.record_action(
                                    "create_goal", user_id=str(user_id)
                                )
                            logger.info(
                                "goal.enqueue_success",
                                chat_id=str(chat_id),
                                objective=text[:80],
                            )
                            response_text = "✅ Objetivo registrado — ejecutando en segundo plano."
                        except Exception as _ge:
                            logger.warning(
                                "goal.enqueue_failure",
                                error=str(_ge)[:120],
                                chat_id=str(chat_id),
                            )
                            response_text = (
                                "⚠️ No se pudo registrar el objetivo en este momento. "
                                "Por favor intenta de nuevo."
                            )
                        return await self._safe_publish_response(
                            text=response_text,
                            chat_id=str(chat_id),
                            correlation_id=correlation_id,
                            skill_results=[],
                            user_text=text,
                            user_lang=_user_lang,
                            outer_trace=_decision_trace,
                            reason="fast_path_goal_enqueue",
                        )

            except Exception:
                _dl_strategy = Strategy.DIRECT_RESPONSE  # Never block on Decision Layer failure
        # ── End Decision Layer ────────────────────────────────────────────────────

        # ── Cross-Turn Retry State + Escalation Ladder (Phases 5 & 6) ────────────
        _action_attempts = 0
        _action_prev_state: dict | None = None
        if self.redis_url and _action_intent.action_commitment and not _is_scheduled_trigger:
            try:
                import json as _acj, time as _act
                _acr_load = aioredis.from_url(self.redis_url, decode_responses=True)
                async with _acr_load as _acrl:
                    _prev_raw = await _acrl.get(f"action_state:{chat_id}")
                if _prev_raw:
                    _prev = _acj.loads(_prev_raw)
                    # Restore attempt count when: same action type OR explicit retry signal
                    if (_prev.get("action_type") == _action_intent.action_type
                            or _action_intent.is_retry_signal):
                        _action_attempts = int(_prev.get("attempts", 0))
                        _action_prev_state = _prev
                        if _action_attempts > 0:
                            logger.info(
                                "action_intent.retry_state_loaded",
                                attempts=_action_attempts,
                                action_type=_action_intent.action_type,
                            )
            except Exception:
                pass  # Never block on Redis failure

        # Escalation: after 3+ failed direct attempts, route to Goal system.
        # Browser/web tasks are excluded — escalating a failing browser task to
        # a Goal just retries the same approach with more overhead.
        _is_browser_action = (_action_intent.primary_skill or "").startswith("browser")
        if (_action_intent.action_commitment
                and _action_attempts >= 3
                and not _is_browser_action
                and self.goal_orchestrator
                and not _is_scheduled_trigger
                and not _planning_mode):
            logger.info(
                "action_commitment.escalating_to_goal",
                attempts=_action_attempts,
                target=(_action_intent.action_target or "")[:60],
            )
            _esc_obj = text
            _esc_cid = str(chat_id)
            _esc_uid = str(user_id)
            _esc_orch = self.goal_orchestrator

            async def _esc_create_goal():
                try:
                    await _esc_orch.create_goal(
                        objective=_esc_obj,
                        chat_id=_esc_cid,
                        user_id=_esc_uid,
                        priority=9,
                        source="action_escalation",
                    )
                except Exception as _ge:
                    logger.warning("action_commitment.escalation_failed", error=str(_ge)[:80])

            asyncio.ensure_future(_esc_create_goal())
            # Save incremented attempt count
            if self.redis_url:
                try:
                    import json as _acj2, time as _act2
                    _acr_esc = aioredis.from_url(self.redis_url, decode_responses=True)
                    async with _acr_esc as _acwe:
                        await _acwe.setex(
                            f"action_state:{chat_id}",
                            1800,
                            _acj2.dumps({
                                "action_type": _action_intent.action_type,
                                "action_target": _action_intent.action_target,
                                "attempts": _action_attempts + 1,
                                "primary_skill": _action_intent.primary_skill,
                                "original_text": text[:200],
                                "timestamp": _act2.time(),
                            }),
                        )
                except Exception:
                    pass
            _esc_response = "Escalando a modo objetivo — ejecutando en segundo plano con plan detallado."
            return await self._safe_publish_response(
                text=_esc_response,
                chat_id=str(chat_id),
                correlation_id=correlation_id,
                skill_results=[],
                user_text=text,
                user_lang=_user_lang,
                outer_trace=_decision_trace,
                reason="action_escalation_to_goal",
            )
        # ── End Cross-Turn Retry / Escalation ────────────────────────────────────

        # Capability Engine: pre-LLM shortcut.
        # Skip for: (a) scheduled triggers, (b) messages classified as
        # SCHEDULED_TASK/SUB_AGENT by decision layer, (c) messages that look
        # like scheduling requests (checked unconditionally — auto_calls may have
        # fired a web_search on a long scheduling message, but we still must not
        # let the Capability Engine re-interpret it as an execution request).
        _is_scheduling_req = (
            _dl_strategy in (Strategy.SCHEDULED_TASK, Strategy.SUB_AGENT)
            or is_scheduling_request(text)
        )
        _cap_hit = None  # tracks whether capability engine produced a result
        # Skip Capability Engine if auto-detect already handled this request —
        # auto_calls fired a skill (e.g. browser_screenshot_full_page), so we must
        # not let the Capability Engine re-execute a different workflow on the same text.
        if not _is_scheduled_trigger and not _is_scheduling_req and not _planning_mode and not auto_calls and self.redis_url and self.skill_executor:
            try:
                from ..skills.capability_engine import CapabilityEngine
                _cap_engine = CapabilityEngine(
                    redis_url=self.redis_url,
                    skill_executor=self.skill_executor,
                )
                _cap_hit = await _cap_engine.try_execute(
                    text=text,
                    user_id=str(user_id),
                    chat_id=str(chat_id),
                )
                if _cap_hit is not None:
                    logger.info("capability_engine.hit", capability=_cap_hit.capability_name,
                                steps=len(_cap_hit.skills_executed))
                    # Communication Intelligence Layer: format raw skill outputs into
                    # natural language before sending to the user.
                    try:
                        _cap_response = await self._get_formatter().format_capability_result(
                            user_request=text,
                            capability_name=_cap_hit.capability_name,
                            raw_output=_cap_hit.output,
                            skills_executed=_cap_hit.skills_executed,
                        )
                        logger.info("formatter.capability_formatted",
                                    cap=_cap_hit.capability_name, chars=len(_cap_response))
                    except Exception as _fmt_err:
                        logger.debug("formatter.capability_fallback", error=str(_fmt_err)[:60])
                        _cap_response = _cap_hit.output  # fallback to raw
                    _cap_response = await self._safe_publish_response(
                        text=_cap_response,
                        chat_id=str(chat_id),
                        correlation_id=correlation_id,
                        skill_results=[],
                        user_text=text,
                        user_lang=_user_lang,
                        outer_trace=_decision_trace,
                        reason="capability_engine_hit",
                    )
                    return _cap_response, {}
            except Exception as _ce:
                logger.info("capability_engine.error", error=str(_ce)[:120])

        # Long-input guaranteed-response: send immediate acknowledgment before
        # starting the LLM processing so the user always sees something quickly.
        if len(text) > 300 and not auto_calls and not _is_scheduled_trigger:
            await _tg_progress({"type": "thinking", "round": 1})

        active_provider = self.model_manager.active_provider

        # v2.5: Adaptive health state — non-blocking, fail-open (must run before build_context)
        _health_tg = None
        try:
            from ..runtime.health_state import evaluate_health_from_redis as _eval_health
            _health_tg = await _eval_health(self.redis_url) if self.redis_url else None
        except Exception:
            pass

        _is_light = bool(_health_tg and _health_tg.mode == "light")
        async with async_session() as session:
            messages = await build_context(
                session, self.memory, text, str(chat_id),
                model_name=active_model, provider_name=active_provider,
                skill_catalog=skill_catalog,
                identity_manager=self.identity_manager,
                redis_url=self.redis_url,
                is_light_mode=_is_light,
            )

        # Inject context reset block into system prompt (hard intent switch guard)
        if _context_reset_block and messages and messages[0].role == "system":
            messages[0] = Message(
                role="system",
                content=messages[0].content + _context_reset_block,
            )
            logger.info("context_reset.block_injected", chat_id=str(chat_id))

        # Inject active flow context lock into system prompt (domain contamination guard)
        if _active_flow and messages:
            try:
                from .flow_state import build_flow_context_block as _build_flow_block
                _flow_block = _build_flow_block(_active_flow)
                if _flow_block and messages[0].role == "system":
                    messages[0] = Message(
                        role="system",
                        content=messages[0].content + "\n\n" + _flow_block,
                    )
                    logger.info("active_flow.context_injected",
                                chat_id=str(chat_id),
                                domain=_active_flow.get("domain"))
            except Exception:
                pass

        # Inject learned examples from positive feedback (few-shots that worked for this user)
        try:
            learned = await get_positive_examples(str(chat_id), limit=3)
            if learned:
                learned_text = format_learned_examples(learned)
                if learned_text and messages:
                    # Insert learned few-shots right before the final user message
                    insert_pos = len(messages) - 1
                    for ex in learned:
                        messages.insert(insert_pos, Message(role="user", content=ex["user_input"]))
                        insert_pos += 1
                        messages.insert(insert_pos, Message(role="assistant", content=ex["skill_calls"]))
                        insert_pos += 1
        except Exception:
            pass  # Learning injection never blocks the main flow

        # Planning mode: inject execution block into system prompt
        if _planning_mode and messages and messages[0].role == "system":
            _pm_block = (
                "\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                "[PLANNING MODE — NO EXECUTION]\n"
                "The user wants to see the plan ONLY. You must NOT execute anything.\n\n"
                "ABSOLUTE RULES — violation is not allowed:\n"
                "1. DO NOT generate any <skill> tags. None. Zero.\n"
                "2. DO NOT create tasks, goals, agents, scheduled jobs, or call any API.\n"
                "3. DO NOT call task_manager, goal creation, agent_manager, or any skill.\n\n"
                "REQUIRED RESPONSE STRUCTURE (use these labels translated to the user's language):\n"
                "1. OVERVIEW — brief description of the approach\n"
                "2. STEPS — numbered list of what would be done and with which tools\n"
                "3. ARCHITECTURE — how the components connect (if applicable)\n"
                "4. End with a clear invitation to execute, e.g. (in the user's language):\n"
                '   "When you want me to run it, tell me: execute / create this / launch it."\n'
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            )
            messages[0] = Message(
                role="system",
                content=messages[0].content + _pm_block,
            )
            logger.info("planning_mode.prompt_injected", chat_id=str(chat_id))

        # ── Action Commitment Injection (Phase 2 + Execution Planner) ───────────
        # If a plan was generated: inject plan-formatted block (system drives execution).
        # Otherwise: inject the classic commitment block (LLM-guided with hints).
        # Planning mode always takes priority — no injection in planning mode.
        if (_action_intent.action_commitment
                and not _planning_mode
                and not _is_scheduled_trigger
                and messages
                and messages[0].role == "system"):
            if _exec_plan is not None:
                # Plan available: structured step-by-step block replaces free-form hints
                _ac_block = _build_action_commitment_block(_action_intent, _action_attempts)
                _plan_block = format_plan_for_prompt(_exec_plan)
                messages[0] = Message(
                    role="system",
                    content=messages[0].content + _ac_block + _plan_block,
                )
                logger.info(
                    "execution_planner.plan_injected",
                    plan_id=_exec_plan.plan_id,
                    action_type=_exec_plan.action_type,
                    confidence=round(_exec_plan.confidence, 2),
                )
            else:
                # No plan: fall back to classic commitment block
                _ac_block = _build_action_commitment_block(_action_intent, _action_attempts)
                messages[0] = Message(
                    role="system",
                    content=messages[0].content + _ac_block,
                )
                logger.info(
                    "action_commitment.block_injected",
                    action_type=_action_intent.action_type,
                    target=(_action_intent.action_target or "")[:60],
                    attempts=_action_attempts,
                )
        # ── End Action Commitment Injection ────────────────────────────────────

        # ── Language directive injection ────────────────────────────────────
        # Inject into system prompt for ANY detected language including English.
        # Bug #6: EN users were getting Spanish output because the previous
        # condition only injected when lang != "en", so the LLM defaulted to
        # whatever the prime.md examples used (mostly Spanish).
        # Scheduled tasks are excluded — they produce internal output, not user messages.
        if _user_lang and not _is_scheduled_trigger and messages and messages[0].role == "system":
            _lang_block = _build_lang_directive(_user_lang)
            messages[0] = Message(
                role="system",
                content=messages[0].content + _lang_block,
            )

        # ── Tool Router cross-turn anti-retry hint ──────────────────────────
        # If the previous turn ended with a tool error (within TTL=120s),
        # inject a system-level note so the LLM doesn't blindly retry the tool.
        if not _is_scheduled_trigger and messages and messages[0].role == "system":
            _prev_tool_err = await _get_tool_error_memory(self.redis_url, str(chat_id))
            if _prev_tool_err:
                _ter_block = (
                    f"\n\n[TOOL_CONTEXT]\n"
                    f"Your previous attempt to use '{_prev_tool_err.get('tool', 'a tool')}' "
                    f"failed because: {_prev_tool_err.get('type', 'missing required input')}. "
                    f"Do NOT attempt the same tool call again UNLESS the missing requirement is now provided. "
                    f"If the user provides the required input (e.g. a valid URL), you may proceed normally. "
                    f"Otherwise, ask the user clearly for the missing information.\n"
                    f"[/TOOL_CONTEXT]"
                )
                messages[0] = Message(
                    role="system",
                    content=messages[0].content + _ter_block,
                )

        # If we have pre-executed skill results, inject few-shot examples + data
        # Small models learn from examples, not rules. Show the pattern first.
        if pre_results_text:
            insert_pos = len(messages) - 1  # Before the final user message
            for fs_user, fs_assistant in WEB_FEWSHOT:
                messages.insert(insert_pos, Message(role="user", content=fs_user))
                insert_pos += 1
                messages.insert(insert_pos, Message(role="assistant", content=fs_assistant))
                insert_pos += 1
            # Replace final user message with data-enriched version
            messages[-1] = Message(
                role="user",
                content=(
                    f"{text}\n\n[DATA]:\n{pre_results_text}\n[/DATA]\n\n"
                    "Analyze the [DATA] above and answer the user's specific question. "
                    "Be concise (2-4 sentences). Never copy raw text. "
                    "If the data doesn't contain what was asked, use <skill>web_search(query=\"...\")</skill> or "
                    "<skill>python_exec(code=\"import requests; r=requests.get('URL', headers={'User-Agent':'Mozilla/5.0'}); print(r.text[:3000])\"))</skill> to find it. "
                    "CRITICAL: NEVER respond with just 'Página abierta', 'No encontré información', or 'Voy a intentar de nuevo' WITHOUT including a <skill> tag. "
                    "If the page shows no products or a login wall, use python_exec with requests to scrape directly."
                ),
            )

        # LLM generation + skill execution loop (for skills model calls explicitly)
        response_text = ""
        _skill_round_count = 0  # tracks rounds where the LLM executed skills
        # Execution Intelligence: initialize trace
        import hashlib as _hashlib
        import json as _json_tr
        _trace_id = str(uuid4())
        _trace_start_ms = int(time.monotonic() * 1000)
        _trace_spans: list[dict] = []
        _task_id_for_trace = ""
        _task_name_for_trace = ""
        if _is_scheduled_trigger:
            # Phase 7: prefer explicit `id=<uuid>` if present (new format),
            # fall back to name (legacy / agent triggers).
            _tm = re.search(r"\[TAREA PROGRAMADA:\s*([^\]]+?)(?:\s*\|\s*id=([0-9a-f-]+))?\s*\]", text)
            if _tm:
                _task_name_for_trace = _tm.group(1).strip()
                _task_id_for_trace = (_tm.group(2) or "").strip() or _task_name_for_trace
        # Pre-populate with screenshots from the auto-detect phase.
        # When a browser capture() runs in auto-detect but falls through to LLM processing,
        # the screenshot must still be sent even though the LLM didn't call any new skills.
        pending_photos = []
        _invalid_photos: set = set()  # paths from [CAPTURE_VALID: false] — display only, not for reports
        _shot_re_pre = re.compile(r"(/data/screenshots/screenshot_\d+\.png)")
        for _pre_r in auto_results:
            if _pre_r.success:
                _pre_out = _pre_r.output or ""
                _pre_capture_invalid = "[CAPTURE_VALID: false]" in _pre_out
                for _pre_m in _shot_re_pre.finditer(_pre_out):
                    _pre_path = _pre_m.group(1)
                    if os.path.exists(_pre_path) and _pre_path not in pending_photos:
                        pending_photos.append(_pre_path)
                        if _pre_capture_invalid:
                            _invalid_photos.add(_pre_path)

        # Execution-scoped artifact registry — single source of truth for this turn.
        # Only VALID screenshots go here — these are used for email attachments and reports.
        _valid_pre = [p for p in pending_photos if p not in _invalid_photos]
        _exec_artifacts: dict[str, list[str]] = {"screenshots": list(_valid_pre)}
        # Capture render_report outputs keyed by format — used to build the final
        # Telegram summary from the same source as the email body.
        _render_report_outputs: dict[str, str] = {}  # format → rendered text
        logger.info(
            "artifact_registry.initialized",
            screenshot_count=len(_exec_artifacts["screenshots"]),
            paths=_exec_artifacts["screenshots"],
        )
        _round_image = image_path  # only inject image on the first round
        _intent_retry_done = False   # one completeness retry per request maximum
        _action_enforcement_done = False  # one enforcement retry per action-committed request
        # System-Controlled Execution Engine state
        _action_all_results: list = []           # accumulates all skill results across rounds
        _action_history: list = []               # ordered record of browser actions with args+outputs
        _action_terminal_detected = False        # set True once terminal state is reached
        _action_terminal_success = False         # terminal success/failure
        _action_terminal_reason = ""             # reason code
        _action_final_prompt_override = ""       # replaces continuation prompt when terminal

        # ── Pre-Execution Phase: run DETERMINISTIC plan steps directly ────────
        # For high-confidence plans (e.g. package tracking with known selectors),
        # execute all leading DETERMINISTIC steps without calling the LLM at all.
        # The LLM loop only starts after deterministic steps complete (or fail).
        # Intent-gate guard (deterministic_sequence path): block plans that imply
        # gmail.send when the user's current message has no explicit intent.
        # Sets _pre_intent_block_missing_recipient so the post-loop response
        # can substitute an explicit "what address?" ask.
        _pre_intent_block_missing_recipient = False
        _pre_intent_skip_plan = False
        if (_plan_executor is not None
                and _exec_plan is not None
                and _exec_plan.confidence >= 0.75
                and not _planning_mode
                and _action_intent.action_commitment
                and getattr(_action_intent, "action_type", "") == "email_send"):
            class _DummyGmailSC:
                def __init__(self):
                    self.skill_name = "gmail"
                    self.arguments = {"action": "send"}
            _ig_ok, _ig_reason, _ig_intent = _intent_gate_check(
                _DummyGmailSC(), text, ctx_messages=messages, chat_id=str(chat_id),
            )
            if not _ig_ok:
                logger.warning(
                    "intent_gate.blocked",
                    skill="gmail", action="send",
                    reason=_ig_reason, intent=_ig_intent,
                    path="deterministic_sequence",
                    text_preview=(text or "")[:80],
                )
                _pre_intent_skip_plan = True
                if _ig_reason == "missing_recipient":
                    _pre_intent_block_missing_recipient = True

        if (_plan_executor is not None
                and _exec_plan is not None
                and _exec_plan.confidence >= 0.75
                and not _planning_mode
                and _action_intent.action_commitment
                and not _pre_intent_skip_plan):
            _pre_step_execs = await _plan_executor.execute_deterministic_sequence(
                user_id=str(user_id), chat_id=str(chat_id)
            )
            for _pse in _pre_step_execs:
                for _pse_sc, _pse_r in zip(_pse.skill_calls, _pse.results):
                    _action_all_results.append(_pse_r)
                    if _pse_r.skill_name == "browser":
                        _sc_args = _pse_sc.arguments if isinstance(_pse_sc.arguments, dict) else {}
                        _full_pre_out = _pse_r.output or ""
                        _stored_pre_out = _full_pre_out[:4096]
                        # Guarantee critical status markers survive truncation
                        for _mk in ("[TRACK_STATUS:", "[FORM_STATUS:"):
                            if _mk in _full_pre_out and _mk not in _stored_pre_out:
                                for _mkl in _full_pre_out.split("\n"):
                                    if _mk in _mkl:
                                        _stored_pre_out += f"\n{_mkl.strip()}"
                                        break
                        _action_history.append({
                            "action": _sc_args.get("action", "unknown"),
                            "url": _sc_args.get("url", ""),
                            "selector": (_sc_args.get("selector", "") or "")[:80],
                            "text": (_sc_args.get("text", "") or "")[:60],
                            "success": _pse_r.success,
                            "output": _stored_pre_out,
                            "round": -1,  # Pre-execution round marker
                        })
                if _pre_step_execs:
                    logger.info(
                        "execution_planner.pre_execution",
                        step_id=_pse.step_id,
                        success=_pse.success,
                        output_snippet=(_pse.output or "")[:80],
                    )
            # Check if pre-execution already reached terminal state
            if _action_all_results and not _action_terminal_detected:
                _pre_is_term, _pre_success, _pre_reason = _check_action_terminal_state(
                    _action_intent, _action_all_results
                )
                if _pre_is_term:
                    _steps_ok_pre, _steps_reason_pre = _verify_required_steps(
                        _action_intent.action_type, _action_history
                    )
                    if _steps_ok_pre:
                        _action_terminal_detected = True
                        _action_terminal_success = _pre_success
                        _action_terminal_reason = _pre_reason
                        _obj_evidence_pre = (
                            _pre_reason.split("objective_confirmed:", 1)[1]
                            if "objective_confirmed:" in _pre_reason else _pre_reason
                        )
                        _action_final_prompt_override = _build_constrained_final_prompt(
                            _action_intent, _action_all_results, _pre_success, _pre_reason
                        )
                        logger.info(
                            "execution_planner.pre_execution_terminal",
                            success=_pre_success,
                            reason=_pre_reason,
                            steps_completed=len(_action_history),
                        )
        # ── End Pre-Execution Phase ────────────────────────────────────────────

        # ── Pre-Execution Short-Circuit ────────────────────────────────────────
        # If pre-execution already reached terminal state, inject the results and
        # final constrained prompt into the message context NOW so the LLM can
        # format the response in round 0 without re-executing any skills.
        # Without this the LLM sees an empty context and wastes a full browser
        # round re-executing what pre-execution already did.
        if _action_terminal_detected and _action_final_prompt_override and _action_all_results:
            _pre_out_lines = []
            for _pre_r in _action_all_results[:3]:
                _pre_snippet = (_pre_r.output or "")[:1500]
                _pre_out_lines.append(f"[pre_exec:{_pre_r.skill_name}] {_pre_snippet}")
            _pre_exec_summary = "\n".join(_pre_out_lines)
            messages.append(Message(
                role="user",
                content=(
                    "[SISTEMA] Ejecución automática completada antes de tu turno:\n\n"
                    f"{_pre_exec_summary}\n\n"
                    f"{_action_final_prompt_override}"
                ),
            ))
            _action_final_prompt_override = ""  # consumed — do not inject again in loop
            logger.info(
                "execution_planner.pre_execution_shortcircuit",
                results=len(_action_all_results),
                reason=_action_terminal_reason,
            )
        # ── End Pre-Execution Short-Circuit ───────────────────────────────────

        # ── Unified LLM Round Loop ────────────────────────────────────────────
        # HIGH-2: Load any confirmed domain lock persisted from the previous turn.
        # If a lock is already present (e.g., from a redirect or intent override),
        # the persisted lock is not used (trust the in-turn derivation over stale).
        _persisted_lock_tg = None
        if self.redis_url:
            try:
                from .control_layer import load_domain_lock as _load_dl
                _persisted_lock_tg = await _load_dl(self.redis_url, str(chat_id))
            except Exception as _dl_load_err:
                logger.debug("domain_lock.load_skip", error=str(_dl_load_err)[:60])

        # ── Domain lock topic-change validation (HARDENED) ────────────────────────
        # Always run validate_lock_for_reuse to clear locks that no longer match the
        # new turn's topic. Then a second pass auto-clears on explicit URL mismatch.
        # This prevents a lock earned during a browser flow from blocking unrelated
        # follow-ups (task adjustments, email, tracking) in subsequent turns.
        if _persisted_lock_tg and self.redis_url:
            try:
                from .control_layer import (
                    validate_lock_for_reuse as _validate_lock,
                    clear_domain_lock as _clear_dl,
                    extract_domains_from_text as _edft,
                )
                import re as _re_dl
                _validated, _validate_reason = _validate_lock(_persisted_lock_tg, text or "")
                if _validated is None:
                    _old_repr = _persisted_lock_tg.serialize() if hasattr(_persisted_lock_tg, "serialize") else str(_persisted_lock_tg)
                    await _clear_dl(
                        self.redis_url,
                        str(chat_id),
                        reason=_validate_reason or "topic_changed",
                        old_domain=_old_repr,
                    )
                    logger.info(
                        "domain_lock.auto_cleared",
                        reason=_validate_reason or "topic_changed",
                        old_domain=_old_repr,
                    )
                    _persisted_lock_tg = None
                else:
                    # Second-pass: explicit URL pointing to a different domain still wins.
                    _explicit_urls = _re_dl.findall(r"https?://[^\s]+", text or "")
                    if _explicit_urls:
                        _url_domains = [_edft(u) for u in _explicit_urls]
                        _url_domains = [d for d in _url_domains if d]
                        if _url_domains and not all(_persisted_lock_tg.allows(d) for d in _url_domains):
                            _old_repr = _persisted_lock_tg.serialize() if hasattr(_persisted_lock_tg, "serialize") else str(_persisted_lock_tg.domains)
                            await _clear_dl(
                                self.redis_url,
                                str(chat_id),
                                reason="user_explicit_url_mismatch",
                                old_domain=_old_repr,
                            )
                            logger.info(
                                "domain_lock.auto_cleared",
                                reason="user_explicit_url_mismatch",
                                old_domain=_old_repr,
                                new_domains=_url_domains,
                            )
                            _persisted_lock_tg = None
            except Exception as _dl_clear_err:
                logger.debug("domain_lock.auto_clear_skip", error=str(_dl_clear_err)[:80])

        # ── Phase 4: Retrieve execution pattern before loop (Telegram) ──────────
        _em_hint_tg = ""
        _em_pattern_id_tg = ""
        if self.redis_url and getattr(_action_intent, "action_commitment", False):
            try:
                from .execution_memory import execution_memory as _em_tg
                _em_pattern_tg = await _em_tg.find_pattern(
                    self.redis_url,
                    _action_intent.action_type,
                    getattr(_action_intent, "objective_spec", None),
                    _persisted_lock_tg,
                )
                if _em_pattern_tg:
                    _em_hint_tg = _em_tg.format_hint(_em_pattern_tg)
                    _em_pattern_id_tg = _em_pattern_tg.get("id", "")
            except Exception as _em_find_err:
                logger.debug("execution_memory.find_skip_tg", error=str(_em_find_err)[:80])

        _lctx = _LoopContext(
            messages=messages,
            text=text,
            user_id=str(user_id),
            chat_id=str(chat_id),
            execution_id=_request_execution_id,
            image_path=_round_image,
            planning_mode=_planning_mode,
            is_scheduled_trigger=_is_scheduled_trigger,
            start_time=time.monotonic(),
            auto_results=list(auto_results),
            exec_artifacts=_exec_artifacts,
            render_report_outputs=_render_report_outputs,
            action_intent=_action_intent,
            exec_plan=_exec_plan,
            progress_callback=_tg_progress,
            active_domain_lock=_persisted_lock_tg,
            invalid_photos=set(_invalid_photos),
            pattern_hint=_em_hint_tg,
            reused_pattern_id=_em_pattern_id_tg,
            health_state=_health_tg,
        )
        # Carry pre-LLM intent-gate state into ctx so post-loop response builder
        # can substitute the "ask for recipient" line if needed.
        if _pre_intent_block_missing_recipient:
            _lctx.intent_missing_recipient_seen = True

        # v2.5: Light-mode — inject hint + cap skill rounds to reduce CPU load
        if _health_tg and _health_tg.mode == "light":
            _light_hint = (
                "\n[SYSTEM_CONSTRAINT: LIGHT_MODE — system under high CPU load. "
                "Prefer direct answers over tool calls. "
                "If you must use a tool, use at most 1 (prefer fetch_url over browser). "
                "Skip non-essential background lookups.]"
            )
            if "SYSTEM_CONSTRAINT: LIGHT_MODE" not in (_lctx.pattern_hint or ""):
                _lctx.pattern_hint = (_lctx.pattern_hint or "") + _light_hint
            # Cap max skill rounds to 2 under load (default is typically 6-12)
            if not hasattr(_lctx, "_light_rounds_capped"):
                _lctx.__dict__["_light_rounds_capped"] = True
                _lctx.__dict__["max_rounds_override"] = 2
            logger.info("light_mode.active", cpu=getattr(_health_tg, "cpu_percent", "?"))

        response_text = await self._run_llm_loop(_lctx)
        # HIGH-2: Persist confirmed domain lock for next turn (TTL 600s).
        if self.redis_url and _lctx.active_domain_lock:
            try:
                from .control_layer import persist_domain_lock as _persist_dl
                await _persist_dl(self.redis_url, str(chat_id), _lctx.active_domain_lock)
            except Exception as _dl_save_err:
                logger.debug("domain_lock.persist_skip", error=str(_dl_save_err)[:60])

        # ── Phase 4: Store / update pattern after loop (Telegram) ────────────
        if self.redis_url and getattr(_action_intent, "action_commitment", False):
            _em_terminal_ok = _lctx.action_terminal_success
            _em_no_block = not _lctx.browser_timed_out
            if _em_terminal_ok and _em_no_block:
                # Verify spec before storing
                try:
                    from .control_layer import validate_against_spec as _em_vas
                    _em_spec = getattr(_action_intent, "objective_spec", None)
                    _em_spec_ok, _ = _em_vas(
                        _lctx.last_results_text, _lctx.action_all_results, _em_spec
                    )
                except Exception:
                    _em_spec_ok = True   # fail-open: if validation errors, allow store
                if _em_spec_ok:
                    try:
                        from .execution_memory import execution_memory as _em_store
                        await _em_store.store_pattern(
                            self.redis_url,
                            _action_intent.action_type,
                            getattr(_action_intent, "objective_spec", None),
                            _lctx.action_history,
                            _lctx.action_all_results,
                            _lctx.active_domain_lock,
                        )
                    except Exception as _em_store_err:
                        logger.debug("execution_memory.store_skip_tg", error=str(_em_store_err)[:80])
            elif not _em_terminal_ok and _lctx.reused_pattern_id:
                # Pattern was injected but execution failed → decay success_rate
                try:
                    from .execution_memory import execution_memory as _em_fail
                    await _em_fail.record_failure(self.redis_url, _lctx.reused_pattern_id)
                except Exception as _em_fail_err:
                    logger.debug("execution_memory.record_failure_skip_tg", error=str(_em_fail_err)[:80])
        # Surface accumulated state for post-loop code
        _skill_round_count        = _lctx.skill_round_count
        _trace_spans              = _lctx.trace_spans
        _action_terminal_detected = _lctx.action_terminal_detected
        _action_terminal_success  = _lctx.action_terminal_success
        _action_all_results       = _lctx.action_all_results
        _invalid_photos.update(_lctx.invalid_photos)
        for _lp in _lctx.media_paths:
            if _lp not in pending_photos:
                pending_photos.append(_lp)

        # ── Save Cross-Turn Retry State (Phase 5) ──────────────────────────────
        if self.redis_url and _action_intent.action_commitment and not _is_scheduled_trigger:
            try:
                import json as _acjs, time as _acts
                _ac_execution_succeeded = _action_terminal_success  # True only on verified terminal success
                # Reset attempt counter on success; increment on failure
                _new_attempts = 0 if _ac_execution_succeeded else (_action_attempts + 1)
                _acr_save = aioredis.from_url(self.redis_url, decode_responses=True)
                async with _acr_save as _acws:
                    await _acws.setex(
                        f"action_state:{chat_id}",
                        1800,  # 30 min TTL
                        _acjs.dumps({
                            "action_type": _action_intent.action_type,
                            "action_target": _action_intent.action_target,
                            "attempts": _new_attempts,
                            "primary_skill": _action_intent.primary_skill,
                            "original_text": text[:200],
                            "last_success": _ac_execution_succeeded,
                            "timestamp": _acts.time(),
                        }),
                    )
                if _ac_execution_succeeded:
                    logger.info(
                        "action_execution.success",
                        action_type=_action_intent.action_type,
                        target=(_action_intent.action_target or "")[:60],
                    )
                else:
                    logger.info(
                        "action_execution.failed",
                        action_type=_action_intent.action_type,
                        target=(_action_intent.action_target or "")[:60],
                        attempts=_new_attempts,
                    )
            except Exception:
                pass  # Never block on Redis failure
        # ── End Save Cross-Turn Retry State ────────────────────────────────────

        # Strip any remaining skill tags and leaked [DATA] blocks from response
        response_text = strip_skill_calls(response_text)
        response_text = _strip_data_blocks(response_text)
        # If the LLM still returned raw JSON, extract the relevant value
        response_text = _extract_from_json_response(response_text)
        response_text = _strip_internal_paths(response_text)
        # Clean markdown artifacts and prompt leakage before sending to Telegram
        response_text = _clean_telegram_output(response_text)

        # ── Schedule honesty enforcement ────────────────────────────────────
        # Combines action-flow results AND full LLM-loop results so we catch
        # both: (a) the LLM hallucinating a scheduled task without calling
        # task_manager, and (b) task_manager called but with a fixed time that
        # task_manager cannot honor. The system is the final authority.
        try:
            _all_turn_results = list(_action_all_results) + list(
                getattr(_lctx, "loop_skill_results", []) or []
            )
            # ── Honesty Layer (NEW, runs FIRST) ───────────────────────────
            # Catches: response talks about a topic (screenshot/email/...) whose
            # supporting skill never ran successfully in THIS turn. Either
            # strips the leaked sentences or replaces with a canonical "I
            # could not complete X" message bound to the actual failure.
            try:
                from .response_binding import apply_honesty_layer as _honesty_tg
                response_text, _hon_trace = _honesty_tg(
                    response_text,
                    skill_results=_all_turn_results,
                    user_text=text,
                    user_lang=_user_lang,
                    action_intent=_action_intent,
                )
                if _hon_trace.get("status") != "passthrough":
                    logger.info(
                        "honesty_layer.applied",
                        chat_id=str(chat_id),
                        status=_hon_trace.get("status"),
                        leaked=_hon_trace.get("leaked_topics", []),
                        grounded=_hon_trace.get("grounded_topics", []),
                        reason=_hon_trace.get("reason", ""),
                    )
                    try:
                        _dtrace = locals().get("_decision_trace")
                        if _dtrace is not None:
                            _dtrace.attach_response_guard({"honesty_layer": _hon_trace})
                    except Exception:
                        pass
            except Exception as _hl_err:
                logger.debug("honesty_layer.skip", error=str(_hl_err)[:120])

            # Single final-response policy entry point. Schedule honesty +
            # side-effect text gate run in order; the trace dict feeds the
            # request DecisionTrace so operators can answer "why".
            response_text, _guard_trace = _apply_final_response_policy(
                response_text,
                user_text=text,
                skill_results=_all_turn_results,
                user_lang=_user_lang,
                chat_id=str(chat_id),
                recent_action_resolver=_get_recent_explicit_action,
            )
            try:
                _dtrace = locals().get("_decision_trace")
                if _dtrace is not None:
                    _dtrace.attach_response_guard(_guard_trace)
            except Exception:
                pass
        except Exception:
            pass  # Never block response on enforcement failure

        # ── Skill-level placeholder block detection ───────────────────────
        # Catches the case where gmail.send was blocked by the skill itself
        # (placeholder subject+body) — the gate didn't fire, but the skill
        # refused. Promote that error to the same "missing_content" UX path.
        try:
            for _sr in _all_turn_results or []:
                if (getattr(_sr, "skill_name", "") == "gmail"
                        and not getattr(_sr, "success", True)
                        and "Placeholder content detected" in (getattr(_sr, "error", "") or "")):
                    _lctx.intent_missing_content_seen = True
                    break
        except Exception:
            pass

        # ── Missing-content honest ask ────────────────────────────────────
        # gmail.send blocked for placeholder/missing subject+body. Substitute
        # an explicit "what content?" question if the LLM did not produce one.
        if getattr(_lctx, "intent_missing_content_seen", False):
            _is_es_mc = _user_lang == "es"
            _ask_c = (
                "¿Qué quieres que envíe en el correo?"
                if _is_es_mc else
                "What do you want me to send in the email?"
            )
            _resp_norm_c = (response_text or "").strip().lower()
            _looks_like_ask_c = (
                "qué" in _resp_norm_c or "que enviar" in _resp_norm_c or
                "what should" in _resp_norm_c or "what do you" in _resp_norm_c or
                "what content" in _resp_norm_c or "qué contenido" in _resp_norm_c or
                "qué texto" in _resp_norm_c
            )
            _looks_like_fallback_c = (
                not response_text or
                "could not generate" in _resp_norm_c or
                "no pude generar" in _resp_norm_c
            )
            if _looks_like_fallback_c or not _looks_like_ask_c:
                response_text = _ask_c

        # ── Missing-recipient honest ask ──────────────────────────────────
        # If gmail.send was blocked at the gate for missing_recipient and the
        # LLM did not generate a clear "what address?" question on its own,
        # substitute a direct ask. The user must be told what's needed.
        if getattr(_lctx, "intent_missing_recipient_seen", False):
            _is_es_mr = _user_lang == "es"
            _ask = (
                "¿A qué dirección de correo quieres que lo envíe?"
                if _is_es_mr else
                "What email address should I send it to?"
            )
            # If the response is empty, generic, or doesn't already ask for an address,
            # replace it with a clean explicit ask.
            _resp_norm = (response_text or "").strip().lower()
            _looks_like_ask = (
                "@" in (response_text or "") or
                "dirección" in _resp_norm or "direccion" in _resp_norm or
                "destinat" in _resp_norm or "address" in _resp_norm or
                "recipient" in _resp_norm or "to whom" in _resp_norm or
                "a quién" in _resp_norm or "a quien" in _resp_norm
            )
            _looks_like_fallback = (
                not response_text or
                "could not generate" in _resp_norm or
                "no pude generar" in _resp_norm
            )
            if _looks_like_fallback or not _looks_like_ask:
                response_text = _ask

        # ── Intent-gate streak note ───────────────────────────────────────
        # If the LLM kept trying side-effects the user did not authorize,
        # surface a short user-facing note and ensure the response asks
        # for the missing authorization (e.g. recipient).
        if getattr(_lctx, "intent_block_streak_exceeded", False):
            from ..communication.translator import apick as _apick_streak
            _streak_seed = (correlation_id or "") + ":streak"
            _streak_note_canonical = await _apick_streak(
                "intent_block_streak_note", _user_lang or "en", _streak_seed,
                self.model_manager, self.redis_url,
            )
            if response_text and response_text.strip():
                response_text = response_text.rstrip() + "\n\n(" + _streak_note_canonical + ")"
            else:
                response_text = await _apick_streak(
                    "intent_block_streak_only", _user_lang or "en", _streak_seed + ":only",
                    self.model_manager, self.redis_url,
                )

        # ── Request-budget partial-result note ────────────────────────────
        # If the per-request round budget stopped the loop early, append an
        # honest "I had to stop" note so the user knows the result is partial.
        if getattr(_lctx, "budget_stopped", False):
            _bs = getattr(_lctx, "budget_stop_state", {}) or {}
            _is_es = _user_lang == "es"
            _partial_note = (
                f"\n\n(Tuve que detenerme tras {_bs.get('used','?')} pasos para no seguir explorando. "
                f"Te entrego el mejor resultado verificado hasta ahora; pídeme una continuación si falta algo.)"
                if _is_es else
                f"\n\n(I had to stop after {_bs.get('used','?')} rounds to avoid further exploration. "
                f"This is the best verified result so far; ask me to continue if something is missing.)"
            )
            if response_text and response_text.strip():
                response_text = response_text.rstrip() + _partial_note
            else:
                response_text = (
                    "Me quedé sin presupuesto de exploración antes de poder responder. "
                    "Pídelo más simple o dime exactamente qué necesitas."
                    if _is_es else
                    "I ran out of exploration budget before I could finish. "
                    "Try asking more simply or tell me exactly what you need."
                )

        # Free the per-request budget slot — the request is finishing.
        try:
            request_budget_release(_request_execution_id)
        except Exception:
            pass

        # Persist the decision trace (Redis, fire-and-forget, fail-safe).
        # Operators can now answer "why did WASP do this?" from one record.
        try:
            _decision_trace.detected_language = _user_lang
            if not getattr(_decision_trace, "_recorded", False):
                _decision_trace._recorded = True
                asyncio.ensure_future(_record_decision_trace(self.redis_url, _decision_trace))
        except Exception:
            pass

        # ── Cognitive note surfacing (silent-intelligence UX fix) ────────────
        # If the cognitive layer steered behavior this turn, append a one-line
        # user-facing note so the operator sees WHY the agent adapted.  The
        # note is consumed (popped) here so it never appears twice.
        try:
            _cog_note_user = await _consume_cognitive_note(self.redis_url, str(chat_id))
            if _cog_note_user and _cog_note_user not in (response_text or ""):
                response_text = f"{response_text}\n\n({_cog_note_user})".strip()
        except Exception:
            pass

        # ── Recent-output dedup ──────────────────────────────────────────────
        # Block exact-duplicate responses sent within a 60s window so the
        # agent doesn't tell the user the same thing twice in adjacent turns.
        # Canonical EN message → translated to user_lang via the central
        # translator. Cached after first call per language.
        try:
            if await _is_duplicate_response(self.redis_url, str(chat_id), response_text):
                logger.info("handler.duplicate_response_suppressed", chat_id=str(chat_id))
                from ..communication.translator import apick as _apick_dedup
                response_text = await _apick_dedup(
                    "dedup_response",
                    _user_lang or "en",
                    str(chat_id) + ":dedup",
                    self.model_manager,
                    self.redis_url,
                )
        except Exception:
            pass

        # ── Hard Response Validation + Auto-Recovery Layer ───────────────────────
        # Validates BEFORE sending. If blocked: attempt_recovery() re-runs the
        # missing skill steps (max 2 retries) before falling back to a safe message.
        # Only count spans where success=True — failed executions do NOT count as completed
        _executed_skills = (
            {s.get("skill", "") for s in _trace_spans if s.get("success")}
            | {r.skill_name for r in auto_results if r.success}
            | (set(_cap_hit.skills_executed) if _cap_hit is not None else set())
        )
        # Add skill:action granularity — only from successful spans
        _executed_skill_actions: set[str] = {
            f"{s.get('skill', '')}:{(s.get('args') or {}).get('action', '')}"
            for s in _trace_spans
            if s.get("success") and (s.get('args') or {}).get('action')
        }
        _executed_skills |= _executed_skill_actions
        _has_skill_data = _skill_round_count > 0 or (_cap_hit is not None) or len(auto_results) > 0
        _validation = ResponseValidator().validate(
            response_text=response_text,
            user_input=text,
            executed_skills=_executed_skills,
            has_any_skill_data=_has_skill_data,
            planning_mode=_planning_mode,
            reset_context=_context_reset_info,
        )
        # Phase 2 — Screenshot completeness check (after main validator)
        if _validation.valid:
            _screenshot_check = check_screenshot_completeness(
                response_text, text, _executed_skills
            )
            if not _screenshot_check.valid:
                _validation = _screenshot_check

        _recovered = False  # track whether recovery succeeded for trace accuracy
        if not _validation.valid:
            logger.warning(
                "response_validator.blocked",
                reason=_validation.reason,
                should_retry=_validation.should_retry,
                user_input=text[:100],
                response_preview=response_text[:100],
            )
            # Phase 3 — Recovery only when validator explicitly allows it
            # Note: drift with should_retry=False (e.g. browser→crypto substitution) must NOT recover
            _can_recover = _validation.should_retry
            if _can_recover and self.skill_executor:
                # Auto-recovery: complete the missing work, then re-validate
                response_text, _recovered = await attempt_recovery(
                    validation_result=_validation,
                    response_text=response_text,
                    user_input=text,
                    messages=messages,
                    model_manager=self.model_manager,
                    skill_executor=self.skill_executor,
                    user_id=str(user_id),
                    chat_id=str(chat_id),
                    cleanup_fn=_clean_telegram_output,
                    redis_url=getattr(self, "redis_url", ""),  # Phase 4 — RecoveryMemory
                )
            else:
                logger.warning(
                    "agent.fallback_response",
                    input=text[:120],
                    reason="unresolved_path",
                    validation_reason=getattr(_validation, "reason", ""),
                    chat_id=str(chat_id),
                )
                response_text = _validation.fallback_response

        # Store episodic memory with actual response
        async with async_session() as session:
            await self.memory.store_episodic(
                session,
                event_type="telegram.message",
                user_input=text,
                agent_response=response_text,
                user_id=str(user_id),
                chat_id=str(chat_id),
            )

        # Execution Intelligence: persist trace
        if self.redis_url:
            _trace_status = "complete" if _trace_spans else "no_skills"
            asyncio.ensure_future(_do_persist_trace(
                redis_url=self.redis_url,
                trace_id=_trace_id,
                chat_id=str(chat_id),
                user_id=str(user_id),
                spans=_trace_spans,
                start_ms=_trace_start_ms,
                is_scheduled=_is_scheduled_trigger,
                task_id=_task_id_for_trace,
                status=_trace_status,
                user_text=text,
            ))

        # Update last-active timestamp (used by Dream Mode inactivity detection)
        # Collect skill sequence info for procedural memory abstraction
        _skill_sequence = []
        _unique_skills_used = set()
        _skill_round_count = 0
        for msg in messages:
            if getattr(msg, 'role', '') == 'assistant':
                for sc in parse_skill_calls(msg.content):
                    _skill_sequence.append({
                        "skill_name": sc.skill_name,
                        "arguments": {k: v for k, v in sc.arguments.items() if k not in ("chat_id", "user_id")},
                    })
                    _unique_skills_used.add(sc.skill_name)
                if parse_skill_calls(msg.content):
                    _skill_round_count += 1

        # Execution validation for scheduled task triggers.
        # After the LLM loop, verify that any skill explicitly named in the task instruction
        # was actually called. If critical skills were skipped, inject one corrective round.
        # This is GENERIC — it scans the task instruction for known skill names.
        if _is_scheduled_trigger and self.skill_executor:
            _KNOWN_SKILLS_RE = re.compile(
                r"\b(gmail|browser|http_request|fetch_url|telegram|notify|"
                r"task_manager|agent_manager|web_search|shell|python_exec)\b",
                re.IGNORECASE,
            )
            _instruction_skills = {
                m.group(1).lower()
                for m in _KNOWN_SKILLS_RE.finditer(text)
            }
            # Use _trace_spans (success=True only) — _skill_sequence records ALL calls
            # including failed ones, which would mask gmail failures from the corrective loop.
            _executed_skills_ok = {
                s.get("skill", "").lower() for s in _trace_spans if s.get("success")
            }
            # For report tasks (informe/reporte/crypto), render_report is also critical
            _is_report_task = bool(re.search(
                r"\b(informe|reporte|report|cripto|crypto|btc|eth|sol)\b", text, re.IGNORECASE
            ))
            _critical_expected = _instruction_skills & {"gmail"}
            if _is_report_task and "gmail" in _instruction_skills:
                _critical_expected.add("render_report")
            _missing_critical = _critical_expected - _executed_skills_ok
            if _missing_critical:
                logger.warning(
                    "handler.scheduled_task_missing_skills",
                    missing=list(_missing_critical),
                    executed=list(_executed_skills),
                )
                # Inject a corrective LLM round to complete the missing step
                _corrective_prompt = (
                    f"[AUTOMATIC VALIDATION] The following steps did not execute and are mandatory: "
                    f"{', '.join(_missing_critical)}. "
                    f"Execute those missing steps NOW using the data already collected. "
                    f"Do NOT take screenshots again. Only complete the missing steps."
                )
                try:
                    messages.append(Message(role="user", content=_corrective_prompt))
                    _corr_resp = await self.model_manager.generate(
                        ModelRequest(messages=messages, max_tokens=1024)
                    )
                    _corr_text = _corr_resp.content if _corr_resp else ""
                    _corr_calls = parse_skill_calls(_corr_text)
                    if _corr_calls and self.skill_executor:
                        _corr_results = await self.skill_executor.execute_batch(
                            _corr_calls, user_id=str(user_id), chat_id=str(chat_id),
                            execution_id=_request_execution_id,
                        )
                        _corr_lines = []
                        for _cr in _corr_results:
                            if _cr.success:
                                _corr_lines.append(f"[skill:{_cr.skill_name}] {_cr.output}")
                                _skill_sequence.append({"skill_name": _cr.skill_name, "arguments": {}})
                            else:
                                _corr_lines.append(f"[skill:{_cr.skill_name}] ERROR: {_cr.error}")
                        if _corr_lines:
                            messages.append(Message(role="assistant", content=_corr_text))
                            messages.append(Message(role="user", content="\n".join(_corr_lines)))
                            _final_corr = await self.model_manager.generate(
                                ModelRequest(messages=messages, max_tokens=512)
                            )
                            if _final_corr and _final_corr.content.strip():
                                response_text = _final_corr.content.strip()
                except Exception as _ve:
                    logger.warning("handler.validation_retry_failed", error=str(_ve))

            # Atomic failsafe: if after corrective round the report task still didn't
            # complete all critical steps, return a clean failsafe message instead of
            # a partial/confusing response. Never send broken output.
            if _is_report_task and _missing_critical:
                _now_executed = {s["skill_name"].lower() for s in _skill_sequence}
                _still_missing = _critical_expected - _now_executed
                if _still_missing:
                    logger.warning(
                        "handler.scheduled_report_failsafe",
                        missing=list(_still_missing),
                        executed=list(_now_executed),
                    )
                    logger.error(
                        "agent.execution_mismatch",
                        action="scheduled_report",
                        result="incomplete",
                        reason=f"critical skills not executed: {list(_still_missing)}",
                        input=text[:120],
                        chat_id=str(chat_id),
                    )
                    response_text = (
                        "No pude completar el informe correctamente en esta ejecucion. "
                        "Intentare nuevamente en el proximo ciclo."
                    )
                    # Store active flow so the user's follow-up stays anchored to this workflow
                    if self.redis_url:
                        try:
                            from .flow_state import (
                                save_active_flow as _save_flow,
                                detect_flow_assets as _detect_assets,
                            )
                            _af_assets = _detect_assets(text)
                            _af_delivery = []
                            _text_lower = text.lower()
                            if any(k in _text_lower for k in ("gmail", "email", "correo", "mail")):
                                _af_delivery.append("email")
                            if "telegram" in _text_lower:
                                _af_delivery.append("telegram")
                            await _save_flow(self.redis_url, str(chat_id), {
                                "domain": "crypto",
                                "flow_type": "CRYPTO_REPORT",
                                "assets": _af_assets or ["BTC", "ETH", "SOL"],
                                "delivery": _af_delivery or ["email"],
                                "last_failure": f"Pasos faltantes: {', '.join(_still_missing)}",
                                "instruction": text[:200],
                            })
                        except Exception:
                            pass

        # ── Scheduled report: unify Telegram output from render_report ───────────
        # For scheduled report tasks that completed successfully, override whatever
        # the LLM wrote as the final response with the deterministic render_report
        # telegram output. This guarantees Telegram always shows the same data as
        # the email — never just "email sent" with no content.
        if _is_scheduled_trigger and _is_report_task and _render_report_outputs:
            # Only override Telegram response when gmail actually succeeded (success=True).
            # _trace_spans has per-call success flags; _skill_sequence does not.
            _gmail_succeeded = any(
                s.get("skill", "").lower() == "gmail" and s.get("success")
                for s in _trace_spans
            )
            if _gmail_succeeded:
                # Prefer explicit telegram format; fall back to email output with trim
                _tg_body = _render_report_outputs.get("telegram", "")
                if not _tg_body and "email" in _render_report_outputs:
                    # Trim email report to first ~10 lines as compact Telegram summary
                    _email_lines = _render_report_outputs["email"].splitlines()
                    _tg_body = "\n".join(_email_lines[:12]).strip()
                if _tg_body:
                    response_text = _tg_body + "\n\nCorreo enviado con el informe completo."
                    logger.info("handler.telegram_unified_from_render_report",
                                fmt="telegram" if "telegram" in _render_report_outputs else "email_trim")
                    # Clear active flow lock — workflow succeeded
                    if self.redis_url:
                        try:
                            from .flow_state import clear_active_flow as _clear_flow_ok
                            await _clear_flow_ok(self.redis_url, str(chat_id))
                        except Exception:
                            pass

        # Planning-only escalation: if LLM produced no skill calls for a complex message,
        # escalate to the goal engine fire-and-forget so Telegram responds immediately.
        if (
            _skill_round_count == 0
            and len(text) > 300
            and self.goal_orchestrator is not None
            and _PLANNING_ONLY_RE.search(response_text)
        ):
            _orch_ref = self.goal_orchestrator
            _text_snap = text
            _chat_snap = str(chat_id)
            _user_snap = str(user_id)

            async def _bg_create_goal():
                try:
                    await _orch_ref.create_goal(
                        objective=_text_snap,
                        chat_id=_chat_snap,
                        user_id=_user_snap,
                        priority=8,
                        source="telegram",
                    )
                except Exception as _ge:
                    logger.warning("handler.planning_escalation_failed", error=str(_ge))

            asyncio.ensure_future(_bg_create_goal())
            response_text = (
                f"✅ Objetivo registrado — trabajando en ello en segundo plano. "
                f"Te notificaré cuando esté listo."
            )
            logger.info("handler.planning_escalated_to_goal_bg", chat_id=_chat_snap, text_len=len(text))

        # Build cognitive trace — snapshot of what happened this message
        _cognitive_trace: dict = {
            "model": self.model_manager.active_model,
            "provider": self.model_manager.active_provider,
            "rounds": _skill_round_count,
            "skills_called": [s["skill_name"] for s in _skill_sequence],
            "unique_skills": list(_unique_skills_used),
            "kg_active": bool(self.redis_url),
            "epistemic_active": bool(self.redis_url),
            "temporal_active": bool(self.redis_url),
            "procedures_checked": bool(self.redis_url),
            "has_image": bool(image_path),
            "has_audio": bool(audio_path),
        }

        # Post-message async fire-and-forget (KG, self-model, temporal, epistemic, procedural)
        if self.redis_url:
            _model_manager_ref = self.model_manager
            _skill_seq_snapshot = list(_skill_sequence)
            _unique_count = len(_unique_skills_used)
            _round_count = _skill_round_count

            async def _post_message_async():
                try:
                    r = aioredis.from_url(self.redis_url, decode_responses=True)
                    await r.set(LAST_ACTIVE_KEY, str(time.time()))
                    await r.aclose()
                except Exception:
                    pass
                try:
                    await kg_extract(text, response_text, str(chat_id), self.redis_url)
                except Exception:
                    pass
                # Phase 5: extract user-declared stable facts (cat name, color, etc.)
                # On contradiction with stored value: do NOT overwrite. The LLM
                # is then expected (via the [USER ATTRIBUTES] block on next turn)
                # to ask the user instead of silently changing.
                try:
                    from ..memory.user_attributes import extract_declarations, declare_attribute
                    from ..db.session import async_session as _ua_async_session
                    _ua_decls = extract_declarations(text)
                    if _ua_decls:
                        async with _ua_async_session() as _ua_sess:
                            for _ua_key, _ua_value in _ua_decls.items():
                                _status, _existing = await declare_attribute(
                                    _ua_sess, str(chat_id), _ua_key, _ua_value,
                                    source="telegram_message",
                                )
                                if _status == "contradicts":
                                    logger.warning(
                                        "user_attribute.contradiction_detected",
                                        chat_id=str(chat_id), key=_ua_key,
                                        existing=_existing, proposed=_ua_value,
                                    )
                except Exception:
                    logger.exception("user_attribute.extract_failed")
                try:
                    await sm_record_message(self.redis_url)
                except Exception:
                    pass
                try:
                    # Extract temporal observations from the conversation
                    combined_text = f"{text} {response_text}"
                    await temporal_extract(combined_text, source="conversation", chat_id=str(chat_id))
                except Exception:
                    pass
                try:
                    # Update epistemic confidence based on skill outcomes
                    combined_text = f"{text} {response_text}"
                    skill_success = bool(_skill_seq_snapshot) and "ERROR" not in response_text[:100]
                    await epistemic_record(combined_text, skill_success, self.redis_url)
                except Exception:
                    pass
                try:
                    # Abstract procedure if this was a complex multi-step task
                    if _round_count >= 2 and _unique_count >= 2 and _skill_seq_snapshot:
                        from ..memory.procedural import abstract_procedure
                        await abstract_procedure(
                            user_input=text,
                            skill_sequence=_skill_seq_snapshot,
                            final_outcome=response_text[:300],
                            chat_id=str(chat_id),
                            model_manager=_model_manager_ref,
                        )
                except Exception:
                    pass
                try:
                    from ..agent.identity import add_xp
                    await add_xp(1)
                except Exception:
                    pass
                try:
                    # Self-Reflection Engine — heuristic analysis of this execution turn
                    from ..reflection_engine import reflect_on_execution as _reflect_exec
                    _refl_duration = int(time.monotonic() * 1000) - _trace_start_ms if _trace_start_ms else 0
                    _refl_skills   = sorted(_executed_skills) if _executed_skills else []
                    _refl_retries  = sum(1 for s in _trace_spans if s.get("round", 0) > 0)
                    _refl_error    = response_text[:120] if response_text and (
                        "error" in response_text[:60].lower() or
                        "failed" in response_text[:60].lower()
                    ) else ""
                    _refl_success  = not bool(_refl_error) and bool(response_text)
                    await _reflect_exec(
                        intent=text[:150],
                        skills_used=_refl_skills,
                        duration_ms=_refl_duration,
                        success=_refl_success,
                        error=_refl_error,
                        retries=_refl_retries,
                        chat_id=str(chat_id),
                    )
                except Exception:
                    pass

            asyncio.ensure_future(_post_message_async())

        # Learning loop: track last exchange for feedback detection on next message
        # Collect raw skill call text from all rounds for positive example storage
        all_skill_calls_raw = " ".join(
            call.raw_text
            for msg in messages if getattr(msg, 'role', '') == 'assistant'
            for call in (parse_skill_calls(msg.content) if hasattr(msg, 'content') else [])
        ) or response_text
        await self._set_last_exchange(str(chat_id), text, all_skill_calls_raw[:1000])

        # Send response — always publish something; empty response must never be silently dropped
        if not response_text:
            response_text = "Sorry, I could not generate a response. Please try again."
            logger.warning("handler.empty_response_replaced_with_fallback", chat_id=str(chat_id))

        # Structured execution trace — full observability per turn (no dashboard required)
        _exec_duration_ms = int(time.monotonic() * 1000) - _trace_start_ms if _trace_start_ms else 0
        logger.info(
            "execution.trace",
            execution_id=_request_execution_id,
            correlation_id=correlation_id,
            chat_id=str(chat_id),
            skills_executed=sorted(_executed_skills),
            skill_rounds=_skill_round_count,
            auto_skills=[r.skill_name for r in auto_results if r.success],
            validation_status="ok" if _recovered else (_validation.reason if _validation else "unknown"),
            validation_passed=_recovered or (_validation.valid if _validation else True),
            has_retries=any(s.get("round", 0) > 0 for s in _trace_spans),
            duration_ms=_exec_duration_ms,
            is_scheduled=_is_scheduled_trigger,
        )

        # Backstop: ensure something always reaches the user. Empty/whitespace
        # responses get the fallback (Bug #2) — a silent drop is the worst UX.
        if not response_text.strip():
            response_text = (
                "No pude generar una respuesta para esa solicitud. ¿Puedes intentarlo de otra forma?"
                if _user_lang == "es"
                else "I could not generate a response for that. Want me to try a different approach?"
            )
            logger.warning("handler.final_publish_empty_fallback",
                           chat_id=str(chat_id), exec_id=_request_execution_id)
        # Phase 8: route through _safe_publish_response so honesty layer + v2
        # checks (capability override, attribute truth, data grounding) ALWAYS
        # apply. Previously this was a direct bus.publish that bypassed every
        # guard for "normal" turns — exactly the path the LLM uses for plain
        # conversational answers like "Rojo." which contradict stored truth.
        try:
            _final_skill_results = list(_skill_seq_snapshot) if _skill_seq_snapshot else []
        except Exception:
            _final_skill_results = []
        response_text = await self._safe_publish_response(
            text=response_text,
            chat_id=str(chat_id),
            correlation_id=correlation_id,
            skill_results=_final_skill_results,
            user_text=text,
            user_lang=_user_lang,
            outer_trace=_decision_trace,
            reason="main_loop_publish",
            action_intent=_action_intent,
        )
        logger.info(
            "telegram_response_published",
            execution_id=_request_execution_id,
            correlation_id=correlation_id,
            chat_id=str(chat_id),
            chars=len(response_text),
        )

        # Send pending screenshots — separate valid from invalid, group valid into an album.
        _valid_pending = [p for p in pending_photos if os.path.exists(p) and p not in _invalid_photos]
        _invalid_pending = [p for p in pending_photos if os.path.exists(p) and p in _invalid_photos]

        _has_invalid = bool(_invalid_pending)
        if _has_invalid:
            from ..communication.translator import apick as _apick_inv2
            _warn_text2 = await _apick_inv2(
                "invalid_capture_warning", _user_lang or "en",
                (correlation_id or "") + ":invwarn2",
                self.model_manager, self.redis_url,
            )
            await self.bus.publish(self.stream_outgoing, {
                "event_type": EventType.TELEGRAM_RESPONSE,
                "correlation_id": str(uuid4()),
                "chat_id": str(chat_id),
                "text": _warn_text2,
            })

        # Album for valid photos (when 2+)
        _mg_id_post = str(uuid4()) if len(_valid_pending) >= 2 else ""
        for _idx, photo_path in enumerate(_valid_pending):
            payload = {
                "event_type": EventType.TELEGRAM_RESPONSE,
                "correlation_id": str(uuid4()),
                "chat_id": str(chat_id),
                "photo_path": photo_path,
                "text": "",
            }
            if _mg_id_post:
                payload["media_group_id"] = _mg_id_post
                payload["media_group_index"] = str(_idx)
                payload["media_group_total"] = str(len(_valid_pending))
            await self.bus.publish(self.stream_outgoing, payload)

        # At most ONE invalid screenshot, with explicit caption
        if _invalid_pending:
            from ..communication.translator import apick as _apick_cap
            _cap_text2 = await _apick_cap(
                "invalid_capture_caption", _user_lang or "en",
                (correlation_id or "") + ":invcap2",
                self.model_manager, self.redis_url,
            )
            await self.bus.publish(self.stream_outgoing, {
                "event_type": EventType.TELEGRAM_RESPONSE,
                "correlation_id": str(uuid4()),
                "chat_id": str(chat_id),
                "photo_path": _invalid_pending[0],
                "text": _cap_text2,
            })

        # Phase 7: Write authoritative execution outcome back to the scheduled
        # task record. Scheduler now sets last_success=None (unknown). This is
        # the success path; timeout/exception paths call _writeback_task_outcome
        # in the outer handler. last_success = (task produced a real outcome
        # AND no failsafe / hallucination markers).
        if _is_scheduled_trigger and _task_id_for_trace and self.redis_url:
            await self._writeback_task_outcome(
                task_id=_task_id_for_trace,
                task_name=_task_name_for_trace,
                response_text=response_text or "",
                trace_spans=_trace_spans,
                outcome="completed",
            )

        # Cleanup temp media files uploaded by the user
        for _media_path in [image_path, audio_path]:
            if _media_path:
                try:
                    os.unlink(_media_path)
                except Exception:
                    pass

        return response_text, _cognitive_trace

    async def _handle_telegram_command(self, data: dict) -> str:
        text = data.get("text", "")
        chat_id = data.get("chat_id", "")
        correlation_id = data.get("correlation_id", "")
        self._current_chat_id = str(chat_id)
        parts = text.split()
        command = parts[0].lower() if parts else ""

        if command == "/ping":
            response = "pong"
        elif command == "/status":
            stats = self.memory.get_stats()
            snapshots = self.memory.list_snapshots()
            model_status = self.model_manager.get_status()
            skills_count = 0
            skills_enabled = 0
            if self.skill_registry:
                all_skills = self.skill_registry.list_all()
                skills_count = len(all_skills)
                skills_enabled = len(self.skill_registry.list_enabled())
            scheduler_status = "Disabled"
            if self.scheduler:
                jobs = self.scheduler.list_jobs()
                active = sum(1 for j in jobs if not j["paused"])
                scheduler_status = f"{active}/{len(jobs)} active"

            response = (
                "Agent Status\n"
                f"Core: Online\n"
                f"Event Bus: Connected\n"
                f"LLM: {model_status['active_provider']}\n"
                f"Model: {model_status['active_model']}\n"
                f"Skills: {skills_enabled}/{skills_count} enabled\n"
                f"Scheduler: {scheduler_status}\n"
                f"Memory: {stats['total']} entries ({stats['size_bytes']} bytes)\n"
                f"  - Facts: {stats['facts']}\n"
                f"  - Episodic: {stats['episodic']}\n"
                f"  - Semantic: {stats['semantic']}\n"
                f"  - Policy: {stats['policy']}\n"
                f"  - Working: {stats['working']}\n"
                f"  - Meta: {stats['meta']}\n"
                f"Snapshots: {len(snapshots)}\n"
                f"Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}"
            )
        elif command == "/model":
            response = await self._handle_model_command(parts[1:])
        elif command == "/memory":
            response = await self._handle_memory_command(parts[1:])
        elif command == "/snapshot":
            response = await self._handle_snapshot_command(parts[1:])
        elif command == "/api":
            response = await self._handle_api_command(parts[1:])
        elif command == "/broker":
            response = await self._handle_broker_command(parts[1:])
        elif command == "/introspect":
            response = await self._handle_introspect_command()
        elif command == "/schedule":
            response = await self._handle_schedule_command(parts[1:])
        elif command == "/task":
            response = await self._handle_task_command(parts[1:], str(chat_id))
        elif command == "/monitor":
            response = await self._handle_monitor_command(parts[1:], str(chat_id))
        elif command == "/openclaw":
            response = await self._handle_openclaw_command(parts[1:])
        elif command == "/identity":
            response = await self._handle_identity_command(parts[1:], str(chat_id))
        elif command == "/skills":
            response = self._handle_skills_command()
        elif command == "/skill":
            response = self._handle_skill_command(parts[1:])
        elif command == "/help":
            _help_lang = await _get_user_lang(self.redis_url, str(chat_id))
            response = _build_help_text(lang=_help_lang)
        else:
            _unk_lang = await _get_user_lang(self.redis_url, str(chat_id))
            if _unk_lang == "es":
                response = f"Comando desconocido: {command}\nUsa /help para ver los comandos disponibles."
            else:
                response = f"Unknown command: {command}\nUse /help for available commands."

        await self.bus.publish(self.stream_outgoing, {
            "event_type": EventType.TELEGRAM_RESPONSE,
            "correlation_id": correlation_id,
            "chat_id": str(chat_id),
            "text": response,
        })
        return response

    async def _handle_broker_command(self, args: list[str]) -> str:
        if not self.broker_client:
            return "Broker not available."
        if not args:
            return (
                "Broker Commands:\n"
                "/broker status <container> - Container status\n"
                "/broker restart <container> - Restart container\n"
                "/broker logs <container> - Recent logs\n"
                "\nContainers: agent-core, agent-telegram, agent-redis, agent-postgres, agent-ollama"
            )

        subcmd = args[0].lower()
        target = args[1] if len(args) > 1 else ""

        if not target:
            return f"Usage: /broker {subcmd} <container_name>"

        if subcmd == "status":
            result = await self.broker_client.container_status(target)
            if result["success"]:
                return f"Container Status: {target}\n{result['result']}"
            return f"Error: {result.get('error', result.get('result', 'Unknown error'))}"

        elif subcmd == "restart":
            result = await self.broker_client.restart_container(target, requested_by="admin")
            if result["success"]:
                return f"Restart: {result['result']}"
            return f"Error: {result.get('error', result.get('result', 'Unknown error'))}"

        elif subcmd == "logs":
            result = await self.broker_client.container_logs(target, requested_by="admin")
            if result["success"]:
                logs = result["result"]
                # Truncate for Telegram (4096 char limit)
                if len(logs) > 3500:
                    logs = logs[-3500:]
                return f"Logs: {target}\n\n{logs}"
            return f"Error: {result.get('error', result.get('result', 'Unknown error'))}"

        else:
            return f"Unknown broker command: {subcmd}\nUse: status, restart, logs"

    async def _handle_identity_command(self, args: list[str], chat_id: str) -> str:
        """Handle /identity show|set|reset|rollback|versions"""
        if not self.identity_manager:
            return "Identity Engine not available."

        subcmd = args[0].lower() if args else "show"

        # /identity show
        if subcmd == "show":
            prompt = await self.identity_manager.get_prompt()
            compiled = await self.identity_manager.get_compiled()
            versions = await self.identity_manager.list_versions()
            lines = [
                "Agent Identity",
                "─" * 28,
                prompt,
                "",
                "Compiled metadata:",
                f"  Style: {compiled.get('style', '?')}",
                f"  Verbosity: {compiled.get('verbosity', '?')}",
                f"  Autonomy: {compiled.get('autonomy_level', '?')}/10",
                f"  Confirmation: {compiled.get('confirmation_threshold', '?')}",
                f"  Risk tolerance: {compiled.get('risk_tolerance', '?')}",
                f"  Cost-aware: {compiled.get('cost_awareness', False)}",
                f"  Proactive: {compiled.get('proactive', False)}",
                f"  Safety enforced: {compiled.get('safety_enforced', False)}",
                f"  Version: {compiled.get('version', '?')}",
                "",
                f"Saved versions: {len(versions)}",
            ]
            return "\n".join(lines)

        # /identity versions
        elif subcmd == "versions":
            versions = await self.identity_manager.list_versions()
            if not versions:
                return "No saved versions found.\nThe identity hasn't been changed yet."
            lines = [f"Identity Versions ({len(versions)}):"]
            for i, v in enumerate(versions[:10], 1):
                ts = v.get("ts", "?")
                preview = v.get("prompt", "")[:60].replace("\n", " ")
                if len(v.get("prompt", "")) > 60:
                    preview += "…"
                lines.append(f"\n{i}. {ts}\n   {preview}")
            lines.append("\nUse: /identity rollback <timestamp>")
            return "\n".join(lines)

        # /identity reset
        elif subcmd == "reset":
            compiled = await self.identity_manager.reset(source="telegram")
            # Audit log
            async with async_session() as session:
                from uuid import uuid4
                from ..db.models import AuditLog
                audit = AuditLog(
                    id=str(uuid4()),
                    event_type="identity.reset",
                    source="telegram",
                    action="identity_reset",
                    input_summary="reset to default",
                    output_summary="Identity reset to default",
                    user_id="",
                    chat_id=chat_id,
                    latency_ms=0,
                )
                session.add(audit)
                await session.commit()
            return (
                "Identity reset to built-in default.\n\n"
                f"Style: {compiled.get('style')} | "
                f"Autonomy: {compiled.get('autonomy_level')}/10 | "
                f"Confirmation: {compiled.get('confirmation_threshold')}"
            )

        # /identity set <text>
        elif subcmd == "set":
            if len(args) < 2:
                return (
                    "Usage: /identity set <prompt text>\n\n"
                    "Example:\n"
                    "/identity set You are a formal assistant. You always confirm before acting. "
                    "You respond only in English. You prioritize accuracy over speed."
                )
            new_prompt = " ".join(args[1:])
            compiled = await self.identity_manager.save(new_prompt, source="telegram")
            # Audit log
            async with async_session() as session:
                from uuid import uuid4
                from ..db.models import AuditLog
                audit = AuditLog(
                    id=str(uuid4()),
                    event_type="identity.updated",
                    source="telegram",
                    action="identity_set",
                    input_summary=new_prompt[:200],
                    output_summary="Identity updated via Telegram",
                    user_id="",
                    chat_id=chat_id,
                    latency_ms=0,
                )
                session.add(audit)
                await session.commit()
            return (
                "Identity updated.\n\n"
                f"Style: {compiled.get('style')} | "
                f"Autonomy: {compiled.get('autonomy_level')}/10 | "
                f"Confirmation: {compiled.get('confirmation_threshold')} | "
                f"Proactive: {compiled.get('proactive')}"
            )

        # /identity rollback <timestamp>
        elif subcmd == "rollback":
            if len(args) < 2:
                versions = await self.identity_manager.list_versions()
                if not versions:
                    return "No versions to rollback to."
                lines = ["Available versions:"]
                for i, v in enumerate(versions[:10], 1):
                    lines.append(f"{i}. {v.get('ts', '?')}")
                lines.append("\nUsage: /identity rollback <timestamp>")
                return "\n".join(lines)

            ts_str = args[1]
            compiled = await self.identity_manager.rollback(ts_str, source="telegram")
            if compiled is None:
                versions = await self.identity_manager.list_versions()
                available = [v.get("ts", "?") for v in versions[:5]]
                return (
                    f"Version '{ts_str}' not found.\n\n"
                    f"Available: {', '.join(available) if available else 'none'}"
                )
            # Audit log
            async with async_session() as session:
                from uuid import uuid4
                from ..db.models import AuditLog
                audit = AuditLog(
                    id=str(uuid4()),
                    event_type="identity.rollback",
                    source="telegram",
                    action="identity_rollback",
                    input_summary=ts_str,
                    output_summary="Identity rolled back via Telegram",
                    user_id="",
                    chat_id=chat_id,
                    latency_ms=0,
                )
                session.add(audit)
                await session.commit()
            return (
                f"Rolled back to version {ts_str}.\n\n"
                f"Style: {compiled.get('style')} | "
                f"Autonomy: {compiled.get('autonomy_level')}/10"
            )

        else:
            return (
                "Identity Commands:\n"
                "/identity show - View current identity\n"
                "/identity set <text> - Update identity prompt\n"
                "/identity reset - Reset to built-in default\n"
                "/identity rollback <timestamp> - Restore a version\n"
                "/identity versions - List saved versions"
            )

    async def _handle_monitor_command(self, args: list[str], chat_id: str) -> str:
        if not args:
            args = ["list"]

        subcmd = args[0].lower()

        if subcmd == "list":
            async with async_session() as session:
                entries = await self.memory.retrieve(
                    session,
                    MemoryQuery(memory_type=MemoryType.WORKING, tags=["monitor", "active"], limit=20),
                )
            if not entries:
                return "No hay monitores activos.\nCrea uno: \"monitorea <url> cada 1 hora\""

            lines = [f"Monitores activos ({len(entries)}):\n"]
            for i, entry in enumerate(entries, 1):
                url = entry.content.get("url", "?")
                mtype = entry.content.get("monitor_type", "change")
                interval = entry.content.get("interval_minutes", 60)
                checks = entry.content.get("check_count", 0)
                changes = entry.content.get("change_count", 0)
                last = entry.content.get("last_checked_at", "nunca")
                if last and last != "nunca":
                    last = last[:19]
                label = entry.content.get("label", url)
                keyword = entry.content.get("keyword", "")

                line = f"{i}. {label}"
                if keyword:
                    line += f" (\"{keyword}\")"
                line += f"\n   Tipo: {mtype} | Cada {interval}min"
                line += f"\n   Checks: {checks} | Cambios: {changes}"
                line += f"\n   Último: {last} | ID: {entry.id[:8]}"
                lines.append(line)

            return "\n".join(lines)

        elif subcmd == "remove":
            target = " ".join(args[1:]) if len(args) > 1 else ""
            if not target:
                return "Uso: /monitor remove <url o id>"

            async with async_session() as session:
                entries = await self.memory.retrieve(
                    session,
                    MemoryQuery(memory_type=MemoryType.WORKING, tags=["monitor", "active"], limit=50),
                )
                error_entries = await self.memory.retrieve(
                    session,
                    MemoryQuery(memory_type=MemoryType.WORKING, tags=["monitor", "error"], limit=50),
                )
                entries.extend(error_entries)

            found = None
            for entry in entries:
                if entry.id.startswith(target) or target in entry.content.get("url", ""):
                    found = entry
                    break

            if not found:
                return f"Monitor no encontrado: {target}"

            label = found.content.get("label", found.content.get("url", "?"))
            async with async_session() as session:
                await self.memory.delete(session, MemoryType.WORKING, found.id)

            return f"Monitor eliminado: {label}"

        else:
            return (
                "Comandos de Monitor:\n"
                "/monitor list - Monitores activos\n"
                "/monitor remove <url|id> - Eliminar monitor\n"
                "\nCrear: \"monitorea <url> cada 1 hora\""
            )

    async def _handle_openclaw_command(self, args: list[str]) -> str:
        if not args:
            args = ["list"]

        subcmd = args[0].lower()

        if subcmd == "search":
            query = " ".join(args[1:]) if len(args) > 1 else ""
            if not query:
                return "Usage: /openclaw search <query>\nExample: /openclaw search docker"
            client = get_clawhub_client()
            results = await client.search(query, limit=8)
            if not results:
                return f"No skills found for: {query}"
            lines = [f"ClawHub: \"{query}\"\n"]
            for r in results:
                stars = f" ({r['stars']}*)" if r.get("stars") else ""
                ver = f" v{r['version']}" if r.get("version") else ""
                lines.append(f"  {r['slug']}{ver}{stars}")
                if r.get("description"):
                    lines.append(f"    {r['description'][:100]}")
            lines.append(f"\nInstall: /openclaw install <slug>")
            return "\n".join(lines)

        elif subcmd == "install":
            slug = args[1] if len(args) > 1 else ""
            if not slug:
                return "Usage: /openclaw install <slug>"
            skill_dir = get_skills_dir() / slug
            if skill_dir.exists():
                return f"Skill '{slug}' is already installed. Use /openclaw remove {slug} first."
            client = get_clawhub_client()
            skill = await client.download(slug)
            if not skill:
                return f"Failed to download '{slug}'. Check the slug and try again."
            missing = check_requirements(skill)
            warning = f"\nMissing: {', '.join(missing)}" if missing else ""
            return (
                f"Installed: {skill.display_name} (v{skill.version})\n"
                f"{skill.description}\n"
                f"Instructions: {len(skill.instructions)} chars{warning}\n\n"
                f"The skill is now active and its instructions are injected into the agent's context."
            )

        elif subcmd == "list":
            skills = load_installed_skills()
            if not skills:
                return "No OpenClaw skills installed.\nSearch: /openclaw search <query>"
            lines = [f"OpenClaw Skills ({len(skills)}):\n"]
            for s in skills:
                src = "[clawhub]" if s.source == "clawhub" else "[local]"
                lines.append(f"  {s.display_name} v{s.version} {src}")
                if s.description:
                    lines.append(f"    {s.description[:80]}")
            return "\n".join(lines)

        elif subcmd == "remove":
            slug = args[1] if len(args) > 1 else ""
            if not slug:
                return "Usage: /openclaw remove <slug>"
            import shutil
            skill_dir = get_skills_dir() / slug
            if not skill_dir.exists():
                return f"Skill '{slug}' is not installed."
            shutil.rmtree(skill_dir)
            return f"Removed: {slug}"

        else:
            return (
                "OpenClaw Commands:\n"
                "/openclaw search <query> - Search ClawHub\n"
                "/openclaw install <slug> - Install skill\n"
                "/openclaw list - Installed skills\n"
                "/openclaw remove <slug> - Remove skill"
            )

    async def _configure_api_key(self, provider: str, api_key: str) -> str:
        """Store a provider credential in Redis and register the provider."""
        label = PROVIDER_LABELS.get(provider, provider)
        try:
            redis_url = getattr(self.model_manager, "redis_url", None)
            if redis_url:
                r = aioredis.from_url(redis_url, decode_responses=True)
                await r.hset(REDIS_APIKEYS_HASH, provider, api_key)
                await r.aclose()

            result = await self.model_manager.register_provider(provider, api_key)
            if result.get("healthy"):
                models = result.get("models", [])
                return (
                    f"{label} configured successfully!\n"
                    f"Status: Connected\n"
                    f"Models: {', '.join(models)}"
                )
            else:
                return (
                    f"{label} credential saved but health check failed.\n"
                    "The credential may be invalid, expired, or the service is down."
                )
        except Exception as e:
            logger.exception("api_key.configure_failed", provider=provider)
            return f"Error configuring {label}: {e}"

    async def _handle_api_command(self, args: list[str]) -> str:
        if not args:
            args = ["list"]

        subcmd = args[0].lower()

        if subcmd == "list":
            providers = await self.model_manager.get_provider_info()
            lines = ["API Providers:\n"]
            for p in providers:
                label = PROVIDER_LABELS.get(p["name"], p["name"])
                if p["configured"]:
                    status = "Connected" if p["healthy"] else "Unhealthy"
                    models = f" ({len(p['models'])} models)" if p["models"] else ""
                    lines.append(f"  {label}: {status}{models}")
                    lines.append(f"    Credential: {p['masked_key']}")
                else:
                    lines.append(f"  {label}: Not configured")
            lines.append(f"\nUse /api set <provider> <credential> to configure.")
            lines.append(f"Providers: openai, openai-codex, anthropic, google, xai")
            return "\n".join(lines)

        elif subcmd == "set":
            if len(args) < 3:
                return "Usage: /api set <provider> <credential>\nProviders: openai, openai-codex, anthropic, google, xai"
            provider = args[1].lower()
            api_key = args[2]
            if provider not in PROVIDER_LABELS:
                return f"Unknown provider: {provider}\nAvailable: openai, openai-codex, anthropic, google, xai"
            return await self._configure_api_key(provider, api_key)

        elif subcmd == "remove":
            if len(args) < 2:
                return "Usage: /api remove <provider>"
            provider = args[1].lower()
            if provider not in PROVIDER_LABELS:
                return f"Unknown provider: {provider}"
            label = PROVIDER_LABELS[provider]

            redis_url = getattr(self.model_manager, "redis_url", None)
            if redis_url:
                r = aioredis.from_url(redis_url, decode_responses=True)
                await r.hdel(REDIS_APIKEYS_HASH, provider)
                await r.aclose()

            removed = self.model_manager.remove_provider(provider)
            if removed:
                return f"{label} removed."
            return f"{label} was not configured."

        elif subcmd == "test":
            if len(args) < 2:
                return "Usage: /api test <provider>"
            provider = args[1].lower()
            p = self.model_manager.providers.get(provider)
            if not p:
                return f"{PROVIDER_LABELS.get(provider, provider)} not configured."
            healthy = await p.health_check()
            if healthy:
                models = p.available_models()
                return f"{PROVIDER_LABELS.get(provider, provider)}: Healthy ({len(models)} models)"
            return f"{PROVIDER_LABELS.get(provider, provider)}: Health check failed"

        else:
            return (
                "API Commands:\n"
                "/api list - Show configured providers\n"
                "/api set <provider> <credential> - Configure provider credential\n"
                "/api remove <provider> - Remove provider credential\n"
                "/api test <provider> - Test connection\n"
                "\nProviders: openai, openai-codex, anthropic, google, xai\n"
                "\nTip: You can also just paste an API key in chat!"
            )

    async def _handle_introspect_command(self) -> str:
        if not self.introspector:
            return "Introspection not available."
        try:
            return await self.introspector.generate_report()
        except Exception as e:
            logger.exception("introspect.failed")
            return f"Introspection error: {e}"

    async def _handle_model_command(self, args: list[str]) -> str:
        if not args:
            args = ["status"]

        subcmd = args[0].lower()

        if subcmd == "status":
            status = self.model_manager.get_status()
            default_model = status.get("default_model", "")
            lines = [
                "Model Status\n",
                f"Active: {status['active_model']} ({status['active_provider']})",
            ]
            if default_model:
                lines.append(f"Default: {default_model} (restored on restart)")
            lines += [
                f"Fallback: {' -> '.join(status['fallback_order'])}",
                "",
                "Installed models:",
            ]
            for name, info in status["providers"].items():
                for m in info.get("models", []):
                    markers = []
                    if m == status["active_model"]:
                        markers.append("active")
                    if m == default_model:
                        markers.append("default ★")
                    suffix = f" ({', '.join(markers)})" if markers else ""
                    lines.append(f"  {m}{suffix}")
            return "\n".join(lines)

        elif subcmd == "list":
            status = self.model_manager.get_status()
            all_models = []
            for name, info in status["providers"].items():
                for m in info.get("models", []):
                    marker = " (active)" if m == status["active_model"] else ""
                    all_models.append(f"  {m}{marker}")
            if not all_models:
                return "No models installed. Use /model available to see downloadable models."
            return "Installed Models:\n" + "\n".join(all_models)

        elif subcmd == "switch":
            model_name = " ".join(args[1:]) if len(args) > 1 else ""
            if not model_name:
                return "Usage: /model switch <model_name>\nExample: /model switch qwen2.5:1.5b"
            result = await self.model_manager.switch_model(model_name)
            return result

        elif subcmd in ("default", "set-default"):
            model_name = " ".join(args[1:]) if len(args) > 1 else ""
            if not model_name:
                current = self.model_manager.default_model
                return (
                    f"Default model: {current or '(none set)'}\n\n"
                    "Usage: /model default <model_name>\n"
                    "The default model is restored automatically on restart."
                )
            result = await self.model_manager.set_default_model(model_name)
            return result

        elif subcmd == "download":
            model_name = " ".join(args[1:]) if len(args) > 1 else ""
            if not model_name:
                return "Usage: /model download <model_name>\nUse /model available to see options."
            catalog = self.model_manager.get_catalog()
            info = next((m for m in catalog if m["name"] == model_name), None)
            size_info = f" ({info['size']})" if info else ""

            # Launch download in background and notify when done
            chat_id = self._current_chat_id
            asyncio.create_task(self._download_and_notify(model_name, chat_id))
            return f"Downloading {model_name}{size_info}...\nI'll notify you when it's ready."

        elif subcmd == "available":
            catalog = self.model_manager.get_catalog()
            status = self.model_manager.get_status()
            installed = set()
            for info in status["providers"].values():
                installed.update(info.get("models", []))

            lines = ["Available Models:\n"]

            current_cat = ""
            for entry in catalog:
                cat = entry["cat"]
                if cat != current_cat:
                    current_cat = cat
                    lines.append(f"\n-- {cat} --")

                tag = " [installed]" if entry["name"] in installed else ""
                lines.append(f"  {entry['name']} ({entry['size']}) - {entry['desc']}{tag}")

            lines.append(f"\nUse: /model download <name>")
            return "\n".join(lines)

        elif subcmd == "delete":
            model_name = " ".join(args[1:]) if len(args) > 1 else ""
            if not model_name:
                return "Usage: /model delete <model_name>"
            result = await self.model_manager.delete_model(model_name)
            return result

        else:
            return (
                "Model Commands:\n"
                "/model status - Current model info\n"
                "/model list - Installed models\n"
                "/model available - Downloadable models\n"
                "/model switch <name> - Switch active model\n"
                "/model default <name> - Set persistent default model\n"
                "/model download <name> - Download a model\n"
                "/model delete <name> - Delete a model"
            )

    async def _download_and_notify(self, model_name: str, chat_id: str):
        """Download a model in background and send notification when done."""
        try:
            result = await self.model_manager.download_model(model_name)
            text = f"Download complete: {model_name}\n{result}\n\nUse /model switch {model_name} to activate it."
            logger.info("download.complete", model=model_name, chat_id=chat_id)
        except Exception as e:
            text = f"Download failed for {model_name}: {str(e)}"
            logger.error("download.failed", model=model_name, error=str(e))

        try:
            await self.bus.publish(self.stream_outgoing, {
                "event_type": EventType.TELEGRAM_RESPONSE,
                "correlation_id": str(uuid4()),
                "chat_id": chat_id,
                "text": text,
            })
            logger.info("download.notification_sent", model=model_name, chat_id=chat_id)
        except Exception as e:
            logger.error("download.notification_failed", model=model_name, error=str(e))

    async def _handle_memory_command(self, args: list[str]) -> str:
        if not args:
            args = ["stats"]

        subcmd = args[0].lower()

        if subcmd == "stats":
            stats = self.memory.get_stats()
            return (
                "Memory Statistics\n"
                f"Total entries: {stats['total']}\n"
                f"Total size: {stats['size_bytes']} bytes\n\n"
                f"By type:\n"
                f"  Facts: {stats['facts']}\n"
                f"  Working: {stats['working']}\n"
                f"  Episodic: {stats['episodic']}\n"
                f"  Semantic: {stats['semantic']}\n"
                f"  Policy: {stats['policy']}\n"
                f"  Meta: {stats['meta']}"
            )
        elif subcmd == "recent":
            limit = int(args[1]) if len(args) > 1 and args[1].isdigit() else 5
            limit = min(limit, 10)
            query = MemoryQuery(memory_type=MemoryType.EPISODIC, limit=limit)
            async with async_session() as session:
                entries = await self.memory.retrieve(session, query)

            if not entries:
                return "No episodic memories found."

            lines = [f"Recent Memories ({len(entries)}):\n"]
            for entry in entries:
                user_input = entry.content.get("user_input", "")[:60]
                ts = entry.created_at[:19]
                lines.append(f"[{ts}] {user_input}")
            return "\n".join(lines)

        elif subcmd == "search":
            search_text = " ".join(args[1:]) if len(args) > 1 else ""
            if not search_text:
                return "Usage: /memory search <text>"

            query = MemoryQuery(text_search=search_text, limit=5)
            async with async_session() as session:
                entries = await self.memory.retrieve(session, query)

            if not entries:
                return f"No memories found for: {search_text}"

            lines = [f"Search results for \"{search_text}\" ({len(entries)}):\n"]
            for entry in entries:
                lines.append(f"[{entry.memory_type.value}] {entry.summary[:80]}")
            return "\n".join(lines)
        else:
            return "Unknown subcommand. Use: /memory stats|recent|search"

    def _handle_skills_command(self) -> str:
        if not self.skill_registry:
            return "Skills system not available."
        all_skills = self.skill_registry.list_all()
        if not all_skills:
            return "No skills registered."
        lines = [f"Skills ({len(all_skills)}):\n"]
        current_cat = ""
        for defn in sorted(all_skills, key=lambda d: (d.category, d.name)):
            if defn.category != current_cat:
                current_cat = defn.category
                lines.append(f"\n-- {current_cat} --")
            status = "ON" if self.skill_registry.is_enabled(defn.name) else "OFF"
            lines.append(f"  [{status}] {defn.name} - {defn.description}")
        lines.append("\nUse /skill enable|disable <name> to toggle.")
        return "\n".join(lines)

    def _handle_skill_command(self, args: list[str]) -> str:
        if not self.skill_registry:
            return "Skills system not available."
        if len(args) < 2:
            return "Usage: /skill enable|disable <name>"
        action = args[0].lower()
        name = args[1].lower()
        if action == "enable":
            if self.skill_registry.enable(name):
                return f"Skill '{name}' enabled."
            return f"Skill '{name}' not found."
        elif action == "disable":
            if self.skill_registry.disable(name):
                return f"Skill '{name}' disabled."
            return f"Skill '{name}' not found."
        else:
            return "Usage: /skill enable|disable <name>"

    async def _handle_schedule_command(self, args: list[str]) -> str:
        if not self.scheduler:
            return "Scheduler not enabled."

        if not args:
            args = ["list"]

        subcmd = args[0].lower()

        if subcmd == "list":
            jobs = self.scheduler.list_jobs()
            if not jobs:
                return "No scheduled jobs."
            lines = ["Scheduled Jobs:\n"]
            for job in jobs:
                status = "PAUSED" if job["paused"] else "ACTIVE"
                interval = self._format_interval(job["interval_seconds"])
                last = job["last_run"][:19] if job["last_run"] else "never"
                health = "OK" if job["last_success"] else "FAIL"
                lines.append(
                    f"  [{status}] {job['name']} (every {interval})\n"
                    f"    Last: {last} | Result: {health} | Runs: {job['run_count']}"
                )
            return "\n".join(lines)

        elif subcmd == "trigger":
            if len(args) < 2:
                return "Usage: /schedule trigger <job_name>"
            job_name = args[1].lower()
            result = await self.scheduler.trigger(job_name)
            return f"Triggered {job_name}:\n{result}"

        elif subcmd == "pause":
            if len(args) < 2:
                return "Usage: /schedule pause <job_name>"
            job_name = args[1].lower()
            if self.scheduler.pause(job_name):
                return f"Job '{job_name}' paused."
            return f"Job '{job_name}' not found."

        elif subcmd == "resume":
            if len(args) < 2:
                return "Usage: /schedule resume <job_name>"
            job_name = args[1].lower()
            if self.scheduler.resume(job_name):
                return f"Job '{job_name}' resumed."
            return f"Job '{job_name}' not found."

        else:
            return (
                "Schedule Commands:\n"
                "/schedule list - Show all jobs\n"
                "/schedule trigger <job> - Run job now\n"
                "/schedule pause <job> - Pause job\n"
                "/schedule resume <job> - Resume job"
            )

    async def _handle_task_command(self, args: list[str], chat_id: str) -> str:
        """Handle /task command for managing custom scheduled tasks."""
        import redis.asyncio as _aioredis
        from ..scheduler.custom_tasks import (
            delete_task as _del_task, fmt_interval as _fmt_iv,
            get_task_by_name as _get_by_name, list_tasks as _list_tasks,
            make_task as _make_task, next_run_from_now as _next_run,
            parse_interval as _parse_iv, save_task as _save_task,
        )
        from datetime import datetime, timezone as _tz

        from ..config import settings as _settings
        r = _aioredis.from_url(_settings.redis_url, decode_responses=True)
        try:
            subcmd = args[0].lower() if args else "list"

            if subcmd == "list":
                tasks = await _list_tasks(r)
                if not tasks:
                    return "No scheduled tasks. Create one with:\n/task create <name> every <interval>: <instruction>"
                lines = ["Tareas programadas:\n"]
                for t in tasks:
                    status = "ACTIVA" if t.get("enabled", True) else "PAUSADA"
                    iv = _fmt_iv(t["interval_seconds"])
                    last = t["last_run"][:19] if t.get("last_run") else "nunca"
                    next_r = t["next_run"][:19] if t.get("next_run") else "—"
                    lines.append(
                        f"  [{status}] {t['name']} (cada {iv})\n"
                        f"    {t['instruction'][:60]}{'…' if len(t['instruction']) > 60 else ''}\n"
                        f"    Última: {last} | Próxima: {next_r} | Runs: {t['run_count']}"
                    )
                return "\n".join(lines)

            elif subcmd == "create":
                # Format: /task create <name> cada <interval>: <instruction>
                # Parse: "nombre cada 2h: instrucción aquí"
                rest = " ".join(args[1:])
                import re as _re
                m = _re.match(r"(.+?)\s+cada\s+(.+?):\s+(.+)", rest, _re.IGNORECASE)
                if not m:
                    return (
                        "Formato: /task create <nombre> cada <intervalo>: <instrucción>\n"
                        "Ejemplo: /task create BTC diario cada día: Busca el precio de Bitcoin y dímelo"
                    )
                name, interval_text, instruction = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
                seconds = _parse_iv(interval_text)
                if not seconds or seconds < 60:
                    return f"Could not parse interval '{interval_text}'. Try: hourly, 2h, daily, weekly, 30 minutes"

                # ── Idempotency check: prevent duplicate tasks ─────────────
                import difflib as _difflib
                existing_tasks = await _list_tasks(r)
                _name_lower = name.lower()
                for _et in existing_tasks:
                    _et_name_lower = _et["name"].lower()
                    _sim = _difflib.SequenceMatcher(None, _name_lower, _et_name_lower).ratio()
                    if _sim >= 0.8:
                        # Same task — offer to update instead
                        _iv = _fmt_iv(_et["interval_seconds"])
                        _dup_base = t("duplicate_task", _user_lang, name=_et["name"])
                        return (
                            f"{_dup_base} ({t('processing', _user_lang)[:-1]}: {_iv})\n"
                            f"  {_et['instruction'][:80]}\n\n"
                            f"/task edit {_et['name']}\n"
                            f"  instruction: {instruction[:60]}"
                        )

                task = _make_task(name=name, instruction=instruction, interval_seconds=seconds, chat_id=chat_id)
                await _save_task(r, task)
                return (
                    f"✓ Tarea '{name}' creada\n"
                    f"  Intervalo: cada {_fmt_iv(seconds)}\n"
                    f"  Instrucción: {instruction[:80]}\n"
                    f"  Primera ejecución: {task['next_run'][:19]}Z"
                )

            elif subcmd in ("delete", "eliminar", "borrar"):
                name = " ".join(args[1:]).strip()
                if not name:
                    return "Uso: /task delete <nombre>"
                task = await _get_by_name(r, name)
                if not task:
                    return f"Tarea '{name}' no encontrada."
                await _del_task(r, task["task_id"])
                return f"Tarea '{name}' eliminada."

            elif subcmd == "pause":
                name = " ".join(args[1:]).strip()
                if not name:
                    return "Uso: /task pause <nombre>"
                task = await _get_by_name(r, name)
                if not task:
                    return f"Tarea '{name}' no encontrada."
                task["enabled"] = False
                await _save_task(r, task)
                return f"Tarea '{name}' pausada."

            elif subcmd == "resume":
                name = " ".join(args[1:]).strip()
                if not name:
                    return "Uso: /task resume <nombre>"
                task = await _get_by_name(r, name)
                if not task:
                    return f"Tarea '{name}' no encontrada."
                task["enabled"] = True
                await _save_task(r, task)
                return f"Tarea '{name}' reanudada."

            elif subcmd in ("trigger", "ejecutar", "run"):
                name = " ".join(args[1:]).strip()
                if not name:
                    return "Uso: /task trigger <nombre>"
                task = await _get_by_name(r, name)
                if not task:
                    return f"Tarea '{name}' no encontrada."
                task["next_run"] = datetime.now(_tz.utc).isoformat()
                await _save_task(r, task)
                return f"Task '{name}' scheduled to run on the next cycle (< 60s)."

            elif subcmd in ("edit", "update", "modificar", "editar"):
                # /task edit <nombre>                          → muestra estado actual
                # /task edit <nombre> instruction: <nueva>     → cambia instrucción
                # /task edit <nombre> interval: <nuevo>        → cambia intervalo
                # /task edit <nombre> name: <nuevo>            → cambia nombre
                # Se pueden combinar: instruction: ... interval: ...
                import re as _re2
                rest = " ".join(args[1:]).strip()
                if not rest:
                    return (
                        "Uso:\n"
                        "/task edit <nombre>\n"
                        "  instruction: <nueva instrucción>\n"
                        "  interval: <nuevo intervalo>\n"
                        "  name: <nuevo nombre>\n\n"
                        "Ejemplo:\n"
                        "/task edit Precio BTC\n"
                        "  instruction: Navega a https://coingecko.com/en/coins/bitcoin y reporta el precio\n"
                        "  interval: cada 30 minutos"
                    )

                # Split task name from field overrides
                # Fields start with "instruction:", "interval:", "name:" on a new token
                field_pattern = _re2.split(r'\b(instruction:|interval:|name:)', rest, maxsplit=1)
                task_name = field_pattern[0].strip()
                if not task_name:
                    return "Debes indicar el nombre de la tarea. Usa /task list para ver los nombres."

                task = await _get_by_name(r, task_name)
                if not task:
                    return f"Tarea '{task_name}' no encontrada. Usa /task list para ver los nombres."

                # If no fields given, show current task details
                if len(field_pattern) < 3:
                    iv = _fmt_iv(task["interval_seconds"])
                    return (
                        f"Tarea: {task['name']}\n"
                        f"Intervalo: cada {iv}\n"
                        f"Instrucción:\n{task['instruction']}\n\n"
                        f"Para editar:\n"
                        f"/task edit {task['name']}\n"
                        f"  instruction: <nueva instrucción>\n"
                        f"  interval: <nuevo intervalo>\n"
                        f"  name: <nuevo nombre>"
                    )

                # Parse all fields from the rest of the string
                fields_str = field_pattern[1] + field_pattern[2]
                new_instruction = ""
                new_interval = ""
                new_name = ""

                m_instr = _re2.search(r'instruction:\s*(.+?)(?=\s*(?:interval:|name:|$))', fields_str, _re2.DOTALL)
                m_intv  = _re2.search(r'interval:\s*(.+?)(?=\s*(?:instruction:|name:|$))', fields_str, _re2.DOTALL)
                m_name  = _re2.search(r'name:\s*(.+?)(?=\s*(?:instruction:|interval:|$))', fields_str, _re2.DOTALL)

                if m_instr: new_instruction = m_instr.group(1).strip()
                if m_intv:  new_interval    = m_intv.group(1).strip()
                if m_name:  new_name        = m_name.group(1).strip()

                changes = []
                if new_instruction:
                    task["instruction"] = new_instruction
                    changes.append("instrucción actualizada")
                if new_interval:
                    seconds = _parse_iv(new_interval)
                    if not seconds or seconds < 60:
                        return f"Could not parse interval '{new_interval}'. Try: hourly, 2h, daily, 30 minutes"
                    task["interval_seconds"] = seconds
                    from ..scheduler.custom_tasks import next_run_from_now as _nrf
                    task["next_run"] = _nrf(seconds)
                    changes.append(f"intervalo → cada {_fmt_iv(seconds)}")
                if new_name:
                    task["name"] = new_name
                    changes.append(f"nombre → '{new_name}'")

                if not changes:
                    return "Nothing to change. Use instruction:, interval: or name: to specify what to modify."

                await _save_task(r, task)
                return f"Tarea actualizada: {', '.join(changes)}."

            else:
                return (
                    "Comandos de tareas:\n"
                    "/task list — lista todas\n"
                    "/task create <nombre> cada <intervalo>: <instrucción>\n"
                    "/task edit <nombre> — ver / editar tarea\n"
                    "/task trigger <nombre> — ejecutar ahora\n"
                    "/task pause <nombre> — pausar\n"
                    "/task resume <nombre> — reanudar\n"
                    "/task delete <nombre> — eliminar"
                )
        finally:
            await r.aclose()

    @staticmethod
    def _format_interval(seconds: float) -> str:
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            return f"{int(seconds / 60)}m"
        elif seconds < 86400:
            return f"{int(seconds / 3600)}h"
        else:
            return f"{int(seconds / 86400)}d"

    async def _handle_snapshot_command(self, args: list[str]) -> str:
        if not args:
            args = ["list"]

        subcmd = args[0].lower()

        if subcmd == "create":
            label = " ".join(args[1:]) if len(args) > 1 else f"manual-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M')}"
            info = self.memory.create_snapshot(label)
            return (
                f"Snapshot created\n"
                f"ID: {info.id[:8]}...\n"
                f"Label: {info.label}\n"
                f"Entries: {info.entry_count}\n"
                f"Size: {info.size_bytes} bytes"
            )
        elif subcmd == "list":
            snapshots = self.memory.list_snapshots()
            if not snapshots:
                return "No snapshots available. Create one with /snapshot create <label>"

            lines = [f"Snapshots ({len(snapshots)}):\n"]
            for snap in snapshots[:10]:
                lines.append(
                    f"[{snap.created_at[:19]}] {snap.label} "
                    f"({snap.entry_count} entries, {snap.trigger})"
                )
            return "\n".join(lines)
        else:
            return "Unknown subcommand. Use: /snapshot create|list"


    # ─────────────────────────────────────────────────────────────────────────
    # Unified LLM Round Loop
    # Single execution pipeline shared by Telegram and dashboard paths.
    # ─────────────────────────────────────────────────────────────────────────
    async def _run_llm_loop(self, ctx: "_LoopContext") -> str:
        """Unified Execution Engine — single deterministic loop for all entrypoints.

        Both Telegram (handle_message) and Dashboard (chat_direct) paths call
        this method exclusively.  All guards, validation, and state mutation
        happen here.

        Per-round stage order (Phase 2 formalised):
          1. PLAN      — LLM generates response + skill calls
          2. PRE-CHECK — pre_execution_check() validates every skill (fail-closed)
          3. EXECUTE   — execute_batch() runs validated skill calls
          4. EVIDENCE  — Phase 2: validate_against_spec() checks ObjectiveSpec;
                         if evidence missing + rounds remain → guided replan
          5. TERMINAL  — _check_action_terminal_state() detects completion
          6. REPLAN    — inject constrained final prompt or failure hint as needed

        Mutates ctx fields (action_history, trace_spans, skill_round_count,
        etc.) in-place.  Returns final response_text after the drift gate and
        failure synthesis are applied.

        Handles both async progress callbacks (Telegram) and sync ones
        (dashboard SSE) transparently.
        """
        # Unified progress emitter — handles sync and async callbacks
        async def _emit(event: dict) -> None:
            cb = ctx.progress_callback
            if cb is None:
                return
            if asyncio.iscoroutinefunction(cb):
                await cb(event)
            else:
                cb(event)

        # Regex compiled once per call (not inside the loop)
        # Also matches https://data/... in case LLM adds https:// prefix to internal paths
        _SHOT_RE_UNIFIED = re.compile(
            r"(?:https?://)?(/data/(?:shared|chat-uploads|screenshots?)/[^\s\)\"\']+"
            r"\.(?:png|jpg|jpeg|gif|webp|bmp|mp4|webm|mov|mp3|ogg|wav|pdf))",
            re.IGNORECASE,
        )
        _SHOT_RE_SIMPLE = re.compile(
            r"(?:https?://)?(/data/screenshots/screenshot_\d+\.png|/data/shared/screenshot\.png)"
        )
        _SENSITIVE_KEY_RE = re.compile(
            r"password|passwd|token|secret|auth|api_key|apikey|credential",
            re.IGNORECASE,
        )

        response_text = ""
        _image_path = ctx.image_path  # consumed after first round
        MAX_SKILL_ROUNDS = _max_skill_rounds()  # read live from settings each execution
        _spec_replan_count = 0         # Phase 2: guided replan counter
        _spec_replan_max = max(2, min(4, MAX_SKILL_ROUNDS // 2))  # Imp-2: dynamic limit (2-4)
        _spec_terminal_override_done = False  # Imp-1: terminal spec gate fires at most once
        _blocking_state_notified = False      # Imp-4: blocking hint fires at most once

        # ── Phase 4: Inject execution pattern hint (guidance only) ────────────
        # Prepend the hint to the last user message so the LLM sees it as context
        # on the first round.  pre_execution_check still applies to all skill calls.
        if ctx.pattern_hint and ctx.messages:
            try:
                from ..models.types import Message as _Msg4
                _last_user_idx = next(
                    (i for i in range(len(ctx.messages) - 1, -1, -1)
                     if ctx.messages[i].role == "user"),
                    None,
                )
                if _last_user_idx is not None:
                    ctx.messages[_last_user_idx] = _Msg4(
                        role="user",
                        content=ctx.pattern_hint + "\n\n" + ctx.messages[_last_user_idx].content,
                    )
            except Exception:
                pass  # fail-open: hint loss is non-critical

        for round_num in range(MAX_SKILL_ROUNDS):
            # ── Request-scoped budget gate ─────────────────────────────────
            # Stop further LLM/skill rounds once the per-request total round
            # budget is exhausted. Returns the best response so far and lets
            # the caller append an honest "partial" note. Per-loop cap still
            # protects this loop too — this is the cross-loop aggregate cap.
            _budget_allowed, _budget_state = request_budget_consume(
                ctx.execution_id, n=1,
            )
            if not _budget_allowed:
                logger.warning(
                    "llm_loop.budget_stop",
                    round=round_num + 1,
                    used=_budget_state.get("used"),
                    cap=_budget_state.get("cap"),
                    tier=_budget_state.get("tier"),
                    has_response=bool(response_text),
                )
                # Tag the ctx so the post-loop response builder can append the
                # honest "I had to stop early" note.
                ctx.__dict__["budget_stopped"] = True
                ctx.__dict__["budget_stop_state"] = _budget_state
                break

            await _emit({"type": "thinking", "round": round_num + 1})

            request = ModelRequest(messages=ctx.messages, image_path=_image_path)
            _image_path = None  # first-round vision only
            llm_response = await self.model_manager.generate(request)
            response_text = llm_response.content

            if not self.skill_executor:
                break

            skill_calls = parse_skill_calls(response_text)

            # ── Tool Router Sanity Check ───────────────────────────────────────
            # Part 1: Loop-breaker — if previous round already had a tool error,
            # force text-only response (no skill calls allowed this round).
            if ctx.tool_error_flag and skill_calls:
                logger.warning(
                    "agent.tool_error_loop_break",
                    blocked_calls=[c.skill_name for c in skill_calls],
                    round=round_num + 1,
                )
                skill_calls = []
                response_text = strip_skill_calls(response_text)
                break

            skill_calls, _router_errors = sanitize_tool_calls(skill_calls, ctx.text)

            # ── Intent boundary gate (BEFORE tool execution) ───────────────
            # Drops side-effecting skill calls (gmail.send, agent_manager.create,
            # task_manager.create) when the user's current message has no
            # explicit intent and no valid short-context reference for them.
            if skill_calls:
                _gated_calls, _gated_dropped = _filter_inferred_side_effects(
                    skill_calls, ctx.text, ctx_messages=ctx.messages, chat_id=ctx.chat_id,
                )
                if _gated_dropped:
                    # Section 3 — enriched logs with intent label.
                    _missing_recipient_seen = False
                    for _gd_sc, _gd_reason, _gd_intent in _gated_dropped:
                        logger.warning(
                            "intent_gate.blocked",
                            skill=_gd_sc.skill_name,
                            action=str((_gd_sc.arguments or {}).get("action", "")
                                       if isinstance(_gd_sc.arguments, dict) else ""),
                            reason=_gd_reason,
                            intent=_gd_intent,
                            round=round_num + 1,
                            text_preview=(ctx.text or "")[:80],
                        )
                        if _gd_reason == "missing_recipient":
                            _missing_recipient_seen = True
                            ctx.intent_missing_recipient_seen = True
                        if _gd_reason == "missing_content":
                            ctx.intent_missing_content_seen = True

                    # Section 4 — block streak control.
                    ctx.intent_block_streak += 1
                    if ctx.intent_block_streak >= 3:
                        ctx.intent_block_streak_exceeded = True
                        logger.warning(
                            "intent_gate.streak_exceeded",
                            streak=ctx.intent_block_streak,
                            round=round_num + 1,
                        )
                        skill_calls = []
                        break

                    # Inject one-time correction so the LLM stops retrying.
                    _blocked_summary = ", ".join(
                        f"{sc.skill_name}({(sc.arguments or {}).get('action','')})"
                        for sc, _, _ in _gated_dropped
                    )
                    _missing_content_seen = any(
                        _r == "missing_content" for _, _r, _ in _gated_dropped
                    )
                    if _missing_recipient_seen:
                        _correction_msg = (
                            f"[INTENT GATE] You attempted {_blocked_summary} but no recipient "
                            f"is present in the user's current message, the recent context, or the "
                            f"call arguments. Ask the user for the recipient address (one short line) "
                            f"before retrying. Do NOT guess the recipient."
                        )
                    elif _missing_content_seen:
                        _correction_msg = (
                            f"[INTENT GATE] You attempted {_blocked_summary} but the email "
                            f"content (subject and body) is missing or placeholder, and the user "
                            f"did not provide concrete content to send. Ask the user what to send "
                            f"(one short line). Do NOT invent or guess the content."
                        )
                    else:
                        _correction_msg = (
                            f"[INTENT GATE] You attempted {_blocked_summary} but the user's "
                            f"current message does not explicitly request this and no short-context "
                            f"reference applies. Deliver the result inline. Do NOT retry the blocked action."
                        )
                    ctx.messages.append(Message(role="assistant", content=response_text))
                    ctx.messages.append(Message(role="user", content=_correction_msg))
                else:
                    # Reset streak on any clean round so a single later block
                    # doesn't trigger the streak cap unfairly.
                    if ctx.intent_block_streak > 0:
                        ctx.intent_block_streak = 0
                skill_calls = _gated_calls

            if _router_errors and not skill_calls:
                # Record tool error state on ctx for loop-breaker next round
                _first_err = _router_errors[0]
                ctx.tool_error_flag = True
                ctx.last_tool_error = {
                    "type": _first_err.get("error", "unknown"),
                    "tool": _first_err.get("tool", ""),
                }
                # Part 3: Persist short-lived cross-turn anti-retry memory
                if self.redis_url:
                    asyncio.ensure_future(_set_tool_error_memory(
                        self.redis_url, ctx.chat_id, ctx.last_tool_error
                    ))
                # Inject error context so LLM explains to user (not retries)
                _err_lines = "\n".join(
                    f"[TOOL_ROUTER_ERROR] {e['error']}: {e['message']} (hint: {e.get('hint', '')})"
                    for e in _router_errors
                )
                ctx.messages.append(Message(role="assistant", content=response_text))
                ctx.messages.append(Message(
                    role="user",
                    content=(
                        f"The skill call could not be executed:\n{_err_lines}\n\n"
                        "Ask the user clearly for the missing URL. "
                        "Do NOT attempt the tool again — just ask."
                    ),
                ))
                continue

            # ── Planning mode: hard execution block ───────────────────────────
            if ctx.planning_mode and skill_calls:
                logger.warning(
                    "planning_mode.execution_blocked",
                    round=round_num + 1,
                    blocked_skills=[c.skill_name for c in skill_calls],
                )
                response_text = strip_skill_calls(response_text)
                break

            # ── Progress: skills planned ──────────────────────────────────────
            if skill_calls:
                _skill_infos = []
                for _sci in skill_calls:
                    _inf: dict = {"name": _sci.skill_name}
                    _sca = _sci.arguments or {}
                    if _sca.get("action"):
                        _inf["action"] = _sca["action"]
                    if _sca.get("url"):
                        _inf["url"] = _sca["url"]
                    if _sca.get("query"):
                        _inf["query"] = _sca["query"][:80]
                    _skill_infos.append(_inf)
                await _emit({
                    "type": "skills_planned",
                    "skills": [c.skill_name for c in skill_calls],
                    "skill_infos": _skill_infos,
                    "round": round_num + 1,
                })

            if not skill_calls:
                # Exit immediately on "not found / already deleted"
                if _ALREADY_GONE_RE.search(response_text):
                    break

                # Retry promise: LLM said "I'll try" but forgot to include a <skill> tag
                if (not ctx.planning_mode
                        and round_num < MAX_SKILL_ROUNDS - 2
                        and _RETRY_PROMISE_RE.search(response_text)):
                    _skip, _why = _correction_should_skip(ctx, "retry_promise", response_text)
                    if _skip:
                        logger.info("llm.retry_promise_skipped", round=round_num + 1, reason=_why)
                        break
                    logger.info("llm.retry_promise_detected", round=round_num + 1)
                    _correction_record(ctx, "retry_promise", response_text)
                    ctx.messages.append(Message(role="assistant", content=response_text))
                    ctx.messages.append(Message(
                        role="user",
                        content=(
                            "Your response says you're going to try again, but you did NOT include "
                            "any skill call. "
                            "You must include ONE skill call NOW to look up the information. "
                            "Example: <skill>browser(action=\"navigate\", url=\"https://...\"))</skill> "
                            "or <skill>python_exec(code=\"import requests; ...\"))</skill>. "
                            "Include the skill tag directly — do not explain, just do it."
                        ),
                    ))
                    continue

                # Action commitment enforcement: user asked for action, LLM gave text-only response
                if (ctx.action_intent is not None
                        and ctx.action_intent.action_commitment
                        and not ctx.planning_mode
                        and not ctx.action_enforcement_done
                        and not ctx.action_terminal_detected
                        and round_num < MAX_SKILL_ROUNDS - 2):
                    _skip, _why = _correction_should_skip(ctx, "action_enforcement", response_text)
                    if _skip:
                        logger.info("action_commitment.enforcement_skipped", round=round_num + 1, reason=_why)
                        ctx.action_enforcement_done = True  # don't try again
                        break
                    ctx.action_enforcement_done = True
                    _correction_record(ctx, "action_enforcement", response_text)
                    logger.info(
                        "action_commitment.enforcement_triggered",
                        round=round_num + 1,
                        action_type=ctx.action_intent.action_type,
                        target=(ctx.action_intent.action_target or "")[:60],
                    )
                    ctx.messages.append(Message(role="assistant", content=response_text))
                    ctx.messages.append(Message(
                        role="user",
                        content=_build_enforcement_prompt(ctx.action_intent),
                    ))
                    continue

                # Intent completeness engine: verify required outputs are present
                if not ctx.intent_retry_done and not ctx.is_scheduled_trigger:
                    try:
                        from ..validation.intent_engine import IntentParser as _ICE
                        _ice_contract = _ICE.parse(ctx.text)
                        if _ice_contract.required:
                            _ice_result = _ice_contract.check(response_text)
                            if not _ice_result.complete:
                                _skip, _why = _correction_should_skip(ctx, "intent_completeness", response_text)
                                if _skip:
                                    logger.info("intent_engine.completeness_skipped",
                                                round=round_num + 1, reason=_why)
                                    ctx.intent_retry_done = True
                                    break
                                ctx.intent_retry_done = True
                                _correction_record(ctx, "intent_completeness", response_text)
                                logger.info(
                                    "intent_engine.completeness_retry",
                                    missing=[r.key for r in _ice_result.missing],
                                    present=[r.key for r in _ice_result.present],
                                    round=round_num + 1,
                                )
                                await _emit({
                                    "type": "thinking",
                                    "round": round_num + 2,
                                    "note": "Filling missing sections…",
                                })
                                ctx.messages.append(Message(role="assistant", content=response_text))
                                ctx.messages.append(Message(
                                    role="user",
                                    content=_ice_result.correction_prompt,
                                ))
                                continue
                    except Exception as _ice_err:
                        logger.debug("intent_engine.error", error=str(_ice_err)[:80])

                # ── Goal Lock: validate grounding at terminal response ─────────
                # When action_terminal_detected, the constrained final prompt
                # was already injected. Validate the LLM respected it before
                # breaking. If not, inject a hard re-anchor prompt (one retry).
                if (
                    ctx.action_terminal_detected
                    and ctx.action_intent is not None
                    and ctx.action_intent.action_commitment
                    and not ctx.planning_mode
                    and round_num < MAX_SKILL_ROUNDS - 1
                    and response_text
                ):
                    try:
                        from .response_grounder import validate_grounding
                        _gl_ground = validate_grounding(
                            response=response_text,
                            action_type=ctx.action_intent.action_type,
                            results=ctx.action_all_results,
                            domain=getattr(ctx.action_intent, "action_target", "") or "",
                            terminal_success=ctx.action_terminal_success,
                            artifacts=ctx.exec_artifacts,
                            user_text=ctx.text,
                        )
                        if not _gl_ground.is_grounded:
                            _skip, _why = _correction_should_skip(ctx, "goal_lock", response_text)
                            if _skip:
                                logger.info(
                                    "goal_lock.reanchor_skipped",
                                    round=round_num + 1,
                                    reason=_why,
                                )
                                break
                            logger.warning(
                                "goal_lock.drift_detected",
                                reason=_gl_ground.reason,
                                round=round_num + 1,
                                preview=response_text[:80],
                            )
                            _correction_record(ctx, "goal_lock", response_text)
                            ctx.messages.append(Message(role="assistant", content=response_text))
                            ctx.messages.append(Message(
                                role="user",
                                content=(
                                    "⚠️ [GOAL LOCK ACTIVE] Your response drifted from the active task.\n"
                                    "You MUST answer ONLY about the execution you just performed.\n"
                                    + _build_constrained_final_prompt(
                                        ctx.action_intent,
                                        ctx.action_all_results,
                                        ctx.action_terminal_success,
                                        ctx.action_terminal_reason,
                                    )
                                ),
                            ))
                            continue  # One retry with hard re-anchor
                    except Exception:
                        pass
                # ── End Goal Lock ─────────────────────────────────────────────

                # ── Improvement 1: Terminal state must satisfy ObjectiveSpec ──
                # If terminal SUCCESS was detected but ObjectiveSpec evidence is
                # still missing → override the break ONCE and force one more
                # execution round with a targeted prompt.
                #
                # Design constraints:
                #  - Only applies to terminal SUCCESS (not failure — let those break)
                #  - Only fires when spec has required_evidence (no-op otherwise)
                #  - Fires at most ONCE per execution (_spec_terminal_override_done)
                #  - Fail-open: any exception → proceed to break normally
                if (
                    ctx.action_terminal_detected
                    and ctx.action_terminal_success
                    and not _spec_terminal_override_done
                    and ctx.action_intent is not None
                    and getattr(ctx.action_intent, "action_commitment", False)
                    and not ctx.planning_mode
                    and round_num < MAX_SKILL_ROUNDS - 1
                ):
                    _spec_t = getattr(ctx.action_intent, "objective_spec", None)
                    if _spec_t is not None and getattr(_spec_t, "required_evidence", None):
                        try:
                            from .control_layer import validate_against_spec as _vas_t
                            _spec_t_ok, _spec_t_missing = _vas_t(
                                response_text, ctx.action_all_results, _spec_t
                            )
                            if not _spec_t_ok:
                                _spec_terminal_override_done = True
                                _spec_t_obj = getattr(_spec_t, "objective", "") or ""
                                logger.warning(
                                    "phase2.terminal_spec_mismatch",
                                    round=round_num + 1,
                                    missing=_spec_t_missing[:80],
                                    action_type=ctx.action_intent.action_type,
                                )
                                # Improvement 3: explicit observability log for terminal override
                                logger.info(
                                    "terminal_override_due_to_missing_evidence",
                                    objective=_spec_t_obj[:120],
                                    missing_evidence=_spec_t_missing[:200],
                                    action_type=ctx.action_intent.action_type,
                                    round=round_num + 1,
                                )
                                ctx.messages.append(Message(role="assistant", content=response_text))
                                ctx.messages.append(Message(
                                    role="user",
                                    content=(
                                        "⚠️ [OBJECTIVE SPEC — NOT SATISFIED]\n"
                                        + (f"Objective: {_spec_t_obj[:120]}\n"
                                           if _spec_t_obj else "")
                                        + f"Missing evidence: {_spec_t_missing[:200]}\n"
                                        "Task completion was detected, but required evidence "
                                        "was not found in the output. Execute one more action "
                                        "to produce the missing evidence, then give your final answer."
                                    ),
                                ))
                                continue  # Override break — one additional round
                        except Exception:
                            pass  # fail-open: fall through to break
                # ── End Improvement 1 ─────────────────────────────────────────

                break  # No skills and no enforcement prompts — exit loop

            # ── Gmail artifact injection: body + attachments ───────────────────
            # Patch gmail:send calls with render_report output as body, and inject
            # captured screenshots as attachments when none are specified.
            for _sc in skill_calls:
                if _sc.skill_name != "gmail":
                    continue
                _sc_args = _sc.arguments or {}
                if _sc_args.get("action", "").strip().lower() != "send":
                    continue
                # Body injection from render_report (prevents placeholder bodies)
                _known_report = ctx.render_report_outputs.get("email", "")
                if _known_report:
                    _current_body = str(_sc_args.get("body", "")).strip()
                    if _current_body != _known_report:
                        _sc.arguments["body"] = _known_report
                        logger.info(
                            "artifact_registry.gmail_body_injected",
                            body_chars=len(_known_report),
                            round=round_num + 1,
                        )
                # Screenshot attachment injection
                _curr_att = _sc_args.get("attachments", "").strip()
                _known_shots = ctx.exec_artifacts.get("screenshots", [])
                if not _curr_att and _known_shots:
                    _sc.arguments["attachments"] = ",".join(_known_shots)
                    logger.info(
                        "artifact_registry.gmail_injected",
                        count=len(_known_shots),
                        paths=_known_shots,
                        round=round_num + 1,
                    )
                elif _curr_att and _known_shots:
                    _specified = [p.strip() for p in _curr_att.split(",") if p.strip()]
                    _valid_specified = [p for p in _specified if os.path.isfile(p)]
                    if not _valid_specified:
                        logger.warning(
                            "artifact_registry.gmail_path_mismatch",
                            specified=_specified,
                            replacing_with=_known_shots,
                            round=round_num + 1,
                        )
                        _sc.arguments["attachments"] = ",".join(_known_shots)

            # ── Phase 8 Hardened: Pre-Execution Guard (STRICT / FAIL-CLOSED) ──
            # EVERY skill call is validated before execute_batch().
            # FAIL-CLOSED: any failure (missing intent, exception, violation)
            # → inject replan message + continue, NEVER fall through to execution.
            #
            # Execution entry contract: _prex_ok must be True to proceed.
            _prex_ok = False
            try:
                from .control_layer import (
                    pre_execution_check, infer_intent_from_text,
                    _REPLAN_MSG_MISSING_INTENT, _REPLAN_MSG_GUARD_EXCEPTION,
                )
                # Guarantee intent is NEVER None — resolve via inference if needed
                _guard_intent = ctx.action_intent
                _intent_source = "explicit"
                if _guard_intent is None or not getattr(_guard_intent, "action_commitment", False):
                    _guard_intent = infer_intent_from_text(ctx.text, skill_calls)
                    _intent_source = "inferred"
                    logger.info(
                        "pre_execution_guard.intent_inferred",
                        action_type=_guard_intent.action_type,
                        target=_guard_intent.action_target[:60] if _guard_intent.action_target else "",
                        round=round_num + 1,
                    )

                # B1 fix: read last_confirmed_domain to anchor follow-ups
                # without URL to the previously-captured site.
                _last_dom = ""
                if self.redis_url and getattr(ctx, "chat_id", ""):
                    try:
                        _r_lcd = aioredis.from_url(self.redis_url, decode_responses=True)
                        try:
                            _last_dom = (await _r_lcd.get(
                                f"last_confirmed_domain:{ctx.chat_id}"
                            )) or ""
                        finally:
                            await _r_lcd.aclose()
                    except Exception:
                        _last_dom = ""

                _prex = pre_execution_check(
                    skill_calls, _guard_intent,
                    user_text=ctx.text,
                    domain_lock_override=ctx.active_domain_lock,
                    last_confirmed_domain=_last_dom,
                )

                if _prex.blocked:
                    logger.warning(
                        "agent.pre_execution_block",
                        tool=str([c.skill_name for c in skill_calls]),
                        reason=_prex.violation_type,
                        input=ctx.text[:120] if hasattr(ctx, "text") else "",
                        blocked=_prex.blocked_skills,
                        action_type=_guard_intent.action_type,
                        intent_source=_intent_source,
                        round=round_num + 1,
                    )
                    # Phase 3 metrics — bump per violation type.
                    try:
                        from ..observability.truth_metrics import bump as _tm_bump_pre
                        await _tm_bump_pre(self.redis_url, f"pre_exec_block_{_prex.violation_type}")
                        if _prex.violation_type == "url_substitution":
                            await _tm_bump_pre(self.redis_url, "url_substitution_blocked")
                    except Exception:
                        pass
                    ctx.messages.append(Message(role="assistant", content=response_text))
                    ctx.messages.append(Message(role="user", content=_prex.replan_message))
                    continue  # REPLAN — never reaches execute_batch
                else:
                    logger.debug(
                        "pre_execution.allowed",
                        action=str([c.skill_name for c in skill_calls]),
                        intent_source=_intent_source,
                        round=round_num + 1,
                    )
                    _prex_ok = True
                    if _prex.active_domain_lock:
                        _new_lock = _prex.active_domain_lock
                        _cur_confirmed = getattr(ctx.active_domain_lock, "confirmed", True)
                        _new_confirmed = getattr(_new_lock, "confirmed", False)
                        if not ctx.active_domain_lock:
                            # H1 Fix case 1: no lock yet — install it
                            ctx.active_domain_lock = _new_lock
                        elif not _cur_confirmed and _new_confirmed:
                            # H1 Fix case 2: existing lock is provisional, new is confirmed
                            # — upgrade to confirmed so post-round validation is accurate
                            ctx.active_domain_lock = _new_lock
                            logger.info(
                                "domain_lock.provisional_upgraded",
                                domain=_new_lock.serialize() if hasattr(_new_lock, "serialize") else str(_new_lock),
                                round=round_num + 1,
                            )
                        _dl = ctx.active_domain_lock
                        logger.info(
                            "domain_lock.updated_from_skill",
                            domain=_dl.serialize() if hasattr(_dl, "serialize") else str(_dl),
                            mode=getattr(_dl, "mode", ""),
                            source=getattr(_dl, "source", _prex.domain_lock_source),
                            confirmed=getattr(_dl, "confirmed", False),
                        )

            except Exception as _prex_err:
                # FAIL-CLOSED: guard exception → block, never execute blindly
                logger.warning(
                    "pre_execution.block",
                    reason="guard_exception",
                    action=str([c.skill_name for c in skill_calls]),
                    error=str(_prex_err)[:80],
                    round=round_num + 1,
                )
                ctx.messages.append(Message(role="assistant", content=response_text))
                ctx.messages.append(Message(
                    role="user",
                    content=_REPLAN_MSG_GUARD_EXCEPTION,
                ))
                continue  # REPLAN — never reaches execute_batch

            # ── Strict execution entry gate ───────────────────────────────────
            # Belt-and-suspenders: execute_batch is ONLY reached when guard passed.
            if not _prex_ok:
                continue  # should be unreachable, but enforces the contract

            # ── Execute skills ────────────────────────────────────────────────
            ctx.skill_round_count += 1
            logger.info(
                "skills.executing",
                round=round_num + 1,
                count=len(skill_calls),
                skills=[c.skill_name for c in skill_calls],
            )
            results = await self.skill_executor.execute_batch(
                skill_calls,
                user_id=str(ctx.user_id),
                chat_id=str(ctx.chat_id),
                execution_id=ctx.execution_id,
            )

            # Accumulate ALL skill results from this LLM loop for post-response
            # guards (schedule honesty, etc.) that need the full per-turn picture.
            try:
                ctx.loop_skill_results.extend(results)
            except Exception:
                pass

            # Execution trace spans (sanitized args, for validation + telemetry)
            for _si, _sr in enumerate(results):
                _sc_arg = skill_calls[_si].arguments if _si < len(skill_calls) else {}
                _clean_args = {
                    k: ("***" if _SENSITIVE_KEY_RE.search(str(k)) else v)
                    for k, v in (_sc_arg.items() if isinstance(_sc_arg, dict) else {})
                }
                _arg_str = json.dumps(
                    sorted(_sc_arg.items()) if isinstance(_sc_arg, dict) else []
                )
                ctx.trace_spans.append({
                    "skill": _sr.skill_name,
                    "arg_hash": hashlib.md5(_arg_str.encode()).hexdigest()[:8],
                    "args": _clean_args,
                    "success": _sr.success,
                    "latency_ms": getattr(_sr, "execution_ms", 0) or 0,
                    "round": round_num,
                })

            # Progress: skill done
            for _pr in results:
                await _emit({
                    "type": "skill_done",
                    "skill": _pr.skill_name,
                    "success": _pr.success,
                    "ms": _pr.execution_ms,
                })

            # Screenshot + media path detection
            for _ridx_m, result in enumerate(results):
                if result.success:
                    _res_out = result.output or ""
                    # [CAPTURE_VALID: false] means the screenshot is diagnostic only —
                    # always send to user (Telegram) but never include in reports or email.
                    _capture_invalid = "[CAPTURE_VALID: false]" in _res_out
                    for _mp in _SHOT_RE_UNIFIED.findall(_res_out):
                        if os.path.exists(_mp) and _mp not in ctx.media_paths:
                            ctx.media_paths.append(_mp)  # always send to user
                            if _capture_invalid:
                                ctx.invalid_photos.add(_mp)
                    for _sm in _SHOT_RE_SIMPLE.finditer(_res_out):
                        _sp = _sm.group(1)
                        if os.path.exists(_sp):
                            if _sp not in ctx.media_paths:
                                ctx.media_paths.append(_sp)  # always send to user
                            if _capture_invalid:
                                ctx.invalid_photos.add(_sp)
                            else:
                                # Only valid screenshots go into exec_artifacts (reports + email)
                                _shots = ctx.exec_artifacts.setdefault("screenshots", [])
                                if _sp not in _shots:
                                    _shots.append(_sp)
                                    logger.info(
                                        "artifact_registry.screenshot_registered",
                                        path=_sp,
                                        valid=True,
                                        total=len(_shots),
                                        round=round_num + 1,
                                    )
                else:
                    if result.skill_name == "browser":
                        _bargs_m = (skill_calls[_ridx_m].arguments
                                    if _ridx_m < len(skill_calls) else {})
                        if isinstance(_bargs_m, dict) and _bargs_m.get("action") == "capture":
                            ctx.browser_timed_out = True

            # Capture render_report outputs by format
            for _ridx, result in enumerate(results):
                if result.success and result.skill_name == "render_report":
                    _rr_args = skill_calls[_ridx].arguments if _ridx < len(skill_calls) else {}
                    _rr_fmt = (
                        str(_rr_args.get("format", "plain"))
                        if isinstance(_rr_args, dict) else "plain"
                    ).lower()
                    ctx.render_report_outputs[_rr_fmt] = result.output

            # Persist successful browser URL for follow-up messages
            for _ridx, result in enumerate(results):
                if result.success and result.skill_name == "browser":
                    _burl = (
                        (skill_calls[_ridx].arguments if _ridx < len(skill_calls) else {})
                        .get("url", "")
                    )
                    if _burl:
                        self._last_browser_url[str(ctx.chat_id)] = _burl

            # ── Post-execution redirect escape check ──────────────────────────
            # H2 Fix: also run when no lock by deriving a retroactive baseline
            # from the intended navigate URL. Catches redirect escapes that happen
            # before the lock is confirmed (e.g. first navigation redirects away).
            _redirect_lock = ctx.active_domain_lock
            if not _redirect_lock:
                # Derive retroactive lock from the navigate URL in this round's calls
                _nav_url_for_redir = ""
                for _sc_r in skill_calls:
                    if (getattr(_sc_r, "skill_name", "") == "browser"
                            and isinstance(getattr(_sc_r, "arguments", None), dict)
                            and _sc_r.arguments.get("action", "").lower() == "navigate"):
                        _nav_url_for_redir = _sc_r.arguments.get("url", "")
                        break
                if _nav_url_for_redir:
                    try:
                        from .control_layer import _extract_domain as _exd, _build_domain_lock
                        _nav_dom = _exd(_nav_url_for_redir)
                        if _nav_dom:
                            _redirect_lock = _build_domain_lock(
                                [_nav_dom], "skill_call", confirmed=False
                            )
                            logger.debug(
                                "domain_lock.retroactive_for_redirect_check",
                                domain=_nav_dom, round=round_num + 1,
                            )
                    except Exception:
                        pass
            if _redirect_lock:
                try:
                    from .control_layer import check_redirect_escape
                    _redir_violated, _redir_msg = check_redirect_escape(
                        results, _redirect_lock
                    )
                    if _redir_violated:
                        logger.warning(
                            "domain_lock.redirect_escape_blocked",
                            lock=_redirect_lock.serialize()
                            if hasattr(_redirect_lock, "serialize")
                            else str(_redirect_lock),
                            round=round_num + 1,
                        )
                        ctx.messages.append(Message(role="assistant", content=response_text))
                        ctx.messages.append(Message(role="user", content=_redir_msg))
                        continue  # REPLAN — redirect escape blocked
                except Exception as _re_err:
                    logger.warning(
                        "domain_lock.redirect_check_error", error=str(_re_err)[:80]
                    )

            # ── Provisional lock confirmation ─────────────────────────────────
            # After execution, try to confirm a provisional domain lock via:
            # Rule A (repeat consistency) or Rule B (useful evidence).
            # Confirmed lock → sticky; provisional → still replaceable next round.
            if ctx.active_domain_lock and not getattr(ctx.active_domain_lock, "confirmed", True):
                try:
                    from .control_layer import maybe_confirm_lock
                    ctx.active_domain_lock, _confirm_reason = maybe_confirm_lock(
                        ctx.active_domain_lock, results
                    )
                    if _confirm_reason:
                        logger.info(
                            "domain_lock.confirmation_reason",
                            domain=ctx.active_domain_lock.serialize()
                            if hasattr(ctx.active_domain_lock, "serialize")
                            else str(ctx.active_domain_lock),
                            reason=_confirm_reason,
                            round=round_num + 1,
                        )
                except Exception as _cl_err:
                    logger.warning(
                        "domain_lock.confirm_error", error=str(_cl_err)[:80]
                    )

            # ── Terminal state detection + step enforcement ───────────────────
            _committed = (
                ctx.action_intent is not None
                and ctx.action_intent.action_commitment
                and not ctx.planning_mode
            )
            if _committed:
                ctx.action_all_results.extend(results)
            elif not ctx.planning_mode:
                # H5 Fix: also accumulate browser results when action_commitment is False.
                # Without this, P1 gate sees zero results when uncommitted browser skills
                # ran and failed — allowing the LLM to hallucinate outcomes unchecked.
                _browser_results = [r for r in results if r.skill_name == "browser"]
                if _browser_results:
                    ctx.action_all_results.extend(_browser_results)
                    logger.debug(
                        "action_all_results.browser_accumulated_uncommitted",
                        count=len(_browser_results),
                        round=round_num + 1,
                    )

                # Record every browser action for step enforcement
                for _sc_idx, _sr in enumerate(results):
                    if _sr.skill_name == "browser":
                        _sc_ref = skill_calls[_sc_idx] if _sc_idx < len(skill_calls) else None
                        _sc_args_h = (
                            _sc_ref.arguments
                            if _sc_ref and isinstance(getattr(_sc_ref, "arguments", None), dict)
                            else {}
                        ) or {}
                        _full_out = _sr.output or ""
                        _stored_out = _full_out[:4096]
                        # Guarantee critical status markers survive truncation
                        for _mk in ("[TRACK_STATUS:", "[FORM_STATUS:"):
                            if _mk in _full_out and _mk not in _stored_out:
                                for _mkl in _full_out.split("\n"):
                                    if _mk in _mkl:
                                        _stored_out += f"\n{_mkl.strip()}"
                                        break
                        ctx.action_history.append({
                            "action": _sc_args_h.get("action", "unknown"),
                            "url": _sc_args_h.get("url", ""),
                            "selector": (_sc_args_h.get("selector", "") or "")[:80],
                            "text": (_sc_args_h.get("text", "") or "")[:60],
                            "success": _sr.success,
                            "output": _stored_out,
                            "round": round_num,
                        })

                # Plan step synchronization
                if ctx.exec_plan is not None:
                    update_plan_from_action_history(ctx.exec_plan, ctx.action_history)

                if not ctx.action_terminal_detected:
                    _is_term, _term_success, _term_reason = _check_action_terminal_state(
                        ctx.action_intent, ctx.action_all_results
                    )

                    # Step enforcement: block premature success claims
                    if _is_term and _term_success:
                        _steps_ok, _steps_reason = _verify_required_steps(
                            ctx.action_intent.action_type, ctx.action_history
                        )
                        if not _steps_ok:
                            _is_term = False
                            _term_success = False
                            ctx.action_final_prompt_override = _build_missing_step_prompt(
                                ctx.action_intent, _steps_reason, ctx.action_history
                            )
                            logger.warning(
                                "step_enforcement.blocked_premature_success",
                                reason=_steps_reason,
                                action_type=ctx.action_intent.action_type,
                                actions_so_far=[h.get("action") for h in ctx.action_history],
                                round=round_num + 1,
                            )

                    if _is_term:
                        ctx.action_terminal_detected = True
                        ctx.action_terminal_success = _term_success
                        ctx.action_terminal_reason = _term_reason
                        _obj_evidence = (
                            _term_reason.split("objective_confirmed:", 1)[1]
                            if "objective_confirmed:" in _term_reason
                            else _term_reason
                        )
                        _wf_persist_prompt = ""
                        if (
                            _term_success
                            and ctx.action_intent.action_type in (
                                "browser_form_workflow", "browser_web_workflow"
                            )
                            and not any(r.skill_name == "skill_manager"
                                        for r in ctx.action_all_results)
                        ):
                            _gate_pass, _gate_reason = _skill_persist_quality_gate(
                                ctx.action_intent, ctx.action_all_results, _obj_evidence
                            )
                            if _gate_pass:
                                _wf_persist_prompt = _build_workflow_persist_prompt(
                                    ctx.action_intent, ctx.action_all_results, _obj_evidence
                                )
                            else:
                                logger.info(
                                    "skill_persist.gate_blocked",
                                    reason=_gate_reason,
                                    action_type=ctx.action_intent.action_type,
                                )
                        ctx.action_final_prompt_override = (
                            (_wf_persist_prompt + "\n\n" if _wf_persist_prompt else "")
                            + _build_constrained_final_prompt(
                                ctx.action_intent, ctx.action_all_results,
                                _term_success, _term_reason,
                            )
                        )
                        logger.info(
                            "action_terminal.detected",
                            success=_term_success,
                            reason=_term_reason,
                            evidence=_obj_evidence[:60],
                            action_type=ctx.action_intent.action_type,
                            workflow_persist=bool(_wf_persist_prompt),
                            round=round_num + 1,
                        )
            # ── End terminal state detection ──────────────────────────────────

            # Build results text
            results_lines = []
            _browser_failed_urls: list[str] = []
            _capture_failed = False
            for _ridx, result in enumerate(results):
                if result.success:
                    results_lines.append(f"[skill:{result.skill_name}] {result.output}")
                else:
                    results_lines.append(
                        f"[skill:{result.skill_name}] ERROR: {result.error}"
                    )
                    if result.skill_name == "browser":
                        _rargs = (
                            skill_calls[_ridx].arguments if _ridx < len(skill_calls) else {}
                        )
                        _rfurl = _rargs.get("url", "") if isinstance(_rargs, dict) else ""
                        if _rfurl:
                            _browser_failed_urls.append(_rfurl)
                        if isinstance(_rargs, dict) and _rargs.get("action") == "capture":
                            _capture_failed = True
                            ctx.browser_timed_out = True
            results_text = "\n".join(results_lines)
            ctx.last_results_text = results_text  # surface to caller for post-loop use

            # ── Improvement 4: Early blocking state detection ─────────────────
            # Detect patterns that cannot be resolved by retrying (captcha, auth,
            # 403, rate-limit, paywall). When found, inject actionable alternative
            # hint ONCE and force the LLM to switch approach immediately.
            # Fail-open: exceptions or no match → continue normally.
            if (
                not _blocking_state_notified
                and ctx.action_intent is not None
                and getattr(ctx.action_intent, "action_commitment", False)
                and not ctx.planning_mode
                and round_num < MAX_SKILL_ROUNDS - 1
            ):
                _bs_blocked, _bs_hint, _bs_type = _detect_blocking_state(results_text)
                if _bs_blocked:
                    _blocking_state_notified = True
                    if _bs_type == "hard":
                        # HARD block: force immediate alternative approach
                        logger.warning(
                            "phase2.hard_blocking_state_detected",
                            hint=_bs_hint[:80],
                            round=round_num + 1,
                            action_type=ctx.action_intent.action_type,
                        )
                        ctx.messages.append(Message(role="assistant", content=response_text))
                        ctx.messages.append(Message(
                            role="user",
                            content=(
                                f"Skill results:\n{results_text}\n\n"
                                f"⚠️ {_bs_hint}\n"
                                "Do NOT retry the same URL or approach. "
                                "Use the alternative described above RIGHT NOW."
                            ),
                        ))
                        continue  # Force alternative approach immediately
                    else:
                        # SOFT block: log hint, allow loop to continue normally.
                        # The LLM sees the failure in results_text and adapts naturally.
                        logger.info(
                            "phase2.soft_blocking_state_detected",
                            hint=_bs_hint[:80],
                            round=round_num + 1,
                            action_type=ctx.action_intent.action_type,
                        )
            # ── End Improvement 4 ─────────────────────────────────────────────

            # ── Phase 2: ObjectiveSpec evidence gap → guided replan ───────────
            # After execute_batch, check whether cumulative evidence satisfies
            # the ObjectiveSpec.required_evidence. If not, and rounds remain,
            # inject a targeted replan hint telling the LLM exactly what's missing.
            #
            # Design:
            #  - Only fires when spec has required_evidence (no-op otherwise)
            #  - Skips when terminal already detected (terminal prompt takes over)
            #  - Skips when action is NOT committed (informational queries)
            #  - At most _spec_replan_max guided replans (Improvement 2: dynamic)
            #  - Corpus normalized before matching (Improvement 5)
            #  - Hint shows found/missing split (Improvement 3: partial evidence)
            #  - Fail-open: any exception → skip silently (existing flow continues)
            if (
                not ctx.action_terminal_detected
                and round_num < MAX_SKILL_ROUNDS - 2
                and _spec_replan_count < _spec_replan_max
                and ctx.action_intent is not None
                and getattr(ctx.action_intent, "action_commitment", False)
                and not ctx.planning_mode
            ):
                _spec = getattr(ctx.action_intent, "objective_spec", None)
                if _spec is not None:
                    try:
                        from .control_layer import validate_against_spec as _vas
                        # Improvement 5: normalize corpus before matching
                        _spec_ok, _spec_missing = _vas(
                            _normalize_for_spec(results_text),
                            ctx.action_all_results,
                            _spec,
                        )
                        if not _spec_ok:
                            _spec_replan_count += 1
                            _spec_objective = getattr(_spec, "objective", "") or ""
                            # Improvement 3: partial evidence — classify done_when as found/missing
                            _ev_corpus = (
                                _normalize_for_spec(results_text) + " "
                                + _normalize_for_spec(" ".join(
                                    getattr(r, "output", None)
                                    or (isinstance(r, dict) and r.get("output", ""))
                                    or ""
                                    for r in (ctx.action_all_results or [])
                                ))
                            )
                            _dw_all = getattr(_spec, "done_when", []) or []
                            _dw_found_items: list = []
                            _dw_miss_items: list = []
                            _stop_words = frozenset({
                                "that", "from", "with", "this", "have", "been",
                                "when", "will", "their", "there", "into", "some",
                            })
                            for _dw_item in _dw_all:
                                _kws = [
                                    w.lower() for w in re.findall(r'\b\w{4,}\b', _dw_item)
                                    if w.lower() not in _stop_words
                                ][:3]
                                if _kws and any(kw in _ev_corpus for kw in _kws):
                                    _dw_found_items.append(_dw_item)
                                else:
                                    _dw_miss_items.append(_dw_item)

                            _found_lines = (
                                "\n".join(f"  ✓ {d}" for d in _dw_found_items)
                                if _dw_found_items else ""
                            )
                            _miss_lines = (
                                "\n".join(f"  ✗ {d}" for d in _dw_miss_items[:3])
                                if _dw_miss_items else _spec_missing[:200]
                            )

                            _spec_hint = (
                                "⚠️ [OBJECTIVE SPEC] Task not complete — required evidence missing.\n"
                                + (f"Objective: {_spec_objective[:120]}\n" if _spec_objective else "")
                                + (f"\nEvidence found:\n{_found_lines}\n" if _found_lines else "")
                                + f"\nStill missing:\n{_miss_lines}\n"
                                "\nYour NEXT action MUST produce real verifiable evidence "
                                "of task completion. Do not describe navigation steps — "
                                "execute and return the actual output."
                            )
                            logger.info(
                                "phase2.spec_evidence_gap",
                                round=round_num + 1,
                                missing=_spec_missing[:80],
                                replan_count=_spec_replan_count,
                                replan_max=_spec_replan_max,
                                found_count=len(_dw_found_items),
                                miss_count=len(_dw_miss_items),
                                action_type=ctx.action_intent.action_type,
                            )
                            ctx.messages.append(Message(role="assistant", content=response_text))
                            ctx.messages.append(Message(
                                role="user",
                                content=f"Skill results:\n{results_text}\n\n{_spec_hint}",
                            ))
                            continue  # Guided replan — LLM must produce evidence
                    except Exception as _p2_err:
                        logger.debug(
                            "phase2.spec_check_error", error=str(_p2_err)[:80]
                        )
            # ── End Phase 2 evidence gap check ───────────────────────────────

            # Browser alternative URL hint
            _browser_alt_hint = ""
            if _capture_failed:
                _browser_alt_hint = (
                    "\n\n⚠️ SCREENSHOT FAILED: The browser could not load the page "
                    "(anti-bot protection or timeout). "
                    "IMPORTANT: Tell the user directly that you could not take the screenshot. "
                    "DO NOT describe page contents. DO NOT invent or hallucinate any data. "
                    "Just say: No pude tomar la captura de pantalla — "
                    "el sitio bloqueó el acceso automático."
                )
            elif _browser_failed_urls:
                _browser_alt_hint = (
                    "\n\n⚠️ BROWSER TIMEOUT: The URL(s) you tried are blocked or timed out: "
                    + ", ".join(_browser_failed_urls)
                    + "\nDO NOT retry these. CoinGecko, Coinbase, and Binance block headless browsers.\n"
                    "Use ONLY these confirmed working sites:\n"
                    "• https://coinmarketcap.com/currencies/bitcoin/ (or ethereum, solana, etc.)\n"
                    "• https://finance.yahoo.com/quote/BTC-USD/ (or ETH-USD, SOL-USD, etc.)\n"
                    "Try browser(action=\"capture\", url=\"https://coinmarketcap.com/currencies/bitcoin/\") NOW."
                )

            # Completed-actions deduplication hint
            _done_actions = sorted({
                f"{s.get('skill')}({(s.get('args') or {}).get('action', '')})"
                for s in ctx.trace_spans[:round_num + 1]
                if s.get("success") and s.get("skill")
            })
            _done_line = (
                f"ALREADY EXECUTED THIS TURN (do NOT repeat): {', '.join(_done_actions)}\n"
                if _done_actions else ""
            )

            # ── Hard wall-clock cap ───────────────────────────────────────────
            _elapsed = time.monotonic() - ctx.start_time
            if _elapsed > MAX_CHAT_SECONDS and round_num < MAX_SKILL_ROUNDS - 1:
                logger.warning(
                    "llm_loop.timeout_forced",
                    elapsed_s=int(_elapsed),
                    round=round_num + 1,
                )
                await _emit({"type": "timeout", "elapsed_s": int(_elapsed)})
                ctx.messages.append(Message(role="assistant", content=response_text))
                ctx.messages.append(Message(
                    role="user",
                    content=(
                        f"Skill results:\n{results_text}\n\n"
                        "[TIME LIMIT REACHED — provide your final answer NOW using what you have. "
                        "Be concise and direct. No more skill calls.]"
                    ),
                ))
                _final_llm = await self.model_manager.generate(
                    ModelRequest(messages=ctx.messages)
                )
                response_text = _final_llm.content
                break

            # Append response + build continuation prompt
            ctx.messages.append(Message(role="assistant", content=response_text))

            if ctx.action_final_prompt_override:
                # Terminal state detected — inject constrained final formatting prompt
                ctx.messages.append(Message(
                    role="user",
                    content=ctx.action_final_prompt_override,
                ))
                ctx.action_final_prompt_override = ""  # consume once
            elif round_num < MAX_SKILL_ROUNDS - 2:
                _all_failed = len(results) > 0 and not any(r.success for r in results)
                if _all_failed:
                    # ALL skills failed — force retry with alternative approach
                    ctx.messages.append(Message(
                        role="user",
                        content=(
                            f"Skill results:\n{results_text}{_browser_alt_hint}\n\n"
                            "The skill failed. DO NOT give up. Try IMMEDIATELY with a different approach:\n"
                            "- Different URL (alternative site with same info)\n"
                            "- python_exec with requests to bypass bot protection\n"
                            "- web_search to find the information another way\n"
                            "Try RIGHT NOW. Include a <skill> tag in your response. Do not explain — just do it."
                        ),
                    ))
                else:
                    _artifact_hint = ""
                    _shots_now = ctx.exec_artifacts.get("screenshots", [])
                    if _shots_now:
                        _shot_list = "\n".join(f"  • {p}" for p in _shots_now)
                        _artifact_hint = (
                            f"\n\nARTIFACT STATE — screenshots captured this turn ({len(_shots_now)}):\n"
                            f"{_shot_list}\n"
                            "Use these EXACT paths when attaching to gmail. Do NOT invent new paths."
                        )
                    _still_needs_shot = (
                        ctx.wants_screenshot
                        and not any("Screenshot saved to" in (r.output or "") for r in results)
                    )
                    _screenshot_continue = (
                        "\n\nSTILL MISSING: The user asked for a SCREENSHOT IMAGE. "
                        "You have NOT called browser(action=\"capture\") yet. "
                        "Call it NOW before sending any final answer."
                        if _still_needs_shot else
                        (" If a screenshot was saved, include it as: ![desc](path)"
                         if ctx.wants_screenshot else "")
                    )
                    ctx.messages.append(Message(
                        role="user",
                        content=(
                            f"Skill results:\n{results_text}{_browser_alt_hint}\n\n"
                            f"{_done_line}"
                            "COMPLETION CHECK — before responding, verify:\n"
                            "• Did you fetch ALL required data? (prices, statistics, content)\n"
                            "• Did you take ALL required screenshots?\n"
                            "• Did you send the email via gmail(action='send', ...)? "
                            "(send_check alone does NOT send)\n"
                            "• Did you create the task/reminder (if requested)?\n"
                            "PARAMETER LOCK: use EXACTLY the recipient, amounts, and names from "
                            "the user's original message. Do NOT substitute or modify them.\n"
                            f"{_artifact_hint}\n"
                            "If ANY item above is still pending → call the skill NOW. "
                            "Do NOT respond until everything is complete. "
                            "STRICT EXECUTION TRUTH — NEVER use future-intent narration:\n"
                            "  ✗ 'Voy a tomar una captura' → take it first, then say 'Tomé la captura'\n"
                            "  ✗ 'Voy a enviar el correo ahora' → send it first, "
                            "then say 'Te envié el correo'\n"
                            "  ✗ 'Procederé a generar el informe' → generate it first, then confirm\n"
                            "  ✓ Only report COMPLETED actions. Only send ONE final response.\n"
                            "NEVER respond with just 'Página abierta' or 'Listo' — "
                            "complete the full request."
                            + _screenshot_continue
                        ),
                    ))
            else:
                # Last rounds — finalize
                ctx.messages.append(Message(
                    role="user",
                    content=(
                        f"Skill results:\n{results_text}\n\n"
                        "Now provide your final answer to the user based on these results. "
                        "IMPORTANT: Format the answer as plain human-readable text. "
                        "NEVER send raw JSON to the user. Extract the relevant value and present it clearly. "
                        "Do not use <skill> tags again."
                        + (" If a screenshot was saved, include its EXACT path in markdown: "
                           "![description](path)"
                           if ctx.wants_screenshot else "")
                    ),
                ))

        # ── Post-loop: Domain Drift Gate ──────────────────────────────────────
        # Active when: action was committed OR recent messages contain browser context.
        # Prevents the LLM from hallucinating weather/crypto/unrelated responses
        # even when the current turn has no explicit action commitment.
        _recent_browser_ctx = any(
            "[TRACK_STATUS:" in (m.content or "")
            or "[Step 1] Navigating" in (m.content or "")
            or "[FORM_STATUS:" in (m.content or "")
            or "[skill:browser]" in (m.content or "")
            for m in ctx.messages[-8:]
        )
        _has_commitment = (
            ctx.action_intent is not None and ctx.action_intent.action_commitment
        )
        if ((_has_commitment or _recent_browser_ctx)
                and not ctx.planning_mode
                and response_text
                and _DRIFT_DURING_ACTION_RE.search(response_text)):
            _drift_diag = (
                _extract_failure_diagnostic(ctx.action_all_results)
                if ctx.action_all_results else ""
            )
            _drift_msg = _drift_diag or "La tarea no pudo completarse."
            logger.warning(
                "action_drift.blocked_post_loop",
                action_type=(ctx.action_intent.action_type
                             if ctx.action_intent else "unknown"),
                preview=response_text[:100],
            )
            response_text = (
                f"No pude completar la tarea solicitada. {_drift_msg} "
                "¿Quieres que lo intente de nuevo?"
            )
        # ── End Domain Drift Gate ─────────────────────────────────────────────

        # ── Post-loop: Failure Synthesis ──────────────────────────────────────
        # When MAX_SKILL_ROUNDS exhausts without terminal state, the LLM's last
        # response may be generic ("Lo siento, no pude...").
        # Replace with a system-generated diagnostic that includes what was
        # attempted, the specific failure reason, and a concrete suggestion.
        if (
            ctx.action_intent is not None
            and ctx.action_intent.action_commitment
            and not ctx.planning_mode
            and not ctx.action_terminal_detected
            and ctx.action_all_results
            and (not response_text.strip() or len(response_text.strip()) < 60)
        ):
            _diag = _extract_failure_diagnostic(ctx.action_all_results)
            _type_label = {
                "browser_package_check": "rastrear el paquete",
                "browser_form_workflow": "completar el formulario",
                "browser_web_workflow": "completar la tarea web",
                "browser_navigation": "navegar al sitio",
            }.get(ctx.action_intent.action_type, "completar la tarea")
            _site = (ctx.action_intent.action_target or "el sitio")[:60]
            _synth = (
                f"No pude {_type_label} en {_site} "
                f"después de {ctx.skill_round_count} intentos."
            )
            if _diag:
                _synth += f" {_diag}"
            else:
                _synth += (
                    " El sitio no devolvió resultados reconocibles "
                    "(posible protección anti-bot o carga JavaScript tardía)."
                )
            _synth += " ¿Quieres que lo intente con un método alternativo?"
            response_text = _synth
            logger.info(
                "post_loop.failure_synthesis",
                action_type=ctx.action_intent.action_type,
                rounds=ctx.skill_round_count,
                diagnostic=_diag[:80] if _diag else "",
            )
        # ── End Failure Synthesis ─────────────────────────────────────────────

        # ── Priority 1: Symmetric Failure-Path Gate ───────────────────────────
        # Fires when execution was attempted but terminal success was NOT achieved.
        # Validates failure-path responses regardless of response length —
        # length is NEVER a reason to skip validation on the failure path.
        # Symmetric to Phase 8, which covers the success path.
        # Gate: results exist AND terminal success was NOT achieved.
        # Does NOT require action_intent — execution results are sufficient signal.
        _fp_active = (
            len(ctx.action_all_results) > 0
            and not ctx.action_terminal_success
            and not ctx.planning_mode
            and response_text
        )
        if _fp_active:
            _fp_action_type = (
                ctx.action_intent.action_type if ctx.action_intent else ""
            )
            logger.info(
                "failure_path.gate_entered",
                action_type=_fp_action_type or "unknown",
                results_count=len(ctx.action_all_results),
                response_length=len(response_text),
            )
            try:
                from .control_layer import enforce_failure_path_contract
                _fp_ref_id = (
                    getattr(ctx.action_intent, "tracking_code", None)
                    or getattr(ctx.action_intent, "action_target", None)
                    or ""
                ) if ctx.action_intent else ""
                _fp_result = enforce_failure_path_contract(
                    response=response_text,
                    action_type=_fp_action_type,
                    results=ctx.action_all_results,
                    artifacts=ctx.exec_artifacts,
                    ref_id=_fp_ref_id,
                    user_text=ctx.text,
                )
                if not _fp_result.approved:
                    response_text = _fp_result.final_response
            except Exception as _fp_err:
                logger.warning("failure_path.gate_error", error=str(_fp_err)[:80])
        # ── End Failure-Path Gate ─────────────────────────────────────────────

        # ── Phase 8: Universal Response Control Layer ────────────────────────
        # Replaces Phase 7 single-type grounding with full multi-layer validation:
        #   Layer A: Semantic grounding (response_grounder — extended to all types)
        #   Layer B: Keyword consistency (action_type markers)
        #   Layer C: Evidence contract + context integrity + hallucination detection
        # Any layer failure → deterministic fallback or honest failure message.
        if (
            ctx.action_terminal_detected
            and ctx.action_intent is not None
            and ctx.action_intent.action_commitment
            and not ctx.planning_mode
            and response_text
        ):
            _p8_ref_id = (
                getattr(ctx.action_intent, "tracking_code", None)
                or getattr(ctx.action_intent, "action_target", None)
                or ""
            )

            # Layer A: Semantic grounding (response_grounder)
            _p8_failed = False
            _p8_reason = ""
            try:
                from .response_grounder import validate_grounding
                _p8_ground = validate_grounding(
                    response=response_text,
                    action_type=ctx.action_intent.action_type,
                    results=ctx.action_all_results,
                    domain=getattr(ctx.action_intent, "action_target", "") or "",
                    terminal_success=ctx.action_terminal_success,
                    artifacts=ctx.exec_artifacts,
                    user_text=ctx.text,
                )
                if not _p8_ground.is_grounded:
                    _p8_failed = True
                    _p8_reason = _p8_ground.reason
            except Exception as _p8_err:
                logger.warning("phase8.grounding_check_error", error=str(_p8_err)[:80])

            # Layer B: Keyword consistency
            if not _p8_failed:
                try:
                    _p8_consist, _p8_consist_reason = _check_response_consistency(
                        response_text,
                        ctx.action_intent.action_type,
                        ctx.action_terminal_success,
                    )
                    if not _p8_consist:
                        _p8_failed = True
                        _p8_reason = _p8_consist_reason
                except Exception as _p8b_err:
                    logger.warning("phase8.consistency_check_error", error=str(_p8b_err)[:80])

            # Layer C: Evidence contract + context integrity + hallucination
            if not _p8_failed:
                try:
                    from .control_layer import enforce_response_contract
                    _p8_ctrl = enforce_response_contract(
                        response=response_text,
                        action_type=ctx.action_intent.action_type,
                        results=ctx.action_all_results,
                        artifacts=ctx.exec_artifacts,
                        terminal_success=ctx.action_terminal_success,
                        user_text=ctx.text,
                        ref_id=_p8_ref_id,
                    )
                    if not _p8_ctrl.approved:
                        _p8_failed = True
                        _p8_reason = _p8_ctrl.reason
                        # Control layer already built the fallback — use it
                        if _p8_ctrl.fallback_used and _p8_ctrl.final_response:
                            response_text = _p8_ctrl.final_response
                            logger.info(
                                "phase8.control_layer_fallback",
                                reason=_p8_reason,
                                action_type=ctx.action_intent.action_type,
                                length=len(response_text),
                            )
                            _p8_failed = False  # fallback already applied
                except Exception as _p8c_err:
                    logger.warning("phase8.control_layer_error", error=str(_p8c_err)[:80])

            # If any layer failed and no fallback was applied, build deterministic response
            if _p8_failed:
                logger.warning(
                    "phase8.response_rejected",
                    reason=_p8_reason,
                    action_type=ctx.action_intent.action_type,
                    terminal_success=ctx.action_terminal_success,
                    preview=response_text[:100],
                )
                try:
                    from .response_grounder import build_deterministic_response
                    _p8_det = build_deterministic_response(
                        results=ctx.action_all_results,
                        action_type=ctx.action_intent.action_type,
                        ref_id=_p8_ref_id,
                        artifacts=ctx.exec_artifacts,
                        terminal_success=ctx.action_terminal_success,
                    )
                    if _p8_det:
                        response_text = _p8_det
                        logger.info(
                            "phase8.deterministic_fallback_used",
                            action_type=ctx.action_intent.action_type,
                            reason=_p8_reason,
                            length=len(response_text),
                        )
                except Exception as _p8_det_err:
                    logger.warning("phase8.det_build_failed", error=str(_p8_det_err)[:80])
        # ── End Phase 8 Universal Control Layer ──────────────────────────────

        # ── C3 Fix: Exhaustion-Path Guard ────────────────────────────────────
        # Fires when execution was attempted (action_commitment=True) but
        # terminal state was NEVER detected (round exhaustion / timeout).
        # Phase 8 requires terminal detection — it never fires here.
        # Priority 1 covers the case with results, but not empty-results exhaustion.
        # This guard blocks hallucinated completion claims on any exhaustion exit.
        if (
            not ctx.action_terminal_detected
            and ctx.action_intent is not None
            and ctx.action_intent.action_commitment
            and not ctx.planning_mode
            and response_text
        ):
            try:
                from .control_layer import enforce_exhaustion_path_contract
                _ex_result = enforce_exhaustion_path_contract(
                    response=response_text,
                    action_type=ctx.action_intent.action_type,
                    results=ctx.action_all_results,
                    user_text=ctx.text,
                )
                if not _ex_result.approved:
                    logger.warning(
                        "exhaustion_path.gate_blocked",
                        reason=_ex_result.reason,
                        action_type=ctx.action_intent.action_type,
                        preview=response_text[:80],
                    )
                    response_text = _ex_result.final_response
            except Exception as _ex_err:
                logger.warning("exhaustion_path.gate_error", error=str(_ex_err)[:80])
        # ── End Exhaustion-Path Guard ─────────────────────────────────────────

        # ── Priority 3: Success-Path Sufficiency Gate ─────────────────────────
        # Fires ONLY on terminal_success=True (execution claimed to succeed).
        # Checks: sufficiency, weak language, strong unsupported claims, entity grounding.
        # Runs AFTER Phase 8 — does not modify failure-path or domain lock.
        if (
            ctx.action_terminal_success
            and ctx.action_intent is not None
            and ctx.action_intent.action_commitment
            and not ctx.planning_mode
            and response_text
        ):
            try:
                from .control_layer import enforce_success_path_contract
                _sp_ref_id = (
                    getattr(ctx.action_intent, "tracking_code", None)
                    or getattr(ctx.action_intent, "action_target", None)
                    or ""
                )
                _sp_result = enforce_success_path_contract(
                    response=response_text,
                    action_type=ctx.action_intent.action_type,
                    results=ctx.action_all_results,
                    artifacts=ctx.exec_artifacts,
                    user_text=ctx.text,
                    ref_id=_sp_ref_id,
                    objective_spec=getattr(ctx.action_intent, "objective_spec", None),
                )
                if not _sp_result.approved:
                    logger.warning(
                        "success_path.gate_blocked",
                        reason=_sp_result.reason,
                        action_type=ctx.action_intent.action_type,
                        preview=response_text[:80],
                    )
                    response_text = _sp_result.final_response
            except Exception as _sp_err:
                logger.warning("success_path.gate_error", error=str(_sp_err)[:80])
        # ── End Priority 3 Success-Path Gate ─────────────────────────────────

        return response_text


    async def chat_direct(self, text: str, user_id: str = "dashboard", progress_callback=None, image_path: str | None = None) -> str:
        """Process a message directly (no Telegram bus) — for dashboard chat.

        Runs the full agent pipeline: skill auto-detect → context build → LLM loop.
        Returns the final response text.
        progress_callback: optional callable(dict) called with progress events during execution.
        image_path: optional path to an image file for vision-capable models.
        """
        if not self.model_manager.active_model:
            return (
                "No model is currently active.\n"
                "Use /model download <name> to install a model, "
                "or configure an API key in /models."
            )

        chat_id = "dashboard"

        # ── Boot sequence: fires exactly once after a panic reset ─────────────
        _boot_msg = await self._run_boot_sequence(chat_id, first_text=text)
        if _boot_msg:
            return _boot_msg

        # ── Fast-path auto-detects (mirrors Telegram handle() shortcuts) ──────
        # API key detection
        detected_api = _detect_api_key(text)
        if detected_api:
            provider, api_key = detected_api
            return await self._configure_api_key(provider, api_key)

        # Model list / status
        _MODEL_LIST_RE = re.compile(
            r"\b(?:que|qué|cuáles?|cuales?|cuantos?|cuántos?|what|which)\s+(?:modelos?|llms?|cerebros?|models?)\s+(?:tienes?|hay|están?|estan?|puedo|disponibles?|instalados?|do you have|are available)\b",
            re.IGNORECASE,
        )
        _MODEL_STATUS_RE = re.compile(
            r"\b(?:que|qué|cual|cuál|what|which)\s+(?:modelo|llm|cerebro|model)\s+(?:usas?|tienes?|estás?\s+usando|estas?\s+usando|es|are you using|do you use)\b",
            re.IGNORECASE,
        )
        if _MODEL_LIST_RE.search(text):
            return await self._handle_model_command(["list"])
        if _MODEL_STATUS_RE.search(text):
            return await self._handle_model_command(["status"])

        # Model switch
        model_request = _detect_model_switch(text)
        if model_request:
            status = self.model_manager.get_status()
            all_models: list[str] = []
            for pinfo in status["providers"].values():
                all_models.extend(pinfo.get("models", []))
            matched = _match_model_name(model_request, all_models)
            if matched:
                await self.model_manager.switch_model(matched)
                new_provider = self.model_manager.active_provider
                return f"✅ Switched to **{matched}** ({new_provider}). Ready."
            else:
                clean = model_request.lower().strip()
                for canon, aliases in _MODEL_ALIASES.items():
                    if any(a in clean for a in aliases) or clean in canon:
                        return (
                            f"Model {canon} is not installed/configured.\n"
                            f"Available: {', '.join(all_models) if all_models else 'none'}\n"
                            "Configure an API key in /models or download a local model."
                        )
                return (
                    f"Model '{model_request}' not found.\n"
                    f"Available: {', '.join(all_models) if all_models else 'none'}"
                )
        # ── End fast-path ─────────────────────────────────────────────────────

        # Pre-LLM fast-path: complex multi-step directives → goal engine (same as Telegram)
        # Skip task-setup requests so the LLM can execute immediately + create recurring task
        if (
            len(text) > 500
            and self.goal_orchestrator is not None
            and _COMPLEX_DIRECTIVE_RE.search(text)
            and not text.startswith("[TAREA PROGRAMADA:")
            and not _TASK_SETUP_RE.search(text)
        ):
            try:
                _goal = await self.goal_orchestrator.create_goal(
                    objective=text,
                    chat_id=chat_id,
                    user_id=user_id,
                    priority=8,
                    source="dashboard",
                )
                logger.info("chat_direct.complex_directive_fast_path", goal_id=_goal.id)
                _fp_response = (
                    f"✅ Objetivo registrado y en ejecución (ID: {_goal.id[:8]}).\n"
                    f"Te notificaré cuando esté listo."
                )
                # Store episodic memory so follow-up messages have context
                try:
                    async with async_session() as _fpm_session:
                        await self.memory.store_episodic(
                            _fpm_session,
                            event_type="dashboard.message",
                            user_input=text,
                            agent_response=_fp_response,
                            user_id=user_id,
                            chat_id=chat_id,
                        )
                except Exception:
                    pass
                return _fp_response
            except Exception as _gex:
                logger.warning("chat_direct.complex_directive_goal_failed", error=str(_gex))
                # Fall through to normal LLM flow

        # ── AGENT WAKEUP (dashboard / direct path) ───────────────────────────
        if text.startswith("[AGENT_WAKEUP:") and self._agent_orchestrator:
            text = await _resolve_agent_wakeup(text, self._agent_orchestrator)

        # Long messages always go to LLM — auto-detects only for short direct commands
        _short_msg = len(text) <= 400

        # Auto-detect "run agent now" — ejecutalo ahora / run it now
        # Skip for scheduled task triggers — "EJECUTA AHORA" is an internal directive, not user intent.
        _is_scheduled_trigger = text.startswith("[TAREA PROGRAMADA:")
        if _detect_agent_run_now(text) and self.skill_executor and not _is_scheduled_trigger:
            from ..skills.types import SkillCall as _SC
            _run_call_d = _SC(
                skill_name="agent_manager",
                arguments={"action": "run_now"},
                raw_text="[auto-detected: run agent now]",
            )
            _run_results_d = await self.skill_executor.execute_batch(
                [_run_call_d], user_id=user_id, chat_id=chat_id,
                execution_id=_dash_execution_id,
            )
            _rrd = _run_results_d[0] if _run_results_d else None
            return _rrd.output if (_rrd and _rrd.success) else (_rrd.output or "No hay agentes para ejecutar." if _rrd else "No hay agentes para ejecutar.")

        # Auto-detect agent creation requests — call agent_manager skill directly
        _agent_params = _detect_agent_create(text) if _short_msg else None
        if _agent_params and self.skill_executor:
            logger.info("agent.auto_create_detected_dashboard", name=_agent_params.get("name"))
            from ..skills.types import SkillCall as _SC
            _agent_call = _SC(
                skill_name="agent_manager",
                arguments={"action": "create", **_agent_params},
                raw_text=f"[auto-detected: create agent '{_agent_params['name']}']",
            )
            _agent_results = await self.skill_executor.execute_batch(
                [_agent_call], user_id="dashboard", chat_id="dashboard",
                execution_id=_dash_execution_id,
            )
            _ar = _agent_results[0] if _agent_results else None
            if _ar and _ar.success:
                return _ar.output
            return f"Could not create agent: {(_ar.output or _ar.error or 'unknown error') if _ar else 'unknown error'}"

        # Auto-detect agent list requests
        if _short_msg and _detect_agent_list(text) and self.skill_executor:
            from ..skills.types import SkillCall as _SC
            _list_call = _SC(
                skill_name="agent_manager",
                arguments={"action": "list"},
                raw_text="[auto-detected: list agents]",
            )
            _list_results = await self.skill_executor.execute_batch(
                [_list_call], user_id="dashboard", chat_id="dashboard",
                execution_id=_dash_execution_id,
            )
            _lr = _list_results[0] if _list_results else None
            if _lr and _lr.success:
                return _lr.output
            return "Could not list agents."

        # Auto-detect delete a specific agent by name (check BEFORE delete-all)
        _del_name = _detect_agent_delete_one(text) if _short_msg else None
        if _del_name and self.skill_executor:
            from ..skills.types import SkillCall as _SC
            _del_call = _SC(
                skill_name="agent_manager",
                arguments={"action": "delete", "agent_id": _del_name},
                raw_text=f"[auto-detected: delete agent '{_del_name}']",
            )
            _del_results = await self.skill_executor.execute_batch(
                [_del_call], user_id="dashboard", chat_id="dashboard",
                execution_id=_dash_execution_id,
            )
            _dr = _del_results[0] if _del_results else None
            if _dr and _dr.success:
                return _dr.output
            return f"Could not delete agent '{_del_name}'."

        # Auto-detect delete ALL agents (or wipe_all if tasks/goals also mentioned)
        if _short_msg and _detect_agent_delete_all(text) and self.skill_executor:
            from ..skills.types import SkillCall as _SC
            _mentions_tasks_or_goals_db = bool(re.search(r"\b(?:tareas?|tasks?|goals?)\b", text, re.IGNORECASE))
            _del_action_db = "wipe_all" if _mentions_tasks_or_goals_db else "delete_all"
            _del_call = _SC(
                skill_name="agent_manager",
                arguments={"action": _del_action_db},
                raw_text=f"[auto-detected: {_del_action_db}]",
            )
            _del_results = await self.skill_executor.execute_batch(
                [_del_call], user_id="dashboard", chat_id="dashboard",
                execution_id=_dash_execution_id,
            )
            _dr = _del_results[0] if _del_results else None
            if _dr and _dr.success:
                return _dr.output
            return "Could not delete agents."

        # Auto-detect single task delete by name — bypass LLM
        _del_task_name_d = _detect_task_delete(text) if (_short_msg and self.skill_executor) else None
        if _del_task_name_d:
            from ..skills.types import SkillCall as _SC
            _dt_call_d = _SC(
                skill_name="task_manager",
                arguments={"action": "delete", "name": _del_task_name_d},
                raw_text=f"[auto-detected: delete task '{_del_task_name_d}']",
            )
            _dt_results_d = await self.skill_executor.execute_batch(
                [_dt_call_d], user_id="dashboard", chat_id="dashboard",
                execution_id=_dash_execution_id,
            )
            _dtr_d = _dt_results_d[0] if _dt_results_d else None
            if _dtr_d and _dtr_d.success:
                return _dtr_d.output
            return _dtr_d.error if (_dtr_d and _dtr_d.error) else f"No se encontró la tarea '{_del_task_name_d}'."

        # Behavioral learning: detect corrections and queue for analysis
        if _detect_correction(text):
            try:
                from ..memory.behavioral import queue_correction as _qc_d
                from ..memory.manager import MemoryQuery, MemoryType
                async with async_session() as _bcs:
                    # Filter by chat_id to avoid cross-chat contamination
                    _recent_d = await self.memory.retrieve(
                        _bcs, MemoryQuery(memory_type=MemoryType.EPISODIC, limit=10)
                    )
                    _prev_req = ""
                    _prev_res = ""
                    for _r in _recent_d or []:
                        if str(_r.content.get("chat_id", "")) == str(chat_id):
                            _prev_req = _r.content.get("user_input", "")
                            _prev_res = _r.content.get("agent_response", "")
                            break
                    if _prev_req or _prev_res:
                        asyncio.ensure_future(
                            _qc_d(user_request=_prev_req, agent_response=_prev_res,
                                  user_correction=text, chat_id=chat_id)
                        )
            except Exception:
                pass

        # Auto-detect Gmail credentials — configure immediately
        _gmail_creds_d = _detect_gmail_configure(text) if self.skill_executor else None
        if _gmail_creds_d:
            from ..skills.types import SkillCall as _SC
            _gm_addr_d, _gm_pw_d = _gmail_creds_d
            _gm_call_d = _SC(
                skill_name="gmail",
                arguments={"action": "configure", "address": _gm_addr_d, "password": _gm_pw_d},
                raw_text="[auto-detected: gmail configure]",
            )
            _gm_results_d = await self.skill_executor.execute_batch(
                [_gm_call_d], user_id=user_id, chat_id=chat_id,
                execution_id=_dash_execution_id,
            )
            _gmr_d = _gm_results_d[0] if _gm_results_d else None
            if _gmr_d and _gmr_d.success:
                return f"✅ Gmail conectado correctamente para {_gm_addr_d}. Ya puedo leer, enviar y buscar correos."
            else:
                err_d = (_gmr_d.error if _gmr_d else "") or "error desconocido"
                return f"❌ No pude conectarme a Gmail: {err_d}."

        # Auto-detect Gmail inbox queries — call gmail(action="inbox") directly
        if _detect_gmail_inbox(text) and self.skill_executor:
            from ..skills.types import SkillCall as _SC
            _gmi_call_d = _SC(
                skill_name="gmail",
                arguments={"action": "inbox", "count": "15"},
                raw_text="[auto-detected: gmail inbox]",
            )
            _gmi_results_d = await self.skill_executor.execute_batch(
                [_gmi_call_d], user_id=user_id, chat_id=chat_id,
                execution_id=_dash_execution_id,
            )
            _gmir_d = _gmi_results_d[0] if _gmi_results_d else None
            if _gmir_d and _gmir_d.success:
                return _gmir_d.output
            elif _gmir_d and "no configurado" in (_gmir_d.error or "").lower():
                return "Gmail is not configured. Share your Gmail address and app password so I can connect."
            else:
                return _gmir_d.error if (_gmir_d and _gmir_d.error) else "No pude acceder al correo."

        # Auto-detect lyrics requests — search and inject results before LLM
        _lyrics_query = _detect_lyrics_request(text)
        if _lyrics_query and self.skill_executor:
            from ..skills.types import SkillCall as _SC
            _s_call = _SC(skill_name="web_search", arguments={"query": _lyrics_query, "max_results": "5"}, raw_text="[auto: lyrics search]")
            _s_results = await self.skill_executor.execute_batch([_s_call], user_id=user_id, chat_id=chat_id, execution_id=_dash_execution_id)
            _sr = _s_results[0] if _s_results else None
            if _sr and _sr.success and _sr.output:
                text = f"{text}\n\n[Resultados de búsqueda de letra]:\n{_sr.output[:3000]}"

        # Auto-detect YouTube link requests — search and inject results before LLM
        _yt_query = _detect_youtube_request(text) if not _lyrics_query else None
        if _yt_query and self.skill_executor:
            from ..skills.types import SkillCall as _SC
            _s_call = _SC(skill_name="web_search", arguments={"query": _yt_query, "max_results": "5"}, raw_text="[auto: youtube search]")
            _s_results = await self.skill_executor.execute_batch([_s_call], user_id=user_id, chat_id=chat_id, execution_id=_dash_execution_id)
            _sr = _s_results[0] if _s_results else None
            if _sr and _sr.success and _sr.output:
                text = f"{text}\n\n[Resultados de búsqueda YouTube]:\n{_sr.output[:2000]}"

        active_model = self.model_manager.active_model
        active_provider = self.model_manager.active_provider

        skill_catalog = ""
        if self.skill_registry:
            skill_catalog = self.skill_registry.format_for_prompt()

        # ── Screenshot intent detection ────────────────────────────────────────
        # When user explicitly asks for a screenshot, inject a mandatory instruction
        # and run a post-loop enforcer in case the LLM still ignores it.
        _SCREENSHOT_INTENT_RE = re.compile(
            r"\b(captur[ae](?:me|la|lo|nos)?|screenshot|pantallazo|pantallaz|screenshoot|"
            r"saca(?:me)?\s+(?:una\s+)?(?:captura|imagen|foto|pantallazo)|"
            r"toma(?:me)?\s+(?:una\s+)?(?:captura|imagen|foto|pantallazo)|"
            r"haz(?:me)?\s+(?:una\s+)?(?:captura|imagen|pantallazo)|"
            r"env[ií]a(?:me)?\s+(?:una\s+)?(?:captura|imagen|pantallazo)|"
            r"manda(?:me)?\s+(?:una\s+)?(?:captura|imagen)|"
            r"quiero\s+(?:ver|la)\s+(?:captura|imagen|pantallazo|grafico|gráfico))\b",
            re.IGNORECASE,
        )
        _URL_IN_TEXT_RE = re.compile(r"https?://[^\s\)\]\"']+")
        _wants_screenshot = bool(_SCREENSHOT_INTENT_RE.search(text))
        _original_text = text  # keep for post-loop URL extraction
        # Unique execution ID for this dashboard request — enables idempotency across skill calls
        _dash_execution_id = str(uuid4())
        # Initialize request-scoped round budget (caps total LLM rounds across
        # cascaded loops in this single user request).
        request_budget_init(_dash_execution_id, text)
        # Decision trace — same one-record-per-request contract as Telegram.
        # Recorded automatically on ANY exit (early returns, exceptions,
        # normal completion) via the asyncio task done-callback below.
        _decision_trace_db = _new_decision_trace(
            path="dashboard",
            chat_id=str(chat_id),
            user_text=text,
            request_tier=_classify_request_tier(text),
        )
        try:
            _curr_task_db = asyncio.current_task()
            if _curr_task_db is not None:
                _redis_url_db = self.redis_url
                _trace_db_capture = _decision_trace_db

                def _record_db_on_done(t):
                    if getattr(_trace_db_capture, "_recorded", False):
                        return
                    _trace_db_capture._recorded = True
                    try:
                        asyncio.ensure_future(_record_decision_trace(_redis_url_db, _trace_db_capture))
                    except Exception:
                        pass

                _curr_task_db.add_done_callback(_record_db_on_done)
        except Exception:
            pass

        # Inject last known URL if user is asking about a page/screenshot without giving URL
        _last_url_dash = self._last_browser_url.get(chat_id, "")
        _BROWSER_CTX_RE = re.compile(
            r"\b(?:captur[ae]|screenshot|pantallazo|screenshoot|"
            r"continu[aá]|sigue|m[aá]s\s+(?:abajo|capturas?)|"
            r"saca(?:me)?\s+(?:una\s+)?(?:captura|imagen|foto|pantallazo)|"
            r"toma(?:me)?\s+(?:una\s+)?(?:captura|imagen|foto|pantallazo)|"
            r"haz(?:me)?\s+(?:una\s+)?(?:captura|imagen|pantallazo)|"
            r"la\s+(?:p[aá]gina|noticia|sitio|url|misma|este\s+sitio)|"
            r"este\s+(?:sitio|art[ií]culo|p[aá]gina|link|enlace)|"
            r"la\s+misma\s+(?:p[aá]gina|url|direcci[oó]n)|"
            r"env[ií]a(?:me)?\s+(?:la|las|los|esas?|estos?)\s+(?:captura|imagen|foto|screenshot))\b",
            re.IGNORECASE,
        )
        _BARE_DOMAIN_RE = re.compile(
            r"\b(?:[a-z0-9-]+\.)+(?:com|net|org|io|co|cl|es|ai|app|dev|me|mx|ar|us|uk|gov|edu)\b",
            re.IGNORECASE,
        )
        _has_any_url_dash = bool(_URL_IN_TEXT_RE.search(text)) or bool(_BARE_DOMAIN_RE.search(text))
        if _last_url_dash and _BROWSER_CTX_RE.search(text) and not _has_any_url_dash:
            text = f"{text}\n[URL activa: {_last_url_dash}]"
        elif _BROWSER_CTX_RE.search(text) and not _has_any_url_dash and not _last_url_dash:
            # No URL anywhere — ask for one instead of guessing.
            logger.info(
                "screenshot.url_required",
                chat_id=chat_id,
                text_preview=text[:80],
                path="dashboard",
            )
            from ..communication.translator import apick as _pp_url_db
            try:
                _db_lang = await _get_user_lang(self.redis_url, str(chat_id)) if chat_id else "en"
            except Exception:
                _db_lang = "en"
            return await _pp_url_db(
                "url_required", _db_lang, text[:60],
                self.model_manager, self.redis_url,
            )

        if _wants_screenshot:
            text = (
                text
                + "\n\n[SCREENSHOT REQUIRED: The user wants an IMAGE, not text. "
                "You MUST call browser(action=\"capture\", url=\"...\", session=\"s1\") "
                "as your FIRST skill. Do NOT use fetch_url or get_text — they return text only. "
                "Only browser(action=\"capture\") produces the screenshot image. "
                "After capture() returns the path, include it as: ![desc](/data/screenshots/screenshot_TIMESTAMP.png)"
            )

        # Auto-detect and pre-execute skills based on user message.
        # Skip auto-detect for scheduling requests — decision layer handles these.
        # Prevents web_search from firing on long monitoring/automation messages.
        pre_results_text = ""
        auto_results = []  # always defined; may be populated by auto-detect below
        _action_intent_db: _ActionIntent = _ActionIntent()  # default: no commitment
        _action_attempts_db: int = 0
        _planning_mode_db: bool = False  # default; overridden inside skill_executor block
        if self.skill_executor:
            _planning_mode_db = bool(_PLANNING_MODE_RE.search(text))
            if _planning_mode_db:
                logger.info("planning_mode.detected_dashboard", text_preview=text[:80])

            # ── Action Intent Classification (dashboard) ──────────────────────
            _action_intent_db = classify_action_intent(text, planning_mode=_planning_mode_db)
            if _action_intent_db.action_commitment:
                logger.info(
                    "action_intent.committed_dashboard",
                    action_type=_action_intent_db.action_type,
                    target=(_action_intent_db.action_target or "")[:80],
                    confidence=round(_action_intent_db.confidence, 2),
                    is_retry=_action_intent_db.is_retry_signal,
                )
            # ── End Action Intent Classification (dashboard) ──────────────────

            # ── Execution Plan Generation (dashboard path) ────────────────────
            _exec_plan_db: _ExecutionPlan | None = generate_plan(_action_intent_db)
            _plan_executor_db: _PlanExecutor | None = (
                _PlanExecutor(_exec_plan_db, self.skill_executor)
                if (_exec_plan_db is not None and self.skill_executor)
                else None
            )
            if _exec_plan_db is not None:
                logger.info(
                    "execution_planner.plan_generated_dashboard",
                    plan_id=_exec_plan_db.plan_id,
                    action_type=_exec_plan_db.action_type,
                    steps=len(_exec_plan_db.steps),
                    confidence=round(_exec_plan_db.confidence, 2),
                )
            # ── End Execution Plan Generation (dashboard) ─────────────────────

            auto_calls = [] if (is_scheduling_request(text) or _planning_mode_db) else detect_skills(text)
            # ── Intent boundary gate (auto-detect path, dashboard) ────────
            if auto_calls:
                _auto_kept_db, _auto_dropped_db = _filter_inferred_side_effects(
                    auto_calls, text, ctx_messages=None, chat_id=chat_id,
                )
                if _auto_dropped_db:
                    for _ad_sc, _ad_reason, _ad_intent in _auto_dropped_db:
                        logger.warning(
                            "intent_gate.blocked",
                            skill=_ad_sc.skill_name,
                            action=str((_ad_sc.arguments or {}).get("action", "")
                                       if isinstance(_ad_sc.arguments, dict) else ""),
                            reason=_ad_reason,
                            intent=_ad_intent,
                            path="auto_detect_dashboard",
                            text_preview=(text or "")[:80],
                        )
                    auto_calls = _auto_kept_db
            if auto_calls:
                logger.info(
                    "dashboard_chat.skills_auto_detected",
                    count=len(auto_calls),
                    skills=[c.skill_name for c in auto_calls],
                )
                if progress_callback:
                    for c in auto_calls:
                        ev = {"type": "skill_start", "skill": c.skill_name, "phase": "pre"}
                        args = c.arguments or {}
                        if args.get("action"): ev["action"] = args["action"]
                        if args.get("url"): ev["url"] = args["url"]
                        if args.get("query"): ev["query"] = args["query"][:80]
                        progress_callback(ev)

                # ── C2 Fix: Pre-execution guard for auto-detect path (dashboard) ─
                # CRIT-2: extended from browser-only to ALL external-interaction
                # skills so that web_search / http_request / fetch_url auto-detect
                # routes also pass domain lock and hard-block validation.
                _EXTERNAL_GUARD_SKILLS_DB = frozenset({
                    "browser", "web_search", "http_request", "fetch_url",
                })
                _external_auto_calls_db = [
                    c for c in auto_calls if c.skill_name in _EXTERNAL_GUARD_SKILLS_DB
                ]
                if _external_auto_calls_db:
                    try:
                        from .control_layer import (
                            pre_execution_check as _prex_fn_db,
                            infer_intent_from_text as _infer_fn_db,
                        )
                        _auto_intent_db = _action_intent_db
                        if not getattr(_auto_intent_db, "action_commitment", False):
                            _auto_intent_db = _infer_fn_db(text, _external_auto_calls_db)
                        _auto_prex_db = _prex_fn_db(
                            _external_auto_calls_db,
                            _auto_intent_db,
                            user_text=text,
                            domain_lock_override=None,
                        )
                        if _auto_prex_db.blocked:
                            _first_args_db = (_external_auto_calls_db[0].arguments or {})
                            logger.warning(
                                "auto_detect.guard_blocked",
                                path="dashboard",
                                reason=_auto_prex_db.violation_type,
                                url=_first_args_db.get("url", "")[:80],
                                query=_first_args_db.get("query", "")[:80],
                            )
                            # Discard blocked auto_detect results — fall through to LLM loop.
                            auto_calls = []
                    except Exception as _auto_guard_err_db:
                        logger.warning(
                            "auto_detect.guard_error",
                            path="dashboard",
                            error=str(_auto_guard_err_db)[:80],
                        )
                        # Fail-closed: guard exception blocks execution
                        return "Could not verify the safety of the requested operation."
                # ── End C2 Pre-execution guard (dashboard) ────────────────────

                auto_results = await self.skill_executor.execute_batch(
                    auto_calls, user_id=user_id, chat_id=chat_id,
                    execution_id=_dash_execution_id,
                )
                if progress_callback:
                    for i, r in enumerate(auto_results):
                        c = auto_calls[i] if i < len(auto_calls) else None
                        extra = {}
                        if c:
                            args = c.arguments or {}
                            if args.get("action"): extra["action"] = args["action"]
                            if args.get("url"): extra["url"] = args["url"]
                            if args.get("query"): extra["query"] = args["query"][:80]
                        progress_callback({"type": "skill_done", "skill": r.skill_name, "success": r.success, "ms": r.execution_ms, **extra})
                # For direct-response skills
                _DIRECT_SKILLS = ("create_reminder", "create_monitor", "list_monitors", "list_reminders")
                if auto_calls and auto_calls[0].skill_name in _DIRECT_SKILLS:
                    r = auto_results[0]
                    return r.output if r.success else f"Error: {r.error}"

                if not pre_results_text:
                    lines = []
                    for result in auto_results:
                        if result.success:
                            lines.append(result.output)
                        else:
                            lines.append(f"[{result.skill_name} error: {result.error}]")
                    pre_results_text = "\n".join(lines)

        # ── Decision Layer (dashboard) ────────────────────────────────────────────
        # Direct routing: SCHEDULED_TASK/SUB_AGENT → direct skill call (no LLM)
        #                 GOAL/SCRIPT → GoalOrchestrator (no LLM round-trip)
        # NOTE: goal_orchestrator is NOT required — SCHEDULED_TASK/SUB_AGENT work
        # with skill_executor alone.  GOAL/SCRIPT fall through to LLM if unavailable.
        _dl_strategy_db = Strategy.DIRECT_RESPONSE
        if not auto_calls and self.skill_executor and not _planning_mode_db:
            try:
                _dl_strategy_db = decide_execution_strategy(text)
                logger.info("decision_layer.dashboard", strategy=_dl_strategy_db.value, text_len=len(text), text_preview=text[:80])

                if _dl_strategy_db == Strategy.SCHEDULED_TASK:
                    if self.governor:
                        _gov_ok_db, _gov_msg_db = await self.governor.check_allow("create_task", user_id=user_id)
                        if not _gov_ok_db:
                            return _gov_msg_db
                    from ..decision_layer import extract_task_params
                    from ..skills.types import SkillCall as _SC
                    _tp_db = extract_task_params(text)
                    _task_call_db = _SC(
                        skill_name="task_manager",
                        arguments={"action": "create", **_tp_db},
                        raw_text=f"[decision_layer: schedule '{_tp_db['name']}']",
                    )
                    _task_res_db = await self.skill_executor.execute_batch(
                        [_task_call_db], user_id=user_id, chat_id=chat_id,
                        execution_id=_dash_execution_id,
                    )
                    _tr_db = _task_res_db[0] if _task_res_db else None
                    # Communication layer: natural language confirmation
                    try:
                        _db_lang = await _get_user_lang(self.redis_url, str(chat_id)) if chat_id else "en"
                    except Exception:
                        _db_lang = "en"
                    return await self._get_formatter().format_task_created(
                        user_request=text,
                        task_params=_tp_db,
                        success=bool(_tr_db and _tr_db.success),
                        raw_output=(_tr_db.output or _tr_db.error or "") if _tr_db else "",
                        user_lang=_db_lang,
                    )

                elif _dl_strategy_db == Strategy.SUB_AGENT:
                    if self.governor:
                        _gov_ok_db, _gov_msg_db = await self.governor.check_allow("create_agent", user_id=user_id)
                        if not _gov_ok_db:
                            return _gov_msg_db
                    from ..decision_layer import extract_agent_params
                    from ..skills.types import SkillCall as _SC
                    _ap_db = extract_agent_params(text)
                    _agent_call_db = _SC(
                        skill_name="agent_manager",
                        arguments={"action": "create", **_ap_db},
                        raw_text=f"[decision_layer: create agent '{_ap_db['name']}']",
                    )
                    _agent_res_db = await self.skill_executor.execute_batch(
                        [_agent_call_db], user_id=user_id, chat_id=chat_id,
                        execution_id=_dash_execution_id,
                    )
                    _ar_db = _agent_res_db[0] if _agent_res_db else None
                    # Communication layer: natural language confirmation
                    return await self._get_formatter().format_agent_created(
                        user_request=text,
                        agent_params=_ap_db,
                        success=bool(_ar_db and _ar_db.success),
                        raw_output=(_ar_db.output or _ar_db.error or "") if _ar_db else "",
                    )

                elif _dl_strategy_db in (Strategy.GOAL, Strategy.SCRIPT):
                    if not self.goal_orchestrator:
                        # No orchestrator — fall through to LLM
                        _dl_strategy_db = Strategy.DIRECT_RESPONSE
                    else:
                        if self.governor:
                            _gov_ok_db, _gov_msg_db = await self.governor.check_allow("create_goal", user_id=user_id)
                            if not _gov_ok_db:
                                return _gov_msg_db
                        # HIGH-3 Fix: await creation so we can report failure to user.
                        try:
                            await self.goal_orchestrator.create_goal(
                                objective=text, chat_id=chat_id,
                                user_id=user_id, priority=8, source="dashboard",
                            )
                            if self.governor:
                                await self.governor.record_action(
                                    "create_goal", user_id=user_id
                                )
                            logger.info(
                                "goal.enqueue_success",
                                chat_id=chat_id,
                                objective=text[:80],
                            )
                            return "✅ Objetivo registrado — ejecutando en segundo plano."
                        except Exception as _ge:
                            logger.warning(
                                "goal.enqueue_failure",
                                error=str(_ge)[:120],
                                chat_id=chat_id,
                            )
                            return (
                                "⚠️ No se pudo registrar el objetivo en este momento. "
                                "Por favor intenta de nuevo."
                            )

            except Exception:
                _dl_strategy_db = Strategy.DIRECT_RESPONSE
        # ── End Decision Layer (dashboard) ────────────────────────────────────────

        # ── Cross-Turn Retry State + Escalation Ladder (dashboard) ───────────
        if self.redis_url and _action_intent_db.action_commitment:
            try:
                import json as _acj_db, time as _act_db
                _acr_load_db = aioredis.from_url(self.redis_url, decode_responses=True)
                async with _acr_load_db as _acrl_db:
                    _prev_raw_db = await _acrl_db.get(f"action_state:{chat_id}")
                if _prev_raw_db:
                    _prev_db = _acj_db.loads(_prev_raw_db)
                    if (_prev_db.get("action_type") == _action_intent_db.action_type
                            or _action_intent_db.is_retry_signal):
                        _action_attempts_db = int(_prev_db.get("attempts", 0))
                        if _action_attempts_db > 0:
                            logger.info(
                                "action_intent.retry_state_loaded_dashboard",
                                attempts=_action_attempts_db,
                                action_type=_action_intent_db.action_type,
                            )
            except Exception:
                pass

        if (_action_intent_db.action_commitment
                and _action_attempts_db >= 2
                and self.goal_orchestrator
                and not _planning_mode_db):
            logger.info(
                "action_commitment.escalating_to_goal_dashboard",
                attempts=_action_attempts_db,
                target=(_action_intent_db.action_target or "")[:60],
            )
            if self.redis_url:
                try:
                    import json as _acj2_db, time as _act2_db
                    _acr_esc_db = aioredis.from_url(self.redis_url, decode_responses=True)
                    async with _acr_esc_db as _acwe_db:
                        await _acwe_db.setex(
                            f"action_state:{chat_id}",
                            1800,
                            _acj2_db.dumps({
                                "action_type": _action_intent_db.action_type,
                                "action_target": _action_intent_db.action_target,
                                "attempts": _action_attempts_db + 1,
                                "primary_skill": _action_intent_db.primary_skill,
                                "original_text": text[:200],
                                "timestamp": _act2_db.time(),
                            }),
                        )
                except Exception:
                    pass
            _esc_orch_db = self.goal_orchestrator
            _esc_obj_db = text
            _esc_uid_db = user_id

            async def _esc_goal_db():
                try:
                    await _esc_orch_db.create_goal(
                        objective=_esc_obj_db,
                        chat_id=chat_id,
                        user_id=_esc_uid_db,
                        priority=9,
                        source="action_escalation",
                    )
                except Exception as _ge:
                    logger.warning("action_commitment.escalation_failed_dashboard", error=str(_ge)[:80])

            asyncio.ensure_future(_esc_goal_db())
            return "Escalando a modo objetivo — ejecutando en segundo plano con plan detallado."
        # ── End Cross-Turn Retry / Escalation (dashboard) ────────────────────

        # Long input acknowledgment: if user sent a complex message that's going
        # to the LLM, send an immediate "processing" signal via progress callback.
        if len(text) > 300 and progress_callback and _dl_strategy_db == Strategy.DIRECT_RESPONSE:
            progress_callback({"type": "thinking", "round": 0, "note": "Analizando solicitud…"})

        # CPI-based automatic light-mode: if cognitive pressure index > 70, skip
        # heavy cognitive blocks to reduce token usage and latency under load.
        _cpi_light = False
        if self.redis_url:
            try:
                import redis.asyncio as _cpi_redis
                _cpi_r = _cpi_redis.from_url(self.redis_url, decode_responses=True)
                try:
                    _cpi_raw = await _cpi_r.get("agent:cpi_score")
                    if _cpi_raw and float(_cpi_raw) > 70:
                        _cpi_light = True
                        logger.info("chat_direct.cpi_light_mode_triggered", cpi=float(_cpi_raw))
                finally:
                    await _cpi_r.aclose()
            except Exception:
                pass

        async with async_session() as session:
            messages = await build_context(
                session, self.memory, text, chat_id,
                model_name=active_model, provider_name=active_provider,
                skill_catalog=skill_catalog,
                identity_manager=self.identity_manager,
                redis_url=self.redis_url,
                is_light_mode=_cpi_light,
            )

        # Planning mode: inject execution block into system prompt (dashboard path)
        if _planning_mode_db and messages and messages[0].role == "system":
            _pm_block_db = (
                "\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                "[PLANNING MODE — NO EXECUTION]\n"
                "The user wants to see the plan ONLY. You must NOT execute anything.\n\n"
                "ABSOLUTE RULES — violation is not allowed:\n"
                "1. DO NOT generate any <skill> tags. None. Zero.\n"
                "2. DO NOT create tasks, goals, agents, scheduled jobs, or call any API.\n"
                "3. DO NOT call task_manager, goal creation, agent_manager, or any skill.\n\n"
                "REQUIRED RESPONSE STRUCTURE (use these labels translated to the user's language):\n"
                "1. OVERVIEW — brief description of the approach\n"
                "2. STEPS — numbered list of what would be done and with which tools\n"
                "3. ARCHITECTURE — how the components connect (if applicable)\n"
                "4. End with a clear invitation to execute, e.g. (in the user's language):\n"
                '   "When you want me to run it, tell me: execute / create this / launch it."\n'
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            )
            messages[0] = Message(
                role="system",
                content=messages[0].content + _pm_block_db,
            )
            logger.info("planning_mode.prompt_injected_dashboard")

        # ── Action Commitment Injection (dashboard) ───────────────────────────
        # If plan available: inject plan-formatted block. Otherwise: classic block.
        if (_action_intent_db.action_commitment
                and not _planning_mode_db
                and messages
                and messages[0].role == "system"):
            if _exec_plan_db is not None:
                _ac_block_db = _build_action_commitment_block(_action_intent_db, _action_attempts_db)
                _plan_block_db = format_plan_for_prompt(_exec_plan_db)
                messages[0] = Message(
                    role="system",
                    content=messages[0].content + _ac_block_db + _plan_block_db,
                )
                logger.info(
                    "execution_planner.plan_injected_dashboard",
                    plan_id=_exec_plan_db.plan_id,
                    action_type=_exec_plan_db.action_type,
                )
            else:
                _ac_block_db = _build_action_commitment_block(_action_intent_db, _action_attempts_db)
                messages[0] = Message(
                    role="system",
                    content=messages[0].content + _ac_block_db,
                )
                logger.info(
                    "action_commitment.block_injected_dashboard",
                    action_type=_action_intent_db.action_type,
                    target=(_action_intent_db.action_target or "")[:60],
                    attempts=_action_attempts_db,
                )
        # ── End Action Commitment Injection (dashboard) ───────────────────────

        if pre_results_text:
            insert_pos = len(messages) - 1
            for fs_user, fs_assistant in WEB_FEWSHOT:
                messages.insert(insert_pos, Message(role="user", content=fs_user))
                insert_pos += 1
                messages.insert(insert_pos, Message(role="assistant", content=fs_assistant))
                insert_pos += 1
            messages[-1] = Message(
                role="user",
                content=(
                    f"{text}\n\n[DATA]:\n{pre_results_text}\n[/DATA]\n\n"
                    "Analyze the [DATA] above and answer the user's specific question. "
                    "Be concise. If the data doesn't contain what was asked, use a skill to find it."
                ),
            )

        # Track media files found in skill results for auto-injection
        _media_from_skills: list[str] = []
        _SKILL_PATH_RE = re.compile(
            r"(/data/(?:shared|chat-uploads|screenshots?)/[^\s\)\"']+\.(?:png|jpg|jpeg|gif|webp|bmp|mp4|webm|mov|mp3|ogg|wav|pdf))",
            re.IGNORECASE,
        )
        # Pre-populate with media from auto-detect phase (e.g. browser capture() results)
        # so the auto-inject at the end of the loop handles them even if LLM forgets the path
        _invalid_photos_db: set = set()
        for _pre_r in auto_results:
            if _pre_r.success:
                _pre_out_db = _pre_r.output or ""
                _pre_cap_inv_db = "[CAPTURE_VALID: false]" in _pre_out_db
                for _pm in _SKILL_PATH_RE.findall(_pre_out_db):
                    if _pm not in _media_from_skills and os.path.exists(_pm):
                        _media_from_skills.append(_pm)
                        if _pre_cap_inv_db:
                            _invalid_photos_db.add(_pm)

        # Execution-scoped artifact registry — shared state across all LLM rounds.
        # Only valid screenshots go here (used for email attachments).
        _exec_artifacts_db: dict[str, list[str]] = {
            "screenshots": [
                p for p in _media_from_skills
                if re.search(r"screenshot_\d+\.png$", p) and p not in _invalid_photos_db
            ]
        }
        _render_report_outputs_db: dict[str, str] = {}  # format → rendered text
        logger.info(
            "artifact_registry.initialized",
            path="dashboard",
            screenshot_count=len(_exec_artifacts_db["screenshots"]),
        )

        response_text = ""
        _browser_timed_out = False  # track if browser already failed — skip enforcer
        _chat_start = time.monotonic()
        _intent_retry_done_db = False  # one completeness retry per request maximum
        _action_enforcement_done_db = False  # one enforcement retry per action-committed request
        _db_skill_round_count = 0  # tracks rounds where skills were executed
        # System-Controlled Execution Engine state (dashboard)
        _action_all_results_db: list = []           # accumulates all skill results across rounds
        _action_history_db: list = []               # ordered record of browser actions with args+outputs
        _action_terminal_detected_db = False        # set True once terminal state is reached
        _action_terminal_success_db = False         # terminal success/failure
        _action_terminal_reason_db = ""             # reason code
        _action_final_prompt_override_db = ""       # replaces continuation prompt when terminal

        # ── Pre-Execution Phase (dashboard): DETERMINISTIC steps ─────────────
        if (_plan_executor_db is not None
                and _exec_plan_db is not None
                and _exec_plan_db.confidence >= 0.75
                and not _planning_mode_db
                and _action_intent_db.action_commitment):
            _pre_step_execs_db = await _plan_executor_db.execute_deterministic_sequence(
                user_id=str(user_id), chat_id=str(chat_id)
            )
            for _pse_db in _pre_step_execs_db:
                for _pse_sc_db, _pse_r_db in zip(_pse_db.skill_calls, _pse_db.results):
                    _action_all_results_db.append(_pse_r_db)
                    if _pse_r_db.skill_name == "browser":
                        _sc_args_db = _pse_sc_db.arguments if isinstance(_pse_sc_db.arguments, dict) else {}
                        _full_pre_out_db = _pse_r_db.output or ""
                        _stored_pre_out_db = _full_pre_out_db[:4096]
                        for _mk_db in ("[TRACK_STATUS:", "[FORM_STATUS:"):
                            if _mk_db in _full_pre_out_db and _mk_db not in _stored_pre_out_db:
                                for _mkl_db in _full_pre_out_db.split("\n"):
                                    if _mk_db in _mkl_db:
                                        _stored_pre_out_db += f"\n{_mkl_db.strip()}"
                                        break
                        _action_history_db.append({
                            "action": _sc_args_db.get("action", "unknown"),
                            "url": _sc_args_db.get("url", ""),
                            "selector": (_sc_args_db.get("selector", "") or "")[:80],
                            "text": (_sc_args_db.get("text", "") or "")[:60],
                            "success": _pse_r_db.success,
                            "output": _stored_pre_out_db,
                            "round": -1,
                        })
                if _pre_step_execs_db:
                    logger.info(
                        "execution_planner.pre_execution_dashboard",
                        step_id=_pse_db.step_id,
                        success=_pse_db.success,
                        output_snippet=(_pse_db.output or "")[:80],
                    )
            if _action_all_results_db and not _action_terminal_detected_db:
                _pre_is_term_db, _pre_success_db, _pre_reason_db = _check_action_terminal_state(
                    _action_intent_db, _action_all_results_db
                )
                if _pre_is_term_db:
                    _steps_ok_pre_db, _ = _verify_required_steps(
                        _action_intent_db.action_type, _action_history_db
                    )
                    if _steps_ok_pre_db:
                        _action_terminal_detected_db = True
                        _action_terminal_success_db = _pre_success_db
                        _action_terminal_reason_db = _pre_reason_db
                        _action_final_prompt_override_db = _build_constrained_final_prompt(
                            _action_intent_db, _action_all_results_db,
                            _pre_success_db, _pre_reason_db
                        )
                        logger.info(
                            "execution_planner.pre_execution_terminal_dashboard",
                            success=_pre_success_db, reason=_pre_reason_db,
                        )
        # ── End Pre-Execution Phase (dashboard) ───────────────────────────────

        # ── Pre-Execution Short-Circuit (dashboard) ───────────────────────────
        if _action_terminal_detected_db and _action_final_prompt_override_db and _action_all_results_db:
            _pre_out_lines_db = []
            for _pre_r_db in _action_all_results_db[:3]:
                _pre_snippet_db = (_pre_r_db.output or "")[:1500]
                _pre_out_lines_db.append(f"[pre_exec:{_pre_r_db.skill_name}] {_pre_snippet_db}")
            _pre_exec_summary_db = "\n".join(_pre_out_lines_db)
            messages.append(Message(
                role="user",
                content=(
                    "[SISTEMA] Ejecución automática completada antes de tu turno:\n\n"
                    f"{_pre_exec_summary_db}\n\n"
                    f"{_action_final_prompt_override_db}"
                ),
            ))
            _action_final_prompt_override_db = ""
            logger.info(
                "execution_planner.pre_execution_shortcircuit_dashboard",
                results=len(_action_all_results_db),
                reason=_action_terminal_reason_db,
            )
        # ── End Pre-Execution Short-Circuit (dashboard) ───────────────────────

        # ── Unified LLM Round Loop (dashboard) ───────────────────────────────
        # HIGH-2: Load any confirmed domain lock persisted from the previous turn.
        _persisted_lock_db = None
        if self.redis_url:
            try:
                from .control_layer import load_domain_lock as _load_dl_db
                _persisted_lock_db = await _load_dl_db(self.redis_url, str(chat_id))
            except Exception as _dl_load_err_db:
                logger.debug("domain_lock.load_skip_db", error=str(_dl_load_err_db)[:60])

        # ── Domain lock topic-change validation (dashboard, HARDENED) ──────────────
        if _persisted_lock_db and self.redis_url:
            try:
                from .control_layer import (
                    validate_lock_for_reuse as _validate_lock_db,
                    extract_domains_from_text as _edft_db,
                    clear_domain_lock as _clear_dl_db,
                )
                import re as _re_dl_db
                _validated_db, _validate_reason_db = _validate_lock_db(_persisted_lock_db, text or "")
                if _validated_db is None:
                    _old_repr_db = _persisted_lock_db.serialize() if hasattr(_persisted_lock_db, "serialize") else str(_persisted_lock_db)
                    await _clear_dl_db(
                        self.redis_url,
                        str(chat_id),
                        reason=_validate_reason_db or "topic_changed",
                        old_domain=_old_repr_db,
                    )
                    logger.info(
                        "domain_lock.auto_cleared_db",
                        reason=_validate_reason_db or "topic_changed",
                        old_domain=_old_repr_db,
                    )
                    _persisted_lock_db = None
                else:
                    _explicit_urls_db = _re_dl_db.findall(r"https?://[^\s]+", text or "")
                    if _explicit_urls_db:
                        _url_domains_db = [_edft_db(u) for u in _explicit_urls_db]
                        _url_domains_db = [d for d in _url_domains_db if d]
                        if _url_domains_db and not all(_persisted_lock_db.allows(d) for d in _url_domains_db):
                            _old_repr_db = _persisted_lock_db.serialize() if hasattr(_persisted_lock_db, "serialize") else str(_persisted_lock_db.domains)
                            await _clear_dl_db(
                                self.redis_url,
                                str(chat_id),
                                reason="user_explicit_url_mismatch",
                                old_domain=_old_repr_db,
                            )
                            logger.info(
                                "domain_lock.auto_cleared_db",
                                reason="user_explicit_url_mismatch",
                                old_domain=_old_repr_db,
                                new_domains=_url_domains_db,
                            )
                            _persisted_lock_db = None
            except Exception as _dl_clear_err_db:
                logger.debug("domain_lock.auto_clear_skip_db", error=str(_dl_clear_err_db)[:80])

        # ── Phase 4: Retrieve execution pattern before loop (dashboard) ─────────
        _em_hint_db = ""
        _em_pattern_id_db = ""
        if self.redis_url and getattr(_action_intent_db, "action_commitment", False):
            try:
                from .execution_memory import execution_memory as _em_db
                _em_pattern_db = await _em_db.find_pattern(
                    self.redis_url,
                    _action_intent_db.action_type,
                    getattr(_action_intent_db, "objective_spec", None),
                    _persisted_lock_db,
                )
                if _em_pattern_db:
                    _em_hint_db = _em_db.format_hint(_em_pattern_db)
                    _em_pattern_id_db = _em_pattern_db.get("id", "")
            except Exception as _em_find_err_db:
                logger.debug("execution_memory.find_skip_db", error=str(_em_find_err_db)[:80])

        # v2.5: Adaptive health state — non-blocking, fail-open
        _health_db = None
        try:
            from ..runtime.health_state import evaluate_health_from_redis as _eval_health_db
            _health_db = await _eval_health_db(self.redis_url) if self.redis_url else None
        except Exception:
            pass

        _lctx_db = _LoopContext(
            messages=messages,
            text=text,
            user_id=user_id,
            chat_id=chat_id,
            execution_id=_dash_execution_id,
            image_path=image_path,
            planning_mode=_planning_mode_db,
            is_scheduled_trigger=_is_scheduled_trigger,
            wants_screenshot=_wants_screenshot,
            start_time=_chat_start,
            auto_results=list(auto_results),
            exec_artifacts=_exec_artifacts_db,
            render_report_outputs=_render_report_outputs_db,
            action_intent=_action_intent_db,
            exec_plan=_exec_plan_db,
            progress_callback=progress_callback,
            active_domain_lock=_persisted_lock_db,
            pattern_hint=_em_hint_db,
            reused_pattern_id=_em_pattern_id_db,
            health_state=_health_db,
            invalid_photos=set(_invalid_photos_db),
        )

        # v2.5: Light-mode soft hint — appended to pattern_hint, never blocks execution
        if _health_db and _health_db.mode == "light" and getattr(_action_intent_db, "primary_skill", "") == "browser":
            _light_hint_db = "\n[SYSTEM_CONSTRAINT: LIGHT_MODE — avoid heavy browser usage, prefer fetch_url or lightweight tools]"
            if "SYSTEM_CONSTRAINT: LIGHT_MODE" not in (_lctx_db.pattern_hint or ""):
                _lctx_db.pattern_hint = (_lctx_db.pattern_hint or "") + _light_hint_db

        response_text = await self._run_llm_loop(_lctx_db)
        # HIGH-2: Persist confirmed domain lock for next turn.
        if self.redis_url and _lctx_db.active_domain_lock:
            try:
                from .control_layer import persist_domain_lock as _persist_dl_db
                await _persist_dl_db(self.redis_url, str(chat_id), _lctx_db.active_domain_lock)
            except Exception as _dl_save_err_db:
                logger.debug("domain_lock.persist_skip_db", error=str(_dl_save_err_db)[:60])

        # ── Phase 4: Store / update pattern after loop (dashboard) ───────────
        if self.redis_url and getattr(_action_intent_db, "action_commitment", False):
            _em_ok_db = _lctx_db.action_terminal_success
            _em_no_block_db = not _lctx_db.browser_timed_out
            if _em_ok_db and _em_no_block_db:
                try:
                    from .control_layer import validate_against_spec as _em_vas_db
                    _em_spec_db = getattr(_action_intent_db, "objective_spec", None)
                    _em_spec_ok_db, _ = _em_vas_db(
                        _lctx_db.last_results_text, _lctx_db.action_all_results, _em_spec_db
                    )
                except Exception:
                    _em_spec_ok_db = True
                if _em_spec_ok_db:
                    try:
                        from .execution_memory import execution_memory as _em_store_db
                        await _em_store_db.store_pattern(
                            self.redis_url,
                            _action_intent_db.action_type,
                            getattr(_action_intent_db, "objective_spec", None),
                            _lctx_db.action_history,
                            _lctx_db.action_all_results,
                            _lctx_db.active_domain_lock,
                        )
                    except Exception as _em_store_err_db:
                        logger.debug("execution_memory.store_skip_db", error=str(_em_store_err_db)[:80])
            elif not _em_ok_db and _lctx_db.reused_pattern_id:
                try:
                    from .execution_memory import execution_memory as _em_fail_db
                    await _em_fail_db.record_failure(self.redis_url, _lctx_db.reused_pattern_id)
                except Exception as _em_fail_err_db:
                    logger.debug("execution_memory.record_failure_skip_db", error=str(_em_fail_err_db)[:80])
        # Surface accumulated state for post-loop code
        _db_skill_round_count        = _lctx_db.skill_round_count
        _action_terminal_detected_db = _lctx_db.action_terminal_detected
        _action_terminal_success_db  = _lctx_db.action_terminal_success
        _action_all_results_db       = _lctx_db.action_all_results
        _browser_timed_out           = _lctx_db.browser_timed_out
        results_text                 = _lctx_db.last_results_text
        _invalid_photos_db.update(_lctx_db.invalid_photos)
        for _lp in _lctx_db.media_paths:
            if _lp not in _media_from_skills:
                _media_from_skills.append(_lp)

        # ── Save Cross-Turn Retry State (dashboard) ──────────────────────────
        if self.redis_url and _action_intent_db.action_commitment:
            try:
                import json as _acjs_db, time as _acts_db
                _ac_exec_ok_db = _action_terminal_success_db  # verified terminal success (unified with Telegram path)
                _new_attempts_db = 0 if _ac_exec_ok_db else (_action_attempts_db + 1)
                _acr_save_db = aioredis.from_url(self.redis_url, decode_responses=True)
                async with _acr_save_db as _acws_db:
                    await _acws_db.setex(
                        f"action_state:{chat_id}",
                        1800,
                        _acjs_db.dumps({
                            "action_type": _action_intent_db.action_type,
                            "action_target": _action_intent_db.action_target,
                            "attempts": _new_attempts_db,
                            "primary_skill": _action_intent_db.primary_skill,
                            "original_text": text[:200],
                            "last_success": _ac_exec_ok_db,
                            "timestamp": _acts_db.time(),
                        }),
                    )
                if _ac_exec_ok_db:
                    logger.info(
                        "action_execution.success",
                        path="dashboard",
                        action_type=_action_intent_db.action_type,
                        target=(_action_intent_db.action_target or "")[:60],
                    )
                else:
                    logger.info(
                        "action_execution.failed",
                        path="dashboard",
                        action_type=_action_intent_db.action_type,
                        target=(_action_intent_db.action_target or "")[:60],
                        attempts=_new_attempts_db,
                    )
            except Exception:
                pass
        # ── End Save Cross-Turn Retry State (dashboard) ───────────────────────

        response_text = strip_skill_calls(response_text)
        response_text = _strip_data_blocks(response_text)
        response_text = _extract_from_json_response(response_text)
        response_text = _strip_internal_paths(response_text)
        # M1 Fix: _clean_telegram_output() strips valid Markdown (**bold**, headers,
        # backticks) that the dashboard renderer preserves. It is NOT applied here.
        # Telegram path still applies it at line ~3810 for its own formatting needs.

        # ── Schedule honesty enforcement (dashboard) ────────────────────────
        # Same guard as the Telegram path. Combines action-flow + full LLM-loop
        # results to catch both confabulated schedules and unhonored fixed times.
        try:
            _all_turn_results_db = list(_action_all_results_db) + list(
                getattr(_lctx_db, "loop_skill_results", []) or []
            )
            # ── Honesty Layer (dashboard) — runs FIRST ────────────────────
            try:
                from .response_binding import apply_honesty_layer as _honesty_db
                response_text, _hon_trace_db = _honesty_db(
                    response_text,
                    skill_results=_all_turn_results_db,
                    user_text=text,
                    user_lang="en",
                    action_intent=_action_intent_db,
                )
                if _hon_trace_db.get("status") != "passthrough":
                    logger.info(
                        "honesty_layer.applied_dashboard",
                        chat_id=str(chat_id),
                        status=_hon_trace_db.get("status"),
                        leaked=_hon_trace_db.get("leaked_topics", []),
                        grounded=_hon_trace_db.get("grounded_topics", []),
                        reason=_hon_trace_db.get("reason", ""),
                    )
                    try:
                        _dtrace_db_hon = locals().get("_decision_trace_db")
                        if _dtrace_db_hon is not None:
                            _dtrace_db_hon.attach_response_guard({"honesty_layer": _hon_trace_db})
                    except Exception:
                        pass
            except Exception as _hl_err_db:
                logger.debug("honesty_layer.skip_db", error=str(_hl_err_db)[:120])

            # Single final-response policy entry point (same as Telegram path).
            response_text, _guard_trace_db = _apply_final_response_policy(
                response_text,
                user_text=text,
                skill_results=_all_turn_results_db,
                user_lang="en",
                chat_id=chat_id,
                recent_action_resolver=_get_recent_explicit_action,
            )
            try:
                _dtrace_db = locals().get("_decision_trace_db")
                if _dtrace_db is not None:
                    _dtrace_db.attach_response_guard(_guard_trace_db)
            except Exception:
                pass
        except Exception:
            pass

        # ── Request-budget partial-result note (dashboard) ───────────────
        if getattr(_lctx_db, "budget_stopped", False):
            _bs_db = getattr(_lctx_db, "budget_stop_state", {}) or {}
            _partial_note_db = (
                f"\n\n(I had to stop after {_bs_db.get('used','?')} rounds to avoid further exploration. "
                f"This is the best verified result so far; ask me to continue if something is missing.)"
            )
            if response_text and response_text.strip():
                response_text = response_text.rstrip() + _partial_note_db
            else:
                response_text = (
                    "I ran out of exploration budget before I could finish. "
                    "Try asking more simply or tell me exactly what you need."
                )

        # Free the per-request budget slot — the request is finishing.
        try:
            request_budget_release(_dash_execution_id)
        except Exception:
            pass

        # Persist decision trace (dashboard path — same contract as Telegram).
        try:
            if not getattr(_decision_trace_db, "_recorded", False):
                _decision_trace_db._recorded = True
                asyncio.ensure_future(_record_decision_trace(self.redis_url, _decision_trace_db))
        except Exception:
            pass

        # ── Priority 1: Symmetric Failure-Path Gate (dashboard) ───────────────
        # Gate: results exist AND terminal success was NOT achieved.
        # Does NOT require action_intent — execution results are sufficient signal.
        _fp_active_db = (
            len(_action_all_results_db) > 0
            and not _action_terminal_success_db
            and not _planning_mode_db
            and response_text
        )
        if _fp_active_db:
            _fp_action_type_db = (
                _action_intent_db.action_type if _action_intent_db else ""
            )
            logger.info(
                "failure_path.gate_entered",
                action_type=_fp_action_type_db or "unknown",
                results_count=len(_action_all_results_db),
                response_length=len(response_text),
                path="dashboard",
            )
            try:
                from .control_layer import enforce_failure_path_contract
                _fp_ref_id_db = (
                    getattr(_action_intent_db, "tracking_code", None)
                    or getattr(_action_intent_db, "action_target", None)
                    or ""
                ) if _action_intent_db else ""
                _fp_result_db = enforce_failure_path_contract(
                    response=response_text,
                    action_type=_fp_action_type_db,
                    results=_action_all_results_db,
                    artifacts=_lctx_db.exec_artifacts,
                    ref_id=_fp_ref_id_db,
                    user_text=text,
                )
                if not _fp_result_db.approved:
                    response_text = _fp_result_db.final_response
            except Exception as _fp_db_err:
                logger.warning(
                    "failure_path.gate_error",
                    error=str(_fp_db_err)[:80],
                    path="dashboard",
                )
        # ── End Failure-Path Gate (dashboard) ─────────────────────────────────

        # ── C3 Fix: Exhaustion-Path Guard (dashboard) ─────────────────────────
        if (
            not _action_terminal_detected_db
            and _action_intent_db is not None
            and getattr(_action_intent_db, "action_commitment", False)
            and not _planning_mode_db
            and response_text
        ):
            try:
                from .control_layer import enforce_exhaustion_path_contract
                _ex_result_db = enforce_exhaustion_path_contract(
                    response=response_text,
                    action_type=_action_intent_db.action_type,
                    results=_action_all_results_db,
                    user_text=text,
                )
                if not _ex_result_db.approved:
                    logger.warning(
                        "exhaustion_path.gate_blocked",
                        reason=_ex_result_db.reason,
                        action_type=_action_intent_db.action_type,
                        path="dashboard",
                        preview=response_text[:80],
                    )
                    response_text = _ex_result_db.final_response
            except Exception as _ex_db_err:
                logger.warning(
                    "exhaustion_path.gate_error",
                    error=str(_ex_db_err)[:80],
                    path="dashboard",
                )
        # ── End Exhaustion-Path Guard (dashboard) ────────────────────────────

        # ── Priority 3: Success-Path Sufficiency Gate (dashboard) ────────────
        if (
            _action_terminal_success_db
            and _action_intent_db is not None
            and getattr(_action_intent_db, "action_commitment", False)
            and not _planning_mode_db
            and response_text
        ):
            try:
                from .control_layer import enforce_success_path_contract
                _sp_ref_id_db = (
                    getattr(_action_intent_db, "tracking_code", None)
                    or getattr(_action_intent_db, "action_target", None)
                    or ""
                )
                _sp_result_db = enforce_success_path_contract(
                    response=response_text,
                    action_type=_action_intent_db.action_type,
                    results=_action_all_results_db,
                    artifacts=_lctx_db.exec_artifacts,
                    user_text=text,
                    ref_id=_sp_ref_id_db,
                    objective_spec=getattr(_action_intent_db, "objective_spec", None),
                )
                if not _sp_result_db.approved:
                    logger.warning(
                        "success_path.gate_blocked",
                        reason=_sp_result_db.reason,
                        action_type=_action_intent_db.action_type,
                        path="dashboard",
                    )
                    response_text = _sp_result_db.final_response
            except Exception as _sp_db_err:
                logger.warning(
                    "success_path.gate_error", error=str(_sp_db_err)[:80], path="dashboard"
                )
        # ── End Priority 3 Success-Path Gate (dashboard) ─────────────────────

        # ── Hard Response Validation + Auto-Recovery Layer (dashboard) ───────────
        # Build executed skills from BOTH auto-detect results AND LLM-round skill calls
        # (matches how Telegram path builds _executed_skills from _trace_spans + auto_results)
        _db_llm_skills: set[str] = set()
        _db_llm_skill_actions: set[str] = set()
        for _dm in messages:
            if getattr(_dm, "role", "") == "assistant":
                for _dsc in parse_skill_calls(_dm.content):
                    _db_llm_skills.add(_dsc.skill_name)
                    _dact = (_dsc.arguments or {}).get("action", "")
                    if _dact:
                        _db_llm_skill_actions.add(f"{_dsc.skill_name}:{_dact}")
        _db_executed_skills = (
            {r.skill_name for r in auto_results if r.success}
            | _db_llm_skills
            | _db_llm_skill_actions
        )
        # has_skill_data: True only if at least one skill actually ran (not just "round_num >= 0")
        _db_has_skill_data = len(auto_results) > 0 or len(_db_llm_skills) > 0
        _db_validation = ResponseValidator().validate(
            response_text=response_text,
            user_input=text,
            executed_skills=_db_executed_skills,
            has_any_skill_data=_db_has_skill_data,
            planning_mode=_planning_mode_db,
        )
        # Phase 2 — Screenshot completeness check (after main validator)
        if _db_validation.valid:
            _db_screenshot_check = check_screenshot_completeness(
                response_text, text, _db_executed_skills
            )
            if not _db_screenshot_check.valid:
                _db_validation = _db_screenshot_check

        _db_recovered = False  # track whether recovery succeeded for trace accuracy
        if not _db_validation.valid:
            logger.warning(
                "response_validator.dashboard.blocked",
                reason=_db_validation.reason,
                user_input=text[:100],
                response_preview=response_text[:100],
            )
            # Phase 3 — Recovery only when validator explicitly allows it
            _db_can_recover = _db_validation.should_retry
            if _db_can_recover and self.skill_executor:
                response_text, _db_recovered = await attempt_recovery(
                    validation_result=_db_validation,
                    response_text=response_text,
                    user_input=text,
                    messages=messages,
                    model_manager=self.model_manager,
                    skill_executor=self.skill_executor,
                    user_id=user_id,
                    chat_id=chat_id,
                    cleanup_fn=_clean_telegram_output,
                    redis_url=getattr(self, "redis_url", ""),  # Phase 4 — RecoveryMemory
                )
            else:
                response_text = _db_validation.fallback_response

        # ── Screenshot post-loop enforcer ──────────────────────────────────────
        # If user asked for a screenshot but the LLM never called capture(), do it now.
        # Skip enforcer if browser already timed out (avoid doubling wait time).
        # Only count paths that ACTUALLY EXIST on disk (prevents fake/hallucinated paths from blocking enforcer).
        _SHOT_PATH_RE = re.compile(r"(?:https?://)?(/data/screenshots/screenshot_\d+\.png)")
        _has_screenshot = any(
            os.path.exists(m.group(1)) for m in _SHOT_PATH_RE.finditer(response_text)
        )
        if _wants_screenshot and not _has_screenshot and not _browser_timed_out and self.skill_executor:
            try:
                from ..skills.types import SkillCall as _SC
                # Extract URL from original user message or from skill results text
                _cap_url = None
                _url_match = _URL_IN_TEXT_RE.search(_original_text)
                if _url_match:
                    _cap_url = _url_match.group(0)
                else:
                    # Scan results_text for navigated URLs
                    _rt = results_text if isinstance(results_text, str) else ""
                    for _line in _rt.split("\n"):
                        _um = _URL_IN_TEXT_RE.search(_line)
                        if _um and any(s in _um.group(0) for s in ("coinbase", "binance", "coingecko", "crypto", "chart")):
                            _cap_url = _um.group(0)
                            break
                if not _cap_url:
                    # Infer URL from keywords in original message using correct working URL patterns
                    _orig_low = _original_text.lower()
                    # Coin slug mapping
                    _COIN_SLUGS = {
                        "eth": "ethereum", "ethereum": "ethereum",
                        "btc": "bitcoin", "bitcoin": "bitcoin",
                        "sol": "solana", "solana": "solana",
                        "bnb": "bnb", "ada": "cardano", "cardano": "cardano",
                        "xrp": "ripple", "ripple": "ripple",
                        "dot": "polkadot", "avax": "avalanche",
                        "matic": "matic-network", "link": "chainlink",
                        "btv": "bitcoin",  # common typo
                    }
                    _coin_slug = "bitcoin"  # default
                    for _tok, _slug in _COIN_SLUGS.items():
                        if _tok in _orig_low:
                            _coin_slug = _slug
                            break
                    # Use only confirmed working sites (no anti-bot):
                    # CoinMarketCap and Yahoo Finance work; CoinGecko/Coinbase/Binance block headless browsers
                    _ticker = _coin_slug.upper()
                    if "yahoo" in _orig_low:
                        _cap_url = f"https://finance.yahoo.com/quote/{_ticker}-USD/"
                    elif "tradingview" in _orig_low:
                        _cap_url = f"https://www.tradingview.com/symbols/{_ticker}USD/"
                    else:
                        # Default to CoinMarketCap (reliable, USD, no anti-bot)
                        _cap_url = f"https://coinmarketcap.com/currencies/{_coin_slug}/"
                logger.info("screenshot_enforcer.executing", url=_cap_url)
                if progress_callback:
                    progress_callback({"type": "skill_start", "skill": "browser", "action": "capture", "url": _cap_url})
                _cap_call = _SC(
                    skill_name="browser",
                    arguments={"action": "capture", "url": _cap_url, "session": "s1", "chat_id": chat_id, "user_id": user_id},
                )
                _cap_results = await self.skill_executor.execute_batch(
                    [_cap_call], user_id=user_id, chat_id=chat_id,
                    execution_id=_dash_execution_id,
                )
                if _cap_results and _cap_results[0].success:
                    _path_m = re.search(r"/data/screenshots/\S+\.png", _cap_results[0].output)
                    if _path_m:
                        _cap_path = _path_m.group(0)
                        _media_from_skills.append(_cap_path)
                        if progress_callback:
                            progress_callback({"type": "skill_done", "skill": "browser", "success": True,
                                               "action": "capture", "url": _cap_url, "ms": _cap_results[0].execution_ms})
                        logger.info("screenshot_enforcer.success", path=_cap_path)
            except Exception as _e:
                logger.warning("screenshot_enforcer.failed", error=str(_e))

        # Replace any non-existent screenshot paths in the response (hallucinated / stale example paths)
        import os as _os
        for _sm in list(_SHOT_PATH_RE.finditer(response_text)):
            _found_path = _sm.group(0)
            if not _os.path.exists(_found_path):
                # Try to substitute a real path from skills
                _real_shot = next(
                    (p for p in _media_from_skills if _os.path.exists(p) and p.endswith(".png")),
                    None,
                )
                if _real_shot:
                    response_text = response_text.replace(_found_path, _real_shot)
                    logger.info("screenshot.fake_path_replaced", fake=_found_path, real=_real_shot)
                else:
                    # Remove the broken markdown image entirely
                    response_text = re.sub(
                        r"!\[[^\]]*\]\(" + re.escape(_found_path) + r"\)",
                        "",
                        response_text,
                    ).strip()
                    logger.info("screenshot.fake_path_removed", fake=_found_path)

        # ── Dashboard scheduled report: unify Telegram/dashboard output ─────────
        if _is_scheduled_trigger and _render_report_outputs_db:
            _is_report_task_db = bool(re.search(
                r"\b(informe|reporte|report|cripto|crypto|btc|eth|sol)\b", text, re.IGNORECASE
            ))
            _db_all_executed = {s.get("name", s.get("skill_name", "")).lower() for s in _skill_sequence_db} if hasattr(locals(), "_skill_sequence_db") else _db_executed_skills
            if _is_report_task_db and "gmail" in _db_all_executed:
                _tg_body_db = _render_report_outputs_db.get("telegram", "")
                if not _tg_body_db and "email" in _render_report_outputs_db:
                    _tg_body_db = "\n".join(_render_report_outputs_db["email"].splitlines()[:12]).strip()
                if _tg_body_db:
                    response_text = _tg_body_db + "\n\nCorreo enviado con el informe completo."
                    logger.info("handler.dashboard_telegram_unified_from_render_report")

        # Auto-inject any media files from skills that the LLM forgot to include
        for mpath in _media_from_skills:
            if mpath not in response_text and _os.path.exists(mpath):
                fname = _os.path.basename(mpath)
                ext = _os.path.splitext(fname)[1].lower()
                if ext in (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"):
                    response_text += f"\n\n![{fname}]({mpath})"
                elif ext in (".mp4", ".webm", ".mov"):
                    response_text += f"\n\n[🎬 {fname}]({mpath})"
                elif ext in (".mp3", ".ogg", ".wav"):
                    response_text += f"\n\n[🔊 {fname}]({mpath})"
                else:
                    response_text += f"\n\n[📎 {fname}]({mpath})"

        async with async_session() as session:
            await self.memory.store_episodic(
                session,
                event_type="dashboard.chat",
                user_input=text,
                agent_response=response_text,
                user_id=user_id,
                chat_id=chat_id,
            )

        # Structured execution trace for dashboard requests
        _db_exec_skills = {r.skill_name for r in auto_results if r.success}
        logger.info(
            "execution.trace",
            execution_id=_dash_execution_id,
            path="dashboard",
            skills_executed=sorted(_db_exec_skills),
            validation_status="ok" if _db_recovered else _db_validation.reason,
            validation_passed=_db_recovered or _db_validation.valid,
        )

        # Learning loop wiring: dashboard path now mirrors Telegram path so
        # learning_examples + behavioral corrections fire from both surfaces.
        try:
            _db_skill_blob = " ".join(
                getattr(r, "skill_name", "") + " " + (getattr(r, "output", "") or "")[:200]
                for r in auto_results if getattr(r, "success", False)
            )
            await self._set_last_exchange(str(user_id), text, _db_skill_blob[:1000])
        except Exception:
            pass

        return response_text
