import time

import structlog

from .base import LLMProvider
from .ollama_provider import OllamaProvider
from .types import ModelRequest, ModelResponse


# Sensible starter model per provider — picked for low cost / decent capability,
# not the most powerful one. User can switch to a stronger model from the
# dashboard /models page. Used by ensure_default_model() on first install when
# the operator has not yet picked anything.
STARTER_MODEL = {
    "openai":     "gpt-4o-mini",
    "openai-codex": "gpt-4o-mini",
    "anthropic":  "claude-haiku-4-5-20251001",
    "google":     "gemini-2.0-flash",
    "xai":        "grok-4-1-fast-non-reasoning",
    "mistral":    "mistral-small-latest",
    "deepseek":   "deepseek-chat",
    "moonshot":   "moonshot-v1-8k",
    "perplexity": "llama-3.1-sonar-small-128k-online",
    "openrouter": "openai/gpt-4o-mini",
    "huggingface": "meta-llama/Llama-3.1-8B-Instruct",
}


class _OpenRouterProvider:
    """Thin wrapper: OpenAI-compatible + HTTP-Referer header for OpenRouter."""

    def __init__(self, api_key: str, models: list[str]):
        from .openai_provider import OpenAICompatibleProvider, OPENROUTER_MODELS
        self._inner = OpenAICompatibleProvider(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            provider_label="openrouter",
            models=models or OPENROUTER_MODELS,
        )
        # Patch extra headers into the inner provider
        self._inner._extra_headers = {"HTTP-Referer": "https://github.com/wasp-agent"}

    def provider_name(self) -> str:
        return "openrouter"

    async def health_check(self) -> bool:
        return await self._inner.health_check()

    def available_models(self) -> list[str]:
        return self._inner.available_models()

    def supports_vision(self, model: str = "") -> bool:
        return self._inner.supports_vision(model)

    def supports_audio(self, model: str = "") -> bool:
        return False

    async def generate(self, request: ModelRequest) -> ModelResponse:
        import httpx, time, base64
        from .types import TokenUsage, ModelResponse as MR

        model = request.model or self._inner._default_model
        start = time.monotonic()
        messages = [{"role": m.role, "content": m.content} for m in request.messages]

        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self._inner._api_key}",
                    "HTTP-Referer": "https://github.com/wasp-agent",
                    "Content-Type": "application/json",
                },
                json={"model": model, "messages": messages,
                      "temperature": request.temperature, "max_tokens": request.max_tokens},
            )
            resp.raise_for_status()
            data = resp.json()

        latency_ms = int((time.monotonic() - start) * 1000)
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        return MR(
            content=content, model_used=model, provider="openrouter",
            usage=TokenUsage(
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
            ),
            latency_ms=latency_ms,
        )

logger = structlog.get_logger()

# TTL for provider health cache in seconds — avoids re-checking on every generate()
_HEALTH_CACHE_TTL = 30.0


class ModelManager:
    """Manages LLM providers with fallback chain.

    Performance improvements:
    - Health check results cached for 30s (avoids round-trip on every generate)
    - get_provider_info() runs health checks in parallel
    """

    def __init__(self, ollama_base_url: str = "http://agent-ollama:11434"):
        self.providers: dict[str, LLMProvider] = {}
        self.fallback_order: list[str] = []
        self.active_provider: str = ""
        self.active_model: str = ""
        self.default_model: str = ""   # persisted default; restored on restart
        self.redis_url: str = ""       # set externally by main.py
        # Health cache: {provider_name: (is_healthy: bool, expires_at: float)}
        self._health_cache: dict[str, tuple[bool, float]] = {}

        # Initialize Ollama (local-first)
        ollama = OllamaProvider(base_url=ollama_base_url)
        self.providers["ollama"] = ollama
        self.fallback_order.append("ollama")
        self.active_provider = "ollama"
        self.active_model = ollama.default_model

    async def initialize(self):
        """Check which providers are available on startup."""
        for name in self.fallback_order:
            provider = self.providers[name]
            healthy = await provider.health_check()
            logger.info(
                "model_manager.provider_check",
                provider=name,
                healthy=healthy,
                models=provider.available_models() if healthy else [],
            )
            if healthy and not self.active_provider:
                self.active_provider = name

        # Restore persisted default model
        if self.redis_url:
            await self.load_default_model()

    async def _health_check_cached(self, provider_name: str) -> bool:
        """Check provider health with TTL cache to avoid redundant round-trips."""
        now = time.monotonic()
        cached = self._health_cache.get(provider_name)
        if cached and now < cached[1]:
            return cached[0]

        provider = self.providers.get(provider_name)
        if not provider:
            return False

        healthy = await provider.health_check()
        self._health_cache[provider_name] = (healthy, now + _HEALTH_CACHE_TTL)
        return healthy

    def invalidate_health_cache(self, provider_name: str | None = None) -> None:
        """Invalidate health cache (e.g. after a provider failure)."""
        if provider_name:
            self._health_cache.pop(provider_name, None)
        else:
            self._health_cache.clear()

    # Keywords that indicate a context-length overflow from any provider
    _OVERFLOW_SIGNALS = (
        "context_length_exceeded",
        "maximum context length",
        "context window",
        "too many tokens",
        "prompt is too long",
        "prompt too long",
        "reduce the length",
        "max_tokens",
        "input length",
        "token limit",
        "exceeds the limit",
        "tokens exceeds",
        "content too long",
    )

    @staticmethod
    def _is_overflow_error(error: Exception) -> bool:
        """Detect if an exception is a context-length overflow."""
        msg = str(error).lower()
        return any(sig in msg for sig in ModelManager._OVERFLOW_SIGNALS)

    @staticmethod
    def _compact_messages(messages: list, keep_exchanges: int = 3) -> list:
        """
        Compress message history to fit within context window.

        Strategy:
        1. Always keep the system message (index 0)
        2. Always keep the final user message (last message)
        3. Keep the last `keep_exchanges` user/assistant pairs
        4. If system message is very long, truncate its non-critical sections
        """
        if len(messages) <= 3:
            return messages

        system_msgs = [m for m in messages if m.role == "system"]
        non_system = [m for m in messages if m.role != "system"]

        if not non_system:
            return messages

        # Keep last N pairs + final user message
        # Each exchange = 1 user + 1 assistant message = 2 messages
        keep_count = keep_exchanges * 2 + 1  # +1 for final user msg
        if len(non_system) > keep_count:
            # Always keep the very last message
            last = non_system[-1:]
            middle = non_system[:-1]
            middle_kept = middle[-keep_count + 1:]
            non_system = middle_kept + last

        compacted = system_msgs + non_system
        logger.info(
            "model_manager.context_compacted",
            original=len(messages),
            compacted=len(compacted),
            keep_exchanges=keep_exchanges,
        )
        return compacted

    async def generate(self, request: ModelRequest) -> ModelResponse:
        """Generate a response using the active provider with fallback.

        Includes Compaction Overflow Recovery: if a provider returns a
        context-length error, the message history is automatically compressed
        and the request is retried before falling back to the next provider.
        """
        # Set active model on the request if not specified
        if not request.model and self.active_model:
            request.model = self.active_model

        original_model = request.model
        original_messages = list(request.messages)

        for provider_name in self._get_chain():
            provider = self.providers.get(provider_name)
            if not provider:
                continue

            healthy = await self._health_check_cached(provider_name)
            if not healthy:
                logger.warning("model_manager.provider_unhealthy", provider=provider_name)
                if (
                    provider_name == self.active_provider
                    and getattr(provider, "fail_closed_on_unhealthy", False)
                ):
                    raise RuntimeError(
                        f"Active provider {provider_name} failed authentication/health check; "
                        "refusing fallback to avoid silently using another provider."
                    )
                continue

            # On fallback, use the provider's default model instead of the
            # original (e.g. don't send "deepseek-r1:7b" to OpenAI)
            if provider_name != self.active_provider:
                available = provider.available_models()
                request.model = available[0] if available else original_model
                logger.info(
                    "model_manager.fallback",
                    from_provider=self.active_provider,
                    to_provider=provider_name,
                    model=request.model,
                )

            # Attempt with progressive compaction on overflow
            for attempt, keep_exchanges in enumerate([None, 4, 2, 1]):
                if keep_exchanges is not None:
                    # Compact messages for retry attempts
                    request.messages = self._compact_messages(original_messages, keep_exchanges)
                else:
                    request.messages = original_messages

                try:
                    response = await provider.generate(request)
                    # Non-blocking economics tracking
                    try:
                        from ..observability.economics import economics as _econ
                        usage = response.usage
                        if usage:
                            _econ.record(
                                model=response.model_used,
                                provider=provider_name,
                                prompt_tokens=usage.prompt_tokens,
                                completion_tokens=usage.completion_tokens,
                            )
                    except Exception:
                        pass
                    if attempt > 0:
                        logger.info(
                            "model_manager.overflow_recovered",
                            provider=provider_name,
                            attempt=attempt,
                            kept_exchanges=keep_exchanges,
                        )
                    return response

                except Exception as e:
                    if self._is_overflow_error(e) and keep_exchanges != 1:
                        # Context overflow — try again with more aggressive compaction
                        logger.warning(
                            "model_manager.overflow_detected",
                            provider=provider_name,
                            attempt=attempt,
                            error=str(e)[:120],
                        )
                        continue
                    else:
                        # Non-overflow error or final compaction attempt failed
                        logger.error(
                            "model_manager.provider_failed",
                            provider=provider_name,
                            error=str(e),
                        )
                        self.invalidate_health_cache(provider_name)
                        request.model = original_model
                        request.messages = original_messages
                        if (
                            provider_name == self.active_provider
                            and getattr(provider, "fail_closed_on_unhealthy", False)
                        ):
                            raise RuntimeError(
                                f"Active provider {provider_name} failed during generation; "
                                "refusing fallback to avoid silently using another provider."
                            ) from e
                        break  # Move to next provider

        raise RuntimeError("All LLM providers failed. No model available.")

    def _get_chain(self) -> list[str]:
        """Get provider chain starting with active, then fallbacks."""
        chain = [self.active_provider] if self.active_provider else []
        for name in self.fallback_order:
            if name not in chain:
                chain.append(name)
        return chain

    def supports_vision(self, model: str = "") -> bool:
        """Return True if the active provider/model supports image input."""
        provider = self.providers.get(self.active_provider)
        if not provider:
            return False
        return provider.supports_vision(model or self.active_model)

    def supports_audio(self) -> bool:
        """Return True if any configured provider can transcribe audio."""
        return any(p.supports_audio() for p in self.providers.values())

    def transcribe_audio_sync(self, audio_path: str) -> str:
        """Synchronous transcription — intended for asyncio.to_thread() use only."""
        for p in self.providers.values():
            if p.supports_audio() and hasattr(p, "transcribe_audio_sync"):
                return p.transcribe_audio_sync(audio_path)
        return ""

    async def transcribe_audio(self, audio_path: str) -> str:
        """Transcribe audio using the first available audio-capable provider."""
        for p in self.providers.values():
            if p.supports_audio() and hasattr(p, "transcribe_audio"):
                return await p.transcribe_audio(audio_path)
        return ""

    def get_status(self) -> dict:
        """Get current model status."""
        return {
            "active_provider": self.active_provider,
            "active_model": self.active_model,
            "default_model": self.default_model,
            "fallback_order": self.fallback_order,
            "providers": {
                name: {
                    "models": p.available_models(),
                    "model_sizes": p.model_sizes() if hasattr(p, "model_sizes") else {},
                }
                for name, p in self.providers.items()
            },
        }

    async def switch_model(self, model_name: str) -> str:
        """Switch to a different model. Returns status message."""
        # Check if model is available in any provider
        for name, provider in self.providers.items():
            await provider.health_check()
            if model_name in provider.available_models():
                self.active_provider = name
                self.active_model = model_name
                logger.info("model_manager.switched", provider=name, model=model_name)
                return f"Switched to {model_name} ({name})"

        return f"Model '{model_name}' not found. Use /model list to see installed models or /model available to see downloadable ones."

    async def set_default_model(self, model_name: str) -> str:
        """Switch to model and persist it as the default across restarts."""
        result = await self.switch_model(model_name)
        if "not found" in result:
            return result
        self.default_model = model_name
        if self.redis_url:
            try:
                import redis.asyncio as aioredis
                r = aioredis.from_url(self.redis_url, decode_responses=True)
                try:
                    await r.set("model:default", model_name)
                finally:
                    await r.aclose()
            except Exception as e:
                logger.warning("model_manager.set_default_failed", error=str(e))
        logger.info("model_manager.default_set", model=model_name)
        return f"Default model set to {model_name}"

    async def load_default_model(self) -> None:
        """Restore the persisted default model on startup."""
        if not self.redis_url:
            return
        try:
            import redis.asyncio as aioredis
            r = aioredis.from_url(self.redis_url, decode_responses=True)
            try:
                saved = await r.get("model:default")
            finally:
                await r.aclose()
            if saved:
                self.default_model = saved
                result = await self.switch_model(saved)
                logger.info("model_manager.default_restored", model=saved, result=result)
        except Exception as e:
            logger.warning("model_manager.load_default_failed", error=str(e))

    async def ensure_default_model(self) -> None:
        """First-run: if no default is persisted, pick a sensible starter model
        for the first registered remote provider so the agent is usable right
        after install without the operator having to touch the /models page.

        Selection rules:
          1. If a default model is already set in Redis (model:default) → keep it.
          2. Otherwise iterate providers in `fallback_order` (skip "ollama" — the
             local model is only useful when downloaded), pick the first whose
             starter model is available, switch to it, and persist as default.
        """
        if self.default_model:
            return  # already restored from Redis in load_default_model()

        for name in self.fallback_order:
            if name == "ollama":
                continue
            starter = STARTER_MODEL.get(name)
            if not starter:
                continue
            provider = self.providers.get(name)
            if not provider:
                continue
            available = provider.available_models()
            if starter not in available:
                # Fall back to first available model for this provider
                starter = available[0] if available else None
            if not starter:
                continue
            self.active_provider = name
            self.active_model = starter
            self.default_model = starter
            if self.redis_url:
                try:
                    import redis.asyncio as aioredis
                    r = aioredis.from_url(self.redis_url, decode_responses=True)
                    try:
                        await r.set("model:default", starter)
                    finally:
                        await r.aclose()
                except Exception as e:
                    logger.warning("model_manager.starter_persist_failed", error=str(e))
            logger.info("model_manager.starter_model_selected", provider=name, model=starter)
            return

        logger.info("model_manager.no_starter_model", note="No remote provider configured")

    async def download_model(self, model_name: str) -> str:
        """Download a model via Ollama. Returns status message."""
        ollama = self.providers.get("ollama")
        if not ollama or not isinstance(ollama, OllamaProvider):
            return "Ollama provider not available."

        if not await ollama.health_check():
            return "Ollama is not running."

        try:
            status = await ollama.pull_model(model_name)
            return f"Download complete: {model_name} ({status})"
        except Exception as e:
            return f"Download failed: {str(e)}"

    async def delete_model(self, model_name: str) -> str:
        """Delete a downloaded model. Auto-switches if active model is deleted."""
        ollama = self.providers.get("ollama")
        if not ollama or not isinstance(ollama, OllamaProvider):
            return "Ollama provider not available."

        was_active = model_name == self.active_model

        success = await ollama.delete_model(model_name)
        if not success:
            return f"Failed to delete: {model_name}"

        # If we deleted the active model, try to switch to another
        if was_active:
            remaining = ollama.available_models()
            if remaining:
                new_model = remaining[0]
                self.active_model = new_model
                return (
                    f"Deleted: {model_name}\n"
                    f"Auto-switched to: {new_model}"
                )
            else:
                self.active_model = ""
                return (
                    f"Deleted: {model_name}\n"
                    "No models remaining. Use /model download <name> to install one "
                    "or connect a remote API."
                )

        return f"Deleted: {model_name}"

    def get_catalog(self) -> list[dict]:
        """Get the curated catalog of downloadable models."""
        ollama = self.providers.get("ollama")
        if ollama and isinstance(ollama, OllamaProvider):
            return ollama.get_catalog()
        return []

    def auto_detect_providers(self, settings) -> None:
        """Register remote providers based on available API keys."""
        from .openai_provider import (
            OpenAICompatibleProvider, OpenAICodexOAuthProvider,
            OPENAI_MODELS, OPENAI_CODEX_MODELS, XAI_MODELS,
            MISTRAL_MODELS, DEEPSEEK_MODELS, OPENROUTER_MODELS,
            PERPLEXITY_MODELS, HUGGINGFACE_MODELS, LMSTUDIO_MODELS, MOONSHOT_MODELS,
        )
        from .anthropic_provider import AnthropicProvider
        from .google_provider import GoogleProvider

        if settings.openai_api_key:
            self.providers["openai"] = OpenAICompatibleProvider(
                api_key=settings.openai_api_key,
                provider_label="openai",
                models=OPENAI_MODELS,
            )
            self.fallback_order.append("openai")
            logger.info("model_manager.provider_registered", provider="openai")

        if getattr(settings, "openai_codex_access_token", ""):
            self.providers["openai-codex"] = OpenAICodexOAuthProvider(
                access_token=settings.openai_codex_access_token,
                base_url=getattr(settings, "openai_codex_base_url", "https://api.openai.com/v1"),
                models=OPENAI_CODEX_MODELS,
                token_expires_at=getattr(settings, "openai_codex_token_expires_at", ""),
                account_label=getattr(settings, "openai_codex_account_label", ""),
            )
            self.fallback_order.append("openai-codex")
            logger.info("model_manager.provider_registered", provider="openai-codex")

        if settings.xai_api_key:
            self.providers["xai"] = OpenAICompatibleProvider(
                api_key=settings.xai_api_key,
                base_url=settings.xai_base_url,
                provider_label="xai",
                models=XAI_MODELS,
            )
            self.fallback_order.append("xai")
            logger.info("model_manager.provider_registered", provider="xai")

        if settings.anthropic_api_key:
            self.providers["anthropic"] = AnthropicProvider(api_key=settings.anthropic_api_key)
            self.fallback_order.append("anthropic")
            logger.info("model_manager.provider_registered", provider="anthropic")

        if settings.google_api_key:
            self.providers["google"] = GoogleProvider(api_key=settings.google_api_key)
            self.fallback_order.append("google")
            logger.info("model_manager.provider_registered", provider="google")

        if settings.mistral_api_key:
            self.providers["mistral"] = OpenAICompatibleProvider(
                api_key=settings.mistral_api_key,
                base_url="https://api.mistral.ai/v1",
                provider_label="mistral",
                models=MISTRAL_MODELS,
            )
            self.fallback_order.append("mistral")
            logger.info("model_manager.provider_registered", provider="mistral")

        if settings.deepseek_api_key:
            self.providers["deepseek"] = OpenAICompatibleProvider(
                api_key=settings.deepseek_api_key,
                base_url="https://api.deepseek.com/v1",
                provider_label="deepseek",
                models=DEEPSEEK_MODELS,
            )
            self.fallback_order.append("deepseek")
            logger.info("model_manager.provider_registered", provider="deepseek")

        if settings.openrouter_api_key:
            self.providers["openrouter"] = _OpenRouterProvider(
                api_key=settings.openrouter_api_key,
                models=OPENROUTER_MODELS,
            )
            self.fallback_order.append("openrouter")
            logger.info("model_manager.provider_registered", provider="openrouter")

        if settings.perplexity_api_key:
            self.providers["perplexity"] = OpenAICompatibleProvider(
                api_key=settings.perplexity_api_key,
                base_url="https://api.perplexity.ai",
                provider_label="perplexity",
                models=PERPLEXITY_MODELS,
            )
            self.fallback_order.append("perplexity")
            logger.info("model_manager.provider_registered", provider="perplexity")

        if settings.huggingface_api_key:
            self.providers["huggingface"] = OpenAICompatibleProvider(
                api_key=settings.huggingface_api_key,
                base_url="https://api-inference.huggingface.co/v1",
                provider_label="huggingface",
                models=HUGGINGFACE_MODELS,
            )
            self.fallback_order.append("huggingface")
            logger.info("model_manager.provider_registered", provider="huggingface")

        if settings.moonshot_api_key:
            self.providers["moonshot"] = OpenAICompatibleProvider(
                api_key=settings.moonshot_api_key,
                base_url="https://api.moonshot.cn/v1",
                provider_label="moonshot",
                models=MOONSHOT_MODELS,
            )
            self.fallback_order.append("moonshot")
            logger.info("model_manager.provider_registered", provider="moonshot")

        # LM Studio — probe local endpoint
        if settings.lmstudio_base_url:
            self.providers["lmstudio"] = OpenAICompatibleProvider(
                api_key="lm-studio",  # LM Studio doesn't require a real key
                base_url=settings.lmstudio_base_url,
                provider_label="lmstudio",
                models=LMSTUDIO_MODELS,
            )
            self.fallback_order.append("lmstudio")
            logger.info("model_manager.provider_registered", provider="lmstudio")

    async def switch_provider(self, provider_name: str) -> bool:
        """Switch the active provider."""
        if provider_name not in self.providers:
            return False
        provider = self.providers[provider_name]
        if not await provider.health_check():
            return False
        self.active_provider = provider_name
        logger.info("model_manager.switched", provider=provider_name)
        return True

    async def register_provider(self, name: str, api_key: str) -> dict:
        """Register or update a remote provider at runtime. Returns status dict."""
        from .openai_provider import (
            OpenAICompatibleProvider, OpenAICodexOAuthProvider,
            OPENAI_MODELS, OPENAI_CODEX_MODELS, XAI_MODELS,
            MISTRAL_MODELS, DEEPSEEK_MODELS, OPENROUTER_MODELS,
            PERPLEXITY_MODELS, HUGGINGFACE_MODELS, MOONSHOT_MODELS,
        )
        from .anthropic_provider import AnthropicProvider
        from .google_provider import GoogleProvider

        factories = {
            "openai": lambda k: OpenAICompatibleProvider(
                api_key=k, provider_label="openai", models=OPENAI_MODELS,
            ),
            "openai-codex": lambda k: OpenAICodexOAuthProvider(
                access_token=k, models=OPENAI_CODEX_MODELS,
            ),
            "xai": lambda k: OpenAICompatibleProvider(
                api_key=k, base_url="https://api.x.ai/v1",
                provider_label="xai", models=XAI_MODELS,
            ),
            "anthropic": lambda k: AnthropicProvider(api_key=k),
            "google":    lambda k: GoogleProvider(api_key=k),
            "mistral":   lambda k: OpenAICompatibleProvider(
                api_key=k, base_url="https://api.mistral.ai/v1",
                provider_label="mistral", models=MISTRAL_MODELS,
            ),
            "deepseek":  lambda k: OpenAICompatibleProvider(
                api_key=k, base_url="https://api.deepseek.com/v1",
                provider_label="deepseek", models=DEEPSEEK_MODELS,
            ),
            "openrouter": lambda k: _OpenRouterProvider(api_key=k, models=OPENROUTER_MODELS),
            "perplexity": lambda k: OpenAICompatibleProvider(
                api_key=k, base_url="https://api.perplexity.ai",
                provider_label="perplexity", models=PERPLEXITY_MODELS,
            ),
            "huggingface": lambda k: OpenAICompatibleProvider(
                api_key=k, base_url="https://api-inference.huggingface.co/v1",
                provider_label="huggingface", models=HUGGINGFACE_MODELS,
            ),
            "lmstudio": lambda k: OpenAICompatibleProvider(
                api_key="lm-studio", base_url=k,  # k = base_url for lmstudio
                provider_label="lmstudio", models=[],
            ),
            "moonshot": lambda k: OpenAICompatibleProvider(
                api_key=k, base_url="https://api.moonshot.cn/v1",
                provider_label="moonshot", models=MOONSHOT_MODELS,
            ),
            "kimi": lambda k: OpenAICompatibleProvider(
                api_key=k, base_url="https://api.moonshot.cn/v1",
                provider_label="moonshot", models=MOONSHOT_MODELS,
            ),
        }

        if name not in factories:
            return {"success": False, "error": f"Unknown provider: {name}. Supported: {list(factories)}"}

        provider = factories[name](api_key)
        self.providers[name] = provider
        if name not in self.fallback_order:
            self.fallback_order.append(name)

        healthy = await provider.health_check()
        logger.info(
            "model_manager.provider_registered",
            provider=name, healthy=healthy,
        )
        return {
            "success": True,
            "healthy": healthy,
            "models": provider.available_models() if healthy else [],
        }

    def remove_provider(self, name: str) -> bool:
        """Remove a remote provider. Returns True if it existed."""
        if name not in self.providers or name == "ollama":
            return False

        del self.providers[name]
        if name in self.fallback_order:
            self.fallback_order.remove(name)

        # If we removed the active provider, switch to first available
        if self.active_provider == name:
            self.active_provider = self.fallback_order[0] if self.fallback_order else ""
            if self.active_provider == "ollama":
                ollama = self.providers.get("ollama")
                if ollama:
                    self.active_model = ollama.default_model

        logger.info("model_manager.provider_removed", provider=name)
        return True

    async def get_provider_info(self) -> list[dict]:
        """Get info for all known provider slots — health checks run in parallel."""
        import asyncio
        from .anthropic_provider import ANTHROPIC_MODELS
        from .google_provider import GOOGLE_MODELS
        from .openai_provider import (
            OPENAI_MODELS, OPENAI_CODEX_MODELS, XAI_MODELS, MISTRAL_MODELS, DEEPSEEK_MODELS,
            MOONSHOT_MODELS, OPENROUTER_MODELS, PERPLEXITY_MODELS, HUGGINGFACE_MODELS,
        )
        _catalogs: dict[str, list[str]] = {
            "openai": OPENAI_MODELS,
            "openai-codex": OPENAI_CODEX_MODELS,
            "anthropic": ANTHROPIC_MODELS,
            "google": GOOGLE_MODELS,
            "xai": XAI_MODELS,
            "mistral": MISTRAL_MODELS,
            "deepseek": DEEPSEEK_MODELS,
            "moonshot": MOONSHOT_MODELS,
            "openrouter": OPENROUTER_MODELS,
            "perplexity": PERPLEXITY_MODELS,
            "huggingface": HUGGINGFACE_MODELS,
            "lmstudio": [],
        }
        all_names = [
            "openai", "openai-codex", "anthropic", "google", "xai",
            "mistral", "deepseek", "moonshot", "openrouter", "perplexity", "huggingface", "lmstudio",
        ]

        async def _check_one(name: str) -> dict:
            provider = self.providers.get(name)
            catalog = _catalogs.get(name, [])
            if not provider:
                return {
                    "name": name, "configured": False,
                    "healthy": False, "masked_key": "", "models": catalog,
                    "auth_status": {},
                }
            healthy = await self._health_check_cached(name)
            key = getattr(provider, "_api_key", "")
            auth_status_fn = getattr(provider, "auth_status", None)
            auth_status = auth_status_fn() if callable(auth_status_fn) else {}
            if name == "openai-codex" and key:
                # OAuth/JWT bearer tokens should not reveal prefix/suffix material.
                masked = "oauth-***REDACTED***"
            else:
                masked = self._mask_key(key) if key else ""
            # Always show full model list; use live list when healthy, catalog otherwise
            models = provider.available_models() if healthy else catalog
            return {
                "name": name,
                "configured": True,
                "healthy": healthy,
                "masked_key": masked,
                "models": models,
                "auth_status": auth_status,
            }

        return list(await asyncio.gather(*[_check_one(n) for n in all_names]))

    @staticmethod
    def _mask_key(key: str) -> str:
        if len(key) <= 10:
            return "***"
        return key[:6] + "..." + key[-4:]
