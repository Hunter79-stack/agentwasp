"""OpenAI Codex OAuth provider regression tests."""
from __future__ import annotations

import time

import pytest

from src.models.base import LLMProvider
from src.models.manager import ModelManager
from src.models.openai_provider import OpenAICodexOAuthProvider, _parse_expires_at
from src.models.types import Message, ModelRequest, ModelResponse
from src.utils.redaction import redact


class _HealthyFallbackProvider(LLMProvider):
    def __init__(self) -> None:
        self.called = False

    def provider_name(self) -> str:
        return "fallback-test"

    def available_models(self) -> list[str]:
        return ["fallback-model"]

    async def health_check(self) -> bool:
        return True

    async def generate(self, request: ModelRequest) -> ModelResponse:
        self.called = True
        return ModelResponse(content="fallback", model_used="fallback-model", provider="fallback-test")


class _HealthyFailingCodexProvider(LLMProvider):
    fail_closed_on_unhealthy = True

    def provider_name(self) -> str:
        return "openai-codex"

    def available_models(self) -> list[str]:
        return ["gpt-4o-mini"]

    async def health_check(self) -> bool:
        return True

    async def generate(self, request: ModelRequest) -> ModelResponse:
        raise RuntimeError("simulated codex 401 during generation")


@pytest.mark.asyncio
async def test_codex_oauth_provider_fails_closed_without_token() -> None:
    provider = OpenAICodexOAuthProvider(access_token="")

    assert provider.provider_name() == "openai-codex"
    assert provider.auth_status()["configured"] is False
    assert await provider.health_check() is False

    with pytest.raises(RuntimeError, match="not configured"):
        await provider.generate(ModelRequest(messages=[Message(role="user", content="hello")]))


@pytest.mark.asyncio
async def test_codex_oauth_provider_fails_closed_when_expired() -> None:
    provider = OpenAICodexOAuthProvider(
        access_token="oauth-test-token-with-enough-length",
        token_expires_at=str(int(time.time()) - 10),
    )

    assert provider.auth_status()["configured"] is True
    assert provider.auth_status()["expired"] is True
    assert await provider.health_check() is False

    with pytest.raises(RuntimeError, match="expired"):
        await provider.generate(ModelRequest(messages=[Message(role="user", content="hello")]))


def test_codex_oauth_expiry_parser_accepts_epoch_and_iso() -> None:
    assert _parse_expires_at("1893456000") == 1893456000.0
    assert _parse_expires_at("2030-01-01T00:00:00Z") == 1893456000.0
    assert _parse_expires_at("not-a-date") == 0.0


@pytest.mark.asyncio
async def test_model_manager_can_register_codex_oauth_provider(monkeypatch) -> None:
    async def fake_health_check(self) -> bool:  # noqa: ANN001 - pytest monkeypatch target
        return False

    monkeypatch.setattr(OpenAICodexOAuthProvider, "health_check", fake_health_check)

    manager = ModelManager()
    result = await manager.register_provider(
        "openai-codex",
        "oauth-test-token-with-enough-length",
    )

    assert result["success"] is True
    assert result["healthy"] is False
    assert "openai-codex" in manager.providers
    assert "openai-codex" in manager.fallback_order


@pytest.mark.asyncio
async def test_model_manager_masks_codex_token_without_prefix_suffix(monkeypatch) -> None:
    async def fake_health_check(self) -> bool:  # noqa: ANN001 - pytest monkeypatch target
        return True

    monkeypatch.setattr(OpenAICodexOAuthProvider, "health_check", fake_health_check)

    raw_token = "oauth-sensitive-token-value-with-enough-length"
    manager = ModelManager()
    await manager.register_provider("openai-codex", raw_token)
    provider_info = await manager.get_provider_info()
    codex_info = next(p for p in provider_info if p["name"] == "openai-codex")

    assert codex_info["configured"] is True
    assert codex_info["masked_key"] == "oauth-***REDACTED***"
    assert "sensitive" not in codex_info["masked_key"]
    assert raw_token[-4:] not in codex_info["masked_key"]
    assert codex_info["auth_status"]["provider"] == "openai-codex"


@pytest.mark.asyncio
async def test_active_codex_auth_failure_refuses_fallback() -> None:
    codex = OpenAICodexOAuthProvider(
        access_token="oauth-test-token-with-enough-length",
        token_expires_at=str(int(time.time()) - 10),
    )
    fallback = _HealthyFallbackProvider()
    manager = ModelManager()
    manager.providers = {"openai-codex": codex, "fallback-test": fallback}
    manager.fallback_order = ["openai-codex", "fallback-test"]
    manager.active_provider = "openai-codex"
    manager.active_model = "gpt-4o-mini"

    with pytest.raises(RuntimeError, match="refusing fallback"):
        await manager.generate(ModelRequest(messages=[Message(role="user", content="hello")]))

    assert fallback.called is False


@pytest.mark.asyncio
async def test_active_codex_generation_failure_refuses_fallback() -> None:
    codex = _HealthyFailingCodexProvider()
    fallback = _HealthyFallbackProvider()
    manager = ModelManager()
    manager.providers = {"openai-codex": codex, "fallback-test": fallback}
    manager.fallback_order = ["openai-codex", "fallback-test"]
    manager.active_provider = "openai-codex"
    manager.active_model = "gpt-4o-mini"

    with pytest.raises(RuntimeError, match="failed during generation"):
        await manager.generate(ModelRequest(messages=[Message(role="user", content="hello")]))

    assert fallback.called is False


def test_oauth_tokens_are_redacted() -> None:
    fake_jwt = ".".join([
        "ey" + "J" + "a" * 30,
        "ey" + "J" + "b" * 30,
        "c" * 30,
    ])
    refresh_value = "refresh-token-" + "x" * 35
    session_value = "session-token-" + "y" * 35
    text = (
        "Authorization: " + "Bearer " + fake_jwt + "\n"
        + "refresh_token=" + refresh_value + "\n"
        + "session_token=" + session_value
    )
    redacted = redact(text)

    assert fake_jwt not in redacted
    assert refresh_value not in redacted
    assert session_value not in redacted
    assert "***REDACTED***" in redacted
