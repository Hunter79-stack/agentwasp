"""Model management routes."""

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

router = APIRouter()

_ALL_PROVIDER_NAMES = [
    "openai", "openai-codex", "anthropic", "google", "xai",
    "mistral", "deepseek", "openrouter", "perplexity", "huggingface", "lmstudio", "moonshot",
]


@router.get("/", response_class=HTMLResponse)
async def models_page(request: Request):
    from ...models.ollama_provider import OllamaProvider
    model_manager = request.app.state.model_manager
    status = model_manager.get_status()
    catalog = model_manager.get_catalog()
    provider_info = await model_manager.get_provider_info()

    default_model = status.get("default_model", "")
    installed = []
    for provider_name, info in status["providers"].items():
        sizes = info.get("model_sizes", {})
        for model in info.get("models", []):
            installed.append({
                "name": model,
                "provider": provider_name,
                "active": model == status["active_model"],
                "is_default": model == default_model,
                "size": sizes.get(model, ""),
            })

    # Ollama-specific data
    ollama = model_manager.providers.get("ollama")
    ollama_available = isinstance(ollama, OllamaProvider)
    ollama_healthy = False
    ollama_installed = []  # models already downloaded
    ollama_total_size = ""
    if ollama_available:
        ollama_healthy = await ollama.health_check()
        sizes = ollama.model_sizes()
        ollama_installed = [
            {"name": m, "size": sizes.get(m, ""), "active": m == status["active_model"]}
            for m in ollama.available_models()
        ]
        # Sum disk usage across installed Ollama models
        total_bytes = sum(
            (info.get("size_bytes") or 0)
            for info in (ollama._model_info or {}).values()
        )
        if total_bytes >= 1_073_741_824:
            ollama_total_size = f"{total_bytes / 1_073_741_824:.1f} GB"
        elif total_bytes >= 1_048_576:
            ollama_total_size = f"{total_bytes / 1_048_576:.0f} MB"
        elif total_bytes > 0:
            ollama_total_size = f"{total_bytes} B"

    # Mark catalog entries as installed
    installed_names = {m["name"] for m in ollama_installed}
    catalog_enriched = [
        {**entry, "installed": entry["name"] in installed_names}
        for entry in catalog
    ]

    return request.app.state.templates.TemplateResponse(request, "models.html", {
        "status": status,
        "installed": installed,
        "catalog": catalog_enriched,
        "provider_info": provider_info,
        "all_provider_names": _ALL_PROVIDER_NAMES,
        "default_model": default_model,
        "ollama_available": ollama_available,
        "ollama_healthy": ollama_healthy,
        "ollama_installed": ollama_installed,
        "ollama_total_size": ollama_total_size,
    })


@router.get("/providers/health")
async def providers_health(request: Request, force: bool = False):
    """Return parallel health check results for all known providers.

    ?force=true bypasses the 30-second TTL cache and calls each provider
    directly, then refreshes the cache with the new result.
    """
    import structlog
    model_manager = request.app.state.model_manager
    if force:
        structlog.get_logger().info("provider.health_forced")
        model_manager.invalidate_health_cache()
    info = await model_manager.get_provider_info()
    return JSONResponse({"ok": True, "providers": info})


async def _get_redis(request: Request):
    import redis.asyncio as aioredis
    if not hasattr(request.app.state, "_redis") or request.app.state._redis is None:
        request.app.state._redis = aioredis.from_url(
            request.app.state.redis_url, decode_responses=True
        )
    return request.app.state._redis


@router.post("/providers/add")
async def add_provider(request: Request):
    """Register or update a provider at runtime and persist to Redis."""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"ok": False, "error": "Invalid JSON"}, status_code=400)
    name = (body.get("name") or "").strip().lower()
    credential = (body.get("api_key") or "").strip()
    if not name or not credential:
        return JSONResponse({"ok": False, "error": "name and provider credential are required"}, status_code=400)
    model_manager = request.app.state.model_manager
    result = await model_manager.register_provider(name, credential)
    # Persist to Redis so credential survives container restarts
    try:
        r = await _get_redis(request)
        await r.hset("apikeys", name, credential)
    except Exception:
        pass
    return JSONResponse(result)


@router.post("/providers/remove")
async def remove_provider(request: Request):
    """Remove a provider at runtime and delete from Redis."""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"ok": False, "error": "Invalid JSON"}, status_code=400)
    name = (body.get("name") or "").strip().lower()
    if not name:
        return JSONResponse({"ok": False, "error": "name is required"}, status_code=400)
    model_manager = request.app.state.model_manager
    removed = model_manager.remove_provider(name)
    # Remove from Redis persistence
    try:
        r = await _get_redis(request)
        await r.hdel("apikeys", name)
    except Exception:
        pass
    return JSONResponse({"ok": removed, "name": name})


@router.post("/ollama/delete")
async def ollama_delete_model(request: Request):
    """Unload from RAM and delete an Ollama model from disk."""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"ok": False, "error": "Invalid JSON"}, status_code=400)
    model_name = (body.get("model") or "").strip()
    if not model_name:
        return JSONResponse({"ok": False, "error": "model is required"}, status_code=400)
    from ...models.ollama_provider import OllamaProvider
    model_manager = request.app.state.model_manager
    ollama = model_manager.providers.get("ollama")
    if not ollama or not isinstance(ollama, OllamaProvider):
        return JSONResponse({"ok": False, "error": "Ollama not available"}, status_code=400)
    ok = await ollama.delete_model(model_name)
    # If this was the active model, switch to default
    if ok and model_manager.active_model == model_name:
        remaining = ollama.available_models()
        if remaining:
            await model_manager.switch_model(remaining[0])
    return JSONResponse({"ok": ok, "model": model_name})


@router.post("/ollama/pull")
async def ollama_pull_model(request: Request):
    """Download an Ollama model (streaming status via SSE not needed — just kick off)."""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"ok": False, "error": "Invalid JSON"}, status_code=400)
    model_name = (body.get("model") or "").strip()
    if not model_name:
        return JSONResponse({"ok": False, "error": "model is required"}, status_code=400)
    from ...models.ollama_provider import OllamaProvider
    model_manager = request.app.state.model_manager
    ollama = model_manager.providers.get("ollama")
    if not ollama or not isinstance(ollama, OllamaProvider):
        return JSONResponse({"ok": False, "error": "Ollama not available"}, status_code=400)
    try:
        status = await ollama.pull_model(model_name)
        return JSONResponse({"ok": True, "model": model_name, "status": status})
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)[:120]}, status_code=500)


@router.post("/switch")
async def switch_model(request: Request):
    try:
        data = await request.json()
        model_name = data.get("model", "")
    except Exception:
        form = await request.form()
        model_name = form.get("model", "")
    if model_name:
        await request.app.state.model_manager.switch_model(model_name)
    if request.headers.get("accept") == "application/json":
        return JSONResponse({"ok": True, "model": model_name})
    return RedirectResponse("/models", status_code=303)


@router.post("/set-default")
async def set_default_model(request: Request):
    """Set a model as the persistent default (survives restarts)."""
    try:
        body = await request.json()
        model_name = (body.get("model") or "").strip()
    except Exception:
        return JSONResponse({"ok": False, "error": "Invalid JSON"}, status_code=400)
    if not model_name:
        return JSONResponse({"ok": False, "error": "model is required"}, status_code=400)
    result = await request.app.state.model_manager.set_default_model(model_name)
    ok = "not found" not in result
    return JSONResponse({"ok": ok, "model": model_name, "message": result})
