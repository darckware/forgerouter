"""Mostly-read-only MCP surface over ForgeRouter's provider/model/agent
admin data.

Mounted at /mcp in app/main.py (mcp_server.streamable_http_app(...)). Every
tool reuses the exact same query/masking functions the equivalent
/admin/* REST endpoint already calls — see each tool's docstring for its
REST counterpart. delete_agent is the one deliberately destructive tool
(gated behind a confirm=True second call); nothing else creates, updates,
or reveals a secret (GET /admin/providers/{name}/key and
GET /admin/agents/{name}/key are deliberately not exposed here, unlike
everything else under /admin/*).

Auth mirrors require_admin() in app/main.py: the caller's Authorization
header must be either a registered agent's own API key or a dashboard
session token. There is no separate MCP credential.
"""

from __future__ import annotations

from typing import Any

from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from app.demand import DEMAND_INFO, DEMANDS, VIRTUAL_MODELS, default_chain
from app.registry import load_registry_with_db_health, mask_secret, provider_readiness
from app.routing_state import model_performance_cached

mcp_server = MCPServer(
    name="forgerouter",
    instructions=(
        "Inspection of ForgeRouter's providers, models and agents (any agent key), "
        "plus administration -- add/remove providers, discover and validate models, "
        "run the health scan and resync, sync pricing -- for admin agents only "
        "(FORGEROUTER_ADMIN_AGENTS, default Athos and Hephaestus) or a dashboard "
        "session. Bearer token = the caller's own ForgeRouter API key; no separate credential."
    ),
)


def _authorize(token: str) -> bool:
    from app.storage import find_agent_by_key, session_user

    if not token:
        return False
    try:
        if session_user(token):
            return True
    except Exception:
        pass
    try:
        return bool(find_agent_by_key(token))
    except Exception:
        return False


def _bearer(ctx: Context) -> str:
    headers = ctx.headers or {}
    authorization = headers.get("authorization") or headers.get("Authorization") or ""
    return authorization[len("Bearer "):].strip() if authorization.startswith("Bearer ") else ""


def _require_admin(ctx: Context) -> None:
    """Write tools: a dashboard session or an allow-listed admin agent
    (app/admin_policy.py) -- the same rule as require_admin() for /admin/*."""
    from app.admin_policy import admin_agent_names, is_admin_token

    if not is_admin_token(_bearer(ctx)):
        raise ToolError(
            "This tool changes ForgeRouter and is limited to admin agents "
            f"({', '.join(sorted(admin_agent_names())) or 'none configured'}) or a dashboard session."
        )


def _require_agent(ctx: Context) -> None:
    headers = ctx.headers or {}
    authorization = headers.get("authorization") or headers.get("Authorization") or ""
    token = authorization[len("Bearer "):].strip() if authorization.startswith("Bearer ") else ""
    if not _authorize(token):
        raise ToolError(
            "A valid Bearer token is required — an agent's own API key or a dashboard session "
            "(same credential as ForgeRouter's /admin/* API)."
        )


@mcp_server.tool()
def list_providers(ctx: Context) -> dict[str, Any]:
    """Full provider/model configuration, API keys masked. Same data as
    GET /admin/providers/registry."""
    _require_agent(ctx)
    from app.main import _mask_registry_providers
    from app.storage import db_providers_with_models

    try:
        return {"providers": _mask_registry_providers(db_providers_with_models()), "source": "database"}
    except Exception:
        from app.registry import load_provider_dicts

        return {"providers": _mask_registry_providers(load_provider_dicts()), "source": "yaml_fallback"}


@mcp_server.tool()
def provider_health(ctx: Context, provider: str = "") -> dict[str, Any]:
    """Latest health-scan verdict per model. Same data as
    GET /admin/providers/health, optionally filtered to one provider name."""
    _require_agent(ctx)
    from app.storage import latest_provider_health_rows

    try:
        rows = latest_provider_health_rows()
    except Exception:
        rows = []
    if provider:
        rows = [row for row in rows if row.get("provider") == provider]
    return {"providers": rows}


@mcp_server.tool()
def provider_readiness(ctx: Context) -> dict[str, Any]:
    """Which providers have their API-key env var actually set — env var
    names and a configured boolean only, never secret values. Same data as
    GET /admin/providers/readiness."""
    _require_agent(ctx)
    from app.registry import provider_readiness as _provider_readiness

    return {"providers": _provider_readiness()}


@mcp_server.tool()
def list_models(ctx: Context) -> dict[str, Any]:
    """Virtual (forgerouter/auto, /simple, ...) and concrete models exactly
    as GET /v1/models reports them, including each virtual model's
    guaranteed context_length. Scoped to the caller's own agent model
    restrictions when the Bearer token identifies a registered agent."""
    _require_agent(ctx)
    from app.main import VIRTUAL_MODELS as _VIRTUAL_MODELS
    from app.main import _virtual_model_context_lengths
    from app.demand import DEMAND_INFO as _DEMAND_INFO
    from app.storage import agent_allowed_models, find_agent_by_key

    headers = ctx.headers or {}
    authorization = headers.get("authorization") or headers.get("Authorization") or ""
    token = authorization[len("Bearer "):].strip() if authorization.startswith("Bearer ") else ""
    allowed: set[str] | None = None
    try:
        agent_name = find_agent_by_key(token)
        if agent_name:
            allowed = agent_allowed_models(agent_name)
    except Exception:
        allowed = None
    registry = load_registry_with_db_health()
    context_lengths = _virtual_model_context_lengths(registry, allowed=allowed)
    virtual = [
        {
            "id": model_id,
            "object": "model",
            "owned_by": "forgerouter",
            "context_length": context_lengths[model_id.split("/", 1)[1]],
            "metadata": {"virtual": True, "description": _DEMAND_INFO.get(model_id.split("/", 1)[1], "Routes by demand class automatically.")},
        }
        for model_id in _VIRTUAL_MODELS
    ]
    return {"object": "list", "data": virtual + registry.openai_models()}


@mcp_server.tool()
def model_pricing(ctx: Context) -> dict[str, Any]:
    """Reference cost lookup coverage per model — which models have a
    resolvable price and from which tier (live/override/catalog). Same data
    as GET /admin/pricing/models."""
    _require_agent(ctx)
    from app.pricing import resolve_price_info
    from app.storage import get_setting

    try:
        registry = load_registry_with_db_health()
    except Exception:
        return {"models": []}
    seen: dict[str, dict[str, Any]] = {}
    for model in registry.models:
        if model.id in seen:
            continue
        try:
            info = resolve_price_info(model.id, model.provider_model)
        except Exception:
            info = None
        seen[model.id] = {
            "public_id": model.id,
            "provider_model": model.provider_model,
            "priced": info is not None,
            "input_cost_per_token": info.get("input_cost_per_token") if info else None,
            "output_cost_per_token": info.get("output_cost_per_token") if info else None,
            "source": info.get("source") if info else None,
        }
    models = sorted(seen.values(), key=lambda item: (not item["priced"], item["public_id"]))
    try:
        last_synced = get_setting("pricing_last_synced")
    except Exception:
        last_synced = None
    return {
        "models": models,
        "priced_count": sum(1 for m in models if m["priced"]),
        "total_count": len(models),
        "last_synced": last_synced,
    }


@mcp_server.tool()
def demand_routes(ctx: Context) -> dict[str, Any]:
    """Configured routing chain per demand class (ai_router.demand_routes),
    plus the rank-derived default chain used when none is configured. Same
    data as GET /admin/demand-routes."""
    _require_agent(ctx)
    from app.storage import get_demand_routes as _get_demand_routes

    try:
        routes = _get_demand_routes()
    except Exception:
        routes = {}
    try:
        registry = load_registry_with_db_health()
        healthy = registry.healthy_for_capability("text")
    except Exception:
        healthy = []
    defaults = {demand: [model.id for model in default_chain(healthy, demand, performance=model_performance_cached())][:8] for demand in DEMANDS}
    return {
        "demands": list(DEMANDS),
        "info": DEMAND_INFO,
        "routes": {demand: routes.get(demand, []) for demand in DEMANDS},
        "defaults": defaults,
        "virtual_models": VIRTUAL_MODELS,
    }


@mcp_server.tool()
def list_agents(ctx: Context, days: int = 30) -> dict[str, Any]:
    """Registered agents with usage stats over the trailing `days`. API keys
    are masked (first 4 chars only) — this never reveals a usable key. Same
    data as GET /admin/agents (minus avatar images)."""
    _require_agent(ctx)
    from app.storage import list_agents_with_usage

    try:
        agents = list_agents_with_usage(max(1, min(days, 365)))
    except Exception:
        return {"agents": [], "source": "db_unavailable"}
    for agent in agents:
        agent["api_key_masked"] = mask_secret(agent.pop("api_key", "") or "")
        agent.pop("avatar_data_url", None)
    return {"agents": agents, "source": "database"}


@mcp_server.tool()
def delete_agent(ctx: Context, name: str, confirm: bool = False) -> dict[str, Any]:
    """Permanently delete an agent (its API key stops working immediately;
    its model/budget controls are gone). The one deliberately destructive
    tool on this otherwise read-only surface — call it once with confirm
    left False to see who you're about to delete, then again with
    confirm=True to actually do it. Same effect as
    DELETE /admin/agents/{name}. Admin agents only."""
    _require_admin(ctx)
    from app.storage import delete_agent as _delete_agent
    from app.storage import list_agents_with_usage

    if not confirm:
        try:
            agents = {agent["name"]: agent for agent in list_agents_with_usage(days=1)}
        except Exception:
            agents = {}
        target = agents.get(name)
        if target is None:
            return {"status": "not_found", "agent": name}
        return {
            "status": "confirm_required",
            "agent": name,
            "kind": target.get("kind"),
            "description": target.get("description"),
            "message": f"This permanently deletes agent {name!r} and revokes its API key. Call delete_agent again with confirm=True to proceed.",
        }
    try:
        deleted = _delete_agent(name)
    except Exception as exc:
        raise ToolError(f"Failed to delete agent {name!r}: {exc}") from exc
    if not deleted:
        return {"status": "not_found", "agent": name}
    return {"status": "deleted", "agent": name}


@mcp_server.tool()
def recent_routes(ctx: Context, limit: int = 25, agent: str = "") -> dict[str, Any]:
    """Recent per-request routing log (the dashboard's Messages page),
    including which model was selected, demand class, tokens/cost, and the
    context-safe-routing diagnostics (context_window/budget/action). Same
    data as GET /admin/routes/recent."""
    _require_agent(ctx)
    from app.storage import recent_route_events

    try:
        safe_limit = max(1, min(limit, 100))
        return {"routes": recent_route_events(safe_limit, agent_name=agent.strip() or None)}
    except Exception:
        return {"routes": []}


@mcp_server.tool()
def usage_summary(ctx: Context, days: int = 30, agent: str = "") -> dict[str, Any]:
    """Aggregated message/token/cost usage over the trailing `days`,
    optionally for one agent. Same data as GET /admin/usage."""
    _require_agent(ctx)
    from app.storage import usage_summary as _usage_summary

    try:
        return _usage_summary(max(1, min(days, 365)), agent_name=agent.strip() or None)
    except Exception:
        return {"days": days, "totals": {"messages": 0, "tokens": 0, "cost": 0.0}, "daily": [], "by_model": [], "by_demand": []}


@mcp_server.tool()
def settings_overview(ctx: Context) -> dict[str, Any]:
    """Current values of the three request-path toggles: context
    compaction, context truncation, and the response cache. Same data as
    GET /admin/settings/context-compaction + context-truncation +
    response-cache combined."""
    _require_agent(ctx)
    from app.main import (
        context_compaction_enabled,
        context_truncation_enabled,
        context_truncation_max_tokens,
        context_truncation_trigger_percent,
        response_cache_enabled,
        response_cache_ttl_seconds,
    )

    try:
        compaction = {"enabled": context_compaction_enabled()}
    except Exception:
        compaction = {"enabled": True}
    try:
        truncation = {
            "enabled": context_truncation_enabled(),
            "max_tokens": context_truncation_max_tokens(),
            "trigger_percent": context_truncation_trigger_percent(),
        }
    except Exception:
        truncation = {"enabled": False, "max_tokens": 32000, "trigger_percent": 80}
    try:
        response_cache = {"enabled": response_cache_enabled(), "ttl_seconds": response_cache_ttl_seconds()}
    except Exception:
        response_cache = {"enabled": False, "ttl_seconds": 300}
    return {"context_compaction": compaction, "context_truncation": truncation, "response_cache": response_cache}


# --- Administration tools (2026-09-28) --------------------------------------------------------
#
# Marcelo: an MCP to administer ForgeRouter -- add a provider, ask for the model scan -- not to
# consume LLMs. Each tool calls the very same /admin/* handler function the dashboard uses (no
# second implementation of the logic, and require_admin() runs again inside it), with a request
# carrying the caller's own token. Not an HTTP loopback: this app runs one uvicorn worker, and a
# request to itself from inside a request could deadlock. The handlers are synchronous and a scan
# can take minutes, so they run on a worker thread to keep the event loop free.
#
# Provider API keys never travel through MCP: `add_provider` takes only the NAME of the
# environment variable holding the key (api_key_env); the value goes in ForgeRouter's .env.


def _admin_request(ctx: Context):
    from starlette.requests import Request

    token = _bearer(ctx)
    return Request({
        "type": "http", "method": "POST", "path": "/mcp", "query_string": b"",
        "headers": [(b"authorization", f"Bearer {token}".encode())],
    })


def _unwrap(result: Any) -> dict[str, Any]:
    from fastapi.responses import JSONResponse
    import json as _json

    if isinstance(result, JSONResponse):
        try:
            body = _json.loads(result.body)
            message = (body.get("error") or {}).get("message") or body
        except Exception:
            message = result.body.decode(errors="replace")[:500]
        raise ToolError(f"ForgeRouter refused ({result.status_code}): {message}")
    return result


async def _run_admin(ctx: Context, handler_name: str, *args: Any) -> dict[str, Any]:
    import anyio
    import app.main as main

    _require_admin(ctx)
    request = _admin_request(ctx)
    handler = getattr(main, handler_name)
    result = await anyio.to_thread.run_sync(lambda: handler(*args, request=request))
    return _unwrap(result)


@mcp_server.tool()
async def discover_provider_models(
    ctx: Context,
    provider_name: str = "",
    base_url: str = "",
    api_key_env: str = "",
    api_format: str = "",
    scan: bool = True,
) -> dict[str, Any]:
    """Discover the models a provider serves (and, with scan=True, health-check the free ones)
    without saving anything. For a stored provider pass provider_name; for a new one pass
    base_url + api_key_env (the NAME of the env var holding its key) + api_format
    (openai | anthropic). Same as POST /admin/providers/discover-models. Admin agents only."""
    import app.main as main

    payload = main.DiscoverModelsPayload(
        provider_name=provider_name.strip(), base_url=base_url.strip(),
        api_key_env=api_key_env.strip(), api_format=api_format.strip(), scan=scan,
    )
    return await _run_admin(ctx, "admin_provider_discover_models", payload)


@mcp_server.tool()
async def add_provider(
    ctx: Context,
    name: str,
    base_url: str,
    api_key_env: str = "",
    tier: int = 2,
    access_type: str = "api_key",
    cost_type: str = "free",
    api_format: str = "openai",
    only_healthy: bool = True,
) -> dict[str, Any]:
    """Register (or update) a provider: discovers its models, keeps the healthy ones enabled
    (all of them with only_healthy=False), saves it and syncs the agents' model lists.
    The key is never passed here -- put it in ForgeRouter's .env under `api_key_env` first
    (access_type=local needs none). access_type: api_key | subscription | local;
    cost_type: free | paid; api_format: openai | anthropic.
    Same as the dashboard's detect + save (discover-models, then PUT /admin/providers/{name}).
    Admin agents only."""
    import os
    import app.main as main

    _require_admin(ctx)  # before anything else -- a non-admin must not learn which env vars exist
    name = name.strip()
    api_key_env = api_key_env.strip()
    if not name or not base_url.strip():
        raise ToolError("name and base_url are required")
    if access_type != "local" and not api_key_env:
        raise ToolError("api_key_env is required (the NAME of the env var in ForgeRouter's .env holding the key)")
    if api_key_env and not os.environ.get(api_key_env):
        raise ToolError(
            f"{api_key_env} is not set in ForgeRouter's environment -- add it to ForgeRouter's .env "
            "and recreate the container, then call add_provider again"
        )
    discovered = await _run_admin(
        ctx, "admin_provider_discover_models",
        main.DiscoverModelsPayload(base_url=base_url.strip(), api_key_env=api_key_env, api_format=api_format, scan=True),
    )
    models = []
    for model in discovered.get("models") or []:
        healthy = (model.get("health") or {}).get("status") == "healthy"
        models.append(main.ProviderModelPayload(
            id=f"{name}/{model['id']}", provider_model=model["id"],
            capabilities=model.get("capabilities") or ["text"],
            enabled=healthy or not only_healthy, health=model.get("health"),
        ))
    if not models:
        raise ToolError(f"No models discovered at {base_url} -- nothing to register")
    payload = main.ProviderPayload(
        name=name, tier=tier, base_url=base_url.strip(), api_key_env=api_key_env,
        access_type=access_type, cost_type=cost_type, api_format=api_format, models=models,
    )
    saved = await _run_admin(ctx, "admin_provider_upsert", name, payload)
    return {
        "provider": name,
        "saved": saved,
        "models": len(models),
        "enabled_models": sum(1 for m in models if m.enabled),
        "discovery": {k: discovered.get(k) for k in ("total", "healthy", "excluded_paid")},
    }


@mcp_server.tool()
async def validate_provider(ctx: Context, name: str) -> dict[str, Any]:
    """Validate a stored provider: credential check plus a real chat completion against each
    enabled model; persists the health results. Same as POST /admin/providers/{name}/validate.
    Admin agents only."""
    return await _run_admin(ctx, "admin_provider_validate", name.strip())


@mcp_server.tool()
async def rescan_providers(ctx: Context) -> dict[str, Any]:
    """Run the LLM health scan across every registered model, switch unhealthy models off (and
    recovered ones back on) and resync the agents' model lists. Can take minutes. Same as
    POST /admin/providers/rescan. Admin agents only."""
    return await _run_admin(ctx, "admin_provider_rescan")


@mcp_server.tool()
async def resync_providers(ctx: Context) -> dict[str, Any]:
    """Full sync: re-discover, catalog and health-scan the free models of every enabled provider
    (new models appear, dead ones go). Slower than rescan_providers. Same as
    POST /admin/providers/resync. Admin agents only."""
    return await _run_admin(ctx, "admin_provider_resync")


@mcp_server.tool()
async def sync_pricing(ctx: Context) -> dict[str, Any]:
    """Refresh the model price catalog and every provider's pricing. Same as
    POST /admin/pricing/sync. Admin agents only."""
    return await _run_admin(ctx, "admin_pricing_sync")


@mcp_server.tool()
async def remove_provider(ctx: Context, name: str, confirm: bool = False) -> dict[str, Any]:
    """Permanently remove a provider and its models (agents lose those models). Call once with
    confirm left False to see what would go, then again with confirm=True. Same as
    DELETE /admin/providers/{name}. Admin agents only."""
    _require_admin(ctx)
    name = name.strip()
    if not confirm:
        from app.registry import load_registry_with_db_health

        try:
            registry = load_registry_with_db_health()
            models = [m.id for m in registry.models if m.provider == name]
        except Exception:
            models = []
        return {
            "status": "confirm_required",
            "provider": name,
            "models": models,
            "message": f"This permanently removes provider {name!r} and its {len(models)} model(s). "
                       "Call remove_provider again with confirm=True to proceed.",
        }
    return await _run_admin(ctx, "admin_provider_delete", name)
