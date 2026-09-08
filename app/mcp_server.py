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
        "Read-only inspection of ForgeRouter's providers, models and agents. "
        "Requires the same Bearer token (agent API key or dashboard session) "
        "as ForgeRouter's own /admin/* API — no separate credential."
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
    DELETE /admin/agents/{name}."""
    _require_agent(ctx)
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
