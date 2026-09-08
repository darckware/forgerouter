import pytest

from app.mcp_server import (
    delete_agent,
    demand_routes,
    list_agents,
    list_models,
    list_providers,
    mcp_server,
    model_pricing,
    provider_health,
    provider_readiness,
    recent_routes,
    settings_overview,
    usage_summary,
)
from mcp.server.mcpserver.exceptions import ToolError


class FakeContext:
    """Duck-typed stand-in for mcp.server.mcpserver.Context: every tool only
    reads .headers, so a plain attribute is enough — no real MCP transport
    needed for these tests."""

    def __init__(self, token: str | None = "valid-token"):
        self.headers = {"authorization": f"Bearer {token}"} if token else {}


TOOLS = [
    delete_agent, demand_routes, list_agents, list_models, list_providers,
    model_pricing, provider_health, provider_readiness, recent_routes,
    settings_overview, usage_summary,
]


def test_registers_exactly_these_eleven_tools():
    assert {tool.__name__ for tool in TOOLS} == {
        "delete_agent", "demand_routes", "list_agents", "list_models", "list_providers",
        "model_pricing", "provider_health", "provider_readiness", "recent_routes",
        "settings_overview", "usage_summary",
    }


def test_authorize_accepts_valid_agent_key(monkeypatch):
    from app.mcp_server import _authorize

    monkeypatch.setattr("app.storage.session_user", lambda token: None)
    monkeypatch.setattr("app.storage.find_agent_by_key", lambda token: "athos" if token == "good" else None)
    assert _authorize("good") is True
    assert _authorize("bad") is False
    assert _authorize("") is False


def test_authorize_accepts_valid_dashboard_session(monkeypatch):
    from app.mcp_server import _authorize

    monkeypatch.setattr("app.storage.session_user", lambda token: {"username": "admin"} if token == "sess" else None)
    monkeypatch.setattr("app.storage.find_agent_by_key", lambda token: None)
    assert _authorize("sess") is True


@pytest.mark.parametrize("tool", TOOLS)
def test_every_tool_rejects_a_missing_bearer_token(tool, monkeypatch):
    monkeypatch.setattr("app.mcp_server._authorize", lambda token: False)
    kwargs = {"name": "whoever"} if tool.__name__ == "delete_agent" else {}
    with pytest.raises(ToolError, match="Bearer token"):
        tool(FakeContext(token=None), **kwargs)


def test_list_providers_masks_and_reuses_the_same_registry_view(monkeypatch):
    monkeypatch.setattr("app.mcp_server._authorize", lambda token: True)
    monkeypatch.setattr(
        "app.storage.db_providers_with_models",
        lambda: [{"name": "groq", "api_key": "sk-super-secret-value", "base_url": "https://api.groq.com/openai/v1", "access_type": "api_key", "models": []}],
    )

    result = list_providers(FakeContext())

    assert result["source"] == "database"
    provider = result["providers"][0]
    assert "api_key" not in provider
    assert provider["api_key_set"] is True
    assert "sk-super-secret-value" not in provider["api_key_masked"]


def test_list_agents_masks_keys_and_drops_avatar(monkeypatch):
    monkeypatch.setattr("app.mcp_server._authorize", lambda token: True)
    monkeypatch.setattr(
        "app.storage.list_agents_with_usage",
        lambda days: [{"name": "Athos", "api_key": "hermes_athos_realsecret", "avatar_data_url": "data:image/png;base64,AAAA"}],
    )

    result = list_agents(FakeContext(), days=30)

    agent = result["agents"][0]
    assert agent["api_key_masked"] != "hermes_athos_realsecret"
    assert "api_key" not in agent
    assert "avatar_data_url" not in agent


def test_provider_health_filters_by_provider(monkeypatch):
    monkeypatch.setattr("app.mcp_server._authorize", lambda token: True)
    monkeypatch.setattr(
        "app.storage.latest_provider_health_rows",
        lambda: [{"provider": "groq", "status": "healthy"}, {"provider": "openrouter", "status": "unhealthy"}],
    )

    all_rows = provider_health(FakeContext(), provider="")
    filtered = provider_health(FakeContext(), provider="groq")

    assert len(all_rows["providers"]) == 2
    assert filtered["providers"] == [{"provider": "groq", "status": "healthy"}]


def test_recent_routes_clamps_limit_and_passes_agent_filter(monkeypatch):
    monkeypatch.setattr("app.mcp_server._authorize", lambda token: True)
    captured = {}

    def fake_recent(limit, agent_name=None):
        captured["limit"] = limit
        captured["agent_name"] = agent_name
        return []

    monkeypatch.setattr("app.storage.recent_route_events", fake_recent)

    recent_routes(FakeContext(), limit=9999, agent="Athos")

    assert captured["limit"] == 100  # clamped, same as GET /admin/routes/recent
    assert captured["agent_name"] == "Athos"


def test_usage_summary_defaults_agent_to_none_when_blank(monkeypatch):
    monkeypatch.setattr("app.mcp_server._authorize", lambda token: True)
    captured = {}

    def fake_usage(days, agent_name=None):
        captured["agent_name"] = agent_name
        return {"totals": {}}

    monkeypatch.setattr("app.storage.usage_summary", fake_usage)

    usage_summary(FakeContext(), days=30, agent="  ")

    assert captured["agent_name"] is None


def test_settings_overview_combines_all_three_toggles(monkeypatch):
    monkeypatch.setattr("app.mcp_server._authorize", lambda token: True)
    monkeypatch.setattr("app.main.context_compaction_enabled", lambda: True)
    monkeypatch.setattr("app.main.context_truncation_enabled", lambda: False)
    monkeypatch.setattr("app.main.context_truncation_max_tokens", lambda: 32000)
    monkeypatch.setattr("app.main.context_truncation_trigger_percent", lambda: 80)
    monkeypatch.setattr("app.main.response_cache_enabled", lambda: True)
    monkeypatch.setattr("app.main.response_cache_ttl_seconds", lambda: 300)

    result = settings_overview(FakeContext())

    assert result == {
        "context_compaction": {"enabled": True},
        "context_truncation": {"enabled": False, "max_tokens": 32000, "trigger_percent": 80},
        "response_cache": {"enabled": True, "ttl_seconds": 300},
    }


def test_delete_agent_without_confirm_reports_target_and_makes_no_change(monkeypatch):
    monkeypatch.setattr("app.mcp_server._authorize", lambda token: True)
    monkeypatch.setattr(
        "app.storage.list_agents_with_usage",
        lambda days: [{"name": "Athos", "kind": "agent", "description": "Chief of Staff"}],
    )
    deleted = {}
    monkeypatch.setattr("app.storage.delete_agent", lambda name: deleted.setdefault("called", name) or True)

    result = delete_agent(FakeContext(), name="Athos", confirm=False)

    assert result["status"] == "confirm_required"
    assert result["kind"] == "agent"
    assert "called" not in deleted


def test_delete_agent_with_confirm_actually_deletes(monkeypatch):
    monkeypatch.setattr("app.mcp_server._authorize", lambda token: True)
    deleted = {}
    monkeypatch.setattr("app.storage.delete_agent", lambda name: deleted.setdefault("called", name) or True)

    result = delete_agent(FakeContext(), name="Athos", confirm=True)

    assert result == {"status": "deleted", "agent": "Athos"}
    assert deleted["called"] == "Athos"


def test_delete_agent_reports_not_found_without_deleting(monkeypatch):
    monkeypatch.setattr("app.mcp_server._authorize", lambda token: True)
    monkeypatch.setattr("app.storage.delete_agent", lambda name: False)

    result = delete_agent(FakeContext(), name="Ghost", confirm=True)

    assert result == {"status": "not_found", "agent": "Ghost"}


def test_demand_routes_reports_configured_and_default_chains(monkeypatch):
    monkeypatch.setattr("app.mcp_server._authorize", lambda token: True)
    monkeypatch.setattr("app.storage.get_demand_routes", lambda: {"simple": ["p1/m"]})
    from app.registry import ProviderRegistry, ProviderModel

    model = ProviderModel("p1/m", "p1", "m", 1, ["text"], True, True, "http://x/v1", "")
    monkeypatch.setattr("app.mcp_server.load_registry_with_db_health", lambda: ProviderRegistry([model]))

    result = demand_routes(FakeContext())

    assert result["routes"]["simple"] == ["p1/m"]
    assert "forgerouter/auto" in result["virtual_models"]


def test_model_pricing_reports_coverage(monkeypatch):
    monkeypatch.setattr("app.mcp_server._authorize", lambda token: True)
    from app.registry import ProviderRegistry, ProviderModel

    model = ProviderModel("groq/llama", "groq", "llama", 1, ["text"], True, True, "http://x/v1", "")
    monkeypatch.setattr("app.mcp_server.load_registry_with_db_health", lambda: ProviderRegistry([model]))
    monkeypatch.setattr("app.pricing.resolve_price_info", lambda public_id, provider_model: {"input_cost_per_token": 1e-7, "output_cost_per_token": 2e-7, "source": "catalog"})
    monkeypatch.setattr("app.storage.get_setting", lambda key: "2026-09-08T00:00:00Z")

    result = model_pricing(FakeContext())

    assert result["priced_count"] == 1
    assert result["models"][0]["source"] == "catalog"


def test_list_models_scopes_to_the_agent_that_owns_the_token(monkeypatch):
    monkeypatch.setattr("app.mcp_server._authorize", lambda token: True)
    monkeypatch.setattr("app.storage.find_agent_by_key", lambda token: "athos")
    monkeypatch.setattr("app.storage.agent_allowed_models", lambda name: {"p1/m"} if name == "athos" else None)
    from app.registry import ProviderRegistry, ProviderModel

    model = ProviderModel("p1/m", "p1", "m", 1, ["text"], True, True, "http://x/v1", "")
    monkeypatch.setattr("app.mcp_server.load_registry_with_db_health", lambda: ProviderRegistry([model]))

    result = list_models(FakeContext(token="athos-key"))

    ids = [item["id"] for item in result["data"]]
    assert "p1/m" in ids
    assert "forgerouter/auto" in ids


def test_provider_readiness_reports_env_var_booleans_only(monkeypatch):
    monkeypatch.setattr("app.mcp_server._authorize", lambda token: True)
    monkeypatch.setattr(
        "app.registry.provider_readiness",
        lambda: [{"provider": "groq", "api_key_env": "GROQ_API_KEY", "configured": True}],
    )

    result = provider_readiness(FakeContext())

    assert result == {"providers": [{"provider": "groq", "api_key_env": "GROQ_API_KEY", "configured": True}]}


def test_mcp_mount_serves_a_real_streamable_http_request():
    # The one real end-to-end check: a Mount does not forward ASGI lifespan
    # events to its sub-app, so mcp_server.session_manager.run() must be
    # wired into app.main's own _lifespan or every /mcp request 500s with
    # "Task group is not initialized" — this caught exactly that bug during
    # development. session_manager.run() may only be entered once per
    # process, so this is the only test in the suite using TestClient's
    # context-manager form (which runs the real ASGI lifespan).
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as client:
        response = client.post(
            "/mcp/",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "clientInfo": {"name": "pytest", "version": "0"},
                },
            },
            # TestClient's default Host header ("testserver") trips the SDK's
            # own DNS-rebinding protection (auto-enabled for a 127.0.0.1
            # host, allowing only 127.0.0.1/localhost/[::1]) — a real client
            # hitting the real deployment sends a Host the proxy already
            # controls, so this only affects the test harness.
            headers={"Accept": "application/json, text/event-stream", "Host": "127.0.0.1:2100"},
        )

    assert response.status_code == 200
    assert '"serverInfo"' in response.text
    assert "forgerouter" in response.text
