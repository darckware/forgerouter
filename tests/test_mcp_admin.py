"""ForgeRouter administration over MCP (2026-09-28): admin allow-list, and tools that reuse
the /admin/* handlers with the caller's own token."""
import asyncio

import pytest
from fastapi.responses import JSONResponse
from mcp.server.mcpserver.exceptions import ToolError

import app.main as main
from app import admin_policy
from app.mcp_server import (
    add_provider,
    discover_provider_models,
    remove_provider,
    rescan_providers,
    resync_providers,
    sync_pricing,
    validate_provider,
)


class FakeContext:
    def __init__(self, token="athos-key"):
        self.headers = {"authorization": f"Bearer {token}"} if token else {}


def run(coro):
    return asyncio.run(coro)


# --- policy ---------------------------------------------------------------------------------

def test_default_admins_are_athos_and_hephaestus(monkeypatch):
    monkeypatch.delenv(admin_policy.ADMIN_AGENTS_ENV, raising=False)
    assert admin_policy.admin_agent_names() == {"athos", "hephaestus"}
    assert admin_policy.is_admin_agent("Athos") and admin_policy.is_admin_agent("HEPHAESTUS")
    assert not admin_policy.is_admin_agent("Lara")
    assert not admin_policy.is_admin_agent(None)


def test_admins_come_from_the_env_when_set(monkeypatch):
    monkeypatch.setenv(admin_policy.ADMIN_AGENTS_ENV, " Athos , ")
    assert admin_policy.admin_agent_names() == {"athos"}
    monkeypatch.setenv(admin_policy.ADMIN_AGENTS_ENV, "")
    assert admin_policy.admin_agent_names() == set()


def test_is_admin_token_accepts_session_or_admin_agent_only(monkeypatch):
    monkeypatch.delenv(admin_policy.ADMIN_AGENTS_ENV, raising=False)
    monkeypatch.setattr("app.storage.session_user", lambda t: {"username": "marcelo"} if t == "sess" else None)
    monkeypatch.setattr("app.storage.find_agent_by_key", lambda t: {"k-athos": "Athos", "k-lara": "Lara"}.get(t))
    assert admin_policy.is_admin_token("sess")
    assert admin_policy.is_admin_token("k-athos")
    assert not admin_policy.is_admin_token("k-lara")
    assert not admin_policy.is_admin_token("")


def _request(token):
    from starlette.requests import Request

    return Request({"type": "http", "method": "GET", "path": "/admin", "query_string": b"",
                    "headers": [(b"authorization", f"Bearer {token}".encode())]})


def test_require_admin_gives_403_to_a_non_admin_agent(monkeypatch):
    monkeypatch.delenv(admin_policy.ADMIN_AGENTS_ENV, raising=False)
    monkeypatch.setattr(main, "session_user", lambda t: None)
    monkeypatch.setattr(main, "find_agent_by_key", lambda t: {"k-athos": "Athos", "k-lara": "Lara"}.get(t))
    monkeypatch.setattr(main, "has_any_agent", lambda: True)
    assert main.require_admin(_request("k-athos")) is None
    refused = main.require_admin(_request("k-lara"))
    assert refused.status_code == 403
    assert main.require_admin(_request("unknown")).status_code == 401


# --- tools ----------------------------------------------------------------------------------

@pytest.mark.parametrize("call", [
    lambda: rescan_providers(FakeContext("k-lara")),
    lambda: resync_providers(FakeContext("k-lara")),
    lambda: sync_pricing(FakeContext("k-lara")),
    lambda: validate_provider(FakeContext("k-lara"), name="groq"),
    lambda: discover_provider_models(FakeContext("k-lara"), provider_name="groq"),
    lambda: remove_provider(FakeContext("k-lara"), name="groq", confirm=True),
    lambda: add_provider(FakeContext("k-lara"), name="new", base_url="https://x", api_key_env="NEW_KEY"),
])
def test_every_admin_tool_refuses_a_non_admin_agent(call, monkeypatch):
    monkeypatch.setattr("app.admin_policy.is_admin_token", lambda t: t != "k-lara")
    with pytest.raises(ToolError, match="admin agents"):
        run(call())


def test_rescan_calls_the_same_handler_with_the_callers_token(monkeypatch):
    monkeypatch.setattr("app.admin_policy.is_admin_token", lambda t: True)
    seen = {}

    def fake_handler(request):
        seen["auth"] = request.headers.get("authorization")
        return {"summary": {"total": 3, "healthy": 2}}

    monkeypatch.setattr(main, "admin_provider_rescan", fake_handler)
    result = run(rescan_providers(FakeContext("athos-key")))
    assert result == {"summary": {"total": 3, "healthy": 2}}
    assert seen["auth"] == "Bearer athos-key"


def test_handler_error_response_becomes_a_tool_error(monkeypatch):
    monkeypatch.setattr("app.admin_policy.is_admin_token", lambda t: True)
    monkeypatch.setattr(
        main, "admin_provider_validate",
        lambda name, request: JSONResponse(status_code=404, content={"error": {"message": f"Provider not found: {name}"}}),
    )
    with pytest.raises(ToolError, match="404.*Provider not found: ghost"):
        run(validate_provider(FakeContext(), name="ghost"))


def test_add_provider_never_takes_a_key_and_requires_the_env_var(monkeypatch):
    monkeypatch.setattr("app.admin_policy.is_admin_token", lambda t: True)
    monkeypatch.delenv("NEW_PROVIDER_KEY", raising=False)
    with pytest.raises(ToolError, match="not set in ForgeRouter's environment"):
        run(add_provider(FakeContext(), name="new", base_url="https://api.new/v1", api_key_env="NEW_PROVIDER_KEY"))
    with pytest.raises(ToolError, match="api_key_env is required"):
        run(add_provider(FakeContext(), name="new", base_url="https://api.new/v1"))


def test_add_provider_discovers_saves_and_enables_only_healthy(monkeypatch):
    monkeypatch.setattr("app.admin_policy.is_admin_token", lambda t: True)
    monkeypatch.setenv("NEW_PROVIDER_KEY", "sk-test")
    saved = {}
    monkeypatch.setattr(main, "admin_provider_discover_models", lambda payload, request: {
        "models": [
            {"id": "good-1", "capabilities": ["text"], "health": {"status": "healthy"}},
            {"id": "bad-1", "capabilities": ["text"], "health": {"status": "unhealthy"}},
        ],
        "total": 2, "healthy": 1, "excluded_paid": 0,
    })

    def fake_upsert(name, payload, request):
        saved["name"], saved["payload"] = name, payload
        return {"status": "saved", "provider": name}

    monkeypatch.setattr(main, "admin_provider_upsert", fake_upsert)
    result = run(add_provider(FakeContext(), name="new", base_url="https://api.new/v1", api_key_env="NEW_PROVIDER_KEY"))
    assert result["models"] == 2 and result["enabled_models"] == 1
    payload = saved["payload"]
    assert payload.api_key == "" and payload.api_key_env == "NEW_PROVIDER_KEY"
    assert {m.id: m.enabled for m in payload.models} == {"new/good-1": True, "new/bad-1": False}


def test_remove_provider_needs_confirmation(monkeypatch):
    monkeypatch.setattr("app.admin_policy.is_admin_token", lambda t: True)
    called = {}
    monkeypatch.setattr(main, "admin_provider_delete", lambda name, request: called.setdefault("name", name) and {"status": "deleted", "provider": name})
    preview = run(remove_provider(FakeContext(), name="groq"))
    assert preview["status"] == "confirm_required" and "name" not in called
    assert run(remove_provider(FakeContext(), name="groq", confirm=True)) == {"status": "deleted", "provider": "groq"}
