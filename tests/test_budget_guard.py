from fastapi.testclient import TestClient

from app.main import app
from app.registry import ProviderModel, ProviderRegistry

client = TestClient(app)


def _model() -> ProviderModel:
    return ProviderModel("p1/model-a", "p1", "model-a", 1, ["text"], True, True, "http://first/v1", "")


def _mock_success_chat(monkeypatch):
    monkeypatch.setattr("app.main.load_registry_with_db_health", lambda: ProviderRegistry([_model()]))
    monkeypatch.setattr("app.main.chat_completion", lambda model, payload: (200, {"choices": [{"message": {"content": "OK"}}]}))
    monkeypatch.setattr("app.main.persist_route_event", lambda *args, **kwargs: None)
    monkeypatch.setattr("app.main.find_agent_by_key", lambda key: "athos" if key == "hermes_k" else None)
    # Known and above the 64k virtual-route floor — these tests are about the
    # budget guard, not context fit.
    monkeypatch.setattr("app.context_policy.context_window", lambda *a: 200_000)


def test_no_budget_configured_routes_normally(monkeypatch):
    _mock_success_chat(monkeypatch)
    monkeypatch.setattr("app.main.get_agent_budget", lambda name: (None, "alert"))

    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer hermes_k"},
        json={"messages": [{"role": "user", "content": "hi"}]},
    )

    assert response.status_code == 200


def test_alert_mode_over_budget_still_routes(monkeypatch):
    _mock_success_chat(monkeypatch)
    monkeypatch.setattr("app.main.get_agent_budget", lambda name: (10.0, "alert"))
    monkeypatch.setattr("app.main.agent_month_spend", lambda name: 25.0)

    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer hermes_k"},
        json={"messages": [{"role": "user", "content": "hi"}]},
    )

    assert response.status_code == 200


def test_block_mode_under_budget_routes_normally(monkeypatch):
    _mock_success_chat(monkeypatch)
    monkeypatch.setattr("app.main.get_agent_budget", lambda name: (10.0, "block"))
    monkeypatch.setattr("app.main.agent_month_spend", lambda name: 3.0)

    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer hermes_k"},
        json={"messages": [{"role": "user", "content": "hi"}]},
    )

    assert response.status_code == 200


def test_block_mode_over_budget_rejects_with_429(monkeypatch):
    _mock_success_chat(monkeypatch)
    monkeypatch.setattr("app.main.get_agent_budget", lambda name: (10.0, "block"))
    monkeypatch.setattr("app.main.agent_month_spend", lambda name: 12.5)

    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer hermes_k"},
        json={"messages": [{"role": "user", "content": "hi"}]},
    )

    assert response.status_code == 429
    body = response.json()
    assert body["error"]["type"] == "budget_exceeded"
    assert "athos" in body["error"]["message"]


def test_budget_lookup_failure_fails_open(monkeypatch):
    # DB failures must never break routing — same rule as every other lookup here.
    _mock_success_chat(monkeypatch)

    def boom(name):
        raise RuntimeError("db unreachable")

    monkeypatch.setattr("app.main.get_agent_budget", boom)

    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer hermes_k"},
        json={"messages": [{"role": "user", "content": "hi"}]},
    )

    assert response.status_code == 200


def test_admin_agent_set_budget_requires_admin(monkeypatch):
    monkeypatch.setattr("app.main.has_any_agent", lambda: True)
    monkeypatch.setattr("app.main.find_agent_by_key", lambda key: None)

    response = client.put("/admin/agents/athos/budget", json={"limit_usd": 10.0, "action": "block"})

    assert response.status_code == 401


def test_admin_agent_set_budget_validates_action(monkeypatch):
    monkeypatch.setattr("app.main.has_any_agent", lambda: True)
    monkeypatch.setattr("app.main.find_agent_by_key", lambda key: "tester" if key == "secret" else None)

    response = client.put(
        "/admin/agents/athos/budget",
        headers={"Authorization": "Bearer secret"},
        json={"limit_usd": 10.0, "action": "delete-everything"},
    )

    assert response.status_code == 422
    assert response.json()["error"]["type"] == "invalid_budget_action"


def test_admin_agent_set_budget_rejects_negative_limit(monkeypatch):
    monkeypatch.setattr("app.main.has_any_agent", lambda: True)
    monkeypatch.setattr("app.main.find_agent_by_key", lambda key: "tester" if key == "secret" else None)

    response = client.put(
        "/admin/agents/athos/budget",
        headers={"Authorization": "Bearer secret"},
        json={"limit_usd": -5.0, "action": "alert"},
    )

    assert response.status_code == 422
    assert response.json()["error"]["type"] == "invalid_budget_limit"


def test_admin_agent_set_budget_saves_and_reports_missing_agent(monkeypatch):
    monkeypatch.setattr("app.main.has_any_agent", lambda: True)
    monkeypatch.setattr("app.main.find_agent_by_key", lambda key: "tester" if key == "secret" else None)
    monkeypatch.setattr("app.main.set_agent_budget", lambda name, limit_usd, action: name == "athos")

    ok = client.put(
        "/admin/agents/athos/budget",
        headers={"Authorization": "Bearer secret"},
        json={"limit_usd": 25.0, "action": "block"},
    )
    assert ok.status_code == 200
    assert ok.json() == {"status": "saved", "agent": "athos", "limit_usd": 25.0, "action": "block"}

    missing = client.put(
        "/admin/agents/ghost/budget",
        headers={"Authorization": "Bearer secret"},
        json={"limit_usd": 25.0, "action": "block"},
    )
    assert missing.status_code == 404
