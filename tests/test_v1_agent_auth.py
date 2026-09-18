from fastapi.testclient import TestClient

from app.main import app
from app.registry import ProviderModel, ProviderRegistry

client = TestClient(app)


def setup_routing(monkeypatch):
    model = ProviderModel("p1/model-a", "p1", "model-a", 1, ["text"], True, True, "http://first/v1", "")
    monkeypatch.setattr("app.main.load_registry_with_db_health", lambda: ProviderRegistry([model]))
    monkeypatch.setattr("app.main.chat_completion", lambda model, payload: (200, {"choices": [{"message": {"content": "OK"}}]}))
    monkeypatch.setattr("app.main.persist_route_event", lambda *args, **kwargs: None)
    monkeypatch.setattr("app.main.get_demand_routes", lambda: {})
    monkeypatch.setattr("app.main.agent_allowed_models", lambda name: None)
    # Known and above the 64k virtual-route floor — this file is about agent
    # key auth, not context fit.
    monkeypatch.setattr("app.context_policy.context_window", lambda *a: 200_000)


def test_v1_rejects_missing_key_once_agents_exist(monkeypatch):
    setup_routing(monkeypatch)
    monkeypatch.setattr("app.main.has_any_agent", lambda: True)
    monkeypatch.setattr("app.main.find_agent_by_key", lambda key: None)

    no_key = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
    bad_key = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer wrong"},
        json={"messages": [{"role": "user", "content": "hi"}]},
    )

    assert no_key.status_code == 401
    assert no_key.json()["error"]["type"] == "invalid_agent_key"
    assert bad_key.status_code == 401


def test_v1_serves_valid_agent_key(monkeypatch):
    setup_routing(monkeypatch)
    monkeypatch.setattr("app.main.has_any_agent", lambda: True)
    monkeypatch.setattr("app.main.find_agent_by_key", lambda key: "athos" if key == "hermes_k" else None)

    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer hermes_k"},
        json={"messages": [{"role": "user", "content": "hi"}]},
    )

    assert response.status_code == 200


def test_v1_stays_open_with_no_agents_or_broken_store(monkeypatch):
    setup_routing(monkeypatch)
    # First-time setup: no agents registered yet.
    monkeypatch.setattr("app.main.has_any_agent", lambda: False)
    assert client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]}).status_code == 200

    # Agent store down during the key lookup: routing must not break.
    def boom(key):
        raise RuntimeError("db down")

    monkeypatch.setattr("app.main.find_agent_by_key", boom)
    monkeypatch.setattr("app.main.has_any_agent", lambda: True)
    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer hermes_k"},
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 200
