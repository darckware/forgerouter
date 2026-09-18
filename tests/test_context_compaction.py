import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.registry import ProviderModel, ProviderRegistry

client = TestClient(app)


@pytest.fixture(autouse=True)
def _known_roomy_context_window(monkeypatch):
    # This file is about compaction, not context fit — give the synthetic
    # model a known window safely above the 64k virtual-route floor (the
    # default "auto" model would otherwise hard-exclude its uncatalogued window).
    monkeypatch.setattr("app.context_policy.context_window", lambda *a: 200_000)


def _model():
    return ProviderModel("p1/model-a", "p1", "model-a", 1, ["text"], True, True, "http://first/v1", "")


def test_chat_normalizes_messages_when_compaction_enabled(monkeypatch):
    monkeypatch.setattr("app.main.load_registry_with_db_health", lambda: ProviderRegistry([_model()]))
    monkeypatch.setattr("app.main.context_compaction_enabled", lambda: True)

    captured = {}

    def fake_chat_completion(model, payload):
        captured["messages"] = payload["messages"]
        return 200, {"choices": [{"message": {"content": "OK"}}]}

    monkeypatch.setattr("app.main.chat_completion", fake_chat_completion)
    monkeypatch.setattr("app.main.persist_route_event", lambda *args, **kwargs: None)

    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi   \n\n\n\nthere"}]},
    )

    assert response.status_code == 200
    assert captured["messages"][0]["content"] == "hi\n\nthere"


def test_chat_passes_raw_messages_when_compaction_disabled(monkeypatch):
    monkeypatch.setattr("app.main.load_registry_with_db_health", lambda: ProviderRegistry([_model()]))
    monkeypatch.setattr("app.main.context_compaction_enabled", lambda: False)

    captured = {}

    def fake_chat_completion(model, payload):
        captured["messages"] = payload["messages"]
        return 200, {"choices": [{"message": {"content": "OK"}}]}

    monkeypatch.setattr("app.main.chat_completion", fake_chat_completion)
    monkeypatch.setattr("app.main.persist_route_event", lambda *args, **kwargs: None)

    raw_content = "hi   \n\n\n\nthere"
    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": raw_content}]},
    )

    assert response.status_code == 200
    assert captured["messages"][0]["content"] == raw_content


def test_chat_persists_tokens_raw(monkeypatch):
    monkeypatch.setattr("app.main.load_registry_with_db_health", lambda: ProviderRegistry([_model()]))
    monkeypatch.setattr("app.main.context_compaction_enabled", lambda: True)
    monkeypatch.setattr("app.main.chat_completion", lambda model, payload: (200, {"choices": [{"message": {"content": "OK"}}]}))

    persisted = {}

    def fake_persist(*args, **kwargs):
        persisted.update(kwargs)

    monkeypatch.setattr("app.main.persist_route_event", fake_persist)

    response = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi there"}]})

    assert response.status_code == 200
    assert isinstance(persisted.get("tokens_raw"), int)
    assert isinstance(persisted.get("tokens_compacted"), int)
    assert persisted["tokens_raw"] > 0


def test_chat_tokens_saved_reflects_compaction_when_enabled(monkeypatch):
    monkeypatch.setattr("app.main.load_registry_with_db_health", lambda: ProviderRegistry([_model()]))
    monkeypatch.setattr("app.main.context_compaction_enabled", lambda: True)
    monkeypatch.setattr("app.main.chat_completion", lambda model, payload: (200, {"choices": [{"message": {"content": "OK"}}]}))

    persisted = {}
    monkeypatch.setattr("app.main.persist_route_event", lambda *a, **kw: persisted.update(kw))

    padded = "hi   \n\n\n\nthere"
    response = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": padded}]})

    assert response.status_code == 200
    assert persisted["tokens_compacted"] < persisted["tokens_raw"]


def test_chat_tokens_saved_is_zero_when_compaction_disabled(monkeypatch):
    monkeypatch.setattr("app.main.load_registry_with_db_health", lambda: ProviderRegistry([_model()]))
    monkeypatch.setattr("app.main.context_compaction_enabled", lambda: False)
    monkeypatch.setattr("app.main.chat_completion", lambda model, payload: (200, {"choices": [{"message": {"content": "OK"}}]}))

    persisted = {}
    monkeypatch.setattr("app.main.persist_route_event", lambda *a, **kw: persisted.update(kw))

    padded = "hi   \n\n\n\nthere"
    response = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": padded}]})

    assert response.status_code == 200
    assert persisted["tokens_compacted"] == persisted["tokens_raw"]


def test_chat_handles_tokenizer_failure_gracefully(monkeypatch):
    monkeypatch.setattr("app.main.load_registry_with_db_health", lambda: ProviderRegistry([_model()]))
    monkeypatch.setattr("app.main.context_compaction_enabled", lambda: True)
    monkeypatch.setattr("app.main.count_tokens", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("no encoding")))
    monkeypatch.setattr("app.main.chat_completion", lambda model, payload: (200, {"choices": [{"message": {"content": "OK"}}]}))
    monkeypatch.setattr("app.main.persist_route_event", lambda *args, **kwargs: None)

    response = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi there"}]})

    assert response.status_code == 200


def test_context_compaction_get_default(monkeypatch):
    monkeypatch.setattr("app.main.context_compaction_enabled", lambda: True)

    response = client.get("/admin/settings/context-compaction")

    assert response.status_code == 200
    assert response.json() == {"enabled": True}


def test_context_compaction_get_handles_db_failure(monkeypatch):
    def boom():
        raise RuntimeError("db unreachable")

    monkeypatch.setattr("app.main.context_compaction_enabled", boom)

    response = client.get("/admin/settings/context-compaction")

    assert response.status_code == 200
    assert response.json() == {"enabled": True}


def test_context_compaction_post_requires_admin(monkeypatch):
    monkeypatch.setattr("app.main.has_any_agent", lambda: True)
    monkeypatch.setattr("app.main.find_agent_by_key", lambda key: None)
    monkeypatch.setattr("app.main.session_user", lambda token: None)

    response = client.post("/admin/settings/context-compaction", json={"enabled": False})

    assert response.status_code == 401


def test_context_compaction_post_persists(monkeypatch):
    monkeypatch.setattr("app.main.has_any_agent", lambda: False)
    saved = {}
    monkeypatch.setattr("app.main.set_context_compaction_enabled", lambda enabled: saved.update(enabled=enabled))

    response = client.post("/admin/settings/context-compaction", json={"enabled": False})

    assert response.status_code == 200
    assert response.json() == {"status": "saved", "enabled": False}
    assert saved == {"enabled": False}
