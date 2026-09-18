import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.registry import ProviderModel, ProviderRegistry

client = TestClient(app)

_PAYLOAD = {"messages": [{"role": "user", "content": "hi"}]}


@pytest.fixture(autouse=True)
def _known_roomy_context_window(monkeypatch):
    # This file is about the response cache, not context fit — give the
    # synthetic model a known window safely above the 64k virtual-route
    # floor (the default "auto" model would otherwise hard-exclude its
    # uncatalogued window).
    monkeypatch.setattr("app.context_policy.context_window", lambda *a: 200_000)


def _one_model():
    return ProviderModel("p1/model-a", "p1", "model-a", 1, ["text"], True, True, "http://first/v1", "")


def test_cache_disabled_by_default_calls_provider_each_time(monkeypatch):
    monkeypatch.setattr("app.main.load_registry_with_db_health", lambda: ProviderRegistry([_one_model()]))
    monkeypatch.setattr("app.main.persist_route_event", lambda *args, **kwargs: None)

    calls = []
    def fake_chat_completion(model, payload):
        calls.append(model.id)
        return 200, {"choices": [{"message": {"content": "OK"}}]}
    monkeypatch.setattr("app.main.chat_completion", fake_chat_completion)

    client.post("/v1/chat/completions", json=_PAYLOAD)
    response = client.post("/v1/chat/completions", json=_PAYLOAD)

    assert calls == ["p1/model-a", "p1/model-a"]
    assert response.headers["x-proxyrouter-cache"] == "off"


def test_cache_hit_skips_provider_call_when_enabled(monkeypatch):
    monkeypatch.setattr("app.main.load_registry_with_db_health", lambda: ProviderRegistry([_one_model()]))
    monkeypatch.setattr("app.main.persist_route_event", lambda *args, **kwargs: None)
    monkeypatch.setattr("app.main.response_cache_enabled", lambda: True)
    monkeypatch.setattr("app.main.response_cache_ttl_seconds", lambda: 60)

    calls = []
    def fake_chat_completion(model, payload):
        calls.append(model.id)
        return 200, {"choices": [{"message": {"content": "OK"}}]}
    monkeypatch.setattr("app.main.chat_completion", fake_chat_completion)

    first = client.post("/v1/chat/completions", json=_PAYLOAD)
    second = client.post("/v1/chat/completions", json=_PAYLOAD)

    assert calls == ["p1/model-a"]  # second request never reached the provider
    assert first.headers["x-proxyrouter-cache"] == "miss"
    assert second.headers["x-proxyrouter-cache"] == "hit"
    assert second.json() == first.json()


def test_cache_differentiates_by_message_content(monkeypatch):
    monkeypatch.setattr("app.main.load_registry_with_db_health", lambda: ProviderRegistry([_one_model()]))
    monkeypatch.setattr("app.main.persist_route_event", lambda *args, **kwargs: None)
    monkeypatch.setattr("app.main.response_cache_enabled", lambda: True)
    monkeypatch.setattr("app.main.response_cache_ttl_seconds", lambda: 60)

    calls = []
    def fake_chat_completion(model, payload):
        calls.append(model.id)
        return 200, {"choices": [{"message": {"content": "OK"}}]}
    monkeypatch.setattr("app.main.chat_completion", fake_chat_completion)

    client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
    client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "bye"}]})

    assert calls == ["p1/model-a", "p1/model-a"]


def test_cache_header_off_bypasses_even_when_globally_enabled(monkeypatch):
    monkeypatch.setattr("app.main.load_registry_with_db_health", lambda: ProviderRegistry([_one_model()]))
    monkeypatch.setattr("app.main.persist_route_event", lambda *args, **kwargs: None)
    monkeypatch.setattr("app.main.response_cache_enabled", lambda: True)
    monkeypatch.setattr("app.main.response_cache_ttl_seconds", lambda: 60)

    calls = []
    def fake_chat_completion(model, payload):
        calls.append(model.id)
        return 200, {"choices": [{"message": {"content": "OK"}}]}
    monkeypatch.setattr("app.main.chat_completion", fake_chat_completion)

    client.post("/v1/chat/completions", json=_PAYLOAD, headers={"X-Proxyrouter-Cache": "off"})
    client.post("/v1/chat/completions", json=_PAYLOAD, headers={"X-Proxyrouter-Cache": "off"})

    assert calls == ["p1/model-a", "p1/model-a"]


def test_cache_header_on_enables_even_when_globally_disabled(monkeypatch):
    monkeypatch.setattr("app.main.load_registry_with_db_health", lambda: ProviderRegistry([_one_model()]))
    monkeypatch.setattr("app.main.persist_route_event", lambda *args, **kwargs: None)
    monkeypatch.setattr("app.main.response_cache_enabled", lambda: False)
    monkeypatch.setattr("app.main.response_cache_ttl_seconds", lambda: 60)

    calls = []
    def fake_chat_completion(model, payload):
        calls.append(model.id)
        return 200, {"choices": [{"message": {"content": "OK"}}]}
    monkeypatch.setattr("app.main.chat_completion", fake_chat_completion)

    client.post("/v1/chat/completions", json=_PAYLOAD, headers={"X-Proxyrouter-Cache": "on"})
    client.post("/v1/chat/completions", json=_PAYLOAD, headers={"X-Proxyrouter-Cache": "on"})

    assert calls == ["p1/model-a"]


def test_streaming_requests_are_never_cached(monkeypatch):
    monkeypatch.setattr("app.main.load_registry_with_db_health", lambda: ProviderRegistry([_one_model()]))
    monkeypatch.setattr("app.main.persist_route_event", lambda *args, **kwargs: None)
    monkeypatch.setattr("app.main.response_cache_enabled", lambda: True)
    monkeypatch.setattr("app.main.response_cache_ttl_seconds", lambda: 60)

    calls = []
    def fake_chat_completion(model, payload):
        calls.append(model.id)
        def chunk_gen():
            yield b"data: {\"choices\": [{\"delta\": {\"content\": \"hi\"}}]}\n\n"
        return 200, chunk_gen()
    monkeypatch.setattr("app.main.chat_completion", fake_chat_completion)

    client.post("/v1/chat/completions", json={**_PAYLOAD, "stream": True})
    client.post("/v1/chat/completions", json={**_PAYLOAD, "stream": True})

    assert calls == ["p1/model-a", "p1/model-a"]
