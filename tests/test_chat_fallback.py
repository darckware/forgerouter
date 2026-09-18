import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.registry import ProviderModel, ProviderRegistry

client = TestClient(app)


@pytest.fixture(autouse=True)
def _known_roomy_context_window(monkeypatch):
    # This whole file is about fallback/retry mechanics on synthetic,
    # catalog-unknown model ids — not context fit. Since a virtual route
    # (the default "auto" model) now hard-excludes an uncatalogued window,
    # give every model here a known window safely above the 64k floor.
    monkeypatch.setattr("app.context_policy.context_window", lambda *a: 200_000)


def test_chat_falls_back_to_next_candidate(monkeypatch):
    first = ProviderModel("p1/model-a", "p1", "model-a", 1, ["text"], True, True, "http://first/v1", "")
    second = ProviderModel("p2/model-b", "p2", "model-b", 2, ["text"], True, True, "http://second/v1", "")
    monkeypatch.setattr("app.main.load_registry_with_db_health", lambda: ProviderRegistry([first, second]))

    calls = []
    def fake_chat_completion(model, payload):
        calls.append(model.id)
        if model.id == "p1/model-a":
            return 500, {"error": {"message": "fail"}}
        return 200, {"choices": [{"message": {"content": "OK"}}]}

    monkeypatch.setattr("app.main.chat_completion", fake_chat_completion)
    monkeypatch.setattr("app.main.persist_route_event", lambda *args, **kwargs: None)

    response = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})

    assert response.status_code == 200
    assert calls == ["p1/model-a", "p2/model-b"]
    assert response.headers["x-proxyrouter-model"] == "p2/model-b"


def test_chat_falls_back_on_200_with_error_body(monkeypatch):
    # A provider can return HTTP 200 with an error object instead of a real
    # completion (no "choices") — this must be treated like a provider error,
    # not forwarded to the caller as a success.
    first = ProviderModel("p1/model-a", "p1", "model-a", 1, ["text"], True, True, "http://first/v1", "")
    second = ProviderModel("p2/model-b", "p2", "model-b", 2, ["text"], True, True, "http://second/v1", "")
    monkeypatch.setattr("app.main.load_registry_with_db_health", lambda: ProviderRegistry([first, second]))

    calls = []
    def fake_chat_completion(model, payload):
        calls.append(model.id)
        if model.id == "p1/model-a":
            return 200, {"error": {"code": 400, "message": "Provider returned error", "metadata": {"error_type": "invalid_request"}}}
        return 200, {"choices": [{"message": {"content": "OK"}}]}

    monkeypatch.setattr("app.main.chat_completion", fake_chat_completion)
    monkeypatch.setattr("app.main.persist_route_event", lambda *args, **kwargs: None)

    unhealthy_calls = []
    monkeypatch.setattr("app.main.mark_runtime_failure_unhealthy", lambda model, status_code, error_message, cooldown_seconds=None: unhealthy_calls.append((model.id, status_code, error_message)))

    response = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})

    assert response.status_code == 200
    assert calls == ["p1/model-a", "p2/model-b"]
    assert response.headers["x-proxyrouter-model"] == "p2/model-b"
    assert unhealthy_calls == [("p1/model-a", 200, "runtime_silent_empty_choices")]


def test_chat_routes_to_requested_model_only(monkeypatch):
    first = ProviderModel("p1/model-a", "p1", "model-a", 1, ["text"], True, True, "http://first/v1", "")
    second = ProviderModel("p2/model-b", "p2", "model-b", 2, ["text"], True, True, "http://second/v1", "")
    monkeypatch.setattr("app.main.load_registry_with_db_health", lambda: ProviderRegistry([first, second]))

    calls = []

    def fake_chat_completion(model, payload):
        calls.append(model.id)
        return 200, {"choices": [{"message": {"content": "OK"}}]}

    monkeypatch.setattr("app.main.chat_completion", fake_chat_completion)
    monkeypatch.setattr("app.main.persist_route_event", lambda *args, **kwargs: None)

    response = client.post("/v1/chat/completions", json={"model": "p2/model-b", "messages": [{"role": "user", "content": "hi"}]})

    assert response.status_code == 200
    assert calls == ["p2/model-b"]


def test_chat_requested_model_falls_back_on_provider_error(monkeypatch):
    # Free-tier limit (429) on the requested model must not stop the caller:
    # the router falls back to the remaining healthy candidates automatically.
    first = ProviderModel("p1/model-a", "p1", "model-a", 1, ["text"], True, True, "http://first/v1", "")
    second = ProviderModel("p2/model-b", "p2", "model-b", 2, ["text"], True, True, "http://second/v1", "")
    monkeypatch.setattr("app.main.load_registry_with_db_health", lambda: ProviderRegistry([first, second]))

    calls = []

    def fake_chat_completion(model, payload):
        calls.append(model.id)
        if model.id == "p2/model-b":
            return 429, {"error": {"message": "Rate limit exceeded: free-models-per-day"}}
        return 200, {"choices": [{"message": {"content": "OK"}}]}

    monkeypatch.setattr("app.main.chat_completion", fake_chat_completion)
    monkeypatch.setattr("app.main.persist_route_event", lambda *args, **kwargs: None)
    monkeypatch.setattr("app.main.mark_runtime_failure_unhealthy", lambda *args, **kwargs: None)

    response = client.post("/v1/chat/completions", json={"model": "p2/model-b", "messages": [{"role": "user", "content": "hi"}]})

    assert response.status_code == 200
    assert calls == ["p2/model-b", "p1/model-a"]
    assert response.headers["x-proxyrouter-model"] == "p1/model-a"


def test_chat_with_image_requires_vision_capability(monkeypatch):
    text_only = ProviderModel("p1/text", "p1", "text", 1, ["text"], True, True, "http://first/v1", "")
    vision = ProviderModel("p2/vision", "p2", "vision", 2, ["text", "vision"], True, True, "http://second/v1", "")
    monkeypatch.setattr("app.main.load_registry_with_db_health", lambda: ProviderRegistry([text_only, vision]))

    calls = []

    def fake_chat_completion(model, payload):
        calls.append(model.id)
        return 200, {"choices": [{"message": {"content": "OK"}}]}

    monkeypatch.setattr("app.main.chat_completion", fake_chat_completion)
    monkeypatch.setattr("app.main.persist_route_event", lambda *args, **kwargs: None)

    response = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "what is in this image?"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,xxx"}},
        ]}],
    })

    assert response.status_code == 200
    assert calls == ["p2/vision"]


def test_chat_completion_stream_success(monkeypatch):
    model = ProviderModel("p1/model-a", "p1", "model-a", 1, ["text"], True, True, "http://first/v1", "")
    monkeypatch.setattr("app.main.load_registry_with_db_health", lambda: ProviderRegistry([model]))

    calls = []
    def fake_chat_completion(model, payload):
        calls.append(model.id)
        def chunk_gen():
            yield b"data: {\"choices\": [{\"delta\": {\"content\": \"hi\"}}]}\n\n"
        return 200, chunk_gen()

    monkeypatch.setattr("app.main.chat_completion", fake_chat_completion)
    monkeypatch.setattr("app.main.persist_route_event", lambda *args, **kwargs: None)

    response = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}], "stream": True})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    content = b"".join(response.iter_bytes())
    assert b"hi" in content
    assert calls == ["p1/model-a"]


def test_auto_inclusion_readmits_runtime_degraded_models(monkeypatch):
    # Healthy pool below the minimum: a model degraded by a runtime failure (429)
    # re-enters as a reserve after the healthy candidates, instead of waiting out
    # the cooldown — deterioration under load must not stall the caller.
    healthy = ProviderModel("p1/model-a", "p1", "model-a", 1, ["text"], True, True, "http://first/v1", "")
    degraded = ProviderModel("p2/model-b", "p2", "model-b", 2, ["text"], True, False, "http://second/v1", "")
    monkeypatch.setattr("app.main.load_registry_with_db_health", lambda: ProviderRegistry([healthy, degraded]))
    monkeypatch.setattr("app.main.runtime_degraded_models", lambda: {"p2/model-b"})
    monkeypatch.setattr("app.main.persist_route_event", lambda *args, **kwargs: None)
    monkeypatch.setattr("app.main.mark_runtime_failure_unhealthy", lambda *args, **kwargs: None)

    calls = []
    def fake_chat_completion(model, payload):
        calls.append(model.id)
        if model.id == "p1/model-a":
            return 429, {"error": {"message": "rate limited"}}
        return 200, {"choices": [{"message": {"content": "OK"}}]}

    monkeypatch.setattr("app.main.chat_completion", fake_chat_completion)

    response = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})

    assert response.status_code == 200
    assert calls == ["p1/model-a", "p2/model-b"]


def test_auto_inclusion_skips_hard_failures(monkeypatch):
    # A model whose unhealthiness is not a runtime failure (e.g. invalid key)
    # is never re-admitted by the auto-inclusion rule.
    healthy = ProviderModel("p1/model-a", "p1", "model-a", 1, ["text"], True, True, "http://first/v1", "")
    broken = ProviderModel("p2/model-b", "p2", "model-b", 2, ["text"], True, False, "http://second/v1", "")
    monkeypatch.setattr("app.main.load_registry_with_db_health", lambda: ProviderRegistry([healthy, broken]))
    monkeypatch.setattr("app.main.runtime_degraded_models", lambda: set())
    monkeypatch.setattr("app.main.persist_route_event", lambda *args, **kwargs: None)
    monkeypatch.setattr("app.main.mark_runtime_failure_unhealthy", lambda *args, **kwargs: None)

    calls = []
    def fake_chat_completion(model, payload):
        calls.append(model.id)
        return 429, {"error": {"message": "rate limited"}}

    monkeypatch.setattr("app.main.chat_completion", fake_chat_completion)

    response = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})

    assert response.status_code == 502
    assert calls == ["p1/model-a"]


def test_chat_all_providers_failed_returns_attempt_trail(monkeypatch):
    # The 502 body must show every candidate tried, not just the last one —
    # otherwise diagnosing "everything failed" means a trip to route_events.
    first = ProviderModel("p1/model-a", "p1", "model-a", 1, ["text"], True, True, "http://first/v1", "")
    second = ProviderModel("p2/model-b", "p2", "model-b", 2, ["text"], True, True, "http://second/v1", "")
    monkeypatch.setattr("app.main.load_registry_with_db_health", lambda: ProviderRegistry([first, second]))
    monkeypatch.setattr("app.main.persist_route_event", lambda *args, **kwargs: None)
    monkeypatch.setattr("app.main.mark_runtime_failure_unhealthy", lambda *args, **kwargs: None)

    def fake_chat_completion(model, payload):
        if model.id == "p1/model-a":
            return 429, {"error": {"message": "rate limited"}}
        raise RuntimeError("connection reset")

    monkeypatch.setattr("app.main.chat_completion", fake_chat_completion)

    response = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})

    assert response.status_code == 502
    body = response.json()["error"]
    assert body["type"] == "all_providers_failed"
    assert body["attempts"] == [
        {"model_id": "p1/model-a", "provider": "p1", "status_code": 429, "error_type": "http_429"},
        {"model_id": "p2/model-b", "provider": "p2", "status_code": None, "error_type": "RuntimeError"},
    ]
    # last_error is kept for backward compat, still pointing at the final attempt.
    assert body["last_error"]["model_id"] == "p2/model-b"


def test_chat_rescues_plain_text_tool_call_from_a_small_model(monkeypatch):
    model = ProviderModel("p1/model-a", "p1", "model-a", 1, ["text", "tool_call"], True, True, "http://first/v1", "")
    monkeypatch.setattr("app.main.load_registry_with_db_health", lambda: ProviderRegistry([model]))
    monkeypatch.setattr("app.main.persist_route_event", lambda *args, **kwargs: None)

    def fake_chat_completion(model, payload):
        return 200, {"choices": [{"message": {"role": "assistant", "content": '{"name": "get_weather", "arguments": {"city": "Rio"}}'}}]}

    monkeypatch.setattr("app.main.chat_completion", fake_chat_completion)

    response = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "weather in Rio?"}],
        "tools": [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}],
    })

    assert response.status_code == 200
    assert response.headers["x-proxyrouter-tool-rescue"] == "true"
    message = response.json()["choices"][0]["message"]
    assert message["content"] is None
    assert message["tool_calls"][0]["function"]["name"] == "get_weather"


def test_rate_ledger_deprioritizes_a_model_after_it_gets_429d(monkeypatch):
    # First call: p1 (tier 1, tried first) gets 429'd — the ledger learns a
    # ceiling of 1 (one request was in-window at the moment of the 429).
    # Second call, still well inside that window: p1 must sort after p2
    # instead of being tried (and failing) again.
    first = ProviderModel("p1/model-a", "p1", "model-a", 1, ["text"], True, True, "http://first/v1", "")
    second = ProviderModel("p2/model-b", "p2", "model-b", 2, ["text"], True, True, "http://second/v1", "")
    monkeypatch.setattr("app.main.load_registry_with_db_health", lambda: ProviderRegistry([first, second]))
    monkeypatch.setattr("app.main.persist_route_event", lambda *args, **kwargs: None)
    monkeypatch.setattr("app.main.mark_runtime_failure_unhealthy", lambda *args, **kwargs: None)

    calls = []
    def fake_chat_completion(model, payload):
        calls.append(model.id)
        if model.id == "p1/model-a":
            return 429, {"error": {"message": "rate limited"}}
        return 200, {"choices": [{"message": {"content": "OK"}}]}
    monkeypatch.setattr("app.main.chat_completion", fake_chat_completion)

    first_response = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
    assert first_response.status_code == 200
    assert calls == ["p1/model-a", "p2/model-b"]

    calls.clear()
    second_response = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
    assert second_response.status_code == 200
    assert calls == ["p2/model-b"]  # p1 sorted last — not retried this time


def test_chat_completion_stream_persists_usage_from_final_chunk(monkeypatch):
    model = ProviderModel("p1/model-a", "p1", "model-a", 1, ["text"], True, True, "http://first/v1", "")
    monkeypatch.setattr("app.main.load_registry_with_db_health", lambda: ProviderRegistry([model]))

    payloads = []
    def fake_chat_completion(model, payload):
        payloads.append(payload)
        def chunk_gen():
            yield b"data: {\"choices\": [{\"delta\": {\"content\": \"hi\"}}]}\n\n"
            yield b"data: {\"choices\": [], \"usage\": {\"prompt_tokens\": 10, \"completion_tokens\": 5, \"total_tokens\": 15}}\n\n"
            yield b"data: [DONE]\n\n"
        return 200, chunk_gen()

    persisted = []
    monkeypatch.setattr("app.main.chat_completion", fake_chat_completion)
    monkeypatch.setattr("app.main.persist_route_event", lambda *args, **kwargs: persisted.append((args, kwargs)))

    response = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}], "stream": True})
    assert response.status_code == 200
    b"".join(response.iter_bytes())

    assert payloads[0]["stream_options"] == {"include_usage": True}
    assert len(persisted) == 1
    args, kwargs = persisted[0]
    assert kwargs["usage"] == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}


def test_chat_completion_stream_marks_unhealthy_on_in_band_error_chunk(monkeypatch):
    # OpenRouter-style aggregators (e.g. opencode zen) can return HTTP 200 and
    # start streaming valid `choices` content, then emit a mid-stream
    # `data: {"error": {...}}` chunk. The OpenAI SDK raises on any such chunk
    # regardless of earlier content, so this must be treated as a provider
    # failure: stop forwarding, persist a failed route event, and mark the
    # model unhealthy so the next request routes elsewhere.
    model = ProviderModel("p1/model-a", "p1", "model-a", 1, ["text"], True, True, "http://first/v1", "")
    monkeypatch.setattr("app.main.load_registry_with_db_health", lambda: ProviderRegistry([model]))

    def fake_chat_completion(model, payload):
        def chunk_gen():
            yield b"data: {\"choices\": [{\"delta\": {\"content\": \"hi\"}}]}\n\n"
            yield b"data: {\"code\": 400, \"message\": \"Provider returned error\", \"error\": {\"code\": 400, \"message\": \"Provider returned error\", \"metadata\": {\"error_type\": \"invalid_request\"}}}\n\n"
            yield b"data: {\"choices\": [{\"delta\": {\"content\": \"should not be forwarded\"}}]}\n\n"
        return 200, chunk_gen()

    persisted = []
    unhealthy_calls = []
    monkeypatch.setattr("app.main.chat_completion", fake_chat_completion)
    monkeypatch.setattr("app.main.persist_route_event", lambda *args, **kwargs: persisted.append((args, kwargs)))
    monkeypatch.setattr("app.main.mark_runtime_failure_unhealthy", lambda model, status_code, error_message, cooldown_seconds=None: unhealthy_calls.append((model.id, status_code, error_message)))

    response = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}], "stream": True})
    assert response.status_code == 200
    content = b"".join(response.iter_bytes())

    assert b"hi" in content
    assert b"should not be forwarded" not in content
    assert b"Provider returned error" not in content
    assert content.endswith(b"data: [DONE]\n\n")

    assert len(persisted) == 1
    args, kwargs = persisted[0]
    assert args[1] == "p1/model-a"
    assert args[2] == "text"
    assert args[3] == "provider_error"
    assert args[4] == "stream_invalid_request"

    assert unhealthy_calls == [("p1/model-a", None, "runtime_stream_invalid_request")]
