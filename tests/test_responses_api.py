import json

from fastapi.testclient import TestClient

from app.main import app
from app.registry import ProviderModel, ProviderRegistry

client = TestClient(app)


def model(model_id: str = "groq/llama-3.3-70b-versatile") -> ProviderModel:
    return ProviderModel(
        id=model_id,
        provider="groq",
        provider_model=model_id.split("/", 1)[1],
        tier=1,
        capabilities=["text", "tool_call"],
        enabled=True,
        healthy=True,
        base_url="https://api.groq.com/openai/v1",
        api_key_env="GROQ_API_KEY",
    )


RESPONSES_TOOLS = [
    {
        "type": "function",
        "name": "get_weather",
        "description": "Weather lookup",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
    }
]


def test_responses_converts_tools_and_function_call_output(monkeypatch):
    monkeypatch.setattr("app.main.load_registry_with_db_health", lambda: ProviderRegistry([model()]))
    monkeypatch.setattr("app.main.persist_route_event", lambda *args, **kwargs: None)
    captured = {}

    def fake_chat_completion(selected, payload):
        captured.update(payload)
        return 200, {
            "choices": [{"message": {"role": "assistant", "content": "25C em SP"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 4},
        }

    monkeypatch.setattr("app.main.chat_completion", fake_chat_completion)

    response = client.post(
        "/v1/responses",
        json={
            "model": "groq/llama-3.3-70b-versatile",
            "max_output_tokens": 64,
            "instructions": "Be concise.",
            "tools": RESPONSES_TOOLS,
            "input": [
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "clima em SP?"}]},
                {"type": "function_call", "call_id": "call_1", "name": "get_weather", "arguments": '{"city": "SP"}'},
                {"type": "function_call_output", "call_id": "call_1", "output": "25C"},
            ],
        },
    )

    assert response.status_code == 200
    # Flat Responses tool shape must reach the provider as OpenAI function parameters.
    tool = captured["tools"][0]
    assert tool["type"] == "function"
    assert tool["function"]["name"] == "get_weather"
    assert tool["function"]["parameters"]["properties"]["city"]["type"] == "string"
    # instructions become the system message; max_output_tokens maps to max_tokens.
    assert captured["messages"][0] == {"role": "system", "content": "Be concise."}
    assert captured["max_tokens"] == 64
    # function_call becomes assistant tool_calls; function_call_output becomes a role:"tool" message.
    assistant = next(m for m in captured["messages"] if m["role"] == "assistant")
    call = assistant["tool_calls"][0]
    assert call["id"] == "call_1"
    assert call["function"]["name"] == "get_weather"
    assert json.loads(call["function"]["arguments"]) == {"city": "SP"}
    tool_message = next(m for m in captured["messages"] if m["role"] == "tool")
    assert tool_message["tool_call_id"] == "call_1"
    assert tool_message["content"] == "25C"


def test_responses_returns_function_call_output_item(monkeypatch):
    monkeypatch.setattr("app.main.load_registry_with_db_health", lambda: ProviderRegistry([model()]))
    monkeypatch.setattr("app.main.persist_route_event", lambda *args, **kwargs: None)

    def fake_chat_completion(selected, payload):
        return 200, {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "SP"}'}}
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 9, "completion_tokens": 3},
        }

    monkeypatch.setattr("app.main.chat_completion", fake_chat_completion)

    response = client.post(
        "/v1/responses",
        json={
            "model": "groq/llama-3.3-70b-versatile",
            "tools": RESPONSES_TOOLS,
            "input": "clima em SP?",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    item = body["output"][0]
    assert item["type"] == "function_call"
    assert item["call_id"] == "call_1"
    assert item["name"] == "get_weather"
    assert json.loads(item["arguments"]) == {"city": "SP"}
    assert body["usage"] == {"input_tokens": 9, "output_tokens": 3, "total_tokens": 12}


def test_responses_plain_text_roundtrip(monkeypatch):
    monkeypatch.setattr("app.main.load_registry_with_db_health", lambda: ProviderRegistry([model()]))
    monkeypatch.setattr("app.main.persist_route_event", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "app.main.chat_completion",
        lambda selected, payload: (
            200,
            {"choices": [{"message": {"role": "assistant", "content": "oi"}, "finish_reason": "stop"}], "usage": {}},
        ),
    )

    response = client.post(
        "/v1/responses",
        json={"model": "groq/llama-3.3-70b-versatile", "input": "oi"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["output"] == [
        {"type": "message", "id": body["output"][0]["id"], "status": "completed", "role": "assistant", "content": [{"type": "output_text", "text": "oi", "annotations": []}]}
    ]
    assert body["status"] == "completed"


def test_responses_incomplete_on_length_finish_reason(monkeypatch):
    monkeypatch.setattr("app.main.load_registry_with_db_health", lambda: ProviderRegistry([model()]))
    monkeypatch.setattr("app.main.persist_route_event", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "app.main.chat_completion",
        lambda selected, payload: (
            200,
            {"choices": [{"message": {"role": "assistant", "content": "partial"}, "finish_reason": "length"}], "usage": {}},
        ),
    )

    response = client.post(
        "/v1/responses",
        json={"model": "groq/llama-3.3-70b-versatile", "input": "oi"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "incomplete"
    assert body["incomplete_details"] == {"reason": "max_output_tokens"}


def test_responses_no_healthy_provider_returns_openai_error_shape(monkeypatch):
    # Empty registry: no candidate for any capability — chat_completions
    # short-circuits with 503 before ever calling chat_completion. /v1/responses
    # must pass that error straight through, same shape Codex's own Responses
    # API client expects ({"error": {"message", "type"}}).
    monkeypatch.setattr("app.main.load_registry_with_db_health", lambda: ProviderRegistry([]))

    response = client.post(
        "/v1/responses",
        json={"model": "groq/llama-3.3-70b-versatile", "input": "oi"},
    )

    assert response.status_code == 503
    body = response.json()
    assert body["error"]["type"] == "no_healthy_provider"
    assert "message" in body["error"]


def test_responses_all_providers_failed_returns_502(monkeypatch):
    monkeypatch.setattr("app.main.load_registry_with_db_health", lambda: ProviderRegistry([model()]))
    monkeypatch.setattr("app.main.persist_route_event", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "app.main.chat_completion",
        lambda selected, payload: (500, {"error": {"message": "upstream on fire", "type": "server_error"}}),
    )

    response = client.post(
        "/v1/responses",
        json={"model": "groq/llama-3.3-70b-versatile", "input": "oi"},
    )

    assert response.status_code == 502
    body = response.json()
    assert body["error"]["type"] == "all_providers_failed"


def _parse_sse_events(raw: bytes) -> list[dict]:
    events = []
    for block in raw.decode("utf-8").split("\n\n"):
        block = block.strip()
        if not block:
            continue
        lines = block.split("\n")
        data_line = next((line[len("data:"):].strip() for line in lines if line.startswith("data:")), None)
        if data_line:
            events.append(json.loads(data_line))
    return events


def test_responses_streams_real_provider_chunks(monkeypatch):
    # Real streaming: text/tool_call deltas must arrive as the provider sends
    # them, not as a synthesized burst replayed after a complete response.
    monkeypatch.setattr("app.main.load_registry_with_db_health", lambda: ProviderRegistry([model()]))
    monkeypatch.setattr("app.main.persist_route_event", lambda *args, **kwargs: None)

    def fake_chat_completion(selected, payload):
        assert payload["stream"] is True

        def chunk_gen():
            yield b'data: {"choices": [{"delta": {"content": "25"}}]}\n\n'
            yield b'data: {"choices": [{"delta": {"content": "C em SP"}}]}\n\n'
            yield b'data: {"choices": [{"delta": {}, "finish_reason": "stop"}]}\n\n'
            yield b'data: {"choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": 4, "total_tokens": 11}}\n\n'
            yield b"data: [DONE]\n\n"

        return 200, chunk_gen()

    monkeypatch.setattr("app.main.chat_completion", fake_chat_completion)

    response = client.post(
        "/v1/responses",
        json={"model": "groq/llama-3.3-70b-versatile", "input": "clima em SP?", "stream": True},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = _parse_sse_events(response.content)
    types = [e["type"] for e in events]
    assert types[0] == "response.created"
    assert types[1] == "response.in_progress"
    assert types[-1] == "response.completed"

    text_deltas = [e["delta"] for e in events if e["type"] == "response.output_text.delta"]
    assert "".join(text_deltas) == "25C em SP"

    completed = next(e for e in events if e["type"] == "response.completed")
    assert completed["response"]["status"] == "completed"
    assert completed["response"]["output"][0]["content"][0]["text"] == "25C em SP"
    assert completed["response"]["usage"] == {"input_tokens": 7, "output_tokens": 4, "total_tokens": 11}


def test_responses_streams_function_call_arguments(monkeypatch):
    monkeypatch.setattr("app.main.load_registry_with_db_health", lambda: ProviderRegistry([model()]))
    monkeypatch.setattr("app.main.persist_route_event", lambda *args, **kwargs: None)

    def fake_chat_completion(selected, payload):
        def chunk_gen():
            yield b'data: {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_1", "function": {"name": "get_weather", "arguments": ""}}]}}]}\n\n'
            yield b'data: {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "{\\"city\\": "}}]}}]}\n\n'
            yield b'data: {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "\\"SP\\"}"}}]}}]}\n\n'
            yield b'data: {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}\n\n'
            yield b"data: [DONE]\n\n"

        return 200, chunk_gen()

    monkeypatch.setattr("app.main.chat_completion", fake_chat_completion)

    response = client.post(
        "/v1/responses",
        json={"model": "groq/llama-3.3-70b-versatile", "input": "clima em SP?", "tools": RESPONSES_TOOLS, "stream": True},
    )

    assert response.status_code == 200
    events = _parse_sse_events(response.content)

    arg_deltas = [e["delta"] for e in events if e["type"] == "response.function_call_arguments.delta"]
    assert json.loads("".join(arg_deltas)) == {"city": "SP"}

    added = next(e for e in events if e["type"] == "response.output_item.added")
    assert added["item"]["type"] == "function_call"
    assert added["item"]["call_id"] == "call_1"
    assert added["item"]["name"] == "get_weather"

    completed = next(e for e in events if e["type"] == "response.completed")
    assert completed["response"]["output"][0]["type"] == "function_call"
    assert json.loads(completed["response"]["output"][0]["arguments"]) == {"city": "SP"}


def test_responses_returns_413_for_context_too_large(monkeypatch):
    # /v1/responses passes chat_completions()' non-200 JSONResponse body
    # through unchanged, so the OpenAI-compatible 413 propagates for free —
    # this locks that contract in for the Responses protocol too.
    monkeypatch.setattr("app.main.load_registry_with_db_health", lambda: ProviderRegistry([model("p/small")]))
    monkeypatch.setattr("app.main.count_tokens", lambda *_: 100_000)
    monkeypatch.setattr("app.context_policy.context_window", lambda *_: 64_000)
    monkeypatch.setattr("app.main.context_truncation_enabled", lambda: False)
    monkeypatch.setattr("app.main.persist_route_event", lambda *args, **kwargs: None)
    calls = []
    monkeypatch.setattr("app.main.chat_completion", lambda selected, payload: calls.append(selected.id) or (200, {}))

    response = client.post("/v1/responses", json={"model": "p/small", "input": "clima em SP?"})

    assert response.status_code == 413
    assert response.json()["error"]["type"] == "context_too_large"
    assert calls == []
