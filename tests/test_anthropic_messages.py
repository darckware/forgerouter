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


ANTHROPIC_TOOLS = [
    {
        "name": "get_weather",
        "description": "Weather lookup",
        "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
    }
]


def test_anthropic_messages_converts_tools_and_tool_results(monkeypatch):
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
        "/v1/messages",
        json={
            "model": "groq/llama-3.3-70b-versatile",
            "max_tokens": 64,
            "tools": ANTHROPIC_TOOLS,
            "messages": [
                {"role": "user", "content": "clima em SP?"},
                {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "SP"}}]},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "25C"}]},
            ],
        },
    )

    assert response.status_code == 200
    # Anthropic input_schema must reach the provider as OpenAI function parameters.
    tool = captured["tools"][0]
    assert tool["type"] == "function"
    assert tool["function"]["name"] == "get_weather"
    assert tool["function"]["parameters"]["properties"]["city"]["type"] == "string"
    # tool_use becomes assistant tool_calls; tool_result becomes a role:"tool" message.
    assistant = next(m for m in captured["messages"] if m["role"] == "assistant")
    call = assistant["tool_calls"][0]
    assert call["id"] == "toolu_1"
    assert call["function"]["name"] == "get_weather"
    assert json.loads(call["function"]["arguments"]) == {"city": "SP"}
    tool_message = next(m for m in captured["messages"] if m["role"] == "tool")
    assert tool_message["tool_call_id"] == "toolu_1"
    assert tool_message["content"] == "25C"


def test_anthropic_messages_returns_tool_use_blocks(monkeypatch):
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
        "/v1/messages",
        json={
            "model": "groq/llama-3.3-70b-versatile",
            "max_tokens": 64,
            "tools": ANTHROPIC_TOOLS,
            "messages": [{"role": "user", "content": "clima em SP?"}],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["stop_reason"] == "tool_use"
    block = body["content"][0]
    assert block["type"] == "tool_use"
    assert block["id"] == "call_1"
    assert block["name"] == "get_weather"
    assert block["input"] == {"city": "SP"}
    assert body["usage"] == {"input_tokens": 9, "output_tokens": 3}


def test_anthropic_messages_plain_text_roundtrip(monkeypatch):
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
        "/v1/messages",
        json={"model": "groq/llama-3.3-70b-versatile", "max_tokens": 16, "messages": [{"role": "user", "content": "oi"}]},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["content"] == [{"type": "text", "text": "oi"}]
    assert body["stop_reason"] == "end_turn"


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


def test_anthropic_messages_streams_real_provider_chunks(monkeypatch):
    # The point of real streaming: text and tool_use deltas must arrive as the
    # provider sends them, not as a synthesized burst replayed after the fact —
    # this feeds a live OpenAI-format chunk generator, not a complete message.
    monkeypatch.setattr("app.main.load_registry_with_db_health", lambda: ProviderRegistry([model()]))
    monkeypatch.setattr("app.main.persist_route_event", lambda *args, **kwargs: None)

    def fake_chat_completion(selected, payload):
        assert payload["stream"] is True

        def chunk_gen():
            yield b'data: {"choices": [{"delta": {"content": "Ol"}}]}\n\n'
            yield b'data: {"choices": [{"delta": {"content": "\xc3\xa1!"}}]}\n\n'
            yield b'data: {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_1", "function": {"name": "get_weather", "arguments": ""}}]}}]}\n\n'
            yield b'data: {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "{\\"city\\": \\"SP\\"}"}}]}}]}\n\n'
            yield b'data: {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}\n\n'
            yield b'data: {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}\n\n'
            yield b"data: [DONE]\n\n"

        return 200, chunk_gen()

    monkeypatch.setattr("app.main.chat_completion", fake_chat_completion)

    response = client.post(
        "/v1/messages",
        json={
            "model": "groq/llama-3.3-70b-versatile",
            "max_tokens": 64,
            "tools": ANTHROPIC_TOOLS,
            "messages": [{"role": "user", "content": "clima em SP?"}],
            "stream": True,
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = _parse_sse_events(response.content)
    types = [e["type"] for e in events]
    assert types[0] == "message_start"
    assert types[-1] == "message_stop"

    text_deltas = [e["delta"]["text"] for e in events if e["type"] == "content_block_delta" and e["delta"]["type"] == "text_delta"]
    assert "".join(text_deltas) == "Olá!"

    tool_start = next(e for e in events if e["type"] == "content_block_start" and e["content_block"]["type"] == "tool_use")
    assert tool_start["content_block"]["id"] == "call_1"
    assert tool_start["content_block"]["name"] == "get_weather"
    assert tool_start["content_block"]["input"] == {}

    json_deltas = [e["delta"]["partial_json"] for e in events if e["type"] == "content_block_delta" and e["delta"]["type"] == "input_json_delta"]
    assert json.loads("".join(json_deltas)) == {"city": "SP"}

    message_delta = next(e for e in events if e["type"] == "message_delta")
    assert message_delta["delta"]["stop_reason"] == "tool_use"
    assert message_delta["usage"] == {"input_tokens": 10, "output_tokens": 5}


def test_anthropic_messages_returns_413_for_context_too_large(monkeypatch):
    # /v1/messages passes chat_completions()' non-200 JSONResponse body
    # through unchanged, so the OpenAI-compatible 413 propagates for free —
    # this locks that contract in for the Anthropic protocol too.
    monkeypatch.setattr("app.main.load_registry_with_db_health", lambda: ProviderRegistry([model("p/small")]))
    monkeypatch.setattr("app.main.count_tokens", lambda *_: 100_000)
    monkeypatch.setattr("app.context_policy.context_window", lambda *_: 64_000)
    monkeypatch.setattr("app.main.context_truncation_enabled", lambda: False)
    monkeypatch.setattr("app.main.persist_route_event", lambda *args, **kwargs: None)
    calls = []
    monkeypatch.setattr("app.main.chat_completion", lambda selected, payload: calls.append(selected.id) or (200, {}))

    response = client.post(
        "/v1/messages",
        json={"model": "p/small", "max_tokens": 64, "messages": [{"role": "user", "content": "clima em SP?"}]},
    )

    assert response.status_code == 413
    assert response.json()["error"]["type"] == "context_too_large"
    assert calls == []
