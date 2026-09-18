from fastapi.testclient import TestClient

from app.main import app
from app.providers.openai_compatible import build_chat_payload
from app.registry import ProviderModel, ProviderRegistry

client = TestClient(app)


def test_chat_payload_omits_null_message_fields(monkeypatch):
    # Strict providers (Mistral, Cloudflare) 422 on explicit nulls in messages
    # ("name": null → extra_forbidden); the payload must omit unset fields.
    model = ProviderModel("p1/m", "p1", "m", 1, ["text"], True, True, "http://x/v1", "")
    monkeypatch.setattr("app.main.load_registry_with_db_health", lambda: ProviderRegistry([model]))
    monkeypatch.setattr("app.main.persist_route_event", lambda *args, **kwargs: None)
    # Known and above the 64k virtual-route floor — this test is about the
    # outgoing payload shape, not context fit.
    monkeypatch.setattr("app.context_policy.context_window", lambda *a: 200_000)
    captured = {}

    def fake_chat_completion(selected, payload):
        captured.update(payload)
        return 200, {"choices": [{"message": {"content": "OK"}}]}

    monkeypatch.setattr("app.main.chat_completion", fake_chat_completion)

    response = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "oi"}]})

    assert response.status_code == 200
    message = captured["messages"][0]
    assert message == {"role": "user", "content": "oi"}
    for forbidden in ("name", "tool_call_id", "tool_calls"):
        assert forbidden not in message


def test_build_chat_payload_uses_provider_model():
    model = ProviderModel(
        id="local/qwen2.5:1.5b", provider="local", provider_model="qwen2.5:1.5b",
        tier=4, capabilities=["text"], enabled=True, healthy=True, base_url="http://127.0.0.1:11434/v1", api_key_env=""
    )

    payload = build_chat_payload(model, [{"role": "user", "content": "OK"}], temperature=0, max_tokens=8)

    assert payload["model"] == "qwen2.5:1.5b"
    assert payload["messages"][0]["content"] == "OK"
    assert payload["temperature"] == 0
    assert payload["max_tokens"] == 8
