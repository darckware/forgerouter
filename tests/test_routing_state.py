import pytest
from fastapi.testclient import TestClient

from app import routing_state
from app.demand import default_chain
from app.main import app
from app.providers.openai_compatible import parse_retry_after
from app.ranking import dynamic_score
from app.registry import ProviderModel, ProviderRegistry

client = TestClient(app)


@pytest.fixture(autouse=True)
def _known_roomy_context_window(monkeypatch):
    # This file is about routing-state mechanics (breakers/sticky/rate
    # ledger), not context fit — give every synthetic model here a known
    # window safely above the 64k virtual-route floor (the default "auto"
    # model would otherwise hard-exclude its uncatalogued window).
    monkeypatch.setattr("app.context_policy.context_window", lambda *a: 200_000)


def model(public_id: str, tier: int, caps: list[str] | None = None) -> ProviderModel:
    return ProviderModel(public_id, public_id.split("/")[0], public_id.split("/", 1)[1], tier, caps or ["text"], True, True, "http://x/v1", "")


def msg(content: str, role: str = "user") -> dict:
    return {"role": role, "content": content}


def test_breaker_opens_at_threshold_and_recovers(monkeypatch):
    monkeypatch.setenv("BREAKER_THRESHOLD", "2")
    monkeypatch.setenv("BREAKER_COOLDOWN_SECONDS", "0.05")
    routing_state.record_provider_failure("p1")
    assert routing_state.breaker_open("p1") is False
    routing_state.record_provider_failure("p1")
    assert routing_state.breaker_open("p1") is True
    import time

    time.sleep(0.06)
    # Half-open: the probe is allowed, one more failure re-opens immediately.
    assert routing_state.breaker_open("p1") is False
    routing_state.record_provider_failure("p1")
    assert routing_state.breaker_open("p1") is True
    routing_state.record_provider_success("p1")
    assert routing_state.breaker_open("p1") is False


def test_sticky_records_and_expires(monkeypatch):
    routing_state.record_sticky("athos", "simple", "p1/m")
    assert routing_state.sticky_model("athos", "simple") == "p1/m"
    assert routing_state.sticky_model("other", "simple") is None
    assert routing_state.sticky_model("athos", None) is None
    monkeypatch.setenv("STICKY_TTL_SECONDS", "0")
    assert routing_state.sticky_model("athos", "simple") is None


def test_parse_retry_after():
    assert parse_retry_after("30") == 30
    assert parse_retry_after("1.5") == 1
    assert parse_retry_after("99999999") == 6 * 3600
    assert parse_retry_after("0") is None
    assert parse_retry_after("Wed, 21 Oct 2026 07:28:00 GMT") is None
    assert parse_retry_after(None) is None


def test_dynamic_score_blends_success_rate_and_latency():
    base = dynamic_score("groq/llama-3.3-70b-versatile")  # 41 static
    assert base == 41.0
    perf = {"groq/llama-3.3-70b-versatile": {"total": 20, "success_rate": 0.5, "latency_ms": 10000}}
    scored = dynamic_score("groq/llama-3.3-70b-versatile", perf)
    assert scored == 41 * 0.8 - 2.0
    # Thin data (< 5 attempts) keeps the static score.
    thin = {"groq/llama-3.3-70b-versatile": {"total": 3, "success_rate": 0.0}}
    assert dynamic_score("groq/llama-3.3-70b-versatile", thin) == 41.0


def test_default_chain_orders_by_dynamic_score_within_band():
    first = model("cloudflare/openai/gpt-oss-120b", 2)   # static 61
    second = model("openrouter/deepseek-r1:free", 2)     # static 60
    # The higher-static model has been failing: it must sink below its band peer.
    perf = {"cloudflare/openai/gpt-oss-120b": {"total": 20, "success_rate": 0.1, "latency_ms": 0}}
    chain = default_chain([first, second], "complex", performance=perf)
    assert chain[0].id == "openrouter/deepseek-r1:free"
    # Band membership stays static: the failing model is still in the complex band.
    assert chain[1].id == "cloudflare/openai/gpt-oss-120b"


def test_chat_deprioritizes_tripped_provider(monkeypatch):
    monkeypatch.setenv("BREAKER_THRESHOLD", "1")
    a = model("p1/model-a", 1)
    b = model("p2/model-b", 2)
    monkeypatch.setattr("app.main.load_registry_with_db_health", lambda: ProviderRegistry([a, b]))
    monkeypatch.setattr("app.main.persist_route_event", lambda *args, **kwargs: None)
    routing_state.record_provider_failure("p1")  # opens p1's breaker

    calls = []

    def fake_chat_completion(selected, payload):
        calls.append(selected.id)
        return 200, {"choices": [{"message": {"content": "OK"}}]}

    monkeypatch.setattr("app.main.chat_completion", fake_chat_completion)

    response = client.post("/v1/chat/completions", json={"messages": [msg("hi")]})

    assert response.status_code == 200
    # p1 is tier 1 but its breaker is open — p2 goes first.
    assert calls == ["p2/model-b"]


def test_chat_all_breakers_open_still_serves(monkeypatch):
    monkeypatch.setenv("BREAKER_THRESHOLD", "1")
    a = model("p1/model-a", 1)
    monkeypatch.setattr("app.main.load_registry_with_db_health", lambda: ProviderRegistry([a]))
    monkeypatch.setattr("app.main.persist_route_event", lambda *args, **kwargs: None)
    routing_state.record_provider_failure("p1")
    monkeypatch.setattr("app.main.chat_completion", lambda selected, payload: (200, {"choices": [{"message": {"content": "OK"}}]}))

    response = client.post("/v1/chat/completions", json={"messages": [msg("hi")]})

    # Deprioritized, never excluded: the only candidate still serves.
    assert response.status_code == 200


def test_chat_sticky_model_goes_first(monkeypatch):
    small = model("local/qwen2.5:1.5b", 4)
    mid = model("groq/llama-3.3-70b-versatile", 1)
    monkeypatch.setattr("app.main.load_registry_with_db_health", lambda: ProviderRegistry([mid, small]))
    monkeypatch.setattr("app.main.get_demand_routes", lambda: {})
    monkeypatch.setattr("app.main.persist_route_event", lambda *args, **kwargs: None)
    # Last success for this demand was the mid model — cache affinity keeps it
    # first even though the simple chain would start at the small model.
    routing_state.record_sticky(None, "simple", "groq/llama-3.3-70b-versatile")

    calls = []

    def fake_chat_completion(selected, payload):
        calls.append(selected.id)
        return 200, {"choices": [{"message": {"content": "OK"}}]}

    monkeypatch.setattr("app.main.chat_completion", fake_chat_completion)

    response = client.post("/v1/chat/completions", json={"model": "forgerouter/simple", "messages": [msg("oi")]})

    assert response.status_code == 200
    assert calls == ["groq/llama-3.3-70b-versatile"]


def test_chat_success_records_sticky_and_closes_breaker(monkeypatch):
    small = model("local/qwen2.5:1.5b", 4)
    monkeypatch.setattr("app.main.load_registry_with_db_health", lambda: ProviderRegistry([small]))
    monkeypatch.setattr("app.main.get_demand_routes", lambda: {})
    monkeypatch.setattr("app.main.persist_route_event", lambda *args, **kwargs: None)
    routing_state.record_provider_failure("local")
    monkeypatch.setattr("app.main.chat_completion", lambda selected, payload: (200, {"choices": [{"message": {"content": "OK"}}]}))

    response = client.post("/v1/chat/completions", json={"model": "forgerouter/simple", "messages": [msg("oi")]})

    assert response.status_code == 200
    assert routing_state.sticky_model(None, "simple") == "local/qwen2.5:1.5b"
    assert routing_state.breaker_open("local") is False


def test_retry_after_becomes_model_cooldown(monkeypatch):
    a = model("p1/model-a", 1)
    b = model("p2/model-b", 2)
    monkeypatch.setattr("app.main.load_registry_with_db_health", lambda: ProviderRegistry([a, b]))
    monkeypatch.setattr("app.main.persist_route_event", lambda *args, **kwargs: None)
    marked = {}

    def fake_mark(selected, http_code, error, cooldown_seconds=None):
        marked[selected.id] = cooldown_seconds

    monkeypatch.setattr("app.main.mark_runtime_failure_unhealthy", fake_mark)

    def fake_chat_completion(selected, payload):
        if selected.id == "p1/model-a":
            return 429, {"error": {"message": "rate limit"}, "_proxyrouter_retry_after": 3600}
        return 200, {"choices": [{"message": {"content": "OK"}}]}

    monkeypatch.setattr("app.main.chat_completion", fake_chat_completion)

    response = client.post("/v1/chat/completions", json={"messages": [msg("hi")]})

    assert response.status_code == 200
    assert marked["p1/model-a"] == 3600
    # The internal marker must never leak to the caller.
    assert "_proxyrouter_retry_after" not in response.text
