import app.validation.context_probe as probe_module
from app.registry import ProviderModel
from app.validation.context_probe import probe_context_window


def _model(model_id: str = "p1/model", provider_model: str = "model") -> ProviderModel:
    return ProviderModel(model_id, "p1", provider_model, 1, ["text"], True, True, "http://first/v1", "")


def _scripted_chat_completion(monkeypatch, responses: list[tuple[int, object]]):
    """Fakes app.providers.openai_compatible.chat_completion with a fixed
    sequence of (status_code, body) responses, one per call in order —
    probe_context_window always tests checkpoints in a fixed, deterministic
    sequence, so call order (not trying to recover the token count from the
    padded prompt) is the reliable thing to script against."""
    calls = {"count": 0}

    def fake_chat_completion(model, payload, timeout=60.0):
        index = calls["count"]
        calls["count"] += 1
        return responses[index]

    monkeypatch.setattr("app.providers.openai_compatible.chat_completion", fake_chat_completion)
    return calls


def test_probe_ascends_past_the_floor_when_it_fits(monkeypatch):
    # Order matches probe_context_window's fixed checkpoint sequence: floor
    # (64k), then ascending (128k, 256k, ...).
    script = [
        (200, {"choices": [{"message": {"content": "OK"}}]}),  # 64_000
        (200, {"choices": [{"message": {"content": "OK"}}]}),  # 128_000
        (400, {"error": {"message": "maximum context length exceeded"}}),  # 256_000
    ]
    _scripted_chat_completion(monkeypatch, script)

    result = probe_context_window(_model(), delay_seconds=0)

    assert result.context_window == 128_000
    assert result.checkpoints[0] == (64_000, "fit")
    assert result.checkpoints[-1] == (256_000, "overflow")


def test_probe_descends_below_the_floor_when_it_overflows(monkeypatch):
    # Order: floor (64k) overflows, then descending (32k, 16k, ...).
    script = [
        (400, {"error": {"message": "context_length_exceeded"}}),  # 64_000
        (400, {"error": {"message": "context_length_exceeded"}}),  # 32_000
        (200, {"choices": [{"message": {"content": "OK"}}]}),  # 16_000
    ]
    _scripted_chat_completion(monkeypatch, script)

    result = probe_context_window(_model(), delay_seconds=0)

    assert result.context_window == 16_000
    assert "below the 64k floor" in result.note


def test_probe_is_inconclusive_on_auth_error_at_the_floor(monkeypatch):
    script = [(401, {"error": {"message": "invalid api key"}})]
    _scripted_chat_completion(monkeypatch, script)

    result = probe_context_window(_model(), delay_seconds=0)

    assert result.context_window is None
    assert "auth_error" in result.note


def test_probe_never_mistakes_rate_limit_for_overflow(monkeypatch):
    # A 429 must never be read as "this checkpoint doesn't fit" — that would
    # report a wrong (too small) window for a model that's simply rate-limited.
    script = [(429, {"error": {"message": "rate limit exceeded"}})]
    _scripted_chat_completion(monkeypatch, script)

    result = probe_context_window(_model(), delay_seconds=0)

    assert result.context_window is None
    assert "rate_limited" in result.note


def test_probe_stops_refining_but_keeps_the_last_confirmed_bound_on_a_later_inconclusive(monkeypatch):
    script = [
        (200, {"choices": [{"message": {"content": "OK"}}]}),  # 64_000
        (500, {"error": {"message": "internal error"}}),  # 128_000
    ]
    _scripted_chat_completion(monkeypatch, script)

    result = probe_context_window(_model(), delay_seconds=0)

    assert result.context_window == 64_000
    assert result.checkpoints[-1] == (128_000, "inconclusive:http_500")


def test_probe_bails_out_without_a_tokenizer(monkeypatch):
    monkeypatch.setattr(probe_module, "count_tokens", lambda *a, **k: None)

    result = probe_context_window(_model(), delay_seconds=0)

    assert result.context_window is None
    assert "tokenizer_unavailable" in result.note
