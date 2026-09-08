from fastapi.testclient import TestClient

from app.demand import classify_request, default_chain, resolve_demand
from app.main import ChatCompletionRequest, ChatMessage, app, infer_capability
from app.normalize import count_tokens
from app.registry import ProviderModel, ProviderRegistry

client = TestClient(app)


def model(public_id: str, tier: int, caps: list[str] | None = None) -> ProviderModel:
    return ProviderModel(public_id, public_id.split("/")[0], public_id.split("/", 1)[1], tier, caps or ["text"], True, True, "http://x/v1", "")


def msg(content: str, role: str = "user") -> dict:
    return {"role": role, "content": content}


def test_classify_request_by_size_and_hints():
    assert classify_request([msg("gere um título curto")], has_tools=False) == "simple"
    assert classify_request([msg("x" * 3000)], has_tools=False) == "standard"
    assert classify_request([msg("x" * 9000)], has_tools=False) == "complex"
    assert classify_request([msg("analise passo a passo este problema")], has_tools=False) == "reasoning"
    assert classify_request([msg("oi")], has_tools=True) == "standard"


def test_classify_discounts_history_for_short_follow_ups():
    # A trivial follow-up in a long conversation must not drift to "complex":
    # history counts at a discount, so 10k chars of transcript + "obrigado, ficou ótimo"
    # stays in the cheap bands instead of burning big-model quota.
    history = [msg("x" * 5000), msg("y" * 5000, role="assistant")]
    assert classify_request([*history, msg("obrigado, ficou ótimo")], has_tools=False) == "standard"
    # Truly long contexts still escalate — just later (discounted, not ignored).
    huge_history = [msg("x" * 20000), msg("y" * 20000, role="assistant")]
    assert classify_request([*huge_history, msg("resuma tudo isso")], has_tools=False) == "complex"


def test_reasoning_hints_match_whole_words_only():
    # "prove" must not fire inside "aprove"/"provedor" (substring false positives).
    assert classify_request([msg("aprove a proposta do provedor")], has_tools=False) == "simple"
    assert classify_request([msg("prove que a raiz de 2 é irracional")], has_tools=False) == "reasoning"


def test_classify_request_detects_code():
    assert classify_request([msg("refatore a função de login")], has_tools=False) == "code"
    assert classify_request([msg("corrija o bug no arquivo app/main.py")], has_tools=False) == "code"
    assert classify_request([msg("o que acha deste trecho?\n```python\nprint('hi')\n```")], has_tools=False) == "code"
    # Code signals win over tools and reasoning hints — coding agents send both.
    assert classify_request([msg("implemente passo a passo o parser.ts")], has_tools=True) == "code"
    assert classify_request([msg("me explique o que é uma API")], has_tools=False) == "simple"


def test_classify_request_strips_context_compression_and_task_lists():
    user_with_task_list = (
        "[Your active task list was preserved across context compression]\n"
        "- [>] 1. Criar script hindsight_prune.py (retenção 90d, preserva imutáveis) (in_progress)\n"
        "- [ ] 2. Registrar cron hindsight-daily-prune (03:00 diário) (pending)\n\n"
        "consegue resolver athos\n\n"
        "continue"
    )
    assert classify_request([msg(user_with_task_list)], has_tools=True) == "standard"

    user_with_compaction = (
        "[CONTEXT COMPACTION — REFERENCE ONLY] Earlier turns were compacted into the summary below.\n"
        "- Created script helper.py\n\n"
        "tudo certo, prossiga"
    )
    assert classify_request([msg(user_with_compaction)], has_tools=False) == "simple"


def test_classify_request_with_images_is_vision():
    image_msg = {"role": "user", "content": [
        {"type": "text", "text": "o que aparece na imagem?"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,xyz"}},
    ]}
    assert classify_request([image_msg], has_tools=False) == "vision"


def test_classify_request_with_audio_is_audio():
    audio_msg = {"role": "user", "content": [
        {"type": "text", "text": "transcreva isso"},
        {"type": "input_audio", "input_audio": {"data": "xyz", "format": "wav"}},
    ]}
    assert classify_request([audio_msg], has_tools=False) == "audio"


def test_infer_capability_prefers_audio_content_over_tool_call():
    # Same hard-requirement treatment as vision: a non-audio-capable model
    # cannot read an input_audio block at all, so it must win over tool_call
    # even when no demand was resolved to "audio" yet.
    request = ChatCompletionRequest(
        model="forgerouter/auto",
        tools=[{"type": "function", "function": {"name": "noop"}}],
        messages=[
            ChatMessage(role="user", content=[
                {"type": "input_audio", "input_audio": {"data": "xyz", "format": "wav"}},
                {"type": "text", "text": "transcreva isso"},
            ])
        ],
    )
    assert infer_capability(request) == "audio"


def test_infer_capability_prefers_vision_over_tool_call():
    # An agentic request commonly attaches both tools and an image (e.g. a
    # function-calling agent describing a screenshot). Images are a hard
    # requirement — a non-vision model cannot read them at all — so vision
    # must win over tool_call, or candidate selection silently drops every
    # vision-capable model that lacks tool_call support.
    request = ChatCompletionRequest(
        model="forgerouter/auto",
        tools=[{"type": "function", "function": {"name": "noop"}}],
        messages=[
            ChatMessage(role="user", content=[
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,xyz"}},
                {"type": "text", "text": "o que aparece na imagem?"},
            ])
        ],
    )
    assert infer_capability(request) == "vision"


def test_infer_capability_prefers_code_and_audio_demand_over_tool_call():
    # Same intersection bug as vision: an agentic harness typically attaches
    # tools to every request regardless of task type, so a code/audio demand
    # must still hard-gate on its own catalog capability (default_chain's
    # existing behaviour) instead of being narrowed to tool_call-only models
    # first.
    request = ChatCompletionRequest(
        model="forgerouter/auto",
        tools=[{"type": "function", "function": {"name": "noop"}}],
        messages=[ChatMessage(role="user", content="refatore a função de login")],
    )
    assert infer_capability(request, demand="code") == "code"
    assert infer_capability(request, demand="audio") == "audio"
    # Demands that aren't catalog-gated (simple/standard/complex/reasoning) keep
    # the existing tool_call/text behaviour.
    assert infer_capability(request, demand="complex") == "tool_call"


def test_default_chain_vision_only_uses_vision_capable():
    plain = model("groq/llama-3.3-70b-versatile", 1)
    seeing = model("groq/meta-llama/llama-4-scout-17b-16e-instruct", 1, ["text", "vision"])
    chain = default_chain([plain, seeing], "vision")
    assert chain == [seeing]


def test_default_chain_audio_only_uses_audio_capable():
    plain = model("groq/llama-3.3-70b-versatile", 1)
    omni = model("openrouter/nemotron-omni:free", 2, ["text", "audio"])
    chain = default_chain([plain, omni], "audio")
    assert chain == [omni]


def test_default_chain_code_only_uses_code_capable():
    plain = model("groq/llama-3.3-70b-versatile", 1)
    coder = model("openrouter/qwen-2.5-coder:free", 2, ["text", "code"])
    chain = default_chain([plain, coder], "code")
    assert chain == [coder]


def test_resolve_demand_virtual_models():
    assert resolve_demand("forgerouter/simple", [], False) == "simple"
    assert resolve_demand("forgerouter/audio", [], False) == "audio"
    assert resolve_demand("forgerouter/auto", [msg("oi")], False) == "simple"
    assert resolve_demand("auto", [msg("x" * 9000)], False) == "complex"
    assert resolve_demand("groq/llama-3.1-8b-instant", [], False) is None


def test_default_chain_bands_by_rank():
    small = model("local/qwen2.5:1.5b", 4)            # score 12
    mid = model("groq/llama-3.3-70b-versatile", 1)    # score 41
    big = model("openrouter/deepseek-r1:free", 2, ["text", "reasoning"])  # score 60
    chain_simple = default_chain([small, mid, big], "simple")
    chain_complex = default_chain([small, mid, big], "complex")
    chain_reasoning = default_chain([small, mid, big], "reasoning")
    assert chain_simple[0].id == "local/qwen2.5:1.5b"
    assert chain_complex[0].id == "openrouter/deepseek-r1:free"
    assert chain_reasoning == [big]


def test_default_chain_is_pure_rank_order_regardless_of_context_window(monkeypatch):
    # Context-window fit (the Hermes-minimum floor and prompt-size fit) is no
    # longer default_chain's job — app.context_policy.partition_candidates()
    # owns it centrally, downstream of this ordering. A tiny/unknown window
    # must not change or drop anything here.
    tiny = model("groq/llama-3.3-70b-versatile", 1)  # score 41
    roomy = model("openrouter/qwen2.5-72b-instruct", 2)  # score 40
    monkeypatch.setattr("app.pricing.context_window", lambda public_id, provider_model: {"groq/llama-3.3-70b-versatile": 8_000}.get(public_id))

    chain = default_chain([roomy, tiny], "standard")

    assert [m.id for m in chain] == ["groq/llama-3.3-70b-versatile", "openrouter/qwen2.5-72b-instruct"]


def test_chat_routes_by_demand_chain(monkeypatch):
    small = model("local/qwen2.5:1.5b", 4)
    mid = model("groq/llama-3.3-70b-versatile", 1)
    monkeypatch.setattr("app.main.load_registry_with_db_health", lambda: ProviderRegistry([mid, small]))
    monkeypatch.setattr("app.main.get_demand_routes", lambda: {})
    # Neutral context windows: this test is about demand-chain ordering, not
    # context fit — a real catalog match for "llama-3.3-70b-versatile" would
    # otherwise let the known-fit-first context policy jump it ahead.
    monkeypatch.setattr("app.context_policy.context_window", lambda *a: None)
    persisted = {}
    monkeypatch.setattr("app.main.persist_route_event", lambda *args, **kwargs: persisted.update(kwargs))

    calls = []

    def fake_chat_completion(selected, payload):
        calls.append(selected.id)
        return 200, {"choices": [{"message": {"content": "OK"}}]}

    monkeypatch.setattr("app.main.chat_completion", fake_chat_completion)

    # Short prompt → simple demand → the small model goes first despite its higher tier.
    response = client.post("/v1/chat/completions", json={"model": "forgerouter/auto", "messages": [msg("titulo curto")]})

    assert response.status_code == 200
    assert calls == ["local/qwen2.5:1.5b"]
    assert response.headers["x-proxyrouter-model"] == "local/qwen2.5:1.5b"
    # The resolved demand class is recorded with the route event for auditing.
    assert persisted["demand"] == "simple"


def test_chat_demand_chain_falls_back(monkeypatch):
    small = model("local/qwen2.5:1.5b", 4)
    mid = model("groq/llama-3.3-70b-versatile", 1)
    monkeypatch.setattr("app.main.load_registry_with_db_health", lambda: ProviderRegistry([mid, small]))
    monkeypatch.setattr("app.main.get_demand_routes", lambda: {"simple": ["local/qwen2.5:1.5b"]})
    monkeypatch.setattr("app.main.persist_route_event", lambda *args, **kwargs: None)
    monkeypatch.setattr("app.main.mark_runtime_failure_unhealthy", lambda *args, **kwargs: None)
    monkeypatch.setattr("app.context_policy.context_window", lambda *a: None)

    calls = []

    def fake_chat_completion(selected, payload):
        calls.append(selected.id)
        if selected.id == "local/qwen2.5:1.5b":
            return 429, {"error": {"message": "rate limit"}}
        return 200, {"choices": [{"message": {"content": "OK"}}]}

    monkeypatch.setattr("app.main.chat_completion", fake_chat_completion)

    response = client.post("/v1/chat/completions", json={"model": "forgerouter/simple", "messages": [msg("oi")]})

    assert response.status_code == 200
    assert calls == ["local/qwen2.5:1.5b", "groq/llama-3.3-70b-versatile"]


def test_chat_code_demand_falls_back_when_no_code_capable_model(monkeypatch):
    # No model in the pool has the "code" capability — the old behaviour hard-failed
    # (503 no_healthy_provider) here. Code-capable is a quality preference, not a
    # hard requirement like vision, so this must still route (to the best general
    # model) instead of blocking, and the persisted event must record the downgrade
    # (demand="code" but required_capability != "code") as the countable
    # "no code option available" signal.
    big = model("openrouter/deepseek-r1:free", 2, ["text", "reasoning"])
    small = model("local/qwen2.5:1.5b", 4)
    monkeypatch.setattr("app.main.load_registry_with_db_health", lambda: ProviderRegistry([small, big]))
    monkeypatch.setattr("app.main.get_demand_routes", lambda: {})
    persisted = {}

    def fake_persist(request_id, model_id, capability, *args, **kwargs):
        persisted.update(kwargs)
        persisted["capability"] = capability

    monkeypatch.setattr("app.main.persist_route_event", fake_persist)

    calls = []

    def fake_chat_completion(selected, payload):
        calls.append(selected.id)
        return 200, {"choices": [{"message": {"content": "OK"}}]}

    monkeypatch.setattr("app.main.chat_completion", fake_chat_completion)

    response = client.post("/v1/chat/completions", json={"model": "forgerouter/code", "messages": [msg("refatore a função de login")]})

    assert response.status_code == 200
    # Ranked best-to-worst across the general pool: the higher-scored model goes first.
    assert calls == ["openrouter/deepseek-r1:free"]
    assert persisted["demand"] == "code"
    assert persisted["capability"] != "code"


# The sub-minimum-context deprioritization case formerly covered here now
# belongs to the request-level context policy (app.context_policy,
# integrated into chat_completions()) instead of demand ordering — see
# tests/test_context_policy.py for the hard virtual-route exclusion, and the
# chat_completions-level fit tests added alongside it.


def test_chat_truncation_targets_largest_viable_budget_and_excludes_smaller_candidates(monkeypatch):
    # Regression for the pre-fix bug: the budget was computed from the
    # SMALLEST window across the whole fallback pool, so a request could get
    # truncated far more than necessary even when the actually-selected
    # candidate had a much bigger real window. It must instead target the
    # LARGEST viable budget (200k * 80% = 160k here), drop history only if
    # that's still not enough, and then exclude any candidate whose own
    # budget (50k * 80% = 40k) still can't hold the trimmed result from the
    # retry pool — rather than trying it and letting the provider reject it.
    big_window = model("groq/llama-3.3-70b-versatile", 1)
    small_window = model("openrouter/qwen-2.5-7b-instruct", 2)
    monkeypatch.setattr("app.main.load_registry_with_db_health", lambda: ProviderRegistry([big_window, small_window]))
    windows = {"groq/llama-3.3-70b-versatile": 200_000, "openrouter/qwen-2.5-7b-instruct": 50_000}
    monkeypatch.setattr("app.context_policy.context_window", lambda public_id, provider_model: windows.get(public_id))
    monkeypatch.setattr("app.main.context_truncation_enabled", lambda: True)
    monkeypatch.setattr("app.main.context_truncation_trigger_percent", lambda: 80)
    monkeypatch.setattr("app.main.persist_route_event", lambda *args, **kwargs: None)

    # Old droppable turn (~70k tokens) + a protected final turn (~100k tokens):
    # over the 160k big-model budget together, but the final turn alone sits
    # between the small model's 40k budget and the big model's 160k one.
    old_turn = "filler word " * 35_000
    final_turn = "filler word " * 50_000
    if count_tokens([msg(old_turn), msg(final_turn)]) is None:
        return  # tiktoken unavailable — degrade gracefully like tests/test_normalize.py

    calls = []
    monkeypatch.setattr(
        "app.main.chat_completion",
        lambda selected, payload: calls.append(selected.id) or (200, {"choices": [{"message": {"content": "OK"}}]}),
    )

    response = client.post(
        "/v1/chat/completions",
        json={"model": "groq/llama-3.3-70b-versatile", "messages": [msg(old_turn), msg(final_turn)]},
    )

    assert response.status_code == 200
    # Only the big-window model is tried — the small one never had a chance
    # to fit the trimmed result and was excluded from the retry pool up front.
    assert calls == ["groq/llama-3.3-70b-versatile"]


def test_chat_never_calls_known_candidate_that_cannot_fit(monkeypatch):
    roomy, small = model("p/roomy", 1), model("p/small", 2)
    monkeypatch.setattr("app.main.load_registry_with_db_health", lambda: ProviderRegistry([roomy, small]))
    monkeypatch.setattr("app.main.count_tokens", lambda *_: 60_000)
    windows = {"p/roomy": 100_000, "p/small": 50_000}
    monkeypatch.setattr("app.context_policy.context_window", lambda public_id, _: windows.get(public_id))
    monkeypatch.setattr("app.main.context_truncation_trigger_percent", lambda: 100)
    monkeypatch.setattr("app.main.context_truncation_max_tokens", lambda: 32_000)
    monkeypatch.setattr("app.main.persist_route_event", lambda *args, **kwargs: None)
    calls = []

    def provider_call(selected, payload):
        calls.append(selected.id)
        return 200, {"choices": [{"message": {"content": "OK"}}]}

    monkeypatch.setattr("app.main.chat_completion", provider_call)

    response = client.post("/v1/chat/completions", json={"model": "p/roomy", "messages": [msg("large")]})

    assert response.status_code == 200
    assert calls == ["p/roomy"]


def test_chat_falls_back_to_unknown_window_candidate_when_known_fit_fails(monkeypatch):
    # Both fit (known's real window, unknown's fallback budget) — known
    # sorts first (a documented window beats a guess), but a 429 there must
    # still fail over to the unknown-but-fitting candidate.
    known, unknown = model("p/known", 1), model("p/unknown", 2)
    monkeypatch.setattr("app.main.load_registry_with_db_health", lambda: ProviderRegistry([known, unknown]))
    monkeypatch.setattr("app.main.count_tokens", lambda *_: 20_000)
    monkeypatch.setattr("app.context_policy.context_window", lambda public_id, _: {"p/known": 100_000}.get(public_id))
    monkeypatch.setattr("app.main.context_truncation_trigger_percent", lambda: 100)
    monkeypatch.setattr("app.main.context_truncation_max_tokens", lambda: 32_000)
    monkeypatch.setattr("app.main.persist_route_event", lambda *args, **kwargs: None)
    monkeypatch.setattr("app.main.mark_runtime_failure_unhealthy", lambda *args, **kwargs: None)
    calls = []

    def provider_call(selected, payload):
        calls.append(selected.id)
        if selected.id == "p/known":
            return 429, {"error": {"message": "rate limit"}}
        return 200, {"choices": [{"message": {"content": "OK"}}]}

    monkeypatch.setattr("app.main.chat_completion", provider_call)

    response = client.post("/v1/chat/completions", json={"model": "p/known", "messages": [msg("hi")]})

    assert response.status_code == 200
    assert calls == ["p/known", "p/unknown"]


def test_chat_rejects_with_413_when_truncation_disabled_and_nothing_fits(monkeypatch):
    small = model("p/small", 1)
    monkeypatch.setattr("app.main.load_registry_with_db_health", lambda: ProviderRegistry([small]))
    monkeypatch.setattr("app.main.count_tokens", lambda *_: 100_000)
    monkeypatch.setattr("app.context_policy.context_window", lambda *_: 50_000)
    monkeypatch.setattr("app.main.context_truncation_trigger_percent", lambda: 100)
    monkeypatch.setattr("app.main.context_truncation_enabled", lambda: False)
    monkeypatch.setattr("app.main.persist_route_event", lambda *args, **kwargs: None)
    calls = []
    monkeypatch.setattr("app.main.chat_completion", lambda selected, payload: calls.append(selected.id) or (200, {}))

    response = client.post("/v1/chat/completions", json={"model": "p/small", "messages": [msg("large")]})

    assert response.status_code == 413
    assert response.json()["error"]["type"] == "context_too_large"
    assert calls == []


def test_chat_rejects_with_413_when_protected_content_alone_overflows(monkeypatch):
    # Truncation is ON, but the only message (system-less, single turn) is
    # itself bigger than the only candidate's budget — nothing droppable can
    # make it fit, so it must reject rather than send an oversized payload.
    small = model("p/small", 1)
    monkeypatch.setattr("app.main.load_registry_with_db_health", lambda: ProviderRegistry([small]))
    monkeypatch.setattr("app.context_policy.context_window", lambda *_: 50_000)
    monkeypatch.setattr("app.main.context_truncation_enabled", lambda: True)
    monkeypatch.setattr("app.main.context_truncation_trigger_percent", lambda: 100)
    monkeypatch.setattr("app.main.persist_route_event", lambda *args, **kwargs: None)
    calls = []
    monkeypatch.setattr("app.main.chat_completion", lambda selected, payload: calls.append(selected.id) or (200, {}))

    huge_final_turn = "filler word " * 40_000  # ~80k tokens — over the 50k budget alone
    if count_tokens([msg(huge_final_turn)]) is None:
        return

    response = client.post("/v1/chat/completions", json={"model": "p/small", "messages": [msg(huge_final_turn)]})

    assert response.status_code == 413
    assert response.json()["error"]["type"] == "context_too_large"
    assert calls == []


def test_models_endpoint_exposes_virtual_models(monkeypatch):
    monkeypatch.setattr("app.main.load_registry_with_db_health", lambda: ProviderRegistry([model("p1/m", 1)]))

    response = client.get("/v1/models")

    payload = response.json()["data"]
    ids = [item["id"] for item in payload]
    virtual = [item for item in payload if item["id"].startswith("forgerouter/")]
    concrete = next(item for item in payload if item["id"] == "p1/m")
    assert "forgerouter/auto" in ids
    assert "forgerouter/reasoning" in ids
    assert "p1/m" in ids
    assert virtual
    assert all(item["context_length"] == 64_000 for item in virtual)
    assert "context_length" not in concrete


def _patch_context_window(monkeypatch, windows: dict[str, int]):
    fake = lambda public_id, provider_model: windows.get(public_id)
    monkeypatch.setattr("app.pricing.context_window", fake)
    monkeypatch.setattr("app.registry.context_window", fake)
    monkeypatch.setattr("app.main.context_window", fake)
    monkeypatch.setattr("app.context_policy.context_window", fake)


def test_models_endpoint_exposes_real_context_length_for_concrete_models(monkeypatch):
    known = model("groq/llama-3.3-70b-versatile", 1)
    unknown = model("p1/m", 1)
    monkeypatch.setattr("app.main.load_registry_with_db_health", lambda: ProviderRegistry([known, unknown]))
    _patch_context_window(monkeypatch, {"groq/llama-3.3-70b-versatile": 131_072})

    payload = client.get("/v1/models").json()["data"]

    known_entry = next(item for item in payload if item["id"] == "groq/llama-3.3-70b-versatile")
    unknown_entry = next(item for item in payload if item["id"] == "p1/m")
    assert known_entry["context_length"] == 131_072
    assert "context_length" not in unknown_entry


def test_models_endpoint_virtual_context_length_reflects_real_windows(monkeypatch):
    # score 60, reasoning-capable → "complex"/"reasoning" bands.
    big = model("openrouter/deepseek-r1:free", 2, ["text", "reasoning"])
    # score 12 → "simple" band.
    small = model("local/qwen2.5:1.5b", 4)
    monkeypatch.setattr("app.main.load_registry_with_db_health", lambda: ProviderRegistry([big, small]))
    _patch_context_window(monkeypatch, {"openrouter/deepseek-r1:free": 300_000, "local/qwen2.5:1.5b": 70_000})

    payload = client.get("/v1/models").json()["data"]
    virtual = {item["id"]: item["context_length"] for item in payload if item["id"].startswith("forgerouter/")}

    # complex/reasoning route only through the big model → its real window.
    assert virtual["forgerouter/complex"] == 300_000
    assert virtual["forgerouter/reasoning"] == 300_000
    # simple routes through the small model → its (smaller, still >=64k) real window.
    assert virtual["forgerouter/simple"] == 70_000
    # auto can land on any chain, so it takes the worst case across all of them.
    assert virtual["forgerouter/auto"] == 70_000


def test_demand_routes_endpoints(monkeypatch):
    monkeypatch.setattr("app.main.get_demand_routes", lambda: {"simple": ["p1/m"]})
    monkeypatch.setattr("app.main.load_registry_with_db_health", lambda: ProviderRegistry([model("p1/m", 1)]))
    saved = {}
    monkeypatch.setattr("app.main.set_demand_routes", lambda demand, models: saved.update(demand=demand, models=models))

    listing = client.get("/admin/demand-routes")
    update = client.put("/admin/demand-routes/simple", json={"models": ["p1/m", " "]})
    invalid = client.put("/admin/demand-routes/nope", json={"models": []})

    assert listing.status_code == 200
    assert listing.json()["routes"]["simple"] == ["p1/m"]
    assert "forgerouter/auto" in listing.json()["virtual_models"]
    assert update.status_code == 200
    assert saved == {"demand": "simple", "models": ["p1/m"]}
    assert invalid.status_code == 400
