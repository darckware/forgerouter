import json

import app.pricing as pricing_module
from app.pricing import context_window, reference_cost
from app.registry import ProviderModel, ProviderRegistry


def test_reference_cost_matches_public_id():
    # groq/llama-3.3-70b-versatile is a real catalog entry: input 5.9e-7, output 7.9e-7 per token.
    cost = reference_cost("groq/llama-3.3-70b-versatile", "llama-3.3-70b-versatile", 1000, 500)

    assert cost == round(1000 * 5.9e-07 + 500 * 7.9e-07, 8)


def test_reference_cost_falls_back_to_bare_provider_model():
    # No provider prefix match, but the bare provider_model alone is a catalog key.
    cost = reference_cost("some-alias/gpt-4o-mini", "gpt-4o-mini", 100, 50)

    assert cost is not None
    assert cost > 0


def test_reference_cost_none_for_unknown_model():
    assert reference_cost("local/totally-made-up-model", "totally-made-up-model", 100, 50) is None


def test_reference_cost_zero_tokens_is_zero_not_none():
    cost = reference_cost("groq/llama-3.3-70b-versatile", "llama-3.3-70b-versatile", 0, 0)

    assert cost == 0.0


def test_reference_cost_uses_curated_override_for_models_missing_from_bulk_catalog():
    # nvidia/z-ai/glm-5.2 has no entry in the bulk LiteLLM snapshot (too new) —
    # it's only priced via the hand-curated config/model_pricing_overrides.json.
    cost = reference_cost("nvidia/z-ai/glm-5.2", "z-ai/glm-5.2", 1000, 500)

    assert cost == round(1000 * 1.4e-06 + 500 * 4.4e-06, 8)


def test_reference_cost_override_takes_priority_over_bulk_catalog():
    # A public_id present in the overrides file always wins, even if some
    # candidate key would also resolve in the bulk catalog.
    cost = reference_cost("nvidia/meta/llama-3.1-8b-instruct", "meta/llama-3.1-8b-instruct", 1000, 500)

    assert cost == round(1000 * 2e-08 + 500 * 5e-08, 8)


def _isolate_context_tiers(monkeypatch, live=None, overrides=None, model_catalog=None, catalog=None, discovered=None):
    """Full isolation across all five context_window() tiers — the real
    config/model_catalog.json (~60+ canonical entries) and
    config/model_context_discovered.json must never leak into a test that
    didn't ask for them."""
    monkeypatch.setattr(pricing_module, "_live", live or {})
    monkeypatch.setattr(pricing_module, "_overrides", overrides or {})
    monkeypatch.setattr(pricing_module, "_model_catalog", model_catalog or {})
    monkeypatch.setattr(pricing_module, "_catalog", catalog or {})
    monkeypatch.setattr(pricing_module, "_discovered", discovered or {})


def test_context_window_falls_through_price_only_live_entry(monkeypatch):
    _isolate_context_tiers(
        monkeypatch,
        live={"agg/model": {"input_cost_per_token": 0.1, "output_cost_per_token": 0.2}},
        catalog={"model": {"input_cost_per_token": 0.3, "output_cost_per_token": 0.4, "context_window": 131_072}},
    )

    assert context_window("agg/model", "model") == 131_072


def test_context_window_prefers_live_value_when_present(monkeypatch):
    _isolate_context_tiers(
        monkeypatch,
        live={"agg/model": {"context_window": 200_000}},
        overrides={"agg/model": {"context_window": 150_000}},
        catalog={"model": {"context_window": 131_072}},
    )

    assert context_window("agg/model", "model") == 200_000


def test_context_window_falls_through_price_only_override_entry(monkeypatch):
    _isolate_context_tiers(
        monkeypatch,
        overrides={"agg/model": {"input_cost_per_token": 0.1, "output_cost_per_token": 0.2}},
        catalog={"model": {"context_window": 131_072}},
    )

    assert context_window("agg/model", "model") == 131_072


def test_context_window_prefers_model_catalog_over_vendored_catalog(monkeypatch):
    # The canonical model catalog is curated/verified per model, same tier
    # confidence as overrides — it must outrank the generic vendored
    # LiteLLM snapshot, which can be stale or match the wrong variant.
    _isolate_context_tiers(
        monkeypatch,
        model_catalog={"vendor/model": {"context_window": 262_144}},
        catalog={"model": {"context_window": 131_072}},
    )

    assert context_window("agg/vendor/model", "vendor/model") == 262_144


def test_context_window_model_catalog_is_shared_across_provider_aliases(monkeypatch):
    # The whole point of canonical_model_key(): the exact same LLM served by
    # two different ForgeRouter providers, spelled identically except for an
    # aggregator's ":free" marketing suffix, must resolve to the one shared
    # catalog entry without needing a second copy.
    _isolate_context_tiers(monkeypatch, model_catalog={"vendor/model": {"context_window": 262_144}})

    assert context_window("Kilo/vendor/model:free", "vendor/model:free") == 262_144
    assert context_window("nvidia/vendor/model", "vendor/model") == 262_144


def test_canonical_model_key_applies_curated_alias(monkeypatch):
    # A vendor slug two providers spell differently for the identical model —
    # canonical_model_key() can't guess this; it's the curated
    # _CANONICAL_ALIASES table's job.
    monkeypatch.setattr(pricing_module, "_CANONICAL_ALIASES", {"vendor-a/model": "vendor-b/model"})

    assert pricing_module.canonical_model_key("vendor-a/model") == "vendor-b/model"
    assert pricing_module.canonical_model_key("vendor-a/model:free") == "vendor-b/model"


def test_sync_provider_pricing_parses_aggregator_pricing(monkeypatch, tmp_path):
    # Isolate the live-tier cache file/globals so this never touches the real
    # config/model_pricing_live.json or leaks state into other tests.
    monkeypatch.setattr(pricing_module, "_LIVE_PATH", tmp_path / "model_pricing_live.json")
    monkeypatch.setattr(pricing_module, "_live", None)
    monkeypatch.setattr(pricing_module, "_live_failed", False)

    class FakeResponse:
        def json(self):
            return {
                "data": [
                    {"id": "some-model", "pricing": {"prompt": "0.000001", "completion": "0.000002"}},
                    {"id": "no-pricing-model"},
                    {"id": "free-model", "pricing": {"prompt": "0", "completion": "0"}},
                ]
            }

    import httpx

    monkeypatch.setattr(httpx, "get", lambda *args, **kwargs: FakeResponse())

    registry = ProviderRegistry(
        [ProviderModel("agg/some-model", "agg", "some-model", 1, ["text"], True, True, "https://agg.example/v1", "")]
    )

    count = pricing_module.sync_provider_pricing(registry)

    assert count == 2  # some-model + free-model; no-pricing-model has no pricing field
    cost = reference_cost("agg/some-model", "some-model", 1000, 500)
    assert cost == round(1000 * 0.000001 + 500 * 0.000002, 8)
    assert reference_cost("agg/free-model", "free-model", 1000, 500) == 0.0


def test_live_tier_takes_priority_over_curated_override(monkeypatch, tmp_path):
    monkeypatch.setattr(pricing_module, "_LIVE_PATH", tmp_path / "model_pricing_live.json")
    monkeypatch.setattr(pricing_module, "_live", None)
    monkeypatch.setattr(pricing_module, "_live_failed", False)

    class FakeResponse:
        def json(self):
            # A different (fresher, live) price than the curated override for this exact model.
            return {"data": [{"id": "z-ai/glm-5.2", "pricing": {"prompt": "0.000002", "completion": "0.000006"}}]}

    import httpx

    monkeypatch.setattr(httpx, "get", lambda *args, **kwargs: FakeResponse())

    registry = ProviderRegistry(
        [ProviderModel("nvidia/z-ai/glm-5.2", "nvidia", "z-ai/glm-5.2", 1, ["text"], True, True, "https://nvidia.example/v1", "")]
    )
    pricing_module.sync_provider_pricing(registry)

    cost = reference_cost("nvidia/z-ai/glm-5.2", "z-ai/glm-5.2", 1000, 500)
    assert cost == round(1000 * 0.000002 + 500 * 0.000006, 8)  # live number, not the 1.4e-6/4.4e-6 override


def test_sync_provider_pricing_also_captures_context_length_without_pricing(monkeypatch, tmp_path):
    # An aggregator /models response can carry context_length on a model that
    # has no pricing object at all (or an unparseable one) — that context data
    # is real and already being fetched; it must not be discarded just because
    # the pricing half of the same entry didn't parse.
    monkeypatch.setattr(pricing_module, "_LIVE_PATH", tmp_path / "model_pricing_live.json")
    monkeypatch.setattr(pricing_module, "_live", None)
    monkeypatch.setattr(pricing_module, "_live_failed", False)
    monkeypatch.setattr(pricing_module, "_model_catalog", {})
    monkeypatch.setattr(pricing_module, "_discovered", {})

    class FakeResponse:
        def json(self):
            return {"data": [{"id": "no-pricing-but-context", "context_length": 200_000}]}

    import httpx

    monkeypatch.setattr(httpx, "get", lambda *args, **kwargs: FakeResponse())

    registry = ProviderRegistry(
        [ProviderModel("agg/no-pricing-but-context", "agg", "no-pricing-but-context", 1, ["text"], True, True, "https://agg.example/v1", "")]
    )
    count = pricing_module.sync_provider_pricing(registry)

    assert count == 1
    assert context_window("agg/no-pricing-but-context", "no-pricing-but-context") == 200_000
    # No cost fields on this entry — reference_cost treats a missing
    # input/output rate as 0, same as any other context-only catalog entry
    # (e.g. the safety-classifier overrides that only carry context_window).
    assert reference_cost("agg/no-pricing-but-context", "no-pricing-but-context", 100, 50) == 0.0


def test_context_window_falls_through_to_discovered_tier_as_last_resort(monkeypatch):
    # Discovered is keyed canonically (by provider_model, decoration-stripped)
    # — same as config/model_catalog.json — not by the exact public_id.
    _isolate_context_tiers(monkeypatch, discovered={"nvidia/some-model": {"context_window": 128_000}})

    assert context_window("nvidia/nvidia/some-model", "nvidia/some-model") == 128_000


def test_context_window_prefers_catalog_over_discovered(monkeypatch):
    # A documented window always wins over a probed lower bound, even if the
    # probe happened to find a bigger number — the catalog entry is real,
    # confirmed data; the probe is just a confirmed floor.
    _isolate_context_tiers(
        monkeypatch,
        catalog={"model": {"context_window": 131_072}},
        discovered={"model": {"context_window": 256_000}},
    )

    assert context_window("agg/model", "model") == 131_072


def test_record_discovered_context_window_persists_and_updates_cache(monkeypatch, tmp_path):
    discovered_path = tmp_path / "model_context_discovered.json"
    monkeypatch.setattr(pricing_module, "_DISCOVERED_PATH", discovered_path)
    monkeypatch.setattr(pricing_module, "_discovered", None)
    monkeypatch.setattr(pricing_module, "_discovered_failed", False)
    monkeypatch.setattr(pricing_module, "_live", {})
    monkeypatch.setattr(pricing_module, "_overrides", {})
    monkeypatch.setattr(pricing_module, "_model_catalog", {})
    monkeypatch.setattr(pricing_module, "_catalog", {})

    pricing_module.record_discovered_context_window("nvidia/nvidia/some-model", "nvidia/some-model", 128_000, "confirmed lower bound")

    assert context_window("nvidia/nvidia/some-model", "nvidia/some-model") == 128_000
    saved = json.loads(discovered_path.read_text())
    # Keyed canonically (the provider_model, unchanged here since it has no
    # aggregator suffix to strip) so a different alias of the same LLM
    # benefits from this one probe result too.
    assert saved["nvidia/some-model"]["context_window"] == 128_000
    assert "confirmed lower bound" in saved["nvidia/some-model"]["source"]


def test_record_discovered_context_window_strips_aggregator_suffix_from_key(monkeypatch, tmp_path):
    discovered_path = tmp_path / "model_context_discovered.json"
    monkeypatch.setattr(pricing_module, "_DISCOVERED_PATH", discovered_path)
    monkeypatch.setattr(pricing_module, "_discovered", None)
    monkeypatch.setattr(pricing_module, "_discovered_failed", False)
    _isolate_context_tiers(monkeypatch)

    pricing_module.record_discovered_context_window("Kilo/vendor/model:free", "vendor/model:free", 128_000)

    # A sibling alias without the aggregator suffix now resolves too.
    assert context_window("nvidia/vendor/model", "vendor/model") == 128_000
