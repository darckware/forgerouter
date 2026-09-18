from fastapi.testclient import TestClient

from app.main import app
from app.registry import ProviderModel, ProviderRegistry

client = TestClient(app)


def _model(model_id: str, provider_model: str) -> ProviderModel:
    return ProviderModel(model_id, "p1", provider_model, 1, ["text"], True, True, "http://first/v1", "")


def test_admin_pricing_models_flags_priced_and_unpriced(monkeypatch):
    registry = ProviderRegistry([
        _model("nvidia/z-ai/glm-5.2", "z-ai/glm-5.2"),  # curated override
        _model("local/totally-made-up-model", "totally-made-up-model"),  # no match anywhere
    ])
    monkeypatch.setattr("app.main.load_registry_with_db_health", lambda: registry)

    response = client.get("/admin/pricing/models")

    assert response.status_code == 200
    body = response.json()
    by_id = {m["public_id"]: m for m in body["models"]}
    assert by_id["nvidia/z-ai/glm-5.2"]["priced"] is True
    assert by_id["nvidia/z-ai/glm-5.2"]["input_cost_per_token"] > 0
    assert by_id["nvidia/z-ai/glm-5.2"]["source"]
    assert by_id["local/totally-made-up-model"]["priced"] is False
    assert body["priced_count"] == 1
    assert body["total_count"] == 2


def test_admin_pricing_sync_requires_admin(monkeypatch):
    monkeypatch.setattr("app.main.has_any_agent", lambda: True)
    monkeypatch.setattr("app.main.find_agent_by_key", lambda key: None)

    response = client.post("/admin/pricing/sync")

    assert response.status_code == 401


def test_admin_pricing_sync_refreshes_catalog_and_backfills(monkeypatch):
    monkeypatch.setattr("app.main.has_any_agent", lambda: True)
    monkeypatch.setattr("app.main.find_agent_by_key", lambda key: "tester" if key == "secret" else None)
    monkeypatch.setattr("app.main.load_registry_with_db_health", lambda: ProviderRegistry([]))
    monkeypatch.setattr("app.pricing.sync_catalog_from_litellm", lambda: 2214)
    monkeypatch.setattr("app.pricing.sync_provider_pricing", lambda registry: 57)
    monkeypatch.setattr("app.main.backfill_reference_costs", lambda: (100, 42))
    monkeypatch.setattr("app.main.set_setting", lambda key, value: None)

    response = client.post("/admin/pricing/sync", headers={"Authorization": "Bearer secret"})

    assert response.status_code == 200
    body = response.json()
    last_synced = body.pop("last_synced")
    assert last_synced  # non-empty ISO timestamp; exact value isn't asserted
    assert body == {
        "status": "synced",
        "catalog_entries": 2214,
        "live_entries": 57,
        "backfill_checked": 100,
        "backfill_priced": 42,
    }


def test_admin_pricing_sync_reports_502_on_fetch_failure(monkeypatch):
    monkeypatch.setattr("app.main.has_any_agent", lambda: True)
    monkeypatch.setattr("app.main.find_agent_by_key", lambda key: "tester" if key == "secret" else None)

    def boom():
        raise RuntimeError("network unreachable")

    monkeypatch.setattr("app.pricing.sync_catalog_from_litellm", boom)

    response = client.post("/admin/pricing/sync", headers={"Authorization": "Bearer secret"})

    assert response.status_code == 502
    assert response.json()["error"]["type"] == "pricing_sync_failed"


def test_admin_discover_context_requires_admin(monkeypatch):
    monkeypatch.setattr("app.main.has_any_agent", lambda: True)
    monkeypatch.setattr("app.main.find_agent_by_key", lambda key: None)

    response = client.post("/admin/pricing/discover-context")

    assert response.status_code == 401


def test_admin_discover_context_skips_already_catalogued_models(monkeypatch):
    from app.validation.context_probe import ContextProbeResult

    monkeypatch.setattr("app.main.has_any_agent", lambda: True)
    monkeypatch.setattr("app.main.find_agent_by_key", lambda key: "tester" if key == "secret" else None)
    registry = ProviderRegistry([
        _model("groq/llama-3.3-70b-versatile", "llama-3.3-70b-versatile"),  # already catalogued
        _model("local/totally-made-up-model", "totally-made-up-model"),  # unknown — must be probed
    ])
    monkeypatch.setattr("app.main.load_registry_with_db_health", lambda: registry)
    probed = []

    def fake_probe(model, timeout=60.0):
        probed.append(model.id)
        return ContextProbeResult(model.id, 128_000, [(64_000, "fit")], "confirmed lower bound")

    monkeypatch.setattr("app.validation.context_probe.probe_context_window", fake_probe)
    persisted = {}
    monkeypatch.setattr("app.pricing.record_discovered_context_window", lambda public_id, window, note="": persisted.update({public_id: window}))

    response = client.post("/admin/pricing/discover-context", headers={"Authorization": "Bearer secret"})

    assert response.status_code == 200
    assert probed == ["local/totally-made-up-model"]
    assert persisted == {"local/totally-made-up-model": 128_000}
    assert response.json()["probed"] == 1


def test_admin_discover_context_does_not_persist_inconclusive_results(monkeypatch):
    from app.validation.context_probe import ContextProbeResult

    monkeypatch.setattr("app.main.has_any_agent", lambda: True)
    monkeypatch.setattr("app.main.find_agent_by_key", lambda key: "tester" if key == "secret" else None)
    registry = ProviderRegistry([_model("local/totally-made-up-model", "totally-made-up-model")])
    monkeypatch.setattr("app.main.load_registry_with_db_health", lambda: registry)
    monkeypatch.setattr(
        "app.validation.context_probe.probe_context_window",
        lambda model, timeout=60.0: ContextProbeResult(model.id, None, [], "floor checkpoint inconclusive: inconclusive:rate_limited"),
    )
    persist_calls = []
    monkeypatch.setattr("app.pricing.record_discovered_context_window", lambda *a, **k: persist_calls.append((a, k)))

    response = client.post("/admin/pricing/discover-context", headers={"Authorization": "Bearer secret"})

    assert response.status_code == 200
    assert persist_calls == []
    assert response.json()["results"][0]["context_window"] is None


def test_admin_discover_context_respects_limit(monkeypatch):
    from app.validation.context_probe import ContextProbeResult

    monkeypatch.setattr("app.main.has_any_agent", lambda: True)
    monkeypatch.setattr("app.main.find_agent_by_key", lambda key: "tester" if key == "secret" else None)
    registry = ProviderRegistry([
        _model(f"local/unknown-{i}", f"unknown-{i}") for i in range(5)
    ])
    monkeypatch.setattr("app.main.load_registry_with_db_health", lambda: registry)
    monkeypatch.setattr(
        "app.validation.context_probe.probe_context_window",
        lambda model, timeout=60.0: ContextProbeResult(model.id, 128_000, [], "confirmed lower bound"),
    )
    monkeypatch.setattr("app.pricing.record_discovered_context_window", lambda *a, **k: None)

    response = client.post("/admin/pricing/discover-context?limit=2", headers={"Authorization": "Bearer secret"})

    assert response.status_code == 200
    assert response.json()["probed"] == 2
