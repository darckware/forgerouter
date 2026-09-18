from app.context_policy import partition_candidates
from app.registry import ProviderModel


def model(public_id: str, tier: int) -> ProviderModel:
    return ProviderModel(
        public_id, public_id.split("/")[0], public_id.split("/", 1)[1],
        tier, ["text"], True, True, "http://provider.test/v1", "",
    )


def test_partition_keeps_known_fit_and_puts_unknown_last_for_concrete_route(monkeypatch):
    # Non-virtual (concrete-model) requests still trust an uncatalogued window —
    # the caller named that model on purpose, unlike a virtual route which could
    # land on any candidate.
    roomy, unknown, small = model("p/roomy", 1), model("p/unknown", 1), model("p/small", 1)
    windows = {"p/roomy": 200_000, "p/small": 70_000}
    monkeypatch.setattr("app.context_policy.context_window", lambda public_id, _: windows.get(public_id))

    result = partition_candidates(
        [unknown, small, roomy], prompt_tokens=60_000,
        trigger_percent=80, unknown_budget=64_000, virtual_route=False,
    )

    assert [m.id for m in result.fitting] == ["p/roomy", "p/unknown"]
    assert [m.id for m in result.excluded] == ["p/small"]
    assert result.max_input_budget == 160_000


def test_partition_excludes_uncatalogued_window_from_virtual_route(monkeypatch):
    # A virtual route (forgerouter/auto and friends) can land on any candidate,
    # so an uncatalogued window can't be trusted to clear the 64k floor the way
    # it's trusted for a concrete-model request — it must be excluded outright,
    # not passed through on the unknown-budget fallback.
    roomy, unknown = model("p/roomy", 1), model("p/unknown", 1)
    windows = {"p/roomy": 200_000}
    monkeypatch.setattr("app.context_policy.context_window", lambda public_id, _: windows.get(public_id))

    result = partition_candidates(
        [unknown, roomy], prompt_tokens=60_000,
        trigger_percent=80, unknown_budget=64_000, virtual_route=True,
    )

    assert [m.id for m in result.fitting] == ["p/roomy"]
    assert [m.id for m in result.excluded] == ["p/unknown"]
    assert result.max_input_budget == 160_000


def test_partition_excludes_known_sub_64k_from_virtual_route(monkeypatch):
    tiny = model("p/tiny", 1)
    monkeypatch.setattr("app.context_policy.context_window", lambda *_: 32_000)
    assert partition_candidates([tiny], 1_000, 80, 32_000, virtual_route=True).fitting == []
    assert partition_candidates([tiny], 1_000, 80, 32_000, virtual_route=False).fitting == [tiny]


def test_partition_excludes_unknown_above_fallback_budget(monkeypatch):
    unknown = model("p/unknown", 1)
    monkeypatch.setattr("app.context_policy.context_window", lambda *_: None)
    result = partition_candidates([unknown], 40_000, 80, 32_000, virtual_route=False)
    assert result.fitting == []
    assert result.max_input_budget == 32_000


def test_partition_preserves_availability_when_count_is_unknown(monkeypatch):
    candidate = model("p/model", 1)
    monkeypatch.setattr("app.context_policy.context_window", lambda *_: 8_000)
    assert partition_candidates([candidate], None, 80, 32_000, virtual_route=True).fitting == [candidate]


def test_partition_excludes_sub_64k_budget_from_truncation_target(monkeypatch):
    # A candidate hard-excluded by the virtual-route floor (known-sub-floor,
    # or — since neither is trustworthy for a virtual route — uncatalogued)
    # is never viable regardless of truncation — its (possibly larger)
    # budget must not inflate max_input_budget, or the truncation target
    # could end up too generous to actually fit the candidates that ARE
    # still in play. Here nothing survives the floor, so there's no budget
    # left to compute at all.
    unknown = model("p/unknown", 1)
    sub_floor = model("p/known-small", 1)
    windows = {"p/known-small": 50_000}
    monkeypatch.setattr("app.context_policy.context_window", lambda public_id, _: windows.get(public_id))

    result = partition_candidates(
        [unknown, sub_floor], prompt_tokens=35_000,
        trigger_percent=100, unknown_budget=32_000, virtual_route=True,
    )

    assert result.fitting == []
    assert [m.id for m in result.excluded] == ["p/unknown", "p/known-small"]
    assert result.max_input_budget is None
