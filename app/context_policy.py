from dataclasses import dataclass

from app.pricing import context_window
from app.registry import ProviderModel

# Hermes Agent's own hard floor (agent/model_metadata.py: MINIMUM_CONTEXT_LENGTH)
# — a virtual route (forgerouter/auto and friends) may land on any of its
# candidates, so none of them may advertise/serve less than Hermes itself
# requires, even when the raw prompt is tiny and would otherwise fit.
MINIMUM_VIRTUAL_CONTEXT = 64_000


@dataclass(frozen=True)
class ContextPartition:
    fitting: list[ProviderModel]
    excluded: list[ProviderModel]
    max_input_budget: int | None


def model_context_budget(model: ProviderModel, trigger_percent: int, unknown_budget: int) -> tuple[int, bool]:
    """(budget, is_known) — the usable input budget for this model: a real
    percentage of its documented window when known, else the flat fallback
    budget used for catalog-unknown models."""
    window = context_window(model.id, model.provider_model)
    if window is None:
        return unknown_budget, False
    return int(window * trigger_percent / 100), True


def partition_candidates(
    candidates: list[ProviderModel],
    prompt_tokens: int | None,
    trigger_percent: int,
    unknown_budget: int,
    virtual_route: bool,
) -> ContextPartition:
    """Split already-ordered candidates into those whose real input budget
    can hold `prompt_tokens` and those that can't — without disturbing the
    incoming order. Known-fit candidates come first, then unknown-fit ones
    (a documented window is trusted over a guess). When prompt_tokens is
    unknown (tokenizer unavailable), nothing is filtered — a counting
    failure must never turn into an outage."""
    if prompt_tokens is None:
        return ContextPartition(list(candidates), [], None)
    known_fit: list[ProviderModel] = []
    unknown_fit: list[ProviderModel] = []
    excluded: list[ProviderModel] = []
    budgets: list[int] = []
    for candidate in candidates:
        window = context_window(candidate.id, candidate.provider_model)
        budget, known = model_context_budget(candidate, trigger_percent, unknown_budget)
        if virtual_route and known and window < MINIMUM_VIRTUAL_CONTEXT:
            # Never viable regardless of truncation — its budget must not
            # inflate max_input_budget, or a truncation target sized off a
            # candidate that can never be selected could under-trim the
            # prompt for every candidate that actually could have fit.
            excluded.append(candidate)
            continue
        budgets.append(budget)
        if prompt_tokens <= budget:
            (known_fit if known else unknown_fit).append(candidate)
        else:
            excluded.append(candidate)
    return ContextPartition(known_fit + unknown_fit, excluded, max(budgets, default=None))
