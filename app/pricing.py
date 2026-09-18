"""Reference (notional) cost estimation.

ForgeRouter only routes to free-tier models — paid models are excluded at
discovery (see `_discover_provider_models` in app/main.py) — so providers
almost never report a billed `usage.cost`. This module estimates what a
request would have cost at public commercial rates for an equivalent model,
purely as an opportunity-cost reference. It is never billed and must never
be confused with the real `cost` field in route_events.

Three catalogs are consulted, in order, for *pricing* (cost genuinely can
differ per provider — the same underlying model can be sold at different
rates by different resellers, so pricing stays keyed by the exact
ForgeRouter public_id, never shared across aliases):

1. config/model_pricing_live.json — pricing read directly from the /models
   response of the providers ForgeRouter actually routes through (OpenRouter,
   Kilo, and any other aggregator whose catalog carries a `pricing` object —
   the same shape app/ranking.py::is_free_model already reads to classify
   free/paid). This is the most authoritative source: it's the literal
   endpoint the request goes to, not a guessed equivalent. Refreshed by
   `sync_provider_pricing()` (wired into POST /admin/pricing/sync), keyed by
   the exact ForgeRouter public_id ("<provider>/<model_id>").
2. config/model_pricing_overrides.json — hand-curated entries for models
   neither of the other two catalogs have (usually because they're too new,
   or the provider's own /models endpoint doesn't publish pricing). Keyed by
   the exact ForgeRouter public_id we saw in route_events, each entry carries
   a `source` — the page the price was verified against — so it's auditable
   and re-checkable later. This file is never touched by sync.
3. config/model_pricing.json — a trimmed snapshot of LiteLLM's public
   `model_prices_and_context_window.json` (chat-capable models only,
   input/output cost per token) — refresh it periodically by re-running
   scripts/update_pricing.py, or via sync, against the upstream file.

Lookup for pricing is a plain id match — no fuzzy matching. A model with no
entry in any catalog gets no reference cost rather than a guessed one.

Context window is a different story: the same underlying LLM is routinely
served by several different ForgeRouter providers under several different
names (the same GLM-5.2 weights show up as "Kilo/z-ai/glm-5.2:free",
"openrouter/z-ai/glm-5.2:free" and "nvidia/z-ai/glm-5.2" — three ForgeRouter
public_ids, one real model). Keying context data per exact public_id like
pricing does meant a real, verified context window had to be hand-copied
into every alias separately (2026-09, the incident that motivated this) —
a model added under a *new* alias of an already-known LLM silently came
back "unknown" and lost forgerouter/auto eligibility until someone
remembered to copy the number over again. `config/model_catalog.json` fixes
that: it's keyed by `canonical_model_key()` — the underlying vendor model
slug with aggregator marketing suffixes (":free", ...) stripped, plus a
small curated alias table (`_CANONICAL_ALIASES`) for the rarer case where
different providers spell the same vendor slug differently (e.g. NVIDIA's
own NIM API calls a model "stepfun-ai/step-3.7-flash" where OpenRouter/Kilo
call the identical weights "stepfun/step-3.7-flash"). One entry there now
covers every alias of that model, present or future, automatically — no
per-alias copy needed. `context_window()` checks it as an additional tier;
`app.validation.context_probe`'s discovered results are keyed the same way,
so a single probe benefits every alias too.
"""

from __future__ import annotations

import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
_LIVE_PATH = _CONFIG_DIR / "model_pricing_live.json"
_OVERRIDES_PATH = _CONFIG_DIR / "model_pricing_overrides.json"
_CATALOG_PATH = _CONFIG_DIR / "model_pricing.json"
_DISCOVERED_PATH = _CONFIG_DIR / "model_context_discovered.json"
_MODEL_CATALOG_PATH = _CONFIG_DIR / "model_catalog.json"

LITELLM_SOURCE_URL = "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"
_KEPT_MODES = ("chat", "completion")

# Aggregator-added marketing decoration on the vendor's own model slug — never
# part of the underlying model's real identity, so it must be stripped before
# two aliases of the same LLM can be recognized as the same canonical key.
_AGGREGATOR_SUFFIXES = (":free", ":extended", ":nitro", ":online", ":beta", ":thinking")

# Curated for the rarer case where different providers spell the exact same
# vendor model slug differently — canonical_model_key() can't guess these,
# so each one needs a one-line entry here (never a full duplicate catalog
# entry). Add to this only when you've confirmed it's really the same model.
_CANONICAL_ALIASES: dict[str, str] = {
    "stepfun-ai/step-3.7-flash": "stepfun/step-3.7-flash",
}

_live: dict[str, Any] | None = None
_live_failed = False
_overrides: dict[str, Any] | None = None
_overrides_failed = False
_catalog: dict[str, Any] | None = None
_catalog_failed = False
_discovered: dict[str, Any] | None = None
_discovered_failed = False
_model_catalog: dict[str, Any] | None = None
_model_catalog_failed = False


def canonical_model_key(provider_model: str) -> str:
    """The underlying vendor model slug, decoration-stripped — the shared key
    every ForgeRouter alias of the same real LLM resolves to in
    config/model_catalog.json, instead of needing its own duplicate entry."""
    key = (provider_model or "").strip()
    for suffix in _AGGREGATOR_SUFFIXES:
        if key.endswith(suffix):
            key = key[: -len(suffix)]
            break
    return _CANONICAL_ALIASES.get(key, key)


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _get_live() -> dict[str, Any]:
    global _live, _live_failed
    if _live is None and not _live_failed:
        try:
            _live = _load_json(_LIVE_PATH)
        except Exception:
            _live_failed = True
            _live = {}
    return _live or {}


def _get_overrides() -> dict[str, Any]:
    global _overrides, _overrides_failed
    if _overrides is None and not _overrides_failed:
        try:
            _overrides = _load_json(_OVERRIDES_PATH)
        except Exception:
            _overrides_failed = True
            _overrides = {}
    return _overrides or {}


def _get_catalog() -> dict[str, Any]:
    global _catalog, _catalog_failed
    if _catalog is None and not _catalog_failed:
        try:
            _catalog = _load_json(_CATALOG_PATH)
        except Exception:
            _catalog_failed = True
            _catalog = {}
    return _catalog or {}


def _get_discovered() -> dict[str, Any]:
    global _discovered, _discovered_failed
    if _discovered is None and not _discovered_failed:
        try:
            _discovered = _load_json(_DISCOVERED_PATH)
        except Exception:
            _discovered_failed = True
            _discovered = {}
    return _discovered or {}


def _get_model_catalog() -> dict[str, Any]:
    global _model_catalog, _model_catalog_failed
    if _model_catalog is None and not _model_catalog_failed:
        try:
            _model_catalog = _load_json(_MODEL_CATALOG_PATH)
        except Exception:
            _model_catalog_failed = True
            _model_catalog = {}
    return _model_catalog or {}


def record_discovered_context_window(public_id: str, provider_model: str, window: int, note: str = "") -> None:
    """Persist a context window app.validation.context_probe actually
    confirmed by testing the live model — the lowest-priority tier in
    context_window() below, since it's a confirmed lower bound from an
    active probe rather than something the provider documents. Never call
    this with a guessed/inferred value; only a probe result.

    Keyed by canonical_model_key(provider_model), not the exact public_id —
    the same real LLM under a different ForgeRouter alias benefits from this
    one probe result too, instead of needing its own separate probe."""
    global _discovered, _discovered_failed
    key = canonical_model_key(provider_model)
    discovered = dict(_get_discovered())
    discovered[key] = {
        "context_window": int(window),
        "source": f"forgerouter context probe against {public_id}, "
        f"{datetime.now(timezone.utc).date().isoformat()}" + (f" ({note})" if note else ""),
    }
    with open(_DISCOVERED_PATH, "w", encoding="utf-8") as fh:
        json.dump(discovered, fh, indent=1, sort_keys=True)
        fh.write("\n")
    _discovered = discovered
    _discovered_failed = False


def _lookup(public_id: str, provider_model: str) -> dict[str, Any] | None:
    live = _get_live()
    if public_id in live:
        return live[public_id]

    overrides = _get_overrides()
    if public_id in overrides:
        return overrides[public_id]

    catalog = _get_catalog()
    candidates = [public_id, provider_model, provider_model.rsplit("/", 1)[-1]]
    for key in candidates:
        entry = catalog.get(key)
        if entry:
            return entry
    return None


def context_window(public_id: str, provider_model: str) -> int | None:
    """The model's real input context window in tokens.

    Resolve this field independently across the tiers below. A higher-tier
    entry may contain authoritative pricing without context metadata; that
    must not hide a context window available from a lower tier.

    Tier order: live (exact public_id, the literal endpoint) → overrides
    (exact public_id, for a deliberately-different single alias) → the
    canonical model catalog (config/model_catalog.json, shared across every
    alias of the same real LLM — see module docstring) → the vendored
    LiteLLM catalog (public_id / provider_model / bare suffix) → discovered
    (config/model_context_discovered.json, written by
    app.validation.context_probe, also canonical-keyed) as the last resort —
    it's a confirmed lower bound from actively testing the live model, not
    something the provider documents, so a documented number always wins
    when one exists.
    """
    public_id = public_id or ""
    provider_model = provider_model or ""
    canonical_key = canonical_model_key(provider_model)
    catalog = _get_catalog()
    entries = [
        _get_live().get(public_id),
        _get_overrides().get(public_id),
        _get_model_catalog().get(canonical_key),
        *(catalog.get(key) for key in (public_id, provider_model, provider_model.rsplit("/", 1)[-1])),
        _get_discovered().get(canonical_key),
    ]
    for entry in entries:
        window = entry.get("context_window") if isinstance(entry, dict) else None
        if isinstance(window, (int, float)) and window > 0:
            return int(window)
    return None


def reference_cost(public_id: str, provider_model: str, prompt_tokens: int, completion_tokens: int) -> float | None:
    """Notional USD cost had this request been billed at public commercial
    rates for an equivalent model. None when the model has no catalog match."""
    entry = _lookup(public_id or "", provider_model or "")
    if entry is None:
        return None
    input_cost = float(entry.get("input_cost_per_token") or 0)
    output_cost = float(entry.get("output_cost_per_token") or 0)
    return round(max(prompt_tokens, 0) * input_cost + max(completion_tokens, 0) * output_cost, 8)


def resolve_price_info(public_id: str, provider_model: str) -> dict[str, Any] | None:
    """Like _lookup, but for display purposes (the admin pricing page) rather
    than a cost computation — returns the raw matched entry (input/output
    cost per token, plus `source` when it came from the curated overrides)
    or None when nothing matched."""
    entry = _lookup(public_id or "", provider_model or "")
    return dict(entry) if entry else None


def _trim_litellm_catalog(raw: dict[str, Any]) -> dict[str, Any]:
    trimmed = {}
    for model_id, entry in raw.items():
        if not isinstance(entry, dict) or entry.get("mode") not in _KEPT_MODES:
            continue
        input_cost = entry.get("input_cost_per_token")
        output_cost = entry.get("output_cost_per_token")
        if not isinstance(input_cost, (int, float)) or not isinstance(output_cost, (int, float)):
            continue
        kept = {"input_cost_per_token": input_cost, "output_cost_per_token": output_cost}
        # Context window — used by app/normalize.py's truncation trigger (percent
        # of the *actual* selected model's window, not a one-size-fits-all
        # constant). max_input_tokens is the input-side limit; max_tokens is
        # older LiteLLM data's only field for models that predate the split.
        window = entry.get("max_input_tokens") or entry.get("max_tokens")
        if isinstance(window, (int, float)) and window > 0:
            kept["context_window"] = int(window)
        trimmed[model_id] = kept
    return trimmed


def sync_catalog_from_litellm(timeout: float = 30.0) -> int:
    """Refresh config/model_pricing.json from LiteLLM's public catalog (the
    same filter as scripts/update_pricing.py). Returns the number of entries
    written. Resets the in-memory cache so subsequent lookups in this process
    see the new data without a restart."""
    global _catalog, _catalog_failed
    with urllib.request.urlopen(LITELLM_SOURCE_URL, timeout=timeout) as response:
        raw = json.load(response)
    trimmed = _trim_litellm_catalog(raw)
    with open(_CATALOG_PATH, "w", encoding="utf-8") as fh:
        json.dump(trimmed, fh, indent=1, sort_keys=True)
        fh.write("\n")
    _catalog = trimmed
    _catalog_failed = False
    return len(trimmed)


def _parse_aggregator_price(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _fetch_provider_pricing(
    provider_name: str, base_url: str, models: list[Any], timeout: float, synced_at: str
) -> dict[str, Any]:
    import os

    import httpx

    result: dict[str, Any] = {}
    if not base_url:
        return result
    api_key = next((m.api_key for m in models if m.api_key), "")
    if not api_key:
        env_name = next((m.api_key_env for m in models if m.api_key_env), "")
        api_key = os.environ.get(env_name, "") if env_name else ""
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        response = httpx.get(base_url.rstrip("/") + "/models", headers=headers, timeout=timeout)
        body = response.json()
    except Exception:
        return result
    items = body.get("data") if isinstance(body, dict) else None
    if not isinstance(items, list):
        return result
    for item in items:
        if not isinstance(item, dict) or not item.get("id"):
            continue
        entry: dict[str, Any] = {}
        pricing = item.get("pricing")
        if isinstance(pricing, dict):
            input_cost = _parse_aggregator_price(pricing.get("prompt"))
            output_cost = _parse_aggregator_price(pricing.get("completion"))
            if input_cost is not None and output_cost is not None:
                entry["input_cost_per_token"] = input_cost
                entry["output_cost_per_token"] = output_cost
                entry["source"] = f"{provider_name} /models pricing (live), synced {synced_at}"
        # Aggregators (OpenRouter, Kilo) also publish context_length in the same
        # /models response — real, documented data already being fetched here for
        # pricing, so capture it too instead of discarding it. This is the same
        # literal endpoint being routed through, the highest-priority tier in
        # app.pricing.context_window().
        context_length = item.get("context_length")
        if isinstance(context_length, (int, float)) and context_length > 0:
            entry["context_window"] = int(context_length)
        if entry:
            public_id = f"{provider_name}/{item['id']}"
            result[public_id] = entry
    return result


def sync_provider_pricing(registry: Any, timeout: float = 10.0, max_workers: int = 8) -> int:
    """Read live pricing straight from the /models response of every provider
    ForgeRouter is currently registered against (one request per distinct
    base_url, not per model, fetched in parallel — same ThreadPoolExecutor
    pattern as _discover_provider_models's health scan in app/main.py, so one
    slow/dead provider doesn't serialize the whole sync behind its timeout).
    Aggregators like OpenRouter and Kilo publish a `pricing: {prompt,
    completion}` object per model — the same field
    app/ranking.py::is_free_model already reads to classify free vs paid —
    this captures the actual numbers instead of discarding them.

    A provider with no /models endpoint, no pricing field, or that errors/
    times out is silently skipped; sync must never fail because one provider
    is unreachable. Returns the number of priced entries written."""
    from concurrent.futures import ThreadPoolExecutor

    by_provider: dict[tuple[str, str], list[Any]] = {}
    for model in registry.models:
        by_provider.setdefault((model.provider, model.base_url), []).append(model)

    synced_at = datetime.now(timezone.utc).date().isoformat()

    def fetch_entry(entry: tuple[tuple[str, str], list[Any]]) -> dict[str, Any]:
        (provider_name, base_url), models = entry
        return _fetch_provider_pricing(provider_name, base_url, models, timeout, synced_at)

    live: dict[str, Any] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for partial in executor.map(fetch_entry, by_provider.items()):
            live.update(partial)

    global _live, _live_failed
    with open(_LIVE_PATH, "w", encoding="utf-8") as fh:
        json.dump(live, fh, indent=1, sort_keys=True)
        fh.write("\n")
    _live = live
    _live_failed = False
    return len(live)
