"""Reference (notional) cost estimation.

ForgeRouter only routes to free-tier models — paid models are excluded at
discovery (see `_discover_provider_models` in app/main.py) — so providers
almost never report a billed `usage.cost`. This module estimates what a
request would have cost at public commercial rates for an equivalent model,
purely as an opportunity-cost reference. It is never billed and must never
be confused with the real `cost` field in route_events.

Three catalogs are consulted, in order:

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

Lookup is a plain id match — no fuzzy matching. A model with no entry in any
catalog gets no reference cost rather than a guessed one.
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

LITELLM_SOURCE_URL = "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"
_KEPT_MODES = ("chat", "completion")

_live: dict[str, Any] | None = None
_live_failed = False
_overrides: dict[str, Any] | None = None
_overrides_failed = False
_catalog: dict[str, Any] | None = None
_catalog_failed = False


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

    Resolve this field independently across the pricing tiers. A higher-tier
    entry may contain authoritative pricing without context metadata; that
    must not hide a context window available from a lower tier.
    """
    public_id = public_id or ""
    provider_model = provider_model or ""
    catalog = _get_catalog()
    entries = [
        _get_live().get(public_id),
        _get_overrides().get(public_id),
        *(catalog.get(key) for key in (public_id, provider_model, provider_model.rsplit("/", 1)[-1])),
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
        pricing = item.get("pricing")
        if not isinstance(pricing, dict):
            continue
        input_cost = _parse_aggregator_price(pricing.get("prompt"))
        output_cost = _parse_aggregator_price(pricing.get("completion"))
        if input_cost is None or output_cost is None:
            continue
        public_id = f"{provider_name}/{item['id']}"
        result[public_id] = {
            "input_cost_per_token": input_cost,
            "output_cost_per_token": output_cost,
            "source": f"{provider_name} /models pricing (live), synced {synced_at}",
        }
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
