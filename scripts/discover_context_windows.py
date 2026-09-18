#!/usr/bin/env python
"""Active context-window discovery for models the pricing catalog has no
documented window for — see app/validation/context_probe.py for why and how.

Costs real provider quota (up to 5 sequential requests per candidate model),
so this is deliberately opt-in and capped per run rather than wired into the
health scanner. Same three-step CLI/cron pattern as scripts/sync_pricing.py:

    docker run --rm --network foundation_network --add-host=host.docker.internal:host-gateway \
        --env-file .env -e PYTHONPATH=/app forgerouter:latest python3 scripts/discover_context_windows.py --limit 10

Suggested host crontab (a handful of models a day, so a big backlog of
newly-uncatalogued models doesn't all get probed the same morning):
    30 7 * * * cd /path/to/forgerouter && docker run --rm --network foundation_network \
        --add-host=host.docker.internal:host-gateway --env-file .env -e PYTHONPATH=/app \
        forgerouter:latest python3 scripts/discover_context_windows.py --limit 5 \
        >> /var/log/forgerouter-context-discovery.log 2>&1
"""

from __future__ import annotations

import argparse

from app.pricing import context_window, record_discovered_context_window
from app.registry import load_registry_with_db_health
from app.validation.context_probe import probe_context_window


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=10, help="Max models to probe this run")
    args = parser.parse_args()

    registry = load_registry_with_db_health()
    candidates = [
        model for model in registry.models
        if model.enabled and context_window(model.id, model.provider_model) is None
    ][:max(0, args.limit)]

    if not candidates:
        print("No uncatalogued models to probe.")
        return

    for model in candidates:
        result = probe_context_window(model)
        if result.context_window is not None:
            record_discovered_context_window(model.id, model.provider_model, result.context_window, result.note)
            print(f"{model.id}: {result.context_window} ({result.note})")
        else:
            print(f"{model.id}: inconclusive ({result.note})")


if __name__ == "__main__":
    main()
