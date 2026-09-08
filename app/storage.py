from __future__ import annotations

import hashlib
import os
import secrets
from datetime import datetime, timezone
from typing import Any

import psycopg
from psycopg.types.json import Json

from app.pricing import reference_cost as estimate_reference_cost
from app.validation.health import HealthResult


def health_to_row(result: HealthResult) -> dict[str, Any]:
    return {
        "model_id": result.model_id,
        "status": result.status,
        "http_code": result.http_code,
        "latency_ms": result.latency_ms,
        "error_message": result.error_message,
        "cooldown_seconds": getattr(result, "cooldown_seconds", None),
    }


def database_url() -> str:
    value = os.environ.get("DATABASE_URL")
    if not value:
        raise RuntimeError("DATABASE_URL is not configured")
    return value


def db_connect() -> psycopg.Connection:
    return psycopg.connect(database_url(), connect_timeout=int(os.environ.get("DATABASE_CONNECT_TIMEOUT", "3")))


def get_postgres_version() -> str:
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT version()")
            row = cur.fetchone()
            return str(row[0]) if row else "unknown"


def persist_health_results(results: list[HealthResult]) -> int:
    if not results:
        return 0

    with db_connect() as conn:
        with conn.cursor() as cur:
            for result in results:
                row = health_to_row(result)
                cur.execute(
                    """
                    INSERT INTO ai_router.provider_health
                        (model_id, status, http_code, latency_ms, error_message, cooldown_seconds)
                    VALUES (
                        (SELECT model_id FROM ai_router.models WHERE public_id = %(model_id)s),
                        %(status)s, %(http_code)s, %(latency_ms)s, %(error_message)s, %(cooldown_seconds)s
                    )
                    """,
                    row,
                )
        conn.commit()
    return len(results)


def set_models_enabled_from_health(results: list[HealthResult]) -> int:
    """Mirror the latest scan verdict into the model on/off selection: unhealthy
    models are unchecked (leave selection and, via the agent sync, every agent's
    list); models that scanned healthy come back on — unless they were turned
    off by hand (manual_off), which only a manual re-enable undoes. Returns
    rows changed."""
    if not results:
        return 0
    with db_connect() as conn:
        with conn.cursor() as cur:
            changed = 0
            for result in results:
                enabled = result.status == "healthy"
                cur.execute(
                    """
                    UPDATE ai_router.models SET enabled = %s
                    WHERE public_id = %s AND enabled IS DISTINCT FROM %s
                      AND NOT (%s AND manual_off)
                    """,
                    (enabled, result.model_id, enabled, enabled),
                )
                changed += cur.rowcount
        conn.commit()
    return changed


def latest_health_by_model() -> dict[str, str]:
    # Runtime failures (rate limits, timeouts) expire after a cooldown: the model becomes
    # routable again and, if it fails once more, gets re-marked. Scanner verdicts persist.
    query = """
    SELECT DISTINCT ON (m.public_id) m.public_id, h.status
    FROM ai_router.provider_health h
    JOIN ai_router.models m ON m.model_id = h.model_id
    WHERE NOT (
        h.status = 'unhealthy'
        AND h.error_message LIKE 'runtime%%'
        AND h.checked_at < now() - make_interval(secs => coalesce(h.cooldown_seconds, 600))
    )
    ORDER BY m.public_id, h.checked_at DESC, h.health_id DESC
    """
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            return {row[0]: row[1] for row in cur.fetchall()}


def model_performance(days: int = 7) -> dict[str, dict[str, Any]]:
    """Per-model routing stats feeding dynamic_score: attempts and success rate
    from route_events, plus the latest health-scan latency."""
    query = """
    SELECT m.public_id,
           count(*) AS total,
           count(*) FILTER (WHERE r.status = 'success') AS ok,
           (
               SELECT h.latency_ms FROM ai_router.provider_health h
               WHERE h.model_id = m.model_id
               ORDER BY h.checked_at DESC, h.health_id DESC
               LIMIT 1
           ) AS latency_ms
    FROM ai_router.route_events r
    JOIN ai_router.models m ON m.model_id = r.selected_model_id
    WHERE r.created_at >= now() - make_interval(days => %(days)s)
    GROUP BY m.public_id, m.model_id
    """
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(query, {"days": days})
            rows = cur.fetchall()
    return {
        row[0]: {
            "total": row[1],
            "success_rate": (row[2] / row[1]) if row[1] else 1.0,
            "latency_ms": row[3],
        }
        for row in rows
    }


def runtime_degraded_models() -> set[str]:
    """Models knocked out by a recent runtime failure (rate limit, timeout) and
    still inside the cooldown window. Normal routing excludes them, but the
    auto-inclusion rule re-admits them as last-resort reserves when the healthy
    pool gets too small — a pool deteriorating under load must not stall callers.
    Hard failures (auth, model not found, scanner verdicts) are never re-admitted."""
    query = """
    SELECT public_id FROM (
        SELECT DISTINCT ON (m.public_id) m.public_id, h.status, h.error_message, h.checked_at, h.cooldown_seconds
        FROM ai_router.provider_health h
        JOIN ai_router.models m ON m.model_id = h.model_id
        ORDER BY m.public_id, h.checked_at DESC, h.health_id DESC
    ) latest
    WHERE status = 'unhealthy'
      AND error_message LIKE 'runtime%%'
      AND checked_at >= now() - make_interval(secs => coalesce(cooldown_seconds, 600))
    """
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            return {row[0] for row in cur.fetchall()}


def get_demand_routes() -> dict[str, list[str]]:
    query = """
    SELECT d.demand, m.public_id
    FROM ai_router.demand_routes d
    JOIN ai_router.models m ON m.model_id = d.model_id
    ORDER BY d.demand, d.position
    """
    routes: dict[str, list[str]] = {}
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            for demand, public_id in cur.fetchall():
                routes.setdefault(demand, []).append(public_id)
    return routes


def set_demand_routes(demand: str, public_ids: list[str]) -> None:
    """Replace the model chain for a demand. Empty list = back to the rank-derived default."""
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM ai_router.demand_routes WHERE demand = %s", (demand,))
            for position, public_id in enumerate(public_ids):
                cur.execute(
                    """
                    INSERT INTO ai_router.demand_routes (demand, position, model_id)
                    SELECT %s, %s, model_id FROM ai_router.models WHERE public_id = %s
                    ON CONFLICT DO NOTHING
                    """,
                    (demand, position, public_id),
                )
        conn.commit()


def route_event_to_row(
    request_id: str,
    selected_model_id: str | None,
    required_capability: str,
    status: str,
    error_type: str | None,
) -> dict[str, Any]:
    return {
        "request_id": request_id,
        "selected_model_id": selected_model_id,
        "required_capability": required_capability,
        "status": status,
        "error_type": error_type,
    }


def persist_route_event(
    request_id: str,
    selected_model_id: str | None,
    required_capability: str,
    status: str,
    error_type: str | None = None,
    usage: dict[str, Any] | None = None,
    agent_name: str | None = None,
    tokens_raw: int | None = None,
    tokens_compacted: int | None = None,
    demand: str | None = None,
    provider_model: str | None = None,
    prompt_preview: str | None = None,
    messages_dropped: int | None = None,
    context_window: int | None = None,
    context_budget: int | None = None,
    context_action: str = "none",
    context_candidates_skipped: int = 0,
) -> None:
    row = route_event_to_row(request_id, selected_model_id, required_capability, status, error_type)
    usage = usage or {}
    row["prompt_tokens"] = usage.get("prompt_tokens")
    row["completion_tokens"] = usage.get("completion_tokens")
    row["total_tokens"] = usage.get("total_tokens")
    row["cost"] = usage.get("cost") or 0
    row["agent_name"] = agent_name
    row["prompt_tokens_raw"] = tokens_raw
    row["prompt_tokens_compacted"] = tokens_compacted
    row["demand"] = demand
    row["prompt_preview"] = prompt_preview
    row["messages_dropped"] = messages_dropped or None
    row["context_window"] = context_window
    row["context_budget"] = context_budget
    row["context_action"] = context_action
    row["context_candidates_skipped"] = context_candidates_skipped
    row["reference_cost"] = None
    if not row["cost"] and selected_model_id and (row["prompt_tokens"] or row["completion_tokens"]):
        try:
            row["reference_cost"] = estimate_reference_cost(
                selected_model_id, provider_model or "", row["prompt_tokens"] or 0, row["completion_tokens"] or 0
            )
        except Exception:
            row["reference_cost"] = None
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO ai_router.route_events
                    (request_id, selected_model_id, required_capability, status, error_type,
                     prompt_tokens, completion_tokens, total_tokens, cost, agent_id,
                     prompt_tokens_raw, prompt_tokens_compacted, demand, reference_cost, prompt_preview,
                     messages_dropped, context_window, context_budget, context_action,
                     context_candidates_skipped)
                VALUES (
                    %(request_id)s,
                    (SELECT model_id FROM ai_router.models WHERE public_id = %(selected_model_id)s),
                    %(required_capability)s, %(status)s, %(error_type)s,
                    %(prompt_tokens)s, %(completion_tokens)s, %(total_tokens)s, %(cost)s,
                    (SELECT agent_id FROM ai_router.agents WHERE name = %(agent_name)s),
                    %(prompt_tokens_raw)s, %(prompt_tokens_compacted)s, %(demand)s, %(reference_cost)s, %(prompt_preview)s,
                    %(messages_dropped)s, %(context_window)s, %(context_budget)s, %(context_action)s,
                    %(context_candidates_skipped)s
                )
                """,
                row,
            )
        conn.commit()


def backfill_reference_costs() -> tuple[int, int]:
    """One-off/re-runnable pass: compute reference_cost for existing rows
    that predate the feature, or predate a model being added to the pricing
    catalog/overrides. Only touches rows where cost is 0/absent and
    reference_cost is still NULL, so re-running after adding new priced
    models only fills gaps. Returns (rows_checked, rows_priced)."""
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT r.route_id, m.public_id, m.provider_model, r.prompt_tokens, r.completion_tokens
                FROM ai_router.route_events r
                JOIN ai_router.models m ON m.model_id = r.selected_model_id
                WHERE r.status = 'success'
                  AND (r.cost = 0 OR r.cost IS NULL)
                  AND r.reference_cost IS NULL
                """
            )
            rows = cur.fetchall()

        priced = 0
        for route_id, public_id, provider_model, prompt_tokens, completion_tokens in rows:
            cost = estimate_reference_cost(public_id or "", provider_model or "", prompt_tokens or 0, completion_tokens or 0)
            if cost is None:
                continue
            with conn.cursor() as cur:
                cur.execute("UPDATE ai_router.route_events SET reference_cost = %s WHERE route_id = %s", (cost, route_id))
            priced += 1
        conn.commit()
    return len(rows), priced


def get_setting(key: str, default: str | None = None) -> str | None:
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT value FROM ai_router.settings WHERE key = %s", (key,))
            row = cur.fetchone()
    return row[0] if row else default


def set_setting(key: str, value: str) -> None:
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO ai_router.settings (key, value, updated_at) VALUES (%s, %s, now())
                ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = now()
                """,
                (key, value),
            )
        conn.commit()


def context_compaction_enabled() -> bool:
    return get_setting("context_compaction_enabled", "true") == "true"


def set_context_compaction_enabled(enabled: bool) -> None:
    set_setting("context_compaction_enabled", "true" if enabled else "false")


# Lossy — distinct from context_compaction above (whitespace-only, nothing ever
# removed). This actually drops the oldest turns once a request's estimated
# tokens cross the cap. Off by default: it changes what the model sees, so an
# operator must opt in per deployment rather than inherit a silent default.
def context_truncation_enabled() -> bool:
    return get_setting("context_truncation_enabled", "false") == "true"


def set_context_truncation_enabled(enabled: bool) -> None:
    set_setting("context_truncation_enabled", "true" if enabled else "false")


def response_cache_enabled() -> bool:
    return get_setting("response_cache_enabled", "false") == "true"


def set_response_cache_enabled(enabled: bool) -> None:
    set_setting("response_cache_enabled", "true" if enabled else "false")


def response_cache_ttl_seconds() -> int:
    try:
        return int(get_setting("response_cache_ttl_seconds", "300") or "300")
    except ValueError:
        return 300


def set_response_cache_ttl_seconds(value: int) -> None:
    set_setting("response_cache_ttl_seconds", str(value))


def context_truncation_max_tokens() -> int:
    """Fallback budget (flat tokens) used only when the selected model's real
    context window isn't in the pricing catalog — see
    context_truncation_trigger_percent for the normal, model-aware path."""
    try:
        return int(get_setting("context_truncation_max_tokens", "32000") or "32000")
    except ValueError:
        return 32000


def set_context_truncation_max_tokens(value: int) -> None:
    set_setting("context_truncation_max_tokens", str(value))


def context_truncation_trigger_percent() -> int:
    """Percent of the selected model's real context window (from the LiteLLM
    catalog) that must be filled before truncation kicks in — mirrors the
    0.8 threshold hermes-agent's own compression already uses, so both
    systems agree on "how full is too full" instead of ForgeRouter guessing
    a flat number that's wrong for a 8k model and wrong again for a 262k one."""
    try:
        return int(get_setting("context_truncation_trigger_percent", "80") or "80")
    except ValueError:
        return 80


def set_context_truncation_trigger_percent(value: int) -> None:
    set_setting("context_truncation_trigger_percent", str(value))


def latest_provider_health_rows() -> list[dict[str, Any]]:
    query = """
    SELECT DISTINCT ON (m.public_id)
        p.name AS provider,
        m.public_id AS model_id,
        p.tier,
        h.status,
        h.http_code,
        h.latency_ms,
        h.error_message,
        h.checked_at
    FROM ai_router.models m
    JOIN ai_router.providers p ON p.provider_id = m.provider_id
    LEFT JOIN ai_router.provider_health h ON h.model_id = m.model_id
    ORDER BY m.public_id, h.checked_at DESC NULLS LAST, h.health_id DESC NULLS LAST
    """
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            rows = cur.fetchall()
    return [
        {
            "provider": row[0],
            "model_id": row[1],
            "tier": row[2],
            "status": row[3] or "unknown",
            "http_code": row[4],
            "latency_ms": row[5],
            "error_message": row[6],
            "checked_at": row[7].isoformat() if row[7] else None,
        }
        for row in rows
    ]


def recent_route_events(limit: int = 25, agent_name: str | None = None) -> list[dict[str, Any]]:
    query = """
    SELECT
        r.route_id,
        r.request_id::text,
        m.public_id,
        r.required_capability,
        r.status,
        r.error_type,
        r.created_at,
        r.total_tokens,
        r.cost,
        a.name,
        r.demand,
        r.reference_cost,
        r.prompt_preview,
        r.messages_dropped,
        r.context_window,
        r.context_budget,
        r.context_action,
        r.context_candidates_skipped
    FROM ai_router.route_events r
    LEFT JOIN ai_router.models m ON m.model_id = r.selected_model_id
    LEFT JOIN ai_router.agents a ON a.agent_id = r.agent_id
    WHERE %(agent_name)s::text IS NULL OR a.name = %(agent_name)s
    ORDER BY r.route_id DESC
    LIMIT %(limit)s
    """
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(query, {"limit": limit, "agent_name": agent_name})
            rows = cur.fetchall()
    return [
        {
            "route_id": row[0],
            "request_id": row[1],
            "model_id": row[2],
            "required_capability": row[3],
            "status": row[4],
            "error_type": row[5],
            "created_at": row[6].isoformat() if row[6] else None,
            "total_tokens": row[7],
            "cost": float(row[8]) if row[8] is not None else 0.0,
            "agent": row[9],
            "demand": row[10],
            "reference_cost": float(row[11]) if row[11] is not None else None,
            "prompt_preview": row[12],
            "messages_dropped": row[13],
            "context_window": row[14],
            "context_budget": row[15],
            "context_action": row[16],
            "context_candidates_skipped": row[17],
        }
        for row in rows
    ]


def usage_summary(days: int = 30, agent_name: str | None = None) -> dict[str, Any]:
    # Usage is an agent-level view: events without an attributed agent (calls made
    # without a valid agent API key) are excluded everywhere — totals, daily chart,
    # cost by model and the per-agent breakdown ("(no agent)" must never appear).
    daily_query = """
    SELECT r.created_at::date AS day,
           count(*) AS messages,
           coalesce(sum(r.total_tokens), 0) AS tokens,
           coalesce(sum(r.cost), 0) AS cost,
           coalesce(sum(r.reference_cost), 0) AS reference_cost
    FROM ai_router.route_events r
    JOIN ai_router.agents a ON a.agent_id = r.agent_id
    WHERE r.created_at >= now() - make_interval(days => %(days)s)
      AND (%(agent_name)s::text IS NULL OR a.name = %(agent_name)s)
    GROUP BY 1
    ORDER BY 1
    """
    by_model_query = """
    SELECT m.public_id AS model_id,
           count(*) AS messages,
           coalesce(sum(r.total_tokens), 0) AS tokens,
           coalesce(sum(r.cost), 0) AS cost,
           coalesce(sum(r.reference_cost), 0) AS reference_cost
    FROM ai_router.route_events r
    JOIN ai_router.models m ON m.model_id = r.selected_model_id
    JOIN ai_router.agents a ON a.agent_id = r.agent_id
    WHERE r.created_at >= now() - make_interval(days => %(days)s)
      AND (%(agent_name)s::text IS NULL OR a.name = %(agent_name)s)
    GROUP BY 1
    ORDER BY 3 DESC, 2 DESC
    """
    by_demand_query = """
    SELECT r.demand,
           count(*) AS messages,
           coalesce(sum(r.total_tokens), 0) AS tokens,
           coalesce(sum(r.cost), 0) AS cost,
           coalesce(sum(r.reference_cost), 0) AS reference_cost
    FROM ai_router.route_events r
    JOIN ai_router.agents a ON a.agent_id = r.agent_id
    WHERE r.created_at >= now() - make_interval(days => %(days)s)
      AND (%(agent_name)s::text IS NULL OR a.name = %(agent_name)s)
      AND r.demand IS NOT NULL
    GROUP BY 1
    ORDER BY 3 DESC, 2 DESC
    """
    by_agent_daily_query = """
    SELECT a.name AS agent,
           r.created_at::date AS day,
           count(*) AS messages,
           coalesce(sum(r.total_tokens), 0) AS tokens,
           coalesce(sum(r.cost), 0) AS cost,
           coalesce(sum(r.reference_cost), 0) AS reference_cost
    FROM ai_router.route_events r
    JOIN ai_router.agents a ON a.agent_id = r.agent_id
    WHERE r.created_at >= now() - make_interval(days => %(days)s)
    GROUP BY 1, 2
    ORDER BY 1, 2
    """
    by_agent_model_query = """
    SELECT a.name AS agent,
           m.public_id AS model_id,
           count(*) AS messages,
           coalesce(sum(r.total_tokens), 0) AS tokens,
           coalesce(sum(r.cost), 0) AS cost,
           coalesce(sum(r.reference_cost), 0) AS reference_cost
    FROM ai_router.route_events r
    JOIN ai_router.models m ON m.model_id = r.selected_model_id
    JOIN ai_router.agents a ON a.agent_id = r.agent_id
    WHERE r.created_at >= now() - make_interval(days => %(days)s)
    GROUP BY 1, 2
    ORDER BY 1, 4 DESC, 3 DESC
    """
    compaction_query = """
    SELECT coalesce(sum(r.prompt_tokens_raw), 0) AS tokens_raw,
           coalesce(sum(r.prompt_tokens_raw - r.prompt_tokens_compacted), 0) AS tokens_saved
    FROM ai_router.route_events r
    JOIN ai_router.agents a ON a.agent_id = r.agent_id
    WHERE r.created_at >= now() - make_interval(days => %(days)s)
      AND (%(agent_name)s::text IS NULL OR a.name = %(agent_name)s)
      AND r.prompt_tokens_raw IS NOT NULL
      AND r.prompt_tokens_compacted IS NOT NULL
    """
    # Requests classified as "code" that had to be served off the general pool
    # because zero code-capable models were healthy at the time (app/main.py's
    # capability downgrade) — required_capability then reads something other
    # than "code". Counted separately from misclassification: this is a
    # capacity signal ("no code option was available"), not a routing mistake.
    code_downgrade_query = """
    SELECT count(*)
    FROM ai_router.route_events r
    JOIN ai_router.agents a ON a.agent_id = r.agent_id
    WHERE r.created_at >= now() - make_interval(days => %(days)s)
      AND (%(agent_name)s::text IS NULL OR a.name = %(agent_name)s)
      AND r.demand = 'code'
      AND r.required_capability IS DISTINCT FROM 'code'
    """
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(daily_query, {"days": days, "agent_name": agent_name})
            daily_rows = cur.fetchall()
            cur.execute(by_model_query, {"days": days, "agent_name": agent_name})
            model_rows = cur.fetchall()
            cur.execute(by_demand_query, {"days": days, "agent_name": agent_name})
            demand_rows = cur.fetchall()
            cur.execute(compaction_query, {"days": days, "agent_name": agent_name})
            compaction_row = cur.fetchone()
            cur.execute(code_downgrade_query, {"days": days, "agent_name": agent_name})
            code_downgrade_row = cur.fetchone()
            agent_daily_rows: list = []
            agent_model_rows: list = []
            if agent_name is None:
                cur.execute(by_agent_daily_query, {"days": days})
                agent_daily_rows = cur.fetchall()
                cur.execute(by_agent_model_query, {"days": days})
                agent_model_rows = cur.fetchall()
    daily = [
        {
            "day": row[0].isoformat(),
            "messages": row[1],
            "tokens": int(row[2]),
            "cost": float(row[3]),
            "reference_cost": float(row[4]),
        }
        for row in daily_rows
    ]
    total_tokens = sum(item["tokens"] for item in daily)
    by_model = [
        {
            "model_id": row[0],
            "messages": row[1],
            "tokens": int(row[2]),
            "cost": float(row[3]),
            "reference_cost": float(row[4]),
            "pct_total": round(100 * int(row[2]) / total_tokens, 1) if total_tokens else 0.0,
        }
        for row in model_rows
    ]
    by_demand = [
        {
            "demand": row[0],
            "messages": row[1],
            "tokens": int(row[2]),
            "cost": float(row[3]),
            "reference_cost": float(row[4]),
            "pct_total": round(100 * int(row[2]) / total_tokens, 1) if total_tokens else 0.0,
        }
        for row in demand_rows
    ]
    by_agent: dict[str, dict[str, Any]] = {}
    for agent, day, messages, tokens, cost, reference_cost in agent_daily_rows:
        entry = by_agent.setdefault(
            agent,
            {
                "agent": agent,
                "totals": {"messages": 0, "tokens": 0, "cost": 0.0, "reference_cost": 0.0},
                "daily": [],
                "by_model": [],
            },
        )
        entry["totals"]["messages"] += messages
        entry["totals"]["tokens"] += int(tokens)
        entry["totals"]["cost"] = round(entry["totals"]["cost"] + float(cost), 6)
        entry["totals"]["reference_cost"] = round(entry["totals"]["reference_cost"] + float(reference_cost), 6)
        entry["daily"].append(
            {
                "day": day.isoformat(),
                "messages": messages,
                "tokens": int(tokens),
                "cost": float(cost),
                "reference_cost": float(reference_cost),
            }
        )
    for agent, model_id, messages, tokens, cost, reference_cost in agent_model_rows:
        entry = by_agent.get(agent)
        if not entry:
            continue
        agent_tokens = entry["totals"]["tokens"]
        entry["by_model"].append(
            {
                "model_id": model_id,
                "messages": messages,
                "tokens": int(tokens),
                "cost": float(cost),
                "reference_cost": float(reference_cost),
                "pct_total": round(100 * int(tokens) / agent_tokens, 1) if agent_tokens else 0.0,
            }
        )
    tokens_raw = int(compaction_row[0]) if compaction_row else 0
    tokens_saved = int(compaction_row[1]) if compaction_row else 0
    code_downgrades = int(code_downgrade_row[0]) if code_downgrade_row else 0
    return {
        "days": days,
        "totals": {
            "messages": sum(item["messages"] for item in daily),
            "tokens": total_tokens,
            "cost": round(sum(item["cost"] for item in daily), 6),
            "reference_cost": round(sum(item["reference_cost"] for item in daily), 6),
            "tokens_raw": tokens_raw,
            "tokens_saved": tokens_saved,
            "pct_saved": round(100 * tokens_saved / tokens_raw, 1) if tokens_raw else 0.0,
            "code_downgrades": code_downgrades,
        },
        "daily": daily,
        "by_model": by_model,
        "by_demand": by_demand,
        "by_agent": list(by_agent.values()),
    }


def yearly_usage_by_agent(year: int | None = None) -> dict[str, Any]:
    # Combines persisted monthly rollups (ai_router.usage_monthly, written by
    # archive_old_route_events) with live route_events for months that haven't
    # been archived yet, so this survives deletion of old raw rows. Mirrors
    # usage_summary's "(no agent)" exclusion.
    resolved_year = year or datetime.now(timezone.utc).year
    query = """
    WITH archived AS (
        SELECT u.agent_id, u.month, u.messages, u.tokens, u.cost, u.reference_cost
        FROM ai_router.usage_monthly u
        WHERE u.year = %(year)s
    ),
    live AS (
        SELECT r.agent_id,
               extract(month FROM r.created_at)::int AS month,
               count(*) AS messages,
               coalesce(sum(r.total_tokens), 0) AS tokens,
               coalesce(sum(r.cost), 0) AS cost,
               coalesce(sum(r.reference_cost), 0) AS reference_cost
        FROM ai_router.route_events r
        WHERE r.agent_id IS NOT NULL
          AND extract(year FROM r.created_at)::int = %(year)s
        GROUP BY 1, 2
    ),
    combined AS (
        SELECT * FROM archived
        UNION ALL
        SELECT * FROM live
    )
    SELECT a.name AS agent,
           c.month,
           sum(c.messages) AS messages,
           sum(c.tokens) AS tokens,
           sum(c.cost) AS cost,
           sum(c.reference_cost) AS reference_cost
    FROM combined c
    JOIN ai_router.agents a ON a.agent_id = c.agent_id
    GROUP BY 1, 2
    ORDER BY 1, 2
    """
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(query, {"year": resolved_year})
            rows = cur.fetchall()

    by_agent: dict[str, dict[str, Any]] = {}
    for agent, month, messages, tokens, cost, reference_cost in rows:
        entry = by_agent.setdefault(
            agent,
            {"agent": agent, "months": {}, "totals": {"messages": 0, "tokens": 0, "cost": 0.0, "reference_cost": 0.0}},
        )
        entry["months"][int(month)] = {
            "messages": messages,
            "tokens": int(tokens),
            "cost": float(cost),
            "reference_cost": float(reference_cost),
        }
        entry["totals"]["messages"] += messages
        entry["totals"]["tokens"] += int(tokens)
        entry["totals"]["cost"] = round(entry["totals"]["cost"] + float(cost), 6)
        entry["totals"]["reference_cost"] = round(entry["totals"]["reference_cost"] + float(reference_cost), 6)

    return {
        "year": resolved_year,
        "by_agent": sorted(by_agent.values(), key=lambda item: item["agent"]),
    }


def yearly_usage_by_demand(year: int | None = None) -> dict[str, Any]:
    # Mirrors yearly_usage_by_agent, keyed by demand instead of agent: combines
    # ai_router.usage_monthly_demand (written by archive_old_route_events) with
    # live route_events for months not yet archived. Requests without a demand
    # (a concrete model id was requested) are excluded — demand-routing-only.
    # Caveat: usage_monthly_demand only exists from db/035 onward, so months
    # archived (and their raw rows deleted) before that migration have no
    # recoverable demand breakdown and will read 0 here even if the per-agent
    # totals for the same month are non-zero.
    resolved_year = year or datetime.now(timezone.utc).year
    query = """
    WITH archived AS (
        SELECT u.demand, u.month, u.messages, u.tokens, u.cost, u.reference_cost
        FROM ai_router.usage_monthly_demand u
        WHERE u.year = %(year)s
    ),
    live AS (
        SELECT r.demand,
               extract(month FROM r.created_at)::int AS month,
               count(*) AS messages,
               coalesce(sum(r.total_tokens), 0) AS tokens,
               coalesce(sum(r.cost), 0) AS cost,
               coalesce(sum(r.reference_cost), 0) AS reference_cost
        FROM ai_router.route_events r
        WHERE r.demand IS NOT NULL
          AND extract(year FROM r.created_at)::int = %(year)s
        GROUP BY 1, 2
    ),
    combined AS (
        SELECT * FROM archived
        UNION ALL
        SELECT * FROM live
    )
    SELECT demand,
           month,
           sum(messages) AS messages,
           sum(tokens) AS tokens,
           sum(cost) AS cost,
           sum(reference_cost) AS reference_cost
    FROM combined
    GROUP BY 1, 2
    ORDER BY 1, 2
    """
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(query, {"year": resolved_year})
            rows = cur.fetchall()

    by_demand: dict[str, dict[str, Any]] = {}
    for demand, month, messages, tokens, cost, reference_cost in rows:
        entry = by_demand.setdefault(
            demand,
            {"demand": demand, "months": {}, "totals": {"messages": 0, "tokens": 0, "cost": 0.0, "reference_cost": 0.0}},
        )
        entry["months"][int(month)] = {
            "messages": messages,
            "tokens": int(tokens),
            "cost": float(cost),
            "reference_cost": float(reference_cost),
        }
        entry["totals"]["messages"] += messages
        entry["totals"]["tokens"] += int(tokens)
        entry["totals"]["cost"] = round(entry["totals"]["cost"] + float(cost), 6)
        entry["totals"]["reference_cost"] = round(entry["totals"]["reference_cost"] + float(reference_cost), 6)

    return {
        "year": resolved_year,
        "by_demand": sorted(by_demand.values(), key=lambda item: item["demand"]),
    }


def archive_old_route_events(keep_months: int = 2) -> dict[str, Any]:
    # Keeps the most recent `keep_months` months of raw route_events for the
    # Messages page; older rows are rolled up per agent/year/month into
    # ai_router.usage_monthly (and per demand/year/month into
    # ai_router.usage_monthly_demand) and then deleted, bounding database/disk growth.
    aggregate_query = """
    SELECT r.agent_id,
           extract(year FROM r.created_at)::int AS year,
           extract(month FROM r.created_at)::int AS month,
           count(*) AS messages,
           coalesce(sum(r.total_tokens), 0) AS tokens,
           coalesce(sum(r.cost), 0) AS cost,
           coalesce(sum(r.reference_cost), 0) AS reference_cost
    FROM ai_router.route_events r
    WHERE r.agent_id IS NOT NULL
      AND r.created_at < %(cutoff)s
    GROUP BY 1, 2, 3
    """
    upsert_query = """
    INSERT INTO ai_router.usage_monthly (agent_id, year, month, messages, tokens, cost, reference_cost)
    VALUES (%(agent_id)s, %(year)s, %(month)s, %(messages)s, %(tokens)s, %(cost)s, %(reference_cost)s)
    ON CONFLICT (agent_id, year, month) DO UPDATE SET
        messages = ai_router.usage_monthly.messages + EXCLUDED.messages,
        tokens = ai_router.usage_monthly.tokens + EXCLUDED.tokens,
        cost = ai_router.usage_monthly.cost + EXCLUDED.cost,
        reference_cost = ai_router.usage_monthly.reference_cost + EXCLUDED.reference_cost,
        archived_at = now()
    """
    # Requests without a demand (a concrete model id was requested, not forgerouter/
    # auto or forgerouter/<demand>) are excluded — this rollup is demand-routing-only.
    demand_aggregate_query = """
    SELECT r.demand,
           extract(year FROM r.created_at)::int AS year,
           extract(month FROM r.created_at)::int AS month,
           count(*) AS messages,
           coalesce(sum(r.total_tokens), 0) AS tokens,
           coalesce(sum(r.cost), 0) AS cost,
           coalesce(sum(r.reference_cost), 0) AS reference_cost
    FROM ai_router.route_events r
    WHERE r.demand IS NOT NULL
      AND r.created_at < %(cutoff)s
    GROUP BY 1, 2, 3
    """
    demand_upsert_query = """
    INSERT INTO ai_router.usage_monthly_demand (demand, year, month, messages, tokens, cost, reference_cost)
    VALUES (%(demand)s, %(year)s, %(month)s, %(messages)s, %(tokens)s, %(cost)s, %(reference_cost)s)
    ON CONFLICT (demand, year, month) DO UPDATE SET
        messages = ai_router.usage_monthly_demand.messages + EXCLUDED.messages,
        tokens = ai_router.usage_monthly_demand.tokens + EXCLUDED.tokens,
        cost = ai_router.usage_monthly_demand.cost + EXCLUDED.cost,
        reference_cost = ai_router.usage_monthly_demand.reference_cost + EXCLUDED.reference_cost,
        archived_at = now()
    """
    delete_query = "DELETE FROM ai_router.route_events WHERE created_at < %(cutoff)s"

    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT date_trunc('month', now()) - make_interval(months => %(offset)s)",
                {"offset": max(keep_months - 1, 0)},
            )
            cutoff = cur.fetchone()[0]
            cur.execute(aggregate_query, {"cutoff": cutoff})
            rows = cur.fetchall()
            for agent_id, year, month, messages, tokens, cost, reference_cost in rows:
                cur.execute(
                    upsert_query,
                    {
                        "agent_id": agent_id,
                        "year": year,
                        "month": month,
                        "messages": messages,
                        "tokens": int(tokens),
                        "cost": float(cost),
                        "reference_cost": float(reference_cost),
                    },
                )
            cur.execute(demand_aggregate_query, {"cutoff": cutoff})
            demand_rows = cur.fetchall()
            for demand, year, month, messages, tokens, cost, reference_cost in demand_rows:
                cur.execute(
                    demand_upsert_query,
                    {
                        "demand": demand,
                        "year": year,
                        "month": month,
                        "messages": messages,
                        "tokens": int(tokens),
                        "cost": float(cost),
                        "reference_cost": float(reference_cost),
                    },
                )
            cur.execute(delete_query, {"cutoff": cutoff})
            deleted_rows = cur.rowcount
        conn.commit()

    return {
        "cutoff": cutoff.isoformat(),
        "archived_months": len(rows),
        "deleted_rows": deleted_rows,
    }


def db_providers_with_models() -> list[dict[str, Any]]:
    query = """
    SELECT
        p.name, p.tier, p.base_url, p.api_key_env, p.enabled, p.api_key,
        p.access_type, p.cost_type, p.auth_config, p.api_format,
        m.public_id, m.provider_model, m.capabilities, m.enabled, m.healthy, m.manual_off
    FROM ai_router.providers p
    LEFT JOIN ai_router.models m ON m.provider_id = p.provider_id
    ORDER BY p.tier, p.name, m.public_id
    """
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            rows = cur.fetchall()
    providers: dict[str, dict[str, Any]] = {}
    for row in rows:
        provider = providers.setdefault(
            row[0],
            {
                "name": row[0],
                "tier": row[1],
                "base_url": row[2],
                "api_key_env": row[3] or "",
                "enabled": row[4],
                "api_key": row[5] or "",
                "access_type": row[6] or "api_key",
                "cost_type": row[7] or "free",
                "auth_config": row[8] or {},
                "api_format": row[9] or "openai",
                "models": [],
            },
        )
        if row[10] is not None:
            provider["models"].append(
                {
                    "id": row[10],
                    "provider_model": row[11],
                    "capabilities": list(row[12] or []),
                    "enabled": row[13],
                    "healthy": row[14],
                    "manual_off": bool(row[15]),
                }
            )
    return list(providers.values())


def upsert_provider(payload: dict[str, Any], manual: bool = False) -> None:
    """Create or replace a provider and its full model list.

    manual=True (dashboard save): the enabled flags are the user's curation —
    an unchecked model becomes manual_off (permanent until re-checked by hand).
    manual=False (resync/automation): manual_off is preserved and keeps the
    model off regardless of the scan verdict."""
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO ai_router.providers (name, tier, base_url, api_key_env, enabled, api_key, access_type, cost_type, auth_config, api_format)
                VALUES (%(name)s, %(tier)s, %(base_url)s, %(api_key_env)s, %(enabled)s, %(api_key)s, %(access_type)s, %(cost_type)s, %(auth_config)s, %(api_format)s)
                ON CONFLICT (name) DO UPDATE SET
                    tier = EXCLUDED.tier,
                    base_url = EXCLUDED.base_url,
                    api_key_env = EXCLUDED.api_key_env,
                    enabled = EXCLUDED.enabled,
                    -- empty api_key in the payload means "keep the stored key"
                    api_key = CASE WHEN EXCLUDED.api_key = '' THEN ai_router.providers.api_key ELSE EXCLUDED.api_key END,
                    access_type = EXCLUDED.access_type,
                    cost_type = EXCLUDED.cost_type,
                    auth_config = EXCLUDED.auth_config,
                    api_format = EXCLUDED.api_format
                RETURNING provider_id
                """,
                {
                    "name": payload["name"],
                    "tier": int(payload["tier"]),
                    "base_url": payload["base_url"],
                    "api_key_env": payload.get("api_key_env") or "",
                    "enabled": bool(payload.get("enabled", True)),
                    "api_key": payload.get("api_key") or "",
                    "access_type": payload.get("access_type") or "api_key",
                    "cost_type": payload.get("cost_type") or "free",
                    "auth_config": Json(payload.get("auth_config") or {}),
                    "api_format": payload.get("api_format") or "openai",
                },
            )
            provider_id = cur.fetchone()[0]
            kept_ids = [model["id"] for model in payload.get("models", [])]
            cur.execute(
                "DELETE FROM ai_router.models WHERE provider_id = %s AND NOT (public_id = ANY(%s))",
                (provider_id, kept_ids),
            )
            for model in payload.get("models", []):
                enabled = bool(model.get("enabled", True))
                cur.execute(
                    """
                    INSERT INTO ai_router.models (provider_id, public_id, provider_model, capabilities, enabled, healthy, manual_off)
                    VALUES (%(provider_id)s, %(public_id)s, %(provider_model)s, %(capabilities)s, %(enabled)s, false, %(manual_off)s)
                    ON CONFLICT (public_id) DO UPDATE SET
                        provider_id = EXCLUDED.provider_id,
                        provider_model = EXCLUDED.provider_model,
                        capabilities = EXCLUDED.capabilities,
                        enabled = CASE WHEN %(manual)s THEN EXCLUDED.enabled
                                       ELSE EXCLUDED.enabled AND NOT ai_router.models.manual_off END,
                        manual_off = CASE WHEN %(manual)s THEN NOT EXCLUDED.enabled
                                          ELSE ai_router.models.manual_off END
                    """,
                    {
                        "provider_id": provider_id,
                        "public_id": model["id"],
                        "provider_model": model.get("provider_model") or model["id"],
                        "capabilities": list(model.get("capabilities", ["text"])),
                        "enabled": enabled,
                        "manual": manual,
                        "manual_off": manual and not enabled,
                    },
                )
                health = model.get("health")
                if health and health.get("status"):
                    cur.execute(
                        """
                        INSERT INTO ai_router.provider_health
                            (model_id, status, http_code, latency_ms, error_message)
                        VALUES (
                            (SELECT model_id FROM ai_router.models WHERE public_id = %(public_id)s),
                            %(status)s, %(http_code)s, %(latency_ms)s, %(error)s
                        )
                        """,
                        {
                            "public_id": model["id"],
                            "status": health["status"],
                            "http_code": health.get("http_code"),
                            "latency_ms": health.get("latency_ms"),
                            "error": health.get("error"),
                        },
                    )
        conn.commit()


def delete_provider(name: str) -> bool:
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM ai_router.providers WHERE name = %s", (name,))
            deleted = cur.rowcount > 0
        conn.commit()
    return deleted


def list_agents_with_usage(days: int = 30) -> list[dict[str, Any]]:
    agents_query = """
    SELECT name, api_key, enabled, created_at, description, aux_tasks, budget_limit_usd, budget_action,
           config_path, config_format, config_key, restart_service, kind, avatar_data_url
    FROM ai_router.agents
    ORDER BY created_at, name
    """
    month_spend_query = """
    SELECT a.name, coalesce(sum(r.cost), 0) + coalesce(sum(r.reference_cost), 0) AS month_spend
    FROM ai_router.route_events r
    JOIN ai_router.agents a ON a.agent_id = r.agent_id
    WHERE date_trunc('month', r.created_at) = date_trunc('month', now())
    GROUP BY 1
    """
    usage_query = """
    SELECT a.name,
           r.created_at::date AS day,
           count(*) AS messages,
           coalesce(sum(r.total_tokens), 0) AS tokens,
           coalesce(sum(r.cost), 0) AS cost,
           coalesce(sum(r.reference_cost), 0) AS reference_cost
    FROM ai_router.route_events r
    JOIN ai_router.agents a ON a.agent_id = r.agent_id
    WHERE r.created_at >= now() - make_interval(days => %(days)s)
    GROUP BY 1, 2
    ORDER BY 1, 2
    """
    models_query = """
    SELECT a.name, m.public_id, am.enabled
    FROM ai_router.agent_models am
    JOIN ai_router.agents a ON a.agent_id = am.agent_id
    JOIN ai_router.models m ON m.model_id = am.model_id
    ORDER BY 1, 2
    """
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(agents_query)
            agent_rows = cur.fetchall()
            cur.execute(usage_query, {"days": days})
            usage_rows = cur.fetchall()
            cur.execute(models_query)
            model_rows = cur.fetchall()
            cur.execute(month_spend_query)
            month_spend_rows = cur.fetchall()
    agents = {
        row[0]: {
            "name": row[0],
            "api_key": row[1],
            "enabled": row[2],
            "created_at": row[3].isoformat() if row[3] else None,
            "description": row[4] or "",
            "aux_tasks": bool(row[5]),
            "budget_limit_usd": float(row[6]) if row[6] is not None else None,
            "budget_action": row[7] or "alert",
            "config_path": row[8] or "",
            "config_format": row[9] or "",
            "config_key": row[10] or "",
            "restart_service": row[11] or "",
            "kind": row[12] or "agent",
            "avatar_data_url": row[13],
            "month_spend": 0.0,
            "messages": 0,
            "tokens": 0,
            "cost": 0.0,
            "reference_cost": 0.0,
            "daily": [],
            "models": [],
            "models_off": [],
        }
        for row in agent_rows
    }
    for name, month_spend in month_spend_rows:
        if name in agents:
            agents[name]["month_spend"] = float(month_spend)
    for name, public_id, model_enabled in model_rows:
        if name in agents:
            agents[name]["models" if model_enabled else "models_off"].append(public_id)
    for name, day, messages, tokens, cost, reference_cost in usage_rows:
        agent = agents.get(name)
        if not agent:
            continue
        agent["messages"] += messages
        agent["tokens"] += int(tokens)
        agent["cost"] = round(agent["cost"] + float(cost), 6)
        agent["reference_cost"] = round(agent["reference_cost"] + float(reference_cost), 6)
        agent["daily"].append(
            {
                "day": day.isoformat(),
                "messages": messages,
                "tokens": int(tokens),
                "cost": float(cost),
                "reference_cost": float(reference_cost),
            }
        )
    return list(agents.values())


def create_agent(
    name: str,
    api_key: str,
    description: str = "",
    kind: str = "agent",
    avatar_data_url: str | None = None,
) -> None:
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO ai_router.agents (name, api_key, description, kind, avatar_data_url) VALUES (%s, %s, %s, %s, %s)",
                (name, api_key, description, kind, avatar_data_url),
            )
        conn.commit()


def set_agent_kind(name: str, kind: str) -> bool:
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute("UPDATE ai_router.agents SET kind = %s WHERE name = %s", (kind, name))
            updated = cur.rowcount > 0
        conn.commit()
    return updated


def set_aux_tasks_agent(name: str) -> bool:
    """Make the agent the auxiliary-tasks agent. The role is exclusive: only one
    agent may hold it, so it is cleared from everyone else first."""
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM ai_router.agents WHERE name = %s", (name,))
            if not cur.fetchone():
                return False
            cur.execute("UPDATE ai_router.agents SET aux_tasks = FALSE WHERE aux_tasks")
            cur.execute("UPDATE ai_router.agents SET aux_tasks = TRUE WHERE name = %s", (name,))
        conn.commit()
    return True


def set_agent_description(name: str, description: str) -> bool:
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute("UPDATE ai_router.agents SET description = %s WHERE name = %s", (description, name))
            updated = cur.rowcount > 0
        conn.commit()
    return updated


def set_agent_avatar(name: str, avatar_data_url: str | None) -> bool:
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute("UPDATE ai_router.agents SET avatar_data_url = %s WHERE name = %s", (avatar_data_url, name))
            updated = cur.rowcount > 0
        conn.commit()
    return updated


def rename_agent(name: str, new_name: str) -> bool:
    """Rename an agent. Safe to do at any time: route_events/agent_models/agent_providers
    reference the agent by its stable agent_id, never by name."""
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute("UPDATE ai_router.agents SET name = %s WHERE name = %s", (new_name, name))
            updated = cur.rowcount > 0
        conn.commit()
    return updated


def rotate_agent_key(name: str, api_key: str) -> bool:
    """Replace only the API key; the agent identity and its model controls are preserved."""
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute("UPDATE ai_router.agents SET api_key = %s WHERE name = %s", (api_key, name))
            updated = cur.rowcount > 0
        conn.commit()
    return updated


def duplicate_agent(source_name: str, new_name: str, api_key: str) -> bool:
    """Create a new agent (own key) copying the source agent's model associations."""
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT agent_id, description, kind FROM ai_router.agents WHERE name = %s", (source_name,))
            row = cur.fetchone()
            if not row:
                return False
            source_id = row[0]
            cur.execute(
                "INSERT INTO ai_router.agents (name, api_key, description, kind) VALUES (%s, %s, %s, %s) RETURNING agent_id",
                (new_name, api_key, row[1] or "", row[2] or "agent"),
            )
            new_id = cur.fetchone()[0]
            cur.execute(
                """
                INSERT INTO ai_router.agent_models (agent_id, model_id, enabled)
                SELECT %s, model_id, enabled FROM ai_router.agent_models WHERE agent_id = %s
                """,
                (new_id, source_id),
            )
        conn.commit()
    return True


def get_agent_models(name: str) -> list[str]:
    query = """
    SELECT m.public_id
    FROM ai_router.agent_models am
    JOIN ai_router.agents a ON a.agent_id = am.agent_id
    JOIN ai_router.models m ON m.model_id = am.model_id
    WHERE a.name = %s AND am.enabled
    ORDER BY m.public_id
    """
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (name,))
            return [row[0] for row in cur.fetchall()]


def set_agent_models(name: str, public_ids: list[str]) -> bool:
    """Set which models are enabled for the agent. Ids in the list are enabled
    (and associated when new); existing associations missing from the list are
    kept but disabled — the per-agent opt-out survives the automatic syncs run
    by Refresh / Run scan / provider saves."""
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT agent_id FROM ai_router.agents WHERE name = %s", (name,))
            row = cur.fetchone()
            if not row:
                return False
            agent_id = row[0]
            cur.execute("UPDATE ai_router.agent_models SET enabled = FALSE WHERE agent_id = %s", (agent_id,))
            for public_id in public_ids:
                cur.execute(
                    """
                    INSERT INTO ai_router.agent_models (agent_id, model_id, enabled)
                    SELECT %s, model_id, TRUE FROM ai_router.models WHERE public_id = %s
                    ON CONFLICT (agent_id, model_id) DO UPDATE SET enabled = TRUE
                    """,
                    (agent_id, public_id),
                )
        conn.commit()
    return True


def sync_agent_model_associations() -> dict[str, int]:
    """Reconcile every agent's model list with the provider registry, so Refresh /
    Run scan / provider saves keep all agents current without manual work:
    - every provider is associated to every agent (agent_providers is the durable
      participation record), so new providers and new agents meet automatically;
    - models switched off on the provider ("on" unchecked) leave every agent's list;
    - enabled models are associated to every agent automatically (new ones arrive
      enabled; per-agent opt-outs — rows kept with enabled = FALSE — are not touched)."""
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO ai_router.agent_providers (agent_id, provider_id)
                SELECT a.agent_id, p.provider_id
                FROM ai_router.agents a
                CROSS JOIN ai_router.providers p
                ON CONFLICT (agent_id, provider_id) DO NOTHING
                """
            )
            cur.execute(
                """
                DELETE FROM ai_router.agent_models am
                USING ai_router.models m
                WHERE am.model_id = m.model_id AND NOT m.enabled
                """
            )
            removed = cur.rowcount
            cur.execute(
                """
                INSERT INTO ai_router.agent_models (agent_id, model_id, enabled)
                SELECT ap.agent_id, m.model_id, TRUE
                FROM ai_router.agent_providers ap
                JOIN ai_router.models m ON m.provider_id = ap.provider_id AND m.enabled
                ON CONFLICT (agent_id, model_id) DO NOTHING
                """
            )
            added = cur.rowcount
        conn.commit()
    return {"associations_added": added, "associations_removed": removed}


def associate_all_healthy_models_to_all_agents() -> dict[str, int]:
    """Associate and enable every healthy, online (enabled) model across all
    registered agents, ensuring all agents have full access to all healthy LLMs."""
    with db_connect() as conn:
        with conn.cursor() as cur:
            # 1. Ensure all agents participate with all enabled providers
            cur.execute(
                """
                INSERT INTO ai_router.agent_providers (agent_id, provider_id)
                SELECT a.agent_id, p.provider_id
                FROM ai_router.agents a
                CROSS JOIN ai_router.providers p
                WHERE a.enabled AND p.enabled
                ON CONFLICT (agent_id, provider_id) DO NOTHING
                """
            )
            # 2. Link every healthy enabled model to every agent (setting enabled = TRUE)
            cur.execute(
                """
                WITH latest_health AS (
                    SELECT DISTINCT ON (model_id) model_id, status
                    FROM ai_router.provider_health
                    ORDER BY model_id, checked_at DESC
                )
                INSERT INTO ai_router.agent_models (agent_id, model_id, enabled)
                SELECT a.agent_id, m.model_id, TRUE
                FROM ai_router.agents a
                CROSS JOIN ai_router.models m
                JOIN ai_router.providers p ON p.provider_id = m.provider_id
                LEFT JOIN latest_health lh ON lh.model_id = m.model_id
                WHERE a.enabled AND m.enabled AND p.enabled
                  AND (lh.status = 'healthy' OR lh.status IS NULL)
                ON CONFLICT (agent_id, model_id) DO UPDATE SET enabled = TRUE
                """
            )
            associations_updated = cur.rowcount
            cur.execute("SELECT COUNT(*) FROM ai_router.agents WHERE enabled")
            agent_count = cur.fetchone()[0] if cur.rowcount else 0
            cur.execute(
                """
                WITH latest_health AS (
                    SELECT DISTINCT ON (model_id) model_id, status
                    FROM ai_router.provider_health
                    ORDER BY model_id, checked_at DESC
                )
                SELECT COUNT(DISTINCT m.model_id)
                FROM ai_router.models m
                JOIN ai_router.providers p ON p.provider_id = m.provider_id
                LEFT JOIN latest_health lh ON lh.model_id = m.model_id
                WHERE m.enabled AND p.enabled AND (lh.status = 'healthy' OR lh.status IS NULL)
                """
            )
            model_count = cur.fetchone()[0] if cur.rowcount else 0
        conn.commit()
    return {
        "agents_updated": agent_count,
        "models_count": model_count,
        "associations_updated": associations_updated,
    }


def agent_allowed_models(name: str) -> set[str] | None:
    """Models the agent may use. An empty set means the agent has no providers
    associated yet and routes to nothing — providers must be registered under
    the agent (or models added on the Agents page) before it can route."""
    return set(get_agent_models(name))


def delete_agent(name: str) -> bool:
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM ai_router.agents WHERE name = %s", (name,))
            deleted = cur.rowcount > 0
        conn.commit()
    return deleted


def get_agent_api_key(name: str) -> str | None:
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT api_key FROM ai_router.agents WHERE name = %s", (name,))
            row = cur.fetchone()
    return row[0] if row else None


def has_any_agent() -> bool:
    """True when at least one enabled agent exists — that is when admin
    endpoints switch from open (first-time setup) to key/session protected."""
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM ai_router.agents WHERE enabled LIMIT 1")
            return cur.fetchone() is not None


def find_agent_by_key(api_key: str) -> str | None:
    if not api_key:
        return None
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT name FROM ai_router.agents WHERE api_key = %s AND enabled",
                (api_key,),
            )
            row = cur.fetchone()
    return row[0] if row else None


def get_agent_budget(agent_name: str) -> tuple[float | None, str] | None:
    """(limit_usd, action) for an agent, or None if the agent doesn't exist.
    limit_usd is None when no budget is configured (the default — opt-in)."""
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT budget_limit_usd, budget_action FROM ai_router.agents WHERE name = %s",
                (agent_name,),
            )
            row = cur.fetchone()
    if not row:
        return None
    limit_usd = float(row[0]) if row[0] is not None else None
    return limit_usd, row[1] or "alert"


def set_agent_budget(agent_name: str, limit_usd: float | None, action: str) -> bool:
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE ai_router.agents SET budget_limit_usd = %s, budget_action = %s WHERE name = %s",
                (limit_usd, action, agent_name),
            )
            updated = cur.rowcount > 0
        conn.commit()
    return updated


def get_agent_deploy_config(agent_name: str) -> dict | None:
    """Where an agent's own runtime config lives, so rotate-key can write the
    freshly rotated key there directly. None fields (the default) mean
    rotate-key stays DB-only for that agent, exactly like before this existed."""
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT config_path, config_format, config_key, restart_service FROM ai_router.agents WHERE name = %s",
                (agent_name,),
            )
            row = cur.fetchone()
    if not row:
        return None
    return {
        "config_path": row[0],
        "config_format": row[1],
        "config_key": row[2],
        "restart_service": row[3],
    }


def set_agent_deploy_config(
    agent_name: str,
    config_path: str | None,
    config_format: str | None,
    config_key: str | None,
    restart_service: str | None,
) -> bool:
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE ai_router.agents
                SET config_path = %s, config_format = %s, config_key = %s, restart_service = %s
                WHERE name = %s
                """,
                (config_path, config_format, config_key, restart_service, agent_name),
            )
            updated = cur.rowcount > 0
        conn.commit()
    return updated


def agent_month_spend(agent_name: str) -> float:
    """Real cost + reference cost for this agent so far in the current
    calendar month. Real cost is almost always 0 (free-tier-only router) —
    reference_cost is what actually drives a budget guard here."""
    query = """
    SELECT coalesce(sum(r.cost), 0) + coalesce(sum(r.reference_cost), 0)
    FROM ai_router.route_events r
    JOIN ai_router.agents a ON a.agent_id = r.agent_id
    WHERE a.name = %(agent_name)s
      AND date_trunc('month', r.created_at) = date_trunc('month', now())
    """
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(query, {"agent_name": agent_name})
            row = cur.fetchone()
    return float(row[0]) if row and row[0] is not None else 0.0


_PBKDF2_ITERATIONS = 200_000


def hash_password(password: str, salt: str | None = None) -> str:
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), _PBKDF2_ITERATIONS).hex()
    return f"pbkdf2${_PBKDF2_ITERATIONS}${salt}${digest}"


def verify_password(stored: str, password: str) -> bool:
    try:
        _, iterations, salt, digest = stored.split("$")
        candidate = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), int(iterations)).hex()
        return secrets.compare_digest(candidate, digest)
    except (ValueError, TypeError):
        return False


def ensure_default_user() -> None:
    """Seed the default admin/admin login (forced password change) when no user exists yet."""
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM ai_router.users")
            if cur.fetchone()[0] == 0:
                cur.execute(
                    "INSERT INTO ai_router.users (username, password_hash, must_change_password, is_admin) VALUES (%s, %s, true, true)",
                    ("admin", hash_password("admin")),
                )
        conn.commit()


def authenticate_user(username: str, password: str) -> dict[str, Any] | None:
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT user_id, username, password_hash, must_change_password, is_admin, full_name, profile_id, avatar_data_url, email"
                " FROM ai_router.users WHERE username = %s AND is_active",
                (username,),
            )
            row = cur.fetchone()
    if not row or not verify_password(row[2], password):
        return None
    return {
        "user_id": row[0],
        "username": row[1],
        "must_change_password": row[3],
        "is_admin": row[4],
        "full_name": row[5],
        "profile_id": row[6],
        "avatar_data_url": row[7],
        "email": row[8],
    }


def first_user() -> dict[str, Any] | None:
    """The dashboard's primary (oldest) user — the account trusted SSO logs
    into. Single-admin model: the seeded admin, possibly renamed since."""
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT user_id, username, must_change_password FROM ai_router.users ORDER BY user_id LIMIT 1"
            )
            row = cur.fetchone()
    if not row:
        return None
    return {"user_id": row[0], "username": row[1], "must_change_password": row[2]}


def create_session(user_id: int, days: int = 7) -> str:
    token = secrets.token_urlsafe(32)
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM ai_router.sessions WHERE expires_at < now()")
            cur.execute(
                "INSERT INTO ai_router.sessions (token, user_id, expires_at) VALUES (%s, %s, now() + make_interval(days => %s))",
                (token, user_id, days),
            )
        conn.commit()
    return token


def session_user(token: str) -> dict[str, Any] | None:
    if not token:
        return None
    query = """
    SELECT u.user_id, u.username, u.must_change_password, u.is_admin, u.full_name, u.profile_id, u.avatar_data_url, u.email
    FROM ai_router.sessions s
    JOIN ai_router.users u ON u.user_id = s.user_id
    WHERE s.token = %s AND s.expires_at >= now() AND u.is_active
    """
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (token,))
            row = cur.fetchone()
    if not row:
        return None
    return {
        "user_id": row[0],
        "username": row[1],
        "must_change_password": row[2],
        "is_admin": row[3],
        "full_name": row[4],
        "profile_id": row[5],
        "avatar_data_url": row[6],
        "email": row[7],
    }


def change_credentials(user_id: int, current_password: str, new_username: str, new_password: str) -> dict[str, Any] | None:
    """Update username/password after verifying the current password. Clears the first-access flag."""
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT username, password_hash FROM ai_router.users WHERE user_id = %s", (user_id,))
            row = cur.fetchone()
            if not row or not verify_password(row[1], current_password):
                return None
            username = new_username.strip() or row[0]
            cur.execute(
                "UPDATE ai_router.users SET username = %s, password_hash = %s, must_change_password = false WHERE user_id = %s",
                (username, hash_password(new_password), user_id),
            )
        conn.commit()
    return {"user_id": user_id, "username": username, "must_change_password": False}


def delete_session(token: str) -> None:
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM ai_router.sessions WHERE token = %s", (token,))
        conn.commit()


# ---------------------------------------------------------------------------
# User accounts + access profiles (ForgeHub-style RBAC, db/025). A profile is
# a named set of per-module view/query/write/delete flags; each user has at
# most one profile. Admins bypass the map entirely; a non-admin without a
# profile sees nothing. Business rules (last-admin protection, self-delete)
# live at the route layer in app/main.py, same split as ForgeHub.
# ---------------------------------------------------------------------------

DASHBOARD_MODULES = ["agents", "overview", "messages", "routing", "tasks", "playground", "users", "profiles"]

_ALL_FLAGS = {"can_view": True, "can_query": True, "can_write": True, "can_delete": True}
_NO_FLAGS = {"can_view": False, "can_query": False, "can_write": False, "can_delete": False}


def user_permissions(user: dict[str, Any]) -> dict[str, dict[str, bool]]:
    """Module → flags map for a session user (as returned by session_user)."""
    if user.get("is_admin"):
        return {module: dict(_ALL_FLAGS) for module in DASHBOARD_MODULES}
    permissions = {module: dict(_NO_FLAGS) for module in DASHBOARD_MODULES}
    profile_id = user.get("profile_id")
    if not profile_id:
        return permissions
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT module, can_view, can_query, can_write, can_delete"
                " FROM ai_router.profile_permissions WHERE profile_id = %s",
                (profile_id,),
            )
            for module, can_view, can_query, can_write, can_delete in cur.fetchall():
                if module in permissions:
                    permissions[module] = {
                        "can_view": can_view,
                        "can_query": can_query,
                        "can_write": can_write,
                        "can_delete": can_delete,
                    }
    return permissions


def list_users() -> list[dict[str, Any]]:
    query = """
    SELECT u.user_id, u.username, u.full_name, u.is_admin, u.is_active,
           u.must_change_password, u.profile_id, p.name, u.created_at, u.avatar_data_url, u.email
    FROM ai_router.users u
    LEFT JOIN ai_router.profiles p ON p.profile_id = u.profile_id
    ORDER BY u.username
    """
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            rows = cur.fetchall()
    return [
        {
            "user_id": r[0], "username": r[1], "full_name": r[2], "is_admin": r[3],
            "is_active": r[4], "must_change_password": r[5], "profile_id": r[6],
            "profile_name": r[7], "created_at": r[8].isoformat() if r[8] else None,
            "avatar_data_url": r[9], "email": r[10],
        }
        for r in rows
    ]


def create_user(
    username: str, password: str, full_name: str | None, is_admin: bool,
    profile_id: int | None, avatar_data_url: str | None = None, email: str | None = None,
) -> int:
    """New users always start with must_change_password so the assigned
    password never outlives first login — same policy as the seeded admin."""
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO ai_router.users (username, password_hash, must_change_password, full_name, is_admin, profile_id, avatar_data_url, email)"
                " VALUES (%s, %s, true, %s, %s, %s, %s, %s) RETURNING user_id",
                (username, hash_password(password), full_name, is_admin, profile_id, avatar_data_url, email),
            )
            user_id = cur.fetchone()[0]
        conn.commit()
    return user_id


def update_user(user_id: int, fields: dict[str, Any]) -> bool:
    """Partial update. `password` is hashed and re-arms must_change_password."""
    allowed = {"username", "full_name", "is_admin", "is_active", "profile_id", "avatar_data_url", "email"}
    sets: list[str] = []
    values: list[Any] = []
    for key, value in fields.items():
        if key == "password":
            sets.append("password_hash = %s")
            values.append(hash_password(value))
            sets.append("must_change_password = true")
        elif key in allowed:
            sets.append(f"{key} = %s")
            values.append(value)
    if not sets:
        return True
    values.append(user_id)
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(f"UPDATE ai_router.users SET {', '.join(sets)} WHERE user_id = %s", values)
            updated = cur.rowcount > 0
        conn.commit()
    return updated


def delete_user(user_id: int) -> bool:
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM ai_router.users WHERE user_id = %s", (user_id,))
            deleted = cur.rowcount > 0
        conn.commit()
    return deleted


def count_active_admins() -> int:
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM ai_router.users WHERE is_admin AND is_active")
            return cur.fetchone()[0]


def list_profiles() -> list[dict[str, Any]]:
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT profile_id, name, description, created_at FROM ai_router.profiles ORDER BY name")
            profile_rows = cur.fetchall()
            cur.execute(
                "SELECT profile_id, module, can_view, can_query, can_write, can_delete FROM ai_router.profile_permissions"
            )
            perm_rows = cur.fetchall()
            cur.execute("SELECT profile_id, count(*) FROM ai_router.users WHERE profile_id IS NOT NULL GROUP BY profile_id")
            usage = dict(cur.fetchall())
    perms_by_profile: dict[int, list[dict[str, Any]]] = {}
    for profile_id, module, can_view, can_query, can_write, can_delete in perm_rows:
        perms_by_profile.setdefault(profile_id, []).append(
            {"module": module, "can_view": can_view, "can_query": can_query, "can_write": can_write, "can_delete": can_delete}
        )
    return [
        {
            "profile_id": r[0], "name": r[1], "description": r[2],
            "created_at": r[3].isoformat() if r[3] else None,
            "permissions": sorted(perms_by_profile.get(r[0], []), key=lambda p: DASHBOARD_MODULES.index(p["module"]) if p["module"] in DASHBOARD_MODULES else 99),
            "users_count": usage.get(r[0], 0),
        }
        for r in profile_rows
    ]


def save_profile(name: str, description: str | None, permissions: list[dict[str, Any]], profile_id: int | None = None) -> int:
    """Create (profile_id None) or fully replace (profile_id set) a profile
    and its permission rows — the form always submits the whole matrix."""
    with db_connect() as conn:
        with conn.cursor() as cur:
            if profile_id is None:
                cur.execute(
                    "INSERT INTO ai_router.profiles (name, description) VALUES (%s, %s) RETURNING profile_id",
                    (name, description),
                )
                profile_id = cur.fetchone()[0]
            else:
                cur.execute(
                    "UPDATE ai_router.profiles SET name = %s, description = %s WHERE profile_id = %s",
                    (name, description, profile_id),
                )
                cur.execute("DELETE FROM ai_router.profile_permissions WHERE profile_id = %s", (profile_id,))
            for perm in permissions:
                module = perm.get("module")
                if module not in DASHBOARD_MODULES:
                    continue
                cur.execute(
                    "INSERT INTO ai_router.profile_permissions (profile_id, module, can_view, can_query, can_write, can_delete)"
                    " VALUES (%s, %s, %s, %s, %s, %s)",
                    (
                        profile_id, module,
                        bool(perm.get("can_view")), bool(perm.get("can_query")),
                        bool(perm.get("can_write")), bool(perm.get("can_delete")),
                    ),
                )
        conn.commit()
    return profile_id


def delete_profile(profile_id: int) -> bool:
    """Users referencing it fall back to no profile (FK ON DELETE SET NULL)."""
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM ai_router.profiles WHERE profile_id = %s", (profile_id,))
            deleted = cur.rowcount > 0
        conn.commit()
    return deleted


def runtime_failure_health_result(model: Any, http_code: int | None, error_message: str, cooldown_seconds: int | None = None) -> HealthResult:
    return HealthResult(
        model_id=model.id,
        status="unhealthy",
        http_code=http_code,
        latency_ms=0,
        error_message=error_message,
        cooldown_seconds=cooldown_seconds,
    )


def mark_runtime_failure_unhealthy(model: Any, http_code: int | None, error_message: str, cooldown_seconds: int | None = None) -> None:
    persist_health_results([runtime_failure_health_result(model, http_code, error_message, cooldown_seconds)])


def get_task_map() -> list[dict[str, Any]]:
    """Hermes task map rows (task → recommended forgerouter group or model)."""
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT task, hint, model FROM ai_router.task_map ORDER BY position, task")
            rows = cur.fetchall()
    return [{"task": row[0], "hint": row[1], "model": row[2]} for row in rows]


def set_task_map_model(task: str, model: str) -> bool:
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE ai_router.task_map SET model = %s, updated_at = now() WHERE task = %s",
                (model, task),
            )
            updated = cur.rowcount > 0
        conn.commit()
    return updated


def list_subscription_catalog() -> list[dict[str, Any]]:
    """Known subscription providers ("coding plans") seeded in 011_provider_access.sql."""
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT name, display_name, plan_hint, base_url, auth_method, token_hint, auth_url, extra_headers
                FROM ai_router.subscription_catalog
                ORDER BY display_name
                """
            )
            rows = cur.fetchall()
    return [
        {
            "name": row[0],
            "display_name": row[1],
            "plan_hint": row[2],
            "base_url": row[3],
            "auth_method": row[4],
            "token_hint": row[5],
            "auth_url": row[6] or "",
            "extra_headers": row[7] or {},
        }
        for row in rows
    ]
