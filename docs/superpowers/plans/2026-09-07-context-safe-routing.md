# Context-safe Routing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent ForgeRouter from knowingly sending oversized prompts, preserving full context when a model fits and summarizing/truncating only when none does.

**Architecture:** Add a pure context-policy module that classifies already-ordered candidates by usable input budget. Integrate it after existing demand/health/routing-state ordering, then either retain the full payload, truncate toward the largest viable budget and rebuild the pool, or return an OpenAI-compatible 413. Make virtual model metadata agent-aware and persist decision diagnostics without storing prompt bodies.

**Tech Stack:** Python 3.12, FastAPI, Pydantic, PostgreSQL SQL migrations, pytest, React 19, TypeScript, Vite, Node test runner.

**Spec:** `docs/superpowers/specs/2026-09-07-context-safe-routing-design.md`

## Global Constraints

- Never knowingly call a model whose documented usable input budget is below the estimated payload size.
- Preserve existing demand, capability, sticky, tier, breaker, rate-limit, fallback, streaming, cache, and agent-auth semantics inside the safe candidate set.
- `context_window` is input-side; do not subtract the output-only `max_tokens` parameter.
- Use `context_truncation_trigger_percent` (default 80) for known windows and `context_truncation_max_tokens` (default 32k) for unknown windows.
- Unknown-window candidates sort after known-fitting candidates and are allowed only within the unknown-window budget.
- Known sub-64k models do not serve virtual routes; concrete preferences may use them when the payload fits.
- System messages and the final user turn are protected; return HTTP 413 if protected content cannot fit.
- Database failures and token-counting failures must never break otherwise viable routing.
- Do not persist secrets or conversation bodies.
- Preserve unrelated dirty-worktree changes and stage only task-owned files in each commit.

---

## File structure

- Create `app/context_policy.py`: pure budget calculation and candidate partitioning.
- Create `tests/test_context_policy.py`: unit tests for known, unknown, virtual-floor, and counting-unavailable cases.
- Modify `app/main.py`: normalize before policy evaluation; safe-pool selection; target-budget truncation; 413 response; agent-aware model metadata; context diagnostics propagation.
- Modify `app/demand.py`: leave demand/rank ordering to this module and remove duplicated prompt-fit/minimum-window filtering once the central policy owns it.
- Modify `tests/test_demand_routing.py`: endpoint-level routing, truncation, rejection, and virtual metadata regressions.
- Modify `tests/test_anthropic_messages.py` and `tests/test_responses_api.py`: translated-protocol 413 behavior.
- Create `db/053_context_routing_diagnostics.sql`: route-event diagnostic columns.
- Modify `app/storage.py`: persist and return context diagnostics.
- Create `tests/test_storage_context.py`: SQL row/response mapping tests.
- Modify `frontend/src/main.tsx`: type and display context decisions in Messages.
- Rebuild `frontend/dist/`: deployable dashboard bundle.
- Modify `CLAUDE.md`: document context-safe candidate filtering and 413 behavior.

---

### Task 1: Pure context policy

**Files:**
- Create: `app/context_policy.py`
- Create: `tests/test_context_policy.py`
- Modify: `app/demand.py`
- Modify: `tests/test_demand_routing.py`

**Interfaces:**
- Consumes: `ProviderModel`, `app.pricing.context_window`, ordered candidate lists.
- Produces: `model_context_budget(model, trigger_percent, unknown_budget) -> tuple[int, bool]`, `partition_candidates(candidates, prompt_tokens, trigger_percent, unknown_budget, virtual_route) -> ContextPartition`, and `ContextPartition(fitting, excluded, max_input_budget)`.

- [ ] **Step 1: Write failing policy tests**

```python
from app.context_policy import partition_candidates
from app.registry import ProviderModel


def model(public_id: str, tier: int) -> ProviderModel:
    return ProviderModel(
        public_id, public_id.split("/")[0], public_id.split("/", 1)[1],
        tier, ["text"], True, True, "http://provider.test/v1", "",
    )


def test_partition_keeps_known_fit_and_puts_unknown_last(monkeypatch):
    roomy, unknown, small = model("p/roomy", 1), model("p/unknown", 1), model("p/small", 1)
    windows = {"p/roomy": 200_000, "p/small": 70_000}
    monkeypatch.setattr("app.context_policy.context_window", lambda public_id, _: windows.get(public_id))

    result = partition_candidates(
        [unknown, small, roomy], prompt_tokens=60_000,
        trigger_percent=80, unknown_budget=64_000, virtual_route=True,
    )

    assert [m.id for m in result.fitting] == ["p/roomy", "p/unknown"]
    assert [m.id for m in result.excluded] == ["p/small"]
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
```

The production mutation each test catches is respectively: unknown candidates incorrectly preceding known fits; the virtual 64k floor not enforced centrally; unsafe unknown candidates being attempted; and tokenizer failure causing an outage.

- [ ] **Step 2: Run the policy tests and verify RED**

Run:

```bash
docker compose run --rm forgerouter pytest tests/test_context_policy.py -q
```

Expected: FAIL because `app.context_policy` does not exist.

- [ ] **Step 3: Implement the policy module minimally**

```python
from dataclasses import dataclass

from app.pricing import context_window
from app.registry import ProviderModel

MINIMUM_VIRTUAL_CONTEXT = 64_000


@dataclass(frozen=True)
class ContextPartition:
    fitting: list[ProviderModel]
    excluded: list[ProviderModel]
    max_input_budget: int | None


def model_context_budget(model: ProviderModel, trigger_percent: int, unknown_budget: int) -> tuple[int, bool]:
    window = context_window(model.id, model.provider_model)
    if window is None:
        return unknown_budget, False
    return int(window * trigger_percent / 100), True


def partition_candidates(candidates, prompt_tokens, trigger_percent, unknown_budget, virtual_route):
    if prompt_tokens is None:
        return ContextPartition(list(candidates), [], None)
    known_fit, unknown_fit, excluded, budgets = [], [], [], []
    for candidate in candidates:
        window = context_window(candidate.id, candidate.provider_model)
        budget, known = model_context_budget(candidate, trigger_percent, unknown_budget)
        budgets.append(budget)
        if virtual_route and known and window < MINIMUM_VIRTUAL_CONTEXT:
            excluded.append(candidate)
        elif prompt_tokens <= budget:
            (known_fit if known else unknown_fit).append(candidate)
        else:
            excluded.append(candidate)
    return ContextPartition(known_fit + unknown_fit, excluded, max(budgets, default=None))
```

- [ ] **Step 4: Centralize the 64k rule**

Remove `_meets_minimum_context`, `_fits_prompt`, and `estimated_tokens` from `app/demand.py:218-267`. Restore `default_chain()` to pure demand/rank ordering. Update old tests that assert “deprioritize but retain” so they instead exercise `partition_candidates()` and the new hard virtual-route exclusion.

- [ ] **Step 5: Run policy and demand tests**

Run:

```bash
docker compose run --rm forgerouter pytest tests/test_context_policy.py tests/test_demand_routing.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit Task 1**

```bash
git add app/context_policy.py app/demand.py tests/test_context_policy.py tests/test_demand_routing.py
git commit -m "feat(routing): centralize context-fit policy"
```

---

### Task 2: Safe routing, largest-budget truncation, and 413

**Files:**
- Modify: `app/main.py:1024-1375`
- Modify: `app/normalize.py:70-115`
- Modify: `tests/test_demand_routing.py`

**Interfaces:**
- Consumes: Task 1's `partition_candidates()` and `model_context_budget()`.
- Produces: `_context_too_large(prompt_tokens, max_input_budget) -> JSONResponse` and `_prepare_context_payload(messages, tools, candidates, registry, truncation_on, trigger_percent, unknown_budget, virtual_route) -> ContextPayload`.
- `ContextPayload` contains `messages`, `candidates`, `tokens`, `messages_dropped`, `action`, `skipped`, and `selected_budget`.

- [ ] **Step 1: Write failing endpoint tests for complete-payload filtering**

```python
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
```

Add a companion test where a known-fit model and an unknown-fit model both exist and the known model fails with 429; assert the unknown model is attempted second.

- [ ] **Step 2: Run the new filtering tests and verify RED**

Run the two exact pytest node IDs. Expected: FAIL because the small/unknown candidate remains in current fallback ordering.

- [ ] **Step 3: Move normalization and counting before context filtering**

In `chat_completions()`, build `raw_messages`, apply `normalize_messages()` when enabled, and count `messages_for_payload` plus tools before context policy. Retain `tokens_raw` for accounting and name the normalized estimate `tokens_compacted`. Do not change cache-key input, which intentionally reflects the caller's original request.

- [ ] **Step 4: Add the payload-preparation result type and 413 helper**

```python
@dataclass(frozen=True)
class ContextPayload:
    messages: list[dict[str, Any]]
    candidates: list[ProviderModel]
    tokens: int | None
    messages_dropped: int
    action: str
    skipped: int
    selected_budget: int | None


def _context_too_large(prompt_tokens: int | None, max_input_budget: int | None) -> JSONResponse:
    return JSONResponse(status_code=413, content={"error": {
        "message": "The protected prompt content exceeds every available model context window.",
        "type": "context_too_large",
        "prompt_tokens": prompt_tokens,
        "max_input_budget": max_input_budget,
    }})
```

- [ ] **Step 5: Implement the no-loss fitting path**

Call `partition_candidates()` after all existing candidate ordering. When `partition.fitting` is non-empty, return a `ContextPayload` with the original normalized messages and only fitting candidates. `action="none"`; `skipped=len(partition.excluded)`.

- [ ] **Step 6: Write failing 413 tests**

Cover truncation disabled and “system plus final turn alone is too large.” Assert status 413, `error.type == "context_too_large"`, zero provider calls, and zero unhealthy calls.

- [ ] **Step 7: Run the 413 tests and verify RED**

Expected: current code calls a provider or returns 502 rather than 413.

- [ ] **Step 8: Implement largest-budget truncation and recount**

When no candidate fits:

```python
if not truncation_on:
    return _context_too_large(tokens, partition.max_input_budget)
target_budget = partition.max_input_budget
trimmed, dropped, dropped_messages, fits = truncate_messages(messages, target_budget, tools)
if not fits:
    return _context_too_large(tokens, target_budget)
if dropped:
    summary = _summarize_dropped_context(dropped_messages, registry)
    action = "truncated"
    if summary:
        insert_at = sum(1 for message in trimmed if message.get("role") == "system")
        summarized = trimmed[:insert_at] + [{
            "role": "system",
            "content": f"[Earlier conversation summary — {dropped} message(s) condensed to save context]\n{summary}",
        }] + trimmed[insert_at:]
        summarized_tokens = count_tokens(summarized, tools)
        if summarized_tokens is not None and summarized_tokens <= target_budget:
            trimmed, action = summarized, "summarized"
trimmed_tokens = count_tokens(trimmed, tools)
safe = partition_candidates(candidates, trimmed_tokens, trigger_percent, unknown_budget, virtual_route)
if not safe.fitting:
    return _context_too_large(trimmed_tokens, target_budget)
```

Call `count_tokens()` once per branch and store the result instead of repeating it. If counting unexpectedly becomes unavailable during recount, preserve availability and use the ordered candidate list.

- [ ] **Step 9: Make `truncate_messages()` report an unfit protected tail**

Extend its return type to include `fits: bool`, computed by a final count after oldest removable turns are exhausted. Update all existing callers/tests. This distinguishes “nothing more can be removed” from successful truncation without duplicating turn-protection logic in `main.py`.

- [ ] **Step 10: Replace the obsolete smallest-window regression**

Replace `test_chat_truncation_budget_uses_minimum_window_across_all_fallback_candidates` with a test asserting a 200k/50k pool targets 160k (at 80%), drops history only if needed, and excludes the 50k model from the rebuilt fallback pool.

- [ ] **Step 11: Run routing and normalization tests**

```bash
docker compose run --rm forgerouter pytest tests/test_context_policy.py tests/test_demand_routing.py tests/test_normalize.py tests/test_chat_fallback.py -q
```

Expected: PASS.

- [ ] **Step 12: Commit Task 2**

```bash
git add app/main.py app/normalize.py tests/test_demand_routing.py tests/test_normalize.py
git commit -m "feat(routing): reject or compact oversized context safely"
```

---

### Task 3: Agent-aware virtual context metadata

**Files:**
- Modify: `app/main.py:150-168,376-390`
- Modify: `tests/test_demand_routing.py`
- Modify: `tests/test_v1_agent_auth.py`

**Interfaces:**
- Consumes: `find_agent_by_key()`, `agent_allowed_models()`, `get_demand_routes()`, and Task 1's virtual eligibility rule.
- Produces: `_virtual_model_context_lengths(registry, allowed=None, configured_routes=None) -> dict[str, int]` and `_optional_agent_models(request) -> set[str] | None`.

- [ ] **Step 1: Write failing model-metadata tests**

Add tests that:

```python
def test_models_virtual_context_uses_authenticated_agent_pool(monkeypatch):
    # agent key allows only the 300k model; global pool also has a 70k model
    response = client.get("/v1/models", headers={"Authorization": "Bearer agent-key"})
    entries = {item["id"]: item for item in response.json()["data"]}
    virtual = entries["forgerouter/auto"]
    assert virtual["context_length"] == 300_000


def test_models_virtual_context_uses_configured_chain_before_default(monkeypatch):
    # configured reasoning head is included in the actual ordered route, with
    # remaining eligible fallbacks included in its guaranteed minimum
    entries = {item["id"]: item for item in client.get("/v1/models").json()["data"]}
    assert entries["forgerouter/reasoning"]["context_length"] == 128_000
```

Also cover DB lookup failure: endpoint remains 200 and uses the unrestricted pool.

- [ ] **Step 2: Run the exact tests and verify RED**

Expected: `/v1/models` ignores Authorization and `_virtual_model_context_lengths()` ignores configured routes.

- [ ] **Step 3: Add optional agent scoping to `/v1/models`**

Change the endpoint to `def models(raw_request: Request)`. Resolve a valid bearer key with `find_agent_by_key`; if resolved, load `agent_allowed_models`. Any lookup exception yields `allowed=None` and preserves the unrestricted discovery behavior. Unlike chat completion, model discovery remains public and does not return 401 for an absent/invalid key.

- [ ] **Step 4: Make virtual metadata mirror reachable ordered pools**

Pass `allowed` and a best-effort `get_demand_routes()` mapping into `_virtual_model_context_lengths()`. For each demand, build the configured/default ordered chain, append remaining capability-compatible healthy fallbacks, apply the virtual 64k eligibility rule, then take the minimum known raw window. An allowed unknown-window candidate lowers the advertised guarantee to 64k. `auto` remains the minimum of all demand results.

- [ ] **Step 5: Run metadata and auth tests**

```bash
docker compose run --rm forgerouter pytest tests/test_demand_routing.py tests/test_v1_agent_auth.py tests/test_registry_and_chat.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit Task 3**

```bash
git add app/main.py tests/test_demand_routing.py tests/test_v1_agent_auth.py
git commit -m "feat(models): scope virtual context to reachable agent routes"
```

---

### Task 4: Persist and display context-routing diagnostics

**Files:**
- Create: `db/053_context_routing_diagnostics.sql`
- Modify: `app/storage.py:217-280,435-482`
- Modify: `app/main.py` persistence calls and `_stream_and_persist_usage()` signature
- Create: `tests/test_storage_context.py`
- Modify: `frontend/src/main.tsx:12,3460-3480`
- Rebuild: `frontend/dist/index.html`, `frontend/dist/assets/*`

**Interfaces:**
- Extends `persist_route_event(..., context_window=None, context_budget=None, context_action="none", context_candidates_skipped=0)`.
- Extends `/admin/routes/recent` rows with the same four fields.

- [ ] **Step 1: Write the idempotent migration**

```sql
ALTER TABLE ai_router.route_events
    ADD COLUMN IF NOT EXISTS context_window INTEGER,
    ADD COLUMN IF NOT EXISTS context_budget INTEGER,
    ADD COLUMN IF NOT EXISTS context_action TEXT,
    ADD COLUMN IF NOT EXISTS context_candidates_skipped INTEGER NOT NULL DEFAULT 0;
```

- [ ] **Step 2: Write failing storage mapping tests**

Create `tests/test_storage_context.py` with a minimal context-manager fake around `app.storage.db_connect`:

```python
from contextlib import contextmanager
from datetime import datetime, timezone

import app.storage as storage


class FakeCursor:
    def __init__(self, rows=()):
        self.rows = rows
        self.query = None
        self.params = None
    def __enter__(self): return self
    def __exit__(self, *_): return False
    def execute(self, query, params): self.query, self.params = query, params
    def fetchall(self): return self.rows


class FakeConnection:
    def __init__(self, cursor): self.fake_cursor = cursor
    def __enter__(self): return self
    def __exit__(self, *_): return False
    def cursor(self): return self.fake_cursor
    def commit(self): pass


def patch_connection(monkeypatch, cursor):
    @contextmanager
    def connect():
        yield FakeConnection(cursor)
    monkeypatch.setattr(storage, "db_connect", connect)


def test_persist_route_event_binds_context_diagnostics(monkeypatch):
    cursor = FakeCursor()
    patch_connection(monkeypatch, cursor)
    storage.persist_route_event(
        "00000000-0000-0000-0000-000000000001", "p/model", "text", "success",
        context_window=200_000, context_budget=160_000,
        context_action="summarized", context_candidates_skipped=2,
    )
    assert cursor.params["context_window"] == 200_000
    assert cursor.params["context_budget"] == 160_000
    assert cursor.params["context_action"] == "summarized"
    assert cursor.params["context_candidates_skipped"] == 2


def test_recent_routes_returns_context_diagnostics(monkeypatch):
    row = (1, "request", "p/model", "text", "success", None,
           datetime.now(timezone.utc), 10, 0, "agent", "simple", None,
           "preview", 3, 200_000, 160_000, "summarized", 2)
    cursor = FakeCursor([row])
    patch_connection(monkeypatch, cursor)
    event = storage.recent_route_events()[0]
    assert event["context_window"] == 200_000
    assert event["context_budget"] == 160_000
    assert event["context_action"] == "summarized"
    assert event["context_candidates_skipped"] == 2
```

The mutation caught is silently dropping diagnostics at the SQL boundary.

- [ ] **Step 3: Run the storage tests and verify RED**

Expected: unexpected keyword arguments or missing response keys.

- [ ] **Step 4: Implement storage persistence and reads**

Add the parameters, row keys, INSERT columns and bound values, SELECT columns, and response mapping. Use `None` for unknown window/budget, `"none"` for no context mutation, and `0` for no skipped candidates.

- [ ] **Step 5: Propagate diagnostics through every persistence path**

Pass `ContextPayload.action`, `.skipped`, and `.selected_budget`; pass the selected model's raw `context_window`. Extend `_stream_and_persist_usage()` so streaming success/failure persists identical diagnostics. For a pre-provider 413, persist one `selected_model_id=None`, `status="rejected"`, `error_type="context_too_large"` event in a best-effort try/except.

- [ ] **Step 6: Run backend diagnostic tests**

```bash
docker compose run --rm forgerouter pytest tests/test_storage_context.py tests/test_demand_routing.py tests/test_chat_fallback.py -q
```

Expected: PASS.

- [ ] **Step 7: Extend the Messages UI**

Add to `RouteEvent`:

```ts
context_window: number | null;
context_budget: number | null;
context_action: 'none' | 'summarized' | 'truncated' | 'rejected' | null;
context_candidates_skipped: number;
```

In the expanded message detail, show a compact line only when an action occurred or candidates were skipped, for example: `context summarized · budget 160k · 2 incompatible candidates skipped`. Reuse `formatContextLength()` and existing status classes.

- [ ] **Step 8: Test and build the frontend**

```bash
cd frontend
npm test
npm run build
```

Expected: all Node tests pass and Vite emits a production bundle without TypeScript/build errors.

- [ ] **Step 9: Commit Task 4**

```bash
git add db/053_context_routing_diagnostics.sql app/storage.py app/main.py tests frontend/src/main.tsx frontend/dist
git commit -m "feat(observability): expose context routing decisions"
```

Stage only test files actually changed by this task; do not stage unrelated dirty files.

---

### Task 5: Protocol compatibility, documentation, and full verification

**Files:**
- Modify: `tests/test_anthropic_messages.py`
- Modify: `tests/test_responses_api.py`
- Modify: `CLAUDE.md`

**Interfaces:**
- Consumes: Task 2's OpenAI-compatible 413 response.
- Produces: stable Anthropic Messages and OpenAI Responses error translations for `context_too_large`.

- [ ] **Step 1: Add translator regression tests**

For `/v1/messages`, submit an oversized request under patched counts/windows and assert HTTP 413 with the existing pass-through body retaining `error.type == "context_too_large"`. For `/v1/responses`, assert the same HTTP 413 OpenAI-compatible envelope. Assert zero upstream provider calls for both.

Use this setup in each existing protocol test file, adapting only that endpoint's request body:

```python
candidate = model("p/small", 1)
monkeypatch.setattr("app.main.load_registry_with_db_health", lambda: ProviderRegistry([candidate]))
monkeypatch.setattr("app.main.count_tokens", lambda *_: 100_000)
monkeypatch.setattr("app.context_policy.context_window", lambda *_: 64_000)
monkeypatch.setattr("app.main.context_truncation_enabled", lambda: False)
calls = []
monkeypatch.setattr("app.main.chat_completion", lambda *args: calls.append(args) or (200, {}))

response = client.post(PROTOCOL_PATH, json=PROTOCOL_OVERSIZED_BODY)

assert response.status_code == 413
assert response.json()["error"]["type"] == "context_too_large"
assert calls == []
```

For Messages, `PROTOCOL_PATH` is `/v1/messages` and the body contains `model`, `max_tokens`, and a user message. For Responses, it is `/v1/responses` and the body contains `model` and string `input`. Use each file's existing `model()` fixture/helper and request conventions rather than adding production-only test hooks.

- [ ] **Step 2: Run both protocol tests**

```bash
docker compose run --rm forgerouter pytest tests/test_anthropic_messages.py tests/test_responses_api.py -q
```

Expected: PASS because both endpoints already pass non-200 `JSONResponse` bodies through. If a regression is exposed, minimally repair the existing non-200 branch; do not duplicate context policy in either adapter.

- [ ] **Step 3: Document the final routing contract**

Update the Context truncation and Demand routing sections in `CLAUDE.md` with:

- known incompatible candidates are skipped, not marked unhealthy;
- unknown candidates are limited by the fallback budget;
- truncation targets the largest usable budget and rebuilds the pool;
- protected-content overflow returns 413;
- `/v1/models` virtual windows are agent/reachable-route aware.

- [ ] **Step 4: Run focused verification**

```bash
docker compose run --rm forgerouter pytest \
  tests/test_context_policy.py tests/test_demand_routing.py tests/test_normalize.py \
  tests/test_chat_fallback.py tests/test_anthropic_messages.py tests/test_responses_api.py -q
```

Expected: PASS with zero failures.

- [ ] **Step 5: Run full verification**

```bash
docker compose run --rm forgerouter pytest -q
cd frontend && npm test && npm run build
cd .. && git diff --check
```

Expected: all backend and frontend tests pass, Vite build succeeds, and `git diff --check` emits no output.

- [ ] **Step 6: Review database rollout requirement**

Confirm `db/053_context_routing_diagnostics.sql` is listed in the handoff as a required manual migration before deploying code that writes the new columns. Do not apply it to production automatically.

- [ ] **Step 7: Commit Task 5**

```bash
git add CLAUDE.md tests/test_anthropic_messages.py tests/test_responses_api.py frontend/dist
git commit -m "docs: describe context-safe routing contract"
```

- [ ] **Step 8: Request final code review**

Use `superpowers:requesting-code-review` against the implementation commit range. Fix all Critical and Important findings, rerun the affected focused tests, then rerun the full verification command before claiming completion.
