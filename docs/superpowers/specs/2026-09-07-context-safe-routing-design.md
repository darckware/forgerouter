# Context-safe routing design

## Objective

ForgeRouter must never knowingly send a prompt to a model whose documented input context cannot hold it. It should preserve the complete conversation whenever at least one compatible model can accept it, summarize and truncate only when no compatible model can accept it intact, and return a clear client error when protected content cannot be made to fit.

This design applies to virtual routes (`auto` and `forgerouter/*`) and concrete-model preferences. It preserves ForgeRouter's existing automatic fallback contract: a requested concrete model remains a preference rather than an exclusive target.

## Context concepts

`context_window()` is the model's input-side limit sourced from provider metadata, overrides, or the LiteLLM catalog. Candidate-fit decisions compare that limit with the actual estimated input payload, including tools. Output limits remain governed separately by `max_tokens`; they are not subtracted from an input-specific limit.

ForgeRouter will derive a usable input budget as `trigger_percent` of a known context window. The default remains 80%, providing tokenizer-estimation and provider-overhead headroom. For an unknown window, the existing `context_truncation_max_tokens` setting (default 32k) is the only safe budget.

Hermes continues to discover the virtual model's conservative window from `/v1/models` and compact its own conversation at its configured threshold. ForgeRouter remains the final safety boundary.

## Candidate classification

After lossless message normalization, ForgeRouter estimates the actual payload size and classifies capability-compatible candidates as:

- `fit`: known context budget is at least the payload size.
- `unknown_fit`: context is unknown and the payload is no larger than the configured unknown-window budget.
- `too_small`: known context budget is below the payload size.
- `unknown_unsafe`: context is unknown and the payload exceeds the configured unknown-window budget.

Known `fit` candidates retain the existing demand, sticky, tier, breaker, performance, and rate-limit ordering. `unknown_fit` candidates come after known-fit candidates. `too_small` and `unknown_unsafe` candidates are not called with the unmodified payload.

Models with a known raw window below Hermes's 64k virtual-route floor do not participate in virtual routes. They remain addressable as concrete-model preferences, subject to the same fit check and fallback behavior. Unknown-window models are allowed only under the conservative unknown-window budget and sort last.

## Routing and compaction flow

1. Resolve capability, agent model permissions, health, and demand as today.
2. Apply lossless normalization before context-aware ordering and count the normalized payload plus tools.
3. Classify and order candidates using their usable context budgets.
4. If at least one candidate is `fit` or `unknown_fit`, send the complete normalized payload only to those candidates. Known-incompatible candidates are excluded from fallback attempts.
5. If no candidate fits and context truncation is disabled, return HTTP 413 `context_too_large` without making a provider call.
6. If no candidate fits and context truncation is enabled, select the largest known usable context budget. Existing routing priority breaks ties. Summarize and remove oldest complete turns until the payload fits that target budget.
7. Recount after inserting the summary. If the summary causes overflow, shorten or omit it and recount. System messages and the final user turn remain protected.
8. Reclassify candidates against the resulting payload. Retain only candidates that can safely accept it; this prevents a smaller fallback from receiving a payload prepared for a larger model.
9. If protected content alone exceeds every known budget and the unknown budget, return HTTP 413 instead of calling a provider.

The truncation target is deliberately not the smallest model in the original pool. Shrinking every request to preserve a tiny fallback wastes the larger models' usable context. Fallback breadth is reduced only for that oversized request.

## HTTP 413 contract

The error uses the existing OpenAI-compatible envelope:

```json
{
  "error": {
    "message": "The protected prompt content exceeds every available model context window.",
    "type": "context_too_large",
    "prompt_tokens": 120000,
    "max_input_budget": 100000
  }
}
```

No model is marked unhealthy for this response because no provider call failed.

## Virtual model metadata

`/v1/models` will calculate virtual context from the candidates that can actually serve the authenticated agent:

- Apply agent model permissions when a valid agent bearer key is present.
- Honor configured demand routes before falling back to rank-derived default chains.
- Exclude known sub-64k models from virtual-route guarantees.
- Treat any eligible unknown-window candidate as forcing the advertised compatibility floor of 64k. This is a client-compatibility value rather than proof of that backend's capacity; request-time unknown-budget enforcement remains authoritative.
- Advertise each demand route's minimum guaranteed window; `forgerouter/auto` advertises the minimum across all demand routes it may select.

Unauthenticated discovery retains the current all-model view. Empty or entirely unknown pools advertise the 64k floor.

## Observability

Route diagnostics will expose enough information to explain the decision without persisting conversation content:

- normalized estimated prompt tokens;
- selected model context window and usable budget;
- context action: `none`, `summarized`, `truncated`, or `rejected`;
- count of context-incompatible candidates skipped;
- messages dropped, using the existing field.

Skipped candidates are not recorded as failed provider attempts and do not affect model health, breaker state, or success-rate statistics.

## Failure handling

- Token counting failure: preserve existing availability behavior, skip proactive filtering, and continue routing. Provider context errors can still trigger normal fallback.
- Context metadata missing: allow the model only when the prompt is within the configured unknown-window budget.
- Summarizer failure: fall back to dropping complete oldest turns, as today.
- Summary overflow: omit the summary rather than violate the selected budget.
- No removable history: return 413.
- Provider reports a smaller authoritative limit: retain Hermes's existing error learning and cache update behavior; subsequent requests use the corrected limit once catalog/cache integration supplies it.

## Compatibility

- Existing demand classification, capability filtering, sticky routing, circuit breakers, rate-limit handling, and provider fallback semantics remain unchanged inside the safe candidate set.
- Context truncation remains controlled by the existing setting. Disabled means reject oversized requests rather than silently discard content.
- Lossless whitespace compaction remains enabled independently.
- No secrets or message bodies are added to logs or error responses.

## Testing

Backend regression tests will cover:

1. A fitting model receives the complete payload while smaller candidates are never called.
2. Unknown-window candidates sort after known fitting models and are excluded above the unknown budget.
3. Known sub-64k models are excluded from virtual routes but remain usable through a concrete preference when the payload fits.
4. No-fit plus truncation disabled returns 413 without provider calls or unhealthy marks.
5. No-fit plus truncation enabled targets the largest usable window, summarizes/truncates, recounts, and rebuilds the fallback pool.
6. Protected content that cannot fit returns 413.
7. Summary overflow is handled without exceeding the budget.
8. `/v1/models` honors configured demand routes and authenticated agent model permissions.
9. Token-count failure preserves the existing availability fallback.
10. Existing routing, streaming, Anthropic Messages, Responses API, caching, usage accounting, and frontend suites remain green.

## Rollout

Candidate safety filtering and explicit 413 responses are always active; only the lossy summarization/truncation path remains behind the existing context-truncation setting. No new feature flag is required. Enable truncation in the target environment after deployment. Monitor 413 counts, skipped-candidate counts, and messages dropped before considering a change to the default setting.
