"""Active context-window discovery.

`app/pricing.py: context_window()` only ever reports a window that some real
source (a provider's own /models response, a hand-curated override, or the
vendored LiteLLM catalog) actually documents — it never guesses. But some
providers (mostly NVIDIA's direct NIM API, unlike its OpenRouter/Kilo
aggregator aliases) publish no context-length metadata anywhere ForgeRouter
can read it. For those, this module finds out the real number by actually
asking the model: send it a real prompt padded to a target token count and
see whether the provider accepts it or rejects it as too long.

This is deliberately never run inline during a live chat_completions request
— it costs real provider quota and can take several sequential calls per
model — only from an explicit admin action (POST /admin/providers/discover-
context) or its cron counterpart (scripts/discover_context_windows.py).

Checkpoints start at MINIMUM_VIRTUAL_CONTEXT (app.context_policy) since
that's the only decision that actually matters for routing: is this model
even eligible for a virtual route. Everything past that just refines how
much bigger it is. A checkpoint result is only ever a confirmed lower bound
(the model may support more than the largest checkpoint that fit) — never an
exact number, and never a guess: an inconclusive response (auth error, rate
limit, an unrecognized error) stops refinement rather than reporting
something unverified.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from app.context_policy import MINIMUM_VIRTUAL_CONTEXT
from app.normalize import count_tokens
from app.registry import ProviderModel

_ABOVE_FLOOR_CHECKPOINTS = (128_000, 256_000, 512_000, 1_000_000)
_BELOW_FLOOR_CHECKPOINTS = (32_000, 16_000, 8_000, 4_000)

_PADDING_UNIT = "The quick brown fox jumps over the lazy dog. "

# Substrings (checked case-insensitively against the error body) that mean
# "the provider rejected this specific prompt as too long" — as opposed to
# any other 4xx (auth, rate limit, malformed request) which must never be
# misread as "the context window is smaller than this checkpoint".
_OVERFLOW_PATTERNS = (
    "context_length_exceeded",
    "context_length",
    "context window",
    "context length",
    "maximum context",
    "max_tokens",
    "too many tokens",
    "too long",
    "reduce the length",
    "reduce the number of tokens",
    "input is too long",
    "please reduce",
    "exceeds the model",
    "exceeds context",
    "token limit",
    "prompt is too long",
)


@dataclass(frozen=True)
class ContextProbeResult:
    model_id: str
    # A confirmed lower bound ("at least this many tokens fit"), or None when
    # nothing could be confirmed at all (e.g. the floor checkpoint itself was
    # inconclusive).
    context_window: int | None
    checkpoints: list[tuple[int, str]] = field(default_factory=list)  # (tokens, "fit"|"overflow"|"inconclusive:<reason>")
    note: str = ""


def _padded_messages(target_tokens: int) -> list[dict[str, Any]] | None:
    """A single user message padded to approximately `target_tokens` (tiktoken
    cl100k_base estimate — provider tokenizers vary, so this is a calibration,
    not an exact match; slight overshoot is fine and even preferable, since it
    biases toward under- rather than over-claiming what fits). None when the
    tokenizer itself is unavailable — discovery cannot calibrate without it."""
    unit_tokens = count_tokens([{"role": "user", "content": _PADDING_UNIT}])
    if not unit_tokens:
        return None
    repeats = max(1, -(-target_tokens // unit_tokens))  # ceil division
    content = _PADDING_UNIT * repeats + "\n\nReply with just the word OK."
    return [{"role": "user", "content": content}]


def _looks_like_overflow(body: Any) -> bool:
    text = json.dumps(body).lower() if isinstance(body, (dict, list)) else str(body).lower()
    return any(pattern in text for pattern in _OVERFLOW_PATTERNS)


def _classify_checkpoint(status_code: int | None, body: Any) -> str:
    """"fit" | "overflow" | "inconclusive:<reason>" for one checkpoint call."""
    if status_code == 200:
        return "fit"
    if status_code in (400, 413, 422) and _looks_like_overflow(body):
        return "overflow"
    if status_code is None:
        return "inconclusive:network_error"
    if status_code in (401, 403):
        return "inconclusive:auth_error"
    if status_code == 429:
        return "inconclusive:rate_limited"
    return f"inconclusive:http_{status_code}"


def _probe_checkpoint(model: ProviderModel, tokens: int, timeout: float) -> tuple[str, Any]:
    from app.providers.openai_compatible import chat_completion

    messages = _padded_messages(tokens)
    if messages is None:
        return "inconclusive:tokenizer_unavailable", None
    payload = {"model": model.provider_model, "messages": messages, "temperature": 0, "max_tokens": 4}
    try:
        status_code, body = chat_completion(model, payload, timeout=timeout)
    except Exception as exc:
        return f"inconclusive:{type(exc).__name__}", None
    return _classify_checkpoint(status_code, body), body


def probe_context_window(model: ProviderModel, timeout: float = 60.0, delay_seconds: float = 1.0) -> ContextProbeResult:
    """Discover a model's real context window by testing it directly, starting
    at the floor that actually matters (MINIMUM_VIRTUAL_CONTEXT) and refining
    from there. Sequential, one checkpoint at a time — this spends real
    provider quota, so it never parallelizes across checkpoints or races
    ahead past an inconclusive result."""
    checkpoints: list[tuple[int, str]] = []

    outcome, _ = _probe_checkpoint(model, MINIMUM_VIRTUAL_CONTEXT, timeout)
    checkpoints.append((MINIMUM_VIRTUAL_CONTEXT, outcome))

    if outcome.startswith("inconclusive"):
        return ContextProbeResult(model.id, None, checkpoints, f"floor checkpoint inconclusive: {outcome}")

    if outcome == "fit":
        confirmed = MINIMUM_VIRTUAL_CONTEXT
        for tokens in _ABOVE_FLOOR_CHECKPOINTS:
            time.sleep(delay_seconds)
            outcome, _ = _probe_checkpoint(model, tokens, timeout)
            checkpoints.append((tokens, outcome))
            if outcome != "fit":
                break
            confirmed = tokens
        return ContextProbeResult(model.id, confirmed, checkpoints, "confirmed lower bound — real window may be larger")

    # outcome == "overflow" at the floor: find how far below it actually goes.
    confirmed = None
    for tokens in _BELOW_FLOOR_CHECKPOINTS:
        time.sleep(delay_seconds)
        outcome, _ = _probe_checkpoint(model, tokens, timeout)
        checkpoints.append((tokens, outcome))
        if outcome == "fit":
            confirmed = tokens
            break
        if outcome.startswith("inconclusive"):
            break
    if confirmed is None:
        return ContextProbeResult(model.id, None, checkpoints, "below the 64k floor, but couldn't confirm how far — smallest checkpoint tested still overflowed or was inconclusive")
    return ContextProbeResult(model.id, confirmed, checkpoints, "confirmed lower bound, below the 64k floor")
