"""Lossless context compaction.

Strips whitespace formatting that carries no information for the model
(trailing spaces, runs of blank lines, pretty-printing in tool-result JSON)
before a chat payload is forwarded to a provider. Nothing semantic is removed
— the model sees the same content, just without incidental formatting bytes.

Token counts are estimated with tiktoken's cl100k_base encoding purely as a
common yardstick for the "tokens saved" dashboard indicator, independent of
which provider/tokenizer actually serves the request. If the encoding can't be
loaded (e.g. no network on first use), counting is skipped — this must never
break routing.

Multimodal content parts (`image_url`, `input_audio`) never reach the
tokenizer as their raw payload — an embedded base64 image/audio blob is not
English-like text, so tiktoken's BPE tokenizes it far denser than real
prose; a single few-hundred-KB screenshot counted this way turned into
hundreds of thousands of "tokens" (measured live: vision-demand route_events
averaging 94k-350k prompt_tokens_compacted for what was an ordinary chat
message plus one image, and 413-rejecting requests that were nowhere near
actually too large). `_strip_multimodal_content` replaces each such part
with a short placeholder before it's ever JSON-dumped for encoding, and
`count_tokens` adds a flat per-item estimate back on top instead —
approximating real provider vision/audio token cost (resolution/duration-
based, not proportional to encoded size) far better than either tokenizing
the raw bytes or ignoring them outright.
"""

from __future__ import annotations

import json
import re
from typing import Any

_TRAILING_WS = re.compile(r"[ \t]+\n")
_BLANK_LINES = re.compile(r"\n{3,}")

# Flat per-item estimates standing in for real provider vision/audio token
# cost — deliberately conservative order-of-magnitude figures (see module
# docstring), not a measured per-provider number. Never proportional to the
# item's actual encoded size.
_IMAGE_TOKEN_ESTIMATE = 1500
_AUDIO_TOKEN_ESTIMATE = 300

_encoding: Any = None
_encoding_failed = False


def _get_encoding() -> Any:
    global _encoding, _encoding_failed
    if _encoding is None and not _encoding_failed:
        try:
            import tiktoken

            _encoding = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _encoding_failed = True
    return _encoding


def _strip_multimodal_content(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """(stripped_messages, extra_token_estimate) — replaces every
    image_url/input_audio content part's payload with a short placeholder
    (see module docstring for why the real payload must never be
    tokenized), returning a flat estimate to add back per item. Messages
    with plain string content (no multimodal parts) pass through
    unchanged."""
    extra_tokens = 0
    stripped: list[dict[str, Any]] = []
    for message in messages:
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            stripped.append(message)
            continue
        new_content = []
        for part in content:
            part_type = part.get("type") if isinstance(part, dict) else None
            if part_type == "image_url":
                new_content.append({"type": "image_url", "image_url": {"url": "<image omitted>"}})
                extra_tokens += _IMAGE_TOKEN_ESTIMATE
            elif part_type == "input_audio":
                new_content.append({"type": "input_audio", "input_audio": {"data": "<audio omitted>"}})
                extra_tokens += _AUDIO_TOKEN_ESTIMATE
            else:
                new_content.append(part)
        stripped.append({**message, "content": new_content})
    return stripped, extra_tokens


def count_tokens(messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> int | None:
    encoding = _get_encoding()
    if encoding is None:
        return None
    stripped, extra_tokens = _strip_multimodal_content(messages)
    text = json.dumps(stripped, ensure_ascii=False)
    if tools:
        text += json.dumps(tools, ensure_ascii=False)
    return len(encoding.encode(text)) + extra_tokens


def _compact_text(text: str) -> str:
    text = _TRAILING_WS.sub("\n", text)
    return _BLANK_LINES.sub("\n\n", text)


def _compact_json_if_valid(text: str) -> str:
    stripped = text.strip()
    if not stripped or stripped[0] not in "{[":
        return text
    try:
        parsed = json.loads(stripped)
    except Exception:
        return text
    return json.dumps(parsed, separators=(",", ":"), ensure_ascii=False)


def truncate_messages(
    messages: list[dict[str, Any]],
    max_tokens: int,
    tools: list[dict[str, Any]] | None = None,
    total_tokens: int | None = None,
) -> tuple[list[dict[str, Any]], int, list[dict[str, Any]], bool]:
    """Lossy safety valve for a runaway conversation history — distinct from
    normalize_messages above (that one never removes content). Drops the
    oldest *turns* (a user message plus everything up to, but not including,
    the next user message — keeps an assistant tool_call paired with its tool
    response, never split) once the estimated token count exceeds max_tokens.
    System messages and the final turn (the request actually being answered)
    are always kept, even if that alone still exceeds max_tokens — there is
    nothing left to safely cut at that point. Returns (messages, turns_dropped,
    dropped_messages, fits) — callers that want to summarize instead of
    discarding outright use dropped_messages as the source text; fits is False
    when the protected content alone still exceeds max_tokens after every
    removable turn is gone (a token-counting failure never reports unfit).

    `total_tokens`: skip the initial count when the caller already has it
    for this exact `messages`/`tools` pair (chat_completions always does) —
    avoids re-tokenizing the same, often very large, prompt twice.
    """
    total = total_tokens if total_tokens is not None else count_tokens(messages, tools)
    if total is None or total <= max_tokens:
        return messages, 0, [], True
    system_messages = [m for m in messages if m.get("role") == "system"]
    rest = [m for m in messages if m.get("role") != "system"]
    turns: list[list[dict[str, Any]]] = []
    for message in rest:
        if message.get("role") == "user" or not turns:
            turns.append([message])
        else:
            turns[-1].append(message)
    dropped_messages: list[dict[str, Any]] = []
    while len(turns) > 1:
        current = system_messages + [m for turn in turns for m in turn]
        if (count_tokens(current, tools) or 0) <= max_tokens:
            break
        dropped_messages.extend(turns[0])
        turns.pop(0)
    result = system_messages + [m for turn in turns for m in turn]
    final_tokens = count_tokens(result, tools)
    fits = final_tokens is None or final_tokens <= max_tokens
    return result, len(dropped_messages), dropped_messages, fits


def normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for message in messages:
        item = dict(message)
        content = item.get("content")
        if isinstance(content, str):
            if item.get("role") == "tool":
                # Tool results are data payloads — pretty-printing is incidental.
                compacted = _compact_json_if_valid(content)
                item["content"] = compacted if compacted != content else _compact_text(content)
            else:
                item["content"] = _compact_text(content)
        elif isinstance(content, list):
            item["content"] = [
                {**part, "text": _compact_text(part["text"])}
                if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str)
                else part
                for part in content
            ]
        normalized.append(item)
    return normalized
