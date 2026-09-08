from app.normalize import count_tokens, normalize_messages, truncate_messages


def test_normalize_strips_trailing_whitespace_and_collapses_blank_lines():
    messages = [{"role": "user", "content": "line one   \nline two\t\n\n\n\nline three\n"}]

    normalized = normalize_messages(messages)

    assert normalized[0]["content"] == "line one\nline two\n\nline three\n"


def test_normalize_minifies_json_tool_results():
    messages = [
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": '{\n  "result": "ok",\n  "items": [\n    1,\n    2\n  ]\n}',
        }
    ]

    normalized = normalize_messages(messages)

    assert normalized[0]["content"] == '{"result":"ok","items":[1,2]}'


def test_normalize_leaves_non_json_tool_content_alone_except_whitespace():
    messages = [{"role": "tool", "tool_call_id": "call_1", "content": "plain text   \nwith trailing spaces"}]

    normalized = normalize_messages(messages)

    assert normalized[0]["content"] == "plain text\nwith trailing spaces"


def test_normalize_does_not_minify_json_in_non_tool_messages():
    raw = '{\n  "a": 1\n}'
    messages = [{"role": "user", "content": raw}]

    normalized = normalize_messages(messages)

    # Only whitespace normalization applies to non-tool roles: no trailing
    # whitespace and no 3+ blank line runs here, so the content is unchanged.
    assert normalized[0]["content"] == raw


def test_normalize_handles_list_content_parts():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "hello   \nworld\n\n\n\n!"},
                {"type": "image_url", "image_url": {"url": "http://example.com/x.png"}},
            ],
        }
    ]

    normalized = normalize_messages(messages)

    assert normalized[0]["content"][0]["text"] == "hello\nworld\n\n!"
    assert normalized[0]["content"][1] == {"type": "image_url", "image_url": {"url": "http://example.com/x.png"}}


def test_normalize_preserves_messages_without_string_or_list_content():
    messages = [{"role": "assistant", "content": None, "tool_calls": [{"id": "call_1"}]}]

    normalized = normalize_messages(messages)

    assert normalized == messages


def test_count_tokens_reflects_compaction_savings():
    padded = [{"role": "user", "content": "hello   \nworld\n\n\n\n!"}]
    compact = normalize_messages(padded)

    raw_tokens = count_tokens(padded)
    compact_tokens = count_tokens(compact)

    if raw_tokens is None or compact_tokens is None:
        return  # tiktoken encoding unavailable in this environment — degrade gracefully
    assert compact_tokens < raw_tokens


def _big(label: str) -> str:
    # ~2k tokens of filler per message — big enough that a handful of turns
    # reliably crosses a small test budget without depending on exact tiktoken counts.
    return f"{label} " + ("filler word " * 2000)


def test_truncate_messages_leaves_small_conversations_untouched():
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello!"},
    ]

    result, dropped, dropped_messages, fits = truncate_messages(messages, max_tokens=50_000)

    if count_tokens(messages) is None:
        return  # tiktoken unavailable — degrade gracefully like the rest of this file
    assert result == messages
    assert dropped == 0


def test_truncate_messages_drops_oldest_turns_keeps_system_and_last_turn():
    messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": _big("turn1")},
        {"role": "assistant", "content": _big("turn1-reply")},
        {"role": "user", "content": _big("turn2")},
        {"role": "assistant", "content": _big("turn2-reply")},
        {"role": "user", "content": "final question"},
    ]
    if count_tokens(messages) is None:
        return

    result, dropped, dropped_messages, fits = truncate_messages(messages, max_tokens=3_000)

    assert dropped > 0
    assert dropped_messages
    # System prompt always survives.
    assert result[0] == {"role": "system", "content": "system prompt"}
    # The final turn (what's actually being answered) always survives.
    assert result[-1] == {"role": "user", "content": "final question"}
    assert count_tokens(result) <= count_tokens(messages)
    assert fits is True


def test_truncate_messages_reports_unfit_when_protected_content_alone_overflows():
    # System + final turn alone still exceeds the budget after every
    # removable turn is gone — fits must say so instead of pretending the
    # (still oversized) result is safe to send.
    messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": _big("only turn, way over budget on its own")},
    ]
    if count_tokens(messages) is None:
        return

    result, dropped, dropped_messages, fits = truncate_messages(messages, max_tokens=10)

    assert dropped == 0
    assert result == messages
    assert dropped_messages == []
    assert fits is False


def test_truncate_messages_keeps_tool_response_paired_with_its_assistant_turn():
    messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": _big("turn1")},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1"}]},
        {"role": "tool", "tool_call_id": "call_1", "content": _big("tool result")},
        {"role": "assistant", "content": _big("turn1-reply")},
        {"role": "user", "content": "final question"},
    ]
    if count_tokens(messages) is None:
        return

    result, dropped, dropped_messages, fits = truncate_messages(messages, max_tokens=3_000)

    # Either the whole first turn (user + tool_call + tool response + reply)
    # survives, or it's dropped as one unit — the tool message is never left
    # without its assistant tool_call.
    tool_present = any(m.get("role") == "tool" for m in result)
    tool_call_present = any(m.get("role") == "assistant" and m.get("tool_calls") for m in result)
    assert tool_present == tool_call_present
    assert dropped in (0, 4)
