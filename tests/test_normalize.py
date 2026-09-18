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


def test_count_tokens_never_tokenizes_a_base64_image_payload():
    # Regression: an embedded image was previously counted as literal text —
    # tiktoken's BPE tokenizes base64 far denser than prose, so a few-hundred-KB
    # screenshot turned into hundreds of thousands of "tokens" (measured live:
    # vision route_events averaging 94k-350k tokens for an ordinary chat message
    # plus one image, some wrongly 413-rejected). A huge base64 payload must add
    # only the flat per-image estimate, not scale with its size.
    huge_base64 = "A" * 500_000  # ~half a million chars of "image data"
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "what is in this image?"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{huge_base64}"}},
        ],
    }]

    tokens = count_tokens(messages)

    if tokens is None:
        return  # tiktoken encoding unavailable in this environment — degrade gracefully
    assert tokens < 2_000  # the flat image estimate plus a few words of text, nowhere near 500k chars' worth


def test_count_tokens_adds_flat_estimate_per_image_and_audio_part():
    text_only = count_tokens([{"role": "user", "content": [{"type": "text", "text": "hi"}]}])
    with_image = count_tokens([{"role": "user", "content": [
        {"type": "text", "text": "hi"},
        {"type": "image_url", "image_url": {"url": "https://example.com/small.png"}},
    ]}])
    with_audio = count_tokens([{"role": "user", "content": [
        {"type": "text", "text": "hi"},
        {"type": "input_audio", "input_audio": {"data": "YWJj", "format": "wav"}},
    ]}])

    if text_only is None or with_image is None or with_audio is None:
        return  # tiktoken encoding unavailable in this environment — degrade gracefully
    # A flat per-item estimate is added regardless of whether the image is a
    # short external URL or embedded base64 — the real vision/audio token
    # cost is about the item, not proportional to how it's referenced. A
    # small amount on top of the flat estimate is the placeholder's own
    # (short, fixed) JSON encoding — not the exact number, since it isn't
    # what's being verified here.
    assert 1_500 <= with_image - text_only < 1_600
    assert 300 <= with_audio - text_only < 400


def test_count_tokens_leaves_plain_string_content_unaffected():
    # No multimodal parts to strip — plain string content must take the
    # unchanged pass-through path, not the list-stripping one.
    tokens = count_tokens([{"role": "user", "content": "hello"}])

    if tokens is None:
        return  # tiktoken encoding unavailable in this environment — degrade gracefully
    assert tokens > 0


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


def test_truncate_messages_uses_precomputed_total_instead_of_recounting():
    # A caller that already knows the token count (chat_completions always
    # does) can pass it in to skip the redundant initial count — prove it's
    # actually consulted, not silently ignored, by passing a wrong total
    # that claims "already fits" for a payload that would otherwise need
    # truncating.
    messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": _big("only turn")},
    ]
    if count_tokens(messages) is None:
        return

    result, dropped, dropped_messages, fits = truncate_messages(messages, max_tokens=10, total_tokens=5)

    assert result == messages
    assert dropped == 0
    assert fits is True


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
