from __future__ import annotations

import re
from typing import Any

from app.ranking import dynamic_score, intelligence_score
from app.registry import ProviderModel

# Demand classes, from cheapest to deepest. Each routes to a chain of models:
# a custom chain stored in ai_router.demand_routes, or a rank-derived default.
DEMANDS = ("simple", "standard", "complex", "reasoning", "vision", "audio", "code")

DEMAND_INFO: dict[str, str] = {
    "simple": "Short utility jobs: titles, compression, quick extraction, yes/no checks.",
    "standard": "Day-to-day chat, summaries and tool calls.",
    "complex": "Coding, long context and multi-step work.",
    "reasoning": "Deep analysis and step-by-step thinking.",
    "vision": "Requests with images — routed only through vision-capable models (catalog).",
    "audio": "Audio work (TTS tag insertion, transcription post-processing) — routed only through audio-capable models (catalog).",
    "code": "Code generation and editing — routed only through code-capable models (catalog).",
}

VIRTUAL_PREFIX = "forgerouter/"
VIRTUAL_MODELS = [VIRTUAL_PREFIX + "auto", *(VIRTUAL_PREFIX + demand for demand in DEMANDS)]

REASONING_HINTS = (
    "step by step",
    "passo a passo",
    "think carefully",
    "raciocine",
    "chain of thought",
    "prove",
    "demonstre",
    "deep analysis",
    "análise profunda",
    "analise profundamente",
)

# Hints match whole words only: "prove" must not fire inside "aprove"/"provedor".
_REASONING_HINTS_RE = re.compile(r"\b(?:" + "|".join(re.escape(hint) for hint in REASONING_HINTS) + r")\b")

CODE_HINTS = (
    "refactor",
    "refatore",
    "refatorar",
    "implemente",
    "implementar",
    "escreva uma função",
    "write a function",
    "corrija o bug",
    "fix the bug",
    "stack trace",
    "traceback",
    "code review",
    "pull request",
    "teste unitário",
    "unit test",
)
_CODE_HINTS_RE = re.compile(r"\b(?:" + "|".join(re.escape(hint) for hint in CODE_HINTS) + r")\b")
_FILE_EXT_RE = re.compile(r"\b[\w./-]+\.(?:py|ts|tsx|js|jsx|java|go|rs|c|cpp|h|cs|rb|php|sql|sh|css|html|json|ya?ml|toml)\b")

# Prompt-size thresholds (chars, ~4 chars per token): keep small jobs on small models
# so the free-tier quotas of the big models survive the day.
SIMPLE_MAX_CHARS = 1_500
STANDARD_MAX_CHARS = 8_000
# Conversation history counts at a discount: task difficulty lives in the last user
# message, and weighing the full transcript made every follow-up in a long
# conversation drift to "complex". Discounted history still escalates truly long
# contexts (which need a long-context model) — just HISTORY_DISCOUNT times later.
HISTORY_DISCOUNT = 4


# Some clients (e.g. Hermes agents with the Hindsight memory plugin) inject recalled
# memory/context directly into the user message's own text rather than a separate
# system message. That block reflects the agent's persistent memory, not what the
# human actually typed — for a software-engineering ecosystem it is near-guaranteed
# to mention code, file paths or extensions, which made demand classification return
# "code" for every single message regardless of content ("Oi", "Sim", ...), sending
# ordinary chat to a fill-in-the-middle code model. Stripped before classification.
_MEMORY_CONTEXT_RE = re.compile(r"<memory-context>.*?</memory-context>", re.DOTALL | re.IGNORECASE)

# Hermes / agent context compression handoff and preserved task list prefixes
_TASK_LIST_PRESERVED_RE = re.compile(r"\[Your active task list was preserved across context compression\].*?(?=\n\n\S|\Z)", re.DOTALL | re.IGNORECASE)
_CONTEXT_COMPACTION_BLOCK_RE = re.compile(r"\[CONTEXT COMPACTION — REFERENCE ONLY\].*?(?=\n\n\S|\Z)", re.DOTALL | re.IGNORECASE)

# A second, untagged injection source: each Hermes profile's `pre_llm_call` hook
# (knowledge_cycle.py, on every profile — not just Hindsight/Athos) appends its own
# Foundation/Knowledge-Base excerpt to the user message with no wrapper at all, only
# a fixed literal header, and always as the last thing in the message (turn_context.py
# appends memory-context, then this hook's output, in that order). No tag to match —
# truncate from the literal marker to the end of the string instead. Root-caused
# 2026-07-23 on Athos: the memory-context strip alone didn't stop demand="code",
# because this second block (which quotes file paths/skill docs) was still intact.
_FOUNDATION_CONTEXT_MARKER = "[FOUNDATION / KNOWLEDGE BASE — CONTEXTO OPERACIONAL AUTOMÁTICO]"


def _strip_injected_context(text: str) -> str:
    text = _MEMORY_CONTEXT_RE.sub("", text)
    text = _TASK_LIST_PRESERVED_RE.sub("", text)
    text = _CONTEXT_COMPACTION_BLOCK_RE.sub("", text)
    marker_index = text.find(_FOUNDATION_CONTEXT_MARKER)
    if marker_index != -1:
        text = text[:marker_index]
    return text


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return _strip_injected_context(content)
    if isinstance(content, list):
        return _strip_injected_context(
            " ".join(
                str(part.get("text", ""))
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            )
        )
    return ""


def _content_has_image(content: Any) -> bool:
    if not isinstance(content, list):
        return False
    return any(
        isinstance(part, dict) and part.get("type") in ("image_url", "input_image", "image")
        for part in content
    )


def messages_have_images(messages: list[Any]) -> bool:
    for message in messages:
        content = getattr(message, "content", None) if not isinstance(message, dict) else message.get("content")
        if _content_has_image(content):
            return True
    return False


def _content_has_audio(content: Any) -> bool:
    if not isinstance(content, list):
        return False
    return any(
        isinstance(part, dict) and part.get("type") in ("input_audio", "audio", "audio_url")
        for part in content
    )


def messages_have_audio(messages: list[Any]) -> bool:
    # A non-audio-capable model cannot read an input_audio block at all — same
    # hard-requirement treatment as messages_have_images, just for models that
    # take audio as native chat input (not the separate Whisper/TTS-style REST
    # API OpenAI exposes, which this router doesn't implement).
    for message in messages:
        content = getattr(message, "content", None) if not isinstance(message, dict) else message.get("content")
        if _content_has_audio(content):
            return True
    return False


# How far into the last user message the code-hint/file-extension scan looks.
# Clients that inject extra context (not just the known <memory-context> block —
# e.g. a Hermes plugin's pre_llm_call "user_message" target, appended raw with no
# wrapper tag at all) always append it *after* the human's own text
# (turn_context.py: compose_user_api_content does `content + "\n\n" + injections`).
# A software-engineering ecosystem's injected boilerplate is near-guaranteed to
# contain a file extension or code-ish word somewhere in a multi-KB dump, which
# made an unrelated one-word reply ("teste") classify as "code" — same failure
# class as the <memory-context> incident, just via an injection with no tag to
# strip. Bounding the scan to what a human plausibly typed avoids needing to know
# every injection format a client might use.
CODE_HINT_SCAN_CHARS = 2_000


def _looks_like_code(text: str, lowered: str) -> bool:
    text = text[:CODE_HINT_SCAN_CHARS]
    lowered = lowered[:CODE_HINT_SCAN_CHARS]
    if "```" in text:
        return True
    return bool(_CODE_HINTS_RE.search(lowered) or _FILE_EXT_RE.search(lowered))


def classify_request(messages: list[Any], has_tools: bool) -> str:
    """Classify a chat request into a demand class (forgerouter/auto behaviour)."""
    if messages_have_images(messages):
        return "vision"
    if messages_have_audio(messages):
        return "audio"
    total_chars = 0
    last_user = ""
    for message in messages:
        role = getattr(message, "role", None) or (message.get("role") if isinstance(message, dict) else "")
        content = getattr(message, "content", None) if not isinstance(message, dict) else message.get("content")
        text = _message_text(content)
        total_chars += len(text)
        if role == "user" and text:
            last_user = text
    lowered = last_user.lower()
    effective_chars = len(last_user) + (total_chars - len(last_user)) // HISTORY_DISCOUNT
    if _looks_like_code(last_user, lowered):
        return "code"
    if _REASONING_HINTS_RE.search(lowered):
        return "reasoning"
    if has_tools:
        return "standard" if effective_chars < STANDARD_MAX_CHARS else "complex"
    if effective_chars < SIMPLE_MAX_CHARS:
        return "simple"
    if effective_chars < STANDARD_MAX_CHARS:
        return "standard"
    return "complex"


def default_chain(
    models: list[ProviderModel],
    demand: str,
    performance: dict[str, Any] | None = None,
) -> list[ProviderModel]:
    """Rank-derived chain for a demand when no custom chain is configured.

    simple: best of the small models (economy first); standard: mid band;
    complex: high rank; reasoning: reasoning-capable, highest rank first.
    Band membership uses the static score (a big model stays a "complex" model
    even while misbehaving); the order *within* the chain uses dynamic_score,
    so recently failing/slow models sink without changing class.

    Pure demand/rank ordering — context-window fit and the Hermes-minimum
    floor are enforced centrally by app.context_policy.partition_candidates()
    downstream of this ordering, not duplicated here.
    """
    by_score_desc = sorted(models, key=lambda model: -dynamic_score(model.id, performance))
    if demand in ("vision", "audio", "code"):
        # Vision/audio/code are catalog particularities: only models with the capability may serve them.
        return [model for model in by_score_desc if demand in model.capabilities]
    if demand == "reasoning":
        band = [model for model in by_score_desc if "reasoning" in model.capabilities]
        return band or by_score_desc[:3]
    if demand == "complex":
        band = [model for model in by_score_desc if intelligence_score(model.id) >= 50]
        return band or by_score_desc[:3]
    if demand == "standard":
        band = [model for model in by_score_desc if 30 <= intelligence_score(model.id) < 50]
        return band or by_score_desc
    small = [model for model in by_score_desc if intelligence_score(model.id) < 30]
    return small or list(reversed(by_score_desc))[:3]


def resolve_demand(requested_model: str, messages: list[Any], has_tools: bool) -> str | None:
    """Demand class for the request, or None when a concrete model id was requested."""
    name = (requested_model or "auto").strip()
    if name in ("auto", VIRTUAL_PREFIX + "auto"):
        return classify_request(messages, has_tools)
    if name.startswith(VIRTUAL_PREFIX):
        suffix = name[len(VIRTUAL_PREFIX):]
        if suffix in DEMANDS:
            return suffix
        return classify_request(messages, has_tools)
    return None
