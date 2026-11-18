"""
Token counts a provider reported for the request just made on THIS thread - handed from an adapter to the AIProviderManager
without changing the AIProvider interface (generate_reply still returns plain text).

An adapter calls `note(...)` after reading a successful response; the manager calls `take()` right after the attempt, on the
same thread, so concurrent requests on other threads never mix their numbers. Only numbers are kept - never text.
"""

from __future__ import annotations

import threading

_local = threading.local()


def _count(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def note(prompt_tokens: object = None, completion_tokens: object = None, total_tokens: object = None) -> None:
    """Record what the provider reported for this thread's current request. Missing or odd values are ignored."""
    prompt, completion, total = _count(prompt_tokens), _count(completion_tokens), _count(total_tokens)
    if total is None and (prompt is not None or completion is not None):
        total = (prompt or 0) + (completion or 0)
    _local.total = total


def note_openai(data: object) -> None:
    """The OpenAI-compatible `usage` object (Groq, OpenRouter, FreeLLMAPI, Cloudflare's newer models)."""
    usage = data.get("usage") if isinstance(data, dict) else None
    if isinstance(usage, dict):
        note(usage.get("prompt_tokens"), usage.get("completion_tokens"), usage.get("total_tokens"))


def note_gemini(data: object) -> None:
    """Gemini's `usageMetadata` object."""
    usage = data.get("usageMetadata") if isinstance(data, dict) else None
    if isinstance(usage, dict):
        note(usage.get("promptTokenCount"), usage.get("candidatesTokenCount"), usage.get("totalTokenCount"))


def take() -> int | None:
    """The total noted on this thread since the last take (None if none), clearing it."""
    total = getattr(_local, "total", None)
    _local.total = None
    return total


def clear() -> None:
    _local.total = None
