"""
Generic helpers for writing notification text - deterministic, never
model-generated. The actual sentences for Jobs and interactions live with the
code that knows about them (app/orchestration/notification_events.py); these
are the building blocks they share.

Rules the text built from these follows:

- Addressed ONLY as "Sir" or "Boss" (never the user's name). Which of the two
  is chosen deterministically from the event key, so the same event always
  reads the same way.
- Short, and fine both read on a screen and heard aloud: whole sentences, no
  symbols, no markup, no identifiers.
- Untrusted text (goals, prompts, tool output) only ever gets in through
  `clean_text`: one line, no control/format characters, bounded.
"""

from __future__ import annotations

import unicodedata
import zlib

_ADDRESSES = ("Sir", "Boss")


def address_for(key: str) -> str:
    """"Sir" or "Boss", fixed for a given event key."""
    return _ADDRESSES[zlib.crc32(key.encode("utf-8")) % len(_ADDRESSES)]


def clean_text(text: str | None, limit: int) -> str:
    """Single line, no control/format characters, bounded."""
    cleaned = "".join(
        " " if unicodedata.category(ch) in ("Cc", "Cf", "Zl", "Zp") else ch for ch in (text or "")
    )
    collapsed = " ".join(cleaned.split())
    return collapsed if len(collapsed) <= limit else collapsed[: limit - 3].rstrip() + "..."


def sentence(text: str) -> str:
    """`text` ending in sentence punctuation (so more can follow it)."""
    return text if text[-1:] in ".?!" else text + "."


def humanize_remaining(seconds: float) -> str:
    """"in about 4 hours", "tomorrow", ... for a time still to come."""
    seconds = max(0.0, seconds)
    minutes = seconds / 60
    if minutes < 2:
        return "in about a minute"
    if minutes < 50:
        return f"in about {round(minutes)} minutes"
    hours = seconds / 3600
    if hours < 1.5:
        return "in about an hour"
    if hours < 20:
        return f"in about {round(hours)} hours"
    if hours < 36:
        return "tomorrow"
    return f"in about {round(hours / 24)} days"
