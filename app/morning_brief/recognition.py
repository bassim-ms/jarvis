"""
Recognising a request for the Morning Brief - deterministic, bounded, whole-message.

The message must BE the request ("morning brief", "give me my morning brief", "what do I need to know
today?", "catch me up for today"); a longer sentence that merely contains those words is not ("what do I need
to know about Python decorators?", "what is a morning brief?", "search the web for: morning brief"). No model
is asked, no I/O is done, and nothing here decides what to do - it only says whether the words are the request.
"""

from __future__ import annotations

import re

MAX_MESSAGE_CHARS = 120

_LEAD = r"^\s*(?:(?:ok|okay|hey|hi|so|well|jarvis|good\s+morning|morning)[,.!\s]+)*"
_POLITE = r"(?:(?:can|could|would|will)\s+you\s+)?(?:please\s+)?"
_TAIL = r"(?:\s+(?:please|now|for\s+today|for\s+this\s+morning|today|this\s+morning))*\s*[.!?]*\s*$"

_BRIEF = r"(?:(?:my|the|a)\s+)?(?:(?:daily|morning|day|today'?s)\s+){1,2}(?:brief(?:ing)?)"
_VERB = r"(?:give|show|get|send|read|tell|run|start|prepare|do|let)\s+(?:me\s+)?(?:have\s+)?"

_PATTERNS = tuple(
    re.compile(_LEAD + _POLITE + body + _TAIL, re.IGNORECASE)
    for body in (
        _BRIEF,  # "morning brief", "my morning briefing"
        _VERB + _BRIEF,  # "give me my morning brief"
        r"brief\s+me",  # "brief me", "good morning, brief me"
        r"what\s+do\s+i\s+need\s+to\s+know",
        r"what\s+should\s+i\s+know",
        r"(?:is\s+there\s+)?anything\s+i\s+(?:need|should)\s+to\s+know",
        r"catch\s+me\s+up",
    )
)

#: "what do I need to know" / "catch me up" alone are far too general: they only count with a day qualifier.
_DAY_QUALIFIER = re.compile(r"\b(?:today|this\s+morning|for\s+the\s+day)\b", re.IGNORECASE)
_BRIEF_WORD = re.compile(r"\bbrief(?:ing)?\b", re.IGNORECASE)


def is_morning_brief_request(message: object) -> bool:
    """True only if the whole message is a request for the morning brief. Never raises."""
    if not isinstance(message, str):
        return False
    text = message.strip()
    if not text or len(text) > MAX_MESSAGE_CHARS or "\n" in text:
        return False
    if not any(p.match(text) for p in _PATTERNS):
        return False
    if _BRIEF_WORD.search(text):
        return True
    # A general "what do I need to know / catch me up" needs an explicit "today" (or this morning).
    return bool(_DAY_QUALIFIER.search(text))
