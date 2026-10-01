"""
Deterministic recognition of the user's questions about their OWN memory, and of an explicit request to forget something.

The read / forget counterpart of extractor.py: it only reads the USER'S sentence with fixed patterns (no AI, nothing from a page, a
tool or a model), and only the shapes below; anything else returns None and is ordinary conversation.

  recall   "What did I ask you to note down?"   "What preferences do you remember about me?"   "What are my preferences?"
           "Did I ever tell you that I prefer X?"   "How old am I?"   "When is my birthday?"
  forget   "Forget that preference."   "Forget the preference about X."   "Forget that I prefer X."   "Forget my birthday."

A forget request must name what it is about with a memory word (preference / note / fact / memory, or "that I prefer ..."): a bare
"forget it" or "forget the laptop, find me a printer" is the existing cancel-a-waiting-task language and is never read here.
Bulk wipes ("forget everything") are recognised only so they can be REFUSED - one sentence never clears a memory.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_I = re.IGNORECASE


def _rx(pattern: str) -> re.Pattern:
    return re.compile(pattern, _I)


_LEAD = r"^\s*(?:(?:please|ok|okay|hey\s+jarvis|jarvis|so|and|well)[,.!\s]+)*"
_END = r"\s*[.!?]*\s*$"

MAX_MESSAGE_CHARS = 300


@dataclass(frozen=True)
class RecallQuery:
    kind: str  # everything | preferences | did_i | age | birthday
    topic: str = ""  # did_i: what the user asks whether they said


@dataclass(frozen=True)
class ForgetRequest:
    what: str  # preference | note | fact | birthday | bulk
    topic: str = ""  # what it is about ("" = "that one": the last thing written, or the only one)


_EVERYTHING = [
    _rx(_LEAD + r"what\s+(?:did|have)\s+i\s+(?:ask(?:ed)?\s+you\s+to\s+)?(?:note|jot|write)(?:\s+(?:this|that|it))?\s+down" + _END),
    _rx(_LEAD + r"what\s+(?:did|have)\s+i\s+(?:ask(?:ed)?\s+you\s+to\s+remember|told\s+you(?:\s+to\s+remember)?|noted|told\s+you\s+to\s+note)" + _END),
    _rx(_LEAD + r"what\s+(?:do|did)\s+you\s+(?:remember|know)(?:\s+about\s+me)?" + _END),
    _rx(_LEAD + r"(?:what\s+(?:notes?\s+)?(?:have\s+i|did\s+i)\s+(?:saved|kept)|(?:list|show)\s+(?:me\s+)?(?:all\s+)?my\s+notes)" + _END),
]
_PREFERENCES = [
    _rx(_LEAD + r"what\s+(?:are|is)\s+my\s+(?:\w+\s+)?preferences?" + _END),
    _rx(_LEAD + r"what\s+preferences?\s+(?:do\s+you\s+(?:remember|know|have)|have\s+i\s+(?:told|given)\s+you)(?:\s+about\s+me)?" + _END),
    _rx(_LEAD + r"(?:list|show)\s+(?:me\s+)?my\s+preferences?" + _END),
    _rx(_LEAD + r"what\s+do\s+i\s+prefer" + _END),
]
_DID_I = [
    _rx(_LEAD + r"(?:did|have)\s+i\s+(?:ever\s+)?(?:tell|told|say|said|mention(?:ed)?)(?:\s+to)?\s+you\s+(?:that\s+)?(?P<x>[^?]{3,120}?)" + _END),
    _rx(_LEAD + r"do\s+you\s+(?:remember|know)\s+(?:that\s+)?(?P<x>i\s+(?:prefer|like|love|enjoy)[^?]{2,120}?)" + _END),
]
_AGE = [_rx(_LEAD + r"(?:how\s+old\s+am\s+i|what(?:'s|\s+is)\s+my\s+age|what\s+age\s+am\s+i)" + _END)]
_BIRTHDAY = [
    _rx(_LEAD + r"(?:when\s+is\s+my\s+(?:next\s+)?birthday|how\s+(?:long|many\s+days)\s+(?:until|till|to)\s+my\s+birthday)" + _END),
    _rx(_LEAD + r"(?:what(?:'s|\s+is)\s+my\s+(?:date\s+of\s+birth|birthday|dob)|when\s+was\s+i\s+born)" + _END),
]


def parse_recall(message: object) -> RecallQuery | None:
    if not isinstance(message, str) or not message.strip() or len(message) > MAX_MESSAGE_CHARS:
        return None
    for pattern in _AGE:
        if pattern.match(message):
            return RecallQuery("age")
    for pattern in _BIRTHDAY:
        if pattern.match(message):
            return RecallQuery("birthday")
    for pattern in _PREFERENCES:
        if pattern.match(message):
            return RecallQuery("preferences")
    for pattern in _EVERYTHING:
        if pattern.match(message):
            return RecallQuery("everything")
    for pattern in _DID_I:
        if m := pattern.match(message):
            return RecallQuery("did_i", m.group("x").strip())
    return None


_FORGET_VERB = r"(?:forget|delete|remove|erase)"
_FORGET_KIND = [
    _rx(
        _LEAD + _FORGET_VERB + r"\s+(?:that\s+|about\s+)?(?:the\s+|my\s+|this\s+)?(?P<kind>preference|note|fact|memory)"
        r"(?:\s+(?:about|on|for|regarding|that)\s+(?P<x>[^.!?]+?))?" + _END
    ),
]
_FORGET_THAT_I = [
    _rx(_LEAD + _FORGET_VERB + r"\s+that\s+i\s+(?:prefer|like|love|enjoy)\s+(?P<x>[^.!?]{2,120}?)" + _END),
]
_FORGET_BIRTHDAY = [
    _rx(_LEAD + _FORGET_VERB + r"\s+(?:about\s+)?my\s+(?:date\s+of\s+birth|birthday|birth\s?date|dob|age)" + _END),
]
_FORGET_BULK = [
    _rx(_LEAD + _FORGET_VERB + r"\s+(?:absolutely\s+)?everything(?:\s+you\s+know(?:\s+about\s+me)?|\s+about\s+me)?" + _END),
    _rx(_LEAD + _FORGET_VERB + r"\s+(?:all|every)(?:\s+of)?\s+(?:my\s+|the\s+)?(?:memory|memories|notes|preferences|facts|things)" + _END),
    _rx(_LEAD + r"(?:clear|wipe|reset)\s+(?:all\s+of\s+)?(?:my\s+|your\s+)?(?:whole\s+|entire\s+)?memory" + _END),
]


def parse_forget(message: object) -> ForgetRequest | None:
    if not isinstance(message, str) or not message.strip() or len(message) > MAX_MESSAGE_CHARS:
        return None
    for pattern in _FORGET_BULK:
        if pattern.match(message):
            return ForgetRequest("bulk")
    for pattern in _FORGET_BIRTHDAY:
        if pattern.match(message):
            return ForgetRequest("birthday")
    for pattern in _FORGET_THAT_I:
        if m := pattern.match(message):
            return ForgetRequest("preference", m.group("x").strip())
    for pattern in _FORGET_KIND:
        if m := pattern.match(message):
            kind = m.group("kind").lower()
            return ForgetRequest("fact" if kind == "memory" else kind, (m.group("x") or "").strip())
    return None


_STOPWORDS = frozenset(
    "a an the and or but to of in on at for from with about that this these those it its is are was were be i me my we you your "
    "prefer prefers preferred like likes love enjoy ever told tell said say mention mentioned thing things".split()
)


def topic_words(text: str) -> list[str]:
    """The meaningful lower-case words of `text` (stop-words and "I prefer"-style framing removed)."""
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOPWORDS]
