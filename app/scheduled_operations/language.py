"""
The words of a Scheduled Job request, as DATA - deterministic and pure (no clock, no I/O, no model, no storage).

It reads a whole chat message into one of two things, or says "not mine" (None) and leaves the message to the rest of Jarvis:

  CREATE   a WORK instruction together with explicit timing: "Every morning at 8, research AI robotics news and give me a summary",
           "Every Monday at 6 PM, check my upcoming calendar and summarize tomorrow", "Tomorrow at 5 PM, research X and report the
           findings". The timing words are the reminder grammar's (app/reminders/language.py - one set of day/time/repeat words for
           the whole product), read only at the START of the request, or - when the request has none there - only at its END, so
           the instruction's own closing words ("...and summarize tomorrow") are never mistaken for a second timing.
  MANAGE   "what scheduled jobs do I have?", "show / pause / resume / cancel my <topic> scheduled job".

A message is only ever a CREATE request when ALL of these hold - so ordinary chat is never taken for one:
  * the timing says when it HAPPENS: a repetition ("every day", "every Monday", "every 2 hours") or a time of day. A bare day
    ("tomorrow") or a relative time ("in 10 minutes") is not enough (the first is usually just conversation, and a relative start is
    not something V1 schedules);
  * what is left is an instruction: it starts with an action verb ("research", "check", "summarize", "send" ...). The verb list is
    deliberately wider than what V1 can do - "send me X every morning" is recognised so it can be REFUSED aloud, never quietly
    started as an immediate job instead.

Nothing here decides anything about time zones, today's date or what is allowed; it only reads words into fields. The instruction's own
words are DATA: stored and shown as plain text, never parsed as a command.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from app.reminders.language import When, split_leading_when, split_when

MAX_REQUEST_CHARS = 600

_I = re.IGNORECASE


class ManageAction(Enum):
    LIST = "list"
    SHOW = "show"
    PAUSE = "pause"
    RESUME = "resume"
    CANCEL = "cancel"


@dataclass(frozen=True)
class CreateRequest:
    text: str  # the instruction, without its timing words
    when: When
    reads_only: bool = True  # it starts with a READ verb (research, check, summarize ...); False: it asks to DO something (send, add, buy ...)


@dataclass(frozen=True)
class ManageRequest:
    action: ManageAction
    topic: str = ""  # what the user called it ("ai news")
    everything: bool = False  # "all"/"every" - never honoured, only recognised so it can be declined


_LEAD = r"^\s*(?:(?:ok|okay|hey|hi|so|well|jarvis|please)[,.!\s]+)*(?:(?:can|could|would|will)\s+you\s+)?(?:please\s+)?"
_END = r"\s*[.!?]*\s*$"

# ====================================== creating =========================================================================

#: "every morning at 8" is "every day at 8 in the morning" - the part of the day pins AM/PM, so it is never a question.
_DAYPART_AT = re.compile(
    r"\b(?P<every>every|each)\s+(?P<part>morning|afternoon|evening|night)\b[\s,]*(?:at\s+)?(?P<h>\d{1,2})(?::(?P<mi>\d{2}))?"
    r"(?!\s*(?:am|pm|a\.m\.|p\.m\.))(?![\d:])",
    _I,
)
#: "every evening at 7pm" / "every morning" (no time): the part of the day adds nothing the clock does not already say.
_DAYPART = re.compile(r"\b(?P<every>every|each)\s+(?:morning|afternoon|evening|night)\b", _I)

#: What an instruction starts with. The READ verbs are what V1 can do. The ACT verbs are recognised too, on purpose: "send me X every
#: morning" or "add a task every day" must be REFUSED aloud (the family-level capability check cannot tell "list my tasks" from "add a
#: task"), never proposed and never quietly started as an immediate job instead.
_READ_VERBS = (
    r"research|investigate|look\s+(?:up|into|for|at)|find|search|check|review|summari[sz]e|analy[sz]e|compare|compile|gather|collect|"
    r"list|go\s+through|give\s+me|tell\s+me|let\s+me\s+know|brief\s+me|update\s+me|report|prepare|draft|monitor|watch|track|"
    r"read|scan|pull|write|keep\s+(?:an\s+eye|track)|show\s+me"
)
_ACT_VERBS = (
    r"send|email|e-mail|message|text|call|notify|post|tweet|buy|order|purchase|book|pay|open|launch|click|type|press|turn|switch|"
    r"print|delete|remove|install|run|start|stop|move|copy|save|create|add|set|update|upload|download"
)
#: An instruction that STARTS with a read verb can still ask for a side effect further on ("research X and email me the result", "find my
#: files and delete the old ones"). The profile would stop the side effect at run time, but the user would have approved something that can
#: never fully happen - so the clearly side-effecting verbs, after a conjunction, make it a request to DO something. (A bounded list: the
#: READ-only profile, not this grammar, is what enforces V1's scope; this only makes the refusal honest and early.)
_SIDE_EFFECT = re.compile(
    r"(?:\b(?:and|then|also)|[,;])\s+(?:(?:please|also|then)\s+)*(?:send|email|e-mail|message\s+me|text\s+me|call|notify|post|tweet|buy|order|"
    r"purchase|book|pay|delete|remove|install|save|upload|download|print|open|click|type|press|launch|turn\s+(?:on|off)|switch\s+(?:on|off)|"
    r"add\b[^.;]{0,40}\b(?:to|on)\s+my|create\s+(?:a|an)\s+(?:task|note|reminder|event)|set\s+(?:a|an)\s+(?:reminder|alarm|timer))\b",
    _I,
)
_LEADING = r"^(?:(?:please|to)\s+)*"
_READ_INSTRUCTION = re.compile(rf"{_LEADING}(?:{_READ_VERBS})\b", _I)
_INSTRUCTION = re.compile(rf"{_LEADING}(?:{_READ_VERBS}|{_ACT_VERBS})\b", _I)


def _normalise(rest: str) -> str:
    """Read 'every morning at 8' as 'every day at 8 in the morning', and 'every evening' as 'every day', wherever it stands (start or end)."""

    def with_time(m: re.Match) -> str:
        minutes = f":{m.group('mi')}" if m.group("mi") else ""
        return f"{m.group('every')} day at {m.group('h')}{minutes} in the {m.group('part').lower()}"

    changed = _DAYPART_AT.sub(with_time, rest, count=1)
    return changed if changed != rest else _DAYPART.sub(lambda m: f"{m.group('every')} day", rest, count=1)


def _schedules(when: When) -> bool:
    """The timing says when something HAPPENS (see the module docstring)."""
    if when.relative_minutes is not None or when.relative_days is not None:
        return False
    return bool(when.repeat or when.every or when.clock)


def parse_create(message: str) -> CreateRequest | None:
    lead = re.match(_LEAD, message, _I)
    rest = message[lead.end():] if lead else message
    rest = _normalise(rest)
    leading, text = split_leading_when(rest)
    if leading.empty:
        leading, text = split_when(rest)  # the timing at the END ("research X every day at 7")
    text = " ".join(text.split())
    if not text or not _schedules(leading) or not _INSTRUCTION.match(text):
        return None
    return CreateRequest(text=text, when=leading, reads_only=bool(_READ_INSTRUCTION.match(text)) and not _SIDE_EFFECT.search(text))


# ====================================== managing =========================================================================

_NOUN = r"scheduled\s+(?:jobs?|tasks?)"
_NOUN_RX = re.compile(rf"\b{_NOUN}\b", _I)
_LIST = re.compile(
    _LEAD
    + rf"(?:(?:what|which)\s+{_NOUN}\s+do\s+i\s+have(?:\s+(?:set|now|right\s+now|coming\s+up))?|"
    rf"what\s+(?:are|is)\s+my\s+(?:active\s+|upcoming\s+)?{_NOUN}|what'?s\s+my\s+(?:active\s+|upcoming\s+)?{_NOUN}|"
    rf"(?:show|list|tell)(?:\s+me)?\s+(?:all\s+)?(?:of\s+)?(?:my\s+)?(?:active\s+|upcoming\s+)?{_NOUN}|"
    rf"do\s+i\s+have\s+any\s+(?:active\s+|upcoming\s+)?{_NOUN}(?:\s+set)?|"
    rf"my\s+(?:active\s+|upcoming\s+)?{_NOUN})" + _END,
    _I,
)
_VERB_ACTIONS = {
    "pause": ManageAction.PAUSE, "suspend": ManageAction.PAUSE, "hold": ManageAction.PAUSE,
    "resume": ManageAction.RESUME, "unpause": ManageAction.RESUME, "restart": ManageAction.RESUME,
    "cancel": ManageAction.CANCEL, "delete": ManageAction.CANCEL, "remove": ManageAction.CANCEL, "stop": ManageAction.CANCEL,
    "show": ManageAction.SHOW, "inspect": ManageAction.SHOW, "describe": ManageAction.SHOW,
}
_MANAGE = re.compile(
    _LEAD + r"(?P<verb>" + "|".join(_VERB_ACTIONS) + r")\s+(?:(?:my|the|that|this|a|an)\s+)?(?P<body>.+?)" + _END, _I
)
_TOPIC_NOISE = frozenset("my the a an that this about for to of on regarding please scheduled job jobs task tasks called named titled one ones".split())


def topic_words(topic: str) -> tuple[str, ...]:
    return tuple(w for w in re.findall(r"[a-z0-9']+", topic.lower()) if w not in _TOPIC_NOISE)


def _manage(match: re.Match) -> ManageRequest | None:
    body = match.group("body").strip()
    noun = _NOUN_RX.search(body)
    if noun is None:
        return None  # "cancel the meeting", "pause my music": not about scheduled jobs at all
    before, after = body[: noun.start()].strip(), body[noun.end():].strip()
    topic = (before if before else after).strip()
    everything = bool(re.search(r"\b(?:all|every|everything|each)\b", before, _I))
    action = _VERB_ACTIONS[match.group("verb").lower()]
    plural = noun.group(0).lower().rstrip().endswith("s")
    if action is ManageAction.SHOW and plural and not topic:
        return ManageRequest(ManageAction.LIST)
    cleaned = "" if everything else " ".join(w for w in topic.split() if w.lower() not in ("all", "every", "of", "them", "my", "the"))
    return ManageRequest(action, topic=cleaned, everything=everything or (plural and not cleaned))


def parse_request(message: object) -> CreateRequest | ManageRequest | None:
    """The Scheduled Job request a message is, in its entirety - or None (ordinary conversation, left to the rest of Jarvis)."""
    if not isinstance(message, str) or not message.strip() or len(message) > MAX_REQUEST_CHARS:
        return None
    if _LIST.match(message):
        return ManageRequest(ManageAction.LIST)
    manage = _MANAGE.match(message)
    managed = _manage(manage) if manage else None
    return managed if managed is not None else parse_create(message)
