"""
Reading what the user said to (or about) an Action Proposal - deterministic, bounded, whole-message, no model.

Two small grammars, both closed:

  * a REQUEST for a plan: "make a plan", "make a plan for my Arduino project", "help me plan my Arduino project" - optionally ending in
    "and write it in a note" (the only way a note step is ever offered);
  * a reply while a proposal is WAITING: approve, reject, a not-quite-yes ("ok"), a question ABOUT the plan, "skip step N", an
    unsupported change, or anything else (= unrelated).

A longer sentence that merely contains these words is not one of them. In particular a yes with a "but" ("yes but skip step 2") is
NOT an approval: it matches nothing here, so it is "unrelated" and the pending proposal is dropped rather than guessed at.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.focus.recognition import MAX_SUBJECT_TOKENS

MAX_MESSAGE_CHARS = 120

APPROVE = "approve"
REJECT = "reject"
ACK = "ack"  # "ok", "sure": a nod, not an approval
SKIP = "skip"
EXPLAIN_STEP = "explain_step"
EXPLAIN_AFTER = "explain_after"
EXPLAIN_WHY = "explain_why"
SHOW = "show"
CHANGE = "change"  # a change other than skipping, which is not supported yet
UNRELATED = "unrelated"

LAST = -1  # "the last step"

_LEAD = r"^\s*(?:(?:ok|okay|hey|hi|so|well|jarvis)[,.!\s]+)*"
#: A nod in front of a real approval is fine ("sure, go ahead"); a nod ALONE is not (see _ACK).
_APPROVE_LEAD = r"^\s*(?:(?:ok|okay|sure|alright|hey|hi|so|well|jarvis)[,.!\s]+)*"
_END = r"\s*[.!?]*\s*$"
#: A yes / no / nod is a statement: with a question mark ("yes?") it is not one.
_STATEMENT_END = r"\s*[.!]*\s*$"
_POLITE = r"(?:(?:can|could|would)\s+(?:you|we)\s+|let'?s\s+|please\s+)?"
_ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "last": LAST, "final": LAST}
_STEPREF = r"(?:step\s*(?P<n>\d{1,2})|(?:the\s+)?(?P<ord>first|second|third|fourth|last|final)\s+step)"


def _compile(body: str, *, lead: str = _LEAD, end: str = _END) -> re.Pattern[str]:
    return re.compile(lead + body + end, re.IGNORECASE)


_APPROVE = tuple(
    _compile(body, lead=_APPROVE_LEAD, end=_STATEMENT_END)
    for body in (
        r"(?:(?:yes|yeah|yep|yup)[,\s]+)?(?:go\s+ahead|do\s+it|let'?s\s+do\s+it)",
        r"(?:yes|yeah|yep|yup)(?:[,\s]+please)?",
        r"approved?",
    )
)
_REJECT = tuple(
    _compile(body, end=_STATEMENT_END)
    for body in (
        r"(?:no|nope|nah)(?:\s+thanks|\s+thank\s+you)?",
        r"cancel(?:\s+(?:that|it|the\s+plan))?",
        r"stop",
        r"not\s+now",
        r"never\s*mind",
        r"forget\s+it",
        r"don'?t(?:\s+do\s+(?:that|it))?",
        r"decline[d]?",
        r"reject(?:ed)?",
        r"drop\s+(?:it|that|the\s+plan)",
    )
)
_ACK = _compile(r"(?:ok(?:ay)?|k|kk|sure|alright|all\s+right|fine|sounds\s+good|proceed|send\s+it)", end=_STATEMENT_END)
_SKIP = _compile(_POLITE + r"(?:skip|drop|remove|delete|leave\s+out|cut)\s+" + _STEPREF + r"(?:\s+please)?")
_CHANGE = _compile(
    _POLITE
    + r"(?:change|swap|replace|reorder|move|add|edit|modify|rewrite|insert)\s+"
    + r"(?:" + _STEPREF + r"|(?:an?\s+)?(?:new\s+|extra\s+|another\s+)?steps?|the\s+(?:plan|order|steps))"
    + r"[^?.!]{0,40}"
)
_EXPLAIN_STEP = tuple(
    _compile(body)
    for body in (
        r"(?:(?:can|could)\s+you\s+)?(?:please\s+)?(?:explain|clarify|describe|tell\s+me\s+(?:more\s+)?about)\s+" + _STEPREF,
        r"(?:(?:can|could)\s+you\s+)?(?:please\s+)?explain\s+what\s+" + _STEPREF + r"\s+(?:means|does|involves)",
        r"what\s+(?:does|is|do)\s+" + _STEPREF + r"\s+(?:mean|do|involve|about)",
        r"what(?:'s|\s+is)\s+" + _STEPREF,
    )
)
_EXPLAIN_AFTER = tuple(
    _compile(body)
    for body in (
        r"what\s+happens\s+after\s+" + _STEPREF,
        r"what(?:'s|\s+is)\s+(?:next\s+)?after\s+" + _STEPREF,
    )
)
_EXPLAIN_WHY = tuple(
    _compile(body)
    for body in (
        r"why\s+(?:this|that|the|these)\s+(?:plan|order|steps?)",
        r"why\s+(?:is\s+|are\s+)?" + _STEPREF + r"(?:\s+(?:first|there|here|needed|included|in\s+the\s+plan))?",
    )
)
_SHOW = tuple(
    _compile(body)
    for body in (
        r"(?:please\s+)?(?:show|repeat|read|display|give)\s+(?:me\s+)?(?:the|that|this)\s+(?:plan|steps)(?:\s+again)?",
        r"what(?:'s|\s+is|\s+was)\s+the\s+plan",
        r"(?:say|show)\s+(?:that|it)\s+again",
    )
)


@dataclass(frozen=True)
class FollowUp:
    kind: str
    step: int | None = None  # 1-based, or LAST; None when the reply names no step


def _step_of(match: re.Match[str]) -> int:
    ordinal = match.groupdict().get("ord")
    if ordinal:
        return _ORDINALS[ordinal.lower()]
    return int(match.group("n"))


def classify_follow_up(message: object) -> FollowUp:
    """What `message` is, as a reply to a waiting proposal. Never raises; anything not recognised is UNRELATED."""
    if not isinstance(message, str):
        return FollowUp(UNRELATED)
    text = message.strip()
    if not text or len(text) > MAX_MESSAGE_CHARS or "\n" in text or "\r" in text:
        return FollowUp(UNRELATED)
    if any(p.match(text) for p in _APPROVE):
        return FollowUp(APPROVE)
    if any(p.match(text) for p in _REJECT):
        return FollowUp(REJECT)
    if _ACK.match(text):
        return FollowUp(ACK)
    if (match := _SKIP.match(text)) is not None:
        return FollowUp(SKIP, _step_of(match))
    if (match := _CHANGE.match(text)) is not None:
        return FollowUp(CHANGE, _step_of(match) if (match.groupdict().get("n") or match.groupdict().get("ord")) else None)
    for pattern in _EXPLAIN_STEP:
        if (match := pattern.match(text)) is not None:
            return FollowUp(EXPLAIN_STEP, _step_of(match))
    for pattern in _EXPLAIN_AFTER:
        if (match := pattern.match(text)) is not None:
            return FollowUp(EXPLAIN_AFTER, _step_of(match))
    for pattern in _EXPLAIN_WHY:
        if (match := pattern.match(text)) is not None:
            named = match.groupdict().get("n") or match.groupdict().get("ord")
            return FollowUp(EXPLAIN_WHY, _step_of(match) if named else None)
    if any(p.match(text) for p in _SHOW):
        return FollowUp(SHOW)
    return FollowUp(UNRELATED)


# --- a request for a plan -----------------------------------------------------------------------------------------------------------

_TIME_WORD = r"(?:please|today|tonight|now|right\s+now|this\s+(?:morning|afternoon|evening))"
_TAIL = rf"(?:\s+{_TIME_WORD})*"
_WORD = r"[a-z0-9][a-z0-9'\-]*"
_SUBJECT = rf"(?P<subject>{_WORD}(?:\s+{_WORD}){{0,{MAX_SUBJECT_TOKENS - 1}}}?)"
_MAKE = r"(?:make|give)(?:\s+me)?\s+a\s+plan"

#: The one optional rider on a plan request: "... and write it in a note". Closed: nothing else may follow the plan request. These are the
#: wordings the capability router leaves as ordinary conversation; a "save it ..." / "store it ..." / "keep it ..." request is routed elsewhere
#: (a multi-step goal -> a Job) BEFORE this door sees it, so it is deliberately not part of this grammar.
_IT = r"(?:it|this\s+plan|the\s+plan)"
_NOTE = (
    r"(?P<note>\s+(?:and\s+(?:(?:also|then)\s+)?(?:"
    + rf"write\s+{_IT}\s+(?:down\s+)?(?:in|as)\s+a\s+note|put\s+{_IT}\s+in\s+a\s+note|make\s+a\s+note\s+of\s+(?:it|this|the\s+plan)|note\s+{_IT}\s+down"
    + r")|with\s+a\s+note))?"
)

_PLAN_GLOBAL = re.compile(_LEAD + _MAKE + _TAIL + _NOTE + _TAIL + _END, re.IGNORECASE)
_PLAN_FOR = tuple(
    re.compile(_LEAD + body + _TAIL + _NOTE + _TAIL + _END, re.IGNORECASE)
    for body in (
        _MAKE + r"\s+(?:for|to\s+finish|to\s+complete)\s+(?:my|the|our)\s+" + _SUBJECT,
        r"help\s+me\s+plan\s+(?:my|the|our)\s+" + _SUBJECT,
    )
)


@dataclass(frozen=True)
class PlanRequest:
    subject: str = ""  # what the user called it ("Arduino project"); empty for a plain "make a plan"
    save_note: bool = False  # "... and write it in a note": the user asked for the plan to be kept as a note


def parse_plan_request(message: object) -> PlanRequest | None:
    """The plan request `message` is, or None. Never raises. A message that is not, in its entirety, one of the closed forms is
    ordinary conversation."""
    if not isinstance(message, str):
        return None
    text = message.strip()
    if not text or len(text) > MAX_MESSAGE_CHARS or "\n" in text or "\r" in text:
        return None
    match = _PLAN_GLOBAL.match(text)
    if match is not None:
        return PlanRequest(save_note=bool(match.group("note")))
    for pattern in _PLAN_FOR:
        match = pattern.match(text)
        if match is not None:
            return PlanRequest(subject=match.group("subject").strip(), save_note=bool(match.group("note")))
    return None
