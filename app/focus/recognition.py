"""
Recognising a focus request - deterministic, bounded, whole-message.

Two closed forms only. The message must BE the request; a longer sentence that merely contains these words is
ordinary conversation:

  objective   "I need to finish my Arduino project today."   (a first-person need + a work verb + "my/the/our" + a subject)
  question    "What should I work on right now?"

Nothing else counts: opinions ("my Arduino project is annoying"), past tense ("I finished..."), questions about
concepts ("what does a project deadline mean?"), imperatives, compound requests ("...and then email Sam") and
anything with a clause in the subject are NOT focus requests. No model is asked, no I/O is done, and nothing here
decides what to do - it only says whether the words are the request, and what the user called the subject.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

MAX_MESSAGE_CHARS = 160
MAX_SUBJECT_TOKENS = 5

OBJECTIVE = "objective"
QUESTION = "question"

_LEAD = r"^\s*(?:(?:ok|okay|hey|hi|so|well|jarvis)[,.!\s]+)*"
_TIME_WORD = r"(?:please|today|tonight|now|right\s+now|this\s+(?:morning|afternoon|evening))"
_TAIL = rf"(?:\s+{_TIME_WORD})*\s*[.!?]*\s*$"

_NEED = r"i\s+(?:really\s+|just\s+)?(?:(?:need|have|want)\s+to|must|should)\s+"
_WORK = r"(?:finish|complete|wrap\s+up|work\s+on|continue|get\s+back\s+to|focus\s+on)\s+"
_DETERMINER = r"(?:my|the|our)\s+"
_WORD = r"[a-z0-9][a-z0-9'\-]*"
_SUBJECT = rf"(?P<subject>{_WORD}(?:\s+{_WORD}){{0,{MAX_SUBJECT_TOKENS - 1}}}?)"

_OBJECTIVE = re.compile(_LEAD + _NEED + _WORK + _DETERMINER + _SUBJECT + _TAIL, re.IGNORECASE)

_PICK = r"(?:work\s+on|focus\s+on|do|tackle|start\s+(?:with|on)|prioriti[sz]e)(?:\s+(?:first|next))?"
_QUESTIONS = tuple(
    re.compile(_LEAD + body + _TAIL, re.IGNORECASE)
    for body in (
        rf"what\s+(?:should|shall)\s+i\s+{_PICK}",
        rf"what\s+do\s+i\s+(?:need|have)\s+to\s+{_PICK}",
    )
)

#: Words that turn a "subject" into a clause or a second request ("...report to Sam", "...it and then..."): never a focus subject.
_CLAUSE_WORDS = frozenset(
    "and or but then so because that which if when while until after before to for with about from into than also plus by at in on during within over via using".split()
)

#: Too general to name anything: dropped from a subject, a project name or a task when comparing them. ("project",
#: "work", "stuff" would otherwise tie every project and every task to every request.)
GENERIC_WORDS = frozenset(
    """the and for with from about into this that these those have need want task tasks project projects work working
    thing things stuff finish complete start continue make get doing done today tonight tomorrow morning afternoon evening
    now please its them they some something anything everything all one ones own new old really just much more most
    first next""".split()
)
MIN_TOKEN_CHARS = 3

_TOKEN = re.compile(r"[a-z0-9]+")


def distinctive_tokens(text: object) -> tuple[str, ...]:
    """The lower-case words of `text` that can tell one thing from another (3+ characters, not generic), in order,
    without repeats. Whole words only: "art" is not "start"."""
    if not isinstance(text, str):
        return ()
    seen: dict[str, None] = {}
    for token in _TOKEN.findall(text.casefold()):
        if len(token) >= MIN_TOKEN_CHARS and token not in GENERIC_WORDS:
            seen.setdefault(token)
    return tuple(seen)


@dataclass(frozen=True)
class FocusRequest:
    kind: str  # OBJECTIVE | QUESTION
    subject: str = ""  # what the user called it, without a trailing "project" ("Arduino")
    named_project: bool = False  # the user said "project" ("my Arduino project")
    hint: tuple[str, ...] = ()  # distinctive tokens of the subject


def parse_focus_request(message: object) -> FocusRequest | None:
    """The focus request `message` is, or None. Never raises."""
    if not isinstance(message, str):
        return None
    text = message.strip()
    if not text or len(text) > MAX_MESSAGE_CHARS or "\n" in text or "\r" in text:
        return None
    if any(pattern.match(text) for pattern in _QUESTIONS):
        return FocusRequest(QUESTION)
    match = _OBJECTIVE.match(text)
    if match is None:
        return None
    words = match.group("subject").split()
    if any(word.casefold() in _CLAUSE_WORDS for word in words):
        return None
    named_project = words[-1].casefold() in ("project", "projects")
    if named_project:
        words = words[:-1]
    subject = " ".join(words)
    hint = distinctive_tokens(subject)
    if not hint:  # "my project", "the thing", "it": nothing that says WHICH
        return None
    return FocusRequest(OBJECTIVE, subject, named_project, hint)
