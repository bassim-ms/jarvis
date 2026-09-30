"""
Deterministic recognition of the user's OWN words as proposed writes.

Two shapes, both anchored to one sentence of the USER'S message - never to text
from a page, an email, a tool or a model:

  commands     "Remember that I prefer X."   "Set my goal to X."   ("anyway" changes nothing:
               sensitive information is only ever kept after an explicit confirmation)
  statements   "I prefer dark mode."   "My budget is now 2500 AED."   "I am working on LunitiAI."

Conservative on purpose: a question, a conditional, a negation, or anything that
is not plainly a first-person statement of the supported shapes yields nothing (the
message is then just conversation). A hedge ("maybe", "I think") or a time-bound
word ("today", "right now") is carried on the proposal - the policy then stages it
as a candidate or treats it as temporary. No AI is involved, so no model can
choose what a sentence "means" for memory.
"""

from __future__ import annotations

import re

from app.memory_policy.base import MAX_KEY_CHARS, Kind, Origin, WriteProposal

_I = re.IGNORECASE
MAX_PROPOSALS = 3
MAX_PROJECT_NAME_WORDS = 8


def _rx(pattern: str) -> re.Pattern:
    return re.compile(pattern, _I)


_HEDGE = _rx(
    r"\b(?:maybe|perhaps|might|may|probably|possibly|i\s+think|i\s+guess|i\s+suppose|not\s+sure|"
    r"kind\s+of|sort\s+of|i\s+believe|it\s+seems|i\s+feel\s+like|not\s+certain)\b"
)
_TEMPORARY = _rx(
    r"\b(?:today|tonight|tomorrow|right\s+now|at\s+the\s+moment|this\s+(?:morning|afternoon|evening|week|weekend|month)|"
    r"for\s+now|for\s+today|just\s+for\s+this|later\s+on|at\s+present)\b"
)
_CHANGE = _rx(r"\b(?:now|actually|no\s+longer|has\s+changed|have\s+changed|changed\s+to|from\s+now\s+on|instead)\b")
_NEGATION = _rx(r"\b(?:don't|do\s+not|doesn't|does\s+not|never|not|no\s+longer|won't|can't|cannot)\b")
_CONDITIONAL = _rx(r"^\s*(?:if|when|unless|whenever|suppose|imagine|what\s+if)\b")
_LEAD = _rx(
    r"^\s*(?:(?:actually|well|so|ok|okay|also|and|by\s+the\s+way|hey\s+jarvis|jarvis|yes|no|"
    r"i\s+think|i\s+guess|i\s+suppose|i\s+believe|maybe|perhaps|probably)[,.!\s]+)+"
)
_SENTENCES = re.compile(r"(?<=[.!])\s+")

_PREFER = _rx(r"^(?:i|we)\s+(?:really\s+|much\s+|generally\s+|always\s+|now\s+)?prefer\s+(?P<v>[^.!?]{2,120})$")
_BUDGET = _rx(
    r"^my\s+(?:current\s+|maximum\s+|max\s+|monthly\s+|printer\s+)?budget\s+(?:is|=|:)\s*"
    r"(?:now\s+|about\s+|around\s+|up\s+to\s+)?(?P<v>[^.!?]{1,60})$"
)
_PROJECT = _rx(
    r"^(?:i(?:'m|\s+am)\s+(?:currently\s+)?(?:working\s+on|building|developing)|my\s+(?:current\s+)?project\s+is)\s+"
    r"(?P<v>[^.!?]{2,80})$"
)

_CMD_REMEMBER = _rx(
    r"^\s*(?:(?:yes|ok|okay|please|jarvis)[,.!\s]+)*remember(?:\s+(?:that|this|it))?"
    r"(?:\s+anyway)?\s*[:,-]?\s*(?:that\s+)?(?P<clause>[^\n]+?)\s*$"
)
_CMD_GOAL = _rx(r"^\s*(?:please\s+)?(?:set|make)\s+my\s+goal\s+(?:to|as|:)\s*(?P<t>[^\n?]{2,120}?)\s*[.!]?\s*$")
_CMD_GOAL_ADD = _rx(
    r"^\s*(?:please\s+)?add\s+(?:a\s+)?goal\s+to\s+(?:my\s+)?(?P<p>[^:\n?]{2,60}?)(?:\s+project)?\s*[:,-]\s*"
    r"(?P<t>[^\n?]{2,120}?)\s*[.!]?\s*$"
)
_CMD_PROJECT = _rx(
    r"^\s*(?:please\s+)?(?:set|make)\s+my\s+(?:current\s+)?project\s+(?:to|as|:)\s*(?P<v>[^\n?]{2,80}?)\s*[.!]?\s*$"
)


def slug(text: str, max_words: int = 6) -> str:
    words = re.findall(r"[a-z0-9]+", text.lower())[:max_words]
    return "_".join(words)[:MAX_KEY_CHARS]


def _clean(value: str) -> str:
    value = value.strip().strip("\"'`").strip()
    return re.sub(r"[\s.!,;:]+$", "", value)


def _project_name(value: str) -> str:
    name = _clean(value)
    name = re.sub(r"^(?:my|the|a|an|our)\s+", "", name, flags=_I)
    return name


def _flags(sentence: str) -> dict:
    return {
        "hedged": bool(_HEDGE.search(sentence)),
        "temporary": bool(_TEMPORARY.search(sentence)),
        "supersede": bool(_CHANGE.search(sentence)),
    }


def _statement(sentence: str, origin: Origin, evidence: str) -> WriteProposal | None:
    """One sentence -> at most one proposal, from the supported first-person shapes."""
    text = _LEAD.sub("", sentence).strip().rstrip(".!").strip()
    if not text or "?" in text or _CONDITIONAL.match(text) or _NEGATION.search(text):
        return None
    flags = _flags(sentence)
    if origin is Origin.USER_COMMAND:
        flags = {**flags, "supersede": True, "temporary": False}

    if m := _PREFER.match(text):
        value = _clean(m.group("v"))
        if value:
            return WriteProposal(
                Kind.PREFERENCE, "prefers:" + slug(value), value, origin, evidence=evidence, **flags
            )
    if m := _BUDGET.match(text):
        value = _clean(m.group("v"))
        if value and re.search(r"\d", value):
            return WriteProposal(
                Kind.PREFERENCE, "budget", value, origin, category="finance", evidence=evidence, **flags
            )
    if m := _PROJECT.match(text):
        name = _project_name(m.group("v"))
        if name and len(name.split()) <= MAX_PROJECT_NAME_WORDS:
            return WriteProposal(Kind.PROJECT, "project", name, origin, evidence=evidence, **flags)
    return None


def extract_statements(message: str) -> list[WriteProposal]:
    """Proposals from the plain first-person statements in `message`."""
    if not isinstance(message, str) or not message.strip():
        return []
    proposals = []
    for sentence in _SENTENCES.split(message.strip()):
        proposal = _statement(sentence, Origin.USER_STATEMENT, sentence.strip())
        if proposal is not None:
            proposals.append(proposal)
        if len(proposals) >= MAX_PROPOSALS:
            break
    return proposals


def parse_command(message: str) -> list[WriteProposal] | None:
    """The proposals an explicit memory command asks for, or None if `message`
    is not one. "Remember that I prefer X" is read as the statement it contains
    (so it lands in the right place), else it is kept as a note."""
    if not isinstance(message, str) or not message.strip() or "?" in message:
        return None

    if m := _CMD_GOAL_ADD.match(message):
        return [WriteProposal(Kind.GOAL, "goal", _clean(m.group("t")), Origin.USER_COMMAND, evidence=message.strip(), project=_project_name(m.group("p")))]
    if m := _CMD_GOAL.match(message):
        return [WriteProposal(Kind.GOAL, "goal", _clean(m.group("t")), Origin.USER_COMMAND, evidence=message.strip())]
    if m := _CMD_PROJECT.match(message):
        return [WriteProposal(Kind.PROJECT, "project", _project_name(m.group("v")), Origin.USER_COMMAND, evidence=message.strip())]

    m = _CMD_REMEMBER.match(message)
    if not m:
        return None
    clause = m.group("clause").strip()
    if re.match(r"to\b", clause, _I):  # "remember to call mom" is a reminder, not a fact
        return None
    proposals: list[WriteProposal] = []
    for sentence in _SENTENCES.split(clause):
        found = _statement(sentence, Origin.USER_COMMAND, clause)
        if found is None:
            value = _clean(sentence)
            if not value:
                continue
            found = WriteProposal(
                Kind.FACT,
                "note:" + slug(value),
                value,
                Origin.USER_COMMAND,
                evidence=clause,
                hedged=bool(_HEDGE.search(sentence)),
                supersede=True,
            )
        proposals.append(found)
        if len(proposals) >= MAX_PROPOSALS:
            break
    return proposals or None

