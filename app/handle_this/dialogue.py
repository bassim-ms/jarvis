"""
Handle This, Slice 5 - the conversation AROUND a clarification.

A clarification is part of the user's existing objective, not a separate
conversation. While a question is open (or has just been answered) a person
does not only "answer": they change their mind about an answer, say never
mind, drift to something else, or come back later. This module holds the
deterministic reading of those moves - it decides nothing about a Job and holds
no state.

    cancel          "cancel", "never mind", "forget it", "stop"    -> the Job is dropped
    new objective   "forget the laptop, find me a printer instead"  -> NOT an answer; the
                                                                        message is left to the
                                                                        normal router
    revision        "actually, make that 3000 AED"                 -> the latest accepted answer
                                                                        becomes the constraint

Everything here is a pure function of the message (and, for a revision, of which
topics the Job already has an accepted answer for): no AI, no I/O, no store.
The words are DATA. Nothing in a message can select a tool, change a permission
or an autonomy level, or widen a boundary - the only things this module can
produce are "cancel this Job", "this is not an answer", and "replace the stored
text of a budget/time constraint with this short line", and each is carried out
by the engine's own version-guarded transition, never by this module.

Deliberately conservative, in the same spirit as the answer gate
(app/handle_this/continuation.py): a cancel is the WHOLE message (a cancel phrase
inside a longer sentence cancels nothing); a revision needs an explicit revising
word AND the shape of the thing being revised (an amount, a time); when unsure,
it is neither, and the message is ordinary conversation.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Iterable
from dataclasses import dataclass
from typing import Any

from app.handle_this.base import MAX_ITEM_CHARS, Topic, clean_line
from app.handle_this.clarification import QUESTIONS
from app.handle_this.service import address_for
from app.handle_this.signals import STOPWORDS, has_time

#: Longest message that can be a revision (an answer is short; anything longer
#: is a sentence about something else).
MAX_REVISION_CHARS = 300

#: The topics whose accepted answer can be revised: the two whose answer has a
#: distinctive shape (an amount, a time). Nothing else is revised by wording.
REVISABLE_TOPICS = (Topic.BUDGET, Topic.WHEN)

_I = re.IGNORECASE


def _rx(pattern: str) -> re.Pattern:
    return re.compile(pattern, _I)


# --- cancellation ---------------------------------------------------------------------------------------------------------------

_CANCEL_LEAD = r"(?:(?:hey|hi|hello|ok|okay|well|so|oh|hmm|um|uh|actually|no|nope|yeah|yes|thanks|thank\s+you|thx|please|pls|sir|boss|jarvis)\b[\s,!.:;-]*)*"
_CANCEL_OBJECT = (
    r"(?:it|that|this|everything|all\s+(?:of\s+)?(?:it|that|this)|the\s+whole\s+thing|"
    r"(?:the|this|that|my)\s+(?:task|request|job|research|search|thing))"
)
_CANCEL_TAIL = r"(?:[\s,]+(?:please|pls|now|for\s+now|thanks|thank\s+you|sir|boss|jarvis|any\s?more))*"
_CANCEL = _rx(
    rf"^\s*{_CANCEL_LEAD}(?:"
    rf"(?:cancel|stop|abort)(?:\s+(?:about\s+)?{_CANCEL_OBJECT})?"
    rf"|(?:abandon|drop|forget|scrap|ditch|skip|kill)\s+(?:about\s+)?{_CANCEL_OBJECT}"
    rf"|never\s*mind(?:\s+(?:about\s+)?{_CANCEL_OBJECT})?"
    rf"|(?:i\s+)?(?:don'?t|do\s+not)\s+(?:want\s+|need\s+)?(?:you\s+to\s+)?(?:do\s+|bother\s+with\s+|worry\s+about\s+)?{_CANCEL_OBJECT}"
    rf"){_CANCEL_TAIL}[\s.!]*$"
)


_CANCEL_VERBS = r"(?:cancel|stop|abort|forget|drop|scrap|skip|ditch)"
_CANCEL_NAMED = _rx(
    rf"^\s*{_CANCEL_LEAD}{_CANCEL_VERBS}\s+(?:about\s+)?(?:the\s+|that\s+|my\s+)?(?P<target>[^.!?]{{2,80}}?)"
    r"(?:\s+(?:one|task|request|research|search|job|please))*[\s.!]*$"
)


def cancel_target(message: object) -> str | None:
    """For "cancel the printer one" - a cancel that NAMES what to cancel - the
    naming words ("printer"); None for anything else (including a bare
    cancel). Only ever used to tell already-waiting tasks apart: it can name
    nothing that is not one of them."""
    if not isinstance(message, str) or not message.strip() or len(message) > 120 or is_cancel(message):
        return None
    found = _CANCEL_NAMED.match(message)
    return found.group("target") if found else None


def is_cancel(message: object) -> bool:
    """Whether `message` is, in its entirety, a request to drop the current
    task. "cancel the meeting" or "stop by the shop" are not."""
    if not isinstance(message, str) or not message.strip() or len(message) > 120:
        return False
    return bool(_CANCEL.match(message))


# --- a new objective (which is NOT an answer) ---------------------------------------------------------------------------------

_REQUEST_VERBS = (
    r"(?:find|get\s+me|research|look\s+(?:up|for|into)|search|show\s+me|recommend|compare|remind\s+me|set\s+up|"
    r"create|build|plan|book|write|analy[sz]e|investigate|suggest|pick|choose|order|buy|help\s+me)"
)
#: "forget the laptop, find me a printer": setting one thing aside and asking for another.
_ABANDON_THEN_REQUEST = _rx(
    rf"\b(?:forget|scrap|ignore|drop|ditch|abandon|skip)\s+(?:about\s+)?(?:the|that|this|my|it|those|these)\b.*?\b{_REQUEST_VERBS}\b"
)
_INSTEAD_REQUEST = _rx(rf"\binstead\b.*?\b{_REQUEST_VERBS}\b|\b{_REQUEST_VERBS}\b.*?\binstead\b")
_CHANGE_OF_SUBJECT = _rx(
    r"\b(?:change\s+of\s+plans?|new\s+(?:request|task|question|topic)|different\s+(?:request|task|question|topic)|"
    r"switch(?:ing)?\s+(?:to|gears)|on\s+another\s+note|separate\s+(?:request|question))\b"
)
#: A message that STARTS with a request ("find me a printer under 500 AED"). Only
#: held against the two topics whose answer is an amount / a time: for a question
#: like "what would you like me to look into?" a request IS the answer.
_STARTS_WITH_REQUEST = _rx(
    rf"^\s*(?:(?:hey|hi|ok|okay|well|so|actually|please|jarvis|can\s+you|could\s+you|would\s+you)\b[\s,!.:;-]*)*{_REQUEST_VERBS}\b(?:\W+\w+){{2,}}"
)


def is_new_objective(message: object, topic: Topic | None = None) -> bool:
    """Whether `message` is a fresh request (or sets the current one aside for
    another) rather than an answer to a question about the current one. When
    true the message is never taken as an answer, so it can never be folded into
    the old Job as a constraint."""
    if not isinstance(message, str) or not message.strip():
        return False
    if _ABANDON_THEN_REQUEST.search(message) or _INSTEAD_REQUEST.search(message) or _CHANGE_OF_SUBJECT.search(message):
        return True
    return topic in (Topic.BUDGET, Topic.WHEN) and bool(_STARTS_WITH_REQUEST.match(message))


# --- revising an accepted answer -----------------------------------------------------------------------------------------------

_REVISION_MARKER = _rx(
    r"\b(?:actually|instead|on\s+second\s+thought|scratch\s+that|correction|make\s+(?:that|it)|"
    r"change\s+(?:that|it|my\s+\w+|the\s+\w+)|update\s+(?:that|it|my\s+\w+|the\s+\w+)|"
    r"(?:raise|lower|bump|increase|decrease)\s+(?:that|it|my\s+\w+|the\s+\w+)|"
    r"(?:i|we)\s+can\s+(?:stretch|push|go)\b|stretch\s+to|let'?s\s+(?:say|make\s+it|go\s+with))\b"
)
_AMOUNT = _rx(
    r"[$€£]\s?\d|\d[\d,.]*\s?(?:k|m|thousand|million|grand|aed|usd|eur|gbp|dollars?|dirhams?|euros?|pounds?|bucks)\b|"
    r"\b(?:aed|usd|eur|gbp)\s?\d"
)
_NO_LIMIT = _rx(r"\b(?:no\s+(?:limit|budget|maximum|cap)|unlimited|any\s+budget|no\s+upper\s+limit)\b")
_DIGIT = re.compile(r"\d")

# "no" is filler only when set off ("no, make that 3000"): "no limit" is the answer
_LEAD_FILLER = _rx(
    r"^\s*(?:(?:(?:actually|well|oh|hmm|ok|okay|sorry|wait|hold\s+on|so|on\s+second\s+thought|scratch\s+that|correction)\b"
    r"|no(?=[,.!:;]))[\s,:;.!-]*)+"
)
_LEAD_PHRASE = _rx(
    r"^\s*(?:(?:make|change|update|set|raise|lower|bump|increase|decrease)\s+(?:that|it|my\s+\w+|the\s+\w+)(?:\s+(?:to|as|be))?\s*"
    r"|(?:i|we)\s+can\s+(?:stretch|push|go)(?:\s+up)?(?:\s+to)?\s*"
    r"|stretch\s+to\s*|let'?s\s+(?:say|make\s+it|go\s+with)\s*)"
)


@dataclass(frozen=True)
class Revision:
    """`topic` is the topic whose accepted answer is replaced; `text` the new
    answer as a short single line (the message with its revising words taken
    off the front)."""

    topic: Topic
    text: str


def _strip_revising_words(message: str) -> str:
    text = _LEAD_FILLER.sub("", message.strip(), count=1)
    text = _LEAD_PHRASE.sub("", text, count=1)
    text = _LEAD_FILLER.sub("", text, count=1)
    return clean_line(text.strip(" .!,;:"), MAX_ITEM_CHARS) or clean_line(message, MAX_ITEM_CHARS)


def detect_revision(message: object, accepted: Collection[Topic]) -> Revision | None:
    """The revision `message` makes to an answer the user already gave, or
    None. `accepted` names the topics that have an accepted answer. Needs an
    explicit revising word AND the shape of the thing revised; a message that
    could revise more than one topic revises none."""
    if not isinstance(message, str):
        return None
    text = message.strip()
    if not text or len(text) > MAX_REVISION_CHARS or "?" in text:
        return None
    if is_cancel(text) or is_new_objective(text) or not _REVISION_MARKER.search(text):
        return None

    timed = has_time(text)
    amountish = bool(_AMOUNT.search(text) or _NO_LIMIT.search(text))
    # a bare number ("make that 3000") is a budget only if no time could be meant
    bare = bool(_DIGIT.search(text)) and Topic.WHEN not in accepted and not timed
    matches: list[Topic] = []
    if Topic.BUDGET in accepted and (amountish or bare):
        matches.append(Topic.BUDGET)
    if Topic.WHEN in accepted and timed:
        matches.append(Topic.WHEN)
    if len(matches) != 1:  # nothing to revise, or it could be either: neither is changed
        return None
    return Revision(matches[0], _strip_revising_words(text))


# --- what the resolver is given ---------------------------------------------------------------------------------------------------

ORIGIN = "handle_this"

_NOISE = frozenset(
    "cancel stop abort forget drop scrap skip ditch task tasks request job research search one thing please never "
    "mind about the that this those these first second last latest".split()
)
_OLDER = frozenset({"first", "older", "oldest", "earlier", "earliest"})
_NEWER = frozenset({"latest", "newer", "newest", "recent", "last"})


def _content_words(text: str) -> set[str]:
    """The words of `text` that could name a task: lower-case, 3+ characters,
    not a stop word or a word of cancelling itself. (The ordinals are kept
    apart, in _OLDER / _NEWER.)"""
    words = set(re.findall(r"[a-z0-9]+", text.lower()))
    return {w for w in words if len(w) >= 3 and w not in (STOPWORDS | _NOISE)} | (words & (_OLDER | _NEWER | {"second"}))


class ClarificationDialogue:
    """The resolver's view of all of the above, over the interactions Handle
    This asked (`detail["origin"] == "handle_this"`). Stateless; every method is
    a pure function of its arguments. The resolver calls it duck-typed, so the
    interaction layer stays free of any dependency on this package."""

    def cancel_target(self, message: str) -> str | None:
        return cancel_target(message)

    def choose_task(self, text: str, jobs: list[Any]) -> Any | None:
        """Which ONE of `jobs` (oldest first) `text` picks out, or None. Read
        against the user's own words for each task: only words that tell the
        tasks apart count ("printer", not "robotics" if both say it), plus
        "the first / older one" and "the latest / newer / second one". No
        unique answer means no answer - nothing is guessed."""
        if not isinstance(text, str) or len(jobs) < 2:
            return None
        ordered = sorted(jobs, key=lambda j: j.created_at)
        words = _content_words(text)
        if words & _OLDER:
            return ordered[0]
        if words & _NEWER or ("second" in words and len(ordered) == 2):
            return ordered[-1]
        each = [_content_words(getattr(j, "goal", "") or "") for j in ordered]
        common = set.intersection(*each)
        scores = [len(words & (mine - common)) for mine in each]
        best = max(scores)
        if best == 0 or scores.count(best) != 1:
            return None
        return ordered[scores.index(best)]

    def which_task_reply(self, jobs: list[Any]) -> str:
        """The short question put to the user: which of the waiting tasks to
        drop, each named by its own request in the user's own (cleaned) words."""
        ordered = sorted(jobs, key=lambda j: j.created_at)[:3]
        labels = " or ".join(f"“{clean_line(getattr(j, 'goal', ''), 60)}”" for j in ordered)
        who = address_for("".join(j.id for j in ordered))
        return f"{who}, which task should I cancel? {labels}"

    def owns(self, interaction: Any) -> bool:
        """Whether Handle This asked this question."""
        return (getattr(interaction, "detail", None) or {}).get("origin") == ORIGIN

    def cancel_requested(self, message: str) -> bool:
        return is_cancel(message)

    def could_revise(self, message: str) -> bool:
        """A cheap first look, so no Job is loaded for ordinary chat: does the
        message carry an explicit revising word at all?"""
        return (
            isinstance(message, str)
            and 0 < len(message.strip()) <= MAX_REVISION_CHARS
            and "?" not in message
            and bool(_REVISION_MARKER.search(message))
        )

    def topic_of(self, interaction: Any) -> str | None:
        topic = (getattr(interaction, "detail", None) or {}).get("topic")
        try:
            return Topic(topic).value
        except ValueError:
            return None

    def accepted_answers(self, interactions: Iterable[Any]) -> dict[str, Any]:
        """Topic value -> the LATEST resolved interaction of that topic whose
        answer was actually usable (not one that had to be asked again)."""
        # imported here: continuation imports this module at load time
        from app.handle_this.continuation import answer_satisfies

        accepted: dict[str, Any] = {}
        for interaction in interactions:
            if not self.owns(interaction) or getattr(interaction.status, "value", None) != "resolved":
                continue
            value = self.topic_of(interaction)
            if value is None:
                continue
            if answer_satisfies(Topic(value), interaction.response or ""):
                accepted[value] = interaction  # later ones replace earlier ones (oldest first)
        return accepted

    def revision(self, message: str, accepted_topics: Collection[str]) -> tuple[str, str] | None:
        """(topic value, new answer text) if `message` revises an accepted
        answer, else None."""
        topics = {Topic(v) for v in accepted_topics if v in {t.value for t in REVISABLE_TOPICS}}
        found = detect_revision(message, topics)
        return None if found is None else (found.topic.value, found.text)

    def choose(self, candidates: list[Any]) -> Any | None:
        """When several of Handle This's questions could take the same answer:
        the most recently put to the user - but only when they ask the SAME
        thing (two budget questions: the newest is the one in front of the user).
        Different questions stay ambiguous, and the user is asked."""
        if len(candidates) < 2 or not all(self.owns(c) for c in candidates):
            return None
        topics = {self.topic_of(c) for c in candidates}
        if len(topics) != 1 or None in topics:
            return None
        stamped = [c for c in candidates if getattr(c, "presented_at", None)]
        if len(stamped) != len(candidates):
            return None
        ordered = sorted(candidates, key=lambda c: c.presented_at)
        if ordered[-1].presented_at == ordered[-2].presented_at:
            return None  # cannot tell which is newer: ask
        return ordered[-1]

    def reminder(self, interaction: Any) -> str:
        """What the user is told when they answer a question that has gone
        stale: the question again, in plain words. The prompt is one of Handle
        This's fixed templates."""
        prompt = (interaction.prompt or "").strip()
        known = {q for q in QUESTIONS.values()}
        question = prompt if prompt in known else ""
        if not question:
            return "It's been a while since I asked you something, so I haven't used that yet. Could you tell me again?"
        return (
            f"It's been a while since I asked this, so I haven't used that answer yet. {question} "
            "Just tell me again and I'll carry on."
        )
