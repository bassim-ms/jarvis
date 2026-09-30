"""
Handle This continuation: what happens to a parked objective once the user has
ANSWERED the question it was waiting on.

When Handle This needs something material it does not just refuse or guess: the
request becomes an ordinary Job that WAITS (status AWAITING_INPUT) with one
INPUT interaction attached (app/orchestration/interactions.py), carrying the
structured objective Handle This had already understood. The answer arrives
through the existing interaction resolver, which resolves the interaction and
moves the SAME Job back to PENDING in one atomic transaction. From there the
engine calls `ObjectiveContinuer.continue_objective` once, BEFORE planning, to
answer one question: with this answer folded in, is the ORIGINAL objective now
ready, or is something else still materially missing?

    ready  -> the Job goes on to the existing planner exactly as any Job does,
              carrying the updated objective (and the answer, which the engine
              also hands the planner as a clarification).
    ask    -> the same Job waits again on ONE more question (bounded by the
              per-Job interaction limits, enforced atomically by the engine).
    fail   -> the Job fails safely (an unusable objective, or a fault here).

Guarantees:

- The original request is authoritative. The answer is folded in as a
  CONSTRAINT on that objective (a budget, a purpose, a time) - it never becomes
  the objective, and it is never re-interpreted as a new request.
- Deterministic and side-effect free: no AI call, no tool, no store, no Job.
  Everything it needs is passed in as plain data, and it returns plain data, so
  a restart between the answer and the continuation loses nothing.
- The answer is untrusted text. It is kept as a short single-line constraint,
  checked only for whether it plausibly answers the question that was asked
  (a budget needs an amount or an explicit "no limit"), and otherwise the same
  question is asked again - within the same bounded number of questions. An
  answer can never select a tool, change a permission or widen the boundary:
  the objective it lands in carries no such field.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from app.handle_this.base import (
    MAX_ITEM_CHARS,
    MAX_LIST_ITEMS,
    Confidence,
    Constraint,
    ConstraintSource,
    GoalFamily,
    Topic,
    clean_line,
)
from app.handle_this.clarification import QUESTIONS, apply_policy
from app.handle_this.dialogue import REVISABLE_TOPICS, is_cancel, is_new_objective
from app.handle_this.interpretation import Interpretation
from app.handle_this.signals import STOPWORDS, has_subject, has_time

_NO_LIMIT = re.compile(
    r"\b(?:no\s+(?:limit|budget|maximum|cap)|unlimited|flexible|doesn'?t\s+matter|does\s+not\s+matter|any\s+budget)\b",
    re.IGNORECASE,
)
_HAS_WORD = re.compile(r"[A-Za-z]{2,}")

#: Topics whose answer becomes a constraint on the objective, and which
#: constraint topic it is recorded under.
_ANSWER_AS_CONSTRAINT = {
    Topic.BUDGET: Topic.BUDGET,
    Topic.USE_CASE: Topic.USE_CASE,
    Topic.WHEN: Topic.TIMEFRAME,
}


# --- step 1: is this message plausibly an ANSWER at all? ("arming") ------------------------------------
#
# A pending question must not swallow ordinary conversation: "hey Jarvis",
# "thanks", "what's up" are not attempts to answer "what's your maximum
# budget?", and treating them as one would use up one of the user's (two)
# questions for nothing. So the resolver asks this FIRST, and only a message
# that passes is treated as an answer (and so consumes anything). What
# happens to an answer AFTER that - is it usable? - is `answer_satisfies`
# below, a separate step with a separate outcome (ask again).
#
# Deliberately shallow and conservative. One rule applies to every topic: a
# message made only of greeting / thanks / filler / small talk, or asking to
# change the subject, is conversation. Only the two topics with a distinctive
# shape (an amount, a time) add a signal of their own - there is no parser per
# topic. Unsure means "not an answer": nothing is consumed and the message is
# ordinary chat while the question stays open.

_CONVERSATIONAL = frozenset(
    """hey hi hello hiya yo jarvis thanks thank thx ty ok okay k sure yes yep yeah yup no nope nah cool great
    nice good fine alright right please pls hmm hm um uh well lol haha bye goodbye morning afternoon evening
    night up doing going there again anything something tell help what see know got get too also then now just
    really very oh ah wow mm whatever anyway anyways understood noted""".split()
)
_META_CONVERSATION = re.compile(
    r"\b(?:something\s+else|tell\s+me\s+something|help\s+me\s+with|can\s+you\s+help|what'?s\s+up|"
    r"how\s+are\s+you|never\s*mind|forget\s+it|another\s+question|change\s+the\s+subject|"
    r"talk\s+about|by\s+the\s+way)\b",
    re.IGNORECASE,
)
_WORD = re.compile(r"[a-z0-9']+")

#: An amount, a magnitude word, a currency, or the word budget/afford - the
#: shape of an attempt to state a budget (usable or not).
_BUDGET_SIGNAL = re.compile(
    r"\d|\b(?:hundred|thousand|million|grand|aed|usd|eur|gbp|dollars?|dirhams?|euros?|pounds?|bucks|"
    r"budget|afford|unlimited|flexible|no\s+(?:limit|budget|maximum|cap))\b",
    re.IGNORECASE,
)


def _has_substance(text: str) -> bool:
    """Whether anything is left of `text` once greetings, thanks, filler and
    stop words are set aside (and it is not a request to talk about something
    else)."""
    if _META_CONVERSATION.search(text):
        return False
    return any(w not in _CONVERSATIONAL and w not in STOPWORDS for w in _WORD.findall(text.lower()))


def is_plausible_answer(topic: Topic, text: str) -> bool:
    """Whether `text` plausibly ATTEMPTS to answer the question about `topic`
    (not whether it is a good answer - see `answer_satisfies`). Pure."""
    if not isinstance(text, str) or not text.strip():
        return False
    # Slice 5: a request to drop the task, or a fresh request that sets this one
    # aside, is never an attempt to answer - whatever amount or time it mentions.
    if is_cancel(text) or is_new_objective(text, topic):
        return False
    if topic is Topic.BUDGET:
        return bool(_BUDGET_SIGNAL.search(text))
    if topic is Topic.WHEN:
        return bool(has_time(text) or re.search(r"\d", text))
    return _has_substance(text)  # every other topic: anything but conversation


def plausible_input_answer(interaction, message: str) -> bool:
    """The resolver's answer gate (see InteractionResolver). For a question
    Handle This asked, whether `message` plausibly answers it. A question that
    is not Handle This's has no opinion here (True: the resolver's own rules
    apply, as always); one of ours whose topic cannot be read is treated as
    "no" - when unsure, nothing is consumed."""
    detail = getattr(interaction, "detail", None) or {}
    if detail.get("origin") != "handle_this":
        return True
    try:
        return is_plausible_answer(Topic(detail.get("topic")), message)
    except ValueError:
        return False


def answer_satisfies(topic: Topic, answer: str) -> bool:
    """Whether `answer` plausibly answers the question asked about `topic`.
    Deliberately minimal - it recognises an answer that is unusable for the
    question, it does not judge whether it is good."""
    if not isinstance(answer, str) or not answer.strip():
        return False
    if topic is Topic.BUDGET:
        return bool(re.search(r"\d", answer) or _NO_LIMIT.search(answer))
    if topic is Topic.WHEN:
        return bool(has_time(answer) or re.search(r"\d", answer))
    if topic is Topic.SUBJECT:
        return has_subject(answer)
    return bool(_HAS_WORD.search(answer))  # USE_CASE, INTENT: anything worded


@dataclass(frozen=True)
class Continuation:
    """The verdict. `kind` is "ready", "ask" or "fail". `objective` is the
    updated, schema-conformant objective (ready/ask). `question` and `detail`
    describe the next INPUT to open (ask only)."""

    kind: str
    objective: dict[str, Any] | None = None
    question: str | None = None
    detail: dict[str, Any] | None = None


class ObjectiveContinuer:
    """See the module docstring. Stateless."""

    def revise_objective(self, objective: dict[str, Any], topic: str, answer: str) -> dict[str, Any] | None:
        """The stored objective with the user's REVISED answer for `topic` in
        place of the earlier one (a budget or a time - the only answers with a
        shape that can be checked), or None if this cannot be applied. Nothing
        but the one constraint of that topic changes: the original request, the
        goal, the family and everything else stay as they were, and the answer
        remains a short single-line piece of data. Pure."""
        try:
            kind = Topic(topic)
            if kind not in REVISABLE_TOPICS:
                return None
            line = clean_line(answer, MAX_ITEM_CHARS)
            if not line or not answer_satisfies(kind, line):
                return None
            target = _ANSWER_AS_CONSTRAINT[kind].value
            constraints = [c for c in objective["constraints"] if c["topic"] != target]
            constraints.append({"topic": target, "text": line, "source": ConstraintSource.REQUEST.value})
            if len(constraints) > MAX_LIST_ITEMS:
                return None
            updated = dict(objective)
            updated["constraints"] = constraints
            return updated
        except (KeyError, TypeError, ValueError):
            return None

    def continue_objective(
        self, objective: dict[str, Any], answers: list[dict[str, Any]]
    ) -> Continuation:
        """`objective` is the stored objective (Job.context); `answers` are the
        NEW resolved answers, each {"topic", "answer", "interaction_id"}."""
        try:
            return self._continue(objective, answers)
        except (KeyError, TypeError, ValueError):
            return Continuation("fail")

    def _continue(self, objective: dict[str, Any], answers: list[dict[str, Any]]) -> Continuation:
        original = objective["original_request"]
        constraints = [
            Constraint(Topic(c["topic"]), c["text"], ConstraintSource(c["source"]))
            for c in objective["constraints"]
        ]
        resolved: set[Topic] = set()
        for item in answers:
            topic = Topic(item["topic"])
            answer = item["answer"]
            if not answer_satisfies(topic, answer):
                continue  # unusable for the question: it will be asked again
            if topic in _ANSWER_AS_CONSTRAINT:
                text = clean_line(answer, MAX_ITEM_CHARS)
                if text:
                    # The latest accepted answer is THE constraint for its topic
                    # (a revised answer replaces, never sits beside, the old one).
                    target = _ANSWER_AS_CONSTRAINT[topic]
                    constraints = [c for c in constraints if c.topic is not target]
                    constraints.append(Constraint(target, text, ConstraintSource.REQUEST))
            else:
                resolved.add(topic)

        families = tuple(GoalFamily(v) for v in objective["required_families"])
        interpretation = Interpretation(
            interpreted_goal=objective["interpreted_goal"],
            goal_family=GoalFamily(objective["goal_family"]),
            required_families=families,
            constraints=tuple(constraints),
            success_criteria=tuple(objective["success_criteria"]),
            confidence=Confidence.HIGH,  # an "intent" doubt was the question just answered
            method="deterministic",
        )
        # Context-known constraints are already inside `constraints` (source
        # "context"), so no fresh context lookup is needed or made.
        outcome = apply_policy(original, interpretation, None, frozenset(resolved))

        updated = dict(objective)
        updated["constraints"] = [
            {"topic": c.topic.value, "text": c.text, "source": c.source.value} for c in outcome.constraints
        ]
        updated["assumptions"] = [{"topic": a.topic.value, "text": a.text} for a in outcome.assumptions]

        if outcome.needs:
            topic = outcome.needs[0].topic
            return Continuation(
                "ask",
                objective=updated,
                question=QUESTIONS[topic],
                detail={"origin": "handle_this", "topic": topic.value},
            )
        return Continuation("ready", objective=updated)
