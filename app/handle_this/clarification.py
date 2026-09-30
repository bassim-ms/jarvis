"""
The clarification policy: WHEN is it right to ask the user something?

Jarvis inspects the request, uses the relevant context it retrieved, uses the
constraints it already knows, infers safe defaults where that is reasonable,
and asks ONLY when missing information materially affects the result or
blocks safe continuation. It does not ask merely because some information is
absent.

This is a pure function of (request text, interpretation, context): no I/O and
no AI. Crucially it is also the ONLY place that decides whether a question is
warranted - an AI interpretation can report a low confidence, but it cannot
make Jarvis ask about an irrelevant topic, because what counts as "material"
is decided here, from the request's own wording and from a fixed set of rules:

- SUBJECT    Material if the request names nothing it is about ("research it").
             Asked FIRST and alone: nothing else is worth asking before this.
- INTENT     Material if the interpretation itself is low-confidence.
- BUDGET     Material only for an objective about CHOOSING something to acquire
             (price changes which options are worth researching) and only if
             neither the request nor the context gives one.
- USE_CASE   What the thing is for changes which option is best - but it is
             asked only for a wholly open "pick me something" request, i.e.
             when the budget is ALSO unresolved. Once a budget is known (stated
             or from context) the request has a criterion to research against,
             and asking what it is for as well would be over-asking. Never asked
             if the request or an explicitly named project already says.
- WHEN       Material if the request asks for a reminder or scheduled item and
             gives no time (it cannot be created without one).

Everything else that might be missing is either settled by a safe DEFAULT that
is recorded as a visible Assumption (e.g. "shortlist up to 3 candidates", "use
current information") or is simply not needed. The questions themselves are
fixed, bounded templates - never text taken from the request or a model.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.handle_this.base import (
    MAX_CLARIFICATION_NEEDS,
    MAX_LIST_ITEMS,
    Assumption,
    ClarificationNeed,
    Confidence,
    ConstraintSource,
    Constraint,
    ContextHandoff,
    GoalFamily,
    Topic,
)
from app.handle_this.interpretation import Interpretation
from app.handle_this.signals import has_subject, has_time, is_purchase_oriented, needs_time

#: Fixed (what is missing, why it is needed) per topic.
_NEED_TEXT: dict[Topic, tuple[str, str]] = {
    Topic.SUBJECT: ("What this is about", "I cannot tell what to work on without knowing the subject."),
    Topic.INTENT: ("What outcome you want", "I am not sure what result you are after."),
    Topic.BUDGET: ("Your budget", "Price materially changes which options are worth researching."),
    Topic.USE_CASE: ("What it is for", "What you will use it for changes which option is best."),
    Topic.WHEN: ("When it should happen", "A reminder or calendar item needs a time to be set."),
}


#: The question actually put to the user for each topic (an INPUT
#: interaction's prompt). Fixed, short templates: never text from the request
#: or from a model, so what is asked can only ever be one of these.
QUESTIONS: dict[Topic, str] = {
    Topic.SUBJECT: "What would you like me to look into?",
    Topic.INTENT: "What outcome are you looking for?",
    Topic.BUDGET: "What's your maximum budget?",
    Topic.USE_CASE: "What will you mainly use it for?",
    Topic.WHEN: "When should I set it for?",
}


@dataclass(frozen=True)
class PolicyOutcome:
    """What the policy decided: the constraints in force (stated ones, plus
    ones the context settles for topics the request left open), the safe
    defaults it will proceed on, and the material needs that must be asked
    (empty means nothing must be)."""

    constraints: tuple[Constraint, ...]
    assumptions: tuple[Assumption, ...]
    needs: tuple[ClarificationNeed, ...]


def _need(topic: Topic) -> ClarificationNeed:
    missing, why = _NEED_TEXT[topic]
    return ClarificationNeed(topic=topic, missing=missing, why=why, material=True)


def apply_policy(
    message: str,
    interpretation: Interpretation,
    context: ContextHandoff | None,
    resolved: frozenset[Topic] = frozenset(),
) -> PolicyOutcome:
    """`resolved` names topics the user has since ANSWERED in a way the
    request's own wording cannot show (a subject, an intent): they are not
    asked again. Empty for an ordinary first look at a request."""
    families = set(interpretation.required_families)

    # Constraints in force: the request's own, then whatever the context knows
    # about a topic the request did not settle.
    constraints = list(interpretation.constraints)
    stated = {c.topic for c in constraints}
    if context is not None:
        for known in context.known_constraints:
            if known.topic not in stated and len(constraints) < MAX_LIST_ITEMS:
                constraints.append(Constraint(known.topic, known.text, ConstraintSource.CONTEXT))
                stated.add(known.topic)

    needs: list[ClarificationNeed] = []

    if Topic.SUBJECT not in resolved and not has_subject(message):
        needs.append(_need(Topic.SUBJECT))
    if interpretation.confidence is Confidence.LOW and Topic.INTENT not in resolved:
        needs.append(_need(Topic.INTENT))

    if not needs:
        choosing = bool(families & {GoalFamily.RESEARCH, GoalFamily.ANALYSIS}) and is_purchase_oriented(message)
        budget_open = choosing and Topic.BUDGET not in stated
        if budget_open:
            needs.append(_need(Topic.BUDGET))
        if budget_open and Topic.USE_CASE not in stated:
            needs.append(_need(Topic.USE_CASE))
        if (
            GoalFamily.ORGANIZATION in families
            and needs_time(message)
            and not has_time(message)
            and Topic.TIMEFRAME not in stated
        ):
            needs.append(_need(Topic.WHEN))

    assumptions: list[Assumption] = []
    if GoalFamily.RESEARCH in families:
        if Topic.QUANTITY not in stated:
            assumptions.append(Assumption(Topic.QUANTITY, "Shortlist up to 3 candidates"))
        if Topic.TIMEFRAME not in stated:
            assumptions.append(Assumption(Topic.TIMEFRAME, "Use current information"))

    return PolicyOutcome(
        constraints=tuple(constraints),
        assumptions=tuple(assumptions),
        needs=tuple(needs[:MAX_CLARIFICATION_NEEDS]),
    )
