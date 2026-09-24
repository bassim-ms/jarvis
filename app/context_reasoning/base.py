"""
Context-Aware Reasoning abstraction (M16.3): ReasoningResult - a
structured, validated answer to "given what the user asked and the
context M16.2 selected, what does that context mean for this request?"

This module is architecture only, the same "types and pure validation, no
I/O" role app/context_engine/base.py and app/world_model/base.py play for
their own milestones - app/context_reasoning/deterministic.py derives
bounded, explainable facts from a ContextSnapshot; app/context_reasoning/
reasoner.py:ContextReasoner consumes those (plus the raw request and
RequestContext) to build a ReasoningResult. Nothing here calls an AI
provider, queries the World Model, or executes anything.

Every fact/constraint/conflict/opportunity/conclusion carries `provenance`
- a tuple of small, stable, deterministic identifier strings (e.g.
"blocker:<id>", "goal:<id>", "deterministic:overdue_commitment",
"request") - never free-form LLM-generated prose. This mirrors
app.context_engine.base.ItemProvenance.reasons exactly: provenance is
"why is this here," traceable back to a concrete context item, not a
second layer of explanation text. See app/context_reasoning/reasoner.py
for how a ReasoningResult's provenance is validated against what was
actually supplied to the reasoner (the anti-hallucination gate).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class ContextReasoningError(ValueError):
    """Raised by ReasoningResult.__post_init__ (and its nested dataclasses'
    own __post_init__) for a structurally invalid field - never a parsing/
    LLM-output-validation error (that's app/context_reasoning/reasoner.py's
    job, kept entirely separate - see RequestContextError's own docstring
    in app/context_engine/base.py for the identical split this mirrors).
    This only validates that a directly-constructed ReasoningResult is
    well-formed."""


def _validate_text(value: Any, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ContextReasoningError(f"{field_name} must be a non-empty string, got: {value!r}")


def _validate_provenance(value: Any, field_name: str) -> None:
    if not isinstance(value, tuple) or any(not isinstance(v, str) or not v for v in value):
        raise ContextReasoningError(
            f"{field_name} must be a tuple of non-empty strings, got: {value!r}"
        )


@dataclass(frozen=True)
class RelevantFact:
    """One fact the reasoner treated as relevant to this request -
    grounded in `provenance` (see this module's own docstring). Never
    constructed directly from raw LLM output without validation - see
    app/context_reasoning/reasoner.py."""

    text: str
    provenance: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _validate_text(self.text, "RelevantFact.text")
        _validate_provenance(self.provenance, "RelevantFact.provenance")

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "provenance": list(self.provenance)}

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "RelevantFact":
        return RelevantFact(text=data["text"], provenance=tuple(data.get("provenance", ())))


@dataclass(frozen=True)
class Constraint:
    """Something that limits what the user can/should do right now (an
    active blocker, a resource that isn't available, a due date already
    past) - distinct from a RelevantFact in that a Constraint specifically
    bears on what's POSSIBLE/ADVISABLE, not merely true."""

    text: str
    provenance: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _validate_text(self.text, "Constraint.text")
        _validate_provenance(self.provenance, "Constraint.provenance")

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "provenance": list(self.provenance)}

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "Constraint":
        return Constraint(text=data["text"], provenance=tuple(data.get("provenance", ())))


@dataclass(frozen=True)
class Conflict:
    """Two or more things pulling against each other - overlapping
    events, a commitment due during an event, competing priorities `raw_
    request` itself surfaces ("should I work on Luniti or robotics
    tonight?"). `provenance` may legitimately name more than one item -
    that's the whole point of a conflict."""

    text: str
    provenance: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _validate_text(self.text, "Conflict.text")
        _validate_provenance(self.provenance, "Conflict.provenance")

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "provenance": list(self.provenance)}

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "Conflict":
        return Conflict(text=data["text"], provenance=tuple(data.get("provenance", ())))


@dataclass(frozen=True)
class Opportunity:
    """Something favorable the reasoner noticed - a resource that's free,
    a dependency that just cleared, a window of time with nothing else
    competing for it. Never a directive ("do X now") - that's
    `recommended_next_step`'s job; an Opportunity only names the
    favorable condition itself."""

    text: str
    provenance: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _validate_text(self.text, "Opportunity.text")
        _validate_provenance(self.provenance, "Opportunity.provenance")

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "provenance": list(self.provenance)}

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "Opportunity":
        return Opportunity(text=data["text"], provenance=tuple(data.get("provenance", ())))


@dataclass(frozen=True)
class Conclusion:
    """A conclusion the reasoner actually drew by connecting the facts/
    constraints/conflicts/opportunities above to the user's request -
    this is the "so what" ReasoningResult exists to produce.
    `recommended_next_step` (on ReasoningResult) is the single
    highest-level one of these, when there is one; `conclusions` holds
    every conclusion reached, including ones that don't rise to "the"
    next step (e.g. "the printer being broken doesn't block this
    request")."""

    text: str
    provenance: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _validate_text(self.text, "Conclusion.text")
        _validate_provenance(self.provenance, "Conclusion.provenance")

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "provenance": list(self.provenance)}

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "Conclusion":
        return Conclusion(text=data["text"], provenance=tuple(data.get("provenance", ())))


_LIST_FIELDS: tuple[tuple[str, type], ...] = (
    ("relevant_facts", RelevantFact),
    ("constraints", Constraint),
    ("conflicts", Conflict),
    ("opportunities", Opportunity),
    ("conclusions", Conclusion),
)


@dataclass(frozen=True)
class ReasoningResult:
    """What M16.3 hands back to "existing response generation" (app/
    core.py) for one request: an interpretation of what the selected
    ContextSnapshot means for it, plus the bounded, grounded facts/
    constraints/conflicts/opportunities/conclusions that interpretation
    rests on. Deterministic, serializable (see `to_dict`/`from_dict`),
    and validated at construction - never built by directly trusting raw
    LLM output (see app/context_reasoning/reasoner.py for the strict
    parse-and-validate step that stands between an AI response and a
    ReasoningResult).

    `confidence` is bounded to [0.0, 1.0] and represents confidence in
    the reasoning CONCLUSION, never a claim of mathematical certainty -
    the same non-goal app.world_model.base.validate_confidence's own
    docstring states for Preference.confidence.

    `requires_clarification` is `True` only when the supplied context
    genuinely can't resolve an important ambiguity in the request (e.g.
    "should I work on Luniti or robotics tonight?" with nothing in the
    selected context favoring either) - see `clarification_question`,
    which must be set exactly when this is `True`. M16.3 never silently
    picks one option in that situation; representing the ambiguity is
    the entire point of this field. It's `False` for the ordinary case
    where the request is already resolved by what's known.

    `recommended_next_step` is a plain, single reasoning conclusion
    ("the deployment task appears to be the next logical step") - never
    an instruction executed by anything. ReasoningResult carries no
    executable action of any kind; see this package's own module
    docstring and app/context_reasoning/reasoner.py's M15 boundary notes."""

    interpretation: str
    relevant_facts: tuple[RelevantFact, ...] = ()
    constraints: tuple[Constraint, ...] = ()
    conflicts: tuple[Conflict, ...] = ()
    opportunities: tuple[Opportunity, ...] = ()
    conclusions: tuple[Conclusion, ...] = ()
    recommended_next_step: str | None = None
    confidence: float = 1.0
    requires_clarification: bool = False
    clarification_question: str | None = None

    def __post_init__(self) -> None:
        _validate_text(self.interpretation, "ReasoningResult.interpretation")

        for field_name, item_type in _LIST_FIELDS:
            items = getattr(self, field_name)
            if not isinstance(items, tuple) or any(not isinstance(i, item_type) for i in items):
                raise ContextReasoningError(
                    f"ReasoningResult.{field_name} must be a tuple of {item_type.__name__}, "
                    f"got: {items!r}"
                )

        if (
            not isinstance(self.confidence, (int, float))
            or isinstance(self.confidence, bool)
            or not (0.0 <= self.confidence <= 1.0)
        ):
            raise ContextReasoningError(
                f"ReasoningResult.confidence must be a float in [0.0, 1.0], got: "
                f"{self.confidence!r}"
            )

        if not isinstance(self.requires_clarification, bool):
            raise ContextReasoningError(
                f"ReasoningResult.requires_clarification must be a bool, got: "
                f"{self.requires_clarification!r}"
            )

        if self.requires_clarification:
            if not isinstance(self.clarification_question, str) or not self.clarification_question.strip():
                raise ContextReasoningError(
                    "ReasoningResult.clarification_question must be a non-empty string "
                    "when requires_clarification is True."
                )
        elif self.clarification_question is not None:
            raise ContextReasoningError(
                "ReasoningResult.clarification_question must be None when "
                "requires_clarification is False, got: "
                f"{self.clarification_question!r}"
            )

        if self.recommended_next_step is not None and (
            not isinstance(self.recommended_next_step, str)
            or not self.recommended_next_step.strip()
        ):
            raise ContextReasoningError(
                "ReasoningResult.recommended_next_step must be None or a non-empty "
                f"string, got: {self.recommended_next_step!r}"
            )

    def is_empty(self) -> bool:
        """True when this result carries nothing that would meaningfully
        change downstream reasoning/prompt content - no facts,
        constraints, conflicts, opportunities, conclusions, or genuine
        ambiguity. Used by app/core.py to preserve byte-identical prompt
        text for a request whose ContextSnapshot had nothing relevant
        (M16.3's own empty-context backward-compatibility requirement) -
        never used to change what THIS object itself contains."""
        return (
            not self.relevant_facts
            and not self.constraints
            and not self.conflicts
            and not self.opportunities
            and not self.conclusions
            and not self.requires_clarification
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a small, JSON-friendly dict - the same
        `to_dict`/`from_dict` convention app.automation.base.Condition
        already uses - so a later milestone (M16.4 planning) can persist
        or transmit a ReasoningResult without depending on this module's
        own dataclasses."""
        return {
            "interpretation": self.interpretation,
            "relevant_facts": [f.to_dict() for f in self.relevant_facts],
            "constraints": [c.to_dict() for c in self.constraints],
            "conflicts": [c.to_dict() for c in self.conflicts],
            "opportunities": [o.to_dict() for o in self.opportunities],
            "conclusions": [c.to_dict() for c in self.conclusions],
            "recommended_next_step": self.recommended_next_step,
            "confidence": self.confidence,
            "requires_clarification": self.requires_clarification,
            "clarification_question": self.clarification_question,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "ReasoningResult":
        """Reconstruct a ReasoningResult from its `to_dict()` form - the
        inverse. Raises ContextReasoningError (via __post_init__) for a
        structurally invalid dict, exactly as direct construction would."""
        return ReasoningResult(
            interpretation=data["interpretation"],
            relevant_facts=tuple(
                RelevantFact.from_dict(f) for f in data.get("relevant_facts", ())
            ),
            constraints=tuple(
                Constraint.from_dict(c) for c in data.get("constraints", ())
            ),
            conflicts=tuple(Conflict.from_dict(c) for c in data.get("conflicts", ())),
            opportunities=tuple(
                Opportunity.from_dict(o) for o in data.get("opportunities", ())
            ),
            conclusions=tuple(Conclusion.from_dict(c) for c in data.get("conclusions", ())),
            recommended_next_step=data.get("recommended_next_step"),
            confidence=data.get("confidence", 1.0),
            requires_clarification=data.get("requires_clarification", False),
            clarification_question=data.get("clarification_question"),
        )
