"""
Context-Aware Planning abstraction (M16.4): PlanningDirective - a
structured, validated answer to "given what the user asked, what M16.2
selected, and what M16.3 concluded, what STRATEGY does this request
call for?"

This is architecture only, the same "types and pure validation, no I/O"
role app/context_reasoning/base.py plays for M16.3, one layer up.
app/context_planning/planner.py:ContextPlanner consumes a
ReasoningResult (never queries the World Model or MemoryStore itself) to
build a PlanningDirective. Nothing here calls an AI provider, executes a
tool, or creates a Job - see this package's own module docstring.

PlanningDirective is deliberately NOT a competing Plan/PlanStep system:
it has no steps, no tool calls, no dependency graph to execute - only a
`planning_type` classification and a handful of bounded, grounded
observations (`contextual_constraints`/`relevant_resources`/
`dependencies`) plus free-text `assumptions`/`suggested_approach`. If a
request genuinely needs executable steps, that's exactly what M15's own
Planner/Plan/PlanStep (app/orchestration/) already do - a
PlanningDirective's `requires_execution` flag only ever NAMES that fact;
it never itself creates, validates, or runs anything (see
app/core.py:JarvisCore._plan_about_context for how it reaches "existing
response generation" instead)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class PlanningType(Enum):
    """What KIND of planning strategy a request calls for - a small,
    fixed, closed set (M16.4's own spec), never inferred or extended
    ad-hoc.

    NONE - purely informational; nothing to plan ("what's happening with
      my robotics project?").
    DECISION - the user is choosing between options; the value here is
      framing the decision, not picking or executing anything ("should
      I work on robotics or Luniti tonight?").
    ACTION - a single, mostly self-contained thing to get done ("finish
      my robotics arm").
    MULTI_STEP - a goal whose accomplishment genuinely involves several
      dependent parts, especially once weighed against other context
      ("finish my robotics arm around my other commitments").
    """

    NONE = "none"
    DECISION = "decision"
    ACTION = "action"
    MULTI_STEP = "multi_step"


_VALID_PLANNING_TYPE_VALUES = {member.value for member in PlanningType}


class PlanningTypeError(ValueError):
    """Raised by `parse_planning_type` for an unrecognized planning type."""


def parse_planning_type(value: str) -> PlanningType:
    """Case/whitespace-tolerant parse (LLM output normalization only -
    e.g. "ACTION"/"Action"/"action" all resolve the same way); the VALUE
    itself must still be one of the fixed four, exactly like every other
    `parse_*` function in this codebase (app/world_model/base.py,
    app/context_engine/base.py)."""
    normalized = value.strip().lower()
    if normalized not in _VALID_PLANNING_TYPE_VALUES:
        raise PlanningTypeError(
            f"Unknown planning type: '{value}'. Must be one of: "
            f"{sorted(_VALID_PLANNING_TYPE_VALUES)}"
        )
    return PlanningType(normalized)


class PlanningDirectiveError(ValueError):
    """Raised by PlanningDirective.__post_init__ (and GroundedReference's
    own __post_init__) for a structurally invalid field, OR for a
    deterministic coherence violation (e.g. NONE/DECISION claiming
    `requires_execution=True`) - never a parsing/LLM-output-validation
    error (that's app/context_planning/planner.py's job, kept entirely
    separate - mirrors ContextReasoningError/ContextReasonerError's own
    split in app/context_reasoning/, M16.3)."""


def _validate_text(value: Any, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise PlanningDirectiveError(f"{field_name} must be a non-empty string, got: {value!r}")


def _validate_provenance(value: Any, field_name: str) -> None:
    if not isinstance(value, tuple) or any(not isinstance(v, str) or not v for v in value):
        raise PlanningDirectiveError(
            f"{field_name} must be a tuple of non-empty strings, got: {value!r}"
        )


@dataclass(frozen=True)
class GroundedReference:
    """One grounded observation - a contextual constraint, a relevant
    resource, or a dependency - carrying `provenance` (stable,
    deterministic identifier strings, e.g. "blocker:<id>", the SAME
    convention app.context_reasoning.base.RelevantFact already
    established for M16.3) so it can be traced back to something that
    was actually supplied, never invented. See
    app/context_planning/planner.py for how provenance is validated
    against what was actually available."""

    text: str
    provenance: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _validate_text(self.text, "GroundedReference.text")
        _validate_provenance(self.provenance, "GroundedReference.provenance")

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "provenance": list(self.provenance)}

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "GroundedReference":
        return GroundedReference(text=data["text"], provenance=tuple(data.get("provenance", ())))


#: `requires_execution` may only be True for these planning types - a
#: NONE/DECISION directive claiming execution is required is a
#: deterministic coherence violation (M16.4's own "no impossible/
#: contradictory requirements" validation requirement), enforced
#: structurally in `PlanningDirective.__post_init__` regardless of what
#: an LLM claimed.
_EXECUTABLE_PLANNING_TYPES = frozenset({PlanningType.ACTION, PlanningType.MULTI_STEP})


@dataclass(frozen=True)
class PlanningDirective:
    """What M16.4 hands back to "existing response generation" (app/
    core.py) for one request: a strategic classification of what kind of
    planning it calls for, plus the bounded, grounded observations that
    classification rests on. Deterministic, serializable (`to_dict`/
    `from_dict`), and validated at construction - never built by
    directly trusting raw LLM output (see app/context_planning/
    planner.py for the strict parse-and-validate step).

    `confidence` is bounded to [0.0, 1.0] and represents confidence in
    the planning classification, never a claim of certainty - the same
    non-goal ReasoningResult.confidence's own docstring states
    (app/context_reasoning/base.py).

    `requires_execution` only ever NAMES that executable follow-through
    would be needed - it never triggers anything. If `True`, "existing
    response generation" may mention that a task could be created, but
    actually creating one still goes through the entirely separate,
    pre-existing TASK capability path (JarvisCore._handle_task ->
    ToolRegistry's `create_job` - app/core.py, app/orchestration/tools.py),
    completely untouched by M16.4. See `_EXECUTABLE_PLANNING_TYPES` above
    for the one structural coherence rule this field is checked against."""

    objective: str
    planning_type: PlanningType
    contextual_constraints: tuple[GroundedReference, ...] = ()
    relevant_resources: tuple[GroundedReference, ...] = ()
    dependencies: tuple[GroundedReference, ...] = ()
    assumptions: tuple[str, ...] = ()
    suggested_approach: str | None = None
    requires_execution: bool = False
    confidence: float = 1.0

    def __post_init__(self) -> None:
        _validate_text(self.objective, "PlanningDirective.objective")

        if not isinstance(self.planning_type, PlanningType):
            raise PlanningDirectiveError(
                f"PlanningDirective.planning_type must be a PlanningType member, "
                f"got: {self.planning_type!r}"
            )

        for field_name in ("contextual_constraints", "relevant_resources", "dependencies"):
            items = getattr(self, field_name)
            if not isinstance(items, tuple) or any(
                not isinstance(i, GroundedReference) for i in items
            ):
                raise PlanningDirectiveError(
                    f"PlanningDirective.{field_name} must be a tuple of "
                    f"GroundedReference, got: {items!r}"
                )

        if not isinstance(self.assumptions, tuple) or any(
            not isinstance(a, str) or not a.strip() for a in self.assumptions
        ):
            raise PlanningDirectiveError(
                f"PlanningDirective.assumptions must be a tuple of non-empty "
                f"strings, got: {self.assumptions!r}"
            )

        if self.suggested_approach is not None and (
            not isinstance(self.suggested_approach, str) or not self.suggested_approach.strip()
        ):
            raise PlanningDirectiveError(
                "PlanningDirective.suggested_approach must be None or a "
                f"non-empty string, got: {self.suggested_approach!r}"
            )

        if not isinstance(self.requires_execution, bool):
            raise PlanningDirectiveError(
                f"PlanningDirective.requires_execution must be a bool, got: "
                f"{self.requires_execution!r}"
            )
        if self.requires_execution and self.planning_type not in _EXECUTABLE_PLANNING_TYPES:
            raise PlanningDirectiveError(
                f"requires_execution=True is incoherent with planning_type="
                f"{self.planning_type.value!r} - only "
                f"{sorted(t.value for t in _EXECUTABLE_PLANNING_TYPES)} may require "
                "execution."
            )

        if (
            not isinstance(self.confidence, (int, float))
            or isinstance(self.confidence, bool)
            or not (0.0 <= self.confidence <= 1.0)
        ):
            raise PlanningDirectiveError(
                f"PlanningDirective.confidence must be a float in [0.0, 1.0], got: "
                f"{self.confidence!r}"
            )

    def is_empty(self) -> bool:
        """True when this directive carries nothing that would
        meaningfully change downstream prompt content - NONE, no
        constraints/resources/dependencies/assumptions, no suggested
        approach, and no execution requirement. Used by app/core.py to
        preserve byte-identical prompt text for an informational
        request M16.4 was never meant to enrich - the same "empty means
        nothing to add" guarantee ReasoningResult.is_empty() already
        gives M16.3 one layer down."""
        return (
            self.planning_type is PlanningType.NONE
            and not self.contextual_constraints
            and not self.relevant_resources
            and not self.dependencies
            and not self.assumptions
            and self.suggested_approach is None
            and not self.requires_execution
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "objective": self.objective,
            "planning_type": self.planning_type.value,
            "contextual_constraints": [c.to_dict() for c in self.contextual_constraints],
            "relevant_resources": [r.to_dict() for r in self.relevant_resources],
            "dependencies": [d.to_dict() for d in self.dependencies],
            "assumptions": list(self.assumptions),
            "suggested_approach": self.suggested_approach,
            "requires_execution": self.requires_execution,
            "confidence": self.confidence,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "PlanningDirective":
        return PlanningDirective(
            objective=data["objective"],
            planning_type=parse_planning_type(data["planning_type"]),
            contextual_constraints=tuple(
                GroundedReference.from_dict(c) for c in data.get("contextual_constraints", ())
            ),
            relevant_resources=tuple(
                GroundedReference.from_dict(r) for r in data.get("relevant_resources", ())
            ),
            dependencies=tuple(
                GroundedReference.from_dict(d) for d in data.get("dependencies", ())
            ),
            assumptions=tuple(data.get("assumptions", ())),
            suggested_approach=data.get("suggested_approach"),
            requires_execution=data.get("requires_execution", False),
            confidence=data.get("confidence", 1.0),
        )
