"""
Context Engine abstraction (M16.2): RequestContext - a structured
representation of what a request needs Rhasspy to UNDERSTAND, as opposed
to RequestRequirements (app/providers/requirements.py), which represents
what a request needs Rhasspy to DO.

The two are deliberately separate types, never merged into one enum:
RequestRequirements' TaskType (GENERAL/REASONING/CODING/TOOL_USE/
STRUCTURED_OUTPUT/SEARCH_SYNTHESIS) is about provider selection - which
AI backend can best execute this request. ContextIntent (this module) is
about retrieval - which slice of the Personal World Model
(app/world_model/) is relevant to understanding this request. A single
message can independently be TaskType.REASONING (needs a provider good
at reasoning) AND ContextIntent.PLANNING (needs project/goal/commitment
context) - collapsing them would force one axis to stand in for the
other.

This module is architecture only, the same "types and pure validation,
no I/O" role app/orchestration/base.py and app/world_model/base.py play
for their own milestones - app/context_engine/request_builder.py builds
a RequestContext deterministically; app/context_engine/engine.py
consumes one to build a ContextSnapshot (app/world_model/base.py).
Nothing here calls an AI provider, queries the World Model, or touches
MemoryStore.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class RequestContextError(ValueError):
    """Raised by RequestContext.__post_init__ (and TimeScope's/
    ContextBudget's own __post_init__) for a structurally invalid field -
    never a parsing/classification error (that's request_builder.py's
    job, kept entirely separate - see RequestContext's own docstring).
    This only validates that a directly-constructed RequestContext/
    TimeScope/ContextBudget is well-formed: real enum members where an
    enum is expected, non-negative budget values. It never re-derives or
    second-guesses what a value SHOULD be, only whether what's given is
    valid."""


class ContextIntent(Enum):
    """What KIND of understanding a request needs from the World Model -
    orthogonal to Capability (app/capability.py, "who handles this
    message") and to RequestRequirements.TaskType (app/providers/
    requirements.py, "what does the provider need to be good at").

    GENERAL - no specific context shape identified; a light, broad
      snapshot (see `default_required_context` below).
    PROJECT_FOCUS - the request centers on one or more named/implied
      projects ("tell me about my robotics arm project").
    TASK_EXECUTION - about doing/advancing a specific task/job
      ("finish the gripper print", "work on the assignment").
    PLANNING - about deciding what to do / sequencing work ("what should
      I work on next", "plan my week").
    SCHEDULING - about time/calendar-shaped questions ("what's on my
      calendar today", "when is X due").
    STATUS - about current state/progress ("what's blocking me",
      "how's the robotics project going").
    DECISION_SUPPORT - the user is choosing between options and needs
      broader context (projects, blockers, resources, preferences) to
      decide - "should I work on X or Y".
    """

    GENERAL = "general"
    PROJECT_FOCUS = "project_focus"
    TASK_EXECUTION = "task_execution"
    PLANNING = "planning"
    SCHEDULING = "scheduling"
    STATUS = "status"
    DECISION_SUPPORT = "decision_support"


_VALID_CONTEXT_INTENT_VALUES = {member.value for member in ContextIntent}


class ContextIntentError(ValueError):
    """Raised by `parse_context_intent` for an unrecognized intent."""


def parse_context_intent(value: str) -> ContextIntent:
    if value not in _VALID_CONTEXT_INTENT_VALUES:
        raise ContextIntentError(
            f"Unknown context intent: '{value}'. Must be one of: "
            f"{sorted(_VALID_CONTEXT_INTENT_VALUES)}"
        )
    return ContextIntent(value)


class TimeScopeKind(Enum):
    """What kind of time boundary a request implies - deterministically
    parsed from a small, fixed set of temporal phrases (see
    app/context_engine/request_builder.py); never inferred by an AI
    call.

    NONE - no temporal phrase detected; no time filtering applied.
    NOW - "right now" / "currently" / "at the moment".
    TODAY - "today".
    TOMORROW - "tomorrow".
    THIS_WEEK - "this week".
    DATE_RANGE - an explicit "from X to Y" / "between X and Y" date
      range was given; `start`/`end` are populated from it.
    UPCOMING - "upcoming" / "coming up" / "what's next" - a forward-
      looking window with no fixed end.
    OVERDUE - "overdue" / "late" / "past due" - things whose due time
      has already passed. Overdue items are never excluded from
      relevance for being in the past - see
      app/context_engine/scoring.py's own time-relevance docstring.
    """

    NONE = "none"
    NOW = "now"
    TODAY = "today"
    TOMORROW = "tomorrow"
    THIS_WEEK = "this_week"
    DATE_RANGE = "date_range"
    UPCOMING = "upcoming"
    OVERDUE = "overdue"


@dataclass(frozen=True)
class TimeScope:
    """A structured time boundary. `start`/`end` are ISO 8601 UTC
    timestamps (the same string convention every timestamp in this
    codebase already uses - see app/world_model/base.py's entities),
    populated whenever `kind` implies a concrete boundary (TODAY,
    TOMORROW, THIS_WEEK, DATE_RANGE); left `None` for kinds that don't
    (NONE, NOW is a point not a range, UPCOMING/OVERDUE are open-ended
    by design)."""

    kind: TimeScopeKind = TimeScopeKind.NONE
    start: str | None = None
    end: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, TimeScopeKind):
            raise RequestContextError(
                f"TimeScope.kind must be a TimeScopeKind member, got: {self.kind!r}"
            )


class RequiredContextCategory(Enum):
    """One World Model context category a request may need. Deliberately
    a closed enum, not a free-form string set - see ContextBudget for
    why each one needs a centrally-defined default limit too."""

    PROJECTS = "projects"
    GOALS = "goals"
    TASKS = "tasks"
    COMMITMENTS = "commitments"
    EVENTS = "events"
    BLOCKERS = "blockers"
    RESOURCES = "resources"
    PREFERENCES = "preferences"
    MEMORY = "memory"


@dataclass(frozen=True)
class ContextBudget:
    """Per-category maximum number of items a ContextSnapshot may
    include - the ONE centralized place these limits are defined (see
    DEFAULT_CONTEXT_BUDGET below); app/context_engine/engine.py reads
    `for_category(...)` rather than any category having its own
    hardcoded number. Defaults are reasonable starting values (M16.2's
    own spec), not fixed architecture - callers needing different limits
    construct their own ContextBudget instead of this module growing
    per-caller special cases.
    """

    projects: int = 5
    goals: int = 8
    tasks: int = 10
    commitments: int = 5
    events: int = 8
    blockers: int = 8
    resources: int = 8
    preferences: int = 10
    memory: int = 10

    def __post_init__(self) -> None:
        negative = {
            name: value
            for name in (
                "projects",
                "goals",
                "tasks",
                "commitments",
                "events",
                "blockers",
                "resources",
                "preferences",
                "memory",
            )
            if (value := getattr(self, name)) < 0
        }
        if negative:
            raise RequestContextError(
                f"ContextBudget values must be non-negative, got: {negative!r}"
            )

    def for_category(self, category: RequiredContextCategory) -> int:
        return getattr(self, category.value)


#: The single default budget every RequestContext uses unless a caller
#: explicitly overrides it - see ContextBudget's own docstring.
DEFAULT_CONTEXT_BUDGET = ContextBudget()

#: A staged candidate (app/memory_policy/) reaches a ContextSnapshot only once it has
#: been corroborated to at least this confidence (a single uncertain statement or
#: inference starts at 0.5), and at most this many are included. They are always
#: labelled unconfirmed and never placed in a fact category.
CANDIDATE_CONTEXT_MIN_CONFIDENCE = 0.6
MAX_CONTEXT_CANDIDATES = 5

#: The default RequiredContextCategory set for each ContextIntent -
#: centralizes "don't retrieve categories unnecessarily" in one place
#: (app/context_engine/request_builder.py:default_required_context)
#: rather than scattering per-intent category lists across callers.
#: MEMORY is deliberately excluded from every default (it stays a
#: separate, explicit opt-in retrieval source - see this package's own
#: module docstring in engine.py for why), and PREFERENCES only appears
#: where a request is plausibly about making a choice/being helped
#: (DECISION_SUPPORT) - every other category set matches the intent's
#: own worked description above.
_DEFAULT_REQUIRED_CONTEXT: dict[ContextIntent, frozenset[RequiredContextCategory]] = {
    ContextIntent.GENERAL: frozenset(
        {
            RequiredContextCategory.PROJECTS,
            RequiredContextCategory.GOALS,
            RequiredContextCategory.TASKS,
            RequiredContextCategory.COMMITMENTS,
            RequiredContextCategory.EVENTS,
            RequiredContextCategory.BLOCKERS,
        }
    ),
    ContextIntent.PROJECT_FOCUS: frozenset(
        {
            RequiredContextCategory.PROJECTS,
            RequiredContextCategory.GOALS,
            RequiredContextCategory.TASKS,
            RequiredContextCategory.BLOCKERS,
            RequiredContextCategory.RESOURCES,
        }
    ),
    ContextIntent.TASK_EXECUTION: frozenset(
        {
            RequiredContextCategory.TASKS,
            RequiredContextCategory.GOALS,
            RequiredContextCategory.BLOCKERS,
            RequiredContextCategory.RESOURCES,
        }
    ),
    ContextIntent.PLANNING: frozenset(
        {
            RequiredContextCategory.PROJECTS,
            RequiredContextCategory.GOALS,
            RequiredContextCategory.TASKS,
            RequiredContextCategory.COMMITMENTS,
            RequiredContextCategory.EVENTS,
        }
    ),
    ContextIntent.SCHEDULING: frozenset(
        {
            RequiredContextCategory.EVENTS,
            RequiredContextCategory.COMMITMENTS,
            RequiredContextCategory.TASKS,
        }
    ),
    ContextIntent.STATUS: frozenset(
        {
            RequiredContextCategory.PROJECTS,
            RequiredContextCategory.GOALS,
            RequiredContextCategory.TASKS,
            RequiredContextCategory.BLOCKERS,
            RequiredContextCategory.COMMITMENTS,
        }
    ),
    ContextIntent.DECISION_SUPPORT: frozenset(
        {
            RequiredContextCategory.PROJECTS,
            RequiredContextCategory.GOALS,
            RequiredContextCategory.BLOCKERS,
            RequiredContextCategory.RESOURCES,
            RequiredContextCategory.PREFERENCES,
        }
    ),
}


def default_required_context(intent: ContextIntent) -> frozenset[RequiredContextCategory]:
    """The default RequiredContextCategory set for `intent` - see
    `_DEFAULT_REQUIRED_CONTEXT` above."""
    return _DEFAULT_REQUIRED_CONTEXT[intent]


# ---------------------------------------------------------------------------
# ContextSnapshot metadata (retrieval/scoring concepts, not World Model
# state - see app/world_model/base.py:ContextSnapshot's own docstring for
# why these live here rather than there. ContextSnapshot itself stays in
# app/world_model/base.py as the shared output contract; only ItemProvenance/
# CategoryRetrieval/ContextWindow - the "how the Context Engine picked
# this" bookkeeping - live in this module.
# ---------------------------------------------------------------------------


@dataclass
class ItemProvenance:
    """Why ONE selected item ended up in a ContextSnapshot - the per-item
    half of "explainability/provenance" (app.context_engine.engine
    populates this; this module only defines the shape). `reasons` are
    deterministic identifiers/short strings (e.g. "explicit_match",
    "has_active_blocker" - see app.context_engine.scoring's REASON_*
    constants for the fixed vocabulary), never LLM-generated prose."""

    relevance_score: float
    reasons: tuple[str, ...] = ()


@dataclass
class CategoryRetrieval:
    """Retrieval bookkeeping for one context category.

    `considered` is the number of candidates that entered the retrieval
    pipeline BEFORE hard filtering, scoring, or budget selection - i.e.
    exactly `len(candidates retrieved from the store)`, not the
    post-hard-filter count. `selected` is how many actually ended up in
    the ContextSnapshot (after hard filtering + budget truncation).
    `items` holds the selected ones' own ItemProvenance, keyed by entity
    id - deliberately bounded to `selected` entries (never the full
    `considered` set), so this stays small, explainability metadata, not
    a full audit log - see ContextSnapshot's own docstring."""

    considered: int = 0
    selected: int = 0
    items: dict[str, ItemProvenance] = field(default_factory=dict)


@dataclass
class ContextWindow:
    """The effective time boundary a ContextSnapshot was retrieved for -
    mirrors RequestContext.time_scope's own start/end, so a snapshot is
    self-describing about what time range it reflects without a caller
    needing to keep the originating RequestContext around. `None`/`None`
    means no explicit time boundary was applied (e.g. TimeScopeKind.NONE)."""

    start: str | None = None
    end: str | None = None


@dataclass
class RequestContext:
    """A structured representation of what a request needs Rhasspy to
    UNDERSTAND (as opposed to RequestRequirements: what it needs Rhasspy
    to DO - see this module's own docstring). Built deterministically by
    app/context_engine/request_builder.py; consumed by
    app/context_engine/engine.py:ContextEngine.build_snapshot.

    `mentioned_*` are request-level TEXT HINTS ("robotics arm", "the
    physics assignment") - never World Model ids. The request parser
    that builds this has no business knowing internal database ids;
    resolving a hint against the actual World Model (deterministic
    substring matching, preserving ambiguity when more than one entity
    plausibly matches - see app/context_engine/engine.py) is entirely
    the Context Engine's job, not this dataclass's or the builder's.
    """

    raw_request: str
    context_intent: ContextIntent = ContextIntent.GENERAL
    mentioned_projects: list[str] = field(default_factory=list)
    mentioned_goals: list[str] = field(default_factory=list)
    mentioned_tasks: list[str] = field(default_factory=list)
    mentioned_commitments: list[str] = field(default_factory=list)
    mentioned_resources: list[str] = field(default_factory=list)
    time_scope: TimeScope = field(default_factory=TimeScope)
    required_context: frozenset[RequiredContextCategory] = field(
        default_factory=lambda: default_required_context(ContextIntent.GENERAL)
    )
    context_budget: ContextBudget = DEFAULT_CONTEXT_BUDGET

    def __post_init__(self) -> None:
        """Lightweight structural validation only - never re-classifies
        or re-derives anything (that stays request_builder.py's job,
        entirely separate from this model-level check). Each nested
        dataclass (TimeScope, ContextBudget) already validated itself at
        its OWN construction time (see their own __post_init__ above);
        this only needs to confirm the right TYPE was actually passed in
        for each field, plus that every `required_context` member is a
        real RequiredContextCategory."""
        if not isinstance(self.context_intent, ContextIntent):
            raise RequestContextError(
                f"context_intent must be a ContextIntent member, got: "
                f"{self.context_intent!r}"
            )
        if not isinstance(self.time_scope, TimeScope):
            raise RequestContextError(
                f"time_scope must be a TimeScope, got: {self.time_scope!r}"
            )
        if not isinstance(self.context_budget, ContextBudget):
            raise RequestContextError(
                f"context_budget must be a ContextBudget, got: {self.context_budget!r}"
            )
        invalid_categories = [
            c for c in self.required_context if not isinstance(c, RequiredContextCategory)
        ]
        if invalid_categories:
            raise RequestContextError(
                f"required_context contains non-RequiredContextCategory values: "
                f"{invalid_categories!r}"
            )
