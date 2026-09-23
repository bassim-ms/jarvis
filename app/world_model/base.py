"""
Personal World Model abstraction (M16.1): a deterministic, structured
record of what Rhasspy currently believes is true about the user's
life/projects - "what IS true," never "what was said" (that's
MemoryStore, app/memory/base.py) and never "what should be done about
it" (that's a future Context Engine, M16.2+, reasoning over a
ContextSnapshot built from this module).

Built entirely on top of architecture M1-M15 already established, not a
parallel system:

- Persistence follows the exact SQLiteJobStore/SQLiteTaskStore pattern
  (see app/world_model/sqlite_store.py): one connection held for the
  store's lifetime, guarded by a lock, typed columns for fixed-shape
  fields, JSON only where the shape genuinely varies (`metadata`).
- Task/Job linkage (see TaskLink below) references app.tasks.Task (M9)
  and app.orchestration.Job (M15) by id only - this module never
  duplicates either one's own state. M15 remains the sole execution
  authority for Jobs; M9 remains the sole authority for Tasks. The World
  Model only records "this task/job relates to this project/goal."
- Domain errors are plain ValueError subclasses, the same convention
  AutonomyError/ConditionError/JobPriorityError (app/orchestration/base.py,
  app/automation/base.py) already use - never a bare ValueError or a
  silently-swallowed invalid value.

Deliberately NOT a generic graph database: relationships are a handful of
small, explicit, fixed-shape tables (TaskLink/GoalDependency/BlockerLink/
ResourceLink) - see app/world_model/sqlite_store.py - not an arbitrary
edge-typed graph. Project/Commitment/Event each carry a direct, optional
`project_id` field rather than a relationship table, since "belongs to at
most one project" is a plain foreign key, not a many-to-many
relationship - the same "prefer a direct field over a table when the
relationship is genuinely 1:1/1:many" judgment Goal.project_id already
makes.

Every entity here is deterministic, plain-Python state: nothing in this
module calls an AI provider, and nothing here decides what to DO with
this state (planning/scheduling/notifying) - see this module's own
docstring section in the M16.1 implementation notes for the full
AI/autonomy boundary. `EntitySource.INFERENCE` exists so a future
Context Engine can record "the system guessed this," but this module
never upgrades an INFERENCE-sourced fact to a stronger source on its
own - only an explicit caller passing a different `source` does that.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Shared enums
# ---------------------------------------------------------------------------


class Priority(Enum):
    """A simple, fixed three-level priority - display/ordering only,
    shared by Project/Goal/Commitment since all three are part of this
    one cohesive module (unlike Task vs Job, which deliberately keep
    independent copies because they're separate milestones/domains - see
    app/orchestration/base.py:JobPriority's own docstring)."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


_VALID_PRIORITY_VALUES = {member.value for member in Priority}


class PriorityError(ValueError):
    """Raised by `parse_priority` for an unrecognized priority."""


def parse_priority(value: str) -> Priority:
    if value not in _VALID_PRIORITY_VALUES:
        raise PriorityError(
            f"Unknown priority: '{value}'. Must be one of: {sorted(_VALID_PRIORITY_VALUES)}"
        )
    return Priority(value)


class EntitySource(Enum):
    """Where a fact/entity came from - provenance, not confidence (see
    `validate_confidence` below for the separate, optional 0.0-1.0
    signal). Every entity that can plausibly be system-derived rather
    than a direct user statement (Commitment, Event, Preference) carries
    one of these.

    CRITICAL boundary this enum exists to enforce: INFERENCE must stay
    distinguishable from a user-confirmed fact everywhere it's read -
    nothing in this module ever rewrites an INFERENCE-sourced entity's
    `source` to something stronger on its own; only an explicit caller
    (a future Context Engine, or the user directly confirming it)
    changes it, via an ordinary update call, same as any other field.
    """

    USER = "user"
    CALENDAR = "calendar"
    TASK_SYSTEM = "task_system"
    REMINDER_SYSTEM = "reminder_system"
    MEMORY = "memory"
    TOOL = "tool"
    INFERENCE = "inference"
    SYSTEM = "system"


_VALID_SOURCE_VALUES = {member.value for member in EntitySource}


class EntitySourceError(ValueError):
    """Raised by `parse_source` for an unrecognized source."""


def parse_source(value: str) -> EntitySource:
    if value not in _VALID_SOURCE_VALUES:
        raise EntitySourceError(
            f"Unknown source: '{value}'. Must be one of: {sorted(_VALID_SOURCE_VALUES)}"
        )
    return EntitySource(value)


class InvalidConfidenceError(ValueError):
    """Raised by `validate_confidence` for a value outside [0.0, 1.0]."""


def validate_confidence(value: float) -> None:
    """Validate a confidence score - a plain float in [0.0, 1.0]. Raises
    rather than clamping, so an out-of-range value (a caller bug, not a
    legitimate edge case) is never silently coerced into a misleadingly
    "valid-looking" number."""
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not (0.0 <= value <= 1.0):
        raise InvalidConfidenceError(
            f"confidence must be a float in [0.0, 1.0], got: {value!r}"
        )


#: Default confidence for a newly created Preference when the caller
#: doesn't specify one - matches EntitySource.USER's own weight (a
#: directly user-stated preference is taken at face value); a caller
#: recording an INFERENCE-sourced preference should pass an explicit,
#: lower confidence instead of relying on this default.
DEFAULT_CONFIDENCE = 1.0


class LinkedEntityKind(Enum):
    """What kind of entity the *other* side of a relationship table row
    refers to - used by BlockerLink/ResourceLink, whose target can be a
    World-Model-internal entity (PROJECT/GOAL) or an external M9 Task/M15
    Job. Not every kind is valid for every relationship (see
    `VALID_BLOCKER_LINK_KINDS`/`VALID_RESOURCE_LINK_KINDS` below) - this
    enum only names the universe of possible kinds, it doesn't itself
    constrain any one relationship."""

    PROJECT = "project"
    GOAL = "goal"
    TASK = "task"
    JOB = "job"


_VALID_LINKED_ENTITY_KIND_VALUES = {member.value for member in LinkedEntityKind}


class LinkedEntityKindError(ValueError):
    """Raised by `parse_linked_entity_kind` for an unrecognized kind."""


def parse_linked_entity_kind(value: str) -> LinkedEntityKind:
    if value not in _VALID_LINKED_ENTITY_KIND_VALUES:
        raise LinkedEntityKindError(
            f"Unknown entity kind: '{value}'. Must be one of: "
            f"{sorted(_VALID_LINKED_ENTITY_KIND_VALUES)}"
        )
    return LinkedEntityKind(value)


#: A Blocker may affect any of these - task/job execution, a goal, or the
#: project itself (spec: "Task/Job -> Blocker, Goal -> Blocker, Project ->
#: Blocker").
VALID_BLOCKER_LINK_KINDS = frozenset(
    {LinkedEntityKind.PROJECT, LinkedEntityKind.GOAL, LinkedEntityKind.TASK, LinkedEntityKind.JOB}
)

#: A Resource may be attached to a task/job or a project (spec:
#: "Task/Job -> Resource, Project -> Resource") - deliberately narrower
#: than VALID_BLOCKER_LINK_KINDS (no direct Goal->Resource relationship
#: was asked for; a resource needed for a goal is reachable through the
#: goal's own tasks/jobs).
VALID_RESOURCE_LINK_KINDS = frozenset(
    {LinkedEntityKind.PROJECT, LinkedEntityKind.TASK, LinkedEntityKind.JOB}
)


class TaskRefType(Enum):
    """Which existing execution system a TaskLink's `ref_id` belongs to -
    app.tasks.Task (M9) or app.orchestration.Job (M15). Deliberately its
    own small enum rather than reusing LinkedEntityKind's TASK/JOB
    members directly in TaskLink's own type annotation, so a TaskLink can
    never accidentally be constructed with PROJECT/GOAL - only the two
    execution-system kinds a task/job linkage is actually about."""

    TASK = "task"
    JOB = "job"


_VALID_TASK_REF_TYPE_VALUES = {member.value for member in TaskRefType}


class TaskRefTypeError(ValueError):
    """Raised by `parse_task_ref_type` for an unrecognized ref type."""


def parse_task_ref_type(value: str) -> TaskRefType:
    if value not in _VALID_TASK_REF_TYPE_VALUES:
        raise TaskRefTypeError(
            f"Unknown task ref type: '{value}'. Must be one of: "
            f"{sorted(_VALID_TASK_REF_TYPE_VALUES)}"
        )
    return TaskRefType(value)


# ---------------------------------------------------------------------------
# Transition validation - shared, reusable machinery
# ---------------------------------------------------------------------------


class InvalidTransitionError(ValueError):
    """Raised by any of the `validate_*_transition` functions below for a
    transition not present in that entity's own transition map. Never
    raised for a same-status "transition" that a caller didn't actually
    request - callers decide whether a no-op transition is worth calling
    at all; this function only validates the (current, target) pair it's
    given."""


def _validate_transition(
    entity_name: str,
    current: Enum,
    target: Enum,
    allowed: dict[Enum, frozenset[Enum]],
) -> None:
    """The one reusable transition-validation rule every entity below
    builds on: `target` must be in `allowed.get(current, frozenset())`.
    A status with no entry in `allowed` (a terminal state) always yields
    an empty allowed-set, so every transition out of it is rejected
    without needing a repeated explicit empty entry for each terminal
    state."""
    if target not in allowed.get(current, frozenset()):
        raise InvalidTransitionError(
            f"Cannot transition {entity_name} from '{current.value}' to '{target.value}'."
        )


class ProjectStatus(Enum):
    """Lifecycle state of a Project.

    ACTIVE - the normal working state.
    PAUSED - temporarily set aside; can resume to ACTIVE or be archived.
    COMPLETED - terminal; the project's goal was achieved.
    ARCHIVED - terminal; set aside permanently without being completed.
    """

    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    ARCHIVED = "archived"


#: Exactly the transitions M16.1's spec names - ACTIVE can pause/
#: complete/archive; PAUSED can resume or archive; COMPLETED/ARCHIVED are
#: terminal (absent here, so `_validate_transition` rejects anything out
#: of them via the empty-frozenset default).
_PROJECT_TRANSITIONS: dict[ProjectStatus, frozenset[ProjectStatus]] = {
    ProjectStatus.ACTIVE: frozenset(
        {ProjectStatus.PAUSED, ProjectStatus.COMPLETED, ProjectStatus.ARCHIVED}
    ),
    ProjectStatus.PAUSED: frozenset({ProjectStatus.ACTIVE, ProjectStatus.ARCHIVED}),
}

TERMINAL_PROJECT_STATUSES = frozenset({ProjectStatus.COMPLETED, ProjectStatus.ARCHIVED})


def validate_project_transition(current: ProjectStatus, target: ProjectStatus) -> None:
    _validate_transition("project", current, target, _PROJECT_TRANSITIONS)


class GoalStatus(Enum):
    """Lifecycle state of a Goal.

    ACTIVE - being pursued.
    PAUSED - temporarily set aside; can resume or be abandoned.
    ACHIEVED - terminal; the goal was reached.
    ABANDONED - terminal; given up on without being reached.
    """

    ACTIVE = "active"
    PAUSED = "paused"
    ACHIEVED = "achieved"
    ABANDONED = "abandoned"


#: Mirrors ProjectStatus's own shape (ACTIVE <-> PAUSED, both can reach a
#: terminal state) - a Goal's lifecycle is deliberately the same shape as
#: a Project's, just with different terminal-state names (ACHIEVED/
#: ABANDONED vs COMPLETED/ARCHIVED) reflecting that a Goal is reached or
#: given up on, not "finished" in the generic Project sense.
_GOAL_TRANSITIONS: dict[GoalStatus, frozenset[GoalStatus]] = {
    GoalStatus.ACTIVE: frozenset(
        {GoalStatus.PAUSED, GoalStatus.ACHIEVED, GoalStatus.ABANDONED}
    ),
    GoalStatus.PAUSED: frozenset({GoalStatus.ACTIVE, GoalStatus.ABANDONED}),
}

TERMINAL_GOAL_STATUSES = frozenset({GoalStatus.ACHIEVED, GoalStatus.ABANDONED})


def validate_goal_transition(current: GoalStatus, target: GoalStatus) -> None:
    _validate_transition("goal", current, target, _GOAL_TRANSITIONS)


class CommitmentStatus(Enum):
    """Lifecycle state of a Commitment (an external obligation - "submit
    the assignment by Thursday" - never the same thing as an internal
    Task/Job; see the Commitment dataclass's own docstring).

    OPEN - not yet resolved either way.
    FULFILLED - terminal; the commitment was met.
    MISSED - terminal; the commitment's deadline passed unmet.
    CANCELLED - terminal; the obligation no longer applies (e.g. the
      counterparty released the user from it).
    """

    OPEN = "open"
    FULFILLED = "fulfilled"
    MISSED = "missed"
    CANCELLED = "cancelled"


#: A Commitment only ever leaves OPEN once, into exactly one terminal
#: state - unlike Project/Goal, there's no "resume" path back from any of
#: FULFILLED/MISSED/CANCELLED (a missed commitment doesn't un-miss
#: itself; a fulfilled one doesn't un-fulfill).
_COMMITMENT_TRANSITIONS: dict[CommitmentStatus, frozenset[CommitmentStatus]] = {
    CommitmentStatus.OPEN: frozenset(
        {CommitmentStatus.FULFILLED, CommitmentStatus.MISSED, CommitmentStatus.CANCELLED}
    ),
}

TERMINAL_COMMITMENT_STATUSES = frozenset(
    {CommitmentStatus.FULFILLED, CommitmentStatus.MISSED, CommitmentStatus.CANCELLED}
)


def validate_commitment_transition(current: CommitmentStatus, target: CommitmentStatus) -> None:
    _validate_transition("commitment", current, target, _COMMITMENT_TRANSITIONS)


class BlockerStatus(Enum):
    """Lifecycle state of a Blocker.

    ACTIVE - currently blocking whatever it's linked to.
    RESOLVED - terminal; the underlying problem was fixed.
    DISMISSED - terminal; determined not to actually be a blocker (or no
      longer relevant), without being "resolved" in the fixed-it sense.
    """

    ACTIVE = "active"
    RESOLVED = "resolved"
    DISMISSED = "dismissed"


_BLOCKER_TRANSITIONS: dict[BlockerStatus, frozenset[BlockerStatus]] = {
    BlockerStatus.ACTIVE: frozenset({BlockerStatus.RESOLVED, BlockerStatus.DISMISSED}),
}

TERMINAL_BLOCKER_STATUSES = frozenset({BlockerStatus.RESOLVED, BlockerStatus.DISMISSED})


def validate_blocker_transition(current: BlockerStatus, target: BlockerStatus) -> None:
    _validate_transition("blocker", current, target, _BLOCKER_TRANSITIONS)


class BlockerSeverity(Enum):
    """How much a Blocker matters - display/prioritization only, never
    gates any automatic behavior (M16.1 has none - see this module's own
    AUTONOMY boundary notes)."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


_VALID_BLOCKER_SEVERITY_VALUES = {member.value for member in BlockerSeverity}


class BlockerSeverityError(ValueError):
    """Raised by `parse_blocker_severity` for an unrecognized severity."""


def parse_blocker_severity(value: str) -> BlockerSeverity:
    if value not in _VALID_BLOCKER_SEVERITY_VALUES:
        raise BlockerSeverityError(
            f"Unknown severity: '{value}'. Must be one of: "
            f"{sorted(_VALID_BLOCKER_SEVERITY_VALUES)}"
        )
    return BlockerSeverity(value)


class ResourceType(Enum):
    """What kind of thing a Resource represents - exactly the eight
    kinds M16.1's spec names."""

    TIME = "time"
    MONEY = "money"
    DEVICE = "device"
    HARDWARE = "hardware"
    FILE = "file"
    PERSON = "person"
    SERVICE = "service"
    API = "api"
    OTHER = "other"


_VALID_RESOURCE_TYPE_VALUES = {member.value for member in ResourceType}


class ResourceTypeError(ValueError):
    """Raised by `parse_resource_type` for an unrecognized type."""


def parse_resource_type(value: str) -> ResourceType:
    if value not in _VALID_RESOURCE_TYPE_VALUES:
        raise ResourceTypeError(
            f"Unknown resource type: '{value}'. Must be one of: "
            f"{sorted(_VALID_RESOURCE_TYPE_VALUES)}"
        )
    return ResourceType(value)


class ResourceStatus(Enum):
    """A controlled (not free-text) status for a Resource's current
    availability - the spec leaves this "appropriate to the existing
    architecture" rather than naming exact values, so this is a plain,
    fixed three-state enum, the same shape as every other controlled
    status in this module. No transition graph is enforced for it (the
    spec doesn't ask for one) - `parse_resource_status` only validates
    membership, exactly like `parse_priority`."""

    AVAILABLE = "available"
    IN_USE = "in_use"
    UNAVAILABLE = "unavailable"


_VALID_RESOURCE_STATUS_VALUES = {member.value for member in ResourceStatus}


class ResourceStatusError(ValueError):
    """Raised by `parse_resource_status` for an unrecognized status."""


def parse_resource_status(value: str) -> ResourceStatus:
    if value not in _VALID_RESOURCE_STATUS_VALUES:
        raise ResourceStatusError(
            f"Unknown resource status: '{value}'. Must be one of: "
            f"{sorted(_VALID_RESOURCE_STATUS_VALUES)}"
        )
    return ResourceStatus(value)


# ---------------------------------------------------------------------------
# Entities
# ---------------------------------------------------------------------------


@dataclass
class Project:
    """A user's project - the top of the World Model's own hierarchy
    (Project -> Goal -> linked Task/Job -> M15 Plan -> execution).
    `metadata` is a small, JSON-safe dict for anything not worth its own
    typed column, the same role Job.context plays for orchestration."""

    id: str
    name: str
    description: str
    status: ProjectStatus
    priority: Priority
    created_at: str
    updated_at: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Goal:
    """A goal belonging to exactly one Project. `target_date` is
    display-only (an optional ISO 8601 date/timestamp) - nothing polls it
    or fires anything from it, the same "display-only, no background
    process of its own" convention Task.due_date already uses
    (app/tasks/base.py)."""

    id: str
    project_id: str
    title: str
    description: str
    status: GoalStatus
    priority: Priority
    target_date: str | None
    created_at: str
    updated_at: str


@dataclass
class Commitment:
    """An external obligation - "submit the physics assignment by
    Thursday" - deliberately NOT the same thing as an internal Task/Job:
    a Task/Job is work Rhasspy tracks/executes; a Commitment is an
    obligation that exists whether or not Rhasspy ever does anything
    about it. `project_id` is optional (a commitment need not belong to
    any project)."""

    id: str
    title: str
    description: str
    status: CommitmentStatus
    priority: Priority
    due_at: str | None
    source: EntitySource
    created_at: str
    updated_at: str
    project_id: str | None = None


@dataclass
class Event:
    """A time-bound occurrence - not a task, never executed. `external_id`
    is an opaque string reserved for a future sync integration (e.g.
    Google Calendar's own event id) - M16.1 never reads or writes an
    external calendar itself (see app/calendar/ for the actual M9 Google
    Calendar integration, unrelated to and untouched by this module)."""

    id: str
    title: str
    description: str
    starts_at: str
    ends_at: str
    location: str
    source: EntitySource
    external_id: str | None
    created_at: str
    updated_at: str
    project_id: str | None = None


@dataclass
class Resource:
    """Something a project/task/job may need or use - a device, a block
    of time, a file, a person, a service/API. Deliberately no
    availability *scheduling* (no "busy until X") - just a controlled
    current status (see ResourceStatus); anything more belongs to a
    later milestone that actually reasons about resource contention."""

    id: str
    name: str
    type: ResourceType
    status: ResourceStatus
    description: str
    created_at: str
    updated_at: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Blocker:
    """Something actively preventing progress on a task/job, goal, or
    project - represented as its own first-class entity (with its own
    lifecycle, severity, and explicit links) rather than a boolean
    `blocked=true` flag anywhere else, exactly per this milestone's own
    spec: a Blocker like "printer nozzle isn't heating" carries real
    information (what/why/how severe/when) a flag can't."""

    id: str
    title: str
    description: str
    status: BlockerStatus
    severity: BlockerSeverity
    detected_at: str
    created_at: str
    updated_at: str
    resolved_at: str | None = None


@dataclass
class Preference:
    """A stable, operational preference - "response_style: concise" -
    never an arbitrary conversation fact (those stay in MemoryStore; see
    this module's own docstring's Memory/World-Model boundary). `key` is
    not unique on its own (the same key can exist more than once across
    different `category`s) - see WorldModelStore.list_preferences for how
    a caller finds "the" current value for a key."""

    id: str
    key: str
    value: str
    category: str
    source: EntitySource
    confidence: float
    updated_at: str


# ---------------------------------------------------------------------------
# Relationships
# ---------------------------------------------------------------------------


@dataclass
class TaskLink:
    """Associates an existing app.tasks.Task (M9) or app.orchestration.Job
    (M15) with a Project and/or Goal. `ref_id` is stored opaquely - this
    module never validates it against the Task/Job's own store (a
    separate SQLite file with its own lifecycle; see this module's own
    docstring) - M9/M15 each remain the sole authority for whether that
    id actually exists or what state it's in. At least one of
    `project_id`/`goal_id` is set (WorldModelService enforces this at
    creation; see app/world_model/service.py:link_task)."""

    id: str
    ref_type: TaskRefType
    ref_id: str
    created_at: str
    project_id: str | None = None
    goal_id: str | None = None


@dataclass
class GoalDependency:
    """`goal_id` depends on `depends_on_goal_id` - both are World-Model
    Goal ids (validated to exist; see WorldModelService.add_goal_dependency).
    Purely descriptive (M16.1 has no scheduler that acts on this) - a
    future Context Engine/planner can use it to sequence goals, but
    nothing here enforces or checks the dependency automatically."""

    id: str
    goal_id: str
    depends_on_goal_id: str
    created_at: str


@dataclass
class BlockerLink:
    """A Blocker affecting a task/job/goal/project - see
    VALID_BLOCKER_LINK_KINDS for which `entity_kind`s are valid here."""

    id: str
    blocker_id: str
    entity_kind: LinkedEntityKind
    entity_id: str
    created_at: str


@dataclass
class ResourceLink:
    """A Resource attached to a task/job/project - see
    VALID_RESOURCE_LINK_KINDS for which `entity_kind`s are valid here."""

    id: str
    resource_id: str
    entity_kind: LinkedEntityKind
    entity_id: str
    created_at: str


# ---------------------------------------------------------------------------
# ContextSnapshot (M16.1 shape only) - populated by app.context_engine (M16.2)
# ---------------------------------------------------------------------------


@dataclass
class ContextSnapshot:
    """The shape a Context Engine (M16.2, app/context_engine/) populates
    by querying this module and MemoryStore together, then hands to AI
    reasoning ahead of M15 planning/execution.

    M16.1 defined this shape only, with no builder; M16.2's
    `app.context_engine.engine.ContextEngine.build_snapshot` is the one
    place a "real" ContextSnapshot gets constructed - this module itself
    still never builds one. `relevant_memory` is a plain list of strings
    (not app.memory.base.Message) so this module never has to import
    MemoryStore's own types - the Context Engine that actually populates
    this field owns translating memory into whatever form it puts here.

    `context_window`/`retrieval_metadata` (added for M16.2) are
    deliberately bounded debug/explainability metadata, not a full
    audit log - see app.context_engine.base.CategoryRetrieval's own
    docstring for exactly what they hold. Their concrete types
    (ContextWindow, CategoryRetrieval) live in app/context_engine/base.py,
    not here: they're retrieval/scoring concepts (how the Context Engine
    selected something), not World Model state (what's true) - this
    module only needs to know their SHAPE well enough to hold them, via
    a forward-reference string annotation (safe because this file starts
    with `from __future__ import annotations`, so these annotations are
    never evaluated at runtime and this module never has to import
    app.context_engine - keeping the dependency direction World Model
    -> nothing, Context Engine -> World Model, never the other way).
    """

    current_time: str
    active_projects: list[Project] = field(default_factory=list)
    relevant_goals: list[Goal] = field(default_factory=list)
    relevant_tasks: list[TaskLink] = field(default_factory=list)
    upcoming_commitments: list[Commitment] = field(default_factory=list)
    upcoming_events: list[Event] = field(default_factory=list)
    active_blockers: list[Blocker] = field(default_factory=list)
    relevant_resources: list[Resource] = field(default_factory=list)
    relevant_preferences: list[Preference] = field(default_factory=list)
    relevant_memory: list[str] = field(default_factory=list)
    #: app.context_engine.base.ContextWindow | None - forward reference,
    #: see this class's own docstring for why no import exists here.
    context_window: "ContextWindow | None" = None
    #: dict[str, app.context_engine.base.CategoryRetrieval] - forward
    #: reference, see this class's own docstring.
    retrieval_metadata: "dict[str, CategoryRetrieval]" = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Store interface
# ---------------------------------------------------------------------------


class WorldModelStore(ABC):
    """Persistent storage for every World Model entity and relationship.

    One store (not one per entity) for the whole module - the same
    judgment call app.orchestration.base.JobStore already makes for
    Job/Plan/PlanStep/JobEvent together, rather than a store per nested
    concept. Deliberately whole-object `save_*` upserts (mirroring
    JobStore.save) rather than narrow per-field setters: callers
    (WorldModelService) load an entity, mutate the in-memory dataclass,
    then save it back as one cohesive write - the same rationale
    JobStore's own docstring gives for why a Job's state changes as one
    unit rather than through a dozen narrow methods.

    This class does NOT validate transitions, confidence, or foreign-key
    existence - that's WorldModelService's job (app/world_model/service.py).
    A store implementation only needs to persist and return exactly what
    it's given/asked for.
    """

    # -- Project --------------------------------------------------------

    @abstractmethod
    def create_project(self, project: Project) -> None:
        """Persist a new Project. `project.id` must not already exist."""

    @abstractmethod
    def get_project(self, project_id: str) -> Project | None:
        """Look up a Project by id, or None if it doesn't exist."""

    @abstractmethod
    def list_projects(self, status: ProjectStatus | None = None) -> list[Project]:
        """Return Projects, optionally filtered to one status, most
        recently created first."""

    @abstractmethod
    def save_project(self, project: Project) -> None:
        """Persist every mutable field of an existing Project."""

    # -- Goal -------------------------------------------------------------

    @abstractmethod
    def create_goal(self, goal: Goal) -> None:
        """Persist a new Goal. `goal.id` must not already exist."""

    @abstractmethod
    def get_goal(self, goal_id: str) -> Goal | None:
        """Look up a Goal by id, or None if it doesn't exist."""

    @abstractmethod
    def list_goals(
        self, project_id: str | None = None, status: GoalStatus | None = None
    ) -> list[Goal]:
        """Return Goals, optionally filtered by project and/or status,
        most recently created first."""

    @abstractmethod
    def save_goal(self, goal: Goal) -> None:
        """Persist every mutable field of an existing Goal."""

    # -- Commitment ---------------------------------------------------------

    @abstractmethod
    def create_commitment(self, commitment: Commitment) -> None:
        """Persist a new Commitment. `commitment.id` must not already exist."""

    @abstractmethod
    def get_commitment(self, commitment_id: str) -> Commitment | None:
        """Look up a Commitment by id, or None if it doesn't exist."""

    @abstractmethod
    def list_commitments(
        self, project_id: str | None = None, status: CommitmentStatus | None = None
    ) -> list[Commitment]:
        """Return Commitments, optionally filtered by project and/or
        status, most recently created first."""

    @abstractmethod
    def save_commitment(self, commitment: Commitment) -> None:
        """Persist every mutable field of an existing Commitment."""

    # -- Event ------------------------------------------------------------

    @abstractmethod
    def create_event(self, event: Event) -> None:
        """Persist a new Event. `event.id` must not already exist."""

    @abstractmethod
    def get_event(self, event_id: str) -> Event | None:
        """Look up an Event by id, or None if it doesn't exist."""

    @abstractmethod
    def list_events(self, project_id: str | None = None) -> list[Event]:
        """Return Events, optionally filtered by project, soonest
        `starts_at` first."""

    @abstractmethod
    def save_event(self, event: Event) -> None:
        """Persist every mutable field of an existing Event."""

    # -- Resource -----------------------------------------------------------

    @abstractmethod
    def create_resource(self, resource: Resource) -> None:
        """Persist a new Resource. `resource.id` must not already exist."""

    @abstractmethod
    def get_resource(self, resource_id: str) -> Resource | None:
        """Look up a Resource by id, or None if it doesn't exist."""

    @abstractmethod
    def list_resources(
        self, type: ResourceType | None = None, status: ResourceStatus | None = None
    ) -> list[Resource]:
        """Return Resources, optionally filtered by type and/or status,
        most recently created first."""

    @abstractmethod
    def save_resource(self, resource: Resource) -> None:
        """Persist every mutable field of an existing Resource."""

    # -- Blocker ------------------------------------------------------------

    @abstractmethod
    def create_blocker(self, blocker: Blocker) -> None:
        """Persist a new Blocker. `blocker.id` must not already exist."""

    @abstractmethod
    def get_blocker(self, blocker_id: str) -> Blocker | None:
        """Look up a Blocker by id, or None if it doesn't exist."""

    @abstractmethod
    def list_blockers(self, status: BlockerStatus | None = None) -> list[Blocker]:
        """Return Blockers, optionally filtered to one status, most
        recently created first."""

    @abstractmethod
    def save_blocker(self, blocker: Blocker) -> None:
        """Persist every mutable field of an existing Blocker."""

    # -- Preference ---------------------------------------------------------

    @abstractmethod
    def create_preference(self, preference: Preference) -> None:
        """Persist a new Preference. `preference.id` must not already exist."""

    @abstractmethod
    def get_preference(self, preference_id: str) -> Preference | None:
        """Look up a Preference by id, or None if it doesn't exist."""

    @abstractmethod
    def list_preferences(
        self, key: str | None = None, category: str | None = None
    ) -> list[Preference]:
        """Return Preferences, optionally filtered by key and/or
        category, most recently updated first."""

    @abstractmethod
    def save_preference(self, preference: Preference) -> None:
        """Persist every mutable field of an existing Preference."""

    # -- Relationships: TaskLink ----------------------------------------

    @abstractmethod
    def create_task_link(self, link: TaskLink) -> None:
        """Persist a new TaskLink. `link.id` must not already exist."""

    @abstractmethod
    def get_task_link(self, link_id: str) -> TaskLink | None:
        """Look up a TaskLink by id, or None if it doesn't exist."""

    @abstractmethod
    def list_task_links(
        self, project_id: str | None = None, goal_id: str | None = None
    ) -> list[TaskLink]:
        """Return TaskLinks, optionally filtered by project and/or goal."""

    @abstractmethod
    def delete_task_link(self, link_id: str) -> bool:
        """Remove a TaskLink by id. Returns False if it didn't exist."""

    # -- Relationships: GoalDependency ------------------------------------

    @abstractmethod
    def create_goal_dependency(self, dependency: GoalDependency) -> None:
        """Persist a new GoalDependency. `dependency.id` must not already
        exist."""

    @abstractmethod
    def list_goal_dependencies(self, goal_id: str) -> list[GoalDependency]:
        """Return every GoalDependency where `goal_id` is the dependent
        (i.e. goals `goal_id` itself depends on)."""

    @abstractmethod
    def list_goal_dependents(self, goal_id: str) -> list[GoalDependency]:
        """Return every GoalDependency where `goal_id` is the one being
        depended on (i.e. goals that depend on `goal_id`)."""

    @abstractmethod
    def delete_goal_dependency(self, dependency_id: str) -> bool:
        """Remove a GoalDependency by id. Returns False if it didn't
        exist."""

    # -- Relationships: BlockerLink ---------------------------------------

    @abstractmethod
    def create_blocker_link(self, link: BlockerLink) -> None:
        """Persist a new BlockerLink. `link.id` must not already exist."""

    @abstractmethod
    def list_blocker_links_for_blocker(self, blocker_id: str) -> list[BlockerLink]:
        """Return every BlockerLink for one Blocker (every entity it
        affects)."""

    @abstractmethod
    def list_blocker_links_for_entity(
        self, entity_kind: LinkedEntityKind, entity_id: str
    ) -> list[BlockerLink]:
        """Return every BlockerLink naming this (kind, id) as the
        affected entity (every Blocker affecting it)."""

    @abstractmethod
    def delete_blocker_link(self, link_id: str) -> bool:
        """Remove a BlockerLink by id. Returns False if it didn't exist."""

    # -- Relationships: ResourceLink ---------------------------------------

    @abstractmethod
    def create_resource_link(self, link: ResourceLink) -> None:
        """Persist a new ResourceLink. `link.id` must not already exist."""

    @abstractmethod
    def list_resource_links_for_resource(self, resource_id: str) -> list[ResourceLink]:
        """Return every ResourceLink for one Resource (every entity it's
        attached to)."""

    @abstractmethod
    def list_resource_links_for_entity(
        self, entity_kind: LinkedEntityKind, entity_id: str
    ) -> list[ResourceLink]:
        """Return every ResourceLink naming this (kind, id) as the
        attached-to entity (every Resource attached to it)."""

    @abstractmethod
    def delete_resource_link(self, link_id: str) -> bool:
        """Remove a ResourceLink by id. Returns False if it didn't exist."""
