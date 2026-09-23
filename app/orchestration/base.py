"""
Orchestration abstraction (M15): a persistent, goal-driven Job - "the
user wants X; plan how to get there using existing tools; execute the
plan; track progress; recover from failure" - built entirely on top of
architecture M1-M14 already established, not a parallel system.

Deliberately NOT the same thing as app.tasks.Task ("Rhasspy Tasks", M9):
that's a plain user-facing todo item with no plan, no steps, and no
execution of its own - just create/list/complete/cancel free text. A Job
here is the opposite: it has no meaning to a user as a standalone note,
only as "work Jarvis is doing/has done toward a goal." Keeping the two
independent avoids Job inheriting Task's simple-todo semantics it
doesn't need, and avoids Task inheriting Job's plan/execution complexity
it doesn't need - the same reasoning that already keeps Reminder and
Task apart (see app/tasks/base.py's own module docstring). A Job is also
NOT an Automation (app/automation/base.py): an Automation's action is a
single, fixed ToolCall chosen once at creation time and re-run
unattended on a trigger; a Job's plan is proposed by an AI planner at
run time, can consist of many dependent steps, and runs once (subject to
retry/replan) rather than repeatedly.

Executor (app/orchestration/engine.py:JobEngine), Planner
(app/orchestration/planner.py), and Plan Validator
(app/orchestration/validator.py) all depend only on the types defined
here, never on SQLite directly - same "Core/Registry never knows the
storage engine" split as MemoryStore/ReminderStore/AutomationStore/
TaskStore.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from app.tools.base import PermissionLevel


class AutonomyLevel(Enum):
    """How much Jarvis is trusted to act on a Job without asking first.

    MANUAL - planning is allowed, but any step whose tool isn't
      automatic (see app.tools.base.AUTOMATIC_PERMISSIONS) requires
      confirmation - "meaningful execution requires confirmation."
    ASSISTED (the default) - READ/SAFE_ACTION steps run automatically;
      CONFIRM/DANGEROUS steps require confirmation. Identical to MANUAL
      for step-gating purposes today (both require confirmation for any
      non-automatic step) - the distinction exists for future milestones
      that give MANUAL stricter behavior (e.g. confirming automatic
      steps too) without disturbing ASSISTED's contract.
    AUTONOMOUS - a step whose permission is in this Job's own
      `autonomous_permissions` (see below) runs automatically; anything
      else still requires confirmation. DANGEROUS steps ALWAYS require
      confirmation regardless of autonomy or configuration - irreversible/
      dangerous actions never bypass the existing permission architecture
      (app/tools/base.py), no matter how much autonomy a Job is granted.
      See JobEngine._needs_confirmation and `parse_autonomous_permissions`
      below for the structural (not just conventional) enforcement of
      that boundary.
    """

    MANUAL = "manual"
    ASSISTED = "assisted"
    AUTONOMOUS = "autonomous"


_VALID_AUTONOMY_VALUES = {member.value for member in AutonomyLevel}


class AutonomyError(ValueError):
    """Raised by `parse_autonomy` for an unrecognized level."""


def parse_autonomy(value: str) -> AutonomyLevel:
    """Validate and construct an AutonomyLevel from a raw tool parameter.
    Never silently coerced or defaulted - callers (CreateJobTool) reject
    the request instead."""
    if value not in _VALID_AUTONOMY_VALUES:
        raise AutonomyError(
            f"Unknown autonomy level: '{value}'. Must be one of: "
            f"{sorted(_VALID_AUTONOMY_VALUES)}"
        )
    return AutonomyLevel(value)


#: The only permission level a Job's `autonomous_permissions` can ever
#: name. READ/SAFE_ACTION are excluded because they're already
#: unconditionally automatic (app.tools.base.AUTOMATIC_PERMISSIONS)
#: regardless of autonomy or this configuration - listing them here would
#: be a no-op that implies more control than actually exists. DANGEROUS
#: is deliberately, permanently excluded: no per-Job configuration can
#: ever add it, at any autonomy level. That boundary is enforced twice -
#: structurally, here, so a caller can never even construct a value
#: containing it (see `parse_autonomous_permissions`), and again in
#: JobEngine._needs_confirmation, which treats DANGEROUS as always
#: requiring confirmation regardless of what a Job's configuration says.
#: A genuinely different rule for some future class of action (e.g. a
#: new PermissionLevel a later milestone adds) would need its own
#: deliberate change to the existing M5 permission architecture itself -
#: never a per-Job setting quietly widening what DANGEROUS means today.
_CONFIGURABLE_AUTONOMOUS_PERMISSION_VALUES = frozenset({PermissionLevel.CONFIRM.value})

#: Preserves M15's original AUTONOMOUS behavior (auto-run CONFIRM, never
#: DANGEROUS) as the default whenever a Job doesn't configure anything
#: different - existing callers/tests that never mention this setting see
#: byte-for-byte the same gating decisions as before this was made
#: configurable.
DEFAULT_AUTONOMOUS_PERMISSIONS: frozenset[str] = frozenset({PermissionLevel.CONFIRM.value})


class AutonomousPermissionError(ValueError):
    """Raised by `parse_autonomous_permissions` for a value outside
    `_CONFIGURABLE_AUTONOMOUS_PERMISSION_VALUES` - in particular, for any
    attempt to include "dangerous". Never silently dropped or clamped."""


def parse_autonomous_permissions(values: list[str] | None) -> frozenset[str]:
    """Validate and construct the set of permission levels this Job's
    AUTONOMOUS autonomy (if any) may auto-run without confirmation.
    `None` means "not configured" - returns `DEFAULT_AUTONOMOUS_PERMISSIONS`,
    not an empty set, so an unconfigured Job keeps today's AUTONOMOUS
    behavior. An explicit empty list is different from `None`: it means
    "AUTONOMOUS should behave like ASSISTED for this Job" - a real,
    intentional per-Job opt-out, not an error."""
    if values is None:
        return DEFAULT_AUTONOMOUS_PERMISSIONS
    parsed = frozenset(values)
    invalid = parsed - _CONFIGURABLE_AUTONOMOUS_PERMISSION_VALUES
    if invalid:
        raise AutonomousPermissionError(
            f"Invalid autonomous permission(s): {sorted(invalid)}. Must be a "
            f"subset of {sorted(_CONFIGURABLE_AUTONOMOUS_PERMISSION_VALUES)} - "
            "'dangerous' can never be auto-approved, regardless of autonomy "
            "or configuration."
        )
    return parsed


class JobPriority(Enum):
    """A simple, fixed three-level priority - display/ordering only,
    mirrors app.tasks.base.TaskPriority's three values but is its own
    independent enum (see this module's docstring for why Job stays
    independent of Task)."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


_VALID_PRIORITY_VALUES = {member.value for member in JobPriority}


class JobPriorityError(ValueError):
    """Raised by `parse_priority` for an unrecognized priority."""


def parse_priority(value: str) -> JobPriority:
    if value not in _VALID_PRIORITY_VALUES:
        raise JobPriorityError(
            f"Unknown priority: '{value}'. Must be one of: "
            f"{sorted(_VALID_PRIORITY_VALUES)}"
        )
    return JobPriority(value)


class JobStatus(Enum):
    """Lifecycle state of a Job - exactly the eight states M15 requires.

    PENDING - created, not yet planned.
    PLANNING - a plan is currently being requested/validated. Never
      persisted across a poll in normal operation (planning is
      synchronous within one JobEngine tick); only ever observed
      mid-crash, in which case the next tick safely retries planning
      from scratch (see JobEngine._process) - planning has no side
      effects of its own.
    AWAITING_CONFIRMATION - one or more ready-to-run steps need
      confirmation before they can proceed (see JobEngine._needs_
      confirmation); JobEngine takes no further action on this Job until
      ConfirmJobTool resolves it.
    RUNNING - being actively advanced by JobEngine (planned, and no
      steps are currently blocked on confirmation).
    PAUSED - explicitly paused by the user; JobEngine skips it entirely
      until resumed.
    FAILED - terminal; a step failed in a way retries/replanning
      couldn't recover from, confirmation was declined, or planning/
      validation itself failed. `error` explains why.
    COMPLETED - terminal; every step succeeded (or was safely skipped
      because a sibling branch failed without affecting it). `final_
      result` summarizes the outcome.
    CANCELLED - terminal; the user explicitly cancelled it.
    """

    PENDING = "pending"
    PLANNING = "planning"
    AWAITING_CONFIRMATION = "awaiting_confirmation"
    RUNNING = "running"
    PAUSED = "paused"
    FAILED = "failed"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


#: Terminal states a Job can never leave - excluded from JobEngine's
#: active-work queries, same convention as TERMINAL_TASK_STATUSES/
#: TERMINAL_AUTOMATION_STATUSES/TERMINAL_STATUSES (reminders).
TERMINAL_JOB_STATUSES = frozenset(
    {JobStatus.FAILED, JobStatus.COMPLETED, JobStatus.CANCELLED}
)


class StepStatus(Enum):
    """Lifecycle state of one PlanStep."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"  # a dependency failed/was skipped; this step never ran


#: Statuses that make a step's own contribution to the plan "done" for
#: dependency-resolution purposes - SUCCEEDED unblocks dependents, FAILED/
#: SKIPPED cascade a skip to dependents (see JobEngine._advance).
TERMINAL_STEP_STATUSES = frozenset(
    {StepStatus.SUCCEEDED, StepStatus.FAILED, StepStatus.SKIPPED}
)


@dataclass
class RetryPolicy:
    """How many times a step is attempted before it's considered
    permanently failed, and how long to wait between attempts.

    `max_attempts=1` means no retry - the same "least demanding default"
    discipline RequestRequirements (app/providers/requirements.py) uses,
    though the actual default applied when a planner doesn't specify one
    is settings.orchestration_max_retries_per_step (see
    app/orchestration/validator.py), not this dataclass's own default.
    """

    max_attempts: int = 1
    delay_seconds: float = 0.0


@dataclass
class PlanStep:
    """One executable unit of a Plan - a single registered tool call.

    `id` is a plan-local identifier (e.g. "step_1"), not a database
    primary key - used only to express `depends_on` edges and to look a
    step up when a batch of steps is approved via ConfirmJobTool.

    `permission` is filled in from the tool's own PermissionLevel at
    validation time (app/orchestration/validator.py) - never trusted
    from planner output - and is what JobEngine gates confirmation on.

    `confirmed` is set True only by an explicit ConfirmJobTool approval
    for this exact step; it's the one piece of state that lets a
    previously-gated CONFIRM/DANGEROUS step actually run once approved
    (see JobEngine._needs_confirmation/_confirmed_value). It never
    resets itself and is never set by anything but that explicit
    approval - autonomy alone can make a CONFIRM step skip confirmation
    (see AutonomyLevel.AUTONOMOUS), but never sets this flag.

    `result` is a small JSON-safe summary of the ToolResult the step
    produced (status/message/data), never the live ToolResult object -
    see app/orchestration/engine.py:_serialize_tool_result.
    """

    id: str
    tool_name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    depends_on: list[str] = field(default_factory=list)
    permission: str | None = None  # PermissionLevel.value, filled by the validator
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)
    status: StepStatus = StepStatus.PENDING
    attempts: int = 0
    confirmed: bool = False
    result: dict[str, Any] | None = None
    error: str | None = None


@dataclass
class Plan:
    """An ordered, validated collection of PlanSteps - never constructed
    directly from raw planner output (see app/orchestration/validator.py:
    validate_plan, the only place a Plan is built)."""

    steps: list[PlanStep] = field(default_factory=list)

    def get(self, step_id: str) -> PlanStep | None:
        for step in self.steps:
            if step.id == step_id:
                return step
        return None


@dataclass
class JobEvent:
    """One entry in a Job's audit trail - timestamped, human-readable,
    and step-scoped when relevant. Purely observational (never read by
    any control-flow logic) - exists so a future dashboard/UI, or a
    human debugging a Job, has a full timeline without replaying the
    Plan's own current-state fields. See JobStore's own docstring for
    why this is stored as part of the Job rather than a separate table.
    """

    timestamp: str
    kind: str
    detail: str = ""
    step_id: str | None = None


@dataclass
class Job:
    """A single persisted goal-driven task.

    `plan` is None until planning succeeds at least once. `context` is a
    small, JSON-safe snapshot the planner is given alongside the goal
    (e.g. known user facts) - built once at creation time (CreateJobTool),
    the same "compute shared context once, pass it down" shape
    ConditionContext (app/automation/base.py) already uses.

    `total_step_count` is the cumulative number of steps ever added to
    this Job's plan across the original plan and every replan - the
    running total JobEngine checks against
    settings.orchestration_max_total_steps, so replanning can't be used
    to bypass the safety limit one plan at a time. `replan_count` is how
    many times replanning has actually happened, checked against
    settings.orchestration_max_replans.

    `pending_step_ids`/`confirmation_message` are only meaningful while
    `status` is AWAITING_CONFIRMATION - the current grouped batch of
    ready steps waiting on ConfirmJobTool, and the human-readable summary
    JobEngine built for them (see JobEngine._build_confirmation_message).

    `final_result`/`error` are mutually exclusive in practice:
    `final_result` is set only when `status` becomes COMPLETED;
    `error` is set only when it becomes FAILED. Neither is ever set to
    claim an outcome that didn't actually happen - see JobEngine's own
    module docstring for the "never claim success when an action failed"
    discipline this whole module exists to uphold.

    `conversation_id` (added after M15's initial pass) links a Job back
    to the conversation it was created from - not for access control (this
    is a single-user assistant), but so a natural-language status query
    or confirmation reply ("is that finished?", "yes", "send it") can be
    resolved against "the task from THIS conversation" first, before
    falling back to considering every Job (see
    JarvisCore._handle_task_status/_handle_confirmation, app/core.py).
    `None` for a Job created without one (e.g. directly via a test, or a
    future non-chat entry point).

    `autonomous_permissions` is the configurable boundary
    AutonomyLevel.AUTONOMOUS actually checks (see
    `parse_autonomous_permissions` above) - a genuine per-Job setting,
    not a hardcoded engine constant. Meaningless for MANUAL/ASSISTED
    (they never consult it).
    """

    id: str
    goal: str
    status: JobStatus
    priority: JobPriority
    autonomy: AutonomyLevel
    created_at: str
    updated_at: str
    plan: Plan | None = None
    context: dict[str, Any] = field(default_factory=dict)
    replan_count: int = 0
    total_step_count: int = 0
    pending_step_ids: list[str] = field(default_factory=list)
    confirmation_message: str | None = None
    final_result: str | None = None
    error: str | None = None
    events: list[JobEvent] = field(default_factory=list)
    conversation_id: str | None = None
    autonomous_permissions: frozenset[str] = field(
        default_factory=lambda: DEFAULT_AUTONOMOUS_PERMISSIONS
    )


class JobStore(ABC):
    """Persistent storage for Jobs.

    Deliberately a single `save(job)` upsert rather than the narrow,
    per-transition methods ReminderStore/AutomationStore/TaskStore each
    use (record_trigger/complete/enable/...): a Job's plan/step state
    changes as one cohesive unit on every JobEngine tick (a step's
    status+result+attempts, possibly the Job's own status, possibly
    pending_step_ids, all together) - narrow methods here would multiply
    into a dozen methods used only by JobEngine's own internal
    bookkeeping, for no benefit a whole-object save doesn't already give.
    `id`/`goal`/`created_at` are immutable after `create`; every other
    field is whatever `save` last wrote.
    """

    @abstractmethod
    def create(
        self,
        goal: str,
        priority: JobPriority = JobPriority.MEDIUM,
        autonomy: AutonomyLevel = AutonomyLevel.ASSISTED,
        context: dict[str, Any] | None = None,
        conversation_id: str | None = None,
        autonomous_permissions: frozenset[str] | None = None,
    ) -> Job:
        """Create and persist a new PENDING Job with no plan yet.
        `autonomous_permissions=None` means "use the default" (see
        `parse_autonomous_permissions`), matching every other optional
        field's "unset means the ordinary default" convention here."""

    @abstractmethod
    def get(self, job_id: str) -> Job | None:
        """Look up a Job by ID, or None if it doesn't exist."""

    @abstractmethod
    def list(self, status: JobStatus | None = None) -> list[Job]:
        """Return Jobs, optionally filtered to one status, most recently
        created first."""

    @abstractmethod
    def save(self, job: Job) -> None:
        """Persist every mutable field of `job` (see class docstring).
        `job.updated_at` is set by the caller before calling this - the
        store never invents a timestamp on save, so JobEngine's own
        event timestamps and `updated_at` always agree."""
