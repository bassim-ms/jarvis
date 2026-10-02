"""
Scheduled Jobs (V1, Slice 1 - the FOUNDATION): the model, the state machine and the storage contract. Nothing here runs anything.

A Scheduled Job is a standing, APPROVED INTENT plus a Schedule. It is not a Job: it owns no plan, runs no tool and never calls a
Planner. Each time one of its occurrences comes due, a later slice creates one ordinary, fresh Job from it (the existing JobEngine
plans and runs that Job). So the concepts stay distinct:

    Task          something the user wants done (a passive to-do)          app/tasks
    Reminder      tell the user something at a time                         app/reminders
    Schedule      WHEN something should happen (a pure value)               app/scheduling
    Job           ONE execution instance: plan, steps, a terminal state     app/orchestration
    Scheduled Job a Job created from a Schedule, once per occurrence        THIS package (the standing order, not the run)
    Automation    one fixed, pre-validated tool call on a trigger           app/automation (untouched)

This module is data and rules only: it imports no tool, no planner, no engine, no notification, no model, no World Model. The
storage contract (`ScheduledJobStore`) is implemented in sqlite_store.py; the pure decisions (missed runs, limits, the circuit
breaker) are in policy.py.

LIFECYCLE (terminal states are final; every transition is optimistic-concurrency guarded, see the store):

    PROPOSED --approve--> ACTIVE <--pause/resume--> PAUSED
       |                    |  \\                     |
       | decline/expire     |   \\--> COMPLETED       +--> COMPLETED / CANCELLED
       v                    v        (reason below)
    CANCELLED / EXPIRED   CANCELLED

    PROPOSED  stored and waiting for the user's explicit approval. No Job exists, nothing can run. Expires on its own.
    ACTIVE    approved; its occurrences may create Jobs. The 90-day V1 lifetime starts at approval (`expires_at`).
    PAUSED    no occurrence creates a Job. Why is in `status_reason`: the user, the circuit breaker, a revoked creating device, or a
              capability that is no longer available. Resuming goes to the next FUTURE occurrence - nothing missed is replayed.
    COMPLETED a one-time schedule ran (or was missed), or the lifetime ended. Renewal is a fresh explicit approval (a later slice).
    CANCELLED the user dropped it (or declined the proposal).
    EXPIRED   a proposal nobody approved in time.

THE LEDGER. Every occurrence that is acted on gets exactly one `ScheduledJobRun` row. `UNIQUE(schedule_id, occurrence)` is the
authority for "one occurrence -> at most one Job": the row and the Job it names are written in ONE transaction (see the store's
`record_run`), so a duplicate poll, a second process or a restart cannot create a second Job for the same occurrence.

APPROVAL GRANTS AN INTENT, NOT AUTHORITY. Nothing stored here widens what a Job may do: the capability profile is re-derived from
the live ToolRegistry on every run (a later slice), and `profile` only NAMES the (reduce-only) profile the intent was approved under.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from app.scheduling.schedule import Schedule

# --- bounds (one place, easy to change) -----------------------------------------------------------------------------------------

MAX_TITLE_CHARS = 120
MAX_INTENT_CHARS = 2000
#: The serialised size of a stored objective (the structured understanding of the request), as for a Job's own handoff.
MAX_OBJECTIVE_CHARS = 8000
MAX_EVENTS = 50
MAX_EVENT_DETAIL_CHARS = 200
MAX_REASON_DETAIL_CHARS = 200
MAX_ID_CHARS = 128

#: The one (reduce-only) capability profile V1 defines. The profile's tool list is defined with the unattended Job profile in a later
#: slice; here it is only a NAME that a stored intent was approved under. Unknown names are refused - fail closed.
PROFILE_UNATTENDED_V1 = "unattended_v1"
KNOWN_PROFILES = frozenset({PROFILE_UNATTENDED_V1})

#: `via` identifies the actor/channel of a transition ("chat", "scheduler", ...) - deliberately not free text.
_VIA_PATTERN = re.compile(r"^[a-z0-9_.:-]{1,50}$")


class ScheduledJobError(ValueError):
    """Base class: a stored Scheduled Job or ledger row that is malformed, or an operation that cannot be honoured. Messages are
    safe to log; user-facing wording is the caller's."""


class InvalidTransitionError(ScheduledJobError):
    """The requested status change is not an allowed move from the current status (or the reason does not fit it). Nothing changed."""


class StaleScheduledJobError(ScheduledJobError):
    """The row changed after the caller read it (or no longer exists). The write was NOT applied; re-read and decide again."""


class DuplicateOccurrenceError(ScheduledJobError):
    """This occurrence of this Scheduled Job was already acted on. Nothing was written - in particular no second Job exists."""


class ScheduledJobStatus(Enum):
    PROPOSED = "proposed"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


TERMINAL_STATUSES = frozenset(
    {ScheduledJobStatus.COMPLETED, ScheduledJobStatus.CANCELLED, ScheduledJobStatus.EXPIRED}
)
#: Statuses that count against the "at most N Scheduled Jobs" cap: a proposal, a live one and a paused one all still exist.
OPEN_STATUSES = frozenset(
    {ScheduledJobStatus.PROPOSED, ScheduledJobStatus.ACTIVE, ScheduledJobStatus.PAUSED}
)


class StatusReason(Enum):
    """Why a Scheduled Job is in a PAUSED / COMPLETED / CANCELLED / EXPIRED status. ACTIVE and PROPOSED carry none."""

    # PAUSED
    USER = "user"
    BREAKER = "breaker"  # too many consecutive failed occurrences
    DEVICE_REVOKED = "device_revoked"  # the device it was created from is no longer trusted
    BLOCKED = "blocked"  # a capability it needs is no longer available
    # COMPLETED
    ONE_TIME_DONE = "one_time_done"
    ONE_TIME_MISSED = "one_time_missed"
    LIFETIME_ENDED = "lifetime_ended"
    # CANCELLED (a proposal the user turned down; USER covers cancelling a live one)
    DECLINED = "declined"
    # EXPIRED
    PROPOSAL_EXPIRED = "proposal_expired"


_S, _R = ScheduledJobStatus, StatusReason

#: EVERY allowed move: current status -> {next status -> the reasons that fit it}. `frozenset({None})` means "no reason". A move that
#: is not here is refused (InvalidTransitionError), including any move out of a terminal status. PROPOSED -> ACTIVE and
#: PAUSED -> ACTIVE are listed for completeness; they are done by the store's `activate` / `resume`, which also set the fields that
#: go with them (approval, lifetime, the next due time).
ALLOWED_TRANSITIONS: dict[ScheduledJobStatus, dict[ScheduledJobStatus, frozenset]] = {
    _S.PROPOSED: {
        _S.ACTIVE: frozenset({None}),
        _S.CANCELLED: frozenset({_R.DECLINED}),
        _S.EXPIRED: frozenset({_R.PROPOSAL_EXPIRED}),
    },
    _S.ACTIVE: {
        _S.PAUSED: frozenset({_R.USER, _R.BREAKER, _R.DEVICE_REVOKED, _R.BLOCKED}),
        _S.COMPLETED: frozenset({_R.ONE_TIME_DONE, _R.ONE_TIME_MISSED, _R.LIFETIME_ENDED}),
        _S.CANCELLED: frozenset({_R.USER}),
    },
    _S.PAUSED: {
        _S.ACTIVE: frozenset({None}),
        _S.COMPLETED: frozenset({_R.LIFETIME_ENDED}),
        _S.CANCELLED: frozenset({_R.USER}),
    },
    _S.COMPLETED: {},
    _S.CANCELLED: {},
    _S.EXPIRED: {},
}


def check_transition(current: ScheduledJobStatus, target: ScheduledJobStatus, reason: StatusReason | None) -> None:
    """Raise InvalidTransitionError unless `current -> target` (with `reason`) is an allowed move."""
    allowed = ALLOWED_TRANSITIONS.get(current, {})
    if target not in allowed:
        raise InvalidTransitionError(f"A {current.value} Scheduled Job cannot become {target.value}.")
    if reason not in allowed[target]:
        shown = reason.value if reason is not None else "none"
        raise InvalidTransitionError(f"'{shown}' is not a valid reason for {current.value} -> {target.value}.")


class RunKind(Enum):
    SCHEDULED = "scheduled"  # the schedule's own occurrence
    MANUAL = "manual"  # "run it now": same checks, does not move the schedule


class RunOutcome(Enum):
    """What was done about one occurrence. Only the first two create a Job."""

    CREATED = "created"  # on time (within the grace window)
    CREATED_LATE = "created_late"  # the one catch-up run after a missed occurrence
    SKIPPED_MISSED = "skipped_missed"  # missed, and too late to catch up
    SKIPPED_OVERLAP = "skipped_overlap"  # the previous run of this Scheduled Job had not finished
    SKIPPED_BLOCKED = "skipped_blocked"  # a re-check failed (capability gone, device revoked, ...); `reason` says which


JOB_OUTCOMES = frozenset({RunOutcome.CREATED, RunOutcome.CREATED_LATE})

#: The only values a run's `result_state` takes: the terminal state of the Job it created, copied once it finished.
RESULT_COMPLETED = "completed"
RESULT_FAILED = "failed"  # includes a run that timed out
RESULT_CANCELLED = "cancelled"
RESULT_STATES = frozenset({RESULT_COMPLETED, RESULT_FAILED, RESULT_CANCELLED})


@dataclass(frozen=True)
class ScheduledJobEvent:
    """One entry in a Scheduled Job's audit trail: when, what, by which actor. Purely observational."""

    timestamp: str
    kind: str
    detail: str = ""
    via: str = "system"


@dataclass
class ScheduledJob:
    """A standing, approved intent plus its Schedule.

    `intent_text` is the user's own words for WHAT to do (without the schedule phrase) and `objective` the stable, validated
    understanding of it - never a snapshot of the world: context is built fresh when an occurrence runs. `title` is a short label in
    the user's words, used to refer to it later. `profile` names the reduce-only capability profile it was approved under.

    `schedule` carries its own time zone; `due_at` is the next occurrence as an aware UTC datetime (None until approved).
    `requested_until` is how long the user asked for it to go on ("for the next two years"); `expires_at` is what V1 actually
    activates - never later than approval + the V1 lifetime - so a longer request is understood and honoured for the first period,
    and renewal (fresh approval) is needed after it. One-time schedules have neither.

    `created_via_*` record where it came from, for the revocation re-check and for referring to it in conversation. Counters:
    `runs_total` Jobs created, `consecutive_failures` the current failure streak (the circuit breaker's input).

    `version` is the optimistic-concurrency counter, set by the store, never by callers.
    """

    id: str
    status: ScheduledJobStatus
    title: str
    intent_text: str
    objective: dict[str, Any]
    schedule: Schedule
    profile: str
    created_at: datetime
    updated_at: datetime
    status_reason: StatusReason | None = None
    due_at: datetime | None = None
    approved_at: datetime | None = None
    approval_interaction_id: str | None = None
    proposal_expires_at: datetime | None = None
    requested_until: datetime | None = None
    expires_at: datetime | None = None
    created_via_conversation_id: str | None = None
    created_via_device_id: str | None = None
    runs_total: int = 0
    consecutive_failures: int = 0
    last_job_id: str | None = None
    last_run_at: datetime | None = None
    events: list[ScheduledJobEvent] = field(default_factory=list)
    version: int = 0

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES


@dataclass(frozen=True)
class NewRun:
    """One occurrence about to be acted on, as the caller decides it (see the store's `record_run`).

    `occurrence` is the deterministic identity (app.scheduling.runs.occurrence_key); `scheduled_for` the occurrence's UTC instant.
    A run that creates a Job (CREATED / CREATED_LATE) must name that Job's id - and carry the Job to insert - and any other outcome
    must not: a skipped occurrence has no Job. `reason` says why for SKIPPED_BLOCKED."""

    occurrence: str
    scheduled_for: datetime
    outcome: RunOutcome
    kind: RunKind = RunKind.SCHEDULED
    job_id: str | None = None
    reason: str | None = None
    missed_count: int = 0
    late: bool = False


@dataclass(frozen=True)
class ScheduledJobRun:
    """The ledger row for one acted-on occurrence."""

    id: int
    schedule_id: str
    occurrence: str
    kind: RunKind
    scheduled_for: datetime
    outcome: RunOutcome
    job_id: str | None
    reason: str | None
    missed_count: int
    late: bool
    created_at: datetime
    result_state: str | None = None
    settled_at: datetime | None = None
    reported_at: datetime | None = None


@dataclass(frozen=True)
class Settled:
    """The outcome of settling a run (see the store's `settle_run`): the updated Scheduled Job, whether this call was the one that
    settled it, and whether it tripped the circuit breaker (and so paused the Scheduled Job)."""

    scheduled_job: ScheduledJob
    changed: bool
    tripped: bool


def valid_via(via: str) -> str:
    if not isinstance(via, str) or not _VIA_PATTERN.match(via):
        raise ScheduledJobError("`via` must be a short identifier such as 'chat' or 'scheduler'.")
    return via


class ScheduledJobStore(ABC):
    """Persistent storage for Scheduled Jobs and their occurrence ledger. Every mutation is a compare-and-set on `version`: it
    applies only if the row is exactly as the caller read it, and otherwise raises StaleScheduledJobError leaving it untouched."""

    @abstractmethod
    def create_proposal(
        self,
        *,
        title: str,
        intent_text: str,
        objective: dict[str, Any],
        schedule: Schedule,
        proposal_expires_at: datetime,
        requested_until: datetime | None = None,
        profile: str = PROFILE_UNATTENDED_V1,
        created_via_conversation_id: str | None = None,
        created_via_device_id: str | None = None,
        now: datetime | None = None,
    ) -> ScheduledJob:
        """Store a new PROPOSED Scheduled Job. Nothing can run from it until it is approved (`activate`)."""

    @abstractmethod
    def get(self, scheduled_job_id: str) -> ScheduledJob | None: ...

    @abstractmethod
    def list(self, statuses: frozenset[ScheduledJobStatus] | None = None) -> list[ScheduledJob]:
        """Most recently created first; optionally only the given statuses."""

    @abstractmethod
    def list_due(self, now: datetime, limit: int = 20) -> list[ScheduledJob]:
        """ACTIVE Scheduled Jobs whose `due_at` is at or before `now`, earliest first."""

    @abstractmethod
    def count(self, statuses: frozenset[ScheduledJobStatus]) -> int: ...

    @abstractmethod
    def activate(
        self,
        scheduled_job_id: str,
        *,
        expected_version: int,
        approved_at: datetime,
        approval_interaction_id: str,
        due_at: datetime,
        expires_at: datetime | None,
        via: str,
    ) -> ScheduledJob:
        """PROPOSED -> ACTIVE: record the approval, the first due time and the lifetime in one step."""

    @abstractmethod
    def transition(
        self,
        scheduled_job_id: str,
        target: ScheduledJobStatus,
        reason: StatusReason | None,
        *,
        expected_version: int,
        via: str,
        detail: str = "",
        now: datetime | None = None,
    ) -> ScheduledJob:
        """Any allowed move except approval and resume (`check_transition`)."""

    @abstractmethod
    def resume(self, scheduled_job_id: str, *, next_due_at: datetime, expected_version: int, via: str) -> ScheduledJob:
        """PAUSED -> ACTIVE at `next_due_at` (the caller computes the next FUTURE occurrence). Clears the failure streak when the
        pause was the circuit breaker's."""

    @abstractmethod
    def reschedule(
        self, scheduled_job_id: str, *, schedule: Schedule, due_at: datetime, expected_version: int, via: str
    ) -> ScheduledJob:
        """Change WHEN an ACTIVE or PAUSED Scheduled Job happens. It does not extend the lifetime and does not change what it does."""

    @abstractmethod
    def expire_proposals(self, now: datetime, via: str = "system") -> list[ScheduledJob]:
        """Move every PROPOSED Scheduled Job whose proposal time has passed to EXPIRED."""

    @abstractmethod
    def record_run(
        self,
        scheduled_job_id: str,
        run: NewRun,
        *,
        expected_version: int,
        expected_due_at: datetime | None,
        next_due_at: datetime | None,
        complete_reason: StatusReason | None = None,
        insert_job: Callable[[Any], None] | None = None,
        via: str = "scheduler",
        now: datetime | None = None,
    ) -> ScheduledJobRun:
        """Act on one occurrence ATOMICALLY: in a single transaction, (1) compare-and-set the Scheduled Job (its version, and for a
        scheduled run its `due_at`), advance it to `next_due_at` (or complete it with `complete_reason` when there is none),
        (2) insert the ledger row, (3) call `insert_job(connection)` to insert the Job the row names. Any failure - a stale row, a
        duplicate occurrence, an error inside `insert_job` - rolls back ALL of it. A MANUAL run does not move the schedule.

        `insert_job` receives the open connection WHILE the store's lock is held (the Job store's lock, when shared): it must only
        write through that connection - e.g. the Job store's static `insert_job_locked(connection, job)` - and must not call back into
        any store method that takes the lock (that would deadlock)."""

    @abstractmethod
    def get_run(self, schedule_id: str, occurrence: str) -> ScheduledJobRun | None: ...

    @abstractmethod
    def runs_for(self, schedule_id: str, limit: int = 50) -> list[ScheduledJobRun]:
        """Most recent first."""

    @abstractmethod
    def unsettled_runs(self, schedule_id: str | None = None) -> list[ScheduledJobRun]:
        """Runs that created a Job whose result has not been recorded yet (the Job may still be running)."""

    @abstractmethod
    def count_runs_since(self, since: datetime) -> int:
        """Jobs created (CREATED / CREATED_LATE) at or after `since`, across all Scheduled Jobs - the daily cap's input."""

    @abstractmethod
    def settle_run(
        self, run_id: int, result_state: str, *, breaker_limit: int, via: str = "scheduler", now: datetime | None = None
    ) -> Settled:
        """Record, once, how the Job a run created ended, update the failure streak, and pause the Scheduled Job (reason BREAKER)
        when the streak reaches `breaker_limit`. Idempotent: settling an already settled run changes nothing."""

    @abstractmethod
    def mark_reported(self, run_id: int, now: datetime | None = None) -> bool:
        """Record that the user was told about this run. True the first time, False after."""
