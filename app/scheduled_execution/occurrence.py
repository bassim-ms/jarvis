"""
A Scheduled Occurrence: one due occurrence of one approved Scheduled Job, already DECIDED.

It is built from the Slice 1 policy's `DueDecision` (`decide_due`), so the decisions about timing - on time, the one catch-up run after
a missed occurrence, where the schedule goes next, whether it ends - are made by that pure policy and only carried here. An
occurrence exists only for the two decisions that create a Job (RUN, RUN_LATE); a skip or a lifetime end creates nothing, so there is
no occurrence to build for them.

`occurrence` is the deterministic identity (`app.scheduling.runs.occurrence_key`): derived from the Scheduled Job id and the UTC
instant the occurrence resolved to, never from wall-clock text. The same occurrence therefore always has the same key, which is what the
ledger's UNIQUE constraint keys on.

Pure: no clock, no I/O, no storage.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from app.scheduled_jobs.base import RunKind, RunOutcome, ScheduledJob, ScheduledJobError, ScheduledJobStatus, StatusReason
from app.scheduled_jobs.policy import DueAction, DueDecision
from app.scheduling.runs import occurrence_key


def manual_occurrence_key(scheduled_job_id: str, version: int) -> str:
    occurrence_key(scheduled_job_id, datetime(2000, 1, 1, tzinfo=timezone.utc))  # the same id rules as every occurrence key
    return f"{scheduled_job_id}@manual:v{int(version)}"


@dataclass(frozen=True)
class ScheduledOccurrence:
    scheduled_job_id: str
    occurrence: str  # the deterministic identity
    scheduled_for: datetime  # the instant the occurrence resolved to (UTC)
    outcome: RunOutcome  # CREATED (on time) or CREATED_LATE (the one catch-up run)
    expected_due_at: datetime  # the `due_at` the Scheduled Job had when this was decided (the compare-and-set)
    next_due_at: datetime | None  # where the schedule goes next; None when this occurrence ends it
    complete_reason: StatusReason | None = None  # set exactly when there is no next run
    missed_count: int = 0
    kind: RunKind = RunKind.SCHEDULED

    @property
    def late(self) -> bool:
        return self.outcome is RunOutcome.CREATED_LATE

    @classmethod
    def manual(cls, scheduled_job: ScheduledJob, now: datetime) -> "ScheduledOccurrence":
        """A "run it now" request for an ACTIVE Scheduled Job. It is the ledger's own MANUAL kind (Slice 1): same checks, same launch, but it
        does not move the schedule - no next due time, no completion - so the next REAL occurrence is untouched and nothing is consumed.

        Its identity is `<id>@manual:v<version>`: derived from the Scheduled Job's version as it was READ. Two requests that read the same
        state share one identity, so the ledger's UNIQUE constraint (and the compare-and-set on that same version) lets exactly one launch
        - the other is a duplicate and writes nothing. Every launch bumps the version, so a LATER request gets a new identity; and it can
        never equal a scheduled occurrence's key (`<id>@<instant>`). It is not a historical occurrence: `scheduled_for` is simply now."""
        if scheduled_job.status is not ScheduledJobStatus.ACTIVE or scheduled_job.due_at is None:
            raise ScheduledJobError("Only an active Scheduled Job can be run now.")
        return cls(
            scheduled_job_id=scheduled_job.id,
            occurrence=manual_occurrence_key(scheduled_job.id, scheduled_job.version),
            scheduled_for=now,
            outcome=RunOutcome.CREATED,
            expected_due_at=scheduled_job.due_at,
            next_due_at=None,
            complete_reason=None,
            kind=RunKind.MANUAL,
        )

    @classmethod
    def from_decision(cls, scheduled_job: ScheduledJob, decision: DueDecision) -> "ScheduledOccurrence":
        """The occurrence a RUN / RUN_LATE decision stands for. Any other decision (not due, skipped, lifetime ended) creates no
        Job, so it is refused here rather than turned into one."""
        if decision.action not in (DueAction.RUN, DueAction.RUN_LATE):
            raise ScheduledJobError(f"A '{decision.action.value}' decision creates no Job, so it has no occurrence to run.")
        if scheduled_job.status is not ScheduledJobStatus.ACTIVE or scheduled_job.due_at is None:
            raise ScheduledJobError("Only an active Scheduled Job has occurrences.")
        if (decision.next_due is None) != (decision.complete_reason is not None):
            raise ScheduledJobError("A decision either has a next run or ends the Scheduled Job - exactly one of the two.")
        return cls(
            scheduled_job_id=scheduled_job.id,
            occurrence=occurrence_key(scheduled_job.id, decision.occurrence_at),
            scheduled_for=decision.occurrence_at,
            outcome=RunOutcome.CREATED if decision.action is DueAction.RUN else RunOutcome.CREATED_LATE,
            expected_due_at=scheduled_job.due_at,
            next_due_at=decision.next_due,
            complete_reason=decision.complete_reason,
            missed_count=decision.missed_count,
        )
