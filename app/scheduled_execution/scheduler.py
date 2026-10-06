"""
The Scheduled Jobs scheduler: decides, on each tick, which due occurrences become a Job - and settles the ones that finished.

    tick():   settle finished runs  ->  for each due Scheduled Job:  gates, in order  ->  ScheduledExecutionAdapter.launch

It runs inside the existing local runtime - it is hung on the JobEngine's own tick (`JobEngine.add_tick_hook`), so there is no second
loop, no thread, no timer and no external service. Jarvis running = it runs; Jarvis stopped = it does not. Nothing is held in memory
that matters: what is due, what already ran, what is still running and what failed are all read from the database on every tick, so
a restart or a crash loses nothing and a second tick (or a second process) only re-reads the same truth.

WHAT IT DOES NOT DO. It never executes a tool, plans, validates a plan, touches the ToolRegistry, writes memory, writes organizer
state, notifies, or enforces a deadline: it decides WHETHER an occurrence launches, the adapter makes the ordinary Job, and the existing
JobEngine plans and runs it (and owns its deadline, its cancellation and its security). The kill switch, overlap, breaker, cap and
device checks here are further REDUCTIONS - each can only stop a launch, none can widen anything.

THE GATES, in this order, for an occurrence that is due (the first that fails consumes the occurrence with ONE terminal settlement - a
skipped ledger row - and the Scheduled Job moves on to its next FUTURE occurrence; nothing is queued and nothing is replayed):

    1  kill switch             off                                   -> SKIPPED_DISABLED
    2  eligible and due        not ACTIVE / not due yet              -> nothing (not this tick's business)
    3  expiry                  the approved period is over           -> SKIPPED_EXPIRED (and the Scheduled Job completes)
       missed                  too late for the one catch-up run     -> SKIPPED_MISSED
    4  overlap                 the previous run's Job is in flight   -> SKIPPED_OVERLAP
    5  failure breaker         the streak is at the limit            -> the Scheduled Job is PAUSED (reason breaker), no run
    6  execution caps          THIS Scheduled Job's 24-hour cap      -> SKIPPED_CAP
                               the GLOBAL 24-hour cap (Slice 1)      -> stays due (bounded by the catch-up window)
                               the GLOBAL in-flight limit (Slice 1)  -> stays due (bounded by the catch-up window)
    7  device                  the creating device is not trusted    -> SKIPPED_BLOCKED (the Scheduled Job stays ACTIVE)
    8  capability / boundary   the adapter refuses (Slice 2)         -> SKIPPED_BLOCKED
    9  launch                  the adapter makes the Job and the ledger row in ONE transaction

IN FLIGHT. A run counts as in flight - for overlap and for the global limit - only while its Job is in one of the states the JobEngine
itself advances (pending, planning, running: its own `_ACTIVE_STATUSES`). A Job that is paused, or waiting for an answer or a
confirmation, is not being advanced and may never be (the engine skips it), so it must not block a Scheduled Job forever; a Job that is
finished is settled first. The run stays unsettled (and the Job untouched) until the Job actually ends, whenever that is.

THE APPROVED PERIOD NEVER STOPS WHILE PAUSED. Pausing does not extend a Scheduled Job's 90-day period. The tick completes a paused one
whose period has ended (COMPLETED, reason lifetime_ended - the same end an active one reaches), and the store refuses to resume it
after that: only a fresh approval can start a new period.

ONE OCCURRENCE -> AT MOST ONE LAUNCH -> AT MOST ONE TERMINAL SETTLEMENT is the database's guarantee, not this class's: the ledger's
UNIQUE(schedule, occurrence), the compare-and-set on the Scheduled Job's version and due time, and `settle_run`'s "only while unsettled"
update. Two ticks, two schedulers or two processes racing for the same occurrence therefore produce one winner, and the loser writes
nothing.

SETTLEMENT. A launched run is settled once, from the state its Job ended in (completed / failed - a timeout is a failure / cancelled),
and the failure streak follows it: a completed run resets it, a failed one extends it, a cancelled one (the user's own choice) is neutral.
A skipped occurrence is its own settlement (its ledger row is final) and is never an execution and never a failure.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta, timezone

from app.orchestration.base import TERMINAL_JOB_STATUSES, JobStatus
from app.scheduled_execution.adapter import ExecutionRefused, ScheduledExecutionAdapter
from app.scheduled_execution.occurrence import ScheduledOccurrence
from app.scheduled_jobs.base import (
    MAX_REASON_DETAIL_CHARS,
    RESULT_CANCELLED,
    RESULT_COMPLETED,
    RESULT_FAILED,
    DuplicateOccurrenceError,
    NewRun,
    RunOutcome,
    ScheduledJob,
    ScheduledJobRun,
    ScheduledJobStatus,
    ScheduledJobStore,
    StaleScheduledJobError,
    StatusReason,
)
from app.scheduled_jobs.policy import DAILY_WINDOW, Admission, DueAction, DueDecision, Limits, admit, decide_due
from app.scheduling.runs import occurrence_key
from app.scheduling.schedule import ScheduleKind

logger = logging.getLogger(__name__)

#: How many due Scheduled Jobs one tick looks at. Bounded: whatever is left is simply seen by the next tick.
DEFAULT_BATCH = 20

#: The state a Job ended in -> what the ledger records for the run it belongs to.
_RESULT_BY_JOB_STATUS = {
    JobStatus.COMPLETED: RESULT_COMPLETED,
    JobStatus.FAILED: RESULT_FAILED,
    JobStatus.CANCELLED: RESULT_CANCELLED,
}

#: The Job states in which a run is IN FLIGHT: exactly the states JobEngine advances (`app.orchestration.engine._ACTIVE_STATUSES`; a test
#: pins the two together, as the scheduler may not import the engine). Every other non-terminal state - paused, awaiting confirmation or
#: input - is a Job nobody is advancing, so it is not in flight.
IN_FLIGHT_JOB_STATUSES = frozenset({JobStatus.PENDING, JobStatus.PLANNING, JobStatus.RUNNING})

_MICROSECOND = timedelta(microseconds=1)


@dataclass
class TickResult:
    """What one tick did - for tests and logs only; nothing reads it to decide anything."""

    launched: int = 0
    settled: int = 0
    paused: int = 0
    deferred: int = 0
    expired: int = 0
    lost_races: int = 0
    errors: int = 0
    skipped: dict[str, int] = field(default_factory=dict)

    def count_skip(self, outcome: RunOutcome) -> None:
        self.skipped[outcome.value] = self.skipped.get(outcome.value, 0) + 1


class ManualRefusal(Enum):
    """Why a "run it now" request did not start a run. Every one of them wrote NOTHING: no ledger row, no Job, no change to the Scheduled
    Job - a refused request is not an occurrence."""

    NOT_FOUND = "not_found"
    NOT_ACTIVE = "not_active"  # paused (by the user or the breaker), finished, cancelled, or still a proposal
    ONE_TIME = "one_time"  # a one-time Scheduled Job runs ONCE, at its time: a manual run would make it run twice
    DISABLED = "disabled"  # the global kill switch is off
    EXPIRED = "expired"  # its approved period has ended
    OVERLAP = "overlap"  # its previous run is still in flight
    BREAKER = "breaker"  # its failure streak is at the limit
    CAP = "cap"  # its own execution cap for the rolling 24 hours
    GLOBAL_CAP = "global_cap"  # the daily cap across all Scheduled Jobs
    BUSY = "busy"  # the global in-flight limit
    DEVICE = "device"  # the device it was created from is not trusted
    BLOCKED = "blocked"  # the adapter's runtime capability / boundary check refused
    CHANGED = "changed"  # another request or tick acted on it first (a duplicate request, or a race)


@dataclass(frozen=True)
class ManualRunResult:
    launched: bool
    refusal: ManualRefusal | None = None
    job_id: str | None = None


class ScheduledJobScheduler:
    """`scheduled_jobs` is the Slice 1 store (the ledger); `jobs` anything with `get(job_id) -> Job | None` (the existing Job store,
    only ever READ); `adapter` the Slice 2 adapter, the one thing that makes a Job; `limits` the policy's `Limits`; `enabled` a
    zero-argument callable read on every tick (the kill switch - in the application, `settings.scheduled_jobs_enabled`);
    `device_trusted` maps a device id to whether it is still trusted (the existing device registry, injected; None means no
    device can be vouched for, so a Scheduled Job created from a device is blocked). All injected: this class owns no registry, no
    clock and no permission of its own."""

    def __init__(
        self,
        *,
        scheduled_jobs: ScheduledJobStore,
        jobs,
        adapter: ScheduledExecutionAdapter,
        limits: Limits,
        enabled: Callable[[], bool] = lambda: True,
        device_trusted: Callable[[str], bool] | None = None,
        clock: Callable[[], datetime] | None = None,
        batch: int = DEFAULT_BATCH,
    ):
        if not isinstance(limits, Limits):
            raise TypeError("The scheduler needs the policy's Limits.")
        self._store = scheduled_jobs
        self._jobs = jobs
        self._adapter = adapter
        self._limits = limits
        self._enabled = enabled
        self._device_trusted = device_trusted
        self._clock = clock
        self._batch = max(1, int(batch))

    def _now(self) -> datetime:
        return (self._clock() if self._clock is not None else datetime.now(timezone.utc)).astimezone(timezone.utc)

    # -- the tick ---------------------------------------------------------------------------------------------------------------

    def tick(self) -> TickResult:
        """One bounded pass: settle what finished, then decide each due Scheduled Job once. Never raises: one Scheduled Job's failure
        is isolated and the rest of the tick goes on. Nothing is retried here - an occurrence that could not be decided simply stays
        due and the next tick looks again (and the missed-run policy ends that after the catch-up window)."""
        result = TickResult()
        self._settle_finished(result)
        self._complete_expired_paused(result)
        try:
            due = self._store.list_due(self._now(), self._batch)
        except Exception:  # noqa: BLE001 - a failed read must not break the engine's tick; the next one looks again
            logger.warning("scheduled jobs: could not list what is due", exc_info=True)
            result.errors += 1
            return result
        for scheduled_job in due:
            try:
                self._handle(scheduled_job.id, result)
            except Exception:  # noqa: BLE001 - isolate one Scheduled Job's failure
                logger.warning("scheduled jobs: could not handle %s", scheduled_job.id, exc_info=True)
                result.errors += 1
        return result

    # -- settlement -------------------------------------------------------------------------------------------------------------

    def _settle_finished(self, result: TickResult) -> None:
        """Record, once, how every finished run's Job ended. A run whose Job is still going is left alone."""
        try:
            runs = self._store.unsettled_runs()
        except Exception:  # noqa: BLE001
            logger.warning("scheduled jobs: could not list unsettled runs", exc_info=True)
            result.errors += 1
            return
        for run in runs:
            try:
                self._settle(run, result)
            except Exception:  # noqa: BLE001 - one run's settlement failing never stops the others
                logger.warning("scheduled jobs: could not settle run %s", run.id, exc_info=True)
                result.errors += 1

    def _settle(self, run: ScheduledJobRun, result: TickResult) -> bool:
        """Settle `run` if its Job has ended. Returns True when the run is still IN FLIGHT (its Job is being advanced by the engine);
        False when it is finished (and now settled) or when its Job is not in flight at all (paused, awaiting an answer or a
        confirmation) - that run is left unsettled, its Job untouched, and it no longer blocks anything. The write is `settle_run`'s: it
        applies only while the run is unsettled, so a second settler changes nothing."""
        job = self._jobs.get(run.job_id)
        if job is None:
            state = RESULT_FAILED  # a Job that no longer exists did not complete - and it is certainly not in flight
        elif job.status in TERMINAL_JOB_STATUSES:
            state = _RESULT_BY_JOB_STATUS[job.status]
        else:
            return job.status in IN_FLIGHT_JOB_STATUSES
        settled = self._store.settle_run(run.id, state, breaker_limit=self._limits.breaker_failures, via="scheduler", now=self._now())
        if settled.changed:
            result.settled += 1
            if settled.tripped:
                result.paused += 1
        return False

    def in_flight(self, scheduled_job_id: str) -> bool:
        """Whether a run of this Scheduled Job is IN FLIGHT right now: the very definition overlap uses (its Job is pending, planning or
        running - the states the JobEngine itself advances), after settling any run whose Job has already ended. A question only: it starts,
        stops and changes nothing but that settlement."""
        return self._in_flight(scheduled_job_id, TickResult()) > 0

    def _in_flight(self, schedule_id: str | None, result: TickResult) -> int:
        """How many runs are IN FLIGHT - of one Scheduled Job, or of all of them - after settling every one whose Job has already ended,
        so that a completed or failed Job is never mistaken for one still going, and a Job nobody is advancing is never counted."""
        count = 0
        for run in self._store.unsettled_runs(schedule_id):
            if self._settle(run, result):
                count += 1
        return count

    # -- the approved period ----------------------------------------------------------------------------------------------------

    def _complete_expired_paused(self, result: TickResult) -> None:
        """A paused Scheduled Job whose approved period has ended is COMPLETED (reason lifetime_ended) - the end an active one reaches
        through `SKIPPED_EXPIRED`. A paused one has no occurrence in play, so there is no ledger row: the state and its event are the
        record. Pausing never extends the period, so it cannot be resumed (the store refuses) or run past it. This is bookkeeping, not
        a launch, so it does not depend on the kill switch."""
        try:
            paused = self._store.list(frozenset({ScheduledJobStatus.PAUSED}))
        except Exception:  # noqa: BLE001 - the next tick looks again
            logger.warning("scheduled jobs: could not list paused Scheduled Jobs", exc_info=True)
            result.errors += 1
            return
        now = self._now()
        for scheduled_job in paused:
            if scheduled_job.expires_at is None or scheduled_job.expires_at > now:
                continue
            try:
                self._store.transition(
                    scheduled_job.id, ScheduledJobStatus.COMPLETED, StatusReason.LIFETIME_ENDED,
                    expected_version=scheduled_job.version, via="scheduler", detail="the approved period ended while it was paused", now=now,
                )
            except StaleScheduledJobError:
                continue  # resumed, cancelled or changed meanwhile: the newer state stands
            except Exception:  # noqa: BLE001 - isolate one Scheduled Job's failure
                logger.warning("scheduled jobs: could not complete %s", scheduled_job.id, exc_info=True)
                result.errors += 1
                continue
            result.expired += 1

    # -- one due Scheduled Job --------------------------------------------------------------------------------------------------

    def _fresh(self, scheduled_job_id: str, now: datetime) -> ScheduledJob | None:
        current = self._store.get(scheduled_job_id)
        if current is None or current.status is not ScheduledJobStatus.ACTIVE or current.due_at is None or current.due_at > now:
            return None  # gone, paused, finished, or not due (any more): not this tick's business
        return current

    def _handle(self, scheduled_job_id: str, result: TickResult) -> None:
        now = self._now()
        scheduled_job = self._fresh(scheduled_job_id, now)
        if scheduled_job is None:
            return
        decision = decide_due(
            scheduled_job.schedule, scheduled_job.due_at, now, expires_at=scheduled_job.expires_at, limits=self._limits
        )
        if decision.action is DueAction.NOT_DUE:
            return

        # 1. the kill switch: the first and cheapest gate. Off means nothing launches; the occurrence is consumed, not queued.
        if not self._is_enabled():
            self._skip(scheduled_job, decision, RunOutcome.SKIPPED_DISABLED, "kill_switch", now, result)
            return
        # 3. expiry and the missed-run policy (pure decisions made by the Slice 1 policy).
        if decision.action is DueAction.LIFETIME_ENDED:
            self._skip(scheduled_job, decision, RunOutcome.SKIPPED_EXPIRED, "lifetime_ended", now, result)
            return
        if decision.action is DueAction.SKIP_MISSED:
            self._skip(scheduled_job, decision, RunOutcome.SKIPPED_MISSED, "too_late_to_catch_up", now, result)
            return

        # 4. overlap: only a run whose Job is IN FLIGHT counts (see the module docstring). Settling what has ended changes the Scheduled
        # Job's version (and may trip the breaker), so it is read again afterwards.
        previous_in_flight = self._in_flight(scheduled_job.id, result) > 0
        scheduled_job = self._fresh(scheduled_job_id, self._now())
        if scheduled_job is None:
            return
        # 5/6. overlap -> breaker -> this Scheduled Job's cap -> the global daily cap -> the global in-flight limit (`admit` holds the
        # order of everything but the breaker).
        window_start = now - DAILY_WINDOW + _MICROSECOND  # a run exactly 24 hours old is outside the window
        admission = admit(
            previous_run_unsettled=previous_in_flight,
            unsettled_total=self._in_flight(None, result),
            runs_in_last_day=self._store.count_runs_since(window_start),
            schedule_runs_in_last_day=self._store.count_runs_since(window_start, scheduled_job.id),
            limits=self._limits,
        )
        if admission is Admission.SKIP_OVERLAP:
            self._skip(scheduled_job, decision, RunOutcome.SKIPPED_OVERLAP, "previous_run_active", now, result)
            return
        if scheduled_job.consecutive_failures >= self._limits.breaker_failures:
            self._pause(scheduled_job, StatusReason.BREAKER, f"{scheduled_job.consecutive_failures} failed runs in a row", result)
            return
        if admission is Admission.SKIP_CAP:
            self._skip(scheduled_job, decision, RunOutcome.SKIPPED_CAP, "execution_cap", now, result)
            return
        if admission in (Admission.DEFER_DAILY_CAP, Admission.DEFER_CONCURRENCY):
            result.deferred += 1  # a GLOBAL limit: stays due, not decided, not recorded (the catch-up window ends it)
            return

        # 7. the device the Scheduled Job was created from must still be trusted. Only THIS occurrence is blocked: the Scheduled Job
        # stays active and the next occurrence is asked the same question again (no device policy beyond the existing trust state).
        if scheduled_job.created_via_device_id is not None and not self._is_trusted(scheduled_job.created_via_device_id):
            self._skip(scheduled_job, decision, RunOutcome.SKIPPED_BLOCKED, "device_not_trusted", now, result)
            return

        # the kill switch once more, immediately before the launch: a switch thrown while this tick was deciding still stops it.
        if not self._is_enabled():
            self._skip(scheduled_job, decision, RunOutcome.SKIPPED_DISABLED, "kill_switch", now, result)
            return

        # 8/9. the adapter re-checks the capability boundary against the live profile-filtered tool set, then makes the Job and its
        # ledger row in one transaction. A refusal wrote nothing, so the occurrence is consumed here, as blocked.
        occurrence = ScheduledOccurrence.from_decision(scheduled_job, decision)
        try:
            self._adapter.launch(scheduled_job, occurrence)
        except ExecutionRefused as refusal:
            self._skip(scheduled_job, decision, RunOutcome.SKIPPED_BLOCKED, f"refused: {refusal}", now, result)
            return
        except (DuplicateOccurrenceError, StaleScheduledJobError):
            result.lost_races += 1  # another tick / scheduler / process acted on it first: it is theirs, nothing was written here
            return
        result.launched += 1
        logger.info("scheduled jobs: launched %s (%s)", scheduled_job.id, occurrence.occurrence)

    # -- run it now ---------------------------------------------------------------------------------------------------------------

    def run_now(self, scheduled_job_id: str) -> ManualRunResult:
        """Start ONE run of an approved, ACTIVE Scheduled Job right now - through the very same gates as a due occurrence, in the same
        order, and then the same adapter launch (so the same Job, profile, boundary check and ledger transaction):

            kill switch -> active and inside its approved period -> overlap -> failure breaker -> this Scheduled Job's cap ->
            the two GLOBAL limits -> device trust -> kill switch again -> the adapter's capability check and atomic launch

        The ONE difference: a gate that stops a due occurrence CONSUMES it (a skipped ledger row, the schedule moves on); here a gate
        merely REFUSES the request and writes nothing, because a refused request is not an occurrence - there is nothing to consume and
        no history to fake. A manual run is the ledger's MANUAL kind: it counts as an execution for every cap, overlap and the breaker
        (it IS one), but it does not move the schedule, so the next real occurrence is exactly where it was. Two requests racing for the
        same state share one identity (see `ScheduledOccurrence.manual`): one launches, the other is refused as CHANGED."""
        now = self._now()
        if not self._is_enabled():
            return ManualRunResult(False, ManualRefusal.DISABLED)
        scheduled_job = self._store.get(scheduled_job_id)
        if scheduled_job is None:
            return ManualRunResult(False, ManualRefusal.NOT_FOUND)
        if scheduled_job.status is not ScheduledJobStatus.ACTIVE or scheduled_job.due_at is None:
            return ManualRunResult(False, ManualRefusal.NOT_ACTIVE)
        if scheduled_job.schedule.kind is ScheduleKind.ONCE:
            return ManualRunResult(False, ManualRefusal.ONE_TIME)
        if scheduled_job.expires_at is not None and now >= scheduled_job.expires_at:
            return ManualRunResult(False, ManualRefusal.EXPIRED)

        result = TickResult()  # settling what has ended is bookkeeping the tick does too; its counts are not needed here
        previous_in_flight = self._in_flight(scheduled_job.id, result) > 0
        scheduled_job = self._store.get(scheduled_job_id)  # settling may have changed its version (or tripped the breaker)
        if scheduled_job is None or scheduled_job.status is not ScheduledJobStatus.ACTIVE or scheduled_job.due_at is None:
            return ManualRunResult(False, ManualRefusal.NOT_ACTIVE)
        window_start = now - DAILY_WINDOW + _MICROSECOND
        admission = admit(
            previous_run_unsettled=previous_in_flight,
            unsettled_total=self._in_flight(None, result),
            runs_in_last_day=self._store.count_runs_since(window_start),
            schedule_runs_in_last_day=self._store.count_runs_since(window_start, scheduled_job.id),
            limits=self._limits,
        )
        if admission is Admission.SKIP_OVERLAP:
            return ManualRunResult(False, ManualRefusal.OVERLAP)
        if scheduled_job.consecutive_failures >= self._limits.breaker_failures:
            return ManualRunResult(False, ManualRefusal.BREAKER)
        if admission is Admission.SKIP_CAP:
            return ManualRunResult(False, ManualRefusal.CAP)
        if admission is Admission.DEFER_DAILY_CAP:
            return ManualRunResult(False, ManualRefusal.GLOBAL_CAP)
        if admission is Admission.DEFER_CONCURRENCY:
            return ManualRunResult(False, ManualRefusal.BUSY)
        if scheduled_job.created_via_device_id is not None and not self._is_trusted(scheduled_job.created_via_device_id):
            return ManualRunResult(False, ManualRefusal.DEVICE)
        if not self._is_enabled():  # thrown while we were deciding: still stops it
            return ManualRunResult(False, ManualRefusal.DISABLED)

        try:
            launched = self._adapter.launch(scheduled_job, ScheduledOccurrence.manual(scheduled_job, self._now()))
        except ExecutionRefused as refusal:
            logger.info("scheduled jobs: run now refused for %s (%s)", scheduled_job.id, refusal)
            return ManualRunResult(False, ManualRefusal.BLOCKED)
        except (DuplicateOccurrenceError, StaleScheduledJobError):
            return ManualRunResult(False, ManualRefusal.CHANGED)
        logger.info("scheduled jobs: launched %s manually (%s)", scheduled_job.id, launched.run.occurrence)
        return ManualRunResult(True, job_id=launched.job.id)

    # -- consuming an occurrence ------------------------------------------------------------------------------------------------

    def _skip(
        self, scheduled_job: ScheduledJob, decision: DueDecision, outcome: RunOutcome, reason: str, now: datetime, result: TickResult
    ) -> bool:
        """Consume the occurrence WITHOUT a Job: one ledger row (the occurrence's terminal settlement) and the Scheduled Job moved to its
        next future occurrence (or completed), compare-and-set in one transaction. True if this call recorded it; False if someone else
        already had (nothing is written then)."""
        complete_reason = decision.complete_reason
        if complete_reason is StatusReason.ONE_TIME_DONE:
            complete_reason = StatusReason.ONE_TIME_MISSED  # a one-time Scheduled Job that did not run was missed, not done
        line = " ".join(str(reason).split())[:MAX_REASON_DETAIL_CHARS]
        try:
            self._store.record_run(
                scheduled_job.id,
                NewRun(
                    occurrence=occurrence_key(scheduled_job.id, decision.occurrence_at),
                    scheduled_for=decision.occurrence_at,
                    outcome=outcome,
                    reason=line,
                    missed_count=decision.missed_count,
                ),
                expected_version=scheduled_job.version,
                expected_due_at=scheduled_job.due_at,
                next_due_at=decision.next_due,
                complete_reason=complete_reason,
                via="scheduler",
                now=now,
            )
        except (DuplicateOccurrenceError, StaleScheduledJobError):
            result.lost_races += 1
            return False
        result.count_skip(outcome)
        logger.info("scheduled jobs: %s %s (%s)", outcome.value, scheduled_job.id, line)
        return True

    def _pause(self, scheduled_job: ScheduledJob, reason: StatusReason, detail: str, result: TickResult) -> None:
        try:
            self._store.transition(
                scheduled_job.id, ScheduledJobStatus.PAUSED, reason, expected_version=scheduled_job.version, via="scheduler", detail=detail
            )
        except StaleScheduledJobError:
            return  # it changed while we looked (paused, resumed, run): the newer state stands
        result.paused += 1

    # -- the two injected questions, asked fail-closed ----------------------------------------------------------------------------

    def _is_enabled(self) -> bool:
        try:
            return bool(self._enabled())
        except Exception:  # noqa: BLE001 - if the switch cannot be read, it is OFF
            return False

    def _is_trusted(self, device_id: str) -> bool:
        if self._device_trusted is None:
            return False
        try:
            return bool(self._device_trusted(device_id))
        except Exception:  # noqa: BLE001 - a device that cannot be vouched for is not trusted
            return False
