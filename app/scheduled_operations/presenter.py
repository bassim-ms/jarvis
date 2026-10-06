"""
Presenting what became of a Scheduled Job's occurrences - in Jarvis's words, from facts that exist.

    present_run(scheduled_job, run, facts, limits)  ->  Presented | None      PURE: words from facts; changes nothing
    ScheduledJobReporter.report()                    ->  tells the user ONCE per final occurrence, through the EXISTING notifications

There is no new notification framework. A Presented is an ordinary `NotificationEvent` (app/notifications/base.py) handed to the router
the Job engine and the interaction lifecycle already use - so the attention policy, the channels, the at-most-once ledger and the
desktop toast all apply unchanged. The only vocabulary is the existing one: a run that finished is COMPLETED, one that failed is FAILED,
and an occurrence that did not run (skipped) is EXPIRED - "something that was due lapsed".

WHAT IT MAY SAY. Only what the records hold: the outcome of the occurrence (ledger), how the Job ended and how long it ran (the Job's own
status and times), the Scheduled Job's own title, state and failure streak, and the configured limits. It never says why a tool failed or
what a page contained: a Job's error text and output are tool-derived, untrusted, and do not belong in something that may be spoken aloud
or shown on a lock screen (the same rule as the engine's own notifications). What a finished run FOUND is available on request - the user
asks to see the Scheduled Job and gets a bounded excerpt of the Job's own report, quoted as data.

WHAT IT MAY NOT DO. The presenter is pure: it reads, it words, it returns. The reporter changes exactly one thing - `reported_at` on the
ledger row, once the user has been told (that is what the column is for). It never changes a Job, a Scheduled Job, a priority, a
permission or the security decision that was made; it never retries anything, writes memory, creates a task or a note, or contacts anyone.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from app.notifications.base import DEFAULT_USER, NotificationEvent, NotificationKind, NotificationPriority
from app.notifications.text import address_for, clean_text
from app.orchestration.base import TIMED_OUT_ERROR, Job, JobStatus
from app.scheduled_jobs.base import (
    RESULT_COMPLETED,
    RESULT_FAILED,
    RunKind,
    RunOutcome,
    ScheduledJob,
    ScheduledJobRun,
    ScheduledJobStatus,
    ScheduledJobStore,
    StatusReason,
)
from app.scheduled_jobs.policy import Limits
from app.scheduled_operations.language import hint_name

logger = logging.getLogger("jarvis.scheduled_operations")

_TITLE = 60
_EXCERPT = 300

#: A final outcome older than this when it comes to be reported is not news any more (Jarvis was off, or no channel took it): it is marked as
#: handled without a notification, so a long absence never produces a burst of stale announcements.
STALE_AFTER = timedelta(hours=24)


@dataclass(frozen=True)
class JobFacts:
    """What the Job a run created says about how it ended - only facts the Job itself records."""

    ended_in: str  # "completed" | "failed" | "cancelled" | anything else (still going / unknown)
    timed_out: bool = False
    seconds: float | None = None  # how long it ran (created -> last update)
    deadline_seconds: float | None = None  # the time limit it was given
    result_excerpt: str | None = None  # a bounded, single-line excerpt of its report (only for a completed run)


def _seconds_between(start: str | None, end: str | None) -> float | None:
    try:
        return max(0.0, (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds())
    except (TypeError, ValueError):
        return None


def facts_from_job(job: Job | None) -> JobFacts | None:
    """The facts of a Job, for the presenter. Read-only; None when there is no Job (it no longer exists)."""
    if job is None:
        return None
    ended = job.status.value if job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED) else "unfinished"
    return JobFacts(
        ended_in=ended,
        timed_out=ended == "failed" and job.error == TIMED_OUT_ERROR,
        seconds=_seconds_between(job.created_at, job.updated_at),
        deadline_seconds=_seconds_between(job.created_at, job.deadline_at) if job.deadline_at else None,
        result_excerpt=(clean_text(job.final_result, _EXCERPT) or None) if ended == "completed" else None,
    )


def span(seconds: float | None) -> str:
    """"under a minute" / "3 minutes" / "2 hours" - words, never raw seconds."""
    if seconds is None:
        return ""
    if seconds < 60:
        return "under a minute"
    minutes = round(seconds / 60)
    if minutes < 60:
        return f"{minutes} minute{'' if minutes == 1 else 's'}"
    hours = round(minutes / 60)
    return f"{hours} hour{'' if hours == 1 else 's'}"


@dataclass(frozen=True)
class Presented:
    key: str
    kind: NotificationKind
    priority: NotificationPriority
    text: str

    def event(self, run: ScheduledJobRun) -> NotificationEvent:
        return NotificationEvent(key=self.key, kind=self.kind, priority=self.priority, text=self.text, job_id=run.job_id)


# ====================================== the words ========================================================================

#: Why an occurrence was skipped as BLOCKED, keyed by the ledger's own reason codes (see the scheduler). Anything else is the adapter's
#: "refused: ..." text, which names internals and is deliberately never repeated.
_BLOCKED = {
    "device_not_trusted": "the device it was set up from isn't trusted any more",
}
_BLOCKED_DEFAULT = "something it needs isn't available to an unattended run right now"


#: ROUTINE scheduler decisions that are never announced, however often they happen: the kill switch is the user's own setting; an overlap
#: is followed by the result of the run that caused it; and the per-schedule cap recurs daily for a busy schedule. They stay in the ledger
#: (and in "show my X scheduled job") - this is about not interrupting the user, not about hiding facts.
SILENT_SKIPS = frozenset({RunOutcome.SKIPPED_DISABLED, RunOutcome.SKIPPED_OVERLAP, RunOutcome.SKIPPED_CAP})
#: A problem the user may want to act on, told ONCE when it begins: a repeat of the same outcome as the occurrence just before it is the
#: same news and stays silent (it is told again after anything different - a run that worked, say - happened in between).
ONCE_PER_STREAK = frozenset({RunOutcome.SKIPPED_BLOCKED, RunOutcome.SKIPPED_MISSED})


def _next_time(scheduled_job: ScheduledJob) -> str:
    return " It stays scheduled, and I'll try again at its next time." if scheduled_job.status is ScheduledJobStatus.ACTIVE else ""


def _skip_words(scheduled_job: ScheduledJob, run: ScheduledJobRun) -> tuple[str, NotificationPriority]:
    outcome = run.outcome
    if outcome is RunOutcome.SKIPPED_BLOCKED:
        why = _BLOCKED.get(run.reason or "", _BLOCKED_DEFAULT)
        return f"couldn't run this time: {why}.{_next_time(scheduled_job)}", NotificationPriority.NORMAL
    if outcome is RunOutcome.SKIPPED_MISSED:
        extra = f" ({run.missed_count} runs were missed, and I don't replay them)" if run.missed_count > 1 else ""
        return f"was due, but I wasn't running at the time{extra}, so it didn't run.{_next_time(scheduled_job)}", NotificationPriority.NORMAL
    if outcome is RunOutcome.SKIPPED_EXPIRED:
        return (
            "has reached the end of its approved period, so it has stopped. "
            f"Say \"renew my {hint_name(scheduled_job.title)} scheduled job\" if you want it to keep going.",
            NotificationPriority.NORMAL,
        )
    return "didn't run this time.", NotificationPriority.LOW  # an outcome with no wording of its own is still never shown raw


def _failure_words(scheduled_job: ScheduledJob, facts: JobFacts | None, limits: Limits) -> str:
    if facts is not None and facts.timed_out:
        # The limit the Job was given - unless its record does not give a sensible duration, then the configured one (never "after under a minute").
        seconds = facts.deadline_seconds if facts.deadline_seconds is not None and facts.deadline_seconds >= 60 else limits.run_deadline.total_seconds()
        text = f"ran but didn't finish: it timed out after {span(seconds)}."
    else:
        text = "ran but couldn't finish."
    streak = scheduled_job.consecutive_failures
    if scheduled_job.status is ScheduledJobStatus.PAUSED and scheduled_job.status_reason is StatusReason.BREAKER:
        text += f" That was {streak} failed runs in a row, so I've paused it. Resume it when you're ready."
    elif streak > 0:
        text += f" That's {streak} failed run{'' if streak == 1 else 's'} in a row; I'll pause it after {limits.breaker_failures}."
    return text


def present_run(
    scheduled_job: ScheduledJob, run: ScheduledJobRun, facts: JobFacts | None, limits: Limits, previous: ScheduledJobRun | None = None
) -> Presented | None:
    """What to tell the user about one FINAL occurrence, or None when there is nothing to say: a run the user cancelled themselves, a run
    that has not finished, a ROUTINE skip (`SILENT_SKIPS`), or a repeat of a problem already told (`ONCE_PER_STREAK`: `previous` is the
    occurrence just before this one). Results - a run that finished or failed - and meaningful changes of state (a problem beginning, a
    period ending) are what get announced. Pure: the same inputs always give the same words, and nothing is changed."""
    title = clean_text(scheduled_job.title, _TITLE)
    manual = run.kind is RunKind.MANUAL
    if run.job_id is None:  # a skipped occurrence
        if run.outcome in SILENT_SKIPS:
            return None
        if run.outcome in ONCE_PER_STREAK and previous is not None and previous.outcome is run.outcome:
            return None
        key = f"scheduled:{run.id}:{run.outcome.value}"
        body, priority = _skip_words(scheduled_job, run)
        return Presented(key, NotificationKind.EXPIRED, priority, f'{address_for(key)}, your scheduled job "{title}" {body}')
    state = run.result_state
    if state is None:  # the Job has not finished: nothing to say yet
        return None
    key = f"scheduled:{run.id}:{state}"
    who = address_for(key)
    lead = f'the run you asked for - "{title}" -' if manual else f'your scheduled job "{title}"'
    if state == RESULT_COMPLETED:
        took = span(facts.seconds) if facts is not None else ""
        took = f" in {took}" if took and took != "under a minute" else ""
        return Presented(
            key, NotificationKind.COMPLETED, NotificationPriority.LOW,
            f'{who}, {lead} finished{took}. Ask me to show it and I\'ll tell you what it found.',
        )
    if state == RESULT_FAILED:
        return Presented(key, NotificationKind.FAILED, NotificationPriority.HIGH, f"{who}, {lead} {_failure_words(scheduled_job, facts, limits)}")
    return None  # a cancelled run is the user's own choice: nothing to announce


# ====================================== telling the user, once ==================================================================


class ScheduledJobReporter:
    """Reports final occurrences through the notification router, once each. `notify(user, event)` is the router the application already
    has (the attention-gated one); `job_facts(job_id)` reads a Job's facts (None if it is gone). Hung on the engine's own tick like the
    scheduler - no loop, thread or timer of its own - and it never raises: a broken channel or read leaves the occurrence unreported and
    the next tick looks again (until it is stale)."""

    def __init__(
        self,
        *,
        scheduled_jobs: ScheduledJobStore,
        notify: Callable[[str, NotificationEvent], object],
        job_facts: Callable[[str], JobFacts | None],
        limits: Limits,
        clock: Callable[[], datetime] | None = None,
        user: str = DEFAULT_USER,
        batch: int = 20,
    ):
        self._store = scheduled_jobs
        self._notify = notify
        self._job_facts = job_facts
        self._limits = limits
        self._clock = clock
        self._user = user
        self._batch = max(1, int(batch))

    def _now(self) -> datetime:
        return (self._clock() if self._clock is not None else datetime.now(timezone.utc)).astimezone(timezone.utc)

    def report(self) -> int:
        """Tell the user about every final occurrence they have not been told about. Returns how many notifications were handed over."""
        told = 0
        try:
            pending = self._store.unreported_runs(self._batch)
        except Exception:  # noqa: BLE001 - the next tick looks again
            logger.warning("scheduled jobs: could not list unreported runs", exc_info=True)
            return 0
        for run in pending:
            try:
                told += self._report_one(run)
            except Exception:  # noqa: BLE001 - one occurrence never stops the others, and stays unreported for the next tick
                logger.warning("scheduled jobs: could not report run %s", run.id, exc_info=True)
        return told

    def _previous(self, run: ScheduledJobRun) -> ScheduledJobRun | None:
        """The occurrence of the same Scheduled Job recorded just before `run`."""
        earlier = [r for r in self._store.runs_for(run.schedule_id, 50) if r.id < run.id]
        return max(earlier, key=lambda r: r.id) if earlier else None

    def _report_one(self, run: ScheduledJobRun) -> int:
        now = self._now()
        happened = run.settled_at or run.created_at
        scheduled_job = self._store.get(run.schedule_id)
        if scheduled_job is None or now - happened > STALE_AFTER:
            self._store.mark_reported(run.id, now)  # not news any more (or nothing left to refer to): handled, silently
            return 0
        facts = self._job_facts(run.job_id) if run.job_id is not None else None
        previous = self._previous(run) if run.job_id is None and run.outcome in ONCE_PER_STREAK else None
        presented = present_run(scheduled_job, run, facts, self._limits, previous)
        if presented is None:
            self._store.mark_reported(run.id, now)  # nothing to say (a routine skip, a repeat, a cancelled run): handled, silently
            return 0
        delivery = self._notify(self._user, presented.event(run))
        if getattr(delivery, "accepted", False) or getattr(delivery, "duplicate", False):
            self._store.mark_reported(run.id, now)  # handed over (or already was): never again
            return 1 if getattr(delivery, "accepted", False) else 0
        return 0  # no channel took it: stays unreported, the next tick tries again
