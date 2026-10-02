"""
Scheduled Jobs policy: the PURE decisions - no clock (`now` is always an argument), no I/O, no storage, no tool, no model.

Each function answers one question from plain values, so every rule can be tested exhaustively and nothing here can act:

  lifetime_for         how long an approved recurring Scheduled Job stays active (never more than the V1 lifetime), and whether the
                       user's longer request needs a renewal after it
  validate_new_schedule  is this NEW schedule acceptable (not too frequent, not too far ahead, not beyond the first lifetime period)
  check_proposal_cap   may another Scheduled Job exist
  decide_due           an occurrence is due: run it, run it late ONCE, skip it, or the lifetime is over - and what comes next
  admit                may a run start now (overlap, daily cap, concurrency cap)
  breaker_decision     what a finished run does to the failure streak, and whether it pauses the Scheduled Job
  next_future_due      where a paused Scheduled Job resumes (never a replay)

THE MISSED-RUN POLICY (V1, one policy, no per-schedule switch). Built on `plan_run`'s facts. An occurrence within the grace window just
runs. A MISSED one - Jarvis was not running - is NEVER replayed occurrence by occurrence: at most ONE catch-up run happens, for the
most recent missed occurrence, and only while that occurrence is no older than the catch-up window; otherwise it is skipped. Either way
the Scheduled Job moves to its next FUTURE occurrence, and the catch-up run's identity (`occurrence_at`) is a single deterministic
instant, so a repeated poll cannot make a second one.

All limits come from one `Limits` value (settings in the running app), so tests and later slices never read configuration here.
"""

from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

from app.scheduled_jobs.base import (
    MAX_INTENT_CHARS,
    MAX_OBJECTIVE_CHARS,
    MAX_TITLE_CHARS,
    RESULT_CANCELLED,
    RESULT_COMPLETED,
    RESULT_FAILED,
    ScheduledJobError,
    StatusReason,
)
from app.scheduling.builder import MAX_HORIZON
from app.scheduling.runs import DEFAULT_GRACE, RunDecision, plan_run
from app.scheduling.schedule import Schedule, ScheduleKind, next_due

#: "Runs today" is counted over a rolling 24 hours, so the cap needs no time zone and cannot be dodged by crossing midnight.
DAILY_WINDOW = timedelta(hours=24)


class ScheduledJobPolicyError(ScheduledJobError):
    """A request the V1 limits refuse. The message is plain and safe to tell the user."""


@dataclass(frozen=True)
class Limits:
    """Every V1 limit in one place. Defaults are the approved V1 values; `from_settings` reads the configured ones."""

    min_interval: timedelta = timedelta(minutes=60)  # NEW repeating Scheduled Jobs only
    lifetime: timedelta = timedelta(days=90)  # a recurring one's active life, from approval
    catch_up_window: timedelta = timedelta(hours=6)  # how late the ONE catch-up run may still be
    grace: timedelta = DEFAULT_GRACE  # how late an occurrence may be and still simply run
    horizon: timedelta = MAX_HORIZON  # how far ahead a one-time Scheduled Job may be set
    max_open: int = 10
    max_runs_per_day: int = 10
    max_concurrent: int = 2
    breaker_failures: int = 3

    @classmethod
    def from_settings(cls) -> "Limits":
        from app.config import settings

        return cls(
            min_interval=timedelta(minutes=max(1, int(settings.scheduled_job_min_interval_minutes))),
            lifetime=timedelta(days=max(1, int(settings.scheduled_job_lifetime_days))),
            catch_up_window=timedelta(seconds=max(0, int(settings.scheduled_job_catch_up_window_seconds))),
            grace=timedelta(seconds=max(0, int(settings.schedule_grace_seconds))),
            max_open=max(1, int(settings.scheduled_job_max_active)),
            max_runs_per_day=max(1, int(settings.scheduled_job_max_runs_per_day)),
            max_concurrent=max(1, int(settings.scheduled_job_max_concurrent)),
            breaker_failures=max(1, int(settings.scheduled_job_breaker_failures)),
        )


def _aware(value: object, name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ScheduledJobPolicyError(f"`{name}` must be a timezone-aware datetime.")
    return value


# -- text a Scheduled Job stores ------------------------------------------------------------------------------------------------


def _one_line(text: object) -> str:
    cleaned = "".join(" " if unicodedata.category(ch) in ("Cc", "Cf", "Zl", "Zp") else ch for ch in (text if isinstance(text, str) else ""))
    return " ".join(cleaned.split())


def clean_title(text: object) -> str:
    """The short label a Scheduled Job is referred to by. Single line; refused (never silently cut) when empty or too long."""
    title = _one_line(text)
    if not title:
        raise ScheduledJobPolicyError("A Scheduled Job needs a short title.")
    if len(title) > MAX_TITLE_CHARS:
        raise ScheduledJobPolicyError(f"The title is too long - keep it under {MAX_TITLE_CHARS} characters.")
    return title


def clean_intent(text: object) -> str:
    """The user's own words for WHAT to do. Single line, never silently cut."""
    intent = _one_line(text)
    if not intent:
        raise ScheduledJobPolicyError("Tell me what you want done.")
    if len(intent) > MAX_INTENT_CHARS:
        raise ScheduledJobPolicyError(f"That is too long - please keep it under {MAX_INTENT_CHARS} characters.")
    return intent


def validate_objective(objective: object) -> dict[str, Any]:
    """The stored understanding of the request: a JSON-safe object with text keys, within the size bound. (What is IN it is validated
    where it is made - Handle This - and again when it is used; this only keeps storage well-formed.)"""
    if not isinstance(objective, dict) or any(not isinstance(key, str) or not key for key in objective):
        raise ScheduledJobPolicyError("The objective must be an object with text keys.")
    try:
        text = json.dumps(objective, sort_keys=True, ensure_ascii=False)
    except (TypeError, ValueError):
        raise ScheduledJobPolicyError("The objective must be plain data (text, numbers, lists, objects).") from None
    if len(text) > MAX_OBJECTIVE_CHARS:
        raise ScheduledJobPolicyError("The objective is too large.")
    return json.loads(text)


# -- lifetime -------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Lifetime:
    """`expires_at` is None for a one-time schedule. `renewal_required`: the user asked for longer than V1 activates (or for no end),
    so they must be told that fresh approval is needed after `expires_at`."""

    expires_at: datetime | None
    renewal_required: bool


def lifetime_for(schedule: Schedule, approved_at: datetime, requested_until: datetime | None, limits: Limits) -> Lifetime:
    """The 90-day (V1) lifetime is an EXECUTION boundary, not an understanding one: a request for longer ("every Monday for two years") is
    understood and activated for the first period only, and `renewal_required` says renewal will be needed. It is never rejected for
    being long. It starts at approval, not at the first run. A one-time schedule has no lifetime (its horizon is checked separately)."""
    approved_at = _aware(approved_at, "approved_at")
    if schedule.kind is ScheduleKind.ONCE:
        return Lifetime(None, False)
    cap = approved_at + limits.lifetime
    if requested_until is None:
        return Lifetime(cap, True)  # an open-ended request still ends at the cap
    requested_until = _aware(requested_until, "requested_until")
    if requested_until <= approved_at:
        raise ScheduledJobPolicyError("The end you asked for is not in the future.")
    if requested_until > cap:
        return Lifetime(cap, True)
    return Lifetime(requested_until, False)


def validate_new_schedule(schedule: Schedule, *, now: datetime, first_due: datetime | None, limits: Limits) -> None:
    """Refuse a NEW schedule that is too frequent, too far ahead, or has no future occurrence. (Never applied to a stored one.)"""
    now = _aware(now, "now")
    if first_due is None:
        raise ScheduledJobPolicyError("That time has no occurrence ahead of it.")
    first_due = _aware(first_due, "first_due")
    if first_due <= now:
        raise ScheduledJobPolicyError("The first run has to be in the future.")
    if schedule.kind is ScheduleKind.ONCE:
        if first_due - now > limits.horizon:
            raise ScheduledJobPolicyError("That is more than two years away - please pick a nearer date.")
        return
    second = next_due(schedule, first_due)
    if second is not None and second - first_due < limits.min_interval:
        minutes = int(limits.min_interval.total_seconds() // 60)
        raise ScheduledJobPolicyError(f"A repeating Scheduled Job has to be at least {minutes} minutes apart.")
    if first_due - now > limits.lifetime:
        days = int(limits.lifetime.total_seconds() // 86400)
        raise ScheduledJobPolicyError(f"Its first run would fall after the first {days} days, which is as long as one approval lasts.")


def check_proposal_cap(open_count: int, limits: Limits) -> None:
    """`open_count` is how many Scheduled Jobs are proposed, active or paused."""
    if open_count >= limits.max_open:
        raise ScheduledJobPolicyError(
            f"You already have {open_count} scheduled jobs - please cancel one before adding another."
        )


# -- an occurrence is due -------------------------------------------------------------------------------------------------------


class DueAction(Enum):
    NOT_DUE = "not_due"  # the clock is before `due_at`: leave it for a later poll
    RUN = "run"  # on time (within grace): create the Job
    RUN_LATE = "run_late"  # missed, but within the catch-up window: create the ONE catch-up Job
    SKIP_MISSED = "skip_missed"  # missed, and too late to catch up: record it and move on
    LIFETIME_ENDED = "lifetime_ended"  # the approved period is over: create nothing, complete the Scheduled Job


@dataclass(frozen=True)
class DueDecision:
    """`occurrence_at` is the instant whose `occurrence_key` identifies this run (for a catch-up or skip: the MOST RECENT missed
    occurrence - one deterministic identity, however many were missed). `next_due` is where the Scheduled Job goes next (None: there is
    none); `complete_reason` is set when this decision ends it."""

    action: DueAction
    occurrence_at: datetime
    missed_count: int = 0
    next_due: datetime | None = None
    complete_reason: StatusReason | None = None


def decide_due(schedule: Schedule, due_at: datetime, now: datetime, *, expires_at: datetime | None, limits: Limits) -> DueDecision:
    """What to do about the occurrence stored as `due_at`, seen at `now`. Pure and deterministic: the same inputs always give the same
    decision and the same `occurrence_at`."""
    due_at, now = _aware(due_at, "due_at"), _aware(now, "now")
    if expires_at is not None and now >= _aware(expires_at, "expires_at"):
        return DueDecision(DueAction.LIFETIME_ENDED, due_at, complete_reason=StatusReason.LIFETIME_ENDED)
    plan = plan_run(schedule, due_at, now, limits.grace)
    if plan.decision is RunDecision.NOT_DUE:
        return DueDecision(DueAction.NOT_DUE, due_at, next_due=due_at)

    upcoming = plan.next_due
    ended_by_lifetime = False
    if upcoming is not None and expires_at is not None and upcoming >= expires_at:
        upcoming, ended_by_lifetime = None, True  # no occurrence after the approved period ever runs

    def finish(ran: bool) -> StatusReason | None:
        if upcoming is not None:
            return None
        if ended_by_lifetime or schedule.kind is not ScheduleKind.ONCE:
            return StatusReason.LIFETIME_ENDED
        return StatusReason.ONE_TIME_DONE if ran else StatusReason.ONE_TIME_MISSED

    if plan.decision is RunDecision.RUN:
        return DueDecision(DueAction.RUN, plan.scheduled_for, 0, upcoming, finish(True))

    latest = plan.latest_missed or due_at
    if now - latest <= limits.catch_up_window:
        return DueDecision(DueAction.RUN_LATE, latest, plan.missed_count, upcoming, finish(True))
    return DueDecision(DueAction.SKIP_MISSED, latest, plan.missed_count, upcoming, finish(False))


def next_future_due(schedule: Schedule, now: datetime, expires_at: datetime | None) -> datetime | None:
    """Where a paused Scheduled Job resumes: its next occurrence strictly after `now`, or None when none is left inside its lifetime.
    Whatever came due while it was paused is never replayed."""
    upcoming = next_due(schedule, _aware(now, "now"))
    if upcoming is not None and expires_at is not None and upcoming >= expires_at:
        return None
    return upcoming


# -- may a run start now --------------------------------------------------------------------------------------------------------


class Admission(Enum):
    ADMIT = "admit"
    SKIP_OVERLAP = "skip_overlap"  # the previous run of this Scheduled Job has not finished: this occurrence is skipped, not queued
    DEFER_DAILY_CAP = "defer_daily_cap"  # leave it due; if it stays blocked past its window it becomes a skipped occurrence
    DEFER_CONCURRENCY = "defer_concurrency"  # likewise


def admit(*, previous_run_unsettled: bool, unsettled_total: int, runs_in_last_day: int, limits: Limits) -> Admission:
    """Single-flight per Scheduled Job first, then the global caps. A deferral is not a decision to skip: the occurrence stays due and
    the next poll decides again (and, once it is too late to catch up, `decide_due` turns it into a skip)."""
    if previous_run_unsettled:
        return Admission.SKIP_OVERLAP
    if runs_in_last_day >= limits.max_runs_per_day:
        return Admission.DEFER_DAILY_CAP
    if unsettled_total >= limits.max_concurrent:
        return Admission.DEFER_CONCURRENCY
    return Admission.ADMIT


# -- the circuit breaker --------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class BreakerResult:
    consecutive_failures: int
    trip: bool


def breaker_decision(current_streak: int, result_state: str, limit: int) -> BreakerResult:
    """A completed run resets the failure streak; a failed one (a timeout is a failure) extends it and, at `limit`, pauses the Scheduled
    Job; a cancelled one (the user's own choice) is neutral. There is no retry here: the next occurrence is the retry."""
    if not isinstance(current_streak, int) or isinstance(current_streak, bool) or current_streak < 0:
        raise ScheduledJobPolicyError("The failure streak must be a non-negative number.")
    if result_state == RESULT_COMPLETED:
        return BreakerResult(0, False)
    if result_state == RESULT_FAILED:
        streak = current_streak + 1
        return BreakerResult(streak, streak >= limit)
    if result_state == RESULT_CANCELLED:
        return BreakerResult(current_streak, False)
    raise ScheduledJobPolicyError(f"Unknown run result '{result_state}'.")
