"""
Run planning: given a schedule, the occurrence that was due, and "now", what is the situation - and how do two runs of the same
occurrence get recognised as the same one?

PURE and policy-free. `plan_run` only states the FACTS (on time, late within the grace window, or missed; how many occurrences were
skipped; when the next one is). What to DO about a missed run - one catch-up notice for a reminder, skip a scheduled Job - is the
caller's policy, applied in later slices on top of these facts.

Restart behaviour follows from that: a stored item is just (schedule, `due_at`). After any outage `plan_run` collapses every missed
occurrence into ONE `MISSED` verdict (never a replay storm) and names the next future occurrence.

Duplicate prevention is identity by the RESOLVED INSTANT: `occurrence_key` is derived from the UTC instant an occurrence resolved
to, never from local wall-clock text. In a daylight-saving overlap a local time happens twice; the schedule only ever resolves it
to its first occurrence, so one occurrence, one key, one run.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

from app.scheduling.schedule import Schedule, ScheduleError, ScheduleKind, _as_utc, interval_delta, next_due, previous_due
from app.scheduling.timezones import UTC

#: The grace window a late run is still simply "run" (settings.schedule_grace_seconds overrides it).
DEFAULT_GRACE = timedelta(minutes=10)
#: `missed_count` stops counting here (a schedule offline for years is still one MISSED verdict).
MAX_MISSED_COUNT = 1000
MAX_OWNER_ID_CHARS = 128


class RunDecision(Enum):
    NOT_DUE = "not_due"  # `now` is before `due_at`
    RUN = "run"  # due, and no later than the grace window
    MISSED = "missed"  # due, but later than the grace window


@dataclass(frozen=True)
class RunPlan:
    decision: RunDecision
    scheduled_for: datetime  # the occurrence being handled: the stored `due_at` (UTC)
    lateness: timedelta  # how late `now` is (zero when NOT_DUE)
    next_due: datetime | None  # NOT_DUE: `due_at` itself; otherwise the first occurrence strictly after `now` (None: finished)
    latest_missed: datetime | None = None  # MISSED: the most recent occurrence at or before `now`
    missed_count: int = 0  # MISSED: occurrences from `due_at` up to `now`, capped at MAX_MISSED_COUNT


def plan_run(schedule: Schedule, due_at: datetime, now: datetime, grace: timedelta = DEFAULT_GRACE) -> RunPlan:
    """What the situation is for an item due at `due_at`, seen at `now`. A run exactly `grace` late is still a RUN; one
    microsecond later is MISSED. Everything is UTC-normalised; naive times are refused."""
    if not isinstance(grace, timedelta) or grace < timedelta(0):
        raise ScheduleError("grace must be a non-negative duration")
    due_at, now = _as_utc(due_at, "due_at"), _as_utc(now, "now")
    if now < due_at:
        return RunPlan(RunDecision.NOT_DUE, due_at, timedelta(0), due_at)
    lateness = now - due_at
    upcoming = next_due(schedule, now)
    if lateness <= grace:
        return RunPlan(RunDecision.RUN, due_at, lateness, upcoming)
    return RunPlan(
        RunDecision.MISSED, due_at, lateness, upcoming,
        latest_missed=previous_due(schedule, now) or due_at,
        missed_count=_count_from(schedule, due_at, now),
    )


def _count_from(schedule: Schedule, due_at: datetime, now: datetime) -> int:
    """Occurrences from `due_at` (counted as the first) up to and including `now`, capped."""
    if schedule.kind is ScheduleKind.INTERVAL:  # exact arithmetic, no loop
        return min(MAX_MISSED_COUNT, 1 + (now - due_at) // interval_delta(schedule.interval))
    count, occurrence = 1, due_at
    while count < MAX_MISSED_COUNT:
        occurrence = next_due(schedule, occurrence)
        if occurrence is None or occurrence > now:
            break
        count += 1
    return count


def occurrence_key(owner_id: str, scheduled_for: datetime) -> str:
    """The identity of one occurrence of one reminder / automation: `<owner>@<UTC instant to the second>`. The same instant gives
    the same key whatever zone it was written in, and a daylight-saving overlap's single run has a single key."""
    if not isinstance(owner_id, str) or not owner_id or len(owner_id) > MAX_OWNER_ID_CHARS or any(c in owner_id for c in "@\n\r"):
        raise ScheduleError("an occurrence key needs a short owner id without '@' or line breaks")
    instant = _as_utc(scheduled_for, "scheduled_for").astimezone(UTC)
    return f"{owner_id}@{instant.strftime('%Y-%m-%dT%H:%M:%SZ')}"
