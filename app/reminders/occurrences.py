"""
Which schedule describes a stored reminder - pure (no clock, no I/O).

A Reminder V2 reminder carries its own Schedule. A reminder from before V2 has none: a one-time one is simply due at its
`due_at`; a recurring one is "every N units", which is exactly an INTERVAL schedule anchored at its current `due_at`. That
derived schedule is used to DESCRIBE such a reminder and to work out where a PAUSED one resumes (pausing is new in V2). It is
never used to fire one: the scheduler leaves a pre-V2 reminder on its original path, so its missed-run behaviour is unchanged.
"""

from __future__ import annotations

from datetime import datetime

from app.reminders.base import Reminder
from app.scheduling.schedule import Schedule, ScheduleError, ScheduleKind
from app.scheduling.timezones import UTC


def parse_instant(value: str) -> datetime:
    """A stored ISO timestamp as an aware UTC datetime (a naive one is read as UTC, as the rest of the reminder code does)."""
    parsed = datetime.fromisoformat(value)
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


def effective_schedule(reminder: Reminder) -> Schedule | None:
    """The reminder's Schedule: its own (V2), or an interval schedule derived from a legacy recurrence. None for a legacy
    one-time reminder, and for a legacy recurrence too extreme to express (which keeps the original handling)."""
    if reminder.schedule is not None:
        return reminder.schedule
    if reminder.recurrence is None:
        return None
    try:
        return Schedule(ScheduleKind.INTERVAL, "UTC", interval=reminder.recurrence, anchor=parse_instant(reminder.due_at))
    except (ScheduleError, ValueError, OverflowError):
        return None
