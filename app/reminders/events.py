"""
The notification a due reminder becomes: a plain, channel-neutral `NotificationEvent`
(app/notifications/base.py), like the Job and interaction ones. The reminder's own text is the
user's, cleaned and bounded; it is display text only - it can never mark the event important or
urgent (a reminder is NORMAL to the attention layer, by its kind). Reminders carry no structured
"deadline" or "important" flag today, so none is invented from their words.

A reminder that came due while Jarvis was not running (or could not deliver) is announced ONCE, as a catch-up: the same
event key as the occurrence it stands for, wording that says it was missed and how many times it came due. Missed
occurrences are never announced one by one.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from app.notifications.base import NotificationEvent, NotificationKind, NotificationPriority
from app.notifications.text import address_for, clean_text
from app.reminders.base import Reminder
from app.reminders.timefmt import local_phrase

_TEXT_LIMIT = 300


@dataclass(frozen=True)
class CatchUp:
    """A reminder announced late: how many times it came due while it could not be announced, and the latest of them."""

    missed_count: int
    latest_missed: datetime
    tz: str | None = None


def reminder_key(reminder: Reminder) -> str:
    """One key per OCCURRENCE (a recurring reminder is a new event each time it comes due)."""
    return f"reminder:{reminder.id}:{reminder.due_at}"


def reminder_event(reminder: Reminder, catch_up: CatchUp | None = None) -> NotificationEvent:
    key = reminder_key(reminder)
    body = clean_text(reminder.text, _TEXT_LIMIT) or "you asked me to remind you"
    if catch_up is None:
        text = f"{address_for(key)}, a reminder: {body}"
    else:
        how = "it was due" if catch_up.missed_count <= 1 else f"it came due {catch_up.missed_count} times"
        last = local_phrase(catch_up.latest_missed, catch_up.tz)
        text = f"{address_for(key)}, a reminder I missed while I was offline ({how}; the last was {last}): {body}"
    return NotificationEvent(
        key=key,
        kind=NotificationKind.REMINDER,
        priority=NotificationPriority.NORMAL,
        text=text,
    )
