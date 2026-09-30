"""
The notification a due reminder becomes: a plain, channel-neutral `NotificationEvent`
(app/notifications/base.py), like the Job and interaction ones. The reminder's own text is the
user's, cleaned and bounded; it is display text only - it can never mark the event important or
urgent (a reminder is NORMAL to the attention layer, by its kind). Reminders carry no structured
"deadline" or "important" flag today, so none is invented from their words.
"""

from __future__ import annotations

from app.notifications.base import NotificationEvent, NotificationKind, NotificationPriority
from app.notifications.text import address_for, clean_text
from app.reminders.base import Reminder

_TEXT_LIMIT = 300


def reminder_key(reminder: Reminder) -> str:
    """One key per OCCURRENCE (a recurring reminder is a new event each time it comes due)."""
    return f"reminder:{reminder.id}:{reminder.due_at}"


def reminder_event(reminder: Reminder) -> NotificationEvent:
    key = reminder_key(reminder)
    body = clean_text(reminder.text, _TEXT_LIMIT) or "you asked me to remind you"
    return NotificationEvent(
        key=key,
        kind=NotificationKind.REMINDER,
        priority=NotificationPriority.NORMAL,
        text=f"{address_for(key)}, a reminder: {body}",
    )
