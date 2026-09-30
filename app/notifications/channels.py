"""
Concrete notification channels. Today: the desktop toast, reusing the
Notifier reminders already use. Future channels (speech, phone, watch,
earbuds, glasses) are further NotificationChannel subclasses; none exists yet.
"""

from __future__ import annotations

from app.notifications.base import NotificationChannel, NotificationEvent, NotificationPriority
from app.reminders.notifier import Notifier


class DesktopToastChannel(NotificationChannel):
    """Shows an event's text as a desktop notification. `min_priority` lets a
    deployment keep LOW (informational) events off the desktop."""

    name = "desktop"

    def __init__(
        self,
        notifier: Notifier,
        *,
        min_priority: NotificationPriority = NotificationPriority.LOW,
        title: str = "Jarvis",
    ):
        self.notifier = notifier
        self.min_priority = min_priority
        self.title = title

    def deliver(self, user: str, event: NotificationEvent) -> None:
        self.notifier.notify(self.title, event.text)
