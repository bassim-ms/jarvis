"""
Builds the application's default ReminderStore/Notifier/ReminderScheduler
from config - the one place a platform check happens, same role
app/search/factory.py plays for the default search provider/cache chain.
"""

import platform

from app.reminders.base import ReminderStore
from app.reminders.notifier import ConsoleNotifier, Notifier, WindowsToastNotifier
from app.reminders.scheduler import ReminderScheduler
from app.reminders.sqlite_store import SQLiteReminderStore


def create_default_notifier() -> Notifier:
    """WindowsToastNotifier on Windows (the only platform Jarvis targets
    today - see README); ConsoleNotifier elsewhere, so tests and any
    non-Windows dev environment still run without a native toast backend."""
    if platform.system() == "Windows":
        return WindowsToastNotifier()
    return ConsoleNotifier()


def create_default_reminder_store(db_path: str) -> ReminderStore:
    return SQLiteReminderStore(db_path)


def create_default_scheduler(
    store: ReminderStore, poll_interval_seconds: float, router=None
) -> ReminderScheduler:
    return ReminderScheduler(
        store=store,
        notifier=create_default_notifier(),
        poll_interval_seconds=poll_interval_seconds,
        router=router,
    )
