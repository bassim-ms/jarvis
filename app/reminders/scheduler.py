"""
Reminder Scheduler (M7): a lightweight background poller that turns due
Reminders into notifications.

Deliberately not a general task/automation engine: it only ever calls
`Notifier.notify(...)` (app/reminders/notifier.py) for a Reminder that's
already persisted and already come due (per `ReminderStore.list_due`) - it
never executes a registered Tool, never runs arbitrary code, and never
accepts a schedule for anything other than a Reminder's own free text.
That narrowness is the whole security boundary this milestone adds:
persisted "do this later" state can only ever mean "show a notification,"
never "run this tool/command" - see app/reminders/base.py.

Survives a Jarvis restart with no special recovery code: `list_due` is a
plain "due_at <= now" query against ReminderStore's persistent SQLite
backing, so a Reminder that became due while Jarvis was offline is simply
found and triggered on the first poll after startup, exactly like any
other due reminder - restart recovery is a property of persistence, not a
poller feature.

`advance_by_recurrence` (app/reminders/base.py) is shared with
AutomationEngine (app/automation/engine.py, M8) - a recurring Automation's
`next_run_at` advances exactly the same way a recurring Reminder's
`due_at` does, rather than a second copy of the same interval math.
"""

import threading
from datetime import datetime, timezone

from app.reminders.base import Reminder, ReminderStore, advance_by_recurrence
from app.reminders.notifier import Notifier, NotifierError

#: How often ReminderScheduler polls by default when none is configured -
#: frequent enough that a reminder fires within about half a minute of
#: coming due, without hammering ReminderStore's SQLite connection.
DEFAULT_POLL_INTERVAL_SECONDS = 30.0


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ReminderScheduler:
    """Polls `store` for due reminders and triggers `notifier` for each, on
    its own background thread.

    `tick()` is the actual check-and-fire step, fully synchronous and with
    all its side effects visible before it returns - tests call it
    directly instead of racing a background thread; `start()`/`stop()`
    just wrap it in a daemon polling loop for the running application.
    """

    def __init__(
        self,
        store: ReminderStore,
        notifier: Notifier,
        poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
        router=None,
    ):
        self.store = store
        self.notifier = notifier
        # Optional (duck-typed like a NotificationRouter: `notify(user, event)`). When given, a due
        # reminder is emitted as a structured NotificationEvent and handed to it - in the app, the
        # attention-gated router, which decides whether the user is interrupted or the reminder waits.
        # None (the default, every existing caller/test): the notifier is called directly, as always.
        self.router = router
        self.poll_interval_seconds = poll_interval_seconds
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        """Start the background polling thread. No-op if already running."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name="jarvis-reminder-scheduler", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float | None = 5.0) -> None:
        """Signal the background thread to stop and wait for it to exit.
        No-op if it isn't running."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.tick()
            except Exception:  # noqa: BLE001 - a failed poll must not kill the loop
                # E.g. store.list_due() itself failing (a transient SQLite
                # error, a closed connection, ...) - outside tick()'s own
                # per-reminder isolation below. Swallowed rather than
                # logged since this codebase has no logging setup yet;
                # the next poll simply retries.
                pass
            self._stop_event.wait(self.poll_interval_seconds)

    def _emit(self, reminder: Reminder) -> tuple[bool, str]:
        """Hand the reminder to the router as an event. "Success" means the router took responsibility for
        it (delivered it or held it for a better moment) - it is not lost either way."""
        from app.notifications.base import DEFAULT_USER
        from app.reminders.events import reminder_event

        try:
            result = self.router.notify(DEFAULT_USER, reminder_event(reminder))
        except Exception as exc:  # noqa: BLE001 - notifying never breaks the scheduler
            return False, str(exc)
        if result.accepted or result.duplicate:
            return True, ""
        return False, "the notification could not be accepted"

    def tick(self) -> int:
        """Trigger every currently-due reminder once and return how many
        were successfully processed (regardless of whether notifying each
        one succeeded).

        Never raises past a single reminder: each one is isolated in its
        own try/except, the same failure-isolation philosophy
        ToolRegistry.execute (app/tools/registry.py) already applies to a
        single misbehaving tool - so one reminder failing in an
        unexpected way (e.g. a corrupt recurrence, a transient store
        error) can't stop the rest of this poll, and can't kill the
        background thread (see `_run`)."""
        due = self.store.list_due(_now())
        processed = 0
        for reminder in due:
            try:
                self._trigger(reminder)
            except Exception:  # noqa: BLE001 - isolate one reminder's failure
                continue
            processed += 1
        return processed

    def _trigger(self, reminder: Reminder) -> None:
        if self.router is not None:
            success, detail = self._emit(reminder)
        else:
            try:
                self.notifier.notify(title="Jarvis Reminder", message=reminder.text)
                success, detail = True, ""
            except NotifierError as exc:
                success, detail = False, str(exc)

        next_due_at = (
            advance_by_recurrence(reminder.due_at, reminder.recurrence)
            if success and reminder.recurrence is not None
            else None
        )

        self.store.record_trigger(
            reminder.id,
            triggered_at=_now(),
            success=success,
            detail=detail,
            next_due_at=next_due_at,
        )
