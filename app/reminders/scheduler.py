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

Reminder V2: a reminder with a Schedule is handled with `plan_run` (app/scheduling/runs.py).
On time, or up to the grace window late: it is announced and moved to its next
occurrence. Later than that (Jarvis was off): ONE catch-up announcement stands for
every occurrence that was missed, and the reminder jumps to its next FUTURE
occurrence - there is never a replay of each missed one. The update that records
"handled" is a compare-and-set on the occurrence, and an occurrence is recorded as
delivered at most once, so two polls (or two processes) cannot announce it twice.
A reminder created before V2 has no Schedule and is NOT touched by any of this - one-time or recurring, it is handled
exactly as it always was (a recurring one moves on by one interval per announcement, so what it missed is replayed one
occurrence per poll, as before).
This is still ONLY a notification: it never makes a Job, calls a Planner or runs a tool.
"""

import threading
from datetime import datetime, timedelta, timezone

from app.reminders.base import Reminder, ReminderStore, advance_by_recurrence
from app.reminders.events import CatchUp
from app.reminders.notifier import Notifier, NotifierError
from app.reminders.occurrences import parse_instant
from app.scheduling.runs import DEFAULT_GRACE, RunDecision, occurrence_key, plan_run

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
        clock=None,
        grace: timedelta = DEFAULT_GRACE,
    ):
        self.store = store
        self.notifier = notifier
        # An injectable clock (a callable returning an aware datetime) so restart and missed-run behaviour is testable
        # without waiting; None (every existing caller) is the real clock. `grace` is how late an occurrence may be and
        # still simply run (settings.schedule_grace_seconds in the app).
        self._clock = clock
        self.grace = grace
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

    def is_running(self) -> bool:
        """True while the background thread is alive (read by the System health view - app/diagnostics)."""
        return self._thread is not None and self._thread.is_alive()

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

    def _now_dt(self) -> datetime:
        return (self._clock() if self._clock is not None else datetime.now(timezone.utc)).astimezone(timezone.utc)

    def _emit(self, reminder: Reminder, catch_up: CatchUp | None = None) -> tuple[bool, str]:
        """Hand the reminder to the router as an event. "Success" means the router took responsibility for
        it (delivered it or held it for a better moment) - it is not lost either way."""
        from app.notifications.base import DEFAULT_USER
        from app.reminders.events import reminder_event

        try:
            result = self.router.notify(DEFAULT_USER, reminder_event(reminder, catch_up))
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
        now = self._now_dt()
        due = self.store.list_due(now.isoformat())
        processed = 0
        for reminder in due:
            try:
                self._trigger(reminder, now)
            except Exception:  # noqa: BLE001 - isolate one reminder's failure
                continue
            processed += 1
        return processed

    def _trigger(self, reminder: Reminder, now: datetime | None = None) -> None:
        now = now or self._now_dt()
        if reminder.schedule_invalid:
            raise ValueError("this reminder's stored schedule could not be read")
        schedule = reminder.schedule  # only a V2 reminder has one; a pre-V2 reminder takes the original path below
        if schedule is not None:
            self._trigger_scheduled(reminder, schedule, now)
            return
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

    def _trigger_scheduled(self, reminder: Reminder, schedule, now: datetime) -> None:
        """Handle one due occurrence of a scheduled (V2) reminder: announce it once, then move to the next
        FUTURE occurrence. See the module docstring for the missed-run rule."""
        due_at = parse_instant(reminder.due_at)
        plan = plan_run(schedule, due_at, now, self.grace)
        if plan.decision is RunDecision.NOT_DUE:
            return  # the clock moved back since it was listed: leave it for a later poll
        catch_up = None
        detail_ok = ""
        if plan.decision is RunDecision.MISSED:
            catch_up = CatchUp(plan.missed_count, plan.latest_missed or due_at, reminder.timezone)
            detail_ok = f"caught up: it came due {plan.missed_count} time(s) while Jarvis could not announce it"

        if self.router is not None:
            success, detail = self._emit(reminder, catch_up)
        else:
            try:
                self.notifier.notify(title="Jarvis Reminder", message=reminder.text)
                success, detail = True, ""
            except NotifierError as exc:
                success, detail = False, str(exc)

        self.store.record_trigger(
            reminder.id,
            triggered_at=now.isoformat(),
            success=success,
            detail=detail if not success else detail_ok,
            next_due_at=plan.next_due.isoformat() if success and plan.next_due is not None else None,
            occurrence=occurrence_key(reminder.id, due_at),
            expected_due_at=reminder.due_at,
        )
