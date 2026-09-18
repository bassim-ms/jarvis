"""
Tests for ReminderScheduler (app/reminders/scheduler.py): due-reminder
polling, one-time vs. recurring triggering, failure handling, restart
recovery, and the background start()/stop() thread lifecycle. Uses a real
SQLiteReminderStore (":memory:" or a temp file) and a FakeNotifier -
`tick()` is called directly wherever the test doesn't specifically need
the background thread, matching this module's own documented "tests call
tick() instead of racing a thread" design.
"""

import time
from datetime import datetime, timedelta, timezone

from app.reminders.base import Recurrence, RecurrenceUnit, ReminderStatus
from app.reminders.notifier import Notifier, NotifierError
from app.reminders.scheduler import ReminderScheduler
from app.reminders.sqlite_store import SQLiteReminderStore


def _iso(delta: timedelta) -> str:
    return (datetime.now(timezone.utc) + delta).isoformat()


class FakeNotifier(Notifier):
    def __init__(self, error: Exception | None = None):
        self.error = error
        self.calls: list[tuple[str, str]] = []

    def notify(self, title: str, message: str) -> None:
        self.calls.append((title, message))
        if self.error:
            raise self.error


# --- tick(): basic due handling ----------------------------------------


def test_tick_notifies_for_a_due_reminder():
    store = SQLiteReminderStore(":memory:")
    store.create(text="walk the dog", due_at=_iso(timedelta(minutes=-1)))
    notifier = FakeNotifier()
    scheduler = ReminderScheduler(store=store, notifier=notifier)

    processed = scheduler.tick()

    assert processed == 1
    assert notifier.calls == [("Jarvis Reminder", "walk the dog")]


def test_tick_ignores_reminders_not_yet_due():
    store = SQLiteReminderStore(":memory:")
    store.create(text="later", due_at=_iso(timedelta(minutes=30)))
    notifier = FakeNotifier()
    scheduler = ReminderScheduler(store=store, notifier=notifier)

    processed = scheduler.tick()

    assert processed == 0
    assert notifier.calls == []


def test_tick_processes_every_due_reminder_in_one_pass():
    store = SQLiteReminderStore(":memory:")
    store.create(text="first", due_at=_iso(timedelta(minutes=-5)))
    store.create(text="second", due_at=_iso(timedelta(minutes=-1)))
    notifier = FakeNotifier()
    scheduler = ReminderScheduler(store=store, notifier=notifier)

    processed = scheduler.tick()

    assert processed == 2
    assert {call[1] for call in notifier.calls} == {"first", "second"}


# --- one-time vs. recurring ------------------------------------------------


def test_one_time_reminder_is_not_retriggered_on_a_later_tick():
    store = SQLiteReminderStore(":memory:")
    reminder = store.create(text="once", due_at=_iso(timedelta(minutes=-1)))
    notifier = FakeNotifier()
    scheduler = ReminderScheduler(store=store, notifier=notifier)

    scheduler.tick()
    scheduler.tick()

    assert len(notifier.calls) == 1
    assert store.get(reminder.id).status is ReminderStatus.TRIGGERED


def test_recurring_reminder_advances_and_is_not_immediately_due_again():
    store = SQLiteReminderStore(":memory:")
    reminder = store.create(
        text="stand up",
        due_at=_iso(timedelta(minutes=-1)),
        recurrence=Recurrence(unit=RecurrenceUnit.HOURS, interval=1),
    )
    notifier = FakeNotifier()
    scheduler = ReminderScheduler(store=store, notifier=notifier)

    scheduler.tick()

    assert len(notifier.calls) == 1
    updated = store.get(reminder.id)
    assert updated.status is ReminderStatus.PENDING
    # Next due_at should be about an hour in the future - definitely not due now.
    assert scheduler.tick() == 0
    assert len(notifier.calls) == 1
    assert datetime.fromisoformat(updated.due_at) > datetime.now(timezone.utc)


def test_recurring_reminder_fires_again_once_its_new_due_at_passes():
    store = SQLiteReminderStore(":memory:")
    store.create(
        text="stand up",
        due_at=_iso(timedelta(minutes=-61)),
        recurrence=Recurrence(unit=RecurrenceUnit.HOURS, interval=1),
    )
    notifier = FakeNotifier()
    scheduler = ReminderScheduler(store=store, notifier=notifier)

    # First tick fires it and advances due_at by 1 hour from the (already
    # past) original due_at - which is itself already in the past again,
    # so a second tick fires it once more without waiting a real hour.
    scheduler.tick()
    scheduler.tick()

    assert len(notifier.calls) == 2


# --- notifier failure --------------------------------------------------


def test_failed_notification_is_recorded_but_reminder_stays_due():
    store = SQLiteReminderStore(":memory:")
    reminder = store.create(text="flaky", due_at=_iso(timedelta(minutes=-1)))
    notifier = FakeNotifier(error=NotifierError("toast backend unavailable"))
    scheduler = ReminderScheduler(store=store, notifier=notifier)

    processed = scheduler.tick()

    assert processed == 1
    assert store.get(reminder.id).status is ReminderStatus.PENDING
    executions = store.get_executions(reminder.id)
    assert executions[-1].success is False
    assert "toast backend unavailable" in executions[-1].detail
    # Still due - retried on the very next tick.
    assert scheduler.tick() == 1


# --- unexpected-exception isolation (regression: must not kill the loop) --


def test_tick_isolates_an_unexpected_exception_from_one_reminder():
    """Unlike NotifierError (handled explicitly), a *different* exception
    type escaping Notifier.notify() - simulating an unforeseen bug -
    must not propagate out of tick() itself."""
    store = SQLiteReminderStore(":memory:")
    store.create(text="boom", due_at=_iso(timedelta(minutes=-1)))
    notifier = FakeNotifier(error=RuntimeError("unexpected notifier bug"))
    scheduler = ReminderScheduler(store=store, notifier=notifier)

    processed = scheduler.tick()  # must not raise

    assert processed == 0
    assert notifier.calls == [("Jarvis Reminder", "boom")]


def test_tick_continues_processing_other_reminders_after_one_fails_unexpectedly():
    store = SQLiteReminderStore(":memory:")
    store.create(text="boom", due_at=_iso(timedelta(minutes=-2)))
    ok_reminder = store.create(text="fine", due_at=_iso(timedelta(minutes=-1)))

    class SelectivelyExplodingNotifier(Notifier):
        def __init__(self):
            self.calls: list[tuple[str, str]] = []

        def notify(self, title: str, message: str) -> None:
            self.calls.append((title, message))
            if message == "boom":
                raise RuntimeError("unexpected notifier bug")

    notifier = SelectivelyExplodingNotifier()
    scheduler = ReminderScheduler(store=store, notifier=notifier)

    processed = scheduler.tick()  # must not raise

    assert processed == 1
    assert ("Jarvis Reminder", "fine") in notifier.calls
    assert store.get(ok_reminder.id).status is ReminderStatus.TRIGGERED


def test_tick_isolates_an_overflow_error_computing_the_next_occurrence():
    """A reminder with an extreme recurrence interval (bypassing
    CreateReminderTool's own new validation - e.g. persisted some other
    way) makes _advance_due_at raise OverflowError. That must not escape
    tick() and kill the background thread either."""
    store = SQLiteReminderStore(":memory:")
    store.create(
        text="extreme",
        due_at=_iso(timedelta(minutes=-1)),
        recurrence=Recurrence(unit=RecurrenceUnit.WEEKS, interval=10**9),
    )
    notifier = FakeNotifier()
    scheduler = ReminderScheduler(store=store, notifier=notifier)

    processed = scheduler.tick()  # must not raise

    assert processed == 0
    assert notifier.calls == [("Jarvis Reminder", "extreme")]


def test_background_thread_survives_a_failed_poll_and_keeps_running():
    """The outer safety net in _run(): a store failure *outside* any
    single reminder (e.g. list_due() itself raising, simulating a
    transient SQLite error) must not kill the polling thread - it should
    simply retry on the next scheduled poll."""
    import sqlite3

    class FlakyListDueStore(SQLiteReminderStore):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.list_due_calls = 0

        def list_due(self, now):
            self.list_due_calls += 1
            if self.list_due_calls == 1:
                raise sqlite3.OperationalError("database is locked")
            return super().list_due(now)

    store = FlakyListDueStore(":memory:")
    store.create(text="after recovery", due_at=_iso(timedelta(minutes=-1)))
    notifier = FakeNotifier()
    scheduler = ReminderScheduler(
        store=store, notifier=notifier, poll_interval_seconds=0.05
    )

    scheduler.start()
    try:
        deadline = time.monotonic() + 2.0
        while not notifier.calls and time.monotonic() < deadline:
            time.sleep(0.01)
    finally:
        scheduler.stop()

    assert store.list_due_calls >= 2
    assert notifier.calls == [("Jarvis Reminder", "after recovery")]


# --- restart recovery --------------------------------------------------


def test_scheduler_recovers_reminders_due_while_jarvis_was_offline(tmp_path):
    """Simulates a restart: a reminder becomes due while no scheduler is
    running, then a brand-new store/scheduler pointed at the same on-disk
    database (as app/main.py's real wiring would build after a process
    restart) picks it up on the very first poll - no separate recovery
    step required."""
    db_path = str(tmp_path / "reminders.db")

    startup_store = SQLiteReminderStore(db_path)
    reminder = startup_store.create(
        text="missed while offline", due_at=_iso(timedelta(hours=-2))
    )
    del startup_store  # simulate the process exiting

    restarted_store = SQLiteReminderStore(db_path)
    notifier = FakeNotifier()
    scheduler = ReminderScheduler(store=restarted_store, notifier=notifier)

    processed = scheduler.tick()

    assert processed == 1
    assert notifier.calls == [("Jarvis Reminder", "missed while offline")]
    assert restarted_store.get(reminder.id).status is ReminderStatus.TRIGGERED


def test_scheduler_recovers_a_recurring_reminder_after_restart(tmp_path):
    db_path = str(tmp_path / "reminders.db")

    startup_store = SQLiteReminderStore(db_path)
    startup_store.create(
        text="daily check-in",
        due_at=_iso(timedelta(hours=-25)),
        recurrence=Recurrence(unit=RecurrenceUnit.DAYS, interval=1),
    )
    del startup_store

    restarted_store = SQLiteReminderStore(db_path)
    notifier = FakeNotifier()
    scheduler = ReminderScheduler(store=restarted_store, notifier=notifier)

    processed = scheduler.tick()

    assert processed == 1
    assert notifier.calls == [("Jarvis Reminder", "daily check-in")]


# --- background thread lifecycle ----------------------------------------


def test_start_runs_tick_in_the_background_until_stopped():
    store = SQLiteReminderStore(":memory:")
    store.create(text="background", due_at=_iso(timedelta(minutes=-1)))
    notifier = FakeNotifier()
    scheduler = ReminderScheduler(store=store, notifier=notifier, poll_interval_seconds=0.05)

    scheduler.start()
    try:
        deadline = time.monotonic() + 2.0
        while not notifier.calls and time.monotonic() < deadline:
            time.sleep(0.01)
    finally:
        scheduler.stop()

    assert notifier.calls == [("Jarvis Reminder", "background")]


def test_start_is_a_no_op_when_already_running():
    store = SQLiteReminderStore(":memory:")
    scheduler = ReminderScheduler(store=store, notifier=FakeNotifier(), poll_interval_seconds=1)

    scheduler.start()
    first_thread = scheduler._thread
    scheduler.start()

    assert scheduler._thread is first_thread
    scheduler.stop()


def test_stop_is_a_no_op_when_not_running():
    store = SQLiteReminderStore(":memory:")
    scheduler = ReminderScheduler(store=store, notifier=FakeNotifier())

    scheduler.stop()  # must not raise
