"""
Unit tests for SQLiteReminderStore (app/reminders/sqlite_store.py):
persistence, one-time vs. recurring due handling, lifecycle/execution
logging, and complete/cancel. Uses real SQLite (":memory:" or a temp
file), never mocked - same convention as test_memory.py/test_search_cache.py.
"""

from datetime import datetime, timedelta, timezone

from app.reminders.base import Recurrence, RecurrenceUnit, ReminderStatus
from app.reminders.sqlite_store import SQLiteReminderStore


def _iso(delta: timedelta) -> str:
    return (datetime.now(timezone.utc) + delta).isoformat()


def make_store() -> SQLiteReminderStore:
    return SQLiteReminderStore(":memory:")


# --- create/get/list -------------------------------------------------------


def test_create_returns_a_pending_reminder():
    store = make_store()

    reminder = store.create(text="water the plants", due_at=_iso(timedelta(minutes=5)))

    assert reminder.id
    assert reminder.text == "water the plants"
    assert reminder.status is ReminderStatus.PENDING
    assert reminder.recurrence is None


def test_create_persists_recurrence():
    store = make_store()

    reminder = store.create(
        text="stand up",
        due_at=_iso(timedelta(minutes=1)),
        recurrence=Recurrence(unit=RecurrenceUnit.HOURS, interval=2),
    )

    fetched = store.get(reminder.id)
    assert fetched.recurrence == Recurrence(unit=RecurrenceUnit.HOURS, interval=2)


def test_get_unknown_reminder_returns_none():
    store = make_store()

    assert store.get("does-not-exist") is None


def test_list_returns_all_reminders_most_recent_first():
    store = make_store()
    first = store.create(text="first", due_at=_iso(timedelta(minutes=1)))
    second = store.create(text="second", due_at=_iso(timedelta(minutes=2)))

    reminders = store.list()

    assert [r.id for r in reminders] == [second.id, first.id]


def test_list_filters_by_status():
    store = make_store()
    pending = store.create(text="pending one", due_at=_iso(timedelta(minutes=5)))
    completed = store.create(text="completed one", due_at=_iso(timedelta(minutes=5)))
    store.complete(completed.id)

    pending_only = store.list(status=ReminderStatus.PENDING)

    assert [r.id for r in pending_only] == [pending.id]


# --- list_due ---------------------------------------------------------------


def test_list_due_returns_only_past_due_pending_reminders():
    store = make_store()
    due = store.create(text="overdue", due_at=_iso(timedelta(minutes=-5)))
    store.create(text="future", due_at=_iso(timedelta(minutes=30)))

    due_reminders = store.list_due(_iso(timedelta()))

    assert [r.id for r in due_reminders] == [due.id]


def test_list_due_excludes_non_pending_reminders():
    store = make_store()
    reminder = store.create(text="overdue", due_at=_iso(timedelta(minutes=-5)))
    store.cancel(reminder.id)

    assert store.list_due(_iso(timedelta())) == []


# --- record_trigger: one-time vs. recurring ---------------------------------


def test_record_trigger_marks_one_time_reminder_triggered():
    store = make_store()
    reminder = store.create(text="one-time", due_at=_iso(timedelta(minutes=-1)))

    store.record_trigger(reminder.id, triggered_at=_iso(timedelta()), success=True)

    fetched = store.get(reminder.id)
    assert fetched.status is ReminderStatus.TRIGGERED
    assert store.list_due(_iso(timedelta())) == []


def test_record_trigger_advances_recurring_reminder_and_stays_pending():
    store = make_store()
    original_due = _iso(timedelta(minutes=-1))
    reminder = store.create(
        text="recurring",
        due_at=original_due,
        recurrence=Recurrence(unit=RecurrenceUnit.HOURS, interval=1),
    )
    next_due = _iso(timedelta(hours=1))

    store.record_trigger(
        reminder.id,
        triggered_at=_iso(timedelta()),
        success=True,
        next_due_at=next_due,
    )

    fetched = store.get(reminder.id)
    assert fetched.status is ReminderStatus.PENDING
    assert fetched.due_at == next_due
    # No longer due immediately - it won't refire until the new due_at.
    assert store.list_due(_iso(timedelta())) == []


def test_record_trigger_failure_leaves_reminder_pending_and_due():
    store = make_store()
    reminder = store.create(text="flaky", due_at=_iso(timedelta(minutes=-1)))

    store.record_trigger(
        reminder.id,
        triggered_at=_iso(timedelta()),
        success=False,
        detail="notifier exploded",
    )

    fetched = store.get(reminder.id)
    assert fetched.status is ReminderStatus.PENDING
    # Still due - the next poll should retry it.
    assert [r.id for r in store.list_due(_iso(timedelta()))] == [reminder.id]


def test_record_trigger_always_logs_an_execution():
    store = make_store()
    reminder = store.create(text="logged", due_at=_iso(timedelta(minutes=-1)))

    store.record_trigger(
        reminder.id, triggered_at="2026-01-01T00:00:00+00:00", success=True
    )
    store.record_trigger(
        reminder.id,
        triggered_at="2026-01-01T01:00:00+00:00",
        success=False,
        detail="oops",
    )

    executions = store.get_executions(reminder.id)
    assert [e.triggered_at for e in executions] == [
        "2026-01-01T00:00:00+00:00",
        "2026-01-01T01:00:00+00:00",
    ]
    assert executions[0].success is True
    assert executions[1].success is False
    assert executions[1].detail == "oops"


# --- complete/cancel ---------------------------------------------------------


def test_complete_marks_reminder_completed_and_returns_true():
    store = make_store()
    reminder = store.create(text="done soon", due_at=_iso(timedelta(minutes=5)))

    assert store.complete(reminder.id) is True
    assert store.get(reminder.id).status is ReminderStatus.COMPLETED


def test_complete_unknown_reminder_returns_false():
    store = make_store()

    assert store.complete("does-not-exist") is False


def test_complete_already_completed_reminder_returns_false():
    store = make_store()
    reminder = store.create(text="done soon", due_at=_iso(timedelta(minutes=5)))
    store.complete(reminder.id)

    assert store.complete(reminder.id) is False


def test_cancel_marks_reminder_cancelled_and_returns_true():
    store = make_store()
    reminder = store.create(text="never mind", due_at=_iso(timedelta(minutes=5)))

    assert store.cancel(reminder.id) is True
    assert store.get(reminder.id).status is ReminderStatus.CANCELLED


def test_cancel_stops_a_recurring_reminder_from_ever_being_due_again():
    store = make_store()
    reminder = store.create(
        text="recurring",
        due_at=_iso(timedelta(minutes=-1)),
        recurrence=Recurrence(unit=RecurrenceUnit.DAYS, interval=1),
    )

    store.cancel(reminder.id)

    assert store.list_due(_iso(timedelta())) == []


# --- persistence across store instances (same file) -------------------------


def test_reminders_persist_across_store_instances(tmp_path):
    db_path = str(tmp_path / "reminders.db")
    first_store = SQLiteReminderStore(db_path)
    reminder = first_store.create(text="survive restart", due_at=_iso(timedelta(minutes=5)))

    second_store = SQLiteReminderStore(db_path)

    fetched = second_store.get(reminder.id)
    assert fetched is not None
    assert fetched.text == "survive restart"
