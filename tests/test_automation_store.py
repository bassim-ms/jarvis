"""
Unit tests for SQLiteAutomationStore (app/automation/sqlite_store.py):
persistence, trigger/condition/action round-tripping, execution-history
logging, and enable/disable/cancel. Uses real SQLite (":memory:" or a
temp file), never mocked - same convention as test_reminders_store.py.
"""

from datetime import datetime, timedelta, timezone

from app.automation.base import (
    AutomationStatus,
    ExecutionOutcome,
    FactEqualsCondition,
    TimeWindowCondition,
    Trigger,
    TriggerType,
)
from app.automation.sqlite_store import SQLiteAutomationStore
from app.reminders.base import Recurrence, RecurrenceUnit
from app.tools.decision import ToolCall


def _iso(delta: timedelta) -> str:
    return (datetime.now(timezone.utc) + delta).isoformat()


def make_store() -> SQLiteAutomationStore:
    return SQLiteAutomationStore(":memory:")


def _manual_trigger() -> Trigger:
    return Trigger(type=TriggerType.MANUAL)


def _action() -> ToolCall:
    return ToolCall(tool_name="echo", parameters={"text": "hi"})


# --- create/get/list ---------------------------------------------------


def test_create_returns_an_enabled_automation():
    store = make_store()

    automation = store.create(
        name="say hi", trigger=_manual_trigger(), conditions=[], action=_action()
    )

    assert automation.id
    assert automation.name == "say hi"
    assert automation.status is AutomationStatus.ENABLED
    assert automation.trigger.type is TriggerType.MANUAL
    assert automation.action == _action()
    assert automation.next_run_at is None


def test_create_persists_a_time_trigger_with_recurrence():
    store = make_store()
    next_run = _iso(timedelta(minutes=5))

    automation = store.create(
        name="daily",
        trigger=Trigger(
            type=TriggerType.TIME,
            recurrence=Recurrence(unit=RecurrenceUnit.DAYS, interval=1),
        ),
        conditions=[],
        action=_action(),
        next_run_at=next_run,
    )

    fetched = store.get(automation.id)
    assert fetched.trigger.type is TriggerType.TIME
    assert fetched.trigger.recurrence == Recurrence(unit=RecurrenceUnit.DAYS, interval=1)
    assert fetched.next_run_at == next_run


def test_create_persists_a_one_time_time_trigger_with_no_recurrence():
    store = make_store()

    automation = store.create(
        name="once", trigger=Trigger(type=TriggerType.TIME), conditions=[], action=_action()
    )

    fetched = store.get(automation.id)
    assert fetched.trigger.recurrence is None


def test_create_persists_a_reminder_completed_trigger():
    store = make_store()

    automation = store.create(
        name="on complete",
        trigger=Trigger(type=TriggerType.REMINDER_COMPLETED, reminder_id="rem-1"),
        conditions=[],
        action=_action(),
    )

    fetched = store.get(automation.id)
    assert fetched.trigger.type is TriggerType.REMINDER_COMPLETED
    assert fetched.trigger.reminder_id == "rem-1"


def test_create_persists_conditions_round_trip():
    store = make_store()
    conditions = [
        FactEqualsCondition(key="mode", value="work"),
        TimeWindowCondition(start_hour=9, end_hour=18),
    ]

    automation = store.create(
        name="gated", trigger=_manual_trigger(), conditions=conditions, action=_action()
    )

    fetched = store.get(automation.id)
    assert fetched.conditions == conditions


def test_get_unknown_automation_returns_none():
    store = make_store()

    assert store.get("does-not-exist") is None


def test_list_returns_all_automations_most_recent_first():
    store = make_store()
    first = store.create(
        name="first", trigger=_manual_trigger(), conditions=[], action=_action()
    )
    second = store.create(
        name="second", trigger=_manual_trigger(), conditions=[], action=_action()
    )

    automations = store.list()

    assert [a.id for a in automations] == [second.id, first.id]


def test_list_filters_by_status():
    store = make_store()
    enabled = store.create(
        name="enabled one", trigger=_manual_trigger(), conditions=[], action=_action()
    )
    cancelled = store.create(
        name="cancelled one", trigger=_manual_trigger(), conditions=[], action=_action()
    )
    store.cancel(cancelled.id)

    enabled_only = store.list(status=AutomationStatus.ENABLED)

    assert [a.id for a in enabled_only] == [enabled.id]


# --- record_execution: recurring vs one-shot ----------------------------


def test_record_execution_advances_a_recurring_time_trigger():
    store = make_store()
    automation = store.create(
        name="daily",
        trigger=Trigger(
            type=TriggerType.TIME,
            recurrence=Recurrence(unit=RecurrenceUnit.DAYS, interval=1),
        ),
        conditions=[],
        action=_action(),
        next_run_at=_iso(timedelta(minutes=-1)),
    )
    next_run = _iso(timedelta(days=1))

    store.record_execution(
        automation.id,
        executed_at=_iso(timedelta()),
        outcome=ExecutionOutcome.SUCCESS,
        next_run_at=next_run,
    )

    fetched = store.get(automation.id)
    assert fetched.status is AutomationStatus.ENABLED
    assert fetched.next_run_at == next_run


def test_record_execution_disables_a_one_time_time_trigger():
    store = make_store()
    automation = store.create(
        name="once",
        trigger=Trigger(type=TriggerType.TIME),
        conditions=[],
        action=_action(),
        next_run_at=_iso(timedelta(minutes=-1)),
    )

    store.record_execution(
        automation.id,
        executed_at=_iso(timedelta()),
        outcome=ExecutionOutcome.SUCCESS,
        disable=True,
    )

    assert store.get(automation.id).status is AutomationStatus.DISABLED


def test_record_execution_disables_a_reminder_completed_trigger():
    store = make_store()
    automation = store.create(
        name="on complete",
        trigger=Trigger(type=TriggerType.REMINDER_COMPLETED, reminder_id="rem-1"),
        conditions=[],
        action=_action(),
    )

    store.record_execution(
        automation.id,
        executed_at=_iso(timedelta()),
        outcome=ExecutionOutcome.SUCCESS,
        disable=True,
    )

    assert store.get(automation.id).status is AutomationStatus.DISABLED


def test_record_execution_disable_never_resurrects_a_cancelled_automation():
    store = make_store()
    automation = store.create(
        name="x", trigger=_manual_trigger(), conditions=[], action=_action()
    )
    store.cancel(automation.id)

    store.record_execution(
        automation.id,
        executed_at=_iso(timedelta()),
        outcome=ExecutionOutcome.SUCCESS,
        disable=True,
    )

    assert store.get(automation.id).status is AutomationStatus.CANCELLED


def test_record_execution_always_logs_an_execution():
    store = make_store()
    automation = store.create(
        name="x", trigger=_manual_trigger(), conditions=[], action=_action()
    )

    store.record_execution(
        automation.id,
        executed_at="2026-01-01T00:00:00+00:00",
        outcome=ExecutionOutcome.SUCCESS,
        detail="ok",
    )
    store.record_execution(
        automation.id,
        executed_at="2026-01-01T01:00:00+00:00",
        outcome=ExecutionOutcome.FAILURE,
        detail="boom",
    )
    store.record_execution(
        automation.id,
        executed_at="2026-01-01T02:00:00+00:00",
        outcome=ExecutionOutcome.SKIPPED,
        detail="condition not met",
    )

    executions = store.get_executions(automation.id)
    assert [e.outcome for e in executions] == [
        ExecutionOutcome.SUCCESS,
        ExecutionOutcome.FAILURE,
        ExecutionOutcome.SKIPPED,
    ]
    assert executions[1].detail == "boom"


# --- enable/disable/cancel ----------------------------------------------


def test_disable_then_enable_round_trip():
    store = make_store()
    automation = store.create(
        name="x", trigger=_manual_trigger(), conditions=[], action=_action()
    )

    assert store.disable(automation.id) is True
    assert store.get(automation.id).status is AutomationStatus.DISABLED

    assert store.enable(automation.id) is True
    assert store.get(automation.id).status is AutomationStatus.ENABLED


def test_enable_unknown_automation_returns_false():
    store = make_store()

    assert store.enable("does-not-exist") is False


def test_cancel_is_terminal_and_blocks_enable_and_disable():
    store = make_store()
    automation = store.create(
        name="x", trigger=_manual_trigger(), conditions=[], action=_action()
    )

    assert store.cancel(automation.id) is True
    assert store.get(automation.id).status is AutomationStatus.CANCELLED
    assert store.cancel(automation.id) is False
    assert store.enable(automation.id) is False
    assert store.disable(automation.id) is False


# --- persistence across store instances (restart) ------------------------


def test_automations_persist_across_store_instances(tmp_path):
    db_path = str(tmp_path / "automations.db")
    first_store = SQLiteAutomationStore(db_path)
    automation = first_store.create(
        name="survive restart",
        trigger=Trigger(
            type=TriggerType.TIME,
            recurrence=Recurrence(unit=RecurrenceUnit.HOURS, interval=2),
        ),
        conditions=[FactEqualsCondition(key="k", value="v")],
        action=ToolCall(tool_name="echo", parameters={"text": "hi"}),
        next_run_at=_iso(timedelta(minutes=5)),
    )

    second_store = SQLiteAutomationStore(db_path)

    fetched = second_store.get(automation.id)
    assert fetched is not None
    assert fetched.name == "survive restart"
    assert fetched.trigger.recurrence == Recurrence(unit=RecurrenceUnit.HOURS, interval=2)
    assert fetched.conditions == [FactEqualsCondition(key="k", value="v")]
    assert fetched.action == ToolCall(tool_name="echo", parameters={"text": "hi"})
