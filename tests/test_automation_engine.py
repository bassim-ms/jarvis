"""
Tests for AutomationEngine (app/automation/engine.py): due-automation
collection across all three trigger types, condition gating, trigger
advancement/retirement, permission enforcement (defensive, even for a
hand-crafted automation that bypassed CreateAutomationTool's own
validation), restart recovery, failure isolation, and the background
start()/stop() thread lifecycle. Uses a real SQLiteAutomationStore/
SQLiteReminderStore/SQLiteMemoryStore (":memory:" or a temp file) and
small fake Tools - `tick()`/`run_now()` are called directly wherever a
test doesn't specifically need the background thread, matching
ReminderScheduler's own tests (tests/test_reminders_scheduler.py).
"""

import time
from datetime import datetime, timedelta, timezone

import pytest

from app.automation.base import (
    Automation,
    AutomationStatus,
    ExecutionOutcome,
    FactEqualsCondition,
    Trigger,
    TriggerType,
)
from app.automation.engine import AutomationEngine, AutomationRunError
from app.automation.sqlite_store import SQLiteAutomationStore
from app.memory.sqlite_store import SQLiteMemoryStore
from app.reminders.base import Recurrence, RecurrenceUnit
from app.reminders.sqlite_store import SQLiteReminderStore
from app.tools.base import PermissionLevel, Tool, ToolParameter, ToolResult
from app.tools.decision import ToolCall
from app.tools.registry import ToolRegistry


def _iso(delta: timedelta) -> str:
    return (datetime.now(timezone.utc) + delta).isoformat()


class RecordingTool(Tool):
    """Fake SAFE_ACTION tool: records every call, always succeeds."""

    name = "recording_tool"
    description = "Records each call."
    permission = PermissionLevel.SAFE_ACTION

    def __init__(self):
        self.calls: list[str] = []

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="value", type="string", required=False, default="", description=""
            )
        ]

    def execute(self, value: str = "") -> ToolResult:
        self.calls.append(value)
        return ToolResult.success(message=f"recorded {value}")


class FailingTool(Tool):
    """Fake SAFE_ACTION tool: records every call, always fails."""

    name = "failing_tool"
    description = "Always fails."
    permission = PermissionLevel.SAFE_ACTION

    def __init__(self):
        self.calls = 0

    def execute(self) -> ToolResult:
        self.calls += 1
        return ToolResult.error("nope")


class ConfirmTool(Tool):
    """Fake CONFIRM tool: proves the engine never auto-approves it."""

    name = "confirm_tool"
    description = "Requires confirmation."
    permission = PermissionLevel.CONFIRM

    def __init__(self):
        self.calls = 0

    def execute(self) -> ToolResult:
        self.calls += 1
        return ToolResult.success(message="ran")


class BrokenTool(Tool):
    """Fake tool whose execute() raises - proves ToolRegistry's own
    isolation (not the engine's) already turns this into a ToolResult
    error, same as test_tools.py::test_execute_isolates_exceptions."""

    name = "broken_tool"
    description = "Always raises."
    permission = PermissionLevel.SAFE_ACTION

    def execute(self) -> ToolResult:
        raise RuntimeError("boom")


def make_engine(poll_interval_seconds: float = 30.0):
    registry = ToolRegistry()
    recording = RecordingTool()
    failing = FailingTool()
    confirm = ConfirmTool()
    registry.register(recording)
    registry.register(failing)
    registry.register(confirm)
    registry.register(BrokenTool())

    store = SQLiteAutomationStore(":memory:")
    reminder_store = SQLiteReminderStore(":memory:")
    memory = SQLiteMemoryStore(":memory:")
    engine = AutomationEngine(
        store=store,
        tools=registry,
        reminder_store=reminder_store,
        memory=memory,
        poll_interval_seconds=poll_interval_seconds,
    )
    return engine, store, registry, reminder_store, memory, recording, failing, confirm


def _action(tool_name: str = "recording_tool", value: str = "hi") -> ToolCall:
    return ToolCall(tool_name=tool_name, parameters={"value": value} if value else {})


# --- tick(): TIME trigger ------------------------------------------------


def test_tick_runs_a_due_time_automation():
    engine, store, *_ , recording, _, _ = make_engine()
    automation = store.create(
        name="due now",
        trigger=Trigger(type=TriggerType.TIME),
        conditions=[],
        action=_action(),
        next_run_at=_iso(timedelta(minutes=-1)),
    )

    processed = engine.tick()

    assert processed == 1
    assert recording.calls == ["hi"]
    executions = store.get_executions(automation.id)
    assert executions[-1].outcome is ExecutionOutcome.SUCCESS


def test_tick_ignores_a_time_automation_not_yet_due():
    engine, store, *_, recording, _, _ = make_engine()
    store.create(
        name="later",
        trigger=Trigger(type=TriggerType.TIME),
        conditions=[],
        action=_action(),
        next_run_at=_iso(timedelta(minutes=30)),
    )

    assert engine.tick() == 0
    assert recording.calls == []


def test_one_time_time_automation_self_disables_after_running():
    engine, store, *_, recording, _, _ = make_engine()
    automation = store.create(
        name="once",
        trigger=Trigger(type=TriggerType.TIME),
        conditions=[],
        action=_action(),
        next_run_at=_iso(timedelta(minutes=-1)),
    )

    engine.tick()
    engine.tick()

    assert recording.calls == ["hi"]
    assert store.get(automation.id).status is AutomationStatus.DISABLED


def test_recurring_time_automation_advances_and_stays_enabled():
    engine, store, *_, recording, _, _ = make_engine()
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

    engine.tick()

    updated = store.get(automation.id)
    assert updated.status is AutomationStatus.ENABLED
    assert datetime.fromisoformat(updated.next_run_at) > datetime.now(timezone.utc)
    # Not due again immediately.
    assert engine.tick() == 0
    assert recording.calls == ["hi"]


# --- tick(): MANUAL trigger is never auto-due -----------------------------


def test_tick_never_runs_a_manual_automation():
    engine, store, *_, recording, _, _ = make_engine()
    store.create(
        name="manual only",
        trigger=Trigger(type=TriggerType.MANUAL),
        conditions=[],
        action=_action(),
    )

    assert engine.tick() == 0
    assert recording.calls == []


# --- tick(): REMINDER_COMPLETED trigger -----------------------------------


def test_reminder_completed_trigger_not_due_while_reminder_pending():
    engine, store, _, reminder_store, *_, recording, _, _ = make_engine()
    reminder = reminder_store.create(text="task", due_at=_iso(timedelta(minutes=5)))
    store.create(
        name="on complete",
        trigger=Trigger(type=TriggerType.REMINDER_COMPLETED, reminder_id=reminder.id),
        conditions=[],
        action=_action(),
    )

    assert engine.tick() == 0
    assert recording.calls == []


def test_reminder_completed_trigger_due_when_reminder_triggered():
    engine, store, _, reminder_store, *_, recording, _, _ = make_engine()
    reminder = reminder_store.create(text="task", due_at=_iso(timedelta(minutes=-1)))
    reminder_store.record_trigger(reminder.id, triggered_at=_iso(timedelta()), success=True)
    automation = store.create(
        name="on complete",
        trigger=Trigger(type=TriggerType.REMINDER_COMPLETED, reminder_id=reminder.id),
        conditions=[],
        action=_action(),
    )

    processed = engine.tick()

    assert processed == 1
    assert recording.calls == ["hi"]
    assert store.get(automation.id).status is AutomationStatus.DISABLED


def test_reminder_completed_trigger_due_when_reminder_explicitly_completed():
    engine, store, _, reminder_store, *_, recording, _, _ = make_engine()
    reminder = reminder_store.create(text="task", due_at=_iso(timedelta(minutes=5)))
    reminder_store.complete(reminder.id)
    store.create(
        name="on complete",
        trigger=Trigger(type=TriggerType.REMINDER_COMPLETED, reminder_id=reminder.id),
        conditions=[],
        action=_action(),
    )

    assert engine.tick() == 1
    assert recording.calls == ["hi"]


def test_reminder_completed_trigger_not_due_when_reminder_cancelled():
    """Cancelling a reminder means "never mind," not "done" - must not
    trigger the automation."""
    engine, store, _, reminder_store, *_, recording, _, _ = make_engine()
    reminder = reminder_store.create(text="task", due_at=_iso(timedelta(minutes=5)))
    reminder_store.cancel(reminder.id)
    store.create(
        name="on complete",
        trigger=Trigger(type=TriggerType.REMINDER_COMPLETED, reminder_id=reminder.id),
        conditions=[],
        action=_action(),
    )

    assert engine.tick() == 0
    assert recording.calls == []


def test_reminder_completed_trigger_fires_only_once():
    engine, store, _, reminder_store, *_, recording, _, _ = make_engine()
    reminder = reminder_store.create(text="task", due_at=_iso(timedelta(minutes=5)))
    reminder_store.complete(reminder.id)
    store.create(
        name="on complete",
        trigger=Trigger(type=TriggerType.REMINDER_COMPLETED, reminder_id=reminder.id),
        conditions=[],
        action=_action(),
    )

    engine.tick()
    engine.tick()

    assert recording.calls == ["hi"]


def test_collect_due_isolates_a_broken_reminder_lookup_from_other_automations():
    """A single automation's due-check failing (e.g. reminder_store.get()
    raising for a corrupt/unexpected trigger) must not stop *other*
    automations from being collected and run in the same poll - the same
    per-item isolation tick() already applies to executing a due
    automation, extended to checking one (see
    AutomationEngine._collect_due)."""

    class BrokenReminderStore(SQLiteReminderStore):
        def get(self, reminder_id):
            if reminder_id == "boom":
                raise RuntimeError("unexpected reminder store failure")
            return super().get(reminder_id)

    registry = ToolRegistry()
    recording = RecordingTool()
    registry.register(recording)
    store = SQLiteAutomationStore(":memory:")
    reminder_store = BrokenReminderStore(":memory:")
    memory = SQLiteMemoryStore(":memory:")
    engine = AutomationEngine(
        store=store, tools=registry, reminder_store=reminder_store, memory=memory
    )

    bad = store.create(
        name="bad",
        trigger=Trigger(type=TriggerType.REMINDER_COMPLETED, reminder_id="boom"),
        conditions=[],
        action=_action(),
    )
    good = store.create(
        name="good",
        trigger=Trigger(type=TriggerType.TIME),
        conditions=[],
        action=_action(value="ok"),
        next_run_at=_iso(timedelta(minutes=-1)),
    )

    processed = engine.tick()  # must not raise

    assert processed == 1
    assert recording.calls == ["ok"]
    assert store.get(good.id).status is AutomationStatus.DISABLED
    assert store.get(bad.id).status is AutomationStatus.ENABLED  # never even ran


# --- conditions gate the action, never the trigger's own scheduling ------


def test_condition_false_skips_the_action_but_still_advances_a_recurring_trigger():
    engine, store, *_, recording, _, _ = make_engine()
    automation = store.create(
        name="gated",
        trigger=Trigger(
            type=TriggerType.TIME,
            recurrence=Recurrence(unit=RecurrenceUnit.DAYS, interval=1),
        ),
        conditions=[FactEqualsCondition(key="mode", value="work")],
        action=_action(),
        next_run_at=_iso(timedelta(minutes=-1)),
    )

    processed = engine.tick()

    assert processed == 1
    assert recording.calls == []  # action never ran
    updated = store.get(automation.id)
    assert updated.status is AutomationStatus.ENABLED
    assert datetime.fromisoformat(updated.next_run_at) > datetime.now(timezone.utc)
    executions = store.get_executions(automation.id)
    assert executions[-1].outcome is ExecutionOutcome.SKIPPED


def test_condition_false_still_retires_a_one_time_trigger():
    engine, store, *_, recording, _, _ = make_engine()
    automation = store.create(
        name="gated once",
        trigger=Trigger(type=TriggerType.TIME),
        conditions=[FactEqualsCondition(key="mode", value="work")],
        action=_action(),
        next_run_at=_iso(timedelta(minutes=-1)),
    )

    engine.tick()

    assert recording.calls == []
    assert store.get(automation.id).status is AutomationStatus.DISABLED


def test_condition_true_runs_the_action():
    engine, store, _, _, memory, recording, _, _ = make_engine()
    memory.save_fact("mode", "work")
    store.create(
        name="gated",
        trigger=Trigger(type=TriggerType.TIME),
        conditions=[FactEqualsCondition(key="mode", value="work")],
        action=_action(),
        next_run_at=_iso(timedelta(minutes=-1)),
    )

    engine.tick()

    assert recording.calls == ["hi"]


# --- action failure ---------------------------------------------------


def test_failed_action_is_recorded_as_failure_outcome():
    engine, store, registry, reminder_store, memory, recording, failing, confirm = (
        make_engine()
    )
    automation = store.create(
        name="fails",
        trigger=Trigger(type=TriggerType.TIME),
        conditions=[],
        action=_action(tool_name="failing_tool", value=""),
        next_run_at=_iso(timedelta(minutes=-1)),
    )

    engine.tick()

    assert failing.calls == 1
    executions = store.get_executions(automation.id)
    assert executions[-1].outcome is ExecutionOutcome.FAILURE
    # A recurring/one-time trigger still advances/retires on failure,
    # same as a skipped condition - see AutomationEngine._advance_trigger.
    assert store.get(automation.id).status is AutomationStatus.DISABLED


# --- permission enforcement (defensive, even bypassing CreateAutomationTool) --


def test_engine_never_runs_a_confirm_tool_even_for_a_hand_crafted_automation():
    """CreateAutomationTool rejects a CONFIRM/DANGEROUS action tool at
    creation time (see test_automation_tools.py) - this proves the engine
    itself is ALSO defensive: even an automation that bypassed that
    validation (constructed directly against the store, as a bug or a
    future caller might) can never actually run a CONFIRM tool, because
    the engine never passes confirmed=True to ToolRegistry.execute."""
    engine, store, registry, reminder_store, memory, recording, failing, confirm = (
        make_engine()
    )
    automation = store.create(
        name="sneaky",
        trigger=Trigger(type=TriggerType.TIME),
        conditions=[],
        action=ToolCall(tool_name="confirm_tool", parameters={}),
        next_run_at=_iso(timedelta(minutes=-1)),
    )

    engine.tick()

    assert confirm.calls == 0  # never actually executed
    executions = store.get_executions(automation.id)
    assert executions[-1].outcome is ExecutionOutcome.FAILURE


# --- failure isolation (regression: must not kill the engine loop) --------


def test_tick_isolates_an_unexpected_exception_from_one_automation():
    engine, store, registry, reminder_store, memory, recording, failing, confirm = (
        make_engine()
    )
    bad = store.create(
        name="bad",
        trigger=Trigger(type=TriggerType.TIME, recurrence=Recurrence(unit=RecurrenceUnit.WEEKS, interval=10**9)),
        conditions=[],
        action=_action(),
        next_run_at=_iso(timedelta(minutes=-1)),
    )
    good = store.create(
        name="good",
        trigger=Trigger(type=TriggerType.TIME),
        conditions=[],
        action=_action(value="ok"),
        next_run_at=_iso(timedelta(minutes=-1)),
    )

    processed = engine.tick()  # must not raise

    assert processed == 1
    assert "ok" in recording.calls
    assert store.get(good.id).status is AutomationStatus.DISABLED


def test_background_thread_survives_a_failed_poll_and_keeps_running():
    import sqlite3

    class FlakyListStore(SQLiteAutomationStore):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.list_calls = 0

        def list(self, status=None):
            self.list_calls += 1
            if self.list_calls == 1:
                raise sqlite3.OperationalError("database is locked")
            return super().list(status=status)

    store = FlakyListStore(":memory:")
    registry = ToolRegistry()
    recording = RecordingTool()
    registry.register(recording)
    reminder_store = SQLiteReminderStore(":memory:")
    memory = SQLiteMemoryStore(":memory:")

    store.create(
        name="after recovery",
        trigger=Trigger(type=TriggerType.TIME),
        conditions=[],
        action=_action(value="recovered"),
        next_run_at=_iso(timedelta(minutes=-1)),
    )

    engine = AutomationEngine(
        store=store,
        tools=registry,
        reminder_store=reminder_store,
        memory=memory,
        poll_interval_seconds=0.05,
    )

    engine.start()
    try:
        deadline = time.monotonic() + 2.0
        while not recording.calls and time.monotonic() < deadline:
            time.sleep(0.01)
    finally:
        engine.stop()

    assert store.list_calls >= 2
    assert recording.calls == ["recovered"]


# --- restart recovery --------------------------------------------------


def test_engine_recovers_a_due_automation_after_restart(tmp_path):
    db_path = str(tmp_path / "automations.db")
    startup_store = SQLiteAutomationStore(db_path)
    automation = startup_store.create(
        name="missed while offline",
        trigger=Trigger(type=TriggerType.TIME),
        conditions=[],
        action=_action(value="recovered"),
        next_run_at=_iso(timedelta(hours=-2)),
    )
    del startup_store

    restarted_store = SQLiteAutomationStore(db_path)
    registry = ToolRegistry()
    recording = RecordingTool()
    registry.register(recording)
    engine = AutomationEngine(
        store=restarted_store,
        tools=registry,
        reminder_store=SQLiteReminderStore(":memory:"),
        memory=SQLiteMemoryStore(":memory:"),
    )

    processed = engine.tick()

    assert processed == 1
    assert recording.calls == ["recovered"]
    assert restarted_store.get(automation.id).status is AutomationStatus.DISABLED


# --- run_now() -----------------------------------------------------------


def test_run_now_runs_a_manual_automation():
    engine, store, *_, recording, _, _ = make_engine()
    automation = store.create(
        name="manual",
        trigger=Trigger(type=TriggerType.MANUAL),
        conditions=[],
        action=_action(value="manual run"),
    )

    engine.run_now(automation.id)

    assert recording.calls == ["manual run"]
    # MANUAL never auto-disables - can be run again.
    assert store.get(automation.id).status is AutomationStatus.ENABLED


def test_run_now_can_force_a_time_automation_that_isnt_due_yet():
    engine, store, *_, recording, _, _ = make_engine()
    automation = store.create(
        name="not due yet",
        trigger=Trigger(type=TriggerType.TIME),
        conditions=[],
        action=_action(value="forced"),
        next_run_at=_iso(timedelta(hours=5)),
    )

    engine.run_now(automation.id)

    assert recording.calls == ["forced"]


def test_run_now_raises_for_unknown_automation():
    engine, *_ = make_engine()

    with pytest.raises(AutomationRunError):
        engine.run_now("does-not-exist")


def test_run_now_raises_for_a_disabled_automation():
    engine, store, *_ = make_engine()
    automation = store.create(
        name="paused", trigger=Trigger(type=TriggerType.MANUAL), conditions=[], action=_action()
    )
    store.disable(automation.id)

    with pytest.raises(AutomationRunError):
        engine.run_now(automation.id)


def test_run_now_raises_for_a_cancelled_automation():
    engine, store, *_ = make_engine()
    automation = store.create(
        name="gone", trigger=Trigger(type=TriggerType.MANUAL), conditions=[], action=_action()
    )
    store.cancel(automation.id)

    with pytest.raises(AutomationRunError):
        engine.run_now(automation.id)


# --- background thread lifecycle ----------------------------------------


def test_start_is_a_no_op_when_already_running():
    engine, *_ = make_engine()

    engine.start()
    first_thread = engine._thread
    engine.start()

    assert engine._thread is first_thread
    engine.stop()


def test_stop_is_a_no_op_when_not_running():
    engine, *_ = make_engine()

    engine.stop()  # must not raise
