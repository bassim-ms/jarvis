"""
Tests for the reminder Tools (app/reminders/tools.py): CreateReminderTool,
ListRemindersTool, CompleteReminderTool, CancelReminderTool - both directly
and through ToolRegistry, proving permission levels and parameter
validation match every other Tool in this codebase (see test_tools.py).
"""

from datetime import datetime, timedelta, timezone

from app.reminders.base import ReminderStatus
from app.reminders.sqlite_store import SQLiteReminderStore
from app.reminders.tools import (
    CancelReminderTool,
    CompleteReminderTool,
    CreateReminderTool,
    ListRemindersTool,
)
from app.tools.base import PermissionLevel, ToolResultStatus
from app.tools.registry import ToolRegistry


def make_registry_and_store() -> tuple[ToolRegistry, SQLiteReminderStore]:
    store = SQLiteReminderStore(":memory:")
    registry = ToolRegistry()
    registry.register(CreateReminderTool(store=store))
    registry.register(ListRemindersTool(store=store))
    registry.register(CompleteReminderTool(store=store))
    registry.register(CancelReminderTool(store=store))
    return registry, store


# --- permissions -------------------------------------------------------


def test_create_reminder_is_safe_action():
    assert CreateReminderTool(store=SQLiteReminderStore(":memory:")).permission is (
        PermissionLevel.SAFE_ACTION
    )


def test_list_reminders_is_read():
    assert ListRemindersTool(store=SQLiteReminderStore(":memory:")).permission is (
        PermissionLevel.READ
    )


def test_complete_and_cancel_reminder_are_safe_action():
    store = SQLiteReminderStore(":memory:")
    assert CompleteReminderTool(store=store).permission is PermissionLevel.SAFE_ACTION
    assert CancelReminderTool(store=store).permission is PermissionLevel.SAFE_ACTION


# --- create_reminder -----------------------------------------------------


def test_create_reminder_via_registry_creates_a_pending_reminder():
    registry, store = make_registry_and_store()

    result = registry.execute(
        "create_reminder", params={"text": "buy milk", "due_in_minutes": 10}
    )

    assert result.status == ToolResultStatus.SUCCESS
    assert result.data["text"] == "buy milk"
    assert result.data["status"] == "pending"
    reminders = store.list()
    assert len(reminders) == 1
    assert reminders[0].text == "buy milk"


def test_create_reminder_computes_due_at_from_due_in_minutes():
    registry, _ = make_registry_and_store()
    before = datetime.now(timezone.utc)

    result = registry.execute(
        "create_reminder", params={"text": "x", "due_in_minutes": 30}
    )

    due_at = datetime.fromisoformat(result.data["due_at"])
    assert due_at - before > timedelta(minutes=29)
    assert due_at - before < timedelta(minutes=31)


def test_create_reminder_with_recurrence():
    registry, _ = make_registry_and_store()

    result = registry.execute(
        "create_reminder",
        params={
            "text": "stand up",
            "due_in_minutes": 1,
            "recurrence_unit": "hours",
            "recurrence_interval": 2,
        },
    )

    assert result.data["recurrence_unit"] == "hours"
    assert result.data["recurrence_interval"] == 2


def test_create_reminder_rejects_negative_delay():
    registry, _ = make_registry_and_store()

    result = registry.execute(
        "create_reminder", params={"text": "x", "due_in_minutes": -5}
    )

    assert result.status == ToolResultStatus.ERROR


def test_create_reminder_rejects_unknown_recurrence_unit():
    registry, _ = make_registry_and_store()

    result = registry.execute(
        "create_reminder",
        params={"text": "x", "due_in_minutes": 5, "recurrence_unit": "fortnights"},
    )

    assert result.status == ToolResultStatus.ERROR
    assert "fortnights" in result.message


def test_create_reminder_rejects_zero_recurrence_interval():
    """A zero interval would never advance due_at into the future (see
    ReminderScheduler._advance_due_at), so the reminder would re-fire on
    every single poll forever - must be rejected at creation time."""
    registry, store = make_registry_and_store()

    result = registry.execute(
        "create_reminder",
        params={
            "text": "x",
            "due_in_minutes": 5,
            "recurrence_unit": "hours",
            "recurrence_interval": 0,
        },
    )

    assert result.status == ToolResultStatus.ERROR
    assert "recurrence_interval" in result.message
    assert store.list() == []


def test_create_reminder_rejects_negative_recurrence_interval():
    registry, _ = make_registry_and_store()

    result = registry.execute(
        "create_reminder",
        params={
            "text": "x",
            "due_in_minutes": 5,
            "recurrence_unit": "hours",
            "recurrence_interval": -3,
        },
    )

    assert result.status == ToolResultStatus.ERROR
    assert "recurrence_interval" in result.message


def test_create_reminder_rejects_a_fractional_interval_that_truncates_to_zero():
    """int(0.5) == 0 - a sub-unit interval must also be rejected, not
    silently truncated into the same zero-interval spam loop."""
    registry, _ = make_registry_and_store()

    result = registry.execute(
        "create_reminder",
        params={
            "text": "x",
            "due_in_minutes": 5,
            "recurrence_unit": "hours",
            "recurrence_interval": 0.5,
        },
    )

    assert result.status == ToolResultStatus.ERROR


def test_create_reminder_accepts_the_default_recurrence_interval():
    """The default (1) must still work - the new validation shouldn't
    reject ordinary recurring reminders that don't pass an interval at all."""
    registry, _ = make_registry_and_store()

    result = registry.execute(
        "create_reminder",
        params={"text": "x", "due_in_minutes": 5, "recurrence_unit": "hours"},
    )

    assert result.status == ToolResultStatus.SUCCESS
    assert result.data["recurrence_interval"] == 1


# --- list_reminders -------------------------------------------------------


def test_list_reminders_returns_all_by_default():
    registry, store = make_registry_and_store()
    registry.execute("create_reminder", params={"text": "a", "due_in_minutes": 5})
    registry.execute("create_reminder", params={"text": "b", "due_in_minutes": 5})

    result = registry.execute("list_reminders")

    assert result.status == ToolResultStatus.SUCCESS
    assert len(result.data) == 2


def test_list_reminders_filters_by_status():
    registry, store = make_registry_and_store()
    create_result = registry.execute(
        "create_reminder", params={"text": "a", "due_in_minutes": 5}
    )
    registry.execute("create_reminder", params={"text": "b", "due_in_minutes": 5})
    registry.execute(
        "complete_reminder", params={"reminder_id": create_result.data["id"]}
    )

    pending = registry.execute("list_reminders", params={"status": "pending"})
    completed = registry.execute("list_reminders", params={"status": "completed"})

    assert len(pending.data) == 1
    assert pending.data[0]["text"] == "b"
    assert len(completed.data) == 1
    assert completed.data[0]["text"] == "a"


def test_list_reminders_rejects_unknown_status():
    registry, _ = make_registry_and_store()

    result = registry.execute("list_reminders", params={"status": "someday"})

    assert result.status == ToolResultStatus.ERROR


# --- complete_reminder -----------------------------------------------------


def test_complete_reminder_marks_it_completed():
    registry, store = make_registry_and_store()
    created = registry.execute(
        "create_reminder", params={"text": "x", "due_in_minutes": 5}
    )

    result = registry.execute(
        "complete_reminder", params={"reminder_id": created.data["id"]}
    )

    assert result.status == ToolResultStatus.SUCCESS
    assert store.get(created.data["id"]).status is ReminderStatus.COMPLETED


def test_complete_reminder_unknown_id_returns_error():
    registry, _ = make_registry_and_store()

    result = registry.execute(
        "complete_reminder", params={"reminder_id": "does-not-exist"}
    )

    assert result.status == ToolResultStatus.ERROR


# --- cancel_reminder -----------------------------------------------------


def test_cancel_reminder_marks_it_cancelled():
    registry, store = make_registry_and_store()
    created = registry.execute(
        "create_reminder", params={"text": "x", "due_in_minutes": 5}
    )

    result = registry.execute(
        "cancel_reminder", params={"reminder_id": created.data["id"]}
    )

    assert result.status == ToolResultStatus.SUCCESS
    assert store.get(created.data["id"]).status is ReminderStatus.CANCELLED


def test_cancel_reminder_unknown_id_returns_error():
    registry, _ = make_registry_and_store()

    result = registry.execute(
        "cancel_reminder", params={"reminder_id": "does-not-exist"}
    )

    assert result.status == ToolResultStatus.ERROR
