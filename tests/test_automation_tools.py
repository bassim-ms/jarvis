"""
Tests for the automation Tools (app/automation/tools.py):
CreateAutomationTool, ListAutomationsTool, EnableAutomationTool,
DisableAutomationTool, CancelAutomationTool, RunAutomationTool - both
directly and through ToolRegistry, proving permission levels, safe-action
enforcement (rejecting CONFIRM/DANGEROUS and self-referential automation
tools as an automation's action), and parameter validation.
"""

from app.automation.base import AutomationStatus
from app.automation.engine import AutomationEngine
from app.automation.sqlite_store import SQLiteAutomationStore
from app.automation.tools import (
    CancelAutomationTool,
    CreateAutomationTool,
    DisableAutomationTool,
    EnableAutomationTool,
    ListAutomationsTool,
    RunAutomationTool,
)
from app.memory.sqlite_store import SQLiteMemoryStore
from app.reminders.sqlite_store import SQLiteReminderStore
from app.tools.base import PermissionLevel, Tool, ToolParameter, ToolResult, ToolResultStatus
from app.tools.registry import ToolRegistry


class RecordingTool(Tool):
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


class ConfirmTool(Tool):
    name = "confirm_tool"
    description = "Requires confirmation."
    permission = PermissionLevel.CONFIRM

    def execute(self) -> ToolResult:
        return ToolResult.success(message="ran")


class DangerousTool(Tool):
    name = "dangerous_tool"
    description = "Very dangerous."
    permission = PermissionLevel.DANGEROUS

    def execute(self) -> ToolResult:
        return ToolResult.success(message="ran")


def make_world():
    registry = ToolRegistry()
    recording = RecordingTool()
    registry.register(recording)
    registry.register(ConfirmTool())
    registry.register(DangerousTool())

    automation_store = SQLiteAutomationStore(":memory:")
    reminder_store = SQLiteReminderStore(":memory:")
    memory = SQLiteMemoryStore(":memory:")
    engine = AutomationEngine(
        store=automation_store, tools=registry, reminder_store=reminder_store, memory=memory
    )

    registry.register(
        CreateAutomationTool(
            store=automation_store, tools=registry, reminder_store=reminder_store
        )
    )
    registry.register(ListAutomationsTool(store=automation_store))
    registry.register(EnableAutomationTool(store=automation_store))
    registry.register(DisableAutomationTool(store=automation_store))
    registry.register(CancelAutomationTool(store=automation_store))
    registry.register(RunAutomationTool(engine=engine))

    return registry, automation_store, reminder_store, recording


def _create(registry, **overrides):
    params = {
        "name": "x",
        "trigger_type": "manual",
        "action_tool_name": "recording_tool",
    }
    params.update(overrides)
    return registry.execute("create_automation", params=params)


# --- permissions -------------------------------------------------------


def test_all_automation_tools_use_only_automatic_permissions():
    registry, *_ = make_world()
    for name in (
        "create_automation",
        "list_automations",
        "enable_automation",
        "disable_automation",
        "cancel_automation",
        "run_automation",
    ):
        assert registry.get(name).permission in (
            PermissionLevel.READ,
            PermissionLevel.SAFE_ACTION,
        )
    assert registry.get("list_automations").permission is PermissionLevel.READ
    assert registry.get("create_automation").permission is PermissionLevel.SAFE_ACTION


# --- create_automation: trigger validation -------------------------------


def test_create_manual_automation():
    registry, store, _, _ = make_world()

    result = _create(registry, name="manual one", action_params={"value": "hi"})

    assert result.status == ToolResultStatus.SUCCESS
    assert result.data["trigger_type"] == "manual"
    assert result.data["status"] == "enabled"
    assert len(store.list()) == 1


def test_create_automation_rejects_unknown_trigger_type():
    registry, *_ = make_world()

    result = _create(registry, trigger_type="when_the_stars_align")

    assert result.status == ToolResultStatus.ERROR


def test_create_automation_rejects_unknown_action_tool():
    registry, *_ = make_world()

    result = _create(registry, action_tool_name="does_not_exist")

    assert result.status == ToolResultStatus.ERROR


def test_create_automation_rejects_a_confirm_permission_action_tool():
    registry, *_ = make_world()

    result = _create(registry, action_tool_name="confirm_tool")

    assert result.status == ToolResultStatus.ERROR
    assert "confirm" in result.message.lower()


def test_create_automation_rejects_a_dangerous_permission_action_tool():
    registry, *_ = make_world()

    result = _create(registry, action_tool_name="dangerous_tool")

    assert result.status == ToolResultStatus.ERROR


def test_create_automation_rejects_automation_tools_as_the_action():
    """The whole point of this guard: an automation can never invoke
    create_automation/enable_automation/... as its own action - see
    app/automation/tools.py's module docstring (prevents unbounded
    self-replication)."""
    registry, *_ = make_world()

    for meta_tool_name in (
        "create_automation",
        "list_automations",
        "enable_automation",
        "disable_automation",
        "cancel_automation",
        "run_automation",
    ):
        result = _create(registry, action_tool_name=meta_tool_name)
        assert result.status == ToolResultStatus.ERROR, meta_tool_name


def test_create_time_automation_requires_due_in_minutes():
    registry, *_ = make_world()

    result = _create(registry, trigger_type="time")

    assert result.status == ToolResultStatus.ERROR
    assert "due_in_minutes" in result.message


def test_create_time_automation_rejects_negative_due_in_minutes():
    registry, *_ = make_world()

    result = _create(registry, trigger_type="time", due_in_minutes=-5)

    assert result.status == ToolResultStatus.ERROR


def test_create_time_automation_with_recurrence():
    registry, *_ = make_world()

    result = _create(
        registry,
        name="daily",
        trigger_type="time",
        due_in_minutes=5,
        recurrence_unit="days",
        recurrence_interval=1,
    )

    assert result.status == ToolResultStatus.SUCCESS
    assert result.data["recurrence_unit"] == "days"
    assert result.data["next_run_at"] is not None


def test_create_time_automation_rejects_non_positive_recurrence_interval():
    registry, *_ = make_world()

    result = _create(
        registry,
        trigger_type="time",
        due_in_minutes=5,
        recurrence_unit="days",
        recurrence_interval=0,
    )

    assert result.status == ToolResultStatus.ERROR
    assert "recurrence_interval" in result.message


def test_create_reminder_completed_automation_requires_reminder_id():
    registry, *_ = make_world()

    result = _create(registry, trigger_type="reminder_completed")

    assert result.status == ToolResultStatus.ERROR
    assert "reminder_id" in result.message


def test_create_reminder_completed_automation_rejects_unknown_reminder():
    registry, *_ = make_world()

    result = _create(
        registry, trigger_type="reminder_completed", reminder_id="does-not-exist"
    )

    assert result.status == ToolResultStatus.ERROR


def test_create_reminder_completed_automation_with_real_reminder():
    registry, store, reminder_store, _ = make_world()
    reminder = reminder_store.create(text="task", due_at="2026-01-01T00:00:00+00:00")

    result = _create(
        registry, trigger_type="reminder_completed", reminder_id=reminder.id
    )

    assert result.status == ToolResultStatus.SUCCESS
    assert result.data["reminder_id"] == reminder.id


# --- create_automation: conditions ----------------------------------------


def test_create_automation_with_fact_equals_condition():
    registry, *_ = make_world()

    result = _create(registry, condition_fact_key="mode", condition_fact_value="work")

    assert result.status == ToolResultStatus.SUCCESS
    assert result.data["conditions"] == [
        {"type": "fact_equals", "key": "mode", "value": "work"}
    ]


def test_create_automation_rejects_partial_fact_condition():
    registry, *_ = make_world()

    result = _create(registry, condition_fact_key="mode")

    assert result.status == ToolResultStatus.ERROR


def test_create_automation_with_time_window_condition():
    registry, *_ = make_world()

    result = _create(registry, condition_start_hour=9, condition_end_hour=18)

    assert result.status == ToolResultStatus.SUCCESS
    assert result.data["conditions"] == [
        {"type": "time_window", "start_hour": 9, "end_hour": 18}
    ]


def test_create_automation_rejects_partial_time_window_condition():
    registry, *_ = make_world()

    result = _create(registry, condition_start_hour=9)

    assert result.status == ToolResultStatus.ERROR


def test_create_automation_rejects_out_of_range_time_window_hours():
    registry, *_ = make_world()

    result = _create(registry, condition_start_hour=9, condition_end_hour=25)

    assert result.status == ToolResultStatus.ERROR


def test_create_automation_with_both_condition_types():
    registry, *_ = make_world()

    result = _create(
        registry,
        condition_fact_key="mode",
        condition_fact_value="work",
        condition_start_hour=9,
        condition_end_hour=18,
    )

    assert result.status == ToolResultStatus.SUCCESS
    assert len(result.data["conditions"]) == 2


# --- list_automations -----------------------------------------------------


def test_list_automations_returns_all_by_default():
    registry, *_ = make_world()
    _create(registry, name="a")
    _create(registry, name="b")

    result = registry.execute("list_automations")

    assert result.status == ToolResultStatus.SUCCESS
    assert len(result.data) == 2


def test_list_automations_filters_by_status():
    registry, *_ = make_world()
    a = _create(registry, name="a")
    _create(registry, name="b")
    registry.execute("cancel_automation", params={"automation_id": a.data["id"]})

    enabled = registry.execute("list_automations", params={"status": "enabled"})
    cancelled = registry.execute("list_automations", params={"status": "cancelled"})

    assert len(enabled.data) == 1
    assert enabled.data[0]["name"] == "b"
    assert len(cancelled.data) == 1
    assert cancelled.data[0]["name"] == "a"


def test_list_automations_rejects_unknown_status():
    registry, *_ = make_world()

    result = registry.execute("list_automations", params={"status": "someday"})

    assert result.status == ToolResultStatus.ERROR


# --- enable/disable/cancel -------------------------------------------------


def test_disable_then_enable_automation():
    registry, store, *_ = make_world()
    created = _create(registry)

    disabled = registry.execute(
        "disable_automation", params={"automation_id": created.data["id"]}
    )
    assert disabled.status == ToolResultStatus.SUCCESS
    assert store.get(created.data["id"]).status is AutomationStatus.DISABLED

    enabled = registry.execute(
        "enable_automation", params={"automation_id": created.data["id"]}
    )
    assert enabled.status == ToolResultStatus.SUCCESS
    assert store.get(created.data["id"]).status is AutomationStatus.ENABLED


def test_enable_unknown_automation_returns_error():
    registry, *_ = make_world()

    result = registry.execute("enable_automation", params={"automation_id": "nope"})

    assert result.status == ToolResultStatus.ERROR


def test_cancel_automation():
    registry, store, *_ = make_world()
    created = _create(registry)

    result = registry.execute(
        "cancel_automation", params={"automation_id": created.data["id"]}
    )

    assert result.status == ToolResultStatus.SUCCESS
    assert store.get(created.data["id"]).status is AutomationStatus.CANCELLED


def test_cancel_unknown_automation_returns_error():
    registry, *_ = make_world()

    result = registry.execute("cancel_automation", params={"automation_id": "nope"})

    assert result.status == ToolResultStatus.ERROR


# --- run_automation --------------------------------------------------------


def test_run_automation_runs_a_manual_automation():
    registry, store, _, recording = make_world()
    created = _create(
        registry, name="manual", action_params={"value": "ran manually"}
    )

    result = registry.execute(
        "run_automation", params={"automation_id": created.data["id"]}
    )

    assert result.status == ToolResultStatus.SUCCESS
    assert recording.calls == ["ran manually"]


def test_run_automation_returns_error_for_unknown_id():
    registry, *_ = make_world()

    result = registry.execute("run_automation", params={"automation_id": "nope"})

    assert result.status == ToolResultStatus.ERROR


def test_run_automation_returns_error_for_disabled_automation():
    registry, store, *_ = make_world()
    created = _create(registry)
    registry.execute("disable_automation", params={"automation_id": created.data["id"]})

    result = registry.execute(
        "run_automation", params={"automation_id": created.data["id"]}
    )

    assert result.status == ToolResultStatus.ERROR


def test_run_automation_returns_error_for_cancelled_automation():
    registry, store, *_ = make_world()
    created = _create(registry)
    registry.execute("cancel_automation", params={"automation_id": created.data["id"]})

    result = registry.execute(
        "run_automation", params={"automation_id": created.data["id"]}
    )

    assert result.status == ToolResultStatus.ERROR
