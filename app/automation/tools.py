"""
Automation tools (M8): adapt AutomationStore/AutomationEngine into the
Tool interface, so create/list/enable/disable/cancel/run reach an
automation through the same ToolRegistry (app/tools/registry.py) and
TOOL capability (app/capability.py, app/tools/decision.py) every other
tool already uses - no special-casing in JarvisCore, exactly like
web_search (M6.9-6.16) and the reminder tools (M7).

CreateAutomationTool is the one place that enforces this milestone's
"safe actions only, no unrestricted scheduled tools" requirement: it
rejects an `action_tool_name` that either isn't registered, isn't
already an automatic-permission tool (READ/SAFE_ACTION -
app/tools/base.py's AUTOMATIC_PERMISSIONS), or is one of the automation
tools defined here themselves - the last of which specifically prevents
an automation from creating, enabling, disabling, cancelling, or running
*other* automations, which would otherwise let a recurring automation
self-replicate (an automation whose action creates another automation
with the same recurring trigger, which creates another, ...) - exactly
the kind of unattended, unbounded, planning-loop-like growth this
milestone's scope explicitly excludes, not just a generic permission
check.
"""

from datetime import datetime, timedelta, timezone
from typing import Any

from app.automation.base import (
    Automation,
    AutomationStatus,
    AutomationStore,
    Condition,
    FactEqualsCondition,
    TimeWindowCondition,
    Trigger,
    TriggerType,
)
from app.automation.engine import AutomationEngine, AutomationRunError
from app.reminders.base import RecurrenceError, ReminderStore, parse_recurrence
from app.tools.base import (
    AUTOMATIC_PERMISSIONS,
    PermissionLevel,
    Tool,
    ToolParameter,
    ToolResult,
)
from app.tools.decision import ToolCall
from app.tools.registry import ToolRegistry

_VALID_TRIGGER_TYPES = {member.value for member in TriggerType}
_VALID_STATUSES = {member.value for member in AutomationStatus}

#: Automation tools can never be scheduled as an automation's own action -
#: see this module's docstring. Filled in once, after the tool classes
#: below are defined (their `name` attributes are class-level constants).
_AUTOMATION_TOOL_NAMES: frozenset[str] = frozenset()


def _automation_to_dict(automation: Automation) -> dict[str, Any]:
    return {
        "id": automation.id,
        "name": automation.name,
        "status": automation.status.value,
        "trigger_type": automation.trigger.type.value,
        "recurrence_unit": (
            automation.trigger.recurrence.unit.value
            if automation.trigger.recurrence
            else None
        ),
        "recurrence_interval": (
            automation.trigger.recurrence.interval
            if automation.trigger.recurrence
            else None
        ),
        "reminder_id": automation.trigger.reminder_id,
        "conditions": [c.to_dict() for c in automation.conditions],
        "action_tool_name": automation.action.tool_name,
        "action_params": automation.action.parameters,
        "next_run_at": automation.next_run_at,
    }


class CreateAutomationTool(Tool):
    """Creates a new ENABLED automation: a trigger (time/manual/
    reminder_completed), zero or more basic conditions, and an action
    (an already-registered, already-automatic-permission tool + params).
    SAFE_ACTION: it only ever adds a new automation row - running it is
    entirely AutomationEngine's separate, later job."""

    name = "create_automation"
    description = (
        "Create an automation that runs a safe tool when a trigger fires "
        "(on a schedule, when a reminder completes, or only when run "
        "manually), optionally gated by a basic condition."
    )
    permission = PermissionLevel.SAFE_ACTION

    def __init__(
        self, store: AutomationStore, tools: ToolRegistry, reminder_store: ReminderStore
    ):
        self.store = store
        self.tools = tools
        self.reminder_store = reminder_store

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="name", type="string", description="Human-readable automation name"
            ),
            ToolParameter(
                name="trigger_type",
                type="string",
                description="time, manual, or reminder_completed",
            ),
            ToolParameter(
                name="action_tool_name",
                type="string",
                description="Name of the registered tool this automation runs",
            ),
            ToolParameter(
                name="action_params",
                type="object",
                description="Parameters to pass to action_tool_name",
                required=False,
                default=None,
            ),
            ToolParameter(
                name="due_in_minutes",
                type="number",
                description="For a time trigger: minutes from now it first fires",
                required=False,
                default=None,
            ),
            ToolParameter(
                name="recurrence_unit",
                type="string",
                description=(
                    "For a time trigger: repeat unit (minutes, hours, days, "
                    "weeks). Omit for a one-time trigger."
                ),
                required=False,
                default=None,
            ),
            ToolParameter(
                name="recurrence_interval",
                type="number",
                description="For a time trigger: repeat every N units (default 1)",
                required=False,
                default=1,
            ),
            ToolParameter(
                name="reminder_id",
                type="string",
                description="For a reminder_completed trigger: which reminder",
                required=False,
                default=None,
            ),
            ToolParameter(
                name="condition_fact_key",
                type="string",
                description="Optional condition: only run if this stored fact ...",
                required=False,
                default=None,
            ),
            ToolParameter(
                name="condition_fact_value",
                type="string",
                description="Optional condition: ... equals this value",
                required=False,
                default=None,
            ),
            ToolParameter(
                name="condition_start_hour",
                type="number",
                description="Optional condition: only run at/after this UTC hour (0-23)",
                required=False,
                default=None,
            ),
            ToolParameter(
                name="condition_end_hour",
                type="number",
                description="Optional condition: only run before this UTC hour (0-23)",
                required=False,
                default=None,
            ),
        ]

    def execute(
        self,
        name: str,
        trigger_type: str,
        action_tool_name: str,
        action_params: dict[str, Any] | None = None,
        due_in_minutes: float | None = None,
        recurrence_unit: str | None = None,
        recurrence_interval: float = 1,
        reminder_id: str | None = None,
        condition_fact_key: str | None = None,
        condition_fact_value: str | None = None,
        condition_start_hour: float | None = None,
        condition_end_hour: float | None = None,
    ) -> ToolResult:
        if trigger_type not in _VALID_TRIGGER_TYPES:
            return ToolResult.error(
                f"Unknown trigger_type: '{trigger_type}'. Must be one of: "
                f"{sorted(_VALID_TRIGGER_TYPES)}"
            )

        action_tool = self.tools.get(action_tool_name)
        if action_tool is None:
            return ToolResult.error(f"Unknown tool: '{action_tool_name}'")
        if action_tool_name in _AUTOMATION_TOOL_NAMES:
            return ToolResult.error(
                f"Tool '{action_tool_name}' manages automations itself and "
                "can't be used as an automation's action."
            )
        if action_tool.permission not in AUTOMATIC_PERMISSIONS:
            return ToolResult.error(
                f"Tool '{action_tool_name}' requires confirmation "
                f"({action_tool.permission.value}) and can't be used as an "
                "automation action - only read/safe_action tools are allowed."
            )

        trigger_kind = TriggerType(trigger_type)
        recurrence = None
        reminder_ref = None
        next_run_at = None

        if trigger_kind is TriggerType.TIME:
            if due_in_minutes is None:
                return ToolResult.error("due_in_minutes is required for a time trigger.")
            if due_in_minutes < 0:
                return ToolResult.error("due_in_minutes must not be negative.")
            try:
                recurrence = parse_recurrence(recurrence_unit, recurrence_interval)
            except RecurrenceError as exc:
                return ToolResult.error(str(exc))
            next_run_at = (
                datetime.now(timezone.utc) + timedelta(minutes=due_in_minutes)
            ).isoformat()
        elif trigger_kind is TriggerType.REMINDER_COMPLETED:
            if not reminder_id:
                return ToolResult.error(
                    "reminder_id is required for a reminder_completed trigger."
                )
            if self.reminder_store.get(reminder_id) is None:
                return ToolResult.error(f"Unknown reminder: '{reminder_id}'")
            reminder_ref = reminder_id

        conditions: list[Condition] = []
        if condition_fact_key is not None or condition_fact_value is not None:
            if condition_fact_key is None or condition_fact_value is None:
                return ToolResult.error(
                    "condition_fact_key and condition_fact_value must be given together."
                )
            conditions.append(
                FactEqualsCondition(key=condition_fact_key, value=condition_fact_value)
            )
        if condition_start_hour is not None or condition_end_hour is not None:
            if condition_start_hour is None or condition_end_hour is None:
                return ToolResult.error(
                    "condition_start_hour and condition_end_hour must be given together."
                )
            start_hour, end_hour = int(condition_start_hour), int(condition_end_hour)
            if not (0 <= start_hour <= 23 and 0 <= end_hour <= 23):
                return ToolResult.error(
                    "condition_start_hour/condition_end_hour must be between 0 and 23."
                )
            conditions.append(TimeWindowCondition(start_hour=start_hour, end_hour=end_hour))

        trigger = Trigger(type=trigger_kind, recurrence=recurrence, reminder_id=reminder_ref)
        action = ToolCall(tool_name=action_tool_name, parameters=action_params or {})

        automation = self.store.create(
            name=name,
            trigger=trigger,
            conditions=conditions,
            action=action,
            next_run_at=next_run_at,
        )
        return ToolResult.success(
            data=_automation_to_dict(automation), message=f"Automation '{name}' created."
        )


class ListAutomationsTool(Tool):
    """Lists automations, optionally filtered to one status. READ: no
    side effects."""

    name = "list_automations"
    description = (
        "List automations, optionally filtered by status "
        "(enabled, disabled, cancelled)."
    )
    permission = PermissionLevel.READ

    def __init__(self, store: AutomationStore):
        self.store = store

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="status",
                type="string",
                description="Filter: enabled, disabled, or cancelled",
                required=False,
                default=None,
            )
        ]

    def execute(self, status: str | None = None) -> ToolResult:
        status_filter = None
        if status is not None:
            if status not in _VALID_STATUSES:
                return ToolResult.error(f"Unknown status: '{status}'")
            status_filter = AutomationStatus(status)

        automations = self.store.list(status=status_filter)
        return ToolResult.success(data=[_automation_to_dict(a) for a in automations])


class EnableAutomationTool(Tool):
    """Re-enables a paused automation. SAFE_ACTION: it only ever changes
    that one automation's own status."""

    name = "enable_automation"
    description = "Enable (un-pause) an automation, by its ID."
    permission = PermissionLevel.SAFE_ACTION

    def __init__(self, store: AutomationStore):
        self.store = store

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="automation_id", type="string", description="ID of the automation"
            )
        ]

    def execute(self, automation_id: str) -> ToolResult:
        if self.store.enable(automation_id):
            return ToolResult.success(message=f"Automation '{automation_id}' enabled.")
        return ToolResult.error(
            f"Automation '{automation_id}' not found or cancelled."
        )


class DisableAutomationTool(Tool):
    """Pauses an automation without cancelling it. SAFE_ACTION: it only
    ever changes that one automation's own status."""

    name = "disable_automation"
    description = "Disable (pause) an automation, by its ID."
    permission = PermissionLevel.SAFE_ACTION

    def __init__(self, store: AutomationStore):
        self.store = store

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="automation_id", type="string", description="ID of the automation"
            )
        ]

    def execute(self, automation_id: str) -> ToolResult:
        if self.store.disable(automation_id):
            return ToolResult.success(message=f"Automation '{automation_id}' disabled.")
        return ToolResult.error(
            f"Automation '{automation_id}' not found or cancelled."
        )


class CancelAutomationTool(Tool):
    """Terminally cancels an automation. SAFE_ACTION: it only ever
    changes that one automation's own status."""

    name = "cancel_automation"
    description = "Cancel an automation, by its ID."
    permission = PermissionLevel.SAFE_ACTION

    def __init__(self, store: AutomationStore):
        self.store = store

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="automation_id", type="string", description="ID of the automation"
            )
        ]

    def execute(self, automation_id: str) -> ToolResult:
        if self.store.cancel(automation_id):
            return ToolResult.success(message=f"Automation '{automation_id}' cancelled.")
        return ToolResult.error(
            f"Automation '{automation_id}' not found or already cancelled."
        )


class RunAutomationTool(Tool):
    """Runs one automation immediately, through the exact same
    AutomationEngine execution path (conditions, permission-gated action
    dispatch, execution-history logging, trigger advancement) a
    poller-triggered run uses - the only way a MANUAL-trigger automation
    ever runs, and also usable to force an early run of a time/
    reminder_completed one. SAFE_ACTION: it never bypasses the target
    action's own permission check (still ToolRegistry.execute, still no
    confirmed=True - see app/automation/engine.py)."""

    name = "run_automation"
    description = "Run an automation immediately, regardless of its schedule."
    permission = PermissionLevel.SAFE_ACTION

    def __init__(self, engine: AutomationEngine):
        self.engine = engine

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="automation_id", type="string", description="ID of the automation"
            )
        ]

    def execute(self, automation_id: str) -> ToolResult:
        try:
            self.engine.run_now(automation_id)
        except AutomationRunError as exc:
            return ToolResult.error(str(exc))
        return ToolResult.success(message=f"Automation '{automation_id}' ran.")


_AUTOMATION_TOOL_NAMES = frozenset(
    {
        CreateAutomationTool.name,
        ListAutomationsTool.name,
        EnableAutomationTool.name,
        DisableAutomationTool.name,
        CancelAutomationTool.name,
        RunAutomationTool.name,
    }
)
