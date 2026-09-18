"""
Reminder tools (M7): adapt ReminderStore into the Tool interface, so
create/list/complete/cancel reach a reminder through the same
ToolRegistry (app/tools/registry.py) and TOOL capability
(app/capability.py, app/tools/decision.py) every other tool already uses -
no special-casing in JarvisCore, exactly like web_search (M6.9-6.16).

None of these tools start, stop, or otherwise touch ReminderScheduler -
creating a reminder here just persists a row; the separately-running
scheduler (app/reminders/scheduler.py) is what later notices it's due and
notifies. That split ("storage vs. background process") mirrors
SearchCache/SQLiteSearchCache vs. the search pipeline that reads it.
"""

from datetime import datetime, timedelta, timezone
from typing import Any

from app.reminders.base import (
    RecurrenceError,
    Reminder,
    ReminderStatus,
    ReminderStore,
    parse_recurrence,
)
from app.tools.base import PermissionLevel, Tool, ToolParameter, ToolResult

_VALID_STATUSES = {member.value for member in ReminderStatus}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _reminder_to_dict(reminder: Reminder) -> dict[str, Any]:
    return {
        "id": reminder.id,
        "text": reminder.text,
        "due_at": reminder.due_at,
        "status": reminder.status.value,
        "recurrence_unit": (
            reminder.recurrence.unit.value if reminder.recurrence else None
        ),
        "recurrence_interval": (
            reminder.recurrence.interval if reminder.recurrence else None
        ),
    }


class CreateReminderTool(Tool):
    """Creates a new PENDING reminder due `due_in_minutes` from now,
    optionally repeating every `recurrence_interval` `recurrence_unit`s
    after that. SAFE_ACTION: it only ever adds a new row - it can't affect
    any other reminder, and what it creates can only ever become a
    notification (see ReminderScheduler), never a Tool invocation."""

    name = "create_reminder"
    description = (
        "Create a reminder that notifies the user after a delay, "
        "optionally repeating on a fixed interval."
    )
    permission = PermissionLevel.SAFE_ACTION

    def __init__(self, store: ReminderStore):
        self.store = store

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="text", type="string", description="What to remind the user"
            ),
            ToolParameter(
                name="due_in_minutes",
                type="number",
                description="Minutes from now the reminder should first fire",
            ),
            ToolParameter(
                name="recurrence_unit",
                type="string",
                description=(
                    "Repeat unit: minutes, hours, days, or weeks. Omit for a "
                    "one-time reminder."
                ),
                required=False,
                default=None,
            ),
            ToolParameter(
                name="recurrence_interval",
                type="number",
                description="Repeat every N units (default 1)",
                required=False,
                default=1,
            ),
        ]

    def execute(
        self,
        text: str,
        due_in_minutes: float,
        recurrence_unit: str | None = None,
        recurrence_interval: float = 1,
    ) -> ToolResult:
        if due_in_minutes < 0:
            return ToolResult.error("due_in_minutes must not be negative.")

        try:
            recurrence = parse_recurrence(recurrence_unit, recurrence_interval)
        except RecurrenceError as exc:
            return ToolResult.error(str(exc))

        due_at = (
            datetime.now(timezone.utc) + timedelta(minutes=due_in_minutes)
        ).isoformat()

        reminder = self.store.create(text=text, due_at=due_at, recurrence=recurrence)
        return ToolResult.success(
            data=_reminder_to_dict(reminder),
            message=f"Reminder set: '{text}' at {due_at}.",
        )


class ListRemindersTool(Tool):
    """Lists reminders, optionally filtered to one status. READ: no side
    effects."""

    name = "list_reminders"
    description = (
        "List reminders, optionally filtered by status "
        "(pending, triggered, completed, cancelled)."
    )
    permission = PermissionLevel.READ

    def __init__(self, store: ReminderStore):
        self.store = store

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="status",
                type="string",
                description="Filter: pending, triggered, completed, or cancelled",
                required=False,
                default=None,
            )
        ]

    def execute(self, status: str | None = None) -> ToolResult:
        status_filter = None
        if status is not None:
            if status not in _VALID_STATUSES:
                return ToolResult.error(f"Unknown status: '{status}'")
            status_filter = ReminderStatus(status)

        reminders = self.store.list(status=status_filter)
        return ToolResult.success(data=[_reminder_to_dict(r) for r in reminders])


class CompleteReminderTool(Tool):
    """Marks a reminder COMPLETED, stopping any future recurrence.
    SAFE_ACTION: it only ever changes that one reminder's own status."""

    name = "complete_reminder"
    description = "Mark a reminder as completed, by its ID."
    permission = PermissionLevel.SAFE_ACTION

    def __init__(self, store: ReminderStore):
        self.store = store

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="reminder_id", type="string", description="ID of the reminder"
            )
        ]

    def execute(self, reminder_id: str) -> ToolResult:
        if self.store.complete(reminder_id):
            return ToolResult.success(message=f"Reminder '{reminder_id}' completed.")
        return ToolResult.error(
            f"Reminder '{reminder_id}' not found or already completed/cancelled."
        )


class CancelReminderTool(Tool):
    """Marks a reminder CANCELLED. Same shape as CompleteReminderTool, kept
    as a separate tool/name since "completed" and "cancelled" are
    different user intents even though both stop future occurrences."""

    name = "cancel_reminder"
    description = "Cancel a reminder, by its ID."
    permission = PermissionLevel.SAFE_ACTION

    def __init__(self, store: ReminderStore):
        self.store = store

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="reminder_id", type="string", description="ID of the reminder"
            )
        ]

    def execute(self, reminder_id: str) -> ToolResult:
        if self.store.cancel(reminder_id):
            return ToolResult.success(message=f"Reminder '{reminder_id}' cancelled.")
        return ToolResult.error(
            f"Reminder '{reminder_id}' not found or already completed/cancelled."
        )
