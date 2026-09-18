"""
Task tools (M9): adapt TaskStore into the Tool interface, so
create/list/complete/cancel reach a task through the same ToolRegistry
(app/tools/registry.py) and TOOL capability (app/capability.py,
app/tools/decision.py) every other tool already uses - no special-casing
in JarvisCore, exactly like the reminder tools (M7).
"""

from typing import Any

from app.tasks.base import Task, TaskPriority, TaskStatus, TaskStore
from app.tools.base import PermissionLevel, Tool, ToolParameter, ToolResult

_VALID_PRIORITIES = {member.value for member in TaskPriority}
_VALID_STATUSES = {member.value for member in TaskStatus}


def _task_to_dict(task: Task) -> dict[str, Any]:
    return {
        "id": task.id,
        "text": task.text,
        "status": task.status.value,
        "priority": task.priority.value,
        "due_date": task.due_date,
    }


class CreateTaskTool(Tool):
    """Creates a new TODO task. SAFE_ACTION: it only ever adds a new
    row - purely local, no scheduling, no notification.

    Unlike create_reminder/create_automation/create_event, `text` is the
    only required parameter (priority/due_date are optional, with plain
    defaults) - so ToolCallBuilder (app/tools/decision.py) can always
    build the call deterministically from trailing message text, zero AI
    calls, the same single-required-string-param shape as EchoTool."""

    name = "create_task"
    description = "Create a task, optionally with a due date and priority."
    permission = PermissionLevel.SAFE_ACTION

    def __init__(self, store: TaskStore):
        self.store = store

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(name="text", type="string", description="What the task is"),
            ToolParameter(
                name="priority",
                type="string",
                description="low, medium, or high (default medium)",
                required=False,
                default="medium",
            ),
            ToolParameter(
                name="due_date",
                type="string",
                description="Optional due date/time, ISO 8601 - display only",
                required=False,
                default=None,
            ),
        ]

    def execute(
        self, text: str, priority: str = "medium", due_date: str | None = None
    ) -> ToolResult:
        if priority not in _VALID_PRIORITIES:
            return ToolResult.error(
                f"Unknown priority: '{priority}'. Must be one of: "
                f"{sorted(_VALID_PRIORITIES)}"
            )

        task = self.store.create(
            text=text, priority=TaskPriority(priority), due_date=due_date
        )
        return ToolResult.success(
            data=_task_to_dict(task), message=f"Task '{text}' created."
        )


class ListTasksTool(Tool):
    """Lists tasks, optionally filtered to one status. READ: no side
    effects."""

    name = "list_tasks"
    description = "List tasks, optionally filtered by status (todo, done, cancelled)."
    permission = PermissionLevel.READ

    def __init__(self, store: TaskStore):
        self.store = store

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="status",
                type="string",
                description="Filter: todo, done, or cancelled",
                required=False,
                default=None,
            )
        ]

    def execute(self, status: str | None = None) -> ToolResult:
        status_filter = None
        if status is not None:
            if status not in _VALID_STATUSES:
                return ToolResult.error(f"Unknown status: '{status}'")
            status_filter = TaskStatus(status)

        tasks = self.store.list(status=status_filter)
        return ToolResult.success(data=[_task_to_dict(t) for t in tasks])


class CompleteTaskTool(Tool):
    """Marks a task DONE. SAFE_ACTION: it only ever changes that one
    task's own status."""

    name = "complete_task"
    description = "Mark a task as done, by its ID."
    permission = PermissionLevel.SAFE_ACTION

    def __init__(self, store: TaskStore):
        self.store = store

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(name="task_id", type="string", description="ID of the task")
        ]

    def execute(self, task_id: str) -> ToolResult:
        if self.store.complete(task_id):
            return ToolResult.success(message=f"Task '{task_id}' completed.")
        return ToolResult.error(
            f"Task '{task_id}' not found or already completed/cancelled."
        )


class CancelTaskTool(Tool):
    """Marks a task CANCELLED. Same shape as CompleteTaskTool, kept as a
    separate tool/name since "done" and "cancelled" are different user
    intents."""

    name = "cancel_task"
    description = "Cancel a task, by its ID."
    permission = PermissionLevel.SAFE_ACTION

    def __init__(self, store: TaskStore):
        self.store = store

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(name="task_id", type="string", description="ID of the task")
        ]

    def execute(self, task_id: str) -> ToolResult:
        if self.store.cancel(task_id):
            return ToolResult.success(message=f"Task '{task_id}' cancelled.")
        return ToolResult.error(
            f"Task '{task_id}' not found or already completed/cancelled."
        )
