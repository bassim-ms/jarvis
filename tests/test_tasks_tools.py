"""
Tests for the Task tools (app/tasks/tools.py): CreateTaskTool,
ListTasksTool, CompleteTaskTool, CancelTaskTool - both directly and
through ToolRegistry, proving permission levels and parameter validation
match every other Tool in this codebase (see test_tools.py/
test_reminders_tools.py).
"""

from app.tasks.base import TaskStatus
from app.tasks.sqlite_store import SQLiteTaskStore
from app.tasks.tools import CancelTaskTool, CompleteTaskTool, CreateTaskTool, ListTasksTool
from app.tools.base import PermissionLevel, ToolResultStatus
from app.tools.registry import ToolRegistry


def make_registry_and_store() -> tuple[ToolRegistry, SQLiteTaskStore]:
    store = SQLiteTaskStore(":memory:")
    registry = ToolRegistry()
    registry.register(CreateTaskTool(store=store))
    registry.register(ListTasksTool(store=store))
    registry.register(CompleteTaskTool(store=store))
    registry.register(CancelTaskTool(store=store))
    return registry, store


# --- permissions -------------------------------------------------------


def test_permission_levels():
    registry, _ = make_registry_and_store()

    assert registry.get("create_task").permission is PermissionLevel.SAFE_ACTION
    assert registry.get("list_tasks").permission is PermissionLevel.READ
    assert registry.get("complete_task").permission is PermissionLevel.SAFE_ACTION
    assert registry.get("cancel_task").permission is PermissionLevel.SAFE_ACTION


# --- create_task -----------------------------------------------------


def test_create_task_via_registry():
    registry, store = make_registry_and_store()

    result = registry.execute("create_task", params={"text": "buy milk"})

    assert result.status == ToolResultStatus.SUCCESS
    assert result.data["text"] == "buy milk"
    assert result.data["priority"] == "medium"
    assert result.data["status"] == "todo"
    assert len(store.list()) == 1


def test_create_task_with_priority_and_due_date():
    registry, _ = make_registry_and_store()

    result = registry.execute(
        "create_task", params={"text": "ship it", "priority": "high", "due_date": "2026-01-01"}
    )

    assert result.data["priority"] == "high"
    assert result.data["due_date"] == "2026-01-01"


def test_create_task_rejects_unknown_priority():
    registry, store = make_registry_and_store()

    result = registry.execute("create_task", params={"text": "x", "priority": "urgent"})

    assert result.status == ToolResultStatus.ERROR
    assert "urgent" in result.message
    assert store.list() == []


# --- list_tasks -----------------------------------------------------


def test_list_tasks_returns_all_by_default():
    registry, _ = make_registry_and_store()
    registry.execute("create_task", params={"text": "a"})
    registry.execute("create_task", params={"text": "b"})

    result = registry.execute("list_tasks")

    assert result.status == ToolResultStatus.SUCCESS
    assert len(result.data) == 2


def test_list_tasks_filters_by_status():
    registry, _ = make_registry_and_store()
    a = registry.execute("create_task", params={"text": "a"})
    registry.execute("create_task", params={"text": "b"})
    registry.execute("complete_task", params={"task_id": a.data["id"]})

    todo = registry.execute("list_tasks", params={"status": "todo"})
    done = registry.execute("list_tasks", params={"status": "done"})

    assert len(todo.data) == 1
    assert todo.data[0]["text"] == "b"
    assert len(done.data) == 1
    assert done.data[0]["text"] == "a"


def test_list_tasks_rejects_unknown_status():
    registry, _ = make_registry_and_store()

    result = registry.execute("list_tasks", params={"status": "someday"})

    assert result.status == ToolResultStatus.ERROR


# --- complete_task -----------------------------------------------------


def test_complete_task_marks_it_done():
    registry, store = make_registry_and_store()
    created = registry.execute("create_task", params={"text": "x"})

    result = registry.execute("complete_task", params={"task_id": created.data["id"]})

    assert result.status == ToolResultStatus.SUCCESS
    assert store.get(created.data["id"]).status is TaskStatus.DONE


def test_complete_task_unknown_id_returns_error():
    registry, _ = make_registry_and_store()

    result = registry.execute("complete_task", params={"task_id": "does-not-exist"})

    assert result.status == ToolResultStatus.ERROR


# --- cancel_task -----------------------------------------------------


def test_cancel_task_marks_it_cancelled():
    registry, store = make_registry_and_store()
    created = registry.execute("create_task", params={"text": "x"})

    result = registry.execute("cancel_task", params={"task_id": created.data["id"]})

    assert result.status == ToolResultStatus.SUCCESS
    assert store.get(created.data["id"]).status is TaskStatus.CANCELLED


def test_cancel_task_unknown_id_returns_error():
    registry, _ = make_registry_and_store()

    result = registry.execute("cancel_task", params={"task_id": "does-not-exist"})

    assert result.status == ToolResultStatus.ERROR
