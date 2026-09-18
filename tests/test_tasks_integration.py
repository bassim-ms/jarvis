"""
Integration tests for M9's task tools: default-registry wiring,
end-to-end Core TOOL dispatch, and independence from M7 Reminders (no
shared store/table - see app/tasks/base.py's module docstring).
"""

from app.capability import Capability, CapabilityDecision, CapabilityRouter
from app.core import JarvisCore
from app.memory.sqlite_store import SQLiteMemoryStore
from app.providers.base import AIProvider
from app.reminders.sqlite_store import SQLiteReminderStore
from app.tasks.base import TaskStatus
from app.tasks.sqlite_store import SQLiteTaskStore
from app.tasks.tools import CompleteTaskTool, CreateTaskTool
from app.tools.registry import ToolRegistry


class FakeProvider(AIProvider):
    def generate_reply(self, message: str) -> str:
        return "direct reply"


class FakeCapabilityRouter(CapabilityRouter):
    def __init__(self, capability: Capability, tool_name: str | None = None):
        self.capability = capability
        self.tool_name = tool_name

    def route(self, message: str) -> CapabilityDecision:
        return CapabilityDecision(capability=self.capability, tool_name=self.tool_name)


# --- default tool registration -------------------------------------------


def test_register_default_tools_adds_all_four_task_tools():
    from app.tools.defaults import register_default_tools

    registry = ToolRegistry()
    register_default_tools(registry)

    for name in ("create_task", "list_tasks", "complete_task", "cancel_task"):
        assert registry.has(name), f"{name} not registered"


def test_register_default_tools_uses_the_given_task_store():
    from app.tools.defaults import register_default_tools

    store = SQLiteTaskStore(":memory:")
    registry = ToolRegistry()
    register_default_tools(registry, task_store=store)

    registry.execute("create_task", params={"text": "x"})

    assert len(store.list()) == 1


# --- independence from M7 Reminders --------------------------------------


def test_tasks_and_reminders_use_independent_stores():
    """A task and a reminder never share an ID/table - completing one
    must never affect the other."""
    task_store = SQLiteTaskStore(":memory:")
    reminder_store = SQLiteReminderStore(":memory:")

    task = task_store.create(text="a task")
    reminder = reminder_store.create(text="a reminder", due_at="2026-01-01T00:00:00+00:00")

    assert task_store.get(reminder.id) is None
    assert reminder_store.get(task.id) is None
    assert len(task_store.list()) == 1
    assert len(reminder_store.list()) == 1


# --- end-to-end Core TOOL dispatch -----------------------------------------


def test_core_completes_a_task_through_deterministic_tool_dispatch():
    """complete_task has exactly one required string parameter, so
    ToolCallBuilder fills it deterministically from trailing message
    text, zero AI calls - exactly like complete_reminder."""
    store = SQLiteTaskStore(":memory:")
    task = store.create(text="water the plants")

    registry = ToolRegistry()
    registry.register(CompleteTaskTool(store=store))

    jarvis = JarvisCore(
        provider=FakeProvider(),
        memory=SQLiteMemoryStore(":memory:"),
        tools=registry,
        capability_router=FakeCapabilityRouter(Capability.TOOL, tool_name="complete_task"),
    )

    reply, _, sources = jarvis.handle_chat(f"use tool complete_task {task.id}")

    assert "completed" in reply.lower()
    assert sources == []
    assert store.get(task.id).status is TaskStatus.DONE


def test_core_creates_a_task_through_deterministic_tool_dispatch():
    """create_task has exactly one required parameter (text) - unlike
    create_reminder/create_automation/create_event, ToolCallBuilder can
    build it deterministically too, with priority/due_date left at their
    tool-declared defaults (see CreateTaskTool's own docstring)."""
    store = SQLiteTaskStore(":memory:")
    registry = ToolRegistry()
    registry.register(CreateTaskTool(store=store))

    jarvis = JarvisCore(
        provider=FakeProvider(),
        memory=SQLiteMemoryStore(":memory:"),
        tools=registry,
        capability_router=FakeCapabilityRouter(Capability.TOOL, tool_name="create_task"),
    )

    jarvis.handle_chat("use tool create_task remember to stretch")

    tasks = store.list()
    assert len(tasks) == 1
    assert tasks[0].text == "remember to stretch"
    assert tasks[0].priority.value == "medium"
