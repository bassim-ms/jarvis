"""
Integration tests for M7: reminder tools wired into the default tool
registry and the running application (app/main.py), plus end-to-end
JarvisCore TOOL-capability dispatch onto a real ReminderStore - proving
reminders integrate through the exact same TOOL capability / Tool Decision
Layer machinery every other tool uses (see test_core.py/test_tools.py),
with no reminder-specific branching added anywhere in Core.
"""

from app.capability import Capability, CapabilityDecision, CapabilityRouter
from app.core import JarvisCore
from app.memory.sqlite_store import SQLiteMemoryStore
from app.providers.base import AIProvider
from app.reminders.base import ReminderStatus
from app.reminders.sqlite_store import SQLiteReminderStore
from app.reminders.tools import (
    CancelReminderTool,
    CompleteReminderTool,
    CreateReminderTool,
    ListRemindersTool,
)
from app.tools.decision import ToolCall, ToolCallBuilder, ToolCallProvider
from app.tools.registry import ToolRegistry


class FakeProvider(AIProvider):
    def generate_reply(self, message: str) -> str:
        return "direct reply"


class FakeCapabilityRouter(CapabilityRouter):
    """Same fixed-decision fake as test_core.py's - avoids depending on
    DeterministicCapabilityRouter's exact phrasing rules here."""

    def __init__(self, capability: Capability, tool_name: str | None = None):
        self.capability = capability
        self.tool_name = tool_name

    def route(self, message: str) -> CapabilityDecision:
        return CapabilityDecision(capability=self.capability, tool_name=self.tool_name)


class FakeToolCallProvider(ToolCallProvider):
    """Returns a fixed ToolCall - stands in for GeminiToolCallProvider so
    these tests never make a real AI call, matching test_core.py's own
    ToolCallProvider fakes."""

    def __init__(self, tool_call: ToolCall):
        self.tool_call = tool_call

    def decide(self, message, candidate_tools):
        return self.tool_call


# --- default tool registration -------------------------------------------


def test_register_default_tools_adds_all_four_reminder_tools():
    from app.tools.defaults import register_default_tools

    registry = ToolRegistry()
    register_default_tools(registry)

    for name in (
        "create_reminder",
        "list_reminders",
        "complete_reminder",
        "cancel_reminder",
    ):
        assert registry.has(name), f"{name} not registered"


def test_register_default_tools_uses_the_given_reminder_store():
    from app.tools.defaults import register_default_tools

    store = SQLiteReminderStore(":memory:")
    registry = ToolRegistry()

    register_default_tools(registry, reminder_store=store)
    registry.execute("create_reminder", params={"text": "x", "due_in_minutes": 5})

    assert len(store.list()) == 1


# --- the actual running application --------------------------------------


def test_application_core_has_reminder_tools_registered():
    from app.main import core as app_core

    for name in (
        "create_reminder",
        "list_reminders",
        "complete_reminder",
        "cancel_reminder",
    ):
        assert app_core.tools.has(name)


def test_application_reminder_scheduler_shares_the_tools_store():
    """The background scheduler and the create/list/complete/cancel tools
    must operate on the exact same persisted reminders, not two
    independent SQLite connections that happen to point at the same
    file."""
    from app.main import core as app_core
    from app.main import reminder_scheduler

    tool_store = app_core.tools.get("create_reminder").store

    assert reminder_scheduler.store is tool_store


# --- end-to-end Core TOOL dispatch onto a real ReminderStore --------------


def test_core_completes_a_reminder_through_deterministic_tool_dispatch():
    """complete_reminder has exactly one required string parameter, so
    ToolCallBuilder fills it deterministically from trailing message text
    - zero AI calls - exactly like EchoTool in test_core.py/test_tool_decision.py."""
    store = SQLiteReminderStore(":memory:")
    reminder = store.create(text="water the plants", due_at="2026-01-01T00:00:00+00:00")

    registry = ToolRegistry()
    registry.register(CompleteReminderTool(store=store))

    jarvis = JarvisCore(
        provider=FakeProvider(),
        memory=SQLiteMemoryStore(":memory:"),
        tools=registry,
        capability_router=FakeCapabilityRouter(
            Capability.TOOL, tool_name="complete_reminder"
        ),
    )

    reply, _, sources = jarvis.handle_chat(f"use tool complete_reminder {reminder.id}")

    assert "completed" in reply.lower()
    assert sources == []
    assert store.get(reminder.id).status is ReminderStatus.COMPLETED


def test_core_cancels_a_reminder_through_deterministic_tool_dispatch():
    store = SQLiteReminderStore(":memory:")
    reminder = store.create(text="water the plants", due_at="2026-01-01T00:00:00+00:00")

    registry = ToolRegistry()
    registry.register(CancelReminderTool(store=store))

    jarvis = JarvisCore(
        provider=FakeProvider(),
        memory=SQLiteMemoryStore(":memory:"),
        tools=registry,
        capability_router=FakeCapabilityRouter(
            Capability.TOOL, tool_name="cancel_reminder"
        ),
    )

    jarvis.handle_chat(f"use tool cancel_reminder {reminder.id}")

    assert store.get(reminder.id).status is ReminderStatus.CANCELLED


def test_core_creates_a_reminder_through_ai_assisted_tool_dispatch():
    """create_reminder has multiple required parameters, so ToolCallBuilder
    falls back to its injected call_provider (faked here, never a real
    Gemini call) to fill them - the same AI-assisted path AddTool exercises
    in test_core.py."""
    store = SQLiteReminderStore(":memory:")
    registry = ToolRegistry()
    registry.register(CreateReminderTool(store=store))

    call_provider = FakeToolCallProvider(
        ToolCall(
            tool_name="create_reminder",
            parameters={"text": "stretch", "due_in_minutes": 10},
        )
    )

    jarvis = JarvisCore(
        provider=FakeProvider(),
        memory=SQLiteMemoryStore(":memory:"),
        tools=registry,
        capability_router=FakeCapabilityRouter(
            Capability.TOOL, tool_name="create_reminder"
        ),
        tool_call_builder=ToolCallBuilder(call_provider),
    )

    jarvis.handle_chat("use tool create_reminder remind me to stretch in 10 minutes")

    reminders = store.list()
    assert len(reminders) == 1
    assert reminders[0].text == "stretch"
    assert reminders[0].status is ReminderStatus.PENDING
