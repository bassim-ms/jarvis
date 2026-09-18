"""
Integration tests for M9's calendar tools: default-registry wiring, the
existing M8 automation-permission guard automatically excluding
create_event/update_event/delete_event from being used as an automation
action (no changes to app/automation/* needed or made), and end-to-end
JarvisCore TOOL-capability dispatch onto a real (faked-provider) calendar
Tool set.
"""

from app.automation.sqlite_store import SQLiteAutomationStore
from app.automation.tools import CreateAutomationTool
from app.calendar.base import CalendarEvent, CalendarProvider
from app.calendar.tools import CreateEventTool, DeleteEventTool, ListEventsTool, UpdateEventTool
from app.capability import Capability, CapabilityDecision, CapabilityRouter
from app.core import JarvisCore
from app.memory.sqlite_store import SQLiteMemoryStore
from app.providers.base import AIProvider
from app.reminders.sqlite_store import SQLiteReminderStore
from app.tools.decision import ToolCall, ToolCallBuilder, ToolCallProvider
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


class FakeToolCallProvider(ToolCallProvider):
    def __init__(self, tool_call: ToolCall):
        self.tool_call = tool_call

    def decide(self, message, candidate_tools):
        return self.tool_call


class FakeCalendarProvider(CalendarProvider):
    def __init__(self):
        self.deleted: list[str] = []

    def list_events(self, time_min=None, time_max=None, max_results=20):
        return []

    def create_event(self, title, start, end, description="", location=""):
        return CalendarEvent(id="e1", title=title, start=start, end=end)

    def update_event(self, event_id, title=None, start=None, end=None, description=None, location=None):
        return CalendarEvent(id=event_id, title=title or "x", start=start or "s", end=end or "e")

    def delete_event(self, event_id):
        self.deleted.append(event_id)


# --- default tool registration -------------------------------------------


def test_register_default_tools_adds_all_four_calendar_tools():
    from app.tools.defaults import register_default_tools

    registry = ToolRegistry()
    register_default_tools(registry)

    for name in ("list_events", "create_event", "update_event", "delete_event"):
        assert registry.has(name), f"{name} not registered"


def test_register_default_tools_uses_the_given_calendar_provider():
    from app.tools.defaults import register_default_tools

    provider = FakeCalendarProvider()
    registry = ToolRegistry()
    register_default_tools(registry, calendar_provider=provider)

    registry.execute("delete_event", params={"event_id": "e1"}, confirmed=True)

    assert provider.deleted == ["e1"]


# --- calendar writes can never be automation actions (M8 guard, unchanged) --


def _make_automation_world(provider: CalendarProvider):
    registry = ToolRegistry()
    registry.register(ListEventsTool(provider=provider))
    registry.register(CreateEventTool(provider=provider))
    registry.register(UpdateEventTool(provider=provider))
    registry.register(DeleteEventTool(provider=provider))

    store = SQLiteAutomationStore(":memory:")
    reminder_store = SQLiteReminderStore(":memory:")
    registry.register(
        CreateAutomationTool(store=store, tools=registry, reminder_store=reminder_store)
    )
    return registry


def test_calendar_write_tools_are_rejected_as_automation_actions():
    registry = _make_automation_world(FakeCalendarProvider())

    for tool_name in ("create_event", "update_event", "delete_event"):
        result = registry.execute(
            "create_automation",
            params={"name": "x", "trigger_type": "manual", "action_tool_name": tool_name},
        )
        assert result.status.value == "error", tool_name


def test_list_events_can_be_used_as_an_automation_action():
    registry = _make_automation_world(FakeCalendarProvider())

    result = registry.execute(
        "create_automation",
        params={"name": "x", "trigger_type": "manual", "action_tool_name": "list_events"},
    )

    assert result.status.value == "success"


# --- end-to-end Core TOOL dispatch -----------------------------------------


def test_core_never_deletes_an_event_without_confirmation():
    """delete_event has exactly one required string parameter, so
    ToolCallBuilder fills it deterministically - zero AI calls - exactly
    like cancel_reminder/cancel_automation. But its DANGEROUS permission
    still requires confirmation regardless of how the call was built -
    since /chat has no confirmation round-trip yet, Core surfaces
    CONFIRMATION_REQUIRED as the reply text and the tool never actually
    runs, the exact same outcome test_core.py already proves for a fake
    CONFIRM tool."""
    provider = FakeCalendarProvider()
    registry = ToolRegistry()
    registry.register(DeleteEventTool(provider=provider))

    jarvis = JarvisCore(
        provider=FakeProvider(),
        memory=SQLiteMemoryStore(":memory:"),
        tools=registry,
        capability_router=FakeCapabilityRouter(Capability.TOOL, tool_name="delete_event"),
    )

    reply, _, sources = jarvis.handle_chat("use tool delete_event evt-123")

    assert "confirm" in reply.lower()
    assert provider.deleted == []
    assert sources == []


def test_core_creates_an_event_through_ai_assisted_tool_dispatch():
    """create_event has multiple required parameters, so ToolCallBuilder
    falls back to its injected call_provider (faked here, never a real
    Gemini call) to fill them - the same AI-assisted path create_reminder/
    create_automation exercise."""
    provider = FakeCalendarProvider()
    registry = ToolRegistry()
    registry.register(CreateEventTool(provider=provider))

    call_provider = FakeToolCallProvider(
        ToolCall(
            tool_name="create_event",
            parameters={
                "title": "Standup",
                "start": "2026-01-01T09:00:00+00:00",
                "end": "2026-01-01T09:30:00+00:00",
            },
        )
    )

    jarvis = JarvisCore(
        provider=FakeProvider(),
        memory=SQLiteMemoryStore(":memory:"),
        tools=registry,
        capability_router=FakeCapabilityRouter(Capability.TOOL, tool_name="create_event"),
        tool_call_builder=ToolCallBuilder(call_provider),
    )

    reply, _, _ = jarvis.handle_chat("use tool create_event standup tomorrow at 9")

    # CONFIRM permission -> still requires confirmation even though the
    # ToolCall itself was built successfully via the AI fallback.
    assert "confirm" in reply.lower()
