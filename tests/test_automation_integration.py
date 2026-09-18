"""
Integration tests for M8: automation tools wired into the default tool
registry and the running application (app/main.py), plus end-to-end
JarvisCore TOOL-capability dispatch onto a real AutomationEngine -
proving automations integrate through the exact same TOOL capability /
Tool Decision Layer machinery every other tool uses (see
tests/test_reminders_integration.py), with no automation-specific
branching added anywhere in Core.
"""

from app.automation.base import AutomationStatus
from app.automation.engine import AutomationEngine
from app.automation.sqlite_store import SQLiteAutomationStore
from app.automation.tools import EnableAutomationTool, RunAutomationTool
from app.capability import Capability, CapabilityDecision, CapabilityRouter
from app.core import JarvisCore
from app.memory.sqlite_store import SQLiteMemoryStore
from app.providers.base import AIProvider
from app.reminders.sqlite_store import SQLiteReminderStore
from app.tools.base import PermissionLevel, Tool, ToolParameter, ToolResult
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


# --- default tool registration -------------------------------------------


def test_register_default_tools_adds_all_six_automation_tools():
    from app.tools.defaults import register_default_tools

    registry = ToolRegistry()
    register_default_tools(registry)

    for name in (
        "create_automation",
        "list_automations",
        "enable_automation",
        "disable_automation",
        "cancel_automation",
        "run_automation",
    ):
        assert registry.has(name), f"{name} not registered"


def test_register_default_tools_returns_the_automation_engine():
    from app.tools.defaults import register_default_tools

    registry = ToolRegistry()
    engine = register_default_tools(registry)

    assert isinstance(engine, AutomationEngine)


def test_register_default_tools_uses_the_given_automation_store():
    from app.tools.defaults import register_default_tools

    store = SQLiteAutomationStore(":memory:")
    registry = ToolRegistry()

    register_default_tools(registry, automation_store=store)
    registry.execute(
        "create_automation",
        params={
            "name": "x",
            "trigger_type": "manual",
            "action_tool_name": "web_search",
        },
    )

    assert len(store.list()) == 1


def test_run_automation_tool_dispatches_through_the_shared_registry():
    """RunAutomationTool is constructed with a reference to the same
    registry it's registered into - even though the engine is built
    before every tool is registered, a later-registered tool must still
    be usable as an automation action (see app/tools/defaults.py's
    docstring on this)."""
    from app.tools.defaults import register_default_tools

    registry = ToolRegistry()
    recording = RecordingTool()
    registry.register(recording)
    store = SQLiteAutomationStore(":memory:")

    register_default_tools(registry, automation_store=store)
    created = registry.execute(
        "create_automation",
        params={
            "name": "x",
            "trigger_type": "manual",
            "action_tool_name": "recording_tool",
            "action_params": {"value": "hi"},
        },
    )

    result = registry.execute(
        "run_automation", params={"automation_id": created.data["id"]}
    )

    assert result.status.value == "success"
    assert recording.calls == ["hi"]


# --- the actual running application --------------------------------------


def test_application_core_has_automation_tools_registered():
    from app.main import core as app_core

    for name in (
        "create_automation",
        "list_automations",
        "enable_automation",
        "disable_automation",
        "cancel_automation",
        "run_automation",
    ):
        assert app_core.tools.has(name)


def test_application_automation_engine_shares_the_tools_store():
    from app.main import automation_engine
    from app.main import core as app_core

    tools_store = app_core.tools.get("create_automation").store

    assert automation_engine.store is tools_store


def test_application_automation_engine_uses_core_memory():
    """FactEqualsCondition (app/automation/base.py) must see the same
    facts JarvisCore's own DIRECT/SEARCH prompts use, not a separate,
    empty MemoryStore."""
    from app.main import automation_engine
    from app.main import core as app_core

    assert automation_engine.memory is app_core.memory


# --- end-to-end Core TOOL dispatch onto a real AutomationEngine ----------


def test_core_enables_an_automation_through_deterministic_tool_dispatch():
    """enable_automation has exactly one required string parameter, so
    ToolCallBuilder fills it deterministically from trailing message text
    - zero AI calls - exactly like complete_reminder in
    tests/test_reminders_integration.py."""
    automation_store = SQLiteAutomationStore(":memory:")
    reminder_store = SQLiteReminderStore(":memory:")
    memory = SQLiteMemoryStore(":memory:")
    registry = ToolRegistry()
    registry.register(RecordingTool())
    engine = AutomationEngine(
        store=automation_store, tools=registry, reminder_store=reminder_store, memory=memory
    )
    registry.register(EnableAutomationTool(store=automation_store))

    from app.automation.base import Trigger, TriggerType

    automation = automation_store.create(
        name="x",
        trigger=Trigger(type=TriggerType.MANUAL),
        conditions=[],
        action=ToolCall(tool_name="recording_tool", parameters={}),
    )
    automation_store.disable(automation.id)

    jarvis = JarvisCore(
        provider=FakeProvider(),
        memory=SQLiteMemoryStore(":memory:"),
        tools=registry,
        capability_router=FakeCapabilityRouter(
            Capability.TOOL, tool_name="enable_automation"
        ),
    )

    reply, _, sources = jarvis.handle_chat(f"use tool enable_automation {automation.id}")

    assert "enabled" in reply.lower()
    assert sources == []
    assert automation_store.get(automation.id).status is AutomationStatus.ENABLED


def test_core_runs_an_automation_through_deterministic_tool_dispatch():
    automation_store = SQLiteAutomationStore(":memory:")
    reminder_store = SQLiteReminderStore(":memory:")
    memory = SQLiteMemoryStore(":memory:")
    registry = ToolRegistry()
    recording = RecordingTool()
    registry.register(recording)
    engine = AutomationEngine(
        store=automation_store, tools=registry, reminder_store=reminder_store, memory=memory
    )
    registry.register(RunAutomationTool(engine=engine))

    from app.automation.base import Trigger, TriggerType

    automation = automation_store.create(
        name="x",
        trigger=Trigger(type=TriggerType.MANUAL),
        conditions=[],
        action=ToolCall(tool_name="recording_tool", parameters={"value": "hi"}),
    )

    jarvis = JarvisCore(
        provider=FakeProvider(),
        memory=SQLiteMemoryStore(":memory:"),
        tools=registry,
        capability_router=FakeCapabilityRouter(
            Capability.TOOL, tool_name="run_automation"
        ),
    )

    jarvis.handle_chat(f"use tool run_automation {automation.id}")

    assert recording.calls == ["hi"]


def test_core_creates_an_automation_through_ai_assisted_tool_dispatch():
    """create_automation has multiple required parameters, so
    ToolCallBuilder falls back to its injected call_provider (faked here,
    never a real Gemini call) to fill them - the same AI-assisted path
    create_reminder exercises in test_reminders_integration.py."""
    from app.tools.defaults import register_default_tools

    registry = ToolRegistry()
    automation_store = SQLiteAutomationStore(":memory:")
    register_default_tools(registry, automation_store=automation_store)

    call_provider = FakeToolCallProvider(
        ToolCall(
            tool_name="create_automation",
            parameters={
                "name": "daily search",
                "trigger_type": "time",
                "action_tool_name": "web_search",
                "action_params": {"query": "jarvis project"},
                "due_in_minutes": 10,
            },
        )
    )

    jarvis = JarvisCore(
        provider=FakeProvider(),
        memory=SQLiteMemoryStore(":memory:"),
        tools=registry,
        capability_router=FakeCapabilityRouter(
            Capability.TOOL, tool_name="create_automation"
        ),
        tool_call_builder=ToolCallBuilder(call_provider),
    )

    jarvis.handle_chat("use tool create_automation search daily in 10 minutes")

    automations = automation_store.list()
    assert len(automations) == 1
    assert automations[0].name == "daily search"
    assert automations[0].action.tool_name == "web_search"
