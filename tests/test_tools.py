"""
Tests for the Tool System: the Tool abstraction, the Registry, and the
permission model. Uses only tiny fake tools defined here - proving the
architecture works generically, without any real integration.
"""

import pytest

from app.core import JarvisCore
from app.tools.base import (
    PermissionLevel,
    Tool,
    ToolParameter,
    ToolRequirements,
    ToolResult,
    ToolResultStatus,
)
from app.tools.registry import ToolAlreadyRegisteredError, ToolRegistry


class EchoTool(Tool):
    """Fake READ tool: automatic, no confirmation needed."""

    name = "echo"
    description = "Echoes back whatever text it's given."
    permission = PermissionLevel.READ

    @property
    def parameters(self) -> list[ToolParameter]:
        return [ToolParameter(name="text", type="string", description="Text to echo")]

    def execute(self, text: str = "") -> ToolResult:
        return ToolResult.success(data=text)


class DeleteEverythingTool(Tool):
    """Fake DANGEROUS tool: requires explicit confirmation to run."""

    name = "delete_everything"
    description = "Pretends to delete all data. Never actually does anything."
    permission = PermissionLevel.DANGEROUS
    requirements = ToolRequirements(dependencies=["fake_disk_sdk"], network=False)

    def execute(self) -> ToolResult:
        return ToolResult.success(message="Everything was 'deleted'.")


class BrokenTool(Tool):
    """Fake tool that always raises, to prove the Registry isolates failures."""

    name = "broken"
    description = "Always raises."
    permission = PermissionLevel.READ

    def execute(self) -> ToolResult:
        raise RuntimeError("boom")


# --- Tool abstraction -------------------------------------------------


def test_tool_declares_name_description_permission_and_parameters():
    tool = EchoTool()

    assert tool.name == "echo"
    assert tool.description
    assert tool.permission == PermissionLevel.READ
    assert tool.parameters[0].name == "text"


def test_tool_default_parameters_and_requirements_are_empty():
    class MinimalTool(Tool):
        name = "minimal"
        description = "Does nothing."
        permission = PermissionLevel.READ

        def execute(self) -> ToolResult:
            return ToolResult.success()

    tool = MinimalTool()

    assert tool.parameters == []
    assert tool.requirements.dependencies == []
    assert tool.requirements.network is False


def test_tool_requirements_metadata_is_readable():
    tool = DeleteEverythingTool()

    assert tool.requirements.dependencies == ["fake_disk_sdk"]
    assert tool.requirements.network is False


# --- Registry: registration, lookup, listing ---------------------------


def test_register_and_get_tool():
    registry = ToolRegistry()
    tool = EchoTool()

    registry.register(tool)

    assert registry.get("echo") is tool


def test_get_unknown_tool_returns_none():
    registry = ToolRegistry()

    assert registry.get("does_not_exist") is None


def test_has_reflects_registration_state():
    registry = ToolRegistry()

    assert registry.has("echo") is False

    registry.register(EchoTool())

    assert registry.has("echo") is True


def test_list_tools_returns_all_registered_tools():
    registry = ToolRegistry()
    registry.register(EchoTool())
    registry.register(DeleteEverythingTool())

    names = {tool.name for tool in registry.list_tools()}

    assert names == {"echo", "delete_everything"}


def test_registering_duplicate_name_raises():
    registry = ToolRegistry()
    registry.register(EchoTool())

    with pytest.raises(ToolAlreadyRegisteredError):
        registry.register(EchoTool())


def test_unregister_removes_tool():
    registry = ToolRegistry()
    registry.register(EchoTool())

    registry.unregister("echo")

    assert registry.has("echo") is False


# --- Registry: execution and unknown tools ------------------------------


def test_execute_runs_registered_read_tool_automatically():
    registry = ToolRegistry()
    registry.register(EchoTool())

    result = registry.execute("echo", params={"text": "hello"})

    assert result.status == ToolResultStatus.SUCCESS
    assert result.data == "hello"


def test_execute_unknown_tool_returns_error_result():
    registry = ToolRegistry()

    result = registry.execute("does_not_exist")

    assert result.status == ToolResultStatus.ERROR
    assert "does_not_exist" in result.message


def test_execute_isolates_exceptions_raised_by_a_tool():
    registry = ToolRegistry()
    registry.register(BrokenTool())

    result = registry.execute("broken")

    assert result.status == ToolResultStatus.ERROR
    assert "boom" in result.message


# --- Permissions ---------------------------------------------------------


def test_confirm_level_requires_confirmation_before_running():
    class ArchiveTool(Tool):
        name = "archive"
        description = "Archives something."
        permission = PermissionLevel.CONFIRM

        def execute(self) -> ToolResult:
            return ToolResult.success(message="archived")

    registry = ToolRegistry()
    registry.register(ArchiveTool())

    pending = registry.execute("archive")
    assert pending.status == ToolResultStatus.CONFIRMATION_REQUIRED
    assert pending.permission == PermissionLevel.CONFIRM

    approved = registry.execute("archive", confirmed=True)
    assert approved.status == ToolResultStatus.SUCCESS

    declined = registry.execute("archive", confirmed=False)
    assert declined.status == ToolResultStatus.DENIED


def test_dangerous_level_requires_confirmation_before_running():
    registry = ToolRegistry()
    registry.register(DeleteEverythingTool())

    pending = registry.execute("delete_everything")
    assert pending.status == ToolResultStatus.CONFIRMATION_REQUIRED
    assert pending.permission == PermissionLevel.DANGEROUS

    approved = registry.execute("delete_everything", confirmed=True)
    assert approved.status == ToolResultStatus.SUCCESS


def test_safe_action_level_runs_automatically_like_read():
    class LogTool(Tool):
        name = "log"
        description = "Writes a log line."
        permission = PermissionLevel.SAFE_ACTION

        def execute(self) -> ToolResult:
            return ToolResult.success(message="logged")

    registry = ToolRegistry()
    registry.register(LogTool())

    result = registry.execute("log")

    assert result.status == ToolResultStatus.SUCCESS


# --- Core integration ------------------------------------------------


def test_core_exposes_a_tool_registry_by_default():
    core = JarvisCore(provider=object(), memory=_unused_memory())

    assert isinstance(core.tools, ToolRegistry)
    assert core.tools.list_tools() == []


def test_core_accepts_an_injected_tool_registry():
    registry = ToolRegistry()
    registry.register(EchoTool())

    core = JarvisCore(provider=object(), memory=_unused_memory(), tools=registry)

    assert core.tools is registry
    assert core.tools.has("echo")


def _unused_memory():
    from app.memory.sqlite_store import SQLiteMemoryStore

    return SQLiteMemoryStore(":memory:")


# --- Extensibility: adding a new tool needs no core/registry changes ----


def test_adding_a_brand_new_tool_requires_no_infrastructure_changes():
    """A tool type never seen by base.py or registry.py registers and runs
    exactly like the fakes above - proving the architecture scales to new
    tools without modification."""

    class WeatherStub(Tool):
        name = "weather_stub"
        description = "Stands in for a future real weather tool."
        permission = PermissionLevel.READ
        requirements = ToolRequirements(dependencies=["requests"], network=True)

        @property
        def parameters(self) -> list[ToolParameter]:
            return [ToolParameter(name="city", type="string")]

        def execute(self, city: str) -> ToolResult:
            return ToolResult.success(data=f"weather for {city}")

    registry = ToolRegistry()
    registry.register(WeatherStub())

    result = registry.execute("weather_stub", params={"city": "London"})

    assert result.status == ToolResultStatus.SUCCESS
    assert result.data == "weather for London"
    assert registry.get("weather_stub").requirements.network is True
