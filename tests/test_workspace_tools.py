"""
Automation Slice A1 - Workspace Files, the TOOL and REGISTRY layer: the three tools are ordinary tools on the ordinary ToolRegistry
(list/read are READ, write is SAFE_ACTION like `save_note`), they answer with a file name and a size - never a path - and they are reachable
only where they should be: a Job's plan and an explicit chat command, but not a schedule, an automation or an unattended run.

Real: the tools, ToolRegistry, register_default_tools, JarvisCore's explicit-command path, the automation tool and the unattended profile.
Nothing here writes outside tmp_path.
"""

import pytest

from app.automation.sqlite_store import SQLiteAutomationStore
from app.automation.tools import CreateAutomationTool
from app.config import settings
from app.core import JarvisCore
from app.memory.sqlite_store import SQLiteMemoryStore
from app.orchestration.profiles import UNATTENDED_V1
from app.providers.base import AIProvider
from app.reminders.sqlite_store import SQLiteReminderStore
from app.tools.base import PermissionLevel, ToolArgumentError, ToolResultStatus
from app.tools.decision import ToolCallBuilder, ToolCallError, ToolCallProvider
from app.tools.defaults import create_default_registry, register_default_tools
from app.tools.registry import ToolRegistry
from app.workspace.base import MAX_NAME_CHARS, Workspace
from app.workspace.tools import (
    ListWorkspaceFilesTool,
    ReadWorkspaceFileTool,
    WriteWorkspaceFileTool,
    workspace_tools,
)

NAMES = ("list_workspace_files", "read_workspace_file", "write_workspace_file")


@pytest.fixture
def ws(tmp_path):
    return Workspace(tmp_path / "ws", max_file_bytes=200)


@pytest.fixture
def registry(ws):
    registry = ToolRegistry()
    for tool in workspace_tools(ws):
        registry.register(tool)
    return registry


def run(registry, tool, /, **params):
    return registry.execute(tool, params=params)


# ===================================================== what each tool is =========================================================


def test_the_three_tools_have_the_agreed_names_permissions_and_reach(registry):
    seen = {n: registry.get(n) for n in NAMES}

    assert {n: t.permission for n, t in seen.items()} == {
        "list_workspace_files": PermissionLevel.READ,
        "read_workspace_file": PermissionLevel.READ,
        "write_workspace_file": PermissionLevel.SAFE_ACTION,  # the level save_note has: it only ever CREATES, in one folder
    }
    assert all(t.plannable for t in seen.values())  # a Job's plan may use them
    assert seen["write_workspace_file"].automatable is False  # a schedule or an automation never writes files on its own
    assert seen["list_workspace_files"].automatable and seen["read_workspace_file"].automatable  # read tools: unchanged default


def test_the_declared_parameters_are_exactly_the_ones_the_workspace_needs(registry):
    params = {n: [(p.name, p.type, p.required, p.free_text) for p in registry.get(n).parameters] for n in NAMES}

    assert params == {
        "list_workspace_files": [],
        "read_workspace_file": [("name", "string", True, False)],
        "write_workspace_file": [("name", "string", True, False), ("content", "string", True, True)],  # only the content is prose
    }
    assert not any(p.name in ("path", "directory", "folder", "overwrite", "append", "mode", "encoding") for n in NAMES for p in registry.get(n).parameters)


def test_the_descriptions_tell_a_planner_the_rules_and_never_reveal_a_path(registry, tmp_path):
    for name in NAMES:
        description = registry.get(name).description
        assert str(tmp_path) not in description and "\\" not in description
    write = registry.get("write_workspace_file").description
    assert "NEW" in write and "never overwrites" in write and ".txt, .md, .csv and .json" in write
    assert "never read in part" in registry.get("read_workspace_file").description


# ===================================================== behaviour through the registry ============================================


def test_write_then_list_then_read_through_the_registry(registry):
    wrote = run(registry, "write_workspace_file", name="study-plan.md", content="# Plan\n- chapter 3")
    listed = run(registry, "list_workspace_files")
    read = run(registry, "read_workspace_file", name="study-plan.md")

    assert wrote.status is ToolResultStatus.SUCCESS and wrote.message == "Created 'study-plan.md' (18 bytes)."
    assert wrote.data == {"name": "study-plan.md", "bytes": 18}
    assert listed.data == [{"name": "study-plan.md", "bytes": 18, "modified": listed.data[0]["modified"]}] and listed.message == "1 file(s) in the workspace."
    assert read.data == {"name": "study-plan.md", "bytes": 18, "text": "# Plan\n- chapter 3", "redacted": False}
    assert read.message == "Read 'study-plan.md' (18 bytes)."


def test_a_safe_action_write_runs_without_a_confirmation_round_trip(registry):
    result = run(registry, "write_workspace_file", name="a.txt", content="x")

    assert result.status is ToolResultStatus.SUCCESS  # SAFE_ACTION is automatic, exactly like save_note


def test_results_name_a_file_and_a_size_never_a_path(registry, tmp_path):
    results = [
        run(registry, "write_workspace_file", name="a.md", content="x"),
        run(registry, "list_workspace_files"),
        run(registry, "read_workspace_file", name="a.md"),
        run(registry, "write_workspace_file", name="a.md", content="again"),  # refused
        run(registry, "read_workspace_file", name="nope.md"),  # refused
    ]

    for result in results:
        assert str(tmp_path) not in repr((result.message, result.data)) and "Users" not in repr((result.message, result.data))


def test_create_only_is_enforced_through_the_tool(registry, ws):
    run(registry, "write_workspace_file", name="keep.md", content="original")

    again = run(registry, "write_workspace_file", name="keep.md", content="replacement")

    assert again.status is ToolResultStatus.ERROR and "already exists" in again.message
    assert (ws._root / "keep.md").read_text(encoding="utf-8") == "original"


@pytest.mark.parametrize("name", ["../x.md", "a/b.md", "x.py", "CON.txt", "x.md ", "", "a" * MAX_NAME_CHARS + ".md"])
def test_a_bad_name_is_an_error_result_and_creates_nothing(registry, ws, tmp_path, name):
    result = run(registry, "write_workspace_file", name=name, content="text")

    assert result.status is ToolResultStatus.ERROR
    assert not ws._root.exists() and list(tmp_path.iterdir()) == []


def test_validate_arguments_rejects_bad_values_and_has_no_side_effects(ws, tmp_path):
    write, read = WriteWorkspaceFileTool(ws), ReadWorkspaceFileTool(ws)

    for name, content in (("../x.md", "ok"), ("x.md", ""), ("x.md", "a\x07b"), ("x.md", "z" * 201), ("x.py", "ok")):
        with pytest.raises(ToolArgumentError):
            write.validate_arguments(name=name, content=content)
    with pytest.raises(ToolArgumentError):
        read.validate_arguments(name="../x.md")
    write.validate_arguments(name="ok.md", content="fine")  # a valid pair passes...

    assert list(tmp_path.iterdir()) == []  # ...and none of it touched the disk


def test_the_list_tool_has_nothing_to_validate_and_lists_an_empty_workspace(registry):
    result = run(registry, "list_workspace_files")

    assert result.status is ToolResultStatus.SUCCESS and result.data == [] and result.message == "0 file(s) in the workspace."


# ===================================================== the kill switch and the defaults ==========================================


def test_the_default_registry_has_the_three_tools_and_building_it_creates_no_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "workspace_dir", str(tmp_path / "default-ws"))

    registry = create_default_registry()

    assert all(registry.has(n) for n in NAMES)
    assert not (tmp_path / "default-ws").exists()  # the folder appears with the first write, never at startup


def test_the_settings_switch_is_read_on_every_call_through_the_default_wiring(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "workspace_dir", str(tmp_path / "default-ws"))
    monkeypatch.setattr(settings, "workspace_enabled", False)
    registry = create_default_registry()

    off = run(registry, "write_workspace_file", name="a.md", content="x")
    monkeypatch.setattr(settings, "workspace_enabled", True)  # same tool instances, no rebuild
    on = run(registry, "write_workspace_file", name="a.md", content="x")

    assert off.status is ToolResultStatus.ERROR and "switched off" in off.message
    assert on.status is ToolResultStatus.SUCCESS and (tmp_path / "default-ws" / "a.md").read_text(encoding="utf-8") == "x"


def test_the_settings_bounds_reach_the_default_workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "workspace_dir", str(tmp_path / "default-ws"))
    monkeypatch.setattr(settings, "workspace_max_file_bytes", 10)
    monkeypatch.setattr(settings, "workspace_max_files", 1)
    registry = create_default_registry()

    too_big = run(registry, "write_workspace_file", name="a.md", content="x" * 11)
    first = run(registry, "write_workspace_file", name="a.md", content="x" * 10)
    second = run(registry, "write_workspace_file", name="b.md", content="x")

    assert (too_big.status, first.status, second.status) == (ToolResultStatus.ERROR, ToolResultStatus.SUCCESS, ToolResultStatus.ERROR)


def test_register_default_tools_uses_the_workspace_it_is_given(ws):
    registry = ToolRegistry()
    register_default_tools(registry, workspace=ws)

    run(registry, "write_workspace_file", name="given.md", content="x")

    assert (ws._root / "given.md").exists()


def test_the_workspace_settings_have_the_agreed_defaults():
    assert settings.workspace_enabled is True
    assert settings.workspace_max_file_bytes == 64 * 1024 and settings.workspace_max_files == 200
    assert settings.workspace_max_total_bytes == 20 * 1024 * 1024 and settings.workspace_writes_per_minute == 10
    assert settings.workspace_dir.endswith("Jarvis Workspace") and "Documents" in settings.workspace_dir  # in the user's folder, not the project


# ===================================================== where they must NOT be reachable ==========================================


def test_an_unattended_scheduled_run_may_use_none_of_them(registry):
    for name in NAMES:
        assert name not in UNATTENDED_V1.allowed_tools
        assert UNATTENDED_V1.denial(name, registry.get(name)) is not None  # not even the READ ones: the approved surface is unchanged


def test_an_automation_can_never_be_set_up_to_write_a_file(registry):
    tool = CreateAutomationTool(store=SQLiteAutomationStore(":memory:"), tools=registry, reminder_store=SQLiteReminderStore(":memory:"))

    result = tool.execute(
        name="nightly file", trigger_type="time", action_tool_name="write_workspace_file",
        action_params={"name": "x.md", "content": "x"}, due_in_minutes=5,
    )

    assert result.status is ToolResultStatus.ERROR and "can't be run by an automation" in result.message


# ===================================================== an explicit chat command ==================================================


class _Unused(AIProvider):
    def generate_reply(self, message: str) -> str:
        raise AssertionError("no AI may be used for an explicit tool command")


class _ExplodingCalls(ToolCallProvider):
    def decide(self, message, candidate_tools):
        raise ToolCallError("an explicit command must never be handed to the AI")


def chat(registry):
    jarvis = JarvisCore(provider=_Unused(), memory=SQLiteMemoryStore(":memory:"), tools=registry, tool_call_builder=ToolCallBuilder(_ExplodingCalls()))
    return lambda message: jarvis.handle_chat(message)[0]


def test_an_explicit_command_can_create_list_and_read_a_file(registry, ws):
    say = chat(registry)

    assert say("use tool write_workspace_file notes.md hello from the workspace") == "Created 'notes.md' (24 bytes)."
    assert (ws._root / "notes.md").read_text(encoding="utf-8") == "hello from the workspace"
    assert say("use tool list_workspace_files") == "1 file(s) in the workspace."
    assert say("use tool read_workspace_file notes.md") == "Read 'notes.md' (24 bytes)."


def test_an_explicit_command_cannot_overwrite_or_leave_the_folder(registry, ws, tmp_path):
    say = chat(registry)
    say("use tool write_workspace_file notes.md original")

    again = say("use tool write_workspace_file notes.md replacement")
    outside = say("use tool write_workspace_file ../outside.md pwned")

    assert "already exists" in again and (ws._root / "notes.md").read_text(encoding="utf-8") == "original"
    assert "plain name" in outside and not (tmp_path / "outside.md").exists()
