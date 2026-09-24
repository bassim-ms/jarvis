"""
Tests for JarvisCore's Self-Development Engine integration point
(app/core.py): `self_development_engine` is an optional, additive
constructor parameter; SELF_DEVELOPMENT-capability messages reach it
only when configured, falling back to ordinary DIRECT handling
otherwise - the exact same "hint, not a guarantee" fallback discipline
CONFIRMATION's own branch already uses.

Covers focused-test category 18 (permission boundaries) at the Core
level: the self-development engine's own tools are never reachable
through JarvisCore's ordinary `self.tools`, and vice versa.
"""

import json
import subprocess

from app.capability import Capability, CapabilityDecision, CapabilityRouter
from app.core import JarvisCore
from app.memory.sqlite_store import SQLiteMemoryStore
from app.providers.base import AIProvider
from app.self_development.factory import create_default_self_development_engine
from app.tools.base import PermissionLevel, Tool, ToolResult
from app.tools.registry import ToolRegistry


class _RecordingProvider(AIProvider):
    def __init__(self, reply: str = "fake reply"):
        self.reply = reply
        self.prompts: list[str] = []

    def generate_reply(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.reply


class _PlanProvider(AIProvider):
    def __init__(self, plan: dict | None = None):
        self.plan = plan or {
            "objective": "Add a helper function",
            "files_affected": [{"path": "helper.py", "description": "add it"}],
            "dependencies": [],
            "tests_to_run": [],
            "risks": [],
            "required_permissions": ["code_edit"],
        }
        self.calls = 0

    def generate_reply(self, message: str) -> str:
        self.calls += 1
        return json.dumps(self.plan)


class _FixedCapabilityRouter(CapabilityRouter):
    def __init__(self, capability: Capability, goal: str | None = None):
        self.decision = CapabilityDecision(capability=capability, goal=goal)

    def route(self, message: str) -> CapabilityDecision:
        return self.decision


def _init_repo(path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Tester"], cwd=path, check=True)
    (path / "helper.py").write_text("# empty\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=path, check=True)


def _make_engine(repo_path, provider):
    return create_default_self_development_engine(
        provider,
        project_root=str(repo_path),
        max_repair_iterations=1,
        command_timeout_seconds=30.0,
        test_command="git status --porcelain",
    )


def _make_core(engine=None, capability_router=None):
    return JarvisCore(
        provider=_RecordingProvider(),
        memory=SQLiteMemoryStore(":memory:"),
        capability_router=capability_router,
        self_development_engine=engine,
    )


# -- defaults / additive presence --------------------------------------------


def test_self_development_engine_defaults_to_none():
    core = JarvisCore(provider=_RecordingProvider())
    assert core.self_development_engine is None


# -- fallback when no engine is configured -----------------------------


def test_self_development_capability_falls_back_to_direct_without_an_engine():
    router = _FixedCapabilityRouter(Capability.SELF_DEVELOPMENT, goal="fix the voice module")
    core = _make_core(engine=None, capability_router=router)

    reply, conversation_id, sources = core.handle_chat("fix the voice module")

    assert reply == "fake reply"  # ordinary DIRECT reply, not a dead end
    assert sources == []


# -- engine configured: dispatches, proposes, reports -------------------


def test_self_development_capability_dispatches_to_the_engine(tmp_path):
    _init_repo(tmp_path)
    plan_provider = _PlanProvider()
    engine = _make_engine(tmp_path, plan_provider)
    router = _FixedCapabilityRouter(Capability.SELF_DEVELOPMENT, goal="add a helper function")
    core = _make_core(engine=engine, capability_router=router)

    reply, conversation_id, sources = core.handle_chat("add a helper function")

    assert plan_provider.calls == 1
    assert "proposed development plan" in reply.lower()
    assert "helper.py" in reply
    assert sources == []
    # Nothing was actually touched - only a plan was proposed.
    assert (tmp_path / "helper.py").read_text(encoding="utf-8") == "# empty\n"


def test_self_development_uses_decisions_own_goal_text_not_the_raw_message(tmp_path):
    """`decision.goal` (the explicit-invocation phrasing's captured
    text) is what's passed to the engine, not the raw message - the
    same convention TASK's own `goal` already uses."""
    _init_repo(tmp_path)
    plan_provider = _PlanProvider()
    engine = _make_engine(tmp_path, plan_provider)
    router = _FixedCapabilityRouter(Capability.SELF_DEVELOPMENT, goal="add a helper function")
    core = _make_core(engine=engine, capability_router=router)

    core.handle_chat("self-develop: add a helper function")

    session = next(iter(engine.sessions.values()))
    assert session.request == "add a helper function"


def test_reports_a_blocked_dirty_tree_session(tmp_path):
    _init_repo(tmp_path)
    (tmp_path / "helper.py").write_text("dirty\n", encoding="utf-8")
    engine = _make_engine(tmp_path, _PlanProvider())
    router = _FixedCapabilityRouter(Capability.SELF_DEVELOPMENT, goal="add a helper function")
    core = _make_core(engine=engine, capability_router=router)

    reply, _, _ = core.handle_chat("add a helper function")

    assert "uncommitted changes" in reply.lower()


# -- permission boundaries (category 18): the two registries never mix -----


class _SpyTool(Tool):
    name = "core_only_tool"
    description = "A tool only on JarvisCore's own registry."
    permission = PermissionLevel.READ

    def execute(self) -> ToolResult:
        return ToolResult.success(message="ran")


def test_core_tools_registry_and_self_development_registry_are_independent(tmp_path):
    _init_repo(tmp_path)
    core_tools = ToolRegistry()
    core_tools.register(_SpyTool())
    engine = _make_engine(tmp_path, _PlanProvider())
    router = _FixedCapabilityRouter(Capability.SELF_DEVELOPMENT, goal="add a helper function")
    core = JarvisCore(
        provider=_RecordingProvider(),
        memory=SQLiteMemoryStore(":memory:"),
        tools=core_tools,
        capability_router=router,
        self_development_engine=engine,
    )

    core.handle_chat("add a helper function")

    # JarvisCore's own tool is not visible on the engine's registry...
    assert engine.tools.get("core_only_tool") is None
    # ...and the engine's own privileged tools are not visible on
    # JarvisCore's ordinary registry (never reachable through the
    # normal TOOL capability/ToolCallBuilder path).
    assert core.tools.get("self_dev_edit_file") is None
    assert core.tools.get("self_dev_commit") is None


def test_self_development_never_calls_jarviscores_own_tool_registry(tmp_path):
    _init_repo(tmp_path)
    calls: list[str] = []
    core_tools = ToolRegistry()
    original_execute = core_tools.execute

    def _recording_execute(*args, **kwargs):
        calls.append("core_tools_execute_called")
        return original_execute(*args, **kwargs)

    core_tools.execute = _recording_execute  # type: ignore[method-assign]

    engine = _make_engine(tmp_path, _PlanProvider())
    router = _FixedCapabilityRouter(Capability.SELF_DEVELOPMENT, goal="add a helper function")
    core = JarvisCore(
        provider=_RecordingProvider(),
        memory=SQLiteMemoryStore(":memory:"),
        tools=core_tools,
        capability_router=router,
        self_development_engine=engine,
    )

    core.handle_chat("add a helper function")

    assert calls == []


def test_self_development_never_mutates_the_world_model_or_creates_a_job(tmp_path):
    """No World Model / M15 Job dependency exists anywhere in this
    dispatch path at all - proven by simply not injecting either and
    confirming the call still completes without error."""
    _init_repo(tmp_path)
    engine = _make_engine(tmp_path, _PlanProvider())
    router = _FixedCapabilityRouter(Capability.SELF_DEVELOPMENT, goal="add a helper function")
    core = _make_core(engine=engine, capability_router=router)

    reply, conversation_id, sources, job_id = core.handle_chat_with_job_id(
        "add a helper function"
    )
    assert reply
    assert job_id is None  # SELF_DEVELOPMENT never creates an M15 Job
