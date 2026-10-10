"""
Runtime-path tests for explicit tool-command understanding.

These go through the REAL chain - JarvisCore (default DeterministicCapability
Router, default planner wiring) -> ExplicitToolCommandPlanner -> ToolSequence
-> ToolRegistry.execute -> Tool.execute - and, for the final class, through
the real FastAPI `/chat` endpoint. Every tool records its own `execute()`
calls into one shared log, so the tests prove what actually RAN, in what
order, with what arguments - not merely what was parsed.

Nothing touches the OS: computer tools run against a recording fake backend.
"""

import pytest
from fastapi.testclient import TestClient

from app.computer.base import (
    AllowedApplication,
    ComputerBackend,
    ComputerControlError,
    MouseButton,
    ScreenInfo,
    WindowInfo,
)
from app.computer.defaults import create_computer_tools_registry
from app.core import JarvisCore
from app.memory.sqlite_store import SQLiteMemoryStore
from app.providers.base import AIProvider, ProviderError
from app.tools.base import PermissionLevel, Tool, ToolParameter, ToolResult
from app.tools.decision import ToolCall, ToolCallBuilder, ToolCallError, ToolCallProvider
from app.tools.registry import ToolRegistry

# --- fakes --------------------------------------------------------------------


class UnusedProvider(AIProvider):
    """The chat provider must never be called for an explicit tool command."""

    def generate_reply(self, message: str) -> str:
        raise AssertionError(f"chat provider must not be called, got: {message!r}")


class ExplodingToolCallProvider(ToolCallProvider):
    def __init__(self):
        self.calls: list[str] = []

    def decide(self, message, candidate_tools):
        self.calls.append(message)
        raise AssertionError(f"AI tool-call construction must not run, got: {message!r}")


class FakeComputerBackend(ComputerBackend):
    """Records every backend call, in order, in one list."""

    def __init__(self):
        self.calls: list[tuple] = []

    def get_active_window(self) -> WindowInfo:
        self.calls.append(("get_active_window",))
        return WindowInfo(title="Notepad", process_name="notepad.exe", is_maximized=True)

    def get_screen_info(self) -> ScreenInfo:
        return ScreenInfo(width=1920, height=1080)

    def open_application(self, app: AllowedApplication) -> None:
        self.calls.append(("open_application", app))

    def open_url(self, url: str) -> None:
        self.calls.append(("open_url", url))

    def type_text(self, text: str) -> None:
        self.calls.append(("type_text", text))

    def press_key(self, key: str) -> None:
        self.calls.append(("press_key", key))

    def move_mouse(self, x: int, y: int) -> None:
        self.calls.append(("move_mouse", x, y))

    def click_mouse(self, x, y, button: MouseButton) -> None:
        self.calls.append(("click_mouse", x, y, button))

    def double_click(self, x: int, y: int, button: MouseButton) -> None:
        self.calls.append(("double_click", x, y, button))

    def scroll(self, amount: int) -> None:
        self.calls.append(("scroll", amount))


class LoggedTool(Tool):
    """Base for tools that append (name, kwargs) to a shared log on execute()."""

    permission = PermissionLevel.READ
    description = "a logging test tool"

    def __init__(self, log: list):
        self.log = log


class AlphaTool(LoggedTool):
    name = "alpha"

    def execute(self) -> ToolResult:
        self.log.append(("alpha", {}))
        return ToolResult.success(message="alpha done")


class BetaTool(LoggedTool):
    name = "beta"

    def execute(self) -> ToolResult:
        self.log.append(("beta", {}))
        return ToolResult.success(message="beta done")


class GammaTool(LoggedTool):
    name = "gamma"

    def execute(self) -> ToolResult:
        self.log.append(("gamma", {}))
        return ToolResult.success(message="gamma done")


class FailingTool(LoggedTool):
    """Executes, records, and returns ToolResult.error."""

    name = "fails"

    def execute(self) -> ToolResult:
        self.log.append(("fails", {}))
        return ToolResult.error("it broke")


class RaisingTool(LoggedTool):
    name = "raises"

    def execute(self) -> ToolResult:
        self.log.append(("raises", {}))
        raise RuntimeError("boom")


class ConfirmingTool(LoggedTool):
    name = "gated"
    permission = PermissionLevel.CONFIRM

    def execute(self) -> ToolResult:
        self.log.append(("gated", {}))
        return ToolResult.success(message="gated ran")


class DangerousTool(LoggedTool):
    name = "dangerous"
    permission = PermissionLevel.DANGEROUS

    def execute(self) -> ToolResult:
        self.log.append(("dangerous", {}))
        return ToolResult.success(message="dangerous ran")


class NoteTool(LoggedTool):
    """Free-text string + a number, to prove arguments arrive intact."""

    name = "note"

    @property
    def parameters(self):
        return [
            ToolParameter(name="text", type="string", free_text=True),
        ]

    def execute(self, text: str) -> ToolResult:
        self.log.append(("note", {"text": text}))
        return ToolResult.success(message=f"noted: {text}")


class DenyAllValueTool(LoggedTool):
    """Its value check always rejects - proves validate_arguments gates the
    WHOLE sequence, before any earlier step executes."""

    name = "picky"

    @property
    def parameters(self):
        return [ToolParameter(name="value", type="string")]

    def validate_arguments(self, value: str) -> None:
        from app.tools.base import ToolArgumentError

        raise ToolArgumentError(f"value {value!r} is never acceptable")

    def execute(self, value: str) -> ToolResult:
        self.log.append(("picky", {"value": value}))
        return ToolResult.success(message="picky ran")


def make_jarvis(tools=None, computer_tools=None, provider=None) -> tuple[JarvisCore, ExplodingToolCallProvider]:
    call_provider = provider or ExplodingToolCallProvider()
    jarvis = JarvisCore(
        provider=UnusedProvider(),
        memory=SQLiteMemoryStore(":memory:"),
        tools=tools if tools is not None else ToolRegistry(),
        computer_tools=computer_tools,
        tool_call_builder=ToolCallBuilder(call_provider),
    )
    return jarvis, call_provider


def logged_registry(log: list, *tool_classes) -> ToolRegistry:
    registry = ToolRegistry()
    for cls in tool_classes or (AlphaTool, BetaTool, GammaTool):
        registry.register(cls(log))
    return registry


# --- the two regression commands, through the real chat pathway ----------------------


def test_regression_open_url_then_scroll_executes_exactly_open_url_then_scroll():
    backend = FakeComputerBackend()
    jarvis, provider = make_jarvis(computer_tools=create_computer_tools_registry(backend=backend))

    reply, _, sources = jarvis.handle_chat(
        "use tool open_url https://luniti.app, then use tool scroll -600"
    )

    assert backend.calls == [("open_url", "https://luniti.app"), ("scroll", -600)]
    assert reply == "1. Opened 'https://luniti.app'.\n2. Scrolled by -600."
    assert sources == []
    assert provider.calls == []  # no AI interpreted "-600" or the URL


def test_regression_open_url_wait_scroll_executes_nothing_and_opens_no_url():
    backend = FakeComputerBackend()
    jarvis, provider = make_jarvis(computer_tools=create_computer_tools_registry(backend=backend))

    reply, _, _ = jarvis.handle_chat(
        "use tool open_url https://luniti.app, wait for 5 seconds, then use tool scroll -600"
    )

    assert backend.calls == []  # not even a malformed URL was opened
    assert "Nothing was executed" in reply
    assert "wait" in reply and "not an available tool" in reply
    assert provider.calls == []


def test_there_is_no_wait_tool_in_the_real_application_registries():
    """Verified against the REAL app.main wiring: the "wait" the user asked
    for genuinely does not exist, in either registry."""
    import app.main as live_app

    for name in ("wait", "sleep", "delay", "pause", "timer"):
        assert live_app.core.tools.has(name) is False
        assert live_app.core.computer_tools.has(name) is False


# --- 25/26: exact order, and execute() invoked for every step ---------------------------


def test_steps_execute_in_exact_order_and_each_executes_exactly_once():
    log: list = []
    jarvis, _ = make_jarvis(tools=logged_registry(log))

    reply, _, _ = jarvis.handle_chat("use tool gamma, then use tool alpha, then use tool beta")

    assert [name for name, _ in log] == ["gamma", "alpha", "beta"]
    assert reply == "1. gamma done\n2. alpha done\n3. beta done"


@pytest.mark.parametrize(
    "message",
    [
        "use tool alpha, then use tool beta, then use tool gamma",
        "use tool alpha, then beta, then gamma",
        "use tool alpha and then beta and then gamma",
        "use tool alpha followed by beta followed by gamma",
        "use tool alpha, then use tool beta and then gamma",
    ],
)
def test_every_separator_form_runs_all_three_steps_in_order(message):
    log: list = []
    jarvis, _ = make_jarvis(tools=logged_registry(log))

    jarvis.handle_chat(message)

    assert [name for name, _ in log] == ["alpha", "beta", "gamma"]


def test_a_single_step_command_replies_with_just_the_result():
    log: list = []
    jarvis, _ = make_jarvis(tools=logged_registry(log))

    reply, _, _ = jarvis.handle_chat("use tool beta")

    assert reply == "beta done"
    assert log == [("beta", {})]


def test_free_text_arguments_reach_execute_intact_and_stop_at_the_separator():
    log: list = []
    registry = logged_registry(log, NoteTool, AlphaTool)
    jarvis, _ = make_jarvis(tools=registry)

    jarvis.handle_chat("use tool note remember: milk, eggs, then use tool alpha")

    assert log == [("note", {"text": "remember: milk, eggs"}), ("alpha", {})]


# --- 15-17, 18-20: nothing executes when any step is invalid -----------------------------


@pytest.mark.parametrize(
    "message",
    [
        "use tool fake_tool, then use tool alpha",  # unknown first
        "use tool alpha, then use tool fake_tool, then use tool beta",  # unknown middle
        "use tool alpha, then use tool beta, then use tool fake_tool",  # unknown last
        "use tool alpha, then use tool definitely_not_a_real_tool",
        "use tool alpha, wait for 5 seconds, then use tool beta",  # unsupported "wait"
        "use tool alpha, then wait for 5 seconds, then use tool beta",
        "use tool alpha, then",  # malformed
        "use tool alpha, then, then use tool beta",
        "use tool alpha, then use tool",
        'use tool note "unterminated, then use tool alpha',
        "use tool alpha please, then use tool beta",  # extra words
        "use tool alpha, then use tool beta please",
    ],
)
def test_any_invalid_step_rejects_the_whole_sequence_and_executes_nothing(message):
    log: list = []
    jarvis, provider = make_jarvis(tools=logged_registry(log, AlphaTool, BetaTool, NoteTool))

    reply, _, _ = jarvis.handle_chat(message)

    assert log == []
    assert "nothing was executed" in reply.lower()
    assert provider.calls == []


def test_an_invalid_argument_value_rejects_the_sequence_before_any_step_runs():
    log: list = []
    registry = logged_registry(log, AlphaTool, DenyAllValueTool)
    jarvis, _ = make_jarvis(tools=registry)

    reply, _, _ = jarvis.handle_chat("use tool alpha, then use tool picky anything")

    assert log == []
    assert "Step 2 (picky)" in reply and "never acceptable" in reply


def test_an_unknown_tool_reply_names_the_tool_and_says_nothing_ran():
    log: list = []
    jarvis, _ = make_jarvis(tools=logged_registry(log))

    reply, _, _ = jarvis.handle_chat("use tool alpha, then use tool fake_tool")

    assert "Step 2 (fake_tool): Unknown tool: 'fake_tool'" in reply
    assert "Nothing was executed" in reply


# --- 23: prose is not a tool command -----------------------------------------------------


def test_my_tool_is_great_does_not_invoke_a_tool_and_reaches_the_normal_chat_path():
    log: list = []
    replies: list[str] = []

    class RecordingProvider(AIProvider):
        def generate_reply(self, message: str) -> str:
            replies.append(message)
            return "glad you like it"

    jarvis = JarvisCore(
        provider=RecordingProvider(),
        memory=SQLiteMemoryStore(":memory:"),
        tools=logged_registry(log),
    )

    reply, _, _ = jarvis.handle_chat("my tool is great")

    assert reply == "glad you like it"
    assert log == []
    assert len(replies) == 1


# --- 27/28: permission enforcement for every step ----------------------------------------


@pytest.mark.parametrize("gated", [ConfirmingTool, DangerousTool])
@pytest.mark.parametrize("position", [0, 1, 2])
def test_one_unauthorized_step_anywhere_rejects_the_entire_sequence(gated, position):
    log: list = []
    registry = logged_registry(log, AlphaTool, BetaTool, gated)
    names = ["alpha", "beta"]
    names.insert(position, gated.name)
    jarvis, _ = make_jarvis(tools=registry)

    reply, _, _ = jarvis.handle_chat(", then ".join(f"use tool {n}" for n in names))

    assert log == []  # no step - authorized or not - executed
    assert f"Step {position + 1} ({gated.name})" in reply
    assert "requires confirmation" in reply
    assert "Nothing was executed" in reply


def test_every_authorized_step_still_goes_through_toolregistry_execute():
    """The planner pre-checks permission, but execution itself is still
    ToolRegistry.execute per step (defence in depth)."""
    log: list = []
    executed: list[str] = []

    class SpyRegistry(ToolRegistry):
        def execute(self, name, params=None, confirmed=None):
            executed.append(name)
            return super().execute(name, params=params, confirmed=confirmed)

    registry = SpyRegistry()
    for cls in (AlphaTool, BetaTool, GammaTool):
        registry.register(cls(log))
    jarvis, _ = make_jarvis(tools=registry)

    jarvis.handle_chat("use tool alpha, then use tool beta, then use tool gamma")

    assert executed == ["alpha", "beta", "gamma"]


def test_a_single_unauthorized_tool_is_still_not_executed():
    log: list = []
    jarvis, _ = make_jarvis(tools=logged_registry(log, ConfirmingTool))

    reply, _, _ = jarvis.handle_chat("use tool gated")

    assert log == []
    assert "requires confirmation" in reply


# --- 29: execution failure policy (phase 2) ----------------------------------------------


def test_a_step_that_fails_at_execution_stops_the_sequence_and_keeps_earlier_results():
    log: list = []
    registry = logged_registry(log, AlphaTool, FailingTool, GammaTool)
    jarvis, _ = make_jarvis(tools=registry)

    reply, _, _ = jarvis.handle_chat("use tool alpha, then use tool fails, then use tool gamma")

    assert [name for name, _ in log] == ["alpha", "fails"]  # gamma never ran
    lines = reply.splitlines()
    assert lines[0] == "1. alpha done"  # earlier result preserved
    assert lines[1] == "2. it broke"  # the real failure, reported as-is
    assert "Stopped after step 2; step(s) 3 were not run." in reply
    assert "gamma done" not in reply  # never claims a step that didn't run


def test_a_tool_that_raises_is_reported_as_an_error_and_stops_the_sequence():
    log: list = []
    registry = logged_registry(log, RaisingTool, AlphaTool)
    jarvis, _ = make_jarvis(tools=registry)

    reply, _, _ = jarvis.handle_chat("use tool raises, then use tool alpha")

    assert [name for name, _ in log] == ["raises"]
    assert "boom" in reply and "Stopped after step 1" in reply


def test_a_failure_in_the_last_step_adds_no_stopped_note():
    log: list = []
    registry = logged_registry(log, AlphaTool, FailingTool)
    jarvis, _ = make_jarvis(tools=registry)

    reply, _, _ = jarvis.handle_chat("use tool alpha, then use tool fails")

    assert reply == "1. alpha done\n2. it broke"


def test_a_computer_tool_that_fails_at_execution_stops_the_rest():
    """The backend-level failure (validation passed, the OS call failed)
    is an EXECUTION failure: earlier steps stay reported, later ones don't run."""

    class FailingScrollBackend(FakeComputerBackend):
        def scroll(self, amount: int) -> None:
            raise ComputerControlError("scroll refused")

    backend = FailingScrollBackend()
    jarvis, _ = make_jarvis(computer_tools=create_computer_tools_registry(backend=backend))

    reply, _, _ = jarvis.handle_chat(
        "use tool open_url https://luniti.app, then use tool scroll -600, then use tool get_active_window"
    )

    assert backend.calls == [("open_url", "https://luniti.app")]
    assert reply.splitlines()[0] == "1. Opened 'https://luniti.app'."
    assert "scroll refused" in reply
    assert "Stopped after step 2; step(s) 3 were not run." in reply


# --- the executor runs the ToolSequence, not the sentence ---------------------------------


def test_execution_never_reparses_the_original_message(monkeypatch):
    """Once validated, the ToolSequence is the source of truth: after the
    planner has run, re-parsing (of any kind) would be a bug."""
    import app.tools.command as command

    log: list = []
    jarvis, _ = make_jarvis(tools=logged_registry(log))
    real_parse = command.parse_explicit_command
    parse_calls: list[str] = []

    def counting_parse(message, lookup):
        parse_calls.append(message)
        return real_parse(message, lookup)

    monkeypatch.setattr(command, "parse_explicit_command", counting_parse)

    jarvis.handle_chat("use tool alpha, then use tool beta")

    assert parse_calls == ["use tool alpha, then use tool beta"]  # exactly once
    assert [name for name, _ in log] == ["alpha", "beta"]


def test_the_executor_cannot_invent_or_repeat_steps():
    log: list = []
    jarvis, _ = make_jarvis(tools=logged_registry(log))

    jarvis.handle_chat("use tool alpha, then use tool alpha")
    assert [name for name, _ in log] == ["alpha", "alpha"]  # exactly what was asked, no more

    log.clear()
    jarvis.handle_chat("use tool beta")
    assert [name for name, _ in log] == ["beta"]


# --- AI construction failures happen in PHASE 1: nothing has run --------------------------


def test_a_provider_failure_while_planning_raises_before_any_step_executes():
    log: list = []

    class ProseTool(LoggedTool):
        name = "prose"
        natural_language_arguments = True

        @property
        def parameters(self):
            return [
                ToolParameter(name="text", type="string"),
                ToolParameter(name="minutes", type="number"),
            ]

        def execute(self, text: str, minutes) -> ToolResult:
            self.log.append(("prose", {"text": text, "minutes": minutes}))
            return ToolResult.success(message="ok")

    class DownProvider(ToolCallProvider):
        def decide(self, message, candidate_tools):
            raise ToolCallError("all providers 429")

    registry = ToolRegistry()
    registry.register(AlphaTool(log))
    registry.register(ProseTool(log))
    jarvis, _ = make_jarvis(tools=registry, provider=DownProvider())

    with pytest.raises(ProviderError, match="all providers 429"):
        jarvis.handle_chat("use tool alpha, then use tool prose remind me to stretch")

    assert log == []  # alpha did NOT run


# --- 30/31: M15 and AutomationEngine isolation ----------------------------------------------


COMPUTER_TOOL_NAMES = (
    "get_active_window",
    "get_screen_info",
    "get_screen_size",
    "open_application",
    "open_url",
    "type_text",
    "press_key",
    "move_mouse",
    "click_mouse",
    "double_click",
    "scroll",
)


def test_a_command_naming_computer_tools_cannot_run_them_when_no_computer_registry_exists():
    """core.tools-shaped registry only (what M15/AutomationEngine hold): the
    whole command is rejected as unknown tools and nothing executes."""
    log: list = []
    jarvis, _ = make_jarvis(tools=logged_registry(log))

    reply, _, _ = jarvis.handle_chat("use tool alpha, then use tool open_url https://luniti.app")

    assert log == []
    assert "Unknown tool: 'open_url'" in reply


def test_the_planner_never_adds_computer_tools_to_the_ordinary_registry():
    backend = FakeComputerBackend()
    ordinary = logged_registry([])
    jarvis, _ = make_jarvis(
        tools=ordinary, computer_tools=create_computer_tools_registry(backend=backend)
    )

    jarvis.handle_chat("use tool open_url https://luniti.app, then use tool scroll -600")

    for name in COMPUTER_TOOL_NAMES:
        assert ordinary.has(name) is False
    assert backend.calls == [("open_url", "https://luniti.app"), ("scroll", -600)]


def test_m15_and_automation_engines_still_hold_no_computer_tool_in_the_real_app():
    import app.main as live_app

    assert live_app.job_engine.tools is live_app.core.tools
    assert live_app.automation_engine.tools is live_app.core.tools
    assert live_app.core.computer_tools is not live_app.core.tools
    from app.computer.defaults import JOB_COMPUTER_TOOL_NAMES

    for name in COMPUTER_TOOL_NAMES:  # Slice 4: Jobs hold exactly the narrow allowlist (no input tool)
        assert live_app.core.tools.has(name) is (name in JOB_COMPUTER_TOOL_NAMES)
        assert live_app.job_engine.tools.has(name) is (name in JOB_COMPUTER_TOOL_NAMES)
        assert live_app.core.computer_tools.has(name) is True


def test_the_command_layer_is_not_imported_by_m15_or_the_automation_engine():
    """Structural isolation: neither engine's package refers to the
    explicit-command layer or the computer tools at all."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent / "app"
    for package in ("orchestration", "automation"):
        for path in (root / package).rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            assert "app.tools.command" not in text, path
            assert "app.computer" not in text, path


def test_a_m15_plan_step_still_cannot_name_a_computer_tool():
    from app.orchestration.base import RetryPolicy
    from app.orchestration.validator import PlanValidationError, validate_plan

    import app.main as live_app

    from app.computer.defaults import JOB_COMPUTER_TOOL_NAMES

    for name in COMPUTER_TOOL_NAMES:
        if name in JOB_COMPUTER_TOOL_NAMES:
            continue  # Slice 4: the allowlist is plannable; every other computer tool is still rejected
        with pytest.raises(PlanValidationError, match="unavailable tool"):
            validate_plan(
                [{"id": "s1", "tool": name, "arguments": {}}],
                live_app.job_engine.tools,
                max_steps=20,
                default_retry=RetryPolicy(max_attempts=1, delay_seconds=0.1),
            )


def test_an_automation_cannot_run_a_computer_tool():
    import app.main as live_app

    # Slice 4: open_url / scroll are Job-reachable now, so the automation boundary is the `automatable` flag that
    # AutomationEngine and CreateAutomation read (never executed here - that would drive the real browser).
    for name in ("open_url", "scroll", "get_active_window"):
        assert live_app.automation_engine.tools.get(name).automatable is False
    # every input tool is still simply unknown to an automation
    for name, params in (("type_text", {"text": "x"}), ("click_mouse", {"x": 1, "y": 1}), ("press_key", {"key": "enter"})):
        assert "Unknown tool" in live_app.automation_engine.tools.execute(name, params=params).message


# --- 32: the actual /chat endpoint ------------------------------------------------------------


@pytest.fixture
def chat_client(monkeypatch):
    """The real FastAPI app and the real `core` singleton, with only its
    storage (in-memory) and computer backend (recording fake) swapped."""
    import app.main as main

    backend = FakeComputerBackend()
    monkeypatch.setattr(main.core, "memory", SQLiteMemoryStore(":memory:"))
    monkeypatch.setattr(
        main.core, "computer_tools", create_computer_tools_registry(backend=backend)
    )
    return TestClient(main.app, base_url="http://localhost"), backend


def test_chat_endpoint_runs_the_regression_sequence_through_the_real_pipeline(chat_client):
    client, backend = chat_client

    response = client.post(
        "/chat", json={"message": "use tool open_url https://luniti.app, then use tool scroll -600"}
    )

    assert response.status_code == 200
    assert response.json()["reply"] == "1. Opened 'https://luniti.app'.\n2. Scrolled by -600."
    assert backend.calls == [("open_url", "https://luniti.app"), ("scroll", -600)]


def test_chat_endpoint_rejects_the_wait_sequence_with_a_200_reply_and_no_execution(chat_client):
    client, backend = chat_client

    response = client.post(
        "/chat",
        json={
            "message": "use tool open_url https://luniti.app, wait for 5 seconds, then use tool scroll -600"
        },
    )

    assert response.status_code == 200
    reply = response.json()["reply"]
    assert "Nothing was executed" in reply
    assert "not an available tool" in reply
    assert backend.calls == []


def test_chat_endpoint_rejects_an_unknown_step_before_the_first_step_runs(chat_client):
    client, backend = chat_client

    response = client.post(
        "/chat", json={"message": "use tool open_url https://luniti.app, then use tool fake_tool"}
    )

    assert response.status_code == 200
    assert "Unknown tool: 'fake_tool'" in response.json()["reply"]
    assert backend.calls == []


@pytest.mark.parametrize(
    "message",
    [
        "use tool fake_tool, then use tool scroll -600",
        "use tool scroll -600 please",
        "use tool scroll, then use tool scroll -600",
        "use tool get_active_window, then",
        "use tool move_mouse somewhere",
        "use tool open_url ftp://example.com, then use tool scroll 5",
    ],
)
def test_chat_endpoint_fails_closed_for_the_whole_class_of_bad_commands(chat_client, message):
    client, backend = chat_client

    response = client.post("/chat", json={"message": message})

    assert response.status_code == 200
    assert "Nothing was executed" in response.json()["reply"] or "nothing was executed" in response.json()["reply"]
    assert backend.calls == []


def test_chat_endpoint_and_a_prose_message_are_not_treated_as_a_tool_command(chat_client, monkeypatch):
    import app.main as main

    client, backend = chat_client

    class Canned(AIProvider):
        def generate_reply(self, message: str) -> str:
            return "canned"

    monkeypatch.setattr(main.core, "provider", Canned())

    response = client.post("/chat", json={"message": "my tool is great"})

    assert response.status_code == 200
    assert response.json()["reply"] == "canned"
    assert backend.calls == []
