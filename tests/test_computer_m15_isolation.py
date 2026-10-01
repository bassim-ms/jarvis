"""
Tests for M17.1-C: Connect Computer Control to Jarvis's real chat
execution path while keeping M15/AutomationEngine from gaining
computer-control access.

Architecture under test (see app/core.py's own module docstring and
`JarvisCore._resolve_tool_registry`, and app/main.py's `core.computer_tools`
wiring): `JarvisCore.computer_tools` is a SEPARATE ToolRegistry from
`JarvisCore.tools`. `self.tools` is the exact registry object M15's
JobEngine and AutomationEngine are constructed with (app/main.py) -
`_resolve_tool_registry` only ever consults `self.computer_tools` for an
EXPLICITLY named tool that registry actually contains; every other case
(including the ambiguous "nothing named" case, and every M15/automation
code path, which never holds a reference to `self.computer_tools` at
all) uses `self.tools`, unchanged. This file proves that separation is
real, deterministic (a pure function of registry membership, never of
message/LLM content), and doesn't weaken or duplicate any existing
permission check.

Everything here runs against FakeComputerBackend - no real mouse
movement, keystrokes, application launch, or screen access anywhere in
this file. The one exception is importing `app.main` itself (to prove
the REAL production wiring, not just a hand-built JarvisCore) - that
import only performs registry membership checks (`.has(...)`), never
`.execute(...)`, so it triggers no OS action even though the real
WindowsComputerBackend is constructed (construction alone has no side
effects - see app/computer/windows_backend.py).
"""

import pytest

from app.computer.base import (
    AllowedApplication,
    ComputerBackend,
    ComputerControlError,
    MouseButton,
    ScreenInfo,
    WindowInfo,
)
from app.computer.defaults import (
    JOB_COMPUTER_TOOL_NAMES,
    create_computer_tools_registry,
    register_computer_tools,
)
from app.computer.tools import PressKeyTool
from app.core import JarvisCore
from app.memory.sqlite_store import SQLiteMemoryStore
from app.orchestration.base import RetryPolicy
from app.orchestration.validator import PlanValidationError, validate_plan
from app.providers.base import AIProvider
from app.tools.base import PermissionLevel, Tool, ToolResult, ToolResultStatus
from app.tools.decision import ToolCall, ToolCallBuilder, ToolCallError, ToolCallProvider
from app.tools.registry import ToolRegistry

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


class FakeComputerBackend(ComputerBackend):
    """Records every call instead of touching the real OS."""

    def __init__(self):
        self.opened_applications: list[AllowedApplication] = []
        self.opened_urls: list[str] = []
        self.typed_text: list[str] = []
        self.pressed_keys: list[str] = []
        self.moved_to: list[tuple[int, int]] = []
        self.clicks: list[tuple[int | None, int | None, MouseButton]] = []
        self.double_clicks: list[tuple[int, int, MouseButton]] = []
        self.scrolls: list[int] = []

    def get_active_window(self) -> WindowInfo:
        return WindowInfo(title="Notepad", process_name="notepad.exe", is_maximized=True)

    def get_screen_info(self) -> ScreenInfo:
        return ScreenInfo(width=1920, height=1080)

    def open_application(self, app: AllowedApplication) -> None:
        self.opened_applications.append(app)

    def open_url(self, url: str) -> None:
        self.opened_urls.append(url)

    def type_text(self, text: str) -> None:
        self.typed_text.append(text)

    def press_key(self, key: str) -> None:
        self.pressed_keys.append(key)

    def move_mouse(self, x: int, y: int) -> None:
        self.moved_to.append((x, y))

    def click_mouse(self, x, y, button: MouseButton) -> None:
        self.clicks.append((x, y, button))

    def double_click(self, x: int, y: int, button: MouseButton) -> None:
        self.double_clicks.append((x, y, button))

    def scroll(self, amount: int) -> None:
        self.scrolls.append(amount)


class UnusedProvider(AIProvider):
    """Fails the test if the chat AI provider is ever actually called -
    a deterministic TOOL dispatch never needs it."""

    def generate_reply(self, message: str) -> str:
        raise AssertionError(
            f"AIProvider.generate_reply should not have been called, got: {message!r}"
        )


def make_jarvis(computer_backend: FakeComputerBackend | None = None, tools: ToolRegistry | None = None):
    computer_backend = computer_backend or FakeComputerBackend()
    computer_tools = create_computer_tools_registry(backend=computer_backend)
    jarvis = JarvisCore(
        provider=UnusedProvider(),
        memory=SQLiteMemoryStore(":memory:"),
        tools=tools or ToolRegistry(),
        computer_tools=computer_tools,
    )
    return jarvis, computer_backend


# --- 1/2. Ordinary interactive chat reaches computer tools ----------------


def test_interactive_chat_reaches_a_read_computer_tool():
    jarvis, backend = make_jarvis()

    reply, _, _ = jarvis.handle_chat("use tool get_active_window")

    assert "Notepad" in reply


def test_interactive_chat_reaches_a_safe_action_computer_tool_and_the_fake_backend():
    jarvis, backend = make_jarvis()

    jarvis.handle_chat("use tool open_url https://example.com")

    assert backend.opened_urls == ["https://example.com"]


def test_interactive_chat_dispatch_uses_the_computer_tools_registry_not_the_tools_registry():
    jarvis, backend = make_jarvis()

    assert jarvis.tools.has("open_url") is False
    assert jarvis.computer_tools.has("open_url") is True

    jarvis.handle_chat("use tool press_key enter")

    assert backend.pressed_keys == ["enter"]


# --- 2b. explicit multi-tool sequences reach computer tools ----------------


class ExplodingToolCallProvider(ToolCallProvider):
    """Fails the test if an AI ToolCallProvider is ever consulted. Every
    computer tool's arguments are bound deterministically from its own
    schema, so a computer-tool command must never reach an LLM."""

    def __init__(self):
        self.calls: list[str] = []

    def decide(self, message: str, candidate_tools: list[Tool]) -> ToolCall:
        self.calls.append(message)
        raise AssertionError(f"AI tool-call construction must not run, got: {message!r}")


def make_sequence_jarvis(tools: ToolRegistry | None = None):
    backend = FakeComputerBackend()
    provider = ExplodingToolCallProvider()
    jarvis = JarvisCore(
        provider=UnusedProvider(),
        memory=SQLiteMemoryStore(":memory:"),
        tools=tools if tools is not None else ToolRegistry(),
        computer_tools=create_computer_tools_registry(backend=backend),
        tool_call_builder=ToolCallBuilder(provider),
    )
    return jarvis, backend, provider


def test_interactive_chat_executes_the_reported_move_then_click_sequence():
    """"use tool move_mouse 500 500, then use tool click_mouse 500 500":
    both steps run, in order, through the real `computer_tools` registry,
    each landing at exactly the coordinates its own step named - with no AI
    call to interpret either step."""
    jarvis, backend, provider = make_sequence_jarvis()

    reply, _, _ = jarvis.handle_chat(
        "use tool move_mouse 500 500, then use tool click_mouse 500 500"
    )

    assert backend.moved_to == [(500, 500)]
    assert backend.clicks == [(500, 500, MouseButton.LEFT)]
    assert "Moved mouse to (500, 500)" in reply
    assert "Clicked left at (500, 500)" in reply
    assert provider.calls == []


def test_interactive_chat_computer_tool_sequence_has_no_argument_leakage():
    """Two DISTINCT computer tools that both take numbers (move_mouse,
    scroll): each step's arguments come only from its own step."""
    jarvis, backend, provider = make_sequence_jarvis()

    jarvis.handle_chat("use tool move_mouse 500 500, then use tool scroll 20")

    assert backend.moved_to == [(500, 500)]
    assert backend.scrolls == [20]
    assert provider.calls == []


def test_a_computer_tool_sequence_still_never_reaches_m15_or_automation():
    """The registry M15's JobEngine/AutomationEngine actually hold
    (core.tools-shaped) never gains move_mouse/click_mouse just because a
    sequence named them - computer tools stay reachable ONLY through
    core.computer_tools, exactly as for a single invocation (see
    COMPUTER_TOOL_NAMES checks elsewhere in this file)."""
    core_tools_shaped_registry = ToolRegistry()  # same shape as the real core.tools
    jarvis, backend, _ = make_sequence_jarvis(tools=core_tools_shaped_registry)

    jarvis.handle_chat("use tool move_mouse 1 1, then use tool click_mouse 1 1")

    assert core_tools_shaped_registry.has("move_mouse") is False
    assert core_tools_shaped_registry.has("click_mouse") is False
    assert jarvis.computer_tools.has("move_mouse") is True
    assert jarvis.computer_tools.has("click_mouse") is True
    assert backend.moved_to == [(1, 1)]


# --- the exact reported "open_url, then scroll" live command ----------------


def test_the_exact_reported_open_url_then_scroll_sequence_executes_both_steps():
    """"use tool open_url https://luniti.app, then use tool scroll -600":
    open_url receives exactly the URL, scroll exactly -600, in that order,
    against the REAL computer tools - and no AI call interprets either
    argument."""
    jarvis, backend, provider = make_sequence_jarvis()

    reply, _, _ = jarvis.handle_chat(
        "use tool open_url https://luniti.app, then use tool scroll -600"
    )

    assert backend.opened_urls == ["https://luniti.app"]
    assert backend.scrolls == [-600]
    assert reply == "1. Opened 'https://luniti.app'.\n2. Scrolled by -600."
    assert provider.calls == []


def test_open_url_wait_scroll_is_rejected_whole_and_never_opens_a_malformed_url():
    """The command that used to open a random .lnk target: there is no
    "wait" tool, so the WHOLE command is rejected at validation - the
    browser is not opened (with or without a mangled URL), nothing scrolls."""
    jarvis, backend, provider = make_sequence_jarvis()

    reply, _, _ = jarvis.handle_chat(
        "use tool open_url https://luniti.app, wait for 5 seconds, then use tool scroll -600"
    )

    assert backend.opened_urls == []
    assert backend.scrolls == []
    assert provider.calls == []
    assert '"wait"' in reply and "not an available tool" in reply
    assert "Nothing was executed" in reply


def test_a_step_whose_value_is_invalid_prevents_every_earlier_step_from_running():
    """Validation covers argument VALUES, not just shape: "scroll 999999"
    exceeds the tool's own bound, so open_url - a perfectly valid first
    step - must not run either."""
    jarvis, backend, _ = make_sequence_jarvis()

    reply, _, _ = jarvis.handle_chat(
        "use tool open_url https://luniti.app, then use tool scroll 999999"
    )

    assert backend.opened_urls == []
    assert backend.scrolls == []
    assert "Step 2 (scroll)" in reply
    assert "Nothing was executed" in reply


def test_open_url_then_scroll_sequence_still_resolves_only_against_computer_tools():
    """M15/AutomationEngine must remain unable to reach either tool named
    in this exact sequence - checked against the real app.main singleton
    (not a hand-built stand-in), without executing anything or making an
    AI call (see this file's own module docstring)."""
    import app.main as live_app
    from app.tools.command import ParsedCommand, parse_explicit_command

    message = "use tool open_url https://luniti.app, then use tool scroll -600"

    def lookup(name):
        return live_app.core._resolve_tool_registry(name).get(name)

    parsed = parse_explicit_command(message, lookup)
    assert isinstance(parsed, ParsedCommand)
    names = [step.tool.name for step in parsed.steps]
    assert names == ["open_url", "scroll"]
    assert [step.arguments for step in parsed.steps] == [
        {"url": "https://luniti.app"},
        {"amount": -600},
    ]

    for name in names:
        # Slice 4: both are on the Job allowlist - but never automatable, and still in the interactive registry.
        assert live_app.job_engine.tools.has(name) is True
        assert live_app.automation_engine.tools.get(name).automatable is False
        assert live_app.core.computer_tools.has(name) is True


# --- 3. Invalid computer arguments are still rejected ----------------------


def test_invalid_application_is_still_rejected_through_interactive_chat():
    jarvis, backend = make_jarvis()

    jarvis.handle_chat("use tool open_application cmd")

    assert backend.opened_applications == []


def test_invalid_url_scheme_is_still_rejected_through_interactive_chat():
    jarvis, backend = make_jarvis()

    jarvis.handle_chat("use tool open_url file:///C:/secrets.txt")

    assert backend.opened_urls == []


def test_disallowed_key_is_still_rejected_through_interactive_chat():
    jarvis, backend = make_jarvis()

    jarvis.handle_chat("use tool press_key banana")

    assert backend.pressed_keys == []


def test_disallowed_key_combination_is_still_rejected_through_interactive_chat():
    """"ctrl+alt+delete" is a valid three-token combo shape (two
    modifiers + a target), but "l+ctrl" has no leading modifier at all -
    proving Fix B's combo support didn't accidentally loosen the "must
    start with a modifier" rule when reached through the real chat path."""
    jarvis, backend = make_jarvis()

    jarvis.handle_chat("use tool press_key l+ctrl")

    assert backend.pressed_keys == []


# --- 4. A denied computer capability prevents execution -------------------


class _ConfirmTierPressKeyTool(PressKeyTool):
    """Test-only: same PressKeyTool, classified CONFIRM instead of
    SAFE_ACTION - proves the interactive path's routing to
    `computer_tools` still goes through ToolRegistry.execute's ordinary
    permission gate, with no special-case auto-approval."""

    name = "press_key_confirm_tier"
    permission = PermissionLevel.CONFIRM


def test_a_denied_computer_capability_prevents_execution_via_interactive_chat():
    backend = FakeComputerBackend()
    computer_tools = ToolRegistry()
    computer_tools.register(_ConfirmTierPressKeyTool(backend=backend))
    jarvis = JarvisCore(
        provider=UnusedProvider(),
        memory=SQLiteMemoryStore(":memory:"),
        tools=ToolRegistry(),
        computer_tools=computer_tools,
    )

    jarvis.handle_chat("use tool press_key_confirm_tier enter")

    assert backend.pressed_keys == []


def test_explicit_decline_denies_a_gated_computer_tool():
    backend = FakeComputerBackend()
    computer_tools = ToolRegistry()
    computer_tools.register(_ConfirmTierPressKeyTool(backend=backend))

    result = computer_tools.execute(
        "press_key_confirm_tier", params={"key": "enter"}, confirmed=False
    )

    assert result.status == ToolResultStatus.DENIED
    assert backend.pressed_keys == []


# --- 5/6. M15 (JobEngine) and AutomationEngine cannot execute computer tools


def test_the_real_app_gives_job_engine_and_automation_engine_only_core_tools():
    """Imports the REAL, running application module (app/main.py) - the
    actual production wiring, not a hand-built test double - and proves
    JobEngine/AutomationEngine share `core.tools` (which never contains a
    computer tool), while `core.computer_tools` is a distinct object
    neither of them was ever given a reference to."""
    import app.main as live_app

    assert live_app.job_engine.tools is live_app.core.tools
    assert live_app.automation_engine.tools is live_app.core.tools
    assert live_app.core.computer_tools is not live_app.core.tools

    # Slice 4: Jobs hold EXACTLY the narrow allowlist of computer tools - never an input tool.
    for name in COMPUTER_TOOL_NAMES:
        in_jobs = name in JOB_COMPUTER_TOOL_NAMES
        assert live_app.core.tools.has(name) is in_jobs
        assert live_app.job_engine.tools.has(name) is in_jobs
        assert live_app.core.computer_tools.has(name) is True
    for name in ("type_text", "press_key", "move_mouse", "click_mouse", "double_click", "open_application"):
        assert live_app.job_engine.tools.has(name) is False
    # ...and an automation never runs any of them (AutomationEngine / CreateAutomation read `automatable`).
    for name in JOB_COMPUTER_TOOL_NAMES:
        assert live_app.automation_engine.tools.get(name).automatable is False


def test_get_screen_size_resolves_through_the_real_apps_chat_capability_path():
    """Regression test for a live smoke-test failure: `/chat` with "use
    tool get_screen_size" returned "Unknown tool: 'get_screen_size'"
    because no tool by that exact name existed (only `get_screen_info`
    did) - every routing layer (capability decision, registry
    resolution, ToolCall construction) was already working correctly;
    the tool itself was simply missing under that name. This exercises
    the SAME sequence app/core.py's `_handle_tool` uses, against the
    REAL app.main singleton, stopping short of `registry.execute(...)`
    (this file never executes against the real backend - see its own
    module docstring) - `get_screen_size` must resolve to
    `core.computer_tools` and build a valid, empty-parameter ToolCall,
    exactly like `get_screen_info` already does."""
    import app.main as live_app

    decision = live_app.core.capability_router.route("use tool get_screen_size")
    assert decision.capability.value == "tool"
    assert decision.tool_name == "get_screen_size"

    registry = live_app.core._resolve_tool_registry(decision.tool_name)
    assert registry is live_app.core.computer_tools
    assert registry.has("get_screen_size") is True
    assert registry.get("get_screen_size").permission is PermissionLevel.READ

    tool_call = live_app.core.tool_call_builder.build(
        "use tool get_screen_size", registry, decision.tool_name
    )
    assert tool_call.tool_name == "get_screen_size"
    assert tool_call.parameters == {}


def test_scroll_registers_under_its_real_name_and_scroll_mouse_is_not_a_tool():
    """Investigated as a reported live smoke-test failure: `/chat` with
    "use tool scroll_mouse 5" returned "Unknown tool: 'scroll_mouse'".
    Unlike the get_screen_size regression above, this was NOT a
    registration bug: `ScrollTool` (app/computer/tools.py) has always
    been named "scroll" and IS correctly registered by
    register_computer_tools (app/computer/defaults.py) - "scroll_mouse"
    has never existed anywhere in this codebase or its tests
    (COMPUTER_TOOL_NAMES above has always listed "scroll", matching
    "double_click"'s own no-"_mouse"-suffix naming). Confirmed with the
    user: no rename/alias is wanted - the live test simply needs to
    invoke "scroll", not "scroll_mouse". This test locks in exactly that
    finding as a permanent regression guard, using the SAME real
    app.main singleton and routing sequence
    (capability decision -> registry resolution) the get_screen_size
    regression above uses, stopping short of `registry.execute(...)`
    for the same reason (see this file's own module docstring)."""
    import app.main as live_app

    decision = live_app.core.capability_router.route("use tool scroll 5")
    assert decision.capability.value == "tool"
    assert decision.tool_name == "scroll"

    registry = live_app.core._resolve_tool_registry(decision.tool_name)
    assert registry is live_app.core.computer_tools
    assert registry.has("scroll") is True
    assert registry.get("scroll").permission is PermissionLevel.SAFE_ACTION

    # "scroll_mouse" is not, and was never, a registered tool name -
    # under any registry the real app exposes.
    assert registry.has("scroll_mouse") is False
    assert live_app.core.tools.has("scroll_mouse") is False
    assert live_app.core.computer_tools.has("scroll_mouse") is False


def test_scroll_is_registered_and_executable_through_the_tool_registry():
    """Complements the routing-only regression above by proving "scroll"
    is not just present but actually runnable through ToolRegistry.execute
    (params flowing through unchanged, SAFE_ACTION running immediately, no
    confirmation gate) - against a FakeComputerBackend, never the real
    backend (see this file's own module docstring on OS side effects).
    Also proves "scroll_mouse" fails the same safe, typed way any unknown
    tool name already does - a clean ToolResult.error, never a crash -
    reproducing the exact live failure message under test."""
    backend = FakeComputerBackend()
    registry = create_computer_tools_registry(backend=backend)

    result = registry.execute("scroll", params={"amount": 5})

    assert result.status == ToolResultStatus.SUCCESS
    assert backend.scrolls == [5]

    unknown_result = registry.execute("scroll_mouse", params={"amount": 5})

    assert unknown_result.status == ToolResultStatus.ERROR
    assert unknown_result.message == "Unknown tool: 'scroll_mouse'"
    assert backend.scrolls == [5]  # the unknown-name call never reached the backend


def test_create_default_registry_never_includes_a_computer_tool():
    """Regression guard directly on app/tools/defaults.py - the function
    that builds the registry M15/AutomationEngine actually receive
    (app/main.py) - independent of app.main's own wiring above."""
    from app.tools.defaults import create_default_registry

    registry = create_default_registry()

    for name in COMPUTER_TOOL_NAMES:
        assert registry.has(name) is False


def test_automation_engine_cannot_run_a_computer_tool_by_name():
    """Even if an automation somehow specified a computer-tool action
    name, AutomationEngine's own registry (== core.tools) simply doesn't
    have it - the same "Unknown tool" outcome any nonexistent tool name
    gets, not a special denial."""
    import app.main as live_app

    result = live_app.automation_engine.tools.execute(
        "open_application", params={"app": "notepad"}
    )

    assert result.status == ToolResultStatus.ERROR
    assert "Unknown tool" in result.message


def test_job_engines_registry_cannot_run_a_computer_tool_by_name():
    import app.main as live_app

    result = live_app.job_engine.tools.execute("open_application", params={"app": "notepad"})

    assert result.status == ToolResultStatus.ERROR
    assert "Unknown tool" in result.message


# --- 7/10. An M15 plan cannot inject a computer tool call, deterministically


@pytest.mark.parametrize("tool_name", COMPUTER_TOOL_NAMES)
def test_a_plan_step_naming_any_computer_tool_is_rejected_by_plan_validation(tool_name):
    """`validate_plan` (app/orchestration/validator.py) is the one gate
    every Job plan - however it was produced, including a hallucinated or
    adversarially-crafted planner response - must pass before JobEngine
    will ever run a step. Run against the SAME kind of registry JobEngine
    actually holds (one built without computer tools), naming EVERY
    computer tool is rejected uniformly - proving this doesn't depend on
    the planner/LLM choosing not to try, but on the tool simply not
    existing in that registry at all."""
    registry = ToolRegistry()  # the same shape as the real core.tools: no computer tools

    with pytest.raises(PlanValidationError, match="unavailable tool"):
        validate_plan(
            [{"id": "s1", "tool": tool_name, "arguments": {}}],
            registry,
            max_steps=20,
            default_retry=RetryPolicy(max_attempts=1, delay_seconds=0.1),
        )


def test_plan_validation_still_accepts_a_step_for_a_tool_the_registry_actually_has():
    """Contrast case: validate_plan works normally for a tool that IS
    registered - proving the rejection above is about registry
    membership, not a blanket/broken validator."""

    class PingTool(Tool):
        name = "ping"
        description = "ping"
        permission = PermissionLevel.READ

        def execute(self):
            return ToolResult.success(message="pong")

    registry = ToolRegistry()
    registry.register(PingTool())

    steps = validate_plan(
        [{"id": "s1", "tool": "ping", "arguments": {}}],
        registry,
        max_steps=20,
        default_retry=RetryPolicy(max_attempts=1, delay_seconds=0.1),
    )

    assert steps[0].tool_name == "ping"


def test_the_real_app_registry_rejects_an_injected_computer_tool_plan_step():
    """Same proof as above, against the REAL app.main registry M15
    actually uses in production, not a freshly built stand-in."""
    import app.main as live_app

    with pytest.raises(PlanValidationError, match="unavailable tool"):
        validate_plan(
            [{"id": "s1", "tool": "open_application", "arguments": {"app": "notepad"}}],
            live_app.job_engine.tools,
            max_steps=20,
            default_retry=RetryPolicy(max_attempts=1, delay_seconds=0.1),
        )


# --- _resolve_tool_registry: pure, deterministic, LLM-independent ---------


def test_resolve_tool_registry_is_a_pure_function_of_registry_membership():
    """Directly exercises `_resolve_tool_registry` across every
    combination that matters - proving the routing decision depends ONLY
    on (tool_name, what's registered where), never on anything else in
    the message/decision, and therefore can't be swayed by an LLM's
    phrasing, confidence, or any other content it controls."""
    backend = FakeComputerBackend()
    computer_tools = create_computer_tools_registry(backend=backend)
    ordinary_tools = ToolRegistry()

    class PingTool(Tool):
        name = "ping"
        description = "ping"
        permission = PermissionLevel.READ

        def execute(self):
            return ToolResult.success(message="pong")

    ordinary_tools.register(PingTool())

    jarvis_with_computer_tools = JarvisCore(
        provider=UnusedProvider(),
        memory=SQLiteMemoryStore(":memory:"),
        tools=ordinary_tools,
        computer_tools=computer_tools,
    )
    jarvis_without_computer_tools = JarvisCore(
        provider=UnusedProvider(),
        memory=SQLiteMemoryStore(":memory:"),
        tools=ordinary_tools,
    )

    # A named computer tool, with computer_tools configured -> computer_tools.
    assert (
        jarvis_with_computer_tools._resolve_tool_registry("open_application")
        is computer_tools
    )
    # The ambiguous "nothing named" case -> always self.tools, even with
    # computer_tools configured.
    assert jarvis_with_computer_tools._resolve_tool_registry(None) is ordinary_tools
    # An ordinary tool name that computer_tools doesn't have -> self.tools.
    assert jarvis_with_computer_tools._resolve_tool_registry("ping") is ordinary_tools
    # A name nothing has at all -> self.tools (unregistered-tool handling
    # happens inside ToolRegistry.execute, unchanged).
    assert jarvis_with_computer_tools._resolve_tool_registry("nonexistent") is ordinary_tools
    # No computer_tools configured at all (every pre-M17.1-C caller) ->
    # always self.tools, regardless of tool_name.
    assert (
        jarvis_without_computer_tools._resolve_tool_registry("open_application")
        is ordinary_tools
    )


# --- 8. Existing M15 tools still work exactly as before -------------------


def test_task_capability_still_creates_a_job_when_computer_tools_is_also_configured():
    from app.orchestration.sqlite_store import SQLiteJobStore
    from app.orchestration.tools import CreateJobTool

    job_store = SQLiteJobStore(":memory:")
    tools = ToolRegistry()
    tools.register(CreateJobTool(store=job_store))
    computer_tools = create_computer_tools_registry(backend=FakeComputerBackend())
    jarvis = JarvisCore(
        provider=UnusedProvider(),
        memory=SQLiteMemoryStore(":memory:"),
        tools=tools,
        computer_tools=computer_tools,
    )

    jarvis.handle_chat("Prepare everything for tomorrow's robotics meeting.")

    jobs = job_store.list()
    assert len(jobs) == 1
    assert jobs[0].goal == "Prepare everything for tomorrow's robotics meeting."


# --- 9. Existing interactive tools still work exactly as before -----------


def test_an_ordinary_registered_tool_still_dispatches_through_self_tools():
    class PingTool(Tool):
        name = "ping"
        description = "ping"
        permission = PermissionLevel.READ

        def execute(self):
            return ToolResult.success(message="pong")

    ordinary_tools = ToolRegistry()
    ordinary_tools.register(PingTool())
    jarvis, _ = make_jarvis(tools=ordinary_tools)

    reply, _, _ = jarvis.handle_chat("use tool ping")

    assert "pong" in reply


def test_jarvis_core_without_computer_tools_behaves_exactly_as_before():
    """No `computer_tools` argument at all (every pre-M17.1-C caller) -
    `self.computer_tools` is `None`, and `_handle_tool` behaves exactly
    as it did before this milestone existed."""

    class PingTool(Tool):
        name = "ping"
        description = "ping"
        permission = PermissionLevel.READ

        def execute(self):
            return ToolResult.success(message="pong")

    tools = ToolRegistry()
    tools.register(PingTool())
    jarvis = JarvisCore(
        provider=UnusedProvider(), memory=SQLiteMemoryStore(":memory:"), tools=tools
    )

    assert jarvis.computer_tools is None
    reply, _, _ = jarvis.handle_chat("use tool ping")
    assert "pong" in reply


# --- 12. No duplicate permission logic was introduced ----------------------


def test_computer_backend_interface_has_no_permission_parameters():
    """ComputerBackend's abstract methods (app/computer/base.py) take no
    permission/confirmation/approval-shaped parameter anywhere - proving
    neither M17.1-C nor M17.1-D's `double_click` addition introduced a
    second, backend-level permission mechanism alongside ToolRegistry's -
    this is a regression guard on that invariant."""
    import inspect

    disallowed_names = {"confirmed", "approve", "approved", "permission", "authorized"}
    for method_name in (
        "get_active_window",
        "get_screen_info",
        "open_application",
        "open_url",
        "type_text",
        "press_key",
        "move_mouse",
        "click_mouse",
        "double_click",
        "scroll",
    ):
        method = getattr(ComputerBackend, method_name)
        params = set(inspect.signature(method).parameters) - {"self"}
        assert not (params & disallowed_names), (
            f"{method_name} unexpectedly has a permission-shaped parameter: {params}"
        )


def test_register_computer_tools_still_relies_solely_on_toolregistry_for_permission():
    """Registering computer tools (app/computer/defaults.py) attaches no
    permission wrapper of its own - each tool's `permission` attribute
    (enforced only by ToolRegistry.execute) is unchanged by registration."""
    registry = ToolRegistry()
    register_computer_tools(registry, backend=FakeComputerBackend())

    assert registry.get("get_active_window").permission is PermissionLevel.READ
    assert registry.get("open_application").permission is PermissionLevel.SAFE_ACTION
