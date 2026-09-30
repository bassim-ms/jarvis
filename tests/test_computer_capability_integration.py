"""
Tests for M17.1-B: Computer Control Permission Integration.

Proves the M17.1-A computer-control tools are reachable through Jarvis's
REAL, EXISTING capability/permission machinery end-to-end -
DeterministicCapabilityRouter deciding Capability.TOOL,
ToolCallBuilder constructing a validated ToolCall, and
ToolRegistry.execute enforcing PermissionLevel - exactly the same path
every other tool family (calendar, reminders, web_search, orchestration)
already goes through. app/core.py, app/capability.py, and
app/tools/registry.py are completely UNMODIFIED by this milestone; this
file only proves the existing machinery already handles computer tools
correctly once they're registered on an ordinary ToolRegistry, via
app/computer/defaults.py.

Every test builds its own fresh JarvisCore + ToolRegistry (the same
pattern tests/test_core.py's own TASK/TASK_STATUS/CONFIRMATION
integration tests already use for orchestration tools) - never the real
running app's `core.tools` singleton, which is shared with M15's
JobEngine (see app/computer/defaults.py's own module docstring for why
that matters and is deliberately out of scope here).

Everything runs against FakeComputerBackend - no real mouse movement,
keystrokes, application launch, or screen access anywhere in this file.
"""

import pytest

from app.computer.base import (
    AllowedApplication,
    ComputerBackend,
    MouseButton,
    ScreenInfo,
    WindowInfo,
)
from app.computer.defaults import create_computer_tools_registry, register_computer_tools
from app.computer.tools import PressKeyTool
from app.core import JarvisCore
from app.memory.sqlite_store import SQLiteMemoryStore
from app.providers.base import AIProvider
from app.tools.base import PermissionLevel, ToolArgumentError, ToolResultStatus
from app.tools.decision import ToolCall, ToolCallBuilder, ToolCallProvider
from app.tools.registry import ToolRegistry


class FakeComputerBackend(ComputerBackend):
    """Records every call instead of touching the real OS - the same
    fake used in tests/test_computer_tools.py, duplicated locally so this
    file has no cross-test-module import dependency."""

    def __init__(self):
        self.opened_applications: list[AllowedApplication] = []
        self.opened_urls: list[str] = []
        self.typed_text: list[str] = []
        self.pressed_keys: list[str] = []
        self.moved_to: list[tuple[int, int]] = []
        self.clicks: list[tuple[int | None, int | None, MouseButton]] = []
        self.double_clicks: list[tuple[int, int, MouseButton]] = []
        self.scrolls: list[int] = []
        self.active_window_reads = 0
        self.screen_info_reads = 0

    def get_active_window(self) -> WindowInfo:
        self.active_window_reads += 1
        return WindowInfo(title="Notepad", process_name="notepad.exe", is_maximized=True)

    def get_screen_info(self) -> ScreenInfo:
        self.screen_info_reads += 1
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
    """An AIProvider that fails the test if it's ever actually called -
    proves a deterministic TOOL dispatch never reaches the chat AI
    provider, the same guarantee test_core.py's own TASK tests already
    check for orchestration tools (`provider.last_message is None`)."""

    def generate_reply(self, message: str) -> str:
        raise AssertionError(
            f"AIProvider.generate_reply should not have been called, got: {message!r}"
        )


class FixedToolCallProvider(ToolCallProvider):
    """A ToolCallProvider that always returns one fixed ToolCall,
    regardless of the message - used only to prove a multi-required-
    parameter computer tool (e.g. move_mouse) also works end-to-end
    through handle_chat, without making a real Gemini call."""

    def __init__(self, tool_call: ToolCall):
        self.tool_call = tool_call
        self.calls: list[tuple[str, list]] = []

    def decide(self, message, candidate_tools):
        self.calls.append((message, candidate_tools))
        return self.tool_call


def make_jarvis(
    registry: ToolRegistry | None = None,
    backend: FakeComputerBackend | None = None,
    tool_call_builder: ToolCallBuilder | None = None,
):
    backend = backend or FakeComputerBackend()
    registry = registry or create_computer_tools_registry(backend=backend)
    jarvis = JarvisCore(
        provider=UnusedProvider(),
        memory=SQLiteMemoryStore(":memory:"),
        tools=registry,
        tool_call_builder=tool_call_builder or ToolCallBuilder(),
    )
    return jarvis, backend


# --- 1. READ capabilities through the normal capability system -----------


def test_get_active_window_runs_through_capability_router_and_registry():
    jarvis, backend = make_jarvis()

    reply, _, sources = jarvis.handle_chat("use tool get_active_window")

    assert backend.active_window_reads == 1
    assert "Notepad" in reply
    assert sources == []


def test_get_screen_info_runs_through_capability_router_and_registry():
    jarvis, backend = make_jarvis()

    reply, _, _ = jarvis.handle_chat("use tool get_screen_info")

    assert backend.screen_info_reads == 1


def test_get_screen_size_runs_through_capability_router_and_registry():
    """Regression test for the exact live smoke-test failure: "use tool
    get_screen_size" must resolve and execute through the real
    capability path, not return "Unknown tool"."""
    jarvis, backend = make_jarvis()

    reply, _, _ = jarvis.handle_chat("use tool get_screen_size")

    assert backend.screen_info_reads == 1
    assert "Unknown tool" not in reply


def test_read_tools_are_registered_as_read_permission():
    registry = create_computer_tools_registry(backend=FakeComputerBackend())

    assert registry.get("get_active_window").permission is PermissionLevel.READ
    assert registry.get("get_screen_info").permission is PermissionLevel.READ
    assert registry.get("get_screen_size").permission is PermissionLevel.READ


# --- 2. SAFE_ACTION capabilities through the normal capability system ----


def test_open_url_runs_through_capability_router_and_registry():
    jarvis, backend = make_jarvis()

    jarvis.handle_chat("use tool open_url https://example.com")

    assert backend.opened_urls == ["https://example.com"]


def test_press_key_runs_through_capability_router_and_registry():
    jarvis, backend = make_jarvis()

    jarvis.handle_chat("use tool press_key enter")

    assert backend.pressed_keys == ["enter"]


def test_open_application_runs_through_capability_router_and_registry():
    jarvis, backend = make_jarvis()

    jarvis.handle_chat("use tool open_application notepad")

    assert backend.opened_applications == [AllowedApplication.NOTEPAD]


def test_type_text_runs_through_capability_router_and_registry():
    jarvis, backend = make_jarvis()

    jarvis.handle_chat("use tool type_text hello world")

    assert backend.typed_text == ["hello world"]


def test_a_multi_parameter_safe_action_tool_also_works_end_to_end():
    """move_mouse takes two required numbers. Both are bound
    deterministically from the tool's own schema - the AI provider (a
    FixedToolCallProvider that would return DIFFERENT coordinates) is never
    consulted, so the mouse lands exactly where the message said."""
    backend = FakeComputerBackend()
    registry = create_computer_tools_registry(backend=backend)
    fixed_provider = FixedToolCallProvider(
        ToolCall(tool_name="move_mouse", parameters={"x": 1, "y": 2})
    )
    jarvis, backend = make_jarvis(
        registry=registry,
        backend=backend,
        tool_call_builder=ToolCallBuilder(call_provider=fixed_provider),
    )

    jarvis.handle_chat("use tool move_mouse 100 200")

    assert backend.moved_to == [(100, 200)]
    assert fixed_provider.calls == []


def test_move_mouse_with_no_coordinates_fails_closed_and_never_asks_an_ai_to_invent_them():
    """"use tool move_mouse somewhere" used to hand the message to an LLM,
    which could invent coordinates. It is now rejected at validation: no
    AI call, no mouse movement."""
    backend = FakeComputerBackend()
    registry = create_computer_tools_registry(backend=backend)
    fixed_provider = FixedToolCallProvider(
        ToolCall(tool_name="move_mouse", parameters={"x": 100, "y": 200})
    )
    jarvis, backend = make_jarvis(
        registry=registry,
        backend=backend,
        tool_call_builder=ToolCallBuilder(call_provider=fixed_provider),
    )

    reply, _, _ = jarvis.handle_chat("use tool move_mouse somewhere")

    assert backend.moved_to == []
    assert fixed_provider.calls == []
    assert "must be a number" in reply
    assert "Nothing was executed" in reply


def test_click_mouse_lands_at_exactly_the_named_coordinates_end_to_end():
    """Regression test for the "clicks at the current position instead
    of the given coordinates" bug: click_mouse's x/y are now required
    (matching move_mouse/double_click's own shape), so this exercises
    the exact same AI-backed construction path as
    test_a_multi_parameter_safe_action_tool_also_works_end_to_end above,
    and must land at EXACTLY (100, 100) - never the current cursor
    position."""
    backend = FakeComputerBackend()
    registry = create_computer_tools_registry(backend=backend)
    fixed_provider = FixedToolCallProvider(
        ToolCall(tool_name="click_mouse", parameters={"x": 100, "y": 100})
    )
    jarvis, backend = make_jarvis(
        registry=registry,
        backend=backend,
        tool_call_builder=ToolCallBuilder(call_provider=fixed_provider),
    )

    jarvis.handle_chat("use tool click_mouse 100 100")

    assert backend.clicks == [(100, 100, MouseButton.LEFT)]


def test_double_click_also_works_end_to_end_through_the_capability_path():
    """double_click (M17.1-D) has the same shape as move_mouse: its two
    required numbers are bound deterministically, never by an AI."""
    backend = FakeComputerBackend()
    registry = create_computer_tools_registry(backend=backend)
    fixed_provider = FixedToolCallProvider(
        ToolCall(tool_name="double_click", parameters={"x": 1, "y": 2})
    )
    jarvis, backend = make_jarvis(
        registry=registry,
        backend=backend,
        tool_call_builder=ToolCallBuilder(call_provider=fixed_provider),
    )

    jarvis.handle_chat("use tool double_click 30 40")

    assert backend.double_clicks == [(30, 40, MouseButton.LEFT)]
    assert fixed_provider.calls == []


def test_safe_action_tools_are_registered_as_safe_action_permission():
    registry = create_computer_tools_registry(backend=FakeComputerBackend())

    for name in (
        "open_application",
        "open_url",
        "type_text",
        "press_key",
        "move_mouse",
        "click_mouse",
        "double_click",
        "scroll",
    ):
        assert registry.get(name).permission is PermissionLevel.SAFE_ACTION


# --- 3. Capability decision is made BEFORE computer-tool execution -------


def test_an_ordinary_message_never_reaches_the_backend_without_tool_invocation():
    """Merely mentioning a computer tool's name, or an application/URL,
    in ordinary conversation - without the explicit TOOL-invocation
    syntax DeterministicCapabilityRouter requires - must never reach the
    backend. This is what "the capability decision gates execution"
    means: CapabilityRouter runs FIRST, and only a TOOL decision ever
    causes ToolRegistry to be consulted at all."""
    registry = create_computer_tools_registry(backend=FakeComputerBackend())
    backend = registry.get("open_application").backend
    provider = _CannedReplyProvider("Sure, I can help with that.")
    jarvis = JarvisCore(
        provider=provider, memory=SQLiteMemoryStore(":memory:"), tools=registry
    )

    reply, _, _ = jarvis.handle_chat(
        "Can you open notepad for me and press enter in it?"
    )

    assert reply == "Sure, I can help with that."
    assert backend.opened_applications == []
    assert backend.pressed_keys == []


def test_capability_router_decision_precedes_any_registry_lookup(monkeypatch):
    """Directly proves ordering: CapabilityRouter.route is called, and
    its result inspected, before ToolRegistry.execute is ever invoked -
    not just "the backend happened not to be called" but the actual
    call-order invariant."""
    backend = FakeComputerBackend()
    registry = create_computer_tools_registry(backend=backend)
    jarvis, _ = make_jarvis(registry=registry, backend=backend)

    call_order: list[str] = []
    original_route = jarvis.capability_router.route
    original_execute = registry.execute

    def recording_route(message):
        call_order.append("capability_decision")
        return original_route(message)

    def recording_execute(name, params=None, confirmed=None):
        call_order.append("registry_execute")
        return original_execute(name, params=params, confirmed=confirmed)

    monkeypatch.setattr(jarvis.capability_router, "route", recording_route)
    monkeypatch.setattr(registry, "execute", recording_execute)

    jarvis.handle_chat("use tool press_key enter")

    assert call_order == ["capability_decision", "registry_execute"]


class _CannedReplyProvider(AIProvider):
    def __init__(self, reply: str):
        self.reply = reply

    def generate_reply(self, message: str) -> str:
        return self.reply


# --- 4/5. The model cannot bypass the check or invoke the backend directly


class _ConfirmTierPressKeyTool(PressKeyTool):
    """Test-only: identical to PressKeyTool, but classified CONFIRM
    instead of SAFE_ACTION. Proves ToolRegistry.execute's permission gate
    applies to a computer tool with no special-case exemption, and that
    nothing in the ordinary chat/TOOL-capability path can ever supply
    `confirmed=True` on a message's behalf - only an explicit caller-side
    `confirmed=` argument to ToolRegistry.execute can, and the TOOL
    capability path never passes one (see app/core.py:_handle_tool)."""

    name = "press_key_confirm_tier"
    permission = PermissionLevel.CONFIRM


def test_a_gated_tool_is_never_auto_approved_through_the_chat_path():
    backend = FakeComputerBackend()
    registry = ToolRegistry()
    registry.register(_ConfirmTierPressKeyTool(backend=backend))
    jarvis, _ = make_jarvis(registry=registry, backend=backend)

    jarvis.handle_chat("use tool press_key_confirm_tier enter")

    assert backend.pressed_keys == []


def test_registry_execute_requires_explicit_confirmation_for_a_gated_tool():
    backend = FakeComputerBackend()
    registry = ToolRegistry()
    registry.register(_ConfirmTierPressKeyTool(backend=backend))

    result = registry.execute("press_key_confirm_tier", params={"key": "enter"})

    assert result.status == ToolResultStatus.CONFIRMATION_REQUIRED
    assert backend.pressed_keys == []


def test_registry_execute_denies_a_gated_tool_on_explicit_decline():
    backend = FakeComputerBackend()
    registry = ToolRegistry()
    registry.register(_ConfirmTierPressKeyTool(backend=backend))

    result = registry.execute(
        "press_key_confirm_tier", params={"key": "enter"}, confirmed=False
    )

    assert result.status == ToolResultStatus.DENIED
    assert backend.pressed_keys == []


def test_a_gated_tool_runs_only_with_explicit_caller_side_confirmation():
    backend = FakeComputerBackend()
    registry = ToolRegistry()
    registry.register(_ConfirmTierPressKeyTool(backend=backend))

    result = registry.execute(
        "press_key_confirm_tier", params={"key": "enter"}, confirmed=True
    )

    assert result.status == ToolResultStatus.SUCCESS
    assert backend.pressed_keys == ["enter"]


def test_a_smuggled_confirmed_parameter_cannot_reach_the_backend():
    """Even if a parameter literally named "confirmed" (or anything else
    outside a tool's own declared schema) ended up in `params`, it can
    never influence ToolRegistry.execute's actual permission gate - that
    gate is driven solely by its own separate `confirmed=` keyword
    argument, never by anything inside `params`. Tool.execute's fixed
    signature (no **kwargs anywhere in app/computer/tools.py) means an
    unexpected key is a clean ToolResult.error, not a bypass."""
    backend = FakeComputerBackend()
    registry = ToolRegistry()
    registry.register(PressKeyTool(backend=backend))

    result = registry.execute("press_key", params={"key": "enter", "confirmed": True})

    assert result.status == ToolResultStatus.ERROR
    assert backend.pressed_keys == []


def test_model_cannot_invoke_the_backend_without_going_through_the_registry():
    """There is no method on any computer Tool that calls the backend
    other than `execute()`, and `execute()` is only ever invoked by
    ToolRegistry.execute (see app/tools/registry.py) - never directly by
    JarvisCore or anything else. Constructing a Tool and inspecting its
    public surface confirms there is no alternate, ungated entry point."""
    backend = FakeComputerBackend()
    tool = PressKeyTool(backend=backend)

    public_methods = [
        name
        for name in dir(tool)
        if not name.startswith("_")
        and callable(getattr(tool, name))
        and name not in ("execute", "validate_arguments")
    ]
    # name/description/permission/parameters are plain attributes/
    # properties, not callables that reach the backend - only `execute`
    # does, and it always goes through ToolRegistry's permission gate.
    # `validate_arguments` is the side-effect-free pre-flight check (see
    # app/tools/base.py): it must never act on the backend.
    assert public_methods == []

    tool.validate_arguments(key="enter")
    with pytest.raises(ToolArgumentError):
        tool.validate_arguments(key="banana")
    assert backend.pressed_keys == []


# --- 6. Existing unrelated capability behavior is unchanged ---------------


def test_direct_capability_is_unaffected_by_computer_tools_being_registered():
    registry = create_computer_tools_registry(backend=FakeComputerBackend())
    provider = _CannedReplyProvider("The capital of Japan is Tokyo.")
    jarvis = JarvisCore(
        provider=provider, memory=SQLiteMemoryStore(":memory:"), tools=registry
    )

    reply, _, sources = jarvis.handle_chat("What is the capital of Japan?")

    assert reply == "The capital of Japan is Tokyo."
    assert sources == []


def test_an_unrelated_registered_tool_is_unaffected_by_computer_tools():
    from app.tools.base import PermissionLevel as PL
    from app.tools.base import Tool, ToolResult

    class PingTool(Tool):
        name = "ping"
        description = "ping"
        permission = PL.READ

        def execute(self):
            return ToolResult.success(message="pong")

    registry = create_computer_tools_registry(backend=FakeComputerBackend())
    registry.register(PingTool())
    jarvis, _ = make_jarvis(registry=registry)

    reply, _, _ = jarvis.handle_chat("use tool ping")

    assert "pong" in reply


# --- 7. Denied capabilities prevent execution (covered above too) --------


def test_denied_confirm_tier_tool_leaves_no_trace_of_execution():
    backend = FakeComputerBackend()
    registry = ToolRegistry()
    registry.register(_ConfirmTierPressKeyTool(backend=backend))

    registry.execute("press_key_confirm_tier", params={"key": "enter"}, confirmed=False)

    assert backend.pressed_keys == []
    assert backend.active_window_reads == 0


# --- 8. Computer tools retain all M17.1-A input validation ---------------


def test_open_application_still_rejects_a_non_allowlisted_app_through_chat():
    jarvis, backend = make_jarvis()

    reply, _, _ = jarvis.handle_chat("use tool open_application cmd")

    assert backend.opened_applications == []
    assert "not an allowed application" in reply.lower() or "cmd" in reply


def test_open_url_still_rejects_a_non_http_scheme_through_chat():
    jarvis, backend = make_jarvis()

    jarvis.handle_chat("use tool open_url file:///C:/secrets.txt")

    assert backend.opened_urls == []


def test_press_key_still_rejects_a_disallowed_key_through_chat():
    jarvis, backend = make_jarvis()

    jarvis.handle_chat("use tool press_key rm_rf")

    assert backend.pressed_keys == []


def test_registry_still_enforces_permission_for_a_directly_executed_computer_tool():
    """Direct ToolRegistry.execute (bypassing chat entirely) still goes
    through the exact same permission/validation machinery - proving
    computer tools don't rely on JarvisCore for their safety, only on
    ToolRegistry itself."""
    backend = FakeComputerBackend()
    registry = create_computer_tools_registry(backend=backend)

    result = registry.execute("open_application", params={"app": "cmd"})

    assert result.status == ToolResultStatus.ERROR
    assert backend.opened_applications == []


# --- 9. No real OS interaction (structural check) --------------------------


def test_register_computer_tools_accepts_an_injected_fake_backend():
    """register_computer_tools never hardcodes WindowsComputerBackend -
    every test in this file (and test_computer_tools.py) can run without
    ever constructing a real backend."""
    backend = FakeComputerBackend()
    registry = ToolRegistry()

    register_computer_tools(registry, backend=backend)

    for tool in registry.list_tools():
        assert tool.backend is backend
