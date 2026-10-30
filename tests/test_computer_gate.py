"""
Computer Input Gate (app/computer/gate.py), the SWITCH and LIMIT half: the seven INPUT methods - type_text, press_key, move_mouse, click_mouse, double_click,
scroll, open_application - may reach the desktop only when `computer_input_enabled` is on (default OFF) and the one shared limit (default 10 admitted actions in any
rolling 60 seconds) has room. (What a grant covers is tested in test_computer_authorization.py and test_computer_gate_no_bypass.py. There is no budget-only gate any more: the
tests HERE keep asserting exactly the two rules they were written for, but every input call goes through the real gate with a real one-entry grant that tests/grant_support.py
mints for exactly that call.) The gate is enforced at EXECUTION time, inside the backend every tool ends in, so it cannot be skipped by a direct `execute`, a registry call, or
a chained "use tool ..., then ..." command; the tools' `validate_arguments` only PRE-checks (consuming nothing) to turn a whole command away early.

Real: the gate, the wrapper, the tools, ToolRegistry, JarvisCore with its explicit-command planner, and (for the wiring tests) the production objects in
app/main.py. Faked: ONLY the desktop - a recording backend. Every test here also replaces the real `pyautogui`, `subprocess` and `webbrowser` inside the
Windows backend with objects that raise, so a bug in this slice's wiring could never move a real mouse or press a real key during testing.
"""

import ast
import logging
import pathlib
import re
import threading
import time
from collections import deque

import pytest

import app.computer.windows_backend as windows_backend
from app.computer.base import (
    AllowedApplication,
    ComputerBackend,
    ComputerControlError,
    MouseButton,
    ScreenInfo,
    WindowInfo,
)
from app.computer.defaults import JOB_COMPUTER_TOOL_NAMES, create_computer_tools_registry, register_computer_tools
from app.computer.gate import (
    DEFAULT_ACTIONS_PER_WINDOW,
    GATED_METHODS,
    SWITCHED_OFF,
    UNGATED_METHODS,
    WINDOW_SECONDS,
    GatedComputerBackend,
    InputGate,
    create_default_input_gate,
    default_input_gate,
)
from app.computer.windows_backend import WindowsComputerBackend
from app.config import Settings, settings
from app.core import JarvisCore
from app.memory.sqlite_store import SQLiteMemoryStore
from app.providers.base import AIProvider
from app.tools.base import PermissionLevel, ToolArgumentError, ToolResultStatus
from app.tools.decision import ToolCallBuilder, ToolCallError, ToolCallProvider
from app.tools.registry import ToolRegistry
from grant_support import PerCallGrants, authorize, budget_trail, check_now, admit_now, without_chat_trace_noise, without_grant_ids

INPUT_TOOLS = ("type_text", "press_key", "move_mouse", "click_mouse", "double_click", "open_application", "scroll")
#: Every method that ACTS on the desktop: the input tools plus the two that change which window is in front (each needs a grant that binds its exact arguments).
ACTING_METHODS = (*INPUT_TOOLS, "open_url", "switch_window")

#: (tool name, the parameters of a valid call, the single effect it must have on the desktop)
INPUT_CALLS = [
    ("type_text", {"text": "hello"}, ("type_text", "hello")),
    ("press_key", {"key": "enter"}, ("press_key", "enter")),
    ("move_mouse", {"x": 10, "y": 20}, ("move_mouse", 10, 20)),
    ("click_mouse", {"x": 10, "y": 20}, ("click_mouse", 10, 20, MouseButton.LEFT)),
    ("double_click", {"x": 10, "y": 20}, ("double_click", 10, 20, MouseButton.LEFT)),
    ("open_application", {"app": "notepad"}, ("open_application", AllowedApplication.NOTEPAD)),
    ("scroll", {"amount": 5}, ("scroll", 5)),
]
IDS = [c[0] for c in INPUT_CALLS]


# ===================================================== doubles ==========================================================================


class _NoDesktop:
    def __getattr__(self, name):
        raise AssertionError(f"a test reached the REAL desktop: {name}")


@pytest.fixture(autouse=True)
def no_real_desktop(monkeypatch):
    """Inside the real backend module, the three things that act on the machine become tripwires, for every test in this file."""
    for attribute in ("pyautogui", "subprocess", "webbrowser"):
        monkeypatch.setattr(windows_backend, attribute, _NoDesktop())


class RecordingBackend(ComputerBackend):
    """The fake desktop: `effects` is everything that DID something, in order; `reads` is everything that only looked."""

    def __init__(self, error: Exception | None = None):
        self.error = error
        self.effects: list[tuple] = []
        self.reads: list[str] = []

    def _act(self, *call):
        if self.error is not None:
            raise self.error
        self.effects.append(call)

    def get_active_window(self):
        self.reads.append("get_active_window")
        return WindowInfo(title="Notepad", process_name="notepad.exe", is_maximized=False)

    def get_screen_info(self):
        self.reads.append("get_screen_info")
        return ScreenInfo(width=1920, height=1080)

    def open_application(self, app):
        self._act("open_application", app)

    def open_url(self, url):
        self._act("open_url", url)

    def switch_window(self, title, process):
        self._act("switch_window", title, process)
        return WindowInfo(title=title, process_name=process, is_maximized=False)

    def type_text(self, text):
        self._act("type_text", text)

    def press_key(self, key):
        self._act("press_key", key)

    def move_mouse(self, x, y):
        self._act("move_mouse", x, y)

    def click_mouse(self, x, y, button):
        self._act("click_mouse", x, y, button)

    def double_click(self, x, y, button):
        self._act("double_click", x, y, button)

    def scroll(self, amount):
        self._act("scroll", amount)


class Clock:
    def __init__(self, now: float = 1000.0):
        self.now = now

    def __call__(self) -> float:
        return self.now


class SlowCountDeque(deque):
    """Reports its length only after a pause. An unlocked `if len(events) >= limit: refuse; else events.append(now)` then acts on a count that is already
    stale - several threads all see room, all record - which is exactly the race the gate's lock exists to prevent. With the lock it only costs time."""

    def __len__(self):
        size = super().__len__()
        time.sleep(0.0003)
        return size


def make_gated(limit=10, enabled=True, clock=None, inner=None):
    """(gated backend, the recording backend behind it, a dict that changes the settings live, the clock).

    These tests are about the switch and the limit. Every input call still goes through the REAL gate with a REAL grant: `PerCallGrants` mints a one-entry grant for exactly
    the call being made (tests/grant_support.py) and hands it to the production `GatedComputerBackend`."""
    clock = clock or Clock()
    state = {"enabled": enabled, "limit": limit}
    gate = InputGate(lambda: state["enabled"], lambda: state["limit"], clock=clock)
    inner = inner or RecordingBackend()
    return PerCallGrants(lambda grant: GatedComputerBackend(inner, gate, grant), gate), inner, state, clock


BACKEND_CALLS = {
    "type_text": lambda b: b.type_text("hello"),
    "press_key": lambda b: b.press_key("enter"),
    "move_mouse": lambda b: b.move_mouse(10, 20),
    "click_mouse": lambda b: b.click_mouse(10, 20, MouseButton.LEFT),
    "double_click": lambda b: b.double_click(10, 20, MouseButton.LEFT),
    "open_application": lambda b: b.open_application(AllowedApplication.NOTEPAD),
    "scroll": lambda b: b.scroll(5),
}
#: The exact (normalized) arguments each BACKEND_CALLS call passes to the gate, for a test that mints its grants ahead of time.
CALL_ARGS = {
    "type_text": ("hello",),
    "press_key": (("enter",),),
    "move_mouse": (10, 20),
    "click_mouse": (10, 20, MouseButton.LEFT),
    "double_click": (10, 20, MouseButton.LEFT),
    "open_application": (AllowedApplication.NOTEPAD,),
    "scroll": (5,),
}


# ===================================================== the defaults ======================================================================


def test_the_declared_defaults_are_off_and_ten_per_rolling_minute():
    fields = Settings.model_fields

    assert fields["computer_input_enabled"].default is False  # OFF unless the owner turns it on
    assert fields["computer_input_actions_per_minute"].default == 10
    assert DEFAULT_ACTIONS_PER_WINDOW == 10 and WINDOW_SECONDS == 60.0


def test_the_default_gate_requires_authority_and_is_built_from_the_settings(monkeypatch):
    monkeypatch.setattr(settings, "computer_input_enabled", True)  # the switch alone is not enough, on a gate built from the settings
    monkeypatch.setattr(settings, "computer_input_actions_per_minute", 10**6)
    for gate in (create_default_input_gate(), default_input_gate()):
        assert not hasattr(gate, "_require_authority")  # fail closed always: there is no mode in which a grant is optional
        with pytest.raises(ComputerControlError, match="approved authorization"):
            gate.admit("type_text", authority=None, canonical=None)


def test_the_default_gate_reads_both_settings_on_every_call(monkeypatch):
    gate = create_default_input_gate()  # (each admission below carries a real one-entry grant; this test is about the two SETTINGS)
    monkeypatch.setattr(settings, "computer_input_enabled", False)
    monkeypatch.setattr(settings, "computer_input_actions_per_minute", 2)
    with pytest.raises(ComputerControlError, match="switched off"):
        admit_now(gate, "type_text")

    monkeypatch.setattr(settings, "computer_input_enabled", True)  # the SAME gate object, no rebuild
    admit_now(gate, "type_text")
    admit_now(gate, "type_text")
    with pytest.raises(ComputerControlError, match="limited to 2 actions per minute"):
        admit_now(gate, "type_text")

    monkeypatch.setattr(settings, "computer_input_actions_per_minute", 3)  # raised live: the third now fits
    admit_now(gate, "type_text")
    monkeypatch.setattr(settings, "computer_input_enabled", False)  # switched off live: nothing more, even with room
    with pytest.raises(ComputerControlError, match="switched off"):
        admit_now(gate, "type_text")


# ===================================================== the gate: switch, limit, boundary =====================================================


def test_a_switched_off_gate_refuses_everything_and_uses_up_nothing():
    backend, inner, state, _ = make_gated(limit=10, enabled=False)

    for _ in range(100):
        with pytest.raises(ComputerControlError, match="switched off"):
            backend.type_text("x")
        with pytest.raises(ComputerControlError, match="switched off"):
            check_now(backend.gate, "type_text")
    state["enabled"] = True
    for _ in range(10):
        backend.type_text("x")  # all ten slots are still free: not one refusal was counted

    assert len(inner.effects) == 10
    with pytest.raises(ComputerControlError, match="limited to 10"):
        backend.type_text("x")


def test_ten_admitted_in_the_same_instant_then_the_eleventh_is_refused_until_exactly_sixty_seconds_later():
    backend, inner, _, clock = make_gated(limit=10)
    for _ in range(10):
        backend.press_key("enter")

    for later in (0.0, 1.0, 30.0, 59.0, 59.999999):
        clock.now = 1000.0 + later
        with pytest.raises(ComputerControlError, match="limited to 10"):
            backend.press_key("enter")
    assert len(inner.effects) == 10

    clock.now = 1000.0 + 60.0  # age == 60.000: the ten have expired
    for _ in range(10):
        backend.press_key("enter")
    with pytest.raises(ComputerControlError):
        backend.press_key("enter")
    assert len(inner.effects) == 20


def test_the_rolling_window_boundary_is_exact_at_every_edge():
    """ONE story, each row following the previous: a limit of 3 and an attempt at each time shown (seconds after the start)."""
    table = [
        (0.0, True), (10.0, True), (20.0, True),            # the window fills: admitted at t = 0, 10, 20
        (20.0, False), (59.0, False), (59.999, False),       # full - and a refusal uses up nothing
        (60.0, True),                                         # the t=0 action is exactly 60 s old: its slot is free
        (60.0, False),                                        # ...and that one slot is all that freed
        (69.999, False),                                      # the t=10 action is 59.999 s old: it still counts
        (70.0, True),                                         # exactly 60 s: free
        (79.999, False), (80.0, True),                        # the t=20 action, the same way
        (80.0, False),
    ]
    backend, _, _, clock = make_gated(limit=3)

    for at, expected in table:
        clock.now = 1000.0 + at
        try:
            backend.press_key("enter")
            actual = True
        except ComputerControlError:
            actual = False
        assert actual is expected, f"at t={at}: expected {'admitted' if expected else 'refused'}"


def test_a_refused_attempt_never_extends_the_window():
    backend, _, _, clock = make_gated(limit=1)
    backend.type_text("x")  # at t=0
    for second in range(1, 60):
        clock.now = 1000.0 + second
        with pytest.raises(ComputerControlError):
            backend.type_text("x")  # fifty-nine refusals

    clock.now = 1000.0 + 60.0

    backend.type_text("x")  # the slot frees at 60 s after the ONE admitted action, not 60 s after the last refusal


def test_one_budget_is_shared_by_all_six_tools_not_one_each():
    for spent_on, spend in BACKEND_CALLS.items():
        backend, inner, _, _ = make_gated(limit=3)
        for _ in range(3):
            spend(backend)  # the whole budget goes on ONE tool...

        for other, call in BACKEND_CALLS.items():  # ...so every tool, that one included, is refused
            with pytest.raises(ComputerControlError, match="limited to 3"):
                call(backend)
        assert len(inner.effects) == 3, spent_on


def test_a_mixed_run_of_all_seven_uses_one_count():
    backend, inner, _, _ = make_gated(limit=7)
    for call in BACKEND_CALLS.values():
        call(backend)

    with pytest.raises(ComputerControlError):
        backend.type_text("eighth")

    assert [e[0] for e in inner.effects] == list(BACKEND_CALLS)


def test_an_attempt_that_passed_the_gate_counts_even_if_the_desktop_then_failed():
    inner = RecordingBackend(error=ComputerControlError("the app is not installed"))
    backend, _, _, _ = make_gated(limit=1, inner=inner)

    with pytest.raises(ComputerControlError, match="not installed"):
        backend.open_application(AllowedApplication.NOTEPAD)

    inner.error = None
    with pytest.raises(ComputerControlError, match="limited to 1"):
        backend.open_application(AllowedApplication.NOTEPAD)  # the failed attempt still used the slot: a failing loop cannot spin


def test_the_retry_hint_counts_down_to_the_free_slot():
    backend, _, _, clock = make_gated(limit=1)
    backend.type_text("x")

    clock.now = 1000.0 + 20.0
    with pytest.raises(ComputerControlError, match=r"Try again in 40 second\(s\)"):
        backend.type_text("x")
    clock.now = 1000.0 + 59.5
    with pytest.raises(ComputerControlError, match=r"Try again in 1 second\(s\)"):
        backend.type_text("x")


def test_check_consumes_nothing_but_refuses_when_off_or_full():
    backend, inner, state, _ = make_gated(limit=2)
    for _ in range(500):
        check_now(backend.gate, "type_text")  # a pre-check can never be what limits anyone

    backend.type_text("x")
    backend.type_text("x")
    with pytest.raises(ComputerControlError, match="limited to 2"):
        check_now(backend.gate, "type_text")
    state["enabled"] = False
    with pytest.raises(ComputerControlError, match="switched off"):
        check_now(backend.gate, "type_text")

    assert len(inner.effects) == 2


def test_the_limit_can_be_changed_while_running_and_a_lowered_limit_holds_until_the_window_drains():
    backend, inner, state, clock = make_gated(limit=5)
    for _ in range(4):
        backend.type_text("x")

    state["limit"] = 2  # lowered below what is already in the window
    with pytest.raises(ComputerControlError, match="limited to 2"):
        backend.type_text("x")
    clock.now = 1000.0 + 60.0  # the four have expired
    backend.type_text("x")
    backend.type_text("x")
    with pytest.raises(ComputerControlError):
        backend.type_text("x")
    state["limit"] = 3  # raised: one more fits at once
    backend.type_text("x")

    assert len(inner.effects) == 7


@pytest.mark.parametrize("bad", [None, -1, -(2**40), 10.0, "10", True, False, []])
def test_a_limit_that_is_not_a_plain_non_negative_whole_number_allows_nothing(bad):
    backend, inner, _, _ = make_gated(limit=bad)

    with pytest.raises(ComputerControlError):
        backend.type_text("x")

    assert inner.effects == []


@pytest.mark.parametrize("bad", [1, "yes", "true", None, 0, [], object()])
def test_a_switch_that_is_not_exactly_true_is_off(bad):
    backend, inner, _, _ = make_gated(enabled=bad)

    with pytest.raises(ComputerControlError, match="switched off"):
        backend.type_text("x")

    assert inner.effects == []


def test_a_setting_that_cannot_be_read_fails_closed():
    def boom():
        raise RuntimeError("config unavailable")

    for gate in (InputGate(boom, lambda: 10), InputGate(lambda: True, boom)):
        backend = GatedComputerBackend(RecordingBackend(), gate)
        with pytest.raises(ComputerControlError):
            backend.type_text("x")
        assert backend.inner.effects == []


# ===================================================== concurrency ==============================================================================


@pytest.mark.parametrize("round_number", range(10))
def test_concurrent_calls_admit_exactly_the_limit_and_no_more(round_number):
    threads_count, limit = 64, 10
    gate = InputGate(lambda: True, lambda: limit, clock=Clock(), max_grants=threads_count + 8)  # a clock that never moves: nothing expires mid-test
    gate._events = SlowCountDeque()  # see the class: makes a missing lock show up every time instead of once in a blue moon
    inner = RecordingBackend()
    # One real grant per thread, minted BEFORE the race (the race is on the gate's lock, not on the issuer): every thread holds exactly the entry for its own call.
    backends = [GatedComputerBackend(inner, gate, authorize(gate, (tool, *CALL_ARGS[tool]))) for tool in (INPUT_TOOLS[n % len(INPUT_TOOLS)] for n in range(threads_count))]
    barrier = threading.Barrier(threads_count)
    results = []

    def attempt(n):
        barrier.wait()
        try:
            BACKEND_CALLS[INPUT_TOOLS[n % len(INPUT_TOOLS)]](backends[n])  # all seven tools at once, sharing one budget
            results.append(True)
        except ComputerControlError:
            results.append(False)

    threads = [threading.Thread(target=attempt, args=(n,)) for n in range(threads_count)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert results.count(True) == limit and results.count(False) == threads_count - limit
    assert len(inner.effects) == limit  # and exactly that many reached the desktop


# ===================================================== the wrapper: every method classified, gated ones gated =====================================


def test_every_method_of_the_backend_interface_is_classified_exactly_once():
    interface = set(ComputerBackend.__abstractmethods__) | {"switch_window"}

    assert GATED_METHODS | UNGATED_METHODS == interface  # a method added later cannot reach the desktop unclassified
    assert GATED_METHODS & UNGATED_METHODS == set()
    assert GATED_METHODS == set(ACTING_METHODS)
    assert UNGATED_METHODS == {"get_active_window", "get_screen_info"}  # open_url and switch_window moved to GATED_METHODS: each changes which window is in front


@pytest.mark.parametrize("tool, params, effect", INPUT_CALLS, ids=IDS)
def test_each_gated_method_is_admitted_first_and_then_passed_on_unchanged(tool, params, effect):
    order = []
    backend, inner, _, _ = make_gated()
    real_admit = backend.gate.admit
    backend.gate.admit = lambda action, **kwargs: (order.append(("admit", action)), real_admit(action, **kwargs))[1]

    BACKEND_CALLS[tool](backend)

    assert order == [("admit", tool)] and inner.effects == [effect]


@pytest.mark.parametrize("tool, params, effect", INPUT_CALLS, ids=IDS)
def test_each_gated_method_reaches_nothing_when_refused(tool, params, effect):
    backend, inner, _, _ = make_gated(enabled=False)

    with pytest.raises(ComputerControlError):
        BACKEND_CALLS[tool](backend)

    assert inner.effects == []


def test_only_the_reads_pass_straight_through_even_when_input_is_off():
    backend, inner, _, _ = make_gated(enabled=False, limit=0)

    assert backend.get_active_window().title == "Notepad" and backend.get_screen_info().width == 1920
    with pytest.raises(ComputerControlError, match="switched off"):
        backend.open_url("https://example.com")  # open_url and switch_window change which window is in front: gated like the rest
    with pytest.raises(ComputerControlError, match="switched off"):
        backend.switch_window("Notes", "notes.exe")

    assert inner.effects == [] and inner.reads == ["get_active_window", "get_screen_info"]


def test_the_reads_use_none_of_the_input_budget():
    backend, inner, _, _ = make_gated(limit=1)
    for _ in range(25):
        backend.get_active_window()
        backend.get_screen_info()

    backend.type_text("the one slot is still free")  # 50 pass-through reads later
    with pytest.raises(ComputerControlError, match="limited to 1"):
        backend.type_text("and now it is used")


def test_open_url_and_switch_window_share_the_one_budget_with_the_other_seven():
    backend, inner, _, _ = make_gated(limit=2)

    backend.open_url("https://example.com")
    backend.switch_window("Notes", "notes.exe")
    for refused in (lambda: backend.type_text("x"), lambda: backend.open_url("https://example.com"), lambda: backend.switch_window("Notes", "notes.exe")):
        with pytest.raises(ComputerControlError, match="limited to 2"):
            refused()

    assert inner.effects == [("open_url", "https://example.com"), ("switch_window", "Notes", "notes.exe")]


# ===================================================== the tools: direct calls cannot bypass the gate ============================================


def tools_over(backend):
    return create_computer_tools_registry(backend=backend)


@pytest.mark.parametrize("tool, params, effect", INPUT_CALLS, ids=IDS)
def test_a_registry_call_is_refused_when_input_is_off(tool, params, effect):
    backend, inner, _, _ = make_gated(enabled=False)

    result = tools_over(backend).execute(tool, params=params)

    assert result.status is ToolResultStatus.ERROR and result.message == SWITCHED_OFF
    assert inner.effects == []


@pytest.mark.parametrize("tool, params, effect", INPUT_CALLS, ids=IDS)
def test_a_registry_call_reaches_the_desktop_once_when_input_is_on(tool, params, effect):
    backend, inner, _, _ = make_gated()

    result = tools_over(backend).execute(tool, params=params)

    assert result.status is ToolResultStatus.SUCCESS and inner.effects == [effect]


@pytest.mark.parametrize("tool, params, effect", INPUT_CALLS, ids=IDS)
def test_calling_the_tool_object_directly_is_gated_too(tool, params, effect):
    backend, inner, state, _ = make_gated(enabled=False)
    instance = tools_over(backend).get(tool)

    off = instance.execute(**params)  # not through the registry at all
    state["enabled"] = True
    on = instance.execute(**params)

    assert off.status is ToolResultStatus.ERROR and on.status is ToolResultStatus.SUCCESS and inner.effects == [effect]


@pytest.mark.parametrize("tool, params, effect", INPUT_CALLS, ids=IDS)
def test_each_tools_validation_pre_checks_the_gate_on_its_own_and_consumes_nothing(tool, params, effect):
    backend, inner, state, _ = make_gated(limit=1)
    instance = tools_over(backend).get(tool)

    for _ in range(50):
        instance.validate_arguments(**params)  # on, with room: valid, and a pre-check never uses the slot
    state["enabled"] = False
    with pytest.raises(ToolArgumentError, match="switched off"):
        instance.validate_arguments(**params)
    state["enabled"] = True
    instance.execute(**params)  # the single slot is still there...
    with pytest.raises(ToolArgumentError, match="limited to 1"):
        instance.validate_arguments(**params)  # ...and now a pre-check sees it is spent

    assert inner.effects == [effect]


def test_the_limit_holds_at_execution_even_if_validation_is_skipped_or_stale(monkeypatch):
    backend, inner, _, _ = make_gated(limit=2)
    registry = tools_over(backend)
    for name in INPUT_TOOLS:  # a caller that never validates, or a pre-check that is out of date
        monkeypatch.setattr(type(registry.get(name)), "validate_arguments", lambda self, **kw: None)
    monkeypatch.setattr(backend.gate, "check", lambda action, **kwargs: None)

    results = [registry.execute(n, params=p) for n, p, _ in INPUT_CALLS[:5]]

    assert [r.status for r in results] == [ToolResultStatus.SUCCESS, ToolResultStatus.SUCCESS] + [ToolResultStatus.ERROR] * 3
    assert all("limited to 2" in r.message for r in results[2:])
    assert len(inner.effects) == 2


def test_eleven_registry_calls_of_mixed_tools_reach_the_desktop_ten_times_and_the_window_slides():
    backend, inner, _, clock = make_gated(limit=10)
    registry = tools_over(backend)
    calls = [INPUT_CALLS[n % 6] for n in range(11)]

    outcomes = [registry.execute(name, params=params).status for name, params, _ in calls]

    assert outcomes == [ToolResultStatus.SUCCESS] * 10 + [ToolResultStatus.ERROR]
    assert len(inner.effects) == 10
    clock.now += 59.999
    assert registry.execute("press_key", params={"key": "enter"}).status is ToolResultStatus.ERROR
    clock.now += 0.001  # exactly 60.000 after the first ten
    assert registry.execute("press_key", params={"key": "enter"}).status is ToolResultStatus.SUCCESS


def test_validation_failures_use_no_slot():
    backend, inner, _, _ = make_gated(limit=2)
    registry = tools_over(backend)
    refused = [
        registry.execute("press_key", params={"key": "a"}),  # not an allowed key
        registry.execute("click_mouse", params={"x": 99999, "y": 5}),  # off screen
        registry.execute("type_text", params={"text": "x" * 1001}),  # too long
        registry.execute("open_application", params={"app": "cmd"}),  # not an allowed app
    ]

    assert all(r.status is ToolResultStatus.ERROR for r in refused) and inner.effects == []
    assert registry.execute("press_key", params={"key": "enter"}).status is ToolResultStatus.SUCCESS
    assert registry.execute("press_key", params={"key": "tab"}).status is ToolResultStatus.SUCCESS  # both slots were still free


def test_register_computer_tools_with_no_backend_gets_the_self_gating_default():
    registry = ToolRegistry()
    register_computer_tools(registry)  # constructing the real backend has no side effect; nothing is called on it

    for name in INPUT_TOOLS:
        backend = registry.get(name).backend
        assert isinstance(backend, WindowsComputerBackend) and backend.gate is default_input_gate()


def test_the_input_tools_keep_their_permission_but_are_never_automatable_and_the_read_tools_are_unchanged():
    registry = tools_over(make_gated()[0])

    for name in INPUT_TOOLS:
        tool = registry.get(name)
        assert tool.permission is PermissionLevel.SAFE_ACTION  # the control is the gate, not a confirmation
        assert tool.automatable is False
    assert registry.get("get_screen_info").permission is PermissionLevel.READ and registry.get("get_screen_info").automatable is True
    for name in JOB_COMPUTER_TOOL_NAMES:
        assert registry.get(name).automatable is False  # as before this slice


# ===================================================== chained commands, through the real JarvisCore ==============================================


class _NoAI(AIProvider):
    def generate_reply(self, message):
        raise AssertionError("no AI may be used for an explicit tool command")


class _NoToolAI(ToolCallProvider):
    consulted = 0

    def decide(self, message, candidate_tools):
        type(self).consulted += 1
        raise ToolCallError("an explicit command must never be handed to the AI")


def chat_over(limit=10, enabled=True):
    backend, inner, state, clock = make_gated(limit=limit, enabled=enabled)
    jarvis = JarvisCore(
        provider=_NoAI(), memory=SQLiteMemoryStore(":memory:"), tools=ToolRegistry(),
        computer_tools=create_computer_tools_registry(backend=backend), tool_call_builder=ToolCallBuilder(_NoToolAI()),
    )
    return (lambda message: jarvis.handle_chat(message)[0]), inner, state, clock


CHAIN = "use tool press_key win+r, then use tool type_text notepad, then use tool press_key enter"


def test_a_chained_command_is_turned_away_whole_when_input_is_off():
    say, inner, _, _ = chat_over(enabled=False)

    reply = say(CHAIN)

    assert "switched off" in reply and "nothing was executed" in reply.lower()
    assert inner.effects == []


def test_a_chain_that_mixes_in_an_allowed_tool_is_still_turned_away_before_its_first_step():
    say, inner, _, _ = chat_over(enabled=False)

    reply = say("use tool scroll -600, then use tool press_key enter")

    assert "nothing was executed" in reply.lower()
    assert inner.effects == []  # not even the scroll ran: the pre-check rejects the whole command


def test_the_reads_still_work_while_input_is_off_but_scrolling_and_opening_a_url_do_not():
    say, inner, _, _ = chat_over(enabled=False)

    assert "switched off" in say("use tool scroll -600")  # scrolling is mouse-wheel input: gated
    assert "switched off" in say("use tool open_url https://example.com")  # and so is opening the browser: it changes which window is in front
    say("use tool get_active_window")

    assert inner.effects == []  # nothing acted; only the read ran (open_url and switch_window are no longer outside the gate)


def test_a_chain_longer_than_the_remaining_budget_runs_only_what_fits_and_says_so():
    say, inner, _, _ = chat_over(limit=2)

    reply = say(CHAIN)  # three input steps, two slots: the pre-check passes (it consumes nothing); the third is refused when it runs

    assert inner.effects == [("press_key", "win+r"), ("type_text", "notepad")]  # 'enter' never reached the desktop
    assert reply.startswith("1. Pressed 'win+r'.\n2. Text typed.\n3. ") and "limited to 2 actions per minute" in reply


def test_a_longer_chain_stops_at_the_refusal_and_reports_the_steps_not_run():
    say, inner, _, _ = chat_over(limit=2)

    reply = say("use tool press_key win+r, then use tool type_text a, then use tool press_key enter, then use tool press_key tab")

    assert len(inner.effects) == 2
    assert "Stopped after step 3; step(s) 4 were not run." in reply


def test_the_budget_is_shared_across_messages_and_a_full_window_turns_the_next_command_away_early():
    say, inner, _, clock = chat_over(limit=2)
    say("use tool press_key enter, then use tool press_key tab")

    reply = say("use tool type_text hello")

    assert "limited to 2" in reply and "nothing was executed" in reply.lower() and len(inner.effects) == 2
    clock.now += 60.0
    assert say("use tool type_text hello") == "Text typed."
    assert len(inner.effects) == 3


def test_ten_single_step_messages_pass_the_eleventh_does_not_and_sixty_seconds_later_one_does():
    say, inner, _, clock = chat_over(limit=10)
    for _ in range(10):
        assert say("use tool press_key enter") == "Pressed 'enter'."

    assert "limited to 10" in say("use tool press_key enter")
    clock.now += 59.9
    assert "limited to 10" in say("use tool press_key enter")
    clock.now += 0.1
    assert say("use tool press_key enter") == "Pressed 'enter'."
    assert len(inner.effects) == 11


def test_a_command_the_validators_refuse_uses_no_slot_and_no_ai_is_ever_consulted():
    say, inner, _, _ = chat_over(limit=1)

    say("use tool press_key a")
    say("use tool open_application cmd")
    ok = say("use tool press_key enter")

    assert ok == "Pressed 'enter'." and len(inner.effects) == 1 and _NoToolAI.consulted == 0


# ===================================================== the REAL backend enforces the gate itself ==================================================


class RecordingOS:
    """Stands in for pyautogui, subprocess and webbrowser INSIDE the real WindowsComputerBackend module. The real backend's own code runs; this only records
    what it finally asks the OS to do, and (if `fail` is set) fails the way a real OS call can."""

    FAILSAFE = True

    def __init__(self):
        self.calls: list[tuple] = []
        self.fail: Exception | None = None

    def _record(self, *call):
        if self.fail is not None:
            raise self.fail
        self.calls.append(call)

    def size(self):
        return (1920, 1080)

    def write(self, text, interval=0.0):  # the real backend now paces typing (windows_backend.TYPING_INTERVAL_SECONDS); what is recorded is still just the text
        self._record("write", text)

    def press(self, key):
        self._record("press", key)

    def hotkey(self, *keys):
        self._record("hotkey", keys)

    def moveTo(self, x, y):
        self._record("moveTo", x, y)

    def click(self, **kwargs):
        self._record("click", kwargs)

    def doubleClick(self, **kwargs):
        self._record("doubleClick", kwargs)

    def scroll(self, amount):
        self._record("scroll", amount)

    def Popen(self, command, shell=False):
        self._record("Popen", command, shell)

    def open(self, url):
        self._record("browser", url)
        return True


#: What each valid INPUT_CALLS call finally asks the OS to do, as the real backend sees it.
OS_EFFECT = {
    "type_text": ("write", "hello"),
    "press_key": ("press", "enter"),
    "move_mouse": ("moveTo", 10, 20),
    "click_mouse": ("click", {"x": 10, "y": 20, "button": "left"}),
    "double_click": ("doubleClick", {"x": 10, "y": 20, "button": "left"}),
    "open_application": ("Popen", ["notepad.exe"], False),
}


@pytest.fixture
def recording_os(monkeypatch):
    recorder = RecordingOS()
    for attribute in ("pyautogui", "subprocess", "webbrowser"):
        monkeypatch.setattr(windows_backend, attribute, recorder)  # replaces the autouse tripwire for this test
    return recorder


@pytest.fixture
def process_gate(monkeypatch):
    """The process-wide gate with its history cleared for this test (and restored afterwards), and both settings under the test's control."""
    gate = default_input_gate()
    monkeypatch.setattr(gate, "_events", deque())
    monkeypatch.setattr(gate, "_grants", {})  # and no grant minted by an earlier test, or by this one, outlives it
    monkeypatch.setattr(settings, "computer_input_enabled", False)
    monkeypatch.setattr(settings, "computer_input_actions_per_minute", 10)
    return gate


def authorized_real_backend(gate=None):
    """The REAL `WindowsComputerBackend` over the process-wide gate (or `gate`), given a real one-entry grant for each call it is asked to make."""
    gate = gate if gate is not None else default_input_gate()
    return PerCallGrants(lambda grant: WindowsComputerBackend(gate=gate, authority=grant), gate)


@pytest.mark.parametrize("tool", INPUT_TOOLS)
def test_a_bare_windows_backend_is_switched_off_and_nothing_reaches_the_os(tool, recording_os, process_gate):
    bare = WindowsComputerBackend()  # no factory, no wrapper, no registry, no tool: exactly what a careless future caller would write

    with pytest.raises(ComputerControlError, match="switched off"):
        BACKEND_CALLS[tool](bare)

    assert recording_os.calls == []


def test_the_real_backends_click_at_the_current_position_is_gated_too(recording_os, process_gate, monkeypatch):
    """`click_mouse(None, None, ...)` is a separate branch (it clicks wherever the cursor is): it has its own gate call, and its own test."""
    bare = authorized_real_backend()

    with pytest.raises(ComputerControlError, match="switched off"):
        bare.click_mouse(None, None, MouseButton.LEFT)
    assert recording_os.calls == []

    monkeypatch.setattr(settings, "computer_input_enabled", True)
    with pytest.raises(ComputerControlError, match="approved authorization"):
        WindowsComputerBackend().click_mouse(None, None, MouseButton.LEFT)  # on, but this instance carries no grant
    assert recording_os.calls == []
    bare.click_mouse(None, None, MouseButton.LEFT)
    assert recording_os.calls == [("click", {"button": "left"})]


@pytest.mark.parametrize("tool, params, effect", INPUT_CALLS, ids=IDS)
def test_each_tools_pre_check_works_on_the_real_backend_too_and_consumes_nothing(tool, params, effect, recording_os, process_gate, monkeypatch):
    instance = create_computer_tools_registry().get(tool)  # the production default backend: the real one

    with pytest.raises(ToolArgumentError, match="switched off"):
        instance.validate_arguments(**params)
    monkeypatch.setattr(settings, "computer_input_enabled", True)
    with pytest.raises(ToolArgumentError, match="approved authorization"):
        instance.validate_arguments(**params)  # on, but the production default carries no grant, so even the pre-check says no
    authorized = create_computer_tools_registry(backend=authorized_real_backend()).get(tool)  # the same real backend and gate, handed a grant per call
    for _ in range(25):
        authorized.validate_arguments(**params)  # valid, repeatedly, and it never uses a slot

    assert recording_os.calls == [] and len(process_gate._events) == 0


def test_a_bare_windows_backend_obeys_the_settings_and_every_instance_shares_one_budget(recording_os, process_gate, monkeypatch):
    monkeypatch.setattr(settings, "computer_input_enabled", True)
    monkeypatch.setattr(settings, "computer_input_actions_per_minute", 3)
    first, second = authorized_real_backend(), authorized_real_backend()

    first.type_text("hello")
    second.press_key("enter")
    first.move_mouse(10, 20)
    with pytest.raises(ComputerControlError, match="limited to 3"):
        second.click_mouse(10, 20, MouseButton.LEFT)  # a different instance, a different tool: the same three-action budget
    with pytest.raises(ComputerControlError, match="limited to 3"):
        first.open_application(AllowedApplication.NOTEPAD)

    assert recording_os.calls == [OS_EFFECT["type_text"], OS_EFFECT["press_key"], OS_EFFECT["move_mouse"]]


def test_a_bare_windows_backend_rereads_the_switch_on_every_single_call(recording_os, process_gate, monkeypatch):
    bare = authorized_real_backend()

    for state in (False, True, False, True, True, False):
        monkeypatch.setattr(settings, "computer_input_enabled", state)
        if state:
            bare.press_key("enter")
        else:
            with pytest.raises(ComputerControlError, match="switched off"):
                bare.press_key("enter")

    assert recording_os.calls == [OS_EFFECT["press_key"]] * 3


def test_a_gate_handed_to_the_backend_is_the_only_one_it_uses(recording_os, process_gate):
    own = InputGate(enabled=lambda: True, limit=lambda: 1)
    bare = authorized_real_backend(own)  # the process-wide gate is OFF here, and is never consulted

    bare.type_text("hello")
    with pytest.raises(ComputerControlError, match="limited to 1"):
        bare.type_text("hello")

    assert recording_os.calls == [OS_EFFECT["type_text"]] and len(process_gate._events) == 0


@pytest.mark.parametrize("round_number", range(5))
def test_concurrent_calls_on_the_real_backend_admit_exactly_the_limit_across_instances_and_tools(round_number, recording_os, process_gate, monkeypatch):
    """The production configuration: several real backends, NO gate handed to any of them, so they all share the process-wide gate - hammered by 64 threads
    using all six tools. Exactly `limit` reach the OS, however the threads interleave (the slow count makes a missing lock show up every time)."""
    monkeypatch.setattr(settings, "computer_input_enabled", True)
    monkeypatch.setattr(settings, "computer_input_actions_per_minute", 10)
    monkeypatch.setattr(process_gate, "_events", SlowCountDeque())
    monkeypatch.setattr(process_gate, "_max_grants", 80)  # capacity for the 64 grants minted below, which are all live at once (the cap is not what is under test)
    # 64 real backends (a bare `WindowsComputerBackend(authority=...)` each, none handed a gate), every one holding a real grant minted ahead of the race for its own call.
    instances = [
        WindowsComputerBackend(authority=authorize(process_gate, (tool, *CALL_ARGS[tool])))
        for tool in (INPUT_TOOLS[n % len(INPUT_TOOLS)] for n in range(64))
    ]
    assert all(instance.gate is process_gate for instance in instances)
    barrier = threading.Barrier(64)
    outcomes = []

    def attempt(n):
        barrier.wait()
        try:
            BACKEND_CALLS[INPUT_TOOLS[n % len(INPUT_TOOLS)]](instances[n])
            outcomes.append(True)
        except ComputerControlError:
            outcomes.append(False)

    threads = [threading.Thread(target=attempt, args=(n,)) for n in range(64)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert outcomes.count(True) == 10 and outcomes.count(False) == 54
    assert len(recording_os.calls) == 10


def test_the_real_backends_boundary_is_exact_at_59_999_and_60_seconds(recording_os):
    clock = Clock()
    bare = authorized_real_backend(InputGate(enabled=lambda: True, limit=lambda: 2, clock=clock))
    bare.type_text("hello")  # t = 0
    clock.now += 10.0
    bare.type_text("hello")  # t = 10

    clock.now = 1000.0 + 59.999
    with pytest.raises(ComputerControlError):
        bare.type_text("hello")
    clock.now = 1000.0 + 60.0  # the t=0 action is exactly 60 s old
    bare.type_text("hello")
    with pytest.raises(ComputerControlError):
        bare.type_text("hello")  # that freed ONE slot, no more
    clock.now = 1000.0 + 69.999
    with pytest.raises(ComputerControlError):
        bare.type_text("hello")
    clock.now = 1000.0 + 70.0  # the t=10 action is exactly 60 s old
    bare.type_text("hello")

    assert len(recording_os.calls) == 4


def test_requests_that_fail_validation_use_no_slot_at_the_real_backend(recording_os):
    bare = authorized_real_backend(InputGate(enabled=lambda: True, limit=lambda: 1))
    invalid = [
        lambda: bare.type_text(""), lambda: bare.type_text("x" * 1001), lambda: bare.press_key("a"), lambda: bare.press_key("ctrl+alt"),
        lambda: bare.move_mouse(99999, 1), lambda: bare.click_mouse(99999, 1, MouseButton.LEFT), lambda: bare.click_mouse(10, 20, "left"),
        lambda: bare.double_click(99999, 1, MouseButton.LEFT), lambda: bare.open_application("cmd"),
    ]

    for attempt in invalid:
        with pytest.raises(ComputerControlError):
            attempt()
    assert recording_os.calls == []

    bare.press_key("enter")  # the single slot was never touched
    with pytest.raises(ComputerControlError, match="limited to 1"):
        bare.press_key("tab")


def test_an_action_counts_exactly_once_through_the_default_backend_and_the_tools(recording_os, process_gate, monkeypatch):
    """No double counting: the backend is the real one (not wrapped in the gate wrapper), so four allowed actions use four slots, not eight."""
    monkeypatch.setattr(settings, "computer_input_enabled", True)
    monkeypatch.setattr(settings, "computer_input_actions_per_minute", 4)
    registry = create_computer_tools_registry(backend=authorized_real_backend())  # the real backend and the process-wide gate, given a grant per call

    results = [registry.execute(name, params=params).status for name, params, _ in INPUT_CALLS[:5]]

    assert results == [ToolResultStatus.SUCCESS] * 4 + [ToolResultStatus.ERROR]
    assert recording_os.calls == [OS_EFFECT[name] for name, _, _ in INPUT_CALLS[:4]]


def test_the_wrapper_refuses_to_wrap_a_backend_that_already_enforces_the_gate():
    gate = InputGate(enabled=lambda: True, limit=lambda: 10)

    with pytest.raises(ValueError, match="twice"):
        GatedComputerBackend(WindowsComputerBackend(), gate)
    with pytest.raises(ValueError, match="twice"):
        GatedComputerBackend(make_gated()[0], gate)  # nor a wrapper around a wrapper
    GatedComputerBackend(RecordingBackend(), gate)  # a backend that does NOT gate itself is exactly what the wrapper is for


def test_the_default_backend_is_one_process_wide_gate_not_one_per_object(process_gate):
    assert default_input_gate() is default_input_gate()
    assert WindowsComputerBackend().gate is WindowsComputerBackend().gate is default_input_gate()
    assert create_default_input_gate() is not default_input_gate()  # the constructor makes a NEW gate; the process-wide one is the shared instance


# ===================================================== the production reach, by reading the code ==================================================

APP_DIR = pathlib.Path(__file__).resolve().parent.parent / "app"
OS_INPUT_FUNCTIONS = frozenset({
    "write", "typewrite", "press", "hotkey", "keyDown", "keyUp", "moveTo", "moveRel", "move", "click", "doubleClick", "tripleClick", "rightClick",
    "middleClick", "mouseDown", "mouseUp", "dragTo", "dragRel", "drag", "scroll", "hscroll", "vscroll",
})
GATED_BACKEND_METHODS = ("open_application", "type_text", "press_key", "move_mouse", "click_mouse", "double_click", "scroll", "open_url", "switch_window")


def app_trees():
    for path in sorted(APP_DIR.rglob("*.py")):
        yield path.relative_to(APP_DIR).as_posix(), ast.parse(path.read_text(encoding="utf-8"))


def test_production_code_builds_the_real_backend_in_exactly_one_place():
    sites = [
        rel for rel, tree in app_trees() for node in ast.walk(tree)
        if isinstance(node, ast.Call) and getattr(node.func, "id", getattr(node.func, "attr", None)) == "WindowsComputerBackend"
    ]

    assert sites == ["computer/factory.py", "computer/factory.py"]  # exactly the default (no grant) and the explicitly authorized constructor, both in the factory


def test_production_code_never_references_the_wrapper_because_the_real_backend_gates_itself():
    users = sorted({
        rel for rel, tree in app_trees() for node in ast.walk(tree)
        if (isinstance(node, ast.Name) and node.id == "GatedComputerBackend") or (isinstance(node, ast.Attribute) and node.attr == "GatedComputerBackend")
    })

    assert users == []  # it is only DEFINED (computer/gate.py), for backends that do not gate themselves; nothing in production uses it


def test_no_other_module_in_the_application_calls_the_os_input_functions():
    offenders = [
        (rel, node.attr) for rel, tree in app_trees() for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "pyautogui"
        and node.attr in OS_INPUT_FUNCTIONS and rel != "computer/windows_backend.py"
    ]

    assert offenders == []  # observation/screen.py only ever calls pyautogui.screenshot (read-only)


def test_every_os_input_call_in_the_real_backend_sits_behind_a_gate_admission_in_the_same_method():
    tree = ast.parse((APP_DIR / "computer" / "windows_backend.py").read_text(encoding="utf-8"))
    backend = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "WindowsComputerBackend")
    methods = {m.name: m for m in backend.body if isinstance(m, ast.FunctionDef)}

    def os_calls(method):
        found = []
        for node in ast.walk(method):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "_bring_to_foreground":  # the focus-changing call `switch_window` ends in
                found.append(node.lineno)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                owner, name = node.func.value.id, node.func.attr
                if (owner == "pyautogui" and name in OS_INPUT_FUNCTIONS) or (owner == "subprocess" and name == "Popen") or (owner == "webbrowser" and name == "open"):
                    found.append(node.lineno)
        return found

    def admissions(method):
        return [
            n.lineno for n in ast.walk(method)
            if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "admit_input"  # gate.admit_input(gate, authority, action, *validated args)
        ]

    for name in GATED_BACKEND_METHODS:
        calls, admitted = os_calls(methods[name]), admissions(methods[name])
        assert calls and len(admitted) >= 1, name
        assert all(any(a < c for a in admitted) for c in calls), f"{name}: an OS call comes before any gate admission"
    ungated = {name for name, method in methods.items() if name not in GATED_BACKEND_METHODS and os_calls(method)}
    # `_activate_url_handler_window` brings the browser forward and is called ONLY by `open_url`, after its admission (test_computer_focus_authorization.py pins that order).
    assert ungated - {"_activate_url_handler_window"} == set(), ungated  # no other method of the real backend acts on the OS (scroll, open_url and switch_window are gated now)


# ===================================================== the production wiring ======================================================================


@pytest.fixture
def live(monkeypatch, recording_os, process_gate):
    """The REAL production objects in app/main.py: its real backend (the real WindowsComputerBackend code, its real process-wide gate and the real settings
    path), with only the OS libraries replaced by recorders."""
    import app.main as live_app

    backend = live_app._computer_backend
    assert isinstance(backend, WindowsComputerBackend) and backend.gate is default_input_gate()
    return live_app, backend, recording_os


def test_every_production_path_to_the_desktop_goes_through_the_one_self_gating_backend(live):
    live_app, backend, _ = live

    for name in INPUT_TOOLS:
        assert live_app.core.computer_tools.get(name).backend is backend
        if name != "scroll":
            assert live_app.core.tools.has(name) is False  # and no keyboard/mouse/launch tool is in the Job/automation registry: the allowlist did not widen
    for name in JOB_COMPUTER_TOOL_NAMES:
        assert live_app.core.tools.get(name).backend is backend  # the Job tools share it (scroll is gated, and this shared backend carries no grant)
    assert set(JOB_COMPUTER_TOOL_NAMES) == {"open_url", "scroll", "get_active_window", "switch_window"}


def test_with_the_default_off_the_production_registry_refuses_input_and_nothing_reaches_the_os(live):
    live_app, _, recorder = live

    results = [live_app.core.computer_tools.execute(name, params=params) for name, params, _ in INPUT_CALLS]

    assert all(r.status is ToolResultStatus.ERROR and "switched off" in r.message for r in results)
    assert recorder.calls == []


def test_switched_on_the_production_registry_still_admits_nothing_because_it_carries_no_grant(live, monkeypatch):
    live_app, _, recorder = live
    monkeypatch.setattr(settings, "computer_input_enabled", True)
    monkeypatch.setattr(settings, "computer_input_actions_per_minute", 2)

    results = [live_app.core.computer_tools.execute(name, params=params) for name, params, _ in INPUT_CALLS[:4]]

    assert all(r.status is ToolResultStatus.ERROR and "approved authorization" in r.message for r in results)  # the switch is necessary, not sufficient
    assert recorder.calls == []


def test_a_registry_over_an_authorized_real_backend_obeys_the_settings_limit(live, monkeypatch):
    _, _, recorder = live
    monkeypatch.setattr(settings, "computer_input_enabled", True)
    monkeypatch.setattr(settings, "computer_input_actions_per_minute", 2)
    registry = create_computer_tools_registry(backend=authorized_real_backend())  # the real backend code and the real process-wide gate, handed a grant per call

    results = [registry.execute(name, params=params) for name, params, _ in INPUT_CALLS[:4]]

    assert [r.status for r in results] == [ToolResultStatus.SUCCESS, ToolResultStatus.SUCCESS, ToolResultStatus.ERROR, ToolResultStatus.ERROR]
    assert recorder.calls == [OS_EFFECT["type_text"], OS_EFFECT["press_key"]]


def test_a_chained_chat_command_through_the_production_core_is_gated_too(live, monkeypatch):
    live_app, _, recorder = live
    monkeypatch.setattr(live_app.core, "memory", SQLiteMemoryStore(":memory:"))

    off = live_app.core.handle_chat(CHAIN)[0]
    assert "switched off" in off and "nothing was executed" in off.lower() and recorder.calls == []

    monkeypatch.setattr(settings, "computer_input_enabled", True)
    monkeypatch.setattr(settings, "computer_input_actions_per_minute", 2)
    on = live_app.core.handle_chat(CHAIN)[0]  # the production core carries no grant: switched on, the whole chain is still turned away before its first step

    assert recorder.calls == [] and "approved authorization" in on and "nothing was executed" in on.lower()


def test_a_chained_command_over_an_authorized_real_backend_stops_at_the_limit(live, monkeypatch):
    _, _, recorder = live
    say = real_chat(monkeypatch, enabled=True, limit=2)

    on = say(CHAIN)  # three input steps, a budget of two

    assert recorder.calls == [("hotkey", ("win", "r")), ("write", "notepad")]  # 'enter' never reached the OS
    assert "limited to 2 actions per minute" in on


# ===================================================== the audit log ============================================================================

SECRET = "TOP-SECRET-PASSWORD-9d2f"


def flattened(caplog) -> str:
    """Everything a log handler could ever emit for these records. (An 'allowed' line now carries the grant's random 8-hex-digit id prefix; it is masked so that a check such as
    'the text 777 is nowhere in the log' cannot be tripped by a random id that happens to contain it. The id is not an argument value.)"""
    return without_chat_trace_noise(without_grant_ids("\n".join(f"{r.getMessage()} | {r.msg} | {r.args!r} | {r.exc_text or ''}" for r in caplog.records)))


def test_the_audit_log_names_the_action_and_outcome_but_never_a_value(caplog):
    caplog.set_level(logging.DEBUG, logger="jarvis.computer")
    backend, inner, state, _ = make_gated(limit=3)
    registry = tools_over(backend)

    registry.execute("type_text", params={"text": SECRET})
    registry.execute("press_key", params={"key": "ctrl+shift+esc"})
    registry.execute("click_mouse", params={"x": 777, "y": 513})
    registry.execute("open_application", params={"app": "calculator"})  # over the limit
    state["enabled"] = False
    registry.execute("move_mouse", params={"x": 777, "y": 513})  # switched off
    text = flattened(caplog)

    assert "action=type_text" in text and "allowed" in text
    assert "action=press_key" in text
    assert "chars=" not in text and "parts=" not in text and "shape=" not in text  # no length or other metadata derived from an argument
    assert "reason=rate_limited" in text and "reason=disabled" in text and "stage=preflight" not in text  # registry calls skip the pre-check
    for value in (SECRET, "ctrl", "shift", "esc", "777", "513", "calculator"):
        assert value not in text, value
    assert all(r.name == "jarvis.computer.input" for r in caplog.records)
    levels = {("allowed" in r.getMessage()): r.levelno for r in caplog.records}
    assert levels == {True: logging.INFO, False: logging.WARNING}  # admitted attempts are INFO; every refusal is a WARNING


def test_a_failure_from_the_desktop_with_the_typed_text_in_it_is_not_logged(caplog):
    caplog.set_level(logging.DEBUG, logger="jarvis.computer")
    inner = RecordingBackend(error=ComputerControlError(f"the target field rejected {SECRET}"))
    backend, _, _, _ = make_gated(inner=inner)

    result = tools_over(backend).execute("type_text", params={"text": SECRET})

    assert result.status is ToolResultStatus.ERROR  # the caller still learns what happened...
    assert SECRET not in flattened(caplog) and any("action=type_text" in r.getMessage() for r in caplog.records)  # ...the audit trail does not carry it


def test_a_chained_command_turned_away_at_validation_leaves_one_line_per_refused_step_and_no_value(caplog):
    caplog.set_level(logging.DEBUG, logger="jarvis.computer")
    say, inner, _, _ = chat_over(enabled=False)

    say(f"use tool press_key win+r, then use tool type_text {SECRET}, then use tool press_key enter")
    text = flattened(caplog)

    assert [r.getMessage() for r in caplog.records] == [
        "computer_input refused action=press_key reason=disabled stage=preflight",
        "computer_input refused action=type_text reason=disabled stage=preflight",
        "computer_input refused action=press_key reason=disabled stage=preflight",
    ]  # the planner pre-checks each of the three steps; nothing was admitted, so there is no execution-time line to duplicate
    assert {r.levelno for r in caplog.records} == {logging.WARNING}  # a turned-away attempt is a WARNING, not a debug note
    assert SECRET not in text and "win" not in text and inner.effects == []


def test_log_records_carry_no_extra_fields_that_could_hold_a_value(caplog):
    caplog.set_level(logging.DEBUG, logger="jarvis.computer")
    backend, _, _, _ = make_gated(limit=1)
    tools_over(backend).execute("type_text", params={"text": SECRET})
    tools_over(backend).execute("type_text", params={"text": SECRET})
    standard = set(logging.LogRecord("n", logging.INFO, "p", 1, "m", None, None).__dict__) | {"message", "asctime", "taskName"}

    for record in caplog.records:
        assert set(record.__dict__) - standard == set()
        assert SECRET not in repr(record.__dict__)


# ===================================================== privacy: ANY logger, the real stack, every path ============================================


def everything_logged_anywhere(caplog) -> str:
    """Every record of every logger, as a handler would render it - message, arguments and any traceback."""
    formatter = logging.Formatter("%(name)s %(levelname)s %(message)s")
    # (grant id prefixes, and the chat trace's random conversation id and duration, are masked: see `flattened` and grant_support.without_chat_trace_noise)
    return without_chat_trace_noise(without_grant_ids("\n".join(f"{formatter.format(r)} | {r.args!r} | {r.exc_info!r}" for r in caplog.records)))


def real_chat(monkeypatch, *, enabled, limit):
    """A JarvisCore over the REAL WindowsComputerBackend code and the process-wide gate - handed a real grant per call, since the production default carries none - with only the OS
    libraries recorded."""
    monkeypatch.setattr(settings, "computer_input_enabled", enabled)
    monkeypatch.setattr(settings, "computer_input_actions_per_minute", limit)
    jarvis = JarvisCore(
        provider=_NoAI(), memory=SQLiteMemoryStore(":memory:"), tools=ToolRegistry(),
        computer_tools=create_computer_tools_registry(backend=authorized_real_backend()), tool_call_builder=ToolCallBuilder(_NoToolAI()),
    )
    return lambda message: jarvis.handle_chat(message)[0]


TYPE_SECRET = f"use tool type_text {SECRET}"
PRIVACY_SCENARIOS = {
    "allowed": (True, 10, [TYPE_SECRET], None),
    "switched_off": (False, 10, [TYPE_SECRET], None),
    "over_the_limit": (True, 1, ["use tool press_key enter", TYPE_SECRET], None),
    "chain_turned_away_at_validation": (False, 10, [f"use tool press_key win+r, then use tool type_text {SECRET}, then use tool press_key enter"], None),
    "chain_stopped_midway": (True, 1, [f"use tool press_key enter, then use tool type_text {SECRET}"], None),
    "os_call_fails_with_a_control_error": (True, 10, [TYPE_SECRET], ComputerControlError(f"the OS rejected {SECRET}")),
    "os_library_crashes_with_the_text": (True, 10, [TYPE_SECRET], RuntimeError(f"the input library crashed typing {SECRET}")),
}


@pytest.mark.parametrize("name", list(PRIVACY_SCENARIOS))
def test_no_log_record_from_any_logger_ever_holds_the_typed_text(name, caplog, monkeypatch, recording_os, process_gate):
    enabled, limit, messages, failure = PRIVACY_SCENARIOS[name]
    caplog.set_level(logging.DEBUG)  # the ROOT logger: every logger in the application, not only ours
    say = real_chat(monkeypatch, enabled=enabled, limit=limit)
    recording_os.fail = failure

    for message in messages:
        say(message)  # the caller may be told what happened, including (for an OS error) the text - that is the reply, not the log

    assert SECRET not in everything_logged_anywhere(caplog)


def test_no_log_record_holds_a_key_a_coordinate_an_application_or_any_other_argument_value(caplog, monkeypatch, recording_os, process_gate):
    caplog.set_level(logging.DEBUG)
    say = real_chat(monkeypatch, enabled=True, limit=3)

    for message in (
        "use tool click_mouse 777 513", "use tool press_key ctrl+shift+esc", f"use tool type_text {SECRET}",
        "use tool open_application calculator", "use tool move_mouse 777 513",  # the last two are over the limit of three
    ):
        say(message)
    text = everything_logged_anywhere(caplog)

    assert "action=click_mouse" in text and "action=press_key" in text and "reason=rate_limited" in text  # the trail is useful...
    for value in (SECRET, "777", "513", "ctrl+shift+esc", "calculator", "left"):
        assert value not in text, value  # ...and carries no argument value
    for token in ("ctrl", "shift", "esc", "calculator"):
        assert re.search(rf"\b{token}\b", text) is None, token  # not even the key names on their own, in any spelling (a tuple, a list, a join)


def test_the_exact_audit_trail_through_config_changes_and_a_crashing_backend(caplog, monkeypatch, recording_os, process_gate):
    """One mixed sequence on the real backend, asserting the log line for EVERY attempt: off by default, switched on, the OS library crashing AFTER the gate
    admitted the action, the limit lowered live below what is used, raised live, and switched off again. Tool name, outcome, reason and count - never a value
    and never anything derived from one (no lengths)."""
    caplog.set_level(logging.DEBUG, logger="jarvis.computer")
    bare = authorized_real_backend()  # real backend, real process-wide gate, a real one-entry grant per call
    outcomes = []

    def attempt(call):
        try:
            call()
            outcomes.append("ok")
        except ComputerControlError:
            outcomes.append("refused")
        except Exception:  # noqa: BLE001 - the OS library crashing: not ours to swallow, only to record that it happened
            outcomes.append("crashed")

    attempt(lambda: bare.type_text(SECRET))  # 1. switched off (the default)
    monkeypatch.setattr(settings, "computer_input_enabled", True)  # 2. configuration changes between calls: on, three per minute
    monkeypatch.setattr(settings, "computer_input_actions_per_minute", 3)
    attempt(lambda: bare.type_text(SECRET))
    recording_os.fail = RuntimeError(f"the input library crashed on {SECRET}")  # 3. the OS call raises AFTER the gate admitted the action
    attempt(lambda: bare.press_key("ctrl+shift+esc"))
    recording_os.fail = None
    monkeypatch.setattr(settings, "computer_input_actions_per_minute", 2)  # 4. the limit lowered live, below the two already used
    attempt(lambda: bare.move_mouse(777, 513))
    monkeypatch.setattr(settings, "computer_input_actions_per_minute", 4)  # 5. raised live: one more fits
    attempt(lambda: bare.move_mouse(777, 513))
    monkeypatch.setattr(settings, "computer_input_enabled", False)  # 6. switched off again, with budget to spare
    attempt(lambda: bare.open_application(AllowedApplication.CALCULATOR))
    lines = budget_trail([r.getMessage() for r in caplog.records])  # (an 'allowed' line also names the grant and its entry; the trail below is the action/outcome/count part)

    assert outcomes == ["refused", "ok", "crashed", "refused", "ok", "refused"]
    assert lines == [
        "computer_input refused action=type_text reason=disabled",
        "computer_input allowed action=type_text used=1/3",
        "computer_input allowed action=press_key used=2/3",  # admitted (and counted) before the library failed
        "computer_input refused action=move_mouse reason=rate_limited",
        "computer_input allowed action=move_mouse used=3/4",
        "computer_input refused action=open_application reason=disabled",
    ]
    text = "\n".join(lines) + everything_logged_anywhere(caplog)
    assert SECRET not in text and "777" not in text and "513" not in text and "calculator" not in text
    for token in ("ctrl", "shift", "esc"):
        assert re.search(rf"\b{token}\b", text) is None, token


def test_one_chain_using_all_six_tools_shares_one_budget_through_the_real_stack(monkeypatch, recording_os, process_gate):
    """The production path end to end: chat command -> explicit-command planner -> the real tools -> the real backend -> its gate. Four slots, six tools."""
    say = real_chat(monkeypatch, enabled=True, limit=4)

    reply = say(
        "use tool move_mouse 10 20, then use tool click_mouse 10 20, then use tool double_click 10 20, "
        "then use tool open_application notepad, then use tool press_key enter, then use tool type_text hello"
    )

    assert recording_os.calls == [
        ("moveTo", 10, 20),
        ("click", {"x": 10, "y": 20, "button": "left"}),
        ("doubleClick", {"x": 10, "y": 20, "button": "left"}),
        ("Popen", ["notepad.exe"], False),
    ]  # the fifth and sixth tools never reached the OS
    assert "limited to 4 actions per minute" in reply and "Stopped after step 5; step(s) 6 were not run." in reply
