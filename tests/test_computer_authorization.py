"""
Authorization grants, the ENFORCEMENT half: the global switch is necessary but no longer sufficient. OS input needs a grant that covers THIS exact action with THESE
exact arguments, next in its approved order - checked by the gate, inside the backend, immediately before every OS-input call, atomically with the rate limit.

Real: the gate, the grants, the issuer, the real `WindowsComputerBackend` code, the tools, `ToolRegistry`, `JarvisCore` with its explicit-command planner, the
production objects in app/main.py. Faked: ONLY the OS - `pyautogui`, `subprocess` and `webbrowser` inside the backend module are replaced by a recorder for every test
here, so nothing in this file can move a real mouse, press a real key, launch a program or open a browser. The monotonic clock is injected.
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
from app.computer.authority import MAX_GRANT_ENTRIES, MAX_GRANT_SECONDS, AuthorityError, InputGrant, InputStep, canonical_input
from app.computer.base import (
    AllowedApplication,
    ComputerBackend,
    ComputerControlError,
    MouseButton,
    ScreenInfo,
    WindowInfo,
    normalize_key_input,
)
from app.computer.defaults import JOB_COMPUTER_TOOL_NAMES, create_computer_tools_registry
from app.computer.factory import create_authorized_computer_backend, create_default_computer_backend
from app.computer.gate import (
    GATED_METHODS,
    NOT_AUTHORIZED,
    READ_ONLY_METHODS,
    SWITCHED_OFF,
    UNGATED_METHODS,
    UNPROTECTED_ACTIONS,
    GatedComputerBackend,
    InputGate,
    admit_input,
    default_input_gate,
)
from app.computer.issuer import GrantIssuer
from app.computer.tools import ScrollTool
from app.computer.windows_backend import WindowsComputerBackend
from app.config import settings
from app.core import JarvisCore
from app.memory.sqlite_store import SQLiteMemoryStore
from app.orchestration.base import TERMINAL_JOB_STATUSES, AutonomyLevel, JobStatus, RetryPolicy, StepStatus
from app.orchestration.engine import JobEngine
from app.orchestration.planner import Planner
from app.orchestration.sqlite_store import SQLiteJobStore
from app.orchestration.validator import validate_plan
from app.providers.base import AIProvider
from app.tools.base import PermissionLevel, ToolArgumentError, ToolResultStatus
from app.tools.decision import ToolCallBuilder, ToolCallProvider
from app.tools.registry import ToolRegistry

APP_DIR = pathlib.Path(__file__).resolve().parent.parent / "app"
SECRET = "TOP-SECRET-PASSWORD-9d2f"

# ====================================================== the fake OS and the helpers =====================================================


class FakeOS:
    """Stands in for pyautogui, subprocess and webbrowser INSIDE the real backend module: the backend's own code runs, this only records what it finally asks the OS."""

    FAILSAFE = True

    def __init__(self):
        self.calls: list[tuple] = []
        self.lock = threading.Lock()

    def _record(self, *call):
        with self.lock:
            self.calls.append(call)

    def size(self):
        return (1920, 1080)

    def write(self, text, interval=0.0):
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


@pytest.fixture(autouse=True)
def os_recorder(monkeypatch):
    recorder = FakeOS()
    for attribute in ("pyautogui", "subprocess", "webbrowser"):
        monkeypatch.setattr(windows_backend, attribute, recorder)
    return recorder


class Clock:
    def __init__(self, now: float = 1000.0):
        self.now = now

    def __call__(self) -> float:
        return self.now


def make_gate(*, enabled=True, limit=10, clock=None):
    """A gate that REQUIRES authority (the default), with the switch and the limit under the test's control."""
    clock = clock or Clock()
    state = {"enabled": enabled, "limit": limit}
    return InputGate(lambda: state["enabled"], lambda: state["limit"], clock=clock), state, clock


#: action -> (the call on a backend, the InputStep that authorizes exactly that call, a DIFFERENT call of the same action, the OS effect count of the real call)
CALLS = {
    "type_text": (lambda b: b.type_text("hello"), InputStep("type_text", ("hello",)), lambda b: b.type_text("hellp")),
    "press_key": (lambda b: b.press_key("ctrl+n"), InputStep("press_key", (normalize_key_input("ctrl+n"),)), lambda b: b.press_key("ctrl+s")),
    "move_mouse": (lambda b: b.move_mouse(10, 20), InputStep("move_mouse", (10, 20)), lambda b: b.move_mouse(10, 21)),
    "click_mouse": (
        lambda b: b.click_mouse(10, 20, MouseButton.LEFT),
        InputStep("click_mouse", (10, 20, MouseButton.LEFT)),
        lambda b: b.click_mouse(10, 20, MouseButton.RIGHT),
    ),
    "double_click": (
        lambda b: b.double_click(10, 20, MouseButton.LEFT),
        InputStep("double_click", (10, 20, MouseButton.LEFT)),
        lambda b: b.double_click(11, 20, MouseButton.LEFT),
    ),
    "scroll": (lambda b: b.scroll(5), InputStep("scroll", (5,)), lambda b: b.scroll(6)),
    "open_application": (
        lambda b: b.open_application(AllowedApplication.NOTEPAD),
        InputStep("open_application", (AllowedApplication.NOTEPAD,)),
        lambda b: b.open_application(AllowedApplication.CALCULATOR),
    ),
}
ACTIONS = list(CALLS)
TOOL_PARAMS = {
    "type_text": {"text": "hello"},
    "press_key": {"key": "ctrl+n"},
    "move_mouse": {"x": 10, "y": 20},
    "click_mouse": {"x": 10, "y": 20},
    "double_click": {"x": 10, "y": 20},
    "scroll": {"amount": 5},
    "open_application": {"app": "notepad"},
}


def issue(gate, actions, *, ttl=120, purpose="test", actor="test"):
    return GrantIssuer(gate).issue([CALLS[a][1] for a in actions], ttl_seconds=ttl, purpose=purpose, actor=actor)


def authorized(gate, actions, **kwargs):
    grant = issue(gate, actions, **kwargs)
    return grant, WindowsComputerBackend(gate=gate, authority=grant)


def denied(call, backend, *, text=NOT_AUTHORIZED):
    with pytest.raises(ComputerControlError) as caught:
        call(backend)
    assert str(caught.value) == text
    return caught.value


class SlowCountDeque(deque):
    """Reports its length only after a pause: an unlocked check-then-append then acts on a stale count, so a missing lock shows up every time."""

    def __len__(self):
        size = super().__len__()
        time.sleep(0.0003)
        return size


def hammer(backend, call, threads):
    """Run `call(backend)` from `threads` threads released at the same instant; return how many were admitted."""
    barrier = threading.Barrier(threads)
    results = []

    def attempt():
        barrier.wait()
        try:
            call(backend)
            results.append(True)
        except ComputerControlError:
            results.append(False)

    workers = [threading.Thread(target=attempt) for _ in range(threads)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    return results.count(True)


# ============================================ 1 + 2. default deny: the switch AND a covering grant, for every input method ==========================


@pytest.mark.parametrize("action", ACTIONS)
def test_every_input_method_is_denied_when_the_switch_is_off_even_with_a_grant(action, os_recorder):
    gate, state, _ = make_gate()
    grant, backend = authorized(gate, [action])
    state["enabled"] = False

    denied(CALLS[action][0], backend, text=SWITCHED_OFF)
    denied(CALLS[action][0], WindowsComputerBackend(gate=gate), text=SWITCHED_OFF)  # and a bare backend, of course

    assert os_recorder.calls == [] and len(gate._events) == 0


@pytest.mark.parametrize("action", ACTIONS)
def test_every_input_method_is_denied_with_the_switch_on_but_no_grant(action, os_recorder):
    gate, _, _ = make_gate()
    bare = WindowsComputerBackend(gate=gate)

    denied(CALLS[action][0], bare)

    assert os_recorder.calls == [] and len(gate._events) == 0  # nothing reached the OS and no rate-limit slot was used


@pytest.mark.parametrize("action", ACTIONS)
def test_the_wrapper_for_other_backends_enforces_the_same_rule(action):
    gate, _, _ = make_gate()

    class Recording(ComputerBackend):
        effects: list = []

        def get_active_window(self):
            return WindowInfo("t", "p.exe", False)

        def get_screen_info(self):
            return ScreenInfo(1920, 1080)

        def open_application(self, app): self.effects.append("open_application")
        def open_url(self, url): self.effects.append("open_url")
        def switch_window(self, title, process): self.effects.append("switch_window"); return WindowInfo(title, process, False)
        def type_text(self, text): self.effects.append("type_text")
        def press_key(self, key): self.effects.append("press_key")
        def move_mouse(self, x, y): self.effects.append("move_mouse")
        def click_mouse(self, x, y, button): self.effects.append("click_mouse")
        def double_click(self, x, y, button): self.effects.append("double_click")
        def scroll(self, amount): self.effects.append("scroll")

    inner = Recording()
    inner.effects = []
    denied(CALLS[action][0], GatedComputerBackend(inner, gate))
    grant = issue(gate, [action])
    CALLS[action][0](GatedComputerBackend(inner, gate, authority=grant))

    assert inner.effects == [action]


def test_issuing_a_grant_is_refused_while_the_switch_is_off_and_registers_nothing():
    gate, state, _ = make_gate(enabled=False)

    with pytest.raises(ComputerControlError, match="switched off"):
        issue(gate, ["type_text"])

    assert gate._grants == {}
    state["enabled"] = True
    assert gate.remaining() == 0  # still nothing: no grant exists to cover anything


def test_the_default_and_the_production_backends_carry_no_authority_so_they_admit_nothing(monkeypatch):
    import app.main as live_app

    assert create_default_computer_backend()._authority is None and WindowsComputerBackend()._authority is None
    assert live_app._computer_backend._authority is None
    assert not hasattr(default_input_gate(), "_require_authority")  # there is no authorization mode to be in: every gate requires a grant

    monkeypatch.setattr(settings, "computer_input_enabled", True)
    monkeypatch.setattr(settings, "computer_input_actions_per_minute", 10)
    denied(CALLS["type_text"][0], live_app._computer_backend)  # the switch is on and the limit has room: still nothing


def test_an_authorized_backend_must_be_given_a_real_grant():
    with pytest.raises(TypeError):
        create_authorized_computer_backend("not a grant")
    gate, _, _ = make_gate()
    backend = WindowsComputerBackend(gate=gate, authority="deadbeef" * 4)  # anything that is not the registered grant object is nobody
    denied(CALLS["type_text"][0], backend)


# ============================================ nothing is left outside the gate any more (open_url and switch_window were the documented gap) ======================


def test_nothing_is_left_unprotected_and_only_the_two_reads_are_outside_the_grant_gate():
    assert UNPROTECTED_ACTIONS == frozenset()  # open_url and switch_window were listed here until they were put behind the gate (tests/test_computer_focus_authorization.py)
    assert UNGATED_METHODS == READ_ONLY_METHODS == {"get_active_window", "get_screen_info"}
    assert GATED_METHODS.isdisjoint(UNPROTECTED_ACTIONS | READ_ONLY_METHODS) and {"open_url", "switch_window"} <= GATED_METHODS


def test_open_url_and_switch_window_no_longer_pass_through_the_wrapper_with_no_grant():
    gate, _, _ = make_gate(enabled=True)
    seen = []

    class Inner(ComputerBackend):
        def get_active_window(self): return WindowInfo("t", "p.exe", False)
        def get_screen_info(self): return ScreenInfo(1, 1)
        def open_application(self, app): raise AssertionError
        def open_url(self, url): seen.append(("open_url", url))
        def switch_window(self, title, process): seen.append(("switch_window", title, process)); return WindowInfo(title, process, False)
        def type_text(self, text): raise AssertionError
        def press_key(self, key): raise AssertionError
        def move_mouse(self, x, y): raise AssertionError
        def click_mouse(self, x, y, button): raise AssertionError
        def double_click(self, x, y, button): raise AssertionError
        def scroll(self, amount): raise AssertionError

    wrapper = GatedComputerBackend(Inner(), gate)
    with pytest.raises(ComputerControlError, match="approved authorization"):
        wrapper.open_url("https://example.com")
    with pytest.raises(ComputerControlError, match="approved authorization"):
        wrapper.switch_window("Notes", "notes.exe")

    assert seen == [] and len(gate._events) == 0


# ============================================ 3. chat, chained commands, registry and tool calls cannot bypass it ===================================


@pytest.mark.parametrize("action", ACTIONS)
def test_a_direct_tool_call_and_a_registry_call_cannot_bypass_authorization(action, os_recorder):
    gate, _, _ = make_gate()
    registry = create_computer_tools_registry(backend=WindowsComputerBackend(gate=gate))

    direct = registry.get(action).execute(**TOOL_PARAMS[action])  # skips the registry AND validate_arguments
    via_registry = registry.execute(action, params=TOOL_PARAMS[action])

    assert direct.status is ToolResultStatus.ERROR and direct.message == NOT_AUTHORIZED
    assert via_registry.status is ToolResultStatus.ERROR and via_registry.message == NOT_AUTHORIZED
    assert os_recorder.calls == [] and len(gate._events) == 0


@pytest.mark.parametrize("action", ACTIONS)
def test_each_tools_early_pre_check_turns_an_unauthorized_call_away_but_it_is_only_a_convenience(action, os_recorder):
    gate, _, _ = make_gate()
    tool = create_computer_tools_registry(backend=WindowsComputerBackend(gate=gate)).get(action)

    with pytest.raises(ToolArgumentError, match="needs an approved authorization"):
        tool.validate_arguments(**TOOL_PARAMS[action])

    assert os_recorder.calls == [] and len(gate._events) == 0


class _NoAI(AIProvider):
    def generate_reply(self, message):
        raise AssertionError("no AI may be used for an explicit tool command")


class _NoToolAI(ToolCallProvider):
    def decide(self, message, candidate_tools):
        raise AssertionError("no AI may be used for an explicit tool command")


def chat_over(backend):
    jarvis = JarvisCore(
        provider=_NoAI(), memory=SQLiteMemoryStore(":memory:"), tools=ToolRegistry(),
        computer_tools=create_computer_tools_registry(backend=backend), tool_call_builder=ToolCallBuilder(_NoToolAI()),
    )
    return lambda message: jarvis.handle_chat(message)[0]


CHAIN = "use tool press_key win+r, then use tool type_text notepad, then use tool press_key enter"


def test_chat_commands_single_and_chained_cannot_type_even_with_the_switch_on(os_recorder):
    gate, _, _ = make_gate()
    say = chat_over(WindowsComputerBackend(gate=gate))

    single = say(f"use tool type_text {SECRET}")
    chained = say(CHAIN)
    scrolled = say("use tool scroll -600")

    assert "needs an approved authorization" in single and SECRET not in single
    assert "needs an approved authorization" in chained and "nothing was executed" in chained.lower()  # the whole chain is turned away before its first step
    assert "needs an approved authorization" in scrolled
    assert os_recorder.calls == [] and len(gate._events) == 0


def test_a_grant_for_the_demos_backend_does_not_authorize_chat_that_uses_another_backend(os_recorder):
    gate, _, _ = make_gate()
    grant, demo_backend = authorized(gate, ["type_text", "press_key"])
    say = chat_over(WindowsComputerBackend(gate=gate))  # chat's backend: same gate, same switch, no grant handed to it

    reply = say(f"use tool type_text {SECRET}")

    assert "needs an approved authorization" in reply and os_recorder.calls == []
    assert gate.unspent(grant) == 2  # and the demo's grant was not touched by the refused attempt


class _NoPlanner(Planner):
    def plan(self, goal, *, available_tools, context, autonomy, replan_context=None):
        raise AssertionError("a fixed plan: nothing may re-plan it")


def test_a_background_job_with_a_scroll_step_runs_the_step_and_is_refused_by_the_backend_not_by_the_plan_validator(os_recorder):
    """Jobs' scroll goes through the shared, grantless backend. The Job plan validator checks argument TYPES only (it never calls a tool's own pre-check), so the plan is
    accepted - and the enforcement that matters is the backend's, at execution: the JobEngine's own step fails and nothing reaches the OS."""
    gate, _, _ = make_gate()
    registry = ToolRegistry()
    registry.register(ScrollTool(WindowsComputerBackend(gate=gate)))  # exactly how the Job registry holds it
    steps = validate_plan(
        [{"id": "s1", "tool": "scroll", "arguments": {"amount": 5}, "depends_on": [], "max_attempts": 1, "retry_delay_seconds": 0.0}],
        registry, max_steps=5, default_retry=RetryPolicy(),
    )
    store = SQLiteJobStore(":memory:")
    job = store.create_running_job("scroll something", steps, autonomy=AutonomyLevel.ASSISTED)
    engine = JobEngine(
        store=store, tools=registry, planner=_NoPlanner(), max_steps=5, max_total_steps=10, max_retries_per_step=1, max_replans=0,
        retry_delay_seconds=0.0, poll_interval_seconds=1000.0, sleep=lambda seconds: None,
    )

    for _ in range(20):
        engine.tick()
        job = store.get(job.id)
        if job.status in TERMINAL_JOB_STATUSES:
            break

    step = job.plan.steps[0]
    assert job.status is JobStatus.FAILED and step.status is StepStatus.FAILED and step.error == NOT_AUTHORIZED
    assert os_recorder.calls == [] and len(gate._events) == 0 and registry.execute("scroll", params={"amount": 5}).message == NOT_AUTHORIZED


def test_the_production_registries_in_app_main_cannot_type_scroll_or_click_with_the_switch_on(monkeypatch, os_recorder):
    import app.main as live_app

    process = default_input_gate()
    monkeypatch.setattr(process, "_events", deque())
    monkeypatch.setattr(settings, "computer_input_enabled", True)
    monkeypatch.setattr(settings, "computer_input_actions_per_minute", 10)

    for action in ACTIONS:
        result = live_app.core.computer_tools.execute(action, params=TOOL_PARAMS[action])  # the interactive registry chat reaches
        assert result.status is ToolResultStatus.ERROR and result.message == NOT_AUTHORIZED, action
    job_scroll = live_app.core.tools.execute("scroll", params={"amount": 5})  # the registry Jobs execute through
    assert job_scroll.status is ToolResultStatus.ERROR and job_scroll.message == NOT_AUTHORIZED
    assert os_recorder.calls == [] and len(process._events) == 0


# ============================================ 4. the grant is exact: wrong action / arguments / expired / exhausted / forged / replayed ==============


def test_a_grant_covering_all_seven_actions_in_order_lets_exactly_those_through_once_each(os_recorder):
    gate, _, _ = make_gate()
    grant, backend = authorized(gate, ACTIONS)

    for action in ACTIONS:
        CALLS[action][0](backend)

    assert len(os_recorder.calls) == 7 and gate.unspent(grant) == 0 and len(gate._events) == 7
    denied(CALLS["type_text"][0], backend)  # the grant is spent: it covers nothing now


@pytest.mark.parametrize("action", ACTIONS)
def test_a_wrong_argument_is_refused_and_revokes_the_whole_grant(action, os_recorder):
    gate, _, _ = make_gate()
    grant, backend = authorized(gate, [action, "type_text"])

    denied(CALLS[action][2], backend)  # the same action, a different argument

    assert os_recorder.calls == [] and len(gate._events) == 0  # nothing ran and no slot was used
    assert gate._grants[grant.id].index == 0 and gate._grants[grant.id].closed == "mismatch"  # no entry was consumed, and the grant is over
    denied(CALLS[action][0], backend)  # even the exactly-right call is refused now
    assert gate.unspent(grant) == 0 and os_recorder.calls == []


def test_a_wrong_action_is_refused_and_revokes_the_whole_grant(os_recorder):
    gate, _, _ = make_gate()
    grant, backend = authorized(gate, ["type_text", "press_key"])

    denied(CALLS["scroll"][0], backend)  # an action the grant never listed

    assert os_recorder.calls == [] and gate._grants[grant.id].closed == "mismatch"
    denied(CALLS["type_text"][0], backend)


def test_the_right_actions_in_the_wrong_order_are_refused_and_revoke_the_grant(os_recorder):
    gate, _, _ = make_gate()
    grant, backend = authorized(gate, ["type_text", "press_key"])

    denied(CALLS["press_key"][0], backend)  # the second approved action, asked for first

    assert os_recorder.calls == [] and gate._grants[grant.id].index == 0 and gate._grants[grant.id].closed == "mismatch"
    denied(CALLS["type_text"][0], backend)


def test_a_grant_expires_on_the_monotonic_clock_at_exactly_its_lifetime(os_recorder):
    gate, _, clock = make_gate()
    grant, backend = authorized(gate, ["type_text", "type_text", "type_text"], ttl=120)

    clock.now += 119.999
    CALLS["type_text"][0](backend)  # still valid 1 ms before it ends
    clock.now += 0.001  # exactly 120 s after issue
    denied(CALLS["type_text"][0], backend)

    assert len(os_recorder.calls) == 1 and gate.unspent(grant) == 0 and gate._grants == {}  # expired and forgotten


def test_a_grant_can_never_be_asked_for_more_than_the_maximum_lifetime_or_size():
    gate, _, _ = make_gate()
    assert MAX_GRANT_SECONDS == 300.0
    issue(gate, ["type_text"], ttl=300)  # the cap itself is fine
    for bad_ttl in (0, -1, 300.001, 3600, True, float("nan"), float("inf"), None, "120"):
        with pytest.raises(AuthorityError):
            issue(gate, ["type_text"], ttl=bad_ttl)
    issuer = GrantIssuer(gate)
    with pytest.raises(AuthorityError):
        issuer.issue([], ttl_seconds=10, purpose="test", actor="test")
    with pytest.raises(AuthorityError):
        issuer.issue([CALLS["scroll"][1]] * (MAX_GRANT_ENTRIES + 1), ttl_seconds=10, purpose="test", actor="test")
    with pytest.raises(AuthorityError):
        issuer.issue(["type_text"], ttl_seconds=10, purpose="test", actor="test")  # not InputSteps
    for bad_label in ("Has Space", "x\ny", ""):
        with pytest.raises(AuthorityError):
            issuer.issue([CALLS["scroll"][1]], ttl_seconds=10, purpose=bad_label, actor="test")
        with pytest.raises(AuthorityError):
            issuer.issue([CALLS["scroll"][1]], ttl_seconds=10, purpose="test", actor=bad_label)
    issuer.issue([CALLS["scroll"][1]] * MAX_GRANT_ENTRIES, ttl_seconds=10, purpose="test", actor="test")  # the size cap itself is fine


def test_the_number_of_active_grants_is_bounded_and_a_revoked_one_frees_a_place():
    gate, _, _ = make_gate()
    grants = [issue(gate, ["type_text"]) for _ in range(8)]

    with pytest.raises(AuthorityError, match="too many"):
        issue(gate, ["type_text"])
    gate.revoke(grants[0])
    issue(gate, ["type_text"])


def test_an_exhausted_grant_admits_nothing_more_and_a_replay_on_a_new_backend_is_refused_too(os_recorder):
    gate, _, _ = make_gate()
    grant, backend = authorized(gate, ["type_text"])
    CALLS["type_text"][0](backend)

    denied(CALLS["type_text"][0], backend)  # the same call again: a replay
    denied(CALLS["type_text"][0], WindowsComputerBackend(gate=gate, authority=grant))  # the same grant handed to a fresh backend: still spent

    assert len(os_recorder.calls) == 1 and len(gate._events) == 1


def test_two_identical_entries_allow_exactly_two_identical_calls(os_recorder):
    gate, _, _ = make_gate()
    _, backend = authorized(gate, ["type_text", "type_text"])

    CALLS["type_text"][0](backend)
    CALLS["type_text"][0](backend)
    denied(CALLS["type_text"][0], backend)

    assert len(os_recorder.calls) == 2


def test_a_forged_grant_is_nobody_whether_a_lookalike_a_copied_id_or_from_another_gate(os_recorder):
    gate, _, _ = make_gate()
    real, real_backend = authorized(gate, ["type_text"])
    never_registered = InputGrant()
    lookalike = InputGrant()
    lookalike._id = real.id  # a different object wearing the real grant's id
    other_gate, _, _ = make_gate()
    foreign = issue(other_gate, ["type_text"])

    for fake in (never_registered, lookalike, foreign):
        denied(CALLS["type_text"][0], WindowsComputerBackend(gate=gate, authority=fake))

    assert os_recorder.calls == [] and len(gate._events) == 0
    assert gate.unspent(real) == 1  # none of the forgeries touched the real grant or revoked it
    CALLS["type_text"][0](real_backend)  # and it still works for its holder
    assert len(os_recorder.calls) == 1


def test_a_right_grant_with_arguments_built_for_a_different_action_is_a_mismatch(os_recorder):
    gate, _, _ = make_gate()
    grant, _ = authorized(gate, ["type_text"])

    with pytest.raises(ComputerControlError):
        gate.admit("type_text", authority=grant, canonical=canonical_input("press_key", ("enter",)))  # a press_key's form presented as a type_text

    assert gate._grants[grant.id].closed == "mismatch" and len(gate._events) == 0


def test_an_unknown_action_and_unauthorizable_arguments_are_refused_without_consuming_anything(os_recorder):
    gate, _, _ = make_gate()
    grant, _ = authorized(gate, ["scroll"])

    with pytest.raises(ComputerControlError):
        gate.admit("sudo_type", authority=grant, canonical=canonical_input("scroll", 5))  # not a gated action at all
    with pytest.raises(ComputerControlError):
        admit_input(gate, grant, "scroll", True)  # a bool cannot be put in canonical form, so it cannot be authorized
    with pytest.raises(ComputerControlError):
        admit_input(gate, grant, "scroll", 5.5)  # nor can a float

    assert gate.unspent(grant) == 1 and len(gate._events) == 0  # the grant is intact: these never reached a match
    CALLS["scroll"][0](WindowsComputerBackend(gate=gate, authority=grant))
    assert len(os_recorder.calls) == 1


# ============================================ 5. concurrency: no entry spent twice, no way around the limit =========================================


@pytest.mark.parametrize("round_number", range(4))
def test_concurrent_calls_cannot_spend_one_grant_entry_twice(round_number, os_recorder):
    gate, _, _ = make_gate(limit=10**6)
    gate._events = SlowCountDeque()
    grant, backend = authorized(gate, ["type_text"])

    admitted = hammer(backend, CALLS["type_text"][0], threads=48)

    assert admitted == 1 and len(os_recorder.calls) == 1 and gate.unspent(grant) == 0


@pytest.mark.parametrize("round_number", range(4))
def test_concurrent_calls_admit_exactly_as_many_as_the_grant_lists(round_number, os_recorder):
    gate, _, _ = make_gate(limit=10**6)
    gate._events = SlowCountDeque()
    _, backend = authorized(gate, ["type_text"] * 5)

    admitted = hammer(backend, CALLS["type_text"][0], threads=48)

    assert admitted == 5 and len(os_recorder.calls) == 5


@pytest.mark.parametrize("round_number", range(4))
def test_concurrent_calls_cannot_exceed_the_rate_limit_and_refused_ones_do_not_burn_entries(round_number, os_recorder):
    gate, _, clock = make_gate(limit=3)
    gate._events = SlowCountDeque()
    grant, backend = authorized(gate, ["type_text"] * 10)

    admitted = hammer(backend, CALLS["type_text"][0], threads=48)

    assert admitted == 3 and len(os_recorder.calls) == 3 and len(gate._events) == 3
    assert gate.unspent(grant) == 7  # 45 refusals spent no entry: only the 3 admitted ones did
    clock.now += 60.0  # the window slides: the unspent entries are still there
    assert hammer(backend, CALLS["type_text"][0], threads=48) == 3 and gate.unspent(grant) == 4


# ============================================ 6. a rejected action consumes neither a grant entry nor a rate-limit slot ===============================


def test_a_rate_limited_refusal_consumes_no_entry_and_no_slot_and_does_not_revoke(os_recorder):
    gate, _, clock = make_gate(limit=1)
    grant, backend = authorized(gate, ["type_text", "press_key"])
    CALLS["type_text"][0](backend)

    with pytest.raises(ComputerControlError, match="limited to 1"):
        CALLS["press_key"][0](backend)

    assert gate.unspent(grant) == 1 and len(gate._events) == 1 and gate._grants[grant.id].closed is None
    clock.now += 60.0
    CALLS["press_key"][0](backend)  # the entry was waiting, intact
    assert len(os_recorder.calls) == 2


def test_the_switch_being_off_refuses_without_a_slot_and_ends_every_grant_for_good(os_recorder):
    gate, state, _ = make_gate()
    grant, backend = authorized(gate, ["type_text", "press_key"])
    other, _ = authorized(gate, ["scroll"])
    state["enabled"] = False

    denied(CALLS["type_text"][0], backend, text=SWITCHED_OFF)
    state["enabled"] = True  # turned back on: no grant survived having been seen off

    denied(CALLS["type_text"][0], backend)
    assert len(gate._events) == 0 and os_recorder.calls == [] and gate.unspent(grant) == 0 and gate.unspent(other) == 0


def test_the_non_consuming_questions_never_consume_revoke_or_log(os_recorder, caplog):
    gate, _, _ = make_gate()
    grant, _ = authorized(gate, ["open_application", "type_text"])
    caplog.clear()
    caplog.set_level(logging.DEBUG)

    gate.check("type_text", authority=grant)  # ORDER-AGNOSTIC: a command is pre-checked whole before its first step runs
    gate.check("open_application", authority=grant)
    assert gate.remaining(authority=grant) == 2 and gate.unspent(grant) == 2 and gate.budget_remaining() == 10
    with pytest.raises(ComputerControlError):
        gate.check("scroll", authority=grant)  # not covered by any unspent entry: refused, but it is only a question - the grant stays alive
    with pytest.raises(ComputerControlError):
        gate.check("type_text")  # no authority at all

    assert gate.unspent(grant) == 2 and len(gate._events) == 0 and gate._grants[grant.id].closed is None
    assert [r for r in caplog.records if "allowed" in r.getMessage() or "revoked" in r.getMessage()] == []


def test_a_spent_entry_is_no_longer_covered_in_a_pre_check(os_recorder):
    gate, _, _ = make_gate()
    grant, backend = authorized(gate, ["open_application", "type_text"])
    CALLS["open_application"][0](backend)

    gate.check("type_text", authority=grant)
    with pytest.raises(ComputerControlError):
        gate.check("open_application", authority=grant)  # its one entry is spent
    assert gate.remaining(authority=grant) == 1


# ============================================ 7. nothing is inherited by threads, requests, schedulers or jobs ======================================


def test_a_thread_using_the_shared_backend_gains_nothing_from_a_grant_held_elsewhere(os_recorder):
    gate, _, _ = make_gate()
    grant, demo_backend = authorized(gate, ["type_text"])
    shared = WindowsComputerBackend(gate=gate)  # the backend the rest of the application uses
    outcomes = []

    def background():
        try:
            CALLS["type_text"][0](shared)
            outcomes.append("ran")
        except ComputerControlError as exc:
            outcomes.append(str(exc))

    worker = threading.Thread(target=background)
    worker.start()
    worker.join()

    assert outcomes == [NOT_AUTHORIZED] and os_recorder.calls == [] and gate.unspent(grant) == 1


def test_authority_is_only_what_was_handed_over_by_reference_a_thread_given_the_backend_can_act_one_that_is_not_cannot(os_recorder):
    gate, _, _ = make_gate()
    _, handed = authorized(gate, ["type_text"])
    outcomes = {}

    def run(name, backend):
        try:
            CALLS["type_text"][0](backend)
            outcomes[name] = "ran"
        except ComputerControlError:
            outcomes[name] = "denied"

    stranger = threading.Thread(target=run, args=("stranger", WindowsComputerBackend(gate=gate)))
    stranger.start()
    stranger.join()
    holder = threading.Thread(target=run, args=("holder", handed))
    holder.start()
    holder.join()

    assert outcomes == {"stranger": "denied", "holder": "ran"} and len(os_recorder.calls) == 1


def test_a_copied_context_a_thread_pool_and_an_event_loop_carry_no_authority(os_recorder):
    import asyncio
    import contextvars
    from concurrent.futures import ThreadPoolExecutor

    gate, _, _ = make_gate()
    grant, _ = authorized(gate, ["type_text"] * 3)
    shared = WindowsComputerBackend(gate=gate)
    attempts = []

    def attempt():
        try:
            CALLS["type_text"][0](shared)
            attempts.append("ran")
        except ComputerControlError:
            attempts.append("denied")

    contextvars.copy_context().run(attempt)
    with ThreadPoolExecutor(max_workers=2) as pool:
        pool.submit(attempt).result()
        pool.submit(contextvars.copy_context().run, attempt).result()
    asyncio.run(asyncio.to_thread(attempt))

    assert attempts == ["denied"] * 4 and os_recorder.calls == [] and gate.unspent(grant) == 3


def test_authority_is_never_ambient_the_computer_modules_use_no_context_variables():
    for name in ("authority.py", "gate.py", "issuer.py", "windows_backend.py", "factory.py"):
        source = (APP_DIR / "computer" / name).read_text(encoding="utf-8")
        assert "contextvars" not in source and "ContextVar" not in source and "threading.local" not in source, name


def test_one_backends_grant_is_not_another_backends_even_in_the_same_process(os_recorder):
    gate, _, _ = make_gate()
    grant_a, backend_a = authorized(gate, ["type_text"])
    backend_b = WindowsComputerBackend(gate=gate)

    CALLS["type_text"][0](backend_a)

    denied(CALLS["type_text"][0], backend_b)
    assert len(os_recorder.calls) == 1


# ============================================ 9. revocation, and the global switch ==================================================================


def test_revoking_a_grant_stops_the_very_next_action_and_is_idempotent(os_recorder):
    gate, _, _ = make_gate()
    grant, backend = authorized(gate, ["type_text", "type_text"])
    CALLS["type_text"][0](backend)

    assert gate.revoke(grant) is True and gate.revoke(grant) is False
    denied(CALLS["type_text"][0], backend)

    assert len(os_recorder.calls) == 1 and gate.unspent(grant) == 0
    with pytest.raises(ComputerControlError):
        gate.check("type_text", authority=grant)


def test_revoking_a_forged_or_unknown_grant_changes_nothing(os_recorder):
    gate, _, _ = make_gate()
    real, backend = authorized(gate, ["type_text"])
    lookalike = InputGrant()
    lookalike._id = real.id

    assert gate.revoke(InputGrant()) is False and gate.revoke(lookalike) is False and gate.revoke(None) is False
    CALLS["type_text"][0](backend)  # the real grant never noticed
    assert len(os_recorder.calls) == 1


def test_revoke_all_ends_every_active_grant_at_once(os_recorder):
    gate, _, _ = make_gate()
    pairs = [authorized(gate, ["type_text"]) for _ in range(3)]

    assert gate.revoke_all() == 3 and gate.revoke_all() == 0
    for _, backend in pairs:
        denied(CALLS["type_text"][0], backend)
    assert os_recorder.calls == []


def test_remaining_is_the_smaller_of_the_rate_budget_and_the_grants_unspent_entries(os_recorder):
    gate, _, clock = make_gate(limit=4)
    grant, backend = authorized(gate, ["type_text"] * 6)

    assert gate.remaining(authority=grant) == 4 and gate.remaining() == 0 and gate.budget_remaining() == 4  # no grant given: nothing could be admitted
    for _ in range(3):
        CALLS["type_text"][0](backend)
    assert gate.remaining(authority=grant) == 1
    clock.now += 60.0
    assert gate.remaining(authority=grant) == 3  # budget back to 4, but only 3 entries are left


def test_a_demo_style_run_ends_with_its_grant_revoked_and_nothing_left_usable(os_recorder):
    gate, _, _ = make_gate()
    grant, backend = authorized(gate, ["type_text", "press_key", "scroll"])
    CALLS["type_text"][0](backend)
    try:
        raise RuntimeError("the run failed half-way")
    except RuntimeError:
        pass
    finally:
        gate.revoke(grant)

    denied(CALLS["press_key"][0], backend)
    assert gate.unspent(grant) == 0 and len(os_recorder.calls) == 1


# ============================================ 10. the logs carry no sensitive value ==================================================================


def test_no_log_record_of_any_logger_carries_a_value_a_digest_a_key_or_a_length(os_recorder, caplog):
    caplog.set_level(logging.DEBUG)
    gate, state, clock = make_gate(limit=4)
    steps = [
        InputStep("type_text", (SECRET,)),
        InputStep("press_key", (normalize_key_input("ctrl+shift+esc"),)),
        InputStep("click_mouse", (777, 513, MouseButton.RIGHT)),
        InputStep("open_application", (AllowedApplication.CALCULATOR,)),
        InputStep("scroll", (-1234,)),
        InputStep("move_mouse", (777, 513)),
    ]
    grant = GrantIssuer(gate).issue(steps, ttl_seconds=120, purpose="privacy_test", actor="test")
    backend = WindowsComputerBackend(gate=gate, authority=grant)
    backend.type_text(SECRET)
    backend.press_key("ctrl+shift+esc")
    backend.click_mouse(777, 513, MouseButton.RIGHT)
    backend.open_application(AllowedApplication.CALCULATOR)
    with pytest.raises(ComputerControlError):
        backend.scroll(-1234)  # the 5th action is over the limit of four: a refusal
    clock.now += 60.0
    with pytest.raises(ComputerControlError):
        backend.type_text(SECRET + "-DIFFERENT")  # a mismatch: revokes the grant
    state["enabled"] = False
    with pytest.raises(ComputerControlError):
        backend.move_mouse(777, 513)  # switched off
    registry_text = "\n".join(f"{r.name} {r.getMessage()} | {r.msg} | {r.args!r} | {r.exc_text or ''} | {r.exc_info!r}" for r in caplog.records)
    everything = registry_text + repr(grant) + repr(backend.gate) + str(vars(caplog.records[0]))

    macs = [grant._mac(step.canonical()).hex() for step in steps]
    assert any("grant issued" in r.getMessage() for r in caplog.records) and any("action=type_text" in r.getMessage() for r in caplog.records)
    for value in (SECRET, "DIFFERENT", "777", "513", "1234", "calculator", "ctrl", "shift", "esc", "left", "right", grant.id, grant._key.hex(), "notepad"):
        assert value not in everything, value
    for mac in macs:
        assert mac not in everything and mac[:16] not in everything  # not a digest, not even a prefix of one
    for token in ("chars=", "parts=", "shape=", "digest", "mac=", "key=", "len="):
        assert token not in everything, token  # no metadata derived from an argument, no lengths
    assert {r.name for r in caplog.records} == {"jarvis.computer.input"}


def test_the_audit_lines_name_the_action_the_grant_by_a_short_prefix_and_a_reason_code_only(os_recorder, caplog):
    caplog.set_level(logging.DEBUG, logger="jarvis.computer")
    gate, _, _ = make_gate(limit=1)
    grant, backend = authorized(gate, ["type_text", "press_key"], purpose="notepad_demo", actor="cli")
    short = grant.short_id
    CALLS["type_text"][0](backend)
    with pytest.raises(ComputerControlError):
        CALLS["press_key"][0](backend)  # rate limited
    with pytest.raises(ComputerControlError):
        CALLS["scroll"][0](backend)  # a mismatch (the 2nd entry is a press_key), after the limit check order: mismatch is decided first
    lines = [r.getMessage() for r in caplog.records]

    assert lines == [
        f"computer_input grant issued id={short} entries=2 ttl=120 purpose=notepad_demo actor=cli",
        f"computer_input allowed action=type_text grant={short} entry=1/2 used=1/1",
        "computer_input refused action=press_key reason=rate_limited",
        f"computer_input grant revoked id={short} reason=mismatch",
        "computer_input refused action=scroll reason=mismatch",
    ]
    for record in caplog.records:
        assert set(record.__dict__) - (set(logging.LogRecord("n", logging.INFO, "p", 1, "m", None, None).__dict__) | {"message", "asctime", "taskName"}) == set()
    assert grant.id not in "\n".join(lines)


# ============================================ the static guards: accidental architectural drift (NOT a security boundary) ============================


def app_trees():
    for path in sorted(APP_DIR.rglob("*.py")):
        yield path.relative_to(APP_DIR).as_posix(), path.read_text(encoding="utf-8"), ast.parse(path.read_text(encoding="utf-8"))


def test_only_the_notepad_demos_command_line_imports_the_grant_issuer():
    importers = set()
    for rel, source, tree in app_trees():
        for node in ast.walk(tree):
            if isinstance(node, ast.Import) and any(alias.name.startswith("app.computer.issuer") for alias in node.names):
                importers.add(rel)
            if isinstance(node, ast.ImportFrom) and (
                (node.module or "").startswith("app.computer.issuer")
                or (node.module == "app.computer" and any(alias.name == "issuer" for alias in node.names))
                or (node.level and (node.module or "").endswith("issuer"))
            ):
                importers.add(rel)
        if rel != "computer/issuer.py" and ("computer.issuer" in source or "import_module" in source and "issuer" in source):
            importers.add(rel)  # an indirect import by name counts too
    assert importers == {"demos/notepad_note.py"}  # no chat, Core, tool, Job, scheduler, automation, proposal or main module may mint a grant


def test_only_the_issuer_registers_grants_and_only_the_issuer_builds_them():
    registering, constructing = set(), set()
    for rel, _, tree in app_trees():
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr == "_register_grant":
                registering.add(rel)
            if isinstance(node, ast.Call) and getattr(node.func, "id", getattr(node.func, "attr", None)) == "InputGrant":
                constructing.add(rel)
    assert registering == {"computer/issuer.py"} and constructing == {"computer/issuer.py"}


def test_the_gates_admission_is_called_only_by_the_gate_module_and_the_backend_helper_only_by_the_two_backends():
    admits, helper = set(), set()
    for rel, _, tree in app_trees():
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "admit":
                admits.add(rel)
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "admit_input":
                helper.add(rel)
    assert admits == {"computer/gate.py"}
    assert helper == {"computer/gate.py", "computer/windows_backend.py"}


INPUT_CAPABLE = frozenset({
    "write", "typewrite", "press", "hotkey", "keyDown", "keyUp", "moveTo", "moveRel", "move", "click", "doubleClick", "tripleClick", "rightClick",
    "middleClick", "mouseDown", "mouseUp", "dragTo", "dragRel", "drag", "scroll", "hscroll", "vscroll",
})


def test_input_capable_pyautogui_calls_exist_only_in_the_one_approved_backend():
    offenders, importers = [], set()
    for rel, _, tree in app_trees():
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] == "pyautogui":
                        importers.add(rel)
                        if alias.asname is not None and rel != "computer/windows_backend.py":
                            offenders.append((rel, f"import pyautogui as {alias.asname}"))
            if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "pyautogui":
                importers.add(rel)
                offenders.append((rel, "from pyautogui import ..."))  # would hide which functions are called
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "pyautogui" and node.attr in INPUT_CAPABLE:
                if rel != "computer/windows_backend.py":
                    offenders.append((rel, node.attr))
    assert offenders == []
    assert importers == {"computer/windows_backend.py", "observation/screen.py"}  # the second only ever takes a screenshot (read-only)
    screen = next(tree for rel, _, tree in app_trees() if rel == "observation/screen.py")
    used = {n.attr for n in ast.walk(screen) if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "pyautogui"}
    assert used <= {"screenshot"}


def test_no_module_sends_raw_keyboard_or_mouse_events_by_any_other_route():
    forbidden_attributes = {"keybd_event", "mouse_event", "SendInput", "SetCursorPos", "PostMessage", "PostMessageW", "SendMessage"}
    forbidden_modules = {"pydirectinput", "pynput", "keyboard", "mouse", "autoit", "pywinauto"}
    offenders = []
    for rel, _, tree in app_trees():
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in forbidden_attributes:
                offenders.append((rel, node.attr))
            if isinstance(node, ast.Import):
                offenders += [(rel, a.name) for a in node.names if a.name.split(".")[0] in forbidden_modules]
            if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] in forbidden_modules:
                offenders.append((rel, node.module))
    assert offenders == []


# ============================================ 11. the permission model is untouched ===============================================================


def test_no_permission_level_changed_the_grant_is_an_additional_layer():
    registry = create_computer_tools_registry(backend=WindowsComputerBackend(gate=make_gate()[0]))

    for name in ACTIONS:
        tool = registry.get(name)
        assert tool.permission is PermissionLevel.SAFE_ACTION and tool.automatable is False, name
    assert registry.get("get_screen_info").permission is PermissionLevel.READ
    assert JOB_COMPUTER_TOOL_NAMES == frozenset({"open_url", "scroll", "get_active_window", "switch_window"})


def test_the_production_job_registry_still_holds_no_keyboard_mouse_or_launch_tool():
    import app.main as live_app

    for name in ACTIONS:
        assert live_app.core.tools.has(name) is (name == "scroll")  # scroll is the one input tool Jobs always had; it is gated now


def test_the_readme_documents_the_grant_model_its_limits_and_the_gated_window_actions():
    readme = (pathlib.Path(__file__).resolve().parent.parent / "README.md").read_text(encoding="utf-8")
    section = readme[readme.index("### Computer Input Gate"):]
    section = section[: section.index("\n### ", 10)] if "\n### " in section[10:] else section

    for needle in ("authorization grant", "switch_window", "open_url", "not a security boundary", "revoke", "exact window title", "not a window instance", "no topmost-window fallback"):
        assert needle in section.lower(), needle
    assert "need a later security review" not in section.lower() and "not** behind the gate" not in section  # open_url and switch_window are no longer described as outside the gate


def test_the_wrappers_invalid_key_is_a_validation_failure_not_a_grant_mismatch_just_like_the_real_backend(os_recorder):
    gate, _, _ = make_gate()
    grant = issue(gate, ["press_key"])
    real = WindowsComputerBackend(gate=gate, authority=grant)
    calls = []

    class Inner(ComputerBackend):
        def get_active_window(self): return WindowInfo("t", "p.exe", False)
        def get_screen_info(self): return ScreenInfo(1, 1)
        def open_application(self, app): pass
        def open_url(self, url): pass
        def switch_window(self, title, process): return WindowInfo(title, process, False)
        def type_text(self, text): pass
        def press_key(self, key): calls.append(key)
        def move_mouse(self, x, y): pass
        def click_mouse(self, x, y, button): pass
        def double_click(self, x, y, button): pass
        def scroll(self, amount): pass

    wrapper = GatedComputerBackend(Inner(), gate, authority=grant)

    for backend in (real, wrapper):
        for invalid in ("a", "ctrl+alt", "not-a-key", ""):
            with pytest.raises(ComputerControlError) as caught:
                backend.press_key(invalid)
            assert str(caught.value) != NOT_AUTHORIZED  # a validation message, never an authorization one
    assert gate.unspent(grant) == 1 and gate._grants[grant.id].closed is None and len(gate._events) == 0 and calls == []
    wrapper.press_key("ctrl+n")  # the grant is entirely intact: the one call it approved still goes through, once
    assert calls == ["ctrl+n"] and gate.unspent(grant) == 0 and len(gate._events) == 1


def test_closed_grant_records_do_not_pile_up_when_grants_are_issued_and_revoked_in_a_loop():
    gate, _, _ = make_gate()

    for _ in range(500):
        gate.revoke(issue(gate, ["type_text"]))

    assert len(gate._grants) <= 4 * 8  # bounded by the cap, however long the loop


# ============================== hardening: failure AFTER admission, a refusal MID-sequence, and the admission step itself raising ==========================

#: action -> (the FakeOS attribute that action finally calls, the exception that call raises). open_application's Popen failure is turned into a ComputerControlError by the backend.
OS_FAILURE = {
    "type_text": ("write", RuntimeError("the input library crashed")),
    "press_key": ("hotkey", RuntimeError("the input library crashed")),
    "move_mouse": ("moveTo", RuntimeError("the input library crashed")),
    "click_mouse": ("click", RuntimeError("the input library crashed")),
    "double_click": ("doubleClick", RuntimeError("the input library crashed")),
    "scroll": ("scroll", RuntimeError("the input library crashed")),
    "open_application": ("Popen", OSError("not installed")),
}


@pytest.mark.parametrize("action", ACTIONS)
def test_an_os_failure_after_admission_keeps_the_entry_and_the_slot_spent_so_a_retry_cannot_reuse_them(action, os_recorder, monkeypatch):
    gate, _, _ = make_gate()
    grant, backend = authorized(gate, [action])  # exactly one entry
    attribute, failure = OS_FAILURE[action]
    attempts = []

    def failing_os_call(*args, **kwargs):
        attempts.append(1)
        raise failure

    monkeypatch.setattr(os_recorder, attribute, failing_os_call)

    with pytest.raises((RuntimeError, ComputerControlError)) as first:
        CALLS[action][0](backend)  # admitted, then the OS call itself fails
    assert str(first.value) != NOT_AUTHORIZED and len(attempts) == 1  # it really got as far as the OS call: this was an OS failure, not a refusal

    state = gate._grants[grant.id]
    assert state.index == 1 and state.closed == "exhausted" and len(gate._events) == 1  # the entry and the slot were spent by the admission
    denied(CALLS[action][0], backend)  # the retry: refused, the entry is gone
    denied(CALLS[action][0], WindowsComputerBackend(gate=gate, authority=grant))  # and no fresh backend instance can reuse it either
    assert len(attempts) == 1 and os_recorder.calls == [] and len(gate._events) == 1  # the OS function was attempted exactly once, and nothing was recorded after


def test_an_os_failure_in_the_middle_of_a_run_spends_that_entry_but_leaves_the_next_ones_intact(os_recorder, monkeypatch):
    gate, _, _ = make_gate()
    grant, backend = authorized(gate, ["type_text", "press_key", "scroll"])
    real_write, outcomes = os_recorder.write, []
    monkeypatch.setattr(os_recorder, "write", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("the input library crashed")))

    with pytest.raises(RuntimeError):
        CALLS["type_text"][0](backend)  # entry 1 spent, then the OS failed
    assert gate.unspent(grant) == 2 and gate._grants[grant.id].closed is None  # a failure of the OS is not a mismatch: the grant stays alive
    CALLS["press_key"][0](backend)  # the NEXT approved entry still works, in order
    CALLS["scroll"][0](backend)
    denied(CALLS["type_text"][0], backend)  # while the failed entry is not given back

    assert [c[0] for c in os_recorder.calls] == ["hotkey", "scroll"] and gate.unspent(grant) == 0 and len(gate._events) == 3


@pytest.mark.parametrize("bad_position", [0, 1, 2, 3])
def test_a_refusal_in_the_middle_of_a_sequence_stops_every_later_action_even_the_correctly_approved_ones(bad_position, os_recorder):
    plan = ["open_application", "press_key", "type_text", "scroll"]
    gate, _, _ = make_gate()
    grant, backend = authorized(gate, plan)
    calls_made = []

    def attempt(action, call):
        try:
            call(backend)
            calls_made.append((action, "ran"))
        except ComputerControlError:
            calls_made.append((action, "refused"))

    for position, action in enumerate(plan):
        attempt(action, CALLS[action][2] if position == bad_position else CALLS[action][0])  # one call carries a wrong argument
    for action in plan:  # a persistent caller that ignores the refusal and keeps going, with every correct call in turn
        attempt(action, CALLS[action][0])

    ran = [a for a, outcome in calls_made if outcome == "ran"]
    assert ran == plan[:bad_position]  # only the calls BEFORE the bad one ever happened
    assert all(outcome == "refused" for _, outcome in calls_made[bad_position:])  # the bad call and EVERY later one - original or retried - was refused
    assert len(os_recorder.calls) == bad_position and len(gate._events) == bad_position
    assert gate._grants[grant.id].closed == "mismatch" and gate.unspent(grant) == 0
    assert gate._grants[grant.id].index == bad_position  # and not one entry was consumed by a refusal


def test_a_chain_whose_second_step_is_refused_at_the_backend_never_runs_the_third(os_recorder, caplog):
    """The pre-check is order-agnostic and looks at action NAMES, so this whole chain passes it. Step 2 types something other than what the grant approved: refused at the
    backend, the grant is revoked, and step 3 (the Enter) never reaches the OS - the executor stops, and even without it the revoked grant would refuse."""
    caplog.set_level(logging.DEBUG)
    gate, _, _ = make_gate()
    grant = GrantIssuer(gate).issue(
        [InputStep("press_key", (normalize_key_input("ctrl+n"),)), InputStep("type_text", ("approved text",)), InputStep("press_key", (normalize_key_input("enter"),))],
        ttl_seconds=120, purpose="chain_test", actor="test",
    )
    say = chat_over(WindowsComputerBackend(gate=gate, authority=grant))

    reply = say("use tool press_key ctrl+n, then use tool type_text something else, then use tool press_key enter")

    assert reply.startswith("1. Pressed 'ctrl+n'.\n2. ") and "needs an approved authorization" in reply
    assert "Stopped after step 2; step(s) 3 were not run." in reply
    assert os_recorder.calls == [("hotkey", ("ctrl", "n"))]  # step 1 only: no typing, no Enter
    assert gate._grants[grant.id].closed == "mismatch" and gate._grants[grant.id].index == 1 and len(gate._events) == 1
    denied(CALLS["type_text"][0], WindowsComputerBackend(gate=gate, authority=grant))  # the grant is over, whoever holds it


def test_a_chain_refused_by_the_rate_limit_mid_way_runs_nothing_after_the_refusal(os_recorder):
    gate, _, _ = make_gate(limit=1)
    grant = GrantIssuer(gate).issue(
        [InputStep("press_key", (normalize_key_input("ctrl+n"),)), InputStep("type_text", ("approved text",)), InputStep("press_key", (normalize_key_input("enter"),))],
        ttl_seconds=120, purpose="chain_test", actor="test",
    )
    say = chat_over(WindowsComputerBackend(gate=gate, authority=grant))

    reply = say("use tool press_key ctrl+n, then use tool type_text approved text, then use tool press_key enter")

    assert "limited to 1 actions per minute" in reply and "Stopped after step 2; step(s) 3 were not run." in reply
    assert os_recorder.calls == [("hotkey", ("ctrl", "n"))]
    assert gate.unspent(grant) == 2 and gate._grants[grant.id].closed is None  # a limit refusal is not a mismatch: the rest of the grant is intact for later


# ------------------------------------------------ the admission step itself raising: fail closed, nothing half-done ---------------------------------------------


class FlakyClock(Clock):
    """A monotonic clock that can be made to fail."""

    def __init__(self, now: float = 1000.0):
        super().__init__(now)
        self.failing = False
        self.calls_while_failing = 0

    def __call__(self) -> float:
        if self.failing:
            self.calls_while_failing += 1
            raise RuntimeError("the clock failed")
        return self.now


def lock_is_free(gate):
    got = gate._lock.acquire(blocking=False)
    if got:
        gate._lock.release()
    return got


def snapshot(gate, grant):
    state = gate._grants[grant.id]
    return state.index, state.closed, len(gate._events), gate.unspent(grant) if gate._clock.failing is False else None


@pytest.mark.parametrize("action", ACTIONS)
def test_a_failing_clock_inside_admission_fails_closed_with_no_os_call_and_no_state_change(action, os_recorder):
    clock = FlakyClock()
    gate, _, _ = make_gate(clock=clock)
    grant, backend = authorized(gate, [action, "type_text"])
    before = snapshot(gate, grant)
    clock.failing = True

    with pytest.raises(RuntimeError, match="the clock failed"):
        CALLS[action][0](backend)  # the admission itself raised: the call must not reach the OS

    clock.failing = False
    assert clock.calls_while_failing >= 1 and os_recorder.calls == []  # it really was the admission that failed, and nothing was done
    assert snapshot(gate, grant) == before == (0, None, 0, 2)  # no entry spent, no slot taken, the grant neither revoked nor altered
    assert lock_is_free(gate)  # and the lock was not left held
    CALLS[action][0](backend)  # once the clock is back, the very same approved call is admitted normally, spending exactly one entry and one slot
    assert len(os_recorder.calls) == 1 and snapshot(gate, grant) == (1, None, 1, 1)


def test_a_failing_clock_in_the_non_consuming_questions_changes_nothing_either(os_recorder):
    clock = FlakyClock()
    gate, _, _ = make_gate(clock=clock)
    grant, _ = authorized(gate, ["type_text", "press_key"])
    before = snapshot(gate, grant)
    clock.failing = True

    for question in (lambda: gate.check("type_text", authority=grant), lambda: gate.check_budget(), lambda: gate.remaining(authority=grant),
                     lambda: gate.budget_remaining(), lambda: gate.unspent(grant)):
        with pytest.raises(RuntimeError, match="the clock failed"):
            question()  # a question that cannot read the time gives no answer - it does not guess "yes"

    clock.failing = False
    assert snapshot(gate, grant) == before and lock_is_free(gate) and os_recorder.calls == []
    assert gate.remaining(authority=grant) == 2  # everything is exactly as it was


def test_a_failing_clock_while_issuing_registers_nothing_and_a_later_issue_works():
    clock = FlakyClock()
    gate, _, _ = make_gate(clock=clock)
    clock.failing = True

    with pytest.raises(RuntimeError, match="the clock failed"):
        issue(gate, ["type_text"])

    clock.failing = False
    assert gate._grants == {} and len(gate._events) == 0 and lock_is_free(gate)  # no half-registered grant
    grant = issue(gate, ["type_text"])
    assert gate.unspent(grant) == 1


def test_a_failing_clock_during_admission_cannot_be_used_to_slip_an_action_past_the_limit_or_the_grant(os_recorder):
    clock = FlakyClock()
    gate, _, _ = make_gate(limit=2, clock=clock)
    grant, backend = authorized(gate, ["type_text"] * 4)
    outcomes = []

    for attempt_number in range(6):
        clock.failing = attempt_number % 2 == 0  # the clock fails on every other attempt
        try:
            CALLS["type_text"][0](backend)
            outcomes.append("ran")
        except ComputerControlError:  # (a subclass of RuntimeError, so it must be caught first)
            outcomes.append("refused")
        except RuntimeError:
            outcomes.append("clock")

    clock.failing = False
    assert outcomes == ["clock", "ran", "clock", "ran", "clock", "refused"]  # two admitted, the third over the limit; failures admitted nothing
    assert len(os_recorder.calls) == 2 and len(gate._events) == 2 and gate._grants[grant.id].index == 2  # entries and slots moved together, never apart


# ============================== the switch and the limit are CALLABLES read on every call: what happens when they raise or misbehave while a grant is active ============


class Flaky:
    """A configuration callable (the switch, or the limit) that can be made to raise or to return something odd, and counts how often it was read."""

    def __init__(self, value):
        self.value = value
        self.mode = "ok"  # ok | raise | odd
        self.odd = None
        self.failure: BaseException = RuntimeError("the configuration could not be read")
        self.reads = 0

    def __call__(self):
        self.reads += 1
        if self.mode == "raise":
            raise self.failure
        if self.mode == "odd":
            return self.odd
        return self.value


def flaky_gate(limit=10):
    enabled, limit_callable, clock = Flaky(True), Flaky(limit), Clock()
    return InputGate(enabled, limit_callable, clock=clock), enabled, limit_callable, clock


def grant_state(gate, grant):
    state = gate._grants[grant.id]
    return state.index, state.closed, len(gate._events)


LIMITED_TO_ZERO = "limited to 0 actions per minute"


@pytest.mark.parametrize("action", ACTIONS)
def test_a_switch_that_cannot_be_read_refuses_everything_and_ends_every_grant_for_good(action, os_recorder):
    gate, enabled, _, _ = flaky_gate()
    grant, backend = authorized(gate, [action, "type_text"])
    unrelated, _ = authorized(gate, ["scroll"])
    enabled.mode = "raise"

    denied(CALLS[action][0], backend, text=SWITCHED_OFF)  # fail closed: an unreadable switch is a switch that is off

    assert os_recorder.calls == [] and len(gate._events) == 0 and lock_is_free(gate)
    assert grant_state(gate, grant) == (0, "switch_off", 0) and grant_state(gate, unrelated) == (0, "switch_off", 0)  # no entry spent, no slot taken; EVERY grant ended
    enabled.mode = "ok"  # the callable recovers ...
    denied(CALLS[action][0], backend)  # ... but the grants do not come back with it
    assert os_recorder.calls == [] and gate.unspent(grant) == 0 and gate.unspent(unrelated) == 0 and len(gate._events) == 0
    fresh, fresh_backend = authorized(gate, [action])  # a new approval works normally once the switch is readable again
    CALLS[action][0](fresh_backend)
    assert len(os_recorder.calls) == 1 and len(gate._events) == 1 and gate._grants[fresh.id].closed == "exhausted"


def test_the_non_consuming_questions_with_an_unreadable_switch_refuse_or_report_zero_but_change_nothing(os_recorder, caplog):
    gate, enabled, _, _ = flaky_gate()
    grant, backend = authorized(gate, ["type_text", "press_key"])
    before = grant_state(gate, grant)
    enabled.mode = "raise"
    caplog.clear()
    caplog.set_level(logging.DEBUG)

    with pytest.raises(ComputerControlError, match="switched off"):
        gate.check("type_text", authority=grant)
    with pytest.raises(ComputerControlError, match="switched off"):
        gate.check_budget()
    assert gate.remaining(authority=grant) == 0 and gate.remaining() == 0 and gate.budget_remaining() == 0  # "nothing could be admitted", never "unknown"
    assert gate.unspent(grant) == 2  # (it describes the grant, not the switch)

    assert grant_state(gate, grant) == before and lock_is_free(gate) and os_recorder.calls == []  # only ADMISSION ends grants; a question never does
    assert [r for r in caplog.records if "revoked" in r.getMessage() or "allowed" in r.getMessage()] == []
    enabled.mode = "ok"
    gate.check("type_text", authority=grant)
    assert gate.remaining(authority=grant) == 2
    CALLS["type_text"][0](backend)  # and the grant, never touched, works exactly as approved
    assert len(os_recorder.calls) == 1 and grant_state(gate, grant) == (1, None, 1)


def test_a_switch_that_cannot_be_read_while_issuing_registers_nothing_and_a_later_issue_works():
    gate, enabled, _, _ = flaky_gate()
    enabled.mode = "raise"

    with pytest.raises(ComputerControlError, match="switched off"):
        issue(gate, ["type_text"])

    assert gate._grants == {} and len(gate._events) == 0 and lock_is_free(gate)
    enabled.mode = "ok"
    assert gate.unspent(issue(gate, ["type_text"])) == 1


@pytest.mark.parametrize("odd", [1, "true", "yes", [True], None, 1.0, "True"], ids=repr)
def test_only_a_real_true_counts_as_the_switch_being_on_anything_else_is_off_and_ends_the_grants(odd, os_recorder):
    gate, enabled, _, _ = flaky_gate()
    grant, backend = authorized(gate, ["type_text"])
    enabled.mode, enabled.odd = "odd", odd

    denied(CALLS["type_text"][0], backend, text=SWITCHED_OFF)

    assert os_recorder.calls == [] and grant_state(gate, grant) == (0, "switch_off", 0)
    enabled.mode = "ok"
    denied(CALLS["type_text"][0], backend)


@pytest.mark.parametrize("action", ACTIONS)
def test_a_limit_that_cannot_be_read_refuses_without_consuming_anything_and_the_grant_survives(action, os_recorder):
    gate, _, limit, _ = flaky_gate()
    grant, backend = authorized(gate, [action, "type_text"])
    limit.mode = "raise"

    denied(CALLS[action][0], backend, text=f"Computer input is limited to 0 actions per minute, and that limit has been reached. Try again in 60 second(s).")

    assert os_recorder.calls == [] and lock_is_free(gate)
    assert grant_state(gate, grant) == (0, None, 0) and gate.unspent(grant) == 2  # no entry spent, no slot taken, and a limit trouble does not revoke
    limit.mode = "ok"  # the callable recovers: the same approved call is admitted, spending exactly one entry and one slot
    CALLS[action][0](backend)
    assert len(os_recorder.calls) == 1 and grant_state(gate, grant) == (1, None, 1)


@pytest.mark.parametrize("odd", [-1, True, False, "10", 10.0, 3.5, None, [10], float("nan")], ids=repr)
def test_a_limit_that_is_not_a_plain_non_negative_whole_number_allows_nothing_and_consumes_nothing(odd, os_recorder):
    gate, _, limit, _ = flaky_gate()
    grant, backend = authorized(gate, ["type_text", "type_text"])
    limit.mode, limit.odd = "odd", odd

    with pytest.raises(ComputerControlError, match=LIMITED_TO_ZERO):
        CALLS["type_text"][0](backend)

    assert os_recorder.calls == [] and grant_state(gate, grant) == (0, None, 0)
    assert gate.remaining(authority=grant) == 0 and gate.budget_remaining() == 0
    limit.mode = "ok"
    CALLS["type_text"][0](backend)
    assert len(os_recorder.calls) == 1 and grant_state(gate, grant) == (1, None, 1)


def test_the_non_consuming_questions_with_an_unreadable_limit_refuse_or_report_zero_but_change_nothing(os_recorder):
    gate, _, limit, _ = flaky_gate()
    grant, _ = authorized(gate, ["type_text", "press_key"])
    before = grant_state(gate, grant)
    limit.mode = "raise"

    with pytest.raises(ComputerControlError, match=LIMITED_TO_ZERO):
        gate.check("type_text", authority=grant)
    with pytest.raises(ComputerControlError, match=LIMITED_TO_ZERO):
        gate.check_budget()
    assert gate.remaining(authority=grant) == 0 and gate.budget_remaining() == 0 and gate.unspent(grant) == 2

    assert grant_state(gate, grant) == before and lock_is_free(gate) and os_recorder.calls == []
    limit.mode = "ok"
    gate.check("type_text", authority=grant)
    assert gate.remaining(authority=grant) == 2 and gate.budget_remaining() == 10


def test_a_wrong_call_while_the_limit_is_unreadable_still_ends_the_grant_because_the_match_is_decided_before_the_limit(os_recorder):
    gate, _, limit, _ = flaky_gate()
    grant, backend = authorized(gate, ["type_text", "press_key"])
    limit.mode = "raise"

    denied(CALLS["type_text"][2], backend)  # not what was approved

    limit.mode = "ok"
    assert grant_state(gate, grant) == (0, "mismatch", 0) and os_recorder.calls == []
    denied(CALLS["type_text"][0], backend)  # the approved call is refused now: the grant is over


def test_a_limit_that_flaps_between_unreadable_and_fine_never_admits_more_than_the_grant_and_keeps_entries_and_slots_together(os_recorder):
    gate, _, limit, _ = flaky_gate(limit=10)
    grant, backend = authorized(gate, ["type_text"] * 3)
    outcomes = []

    for attempt_number in range(8):
        limit.mode = "raise" if attempt_number % 2 == 0 else "ok"
        try:
            CALLS["type_text"][0](backend)
            outcomes.append("ran")
        except ComputerControlError:
            outcomes.append("refused")
        assert grant_state(gate, grant)[0] == len(gate._events) == len([o for o in outcomes if o == "ran"])  # entries and slots move together, always

    assert outcomes == ["refused", "ran", "refused", "ran", "refused", "ran", "refused", "refused"]  # three entries, three admissions; the 8th: the grant is spent
    assert len(os_recorder.calls) == 3 and gate._grants[grant.id].closed == "exhausted"


def test_both_callables_unreadable_at_once_is_the_switch_message_and_everything_stays_denied(os_recorder):
    gate, enabled, limit, _ = flaky_gate()
    grant, backend = authorized(gate, ["type_text"])
    enabled.mode = limit.mode = "raise"

    denied(CALLS["type_text"][0], backend, text=SWITCHED_OFF)  # the switch is read first

    enabled.mode = limit.mode = "ok"
    assert os_recorder.calls == [] and grant_state(gate, grant) == (0, "switch_off", 0)
    denied(CALLS["type_text"][0], backend)


@pytest.mark.parametrize("which", ["enabled", "limit"])
@pytest.mark.parametrize("failure", [KeyboardInterrupt(), SystemExit(3)], ids=["KeyboardInterrupt", "SystemExit"])
def test_an_exception_that_is_not_an_ordinary_error_from_a_callable_propagates_and_leaves_everything_untouched(which, failure, os_recorder):
    """Ctrl+C must never be swallowed by the gate's fail-closed handling. It propagates out of the admission, BEFORE any OS call, with the lock released and no state half-changed."""
    gate, enabled, limit, _ = flaky_gate()
    grant, backend = authorized(gate, ["type_text", "press_key"])
    flaky = enabled if which == "enabled" else limit
    flaky.mode, flaky.failure = "raise", failure
    before = grant_state(gate, grant)

    with pytest.raises(type(failure)):
        CALLS["type_text"][0](backend)

    assert os_recorder.calls == [] and lock_is_free(gate) and grant_state(gate, grant) == before == (0, None, 0)
    flaky.mode = "ok"
    CALLS["type_text"][0](backend)  # nothing was lost: the approved call is admitted normally afterwards
    assert len(os_recorder.calls) == 1 and grant_state(gate, grant) == (1, None, 1)


# ============================================ unencodable text (a lone surrogate) through the REAL authorization paths ====================================


def surrogate_free(text: str) -> bool:
    return not any(0xD800 <= ord(character) <= 0xDFFF for character in text)


def test_a_plan_with_unencodable_text_cannot_be_issued_a_grant_and_registers_nothing():
    gate, _, _ = make_gate()
    issuer = GrantIssuer(gate)

    with pytest.raises(AuthorityError) as caught:
        issuer.issue([InputStep("type_text", ("a\ud800b",))], ttl_seconds=60, purpose="test", actor="test")
    with pytest.raises(AuthorityError):  # all or nothing: a good entry in front of the bad one is not registered either
        issuer.issue([InputStep("type_text", ("fine",)), InputStep("open_url", ("https://example.com/\ud800",))], ttl_seconds=60, purpose="test", actor="test")

    assert type(caught.value) is AuthorityError and surrogate_free(str(caught.value))
    assert gate._grants == {} and len(gate._events) == 0
    assert issue(gate, ["type_text"]) is not None  # and the issuer is perfectly usable afterwards


UNENCODABLE_CALLS = {
    "type_text": (lambda b: b.type_text("a\ud800b"), InputStep("type_text", ("hello",))),
    "open_url": (lambda b: b.open_url("https://example.com/a\ud800"), InputStep("open_url", ("https://example.com/ok",))),
}


@pytest.mark.parametrize("action", list(UNENCODABLE_CALLS))
def test_unencodable_text_through_the_real_backend_is_refused_before_the_gate_and_consumes_no_entry_and_no_slot(action, os_recorder, caplog):
    caplog.set_level(logging.DEBUG, logger="jarvis.computer")
    call, step = UNENCODABLE_CALLS[action]
    gate, _, _ = make_gate()
    grant = GrantIssuer(gate).issue([step], ttl_seconds=120, purpose="test", actor="test")
    backend = WindowsComputerBackend(gate=gate, authority=grant)

    denied(call, backend)  # exactly the standard "needs an approved authorization" refusal: a ComputerControlError, not a UnicodeEncodeError

    assert os_recorder.calls == []  # no keystroke, no browser
    state = gate._grants[grant.id]
    assert state.closed is None and state.index == 0 and gate.unspent(grant) == 1 and len(gate._events) == 0  # the grant is intact: this was never a match, and not a mismatch
    refusals = [r.getMessage() for r in caplog.records if r.name == "jarvis.computer.input" and "refused" in r.getMessage()]
    assert refusals == [f"computer_input refused action={action} reason=invalid_arguments"]  # audited like any other invalid argument ...
    assert all(surrogate_free(r.getMessage()) and "ud800" not in r.getMessage().lower() for r in caplog.records)  # ... and the value is never logged


def test_after_an_unencodable_attempt_the_one_call_the_grant_approved_still_goes_through_exactly_once(os_recorder):
    gate, _, _ = make_gate()
    grant, backend = authorized(gate, ["type_text"])

    denied(lambda b: b.type_text("a\ud800b"), backend)
    backend.type_text("hello")  # the approved text: the entry was never touched

    assert os_recorder.calls == [("write", "hello")] and gate.unspent(grant) == 0 and len(gate._events) == 1


def test_unencodable_text_through_the_wrapper_and_through_the_tools_is_refused_with_nothing_reaching_a_backend(os_recorder):
    from app.computer.tools import TypeTextTool

    gate, _, _ = make_gate()
    seen = []

    class Inner(ComputerBackend):
        def get_active_window(self): return WindowInfo("t", "p.exe", False)
        def get_screen_info(self): return ScreenInfo(1, 1)
        def open_application(self, app): seen.append("open_application")
        def open_url(self, url): seen.append("open_url")
        def switch_window(self, title, process): seen.append("switch_window"); return WindowInfo(title, process, False)
        def type_text(self, text): seen.append("type_text")
        def press_key(self, key): seen.append("press_key")
        def move_mouse(self, x, y): seen.append("move_mouse")
        def click_mouse(self, x, y, button): seen.append("click_mouse")
        def double_click(self, x, y, button): seen.append("double_click")
        def scroll(self, amount): seen.append("scroll")

    grant = issue(gate, ["type_text"])
    wrapper = GatedComputerBackend(Inner(), gate, authority=grant)
    denied(lambda b: b.type_text("a\ud800b"), wrapper)
    assert seen == [] and gate.unspent(grant) == 1 and len(gate._events) == 0

    backend = WindowsComputerBackend(gate=gate, authority=grant)
    result = TypeTextTool(backend).execute(text="a\ud800b")  # the tool reports it as an error result, never an exception
    registry_result = create_computer_tools_registry(backend=backend).execute("type_text", params={"text": "a\ud800b"})

    for outcome in (result, registry_result):
        assert outcome.status is ToolResultStatus.ERROR and NOT_AUTHORIZED in outcome.message and surrogate_free(outcome.message)
    assert os_recorder.calls == [] and gate.unspent(grant) == 1 and len(gate._events) == 0
