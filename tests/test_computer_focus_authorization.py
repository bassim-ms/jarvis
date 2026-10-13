"""
`open_url` and `switch_window` need an authorization grant, like every other action that acts on the desktop.

Both used to be listed in `UNPROTECTED_ACTIONS`: they needed neither the switch nor a grant, yet each one changes which window is in front. Now:

  * `open_url(url)`                 the grant binds the EXACT URL string handed to the browser.
  * `switch_window(title, process)` the grant binds the EXACT window title and the EXACT process name. The window is found by one enumeration, AFTER admission, and only an exact
                                    title-plus-process match is switched to: zero matches and more than one match are both errors, with no focus operation, no fallback and
                                    no second look. The entry is spent at admission, before the enumeration, so a failed lookup is a spent entry and is never retried.
  * The shared backend carries no grant, so chat, chains and Jobs - which keep the tools registered - are refused at execution. Registration grants nothing.

Everything here runs on recorders: the browser, the window list and the focus calls are replaced by objects that only write an event, and the fake input library's tripwires
cover the real ones. "Refused before any operation" means the event list holds the admission and nothing else.
"""

import ast
import pathlib
from types import SimpleNamespace

import pytest

import fake_input_library as fake
import grant_support as gs
from app.computer import windows_backend
from app.computer.authority import AuthorityError, canonical_input
from app.computer.base import (
    AllowedApplication,
    ComputerBackend,
    ComputerControlError,
    MouseButton,
    WindowInfo,
    exact_window_matches,
    validate_process_name,
)
from app.computer.defaults import JOB_COMPUTER_TOOL_NAMES, create_computer_tools_registry, register_job_computer_tools
from app.computer.factory import create_default_computer_backend
from app.computer.gate import (
    GATED_METHODS,
    NOT_AUTHORIZED,
    SWITCHED_OFF,
    UNGATED_METHODS,
    UNPROTECTED_ACTIONS,
    GatedComputerBackend,
    InputGate,
    _limited,
    default_input_gate,
)
from app.computer.tools import JobOpenUrlTool, OpenUrlTool, SwitchWindowTool
from app.computer.windows_backend import WindowsComputerBackend
from app.config import settings
from app.core import JarvisCore
from app.memory.sqlite_store import SQLiteMemoryStore
from app.providers.base import AIProvider
from app.tools.base import ToolArgumentError, ToolResultStatus
from app.tools.decision import ToolCallBuilder, ToolCallError, ToolCallProvider
from app.tools.registry import ToolRegistry

APP_DIR = pathlib.Path(__file__).resolve().parent.parent / "app"
URL = "https://example.com/a?x=1#top"
TITLE, PROCESS = "Untitled - Notepad", "notepad.exe"
NP = (11, "Untitled - Notepad", "Notepad.exe")  # (hwnd, title, process) as the window list reports it: the file name keeps its case
ACTIONS = ("open_url", "switch_window")
ENTRY = {"open_url": ("open_url", URL), "switch_window": ("switch_window", TITLE, PROCESS)}
CALL = {"open_url": lambda backend: backend.open_url(URL), "switch_window": lambda backend: backend.switch_window(TITLE, PROCESS)}


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


class World:
    """Everything the two actions touch, as recorders. `events` is the order in which things happened."""

    def __init__(self, windows=(), foreground=None):
        self.events: list[tuple] = []
        self.windows = list(windows)
        self.foreground = foreground
        self.refuse_foreground = False

    def enumerate(self):
        self.events.append(("enumerate",))
        return list(self.windows)

    def bring(self, hwnd):
        self.events.append(("bring", hwnd))
        if not self.refuse_foreground:
            self.foreground = hwnd

    @property
    def os_events(self):
        """What happened to the desktop or the browser (everything except the gate being asked)."""
        return [event for event in self.events if event[0] != "admit"]


@pytest.fixture(autouse=True)
def isolation_holds():
    assert fake.isolation_report()["all_ok"], "the OS-input isolation is not in place; refusing to run"


@pytest.fixture
def world(monkeypatch):
    recorded = World([NP])
    monkeypatch.setattr(windows_backend, "_list_switchable_windows", recorded.enumerate)
    monkeypatch.setattr(windows_backend, "_foreground_window", lambda: recorded.foreground)
    monkeypatch.setattr(windows_backend, "_bring_to_foreground", recorded.bring)
    monkeypatch.setattr(windows_backend, "webbrowser", SimpleNamespace(open=lambda url: (recorded.events.append(("browser", url)), True)[1]))
    monkeypatch.setattr(WindowsComputerBackend, "_activate_url_handler_window", staticmethod(lambda url: recorded.events.append(("activate", url))))
    monkeypatch.setattr(WindowsComputerBackend, "_is_maximized", staticmethod(lambda hwnd: False))
    return recorded


def rig(world, *entries, ttl=60.0, enabled=True, limit=10**6):
    """(real backend, gate, grant or None, clock, switch state). Every call to the gate's `admit` is written into `world.events` first."""
    clock, state = Clock(), {"on": enabled, "limit": limit}
    gate = InputGate(lambda: state["on"], lambda: state["limit"], clock=clock)
    grant = gs.authorize(gate, *entries, ttl_seconds=ttl) if entries else None
    real_admit = gate.admit

    def logged(action, **kwargs):
        world.events.append(("admit", action))
        return real_admit(action, **kwargs)

    gate.admit = logged
    return WindowsComputerBackend(gate=gate, authority=grant), gate, grant, clock, state


def refused_untouched(world, gate, call, *, text=NOT_AUTHORIZED):
    with pytest.raises(ComputerControlError) as refused:
        call()
    assert str(refused.value) == text
    assert world.os_events == [] and len(gate._events) == 0


# ================================================ nothing is authorized by being registered, switched on, or asked by chat or a Job =============================


@pytest.mark.parametrize("action", ACTIONS)
def test_a_real_backend_with_no_grant_refuses_even_with_the_switch_on_and_a_huge_limit_and_touches_nothing(world, action):
    backend, gate, _, _, _ = rig(world)

    refused_untouched(world, gate, lambda: CALL[action](backend))

    assert world.events == [("admit", action)]  # the gate was asked, said no, and that is all that happened


@pytest.mark.parametrize("action", ACTIONS)
def test_refusal_happens_before_any_real_focus_or_browser_call_not_just_before_the_recorders(monkeypatch, action):
    """No recorders here: the REAL helpers stand behind tripwires, and the window list raises. A refusal that came one line too late would show as a tripwire hit."""
    monkeypatch.setattr(windows_backend, "_list_switchable_windows", lambda: (_ for _ in ()).throw(AssertionError("a refused call must not even list windows")))
    backend = WindowsComputerBackend(gate=InputGate(lambda: True, lambda: 10**6))
    hits = len(fake.TRIPWIRE_HITS)

    with pytest.raises(ComputerControlError, match="approved authorization"):
        CALL[action](backend)

    assert len(fake.TRIPWIRE_HITS) == hits


@pytest.fixture
def switched_on(monkeypatch):
    monkeypatch.setattr(settings, "computer_input_enabled", True)
    monkeypatch.setattr(settings, "computer_input_actions_per_minute", 10**6)
    gate = default_input_gate()
    monkeypatch.setattr(gate, "_events", type(gate._events)())
    monkeypatch.setattr(gate, "_grants", {})
    return gate


@pytest.mark.parametrize("action", ACTIONS)
def test_every_production_backend_refuses_both_actions_with_the_switch_on(world, switched_on, action):
    import app.main as live_app

    for name, backend in {"bare": WindowsComputerBackend(), "factory": create_default_computer_backend(), "app/main.py": live_app._computer_backend}.items():
        assert backend._authority is None, name
        with pytest.raises(ComputerControlError) as refused:
            CALL[action](backend)
        assert str(refused.value) == NOT_AUTHORIZED, name

    assert world.os_events == [] and len(switched_on._events) == 0


def test_the_tools_stay_registered_for_jobs_and_chat_but_registration_grants_nothing(world, switched_on, monkeypatch):
    import app.main as live_app

    # The Job `open_url` resolves the host through DNS before anything else; the live one would do a REAL lookup. Give the live Job tool a fixed resolver so this test never
    # touches the network (a failed lookup would be refused by the public-host guard BEFORE the gate, and the test would then say nothing about the gate).
    monkeypatch.setattr(live_app.core.tools.get("open_url"), "_resolver", lambda host: ["93.184.216.34"])
    assert JOB_COMPUTER_TOOL_NAMES == {"open_url", "scroll", "get_active_window", "switch_window"}
    assert live_app.core.tools.has("open_url") and live_app.core.tools.has("switch_window")  # the Job allowlist
    assert live_app.core.computer_tools.has("open_url") and live_app.core.computer_tools.has("switch_window")  # interactive chat
    jobs = ToolRegistry()
    register_job_computer_tools(jobs, live_app._computer_backend, resolver=lambda host: ["93.184.216.34"])

    for registry in (jobs, live_app.core.tools, live_app.core.computer_tools):
        opened = registry.execute("open_url", params={"url": "https://example.com/"})
        switched = registry.execute("switch_window", params={"title": TITLE, "process": PROCESS})
        for result in (opened, switched):
            assert result.status is ToolResultStatus.ERROR and NOT_AUTHORIZED in result.message
    assert world.os_events == []


class _NoAI(AIProvider):
    def generate_reply(self, message):
        raise AssertionError("no AI may be used for an explicit tool command")


class _NoToolAI(ToolCallProvider):
    def decide(self, message, candidate_tools):
        raise ToolCallError("an explicit command must never be handed to the AI")


def test_a_chained_chat_command_is_turned_away_whole_before_its_first_step(world, switched_on):
    backend = WindowsComputerBackend()
    jarvis = JarvisCore(
        provider=_NoAI(), memory=SQLiteMemoryStore(":memory:"), tools=ToolRegistry(),
        computer_tools=create_computer_tools_registry(backend=backend), tool_call_builder=ToolCallBuilder(_NoToolAI()),
    )

    reply = jarvis.handle_chat('use tool open_url https://example.com/, then use tool switch_window "Untitled - Notepad" notepad.exe')[0]

    assert "approved authorization" in reply and "nothing was executed" in reply.lower()
    assert world.os_events == [] and len(switched_on._events) == 0


@pytest.mark.parametrize("make", [
    pytest.param(lambda b: OpenUrlTool(b), id="open_url"),
    pytest.param(lambda b: JobOpenUrlTool(b, resolver=lambda host: ["93.184.216.34"]), id="job-open_url"),
    pytest.param(lambda b: SwitchWindowTool(b), id="switch_window"),
])
def test_each_tools_pre_check_turns_a_command_away_and_consumes_nothing(world, make):
    backend, gate, grant, _, state = rig(world)
    tool = make(backend)
    params = {"title": TITLE, "process": PROCESS} if tool.name == "switch_window" else {"url": "https://example.com/"}

    with pytest.raises(ToolArgumentError, match="approved authorization"):
        tool.validate_arguments(**params)
    state["on"] = False
    with pytest.raises(ToolArgumentError, match="switched off"):
        tool.validate_arguments(**params)

    assert world.os_events == [] and len(gate._events) == 0


@pytest.mark.parametrize("action", ACTIONS)
def test_a_pre_check_under_a_covering_grant_passes_and_spends_nothing(world, action):
    backend, gate, grant, _, _ = rig(world, ENTRY[action])
    tool = OpenUrlTool(backend) if action == "open_url" else SwitchWindowTool(backend)
    params = {"url": URL} if action == "open_url" else {"title": TITLE, "process": PROCESS}

    for _ in range(50):
        tool.validate_arguments(**params)

    assert gate.unspent(grant) == 1 and len(gate._events) == 0 and world.os_events == []


def test_no_tool_schema_carries_authority_and_the_switch_window_tool_requires_both_title_and_process():
    backend = WindowsComputerBackend(gate=InputGate(lambda: False, lambda: 0))
    registry = create_computer_tools_registry(backend=backend)
    for name in [tool.name for tool in registry.list_tools()]:
        parameters = {p.name for p in registry.get(name).parameters}
        assert not parameters & {"grant", "authority", "authorization", "token", "ticket", "credential", "key_material"}, name
    switch = registry.get("switch_window")
    assert {p.name: p.required for p in switch.parameters} == {"title": True, "process": True}


# ================================================ open_url: the grant binds the exact URL ========================================================================


def test_a_covering_grant_opens_exactly_that_url_after_admission_and_is_then_spent(world):
    backend, gate, grant, _, _ = rig(world, ENTRY["open_url"])

    backend.open_url(URL)

    assert world.events == [("admit", "open_url"), ("browser", URL), ("activate", URL)]  # the order: ask the gate, then the browser, then bring it forward
    assert gate.unspent(grant) == 0 and len(gate._events) == 1
    with pytest.raises(ComputerControlError):
        backend.open_url(URL)  # single use
    assert world.os_events == [("browser", URL), ("activate", URL)]


@pytest.mark.parametrize("other", [
    "https://example.com/a?x=1#other",  # a different fragment
    "https://example.com/a?x=2#top",  # a different query
    "https://example.com/a?x=1",  # no fragment
    "https://example.com/a/?x=1#top",  # a trailing slash
    "http://example.com/a?x=1#top",  # http, not https
    "https://EXAMPLE.com/a?x=1#top",  # host case
    "https://example.org/a?x=1#top",  # another host
    " https://example.com/a?x=1#top",  # a leading space: the browser would be handed a different string
], ids=lambda u: u[:40])
def test_a_grant_for_one_url_cannot_open_another_and_the_mismatch_ends_the_grant(world, other):
    backend, gate, grant, _, _ = rig(world, ENTRY["open_url"])

    with pytest.raises(ComputerControlError) as refused:
        backend.open_url(other)

    assert str(refused.value) == NOT_AUTHORIZED
    assert world.os_events == [] and gate.unspent(grant) == 0 and len(gate._events) == 0
    with pytest.raises(ComputerControlError):
        backend.open_url(URL)  # the grant is gone: even the right URL is now refused
    assert world.os_events == []


@pytest.mark.parametrize("bad", ["file:///C:/secrets.txt", "javascript:alert(1)", "https://", "", "   ", "ftp://example.com/"])
def test_an_invalid_url_is_refused_before_the_gate_and_costs_nothing(world, bad):
    backend, gate, grant, _, _ = rig(world, ENTRY["open_url"])

    with pytest.raises(ComputerControlError):
        backend.open_url(bad)

    assert world.events == [] and gate.unspent(grant) == 1 and len(gate._events) == 0  # not even the gate was asked


def test_the_job_variants_guards_still_run_first_and_a_granted_public_url_passes_them(world):
    backend, gate, grant, _, _ = rig(world, ("open_url", "https://example.com/"))
    tool = JobOpenUrlTool(backend, resolver=lambda host: ["93.184.216.34"])

    local = tool.execute(url="http://localhost:8000/")
    opened = tool.execute(url="https://example.com/")

    assert local.status is ToolResultStatus.ERROR and "open_url refused" in local.message
    assert ("admit", "open_url") in world.events and world.events.count(("admit", "open_url")) == 1  # only the public URL reached the gate; the local one never did
    assert opened.status is ToolResultStatus.SUCCESS and gate.unspent(grant) == 0


def test_a_failure_to_bring_the_browser_forward_after_admission_still_spends_the_entry(world, monkeypatch):
    backend, gate, grant, _, _ = rig(world, ENTRY["open_url"])
    monkeypatch.setattr(WindowsComputerBackend, "_activate_url_handler_window", staticmethod(lambda url: (_ for _ in ()).throw(ComputerControlError("not forward"))))

    with pytest.raises(ComputerControlError, match="not forward"):
        backend.open_url(URL)

    assert ("browser", URL) in world.events and gate.unspent(grant) == 0 and len(gate._events) == 1  # the browser was opened, and the entry stays spent


# ================================================ switch_window: the grant binds the exact title AND the exact process ===========================================


def test_a_covering_grant_switches_to_the_one_exact_match_after_admission_and_before_nothing_else(world):
    backend, gate, grant, _, _ = rig(world, ENTRY["switch_window"])

    info = backend.switch_window(TITLE, PROCESS)

    assert info == WindowInfo(title=TITLE, process_name="Notepad.exe", is_maximized=False)
    assert world.events == [("admit", "switch_window"), ("enumerate",), ("bring", 11)]  # ask the gate, ONE enumeration, then the focus call
    assert gate.unspent(grant) == 0 and len(gate._events) == 1


def test_an_exact_match_that_is_already_in_front_still_spends_the_entry_and_changes_no_focus(world):
    world.foreground = 11
    backend, gate, grant, _, _ = rig(world, ENTRY["switch_window"])

    backend.switch_window(TITLE, PROCESS)

    assert world.os_events == [("enumerate",)] and gate.unspent(grant) == 0


def test_the_process_name_is_case_folded_because_windows_file_names_are_but_nothing_else_is(world):
    backend, gate, grant, _, _ = rig(world, ENTRY["switch_window"])

    backend.switch_window(TITLE, "NOTEPAD.EXE")  # the same file name: the same authorization

    assert ("bring", 11) in world.events and gate.unspent(grant) == 0


@pytest.mark.parametrize("title, process", [
    ("untitled - notepad", PROCESS),  # the title is case-sensitive
    ("Untitled - Notepad - extra", PROCESS),  # a different title
    ("Notepad", PROCESS),  # a fragment is not the title
    (TITLE, "notepad++.exe"),  # a different process
    (TITLE, "code.exe"),
    ("Other", "code.exe"),
], ids=lambda value: str(value)[:30])
def test_a_grant_for_one_title_and_process_cannot_be_spent_on_another_and_the_mismatch_ends_the_grant(world, title, process):
    backend, gate, grant, _, _ = rig(world, ENTRY["switch_window"])

    with pytest.raises(ComputerControlError) as refused:
        backend.switch_window(title, process)

    assert str(refused.value) == NOT_AUTHORIZED
    assert world.os_events == [] and gate.unspent(grant) == 0 and len(gate._events) == 0  # no enumeration, no focus: refused at the gate
    with pytest.raises(ComputerControlError):
        backend.switch_window(TITLE, PROCESS)  # and the right pair is refused now
    assert world.os_events == []


@pytest.mark.parametrize("title, process", [
    (" Untitled - Notepad", PROCESS), ("Untitled - Notepad ", PROCESS), ("Untitled  - Notepad", PROCESS),  # padded or repeated whitespace: not the exact title
    ("", PROCESS), ("   ", PROCESS), ("x" * 101, PROCESS), ("Untitled\n- Notepad", PROCESS), (None, PROCESS), (5, PROCESS),
    (TITLE, ""), (TITLE, "notepad"), (TITLE, " notepad.exe"), (TITLE, "notepad.exe "), (TITLE, "..\\notepad.exe"), (TITLE, "c:\\x\\notepad.exe"), (TITLE, None), (TITLE, 5),
], ids=lambda value: repr(value)[:24])
def test_a_malformed_title_or_process_is_refused_before_the_gate_and_costs_nothing(world, title, process):
    backend, gate, grant, _, _ = rig(world, ENTRY["switch_window"])

    with pytest.raises(ComputerControlError):
        backend.switch_window(title, process)

    assert world.events == [] and gate.unspent(grant) == 1 and len(gate._events) == 0  # the gate was not even asked, so the entry is intact


FAILING_LOOKUPS = {
    "no window at all": [],
    "only another title": [(21, "Untitled - Notepad++", "notepad++.exe")],
    "only a case variant": [(22, "untitled - notepad", "Notepad.exe")],
    "only a longer title": [(23, "Untitled - Notepad - extra", "Notepad.exe")],
    "right title, wrong process": [(24, "Untitled - Notepad", "notepad++.exe")],
    "two identical windows": [NP, (12, "Untitled - Notepad", "notepad.exe")],
    "a near match first": [(25, "Untitled - Notepad (2)", "Notepad.exe"), (26, "Notepad", "Notepad.exe")],
}


@pytest.mark.parametrize("name", list(FAILING_LOOKUPS))
def test_zero_or_several_exact_matches_fail_without_any_focus_operation_and_the_entry_is_spent_before_the_enumeration(world, name):
    world.windows = FAILING_LOOKUPS[name]
    backend, gate, grant, _, _ = rig(world, ENTRY["switch_window"], ("scroll", 5))

    with pytest.raises(ComputerControlError) as failed:
        backend.switch_window(TITLE, PROCESS)

    assert world.events == [("admit", "switch_window"), ("enumerate",)]  # admitted FIRST, one enumeration, no focus call, no second look
    assert str(failed.value) != NOT_AUTHORIZED  # it was authorized: the lookup is what failed
    assert gate.unspent(grant) == 1 and len(gate._events) == 1  # the switch_window entry is spent, the slot too; only the scroll entry is left
    with pytest.raises(ComputerControlError):
        backend.switch_window(TITLE, PROCESS)  # the same entry cannot be tried again ...
    assert world.events.count(("enumerate",)) == 1 and ("bring", 11) not in world.events and not [e for e in world.events if e[0] == "bring"]  # ... no second look, no focus
    assert gate.unspent(grant) == 0  # (and asking for switch_window where the grant's next entry is scroll ended the grant)


def test_an_ambiguous_lookup_does_not_choose_the_topmost_window(world):
    world.windows = [NP, (12, "Untitled - Notepad", "notepad.exe")]
    backend, gate, grant, _, _ = rig(world, ENTRY["switch_window"])

    with pytest.raises(ComputerControlError, match="(?i)more than one"):
        backend.switch_window(TITLE, PROCESS)

    assert [e for e in world.events if e[0] == "bring"] == []


def test_a_window_that_the_deny_list_hides_is_never_a_match_even_with_an_exact_grant(world):
    world.windows = [(31, "Notepad - password.txt", "Notepad.exe")]
    backend, gate, grant, _, _ = rig(world, ("switch_window", "Notepad - password.txt", PROCESS))

    with pytest.raises(ComputerControlError):
        backend.switch_window("Notepad - password.txt", PROCESS)

    assert [e for e in world.events if e[0] == "bring"] == [] and gate.unspent(grant) == 0  # nothing focused, and the entry was spent


def test_windows_refusing_to_bring_the_window_forward_fails_closed_after_admission_and_spends_the_entry(world):
    world.refuse_foreground = True
    backend, gate, grant, _, _ = rig(world, ENTRY["switch_window"])

    with pytest.raises(ComputerControlError, match="did not let"):
        backend.switch_window(TITLE, PROCESS)

    assert world.events.count(("bring", 11)) == 1 and gate.unspent(grant) == 0 and len(gate._events) == 1
    with pytest.raises(ComputerControlError):
        backend.switch_window(TITLE, PROCESS)
    assert world.events.count(("bring", 11)) == 1  # not tried again


# ================================================ the grant's lifecycle applies to both actions ==================================================================


@pytest.mark.parametrize("action", ACTIONS)
def test_a_revoked_grant_is_refused(world, action):
    backend, gate, grant, _, _ = rig(world, ENTRY[action])
    gate.revoke(grant)

    refused_untouched(world, gate, lambda: CALL[action](backend))


@pytest.mark.parametrize("action", ACTIONS)
def test_an_expired_grant_is_refused_exactly_at_the_end_of_its_lifetime(world, action):
    backend, gate, grant, clock, _ = rig(world, ENTRY[action], ttl=10.0)
    clock.now += 9.999
    assert gate.unspent(grant) == 1
    clock.now += 0.001  # exactly the end

    refused_untouched(world, gate, lambda: CALL[action](backend))


@pytest.mark.parametrize("action", ACTIONS)
def test_a_switched_off_gate_refuses_and_ends_every_grant(world, action):
    backend, gate, grant, _, state = rig(world, ENTRY[action])
    other = gs.authorize(gate, "scroll")
    state["on"] = False

    refused_untouched(world, gate, lambda: CALL[action](backend), text=SWITCHED_OFF)

    state["on"] = True
    assert gate.unspent(grant) == 0 and gate.unspent(other) == 0  # nothing outlives a switch seen off


@pytest.mark.parametrize("action", ACTIONS)
def test_an_exhausted_grant_admits_nothing_more(world, action):
    backend, gate, grant, _, _ = rig(world, ENTRY[action])
    CALL[action](backend)
    seen = list(world.os_events)

    with pytest.raises(ComputerControlError):
        CALL[action](backend)

    assert world.os_events == seen and len(gate._events) == 1


@pytest.mark.parametrize("granted, used", [("open_url", "switch_window"), ("switch_window", "open_url")])
def test_a_grant_for_one_action_cannot_be_spent_on_the_other(world, granted, used):
    backend, gate, grant, _, _ = rig(world, ENTRY[granted])

    with pytest.raises(ComputerControlError):
        CALL[used](backend)

    assert world.os_events == [] and gate.unspent(grant) == 0


def test_the_entries_of_a_grant_are_used_in_their_order(world):
    backend, gate, grant, _, _ = rig(world, ENTRY["switch_window"], ENTRY["open_url"])

    with pytest.raises(ComputerControlError):
        backend.open_url(URL)  # the grant's next entry is the switch

    assert world.os_events == [] and gate.unspent(grant) == 0


def test_the_rate_limit_is_shared_and_refuses_both_actions_when_full(world):
    backend, gate, grant, _, _ = rig(world, ENTRY["open_url"], limit=0)

    refused_untouched(world, gate, lambda: backend.open_url(URL), text=_limited(0, 60))


# ================================================ the wrapper used for test doubles applies the same rules ========================================================


class Inner(ComputerBackend):
    def __init__(self):
        self.calls = []

    def get_active_window(self): raise AssertionError("not used")  # noqa: E704
    def get_screen_info(self): raise AssertionError("not used")  # noqa: E704
    def open_application(self, app): raise AssertionError("not used")  # noqa: E704
    def type_text(self, text): raise AssertionError("not used")  # noqa: E704
    def press_key(self, key): raise AssertionError("not used")  # noqa: E704
    def move_mouse(self, x, y): raise AssertionError("not used")  # noqa: E704
    def click_mouse(self, x, y, button): raise AssertionError("not used")  # noqa: E704
    def double_click(self, x, y, button): raise AssertionError("not used")  # noqa: E704
    def scroll(self, amount): raise AssertionError("not used")  # noqa: E704

    def open_url(self, url):
        self.calls.append(("open_url", url))

    def switch_window(self, title, process):
        self.calls.append(("switch_window", title, process))
        return WindowInfo(title, process, False)


@pytest.mark.parametrize("action", ACTIONS)
def test_the_wrapper_refuses_without_a_grant_and_reaches_the_double_only_with_the_exact_one(action):
    gate = InputGate(lambda: True, lambda: 10**6)
    inner = Inner()

    with pytest.raises(ComputerControlError, match="approved authorization"):
        CALL[action](GatedComputerBackend(inner, gate))
    assert inner.calls == []

    CALL[action](GatedComputerBackend(inner, gate, gs.authorize(gate, ENTRY[action])))
    assert inner.calls == [ENTRY[action]]


def test_the_wrapper_validates_first_like_the_real_backend_and_a_mismatch_never_reaches_the_double():
    gate = InputGate(lambda: True, lambda: 10**6)
    inner = Inner()
    grant = gs.authorize(gate, ENTRY["switch_window"])
    wrapper = GatedComputerBackend(inner, gate, grant)

    with pytest.raises(ComputerControlError):
        wrapper.switch_window(" Untitled - Notepad", PROCESS)  # malformed: refused before the gate, nothing spent
    assert gate.unspent(grant) == 1
    with pytest.raises(ComputerControlError):
        wrapper.switch_window(TITLE, "code.exe")  # a mismatch: refused and the grant ends
    assert inner.calls == [] and gate.unspent(grant) == 0


# ================================================ the pure helpers ===============================================================================================


@pytest.mark.parametrize("good", ["notepad.exe", "NOTEPAD.EXE", "notepad++.exe", "chrome.exe", "my app.exe", "a.b-c_d.exe"])
def test_a_process_name_is_a_plain_file_name_ending_in_exe_and_is_lower_cased(good):
    assert validate_process_name(good) == good.lower()


@pytest.mark.parametrize("bad", [None, 5, "", " ", "notepad", "notepad.exe ", " notepad.exe", ".exe", "a\\b.exe", "a/b.exe", "a:b.exe", "a" * 70 + ".exe", "no\ttab.exe", "unknown"])
def test_anything_else_is_not_a_process_name(bad):
    with pytest.raises(ComputerControlError):
        validate_process_name(bad)


def test_exact_window_matches_is_exact_on_the_title_and_the_lower_cased_process_and_skips_the_deny_list():
    windows = [
        (1, "Untitled - Notepad", "Notepad.exe"), (2, "untitled - notepad", "Notepad.exe"), (3, "Untitled - Notepad ", "Notepad.exe"), (4, "Untitled - Notepad", "notepad++.exe"),
        (5, "Untitled - Notepad", "unknown"), (6, "Vault - Untitled - Notepad", "Notepad.exe"), (7, "Untitled - Notepad", "Notepad.exe"),
    ]

    assert exact_window_matches(windows, "Untitled - Notepad", "notepad.exe") == [windows[0], windows[6]]
    assert exact_window_matches(windows, "Notepad", "notepad.exe") == []
    assert exact_window_matches([(8, "Sign in - Notepad", "Notepad.exe")], "Sign in - Notepad", "notepad.exe") == []  # a deny-listed title is never a match
    assert exact_window_matches([(9, "Anything", "KeePass.exe")], "Anything", "keepass.exe") == []  # nor a deny-listed process


# ================================================ the classification, the encodings and the order, by reading the code ============================================


def test_all_nine_acting_methods_are_gated_and_nothing_is_left_unprotected():
    assert GATED_METHODS == {"open_application", "type_text", "press_key", "move_mouse", "click_mouse", "double_click", "scroll", "open_url", "switch_window"}
    assert UNPROTECTED_ACTIONS == frozenset() and UNGATED_METHODS == {"get_active_window", "get_screen_info"}


def test_the_existing_encodings_are_byte_for_byte_what_they_were_before_the_two_actions_were_added():
    expected = {
        ("open_application", (AllowedApplication.NOTEPAD,)): "6a61727669732e636f6d70757465722e696e7075742e763100000000106f70656e5f6170706c69636174696f6e65000000076e6f7465706164",
        ("type_text", ("hello",)): "6a61727669732e636f6d70757465722e696e7075742e76310000000009747970655f74657874730000000568656c6c6f",
        ("press_key", (("ctrl", "n"),)): "6a61727669732e636f6d70757465722e696e7075742e7631000000000970726573735f6b6579740002000000046374726c000000016e",
        ("move_mouse", (10, 20)): "6a61727669732e636f6d70757465722e696e7075742e7631000000000a6d6f76655f6d6f7573656900000002313069000000023230",
        ("click_mouse", (10, 20, MouseButton.LEFT)): "6a61727669732e636f6d70757465722e696e7075742e7631000000000b636c69636b5f6d6f757365690000000231306900000002323065000000046c656674",
        ("click_mouse", (None, None, MouseButton.LEFT)): "6a61727669732e636f6d70757465722e696e7075742e7631000000000b636c69636b5f6d6f7573656e6e65000000046c656674",
        ("double_click", (10, 20, MouseButton.RIGHT)): "6a61727669732e636f6d70757465722e696e7075742e7631000000000c646f75626c655f636c69636b690000000231306900000002323065000000057269676874",
        ("scroll", (-5,)): "6a61727669732e636f6d70757465722e696e7075742e763100000000067363726f6c6c69000000022d35",
    }

    for (action, args), digest in expected.items():
        assert canonical_input(action, *args).hex() == digest, action


def test_the_two_new_actions_have_their_own_arity_and_typed_string_encoding():
    assert canonical_input("open_url", URL) != canonical_input("open_url", URL + "/")
    assert canonical_input("switch_window", TITLE, PROCESS) != canonical_input("switch_window", PROCESS, TITLE)  # order matters
    assert canonical_input("switch_window", "ab", "c") != canonical_input("switch_window", "a", "bc")  # length-prefixed: no field can run into the next
    for action, args in (("open_url", ()), ("open_url", (URL, URL)), ("switch_window", (TITLE,)), ("switch_window", (TITLE, PROCESS, PROCESS))):
        with pytest.raises(AuthorityError):
            canonical_input(action, *args)


def test_in_the_backend_each_action_asks_the_gate_before_anything_that_opens_or_focuses():
    tree = ast.parse((APP_DIR / "computer" / "windows_backend.py").read_text(encoding="utf-8"))
    cls = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "WindowsComputerBackend")
    methods = {m.name: m for m in cls.body if isinstance(m, ast.FunctionDef)}
    effects = ("webbrowser.open", "self._activate_url_handler_window", "_list_switchable_windows", "_bring_to_foreground", "win32gui.", "_foreground_window")

    for name in ACTIONS:
        calls = [n for n in ast.walk(methods[name]) if isinstance(n, ast.Call)]
        admissions = [n.lineno for n in calls if getattr(n.func, "id", None) == "admit_input"]
        touching = [n.lineno for n in calls if any(ast.unparse(n.func).startswith(effect) or ast.unparse(n.func) == effect for effect in effects)]
        assert len(admissions) == 1, name
        assert touching and all(admissions[0] < line for line in touching), f"{name}: something touches the desktop before the gate is asked"
    switch = [n for n in ast.walk(methods["switch_window"]) if isinstance(n, ast.Call) and "_list_switchable_windows" in ast.unparse(n.func)]
    assert len(switch) == 1  # ONE enumeration per call, so no second look


def test_the_documentation_states_the_exact_binding_and_the_window_instance_limitation():
    readme = (APP_DIR.parent / "README.md").read_text(encoding="utf-8")
    section = readme[readme.index("### Computer Input Gate"):].lower()
    backend = (APP_DIR / "computer" / "windows_backend.py").read_text(encoding="utf-8").lower()
    tools = (APP_DIR / "computer" / "tools.py").read_text(encoding="utf-8").lower()

    assert "not a window instance" in section and "more than one" in section and "no second look" in section  # the README
    assert "not a window instance" in backend and "no fallback" in backend  # the backend's own docstring
    assert "does not identify one window" in tools and "exact process name" in tools  # the tools' module docstring


# ================================================ unencodable text (a lone surrogate) in a URL or a title ===================================================


def test_a_url_with_unencodable_text_is_refused_before_the_gate_decides_and_before_the_browser_is_touched(world):
    backend, gate, grant, _, _ = rig(world, ("open_url", "https://example.com/ok"))

    with pytest.raises(ComputerControlError) as refused:
        backend.open_url("https://example.com/a\ud800")

    assert str(refused.value) == NOT_AUTHORIZED
    assert world.events == []  # not even an admission was attempted: nothing was opened, listed or brought forward
    assert gate.unspent(grant) == 1 and len(gate._events) == 0 and gate._grants[grant.id].closed is None  # the entry is intact; this was not a mismatch
    backend.open_url("https://example.com/ok")  # the URL the grant names still works, once
    assert world.events == [("admit", "open_url"), ("browser", "https://example.com/ok"), ("activate", "https://example.com/ok")] and gate.unspent(grant) == 0


@pytest.mark.parametrize("title, process", [("a\ud800", "notepad.exe"), (TITLE, "\udc00.exe")])
def test_a_window_title_or_process_with_unencodable_text_is_refused_by_validation_with_nothing_spent(world, title, process):
    backend, gate, grant, _, _ = rig(world, ENTRY["switch_window"])

    with pytest.raises(ComputerControlError):
        backend.switch_window(title, process)

    assert world.events == [] and gate.unspent(grant) == 1 and len(gate._events) == 0  # never reached the gate, never listed a window


def test_the_job_variant_with_unencodable_text_in_a_url_reports_an_error_and_opens_nothing(world):
    backend, gate, grant, _, _ = rig(world, ("open_url", "https://example.com/"))
    tool = JobOpenUrlTool(backend, resolver=lambda host: ["93.184.216.34"])

    result = tool.execute(url="https://example.com/a\ud800")

    assert result.status is ToolResultStatus.ERROR and world.events == [] and gate.unspent(grant) == 1
