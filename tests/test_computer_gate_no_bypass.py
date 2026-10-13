"""
There is NO budget-only way through the Computer Input Gate.

The gate used to have a constructor flag (`require_authority=False`) that dropped the grant requirement and left only the switch and the rate limit; it existed for unit tests.
It is gone, and so is every other way to ask the gate to admit an action without a grant that covers that exact action:

  * `InputGate.__init__` has no authorization option, and no gate has an authorization-mode attribute;
  * `admit` / `check` / `remaining` refuse without a real, registered grant - with the switch on and a huge limit;
  * the production backends (a bare `WindowsComputerBackend()`, the factory's, the one in app/main.py) carry no grant, so with the switch ON they still admit nothing;
  * the gate is a final class, and a backend refuses anything that is not an `InputGate`, so a stand-in whose `admit` says yes cannot be slipped in;
  * only `create_default_input_gate()` builds a gate in production code, and nothing in `app/` detects a test run or imports test code.

None of this depends on whether pytest is running. The tests that exercise the switch and the rate limit go through the same real path with real grants (tests/grant_support.py).
Everything here runs against the fake input library (tests/conftest.py): a mistake would show up as a recorded call, never as real input.
"""

import ast
import inspect
import pathlib
from collections import deque
from types import SimpleNamespace

import pytest

import fake_input_library as fake
import grant_support as gs
from app.computer.authority import InputGrant, InputStep, canonical_input
from app.computer.base import AllowedApplication, ComputerBackend, ComputerControlError, MouseButton
from app.computer.factory import create_default_computer_backend
from app.computer.gate import GATED_METHODS, NOT_AUTHORIZED, GatedComputerBackend, InputGate, admit_input, create_default_input_gate, default_input_gate
from app.computer.issuer import GrantIssuer
from app.computer.windows_backend import WindowsComputerBackend
from app.config import settings

APP_DIR = pathlib.Path(__file__).resolve().parent.parent / "app"

CALLS = {
    "type_text": lambda b: b.type_text("hello"),
    "press_key": lambda b: b.press_key("enter"),
    "move_mouse": lambda b: b.move_mouse(10, 20),
    "click_mouse": lambda b: b.click_mouse(10, 20, MouseButton.LEFT),
    "double_click": lambda b: b.double_click(10, 20, MouseButton.LEFT),
    "open_application": lambda b: b.open_application(AllowedApplication.NOTEPAD),
    "scroll": lambda b: b.scroll(5),
    "open_url": lambda b: b.open_url("https://example.com/"),
    "switch_window": lambda b: b.switch_window("Untitled - Notepad", "notepad.exe"),
}
ACTIONS = sorted(GATED_METHODS)


@pytest.fixture(autouse=True)
def isolation_holds():
    assert fake.isolation_report()["all_ok"], "the OS-input isolation is not in place; refusing to run"


def app_sources():
    for path in sorted(APP_DIR.rglob("*.py")):
        yield path.relative_to(APP_DIR).as_posix(), path.read_text(encoding="utf-8")


# ================================================ the constructor and the object have no authorization mode ===================================================


def test_the_gate_constructor_offers_no_way_to_turn_authorization_off():
    names = [name for name in inspect.signature(InputGate.__init__).parameters if name != "self"]

    assert names == ["enabled", "limit", "window_seconds", "clock", "max_grants"]


@pytest.mark.parametrize("option", ["require_authority", "budget_only", "authority_required", "bypass", "authorize", "strict"])
def test_no_spelling_of_a_budget_only_option_is_accepted(option):
    with pytest.raises(TypeError):
        InputGate(lambda: True, lambda: 10, **{option: False})


def test_a_gate_carries_no_authorization_mode_to_flip_afterwards():
    gate = InputGate(lambda: True, lambda: 10)

    assert not [name for name in vars(gate) if any(word in name for word in ("authority", "budget", "bypass", "strict", "mode"))]
    assert not hasattr(gate, "_require_authority")
    for factory in (create_default_input_gate, default_input_gate):
        assert not hasattr(factory(), "_require_authority")


def test_admit_makes_the_grant_and_the_canonical_form_required_keyword_arguments():
    parameters = inspect.signature(InputGate.admit).parameters

    for name in ("authority", "canonical"):
        assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY and parameters[name].default is inspect.Parameter.empty, name
    gate = InputGate(lambda: True, lambda: 10**9)
    with pytest.raises(TypeError):
        gate.admit("scroll")  # no way to even ask without saying which grant and which exact action


# ================================================ the gate itself: no grant, no admission, whatever the switch and the limit say ==============================


def a_lookalike_of_a_registered_grant(gate):
    real = gs.authorize(gate, "scroll")
    twin = InputGrant()
    twin._id = real.id  # the same id, a different object (and a different key)
    return twin


@pytest.mark.parametrize("action", ACTIONS)
@pytest.mark.parametrize("who", ["nothing", "a string", "a dict", "a number", "a bare object", "a lookalike", "a stand-in object"])
def test_admit_refuses_every_action_without_a_real_registered_grant_with_the_switch_on_and_a_huge_limit(action, who):
    gate = InputGate(lambda: True, lambda: 10**9)
    authority = {
        "nothing": None, "a string": "grant", "a dict": {"id": "x"}, "a number": 5, "a bare object": object(),
        "a lookalike": a_lookalike_of_a_registered_grant(gate), "a stand-in object": SimpleNamespace(id="x", _mac=lambda c: b"x" * 32),
    }[who]

    with pytest.raises(ComputerControlError) as refused:
        gate.admit(action, authority=authority, canonical=canonical_input(action, *gs.PROBE_ARGS[action]))

    assert str(refused.value) == NOT_AUTHORIZED
    assert len(gate._events) == 0  # and it took no rate-limit slot


@pytest.mark.parametrize("action", ACTIONS)
def test_check_and_remaining_say_no_without_a_grant_even_with_the_switch_on_and_a_huge_limit(action):
    gate = InputGate(lambda: True, lambda: 10**9)

    with pytest.raises(ComputerControlError) as refused:
        gate.check(action)
    with pytest.raises(ComputerControlError):
        gate.check(action, authority=None)

    assert str(refused.value) == NOT_AUTHORIZED
    assert gate.remaining() == 0 and gate.remaining(authority=None) == 0
    assert gate.budget_remaining() == 10**9  # the READ-ONLY budget question is not an admission and still answers


@pytest.mark.parametrize("action", ACTIONS)
def test_a_grant_for_another_action_is_not_a_way_in_and_ends_that_grant(action):
    gate = InputGate(lambda: True, lambda: 10**9)
    other = next(name for name in ACTIONS if name != action)
    grant = gs.authorize(gate, other)

    with pytest.raises(ComputerControlError):
        gs.admit_with(gate, grant, action)

    assert gate.unspent(grant) == 0 and len(gate._events) == 0


def test_the_switch_is_still_necessary_with_a_perfectly_good_grant():
    state = {"on": True}
    gate = InputGate(lambda: state["on"], lambda: 10)
    grant = gs.authorize(gate, "scroll")
    state["on"] = False

    with pytest.raises(ComputerControlError, match="switched off"):
        gs.admit_with(gate, grant, "scroll")


# ================================================ the production backends carry no grant, so the switch alone opens nothing ===================================


@pytest.fixture
def switch_on_with_a_huge_limit(monkeypatch):
    monkeypatch.setattr(settings, "computer_input_enabled", True)
    monkeypatch.setattr(settings, "computer_input_actions_per_minute", 10**6)
    gate = default_input_gate()
    monkeypatch.setattr(gate, "_events", deque())
    return gate


@pytest.mark.parametrize("action", ACTIONS)
def test_with_the_switch_on_every_production_backend_still_admits_nothing(action, switch_on_with_a_huge_limit):
    import app.main as live_app

    gate = switch_on_with_a_huge_limit
    backends = {"WindowsComputerBackend()": WindowsComputerBackend(), "the factory's": create_default_computer_backend(), "app/main.py's": live_app._computer_backend}

    for name, backend in backends.items():
        assert backend.gate is gate, name
        with pytest.raises(ComputerControlError) as refused:
            CALLS[action](backend)
        assert str(refused.value) == NOT_AUTHORIZED, name

    assert fake.FAKE.calls == [] and len(gate._events) == 0


def test_the_production_backend_pre_check_refuses_too_so_a_command_is_turned_away_whole(switch_on_with_a_huge_limit):
    backend = WindowsComputerBackend()

    for action in ACTIONS:
        with pytest.raises(ComputerControlError, match="needs an approved authorization"):
            backend.check_input(action)


# ================================================ a stand-in gate cannot be slipped in (3b) ===================================================================


class _Inner(ComputerBackend):
    def get_active_window(self): raise AssertionError("not called")  # noqa: E704
    def get_screen_info(self): raise AssertionError("not called")  # noqa: E704
    def open_application(self, app): raise AssertionError("not called")  # noqa: E704
    def open_url(self, url): raise AssertionError("not called")  # noqa: E704
    def type_text(self, text): raise AssertionError("not called")  # noqa: E704
    def press_key(self, key): raise AssertionError("not called")  # noqa: E704
    def move_mouse(self, x, y): raise AssertionError("not called")  # noqa: E704
    def click_mouse(self, x, y, button): raise AssertionError("not called")  # noqa: E704
    def double_click(self, x, y, button): raise AssertionError("not called")  # noqa: E704
    def scroll(self, amount): raise AssertionError("not called")  # noqa: E704


class _AlwaysYes:
    """What a bypass would look like: it has the gate's methods and says yes to everything."""

    gate = None

    def admit(self, action, **kwargs):
        return None

    def check(self, action, **kwargs):
        return None


STAND_INS = {"an always-yes object": _AlwaysYes(), "a namespace": SimpleNamespace(admit=lambda *a, **k: None, check=lambda *a, **k: None), "a bare object": object(), "a string": "gate"}


def test_the_gate_class_is_final():
    with pytest.raises(TypeError):

        class Permissive(InputGate):
            def admit(self, action, *, authority, canonical):
                return None


@pytest.mark.parametrize("name", list(STAND_INS))
def test_the_real_backend_refuses_anything_that_is_not_an_input_gate(name):
    with pytest.raises(TypeError):
        WindowsComputerBackend(gate=STAND_INS[name])


@pytest.mark.parametrize("name", list(STAND_INS))
def test_the_wrapper_refuses_anything_that_is_not_an_input_gate(name):
    with pytest.raises(TypeError):
        GatedComputerBackend(_Inner(), STAND_INS[name])


@pytest.mark.parametrize("name", list(STAND_INS))
def test_admission_refuses_anything_that_is_not_an_input_gate(name):
    with pytest.raises(TypeError):
        admit_input(STAND_INS[name], None, "scroll", 5)
    assert fake.FAKE.calls == []


def test_a_real_gate_is_still_accepted_everywhere():
    gate = InputGate(lambda: True, lambda: 10)

    assert WindowsComputerBackend(gate=gate).gate is gate
    assert GatedComputerBackend(_Inner(), gate).gate is gate


# ================================================ production code: one place builds a gate, and nothing detects or imports tests ==============================


def test_only_the_default_gate_factory_builds_an_input_gate_in_production_code_and_it_passes_nothing_but_the_two_settings():
    class Finder(ast.NodeVisitor):
        def __init__(self, rel):
            self.rel, self.inside, self.sites = rel, [], []

        def visit_FunctionDef(self, node):
            self.inside.append(node.name)
            self.generic_visit(node)
            self.inside.pop()

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_Call(self, node):
            if getattr(node.func, "id", getattr(node.func, "attr", None)) == "InputGate":
                self.sites.append((self.rel, self.inside[-1] if self.inside else "<module>", tuple(sorted(k.arg for k in node.keywords)), len(node.args)))
            self.generic_visit(node)

    sites = []
    for rel, source in app_sources():
        finder = Finder(rel)
        finder.visit(ast.parse(source))
        sites += finder.sites

    assert sites == [("computer/gate.py", "create_default_input_gate", ("enabled", "limit"), 0)]


def test_the_application_has_no_authorization_off_switch_in_its_source():
    offenders = []
    for rel, source in app_sources():
        lowered = source.lower()
        for phrase in ("require_authority", "budget_only"):  # identifiers (the prose may say there is NO budget-only mode)
            if phrase in lowered:
                offenders.append((rel, phrase))

    assert offenders == []


def test_nothing_in_the_application_detects_a_test_run_or_imports_test_code():
    offenders = []
    for rel, source in app_sources():
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                modules = [alias.name for alias in node.names] if isinstance(node, ast.Import) else [node.module or ""]
                for module in modules:
                    if module.split(".")[0] in {"pytest", "_pytest", "tests", "conftest", "grant_support", "fake_input_library", "unittest"}:
                        offenders.append((rel, "imports " + module))
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and "PYTEST_CURRENT_TEST" in node.value:
                offenders.append((rel, "mentions " + node.value))
            if isinstance(node, (ast.Compare, ast.Call, ast.Subscript)) and "sys.modules" in ast.unparse(node) and "pytest" in ast.unparse(node).lower():
                offenders.append((rel, "asks whether pytest is loaded"))  # (the data in self_development/safety.py that merely NAMES pytest as an allowed program is not a detection)

    assert offenders == []


# ================================================ the test helpers are not a way around the gate =============================================================


class Recording(_Inner):
    def __init__(self):
        self.effects = []

    def type_text(self, text):
        self.effects.append(("type_text", text))

    def scroll(self, amount):
        self.effects.append(("scroll", amount))


def test_the_helper_with_the_switch_off_reaches_the_gate_with_no_grant_and_the_gate_refuses():
    gate = InputGate(lambda: False, lambda: 10)
    inner = Recording()
    backend = gs.PerCallGrants(lambda grant: GatedComputerBackend(inner, gate, grant), gate)

    with pytest.raises(ComputerControlError, match="switched off"):
        backend.type_text("x")
    with pytest.raises(ComputerControlError, match="switched off"):
        gs.admit_now(gate, "scroll")

    assert inner.effects == [] and len(gate._events) == 0


def test_the_helper_mints_exactly_one_matching_entry_per_call_and_leaves_no_live_grant():
    gate = InputGate(lambda: True, lambda: 2)
    inner = Recording()
    backend = gs.PerCallGrants(lambda grant: GatedComputerBackend(inner, gate, grant), gate)

    backend.type_text("a")
    backend.scroll(3)
    with pytest.raises(ComputerControlError, match="limited to 2"):
        backend.type_text("b")  # the LIMIT is the production gate's: the helper adds no capacity

    assert inner.effects == [("type_text", "a"), ("scroll", 3)]
    assert not [state for state in gate._grants.values() if state.usable]


def test_a_grant_the_helper_mints_has_the_same_exact_scope_as_any_other():
    gate = InputGate(lambda: True, lambda: 10)
    inner = Recording()
    grant = gs.authorize(gate, ("type_text", "approved"))

    with pytest.raises(ComputerControlError):
        GatedComputerBackend(inner, gate, grant).type_text("something else")

    assert inner.effects == [] and gate.unspent(grant) == 0  # refused, and the whole grant ended
    assert isinstance(GrantIssuer(gate).issue([InputStep("scroll", (1,))], ttl_seconds=5, purpose="x", actor="y"), InputGrant)


def test_the_gate_level_helpers_refuse_without_a_grant_just_like_the_gate():
    gate = InputGate(lambda: True, lambda: 10**9)

    with pytest.raises(ComputerControlError):
        gs.admit_with(gate, None, "type_text")
    with pytest.raises(ComputerControlError):
        gate.check("type_text", authority=None)


def test_the_helper_module_is_test_only_and_no_conftest_hook_installs_a_grant_or_patches_the_gate():
    conftest = (pathlib.Path(__file__).resolve().parent / "conftest.py").read_text(encoding="utf-8")

    assert "grant_support" not in conftest and "InputGate" not in conftest and "admit" not in conftest
