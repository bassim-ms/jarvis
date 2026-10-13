"""
Test-only helpers for the tests that are about the Computer Input Gate's SWITCH and RATE LIMIT (not about what a grant covers).

Those tests used to run on a "budget-only" gate that needed no authorization. That mode no longer exists: `InputGate.admit` ALWAYS requires a grant that covers the exact
action. These helpers let a budget test keep its assertions while going through the REAL path: a real grant, minted by the real `GrantIssuer`, checked by the real
`InputGate.admit` (HMAC match, single use, expiry, switch, shared rate limit, one lock).

  * Nothing here is imported by `app/` (a test in tests/test_computer_gate_no_bypass.py pins that), and nothing here patches or replaces the gate.
  * Nothing is ambient: a test builds the helper explicitly and hands it the gate. There is no conftest hook, no environment variable and no pytest detection.
  * A helper that cannot mint a grant because the SWITCH IS OFF (the issuer refuses then) does NOT skip the gate: it makes the call with no grant at all, so the gate's own
    refusal (the "switched off" message, nothing spent) is what the test observes. Any other minting problem is raised, never hidden.
  * What the helpers cannot do is hide a missing grant: `PerCallGrants(...)` mints exactly what the call needs, so the question these tests ask - "does the switch / the
    limit / the lock behave" - is answered by the production code, while "does a grant protect" is answered in tests/test_computer_authorization.py and
    tests/test_computer_gate_no_bypass.py.
"""

from __future__ import annotations

import logging
import re
from contextlib import contextmanager

from app.computer.authority import MAX_GRANT_SECONDS, AuthorityError, InputGrant, InputStep, canonical_input
from app.computer.base import (
    AllowedApplication,
    ComputerBackend,
    ComputerControlError,
    MouseButton,
    normalize_key_input,
    validate_exact_window_title,
    validate_process_name,
    validate_url,
)
from app.computer.gate import InputGate, admit_input
from app.computer.issuer import GrantIssuer

PURPOSE = "budget-test"
ACTOR = "test-helper"

#: Valid arguments for each grantable action, for the callers that only care about the action NAME (a pre-check, or a gate-level test).
PROBE_ARGS = {
    "open_application": (AllowedApplication.NOTEPAD,),
    "type_text": ("x",),
    "press_key": (("enter",),),
    "move_mouse": (10, 20),
    "click_mouse": (10, 20, MouseButton.LEFT),
    "double_click": (10, 20, MouseButton.LEFT),
    "scroll": (5,),
    "open_url": ("https://example.com/",),
    "switch_window": ("Example Window", "example.exe"),
}

_LIFECYCLE = "computer_input grant "  # issued | refused | revoked | spent: the lifecycle lines, as opposed to the allowed / refused lines of an ACTION
_GRANT_FIELDS = re.compile(r" grant=[0-9a-f]{8} entry=\d+/\d+")
_SHORT_ID = re.compile(r"\b[0-9a-f]{8}\b")


class _DropLifecycle(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return not record.getMessage().startswith(_LIFECYCLE)


@contextmanager
def lifecycle_quiet():
    """While active, the audit logger drops the grant LIFECYCLE lines (issued / refused / revoked / spent) that these helpers cause, so a test that counts the audit lines
    of ACTIONS sees only those. The allowed / refused lines of actions are untouched."""
    audit = logging.getLogger("jarvis.computer.input")
    dropper = _DropLifecycle()
    audit.addFilter(dropper)
    try:
        yield
    finally:
        audit.removeFilter(dropper)


def budget_trail(lines):
    """Audit lines with the grant fields of an 'allowed' line removed, so `computer_input allowed action=type_text used=1/3` compares as it always did."""
    return [_GRANT_FIELDS.sub("", line) for line in lines]


def without_grant_ids(text: str) -> str:
    """A log text with every 8-hex-digit grant id prefix replaced, so a 'no value appears in the log' check cannot trip over a random id that happens to contain '777'."""
    return _SHORT_ID.sub("<id>", text)


def _steps(calls) -> list[InputStep]:
    steps = []
    for call in calls:
        action, args = (call, PROBE_ARGS[call]) if isinstance(call, str) else (call[0], tuple(call[1:]))
        steps.append(InputStep(action, args))
    return steps


def authorize(gate: InputGate, *calls, ttl_seconds: float = MAX_GRANT_SECONDS) -> InputGrant:
    """A real grant, from the real issuer, covering `calls` in order. A call is an action name (valid probe arguments) or `(action, *args)`. Needs the switch ON, as always."""
    return GrantIssuer(gate).issue(_steps(calls), ttl_seconds=ttl_seconds, purpose=PURPOSE, actor=ACTOR)


def try_authorize(gate: InputGate, *calls, ttl_seconds: float = MAX_GRANT_SECONDS) -> InputGrant | None:
    """`authorize`, or None while the switch is off (then there can be no grant, and the gate is asked with none). Other problems are raised."""
    try:
        return authorize(gate, *calls, ttl_seconds=ttl_seconds)
    except ComputerControlError:
        return None


def admit_with(gate: InputGate, grant: InputGrant | None, action: str, *args) -> None:
    """The gate-level admission exactly as a backend does it (`admit_input`): the arguments are put in canonical form and checked against `grant`."""
    admit_input(gate, grant, action, *(args or PROBE_ARGS[action]))


def admit_now(gate: InputGate, action: str, *args) -> None:
    """Admit ONE action through the real `admit`, with a one-entry grant minted for exactly it (none if the switch is off, so the gate refuses it itself)."""
    with lifecycle_quiet():
        grant = try_authorize(gate, (action, *(args or PROBE_ARGS[action])))
        try:
            admit_with(gate, grant, action, *args)
        finally:
            if grant is not None:
                gate.revoke(grant)


def check_now(gate: InputGate, action: str) -> None:
    """The gate's NON-CONSUMING pre-check for `action`, with a grant minted for it and revoked straight after (so a pre-check leaves no live grant behind)."""
    with lifecycle_quiet():
        grant = try_authorize(gate, action)
        try:
            gate.check(action, authority=grant)
        finally:
            if grant is not None:
                gate.revoke(grant)


class PerCallGrants(ComputerBackend):
    """A backend that, before each of the nine acting methods, mints a ONE-ENTRY grant for exactly that call, asks `build(grant)` for a backend that carries it and makes the
    call there. `build` is `lambda grant: GatedComputerBackend(inner, gate, grant)` or `lambda grant: WindowsComputerBackend(gate=gate, authority=grant)`: the production
    enforcement, unchanged. The grant is revoked afterwards, so a refused call leaves no live grant behind (the issuer allows only a few live grants at once).

    It carries `gate`, like a backend that enforces one, so the production wrapper refuses to wrap it a second time."""

    def __init__(self, build, gate: InputGate):
        self.gate = gate
        self._build = build
        self._issuer = GrantIssuer(gate)

    @property
    def inner(self) -> ComputerBackend:
        return self._build(None).inner

    def _mint(self, action: str, args: tuple) -> InputGrant | None:
        try:
            canonical_input(action, *args)
        except AuthorityError:  # arguments that cannot be authorized at all: the backend's own validation answers them
            return None
        try:
            return self._issuer.issue([InputStep(action, args)], ttl_seconds=MAX_GRANT_SECONDS, purpose=PURPOSE, actor=ACTOR)
        except ComputerControlError:  # the switch is off: no grant can exist; the call goes to the gate with none and the GATE refuses it
            return None

    def _call(self, action: str, args: tuple, call):
        with lifecycle_quiet():
            grant = self._mint(action, args)
            try:
                return call(self._build(grant))
            finally:
                if grant is not None:
                    self.gate.revoke(grant)

    def check_input(self, action: str) -> None:
        with lifecycle_quiet():
            grant = self._mint(action, PROBE_ARGS[action])
            try:
                self._build(grant).check_input(action)
            finally:
                if grant is not None:
                    self.gate.revoke(grant)

    # -- the reads: passed straight through ------------------------------------------------------------------

    def get_active_window(self):
        return self._build(None).get_active_window()

    def get_screen_info(self):
        return self._build(None).get_screen_info()


    # -- the seven input methods -------------------------------------------------------------------------------------------------

    def open_application(self, app):
        return self._call("open_application", (app,), lambda backend: backend.open_application(app))

    def type_text(self, text):
        return self._call("type_text", (text,), lambda backend: backend.type_text(text))

    def press_key(self, key):
        try:
            tokens = normalize_key_input(key)
        except ComputerControlError:  # an invalid key is turned away by the backend before the gate, with no grant involved
            return self._build(None).press_key(key)
        return self._call("press_key", (tokens,), lambda backend: backend.press_key(key))

    def move_mouse(self, x, y):
        return self._call("move_mouse", (x, y), lambda backend: backend.move_mouse(x, y))

    def click_mouse(self, x, y, button):
        return self._call("click_mouse", (x, y, button), lambda backend: backend.click_mouse(x, y, button))

    def double_click(self, x, y, button):
        return self._call("double_click", (x, y, button), lambda backend: backend.double_click(x, y, button))

    def scroll(self, amount):
        return self._call("scroll", (amount,), lambda backend: backend.scroll(amount))

    def open_url(self, url):
        try:
            validate_url(url)
        except ComputerControlError:  # an invalid URL is turned away by the backend before the gate, with no grant involved
            return self._build(None).open_url(url)
        return self._call("open_url", (url,), lambda backend: backend.open_url(url))

    def switch_window(self, title, process):
        try:
            exact = (validate_exact_window_title(title), validate_process_name(process))  # the values the backend will present to the gate
        except ComputerControlError:  # malformed: turned away before the gate, with no grant involved
            return self._build(None).switch_window(title, process)
        return self._call("switch_window", exact, lambda backend: backend.switch_window(title, process))
