"""
The Computer Input Gate: the ONE place that decides whether Jarvis may send input to the desktop.

    a chat command / a registry call / a chained command  ->  an input TOOL  ->  a backend that enforces the gate  ->  InputGate.admit()  ->  the desktop

The gate is enforced by the backend ITSELF, at its lowest layer: `WindowsComputerBackend` calls `InputGate.admit()` immediately before each OS call that sends
input, after the arguments were validated, using ONE process-wide gate unless it is handed another. `GatedComputerBackend` (below) does the same job for any OTHER
backend that does not enforce the gate itself (a test double, a future backend); it refuses to wrap one that does, so an action can never be counted twice.

The nine ACTING methods - `type_text`, `press_key`, `move_mouse`, `click_mouse`, `double_click`, `scroll`, `open_application`, `open_url` and `switch_window` - can drive the
keyboard, the mouse wheel and the mouse, launch programs, open the default browser and change which window is in front. They are SAFE_ACTION (no confirmation), so one anonymous
chat message could press Win+R, type a command and press Enter, or move the focus somewhere a later keystroke will land.
Two locks close that path without adding any capability, and BOTH must open:

  * SWITCH     `computer_input_enabled` (default OFF) is read on EVERY call. Off, no input method reaches the desktop, and any grants are revoked the moment an
               admission sees the switch off.
  * GRANT      An authorization grant (app/computer/authority.py) that covers THIS exact action with THESE exact arguments, next in its approved order. The backend
               instance carries the grant it was handed (`WindowsComputerBackend(authority=grant)`); the shared production backend carries none, so it admits nothing.
               Grants are single-use per entry, expire on a monotonic clock (300 s at most), and are revoked by `revoke()` / `revoke_all()`. A wrong action or wrong
               arguments is refused AND revokes the whole grant. Nothing is inherited: authority is a reference somebody passed, not ambient context.
  * LIMIT      ONE limit shared by all nine: at most `computer_input_actions_per_minute` (default 10) admitted actions in any rolling 60 seconds. An action admitted at
               time t stops counting at exactly t + 60 s. A refused attempt - for ANY reason - uses up NO slot and NO grant entry.
  * HERE       Enforcement is at EXECUTION time, in the backend every tool ends in, so it cannot be skipped by calling a tool's `execute` directly, by
               `ToolRegistry.execute`, by a chained "use tool ..., then ..." command, by a caller that never ran `validate_arguments`, or by building a backend somewhere
               the factory never saw. The tools ALSO pre-check (`check_input`, which consumes nothing and is order-agnostic) from `validate_arguments`, so a whole command
               is turned away before its first step runs; that pre-check is a convenience, never the enforcement.
  * ATOMIC     Validating the grant, spending its entry, and recording the rate-limit slot are ONE step under one lock, so any number of concurrent calls admit at most
               what the grant and the limit allow, and a refusal changes neither.
  * IN FLIGHT  `type_text` takes time, so the real backend types it in bounded chunks (windows_backend.TYPING_CHUNK_CHARACTERS) and, before EVERY chunk, asks the admission's
               `AdmissionTicket` whether it may continue. That recheck is non-consuming (the whole text is still ONE grant entry and ONE rate-limit slot) and refuses once the
               grant was revoked or ended by a mismatch, its lifetime has passed, or the switch is off - and then no further chunk is sent. HONEST LIMITS: characters already
               sent to the operating system cannot be recalled; revocation, expiry and switch-off are observed at the NEXT recheck, so at most one chunk (and the OS call that is
               already running) can follow the moment they happen; the other eight acting methods are a single OS call (or one lookup) with nothing in between to interrupt. `GatedComputerBackend`
               wraps a double that does its own typing, so it cannot interrupt that double.
  * AUDIT      Every admitted and every refused attempt is logged (logger `jarvis.computer.input`): the action's NAME, the outcome, a reason code and counts, with a
               short grant-id prefix. NEVER typed text, key names, coordinates, application names, argument digests or any metadata derived from an argument (no lengths).

What it does NOT do: it grants nothing by itself (grants are minted only by app/computer/issuer.py), adds no tool or permission and changes no permission level, keeps no
state beyond bounded in-memory lists (a restart forgets every grant, which is the safe direction), and covers every acting method (UNPROTECTED_ACTIONS is empty).

A grant is never optional: the gate has no budget-only mode, no constructor option or argument default that skips it, and nothing that asks whether a test is running; the
class is final and the backends accept nothing but an `InputGate`. Tests of the switch and the rate limit go through the same path with real grants (tests/grant_support.py).

NOT A SECURITY BOUNDARY AGAINST CODE IN THIS PROCESS: anything running here can set the switch, read a grant's key or call these internals. The import and call-site
guards in the tests catch accidental drift only.
"""

from __future__ import annotations

import hmac
import logging
import math
import threading
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from app.computer.authority import GRANTABLE_ACTIONS, MAX_GRANT_ENTRIES, MAX_GRANT_SECONDS, AuthorityError, InputGrant, InputStep, canonical_input
from app.computer.base import (
    AllowedApplication,
    ComputerBackend,
    ComputerControlError,
    MouseButton,
    ScreenInfo,
    WindowInfo,
    normalize_key_input,
    validate_exact_window_title,
    validate_process_name,
    validate_url,
)

logger = logging.getLogger("jarvis.computer.input")

WINDOW_SECONDS = 60.0
DEFAULT_ACTIONS_PER_WINDOW = 10
MAX_ACTIVE_GRANTS = 8

#: The nine methods that act on the desktop - they send input, launch a program, open the browser or change which window is in front: gated by the switch, the limit AND a
#: grant. `open_url` binds the exact URL; `switch_window` binds the exact window title and the exact process name.
GATED_METHODS = frozenset(
    {"open_application", "type_text", "press_key", "move_mouse", "click_mouse", "double_click", "scroll", "open_url", "switch_window"}
)
#: Everything else on the interface: passed straight through. A test pins that every method of `ComputerBackend` is in exactly one of the two sets, so a method added
#: later cannot reach the desktop without somebody deciding which set it belongs to.
UNGATED_METHODS = frozenset({"get_active_window", "get_screen_info"})
#: Read-only: they change nothing.
READ_ONLY_METHODS = frozenset({"get_active_window", "get_screen_info"})
#: OS-affecting actions that are NOT protected. None: `open_url` and `switch_window` were listed here until they were put behind the gate. The name stays so that the
#: classification check below, and the tests that pin it, say "nothing is left out" rather than silently losing the concept.
UNPROTECTED_ACTIONS: frozenset[str] = frozenset()

if GATED_METHODS != GRANTABLE_ACTIONS or UNGATED_METHODS != READ_ONLY_METHODS | UNPROTECTED_ACTIONS:  # pragma: no cover - a drifted constant must never import quietly
    raise RuntimeError("the gate's action classification and the authority module's grantable actions disagree")

SWITCHED_OFF = (
    "Computer input is switched off, so nothing was typed, pressed, clicked, scrolled or launched. "
    "(The owner can enable it with JARVIS_COMPUTER_INPUT_ENABLED=true.)"
)
NOT_AUTHORIZED = (
    "Computer input needs an approved authorization that covers this exact action, so nothing was typed, pressed, clicked, scrolled or launched."
)

# Reason codes in the audit log: disabled | rate_limited | no_authority | revoked | expired | exhausted | mismatch | switch_off | not_covered | unknown_action |
# invalid_arguments. The CALLER sees SWITCHED_OFF or the limit message for the first two and NOT_AUTHORIZED for every other code.


def _limited(limit: int, retry_after: int) -> str:
    return f"Computer input is limited to {limit} actions per minute, and that limit has been reached. Try again in {retry_after} second(s)."


@dataclass
class _GrantState:
    """The gate's private record of one registered grant. Never exposed: it holds the keyed entries."""

    grant: InputGrant
    actions: tuple[str, ...]
    macs: tuple[bytes, ...]
    expires_at: float
    purpose: str
    index: int = 0
    closed: str | None = None  # None while usable; else why not: revoked | expired | exhausted | mismatch | switch_off

    @property
    def usable(self) -> bool:
        return self.closed is None and self.index < len(self.macs)


class AdmissionTicket:
    """One ADMITTED action that is still being carried out - today only `type_text`, which the real backend sends in bounded chunks.

    The gate admits the action once (one grant entry, one rate-limit slot) and returns this ticket. Between chunks the backend calls `still_authorized()`, a NON-CONSUMING
    recheck: it spends no entry and takes no slot, and it refuses once the grant was revoked (`revoke` / `revoke_all`), ended by a mismatch, has expired on the gate's
    clock, or the global switch is off. A grant that merely ran out of entries by admitting this very action is NOT a reason to stop: that is how a one-action grant is spent.

    A ticket only ever CONTINUES an action the gate already admitted; it cannot admit anything. It is created by `InputGate.admit(..., keep_open=True)` and nowhere else.
    Revocation is observed at the next `still_authorized()` call: characters already sent to the operating system cannot be recalled."""

    __slots__ = ("_gate", "_grant", "_action", "_expires_at", "_cancelled", "_finished")

    def __init__(self, gate: "InputGate", grant: InputGrant, action: str, expires_at: float):
        self._gate = gate
        self._grant = grant
        self._action = action
        self._expires_at = expires_at
        self._cancelled: str | None = None  # why it was ended: revoked | mismatch | switch_off
        self._finished = False

    def still_authorized(self) -> None:
        """Raise `ComputerControlError` if the admitted action must not send anything more; otherwise return. Consumes nothing."""
        self._gate._recheck(self)

    def close(self) -> None:
        """The action is over (finished, failed or stopped). Idempotent."""
        self._gate._finish(self)

    def __repr__(self) -> str:
        return f"AdmissionTicket(action={self._action})"


class InputGate:
    """Decides, and records, whether one more input action may happen now. See the module docstring for the exact semantics.

    An action is admitted ONLY with a registered grant that covers it (and with the switch on and room in the rolling limit). There is no mode in which a grant is optional: no
    constructor option, no attribute and no argument default turns that requirement off, and nothing here asks whether a test is running. The class is final, so it cannot be
    subclassed to override `admit`; the backends accept nothing but an `InputGate`."""

    def __init_subclass__(cls, **kwargs):
        raise TypeError("InputGate is final: a subclass could override the admission decision, so none is allowed")

    def __init__(
        self,
        enabled: Callable[[], bool],
        limit: Callable[[], int],
        *,
        window_seconds: float = WINDOW_SECONDS,
        clock: Callable[[], float] = time.monotonic,
        max_grants: int = MAX_ACTIVE_GRANTS,
    ):
        self._enabled = enabled
        self._limit = limit
        self._window = window_seconds
        self._clock = clock
        self._max_grants = max_grants
        self._lock = threading.Lock()
        self._events: deque[float] = deque()
        self._grants: dict[str, _GrantState] = {}
        self._open: set[AdmissionTicket] = set()  # admitted actions still being carried out; each is closed by its backend, and one past its grant's lifetime is forgotten

    # -- reading the configuration, failing CLOSED ----------------------------------------------------------------------------

    def _is_enabled(self) -> bool:
        try:
            return self._enabled() is True
        except Exception:  # noqa: BLE001 - a switch that cannot be read is a switch that is off
            return False

    def _current_limit(self) -> int:
        try:
            value = self._limit()
        except Exception:  # noqa: BLE001
            return 0
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return 0  # anything that is not a plain non-negative whole number allows nothing
        return value

    def _expire(self, now: float) -> None:
        # An action stops counting at EXACTLY `window` seconds old: at age 59.999 it counts, at age 60.0 it does not.
        while self._events and now - self._events[0] >= self._window:
            self._events.popleft()

    def _expire_grants(self, now: float) -> None:
        """Mark grants past their lifetime, and forget terminal ones once their lifetime is over (they then read as unknown). Caller holds the lock."""
        for key, state in list(self._grants.items()):
            if state.closed is None and now >= state.expires_at:
                state.closed = "expired"
            if state.closed is not None and now >= state.expires_at:
                del self._grants[key]
        if self._open:
            self._open = {ticket for ticket in self._open if now < ticket._expires_at}  # (a ticket past its lifetime already reads as expired)

    def _cancel_open(self, reason: str, grant: InputGrant | None = None) -> None:
        """End the in-flight actions of `grant` (all of them if None). They stop at their next recheck. Caller holds the lock."""
        for ticket in self._open:
            if ticket._cancelled is None and (grant is None or ticket._grant is grant):
                ticket._cancelled = reason

    def _end_everything_for_switch_off(self, lines: list) -> None:
        """The switch was seen off: no grant and no in-flight action outlives that. Caller holds the lock; `lines` is logged after it is released."""
        for other in self._grants.values():
            if other.closed is None:
                other.closed = "switch_off"
                lines.append((logging.WARNING, "computer_input grant revoked id=%s reason=switch_off", (other.grant.short_id,)))
        self._cancel_open("switch_off")

    def _budget_refusal(self, now: float, limit: int) -> tuple[str, str] | None:
        """(code, text) for why the SWITCH or the LIMIT says no right now, or None. Caller holds the lock and has already expired old events."""
        if not self._is_enabled():
            return "disabled", SWITCHED_OFF
        if len(self._events) >= limit:
            oldest = self._events[0] if self._events else now
            return "rate_limited", _limited(limit, max(1, math.ceil(self._window - (now - oldest))))
        return None

    def _state_of(self, authority: object) -> _GrantState | None:
        """The registered state for exactly THIS grant object, else None. Found by object identity, never by id alone, so a lookalike carrying a copied id is nobody."""
        if not isinstance(authority, InputGrant):
            return None
        state = self._grants.get(authority.id)
        return state if state is not None and state.grant is authority else None

    # -- the non-consuming questions ----------------------------------------------------------------------------------------------

    def check_budget(self) -> None:
        """NON-CONSUMING: raises if the switch or the limit would refuse right now (whatever the grant). Records nothing, so it can never be what limits anyone."""
        with self._lock:
            now = self._clock()
            self._expire(now)
            refusal = self._budget_refusal(now, self._current_limit())
        if refusal is not None:
            logger.warning("computer_input refused action=%s reason=%s stage=preflight", "budget", refusal[0])
            raise ComputerControlError(refusal[1])

    def check(self, action: str, *, authority: InputGrant | None = None) -> None:
        """NON-CONSUMING pre-check: raises if input would be refused right now. It records nothing and spends nothing. With a grant it asks only whether the grant is
        usable and still holds an UNSPENT entry for this action name - not whether it is next, because a whole command is pre-checked before its first step runs."""
        with self._lock:
            now = self._clock()
            self._expire(now)
            self._expire_grants(now)
            refusal = self._budget_refusal(now, self._current_limit())
            code, text = refusal if refusal is not None else (None, None)
            if code is None:
                state = self._state_of(authority)
                if state is None:
                    code = "no_authority"
                elif not state.usable:
                    code = state.closed or "exhausted"
                elif action not in state.actions[state.index :]:
                    code = "not_covered"
                if code is not None:
                    text = NOT_AUTHORIZED
        if code is not None:
            # Logged here too: a command turned away at validation never reaches `admit`, and that attempt must leave a trace as well.
            logger.warning("computer_input refused action=%s reason=%s stage=preflight", action, code)
            raise ComputerControlError(text or NOT_AUTHORIZED)

    def budget_remaining(self) -> int:
        """READ-ONLY: how many more actions the SWITCH and the LIMIT would admit right now - 0 while input is switched off or the limit cannot be read. Takes no slot."""
        with self._lock:
            now = self._clock()
            self._expire(now)
            if not self._is_enabled():
                return 0
            return max(0, self._current_limit() - len(self._events))

    def remaining(self, *, authority: InputGrant | None = None) -> int:
        """READ-ONLY: how many more actions would be admitted right now: the smaller of the rate budget and the grant's unspent entries, and 0 without a usable grant. It
        records nothing, logs nothing and takes no slot; `admit` still decides each action when it happens. (`budget_remaining` answers the rate budget alone.)"""
        with self._lock:
            now = self._clock()
            self._expire(now)
            self._expire_grants(now)
            if not self._is_enabled():
                return 0
            budget = max(0, self._current_limit() - len(self._events))
            state = self._state_of(authority)
            if state is None or not state.usable:
                return 0
            return min(budget, len(state.macs) - state.index)

    def unspent(self, grant: InputGrant) -> int:
        """READ-ONLY: how many entries of THIS grant are still unspent and usable (0 for an unknown, closed, expired or exhausted grant). Reveals nothing about them."""
        with self._lock:
            self._expire_grants(self._clock())
            state = self._state_of(grant)
            return 0 if state is None or not state.usable else len(state.macs) - state.index

    # -- the enforcement ---------------------------------------------------------------------------------------------------------

    def admit(self, action: str, *, authority: InputGrant | None, canonical: bytes | None, keep_open: bool = False) -> AdmissionTicket | None:
        """THE enforcement. Atomically: refuse (raising ComputerControlError, spending no grant entry and taking no rate-limit slot) or spend one grant entry and record
        one action together, and return. A wrong action or argument under a registered grant ALSO revokes that whole grant. `authority` and `canonical` have no defaults: a
        caller must say which grant and which exact action it presents (`None` is refused), so there is no way to ask without one.

        `keep_open=True` is for an action that takes time and is carried out in pieces (`type_text`): it returns an `AdmissionTicket` whose `still_authorized()` the caller
        calls between pieces. The ticket is part of THIS admission - one entry, one slot - and admits nothing further. Without it the return value is None."""
        mac = None
        if isinstance(authority, InputGrant) and isinstance(canonical, bytes):
            try:
                mac = authority._mac(canonical)  # pure, outside the lock: the key never changes
            except Exception:  # noqa: BLE001 - fail closed: no match
                mac = None
        lines: list[tuple[int, str, tuple]] = []  # logged after the lock is released
        with self._lock:
            now = self._clock()
            self._expire(now)
            self._expire_grants(now)
            limit = self._current_limit()
            code: str | None = None
            text = NOT_AUTHORIZED
            state: _GrantState | None = None
            ticket: AdmissionTicket | None = None
            if action not in GATED_METHODS:
                code = "unknown_action"
            elif not self._is_enabled():
                code, text = "disabled", SWITCHED_OFF
                self._end_everything_for_switch_off(lines)  # the switch was seen off: no grant, and no action still being carried out, outlives that
            else:
                state = self._state_of(authority)
                if state is None:
                    code = "no_authority"
                elif not state.usable:
                    code = state.closed or "exhausted"
                elif mac is None or state.actions[state.index] != action or not _same(mac, state.macs[state.index]):
                    code = "mismatch"
                    state.closed = "mismatch"  # a wrong action or argument ends the whole grant
                    self._cancel_open("mismatch", state.grant)  # ... including an action of this grant that is still being carried out
                    lines.append((logging.WARNING, "computer_input grant revoked id=%s reason=mismatch", (state.grant.short_id,)))
                if code is None and len(self._events) >= limit:
                    oldest = self._events[0] if self._events else now
                    code, text = "rate_limited", _limited(limit, max(1, math.ceil(self._window - (now - oldest))))
            if code is None and state is None:  # pragma: no cover - unreachable (no grant sets "no_authority" above); here so that a later edit can never admit without one
                code = "no_authority"
            if code is None:
                self._events.append(now)
                used = len(self._events)
                state.index += 1
                if state.index >= len(state.macs):
                    state.closed = "exhausted"
                    lines.append((logging.INFO, "computer_input grant spent id=%s", (state.grant.short_id,)))
                entry, total = state.index, len(state.macs)
                if keep_open:
                    ticket = AdmissionTicket(self, state.grant, action, state.expires_at)
                    self._open.add(ticket)
        for level, message, args in lines:
            logger.log(level, message, *args)
        if code is not None:
            logger.warning("computer_input refused action=%s reason=%s", action, code)
            raise ComputerControlError(text)
        logger.info("computer_input allowed action=%s grant=%s entry=%d/%d used=%d/%d", action, state.grant.short_id, entry, total, used, limit)
        return ticket

    def _recheck(self, ticket: AdmissionTicket) -> None:
        """NON-CONSUMING: may the action behind `ticket` send its next piece? Spends no grant entry and takes no slot. Refuses (ComputerControlError) when the action is
        already over, its grant was revoked or ended by a mismatch, the grant's lifetime has passed, or the switch is off (which, as at admission, also ends every grant)."""
        if not isinstance(ticket, AdmissionTicket) or ticket._gate is not self:
            raise ComputerControlError(NOT_AUTHORIZED)
        lines: list[tuple[int, str, tuple]] = []
        reason: str | None = None
        with self._lock:
            now = self._clock()
            if ticket._finished:
                reason = "finished"
            elif ticket._cancelled is not None:
                reason = ticket._cancelled
            elif now >= ticket._expires_at:
                reason = ticket._cancelled = "expired"
            elif not self._is_enabled():
                reason = "switch_off"
                self._end_everything_for_switch_off(lines)
        for level, message, args in lines:
            logger.log(level, message, *args)
        if reason is not None:
            logger.warning("computer_input refused action=%s reason=%s stage=continuation", ticket._action, reason)  # (no argument, count or length is ever logged)
            raise ComputerControlError(SWITCHED_OFF if reason == "switch_off" else NOT_AUTHORIZED)

    def _finish(self, ticket: AdmissionTicket) -> None:
        with self._lock:
            ticket._finished = True
            self._open.discard(ticket)

    # -- grants: registered only by the issuer, revoked by anyone holding the grant ----------------------------------------------

    def _register_grant(self, grant: InputGrant, steps: Sequence[InputStep], *, ttl_seconds: float, purpose: str, actor: str) -> None:
        """Called ONLY by `GrantIssuer` (a test pins that no other module touches it). Computes the keyed entries, then registers the grant under the lock. Refuses while the
        switch is off, past the active-grant cap, or for an out-of-range lifetime or size."""
        if isinstance(ttl_seconds, bool) or not isinstance(ttl_seconds, (int, float)) or not 0 < ttl_seconds <= MAX_GRANT_SECONDS:
            raise AuthorityError(f"a grant must last more than 0 and at most {MAX_GRANT_SECONDS:g} seconds.")
        if not isinstance(grant, InputGrant) or not 0 < len(steps) <= MAX_GRANT_ENTRIES:
            raise AuthorityError(f"a grant must list between 1 and {MAX_GRANT_ENTRIES} actions.")
        macs = tuple(grant._mac(step.canonical()) for step in steps)  # raises AuthorityError for anything that cannot be authorized
        actions = tuple(step.action for step in steps)
        with self._lock:
            now = self._clock()
            self._expire(now)
            self._expire_grants(now)
            if len(self._grants) >= 4 * self._max_grants:  # closed records are only kept to explain a refusal: forget them rather than let them pile up
                for key in [key for key, state in self._grants.items() if state.closed is not None]:
                    del self._grants[key]
            if not self._is_enabled():
                refused = SWITCHED_OFF
            elif sum(1 for state in self._grants.values() if state.usable) >= self._max_grants:
                raise AuthorityError("too many authorization grants are active.")
            elif grant.id in self._grants:
                raise AuthorityError("that grant is already registered.")
            else:
                refused = None
                self._grants[grant.id] = _GrantState(grant=grant, actions=actions, macs=macs, expires_at=now + float(ttl_seconds), purpose=purpose)
        if refused is not None:
            logger.warning("computer_input grant refused reason=disabled purpose=%s actor=%s", purpose, actor)
            raise ComputerControlError(refused)
        logger.info("computer_input grant issued id=%s entries=%d ttl=%d purpose=%s actor=%s", grant.short_id, len(steps), int(ttl_seconds), purpose, actor)

    def revoke(self, grant: InputGrant) -> bool:
        """Revoke THIS grant now: no further action is admitted under it, and an action it admitted that is still being carried out (a long `type_text`) sends nothing more
        once it next rechecks. Characters already sent to the operating system cannot be recalled. Idempotent; True if it was still usable."""
        with self._lock:
            state = self._state_of(grant)
            changed = state is not None and state.closed is None
            if changed:
                state.closed = "revoked"
            self._cancel_open("revoked", grant)  # also when the grant was already spent: its last action may still be typing
        if changed:
            logger.warning("computer_input grant revoked id=%s reason=revoked", grant.short_id)
        return bool(changed)

    def revoke_all(self) -> int:
        """Revoke every active grant, and stop every action still being carried out at its next recheck. Returns how many grants were still usable."""
        with self._lock:
            hit = [state for state in self._grants.values() if state.closed is None]
            for state in hit:
                state.closed = "revoked"
            self._cancel_open("revoked")
        for state in hit:
            logger.warning("computer_input grant revoked id=%s reason=revoked_all", state.grant.short_id)
        return len(hit)


def _same(a: bytes, b: bytes) -> bool:
    return hmac.compare_digest(a, b)


def require_input_gate(gate: object) -> InputGate:
    """A backend accepts a gate only if it IS an `InputGate` (a final class), never something that merely has the same methods: a stand-in whose `admit` says yes would be a
    way around every rule. Raises TypeError for anything else."""
    if not isinstance(gate, InputGate):
        raise TypeError("a backend can only enforce a real InputGate")
    return gate


def admit_input(gate: InputGate, authority: InputGrant | None, action: str, *args: object, keep_open: bool = False) -> AdmissionTicket | None:
    """What every backend calls immediately before an OS-input call, AFTER validating its arguments: turn the validated, normalized arguments into their one canonical form
    and ask the gate. An argument that cannot be put in canonical form cannot be authorized, so it is refused. With `keep_open=True` (an action carried out in pieces) the
    admission's `AdmissionTicket` is returned; the caller rechecks it between pieces and closes it when done."""
    require_input_gate(gate)
    try:
        canonical = canonical_input(action, *args)
    except AuthorityError:
        logger.warning("computer_input refused action=%s reason=invalid_arguments", action if action in GATED_METHODS else "unknown")
        raise ComputerControlError(NOT_AUTHORIZED) from None
    if keep_open:
        return gate.admit(action, authority=authority, canonical=canonical, keep_open=True)
    gate.admit(action, authority=authority, canonical=canonical)
    return None


class GatedComputerBackend(ComputerBackend):
    """Wraps a backend that does NOT enforce the gate itself. The nine acting methods are admitted by the gate first (validating their arguments first, as the real
    backend does); the two read-only methods are passed straight through. (`WindowsComputerBackend` enforces the gate itself, so it is never wrapped.)"""

    def __init__(self, inner: ComputerBackend, gate: InputGate, authority: InputGrant | None = None):
        require_input_gate(gate)
        if getattr(inner, "gate", None) is not None:
            raise ValueError("this backend already enforces a Computer Input Gate; wrapping it would count every action twice")
        self._inner = inner
        self.gate = gate
        self._authority = authority

    @property
    def inner(self) -> ComputerBackend:
        return self._inner

    def check_input(self, action: str) -> None:
        """What the tools call from `validate_arguments`: the gate's non-consuming pre-check, with this backend's own grant (if any)."""
        self.gate.check(action, authority=self._authority)

    # -- passed straight through -------------------------------------------------------------------------------------------------

    def get_active_window(self) -> WindowInfo:
        return self._inner.get_active_window()

    def get_screen_info(self) -> ScreenInfo:
        return self._inner.get_screen_info()

    # -- gated: the nine acting methods ------------------------------------------------------------------------------------------

    def open_url(self, url: str) -> None:
        validate_url(url)  # validation FIRST, as in the real backend: an invalid URL is turned away before the gate and costs no slot or entry
        admit_input(self.gate, self._authority, "open_url", url)
        self._inner.open_url(url)

    def switch_window(self, title: str, process: str) -> WindowInfo:
        exact_title, exact_process = validate_exact_window_title(title), validate_process_name(process)
        admit_input(self.gate, self._authority, "switch_window", exact_title, exact_process)
        return self._inner.switch_window(exact_title, exact_process)

    def open_application(self, app: AllowedApplication) -> None:
        admit_input(self.gate, self._authority, "open_application", app)
        self._inner.open_application(app)

    def type_text(self, text: str) -> None:
        admit_input(self.gate, self._authority, "type_text", text)
        self._inner.type_text(text)

    def press_key(self, key: str) -> None:
        tokens = normalize_key_input(key)  # validation FIRST, as in the real backend: an invalid key is turned away before the gate and costs no slot or entry
        admit_input(self.gate, self._authority, "press_key", tokens)
        self._inner.press_key(key)

    def move_mouse(self, x: int, y: int) -> None:
        admit_input(self.gate, self._authority, "move_mouse", x, y)
        self._inner.move_mouse(x, y)

    def click_mouse(self, x: int | None, y: int | None, button: MouseButton) -> None:
        admit_input(self.gate, self._authority, "click_mouse", x, y, button)
        self._inner.click_mouse(x, y, button)

    def double_click(self, x: int, y: int, button: MouseButton) -> None:
        admit_input(self.gate, self._authority, "double_click", x, y, button)
        self._inner.double_click(x, y, button)

    def scroll(self, amount: int) -> None:
        admit_input(self.gate, self._authority, "scroll", amount)
        self._inner.scroll(amount)


def create_default_input_gate() -> InputGate:
    """A NEW gate bound to the settings - the ONLY place production code builds a gate. Both settings are read at every call, so changing one takes effect at once."""
    from app.config import settings  # imported here so this module has no import-time dependency on configuration

    return InputGate(
        enabled=lambda: settings.computer_input_enabled,
        limit=lambda: settings.computer_input_actions_per_minute,
    )


_process_gate: InputGate | None = None
_process_gate_lock = threading.Lock()


def default_input_gate() -> InputGate:
    """THE process-wide gate: one counter for every backend in this process that was not handed a gate of its own - which is what makes the limit
    ONE shared budget however many backends or tools exist, and what lets `WindowsComputerBackend()` be gated wherever it is constructed."""
    global _process_gate
    with _process_gate_lock:
        if _process_gate is None:
            _process_gate = create_default_input_gate()
        return _process_gate
