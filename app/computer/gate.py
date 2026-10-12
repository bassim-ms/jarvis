"""
The Computer Input Gate: the ONE place that decides whether Jarvis may send input to the desktop.

    a chat command / a registry call / a chained command  ->  an input TOOL  ->  a backend that enforces the gate  ->  InputGate.admit()  ->  the desktop

The gate is enforced by the backend ITSELF, at its lowest layer: `WindowsComputerBackend` calls `InputGate.admit()` immediately before each OS call that
sends input, using ONE process-wide gate unless it is handed another. `GatedComputerBackend` (below) does the same job for any OTHER backend that does
not enforce the gate itself (a test double, a future backend); it refuses to wrap one that does, so an action can never be counted twice.

The six INPUT methods - `type_text`, `press_key`, `move_mouse`, `click_mouse`, `double_click`, `open_application` - can drive the keyboard, the mouse
and program launches. They were SAFE_ACTION (no confirmation), so one anonymous chat message could press Win+R, type a command and press Enter. This
gate closes that path without adding any capability:

  * SWITCH     `computer_input_enabled` (default OFF) is read on EVERY call. Off, no input method reaches the desktop.
  * LIMIT      ONE limit shared by all six: at most `computer_input_actions_per_minute` (default 10) admitted actions in any rolling 60 seconds.
               An action admitted at time t stops counting at exactly t + 60 s (so at t + 59.999 s it still counts; at t + 60 s its slot is free).
               A refused attempt - switched off or over the limit - uses up NO slot.
  * HERE       Enforcement is at EXECUTION time, in the backend every tool ends in, so it cannot be skipped by calling a tool's `execute`
               directly, by `ToolRegistry.execute`, by a chained "use tool ..., then ..." command, by a caller that never ran `validate_arguments`, or
               by building a backend somewhere the factory never saw.
               The tools ALSO pre-check (`check_input`, which consumes nothing) from `validate_arguments`, so a whole command is turned away before its
               first step runs when input is off or the limit is already reached; that pre-check is a convenience, never the enforcement.
  * ATOMIC     Counting and recording one action is a single step under a lock, so any number of concurrent calls admit at most the limit.
  * AUDIT      Every admitted and every refused attempt is logged (logger `jarvis.computer.input`): the action's name, the outcome, the reason and
               the count. NEVER typed text, key names, coordinates, application names or any other argument VALUE - at most a coarse SHAPE
               ("12 characters", "2 parts") that says how much, not what.

What it does NOT do: it grants nothing, adds no tool or permission, keeps no state beyond a bounded in-memory list of recent timestamps (a restart
forgets it, and the switch is off by default anyway), and does not touch the read-only tools or the Job allowlist (`open_url`, `scroll`,
`switch_window`, `get_active_window`, screen info): those methods are passed straight through.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from collections.abc import Callable

from app.computer.base import (
    AllowedApplication,
    ComputerBackend,
    ComputerControlError,
    MouseButton,
    ScreenInfo,
    WindowInfo,
)

logger = logging.getLogger("jarvis.computer.input")

WINDOW_SECONDS = 60.0
DEFAULT_ACTIONS_PER_WINDOW = 10

#: The six methods that send input to the desktop: gated.
GATED_METHODS = frozenset({"open_application", "type_text", "press_key", "move_mouse", "click_mouse", "double_click"})
#: Everything else on the interface: passed straight through. A test pins that every method of `ComputerBackend` is in exactly one of the two
#: sets, so a method added later cannot reach the desktop without somebody deciding which set it belongs to.
UNGATED_METHODS = frozenset({"get_active_window", "get_screen_info", "open_url", "switch_window", "scroll"})

SWITCHED_OFF = (
    "Computer input is switched off, so nothing was typed, pressed, clicked or launched. "
    "(The owner can enable it with JARVIS_COMPUTER_INPUT_ENABLED=true.)"
)


def _limited(limit: int, retry_after: int) -> str:
    return f"Computer input is limited to {limit} actions per minute, and that limit has been reached. Try again in {retry_after} second(s)."


class InputGate:
    """Decides, and records, whether one more input action may happen now. See the module docstring for the exact semantics."""

    def __init__(
        self,
        enabled: Callable[[], bool],
        limit: Callable[[], int],
        *,
        window_seconds: float = WINDOW_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._enabled = enabled
        self._limit = limit
        self._window = window_seconds
        self._clock = clock
        self._lock = threading.Lock()
        self._events: deque[float] = deque()

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

    def _refusal(self, now: float, limit: int) -> str | None:
        """The reason input is not allowed right now, or None. Caller holds the lock and has already expired old events."""
        if not self._is_enabled():
            return SWITCHED_OFF
        if len(self._events) >= limit:
            oldest = self._events[0] if self._events else now
            return _limited(limit, max(1, math.ceil(self._window - (now - oldest))))
        return None

    # -- the two operations ------------------------------------------------------------------------------------------------------

    def check(self, action: str) -> None:
        """NON-CONSUMING pre-check: raises if input would be refused right now. It records nothing, so it can never be what limits anyone."""
        with self._lock:
            now = self._clock()
            self._expire(now)
            reason = self._refusal(now, self._current_limit())
        if reason is not None:
            # Logged here too: a command turned away at validation never reaches `admit`, and that attempt must leave a trace as well. The command
            # planner pre-checks EVERY step, so a refused three-step command leaves three lines (one per step, none carrying a value); and a refusal
            # here stops the whole command, so `admit` never runs for it and no execution-time line duplicates these.
            logger.warning("computer_input refused action=%s reason=%s stage=preflight", action, "disabled" if reason is SWITCHED_OFF else "rate_limited")
            raise ComputerControlError(reason)

    def remaining(self) -> int:
        """READ-ONLY: how many more actions would be admitted if they were attempted right now - 0 while input is switched off or the limit cannot be
        read. It records nothing, logs nothing and takes no slot, so it can never be what limits anyone and is never the enforcement (`admit` is). A
        caller that needs several actions in a row (a multi-step task) can use it to refuse to START one the budget cannot finish; the answer is a
        snapshot, and `admit` still decides each action when it happens."""
        with self._lock:
            now = self._clock()
            self._expire(now)
            if not self._is_enabled():
                return 0
            return max(0, self._current_limit() - len(self._events))

    def admit(self, action: str, shape: str = "") -> None:
        """THE enforcement. Atomically: refuse (raising ComputerControlError, consuming nothing) or record one action and return."""
        with self._lock:
            now = self._clock()
            self._expire(now)
            limit = self._current_limit()
            reason = self._refusal(now, limit)
            if reason is None:
                self._events.append(now)
                used = len(self._events)
        if reason is not None:
            kind = "disabled" if reason is SWITCHED_OFF else "rate_limited"
            logger.warning("computer_input refused action=%s reason=%s", action, kind)
            raise ComputerControlError(reason)
        logger.info("computer_input allowed action=%s used=%d/%d%s", action, used, limit, f" shape={shape}" if shape else "")


class GatedComputerBackend(ComputerBackend):
    """Wraps a backend that does NOT enforce the gate itself. The six input methods are admitted by the gate first; every other method is passed
    straight through. (`WindowsComputerBackend` enforces the gate itself, so it is never wrapped.)"""

    def __init__(self, inner: ComputerBackend, gate: InputGate):
        if getattr(inner, "gate", None) is not None:
            raise ValueError("this backend already enforces a Computer Input Gate; wrapping it would count every action twice")
        self._inner = inner
        self.gate = gate

    @property
    def inner(self) -> ComputerBackend:
        return self._inner

    def check_input(self, action: str) -> None:
        """What the tools call from `validate_arguments`: the gate's non-consuming pre-check."""
        self.gate.check(action)

    # -- passed straight through -------------------------------------------------------------------------------------------------

    def get_active_window(self) -> WindowInfo:
        return self._inner.get_active_window()

    def get_screen_info(self) -> ScreenInfo:
        return self._inner.get_screen_info()

    def open_url(self, url: str) -> None:
        self._inner.open_url(url)

    def switch_window(self, title: str) -> WindowInfo:
        return self._inner.switch_window(title)

    def scroll(self, amount: int) -> None:
        self._inner.scroll(amount)

    # -- gated: the six input methods --------------------------------------------------------------------------------------------

    def open_application(self, app: AllowedApplication) -> None:
        self.gate.admit("open_application")
        self._inner.open_application(app)

    def type_text(self, text: str) -> None:
        self.gate.admit("type_text", f"chars={len(text) if isinstance(text, str) else 'n/a'}")
        self._inner.type_text(text)

    def press_key(self, key: str) -> None:
        self.gate.admit("press_key", f"parts={len(key.split('+')) if isinstance(key, str) else 'n/a'}")
        self._inner.press_key(key)

    def move_mouse(self, x: int, y: int) -> None:
        self.gate.admit("move_mouse")
        self._inner.move_mouse(x, y)

    def click_mouse(self, x: int | None, y: int | None, button: MouseButton) -> None:
        self.gate.admit("click_mouse")
        self._inner.click_mouse(x, y, button)

    def double_click(self, x: int, y: int, button: MouseButton) -> None:
        self.gate.admit("double_click")
        self._inner.double_click(x, y, button)


def create_default_input_gate() -> InputGate:
    """A NEW gate bound to the settings. Both settings are read at every call, so changing one takes effect at once."""
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
