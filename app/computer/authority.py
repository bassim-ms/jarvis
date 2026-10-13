"""
Authorization grants for computer input: WHAT a grant is, and how an action is matched against it. (Where grants are minted: `app/computer/issuer.py`. Where they
are enforced: `app/computer/gate.py`, called by the backend immediately before every OS-input call.)

    switch on  +  a grant that covers THIS exact action, with THESE exact arguments, next in order  =  the action may run

The global switch (`JARVIS_COMPUTER_INPUT_ENABLED`) stays necessary but is no longer sufficient. A grant is a short-lived, single-use, ordered list of exact actions:

  * EXACT SCOPE   Each entry is an action name plus its canonical arguments. A grant for "type this text" cannot be used to type different text, to press a key, or
                  to do the same action later in the order than it was approved for.
  * KEYED MATCH   The grant never stores or compares raw arguments. Each entry is an HMAC-SHA256 (a random per-grant key) of the canonical form, and a presented action is
                  matched by recomputing that HMAC. No argument value, no digest and no length is ever logged, so a log line cannot reveal what was typed.
  * UNGUESSABLE   A grant's identity is 128 random bits and its key 256 random bits; neither is derived from anything. The grant OBJECT is the credential: it is held
                  by reference, never looked up by id alone, and it cannot be copied or pickled.
  * BY REFERENCE  Authority lives in an explicit handle passed to a backend instance (`WindowsComputerBackend(authority=grant)`), never in ambient context, so a thread,
                  a request, a scheduler or a Job gains nothing unless someone hands it that exact backend.

HONEST LIMITS. This is a defense against accidental and prompted misuse inside a cooperating program. It is NOT a security boundary against arbitrary code running in
this Python process: such code can read a grant's key, call the gate's internals, set `settings.computer_input_enabled`, or drive the OS directly. The import and
call-site guards in the tests catch ACCIDENTAL architectural drift only. Stopping a hostile in-process actor would need a separate broker process, which is out of scope.

WINDOW AND BROWSER ACTIONS. `open_url` (opens the default browser on a URL and forces it to the front) and `switch_window` (brings a window to the front, restoring it if it
is minimized) change which window is in front, so they are grantable like every other action that acts on the desktop. `open_url` binds the EXACT URL string. `switch_window`
binds the EXACT window title and the EXACT process name: the window is looked up once, after admission, and only an exact title-plus-process match is switched to. What a
grant does NOT bind is a window INSTANCE: two windows with the same title and process are an error, never a choice.
"""

from __future__ import annotations

import hashlib
import hmac
import re
import secrets
from dataclasses import dataclass
from typing import Any

from app.computer.base import AllowedApplication, MouseButton

#: Every action a grant can cover: the backend methods that act on the desktop (send input, launch a program, open the browser, change which window is in front).
GRANTABLE_ACTIONS = frozenset(
    {"open_application", "type_text", "press_key", "move_mouse", "click_mouse", "double_click", "scroll", "open_url", "switch_window"}
)

MAX_GRANT_SECONDS = 300.0  # a grant never lives longer than this, whatever was asked for
MAX_GRANT_ENTRIES = 32  # and never lists more actions than this
_LABEL = re.compile(r"[a-z0-9][a-z0-9_.-]{0,39}")  # purpose / actor labels: short, fixed vocabulary, safe to log

_DOMAIN = b"jarvis.computer.input.v1\x00"
_ARITY = {
    "open_application": 1, "type_text": 1, "press_key": 1, "move_mouse": 2, "click_mouse": 3, "double_click": 3, "scroll": 1,
    "open_url": 1,  # the exact URL string
    "switch_window": 2,  # the exact window title, the exact (lower-cased) process file name
}


class AuthorityError(ValueError):
    """A grant, a step or a label is not acceptable. The message never contains an argument value."""


def check_label(label: object, what: str) -> str:
    if not isinstance(label, str) or not _LABEL.fullmatch(label):
        raise AuthorityError(f"{what} must be a short lowercase label (letters, digits, '_', '.', '-').")
    return label


def _field(data: bytes) -> bytes:
    return len(data).to_bytes(4, "big") + data


def _utf8(text: str) -> bytes:
    """UTF-8 for one string argument. Text that cannot be encoded (a lone surrogate, which `validate_text` and `validate_url` let through) cannot be authorized, so it is the same
    `AuthorityError` as any other argument that cannot be put in canonical form - never a raw `UnicodeEncodeError`, and never the offending character in the message."""
    try:
        return text.encode("utf-8")
    except UnicodeEncodeError:
        raise AuthorityError("an action argument contains text that is not valid Unicode, so it cannot be authorized.") from None


def _encode(value: Any) -> bytes:
    """One argument, as typed, length-prefixed bytes: a string can never be mistaken for a number, a number for a key list, or one field for two."""
    if value is None:
        return b"n"
    if isinstance(value, bool):
        raise AuthorityError("a boolean is not a valid action argument.")
    if isinstance(value, (AllowedApplication, MouseButton)):
        return b"e" + _field(value.value.encode("utf-8"))
    if isinstance(value, str):
        return b"s" + _field(_utf8(value))
    if isinstance(value, int):
        return b"i" + _field(str(value).encode("ascii"))
    if isinstance(value, (tuple, list)) and all(isinstance(item, str) for item in value):
        return b"t" + len(value).to_bytes(2, "big") + b"".join(_field(_utf8(item)) for item in value)
    raise AuthorityError("an action argument has a type that cannot be authorized.")


def canonical_input(action: str, *args: Any) -> bytes:
    """The ONE canonical form of an input action and its (already validated, already normalized) arguments. The plan author and the backend both call this, so
    they cannot disagree. `press_key` takes the NORMALIZED token tuple (`normalize_key_input`); `open_application` the enum member; the rest as validated."""
    if action not in GRANTABLE_ACTIONS:
        raise AuthorityError("that is not an action a grant can cover.")
    if len(args) != _ARITY[action]:
        raise AuthorityError("an action was given the wrong number of arguments.")
    return _DOMAIN + _field(action.encode("ascii")) + b"".join(_encode(arg) for arg in args)


@dataclass(frozen=True)
class InputStep:
    """One approved action: its name and its exact, normalized arguments. Built by whoever proposes the plan; it holds the values only until it is turned into a keyed entry."""

    action: str
    args: tuple = ()

    def canonical(self) -> bytes:
        return canonical_input(self.action, *self.args)


class InputGrant:
    """The credential. Its identity and key are random; it is registered with ONE gate, which finds it by this object, not by its id. It cannot be copied or pickled,
    and its repr and logs show a short id prefix only - never the key, never an entry."""

    __slots__ = ("_id", "_key", "__weakref__")

    def __init__(self) -> None:
        self._id = secrets.token_hex(16)
        self._key = secrets.token_bytes(32)

    @property
    def id(self) -> str:
        return self._id

    @property
    def short_id(self) -> str:
        return self._id[:8]

    def _mac(self, canonical: bytes) -> bytes:
        return hmac.new(self._key, canonical, hashlib.sha256).digest()

    def __repr__(self) -> str:
        return f"InputGrant(id={self.short_id}...)"

    def __copy__(self) -> "InputGrant":
        raise TypeError("an InputGrant cannot be copied")

    def __deepcopy__(self, memo: object) -> "InputGrant":
        raise TypeError("an InputGrant cannot be copied")

    def __reduce__(self) -> Any:
        raise TypeError("an InputGrant cannot be pickled")
