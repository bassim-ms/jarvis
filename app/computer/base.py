"""
Computer Control abstraction (M17.1-A): a narrow, read-mostly interface
for observing and lightly driving the local Windows desktop, kept fully
decoupled from any specific automation library - the same "interface +
swappable backend" shape as AIProvider (app/providers/base.py),
SearchProvider (app/search/base.py), and CalendarProvider
(app/calendar/base.py).

`ComputerBackend` is the only thing computer-control Tools
(app/computer/tools.py) depend on; `WindowsComputerBackend`
(app/computer/windows_backend.py, built on PyAutoGUI + pywin32) is the
only implementation for now, but nothing above this boundary ever sees a
PyAutoGUI/pywin32 type or a raw OS error - only the dataclasses/errors
below. A fake backend (see tests/test_computer_tools.py) implements the
same interface with no OS interaction at all, so Tool behavior is fully
testable without ever moving a real mouse, typing into a real keyboard,
launching a real application, or touching a real screen.

This is a FOUNDATION milestone only: the LLM must never gain arbitrary OS
access through this module. Concretely:

- `open_application` takes an `AllowedApplication` - a small, fixed,
  closed enum - never a caller-supplied executable path or command
  string. A concrete backend maps each member to one, single, hardcoded
  launch command; there is no way to reach this method with an arbitrary
  path, and no "escape hatch" parameter that accepts one.
- `open_url` only ever opens a URL through the normal OS/browser
  mechanism (`webbrowser.open` in WindowsComputerBackend) - never a shell
  command, and `validate_url` below rejects every scheme except http/
  https before a concrete backend ever sees the URL.
- `press_key` accepts either ONE key from `ALLOWED_KEYS` - a small,
  fixed, curated set of named keys (enter, tab, arrows, function keys,
  ...), matched case-insensitively and validated by `validate_key`
  below - or, since M17.1-D, a narrowly-scoped "+"-joined modifier
  combination (e.g. "ctrl+l", "ctrl+shift+esc"), validated by
  `validate_key_combo` below: at most `MAX_COMBO_TOKENS` tokens, every
  token but the last drawn from the closed `MODIFIER_KEYS` set, and the
  last drawn from `ALLOWED_COMBO_TARGETS` (a single a-z/0-9 character or
  one of `ALLOWED_KEYS`' own non-modifier names) - never an arbitrary
  string, path, URL, or shell token passed straight to the underlying
  automation library either way.
- `type_text` takes plain text only, bounded by `MAX_TEXT_LENGTH`
  (`validate_text` below) - it types characters, it never interprets the
  text as a command, path, or key combination.
- There is no shell-execution method, no filesystem-write method, no
  credential-access method, and no method that accepts a caller-supplied
  path or command anywhere in this interface. Nothing in this module
  ever imports `subprocess` with `shell=True`.

`validate_url`/`validate_key`/`validate_key_combo`/`validate_text`/
`validate_scroll_amount`/`validate_coordinates` live here, not
duplicated per-backend or per-tool, so there is exactly ONE definition
of "safe input" for each of those things - both app/computer/tools.py
(the boundary that receives raw, LLM/user-supplied parameters) and
WindowsComputerBackend (defense in depth, in case a future caller ever
invokes a backend directly) call the same functions, via
`normalize_key_input` for `press_key` specifically (see below).
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from urllib.parse import urlsplit


class ComputerControlError(RuntimeError):
    """Raised when a computer-control backend can't complete a request -
    an application that isn't installed, an out-of-bounds coordinate, a
    key/URL/text that fails validation, or an underlying OS/automation
    library failure. Never a raw PyAutoGUI/pywin32 exception past this
    boundary. Callers (Tools) turn this into a normal ToolResult.error,
    the same convention CalendarProviderError/SearchProviderError already
    use at their own boundaries."""


@dataclass(frozen=True)
class WindowInfo:
    """The single foreground (active) window, in a shape common to every
    backend. `process_name` is the owning process's executable filename
    only (e.g. "notepad.exe") - never a full path, since a full path can
    reveal local usernames/directory layout and nothing here needs it.
    `process_name` is `"unknown"` when a backend can't resolve it (e.g. a
    protected process) rather than raising - this is a read tool, so a
    partial answer beats a failure."""

    title: str
    process_name: str
    is_maximized: bool


@dataclass(frozen=True)
class ScreenInfo:
    """The primary display's size in pixels. Used by the Tools layer to
    validate that a mouse-move/click coordinate is actually on-screen
    before it ever reaches a backend - see `MoveMouseTool`/`ClickMouseTool`
    in app/computer/tools.py."""

    width: int
    height: int


class MouseButton(str, Enum):
    """The only three mouse buttons a click can name - a closed set,
    never an arbitrary string passed through to the automation library."""

    LEFT = "left"
    RIGHT = "right"
    MIDDLE = "middle"


class AllowedApplication(str, Enum):
    """The ENTIRE closed set of applications `open_application` can ever
    launch. This is the "narrow, explicit mechanism" the milestone
    requires instead of accepting an arbitrary executable path: a caller
    (ultimately an LLM, via a Tool parameter) can only ever name one of
    these fixed members - there is no member, parameter, or code path
    that accepts a free-form path or command string. Extending this set
    later means adding an enum member plus one fixed launch command in
    WindowsComputerBackend, never widening it to accept caller input.
    """

    NOTEPAD = "notepad"
    CALCULATOR = "calculator"
    FILE_EXPLORER = "file_explorer"
    PAINT = "paint"
    CHROME = "chrome"
    EDGE = "edge"
    FIREFOX = "firefox"


#: The entire closed set of key names a STANDALONE `press_key` call
#: accepts - navigation/control keys only (not a full keyboard), matching
#: the names PyAutoGUI itself recognizes (see WindowsComputerBackend).
#: Ordinary text entry goes through `type_text` instead; `press_key`
#: exists for the keys `type_text` can't express (Enter, Tab, Escape,
#: arrows, function keys, ...). A bare letter/digit is deliberately NOT
#: included here - it has no purpose `type_text` doesn't already cover,
#: and keeping this set to "keys with no printable-text meaning" is what
#: makes it reviewable at a glance. Matched CASE-INSENSITIVELY by
#: `validate_key` (M17.1-D) - this set itself stays lowercase-only and is
#: never expanded to widen what a single key press can do; see
#: `MODIFIER_KEYS`/`ALLOWED_COMBO_TARGETS` below for the SEPARATE,
#: narrowly-scoped set a "+"-joined combination's tokens are checked
#: against instead.
ALLOWED_KEYS = frozenset(
    {
        "enter",
        "tab",
        "esc",
        "escape",
        "space",
        "backspace",
        "delete",
        "up",
        "down",
        "left",
        "right",
        "home",
        "end",
        "pageup",
        "pagedown",
        "capslock",
        "shift",
        "ctrl",
        "alt",
        "win",
        "insert",
        "f1",
        "f2",
        "f3",
        "f4",
        "f5",
        "f6",
        "f7",
        "f8",
        "f9",
        "f10",
        "f11",
        "f12",
    }
)

#: The ONLY four tokens allowed in a key combination's modifier
#: position(s) - i.e. every token but the last (M17.1-D). A closed set,
#: distinct from `ALLOWED_KEYS`: each of these can ALSO still be pressed
#: alone via the standalone `press_key` path (they're members of
#: `ALLOWED_KEYS` too), but only these four may appear as a combination's
#: leading token(s).
MODIFIER_KEYS = frozenset({"ctrl", "alt", "shift", "win"})

#: The closed set of tokens allowed as a key combination's FINAL
#: (target) token - a single lowercase letter or digit, or one of
#: `ALLOWED_KEYS`' own named keys, EXCLUDING the modifier names
#: themselves: a combination's last token is always the key actually
#: being "pressed while the modifiers are held," never another modifier
#: - "ctrl+alt" (two modifiers, no real target) is rejected, not treated
#: as a valid combination. Derived from `ALLOWED_KEYS` rather than a
#: second hand-maintained list, so the two can never silently drift
#: apart.
ALLOWED_COMBO_TARGETS = frozenset("abcdefghijklmnopqrstuvwxyz0123456789") | (
    ALLOWED_KEYS - MODIFIER_KEYS
)

#: A key combination has at most this many "+"-separated tokens - one
#: modifier plus a target ("ctrl+l"), or two modifiers plus a target
#: ("ctrl+shift+esc") - never an unbounded chain.
MAX_COMBO_TOKENS = 3

#: `type_text` is bounded so a single call can't be used to paste an
#: unbounded amount of text (or, e.g., a whole file's contents) through
#: the keyboard - a generous bound for real typing, not a meaningful
#: limit on ordinary use.
MAX_TEXT_LENGTH = 1000

#: `scroll`'s magnitude is bounded to a single reasonable "scroll
#: gesture" worth of movement - never unbounded.
MAX_SCROLL_AMOUNT = 2000


#: `switch_window` is bounded: the query is a short plain-text title fragment...
MAX_WINDOW_QUERY_CHARS = 100

#: ...and a window is never a switch target if its process is a credential store, a login / elevation prompt, a
#: shell or a system-management tool (switching there is "arbitrary process manipulation" territory) - or if its
#: title looks like a credential / sign-in window. A fixed, reviewable deny-list, not a permission system.
SWITCH_BLOCKED_PROCESSES = frozenset(
    {
        "keepass.exe", "keepassxc.exe", "1password.exe", "bitwarden.exe", "lastpass.exe", "dashlane.exe", "enpass.exe",
        "roboform.exe", "credentialuibroker.exe", "consent.exe", "logonui.exe", "winlogon.exe", "lockapp.exe",
        "cmd.exe", "powershell.exe", "pwsh.exe", "windowsterminal.exe", "wt.exe", "conhost.exe", "regedit.exe",
        "taskmgr.exe", "mmc.exe", "services.exe",
    }
)
SWITCH_BLOCKED_TITLE_WORDS = (
    "password", "passcode", "credential", "keychain", "vault", "authenticator", "sign in", "sign-in", "log in",
    "login", "two-factor", "2fa", "private key", "wallet", "user account control",
)


def validate_window_query(query: str) -> str:
    """The only validation `switch_window` applies to its argument: a non-empty, short, printable title fragment."""
    if not isinstance(query, str) or not query.strip():
        raise ComputerControlError("window title must be a non-empty string.")
    cleaned = " ".join(query.split())
    if len(cleaned) > MAX_WINDOW_QUERY_CHARS:
        raise ComputerControlError(f"window title exceeds the maximum length of {MAX_WINDOW_QUERY_CHARS} characters.")
    if not cleaned.isprintable():
        raise ComputerControlError("window title must be plain printable text.")
    return cleaned


def validate_exact_window_title(title: str) -> str:
    """The title `switch_window` is asked for, which must be given EXACTLY: `validate_window_query`'s rules (non-empty, short, printable) and no leading, trailing or repeated
    whitespace, so what is requested, what a grant binds and what a window must equal are one and the same string."""
    cleaned = validate_window_query(title)
    if cleaned != title:
        raise ComputerControlError("window title must be given exactly: no leading, trailing or repeated whitespace.")
    return cleaned


#: A process is named by its file name: letters, digits and a few punctuation marks, ending in `.exe`, at most this long. No path, no wildcard.
MAX_PROCESS_NAME_CHARS = 64
_PROCESS_NAME = re.compile(r"[a-z0-9_][a-z0-9_ .+()~-]*\.exe")


def validate_process_name(process: str) -> str:
    """The process file name `switch_window` is asked for, lower-cased (Windows file names are case-insensitive, so `Notepad.EXE` and `notepad.exe` are the same name and
    one authorization). Anything that is not a plain ASCII `*.exe` file name is refused: no path, no wildcard, no padding, and never "unknown"."""
    if not isinstance(process, str) or not process.isascii():
        raise ComputerControlError("process must be a plain ASCII file name ending in .exe.")
    lowered = process.lower()
    if len(lowered) > MAX_PROCESS_NAME_CHARS or not _PROCESS_NAME.fullmatch(lowered):
        raise ComputerControlError("process must be a plain file name ending in .exe (no path, wildcard or padding).")
    return lowered


def exact_window_matches(windows, title: str, process: str) -> list:
    """The windows (of `(hwnd, title, process)` tuples) whose FULL title equals `title` exactly (case-sensitive) and whose process file name, lower-cased, equals `process`,
    that may be a `switch_window` target at all (`is_switchable`). No substring, prefix, wildcard or fuzzy matching; a pure function that touches nothing. The real backend and
    the Notepad demo's read-only pre-wait both call exactly this, so they cannot disagree about what a match is."""
    return [window for window in windows_of_process(windows, process) if window[1] == title]


def windows_of_process(windows, process: str) -> list:
    """The windows whose process file name, lower-cased, equals `process` and that may be a `switch_window` target at all (`is_switchable`): the first half of the exact match,
    and what the Notepad demo lists as "the notepad.exe windows that are visible" in a failure report. Pure; touches nothing."""
    wanted = process.lower()
    return [window for window in windows if str(window[2]).lower() == wanted and is_switchable(window[1], str(window[2]))]


def is_switchable(title: str, process_name: str) -> bool:
    """Whether a window with this title / process may ever be a `switch_window` target."""
    lowered = (title or "").lower()
    if (process_name or "").lower() in SWITCH_BLOCKED_PROCESSES:
        return False
    return not any(word in lowered for word in SWITCH_BLOCKED_TITLE_WORDS)


def validate_url(url: str) -> str:
    """The only validation `open_url` ever applies: `url` must be a
    string with an http/https scheme and a host. Every other scheme
    (file:, javascript:, a custom/registered protocol handler that could
    itself launch an arbitrary application, ...) is rejected here, before
    any backend ever sees it - opening a URL must only ever open it
    through the normal browser mechanism, never anything else a URL
    string could otherwise trigger."""
    if not isinstance(url, str) or not url.strip():
        raise ComputerControlError("url must be a non-empty string.")

    parsed = urlsplit(url)
    if parsed.scheme not in ("http", "https"):
        raise ComputerControlError(
            f"url must use http or https, got scheme: {parsed.scheme!r}"
        )
    if not parsed.netloc:
        raise ComputerControlError(f"url must include a host: {url!r}")
    return url


def validate_key(key: str) -> str:
    """Validates a STANDALONE (non-combination) key press: `key`,
    case-folded, must be a member of `ALLOWED_KEYS` above - never an
    arbitrary string forwarded to the automation library. Matching is
    case-INSENSITIVE (M17.1-D) - "ENTER"/"Enter"/"enter" are all
    accepted - but the returned value is always the canonical lowercase
    form; every caller MUST use that RETURNED value, never the original
    `key` argument, since PyAutoGUI's own key names are lowercase-only
    and silently no-op (rather than error) on an unrecognized casing."""
    if not isinstance(key, str):
        raise ComputerControlError(f"key is not an allowed key: {key!r}")
    normalized = key.lower()
    if normalized not in ALLOWED_KEYS:
        raise ComputerControlError(f"key is not an allowed key: {key!r}")
    return normalized


def validate_key_combo(key: str) -> tuple[str, ...]:
    """Validates a "+"-joined key combination (M17.1-D) - e.g. "ctrl+l",
    "ctrl+shift+esc" - narrowly scoped so it can never carry an
    arbitrary executable command, shell syntax, path, or URL:

    - every token is case-folded before any check, so "CTRL+L" and
      "ctrl+l" validate identically;
    - an empty token anywhere (a leading, trailing, or doubled "+" -
      "ctrl+", "+l", "ctrl++l") is rejected;
    - at most `MAX_COMBO_TOKENS` tokens total ("ctrl+shift+alt+l" is
      rejected as too long);
    - every token but the last must be a member of `MODIFIER_KEYS` - the
      combination must have at least one (a combination naming no
      modifier at all, e.g. "l+ctrl", is rejected: "l" isn't a modifier
      and can't lead);
    - the LAST token must be a member of `ALLOWED_COMBO_TARGETS` (a
      single a-z/0-9 character, or one of `ALLOWED_KEYS`' own
      non-modifier names) - a combination of modifiers with no real
      target, e.g. "ctrl+alt", is rejected.

    Returns the normalized tokens as a tuple, in order, ready for
    `pyautogui.hotkey(*tokens)`. Raises ComputerControlError - the same
    type every other validation failure in this module raises - for
    anything that fails any rule above."""
    if not isinstance(key, str):
        raise ComputerControlError(f"key combination is not valid: {key!r}")

    tokens = [token.lower() for token in key.split("+")]

    if any(token == "" for token in tokens):
        raise ComputerControlError(
            "key combination has an empty token (check for a leading, "
            f"trailing, or doubled '+'): {key!r}"
        )
    if len(tokens) > MAX_COMBO_TOKENS:
        raise ComputerControlError(
            f"key combination has {len(tokens)} tokens, exceeding the "
            f"maximum of {MAX_COMBO_TOKENS}: {key!r}"
        )

    *modifier_tokens, target_token = tokens

    if not modifier_tokens:
        raise ComputerControlError(f"key combination has no modifier: {key!r}")
    for token in modifier_tokens:
        if token not in MODIFIER_KEYS:
            raise ComputerControlError(
                f"key combination token {token!r} is not an allowed "
                f"modifier (expected one of {sorted(MODIFIER_KEYS)}): {key!r}"
            )
    if target_token not in ALLOWED_COMBO_TARGETS:
        raise ComputerControlError(
            f"key combination's final key {target_token!r} is not an "
            f"allowed combination target: {key!r}"
        )

    return tuple(tokens)


def normalize_key_input(key: str) -> tuple[str, ...]:
    """The ONE place `press_key`'s two accepted forms are told apart -
    by whether `key` contains "+" - so app/computer/tools.py and
    WindowsComputerBackend both dispatch identically instead of each
    re-implementing the split. Returns a 1-tuple (a standalone key, via
    `validate_key`) or a 2-or-3-tuple (a combination, via
    `validate_key_combo`) - callers branch on `len(...)` to decide
    between `pyautogui.press` and `pyautogui.hotkey`."""
    if isinstance(key, str) and "+" in key:
        return validate_key_combo(key)
    return (validate_key(key),)


def validate_text(text: str) -> str:
    """The only validation `type_text` ever applies: `text` must be a
    non-empty string no longer than `MAX_TEXT_LENGTH`."""
    if not isinstance(text, str) or not text:
        raise ComputerControlError("text must be a non-empty string.")
    if len(text) > MAX_TEXT_LENGTH:
        raise ComputerControlError(
            f"text exceeds the maximum length of {MAX_TEXT_LENGTH} characters."
        )
    return text


def validate_coordinates(x: int, y: int, screen: ScreenInfo) -> tuple[int, int]:
    """The only validation `move_mouse`/`click_mouse` ever apply to a
    given (x, y): both must be plain integers (never a bool, float, or
    string), and both must fall within `screen`'s bounds. Shared by
    app/computer/tools.py (which fetches the backend's own current
    ScreenInfo first) and WindowsComputerBackend (defense in depth) -
    the same "exactly ONE definition of safe input" discipline this
    module's own docstring describes."""
    if not isinstance(x, int) or isinstance(x, bool) or not isinstance(y, int) or isinstance(y, bool):
        raise ComputerControlError(f"x and y must be integers, got: x={x!r}, y={y!r}")
    if not (0 <= x < screen.width) or not (0 <= y < screen.height):
        raise ComputerControlError(
            f"({x}, {y}) is outside the screen bounds "
            f"(0-{screen.width - 1}, 0-{screen.height - 1})."
        )
    return x, y


def validate_scroll_amount(amount: int) -> int:
    """The only validation `scroll` ever applies: `amount` must be a
    (possibly negative) integer within `MAX_SCROLL_AMOUNT` of zero.
    Positive scrolls up, negative scrolls down - the same convention
    PyAutoGUI's own `scroll()` uses."""
    if not isinstance(amount, int) or isinstance(amount, bool):
        raise ComputerControlError(f"scroll amount must be an integer, got: {amount!r}")
    if abs(amount) > MAX_SCROLL_AMOUNT:
        raise ComputerControlError(
            f"scroll amount must be within +/-{MAX_SCROLL_AMOUNT}, got: {amount}"
        )
    return amount


class ComputerBackend(ABC):
    """A single local Windows desktop backend. Every method here is
    either READ (observes state, never changes anything) or a narrowly-
    scoped SAFE_ACTION (see app/computer/tools.py for the exact
    permission each Tool declares) - there is deliberately no method for
    shell execution, filesystem writes, credential access, registry
    changes, process injection, or anything else this milestone's
    security requirements rule out. See this module's own docstring for
    the specific narrowing each method below applies.
    """

    @abstractmethod
    def get_active_window(self) -> WindowInfo:
        """Return info about the current foreground window. Read-only."""

    @abstractmethod
    def get_screen_info(self) -> ScreenInfo:
        """Return the primary display's size. Read-only."""

    @abstractmethod
    def open_application(self, app: AllowedApplication) -> None:
        """Launch one of the fixed `AllowedApplication` members. Raises
        ComputerControlError if that application isn't available on this
        machine (e.g. a browser that isn't installed)."""

    @abstractmethod
    def open_url(self, url: str) -> None:
        """Open `url` (already validated by `validate_url`) through the
        normal OS/browser mechanism, leaving the browser that received it
        as the active window. Raises ComputerControlError if that can't be
        established, so a following scroll/click/type is never run against
        whichever window happened to have focus instead."""

    def switch_window(self, title: str, process: str) -> WindowInfo:
        """Make the one VISIBLE, local, titled window whose FULL title equals `title` exactly (already validated by `validate_exact_window_title`) and whose process file
        name equals `process` (validated by `validate_process_name`) the active window, and return it. Never a credential / system / shell window (`is_switchable`).
        Raises ComputerControlError if NO window matches, if MORE THAN ONE does (it never chooses), or if Windows will not activate it. Not abstract: a backend that cannot
        switch windows simply refuses, so every other backend keeps working."""
        raise ComputerControlError("this backend cannot switch windows.")

    @abstractmethod
    def type_text(self, text: str) -> None:
        """Type `text` (already validated by `validate_text`) via the
        keyboard, into whatever currently has focus."""

    @abstractmethod
    def press_key(self, key: str) -> None:
        """Press and release a single key, or (M17.1-D) a "+"-joined
        modifier combination such as "ctrl+l" - already validated by
        `normalize_key_input` (`validate_key`/`validate_key_combo`)."""

    @abstractmethod
    def move_mouse(self, x: int, y: int) -> None:
        """Move the mouse cursor to screen coordinates (x, y). The Tools
        layer has already checked these are on-screen (see
        `ScreenInfo`) before calling this."""

    @abstractmethod
    def click_mouse(
        self, x: int | None, y: int | None, button: MouseButton
    ) -> None:
        """Click `button` at (x, y), or at the current cursor position if
        both are `None`. The Tools layer has already checked non-`None`
        coordinates are on-screen before calling this."""

    @abstractmethod
    def double_click(self, x: int, y: int, button: MouseButton) -> None:
        """Move to (x, y) and perform exactly ONE double-click of
        `button` (M17.1-D) - never an arbitrary click count. Unlike
        `click_mouse`, coordinates are always required here (there is no
        "double-click at the current position" variant) - the Tools
        layer has already checked they're on-screen before calling
        this."""

    @abstractmethod
    def scroll(self, amount: int) -> None:
        """Scroll by `amount` (already validated by
        `validate_scroll_amount`) at the current cursor position."""
