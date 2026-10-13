"""
Suite-wide protection against REAL OS input: a fake input library, tripwires on the raw Windows input functions, and an import blocker.

WHY. Computer-control code ends in `pyautogui` (keyboard, mouse, scroll), `subprocess.Popen` (launching programs) and `webbrowser.open`. A test that reaches the real
thing moves a real mouse or presses a real key. Individual tests used to protect themselves by patching `windows_backend.pyautogui`, which only works if the patch is
in place at the instant of the call. That is not enough for mutation testing, where a deliberately broken method can be built in a namespace the patch never reaches.
This module closes the door at the SOURCE, once, before anything can import the real library:

  * `sys.modules["pyautogui"]` is a FAKE that records every call and does nothing. Every `import pyautogui` anywhere in the process - the application, a test, a
    namespace copied for a mutant - gets this object. It has no `__file__` and no `__path__`, so `import pyautogui._pyautogui_win` cannot reach the real internals.
  * A meta-path finder refuses to import the real `pyautogui` (or `pydirectinput` / `pynput`) under any other route.
  * The raw Win32 input functions (`user32.keybd_event`, `mouse_event`, `SendInput`, `SetCursorPos`, ... and their `win32api` twins) are replaced by tripwires, both on
    the shared `ctypes.windll.user32` and on any freshly opened `ctypes.WinDLL`/`CDLL` handle.
  * The calls that change which window is in front (`user32.SetForegroundWindow` / `BringWindowToTop` / `AttachThreadInput`, `win32gui.SetForegroundWindow` /
    `ShowWindow` / ...) are tripwires too: a test that reaches them would steal the real focus.
  * `webbrowser.open*` and `os.startfile` are tripwires.
  * `subprocess.Popen` refuses to start the programs `open_application` can launch (Notepad, Calculator, Explorer, Paint, Chrome, Edge, Firefox) and nothing else.

A tripwire raises `RealInputBlocked`, a BaseException on purpose: an `except Exception` somewhere in application code cannot swallow it, so reaching real input is
always loud. Nothing here ever calls a native input function: the checks only LOOK at what is installed.

`isolation_report()` says what is in place (used by tests/conftest.py, the tests, and the mutation runner, which refuses to apply any mutant unless it is all true).
`tests/conftest.py` calls `install()` when pytest starts. Running pytest with `--noconftest` bypasses it; tests/test_no_real_input_library.py then fails loudly.
"""

from __future__ import annotations

import ctypes
import importlib.abc
import importlib.machinery
import os
import shlex
import subprocess
import sys
import types
import webbrowser
from pathlib import PureWindowsPath

TRIPWIRE_MARK = "__jarvis_tripwire__"
GETITEM_GUARD_MARK = "__jarvis_getitem_guard__"

#: Real input libraries that must never be imported while the fake is in place.
BLOCKED_ROOTS = frozenset({"pyautogui", "pydirectinput", "pynput"})
#: Raw Win32 functions that synthesize input, or change which window is in front; reaching any of them from a test is a bug.
BLOCKED_USER32 = frozenset({
    "keybd_event", "mouse_event", "SendInput", "SetCursorPos", "SetPhysicalCursorPos", "BlockInput",
    "SetForegroundWindow", "BringWindowToTop", "AttachThreadInput",  # focus: what `switch_window` and `open_url` end in
})
BLOCKED_WIN32API = frozenset({"keybd_event", "mouse_event", "SetCursorPos"})
#: The pywin32 window calls that change which window is shown or in front (the backend's `_bring_to_foreground` uses ShowWindow).
BLOCKED_WIN32GUI = frozenset({"SetForegroundWindow", "ShowWindow", "BringWindowToTop", "SwitchToThisWindow", "SetActiveWindow"})
#: The programs `WindowsComputerBackend.open_application` can launch (plus their bare names).
LAUNCH_TARGETS = frozenset({
    "notepad", "notepad.exe", "calc", "calc.exe", "calculator", "explorer", "explorer.exe",
    "mspaint", "mspaint.exe", "chrome", "chrome.exe", "msedge", "msedge.exe", "firefox", "firefox.exe",
})
#: What the fake records: every function of the real library that could act on the machine.
RECORDED_FUNCTIONS = (
    "write", "typewrite", "press", "hotkey", "keyDown", "keyUp", "moveTo", "moveRel", "move", "click", "doubleClick", "tripleClick", "rightClick",
    "middleClick", "mouseDown", "mouseUp", "dragTo", "dragRel", "drag", "scroll", "hscroll", "vscroll",
)

#: Every tripwire that fired, in order (strings naming what was reached). The mutation runner reports how many.
TRIPWIRE_HITS: list[str] = []


class RealInputBlocked(BaseException):
    """A test or a mutant reached REAL OS input. A BaseException so that no `except Exception` can hide it."""


def tripwire(what: str):
    def blocked(*args, **kwargs):
        TRIPWIRE_HITS.append(what)
        raise RealInputBlocked(f"REAL OS input was reached ({what}); it is blocked in tests (see tests/fake_input_library.py)")

    blocked.__jarvis_tripwire__ = True
    blocked.__name__ = f"tripwire[{what}]"
    return blocked


def is_tripwire(obj: object) -> bool:
    """Whether `obj` is one of our tripwires. Only reads an attribute; never calls anything."""
    return getattr(obj, TRIPWIRE_MARK, False) is True


class FakeInputLibrary(types.ModuleType):
    """Stands in for `pyautogui` everywhere. Records `(function name, args, kwargs)` in `calls`; sends nothing; has no file on disk."""

    IS_JARVIS_FAKE_INPUT_LIBRARY = True

    def __init__(self) -> None:
        super().__init__("pyautogui", "A fake input library: records calls, does nothing.")
        self.__spec__ = importlib.machinery.ModuleSpec("pyautogui", loader=None)  # so importlib.util.find_spec("pyautogui") does not choke on a None spec
        self.calls: list[tuple] = []
        self.FAILSAFE = True
        self.PAUSE = 0.1
        self.MINIMUM_DURATION = 0.1
        self.FAILSAFE_POINTS = [(0, 0)]

        class FailSafeException(Exception):
            pass

        self.FailSafeException = FailSafeException
        for name in RECORDED_FUNCTIONS:
            setattr(self, name, self._recorder(name))
        self.size = lambda: (1920, 1080)
        self.position = lambda: (500, 500)
        self.onScreen = lambda x, y=None: True

        def screenshot(*args, **kwargs):
            raise RuntimeError("the fake input library has no screen: a test that needs a screenshot must supply its own")

        self.screenshot = screenshot

    def _recorder(self, name: str):
        def record(*args, **kwargs):
            self.calls.append((name, args, kwargs))

        record.__name__ = name
        return record

    def __getattr__(self, name: str):
        if name.startswith("__"):
            raise AttributeError(name)
        raise AttributeError(f"the fake input library has no attribute {name!r}; add it deliberately to tests/fake_input_library.py if a test truly needs it")


FAKE = FakeInputLibrary()


class _BlockRealInputLibraries(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.partition(".")[0] in BLOCKED_ROOTS:
            raise ImportError(f"the real input library {fullname!r} cannot be imported in tests (tests/fake_input_library.py)", name=fullname)
        return None


def _launch_name(args: object) -> str:
    """The lower-case file name of the program a Popen call would start, or '' if it cannot tell."""
    try:
        if isinstance(args, (list, tuple)):
            first = os.fsdecode(args[0]) if args else ""
        else:
            text = os.fsdecode(args) if isinstance(args, (str, bytes, os.PathLike)) else ""
            first = shlex.split(text, posix=False)[0] if text.strip() else ""
        return PureWindowsPath(first.strip().strip('"').strip("'")).name.lower()
    except Exception:  # noqa: BLE001 - cannot tell: let it through; only a recognised launch target is blocked
        return ""


def _patch_user32() -> None:
    if not hasattr(ctypes, "windll"):
        return
    if not getattr(ctypes.CDLL.__getitem__, GETITEM_GUARD_MARK, False):
        original = ctypes.CDLL.__getitem__

        def guarded(self, name_or_ordinal):
            if isinstance(name_or_ordinal, str) and name_or_ordinal in BLOCKED_USER32:
                return tripwire(f"user32.{name_or_ordinal}")
            return original(self, name_or_ordinal)

        setattr(guarded, GETITEM_GUARD_MARK, True)
        ctypes.CDLL.__getitem__ = guarded  # covers every freshly opened handle, and WinDLL (a subclass)
    user32 = ctypes.windll.user32
    for name in BLOCKED_USER32:
        setattr(user32, name, tripwire(f"user32.{name}"))  # and the shared handle, whatever was cached on it before


def _patch_win32api() -> None:
    try:
        import win32api
    except ImportError:  # pragma: no cover
        return
    for name in BLOCKED_WIN32API:
        if hasattr(win32api, name):
            setattr(win32api, name, tripwire(f"win32api.{name}"))


def _patch_win32gui() -> None:
    try:
        import win32gui
    except ImportError:  # pragma: no cover
        return
    for name in BLOCKED_WIN32GUI:
        if hasattr(win32gui, name):
            setattr(win32gui, name, tripwire(f"win32gui.{name}"))


def _patch_browser_and_shell() -> None:
    for name in ("open", "open_new", "open_new_tab"):
        setattr(webbrowser, name, tripwire(f"webbrowser.{name}"))
    if hasattr(os, "startfile"):
        os.startfile = tripwire("os.startfile")


def _patch_popen() -> None:
    if getattr(subprocess.Popen, TRIPWIRE_MARK, False):
        return
    real = subprocess.Popen

    class GuardedPopen(real):
        __jarvis_tripwire__ = True

        def __init__(self, args, *a, **k):
            name = _launch_name(args)
            if name in LAUNCH_TARGETS:
                TRIPWIRE_HITS.append(f"subprocess.Popen({name})")
                raise RealInputBlocked(f"a program launch ({name}) was attempted; blocked in tests (tests/fake_input_library.py)")
            super().__init__(args, *a, **k)

    GuardedPopen.__name__ = GuardedPopen.__qualname__ = "Popen"
    subprocess.Popen = GuardedPopen


def install() -> None:
    """Put the fake and every tripwire in place. Idempotent; call it before anything imports the application."""
    sys.modules["pyautogui"] = FAKE
    if not any(isinstance(finder, _BlockRealInputLibraries) for finder in sys.meta_path):
        sys.meta_path.insert(0, _BlockRealInputLibraries())
    _patch_user32()
    _patch_win32api()
    _patch_win32gui()
    _patch_browser_and_shell()
    _patch_popen()


def isolation_report() -> dict:
    """What is in place, as plain booleans. Looks only; calls nothing native. `all_ok` is what a mutation runner must see before it applies any mutant."""
    modules = sys.modules
    current = modules.get("pyautogui")
    user32_vars = vars(ctypes.windll.user32) if hasattr(ctypes, "windll") else {}
    report = {
        "fake_installed": current is FAKE and "__file__" not in vars(FAKE) and not hasattr(FAKE, "__path__"),
        "real_library_not_loaded": not any(name.startswith("pyautogui.") for name in modules) and getattr(current, "__file__", None) is None,
        "import_blocker_installed": any(isinstance(finder, _BlockRealInputLibraries) for finder in sys.meta_path),
        "user32_shared_handle_tripwired": all(is_tripwire(user32_vars.get(name)) for name in BLOCKED_USER32) if user32_vars else True,
        "user32_new_handles_guarded": getattr(ctypes.CDLL.__getitem__, GETITEM_GUARD_MARK, False) is True,
        "win32api_tripwired": _win32api_tripwired(),
        "win32gui_focus_tripwired": _win32gui_tripwired(),
        "webbrowser_tripwired": all(is_tripwire(getattr(webbrowser, name, None)) for name in ("open", "open_new", "open_new_tab")),
        "startfile_tripwired": is_tripwire(getattr(os, "startfile", None)) if hasattr(os, "startfile") else True,
        "program_launch_guarded": is_tripwire(subprocess.Popen),
    }
    backend = modules.get("app.computer.windows_backend")
    report["backend_pyautogui_is_fake"] = None if backend is None else (vars(backend).get("pyautogui") is FAKE)
    report["all_ok"] = all(value for key, value in report.items() if key != "backend_pyautogui_is_fake") and report["backend_pyautogui_is_fake"] in (None, True)
    return report


def _win32gui_tripwired() -> bool:
    try:
        import win32gui
    except ImportError:  # pragma: no cover
        return True
    return all(is_tripwire(getattr(win32gui, name, None)) for name in BLOCKED_WIN32GUI if hasattr(win32gui, name))


def _win32api_tripwired() -> bool:
    try:
        import win32api
    except ImportError:  # pragma: no cover
        return True
    return all(is_tripwire(getattr(win32api, name, None)) for name in BLOCKED_WIN32API if hasattr(win32api, name))
