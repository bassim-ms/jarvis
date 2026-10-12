"""
WindowsComputerBackend (M17.1-A): the only concrete ComputerBackend
(app/computer/base.py) for now - local-machine, Windows-only, built on
PyAutoGUI (mouse/keyboard/screen) and pywin32 (active-window info).

Every method here re-validates its input with the SAME `validate_*`
functions app/computer/base.py already exports (defense in depth, in case
a future caller ever invokes this backend directly instead of through
app/computer/tools.py, which validates first) before touching PyAutoGUI,
`subprocess`, or `webbrowser` - never a raw, unvalidated value forwarded
to any of those.

Application launching (`open_application`) only ever runs one of a fixed,
hardcoded set of launch commands (`_APP_LAUNCH_COMMANDS` below), keyed by
the closed `AllowedApplication` enum - never a caller-supplied path or
command string, and every `subprocess.Popen` call here passes a list of
args with `shell=False` (the default), never a shell string. URL opening
(`open_url`) only ever calls `webbrowser.open`, the standard library's own
"hand this to the OS's normal URL handler" mechanism - never a subprocess,
never a shell. After it succeeds, `open_url` also makes the browser that
received the URL the active window (M17.1-H) - see
`_activate_url_handler_window` for why Windows won't do that on its own
when the request comes from a background server process.
"""

import ctypes
import subprocess
import webbrowser
from ctypes import wintypes
from urllib.parse import urlsplit

import pyautogui
import win32api
import win32con
import win32gui
import win32process

from app.computer.base import (
    AllowedApplication,
    ComputerBackend,
    ComputerControlError,
    MouseButton,
    ScreenInfo,
    WindowInfo,
    is_switchable,
    normalize_key_input,
    validate_window_query,
    validate_coordinates,
    validate_scroll_amount,
    validate_text,
    validate_url,
)
from app.computer.gate import InputGate, default_input_gate

# PyAutoGUI's own safety net: moving the mouse into a screen corner
# raises pyautogui.FailSafeException, giving a human a way to abort any
# in-progress automation. Left at its library default (True) - never
# disabled here.
pyautogui.FAILSAFE = True
# This backend doesn't change PyAutoGUI's global timing behavior (its
# default pause applies once after each call). `write()` is the exception
# that needed care: left alone it sends every character back-to-back with
# no delay at all, which a busy application (e.g. the Windows 11 Notepad
# editor) can drop characters from. So `type_text` passes this interval.
TYPING_INTERVAL_SECONDS = 0.03

#: The ENTIRE closed mapping from an `AllowedApplication` member to its
#: one, fixed launch command - see this module's own docstring. Every
#: value is a list (never a shell string) of a single, hardcoded
#: executable filename; Popen resolves it against the system PATH
#: (`shell=False` still lets Windows' own CreateProcess search PATH for a
#: bare filename) - the resolution target is fixed by this dict, never by
#: caller input, so there is no path-injection surface even though the
#: exact installed location varies by machine.
_APP_LAUNCH_COMMANDS: dict[AllowedApplication, list[str]] = {
    AllowedApplication.NOTEPAD: ["notepad.exe"],
    AllowedApplication.CALCULATOR: ["calc.exe"],
    AllowedApplication.FILE_EXPLORER: ["explorer.exe"],
    AllowedApplication.PAINT: ["mspaint.exe"],
    AllowedApplication.CHROME: ["chrome.exe"],
    AllowedApplication.EDGE: ["msedge.exe"],
    AllowedApplication.FIREFOX: ["firefox.exe"],
}

# Private loader instances, so the argtypes set here can never alter the
# shared `ctypes.windll.user32` prototypes PyAutoGUI itself calls through.
_user32 = ctypes.WinDLL("user32", use_last_error=True)
_user32.SetForegroundWindow.argtypes = [wintypes.HWND]
_user32.SetForegroundWindow.restype = wintypes.BOOL
_user32.BringWindowToTop.argtypes = [wintypes.HWND]
_user32.BringWindowToTop.restype = wintypes.BOOL
_user32.AttachThreadInput.argtypes = [wintypes.DWORD, wintypes.DWORD, wintypes.BOOL]
_user32.AttachThreadInput.restype = wintypes.BOOL

_shlwapi = ctypes.WinDLL("shlwapi")
_shlwapi.AssocQueryStringW.argtypes = [
    wintypes.DWORD,
    wintypes.DWORD,
    wintypes.LPCWSTR,
    wintypes.LPCWSTR,
    wintypes.LPWSTR,
    ctypes.POINTER(wintypes.DWORD),
]
_shlwapi.AssocQueryStringW.restype = ctypes.c_long

_dwmapi = ctypes.WinDLL("dwmapi")
_dwmapi.DwmGetWindowAttribute.argtypes = [
    wintypes.HWND,
    wintypes.DWORD,
    ctypes.c_void_p,
    wintypes.DWORD,
]
_dwmapi.DwmGetWindowAttribute.restype = ctypes.c_long

_ASSOCF_IS_PROTOCOL = 0x00001000
_ASSOCSTR_EXECUTABLE = 2
_DWMWA_CLOAKED = 14


def _url_handler_executable(scheme: str) -> str | None:
    """Lowercase executable filename Windows' Shell associates with `scheme`
    - the same resolution os.startfile (so webbrowser.open) goes through.
    Not read from the UserChoice registry key, which can be stale: on the
    machine this was diagnosed on it named Edge while every URL opened in
    Comet."""
    size = wintypes.DWORD(1024)
    buffer = ctypes.create_unicode_buffer(size.value)
    result = _shlwapi.AssocQueryStringW(
        _ASSOCF_IS_PROTOCOL, _ASSOCSTR_EXECUTABLE, scheme, "open", buffer, ctypes.byref(size)
    )
    if result != 0 or not buffer.value:
        return None
    return buffer.value.rsplit("\\", 1)[-1].lower()


def _is_cloaked(hwnd: int) -> bool:
    """True for a window DWM hides, e.g. one on another virtual desktop."""
    cloaked = wintypes.DWORD()
    result = _dwmapi.DwmGetWindowAttribute(
        hwnd, _DWMWA_CLOAKED, ctypes.byref(cloaked), ctypes.sizeof(cloaked)
    )
    return result == 0 and cloaked.value != 0


def _find_window_for_executable(executable: str) -> int | None:
    """Topmost visible, titled, non-cloaked top-level window owned by
    `executable` (EnumWindows walks the Z-order topmost-first)."""
    found: list[int] = []

    def visit(hwnd: int, _extra: object) -> bool:
        if (
            not found
            and win32gui.IsWindowVisible(hwnd)
            and win32gui.GetWindowText(hwnd)
            and not _is_cloaked(hwnd)
            and WindowsComputerBackend._resolve_process_name(hwnd).lower() == executable
        ):
            found.append(hwnd)
        return True

    win32gui.EnumWindows(visit, None)
    return found[0] if found else None


def _list_switchable_windows() -> list[tuple[int, str, str]]:
    """(hwnd, title, process) of every visible, titled, non-cloaked top-level window a `switch_window` may target,
    topmost first (EnumWindows walks the Z-order). Credential / shell / system windows are filtered out here."""
    found: list[tuple[int, str, str]] = []

    def visit(hwnd: int, _extra: object) -> bool:
        if win32gui.IsWindowVisible(hwnd) and not _is_cloaked(hwnd):
            title = win32gui.GetWindowText(hwnd)
            if title:
                process = WindowsComputerBackend._resolve_process_name(hwnd)
                if is_switchable(title, process):
                    found.append((hwnd, title, process))
        return True

    win32gui.EnumWindows(visit, None)
    return found


def _foreground_window() -> int:
    return win32gui.GetForegroundWindow()


def _bring_to_foreground(hwnd: int) -> None:
    """Windows refuses SetForegroundWindow from a background process (this
    server) unless the calling thread shares the current foreground
    thread's input state - so attach only for the duration of the call.
    SetForegroundWindow's own return value is unreliable while attached
    (it can report failure after succeeding), so callers must re-check
    `_foreground_window()` rather than trust it."""
    foreground = _foreground_window()
    foreground_thread = win32process.GetWindowThreadProcessId(foreground)[0] if foreground else 0
    current_thread = win32api.GetCurrentThreadId()
    attached = foreground_thread not in (0, current_thread) and bool(
        _user32.AttachThreadInput(current_thread, foreground_thread, True)
    )
    try:
        if win32gui.IsIconic(hwnd):
            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
        _user32.SetForegroundWindow(hwnd)
        _user32.BringWindowToTop(hwnd)
    finally:
        if attached:
            _user32.AttachThreadInput(current_thread, foreground_thread, False)


class WindowsComputerBackend(ComputerBackend):
    """The default, real ComputerBackend - see app/computer/factory.py
    for how it's constructed for the running application.

    COMPUTER INPUT GATE (app/computer/gate.py): this class enforces the gate ITSELF, at the exact line before each call that would send input to the
    desktop - `open_application`, `type_text`, `press_key`, `move_mouse`, `click_mouse`, `double_click`. So it does not matter how an instance was
    obtained: any `WindowsComputerBackend()`, built by the factory or by any other code, is switched off by default and shares ONE process-wide rate
    limit with every other instance (unless a caller deliberately passes its own `gate`, as a test does). A request that fails validation is turned
    away BEFORE the gate and uses no slot; one the gate admits counts even if the OS then fails. Construction itself touches nothing."""

    def __init__(self, gate: InputGate | None = None):
        self.gate = gate if gate is not None else default_input_gate()

    def check_input(self, action: str) -> None:
        """What the tools call from `validate_arguments`: the gate's non-consuming pre-check (the enforcement is in the methods below)."""
        self.gate.check(action)

    def get_active_window(self) -> WindowInfo:
        hwnd = win32gui.GetForegroundWindow()
        if not hwnd:
            return WindowInfo(title="", process_name="unknown", is_maximized=False)

        title = win32gui.GetWindowText(hwnd)
        process_name = self._resolve_process_name(hwnd)
        is_maximized = self._is_maximized(hwnd)
        return WindowInfo(
            title=title, process_name=process_name, is_maximized=is_maximized
        )

    @staticmethod
    def _resolve_process_name(hwnd: int) -> str:
        try:
            _, pid = win32process.GetWindowThreadProcessId(hwnd)
            handle = win32api.OpenProcess(
                win32con.PROCESS_QUERY_INFORMATION | win32con.PROCESS_VM_READ,
                False,
                pid,
            )
            try:
                path = win32process.GetModuleFileNameEx(handle, 0)
            finally:
                win32api.CloseHandle(handle)
        except Exception:  # noqa: BLE001 - a protected/inaccessible process
            return "unknown"
        return path.rsplit("\\", 1)[-1] if path else "unknown"

    @staticmethod
    def _is_maximized(hwnd: int) -> bool:
        try:
            placement = win32gui.GetWindowPlacement(hwnd)
        except Exception:  # noqa: BLE001
            return False
        show_cmd = placement[1]
        return show_cmd == win32con.SW_SHOWMAXIMIZED

    def get_screen_info(self) -> ScreenInfo:
        width, height = pyautogui.size()
        return ScreenInfo(width=width, height=height)

    def open_application(self, app: AllowedApplication) -> None:
        if not isinstance(app, AllowedApplication):
            raise ComputerControlError(f"Unknown application: {app!r}")

        command = _APP_LAUNCH_COMMANDS[app]
        self.gate.admit("open_application")
        try:
            subprocess.Popen(command, shell=False)
        except (OSError, FileNotFoundError) as exc:
            raise ComputerControlError(
                f"Could not launch '{app.value}' - it may not be installed: {exc}"
            ) from exc

    def open_url(self, url: str) -> None:
        validate_url(url)
        opened = webbrowser.open(url)
        if not opened:
            raise ComputerControlError(f"Could not open URL: {url!r}")
        self._activate_url_handler_window(url)

    @staticmethod
    def _activate_url_handler_window(url: str) -> None:
        """Make the browser that received `url` the active window.
        Windows won't let the browser do it itself when the request came
        from a background process (confirmed live: with the default
        foreground-lock rules it opens BEHIND the window the user was in,
        which keeps both focus and the cursor). Fails closed: if this
        can't be verified, later steps (scroll/click/type) would otherwise
        act on whatever still has focus - e.g. the terminal the request
        was sent from."""
        scheme = urlsplit(url).scheme
        executable = _url_handler_executable(scheme)
        if executable is None:
            raise ComputerControlError(
                f"Opened {url!r}, but could not determine which application "
                f"handles {scheme} URLs, so it was not made the active window."
            )

        hwnd = _find_window_for_executable(executable)
        if hwnd is None:
            raise ComputerControlError(
                f"Opened {url!r}, but found no {executable} window to make "
                "the active window."
            )

        if _foreground_window() != hwnd:
            _bring_to_foreground(hwnd)
        if _foreground_window() != hwnd:
            raise ComputerControlError(
                f"Opened {url!r}, but Windows did not let {executable} become "
                "the active window."
            )

    def switch_window(self, title: str) -> WindowInfo:
        """Composes the helpers `open_url` already uses (`_bring_to_foreground`, `_foreground_window`): the topmost
        matching local window is activated, and success is only claimed once Windows really made it foreground."""
        query = validate_window_query(title).lower()
        matches = [w for w in _list_switchable_windows() if query in w[1].lower()]
        if not matches:
            raise ComputerControlError(f"No visible window title contains {title!r}.")
        hwnd, window_title, process = matches[0]
        if _foreground_window() != hwnd:
            _bring_to_foreground(hwnd)
        if _foreground_window() != hwnd:
            raise ComputerControlError(f"Windows did not let '{window_title}' become the active window.")
        return WindowInfo(title=window_title, process_name=process, is_maximized=self._is_maximized(hwnd))

    def type_text(self, text: str) -> None:
        validate_text(text)
        self.gate.admit("type_text", f"chars={len(text)}")
        pyautogui.write(text, interval=TYPING_INTERVAL_SECONDS)

    def press_key(self, key: str) -> None:
        tokens = normalize_key_input(key)
        self.gate.admit("press_key", f"parts={len(tokens)}")
        if len(tokens) == 1:
            pyautogui.press(tokens[0])
        else:
            pyautogui.hotkey(*tokens)

    def move_mouse(self, x: int, y: int) -> None:
        validate_coordinates(x, y, self.get_screen_info())
        self.gate.admit("move_mouse")
        pyautogui.moveTo(x, y)

    def click_mouse(
        self, x: int | None, y: int | None, button: MouseButton
    ) -> None:
        if not isinstance(button, MouseButton):
            raise ComputerControlError(f"Unknown mouse button: {button!r}")
        if x is None and y is None:
            self.gate.admit("click_mouse")
            pyautogui.click(button=button.value)
        else:
            validate_coordinates(x, y, self.get_screen_info())
            self.gate.admit("click_mouse")
            pyautogui.click(x=x, y=y, button=button.value)

    def double_click(self, x: int, y: int, button: MouseButton) -> None:
        if not isinstance(button, MouseButton):
            raise ComputerControlError(f"Unknown mouse button: {button!r}")
        validate_coordinates(x, y, self.get_screen_info())
        self.gate.admit("double_click")
        # PyAutoGUI's own dedicated double-click primitive - never
        # `click(clicks=2)`, which would expose an arbitrary click count
        # this milestone deliberately doesn't (see app/computer/tools.py:
        # DoubleClickTool).
        pyautogui.doubleClick(x=x, y=y, button=button.value)

    def scroll(self, amount: int) -> None:
        validate_scroll_amount(amount)
        pyautogui.scroll(amount)
