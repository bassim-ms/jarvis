"""
Tests for WindowsComputerBackend (app/computer/windows_backend.py).

Every PyAutoGUI/pywin32/subprocess/webbrowser call is mocked - this file
must NEVER move the real mouse, type into the real keyboard, launch a
real application, or touch the real screen. These tests instead prove:
- the backend re-validates input itself (defense in depth) and never
  forwards an invalid value to PyAutoGUI/subprocess/webbrowser,
- `open_application` only ever calls `subprocess.Popen` with a fixed,
  hardcoded argument list and `shell=False` - never a shell string, and
  never anything derived from caller input,
- `open_url` only ever calls `webbrowser.open` - never `subprocess`,
- the active-window/screen-info reads translate pywin32/PyAutoGUI data
  into the plain WindowInfo/ScreenInfo dataclasses correctly,
- `press_key` (M17.1-D) forwards the NORMALIZED lowercase key/combo
  tokens to PyAutoGUI, never the original casing - PyAutoGUI's own key
  names are lowercase-only and silently no-op on anything else - and
  dispatches a standalone key to `pyautogui.press` vs a "+"-joined
  combination to `pyautogui.hotkey`,
- `open_url` (M17.1-H) makes the browser that received the URL the
  active window - after, never before, `webbrowser.open` - and fails
  closed when that can't be verified. The four OS-touching helpers it
  uses are replaced for EVERY test here by the autouse `fake_desktop`
  fixture, so no test can ever activate a real window.
"""

import ctypes
import re
from unittest.mock import MagicMock, patch

import pytest

from app.computer.base import (
    AllowedApplication,
    ComputerControlError,
    MouseButton,
    ScreenInfo,
)
from app.computer.defaults import create_computer_tools_registry
from app.computer.windows_backend import WindowsComputerBackend
from app.core import JarvisCore
from app.memory.sqlite_store import SQLiteMemoryStore
from app.providers.base import AIProvider
from app.tools.decision import ToolCall, ToolCallBuilder, ToolCallProvider
from app.tools.registry import ToolRegistry

BROWSER_HWND = 1001
PREVIOUSLY_FOCUSED_HWND = 2002


class FakeDesktop:
    """Stands in for `_url_handler_executable`, `_find_window_for_executable`,
    `_foreground_window` and `_bring_to_foreground`. Starts in the state
    observed live: the browser window exists but something else (e.g. the
    editor the request was sent from) is in the foreground. `events` is a
    single ordered log tests can extend, to prove call ORDER across
    webbrowser/activation/PyAutoGUI."""

    def __init__(self):
        self.executable: str | None = "comet.exe"
        self.browser_hwnd: int | None = BROWSER_HWND
        self.foreground = PREVIOUSLY_FOCUSED_HWND
        self.activation_allowed = True
        self.resolved_schemes: list[str] = []
        self.events: list[tuple] = []

    def url_handler_executable(self, scheme):
        self.resolved_schemes.append(scheme)
        return self.executable

    def find_window(self, executable):
        return self.browser_hwnd

    def foreground_window(self):
        return self.foreground

    def bring_to_foreground(self, hwnd):
        self.events.append(("activate", hwnd))
        if self.activation_allowed:
            self.foreground = hwnd

    @property
    def activations(self):
        return [hwnd for name, hwnd in self.events if name == "activate"]


@pytest.fixture(autouse=True)
def fake_desktop():
    desktop = FakeDesktop()
    with patch(
        "app.computer.windows_backend._url_handler_executable",
        side_effect=desktop.url_handler_executable,
    ), patch(
        "app.computer.windows_backend._find_window_for_executable",
        side_effect=desktop.find_window,
    ), patch(
        "app.computer.windows_backend._foreground_window",
        side_effect=desktop.foreground_window,
    ), patch(
        "app.computer.windows_backend._bring_to_foreground",
        side_effect=desktop.bring_to_foreground,
    ):
        yield desktop


@pytest.fixture
def backend():
    return WindowsComputerBackend()


# --- open_application: no shell execution, no arbitrary paths ------------


def test_open_application_calls_popen_with_a_fixed_arg_list_and_no_shell(backend):
    with patch("app.computer.windows_backend.subprocess.Popen") as popen:
        backend.open_application(AllowedApplication.NOTEPAD)

    popen.assert_called_once_with(["notepad.exe"], shell=False)


def test_open_application_never_uses_shell_true_for_any_allowed_app(backend):
    with patch("app.computer.windows_backend.subprocess.Popen") as popen:
        for app in AllowedApplication:
            backend.open_application(app)

    for call in popen.call_args_list:
        args, kwargs = call
        assert kwargs.get("shell") is not True
        assert isinstance(args[0], list)  # a list of args, never a shell string


def test_open_application_rejects_a_non_enum_value_before_touching_popen(backend):
    with patch("app.computer.windows_backend.subprocess.Popen") as popen:
        with pytest.raises(ComputerControlError):
            backend.open_application("C:/Windows/System32/cmd.exe")  # type: ignore[arg-type]

    popen.assert_not_called()


def test_open_application_wraps_a_missing_executable_as_computer_control_error(backend):
    with patch("app.computer.windows_backend.subprocess.Popen", side_effect=FileNotFoundError()):
        with pytest.raises(ComputerControlError):
            backend.open_application(AllowedApplication.FIREFOX)


# --- open_url: only ever the browser mechanism, never a subprocess -------


def test_open_url_calls_webbrowser_open_only(backend):
    with patch("app.computer.windows_backend.webbrowser.open", return_value=True) as wb_open, patch(
        "app.computer.windows_backend.subprocess.Popen"
    ) as popen:
        backend.open_url("https://example.com")

    wb_open.assert_called_once_with("https://example.com")
    popen.assert_not_called()


def test_open_url_rejects_a_non_http_scheme_before_touching_webbrowser(backend):
    with patch("app.computer.windows_backend.webbrowser.open") as wb_open:
        with pytest.raises(ComputerControlError):
            backend.open_url("file:///C:/secrets.txt")

    wb_open.assert_not_called()


def test_open_url_raises_when_webbrowser_reports_failure(backend, fake_desktop):
    with patch("app.computer.windows_backend.webbrowser.open", return_value=False):
        with pytest.raises(ComputerControlError):
            backend.open_url("https://example.com")

    assert fake_desktop.activations == []  # nothing opened, so nothing to activate


# --- open_url: the browser becomes the active window (M17.1-H) ------------
#
# Live-diagnosed root cause of "open_url, then scroll 50 opens the page but
# never scrolls it": webbrowser.open returns immediately, and Windows won't
# let the browser take focus when the request came from a background
# process - it opened BEHIND the user's editor, which kept both focus and
# the cursor, so the injected wheel event went to the editor.


def test_open_url_makes_the_browser_the_active_window_after_opening(backend, fake_desktop):
    with patch(
        "app.computer.windows_backend.webbrowser.open",
        side_effect=lambda url: fake_desktop.events.append(("webbrowser.open", url)) or True,
    ):
        backend.open_url("https://luniti.app")

    assert fake_desktop.events == [
        ("webbrowser.open", "https://luniti.app"),
        ("activate", BROWSER_HWND),
    ]
    assert fake_desktop.foreground == BROWSER_HWND


def test_open_url_resolves_the_handler_from_the_urls_own_scheme(backend, fake_desktop):
    with patch("app.computer.windows_backend.webbrowser.open", return_value=True):
        backend.open_url("http://example.com")
        backend.open_url("https://example.com")

    assert fake_desktop.resolved_schemes == ["http", "https"]


def test_open_url_does_not_reactivate_a_browser_that_is_already_active(backend, fake_desktop):
    fake_desktop.foreground = BROWSER_HWND

    with patch("app.computer.windows_backend.webbrowser.open", return_value=True):
        backend.open_url("https://luniti.app")

    assert fake_desktop.activations == []


def test_open_url_fails_closed_when_windows_refuses_activation(backend, fake_desktop):
    """Must raise - not report success - so a following scroll/click/type
    is never run against the window that kept focus instead."""
    fake_desktop.activation_allowed = False

    with patch("app.computer.windows_backend.webbrowser.open", return_value=True) as wb_open:
        with pytest.raises(
            ComputerControlError,
            match=r"Opened 'https://luniti\.app', but Windows did not let comet\.exe become",
        ):
            backend.open_url("https://luniti.app")

    wb_open.assert_called_once_with("https://luniti.app")
    assert fake_desktop.activations == [BROWSER_HWND]


def test_open_url_fails_closed_when_no_browser_window_is_found(backend, fake_desktop):
    fake_desktop.browser_hwnd = None

    with patch("app.computer.windows_backend.webbrowser.open", return_value=True):
        with pytest.raises(ComputerControlError, match="found no comet.exe window"):
            backend.open_url("https://luniti.app")

    assert fake_desktop.activations == []


def test_open_url_fails_closed_when_the_url_handler_cannot_be_resolved(backend, fake_desktop):
    fake_desktop.executable = None

    with patch("app.computer.windows_backend.webbrowser.open", return_value=True):
        with pytest.raises(ComputerControlError, match="could not determine which application"):
            backend.open_url("https://luniti.app")

    assert fake_desktop.activations == []


# --- keyboard/mouse/scroll: validated, then forwarded to PyAutoGUI -------


def test_type_text_forwards_validated_text(backend):
    with patch("app.computer.windows_backend.pyautogui.write") as write:
        backend.type_text("hello")

    write.assert_called_once_with("hello")


def test_type_text_rejects_invalid_text_before_touching_pyautogui(backend):
    with patch("app.computer.windows_backend.pyautogui.write") as write:
        with pytest.raises(ComputerControlError):
            backend.type_text("")

    write.assert_not_called()


def test_press_key_forwards_an_allowed_key(backend):
    with patch("app.computer.windows_backend.pyautogui.press") as press:
        backend.press_key("enter")

    press.assert_called_once_with("enter")


def test_press_key_rejects_a_disallowed_key_before_touching_pyautogui(backend):
    with patch("app.computer.windows_backend.pyautogui.press") as press:
        with pytest.raises(ComputerControlError):
            backend.press_key("banana")

    press.assert_not_called()


# --- press_key: case-insensitivity reaches PyAutoGUI already normalized
# (Fix A) - the exact gap that let "ENTER" silently no-op before: PyAutoGUI's
# own key names are lowercase-only, so the value actually forwarded matters,
# not just whether validation raises.


@pytest.mark.parametrize(
    "raw,expected",
    [("ENTER", "enter"), ("Enter", "enter"), ("TAB", "tab"), ("F5", "f5")],
)
def test_press_key_normalizes_case_before_forwarding_to_pyautogui(backend, raw, expected):
    with patch("app.computer.windows_backend.pyautogui.press") as press:
        backend.press_key(raw)

    press.assert_called_once_with(expected)


def test_press_key_case_insensitivity_never_reaches_pyautogui_with_the_original_casing(backend):
    """Regression guard for the exact bug this fix addresses: forwarding
    the RAW (unnormalized) key string would make PyAutoGUI silently
    no-op (its own key map is lowercase-only) instead of pressing
    anything - proving the normalized value, not the original, is what
    PyAutoGUI actually receives."""
    with patch("app.computer.windows_backend.pyautogui.press") as press:
        backend.press_key("ENTER")

    assert press.call_args != (("ENTER",), {})
    press.assert_called_once_with("enter")


# --- press_key: modifier combinations (Fix B) -----------------------------


@pytest.mark.parametrize(
    "raw,expected_tokens",
    [
        ("ctrl+l", ("ctrl", "l")),
        ("CTRL+L", ("ctrl", "l")),
        ("ctrl+c", ("ctrl", "c")),
        ("ctrl+shift+esc", ("ctrl", "shift", "esc")),
        ("alt+tab", ("alt", "tab")),
        ("win+d", ("win", "d")),
        ("ctrl+enter", ("ctrl", "enter")),
    ],
)
def test_press_key_dispatches_a_combination_to_hotkey(backend, raw, expected_tokens):
    with patch("app.computer.windows_backend.pyautogui.press") as press, patch(
        "app.computer.windows_backend.pyautogui.hotkey"
    ) as hotkey:
        backend.press_key(raw)

    hotkey.assert_called_once_with(*expected_tokens)
    press.assert_not_called()


def test_press_key_dispatches_a_standalone_key_to_press_not_hotkey(backend):
    with patch("app.computer.windows_backend.pyautogui.press") as press, patch(
        "app.computer.windows_backend.pyautogui.hotkey"
    ) as hotkey:
        backend.press_key("enter")

    press.assert_called_once_with("enter")
    hotkey.assert_not_called()


@pytest.mark.parametrize(
    "combo",
    [
        "l+ctrl",
        "ctrl+",
        "+l",
        "ctrl++l",
        "ctrl+shift+alt+l",
        "banana+l",
        "ctrl+alt",
        "ctrl+/bin/sh",
        "ctrl+; shutdown /s",
    ],
)
def test_press_key_rejects_disallowed_combinations_before_touching_pyautogui(backend, combo):
    with patch("app.computer.windows_backend.pyautogui.press") as press, patch(
        "app.computer.windows_backend.pyautogui.hotkey"
    ) as hotkey:
        with pytest.raises(ComputerControlError):
            backend.press_key(combo)

    press.assert_not_called()
    hotkey.assert_not_called()


def test_move_mouse_forwards_to_pyautogui(backend):
    with patch("app.computer.windows_backend.pyautogui.size", return_value=(1920, 1080)), patch(
        "app.computer.windows_backend.pyautogui.moveTo"
    ) as move_to:
        backend.move_mouse(10, 20)

    move_to.assert_called_once_with(10, 20)


def test_move_mouse_rejects_off_screen_coordinates_before_touching_pyautogui(backend):
    """WindowsComputerBackend re-validates coordinates itself (defense in
    depth) - a direct caller that bypasses the Tools layer (see
    app/computer/tools.py) still can't push the real cursor off-screen or
    forward a non-integer value straight to PyAutoGUI."""
    with patch("app.computer.windows_backend.pyautogui.size", return_value=(1920, 1080)), patch(
        "app.computer.windows_backend.pyautogui.moveTo"
    ) as move_to:
        with pytest.raises(ComputerControlError):
            backend.move_mouse(-999999, -999999)

    move_to.assert_not_called()


def test_move_mouse_rejects_non_integer_coordinates_before_touching_pyautogui(backend):
    with patch("app.computer.windows_backend.pyautogui.size", return_value=(1920, 1080)), patch(
        "app.computer.windows_backend.pyautogui.moveTo"
    ) as move_to:
        with pytest.raises(ComputerControlError):
            backend.move_mouse(1.5, 20)  # type: ignore[arg-type]

    move_to.assert_not_called()


def test_click_mouse_with_coordinates(backend):
    with patch("app.computer.windows_backend.pyautogui.size", return_value=(1920, 1080)), patch(
        "app.computer.windows_backend.pyautogui.click"
    ) as click:
        backend.click_mouse(10, 20, MouseButton.RIGHT)

    click.assert_called_once_with(x=10, y=20, button="right")


def test_click_mouse_without_coordinates_clicks_current_position(backend):
    with patch("app.computer.windows_backend.pyautogui.click") as click:
        backend.click_mouse(None, None, MouseButton.LEFT)

    click.assert_called_once_with(button="left")


def test_click_mouse_rejects_off_screen_coordinates_before_touching_pyautogui(backend):
    with patch("app.computer.windows_backend.pyautogui.size", return_value=(1920, 1080)), patch(
        "app.computer.windows_backend.pyautogui.click"
    ) as click:
        with pytest.raises(ComputerControlError):
            backend.click_mouse(99999, 99999, MouseButton.LEFT)

    click.assert_not_called()


def test_click_mouse_rejects_a_non_enum_button(backend):
    with patch("app.computer.windows_backend.pyautogui.click") as click:
        with pytest.raises(ComputerControlError):
            backend.click_mouse(None, None, "left")  # type: ignore[arg-type]

    click.assert_not_called()


# --- double_click: dedicated PyAutoGUI primitive, never click(clicks=N) ---


def test_double_click_calls_pyautogui_doublclick_with_validated_coordinates(backend):
    with patch("app.computer.windows_backend.pyautogui.size", return_value=(1920, 1080)), patch(
        "app.computer.windows_backend.pyautogui.doubleClick"
    ) as double_click, patch("app.computer.windows_backend.pyautogui.click") as click:
        backend.double_click(10, 20, MouseButton.RIGHT)

    double_click.assert_called_once_with(x=10, y=20, button="right")
    click.assert_not_called()


def test_double_click_rejects_off_screen_coordinates_before_touching_pyautogui(backend):
    with patch("app.computer.windows_backend.pyautogui.size", return_value=(1920, 1080)), patch(
        "app.computer.windows_backend.pyautogui.doubleClick"
    ) as double_click:
        with pytest.raises(ComputerControlError):
            backend.double_click(99999, 99999, MouseButton.LEFT)

    double_click.assert_not_called()


def test_double_click_rejects_non_integer_coordinates_before_touching_pyautogui(backend):
    with patch("app.computer.windows_backend.pyautogui.size", return_value=(1920, 1080)), patch(
        "app.computer.windows_backend.pyautogui.doubleClick"
    ) as double_click:
        with pytest.raises(ComputerControlError):
            backend.double_click(float("nan"), 20, MouseButton.LEFT)  # type: ignore[arg-type]

    double_click.assert_not_called()


def test_double_click_rejects_a_non_enum_button(backend):
    with patch("app.computer.windows_backend.pyautogui.size", return_value=(1920, 1080)), patch(
        "app.computer.windows_backend.pyautogui.doubleClick"
    ) as double_click:
        with pytest.raises(ComputerControlError):
            backend.double_click(10, 20, "left")  # type: ignore[arg-type]

    double_click.assert_not_called()


def test_scroll_forwards_a_validated_amount(backend):
    with patch("app.computer.windows_backend.pyautogui.scroll") as scroll:
        backend.scroll(-50)

    scroll.assert_called_once_with(-50)


def test_scroll_rejects_out_of_range_amount_before_touching_pyautogui(backend):
    from app.computer.base import MAX_SCROLL_AMOUNT

    with patch("app.computer.windows_backend.pyautogui.scroll") as scroll:
        with pytest.raises(ComputerControlError):
            backend.scroll(MAX_SCROLL_AMOUNT + 1)

    scroll.assert_not_called()


# --- get_screen_info -------------------------------------------------------


def test_get_screen_info_reads_from_pyautogui(backend):
    with patch("app.computer.windows_backend.pyautogui.size", return_value=(1920, 1080)):
        info = backend.get_screen_info()

    assert info == ScreenInfo(width=1920, height=1080)


# --- get_active_window -----------------------------------------------------


def test_get_active_window_builds_window_info_from_pywin32():
    backend = WindowsComputerBackend()
    fake_handle = object()

    with patch("app.computer.windows_backend.win32gui.GetForegroundWindow", return_value=42), patch(
        "app.computer.windows_backend.win32gui.GetWindowText", return_value="Notepad"
    ), patch(
        "app.computer.windows_backend.win32gui.GetWindowPlacement",
        return_value=(0, 3, (0, 0), (0, 0), (0, 0, 0, 0)),
    ), patch(
        "app.computer.windows_backend.win32process.GetWindowThreadProcessId",
        return_value=(0, 1234),
    ), patch(
        "app.computer.windows_backend.win32api.OpenProcess", return_value=fake_handle
    ), patch(
        "app.computer.windows_backend.win32process.GetModuleFileNameEx",
        return_value=r"C:\Windows\System32\notepad.exe",
    ), patch(
        "app.computer.windows_backend.win32api.CloseHandle"
    ), patch(
        "app.computer.windows_backend.win32con.SW_SHOWMAXIMIZED", 3
    ):
        window = backend.get_active_window()

    assert window.title == "Notepad"
    assert window.process_name == "notepad.exe"
    assert window.is_maximized is True


def test_get_active_window_returns_unknown_process_name_on_resolution_failure():
    backend = WindowsComputerBackend()

    with patch("app.computer.windows_backend.win32gui.GetForegroundWindow", return_value=42), patch(
        "app.computer.windows_backend.win32gui.GetWindowText", return_value="Something"
    ), patch(
        "app.computer.windows_backend.win32gui.GetWindowPlacement",
        return_value=(0, 1, (0, 0), (0, 0), (0, 0, 0, 0)),
    ), patch(
        "app.computer.windows_backend.win32process.GetWindowThreadProcessId",
        side_effect=RuntimeError("access denied"),
    ):
        window = backend.get_active_window()

    assert window.title == "Something"
    assert window.process_name == "unknown"
    assert window.is_maximized is False


def test_get_active_window_with_no_foreground_window_returns_empty_info():
    backend = WindowsComputerBackend()

    with patch("app.computer.windows_backend.win32gui.GetForegroundWindow", return_value=0):
        window = backend.get_active_window()

    assert window.title == ""
    assert window.process_name == "unknown"
    assert window.is_maximized is False


# --- M17.1-H: the live request, traced through the REAL backend ------------


class UnusedProvider(AIProvider):
    def generate_reply(self, message: str) -> str:
        raise AssertionError(f"chat AI provider should not be called: {message!r}")


class ClauseParsingToolCallProvider(ToolCallProvider):
    """Stands in for the AI call scroll's construction needs (its one
    required parameter is a number, so the deterministic fast path can't
    fill it): zips the integers in the step's OWN clause into the tool's
    required parameters, and records every clause it was given."""

    def __init__(self):
        self.clauses: list[str] = []

    def decide(self, message, candidate_tools):
        self.clauses.append(message)
        tool = candidate_tools[0]
        numbers = [int(n) for n in re.findall(r"-?\d+", message)]
        required = [p.name for p in tool.parameters if p.required]
        return ToolCall(tool_name=tool.name, parameters=dict(zip(required, numbers)))


def make_real_backend_jarvis():
    call_provider = ClauseParsingToolCallProvider()
    jarvis = JarvisCore(
        provider=UnusedProvider(),
        memory=SQLiteMemoryStore(":memory:"),
        tools=ToolRegistry(),
        computer_tools=create_computer_tools_registry(backend=WindowsComputerBackend()),
        tool_call_builder=ToolCallBuilder(call_provider),
    )
    return jarvis, call_provider


def test_open_url_then_scroll_reaches_the_windows_backend_in_order_with_amount_unchanged(
    fake_desktop,
):
    """The exact live request through JarvisCore and the REAL
    WindowsComputerBackend: open_url executes, the browser is made active
    BEFORE the second step runs, and scroll reaches PyAutoGUI with
    amount=50 unchanged and no x/y - i.e. at the current cursor position."""
    jarvis, call_provider = make_real_backend_jarvis()

    with patch(
        "app.computer.windows_backend.webbrowser.open",
        side_effect=lambda url: fake_desktop.events.append(("webbrowser.open", url)) or True,
    ), patch(
        "app.computer.windows_backend.pyautogui.scroll",
        side_effect=lambda *a, **kw: fake_desktop.events.append(("pyautogui.scroll", a, kw)),
    ):
        reply, _, _ = jarvis.handle_chat(
            "use tool open_url https://luniti.app, then use tool scroll 50"
        )

    assert fake_desktop.events == [
        ("webbrowser.open", "https://luniti.app"),
        ("activate", BROWSER_HWND),
        ("pyautogui.scroll", (50,), {}),
    ]
    assert reply == "1. Opened 'https://luniti.app'.\n2. Scrolled by 50."
    # Both steps' arguments are bound deterministically from the tools' own
    # schemas - no AI call is made to interpret "50".
    assert call_provider.clauses == []


def test_scroll_never_runs_when_the_browser_could_not_be_made_active(fake_desktop):
    """Fail closed: if Windows refuses activation, the sequence stops at
    open_url - the wheel event is never injected into whatever kept focus."""
    fake_desktop.activation_allowed = False
    jarvis, _ = make_real_backend_jarvis()

    with patch("app.computer.windows_backend.webbrowser.open", return_value=True), patch(
        "app.computer.windows_backend.pyautogui.scroll"
    ) as scroll:
        reply, _, _ = jarvis.handle_chat(
            "use tool open_url https://luniti.app, then use tool scroll 50"
        )

    scroll.assert_not_called()
    assert reply.startswith(
        "1. Opened 'https://luniti.app', but Windows did not let comet.exe"
    )
    assert "\n2." not in reply


# --- scroll: the exact Win32 call, pinning direction and units ---------------


def test_scroll_sends_one_raw_wheel_delta_equal_to_the_amount_positive_meaning_up(
    backend, monkeypatch
):
    """What pyautogui.scroll(50) actually sends on Windows: ONE
    mouse_event(MOUSEEVENTF_WHEEL, ..., dwData=50) at the cursor, with no
    window handle involved. dwData is raw - one wheel notch is
    WHEEL_DELTA=120, so 50 is under half a notch - and positive is
    wheel-forward, i.e. UP, which can't move a page that just loaded at
    its top. Pins the current contract; changing it must be deliberate."""
    import pyautogui

    monkeypatch.setattr(pyautogui, "PAUSE", 0)
    with patch("pyautogui._pyautogui_win._position", return_value=(100, 100)), patch(
        "pyautogui._pyautogui_win._size", return_value=(1920, 1080)
    ), patch.object(ctypes.windll.user32, "mouse_event") as mouse_event:
        backend.scroll(50)

    mouse_event.assert_called_once()
    event_flags, _dx, _dy, wheel_delta, extra_info = mouse_event.call_args.args
    assert event_flags == 0x0800  # MOUSEEVENTF_WHEEL
    assert wheel_delta == 50
    assert extra_info == 0
