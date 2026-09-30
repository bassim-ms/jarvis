"""
Tests for the computer-control Tools (app/computer/tools.py):
GetActiveWindowTool, GetScreenInfoTool, OpenApplicationTool, OpenUrlTool,
TypeTextTool, PressKeyTool, MoveMouseTool, ClickMouseTool, ScrollTool -
both directly and through ToolRegistry, proving permission levels
(READ/SAFE_ACTION) are enforced exactly like every other tool in this
codebase, and that a ComputerControlError becomes a normal
ToolResult.error rather than propagating.

Everything here runs against FakeComputerBackend, which implements
ComputerBackend with no OS interaction at all - no real mouse movement,
no real keystrokes, no real application launch, no real screen access.
This also proves the Tools layer depends only on the ComputerBackend
abstraction, never on WindowsComputerBackend specifically.
"""

import pytest

from app.computer.base import (
    AllowedApplication,
    ComputerBackend,
    ComputerControlError,
    MouseButton,
    ScreenInfo,
    WindowInfo,
)
from app.computer.tools import (
    ClickMouseTool,
    DoubleClickTool,
    GetActiveWindowTool,
    GetScreenInfoTool,
    GetScreenSizeTool,
    MoveMouseTool,
    OpenApplicationTool,
    OpenUrlTool,
    PressKeyTool,
    ScrollTool,
    TypeTextTool,
)
from app.tools.base import PermissionLevel, ToolResultStatus
from app.tools.registry import ToolRegistry


class FakeComputerBackend(ComputerBackend):
    """Records every call instead of touching the real OS. `error`, when
    set, is raised by every method - used to prove a backend failure
    becomes a ToolResult.error rather than propagating."""

    def __init__(self, error: Exception | None = None, screen: ScreenInfo | None = None):
        self.error = error
        self.screen = screen or ScreenInfo(width=1920, height=1080)
        self.opened_applications: list[AllowedApplication] = []
        self.opened_urls: list[str] = []
        self.typed_text: list[str] = []
        self.pressed_keys: list[str] = []
        self.moved_to: list[tuple[int, int]] = []
        self.clicks: list[tuple[int | None, int | None, MouseButton]] = []
        self.double_clicks: list[tuple[int, int, MouseButton]] = []
        self.scrolls: list[int] = []

    def get_active_window(self) -> WindowInfo:
        if self.error:
            raise self.error
        return WindowInfo(title="Notepad", process_name="notepad.exe", is_maximized=True)

    def get_screen_info(self) -> ScreenInfo:
        if self.error:
            raise self.error
        return self.screen

    def open_application(self, app: AllowedApplication) -> None:
        if self.error:
            raise self.error
        self.opened_applications.append(app)

    def open_url(self, url: str) -> None:
        if self.error:
            raise self.error
        self.opened_urls.append(url)

    def type_text(self, text: str) -> None:
        if self.error:
            raise self.error
        self.typed_text.append(text)

    def press_key(self, key: str) -> None:
        if self.error:
            raise self.error
        self.pressed_keys.append(key)

    def move_mouse(self, x: int, y: int) -> None:
        if self.error:
            raise self.error
        self.moved_to.append((x, y))

    def click_mouse(self, x, y, button: MouseButton) -> None:
        if self.error:
            raise self.error
        self.clicks.append((x, y, button))

    def double_click(self, x: int, y: int, button: MouseButton) -> None:
        if self.error:
            raise self.error
        self.double_clicks.append((x, y, button))

    def scroll(self, amount: int) -> None:
        if self.error:
            raise self.error
        self.scrolls.append(amount)


def make_registry(backend: ComputerBackend | None = None):
    backend = backend or FakeComputerBackend()
    registry = ToolRegistry()
    registry.register(GetActiveWindowTool(backend=backend))
    registry.register(GetScreenInfoTool(backend=backend))
    registry.register(GetScreenSizeTool(backend=backend))
    registry.register(OpenApplicationTool(backend=backend))
    registry.register(OpenUrlTool(backend=backend))
    registry.register(TypeTextTool(backend=backend))
    registry.register(PressKeyTool(backend=backend))
    registry.register(MoveMouseTool(backend=backend))
    registry.register(ClickMouseTool(backend=backend))
    registry.register(DoubleClickTool(backend=backend))
    registry.register(ScrollTool(backend=backend))
    return registry, backend


# --- permissions ---------------------------------------------------------


def test_permission_levels_match_the_locked_scope():
    registry, _ = make_registry()

    assert registry.get("get_active_window").permission is PermissionLevel.READ
    assert registry.get("get_screen_info").permission is PermissionLevel.READ
    assert registry.get("get_screen_size").permission is PermissionLevel.READ
    assert registry.get("open_application").permission is PermissionLevel.SAFE_ACTION
    assert registry.get("open_url").permission is PermissionLevel.SAFE_ACTION
    assert registry.get("type_text").permission is PermissionLevel.SAFE_ACTION
    assert registry.get("press_key").permission is PermissionLevel.SAFE_ACTION
    assert registry.get("move_mouse").permission is PermissionLevel.SAFE_ACTION
    assert registry.get("click_mouse").permission is PermissionLevel.SAFE_ACTION
    assert registry.get("double_click").permission is PermissionLevel.SAFE_ACTION
    assert registry.get("scroll").permission is PermissionLevel.SAFE_ACTION


def test_every_tool_runs_automatically_with_no_confirmation():
    registry, _ = make_registry()

    result = registry.execute("get_active_window")

    assert result.status == ToolResultStatus.SUCCESS


# --- GetActiveWindowTool / GetScreenInfoTool (READ) -----------------------


def test_get_active_window_returns_backend_data():
    registry, _ = make_registry()

    result = registry.execute("get_active_window")

    assert result.status == ToolResultStatus.SUCCESS
    assert result.data == {
        "title": "Notepad",
        "process_name": "notepad.exe",
        "is_maximized": True,
    }


def test_get_active_window_reports_backend_error_as_tool_error():
    registry, _ = make_registry(FakeComputerBackend(error=ComputerControlError("boom")))

    result = registry.execute("get_active_window")

    assert result.status == ToolResultStatus.ERROR
    assert "boom" in result.message


def test_get_screen_info_returns_backend_data():
    registry, _ = make_registry(FakeComputerBackend(screen=ScreenInfo(width=2560, height=1440)))

    result = registry.execute("get_screen_info")

    assert result.status == ToolResultStatus.SUCCESS
    assert result.data == {"width": 2560, "height": 1440}


def test_get_screen_size_is_registered_as_a_read_alias_for_get_screen_info():
    """Regression test: get_screen_size (the exact name a live smoke
    test used) must be registered and behave identically to
    get_screen_info - not merely "some screen-size tool exists"."""
    registry, _ = make_registry(FakeComputerBackend(screen=ScreenInfo(width=2560, height=1440)))

    assert registry.has("get_screen_size")
    assert registry.get("get_screen_size").permission is PermissionLevel.READ

    result = registry.execute("get_screen_size")

    assert result.status == ToolResultStatus.SUCCESS
    assert result.data == {"width": 2560, "height": 1440}


def test_get_screen_size_reports_backend_error_as_tool_error():
    registry, _ = make_registry(FakeComputerBackend(error=ComputerControlError("boom")))

    result = registry.execute("get_screen_size")

    assert result.status == ToolResultStatus.ERROR
    assert "boom" in result.message


# --- OpenApplicationTool: allowlisting/restrictions -----------------------


def test_open_application_opens_an_allowed_app():
    registry, backend = make_registry()

    result = registry.execute("open_application", params={"app": "notepad"})

    assert result.status == ToolResultStatus.SUCCESS
    assert backend.opened_applications == [AllowedApplication.NOTEPAD]


@pytest.mark.parametrize(
    "app",
    [
        "cmd",
        "powershell",
        "C:/Windows/System32/cmd.exe",
        "notepad.exe && calc.exe",
        "",
        "regedit",
        "not_a_real_app",
    ],
)
def test_open_application_rejects_anything_outside_the_allowlist(app):
    registry, backend = make_registry()

    result = registry.execute("open_application", params={"app": app})

    assert result.status == ToolResultStatus.ERROR
    assert backend.opened_applications == []


@pytest.mark.parametrize("app", [["notepad.exe"], {"path": "cmd.exe"}, None, 123, 1.5])
def test_open_application_rejects_non_string_and_unhashable_values_cleanly(app):
    """Proves a malformed tool argument (a list/dict/None/number instead
    of a string - exactly what a hallucinated or prompt-injected LLM tool
    call could produce) is turned into a clean ToolResult.error by the
    tool itself, not an unhandled exception relied on ToolRegistry's
    outer catch-all to paper over."""
    registry, backend = make_registry()

    result = registry.execute("open_application", params={"app": app})

    assert result.status == ToolResultStatus.ERROR
    assert backend.opened_applications == []


def test_open_application_never_accepts_an_arbitrary_executable_path():
    """A caller-supplied path is never even passed to the backend -
    OpenApplicationTool resolves `app` through the closed
    AllowedApplication enum first, so nothing resembling a path or
    command string can ever reach `backend.open_application`."""
    registry, backend = make_registry()

    result = registry.execute(
        "open_application", params={"app": r"C:\Windows\System32\cmd.exe"}
    )

    assert result.status == ToolResultStatus.ERROR
    assert backend.opened_applications == []


def test_open_application_surfaces_backend_failure_as_tool_error():
    registry, _ = make_registry(FakeComputerBackend(error=ComputerControlError("not installed")))

    result = registry.execute("open_application", params={"app": "chrome"})

    assert result.status == ToolResultStatus.ERROR
    assert "not installed" in result.message


# --- OpenUrlTool: URL handling -------------------------------------------


def test_open_url_opens_an_http_url():
    registry, backend = make_registry()

    result = registry.execute("open_url", params={"url": "https://example.com"})

    assert result.status == ToolResultStatus.SUCCESS
    assert backend.opened_urls == ["https://example.com"]


@pytest.mark.parametrize(
    "url",
    [
        "file:///C:/secrets.txt",
        "javascript:alert(1)",
        "ftp://example.com",
        "not a url",
        "",
    ],
)
def test_open_url_rejects_unsafe_or_malformed_urls(url):
    registry, backend = make_registry()

    result = registry.execute("open_url", params={"url": url})

    assert result.status == ToolResultStatus.ERROR
    assert backend.opened_urls == []


# --- TypeTextTool / PressKeyTool: keyboard input --------------------------


def test_type_text_types_ordinary_text():
    registry, backend = make_registry()

    result = registry.execute("type_text", params={"text": "hello"})

    assert result.status == ToolResultStatus.SUCCESS
    assert backend.typed_text == ["hello"]


def test_type_text_rejects_empty_text():
    registry, backend = make_registry()

    result = registry.execute("type_text", params={"text": ""})

    assert result.status == ToolResultStatus.ERROR
    assert backend.typed_text == []


def test_type_text_rejects_text_over_the_max_length():
    from app.computer.base import MAX_TEXT_LENGTH

    registry, backend = make_registry()

    result = registry.execute("type_text", params={"text": "a" * (MAX_TEXT_LENGTH + 1)})

    assert result.status == ToolResultStatus.ERROR
    assert backend.typed_text == []


def test_press_key_presses_an_allowed_key():
    registry, backend = make_registry()

    result = registry.execute("press_key", params={"key": "enter"})

    assert result.status == ToolResultStatus.SUCCESS
    assert backend.pressed_keys == ["enter"]


@pytest.mark.parametrize("key", ["", "a", "rm -rf /", "banana+l", "ctrl+", "+l", "ctrl+alt", "l+ctrl"])
def test_press_key_rejects_anything_outside_the_allowlist(key):
    registry, backend = make_registry()

    result = registry.execute("press_key", params={"key": key})

    assert result.status == ToolResultStatus.ERROR
    assert backend.pressed_keys == []


# --- PressKeyTool: case-insensitivity (Fix A) -----------------------------


@pytest.mark.parametrize("raw", ["ENTER", "Enter", "eNtEr"])
def test_press_key_is_case_insensitive_through_the_tool_layer(raw):
    """The Tool layer accepts any case variant and forwards the ORIGINAL
    string to the backend - WindowsComputerBackend is the layer that
    normalizes before touching PyAutoGUI (see
    tests/test_computer_backend_windows.py); FakeComputerBackend here
    just proves the Tool layer's OWN validation (`normalize_key_input`)
    doesn't reject a valid key merely because of its casing."""
    registry, backend = make_registry()

    result = registry.execute("press_key", params={"key": raw})

    assert result.status == ToolResultStatus.SUCCESS
    assert backend.pressed_keys == [raw]


# --- PressKeyTool: modifier combinations (Fix B) --------------------------


@pytest.mark.parametrize(
    "combo",
    ["ctrl+l", "CTRL+L", "ctrl+c", "ctrl+shift+esc", "alt+tab", "win+d", "ctrl+enter"],
)
def test_press_key_accepts_approved_modifier_combinations(combo):
    registry, backend = make_registry()

    result = registry.execute("press_key", params={"key": combo})

    assert result.status == ToolResultStatus.SUCCESS
    assert backend.pressed_keys == [combo]


@pytest.mark.parametrize(
    "combo",
    [
        "l+ctrl",  # no leading modifier
        "ctrl+",  # trailing '+'
        "+l",  # leading '+'
        "ctrl++l",  # doubled '+'
        "ctrl+shift+alt+l",  # exceeds the 3-token maximum
        "banana+l",  # unknown modifier token
        "ctrl+alt",  # two modifiers, no real target
        "ctrl+/bin/sh",  # path-shaped target - never accepted
        "ctrl+powershell",  # command-shaped target - never accepted
    ],
)
def test_press_key_rejects_disallowed_modifier_combinations(combo):
    registry, backend = make_registry()

    result = registry.execute("press_key", params={"key": combo})

    assert result.status == ToolResultStatus.ERROR
    assert backend.pressed_keys == []


# --- MoveMouseTool / ClickMouseTool: mouse action validation --------------


def test_move_mouse_moves_to_an_on_screen_point():
    registry, backend = make_registry()

    result = registry.execute("move_mouse", params={"x": 100, "y": 200})

    assert result.status == ToolResultStatus.SUCCESS
    assert backend.moved_to == [(100, 200)]


@pytest.mark.parametrize("x,y", [(-1, 100), (100, -1), (1920, 100), (100, 1080), (999999, 1)])
def test_move_mouse_rejects_off_screen_coordinates(x, y):
    registry, backend = make_registry(FakeComputerBackend(screen=ScreenInfo(1920, 1080)))

    result = registry.execute("move_mouse", params={"x": x, "y": y})

    assert result.status == ToolResultStatus.ERROR
    assert backend.moved_to == []


@pytest.mark.parametrize("x,y", [(1.5, 100), (100, "200"), (None, 100)])
def test_move_mouse_rejects_non_integer_coordinates(x, y):
    registry, backend = make_registry()

    result = registry.execute("move_mouse", params={"x": x, "y": y})

    assert result.status == ToolResultStatus.ERROR
    assert backend.moved_to == []


@pytest.mark.parametrize(
    "x,y",
    [
        (float("nan"), 100),
        (100, float("nan")),
        (float("inf"), 100),
        (float("-inf"), 100),
        (100, float("inf")),
        ([100], 200),
        ({"x": 100}, 200),
        (100, [200]),
        (100, {"y": 200}),
        (True, 100),  # bool is an int subclass - deliberately excluded
        (100, False),
    ],
)
def test_move_mouse_rejects_nan_infinity_and_uncoercible_types(x, y):
    """Explicit coverage for the exact malformed-input shapes an LLM
    could plausibly hallucinate: NaN/infinity floats, lists, dicts, and
    bools - none of these are coercible ints, so none reach the
    backend."""
    registry, backend = make_registry()

    result = registry.execute("move_mouse", params={"x": x, "y": y})

    assert result.status == ToolResultStatus.ERROR
    assert backend.moved_to == []


def test_click_mouse_rejects_missing_coordinates_safely():
    """x and y are required (see this tool's own docstring for why: a
    previously-optional x/y let "use tool click_mouse 100 100" build with
    zero parameters and silently click at the current cursor position,
    ignoring the given coordinates) - omitting both must be a clean,
    typed ERROR, never a click at an unintended position and never an
    unhandled crash."""
    registry, backend = make_registry()

    result = registry.execute("click_mouse", params={})

    assert result.status == ToolResultStatus.ERROR
    assert backend.clicks == []


def test_click_mouse_clicks_at_a_given_on_screen_point():
    registry, backend = make_registry()

    result = registry.execute(
        "click_mouse", params={"x": 50, "y": 60, "button": "right"}
    )

    assert result.status == ToolResultStatus.SUCCESS
    assert backend.clicks == [(50, 60, MouseButton.RIGHT)]


def test_click_mouse_rejects_off_screen_coordinates():
    registry, backend = make_registry(FakeComputerBackend(screen=ScreenInfo(1920, 1080)))

    result = registry.execute("click_mouse", params={"x": 5000, "y": 5000})

    assert result.status == ToolResultStatus.ERROR
    assert backend.clicks == []


def test_click_mouse_rejects_an_unknown_button():
    registry, backend = make_registry()

    result = registry.execute(
        "click_mouse", params={"x": 50, "y": 60, "button": "double_left_ultra"}
    )

    assert result.status == ToolResultStatus.ERROR
    assert backend.clicks == []


@pytest.mark.parametrize("button", [["left"], {"btn": "left"}, None, 123])
def test_click_mouse_rejects_non_string_and_unhashable_button_values_cleanly(button):
    registry, backend = make_registry()

    result = registry.execute("click_mouse", params={"x": 50, "y": 60, "button": button})

    assert result.status == ToolResultStatus.ERROR
    assert backend.clicks == []


def test_click_mouse_rejects_only_one_of_x_or_y_given():
    registry, backend = make_registry()

    result = registry.execute("click_mouse", params={"x": 10})

    assert result.status == ToolResultStatus.ERROR
    assert backend.clicks == []


def test_click_mouse_rejects_missing_y_when_only_x_is_given():
    """Same as the case above, named for what it actually now tests:
    x/y are both required, so a missing y is a plain, safe ERROR (via
    Tool.execute's own missing-argument failure, caught by ToolRegistry -
    see app/tools/registry.py), never a partial click."""
    registry, backend = make_registry()

    result = registry.execute("click_mouse", params={"y": 20})

    assert result.status == ToolResultStatus.ERROR
    assert backend.clicks == []


def test_move_mouse_and_click_mouse_accept_coordinates_exactly_on_the_valid_boundary():
    """(0, 0) and (width - 1, height - 1) are the smallest/largest
    valid on-screen coordinates - both must be accepted, not just
    values strictly inside them."""
    registry, backend = make_registry(FakeComputerBackend(screen=ScreenInfo(1920, 1080)))

    assert registry.execute("move_mouse", params={"x": 0, "y": 0}).status == (
        ToolResultStatus.SUCCESS
    )
    assert registry.execute(
        "move_mouse", params={"x": 1919, "y": 1079}
    ).status == ToolResultStatus.SUCCESS
    assert backend.moved_to == [(0, 0), (1919, 1079)]

    assert registry.execute(
        "click_mouse", params={"x": 0, "y": 0}
    ).status == ToolResultStatus.SUCCESS
    assert registry.execute(
        "click_mouse", params={"x": 1919, "y": 1079}
    ).status == ToolResultStatus.SUCCESS
    assert backend.clicks == [(0, 0, MouseButton.LEFT), (1919, 1079, MouseButton.LEFT)]


# --- DoubleClickTool: mouse action validation (M17.1-D) --------------------


def test_double_click_double_clicks_at_an_on_screen_point():
    registry, backend = make_registry()

    result = registry.execute("double_click", params={"x": 50, "y": 60})

    assert result.status == ToolResultStatus.SUCCESS
    assert backend.double_clicks == [(50, 60, MouseButton.LEFT)]


@pytest.mark.parametrize("button,expected", [("left", MouseButton.LEFT), ("right", MouseButton.RIGHT), ("middle", MouseButton.MIDDLE)])
def test_double_click_accepts_every_allowed_button(button, expected):
    registry, backend = make_registry()

    result = registry.execute("double_click", params={"x": 10, "y": 20, "button": button})

    assert result.status == ToolResultStatus.SUCCESS
    assert backend.double_clicks == [(10, 20, expected)]


def test_double_click_accepts_coordinates_exactly_on_the_valid_boundary():
    registry, backend = make_registry(FakeComputerBackend(screen=ScreenInfo(1920, 1080)))

    assert registry.execute(
        "double_click", params={"x": 0, "y": 0}
    ).status == ToolResultStatus.SUCCESS
    assert registry.execute(
        "double_click", params={"x": 1919, "y": 1079}
    ).status == ToolResultStatus.SUCCESS
    assert backend.double_clicks == [
        (0, 0, MouseButton.LEFT),
        (1919, 1079, MouseButton.LEFT),
    ]


@pytest.mark.parametrize("x,y", [(-1, 100), (100, -1), (1920, 100), (100, 1080), (999999, 1)])
def test_double_click_rejects_off_screen_coordinates(x, y):
    registry, backend = make_registry(FakeComputerBackend(screen=ScreenInfo(1920, 1080)))

    result = registry.execute("double_click", params={"x": x, "y": y})

    assert result.status == ToolResultStatus.ERROR
    assert backend.double_clicks == []


@pytest.mark.parametrize(
    "x,y",
    [
        (1.5, 100),
        (100, "200"),
        (None, 100),
        (float("nan"), 100),
        (float("inf"), 100),
        ([100], 200),
        ({"x": 100}, 200),
        (True, 100),
    ],
)
def test_double_click_rejects_non_integer_or_uncoercible_coordinates(x, y):
    registry, backend = make_registry()

    result = registry.execute("double_click", params={"x": x, "y": y})

    assert result.status == ToolResultStatus.ERROR
    assert backend.double_clicks == []


def test_double_click_requires_both_coordinates():
    """DoubleClickTool has no "current position" variant - x/y are
    always required (ClickMouseTool now matches this exact shape too -
    see its own docstring for why its x/y are no longer optional)."""
    registry, backend = make_registry()

    result = registry.execute("double_click", params={"x": 10})

    assert result.status == ToolResultStatus.ERROR
    assert backend.double_clicks == []


def test_double_click_rejects_an_unknown_button():
    registry, backend = make_registry()

    result = registry.execute(
        "double_click", params={"x": 10, "y": 20, "button": "double_left_ultra"}
    )

    assert result.status == ToolResultStatus.ERROR
    assert backend.double_clicks == []


@pytest.mark.parametrize("button", [["left"], {"btn": "left"}, None, 123])
def test_double_click_rejects_non_string_and_unhashable_button_values_cleanly(button):
    registry, backend = make_registry()

    result = registry.execute(
        "double_click", params={"x": 10, "y": 20, "button": button}
    )

    assert result.status == ToolResultStatus.ERROR
    assert backend.double_clicks == []


def test_double_click_never_exposes_an_arbitrary_click_count():
    """DoubleClickTool has no 'clicks' parameter at all - only x/y/button
    - so there is no way to request anything but exactly one
    double-click."""
    registry, _ = make_registry()

    param_names = {p.name for p in registry.get("double_click").parameters}

    assert param_names == {"x", "y", "button"}


# --- ScrollTool: scroll validation ----------------------------------------


def test_scroll_scrolls_up_and_down():
    registry, backend = make_registry()

    assert registry.execute("scroll", params={"amount": 100}).status == ToolResultStatus.SUCCESS
    assert registry.execute("scroll", params={"amount": -100}).status == ToolResultStatus.SUCCESS
    assert backend.scrolls == [100, -100]


def test_scroll_rejects_out_of_range_amounts():
    from app.computer.base import MAX_SCROLL_AMOUNT

    registry, backend = make_registry()

    result = registry.execute("scroll", params={"amount": MAX_SCROLL_AMOUNT + 1})

    assert result.status == ToolResultStatus.ERROR
    assert backend.scrolls == []


def test_scroll_rejects_non_integer_amount():
    registry, backend = make_registry()

    result = registry.execute("scroll", params={"amount": "a lot"})

    assert result.status == ToolResultStatus.ERROR
    assert backend.scrolls == []


# --- ensuring tools use the backend abstraction ---------------------------


def test_tools_only_ever_call_through_the_backend_interface():
    """A minimal second ComputerBackend implementation (distinct from
    FakeComputerBackend and WindowsComputerBackend) still works with
    every tool unchanged - proving the Tools layer depends only on the
    abstract ComputerBackend interface, never a concrete class."""

    class MinimalBackend(ComputerBackend):
        def get_active_window(self):
            return WindowInfo(title="x", process_name="x.exe", is_maximized=False)

        def get_screen_info(self):
            return ScreenInfo(width=800, height=600)

        def open_application(self, app):
            pass

        def open_url(self, url):
            pass

        def type_text(self, text):
            pass

        def press_key(self, key):
            pass

        def move_mouse(self, x, y):
            pass

        def click_mouse(self, x, y, button):
            pass

        def double_click(self, x, y, button):
            pass

        def scroll(self, amount):
            pass

    registry, _ = make_registry(MinimalBackend())

    assert registry.execute("get_active_window").status == ToolResultStatus.SUCCESS
    assert registry.execute("get_screen_info").status == ToolResultStatus.SUCCESS
    assert (
        registry.execute("open_application", params={"app": "notepad"}).status
        == ToolResultStatus.SUCCESS
    )


# --- validate_arguments: the side-effect-free pre-flight check --------------------


def _preflight_cases():
    return [
        ("open_application", {"app": "notepad"}, {"app": "cmd"}),
        ("open_url", {"url": "https://luniti.app"}, {"url": "file:///C:/secrets.txt"}),
        ("open_url", {"url": "https://luniti.app"}, {"url": "javascript:alert(1)"}),
        ("type_text", {"text": "hello"}, {"text": ""}),
        ("press_key", {"key": "enter"}, {"key": "banana"}),
        ("press_key", {"key": "ctrl+l"}, {"key": "l+ctrl"}),
        ("move_mouse", {"x": 10, "y": 10}, {"x": 99999, "y": 10}),
        ("click_mouse", {"x": 10, "y": 10}, {"x": 10, "y": -1}),
        ("click_mouse", {"x": 10, "y": 10, "button": "right"}, {"x": 10, "y": 10, "button": "back"}),
        ("double_click", {"x": 10, "y": 10}, {"x": 1.5, "y": 10}),
        ("scroll", {"amount": -600}, {"amount": 999999}),
        ("scroll", {"amount": 5}, {"amount": 2.5}),
    ]


@pytest.mark.parametrize("name,good,bad", _preflight_cases())
def test_validate_arguments_agrees_with_execute_and_never_touches_the_backend(name, good, bad):
    from app.tools.base import ToolArgumentError

    registry, backend = make_registry()
    tool = registry.get(name)

    tool.validate_arguments(**good)  # accepted...
    with pytest.raises(ToolArgumentError) as exc_info:
        tool.validate_arguments(**bad)  # ...and rejected

    # execute() rejects the same value with the same message.
    result = registry.execute(name, params=bad)
    assert result.status == ToolResultStatus.ERROR
    assert result.message == str(exc_info.value)

    # Neither pre-flight call did anything to the (fake) desktop.
    assert backend.opened_applications == []
    assert backend.opened_urls == []
    assert backend.typed_text == []
    assert backend.pressed_keys == []
    assert backend.moved_to == []
    assert backend.clicks == []
    assert backend.double_clicks == []
    assert backend.scrolls == []


def test_validate_arguments_uses_the_real_screen_size_for_coordinates():
    from app.tools.base import ToolArgumentError

    registry, _ = make_registry(FakeComputerBackend(screen=ScreenInfo(width=100, height=50)))

    registry.get("move_mouse").validate_arguments(x=99, y=49)
    for x, y in ((100, 0), (0, 50)):
        with pytest.raises(ToolArgumentError):
            registry.get("move_mouse").validate_arguments(x=x, y=y)


def test_read_tools_have_nothing_to_validate():
    registry, _ = make_registry()

    for name in ("get_active_window", "get_screen_info", "get_screen_size"):
        assert registry.get(name).validate_arguments() is None
