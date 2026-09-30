"""
Computer Control tools (M17.1-A): adapt ComputerBackend
(app/computer/base.py) into the Tool interface, the same "Tool depends
only on an abstract backend, never a concrete implementation" shape
web_search/calendar/reminders/automation tools already use.

This module is the boundary that receives raw, string/number parameters
(ultimately from an LLM's tool call, once a later milestone wires that
up - see this module's own security note below) - every parameter is
validated here, using the SAME `validate_*` functions/constants
app/computer/base.py exports, before a backend ever sees it. A backend
(WindowsComputerBackend) re-validates too, as defense in depth, but this
is the layer that turns a bad parameter into a clean ToolResult.error
instead of an exception or an unsafe call.

Permission levels: `get_active_window`/`get_screen_info` are READ (no
side effects). Every other tool here is SAFE_ACTION - a real but narrowly
-scoped, automatically-approved side effect (opening an allowed app/URL,
typing, pressing one allowed key, moving/clicking the mouse, scrolling) -
per this milestone's own scope. None of that is enforced here;
ToolRegistry.execute (app/tools/registry.py) already does it, unchanged,
for every tool, using AUTOMATIC_PERMISSIONS (READ + SAFE_ACTION run
immediately, no confirmation round-trip) exactly as it does for every
other tool in this codebase.

M17.1-A note (see this milestone's own scope): these Tool classes are
NOT registered in app/tools/defaults.py, so they are not yet reachable
through the running application's TOOL capability, AI tool-calling, or
M15 Job execution - that wiring, and any confirmation-flow changes it
implies, is deliberately left to M17.1-B/C. Today they exist only to be
constructed and exercised directly (e.g. in tests), against either
WindowsComputerBackend or a fake backend.
"""

from typing import Any

from app.computer.base import (
    AllowedApplication,
    ComputerBackend,
    ComputerControlError,
    MouseButton,
    normalize_key_input,
    validate_coordinates,
    validate_scroll_amount,
    validate_text,
    validate_url,
)
from app.tools.base import PermissionLevel, Tool, ToolArgumentError, ToolParameter, ToolResult


class GetActiveWindowTool(Tool):
    """Reads the current foreground window's title/process/maximized
    state. READ: no side effects."""

    name = "get_active_window"
    description = "Get the title, owning process, and maximized state of the active window."
    permission = PermissionLevel.READ

    def __init__(self, backend: ComputerBackend):
        self.backend = backend

    def execute(self) -> ToolResult:
        try:
            window = self.backend.get_active_window()
        except ComputerControlError as exc:
            return ToolResult.error(str(exc))

        return ToolResult.success(
            data={
                "title": window.title,
                "process_name": window.process_name,
                "is_maximized": window.is_maximized,
            }
        )


class GetScreenInfoTool(Tool):
    """Reads the primary display's size. READ: no side effects."""

    name = "get_screen_info"
    description = "Get the primary screen's width and height, in pixels."
    permission = PermissionLevel.READ

    def __init__(self, backend: ComputerBackend):
        self.backend = backend

    def execute(self) -> ToolResult:
        try:
            screen = self.backend.get_screen_info()
        except ComputerControlError as exc:
            return ToolResult.error(str(exc))

        return ToolResult.success(data={"width": screen.width, "height": screen.height})


class GetScreenSizeTool(GetScreenInfoTool):
    """Identical to GetScreenInfoTool - same backend call, same READ
    permission, same data shape - registered under the literal name
    "get_screen_size" (M17.1-D's original spec name for this
    capability). Not a new capability: zero new logic, zero new backend
    method, zero new validation - purely a second registered name for
    the exact same `ComputerBackend.get_screen_info()` read, so a
    caller/tool-invocation using either name works."""

    name = "get_screen_size"


#: Shared by OpenApplicationTool's `parameters` and `execute` - the exact
#: set of names a caller may pass, spelled out once rather than derived
#: implicitly, so the declared parameter description and the actual
#: accepted values can never drift apart.
_ALLOWED_APPLICATION_NAMES = sorted(app.value for app in AllowedApplication)


class OpenApplicationTool(Tool):
    """Launches one application from the fixed AllowedApplication set.
    SAFE_ACTION: a real side effect, but bounded to a small, closed,
    reviewed list - never an arbitrary executable path (see
    app/computer/base.py's own docstring for why)."""

    name = "open_application"
    description = (
        "Open one of a fixed set of applications: "
        + ", ".join(_ALLOWED_APPLICATION_NAMES)
    )
    permission = PermissionLevel.SAFE_ACTION

    def __init__(self, backend: ComputerBackend):
        self.backend = backend

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="app",
                type="string",
                description="Which application to open: " + ", ".join(_ALLOWED_APPLICATION_NAMES),
            )
        ]

    def validate_arguments(self, app: str) -> None:
        try:
            AllowedApplication(app)
        except ValueError:
            raise ToolArgumentError(
                f"'{app}' is not an allowed application. Allowed: "
                + ", ".join(_ALLOWED_APPLICATION_NAMES)
            ) from None

    def execute(self, app: str) -> ToolResult:
        try:
            application = AllowedApplication(app)
        except ValueError:
            return ToolResult.error(
                f"'{app}' is not an allowed application. Allowed: "
                + ", ".join(_ALLOWED_APPLICATION_NAMES)
            )

        try:
            self.backend.open_application(application)
        except ComputerControlError as exc:
            return ToolResult.error(str(exc))

        return ToolResult.success(message=f"Opened '{application.value}'.")


class OpenUrlTool(Tool):
    """Opens a URL through the normal browser mechanism. SAFE_ACTION:
    restricted to http/https URLs only (see `validate_url`)."""

    name = "open_url"
    description = "Open a URL (http/https only) in the default browser."
    permission = PermissionLevel.SAFE_ACTION

    def __init__(self, backend: ComputerBackend):
        self.backend = backend

    @property
    def parameters(self) -> list[ToolParameter]:
        return [ToolParameter(name="url", type="string", description="The http/https URL to open")]

    def validate_arguments(self, url: str) -> None:
        try:
            validate_url(url)
        except ComputerControlError as exc:
            raise ToolArgumentError(str(exc)) from exc

    def execute(self, url: str) -> ToolResult:
        try:
            validate_url(url)
            self.backend.open_url(url)
        except ComputerControlError as exc:
            return ToolResult.error(str(exc))

        return ToolResult.success(message=f"Opened '{url}'.")


class TypeTextTool(Tool):
    """Types text via the keyboard into whatever currently has focus.
    SAFE_ACTION: bounded length (see `validate_text`/`MAX_TEXT_LENGTH`)."""

    name = "type_text"
    description = "Type text via the keyboard into whatever currently has focus."
    permission = PermissionLevel.SAFE_ACTION

    def __init__(self, backend: ComputerBackend):
        self.backend = backend

    @property
    def parameters(self) -> list[ToolParameter]:
        return [ToolParameter(
                name="text", type="string", description="The text to type", free_text=True
            )]

    def validate_arguments(self, text: str) -> None:
        try:
            validate_text(text)
        except ComputerControlError as exc:
            raise ToolArgumentError(str(exc)) from exc

    def execute(self, text: str) -> ToolResult:
        try:
            validate_text(text)
            self.backend.type_text(text)
        except ComputerControlError as exc:
            return ToolResult.error(str(exc))

        return ToolResult.success(message="Text typed.")


class PressKeyTool(Tool):
    """Presses a single named key, or (M17.1-D) a narrowly-scoped
    "+"-joined modifier combination. SAFE_ACTION: restricted to
    `ALLOWED_KEYS` for a standalone key, or `MODIFIER_KEYS` +
    `ALLOWED_COMBO_TARGETS` for a combination (see app/computer/base.py)."""

    name = "press_key"
    description = (
        "Press a single named key (e.g. enter, tab, esc, up, f5), or a "
        "'+'-joined modifier combination (e.g. ctrl+l, ctrl+shift+esc, "
        "alt+tab)."
    )
    permission = PermissionLevel.SAFE_ACTION

    def __init__(self, backend: ComputerBackend):
        self.backend = backend

    @property
    def parameters(self) -> list[ToolParameter]:
        return [ToolParameter(name="key", type="string", description="The key to press")]

    def validate_arguments(self, key: str) -> None:
        try:
            normalize_key_input(key)
        except ComputerControlError as exc:
            raise ToolArgumentError(str(exc)) from exc

    def execute(self, key: str) -> ToolResult:
        try:
            normalize_key_input(key)
            self.backend.press_key(key)
        except ComputerControlError as exc:
            return ToolResult.error(str(exc))

        return ToolResult.success(message=f"Pressed '{key}'.")


def _validate_coordinates(backend: ComputerBackend, x: Any, y: Any) -> tuple[int, int]:
    """Shared by MoveMouseTool/ClickMouseTool: fetches the backend's own
    current ScreenInfo and delegates to `validate_coordinates`
    (app/computer/base.py) - the same function WindowsComputerBackend
    itself calls as defense in depth, so there is exactly one definition
    of "on-screen" for both layers."""
    screen = backend.get_screen_info()
    return validate_coordinates(x, y, screen)


class MoveMouseTool(Tool):
    """Moves the mouse cursor to an on-screen coordinate. SAFE_ACTION:
    coordinates are validated against the real screen size before the
    backend ever sees them."""

    name = "move_mouse"
    description = "Move the mouse cursor to screen coordinates (x, y)."
    permission = PermissionLevel.SAFE_ACTION

    def __init__(self, backend: ComputerBackend):
        self.backend = backend

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(name="x", type="number", description="Target x coordinate, in pixels"),
            ToolParameter(name="y", type="number", description="Target y coordinate, in pixels"),
        ]

    def validate_arguments(self, x: int, y: int) -> None:
        try:
            _validate_coordinates(self.backend, x, y)
        except ComputerControlError as exc:
            raise ToolArgumentError(str(exc)) from exc

    def execute(self, x: int, y: int) -> ToolResult:
        try:
            x, y = _validate_coordinates(self.backend, x, y)
            self.backend.move_mouse(x, y)
        except ComputerControlError as exc:
            return ToolResult.error(str(exc))

        return ToolResult.success(message=f"Moved mouse to ({x}, {y}).")


class ClickMouseTool(Tool):
    """Clicks a mouse button at an on-screen coordinate (x, y required -
    see this class's own regression tests for why: a required-but-
    declared-optional x/y let "use tool click_mouse 100 100" build with
    zero parameters and silently click at the current cursor position
    instead, ignoring the given coordinates entirely). SAFE_ACTION:
    coordinates are validated against the real screen size (the same
    `_validate_coordinates` helper MoveMouseTool/DoubleClickTool already
    use) and the button against the same closed `MouseButton` set -
    mirrors DoubleClickTool's own parameter shape exactly."""

    name = "click_mouse"
    description = "Click the mouse (left/right/middle) at (x, y)."
    permission = PermissionLevel.SAFE_ACTION

    def __init__(self, backend: ComputerBackend):
        self.backend = backend

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(name="x", type="number", description="Target x coordinate"),
            ToolParameter(name="y", type="number", description="Target y coordinate"),
            ToolParameter(
                name="button",
                type="string",
                description="Which button to click: left, right, or middle",
                required=False,
                default=MouseButton.LEFT.value,
            ),
        ]

    def validate_arguments(
        self, x: int, y: int, button: str = MouseButton.LEFT.value
    ) -> None:
        try:
            _validate_coordinates(self.backend, x, y)
        except ComputerControlError as exc:
            raise ToolArgumentError(str(exc)) from exc
        try:
            MouseButton(button)
        except ValueError:
            raise ToolArgumentError(
                f"'{button}' is not an allowed button. Allowed: "
                + ", ".join(b.value for b in MouseButton)
            ) from None

    def execute(self, x: int, y: int, button: str = MouseButton.LEFT.value) -> ToolResult:
        try:
            x, y = _validate_coordinates(self.backend, x, y)

            try:
                mouse_button = MouseButton(button)
            except ValueError:
                raise ComputerControlError(
                    f"'{button}' is not an allowed button. Allowed: "
                    + ", ".join(b.value for b in MouseButton)
                ) from None

            self.backend.click_mouse(x, y, mouse_button)
        except ComputerControlError as exc:
            return ToolResult.error(str(exc))

        return ToolResult.success(message=f"Clicked {mouse_button.value} at ({x}, {y}).")


class DoubleClickTool(Tool):
    """Double-clicks a mouse button at an on-screen coordinate
    (M17.1-D). SAFE_ACTION: coordinates are validated against the real
    screen size (the same `_validate_coordinates` helper ClickMouseTool/
    MoveMouseTool already use) and the button against the same closed
    `MouseButton` set - always exactly one double-click, never an
    arbitrary click count (there is no "clicks" parameter anywhere on
    this Tool)."""

    name = "double_click"
    description = "Double-click the mouse (left/right/middle) at (x, y)."
    permission = PermissionLevel.SAFE_ACTION

    def __init__(self, backend: ComputerBackend):
        self.backend = backend

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(name="x", type="number", description="Target x coordinate"),
            ToolParameter(name="y", type="number", description="Target y coordinate"),
            ToolParameter(
                name="button",
                type="string",
                description="Which button to double-click: left, right, or middle",
                required=False,
                default=MouseButton.LEFT.value,
            ),
        ]

    def validate_arguments(
        self, x: int, y: int, button: str = MouseButton.LEFT.value
    ) -> None:
        try:
            _validate_coordinates(self.backend, x, y)
        except ComputerControlError as exc:
            raise ToolArgumentError(str(exc)) from exc
        try:
            MouseButton(button)
        except ValueError:
            raise ToolArgumentError(
                f"'{button}' is not an allowed button. Allowed: "
                + ", ".join(b.value for b in MouseButton)
            ) from None

    def execute(self, x: int, y: int, button: str = MouseButton.LEFT.value) -> ToolResult:
        try:
            x, y = _validate_coordinates(self.backend, x, y)

            try:
                mouse_button = MouseButton(button)
            except ValueError:
                raise ComputerControlError(
                    f"'{button}' is not an allowed button. Allowed: "
                    + ", ".join(b.value for b in MouseButton)
                ) from None

            self.backend.double_click(x, y, mouse_button)
        except ComputerControlError as exc:
            return ToolResult.error(str(exc))

        return ToolResult.success(
            message=f"Double-clicked {mouse_button.value} at ({x}, {y})."
        )


class ScrollTool(Tool):
    """Scrolls at the current cursor position. SAFE_ACTION: bounded
    magnitude (see `validate_scroll_amount`/`MAX_SCROLL_AMOUNT`)."""

    name = "scroll"
    description = "Scroll up (positive) or down (negative) at the current cursor position."
    permission = PermissionLevel.SAFE_ACTION

    def __init__(self, backend: ComputerBackend):
        self.backend = backend

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="amount",
                type="number",
                description="Scroll amount: positive scrolls up, negative scrolls down",
            )
        ]

    def validate_arguments(self, amount: int) -> None:
        try:
            validate_scroll_amount(amount)
        except ComputerControlError as exc:
            raise ToolArgumentError(str(exc)) from exc

    def execute(self, amount: int) -> ToolResult:
        try:
            validate_scroll_amount(amount)
            self.backend.scroll(amount)
        except ComputerControlError as exc:
            return ToolResult.error(str(exc))

        return ToolResult.success(message=f"Scrolled by {amount}.")
