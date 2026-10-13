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

COMPUTER INPUT GATE (app/computer/gate.py): the nine ACTING tools here - type_text, press_key, move_mouse, click_mouse, double_click, scroll,
open_application, open_url and switch_window - are still SAFE_ACTION (so ToolRegistry asks for no confirmation), but the real backend enforces the Input Gate ITSELF: the switch
is OFF by default, an authorization grant that covers the exact action is required, and one rate limit is shared by all nine (and by every backend instance in the process).
`open_url` is bound to the exact URL; `switch_window` to the exact window title AND the exact process name (a grant does not identify one window: two windows with that title
and process are an error). Registering a tool, here or in the Job allowlist, grants nothing: the shared backend carries no grant, so chat and Jobs are refused at execution. The gate enforces at EXECUTION time, inside the
backend, immediately before the OS call, so a direct `execute`, a registry call or a chained command cannot skip it, and neither can a backend built
somewhere the factory never saw. Each tool's `validate_arguments` additionally calls `_preflight_input`, a
non-consuming pre-check, so a whole command is turned away before its first step runs; that is a convenience, not the enforcement. All of them are also
`automatable = False`: until now they were kept away from automations only by living in a separate registry.
"""

from collections.abc import Callable
from typing import Any

from app.computer.base import (
    AllowedApplication,
    ComputerBackend,
    ComputerControlError,
    MouseButton,
    normalize_key_input,
    validate_coordinates,
    validate_exact_window_title,
    validate_process_name,
    validate_scroll_amount,
    validate_text,
    validate_url,
)
from app.tools.base import PermissionLevel, Tool, ToolArgumentError, ToolParameter, ToolResult


def _preflight_input(backend: ComputerBackend, action: str) -> None:
    """The Input Gate's NON-CONSUMING pre-check, for `validate_arguments`: turn a command away before any of its steps run when computer input is
    switched off or the limit is already reached. It records nothing and is never the enforcement (that is the backend's own `admit`, at execution
    time). A backend with no gate (a test double) has nothing to pre-check."""
    check = getattr(backend, "check_input", None)
    if check is None:
        return
    try:
        check(action)
    except ComputerControlError as exc:
        raise ToolArgumentError(str(exc)) from exc


class GetActiveWindowTool(Tool):
    """Reads the current foreground window's title/process/maximized
    state. READ: no side effects."""

    name = "get_active_window"
    description = "Get the title, owning process, and maximized state of the active window."
    permission = PermissionLevel.READ
    automatable = False  # Slice 4: Jobs may use this tool; a scheduled automation never does

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
    automatable = False  # Computer Input Gate: an input tool is never run by a schedule or an automation (it used to be kept away only by living in another registry)

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
        _preflight_input(self.backend, "open_application")
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
    restricted to http/https URLs only (see `validate_url`). It needs an authorization grant for that EXACT URL (see the module docstring): registering this tool grants nothing."""

    name = "open_url"
    description = "Open a URL (http/https only) in the default browser."
    permission = PermissionLevel.SAFE_ACTION
    automatable = False  # Slice 4: Jobs may use this tool; a scheduled automation never does

    def __init__(self, backend: ComputerBackend):
        self.backend = backend

    @property
    def parameters(self) -> list[ToolParameter]:
        return [ToolParameter(name="url", type="string", description="The http/https URL to open")]

    def validate_arguments(self, url: str) -> None:
        _preflight_input(self.backend, "open_url")  # non-consuming: turns a whole command away before its first step when nothing authorizes it
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


class JobOpenUrlTool(OpenUrlTool):
    """`open_url` as a Job (and so a page-influenced plan) may use it: the same tool, name and permission, plus the
    PUBLIC-HOST guard the page observer already applies to every fetch (app/observation/page.py) - a URL that is
    not a public http(s) host (localhost, a private / link-local address, credentials in the URL, an odd port) is
    refused, so a link found on an untrusted page can never steer the browser at the local network. Interactive
    chat keeps the plain OpenUrlTool. Redirects happen inside the browser and are not re-checked here."""

    automatable = False
    description = (
        "Open a public http/https URL in the user's default browser. Only open a URL the user gave or that an "
        "observation returned - never guess or invent one. To go to a named section of a page, prefer the same-page "
        "link's href from observe_page (it keeps its #anchor, e.g. https://site/#workflow) over scrolling. A #anchor is "
        "refused unless observe_page already exposed that exact anchor for that page (see its `anchors`): never invent "
        "one from a label. Localhost, private addresses and URLs with credentials are refused. Verify the result "
        "afterwards (verify_observation with window_title_contains)."
    )

    def __init__(
        self,
        backend: ComputerBackend,
        resolver: Callable[[str], list[str]] | None = None,
        anchors: Callable[[str], tuple[str, ...] | None] | None = None,
    ):
        super().__init__(backend)
        self._resolver = resolver
        #: `anchors(url)` -> the section anchors an observation of that page exposed, or None if it was never observed
        #: (or its entry expired). Without one, no #fragment can be opened at all.
        self._anchors = anchors

    def _check_public(self, url: str) -> None:
        from urllib.parse import urlsplit

        from app.observation.base import ObservationError, safe_anchor
        from app.observation.page import _resolve, validate_fetch_url

        try:
            validate_fetch_url(url, self._resolver or _resolve)
        except ObservationError as exc:
            raise ComputerControlError(f"open_url refused: {exc}") from exc
        fragment = urlsplit(url).fragment
        if not fragment:
            return
        if safe_anchor(fragment) != fragment:
            raise ComputerControlError(
                "open_url refused: a #fragment may only be a simple section anchor (letters, digits, '-' and '_')."
            )
        observed = self._anchors(url) if self._anchors is not None else None
        if observed is None:
            raise ComputerControlError(
                f"open_url refused: '#{fragment}' cannot be opened because that page has not been observed (or the "
                "observation expired). Use observe_page on it first, or open the page without a #fragment."
            )
        if fragment not in observed:
            listed = ", ".join(observed) if observed else "none"
            raise ComputerControlError(
                f"open_url refused: '#{fragment}' was not observed on that page. Anchors it exposed: {listed}. Use one "
                "of those, or open the page without a #fragment."
            )

    def validate_arguments(self, url: str) -> None:
        _preflight_input(self.backend, "open_url")
        try:
            validate_url(url)
            self._check_public(url)
        except ComputerControlError as exc:
            raise ToolArgumentError(str(exc)) from exc

    def execute(self, url: str) -> ToolResult:
        try:
            validate_url(url)
            self._check_public(url)
        except ComputerControlError as exc:
            return ToolResult.error(str(exc))
        return super().execute(url)


class SwitchWindowTool(Tool):
    """Brings an existing, visible, local window to the front. SAFE_ACTION, but it needs an authorization grant for the EXACT window title AND the EXACT process name (see the
    module docstring). The title must equal the window's full title exactly and the process must be its exact file name: no fragment, no wildcard. It is bounded
    (`validate_exact_window_title`, `validate_process_name`), never a credential / shell / system window (`is_switchable`, enforced by the backend) and never launches or
    closes anything. Zero matching windows, or more than one, is an error; it never chooses. A grant does not identify one window INSTANCE. Success is only reported once the
    backend confirmed that window is active. The title shown back is scrubbed: a window title is untrusted text."""

    name = "switch_window"
    description = (
        "Switch to an already open, visible window whose FULL title equals the given title exactly (case-sensitive, e.g. 'Example Domain - Browser') and whose process file name "
        "is the given process (e.g. 'chrome.exe'); take both from get_active_window or an observation, never guess them. It needs an authorization grant for that exact "
        "title and process. It never opens, closes or starts anything, refuses password-manager, sign-in, shell and system windows, and fails if no window - or more than "
        "one - matches. Verify it afterwards with verify_observation (window_title_contains / window_process_is)."
    )
    permission = PermissionLevel.SAFE_ACTION
    automatable = False

    def __init__(self, backend: ComputerBackend):
        self.backend = backend

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(name="title", type="string", description="The target window's exact, full title", free_text=True),
            ToolParameter(name="process", type="string", description="The target window's exact process file name, e.g. notepad.exe"),
        ]

    def validate_arguments(self, title: str, process: str) -> None:
        _preflight_input(self.backend, "switch_window")
        try:
            validate_exact_window_title(title)
            validate_process_name(process)
        except ComputerControlError as exc:
            raise ToolArgumentError(str(exc)) from exc

    def execute(self, title: str, process: str) -> ToolResult:
        from app.observation.base import scrub

        try:
            exact_title, exact_process = validate_exact_window_title(title), validate_process_name(process)
            window = self.backend.switch_window(exact_title, exact_process)
        except ComputerControlError as exc:
            return ToolResult.error(str(exc))
        shown = {"title": scrub(window.title, 120), "process_name": scrub(window.process_name, 60)}
        return ToolResult.success(data=shown, message=f"Switched to '{shown['title']}' ({shown['process_name']}).")


class TypeTextTool(Tool):
    """Types text via the keyboard into whatever currently has focus.
    SAFE_ACTION: bounded length (see `validate_text`/`MAX_TEXT_LENGTH`)."""

    name = "type_text"
    description = "Type text via the keyboard into whatever currently has focus."
    permission = PermissionLevel.SAFE_ACTION
    automatable = False  # Computer Input Gate: an input tool is never run by a schedule or an automation (it used to be kept away only by living in another registry)

    def __init__(self, backend: ComputerBackend):
        self.backend = backend

    @property
    def parameters(self) -> list[ToolParameter]:
        return [ToolParameter(
                name="text", type="string", description="The text to type", free_text=True
            )]

    def validate_arguments(self, text: str) -> None:
        _preflight_input(self.backend, "type_text")
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
    automatable = False  # Computer Input Gate: an input tool is never run by a schedule or an automation (it used to be kept away only by living in another registry)

    def __init__(self, backend: ComputerBackend):
        self.backend = backend

    @property
    def parameters(self) -> list[ToolParameter]:
        return [ToolParameter(name="key", type="string", description="The key to press")]

    def validate_arguments(self, key: str) -> None:
        _preflight_input(self.backend, "press_key")
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
    automatable = False  # Computer Input Gate: an input tool is never run by a schedule or an automation (it used to be kept away only by living in another registry)

    def __init__(self, backend: ComputerBackend):
        self.backend = backend

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(name="x", type="number", description="Target x coordinate, in pixels"),
            ToolParameter(name="y", type="number", description="Target y coordinate, in pixels"),
        ]

    def validate_arguments(self, x: int, y: int) -> None:
        _preflight_input(self.backend, "move_mouse")
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
    automatable = False  # Computer Input Gate: an input tool is never run by a schedule or an automation (it used to be kept away only by living in another registry)

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
        _preflight_input(self.backend, "click_mouse")
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
    automatable = False  # Computer Input Gate: an input tool is never run by a schedule or an automation (it used to be kept away only by living in another registry)

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
        _preflight_input(self.backend, "double_click")
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
    description = (
        "Scroll up (positive) or down (negative) at the current cursor position, by at most 2000 per call (use "
        "several steps for more). To reach a named section prefer open_url with that section's #anchor link from "
        "observe_page: scrolling is blind. It does not say what is now visible: check it afterwards (observe_screen "
        "before, then verify_observation check=screen_changed)."
    )
    permission = PermissionLevel.SAFE_ACTION
    automatable = False  # Slice 4: Jobs may use this tool; a scheduled automation never does

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
        _preflight_input(self.backend, "scroll")  # scrolling is input (mouse wheel): turned away early, never enforced here
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
