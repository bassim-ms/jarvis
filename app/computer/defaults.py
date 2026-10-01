"""
Default computer-control tool registration (M17.1-B): the one place that
knows the concrete set of computer-control tools (M17.1-A,
app/computer/tools.py) and wires them onto a ToolRegistry - the same
"Tool depends only on the Registry/PermissionLevel machinery, never a
concrete implementation" pattern app/tools/defaults.py already
establishes for every other tool family (web_search, reminders,
automation, calendar, tasks, orchestration).

Deliberately its OWN function/module rather than an addition to
app/tools/defaults.py:register_default_tools - that function populates
the real running application's `core.tools` registry (see app/main.py),
which is also the EXACT registry app/main.py hands to JobEngine (M15):
JobEngine.plan calls `self.tools.list_tools()` to build a Job's planning
candidates (app/orchestration/engine.py), and AutomationEngine can run
any registered READ/SAFE_ACTION tool as an automation's action
(app/automation/engine.py). Adding computer tools to that shared
registry would silently make them selectable by M15 Job planning and
automations - which M17.1-B explicitly does not do (that decision, and
whatever gating it needs, belongs to a later milestone, not to an
accidental side effect of registering a tool here).

`register_computer_tools`/`create_computer_tools_registry` instead
populate whatever ToolRegistry a caller gives them (or a fresh one),
proving the computer tools integrate correctly with the SAME
capability/permission machinery - DeterministicCapabilityRouter deciding
TOOL, ToolCallBuilder constructing a validated ToolCall, and
ToolRegistry.execute enforcing PermissionLevel (READ/SAFE_ACTION always
run; nothing higher exists among these tools) - every other tool family
already goes through (see app/core.py's own module docstring). No
permission logic is duplicated here or in app/computer/: ToolRegistry.
execute remains the sole enforcement point, exactly as
app/computer/base.py's own module docstring already states.

This module is NOT called from app/main.py yet - wiring the running
application's own singleton (and deciding how to keep it out of M15's
reach there) is M17.1-C's job, not this one's.
"""

from app.computer.base import ComputerBackend
from app.computer.factory import create_default_computer_backend
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
    SwitchWindowTool,
    TypeTextTool,
    JobOpenUrlTool,
)
from app.tools.registry import ToolRegistry


def register_computer_tools(
    registry: ToolRegistry, backend: ComputerBackend | None = None
) -> None:
    """Register every M17.1-A computer-control tool onto `registry`, all
    sharing one backend instance. `backend` defaults to a real
    WindowsComputerBackend (app/computer/factory.py); callers that want a
    fake/mock backend (tests) pass one explicitly - the same "takes a
    registry/backend, never owns global state" convention
    register_default_tools already uses for its own stores/providers."""
    backend = backend or create_default_computer_backend()
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
    registry.register(SwitchWindowTool(backend=backend))


#: The ONLY computer tools a Job plan may use (Slice 4): opening a page, scrolling it, and reading which window is
#: active. Everything that types, presses keys, moves or clicks the mouse, or launches an application stays in the
#: isolated interactive registry. This is a plain, reviewable list - not a permission system; ToolRegistry still
#: decides whether and how each of these runs.
JOB_COMPUTER_TOOL_NAMES = frozenset({"open_url", "scroll", "get_active_window", "switch_window"})


def register_job_computer_tools(
    registry: ToolRegistry, backend: ComputerBackend | None = None, *, resolver=None, anchors=None
) -> None:
    """Register ONLY `JOB_COMPUTER_TOOL_NAMES` onto `registry` (the Job-facing one), sharing `backend`. The Job's
    `open_url` is the public-host-guarded variant (`resolver` is the DNS lookup it uses; tests inject one). `anchors` is
    the observed-anchor lookup (ObservationEngine.anchors.observed): the Job's `open_url` opens a #fragment only if
    that exact anchor was observed for that page."""
    backend = backend or create_default_computer_backend()
    registry.register(GetActiveWindowTool(backend=backend))
    registry.register(JobOpenUrlTool(backend=backend, resolver=resolver, anchors=anchors))
    registry.register(ScrollTool(backend=backend))
    registry.register(SwitchWindowTool(backend=backend))


def create_computer_tools_registry(backend: ComputerBackend | None = None) -> ToolRegistry:
    """Build a fresh ToolRegistry populated with only the computer-control
    tools - lets a caller (a test, or a future dedicated integration)
    exercise them through JarvisCore's TOOL capability without any of
    Jarvis's other default tools, AutomationEngine, or M15 JobEngine
    present."""
    registry = ToolRegistry()
    register_computer_tools(registry, backend=backend)
    return registry
