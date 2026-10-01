"""
Default observation wiring: build the engine, and say exactly which registry gets which observation tool.

`register_observation_tools` registers the three READ tools onto the JOB-facing registry (`core.tools`): observe a page,
verify a state, and take one bounded in-memory screen fingerprint. It adds no permission rule, loop or executor - the
tools run through ToolRegistry like every other tool.
"""

from __future__ import annotations

from app.computer.base import ComputerBackend
from app.computer.factory import create_default_computer_backend
from app.observation.engine import ObservationEngine
from app.observation.page import HttpPageFetcher
from app.observation.screen import PyAutoGuiScreenCapturer, VisionInterpreter
from app.observation.tools import ObservePageTool, ObserveScreenTool, VerifyObservationTool
from app.tools.registry import ToolRegistry

#: The observation tools a Job plan may use.
JOB_OBSERVATION_TOOL_NAMES = frozenset({"observe_page", "verify_observation", "observe_screen"})


def create_default_observation_engine(
    backend: ComputerBackend | None = None, *, vision: VisionInterpreter | None = None
) -> ObservationEngine:
    """The real engine: guarded HTTP page fetch, the shared computer backend (read-only use), pyautogui capture.
    `vision` stays None until a vision interpreter is deliberately wired (none is, in this slice)."""
    return ObservationEngine(
        fetcher=HttpPageFetcher(),
        backend=backend or create_default_computer_backend(),
        capturer=PyAutoGuiScreenCapturer(),
        vision=vision,
    )


def register_observation_tools(registry: ToolRegistry, engine: ObservationEngine) -> None:
    registry.register(ObservePageTool(engine))
    registry.register(VerifyObservationTool(engine))
    registry.register(ObserveScreenTool(engine))
