"""
Default observation wiring: build the engine, and say exactly which registry gets which observation tool.

`register_observation_tools` is for the JOB-facing registry (`core.tools`): the two READ tools a plan may use to
observe and verify. `register_interactive_observation_tools` is for the isolated interactive computer registry:
screen capture is never offered to a Job. Neither adds a permission rule, a loop or an executor - the tools run
through ToolRegistry like every other tool.
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
JOB_OBSERVATION_TOOL_NAMES = frozenset({"observe_page", "verify_observation"})


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


def register_interactive_observation_tools(registry: ToolRegistry, engine: ObservationEngine) -> None:
    registry.register(ObserveScreenTool(engine))
