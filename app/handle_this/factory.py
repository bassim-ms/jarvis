"""
Builds a HandleThisIntake from the application's existing pieces - the one
place construction happens, same role every other app/*/factory.py plays.
Nothing here is wired into chat yet: exposing the intake is a later slice.
"""

from __future__ import annotations

from app.context_engine.engine import ContextEngine
from app.handle_this.context import ContextEngineSource
from app.handle_this.intake import HandleThisIntake
from app.handle_this.interpretation import LLMGoalInterpreter
from app.providers.base import AIProvider
from app.tools.registry import ToolRegistry


def planner_tool_names(registry: ToolRegistry) -> list[str]:
    """The names of the tools a Planner would be OFFERED from `registry` (its
    `plannable` tools) - the same set JobEngine offers, and so the honest
    answer to "what could Handle This's objectives actually run on?". Names
    only: this reads nothing about permissions and executes nothing."""
    return [tool.name for tool in registry.list_tools() if tool.plannable]


def create_default_intake(
    *,
    provider: AIProvider | None = None,
    context_engine: ContextEngine | None = None,
    tools: ToolRegistry | None = None,
) -> HandleThisIntake:
    """`provider` (normally `core.provider`, the AIProviderManager with its
    ordinary fallback) enables the AI interpreter for wording the
    deterministic pass cannot place; `context_engine` enables the context
    handoff; `tools` should be the registry the JobEngine plans against (NOT
    the isolated computer-control registry), so unavailable families are
    reported honestly."""
    return HandleThisIntake(
        llm_interpreter=LLMGoalInterpreter(provider) if provider is not None else None,
        context_source=ContextEngineSource(context_engine) if context_engine is not None else None,
        available_tools=(lambda: planner_tool_names(tools)) if tools is not None else None,
    )
