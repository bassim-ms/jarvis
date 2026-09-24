"""
Builds the application's default ContextPlanner - the one place
construction happens, mirroring app/context_reasoning/factory.py's own
role for ContextReasoner. Takes the AIProvider to use rather than
constructing one itself, so the caller (app/main.py, attaching to the
already-built `core` singleton - see app/core.py) decides which
provider/provider-manager it's routed through.
"""

from app.context_planning.planner import ContextPlanner, LLMContextPlanner
from app.providers.base import AIProvider


def create_default_context_planner(provider: AIProvider) -> ContextPlanner:
    return LLMContextPlanner(provider)
