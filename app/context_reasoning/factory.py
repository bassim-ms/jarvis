"""
Builds the application's default ContextReasoner - the one place
construction happens, mirroring app/context_engine/factory.py's own role
for ContextEngine. Takes the AIProvider to use rather than constructing
one itself, so the caller (app/main.py, attaching to the already-built
`core` singleton - see app/core.py) decides which provider/provider-
manager it's routed through, exactly like create_default_context_engine
takes its WorldModelService rather than building one.
"""

from app.context_reasoning.reasoner import ContextReasoner, LLMContextReasoner
from app.providers.base import AIProvider


def create_default_context_reasoner(provider: AIProvider) -> ContextReasoner:
    return LLMContextReasoner(provider)
