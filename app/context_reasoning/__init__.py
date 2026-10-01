"""
Context-Aware Reasoning (M16.3): a bounded, read-only reasoning layer
between a ContextSnapshot (M16.2, app/context_engine/) and Jarvis's
existing response generation (app/core.py).

See app/context_reasoning/base.py for the ReasoningResult contract,
app/context_reasoning/deterministic.py for the bounded deterministic
analysis pass, app/context_reasoning/reasoner.py for the ContextReasoner
abstraction/implementation, and app/context_reasoning/factory.py for how
the application builds its default one.
"""
