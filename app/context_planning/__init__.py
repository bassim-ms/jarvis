"""
Context-Aware Planning (M16.4): a bounded, read-only, NON-EXECUTING
layer that turns a ReasoningResult (M16.3, app/context_reasoning/) into
a strategic/contextual PlanningDirective - never a competing executable
plan. M15 (app/orchestration/) remains the sole authoritative executable
planning/execution system: its own Plan/PlanStep/Job/JobEngine are
untouched and unduplicated by anything here.

See app/context_planning/base.py for the PlanningDirective contract,
app/context_planning/planner.py for the ContextPlanner abstraction/
implementation, and app/context_planning/factory.py for how the
application builds its default one.
"""
