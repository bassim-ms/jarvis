"""
Tests for JarvisCore's M16.4 integration point (app/core.py):
`context_planner` is an optional, additive constructor parameter, and
`handle_chat_with_planning` is the one new method that exposes the
PlanningDirective `_dispatch` builds. Crucially, `_plan_about_context`
must only ever be reached for planning/action-oriented requests
(PLANNING/TASK_EXECUTION/DECISION_SUPPORT intents) - never for ordinary
informational DIRECT requests (PROJECT_FOCUS/STATUS/SCHEDULING/GENERAL),
even though those still get M16.3 reasoning enrichment. Mirrors
tests/test_core_context_reasoning_capability_gating.py's own discipline
one milestone up.
"""

import json

from app.capability import Capability, CapabilityDecision, CapabilityRouter
from app.context_engine.engine import ContextEngine
from app.context_planning.planner import LLMContextPlanner
from app.context_reasoning.reasoner import LLMContextReasoner
from app.core import JarvisCore
from app.orchestration.sqlite_store import SQLiteJobStore
from app.orchestration.tools import CreateJobTool
from app.providers.base import AIProvider, ProviderError
from app.tools.registry import ToolRegistry
from app.world_model.service import WorldModelService
from app.world_model.sqlite_store import SQLiteWorldModelStore


class _RecordingProvider(AIProvider):
    def __init__(self, reply: str = "fake reply"):
        self.reply = reply
        self.prompts: list[str] = []

    def generate_reply(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.reply


class _CannedProvider(AIProvider):
    """A fake AIProvider that always returns the same canned JSON (or
    raises) - used independently for the reasoner's and the planner's
    own provider, so a test can tell exactly which one was called."""

    def __init__(self, payload: dict | None = None, error: Exception | None = None):
        self.payload = payload
        self.error = error
        self.calls = 0

    def generate_reply(self, prompt: str) -> str:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return json.dumps(self.payload)


class _FixedCapabilityRouter(CapabilityRouter):
    """Always returns a fixed CapabilityDecision, ignoring the message -
    isolates "which capability" from "what ContextIntent the message's
    own wording implies," the same test double
    tests/test_core_context_reasoning_capability_gating.py already uses."""

    def __init__(self, capability: Capability):
        self.decision = CapabilityDecision(capability=capability)

    def route(self, message: str) -> CapabilityDecision:
        return self.decision


def _valid_reasoning_payload(goal_id: str) -> dict:
    return {
        "interpretation": "Reasoning about the request.",
        "relevant_facts": [
            {"text": "The robotics arm goal is active.", "provenance": [f"goal:{goal_id}"]}
        ],
        "constraints": [],
        "conflicts": [],
        "opportunities": [],
        "conclusions": [],
        "recommended_next_step": None,
        "confidence": 1.0,
        "requires_clarification": False,
        "clarification_question": None,
    }


def _valid_planning_payload(goal_id: str, **overrides) -> dict:
    payload = {
        "objective": "Finish the robotics arm project.",
        "planning_type": "ACTION",
        "contextual_constraints": [],
        "relevant_resources": [
            {"text": "The robotics arm goal.", "provenance": [f"goal:{goal_id}"]}
        ],
        "dependencies": [],
        "assumptions": [],
        "suggested_approach": "Work on the remaining steps.",
        "requires_execution": True,
        "confidence": 0.9,
    }
    payload.update(overrides)
    return payload


def _world_model_with_project() -> WorldModelService:
    """A project WITH a goal under it - a goal is in every one of the
    three "planning-oriented" intents' required_context (PLANNING/
    TASK_EXECUTION/DECISION_SUPPORT all include RequiredContextCategory.
    GOALS - app/context_engine/base.py), unlike PROJECTS (which
    TASK_EXECUTION deliberately excludes), so grounding test payloads
    against the goal's id works uniformly across all three."""
    world_model = WorldModelService(SQLiteWorldModelStore(":memory:"))
    project = world_model.create_project("Robotics Arm")
    world_model.create_goal(project.id, "Finish robotics arm")
    return world_model


def _active_goal_id(world_model: WorldModelService) -> str:
    return world_model.get_active_goals()[0].id


def _make_core(
    *,
    world_model: WorldModelService | None = None,
    with_context_reasoner: bool = True,
    with_context_planner: bool = True,
    planner_error: Exception | None = None,
    reasoner_error: Exception | None = None,
    tools: ToolRegistry | None = None,
    capability: Capability = Capability.DIRECT,
) -> tuple[JarvisCore, _RecordingProvider, _CannedProvider, _CannedProvider]:
    world_model = world_model or _world_model_with_project()
    goal_id = _active_goal_id(world_model)
    context_engine = ContextEngine(world_model)

    reasoner_provider = _CannedProvider(
        payload=_valid_reasoning_payload(goal_id), error=reasoner_error
    )
    context_reasoner = LLMContextReasoner(reasoner_provider) if with_context_reasoner else None

    planner_provider = _CannedProvider(
        payload=_valid_planning_payload(goal_id), error=planner_error
    )
    context_planner = LLMContextPlanner(planner_provider) if with_context_planner else None

    reply_provider = _RecordingProvider()
    core = JarvisCore(
        provider=reply_provider,
        tools=tools or ToolRegistry(),
        capability_router=_FixedCapabilityRouter(capability),
        context_engine=context_engine,
        context_reasoner=context_reasoner,
        context_planner=context_planner,
    )
    return core, reply_provider, reasoner_provider, planner_provider


# -- defaults / additive presence --------------------------------------------


def test_context_planner_defaults_to_none():
    core = JarvisCore(provider=_RecordingProvider())
    assert core.context_planner is None


def test_handle_chat_with_planning_returns_none_when_no_context_planner_configured():
    core, _reply, _reasoner, planner_provider = _make_core(with_context_planner=False)

    result = core.handle_chat_with_planning("finish my robotics arm")

    assert len(result) == 7
    assert result[6] is None
    assert planner_provider.calls == 0


def test_handle_chat_with_planning_returns_none_when_no_reasoning_result_exists():
    """A PlanningDirective needs a ReasoningResult to consume - with no
    context_reasoner (even with a context_planner configured), planning
    never even asks."""
    core, _reply, _reasoner, planner_provider = _make_core(with_context_reasoner=False)

    result = core.handle_chat_with_planning("finish my robotics arm")

    assert result[6] is None
    assert planner_provider.calls == 0


# -- intent gating: planning/action-oriented invokes the planner ------------


def test_task_execution_intent_invokes_the_planner():
    core, _reply, _reasoner, planner_provider = _make_core()

    reply, conv_id, sources, job_id, snapshot, reasoning, plan = core.handle_chat_with_planning(
        "finish my robotics arm"
    )

    assert planner_provider.calls == 1
    assert plan is not None
    assert plan.planning_type.value == "action"


def test_decision_support_intent_invokes_the_planner():
    core, _reply, _reasoner, planner_provider = _make_core()

    result = core.handle_chat_with_planning("should I work on robotics or Luniti tonight?")

    assert planner_provider.calls == 1
    assert result[6] is not None


def test_planning_intent_invokes_the_planner():
    core, _reply, _reasoner, planner_provider = _make_core()

    result = core.handle_chat_with_planning("what should I work on next")

    assert planner_provider.calls == 1
    assert result[6] is not None


# -- intent gating: informational requests never reach M16.4 ----------------


def test_project_focus_intent_does_not_invoke_the_planner():
    core, reply_provider, reasoner_provider, planner_provider = _make_core()

    reply, conv_id, sources, job_id, snapshot, reasoning, plan = core.handle_chat_with_planning(
        "what's happening with my robotics project?"
    )

    assert reasoner_provider.calls == 1  # M16.3 still reasons about this
    assert planner_provider.calls == 0  # M16.4 never does
    assert plan is None
    assert len(reply_provider.prompts) == 1


def test_status_intent_does_not_invoke_the_planner():
    core, _reply, reasoner_provider, planner_provider = _make_core()

    result = core.handle_chat_with_planning("what's the status of my project?")

    assert reasoner_provider.calls == 1
    assert planner_provider.calls == 0
    assert result[6] is None


def test_scheduling_intent_does_not_invoke_the_planner():
    core, _reply, _reasoner, planner_provider = _make_core()

    result = core.handle_chat_with_planning("what's on my calendar today?")

    assert planner_provider.calls == 0
    assert result[6] is None


def test_general_intent_does_not_invoke_the_planner():
    core, _reply, _reasoner, planner_provider = _make_core()

    result = core.handle_chat_with_planning("hello there")

    assert planner_provider.calls == 0
    assert result[6] is None


# -- provider/output failures degrade safely ---------------------------------


def test_planner_provider_error_still_produces_an_ordinary_reply():
    core, reply_provider, _reasoner, _planner = _make_core(
        planner_error=ProviderError("boom")
    )

    reply, conv_id, sources, job_id, snapshot, reasoning, plan = core.handle_chat_with_planning(
        "finish my robotics arm"
    )

    assert reply == "fake reply"
    assert plan is None
    assert len(reply_provider.prompts) == 1


def test_planner_malformed_output_still_produces_an_ordinary_reply():
    world_model = _world_model_with_project()
    context_engine = ContextEngine(world_model)
    goal_id = _active_goal_id(world_model)
    reasoner_provider = _CannedProvider(payload=_valid_reasoning_payload(goal_id))
    planner_provider = _CannedProvider(payload=None)
    planner_provider.generate_reply = lambda prompt: "not json"  # type: ignore[method-assign]
    reply_provider = _RecordingProvider()

    core = JarvisCore(
        provider=reply_provider,
        capability_router=_FixedCapabilityRouter(Capability.DIRECT),
        context_engine=context_engine,
        context_reasoner=LLMContextReasoner(reasoner_provider),
        context_planner=LLMContextPlanner(planner_provider),
    )

    reply, conv_id, sources, job_id, snapshot, reasoning, plan = core.handle_chat_with_planning(
        "finish my robotics arm"
    )

    assert reply == "fake reply"
    assert plan is None


# -- planning directive reaches the prompt -----------------------------------


def test_planning_directive_reaches_the_direct_prompt():
    core, reply_provider, _reasoner, _planner = _make_core()

    core.handle_chat_with_planning("finish my robotics arm")

    assert len(reply_provider.prompts) == 1
    prompt = reply_provider.prompts[0]
    assert "What Jarvis's planning about this request concluded" in prompt
    assert "Work on the remaining steps." in prompt


# -- read-only / no execution / no duplicate Job creation --------------------


def test_context_planner_never_mutates_the_world_model():
    world_model = _world_model_with_project()
    project = world_model.get_active_projects()[0]
    before = world_model.get_project(project.id)
    core, _reply, _reasoner, _planner = _make_core(world_model=world_model)

    core.handle_chat_with_planning("finish my robotics arm")

    after = world_model.get_project(project.id)
    assert after == before


def test_context_planner_never_calls_tool_registry_execute():
    tools = ToolRegistry()
    calls: list[str] = []
    original_execute = tools.execute

    def _recording_execute(*args, **kwargs):
        calls.append("execute_called")
        return original_execute(*args, **kwargs)

    tools.execute = _recording_execute  # type: ignore[method-assign]

    core, _reply, _reasoner, planner_provider = _make_core(tools=tools)

    core.handle_chat_with_planning("finish my robotics arm")

    assert calls == []
    assert planner_provider.calls == 1  # planning still ran - just never via ToolRegistry


def test_context_planner_never_creates_a_duplicate_job():
    """Even with `requires_execution=True` in the canned planning
    payload, M16.4 must never itself create a Job - only the existing,
    separate TASK-capability path (`_handle_task` -> `create_job`) does
    that, and this test forces DIRECT, never TASK."""
    tools = ToolRegistry()
    job_store = SQLiteJobStore(":memory:")
    tools.register(CreateJobTool(store=job_store))

    core, _reply, _reasoner, planner_provider = _make_core(tools=tools)

    reply, conv_id, sources, job_id, snapshot, reasoning, plan = core.handle_chat_with_planning(
        "finish my robotics arm"
    )

    assert plan.requires_execution is True
    assert job_id is None
    assert job_store.list() == []


# -- existing handle_chat*/behavior remains correct --------------------------


def test_handle_chat_contract_is_unaffected_by_a_configured_context_planner():
    core_without, _r1, _x1, _p1 = _make_core(with_context_planner=False)
    core_with, _r2, _x2, _p2 = _make_core()

    reply_a, conv_a, sources_a = core_without.handle_chat("finish my robotics arm")
    reply_b, conv_b, sources_b = core_with.handle_chat("finish my robotics arm")

    assert reply_a == reply_b == "fake reply"
    assert sources_a == sources_b == []


def test_handle_chat_with_reasoning_contract_unaffected_by_a_configured_context_planner():
    core_without, _r1, _x1, _p1 = _make_core(with_context_planner=False)
    core_with, _r2, _x2, _p2 = _make_core()

    result_without = core_without.handle_chat_with_reasoning("finish my robotics arm")
    result_with = core_with.handle_chat_with_reasoning("finish my robotics arm")

    assert len(result_without) == len(result_with) == 6
    assert result_without[0] == result_with[0] == "fake reply"


def test_handle_chat_with_planning_returns_a_7_tuple():
    core, _reply, _reasoner, _planner = _make_core()
    result = core.handle_chat_with_planning("finish my robotics arm")
    assert len(result) == 7
