"""
Tests for app/context_planning/planner.py: LLMContextPlanner - strict
parsing/validation of AI output into a PlanningDirective, the
empty-context no-call short circuit, and safe failure behavior.

Uses a plain fake AIProvider (never a real network call) - `plan()`
calls `call_with_requirements`, which treats a plain AIProvider as a
no-op pass-through and calls `generate_reply(prompt)` directly, so
these tests see the exact prompt built and can hand back whatever
canned JSON they need.
"""

import json

import pytest

from app.context_engine.base import RequestContext
from app.context_planning.planner import ContextPlanningError, LLMContextPlanner
from app.context_reasoning.base import ReasoningResult, RelevantFact
from app.providers.base import AIProvider, ProviderError
from app.world_model.base import ContextSnapshot, Priority, Project, ProjectStatus

_NOW = "2026-01-01T12:00:00+00:00"


class _FakeProvider(AIProvider):
    def __init__(self, reply: str | None = None, error: Exception | None = None):
        self.reply = reply
        self.error = error
        self.calls = 0
        self.last_prompt: str | None = None

    def generate_reply(self, message: str) -> str:
        self.calls += 1
        self.last_prompt = message
        if self.error is not None:
            raise self.error
        return self.reply


def _project(id_="project-1") -> Project:
    return Project(
        id=id_,
        name="Robotics Arm",
        description="",
        status=ProjectStatus.ACTIVE,
        priority=Priority.MEDIUM,
        created_at=_NOW,
        updated_at=_NOW,
    )


def _non_empty_snapshot() -> ContextSnapshot:
    return ContextSnapshot(current_time=_NOW, active_projects=[_project()])


def _rc(message: str = "finish my robotics arm") -> RequestContext:
    return RequestContext(raw_request=message)


def _non_empty_reasoning(project_id: str = "project-1") -> ReasoningResult:
    return ReasoningResult(
        interpretation="The user wants to finish the robotics arm project.",
        relevant_facts=(
            RelevantFact(text="The robotics arm project is active.", provenance=(f"project:{project_id}",)),
        ),
    )


def _valid_payload(project_id: str = "project-1", **overrides) -> dict:
    payload = {
        "objective": "Finish the robotics arm project.",
        "planning_type": "ACTION",
        "contextual_constraints": [],
        "relevant_resources": [
            {"text": "The robotics arm project itself.", "provenance": [f"project:{project_id}"]}
        ],
        "dependencies": [],
        "assumptions": [],
        "suggested_approach": "Work on the remaining assembly steps.",
        "requires_execution": True,
        "confidence": 0.9,
    }
    payload.update(overrides)
    return payload


# -- empty-context short circuit ---------------------------------------------


def test_empty_snapshot_and_reasoning_never_calls_the_provider():
    provider = _FakeProvider(reply="should never be used")
    planner = LLMContextPlanner(provider)

    result = planner.plan(
        "hello there",
        _rc("hello there"),
        ContextSnapshot(current_time=_NOW),
        ReasoningResult(interpretation="Nothing relevant."),
    )

    assert provider.calls == 0
    assert result.is_empty()


def test_non_empty_snapshot_with_empty_reasoning_still_calls_the_provider():
    """The snapshot alone being non-empty is enough to warrant asking -
    the short circuit requires BOTH to be empty."""
    provider = _FakeProvider(reply=json.dumps(_valid_payload()))
    planner = LLMContextPlanner(provider)

    planner.plan(
        "finish my robotics arm",
        _rc(),
        _non_empty_snapshot(),
        ReasoningResult(interpretation="Nothing relevant."),
    )

    assert provider.calls == 1


# -- valid structured response: decision / action / multi-step --------------


def test_valid_action_response_produces_a_matching_directive():
    provider = _FakeProvider(reply=json.dumps(_valid_payload()))
    planner = LLMContextPlanner(provider)

    directive = planner.plan(
        "finish my robotics arm", _rc(), _non_empty_snapshot(), _non_empty_reasoning()
    )

    assert directive.planning_type.value == "action"
    assert directive.requires_execution is True
    assert directive.confidence == 0.9
    assert directive.relevant_resources[0].provenance == ("project:project-1",)


def test_valid_decision_response_never_requires_execution():
    payload = _valid_payload(
        objective="Decide between robotics and Luniti tonight.",
        planning_type="DECISION",
        requires_execution=False,
        suggested_approach=None,
    )
    provider = _FakeProvider(reply=json.dumps(payload))
    planner = LLMContextPlanner(provider)

    directive = planner.plan(
        "should I work on robotics or Luniti tonight?",
        _rc(),
        _non_empty_snapshot(),
        _non_empty_reasoning(),
    )

    assert directive.planning_type.value == "decision"
    assert directive.requires_execution is False


def test_valid_multi_step_response_with_contextual_constraints_and_dependencies():
    payload = _valid_payload(
        planning_type="MULTI_STEP",
        contextual_constraints=[
            {"text": "A commitment is due Friday.", "provenance": ["project:project-1"]}
        ],
        dependencies=[
            {"text": "Needs the gripper part first.", "provenance": ["project:project-1"]}
        ],
    )
    provider = _FakeProvider(reply=json.dumps(payload))
    planner = LLMContextPlanner(provider)

    directive = planner.plan(
        "finish my robotics arm around my other commitments",
        _rc(),
        _non_empty_snapshot(),
        _non_empty_reasoning(),
    )

    assert directive.planning_type.value == "multi_step"
    assert directive.contextual_constraints[0].text == "A commitment is due Friday."
    assert directive.dependencies[0].text == "Needs the gripper part first."


def test_response_wrapped_in_markdown_code_fence_is_tolerated():
    fenced = "```json\n" + json.dumps(_valid_payload()) + "\n```"
    provider = _FakeProvider(reply=fenced)
    planner = LLMContextPlanner(provider)

    directive = planner.plan("hello", _rc(), _non_empty_snapshot(), _non_empty_reasoning())
    assert directive.confidence == 0.9


# -- malformed response --------------------------------------------------------


def test_invalid_json_raises():
    provider = _FakeProvider(reply="not json at all")
    planner = LLMContextPlanner(provider)

    with pytest.raises(ContextPlanningError):
        planner.plan("hello", _rc(), _non_empty_snapshot(), _non_empty_reasoning())


def test_non_object_json_raises():
    provider = _FakeProvider(reply="[1, 2, 3]")
    planner = LLMContextPlanner(provider)

    with pytest.raises(ContextPlanningError):
        planner.plan("hello", _rc(), _non_empty_snapshot(), _non_empty_reasoning())


def test_unexpected_top_level_field_raises():
    payload = _valid_payload()
    payload["extra_field"] = "not allowed"
    provider = _FakeProvider(reply=json.dumps(payload))
    planner = LLMContextPlanner(provider)

    with pytest.raises(ContextPlanningError):
        planner.plan("hello", _rc(), _non_empty_snapshot(), _non_empty_reasoning())


# -- missing fields -------------------------------------------------------


def test_missing_objective_raises():
    payload = _valid_payload()
    del payload["objective"]
    provider = _FakeProvider(reply=json.dumps(payload))
    planner = LLMContextPlanner(provider)

    with pytest.raises(ContextPlanningError):
        planner.plan("hello", _rc(), _non_empty_snapshot(), _non_empty_reasoning())


def test_missing_requires_execution_raises():
    payload = _valid_payload()
    del payload["requires_execution"]
    provider = _FakeProvider(reply=json.dumps(payload))
    planner = LLMContextPlanner(provider)

    with pytest.raises(ContextPlanningError):
        planner.plan("hello", _rc(), _non_empty_snapshot(), _non_empty_reasoning())


def test_missing_planning_type_raises():
    payload = _valid_payload()
    del payload["planning_type"]
    provider = _FakeProvider(reply=json.dumps(payload))
    planner = LLMContextPlanner(provider)

    with pytest.raises(ContextPlanningError):
        planner.plan("hello", _rc(), _non_empty_snapshot(), _non_empty_reasoning())


# -- invalid values -------------------------------------------------------


def test_unknown_planning_type_raises():
    payload = _valid_payload(planning_type="URGENT")
    provider = _FakeProvider(reply=json.dumps(payload))
    planner = LLMContextPlanner(provider)

    with pytest.raises(ContextPlanningError):
        planner.plan("hello", _rc(), _non_empty_snapshot(), _non_empty_reasoning())


def test_confidence_above_one_raises():
    payload = _valid_payload(confidence=1.5)
    provider = _FakeProvider(reply=json.dumps(payload))
    planner = LLMContextPlanner(provider)

    with pytest.raises(ContextPlanningError):
        planner.plan("hello", _rc(), _non_empty_snapshot(), _non_empty_reasoning())


def test_confidence_below_zero_raises():
    payload = _valid_payload(confidence=-0.5)
    provider = _FakeProvider(reply=json.dumps(payload))
    planner = LLMContextPlanner(provider)

    with pytest.raises(ContextPlanningError):
        planner.plan("hello", _rc(), _non_empty_snapshot(), _non_empty_reasoning())


def test_confidence_boundaries_are_accepted():
    for boundary in (0.0, 1.0):
        payload = _valid_payload(confidence=boundary, requires_execution=False)
        provider = _FakeProvider(reply=json.dumps(payload))
        planner = LLMContextPlanner(provider)
        directive = planner.plan("hello", _rc(), _non_empty_snapshot(), _non_empty_reasoning())
        assert directive.confidence == boundary


# -- no impossible/contradictory requirements (coherence) --------------------


def test_none_planning_type_requiring_execution_raises():
    payload = _valid_payload(planning_type="NONE", requires_execution=True)
    provider = _FakeProvider(reply=json.dumps(payload))
    planner = LLMContextPlanner(provider)

    with pytest.raises(ContextPlanningError):
        planner.plan("hello", _rc(), _non_empty_snapshot(), _non_empty_reasoning())


def test_decision_planning_type_requiring_execution_raises():
    payload = _valid_payload(planning_type="DECISION", requires_execution=True)
    provider = _FakeProvider(reply=json.dumps(payload))
    planner = LLMContextPlanner(provider)

    with pytest.raises(ContextPlanningError):
        planner.plan("hello", _rc(), _non_empty_snapshot(), _non_empty_reasoning())


# -- unsupported/hallucinated references -------------------------------------


def test_provenance_citing_an_unknown_entity_id_raises():
    payload = _valid_payload(
        relevant_resources=[
            {"text": "A fabricated resource.", "provenance": ["resource:does-not-exist"]}
        ]
    )
    provider = _FakeProvider(reply=json.dumps(payload))
    planner = LLMContextPlanner(provider)

    with pytest.raises(ContextPlanningError):
        planner.plan("hello", _rc(), _non_empty_snapshot(), _non_empty_reasoning())


def test_dependency_with_empty_provenance_raises():
    payload = _valid_payload(
        dependencies=[{"text": "An ungrounded dependency.", "provenance": []}]
    )
    provider = _FakeProvider(reply=json.dumps(payload))
    planner = LLMContextPlanner(provider)

    with pytest.raises(ContextPlanningError):
        planner.plan("hello", _rc(), _non_empty_snapshot(), _non_empty_reasoning())


def test_constraint_citing_a_real_and_a_fake_id_raises():
    payload = _valid_payload(
        contextual_constraints=[
            {
                "text": "Partially grounded, partially invented.",
                "provenance": ["project:project-1", "goal:does-not-exist"],
            }
        ]
    )
    provider = _FakeProvider(reply=json.dumps(payload))
    planner = LLMContextPlanner(provider)

    with pytest.raises(ContextPlanningError):
        planner.plan("hello", _rc(), _non_empty_snapshot(), _non_empty_reasoning())


def test_known_entity_grounded_reference_is_accepted():
    """Sanity check for the hallucination tests above: citing the REAL
    project id that was actually supplied is accepted."""
    payload = _valid_payload(
        relevant_resources=[{"text": "The robotics arm project.", "provenance": ["project:project-1"]}]
    )
    provider = _FakeProvider(reply=json.dumps(payload))
    planner = LLMContextPlanner(provider)

    directive = planner.plan("hello", _rc(), _non_empty_snapshot(), _non_empty_reasoning())
    assert directive.relevant_resources[0].provenance == ("project:project-1",)


# -- provider failures ---------------------------------------------------


def test_provider_error_raises_context_planning_error():
    provider = _FakeProvider(error=ProviderError("boom"))
    planner = LLMContextPlanner(provider)

    with pytest.raises(ContextPlanningError):
        planner.plan("hello", _rc(), _non_empty_snapshot(), _non_empty_reasoning())


# -- prompt construction --------------------------------------------------


def test_prompt_sent_to_provider_includes_the_raw_request_reasoning_and_allowed_provenance():
    provider = _FakeProvider(reply=json.dumps(_valid_payload()))
    planner = LLMContextPlanner(provider)

    planner.plan(
        "finish my robotics arm", _rc(), _non_empty_snapshot(), _non_empty_reasoning()
    )

    assert "finish my robotics arm" in provider.last_prompt
    assert "project:project-1" in provider.last_prompt
    assert "The user wants to finish the robotics arm project." in provider.last_prompt
