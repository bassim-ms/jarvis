"""
Tests for app/context_planning/base.py: PlanningDirective and its nested
value types are pure, validated, serializable data - no I/O, no AI call.
"""

import pytest

from app.context_planning.base import (
    GroundedReference,
    PlanningDirective,
    PlanningDirectiveError,
    PlanningType,
    PlanningTypeError,
    parse_planning_type,
)


def _minimal_directive(**overrides) -> PlanningDirective:
    defaults = dict(objective="Decide what to do tonight.", planning_type=PlanningType.NONE)
    defaults.update(overrides)
    return PlanningDirective(**defaults)


# -- planning type selection / parsing ---------------------------------------


def test_parse_planning_type_accepts_every_member():
    for member in PlanningType:
        assert parse_planning_type(member.value) is member


def test_parse_planning_type_is_case_and_whitespace_tolerant():
    assert parse_planning_type(" ACTION ") is PlanningType.ACTION
    assert parse_planning_type("Multi_Step") is PlanningType.MULTI_STEP


def test_parse_planning_type_rejects_unknown_value():
    with pytest.raises(PlanningTypeError):
        parse_planning_type("urgent")


# -- valid construction -------------------------------------------------------


def test_minimal_none_directive_constructs():
    directive = _minimal_directive()
    assert directive.planning_type is PlanningType.NONE
    assert directive.requires_execution is False
    assert directive.confidence == 1.0


def test_fully_populated_multi_step_directive_constructs():
    directive = PlanningDirective(
        objective="Finish the robotics arm around other commitments.",
        planning_type=PlanningType.MULTI_STEP,
        contextual_constraints=(
            GroundedReference(text="Report due Friday", provenance=("commitment:1",)),
        ),
        relevant_resources=(GroundedReference(text="3D printer", provenance=("resource:1",)),),
        dependencies=(GroundedReference(text="Needs the gripper part", provenance=("goal:1",)),),
        assumptions=("Assumes the printer is working.",),
        suggested_approach="Work on it between commitments this week.",
        requires_execution=True,
        confidence=0.75,
    )
    assert directive.planning_type is PlanningType.MULTI_STEP
    assert directive.requires_execution is True
    assert directive.confidence == 0.75


@pytest.mark.parametrize("planning_type", [PlanningType.ACTION, PlanningType.MULTI_STEP])
def test_requires_execution_true_is_accepted_for_action_and_multi_step(planning_type):
    directive = _minimal_directive(planning_type=planning_type, requires_execution=True)
    assert directive.requires_execution is True


# -- coherence: requires_execution vs planning_type --------------------------


@pytest.mark.parametrize("planning_type", [PlanningType.NONE, PlanningType.DECISION])
def test_requires_execution_true_is_incoherent_for_none_and_decision(planning_type):
    with pytest.raises(PlanningDirectiveError):
        _minimal_directive(planning_type=planning_type, requires_execution=True)


def test_requires_execution_false_is_always_fine():
    for planning_type in PlanningType:
        _minimal_directive(planning_type=planning_type, requires_execution=False)


# -- confidence bounds --------------------------------------------------------


def test_confidence_below_zero_rejected():
    with pytest.raises(PlanningDirectiveError):
        _minimal_directive(confidence=-0.01)


def test_confidence_above_one_rejected():
    with pytest.raises(PlanningDirectiveError):
        _minimal_directive(confidence=1.01)


def test_confidence_boundaries_accepted():
    assert _minimal_directive(confidence=0.0).confidence == 0.0
    assert _minimal_directive(confidence=1.0).confidence == 1.0


def test_confidence_bool_rejected():
    with pytest.raises(PlanningDirectiveError):
        _minimal_directive(confidence=True)


# -- required fields / structural validation ---------------------------------


def test_empty_objective_rejected():
    with pytest.raises(PlanningDirectiveError):
        _minimal_directive(objective="")


def test_non_planning_type_member_rejected():
    with pytest.raises(PlanningDirectiveError):
        _minimal_directive(planning_type="action")  # a raw string, not the enum


def test_contextual_constraints_must_be_grounded_reference_instances():
    with pytest.raises(PlanningDirectiveError):
        _minimal_directive(contextual_constraints=("not a GroundedReference",))


def test_grounded_reference_empty_text_rejected():
    with pytest.raises(PlanningDirectiveError):
        GroundedReference(text="")


def test_grounded_reference_provenance_must_be_strings():
    with pytest.raises(PlanningDirectiveError):
        GroundedReference(text="ok", provenance=(1,))


def test_blank_assumption_rejected():
    with pytest.raises(PlanningDirectiveError):
        _minimal_directive(assumptions=("   ",))


def test_blank_suggested_approach_rejected():
    with pytest.raises(PlanningDirectiveError):
        _minimal_directive(suggested_approach="   ")


# -- is_empty() ---------------------------------------------------------------


def test_is_empty_true_for_minimal_none_directive():
    assert _minimal_directive().is_empty() is True


def test_is_empty_false_when_a_constraint_is_present():
    directive = _minimal_directive(
        contextual_constraints=(GroundedReference(text="x", provenance=("request",)),)
    )
    assert directive.is_empty() is False


def test_is_empty_false_for_action_requiring_execution():
    directive = _minimal_directive(planning_type=PlanningType.ACTION, requires_execution=True)
    assert directive.is_empty() is False


def test_is_empty_false_when_suggested_approach_present():
    directive = _minimal_directive(suggested_approach="Do X first.")
    assert directive.is_empty() is False


# -- serialization/deserialization -------------------------------------------


def test_to_dict_round_trips_through_from_dict():
    original = PlanningDirective(
        objective="Finish the robotics arm around other commitments.",
        planning_type=PlanningType.MULTI_STEP,
        contextual_constraints=(
            GroundedReference(text="Report due Friday", provenance=("commitment:1",)),
        ),
        relevant_resources=(GroundedReference(text="3D printer", provenance=("resource:1",)),),
        dependencies=(GroundedReference(text="Needs the gripper part", provenance=("goal:1",)),),
        assumptions=("Assumes the printer is working.",),
        suggested_approach="Work on it between commitments this week.",
        requires_execution=True,
        confidence=0.75,
    )
    restored = PlanningDirective.from_dict(original.to_dict())
    assert restored == original


def test_to_dict_is_json_safe():
    import json

    directive = _minimal_directive(
        planning_type=PlanningType.DECISION,
        contextual_constraints=(GroundedReference(text="x", provenance=("request",)),),
    )
    json.dumps(directive.to_dict())  # must not raise


def test_from_dict_missing_optional_fields_uses_defaults():
    restored = PlanningDirective.from_dict({"objective": "hello", "planning_type": "none"})
    assert restored.confidence == 1.0
    assert restored.requires_execution is False
    assert restored.contextual_constraints == ()
