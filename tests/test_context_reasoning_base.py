"""
Tests for app/context_reasoning/base.py: ReasoningResult and its nested
value types are pure, validated, serializable data - no I/O, no AI call.
"""

import pytest

from app.context_reasoning.base import (
    Conclusion,
    Conflict,
    Constraint,
    ContextReasoningError,
    Opportunity,
    RelevantFact,
    ReasoningResult,
)


def _minimal_result(**overrides) -> ReasoningResult:
    defaults = dict(interpretation="The user wants to know about their robotics project.")
    defaults.update(overrides)
    return ReasoningResult(**defaults)


# -- valid construction -----------------------------------------------------


def test_minimal_valid_result_constructs():
    result = _minimal_result()
    assert result.interpretation.startswith("The user")
    assert result.confidence == 1.0
    assert result.requires_clarification is False
    assert result.clarification_question is None


def test_fully_populated_result_constructs():
    result = ReasoningResult(
        interpretation="Deciding what to work on tonight.",
        relevant_facts=(RelevantFact(text="Blocker X is active", provenance=("blocker:1",)),),
        constraints=(Constraint(text="Printer unavailable", provenance=("blocker:1",)),),
        conflicts=(Conflict(text="Two events overlap", provenance=("event:1", "event:2")),),
        opportunities=(Opportunity(text="Free evening", provenance=("request",)),),
        conclusions=(Conclusion(text="Robotics is blocked", provenance=("blocker:1",)),),
        recommended_next_step="Work on Luniti instead.",
        confidence=0.8,
        requires_clarification=False,
    )
    assert result.confidence == 0.8
    assert result.relevant_facts[0].text == "Blocker X is active"


def test_requires_clarification_with_question_constructs():
    result = _minimal_result(
        requires_clarification=True, clarification_question="Luniti or robotics tonight?"
    )
    assert result.requires_clarification is True
    assert result.clarification_question == "Luniti or robotics tonight?"


# -- confidence bounds --------------------------------------------------------


def test_confidence_below_zero_rejected():
    with pytest.raises(ContextReasoningError):
        _minimal_result(confidence=-0.01)


def test_confidence_above_one_rejected():
    with pytest.raises(ContextReasoningError):
        _minimal_result(confidence=1.01)


def test_confidence_boundaries_accepted():
    assert _minimal_result(confidence=0.0).confidence == 0.0
    assert _minimal_result(confidence=1.0).confidence == 1.0


def test_confidence_bool_rejected():
    """bool is a subclass of int in Python - must not silently pass as
    a valid confidence float, matching validate_confidence's own
    discipline in app/world_model/base.py."""
    with pytest.raises(ContextReasoningError):
        _minimal_result(confidence=True)


# -- required fields ----------------------------------------------------------


def test_empty_interpretation_rejected():
    with pytest.raises(ContextReasoningError):
        _minimal_result(interpretation="")


def test_blank_interpretation_rejected():
    with pytest.raises(ContextReasoningError):
        _minimal_result(interpretation="   ")


def test_requires_clarification_true_without_question_rejected():
    with pytest.raises(ContextReasoningError):
        _minimal_result(requires_clarification=True, clarification_question=None)


def test_requires_clarification_false_with_question_rejected():
    with pytest.raises(ContextReasoningError):
        _minimal_result(requires_clarification=False, clarification_question="Which one?")


def test_relevant_facts_must_be_relevant_fact_instances():
    with pytest.raises(ContextReasoningError):
        _minimal_result(relevant_facts=("not a RelevantFact",))


def test_relevant_fact_empty_text_rejected():
    with pytest.raises(ContextReasoningError):
        RelevantFact(text="")


def test_relevant_fact_provenance_must_be_strings():
    with pytest.raises(ContextReasoningError):
        RelevantFact(text="ok", provenance=(1,))


# -- is_empty() ---------------------------------------------------------------


def test_is_empty_true_for_minimal_result():
    assert _minimal_result().is_empty() is True


def test_is_empty_false_when_a_fact_is_present():
    result = _minimal_result(
        relevant_facts=(RelevantFact(text="x", provenance=("request",)),)
    )
    assert result.is_empty() is False


def test_is_empty_false_when_requires_clarification():
    result = _minimal_result(
        requires_clarification=True, clarification_question="Which one?"
    )
    assert result.is_empty() is False


# -- serialization/deserialization -------------------------------------------


def test_to_dict_round_trips_through_from_dict():
    original = ReasoningResult(
        interpretation="Deciding what to work on tonight.",
        relevant_facts=(RelevantFact(text="Blocker X is active", provenance=("blocker:1",)),),
        constraints=(Constraint(text="Printer unavailable", provenance=("blocker:1",)),),
        conflicts=(Conflict(text="Two events overlap", provenance=("event:1", "event:2")),),
        opportunities=(Opportunity(text="Free evening", provenance=("request",)),),
        conclusions=(Conclusion(text="Robotics is blocked", provenance=("blocker:1",)),),
        recommended_next_step="Work on Luniti instead.",
        confidence=0.8,
        requires_clarification=False,
    )
    restored = ReasoningResult.from_dict(original.to_dict())
    assert restored == original


def test_to_dict_is_json_safe():
    import json

    result = _minimal_result(
        relevant_facts=(RelevantFact(text="x", provenance=("request",)),)
    )
    json.dumps(result.to_dict())  # must not raise


def test_from_dict_missing_optional_fields_uses_defaults():
    restored = ReasoningResult.from_dict({"interpretation": "hello"})
    assert restored.confidence == 1.0
    assert restored.requires_clarification is False
    assert restored.relevant_facts == ()
