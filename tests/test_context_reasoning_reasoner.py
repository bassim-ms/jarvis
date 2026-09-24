"""
Tests for app/context_reasoning/reasoner.py: LLMContextReasoner - strict
parsing/validation of AI output into a ReasoningResult, the
empty-ContextSnapshot no-call short circuit, and safe failure behavior.

Uses a plain fake AIProvider (never a real network call) - `reason()`
calls `call_with_requirements`, which treats a plain AIProvider as a
no-op pass-through and calls `generate_reply(prompt)` directly (see
app/providers/manager.py:call_with_requirements), so these tests see the
exact prompt built and can hand back whatever canned JSON they need.
"""

import json

import pytest

from app.context_engine.base import RequestContext
from app.context_reasoning.reasoner import (
    ContextReasonerError,
    LLMContextReasoner,
)
from app.providers.base import AIProvider, ProviderError
from app.world_model.base import (
    ContextSnapshot,
    Priority,
    Project,
    ProjectStatus,
)

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


def _rc(message: str = "tell me about my robotics arm project") -> RequestContext:
    return RequestContext(raw_request=message)


def _valid_payload(project_id: str = "project-1") -> dict:
    return {
        "interpretation": "The user wants to know about their robotics arm project.",
        "relevant_facts": [
            {"text": "The robotics arm project is active.", "provenance": [f"project:{project_id}"]}
        ],
        "constraints": [],
        "conflicts": [],
        "opportunities": [],
        "conclusions": [
            {"text": "Nothing is blocking this project right now.", "provenance": [f"project:{project_id}"]}
        ],
        "recommended_next_step": "Ask what specifically they want to know.",
        "confidence": 0.9,
        "requires_clarification": False,
        "clarification_question": None,
    }


# -- empty-context short circuit ---------------------------------------------


def test_empty_snapshot_never_calls_the_provider():
    provider = _FakeProvider(reply="should never be used")
    reasoner = LLMContextReasoner(provider)

    result = reasoner.reason("hello there", _rc("hello there"), ContextSnapshot(current_time=_NOW))

    assert provider.calls == 0
    assert result.is_empty()


# -- valid structured response ------------------------------------------------


def test_valid_response_produces_a_matching_reasoning_result():
    provider = _FakeProvider(reply=json.dumps(_valid_payload()))
    reasoner = LLMContextReasoner(provider)

    result = reasoner.reason(
        "tell me about my robotics arm project", _rc(), _non_empty_snapshot()
    )

    assert provider.calls == 1
    assert "robotics arm project" in result.interpretation
    assert result.relevant_facts[0].text == "The robotics arm project is active."
    assert result.confidence == 0.9
    assert result.requires_clarification is False


def test_response_wrapped_in_markdown_code_fence_is_tolerated():
    fenced = "```json\n" + json.dumps(_valid_payload()) + "\n```"
    provider = _FakeProvider(reply=fenced)
    reasoner = LLMContextReasoner(provider)

    result = reasoner.reason("hello", _rc(), _non_empty_snapshot())
    assert result.confidence == 0.9


# -- malformed response --------------------------------------------------------


def test_invalid_json_raises_context_reasoner_error():
    provider = _FakeProvider(reply="not json at all")
    reasoner = LLMContextReasoner(provider)

    with pytest.raises(ContextReasonerError):
        reasoner.reason("hello", _rc(), _non_empty_snapshot())


def test_non_object_json_raises_context_reasoner_error():
    provider = _FakeProvider(reply="[1, 2, 3]")
    reasoner = LLMContextReasoner(provider)

    with pytest.raises(ContextReasonerError):
        reasoner.reason("hello", _rc(), _non_empty_snapshot())


def test_unexpected_top_level_field_raises():
    payload = _valid_payload()
    payload["extra_field"] = "not allowed"
    provider = _FakeProvider(reply=json.dumps(payload))
    reasoner = LLMContextReasoner(provider)

    with pytest.raises(ContextReasonerError):
        reasoner.reason("hello", _rc(), _non_empty_snapshot())


# -- missing fields -------------------------------------------------------


def test_missing_interpretation_raises():
    payload = _valid_payload()
    del payload["interpretation"]
    provider = _FakeProvider(reply=json.dumps(payload))
    reasoner = LLMContextReasoner(provider)

    with pytest.raises(ContextReasonerError):
        reasoner.reason("hello", _rc(), _non_empty_snapshot())


def test_missing_requires_clarification_raises():
    payload = _valid_payload()
    del payload["requires_clarification"]
    provider = _FakeProvider(reply=json.dumps(payload))
    reasoner = LLMContextReasoner(provider)

    with pytest.raises(ContextReasonerError):
        reasoner.reason("hello", _rc(), _non_empty_snapshot())


# -- invalid values -------------------------------------------------------


def test_confidence_above_one_raises():
    payload = _valid_payload()
    payload["confidence"] = 1.5
    provider = _FakeProvider(reply=json.dumps(payload))
    reasoner = LLMContextReasoner(provider)

    with pytest.raises(ContextReasonerError):
        reasoner.reason("hello", _rc(), _non_empty_snapshot())


def test_confidence_below_zero_raises():
    payload = _valid_payload()
    payload["confidence"] = -0.5
    provider = _FakeProvider(reply=json.dumps(payload))
    reasoner = LLMContextReasoner(provider)

    with pytest.raises(ContextReasonerError):
        reasoner.reason("hello", _rc(), _non_empty_snapshot())


def test_non_numeric_confidence_raises():
    payload = _valid_payload()
    payload["confidence"] = "high"
    provider = _FakeProvider(reply=json.dumps(payload))
    reasoner = LLMContextReasoner(provider)

    with pytest.raises(ContextReasonerError):
        reasoner.reason("hello", _rc(), _non_empty_snapshot())


def test_confidence_boundaries_are_accepted():
    for boundary in (0.0, 1.0):
        payload = _valid_payload()
        payload["confidence"] = boundary
        provider = _FakeProvider(reply=json.dumps(payload))
        reasoner = LLMContextReasoner(provider)
        result = reasoner.reason("hello", _rc(), _non_empty_snapshot())
        assert result.confidence == boundary


# -- unsupported entities / unsupported facts (anti-hallucination) -----------


def test_provenance_citing_an_unknown_entity_id_raises():
    payload = _valid_payload()
    payload["relevant_facts"] = [
        {"text": "Fabricated fact about a made-up goal.", "provenance": ["goal:does-not-exist"]}
    ]
    provider = _FakeProvider(reply=json.dumps(payload))
    reasoner = LLMContextReasoner(provider)

    with pytest.raises(ContextReasonerError):
        reasoner.reason("hello", _rc(), _non_empty_snapshot())


def test_fact_with_empty_provenance_raises():
    payload = _valid_payload()
    payload["relevant_facts"] = [{"text": "Ungrounded claim.", "provenance": []}]
    provider = _FakeProvider(reply=json.dumps(payload))
    reasoner = LLMContextReasoner(provider)

    with pytest.raises(ContextReasonerError):
        reasoner.reason("hello", _rc(), _non_empty_snapshot())


def test_conflict_citing_a_real_and_a_fake_id_raises():
    payload = _valid_payload()
    payload["conflicts"] = [
        {
            "text": "A conflict partially grounded, partially invented.",
            "provenance": ["project:project-1", "event:does-not-exist"],
        }
    ]
    provider = _FakeProvider(reply=json.dumps(payload))
    reasoner = LLMContextReasoner(provider)

    with pytest.raises(ContextReasonerError):
        reasoner.reason("hello", _rc(), _non_empty_snapshot())


def test_known_entity_grounded_fact_is_accepted():
    """Sanity check for the anti-hallucination test above: citing the
    REAL project id that was actually supplied is accepted."""
    payload = _valid_payload(project_id="project-1")
    provider = _FakeProvider(reply=json.dumps(payload))
    reasoner = LLMContextReasoner(provider)

    result = reasoner.reason("hello", _rc(), _non_empty_snapshot())
    assert result.relevant_facts[0].provenance == ("project:project-1",)


# -- clarification-required reasoning -----------------------------------------


def test_requires_clarification_true_is_accepted_with_a_question():
    payload = _valid_payload()
    payload["requires_clarification"] = True
    payload["clarification_question"] = "Do you mean Luniti or robotics tonight?"
    provider = _FakeProvider(reply=json.dumps(payload))
    reasoner = LLMContextReasoner(provider)

    result = reasoner.reason("Luniti or robotics tonight?", _rc(), _non_empty_snapshot())
    assert result.requires_clarification is True
    assert result.clarification_question == "Do you mean Luniti or robotics tonight?"


def test_requires_clarification_true_without_question_raises():
    payload = _valid_payload()
    payload["requires_clarification"] = True
    payload["clarification_question"] = None
    provider = _FakeProvider(reply=json.dumps(payload))
    reasoner = LLMContextReasoner(provider)

    with pytest.raises(ContextReasonerError):
        reasoner.reason("hello", _rc(), _non_empty_snapshot())


def test_requires_clarification_false_with_question_raises():
    payload = _valid_payload()
    payload["requires_clarification"] = False
    payload["clarification_question"] = "This shouldn't be here."
    provider = _FakeProvider(reply=json.dumps(payload))
    reasoner = LLMContextReasoner(provider)

    with pytest.raises(ContextReasonerError):
        reasoner.reason("hello", _rc(), _non_empty_snapshot())


# -- provider failures ---------------------------------------------------


def test_provider_error_raises_context_reasoner_error():
    provider = _FakeProvider(error=ProviderError("boom"))
    reasoner = LLMContextReasoner(provider)

    with pytest.raises(ContextReasonerError):
        reasoner.reason("hello", _rc(), _non_empty_snapshot())


# -- prompt construction --------------------------------------------------


def test_prompt_sent_to_provider_includes_the_raw_request_and_allowed_provenance():
    provider = _FakeProvider(reply=json.dumps(_valid_payload()))
    reasoner = LLMContextReasoner(provider)

    reasoner.reason("tell me about my robotics arm project", _rc(), _non_empty_snapshot())

    assert "tell me about my robotics arm project" in provider.last_prompt
    assert "project:project-1" in provider.last_prompt
