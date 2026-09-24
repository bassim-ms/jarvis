"""
Tests for Capability.SELF_DEV_APPROVAL routing
(app/capability.py:_SELF_DEV_APPROVAL_PATTERN): the explicit "approve/
reject/decline/deny the plan/security edit/commit" phrasing must be
the ONLY thing that triggers it - a bare "yes"/"approve" stays
CONFIRMATION (M15), and any message with extra text around the
phrase stays out of this capability entirely, never partially
matched. Purely routing-level (no engine, no JarvisCore) - see
tests/test_core_self_dev_approval_integration.py for the full
chat-driven approval flow.
"""

import pytest

from app.capability import Capability, DeterministicCapabilityRouter


@pytest.fixture
def router():
    return DeterministicCapabilityRouter()


# -- each gate, each verb --------------------------------------------------


@pytest.mark.parametrize(
    "message,expected_gate,expected_approve",
    [
        ("Approve the plan.", "plan", True),
        ("approve the plan", "plan", True),
        ("Reject the plan.", "plan", False),
        ("decline the plan.", "plan", False),
        ("deny the plan", "plan", False),
        ("Approve the security edit.", "security_edit", True),
        ("Reject the security edit.", "security_edit", False),
        ("decline the security edit", "security_edit", False),
        ("deny the security edit.", "security_edit", False),
        ("Approve the commit.", "commit", True),
        ("Reject the commit.", "commit", False),
        ("decline the commit.", "commit", False),
        ("please approve the plan", "plan", True),
        ("Approve plan", "plan", True),  # "the" is optional
        ("Approve commit", "commit", True),
    ],
)
def test_recognizes_each_gate_and_verb(router, message, expected_gate, expected_approve):
    decision = router.route(message)
    assert decision.capability is Capability.SELF_DEV_APPROVAL
    assert decision.self_dev_gate == expected_gate
    assert decision.approve is expected_approve


# -- never a bare/ambiguous word -------------------------------------------


@pytest.mark.parametrize("message", ["yes", "approve", "approved", "ok", "confirm", "no", "decline"])
def test_bare_words_stay_confirmation_not_self_dev_approval(router, message):
    decision = router.route(message)
    assert decision.capability is not Capability.SELF_DEV_APPROVAL


# -- never a partial/substring match ---------------------------------------


@pytest.mark.parametrize(
    "message",
    [
        "Approve the plan, and also tell me a joke",
        "I approve the plan for tomorrow's meeting",
        "Should I approve the plan?",
        "Approve the plan for the trip",
    ],
)
def test_extra_surrounding_text_never_matches(router, message):
    decision = router.route(message)
    assert decision.capability is not Capability.SELF_DEV_APPROVAL


def test_unknown_gate_name_never_matches(router):
    decision = router.route("Approve the rollback.")
    assert decision.capability is not Capability.SELF_DEV_APPROVAL


# -- priority: checked before SELF_DEVELOPMENT's own verb+target ----------


def test_takes_priority_when_it_is_the_whole_message(router):
    decision = router.route("Approve the security edit.")
    assert decision.capability is Capability.SELF_DEV_APPROVAL


def test_tool_invocation_still_takes_priority():
    router = DeterministicCapabilityRouter()
    decision = router.route("use tool approve_plan")
    assert decision.capability is Capability.TOOL
