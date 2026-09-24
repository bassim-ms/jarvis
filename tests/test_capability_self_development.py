"""
Tests for Capability.SELF_DEVELOPMENT routing (app/capability.py):
a clearly-identified development request must trigger it; an ordinary
conversational request must NEVER accidentally trigger it. Mirrors
the worked trigger/non-trigger examples from V1's own spec.
"""

import pytest

from app.capability import Capability, DeterministicCapabilityRouter


@pytest.fixture
def router():
    return DeterministicCapabilityRouter()


# -- explicit invocation phrasing --------------------------------------


def test_explicit_self_develop_invocation():
    router = DeterministicCapabilityRouter()
    decision = router.route("self-develop: add a retry helper to command_runner.py")
    assert decision.capability is Capability.SELF_DEVELOPMENT
    assert decision.goal == "add a retry helper to command_runner.py"


def test_explicit_dev_task_invocation():
    router = DeterministicCapabilityRouter()
    decision = router.route("dev task: fix the flaky test")
    assert decision.capability is Capability.SELF_DEVELOPMENT


# -- verb+target trigger examples (V1's own worked examples) --------------


@pytest.mark.parametrize(
    "message",
    [
        "Jarvis, add X to your code",
        "Jarvis, fix the voice module",
        "Jarvis, inspect why this test is failing",
        "Jarvis, modify the personality system",
        "please update the context engine",
        "can you refactor the planner module",
        "patch the reasoning system",
        "improve your codebase",
    ],
)
def test_triggers_self_development(message):
    router = DeterministicCapabilityRouter()
    decision = router.route(message)
    assert decision.capability is Capability.SELF_DEVELOPMENT
    assert decision.goal == message.strip()


# -- ordinary conversational requests must NEVER trigger it ---------------


@pytest.mark.parametrize(
    "message",
    [
        "search for X",
        "remind me tomorrow",
        "what's on my calendar?",
        "add 1 and 2",
        "can you help me debug this function",
        "can you fix my schedule tomorrow",
        "update the reminder for 5pm",
        "what's happening with my robotics project?",
        "hello there",
        "add milk to my shopping list",
        "fix my sleep schedule",
        "modify the thermostat",
    ],
)
def test_does_not_trigger_self_development(message):
    router = DeterministicCapabilityRouter()
    decision = router.route(message)
    assert decision.capability is not Capability.SELF_DEVELOPMENT


def test_tool_invocation_takes_priority_over_self_development_wording():
    """"use tool" is checked first - an explicit TOOL invocation always
    wins even if it also happens to contain a self-dev-flavored word."""
    router = DeterministicCapabilityRouter()
    decision = router.route("use tool fix_module")
    assert decision.capability is Capability.TOOL


def test_verb_alone_without_a_target_does_not_trigger():
    router = DeterministicCapabilityRouter()
    decision = router.route("please fix the printer")
    assert decision.capability is not Capability.SELF_DEVELOPMENT


def test_target_alone_without_a_verb_does_not_trigger():
    router = DeterministicCapabilityRouter()
    decision = router.route("your code is really impressive")
    assert decision.capability is not Capability.SELF_DEVELOPMENT
