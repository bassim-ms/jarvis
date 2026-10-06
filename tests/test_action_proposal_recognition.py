"""
Action Proposal recognition: the two closed grammars (a request for a plan, and a reply while a proposal waits) and how those phrases route
through the production capability router. Pure and deterministic - no store, no clock, no model.
"""

import pytest

from app.action_proposal.recognition import (
    ACK,
    APPROVE,
    CHANGE,
    EXPLAIN_AFTER,
    EXPLAIN_STEP,
    EXPLAIN_WHY,
    LAST,
    MAX_MESSAGE_CHARS,
    REJECT,
    SHOW,
    SKIP,
    UNRELATED,
    classify_follow_up,
    parse_plan_request,
)
from app.capability import Capability, DeterministicCapabilityRouter
from app.handle_this.routing import ObjectiveAwareCapabilityRouter

APPROVALS = [
    "yes", "Yes.", "YES", "yeah", "Yeah!", "yep", "yup", "yes please", "Yes, please.", "go ahead", "Go ahead.", "do it", "Do it!",
    "let's do it", "Let's do it.", "lets do it", "Yeah, let's do it.", "yes, go ahead", "yeah go ahead", "yes do it", "approve", "approved",
    "Jarvis, yes", "ok, yes", "ok let's do it", "okay, go ahead", "sure, go ahead", "Sure, let's do it.",
]
REJECTIONS = [
    "no", "No.", "nope", "nah", "no thanks", "cancel", "cancel that", "cancel the plan", "stop", "not now", "never mind", "nevermind",
    "forget it", "don't", "don't do that", "decline", "reject", "drop it", "drop the plan",
]
NODS = ["ok", "okay", "OK.", "k", "sure", "Sure!", "alright", "all right", "fine", "sounds good", "proceed", "send it"]
NOT_APPROVALS = [
    # a reservation, an addition, a second request: never an approval
    "yes but skip step 2", "yes, but only the first one", "yes and also email Sam", "go ahead and email Sam", "yes, skip step 2",
    "do it later", "let's do it tomorrow", "yes?", "yes or no", "not yes", "yes no", "yesterday", "yes please do it now and then tell me",
    "ok", "sure", "ok then", "sure thing",
    # other phrases that merely resemble one
    "why do it", "do it for me", "I did it", "let's not do it", "go ahead of me", "approve of that",
    "", "   ", "yes\nno", "x" * (MAX_MESSAGE_CHARS + 1), None, 42, ["yes"], b"yes",
]


@pytest.mark.parametrize("message", APPROVALS)
def test_an_approval_is_recognised_only_as_a_whole_message(message):
    assert classify_follow_up(message).kind == APPROVE


@pytest.mark.parametrize("message", REJECTIONS)
def test_a_rejection_is_recognised(message):
    assert classify_follow_up(message).kind == REJECT


@pytest.mark.parametrize("message", NODS)
def test_ok_and_sure_are_a_nod_never_an_approval(message):
    assert classify_follow_up(message).kind == ACK


@pytest.mark.parametrize("message", NOT_APPROVALS)
def test_a_reservation_an_addition_or_a_lookalike_never_approves(message):
    assert classify_follow_up(message).kind != APPROVE


@pytest.mark.parametrize("message", ["yes but skip step 2", "yes, and skip step 2", "go ahead and email Sam", "yes but"])
def test_a_yes_with_a_but_is_unrelated_so_the_proposal_is_dropped_rather_than_guessed_at(message):
    assert classify_follow_up(message).kind == UNRELATED


@pytest.mark.parametrize(
    "message,kind,step",
    [
        ("What does step 2 mean?", EXPLAIN_STEP, 2),
        ("what is step 3?", EXPLAIN_STEP, 3),
        ("What's step 1", EXPLAIN_STEP, 1),
        ("Can you explain step 3?", EXPLAIN_STEP, 3),
        ("explain step 2", EXPLAIN_STEP, 2),
        ("Could you please explain the second step?", EXPLAIN_STEP, 2),
        ("describe the last step", EXPLAIN_STEP, LAST),
        ("What happens after step 2?", EXPLAIN_AFTER, 2),
        ("what's after step 1", EXPLAIN_AFTER, 1),
        ("Why this plan?", EXPLAIN_WHY, None),
        ("why this order", EXPLAIN_WHY, None),
        ("why step 2?", EXPLAIN_WHY, 2),
        ("Why is step 3 there?", EXPLAIN_WHY, 3),
        ("Show the plan again", SHOW, None),
        ("repeat the plan", SHOW, None),
        ("what was the plan?", SHOW, None),
        ("Can we skip step 1?", SKIP, 1),
        ("skip step 2", SKIP, 2),
        ("Please skip the first step", SKIP, 1),
        ("let's skip step 3", SKIP, 3),
        ("drop step 2", SKIP, 2),
        ("skip the last step", SKIP, LAST),
        ("change step 2", CHANGE, 2),
        ("add a step", CHANGE, None),
        ("can we swap the order", CHANGE, None),
        ("could you replace step 1 with something shorter", CHANGE, 1),
    ],
)
def test_a_question_about_the_plan_or_a_skip_is_a_follow_up_that_keeps_it_alive(message, kind, step):
    follow = classify_follow_up(message)
    assert (follow.kind, follow.step) == (kind, step)


@pytest.mark.parametrize(
    "message",
    [
        "What's the weather tomorrow?", "Remind me to call Mom at 6", "What should I work on?", "hello", "tell me a joke",
        "explain step 2 and then email Sam", "skip step 2 and step 3", "what does step 2 of my homework mean to my teacher",
        "how do I wire a joystick?", "make step 2 shorter", "I'm back", "Can you explain what a project deadline means?",
        "search the web for servo wiring", "What happens after the exam?", "why did my servo burn out?",
    ],
)
def test_anything_else_is_unrelated(message):
    assert classify_follow_up(message).kind == UNRELATED


def test_classification_never_raises_on_hostile_input():
    for message in ("\x00" * 50, "skip step " + "9" * 200, "explain ‮step 2‬", "\ud800", "step 99999999999999999999 mean"):
        classify_follow_up(message)


@pytest.mark.parametrize(
    "message,subject",
    [
        ("Make a plan", ""),
        ("make me a plan", ""),
        ("Give me a plan.", ""),
        ("make a plan please", ""),
        ("Jarvis, make a plan", ""),
        ("Make a plan for my Arduino project", "Arduino project"),
        ("Make a plan for my Arduino project today", "Arduino project"),
        ("make me a plan for the report", "report"),
        ("Make a plan to finish the report", "report"),
        ("help me plan my Arduino project", "Arduino project"),
        ("Help me plan our thesis project", "thesis project"),
    ],
)
def test_a_request_for_a_plan_is_recognised(message, subject):
    request = parse_plan_request(message)
    assert request is not None and request.subject == subject


@pytest.mark.parametrize(
    "message",
    [
        "plan it", "what's the plan", "Plan my Arduino project", "I need a plan", "make a plan for my project and then email Sam, please",
        "Make a plan for my trip to Paris and book flights", "can you make a plan for next year's budget at the bank",
        "what is a plan?", "make a planet", "Make a plan.\nAnd email Sam", "", "x" * (MAX_MESSAGE_CHARS + 1), None, 42,
    ],
)
def test_a_sentence_that_only_contains_the_words_is_not_a_plan_request(message):
    request = parse_plan_request(message)
    # (a subject with a clause in it is turned away later, by the same reading of "my <X> project" that Focus uses)
    assert request is None or "and" in request.subject.lower().split()


# --- how these phrases reach the door: the production router ---------------------------------------------------------------------------


def _capability(message):
    return ObjectiveAwareCapabilityRouter(DeterministicCapabilityRouter()).route(message).capability


@pytest.mark.parametrize("message", ["yes", "Yes.", "yeah", "go ahead", "do it", "approve", "no", "nope", "ok", "sure", "stop", "cancel"])
def test_these_replies_reach_a_waiting_proposal_through_the_confirmation_branch(message):
    assert _capability(message) is Capability.CONFIRMATION


@pytest.mark.parametrize(
    "message",
    ["let's do it", "Yeah, let's do it.", "yes please", "yes, go ahead", "never mind", "forget it", "What does step 2 mean?", "Can we skip step 1?",
     "Why this plan?", "Show the plan again", "What happens after step 2?", "Make a plan", "Make a plan for my Arduino project",
     "help me plan my Arduino project"],
)
def test_these_reach_it_through_the_ordinary_conversation_door(message):
    assert _capability(message) is Capability.DIRECT


@pytest.mark.parametrize("message", ["not now", "Make a plan for my Arduino project today"])
def test_a_freshness_word_sends_a_phrase_down_the_search_route_where_the_door_still_sees_it(message):
    assert _capability(message) is Capability.SEARCH


@pytest.mark.parametrize("message", ["plan it", "plan my Arduino project", "Plan the Arduino project"])
def test_a_goal_verb_phrase_belongs_to_handle_this_not_to_the_proposal_door(message):
    assert _capability(message) is Capability.TASK  # which is why the request grammar does not claim it
