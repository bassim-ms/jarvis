"""
Focus recognition: the whole-message grammar, and how it sits with the capability router.

Pure and deterministic - no store, no clock, no model. The point of these tests is the BOUNDARY: a request for focus is
claimed; a sentence that merely contains "project", "finish" or "today" is ordinary conversation.
"""

import pytest

from app.capability import Capability, DeterministicCapabilityRouter
from app.focus.recognition import (
    MAX_MESSAGE_CHARS,
    OBJECTIVE,
    QUESTION,
    distinctive_tokens,
    parse_focus_request,
)
from app.handle_this.routing import ObjectiveAwareCapabilityRouter
from app.search.router import is_explicit_search_request

OBJECTIVES = [
    ("I need to finish my Arduino project today.", "Arduino", True),
    ("Jarvis, I need to finish my Arduino project today.", "Arduino", True),
    ("ok, I need to finish my Arduino project", "Arduino", True),
    ("I want to work on my Arduino project now.", "Arduino", True),
    ("I have to wrap up the robotics project tonight", "robotics", True),
    ("I really need to complete our Arduino weather station project right now please", "Arduino weather station", True),
    ("I should continue my thesis this evening", "thesis", False),
    ("I have to finish the report today", "report", False),
]

QUESTIONS = [
    "What should I work on right now?",
    "What should I work on?",
    "What should I focus on today?",
    "what should I do next",
    "What do I need to do today?",
    "what do I have to work on this morning",
    "Jarvis, what should I tackle first?",
    "What should I start with?",
]

NOT_FOCUS = [
    # opinions, past tense, descriptions: contain the words but ask for nothing
    "My Arduino project is annoying.",
    "My Arduino project is annoying today.",
    "I finished my Arduino project today.",
    "I needed to finish my Arduino project today.",
    "I had to finish my Arduino project today.",
    "I wanted to finish my Arduino project today.",
    "I need a break from my Arduino project today.",
    "The project deadline is today.",
    "I'm working on my Arduino project now",
    # questions about concepts
    "Can you explain what a project deadline means?",
    "What is a project?",
    "What should I do about my landlord?",
    "what do I do?",
    "How do I finish a project faster?",
    # imperatives, requests for help, other doors' phrases
    "Finish my Arduino project",
    "Help me finish my Arduino project today",
    "Add a task to finish my Arduino project",
    "Remind me to finish my Arduino project tomorrow",
    "Create a task: finish Arduino project",
    "Search for Arduino project ideas today",
    "morning brief",
    "what do I need to know today?",
    # compound / clause-bearing: more than one request, not a focus request
    "I need to finish my Arduino project today and then email the report to Sam",
    "I need to finish my Arduino project every day at 8",
    "I need to finish my Arduino project by Friday",
    "I need to finish my report to Sam today",
    "I need to finish my Arduino project because it is due",
    # nothing that says WHICH
    "I need to finish my project today",
    "I need to finish it today",
    "I need to finish my stuff today",
    "I need to finish this today",
    # confirmations and continuations
    "yes", "Let's do it.", "Okay, continue.", "go ahead", "do it",
    # junk
    "", "   ", "hello", "x" * (MAX_MESSAGE_CHARS + 1), "I need to finish my Arduino project today\nand email Sam",
    None, 42, ["What should I work on?"], b"What should I work on?",
]


@pytest.mark.parametrize("message,subject,named_project", OBJECTIVES)
def test_an_objective_statement_is_recognised_with_what_the_user_called_it(message, subject, named_project):
    request = parse_focus_request(message)

    assert request is not None and request.kind == OBJECTIVE
    assert request.subject == subject and request.named_project is named_project
    assert request.hint == distinctive_tokens(subject) and request.hint


@pytest.mark.parametrize("message", QUESTIONS)
def test_a_question_about_what_to_work_on_is_recognised(message):
    request = parse_focus_request(message)

    assert request is not None and request.kind == QUESTION and request.hint == () and request.subject == ""


@pytest.mark.parametrize("message", NOT_FOCUS)
def test_everything_else_is_not_a_focus_request(message):
    assert parse_focus_request(message) is None


def test_recognition_never_raises_on_hostile_input():
    for message in ("\x00" * 50, "I need to finish my " + "a" * 150, "I need to finish my " + "(" * 100, "\ud800", "I need to finish my ‮project‬ today"):
        parse_focus_request(message)  # no exception is the assertion


def test_distinctive_tokens_are_whole_words_without_generic_ones():
    assert distinctive_tokens("Finish the Arduino weather-station project, today!") == ("arduino", "weather", "station")
    assert distinctive_tokens("my project") == () and distinctive_tokens("it") == () and distinctive_tokens(None) == ()
    assert "art" not in distinctive_tokens("start")  # whole words only
    assert distinctive_tokens("Arduino arduino ARDUINO") == ("arduino",)


# --- how the grammar sits with the production capability router --------------------------------------------------------------------


def _production_router():
    return ObjectiveAwareCapabilityRouter(DeterministicCapabilityRouter())


def _door_admits(message):
    """The capability gate the chat door uses (the same condition as the Brief / reminder / recall doors)."""
    capability = _production_router().route(message).capability
    return capability is Capability.DIRECT or (capability is Capability.SEARCH and not is_explicit_search_request(message))


@pytest.mark.parametrize("message", [m for m, *_ in OBJECTIVES] + QUESTIONS)
def test_every_recognised_request_reaches_the_door_as_direct_or_a_non_explicit_search(message):
    assert parse_focus_request(message) is not None
    assert _door_admits(message), f"routed to {_production_router().route(message).capability}"


@pytest.mark.parametrize(
    "message,capability",
    [
        ("Create a task: finish Arduino project", Capability.TASK),
        ("Use tool list_tasks", Capability.TOOL),
        ("yes", Capability.CONFIRMATION),
    ],
)
def test_task_tool_and_confirmation_messages_are_not_admitted_to_the_door(message, capability):
    assert _production_router().route(message).capability is capability
    assert not _door_admits(message)


def test_an_explicit_search_is_not_admitted_even_when_it_contains_the_words():
    message = "Search for Arduino project ideas today"
    assert is_explicit_search_request(message) and not _door_admits(message)
    # ...and the grammar would not have claimed it anyway: two independent locks
    assert parse_focus_request(message) is None


def test_no_message_is_both_recognised_and_routed_to_a_command_capability():
    """Whatever the router calls a TOOL / TASK / TASK_STATUS / CONFIRMATION / self-development message, the door never gets it."""
    router = _production_router()
    for message in [m for m, *_ in OBJECTIVES] + QUESTIONS + NOT_FOCUS:
        if not isinstance(message, str):
            continue
        if parse_focus_request(message) is None:
            continue
        assert router.route(message).capability in (Capability.DIRECT, Capability.SEARCH), message
