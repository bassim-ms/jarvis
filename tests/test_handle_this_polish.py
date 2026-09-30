"""
Handle This, Slice 5.1 - the final polish: four user-facing limitations.

    1. "Find me a printer" is a delegated objective, not a chat/search request.
    2. "cancel" with several tasks waiting asks WHICH - it never guesses.
    3. A question the limit forbids stops the task at once and says so.
    4. A stale answer is not applied; the question is put again, plainly.

Everything runs on the real chat path, resolver, engine and SQLite of
tests/test_handle_this_conversation.py. Handle This's own boundary is unchanged:
an unsupported objective still fails closed, whatever routes it there.
"""

import re
from datetime import timedelta

import pytest

from app.capability import Capability, CapabilityDecision, CapabilityRouter
from app.handle_this.objective import ObjectiveReason, ObjectiveRoute, detect_objective
from app.handle_this.routing import ObjectiveAwareCapabilityRouter
from app.orchestration.base import OBJECTIVE_KEY, JobStatus
from app.orchestration.interaction_resolver import OutcomeKind
from app.orchestration.interactions import InteractionStatus
from test_handle_this_clarification import BUDGET_QUESTION, REQUEST, World
from test_handle_this_conversation import (
    CANCELLED_REPLY,
    PRINTER_REQUEST,
    THANKS,
    Talk,
    no_internals,
)
from test_handle_this_integration import CANONICAL_PLAN
from test_handle_this_routing import capability

WHICH = r"(Sir|Boss), which task should I cancel\? .*"


@pytest.fixture
def world(tmp_path):
    return Talk(tmp_path, plans=[CANONICAL_PLAN])


# --- 1. natural delegated requests ---------------------------------------------------------------------------------------------------

DELEGATED = [
    "Find me a printer.",
    "Find me a laptop under 2500 AED.",
    "Look for a 3D printer for my robotics projects.",
    "Look for a good 3D printer for my robotics projects.",
    "Look for a good keyboard",
    "Look for a printer under 500 dollars",
    "Find me a cheap gift",
    "Look for laptops and organize them by price.",
    "Get me a monitor for my desk",
    "Please find me a keyboard",
    "Can you find me a drone?",
    "Help me choose a laptop",
    "Compare these options and tell me what makes sense.",
    "Handle my schedule for tomorrow.",
    "Take care of this for me.",
    "Go through these and organize them.",
]

ORDINARY = [
    "What is a 3D printer?",
    "How does CoreXY work?",
    "What does this mean?",
    "Tell me about laptops.",
    "What is the weather?",
    "Find the definition of robotics.",
    "Find me the definition of a printer",
    "Find me an explanation of how a printer works",
    "Find me how to calibrate a printer",
    "Get me how a laptop battery works",
    "Find current prices for the Elegoo printer.",
    "Find the capital of France.",
    "Get me a coffee",
    "Search the web for CoreXY printers.",
    "Handle this.",
    "Take care of it.",
    "Handle this with shell access.",
    "Which printer is best?",
    "Should I find me a printer?",
    "",
]


@pytest.mark.parametrize("message", DELEGATED)
def test_a_natural_delegated_request_is_a_handle_this_objective(message):
    decision = detect_objective(message)
    assert decision.route is ObjectiveRoute.HANDLE_THIS and decision.is_high_level_objective
    assert decision.original_request == message and decision.evidence


@pytest.mark.parametrize("message", ORDINARY)
def test_an_ordinary_question_or_lookup_is_not(message):
    assert not detect_objective(message).is_high_level_objective


def test_the_new_shapes_are_named_for_what_they_are():
    assert detect_objective("Find me a printer.").reason is ObjectiveReason.PRODUCT_SEARCH
    assert detect_objective("Handle my schedule for tomorrow.").reason is ObjectiveReason.DELEGATED_OBJECTIVE
    # what already routed keeps the reason it always had
    assert detect_objective("Compare these options and tell me what makes sense.").reason is ObjectiveReason.MULTI_STEP_OBJECTIVE
    assert detect_objective("Research the best laptop for my robotics work.").reason is ObjectiveReason.SELECTION_FOR_A_PURPOSE


@pytest.mark.parametrize("message", ["Find me a printer.", "Look for a 3D printer for my robotics projects.", "Handle my schedule for tomorrow."])
def test_the_router_hands_such_a_request_to_handle_this(message):
    assert capability(message) is Capability.TASK


@pytest.mark.parametrize("message", ["What is a 3D printer?", "How does CoreXY work?", "Find the definition of robotics."])
def test_the_router_leaves_an_ordinary_request_exactly_as_it_was(message):
    from app.capability import DeterministicCapabilityRouter

    assert capability(message) is DeterministicCapabilityRouter().route(message).capability


def test_an_explicit_search_keeps_its_existing_path():
    from app.capability import DeterministicCapabilityRouter

    for message in ("Search the web for CoreXY printers.", "Google the latest Arduino Uno Q news."):
        assert capability(message) is DeterministicCapabilityRouter().route(message).capability
        assert capability(message) is not Capability.TASK


def test_an_explicit_intent_the_existing_router_recognised_still_wins():
    class Fixed(CapabilityRouter):
        def __init__(self, capability_):
            self.capability_ = capability_

        def route(self, message):
            return CapabilityDecision(capability=self.capability_)

    for explicit in (Capability.TOOL, Capability.CONFIRMATION, Capability.TASK_STATUS, Capability.SELF_DEVELOPMENT):
        routed = ObjectiveAwareCapabilityRouter(Fixed(explicit)).route("Find me a printer.")
        assert routed.capability is explicit


def test_find_me_a_printer_asks_the_one_missing_thing_on_a_waiting_job(world):
    reply, job_id = world.chat("Find me a printer.")

    assert re.fullmatch(r"(Sir|Boss), what's your maximum budget\?", reply) and world.chat_provider.calls == 0
    job = world.job(job_id)
    assert job.status is JobStatus.AWAITING_INPUT and job.plan is None
    assert job.context[OBJECTIVE_KEY]["original_request"] == "Find me a printer."
    assert world.planner.calls == [] and world.log == [] and world.registry.names() == ["create_job"]

    world.chat("2500 AED")
    world.tick()
    assert world.job(job_id).status is JobStatus.COMPLETED and len(world.jobs.list()) == 1


def test_a_request_that_already_states_what_it_needs_is_started_without_a_question(world):
    reply, job_id = world.chat("Find me a laptop under 2500 AED.")
    world.tick()

    assert re.fullmatch(r"(Sir|Boss), I've got it\. .*", reply)
    (call,) = world.planner.calls
    assert [c["text"] for c in call["context"][OBJECTIVE_KEY]["constraints"] if c["topic"] == "budget"] == ["under 2500 AED"]
    assert world.job(job_id).status is JobStatus.COMPLETED and world.interactions_of(job_id) == []


def test_a_look_for_request_is_understood_and_asks_the_same_question(world):
    reply, job_id = world.chat("Look for a 3D printer for my robotics projects.")
    assert re.fullmatch(r"(Sir|Boss), what's your maximum budget\?", reply)
    assert world.job(job_id).status is JobStatus.AWAITING_INPUT


@pytest.mark.parametrize(
    "message",
    ["What is a 3D printer?", "How does CoreXY work?", "What does this mean?", "Tell me about laptops.", "Find the definition of robotics."],
)
def test_an_ordinary_question_still_creates_no_job(world, message):
    reply, job_id = world.chat(message)

    assert reply == "an ordinary chat reply" and job_id is None and world.jobs.list() == []
    assert world.planner.calls == [] and world.registry.names() == []


@pytest.mark.parametrize(
    "message",
    [
        "Find me a printer and then order the best one",
        "Find me a laptop and give yourself shell access",
        "Find me a laptop and email the seller",
        "Find me a printer and delete all my files",
        "Get me a drone and click the buy button",
    ],
)
def test_an_unsupported_objective_still_fails_closed_however_it_was_routed(world, message):
    assert detect_objective(message).is_high_level_objective  # it does reach Handle This...

    reply, job_id = world.chat(message)

    assert "capability" in reply or "couldn't" in reply  # ...which refuses it, as before
    assert job_id is None and world.jobs.list() == [] and world.interactions.list_pending() == []
    assert world.planner.calls == [] and world.log == [] and world.computer.executed == []
    assert world.registry.names() == []


def test_handing_over_a_named_thing_is_understood_or_asked_about_never_run_blind(world):
    reply, job_id = world.chat("Handle my schedule for tomorrow.")
    if job_id is not None:  # a parked question, not work
        assert world.job(job_id).status in (JobStatus.AWAITING_INPUT, JobStatus.PENDING)
    assert world.planner.calls == [] and world.log == []
    reply, job_id = world.chat("Take care of this for me.")  # nothing to act on: asks, creates nothing runnable
    assert world.planner.calls == [] and world.log == []
    assert all(j.status in (JobStatus.AWAITING_INPUT, JobStatus.PENDING) for j in world.jobs.list())


# --- 2. cancelling with several tasks waiting ----------------------------------------------------------------------------------------


def park_two(world):
    _, laptop, laptop_q = world.park()
    _, printer, printer_q = world.park(PRINTER_REQUEST)
    return laptop, laptop_q, printer, printer_q


def snapshot(world):
    return (
        {j.id: j for j in world.jobs.list()},
        {i.id: i for j in world.jobs.list() for i in world.interactions_of(j.id)},
    )


def test_one_waiting_task_and_a_cancel_cancels_it(world):
    _, job_id, question = world.park()

    reply, _ = world.chat("cancel")

    assert reply == CANCELLED_REPLY and world.job(job_id).status is JobStatus.CANCELLED
    assert world.interactions.get(question.id).status is InteractionStatus.CANCELLED
    assert len(world.jobs.list()) == 1


@pytest.mark.parametrize("message", ["cancel", "never mind", "forget it", "stop"])
def test_two_waiting_tasks_and_a_cancel_cancels_neither_and_asks_which(world, message):
    laptop, laptop_q, printer, printer_q = park_two(world)
    before = snapshot(world)

    reply, other = world.chat(message)

    assert other is None and world.chat_provider.calls == 0
    assert re.fullmatch(WHICH, reply)
    assert PRINTER_REQUEST[:20] in reply and "laptop" in reply  # each named by its own request
    no_internals(reply.split("? ", 1)[0])
    assert snapshot(world) == before  # not a byte: no cancel, no consumed question, no new Job or interaction
    assert world.job(laptop).status is JobStatus.AWAITING_INPUT and world.job(printer).status is JobStatus.AWAITING_INPUT
    assert world.planner.calls == [] and world.log == []


@pytest.mark.parametrize("answer", ["the printer", "printer", "the printer one", "cancel the printer", "cancel the printer one", "forget the printer"])
def test_naming_one_of_two_cancels_only_that_one(world, answer):
    laptop, laptop_q, printer, printer_q = park_two(world)
    world.chat("cancel")

    reply, _ = world.chat(answer)

    assert reply == CANCELLED_REPLY
    assert world.job(printer).status is JobStatus.CANCELLED
    assert world.interactions.get(printer_q.id).status is InteractionStatus.CANCELLED and world.interactions.get(printer_q.id).response is None
    assert world.job(laptop).status is JobStatus.AWAITING_INPUT
    assert world.interactions.get(laptop_q.id).status is InteractionStatus.PENDING
    assert len(world.jobs.list()) == 2 and len(world.interactions_of(laptop)) == 1 and len(world.interactions_of(printer)) == 1


def test_naming_one_in_the_same_breath_needs_no_question_first(world):
    laptop, laptop_q, printer, printer_q = park_two(world)

    reply, _ = world.chat("cancel the laptop one")

    assert reply == CANCELLED_REPLY
    assert world.job(laptop).status is JobStatus.CANCELLED and world.job(printer).status is JobStatus.AWAITING_INPUT


@pytest.mark.parametrize(
    "answer,dropped",
    [("the first one", "laptop"), ("the older one", "laptop"), ("the latest", "printer"), ("the second one", "printer"), ("the newer one", "printer")],
)
def test_which_can_also_be_answered_by_order(world, answer, dropped):
    laptop, _, printer, _ = park_two(world)
    world.chat("cancel")

    reply, _ = world.chat(answer)

    assert reply == CANCELLED_REPLY
    gone, kept = (laptop, printer) if dropped == "laptop" else (printer, laptop)
    assert world.job(gone).status is JobStatus.CANCELLED and world.job(kept).status is JobStatus.AWAITING_INPUT


def test_cancelling_the_named_one_leaves_the_other_for_a_plain_cancel(world):
    laptop, _, printer, _ = park_two(world)
    world.chat("cancel")
    world.chat("the printer")

    reply, _ = world.chat("cancel")  # only one is left now: no need to ask

    assert reply == CANCELLED_REPLY and world.job(laptop).status is JobStatus.CANCELLED
    assert world.job(printer).status is JobStatus.CANCELLED


@pytest.mark.parametrize("answer", ["hello", "thanks", "robotics", "the meeting", "both of them", "2500 AED", "I'm not sure"])
def test_an_answer_that_does_not_tell_them_apart_cancels_nothing(world, answer):
    park_two(world)
    world.chat("cancel")
    before = snapshot(world)

    world.chat(answer)

    assert all(j.status is JobStatus.AWAITING_INPUT or j.status is JobStatus.PENDING for j in world.jobs.list())
    assert [j.status for j in world.jobs.list()].count(JobStatus.CANCELLED) == 0
    if answer != "2500 AED":  # (that one is a plausible answer to a question that is still open, and is taken as one)
        assert snapshot(world) == before


def test_naming_something_that_is_neither_cancels_nothing(world):
    park_two(world)
    before = snapshot(world)

    for message in ("cancel my 3pm meeting", "stop by the shop on the way", "forget the meeting"):
        outcome = world.resolver.handle(message, world.conversation, may_be_answer=False)  # (a digit could be an answer: that is not this)
        assert outcome.kind is not OutcomeKind.CANCELLED, message
    assert snapshot(world) == before


def test_the_question_is_only_open_for_a_while(world):
    laptop, _, printer, _ = park_two(world)
    world.chat("cancel")
    world.clock.advance(minutes=31)

    world.chat("the printer")  # far too late to be an answer to "which?"

    assert world.job(printer).status is not JobStatus.CANCELLED


def test_a_restart_forgets_the_question_and_never_a_task(world):
    laptop, _, printer, _ = park_two(world)
    world.chat("cancel")
    restarted = world.reopen(plans=[CANONICAL_PLAN])

    restarted.chat("the printer")
    assert restarted.job(printer).status is JobStatus.AWAITING_INPUT and restarted.job(laptop).status is JobStatus.AWAITING_INPUT

    reply, _ = restarted.chat("cancel")  # asked again, as if for the first time
    assert re.fullmatch(WHICH, reply)
    reply, _ = restarted.chat("the printer")
    assert reply == CANCELLED_REPLY and restarted.job(printer).status is JobStatus.CANCELLED
    again = restarted.reopen(plans=[CANONICAL_PLAN])
    assert again.job(printer).status is JobStatus.CANCELLED and again.job(laptop).status is JobStatus.AWAITING_INPUT


def test_cancelling_uses_up_no_question_and_makes_no_task(world):
    laptop, laptop_q, printer, _ = park_two(world)
    world.chat("cancel")
    world.chat("the printer")

    assert len(world.jobs.list()) == 2 and len(world.interactions_of(laptop)) == 1  # the other task still has its whole budget
    reply, _ = world.chat("I haven't decided my budget yet")  # one unusable attempt: its question 1 of 2 ...
    assert reply == f"I still need that to get started: {BUDGET_QUESTION}"  # ... and a second is still allowed
    world.chat("2500 AED")
    world.tick()
    assert world.job(laptop).status is JobStatus.COMPLETED and len(world.jobs.list()) == 2


def test_a_task_that_was_only_just_answered_counts_as_one_the_user_could_mean(world):
    _, laptop, _ = world.park()
    world.chat("2500 AED")  # answered, about to be planned
    _, printer, _ = world.park(PRINTER_REQUEST)

    reply, _ = world.chat("cancel")

    assert re.fullmatch(WHICH, reply)
    assert world.job(laptop).status is JobStatus.PENDING and world.job(printer).status is JobStatus.AWAITING_INPUT


def test_a_stale_waiting_task_still_counts_as_waiting(world):
    laptop, _, printer, _ = park_two(world)
    world.clock.advance(hours=2)  # neither question was surfaced recently

    reply, _ = world.chat("cancel")

    # (the ordinary "waiting on you" notice for questions not surfaced lately follows it, as it always does)
    assert re.match(WHICH, reply, re.DOTALL)
    assert world.job(laptop).status is JobStatus.AWAITING_INPUT and world.job(printer).status is JobStatus.AWAITING_INPUT


def test_other_conversations_tasks_are_never_offered_or_cancelled(world):
    laptop, _, printer, _ = park_two(world)
    other = world.other_conversation()

    reply, _ = world.chat("cancel", conversation=other)

    assert not re.fullmatch(WHICH, reply) and reply != CANCELLED_REPLY
    assert world.job(laptop).status is JobStatus.AWAITING_INPUT and world.job(printer).status is JobStatus.AWAITING_INPUT


def test_three_waiting_tasks_are_all_offered_and_one_can_be_named(world):
    laptop, _, printer, _ = park_two(world)
    _, monitor, _ = world.park("Research the best monitor for my design work.")

    reply, _ = world.chat("cancel")
    assert re.fullmatch(WHICH, reply) and reply.count("“") == 3

    world.chat("the monitor")
    assert world.job(monitor).status is JobStatus.CANCELLED
    assert world.job(laptop).status is JobStatus.AWAITING_INPUT and world.job(printer).status is JobStatus.AWAITING_INPUT


# --- 3. the clarification limit ------------------------------------------------------------------------------------------------------


def test_reaching_the_limit_says_so_immediately_and_stops_the_task(world):
    _, job_id, _ = world.park()
    world.chat("I haven't decided my budget yet")  # question 1 answered unusably; question 2 opened

    reply, _ = world.chat("still deciding my budget")  # a third question is not allowed

    assert reply.startswith("I'm sorry") and "stopped" in reply
    assert "continuing" not in reply and "passed your answer on" not in reply and not reply.startswith(THANKS)
    no_internals(reply)
    job = world.job(job_id)  # already terminal - not "later, on a tick"
    assert job.status is JobStatus.FAILED and "more information than the limit" in job.error
    assert len(world.interactions_of(job_id)) == 2 and world.planner.calls == [] and world.log == []
    assert len(world.jobs.list()) == 1


def test_nothing_runs_later_and_the_terminal_state_survives_a_restart(world):
    _, job_id, _ = world.park()
    world.chat("I haven't decided my budget yet")
    world.chat("still deciding my budget")
    failed = world.job(job_id)

    world.tick(5)
    restarted = world.reopen(plans=[CANONICAL_PLAN])
    restarted.tick(5)

    assert world.job(job_id) == failed and restarted.job(job_id) == failed
    assert restarted.job(job_id).status is JobStatus.FAILED
    assert world.planner.calls == [] and restarted.planner.calls == [] and world.log == [] and restarted.log == []
    assert restarted.interactions.list_pending() == [] and len(restarted.interactions_of(job_id)) == 2


def test_the_limits_themselves_are_unchanged():
    from app.orchestration.interactions import MAX_INPUT_QUESTIONS_PER_JOB, MAX_INTERACTIONS_PER_JOB

    assert (MAX_INPUT_QUESTIONS_PER_JOB, MAX_INTERACTIONS_PER_JOB) == (2, 3)


def test_the_first_unusable_answer_is_still_asked_again_not_stopped(world):
    _, job_id, _ = world.park()

    reply, _ = world.chat("I haven't decided my budget yet")

    assert reply == f"I still need that to get started: {BUDGET_QUESTION}"
    assert world.job(job_id).status is JobStatus.AWAITING_INPUT


def test_a_task_that_the_limit_stopped_can_be_asked_for_again_from_scratch(world):
    _, first, _ = world.park()
    world.chat("I haven't decided my budget yet")
    world.chat("still deciding my budget")

    _, second = world.chat(REQUEST)

    assert second is not None and second != first
    assert world.job(second).status is JobStatus.AWAITING_INPUT and len(world.jobs.list()) == 2


def test_without_the_dialogue_the_engine_still_fails_the_job_on_its_tick_as_in_slice_4(tmp_path):
    plain = World(tmp_path, plans=[CANONICAL_PLAN])
    _, job_id, _ = plain.park()
    plain.chat("I haven't decided my budget yet")
    plain.tick()
    second = plain.interactions_of(job_id)[1]
    plain.interactions.mark_presented(second.id, plain.conversation)
    plain.chat("still deciding my budget")
    assert plain.job(job_id).status is JobStatus.PENDING  # not stopped in the reply: nothing changed there
    plain.tick()
    assert plain.job(job_id).status is JobStatus.FAILED


# --- 4. a stale answer ------------------------------------------------------------------------------------------------------------------


def test_a_stale_answer_is_not_applied_and_the_user_is_told_it_needs_answering_again(world):
    _, job_id, question = world.park()
    before = world.job(job_id)
    world.clock.advance(hours=3)

    reply, other = world.chat("2500 AED")

    assert other is None and world.chat_provider.calls == 0
    assert reply.startswith("It's been a while since I asked this")
    assert "haven't used that answer yet" in reply and BUDGET_QUESTION in reply and "tell me again" in reply
    no_internals(reply)
    again = world.interactions.get(question.id)
    assert again.status is InteractionStatus.PENDING and again.response is None
    assert again.version == question.version and again.id == question.id and again.code == question.code
    assert len(world.interactions_of(job_id)) == 1 and len(world.jobs.list()) == 1
    assert world.job(job_id) == before and world.constraints(job_id) == [] and world.planner.calls == []
    assert again.presented_at != question.presented_at  # put in front of the user again


def test_the_next_fresh_answer_goes_through_the_normal_path(world):
    _, job_id, question = world.park()
    world.clock.advance(hours=3)
    world.chat("2500 AED")

    reply, _ = world.chat("about 2500 AED")

    assert reply.startswith(THANKS)
    assert world.interactions.get(question.id).response == "about 2500 AED"
    assert world.interactions.get(question.id).status is InteractionStatus.RESOLVED
    world.tick()
    assert world.job(job_id).status is JobStatus.COMPLETED and len(world.jobs.list()) == 1
    assert len(world.interactions_of(job_id)) == 1


def test_after_the_reminder_an_unusable_answer_is_asked_again_as_in_slice_4_1(world):
    _, job_id, question = world.park()
    world.clock.advance(hours=3)
    world.chat("2500 AED")

    reply, _ = world.chat("I haven't decided my budget yet")  # plausible, not usable: the existing re-ask

    assert reply == f"I still need that to get started: {BUDGET_QUESTION}"
    assert world.interactions.get(question.id).status is InteractionStatus.RESOLVED
    assert len(world.interactions_of(job_id)) == 2 and len(world.jobs.list()) == 1


@pytest.mark.parametrize("message", ["hello", "thanks", "what's up?", "Hey Jarvis", "okay", "cool"])
def test_chatter_after_a_long_gap_consumes_and_reminds_of_nothing(world, message):
    _, job_id, question = world.park()
    world.clock.advance(hours=3)
    before = world.job(job_id)

    reply, _ = world.chat(message)

    assert reply.startswith("an ordinary chat reply") and "been a while since I asked" not in reply
    assert world.interactions.get(question.id).response is None and world.interactions.get(question.id).status is InteractionStatus.PENDING
    assert world.job(job_id) == before and len(world.interactions_of(job_id)) == 1


def test_a_stale_answer_never_creates_a_second_task_or_question_however_often_it_is_sent(world):
    _, job_id, question = world.park()
    world.clock.advance(hours=3)
    world.chat("2500 AED")
    world.clock.advance(hours=3)  # and stale again
    reply, _ = world.chat("2500 AED")

    assert reply.startswith("It's been a while since I asked this")
    assert len(world.jobs.list()) == 1 and len(world.interactions_of(job_id)) == 1
    assert world.interactions.get(question.id).version == question.version


def test_an_objective_that_can_no_longer_be_completed_stops_the_job_at_once_too(world):
    _, job_id, question = world.park()
    world.chat("2500 AED")
    job = world.job(job_id)
    job.context = {**job.context, OBJECTIVE_KEY: {k: v for k, v in job.context[OBJECTIVE_KEY].items() if k != "constraints"}}
    world.jobs.save(job)

    outcome = world.engine.follow_up(job_id)

    assert outcome.stopped is True and outcome.question is None
    failed = world.job(job_id)
    assert failed.status is JobStatus.FAILED and "could not continue" in failed.error.lower()
    world.tick(3)
    assert world.job(job_id) == failed and world.planner.calls == []


def test_a_job_that_is_ready_is_never_touched_by_the_follow_up_check(world):
    _, job_id, _ = world.park()
    world.chat("2500 AED")
    before = world.job(job_id)

    outcome = world.engine.follow_up(job_id)

    assert outcome.stopped is False and outcome.question is None and world.job(job_id) == before


def test_a_question_past_its_lifetime_is_not_a_task_the_user_could_mean(world):
    _, old_job, old_q = world.park()
    _, new_job, _ = world.park(PRINTER_REQUEST)
    past = (world.clock() - timedelta(hours=1)).isoformat()
    world.jobs._conn.execute("UPDATE interactions SET expires_at = ? WHERE id = ?", (past, old_q.id))
    world.jobs._conn.commit()  # expired; the lifecycle has not closed it yet

    reply, _ = world.chat("cancel")

    assert reply == CANCELLED_REPLY and world.job(new_job).status is JobStatus.CANCELLED  # one to ask about: none needed
    assert world.job(old_job).status is JobStatus.AWAITING_INPUT


def test_a_task_that_is_no_longer_waiting_on_its_question_is_not_offered(world):
    _, first, first_q = world.park()
    _, second, _ = world.park(PRINTER_REQUEST)
    job = world.job(first)
    job.status = JobStatus.PAUSED  # its question is still pending, but the task is not waiting on it
    world.jobs.save(job)

    reply, _ = world.chat("cancel")

    assert reply == CANCELLED_REPLY and world.job(second).status is JobStatus.CANCELLED
    assert world.job(first).status is JobStatus.PAUSED


def test_a_bare_answer_picks_only_among_the_tasks_that_were_offered_but_a_named_cancel_may_name_any(world):
    laptop, _, printer, _ = park_two(world)
    world.chat("cancel")  # offers the laptop and the printer
    _, monitor, _ = world.park("Research the best monitor for my design work.")

    world.chat("the monitor")  # not on offer: a bare answer does not reach it
    assert world.job(monitor).status is JobStatus.AWAITING_INPUT

    reply, _ = world.chat("cancel the monitor")  # but naming it outright is a cancel of it
    assert reply == CANCELLED_REPLY and world.job(monitor).status is JobStatus.CANCELLED
    assert world.job(laptop).status is JobStatus.AWAITING_INPUT and world.job(printer).status is JobStatus.AWAITING_INPUT


def test_the_question_is_forgotten_once_it_has_been_answered(world):
    park_two(world)
    world.chat("cancel")
    assert world.resolver._choices

    world.chat("the printer")

    assert world.resolver._choices == {}


def test_second_only_means_something_when_there_are_exactly_two(world):
    laptop, _, printer, _ = park_two(world)
    _, monitor, _ = world.park("Research the best monitor for my design work.")
    world.chat("cancel")
    before = snapshot(world)

    world.chat("the second one")  # which of three? not said: nothing is guessed

    assert snapshot(world) == before
    world.chat("the first one")  # "first" is still the oldest
    assert world.job(laptop).status is JobStatus.CANCELLED
