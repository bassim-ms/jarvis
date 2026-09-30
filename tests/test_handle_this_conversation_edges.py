"""
Handle This, Slice 5 - the edges of the conversation around a question.

The behaviour itself is in tests/test_handle_this_conversation.py; this file pins
the boundaries a race, an odd state or a faulty helper could push it across:
retries when the Job moved between reading and writing, a lost race that is only
a repeat, a dialogue that breaks, a question that expired before the lifecycle
closed it, and the engine's own refusals.
"""

import pytest

from app.handle_this.dialogue import ClarificationDialogue
from app.orchestration.base import (
    CLARIFICATION_STATE_KEY,
    CLARIFICATIONS_KEY,
    OBJECTIVE_KEY,
    JobStatus,
    Plan,
    PlanStep,
    StaleJobError,
    StepStatus,
)
from app.orchestration.interaction_resolver import InteractionResolver, OutcomeKind
from app.orchestration.interactions import InteractionStatus, InteractionType
from test_handle_this_clarification import BUDGET_QUESTION, REQUEST
from test_handle_this_conversation import (
    CANCELLED_REPLY,
    MULTI_REQUEST,
    PRINTER_REQUEST,
    REVISED_REPLY,
    THANKS,
    Talk,
)
from test_handle_this_integration import CANONICAL_PLAN


@pytest.fixture
def world(tmp_path):
    return Talk(tmp_path, plans=[CANONICAL_PLAN])


def bump(world, job_id):
    """Something else writes the Job (the engine's tick, say): its version moves on."""
    job = world.job(job_id)
    world.jobs.save(job)


# --- the Job moved between reading it and writing it -----------------------------------------------------------------------


def test_a_cancel_is_retried_when_the_job_moved_in_between(world, monkeypatch):
    _, job_id, interaction = world.park()
    real = world.engine.cancel_job
    attempts = []

    def moved_first(job_id_, *, expected_job_version, **kwargs):
        attempts.append(expected_job_version)
        if len(attempts) == 1:
            bump(world, job_id_)  # the write below now sees a newer Job than the one this attempt read
        return real(job_id_, expected_job_version=expected_job_version, **kwargs)

    monkeypatch.setattr(world.engine, "cancel_job", moved_first)
    outcome = world.resolver.handle("cancel", world.conversation)

    assert outcome.kind is OutcomeKind.CANCELLED and outcome.reply == CANCELLED_REPLY
    assert len(attempts) == 2 and attempts[1] > attempts[0]  # read again, not blindly repeated
    assert world.job(job_id).status is JobStatus.CANCELLED
    assert world.interactions.get(interaction.id).status is InteractionStatus.CANCELLED


def test_a_cancel_that_keeps_losing_the_race_gives_up_safely(world, monkeypatch):
    _, job_id, interaction = world.park()
    calls = []

    def always_stale(job_id_, **kwargs):
        calls.append(1)
        raise StaleJobError("moved")

    monkeypatch.setattr(world.engine, "cancel_job", always_stale)
    outcome = world.resolver.handle("cancel", world.conversation)

    assert outcome.kind is OutcomeKind.REJECTED and len(calls) == 3  # bounded
    assert world.job(job_id).status is JobStatus.AWAITING_INPUT
    assert world.interactions.get(interaction.id) == interaction


def test_a_cancel_that_lost_only_to_an_identical_cancel_is_still_a_success(world, monkeypatch):
    _, job_id, _ = world.park()
    real = world.engine.cancel_job

    def other_copy_won(job_id_, *, expected_job_version, **kwargs):
        real(job_id_, expected_job_version=expected_job_version, **kwargs)  # the other copy of the message
        raise StaleJobError("this copy lost")

    monkeypatch.setattr(world.engine, "cancel_job", other_copy_won)
    outcome = world.resolver.handle("never mind", world.conversation)

    assert outcome.kind is OutcomeKind.CANCELLED and outcome.reply == CANCELLED_REPLY
    assert world.job(job_id).status is JobStatus.CANCELLED


def test_a_cancel_of_something_that_finished_meanwhile_says_so_and_changes_nothing(world, monkeypatch):
    _, job_id, _ = world.park()
    world.chat("2500 AED")
    world.tick()
    done = world.job(job_id)
    assert done.status is JobStatus.COMPLETED
    job = world.jobs.list()[0]

    outcome = world.resolver._cancel(job)  # a cancel that read the Job just before it finished

    assert outcome.kind is OutcomeKind.REJECTED and "already finished" in outcome.reply
    assert world.job(job_id) == done


def test_a_revision_is_retried_when_the_job_moved_in_between(world, monkeypatch):
    _, job_id, _ = world.park()
    world.chat("2500 AED")
    real = world.engine.revise_answer
    attempts = []

    def moved_first(job_id_, topic, answer, *, expected_job_version):
        attempts.append(expected_job_version)
        if len(attempts) == 1:
            bump(world, job_id_)
        return real(job_id_, topic, answer, expected_job_version=expected_job_version)

    monkeypatch.setattr(world.engine, "revise_answer", moved_first)
    outcome = world.resolver.handle("Actually, make that 3000 AED.", world.conversation)

    assert outcome.kind is OutcomeKind.REVISED and outcome.reply == REVISED_REPLY
    assert len(attempts) == 2 and attempts[1] > attempts[0]
    assert [c["text"] for c in world.constraints(job_id)] == ["3000 AED"]


def test_a_revision_that_keeps_losing_the_race_gives_up_safely(world, monkeypatch):
    _, job_id, _ = world.park()
    world.chat("2500 AED")
    before = world.job(job_id)
    calls = []

    def always_stale(*args, **kwargs):
        calls.append(1)
        raise StaleJobError("moved")

    monkeypatch.setattr(world.engine, "revise_answer", always_stale)
    outcome = world.resolver.handle("Actually, make that 3000 AED.", world.conversation)

    assert outcome.kind is OutcomeKind.REJECTED and len(calls) == 3
    assert world.job(job_id) == before


def test_a_revision_the_engine_refuses_falls_back_to_ordinary_conversation(world, monkeypatch):
    from app.orchestration.interactions import InteractionJobStateError

    _, job_id, _ = world.park()
    world.chat("2500 AED")

    def refuses(*args, **kwargs):
        raise InteractionJobStateError("no")

    monkeypatch.setattr(world.engine, "revise_answer", refuses)
    assert world.resolver.handle("Actually, make that 3000 AED.", world.conversation).kind is OutcomeKind.NOT_HANDLED


def test_a_change_is_never_taken_from_a_message_the_router_called_a_command_or_an_explicit_search(world):
    _, job_id, _ = world.park()
    world.chat("2500 AED")
    before = world.job(job_id)

    outcome = world.resolver.handle(
        "Actually, make that 3000 AED.", world.conversation, may_be_answer=False, may_answer_if_recognised=False
    )
    assert outcome.kind is OutcomeKind.NOT_HANDLED and world.job(job_id) == before

    # ...but one the router only called a search because of a freshness word still can be one
    outcome = world.resolver.handle(
        "Actually, make that 3000 AED right now", world.conversation, may_be_answer=False, may_answer_if_recognised=True
    )
    assert outcome.kind is OutcomeKind.REVISED


def test_only_an_answer_that_was_actually_accepted_can_be_changed(world):
    _, job_id, first = world.park()
    world.chat("I haven't decided my budget yet")  # answered, but unusable: it was asked again
    second = world.interactions_of(job_id)[1]

    reply, _ = world.chat("Actually, make that 3000 AED.")  # nothing to change yet: it is the answer to the new question

    assert reply.startswith(THANKS)
    assert world.interactions.get(second.id).status is InteractionStatus.RESOLVED
    assert world.job(job_id).context.get(CLARIFICATION_STATE_KEY) is None or (
        "revisions" not in world.job(job_id).context[CLARIFICATION_STATE_KEY]
    )


def test_a_time_is_not_a_change_to_a_budget_answer_and_the_other_way_round(world):
    _, job_id, _ = world.park()
    world.chat("2500 AED")
    before = world.job(job_id)

    reply, _ = world.chat("Actually, make that tomorrow at 5pm")  # only a budget was answered

    assert reply.startswith("an ordinary chat reply") and world.job(job_id) == before


# --- a lost race that is only a repeat --------------------------------------------------------------------------------------------


def test_the_same_answer_arriving_after_it_was_applied_is_recognised_as_a_repeat(world):
    _, job_id, interaction = world.park()
    job = world.job(job_id)
    world.chat("2500 AED")  # the first copy wins

    late = world.resolver._apply(
        interaction, job, approved=True, response="2500 AED", via="chat", conversation_id=world.conversation
    )  # the second copy, still holding what it read before

    assert late.kind is OutcomeKind.RESOLVED and late.reply == "Got it - I already have that."
    assert world.interactions.get(interaction.id).response == "2500 AED" and world.interactions.get(interaction.id).version == 1
    assert len(world.interactions_of(job_id)) == 1


def test_a_different_answer_arriving_late_is_not_mistaken_for_a_repeat(world):
    _, job_id, interaction = world.park()
    job = world.job(job_id)
    world.chat("2500 AED")

    late = world.resolver._apply(
        interaction, job, approved=True, response="9999 AED", via="chat", conversation_id=world.conversation
    )

    assert late.kind is OutcomeKind.REJECTED and "changed just now" in late.reply
    assert world.interactions.get(interaction.id).response == "2500 AED"


def test_without_a_dialogue_a_lost_race_is_reported_as_before(tmp_path):
    from test_handle_this_clarification import World

    plain = World(tmp_path, plans=[CANONICAL_PLAN])
    _, job_id, interaction = plain.park()
    job = plain.job(job_id)
    plain.chat("2500 AED")

    late = plain.resolver._apply(interaction, job, approved=True, response="2500 AED", via="chat")

    assert late.kind is OutcomeKind.REJECTED


# --- a helper that breaks must not lose or half-apply a message --------------------------------------------------------------------


class BrokenDialogue(ClarificationDialogue):
    def cancel_requested(self, message):
        raise RuntimeError("bug")


def test_a_dialogue_that_raises_leaves_the_message_to_ordinary_conversation(world):
    _, job_id, interaction = world.park()
    world.resolver.dialogue = BrokenDialogue()
    before = world.job(job_id)

    outcome = world.resolver.handle("never mind", world.conversation)
    reply, _ = world.chat("hello")

    assert outcome.kind is OutcomeKind.NOT_HANDLED and reply.startswith("an ordinary chat reply")
    assert world.job(job_id) == before and world.interactions.get(interaction.id) == interaction


# --- which question a "cancel" is about -------------------------------------------------------------------------------------------


def test_a_cancel_never_drops_a_question_another_part_of_the_system_asked(world):
    job = world.jobs.create("some other work", conversation_id=world.conversation)
    job.status = JobStatus.AWAITING_INPUT
    world.jobs.save(job)
    other = world.interactions.create(
        job.id, InteractionType.INPUT, "Which folder?", detail={"origin": "someone-else"}, now=world.clock()
    )
    world.interactions.mark_presented(other.id, world.conversation, now=world.clock())

    assert world.resolver._handle_cancel(world.conversation) is None  # not a question Handle This asked
    outcome = world.resolver.handle("cancel", world.conversation)

    # (a question with no gate is, as ever, answerable by any armed reply - that is not this feature)
    assert outcome.kind is not OutcomeKind.CANCELLED
    assert world.job(job.id).status is not JobStatus.CANCELLED


def test_a_cancel_from_another_conversation_does_not_drop_an_answered_task(world):
    _, job_id, _ = world.park()
    world.chat("2500 AED")

    outcome = world.resolver.handle("cancel", world.other_conversation())

    assert outcome.kind is OutcomeKind.NOT_HANDLED and world.job(job_id).status is JobStatus.PENDING


def test_a_cancel_long_after_the_answer_is_only_a_word(world):
    _, job_id, _ = world.park()
    world.chat("2500 AED")
    world.clock.advance(minutes=31)

    outcome = world.resolver.handle("cancel", world.conversation)

    assert outcome.kind is OutcomeKind.NOT_HANDLED and world.job(job_id).status is JobStatus.PENDING


def test_a_change_goes_to_the_task_most_recently_answered(tmp_path):
    world = Talk(tmp_path, plans=[CANONICAL_PLAN])
    _, older, _ = world.park()
    world.chat("2500 AED")
    _, newer, _ = world.park(PRINTER_REQUEST)
    world.chat("500 AED")

    world.chat("Actually, make that 700 AED.")

    assert [c["text"] for c in world.constraints(newer)] == ["700 AED"]
    assert [c["text"] for c in world.constraints(older)] == []  # its answer is folded in later, by the engine
    assert world.job(older).context.get(CLARIFICATION_STATE_KEY) is None


# --- a stale question, at the edges ------------------------------------------------------------------------------------------------------


def test_a_question_past_its_lifetime_but_not_yet_closed_is_neither_answered_nor_reminded(world):
    _, job_id, interaction = world.park()
    world.clock.advance(days=8)  # expired; the lifecycle has not run yet

    reply, _ = world.chat("2500 AED")

    assert reply.startswith("an ordinary chat reply")
    assert world.interactions.get(interaction.id).response is None
    assert world.job(job_id).status is JobStatus.AWAITING_INPUT


def test_a_stale_question_is_not_reminded_for_something_that_is_a_question(world):
    _, job_id, interaction = world.park()
    world.clock.advance(minutes=31)

    reply, _ = world.chat("is 2500 AED enough?")

    assert reply.startswith("an ordinary chat reply") and "been a while" not in reply


def test_with_two_stale_questions_the_newest_is_shown_and_armed(tmp_path):
    world = Talk(tmp_path, plans=[CANONICAL_PLAN])
    _, _, older = world.park()
    _, _, newer = world.park(PRINTER_REQUEST)
    world.clock.advance(minutes=31)

    reply, _ = world.chat("500 AED")

    assert reply.startswith("It's been a while") and BUDGET_QUESTION in reply
    # the newest was shown (and armed again) by the reminder itself; the other one only afterwards,
    # by the ordinary "waiting on you" notice
    assert world.interactions.get(newer.id).presented_at != newer.presented_at
    assert world.interactions.get(newer.id).presented_at < world.interactions.get(older.id).presented_at
    assert world.interactions.get(older.id).response is None and world.interactions.get(newer.id).response is None


def test_a_resolver_without_an_answer_gate_never_reminds(world):
    _, job_id, interaction = world.park()
    world.clock.advance(minutes=31)
    ungated = InteractionResolver(
        world.interactions, world.jobs, world.engine, clock=world.clock, dialogue=ClarificationDialogue()
    )

    assert ungated.handle("2500 AED", world.conversation).kind is OutcomeKind.NOT_HANDLED
    assert world.interactions.get(interaction.id).response is None


# --- the engine's own refusals and guards ------------------------------------------------------------------------------------------------


def test_the_engine_refuses_to_change_an_answer_on_a_finished_job(world):
    _, job_id, _ = world.park()
    world.chat("2500 AED")
    world.tick()
    done = world.job(job_id)

    from app.orchestration.interactions import InteractionJobStateError

    with pytest.raises(InteractionJobStateError):
        world.engine.revise_answer(job_id, "budget", "3000 AED", expected_job_version=done.version)
    assert world.job(job_id) == done


def _started(world, job_id, status, steps=(("s1", StepStatus.PENDING),), pending=()):
    job = world.fold_and_start(job_id, [(i, "web_search", s, []) for i, s in steps])
    job.status = status
    job.pending_step_ids = list(pending)
    world.jobs.save(job)
    return world.job(job_id)


@pytest.mark.parametrize(
    "status,pending",
    [(JobStatus.PAUSED, ()), (JobStatus.AWAITING_CONFIRMATION, ("s1",)), (JobStatus.RUNNING, ("s1",))],
)
def test_a_plan_is_only_dropped_for_a_running_job_that_is_not_waiting_on_a_confirmation(world, status, pending):
    _, job_id, _ = world.park()
    world.chat("2500 AED")
    job = _started(world, job_id, status, pending=pending)

    revised = world.engine.revise_answer(job_id, "budget", "3000 AED", expected_job_version=job.version)

    assert revised.plan is not None and revised.status is status and revised.total_step_count == 1
    assert world.job(job_id).plan is not None


def test_two_different_answers_can_each_be_changed_and_both_changes_are_kept(world):
    _, job_id, _ = world.park(MULTI_REQUEST)
    world.chat("under 900 dollars")
    world.chat("tomorrow morning")
    job = world.job(job_id)

    job = world.engine.revise_answer(job_id, "budget", "1200 dollars", expected_job_version=job.version)
    job = world.engine.revise_answer(job_id, "when", "next friday", expected_job_version=job.version)

    assert job.context[CLARIFICATION_STATE_KEY]["revisions"] == {"budget": "1200 dollars", "when": "next friday"}
    assert [c["text"] for c in world.constraints(job_id)] == ["1200 dollars"]
    assert [c["text"] for c in world.constraints(job_id, "timeframe")] == ["next friday"]


def test_the_planner_is_given_one_answer_per_topic_even_after_an_unusable_first_attempt(world):
    _, job_id, _ = world.park()
    world.chat("I haven't decided my budget yet")  # unusable: asked again
    world.chat("2500 AED")
    world.chat("Actually, make that 3000 AED.")

    world.tick()

    (call,) = world.planner.calls
    assert call["context"][CLARIFICATIONS_KEY] == [{"question": BUDGET_QUESTION, "answer": "3000 AED"}]
    assert [c["text"] for c in call["context"][OBJECTIVE_KEY]["constraints"] if c["topic"] == "budget"] == ["3000 AED"]


def test_a_follow_up_is_only_opened_for_a_job_that_is_waiting_to_be_continued(world):
    _, job_id, first = world.park(MULTI_REQUEST)
    answered = world.engine.resolve_interaction(
        first.id, expected_version=first.version, expected_job_version=world.job(job_id).version,
        response="under 900 dollars", via="chat",
    )
    job = answered.job
    assert job.status is JobStatus.PENDING
    job.status = JobStatus.PAUSED
    world.jobs.save(job)

    assert world.engine.open_follow_up(job_id) is None  # paused: not this call's to continue
    assert len(world.interactions_of(job_id)) == 1

    job = world.job(job_id)
    job.status = JobStatus.PENDING
    world.jobs.save(job)
    follow_up = world.engine.open_follow_up(job_id)
    assert follow_up is not None and follow_up.detail["topic"] == "when"


def test_a_follow_up_past_the_question_limit_stops_the_job_at_once_and_says_so(world):
    _, job_id, first = world.park()
    world.chat("I haven't decided my budget yet")  # question 2 opened
    second = world.interactions_of(job_id)[1]
    world.engine.resolve_interaction(
        second.id, expected_version=second.version, expected_job_version=world.job(job_id).version,
        response="still deciding my budget", via="chat",
    )
    job = world.job(job_id)
    assert job.status is JobStatus.PENDING

    stopped = world.engine.follow_up(job_id)  # a third question would exceed the limit
    assert stopped.stopped is True and stopped.question is None
    failed = world.job(job_id)
    assert failed.status is JobStatus.FAILED and "more information than the limit" in failed.error
    world.tick(3)
    assert world.job(job_id) == failed and world.planner.calls == [] and world.log == []  # nothing runs later
