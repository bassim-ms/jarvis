"""
Focus never takes a message a waiting interaction owns.

If Jarvis has put a question (INPUT) or a confirmation to the user in this conversation and is waiting on it, the next message belongs to
the interaction flow - even when the resolver itself does not take it (a SEARCH-routed sentence its answer gate does not recognise).
Real InteractionResolver / JobEngine / stores (the resolver tests' own World), the production capability router, the real FocusService on
real task and World Model stores.
"""

from datetime import timedelta

import pytest

from app.capability import DeterministicCapabilityRouter
from app.core import JarvisCore
from app.focus import FocusService
from app.handle_this.routing import ObjectiveAwareCapabilityRouter
from app.memory.sqlite_store import SQLiteMemoryStore
from app.orchestration.base import JobStatus
from app.orchestration.interaction_resolver import InteractionResolver
from app.orchestration.interactions import InteractionStatus
from app.orchestration.tools import ConfirmJobTool, ListJobsTool
from app.tasks.base import TaskPriority
from app.tasks.sqlite_store import SQLiteTaskStore
from app.tools.base import PermissionLevel
from app.world_model.factory import create_default_world_model_service, create_default_world_model_store
from test_interaction_resolver import ChatProvider, World, make_tool, untouched

CONVERSATION = "c1"
ARDUINO = "I need to finish my Arduino project today."  # routed to SEARCH only because of "today"
ARDUINO_DIRECT = "I need to finish my Arduino project."
WHAT_NOW = "What should I work on right now?"
FOCUS_START = 'This sounds like your project "Arduino project". Start with: "Arduino: test the servo".'


class SpyFocus:
    """The real FocusService, recording what reached it and whether it answered."""

    def __init__(self, service):
        self.service = service
        self.reached = []
        self.answered = []

    def command_reply(self, message, conversation_id=None):
        self.reached.append(message)
        reply = self.service.command_reply(message, conversation_id)
        if reply is not None:
            self.answered.append(message)
        return reply


class Rig:
    def __init__(self, tmp_path, *, answer_gate=None, with_focus=True, resolver=None):
        self.world = World()
        if answer_gate is not None:
            self.world.resolver = InteractionResolver(
                self.world.interactions, self.world.jobs, self.world.engine,
                surface_window_seconds=1800, clock=self.world.clock, answer_gate=answer_gate,
            )
        self.tasks = SQLiteTaskStore(str(tmp_path / "tasks.db"))
        self.wm = create_default_world_model_service(create_default_world_model_store(str(tmp_path / "wm.db")))
        self.wm.create_project("Arduino project")
        self.tasks.create(text="Arduino: test the servo", priority=TaskPriority.HIGH, due_date="2026-09-29")
        self.focus = SpyFocus(FocusService(tasks=self.tasks, calendar=None, world_model=self.wm, clock=self.world.clock))
        self.world.registry.register(ListJobsTool(store=self.world.jobs))
        self.world.registry.register(ConfirmJobTool(store=self.world.jobs))
        self.world.registry.register(make_tool("create_job", PermissionLevel.SAFE_ACTION, self.world.log))  # a stand-in: the TASK path calls it
        self.provider = ChatProvider()
        self.memory = SQLiteMemoryStore(":memory:")
        self.core = JarvisCore(
            provider=self.provider,
            memory=self.memory,
            tools=self.world.registry,
            capability_router=ObjectiveAwareCapabilityRouter(DeterministicCapabilityRouter()),
            interaction_resolver=resolver or self.world.resolver,
            focus=self.focus if with_focus else None,
        )
        self.conversation = self.memory.create_conversation()

    def say(self, message):
        try:
            return self.core.handle_chat(message, self.conversation)[0]
        except Exception as exc:  # noqa: BLE001 - e.g. a SEARCH with no search tool: whatever the existing path does
            return exc

    def input_waiting(self, prompt="Which Arduino project do you mean?", present=True):
        job, interaction = self.world.input_job(conversation_id=self.conversation, prompt=prompt)
        if present:
            self.world.present(interaction, self.conversation)
        return job, interaction

    def confirmation_waiting(self, present=True):
        job, interaction = self.world.confirm_job(conversation_id=self.conversation)
        if present:
            self.world.present(interaction, self.conversation)
        return job, interaction

    def observable(self, message):
        """Everything a message visibly did: the reply (or the error), the model calls, the tools run, the interaction states."""
        reply = self.say(message)
        shown = (type(reply).__name__, str(reply)) if isinstance(reply, Exception) else reply
        states = sorted(i.status.value for i in self.world.interactions.list_pending())  # (ids differ between worlds)
        return shown, self.provider.calls, [name for name, _ in self.world.registry.executed], states


def pair(tmp_path, setup, message, **options):
    """The same world with and without focus: a message focus does not take must be handled exactly as it was before focus existed."""
    results = []
    for name in ("with", "without"):
        directory = tmp_path / name
        directory.mkdir()
        rig = Rig(directory, with_focus=name == "with", **options)
        setup(rig)
        results.append((rig, rig.observable(message)))
    return results


# ===================================================== 1. active INPUT: the interaction wins =================================================


def test_an_input_waiting_for_its_answer_takes_a_focus_looking_reply_and_focus_never_sees_it(tmp_path):
    rig = Rig(tmp_path)
    job, interaction = rig.input_waiting("Which Arduino project do you mean?")

    reply = rig.say(ARDUINO_DIRECT)

    resolved = rig.world.interactions.get(interaction.id)
    assert resolved.status is InteractionStatus.RESOLVED and resolved.response == ARDUINO_DIRECT  # the answer, exactly as typed
    assert rig.focus.reached == [] and FOCUS_START not in str(reply)
    assert rig.world.jobs.get(job.id).status is JobStatus.PENDING  # the Job resumes through normal planning, as for any answer


def test_the_users_own_example_a_sentence_with_today_also_answers_the_question_when_the_answer_gate_recognises_it(tmp_path):
    rig = Rig(tmp_path, answer_gate=lambda interaction, message: "Arduino" in message)
    _, interaction = rig.input_waiting("Which Arduino project do you mean?")

    rig.say(ARDUINO)  # SEARCH-routed (the word "today"): the resolver may answer it, but only if its gate recognises it

    assert rig.world.interactions.get(interaction.id).status is InteractionStatus.RESOLVED
    assert rig.world.interactions.get(interaction.id).response == ARDUINO
    assert rig.focus.reached == []


def test_a_search_routed_sentence_the_gate_does_not_recognise_is_still_not_focuss_it_goes_the_way_it_always_went(tmp_path):
    (rig, with_focus), (_, without) = pair(
        tmp_path, lambda r: r.input_waiting(), ARDUINO, answer_gate=lambda interaction, message: False
    )

    assert rig.focus.reached == [] and with_focus == without  # focus stood down; nothing about this message changed


@pytest.mark.parametrize("message", [ARDUINO, ARDUINO_DIRECT, WHAT_NOW])
def test_while_a_question_is_waiting_focus_changes_nothing_for_any_focus_request(tmp_path, message):
    (rig, with_focus), (_, without) = pair(tmp_path, lambda r: r.input_waiting(), message)

    assert rig.focus.reached == [] and with_focus == without


def test_the_explicit_code_form_still_answers_exactly_as_before(tmp_path):
    rig = Rig(tmp_path)
    _, interaction = rig.input_waiting()

    rig.say(f"answer {interaction.code}: {ARDUINO}")

    assert rig.world.interactions.get(interaction.id).response == ARDUINO and rig.focus.reached == []


# ===================================================== 2. no active INPUT: focus works as before =============================================


def test_with_nothing_waiting_the_same_message_is_answered_by_focus(tmp_path):
    rig = Rig(tmp_path)

    reply = rig.say(ARDUINO)

    assert reply.startswith(FOCUS_START) and rig.focus.answered == [ARDUINO] and rig.provider.calls == 0


def test_focus_is_available_again_once_the_question_has_been_answered(tmp_path):
    rig = Rig(tmp_path)
    _, interaction = rig.input_waiting()
    rig.say("Chrome")  # answers it
    assert rig.world.interactions.get(interaction.id).status is InteractionStatus.RESOLVED

    assert rig.say(ARDUINO).startswith(FOCUS_START)


def test_a_question_that_was_never_put_to_the_user_does_not_hold_focus_back(tmp_path):
    rig = Rig(tmp_path)
    rig.input_waiting(present=False)  # created, but not yet shown to this conversation: the user has not been asked anything

    assert rig.say(ARDUINO).startswith(FOCUS_START)


def test_a_question_put_to_another_conversation_does_not_hold_focus_back(tmp_path):
    rig = Rig(tmp_path)
    job, interaction = rig.world.input_job(conversation_id="elsewhere")
    rig.world.present(interaction, "elsewhere")

    assert rig.say(ARDUINO).startswith(FOCUS_START)
    assert rig.world.interactions.get(interaction.id).status is InteractionStatus.PENDING


def test_a_question_whose_window_has_passed_no_longer_holds_focus_back(tmp_path):
    rig = Rig(tmp_path)
    rig.input_waiting()
    rig.world.clock.advance(minutes=31)  # the resolver's own 30-minute window

    assert rig.say(ARDUINO).startswith(FOCUS_START)


def test_a_waiting_question_does_not_change_what_focus_says_once_it_may_speak(tmp_path):
    rig = Rig(tmp_path)
    rig.say(ARDUINO)  # nothing waiting
    first = rig.say(WHAT_NOW)

    assert first.startswith('Start with: "Arduino: test the servo".') and rig.focus.answered == [ARDUINO, WHAT_NOW]


# ===================================================== 3. active confirmation: the confirmation flow wins ====================================


def test_a_confirmation_waiting_for_its_yes_is_untouched_by_a_focus_looking_message_and_focus_stands_down(tmp_path):
    (rig, with_focus), (without_rig, without) = pair(tmp_path, lambda r: r.confirmation_waiting(), ARDUINO)

    assert rig.focus.reached == [] and with_focus == without
    (confirmation,) = [i for i in rig.world.interactions.list_pending()]
    assert confirmation.status is InteractionStatus.PENDING  # neither approved nor declined by a message that is not a yes or a no
    assert rig.world.jobs.get(confirmation.job_id).status is JobStatus.AWAITING_CONFIRMATION


def test_the_confirmation_is_still_answered_by_its_yes_through_the_existing_flow(tmp_path):
    rig = Rig(tmp_path)
    job, interaction = rig.confirmation_waiting()
    rig.say(ARDUINO)
    assert untouched(rig.world, job, interaction)  # a message that is not a yes or a no changes nothing about it

    rig.say("yes")

    assert rig.world.interactions.get(interaction.id).status is InteractionStatus.RESOLVED
    assert rig.focus.reached == []  # "yes" is a CONFIRMATION, and was never offered to focus either


def test_a_confirmation_that_was_never_put_to_the_user_does_not_hold_focus_back(tmp_path):
    rig = Rig(tmp_path)
    rig.confirmation_waiting(present=False)

    assert rig.say(ARDUINO).startswith(FOCUS_START)


# ===================================================== 4. task flows: the task flow wins =====================================================


def test_a_task_flow_waiting_on_an_answer_takes_the_reply_and_focus_stands_down(tmp_path):
    """A Job that has asked the user something (a Handle This clarification is exactly this) owns the next message."""
    rig = Rig(tmp_path)
    job, interaction = rig.input_waiting("Which Arduino project, and what should I do with it?")

    rig.say(ARDUINO_DIRECT)

    assert rig.world.interactions.get(interaction.id).response == ARDUINO_DIRECT and rig.focus.reached == []


def test_a_task_request_with_the_same_words_goes_to_the_task_flow_not_to_focus(tmp_path):
    rig = Rig(tmp_path)

    rig.say("Create a task: finish my Arduino project today")

    assert "create_job" in [name for name, _ in rig.world.registry.executed]  # the ordinary TASK path ran
    assert rig.focus.reached == []


def test_a_job_that_is_merely_running_does_not_own_the_message_and_is_not_touched_by_focus(tmp_path):
    rig = Rig(tmp_path)
    job = rig.world.jobs.create("Compile the firmware", conversation_id=rig.conversation)
    job.status = JobStatus.RUNNING
    rig.world.jobs.save(job)
    before = rig.world.jobs.get(job.id)

    reply = rig.say(ARDUINO)

    assert reply.startswith(FOCUS_START)  # nothing is waiting for an answer, so focus is free to answer
    assert rig.world.jobs.get(job.id) == before  # and the Job was not touched


# ===================================================== 5. everything else is unchanged ======================================================


def test_other_flows_keep_their_precedence_while_a_question_waits(tmp_path):
    rig = Rig(tmp_path)
    job, interaction = rig.input_waiting()

    rig.say("use tool list_jobs")
    rig.say("Search for Arduino project ideas today")
    rig.say("yes")

    assert rig.focus.reached == []  # none of them is something focus could have taken either


@pytest.mark.parametrize("message", ["What is the capital of Japan?", "My Arduino project is annoying.", "Let's do it.", "Okay, continue."])
def test_ordinary_messages_are_handled_the_same_with_and_without_focus_whether_or_not_a_question_waits(tmp_path, message):
    for waiting in (False, True):
        directory = tmp_path / f"w{waiting}"
        directory.mkdir()
        results = []
        for name in ("with", "without"):
            sub = directory / name
            sub.mkdir()
            rig = Rig(sub, with_focus=name == "with")
            if waiting:
                rig.input_waiting()
            results.append(rig.observable(message))
        assert results[0] == results[1]


# ===================================================== the resolver's question, directly ====================================================


def test_awaiting_reply_is_the_resolvers_own_definition_of_armed_and_changes_nothing():
    world = World()
    resolver = world.resolver
    assert resolver.awaiting_reply("c1") is False

    job, interaction = world.input_job(conversation_id="c1")
    assert resolver.awaiting_reply("c1") is False  # created but not yet put to the user
    world.present(interaction, "c1")
    assert resolver.awaiting_reply("c1") is True and resolver.awaiting_reply("c2") is False  # armed for c1 only
    assert untouched(world, job, interaction)  # a question only

    confirm_job, confirmation = world.confirm_job(conversation_id="c3")
    world.present(confirmation, "c3")
    assert resolver.awaiting_reply("c3") is True  # a confirmation waits on the next message too

    world.clock.advance(minutes=31)
    assert resolver.awaiting_reply("c1") is False and resolver.awaiting_reply("c3") is False  # the window has passed


def test_awaiting_reply_counts_a_background_job_surfaced_to_the_conversation():
    world = World()
    job, interaction = world.input_job(conversation_id=None)  # a background Job: answerable from any conversation
    world.present(interaction, "c1")

    assert world.resolver.awaiting_reply("c1") is True and world.resolver.awaiting_reply("c9") is False


def test_a_resolver_that_cannot_say_makes_focus_stand_down_rather_than_guess(tmp_path):
    class Opaque:
        def handle(self, *args, **kwargs):
            from types import SimpleNamespace

            return SimpleNamespace(handled=False, reply=None)

        def surface(self, conversation_id):
            return None

        def awaiting_reply(self, conversation_id):
            raise RuntimeError("cannot tell")

    rig = Rig(tmp_path, resolver=Opaque())

    reply = rig.say(WHAT_NOW)

    assert rig.focus.reached == [] and not str(reply).startswith("Start with")


def test_a_resolver_without_the_question_is_treated_the_same_way(tmp_path):
    class Old:
        def handle(self, *args, **kwargs):
            from types import SimpleNamespace

            return SimpleNamespace(handled=False, reply=None)

        def surface(self, conversation_id):
            return None

    rig = Rig(tmp_path, resolver=Old())

    assert rig.say(WHAT_NOW) and rig.focus.reached == []


def test_without_a_resolver_focus_works_as_before(tmp_path):
    rig = Rig(tmp_path)
    rig.core.interaction_resolver = None

    assert rig.say(ARDUINO).startswith(FOCUS_START)


# ===================================================== the lookup is only made for a message Focus could take ===============================


class CountingResolver(InteractionResolver):
    """The real resolver, counting how often it is asked whether something is waiting."""

    asked = 0

    def awaiting_reply(self, conversation_id):
        self.asked += 1
        return super().awaiting_reply(conversation_id)


def counting(rig):
    world = rig.world
    resolver = CountingResolver(world.interactions, world.jobs, world.engine, surface_window_seconds=1800, clock=world.clock)
    rig.core.interaction_resolver = resolver
    return resolver


@pytest.mark.parametrize(
    "message",
    [
        "use tool list_jobs",  # TOOL
        "Create a task: finish my Arduino project today",  # TASK
        "what is the status of my task?",  # TASK_STATUS
        "yes",  # CONFIRMATION
        "approve the plan",  # SELF_DEV_APPROVAL
        "self-develop: add a dark mode toggle",  # SELF_DEVELOPMENT
        "Search for Arduino project ideas today",  # an explicit SEARCH
    ],
)
def test_a_message_focus_can_never_take_costs_no_interaction_lookup(tmp_path, message):
    rig = Rig(tmp_path)
    resolver = counting(rig)

    rig.say(message)

    assert resolver.asked == 0 and rig.focus.reached == []


@pytest.mark.parametrize("message", [ARDUINO, ARDUINO_DIRECT, WHAT_NOW, "What is the capital of Japan?"])
def test_the_lookup_is_made_once_for_a_message_that_could_be_focuss(tmp_path, message):
    rig = Rig(tmp_path)
    resolver = counting(rig)

    rig.say(message)

    assert resolver.asked == 1


def test_a_message_another_door_already_claimed_costs_no_lookup_either(tmp_path):
    from types import SimpleNamespace

    rig = Rig(tmp_path)
    resolver = counting(rig)
    rig.core.reminder_operations = SimpleNamespace(command_reply=lambda message, conversation_id=None: "REMINDER ANSWER")

    assert rig.say(ARDUINO_DIRECT) == "REMINDER ANSWER" and resolver.asked == 0


def test_the_capability_check_comes_before_the_lookup_in_the_door_and_the_precedence_is_unchanged():
    import inspect

    source = inspect.getsource(JarvisCore._dispatch)
    block = source[source.index("self.focus is not None") : source.index("self.focus.command_reply")]

    assert block.index("decision.capability is Capability.DIRECT") < block.index("is_explicit_search_request") < block.index("_interaction_awaiting_reply")
    for earlier in ("memory_reply is None", "interaction_reply is None"):
        assert earlier in source[source.index("self.focus is not None") - 120 : source.index("self.focus is not None") + 140]
    # ...and it is still the last door, after every other one (the order itself did not move)
    for door in ("self.morning_brief.command_reply", "self.reminder_operations.command_reply", "self.scheduled_job_operations.command_reply", "self.memory_writer.recall_reply"):
        assert source.index(door) < source.index("self.focus.command_reply")
