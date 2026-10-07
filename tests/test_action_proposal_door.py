"""
The Action Proposal door in the real JarvisCore: where it sits, what may reach it, what it must never take from the other owners of a
message (a waiting question, a real confirmation, a Task flow, a tool, a search, the other doors), and that no model, tool or Job is
involved. Real InteractionResolver / JobEngine / stores (the resolver tests' own World), the production capability router, the real
FocusService and ActionProposalService on real task and World Model stores.
"""

import inspect
from types import SimpleNamespace

import pytest

from app.action_proposal import ActionProposalService, ExecutionHandoff
from app.capability import DeterministicCapabilityRouter
from app.core import JarvisCore
from app.focus import FocusService
from app.handle_this.routing import ObjectiveAwareCapabilityRouter
from app.memory.sqlite_store import SQLiteMemoryStore
from app.orchestration.base import JobStatus, Plan, PlanStep
from app.orchestration.interactions import InteractionStatus
from app.orchestration.tools import ConfirmJobTool, ListJobsTool
from app.reminders.service import ReminderService
from app.reminders.sqlite_store import SQLiteReminderStore
from app.reminders.tools import CreateReminderTool
from app.scheduling.timezones import ZoneChoice
from app.tasks.base import TaskPriority
from app.tasks.sqlite_store import SQLiteTaskStore
from app.tools.base import PermissionLevel
from app.world_model.factory import create_default_world_model_service, create_default_world_model_store
from test_interaction_resolver import ChatProvider, World, make_tool

PLAN = "Make a plan for my Arduino project"
SOON = "2026-09-29T15:00:00+00:00"  # the resolver tests' World clock is 12:00 UTC: a reminder at 14:30 is comfortably ahead


class CoreRig:
    def __init__(self, tmp_path, *, proposals=True):
        self.world = World()
        self.tasks = SQLiteTaskStore(str(tmp_path / "tasks.db"))
        self.wm = create_default_world_model_service(create_default_world_model_store(str(tmp_path / "wm.db")))
        self.wm.create_project("Arduino project")
        self.first = self.tasks.create(text="Test Arduino servo wiring", priority=TaskPriority.HIGH, due_date=SOON)
        self.second = self.tasks.create(text="Order Arduino sensor kit", priority=TaskPriority.MEDIUM)
        registry = self.world.registry
        registry.register(ListJobsTool(store=self.world.jobs))
        registry.register(ConfirmJobTool(store=self.world.jobs))
        registry.register(make_tool("create_job", PermissionLevel.SAFE_ACTION, self.world.log))
        self.reminders = SQLiteReminderStore(":memory:")
        registry.register(
            CreateReminderTool(
                self.reminders, ReminderService(self.reminders, clock=self.world.clock, zone_resolver=lambda: ZoneChoice("UTC", "configured"))
            )
        )
        self.provider = ChatProvider()
        self.memory = SQLiteMemoryStore(":memory:")
        self.handed = []
        self.focus = FocusService(tasks=self.tasks, calendar=None, world_model=self.wm, clock=self.world.clock)
        self.service = ActionProposalService(
            focus=self.focus, tasks=self.tasks, tools=registry, clock=self.world.clock, on_approved=self._handoff,
        )
        self.core = JarvisCore(
            provider=self.provider,
            memory=self.memory,
            tools=registry,
            capability_router=ObjectiveAwareCapabilityRouter(DeterministicCapabilityRouter()),
            interaction_resolver=self.world.resolver,
            focus=self.focus,
            action_proposals=self.service if proposals else None,
        )
        self.conversation = self.memory.create_conversation()

    def _handoff(self, approved):
        self.handed.append(approved)
        return "HANDED OFF"

    def say(self, message):
        try:
            return self.core.handle_chat(message, self.conversation)[0]
        except Exception as exc:  # noqa: BLE001 - e.g. an explicit search with no search tool: whatever the existing path does
            return exc

    def observable(self, message):
        reply = self.say(message)
        shown = (type(reply).__name__, str(reply)) if isinstance(reply, Exception) else reply
        return shown, self.provider.calls, [name for name, _ in self.world.registry.executed]

    def pending(self):
        return self.service._pending.get(self.conversation)

    def planned(self):
        reply = self.say(PLAN)
        assert self.pending() is not None, reply
        return reply

    def executed(self):
        return [name for name, _ in self.world.registry.executed]


def pair(tmp_path, setup, message):
    """The same world with the proposal door and without it: a message the door does not take is handled exactly as before it existed."""
    results = []
    for name in ("with", "without"):
        directory = tmp_path / name
        directory.mkdir()
        rig = CoreRig(directory, proposals=name == "with")
        setup(rig)
        results.append((rig, rig.observable(message)))
    return results


# ======================================================= the happy path, with nothing executed ==========================================


def test_a_request_through_the_real_core_shows_the_plan_without_a_model_a_tool_or_a_job(tmp_path):
    rig = CoreRig(tmp_path)
    before = (rig.tasks.list(), rig.world.jobs.list())

    reply = rig.planned()

    assert reply.startswith('Here\'s what I\'d do for "Arduino project":\n1. Your part: work on "Test Arduino servo wiring".')
    assert "3. My part: I'll set a reminder for 14:30 today:" in reply
    assert rig.provider.calls == 0 and rig.world.jobs.list() == [] and rig.executed() == []
    assert (rig.tasks.list(), rig.world.jobs.list()) == before


def test_the_core_counts_every_message_for_the_adjacency_lock_even_one_that_fails(tmp_path):
    rig = CoreRig(tmp_path)
    rig.planned()
    assert rig.pending().messages_since == 0  # nothing has arrived since the plan was shown

    rig.say("Why this plan?")  # answered by the door: the plan waits for the next message again
    assert rig.pending().messages_since == 0

    stored = len(rig.memory.get_messages(rig.conversation))
    failed = rig.say("Search for Arduino servo wiring")  # no search tool here: this message FAILS and stores nothing in the history
    assert isinstance(failed, Exception) and len(rig.memory.get_messages(rig.conversation)) == stored
    assert rig.pending().messages_since == 1  # ...yet it was counted

    assert rig.say("yes") != "HANDED OFF" and rig.handed == []  # so the plan can no longer be approved


def test_a_failed_message_in_between_still_ends_the_plans_chance(tmp_path):
    rig = CoreRig(tmp_path)
    rig.planned()
    rig.core.provider = SimpleNamespace(generate_reply=lambda message: (_ for _ in ()).throw(RuntimeError("model down")))

    assert isinstance(rig.say("tell me a joke"), Exception)  # the unrelated message blows up before anything is stored
    assert rig.say("yes") != "HANDED OFF" and rig.handed == []


@pytest.mark.parametrize("message", ["yes", "go ahead", "do it", "let's do it", "Yeah, let's do it.", "yes please", "yeah"])
def test_each_approval_phrase_approves_through_whichever_route_it_takes(tmp_path, message):
    rig = CoreRig(tmp_path)
    rig.planned()

    reply = rig.say(message)

    assert reply == "HANDED OFF" and len(rig.handed) == 1 and rig.pending() is None
    assert rig.provider.calls == 0  # no model for an approval
    assert rig.world.jobs.list() == [] and "create_reminder" not in rig.executed()  # approving created no Job and ran nothing


@pytest.mark.parametrize("message", ["ok", "sure"])
def test_ok_and_sure_through_the_core_do_not_approve(tmp_path, message):
    rig = CoreRig(tmp_path)
    rig.planned()

    assert "I need a clear yes" in rig.say(message)
    assert rig.handed == [] and rig.pending() is not None and rig.provider.calls == 0
    assert rig.say("yes") == "HANDED OFF"


@pytest.mark.parametrize("message", ["yes but skip step 2", "go ahead and email Sam"])
def test_a_yes_with_a_but_through_the_core_drops_the_plan_and_is_ordinary_conversation(tmp_path, message):
    rig = CoreRig(tmp_path)
    rig.planned()

    reply = rig.say(message)

    assert reply == "an ordinary chat reply" and rig.provider.calls == 1  # the model, as for any ordinary message
    assert rig.handed == [] and rig.pending() is None
    assert rig.say("yes") != "HANDED OFF" and rig.handed == []


@pytest.mark.parametrize("message", ["no", "nope", "cancel", "not now", "never mind"])
def test_a_rejection_through_the_core_cancels_it(tmp_path, message):
    rig = CoreRig(tmp_path)
    rig.planned()

    assert "I've dropped that plan" in rig.say(message)
    assert rig.pending() is None and rig.handed == [] and rig.provider.calls == 0


@pytest.mark.parametrize(
    "message", ["What does step 2 mean?", "Can you explain step 3?", "What happens after step 2?", "Why this plan?", "Show the plan again", "Can we skip step 2"]
)
def test_proposal_related_questions_are_answered_without_the_model_and_keep_it_alive(tmp_path, message):
    rig = CoreRig(tmp_path)
    rig.planned()

    reply = rig.say(message)

    assert "Say yes and I'll do my part" in reply and rig.provider.calls == 0 and rig.pending() is not None
    assert rig.say("yes") == "HANDED OFF" and len(rig.handed) == 1


# ======================================================= unrelated messages and adjacency ==============================================


@pytest.mark.parametrize("message", ["What's 2 plus 2?", "tell me a joke", "What's the weather tomorrow?"])
def test_an_unrelated_message_drops_the_plan_and_a_later_yes_approves_nothing(tmp_path, message):
    rig = CoreRig(tmp_path)
    rig.planned()

    rig.say(message)

    assert rig.pending() is None  # the door saw it, classified it unrelated, and dropped the plan
    assert rig.say("yes") != "HANDED OFF" and rig.handed == []


def test_a_message_another_door_claims_never_reaches_the_proposal_but_still_stops_a_later_yes(tmp_path):
    rig = CoreRig(tmp_path)
    rig.core.reminder_operations = SimpleNamespace(
        command_reply=lambda message, conversation_id=None: "REMINDER SET" if message.lower().startswith("remind me") else None
    )
    rig.planned()

    assert rig.say("Remind me to call Mom at 6") == "REMINDER SET"  # the reminder door owns it, as always
    assert rig.say("yes") != "HANDED OFF" and rig.handed == []  # ...and the plan, which never saw that message, can no longer be approved


@pytest.mark.parametrize("door", ["morning_brief", "reminder_operations", "device_operations", "activity_context"])
def test_another_door_that_claims_the_request_itself_comes_first(tmp_path, door):
    rig = CoreRig(tmp_path)
    setattr(rig.core, door, SimpleNamespace(command_reply=lambda message, conversation_id=None, **kwargs: f"CLAIMED BY {door}"))

    assert rig.say(PLAN) == f"CLAIMED BY {door}" and rig.pending() is None


@pytest.mark.parametrize(
    "message",
    ["use tool list_jobs", "Create a task: finish my Arduino project today", "Search for Arduino servo wiring", "what is the status of my task?"],
)
def test_a_tool_task_status_or_search_message_keeps_its_own_flow_and_ends_the_plans_chance(tmp_path, message):
    (rig, with_proposals), (_, without) = pair(tmp_path, lambda r: r.say(PLAN), message)

    assert with_proposals[0] == without[0] and with_proposals[2] == without[2]  # the same reply, the same tools: nothing was taken
    assert rig.say("yes") != "HANDED OFF" and rig.handed == []  # the plan never saw that message, so it can no longer be approved


def test_without_the_proposal_service_everything_is_exactly_as_before(tmp_path):
    rig = CoreRig(tmp_path, proposals=False)

    assert rig.say(PLAN) == "an ordinary chat reply" and rig.say("yes") != "HANDED OFF" and rig.handed == []


# ======================================================= interaction precedence ==========================================================


def test_a_waiting_question_owns_the_next_message_and_no_plan_is_made(tmp_path):
    rig = CoreRig(tmp_path)
    job, interaction = rig.world.input_job(conversation_id=rig.conversation, prompt="Which Arduino project do you mean?")
    rig.world.present(interaction, rig.conversation)

    rig.say(PLAN)  # the user answers the question with a plan-looking sentence

    assert rig.world.interactions.get(interaction.id).response == PLAN  # the interaction took it, exactly as typed
    assert rig.pending() is None and rig.handed == []


def test_a_waiting_confirmation_stands_the_door_down_for_a_new_request(tmp_path):
    (rig, with_proposals), (_, without) = pair(tmp_path, lambda r: r.world.present(r.world.confirm_job(conversation_id=r.conversation)[1], r.conversation), PLAN)

    assert rig.pending() is None and with_proposals == without


def test_a_confirmation_that_arrives_after_the_plan_gets_the_yes_and_the_plan_is_not_approved(tmp_path):
    rig = CoreRig(tmp_path)
    rig.planned()
    job, interaction = rig.world.confirm_job(conversation_id=rig.conversation)
    rig.world.present(interaction, rig.conversation)  # a real confirmation is now waiting on this conversation

    rig.say("yes")

    assert rig.world.interactions.get(interaction.id).status is InteractionStatus.RESOLVED  # the existing flow took the yes
    assert rig.handed == []
    assert rig.say("yes") != "HANDED OFF" and rig.handed == []  # and the plan, which never saw it, is dead


def test_the_confirmation_hook_stands_down_while_a_real_confirmation_waits_even_for_a_word_the_resolver_ignores(tmp_path):
    rig = CoreRig(tmp_path)
    rig.planned()
    _, interaction = rig.world.confirm_job(conversation_id=rig.conversation)
    rig.world.present(interaction, rig.conversation)

    reply = rig.say("ok")  # CONFIRMATION-routed, but not a yes the resolver accepts: it is nobody's reply, and certainly not the plan's

    assert "I need a clear yes" not in str(reply) and rig.handed == []
    assert rig.world.interactions.get(interaction.id).status is InteractionStatus.PENDING  # the confirmation is untouched, still waiting


def test_a_question_that_arrives_after_the_plan_gets_the_reply_not_the_plan(tmp_path):
    rig = CoreRig(tmp_path)
    rig.planned()
    job, interaction = rig.world.input_job(conversation_id=rig.conversation, prompt="Which browser?")
    rig.world.present(interaction, rig.conversation)

    rig.say("let's do it")  # DIRECT: the resolver reads it as the answer, never the door

    assert rig.world.interactions.get(interaction.id).response == "let's do it" and rig.handed == []


def test_a_job_waiting_for_a_legacy_confirmation_is_confirmed_first_and_the_plan_is_not_approved(tmp_path):
    rig = CoreRig(tmp_path)
    job = rig.world.jobs.create("Tidy my calendar", conversation_id=rig.conversation)
    job.plan = Plan(steps=[PlanStep(id="s1", tool_name="gated", permission="confirm")])
    job.status = JobStatus.AWAITING_CONFIRMATION
    job.pending_step_ids = ["s1"]
    job.confirmation_message = "confirm?"
    rig.world.jobs.save(job)  # parked by status alone, with no interaction (the older path)
    rig.planned()

    rig.say("yes")

    assert "confirm_job" in rig.executed() and rig.handed == []  # the older confirmation path answered, before the proposal could


def test_an_interaction_never_confuses_two_owners_of_one_yes(tmp_path):
    rig = CoreRig(tmp_path)
    rig.planned()
    _, interaction = rig.world.confirm_job(conversation_id=rig.conversation)
    rig.world.present(interaction, rig.conversation)

    rig.say("yes")
    rig.say("yes")  # a second yes does not reach the (now dead) plan either

    assert rig.handed == []


# ======================================================= where the door sits =============================================================


def test_the_door_follows_focus_and_every_other_door_and_precedes_the_context_engine():
    source = inspect.getsource(JarvisCore._dispatch)
    door = source.index("self.action_proposals.command_reply")

    for earlier in (
        "self.interaction_resolver.handle", "self.activity_context.command_reply", "self.morning_brief.command_reply",
        "self.device_operations.command_reply", "self.reminder_operations.command_reply", "self.scheduled_job_operations.command_reply",
        "self.memory_writer.command_reply", "self.memory_writer.recall_reply", "self.focus.command_reply",
    ):
        assert source.index(earlier) < door, earlier
    assert door < source.index("self.context_engine.build_snapshot")


def test_every_message_is_counted_before_anything_decides_who_answers_it():
    source = inspect.getsource(JarvisCore._dispatch)

    assert source.index("self.action_proposals.note_message(conversation_id)") < source.index("self.capability_router.route(message)")
    assert source.index("self.action_proposals.note_message(conversation_id)") < source.index("self.interaction_resolver.handle")


def test_the_door_has_the_same_gate_as_focus_and_looks_up_interactions_last():
    source = inspect.getsource(JarvisCore._dispatch)
    door = source.index("self.action_proposals.command_reply")
    block = source[source.rindex("if (\n", 0, door) : door]

    assert "memory_reply is None" in block and "interaction_reply is None" in block
    assert block.index("decision.capability is Capability.DIRECT") < block.index("is_explicit_search_request") < block.index("_interaction_awaiting_reply")
    for forbidden in ("Capability.TASK", "Capability.TOOL", "Capability.CONFIRMATION", "SELF_DEV"):
        assert forbidden not in block


def test_the_confirmation_hook_is_reached_only_after_every_real_confirmation_declined_the_message():
    source = inspect.getsource(JarvisCore._dispatch)
    branch = source[source.index("elif decision.capability is Capability.CONFIRMATION:") : source.index("elif decision.capability is Capability.SELF_DEVELOPMENT:")]

    assert branch.index("self._handle_confirmation(") < branch.index("self._proposal_confirmation_reply(") < branch.index("self._fallback_to_direct(")
    assert "if confirmation_result is not None" in branch
    assert source.index("if interaction_reply is not None:") < source.index("elif decision.capability is Capability.CONFIRMATION:")


def test_the_confirmation_hook_stands_down_while_an_interaction_waits_and_on_any_fault():
    source = inspect.getsource(JarvisCore._proposal_confirmation_reply)

    assert "_interaction_awaiting_reply" in source and "except Exception" in source and "return None" in source


def test_a_fault_in_the_proposal_service_is_not_a_proposal_message(tmp_path):
    class Exploding:
        def command_reply(self, message, conversation_id=None):
            raise RuntimeError("boom")

        def confirmation_reply(self, message, conversation_id=None):
            raise RuntimeError("boom")

    (rig, with_it), (_, without) = pair(tmp_path, lambda r: setattr(r.core, "action_proposals", Exploding()), "yes")

    assert with_it == without


# ======================================================= the real application ============================================================


def test_the_real_application_wires_proposals_to_the_execution_handoff_and_adds_no_tool():
    import app.main as live

    service = live.core.action_proposals
    assert isinstance(service, ActionProposalService)
    assert isinstance(service.on_approved, ExecutionHandoff)  # the one boundary an approved proposal leaves through (Slice 3)
    assert service.focus is live.core.focus and service.tools is live.core.tools and service.tasks is live._task_store
    assert not [name for name in live.core.tools._tools if "proposal" in name]
    assert service.command_reply("what is the capital of France?", "c1") is None and service.pending_count() == 0
