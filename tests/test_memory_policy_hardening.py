"""
Memory write policy - the hardening slice.

    1. A confident, staged CANDIDATE may be shown to reasoning - always labelled
       UNCONFIRMED, in its own field, never as a fact.
    2. No autonomous plan and no automation can invoke a memory-write tool: the
       ToolRegistry (`plannable`, `automatable`, `permission`) says so.
    3. Sensitive memory is kept only after the user CONFIRMS it, through the existing
       Job / interaction / ToolRegistry confirmation - "remember anyway" is no bypass.

Real SQLite, real ContextEngine / JobEngine / interaction resolver / ToolRegistry /
AutomationEngine; the only stand-in is the chat model.
"""

import re
import sqlite3
import time

import pytest

from app.automation.base import ExecutionOutcome, Trigger, TriggerType
from app.automation.engine import AutomationEngine
from app.automation.sqlite_store import SQLiteAutomationStore
from app.automation.tools import CreateAutomationTool
from app.context_engine.base import (
    CANDIDATE_CONTEXT_MIN_CONFIDENCE,
    MAX_CONTEXT_CANDIDATES,
    RequestContext,
    RequiredContextCategory,
)
from app.context_engine.engine import ContextEngine
from app.context_reasoning.reasoner import ContextReasonerError
from app.context_reasoning.deterministic import DeterministicAnalysis, analyze
from app.context_reasoning.prompt import build_reasoning_prompt, collect_allowed_provenance
from app.context_reasoning.reasoner import _parse_and_validate
from app.core import JarvisCore
from app.memory.sqlite_store import SQLiteMemoryStore
from app.memory_policy.base import Kind, Origin, WriteProposal
from app.memory_policy.confirmation import COMMIT_TOOL_NAME, GOAL, PROMPT, SensitiveWriteConfirmer
from app.memory_policy.service import MemoryWriteService
from app.memory_policy.store import SQLiteWriteStore
from app.memory_policy.tools import register_memory_tools
from app.orchestration.base import JobStatus
from app.orchestration.engine import JobEngine
from app.orchestration.interaction_resolver import InteractionResolver
from app.orchestration.interactions import InteractionStatus, InteractionType
from app.orchestration.sqlite_interaction_store import SQLiteInteractionStore
from app.orchestration.sqlite_store import SQLiteJobStore
from app.orchestration.validator import PlanValidationError, validate_plan
from app.reminders.sqlite_store import SQLiteReminderStore
from app.tools.base import PermissionLevel, ToolResultStatus
from app.tools.decision import ToolCall
from app.tools.registry import ToolRegistry
from app.world_model.base import CANDIDATE_LABEL, ContextSnapshot
from app.world_model.factory import create_default_world_model_service, create_default_world_model_store
from test_handle_this_integration import ChatProvider, ScriptedPlanner
from test_memory_policy import World

MEMORY_TOOLS = ("remember", "set_current_project", "set_goal", COMMIT_TOOL_NAME)
SENSITIVE_ASK = "That looks sensitive, so I won't keep it unless you confirm. Shall I save it? (yes or no)"
SENSITIVE_WORDS = ("diabetes", "salary", "medication", "tower")


# --- 1. candidates in context ---------------------------------------------------------------------------------------------------------


def _memory_request():
    return RequestContext(
        raw_request="what do you know about my tea",
        required_context=frozenset({RequiredContextCategory.MEMORY, RequiredContextCategory.PREFERENCES}),
    )


def _engine(world, *, source=True):
    return ContextEngine(
        world.wm, world.memory, candidate_source=world.service.candidates_for_context if source else None
    )


def _stage(world, times=1, key="prefers:green_tea", value="green tea", basis="inferred"):
    for _ in range(times):
        world.registry.execute("remember", {"key": key, "value": value, "kind": "preference", "basis": basis})


@pytest.fixture
def world(tmp_path):
    return World(tmp_path)


def test_a_confident_candidate_can_be_retrieved_labelled_unconfirmed(world):
    _stage(world, times=2)  # 0.5, then corroborated to 0.6

    snapshot = _engine(world).build_snapshot(_memory_request())

    (c,) = snapshot.unconfirmed_candidates
    assert (c.key, c.value, c.kind) == ("prefers:green_tea", "green tea", "preference")
    assert c.status == "unconfirmed" and c.confidence == pytest.approx(0.6)
    assert c.confidence >= CANDIDATE_CONTEXT_MIN_CONFIDENCE
    assert c.origin == "inference" and c.created_at and c.updated_at  # provenance kept
    assert CANDIDATE_LABEL == "UNCONFIRMED_CANDIDATE"


def test_a_single_uncertain_observation_is_not_yet_confident_enough_to_be_shown(world):
    _stage(world, times=1)
    world.service.observe("I think I prefer black coffee.")

    assert world.service.store.list_candidates()  # staged...
    assert _engine(world).build_snapshot(_memory_request()).unconfirmed_candidates == []  # ...but not shown


def test_a_candidate_is_never_placed_in_a_fact_category_and_cannot_satisfy_a_fact_lookup(world):
    _stage(world, times=3)

    snapshot = _engine(world).build_snapshot(_memory_request())

    assert snapshot.unconfirmed_candidates
    assert snapshot.relevant_preferences == [] and snapshot.relevant_memory == []
    assert "green tea" not in str(snapshot.relevant_preferences) + str(snapshot.relevant_memory)
    # the fact lookups themselves - the stores everything reads facts from - know nothing of it
    assert world.facts() == {} and world.wm.get_preferences(key="prefers:green_tea") == []
    assert world.wm.store.list_preferences() == []


def test_retrieving_a_candidate_never_promotes_it_or_writes_anything(world):
    _stage(world, times=3)
    before = (len(world.journal()), world.service.store.list_candidates())

    engine = _engine(world)
    for _ in range(5):
        engine.build_snapshot(_memory_request())

    assert (len(world.journal()), world.service.store.list_candidates()) == before
    assert world.nothing_persisted() and world.service.store.list_candidates(status="promoted") == []
    assert world.service.store.list_candidates()[0].status == "candidate"


def test_a_retrieved_candidate_is_not_treated_as_existing_when_the_user_states_it_later(world):
    _stage(world, times=3)
    _engine(world).build_snapshot(_memory_request())

    reply = world.service.observe("I prefer green tea.")

    assert reply == "Noted - I'll remember that."  # not "I already have that": it was never a fact
    assert world.wm.get_preferences(key="prefers:green_tea")[0].source.value == "user"


def test_candidates_need_the_memory_category_and_a_source_at_all(world):
    _stage(world, times=3)
    without_memory = RequestContext(raw_request="x", required_context=frozenset({RequiredContextCategory.PREFERENCES}))

    assert _engine(world).build_snapshot(without_memory).unconfirmed_candidates == []
    assert _engine(world, source=False).build_snapshot(_memory_request()).unconfirmed_candidates == []


def test_only_a_bounded_number_of_the_strongest_candidates_are_shown(world):
    for i in range(MAX_CONTEXT_CANDIDATES + 3):
        _stage(world, times=2 + (i % 3), key=f"prefers:thing_{i}", value=f"thing {i}")

    shown = _engine(world).build_snapshot(_memory_request()).unconfirmed_candidates

    assert len(shown) == MAX_CONTEXT_CANDIDATES
    assert [c.confidence for c in shown] == sorted((c.confidence for c in shown), reverse=True)


def test_a_faulty_candidate_source_only_means_no_candidates(world):
    def broken():
        raise RuntimeError("bug")

    engine = ContextEngine(world.wm, world.memory, candidate_source=broken)
    assert engine.build_snapshot(_memory_request()).unconfirmed_candidates == []


def test_a_promoted_candidate_is_no_longer_shown_as_a_candidate(world):
    _stage(world, times=3)
    world.service.observe("I prefer green tea.")

    assert _engine(world).build_snapshot(_memory_request()).unconfirmed_candidates == []


def _snapshot_with_candidate(world):
    _stage(world, times=2)
    snapshot = _engine(world).build_snapshot(_memory_request())
    return snapshot, snapshot.unconfirmed_candidates[0]


def test_reasoning_is_shown_the_candidate_only_as_an_unconfirmed_hypothesis(world):
    snapshot, c = _snapshot_with_candidate(world)

    prompt = build_reasoning_prompt("what tea do I like", _memory_request(), snapshot, DeterministicAnalysis())

    line = next(l for l in prompt.splitlines() if f"[candidate:{c.id}]" in l)
    assert "UNCONFIRMED_CANDIDATE" in line and "NOT confirmed by the user" in line and "0.60" in line
    assert "from inference" in line and "prefers:green_tea=green tea" in line
    assert "never put it in relevant_facts, constraints or conflicts" in prompt
    assert f"candidate:{c.id}" in collect_allowed_provenance(snapshot, DeterministicAnalysis())
    assert "[preference:" not in line and "[memory]" not in line


def _payload(field, provenance):
    body = {"interpretation": "i", "relevant_facts": [], "constraints": [], "conflicts": [], "opportunities": [],
            "conclusions": [], "recommended_next_step": None, "confidence": 0.5,
            "requires_clarification": False, "clarification_question": None}
    body[field] = [{"text": "They may like green tea.", "provenance": provenance}]
    import json

    return json.dumps(body)


@pytest.mark.parametrize("field", ["relevant_facts", "constraints", "conflicts"])
def test_reasoning_cannot_present_a_candidate_as_established(world, field):
    snapshot, c = _snapshot_with_candidate(world)
    allowed = collect_allowed_provenance(snapshot, DeterministicAnalysis())

    with pytest.raises(ContextReasonerError, match="unconfirmed candidate"):
        _parse_and_validate(_payload(field, [f"candidate:{c.id}"]), allowed)
    with pytest.raises(ContextReasonerError, match="unconfirmed candidate"):  # even alongside real provenance
        _parse_and_validate(_payload(field, ["request", f"candidate:{c.id}"]), allowed)


@pytest.mark.parametrize("field", ["conclusions", "opportunities"])
def test_reasoning_may_use_a_candidate_as_a_clue_or_hypothesis(world, field):
    snapshot, c = _snapshot_with_candidate(world)
    allowed = collect_allowed_provenance(snapshot, DeterministicAnalysis())

    result = _parse_and_validate(_payload(field, [f"candidate:{c.id}"]), allowed)

    assert getattr(result, field)[0].provenance == (f"candidate:{c.id}",)


def test_a_candidate_id_that_was_not_supplied_is_still_a_hallucination(world):
    snapshot, _ = _snapshot_with_candidate(world)
    allowed = collect_allowed_provenance(snapshot, DeterministicAnalysis())
    with pytest.raises(ContextReasonerError):
        _parse_and_validate(_payload("conclusions", ["candidate:made-up"]), allowed)


def test_the_direct_prompt_shows_candidates_as_hunches_not_facts_and_a_job_never_gets_them(world):
    snapshot, c = _snapshot_with_candidate(world)

    text = JarvisCore._format_world_model_context(snapshot)

    assert "Unconfirmed hunches (NOT facts" in text and "prefers:green_tea=green tea (confidence 0.60)" in text
    assert "Relevant preferences" not in text and "Relevant memory" not in text
    # what an autonomous Job's planner is handed carries no candidate at all
    assert JarvisCore._snapshot_to_job_context(snapshot) is None
    with_fact = ContextSnapshot(current_time="t", relevant_memory=["dog: Max"], unconfirmed_candidates=[c])
    assert JarvisCore._snapshot_to_job_context(with_fact) == {"Relevant memory": ["dog: Max"]}


def test_a_snapshot_without_candidates_renders_exactly_as_before():
    assert JarvisCore._format_world_model_context(ContextSnapshot(current_time="t")) == ""
    text = JarvisCore._format_world_model_context(ContextSnapshot(current_time="t", relevant_memory=["dog: Max"]))
    assert "hunches" not in text and text.endswith("- Relevant memory: dog: Max")


# --- 2. autonomous plans and automation cannot write memory ----------------------------------------------------------------------------


@pytest.mark.parametrize("name", MEMORY_TOOLS)
def test_every_memory_write_tool_is_neither_plannable_nor_automatable(world, name):
    tool = world.registry.get(name)
    assert tool.plannable is False and tool.automatable is False


def test_the_planner_is_never_offered_and_validate_plan_rejects_every_memory_write_tool(world):
    offered = {t.name for t in world.registry.list_tools() if t.plannable}
    assert not offered & set(MEMORY_TOOLS)

    for name in MEMORY_TOOLS:
        with pytest.raises(PlanValidationError):
            validate_plan(
                [{"id": "s1", "tool": name, "arguments": {"key": "k", "value": "v", "pending_id": "p"}, "depends_on": []}],
                world.registry,
                max_steps=5,
                default_retry=None,
            )


def _automation_world(world):
    reminders = SQLiteReminderStore(":memory:")
    store = SQLiteAutomationStore(":memory:")
    engine = AutomationEngine(store=store, tools=world.registry, reminder_store=reminders, memory=world.memory)
    create = CreateAutomationTool(store=store, tools=world.registry, reminder_store=reminders)
    return store, engine, create


@pytest.mark.parametrize("name", MEMORY_TOOLS)
def test_an_automation_cannot_be_created_with_a_memory_write_action(world, name):
    store, _engine_, create = _automation_world(world)

    result = create.execute(
        name="sneaky", trigger_type="manual", action_tool_name=name, action_params={"key": "k", "value": "v"}
    )

    assert result.status is ToolResultStatus.ERROR and "can't be run by an automation" in result.message
    assert store.list() == [] and world.nothing_persisted()


@pytest.mark.parametrize("name", ["remember", "set_current_project", "set_goal"])
def test_an_automation_that_got_into_the_store_anyway_still_cannot_run_one(world, name):
    from datetime import datetime, timedelta, timezone

    store, engine, _create = _automation_world(world)
    params = {"key": "k", "value": "v"} if name == "remember" else {"name": "n"} if name == "set_current_project" else {"title": "t"}
    automation = store.create(
        name="sneaky",
        trigger=Trigger(type=TriggerType.TIME),
        conditions=[],
        action=ToolCall(tool_name=name, parameters=params),
        next_run_at=(datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat(),
    )

    engine.tick()

    assert store.get_executions(automation.id)[-1].outcome is ExecutionOutcome.FAILURE
    assert world.nothing_persisted() and world.journal() == []  # not even a decision was made: it never ran


def test_ordinary_automation_actions_are_unaffected(world):
    from app.tools.base import Tool, ToolResult

    class Harmless(Tool):
        name = "harmless"
        description = "d"
        permission = PermissionLevel.SAFE_ACTION

        def execute(self) -> ToolResult:
            return ToolResult.success(message="ok")

    world.registry.register(Harmless())
    _store, _engine_, create = _automation_world(world)

    assert create.execute(name="fine", trigger_type="manual", action_tool_name="harmless").status is ToolResultStatus.SUCCESS


def test_the_registry_stays_authoritative_a_confirm_tool_needs_confirmation_and_the_write_tools_still_work_for_the_user(world):
    unconfirmed = world.registry.execute(COMMIT_TOOL_NAME, {"pending_id": "x"})
    denied = world.registry.execute(COMMIT_TOOL_NAME, {"pending_id": "x"}, confirmed=False)
    assert unconfirmed.status is ToolResultStatus.CONFIRMATION_REQUIRED and denied.status is ToolResultStatus.DENIED
    assert world.registry.get(COMMIT_TOOL_NAME).permission is PermissionLevel.CONFIRM

    # ordinary user-authorised writes are exactly as before
    assert world.registry.execute("remember", {"key": "prefers:dark_mode", "value": "dark mode", "kind": "preference"}).data["persisted"]
    assert world.service.command_reply("Remember that my dog is called Max.") == "Noted - I'll remember that."
    assert world.service.observe("My current project is LunitiAI.") == "Noted - I'll remember that."


def test_the_tools_that_gate_autonomy_are_not_widened_by_the_new_flag():
    from app.tools.base import Tool

    assert Tool.automatable is True  # every existing tool keeps its behaviour: only these four opt out


# --- 3. sensitive memory goes through the existing confirmation ---------------------------------------------------------------------------


class Chain:
    """The real chain: chat -> write policy -> a waiting confirmation Job -> the existing
    resolver / JobEngine / ToolRegistry -> the write."""

    def __init__(self, tmp_path):
        self.dir = tmp_path
        self.db = str(tmp_path / "world.db")
        self.wm = create_default_world_model_service(create_default_world_model_store(self.db))
        self.memory = SQLiteMemoryStore(str(tmp_path / "memory.db"))
        self.jobs = SQLiteJobStore(str(tmp_path / "jobs.db"))
        self.interactions = SQLiteInteractionStore(self.jobs)
        self.registry = ToolRegistry()
        self.confirmer = SensitiveWriteConfirmer(self.interactions)
        self.service = MemoryWriteService(self.wm, self.memory, SQLiteWriteStore(self.db), confirmer=self.confirmer)
        register_memory_tools(self.registry, self.service)
        self.planner = ScriptedPlanner()
        self.engine = JobEngine(
            store=self.jobs, tools=self.registry, planner=self.planner, max_steps=10, max_total_steps=20,
            max_retries_per_step=1, max_replans=2, retry_delay_seconds=0, poll_interval_seconds=1000,
            interaction_store=self.interactions,
        )
        self.resolver = InteractionResolver(self.interactions, self.jobs, self.engine)
        self.confirmer.mark_surfaced = self.resolver.mark_surfaced
        self.provider = ChatProvider()
        self.core = JarvisCore(
            provider=self.provider, memory=self.memory, tools=self.registry,
            memory_writer=self.service, interaction_resolver=self.resolver,
        )
        self.conversation = self.memory.create_conversation()

    def say(self, message, conversation=None):
        reply, _, _ = self.core.handle_chat(message, conversation or self.conversation)
        return reply

    def tick(self, times=3):
        for _ in range(times):
            self.engine.tick()

    def facts(self):
        return self.memory.get_facts()

    def journal(self, key=None):
        return self.service.store.journal(key=key)

    def persisted_nothing(self):
        return self.facts() == {} and self.wm.store.list_preferences() == [] and self.wm.get_active_projects() == []

    def disk_without_chat_history(self) -> str:
        """Everything on disk EXCEPT the conversation log (which holds what the user typed)."""
        chunks = []
        for name in ("world.db", "memory.db", "jobs.db"):
            conn = sqlite3.connect(str(self.dir / name))
            for (table,) in conn.execute("select name from sqlite_master where type='table'"):
                if table in ("messages", "conversations"):
                    continue
                for row in conn.execute(f'select * from "{table}"'):
                    chunks.append(" ".join(str(v) for v in row))
            conn.close()
        return "\n".join(chunks)


@pytest.fixture
def chain(tmp_path):
    return Chain(tmp_path)


def _only_job(chain):
    (job,) = chain.jobs.list()
    (interaction,) = chain.interactions.list_for_job(job.id)
    return job, interaction


def test_a_sensitive_request_asks_for_confirmation_through_the_existing_interaction(chain):
    reply = chain.say("Remember that I have diabetes")

    assert reply == SENSITIVE_ASK
    job, interaction = _only_job(chain)
    assert job.status is JobStatus.AWAITING_CONFIRMATION and job.goal == GOAL and job.conversation_id == chain.conversation
    assert interaction.type is InteractionType.CONFIRMATION and interaction.status is InteractionStatus.PENDING
    assert interaction.prompt == PROMPT and interaction.step_ids == ["s1"] and interaction.detail["origin"] == "memory_policy"
    assert [s.tool_name for s in job.plan.steps] == [COMMIT_TOOL_NAME] and job.plan.steps[0].permission == "confirm"
    assert interaction.presented_conversation_id == chain.conversation  # put to the user, so a plain "yes" answers it
    assert chain.persisted_nothing() and chain.planner.calls == []  # no planner was ever involved


def test_nothing_is_written_before_the_user_confirms(chain):
    chain.say("Remember that I have diabetes")

    chain.tick(5)
    chain.say("hello")
    chain.say("thanks")

    assert chain.persisted_nothing()
    assert chain.service.pending_count() == 1
    assert [e.action for e in chain.journal()] == ["asked"] and chain.journal()[0].value is None
    job, _ = _only_job(chain)
    assert job.status is JobStatus.AWAITING_CONFIRMATION


def test_confirming_persists_it_with_the_right_provenance(chain):
    chain.say("Remember that I have diabetes")

    reply = chain.say("yes")
    chain.tick()

    assert reply.startswith("Approved")
    assert list(chain.facts().values()) == ["I have diabetes"]
    job, interaction = _only_job(chain)
    assert job.status is JobStatus.COMPLETED and interaction.status is InteractionStatus.RESOLVED
    assert job.final_result and "Saved to memory" in job.final_result
    persisted = [e for e in chain.journal() if e.action == "persisted"]
    (entry,) = persisted
    assert entry.origin == "user_command" and entry.source == "user" and entry.confidence == 1.0
    assert "confirmed by the user" in entry.reason and entry.value == "I have diabetes"
    assert [e.action for e in chain.journal()] == ["persisted", "asked"]  # the earlier "asked" stays on record, valueless
    assert chain.journal()[-1].value is None and chain.journal()[-1].key == "(withheld)"
    assert chain.service.pending_count() == 0


def test_declining_persists_nothing_and_it_can_never_be_committed_afterwards(chain):
    chain.say("Remember that I have diabetes")
    job, _ = _only_job(chain)
    held_id = next(iter(chain.service._pending))

    reply = chain.say("no")
    chain.tick(5)

    assert reply.startswith("Declined")
    assert chain.persisted_nothing() and not any(e.action == "persisted" for e in chain.journal())
    done = chain.jobs.get(job.id)
    assert done.status is JobStatus.FAILED  # the existing outcome of a declined step
    assert [s.status.value for s in done.plan.steps] == ["failed"]  # the step never ran
    # ...and not even the right ids, and a confirmed call, can commit it now: the confirmation was not APPROVED
    outcome = chain.service.commit_pending(held_id, job.id)
    assert not outcome.persisted and "only be saved by the confirmation that was asked" in outcome.message
    forged = chain.registry.execute(
        COMMIT_TOOL_NAME, {"pending_id": held_id, "source_job_id": job.id}, confirmed=True
    )
    assert forged.status is ToolResultStatus.ERROR
    assert chain.persisted_nothing()


def test_the_commit_needs_an_approved_confirmation_not_just_the_right_ids(chain):
    chain.say("Remember that I have diabetes")
    job, _ = _only_job(chain)
    held_id = next(iter(chain.service._pending))

    forged = chain.registry.execute(  # right pending id, right job id, but the user has not answered yet
        COMMIT_TOOL_NAME, {"pending_id": held_id, "source_job_id": job.id}, confirmed=True
    )

    assert forged.status is ToolResultStatus.ERROR and chain.persisted_nothing()
    assert held_id in chain.service._pending  # still waiting for the real answer


def test_a_confirmation_that_outlives_its_hold_saves_nothing_and_says_so(chain):
    chain.say("Remember that I have diabetes")
    for held in chain.service._pending.values():
        held.created -= 10 * 86_400  # long ago

    chain.say("yes")
    chain.tick()

    job, _ = _only_job(chain)
    assert chain.persisted_nothing()
    assert job.status is JobStatus.COMPLETED and "expired" in job.final_result and "diabetes" not in job.final_result


def test_the_commit_step_refuses_any_call_that_did_not_come_from_the_confirmed_job(chain):
    chain.say("Remember that I have diabetes")
    job, _ = _only_job(chain)
    held_id = next(iter(chain.service._pending))

    assert chain.registry.execute(COMMIT_TOOL_NAME, {"pending_id": held_id}, confirmed=True).status is ToolResultStatus.ERROR
    wrong = chain.registry.execute(
        COMMIT_TOOL_NAME, {"pending_id": held_id, "source_job_id": "not-that-job"}, confirmed=True
    )
    assert wrong.status is ToolResultStatus.ERROR and "only be saved by the confirmation that was asked" in wrong.message
    assert chain.persisted_nothing() and held_id in chain.service._pending


def test_saying_anyway_does_not_bypass_the_confirmation(chain):
    reply = chain.say("Remember anyway: I have diabetes")

    assert reply == SENSITIVE_ASK and chain.persisted_nothing()
    assert chain.core.memory_writer.pending_count() == 1


def test_a_model_proposed_sensitive_write_goes_through_the_same_confirmation(chain):
    result = chain.registry.execute("remember", {"key": "health", "value": "has diabetes"})

    assert result.status is ToolResultStatus.SUCCESS and result.data == {
        "persisted": False, "destination": "ask", "reason": "it looks sensitive, so it needs the user's say-so",
    }
    assert result.message == SENSITIVE_ASK
    job, interaction = _only_job(chain)
    assert job.status is JobStatus.AWAITING_CONFIRMATION and chain.persisted_nothing()

    assert chain.say("hello").count("Waiting on you") == 1  # a waiting confirmation is put to the user with the next reply
    chain.say("yes")
    chain.tick()
    assert chain.facts() == {"health": "has diabetes"}
    assert chain.jobs.get(job.id).status is JobStatus.COMPLETED


def test_a_secret_is_never_asked_about_never_held_and_never_written_even_if_someone_says_yes(chain):
    for message in ("Remember that my password is hunter2", "Remember anyway: my api key is sk-abcdefghijklmnopqrstuvwx"):
        assert chain.say(message) == "That looks like a credential or a secret, so I won't store it."

    assert chain.jobs.list() == [] and chain.service.pending_count() == 0
    chain.say("yes")
    chain.tick()
    assert chain.persisted_nothing()
    assert "hunter2" not in chain.disk_without_chat_history() and "sk-abcdef" not in chain.disk_without_chat_history()


def test_a_held_request_that_somehow_became_a_secret_is_still_refused_when_confirmed(chain):
    chain.say("Remember that I have diabetes")
    job, _ = _only_job(chain)
    chain.say("yes")  # approved...
    held = next(iter(chain.service._pending.values()))
    held.proposal = WriteProposal(Kind.FACT, "k", "the password is hunter2", Origin.USER_COMMAND)  # ...but now a secret

    chain.tick()

    done = chain.jobs.get(job.id)
    assert done.status is JobStatus.FAILED and chain.planner.calls == []  # refused; no model replanned it
    assert chain.persisted_nothing() and "hunter2" not in chain.disk_without_chat_history()
    assert not any(e.action == "persisted" for e in chain.journal())


def test_a_passing_sensitive_statement_is_not_persisted_and_does_not_start_a_confirmation(chain):
    reply = chain.say("I prefer keeping my salary private.")

    assert "That looks sensitive, so I haven't saved it." in reply and "Shall I" not in reply
    assert chain.jobs.list() == [] and chain.persisted_nothing()


def test_no_sensitive_content_is_in_the_job_the_interaction_the_notification_text_or_the_report(chain):
    chain.say("Remember that I take medication for diabetes")
    job, interaction = _only_job(chain)
    from app.orchestration.notification_events import waiting_event

    texts = [
        job.goal, str(job.confirmation_message), interaction.prompt, str(interaction.detail), str(job.context),
        str([(s.tool_name, s.arguments) for s in job.plan.steps]), waiting_event(interaction, job).text,
        str([e.detail for e in job.events]),
    ]
    chain.say("yes")
    chain.tick()
    done = chain.jobs.get(job.id)
    texts += [str(done.final_result), str([e.detail for e in done.events]), str(done.error)]

    joined = " ".join(texts).lower()
    assert not any(word in joined for word in SENSITIVE_WORDS + ("i take",))
    # the ONLY places the content exists on disk are the two the user asked to keep it in
    assert chain.facts() and set(chain.facts().values()) == {"I take medication for diabetes"}


def test_the_confirmation_follows_the_conversation_it_was_asked_in(chain):
    chain.say("Remember that I have diabetes")
    other = chain.memory.create_conversation()

    chain.say("yes", conversation=other)  # a "yes" from somewhere else does not answer it
    chain.tick()

    assert chain.persisted_nothing()
    assert _only_job(chain)[0].status is JobStatus.AWAITING_CONFIRMATION


def test_a_restart_between_the_question_and_the_yes_saves_nothing_and_fails_closed(tmp_path):
    first = Chain(tmp_path)
    first.say("Remember that I have diabetes")
    job, interaction = _only_job(first)

    # a restart: everything rebuilt over the same files; the held request lived in memory only
    second = Chain(tmp_path)
    second.conversation = first.conversation
    second.resolver.mark_surfaced(interaction.id, first.conversation)
    second.say("yes")
    second.tick()

    assert second.persisted_nothing()
    assert "expired" in second.jobs.get(job.id).final_result


def test_the_held_request_lives_only_in_memory_and_is_bounded(chain):
    chain.say("Remember that I have diabetes")
    assert "diabetes" not in chain.disk_without_chat_history()

    from app.memory_policy.service import MAX_PENDING

    for i in range(MAX_PENDING + 5):
        chain.registry.execute("remember", {"key": f"health{i}", "value": f"has condition {i} diabetes"})
    assert chain.service.pending_count() <= MAX_PENDING


def test_the_existing_confirmation_paths_are_unchanged_for_ordinary_jobs(chain):
    assert chain.service.command_reply("Remember that my dog is called Max.") == "Noted - I'll remember that."
    assert chain.jobs.list() == [] and chain.service.pending_count() == 0
    chain.say("hello")
    assert chain.jobs.list() == []


def test_the_confirmation_uses_the_existing_machinery_and_adds_no_mechanism_of_its_own():
    import pathlib

    import app.memory_policy.confirmation as confirmation

    source = pathlib.Path(confirmation.__file__).read_text(encoding="utf-8")
    imports = "\n".join(l for l in source.splitlines() if l.startswith(("import ", "from ")))
    assert "app.orchestration" in imports and "app.tools.base" in imports
    assert not re.search(r"subprocess|httpx|providers|threading|sqlite3", imports)


def test_the_interaction_store_can_create_only_a_well_formed_waiting_job(chain):
    from app.orchestration.base import Plan, PlanStep

    job = SQLiteJobStore.build_job("g", status=JobStatus.AWAITING_CONFIRMATION)
    with pytest.raises(ValueError):  # no plan
        chain.interactions.create_job_awaiting_confirmation(job, prompt="p?", step_ids=["s1"])
    job.plan = Plan(steps=[PlanStep(id="s1", tool_name=COMMIT_TOOL_NAME, permission="confirm")])
    with pytest.raises(ValueError):  # step not pending
        chain.interactions.create_job_awaiting_confirmation(job, prompt="p?", step_ids=["s1"])
    assert chain.jobs.list() == []  # nothing half-created
    job.pending_step_ids = ["s1"]
    created, interaction = chain.interactions.create_job_awaiting_confirmation(job, prompt="Go ahead?", step_ids=["s1"])
    assert created.id == job.id and interaction.type is InteractionType.CONFIRMATION and interaction.job_id == job.id
    assert chain.interactions.list_pending_for_job(job.id)[0].id == interaction.id
