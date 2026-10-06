"""
Scheduled Jobs V1 / Slice 4 - creating, approving and managing Scheduled Jobs from chat.

The real chain, end to end: JarvisCore -> the Scheduled Job door -> the service -> the real stores (one SQLite file, as in production) ->
the EXISTING confirmation (a waiting Job + interaction, resolved by the real InteractionResolver and JobEngine) -> the CONFIRM-level commit
tool -> the Slice 3 scheduler -> the Slice 2 adapter -> an ordinary Job that the same JobEngine runs under the READ-only profile. Time is a
controllable clock; no network, no model, no second path of any kind.

The invariants under all of them:
    nothing is scheduled by being asked for - only an approved confirmation of THAT proposal makes it live, atomically;
    approval grants an intent, not authority - every run still goes through the profile, the registry, plan validation and audit.
"""

import threading
from datetime import datetime, timedelta, timezone

import pytest

from app.core import JarvisCore
from app.handle_this.factory import create_default_intake
from app.handle_this.routing import ObjectiveAwareCapabilityRouter
from app.handle_this.service import HandleThisService
from app.memory.sqlite_store import SQLiteMemoryStore
from app.orchestration.base import JobStatus
from app.orchestration.interaction_resolver import InteractionResolver
from app.orchestration.interactions import InteractionStatus
from app.scheduled_execution.scheduler import ScheduledJobScheduler
from app.scheduled_jobs.base import RunOutcome, ScheduledJobStatus as S, StatusReason as R
from app.scheduled_jobs.policy import Limits
from app.scheduled_jobs.sqlite_store import SQLiteScheduledJobStore
from app.scheduled_operations.operations import (
    COMMIT_TOOL_NAME,
    ScheduledJobApprovalConfirmer,
    ScheduledJobOperations,
)
from app.scheduled_operations.service import (
    UNATTENDED_REFUSAL,
    ScheduledJobRequestError,
    ScheduledJobService,
    understander,
)
from app.scheduled_operations.tools import CommitScheduledJobApprovalTool
from app.scheduling.timezones import ZoneChoice
from app.tools.base import PermissionLevel, ToolResultStatus
from test_handle_this_integration import ChatProvider
from test_orchestration_engine import step
from test_run_profiles import FakeTool
from test_scheduled_execution_adapter import FIRST, RESEARCH_OBJECTIVE, RESEARCH_TOOLS, Rig

UTC = timezone.utc
DUBAI = "Asia/Dubai"
EVERY_MORNING = "Every morning at 8, research AI robotics news and give me a summary."
TOMORROW_5PM = "Tomorrow at 5 PM, research X and report the findings."
READ_TOOLS = (*RESEARCH_TOOLS, "list_events", "list_notes", "list_reminders")
ASKED = "Please confirm - say yes or no."


def objective_for(text, family="research", **changes):
    objective = dict(RESEARCH_OBJECTIVE, original_request=text, goal_family=family, required_families=[family])
    objective.update(changes)
    return objective


class Chain(Rig):
    """The Slice 2 rig (real stores, registry, JobEngine, adapter; the clock is Monday 2026-10-05 14:01 UTC = 18:01 in Dubai) plus the
    creation layer, a real JarvisCore, and a real scheduler."""

    def __init__(self, tmp_path, *, understand=None, tools=None, limits=None, plans=(), router=None):
        super().__init__(str(tmp_path / "jobs.db"), tools=[FakeTool(n) for n in (READ_TOOLS if tools is None else tools)], plans=list(plans))
        self.path = str(tmp_path / "jobs.db")
        self.understood = []

        def default_understand(text, conversation_id):
            self.understood.append(text)
            return objective_for(text)

        self.limits = limits or Limits(max_concurrent=50, max_runs_per_day=1000)
        self.service = ScheduledJobService(
            self.store, limits=self.limits, understand=understand or default_understand, check_objective=self.adapter.check_objective,
            clock=self.clock, zone_resolver=lambda: ZoneChoice(DUBAI, "test"), proposal_ttl=timedelta(hours=24),
        )
        self.confirmer = ScheduledJobApprovalConfirmer(self.interactions)
        self.ops = ScheduledJobOperations(self.service, self.confirmer)
        self.commit = CommitScheduledJobApprovalTool(self.ops)
        self.registry.register(self.commit)
        self.resolver = InteractionResolver(self.interactions, self.jobs, self.engine)
        self.confirmer.mark_surfaced = self.resolver.mark_surfaced
        self.memory = SQLiteMemoryStore(str(tmp_path / "memory.db"))
        self.core = JarvisCore(
            provider=ChatProvider(), memory=self.memory, tools=self.registry, interaction_resolver=self.resolver,
            scheduled_job_operations=self.ops,
        )
        if router:
            self.core.capability_router = router(self.core.capability_router)
        self.conversation = self.memory.create_conversation()
        self.scheduler = ScheduledJobScheduler(
            scheduled_jobs=self.store, jobs=self.jobs, adapter=self.adapter, limits=self.limits, enabled=lambda: True,
            device_trusted=lambda device_id: True, clock=self.clock,
        )

    # -- the user's side ---------------------------------------------------------------------------------------------------------

    def say(self, message, *, device_context=None):
        return self.core.handle_chat_with_job_id(message, self.conversation, device_context=device_context)[0]

    def tick(self, times=3):
        for _ in range(times):
            self.engine.tick()

    def approve(self):
        """Say yes, and let the real engine run the confirmed step."""
        reply = self.say("yes")
        self.tick()
        return reply

    def propose(self, message=EVERY_MORNING):
        reply = self.say(message)
        assert reply.endswith(ASKED), reply
        return self.only_scheduled()

    def activate(self, message=EVERY_MORNING):
        self.propose(message)
        self.approve()
        return self.only_scheduled()

    # -- looking -------------------------------------------------------------------------------------------------------------------

    def all_scheduled(self):
        return self.store.list()

    def only_scheduled(self):
        (found,) = self.store.list()
        return found

    def confirmation_jobs(self):
        return [j for j in self.jobs.list() if j.goal == "Schedule a job"]

    def research_jobs(self):
        return [j for j in self.jobs.list() if j.goal != "Schedule a job"]

    def pending_confirmation(self):
        (job,) = self.confirmation_jobs()
        (interaction,) = self.interactions.list_for_job(job.id)
        return job, interaction


@pytest.fixture
def chain(tmp_path):
    return Chain(tmp_path)


# ====================================== creating: asked, never just done ========================================================


def test_a_recurring_request_is_understood_stored_as_a_proposal_and_ASKED_in_plain_words(chain):
    reply = chain.say(EVERY_MORNING)

    assert reply.startswith("Here's what I'll set up:") and reply.endswith(ASKED)
    assert "research AI robotics news and give me a summary" in reply  # what
    assert "every day at 8:00 AM (Asia/Dubai)" in reply  # when, and that it repeats
    assert "First run: Tuesday 6 October, 8:00 AM" in reply
    assert "up to 90 days" in reply and "I'll ask you before continuing" in reply  # how long
    assert "read-only" in reply and "can't change anything, send anything or control your computer" in reply  # the V1 scope
    assert "normal security checks" in reply  # approval is not a bypass
    assert "{" not in reply and "objective" not in reply.lower()  # natural language, not a structure

    sj = chain.only_scheduled()
    assert sj.status is S.PROPOSED and sj.due_at is None and sj.approved_at is None and sj.expires_at is None
    assert sj.intent_text == "research AI robotics news and give me a summary" and sj.profile == "unattended_v1"
    assert sj.created_via_conversation_id == chain.conversation
    job, interaction = chain.pending_confirmation()
    assert job.status is JobStatus.AWAITING_CONFIRMATION and interaction.status is InteractionStatus.PENDING
    assert [s.tool_name for s in job.plan.steps] == [COMMIT_TOOL_NAME] and job.plan.steps[0].permission == "confirm"
    assert job.plan.steps[0].arguments == {"scheduled_job_id": sj.id}
    assert interaction.detail["origin"] == "scheduled_jobs" and interaction.detail["scheduled_job_id"] == sj.id
    assert chain.planner.calls == []  # no planner was involved in asking
    assert chain.research_jobs() == []  # and no Job that does the work exists


def test_a_one_time_request_is_asked_too_and_says_it_runs_once(chain):
    reply = chain.say(TOMORROW_5PM)
    assert reply.endswith(ASKED) and "It runs once, then it's done." in reply and "90 days" not in reply
    assert "on Tuesday 6 October 2026 at 5:00 PM (Asia/Dubai)" in reply
    sj = chain.only_scheduled()
    assert sj.status is S.PROPOSED and sj.schedule.kind.value == "once"


def test_the_headline_examples_are_all_supported_and_the_instruction_keeps_its_closing_words(chain):
    chain.say("Every Monday at 6 PM, check my upcoming calendar and summarize tomorrow.")
    assert chain.only_scheduled().intent_text == "check my upcoming calendar and summarize tomorrow"
    assert chain.only_scheduled().schedule.describe() == "every Monday at 6:00 PM (Asia/Dubai)"


def test_nothing_runs_from_a_proposal_however_late_it_gets(chain):
    chain.propose()
    chain.clock.advance(days=3)  # long past the would-be first run
    result = chain.scheduler.tick()
    assert chain.research_jobs() == [] and chain.ledger_count() == 0
    assert chain.only_scheduled().status is not S.ACTIVE


@pytest.mark.parametrize(
    "message, expect",
    [
        ("Research X every 30 minutes", "at least 60 minutes apart"),
        ("Research X every 59 minutes", "at least 60 minutes apart"),
        ("every day at 7, research X", "AM or PM"),
    ],
)
def test_an_invalid_schedule_is_refused_in_words_and_nothing_is_stored_or_asked(chain, message, expect):
    reply = chain.say(message)
    assert expect in reply
    assert chain.all_scheduled() == [] and chain.confirmation_jobs() == []


def test_an_hourly_schedule_is_the_smallest_accepted(chain):
    assert chain.say("Research AI news every 1 hour").endswith(ASKED)
    assert chain.only_scheduled().schedule.describe() == "every 1 hour"


def test_a_time_with_no_day_that_has_passed_today_rolls_to_tomorrow_and_says_so(chain):
    reply = chain.say("At 5 PM research X and report the findings")  # it is already 6 PM in Dubai
    assert "has already passed today, so the first run is tomorrow" in reply


def test_a_date_more_than_two_years_away_is_refused(chain):
    assert "more than two years away" in chain.say("Research X on 2031-01-01 at 9 AM")
    assert chain.all_scheduled() == []


def test_a_request_the_understanding_cannot_make_out_is_answered_with_its_own_words_and_creates_nothing(tmp_path):
    def unclear(text, conversation_id):
        raise ScheduledJobRequestError("Could you tell me a bit more about the outcome you want?")

    chain = Chain(tmp_path, understand=unclear)
    assert chain.say(EVERY_MORNING) == "Could you tell me a bit more about the outcome you want?"
    assert chain.all_scheduled() == [] and chain.confirmation_jobs() == []


def test_what_an_unattended_read_only_run_could_not_do_is_REFUSED_never_weakened(tmp_path):
    # Reading a page is on offer to an unattended run; CLICKING is not - so an understanding that needs a click is refused by the very
    # boundary check every run applies, whatever the door's wording looked like.
    needs_a_click = lambda text, cid: objective_for(text, "computer_workflow", original_request="open my browser and click the login button")
    chain = Chain(tmp_path, understand=needs_a_click)
    assert chain.say("Every morning at 8, research the login page") == UNATTENDED_REFUSAL
    assert chain.all_scheduled() == [] and chain.confirmation_jobs() == []


@pytest.mark.parametrize(
    "message",
    [
        "Every morning at 8, add a task to review my goals",
        "Send me the news every morning at 8",
        "Every Friday at 5 PM, email my summary to the team",
        "Every day at 9, buy more printer filament",
        "Every day at 9 AM, open my browser and click the login button",
    ],
)
def test_a_request_to_DO_something_is_refused_aloud_before_anything_is_stored_or_even_understood(chain, message):
    assert chain.say(message) == UNATTENDED_REFUSAL
    assert chain.all_scheduled() == [] and chain.confirmation_jobs() == [] and chain.understood == []


def test_a_required_tool_that_is_not_available_is_refused(tmp_path):
    chain = Chain(tmp_path, tools=("list_tasks",))  # the research family has nothing to run on
    assert chain.say(EVERY_MORNING) == UNATTENDED_REFUSAL
    assert chain.all_scheduled() == []


def test_stale_world_state_is_never_stored_only_what_the_request_itself_said(tmp_path):
    constraints = [
        {"topic": "budget", "text": "up to 2500 AED", "source": "context"},  # known from the user's context just now: stale tomorrow
        {"topic": "timeframe", "text": "this week only", "source": "request"},  # said in the request: stable
    ]
    chain = Chain(tmp_path, understand=lambda text, cid: objective_for(text, constraints=constraints))
    chain.propose()
    stored = chain.only_scheduled().objective
    assert stored["constraints"] == [constraints[1]]
    assert "request_context" not in str(stored) and "2500" not in str(stored)
    assert set(stored) == set(RESEARCH_OBJECTIVE)  # exactly the understood request: no plan, no tools, no context snapshot


def test_the_creating_device_is_remembered_only_when_it_is_authenticated(chain):
    class Context:
        is_authenticated = True
        device_id = "dev-1"

    chain.say(EVERY_MORNING, device_context=Context())
    assert chain.only_scheduled().created_via_device_id == "dev-1"


def test_an_anonymous_request_records_no_device(chain):
    chain.propose()
    assert chain.only_scheduled().created_via_device_id is None


def test_the_same_request_twice_does_not_create_two_proposals(chain):
    chain.propose()
    again = chain.say(EVERY_MORNING)
    assert "already waiting for your answer" in again
    assert len(chain.all_scheduled()) == 1 and len(chain.confirmation_jobs()) == 1


def test_the_same_request_after_it_is_live_says_so_and_adds_nothing(chain):
    chain.activate()
    again = chain.say(EVERY_MORNING)
    assert again.startswith("You already have that scheduled") and len(chain.all_scheduled()) == 1


def test_the_number_of_scheduled_jobs_is_capped(tmp_path):
    chain = Chain(tmp_path, limits=Limits(max_open=2, max_concurrent=50, max_runs_per_day=1000))
    chain.say("Research A every day at 8 AM")
    chain.say("Research B every day at 9 AM")
    reply = chain.say("Research C every day at 10 AM")
    assert "already have 2 scheduled jobs" in reply and len(chain.all_scheduled()) == 2


def test_a_confirmation_that_cannot_be_asked_leaves_nothing_proposed(chain):
    chain.confirmer.ask = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("store down"))
    assert "nothing was scheduled" in chain.say(EVERY_MORNING)
    assert chain.only_scheduled().status is S.CANCELLED and chain.only_scheduled().status_reason is R.DECLINED
    chain.ops.confirmer = None
    assert "nothing was scheduled" in chain.say("Research other things every day at 9 AM")


# ====================================== approving: atomic, once, only the confirmed proposal ====================================


def test_saying_yes_makes_it_live_atomically_with_its_first_run_and_its_90_day_lifetime(chain):
    sj = chain.propose()
    job, interaction = chain.pending_confirmation()

    reply = chain.say("yes")
    chain.tick()

    assert reply.startswith("Approved")
    live = chain.only_scheduled()
    assert live.status is S.ACTIVE and live.id == sj.id and live.status_reason is None
    assert live.approved_at == chain.clock.now and live.approval_interaction_id == interaction.id
    assert live.due_at == datetime(2026, 10, 6, 4, 0, tzinfo=UTC)  # 8:00 AM in Dubai, tomorrow
    assert live.expires_at == chain.clock.now + timedelta(days=90)
    done = chain.jobs.get(job.id)
    assert done.status is JobStatus.COMPLETED and "Scheduled: " in done.final_result and "First run: Tuesday 6 October" in done.final_result
    assert chain.research_jobs() == []  # approving started no work


def test_a_one_time_approval_has_no_lifetime_of_its_own(chain):
    chain.propose(TOMORROW_5PM)
    chain.approve()
    live = chain.only_scheduled()
    assert live.status is S.ACTIVE and live.expires_at is None and live.due_at == datetime(2026, 10, 6, 13, 0, tzinfo=UTC)


def test_saying_no_schedules_nothing_and_the_proposal_can_never_be_approved_afterwards(chain):
    chain.propose()
    job, interaction = chain.pending_confirmation()
    chain.say("no")
    chain.tick(5)
    assert chain.only_scheduled().status is S.PROPOSED  # never active
    assert chain.commit.execute(chain.only_scheduled().id, source_job_id=job.id).status is ToolResultStatus.ERROR
    assert chain.only_scheduled().status is S.PROPOSED
    chain.clock.advance(hours=25)
    chain.service.sweep()
    assert chain.only_scheduled().status is S.EXPIRED


def test_a_second_approval_of_the_same_confirmation_is_harmless(chain):
    sj = chain.propose()
    chain.approve()
    live = chain.only_scheduled()
    job, interaction = chain.pending_confirmation()

    result = chain.commit.execute(sj.id, source_job_id=job.id)  # the same approval, seen again

    assert result.status is ToolResultStatus.SUCCESS and "already scheduled" in result.message
    again = chain.only_scheduled()
    assert again.version == live.version and again.events == live.events  # nothing was done twice


def test_the_commit_tool_cannot_be_used_directly_or_forged(chain):
    sj = chain.propose()
    job, interaction = chain.pending_confirmation()

    assert chain.commit.execute(sj.id).status is ToolResultStatus.ERROR  # no job provenance
    assert chain.commit.execute(sj.id, source_job_id="some-other-job").status is ToolResultStatus.ERROR  # not the Job that asked
    assert chain.commit.execute(sj.id, source_job_id=job.id).status is ToolResultStatus.ERROR  # asked, but not APPROVED
    assert chain.commit.execute("made-up", source_job_id=job.id).status is ToolResultStatus.ERROR
    chat = chain.say(f"use tool {COMMIT_TOOL_NAME} scheduled_job_id={sj.id}")
    assert chain.only_scheduled().status is S.PROPOSED and "Scheduled:" not in chat


def test_a_confirmation_approved_for_one_proposal_cannot_approve_another(chain):
    first = chain.propose()
    second = chain.service.propose("research other things", _when("every day at 9 AM"), conversation_id=chain.conversation).scheduled_job
    job, interaction = chain.pending_confirmation()
    chain.interactions.resolve(interaction.id, expected_version=interaction.version, response="yes")  # approved - for `first`

    assert chain.commit.execute(second.id, source_job_id=job.id).status is ToolResultStatus.ERROR
    assert chain.store.get(second.id).status is S.PROPOSED
    assert chain.commit.execute(first.id, source_job_id=job.id).status is ToolResultStatus.SUCCESS
    assert chain.store.get(first.id).status is S.ACTIVE and chain.store.get(second.id).status is S.PROPOSED


def test_an_approved_confirmation_of_another_origin_cannot_approve_a_scheduled_job(chain):
    from app.memory_policy.confirmation import SensitiveWriteConfirmer

    sj = chain.propose()
    foreign_job_id = SensitiveWriteConfirmer(chain.interactions).ask("pending-1", None)
    (foreign,) = chain.interactions.list_for_job(foreign_job_id)
    chain.interactions.resolve(foreign.id, expected_version=foreign.version, response="yes")

    assert chain.confirmer.approved(foreign_job_id, sj.id) is None
    assert chain.commit.execute(sj.id, source_job_id=foreign_job_id).status is ToolResultStatus.ERROR
    assert chain.only_scheduled().status is S.PROPOSED


def test_an_approval_after_the_proposal_expired_schedules_nothing(chain):
    sj = chain.propose()
    job, interaction = chain.pending_confirmation()
    chain.interactions.resolve(interaction.id, expected_version=interaction.version, response="yes")
    chain.clock.advance(hours=25)  # the proposal (and its confirmation window) are over
    result = chain.commit.execute(sj.id, source_job_id=job.id)
    assert result.status is ToolResultStatus.ERROR and "expired" in result.message
    assert chain.only_scheduled().status is S.EXPIRED


def test_approval_checks_again_that_the_time_has_not_passed(chain):
    sj = chain.propose(TOMORROW_5PM)
    job, interaction = chain.pending_confirmation()
    chain.interactions.resolve(interaction.id, expected_version=interaction.version, response="yes")
    chain.clock.advance(hours=23, minutes=30)  # inside the 24-hour confirmation window, but 5 PM tomorrow (13:00 UTC) has gone by
    result = chain.commit.execute(sj.id, source_job_id=job.id)
    assert result.status is ToolResultStatus.ERROR and "Nothing was scheduled" in result.message
    assert chain.store.get(sj.id).status is not S.ACTIVE


def test_approval_checks_again_that_an_unattended_run_is_still_possible(chain):
    sj = chain.propose()
    job, interaction = chain.pending_confirmation()
    chain.interactions.resolve(interaction.id, expected_version=interaction.version, response="yes")
    for name in READ_TOOLS:  # the capability disappeared between asking and approving
        chain.registry.unregister(name)
    result = chain.commit.execute(sj.id, source_job_id=job.id)
    assert result.status is ToolResultStatus.ERROR and UNATTENDED_REFUSAL in result.message
    assert chain.store.get(sj.id).status is S.PROPOSED


def test_approving_and_cancelling_race_cleanly_whichever_wins(chain):
    sj = chain.propose()
    chain.service.cancel(sj.id)  # the user withdrew it first
    with pytest.raises(ScheduledJobRequestError):
        chain.service.approve(sj.id, "int-late")
    assert chain.store.get(sj.id).status is S.CANCELLED  # never ACTIVE

    other = chain.service.propose("research other things", _when("every day at 9 AM")).scheduled_job
    chain.service.approve(other.id, "int-1")
    chain.service.cancel(other.id)  # approved first, then the user cancelled the live one
    assert chain.store.get(other.id).status is S.CANCELLED and chain.store.get(other.id).status_reason is R.USER


def test_two_independent_approvals_racing_on_separate_connections_activate_it_exactly_once(tmp_path):
    chain = Chain(tmp_path)
    sj = chain.propose()
    second_store = SQLiteScheduledJobStore.open(chain.path)  # another process's view of the same file
    second = ScheduledJobService(
        second_store, limits=chain.limits, understand=lambda *a: {}, check_objective=chain.adapter.check_objective,
        clock=chain.clock, zone_resolver=lambda: ZoneChoice(DUBAI, "test"),
    )
    outcomes, errors = [], []
    start = threading.Barrier(2)

    def approve(service):
        try:
            start.wait()
            outcomes.append(service.approve(sj.id, "int-race"))
        except BaseException as exc:  # reported on the main thread
            errors.append(exc)

    threads = [threading.Thread(target=approve, args=(s,)) for s in (chain.service, second)]
    [t.start() for t in threads]
    [t.join(timeout=30) for t in threads]

    assert errors == [] and len(outcomes) == 2
    assert sorted(o.already for o in outcomes) == [False, True]  # one did it; the other saw it done
    live = chain.store.get(sj.id)
    assert live.status is S.ACTIVE and live.approval_interaction_id == "int-race"
    assert [e.kind for e in live.events].count("approved") == 1  # approved once


def test_the_scheduler_never_sees_a_proposal_or_a_cancelled_one_only_the_approved_one_when_due(chain):
    sj = chain.propose()
    chain.clock.now = datetime(2026, 10, 6, 5, 0, tzinfo=UTC)  # after the would-be first run
    chain.scheduler.tick()
    assert chain.research_jobs() == [] and chain.ledger_count() == 0  # a PROPOSAL is invisible to it
    assert chain.store.list_due(chain.clock.now) == []

    chain.clock.now = FIRST + timedelta(minutes=1)
    chain.approve()
    assert chain.store.list_due(chain.clock.now) == []  # approved, but not due yet
    chain.scheduler.tick()
    assert chain.research_jobs() == []
    chain.clock.now = datetime(2026, 10, 6, 4, 1, tzinfo=UTC)
    assert [s.id for s in chain.store.list_due(chain.clock.now)] == [sj.id]  # now it is


# ====================================== the whole path ============================================================================


def test_the_full_path_user_request_to_approval_to_a_normal_Job_run_by_the_real_engine(tmp_path):
    chain = Chain(tmp_path, plans=[[step("s1", "web_search", {"q": "AI robotics news"})]])

    chain.activate()  # request -> proposal -> confirmation -> approval
    sj = chain.only_scheduled()
    chain.clock.now = sj.due_at + timedelta(minutes=1)
    chain.scheduler.tick()  # discovers the due occurrence and launches it through the adapter

    (job,) = chain.research_jobs()
    assert job.goal == "research AI robotics news and give me a summary" and job.run_profile == "unattended_v1"
    assert job.conversation_id is None and job.status is JobStatus.PENDING
    assert not job.autonomous_permissions
    chain.tick()  # the SAME JobEngine plans it (existing Planner) and runs it through the existing ToolRegistry
    assert chain.jobs.get(job.id).status is JobStatus.COMPLETED
    assert chain.registry.executed.count("web_search") == 1
    chain.scheduler.tick()  # settles it
    run = chain.store.runs_for(sj.id)[0]
    assert run.outcome is RunOutcome.CREATED and run.result_state == "completed" and run.job_id == job.id
    live = chain.store.get(sj.id)
    assert live.status is S.ACTIVE and live.consecutive_failures == 0 and live.due_at == sj.due_at + timedelta(days=1)


def test_a_one_time_job_runs_once_and_never_again(tmp_path):
    chain = Chain(tmp_path, plans=[[step("s1", "web_search", {"q": "x"})]])
    chain.activate(TOMORROW_5PM)
    sj = chain.only_scheduled()
    chain.clock.now = sj.due_at + timedelta(minutes=1)
    chain.scheduler.tick()
    chain.tick()
    chain.scheduler.tick()
    assert len(chain.research_jobs()) == 1
    done = chain.store.get(sj.id)
    assert done.status is S.COMPLETED and done.status_reason is R.ONE_TIME_DONE and done.expires_at is None

    chain.clock.advance(days=30)
    chain.scheduler.tick()
    assert len(chain.research_jobs()) == 1 and chain.ledger_count() == 1


def test_the_real_Handle_This_understands_the_instruction_and_its_context_is_not_kept(tmp_path):
    chain = Chain(tmp_path)
    handle_this = HandleThisService(create_default_intake(tools=chain.registry))
    chain.service._understand = understander(lambda: handle_this)

    chain.propose("Every Monday at 6 PM, check my upcoming calendar and summarize tomorrow.")

    stored = chain.only_scheduled().objective
    assert stored["original_request"] == "check my upcoming calendar and summarize tomorrow"
    assert stored["required_families"] == ["analysis", "organization"] and "request_context" not in str(stored)


def test_what_the_real_Handle_This_cannot_take_on_is_refused_and_a_vague_request_asks(tmp_path):
    chain = Chain(tmp_path)
    handle_this = HandleThisService(create_default_intake(tools=chain.registry))
    chain.service._understand = understander(lambda: handle_this)
    assert chain.say("Every day at 9 AM, buy a laptop") == UNATTENDED_REFUSAL
    assert "what you'd like me to do" in chain.say("Every day at 9 AM, check the thing")
    assert chain.all_scheduled() == []


# ====================================== the door in the core ====================================================================


def test_ordinary_conversation_is_left_alone(chain):
    assert chain.say("every day I wonder about robots") == "an ordinary chat reply"
    assert chain.say("Tomorrow, research X") != "" and chain.all_scheduled() == []


def test_a_multi_step_recurring_request_is_claimed_not_started_once_immediately(tmp_path):
    chain = Chain(tmp_path, router=ObjectiveAwareCapabilityRouter)
    chain.core.handle_this = HandleThisService(create_default_intake(tools=chain.registry))
    reply = chain.say("Every Monday at 8 AM, research the best 3D printers, compare their specs and give me a shortlist report")
    assert reply.endswith(ASKED)
    assert chain.research_jobs() == []  # Handle This did NOT start it now; it was proposed for approval
    assert chain.only_scheduled().status is S.PROPOSED


def test_an_explicit_tool_command_never_reaches_the_door(chain):
    chain.say("use tool web_search q=research x every day at 8 AM")
    assert chain.all_scheduled() == []
    # `analyze` is BOTH a registered tool and a read verb: written as a command it is the tool's, never a scheduling request
    reply = chain.say("use tool analyze hello every day at 8 AM")
    assert chain.all_scheduled() == [] and chain.confirmation_jobs() == [] and "Here's what I'll set up" not in reply


# ====================================== managing ===========================================================================


def test_listing_shows_what_is_scheduled_when_next_and_the_state(chain):
    assert chain.say("what scheduled jobs do I have?") == "You don't have any scheduled jobs."
    chain.activate()
    chain.service.propose("research prices", _when("every day at 9 AM"), conversation_id=None)  # still only a proposal: not listed
    listing = chain.say("what scheduled jobs do I have?")
    assert listing.startswith("You have 1 scheduled job:")
    assert "research AI robotics news and give me a summary - every day at 8:00 AM (Asia/Dubai); next Tuesday 6 October, 8:00 AM" in listing
    assert "research prices" not in listing


def test_inspecting_one_says_what_when_state_runs_failures_and_the_approved_period(chain):
    chain.activate()
    detail = chain.say("show my robotics scheduled job")
    assert "What: research AI robotics news and give me a summary" in detail
    assert "State: every day at 8:00 AM (Asia/Dubai); next Tuesday 6 October" in detail
    assert "Runs so far: 0; last: it hasn't run yet" in detail
    assert "Approved until: " in detail and "Read-only" in detail and "Failed runs" not in detail
    sj = chain.only_scheduled()
    # (how the failure streak is built is Slice 3's; here only that it is surfaced)
    with chain.jobs.lock:
        chain.jobs.connection.execute("UPDATE scheduled_jobs SET consecutive_failures = 2, runs_total = 5 WHERE id = ?", (sj.id,))
        chain.jobs.connection.commit()
    assert "Failed runs in a row: 2" in chain.say("show my robotics scheduled job")


def test_pause_stops_it_running_and_resume_goes_to_the_next_future_run_with_no_replay(tmp_path):
    chain = Chain(tmp_path, plans=[[step("s1", "web_search")]] * 3)
    chain.activate()
    sj = chain.only_scheduled()

    assert chain.say("pause my robotics scheduled job").startswith("Paused:")
    paused = chain.only_scheduled()
    assert paused.status is S.PAUSED and paused.status_reason is R.USER
    chain.clock.now = sj.due_at + timedelta(days=5)  # five occurrences come and go while paused (the fifth is exactly now: resuming goes strictly AFTER it)
    chain.scheduler.tick()
    assert chain.research_jobs() == [] and chain.ledger_count() == 0
    assert "paused by you" in chain.say("what scheduled jobs do I have?")

    reply = chain.say("resume my robotics scheduled job")
    assert reply.startswith("Resumed:") and "Next run: Monday 12 October, 8:00 AM" in reply
    resumed = chain.only_scheduled()
    assert resumed.status is S.ACTIVE and resumed.due_at == datetime(2026, 10, 12, 4, 0, tzinfo=UTC) and chain.ledger_count() == 0


def test_cancelling_ends_it_for_good(chain):
    chain.activate()
    assert chain.say("cancel my robotics scheduled job").startswith("Cancelled:")
    assert chain.only_scheduled().status is S.CANCELLED
    assert chain.say("what scheduled jobs do I have?") == "You don't have any scheduled jobs."
    assert "couldn't find a scheduled job about \"robotics\"" in chain.say("resume my robotics scheduled job")
    chain.clock.advance(days=2)
    chain.scheduler.tick()
    assert chain.research_jobs() == []


def test_an_expired_schedule_cannot_be_resumed_even_though_it_was_paused_and_nothing_replays(tmp_path):
    chain = Chain(tmp_path)
    chain.activate()
    sj = chain.only_scheduled()
    chain.say("pause my robotics scheduled job")
    chain.clock.now = sj.expires_at + timedelta(minutes=1)  # pausing never stopped the clock on the approved period
    reply = chain.say("resume my robotics scheduled job")
    assert "approved for" in reply and "has ended" in reply and "schedule it again" in reply
    assert chain.only_scheduled().status is S.PAUSED  # unchanged: only a fresh approval can start another period
    chain.scheduler.tick()  # the existing Slice 3 sweep completes it
    assert chain.only_scheduled().status is S.COMPLETED and chain.only_scheduled().status_reason is R.LIFETIME_ENDED
    assert 'say "renew my research ai robotics scheduled job" to start a new one' in chain.say("what scheduled jobs do I have?")


def test_a_paused_one_time_job_keeps_no_90_day_lifetime_and_can_still_be_resumed_while_its_time_is_ahead(chain):
    chain.activate(TOMORROW_5PM)
    chain.say("pause my research scheduled job")
    paused = chain.only_scheduled()
    assert paused.status is S.PAUSED and paused.expires_at is None
    chain.clock.advance(days=40)  # far beyond 90 days would matter for a recurring one - one-time has none
    assert chain.only_scheduled().expires_at is None
    assert "nothing left to run" in chain.say("resume my research scheduled job")  # its one time has gone by, not "expired"


def test_resuming_a_paused_recurring_job_inside_its_period_works(chain):
    chain.activate()
    chain.say("pause my robotics scheduled job")
    chain.clock.advance(days=10)
    assert chain.say("resume my robotics scheduled job").startswith("Resumed:")


def test_several_matches_are_listed_and_nothing_is_changed_and_all_is_declined(chain):
    chain.say("Research AI robotics news every day at 8 AM")
    chain.approve()
    chain.say("Research AI robotics prices every day at 9 AM")
    chain.approve()
    asked = chain.say("pause my robotics scheduled job")
    assert "more than one scheduled job" in asked and asked.count("\n- ") == 2
    assert {s.status for s in chain.all_scheduled()} == {S.ACTIVE}
    assert "won't change several" in chain.say("cancel all my scheduled jobs")
    assert "won't change several" in chain.say("pause my scheduled jobs")
    assert {s.status for s in chain.all_scheduled()} == {S.ACTIVE}
    assert chain.say("pause my prices scheduled job").startswith("Paused:")


def test_managing_something_that_is_not_there_changes_nothing(chain):
    assert "couldn't find a scheduled job" in chain.say("pause my robotics scheduled job")
    chain.activate()
    assert "couldn't find a scheduled job about \"weather\"" in chain.say("cancel my weather scheduled job")
    assert chain.only_scheduled().status is S.ACTIVE


# ====================================== security: approval adds no capability ====================================================


def test_approval_does_not_expand_what_a_run_may_do(tmp_path):
    chain = Chain(tmp_path, plans=[[step("s1", "create_task", {"q": "x"})]])
    write = FakeTool("create_task", PermissionLevel.SAFE_ACTION)
    chain.registry.register(write)
    chain.registry.register(FakeTool("send_email", PermissionLevel.CONFIRM))
    chain.activate()
    sj = chain.only_scheduled()
    chain.clock.now = sj.due_at + timedelta(minutes=1)
    chain.scheduler.tick()
    (job,) = chain.research_jobs()
    assert job.run_profile == "unattended_v1" and not job.autonomous_permissions and job.conversation_id is None

    chain.tick(4)  # the planner proposed a WRITE; the profile and plan validation refuse it
    assert write.calls == [] and "create_task" not in chain.registry.executed
    offered = chain.planner.calls[0]["tools"]
    assert set(offered) <= {"web_search", "observe_page", "verify_observation", "analyze", "list_tasks", "list_events", "list_notes", "list_reminders"}
    assert "create_task" not in offered and "send_email" not in offered and COMMIT_TOOL_NAME not in offered
    assert chain.jobs.get(job.id).status is JobStatus.FAILED  # it could not do what its plan asked: that is what the profile is for


def test_the_commit_tool_is_confirm_level_never_plannable_never_automatable_and_wants_provenance(chain):
    tool = chain.commit
    assert tool.permission is PermissionLevel.CONFIRM and tool.plannable is False and tool.automatable is False
    assert tool.wants_job_provenance is True
    assert COMMIT_TOOL_NAME not in {t.name for t in chain.registry.list_tools() if t.plannable}


def test_the_runtime_boundary_stays_authoritative_after_approval(chain):
    chain.activate()
    sj = chain.only_scheduled()
    for name in READ_TOOLS:  # what the objective needs disappears AFTER it was approved
        chain.registry.unregister(name)
    chain.clock.now = sj.due_at + timedelta(minutes=1)
    chain.scheduler.tick()
    assert chain.research_jobs() == []  # refused at run time: no Job
    run = chain.store.runs_for(sj.id)[0]
    assert run.outcome is RunOutcome.SKIPPED_BLOCKED and run.job_id is None
    assert chain.store.get(sj.id).status is S.ACTIVE  # skipped, not paused: future occurrences are evaluated normally


def test_the_stored_intent_carries_no_plan_no_tools_and_no_steps(chain):
    chain.activate()
    stored = chain.only_scheduled()
    assert stored.profile == "unattended_v1"
    assert stored.objective["required_families"] == ["research"]
    assert "plan" not in stored.objective and "tools" not in stored.objective and "steps" not in stored.objective


def _when(text):
    from app.reminders.language import split_leading_when

    return split_leading_when(text)[0]


# ====================================== more of the creation edges ===============================================================


@pytest.mark.parametrize(
    "message",
    [
        "Every day at 8, find my files and delete the old ones",
        "Every morning at 7, research AI news and email me the summary",
        "Every day at 8, research X, then save it to my notes",
    ],
)
def test_a_read_request_that_goes_on_to_ask_for_a_side_effect_is_refused_whole_not_trimmed(chain, message):
    assert chain.say(message) == UNATTENDED_REFUSAL
    assert chain.all_scheduled() == [] and chain.confirmation_jobs() == [] and chain.understood == []


def test_a_written_report_is_still_a_read_only_request(chain):
    assert chain.say("Every morning at 8, research X, compare Y and write me a report").endswith(ASKED)


def test_a_proposal_nobody_answered_expires_and_no_longer_counts_or_blocks_asking_again(tmp_path):
    chain = Chain(tmp_path, limits=Limits(max_open=1, max_concurrent=50, max_runs_per_day=1000))
    chain.propose()
    blocked = chain.say("Research other things every day at 9 AM")
    assert "already have 1 scheduled jobs" in blocked  # the open proposal holds the only place
    chain.clock.advance(hours=25)  # the proposal and its confirmation lapsed
    assert chain.say(EVERY_MORNING).endswith(ASKED)  # not "already waiting", and the one place is free again
    statuses = sorted(s.status.value for s in chain.all_scheduled())
    assert statuses == ["expired", "proposed"]


def test_a_device_context_that_is_not_authenticated_records_no_device(chain):
    class Context:
        is_authenticated = False
        device_id = "dev-forged"

    chain.say(EVERY_MORNING, device_context=Context())
    assert chain.only_scheduled().created_via_device_id is None


class _Outcome:
    def __init__(self, kind, reply="", fallback_reply=None, job_params=None):
        self.kind = type("Kind", (), {"value": kind})()
        self.reply, self.fallback_reply, self.job_params = reply, fallback_reply, job_params


class _FakeHandleThis:
    def __init__(self, outcome):
        self.outcome, self.calls = outcome, []

    def handle(self, text, conversation_id):
        self.calls.append((text, conversation_id))
        return self.outcome


def test_the_handle_this_adapter_returns_the_objective_only_for_a_ready_request_and_keeps_no_conversation():
    objective = objective_for("research X")
    fake = _FakeHandleThis(_Outcome("start_job", job_params={"objective": objective, "request_context": {"x": ["stale"]}}))
    assert understander(lambda: fake)("research X", "conv-1") == objective
    assert fake.calls == [("research X", None)]  # no conversation: nothing waits on a question


@pytest.mark.parametrize(
    "outcome, message",
    [
        (_Outcome("unsupported", reply="no capability"), UNATTENDED_REFUSAL),
        (_Outcome("needs_clarification", reply="what budget?", fallback_reply="Send it again with the budget."), "Send it again with the budget."),
        (_Outcome("failed", reply="Could you say more?"), "Could you say more?"),
        (_Outcome("start_job", job_params={"goal": "x"}), "I couldn't work out what to schedule."),  # a READY request without an objective is not trusted
        (None, "I can't work out what to schedule right now, so nothing was set up."),
    ],
)
def test_the_handle_this_adapter_turns_everything_else_into_a_plain_refusal(outcome, message):
    fake = _FakeHandleThis(outcome)
    with pytest.raises(ScheduledJobRequestError) as raised:
        understander(lambda: fake if outcome is not None else None)("research X", None)
    assert str(raised.value) == message


# ====================================== hardening: re-validation, approval identity, duplicate races =============================


def test_approval_re_validates_against_the_limits_in_force_NOW(tmp_path):
    chain = Chain(tmp_path)
    chain.say("Research AI news every 1 hour")
    sj = chain.only_scheduled()
    chain.service.limits = Limits(min_interval=timedelta(hours=3), max_concurrent=50, max_runs_per_day=1000)  # tightened since it was asked
    with pytest.raises(ScheduledJobRequestError, match="at least 180 minutes"):
        chain.service.approve(sj.id, "int-1")
    assert chain.store.get(sj.id).status is S.PROPOSED


def test_an_approval_by_a_different_confirmation_than_the_one_that_activated_it_is_refused(chain):
    sj = chain.service.propose("research X", _when("every day at 9 AM")).scheduled_job
    assert chain.service.approve(sj.id, "int-1").already is False
    assert chain.service.approve(sj.id, "int-1").already is True  # the same approval again: harmless
    with pytest.raises(ScheduledJobRequestError, match="already dealt with"):
        chain.service.approve(sj.id, "int-2")  # some other confirmation is not this approval


def test_a_confirmation_of_another_origin_naming_this_proposal_cannot_approve_it(chain, monkeypatch):
    import app.scheduled_operations.operations as operations

    sj = chain.service.propose("research X", _when("every day at 9 AM")).scheduled_job
    monkeypatch.setattr(operations, "ORIGIN", "some_other_feature")
    foreign_job_id = chain.confirmer.ask(sj.id, "Shall I?", None)  # same shape, same proposal id, a different origin
    monkeypatch.undo()
    (foreign,) = chain.interactions.list_for_job(foreign_job_id)
    chain.interactions.resolve(foreign.id, expected_version=foreign.version, response="yes")

    assert chain.confirmer.approved(foreign_job_id, sj.id) is None
    assert chain.commit.execute(sj.id, source_job_id=foreign_job_id).status is ToolResultStatus.ERROR
    assert chain.store.get(sj.id).status is S.PROPOSED


def test_two_identical_requests_racing_keep_exactly_one_proposal(chain):
    """Both pass the duplicate check before either inserts; the earliest wins and the other withdraws its own row."""
    real = chain.store
    rival_text = "research AI robotics news and give me a summary"

    class Racing:
        def __getattr__(self, name):
            return getattr(real, name)

        def create_proposal(self, **kwargs):
            if not getattr(self, "raced", False):
                self.raced = True
                real.create_proposal(**{**kwargs, "now": kwargs["now"] - timedelta(seconds=1), "proposal_expires_at": kwargs["proposal_expires_at"]})
            return real.create_proposal(**kwargs)

    chain.service.store = Racing()
    result = chain.service.propose(rival_text, _when("every day at 8 AM"), conversation_id=None)

    assert result.duplicate is True
    statuses = sorted(sj.status.value for sj in real.list())
    assert statuses == ["cancelled", "proposed"]  # ours was withdrawn; the earlier one stands
    assert result.scheduled_job.status is S.PROPOSED and result.scheduled_job.id == [sj for sj in real.list() if sj.status is S.PROPOSED][0].id


def test_repeating_something_you_already_have_at_the_cap_says_so_instead_of_too_many(tmp_path):
    chain = Chain(tmp_path, limits=Limits(max_open=1, max_concurrent=50, max_runs_per_day=1000))
    chain.propose()
    assert "already waiting" in chain.say(EVERY_MORNING)


# ====================================== the trust model: who can create or approve one ============================================


def test_a_plan_that_names_the_commit_tool_cannot_approve_anything(tmp_path):
    """Web content a Job read, or a confused Planner, can put ANY step in a plan. The approval tool is non-plannable, so no plan can
    carry it: the Job fails, the tool never runs, and the proposal stays exactly as it was."""
    chain = Chain(tmp_path)
    sj = chain.propose()
    chain.planner.plans.append([step("s1", COMMIT_TOOL_NAME, {"scheduled_job_id": sj.id})])
    job = chain.jobs.create("do something a web page told me to")
    chain.tick(4)
    assert chain.jobs.get(job.id).status is JobStatus.FAILED
    assert COMMIT_TOOL_NAME not in chain.registry.executed
    assert chain.store.get(sj.id).status is S.PROPOSED and chain.store.get(sj.id).approved_at is None


def test_only_a_chat_message_reaches_the_door_no_tool_or_job_output_can_create_a_scheduled_job(chain):
    """The door is a hook on what the USER types. Nothing that a tool or a Job returns is ever routed to it, and no registered tool can
    propose one (the only tool in this layer is the confirmation-gated commit)."""
    from app.orchestration.profiles import UNATTENDED_V1

    mine = [t.name for t in chain.registry.list_tools() if "schedul" in t.name]
    assert mine == [COMMIT_TOOL_NAME]
    assert not any("schedul" in name for name in UNATTENDED_V1.allowed_tools)  # an unattended run cannot reach it either


def test_chat_stays_the_anonymous_local_interface_and_this_layer_changes_nothing_about_who_may_call_it():
    """/chat has no credential by design: it is the legacy LOCAL interface and relies on the server only listening on the local machine
    (`uvicorn app.main:app` binds the loopback address). Creating or approving a Scheduled Job needs no more - and no less - than every
    other thing /chat does. These pin the properties that boundary relies on, so a change to them is deliberate: no cross-origin access
    (no CORS), and a request a web page could send without a preflight is refused."""
    from fastapi.testclient import TestClient

    from app.main import app

    names = {m.cls.__name__ for m in app.user_middleware}
    assert not {"CORSMiddleware", "TrustedHostMiddleware"} & names  # unchanged: this slice added no new trust mechanism
    client = TestClient(app)
    simple = client.post("/chat", content='{"message": "what scheduled jobs do I have?"}', headers={"Content-Type": "text/plain"})
    assert simple.status_code == 422  # a cross-site form/text POST is not a chat message
    preflight = client.options("/chat", headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "content-type"})
    assert preflight.status_code == 405 and "access-control-allow-origin" not in preflight.headers  # so JSON from another origin is blocked


def test_nothing_binds_the_server_from_the_unused_host_setting():
    """`config.py` has a `host = "0.0.0.0"` default that nothing reads: the server only listens where its launcher says (the documented
    `uvicorn app.main:app` is loopback). If anything ever starts reading it, /chat - which has no credential by design - would be
    network-reachable, so that change must be a deliberate one (and part of the remote-access design), not a side effect."""
    import pathlib
    import re

    app_dir = pathlib.Path(__file__).resolve().parent.parent / "app"
    readers = [
        str(path.relative_to(app_dir))
        for path in app_dir.rglob("*.py")
        if path.name != "config.py" and re.search(r"settings\.(host|port)\b|JARVIS_(HOST|PORT)", path.read_text(encoding="utf-8"))
    ]
    assert readers == []
    assert "uvicorn.run" not in "".join(path.read_text(encoding="utf-8") for path in app_dir.rglob("*.py"))  # the app never starts a server itself


def test_no_route_or_credential_specific_to_scheduled_jobs_exists():
    import pathlib
    import re

    main = (pathlib.Path(__file__).resolve().parent.parent / "app" / "main.py").read_text(encoding="utf-8")
    assert not re.search(r'@app\.(get|post|put|patch|delete)\([^)]*sched', main, re.IGNORECASE)
    assert "def chat(" in main and main.count("@app.post(") == 3  # /chat, /device/chat, /device/check-in: nothing added


def test_a_reply_to_a_waiting_question_is_that_reply_and_never_also_a_scheduling_request(chain):
    """The door runs only when the message is not an answer to something Jarvis is waiting on: otherwise one sentence would both answer a
    question and silently open a proposal (and a second confirmation) the user never meant to start."""
    from app.orchestration.interactions import InteractionType, default_ttl_seconds

    job, question = chain.interactions.create_job_awaiting_input(
        goal="research printers", context={}, conversation_id=chain.conversation, prompt="When do you want this done?",
        ttl_seconds=default_ttl_seconds(InteractionType.INPUT),
    )
    chain.resolver.mark_surfaced(question.id, chain.conversation)

    reply = chain.say("Every day at 8 AM, research printers")

    assert chain.interactions.get(question.id).status is InteractionStatus.RESOLVED  # it WAS the answer
    assert chain.all_scheduled() == [] and chain.confirmation_jobs() == []  # ...and opened nothing
    assert "Here's what I'll set up" not in reply
