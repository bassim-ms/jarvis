"""
Jarvis - entry point.

FastAPI server with a health check endpoint and a /chat endpoint backed by
Jarvis Core, which routes to an AI provider (Gemini for now) through the
AIProvider interface, and persists conversation history and user facts
through a MemoryStore. Since M7, it also starts a background
ReminderScheduler that polls persisted reminders and fires local
notifications for due ones. Since M8, it also starts a background
AutomationEngine that polls persisted automations and runs their
already-permission-checked tool actions when due. Since M9, it also
wires up Google Calendar (OAuth-authorized separately via
`python -m app.calendar.authorize`) and local Jarvis Tasks - neither
needs a background process of its own. Since M15, it also starts a
background JobEngine that plans and executes goal-driven Jobs (see
app/orchestration/) - a TASK-capability chat message just creates a
PENDING Job and returns immediately; JobEngine does the actual planning/
execution on its own thread, the same way AutomationEngine/
ReminderScheduler already do their own background work. Since M17.1-C,
it also wires up local Windows computer-control tools (see
app/computer/) for ordinary interactive TOOL-capability chat ONLY - a
separate ToolRegistry from `core.tools`, deliberately never reachable
from M15 Job planning or AutomationEngine (see this file's own
`core.computer_tools` wiring below). Since Dashboard Slice 1, it also
serves the local web dashboard (app/dashboard/) at /dashboard. No browser
agent or hardware control beyond that yet - those come in later milestones.
"""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel

from app.automation.factory import create_default_automation_store
from app.calendar.factory import create_default_calendar_provider
from app.computer.defaults import create_computer_tools_registry, register_job_computer_tools
from app.computer.factory import create_default_computer_backend
from app.observation.defaults import (
    create_default_observation_engine,
    register_observation_tools,
)
from app.config import settings
from app.context_engine.factory import create_default_context_engine
from app.context_planning.factory import create_default_context_planner
from app.context_reasoning.factory import create_default_context_reasoner
from app.dashboard import DashboardSources, build_dashboard_router
from app.action_proposal import ActionProposalService, ExecutionHandoff, ResultReports
from app.focus import FocusService
from app.host_guard import LoopbackHostMiddleware
from app.logging_setup import configure_logging
from app.handle_this.base import GoalFamily
from app.handle_this.boundary import DEFAULT_BOUNDARY
from app.handle_this.continuation import ObjectiveContinuer, plausible_input_answer
from app.handle_this.dialogue import ClarificationDialogue
from app.memory_policy.confirmation import SensitiveWriteConfirmer
from app.memory_policy.factory import create_default_memory_write_service
from app.memory_policy.tools import register_memory_tools
from app.handle_this.factory import create_default_intake
from app.handle_this.routing import ObjectiveAwareCapabilityRouter
from app.handle_this.service import HandleThisService
from app.core import ConversationNotFoundError, core
from app.identity import load_identity_context
from app.notes.factory import create_default_note_store
from app.orchestration.factory import (
    create_default_interaction_store,
    create_default_job_engine,
    create_default_job_store,
)
from app.orchestration.interaction_lifecycle import InteractionLifecycle
from app.orchestration.notification_events import still_relevant
from app.orchestration.interaction_resolver import InteractionResolver
from app.scheduled_execution.adapter import ScheduledExecutionAdapter
from app.scheduled_execution.context import FreshContextBuilder
from app.scheduled_execution.scheduler import ScheduledJobScheduler
from app.scheduled_jobs.policy import Limits as ScheduledJobLimits
from app.scheduled_jobs.sqlite_store import SQLiteScheduledJobStore
from datetime import datetime, timedelta

from app.devices.base import DeviceAuthenticationError, DeviceContext, TrustState
from app.devices.gateway import DeviceGateway
from app.devices.operations import CommitDeviceRevocationTool, DeviceOperations, DeviceRevocationConfirmer
from app.devices.registry import DeviceRegistry
from app.devices.store import SQLiteDeviceStore
from app.morning_brief.providers import SearchNewsProvider
from app.morning_brief.service import BriefSources, MorningBriefService
from app.morning_brief.synthesis import BriefSynthesizer
from app.morning_brief.tools import MorningBriefTool
from app.notifications.activity_context import ActivityContextService, SQLiteActivityContextStore
from app.notifications.attention import AttentionPolicy
from app.notifications.attention_config import attention_config_from_settings
from app.notifications.attention_gate import AttentionGate, GatedNotificationRouter
from app.notifications.attention_queue import SQLiteAttentionQueue
from app.notifications.base import (
    DEFAULT_USER,
    NotificationPriority,
    NotificationRouter,
    SQLiteNotificationLedger,
)
from app.notifications.channels import DesktopToastChannel
from app.providers.base import ProviderError
from app.self_development.factory import create_default_self_development_engine
from app.reminders.factory import (
    create_default_notifier,
    create_default_reminder_store,
    create_default_scheduler,
)
from app.reminders.operations import ReminderOperations
from app.orchestration.interactions import InteractionType, default_ttl_seconds
from app.scheduled_operations.operations import ScheduledJobApprovalConfirmer, ScheduledJobOperations
from app.scheduled_operations.service import ScheduledJobService, understander as scheduled_job_understander
from app.scheduled_operations.presenter import ScheduledJobReporter, facts_from_job as scheduled_job_facts_from_job
from app.scheduled_operations.tools import CommitScheduledJobApprovalTool
from app.search.factory import create_default_search_provider
from app.tasks.factory import create_default_task_store
from app.tools.defaults import register_default_tools
from app.world_model.factory import (
    create_default_world_model_service,
    create_default_world_model_store,
)
from app.world_model.sqlite_store import DEFAULT_WORLD_MODEL_DB_PATH

# Application-level wiring: populate the running app's tool registry with
# Jarvis's default tools. JarvisCore's own default (when no registry is
# injected) stays empty - see app/tools/defaults.py.
#
# The reminder/automation/task stores (and the calendar provider) are
# built once here and passed into both register_default_tools (so their
# tools operate on them) and the ReminderScheduler/AutomationEngine below
# (so their background polling sees the exact same persisted rows),
# rather than each independently opening its own SQLite connection to the
# same file. `core.memory` is passed through too, so an automation's
# conditions (app/automation/base.py's FactEqualsCondition) see the same
# stored facts JarvisCore's own DIRECT/SEARCH prompts already use.
_reminder_store = create_default_reminder_store(settings.reminders_db_path)
_automation_store = create_default_automation_store(settings.automations_db_path)
_calendar_provider = create_default_calendar_provider()
_task_store = create_default_task_store(settings.tasks_db_path)
_job_store = create_default_job_store(settings.orchestration_db_path)
# Interactions (questions/confirmations a Job waits on the user for) live in
# the same database, sharing the Job store's connection so the engine can
# change a Job and its interaction in one transaction.
_interaction_store = create_default_interaction_store(_job_store)
_note_store = create_default_note_store(settings.notes_db_path)


def _build_notification_router() -> NotificationRouter:
    """The channel-neutral router Jobs and interactions notify through. Today
    its one channel is the desktop toast (the Notifier reminders use); future
    channels (speech, phone, watch, glasses) are added here. Its delivery
    ledger lives in the Job database, so de-duplication survives restarts."""
    router = NotificationRouter(ledger=SQLiteNotificationLedger.for_job_store(_job_store))
    if settings.notifications_desktop_enabled:
        try:
            floor = NotificationPriority[settings.notifications_desktop_min_priority.strip().upper()]
        except KeyError:
            floor = NotificationPriority.LOW
        router.add_channel(DesktopToastChannel(create_default_notifier(), min_priority=floor))
    return router


_notification_router = _build_notification_router()

# The attention policy in front of the notification router (app/notifications/attention*.py):
# decides whether an event interrupts now, waits in a queue, or stays silent, and on which
# of the router's channels. The producers (Job engine, interaction lifecycle, reminder
# scheduler) are handed `_gated_router` - the same three methods as the router - so they
# still just emit structured events and learn nothing about attention.
#
# The user's current activity is an explicit, TEMPORARY context ("I'm in class", "I'm free
# now"; app/notifications/activity_context.py), never memory and never inferred. When it
# changes - and on every engine tick (below) - held-back notifications are reconsidered.
_activity_context = ActivityContextService(
    SQLiteActivityContextStore.for_job_store(_job_store),
    clock=lambda: datetime.now().astimezone(),
)
attention_gate = AttentionGate(
    _notification_router,
    AttentionPolicy(attention_config_from_settings()),
    SQLiteAttentionQueue.for_job_store(_job_store),
    context_provider=_activity_context.current_name,
    relevance_check=lambda event: still_relevant(event, _interaction_store),
)
_gated_router = GatedNotificationRouter(attention_gate)
_activity_context.on_change = lambda: len(attention_gate.release_queued(DEFAULT_USER))
automation_engine = register_default_tools(
    core.tools,
    reminder_store=_reminder_store,
    automation_store=_automation_store,
    memory=core.memory,
    calendar_provider=_calendar_provider,
    task_store=_task_store,
    job_store=_job_store,
    note_store=_note_store,
    analysis_provider=core.provider,
    interaction_store=_interaction_store,
)

reminder_scheduler = create_default_scheduler(
    _reminder_store,
    settings.reminder_poll_interval_seconds,
    router=_gated_router,
    grace=timedelta(seconds=max(0, settings.schedule_grace_seconds)),
)

# JobEngine (M15) isn't built by register_default_tools itself (see that
# function's own docstring) - built here the same way ReminderScheduler
# is, against the exact same `_job_store`/`core.tools` the create_job/
# confirm_job/... tools above already operate on, so the background
# planner/executor and a user's tool calls always see the same
# persisted Jobs.
# `context_provider=core.memory.get_facts` refreshes a Job's context
# immediately before every planning/replanning call (see
# JobEngine._refresh_context) - the same fact source CreateJobTool's own
# `context_provider` already snapshots at creation time, so a Job never
# relies solely on that one-time snapshot going stale.
job_engine = create_default_job_engine(
    _job_store,
    core.tools,
    max_steps=settings.orchestration_max_steps,
    max_total_steps=settings.orchestration_max_total_steps,
    max_retries_per_step=settings.orchestration_max_retries_per_step,
    max_replans=settings.orchestration_max_replans,
    retry_delay_seconds=settings.orchestration_retry_delay_seconds,
    poll_interval_seconds=settings.orchestration_poll_interval_seconds,
    context_provider=core.memory.get_facts,
    # `core.provider` (the AIProviderManager) drives the planner - planning
    # and replanning - and the result evaluator (which judges a step's
    # result ONLY when its plan entry carries an `expectation`, see
    # app/orchestration/evaluator.py), so both get the same ordered
    # fallback every other AI call in this app already gets, instead of a
    # single unmanaged Gemini client. `analyze` uses it too (see
    # register_default_tools above).
    provider=core.provider,
    interaction_store=_interaction_store,
    # A Job's outcome is announced through the notification router: a
    # failure (only once it can't be recovered) at HIGH priority, a
    # background Job's completion at LOW. Best-effort and off the critical
    # path: the persisted Job is authoritative whether or not anything shows.
    notification_router=_gated_router,
    notify_on_completion=settings.orchestration_notify_on_completion,
    # Continues a Handle This objective that waited on a question once it is
    # answered: folds the answer into the ORIGINAL objective and re-checks it,
    # deterministically, before the existing planner runs.
    objective_continuer=ObjectiveContinuer(),
)

# Tells the user about waiting questions/confirmations (immediately, then a
# few bounded reminders, one final warning) and closes them when they expire.
# Runs at the end of every engine tick; it also opens the confirmation for a
# Job the engine parked in AWAITING_CONFIRMATION.
interaction_lifecycle = InteractionLifecycle(
    _interaction_store, _job_store, job_engine, _gated_router
).attach()

# Held-back notifications are reconsidered on the engine's own tick - no new background loop. It
# re-decides only what is waiting, under the context and time of that moment: a context that has
# ended, quiet hours that are over, or a channel that has become available lets an item through;
# an item that is still suppressed stays put; one that expired or is no longer relevant is expired.
job_engine.add_tick_hook(lambda: attention_gate.release_queued(DEFAULT_USER))

# Routes a user's chat reply to a waiting interaction (`answer <code>: ...`,
# or a yes/no / free-text answer to something surfaced to that conversation)
# and hands it to the engine's atomic transition. It never runs a tool.
# `answer_gate`: a pending clarification is only answered by a message that
# plausibly answers it, so ordinary conversation never uses up a question.
# `dialogue`: the conversation around a question - cancelling, changing an
# answer, coming back to a stale question, the next question - all
# deterministic, all through the engine's own version-guarded transitions.
core.activity_context = _activity_context
core.escalation_surface = attention_gate
core.interaction_resolver = InteractionResolver(
    _interaction_store,
    _job_store,
    job_engine,
    answer_gate=plausible_input_answer,
    dialogue=ClarificationDialogue(),
)

# Personal World Model + Context Engine (M16.1/M16.2) - wired onto the
# already-constructed `core` singleton from outside, the exact same
# "build it here, attach it to core" pattern every store/engine above
# already uses (see e.g. `core.tools` populated by register_default_tools).
# `core.context_engine` defaults to `None` (see app/core.py's own
# JarvisCore.__init__), so this is what actually ACTIVATES the M16.2
# integration for the real running app - without this, `_dispatch` would
# never build a RequestContext/ContextSnapshot at all. No `settings.*`
# field exists for the World Model's db path (deliberately - M16.1's own
# correction: "the World Model should continue receiving db_path
# explicitly through its store/factory APIs," never a config mechanism),
# so `DEFAULT_WORLD_MODEL_DB_PATH` (app/world_model/sqlite_store.py) - the
# one place that literal string is actually defined - is imported and
# passed here explicitly, rather than a second hardcoded copy of it
# living in this file too. `memory=core.memory` mirrors the same "share
# the one real MemoryStore, never open a second connection to the same
# file" convention every store above already follows.
# With JARVIS_DATA_DIR set, the same file name is placed in that folder (settings.in_data_dir) - still passed explicitly.
_WORLD_MODEL_DB_PATH = settings.in_data_dir(DEFAULT_WORLD_MODEL_DB_PATH)
_world_model_store = create_default_world_model_store(_WORLD_MODEL_DB_PATH)
_world_model_service = create_default_world_model_service(_world_model_store)
core.context_engine = create_default_context_engine(_world_model_service, memory=core.memory)

# The write policy for persistent memory (see app/memory_policy/): what the user
# plainly says or asks to be remembered is persisted through the EXISTING World
# Model service and MemoryStore, under a deterministic policy (secrets never,
# sensitive/contradictory only with the user's say-so, inference and uncertainty
# only as staged candidates). A model can only PROPOSE, through tools registered
# on the ToolRegistry - non-plannable, so no Job plan (or web content a Job read)
# can name them.
core.memory_writer = create_default_memory_write_service(
    _world_model_service,
    core.memory,
    _WORLD_MODEL_DB_PATH,
    # A sensitive request is confirmed through the EXISTING Job / interaction /
    # ToolRegistry machinery (never a mechanism of its own): its confirmation is
    # put to the user in the conversation it came from.
    confirmer=SensitiveWriteConfirmer(
        _interaction_store, mark_surfaced=core.interaction_resolver.mark_surfaced
    ),
)
register_memory_tools(core.tools, core.memory_writer)

# Devices (see app/devices/): the registry lives in the application's existing database, the Gateway
# authenticates a device's credential and hands back a DeviceContext (identity only - it grants nothing),
# and the few user-facing device operations ride the ordinary chat path. Revoking a device from chat is
# ASKED through the existing confirmation machinery (a CONFIRM-level, non-plannable tool, like the memory
# policy's); registering and trusting a device happen only in the admin command (python -m app.devices.admin).
_device_registry = DeviceRegistry(SQLiteDeviceStore.for_job_store(_job_store))
device_gateway = DeviceGateway(_device_registry)
device_operations = DeviceOperations(
    _device_registry,
    DeviceRevocationConfirmer(_interaction_store, mark_surfaced=core.interaction_resolver.mark_surfaced),
)
core.device_operations = device_operations

# Reminders (see app/reminders/): "remind me tomorrow at 7 PM to study", "what reminders do I have?", "pause my study
# reminder" ride the ordinary chat path as a whole-message hook. It reads a fixed grammar (no model, no Planner, no Job)
# and acts only through the reminder tools already on `core.tools`, so their permission checks apply. Multi-step goals
# that merely contain a reminder still go to Handle This.
core.reminder_operations = ReminderOperations(
    core.tools,
    family_on_offer=lambda names: DEFAULT_BOUNDARY.assess((GoalFamily.ORGANIZATION,), names).supported,
)
core.tools.register(CommitDeviceRevocationTool(device_operations))
# The Context Engine may show reasoning a confident staged candidate - labelled
# UNCONFIRMED, in its own field, never as a fact.
core.context_engine.candidate_source = core.memory_writer.candidates_for_context

# Scheduled Jobs (see app/scheduled_jobs/ and app/scheduled_execution/): the scheduler is hung on the JobEngine's OWN tick - no
# second loop, thread or timer - so it exists exactly while Jarvis runs. It decides which due occurrences become a Job and settles the
# finished ones; the Job itself is an ordinary one that this same JobEngine plans and runs through `core.tools` (the Job-facing
# registry) under the READ-only unattended profile. The kill switch (`scheduled_jobs_enabled`) is read on every tick; a device a
# Scheduled Job was created from must still be TRUSTED in the existing device registry. A Scheduled Job only exists once the user
# APPROVED it (the creation flow just below), so until then the scheduler finds nothing due.
_scheduled_job_limits = ScheduledJobLimits.from_settings()
_scheduled_job_store = SQLiteScheduledJobStore.for_job_store(_job_store)
_scheduled_job_adapter = ScheduledExecutionAdapter(
    scheduled_jobs=_scheduled_job_store,
    tools=core.tools,
    context_builder=FreshContextBuilder(
        context_engine=core.context_engine,
        snapshot_serializer=core._snapshot_to_job_context,
        facts_provider=core.memory.get_facts,
    ),
    run_deadline=_scheduled_job_limits.run_deadline,
)
scheduled_job_scheduler = ScheduledJobScheduler(
    scheduled_jobs=_scheduled_job_store,
    jobs=_job_store,
    adapter=_scheduled_job_adapter,
    limits=_scheduled_job_limits,
    enabled=lambda: settings.scheduled_jobs_enabled,
    device_trusted=lambda device_id: (
        (device := _device_registry.get(device_id)) is not None and device.trust_state is TrustState.TRUSTED
    ),
)
job_engine.add_tick_hook(scheduled_job_scheduler.tick)

# Creating and managing Scheduled Jobs (see app/scheduled_operations/): "every morning at 8, research X" rides the ordinary chat path as a
# whole-message hook. The request is understood by the EXISTING Handle This (this is the only place that may import it; the service gets
# it as a callable), checked by the SAME adapter check every run applies (so a request an unattended read-only run could never carry out
# is refused up front, never weakened), stored as a PROPOSAL and put to the user through the EXISTING confirmation machinery - only an
# approved confirmation runs the CONFIRM-level, non-plannable `commit_scheduled_job_approval`, which activates it atomically. Approval
# adds no capability: every run still goes through the READ-only profile, the ToolRegistry, Planner validation, device trust and audit.
_scheduled_job_service = ScheduledJobService(
    _scheduled_job_store,
    limits=_scheduled_job_limits,
    understand=scheduled_job_understander(lambda: core.handle_this),
    check_objective=_scheduled_job_adapter.check_objective,
    proposal_ttl=timedelta(seconds=default_ttl_seconds(InteractionType.CONFIRMATION)),
    run_now=scheduled_job_scheduler.run_now,  # "run it now" is the SCHEDULER's: the same gates and the same launch as a due occurrence
    in_flight=scheduled_job_scheduler.in_flight,  # a renewal waits for a run in flight: the scheduler's own definition of it
)

def _scheduled_job_facts(job_id: str):
    return scheduled_job_facts_from_job(_job_store.get(job_id))  # read-only: the Job's own recorded status, times and report


scheduled_job_operations = ScheduledJobOperations(
    _scheduled_job_service,
    ScheduledJobApprovalConfirmer(_interaction_store, mark_surfaced=core.interaction_resolver.mark_surfaced),
    job_facts=_scheduled_job_facts,
)
# Telling the user what became of each occurrence (finished, failed, skipped and why), once, in Jarvis's words - through the EXISTING
# attention-gated notification router, hung on the engine's tick like the scheduler. It changes nothing but the ledger's `reported_at`.
scheduled_job_reporter = ScheduledJobReporter(
    scheduled_jobs=_scheduled_job_store,
    notify=_gated_router.notify,
    job_facts=_scheduled_job_facts,
    limits=_scheduled_job_limits,
)
job_engine.add_tick_hook(scheduled_job_reporter.report)
core.scheduled_job_operations = scheduled_job_operations
core.tools.register(CommitScheduledJobApprovalTool(scheduled_job_operations))

# The Morning Brief (app/morning_brief/): READS the existing calendar, task and reminder stores, the
# Context Engine (World Model, memory, unconfirmed candidates), the Attention queue, Jobs and pending
# interactions, and the user's stated activity - and words the result deterministically. It writes,
# sends, approves and executes nothing. No weather or business integration exists yet, so none is wired;
# news (through the existing search provider) is off unless configured. The brief is worded
# deterministically and then, by default, reworded by the existing AI provider (no tools, output
# validated, deterministic fallback on any failure).
_brief_sources = BriefSources(
    calendar=_calendar_provider,
    tasks=_task_store,
    reminders=_reminder_store,
    context_engine=core.context_engine,
    attention_queue=attention_gate.queue,
    jobs=_job_store,
    interactions=_interaction_store,
    activity=_activity_context,
    news=(
        SearchNewsProvider(create_default_search_provider(), settings.morning_brief_news_query)
        if settings.morning_brief_news_enabled
        else None
    ),
    relevance_check=lambda event: still_relevant(event, _interaction_store),
)
morning_brief_service = MorningBriefService(
    _brief_sources,
    clock=lambda: datetime.now().astimezone(),
    synthesizer=BriefSynthesizer(core.provider) if settings.morning_brief_use_llm else None,
)
core.morning_brief = morning_brief_service
core.tools.register(MorningBriefTool(morning_brief_service))

# Focus (app/focus/): "I need to finish my Arduino project today" / "what should I work on?" - READS the existing task store,
# calendar, World Model projects and goals, and the user's stated activity, ranks deterministically and words the result from
# fixed templates. It registers no tool, starts no Job, writes nothing and calls no model; it is the last chat door.
core.focus = FocusService(
    tasks=_task_store,
    calendar=_calendar_provider,
    world_model=_world_model_service,
    activity=_activity_context,
    clock=lambda: datetime.now().astimezone(),
)

# Action Proposals (app/action_proposal/): "make a plan" - a few honest steps built from what Focus decided, shown, and held IN MEMORY for a
# few minutes, bound to one conversation. Until a yes it creates no Job, runs no tool, calls no model and writes nothing; a valid yes pops the
# proposal, revalidates it against live data and hands the exact approved steps to `on_approved` (Slice 3), which starts ONE Job.
core.action_proposals = ActionProposalService(
    focus=core.focus,
    tasks=_task_store,
    tools=core.tools,
    clock=lambda: datetime.now().astimezone(),
    # An approved, revalidated proposal becomes ONE Job, born RUNNING with exactly the approved steps (never PENDING: the engine would plan it
    # with the model). The existing JobEngine runs it through the existing ToolRegistry and permission rules.
    on_approved=ExecutionHandoff(_job_store.create_running_job, on_started=lambda job: core.result_reports.watch(job)),
)
# Slice 4: how that Job turned out, verified from its own stored state and told ONCE, in the next reply in the same conversation (pulled, in
# memory, nothing pushed). Read-only: it only ever calls the Job store's `get`.
core.result_reports = ResultReports(_job_store.get)
job_engine.add_tick_hook(core.result_reports.poll)

# Observation + the narrow set of computer tools a Job may use (Slices 4-5, see app/observation/): the observe ->
# act -> observe -> verify loop runs on the EXISTING Planner / JobEngine / ToolRegistry, so the JOB-facing registry
# (`core.tools`) gets exactly this allowlist from the computer package and nothing else: open_url (public hosts
# only), scroll, get_active_window and switch_window (app/computer/defaults.py: JOB_COMPUTER_TOOL_NAMES), plus the
# READ tools observe_page, verify_observation and observe_screen (bounded, in-memory). Input tools (type_text,
# press_key, click_mouse, double_click, move_mouse, open_application) stay in the isolated interactive registry
# below. Registered BEFORE Handle This is built so the COMPUTER_WORKFLOW family sees them. One shared backend;
# observation only ever calls its READ methods.
_computer_backend = create_default_computer_backend()
observation_engine = create_default_observation_engine(_computer_backend)
register_job_computer_tools(core.tools, _computer_backend, anchors=observation_engine.anchors.observed)
register_observation_tools(core.tools, observation_engine)

# Context-Aware Reasoning (M16.3) - wired onto `core` the same way M16.2
# just was, immediately above: `core.context_reasoner` defaults to `None`
# (see JarvisCore.__init__), so this is what actually ACTIVATES M16.3 for
# the real running app. Routed through `core.provider` - the same
# AIProviderManager/provider-routing chain every other AI call in this
# app already goes through (app/providers/) - never a second, separately
# configured AI client. See app/context_reasoning/reasoner.py for why a
# request whose ContextSnapshot has nothing relevant never reaches the
# provider at all, keeping this activation backward-compatible with
# every ordinary request the way M16.2's own activation above already is.
core.context_reasoner = create_default_context_reasoner(core.provider)

# Context-Aware Planning (M16.4) - wired onto `core` the same way M16.3
# just was, immediately above: `core.context_planner` defaults to `None`
# (see JarvisCore.__init__), so this is what actually ACTIVATES M16.4 for
# the real running app. Also routed through `core.provider` - never a
# second, separately configured AI client. See app/core.py's own
# `_plan_about_context` for why this only ever runs for planning/action-
# oriented requests (never every chat message), and
# app/context_planning/planner.py for why a request with nothing to plan
# around never reaches the provider at all - both keep this activation
# backward-compatible with every ordinary/informational request the way
# M16.2/M16.3's own activation above already is. M16.4 never creates a
# Job itself: JarvisCore._handle_task's existing `create_job` dispatch
# (untouched by this milestone) remains the only path that does.
core.context_planner = create_default_context_planner(core.provider)

# Handle This (see app/handle_this/): understands a goal-driven request BEFORE
# a Job is started for it - the objective, its family of work, the relevant
# context, whether something material is missing, whether it is supported.
# Only a READY, supported request goes on, and it still becomes an ordinary
# Job through the ordinary `create_job` tool, planned, validated and executed
# by the existing JobEngine/Planner/validator/ToolRegistry. `tools=core.tools`
# is the JOB-facing registry (NOT the isolated computer-control one), so a
# request needing computer control is honestly reported unsupported. Wired
# after the Context Engine above, which it reads (selectively, read-only).
core.handle_this = HandleThisService(
    create_default_intake(
        provider=core.provider, context_engine=core.context_engine, tools=core.tools
    )
)

# Routing INTO Handle This: a wrapper around the router `core` already has. It
# leaves every explicit intent (tool command, confirmation, self-development,
# task status) exactly as decided, and only sends a message to Handle This when
# it is clearly a delegated multi-step objective (structure, not keywords; no
# AI call). A plain question, a simple command or an explicit search keeps its
# existing path. Installed together with `core.handle_this` above.
core.capability_router = ObjectiveAwareCapabilityRouter(core.capability_router)

# Identity context (personality.md/user_profile.md, repo root - see
# app/identity.py) - `core.identity_context` defaults to `None` (see
# JarvisCore.__init__), so this is what actually ACTIVATES it for the
# real running app. Loaded ONCE here (a plain, static string - these are
# small local files, not a per-request lookup), the same "build it here,
# attach it to core" pattern every store/engine above already uses.
# `load_identity_context` degrades to "" for a missing/unreadable file -
# never an error - so a not-yet-created or briefly malformed file never
# breaks server startup or an ordinary chat turn.
core.identity_context = load_identity_context(
    settings.personality_file_path, settings.user_profile_file_path
)

# Self-Development Engine V1 (see app/self_development/) -
# `core.self_development_engine` defaults to `None` (see JarvisCore.
# __init__), so this is what actually ACTIVATES it for the real running
# app: a chat message the deterministic router identifies as
# SELF_DEVELOPMENT (Capability.SELF_DEVELOPMENT, app/capability.py) now
# reaches it instead of falling back to ordinary DIRECT handling. Routed
# through `core.provider` - never a second, separately configured AI
# client. `self_dev_project_root` bounds every file/command this engine
# will ever touch (app/self_development/safety.py) - configurable via
# settings, but re-validated at runtime regardless of what's configured.
#
# Only when JARVIS_SELF_DEV_ENABLED is on (default OFF): its approvals arrive over the anonymous /chat, so any local program could answer
# them. Off, `core.self_development_engine` stays `None` and a development request is answered as ordinary chat.
if settings.self_dev_enabled:
    core.self_development_engine = create_default_self_development_engine(
        core.provider,
        project_root=settings.self_dev_project_root,
        max_repair_iterations=settings.self_dev_max_repair_iterations,
        command_timeout_seconds=settings.self_dev_command_timeout_seconds,
        test_command=settings.self_dev_test_command,
    )

# Computer Control (M17.1-A/B/C, see app/computer/) - `core.computer_tools`
# defaults to `None` (see JarvisCore.__init__), so this is what actually
# ACTIVATES computer-control tools for ordinary interactive chat: a TOOL-
# capability request explicitly naming one (e.g. "use tool open_url ...")
# now reaches it via JarvisCore._handle_tool's `_resolve_tool_registry`.
#
# Deliberately its OWN ToolRegistry, built here and NEVER passed to
# `register_default_tools`/`core.tools` above, nor to `job_engine`/
# `automation_engine` below - those are constructed with `core.tools`
# only, exactly as before this milestone. M15's JobEngine offers
# `core.tools.list_tools()` to its planner and rejects any plan step
# naming a tool `core.tools` doesn't contain (app/orchestration/
# validator.py); AutomationEngine likewise only ever looks tools up on
# the registry it was built with. Since neither is ever given a
# reference to `core.computer_tools`, computer-control tools are
# unreachable from M15 Job planning/execution and from automations - not
# because of a permission check, but because that object simply doesn't
# exist in their scope.
core.computer_tools = create_computer_tools_registry(_computer_backend)


_startup_log = logging.getLogger("jarvis.startup")


@asynccontextmanager
async def _lifespan(app: FastAPI):
    # Logging first, so everything below is recorded (app/logging_setup.py). Configured here, at server start, never on import.
    configure_logging(settings.log_file, settings.log_level)
    _startup_log.info(
        "Jarvis starting: data_dir=%s background_schedulers=%s self_development=%s computer_input=%s",
        settings.data_dir or "(current directory)",
        "on" if settings.background_schedulers_enabled else "OFF",
        "on" if core.self_development_engine is not None else "off",
        "ON" if settings.computer_input_enabled else "off",
    )
    if settings.data_dir:
        # An absolute path (or one climbing out with "..") is honoured as configured - say plainly that it is NOT in the data folder.
        outside = settings.data_files_outside_data_dir({"world_model_db_path": _WORLD_MODEL_DB_PATH})
        if outside:
            _startup_log.warning("data_dir is set, but these data files are OUTSIDE it: %s", ", ".join(outside))
    # JARVIS_BACKGROUND_SCHEDULERS_ENABLED=false starts nothing that runs on its own: due reminders, automations, Jobs and scheduled
    # runs wait, untouched, for a later start with it on. Chat still works; a Job created from chat is stored but not run.
    schedulers = settings.background_schedulers_enabled
    if schedulers:
        reminder_scheduler.start()
        automation_engine.start()
        job_engine.start()
    yield
    if schedulers:
        job_engine.stop()
        automation_engine.stop()
        reminder_scheduler.stop()


app = FastAPI(title=settings.app_name, lifespan=_lifespan)

# Loopback-only Host check (app/host_guard.py): every route answers only a request addressed to localhost / 127.x.x.x / [::1], which closes the DNS-rebinding
# route to the anonymous /chat. It is an ADDITIONAL safeguard only: the server must still be bound to the loopback interface
# (`uvicorn app.main:app --host 127.0.0.1`; the unused `host` default in config.py is not what binds it) and anything that needs trust still needs
# authentication (/device/chat).
app.add_middleware(LoopbackHostMiddleware)

@app.get("/dev/ping")
def dev_ping() -> str:
    return "pong"


@app.get("/dev/status")
def dev_status() -> dict:
    """Read-only status of the REAL, already-wired Self-Development
    Engine (`core.self_development_engine`, set up above at startup) -
    never a fresh engine instantiated per request, and never invented
    data. Reports exactly the sessions that engine is already tracking
    in its own `sessions` dict (the same attribute app/core.py's own
    `_active_self_dev_sessions` already reads directly) - empty if
    none exist yet."""
    engine = core.self_development_engine
    if engine is None:
        return {"configured": False, "sessions": []}
    return {
        "configured": True,
        "sessions": [
            {"id": session.id, "stage": session.stage.value, "dev_branch": session.dev_branch}
            for session in engine.sessions.values()
        ],
    }


@app.get("/providers/status")
def providers_status() -> dict:
    """Read-only state of the AI provider chain, in the exact order every AI call tries it: whether each provider has a key
    and what its most recent REAL answer said about that key (`key_status`, see app/providers/manager.py KEY_STATUS_TEXT:
    "no_key", "present_not_verified", "authenticated", "rejected" or "unavailable" - a non-empty key alone is only ever
    "present_not_verified"), which are enabled and which are resting after a rate limit (and for how long).
    Building this sends nothing to any provider. Names, models, booleans and counts only - never a key, a token, an
    account id or a URL."""
    from app.providers.factory import describe_providers
    from app.providers.manager import KEY_STATUS_TEXT, AIProviderManager, key_status

    manager = core.provider
    if not isinstance(manager, AIProviderManager):
        return {"order": [], "providers": [], "note": "no provider manager is wired"}
    infos = {info.name: info for info in describe_providers()}
    models = {
        "groq": settings.groq_model, "openrouter": settings.openrouter_model, "gemini": settings.gemini_model,
        "cloudflare": settings.cloudflare_model, "freellmapi": "auto",
    }
    now = time.monotonic()
    providers = []
    for position, health in enumerate(manager.get_health(), start=1):
        info = infos.get(health.name)
        resting = max(0.0, health.cooldown_until - now)
        key_present = bool(info and info.configured)
        status = key_status(key_present, health.key_check)
        providers.append({
            "position": position,
            "name": health.name,
            "display_name": info.display_name if info else health.name,
            "model": models.get(health.name, ""),
            "key_present": key_present,
            "key_status": status,
            "key_status_text": KEY_STATUS_TEXT[status],
            "enabled": health.enabled,
            "consecutive_failures": health.consecutive_failures,
            "resting_seconds": round(resting),
        })
    return {
        "order": [entry["name"] for entry in providers],
        "providers": providers,
        "retry_count": manager.retry_count,
        "timeout_seconds": settings.ai_provider_timeout_seconds,
        "default_cooldown_seconds": manager.cooldown_seconds,
        "max_cooldown_seconds": manager.max_cooldown_seconds,
    }


@app.get("/health")
def health_check() -> dict:
    """Simple proof that the server is alive and configured correctly."""
    return {
        "status": "ok",
        "app": settings.app_name,
        "environment": settings.environment,
    }


# The local web dashboard (app/dashboard/): Jarvis's main interface in a browser at http://127.0.0.1:8000/dashboard. It reads the SAME
# stores built above and changes data only through the SAME ToolRegistry chat uses (`core.tools`, so each tool's permission level
# applies); chat goes through the existing POST /chat. Loopback only, like every route (LoopbackHostMiddleware).
app.include_router(build_dashboard_router(DashboardSources(
    tasks=_task_store,
    reminders=_reminder_store,
    notes=_note_store,
    jobs=_job_store,
    interactions=_interaction_store,
    tools=core.tools,
    memory=core.memory,
    provider_status=providers_status,
    health=health_check,
)))


class ChatRequest(BaseModel):
    message: str
    conversation_id: str | None = None


class ChatSource(BaseModel):
    """One citation backing a SEARCH-path reply - mirrors
    app.search.answer.SearchSource verbatim (title/url/query), never
    modified in transit. Always [] for a DIRECT reply."""

    title: str
    url: str
    query: str


class ChatResponse(BaseModel):
    reply: str
    conversation_id: str
    sources: list[ChatSource] = []
    #: The id of the Job `handle_chat_with_job_id` just created (M15), or
    #: `None` for every capability except TASK - a real structured field
    #: instead of a client having to parse the reply's prose for it (see
    #: JarvisCore.handle_chat_with_job_id's own docstring).
    job_id: str | None = None


# ---------------------------------------------------------------------------------------------------
# TWO ENTRY POINTS, NOT EQUIVALENT (see app/devices/__init__.py):
#
#   /chat          the existing LEGACY / LOCAL interface. Anonymous: it never carries a DeviceContext
#                  and is never proof of a device's identity. A request from localhost is NOT "the
#                  trusted PC"; nothing in its body, text or headers can make it one. Unchanged in
#                  Slice 1 because existing clients (including the PC voice client) use it.
#   /device/chat   the AUTHENTICATED device interface: the ONLY route that hands Core a DeviceContext,
#                  and only after the Device Gateway verified a device_id + credential for a TRUSTED
#                  device. (The voice client does not send a credential yet, so it still uses /chat -
#                  an intentional Slice 1 limitation.)
# Do not route /chat through the gateway, and do not attach a device to it.
# ---------------------------------------------------------------------------------------------------
@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Send a user message to Jarvis Core and return its reply.

    Omit conversation_id to start a new conversation; pass back the
    conversation_id from a previous response to continue it with history.

    This is the existing LEGACY / LOCAL interface: anonymous, and never a
    trusted device - it is always called with no device (see /device/chat for
    the authenticated interface).
    """
    return _chat(request, None)


def _authenticated_device(
    x_jarvis_device_id: str | None = Header(default=None),
    x_jarvis_device_token: str | None = Header(default=None),
) -> DeviceContext:
    """Who is calling, from the credential they PROVE - never from the request body or message. Any failure
    (missing, unknown, pending, revoked, wrong credential) is the same 401."""
    try:
        return device_gateway.authenticate(x_jarvis_device_id, x_jarvis_device_token)
    except DeviceAuthenticationError:
        raise HTTPException(status_code=401, detail="device authentication failed")


@app.post("/device/chat", response_model=ChatResponse)
def device_chat(request: ChatRequest, device: DeviceContext = Depends(_authenticated_device)) -> ChatResponse:
    """The same chat, entered through the Device Gateway: the request carries the DeviceContext of the
    authenticated device. Everything after this is the ordinary Jarvis flow and its ordinary limits."""
    return _chat(request, device)


@app.post("/device/check-in")
def device_check_in(
    x_jarvis_device_id: str | None = Header(default=None),
    x_jarvis_device_token: str | None = Header(default=None),
) -> dict:
    """An authenticated "I am alive": marks the device ONLINE and records when. No background service."""
    try:
        device = device_gateway.check_in(x_jarvis_device_id, x_jarvis_device_token)
    except DeviceAuthenticationError:
        raise HTTPException(status_code=401, detail="device authentication failed")
    return {"device_id": device.device_id, "status": device.status.value, "last_seen": device.last_seen.isoformat()}


_chat_log = logging.getLogger("jarvis.chat")


def _chat(request: ChatRequest, device: DeviceContext | None) -> ChatResponse:
    # Only the Gateway makes an authenticated DeviceContext; refuse anything else rather than pass it on.
    if device is not None and not device.is_authenticated:
        raise HTTPException(status_code=401, detail="device authentication failed")
    try:
        # The unauthenticated /chat calls Core exactly as it always has; only an authenticated device
        # adds its DeviceContext.
        extra = {"device_context": device} if device is not None else {}
        reply, conversation_id, sources, job_id = core.handle_chat_with_job_id(
            request.message, request.conversation_id, **extra
        )
    except ConversationNotFoundError as exc:
        _chat_log.info("chat refused: unknown conversation (404)")
        raise HTTPException(status_code=404, detail=str(exc))
    except ProviderError as exc:
        # The exception's type only: its text can describe the request. The detail still goes back to the caller, as before.
        _chat_log.warning("chat failed: %s (502)", type(exc).__name__)
        raise HTTPException(status_code=502, detail=str(exc))
    return ChatResponse(
        reply=reply,
        conversation_id=conversation_id,
        sources=[
            ChatSource(title=s.title, url=s.url, query=s.query) for s in sources
        ],
        job_id=job_id,
    )
