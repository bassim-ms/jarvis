"""
Integration tests for M15: default-registry wiring (register_default_tools
registers all seven orchestration tools using the given job_store),
end-to-end Core TASK dispatch through the real create_job tool, and a
full goal-to-completion run driven by JobEngine.tick() against the same
registry/store Core itself used to create the Job - proving the whole
USER GOAL -> Job -> Planner -> Validator -> Executor -> Tool -> Result
flow works together, not just each piece in isolation.
"""

from app.automation.sqlite_store import SQLiteAutomationStore
from app.capability import Capability, CapabilityDecision, CapabilityRouter
from app.core import JarvisCore
from app.memory.sqlite_store import SQLiteMemoryStore
from app.orchestration.base import JobStatus, StepStatus
from app.orchestration.engine import JobEngine
from app.orchestration.planner import Planner
from app.orchestration.sqlite_store import SQLiteJobStore
from app.orchestration.tools import (
    CancelJobTool,
    ConfirmJobTool,
    CreateJobTool,
    GetJobTool,
    ListJobsTool,
    PauseJobTool,
    ResumeJobTool,
)
from app.providers.base import AIProvider
from app.reminders.sqlite_store import SQLiteReminderStore
from app.tasks.sqlite_store import SQLiteTaskStore
from app.tools.base import PermissionLevel, Tool, ToolResult
from app.tools.defaults import register_default_tools
from app.tools.registry import ToolRegistry


class FakeProvider(AIProvider):
    def generate_reply(self, message: str) -> str:
        return "direct reply"


class FakeCapabilityRouter(CapabilityRouter):
    def __init__(self, capability: Capability, goal: str | None = None):
        self.capability = capability
        self.goal = goal

    def route(self, message: str) -> CapabilityDecision:
        return CapabilityDecision(capability=self.capability, goal=self.goal)


class SendReportTool(Tool):
    """A fake SAFE_ACTION tool standing in for a real capability the
    planner might call - "prepare a report"."""

    name = "prepare_report"
    description = "Prepares a short report."
    permission = PermissionLevel.SAFE_ACTION

    def __init__(self):
        self.calls = 0

    def execute(self) -> ToolResult:
        self.calls += 1
        return ToolResult.success(message="Report prepared.")


class SendEmailTool(Tool):
    """A fake CONFIRM-permission tool - an external, meaningful action
    that needs confirmation under MANUAL/ASSISTED autonomy."""

    name = "send_email"
    description = "Sends the prepared report by email."
    permission = PermissionLevel.CONFIRM

    def __init__(self):
        self.calls = 0

    def execute(self) -> ToolResult:
        self.calls += 1
        return ToolResult.success(message="Email sent.")


class OneShotPlanner(Planner):
    """Always proposes a single call to `prepare_report` - a minimal fake
    Planner standing in for GeminiPlanner in this integration test."""

    def plan(self, goal, *, available_tools, context, autonomy, replan_context=None):
        return [{"id": "s1", "tool": "prepare_report", "arguments": {}, "depends_on": []}]


class SendEmailPlanner(Planner):
    """Always proposes a single call to the CONFIRM-permission
    `send_email` tool."""

    def plan(self, goal, *, available_tools, context, autonomy, replan_context=None):
        return [{"id": "s1", "tool": "send_email", "arguments": {}, "depends_on": []}]


# --- default tool registration -------------------------------------------


def test_register_default_tools_adds_all_seven_job_tools():
    registry = ToolRegistry()
    job_store = SQLiteJobStore(":memory:")

    register_default_tools(
        registry,
        reminder_store=SQLiteReminderStore(":memory:"),
        automation_store=SQLiteAutomationStore(":memory:"),
        memory=SQLiteMemoryStore(":memory:"),
        task_store=SQLiteTaskStore(":memory:"),
        job_store=job_store,
    )

    for name in (
        "create_job",
        "list_jobs",
        "get_job",
        "confirm_job",
        "pause_job",
        "resume_job",
        "cancel_job",
    ):
        assert registry.has(name), f"{name} was not registered"


def test_register_default_tools_uses_the_given_job_store():
    registry = ToolRegistry()
    job_store = SQLiteJobStore(":memory:")

    register_default_tools(
        registry,
        reminder_store=SQLiteReminderStore(":memory:"),
        automation_store=SQLiteAutomationStore(":memory:"),
        memory=SQLiteMemoryStore(":memory:"),
        task_store=SQLiteTaskStore(":memory:"),
        job_store=job_store,
    )

    registry.execute("create_job", params={"goal": "test goal"})

    assert len(job_store.list()) == 1
    assert job_store.list()[0].goal == "test goal"


def test_create_job_tool_context_provider_is_wired_to_memory_facts():
    registry = ToolRegistry()
    job_store = SQLiteJobStore(":memory:")
    memory = SQLiteMemoryStore(":memory:")
    memory.save_fact("favorite_color", "blue")

    register_default_tools(
        registry,
        reminder_store=SQLiteReminderStore(":memory:"),
        automation_store=SQLiteAutomationStore(":memory:"),
        memory=memory,
        task_store=SQLiteTaskStore(":memory:"),
        job_store=job_store,
    )

    registry.execute("create_job", params={"goal": "goal"})

    assert job_store.list()[0].context == {"favorite_color": "blue"}


# --- end-to-end: Core TASK dispatch -> JobEngine execution -----------------


def test_end_to_end_goal_creates_and_completes_a_job():
    job_store = SQLiteJobStore(":memory:")
    tools = ToolRegistry()
    report_tool = SendReportTool()
    tools.register(report_tool)
    tools.register(CreateJobTool(store=job_store))
    tools.register(ListJobsTool(store=job_store))
    tools.register(GetJobTool(store=job_store))
    tools.register(ConfirmJobTool(store=job_store))
    tools.register(PauseJobTool(store=job_store))
    tools.register(ResumeJobTool(store=job_store))
    tools.register(CancelJobTool(store=job_store))

    jarvis = JarvisCore(
        provider=FakeProvider(),
        memory=SQLiteMemoryStore(":memory:"),
        tools=tools,
        capability_router=FakeCapabilityRouter(
            Capability.TASK, goal="prepare a report for the meeting"
        ),
    )

    reply, conversation_id, sources = jarvis.handle_chat("prepare a report for the meeting")

    assert sources == []
    jobs = job_store.list()
    assert len(jobs) == 1
    job = jobs[0]
    assert job.goal == "prepare a report for the meeting"
    assert job.status == JobStatus.PENDING

    # JobEngine now plans and executes it, entirely independently of Core.
    engine = JobEngine(
        store=job_store,
        tools=tools,
        planner=OneShotPlanner(),
        max_steps=10,
        max_total_steps=20,
        max_retries_per_step=2,
        max_replans=1,
        retry_delay_seconds=0,
        poll_interval_seconds=1000,
    )
    engine.tick()

    finished = job_store.get(job.id)
    assert finished.status == JobStatus.COMPLETED
    assert finished.plan.get("s1").status == StepStatus.SUCCEEDED
    assert report_tool.calls == 1
    assert finished.final_result


def test_end_to_end_manual_autonomy_requires_confirmation_before_running():
    job_store = SQLiteJobStore(":memory:")
    tools = ToolRegistry()
    email_tool = SendEmailTool()
    tools.register(email_tool)
    tools.register(CreateJobTool(store=job_store))
    tools.register(ConfirmJobTool(store=job_store))

    jarvis = JarvisCore(
        provider=FakeProvider(),
        memory=SQLiteMemoryStore(":memory:"),
        tools=tools,
        capability_router=FakeCapabilityRouter(Capability.TASK, goal="email the report"),
    )
    jarvis.tools.execute(
        "create_job", params={"goal": "email the report", "autonomy": "manual"}
    )
    job = job_store.list()[0]

    engine = JobEngine(
        store=job_store,
        tools=tools,
        planner=SendEmailPlanner(),
        max_steps=10,
        max_total_steps=20,
        max_retries_per_step=2,
        max_replans=1,
        retry_delay_seconds=0,
        poll_interval_seconds=1000,
    )
    engine.tick()

    awaiting = job_store.get(job.id)
    assert awaiting.status == JobStatus.AWAITING_CONFIRMATION
    assert email_tool.calls == 0

    # The user confirms via the same tool ConfirmJobTool exposes.
    tools.execute("confirm_job", params={"job_id": job.id, "approve": True})
    engine.tick()

    finished = job_store.get(job.id)
    assert finished.status == JobStatus.COMPLETED
    assert email_tool.calls == 1


def test_simple_direct_message_bypasses_orchestration_entirely():
    """A plain DIRECT message never touches JobStore, even when
    orchestration tools are registered."""
    job_store = SQLiteJobStore(":memory:")
    tools = ToolRegistry()
    tools.register(CreateJobTool(store=job_store))

    jarvis = JarvisCore(
        provider=FakeProvider(), memory=SQLiteMemoryStore(":memory:"), tools=tools
    )

    reply, _, sources = jarvis.handle_chat("What is the capital of Japan?")

    assert reply == "direct reply"
    assert sources == []
    assert job_store.list() == []
