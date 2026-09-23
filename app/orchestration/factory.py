"""
Builds the application's default JobStore/JobEngine from config - the one
place construction happens, same role app/automation/factory.py and
app/reminders/factory.py play for their own store/background-poller
pairs.
"""

from app.orchestration.base import JobStore
from app.orchestration.engine import JobEngine
from app.orchestration.planner import GeminiPlanner, Planner
from app.orchestration.sqlite_store import SQLiteJobStore
from app.tools.registry import ToolRegistry


def create_default_job_store(db_path: str) -> JobStore:
    return SQLiteJobStore(db_path)


def create_default_job_engine(
    store: JobStore,
    tools: ToolRegistry,
    *,
    max_steps: int,
    max_total_steps: int,
    max_retries_per_step: int,
    max_replans: int,
    retry_delay_seconds: float,
    poll_interval_seconds: float,
    planner: Planner | None = None,
    context_provider=None,
) -> JobEngine:
    """`context_provider` (added after M15's initial pass) is normally
    `memory.get_facts` - the same source CreateJobTool's own
    `context_provider` already uses (see app/tools/defaults.py) - so a
    Job's planner sees currently-known facts, not just whatever was known
    at the moment the Job was created (see JobEngine._refresh_context)."""
    return JobEngine(
        store=store,
        tools=tools,
        planner=planner or GeminiPlanner(),
        max_steps=max_steps,
        max_total_steps=max_total_steps,
        max_retries_per_step=max_retries_per_step,
        max_replans=max_replans,
        retry_delay_seconds=retry_delay_seconds,
        poll_interval_seconds=poll_interval_seconds,
        context_provider=context_provider,
    )
