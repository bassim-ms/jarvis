"""
Builds the application's default JobStore/JobEngine from config - the one
place construction happens, same role app/automation/factory.py and
app/reminders/factory.py play for their own store/background-poller
pairs.
"""

from app.notifications.base import NotificationRouter
from app.orchestration.base import JobStore
from app.orchestration.engine import JobEngine
from app.orchestration.evaluator import GeminiResultEvaluator, ResultEvaluator
from app.orchestration.planner import GeminiPlanner, Planner
from app.providers.base import AIProvider
from app.reminders.notifier import Notifier
from app.orchestration.interactions import InteractionStore
from app.orchestration.sqlite_interaction_store import SQLiteInteractionStore
from app.orchestration.sqlite_store import SQLiteJobStore
from app.tools.registry import ToolRegistry


def create_default_job_store(db_path: str) -> JobStore:
    return SQLiteJobStore(db_path)


def create_default_interaction_store(job_store: SQLiteJobStore) -> InteractionStore:
    """The interaction store for `job_store`: same database file, same
    connection and lock (see app/orchestration/sqlite_interaction_store.py)."""
    return SQLiteInteractionStore(job_store)


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
    evaluator: ResultEvaluator | None = None,
    provider: AIProvider | None = None,
    notifier: Notifier | None = None,
    interaction_store: InteractionStore | None = None,
    notification_router: NotificationRouter | None = None,
    notify_on_completion: bool = True,
    objective_continuer=None,
) -> JobEngine:
    """`context_provider` (added after M15's initial pass) is normally
    `memory.get_facts` - the same source CreateJobTool's own
    `context_provider` already uses (see app/tools/defaults.py) - so a
    Job's planner sees currently-known facts, not just whatever was known
    at the moment the Job was created (see JobEngine._refresh_context).

    `provider` is the AIProvider the planner (planning AND replanning) and
    the result evaluator both call - in the running app `core.provider`, the
    AIProviderManager, so planning and evaluation get the same ordered
    fallback/cooldown handling every other AI call in this app (DIRECT,
    SEARCH, tool calls, `analyze`) already gets. Without it the planner
    would talk to a single, unmanaged GeminiProvider(): one rate-limited
    provider would then fail planning even while the others are healthy.
    An explicitly passed `planner`/`evaluator` still wins. With no
    `provider` (unit tests, scripts) the behavior is as before: a default
    GeminiPlanner and no evaluator."""
    if evaluator is None and provider is not None:
        evaluator = GeminiResultEvaluator(provider)
    return JobEngine(
        store=store,
        tools=tools,
        planner=planner or GeminiPlanner(provider),
        max_steps=max_steps,
        max_total_steps=max_total_steps,
        max_retries_per_step=max_retries_per_step,
        max_replans=max_replans,
        retry_delay_seconds=retry_delay_seconds,
        poll_interval_seconds=poll_interval_seconds,
        context_provider=context_provider,
        evaluator=evaluator,
        notifier=notifier,
        interaction_store=interaction_store,
        notification_router=notification_router,
        notify_on_completion=notify_on_completion,
        objective_continuer=objective_continuer,
    )
