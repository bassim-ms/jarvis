"""
Asking the user before sensitive information is kept - through the EXISTING
confirmation machinery, not a new one.

A sensitive-but-not-secret memory request becomes an ordinary Job that is
already WAITING on an ordinary CONFIRMATION interaction:

    goal   "Save something to memory"            (generic: never the content)
    plan   one step: `commit_memory_write(pending_id)`   (a CONFIRM-level tool,
           built here by trusted code - no planner is involved, and the tool is
           not plannable, so no planner could produce it either)
    gate   the interaction: "Shall I save what you just told me to memory?"

The user answers it exactly as they answer any confirmation - "yes" / "no" in
the conversation it was asked in - through the existing resolver, JobEngine and
ToolRegistry. Only an approved step ever runs the tool, and only then is anything
persisted. A "no", an expiry, or silence persists nothing.

Nothing sensitive is in the Job, its plan, its events, the interaction, the
notification or the report: the plan carries only an opaque `pending_id`; the
content itself is held in memory (never on disk) by MemoryWriteService until
the step runs.
"""

from __future__ import annotations

from collections.abc import Callable

from app.orchestration.base import Job, JobStatus, Plan, PlanStep
from app.orchestration.interactions import InteractionStatus, InteractionType, default_ttl_seconds
from app.orchestration.sqlite_interaction_store import SQLiteInteractionStore
from app.orchestration.sqlite_store import SQLiteJobStore
from app.tools.base import PermissionLevel

COMMIT_TOOL_NAME = "commit_memory_write"
GOAL = "Save something to memory"
PROMPT = "Shall I save what you just told me to memory?"
_STEP_ID = "s1"
#: The confirmation job must never be re-planned by a model if its step fails:
#: any replan budget is "already spent".
_NO_REPLANS = 10_000


class SensitiveWriteConfirmer:
    """See the module docstring. `mark_surfaced(interaction_id, conversation_id)`
    (the resolver's own) records that the question was just put to the user, so a
    plain "yes"/"no" in that conversation answers it."""

    def __init__(
        self,
        interactions: SQLiteInteractionStore,
        *,
        mark_surfaced: Callable[[str, str], object] | None = None,
    ):
        self.interactions = interactions
        self.mark_surfaced = mark_surfaced

    def approved(self, job_id: str) -> bool:
        """Whether the user APPROVED this Job's confirmation (its interaction is
        RESOLVED - a decline, an expiry or a withdrawal is not). Read from the
        interaction store, so a forged call - or the step of a declined Job - can
        never commit."""
        return any(
            i.type is InteractionType.CONFIRMATION
            and i.status is InteractionStatus.RESOLVED
            and i.detail.get("origin") == "memory_policy"
            for i in self.interactions.list_for_job(job_id)
        )

    def ask(self, pending_id: str, conversation_id: str | None) -> str:
        """Create the waiting Job and its confirmation atomically; returns the Job's
        id. Raises on any fault - the caller then persists nothing."""
        job: Job = SQLiteJobStore.build_job(
            GOAL, conversation_id=conversation_id, status=JobStatus.AWAITING_CONFIRMATION
        )
        job.plan = Plan(
            steps=[
                PlanStep(
                    id=_STEP_ID,
                    tool_name=COMMIT_TOOL_NAME,
                    arguments={"pending_id": pending_id},
                    permission=PermissionLevel.CONFIRM.value,
                )
            ]
        )
        job.total_step_count = 1
        job.pending_step_ids = [_STEP_ID]
        job.replan_count = _NO_REPLANS
        job.confirmation_message = PROMPT
        created, interaction = self.interactions.create_job_awaiting_confirmation(
            job,
            prompt=PROMPT,
            step_ids=[_STEP_ID],
            detail={"origin": "memory_policy", "dangerous": False},
            ttl_seconds=default_ttl_seconds(InteractionType.CONFIRMATION),
        )
        if self.mark_surfaced is not None and conversation_id:
            try:
                self.mark_surfaced(interaction.id, conversation_id)
            except Exception:  # noqa: BLE001 - best effort: the ordinary "waiting on you" notice shows it next
                pass
        return created.id
