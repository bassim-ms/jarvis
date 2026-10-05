"""
The one tool of Scheduled Job creation: `commit_scheduled_job_approval`.

Not for models, plans or automations: CONFIRM-level (the existing JobEngine runs it only after the user APPROVED the confirmation that gates
it, through the existing ToolRegistry check), non-plannable, non-automatable - so no Planner can name it and no web content a Job read can
reach it - and it refuses any call that does not come from the very Job that asked (`source_job_id` is added by JobEngine and cannot be
supplied by a plan or a caller). What it does is only ever ask the service to approve ONE stored proposal; the service re-validates it and
the store activates it atomically. It grants no capability: a Scheduled Job it activates still runs only under the READ-only profile.
"""

from __future__ import annotations

from app.scheduled_operations.operations import COMMIT_TOOL_NAME, ScheduledJobOperations
from app.scheduled_operations.service import ScheduledJobRequestError
from app.tools.base import PermissionLevel, Tool, ToolParameter, ToolResult


class CommitScheduledJobApprovalTool(Tool):
    name = COMMIT_TOOL_NAME
    description = "Make live a scheduled job the user has just confirmed."
    permission = PermissionLevel.CONFIRM
    plannable = False
    automatable = False
    wants_job_provenance = True

    def __init__(self, operations: ScheduledJobOperations):
        self.operations = operations

    @property
    def parameters(self) -> list[ToolParameter]:
        return [ToolParameter(name="scheduled_job_id", type="string", description="The proposed scheduled job's id")]

    def execute(self, scheduled_job_id: str, source_job_id: str | None = None, source_step_id: str | None = None) -> ToolResult:
        if not source_job_id:
            return ToolResult.error("This can only run as an approved confirmation.")
        try:
            return ToolResult.success(message=self.operations.commit_approval(scheduled_job_id, source_job_id))
        except ScheduledJobRequestError as exc:
            return ToolResult.error(str(exc))
