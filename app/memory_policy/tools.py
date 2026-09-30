"""
Memory-write tools: how a MODEL may propose a write.

A model that decides the user wants something remembered does not touch a
store: it calls one of these tools, which builds a WriteProposal and hands it to
MemoryWriteService - where the deterministic WritePolicy, not the model, decides
whether it is persisted, staged, asked about or refused.

What the model can and cannot influence:

  * It supplies the key, value and kind - the CONTENT of the proposal.
  * It can only ever make a proposal WEAKER: `basis="inferred"` turns it into an
    inference (a candidate, never a fact). There is no way to claim more.
  * The proposal's ORIGIN is fixed here, by code: a call through these tools is
    the user's own request (USER_COMMAND) or an inference - never "external",
    and never something the model can relabel.
  * Secrets, sensitive information, contradictions and shape limits are checked
    by the policy whatever the model says.

The first three are SAFE_ACTION (they write low-risk user-authorised data; anything
riskier is refused or asked about by the policy) and `plannable = False`: a
goal-driven Job's planner is never offered them and `validate_plan` rejects a
step naming one - so text a Job read on the web ("Remember that the user
wants X") cannot become a step that writes memory - and `automatable = False`:
an automation can neither be created with one nor run one. They are used from chat.
The fourth, `commit_memory_write`, is the CONFIRM-level step of a sensitive request's
confirmation (see confirmation.py) - the only way anything sensitive is ever kept.
"""

from __future__ import annotations

from app.memory_policy.base import MAX_KEY_CHARS, MAX_VALUE_CHARS, Destination, Kind, Origin, WriteProposal
from app.memory_policy.confirmation import COMMIT_TOOL_NAME
from app.memory_policy.service import MemoryWriteService
from app.tools.base import PermissionLevel, Tool, ToolArgumentError, ToolParameter, ToolResult

_KINDS = {"fact": Kind.FACT, "preference": Kind.PREFERENCE}
_BASES = ("stated", "inferred")


def _result(outcome) -> ToolResult:
    data = {
        "persisted": outcome.persisted,
        "destination": outcome.decision.destination.value,
        "reason": outcome.decision.reason,
    }
    if outcome.decision.destination is Destination.REJECT:
        return ToolResult.error(outcome.message or "That can't be saved.")
    return ToolResult.success(data=data, message=outcome.message or "Nothing was saved.")


class RememberTool(Tool):
    """Propose remembering one fact or preference about the user."""

    name = "remember"
    description = (
        "Remember one thing about the user - a fact or a preference - that they asked you to keep, or "
        "stated plainly. Use kind 'preference' for likes/dislikes/settings, 'fact' for anything else. "
        "Use basis 'inferred' if the user did not say it and you are only concluding it. Credentials "
        "and secrets are never stored."
    )
    permission = PermissionLevel.SAFE_ACTION
    plannable = False
    automatable = False
    natural_language_arguments = True

    def __init__(self, service: MemoryWriteService):
        self.service = service

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(name="key", type="string", description=f"A short name for it (at most {MAX_KEY_CHARS} chars)"),
            ToolParameter(
                name="value", type="string", description=f"What to remember (at most {MAX_VALUE_CHARS} chars)", free_text=True
            ),
            ToolParameter(
                name="kind", type="string", description="'fact' (default) or 'preference'", required=False, default="fact"
            ),
            ToolParameter(
                name="basis",
                type="string",
                description="'stated' (the user said it; default) or 'inferred' (you concluded it)",
                required=False,
                default="stated",
            ),
        ]

    def validate_arguments(self, key: str, value: str, kind: str = "fact", basis: str = "stated") -> None:
        if not isinstance(key, str) or not key.strip() or not isinstance(value, str) or not value.strip():
            raise ToolArgumentError("Both a key and a value are needed.")
        if kind not in _KINDS:
            raise ToolArgumentError(f"kind must be one of: {sorted(_KINDS)}")
        if basis not in _BASES:
            raise ToolArgumentError(f"basis must be one of: {list(_BASES)}")

    def execute(self, key: str, value: str, kind: str = "fact", basis: str = "stated") -> ToolResult:
        try:
            self.validate_arguments(key, value, kind, basis)
        except ToolArgumentError as exc:
            return ToolResult.error(str(exc))
        proposal = WriteProposal(
            _KINDS[kind],
            key.strip(),
            value.strip(),
            Origin.INFERENCE if basis == "inferred" else Origin.USER_COMMAND,
            category="general",
            evidence=value.strip(),
            supersede=basis != "inferred",
        )
        return _result(self.service.apply(proposal))


class SetCurrentProjectTool(Tool):
    """Record a project the user is working on."""

    name = "set_current_project"
    description = "Record a project the user says they are working on (a short name)."
    permission = PermissionLevel.SAFE_ACTION
    plannable = False
    automatable = False

    def __init__(self, service: MemoryWriteService):
        self.service = service

    @property
    def parameters(self) -> list[ToolParameter]:
        return [ToolParameter(name="name", type="string", description="The project's name", free_text=True)]

    def execute(self, name: str) -> ToolResult:
        if not isinstance(name, str) or not name.strip():
            return ToolResult.error("A project needs a name.")
        proposal = WriteProposal(Kind.PROJECT, "project", name.strip(), Origin.USER_COMMAND, evidence=name.strip())
        return _result(self.service.apply(proposal))


class SetGoalTool(Tool):
    """Add a goal to one of the user's projects."""

    name = "set_goal"
    description = (
        "Add a goal to the user's project. Give the project's name unless the user has exactly one active project."
    )
    permission = PermissionLevel.SAFE_ACTION
    plannable = False
    automatable = False

    def __init__(self, service: MemoryWriteService):
        self.service = service

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(name="title", type="string", description="The goal", free_text=True),
            ToolParameter(name="project", type="string", description="The project's name", required=False, default=""),
        ]

    def execute(self, title: str, project: str = "") -> ToolResult:
        if not isinstance(title, str) or not title.strip():
            return ToolResult.error("A goal needs a title.")
        proposal = WriteProposal(
            Kind.GOAL, "goal", title.strip(), Origin.USER_COMMAND, evidence=title.strip(),
            project=(project or "").strip(),
        )
        return _result(self.service.apply(proposal))


class CommitMemoryWriteTool(Tool):
    """Persist a sensitive memory request the user has just CONFIRMED.

    Not for models, plans or automations: it is CONFIRM-level (the engine runs it only
    after the user approved the confirmation that gates it, through the existing
    ToolRegistry check), non-plannable, non-automatable, and it refuses any call that
    does not come from the very Job that asked (`source_job_id` is added by JobEngine
    and cannot be supplied by a plan or a caller). It never echoes what it saved.
    """

    name = COMMIT_TOOL_NAME
    description = "Save to memory something the user has just confirmed."
    permission = PermissionLevel.CONFIRM
    plannable = False
    automatable = False
    wants_job_provenance = True

    def __init__(self, service: MemoryWriteService):
        self.service = service

    @property
    def parameters(self) -> list[ToolParameter]:
        return [ToolParameter(name="pending_id", type="string", description="The held request's id")]

    def execute(
        self, pending_id: str, source_job_id: str | None = None, source_step_id: str | None = None
    ) -> ToolResult:
        if not source_job_id:
            return ToolResult.error("This can only run as an approved confirmation.")
        outcome = self.service.commit_pending(pending_id, source_job_id)
        if outcome.persisted:
            return ToolResult.success(message="Saved to memory.")
        if outcome.decision.destination is Destination.REJECT:
            return ToolResult.error(outcome.message or "That can't be saved.")
        return ToolResult.success(message=outcome.message or "Nothing was saved.")


def register_memory_tools(registry, service: MemoryWriteService) -> None:
    """Register the memory tools on `registry` (the app does this for `core.tools`)."""
    registry.register(RememberTool(service))
    registry.register(SetCurrentProjectTool(service))
    registry.register(SetGoalTool(service))
    registry.register(CommitMemoryWriteTool(service))
