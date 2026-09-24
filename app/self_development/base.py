"""
Self-Development Engine types (V1) - architecture only, the same
"types and pure validation, no I/O" role every other app/*/base.py in
this codebase plays. app/self_development/planner.py builds a
DevelopmentPlan from an AI response (strictly validated first);
app/self_development/engine.py drives a DevelopmentSession through its
own bounded state machine. Nothing here calls an AI provider, executes
a command, touches Git, or writes a file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from app.self_development.safety import (
    is_protected_system_path,
    is_safe_relative_path,
    is_secret_path,
)
from app.tools.base import PermissionLevel


class DevelopmentCapability(Enum):
    """The eight named self-development actions (V1's own spec) plus
    SECURITY_EDIT (added when a hardening pass found that nothing
    protected the engine's OWN implementation from being an ordinary
    edit target) - pure identifiers, never a second permission system.
    Each one maps to exactly one PermissionLevel (app/tools/base.py)
    below, via `permission_for`, and to one or more Tool(s) registered
    on app/self_development/engine.py:SelfDevelopmentEngine's OWN
    ToolRegistry - the SAME PermissionLevel/confirmation enforcement
    every other tool in this codebase already uses (app/tools/
    registry.py), never a parallel mechanism this module invents.

    SECURITY_EDIT is deliberately NOT just "CODE_EDIT at a stronger
    PermissionLevel" - it is its own capability, gated by its own
    engine-level approval call (`SelfDevelopmentEngine.
    approve_security_edit`), entirely separate from ordinary plan
    approval (`approve_plan`). Approving a plan's ordinary CODE_EDIT
    NEVER also authorizes SECURITY_EDIT - see
    app/self_development/engine.py's own module docstring.
    """

    CODE_READ = "code_read"
    CODE_EDIT = "code_edit"
    COMMAND_RUN = "command_run"
    TEST_RUN = "test_run"
    GIT_DIFF = "git_diff"
    BRANCH_CREATE = "branch_create"
    ROLLBACK = "rollback"
    COMMIT = "commit"
    SECURITY_EDIT = "security_edit"


#: The one place each DevelopmentCapability's PermissionLevel is
#: decided - matches V1's own spec exactly: CODE_READ/GIT_DIFF/TEST_RUN/
#: BRANCH_CREATE automatic (READ or SAFE_ACTION - see PermissionLevel's
#: own docstring for the READ-vs-SAFE_ACTION distinction: read-only
#: work is READ, a reversible side effect like running tests or
#: creating a branch is SAFE_ACTION); CODE_EDIT/COMMAND_RUN "controlled"/
#: require approval -> CONFIRM; ROLLBACK requires confirmation ->
#: CONFIRM; COMMIT requires EXPLICIT confirmation -> DANGEROUS, the
#: strongest tier this codebase has, matching that it's the one action
#: in this whole subsystem that's meant to be genuinely hard to trigger
#: by accident. SECURITY_EDIT shares COMMIT's DANGEROUS tier (the
#: strongest PermissionLevel this codebase has) but is enforced through
#: its own, separate Tool/approval call - see the class docstring above.
_CAPABILITY_PERMISSIONS: dict[DevelopmentCapability, PermissionLevel] = {
    DevelopmentCapability.CODE_READ: PermissionLevel.READ,
    DevelopmentCapability.GIT_DIFF: PermissionLevel.READ,
    DevelopmentCapability.TEST_RUN: PermissionLevel.SAFE_ACTION,
    DevelopmentCapability.BRANCH_CREATE: PermissionLevel.SAFE_ACTION,
    DevelopmentCapability.CODE_EDIT: PermissionLevel.CONFIRM,
    DevelopmentCapability.COMMAND_RUN: PermissionLevel.CONFIRM,
    DevelopmentCapability.ROLLBACK: PermissionLevel.CONFIRM,
    DevelopmentCapability.COMMIT: PermissionLevel.DANGEROUS,
    DevelopmentCapability.SECURITY_EDIT: PermissionLevel.DANGEROUS,
}


def permission_for(capability: DevelopmentCapability) -> PermissionLevel:
    return _CAPABILITY_PERMISSIONS[capability]


class SessionStage(Enum):
    """A DevelopmentSession's own bounded state machine - deliberately
    linear (no going "backward" except via an explicit ROLLED_BACK/
    CANCELLED/BLOCKED terminal state), so app/self_development/
    engine.py never has to guess what's allowed next."""

    STARTED = "started"
    BLOCKED_DIRTY_TREE = "blocked_dirty_tree"
    PLAN_PROPOSED = "plan_proposed"
    PLAN_REJECTED = "plan_rejected"
    BRANCH_CREATED = "branch_created"
    #: A plan touching a protected system path (app/self_development/
    #: safety.py:is_protected_system_path) stops HERE after branch
    #: creation - `_apply_plan` is never called until a SEPARATE,
    #: explicit `approve_security_edit(confirmed=True)` call resolves
    #: this stage (see engine.py). A plan with no protected paths skips
    #: this stage entirely.
    AWAITING_SECURITY_APPROVAL = "awaiting_security_approval"
    SECURITY_EDIT_DECLINED = "security_edit_declined"
    EDITING = "editing"
    CHECKS_PASSED = "checks_passed"
    CHECKS_FAILED = "checks_failed"
    AWAITING_COMMIT_APPROVAL = "awaiting_commit_approval"
    COMMITTED = "committed"
    COMMIT_DECLINED = "commit_declined"
    ROLLED_BACK = "rolled_back"


#: Stages from which `rollback` is meaningful - once COMMITTED or
#: already ROLLED_BACK, there is nothing left to roll back on this
#: branch (see engine.py:SelfDevelopmentEngine.rollback).
ROLLBACK_AVAILABLE_STAGES = frozenset(
    {
        SessionStage.BRANCH_CREATED,
        SessionStage.AWAITING_SECURITY_APPROVAL,
        SessionStage.SECURITY_EDIT_DECLINED,
        SessionStage.EDITING,
        SessionStage.CHECKS_PASSED,
        SessionStage.CHECKS_FAILED,
        SessionStage.AWAITING_COMMIT_APPROVAL,
        SessionStage.COMMIT_DECLINED,
    }
)

#: Stages a persisted session record is still considered "abandoned"
#: (a real branch exists, with nothing committed) if the process
#: restarts while sitting in one of them - see
#: engine.py:scan_for_abandoned_sessions. COMMITTED/ROLLED_BACK are
#: deliberately excluded: both are fully resolved, nothing left to
#: recover, and their persisted record is deleted at that point (see
#: engine.py:_delete_session_record).
ABANDONED_SESSION_STAGES = ROLLBACK_AVAILABLE_STAGES


class DevelopmentSessionError(ValueError):
    """Raised when an app/self_development/engine.py caller requests a
    transition a DevelopmentSession's current SessionStage doesn't
    allow (e.g. approving a plan twice, or committing before checks
    ever ran) - never silently reinterpreted as a different, valid
    transition."""


class DevelopmentPlanError(ValueError):
    """Raised by PlannedFileChange/DevelopmentPlan's own __post_init__
    for a structurally invalid field, OR for a file path that fails
    the safety checks (path traversal, secret file) - never a parsing/
    LLM-output-validation error (that's app/self_development/
    planner.py's job, kept entirely separate - the same
    ContextReasoningError/ContextReasonerError split app/
    context_reasoning/ already established)."""


def _validate_text(value: Any, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise DevelopmentPlanError(f"{field_name} must be a non-empty string, got: {value!r}")


@dataclass(frozen=True)
class PlannedFileChange:
    """One file a DevelopmentPlan proposes touching. `path` is always
    validated as a safe, repo-relative path (never `..`, never
    absolute, never a secret-named file) at construction time - a
    DevelopmentPlan can never even be built with an unsafe or secret
    path in its file list, regardless of what an LLM proposed."""

    path: str
    description: str

    def __post_init__(self) -> None:
        _validate_text(self.path, "PlannedFileChange.path")
        _validate_text(self.description, "PlannedFileChange.description")
        if not is_safe_relative_path(self.path):
            raise DevelopmentPlanError(f"Unsafe file path in plan: {self.path!r}")
        if is_secret_path(self.path):
            raise DevelopmentPlanError(
                f"A DevelopmentPlan may never propose touching a secret/config "
                f"file: {self.path!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "description": self.description}

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "PlannedFileChange":
        return PlannedFileChange(path=data["path"], description=data["description"])


@dataclass(frozen=True)
class DevelopmentPlan:
    """A structured, human-approvable proposal for a code change -
    never itself executable. Exactly the seven fields V1's own spec
    names: objective, files_affected ("files likely affected" +
    "intended changes" per-file), dependencies, tests_to_run
    ("tests/checks"), risks, and required_permissions. Approving this
    (see engine.py:SelfDevelopmentEngine.approve_plan) authorizes every
    CODE_EDIT this session makes for the rest of its bounded repair
    loop too - it is never re-requested per edit, the same way
    approving a Job's confirmation batch (M15, app/orchestration/
    tools.py:ConfirmJobTool) authorizes that whole batch at once, not
    step by step."""

    objective: str
    files_affected: tuple[PlannedFileChange, ...] = ()
    dependencies: tuple[str, ...] = ()
    tests_to_run: tuple[str, ...] = ()
    risks: tuple[str, ...] = ()
    required_permissions: tuple[DevelopmentCapability, ...] = ()

    def __post_init__(self) -> None:
        _validate_text(self.objective, "DevelopmentPlan.objective")
        if not isinstance(self.files_affected, tuple) or any(
            not isinstance(f, PlannedFileChange) for f in self.files_affected
        ):
            raise DevelopmentPlanError("files_affected must be a tuple of PlannedFileChange")
        for field_name in ("dependencies", "tests_to_run", "risks"):
            values = getattr(self, field_name)
            if not isinstance(values, tuple) or any(
                not isinstance(v, str) or not v.strip() for v in values
            ):
                raise DevelopmentPlanError(
                    f"DevelopmentPlan.{field_name} must be a tuple of non-empty strings"
                )
        if not isinstance(self.required_permissions, tuple) or any(
            not isinstance(p, DevelopmentCapability) for p in self.required_permissions
        ):
            raise DevelopmentPlanError(
                "required_permissions must be a tuple of DevelopmentCapability"
            )
        self._check_required_permissions_are_not_under_declared()

    def _check_required_permissions_are_not_under_declared(self) -> None:
        """`required_permissions` must be an actual enforcement
        mechanism, never descriptive metadata a plan can under-state -
        checked HERE, at construction, so no DevelopmentPlan can exist
        (however it was built: the planner, a test, a future caller)
        that claims fewer permissions than its own declared operations
        require. This is deliberately about what the plan ITSELF says
        it will touch (`files_affected`) - never about what an LLM
        merely SAYS in `required_permissions` with no corresponding
        operation, which is harmless over-declaration, not a security
        problem."""
        declared = set(self.required_permissions)
        missing: list[str] = []

        if self.files_affected and DevelopmentCapability.CODE_EDIT not in declared:
            missing.append(
                f"files_affected is non-empty but required_permissions is missing "
                f"{DevelopmentCapability.CODE_EDIT.value!r}"
            )

        protected = self.protected_file_paths()
        if protected and DevelopmentCapability.SECURITY_EDIT not in declared:
            missing.append(
                f"files_affected touches protected path(s) {sorted(protected)} but "
                f"required_permissions is missing {DevelopmentCapability.SECURITY_EDIT.value!r}"
            )

        if missing:
            raise DevelopmentPlanError(
                "DevelopmentPlan under-declares required_permissions: " + "; ".join(missing)
            )

    def file_paths(self) -> frozenset[str]:
        return frozenset(f.path for f in self.files_affected)

    def protected_file_paths(self) -> frozenset[str]:
        """Which of `files_affected` fall under a protected system path
        (app/self_development/safety.py:is_protected_system_path) - a
        non-empty result means this plan needs SECURITY_EDIT approval,
        never just ordinary CODE_EDIT (see engine.py:
        SelfDevelopmentEngine.approve_security_edit)."""
        return frozenset(f.path for f in self.files_affected if is_protected_system_path(f.path))

    def to_dict(self) -> dict[str, Any]:
        return {
            "objective": self.objective,
            "files_affected": [f.to_dict() for f in self.files_affected],
            "dependencies": list(self.dependencies),
            "tests_to_run": list(self.tests_to_run),
            "risks": list(self.risks),
            "required_permissions": [p.value for p in self.required_permissions],
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "DevelopmentPlan":
        return DevelopmentPlan(
            objective=data["objective"],
            files_affected=tuple(
                PlannedFileChange.from_dict(f) for f in data.get("files_affected", ())
            ),
            dependencies=tuple(data.get("dependencies", ())),
            tests_to_run=tuple(data.get("tests_to_run", ())),
            risks=tuple(data.get("risks", ())),
            required_permissions=tuple(
                DevelopmentCapability(p) for p in data.get("required_permissions", ())
            ),
        )


@dataclass(frozen=True)
class CommandResult:
    """The outcome of one bounded command execution
    (app/self_development/command_runner.py) - stdout/stderr are
    always plain text, always truncated to a bounded length before
    this object is ever built (never unbounded process output held in
    memory or shown to a user/model)."""

    argv: tuple[str, ...]
    exit_code: int | None
    stdout: str
    stderr: str
    timed_out: bool

    @property
    def succeeded(self) -> bool:
        return not self.timed_out and self.exit_code == 0


@dataclass
class CheckRun:
    """One test/check execution attempt during a session, kept for the
    final DevelopmentSummary's own reporting."""

    command: str
    result: CommandResult


@dataclass
class DevelopmentSummary:
    """What `engine.py:SelfDevelopmentEngine.finalize` produces after
    editing/checking finishes (successfully or not) - exactly the
    fields V1's own spec names for the post-implementation report."""

    files_changed: tuple[str, ...]
    change_summary: str
    checks_run: tuple[CheckRun, ...]
    remaining_failures: tuple[str, ...]
    git_diff_summary: str
    rollback_available: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "files_changed": list(self.files_changed),
            "change_summary": self.change_summary,
            "checks_run": [
                {
                    "command": c.command,
                    "succeeded": c.result.succeeded,
                    "exit_code": c.result.exit_code,
                    "timed_out": c.result.timed_out,
                }
                for c in self.checks_run
            ],
            "remaining_failures": list(self.remaining_failures),
            "git_diff_summary": self.git_diff_summary,
            "rollback_available": self.rollback_available,
        }


@dataclass
class DevelopmentSession:
    """One end-to-end self-development request's own state - created
    once per request (app/self_development/engine.py:
    SelfDevelopmentEngine.start), never shared across requests. This is
    plain mutable session state (not a validated value type like
    DevelopmentPlan above), owned entirely by SelfDevelopmentEngine."""

    id: str
    request: str
    stage: SessionStage
    original_branch: str
    plan: DevelopmentPlan | None = None
    dev_branch: str | None = None
    edited_files: tuple[str, ...] = ()
    repair_attempts: int = 0
    checks_run: tuple[CheckRun, ...] = field(default_factory=tuple)
    summary: DevelopmentSummary | None = None
    messages: tuple[str, ...] = ()

    def log(self, message: str) -> None:
        self.messages = self.messages + (message,)

    def rollback_available(self) -> bool:
        return self.stage in ROLLBACK_AVAILABLE_STAGES and self.dev_branch is not None
