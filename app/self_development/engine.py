"""
SelfDevelopmentEngine (V1) - the ONLY thing in this package that calls
its own ToolRegistry. Drives a DevelopmentSession through its bounded
state machine (app/self_development/base.py:SessionStage): propose a
plan -> wait for approval -> create a branch -> [if the plan touches a
protected path, wait for a SEPARATE SECURITY_EDIT approval] -> edit ->
test -> a SMALL, BOUNDED number of repair attempts -> summarize -> wait
for a SEPARATE commit approval -> commit (or rollback).

This is a controlled development SUBSYSTEM, not a second M15
(app/orchestration/): there is no Plan/PlanStep/Job here, no
dependency graph, no background poller. Every actual action is one
call to `self.tools.execute(name, params=..., confirmed=...)` - the
EXACT SAME PermissionLevel/confirmation enforcement every other tool
in this codebase already goes through (app/tools/registry.py). This
engine decides WHEN to pass `confirmed=True` - only in direct response
to a real approval/confirmation call this class exposes (`approve_plan`,
`approve_security_edit`, `approve_commit`, `rollback`) - it never
grants itself permission or fabricates an approval that didn't happen.

SECURITY_EDIT boundary (hardening pass, added after a security review
found nothing protected this engine's OWN source from being an
ordinary edit target): approving a plan's ordinary CODE_EDIT via
`approve_plan` NEVER authorizes touching a protected system path
(app/self_development/safety.py:is_protected_system_path). If a plan's
`files_affected` includes one, `approve_plan` creates the branch and
STOPS at SessionStage.AWAITING_SECURITY_APPROVAL - `_apply_plan` is
never called until a completely separate `approve_security_edit(
confirmed=True)` call resolves it. The LLM that PROPOSED the plan has
no path to satisfy this itself: `required_permissions` in a
DevelopmentPlan is validated (app/self_development/base.py) but is
descriptive of what the plan needs, never an approval grant, and
`approve_security_edit` only ever proceeds on an explicit `confirmed`
argument supplied by whatever is driving the engine - never inferred
from the plan, the request, or any AI output.

`self.tools` is a SEPARATE ToolRegistry from JarvisCore's own
`self.tools` (app/core.py) - these development tools are never
reachable through the ordinary TOOL capability/ToolCallBuilder path,
and JarvisCore's own tools are never reachable from here either. The
only bridge between the two systems is JarvisCore calling
`SelfDevelopmentEngine.start()`/`approve_plan()`/etc. directly (see
app/core.py's own self-development wiring).
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from app.self_development.base import (
    ABANDONED_SESSION_STAGES,
    CheckRun,
    CommandResult,
    DevelopmentPlan,
    DevelopmentSession,
    DevelopmentSessionError,
    DevelopmentSummary,
    SessionStage,
)
from app.self_development.git import GitClient, branch_name_for
from app.self_development.planner import DevelopmentPlanner, DevelopmentPlanningError
from app.self_development.safety import is_protected_system_path
from app.tools.base import ToolResultStatus
from app.tools.registry import ToolRegistry

#: Failure output bounded before it's ever handed to the planner for a
#: repair attempt or included in a summary - never an unbounded dump.
_MAX_FAILURE_TEXT_CHARS = 8_000


def _new_session_id() -> str:
    return uuid.uuid4().hex[:12]


# ---------------------------------------------------------------------------
# Minimal crash-recovery metadata (hardening pass) - deliberately NOT a
# persistence subsystem: one small JSON file per session, written
# best-effort after each meaningful state change, read only to REPORT
# abandoned sessions - never to resume, reset, or delete anything
# automatically. Stored under `.git/jarvis_selfdev_sessions/` - inside
# Git's own metadata directory, so it's never listed/read/searched by
# this engine's own repo tools, never shown in `git status`/diffs, and
# never mistaken for tracked project content.
# ---------------------------------------------------------------------------


def _session_records_dir(project_root: str) -> Path:
    return Path(project_root) / ".git" / "jarvis_selfdev_sessions"


def _session_record_path(project_root: str, session_id: str) -> Path:
    return _session_records_dir(project_root) / f"{session_id}.json"


def _persist_session_record(project_root: str, session: DevelopmentSession) -> None:
    """Best-effort only - a persistence failure must never break the
    actual workflow. Called AFTER a state change already happened, not
    as part of deciding whether it's allowed to."""
    try:
        directory = _session_records_dir(project_root)
        directory.mkdir(parents=True, exist_ok=True)
        record = {
            "id": session.id,
            "branch_name": session.dev_branch,
            "original_branch": session.original_branch,
            "modified_files": list(session.edited_files),
            "stage": session.stage.value,
        }
        _session_record_path(project_root, session.id).write_text(
            json.dumps(record), encoding="utf-8"
        )
    except OSError:
        pass


def _delete_session_record(project_root: str, session_id: str) -> None:
    """Called only once a session reaches a fully resolved stage
    (COMMITTED/ROLLED_BACK) - nothing left to recover, so nothing left
    to report either."""
    try:
        _session_record_path(project_root, session_id).unlink(missing_ok=True)
    except OSError:
        pass


def scan_for_abandoned_sessions(project_root: str) -> list[dict]:
    """Read-only: reports every persisted session record still sitting
    in a non-terminal stage (app/self_development/base.py:
    ABANDONED_SESSION_STAGES) - a real branch exists, with nothing
    committed and nothing rolled back. NEVER deletes, resets, restores,
    or otherwise acts on anything found - purely informational, safe
    to call at any time, including application startup. A missing or
    unreadable record is silently skipped, never raised."""
    directory = _session_records_dir(project_root)
    if not directory.is_dir():
        return []

    abandoned = []
    for record_path in sorted(directory.glob("*.json")):
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        try:
            stage = SessionStage(record.get("stage"))
        except ValueError:
            continue
        if stage in ABANDONED_SESSION_STAGES:
            abandoned.append(record)
    return abandoned


class SelfDevelopmentEngine:
    def __init__(
        self,
        project_root: str,
        tools: ToolRegistry,
        git_client: GitClient,
        planner: DevelopmentPlanner,
        *,
        max_repair_iterations: int,
    ):
        self.project_root = project_root
        self.tools = tools
        self.git = git_client
        self.planner = planner
        self.max_repair_iterations = max_repair_iterations
        self.sessions: dict[str, DevelopmentSession] = {}

    def scan_for_abandoned_sessions(self) -> list[dict]:
        """See the module-level `scan_for_abandoned_sessions` docstring
        - read-only, reports only, bound to this engine's own
        `project_root`."""
        return scan_for_abandoned_sessions(self.project_root)

    def _persist(self, session: DevelopmentSession) -> None:
        _persist_session_record(self.project_root, session)

    # -- 1-3: understand request -> inspect repo -> create change plan ------

    def start(self, request: str) -> DevelopmentSession:
        """Refuses to start at all if the working tree already has
        uncommitted changes - "detect and report rather than blindly
        resetting" (V1's own explicit requirement). Otherwise proposes
        a DevelopmentPlan and stops there - no branch is created and no
        file is touched until `approve_plan` explicitly says so."""
        original_branch = self.git.current_branch()

        if self.git.has_uncommitted_changes():
            session = DevelopmentSession(
                id=_new_session_id(),
                request=request,
                stage=SessionStage.BLOCKED_DIRTY_TREE,
                original_branch=original_branch,
            )
            session.log(
                "Uncommitted changes already exist in the working tree - "
                "refusing to start a self-development session. Commit or "
                "stash your own changes first."
            )
            self.sessions[session.id] = session
            return session

        session = DevelopmentSession(
            id=_new_session_id(),
            request=request,
            stage=SessionStage.STARTED,
            original_branch=original_branch,
        )
        self.sessions[session.id] = session

        try:
            digest = self._build_repo_digest()
            plan = self.planner.propose_plan(request, digest)
        except DevelopmentPlanningError as exc:
            session.stage = SessionStage.PLAN_REJECTED
            session.log(f"Could not produce a development plan: {exc}")
            return session

        session.plan = plan
        session.stage = SessionStage.PLAN_PROPOSED
        session.log("Development plan proposed - awaiting approval.")

        protected = plan.protected_file_paths()
        if protected:
            session.log(
                "This plan touches protected security-critical file(s): "
                f"{', '.join(sorted(protected))}. Approving the plan will NOT "
                "authorize editing them - a separate SECURITY_EDIT approval "
                "(approve_security_edit) will be required before those "
                "specific files can be touched."
            )
        return session

    def _build_repo_digest(self) -> str:
        result = self.tools.execute("self_dev_list_files")
        if result.status is not ToolResultStatus.SUCCESS or not result.data:
            return "(no files listed)"
        return "\n".join(
            f"{entry['path']}" + (" [secret - contents never shown]" if entry["secret"] else "")
            for entry in result.data
        )

    # -- 4-6: ask user for approval -> create branch -> edit files ----------

    def approve_plan(self, session: DevelopmentSession, approved: bool) -> DevelopmentSession:
        """Authorizes ordinary CODE_EDIT for this session's bounded
        editing/repair loop - NEVER SECURITY_EDIT (see this module's
        own docstring). If the plan touches no protected path, this
        single call runs the entire rest of the workflow synchronously
        (edit -> test -> bounded repair -> finalize). If it does, this
        call stops right after branch creation and
        `approve_security_edit` must be called separately."""
        if session.stage is not SessionStage.PLAN_PROPOSED:
            raise DevelopmentSessionError(
                f"Cannot approve a plan from stage '{session.stage.value}'."
            )
        if not approved:
            session.stage = SessionStage.PLAN_REJECTED
            session.log("Plan rejected - no branch created, no files touched.")
            return session

        branch = branch_name_for(session.id)
        result = self.tools.execute("self_dev_create_branch", params={"name": branch})
        if result.status is not ToolResultStatus.SUCCESS:
            session.stage = SessionStage.PLAN_REJECTED
            session.log(f"Could not create a development branch: {result.message}")
            return session

        session.dev_branch = branch
        session.stage = SessionStage.BRANCH_CREATED
        session.log(f"Created development branch '{branch}'.")
        self._persist(session)

        if session.plan.protected_file_paths():
            session.stage = SessionStage.AWAITING_SECURITY_APPROVAL
            session.log(
                "Branch created. Waiting for a separate SECURITY_EDIT approval "
                "(approve_security_edit) before any protected file is edited."
            )
            self._persist(session)
            return session

        self._proceed_with_editing(session)
        return session

    def approve_security_edit(
        self, session: DevelopmentSession, *, confirmed: bool | None
    ) -> DevelopmentSession:
        """The ONLY way protected-path edits proceed. `confirmed` must
        be explicitly `True` - never inferred from the plan's own
        `required_permissions`, the request text, or any AI output.
        `None` (not yet asked) or `False` (declined) both leave the
        session's protected files untouched; declining marks the
        session SECURITY_EDIT_DECLINED rather than silently retrying."""
        if session.stage is not SessionStage.AWAITING_SECURITY_APPROVAL:
            raise DevelopmentSessionError(
                f"Cannot approve SECURITY_EDIT from stage '{session.stage.value}'."
            )
        if confirmed is not True:
            if confirmed is False:
                session.stage = SessionStage.SECURITY_EDIT_DECLINED
                session.log(
                    "SECURITY_EDIT declined - protected files were not edited. "
                    "The session's branch still exists for manual review or rollback."
                )
            else:
                session.log("SECURITY_EDIT requires an explicit confirmation.")
            self._persist(session)
            return session

        session.log("SECURITY_EDIT approved - proceeding to edit protected file(s).")
        self._persist(session)
        self._proceed_with_editing(session)
        return session

    def _proceed_with_editing(self, session: DevelopmentSession) -> None:
        """Shared tail of `approve_plan` (plans with no protected
        paths) and `approve_security_edit` (plans that needed it) -
        edit -> test -> bounded repair -> finalize."""
        self._apply_plan(session)
        self._persist(session)
        self._run_checks_with_bounded_repair(session)
        self._persist(session)
        self._finalize(session)
        self._persist(session)

    def _apply_plan(self, session: DevelopmentSession) -> None:
        session.stage = SessionStage.EDITING
        edited = list(session.edited_files)
        for change in session.plan.files_affected:
            if self._edit_one_file(session, change.path, change.description, session.plan.objective):
                if change.path not in edited:
                    edited.append(change.path)
        session.edited_files = tuple(edited)

    def _edit_one_file(
        self, session: DevelopmentSession, path: str, description: str, objective: str
    ) -> bool:
        """Routes to the CODE_EDIT tool for an ordinary path, or the
        SEPARATE SECURITY_EDIT tool for a protected one - never the
        other way around (see app/self_development/editor.py, where
        each tool refuses the other's path category outright). Both
        are only ever called with `confirmed=True` here, and this
        method is only ever reached via `_proceed_with_editing`, which
        is only ever reached after the matching real approval
        (`approve_plan` for ordinary paths, `approve_security_edit` for
        protected ones) already happened."""
        read_result = self.tools.execute("self_dev_read_file", params={"path": path})
        current_content = (
            read_result.data["content"]
            if read_result.status is ToolResultStatus.SUCCESS
            else ""
        )
        try:
            new_content = self.planner.propose_file_content(
                objective, description, path, current_content
            )
        except DevelopmentPlanningError as exc:
            session.log(f"Could not generate content for '{path}': {exc}")
            return False

        tool_name = (
            "self_dev_security_edit_file" if is_protected_system_path(path) else "self_dev_edit_file"
        )
        edit_result = self.tools.execute(
            tool_name, params={"path": path, "content": new_content}, confirmed=True
        )
        if edit_result.status is not ToolResultStatus.SUCCESS:
            session.log(f"Edit rejected for '{path}': {edit_result.message}")
            return False
        session.log(f"Edited '{path}'.")
        return True

    # -- 7-9: run tests/checks -> inspect failures -> bounded fixes ---------

    def _run_checks_with_bounded_repair(self, session: DevelopmentSession) -> None:
        while True:
            check = self._run_tests(session)
            if check.result.succeeded:
                session.stage = SessionStage.CHECKS_PASSED
                session.log("Checks passed.")
                return

            if session.repair_attempts >= self.max_repair_iterations:
                session.stage = SessionStage.CHECKS_FAILED
                session.log(
                    f"Checks still failing after {session.repair_attempts} bounded "
                    "repair attempt(s) - stopping (max_repair_iterations reached)."
                )
                return

            session.repair_attempts += 1
            session.log(
                f"Checks failed - bounded repair attempt "
                f"{session.repair_attempts}/{self.max_repair_iterations}."
            )
            self._attempt_repair(session, check.result)

    def _run_tests(self, session: DevelopmentSession) -> CheckRun:
        result = self.tools.execute("self_dev_run_tests")
        if result.status is ToolResultStatus.SUCCESS:
            data = result.data
            command_result = CommandResult(
                argv=(), exit_code=data["exit_code"], stdout=data["stdout"],
                stderr=data["stderr"], timed_out=data["timed_out"],
            )
        else:
            command_result = CommandResult(
                argv=(), exit_code=None, stdout="", stderr=result.message, timed_out=False
            )
        check = CheckRun(command="self_dev_run_tests", result=command_result)
        session.checks_run = session.checks_run + (check,)
        return check

    def _attempt_repair(self, session: DevelopmentSession, failed: CommandResult) -> None:
        failure_text = (failed.stdout + "\n" + failed.stderr)[:_MAX_FAILURE_TEXT_CHARS]
        try:
            digest = self._build_repo_digest()
            fix_plan: DevelopmentPlan = self.planner.propose_fix(
                session.plan, failure_text, digest
            )
        except DevelopmentPlanningError as exc:
            session.log(f"Could not produce a repair fix: {exc}")
            return

        allowed_paths = session.plan.file_paths()
        edited = list(session.edited_files)
        for change in fix_plan.files_affected:
            if change.path not in allowed_paths:
                session.log(
                    f"Ignoring out-of-scope repair edit to '{change.path}' - not "
                    "in the originally approved plan."
                )
                continue
            if self._edit_one_file(session, change.path, change.description, fix_plan.objective):
                if change.path not in edited:
                    edited.append(change.path)
        session.edited_files = tuple(edited)

    # -- 10-11: inspect diff -> show summary ---------------------------------

    def _finalize(self, session: DevelopmentSession) -> None:
        diff_result = self.tools.execute("self_dev_git_diff")
        diff_text = diff_result.data["diff"] if diff_result.status is ToolResultStatus.SUCCESS else ""

        remaining_failures = ()
        if session.stage is SessionStage.CHECKS_FAILED and session.checks_run:
            last = session.checks_run[-1].result
            remaining_failures = ((last.stdout + "\n" + last.stderr)[:_MAX_FAILURE_TEXT_CHARS],)

        session.summary = DevelopmentSummary(
            files_changed=session.edited_files,
            change_summary=session.plan.objective if session.plan else "",
            checks_run=session.checks_run,
            remaining_failures=remaining_failures,
            git_diff_summary=diff_text,
            rollback_available=session.rollback_available(),
        )
        if session.stage is SessionStage.CHECKS_PASSED:
            session.stage = SessionStage.AWAITING_COMMIT_APPROVAL
            session.log("Ready for commit approval.")

    # -- 12: require a SEPARATE explicit approval before committing --------

    def approve_commit(
        self, session: DevelopmentSession, *, confirmed: bool | None, message: str
    ) -> DevelopmentSession:
        if session.stage is not SessionStage.AWAITING_COMMIT_APPROVAL:
            raise DevelopmentSessionError(
                f"Cannot commit from stage '{session.stage.value}' - checks must "
                "have passed and produced a summary first."
            )
        result = self.tools.execute(
            "self_dev_commit", params={"message": message}, confirmed=confirmed
        )
        if result.status is ToolResultStatus.SUCCESS:
            session.stage = SessionStage.COMMITTED
            session.log("Committed.")
            _delete_session_record(self.project_root, session.id)  # fully resolved
        elif result.status is ToolResultStatus.DENIED:
            session.stage = SessionStage.COMMIT_DECLINED
            session.log("Commit declined.")
            self._persist(session)
        else:
            session.log(f"Commit not completed: {result.message}")
            self._persist(session)
        return session

    # -- rollback -------------------------------------------------------------

    def rollback(self, session: DevelopmentSession, *, confirmed: bool | None) -> DevelopmentSession:
        if not session.rollback_available():
            raise DevelopmentSessionError(
                f"Nothing to roll back from stage '{session.stage.value}'."
            )
        result = self.tools.execute(
            "self_dev_rollback",
            params={
                "edited_files": ",".join(session.edited_files),
                "original_branch": session.original_branch,
            },
            confirmed=confirmed,
        )
        if result.status is ToolResultStatus.SUCCESS:
            session.stage = SessionStage.ROLLED_BACK
            session.log(f"Rolled back to '{session.original_branch}'.")
            _delete_session_record(self.project_root, session.id)  # fully resolved
        elif result.status is ToolResultStatus.DENIED:
            session.log("Rollback declined.")
            self._persist(session)
        else:
            session.log(f"Rollback not completed: {result.message}")
            self._persist(session)
        return session
