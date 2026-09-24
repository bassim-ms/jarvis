"""
GitClient (V1) - the only place this engine talks to Git, built
entirely on app.self_development.command_runner.run_command (never a
second subprocess call site, never a `git` library dependency).

Every method here maps to exactly one safety.py-allowed command shape
(git.py never constructs an argv that `validate_command` wouldn't
already accept on its own) - see safety.py's own docstring for the
full allowlist. In particular:

- `create_branch` is the ONLY way this engine creates a branch -
  `git checkout -b <name>`, always from whatever HEAD already is.
- There is no `reset`/`clean` anywhere in this file, or anywhere in
  this whole package - see `rollback`'s own docstring for why a
  targeted per-file restore is used instead.
- `commit` never pushes - there is no push method in this class at
  all, so "never automatically push to GitHub" (V1's own explicit
  requirement) is enforced by absence, not merely by policy.
"""

from __future__ import annotations

import re

from app.self_development.base import CommandResult
from app.self_development.command_runner import run_command
from app.self_development.safety import (
    InvalidBranchNameError,
    PathSafetyError,
    PathTraversalError,
    resolve_within_root,
    validate_branch_name,
)
from app.tools.base import PermissionLevel, Tool, ToolParameter, ToolResult

#: A dev branch name this engine creates - namespaced so it's always
#: obviously distinguishable from a user's own branches, and narrow
#: enough that a caller can't smuggle a path-like or flag-like name
#: through (see `_BRANCH_NAME_PATTERN`).
DEV_BRANCH_PREFIX = "jarvis-selfdev/"

_BRANCH_SLUG_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,60}$")


class GitError(RuntimeError):
    """Raised when a Git operation this engine attempted didn't
    succeed - the caller (engine.py) decides how to report this; never
    silently ignored."""


def branch_name_for(slug: str) -> str:
    """Build a namespaced dev-branch name from a short slug (e.g. a
    session id) - raises ValueError for anything that isn't a plain,
    short, safe token, so a branch name can never carry a path
    separator, a leading `-` (which `git checkout -b` would otherwise
    interpret as a flag), or anything else unexpected."""
    if not _BRANCH_SLUG_PATTERN.match(slug):
        raise ValueError(f"Unsafe branch slug: {slug!r}")
    return f"{DEV_BRANCH_PREFIX}{slug}"


class GitClient:
    """Thin, bounded wrapper - `project_root` is passed through as
    every command's `cwd` unchanged (never a caller-supplied path),
    and `timeout` bounds every single git invocation."""

    def __init__(self, project_root: str, timeout: float = 30.0):
        self.project_root = project_root
        self.timeout = timeout

    def _run(self, argv: list[str]) -> CommandResult:
        return run_command(argv, cwd=self.project_root, timeout=self.timeout)

    def status_porcelain(self) -> str:
        """`git status --porcelain` - machine-readable and
        locale-independent (never git's own translated English
        messages), so "is the tree dirty" never depends on parsing a
        human-facing sentence. Empty output means a clean tree."""
        result = self._run(["git", "status", "--porcelain"])
        if not result.succeeded:
            raise GitError(f"'git status' failed: {result.stderr}")
        return result.stdout

    def has_uncommitted_changes(self) -> bool:
        return bool(self.status_porcelain().strip())

    def current_branch(self) -> str:
        result = self._run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
        if not result.succeeded:
            raise GitError(f"Could not determine current branch: {result.stderr}")
        return result.stdout.strip()

    def diff(self) -> str:
        """Unstaged working-tree diff - since this engine only ever
        stages (`git add -A`) immediately before a commit, every
        change the session has made so far shows up here throughout
        editing/checking, right up until the moment it's committed."""
        result = self._run(["git", "diff"])
        if not result.succeeded:
            raise GitError(f"'git diff' failed: {result.stderr}")
        return result.stdout

    def create_branch(self, name: str) -> None:
        """Enforced HERE, in the Git layer itself - `validate_branch_name`
        (app/self_development/safety.py) is called unconditionally,
        regardless of whether the caller already went through
        `branch_name_for`. A caller invoking this method directly with
        an unnamespaced or unsafe name (e.g. "main", "-x", "../evil")
        is rejected before any `git` process is even started - never
        only by caller discipline."""
        validate_branch_name(name)  # raises InvalidBranchNameError
        result = self._run(["git", "checkout", "-b", name])
        if not result.succeeded:
            raise GitError(f"Could not create branch '{name}': {result.stderr}")

    def switch_branch(self, name: str) -> None:
        """Switch to an EXISTING branch by a bare name - only ever
        called once the working tree is already clean (see
        `rollback`), so this can never carry uncommitted changes onto
        the target branch."""
        result = self._run(["git", "checkout", name])
        if not result.succeeded:
            raise GitError(f"Could not switch to branch '{name}': {result.stderr}")

    def restore_file(self, relative_path: str) -> None:
        """Discard uncommitted changes to exactly one tracked file by
        restoring it from HEAD (`git checkout HEAD -- <path>`) - never
        a wildcard, never "all files". Used only to undo this engine's
        OWN edits during rollback; a file this session never touched
        is never named here.

        A file this session CREATED (never existed in HEAD at all) has
        no HEAD version to restore - `git checkout HEAD -- <path>`
        fails with a "pathspec ... did not match" error for exactly
        this case, distinguishable from any other failure. "Restoring"
        such a file correctly means removing it (undoing its creation),
        done here as a plain, path-validated filesystem delete - never
        `git clean`, which would touch every untracked file in the
        tree, not just this one."""
        result = self._run(["git", "checkout", "HEAD", "--", relative_path])
        if result.succeeded:
            return
        if "did not match any file" in result.stderr:
            self._remove_newly_created_file(relative_path)
            return
        raise GitError(f"Could not restore '{relative_path}': {result.stderr}")

    def _remove_newly_created_file(self, relative_path: str) -> None:
        try:
            resolved = resolve_within_root(self.project_root, relative_path)
        except (PathSafetyError, PathTraversalError) as exc:
            raise GitError(f"Refusing to remove unsafe path '{relative_path}': {exc}") from exc
        try:
            resolved.unlink(missing_ok=True)
        except OSError as exc:
            raise GitError(f"Could not remove newly created file '{relative_path}': {exc}") from exc

    def commit(self, message: str) -> None:
        add_result = self._run(["git", "add", "-A"])
        if not add_result.succeeded:
            raise GitError(f"'git add -A' failed: {add_result.stderr}")
        commit_result = self._run(["git", "commit", "-m", message])
        if not commit_result.succeeded:
            raise GitError(f"'git commit' failed: {commit_result.stderr}")

    def rollback(self, edited_files: list[str], original_branch: str) -> None:
        """Undo every uncommitted edit this session made and switch
        back to `original_branch` - never `git reset --hard`/`git
        clean` (absent from this whole package). Each file this
        session actually edited is individually restored from HEAD
        first, so by the time `switch_branch` runs, the dev branch's
        working tree is guaranteed clean and the switch can never carry
        anything onto `original_branch`."""
        for path in edited_files:
            self.restore_file(path)
        self.switch_branch(original_branch)


# ---------------------------------------------------------------------------
# Tools - the ONLY way app/self_development/engine.py ever reaches
# GitClient. Each maps to exactly one DevelopmentCapability/
# PermissionLevel pair (app/self_development/base.py) - the engine
# calls these through its own ToolRegistry, never GitClient directly,
# so every git action (like every file edit) goes through the same
# permission/confirmation gate as any other tool in this codebase.
# ---------------------------------------------------------------------------


class GitDiffTool(Tool):
    """GIT_DIFF - READ (automatic)."""

    name = "self_dev_git_diff"
    description = "Show the current unstaged working-tree diff."
    permission = PermissionLevel.READ

    def __init__(self, client: GitClient):
        self.client = client

    def execute(self) -> ToolResult:
        try:
            return ToolResult.success(data={"diff": self.client.diff()})
        except GitError as exc:
            return ToolResult.error(str(exc))


class CreateBranchTool(Tool):
    """BRANCH_CREATE - SAFE_ACTION (automatic). Never destroys
    anything - a new branch is always created from the current HEAD."""

    name = "self_dev_create_branch"
    description = "Create and switch to a new, namespaced development branch."
    permission = PermissionLevel.SAFE_ACTION

    def __init__(self, client: GitClient):
        self.client = client

    @property
    def parameters(self) -> list[ToolParameter]:
        return [ToolParameter(name="name", type="string", description="The new branch's name.")]

    def execute(self, name: str) -> ToolResult:
        try:
            self.client.create_branch(name)
        except (GitError, ValueError) as exc:
            return ToolResult.error(str(exc))
        return ToolResult.success(data={"branch": name}, message=f"Created branch '{name}'.")


class CommitTool(Tool):
    """COMMIT - DANGEROUS (V1's own spec: requires EXPLICIT
    confirmation, the strongest tier this codebase has). Never pushes -
    GitClient has no push method at all."""

    name = "self_dev_commit"
    description = "Stage and commit every change on the current development branch."
    permission = PermissionLevel.DANGEROUS

    def __init__(self, client: GitClient):
        self.client = client

    @property
    def parameters(self) -> list[ToolParameter]:
        return [ToolParameter(name="message", type="string", description="The commit message.")]

    def execute(self, message: str) -> ToolResult:
        try:
            self.client.commit(message)
        except GitError as exc:
            return ToolResult.error(str(exc))
        return ToolResult.success(message="Committed.")


class RollbackTool(Tool):
    """ROLLBACK - CONFIRM. Restores every file named in `edited_files`
    (comma-separated - ToolParameter has no array type, see
    app/tools/base.py) from HEAD, then switches back to
    `original_branch`. Never `git reset --hard`/`git clean`."""

    name = "self_dev_rollback"
    description = "Discard this session's uncommitted edits and switch back to the original branch."
    permission = PermissionLevel.CONFIRM

    def __init__(self, client: GitClient):
        self.client = client

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="edited_files",
                type="string",
                description="Comma-separated repo-relative paths this session edited.",
            ),
            ToolParameter(
                name="original_branch", type="string", description="The branch to switch back to."
            ),
        ]

    def execute(self, edited_files: str, original_branch: str) -> ToolResult:
        paths = [p.strip() for p in edited_files.split(",") if p.strip()]
        try:
            self.client.rollback(paths, original_branch)
        except GitError as exc:
            return ToolResult.error(str(exc))
        return ToolResult.success(message=f"Rolled back to '{original_branch}'.")
