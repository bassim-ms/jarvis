"""
Tests for app/self_development/git.py: GitClient and its Tool wrappers,
against real, throwaway temporary git repositories - never the actual
Jarvis repository.

Covers focused-test categories:
10. branch creation
11. Git diff inspection
12. rollback behavior
18. permission boundaries (GIT_DIFF/BRANCH_CREATE automatic, COMMIT
    DANGEROUS, ROLLBACK CONFIRM)
"""

import subprocess

import pytest

from app.self_development.git import (
    CommitTool,
    CreateBranchTool,
    DEV_BRANCH_PREFIX,
    GitClient,
    GitDiffTool,
    GitError,
    RollbackTool,
    branch_name_for,
)
from app.tools.base import AUTOMATIC_PERMISSIONS, PermissionLevel, ToolResultStatus
from app.tools.registry import ToolRegistry


@pytest.fixture
def repo(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Tester"], cwd=tmp_path, check=True)
    (tmp_path / "a.txt").write_text("hello\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=tmp_path, check=True)
    return str(tmp_path)


# -- branch_name_for --------------------------------------------------------


def test_branch_name_for_is_namespaced():
    assert branch_name_for("abc123").startswith(DEV_BRANCH_PREFIX)


@pytest.mark.parametrize("slug", ["../escape", "a/b", "-flag", "", "a" * 100])
def test_branch_name_for_rejects_unsafe_slugs(slug):
    with pytest.raises(ValueError):
        branch_name_for(slug)


# -- GitClient basics --------------------------------------------------


def test_has_uncommitted_changes_false_on_a_clean_repo(repo):
    client = GitClient(repo)
    assert client.has_uncommitted_changes() is False


def test_has_uncommitted_changes_true_after_a_modification(repo):
    client = GitClient(repo)
    (__import__("pathlib").Path(repo) / "a.txt").write_text("changed\n", encoding="utf-8")
    assert client.has_uncommitted_changes() is True


def test_has_uncommitted_changes_true_for_an_untracked_file(repo):
    client = GitClient(repo)
    (__import__("pathlib").Path(repo) / "new.txt").write_text("new\n", encoding="utf-8")
    assert client.has_uncommitted_changes() is True


def test_current_branch_reports_the_real_branch(repo):
    client = GitClient(repo)
    branch = client.current_branch()
    assert branch in ("master", "main")  # git's own configured default


# -- branch creation (category 10) -------------------------------------


def test_create_branch_switches_to_a_new_branch(repo):
    client = GitClient(repo)
    original = client.current_branch()
    client.create_branch(branch_name_for("s1"))
    assert client.current_branch() == branch_name_for("s1")
    assert client.current_branch() != original


def test_create_branch_never_loses_history(repo):
    client = GitClient(repo)
    client.create_branch(branch_name_for("s1"))
    log = subprocess.run(
        ["git", "log", "--oneline"], cwd=repo, capture_output=True, text=True
    )
    assert "init" in log.stdout


def test_create_branch_fails_clearly_on_a_duplicate_name(repo):
    client = GitClient(repo)
    name = branch_name_for("dup")
    client.create_branch(name)
    with pytest.raises(GitError):
        # A branch with this name already exists - proves failures are
        # reported, not swallowed.
        client.create_branch(name)


# -- diff inspection (category 11) ---------------------------------------


def test_diff_is_empty_on_a_clean_tree(repo):
    client = GitClient(repo)
    assert client.diff() == ""


def test_diff_shows_the_real_change(repo):
    client = GitClient(repo)
    (__import__("pathlib").Path(repo) / "a.txt").write_text("hello\nworld\n", encoding="utf-8")
    diff = client.diff()
    assert "a.txt" in diff
    assert "+world" in diff


# -- rollback behavior (category 12) --------------------------------------


def test_rollback_restores_edited_files_and_switches_back(repo):
    client = GitClient(repo)
    original = client.current_branch()
    client.create_branch(branch_name_for("s2"))
    (__import__("pathlib").Path(repo) / "a.txt").write_text("modified\n", encoding="utf-8")
    assert client.has_uncommitted_changes() is True

    client.rollback(["a.txt"], original)

    assert client.current_branch() == original
    assert client.has_uncommitted_changes() is False
    content = (__import__("pathlib").Path(repo) / "a.txt").read_text(encoding="utf-8")
    assert content == "hello\n"


def test_rollback_with_no_edited_files_just_switches_back(repo):
    client = GitClient(repo)
    original = client.current_branch()
    client.create_branch(branch_name_for("s3"))
    client.rollback([], original)
    assert client.current_branch() == original


def test_rollback_removes_a_file_that_was_newly_created_this_session(repo):
    """A file this session CREATED (never existed in HEAD at all) has
    no HEAD version to restore - `restore_file` must remove it instead
    of failing, and rollback must still succeed end to end."""
    client = GitClient(repo)
    original = client.current_branch()
    client.create_branch(branch_name_for("s8"))
    new_file = __import__("pathlib").Path(repo) / "brand_new.txt"
    new_file.write_text("never committed\n", encoding="utf-8")
    assert client.has_uncommitted_changes() is True

    client.rollback(["brand_new.txt"], original)

    assert client.current_branch() == original
    assert not new_file.exists()
    assert client.has_uncommitted_changes() is False


def test_restore_file_removes_a_newly_created_file_directly(repo):
    client = GitClient(repo)
    client.create_branch(branch_name_for("s9"))
    new_file = __import__("pathlib").Path(repo) / "scratch.txt"
    new_file.write_text("temp\n", encoding="utf-8")

    client.restore_file("scratch.txt")

    assert not new_file.exists()


def test_restore_file_still_restores_a_genuinely_tracked_file(repo):
    """Regression: the "newly created" fallback must never fire for an
    ordinary tracked file - it should still restore from HEAD exactly
    as before this fix."""
    client = GitClient(repo)
    client.create_branch(branch_name_for("s10"))
    tracked_file = __import__("pathlib").Path(repo) / "a.txt"
    tracked_file.write_text("modified\n", encoding="utf-8")

    client.restore_file("a.txt")

    assert tracked_file.read_text(encoding="utf-8") == "hello\n"


# -- commit never pushes ---------------------------------------------------


def test_commit_stages_and_commits(repo):
    client = GitClient(repo)
    client.create_branch(branch_name_for("s4"))
    (__import__("pathlib").Path(repo) / "b.txt").write_text("new file\n", encoding="utf-8")
    client.commit("add b.txt")
    assert client.has_uncommitted_changes() is False
    log = subprocess.run(["git", "log", "--oneline"], cwd=repo, capture_output=True, text=True)
    assert "add b.txt" in log.stdout


def test_git_client_has_no_push_method_at_all():
    """Structural, not policy: 'never automatically push' is enforced
    by absence - there is no push capability to accidentally call."""
    assert not hasattr(GitClient, "push")


# -- Tool wrappers: permission boundaries (category 18) -------------------


def test_git_diff_and_create_branch_are_automatic():
    assert GitDiffTool.permission in AUTOMATIC_PERMISSIONS
    assert CreateBranchTool.permission in AUTOMATIC_PERMISSIONS


def test_commit_requires_dangerous_level_confirmation(repo):
    client = GitClient(repo)
    client.create_branch(branch_name_for("s5"))
    registry = ToolRegistry()
    registry.register(CommitTool(client))
    result = registry.execute("self_dev_commit", params={"message": "x"})
    assert result.status is ToolResultStatus.CONFIRMATION_REQUIRED
    assert result.permission is PermissionLevel.DANGEROUS


def test_rollback_tool_requires_confirmation(repo):
    client = GitClient(repo)
    original = client.current_branch()
    client.create_branch(branch_name_for("s6"))
    registry = ToolRegistry()
    registry.register(RollbackTool(client))
    result = registry.execute(
        "self_dev_rollback", params={"edited_files": "", "original_branch": original}
    )
    assert result.status is ToolResultStatus.CONFIRMATION_REQUIRED


def test_rollback_tool_runs_once_confirmed(repo):
    client = GitClient(repo)
    original = client.current_branch()
    client.create_branch(branch_name_for("s7"))
    (__import__("pathlib").Path(repo) / "a.txt").write_text("dirty\n", encoding="utf-8")
    registry = ToolRegistry()
    registry.register(RollbackTool(client))
    result = registry.execute(
        "self_dev_rollback",
        params={"edited_files": "a.txt", "original_branch": original},
        confirmed=True,
    )
    assert result.status is ToolResultStatus.SUCCESS
    assert client.current_branch() == original
