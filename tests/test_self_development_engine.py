"""
Tests for app/self_development/engine.py: SelfDevelopmentEngine - the
end-to-end session/approval/repair-loop orchestrator, against real,
throwaway temporary git repositories (never the actual Jarvis repo).

Covers focused-test categories:
8. unapproved edit rejection
9. unapproved commit rejection
10. branch creation (end-to-end, via the engine)
11. Git diff inspection (end-to-end, via the engine's own summary)
12. rollback behavior (end-to-end)
13. bounded repair loop
14. existing uncommitted changes are protected
15. successful development workflow
16. failed development workflow
17. confirmation boundaries
"""

import json
import subprocess

import pytest

from app.self_development.base import DevelopmentSessionError, SessionStage
from app.self_development.factory import create_default_self_development_engine
from app.providers.base import AIProvider


def _init_repo(path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Tester"], cwd=path, check=True)
    (path / "greet.py").write_text("# empty\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=path, check=True)


def _plan_payload(**overrides) -> dict:
    payload = {
        "objective": "Add a greeting function to greet.py",
        "files_affected": [{"path": "greet.py", "description": "add say_hello"}],
        "dependencies": [],
        "tests_to_run": ["git status --porcelain"],
        "risks": ["none"],
        "required_permissions": ["code_edit", "test_run"],
    }
    payload.update(overrides)
    return payload


class ScriptedProvider(AIProvider):
    """Routes by prompt shape: plan/fix requests get `plan_payload`
    (JSON); file-edit requests get a single whole-file find/replace
    edit (real current content -> `file_content`) - the same shape
    `propose_file_edits` (app/self_development/planner.py) now
    requires, built automatically from whatever the REAL prompt says
    the current content is, so this fixture never has to duplicate
    that text itself."""

    def __init__(self, plan_payload: dict | None = None, file_content: str = "def say_hello():\n    return 'hi'\n"):
        self.plan_payload = plan_payload or _plan_payload()
        self.file_content = file_content
        self.prompts: list[str] = []

    def generate_reply(self, message: str) -> str:
        self.prompts.append(message)
        if "CURRENT CONTENT" in message:
            current = message.split("CURRENT CONTENT (untrusted data):\n", 1)[1]
            return json.dumps({"edits": [{"old_text": current, "new_text": self.file_content}]})
        return json.dumps(self.plan_payload)


def _make_engine(repo_path, provider, *, max_repair_iterations=2, test_command="git status --porcelain"):
    return create_default_self_development_engine(
        provider,
        project_root=str(repo_path),
        max_repair_iterations=max_repair_iterations,
        command_timeout_seconds=30.0,
        test_command=test_command,
    )


@pytest.fixture
def repo(tmp_path):
    _init_repo(tmp_path)
    return tmp_path


# -- 14: existing uncommitted changes are protected --------------------


def test_start_refuses_when_the_tree_is_already_dirty(repo):
    (repo / "greet.py").write_text("# already being edited by the user\n", encoding="utf-8")
    engine = _make_engine(repo, ScriptedProvider())

    session = engine.start("add a greeting function")

    assert session.stage is SessionStage.BLOCKED_DIRTY_TREE
    assert session.dev_branch is None
    # Never touched the user's own uncommitted change.
    assert (repo / "greet.py").read_text(encoding="utf-8") == (
        "# already being edited by the user\n"
    )


def test_start_does_not_create_a_branch_when_blocked(repo):
    (repo / "greet.py").write_text("dirty\n", encoding="utf-8")
    engine = _make_engine(repo, ScriptedProvider())
    engine.start("add a greeting function")
    branch = subprocess.run(
        ["git", "branch"], cwd=repo, capture_output=True, text=True
    ).stdout
    assert "jarvis-selfdev" not in branch


# -- 15: successful development workflow ---------------------------------


def test_full_successful_workflow(repo):
    engine = _make_engine(repo, ScriptedProvider())

    session = engine.start("add a greeting function")
    assert session.stage is SessionStage.PLAN_PROPOSED
    assert session.plan.objective == "Add a greeting function to greet.py"

    session = engine.approve_plan(session, approved=True)
    assert session.stage is SessionStage.AWAITING_COMMIT_APPROVAL
    assert session.dev_branch is not None
    assert "greet.py" in session.edited_files
    # `propose_file_edits` applies the edit's own `new_text` verbatim -
    # unlike the old `propose_file_content`/`_strip_fence`, nothing
    # strips a trailing newline, so the fixture's own trailing "\n"
    # survives exactly.
    assert (repo / "greet.py").read_text(encoding="utf-8") == "def say_hello():\n    return 'hi'\n"
    assert session.summary is not None
    assert session.summary.rollback_available is True
    assert "greet.py" in session.summary.git_diff_summary

    session = engine.approve_commit(session, confirmed=True, message="add greeting function")
    assert session.stage is SessionStage.COMMITTED
    log = subprocess.run(["git", "log", "--oneline"], cwd=repo, capture_output=True, text=True)
    assert "add greeting function" in log.stdout
    status = subprocess.run(["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True)
    assert status.stdout.strip() == ""


def test_successful_workflow_never_pushes(repo):
    """No remote exists in this throwaway repo - if the engine ever
    tried to push, this would fail loudly; it never even attempts it
    (GitClient has no push method - see test_self_development_git.py)."""
    engine = _make_engine(repo, ScriptedProvider())
    session = engine.start("add a greeting function")
    session = engine.approve_plan(session, approved=True)
    engine.approve_commit(session, confirmed=True, message="x")
    # If this test reaches here without any network/remote error, no
    # push was ever attempted against a (nonexistent) remote.


# -- 8: unapproved edit rejection -------------------------------------


def test_declining_the_plan_never_creates_a_branch_or_edits_anything(repo):
    engine = _make_engine(repo, ScriptedProvider())
    session = engine.start("add a greeting function")

    session = engine.approve_plan(session, approved=False)

    assert session.stage is SessionStage.PLAN_REJECTED
    assert session.dev_branch is None
    assert session.edited_files == ()
    assert (repo / "greet.py").read_text(encoding="utf-8") == "# empty\n"
    branch = subprocess.run(["git", "branch"], cwd=repo, capture_output=True, text=True).stdout
    assert "jarvis-selfdev" not in branch


def test_cannot_approve_a_plan_twice(repo):
    engine = _make_engine(repo, ScriptedProvider())
    session = engine.start("add a greeting function")
    engine.approve_plan(session, approved=True)
    with pytest.raises(DevelopmentSessionError):
        engine.approve_plan(session, approved=True)


# -- 9/17: unapproved commit rejection / confirmation boundaries ---------


def test_declining_the_commit_never_commits(repo):
    engine = _make_engine(repo, ScriptedProvider())
    session = engine.start("add a greeting function")
    session = engine.approve_plan(session, approved=True)

    session = engine.approve_commit(session, confirmed=False, message="x")

    assert session.stage is SessionStage.COMMIT_DECLINED
    log = subprocess.run(["git", "log", "--oneline"], cwd=repo, capture_output=True, text=True)
    assert "x" not in log.stdout.splitlines()[0] if log.stdout else True
    status = subprocess.run(["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True)
    assert status.stdout.strip() != ""  # the edit is still there, uncommitted


def test_commit_without_any_confirmation_value_requires_confirmation(repo):
    """`confirmed=None` (never asked yet) reports CONFIRMATION_REQUIRED
    via the exact same ToolRegistry protocol every other CONFIRM/
    DANGEROUS tool in this codebase already uses - never silently
    committed."""
    engine = _make_engine(repo, ScriptedProvider())
    session = engine.start("add a greeting function")
    session = engine.approve_plan(session, approved=True)

    session = engine.approve_commit(session, confirmed=None, message="x")

    assert session.stage is SessionStage.AWAITING_COMMIT_APPROVAL  # unchanged
    status = subprocess.run(["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True)
    assert status.stdout.strip() != ""


def test_cannot_commit_before_checks_have_run(repo):
    engine = _make_engine(repo, ScriptedProvider())
    session = engine.start("add a greeting function")
    with pytest.raises(DevelopmentSessionError):
        engine.approve_commit(session, confirmed=True, message="x")


# -- 12: rollback behavior (end-to-end) ------------------------------


def test_rollback_restores_the_original_file_and_branch(repo):
    engine = _make_engine(repo, ScriptedProvider())
    session = engine.start("add a greeting function")
    session = engine.approve_plan(session, approved=True)
    original_branch = session.original_branch

    session = engine.rollback(session, confirmed=True)

    assert session.stage is SessionStage.ROLLED_BACK
    branch = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo, capture_output=True, text=True
    ).stdout.strip()
    assert branch == original_branch
    assert (repo / "greet.py").read_text(encoding="utf-8") == "# empty\n"


def test_rollback_requires_confirmation(repo):
    engine = _make_engine(repo, ScriptedProvider())
    session = engine.start("add a greeting function")
    session = engine.approve_plan(session, approved=True)

    session = engine.rollback(session, confirmed=None)

    assert session.stage is not SessionStage.ROLLED_BACK
    assert (repo / "greet.py").read_text(encoding="utf-8") != "# empty\n"


def test_rollback_unavailable_after_commit(repo):
    engine = _make_engine(repo, ScriptedProvider())
    session = engine.start("add a greeting function")
    session = engine.approve_plan(session, approved=True)
    engine.approve_commit(session, confirmed=True, message="x")

    assert session.rollback_available() is False
    with pytest.raises(DevelopmentSessionError):
        engine.rollback(session, confirmed=True)


# -- 13/16: bounded repair loop / failed development workflow -----------


def test_bounded_repair_loop_stops_at_the_configured_limit(repo):
    provider = ScriptedProvider(plan_payload=_plan_payload(tests_to_run=["exit 1"]))
    # A test command that always fails, however many times we retry.
    engine = _make_engine(repo, provider, max_repair_iterations=2, test_command="git rev-parse --verify refs/heads/does-not-exist")

    session = engine.start("add a greeting function")
    session = engine.approve_plan(session, approved=True)

    assert session.stage is SessionStage.CHECKS_FAILED
    assert session.repair_attempts == 2  # never more than the configured max
    assert len(session.checks_run) == 3  # initial run + 2 repair attempts
    assert session.summary is not None
    assert session.summary.remaining_failures != ()


def test_repair_loop_never_exceeds_max_repair_iterations_even_with_a_higher_limit(repo):
    provider = ScriptedProvider(plan_payload=_plan_payload())
    engine = _make_engine(
        repo, provider, max_repair_iterations=1,
        test_command="git rev-parse --verify refs/heads/does-not-exist",
    )
    session = engine.start("add a greeting function")
    session = engine.approve_plan(session, approved=True)
    assert session.repair_attempts == 1
    assert session.stage is SessionStage.CHECKS_FAILED


def test_repair_edit_outside_the_original_plan_scope_is_ignored(repo):
    """A repair fix proposing to touch a file OUTSIDE the originally
    approved plan is rejected/ignored by the engine, never applied -
    "never grant itself permissions beyond the approved plan."""
    fix_payload = _plan_payload(
        objective="fix by touching an unrelated file",
        files_affected=[{"path": "unexpected.py", "description": "sneaky"}],
    )

    class TwoPhaseProvider(AIProvider):
        def __init__(self):
            self.calls = 0

        def generate_reply(self, message: str) -> str:
            self.calls += 1
            if "CURRENT CONTENT" in message:
                current = message.split("CURRENT CONTENT (untrusted data):\n", 1)[1]
                return json.dumps(
                    {"edits": [{"old_text": current, "new_text": "def say_hello():\n    return 'hi'\n"}]}
                )
            if "BOUNDED FOLLOW-UP" in message:
                return json.dumps(fix_payload)
            return json.dumps(_plan_payload(tests_to_run=["exit 1"]))

    provider = TwoPhaseProvider()
    engine = _make_engine(
        repo, provider, max_repair_iterations=1,
        test_command="git rev-parse --verify refs/heads/does-not-exist",
    )
    session = engine.start("add a greeting function")
    session = engine.approve_plan(session, approved=True)

    assert "unexpected.py" not in session.edited_files
    assert not (repo / "unexpected.py").exists()


def test_failed_workflow_still_produces_a_summary(repo):
    provider = ScriptedProvider(plan_payload=_plan_payload())
    engine = _make_engine(
        repo, provider, max_repair_iterations=1,
        test_command="git rev-parse --verify refs/heads/does-not-exist",
    )
    session = engine.start("add a greeting function")
    session = engine.approve_plan(session, approved=True)

    assert session.stage is SessionStage.CHECKS_FAILED
    assert session.summary is not None
    assert session.summary.files_changed
    assert session.summary.remaining_failures != ()
    # A failed workflow can still be rolled back - nothing was committed.
    assert session.rollback_available() is True


# -- planning proposal never touches the working tree --------------------


def test_start_never_edits_anything_before_approval(repo):
    engine = _make_engine(repo, ScriptedProvider())
    engine.start("add a greeting function")
    assert (repo / "greet.py").read_text(encoding="utf-8") == "# empty\n"
    status = subprocess.run(["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True)
    assert status.stdout.strip() == ""


def test_start_uses_the_real_repo_listing_as_context(repo):
    # Committed (not left uncommitted) - otherwise the dirty-tree
    # precondition would block `start()` before it ever builds a
    # digest or calls the planner at all (see the "14" tests above).
    (repo / "extra.py").write_text("# extra\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "add extra.py"], cwd=repo, check=True)

    provider = ScriptedProvider()
    engine = _make_engine(repo, provider)
    engine.start("add a greeting function")
    plan_prompt = next(p for p in provider.prompts if "DEVELOPMENT REQUEST" in p)
    assert "extra.py" in plan_prompt
    assert "greet.py" in plan_prompt
