"""
Tests for the Self-Development Engine's security-hardening pass:

1. SECURITY_EDIT - a distinct, non-self-grantable approval required for
   any protected security-critical path (app/self_development/**,
   app/tools/**, app/capability.py, app/core.py, app/config.py).
2. required_permissions as an actual enforcement mechanism (rejecting
   under-declared plans), not descriptive metadata.
3. Git branch-namespace enforcement moved into the Git layer itself.
4. Atomic file writes (crash-resistant, no partial overwrite).
5. Minimal crash-recovery metadata (detect, never auto-act on,
   abandoned sessions).

Every git-touching test here runs against a throwaway temporary
repository - never the real Jarvis repository.
"""

import json
import subprocess

import pytest

from app.self_development.base import (
    DevelopmentCapability,
    DevelopmentPlan,
    DevelopmentPlanError,
    DevelopmentSessionError,
    PlannedFileChange,
    SessionStage,
)
from app.self_development.editor import EditFileTool, SecurityEditFileTool
from app.self_development.engine import (
    _session_record_path,
    _session_records_dir,
    scan_for_abandoned_sessions,
)
from app.self_development.factory import create_default_self_development_engine
from app.self_development.git import GitClient
from app.self_development.planner import DevelopmentPlanner, DevelopmentPlanningError
from app.self_development.safety import InvalidBranchNameError, is_protected_system_path
from app.providers.base import AIProvider, ProviderError
from app.tools.base import PermissionLevel, ToolResultStatus
from app.tools.registry import ToolRegistry


def _init_repo(path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Tester"], cwd=path, check=True)
    (path / "app_self_development_stub.py").write_text("# stub\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=path, check=True)


# ===========================================================================
# 1. SECURITY_EDIT - protected paths, non-self-grantable
# ===========================================================================


def test_is_protected_system_path_matches_the_specified_set():
    for path in (
        "app/self_development/safety.py",
        "app/self_development/engine.py",
        "app/tools/base.py",
        "app/tools/registry.py",
        "app/capability.py",
        "app/core.py",
        "app/config.py",
    ):
        assert is_protected_system_path(path) is True


def test_is_protected_system_path_does_not_match_ordinary_files():
    for path in ("app/greet.py", "app/self_development_notes.md", "tests/test_core.py", "README.md"):
        assert is_protected_system_path(path) is False


def test_plan_touching_a_protected_path_without_security_edit_is_rejected_at_construction():
    with pytest.raises(DevelopmentPlanError):
        DevelopmentPlan(
            objective="x",
            files_affected=(PlannedFileChange(path="app/core.py", description="y"),),
            required_permissions=(DevelopmentCapability.CODE_EDIT,),  # missing SECURITY_EDIT
        )


def test_plan_touching_a_protected_path_with_security_edit_declared_constructs():
    plan = DevelopmentPlan(
        objective="x",
        files_affected=(PlannedFileChange(path="app/core.py", description="y"),),
        required_permissions=(DevelopmentCapability.CODE_EDIT, DevelopmentCapability.SECURITY_EDIT),
    )
    assert plan.protected_file_paths() == {"app/core.py"}


# -- ordinary CODE_EDIT approval cannot authorize SECURITY_EDIT -------------


def test_edit_file_tool_refuses_a_protected_path_outright(tmp_path):
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "core.py").write_text("# original\n", encoding="utf-8")
    tool = EditFileTool(str(tmp_path))

    result = tool.execute(path="app/core.py", content="# hacked\n")

    assert result.status is ToolResultStatus.ERROR
    assert "security_edit" in result.message.lower() or "security-critical" in result.message.lower()
    assert (tmp_path / "app" / "core.py").read_text(encoding="utf-8") == "# original\n"


def test_edit_file_tool_confirmed_true_still_refuses_a_protected_path(tmp_path):
    """Confirmation for ordinary CODE_EDIT can never satisfy
    SECURITY_EDIT - the refusal happens before any permission
    distinction even matters, inside the tool itself."""
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "core.py").write_text("# original\n", encoding="utf-8")
    registry = ToolRegistry()
    registry.register(EditFileTool(str(tmp_path)))

    result = registry.execute(
        "self_dev_edit_file",
        params={"path": "app/core.py", "content": "# hacked\n"},
        confirmed=True,
    )

    assert result.status is ToolResultStatus.ERROR
    assert (tmp_path / "app" / "core.py").read_text(encoding="utf-8") == "# original\n"


def test_security_edit_file_tool_refuses_a_non_protected_path(tmp_path):
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "widget.py").write_text("# original\n", encoding="utf-8")
    tool = SecurityEditFileTool(str(tmp_path))

    result = tool.execute(path="app/widget.py", content="# hacked\n")

    assert result.status is ToolResultStatus.ERROR
    assert (tmp_path / "app" / "widget.py").read_text(encoding="utf-8") == "# original\n"


def test_security_edit_file_tool_is_dangerous_permission():
    assert SecurityEditFileTool.permission is PermissionLevel.DANGEROUS


def test_security_edit_file_tool_requires_confirmation_through_the_registry(tmp_path):
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "core.py").write_text("# original\n", encoding="utf-8")
    registry = ToolRegistry()
    registry.register(SecurityEditFileTool(str(tmp_path)))

    result = registry.execute(
        "self_dev_security_edit_file", params={"path": "app/core.py", "content": "# new\n"}
    )

    assert result.status is ToolResultStatus.CONFIRMATION_REQUIRED
    assert (tmp_path / "app" / "core.py").read_text(encoding="utf-8") == "# original\n"


def test_security_edit_file_tool_writes_once_confirmed(tmp_path):
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "core.py").write_text("# original\n", encoding="utf-8")
    registry = ToolRegistry()
    registry.register(SecurityEditFileTool(str(tmp_path)))

    result = registry.execute(
        "self_dev_security_edit_file",
        params={"path": "app/core.py", "content": "# new\n"},
        confirmed=True,
    )

    assert result.status is ToolResultStatus.SUCCESS
    assert (tmp_path / "app" / "core.py").read_text(encoding="utf-8") == "# new\n"


def test_edit_tools_are_mutually_exclusive_by_path():
    """Every path is acceptable to exactly one of the two tools - never
    both, never neither."""
    ordinary = "app/widget.py"
    protected = "app/core.py"
    assert is_protected_system_path(ordinary) is False
    assert is_protected_system_path(protected) is True


# -- adversarial: self-grant attempts ---------------------------------------


def _protected_plan_payload(**overrides) -> dict:
    payload = {
        "objective": "modify core dispatch",
        "files_affected": [{"path": "app/core.py", "description": "change routing"}],
        "dependencies": [],
        "tests_to_run": [],
        "risks": [],
        "required_permissions": ["code_edit", "security_edit"],
    }
    payload.update(overrides)
    return payload


class _ScriptedProvider(AIProvider):
    def __init__(self, plan_payload=None, file_content="# new content\n", error=None):
        self.plan_payload = plan_payload or _protected_plan_payload()
        self.file_content = file_content
        self.error = error
        self.calls = 0

    def generate_reply(self, message: str) -> str:
        self.calls += 1
        if self.error is not None:
            raise self.error
        if "CURRENT CONTENT" in message:
            return self.file_content
        return json.dumps(self.plan_payload)


def _make_engine(repo_path, provider, **kwargs):
    defaults = dict(
        project_root=str(repo_path),
        max_repair_iterations=1,
        command_timeout_seconds=30.0,
        test_command="git status --porcelain",
    )
    defaults.update(kwargs)
    return create_default_self_development_engine(provider, **defaults)


def test_llm_declaring_security_edit_in_the_plan_does_not_self_grant_it(tmp_path):
    """Even though the plan's OWN required_permissions declares
    security_edit (correctly, satisfying the under-declaration check),
    that declaration is descriptive only - the engine still refuses to
    touch the protected file without a SEPARATE, explicit
    `approve_security_edit(confirmed=True)` call."""
    _init_repo(tmp_path)
    provider = _ScriptedProvider()
    engine = _make_engine(tmp_path, provider)

    session = engine.start("modify core dispatch")
    session = engine.approve_plan(session, approved=True)

    assert session.stage is SessionStage.AWAITING_SECURITY_APPROVAL
    assert session.edited_files == ()
    content = (tmp_path / "app_self_development_stub.py").read_text(encoding="utf-8")
    assert content == "# stub\n"  # untouched; app/core.py in the temp repo doesn't even exist yet
    assert not (tmp_path / "app").exists()


def test_approve_plan_never_calls_the_security_edit_tool(tmp_path):
    _init_repo(tmp_path)
    provider = _ScriptedProvider()
    engine = _make_engine(tmp_path, provider)
    calls = []
    original_execute = engine.tools.execute

    def _spy(name, *args, **kwargs):
        calls.append(name)
        return original_execute(name, *args, **kwargs)

    engine.tools.execute = _spy  # type: ignore[method-assign]

    session = engine.start("modify core dispatch")
    engine.approve_plan(session, approved=True)

    assert "self_dev_security_edit_file" not in calls


def test_declining_security_edit_leaves_protected_files_untouched(tmp_path):
    _init_repo(tmp_path)
    engine = _make_engine(tmp_path, _ScriptedProvider())
    session = engine.start("modify core dispatch")
    session = engine.approve_plan(session, approved=True)

    session = engine.approve_security_edit(session, confirmed=False)

    assert session.stage is SessionStage.SECURITY_EDIT_DECLINED
    assert session.edited_files == ()


def test_security_edit_with_no_confirmation_value_does_not_proceed(tmp_path):
    _init_repo(tmp_path)
    engine = _make_engine(tmp_path, _ScriptedProvider())
    session = engine.start("modify core dispatch")
    session = engine.approve_plan(session, approved=True)

    session = engine.approve_security_edit(session, confirmed=None)

    assert session.stage is SessionStage.AWAITING_SECURITY_APPROVAL  # unchanged
    assert session.edited_files == ()


def test_approving_security_edit_actually_edits_the_protected_file(tmp_path):
    _init_repo(tmp_path)
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "core.py").write_text("# old\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "add core.py"], cwd=tmp_path, check=True)

    engine = _make_engine(tmp_path, _ScriptedProvider())
    session = engine.start("modify core dispatch")
    session = engine.approve_plan(session, approved=True)
    session = engine.approve_security_edit(session, confirmed=True)

    assert "app/core.py" in session.edited_files
    # `propose_file_content`/`_strip_fence` strips surrounding whitespace
    # along with an optional code fence - no trailing newline survives.
    assert (tmp_path / "app" / "core.py").read_text(encoding="utf-8") == "# new content"


def test_cannot_approve_security_edit_from_the_wrong_stage(tmp_path):
    _init_repo(tmp_path)
    engine = _make_engine(tmp_path, _ScriptedProvider())
    session = engine.start("modify core dispatch")
    with pytest.raises(DevelopmentSessionError):
        engine.approve_security_edit(session, confirmed=True)  # plan not even approved yet


def test_cannot_approve_security_edit_twice(tmp_path):
    _init_repo(tmp_path)
    engine = _make_engine(tmp_path, _ScriptedProvider())
    session = engine.start("modify core dispatch")
    session = engine.approve_plan(session, approved=True)
    engine.approve_security_edit(session, confirmed=True)
    with pytest.raises(DevelopmentSessionError):
        engine.approve_security_edit(session, confirmed=True)


# -- regression: plans WITHOUT protected paths still work normally ---------


def test_unrelated_plan_without_protected_paths_still_works_normally(tmp_path):
    _init_repo(tmp_path)
    ordinary_plan = {
        "objective": "add a greeting helper",
        "files_affected": [{"path": "greet.py", "description": "add say_hello"}],
        "dependencies": [],
        "tests_to_run": [],
        "risks": [],
        "required_permissions": ["code_edit"],
    }
    provider = _ScriptedProvider(plan_payload=ordinary_plan, file_content="def say_hello():\n    return 'hi'\n")
    engine = _make_engine(tmp_path, provider)

    session = engine.start("add a greeting helper")
    session = engine.approve_plan(session, approved=True)

    # Never stopped for security approval - went straight through.
    assert session.stage is SessionStage.AWAITING_COMMIT_APPROVAL
    assert "greet.py" in session.edited_files


# ===========================================================================
# 2. required_permissions enforcement (planner-level adversarial tests)
# ===========================================================================


def test_planner_rejects_a_plan_under_declaring_code_edit():
    payload = {
        "objective": "x",
        "files_affected": [{"path": "app/greet.py", "description": "y"}],
        "dependencies": [], "tests_to_run": [], "risks": [],
        "required_permissions": [],  # under-declared
    }
    provider = _FakeSimpleProvider(json.dumps(payload))
    with pytest.raises(DevelopmentPlanningError):
        DevelopmentPlanner(provider).propose_plan("x", "digest")


def test_planner_rejects_a_plan_touching_a_protected_path_without_security_edit():
    payload = {
        "objective": "x",
        "files_affected": [{"path": "app/tools/registry.py", "description": "weaken checks"}],
        "dependencies": [], "tests_to_run": [], "risks": [],
        "required_permissions": ["code_edit"],  # missing security_edit
    }
    provider = _FakeSimpleProvider(json.dumps(payload))
    with pytest.raises(DevelopmentPlanningError):
        DevelopmentPlanner(provider).propose_plan("x", "digest")


def test_planner_accepts_a_correctly_declared_protected_plan():
    provider = _FakeSimpleProvider(json.dumps(_protected_plan_payload()))
    plan = DevelopmentPlanner(provider).propose_plan("x", "digest")
    assert plan.protected_file_paths() == {"app/core.py"}


class _FakeSimpleProvider(AIProvider):
    def __init__(self, reply: str):
        self.reply = reply

    def generate_reply(self, message: str) -> str:
        return self.reply


# ===========================================================================
# 3. Git branch namespace enforcement (in the Git layer itself)
# ===========================================================================


@pytest.fixture
def bare_repo(tmp_path):
    _init_repo(tmp_path)
    return tmp_path


def test_create_branch_rejects_a_non_namespaced_name_even_called_directly(bare_repo):
    client = GitClient(str(bare_repo))
    with pytest.raises(InvalidBranchNameError):
        client.create_branch("main")


def test_create_branch_rejects_a_flag_like_name(bare_repo):
    client = GitClient(str(bare_repo))
    with pytest.raises(InvalidBranchNameError):
        client.create_branch("-x")


def test_create_branch_rejects_a_traversal_like_name(bare_repo):
    client = GitClient(str(bare_repo))
    with pytest.raises(InvalidBranchNameError):
        client.create_branch("../evil")


def test_create_branch_rejects_a_name_without_the_prefix(bare_repo):
    client = GitClient(str(bare_repo))
    with pytest.raises(InvalidBranchNameError):
        client.create_branch("not-namespaced-at-all")


def test_create_branch_accepts_a_valid_namespaced_name(bare_repo):
    client = GitClient(str(bare_repo))
    client.create_branch("jarvis-selfdev/abc123")
    assert client.current_branch() == "jarvis-selfdev/abc123"


def test_create_branch_tool_also_rejects_an_invalid_name(bare_repo):
    """The Tool wrapper doesn't bypass GitClient's own enforcement -
    it's the SAME code path."""
    from app.self_development.git import CreateBranchTool

    client = GitClient(str(bare_repo))
    tool = CreateBranchTool(client)
    result = tool.execute(name="main")
    assert result.status is ToolResultStatus.ERROR
    assert client.current_branch() != "main" or client.current_branch() == "master"


def test_no_git_process_is_started_for_a_rejected_branch_name(bare_repo):
    """The name is validated BEFORE `git checkout -b` ever runs -
    proven by checking no new branch/ref was created at all."""
    client = GitClient(str(bare_repo))
    before = subprocess.run(["git", "branch"], cwd=bare_repo, capture_output=True, text=True).stdout
    with pytest.raises(InvalidBranchNameError):
        client.create_branch("main")
    after = subprocess.run(["git", "branch"], cwd=bare_repo, capture_output=True, text=True).stdout
    assert before == after


# ===========================================================================
# 4. Atomic file writes
# ===========================================================================


def test_edit_file_write_failure_leaves_the_original_file_untouched(tmp_path, monkeypatch):
    (tmp_path / "app").mkdir()
    target = tmp_path / "app" / "widget.py"
    target.write_text("ORIGINAL CONTENT\n", encoding="utf-8")

    import os as os_module

    original_replace = os_module.replace

    def _boom(*args, **kwargs):
        raise OSError("simulated crash during atomic replace")

    monkeypatch.setattr(os_module, "replace", _boom)

    tool = EditFileTool(str(tmp_path))
    result = tool.execute(path="app/widget.py", content="NEW CONTENT\n")

    assert result.status is ToolResultStatus.ERROR
    assert target.read_text(encoding="utf-8") == "ORIGINAL CONTENT\n"
    # No stray temp file left behind in the directory.
    leftovers = [p for p in (tmp_path / "app").iterdir() if p.name.startswith(".jarvis_selfdev_tmp_")]
    assert leftovers == []

    monkeypatch.setattr(os_module, "replace", original_replace)


def test_security_edit_file_write_failure_also_leaves_original_untouched(tmp_path, monkeypatch):
    (tmp_path / "app").mkdir()
    target = tmp_path / "app" / "core.py"
    target.write_text("ORIGINAL PROTECTED CONTENT\n", encoding="utf-8")

    import os as os_module

    original_replace = os_module.replace
    monkeypatch.setattr(os_module, "replace", lambda *a, **k: (_ for _ in ()).throw(OSError("boom")))

    tool = SecurityEditFileTool(str(tmp_path))
    result = tool.execute(path="app/core.py", content="HACKED\n")

    assert result.status is ToolResultStatus.ERROR
    assert target.read_text(encoding="utf-8") == "ORIGINAL PROTECTED CONTENT\n"

    monkeypatch.setattr(os_module, "replace", original_replace)


def test_edit_file_write_succeeds_normally_and_content_is_exact(tmp_path):
    (tmp_path / "app").mkdir()
    target = tmp_path / "app" / "widget.py"
    target.write_text("old\n", encoding="utf-8")

    tool = EditFileTool(str(tmp_path))
    result = tool.execute(path="app/widget.py", content="brand new content\n")

    assert result.status is ToolResultStatus.SUCCESS
    assert target.read_text(encoding="utf-8") == "brand new content\n"
    leftovers = [p for p in (tmp_path / "app").iterdir() if p.name.startswith(".jarvis_selfdev_tmp_")]
    assert leftovers == []


def test_edit_file_can_still_create_a_brand_new_file_atomically(tmp_path):
    (tmp_path / "app").mkdir()
    tool = EditFileTool(str(tmp_path))
    result = tool.execute(path="app/new_module.py", content="def f(): pass\n")
    assert result.status is ToolResultStatus.SUCCESS
    assert (tmp_path / "app" / "new_module.py").read_text(encoding="utf-8") == "def f(): pass\n"


# ===========================================================================
# 5. Minimal crash-recovery metadata
# ===========================================================================


def test_session_record_is_persisted_after_branch_creation(tmp_path):
    _init_repo(tmp_path)
    engine = _make_engine(
        tmp_path,
        _ScriptedProvider(
            plan_payload={
                "objective": "x", "files_affected": [], "dependencies": [],
                "tests_to_run": [], "risks": [], "required_permissions": [],
            }
        ),
    )
    session = engine.start("investigate only")
    session = engine.approve_plan(session, approved=True)

    record_path = _session_record_path(str(tmp_path), session.id)
    assert record_path.exists()
    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert record["id"] == session.id
    assert record["branch_name"] == session.dev_branch
    assert record["original_branch"] == session.original_branch


def test_session_record_is_deleted_after_commit(tmp_path):
    _init_repo(tmp_path)
    ordinary_plan = {
        "objective": "x", "files_affected": [{"path": "greet.py", "description": "y"}],
        "dependencies": [], "tests_to_run": [], "risks": [], "required_permissions": ["code_edit"],
    }
    provider = _ScriptedProvider(plan_payload=ordinary_plan, file_content="x = 1\n")
    engine = _make_engine(tmp_path, provider, test_command="git status --porcelain")
    session = engine.start("add greet")
    session = engine.approve_plan(session, approved=True)
    assert session.stage is SessionStage.AWAITING_COMMIT_APPROVAL
    assert _session_record_path(str(tmp_path), session.id).exists()

    engine.approve_commit(session, confirmed=True, message="add greet")

    assert not _session_record_path(str(tmp_path), session.id).exists()


def test_session_record_is_deleted_after_rollback(tmp_path):
    _init_repo(tmp_path)
    ordinary_plan = {
        "objective": "x", "files_affected": [{"path": "greet.py", "description": "y"}],
        "dependencies": [], "tests_to_run": [], "risks": [], "required_permissions": ["code_edit"],
    }
    provider = _ScriptedProvider(plan_payload=ordinary_plan, file_content="x = 1\n")
    engine = _make_engine(tmp_path, provider, test_command="git status --porcelain")
    session = engine.start("add greet")
    session = engine.approve_plan(session, approved=True)

    engine.rollback(session, confirmed=True)

    assert not _session_record_path(str(tmp_path), session.id).exists()


def test_scan_for_abandoned_sessions_detects_a_pending_session(tmp_path):
    _init_repo(tmp_path)
    engine = _make_engine(
        tmp_path,
        _ScriptedProvider(
            plan_payload={
                "objective": "x", "files_affected": [], "dependencies": [],
                "tests_to_run": [], "risks": [], "required_permissions": [],
            }
        ),
    )
    session = engine.start("investigate only")
    session = engine.approve_plan(session, approved=True)

    abandoned = engine.scan_for_abandoned_sessions()

    assert any(r["id"] == session.id for r in abandoned)


def test_scan_for_abandoned_sessions_ignores_committed_sessions(tmp_path):
    _init_repo(tmp_path)
    ordinary_plan = {
        "objective": "x", "files_affected": [{"path": "greet.py", "description": "y"}],
        "dependencies": [], "tests_to_run": [], "risks": [], "required_permissions": ["code_edit"],
    }
    provider = _ScriptedProvider(plan_payload=ordinary_plan, file_content="x = 1\n")
    engine = _make_engine(tmp_path, provider, test_command="git status --porcelain")
    session = engine.start("add greet")
    session = engine.approve_plan(session, approved=True)
    engine.approve_commit(session, confirmed=True, message="add greet")

    abandoned = engine.scan_for_abandoned_sessions()
    assert not any(r["id"] == session.id for r in abandoned)


def test_scan_for_abandoned_sessions_never_mutates_anything(tmp_path):
    """Purely informational - never deletes/resets/restores anything
    it finds, and never runs a git command of its own."""
    _init_repo(tmp_path)
    engine = _make_engine(
        tmp_path,
        _ScriptedProvider(
            plan_payload={
                "objective": "x", "files_affected": [], "dependencies": [],
                "tests_to_run": [], "risks": [], "required_permissions": [],
            }
        ),
    )
    session = engine.start("investigate only")
    session = engine.approve_plan(session, approved=True)

    before_branches = subprocess.run(
        ["git", "branch"], cwd=tmp_path, capture_output=True, text=True
    ).stdout
    before_records = sorted(p.name for p in _session_records_dir(str(tmp_path)).glob("*.json"))

    engine.scan_for_abandoned_sessions()
    engine.scan_for_abandoned_sessions()  # idempotent - calling twice changes nothing further

    after_branches = subprocess.run(
        ["git", "branch"], cwd=tmp_path, capture_output=True, text=True
    ).stdout
    after_records = sorted(p.name for p in _session_records_dir(str(tmp_path)).glob("*.json"))

    assert before_branches == after_branches
    assert before_records == after_records


def test_scan_for_abandoned_sessions_handles_a_missing_directory():
    assert scan_for_abandoned_sessions("/definitely/does/not/exist") == []


def test_scan_for_abandoned_sessions_skips_a_malformed_record(tmp_path):
    _init_repo(tmp_path)
    directory = _session_records_dir(str(tmp_path))
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "broken.json").write_text("not valid json", encoding="utf-8")

    # Never raises.
    result = scan_for_abandoned_sessions(str(tmp_path))
    assert result == []
