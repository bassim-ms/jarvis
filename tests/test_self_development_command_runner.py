"""
Tests for app/self_development/command_runner.py - the one subprocess
choke point, plus RunTestsTool/RunCommandTool.

Covers focused-test categories:
3. dangerous command rejection (at the tool level)
4. command timeout
5. command output/error capture
18. permission boundaries (TEST_RUN automatic vs COMMAND_RUN CONFIRM)
"""

import subprocess
import tempfile

from app.self_development.command_runner import RunCommandTool, RunTestsTool, run_command
from app.self_development.safety import UnsafeCommandError
from app.tools.base import AUTOMATIC_PERMISSIONS, PermissionLevel, ToolResultStatus
from app.tools.registry import ToolRegistry


def _init_repo(path: str) -> None:
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Tester"], cwd=path, check=True)


# -- run_command: output capture / exit code (category 5) -----------------


def test_run_command_captures_stdout_exit_code(tmp_path):
    _init_repo(str(tmp_path))
    result = run_command(["git", "status", "--porcelain"], cwd=str(tmp_path), timeout=10)
    assert result.exit_code == 0
    assert result.timed_out is False


def test_run_command_captures_a_nonzero_exit_code_and_stderr():
    # `git status` inside a non-repo directory (a fresh tmp dir) fails
    # with a real, captured stderr message and non-zero exit code.
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        result = run_command(["git", "status"], cwd=d, timeout=10)
    assert result.exit_code != 0
    assert result.stderr.strip() != ""
    assert result.succeeded is False


# -- run_command: timeout (category 4) -------------------------------------


def test_run_command_times_out_on_a_command_that_never_finishes():
    # `python -m http.server 0` (an allowed "python -m ..." invocation)
    # starts a server and runs forever until killed - a deliberately
    # unbounded process, bounded here by a short timeout instead.
    result = run_command(["python", "-m", "http.server", "0"], cwd=".", timeout=0.2)
    assert result.timed_out is True
    assert result.exit_code is None
    assert "timed out" in result.stderr.lower()


def test_run_command_never_raises_on_timeout():
    # No try/except needed to reach this assert - a timeout is reported
    # as an ordinary CommandResult, never an unhandled exception (V1's
    # own "no infinite processes, clear failure reporting" requirement).
    result = run_command(["python", "-m", "http.server", "0"], cwd=".", timeout=0.2)
    assert isinstance(result.timed_out, bool)
    assert result.timed_out is True


# -- run_command: dangerous command rejection ------------------------------


def test_run_command_rejects_unsafe_argv_before_executing_anything():
    try:
        run_command(["git", "reset", "--hard"], cwd=".", timeout=10)
        assert False, "should have raised"
    except UnsafeCommandError:
        pass


# -- RunTestsTool -----------------------------------------------------------


def test_run_tests_tool_is_automatic_safe_action():
    assert RunTestsTool.permission is PermissionLevel.SAFE_ACTION
    assert RunTestsTool.permission in AUTOMATIC_PERMISSIONS


def test_run_tests_tool_reports_success_and_failure(tmp_path):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    _init_repo(str(repo_dir))
    ok_tool = RunTestsTool(str(repo_dir), ["git", "status", "--porcelain"], timeout=10)
    result = ok_tool.execute()
    assert result.status is ToolResultStatus.SUCCESS
    assert result.data["succeeded"] is True

    # A genuinely separate, non-git directory (never a subdirectory of
    # `repo_dir` above - git would otherwise find that parent repo by
    # walking upward and the command would trivially succeed).
    with tempfile.TemporaryDirectory() as not_a_repo:
        fail_tool = RunTestsTool(not_a_repo, ["git", "status"], timeout=10)
        result = fail_tool.execute()
    assert result.status is ToolResultStatus.SUCCESS  # the TOOL ran fine
    assert result.data["succeeded"] is False  # the underlying command failed


def test_run_tests_tool_runs_automatically_through_the_registry_without_confirmation(tmp_path):
    _init_repo(str(tmp_path))
    registry = ToolRegistry()
    registry.register(RunTestsTool(str(tmp_path), ["git", "status", "--porcelain"], timeout=10))
    result = registry.execute("self_dev_run_tests")  # no `confirmed=` given
    assert result.status is ToolResultStatus.SUCCESS


# -- RunCommandTool: permission boundary + dangerous rejection -------------


def test_run_command_tool_requires_confirmation():
    assert RunCommandTool.permission is PermissionLevel.CONFIRM
    registry = ToolRegistry()
    registry.register(RunCommandTool(".", timeout=10))
    result = registry.execute("self_dev_run_command", params={"command": "git status"})
    assert result.status is ToolResultStatus.CONFIRMATION_REQUIRED


def test_run_command_tool_runs_once_confirmed(tmp_path):
    _init_repo(str(tmp_path))
    registry = ToolRegistry()
    registry.register(RunCommandTool(str(tmp_path), timeout=10))
    result = registry.execute(
        "self_dev_run_command", params={"command": "git status --porcelain"}, confirmed=True
    )
    assert result.status is ToolResultStatus.SUCCESS
    assert result.data["succeeded"] is True


def test_run_command_tool_rejects_a_dangerous_command_even_when_confirmed(tmp_path):
    """Confirmation can never override safety.validate_command's own
    rejection - the unsafe check runs BEFORE the tool would even
    attempt to execute anything."""
    registry = ToolRegistry()
    registry.register(RunCommandTool(str(tmp_path), timeout=10))
    result = registry.execute(
        "self_dev_run_command", params={"command": "git reset --hard"}, confirmed=True
    )
    assert result.status is ToolResultStatus.ERROR
    assert "rejected" in result.message.lower()


def test_run_command_tool_denied_when_declined(tmp_path):
    registry = ToolRegistry()
    registry.register(RunCommandTool(str(tmp_path), timeout=10))
    result = registry.execute(
        "self_dev_run_command", params={"command": "git status"}, confirmed=False
    )
    assert result.status is ToolResultStatus.DENIED
