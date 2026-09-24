"""
The ONE place a subprocess is ever started anywhere in the
Self-Development Engine (V1) - `run_command` below. Every other module
in this package that needs to run something (git.py, RunTestsTool,
RunCommandTool) calls this, never `subprocess` directly - a single,
auditable choke point.

Deliberately NOT "LLM generates a shell command -> execute it" (V1's
own explicit non-goal): `run_command` takes an argv LIST and always
calls `subprocess.run(..., shell=False)`. Shell metacharacters
(`&&`, `;`, `|`, backticks, redirection) are therefore inert - they're
never interpreted as anything but a literal argument - not merely
filtered by `app.self_development.safety`, though that module's
`validate_command` is still called first, on every invocation, as a
second independent gate (allowlisted executables/subcommands, no
destructive flags).

`RunCommandTool.execute` accepts a single `command` string (the only
shape ToolParameter's plain-string type system supports - see
app/tools/base.py) and tokenizes it with `shlex.split` - quoting-aware
splitting only, never shell evaluation - before validating and running
it exactly the same way.
"""

from __future__ import annotations

import shlex
import subprocess
import time

from app.self_development.base import CommandResult
from app.self_development.safety import UnsafeCommandError, validate_command
from app.tools.base import PermissionLevel, Tool, ToolParameter, ToolResult

#: Hard ceiling on how much of a command's stdout/stderr this engine
#: ever holds in memory or reports - never an unbounded process output
#: (V1's own "clear failure reporting," not "dump everything").
_MAX_OUTPUT_CHARS = 20_000


def _truncate(text: str) -> str:
    if len(text) <= _MAX_OUTPUT_CHARS:
        return text
    return text[:_MAX_OUTPUT_CHARS] + "\n... (truncated)"


def run_command(argv: list[str], *, cwd: str, timeout: float) -> CommandResult:
    """Validate `argv` (raises UnsafeCommandError - the caller decides
    whether that's a ToolResult.error or something stronger) then run
    it bounded: `cwd` fixed to the project root (or a path already
    validated to be inside it - callers never pass an arbitrary cwd),
    `timeout` enforced by `subprocess.run` itself, output captured and
    truncated, exit code always reported. A timeout is reported as a
    CommandResult (`timed_out=True`), never left as an unhandled
    exception - "no infinite processes, clear failure reporting" is
    V1's own explicit requirement."""
    validate_command(argv)

    try:
        completed = subprocess.run(
            argv,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=False,
        )
        return CommandResult(
            argv=tuple(argv),
            exit_code=completed.returncode,
            stdout=_truncate(completed.stdout or ""),
            stderr=_truncate(completed.stderr or ""),
            timed_out=False,
        )
    except subprocess.TimeoutExpired as exc:
        return CommandResult(
            argv=tuple(argv),
            exit_code=None,
            stdout=_truncate(exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")),
            stderr=_truncate(
                (exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or ""))
                + f"\nTimed out after {timeout}s."
            ),
            timed_out=True,
        )
    except FileNotFoundError as exc:
        return CommandResult(
            argv=tuple(argv), exit_code=None, stdout="", stderr=str(exc), timed_out=False
        )


class RunTestsTool(Tool):
    """TEST_RUN - runs the project's own test suite. SAFE_ACTION (V1's
    own spec: automatic, no confirmation) - a test run is read-only
    with respect to the codebase; it never edits or commits anything.
    Always the same fixed command (`test_command`, given by the
    caller/factory - see app/config.py:self_dev_test_command) - never
    an AI- or user-supplied command, so this tool alone can never be
    steered into running something else (see RunCommandTool for the
    general, CONFIRM-gated case)."""

    name = "self_dev_run_tests"
    description = "Run the project's test suite."
    permission = PermissionLevel.SAFE_ACTION

    def __init__(self, project_root: str, test_command: list[str], timeout: float):
        self.project_root = project_root
        self.test_command = list(test_command)
        self.timeout = timeout

    def execute(self) -> ToolResult:
        try:
            result = run_command(self.test_command, cwd=self.project_root, timeout=self.timeout)
        except UnsafeCommandError as exc:  # pragma: no cover - only if misconfigured
            return ToolResult.error(f"Configured test command is unsafe: {exc}")
        return ToolResult.success(
            data={
                "exit_code": result.exit_code,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "timed_out": result.timed_out,
                "succeeded": result.succeeded,
            },
            message="Tests passed." if result.succeeded else "Tests failed.",
        )


class RunCommandTool(Tool):
    """COMMAND_RUN - runs one explicitly allow-listed development
    command (git inspection, Python tooling, tests - see
    app/self_development/safety.py). CONFIRM: "controlled" per V1's own
    spec - every invocation needs approval, even an otherwise-allowed
    one, and `safety.validate_command` is checked BEFORE that
    confirmation gate is ever reached, so a disallowed/dangerous
    command is rejected outright and can never be "confirmed" into
    running (see ToolRegistry.execute, app/tools/registry.py - this
    tool's own `execute` runs only once the Registry has already
    cleared permission/confirmation)."""

    name = "self_dev_run_command"
    description = (
        "Run one explicitly allow-listed development command (git inspection, "
        "Python tooling, or tests) inside the project root."
    )
    permission = PermissionLevel.CONFIRM

    def __init__(self, project_root: str, timeout: float):
        self.project_root = project_root
        self.timeout = timeout

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="command",
                type="string",
                description="The command to run, e.g. 'git status' or 'pytest -q'.",
            )
        ]

    def execute(self, command: str) -> ToolResult:
        try:
            argv = shlex.split(command)
        except ValueError as exc:
            return ToolResult.error(f"Could not parse command: {exc}")

        try:
            result = run_command(argv, cwd=self.project_root, timeout=self.timeout)
        except UnsafeCommandError as exc:
            return ToolResult.error(f"Command rejected: {exc}")

        return ToolResult.success(
            data={
                "exit_code": result.exit_code,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "timed_out": result.timed_out,
                "succeeded": result.succeeded,
            },
            message="Command succeeded." if result.succeeded else "Command failed.",
        )
