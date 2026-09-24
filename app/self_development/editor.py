"""
File-write tools (V1) - the ONLY two places this whole subsystem ever
writes to the filesystem, both built on the same atomic-write helper.

EditFileTool (CODE_EDIT, CONFIRM) writes ordinary files - it REFUSES a
protected security-critical path (app/self_development/safety.py:
is_protected_system_path) outright, telling the caller SECURITY_EDIT is
needed instead. SecurityEditFileTool (SECURITY_EDIT, DANGEROUS) is the
only tool that may write a protected path, and refuses anything that
ISN'T one - the two are mutually exclusive by design, so a plan's
required_permissions/file list unambiguously says which approval tier
each file needed (see app/self_development/base.py:DevelopmentPlan's
own under-declaration check, and engine.py:SelfDevelopmentEngine's own
module docstring for why ordinary plan approval can never also
authorize SECURITY_EDIT).

Writes are atomic: content is written to a temporary file in the SAME
directory (so the final rename is on one filesystem), flushed and
fsync'd, then swapped into place with `os.replace` - atomic on both
POSIX and Windows. A failure at any point before that swap leaves the
ORIGINAL file completely untouched; the temporary file is cleaned up.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

from app.self_development.safety import (
    PathSafetyError,
    PathTraversalError,
    is_protected_system_path,
    is_secret_path,
    resolve_within_root,
)
from app.tools.base import PermissionLevel, Tool, ToolParameter, ToolResult

_MAX_CONTENT_CHARS = 500_000


def _atomic_write(resolved: Path, content: str) -> None:
    """Write `content` to `resolved` atomically - a temp file in the
    same directory, flushed and fsync'd, then `os.replace`d into place.
    Raises OSError on any failure; the ORIGINAL file (if any) is never
    touched until the final, atomic swap succeeds, and the temp file is
    always cleaned up on failure."""
    resolved.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path_str = tempfile.mkstemp(
        dir=str(resolved.parent), prefix=".jarvis_selfdev_tmp_"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        if resolved.exists():
            try:
                shutil.copymode(str(resolved), tmp_path_str)
            except OSError:
                pass  # best-effort only ("preserve permissions where practical")
        os.replace(tmp_path_str, str(resolved))
    except Exception:
        try:
            os.remove(tmp_path_str)
        except OSError:
            pass
        raise


def _resolve_for_write(project_root: str, path: str) -> tuple[Path | None, ToolResult | None]:
    """Shared path validation for both tools below - returns
    `(resolved_path, None)` on success or `(None, error_result)` on
    failure, so each tool's own `execute` can `return error` in one
    line without duplicating this logic."""
    try:
        resolved = resolve_within_root(project_root, path)
    except (PathSafetyError, PathTraversalError) as exc:
        return None, ToolResult.error(str(exc))
    if resolved.is_dir():
        return None, ToolResult.error(f"'{path}' is a directory, not a file.")
    return resolved, None


class EditFileTool(Tool):
    """CODE_EDIT - ordinary files only. Refuses a secret-named path
    (never reads OR writes one) and refuses a PROTECTED system path
    (that needs SecurityEditFileTool/SECURITY_EDIT instead - never
    silently downgraded or allowed through this weaker tool)."""

    name = "self_dev_edit_file"
    description = "Create or overwrite one ordinary repository file with the given content."
    permission = PermissionLevel.CONFIRM

    def __init__(self, project_root: str):
        self.project_root = project_root

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(name="path", type="string", description="Repo-relative file path."),
            ToolParameter(name="content", type="string", description="The file's full new content."),
        ]

    def execute(self, path: str, content: str) -> ToolResult:
        if is_secret_path(path):
            return ToolResult.error(
                f"'{path}' is a protected secret/config file and can never be edited "
                "by the self-development engine."
            )
        if is_protected_system_path(path):
            return ToolResult.error(
                f"'{path}' is a protected security-critical system path - it requires "
                "SECURITY_EDIT approval (self_dev_security_edit_file), never ordinary "
                "CODE_EDIT."
            )
        if len(content) > _MAX_CONTENT_CHARS:
            return ToolResult.error(
                f"Refusing to write '{path}': content exceeds the "
                f"{_MAX_CONTENT_CHARS}-character bound."
            )
        resolved, error = _resolve_for_write(self.project_root, path)
        if error is not None:
            return error

        try:
            _atomic_write(resolved, content)
        except OSError as exc:
            return ToolResult.error(f"Could not write '{path}': {exc}")

        return ToolResult.success(data={"path": path}, message=f"Wrote '{path}'.")


class SecurityEditFileTool(Tool):
    """SECURITY_EDIT - DANGEROUS, the strongest permission tier this
    codebase has, and a SEPARATE approval from ordinary CODE_EDIT (see
    app/self_development/engine.py:SelfDevelopmentEngine.
    approve_security_edit - never satisfied by `approve_plan` alone).
    The ONLY tool that may write a protected security-critical path
    (app/self_development/, app/tools/, app/capability.py, app/core.py,
    app/config.py) - and refuses anything that ISN'T one, so this tool
    is never usable as a generic bypass for ordinary edits either."""

    name = "self_dev_security_edit_file"
    description = (
        "Create or overwrite one PROTECTED security-critical repository file - "
        "requires SECURITY_EDIT approval, separate from ordinary CODE_EDIT."
    )
    permission = PermissionLevel.DANGEROUS

    def __init__(self, project_root: str):
        self.project_root = project_root

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(name="path", type="string", description="Repo-relative file path."),
            ToolParameter(name="content", type="string", description="The file's full new content."),
        ]

    def execute(self, path: str, content: str) -> ToolResult:
        if not is_protected_system_path(path):
            return ToolResult.error(
                f"'{path}' is not a protected security-critical path - use the "
                "ordinary self_dev_edit_file (CODE_EDIT) tool instead."
            )
        if is_secret_path(path):
            return ToolResult.error(
                f"'{path}' is a protected secret/config file and can never be edited "
                "by the self-development engine."
            )
        if len(content) > _MAX_CONTENT_CHARS:
            return ToolResult.error(
                f"Refusing to write '{path}': content exceeds the "
                f"{_MAX_CONTENT_CHARS}-character bound."
            )
        resolved, error = _resolve_for_write(self.project_root, path)
        if error is not None:
            return error

        try:
            _atomic_write(resolved, content)
        except OSError as exc:
            return ToolResult.error(f"Could not write '{path}': {exc}")

        return ToolResult.success(data={"path": path}, message=f"Wrote protected file '{path}'.")
