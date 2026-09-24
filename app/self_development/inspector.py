"""
Read-only repository tools (V1) - CODE_READ, PermissionLevel.READ
(automatic). Every path this module ever touches is resolved with
app.self_development.safety.resolve_within_root first - there is no
other way any of these tools reach the filesystem, so "V1 may operate
only inside the configured project root" is enforced at this single
choke point, not by convention in each tool.

A secret-named file (app.self_development.safety.is_secret_path) is
never read - `ReadFileTool` reports that it exists and is protected,
never its content. `ListRepoFilesTool`/`SearchRepoTool` still show that
such a file EXISTS (V1's own spec: "may detect that a secret/config
file exists without revealing its values") but never its content
either.
"""

from __future__ import annotations

import os

from app.self_development.safety import (
    EXCLUDED_DIRECTORY_NAMES,
    PathSafetyError,
    PathTraversalError,
    is_secret_path,
    resolve_within_root,
)
from app.tools.base import PermissionLevel, Tool, ToolParameter, ToolResult

#: Hard bounds - never an unbounded directory walk or file read (the
#: same "clear, bounded reporting" discipline command_runner.py's own
#: output truncation already applies).
_MAX_LISTED_FILES = 500
_MAX_READ_CHARS = 200_000
_MAX_SEARCH_MATCHES = 100


def _iter_repo_files(root: str):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDED_DIRECTORY_NAMES]
        for filename in filenames:
            full = os.path.join(dirpath, filename)
            yield os.path.relpath(full, root).replace(os.sep, "/")


class ListRepoFilesTool(Tool):
    """CODE_READ - lists repo-relative file paths under the project
    root (optionally scoped to one subdirectory), bounded to
    `_MAX_LISTED_FILES`. A secret-named file's path is still listed
    (its EXISTENCE is not secret - only its content is), tagged so a
    caller/model never mistakes it for a normal, readable file."""

    name = "self_dev_list_files"
    description = "List repository files, optionally scoped to one subdirectory."
    permission = PermissionLevel.READ

    def __init__(self, project_root: str):
        self.project_root = project_root

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="subdirectory",
                type="string",
                description="Repo-relative subdirectory to list (default: the whole repo).",
                required=False,
                default="",
            )
        ]

    def execute(self, subdirectory: str = "") -> ToolResult:
        base = self.project_root
        if subdirectory:
            try:
                resolved = resolve_within_root(self.project_root, subdirectory)
            except (PathSafetyError, PathTraversalError) as exc:
                return ToolResult.error(str(exc))
            if not resolved.is_dir():
                return ToolResult.error(f"Not a directory: {subdirectory!r}")
            base = str(resolved)

        files = []
        for rel_path in _iter_repo_files(base):
            full_rel = f"{subdirectory}/{rel_path}" if subdirectory else rel_path
            files.append({"path": full_rel, "secret": is_secret_path(full_rel)})
            if len(files) >= _MAX_LISTED_FILES:
                break

        return ToolResult.success(data=files, message=f"{len(files)} file(s) listed.")


class ReadFileTool(Tool):
    """CODE_READ - reads one file's text content, bounded to
    `_MAX_READ_CHARS`. Refuses a secret-named file outright (reports
    that it exists and is protected, never its bytes) and refuses
    anything outside the project root (PathTraversalError/
    PathSafetyError -> a plain ToolResult.error, never an unhandled
    exception)."""

    name = "self_dev_read_file"
    description = "Read one repository file's text content."
    permission = PermissionLevel.READ

    def __init__(self, project_root: str):
        self.project_root = project_root

    @property
    def parameters(self) -> list[ToolParameter]:
        return [ToolParameter(name="path", type="string", description="Repo-relative file path.")]

    def execute(self, path: str) -> ToolResult:
        if is_secret_path(path):
            return ToolResult.error(
                f"'{path}' is a protected secret/config file - its contents are "
                "never shown, though it may exist."
            )
        try:
            resolved = resolve_within_root(self.project_root, path)
        except (PathSafetyError, PathTraversalError) as exc:
            return ToolResult.error(str(exc))

        if not resolved.is_file():
            return ToolResult.error(f"No such file: '{path}'")

        try:
            content = resolved.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            return ToolResult.error(f"Could not read '{path}': {exc}")

        truncated = len(content) > _MAX_READ_CHARS
        if truncated:
            content = content[:_MAX_READ_CHARS]

        return ToolResult.success(
            data={"path": path, "content": content, "truncated": truncated}
        )


class SearchRepoTool(Tool):
    """CODE_READ - a plain (non-regex) case-insensitive substring
    search across repo files, bounded to `_MAX_SEARCH_MATCHES` -
    deliberately not a regex engine (no ReDoS surface from an
    AI-supplied pattern). Skips secret-named and binary files."""

    name = "self_dev_search_repo"
    description = "Search repository files for a plain text substring."
    permission = PermissionLevel.READ

    def __init__(self, project_root: str):
        self.project_root = project_root

    @property
    def parameters(self) -> list[ToolParameter]:
        return [ToolParameter(name="query", type="string", description="Plain text to search for.")]

    def execute(self, query: str) -> ToolResult:
        if not query.strip():
            return ToolResult.error("Search query must not be empty.")

        lowered_query = query.lower()
        matches = []
        for rel_path in _iter_repo_files(self.project_root):
            if is_secret_path(rel_path):
                continue
            full_path = os.path.join(self.project_root, rel_path)
            try:
                with open(full_path, "r", encoding="utf-8") as f:
                    for line_number, line in enumerate(f, start=1):
                        if lowered_query in line.lower():
                            matches.append(
                                {"path": rel_path, "line": line_number, "text": line.strip()}
                            )
                            if len(matches) >= _MAX_SEARCH_MATCHES:
                                break
            except (OSError, UnicodeDecodeError):
                continue
            if len(matches) >= _MAX_SEARCH_MATCHES:
                break

        return ToolResult.success(data=matches, message=f"{len(matches)} match(es) found.")
