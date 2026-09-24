"""
Safety primitives for the Self-Development Engine (V1) - the choke
points every other module in this package depends on. Nothing here
executes anything; it only decides what's ALLOWED to.

Two independent boundaries live here:

1. Path safety (`is_safe_relative_path`/`resolve_within_root`) - every
   file the engine ever reads or writes must resolve to somewhere
   inside the configured project root. Checked twice: a pure,
   syntactic check with no filesystem access (rejects `..`, absolute
   paths, drive letters, null bytes - safe to run on unvalidated
   LLM-proposed paths before any I/O happens at all), and a real
   filesystem resolution check (`resolve_within_root`, catches symlink
   tricks and anything the syntactic check alone couldn't).

2. Command safety (`validate_command`) - every command this engine
   might run is an explicit argv LIST, never a shell string. Commands
   are executed with `shell=False` everywhere in this package (see
   app/self_development/command_runner.py) - shell metacharacters
   (`&&`, `;`, `|`, backticks) are therefore inert by construction, not
   merely filtered. `validate_command` additionally enforces a small,
   explicit allowlist of (executable, subcommand) pairs and a
   defense-in-depth denylist scan of every argument, so even a fully
   argv-safe invocation of an allowed executable can't smuggle a
   destructive flag through.
"""

from __future__ import annotations

import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Path safety
# ---------------------------------------------------------------------------


class PathSafetyError(ValueError):
    """Raised for a relative path that is unsafe by construction (`..`,
    absolute, drive letter, null byte, empty) - never a filesystem
    check, see `resolve_within_root` for that."""


class PathTraversalError(ValueError):
    """Raised by `resolve_within_root` when a syntactically-plausible
    relative path actually resolves (after following any symlinks)
    outside the configured project root."""


#: A syntactically safe relative path: forward or back slashes as
#: separators, each component a normal filename (no `.`/`..`, no drive
#: letter, no leading separator, no null byte, no empty component).
_SAFE_COMPONENT_PATTERN = re.compile(r"^[^\\/:*?\"<>|\x00]+$")


def is_safe_relative_path(path: str) -> bool:
    """Pure, no-I/O check that `path` COULD be a safe repo-relative
    path - never raises, returns False for anything unsafe. This is
    deliberately usable on an untrusted, not-yet-resolved string (e.g.
    a path an LLM proposed in a DevelopmentPlan) before any filesystem
    access happens at all."""
    if not path or not isinstance(path, str):
        return False
    if "\x00" in path:
        return False
    # An absolute path (POSIX or Windows) or a drive letter is never a
    # valid repo-relative path, regardless of OS.
    if path.startswith(("/", "\\")):
        return False
    if re.match(r"^[A-Za-z]:[\\/]", path):
        return False
    normalized = path.replace("\\", "/")
    components = normalized.split("/")
    for component in components:
        if component in ("", ".", ".."):
            return False
        if not _SAFE_COMPONENT_PATTERN.match(component):
            return False
    return True


def resolve_within_root(root: str, relative_path: str) -> Path:
    """Resolve `relative_path` against `root` and verify the resolved
    absolute path is genuinely inside `root` - the REAL enforcement
    point (catches symlinks and anything `is_safe_relative_path`'s pure
    string check alone can't). Raises PathSafetyError first for a path
    that fails the syntactic check, or PathTraversalError if it somehow
    still resolves outside the root."""
    if not is_safe_relative_path(relative_path):
        raise PathSafetyError(f"Unsafe relative path: {relative_path!r}")

    resolved_root = Path(root).resolve()
    candidate = (resolved_root / relative_path).resolve()

    if candidate != resolved_root and resolved_root not in candidate.parents:
        raise PathTraversalError(
            f"Path {relative_path!r} resolves outside the project root."
        )
    return candidate


# ---------------------------------------------------------------------------
# Secret-file detection (never reveals contents - see inspector.py)
# ---------------------------------------------------------------------------

#: Filenames/patterns treated as secret/config-sensitive - matched
#: case-insensitively against the path's own components, never its
#: content (content is never inspected to decide this - a file is
#: secret by NAME, deterministically, not by guessing at its bytes).
_SECRET_NAME_PATTERNS = (
    re.compile(r"^\.env(\..*)?$", re.IGNORECASE),
    re.compile(r".*\.pem$", re.IGNORECASE),
    re.compile(r".*\.key$", re.IGNORECASE),
    re.compile(r".*\.p12$", re.IGNORECASE),
    re.compile(r".*\.pfx$", re.IGNORECASE),
    re.compile(r"^id_rsa.*$", re.IGNORECASE),
    re.compile(r".*credentials?.*", re.IGNORECASE),
    re.compile(r".*secret.*", re.IGNORECASE),
    re.compile(r".*token.*", re.IGNORECASE),
    re.compile(r".*password.*", re.IGNORECASE),
)

#: Directories never listed/read/searched/edited at all - never even
#: acknowledged to exist to the model, unlike a secret FILE (which the
#: engine may confirm exists without revealing its content).
EXCLUDED_DIRECTORY_NAMES = frozenset(
    {".git", "venv", "__pycache__", "node_modules", ".pytest_cache"}
)


def is_secret_path(relative_path: str) -> bool:
    """True if any path component's filename looks like a secret/
    credential file by name - `.env`, `*.pem`, `*.key`, anything with
    "secret"/"token"/"password"/"credential" in the name. Used to
    refuse reading OR writing such a file's contents (see
    inspector.py/editor.py) - the engine may still report that such a
    file EXISTS, never what it contains."""
    normalized = relative_path.replace("\\", "/")
    for component in normalized.split("/"):
        if any(pattern.match(component) for pattern in _SECRET_NAME_PATTERNS):
            return True
    return False


# ---------------------------------------------------------------------------
# Protected security-critical paths - a SEPARATE, stronger boundary from
# secrets above. These are ordinary, non-secret source files that
# implement the Self-Development Engine's OWN safety/approval machinery
# (or the wider permission/capability system it's built on) - a plan may
# still propose touching them (V1 must remain able to develop itself),
# but doing so requires DevelopmentCapability.SECURITY_EDIT, a distinct,
# never-self-grantable approval on top of ordinary CODE_EDIT (see
# app/self_development/base.py:DevelopmentPlan.__post_init__, which
# refuses to even construct a plan that touches one of these without
# declaring SECURITY_EDIT, and app/self_development/editor.py, where
# EditFileTool itself refuses these paths outright - only
# SecurityEditFileTool, PermissionLevel.DANGEROUS, may write them).
# ---------------------------------------------------------------------------

#: Exact files and directory prefixes (repo-relative, forward-slash
#: normalized) that implement Jarvis's own security/permission
#: machinery. Directories end in "/" and match anything underneath;
#: single files match exactly. Deliberately a small, fixed, explicit
#: list - never a heuristic - so it's always auditable at a glance.
PROTECTED_SYSTEM_PATH_PREFIXES: tuple[str, ...] = (
    "app/self_development/",
    "app/tools/",
    "app/capability.py",
    "app/core.py",
    "app/config.py",
)


def is_protected_system_path(relative_path: str) -> bool:
    """True if `relative_path` falls under one of
    `PROTECTED_SYSTEM_PATH_PREFIXES` - the Self-Development Engine's own
    implementation, the generic Tool/permission system it's built on
    (app/tools/), the capability router that decides whether a message
    even reaches this engine (app/capability.py), JarvisCore's own
    dispatch (app/core.py), or application configuration (app/config.py).
    Never influenced by file CONTENT, only by path - the same
    deterministic-by-name discipline `is_secret_path` already uses."""
    normalized = relative_path.replace("\\", "/")
    for prefix in PROTECTED_SYSTEM_PATH_PREFIXES:
        if prefix.endswith("/"):
            if normalized == prefix.rstrip("/") or normalized.startswith(prefix):
                return True
        elif normalized == prefix:
            return True
    return False


# ---------------------------------------------------------------------------
# Branch name enforcement - see git.py:GitClient.create_branch, which
# calls `validate_branch_name` itself (never trusting a caller to have
# pre-validated) so an invalid/non-namespaced name is rejected even if
# something calls the Git layer directly, bypassing `branch_name_for`.
# ---------------------------------------------------------------------------

#: Every Self-Development branch this engine ever creates must match
#: this exact shape - "jarvis-selfdev/<safe-slug>". Kept here (not only
#: in git.py) so the pattern is defined once and reused by both the
#: name-building helper (git.py:branch_name_for) and the enforcement
#: point (git.py:GitClient.create_branch).
DEV_BRANCH_NAME_PATTERN = re.compile(r"^jarvis-selfdev/[A-Za-z0-9][A-Za-z0-9_-]{0,60}$")


class InvalidBranchNameError(ValueError):
    """Raised by GitClient.create_branch for any name that doesn't
    match DEV_BRANCH_NAME_PATTERN - enforced in the Git layer itself,
    never only by a caller's own discipline."""


def validate_branch_name(name: str) -> None:
    if not isinstance(name, str) or not DEV_BRANCH_NAME_PATTERN.match(name):
        raise InvalidBranchNameError(
            f"Invalid Self-Development branch name: {name!r}. Must match "
            f"'jarvis-selfdev/<safe-slug>'."
        )


# ---------------------------------------------------------------------------
# Command safety
# ---------------------------------------------------------------------------


class UnsafeCommandError(ValueError):
    """Raised by `validate_command` for any argv this engine refuses to
    run at all - never downgraded to a confirmation prompt. A command
    that fails this check is rejected outright, with no way for a
    caller-supplied `confirmed=True` to override it (see
    app/self_development/command_runner.py:RunCommandTool, which calls
    this BEFORE ToolRegistry's own confirmation gate is even reached)."""


#: The only executables this engine will ever invoke. Deliberately
#: tiny - "tests, Python tooling, Git inspection, and existing project
#: checks" (V1's own scope), nothing else.
_ALLOWED_EXECUTABLES = frozenset({"git", "python", "pytest"})

#: For `git`, the only subcommands ever allowed. Notably absent:
#: "reset" and "clean" - no allowed form of either exists anywhere in
#: this engine (rollback is a targeted per-file restore-from-HEAD plus
#: an ordinary branch switch, never `reset --hard`/`clean` - see
#: `_validate_git_checkout` and git.py); "branch"/"log"/"show" -
#: unused by this engine's own GitClient, so not offered at all, to
#: keep the surface area minimal.
_ALLOWED_GIT_SUBCOMMANDS = frozenset({"status", "diff", "checkout", "add", "commit", "rev-parse"})

#: A flag/token denylist checked against every individual argv element
#: (never a substring-of-the-whole-line scan, which would also reject
#: an innocent commit message containing an ordinary English word like
#: "reset" or "clean") - defense in depth for a dangerous flag slipping
#: through as its own argv element on an otherwise allow-listed command.
_DANGEROUS_TOKENS = frozenset(
    {
        "--hard",
        "--force",
        "--force-with-lease",
        "-f",
        "-rf",
        "-fr",
        "-r",
        "--recursive",
        "-d",
        "-D",
    }
)


def _validate_git_checkout(argv: list[str]) -> None:
    """Exactly three safe shapes are allowed for `git checkout`,
    checked structurally (not by flag presence alone):

    1. `git checkout -b <new-branch>` - create and switch to a new
       branch (BRANCH_CREATE). Never destroys anything.
    2. `git checkout HEAD -- <path>` - restore ONE tracked file to its
       last-committed content (used only by GitClient's own targeted
       rollback of files this engine itself edited - never a wildcard,
       never "." for "everything").
    3. `git checkout <branch-name>` - switch to an ALREADY-EXISTING
       branch by a bare name. Only ever called by GitClient.rollback
       once every edited file has already been restored via shape 2
       above, so the working tree is guaranteed clean and this can
       never carry uncommitted changes onto the target branch (see
       git.py's own rollback docstring).

    Anything else - `git checkout -- <path>` without an explicit
    `HEAD` source, `git checkout .`, or extra/unexpected arguments -
    is rejected."""
    if len(argv) == 4 and argv[2] == "-b" and not argv[3].startswith("-"):
        return
    if (
        len(argv) == 5
        and argv[2] == "HEAD"
        and argv[3] == "--"
        and argv[4] not in (".", "*")
        and not argv[4].startswith("-")
    ):
        return
    if len(argv) == 3 and argv[2] not in (".", "*", "--") and not argv[2].startswith("-"):
        return
    raise UnsafeCommandError(
        f"Unsafe 'git checkout' form: {argv!r}. Only 'git checkout -b <branch>', "
        "'git checkout HEAD -- <path>', or 'git checkout <existing-branch>' are "
        "allowed."
    )


def _validate_git_add(argv: list[str]) -> None:
    if argv[2:] != ["-A"]:
        raise UnsafeCommandError(f"Only 'git add -A' is allowed, got: {argv!r}")


def _validate_git_commit(argv: list[str]) -> None:
    if len(argv) != 4 or argv[2] != "-m" or not argv[3].strip():
        raise UnsafeCommandError(
            f"Only 'git commit -m <non-empty message>' is allowed, got: {argv!r}"
        )


_GIT_SUBCOMMAND_VALIDATORS = {
    "checkout": _validate_git_checkout,
    "add": _validate_git_add,
    "commit": _validate_git_commit,
}


def validate_command(argv: list[str]) -> None:
    """Raise UnsafeCommandError unless `argv` is a fully allow-listed,
    non-destructive command. Never inspects or executes anything -
    pure validation. `argv` must already be a list (never a shell
    string) - see command_runner.py's own docstring for why this
    engine never accepts one."""
    if not isinstance(argv, list) or not argv or not all(isinstance(a, str) for a in argv):
        raise UnsafeCommandError(f"Command must be a non-empty list of strings, got: {argv!r}")

    for token in argv:
        if token in _DANGEROUS_TOKENS:
            raise UnsafeCommandError(f"Command contains a disallowed flag ({token!r}): {argv!r}")

    executable = argv[0]
    if executable not in _ALLOWED_EXECUTABLES:
        raise UnsafeCommandError(
            f"Executable '{executable}' is not in the allowed set: "
            f"{sorted(_ALLOWED_EXECUTABLES)}"
        )

    if executable == "git":
        if len(argv) < 2 or argv[1] not in _ALLOWED_GIT_SUBCOMMANDS:
            raise UnsafeCommandError(
                f"Git subcommand not allowed: {argv[1:2]!r}. Must be one of: "
                f"{sorted(_ALLOWED_GIT_SUBCOMMANDS)}"
            )
        validator = _GIT_SUBCOMMAND_VALIDATORS.get(argv[1])
        if validator is not None:
            validator(argv)

    if executable == "python" and len(argv) >= 2 and argv[1] != "-m":
        raise UnsafeCommandError(
            "Only 'python -m <module> ...' is allowed - never running an "
            "arbitrary script path directly."
        )
