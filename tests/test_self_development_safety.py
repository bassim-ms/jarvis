"""
Tests for app/self_development/safety.py: the security choke points
every other module in this package depends on.

Covers focused-test categories:
1. repository-root enforcement
2. path traversal rejection
3. dangerous command rejection
6. secret-file protection (detection only - see inspector/editor tests
   for "never reveals content")
"""

import os
import tempfile

import pytest

from app.self_development.safety import (
    PathSafetyError,
    PathTraversalError,
    UnsafeCommandError,
    is_safe_relative_path,
    is_secret_path,
    resolve_within_root,
    validate_command,
)


# -- path safety --------------------------------------------------------


def test_ordinary_relative_path_is_safe():
    assert is_safe_relative_path("app/core.py") is True


def test_parent_traversal_is_unsafe():
    assert is_safe_relative_path("../etc/passwd") is False
    assert is_safe_relative_path("app/../../etc/passwd") is False


def test_absolute_posix_path_is_unsafe():
    assert is_safe_relative_path("/etc/passwd") is False


def test_absolute_windows_path_is_unsafe():
    assert is_safe_relative_path("C:\\Windows\\System32") is False


def test_empty_and_dot_paths_are_unsafe():
    assert is_safe_relative_path("") is False
    assert is_safe_relative_path(".") is False
    assert is_safe_relative_path("./app") is False


def test_null_byte_is_unsafe():
    assert is_safe_relative_path("app/core.py\x00.txt") is False


# -- repository-root enforcement (real filesystem resolution) -----------


def test_resolve_within_root_accepts_a_real_nested_path():
    with tempfile.TemporaryDirectory() as root:
        os.makedirs(os.path.join(root, "app"))
        open(os.path.join(root, "app", "x.py"), "w").close()
        resolved = resolve_within_root(root, "app/x.py")
        assert resolved == (
            __import__("pathlib").Path(root).resolve() / "app" / "x.py"
        )


def test_resolve_within_root_rejects_traversal():
    with tempfile.TemporaryDirectory() as root:
        with pytest.raises((PathSafetyError, PathTraversalError)):
            resolve_within_root(root, "../outside.txt")


def test_resolve_within_root_rejects_a_syntactically_unsafe_path_before_any_io():
    with tempfile.TemporaryDirectory() as root:
        with pytest.raises(PathSafetyError):
            resolve_within_root(root, "/etc/passwd")


def test_resolve_within_root_rejects_traversal_that_would_escape_via_symlink(tmp_path):
    """A symlink INSIDE the root pointing OUTSIDE it must still be
    caught - the real filesystem resolution (not just the pure
    syntactic check) is what catches this."""
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (outside / "secret.txt").write_text("secret")
    try:
        (root / "link").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("Symlink creation not permitted in this environment.")

    with pytest.raises(PathTraversalError):
        resolve_within_root(str(root), "link/secret.txt")


# -- secret-file detection ------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        ".env",
        ".env.local",
        "id_rsa",
        "id_rsa.pub",
        "server.pem",
        "server.key",
        "app/credentials.json",
        "config/secrets.yaml",
        "notes/my_token.txt",
        "auth/password_reset.py",
    ],
)
def test_secret_paths_are_detected(path):
    assert is_secret_path(path) is True


@pytest.mark.parametrize(
    "path",
    ["app/core.py", "tests/test_core.py", "README.md", "app/self_development/engine.py"],
)
def test_ordinary_paths_are_not_secret(path):
    assert is_secret_path(path) is False


# -- command safety: allowlist ---------------------------------------------


@pytest.mark.parametrize(
    "argv",
    [
        ["pytest", "-q"],
        ["python", "-m", "pytest", "-q"],
        ["git", "status"],
        ["git", "diff"],
        ["git", "diff", "--cached"],
        ["git", "checkout", "-b", "jarvis-selfdev/abc123"],
        ["git", "checkout", "HEAD", "--", "app/foo.py"],
        ["git", "checkout", "main"],
        ["git", "add", "-A"],
        ["git", "commit", "-m", "a normal commit message"],
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
    ],
)
def test_allowed_commands_pass(argv):
    validate_command(argv)  # must not raise


# -- command safety: dangerous command rejection (category 3) -----------


@pytest.mark.parametrize(
    "argv",
    [
        ["git", "reset", "--hard"],
        ["git", "reset", "--hard", "HEAD~1"],
        ["git", "clean", "-fd"],
        ["git", "push"],
        ["git", "push", "--force"],
        ["git", "checkout", "--", "app/foo.py"],  # discard form, no HEAD source
        ["git", "checkout", "-b", "-x"],  # branch name looks like a flag
        ["git", "branch", "-D", "main"],
        ["git", "add", "."],
        ["git", "commit", "-a", "-m", "x"],
        ["rm", "-rf", "/"],
        ["del", "/f", "/q", "app"],
        ["format", "C:"],
        ["shutdown", "/s", "/t", "0"],
        ["python", "arbitrary_script.py"],
        ["python", "-c", "import os; os.system('rm -rf /')"],
        ["powershell", "-Command", "Remove-Item -Recurse -Force ."],
        ["cmd", "/c", "del /f /q *"],
        ["reg", "add", "HKLM\\Software\\Evil"],
        ["diskpart"],
        [],
    ],
)
def test_dangerous_or_disallowed_commands_are_rejected(argv):
    with pytest.raises(UnsafeCommandError):
        validate_command(argv)


def test_shell_metacharacters_inside_an_argv_element_are_harmless_as_data():
    """Since this engine NEVER uses shell=True, a shell metacharacter
    embedded inside e.g. a commit message is just literal text, not an
    injection vector - validate_command doesn't need to (and
    deliberately doesn't) reject an ordinary commit message merely for
    containing punctuation."""
    validate_command(["git", "commit", "-m", "fix: handle a && b; also |pipes|"])


def test_command_must_be_a_list_not_a_string():
    with pytest.raises(UnsafeCommandError):
        validate_command("git status")  # a shell string, never accepted
