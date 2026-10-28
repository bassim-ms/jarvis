"""
Suite-wide protection for the repository's REAL databases (`jarvis_memory.db`, `jarvis_notes.db`, ...). See tests/conftest.py, which installs this before any test module is imported.

THE PROBLEM. Production code builds its stores from relative names ("jarvis_memory.db", ...) - some from a setting, some from a constant - and the process runs from the repository root,
so any test that builds a default `JarvisCore()`, the default tools, or imports `app.main` (whose module-level core opens every default store) opened the developer's real files,
and a test that chats through them WROTE to the real memory database.

THE FIX (tests only; no production default, setting or schema changes). `sqlite3.connect` is wrapped. A connection to a `jarvis_*.db` file that lives directly in the repository root
or in tests/ is REDIRECTED to a file of the same name inside one per-session temporary directory, so the real file is never opened. Everything else is untouched: ":memory:", a
test's own `tmp_path` databases and any path a test chooses deliberately (a file that merely has the real name in some other folder is NOT redirected). The redirect keeps the
semantics tests relied on - the same name still means the same file for the whole session, so data persists between two stores built in one session - but every session starts empty.

TEETH. Redirecting hides attempts, so two further checks make a leak visible: the metadata of every real database file is recorded when this is installed, and
`pytest_sessionfinish` (tests/conftest.py) FAILS the session if any of them changed; and tests/test_db_isolation.py pins the behaviour.

CLEANUP can never reach a real file: the temporary directory is created here with a fixed prefix directly under the system temp directory, and it is removed only if `is_safe_to_remove`
(a pure check, tested on its own) agrees that it is exactly such a directory and is not the repository or anything above or inside it.

NOT A SECURITY BOUNDARY: a test that opens a database through another library, or copies a real file, is not caught by the redirect (the session-end metadata check would see a changed file).
"""

from __future__ import annotations

import fnmatch
import gc
import os
import shutil
import sqlite3
import tempfile
import urllib.parse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TESTS_DIR = REPO_ROOT / "tests"
TEMP_PREFIX = "jarvis_test_dbs_"
GUARD_MARK = "__jarvis_db_isolation__"
_REAL_NAME_PATTERN = "jarvis_*.db"

#: Every redirected open, in order: (the real file's name, the file actually opened). The tests read it; nothing else does.
REDIRECTS: list[tuple[str, str]] = []
_CONNECTIONS: list = []  # the connections opened on redirected files, closed by `cleanup`

_STATE: dict = {"dir": None, "snapshot": None}


def _folder_key(path: Path) -> str:
    return os.path.normcase(os.path.realpath(path))


def is_real_database(path: object) -> bool:
    """True for a `jarvis_*.db` file directly in the repository root or in tests/ (judged by where the path resolves, relative paths against the current directory). A name that
    only looks like a real database but lives in another folder is not one. Anything that is not a plain path (":memory:", "", bytes, None) is not one."""
    try:
        text = os.fspath(path)  # type: ignore[arg-type]
    except TypeError:
        return False
    if not isinstance(text, str) or text in ("", ":memory:"):
        return False
    resolved = Path(os.path.abspath(text))
    if not fnmatch.fnmatch(resolved.name.lower(), _REAL_NAME_PATTERN):
        return False
    return _folder_key(resolved.parent) in {_folder_key(REPO_ROOT), _folder_key(TESTS_DIR)}


def _uri_path(uri: str) -> tuple[str, str]:
    """(file path, query) of a `file:` URI; the path is returned as a plain path."""
    parts = urllib.parse.urlsplit(uri)
    path = urllib.parse.unquote(parts.path)
    if len(path) > 2 and path[0] == "/" and path[2] == ":":  # file:///C:/dir/x.db  ->  C:/dir/x.db
        path = path[1:]
    return path, parts.query


def session_dir() -> Path:
    directory = _STATE["dir"]
    if directory is None:
        raise RuntimeError("the database isolation is not installed")
    return directory


def redirected_target(name: str) -> Path:
    return session_dir() / name


def _watched_folders() -> tuple[Path, ...]:
    """The folders whose `jarvis_*.db` files the session-end check watches. A function so that a test can point the CHECK at a throwaway repository inside its own tmp_path; the redirect
    itself (`is_real_database`) never consults it, so such a test does not weaken the protection of the real files."""
    return (REPO_ROOT, TESTS_DIR)


def real_database_files() -> list[Path]:
    found: list[Path] = []
    for folder in _watched_folders():
        if folder.is_dir():
            found.extend(sorted(path for path in folder.glob(_REAL_NAME_PATTERN) if path.is_file()))
    return found


def snapshot_real_files() -> dict[str, tuple[int, int]]:
    """Metadata only (size, modification time in ns) of every real database file - their CONTENT is never read here."""
    result = {}
    for path in real_database_files():
        info = path.stat()
        result[str(path)] = (info.st_size, info.st_mtime_ns)
    return result


def changes_between(before: dict, after: dict) -> list[str]:
    """Which real files appeared, disappeared or changed (by size or modification time) between two snapshots."""
    return sorted(
        f"{os.path.basename(name)}: {'created' if name not in before else 'removed' if name not in after else 'changed'}"
        for name in set(before) | set(after)
        if before.get(name) != after.get(name)
    )


def session_changes() -> list[str]:
    snapshot = _STATE["snapshot"]
    if snapshot is None:
        raise RuntimeError("the database isolation is not installed")
    return changes_between(snapshot, snapshot_real_files())


def _system_temp() -> Path:
    return Path(os.path.realpath(tempfile.gettempdir()))


def is_safe_to_remove(path: object) -> bool:
    """A PURE check (it deletes nothing): is `path` exactly a directory this module could have created - named with TEMP_PREFIX, a direct child of the system temp directory, not a link, and
    neither the repository nor anything above or inside it?"""
    try:
        candidate = Path(os.fspath(path))  # type: ignore[arg-type]
    except TypeError:
        return False
    if not candidate.name.startswith(TEMP_PREFIX) or not candidate.is_dir() or candidate.is_symlink():
        return False
    resolved = Path(os.path.realpath(candidate))
    if os.path.normcase(os.path.abspath(candidate)) != os.path.normcase(str(resolved)):  # reached through a link or junction
        return False
    if os.path.normcase(str(resolved.parent)) != os.path.normcase(str(_system_temp())):
        return False
    repo = Path(os.path.realpath(REPO_ROOT))
    return resolved != repo and repo not in resolved.parents and resolved not in repo.parents


def remove_session_dir(path: object) -> bool:
    """Remove a session directory if (and only if) `is_safe_to_remove` says it is one. True if it was removed (files an open connection still locks may remain)."""
    if not is_safe_to_remove(path):
        return False
    gc.collect()  # let unreferenced connections close their files first (Windows cannot delete an open file)
    shutil.rmtree(Path(os.fspath(path)), ignore_errors=True)  # type: ignore[arg-type]
    return True


def _close_all(connections: list) -> None:
    """Close every connection in the list (already-closed or failing ones are ignored) and empty it."""
    for connection in connections:
        try:
            connection.close()
        except Exception:  # noqa: BLE001 - at the end of the session a connection that cannot close changes nothing
            pass
    connections.clear()


def cleanup() -> None:
    """End of the session: close the redirected connections (Windows cannot delete a file that is still open, and module-level stores such as the real app's keep theirs open for the
    whole run), then remove the session directory if it is exactly the directory this module created."""
    _close_all(_CONNECTIONS)
    directory = _STATE["dir"]
    if directory is not None:
        remove_session_dir(directory)


def install() -> None:
    """Idempotent. Create the session directory, record the real files' metadata and wrap `sqlite3.connect`."""
    if getattr(sqlite3.connect, GUARD_MARK, False):
        return
    directory = Path(os.path.realpath(tempfile.mkdtemp(prefix=TEMP_PREFIX)))
    _STATE["dir"] = directory
    _STATE["snapshot"] = snapshot_real_files()
    real_connect = sqlite3.connect

    def connect(*args, **kwargs):
        target = args[0] if args else kwargs.get("database")
        uri = bool(kwargs.get("uri", False))
        redirected = None  # (the real file's name, the plain path actually opened, what to hand to sqlite)
        if uri and isinstance(target, str) and target.startswith("file:"):
            path, query = _uri_path(target)
            if is_real_database(path):
                name = os.path.basename(path)
                redirected = (name, str(redirected_target(name)), redirected_target(name).as_uri() + (f"?{query}" if query else ""))
        elif is_real_database(target):
            name = os.path.basename(os.fspath(target))
            redirected = (name, str(redirected_target(name)), str(redirected_target(name)))
        if redirected is None:
            return real_connect(*args, **kwargs)
        REDIRECTS.append((redirected[0], redirected[1]))
        if args:
            args = (redirected[2], *args[1:])
        else:
            kwargs["database"] = redirected[2]
        connection = real_connect(*args, **kwargs)
        _CONNECTIONS.append(connection)  # (a sqlite3.Connection cannot be weakly referenced, so the list is the only way to close them at the end)
        return connection

    connect.__wrapped_real__ = real_connect  # type: ignore[attr-defined]
    setattr(connect, GUARD_MARK, True)
    sqlite3.connect = connect  # type: ignore[assignment]


def isolation_report() -> dict:
    """Is everything in place? `all_ok` is what tests/conftest.py insists on before running a single test."""
    directory = _STATE["dir"]
    in_place = getattr(sqlite3.connect, GUARD_MARK, False) is True
    dir_ok = directory is not None and is_safe_to_remove(directory)
    snapshot_ok = _STATE["snapshot"] is not None
    return {"redirect_installed": in_place, "session_dir_safe": dir_ok, "real_files_recorded": snapshot_ok, "all_ok": in_place and dir_ok and snapshot_ok}
