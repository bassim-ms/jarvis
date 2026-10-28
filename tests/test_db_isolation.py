"""
Regression tests for the suite-wide database isolation (tests/db_isolation.py, installed by tests/conftest.py).

Before it existed, ordinary tests opened - and, by chatting through a default core, WROTE to - the developer's real `jarvis_memory.db` (and opened eight sibling databases). These tests
pin that a default `JarvisCore`, the default tools and the real app's own stores are backed by TEMPORARY files; that the real files do not change; that paths a test chooses itself are
left alone; that production defaults are untouched; and that cleanup cannot reach a real file.

Every test that writes first asserts WHERE its connection points, so a broken redirect fails the assertion before anything is written.
"""

import hashlib
import importlib
import inspect
import os
import sqlite3
import tempfile
from pathlib import Path

import pytest

import db_isolation as iso
from app.config import Settings, settings
from app.core import JarvisCore
from app.memory.sqlite_store import SQLiteMemoryStore
from app.providers.base import AIProvider
from app.tools.defaults import create_default_registry

REAL_MEMORY = iso.REPO_ROOT / "jarvis_memory.db"


class _Provider(AIProvider):
    def generate_reply(self, prompt: str) -> str:
        return "ok"


def _file_of(connection: sqlite3.Connection) -> Path:
    return Path(connection.execute("PRAGMA database_list").fetchone()[2])


def _is_session_temp(path: Path) -> bool:
    return Path(path).resolve().parent == iso.session_dir().resolve() and iso.REPO_ROOT.resolve() not in Path(path).resolve().parents


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _real_state() -> dict:
    """Metadata of every real database file, and the memory database's content hash (read-only), for before/after comparison."""
    return {"meta": iso.snapshot_real_files(), "memory_sha": _sha(REAL_MEMORY) if REAL_MEMORY.exists() else None}


# ======================================================== the isolation is in place ========================================================


def test_the_isolation_is_installed_and_every_check_passes():
    report = iso.isolation_report()
    assert report == {"redirect_installed": True, "session_dir_safe": True, "real_files_recorded": True, "all_ok": True}
    assert getattr(sqlite3.connect, iso.GUARD_MARK) is True
    assert iso.session_dir().name.startswith(iso.TEMP_PREFIX) and iso.session_dir().is_dir()


def test_installing_twice_changes_nothing():
    wrapped, directory = sqlite3.connect, iso.session_dir()
    iso.install()
    assert sqlite3.connect is wrapped and iso.session_dir() == directory


# ================================== the three construction paths named in the bug: Core, default tools, the real app ===================================


def test_a_default_core_stores_into_a_temporary_file_and_the_real_database_does_not_change():
    before = _real_state()

    core = JarvisCore(provider=_Provider())  # no memory=: exactly the call the 19 test sites make
    file = _file_of(core.memory._conn)

    assert _is_session_temp(file) and file.name == "jarvis_memory.db"  # asserted BEFORE anything is written
    core.memory.save_fact("isolation_probe", "value")
    conversation = core.memory.create_conversation()
    core.memory.add_message(conversation, "user", "isolation probe message")
    assert core.memory.get_facts()["isolation_probe"] == "value"
    assert _real_state() == before


def test_building_the_default_tools_opens_only_temporary_databases_and_changes_no_real_file():
    before, first = _real_state(), len(iso.REDIRECTS)

    create_default_registry()

    opened = iso.REDIRECTS[first:]
    names = {name for name, _ in opened}
    assert {"jarvis_memory.db", "jarvis_reminders.db", "jarvis_automations.db"} <= names  # the default tools really do build these stores
    assert all(_is_session_temp(Path(actual)) for _, actual in opened)
    assert _real_state() == before


def test_the_real_apps_own_stores_are_temporary_files():
    from app import main  # its module-level core opens every default store when first imported (here or earlier in the session)

    assert _is_session_temp(_file_of(main.core.memory._conn))


# ===================================================== what the redirect does, and does not, touch ======================================================


@pytest.mark.parametrize("how", ["absolute", "relative", "keyword", "uri", "tests-folder"])
def test_every_way_of_naming_a_real_database_lands_in_the_temporary_directory(how, monkeypatch):
    monkeypatch.chdir(iso.REPO_ROOT)
    name = f"jarvis_isolation_probe_{how.replace('-', '_')}.db"  # a name no real file has: the probe can never touch one
    in_repo = iso.REPO_ROOT / name
    in_tests = iso.TESTS_DIR / name
    first = len(iso.REDIRECTS)

    connection = {
        "absolute": lambda: sqlite3.connect(str(in_repo)),
        "relative": lambda: sqlite3.connect(name),
        "keyword": lambda: sqlite3.connect(database=name),
        "uri": lambda: sqlite3.connect(f"file:{name}?mode=rwc", uri=True),
        "tests-folder": lambda: sqlite3.connect(str(in_tests)),
    }[how]()
    try:
        connection.execute("CREATE TABLE IF NOT EXISTS probe (x)")
        connection.commit()
        assert _is_session_temp(_file_of(connection)) and _file_of(connection).name == name
    finally:
        connection.close()

    assert not in_repo.exists() and not in_tests.exists()  # nothing was created next to the real files
    assert [entry[0] for entry in iso.REDIRECTS[first:]] == [name]


def test_paths_a_test_chooses_itself_are_not_redirected(tmp_path):
    first = len(iso.REDIRECTS)
    chosen = tmp_path / "chosen.db"
    same_name_elsewhere = tmp_path / "jarvis_memory.db"  # only the FOLDER makes a file "real"

    for path in (chosen, same_name_elsewhere):
        connection = sqlite3.connect(str(path))
        assert Path(_file_of(connection)).resolve() == path.resolve()
        connection.close()
    memory = sqlite3.connect(":memory:")
    assert _file_of(memory).as_posix() in ("", ".")
    memory.close()

    assert len(iso.REDIRECTS) == first


@pytest.mark.parametrize("filename", ["custom_memory.db", "jarvis_memory.db"])
def test_a_deliberately_configured_memory_path_is_preserved_even_when_it_has_the_production_file_name(filename, tmp_path, monkeypatch):
    """The second case is the one that matters: the PRODUCTION basename in a folder this test owns. Only the folder makes a file "real", so it must not be redirected."""
    owned = tmp_path / "owned_by_this_test"
    owned.mkdir()
    custom = owned / filename
    assert not iso.is_real_database(custom)  # the rule itself
    first = len(iso.REDIRECTS)
    monkeypatch.setattr(settings, "memory_db_path", str(custom))

    core = JarvisCore(provider=_Provider())
    session_copy = SQLiteMemoryStore(str(iso.redirected_target("jarvis_memory.db")))  # the session's temporary stand-in for the production-named database
    reopened = None
    try:
        assert Path(_file_of(core.memory._conn)).resolve() == custom.resolve()  # the explicit path was used as given ...
        assert [name for name, _ in iso.REDIRECTS[first:] if name == "jarvis_memory.db"] == []  # ... and never redirected
        core.memory.save_fact("kept", "yes")
        reopened = SQLiteMemoryStore(str(custom))
        assert reopened.get_facts() == {"kept": "yes"}  # it really persisted at the chosen path
        assert custom.stat().st_size > 0
        assert "kept" not in session_copy.get_facts()  # and did not leak into the session's temporary copy
    finally:
        for store in (core.memory, session_copy, reopened):
            if store is not None:
                store._conn.close()  # (the session copy must not stay open: the session directory is removed at the end)


def test_production_defaults_and_settings_are_untouched_by_the_isolation():
    assert Settings.model_fields["memory_db_path"].default == "jarvis_memory.db"
    assert settings.memory_db_path == "jarvis_memory.db"  # the setting still holds the real, relative name: the redirect is below it
    assert inspect.signature(SQLiteMemoryStore.__init__).parameters["db_path"].default == "jarvis_memory.db"
    for field, info in Settings.model_fields.items():
        if field.endswith("_db_path"):
            assert getattr(settings, field) == info.default


# ================================================== persistence still works, with temporary databases ===================================================


def test_sqlite_persistence_works_in_an_explicit_temporary_database(tmp_path):
    path = str(tmp_path / "persist.db")
    first = SQLiteMemoryStore(path)
    first.save_fact("name", "value")
    conversation = first.create_conversation()
    first.add_message(conversation, "user", "hello")

    second = SQLiteMemoryStore(path)
    assert second.get_facts() == {"name": "value"}
    assert [message.content for message in second.get_messages(conversation)] == ["hello"]


def test_the_same_default_name_means_the_same_temporary_file_for_the_whole_session():
    before = _real_state()
    first = SQLiteMemoryStore(settings.memory_db_path)
    assert _is_session_temp(_file_of(first._conn))
    first.save_fact("session_probe", "1")
    try:
        second = SQLiteMemoryStore(settings.memory_db_path)  # a second store built the way production code builds it
        assert second.get_facts()["session_probe"] == "1"
    finally:
        first.delete_fact("session_probe")
    assert _real_state() == before


# ====================================== cleanup can never reach a real file (the check is pure; only a throwaway is ever removed) ===========================


def test_the_safety_check_refuses_the_repository_everything_around_it_and_anything_that_is_not_a_session_directory(tmp_path):
    system_temp = Path(tempfile.gettempdir())
    for refused in (
        iso.REPO_ROOT, iso.TESTS_DIR, iso.REPO_ROOT.parent, iso.REPO_ROOT.parent.parent, system_temp, Path(system_temp.anchor),
        tmp_path,  # a real temp directory, but not a session directory (no prefix, not a direct child of the system temp)
        tmp_path / (iso.TEMP_PREFIX + "inside"),  # the prefix alone is not enough: it must sit directly under the system temp directory
        system_temp / (iso.TEMP_PREFIX + "does_not_exist"),
        REAL_MEMORY,  # a file, not a directory
        None, b"bytes", 42,
    ):
        assert iso.is_safe_to_remove(refused) is False, refused
    assert iso.is_safe_to_remove(iso.session_dir()) is True


def test_only_a_genuine_session_directory_is_ever_removed(tmp_path):
    real_before = _real_state()
    throwaway = Path(tempfile.mkdtemp(prefix=iso.TEMP_PREFIX))
    (throwaway / "x.db").write_bytes(b"not a database")
    (tmp_path / "keep.txt").write_text("keep")

    assert iso.remove_session_dir(throwaway) is True and not throwaway.exists()
    assert iso.remove_session_dir(tmp_path) is False and (tmp_path / "keep.txt").exists()  # refused: not a session directory
    assert iso.remove_session_dir(tmp_path / (iso.TEMP_PREFIX + "inside")) is False
    assert iso.remove_session_dir(None) is False
    assert _real_state() == real_before


def test_the_directory_boundary_is_a_direct_child_of_the_system_temp_directory_judged_on_directories_that_exist(tmp_path, monkeypatch):
    """The rule: a session directory has the prefix AND sits directly under the system temp directory. Every directory here EXISTS, so a refusal can only be the rule's, never a missing path."""
    # (1) The real system temp directory. tmp_path lives below it (pytest's base temp), so a correctly named, existing directory made in tmp_path is nested too deep.
    nested = tmp_path / (iso.TEMP_PREFIX + "nested")
    nested.mkdir()
    assert nested.is_dir() and nested.name.startswith(iso.TEMP_PREFIX)
    assert iso.is_safe_to_remove(nested) is False  # right name, right kind of thing, wrong place

    # (2) Make tmp_path play the system temp directory, so the accepted case can also be exercised without creating anything outside it.
    monkeypatch.setattr(iso, "_system_temp", lambda: Path(os.path.realpath(tmp_path)))
    direct = tmp_path / (iso.TEMP_PREFIX + "direct")
    direct.mkdir()
    deeper = tmp_path / "deeper"
    deeper.mkdir()
    grandchild = deeper / (iso.TEMP_PREFIX + "grandchild")
    grandchild.mkdir()
    unprefixed = tmp_path / "plain_directory"
    unprefixed.mkdir()
    prefixed_file = tmp_path / (iso.TEMP_PREFIX + "a_file")
    prefixed_file.write_bytes(b"not a directory")

    assert iso.is_safe_to_remove(direct) is True  # prefix + direct child + a real directory
    assert iso.is_safe_to_remove(grandchild) is False  # one level too deep
    assert iso.is_safe_to_remove(unprefixed) is False  # right place, wrong name
    assert iso.is_safe_to_remove(prefixed_file) is False  # right name and place, but a file
    assert iso.is_safe_to_remove(tmp_path) is False  # the "system temp" itself is never removable

    # removal honours exactly the same boundary, and nothing else in tmp_path is touched
    assert iso.remove_session_dir(grandchild) is False and grandchild.exists()
    assert iso.remove_session_dir(direct) is True and not direct.exists()
    assert nested.exists() and deeper.exists() and unprefixed.exists() and prefixed_file.exists()  # only the one that satisfied the rule is gone


# ================================================= the session-end check that gives the isolation teeth ===============================================


def test_closing_the_redirected_connections_ignores_ones_that_are_closed_or_cannot_close():
    """(Only the helper is exercised: closing the session's real list here would break every later test that uses a module-level store.)"""

    class _Broken:
        def close(self):
            raise sqlite3.ProgrammingError("cannot close")

    open_one, closed_one = sqlite3.connect(":memory:"), sqlite3.connect(":memory:")
    closed_one.close()
    connections = [open_one, closed_one, _Broken()]

    iso._close_all(connections)

    assert connections == []
    with pytest.raises(sqlite3.ProgrammingError):
        open_one.execute("SELECT 1")  # it really was closed


def test_changes_between_reports_created_removed_and_changed_files_and_nothing_else():
    base = {"a.db": (10, 1), "b.db": (20, 2)}
    assert iso.changes_between(base, dict(base)) == []
    assert iso.changes_between(base, {**base, "a.db": (11, 1)}) == ["a.db: changed"]
    assert iso.changes_between(base, {**base, "b.db": (20, 3)}) == ["b.db: changed"]
    assert iso.changes_between(base, {**base, "c.db": (1, 1)}) == ["c.db: created"]
    assert iso.changes_between(base, {"a.db": (10, 1)}) == ["b.db: removed"]


def test_no_real_database_file_has_changed_so_far_and_every_redirect_went_to_the_temporary_directory():
    assert iso.session_changes() == []
    assert all(_is_session_temp(Path(actual)) for _, actual in iso.REDIRECTS)
    assert not REAL_MEMORY.exists() or str(REAL_MEMORY) in iso.snapshot_real_files()


def test_the_session_end_hook_fails_the_run_when_a_real_file_changed_and_not_otherwise(monkeypatch, capsys):
    conftest = importlib.import_module("conftest")
    cleaned = []
    monkeypatch.setattr(iso, "cleanup", lambda: cleaned.append(True))  # the real session directory must survive this test

    class _Session:
        exitstatus = 0

    quiet = _Session()
    monkeypatch.setattr(iso, "session_changes", lambda: [])
    conftest.pytest_sessionfinish(quiet, 0)
    assert quiet.exitstatus == 0 and cleaned == [True]

    loud = _Session()
    monkeypatch.setattr(iso, "session_changes", lambda: ["jarvis_memory.db: changed"])
    conftest.pytest_sessionfinish(loud, 0)
    assert loud.exitstatus == pytest.ExitCode.TESTS_FAILED
    assert "DATABASE ISOLATION FAILURE" in capsys.readouterr().err


# ======= the REAL session_changes() and the REAL session-end hook, on a throwaway repository that lives entirely inside tmp_path =======


def _scratch_repo(tmp_path: Path, monkeypatch) -> tuple[Path, dict[str, Path]]:
    """A 'repository' of protected-looking files inside tmp_path, watched by the check (and only by the check: `is_real_database`, i.e. the redirect, still guards the real files), with
    the 'session start' snapshot recorded now - exactly as `install()` records it."""
    repo = tmp_path / "scratch_repo"
    (repo / "tests").mkdir(parents=True)
    files = {"root_memory": repo / "jarvis_memory.db", "root_notes": repo / "jarvis_notes.db", "tests_memory": repo / "tests" / "jarvis_memory.db", "unwatched": repo / "other.db"}
    for path in files.values():
        path.write_bytes(b"original bytes")
    monkeypatch.setattr(iso, "_watched_folders", lambda: (repo, repo / "tests"))
    monkeypatch.setitem(iso._STATE, "snapshot", iso.snapshot_real_files())
    return repo, files


def test_the_session_end_check_really_watches_the_repository_root_and_the_tests_folder():
    """The tests below replace `_watched_folders` with a throwaway repository, so this pins the REAL one (it must not depend on whether this machine happens to have real database files)."""
    assert iso._watched_folders() == (iso.REPO_ROOT, iso.TESTS_DIR)
    assert iso.REPO_ROOT.is_dir() and (iso.REPO_ROOT / "app").is_dir() and iso.TESTS_DIR == iso.REPO_ROOT / "tests"


def test_session_changes_compares_the_files_now_with_the_snapshot_recorded_at_session_start(tmp_path, monkeypatch):
    repo, files = _scratch_repo(tmp_path, monkeypatch)
    assert set(iso._STATE["snapshot"]) == {str(files[name]) for name in ("root_memory", "root_notes", "tests_memory")}  # watches jarvis_*.db in both folders, and only those

    assert iso.session_changes() == []  # nothing happened: nothing is reported

    files["unwatched"].write_bytes(b"changed, but it is not a protected name")
    assert iso.session_changes() == []

    files["root_notes"].write_bytes(b"original bytes, now longer")  # a change of SIZE
    assert iso.session_changes() == ["jarvis_notes.db: changed"]
    assert iso.session_changes() == ["jarvis_notes.db: changed"]  # asking does not refresh the snapshot: still compared with the session start

    stat = files["tests_memory"].stat()
    os.utime(files["tests_memory"], ns=(stat.st_atime_ns, stat.st_mtime_ns + 5_000_000_000))  # a change of MODIFICATION TIME only, in the tests/ folder
    assert iso.session_changes() == ["jarvis_memory.db: changed", "jarvis_notes.db: changed"]

    (repo / "jarvis_tasks.db").write_bytes(b"new")  # a file that did not exist at the start
    files["root_memory"].unlink()  # and one that did
    assert iso.session_changes() == ["jarvis_memory.db: changed", "jarvis_memory.db: removed", "jarvis_notes.db: changed", "jarvis_tasks.db: created"]


def test_the_session_end_hook_uses_that_real_comparison_and_neither_restores_nor_deletes_anything(tmp_path, monkeypatch, capsys):
    conftest = importlib.import_module("conftest")
    repo, files = _scratch_repo(tmp_path, monkeypatch)
    monkeypatch.setattr(iso, "cleanup", lambda: None)  # (the real session directory must survive this test; cleanup has its own tests)

    class _Session:
        exitstatus = 0

    quiet = _Session()
    conftest.pytest_sessionfinish(quiet, 0)  # NOT patched: session_changes() runs for real
    assert quiet.exitstatus == 0 and "DATABASE ISOLATION FAILURE" not in capsys.readouterr().err

    files["root_notes"].write_bytes(b"modified after the snapshot")
    files["tests_memory"].unlink()
    loud = _Session()
    conftest.pytest_sessionfinish(loud, 0)

    assert loud.exitstatus == pytest.ExitCode.TESTS_FAILED
    report = capsys.readouterr().err
    assert "DATABASE ISOLATION FAILURE" in report and "jarvis_notes.db: changed" in report and "jarvis_memory.db: removed" in report
    assert files["root_notes"].read_bytes() == b"modified after the snapshot"  # detected, not restored
    assert not files["tests_memory"].exists()  # detected, not recreated
    assert files["root_memory"].exists() and files["unwatched"].exists()  # and nothing else was deleted
