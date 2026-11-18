"""
Reliability and observability (app/diagnostics/; spec 452, 456, 457, 458, 460): database integrity checks, service health,
resources, verified backups and their later verification, and the dashboard's System view. Temporary folders only: the real
databases are never opened for writing, and no backup is ever written into the repository.
"""

import json
import re
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.dashboard.api import DashboardSources, build_dashboard_router
from app.diagnostics import BackupError, Diagnostics, check_database, create_backup, list_backups, verify_backup
from app.diagnostics.backup import MANIFEST
from app.host_guard import LoopbackHostMiddleware
from app.memory.sqlite_store import SQLiteMemoryStore
from app.notes.sqlite_store import SQLiteNoteStore
from app.orchestration.sqlite_interaction_store import SQLiteInteractionStore
from app.orchestration.sqlite_store import SQLiteJobStore
from app.reminders.sqlite_store import SQLiteReminderStore
from app.tasks.sqlite_store import SQLiteTaskStore
from app.tools.registry import ToolRegistry

DASHBOARD = {"X-Jarvis-Dashboard": "1"}
T0 = datetime(2026, 11, 18, 23, 0, tzinfo=timezone(timedelta(hours=4)))


def make_db(path: Path, rows: int) -> Path:
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE notes (id INTEGER PRIMARY KEY, body TEXT)")
    connection.executemany("INSERT INTO notes (body) VALUES (?)", [(f"note {i}",) for i in range(rows)])
    connection.commit()
    connection.close()
    return path


def digest(path: Path) -> bytes:
    return path.read_bytes()


@pytest.fixture
def home(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    memory = make_db(data / "jarvis_memory.db", 12)
    tasks = make_db(data / "jarvis_tasks.db", 3)
    (data / "personality.md").write_text("# Personality\n", encoding="utf-8")
    return data, [("memory_db_path", memory), ("tasks_db_path", tasks), ("notes_db_path", data / "jarvis_notes.db")]


# --- database integrity (457) --------------------------------------------------------------------------------------------


def test_a_sound_database_is_ok_with_its_tables_and_rows(home):
    data, databases = home

    check = check_database("memory_db_path", databases[0][1])

    assert check.ok is True and check.detail == "ok" and check.tables == 1 and check.rows == 12 and check.size_bytes > 0


def test_a_missing_database_is_not_a_failure_it_is_not_created_yet(home):
    _, databases = home

    check = check_database("notes_db_path", databases[2][1])

    assert check.exists is False and check.ok is None and check.detail == "not created yet"


def test_a_file_that_is_not_a_database_fails(tmp_path):
    bad = tmp_path / "jarvis_bad.db"
    bad.write_bytes(b"this is not a database" * 50)

    check = check_database("bad", bad)

    assert check.ok is False and "cannot be read as a database" in check.detail


def test_a_damaged_database_fails_its_check(tmp_path):
    path = make_db(tmp_path / "jarvis_damaged.db", 500)
    raw = bytearray(path.read_bytes())
    for offset in range(4096, min(len(raw), 4096 * 3)):  # scribble over the table's pages, keep the header
        raw[offset] = 0x5A
    path.write_bytes(bytes(raw))

    check = check_database("damaged", path)

    assert check.ok is False


def test_checking_never_changes_a_database(home):
    _, databases = home
    before = digest(databases[0][1])

    check_database("memory_db_path", databases[0][1])

    assert digest(databases[0][1]) == before


# --- verified backups (456) ------------------------------------------------------------------------------------------------


def test_a_backup_copies_every_database_and_identity_file_and_verifies_them(home, tmp_path):
    data, databases = home

    result = create_backup(databases, [data / "personality.md", data / "user_profile.md"], tmp_path / "backups", T0)

    assert result.folder.name == "20261118-230000"
    assert {f["file"] for f in result.files} == {"jarvis_memory.db", "jarvis_tasks.db", "personality.md"}
    assert set(result.skipped) == {"jarvis_notes.db", "user_profile.md"}  # not created yet - reported, not invented
    memory = next(f for f in result.files if f["file"] == "jarvis_memory.db")
    assert memory["rows"] == {"notes": 12} and memory["kind"] == "database" and len(memory["sha256"]) == 64
    manifest = json.loads((result.folder / MANIFEST).read_text(encoding="utf-8"))
    assert manifest["not_included"] == [".env (API keys)", "device credential"]
    assert verify_backup(result.folder) == {"ok": True, "problems": [], "files": 3}


def test_a_backup_never_includes_the_env_or_the_device_credential(home, tmp_path):
    data, databases = home
    (data / ".env").write_text("JARVIS_GROQ_API_KEY=secret\n", encoding="utf-8")
    (data / "jarvis_pc_device.json").write_text("{}", encoding="utf-8")

    result = create_backup(databases, [data / "personality.md"], tmp_path / "backups", T0)

    assert sorted(p.name for p in result.folder.iterdir()) == [MANIFEST, "jarvis_memory.db", "jarvis_tasks.db", "personality.md"]


def test_a_backup_never_changes_or_removes_an_original(home, tmp_path):
    data, databases = home
    before = {name: digest(path) for name, path in databases if path.exists()}

    create_backup(databases, [data / "personality.md"], tmp_path / "backups", T0)

    assert {name: digest(path) for name, path in databases if path.exists()} == before


def test_two_backups_in_the_same_second_never_overwrite_each_other(home, tmp_path):
    _, databases = home

    first = create_backup(databases, [], tmp_path / "backups", T0)
    second = create_backup(databases, [], tmp_path / "backups", T0)

    assert first.folder != second.folder and second.folder.name == "20261118-230000-2"
    assert verify_backup(first.folder)["ok"] and verify_backup(second.folder)["ok"]


def test_a_database_that_cannot_be_copied_stops_the_backup_and_leaves_no_partial_folder(home, tmp_path):
    data, databases = home
    bad = data / "jarvis_bad.db"
    bad.write_bytes(b"not a database" * 100)

    with pytest.raises(BackupError):
        create_backup(databases + [("bad", bad)], [], tmp_path / "backups", T0)

    assert list((tmp_path / "backups").iterdir()) == []


def test_verification_finds_a_changed_a_missing_and_a_damaged_file(home, tmp_path):
    data, databases = home
    result = create_backup(databases, [data / "personality.md"], tmp_path / "backups", T0)
    (result.folder / "personality.md").write_text("tampered", encoding="utf-8")
    (result.folder / "jarvis_tasks.db").unlink()

    report = verify_backup(result.folder)

    assert report["ok"] is False
    assert "personality.md: changed since the backup" in report["problems"]
    assert "jarvis_tasks.db: missing" in report["problems"]


@pytest.mark.parametrize("manifest", ["not json", json.dumps({"version": 99, "files": []}), json.dumps({"files": "x"})])
def test_a_backup_with_an_unreadable_manifest_is_not_trusted(tmp_path, manifest):
    folder = tmp_path / "20261118-230000"
    folder.mkdir()
    (folder / MANIFEST).write_text(manifest, encoding="utf-8")

    assert verify_backup(folder) == {"ok": False, "problems": ["the manifest is missing or unreadable"], "files": 0}


def test_a_manifest_cannot_point_outside_its_backup_folder(home, tmp_path):
    data, databases = home
    result = create_backup(databases, [], tmp_path / "backups", T0)
    manifest = json.loads((result.folder / MANIFEST).read_text(encoding="utf-8"))
    manifest["files"].append({"file": "../../data/jarvis_memory.db", "kind": "database", "sha256": "x"})
    (result.folder / MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")

    report = verify_backup(result.folder)

    assert report["ok"] is False and any("missing" in problem for problem in report["problems"])


def test_backups_are_listed_newest_first(home, tmp_path):
    _, databases = home
    create_backup(databases, [], tmp_path / "backups", T0)
    create_backup(databases, [], tmp_path / "backups", T0 + timedelta(hours=1))
    (tmp_path / "backups" / "not-a-backup").mkdir()

    listed = list_backups(tmp_path / "backups")

    assert [b["name"] for b in listed] == ["20261119-000000", "20261118-230000"]
    assert listed[0]["files"] == 2 and listed[0]["size_bytes"] > 0
    assert list_backups(tmp_path / "nowhere") == []


# --- the report (458, 452, 460) --------------------------------------------------------------------------------------------


def diagnostics(home, tmp_path, services=None, clock=lambda: T0):
    data, databases = home
    return Diagnostics(databases=databases, data_folder=data, backup_root=tmp_path / "backups",
                       services=services or {}, backup_files=[data / "personality.md"], clock=clock,
                       started_at=T0 - timedelta(minutes=5))


def test_the_report_is_ok_when_everything_is_and_says_so_plainly(home, tmp_path):
    d = diagnostics(home, tmp_path, services={"Job engine": lambda: (True, "running")})
    d.backup()

    report = d.report()

    assert report["status"] == "ok" and report["problems"] == [] and report["warnings"] == []
    assert report["uptime_seconds"] == 300 and report["backups"]["count"] == 1
    assert [s["name"] for s in report["services"]] == ["Job engine"]
    assert {db["file"]: db["ok"] for db in report["databases"]} == {"jarvis_memory.db": True, "jarvis_tasks.db": True,
                                                                     "jarvis_notes.db": None}
    assert report["resources"]["disk"]["free_bytes"] > 0 and report["resources"]["threads"] >= 1


def test_a_stopped_service_or_a_bad_database_makes_the_report_failing(home, tmp_path):
    data, _ = home
    (data / "jarvis_notes.db").write_bytes(b"garbage" * 100)
    d = diagnostics(home, tmp_path, services={"Reminder scheduler": lambda: (False, "not running"),
                                              "Automation engine": lambda: (None, "off by setting")})

    report = d.report()

    assert report["status"] == "failing"
    assert "Reminder scheduler: not running" in report["problems"]
    assert any(p.startswith("jarvis_notes.db: cannot be read") for p in report["problems"])
    assert not any("Automation engine" in p for p in report["problems"])  # not expected to run is not a problem


def test_a_probe_that_breaks_is_reported_not_raised(home, tmp_path):
    def broken():
        raise RuntimeError("boom")

    report = diagnostics(home, tmp_path, services={"AI providers": broken}).report()

    assert report["services"][0] == {"name": "AI providers", "ok": False, "detail": "could not be checked (RuntimeError)"}


def test_no_backup_yet_and_low_disk_are_warnings(home, tmp_path, monkeypatch):
    from app.diagnostics import checks

    monkeypatch.setattr(checks, "disk", lambda path: {"free_bytes": 200 * 1024**2, "total_bytes": 100 * 1024**3, "free_percent": 0.2})

    report = diagnostics(home, tmp_path).report()

    assert report["status"] == "warning" and "no backup yet" in report["warnings"]
    assert any(w.startswith("low disk space") for w in report["warnings"])


@pytest.mark.parametrize("name", ["..", "../data", "20261118-230000/../../x", "C:\\Windows", "", "latest"])
def test_verify_accepts_only_a_backup_name_never_a_path(home, tmp_path, name):
    with pytest.raises(ValueError):
        diagnostics(home, tmp_path).verify(name)


# --- the dashboard's System view ----------------------------------------------------------------------------------------------


@pytest.fixture
def system(home, tmp_path):
    jobs = SQLiteJobStore(str(tmp_path / "orchestration.db"))
    sources = DashboardSources(
        tasks=SQLiteTaskStore(str(tmp_path / "tasks.db")), reminders=SQLiteReminderStore(str(tmp_path / "reminders.db")),
        notes=SQLiteNoteStore(str(tmp_path / "notes.db")), jobs=jobs, interactions=SQLiteInteractionStore(jobs),
        tools=ToolRegistry(), memory=SQLiteMemoryStore(str(tmp_path / "memory.db")),
        provider_status=lambda: {"providers": []}, health=lambda: {"status": "ok"},
        diagnostics=diagnostics(home, tmp_path, services={"Job engine": lambda: (True, "running")}),
    )
    app = FastAPI()
    app.add_middleware(LoopbackHostMiddleware)
    app.include_router(build_dashboard_router(sources))
    return TestClient(app, base_url="http://127.0.0.1:8000"), sources


def test_the_system_view_reports_and_backs_up_and_verifies(system):
    http, _ = system

    assert http.get("/api/system").json()["status"] == "warning"  # no backup yet
    made = http.post("/api/system/backup", json={}, headers=DASHBOARD)
    assert made.status_code == 200 and made.json()["verified"] is True
    name = made.json()["name"]
    assert [b["name"] for b in http.get("/api/system/backups").json()["backups"]] == [name]
    verified = http.post(f"/api/system/backups/{name}/verify", json={}, headers=DASHBOARD)
    assert verified.json()["ok"] is True and verified.json()["files"] == 3
    assert http.get("/api/system").json()["status"] == "ok"


@pytest.mark.parametrize("path", ["/api/system/backup", "/api/system/backups/20261118-230000/verify"])
def test_backing_up_and_verifying_are_dashboard_only(system, path):
    http, sources = system

    assert http.post(path, json={}).status_code == 403
    assert http.post(path, json={}, headers={**DASHBOARD, "Origin": "https://evil.example"}).status_code == 403
    assert list_backups(sources.diagnostics.backup_root) == []


def test_verifying_an_unknown_or_malformed_backup_name(system):
    http, _ = system

    assert http.post("/api/system/backups/20991231-000000/verify", json={}, headers=DASHBOARD).status_code == 404
    assert http.post("/api/system/backups/not-a-name/verify", json={}, headers=DASHBOARD).status_code == 400


def test_without_diagnostics_the_view_says_so():
    from app.dashboard.api import DashboardSources as Sources

    assert "diagnostics" in Sources.__dataclass_fields__


def test_the_backups_folder_is_never_committed():
    root = Path(__file__).resolve().parent.parent
    ignored = (root / ".gitignore").read_text(encoding="utf-8")

    assert re.search(r"^backups/$", ignored, re.MULTILINE)


def test_process_memory_is_read_on_this_platform():
    import sys

    from app.diagnostics.checks import process_memory_bytes

    memory = process_memory_bytes()
    if sys.platform == "win32":
        assert isinstance(memory, int) and memory > 10 * 1024**2  # a Python process with FastAPI loaded uses more than 10 MB
    else:
        assert memory is None or memory > 0


def test_the_real_app_reports_on_its_own_databases_and_services_without_writing_anything():
    from app import main

    http = TestClient(main.app, base_url="http://127.0.0.1:8000")

    report = http.get("/api/system").json()

    assert report["available"] is True and report["status"] in ("ok", "warning", "failing")
    files = {db["file"] for db in report["databases"]}
    assert {"jarvis_memory.db", "jarvis_tasks.db", "jarvis_orchestration.db", "jarvis_world_model.db",
            "jarvis_provider_stats.db"} <= files
    assert [s["name"] for s in report["services"]] == ["Reminder scheduler", "Automation engine", "Job engine", "AI providers"]
    assert main.diagnostics.backup_root.name == "backups"
    # This test never asks the real app to back up: that would write copies into the repository's backups/ folder.
