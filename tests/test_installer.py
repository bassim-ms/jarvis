"""
The portable-installer foundation (installer/jarvis_setup.py). Everything runs in pytest's temporary folders: the real
Jarvis folder, its .env and its databases are never written. No network, no pip, no real virtual environment is created
(the commands are recorded, not run) - except the one test that packages the REAL repository into a temporary ZIP and
reads it back, to prove that what would be shipped holds no keys and no personal data. The only sockets opened are to a
throw-away server on 127.0.0.1 that this file starts itself (the running-Jarvis status check); a real running Jarvis is
never asked (see the autouse fixture below).
"""

import hashlib
import http.server
import importlib.util
import json
import os
import shutil
import socket
import sqlite3
import subprocess
import sys
import threading
import zipfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("jarvis_setup", REPO / "installer" / "jarvis_setup.py")
setup = importlib.util.module_from_spec(_spec)
sys.modules["jarvis_setup"] = setup  # dataclasses look their module up while the class is built
_spec.loader.exec_module(setup)

#: Fake keys, assembled at run time so this file itself never contains a key-shaped assignment the package scan would flag.
FAKE_GROQ = "gsk_" + "TESTONLY" * 6
FAKE_OPENROUTER = "sk-or-v1-" + "testonly" * 8
SETTING = "JARVIS_" + "GROQ_API_KEY"


@pytest.fixture(autouse=True)
def no_running_jarvis(monkeypatch):
    """`check` asks a running Jarvis on 127.0.0.1:8000 what it has seen of each key. Whether one happens to be running on
    this computer must never change a test's result: by default, every test sees "not running"."""
    monkeypatch.setattr(setup, "fetch_running_status", lambda port=setup.DEFAULT_PORT, timeout=2.0: None)


def installed_everything(python, packages):
    return {name: "1.0" for name in packages}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_jarvis(root: Path, *, env: str | None = None) -> Path:
    """A minimal Jarvis folder: app/, requirements.txt, .env.example (placeholders only), optionally a .env."""
    (root / "app").mkdir(parents=True)
    (root / "app" / "__init__.py").write_text("", encoding="utf-8")
    (root / "app" / "main.py").write_text("app = None\n", encoding="utf-8")
    (root / "requirements.txt").write_text("fastapi==0.115.6\nuvicorn[standard]==0.32.1\n# a comment\nhttpx==0.28.1\n", encoding="utf-8")
    (root / ".env.example").write_text(
        "# Jarvis settings\n" + SETTING + "=YOUR_GROQ_API_KEY_HERE\nJARVIS_GROQ_MODEL=openai/gpt-oss-120b\n", encoding="utf-8"
    )
    if env is not None:
        (root / ".env").write_text(env, encoding="utf-8")
    return root


def make_db(path: Path, rows: int) -> None:
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE notes (id INTEGER PRIMARY KEY, body TEXT)")
    connection.executemany("INSERT INTO notes (body) VALUES (?)", [(f"note {i}",) for i in range(rows)])
    connection.commit()
    connection.close()


def rows_in(path: Path) -> int:
    connection = sqlite3.connect(path)
    try:
        return connection.execute("SELECT COUNT(*) FROM notes").fetchone()[0]
    finally:
        connection.close()


# --- check -------------------------------------------------------------------------------------------------------------------


def test_check_on_a_fresh_folder_reports_what_is_missing(tmp_path):
    root = make_jarvis(tmp_path / "jarvis")

    findings = {f.name: f for f in setup.check_system(root, python_version=(3, 13, 1), free_bytes=50 * 1024**3)}

    assert findings["Python"].ok and findings["Disk space"].ok and findings["Jarvis files"].ok
    assert not findings["Virtual environment"].ok and "init" in findings["Virtual environment"].detail
    assert not findings["Configuration (.env)"].ok
    assert not findings["At least one AI provider"].ok
    assert all(not findings[f"AI provider: {e['provider']}"].required for e in setup.PROVIDER_KEYS)


def test_check_flags_an_old_python_and_a_full_disk(tmp_path):
    root = make_jarvis(tmp_path / "jarvis")

    findings = {f.name: f for f in setup.check_system(root, python_version=(3, 9, 7), free_bytes=100 * 1024**2)}

    assert not findings["Python"].ok and "3.11" in findings["Python"].detail
    assert not findings["Disk space"].ok


def test_check_reads_dependencies_from_the_environment_and_reports_missing_and_different_versions(tmp_path):
    root = make_jarvis(tmp_path / "jarvis", env=SETTING + "=" + FAKE_GROQ + "\n")
    python = setup.venv_python(root)
    python.parent.mkdir(parents=True)
    python.write_bytes(b"")

    def versions_of(interpreter, packages):
        assert interpreter == python
        return {"fastapi": "0.115.6", "uvicorn": "0.30.0", "httpx": None}

    findings = {f.name: f for f in setup.check_system(root, python_version=(3, 13, 0), free_bytes=10 * 1024**3,
                                                      versions_of=versions_of)}

    assert findings["Virtual environment"].ok
    assert not findings["Dependencies"].ok and "httpx" in findings["Dependencies"].detail
    assert "uvicorn 0.30.0 (wants 0.32.1)" in findings["Dependency versions"].detail
    # A key that is merely present is usable for readiness, but it is NOT reported as working.
    assert findings["AI provider: Groq"].mark == "unverified" and not findings["AI provider: Groq"].ok
    assert findings["At least one AI provider"].ok


def test_check_never_prints_a_key(tmp_path, capsys):
    root = make_jarvis(tmp_path / "jarvis", env=SETTING + "=" + FAKE_GROQ + "\n")

    setup.print_findings(setup.check_system(root, python_version=(3, 13, 0), free_bytes=10 * 1024**3))

    assert FAKE_GROQ not in capsys.readouterr().out


def test_requirements_are_parsed_with_extras_and_comments():
    assert setup.parse_requirements("uvicorn[standard]==0.32.1\n# x\npydantic-settings==2.7.0  # pinned\nnot a pin\n") == {
        "uvicorn": "0.32.1", "pydantic-settings": "2.7.0",
    }


def test_the_real_requirements_file_parses_completely():
    pins = setup.parse_requirements((REPO / "requirements.txt").read_text(encoding="utf-8"))
    assert {"fastapi", "uvicorn", "httpx", "pydantic-settings", "pytest"} <= set(pins)


# --- init --------------------------------------------------------------------------------------------------------------------


def test_init_creates_env_from_the_template_and_installs_through_recorded_commands(tmp_path):
    root = make_jarvis(tmp_path / "jarvis")
    commands = []

    done = setup.initialize(root, run=lambda command, check: commands.append(command), report=lambda message: None,
                            versions_of=installed_everything)

    assert (root / ".env").read_text(encoding="utf-8") == (
        "# Jarvis settings\n" + SETTING + "=\nJARVIS_GROQ_MODEL=openai/gpt-oss-120b\n"
    )  # the template, placeholder emptied, other settings kept
    assert commands[0][1:] == ["-m", "venv", str(root / "venv")]
    assert commands[1][1:] == ["-m", "pip", "install", "--disable-pip-version-check", "-r", str(root / "requirements.txt")]
    assert (root / "backups").is_dir()
    assert "created .env from .env.example (no keys in it yet)" in done
    assert "installed and verified 3 dependencies from requirements.txt" in done


def test_a_placeholder_never_counts_as_a_configured_key(tmp_path):
    root = make_jarvis(tmp_path / "jarvis", env=SETTING + "=YOUR_GROQ_API_KEY_HERE\nJARVIS_GEMINI_API_KEY=<paste>\n")

    findings = {f.name: f for f in setup.check_system(root, python_version=(3, 13, 0), free_bytes=10 * 1024**3)}

    assert not findings["AI provider: Groq"].ok and not findings["AI provider: Gemini"].ok
    assert setup.read_env_values(root / ".env")[SETTING] == ""


def test_init_never_touches_an_existing_env(tmp_path):
    root = make_jarvis(tmp_path / "jarvis", env=SETTING + "=" + FAKE_GROQ + "\nCUSTOM=1\n")
    before = digest(root / ".env")

    done = setup.initialize(root, install_dependencies=False)

    assert digest(root / ".env") == before
    assert "kept the existing .env" in done


def test_init_refuses_an_incomplete_package(tmp_path):
    root = make_jarvis(tmp_path / "jarvis")
    (root / ".env.example").unlink()

    with pytest.raises(setup.SetupError, match=r"not a complete Jarvis package: \.env\.example is missing") as failure:
        setup.initialize(root, install_dependencies=False)
    assert "Extract the whole Jarvis ZIP again" in failure.value.hint


# --- .env writing ------------------------------------------------------------------------------------------------------------


def test_env_values_are_replaced_in_place_appended_and_backed_up(tmp_path):
    env = tmp_path / ".env"
    original = "# keep this comment\n" + SETTING + "=old\nJARVIS_GROQ_MODEL=openai/gpt-oss-120b\n# " + SETTING + "=commented\n"
    env.write_text(original, encoding="utf-8")

    backup = setup.write_env_values(env, {SETTING: FAKE_GROQ, "JARVIS_NEW_SETTING": "x"}, now="20260101000000")

    assert backup.name == ".env.bak.20260101000000" and backup.read_text(encoding="utf-8") == original
    lines = env.read_text(encoding="utf-8").splitlines()
    assert lines == ["# keep this comment", SETTING + "=" + FAKE_GROQ, "JARVIS_GROQ_MODEL=openai/gpt-oss-120b",
                     "# " + SETTING + "=commented", "JARVIS_NEW_SETTING=x"]
    assert setup.read_env_values(env)[SETTING] == FAKE_GROQ


@pytest.mark.parametrize("value", ["a\nJARVIS_EVIL=1", "a\rb"])
def test_a_value_with_a_line_break_is_refused(tmp_path, value):
    env = tmp_path / ".env"
    env.write_text("A=1\n", encoding="utf-8")

    with pytest.raises(ValueError):
        setup.write_env_values(env, {SETTING: value})
    assert env.read_text(encoding="utf-8") == "A=1\n"


def test_a_bad_setting_name_is_refused(tmp_path):
    with pytest.raises(ValueError):
        setup.write_env_values(tmp_path / ".env", {"lower-case": "x"})


# --- keys --------------------------------------------------------------------------------------------------------------------


def scripted(answers):
    queue = list(answers)
    prompts = []

    def ask(prompt):
        prompts.append(prompt)
        return queue.pop(0) if queue else ""

    return ask, prompts


def test_keys_walk_the_providers_in_the_fixed_order_and_never_report_a_typed_value():
    ask, prompts = scripted([FAKE_GROQ, FAKE_OPENROUTER])
    said = []

    updates = setup.collect_keys({}, ask=ask, report=said.append)

    assert updates == {SETTING: FAKE_GROQ, "JARVIS_OPENROUTER_API_KEY": FAKE_OPENROUTER}
    order = [prompt.split(" (")[0].strip() for prompt in prompts]
    assert order == ["JARVIS_GROQ_API_KEY", "JARVIS_OPENROUTER_API_KEY", "JARVIS_GEMINI_API_KEY", "JARVIS_CLOUDFLARE_API_KEY",
                     "JARVIS_CLOUDFLARE_ACCOUNT_ID", "JARVIS_FREELLMAPI_API_KEY"]
    everything_said = "\n".join(said + prompts)
    assert FAKE_GROQ not in everything_said and FAKE_OPENROUTER not in everything_said


def test_enter_keeps_an_existing_key():
    ask, prompts = scripted([""] * 10)

    assert setup.collect_keys({SETTING: FAKE_GROQ}, ask=ask, report=lambda message: None) == {}
    assert "already set" in prompts[0]


def test_a_key_that_looks_wrong_is_asked_for_again_then_accepted_as_typed():
    ask, _ = scripted(["not-a-groq-key-123", "still-not-a-groq-key"])
    said = []

    updates = setup.collect_keys({}, ask=ask, report=said.append)

    assert updates[SETTING] == "still-not-a-groq-key"
    assert any("does not look right" in line for line in said)
    assert not any("not-a-groq-key" in line for line in said)


def test_a_key_that_looks_wrong_can_be_abandoned_with_enter():
    ask, _ = scripted(["typo", ""])

    assert SETTING not in setup.collect_keys({}, ask=ask, report=lambda message: None)


@pytest.mark.parametrize("name, value, prefix, problem", [
    ("X_API_KEY", "abc def ghi jkl", "", "space"),
    ("X_API_KEY", "abc12", "", "too short"),
    ("X_API_KEY", "sk-wrong-prefix-value", "gsk_", "gsk_"),
    ("JARVIS_CLOUDFLARE_ACCOUNT_ID", "not-hex-but-long-enough", "", "32"),
    ("JARVIS_CLOUDFLARE_ACCOUNT_ID", "0123456789abcdef0123456789ABCDEF", "", None),
    ("X_API_KEY", FAKE_GROQ, "gsk_", None),
])
def test_looks_wrong(name, value, prefix, problem):
    reason = setup.looks_wrong(name, value, prefix)
    assert (reason is None) if problem is None else (problem in reason)
    assert reason is None or value not in reason


# --- migrate -----------------------------------------------------------------------------------------------------------------


def make_old_install(root: Path) -> Path:
    root.mkdir(parents=True)
    make_db(root / "jarvis_memory.db", 25)
    make_db(root / "jarvis_tasks.db", 3)
    (root / "jarvis_pc_device.json").write_text('{"device_id": "pc"}', encoding="utf-8")
    (root / "user_profile.md").write_text("# User Profile\n", encoding="utf-8")
    (root / "personality.md").write_text("# Personality\n", encoding="utf-8")
    (root / ".env").write_text(SETTING + "=" + FAKE_GROQ + "\n", encoding="utf-8")
    (root / "unrelated.txt").write_text("not Jarvis data", encoding="utf-8")
    return root


def test_migrate_copies_data_into_a_new_install_and_verifies_each_database(tmp_path):
    old = make_old_install(tmp_path / "old")
    new = make_jarvis(tmp_path / "new")
    source_hashes = {path.name: digest(path) for path in old.iterdir()}

    report = setup.migrate(old, new)

    assert sorted(report.copied) == ["jarvis_memory.db", "jarvis_pc_device.json", "jarvis_tasks.db", "personality.md", "user_profile.md"]
    assert report.failed == {} and report.skipped_existing == [] and report.backup_folder is None
    assert rows_in(new / "jarvis_memory.db") == 25 and rows_in(new / "jarvis_tasks.db") == 3
    assert not (new / ".env").exists()  # keys only with --include-env
    assert not (new / "unrelated.txt").exists()
    assert not list(new.glob("*.partial"))
    assert {path.name: digest(path) for path in old.iterdir()} == source_hashes  # the old folder is never modified


def test_migrate_never_overwrites_existing_data_without_replace(tmp_path):
    old = make_old_install(tmp_path / "old")
    new = make_jarvis(tmp_path / "new")
    make_db(new / "jarvis_memory.db", 2)
    before = digest(new / "jarvis_memory.db")

    report = setup.migrate(old, new)

    assert "jarvis_memory.db" in report.skipped_existing
    assert digest(new / "jarvis_memory.db") == before and rows_in(new / "jarvis_memory.db") == 2


def test_migrate_with_replace_backs_up_the_existing_file_first(tmp_path):
    old = make_old_install(tmp_path / "old")
    new = make_jarvis(tmp_path / "new")
    make_db(new / "jarvis_memory.db", 2)

    report = setup.migrate(old, new, replace=True, now="20260101000000")

    assert report.replaced == ["jarvis_memory.db"]
    assert report.backup_folder == new.resolve() / "backups" / "20260101000000"
    assert rows_in(report.backup_folder / "jarvis_memory.db") == 2  # the replaced data is kept
    assert rows_in(new / "jarvis_memory.db") == 25


def test_migrate_brings_the_env_only_when_asked(tmp_path):
    old = make_old_install(tmp_path / "old")
    new = make_jarvis(tmp_path / "new")

    setup.migrate(old, new, include_env=True)

    assert setup.read_env_values(new / ".env")[SETTING] == FAKE_GROQ


def test_migrate_reports_a_broken_database_and_still_copies_the_rest(tmp_path):
    old = make_old_install(tmp_path / "old")
    (old / "jarvis_notes.db").write_bytes(b"this is not a database" * 100)
    new = make_jarvis(tmp_path / "new")

    report = setup.migrate(old, new)

    assert "jarvis_notes.db" in report.failed
    assert not (new / "jarvis_notes.db").exists() and not (new / "jarvis_notes.db.partial").exists()
    assert "jarvis_memory.db" in report.copied


def test_migrate_reads_a_database_that_still_has_a_write_ahead_log(tmp_path):
    """A database whose latest rows are still in its -wal file (Jarvis running, or not shut down cleanly)."""
    old = tmp_path / "old"
    old.mkdir()
    holder = sqlite3.connect(old / "jarvis_memory.db")
    holder.execute("PRAGMA journal_mode=WAL")
    holder.execute("PRAGMA wal_autocheckpoint=0")
    holder.execute("CREATE TABLE notes (id INTEGER PRIMARY KEY, body TEXT)")
    holder.executemany("INSERT INTO notes (body) VALUES (?)", [(str(i),) for i in range(7)])
    holder.commit()  # rows are in the WAL, not yet in the main file
    new = make_jarvis(tmp_path / "new")
    try:
        report = setup.migrate(old, new)
    finally:
        holder.close()

    assert report.copied == ["jarvis_memory.db"]
    assert rows_in(new / "jarvis_memory.db") == 7


def test_migrate_refuses_the_same_folder_and_a_missing_one(tmp_path):
    root = make_jarvis(tmp_path / "jarvis")
    with pytest.raises(setup.SetupError, match="the same"):
        setup.migrate(root, root)
    with pytest.raises(setup.SetupError, match="The old Jarvis folder was not found") as failure:
        setup.migrate(tmp_path / "nowhere", root)
    assert "in quotes if it has spaces" in failure.value.hint


def test_migrate_refuses_a_file_given_as_the_folder_and_a_folder_with_nothing_to_bring(tmp_path):
    root = make_jarvis(tmp_path / "jarvis")
    a_file = tmp_path / "notes.txt"
    a_file.write_text("x", encoding="utf-8")
    empty = tmp_path / "Documents"
    empty.mkdir()
    (empty / "unrelated.txt").write_text("not Jarvis data", encoding="utf-8")

    with pytest.raises(setup.SetupError, match="This is a file, not a folder"):
        setup.migrate(a_file, root)
    with pytest.raises(setup.SetupError, match="nothing to bring over") as failure:
        setup.migrate(empty, root)
    assert "Nothing was changed" in failure.value.hint
    assert sorted(path.name for path in root.iterdir()) == [".env.example", "app", "requirements.txt"]


# --- package -----------------------------------------------------------------------------------------------------------------


def make_full_folder(root: Path) -> Path:
    make_jarvis(root, env=SETTING + "=" + FAKE_GROQ + "\n")
    (root / "installer").mkdir()
    (root / "installer" / "jarvis_setup.py").write_text("print('setup')\n", encoding="utf-8")
    (root / "README.md").write_text("# Jarvis\nSet " + SETTING + " in .env.\n", encoding="utf-8")
    (root / "personality.md").write_text("# Personality\n", encoding="utf-8")
    (root / "user_profile.md").write_text("# User Profile: private\n", encoding="utf-8")
    (root / "Start-Jarvis.bat").write_text("@echo off\n", encoding="utf-8")
    make_db(root / "jarvis_memory.db", 1)
    (root / "jarvis_pc_device.json").write_text("{}", encoding="utf-8")
    (root / ".env.bak.20260101").write_text(SETTING + "=" + FAKE_GROQ + "\n", encoding="utf-8")
    for folder in ("venv/Lib", "sandbox", ".git", "backups/20260101", "app/__pycache__"):
        (root / folder).mkdir(parents=True, exist_ok=True)
        (root / folder / "x.txt").write_text("x", encoding="utf-8")
    (root / "app" / "jarvis.log").write_text("log", encoding="utf-8")
    (root / "stray-notes.txt").write_text("not part of Jarvis", encoding="utf-8")
    return root


def test_the_package_holds_jarvis_and_nothing_secret_or_personal(tmp_path):
    root = make_full_folder(tmp_path / "jarvis")
    out = tmp_path / "out" / "jarvis.zip"

    result = setup.build_package(out, root)

    with zipfile.ZipFile(out) as archive:
        names = set(archive.namelist())
        manifest = archive.read("jarvis/PACKAGE_MANIFEST.txt").decode()
        blob = b"".join(archive.read(name) for name in names)
    assert {"jarvis/app/main.py", "jarvis/installer/jarvis_setup.py", "jarvis/.env.example", "jarvis/requirements.txt",
            "jarvis/README.md", "jarvis/personality.md", "jarvis/Start-Jarvis.bat"} <= names
    for forbidden in ("jarvis/.env", "jarvis/.env.bak.20260101", "jarvis/jarvis_memory.db", "jarvis/jarvis_pc_device.json",
                      "jarvis/user_profile.md", "jarvis/stray-notes.txt", "jarvis/app/jarvis.log"):
        assert forbidden not in names
    assert not any(part in name for name in names for part in ("/venv/", "/sandbox/", "/.git/", "/backups/", "/__pycache__/"))
    assert FAKE_GROQ.encode() not in blob
    assert result["files"] == len(names) - 1 and "app/main.py" in manifest
    assert not (tmp_path / "out" / "jarvis.zip.partial").exists()


def test_packaging_refuses_a_file_holding_a_real_looking_key(tmp_path):
    root = make_full_folder(tmp_path / "jarvis")
    (root / ".env.example").write_text(SETTING + "=" + FAKE_GROQ + "\n", encoding="utf-8")
    out = tmp_path / "jarvis.zip"

    with pytest.raises(ValueError, match=r"\.env\.example"):
        setup.build_package(out, root)
    assert not out.exists()


@pytest.mark.parametrize("text, secret", [
    (SETTING + "=" + FAKE_GROQ, True),
    ("JARVIS_CLOUDFLARE_" + "API_KEY=abc123xyz", True),
    (SETTING + "=YOUR_GROQ_API_KEY_HERE", False),
    (SETTING + "=", False),
    (SETTING + "=<paste here>", False),
    ("  " + SETTING + "=" + FAKE_GROQ, False),  # not a setting line (indented prose / code)
    ("JARVIS_GROQ_MODEL=openai/gpt-oss-120b", False),
])
def test_looks_like_a_secret(text, secret):
    assert setup.looks_like_a_secret(text) is secret


def test_the_real_repository_packages_cleanly(tmp_path):
    """Package THIS repository into a temporary ZIP: it must build (no file trips the secret scan) and hold no .env, no
    database, no device credential and no user_profile.md. The real folder is only read."""
    out = tmp_path / "jarvis.zip"

    result = setup.build_package(out, REPO)

    with zipfile.ZipFile(out) as archive:
        names = archive.namelist()
        example = archive.read("jarvis/.env.example").decode("utf-8")
    assert result["files"] > 100
    assert "jarvis/app/main.py" in names and "jarvis/installer/jarvis_setup.py" in names
    assert not [n for n in names if n.endswith((".db", "/.env", "user_profile.md", "device.json")) or "/.env.bak" in n]
    assert not setup.looks_like_a_secret(example)


# --- launch ------------------------------------------------------------------------------------------------------------------


def test_launch_binds_to_this_computer_only(tmp_path):
    command = setup.launch_command(tmp_path, port=8123)

    assert command[1:] == ["-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8123"]
    assert "0.0.0.0" not in command


def test_the_command_line_check_exits_non_zero_when_not_ready(tmp_path, monkeypatch, capsys):
    root = make_jarvis(tmp_path / "jarvis")
    monkeypatch.setattr(setup, "INSTALL_ROOT", root)

    assert setup.main(["check"]) == 1
    assert "Not ready yet" in capsys.readouterr().out


def test_the_command_line_migrate_reports_without_printing_keys(tmp_path, monkeypatch, capsys):
    old = make_old_install(tmp_path / "old")
    root = make_jarvis(tmp_path / "jarvis")
    monkeypatch.setattr(setup, "INSTALL_ROOT", root)

    assert setup.main(["migrate", "--from", str(old), "--include-env"]) == 0
    output = capsys.readouterr().out
    assert "copied" in output and FAKE_GROQ not in output


# --- failures: one clear message, a reliable exit code, nothing half-done ------------------------------------------------------


def failing_on(step, error):
    """A recorded `run` that raises `error` for the command whose module is `step` ("venv" or "pip")."""
    commands = []

    def run(command, check):
        commands.append(command)
        if command[2] == step:
            raise error

    return run, commands


def test_a_failed_dependency_install_is_a_clear_setup_error_with_what_to_do(tmp_path):
    root = make_jarvis(tmp_path / "jarvis")
    run, commands = failing_on("pip", subprocess.CalledProcessError(1, ["python.exe", "-m", "pip"]))

    with pytest.raises(setup.SetupError) as failure:
        setup.initialize(root, run=run, report=lambda message: None, versions_of=installed_everything)

    assert str(failure.value) == "Installing the dependencies failed (python.exe stopped with exit code 1)."
    assert "Check the internet connection" in failure.value.hint
    if os.name == "nt":
        assert f"path is {len(str(root))} characters long" in failure.value.hint and "C:\\Jarvis" in failure.value.hint
    assert [command[2] for command in commands] == ["venv", "pip"]  # stopped right there
    assert not (root / "backups").exists()  # nothing after the failed step was done


def test_a_failed_virtual_environment_stops_before_pip(tmp_path):
    root = make_jarvis(tmp_path / "jarvis")
    run, commands = failing_on("venv", subprocess.CalledProcessError(1, ["python.exe", "-m", "venv"]))

    with pytest.raises(setup.SetupError, match=r"Creating the virtual environment failed \(.*exit code 1\)"):
        setup.initialize(root, run=run, report=lambda message: None)

    assert [command[2] for command in commands] == ["venv"]


def test_a_program_that_cannot_be_started_is_a_setup_error_too(tmp_path):
    root = make_jarvis(tmp_path / "jarvis")
    run, _ = failing_on("pip", FileNotFoundError(2, "The system cannot find the file specified"))

    with pytest.raises(setup.SetupError, match="Installing the dependencies failed: The system cannot find the file"):
        setup.initialize(root, run=run, report=lambda message: None)


def test_pip_saying_done_is_not_taken_on_trust(tmp_path):
    root = make_jarvis(tmp_path / "jarvis")

    with pytest.raises(setup.SetupError, match="still not installed: httpx, uvicorn"):
        setup.initialize(root, run=lambda command, check: None, report=lambda message: None,
                         versions_of=lambda python, packages: {"fastapi": "0.115.6", "httpx": None, "uvicorn": None})


def test_init_refuses_an_old_python_before_changing_anything(tmp_path):
    root = make_jarvis(tmp_path / "jarvis")
    commands = []

    with pytest.raises(setup.SetupError, match=r"This Python is 3\.9\.7; Jarvis needs 3\.11 or newer") as failure:
        setup.initialize(root, run=lambda command, check: commands.append(command), python_version=(3, 9, 7))

    assert "python.org" in failure.value.hint
    assert commands == [] and not (root / ".env").exists()


def test_init_refuses_a_package_without_requirements_before_changing_anything(tmp_path):
    root = make_jarvis(tmp_path / "jarvis")
    (root / "requirements.txt").unlink()

    with pytest.raises(setup.SetupError, match="requirements.txt is missing"):
        setup.initialize(root, run=lambda command, check: None)
    assert not (root / ".env").exists()


def test_init_in_a_folder_that_does_not_exist(tmp_path):
    with pytest.raises(setup.SetupError, match="The Jarvis folder was not found"):
        setup.initialize(tmp_path / "missing", install_dependencies=False)


def run_main(monkeypatch, capsys, root, argv):
    monkeypatch.setattr(setup, "INSTALL_ROOT", root)
    code = setup.main(argv)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_the_command_line_init_failure_exits_2_with_a_message_and_no_traceback(tmp_path, monkeypatch, capsys):
    root = make_jarvis(tmp_path / "jarvis")

    def broken_initialize(root, install_dependencies):
        raise setup.SetupError("Installing the dependencies failed (python.exe stopped with exit code 1).",
                               setup.dependency_failure_hint(root))

    monkeypatch.setattr(setup, "initialize", broken_initialize)

    code, out, err = run_main(monkeypatch, capsys, root, ["init"])

    assert code == setup.EXIT_FAILED == 2
    assert "Setup failed: Installing the dependencies failed" in err and "What to do:" in err
    assert "Traceback" not in out + err


def test_the_command_line_init_succeeds_with_0_even_before_any_key_is_typed(tmp_path, monkeypatch, capsys):
    """Keys are the NEXT step: a successful init must not look like a failure to Setup-Jarvis.bat."""
    root = make_jarvis(tmp_path / "jarvis")

    code, out, _ = run_main(monkeypatch, capsys, root, ["init", "--skip-dependencies"])

    assert code == setup.EXIT_OK
    assert "created .env from .env.example" in out and "Not ready yet" in out  # still reported, just not a failure


def test_the_command_line_migrate_from_a_missing_folder_exits_2_with_a_clear_message(tmp_path, monkeypatch, capsys):
    root = make_jarvis(tmp_path / "jarvis")

    code, out, err = run_main(monkeypatch, capsys, root, ["migrate", "--from", str(tmp_path / "old jarvis")])

    assert code == setup.EXIT_FAILED
    assert "Setup failed: The old Jarvis folder was not found:" in err and "old jarvis" in err
    assert "What to do: Give the folder that holds the old jarvis_*.db files" in err
    assert "Traceback" not in out + err


def test_the_command_line_migrate_with_a_failed_file_exits_2_after_copying_the_rest(tmp_path, monkeypatch, capsys):
    old = make_old_install(tmp_path / "old")
    (old / "jarvis_notes.db").write_bytes(b"this is not a database" * 100)
    root = make_jarvis(tmp_path / "jarvis")

    code, out, err = run_main(monkeypatch, capsys, root, ["migrate", "--from", str(old)])

    assert code == setup.EXIT_FAILED
    assert "FAILED jarvis_notes.db" in out and "1 file(s) could not be brought over" in err
    assert (root / "jarvis_memory.db").is_file()


def test_the_command_line_keys_without_an_env_exits_2(tmp_path, monkeypatch, capsys):
    root = make_jarvis(tmp_path / "jarvis")

    code, _, err = run_main(monkeypatch, capsys, root, ["keys"])

    assert code == setup.EXIT_FAILED and "There is no .env" in err and "init" in err


def test_the_command_line_keys_without_a_keyboard_exits_2_and_saves_nothing(tmp_path, monkeypatch, capsys):
    root = make_jarvis(tmp_path / "jarvis", env=SETTING + "=\n")
    before = digest(root / ".env")

    def no_keyboard(prompt):  # what getpass does when there is no console to read from
        raise EOFError

    monkeypatch.setattr(setup, "collect_keys",
                        lambda existing: _COLLECT_KEYS(existing, ask=no_keyboard, report=lambda message: None))

    code, _, err = run_main(monkeypatch, capsys, root, ["keys"])

    assert code == setup.EXIT_FAILED and "No keyboard input is available" in err and "nothing was saved" in err
    assert digest(root / ".env") == before and not list(root.glob(".env.bak.*"))


def test_ctrl_c_exits_130_with_a_message(tmp_path, monkeypatch, capsys):
    root = make_jarvis(tmp_path / "jarvis")

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(setup, "check_system", interrupted)

    code, _, err = run_main(monkeypatch, capsys, root, ["check"])

    assert code == setup.EXIT_CANCELLED == 130 and "Stopped (Ctrl+C)" in err


def test_an_unexpected_bug_still_exits_2_never_1_which_means_not_ready(tmp_path, monkeypatch, capsys):
    root = make_jarvis(tmp_path / "jarvis")

    def buggy(*args, **kwargs):
        raise KeyError("oops")

    monkeypatch.setattr(setup, "check_system", buggy)

    code, out, err = run_main(monkeypatch, capsys, root, ["check"])

    assert code == setup.EXIT_FAILED and "Setup failed: an unexpected KeyError" in err and "Traceback" not in out + err


def test_a_disk_error_is_reported_by_name(tmp_path, monkeypatch, capsys):
    root = make_jarvis(tmp_path / "jarvis")

    def read_only(*args, **kwargs):
        raise PermissionError(13, "Access is denied", str(root / ".env"))

    monkeypatch.setattr(setup, "initialize", read_only)

    code, _, err = run_main(monkeypatch, capsys, root, ["init"])

    assert code == setup.EXIT_FAILED and "Setup failed: Access is denied: " in err and ".env" in err


def test_a_usage_mistake_exits_2(capsys):
    with pytest.raises(SystemExit) as stopped:
        setup.main(["migrate"])  # --from is required
    assert stopped.value.code == setup.EXIT_FAILED


# --- Setup-Jarvis.bat / Start-Jarvis.bat: really run by cmd.exe, against a stand-in jarvis_setup.py ---------------------------

#: Records each command it is given, then exits with the code listed for it in codes.txt (0 when not listed).
STAND_IN_SETUP = """\
import sys
from pathlib import Path
here = Path(__file__).resolve().parent
with open(here / "calls.txt", "a", encoding="utf-8") as log:
    log.write(sys.argv[1] + "\\n")
codes = dict(line.split("=") for line in (here / "codes.txt").read_text(encoding="utf-8").split())
sys.exit(int(codes.get(sys.argv[1], "0")))
"""


def run_batch(tmp_path, batch, codes):
    folder = tmp_path / "jarvis"
    (folder / "installer").mkdir(parents=True)
    shutil.copy2(REPO / batch, folder / batch)
    (folder / "installer" / "jarvis_setup.py").write_text(STAND_IN_SETUP, encoding="utf-8")
    (folder / "installer" / "codes.txt").write_text("\n".join(f"{k}={v}" for k, v in codes.items()) + "\n", encoding="utf-8")
    result = subprocess.run(["cmd", "/c", str(folder / batch)], cwd=folder, stdin=subprocess.DEVNULL, capture_output=True,
                            text=True, timeout=120)
    calls_file = folder / "installer" / "calls.txt"
    calls = calls_file.read_text(encoding="utf-8").split() if calls_file.is_file() else []
    return result, calls


windows_only = pytest.mark.skipif(os.name != "nt" or shutil.which("cmd") is None, reason="cmd.exe batch files")


@windows_only
@pytest.mark.parametrize("codes, expected_calls, expected_code, expected_text", [
    ({}, ["init", "keys", "check"], 0, "To start Jarvis: double-click Start-Jarvis.bat"),
    # A failed install stops setup at once: keys are never asked for.
    ({"init": 2}, ["init"], 2, "Setup stopped at step 1 of 3 - environment and dependencies (exit code 2)"),
    # Any non-zero init stops it - e.g. a Python too old to even run the script exits 1.
    ({"init": 1}, ["init"], 1, "Setup stopped at step 1 of 3 - environment and dependencies (exit code 1)"),
    ({"init": 130}, ["init"], 130, "Setup stopped at step 1 of 3 - environment and dependencies (exit code 130)"),
    ({"keys": 2}, ["init", "keys"], 2, "Setup stopped at step 2 of 3 - API keys (exit code 2)"),
    ({"check": 1}, ["init", "keys", "check"], 1, "Setup finished, but Jarvis is not ready to start yet"),
    ({"check": 2}, ["init", "keys", "check"], 2, "Setup stopped at step 3 of 3 - final check (exit code 2)"),
])
def test_setup_jarvis_bat_stops_at_the_first_failing_step(tmp_path, codes, expected_calls, expected_code, expected_text):
    result, calls = run_batch(tmp_path, "Setup-Jarvis.bat", codes)

    assert calls == expected_calls
    assert result.returncode == expected_code
    assert expected_text in result.stdout


@windows_only
@pytest.mark.parametrize("codes, expected_code", [({}, 0), ({"launch": 1}, 1), ({"launch": 2}, 2)])
def test_start_jarvis_bat_passes_the_exit_code_on(tmp_path, codes, expected_code):
    result, calls = run_batch(tmp_path, "Start-Jarvis.bat", codes)

    assert calls == ["launch"] and result.returncode == expected_code


@windows_only
def test_setup_jarvis_bat_refuses_an_incomplete_folder(tmp_path):
    folder = tmp_path / "jarvis"
    folder.mkdir()
    shutil.copy2(REPO / "Setup-Jarvis.bat", folder / "Setup-Jarvis.bat")

    result = subprocess.run(["cmd", "/c", str(folder / "Setup-Jarvis.bat")], cwd=folder, stdin=subprocess.DEVNULL,
                            capture_output=True, text=True, timeout=60)

    assert result.returncode == 2 and "installer\\jarvis_setup.py is missing" in result.stdout


@pytest.mark.parametrize("batch", ["Setup-Jarvis.bat", "Start-Jarvis.bat"])
def test_the_batch_files_keep_windows_line_endings(batch):
    """cmd.exe can miss a `goto` label in a file with LF-only line endings; .gitattributes keeps these CRLF."""
    data = (REPO / batch).read_bytes()
    assert data.count(b"\n") == data.count(b"\r\n") > 0
    assert "*.bat text eol=crlf" in (REPO / ".gitattributes").read_text(encoding="utf-8")


# --- key status: never "valid" because a key is non-empty ------------------------------------------------------------------


@pytest.mark.parametrize("key_present, running, expected", [
    (False, None, ("no_key", "")),
    (False, {"groq": "authenticated"}, ("no_key", "")),
    (True, None, ("present_not_verified", "not_running")),
    (True, {"groq": "present_not_verified"}, ("present_not_verified", "not_used")),
    (True, {"groq": "no_key"}, ("present_not_verified", "restart")),  # key added after Jarvis started
    (True, {}, ("present_not_verified", "restart")),
    (True, {"groq": "authenticated"}, ("authenticated", "")),
    (True, {"groq": "rejected"}, ("rejected", "")),
    (True, {"groq": "unavailable"}, ("unavailable", "")),
    (True, {"groq": "something new"}, ("present_not_verified", "not_used")),
])
def test_provider_key_status(key_present, running, expected):
    assert setup.provider_key_status(key_present, running, "groq") == expected


def findings_for(root, running):
    return {f.name: f for f in setup.check_system(root, python_version=(3, 13, 0), free_bytes=10 * 1024**3,
                                                  running_status=running)}


def test_a_key_that_is_merely_present_is_never_shown_as_ok(tmp_path, capsys):
    """The regression: `check` used to print "[ok] AI provider: FreeLLMAPI: key set" for a key the provider rejects."""
    root = make_jarvis(tmp_path / "jarvis", env=SETTING + "=" + FAKE_GROQ + "\nJARVIS_FREELLMAPI_API_KEY=" + "f" * 40 + "\n")

    setup.print_findings(setup.check_system(root, python_version=(3, 13, 0), free_bytes=10 * 1024**3, running_status=None))

    out = capsys.readouterr().out
    groq_line = next(line for line in out.splitlines() if "AI provider: Groq" in line)
    assert "[ unverified]" in groq_line and "key present, not verified (Jarvis is not running" in groq_line
    assert "key set" not in out and "[         ok] AI provider" not in out
    assert "2 of 5 have a key (2 not verified)" in out


def test_check_shows_what_the_running_jarvis_has_seen(tmp_path, capsys):
    env = (SETTING + "=" + FAKE_GROQ + "\nJARVIS_OPENROUTER_API_KEY=" + FAKE_OPENROUTER + "\nJARVIS_GEMINI_API_KEY=" + "g" * 39
           + "\nJARVIS_FREELLMAPI_API_KEY=" + "f" * 40 + "\n")
    root = make_jarvis(tmp_path / "jarvis", env=env)
    running = {"groq": "authenticated", "openrouter": "unavailable", "gemini": "present_not_verified",
               "cloudflare": "no_key", "freellmapi": "rejected"}

    findings = findings_for(root, running)

    assert findings["AI provider: Groq"].mark == "ok" and findings["AI provider: Groq"].ok
    assert findings["AI provider: OpenRouter"].mark == "unavailable" and "not verified" in findings["AI provider: OpenRouter"].detail
    assert findings["AI provider: Gemini"].mark == "unverified" and "has not used this provider" in findings["AI provider: Gemini"].detail
    assert findings["AI provider: Cloudflare Workers AI"].mark == "no key"
    assert findings["AI provider: FreeLLMAPI"].mark == "REJECTED" and "refused the key" in findings["AI provider: FreeLLMAPI"].detail
    assert not findings["AI provider: FreeLLMAPI"].ok
    assert findings["At least one AI provider"].detail == (
        "4 of 5 have a key (1 authenticated, 1 not verified, 1 rejected, 1 unavailable)"
    )
    setup.print_findings(list(findings.values()))
    out = capsys.readouterr().out
    assert FAKE_GROQ not in out and FAKE_OPENROUTER not in out and "f" * 40 not in out


def test_every_key_refused_means_not_ready(tmp_path):
    root = make_jarvis(tmp_path / "jarvis", env=SETTING + "=" + FAKE_GROQ + "\n")

    finding = findings_for(root, {"groq": "rejected"})["At least one AI provider"]

    assert not finding.ok and "every key was refused; run: keys" in finding.detail


class _StatusHandler(http.server.BaseHTTPRequestHandler):
    body = b""
    paths: list = []

    def do_GET(self):  # noqa: N802 - the standard library's name
        type(self).paths.append(self.path)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(type(self).body)

    def log_message(self, *args):
        pass


def serve_once(body: bytes):
    handler = type("Handler", (_StatusHandler,), {"body": body, "paths": []})
    server = http.server.HTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, handler


def test_the_running_status_is_read_from_jarvis_on_127_0_0_1(tmp_path):
    body = json.dumps({"order": ["groq"], "providers": [{"name": "groq", "key_status": "rejected"}, {"no": "name"}]}).encode()
    server, handler = serve_once(body)
    try:
        status = _FETCH_RUNNING_STATUS(port=server.server_address[1], timeout=5)
    finally:
        server.shutdown()
        server.server_close()

    assert status == {"groq": "rejected"}
    assert handler.paths == ["/providers/status"]


def test_no_running_jarvis_or_a_stranger_on_the_port_means_not_running():
    server, _ = serve_once(b"<html>not jarvis</html>")
    try:
        assert _FETCH_RUNNING_STATUS(port=server.server_address[1], timeout=5) is None
    finally:
        server.shutdown()
        server.server_close()
    with socket.socket() as probe:  # a port nothing listens on
        probe.bind(("127.0.0.1", 0))
        free_port = probe.getsockname()[1]
    assert _FETCH_RUNNING_STATUS(port=free_port, timeout=2) is None


#: The real functions, kept before the autouse fixture (or a test) replaces the module attributes.
_FETCH_RUNNING_STATUS = setup.fetch_running_status
_COLLECT_KEYS = setup.collect_keys
