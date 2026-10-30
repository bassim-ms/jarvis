"""
Milestone 0, part two - the sandbox really is one, and the log never carries a payload:

  * Settings.data_files_outside_data_dir (app/config.py) names every data file that would NOT be inside JARVIS_DATA_DIR - an absolute
    path elsewhere, or a relative one climbing out with "..";
  * scripts/smoke_check.py refuses (exit 2) BEFORE importing the application when isolation is not guaranteed, and confines the
    Workspace Files folder and every switch as well;
  * app/logging_setup.py renders any exception as its type and code location only - never its text - so a user's message, a Job's
    content, a tool argument or a credential carried by an exception (or by the exception chained behind it) never reaches the log.
"""

import importlib.util
import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest

from app import logging_setup
from app.config import DATA_FILE_SETTINGS, Settings
from app.core import JarvisCore
from app.memory.sqlite_store import SQLiteMemoryStore
from app.providers.base import AIProvider

REPO_ROOT = Path(__file__).resolve().parent.parent
SMOKE_SCRIPT = REPO_ROOT / "scripts" / "smoke_check.py"

PRIVATE_MESSAGE = "please remember my diagnosis is PRIVATE-HEALTH-7731"
JOB_CONTENT = "objective: transfer savings to account PRIVATE-JOB-4410"
TOOL_ARGUMENT = "type_text PRIVATE-ARG-9902"
API_KEY = "AIzaSyFAKE-KEY-0123456789abcdef"
LEAKED_URL = f"https://generativelanguage.googleapis.com/v1beta/models/x:generateContent?key={API_KEY}"
SENSITIVE = ("PRIVATE-HEALTH-7731", "PRIVATE-JOB-4410", "PRIVATE-ARG-9902", API_KEY, "key=")


def _settings(**values) -> Settings:
    """Settings from explicit values only - never the developer's .env."""
    return Settings(_env_file=None, **values)


def _smoke_module():
    spec = importlib.util.spec_from_file_location("smoke_check_under_test", SMOKE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def restore_jarvis_loggers():
    """configure_logging changes process-wide loggers; put them back so later tests' caplog still sees records."""
    saved = {name: (logging.getLogger(name).handlers[:], logging.getLogger(name).level, logging.getLogger(name).propagate)
             for name in logging_setup.JARVIS_LOGGERS}
    yield
    for name, (handlers, level, propagate) in saved.items():
        logger = logging.getLogger(name)
        for handler in logger.handlers[:]:
            if handler not in handlers:
                logger.removeHandler(handler)
                handler.close()
        logger.setLevel(level)
        logger.propagate = propagate


def _read(log_file: Path) -> str:
    for name in logging_setup.JARVIS_LOGGERS:
        for handler in logging.getLogger(name).handlers:
            handler.flush()
    return log_file.read_text(encoding="utf-8")


def _assert_no_payload(text: str) -> None:
    for value in SENSITIVE:
        assert value not in text, value


# ============================================================ which files are outside ===================================================


def test_with_default_relative_paths_every_data_file_is_inside(tmp_path):
    assert _settings(data_dir=str(tmp_path / "sb")).data_files_outside_data_dir() == []


def test_an_absolute_path_elsewhere_is_named_and_one_inside_is_not(tmp_path):
    sandbox = tmp_path / "sb"
    placed = _settings(
        data_dir=str(sandbox),
        memory_db_path=str(tmp_path / "real" / "jarvis_memory.db"),
        tasks_db_path=str(sandbox / "deeper" / "jarvis_tasks.db"),
    )

    assert placed.data_files_outside_data_dir() == ["memory_db_path"]


def test_a_relative_path_climbing_out_is_named(tmp_path):
    placed = _settings(data_dir=str(tmp_path / "sb"), notes_db_path=os.path.join("..", "jarvis_notes.db"))

    assert placed.data_files_outside_data_dir() == ["notes_db_path"]


def test_the_data_folder_itself_is_not_a_file_inside_it(tmp_path):
    sandbox = tmp_path / "sb"
    placed = _settings(data_dir=str(sandbox), orchestration_db_path=str(sandbox))

    assert placed.data_files_outside_data_dir() == ["orchestration_db_path"]


def test_extra_paths_are_checked_and_an_empty_log_file_uses_no_file(tmp_path):
    placed = _settings(data_dir=str(tmp_path / "sb"), log_file="")

    outside = placed.data_files_outside_data_dir({"workspace_dir": str(tmp_path / "Documents" / "Jarvis Workspace")})

    assert outside == ["workspace_dir"]


def test_without_a_data_folder_nothing_is_isolated_so_everything_is_named():
    outside = _settings().data_files_outside_data_dir()

    assert set(outside) == {name for name in DATA_FILE_SETTINGS if getattr(_settings(), name)}


# ============================================================ the smoke check fails closed ==============================================


def _isolated_settings(sandbox: Path, **values) -> Settings:
    smoke = _smoke_module()
    environment = smoke.sandbox_environment(sandbox)
    return _settings(
        data_dir=environment["JARVIS_DATA_DIR"],
        workspace_dir=environment["JARVIS_WORKSPACE_DIR"],
        background_schedulers_enabled=False,
        self_dev_enabled=False,
        computer_input_enabled=False,
        notifications_desktop_enabled=False,
        **values,
    )


def test_the_smoke_environment_confines_everything_and_switches_everything_off(tmp_path):
    smoke = _smoke_module()
    sandbox = tmp_path / "sb"
    settings = _isolated_settings(sandbox)

    assert smoke.isolation_problems(settings, sandbox, settings.in_data_dir("jarvis_world_model.db"), repo_root=tmp_path / "repo") == []
    environment = smoke.sandbox_environment(sandbox)
    assert Path(environment["JARVIS_WORKSPACE_DIR"]).is_relative_to(sandbox)
    for switch in ("JARVIS_BACKGROUND_SCHEDULERS_ENABLED", "JARVIS_SELF_DEV_ENABLED", "JARVIS_COMPUTER_INPUT_ENABLED",
                   "JARVIS_NOTIFICATIONS_DESKTOP_ENABLED"):
        assert environment[switch] == "false"


@pytest.mark.parametrize("override, expected", [
    ({"memory_db_path": "ABSOLUTE_OUTSIDE"}, "memory_db_path resolves outside the sandbox"),
    ({"calendar_token_db_path": os.path.join("..", "jarvis_calendar_token.db")}, "calendar_token_db_path resolves outside the sandbox"),
    ({"workspace_dir": "ABSOLUTE_OUTSIDE"}, "workspace_dir resolves outside the sandbox"),
    ({"self_dev_enabled": True}, "self_dev_enabled is on"),
    ({"computer_input_enabled": True}, "computer_input_enabled is on"),
    ({"background_schedulers_enabled": True}, "background_schedulers_enabled is on"),
    ({"notifications_desktop_enabled": True}, "notifications_desktop_enabled is on"),
])
def test_any_gap_in_isolation_is_reported(tmp_path, override, expected):
    smoke = _smoke_module()
    sandbox = tmp_path / "sb"
    values = {k: (str(tmp_path / "elsewhere" / "x") if v == "ABSOLUTE_OUTSIDE" else v) for k, v in override.items()}
    settings = _isolated_settings(sandbox)
    for name, value in values.items():
        setattr(settings, name, value)

    problems = smoke.isolation_problems(settings, sandbox, settings.in_data_dir("jarvis_world_model.db"), repo_root=tmp_path / "repo")

    assert expected in problems


def test_a_world_model_path_outside_and_a_sandbox_holding_the_repository_are_reported(tmp_path):
    smoke = _smoke_module()
    sandbox = tmp_path / "sb"
    settings = _isolated_settings(sandbox)

    assert "world_model_db_path resolves outside the sandbox" in smoke.isolation_problems(
        settings, sandbox, str(tmp_path / "jarvis_world_model.db"), repo_root=tmp_path / "repo")
    assert any("contains) the repository" in p for p in smoke.isolation_problems(
        settings, sandbox, settings.in_data_dir("jarvis_world_model.db"), repo_root=sandbox / "repo"))


def _run_smoke(tmp_path, extra_env: dict, sandbox: Path):
    env = {**os.environ, **extra_env}
    return subprocess.run(
        [sys.executable, str(SMOKE_SCRIPT), "--data-dir", str(sandbox)],
        capture_output=True, text=True, env=env, timeout=120, cwd=str(REPO_ROOT),
    )


def test_the_script_refuses_an_absolute_database_path_before_starting_the_application(tmp_path):
    outside = tmp_path / "real" / "jarvis_memory.db"
    sandbox = tmp_path / "sb"

    run = _run_smoke(tmp_path, {"JARVIS_MEMORY_DB_PATH": str(outside)}, sandbox)

    assert run.returncode == 2, run.stdout + run.stderr
    assert "memory_db_path resolves outside the sandbox" in run.stdout
    assert "application was NOT started" in run.stdout
    assert not outside.exists() and not outside.parent.exists()
    assert list(sandbox.glob("*.db")) == []  # the application was never imported, so no database was opened anywhere


def test_the_script_refuses_a_path_climbing_out_of_the_sandbox(tmp_path):
    sandbox = tmp_path / "sb"

    run = _run_smoke(tmp_path, {"JARVIS_TASKS_DB_PATH": os.path.join("..", "escaped_tasks.db")}, sandbox)

    assert run.returncode == 2, run.stdout + run.stderr
    assert "tasks_db_path resolves outside the sandbox" in run.stdout
    assert not (tmp_path / "escaped_tasks.db").exists()


# ============================================================ exceptions without their text =============================================


def _raise_chained():
    """A cleaned-up error raised while handling one that carries a private message, a credential URL and a tool argument."""
    try:
        raise ConnectionError(f"POST {LEAKED_URL} failed while sending {PRIVATE_MESSAGE} / {TOOL_ARGUMENT}")
    except ConnectionError as inner:
        raise RuntimeError(f"provider failed for {JOB_CONTENT}") from inner


def test_an_exception_is_described_by_type_and_location_only():
    try:
        _raise_chained()
    except RuntimeError as exc:
        text = logging_setup.describe_exception(exc)

    _assert_no_payload(text)
    assert "builtins.RuntimeError (message withheld)" in text and "builtins.ConnectionError (message withheld)" in text
    assert "which was raised from:" in text
    assert "test_milestone0_isolation_and_log_privacy.py:" in text and "in _raise_chained" in text


@pytest.mark.parametrize("logger_name, call", [
    ("app.scheduled_execution.scheduler", lambda log: log.warning("scheduled jobs: could not handle %s", "sj-123", exc_info=True)),
    ("app.morning_brief.service", lambda log: log.exception("morning brief failed")),
    ("jarvis.scheduled_operations", lambda log: log.warning("scheduled job run-now failed", exc_info=True)),
])
def test_exc_info_and_log_exception_write_no_payload_to_the_file(tmp_path, restore_jarvis_loggers, logger_name, call):
    log_file = tmp_path / "jarvis.log"
    logging_setup.configure_logging(str(log_file), "DEBUG")
    log = logging.getLogger(logger_name)

    try:
        _raise_chained()
    except RuntimeError:
        call(log)
    text = _read(log_file)

    _assert_no_payload(text)
    assert "RuntimeError (message withheld)" in text and "ConnectionError (message withheld)" in text
    assert "in _raise_chained" in text  # still says WHERE it happened


def test_a_record_already_formatted_in_full_elsewhere_is_still_written_without_its_text(restore_jarvis_loggers):
    formatter = logging_setup.PayloadFreeFormatter(logging_setup.FORMAT)
    try:
        _raise_chained()
    except RuntimeError:
        record = logging.getLogger("app.x").makeRecord("app.x", logging.ERROR, __file__, 1, "failed", (), sys.exc_info())
    full = logging.Formatter().format(record)  # another handler's formatter caches the FULL text on the record
    assert API_KEY in full and record.exc_text

    ours = formatter.format(record)

    _assert_no_payload(ours)
    assert record.exc_text and API_KEY in record.exc_text  # the other handler's cache is left as it was


class _Provider(AIProvider):
    def generate_reply(self, message: str) -> str:
        return "fine"


class _LeakyDoor:
    def command_reply(self, message, conversation_id):
        raise ValueError(f"could not parse {message!r} with token {API_KEY}")


def test_a_failing_chat_step_at_debug_level_writes_where_but_not_what(tmp_path, restore_jarvis_loggers):
    log_file = tmp_path / "jarvis.log"
    logging_setup.configure_logging(str(log_file), "DEBUG")
    jarvis = JarvisCore(provider=_Provider(), memory=SQLiteMemoryStore(":memory:"))
    jarvis.morning_brief = _LeakyDoor()

    reply, _, _ = jarvis.handle_chat(f"good morning {PRIVATE_MESSAGE}")
    text = _read(log_file)

    assert reply == "fine"
    _assert_no_payload(text)
    assert "step=morning_brief error=ValueError" in text
    assert "ValueError (message withheld)" in text and "in command_reply" in text
    assert "answered_by=direct" in text and "faults=morning_brief:ValueError" in text
