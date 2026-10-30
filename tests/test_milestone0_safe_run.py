"""
Milestone 0 - a safe first run you can see into:

  * JARVIS_DATA_DIR places every relative data file in one folder (app/config.py: Settings._place_data_files / in_data_dir);
  * JARVIS_BACKGROUND_SCHEDULERS_ENABLED=false starts no background worker (app/main.py: _lifespan);
  * JARVIS_SELF_DEV_ENABLED (default off) decides whether the Self-Development Engine is wired at all;
  * logging writes only Jarvis's own loggers (app/logging_setup.py), and every chat request leaves one trace line with no message text,
    plus a WARNING naming any best-effort step that failed (app/core.py: _RequestTrace).
"""

import asyncio
import logging
import os

import pytest

from app import logging_setup
from app.config import DATA_FILE_SETTINGS, Settings
from app.core import JarvisCore
from app.memory.sqlite_store import SQLiteMemoryStore
from app.providers.base import AIProvider


class _Provider(AIProvider):
    def generate_reply(self, message: str) -> str:
        return "fine"


def _settings(**values) -> Settings:
    """Settings from explicit values only - never the developer's .env."""
    return Settings(_env_file=None, **values)


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


# ============================================================ data folder ===============================================================


def test_without_a_data_dir_every_path_is_exactly_as_before():
    plain = _settings()

    assert plain.data_dir == ""
    assert plain.memory_db_path == "jarvis_memory.db"
    assert plain.orchestration_db_path == "jarvis_orchestration.db"
    assert plain.in_data_dir("jarvis_world_model.db") == "jarvis_world_model.db"


def test_a_data_dir_takes_every_relative_data_file_and_is_created(tmp_path):
    folder = tmp_path / "sandbox"

    placed = _settings(data_dir=str(folder))

    assert folder.is_dir()
    for name in DATA_FILE_SETTINGS:
        value = getattr(placed, name)
        assert os.path.dirname(value) == str(folder), name
    assert placed.in_data_dir("jarvis_world_model.db") == os.path.join(str(folder), "jarvis_world_model.db")


def test_an_absolute_path_is_never_moved_and_an_empty_log_file_stays_empty(tmp_path):
    elsewhere = str(tmp_path / "keep" / "memory.db")

    placed = _settings(data_dir=str(tmp_path / "sandbox"), memory_db_path=elsewhere, log_file="")

    assert placed.memory_db_path == elsewhere
    assert placed.log_file == ""
    assert placed.in_data_dir(elsewhere) == elsewhere


def test_the_data_dir_never_covers_the_identity_files_or_the_workspace(tmp_path):
    placed = _settings(data_dir=str(tmp_path / "sandbox"))

    assert placed.personality_file_path == "personality.md"
    assert placed.user_profile_file_path == "user_profile.md"
    assert "workspace_dir" not in DATA_FILE_SETTINGS and "self_dev_project_root" not in DATA_FILE_SETTINGS


def test_safe_defaults():
    plain = _settings()

    assert plain.self_dev_enabled is False
    assert plain.computer_input_enabled is False
    assert plain.background_schedulers_enabled is True  # unchanged behaviour unless switched off


# ============================================================ app wiring ================================================================


def test_the_running_app_wires_self_development_only_when_enabled():
    from app import main
    from app.config import settings

    assert (main.core.self_development_engine is not None) == settings.self_dev_enabled


def _run_lifespan(main) -> None:
    async def go():
        async with main._lifespan(main.app):
            pass

    asyncio.run(go())


@pytest.mark.parametrize("enabled", [True, False])
def test_the_background_switch_decides_whether_any_worker_starts(enabled, monkeypatch):
    from app import main

    calls = []
    monkeypatch.setattr(main.settings, "background_schedulers_enabled", enabled)
    monkeypatch.setattr(main, "configure_logging", lambda *a, **k: calls.append("logging"))
    for name in ("reminder_scheduler", "automation_engine", "job_engine"):
        worker = getattr(main, name)
        monkeypatch.setattr(worker, "start", lambda n=name: calls.append(f"{n}.start"))
        monkeypatch.setattr(worker, "stop", lambda n=name: calls.append(f"{n}.stop"))

    _run_lifespan(main)

    assert calls[0] == "logging"
    started = [c for c in calls if c.endswith(".start")]
    stopped = [c for c in calls if c.endswith(".stop")]
    if enabled:
        assert started == ["reminder_scheduler.start", "automation_engine.start", "job_engine.start"]
        assert stopped == ["job_engine.stop", "automation_engine.stop", "reminder_scheduler.stop"]
    else:
        assert started == [] and stopped == []


# ============================================================ logging ===================================================================


def test_logging_writes_jarvis_records_to_the_file_and_leaves_other_loggers_alone(tmp_path, restore_jarvis_loggers):
    log_file = tmp_path / "jarvis.log"
    root_handlers = logging.getLogger().handlers[:]

    logging_setup.configure_logging(str(log_file), "INFO")
    logging.getLogger("jarvis.chat").info("from jarvis")
    logging.getLogger("app.something").info("from app")
    logging.getLogger("httpx").warning("from httpx")
    for handler in logging.getLogger("jarvis").handlers:
        handler.flush()

    text = log_file.read_text(encoding="utf-8")
    assert "from jarvis" in text and "from app" in text
    assert "from httpx" not in text
    assert logging.getLogger().handlers == root_handlers


def test_configuring_twice_does_not_duplicate_lines(tmp_path, restore_jarvis_loggers):
    log_file = tmp_path / "jarvis.log"

    logging_setup.configure_logging(str(log_file), "INFO")
    logging_setup.configure_logging(str(log_file), "INFO")
    logging.getLogger("jarvis.chat").info("once")
    for handler in logging.getLogger("jarvis").handlers:
        handler.flush()

    assert log_file.read_text(encoding="utf-8").count("once") == 1


def test_an_unknown_level_falls_back_to_info_and_no_file_means_console_only(restore_jarvis_loggers):
    handlers = logging_setup.configure_logging("", "NOT-A-LEVEL")

    assert logging.getLogger("jarvis").level == logging.INFO
    assert all(not isinstance(h, logging.FileHandler) for h in handlers)


# ============================================================ request trace =============================================================


SECRET_WORDS = "my-private-sentence-xyz"


def test_every_chat_request_leaves_one_trace_line_without_the_message_text(caplog):
    jarvis = JarvisCore(provider=_Provider(), memory=SQLiteMemoryStore(":memory:"))

    with caplog.at_level(logging.INFO, logger="jarvis.chat"):
        reply, conversation_id, _ = jarvis.handle_chat(f"hello {SECRET_WORDS}")

    lines = [r.getMessage() for r in caplog.records if r.name == "jarvis.chat"]
    assert len(lines) == 1
    assert f"conversation={conversation_id}" in lines[0]
    assert "capability=direct" in lines[0] and "answered_by=direct" in lines[0] and "faults=-" in lines[0]
    assert SECRET_WORDS not in lines[0] and "fine" not in lines[0]


class _AnsweringDoor:
    def command_reply(self, message, conversation_id):
        return "the brief"


class _BrokenDoor:
    def command_reply(self, message, conversation_id):
        raise ValueError(f"it broke on {SECRET_WORDS}")


def test_the_trace_names_the_door_that_answered(caplog):
    jarvis = JarvisCore(provider=_Provider(), memory=SQLiteMemoryStore(":memory:"))
    jarvis.morning_brief = _AnsweringDoor()

    with caplog.at_level(logging.INFO, logger="jarvis.chat"):
        reply, _, _ = jarvis.handle_chat("good morning")

    assert reply.startswith("the brief")
    assert "answered_by=morning_brief" in [r.getMessage() for r in caplog.records if r.name == "jarvis.chat"][-1]


def test_a_failing_door_is_still_skipped_but_now_says_so_without_the_message(caplog):
    jarvis = JarvisCore(provider=_Provider(), memory=SQLiteMemoryStore(":memory:"))
    jarvis.morning_brief = _BrokenDoor()

    with caplog.at_level(logging.INFO, logger="jarvis.chat"):
        reply, _, _ = jarvis.handle_chat("good morning")

    assert reply == "fine"  # the chat still answered, exactly as before
    warnings = [r for r in caplog.records if r.name == "jarvis.chat" and r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "step=morning_brief" in warnings[0].getMessage() and "error=ValueError" in warnings[0].getMessage()
    assert all(SECRET_WORDS not in r.getMessage() for r in caplog.records)
    trace = [r.getMessage() for r in caplog.records if r.name == "jarvis.chat" and r.levelno == logging.INFO][-1]
    assert "faults=morning_brief:ValueError" in trace and "answered_by=direct" in trace
