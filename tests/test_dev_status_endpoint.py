"""
Tests for the `/dev/status` route (app/main.py:dev_status): it must
report the REAL, already-wired Self-Development Engine's own current
session state (`main.core.self_development_engine`, set up once at
startup) - never a freshly-instantiated engine with no history, and
never invented/placeholder data.

Follows the exact same "swap the real singleton for one pointed at a
throwaway temporary git repository, never the actual Jarvis
repository" convention tests/test_chat_self_development_integration.py
already established for this same singleton.
"""

import subprocess

from fastapi.testclient import TestClient

from app import main
from app.self_development.base import SessionStage
from app.self_development.factory import create_default_self_development_engine
from app.providers.base import AIProvider

client = TestClient(main.app, base_url="http://localhost")


class _PlanProvider(AIProvider):
    def __init__(self, plan: dict | None = None):
        self.plan = plan or {
            "objective": "Add a helper function",
            "files_affected": [{"path": "helper.py", "description": "add it"}],
            "dependencies": [],
            "tests_to_run": [],
            "risks": [],
            "required_permissions": ["code_edit"],
        }

    def generate_reply(self, message: str) -> str:
        import json

        return json.dumps(self.plan)


def _init_repo(path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Tester"], cwd=path, check=True)
    (path / "helper.py").write_text("# empty\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=path, check=True)


def _swap_engine(tmp_path, monkeypatch):
    _init_repo(tmp_path)
    engine = create_default_self_development_engine(
        _PlanProvider(),
        project_root=str(tmp_path),
        max_repair_iterations=1,
        command_timeout_seconds=30.0,
        test_command="git status --porcelain",
    )
    monkeypatch.setattr(main.core, "self_development_engine", engine)
    return engine


def test_dev_status_reports_none_configured_when_no_engine_is_wired(monkeypatch):
    monkeypatch.setattr(main.core, "self_development_engine", None)

    response = client.get("/dev/status")

    assert response.status_code == 200
    assert response.json() == {"configured": False, "sessions": []}


def test_dev_status_reports_an_empty_session_list_for_a_fresh_engine(tmp_path, monkeypatch):
    """The real, wired engine, with no session started yet - never
    fabricated placeholder session data."""
    _swap_engine(tmp_path, monkeypatch)

    response = client.get("/dev/status")

    assert response.status_code == 200
    assert response.json() == {"configured": True, "sessions": []}


def test_dev_status_reports_the_real_current_session_state(tmp_path, monkeypatch):
    """The core requirement this endpoint exists for: it must reflect
    the ACTUAL current session the real engine is tracking - not a
    freshly-instantiated engine with no memory of it."""
    engine = _swap_engine(tmp_path, monkeypatch)
    session = engine.start("add a helper function")

    response = client.get("/dev/status")

    assert response.status_code == 200
    data = response.json()
    assert data["configured"] is True
    assert data["sessions"] == [
        {"id": session.id, "stage": SessionStage.PLAN_PROPOSED.value, "dev_branch": None}
    ]


def test_dev_status_reflects_state_changes_across_requests(tmp_path, monkeypatch):
    """Proves this is a LIVE view of the real engine, not a snapshot
    taken once - approving the plan (which creates a real branch)
    between two GETs changes what the second one reports."""
    engine = _swap_engine(tmp_path, monkeypatch)
    session = engine.start("add a helper function")

    before = client.get("/dev/status").json()
    engine.approve_plan(session, approved=True)
    after = client.get("/dev/status").json()

    assert before["sessions"][0]["dev_branch"] is None
    assert after["sessions"][0]["dev_branch"] is not None
    assert after["sessions"][0]["stage"] != before["sessions"][0]["stage"]


def test_dev_status_is_read_only_and_never_mutates_engine_state(tmp_path, monkeypatch):
    engine = _swap_engine(tmp_path, monkeypatch)
    engine.start("add a helper function")
    before = dict(engine.sessions)

    client.get("/dev/status")

    assert engine.sessions == before


def test_dev_status_never_exposes_session_request_text_or_file_content(tmp_path, monkeypatch):
    """Harmless by construction: only id/stage/dev_branch are ever
    returned - never the request text, plan, diff, or file content."""
    engine = _swap_engine(tmp_path, monkeypatch)
    engine.start("a request that should never appear in the HTTP response")

    response = client.get("/dev/status")

    assert "a request that should never appear" not in response.text
