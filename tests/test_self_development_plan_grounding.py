"""
Tests for SelfDevelopmentEngine._find_ungrounded_file_paths / start()'s
own grounding check: the planner's LLM call can propose EDITING any
real, existing file, and can propose a brand-new file inside a
directory that already exists - but it can never invent an entirely
new directory/subsystem that doesn't exist in the real repository and
have that accepted as a valid plan.

Added after a live test showed the real planner producing a plan with
invented artifacts (a made-up schema file, a made-up deployment
directory, "staging" steps) for a request whose repo digest contained
none of that - i.e. an ungrounded, hallucinated plan.

Every test runs against a throwaway temporary git repository (never
the actual Jarvis repository).
"""

import json
import subprocess

import pytest

from app.providers.base import AIProvider
from app.self_development.base import SessionStage
from app.self_development.factory import create_default_self_development_engine


def _init_repo(path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Tester"], cwd=path, check=True)
    (path / "app").mkdir()
    (path / "app" / "main.py").write_text("# app\n", encoding="utf-8")
    (path / "greet.py").write_text("# empty\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=path, check=True)


def _plan_payload(**overrides) -> dict:
    payload = {
        "objective": "Add a greeting function",
        "files_affected": [{"path": "greet.py", "description": "add say_hello"}],
        "dependencies": [],
        "tests_to_run": ["git status --porcelain"],
        "risks": [],
        "required_permissions": ["code_edit", "test_run"],
    }
    payload.update(overrides)
    return payload


class ScriptedProvider(AIProvider):
    def __init__(self, plan_payload: dict):
        self.plan_payload = plan_payload

    def generate_reply(self, message: str) -> str:
        return json.dumps(self.plan_payload)


def _make_engine(repo_path, provider):
    return create_default_self_development_engine(
        provider,
        project_root=str(repo_path),
        max_repair_iterations=1,
        command_timeout_seconds=30.0,
        test_command="git status --porcelain",
    )


@pytest.fixture
def repo(tmp_path):
    _init_repo(tmp_path)
    return tmp_path


# -- legitimate, grounded plans are unaffected -----------------------------


def test_editing_a_real_existing_file_is_grounded(repo):
    engine = _make_engine(repo, ScriptedProvider(_plan_payload()))
    session = engine.start("add a greeting function")
    assert session.stage is SessionStage.PLAN_PROPOSED


def test_a_new_file_inside_an_existing_directory_is_grounded(repo):
    engine = _make_engine(
        repo,
        ScriptedProvider(
            _plan_payload(
                files_affected=[{"path": "app/widget.py", "description": "a new helper"}]
            )
        ),
    )
    session = engine.start("add a widget helper")
    assert session.stage is SessionStage.PLAN_PROPOSED


def test_a_new_file_at_the_repo_root_is_grounded(repo):
    engine = _make_engine(
        repo,
        ScriptedProvider(
            _plan_payload(files_affected=[{"path": "newfile.py", "description": "a new file"}])
        ),
    )
    session = engine.start("add a new top-level file")
    assert session.stage is SessionStage.PLAN_PROPOSED


# -- hallucinated / ungrounded plans are rejected --------------------------


@pytest.mark.parametrize(
    "invented_path",
    [
        "schemas/engine-status.schema.json",
        "deploy/status_endpoint.py",
        "docs/status-inventory.md",
        "staging/rollout.yaml",
    ],
)
def test_a_file_under_a_nonexistent_directory_is_rejected(repo, invented_path):
    engine = _make_engine(
        repo,
        ScriptedProvider(
            _plan_payload(files_affected=[{"path": invented_path, "description": "invented"}])
        ),
    )
    session = engine.start("add a /dev/status endpoint")

    assert session.stage is SessionStage.PLAN_REJECTED
    assert invented_path in session.messages[-1]
    assert session.plan is None


def test_reproduces_the_exact_reported_hallucination_and_rejects_it(repo):
    """The exact shape of files a live test's real planner call
    actually produced for "add a /dev/status endpoint" - none of
    these directories exist in this repo (or the real one)."""
    engine = _make_engine(
        repo,
        ScriptedProvider(
            _plan_payload(
                objective="Add a /dev/status endpoint with a full status schema",
                files_affected=[
                    {"path": "schemas/engine-status.schema.json", "description": "schema"},
                    {"path": "status_endpoint.py", "description": "endpoint"},
                    {"path": "deploy/rollout_steps.md", "description": "deployment steps"},
                ],
                risks=["requires a staging deployment"],
            )
        ),
    )

    session = engine.start(
        "self-develop: add a harmless /dev/status endpoint that returns the "
        "current self-development engine status"
    )

    assert session.stage is SessionStage.PLAN_REJECTED
    assert session.plan is None
    assert "not grounded" in session.messages[-1].lower()
    # Nothing was branched, edited, or committed for an ungrounded plan.
    assert session.dev_branch is None
    branches = subprocess.run(
        ["git", "branch"], cwd=repo, capture_output=True, text=True
    ).stdout
    assert "jarvis-selfdev" not in branches


def test_one_ungrounded_file_rejects_the_whole_plan_not_just_that_file(repo):
    """Fail-closed: a plan with one grounded and one ungrounded file is
    rejected ENTIRELY, never silently trimmed down to just the
    grounded file - the same discipline unexpected JSON fields/unknown
    permissions/protected paths already use elsewhere in this package."""
    engine = _make_engine(
        repo,
        ScriptedProvider(
            _plan_payload(
                files_affected=[
                    {"path": "greet.py", "description": "a real file"},
                    {"path": "made_up_dir/ghost.py", "description": "invented"},
                ]
            )
        ),
    )
    session = engine.start("add a greeting function and a ghost file")

    assert session.stage is SessionStage.PLAN_REJECTED
    assert session.plan is None


# -- protected system paths are exempt (a separate, fixed allowlist) -------


def test_a_protected_system_path_is_exempt_from_grounding(repo):
    """app/core.py-style protected paths are drawn from a fixed,
    already-enumerated allowlist (is_protected_system_path) - never a
    free-form path the LLM could hallucinate into existing - so
    grounding doesn't (and needn't) apply to them; SECURITY_EDIT's own
    approval gate is the real protection there."""
    engine = _make_engine(
        repo,
        ScriptedProvider(
            _plan_payload(
                files_affected=[{"path": "app/core.py", "description": "add a marker"}],
                required_permissions=["code_edit", "security_edit", "test_run"],
            )
        ),
    )
    session = engine.start("add a marker to app/core.py")

    assert session.stage is SessionStage.PLAN_PROPOSED  # not rejected as ungrounded
