"""
Tests for SelfDevelopmentEngine's plan-grounding checks
(app/self_development/engine.py: `_find_candidate_files`,
`_discover_symbols_in_file`, `_inspect_relevant_existing_code`,
`_find_hedged_language`, `_find_ungrounded_file_paths`): the planner's
LLM call can propose editing any real, existing file, and can propose
a brand-new file when its own description names a real, statically-
discovered capability - but it can never invent an unrelated file or
attribute.

Added after a live test on commit b279949 showed the real planner
proposing exactly this for "add a harmless /dev/status endpoint that
returns the current self-development engine status":
- Files: app/status.py (invented), app/main.py (real)
- Risk: "if the engine has no status attribute, return a generic
  placeholder"
Revised after a second review: the FIRST version of this fix read
every non-secret file's content on every `start()` call - correct but
slow and imprecise. This version instead: (1) narrows to a small set
of real `.py` files via a cheap, content-free PATH match against the
request's own keywords, ranked by how many keywords each path matches;
(2) parses ONLY that narrow set with `ast` to discover the REAL
classes/functions/attributes actually defined there (e.g.
DevelopmentSession's own `stage` field, generally - never hardcoded to
that one name); (3) requires a brand-new file's own description to
name one of those real, discovered capabilities. A general hedge/
placeholder-language check remains as a SEPARATE backstop, not the
primary mechanism.

Revised AGAIN after a third live test: a real, but wholly UNRELATED,
discovered symbol (GitClient's own `status_porcelain` - plain git-
status plumbing) "grounded" an invented `/dev/status` endpoint file
purely because both happened to share the bare, overly-common word
"status" - the description's OTHER matched words ("engine,"
"development," "self," "test") were every bit as generic. `_find_
ungrounded_file_paths` now refuses to count a match against
`_GENERIC_IDENTIFIER_FRAGMENTS` (a small, fixed, reviewable list) as
sufficient evidence BY ITSELF - a genuinely SPECIFIC word (e.g.
"porcelain," or a real attribute like "stage") is still required.

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


def _init_repo_with_calendar_sync(path) -> None:
    """A repo with one REAL, distinctive module - used to prove real
    symbol discovery/grounding (only meaningful once something real has
    actually been found via static inspection)."""
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Tester"], cwd=path, check=True)
    (path / "app").mkdir()
    (path / "app" / "calendar_sync.py").write_text(
        "class CalendarSync:\n"
        "    def __init__(self):\n"
        "        self.last_synced_at = None\n\n"
        "    def sync_events(self):\n"
        "        '''Sync calendar events.'''\n"
        "        return self.last_synced_at\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "-A"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=path, check=True)


def _init_repo_with_status_porcelain(path) -> None:
    """Mirrors the REAL app/self_development/git.py: a method literally
    named `status_porcelain` that has nothing to do with a "/dev/status"
    endpoint - used to prove a bare, overly-common fragment of a real
    but UNRELATED symbol never counts as grounding by itself."""
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Tester"], cwd=path, check=True)
    (path / "app").mkdir()
    (path / "app" / "repo_status_tools.py").write_text(
        "class RepoStatusTools:\n"
        "    def status_porcelain(self):\n"
        "        '''Return machine-readable git status output.'''\n"
        "        return ''\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "-A"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=path, check=True)


def _init_repo_with_development_session(path) -> None:
    """A DevelopmentSession-shaped fixture with a real `stage` field -
    the positive-evidence example the report specifically named."""
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Tester"], cwd=path, check=True)
    (path / "app").mkdir()
    (path / "app" / "dev_session.py").write_text(
        "from dataclasses import dataclass\n\n"
        "@dataclass\n"
        "class DevelopmentSession:\n"
        "    stage: str\n"
        "    request: str\n",
        encoding="utf-8",
    )
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


@pytest.fixture
def calendar_repo(tmp_path):
    _init_repo_with_calendar_sync(tmp_path)
    return tmp_path


@pytest.fixture
def status_repo(tmp_path):
    _init_repo_with_status_porcelain(tmp_path)
    return tmp_path


@pytest.fixture
def dev_session_repo(tmp_path):
    _init_repo_with_development_session(tmp_path)
    return tmp_path


# -- discovery itself: cheap, targeted, never a full-repo scan -------------


def test_an_unrelated_request_reads_nothing_at_all(calendar_repo):
    """No path in the repo relates to any keyword in this request - the
    engine must not read a single file's content, only the one listing
    call it already needs for the ordinary digest."""
    engine = _make_engine(calendar_repo, ScriptedProvider(_plan_payload()))
    read_calls: list[str] = []
    original_execute = engine.tools.execute

    def _spy(name, *args, **kwargs):
        if name == "self_dev_read_file":
            read_calls.append(kwargs.get("params", {}).get("path"))
        return original_execute(name, *args, **kwargs)

    engine.tools.execute = _spy  # type: ignore[method-assign]

    text, symbols = engine._inspect_relevant_existing_code("what is the weather like today")

    assert read_calls == []
    assert symbols == frozenset()
    assert "nothing was read" in text


def test_a_matching_request_only_reads_the_narrow_candidate_set(calendar_repo):
    """Only the path-matched file(s) are ever read - never every file in
    the repository."""
    engine = _make_engine(calendar_repo, ScriptedProvider(_plan_payload()))
    read_calls: list[str] = []
    original_execute = engine.tools.execute

    def _spy(name, *args, **kwargs):
        if name == "self_dev_read_file":
            read_calls.append(kwargs.get("params", {}).get("path"))
        return original_execute(name, *args, **kwargs)

    engine.tools.execute = _spy  # type: ignore[method-assign]

    text, symbols = engine._inspect_relevant_existing_code(
        "self-develop: sync calendar events more reliably"
    )

    assert read_calls == ["app/calendar_sync.py"]  # not app/main.py, not greet.py-equivalents
    assert "events" in symbols  # discovered from the real sync_events method
    assert "calendarsync" in symbols or "calendar" in symbols


def test_candidate_files_are_ranked_by_keyword_overlap_not_listing_order(calendar_repo):
    """A path matching SEVERAL of the request's own keywords must rank
    above one matching only a single, generic word shared by many
    unrelated files - added after a real run against this repository
    showed a generic word ("engine") crowding out the actually-relevant
    package before the ranking fix."""
    (calendar_repo / "app" / "generic_engine.py").write_text(
        "class GenericEngine:\n    pass\n", encoding="utf-8"
    )
    subprocess.run(
        ["git", "add", "-A"], cwd=calendar_repo, check=True
    )
    subprocess.run(
        ["git", "commit", "-q", "-m", "add unrelated engine"], cwd=calendar_repo, check=True
    )
    engine = _make_engine(calendar_repo, ScriptedProvider(_plan_payload()))

    candidates = engine._find_candidate_files(
        frozenset({"calendar", "sync", "events", "engine"})
    )

    assert candidates[0] == "app/calendar_sync.py"  # 3 keyword hits: calendar, sync, events


# -- legitimate, grounded plans are unaffected -----------------------------


def test_editing_a_real_existing_file_is_grounded(repo):
    engine = _make_engine(repo, ScriptedProvider(_plan_payload()))
    session = engine.start("add a greeting function")
    assert session.stage is SessionStage.PLAN_PROPOSED


def test_a_new_file_inside_an_existing_directory_is_grounded(repo):
    """Nothing in this sparse repo relates to "widget" at all - with no
    discovered symbols either way, the ORIGINAL, simpler bar (parent
    directory exists) still applies and this is accepted."""
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


def test_a_new_file_properly_grounded_in_a_discovered_symbol_is_accepted(calendar_repo):
    """Real capabilities WERE discovered (CalendarSync.sync_events) - a
    new file whose own description names one of them is accepted."""
    engine = _make_engine(
        calendar_repo,
        ScriptedProvider(
            _plan_payload(
                objective="Add better calendar sync retry handling",
                files_affected=[
                    {
                        "path": "app/calendar_retry.py",
                        "description": "retry helper for sync_events failures",
                    }
                ],
            )
        ),
    )
    session = engine.start("self-develop: sync calendar events more reliably")
    assert session.stage is SessionStage.PLAN_PROPOSED


# -- invented files/attributes are rejected --------------------------------


@pytest.mark.parametrize(
    "invented_path",
    ["schemas/engine-status.schema.json", "deploy/status_endpoint.py"],
)
def test_a_file_under_a_nonexistent_directory_is_rejected(repo, invented_path):
    """Pass 1: a brand-new file's parent directory must exist at all."""
    engine = _make_engine(
        repo,
        ScriptedProvider(
            _plan_payload(files_affected=[{"path": invented_path, "description": "invented"}])
        ),
    )
    session = engine.start("add something")

    assert session.stage is SessionStage.PLAN_REJECTED
    assert invented_path in session.messages[-1]
    assert session.plan is None


def test_an_invented_root_level_file_unrelated_to_anything_discovered_is_rejected(calendar_repo):
    """Real capabilities WERE discovered (calendar/sync/events), but
    this new file's own description has nothing to do with any of
    them - a bare "parent directory exists" check alone (repo root
    always exists) would have let this through; the stronger,
    symbol-based bar catches it."""
    engine = _make_engine(
        calendar_repo,
        ScriptedProvider(
            _plan_payload(
                objective="Sync calendar events more reliably",
                files_affected=[
                    {"path": "notes.md", "description": "unrelated scratch notes about lunch"}
                ],
            )
        ),
    )
    session = engine.start("self-develop: sync calendar events more reliably")

    assert session.stage is SessionStage.PLAN_REJECTED
    assert "notes.md" in session.messages[-1]
    assert session.plan is None


def test_an_invented_nested_file_unrelated_to_anything_discovered_is_rejected(calendar_repo):
    """Same as above but nested under a directory (app/) that DOES
    exist - proving this is a genuinely stronger bar than "the parent
    directory happens to exist," not just a rename of pass 1."""
    engine = _make_engine(
        calendar_repo,
        ScriptedProvider(
            _plan_payload(
                objective="Sync calendar events more reliably",
                files_affected=[
                    {"path": "app/ghost_notes.txt", "description": "totally unrelated scratchpad"}
                ],
            )
        ),
    )
    session = engine.start("self-develop: sync calendar events more reliably")

    assert session.stage is SessionStage.PLAN_REJECTED
    assert "app/ghost_notes.txt" in session.messages[-1]
    assert session.plan is None


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


# -- nonexistent attributes / placeholders are rejected (hedge backstop) ---


def test_reproduces_the_exact_reported_hallucination_and_rejects_it(repo):
    """The exact reported shape: an invented file PLUS the literal
    hedge text a live test's real planner call actually produced for
    "add a /dev/status endpoint"."""
    engine = _make_engine(
        repo,
        ScriptedProvider(
            _plan_payload(
                objective="Add a harmless /dev/status endpoint",
                files_affected=[
                    {"path": "app/status.py", "description": "expose engine status"},
                    {"path": "app/main.py", "description": "wire up the new endpoint"},
                ],
                risks=[
                    "if the engine has no status attribute, return a generic placeholder"
                ],
            )
        ),
    )

    session = engine.start(
        "self-develop: add a harmless /dev/status endpoint that returns the "
        "current self-development engine status"
    )

    assert session.stage is SessionStage.PLAN_REJECTED
    assert session.plan is None
    assert "hedge" in session.messages[-1].lower()
    assert session.dev_branch is None
    branches = subprocess.run(
        ["git", "branch"], cwd=repo, capture_output=True, text=True
    ).stdout
    assert "jarvis-selfdev" not in branches


@pytest.mark.parametrize(
    "hedge_text",
    [
        "if the engine has no status attribute, return a generic placeholder",
        "use a fallback value since the config field may not exist",
        "this is a stub implementation for a hypothetical setting",
        "best guess at the schema since it doesn't exist yet",
        "assuming it exists, read the dummy value",
    ],
)
def test_hedged_or_placeholder_language_in_risks_is_rejected(repo, hedge_text):
    engine = _make_engine(repo, ScriptedProvider(_plan_payload(risks=[hedge_text])))
    session = engine.start("add a greeting function")

    assert session.stage is SessionStage.PLAN_REJECTED
    assert session.plan is None


def test_ordinary_confident_language_is_never_flagged_as_hedging(repo):
    """Sanity check: an ordinary, confident, grounded plan - no hedge
    phrase anywhere - is never mistakenly rejected."""
    engine = _make_engine(
        repo,
        ScriptedProvider(
            _plan_payload(
                objective="Add a say_hello function to greet.py",
                risks=["none - purely additive"],
            )
        ),
    )
    session = engine.start("add a greeting function")

    assert session.stage is SessionStage.PLAN_PROPOSED


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


# -- a generic fragment of an UNRELATED real symbol is never sufficient ----


def test_an_unrelated_symbols_generic_fragment_does_not_ground_a_status_endpoint(status_repo):
    """The exact live failure, reproduced generally: a real symbol
    (`status_porcelain`, plain git-status plumbing) IS discovered, and
    its bare "status" fragment DOES textually overlap with a plausible
    description of a "/dev/status" endpoint - but that overlap alone
    must never be treated as grounding, because "status" is one of the
    overly-common fragments `_GENERIC_IDENTIFIER_FRAGMENTS` excludes."""
    engine = _make_engine(
        status_repo,
        ScriptedProvider(
            _plan_payload(
                objective="Add a harmless /dev/status endpoint",
                files_affected=[
                    {
                        "path": "app/dev_status.py",
                        "description": "expose the current status via a new endpoint",
                    }
                ],
            )
        ),
    )

    # Confirm status_porcelain really was discovered - the test only
    # means something if the evidence was genuinely found and still
    # correctly judged insufficient, not merely absent.
    _, discovered = engine._inspect_relevant_existing_code(
        "self-develop: add a /dev/status endpoint using repo status tools"
    )
    assert "status" in discovered

    session = engine.start(
        "self-develop: add a /dev/status endpoint using repo status tools"
    )

    assert session.stage is SessionStage.PLAN_REJECTED
    assert "app/dev_status.py" in session.messages[-1]
    assert session.plan is None


def test_an_unrelated_symbols_generic_fragment_still_blocks_even_with_several_matches(
    status_repo,
):
    """The live plan matched FIVE generic words at once ("status,"
    "engine," "development," "self," "test") - proving "multiple
    matches" alone was never a safe bar; every one of them being
    generic must still reject the plan."""
    engine = _make_engine(
        status_repo,
        ScriptedProvider(
            _plan_payload(
                objective="Add a harmless /dev/status endpoint",
                files_affected=[
                    {
                        "path": "app/dev_status.py",
                        "description": (
                            "add a test endpoint using the self-development "
                            "engine to report status"
                        ),
                    }
                ],
            )
        ),
    )

    session = engine.start(
        "self-develop: add a /dev/status endpoint using repo status tools"
    )

    assert session.stage is SessionStage.PLAN_REJECTED
    assert session.plan is None


# -- a specific, real symbol/attribute grounds the plan (positive case) ----


def test_a_specific_real_attribute_reference_grounds_the_plan(dev_session_repo):
    """DevelopmentSession.stage - a real, specific attribute, not one
    of the generic fragments - is sufficient grounding on its own."""
    engine = _make_engine(
        dev_session_repo,
        ScriptedProvider(
            _plan_payload(
                objective="Expose the development session's current stage",
                files_affected=[
                    {
                        "path": "app/status_endpoint.py",
                        "description": "expose the development session's current stage",
                    }
                ],
            )
        ),
    )

    session = engine.start(
        "self-develop: expose the development session stage via a new endpoint"
    )

    assert session.stage is SessionStage.PLAN_PROPOSED
    assert session.plan is not None


def test_the_unrelated_status_fragment_does_not_leak_grounding_across_repos(status_repo):
    """Sanity check on the mechanism itself, not the whole engine flow:
    a bare generic match is rejected by the acceptance rule directly,
    independent of any other check."""
    from app.self_development.base import DevelopmentCapability, DevelopmentPlan, PlannedFileChange

    engine = _make_engine(status_repo, ScriptedProvider(_plan_payload()))
    _, discovered = engine._inspect_relevant_existing_code(
        "self-develop: add a /dev/status endpoint using repo status tools"
    )
    plan = DevelopmentPlan(
        objective="x",
        files_affected=(
            PlannedFileChange(path="app/dev_status.py", description="expose current status"),
        ),
        required_permissions=(DevelopmentCapability.CODE_EDIT,),
    )

    ungrounded = engine._find_ungrounded_file_paths(plan, discovered)

    assert ungrounded == ("app/dev_status.py",)
