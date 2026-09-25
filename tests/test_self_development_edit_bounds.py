"""
Tests for the Self-Development Engine's EDIT-STAGE bounding
(app/self_development/engine.py: `_content_is_syntactically_valid`,
`_is_change_proportionate`, and their use in `_edit_one_file`; plus
app/self_development/editor.py's own defense-in-depth `_check_syntax`
inside EditFileTool/SecurityEditFileTool).

Added after a live test showed an APPROVED, well-GROUNDED plan still
corrupting app/main.py at the edit stage: the model's raw
`propose_file_content` response dumped hundreds of lines of its own
reasoning into the file, made unrelated changes elsewhere, and
produced a much larger rewrite than the tiny requested change. Plan
grounding (which files/attributes a PLAN may name) and edit bounding
(whether a PROPOSED EDIT's actual content is trustworthy) are separate
concerns - this file covers the second one.

Every test runs against a throwaway temporary git repository (never
the actual Jarvis repository).
"""

import json
import subprocess

import pytest

from app.self_development.base import SessionStage
from app.self_development.editor import EditFileTool, SecurityEditFileTool
from app.self_development.engine import _content_is_syntactically_valid, _is_change_proportionate
from app.self_development.factory import create_default_self_development_engine
from app.providers.base import AIProvider
from app.tools.base import ToolResultStatus


def _init_repo(path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Tester"], cwd=path, check=True)
    (path / "app").mkdir()
    (path / "app" / "__init__.py").write_text("", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=path, check=True)


def _substantial_original_content(n: int = 30) -> str:
    """A file large enough (>= _MIN_LINES_FOR_PROPORTIONALITY_CHECK)
    that the proportionality check actually applies - a real, if
    synthetic, stand-in for something like the real app/main.py."""
    return "\n".join(f"line_{i} = {i}" for i in range(n)) + "\n"


def _plan_payload(**overrides) -> dict:
    payload = {
        "objective": "Add a /dev/status endpoint",
        "files_affected": [{"path": "app/main.py", "description": "add the endpoint"}],
        "dependencies": [],
        "tests_to_run": ["git status --porcelain"],
        "risks": [],
        "required_permissions": ["code_edit", "test_run"],
    }
    payload.update(overrides)
    return payload


class ScriptedProvider(AIProvider):
    def __init__(self, plan_payload: dict, file_content: str):
        self.plan_payload = plan_payload
        self.file_content = file_content

    def generate_reply(self, message: str) -> str:
        if "CURRENT CONTENT" in message:
            return self.file_content
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
def substantial_repo(tmp_path):
    _init_repo(tmp_path)
    original = _substantial_original_content()
    (tmp_path / "app" / "main.py").write_text(original, encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "add main.py"], cwd=tmp_path, check=True)
    return tmp_path, original


# -- 1. reasoning text must never be written into a source file -----------


def test_reasoning_text_is_never_written_into_the_source_file(substantial_repo):
    """The exact reported failure: the model's response is prose - its
    own reasoning - not code at all. Must be rejected before writing,
    and the original file must survive completely untouched."""
    repo, original = substantial_repo
    corrupted = (
        "I need to think about this carefully. The user wants a "
        "/dev/status endpoint. Let me reason step by step about how "
        "to implement this, considering the existing structure first.\n"
        + "\n".join(f"reasoning_step_{i} = 'thinking...'" for i in range(50))
    )
    # Deliberately invalid Python: the prose lines above are not
    # statements at all.
    engine = _make_engine(repo, ScriptedProvider(_plan_payload(), corrupted))

    session = engine.start("add a /dev/status endpoint")
    session = engine.approve_plan(session, approved=True)

    assert "app/main.py" not in session.edited_files
    assert any("not syntactically valid" in m for m in session.messages)
    assert (repo / "app" / "main.py").read_text(encoding="utf-8") == original


# -- 2. an unrelated, oversized rewrite is rejected ------------------------


def test_an_unrelated_oversized_rewrite_is_rejected(substantial_repo):
    """Syntactically VALID Python, but almost none of the original
    file survives and the new content has nothing to do with the
    tiny requested change - rejected as disproportionate, never a
    "trust it because it at least parses" free pass."""
    repo, original = substantial_repo
    unrelated_but_valid = "\n".join(f"unrelated_var_{i} = {i}" for i in range(60)) + "\n"
    engine = _make_engine(repo, ScriptedProvider(_plan_payload(), unrelated_but_valid))

    session = engine.start("add a /dev/status endpoint")
    session = engine.approve_plan(session, approved=True)

    assert "app/main.py" not in session.edited_files
    assert any("replaces far more of the file" in m for m in session.messages)
    assert (repo / "app" / "main.py").read_text(encoding="utf-8") == original


# -- 3. valid small edits still succeed ------------------------------------


def test_a_valid_small_edit_to_a_substantial_file_succeeds(substantial_repo):
    """The ordinary, expected case: the file's own original content is
    fully preserved, plus a small, real addition - never blocked by
    either new check."""
    repo, original = substantial_repo
    new_content = original + "def dev_status():\n    return {'stage': 'ok'}\n"
    engine = _make_engine(repo, ScriptedProvider(_plan_payload(), new_content))

    session = engine.start("add a /dev/status endpoint")
    session = engine.approve_plan(session, approved=True)

    assert "app/main.py" in session.edited_files
    assert (repo / "app" / "main.py").read_text(encoding="utf-8").strip() == new_content.strip()


def test_a_full_rewrite_of_a_small_stub_file_still_succeeds(tmp_path):
    """A tiny stub/placeholder file (below the proportionality check's
    own size floor) being fully replaced by real content is the
    ORDINARY first-edit shape this whole engine is built around - it
    must never be treated as a "disproportionate rewrite"."""
    _init_repo(tmp_path)
    (tmp_path / "app" / "greet.py").write_text("# empty\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "add stub"], cwd=tmp_path, check=True)

    engine = _make_engine(
        tmp_path,
        ScriptedProvider(
            _plan_payload(files_affected=[{"path": "app/greet.py", "description": "add it"}]),
            "def say_hello():\n    return 'hi'\n",
        ),
    )

    session = engine.start("add a greeting function")
    session = engine.approve_plan(session, approved=True)

    assert "app/greet.py" in session.edited_files
    assert "def say_hello" in (tmp_path / "app" / "greet.py").read_text(encoding="utf-8")


# -- 4. syntax-invalid edits are rejected (direct tool-level coverage) ----


def test_edit_file_tool_rejects_syntactically_invalid_python(tmp_path):
    (tmp_path / "app").mkdir()
    target = tmp_path / "app" / "widget.py"
    target.write_text("original = True\n", encoding="utf-8")

    tool = EditFileTool(str(tmp_path))
    result = tool.execute(path="app/widget.py", content="this is not python at all !!\n")

    assert result.status is ToolResultStatus.ERROR
    assert "valid Python" in result.message
    assert target.read_text(encoding="utf-8") == "original = True\n"


def test_security_edit_file_tool_rejects_syntactically_invalid_python(tmp_path):
    (tmp_path / "app").mkdir()
    target = tmp_path / "app" / "core.py"
    target.write_text("original = True\n", encoding="utf-8")

    tool = SecurityEditFileTool(str(tmp_path))
    result = tool.execute(path="app/core.py", content="reasoning text, not code\n")

    assert result.status is ToolResultStatus.ERROR
    assert target.read_text(encoding="utf-8") == "original = True\n"


def test_edit_file_tool_accepts_valid_python(tmp_path):
    (tmp_path / "app").mkdir()
    target = tmp_path / "app" / "widget.py"
    target.write_text("old = True\n", encoding="utf-8")

    tool = EditFileTool(str(tmp_path))
    result = tool.execute(path="app/widget.py", content="new_value = True\n")

    assert result.status is ToolResultStatus.SUCCESS
    assert target.read_text(encoding="utf-8") == "new_value = True\n"


def test_edit_file_tool_never_syntax_checks_a_non_python_file(tmp_path):
    """The syntax gate is Python-specific - an ordinary text/markdown
    file is written exactly as given, never rejected for "not being
    valid Python" (it was never supposed to be)."""
    (tmp_path / "docs").mkdir()
    target = tmp_path / "docs" / "notes.md"
    target.write_text("old notes\n", encoding="utf-8")

    tool = EditFileTool(str(tmp_path))
    result = tool.execute(path="docs/notes.md", content="Just some plain English notes.\n")

    assert result.status is ToolResultStatus.SUCCESS
    assert target.read_text(encoding="utf-8") == "Just some plain English notes.\n"


# -- 5. edits outside the approved plan are rejected -----------------------


def test_apply_plan_never_touches_a_file_outside_files_affected(tmp_path):
    """`_apply_plan` iterates ONLY `session.plan.files_affected` - a
    file the plan never named is structurally unreachable, never a
    matter of the model "deciding" not to touch it."""
    _init_repo(tmp_path)
    (tmp_path / "app" / "greet.py").write_text("# empty\n", encoding="utf-8")
    (tmp_path / "app" / "untouched.py").write_text("# leave me alone\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "add files"], cwd=tmp_path, check=True)

    engine = _make_engine(
        tmp_path,
        ScriptedProvider(
            _plan_payload(files_affected=[{"path": "app/greet.py", "description": "add it"}]),
            "def say_hello():\n    return 'hi'\n",
        ),
    )

    session = engine.start("add a greeting function")
    session = engine.approve_plan(session, approved=True)

    assert session.edited_files == ("app/greet.py",)
    assert (tmp_path / "app" / "untouched.py").read_text(encoding="utf-8") == "# leave me alone\n"


def test_repair_never_touches_a_file_outside_the_original_plan(tmp_path):
    """The bounded repair loop is ALSO scoped to the original plan's
    own files - a follow-up "fix" naming a different file is ignored,
    never applied (see tests/test_self_development_engine.py's own
    `test_repair_edit_outside_the_original_plan_scope_is_ignored` for
    the matching coverage at that layer)."""
    _init_repo(tmp_path)
    (tmp_path / "app" / "greet.py").write_text("# empty\n", encoding="utf-8")
    (tmp_path / "app" / "untouched.py").write_text("# leave me alone\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "add files"], cwd=tmp_path, check=True)

    sneaky_fix = _plan_payload(
        objective="fix by touching an unrelated file",
        files_affected=[{"path": "app/untouched.py", "description": "sneaky"}],
    )

    class TwoPhaseProvider(AIProvider):
        def generate_reply(self, message: str) -> str:
            if "CURRENT CONTENT" in message:
                return "def say_hello():\n    return 'hi'\n"
            if "BOUNDED FOLLOW-UP" in message:
                return json.dumps(sneaky_fix)
            return json.dumps(
                _plan_payload(
                    files_affected=[{"path": "app/greet.py", "description": "add it"}],
                    tests_to_run=["exit 1"],
                )
            )

    engine = _make_engine(tmp_path, TwoPhaseProvider())
    session = engine.start("add a greeting function")
    session = engine.approve_plan(session, approved=True)

    assert (tmp_path / "app" / "untouched.py").read_text(encoding="utf-8") == "# leave me alone\n"


# -- direct unit coverage of the two new helpers ---------------------------


def test_content_is_syntactically_valid_helper():
    assert _content_is_syntactically_valid("app/x.py", "x = 1\n") is True
    assert _content_is_syntactically_valid("app/x.py", "this is not python !!\n") is False
    assert _content_is_syntactically_valid("docs/x.md", "anything at all\n") is True


def test_is_change_proportionate_helper():
    original = _substantial_original_content()
    small_addition = original + "extra_line = True\n"
    unrelated_rewrite = "\n".join(f"unrelated_{i} = {i}" for i in range(60))

    assert _is_change_proportionate(original, small_addition) is True
    assert _is_change_proportionate(original, unrelated_rewrite) is False
    # Below the size floor - always proportionate (a stub being replaced).
    assert _is_change_proportionate("# empty\n", "def real_code():\n    pass\n") is True
