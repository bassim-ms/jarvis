"""
Tests for the Self-Development Engine's EDIT-STAGE bounding
(app/self_development/engine.py: `_apply_file_edits`,
`_content_is_syntactically_valid`, `_is_change_proportionate`, and
their use in `_edit_one_file`; plus app/self_development/editor.py's
own defense-in-depth `_check_syntax` inside EditFileTool/
SecurityEditFileTool).

History:
- A live test showed an APPROVED, well-GROUNDED plan still corrupting
  app/main.py at the edit stage: the model's raw `propose_file_content`
  response (a "produce the complete new file" contract) dumped
  hundreds of lines of its own reasoning into the file. A first fix
  added syntax validation and a similarity-ratio backstop.
- A SECOND live test then showed that fix still wasn't enough: a
  syntactically VALID, still-plausible-looking full-file rewrite
  deleted an unrelated existing endpoint, reformatted unrelated code,
  and invented unrequested models - similar enough overall to stay
  above the similarity threshold.
- The actual fix: `propose_file_content` (whole-file rewrite) is GONE.
  `propose_file_edits` (app/self_development/planner.py) asks for a
  small set of EXACT find-and-replace edits instead, and
  `_apply_file_edits` (engine.py) applies them itself against the REAL
  current content - never the model's own claim of what was there.
  Unrelated code the model never quotes can never be touched AT ALL,
  structurally. Syntax validation and the similarity-ratio check
  remain as a second, independent layer on the FINAL result.

Every test runs against a throwaway temporary git repository (never
the actual Jarvis repository).
"""

import json
import subprocess

import pytest

from app.self_development.base import SessionStage
from app.self_development.editor import EditFileTool, SecurityEditFileTool
from app.self_development.engine import (
    _apply_file_edits,
    _content_is_syntactically_valid,
    _is_change_proportionate,
)
from app.self_development.factory import create_default_self_development_engine
from app.self_development.planner import DevelopmentPlanningError, DevelopmentPlanner
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
    """Plan/fix requests get `plan_payload` (JSON). File-edit requests
    get an EXPLICIT `edits_payload` if given (a raw JSON string, for
    tests that need to control the exact edit shape); otherwise a
    single whole-file find/replace edit is built automatically from
    the REAL current content the prompt shows -> `file_content` - the
    ordinary "replace the whole (small/stub) file" case."""

    def __init__(
        self,
        plan_payload: dict,
        file_content: str | None = None,
        edits_payload: str | None = None,
    ):
        self.plan_payload = plan_payload
        self.file_content = file_content
        self.edits_payload = edits_payload

    def generate_reply(self, message: str) -> str:
        if "CURRENT CONTENT" in message:
            if self.edits_payload is not None:
                return self.edits_payload
            current = message.split("CURRENT CONTENT (untrusted data):\n", 1)[1]
            return json.dumps({"edits": [{"old_text": current, "new_text": self.file_content}]})
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
    """The exact FIRST reported failure: the model's response is prose
    - its own reasoning - not code at all. Must be rejected before
    writing, and the original file must survive completely untouched."""
    repo, original = substantial_repo
    anchor = "line_29 = 29\n"
    reasoning_text = (
        "I need to think about this carefully. The user wants a "
        "/dev/status endpoint. Let me reason step by step about how "
        "to implement this, considering the existing structure first."
    )
    edits_payload = json.dumps({"edits": [{"old_text": anchor, "new_text": reasoning_text}]})
    engine = _make_engine(repo, ScriptedProvider(_plan_payload(), edits_payload=edits_payload))

    session = engine.start("add a /dev/status endpoint")
    session = engine.approve_plan(session, approved=True)

    assert "app/main.py" not in session.edited_files
    assert any("not syntactically valid" in m for m in session.messages)
    assert (repo / "app" / "main.py").read_text(encoding="utf-8") == original


# -- 2. unrelated code changes are rejected ---------------------------------


def test_an_edit_quoting_the_whole_file_as_a_disguised_rewrite_is_rejected(substantial_repo):
    """Syntactically VALID Python, framed as a single "exact match"
    edit whose old_text is the ENTIRE file - the second live failure's
    actual shape: a full-file rewrite dressed up as one big edit,
    deleting/reformatting everything along the way. Rejected by the
    size caps before it's ever applied."""
    repo, original = substantial_repo
    rewritten = "\n".join(f"unrelated_var_{i} = {i}" for i in range(40)) + "\n"
    edits_payload = json.dumps({"edits": [{"old_text": original, "new_text": rewritten}]})
    engine = _make_engine(repo, ScriptedProvider(_plan_payload(), edits_payload=edits_payload))

    session = engine.start("add a /dev/status endpoint")
    session = engine.approve_plan(session, approved=True)

    assert "app/main.py" not in session.edited_files
    assert any("oversized" in m or "didn't match" in m for m in session.messages)
    assert (repo / "app" / "main.py").read_text(encoding="utf-8") == original


def test_an_edit_naming_text_that_does_not_exist_in_the_file_is_rejected(substantial_repo):
    """A hallucinated old_text - text that simply isn't in the real
    file - is rejected outright, never guessed at or fuzzy-matched."""
    repo, original = substantial_repo
    edits_payload = json.dumps(
        {"edits": [{"old_text": "this text does not exist in the file", "new_text": "x = 1\n"}]}
    )
    engine = _make_engine(repo, ScriptedProvider(_plan_payload(), edits_payload=edits_payload))

    session = engine.start("add a /dev/status endpoint")
    session = engine.approve_plan(session, approved=True)

    assert "app/main.py" not in session.edited_files
    assert (repo / "app" / "main.py").read_text(encoding="utf-8") == original


def test_an_ambiguous_edit_matching_more_than_once_is_rejected(tmp_path):
    """old_text appearing MORE than once in the file is ambiguous -
    which occurrence did the model mean? Never guessed; rejected."""
    _init_repo(tmp_path)
    content = "def helper():\n    return 1\n\n\ndef helper():\n    return 1\n"
    (tmp_path / "app" / "dup.py").write_text(content, encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "add dup"], cwd=tmp_path, check=True)

    edits_payload = json.dumps(
        {"edits": [{"old_text": "def helper():\n    return 1\n", "new_text": "def helper():\n    return 2\n"}]}
    )
    engine = _make_engine(
        tmp_path,
        ScriptedProvider(
            _plan_payload(files_affected=[{"path": "app/dup.py", "description": "fix it"}]),
            edits_payload=edits_payload,
        ),
    )

    session = engine.start("fix the helper")
    session = engine.approve_plan(session, approved=True)

    assert "app/dup.py" not in session.edited_files
    assert (tmp_path / "app" / "dup.py").read_text(encoding="utf-8") == content


def test_several_small_edits_that_add_up_to_too_much_are_rejected(tmp_path):
    """No single edit is individually oversized, but their COMBINED
    old_text covers most of the file - rejected in aggregate, not just
    per edit (a live test showed a single disguised rewrite; several
    smaller ones stitched together would defeat a purely per-edit cap)."""
    _init_repo(tmp_path)
    # 8 padded lines (~1,660 chars total) - quoting 6 of them (the
    # per-file edit-count cap) is well over the 30% aggregate ratio
    # cap, even though no SINGLE edit is anywhere near the per-edit
    # size cap.
    lines = [f"line_{i} = '{'x' * 200}'\n" for i in range(8)]
    original = "".join(lines)
    (tmp_path / "app" / "main.py").write_text(original, encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "add main.py"], cwd=tmp_path, check=True)

    edits = [{"old_text": lines[i], "new_text": f"# replaced: {lines[i]}"} for i in range(6)]
    edits_payload = json.dumps({"edits": edits})
    engine = _make_engine(tmp_path, ScriptedProvider(_plan_payload(), edits_payload=edits_payload))

    session = engine.start("add a /dev/status endpoint")
    session = engine.approve_plan(session, approved=True)

    assert "app/main.py" not in session.edited_files
    assert (tmp_path / "app" / "main.py").read_text(encoding="utf-8") == original


# -- 3. valid small edits still succeed ------------------------------------


def test_a_valid_small_edit_to_a_substantial_file_succeeds(substantial_repo):
    """The ordinary, expected case: the file's own original content is
    fully preserved everywhere except the explicitly-quoted anchor,
    plus a small, real addition."""
    repo, original = substantial_repo
    anchor = "line_29 = 29\n"
    new_text = anchor + "\ndef dev_status():\n    return {'stage': 'ok'}\n"
    edits_payload = json.dumps({"edits": [{"old_text": anchor, "new_text": new_text}]})
    engine = _make_engine(repo, ScriptedProvider(_plan_payload(), edits_payload=edits_payload))

    session = engine.start("add a /dev/status endpoint")
    session = engine.approve_plan(session, approved=True)

    assert "app/main.py" in session.edited_files
    result = (repo / "app" / "main.py").read_text(encoding="utf-8")
    assert result == original.replace(anchor, new_text)
    # Everything before the anchor is untouched, byte for byte.
    assert result.startswith(original.split(anchor)[0])


def test_a_full_rewrite_of_a_small_stub_file_still_succeeds(tmp_path):
    """A tiny stub/placeholder file being fully replaced by real
    content is the ORDINARY first-edit shape this whole engine is
    built around - the whole-content-as-one-edit case still works."""
    _init_repo(tmp_path)
    (tmp_path / "app" / "greet.py").write_text("# empty\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "add stub"], cwd=tmp_path, check=True)

    engine = _make_engine(
        tmp_path,
        ScriptedProvider(
            _plan_payload(files_affected=[{"path": "app/greet.py", "description": "add it"}]),
            file_content="def say_hello():\n    return 'hi'\n",
        ),
    )

    session = engine.start("add a greeting function")
    session = engine.approve_plan(session, approved=True)

    assert "app/greet.py" in session.edited_files
    assert "def say_hello" in (tmp_path / "app" / "greet.py").read_text(encoding="utf-8")


def test_a_brand_new_file_is_created_via_the_whole_file_edit_form(tmp_path):
    """A file that doesn't exist at all yet - the ONLY case an edit's
    old_text is allowed to be empty."""
    _init_repo(tmp_path)

    edits_payload = json.dumps(
        {"edits": [{"old_text": "", "new_text": "def dev_status():\n    return {'stage': 'ok'}\n"}]}
    )
    engine = _make_engine(
        tmp_path,
        ScriptedProvider(
            _plan_payload(files_affected=[{"path": "app/status.py", "description": "new file"}]),
            edits_payload=edits_payload,
        ),
    )

    session = engine.start("add a status module")
    session = engine.approve_plan(session, approved=True)

    assert "app/status.py" in session.edited_files
    assert "dev_status" in (tmp_path / "app" / "status.py").read_text(encoding="utf-8")


def test_an_empty_old_text_edit_is_rejected_for_a_file_that_already_has_content(substantial_repo):
    """The whole-file form (`old_text == ""`) must never be usable to
    silently blow away an EXISTING file's real content."""
    repo, original = substantial_repo
    edits_payload = json.dumps({"edits": [{"old_text": "", "new_text": "x = 1\n"}]})
    engine = _make_engine(repo, ScriptedProvider(_plan_payload(), edits_payload=edits_payload))

    session = engine.start("add a /dev/status endpoint")
    session = engine.approve_plan(session, approved=True)

    assert "app/main.py" not in session.edited_files
    assert (repo / "app" / "main.py").read_text(encoding="utf-8") == original


# -- 6. malformed edits are rejected cleanly --------------------------------


def test_malformed_json_edit_response_is_rejected(substantial_repo):
    repo, original = substantial_repo
    engine = _make_engine(
        repo, ScriptedProvider(_plan_payload(), edits_payload="not json at all")
    )

    session = engine.start("add a /dev/status endpoint")
    session = engine.approve_plan(session, approved=True)

    assert "app/main.py" not in session.edited_files
    assert (repo / "app" / "main.py").read_text(encoding="utf-8") == original


def test_a_raw_full_file_string_instead_of_the_edits_envelope_is_rejected(substantial_repo):
    """The OLD `propose_file_content` shape (raw text, no JSON
    envelope at all) must never be silently accepted."""
    repo, original = substantial_repo
    engine = _make_engine(
        repo,
        ScriptedProvider(_plan_payload(), edits_payload="def dev_status():\n    return {}\n"),
    )

    session = engine.start("add a /dev/status endpoint")
    session = engine.approve_plan(session, approved=True)

    assert "app/main.py" not in session.edited_files
    assert (repo / "app" / "main.py").read_text(encoding="utf-8") == original


# -- 7. syntax validation after applying the edit ---------------------------


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


# -- edits outside the approved plan are rejected ---------------------------


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
            file_content="def say_hello():\n    return 'hi'\n",
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
                current = message.split("CURRENT CONTENT (untrusted data):\n", 1)[1]
                return json.dumps(
                    {"edits": [{"old_text": current, "new_text": "def say_hello():\n    return 'hi'\n"}]}
                )
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


# -- direct unit coverage of the core helpers -------------------------------


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


def test_apply_file_edits_helper_happy_path():
    current = "a = 1\nb = 2\nc = 3\n"
    edits = ({"old_text": "b = 2\n", "new_text": "b = 20\n"},)
    assert _apply_file_edits(current, edits) == "a = 1\nb = 20\nc = 3\n"


def test_apply_file_edits_helper_rejects_a_not_found_old_text():
    assert _apply_file_edits("a = 1\n", ({"old_text": "z = 9\n", "new_text": "z = 10\n"},)) is None


def test_apply_file_edits_helper_rejects_an_ambiguous_old_text():
    current = "x = 1\nx = 1\n"
    edits = ({"old_text": "x = 1\n", "new_text": "x = 2\n"},)
    assert _apply_file_edits(current, edits) is None


def test_apply_file_edits_helper_applies_multiple_edits_in_order():
    current = "a = 1\nb = 2\nc = 3\n"
    edits = (
        {"old_text": "a = 1\n", "new_text": "a = 10\n"},
        {"old_text": "c = 3\n", "new_text": "c = 30\n"},
    )
    assert _apply_file_edits(current, edits) == "a = 10\nb = 2\nc = 30\n"


def test_apply_file_edits_helper_new_file_form():
    assert _apply_file_edits("", ({"old_text": "", "new_text": "x = 1\n"},)) == "x = 1\n"


def test_apply_file_edits_helper_rejects_non_whole_file_form_for_a_new_file():
    assert _apply_file_edits("", ({"old_text": "x", "new_text": "y"},)) is None


def test_apply_file_edits_helper_rejects_too_many_edits():
    edits = tuple({"old_text": f"line_{i}", "new_text": f"x_{i}"} for i in range(20))
    assert _apply_file_edits("line_0 line_1 line_2\n" * 20, edits) is None


def test_apply_file_edits_helper_rejects_oversized_single_edit():
    huge = "x" * 10_000
    current = f"before\n{huge}\nafter\n"
    edits = ({"old_text": huge, "new_text": "y" * 10_000},)
    assert _apply_file_edits(current, edits) is None


# -- prompt-level coverage of the new edit contract -------------------------


class _FakeProvider(AIProvider):
    def __init__(self, reply: str):
        self.reply = reply
        self.last_prompt: str | None = None

    def generate_reply(self, message: str) -> str:
        self.last_prompt = message
        return self.reply


def test_propose_file_edits_prompt_asks_for_surgical_edits_not_a_full_rewrite():
    """A live test's root cause: the OLD prompt literally asked for
    "the file's COMPLETE new content." The new one must not."""
    provider = _FakeProvider('{"edits": [{"old_text": "a", "new_text": "b"}]}')
    planner = DevelopmentPlanner(provider)

    planner.propose_file_edits("objective", "desc", "app/x.py", "a")

    assert "COMPLETE new content" not in provider.last_prompt
    assert "surgical" in provider.last_prompt.lower() or "bounded" in provider.last_prompt.lower()


def test_propose_file_edits_rejects_a_response_with_no_edits_field():
    provider = _FakeProvider('{"old_text": "a", "new_text": "b"}')
    planner = DevelopmentPlanner(provider)

    with pytest.raises(DevelopmentPlanningError):
        planner.propose_file_edits("objective", "desc", "app/x.py", "a")
