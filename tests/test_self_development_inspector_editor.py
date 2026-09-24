"""
Tests for app/self_development/inspector.py and editor.py.

Covers focused-test categories:
1. repository-root enforcement
2. path traversal rejection
6. secret-file protection (existence may be reported, content never is)
18. permission boundaries (CODE_READ automatic, CODE_EDIT CONFIRM)
"""

from app.self_development.editor import EditFileTool
from app.self_development.inspector import ListRepoFilesTool, ReadFileTool, SearchRepoTool
from app.tools.base import AUTOMATIC_PERMISSIONS, PermissionLevel, ToolResultStatus
from app.tools.registry import ToolRegistry


def _make_repo(tmp_path):
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "widget.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / ".env").write_text("JARVIS_GEMINI_API_KEY=super-secret-value\n", encoding="utf-8")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("(should never be listed)", encoding="utf-8")
    return str(tmp_path)


# -- ListRepoFilesTool -------------------------------------------------------


def test_list_repo_files_finds_ordinary_files(tmp_path):
    root = _make_repo(tmp_path)
    tool = ListRepoFilesTool(root)
    result = tool.execute()
    paths = {entry["path"] for entry in result.data}
    assert "app/widget.py" in paths


def test_list_repo_files_excludes_the_git_directory(tmp_path):
    root = _make_repo(tmp_path)
    tool = ListRepoFilesTool(root)
    result = tool.execute()
    paths = {entry["path"] for entry in result.data}
    assert not any(p.startswith(".git/") for p in paths)


def test_list_repo_files_tags_secret_files_without_hiding_their_existence(tmp_path):
    """V1's own spec: "may detect that a secret/config file exists
    without revealing its values" - the path itself is listed, tagged."""
    root = _make_repo(tmp_path)
    tool = ListRepoFilesTool(root)
    result = tool.execute()
    entry = next(e for e in result.data if e["path"] == ".env")
    assert entry["secret"] is True


def test_list_repo_files_rejects_a_traversal_subdirectory(tmp_path):
    root = _make_repo(tmp_path)
    tool = ListRepoFilesTool(root)
    result = tool.execute(subdirectory="../outside")
    assert result.status is ToolResultStatus.ERROR


def test_list_repo_files_is_automatic_read():
    assert ListRepoFilesTool.permission is PermissionLevel.READ
    assert ListRepoFilesTool.permission in AUTOMATIC_PERMISSIONS


# -- ReadFileTool -------------------------------------------------------


def test_read_file_returns_real_content(tmp_path):
    root = _make_repo(tmp_path)
    tool = ReadFileTool(root)
    result = tool.execute(path="app/widget.py")
    assert result.status is ToolResultStatus.SUCCESS
    assert result.data["content"] == "x = 1\n"


def test_read_file_never_reveals_secret_content(tmp_path):
    root = _make_repo(tmp_path)
    tool = ReadFileTool(root)
    result = tool.execute(path=".env")
    assert result.status is ToolResultStatus.ERROR
    assert "super-secret-value" not in result.message
    assert "protected" in result.message.lower()


def test_read_file_rejects_path_traversal(tmp_path):
    root = _make_repo(tmp_path)
    outside = tmp_path.parent / "outside_secret.txt"
    outside.write_text("top secret", encoding="utf-8")
    tool = ReadFileTool(root)
    result = tool.execute(path="../outside_secret.txt")
    assert result.status is ToolResultStatus.ERROR
    assert "top secret" not in result.message


def test_read_file_rejects_absolute_path(tmp_path):
    root = _make_repo(tmp_path)
    tool = ReadFileTool(root)
    result = tool.execute(path="/etc/passwd")
    assert result.status is ToolResultStatus.ERROR


def test_read_file_missing_file_reports_cleanly(tmp_path):
    root = _make_repo(tmp_path)
    tool = ReadFileTool(root)
    result = tool.execute(path="does/not/exist.py")
    assert result.status is ToolResultStatus.ERROR


def test_read_file_is_automatic_read():
    assert ReadFileTool.permission is PermissionLevel.READ


# -- SearchRepoTool -------------------------------------------------------


def test_search_repo_finds_a_match(tmp_path):
    root = _make_repo(tmp_path)
    tool = SearchRepoTool(root)
    result = tool.execute(query="x = 1")
    assert any(m["path"] == "app/widget.py" for m in result.data)


def test_search_repo_never_searches_inside_secret_files(tmp_path):
    root = _make_repo(tmp_path)
    tool = SearchRepoTool(root)
    result = tool.execute(query="super-secret-value")
    assert result.data == []


# -- EditFileTool: CODE_EDIT --------------------------------------------


def test_edit_file_writes_new_content(tmp_path):
    root = _make_repo(tmp_path)
    tool = EditFileTool(root)
    result = tool.execute(path="app/widget.py", content="x = 2\n")
    assert result.status is ToolResultStatus.SUCCESS
    assert (tmp_path / "app" / "widget.py").read_text(encoding="utf-8") == "x = 2\n"


def test_edit_file_can_create_a_new_file(tmp_path):
    root = _make_repo(tmp_path)
    tool = EditFileTool(root)
    result = tool.execute(path="app/new_module.py", content="def f(): pass\n")
    assert result.status is ToolResultStatus.SUCCESS
    assert (tmp_path / "app" / "new_module.py").exists()


def test_edit_file_refuses_to_touch_a_secret_file(tmp_path):
    root = _make_repo(tmp_path)
    original = (tmp_path / ".env").read_text(encoding="utf-8")
    tool = EditFileTool(root)
    result = tool.execute(path=".env", content="JARVIS_GEMINI_API_KEY=hacked\n")
    assert result.status is ToolResultStatus.ERROR
    assert (tmp_path / ".env").read_text(encoding="utf-8") == original


def test_edit_file_rejects_path_traversal(tmp_path):
    root = _make_repo(tmp_path)
    outside = tmp_path.parent / "should_not_be_written.txt"
    tool = EditFileTool(root)
    result = tool.execute(path="../should_not_be_written.txt", content="pwned")
    assert result.status is ToolResultStatus.ERROR
    assert not outside.exists()


def test_edit_file_requires_confirmation_through_the_registry(tmp_path):
    root = _make_repo(tmp_path)
    registry = ToolRegistry()
    registry.register(EditFileTool(root))
    result = registry.execute(
        "self_dev_edit_file", params={"path": "app/widget.py", "content": "x = 99\n"}
    )
    assert result.status is ToolResultStatus.CONFIRMATION_REQUIRED
    # Never actually wrote anything while unconfirmed.
    assert (tmp_path / "app" / "widget.py").read_text(encoding="utf-8") == "x = 1\n"


def test_edit_file_denied_when_declined(tmp_path):
    root = _make_repo(tmp_path)
    registry = ToolRegistry()
    registry.register(EditFileTool(root))
    result = registry.execute(
        "self_dev_edit_file",
        params={"path": "app/widget.py", "content": "x = 99\n"},
        confirmed=False,
    )
    assert result.status is ToolResultStatus.DENIED
    assert (tmp_path / "app" / "widget.py").read_text(encoding="utf-8") == "x = 1\n"


def test_edit_file_writes_once_confirmed(tmp_path):
    root = _make_repo(tmp_path)
    registry = ToolRegistry()
    registry.register(EditFileTool(root))
    result = registry.execute(
        "self_dev_edit_file",
        params={"path": "app/widget.py", "content": "x = 99\n"},
        confirmed=True,
    )
    assert result.status is ToolResultStatus.SUCCESS
    assert (tmp_path / "app" / "widget.py").read_text(encoding="utf-8") == "x = 99\n"
