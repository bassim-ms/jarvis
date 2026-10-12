"""
Automation Slice A1 - Workspace Files, the POLICY layer: `Workspace` (app/workspace/base.py) is the whole rule set - what a name may be, what
content may be, that a file is only ever CREATED (never overwritten, appended, renamed or deleted), that nothing outside one folder is touched,
and every bound. The tools (test_workspace_tools.py) and Jobs (test_workspace_jobs.py) are thin layers on top of it.

Real: the real filesystem, under pytest's tmp_path. Nothing here writes outside tmp_path (the default workspace folder is never created).
"""

import os
import pathlib
import stat
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest

import app.workspace.base as base
from app.workspace.base import (
    ALLOWED_EXTENSIONS,
    MAX_NAME_CHARS,
    Workspace,
    WorkspaceError,
    validate_content,
    validate_name,
)

LIMIT = 100  # a small per-file bound keeps the size tests quick
PACKAGE = pathlib.Path(__file__).resolve().parent.parent / "app" / "workspace"


@pytest.fixture
def ws(tmp_path):
    return Workspace(tmp_path / "ws", max_file_bytes=LIMIT)


def folder(ws):
    return ws._root


def listing(ws):
    return sorted(os.listdir(folder(ws))) if folder(ws).exists() else []


# ===================================================== names: ONE definition of a safe filename ====================================

ACCEPTED_NAMES = [
    "notes.md", "a.txt", "Study Guide.md", "data-2026_10.csv", "x.JSON", "Report.MD", "a.b.c.txt", "9lives.md", "CONSOLE.txt", "COM10.txt",
    "a" * (MAX_NAME_CHARS - 4) + ".txt",  # exactly the longest name
]


@pytest.mark.parametrize("name", ACCEPTED_NAMES, ids=[n[:24] for n in ACCEPTED_NAMES])
def test_a_plain_name_with_an_allowed_extension_is_accepted(name):
    assert validate_name(name) == name  # returned unchanged: nothing is normalised or rewritten


REFUSED_NAMES = [
    "", " ", ".md", "..md", "-x.md", "_x.md", "~x.md",  # empty / must start with a letter or digit
    "../x.md", "..\\x.md", "a/b.md", "a\\b.md", "/x.md", "\\x.md", "C:x.md", "C:\\x.md", "x.md:stream", "//server/share/x.md",  # paths, drives, ADS
    "x.md\n", "x\n.md", "x\r.md", "x\t.md", "x\x00.md", "x\x07.md",  # control characters
    "x.md ", " x.md", "x .md", "x..md", "x. .md",  # edge spaces and dots Windows would silently strip
    "x.py", "x.html", "x.htm", "x.exe", "x.bat", "x.cmd", "x.ps1", "x.js", "x.vbs", "x.lnk", "x.svg", "x.xml", "x.yaml", "x.ino",  # not inert text
    "x", "x.", "x.md.exe", "x.markdown", "x.m", "x.txtx",  # no / wrong extension
    "caf\u00e9.md", "\u0430bc.md", "x\u200b.md", "x\u202e.md", "x*.md", "x?.md", 'x".md', "x<.md", "x>.md", "x|.md", "x%20.md", "x$.md",  # odd characters
    "CON.txt", "con.md", "PRN.csv", "AUX.json", "NUL.txt", "COM1.txt", "com9.md", "LPT1.csv", "lpt9.json", "CON.backup.txt", "CON .txt",  # Windows devices
    "a" * (MAX_NAME_CHARS - 3) + ".txt",  # one character too long
]


@pytest.mark.parametrize("name", REFUSED_NAMES, ids=[repr(n)[:28] for n in REFUSED_NAMES])
def test_anything_but_a_plain_allowed_name_is_refused(name):
    with pytest.raises(WorkspaceError):
        validate_name(name)


@pytest.mark.parametrize("value", [None, 5, 5.5, b"x.md", ["x.md"], {"name": "x.md"}, True])
def test_a_non_string_is_never_a_name(value):
    with pytest.raises(WorkspaceError):
        validate_name(value)


def test_the_allowed_extensions_are_exactly_four_inert_text_types():
    assert ALLOWED_EXTENSIONS == frozenset({"txt", "md", "csv", "json"})


@pytest.mark.parametrize("name", REFUSED_NAMES, ids=[repr(n)[:28] for n in REFUSED_NAMES])
def test_a_refused_name_creates_nothing_anywhere(ws, tmp_path, name):
    with pytest.raises(WorkspaceError):
        ws.create(name, "text")

    assert sorted(os.listdir(tmp_path)) == []  # not even the workspace folder: validation comes first


# ===================================================== content ======================================================================


@pytest.mark.parametrize("text", ["line one\nline two", "tab\tseparated\r\nwindows lines", "caf\u00e9 \u2713 \u65e5\u672c\u8a9e", "# Heading\n\n- item", "x"])
def test_ordinary_text_is_accepted(text):
    assert validate_content(text, LIMIT) == text.encode("utf-8")


@pytest.mark.parametrize(
    "text",
    ["", "   ", "\n\t\r\n", "a\x00b", "a\x07b", "a\x1b[31mred", "a\x08b", "a\x0bb", "a\x0cb", "a\x7fb", "a\u0085b", "a\u009fb",
     "a\u200bb", "a\u200eb", "a\u202ab", "a\u202eb", "a\u2060b", "a\u2064b", "a\ufeffb", "\ud800"],
)
def test_empty_control_invisible_and_unencodable_text_is_refused(text):
    with pytest.raises(WorkspaceError):
        validate_content(text, LIMIT)


@pytest.mark.parametrize("value", [None, 5, b"bytes", ["x"], {"a": 1}])
def test_non_text_content_is_refused(value):
    with pytest.raises(WorkspaceError):
        validate_content(value, LIMIT)


def test_the_size_bound_counts_bytes_and_is_inclusive():
    assert len(validate_content("a" * LIMIT, LIMIT)) == LIMIT
    with pytest.raises(WorkspaceError):
        validate_content("a" * (LIMIT + 1), LIMIT)
    assert len(validate_content("\u00e9" * (LIMIT // 2), LIMIT)) == LIMIT  # two bytes each: exactly the bound
    with pytest.raises(WorkspaceError):
        validate_content("\u00e9" * (LIMIT // 2 + 1), LIMIT)  # fewer characters than the bound, more bytes than it


# ===================================================== create: only ever a NEW file =====================================================


def test_create_writes_exactly_the_given_bytes_into_a_folder_it_creates_itself(ws, tmp_path):
    assert not folder(ws).exists()  # nothing existed before the first write

    result = ws.create("notes.md", "# Hi\r\ncaf\u00e9\n")

    assert (result.name, result.bytes) == ("notes.md", len("# Hi\r\ncaf\u00e9\n".encode("utf-8")))
    assert (folder(ws) / "notes.md").read_bytes() == "# Hi\r\ncaf\u00e9\n".encode("utf-8")  # byte for byte, no newline translation
    assert os.listdir(tmp_path) == ["ws"] and listing(ws) == ["notes.md"]  # one folder, one file: no temp file, nothing beside it


def test_create_never_overwrites_appends_or_removes_an_existing_file(ws):
    ws.create("keep.md", "original")
    before = (folder(ws) / "keep.md").read_bytes()

    with pytest.raises(WorkspaceError, match="already exists"):
        ws.create("keep.md", "replacement")

    assert (folder(ws) / "keep.md").read_bytes() == before and listing(ws) == ["keep.md"]


def test_a_name_that_differs_only_in_case_cannot_replace_a_file_on_a_case_insensitive_disk(ws, tmp_path):
    ws.create("Notes.md", "original")
    if not (folder(ws) / "NOTES.MD").exists():
        pytest.skip("a case-sensitive file system treats these as two different files")

    with pytest.raises(WorkspaceError, match="already exists"):
        ws.create("NOTES.MD", "replacement")

    assert (folder(ws) / "Notes.md").read_bytes() == b"original"


def test_a_directory_in_the_way_is_never_replaced_or_written_into(ws):
    folder(ws).mkdir(parents=True)
    (folder(ws) / "taken.md").mkdir()

    with pytest.raises(WorkspaceError):
        ws.create("taken.md", "text")

    assert (folder(ws) / "taken.md").is_dir() and os.listdir(folder(ws) / "taken.md") == []


def test_racing_creates_of_one_name_produce_exactly_one_file(ws):
    outcomes = []

    def attempt(n):
        try:
            ws.create("race.md", f"writer {n}")
            outcomes.append("created")
        except WorkspaceError as exc:
            outcomes.append(str(exc))

    threads = [threading.Thread(target=attempt, args=(n,)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert outcomes.count("created") == 1 and len(outcomes) == 8
    assert listing(ws) == ["race.md"] and (folder(ws) / "race.md").read_text(encoding="utf-8").startswith("writer ")


class _BrokenHandle:
    def __init__(self, handle):
        self.handle = handle

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.handle.close()
        return False

    def write(self, data):
        raise OSError("disk full")


def test_a_failed_write_removes_only_the_file_this_call_created(ws, monkeypatch):
    ws.create("keep.md", "keep me")
    real_open = open

    def fake_open(path, mode="r", *args, **kwargs):
        handle = real_open(path, mode, *args, **kwargs)
        return _BrokenHandle(handle) if mode == "xb" else handle

    monkeypatch.setattr(base, "open", fake_open, raising=False)

    with pytest.raises(WorkspaceError, match="could not be written"):
        ws.create("new.md", "text")

    assert listing(ws) == ["keep.md"] and (folder(ws) / "keep.md").read_bytes() == b"keep me"


def test_a_file_that_does_not_read_back_identically_is_removed_and_never_reported_as_created(ws, monkeypatch):
    real_open = open

    class WrongBytes:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self, n=-1):
            return b"not what was written"

    def fake_open(path, mode="r", *args, **kwargs):
        return WrongBytes() if mode == "rb" else real_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(base, "open", fake_open, raising=False)

    with pytest.raises(WorkspaceError, match="removed it"):
        ws.create("verify.md", "text")

    assert listing(ws) == []


def test_a_file_that_turns_out_not_to_be_plain_is_removed(ws, monkeypatch):
    monkeypatch.setattr(base, "_is_plain_file", lambda st: False)

    with pytest.raises(WorkspaceError, match="removed it"):
        ws.create("odd.md", "text")

    assert listing(ws) == []


def test_nothing_outside_the_folder_is_ever_touched(ws, tmp_path):
    (tmp_path / "outside.md").write_text("outside", encoding="utf-8")
    ws.create("inside.md", "inside")
    for hostile in ("../outside.md", "..\\outside.md", "sub/../../outside.md"):
        with pytest.raises(WorkspaceError):
            ws.create(hostile, "pwned")

    assert (tmp_path / "outside.md").read_text(encoding="utf-8") == "outside"
    assert sorted(os.listdir(tmp_path)) == ["outside.md", "ws"] and listing(ws) == ["inside.md"]


# ===================================================== bounds ======================================================================


def test_the_number_of_files_is_bounded_and_foreign_files_do_not_count(tmp_path):
    ws = Workspace(tmp_path / "ws", max_files=3, max_file_bytes=LIMIT)
    folder(ws).mkdir(parents=True)
    (folder(ws) / "script.py").write_text("print(1)", encoding="utf-8")  # not a workspace file: never counted, never listed
    for n in range(3):
        ws.create(f"f{n}.md", "x")

    with pytest.raises(WorkspaceError, match="already holds 3 files"):
        ws.create("f3.md", "x")

    assert listing(ws) == ["f0.md", "f1.md", "f2.md", "script.py"]


def test_the_total_size_is_bounded(tmp_path):
    ws = Workspace(tmp_path / "ws", max_file_bytes=40, max_total_bytes=50)
    ws.create("a.md", "x" * 30)

    with pytest.raises(WorkspaceError, match="full"):
        ws.create("b.md", "y" * 30)
    ws.create("c.md", "z" * 20)  # exactly the 50 bytes left

    assert listing(ws) == ["a.md", "c.md"]


def test_a_folder_with_too_many_entries_is_treated_as_full_rather_than_scanned_without_end(tmp_path, monkeypatch):
    monkeypatch.setattr(base, "MAX_SCANNED_ENTRIES", 5)
    ws = Workspace(tmp_path / "ws", max_file_bytes=LIMIT)
    folder(ws).mkdir(parents=True)
    for n in range(8):
        (folder(ws) / f"other{n}.bin").write_bytes(b"x")

    with pytest.raises(WorkspaceError, match="already holds"):
        ws.create("new.md", "x")

    assert "new.md" not in listing(ws)


def test_new_files_per_minute_are_bounded_and_the_window_slides(tmp_path):
    now = [1000.0]
    ws = Workspace(tmp_path / "ws", max_file_bytes=LIMIT, writes_per_minute=2, clock=lambda: now[0])
    ws.create("a.md", "x")
    ws.create("b.md", "x")

    with pytest.raises(WorkspaceError, match="Too many"):
        ws.create("c.md", "x")
    now[0] += 61.0
    ws.create("c.md", "x")

    assert listing(ws) == ["a.md", "b.md", "c.md"]


def test_a_refused_name_or_text_does_not_use_up_the_minute(tmp_path):
    ws = Workspace(tmp_path / "ws", max_file_bytes=LIMIT, writes_per_minute=1)
    with pytest.raises(WorkspaceError):
        ws.create("bad name?.md", "x")
    with pytest.raises(WorkspaceError):
        ws.create("ok.md", "")

    ws.create("ok.md", "text")  # the single allowed write is still available

    assert listing(ws) == ["ok.md"]


# ===================================================== the kill switch ====================================================================


def test_the_switch_is_read_on_every_call_and_stops_everything(tmp_path):
    state = {"on": True}
    ws = Workspace(tmp_path / "ws", enabled=lambda: state["on"], max_file_bytes=LIMIT)
    ws.create("a.md", "kept")

    state["on"] = False
    for call in (lambda: ws.list(), lambda: ws.read("a.md"), lambda: ws.create("b.md", "x")):
        with pytest.raises(WorkspaceError, match="switched off"):
            call()
    assert listing(ws) == ["a.md"]

    state["on"] = True  # takes effect at once, no restart
    assert ws.read("a.md").text == "kept"
    ws.create("b.md", "x")
    assert listing(ws) == ["a.md", "b.md"]


def test_a_switched_off_attempt_does_not_use_up_a_write_slot(tmp_path):
    state = {"on": False}
    ws = Workspace(tmp_path / "ws", enabled=lambda: state["on"], max_file_bytes=LIMIT, writes_per_minute=1)
    for _ in range(3):
        with pytest.raises(WorkspaceError, match="switched off"):
            ws.create("a.md", "x")

    state["on"] = True
    ws.create("a.md", "x")  # the one allowed write this minute is still there

    assert listing(ws) == ["a.md"]


def test_a_switched_off_workspace_does_not_even_create_its_folder(tmp_path):
    ws = Workspace(tmp_path / "ws", enabled=lambda: False)

    with pytest.raises(WorkspaceError, match="switched off"):
        ws.create("a.md", "x")

    assert os.listdir(tmp_path) == []


# ===================================================== list ========================================================================


def test_list_returns_only_plain_workspace_files_sorted_and_without_paths(ws):
    assert ws.list() == []  # no folder yet: an empty answer, and still no folder
    assert not folder(ws).exists()
    ws.create("b.md", "bb")
    ws.create("A.txt", "a")
    (folder(ws) / "script.py").write_text("x", encoding="utf-8")  # wrong extension
    (folder(ws) / "noext").write_text("x", encoding="utf-8")
    (folder(ws) / "dir.md").mkdir()  # a directory with a valid-looking name
    (folder(ws) / "sub").mkdir()
    (folder(ws) / "sub" / "inner.md").write_text("x", encoding="utf-8")  # no subfolders are ever listed or read

    files = ws.list()

    assert [(f.name, f.bytes) for f in files] == [("A.txt", 1), ("b.md", 2)]
    assert all(len(f.modified) == 25 and f.modified[4] == "-" and "T" in f.modified for f in files)
    assert str(folder(ws)) not in repr(files)


# ===================================================== read ========================================================================


def test_read_returns_the_text_as_data_with_its_lines(ws):
    ws.create("a.md", "# Title\n\nline two\n  indented   and spaced\n")

    result = ws.read("a.md")

    assert (result.name, result.bytes, result.redacted) == ("a.md", len(b"# Title\n\nline two\n  indented   and spaced\n"), False)
    assert result.text == "# Title\n\nline two\nindented and spaced"  # lines kept (blank ones too); runs of spaces collapsed


def test_read_redacts_secret_shaped_strings_and_says_so(ws):
    folder(ws).mkdir(parents=True)  # a file the user put there, not one Jarvis wrote
    (folder(ws) / "a.md").write_text(
        "plan: buy servos\napi_key = sk-ant-api03-AbCdEfGhIjKlMnOpQrStUvWxYz0123456789\nAuthorization: Bearer abcdefghijklmnop\nend", encoding="utf-8"
    )
    ws.max_file_bytes = 1000

    result = ws.read("a.md")

    assert "sk-ant-api03" not in result.text and "abcdefghijklmnop" not in result.text and "[redacted]" in result.text
    assert result.redacted is True and result.text.splitlines()[0] == "plan: buy servos" and result.text.splitlines()[-1] == "end"


def test_read_removes_control_and_invisible_characters(ws):
    folder(ws).mkdir(parents=True)
    (folder(ws) / "a.md").write_bytes("clean\x1b[31m red\u200b zero\u202e rtl\x00 nul".encode("utf-8"))

    text = ws.read("a.md").text

    assert all(ord(c) >= 32 or c == "\n" for c in text) and "\x1b" not in text and "\u200b" not in text and "\u202e" not in text


def test_a_file_over_the_bound_is_refused_not_cut_short(ws):
    folder(ws).mkdir(parents=True)
    (folder(ws) / "big.md").write_bytes(b"x" * (LIMIT + 1))
    (folder(ws) / "exact.md").write_bytes(b"x" * LIMIT)

    with pytest.raises(WorkspaceError, match="never part of one"):
        ws.read("big.md")
    assert ws.read("exact.md").bytes == LIMIT


def test_a_file_that_is_not_utf8_text_is_refused(ws):
    folder(ws).mkdir(parents=True)
    (folder(ws) / "bin.md").write_bytes(b"\xff\xfe\x00\x01")

    with pytest.raises(WorkspaceError, match="not UTF-8"):
        ws.read("bin.md")


def test_reading_a_missing_file_or_a_missing_folder_says_so_without_a_path(ws, tmp_path):
    for _ in range(2):  # the folder does not exist, then it does
        with pytest.raises(WorkspaceError, match="There is no file named 'nope.md'") as caught:
            ws.read("nope.md")
        assert str(tmp_path) not in str(caught.value)
        folder(ws).mkdir(parents=True, exist_ok=True)


def test_a_directory_is_not_a_readable_file(ws):
    folder(ws).mkdir(parents=True)
    (folder(ws) / "dir.md").mkdir()

    with pytest.raises(WorkspaceError, match="not a plain file"):
        ws.read("dir.md")


@pytest.mark.parametrize("name", ["script.py", "noext", "page.html", "notes.md.exe", "trailing.md ", "CON.txt"])
def test_a_file_that_exists_but_is_not_a_workspace_name_is_still_never_read(ws, name):
    """The name policy is enforced on READ itself, not only by 'the file does not exist': the user's own script or page sitting in the
    folder is not something Jarvis reads."""
    folder(ws).mkdir(parents=True)
    try:
        (folder(ws) / name).write_text("print('private')", encoding="utf-8")
    except OSError:
        pytest.skip("this file system cannot hold a file with that name")

    with pytest.raises(WorkspaceError) as caught:
        ws.read(name)
    assert "no file named" not in str(caught.value)  # refused by the name policy, not because nothing was found


@pytest.mark.parametrize("name", ["../secret.md", "..\\secret.md", "sub/inner.md", "C:\\Windows\\win.ini", "x.py", ""])
def test_reading_outside_the_folder_or_a_disallowed_name_is_refused(ws, tmp_path, name):
    (tmp_path / "secret.md").write_text("outside", encoding="utf-8")
    ws.create("a.md", "x")

    with pytest.raises(WorkspaceError):
        ws.read(name)


# ===================================================== links and reparse points =====================================================


def test_a_stat_with_the_reparse_flag_or_a_link_mode_is_never_a_plain_file():
    assert base._is_plain_file(SimpleNamespace(st_mode=stat.S_IFREG, st_file_attributes=0)) is True
    assert base._is_plain_file(SimpleNamespace(st_mode=stat.S_IFREG)) is True  # a platform without the attribute
    assert base._is_plain_file(SimpleNamespace(st_mode=stat.S_IFREG, st_file_attributes=0x400)) is False  # a symlink / junction / placeholder
    assert base._is_plain_file(SimpleNamespace(st_mode=stat.S_IFLNK, st_file_attributes=0)) is False
    assert base._is_plain_file(SimpleNamespace(st_mode=stat.S_IFDIR, st_file_attributes=0)) is False


def test_a_workspace_folder_that_is_really_a_file_is_refused(tmp_path):
    (tmp_path / "ws").write_text("i am a file", encoding="utf-8")
    ws = Workspace(tmp_path / "ws")

    for call in (lambda: ws.list(), lambda: ws.read("a.md"), lambda: ws.create("a.md", "x")):
        with pytest.raises(WorkspaceError, match="not a plain folder"):
            call()
    assert (tmp_path / "ws").read_text(encoding="utf-8") == "i am a file"


@pytest.mark.skipif(sys.platform != "win32", reason="NTFS junctions exist only on Windows")
def test_a_workspace_folder_that_is_a_junction_is_refused(tmp_path):
    target = tmp_path / "elsewhere"
    target.mkdir()
    (target / "secret.md").write_text("elsewhere", encoding="utf-8")
    link = tmp_path / "ws"
    made = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], capture_output=True, text=True)
    if made.returncode != 0:
        pytest.skip(f"cannot create a junction here: {made.stderr.strip()}")
    ws = Workspace(link)

    for call in (lambda: ws.list(), lambda: ws.read("secret.md"), lambda: ws.create("new.md", "x")):
        with pytest.raises(WorkspaceError, match="not a plain folder"):
            call()
    assert os.listdir(target) == ["secret.md"]  # nothing was read through it and nothing was written into it


@pytest.mark.skipif(sys.platform != "win32", reason="NTFS junctions exist only on Windows")
def test_the_link_check_does_not_need_a_python_that_has_os_path_isjunction(tmp_path, monkeypatch):
    """The project supports Python 3.11, where `os.path.isjunction` does not exist: a junction must still be refused, and an ordinary
    folder must still work, with that function gone."""
    monkeypatch.delattr(os.path, "isjunction", raising=False)
    target = tmp_path / "elsewhere"
    target.mkdir()
    made = subprocess.run(["cmd", "/c", "mklink", "/J", str(tmp_path / "linked"), str(target)], capture_output=True, text=True)
    if made.returncode != 0:
        pytest.skip(f"cannot create a junction here: {made.stderr.strip()}")

    with pytest.raises(WorkspaceError, match="not a plain folder"):
        Workspace(tmp_path / "linked").create("a.md", "x")
    Workspace(tmp_path / "plain").create("a.md", "x")

    assert os.listdir(target) == [] and os.listdir(tmp_path / "plain") == ["a.md"]


def test_the_workspace_code_uses_no_python_version_specific_link_api():
    source = (PACKAGE / "base.py").read_text(encoding="utf-8")

    assert "isjunction" not in source and "is_junction" not in source


@pytest.mark.skipif(sys.platform != "win32", reason="NTFS junctions exist only on Windows")
def test_a_junction_inside_the_folder_is_neither_listed_read_nor_written_through(ws, tmp_path):
    target = tmp_path / "elsewhere"
    target.mkdir()
    (target / "secret.md").write_text("elsewhere", encoding="utf-8")
    ws.create("real.md", "real")
    made = subprocess.run(["cmd", "/c", "mklink", "/J", str(folder(ws) / "link.md"), str(target)], capture_output=True, text=True)
    if made.returncode != 0:
        pytest.skip(f"cannot create a junction here: {made.stderr.strip()}")

    assert [f.name for f in ws.list()] == ["real.md"]  # a reparse point is not a workspace file, whatever it is called
    with pytest.raises(WorkspaceError, match="not a plain file"):
        ws.read("link.md")
    with pytest.raises(WorkspaceError):
        ws.create("link.md", "written through the junction")
    assert os.listdir(target) == ["secret.md"] and (target / "secret.md").read_text(encoding="utf-8") == "elsewhere"


def test_a_workspace_folder_that_is_a_symlink_is_refused(tmp_path):
    target = tmp_path / "elsewhere"
    target.mkdir()
    link = tmp_path / "ws"
    try:
        os.symlink(target, link, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("this account may not create symbolic links")
    ws = Workspace(link)

    with pytest.raises(WorkspaceError, match="not a plain folder"):
        ws.create("new.md", "x")

    assert os.listdir(target) == []


def test_a_symlinked_file_inside_the_folder_is_neither_listed_nor_read(ws, tmp_path):
    outside = tmp_path / "outside.md"
    outside.write_text("outside secret", encoding="utf-8")
    ws.create("real.md", "real")
    try:
        os.symlink(outside, folder(ws) / "link.md")
    except (OSError, NotImplementedError):
        pytest.skip("this account may not create symbolic links")

    assert [f.name for f in ws.list()] == ["real.md"]
    with pytest.raises(WorkspaceError, match="not a plain file"):
        ws.read("link.md")
    with pytest.raises(WorkspaceError):
        ws.create("link.md", "overwrite through the link")
    assert outside.read_text(encoding="utf-8") == "outside secret"


# ===================================================== what an error ever says ======================================================


def test_no_message_ever_carries_a_path_or_an_os_error(ws, tmp_path):
    ws.create("a.md", "x")
    messages = []
    attempts = [
        lambda: ws.create("a.md", "x"), lambda: ws.create("../a.md", "x"), lambda: ws.create("b.md", ""), lambda: ws.read("nope.md"),
        lambda: ws.read("../a.md"), lambda: ws.create("b.py", "x"),
    ]
    for attempt in attempts:
        with pytest.raises(WorkspaceError) as caught:
            attempt()
        messages.append(str(caught.value))

    for text in messages:
        assert str(tmp_path) not in text and "Users" not in text and "\\\\" not in text and "Errno" not in text and "WinError" not in text
