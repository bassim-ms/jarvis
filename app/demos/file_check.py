"""
The demo's confinement rules and its ONE file tool: `verify_demo_file`.

    a dedicated directory  ->  one new, uniquely named file in it  ->  a READ tool bound to exactly that file and the exact text it must hold

This is deliberately NOT a general file reader. The tool takes no parameters, so a plan cannot name a path; it is constructed for one file inside one
dedicated directory and re-checks that binding every time it runs; it opens that single file read-only and compares it with the one text it was built
with. It lives only in the demo's private registry (never `core.tools`), so a Job, a chat command or an automation cannot reach it.

The directory rules exist so the demo can never touch anything else:
  * absolute, ASCII, no `..`, and no link or junction anywhere in the path (the real path must equal the written one);
  * its own name starts with `jarvis_demo` - it must be a directory made for this, never Documents, the home folder or a drive root;
  * it must be FRESH: not there yet (the demo creates it, one level) or already empty. A directory with anything in it - even an earlier demo file - is
    refused, so a new run always starts with a directory holding nothing and a file name that cannot already exist.

Two read-only checks sit on that directory while the run is in progress: `verify_demo_target_free` (just before the save: still empty, still a real
directory, the target still absent) and `verify_demo_file` (after it: the file is there and holds exactly the note). Both are re-checked from the disk at the
moment they run, and both are bound to the one directory and file - they take no arguments.
"""

from __future__ import annotations

import os
import re
import stat
from datetime import datetime, timezone
from pathlib import Path

from app.observation.base import scrub
from app.tools.base import PermissionLevel, Tool, ToolRequirements, ToolResult

DIRECTORY_PREFIX = "jarvis_demo"
FILE_NAME = re.compile(r"jarvis_demo_\d{8}T\d{6}Z_[0-9a-f]{6}\.txt")
MAX_PATH_CHARS = 200
MAX_NOTE_CHARS = 200
MAX_FILE_BYTES = 4096  # far more than the longest note; anything bigger is not what the demo wrote
EXCERPT_CHARS = 120
_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)


class DemoPathError(ValueError):
    """The directory, file or note is not one the demo is willing to use. The message says why, in plain words."""


def _same(a: str | os.PathLike, b: str | os.PathLike) -> bool:
    return os.path.normcase(os.fspath(a)) == os.path.normcase(os.fspath(b))


def is_link(path: str | os.PathLike) -> bool:
    """A symbolic link or a Windows junction / other reparse point. Looks at the entry itself (never follows it)."""
    try:
        if os.path.islink(path):
            return True
        return bool(os.lstat(path).st_file_attributes & _REPARSE_POINT)
    except AttributeError:  # not Windows: st_file_attributes does not exist, and islink already answered
        return False
    except OSError:
        return False


def _plain_ascii(text: str) -> bool:
    return bool(text) and all(0x20 <= ord(ch) <= 0x7E for ch in text)


def validate_note(note: object) -> str:
    """The text the demo types and later expects to find: one plain ASCII line (it must survive being typed on any keyboard layout and read back)."""
    if not isinstance(note, str) or not note or note != note.strip() or not _plain_ascii(note) or len(note) > MAX_NOTE_CHARS:
        raise DemoPathError(f"The note must be one line of plain ASCII text (no leading or trailing space), 1-{MAX_NOTE_CHARS} characters.")
    return note


def validate_directory(raw: object) -> Path:
    """The dedicated directory, or DemoPathError. Creates and changes nothing."""
    text = os.fspath(raw) if isinstance(raw, (str, os.PathLike)) else ""
    if not text or len(text) > MAX_PATH_CHARS or not _plain_ascii(text):
        raise DemoPathError(f"The directory must be a plain ASCII path of at most {MAX_PATH_CHARS} characters.")
    path = Path(text)
    if not path.is_absolute():
        raise DemoPathError("The directory must be an absolute path.")
    if ".." in path.parts:
        raise DemoPathError("The directory must not contain '..'.")
    path = Path(os.path.abspath(text))
    if path.parent == path:
        raise DemoPathError("The directory must not be a drive root.")
    if not path.name.casefold().startswith(DIRECTORY_PREFIX):
        raise DemoPathError(f"The directory's own name must start with '{DIRECTORY_PREFIX}' - it must be a folder made for this demo, not an existing one of yours.")
    if not _same(os.path.realpath(path), path):
        raise DemoPathError("The directory path contains a link or junction; the demo only works in a real directory.")
    if path.exists() or is_link(path):
        if is_link(path) or not path.is_dir():
            raise DemoPathError("The directory path is not a plain directory.")
        try:
            names = [entry.name for entry in os.scandir(path)]
        except OSError as exc:
            raise DemoPathError(f"The directory cannot be listed: {exc.__class__.__name__}.") from None
        if names:  # fresh means fresh: not even an earlier demo file, so nothing in it can ever be mistaken for - or saved over as - this run's file
            raise DemoPathError(f"The directory already holds {len(names)} item(s). Use a new directory, or an empty one, for each run.")
    else:
        parent = path.parent
        if not parent.is_dir() or is_link(parent):
            raise DemoPathError("The directory's parent does not exist (the demo creates one directory level, nothing above it).")
    return path


def new_target(directory: Path, *, now: datetime | None = None, token: str | None = None) -> Path:
    """A fresh, unique file path inside `directory`. Refuses (never reuses) a name that already exists."""
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    token = token if token is not None else os.urandom(3).hex()
    path = directory / f"jarvis_demo_{stamp}_{token}.txt"
    return check_target(directory, path, must_not_exist=True)


def check_target(directory: Path, path: Path, *, must_not_exist: bool) -> Path:
    """`path` is exactly one demo-named file directly inside `directory`, is not a link, and (when asked) does not exist yet."""
    if not FILE_NAME.fullmatch(path.name) or not _same(path.parent, directory):
        raise DemoPathError("The target must be one demo-named file directly inside the dedicated directory.")
    if is_link(path):
        raise DemoPathError("The target is a link; the demo never follows links.")
    if must_not_exist and (path.exists() or is_link(path)):
        raise DemoPathError("The target file already exists; the demo never saves over an existing file.")
    return path


def directory_problem(directory: Path) -> str | None:
    """Why `directory` is no longer the plain, real directory it was validated as (gone, replaced by a file, a link or junction, or reached through
    one), or None. Looked at afresh each time: a directory can be swapped after it was validated."""
    if is_link(directory) or not directory.is_dir():
        return "is no longer a plain directory"
    if not _same(os.path.realpath(directory), directory):
        return "is now reached through a link or junction"
    return None


class VerifyDemoFileTool(Tool):
    """READ-only: does THE file this demo was built for exist, as a plain file, holding exactly THE expected text? No parameters, no other path."""

    name = "verify_demo_file"
    description = "Check, from the disk, that the demo's one target file exists and holds exactly the expected text. Takes no arguments. Read-only."
    permission = PermissionLevel.READ
    requirements = ToolRequirements()
    automatable = False  # a schedule or an automation never runs it (it is also in no shared registry)

    def __init__(self, directory: Path, path: Path, expected: str):
        self._directory = Path(directory)
        self._path = check_target(self._directory, Path(path), must_not_exist=False)
        self._expected = validate_note(expected)

    def execute(self) -> ToolResult:
        # The binding is re-checked at run time, not only at construction.
        problem = directory_problem(self._directory)
        if problem is not None:
            return ToolResult.error(f"Verification failed: the demo directory {problem}, so the file is not trusted.")
        try:
            check_target(self._directory, self._path, must_not_exist=False)
            descriptor = os.open(self._path, os.O_RDONLY | getattr(os, "O_BINARY", 0))
        except FileNotFoundError:
            return ToolResult.error(f"Verification failed: the file {self._path.name} does not exist.")
        except (DemoPathError, OSError) as exc:
            return ToolResult.error(f"Verification failed: the file {self._path.name} cannot be read ({exc.__class__.__name__}).")
        try:
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode):
                return ToolResult.error(f"Verification failed: {self._path.name} is not a plain file.")
            if info.st_size > MAX_FILE_BYTES:
                return ToolResult.error(f"Verification failed: {self._path.name} is {info.st_size} bytes, far more than the note.")
            raw = os.read(descriptor, MAX_FILE_BYTES + 1)
        finally:
            os.close(descriptor)
        try:
            observed = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            return ToolResult.error(f"Verification failed: {self._path.name} is not UTF-8 text.")
        if observed != self._expected:
            shown = scrub(observed, EXCERPT_CHARS)
            return ToolResult.error(
                f"Verification failed: {self._path.name} exists but does not hold the expected text "
                f"(expected {len(self._expected)} characters, found {len(observed)}: '{shown}')."
            )
        return ToolResult.success(
            data={"verified": True, "file": self._path.name, "characters": len(observed)},
            message=f"Verified: {self._path.name} exists and holds exactly the expected {len(observed)} characters.",
        )


class CheckDemoTargetFreeTool(Tool):
    """READ-only, run immediately before the save: the demo directory is still the plain, empty directory it was made as, and THE target file still does not
    exist. If anything has appeared in it, or it was swapped for a link, the save must not happen. No parameters, no other path.

    This narrows the window in which something else could create the target (a save would then raise an overwrite prompt); it cannot close it - see the
    module docstring of notepad_note.py."""

    name = "verify_demo_target_free"
    description = "Check, from the disk, that the demo directory is still empty and its one target file still does not exist. Takes no arguments. Read-only."
    permission = PermissionLevel.READ
    requirements = ToolRequirements()
    automatable = False

    def __init__(self, directory: Path, path: Path):
        self._directory = Path(directory)
        self._path = check_target(self._directory, Path(path), must_not_exist=False)

    def execute(self) -> ToolResult:
        problem = directory_problem(self._directory)
        if problem is not None:
            return ToolResult.error(f"The demo directory {problem}; refusing to save.")
        try:
            check_target(self._directory, self._path, must_not_exist=True)
        except DemoPathError as exc:
            return ToolResult.error(f"The target is not free; refusing to save: {exc}")
        try:
            names = [entry.name for entry in os.scandir(self._directory)]
        except OSError as exc:
            return ToolResult.error(f"The demo directory cannot be listed ({exc.__class__.__name__}); refusing to save.")
        if names:
            return ToolResult.error(f"The demo directory is no longer empty ({len(names)} item(s) appeared); refusing to save.")
        return ToolResult.success(data={"free": True}, message=f"{self._path.name} is still free in an empty directory.")
