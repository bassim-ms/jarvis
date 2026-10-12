"""
Workspace: ONE folder in which Jarvis may list and read small text files and CREATE new ones. Nothing else.

    list_workspace_files / read_workspace_file / write_workspace_file   (tools.py)  ->  Workspace (this module)  ->  the folder

This class is the whole policy; the tools are thin adapters over it (the same shape as NoteStore and its tools), so there is exactly ONE
definition of a safe name, safe content and a safe path. Every failure is a `WorkspaceError` carrying a short, safe sentence - never an
absolute path, never a raw OS error.

WHAT IT WILL DO
  * LIST   the plain text files directly inside the folder (name, size, modified time).
  * READ   one such file (UTF-8, bounded), returned as DATA with secret-shaped strings redacted. Never truncated: a file over the bound
           is refused with a clear reason, so a partial file can never be mistaken for the whole.
  * CREATE one NEW file, atomically (exclusive create): it never overwrites, appends, renames or deletes anything that already exists.
           The bytes are read back and compared before success is reported; a failed write removes only the file this call created.

WHAT IT WILL NEVER DO
  * touch anything outside the folder: a name is a single plain filename (letters, digits, `. _ - space`, 1-64 characters), so it cannot
    hold a separator, drive, `:` (alternate data stream), `..`, or a Windows device name; there are no subfolders; a symlink, junction or
    other reparse point - as the folder itself or as an entry - is refused; and the resolved parent of every path is checked once more.
  * accept a name whose extension is not one of `.txt .md .csv .json` - inert text only, nothing a double-click could run.
  * exceed the bounds (per file, files in the folder, total bytes, new files per minute) or run while `enabled()` is false. The switch is
    read on EVERY call, so turning it off takes effect immediately.

WHAT IT DOES NOT PROMISE
  * Content is written as given: it is the user's request or the model's answer, in a folder the user owns. Secret-shaped strings are
    redacted on READ (so they are not handed to a model or another step), not blocked on write.
  * Reading is not private from the AI: text read here may be passed to the configured AI provider by a later step (for example `analyze`).
"""

from __future__ import annotations

import os
import re
import stat
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from app.observation.base import REDACTED, scrub

ALLOWED_EXTENSIONS = frozenset({"txt", "md", "csv", "json"})
MAX_NAME_CHARS = 64  # the whole name, extension included
DEFAULT_MAX_FILE_BYTES = 64 * 1024
DEFAULT_MAX_FILES = 200
DEFAULT_MAX_TOTAL_BYTES = 20 * 1024 * 1024
DEFAULT_WRITES_PER_MINUTE = 10
MAX_SCANNED_ENTRIES = 2000  # a folder with more entries than this is treated as full: a scan is always bounded

_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._ -]*\.(?P<ext>[A-Za-z]{2,4})", re.ASCII)
_RESERVED = frozenset({"CON", "PRN", "AUX", "NUL", *(f"COM{n}" for n in range(1, 10)), *(f"LPT{n}" for n in range(1, 10))})
#: Controls other than newline / tab / carriage return, DEL and the C1 range, zero-width and bidirectional-override characters.
_BAD_CONTENT = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f\u200b-\u200f\u202a-\u202e\u2060-\u2064\ufeff]")
_REPARSE_POINT = 0x400  # FILE_ATTRIBUTE_REPARSE_POINT: a symlink, a junction, a cloud placeholder...


class WorkspaceError(RuntimeError):
    """Anything the workspace refuses or cannot do. The text is always a short sentence that is safe to show: no path, no OS error."""


@dataclass(frozen=True)
class FileInfo:
    name: str
    bytes: int
    modified: str  # local time, ISO 8601, to the second


@dataclass(frozen=True)
class ReadResult:
    name: str
    bytes: int
    text: str  # normalised line by line, secret-shaped strings redacted
    redacted: bool


@dataclass(frozen=True)
class CreateResult:
    name: str
    bytes: int


def validate_name(name: object) -> str:
    """`name` unchanged if it is a safe workspace filename, else a WorkspaceError. The ONE definition: tools and class both call it."""
    if not isinstance(name, str) or not name:
        raise WorkspaceError("A file name is needed.")
    if len(name) > MAX_NAME_CHARS:
        raise WorkspaceError(f"A file name can be at most {MAX_NAME_CHARS} characters.")
    match = _NAME.fullmatch(name)
    if match is None:
        raise WorkspaceError(
            "A file name must be a plain name - letters, digits, '.', '_', '-' and spaces, starting with a letter or digit - "
            "with an extension of .txt, .md, .csv or .json. Folders and paths are not allowed."
        )
    if match.group("ext").lower() not in ALLOWED_EXTENSIONS:
        raise WorkspaceError("Only .txt, .md, .csv and .json files are allowed.")
    stem = name[: -(len(match.group("ext")) + 1)]
    if stem != stem.rstrip(" ."):
        raise WorkspaceError("A file name cannot end with a space or a dot before its extension.")
    if stem.split(".")[0].strip().upper() in _RESERVED:
        raise WorkspaceError("That name is reserved by Windows. Choose another.")
    return name


def validate_content(content: object, max_bytes: int) -> bytes:
    """The UTF-8 bytes of `content` if it is acceptable text of at most `max_bytes`, else a WorkspaceError."""
    if not isinstance(content, str) or not content.strip():
        raise WorkspaceError("A file needs some text to write.")
    if _BAD_CONTENT.search(content):
        raise WorkspaceError("The text contains control or invisible characters, so it was not written.")
    try:
        data = content.encode("utf-8")
    except UnicodeEncodeError:
        raise WorkspaceError("The text is not valid Unicode, so it was not written.") from None
    if len(data) > max_bytes:
        raise WorkspaceError(f"The text is {len(data)} bytes; a file can be at most {max_bytes} bytes.")
    return data


def _is_plain_file(st: os.stat_result) -> bool:
    return stat.S_ISREG(st.st_mode) and not (getattr(st, "st_file_attributes", 0) & _REPARSE_POINT)


def _is_link(path: os.PathLike | str) -> bool:
    """True for a symlink, a junction or any other reparse point - read from lstat, which does not follow it, on every supported Python."""
    try:
        st = os.lstat(path)
    except OSError:
        return False
    return stat.S_ISLNK(st.st_mode) or bool(getattr(st, "st_file_attributes", 0) & _REPARSE_POINT)


class _Window:
    """At most `limit` events in any sliding minute."""

    def __init__(self, limit: int, clock: Callable[[], float]):
        self._limit, self._clock, self._events = limit, clock, deque()

    def take(self) -> None:
        now = self._clock()
        while self._events and now - self._events[0] > 60.0:
            self._events.popleft()
        if len(self._events) >= self._limit:
            raise WorkspaceError(f"Too many files were created in the last minute (limit {self._limit}). Try again shortly.")
        self._events.append(now)


class Workspace:
    def __init__(
        self,
        root: str | os.PathLike,
        *,
        enabled: Callable[[], bool] = lambda: True,
        max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
        max_files: int = DEFAULT_MAX_FILES,
        max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES,
        writes_per_minute: int = DEFAULT_WRITES_PER_MINUTE,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._root = Path(root)
        self._enabled = enabled
        self.max_file_bytes = max_file_bytes
        self.max_files = max_files
        self.max_total_bytes = max_total_bytes
        self._writes = _Window(writes_per_minute, clock)
        self._lock = threading.Lock()  # one create at a time: the quota check and the create are one step

    # -- the folder ------------------------------------------------------------------------------------------------------------

    def _check_enabled(self) -> None:
        if not self._enabled():
            raise WorkspaceError("The workspace is switched off.")

    def _folder(self, *, create: bool) -> Path | None:
        """The resolved folder, or None if it does not exist (and `create` is false). Refuses when switched off or when the folder
        itself is a link or reparse point."""
        self._check_enabled()
        root = self._root
        try:
            if os.path.lexists(root):
                if _is_link(root) or not os.path.isdir(root):
                    raise WorkspaceError("The workspace folder is not a plain folder, so I will not use it.")
            elif create:
                root.mkdir(parents=True, exist_ok=True)
            else:
                return None
            return Path(os.path.realpath(root))
        except WorkspaceError:
            raise
        except OSError:
            raise WorkspaceError("The workspace folder is not available.") from None

    def _scan(self, folder: Path) -> tuple[list[FileInfo], bool]:
        """The plain, policy-conforming files directly in `folder` (sorted by name), and whether the scan hit its bound."""
        found: list[FileInfo] = []
        full = False
        try:
            with os.scandir(folder) as entries:
                for count, entry in enumerate(entries, start=1):
                    if count > MAX_SCANNED_ENTRIES:
                        full = True
                        break
                    try:
                        validate_name(entry.name)
                        st = entry.stat(follow_symlinks=False)
                    except (WorkspaceError, OSError):
                        continue
                    if _is_plain_file(st):
                        modified = datetime.fromtimestamp(st.st_mtime).astimezone().isoformat(timespec="seconds")
                        found.append(FileInfo(entry.name, st.st_size, modified))
        except OSError:
            raise WorkspaceError("The workspace folder could not be read.") from None
        return sorted(found, key=lambda info: info.name.lower()), full

    # -- list / read / create ----------------------------------------------------------------------------------------------------

    def list(self) -> list[FileInfo]:
        folder = self._folder(create=False)
        if folder is None:
            return []
        return self._scan(folder)[0]

    def read(self, name: object) -> ReadResult:
        validate_name(name)
        folder = self._folder(create=False)
        if folder is None:
            raise WorkspaceError(f"There is no file named '{name}' in the workspace.")
        path = folder / name
        try:
            st = os.lstat(path)
        except FileNotFoundError:
            raise WorkspaceError(f"There is no file named '{name}' in the workspace.") from None
        except OSError:
            raise WorkspaceError(f"'{name}' could not be read.") from None
        if not _is_plain_file(st) or Path(os.path.realpath(path)).parent != folder:
            raise WorkspaceError(f"'{name}' is not a plain file, so I will not read it.")
        if st.st_size > self.max_file_bytes:
            raise WorkspaceError(f"'{name}' is {st.st_size} bytes; I only read files of at most {self.max_file_bytes} bytes, and never part of one.")
        try:
            with open(path, "rb") as handle:
                raw = handle.read(self.max_file_bytes + 1)
        except OSError:
            raise WorkspaceError(f"'{name}' could not be read.") from None
        if len(raw) > self.max_file_bytes:
            raise WorkspaceError(f"'{name}' grew past {self.max_file_bytes} bytes while it was being read, so I did not use it.")
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            raise WorkspaceError(f"'{name}' is not UTF-8 text.") from None
        cleaned = "\n".join(scrub(line) for line in text.splitlines())  # line by line: the file keeps its lines, loses odd characters and secrets
        return ReadResult(name, len(raw), cleaned, cleaned.count(REDACTED) > text.count(REDACTED))

    def create(self, name: object, content: object) -> CreateResult:
        validate_name(name)
        data = validate_content(content, self.max_file_bytes)
        with self._lock:
            self._check_enabled()
            self._writes.take()
            folder = self._folder(create=True)
            files, full = self._scan(folder)
            if full or len(files) >= self.max_files:
                raise WorkspaceError(f"The workspace already holds {self.max_files} files, so nothing was written.")
            if sum(info.bytes for info in files) + len(data) > self.max_total_bytes:
                raise WorkspaceError("The workspace is full, so nothing was written.")
            path = folder / name
            try:
                handle = open(path, "xb")  # exclusive create: it can only succeed if nothing by that name exists
            except FileExistsError:
                raise WorkspaceError(f"'{name}' already exists, so nothing was written. Choose a different name.") from None
            except OSError:
                raise WorkspaceError(f"'{name}' could not be created.") from None
            try:
                with handle:
                    handle.write(data)
                with open(path, "rb") as back:
                    verified = back.read(len(data) + 1) == data
                st = os.lstat(path)
                if not verified or not _is_plain_file(st) or Path(os.path.realpath(path)).parent != folder:
                    raise OSError("verification failed")
            except OSError:
                try:
                    os.remove(path)  # this call created it, so this call removes it - and nothing else
                except OSError:
                    pass
                raise WorkspaceError(f"'{name}' could not be written and checked, so I removed it.") from None
        return CreateResult(name, len(data))
