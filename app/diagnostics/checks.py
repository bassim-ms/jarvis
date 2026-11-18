"""
Database integrity checks (spec 457) and resource readings (452). Read-only: a database is opened with `mode=ro` and only
`PRAGMA quick_check` and row counts are run on it.
"""

from __future__ import annotations

import ctypes
import os
import shutil
import sqlite3
import sys
import threading
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class DatabaseCheck:
    name: str  # the setting's name, e.g. "memory_db_path"
    file: str  # the file name only - the folder is shown once, in the report
    exists: bool
    ok: bool | None  # None = not created yet (normal before first use): nothing to check
    detail: str
    size_bytes: int = 0
    tables: int = 0
    rows: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def open_read_only(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True, timeout=5)


def table_counts(connection: sqlite3.Connection) -> dict[str, int]:
    tables = [row[0] for row in connection.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
    return {table: connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0] for table in tables}


def check_database(name: str, path: str | Path) -> DatabaseCheck:
    """`PRAGMA quick_check` and row counts on one database, read-only. A missing file is not a failure (it is created the
    first time that part of Jarvis is used); an unreadable or damaged one is."""
    file = Path(path)
    if not file.exists():
        return DatabaseCheck(name, file.name, False, None, "not created yet")
    size = file.stat().st_size
    try:
        connection = open_read_only(file)
        try:
            verdict = connection.execute("PRAGMA quick_check").fetchone()[0]
            counts = table_counts(connection) if verdict == "ok" else {}
        finally:
            connection.close()
    except sqlite3.DatabaseError as exc:
        return DatabaseCheck(name, file.name, True, False, f"cannot be read as a database ({type(exc).__name__})", size)
    if verdict != "ok":
        return DatabaseCheck(name, file.name, True, False, f"integrity problem: {str(verdict)[:200]}", size)
    return DatabaseCheck(name, file.name, True, True, "ok", size, len(counts), sum(counts.values()))


def disk(path: str | Path) -> dict:
    usage = shutil.disk_usage(Path(path).resolve())
    return {"free_bytes": usage.free, "total_bytes": usage.total, "free_percent": round(100 * usage.free / usage.total, 1)}


def process_memory_bytes() -> int | None:
    """This process's resident memory, or None when it cannot be read here (no extra dependency is used)."""
    if sys.platform == "win32":
        class Counters(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong), ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t), ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t), ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]
        try:
            from ctypes import wintypes

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE  # a pseudo-handle: must not be truncated to a 32-bit int
            get_info = kernel32.K32GetProcessMemoryInfo
            get_info.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
            get_info.restype = wintypes.BOOL
            counters = Counters()
            counters.cb = ctypes.sizeof(Counters)
            if get_info(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
                return int(counters.WorkingSetSize)
        except (AttributeError, OSError):
            return None
        return None
    try:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(peak if sys.platform == "darwin" else peak * 1024)
    except (ImportError, OSError):
        return None


def resources(data_folder: str | Path, database_bytes: int) -> dict:
    return {
        "disk": disk(data_folder),
        "databases_bytes": database_bytes,
        "process_memory_bytes": process_memory_bytes(),
        "threads": threading.active_count(),
        "python": ".".join(map(str, sys.version_info[:3])),
        "pid": os.getpid(),
    }
