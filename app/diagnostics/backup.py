"""
Verified backups (spec 456): a backup is a NEW folder `<backups>/<YYYYMMDD-HHMMSS>/` holding a consistent copy of every Jarvis
database (SQLite's backup API, so a database in use is still copied consistently), the identity files (personality.md,
user_profile.md), and BACKUP_MANIFEST.json recording each file's SHA-256, size and - for a database - its tables' row counts.

Each copy is verified BEFORE the backup is reported done: integrity check + the same tables and row counts as the original at the
moment of the copy. `verify_backup` re-checks a backup later against its manifest (files present, unchanged, still sound).

Never in a backup: `.env` (API keys) and the device credential - keys are re-entered on a new laptop (installer `keys`) and a
device is re-registered, as the master specification's reality check requires. Nothing here ever deletes or overwrites a file:
an existing backup folder name gets a suffix, and old backups are kept until the user removes them.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from app.diagnostics.checks import open_read_only, table_counts

MANIFEST = "BACKUP_MANIFEST.json"
MANIFEST_VERSION = 1
_NAME_FORMAT = "%Y%m%d-%H%M%S"


class BackupError(RuntimeError):
    """A backup could not be completed and verified; its partial folder is removed (it held only copies)."""


@dataclass
class BackupResult:
    folder: Path
    files: list[dict] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)  # sources that do not exist (yet)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _copy_database(source: Path, destination: Path) -> dict[str, int]:
    """A consistent copy, then proof: integrity ok and the same table row counts as the source had at the copy."""
    source_connection = open_read_only(source)
    try:
        target = sqlite3.connect(destination)
        try:
            source_connection.backup(target)
        finally:
            target.close()
        expected = table_counts(source_connection)
    finally:
        source_connection.close()
    check = open_read_only(destination)
    try:
        verdict = check.execute("PRAGMA integrity_check").fetchone()[0]
        copied = table_counts(check)
    finally:
        check.close()
    if verdict != "ok":
        raise BackupError(f"the copy of {source.name} failed its integrity check")
    # The source may gain rows between the copy and the count (Jarvis is running): a copy may only lag, never invent or lose
    # a table, and never hold MORE rows than the source.
    if set(copied) != set(expected) or any(copied[t] > expected[t] for t in copied):
        raise BackupError(f"the copy of {source.name} does not match its source")
    return copied


def _unique_folder(root: Path, now: datetime) -> Path:
    base = root / now.strftime(_NAME_FORMAT)
    candidate, number = base, 2
    while candidate.exists():
        candidate = base.with_name(f"{base.name}-{number}")
        number += 1
    return candidate


def create_backup(databases: list[tuple[str, Path]], files: list[Path], root: Path, now: datetime) -> BackupResult:
    """Back up `databases` ((setting name, path)) and plain `files` into a new folder under `root`, verified. Raises
    BackupError (and removes the partial new folder) if anything cannot be copied and verified."""
    root.mkdir(parents=True, exist_ok=True)
    folder = _unique_folder(root, now)
    folder.mkdir()
    result = BackupResult(folder=folder)
    try:
        for name, source in databases:
            source = Path(source)
            if not source.is_file():
                result.skipped.append(source.name)
                continue
            target = folder / source.name
            rows = _copy_database(source, target)
            result.files.append({"file": source.name, "kind": "database", "setting": name, "sha256": _sha256(target),
                                 "size_bytes": target.stat().st_size, "rows": rows})
        for source in files:
            source = Path(source)
            if not source.is_file():
                result.skipped.append(source.name)
                continue
            target = folder / source.name
            shutil.copy2(source, target)
            if _sha256(target) != _sha256(source):
                raise BackupError(f"the copy of {source.name} is not identical to the original")
            result.files.append({"file": source.name, "kind": "file", "sha256": _sha256(target),
                                 "size_bytes": target.stat().st_size})
        manifest = {"version": MANIFEST_VERSION, "created_at": now.isoformat(), "files": result.files,
                    "skipped": result.skipped, "not_included": [".env (API keys)", "device credential"]}
        (folder / MANIFEST).write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    except Exception as exc:
        shutil.rmtree(folder, ignore_errors=True)  # only the new folder of copies - never an original
        if isinstance(exc, BackupError):
            raise
        raise BackupError(f"the backup could not be completed ({type(exc).__name__})") from exc
    return result


def verify_backup(folder: Path) -> dict:
    """Re-check a backup against its manifest: every file present and unchanged (SHA-256), every database still sound and
    holding the recorded row counts. Returns {"ok": bool, "problems": [...], "files": n}."""
    manifest_path = Path(folder) / MANIFEST
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entries = manifest["files"]
        if manifest.get("version") != MANIFEST_VERSION or not isinstance(entries, list):
            raise ValueError("unsupported manifest")
    except (OSError, ValueError, KeyError, TypeError):
        return {"ok": False, "problems": ["the manifest is missing or unreadable"], "files": 0}
    problems = []
    for entry in entries:
        name = str(entry.get("file", ""))
        path = Path(folder) / name
        if not name or os.path.basename(name) != name or not path.is_file():
            problems.append(f"{name or '(unnamed)'}: missing")
            continue
        if _sha256(path) != entry.get("sha256"):
            problems.append(f"{name}: changed since the backup")
            continue
        if entry.get("kind") == "database":
            try:
                connection = open_read_only(path)
                try:
                    verdict = connection.execute("PRAGMA integrity_check").fetchone()[0]
                    rows = table_counts(connection)
                finally:
                    connection.close()
            except sqlite3.DatabaseError:
                problems.append(f"{name}: not a readable database")
                continue
            if verdict != "ok":
                problems.append(f"{name}: failed its integrity check")
            elif rows != entry.get("rows"):
                problems.append(f"{name}: row counts differ from the manifest")
    return {"ok": not problems, "problems": problems, "files": len(entries)}


def list_backups(root: Path) -> list[dict]:
    """The backups under `root`, newest first (folders with a manifest only; nothing is verified here - see verify_backup)."""
    if not Path(root).is_dir():
        return []
    found = []
    for folder in Path(root).iterdir():
        manifest_path = folder / MANIFEST
        if not folder.is_dir() or not manifest_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            files = manifest.get("files", [])
            found.append({"name": folder.name, "created_at": manifest.get("created_at"), "files": len(files),
                          "size_bytes": sum(int(f.get("size_bytes", 0)) for f in files if isinstance(f, dict))})
        except (OSError, ValueError, AttributeError, TypeError):
            found.append({"name": folder.name, "created_at": None, "files": 0, "size_bytes": 0, "unreadable": True})
    return sorted(found, key=lambda item: item["name"], reverse=True)
