"""
One reliability report (spec 460, partial) over the checks in this package: service health (458), database integrity (457),
resources (452) and backups (456). Built from callables the application hands in (app/main.py), so it reports on the running
app's own objects and is tested with fakes.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from app.diagnostics.backup import BackupError, create_backup, list_backups, verify_backup
from app.diagnostics.checks import check_database, resources

#: Below either, disk space is reported as a warning (Jarvis's databases, logs and backups need room).
LOW_DISK_BYTES = 1024**3
LOW_DISK_PERCENT = 5.0
_BACKUP_NAME = re.compile(r"\d{8}-\d{6}(-\d+)?")

#: A service probe returns (ok, detail): True = healthy, False = a problem, None = not expected to run (e.g. off by setting).
ServiceProbe = Callable[[], tuple[bool | None, str]]


@dataclass
class Diagnostics:
    databases: list[tuple[str, Path]]
    data_folder: Path
    backup_root: Path
    services: dict[str, ServiceProbe] = field(default_factory=dict)
    backup_files: list[Path] = field(default_factory=list)
    clock: Callable[[], datetime] = lambda: datetime.now().astimezone()
    started_at: datetime | None = None

    def report(self) -> dict:
        now = self.clock()
        problems: list[str] = []
        warnings: list[str] = []

        services = []
        for name, probe in self.services.items():
            try:
                ok, detail = probe()
            except Exception as exc:  # noqa: BLE001 - a probe that breaks is itself a finding, not a crash
                ok, detail = False, f"could not be checked ({type(exc).__name__})"
            services.append({"name": name, "ok": ok, "detail": detail})
            if ok is False:
                problems.append(f"{name}: {detail}")

        databases = [check_database(name, path) for name, path in self.databases]
        for check in databases:
            if check.ok is False:
                problems.append(f"{check.file}: {check.detail}")

        try:
            reading = resources(self.data_folder, sum(check.size_bytes for check in databases))
        except OSError as exc:
            reading = {"disk": None, "error": type(exc).__name__}
            warnings.append("disk space could not be read")
        disk = reading.get("disk")
        if disk and (disk["free_bytes"] < LOW_DISK_BYTES or disk["free_percent"] < LOW_DISK_PERCENT):
            warnings.append(f"low disk space: {disk['free_bytes'] / 1024**3:.1f} GB free")

        backups = list_backups(self.backup_root)
        if not backups:
            warnings.append("no backup yet")

        return {
            "status": "failing" if problems else ("warning" if warnings else "ok"),
            "problems": problems,
            "warnings": warnings,
            "checked_at": now.isoformat(),
            "uptime_seconds": None if self.started_at is None else max(0, round((now - self.started_at).total_seconds())),
            "data_folder": str(self.data_folder),
            "services": services,
            "databases": [check.to_dict() for check in databases],
            "resources": reading,
            "backups": {"count": len(backups), "latest": backups[0] if backups else None},
        }

    def backup(self) -> dict:
        result = create_backup(self.databases, self.backup_files, self.backup_root, self.clock())
        verification = verify_backup(result.folder)
        if not verification["ok"]:  # created and verified once already; a second look must agree before it is reported done
            raise BackupError("the new backup did not pass its own verification")
        return {"name": result.folder.name, "files": len(result.files), "skipped": result.skipped,
                "size_bytes": sum(f["size_bytes"] for f in result.files), "verified": True}

    def backups(self) -> list[dict]:
        return list_backups(self.backup_root)

    def verify(self, name: str) -> dict:
        """Re-verify one backup by its folder name. Only a name this package creates is accepted - no path is ever built from
        anything else (no "..", no separators)."""
        if not _BACKUP_NAME.fullmatch(name or ""):
            raise ValueError("not a backup name")
        folder = self.backup_root / name
        if not folder.is_dir():
            raise FileNotFoundError(name)
        return {"name": name, **verify_backup(folder)}
