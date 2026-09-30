"""
Device storage: one table in the application's EXISTING database (the Job store's connection and lock, the same
way the attention queue and the activity context are stored) - not a second database.

Only a HASH of each device's credential is stored (`credential_hash`), never the credential. It is kept in its
own column and read only by `credential_hash()` (for the gateway's comparison); it is never part of a `Device`,
so no listing, context or log built from a device can contain it. Revoking a device wipes its hash.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from abc import ABC, abstractmethod
from datetime import datetime, timezone

from app.devices.base import Capability, Device, DeviceError, DeviceStatus, DeviceType, TrustState


def name_key(name: str) -> str:
    return " ".join(name.casefold().split())


class DeviceStore(ABC):
    @abstractmethod
    def add(self, device: Device, credential_hash: str) -> None:
        """Store a new device. DeviceError if its id, or its (case-insensitive) name among non-revoked devices, is taken."""

    @abstractmethod
    def get(self, device_id: str) -> Device | None: ...

    @abstractmethod
    def get_by_name(self, name: str) -> Device | None:
        """The device with that name (case-insensitive); a non-revoked one in preference to a revoked one."""

    @abstractmethod
    def list(self) -> list[Device]:
        """Every device, oldest first."""

    @abstractmethod
    def credential_hash(self, device_id: str) -> str | None:
        """The stored credential hash, or None (unknown device, or revoked)."""

    @abstractmethod
    def set_trust(self, device_id: str, state: TrustState, now: datetime) -> None:
        """Move a device to `state`. REVOKED also wipes its credential hash and marks it OFFLINE."""

    @abstractmethod
    def record_check_in(self, device_id: str, now: datetime) -> None:
        """ONLINE, last_seen = now."""

    @abstractmethod
    def set_status(self, device_id: str, status: DeviceStatus, now: datetime) -> None: ...


def _iso(value: datetime | None) -> str | None:
    return value.astimezone(timezone.utc).isoformat() if value is not None else None


def _parse(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


_COLUMNS = "device_id, name, device_type, capabilities, status, trust_state, last_seen, created_at, updated_at"


def _device(row) -> Device:
    return Device(
        device_id=row[0],
        name=row[1],
        device_type=DeviceType(row[2]),
        capabilities=tuple(Capability(c) for c in json.loads(row[3])),
        status=DeviceStatus(row[4]),
        trust_state=TrustState(row[5]),
        last_seen=_parse(row[6]),
        created_at=_parse(row[7]),
        updated_at=_parse(row[8]),
    )


class SQLiteDeviceStore(DeviceStore):
    def __init__(self, connection: sqlite3.Connection, lock: threading.Lock):
        self._conn = connection
        self._lock = lock
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS devices (
                    device_id       TEXT PRIMARY KEY,
                    name            TEXT NOT NULL,
                    name_key        TEXT NOT NULL,
                    device_type     TEXT NOT NULL,
                    capabilities    TEXT NOT NULL,
                    status          TEXT NOT NULL,
                    trust_state     TEXT NOT NULL,
                    last_seen       TEXT,
                    created_at      TEXT NOT NULL,
                    updated_at      TEXT NOT NULL,
                    credential_hash TEXT
                )
                """
            )
            # A name is unique among devices that are not revoked; a revoked device's name can be reused by
            # its replacement (a NEW registration).
            self._conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS ux_devices_active_name ON devices (name_key) "
                "WHERE trust_state != 'revoked'"
            )
            self._conn.commit()

    @classmethod
    def for_job_store(cls, job_store) -> "SQLiteDeviceStore":
        return cls(job_store.connection, job_store.lock)

    def _write(self, sql: str, params: tuple) -> int:
        with self._lock:
            try:
                cursor = self._conn.execute(sql, params)
                self._conn.commit()
                return cursor.rowcount
            except BaseException:
                self._conn.rollback()
                raise

    def add(self, device: Device, credential_hash: str) -> None:
        try:
            self._write(
                f"INSERT INTO devices ({_COLUMNS}, name_key, credential_hash) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    device.device_id, device.name, device.device_type.value,
                    json.dumps([c.value for c in device.capabilities]), device.status.value, device.trust_state.value,
                    _iso(device.last_seen), _iso(device.created_at), _iso(device.updated_at),
                    name_key(device.name), credential_hash,
                ),
            )
        except sqlite3.IntegrityError as exc:
            raise DeviceError("a device with that id or name already exists") from exc

    def get(self, device_id: str) -> Device | None:
        with self._lock:
            row = self._conn.execute(f"SELECT {_COLUMNS} FROM devices WHERE device_id = ?", (device_id,)).fetchone()
        return _device(row) if row else None

    def get_by_name(self, name: str) -> Device | None:
        with self._lock:
            row = self._conn.execute(
                f"SELECT {_COLUMNS} FROM devices WHERE name_key = ? "
                "ORDER BY (trust_state = 'revoked'), created_at DESC, rowid DESC LIMIT 1",
                (name_key(name),),
            ).fetchone()
        return _device(row) if row else None

    def list(self) -> list[Device]:
        with self._lock:
            rows = self._conn.execute(f"SELECT {_COLUMNS} FROM devices ORDER BY created_at, rowid").fetchall()
        return [_device(r) for r in rows]

    def credential_hash(self, device_id: str) -> str | None:
        with self._lock:
            row = self._conn.execute("SELECT credential_hash FROM devices WHERE device_id = ?", (device_id,)).fetchone()
        return row[0] if row else None

    def set_trust(self, device_id: str, state: TrustState, now: datetime) -> None:
        if state is TrustState.REVOKED:
            changed = self._write(
                "UPDATE devices SET trust_state = ?, status = ?, credential_hash = NULL, updated_at = ? WHERE device_id = ?",
                (state.value, DeviceStatus.OFFLINE.value, _iso(now), device_id),
            )
        else:
            changed = self._write(
                "UPDATE devices SET trust_state = ?, updated_at = ? WHERE device_id = ?", (state.value, _iso(now), device_id)
            )
        if not changed:
            raise DeviceError("unknown device")

    def record_check_in(self, device_id: str, now: datetime) -> None:
        self._write(
            "UPDATE devices SET status = ?, last_seen = ?, updated_at = ? WHERE device_id = ?",
            (DeviceStatus.ONLINE.value, _iso(now), _iso(now), device_id),
        )

    def set_status(self, device_id: str, status: DeviceStatus, now: datetime) -> None:
        self._write("UPDATE devices SET status = ?, updated_at = ? WHERE device_id = ?", (status.value, _iso(now), device_id))
