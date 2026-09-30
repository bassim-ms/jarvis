"""
The write journal and the candidate staging area - the ONE small mechanism this
package adds, because the existing stores cannot hold what the policy needs:

- MemoryStore facts are (key -> value, updated_at): an update overwrites, and
  there is no provenance and no history;
- a World Model Preference is READ AS A FACT by the context prompts
  ("key=value"), so an uncertain candidate must not live there.

So:

  write_journal     append-only: every decision the policy made and every write
                    that followed, with the PREVIOUS value when one was replaced
                    (so a contradiction never destroys history), the origin, the
                    provenance and the reason. The values of anything that was
                    refused, asked about or not persisted are NEVER recorded here:
                    a secret or a sensitive line must not reach disk through the
                    audit trail either.
  write_candidates  what Jarvis was told uncertainly or inferred: staged, with
                    confidence, the user's own words, and a timestamp. Nothing
                    reads a candidate as a fact - not the Context Engine, not a
                    planner. It becomes a fact only when the user states it
                    plainly (promotion), and then it is written through the
                    ordinary WorldModelService / MemoryStore path.

Not a memory system: it is provenance for the two that exist.
"""

from __future__ import annotations

import sqlite3
import threading
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone

from app.memory_policy.base import CANDIDATE_CONFIDENCE_CEILING, CORROBORATION_STEP


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class JournalEntry:
    id: str
    at: str
    action: str  # persisted | updated | unchanged | candidate | corroborated | promoted | asked | rejected | temporary
    kind: str
    key: str
    value: str | None
    previous_value: str | None
    origin: str
    destination: str
    source: str | None
    confidence: float | None
    reason: str
    ref: str | None


@dataclass
class Candidate:
    id: str
    kind: str
    key: str
    value: str
    category: str
    origin: str
    confidence: float
    evidence: str
    status: str  # candidate | promoted
    created_at: str
    updated_at: str


class WriteStore(ABC):
    @abstractmethod
    def record(
        self,
        *,
        action: str,
        kind: str,
        key: str,
        value: str | None,
        previous_value: str | None,
        origin: str,
        destination: str,
        source: str | None,
        confidence: float | None,
        reason: str,
        ref: str | None = None,
    ) -> JournalEntry: ...

    @abstractmethod
    def journal(self, key: str | None = None, kind: str | None = None) -> list[JournalEntry]:
        """Newest first."""

    @abstractmethod
    def stage_candidate(
        self, *, kind: str, key: str, value: str, category: str, origin: str, confidence: float, evidence: str
    ) -> tuple[Candidate, bool]:
        """Stage (or corroborate) the ONE candidate for (kind, key). The same
        value again is a corroboration (+CORROBORATION_STEP, never past the
        ceiling); a different value replaces it. Returns (candidate, corroborated)."""

    @abstractmethod
    def get_candidate(self, kind: str, key: str) -> Candidate | None: ...

    @abstractmethod
    def list_candidates(self, status: str | None = "candidate") -> list[Candidate]: ...

    @abstractmethod
    def promote_candidate(self, candidate_id: str) -> None: ...


class SQLiteWriteStore(WriteStore):
    def __init__(self, db_path: str):
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS write_journal (
                    id TEXT PRIMARY KEY,
                    at TEXT NOT NULL,
                    action TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT,
                    previous_value TEXT,
                    origin TEXT NOT NULL,
                    destination TEXT NOT NULL,
                    source TEXT,
                    confidence REAL,
                    reason TEXT NOT NULL,
                    ref TEXT
                );
                CREATE TABLE IF NOT EXISTS write_candidates (
                    id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    category TEXT NOT NULL,
                    origin TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    evidence TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )
            self._conn.commit()

    # -- journal -------------------------------------------------------------------------------

    def record(
        self,
        *,
        action,
        kind,
        key,
        value,
        previous_value,
        origin,
        destination,
        source,
        confidence,
        reason,
        ref=None,
    ) -> JournalEntry:
        entry = JournalEntry(
            id=uuid.uuid4().hex,
            at=_now(),
            action=action,
            kind=kind,
            key=key,
            value=value,
            previous_value=previous_value,
            origin=origin,
            destination=destination,
            source=source,
            confidence=confidence,
            reason=reason,
            ref=ref,
        )
        with self._lock:
            self._conn.execute(
                "INSERT INTO write_journal VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    entry.id, entry.at, entry.action, entry.kind, entry.key, entry.value, entry.previous_value,
                    entry.origin, entry.destination, entry.source, entry.confidence, entry.reason, entry.ref,
                ),
            )
            self._conn.commit()
        return entry

    def journal(self, key: str | None = None, kind: str | None = None) -> list[JournalEntry]:
        clauses, params = [], []
        if key is not None:
            clauses.append("key = ?")
            params.append(key)
        if kind is not None:
            clauses.append("kind = ?")
            params.append(kind)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM write_journal {where} ORDER BY at DESC, rowid DESC", params
            ).fetchall()
        return [JournalEntry(**dict(r)) for r in rows]

    # -- candidates ------------------------------------------------------------------------------

    @staticmethod
    def _candidate(row: sqlite3.Row) -> Candidate:
        return Candidate(**dict(row))

    def get_candidate(self, kind: str, key: str) -> Candidate | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM write_candidates WHERE kind = ? AND key = ? AND status = 'candidate'", (kind, key)
            ).fetchone()
        return self._candidate(row) if row else None

    def list_candidates(self, status: str | None = "candidate") -> list[Candidate]:
        with self._lock:
            if status is None:
                rows = self._conn.execute("SELECT * FROM write_candidates ORDER BY updated_at DESC").fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM write_candidates WHERE status = ? ORDER BY updated_at DESC", (status,)
                ).fetchall()
        return [self._candidate(r) for r in rows]

    def stage_candidate(
        self, *, kind, key, value, category, origin, confidence, evidence
    ) -> tuple[Candidate, bool]:
        from app.memory_policy.policy import normalize

        now = _now()
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM write_candidates WHERE kind = ? AND key = ? AND status = 'candidate'", (kind, key)
            ).fetchone()
            if row is not None and normalize(row["value"]) == normalize(value):
                confidence_now = min(row["confidence"] + CORROBORATION_STEP, CANDIDATE_CONFIDENCE_CEILING)
                self._conn.execute(
                    "UPDATE write_candidates SET confidence = ?, updated_at = ? WHERE id = ?",
                    (confidence_now, now, row["id"]),
                )
                self._conn.commit()
                fresh = self._conn.execute("SELECT * FROM write_candidates WHERE id = ?", (row["id"],)).fetchone()
                return self._candidate(fresh), True
            if row is not None:  # a different value for the same key replaces it
                self._conn.execute(
                    """UPDATE write_candidates SET value = ?, category = ?, origin = ?, confidence = ?,
                       evidence = ?, updated_at = ? WHERE id = ?""",
                    (value, category, origin, confidence, evidence[:200], now, row["id"]),
                )
                self._conn.commit()
                fresh = self._conn.execute("SELECT * FROM write_candidates WHERE id = ?", (row["id"],)).fetchone()
                return self._candidate(fresh), False
            candidate_id = uuid.uuid4().hex
            self._conn.execute(
                "INSERT INTO write_candidates VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (candidate_id, kind, key, value, category, origin, confidence, evidence[:200], "candidate", now, now),
            )
            self._conn.commit()
            fresh = self._conn.execute("SELECT * FROM write_candidates WHERE id = ?", (candidate_id,)).fetchone()
            return self._candidate(fresh), False

    def promote_candidate(self, candidate_id: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE write_candidates SET status = 'promoted', updated_at = ? WHERE id = ?", (_now(), candidate_id)
            )
            self._conn.commit()
