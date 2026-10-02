"""
SQLiteScheduledJobStore: the Scheduled Job record and its occurrence ledger in the orchestration database file.

It is built to share the Job store's connection and lock (`for_job_store`, the same arrangement the device store uses), so that
acting on an occurrence can write the ledger row and the Job in ONE transaction - and can be opened on its own (`open`) for a
stand-alone database. It creates only its own two tables; it never reads or writes the `jobs` table itself: the Job to insert arrives
as a callable (`record_run(insert_job=...)`) that the caller - with the existing Job store's own `insert_job_locked` - runs inside the
transaction. The store therefore cannot create, plan, run or change a Job.

The invariants are enforced by the DATABASE, not only by code:
  * `UNIQUE(schedule_id, occurrence)` - one occurrence of one Scheduled Job is acted on once.
  * `UNIQUE(job_id)` - a Job belongs to at most one run.
  * `CHECK` - a run has a Job exactly when its outcome is CREATED / CREATED_LATE; a result is recorded only for a run with a Job;
    an ACTIVE Scheduled Job always has a next due time; statuses and outcomes are the known ones.
  * every mutation is a compare-and-set on `version` (and, for a scheduled run, on `due_at`), so a stale or racing writer changes nothing.

Datetimes are stored as fixed-width UTC text so that comparing them as text is comparing them in time.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from typing import Any

from app.scheduled_jobs.base import (
    JOB_OUTCOMES,
    KNOWN_PROFILES,
    MAX_EVENT_DETAIL_CHARS,
    MAX_EVENTS,
    MAX_ID_CHARS,
    OPEN_STATUSES,
    RESULT_STATES,
    DuplicateOccurrenceError,
    InvalidTransitionError,
    NewRun,
    RunKind,
    RunOutcome,
    ScheduledJob,
    ScheduledJobError,
    ScheduledJobEvent,
    ScheduledJobRun,
    ScheduledJobStatus,
    ScheduledJobStore,
    Settled,
    StaleScheduledJobError,
    StatusReason,
    check_transition,
    valid_via,
)
from app.scheduled_jobs.policy import breaker_decision, clean_intent, clean_title, validate_objective
from app.scheduling.schedule import Schedule, ScheduleError, ScheduleKind

_S, _R = ScheduledJobStatus, StatusReason

_JOB_COLUMNS = (
    "id, status, status_reason, title, intent_text, objective, schedule, profile, due_at, approved_at, "
    "approval_interaction_id, proposal_expires_at, requested_until, expires_at, created_via_conversation_id, "
    "created_via_device_id, runs_total, consecutive_failures, last_job_id, last_run_at, events, version, created_at, updated_at"
)
_RUN_COLUMNS = (
    "id, schedule_id, occurrence, kind, scheduled_for, outcome, job_id, reason, missed_count, late, created_at, "
    "result_state, settled_at, reported_at"
)

_STATUS_VALUES = ", ".join(f"'{s.value}'" for s in ScheduledJobStatus)
_OUTCOME_VALUES = ", ".join(f"'{o.value}'" for o in RunOutcome)
_JOB_OUTCOME_VALUES = ", ".join(f"'{o.value}'" for o in JOB_OUTCOMES)
_RESULT_VALUES = ", ".join(f"'{r}'" for r in sorted(RESULT_STATES))
_KIND_VALUES = ", ".join(f"'{k.value}'" for k in RunKind)

_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS scheduled_jobs (
    id TEXT PRIMARY KEY,
    status TEXT NOT NULL CHECK (status IN ({_STATUS_VALUES})),
    status_reason TEXT,
    title TEXT NOT NULL,
    intent_text TEXT NOT NULL,
    objective TEXT NOT NULL,
    schedule TEXT NOT NULL,
    profile TEXT NOT NULL,
    due_at TEXT,
    approved_at TEXT,
    approval_interaction_id TEXT,
    proposal_expires_at TEXT,
    requested_until TEXT,
    expires_at TEXT,
    created_via_conversation_id TEXT,
    created_via_device_id TEXT,
    runs_total INTEGER NOT NULL DEFAULT 0,
    consecutive_failures INTEGER NOT NULL DEFAULT 0,
    last_job_id TEXT,
    last_run_at TEXT,
    events TEXT NOT NULL,
    version INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    CHECK (status != 'active' OR due_at IS NOT NULL),
    CHECK (status NOT IN ('active', 'paused', 'completed') OR approved_at IS NOT NULL)
);
CREATE INDEX IF NOT EXISTS ix_scheduled_jobs_due ON scheduled_jobs (status, due_at);
CREATE TABLE IF NOT EXISTS scheduled_job_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    schedule_id TEXT NOT NULL,
    occurrence TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ({_KIND_VALUES})),
    scheduled_for TEXT NOT NULL,
    outcome TEXT NOT NULL CHECK (outcome IN ({_OUTCOME_VALUES})),
    job_id TEXT,
    reason TEXT,
    missed_count INTEGER NOT NULL DEFAULT 0,
    late INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    result_state TEXT CHECK (result_state IS NULL OR result_state IN ({_RESULT_VALUES})),
    settled_at TEXT,
    reported_at TEXT,
    CHECK ((outcome IN ({_JOB_OUTCOME_VALUES})) = (job_id IS NOT NULL)),
    CHECK (result_state IS NULL OR job_id IS NOT NULL),
    UNIQUE (schedule_id, occurrence),
    UNIQUE (job_id)
);
CREATE INDEX IF NOT EXISTS ix_scheduled_job_runs_open ON scheduled_job_runs (result_state, job_id);
"""


def _utc(value: datetime | None = None) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ScheduledJobError("Times must be timezone-aware datetimes.")
    return value.astimezone(timezone.utc)


def _iso(value: datetime | None) -> str | None:
    """Fixed-width UTC text: sorts and compares as time."""
    return None if value is None else _utc(value).strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")


def _parse(value: str | None) -> datetime | None:
    return None if value is None else datetime.fromisoformat(value)


def _event(now: datetime, kind: str, detail: str, via: str) -> dict:
    return {"timestamp": _iso(now), "kind": kind, "detail": " ".join(str(detail).split())[:MAX_EVENT_DETAIL_CHARS], "via": via}


def _events_json(existing: list[ScheduledJobEvent], *new: dict) -> str:
    items = [{"timestamp": e.timestamp, "kind": e.kind, "detail": e.detail, "via": e.via} for e in existing] + list(new)
    return json.dumps(items[-MAX_EVENTS:], separators=(",", ":"))


def _job_from_row(row) -> ScheduledJob:
    (
        id_, status, reason, title, intent, objective, schedule, profile, due_at, approved_at, approval_id, proposal_exp,
        requested_until, expires_at, conv_id, device_id, runs_total, streak, last_job_id, last_run_at, events, version,
        created_at, updated_at,
    ) = tuple(row)
    return ScheduledJob(
        id=id_,
        status=ScheduledJobStatus(status),
        status_reason=StatusReason(reason) if reason else None,
        title=title,
        intent_text=intent,
        objective=json.loads(objective),
        schedule=Schedule.from_dict(json.loads(schedule)),
        profile=profile,
        due_at=_parse(due_at),
        approved_at=_parse(approved_at),
        approval_interaction_id=approval_id,
        proposal_expires_at=_parse(proposal_exp),
        requested_until=_parse(requested_until),
        expires_at=_parse(expires_at),
        created_via_conversation_id=conv_id,
        created_via_device_id=device_id,
        runs_total=runs_total,
        consecutive_failures=streak,
        last_job_id=last_job_id,
        last_run_at=_parse(last_run_at),
        events=[ScheduledJobEvent(e["timestamp"], e["kind"], e.get("detail", ""), e.get("via", "system")) for e in json.loads(events)],
        version=version,
        created_at=_parse(created_at),
        updated_at=_parse(updated_at),
    )


def _run_from_row(row) -> ScheduledJobRun:
    (id_, schedule_id, occurrence, kind, scheduled_for, outcome, job_id, reason, missed, late, created_at, result_state, settled_at, reported_at) = tuple(row)
    return ScheduledJobRun(
        id=id_,
        schedule_id=schedule_id,
        occurrence=occurrence,
        kind=RunKind(kind),
        scheduled_for=_parse(scheduled_for),
        outcome=RunOutcome(outcome),
        job_id=job_id,
        reason=reason,
        missed_count=missed,
        late=bool(late),
        created_at=_parse(created_at),
        result_state=result_state,
        settled_at=_parse(settled_at),
        reported_at=_parse(reported_at),
    )


class SQLiteScheduledJobStore(ScheduledJobStore):
    def __init__(self, connection: sqlite3.Connection, lock: threading.Lock):
        self._conn = connection
        self._lock = lock
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    @classmethod
    def for_job_store(cls, job_store) -> "SQLiteScheduledJobStore":
        """On the Job store's own connection and lock, so a Job and its ledger row can be written in one transaction."""
        return cls(job_store.connection, job_store.lock)

    @classmethod
    def open(cls, db_path: str) -> "SQLiteScheduledJobStore":
        """On a connection of its own (a stand-alone database, or a second process's view of the same file)."""
        connection = sqlite3.connect(db_path, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        return cls(connection, threading.Lock())

    # -- reading ------------------------------------------------------------------------------------------------------------

    def _select(self, scheduled_job_id: str) -> ScheduledJob | None:
        row = self._conn.execute(f"SELECT {_JOB_COLUMNS} FROM scheduled_jobs WHERE id = ?", (scheduled_job_id,)).fetchone()
        return _job_from_row(row) if row is not None else None

    def get(self, scheduled_job_id: str) -> ScheduledJob | None:
        with self._lock:
            return self._select(scheduled_job_id) if isinstance(scheduled_job_id, str) else None

    def list(self, statuses: frozenset[ScheduledJobStatus] | None = None) -> list[ScheduledJob]:
        with self._lock:
            if statuses is None:
                rows = self._conn.execute(f"SELECT {_JOB_COLUMNS} FROM scheduled_jobs ORDER BY created_at DESC, id").fetchall()
            else:
                marks = ", ".join("?" for _ in statuses) or "NULL"
                rows = self._conn.execute(
                    f"SELECT {_JOB_COLUMNS} FROM scheduled_jobs WHERE status IN ({marks}) ORDER BY created_at DESC, id",
                    tuple(s.value for s in statuses),
                ).fetchall()
        return [_job_from_row(r) for r in rows]

    def list_due(self, now: datetime, limit: int = 20) -> list[ScheduledJob]:
        with self._lock:
            rows = self._conn.execute(
                f"SELECT {_JOB_COLUMNS} FROM scheduled_jobs WHERE status = 'active' AND due_at <= ? ORDER BY due_at, id LIMIT ?",
                (_iso(now), max(1, int(limit))),
            ).fetchall()
        return [_job_from_row(r) for r in rows]

    def count(self, statuses: frozenset[ScheduledJobStatus]) -> int:
        marks = ", ".join("?" for _ in statuses) or "NULL"
        with self._lock:
            return self._conn.execute(
                f"SELECT COUNT(*) FROM scheduled_jobs WHERE status IN ({marks})", tuple(s.value for s in statuses)
            ).fetchone()[0]

    # -- creating and approving ----------------------------------------------------------------------------------------------

    def create_proposal(
        self,
        *,
        title: str,
        intent_text: str,
        objective: dict[str, Any],
        schedule: Schedule,
        proposal_expires_at: datetime,
        requested_until: datetime | None = None,
        profile: str = "unattended_v1",
        created_via_conversation_id: str | None = None,
        created_via_device_id: str | None = None,
        now: datetime | None = None,
    ) -> ScheduledJob:
        now = _utc(now)
        if not isinstance(schedule, Schedule):
            raise ScheduledJobError("A Scheduled Job needs a Schedule.")
        if profile not in KNOWN_PROFILES:
            raise ScheduledJobError(f"Unknown capability profile '{profile}'.")
        expires = _utc(proposal_expires_at)
        if expires <= now:
            raise ScheduledJobError("A proposal has to expire in the future.")
        until = _utc(requested_until) if requested_until is not None else None
        for name, value in (("conversation", created_via_conversation_id), ("device", created_via_device_id)):
            if value is not None and (not isinstance(value, str) or not value or len(value) > MAX_ID_CHARS):
                raise ScheduledJobError(f"The {name} id must be short text.")
        record = ScheduledJob(
            id=uuid.uuid4().hex,
            status=_S.PROPOSED,
            title=clean_title(title),
            intent_text=clean_intent(intent_text),
            objective=validate_objective(objective),
            schedule=schedule,
            profile=profile,
            created_at=now,
            updated_at=now,
            proposal_expires_at=expires,
            requested_until=until,
            created_via_conversation_id=created_via_conversation_id,
            created_via_device_id=created_via_device_id,
        )
        events = _events_json([], _event(now, "proposed", "awaiting approval", "system"))
        with self._lock:
            try:
                self._conn.execute(
                    f"INSERT INTO scheduled_jobs ({_JOB_COLUMNS}) VALUES ({', '.join('?' for _ in range(24))})",
                    (
                        record.id, _S.PROPOSED.value, None, record.title, record.intent_text,
                        json.dumps(record.objective, separators=(",", ":"), ensure_ascii=False),
                        json.dumps(schedule.to_dict(), separators=(",", ":")), profile, None, None, None, _iso(expires), _iso(until), None,
                        created_via_conversation_id, created_via_device_id, 0, 0, None, None, events, 0, _iso(now), _iso(now),
                    ),
                )
                self._conn.commit()
            except BaseException:
                self._conn.rollback()
                raise
        record.events = [ScheduledJobEvent(_iso(now), "proposed", "awaiting approval", "system")]
        return record

    def _guarded(self, scheduled_job_id: str, expected_version: int) -> ScheduledJob:
        """The row as the caller read it, or StaleScheduledJobError. Caller holds the lock."""
        current = self._select(scheduled_job_id) if isinstance(scheduled_job_id, str) else None
        if current is None or current.version != expected_version:
            raise StaleScheduledJobError("This Scheduled Job changed (or is gone) since it was read - nothing was changed.")
        return current

    def _apply(self, current: ScheduledJob, assignments: dict[str, Any]) -> ScheduledJob:
        """UPDATE with `version = version + 1`, guarded by the version just read. Caller holds the lock and commits."""
        columns = ", ".join(f"{name} = ?" for name in assignments)
        cursor = self._conn.execute(
            f"UPDATE scheduled_jobs SET {columns}, version = version + 1 WHERE id = ? AND version = ?",
            (*assignments.values(), current.id, current.version),
        )
        if cursor.rowcount != 1:
            raise StaleScheduledJobError("This Scheduled Job changed (or is gone) since it was read - nothing was changed.")
        return self._select(current.id)

    def _mutate(self, scheduled_job_id: str, expected_version: int, build: Callable[[ScheduledJob], dict[str, Any]]) -> ScheduledJob:
        with self._lock:
            try:
                current = self._guarded(scheduled_job_id, expected_version)
                updated = self._apply(current, build(current))
                self._conn.commit()
                return updated
            except BaseException:
                self._conn.rollback()
                raise

    def activate(
        self,
        scheduled_job_id: str,
        *,
        expected_version: int,
        approved_at: datetime,
        approval_interaction_id: str,
        due_at: datetime,
        expires_at: datetime | None,
        via: str,
    ) -> ScheduledJob:
        valid_via(via)
        approved, first, end = _utc(approved_at), _utc(due_at), _utc(expires_at) if expires_at is not None else None
        if not isinstance(approval_interaction_id, str) or not approval_interaction_id or len(approval_interaction_id) > MAX_ID_CHARS:
            raise ScheduledJobError("An approval has to name the confirmation it came from.")

        def build(current: ScheduledJob) -> dict[str, Any]:
            if current.status is not _S.PROPOSED:
                raise InvalidTransitionError(f"Only a proposal can be approved; this one is {current.status.value}.")
            check_transition(current.status, _S.ACTIVE, None)
            if current.proposal_expires_at is not None and approved > current.proposal_expires_at:
                raise InvalidTransitionError("This proposal expired before it was approved.")
            if first <= approved:
                raise ScheduledJobError("The first run has to be after the approval.")
            one_time = current.schedule.kind is ScheduleKind.ONCE
            if one_time and end is not None:
                raise ScheduledJobError("A one-time Scheduled Job has no lifetime.")
            if not one_time and end is None:
                raise ScheduledJobError("A repeating Scheduled Job needs a lifetime (its approval does not last forever).")
            if end is not None and first >= end:
                raise ScheduledJobError("The first run would fall after the lifetime ends.")
            event = _event(approved, "approved", f"first run {_iso(first)}", via)
            return {
                "status": _S.ACTIVE.value, "status_reason": None, "approved_at": _iso(approved),
                "approval_interaction_id": approval_interaction_id, "due_at": _iso(first), "expires_at": _iso(end),
                "events": _events_json(current.events, event), "updated_at": _iso(approved),
            }

        return self._mutate(scheduled_job_id, expected_version, build)

    # -- the lifecycle ---------------------------------------------------------------------------------------------------------

    def transition(
        self,
        scheduled_job_id: str,
        target: ScheduledJobStatus,
        reason: StatusReason | None,
        *,
        expected_version: int,
        via: str,
        detail: str = "",
        now: datetime | None = None,
    ) -> ScheduledJob:
        valid_via(via)
        moment = _utc(now)
        if target is _S.ACTIVE:
            raise InvalidTransitionError("Use activate (approval) or resume to make a Scheduled Job active.")

        def build(current: ScheduledJob) -> dict[str, Any]:
            check_transition(current.status, target, reason)
            label = f"{target.value}:{reason.value}" if reason else target.value
            return {
                "status": target.value, "status_reason": reason.value if reason else None,
                "events": _events_json(current.events, _event(moment, label, detail, via)), "updated_at": _iso(moment),
            }

        return self._mutate(scheduled_job_id, expected_version, build)

    def resume(self, scheduled_job_id: str, *, next_due_at: datetime, expected_version: int, via: str) -> ScheduledJob:
        valid_via(via)
        due = _utc(next_due_at)
        moment = _utc()

        def build(current: ScheduledJob) -> dict[str, Any]:
            if current.status is not _S.PAUSED:
                raise InvalidTransitionError(f"Only a paused Scheduled Job can be resumed; this one is {current.status.value}.")
            check_transition(current.status, _S.ACTIVE, None)
            if current.expires_at is not None and due >= current.expires_at:
                raise ScheduledJobError("The next run would fall after the lifetime ends.")
            changes: dict[str, Any] = {
                "status": _S.ACTIVE.value, "status_reason": None, "due_at": _iso(due),
                "events": _events_json(current.events, _event(moment, "resumed", f"next run {_iso(due)}", via)),
                "updated_at": _iso(moment),
            }
            if current.status_reason is _R.BREAKER:
                changes["consecutive_failures"] = 0  # the user looked at it and chose to carry on: a fresh streak
            return changes

        return self._mutate(scheduled_job_id, expected_version, build)

    def reschedule(
        self, scheduled_job_id: str, *, schedule: Schedule, due_at: datetime, expected_version: int, via: str
    ) -> ScheduledJob:
        valid_via(via)
        due = _utc(due_at)
        moment = _utc()

        def build(current: ScheduledJob) -> dict[str, Any]:
            if current.status not in (_S.ACTIVE, _S.PAUSED):
                raise InvalidTransitionError(f"A {current.status.value} Scheduled Job cannot be rescheduled.")
            if (schedule.kind is ScheduleKind.ONCE) != (current.schedule.kind is ScheduleKind.ONCE):
                raise ScheduledJobError("Changing between a one-time and a repeating schedule needs a fresh approval.")
            if current.expires_at is not None and due >= current.expires_at:
                raise ScheduledJobError("That time falls after the lifetime ends.")
            return {
                "schedule": json.dumps(schedule.to_dict(), separators=(",", ":")), "due_at": _iso(due),
                "events": _events_json(current.events, _event(moment, "rescheduled", f"next run {_iso(due)}", via)),
                "updated_at": _iso(moment),
            }

        return self._mutate(scheduled_job_id, expected_version, build)

    def expire_proposals(self, now: datetime, via: str = "system") -> list[ScheduledJob]:
        valid_via(via)
        moment = _utc(now)
        with self._lock:
            rows = self._conn.execute(
                f"SELECT {_JOB_COLUMNS} FROM scheduled_jobs WHERE status = 'proposed' AND proposal_expires_at <= ?", (_iso(moment),)
            ).fetchall()
        expired = []
        for row in rows:
            current = _job_from_row(row)
            try:
                expired.append(
                    self.transition(
                        current.id, _S.EXPIRED, _R.PROPOSAL_EXPIRED, expected_version=current.version, via=via,
                        detail="not approved in time", now=moment,
                    )
                )
            except StaleScheduledJobError:
                continue  # approved, declined or changed meanwhile: not ours to expire
        return expired

    # -- the ledger: acting on an occurrence -----------------------------------------------------------------------------------

    @staticmethod
    def _check_new_run(run: NewRun, insert_job) -> None:
        if not isinstance(run.occurrence, str) or not run.occurrence or len(run.occurrence) > 2 * MAX_ID_CHARS:
            raise ScheduledJobError("A run needs an occurrence key.")
        _utc(run.scheduled_for)
        creates = run.outcome in JOB_OUTCOMES
        if creates and (not isinstance(run.job_id, str) or not run.job_id or insert_job is None):
            raise ScheduledJobError("A run that creates a Job must name it and supply it to insert.")
        if not creates and (run.job_id is not None or insert_job is not None):
            raise ScheduledJobError("A skipped occurrence has no Job.")
        if run.outcome is RunOutcome.SKIPPED_BLOCKED and not run.reason:
            raise ScheduledJobError("A blocked occurrence has to say why.")
        if run.late != (run.outcome is RunOutcome.CREATED_LATE):
            raise ScheduledJobError("`late` belongs to the catch-up run (CREATED_LATE) and only to it.")
        if run.missed_count < 0:
            raise ScheduledJobError("The missed count cannot be negative.")

    def record_run(
        self,
        scheduled_job_id: str,
        run: NewRun,
        *,
        expected_version: int,
        expected_due_at: datetime | None,
        next_due_at: datetime | None,
        complete_reason: StatusReason | None = None,
        insert_job: Callable[[Any], None] | None = None,
        via: str = "scheduler",
        now: datetime | None = None,
    ) -> ScheduledJobRun:
        valid_via(via)
        self._check_new_run(run, insert_job)
        moment = _utc(now)
        scheduled_for = _utc(run.scheduled_for)
        creates = run.outcome in JOB_OUTCOMES
        with self._lock:
            try:
                current = self._guarded(scheduled_job_id, expected_version)
                if current.status is not _S.ACTIVE:
                    raise StaleScheduledJobError("This Scheduled Job is no longer active - nothing was changed.")
                changes: dict[str, Any] = {"updated_at": _iso(moment)}
                notes = [_event(moment, run.outcome.value, run.occurrence, via)]
                if run.kind is RunKind.SCHEDULED:
                    if expected_due_at is None or current.due_at != _utc(expected_due_at):
                        raise StaleScheduledJobError("This occurrence was already handled - nothing was changed.")
                    if next_due_at is not None:
                        following = _utc(next_due_at)
                        if following <= scheduled_for:
                            raise ScheduledJobError("The next run has to come after the occurrence just handled.")
                        if complete_reason is not None:
                            raise ScheduledJobError("A Scheduled Job with a next run is not complete.")
                        changes["due_at"] = _iso(following)
                    else:
                        if complete_reason is None:
                            raise ScheduledJobError("With no next run, say why the Scheduled Job is complete.")
                        check_transition(_S.ACTIVE, _S.COMPLETED, complete_reason)
                        changes["status"], changes["status_reason"] = _S.COMPLETED.value, complete_reason.value
                        notes.append(_event(moment, f"completed:{complete_reason.value}", "", via))
                elif next_due_at is not None or complete_reason is not None:
                    raise ScheduledJobError("A manual run does not move the schedule.")
                if creates:
                    changes["runs_total"] = current.runs_total + 1
                    changes["last_job_id"] = run.job_id
                    changes["last_run_at"] = _iso(moment)
                changes["events"] = _events_json(current.events, *notes)
                self._apply(current, changes)
                try:
                    cursor = self._conn.execute(
                        f"INSERT INTO scheduled_job_runs (schedule_id, occurrence, kind, scheduled_for, outcome, job_id, reason, "
                        f"missed_count, late, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            current.id, run.occurrence, run.kind.value, _iso(scheduled_for), run.outcome.value, run.job_id,
                            run.reason, run.missed_count, int(run.late), _iso(moment),
                        ),
                    )
                except sqlite3.IntegrityError as exc:
                    if "UNIQUE" in str(exc).upper():
                        raise DuplicateOccurrenceError("This occurrence was already acted on - nothing was written.") from None
                    raise ScheduledJobError(f"The run could not be recorded: {exc}") from None
                run_id = cursor.lastrowid
                if insert_job is not None:
                    insert_job(self._conn)
                self._conn.commit()
            except BaseException:
                self._conn.rollback()
                raise
            return self._select_run(run_id)

    def _select_run(self, run_id: int) -> ScheduledJobRun | None:
        row = self._conn.execute(f"SELECT {_RUN_COLUMNS} FROM scheduled_job_runs WHERE id = ?", (run_id,)).fetchone()
        return _run_from_row(row) if row is not None else None

    def get_run(self, schedule_id: str, occurrence: str) -> ScheduledJobRun | None:
        with self._lock:
            row = self._conn.execute(
                f"SELECT {_RUN_COLUMNS} FROM scheduled_job_runs WHERE schedule_id = ? AND occurrence = ?", (schedule_id, occurrence)
            ).fetchone()
        return _run_from_row(row) if row is not None else None

    def runs_for(self, schedule_id: str, limit: int = 50) -> list[ScheduledJobRun]:
        with self._lock:
            rows = self._conn.execute(
                f"SELECT {_RUN_COLUMNS} FROM scheduled_job_runs WHERE schedule_id = ? ORDER BY scheduled_for DESC, id DESC LIMIT ?",
                (schedule_id, max(1, int(limit))),
            ).fetchall()
        return [_run_from_row(r) for r in rows]

    def unsettled_runs(self, schedule_id: str | None = None) -> list[ScheduledJobRun]:
        query = f"SELECT {_RUN_COLUMNS} FROM scheduled_job_runs WHERE job_id IS NOT NULL AND result_state IS NULL"
        params: tuple = ()
        if schedule_id is not None:
            query += " AND schedule_id = ?"
            params = (schedule_id,)
        with self._lock:
            rows = self._conn.execute(query + " ORDER BY id", params).fetchall()
        return [_run_from_row(r) for r in rows]

    def count_runs_since(self, since: datetime) -> int:
        with self._lock:
            return self._conn.execute(
                "SELECT COUNT(*) FROM scheduled_job_runs WHERE job_id IS NOT NULL AND created_at >= ?", (_iso(since),)
            ).fetchone()[0]

    def settle_run(
        self, run_id: int, result_state: str, *, breaker_limit: int, via: str = "scheduler", now: datetime | None = None
    ) -> Settled:
        valid_via(via)
        if result_state not in RESULT_STATES:
            raise ScheduledJobError(f"Unknown run result '{result_state}'.")
        moment = _utc(now)
        with self._lock:
            try:
                run = self._select_run(run_id)
                if run is None:
                    raise ScheduledJobError("Unknown run.")
                if run.job_id is None:
                    raise ScheduledJobError("A skipped occurrence has no result to settle.")
                current = self._select(run.schedule_id)
                if current is None:
                    raise ScheduledJobError("Unknown Scheduled Job.")
                cursor = self._conn.execute(
                    "UPDATE scheduled_job_runs SET result_state = ?, settled_at = ? WHERE id = ? AND result_state IS NULL",
                    (result_state, _iso(moment), run_id),
                )
                if cursor.rowcount != 1:  # already settled: nothing to do, nothing to repeat
                    self._conn.rollback()
                    return Settled(current, False, False)
                verdict = breaker_decision(current.consecutive_failures, result_state, breaker_limit)
                changes: dict[str, Any] = {
                    "consecutive_failures": verdict.consecutive_failures, "updated_at": _iso(moment),
                }
                notes = [_event(moment, f"run_{result_state}", run.occurrence, via)]
                tripped = verdict.trip and current.status is _S.ACTIVE
                if tripped:
                    changes["status"], changes["status_reason"] = _S.PAUSED.value, _R.BREAKER.value
                    notes.append(_event(moment, "paused:breaker", f"{verdict.consecutive_failures} failed runs in a row", via))
                changes["events"] = _events_json(current.events, *notes)
                updated = self._apply(current, changes)
                self._conn.commit()
                return Settled(updated, True, tripped)
            except BaseException:
                self._conn.rollback()
                raise

    def mark_reported(self, run_id: int, now: datetime | None = None) -> bool:
        with self._lock:
            try:
                cursor = self._conn.execute(
                    "UPDATE scheduled_job_runs SET reported_at = ? WHERE id = ? AND reported_at IS NULL", (_iso(_utc(now)), run_id)
                )
                self._conn.commit()
                return cursor.rowcount == 1
            except BaseException:
                self._conn.rollback()
                raise
