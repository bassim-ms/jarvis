"""
SQLite-backed InteractionStore.

Not a parallel persistence system: it lives in the SAME database file as the
Jobs and shares the SQLiteJobStore's connection and lock (`job_store.
connection` / `job_store.lock`). That is deliberate - a later stage, where
the engine applies a resolved interaction to its Job's steps, needs one
transaction covering both tables, which two connections could never give.

ATOMICITY BOUNDARY (this stage): every transition here is ONE guarded UPDATE
that sets the new status, the response, `resolved_at`, the audit entry and
the bumped version together - so "RESOLVED but the response is gone" cannot
happen, and a stale or racing writer cannot overwrite a newer state (the
UPDATE only matches `id`, `version` AND `status = 'pending'`). A transition
touches ONLY the interaction row: it never writes a Job or a plan step, so
it cannot make a gated step runnable - the Job stays in its waiting state
and the engine never advances it. Coupling a transition to the Job's
step/status changes atomically is the next stage's job (it will do both
writes in one transaction on this shared connection); nothing here pretends
to do it.

The table is created with `CREATE TABLE IF NOT EXISTS`, so an existing
database (jobs only) simply gains it - no ALTER needed.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Callable
from datetime import datetime
from typing import Any

from app.orchestration.base import Job, JobStatus
from app.orchestration.interactions import (
    ALLOWED_TRANSITIONS,
    InteractionError,
    InteractionExpiredError,
    InteractionNotFoundError,
    InteractionRequest,
    InteractionStatus,
    InteractionStore,
    InteractionType,
    InvalidInteractionTransitionError,
    StaleInteractionError,
    expires_at_for,
    generate_interaction_code,
    is_expired,
    utc_now,
    validate_new_interaction,
    validate_transition_input,
)
from app.orchestration.base import StaleJobError  # noqa: F401 - documented failure mode
from app.orchestration.sqlite_store import SQLiteJobStore

#: How many fresh codes to try before giving up on a collision streak. With
#: ~887M possible codes a single retry is already vanishingly unlikely.
_CODE_ATTEMPTS = 10


class SQLiteInteractionStore(InteractionStore):
    """InteractionStore over the Job store's own connection - see the
    module docstring."""

    def __init__(self, job_store: SQLiteJobStore):
        self._job_store = job_store
        self._conn = job_store.connection
        self._lock = job_store.lock
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS interactions (
                    id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL,
                    type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    prompt TEXT NOT NULL,
                    code TEXT NOT NULL,
                    step_ids TEXT NOT NULL DEFAULT '[]',
                    detail TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    expires_at TEXT,
                    resolved_at TEXT,
                    response TEXT,
                    resolution TEXT,
                    version INTEGER NOT NULL DEFAULT 0,
                    presented_at TEXT,
                    presented_conversation_id TEXT,
                    FOREIGN KEY (job_id) REFERENCES jobs (id)
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_interactions_code
                    ON interactions (code);
                CREATE INDEX IF NOT EXISTS idx_interactions_job
                    ON interactions (job_id, status);
                """
            )
            self._conn.commit()
            # An interactions table from an earlier stage lacks the
            # presentation columns: add them in place.
            self._migrate_add_column("presented_at", "TEXT")
            self._migrate_add_column("presented_conversation_id", "TEXT")

    def _migrate_add_column(self, name: str, ddl_type: str) -> None:
        """Same minimal form as SQLiteJobStore._migrate_add_column: try to add
        the column, accept "duplicate column name", surface anything else."""
        try:
            self._conn.execute(f"ALTER TABLE interactions ADD COLUMN {name} {ddl_type}")
            self._conn.commit()
        except sqlite3.OperationalError as exc:
            if "duplicate column name" not in str(exc).lower():
                raise

    @staticmethod
    def _row_to_request(row: sqlite3.Row) -> InteractionRequest:
        return InteractionRequest(
            id=row["id"],
            job_id=row["job_id"],
            type=InteractionType(row["type"]),
            status=InteractionStatus(row["status"]),
            prompt=row["prompt"],
            code=row["code"],
            created_at=row["created_at"],
            expires_at=row["expires_at"],
            resolved_at=row["resolved_at"],
            response=row["response"],
            step_ids=json.loads(row["step_ids"]),
            detail=json.loads(row["detail"]),
            resolution=json.loads(row["resolution"]) if row["resolution"] else None,
            version=row["version"],
            presented_at=row["presented_at"],
            presented_conversation_id=row["presented_conversation_id"],
        )

    def create(
        self,
        job_id: str,
        type: InteractionType,
        prompt: str,
        *,
        step_ids: list[str] | tuple[str, ...] = (),
        detail: dict[str, Any] | None = None,
        ttl_seconds: float | None = None,
        now: datetime | None = None,
    ) -> InteractionRequest:
        detail = validate_new_interaction(type, prompt, step_ids, detail, ttl_seconds)
        created = now or utc_now()
        interaction_id = uuid.uuid4().hex

        with self._lock:
            if self._conn.execute("SELECT 1 FROM jobs WHERE id = ?", (job_id,)).fetchone() is None:
                raise InteractionNotFoundError(f"Job '{job_id}' does not exist.")

            for _ in range(_CODE_ATTEMPTS):
                code = generate_interaction_code()
                try:
                    self._conn.execute(
                        """
                        INSERT INTO interactions (
                            id, job_id, type, status, prompt, code, step_ids,
                            detail, created_at, expires_at, version
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                        """,
                        (
                            interaction_id,
                            job_id,
                            type.value,
                            InteractionStatus.PENDING.value,
                            prompt,
                            code,
                            json.dumps(list(step_ids)),
                            json.dumps(detail),
                            created.isoformat(),
                            expires_at_for(created, ttl_seconds),
                        ),
                    )
                except sqlite3.IntegrityError:
                    self._conn.rollback()
                    if self._conn.execute(
                        "SELECT 1 FROM interactions WHERE id = ?", (interaction_id,)
                    ).fetchone():
                        raise  # not a code collision - the id itself clashed
                    continue  # code collision: draw another
                self._conn.commit()
                break
            else:
                raise InteractionError("Could not allocate a unique interaction code.")
            row = self._conn.execute(
                "SELECT * FROM interactions WHERE id = ?", (interaction_id,)
            ).fetchone()
        return self._row_to_request(row)

    def get(self, interaction_id: str) -> InteractionRequest | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM interactions WHERE id = ?", (interaction_id,)
            ).fetchone()
        return self._row_to_request(row) if row else None

    def list_for_job(self, job_id: str) -> list[InteractionRequest]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM interactions WHERE job_id = ? ORDER BY created_at, rowid",
                (job_id,),
            ).fetchall()
        return [self._row_to_request(row) for row in rows]

    def list_pending_for_job(self, job_id: str) -> list[InteractionRequest]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM interactions WHERE job_id = ? AND status = ? "
                "ORDER BY created_at, rowid",
                (job_id, InteractionStatus.PENDING.value),
            ).fetchall()
        return [self._row_to_request(row) for row in rows]

    # -- transitions ----------------------------------------------------------------------

    def get_by_code(self, code: str) -> InteractionRequest | None:
        if not isinstance(code, str):
            return None
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM interactions WHERE code = ?", (code,)
            ).fetchone()
        return self._row_to_request(row) if row else None

    def list_pending(self) -> list[InteractionRequest]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM interactions WHERE status = ? ORDER BY created_at, rowid",
                (InteractionStatus.PENDING.value,),
            ).fetchall()
        return [self._row_to_request(row) for row in rows]

    def _list_for_job_locked(self, job_id: str) -> list[InteractionRequest]:
        rows = self._conn.execute(
            "SELECT * FROM interactions WHERE job_id = ? ORDER BY created_at, rowid",
            (job_id,),
        ).fetchall()
        return [self._row_to_request(row) for row in rows]

    def mark_presented(
        self,
        interaction_id: str,
        conversation_id: str,
        *,
        now: datetime | None = None,
    ) -> InteractionRequest:
        if not isinstance(conversation_id, str) or not conversation_id or len(conversation_id) > 200:
            raise InvalidInteractionTransitionError(
                "conversation_id must be a non-empty string of at most 200 characters."
            )
        moment = now or utc_now()
        with self._lock:
            current = self._read_locked(interaction_id)
            if current.status is not InteractionStatus.PENDING:
                raise InvalidInteractionTransitionError(
                    f"Interaction '{interaction_id}' is {current.status.value}; it can no "
                    "longer be presented."
                )
            # Metadata only: status and version are untouched, and the write
            # only matches a row that is still pending.
            cursor = self._conn.execute(
                "UPDATE interactions SET presented_at = ?, presented_conversation_id = ? "
                "WHERE id = ? AND status = ?",
                (
                    moment.isoformat(),
                    conversation_id,
                    interaction_id,
                    InteractionStatus.PENDING.value,
                ),
            )
            if cursor.rowcount == 0:
                self._conn.rollback()
                raise InvalidInteractionTransitionError(
                    f"Interaction '{interaction_id}' is no longer pending."
                )
            self._conn.commit()
            return self._read_locked(interaction_id)

    # -- the guards and the write, shared by the plain and the coupled transition ------------

    def _guard_transition(
        self,
        current: InteractionRequest,
        target: InteractionStatus,
        *,
        expected_version: int,
        response: str | None,
        reason: str | None,
        via: str | None,
        moment: datetime,
    ) -> dict[str, Any]:
        """Every rule a transition must satisfy, in a fixed order; returns
        the audit entry to store. Raises before anything is written."""
        # 1. Freshness first: a caller working from an old view is told so,
        #    whatever state the row is in now.
        if current.version != expected_version:
            raise StaleInteractionError(
                f"Interaction '{current.id}' changed since it was read (read version "
                f"{expected_version}, current version {current.version}); nothing was changed."
            )
        # 2. The transition table: only PENDING may move, and only to a terminal state.
        if target not in ALLOWED_TRANSITIONS[current.status]:
            raise InvalidInteractionTransitionError(
                f"Interaction '{current.id}' is {current.status.value}; it cannot "
                f"become {target.value}."
            )
        # 3. Payload and expiry rules.
        validate_transition_input(current.type, target, response, reason, via)
        if target in (InteractionStatus.RESOLVED, InteractionStatus.DECLINED) and is_expired(
            current, moment
        ):
            raise InteractionExpiredError(
                f"Interaction '{current.id}' expired at {current.expires_at}; it can "
                "no longer be answered."
            )
        if target is InteractionStatus.EXPIRED and not is_expired(current, moment):
            raise InvalidInteractionTransitionError(
                f"Interaction '{current.id}' is not due to expire "
                f"(expires_at: {current.expires_at})."
            )
        return {
            "previous_status": current.status.value,
            "status": target.value,
            "previous_version": current.version,
            "at": moment.isoformat(),
            "via": via,
            "reason": reason,
        }

    def _write_transition_locked(
        self,
        current: InteractionRequest,
        target: InteractionStatus,
        *,
        response: str | None,
        audit: dict[str, Any],
        moment: datetime,
    ) -> None:
        """The guarded interaction UPDATE (`id`, `version` AND `status =
        'pending'`), WITHOUT committing. If it matches no row the whole open
        transaction is rolled back and StaleInteractionError raised - so a
        Job write made earlier in the same transaction is undone with it.
        The caller must hold the lock and commit on success."""
        cursor = self._conn.execute(
            """
            UPDATE interactions SET
                status = ?, response = ?, resolved_at = ?, resolution = ?,
                version = ?
            WHERE id = ? AND version = ? AND status = ?
            """,
            (
                target.value,
                response,
                moment.isoformat(),
                json.dumps(audit),
                current.version + 1,
                current.id,
                current.version,
                InteractionStatus.PENDING.value,
            ),
        )
        if cursor.rowcount == 0:
            # Another connection got in between the read and this write: the
            # guard refused it. Nothing is kept from this transaction.
            self._conn.rollback()
            raise StaleInteractionError(
                f"Interaction '{current.id}' changed while it was being updated; "
                "nothing was changed."
            )

    @staticmethod
    def _check_expected_version(expected_version: Any, name: str = "expected_version") -> None:
        if isinstance(expected_version, bool) or not isinstance(expected_version, int) or expected_version < 0:
            raise InvalidInteractionTransitionError(f"{name} must be a non-negative integer.")

    def _read_locked(self, interaction_id: str) -> InteractionRequest:
        row = self._conn.execute(
            "SELECT * FROM interactions WHERE id = ?", (interaction_id,)
        ).fetchone()
        if row is None:
            raise InteractionNotFoundError(f"Interaction '{interaction_id}' does not exist.")
        return self._row_to_request(row)

    # -- transitions ----------------------------------------------------------------------

    def _transition(
        self,
        interaction_id: str,
        target: InteractionStatus,
        *,
        expected_version: int,
        response: str | None,
        reason: str | None,
        via: str | None,
        now: datetime | None,
    ) -> InteractionRequest:
        """The PLAIN transition: changes only the interaction row. Low-level -
        it does not touch the interaction's Job, so on its own it can leave a
        Job waiting for an interaction that is already answered. The
        production path is `transition_with_job` (via JobEngine)."""
        self._check_expected_version(expected_version)
        moment = now or utc_now()

        with self._lock:
            current = self._read_locked(interaction_id)
            audit = self._guard_transition(
                current,
                target,
                expected_version=expected_version,
                response=response,
                reason=reason,
                via=via,
                moment=moment,
            )
            self._write_transition_locked(
                current, target, response=response, audit=audit, moment=moment
            )
            self._conn.commit()
            return self._read_locked(interaction_id)

    def transition_with_job(
        self,
        interaction_id: str,
        target: InteractionStatus,
        *,
        expected_version: int,
        expected_job_version: int,
        mutate_job: Callable[[Job, InteractionRequest], None],
        check: Callable[[Job, InteractionRequest, list[InteractionRequest]], None] | None = None,
        response: str | None = None,
        reason: str | None = None,
        via: str | None = None,
        now: datetime | None = None,
    ) -> tuple[InteractionRequest, Job]:
        """THE ATOMIC BOUNDARY (see the module docstring). One lock
        acquisition, one transaction on the shared connection, one commit:

            read interaction + Job  ->  guards  ->  mutate the Job copy
              ->  guarded Job UPDATE  ->  guarded interaction UPDATE  ->  COMMIT

        Any failure before the COMMIT - a guard, `mutate_job`, a database
        error, even an interrupted process - leaves neither row changed
        (SQLite discards an uncommitted transaction; on an exception we roll
        back explicitly). After the COMMIT both rows reflect the transition.
        The lock is a plain `threading.Lock`, so nothing in here re-enters
        it: the Job is read and written through the job store's lock-free
        helpers on the shared connection."""
        self._check_expected_version(expected_version)
        self._check_expected_version(expected_job_version, "expected_job_version")
        moment = now or utc_now()

        with self._lock:
            try:
                current = self._read_locked(interaction_id)
                audit = self._guard_transition(
                    current,
                    target,
                    expected_version=expected_version,
                    response=response,
                    reason=reason,
                    via=via,
                    moment=moment,
                )
                job = SQLiteJobStore.load_job_locked(self._conn, current.job_id)
                if job is None:
                    raise InteractionNotFoundError(
                        f"Job '{current.job_id}' of interaction '{interaction_id}' does not exist."
                    )
                if job.version != expected_job_version:
                    raise StaleJobError(
                        f"Job '{job.id}' changed since it was read (read version "
                        f"{expected_job_version}, current version {job.version}); "
                        "nothing was changed."
                    )

                if check is not None:
                    check(job, current, self._list_for_job_locked(job.id))  # may refuse
                mutate_job(job, current)  # edits only the in-memory copy

                SQLiteJobStore.write_job_locked(self._conn, job)  # guarded; rolls back on refusal
                self._write_transition_locked(
                    current, target, response=response, audit=audit, moment=moment
                )  # guarded; rolls back BOTH writes on refusal
                self._conn.commit()  # the single commit
            except BaseException:
                self._conn.rollback()  # nothing from this attempt survives
                raise
            job.version += 1
            return self._read_locked(interaction_id), job

    # -- creating an interaction together with the Job it makes wait ----------------------------

    def _allocate_code_locked(self) -> str:
        """A fresh unused code. Checked with a SELECT (the caller holds the
        lock, so nothing can take it in between) instead of relying on a
        unique-index failure: that failure would force a rollback, which here
        would also undo the Job write in the same transaction."""
        for _ in range(_CODE_ATTEMPTS):
            code = generate_interaction_code()
            if self._conn.execute("SELECT 1 FROM interactions WHERE code = ?", (code,)).fetchone() is None:
                return code
        raise InteractionError("Could not allocate a unique interaction code.")

    def _insert_input_locked(
        self,
        job_id: str,
        prompt: str,
        detail: dict[str, Any],
        ttl_seconds: float | None,
        moment: datetime,
    ) -> str:
        """INSERT one PENDING INPUT interaction WITHOUT committing; returns its
        id. The caller holds the lock and commits (or rolls back)."""
        interaction_id = uuid.uuid4().hex
        self._conn.execute(
            """
            INSERT INTO interactions (
                id, job_id, type, status, prompt, code, step_ids,
                detail, created_at, expires_at, version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
            """,
            (
                interaction_id,
                job_id,
                InteractionType.INPUT.value,
                InteractionStatus.PENDING.value,
                prompt,
                self._allocate_code_locked(),
                json.dumps([]),
                json.dumps(detail),
                moment.isoformat(),
                expires_at_for(moment, ttl_seconds),
            ),
        )
        return interaction_id

    def create_job_awaiting_input(
        self,
        *,
        goal: str,
        context: dict[str, Any],
        conversation_id: str | None,
        prompt: str,
        detail: dict[str, Any] | None = None,
        ttl_seconds: float | None = None,
        now: datetime | None = None,
    ) -> tuple[Job, InteractionRequest]:
        clean_detail = validate_new_interaction(InteractionType.INPUT, prompt, (), detail, ttl_seconds)
        moment = now or utc_now()
        job = SQLiteJobStore.build_job(
            goal, context=context, conversation_id=conversation_id, status=JobStatus.AWAITING_INPUT
        )
        with self._lock:
            try:
                SQLiteJobStore.insert_job_locked(self._conn, job)
                interaction_id = self._insert_input_locked(job.id, prompt, clean_detail, ttl_seconds, moment)
                self._conn.commit()  # the single commit: the Job and its question, together
            except BaseException:
                self._conn.rollback()
                raise
            return job, self._read_locked(interaction_id)

    def create_job_awaiting_confirmation(
        self,
        job: Job,
        *,
        prompt: str,
        step_ids: list[str],
        detail: dict[str, Any] | None = None,
        ttl_seconds: float | None = None,
        now: datetime | None = None,
    ) -> tuple[Job, InteractionRequest]:
        """Insert a Job that is ALREADY waiting on a confirmation - built by the caller
        (`SQLiteJobStore.build_job(status=AWAITING_CONFIRMATION)` with its plan and
        `pending_step_ids` set) - together with its ONE pending CONFIRMATION
        interaction, in ONE transaction: both exist or neither does, and the engine
        can never see the Job without the question that gates its step. The
        confirmation-side twin of `create_job_awaiting_input`; used by trusted
        code (the memory write policy) that builds a one-step plan itself, so no
        planner is ever involved. Returns (Job, interaction)."""
        if (
            job.status is not JobStatus.AWAITING_CONFIRMATION
            or job.plan is None
            or not step_ids
            or any(sid not in job.pending_step_ids or job.plan.get(sid) is None for sid in step_ids)
        ):
            raise ValueError("The Job must be awaiting confirmation, with a plan naming every gated step.")
        clean_detail = validate_new_interaction(InteractionType.CONFIRMATION, prompt, step_ids, detail, ttl_seconds)
        moment = now or utc_now()
        with self._lock:
            try:
                SQLiteJobStore.insert_job_locked(self._conn, job)
                interaction_id = uuid.uuid4().hex
                self._conn.execute(
                    """
                    INSERT INTO interactions (
                        id, job_id, type, status, prompt, code, step_ids,
                        detail, created_at, expires_at, version
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                    """,
                    (
                        interaction_id,
                        job.id,
                        InteractionType.CONFIRMATION.value,
                        InteractionStatus.PENDING.value,
                        prompt,
                        self._allocate_code_locked(),
                        json.dumps(list(step_ids)),
                        json.dumps(clean_detail),
                        moment.isoformat(),
                        expires_at_for(moment, ttl_seconds),
                    ),
                )
                self._conn.commit()  # the single commit: the Job and its question, together
            except BaseException:
                self._conn.rollback()
                raise
            return job, self._read_locked(interaction_id)

    def park_job_for_input(
        self,
        job_id: str,
        *,
        expected_job_version: int,
        mutate_job: Callable[[Job], None],
        prompt: str,
        detail: dict[str, Any] | None = None,
        ttl_seconds: float | None = None,
        check: Callable[[Job, list[InteractionRequest]], None] | None = None,
        now: datetime | None = None,
    ) -> tuple[Job, InteractionRequest]:
        self._check_expected_version(expected_job_version, "expected_job_version")
        clean_detail = validate_new_interaction(InteractionType.INPUT, prompt, (), detail, ttl_seconds)
        moment = now or utc_now()
        with self._lock:
            try:
                job = SQLiteJobStore.load_job_locked(self._conn, job_id)
                if job is None:
                    raise InteractionNotFoundError(f"Job '{job_id}' does not exist.")
                if job.version != expected_job_version:
                    raise StaleJobError(
                        f"Job '{job.id}' changed since it was read (read version "
                        f"{expected_job_version}, current version {job.version}); "
                        "nothing was changed."
                    )
                if check is not None:
                    check(job, self._list_for_job_locked(job.id))  # may refuse (a limit)
                mutate_job(job)  # edits only the in-memory copy
                SQLiteJobStore.write_job_locked(self._conn, job)  # guarded; rolls back on refusal
                interaction_id = self._insert_input_locked(job.id, prompt, clean_detail, ttl_seconds, moment)
                self._conn.commit()  # the single commit
            except BaseException:
                self._conn.rollback()
                raise
            job.version += 1
            return job, self._read_locked(interaction_id)

    def cancel_with_job_check(
        self,
        interaction_id: str,
        *,
        expected_version: int,
        check: Callable[[Job, InteractionRequest], None],
        reason: str | None = None,
        via: str | None = None,
        now: datetime | None = None,
    ) -> InteractionRequest:
        self._check_expected_version(expected_version)
        moment = now or utc_now()
        with self._lock:
            try:
                current = self._read_locked(interaction_id)
                audit = self._guard_transition(
                    current,
                    InteractionStatus.CANCELLED,
                    expected_version=expected_version,
                    response=None,
                    reason=reason,
                    via=via,
                    moment=moment,
                )
                job = SQLiteJobStore.load_job_locked(self._conn, current.job_id)
                if job is None:
                    raise InteractionNotFoundError(
                        f"Job '{current.job_id}' of interaction '{interaction_id}' does not exist."
                    )
                check(job, current)  # may refuse; the Job is only read
                self._write_transition_locked(
                    current, InteractionStatus.CANCELLED, response=None, audit=audit, moment=moment
                )
                self._conn.commit()
            except BaseException:
                self._conn.rollback()
                raise
            return self._read_locked(interaction_id)

    def cancel_job_with_interactions(
        self,
        job_id: str,
        *,
        expected_job_version: int,
        mutate_job: Callable[[Job], None],
        reason: str | None = None,
        via: str | None = None,
        now: datetime | None = None,
    ) -> tuple[Job, list[InteractionRequest]]:
        """The Job and all its pending interactions, cancelled in one
        transaction (same discipline as `transition_with_job`: one lock, one
        commit, every write version-guarded, everything rolled back on any
        failure)."""
        self._check_expected_version(expected_job_version, "expected_job_version")
        moment = now or utc_now()

        with self._lock:
            try:
                job = SQLiteJobStore.load_job_locked(self._conn, job_id)
                if job is None:
                    raise InteractionNotFoundError(f"Job '{job_id}' does not exist.")
                if job.version != expected_job_version:
                    raise StaleJobError(
                        f"Job '{job.id}' changed since it was read (read version "
                        f"{expected_job_version}, current version {job.version}); "
                        "nothing was changed."
                    )
                mutate_job(job)  # edits only the in-memory copy
                SQLiteJobStore.write_job_locked(self._conn, job)  # guarded; rolls back on refusal

                withdrawn: list[str] = []
                for current in self._list_for_job_locked(job_id):
                    if current.status is not InteractionStatus.PENDING:
                        continue
                    audit = self._guard_transition(
                        current,
                        InteractionStatus.CANCELLED,
                        expected_version=current.version,
                        response=None,
                        reason=reason,
                        via=via,
                        moment=moment,
                    )
                    self._write_transition_locked(
                        current, InteractionStatus.CANCELLED, response=None, audit=audit, moment=moment
                    )  # guarded; rolls back everything on refusal
                    withdrawn.append(current.id)
                self._conn.commit()  # the single commit
            except BaseException:
                self._conn.rollback()
                raise
            job.version += 1
            return job, [self._read_locked(i) for i in withdrawn]

    def resolve(
        self,
        interaction_id: str,
        *,
        expected_version: int,
        response: str | None = None,
        via: str | None = None,
        now: datetime | None = None,
    ) -> InteractionRequest:
        return self._transition(
            interaction_id,
            InteractionStatus.RESOLVED,
            expected_version=expected_version,
            response=response,
            reason=None,
            via=via,
            now=now,
        )

    def decline(
        self,
        interaction_id: str,
        *,
        expected_version: int,
        response: str | None = None,
        reason: str | None = None,
        via: str | None = None,
        now: datetime | None = None,
    ) -> InteractionRequest:
        return self._transition(
            interaction_id,
            InteractionStatus.DECLINED,
            expected_version=expected_version,
            response=response,
            reason=reason,
            via=via,
            now=now,
        )

    def expire(
        self,
        interaction_id: str,
        *,
        expected_version: int,
        now: datetime | None = None,
    ) -> InteractionRequest:
        return self._transition(
            interaction_id,
            InteractionStatus.EXPIRED,
            expected_version=expected_version,
            response=None,
            reason="expired",
            via="expiry",
            now=now,
        )

    def cancel(
        self,
        interaction_id: str,
        *,
        expected_version: int,
        reason: str | None = None,
        via: str | None = None,
        now: datetime | None = None,
    ) -> InteractionRequest:
        return self._transition(
            interaction_id,
            InteractionStatus.CANCELLED,
            expected_version=expected_version,
            response=None,
            reason=reason,
            via=via,
            now=now,
        )
