"""
Scheduled Jobs V1 / Slice 3 - the schema migration for the new settlement outcomes.

SQLite cannot alter a CHECK constraint, and `CREATE TABLE IF NOT EXISTS` leaves an existing table exactly as it was, so a database made
BEFORE Slice 3 would refuse to record `skipped_disabled` / `skipped_cap` / `skipped_expired`. The store therefore inspects the table it finds
(the repository's convention - there is no schema-version table) and, only when its outcome constraint is behind, rebuilds it in one
transaction without losing a row. These tests build a real OLD-schema database full of data and upgrade it.
"""

import json
import sqlite3
from datetime import datetime, time, timedelta, timezone

import pytest

import app.scheduled_jobs.sqlite_store as store_module
from app.orchestration.sqlite_store import SQLiteJobStore
from app.scheduled_jobs.base import NewRun, RunOutcome
from app.scheduled_jobs.sqlite_store import SQLiteScheduledJobStore
from app.scheduling.schedule import Schedule

UTC = timezone.utc
OLD_OUTCOMES = ("created", "created_late", "skipped_missed", "skipped_overlap", "skipped_blocked")  # the vocabulary before Slice 3
NEW_OUTCOMES = {"skipped_disabled", "skipped_cap", "skipped_expired"}
DUE = datetime(2026, 10, 5, 14, 0, tzinfo=UTC)
SCHEDULE = Schedule.daily("Asia/Dubai", time(18, 0))
TABLE = "scheduled_job_runs"
TEMP_NAME = "scheduled_job_runs__pre_migration"


def stamp(moment):
    return moment.strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")


def old_schema():
    """The schema exactly as Slice 1/2 created it: today's DDL with the outcome CHECK listing only the old five."""
    old = store_module._SCHEMA.replace(store_module._OUTCOME_VALUES, ", ".join(f"'{o}'" for o in OLD_OUTCOMES))
    assert old != store_module._SCHEMA and "skipped_cap" not in old
    return old


SJ_ID = "sj-old-1"


def build_old_database(path, *, runs=True):
    """A database as an earlier version left it: the old table, an active Scheduled Job, and a history of runs (some settled, one
    unsettled, skipped ones, a deleted row so the id counter is ahead of the largest id)."""
    connection = sqlite3.connect(path)
    connection.executescript(old_schema())
    connection.execute(
        "INSERT INTO scheduled_jobs (id, status, title, intent_text, objective, schedule, profile, due_at, approved_at, "
        "approval_interaction_id, expires_at, runs_total, consecutive_failures, events, version, created_at, updated_at) "
        "VALUES (?, 'active', 'Old one', 'check things', '{}', ?, 'unattended_v1', ?, ?, 'int-1', ?, 2, 0, '[]', 3, ?, ?)",
        (SJ_ID, json.dumps(SCHEDULE.to_dict()), stamp(DUE), stamp(DUE - timedelta(days=3)), stamp(DUE + timedelta(days=80)),
         stamp(DUE - timedelta(days=3)), stamp(DUE - timedelta(days=1))),
    )
    if runs:
        rows = [
            # id, occurrence, outcome, job_id, reason, missed, late, result_state, settled_at
            (1, "occ-1", "created", "job-1", None, 0, 0, "completed", stamp(DUE - timedelta(days=2, hours=-1))),
            (2, "occ-2", "created_late", "job-2", None, 3, 1, None, None),
            (3, "occ-3", "skipped_blocked", None, "capability gone", 0, 0, None, None),
            (4, "occ-4", "skipped_overlap", None, None, 0, 0, None, None),
            (5, "occ-5", "skipped_missed", None, None, 2, 0, None, None),
        ]
        for id_, occurrence, outcome, job_id, reason, missed, late, result, settled in rows:
            connection.execute(
                "INSERT INTO scheduled_job_runs (id, schedule_id, occurrence, kind, scheduled_for, outcome, job_id, reason, missed_count, "
                "late, created_at, result_state, settled_at) VALUES (?, ?, ?, 'scheduled', ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (id_, SJ_ID, occurrence, stamp(DUE - timedelta(days=6 - id_)), outcome, job_id, reason, missed, late,
                 stamp(DUE - timedelta(days=6 - id_)), result, settled),
            )
        connection.execute("DELETE FROM scheduled_job_runs WHERE id = 5")  # the counter (5) is now ahead of the largest id (4)
    connection.commit()
    connection.close()


def snapshot(path):
    connection = sqlite3.connect(path)
    try:
        return connection.execute(f"SELECT * FROM {TABLE} ORDER BY id").fetchall()
    finally:
        connection.close()


def definition(path):
    connection = sqlite3.connect(path)
    try:
        return connection.execute("SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?", (TABLE,)).fetchone()[0]
    finally:
        connection.close()


def names(path, kind):
    connection = sqlite3.connect(path)
    try:
        return {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type = ?", (kind,))}
    finally:
        connection.close()


def record_skip(store, outcome, occurrence_at, reason="because"):
    current = store.get(SJ_ID)
    return store.record_run(
        SJ_ID, NewRun(occurrence=f"new@{occurrence_at.isoformat()}", scheduled_for=occurrence_at, outcome=outcome, reason=reason),
        expected_version=current.version, expected_due_at=current.due_at, next_due_at=occurrence_at + timedelta(days=1), via="scheduler",
    )


# ====================================== the old schema really refuses the new outcomes ===========================================


def test_the_old_schema_in_these_tests_is_really_the_old_one_and_refuses_the_new_outcomes(tmp_path):
    path = str(tmp_path / "old.db")
    build_old_database(path)
    connection = sqlite3.connect(path)
    for outcome in NEW_OUTCOMES:
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "INSERT INTO scheduled_job_runs (schedule_id, occurrence, kind, scheduled_for, outcome, created_at) "
                "VALUES (?, ?, 'scheduled', ?, ?, ?)", (SJ_ID, f"x-{outcome}", stamp(DUE), outcome, stamp(DUE)),
            )
    connection.close()


# ====================================== upgrading an old database ================================================================


def test_an_old_database_is_upgraded_in_place_and_every_existing_row_survives_unchanged(tmp_path):
    path = str(tmp_path / "old.db")
    build_old_database(path)
    before = snapshot(path)
    assert len(before) == 4 and "skipped_cap" not in definition(path)

    store = SQLiteScheduledJobStore.open(path)  # opening is what migrates

    assert snapshot(path) == before  # every row, every column, identical (ids included)
    sql = definition(path)
    assert all(f"'{o.value}'" in sql for o in RunOutcome) and NEW_OUTCOMES <= {o.value for o in RunOutcome}
    runs = store.runs_for(SJ_ID)  # and the typed store still reads the old rows
    assert {r.occurrence for r in runs} == {"occ-1", "occ-2", "occ-3", "occ-4"}
    assert store.get_run(SJ_ID, "occ-1").result_state == "completed" and store.get_run(SJ_ID, "occ-2").late
    assert [r.job_id for r in store.unsettled_runs()] == ["job-2"]
    assert store.get(SJ_ID).title == "Old one"  # the Scheduled Job table was not touched


def test_after_the_upgrade_the_new_settlement_outcomes_can_be_written(tmp_path):
    path = str(tmp_path / "old.db")
    build_old_database(path)
    store = SQLiteScheduledJobStore.open(path)

    moment = DUE
    for outcome in (RunOutcome.SKIPPED_DISABLED, RunOutcome.SKIPPED_CAP, RunOutcome.SKIPPED_EXPIRED):
        run = record_skip(store, outcome, moment)
        assert run.outcome is outcome and run.job_id is None
        moment += timedelta(days=1)
    stored = {r.outcome for r in store.runs_for(SJ_ID)}
    assert {RunOutcome.SKIPPED_DISABLED, RunOutcome.SKIPPED_CAP, RunOutcome.SKIPPED_EXPIRED} <= stored


def test_the_constraints_and_indexes_survive_the_rebuild(tmp_path):
    path = str(tmp_path / "old.db")
    build_old_database(path)
    SQLiteScheduledJobStore.open(path)

    assert "ix_scheduled_job_runs_open" in names(path, "index") and "ix_scheduled_jobs_due" in names(path, "index")
    connection = sqlite3.connect(path)
    columns = (
        "INSERT INTO scheduled_job_runs (schedule_id, occurrence, kind, scheduled_for, outcome, job_id, created_at) "
        "VALUES (?, ?, 'scheduled', ?, ?, ?, ?)"
    )
    with pytest.raises(sqlite3.IntegrityError):  # one occurrence of one Scheduled Job, once
        connection.execute(columns, (SJ_ID, "occ-1", stamp(DUE), "skipped_blocked", None, stamp(DUE)))
    with pytest.raises(sqlite3.IntegrityError):  # a Job belongs to at most one run
        connection.execute(columns, (SJ_ID, "fresh", stamp(DUE), "created", "job-1", stamp(DUE)))
    with pytest.raises(sqlite3.IntegrityError):  # a Job-making outcome must name its Job
        connection.execute(columns, (SJ_ID, "fresh-2", stamp(DUE), "created", None, stamp(DUE)))
    with pytest.raises(sqlite3.IntegrityError):  # and an unknown outcome is still refused
        connection.execute(columns, (SJ_ID, "fresh-3", stamp(DUE), "skipped_nonsense", None, stamp(DUE)))
    with pytest.raises(sqlite3.IntegrityError):  # a result only for a run that has a Job
        connection.execute(
            "INSERT INTO scheduled_job_runs (schedule_id, occurrence, kind, scheduled_for, outcome, created_at, result_state) "
            "VALUES (?, 'fresh-4', 'scheduled', ?, 'skipped_cap', ?, 'failed')", (SJ_ID, stamp(DUE), stamp(DUE)),
        )
    connection.close()


def test_ids_keep_counting_from_the_old_counter_never_reusing_a_deleted_one(tmp_path):
    path = str(tmp_path / "old.db")
    build_old_database(path)  # the largest id is 4, but id 5 was used and deleted
    store = SQLiteScheduledJobStore.open(path)
    run = record_skip(store, RunOutcome.SKIPPED_CAP, DUE)
    assert run.id == 6


def test_the_upgrade_is_idempotent_and_a_current_database_is_not_touched_at_all(tmp_path):
    path = str(tmp_path / "old.db")
    build_old_database(path)
    SQLiteScheduledJobStore.open(path)
    after_first = snapshot(path), definition(path)
    assert TEMP_NAME not in names(path, "table")  # nothing is left behind

    again = sqlite3.connect(path)
    again.row_factory = sqlite3.Row
    import threading

    SQLiteScheduledJobStore(again, threading.Lock())  # a second open of an already-current database
    assert again.total_changes == 0  # it changed nothing: the migration did not even run
    again.close()
    assert (snapshot(path), definition(path)) == after_first


def test_a_brand_new_database_needs_no_migration_and_has_the_current_schema(tmp_path):
    path = str(tmp_path / "new.db")
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    import threading

    SQLiteScheduledJobStore(connection, threading.Lock())
    assert connection.total_changes == 0
    sql = definition(path)
    assert all(f"'{o.value}'" in sql for o in RunOutcome)
    connection.close()


def test_an_old_database_with_no_runs_yet_is_upgraded_too(tmp_path):
    path = str(tmp_path / "old-empty.db")
    build_old_database(path, runs=False)
    store = SQLiteScheduledJobStore.open(path)
    assert snapshot(path) == [] and "skipped_cap" in definition(path)
    assert record_skip(store, RunOutcome.SKIPPED_DISABLED, DUE).id == 1


def test_the_upgrade_works_on_the_shared_connection_the_application_uses(tmp_path):
    path = str(tmp_path / "old.db")
    build_old_database(path)
    before = snapshot(path)
    jobs = SQLiteJobStore(path)  # the Job store's own connection and lock, as main.py builds it
    store = SQLiteScheduledJobStore.for_job_store(jobs)
    assert snapshot(path) == before and "skipped_cap" in definition(path)
    assert record_skip(store, RunOutcome.SKIPPED_CAP, DUE).outcome is RunOutcome.SKIPPED_CAP
    assert jobs.list() == []  # the Job table is its own and was not disturbed


# ====================================== a failed upgrade loses nothing ===========================================================


def test_a_migration_that_cannot_complete_rolls_back_and_leaves_the_old_table_and_its_data_untouched(tmp_path):
    path = str(tmp_path / "old.db")
    build_old_database(path)
    blocker = sqlite3.connect(path)
    blocker.execute(f"CREATE TABLE {TEMP_NAME} (x INTEGER)")  # the name the rebuild moves the old table to is taken: it must fail
    blocker.commit()
    blocker.close()
    before, old_sql = snapshot(path), definition(path)

    with pytest.raises(sqlite3.OperationalError):
        SQLiteScheduledJobStore.open(path)

    assert snapshot(path) == before  # every row still there
    assert definition(path) == old_sql  # the old table is still the old table, not half-rebuilt
    assert "ix_scheduled_job_runs_open" in names(path, "index")  # even its index (dropped inside the transaction) was rolled back


def test_the_migration_is_one_transaction_with_nothing_dropped_before_everything_is_copied():
    source = __import__("inspect").getsource(SQLiteScheduledJobStore._migrate_outcome_constraint)
    assert source.index("BEGIN IMMEDIATE") < source.index("RENAME TO") < source.index("INSERT INTO scheduled_job_runs") < source.index("DROP TABLE")
    assert "rollback()" in source and "except BaseException" in source


def test_the_schema_is_inspected_only_after_the_write_lock_is_held():
    source = __import__("inspect").getsource(SQLiteScheduledJobStore._migrate_outcome_constraint)
    code = source.split('"""')[2]  # past the docstring
    assert code.index("BEGIN IMMEDIATE") < code.index("FROM sqlite_master") < code.index("FROM sqlite_sequence")


# ====================================== two connections upgrading the same old database ==========================================


class _Proxy:
    """A connection that runs `before_lock` once, just before it asks for the write lock - the instant at which another connection can
    slip in and finish the upgrade. Everything else is the real connection."""

    def __init__(self, connection, before_lock):
        self._real, self._before_lock = connection, before_lock

    def execute(self, sql, *args):
        if sql.strip().upper() == "BEGIN IMMEDIATE" and self._before_lock is not None:
            hook, self._before_lock = self._before_lock, None
            hook()
        return self._real.execute(sql, *args)

    def __getattr__(self, name):
        return getattr(self._real, name)


def traced_connection(path, rebuilds):
    """A connection of its own that appends to `rebuilds` each time it renames the table aside - the first step of an actual rebuild."""
    connection = sqlite3.connect(path, check_same_thread=False)
    connection.row_factory = sqlite3.Row
    connection.set_trace_callback(lambda statement: rebuilds.append(statement) if "RENAME TO" in statement.upper() else None)
    return connection


def test_a_connection_that_lost_the_race_sees_the_finished_upgrade_and_does_not_rebuild_again(tmp_path):
    import threading

    path = str(tmp_path / "old.db")
    build_old_database(path)
    before = snapshot(path)

    winner_rebuilds, loser_rebuilds = [], []
    winner = traced_connection(path, winner_rebuilds)
    # The loser is about to ask for the lock; at that instant the winner runs its whole upgrade. A check made BEFORE the lock would have
    # already seen the old table and would now rebuild the new one a second time.
    loser = _Proxy(traced_connection(path, loser_rebuilds), lambda: SQLiteScheduledJobStore(winner, threading.Lock()))
    store = SQLiteScheduledJobStore(loser, threading.Lock())

    assert len(winner_rebuilds) == 1 and loser_rebuilds == []  # exactly one actual migration
    assert snapshot(path) == before  # nothing lost, ids intact
    assert TEMP_NAME not in names(path, "table")
    assert record_skip(store, RunOutcome.SKIPPED_CAP, DUE).id == 6  # and the loser's store works on the upgraded table
    winner.close()


def test_independent_openers_racing_on_one_old_database_migrate_it_exactly_once_and_lose_nothing(tmp_path):
    import threading

    path = str(tmp_path / "old.db")
    build_old_database(path)
    before = snapshot(path)

    rebuilds, errors, stores = [], [], []
    start = threading.Barrier(6)

    def open_it():
        try:
            connection = traced_connection(path, rebuilds)
            start.wait()
            stores.append(SQLiteScheduledJobStore(connection, threading.Lock()))
        except BaseException as exc:  # reported on the main thread, below
            errors.append(exc)

    threads = [threading.Thread(target=open_it) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)

    assert errors == [] and len(stores) == 6
    assert len(rebuilds) == 1  # one actual migration among six callers
    assert snapshot(path) == before
    assert TEMP_NAME not in names(path, "table") and "ix_scheduled_job_runs_open" in names(path, "index")
    assert all(f"'{o.value}'" in definition(path) for o in RunOutcome)
    assert record_skip(stores[0], RunOutcome.SKIPPED_EXPIRED, DUE).id == 6

