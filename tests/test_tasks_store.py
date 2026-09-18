"""
Unit tests for SQLiteTaskStore (app/tasks/sqlite_store.py): persistence,
priority/due_date handling, lifecycle transitions, and restart behavior.
Uses real SQLite (":memory:" or a temp file), never mocked - same
convention as test_reminders_store.py.
"""

from app.tasks.base import TaskPriority, TaskStatus
from app.tasks.sqlite_store import SQLiteTaskStore


def make_store() -> SQLiteTaskStore:
    return SQLiteTaskStore(":memory:")


# --- create/get/list -------------------------------------------------------


def test_create_returns_a_todo_task_with_default_priority():
    store = make_store()

    task = store.create(text="buy milk")

    assert task.id
    assert task.text == "buy milk"
    assert task.status is TaskStatus.TODO
    assert task.priority is TaskPriority.MEDIUM
    assert task.due_date is None


def test_create_persists_priority_and_due_date():
    store = make_store()

    task = store.create(text="ship it", priority=TaskPriority.HIGH, due_date="2026-01-01")

    fetched = store.get(task.id)
    assert fetched.priority is TaskPriority.HIGH
    assert fetched.due_date == "2026-01-01"


def test_get_unknown_task_returns_none():
    store = make_store()

    assert store.get("does-not-exist") is None


def test_list_returns_all_tasks_most_recent_first():
    store = make_store()
    first = store.create(text="first")
    second = store.create(text="second")

    tasks = store.list()

    assert [t.id for t in tasks] == [second.id, first.id]


def test_list_filters_by_status():
    store = make_store()
    todo = store.create(text="todo one")
    done = store.create(text="done one")
    store.complete(done.id)

    todo_only = store.list(status=TaskStatus.TODO)

    assert [t.id for t in todo_only] == [todo.id]


# --- complete/cancel ---------------------------------------------------------


def test_complete_marks_task_done_and_returns_true():
    store = make_store()
    task = store.create(text="x")

    assert store.complete(task.id) is True
    assert store.get(task.id).status is TaskStatus.DONE


def test_complete_unknown_task_returns_false():
    store = make_store()

    assert store.complete("does-not-exist") is False


def test_complete_already_completed_task_returns_false():
    store = make_store()
    task = store.create(text="x")
    store.complete(task.id)

    assert store.complete(task.id) is False


def test_cancel_marks_task_cancelled_and_returns_true():
    store = make_store()
    task = store.create(text="x")

    assert store.cancel(task.id) is True
    assert store.get(task.id).status is TaskStatus.CANCELLED


def test_cancel_already_cancelled_task_returns_false():
    store = make_store()
    task = store.create(text="x")
    store.cancel(task.id)

    assert store.cancel(task.id) is False


def test_cancel_a_completed_task_returns_false():
    store = make_store()
    task = store.create(text="x")
    store.complete(task.id)

    assert store.cancel(task.id) is False


# --- persistence across store instances (restart) ------------------------


def test_tasks_persist_across_store_instances(tmp_path):
    db_path = str(tmp_path / "tasks.db")
    first_store = SQLiteTaskStore(db_path)
    task = first_store.create(text="survive restart", priority=TaskPriority.LOW)

    second_store = SQLiteTaskStore(db_path)

    fetched = second_store.get(task.id)
    assert fetched is not None
    assert fetched.text == "survive restart"
    assert fetched.priority is TaskPriority.LOW


def test_a_restarted_store_reflects_completion(tmp_path):
    db_path = str(tmp_path / "tasks.db")
    first_store = SQLiteTaskStore(db_path)
    task = first_store.create(text="finish this")
    first_store.complete(task.id)

    second_store = SQLiteTaskStore(db_path)

    assert second_store.get(task.id).status is TaskStatus.DONE
