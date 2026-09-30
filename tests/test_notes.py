"""
Tests for Notes (app/notes/): the NoteStore, save_note and list_notes, and -
just as important - that notes stay separate from permanent memory/facts and
the World Model.
"""

import pathlib
import re

import pytest

from app.core import JarvisCore
from app.memory.sqlite_store import SQLiteMemoryStore
from app.notes.base import MAX_CONTENT_CHARS, MAX_TITLE_CHARS
from app.notes.sqlite_store import SQLiteNoteStore
from app.notes.tools import MAX_LIST_LIMIT, ListNotesTool, SaveNoteTool
from app.orchestration.base import JobStatus, RetryPolicy
from app.orchestration.engine import JobEngine
from app.orchestration.planner import Planner, PlanningError
from app.orchestration.sqlite_store import SQLiteJobStore
from app.orchestration.validator import PlanValidationError, validate_plan
from app.providers.base import AIProvider
from app.tools.base import PermissionLevel, ToolResultStatus
from app.tools.registry import ToolRegistry

# --- the store ------------------------------------------------------------------------------------


def test_create_and_get_round_trip_with_provenance():
    store = SQLiteNoteStore(":memory:")

    note = store.create("Acme", "founded 1999", source_job_id="job-1", source_step_id="s3")

    fetched = store.get(note.id)
    assert fetched == note
    assert (fetched.title, fetched.content) == ("Acme", "founded 1999")
    assert (fetched.source_job_id, fetched.source_step_id) == ("job-1", "s3")
    assert fetched.created_at


def test_a_note_saved_outside_a_job_has_no_source():
    store = SQLiteNoteStore(":memory:")

    note = store.create("t", "c")

    assert store.get(note.id).source_job_id is None
    assert store.get(note.id).source_step_id is None


def test_get_returns_none_for_an_unknown_id():
    assert SQLiteNoteStore(":memory:").get("nope") is None


def test_list_is_newest_first_and_bounded_by_limit():
    store = SQLiteNoteStore(":memory:")
    for i in range(5):
        store.create(f"n{i}", "c")

    assert [n.title for n in store.list()] == ["n4", "n3", "n2", "n1", "n0"]
    assert [n.title for n in store.list(limit=2)] == ["n4", "n3"]


def test_list_can_be_filtered_to_one_job():
    store = SQLiteNoteStore(":memory:")
    store.create("a", "c", source_job_id="job-1")
    store.create("b", "c", source_job_id="job-2")
    store.create("c", "c", source_job_id="job-1")
    store.create("d", "c")

    assert [n.title for n in store.list(job_id="job-1")] == ["c", "a"]
    assert store.list(job_id="job-9") == []


def test_notes_persist_across_a_new_store_instance(tmp_path):
    db = str(tmp_path / "notes.db")
    first = SQLiteNoteStore(db)
    note = first.create("Acme", "founded 1999", source_job_id="job-1", source_step_id="s3")
    del first

    reopened = SQLiteNoteStore(db)

    assert reopened.get(note.id) == note
    assert [n.id for n in reopened.list()] == [note.id]


# --- save_note ------------------------------------------------------------------------------------------


def test_save_note_is_safe_action_and_list_notes_is_read():
    store = SQLiteNoteStore(":memory:")

    assert SaveNoteTool(store).permission is PermissionLevel.SAFE_ACTION
    assert ListNotesTool(store).permission is PermissionLevel.READ


def test_save_note_stores_the_note_and_returns_it():
    store = SQLiteNoteStore(":memory:")
    registry = ToolRegistry()
    registry.register(SaveNoteTool(store))

    result = registry.execute("save_note", params={"title": "  Acme  ", "content": "founded 1999"})

    assert result.status is ToolResultStatus.SUCCESS
    saved = store.list()[0]
    assert (saved.title, saved.content) == ("Acme", "founded 1999")  # title stripped, content verbatim
    assert result.data["id"] == saved.id
    assert saved.id in result.message


@pytest.mark.parametrize(
    "params",
    [
        {"title": "", "content": "c"},
        {"title": "   ", "content": "c"},
        {"title": "t" * (MAX_TITLE_CHARS + 1), "content": "c"},
        {"title": "t", "content": ""},
        {"title": "t", "content": "   "},
        {"title": "t", "content": "c" * (MAX_CONTENT_CHARS + 1)},
    ],
)
def test_save_note_rejects_invalid_notes_and_stores_nothing(params):
    store = SQLiteNoteStore(":memory:")
    registry = ToolRegistry()
    registry.register(SaveNoteTool(store))

    result = registry.execute("save_note", params=params)

    assert result.status is ToolResultStatus.ERROR
    assert store.list() == []


def test_save_note_content_at_the_limit_is_accepted():
    store = SQLiteNoteStore(":memory:")

    result = SaveNoteTool(store).execute(title="t", content="c" * MAX_CONTENT_CHARS)

    assert result.status is ToolResultStatus.SUCCESS


def test_the_provenance_fields_are_not_part_of_the_advertised_parameters():
    """So a plan cannot supply (or forge) them: validate_plan rejects them
    as unexpected arguments."""
    tool = SaveNoteTool(SQLiteNoteStore(":memory:"))
    assert {p.name for p in tool.parameters} == {"title", "content"}
    registry = ToolRegistry()
    registry.register(tool)

    with pytest.raises(PlanValidationError, match="unexpected argument"):
        validate_plan(
            [
                {
                    "id": "s1",
                    "tool": "save_note",
                    "arguments": {"title": "t", "content": "c", "source_job_id": "someone-else"},
                    "depends_on": [],
                }
            ],
            registry,
            max_steps=5,
            default_retry=RetryPolicy(max_attempts=1),
        )


def test_an_explicit_chat_command_saves_a_note_with_no_source():
    store = SQLiteNoteStore(":memory:")
    registry = ToolRegistry()
    registry.register(SaveNoteTool(store))
    jarvis = JarvisCore(
        provider=type("P", (AIProvider,), {"generate_reply": lambda s, m: "unused"})(),
        memory=SQLiteMemoryStore(":memory:"),
        tools=registry,
    )

    reply, _, _ = jarvis.handle_chat('use tool save_note "Groceries" milk, eggs and bread')

    assert "Saved note" in reply
    note = store.list()[0]
    assert (note.title, note.content) == ("Groceries", "milk, eggs and bread")
    assert note.source_job_id is None


# --- list_notes ---------------------------------------------------------------------------------------------


def test_list_notes_returns_notes_newest_first_with_all_metadata():
    store = SQLiteNoteStore(":memory:")
    store.create("old", "1", source_job_id="job-1", source_step_id="s1")
    store.create("new", "2")
    registry = ToolRegistry()
    registry.register(ListNotesTool(store))

    result = registry.execute("list_notes")

    assert result.status is ToolResultStatus.SUCCESS
    assert [n["title"] for n in result.data] == ["new", "old"]
    assert set(result.data[1]) == {
        "id", "title", "content", "created_at", "source_job_id", "source_step_id"
    }
    assert result.data[1]["source_job_id"] == "job-1"
    assert result.message == "2 note(s)."


def test_list_notes_filters_by_job_and_respects_limit():
    store = SQLiteNoteStore(":memory:")
    for i in range(4):
        store.create(f"a{i}", "c", source_job_id="job-1")
    store.create("other", "c", source_job_id="job-2")
    tool = ListNotesTool(store)

    assert [n["title"] for n in tool.execute(job_id="job-1", limit=2).data] == ["a3", "a2"]
    assert [n["title"] for n in tool.execute(job_id="job-2").data] == ["other"]


@pytest.mark.parametrize("limit", [0, -1, MAX_LIST_LIMIT + 1, 2.5, True, "10"])
def test_list_notes_rejects_a_bad_limit(limit):
    result = ListNotesTool(SQLiteNoteStore(":memory:")).execute(limit=limit)

    assert result.status is ToolResultStatus.ERROR


def test_list_notes_never_modifies_anything():
    store = SQLiteNoteStore(":memory:")
    store.create("a", "c")
    before = [n for n in store.list()]

    ListNotesTool(store).execute()
    ListNotesTool(store).execute(limit=1)

    assert store.list() == before


# --- provenance when saved by a Job step --------------------------------------------------------------------


class OneShotPlanner(Planner):
    def __init__(self, steps):
        self.steps = steps

    def plan(self, goal, *, available_tools, context, autonomy, replan_context=None):
        if replan_context is not None:
            raise PlanningError("no replan")
        return self.steps


def test_a_note_saved_by_a_job_step_records_the_job_and_step():
    notes = SQLiteNoteStore(":memory:")
    jobs = SQLiteJobStore(":memory:")
    registry = ToolRegistry()
    registry.register(SaveNoteTool(notes))
    engine = JobEngine(
        store=jobs,
        tools=registry,
        planner=OneShotPlanner(
            [
                {
                    "id": "save_it",
                    "tool": "save_note",
                    "arguments": {"title": "Findings", "content": "Acme: rockets"},
                    "depends_on": [],
                }
            ]
        ),
        max_steps=5,
        max_total_steps=10,
        max_retries_per_step=1,
        max_replans=0,
        retry_delay_seconds=0,
        poll_interval_seconds=1000,
    )
    job = jobs.create("goal")

    engine.tick()

    assert jobs.get(job.id).status == JobStatus.COMPLETED
    (note,) = notes.list()
    assert (note.source_job_id, note.source_step_id) == (job.id, "save_it")
    assert [n.title for n in notes.list(job_id=job.id)] == ["Findings"]
    # provenance is added at execution time, never written into the plan
    assert jobs.get(job.id).plan.get("save_it").arguments == {
        "title": "Findings",
        "content": "Acme: rockets",
    }


# --- separate from memory / facts / the World Model -----------------------------------------------------------


def test_saving_notes_never_touches_memory_facts():
    memory = SQLiteMemoryStore(":memory:")
    memory.save_fact("name", "Bassim")
    notes = SQLiteNoteStore(":memory:")
    tool = SaveNoteTool(notes)

    tool.execute(title="Acme", content="my favourite_color is now green; remember this forever")

    assert memory.get_facts() == {"name": "Bassim"}


def test_notes_and_memory_use_different_databases_by_default():
    from app.config import settings

    assert settings.notes_db_path != settings.memory_db_path
    assert settings.notes_db_path == "jarvis_notes.db"


def test_notes_do_not_appear_in_the_context_jobs_or_prompts_are_built_from():
    """Nothing reads notes back automatically: a Job's context comes from
    memory facts + the Context Engine, neither of which knows about notes."""
    notes = SQLiteNoteStore(":memory:")
    notes.create("secret finding", "only readable via list_notes")
    memory = SQLiteMemoryStore(":memory:")

    assert "secret finding" not in str(memory.get_facts())
    assert "secret finding" not in str(JarvisCore._snapshot_to_job_context(None))


def test_the_notes_package_does_not_depend_on_memory_the_world_model_or_tasks():
    root = pathlib.Path(__file__).resolve().parent.parent / "app" / "notes"
    for path in root.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        for forbidden in ("app.memory", "app.world_model", "app.tasks", "save_fact"):
            assert forbidden not in text, (path.name, forbidden)


def test_nothing_outside_the_notes_package_promotes_notes_into_memory():
    """No module both imports app.notes and writes facts/World Model state:
    there is no auto-promotion path."""
    app_root = pathlib.Path(__file__).resolve().parent.parent / "app"
    for path in app_root.rglob("*.py"):
        if "notes" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        if re.search(r"^\s*(from|import) app\.notes", text, re.MULTILINE):
            assert "save_fact" not in text, path
            assert "world_model" not in text or path.name in {"main.py"}, path
