"""
User-controlled memory editing (spec 104) and the memory provenance / history view (spec 93): the memory write policy's
dashboard door. What the user asked Jarvis to keep is listed with the wording Jarvis keeps it in, its stored value and where it
came from; ONE exact, user-picked item can be forgotten, only after an explicit confirmation (a single-use token issued by the
policy itself), through the SAME deletion, audit scrub and "forgotten" record as "forget ..." in chat. Real SQLite stores in
temporary folders; no AI provider is involved.
"""

import re
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.dashboard.api import DashboardSources, build_dashboard_router
from app.host_guard import LoopbackHostMiddleware
from app.memory.sqlite_store import SQLiteMemoryStore
from app.memory_policy import service as memory_service
from app.memory_policy.factory import create_default_memory_write_service
from app.notes.sqlite_store import SQLiteNoteStore
from app.orchestration.sqlite_interaction_store import SQLiteInteractionStore
from app.orchestration.sqlite_store import SQLiteJobStore
from app.reminders.sqlite_store import SQLiteReminderStore
from app.tasks.sqlite_store import SQLiteTaskStore
from app.tools.registry import ToolRegistry
from app.world_model.factory import create_default_world_model_service, create_default_world_model_store

DASHBOARD = {"X-Jarvis-Dashboard": "1"}


class Rig:
    """The real memory write service over real stores in one folder - rebuildable, to prove what survives a restart."""

    def __init__(self, folder: Path):
        self.folder = folder
        self.build()

    def build(self):
        db = str(self.folder / "world.db")
        self.wm = create_default_world_model_service(create_default_world_model_store(db))
        self.memory = SQLiteMemoryStore(str(self.folder / "memory.db"))
        self.service = create_default_memory_write_service(self.wm, self.memory, db)
        return self

    def say(self, message):
        return self.service.command_reply(message) or self.service.observe(message)

    def item(self, words):
        found = [i for i in self.service.remembered() if words in i["text"]]
        assert len(found) == 1, (words, self.service.remembered())
        return found[0]


@pytest.fixture
def rig(tmp_path):
    r = Rig(tmp_path)
    r.say("I prefer dark mode.")
    r.say("Remember that my dog is called Max.")
    r.say("Remember that my dog is called Max Junior.")
    return r


# --- what is kept, with its wording and provenance (93) -----------------------------------------------------------------------


def test_what_is_kept_is_listed_with_its_wording_value_and_provenance(rig):
    items = rig.service.remembered()

    texts = [i["text"] for i in items]
    assert "You prefer dark mode" in texts
    assert any(t.startswith("my dog is called Max") for t in texts)
    pref = rig.item("dark mode")
    assert pref["kind"] == "preference" and pref["value"] == "dark mode" and pref["id"] != pref["key"]
    assert pref["provenance"]["origin"] == "user_statement" and pref["provenance"]["saved_at"]
    note = rig.item("Max Junior")
    assert note["kind"] == "note" and note["id"] == note["key"] and note["key"].startswith("note:")
    assert note["provenance"]["origin"] == "user_command" and note["provenance"]["destination"] == "memory"


def test_candidates_are_not_listed_as_kept(rig):
    rig.say("I might prefer green tea.")  # hedged: at most a staged candidate, never a kept item

    assert not any("green tea" in i["text"] for i in rig.service.remembered())


def test_history_shows_what_was_kept_and_why_newest_first(rig):
    history = rig.service.history()

    assert history[0]["at"] >= history[-1]["at"]
    assert {"persisted"} <= {h["action"] for h in history}
    assert all(set(h) == {"at", "action", "kind", "key", "value", "origin", "destination", "reason"} for h in history)


# --- forgetting ONE exact item, only after confirmation (104) ---------------------------------------------------------------


def test_preparing_deletes_nothing_and_names_the_exact_item(rig):
    note = rig.item("Max Junior")

    prepared = rig.service.prepare_forget("note", note["id"])

    assert prepared["text"] == note["text"] and prepared["token"] and prepared["expires_in_seconds"] > 0
    assert any(i["id"] == note["id"] for i in rig.service.remembered())  # still kept until confirmed


def test_confirming_forgets_exactly_that_item_and_nothing_else(rig):
    junior, pref = rig.item("Max Junior"), rig.item("dark mode")
    max_ = next(i for i in rig.service.remembered() if i["key"] == "note:my_dog_is_called_max")  # the near-duplicate
    before = {i["id"] for i in rig.service.remembered()}

    result = rig.service.confirm_forget(rig.service.prepare_forget("note", junior["id"])["token"])

    assert result["forgotten"] is True and "forgotten" in result["message"]
    assert {i["id"] for i in rig.service.remembered()} == before - {junior["id"]}
    assert max_["key"] in rig.memory.get_facts() and junior["key"] not in rig.memory.get_facts()
    assert rig.wm.get_preferences(key=pref["key"])


def test_two_preferences_with_the_same_key_are_told_apart_by_id(tmp_path):
    from app.world_model.base import EntitySource

    r = Rig(tmp_path)
    work = r.wm.create_preference("prefers:editor", "VS Code", category="work", source=EntitySource.USER)
    home = r.wm.create_preference("prefers:editor", "Notepad", category="home", source=EntitySource.USER)

    r.service.confirm_forget(r.service.prepare_forget("preference", work.id)["token"])

    remaining = r.wm.get_preferences(key="prefers:editor")
    assert [p.id for p in remaining] == [home.id]


def test_a_token_works_once(rig):
    token = rig.service.prepare_forget("note", rig.item("Max Junior")["id"])["token"]

    assert rig.service.confirm_forget(token)["forgotten"] is True
    second = rig.service.confirm_forget(token)
    assert second["forgotten"] is False and "already used" in second["message"]


def test_an_expired_token_forgets_nothing(rig, monkeypatch):
    note = rig.item("Max Junior")
    token = rig.service.prepare_forget("note", note["id"])["token"]
    monkeypatch.setattr(memory_service, "FORGET_CONFIRM_SECONDS", -1)

    assert rig.service.confirm_forget(token)["forgotten"] is False
    assert any(i["id"] == note["id"] for i in rig.service.remembered())


@pytest.mark.parametrize("token", ["", "made-up", None, 42])
def test_an_unknown_token_forgets_nothing(rig, token):
    before = rig.service.remembered()

    assert rig.service.confirm_forget(token)["forgotten"] is False
    assert rig.service.remembered() == before


@pytest.mark.parametrize("kind, item_id", [
    ("note", "note:no-such-thing"),
    ("preference", "not-an-id"),
    ("note", ""),
    ("birthday", "dob"),  # no date of birth is kept here
])
def test_preparing_an_item_that_is_not_kept_is_refused(rig, kind, item_id):
    assert rig.service.prepare_forget(kind, item_id) is None


def test_the_kind_must_match_too(rig):
    note = rig.item("Max Junior")

    assert rig.service.prepare_forget("preference", note["id"]) is None


def test_an_item_forgotten_meanwhile_is_not_forgotten_twice(rig):
    note = rig.item("Max Junior")
    token = rig.service.prepare_forget("note", note["id"])["token"]
    rig.memory.delete_fact(note["key"])  # gone by another route between the two steps

    result = rig.service.confirm_forget(token)

    assert result["forgotten"] is False and "no longer kept" in result["message"]


def test_forgetting_records_it_and_scrubs_the_content_from_the_journal(rig):
    note = rig.item("Max Junior")

    rig.service.confirm_forget(rig.service.prepare_forget("note", note["id"])["token"])

    assert rig.service.store.journal(key=note["key"]) == []  # the content-derived key is gone from the audit trail
    forgotten = [e for e in rig.service.store.journal() if e.action == "forgotten"]
    assert len(forgotten) == 1 and forgotten[0].key.startswith("forgotten:")
    assert forgotten[0].reason == "the user asked Jarvis to forget it"
    same_item = rig.service.store.journal(key=forgotten[0].key)
    assert {e.action for e in same_item} == {"persisted", "forgotten"} and all(e.value is None for e in same_item)
    history_text = str(rig.service.history())
    assert "junior" not in history_text.lower() and "forgotten" in history_text


def test_nothing_on_disk_still_says_what_was_forgotten(rig, tmp_path):
    import sqlite3

    note = rig.item("Max Junior")
    rig.service.confirm_forget(rig.service.prepare_forget("note", note["id"])["token"])

    dump = []
    for name in ("world.db", "memory.db"):
        connection = sqlite3.connect(str(rig.folder / name))
        for (table,) in connection.execute("SELECT name FROM sqlite_master WHERE type='table'"):
            dump += [" ".join(map(str, row)) for row in connection.execute(f'SELECT * FROM "{table}"')]
        connection.close()
    text = " ".join(dump).lower()
    assert "junior" not in text  # neither the value nor the key made from its words
    assert "my dog is called max" in text  # the near-duplicate that was NOT forgotten is still kept


def test_a_forget_survives_a_restart_and_so_does_its_record(rig):
    note = rig.item("Max Junior")
    rig.service.confirm_forget(rig.service.prepare_forget("note", note["id"])["token"])

    restarted = Rig(rig.folder)

    assert not any(i["id"] == note["id"] for i in restarted.service.remembered())
    assert [e.action for e in restarted.service.store.journal() if e.key.startswith("forgotten:")].count("forgotten") == 1
    assert any(i["key"] == "note:my_dog_is_called_max" for i in restarted.service.remembered())  # the near-duplicate stays


def test_tokens_do_not_survive_a_restart(rig):
    token = rig.service.prepare_forget("note", rig.item("Max Junior")["id"])["token"]

    assert Rig(rig.folder).service.confirm_forget(token)["forgotten"] is False


# --- the dashboard door ---------------------------------------------------------------------------------------------------------


@pytest.fixture
def http(rig, tmp_path):
    jobs = SQLiteJobStore(str(tmp_path / "orchestration.db"))
    sources = DashboardSources(
        tasks=SQLiteTaskStore(str(tmp_path / "tasks.db")), reminders=SQLiteReminderStore(str(tmp_path / "reminders.db")),
        notes=SQLiteNoteStore(str(tmp_path / "notes.db")), jobs=jobs, interactions=SQLiteInteractionStore(jobs),
        tools=ToolRegistry(), memory=rig.memory, provider_status=lambda: {"providers": []}, health=lambda: {"status": "ok"},
        world_model=rig.wm, memory_policy=rig.service,
    )
    app = FastAPI()
    app.add_middleware(LoopbackHostMiddleware)
    app.include_router(build_dashboard_router(sources))
    return TestClient(app, base_url="http://127.0.0.1:8000")


def test_the_memory_view_lists_kept_items_with_provenance_and_the_history(http, rig):
    body = http.get("/api/memory").json()

    assert {i["text"] for i in body["remembered"]} == {i["text"] for i in rig.service.remembered()}
    assert body["remembered"][0]["provenance"] is not None and body["history"]


def test_the_dashboard_forgets_one_item_in_two_confirmed_steps(http, rig):
    note = rig.item("Max Junior")

    prepared = http.post("/api/memory/forget/prepare", json={"kind": "note", "id": note["id"]}, headers=DASHBOARD)
    assert prepared.status_code == 200 and prepared.json()["text"] == note["text"]
    assert any(i["id"] == note["id"] for i in rig.service.remembered())  # nothing deleted yet

    confirmed = http.post("/api/memory/forget/confirm", json={"token": prepared.json()["token"]}, headers=DASHBOARD)
    assert confirmed.status_code == 200 and confirmed.json()["forgotten"] is True
    assert not any(i["id"] == note["id"] for i in rig.service.remembered())
    assert http.post("/api/memory/forget/confirm", json={"token": prepared.json()["token"]}, headers=DASHBOARD).status_code == 409


def test_the_dashboard_cannot_forget_without_the_confirmation_step(http, rig):
    before = rig.service.remembered()

    assert http.post("/api/memory/forget/confirm", json={"token": "guess"}, headers=DASHBOARD).status_code == 409
    assert http.post("/api/memory/forget", json={"kind": "note", "id": rig.item("Max Junior")["id"]},
                     headers=DASHBOARD).status_code in (404, 405)
    assert rig.service.remembered() == before


def test_preparing_a_missing_item_is_404(http):
    assert http.post("/api/memory/forget/prepare", json={"kind": "note", "id": "note:nothing"}, headers=DASHBOARD).status_code == 404


@pytest.mark.parametrize("headers", [{}, {**DASHBOARD, "Origin": "https://evil.example"}, {**DASHBOARD, "Sec-Fetch-Site": "cross-site"}])
def test_forgetting_is_refused_from_anywhere_but_the_dashboard(http, rig, headers):
    note = rig.item("Max Junior")
    token = rig.service.prepare_forget("note", note["id"])["token"]  # even holding a valid token

    assert http.post("/api/memory/forget/prepare", json={"kind": "note", "id": note["id"]}, headers=headers).status_code == 403
    assert http.post("/api/memory/forget/confirm", json={"token": token}, headers=headers).status_code == 403
    assert any(i["id"] == note["id"] for i in rig.service.remembered())


def test_there_is_no_bulk_forget_anywhere():
    api = (Path(__file__).resolve().parent.parent / "app" / "dashboard" / "api.py").read_text(encoding="utf-8")
    service = (Path(__file__).resolve().parent.parent / "app" / "memory_policy" / "service.py").read_text(encoding="utf-8")

    assert not re.search(r"def\s+(forget_all|clear_memory|delete_all|wipe\w*)|/api/memory/(clear|wipe|forget_all|delete)", api + service)
    # The dashboard never deletes memory itself: only through the policy's two-step door.
    for direct in ("delete_fact", "delete_preference", "_forget_item", "scrub_key"):
        assert direct not in api, direct


def test_forget_key_erases_values_the_key_and_a_staged_candidate_for_that_item_only(tmp_path):
    from app.memory_policy.store import SQLiteWriteStore

    store = SQLiteWriteStore(str(tmp_path / "w.db"))
    store.record(action="persisted", kind="fact", key="note:secret_plan", value="secret plan", previous_value=None,
                 origin="user_command", destination="memory", source="user", confidence=None, reason="stated by the user")
    store.record(action="persisted", kind="fact", key="note:other", value="other", previous_value=None,
                 origin="user_command", destination="memory", source="user", confidence=None, reason="stated by the user")
    store.stage_candidate(kind="fact", key="note:secret_plan", value="secret plan", category="general", origin="inference",
                          confidence=0.5, evidence="I think my secret plan is ...")

    ref = store.forget_key("fact", "note:secret_plan")

    assert ref.startswith("forgotten:") and store.journal(key="note:secret_plan") == []
    assert [e.value for e in store.journal(key=ref)] == [None]
    assert store.get_candidate("fact", "note:secret_plan") is None
    assert store.journal(key="note:other")[0].value == "other"  # an unrelated item is untouched
