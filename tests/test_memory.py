"""
Tests for the SQLite-backed memory store: conversation history and
persistent user facts.
"""

from app.memory.sqlite_store import SQLiteMemoryStore


def test_create_conversation_returns_unique_ids():
    store = SQLiteMemoryStore(":memory:")

    first = store.create_conversation()
    second = store.create_conversation()

    assert first != second


def test_messages_round_trip_in_order():
    store = SQLiteMemoryStore(":memory:")
    conversation_id = store.create_conversation()

    store.add_message(conversation_id, "user", "hello")
    store.add_message(conversation_id, "assistant", "hi there")

    messages = store.get_messages(conversation_id)

    assert [(m.role, m.content) for m in messages] == [
        ("user", "hello"),
        ("assistant", "hi there"),
    ]


def test_get_messages_for_unknown_conversation_is_empty():
    store = SQLiteMemoryStore(":memory:")

    assert store.get_messages("does-not-exist") == []


def test_conversation_exists_reflects_creation():
    store = SQLiteMemoryStore(":memory:")

    conversation_id = store.create_conversation()

    assert store.conversation_exists(conversation_id) is True
    assert store.conversation_exists("does-not-exist") is False


def test_messages_persist_across_store_instances(tmp_path):
    db_path = str(tmp_path / "memory.db")
    store = SQLiteMemoryStore(db_path)
    conversation_id = store.create_conversation()
    store.add_message(conversation_id, "user", "remember this")

    reopened = SQLiteMemoryStore(db_path)

    assert [m.content for m in reopened.get_messages(conversation_id)] == [
        "remember this"
    ]


def test_save_fact_and_get_facts_round_trip():
    store = SQLiteMemoryStore(":memory:")

    store.save_fact("name", "Bassim")

    assert store.get_facts() == {"name": "Bassim"}


def test_save_fact_overwrites_existing_key():
    store = SQLiteMemoryStore(":memory:")

    store.save_fact("name", "Bassim")
    store.save_fact("name", "B")

    assert store.get_facts() == {"name": "B"}


def test_facts_persist_across_store_instances(tmp_path):
    db_path = str(tmp_path / "memory.db")
    store = SQLiteMemoryStore(db_path)
    store.save_fact("timezone", "UTC")

    reopened = SQLiteMemoryStore(db_path)

    assert reopened.get_facts() == {"timezone": "UTC"}
