"""
Tests for SQLiteSearchCache: the concrete SearchCache backend. Uses a fresh
in-memory or temp-file SQLite database per test - no real search provider
or network call involved.
"""

from app.search.base import SearchResult
from app.search.sqlite_cache import SQLiteSearchCache

RESULT = SearchResult(title="Python", url="https://python.org", snippet="lang")


def test_get_on_empty_cache_is_a_miss():
    cache = SQLiteSearchCache(db_path=":memory:")

    assert cache.get("python::5") is None


def test_set_then_get_returns_the_same_results():
    cache = SQLiteSearchCache(db_path=":memory:")

    cache.set("python::5", [RESULT])

    assert cache.get("python::5") == [RESULT]


def test_set_can_store_an_empty_result_list():
    cache = SQLiteSearchCache(db_path=":memory:")

    cache.set("no-hits::5", [])

    assert cache.get("no-hits::5") == []


def test_different_keys_do_not_collide():
    cache = SQLiteSearchCache(db_path=":memory:")

    cache.set("python::5", [RESULT])

    assert cache.get("python::10") is None


def test_expired_entry_is_treated_as_a_miss():
    cache = SQLiteSearchCache(db_path=":memory:", ttl_seconds=-1)

    cache.set("python::5", [RESULT])

    assert cache.get("python::5") is None


def test_set_overwrites_a_previous_value_for_the_same_key():
    cache = SQLiteSearchCache(db_path=":memory:")

    cache.set("python::5", [RESULT])
    cache.set("python::5", [])

    assert cache.get("python::5") == []


def test_default_ttl_is_ten_minutes():
    assert SQLiteSearchCache(db_path=":memory:").ttl_seconds == 600


def test_source_metadata_survives_a_round_trip():
    cache = SQLiteSearchCache(db_path=":memory:")
    result = SearchResult(
        title="Python", url="https://python.org", snippet="lang", source="searxng"
    )

    cache.set("python::5", [result])

    assert cache.get("python::5") == [result]
    assert cache.get("python::5")[0].source == "searxng"


def test_query_provenance_survives_a_round_trip():
    cache = SQLiteSearchCache(db_path=":memory:")
    result = SearchResult(
        title="Python",
        url="https://python.org",
        snippet="lang",
        source="searxng",
        query="python 2026",
    )

    cache.set("python 2026::5", [result])

    assert cache.get("python 2026::5") == [result]
    assert cache.get("python 2026::5")[0].query == "python 2026"


def test_cache_persists_across_instances_pointed_at_the_same_file(tmp_path):
    db_path = str(tmp_path / "cache.db")

    SQLiteSearchCache(db_path=db_path).set("python::5", [RESULT])

    reopened = SQLiteSearchCache(db_path=db_path)
    assert reopened.get("python::5") == [RESULT]
