"""
Tests for the SearXNG search provider. All HTTP calls are monkeypatched, so
no real network access happens.
"""

import httpx
import pytest

from app.search import searxng
from app.search.base import SearchResult
from app.search.searxng import SearXNGError, SearXNGProvider


def test_search_raises_without_searxng_url(monkeypatch):
    monkeypatch.setattr(searxng.settings, "searxng_url", "")

    with pytest.raises(SearXNGError, match="JARVIS_SEARXNG_URL"):
        SearXNGProvider().search("python")


def test_search_returns_standardized_results(monkeypatch):
    monkeypatch.setattr(searxng.settings, "searxng_url", "http://localhost:8080")

    def fake_get(url, params, timeout):
        assert url == "http://localhost:8080/search"
        assert params == {"q": "python", "format": "json"}
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "title": "Python",
                        "url": "https://python.org",
                        "content": "The Python language",
                    },
                    {
                        "title": "Docs",
                        "url": "https://docs.python.org",
                        "content": "Python docs",
                    },
                ]
            },
            request=httpx.Request("GET", url),
        )

    monkeypatch.setattr(searxng.httpx, "get", fake_get)

    results = SearXNGProvider().search("python")

    assert results == [
        SearchResult(
            title="Python",
            url="https://python.org",
            snippet="The Python language",
            source="searxng",
        ),
        SearchResult(
            title="Docs",
            url="https://docs.python.org",
            snippet="Python docs",
            source="searxng",
        ),
    ]


def test_search_strips_trailing_slash_from_base_url(monkeypatch):
    monkeypatch.setattr(searxng.settings, "searxng_url", "http://localhost:8080/")

    def fake_get(url, params, timeout):
        assert url == "http://localhost:8080/search"
        return httpx.Response(
            200, json={"results": []}, request=httpx.Request("GET", url)
        )

    monkeypatch.setattr(searxng.httpx, "get", fake_get)

    assert SearXNGProvider().search("python") == []


def test_search_limits_to_max_results(monkeypatch):
    monkeypatch.setattr(searxng.settings, "searxng_url", "http://localhost:8080")

    def fake_get(url, params, timeout):
        return httpx.Response(
            200,
            json={
                "results": [
                    {"title": f"R{i}", "url": f"https://x.com/{i}", "content": ""}
                    for i in range(10)
                ]
            },
            request=httpx.Request("GET", url),
        )

    monkeypatch.setattr(searxng.httpx, "get", fake_get)

    results = SearXNGProvider().search("python", max_results=3)

    assert len(results) == 3


def test_search_raises_on_unexpected_response_shape(monkeypatch):
    monkeypatch.setattr(searxng.settings, "searxng_url", "http://localhost:8080")

    def fake_get(url, params, timeout):
        return httpx.Response(
            200, json={"unexpected": True}, request=httpx.Request("GET", url)
        )

    monkeypatch.setattr(searxng.httpx, "get", fake_get)

    with pytest.raises(SearXNGError, match="Unexpected SearXNG response"):
        SearXNGProvider().search("python")


def test_search_raises_on_malformed_result_item(monkeypatch):
    monkeypatch.setattr(searxng.settings, "searxng_url", "http://localhost:8080")

    def fake_get(url, params, timeout):
        return httpx.Response(
            200,
            json={"results": [{"title": "Missing url and content"}]},
            request=httpx.Request("GET", url),
        )

    monkeypatch.setattr(searxng.httpx, "get", fake_get)

    with pytest.raises(SearXNGError, match="Unexpected SearXNG result item"):
        SearXNGProvider().search("python")


def test_search_raises_on_network_failure(monkeypatch):
    monkeypatch.setattr(searxng.settings, "searxng_url", "http://localhost:8080")

    def fake_get(url, params, timeout):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(searxng.httpx, "get", fake_get)

    with pytest.raises(SearXNGError, match="Failed to reach SearXNG"):
        SearXNGProvider().search("python")


def test_search_raises_on_http_error_status(monkeypatch):
    monkeypatch.setattr(searxng.settings, "searxng_url", "http://localhost:8080")

    def fake_get(url, params, timeout):
        return httpx.Response(
            500, text="server error", request=httpx.Request("GET", url)
        )

    monkeypatch.setattr(searxng.httpx, "get", fake_get)

    with pytest.raises(SearXNGError, match="Failed to reach SearXNG"):
        SearXNGProvider().search("python")
