"""
Unit tests for the manual provider-chat utility's orchestration logic
(tests/manual_provider_chat.py) - never makes a real network/API call.

`factory.create_provider` is monkeypatched to return an in-memory stub (never
touches httpx/settings), so these tests only exercise ask_provider/
run_providers/format_result/print_report/main - the same code path the
manual script uses, minus the actual HTTP request. A dedicated test also
patches httpx.post to raise if it's ever called, as an explicit guard that
this file never reaches the network even if a future change breaks the
factory patch above.
"""

import pytest

from app.providers.base import AIProvider, ProviderError
from tests import manual_provider_chat as m


class _StubProvider(AIProvider):
    """Stands in for a real provider (as if built by factory.create_provider)
    without any network access."""

    def __init__(self, name, *, answer=None, error=None, raise_exc=None):
        self.name = name
        self.answer = answer
        self.error = error
        self.raise_exc = raise_exc

    def generate_reply(self, message: str) -> str:
        if self.raise_exc is not None:
            raise self.raise_exc
        if self.error is not None:
            raise ProviderError(self.error)
        return self.answer


@pytest.fixture
def stub_factory(monkeypatch):
    """Patches factory.create_provider used by ask_provider so every "real"
    provider is a _StubProvider instead of hitting the network. `providers`
    maps provider name -> _StubProvider instance."""

    def _install(providers):
        def fake_create_provider(name):
            return providers[name]

        monkeypatch.setattr(m.factory, "create_provider", fake_create_provider)

    return _install


def test_ask_provider_uses_factory_for_requested_name(stub_factory):
    stub_factory({"groq": _StubProvider("groq", answer="Tokyo.")})

    result = m.ask_provider("groq", "What is the capital of Japan?")

    assert result.name == "groq"
    assert result.display_name == "Groq"
    assert result.success is True
    assert result.answer == "Tokyo."
    assert result.reason is None


def test_run_providers_all_iterates_every_provider(stub_factory):
    stubs = {name: _StubProvider(name, answer=f"{name} answer") for name in m.PROVIDERS}
    stub_factory(stubs)

    results = m.run_providers(m.PROVIDERS, m.DEFAULT_QUESTION)

    assert [r.name for r in results] == m.PROVIDERS
    assert all(r.success for r in results)


def test_ask_provider_reports_failure_reason(stub_factory):
    stub_factory(
        {"cloudflare": _StubProvider("cloudflare", error="Cloudflare returned 500: boom")}
    )

    result = m.ask_provider("cloudflare", m.DEFAULT_QUESTION)

    assert result.success is False
    assert result.answer is None
    assert "500" in result.reason


def test_ask_provider_missing_credentials(stub_factory):
    stub_factory(
        {
            "openrouter": _StubProvider(
                "openrouter", error="JARVIS_OPENROUTER_API_KEY is not set."
            )
        }
    )

    result = m.ask_provider("openrouter", m.DEFAULT_QUESTION)

    assert result.success is False
    assert "not set" in result.reason


def test_ask_provider_handles_timeout_style_error(stub_factory):
    stub_factory(
        {"gemini": _StubProvider("gemini", error="Failed to reach Gemini: timed out")}
    )

    result = m.ask_provider("gemini", m.DEFAULT_QUESTION)

    assert result.success is False
    assert "timed out" in result.reason.lower()


def test_ask_provider_handles_unexpected_non_provider_exception(stub_factory):
    stub_factory({"gemini": _StubProvider("gemini", raise_exc=TimeoutError("boom detail"))})

    result = m.ask_provider("gemini", m.DEFAULT_QUESTION)

    assert result.success is False
    assert "TimeoutError" in result.reason
    # The raw exception message is deliberately never reused - only the
    # exception's type name - since a non-ProviderError could be holding
    # anything, including a URL/header.
    assert "boom detail" not in result.reason


def test_run_providers_all_continues_after_one_failure(stub_factory):
    stubs = {
        "gemini": _StubProvider("gemini", answer="Tokyo."),
        "groq": _StubProvider("groq", error="Groq returned 500: boom"),
        "openrouter": _StubProvider("openrouter", answer="Tokyo is the capital of Japan."),
        "cloudflare": _StubProvider("cloudflare", answer="Tokyo."),
    }
    stub_factory(stubs)

    results = m.run_providers(m.PROVIDERS, m.DEFAULT_QUESTION)

    assert [r.name for r in results] == m.PROVIDERS
    assert results[0].success is True
    assert results[1].success is False
    assert results[2].success is True
    assert results[3].success is True


def test_print_report_pass_when_all_succeed(stub_factory, capsys):
    stubs = {name: _StubProvider(name, answer=f"{name} answer") for name in m.PROVIDERS}
    stub_factory(stubs)
    results = m.run_providers(m.PROVIDERS, m.DEFAULT_QUESTION)

    passed = m.print_report(m.PROVIDERS, m.DEFAULT_QUESTION, results)

    assert passed is True
    out = capsys.readouterr().out
    assert "=== Jarvis AI Provider Chat Test ===" in out
    assert f"Question: {m.DEFAULT_QUESTION}" in out
    assert "Gemini: SUCCESS" in out
    assert "Groq: SUCCESS" in out
    assert "OpenRouter: SUCCESS" in out
    assert "Cloudflare: SUCCESS" in out
    assert "Provider chat test: PASS" in out


def test_print_report_fail_when_one_provider_fails(stub_factory, capsys):
    stubs = {
        "gemini": _StubProvider("gemini", answer="Tokyo."),
        "groq": _StubProvider("groq", error="Groq returned 401: invalid key"),
        "openrouter": _StubProvider("openrouter", answer="Tokyo."),
        "cloudflare": _StubProvider("cloudflare", answer="Tokyo."),
    }
    stub_factory(stubs)
    results = m.run_providers(m.PROVIDERS, m.DEFAULT_QUESTION)

    passed = m.print_report(m.PROVIDERS, m.DEFAULT_QUESTION, results)

    assert passed is False
    out = capsys.readouterr().out
    assert "Groq: FAILURE" in out
    assert "Reason: Groq returned 401: invalid key" in out
    assert "Provider chat test: FAIL" in out


def test_print_report_never_includes_auth_header_text(stub_factory, capsys):
    # Mirrors the real adapters' own discipline (see
    # app/providers/_openai_compatible.py): a 401 ProviderError message is
    # built from status/body only, never the Authorization header - this
    # pins that the report never adds anything else on top of that message.
    stub_factory(
        {"groq": _StubProvider("groq", error='Groq returned 401: {"error": "invalid_api_key"}')}
    )
    results = m.run_providers(["groq"], m.DEFAULT_QUESTION)

    m.print_report(["groq"], m.DEFAULT_QUESTION, results)

    out = capsys.readouterr().out
    assert "Bearer" not in out
    assert "Authorization" not in out


def test_main_returns_zero_on_pass(stub_factory):
    stub_factory({"groq": _StubProvider("groq", answer="Tokyo.")})

    exit_code = m.main(["--provider", "groq"])

    assert exit_code == 0


def test_main_returns_nonzero_on_fail(stub_factory):
    stub_factory({"groq": _StubProvider("groq", error="Groq returned 500: boom")})

    exit_code = m.main(["--provider", "groq"])

    assert exit_code == 1


def test_main_all_one_failure_does_not_stop_the_rest(stub_factory, capsys):
    stubs = {
        "gemini": _StubProvider("gemini", answer="Tokyo."),
        "groq": _StubProvider("groq", error="Groq returned 500: boom"),
        "openrouter": _StubProvider("openrouter", answer="Tokyo."),
        "cloudflare": _StubProvider("cloudflare", answer="Tokyo."),
    }
    stub_factory(stubs)

    exit_code = m.main(["--provider", "all"])

    assert exit_code == 1
    out = capsys.readouterr().out
    assert "Gemini: SUCCESS" in out
    assert "Groq: FAILURE" in out
    assert "OpenRouter: SUCCESS" in out
    assert "Cloudflare: SUCCESS" in out


def test_main_accepts_custom_question(stub_factory):
    stub_factory({"gemini": _StubProvider("gemini", answer="42.")})

    exit_code = m.main(["--provider", "gemini", "--question", "What is 6*7?"])

    assert exit_code == 0


def test_no_real_network_calls(monkeypatch, stub_factory):
    """Explicit guard: even if the factory patch above were ever removed or
    broken, this pins that running the utility's full main() never reaches
    httpx - the only place any provider makes an actual HTTP request."""

    def fail_post(*args, **kwargs):
        raise AssertionError("a test attempted a real network call via httpx.post")

    monkeypatch.setattr("httpx.post", fail_post)
    stub_factory({name: _StubProvider(name, answer="ok") for name in m.PROVIDERS})

    exit_code = m.main(["--provider", "all"])

    assert exit_code == 0
