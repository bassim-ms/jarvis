"""
Unit tests for the manual provider-fallback utility's orchestration logic
(tests/manual_provider_fallback.py) - never makes a real network/API call.

`factory.create_provider` is monkeypatched to return an in-memory fake
("real" provider stand-ins that either succeed or raise ProviderError, never
touch httpx/settings), so these tests only exercise build_recorders/
run_manager/evaluate/print_report - the same code path the manual script
uses, minus the actual HTTP request.
"""

import pytest

from app.providers.base import AIProvider, ProviderError
from app.providers.manager import AllProvidersFailedError
from tests import manual_provider_fallback as m


class _StubProvider(AIProvider):
    """Stands in for a real provider (as if built by factory.create_provider)
    without any network access."""

    def __init__(self, name, *, succeed=True):
        self.name = name
        self.succeed = succeed

    def generate_reply(self, message: str) -> str:
        if self.succeed:
            return f"{self.name} reply to: {message}"
        raise ProviderError(f"{self.name} returned 500: boom")


@pytest.fixture
def stub_factory(monkeypatch):
    """Patches factory.create_provider used by build_recorders so the
    "real" provider for the target is a _StubProvider instead of hitting
    the network. `outcomes` maps provider name -> succeed bool."""

    def _install(outcomes):
        def fake_create_provider(name):
            return _StubProvider(name, succeed=outcomes.get(name, True))

        monkeypatch.setattr(m.factory, "create_provider", fake_create_provider)

    return _install


def test_build_recorders_only_target_is_real(stub_factory):
    stub_factory({})
    recorders = m.build_recorders("openrouter")

    kinds = {r.name: r.kind for r in recorders}
    assert kinds == {
        "gemini": "fake",
        "groq": "fake",
        "openrouter": "real",
        "cloudflare": "fake",
    }


def test_build_recorders_none_target_all_fake(stub_factory):
    stub_factory({})
    recorders = m.build_recorders("none")

    assert all(r.kind == "fake" for r in recorders)


def test_fake_failing_provider_never_succeeds():
    fake = m.FakeFailingProvider("gemini")
    with pytest.raises(ProviderError, match="intentionally failed"):
        fake.generate_reply("hi")


@pytest.mark.parametrize("target", ["gemini", "groq", "openrouter", "cloudflare"])
def test_run_manager_target_succeeds_earlier_intentionally_fail(stub_factory, target):
    stub_factory({})
    recorders = m.build_recorders(target)
    reply, error = m.run_manager(recorders)

    assert error is None
    assert reply == f"{target} reply to: {m.TEST_PROMPT}"

    target_index = m.ORDER.index(target)
    for recorder in recorders[:target_index]:
        assert recorder.invoked
        assert not recorder.succeeded
        assert recorder.kind == "fake"

    target_recorder = recorders[target_index]
    assert target_recorder.invoked
    assert target_recorder.succeeded

    # Providers after the target were never reached - the manager stops at
    # the first success.
    for recorder in recorders[target_index + 1 :]:
        assert not recorder.invoked


def test_run_manager_none_raises_all_providers_failed(stub_factory):
    stub_factory({})
    recorders = m.build_recorders("none")
    reply, error = m.run_manager(recorders)

    assert reply is None
    assert isinstance(error, AllProvidersFailedError)
    assert all(r.invoked and not r.succeeded for r in recorders)


@pytest.mark.parametrize("target", ["gemini", "groq", "openrouter", "cloudflare"])
def test_evaluate_passes_when_target_succeeds(stub_factory, target):
    stub_factory({})
    recorders = m.build_recorders(target)
    _, error = m.run_manager(recorders)

    passed, reason = m.evaluate(target, recorders, error)

    assert passed is True
    assert reason is None


def test_evaluate_none_passes_when_all_fail(stub_factory):
    stub_factory({})
    recorders = m.build_recorders("none")
    _, error = m.run_manager(recorders)

    passed, reason = m.evaluate("none", recorders, error)

    assert passed is True
    assert reason is None


def test_evaluate_fails_when_target_also_fails(stub_factory):
    # Even the "real" target provider raises here - simulates the target's
    # real API genuinely being down during a manual run.
    stub_factory({"openrouter": False})
    recorders = m.build_recorders("openrouter")
    _, error = m.run_manager(recorders)

    passed, reason = m.evaluate("openrouter", recorders, error)

    assert passed is False
    assert "openrouter" in reason.lower() or "OpenRouter" in reason


def test_evaluate_none_fails_if_a_provider_unexpectedly_succeeds(stub_factory):
    recorders = m.build_recorders("none")
    # Force one of the fakes to behave like it succeeded, without going
    # through generate_reply (simulates a bug rather than re-testing
    # FakeFailingProvider itself).
    recorders[1].invoked = True
    recorders[1].succeeded = True

    passed, reason = m.evaluate("none", recorders, AllProvidersFailedError("x"))

    assert passed is False


def test_print_report_returns_pass_status(stub_factory, capsys):
    stub_factory({})
    recorders = m.build_recorders("groq")
    reply, error = m.run_manager(recorders)

    passed = m.print_report("groq", recorders, reply, error)

    assert passed is True
    out = capsys.readouterr().out
    assert "Gemini: INTENTIONALLY FAILED" in out
    assert "Groq: REAL REQUEST -> SUCCESS" in out
    assert "OpenRouter: NOT ATTEMPTED" in out
    assert "Cloudflare: NOT ATTEMPTED" in out
    assert "Fallback test: PASS" in out
    assert "Successful provider: groq" in out


def test_print_report_none_target(stub_factory, capsys):
    stub_factory({})
    recorders = m.build_recorders("none")
    reply, error = m.run_manager(recorders)

    passed = m.print_report("none", recorders, reply, error)

    assert passed is True
    out = capsys.readouterr().out
    assert "AllProvidersFailedError: CONFIRMED" in out
    assert "Fallback test: PASS" in out


def test_main_returns_zero_on_pass(stub_factory, capsys):
    stub_factory({})
    exit_code = m.main(["--target", "cloudflare"])

    assert exit_code == 0


def test_main_returns_nonzero_on_fail(stub_factory, capsys):
    stub_factory({"cloudflare": False})
    exit_code = m.main(["--target", "cloudflare"])

    assert exit_code == 1


def test_status_line_never_includes_a_bearer_token(stub_factory, capsys):
    """Guards rule 13 (never print secrets): even a failing "real" provider
    can only ever raise a ProviderError whose message this codebase already
    builds without keys/headers (see _openai_compatible.py) - this test
    just pins that the report never adds anything else to that message."""
    stub_factory({"groq": False})
    recorders = m.build_recorders("groq")
    m.run_manager(recorders)

    line = m.status_line(recorders[1])
    assert "Bearer" not in line
    assert "Authorization" not in line
