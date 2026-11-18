"""
Provider statistics from real requests (app/providers/stats.py; spec 70-74, 885): latency, outcomes by kind, rate limits, a
reliability score over the last 50 ACTUAL attempts always paired with its sample size, requests and tokens per day - recorded by
the AIProviderManager for every real attempt, persisted so they survive a restart, shown by GET /providers/status and the
dashboard. Only fakes and temporary databases: no request ever reaches a provider and no quota is used.
"""

import json
import sqlite3
import threading
from datetime import datetime, timedelta, timezone

import httpx
import pytest
from fastapi.testclient import TestClient

from app import main
from app.providers import _openai_compatible, gemini, usage
from app.providers.base import AIProvider, ProviderError
from app.providers.factory import parse_daily_limits
from app.providers.manager import AIProviderEntry, AIProviderManager, AllProvidersFailedError, call_structured
from app.providers.stats import (
    RECENT_WINDOW,
    InvalidStatsState,
    ProviderStats,
    SQLiteProviderStatsStore,
    load_stats,
)

FIXED_ORDER = ["groq", "openrouter", "gemini", "cloudflare", "freellmapi"]
DUBAI = timezone(timedelta(hours=4))
T0 = datetime(2026, 11, 18, 21, 0, tzinfo=DUBAI)


class Scripted(AIProvider):
    """Answers or raises from a script; `tokens` (if given) is noted like a real adapter notes a response's usage."""

    def __init__(self, *outcomes, tokens=None):
        self.outcomes, self.tokens = list(outcomes), tokens

    def generate_reply(self, message: str) -> str:
        outcome = self.outcomes.pop(0) if self.outcomes else "fine"
        if isinstance(outcome, BaseException):
            raise outcome
        if self.tokens is not None:
            usage.note(total_tokens=self.tokens)
        return outcome


class Timer:
    """Every attempt takes the next listed duration (seconds); start/stop calls alternate."""

    def __init__(self, *durations):
        self.durations, self.now, self.stop_next = list(durations), 0.0, False

    def __call__(self):
        if self.stop_next:
            self.now += self.durations.pop(0) if self.durations else 0.1
        self.stop_next = not self.stop_next
        return self.now


def chain(*providers, timer=None, wall=lambda: T0, store=None, **options):
    options.setdefault("sleep", lambda seconds: None)
    return AIProviderManager([AIProviderEntry(name, p) for name, p in zip(FIXED_ORDER, providers)],
                             timer=timer or Timer(), wall_clock=wall, stats_store=store, **options)


def snap(manager, name):
    return manager.stats_snapshot(name)


# --- missing data is reported as missing, never invented --------------------------------------------------------------------


def test_with_no_attempts_there_is_no_reliability_and_no_latency():
    s = ProviderStats().snapshot(T0)

    assert s["reliability"] is None and s["reliability_sample"] == 0 and s["reliability_answered"] == 0
    assert s["avg_latency_ms"] is None and s["last_latency_ms"] is None
    assert s["today"]["requests"] == 0 and s["cost"] is None


def test_a_small_sample_is_reported_as_exactly_that_sample():
    stats = ProviderStats()
    stats.record(True, 120, T0)
    stats.record(True, 80, T0)

    s = stats.snapshot(T0)

    assert s["reliability"] == 1.0 and s["reliability_sample"] == 2 and s["reliability_answered"] == 2
    assert s["recent_window"] == RECENT_WINDOW  # the sample is 2, not 50: the page says "2/2 answered (last 2 requests)"


def test_the_dashboard_never_shows_a_bare_percentage():
    script = (main.__file__.replace("main.py", "dashboard/static/app.js"))
    text = open(script, encoding="utf-8").read()

    assert "reliability_answered}/${stats.reliability_sample} answered" in text
    assert "No requests yet - nothing measured." in text
    assert "stats.reliability *" not in text and "reliability * 100" not in text


# --- counting --------------------------------------------------------------------------------------------------------------


def test_successes_and_failures_are_counted_by_kind_and_latency_is_smoothed_over_answers_only():
    stats = ProviderStats()
    stats.record(True, 400, T0, tokens=50)
    stats.record(True, 200, T0, tokens=30)
    stats.record(False, 30, T0, kind="rate_limit")
    stats.record(False, 30000, T0, kind="network")

    s = stats.snapshot(T0)

    assert (s["attempts"], s["successes"], s["failures"], s["rate_limited"]) == (4, 2, 2, 1)
    assert s["failures_by_kind"] == {"network": 1, "rate_limit": 1}
    assert s["avg_latency_ms"] == round(0.3 * 200 + 0.7 * 400)  # the 30 s timeout does not move the average
    assert s["last_latency_ms"] == 30000 and s["last_failure_kind"] == "network"
    assert s["reliability"] == 0.5 and s["reliability_sample"] == 4
    assert s["tokens_total"] == 80 and s["today"] == {"date": "2026-11-18", "requests": 4, "failures": 2, "rate_limited": 1,
                                                      "tokens": 80, "request_limit": None, "requests_left": None}


def test_reliability_covers_only_the_last_50_actual_attempts():
    stats = ProviderStats()
    for _ in range(RECENT_WINDOW):
        stats.record(False, 10, T0, kind="network")
    for _ in range(RECENT_WINDOW):
        stats.record(True, 10, T0)

    s = stats.snapshot(T0)
    assert s["reliability"] == 1.0 and s["reliability_sample"] == 50 and s["attempts"] == 100 and s["failures"] == 50


def test_today_is_the_local_date_and_starts_again_each_day():
    stats = ProviderStats()
    stats.record(True, 10, T0)
    stats.record(True, 10, T0)

    assert stats.snapshot(T0 + timedelta(days=1))["today"]["requests"] == 0
    stats.record(True, 10, T0 + timedelta(days=1))
    assert stats.snapshot(T0 + timedelta(days=1))["today"]["requests"] == 1
    assert stats.snapshot(T0)["today"]["requests"] == 2  # yesterday's count is kept, per day


def test_only_31_days_of_daily_counts_are_kept():
    stats = ProviderStats()
    for day in range(40):
        stats.record(True, 10, T0 + timedelta(days=day))

    assert len(stats.days) == 31 and min(stats.days) == (T0 + timedelta(days=9)).date().isoformat()


def test_a_configured_daily_limit_shows_what_is_left_and_blocks_nothing():
    stats = ProviderStats()
    for _ in range(3):
        stats.record(True, 10, T0)

    today = stats.snapshot(T0, daily_request_limit=5)["today"]
    assert today["request_limit"] == 5 and today["requests_left"] == 2
    assert stats.snapshot(T0, daily_request_limit=2)["today"]["requests_left"] == 0
    assert stats.snapshot(T0, daily_request_limit=0)["today"]["request_limit"] is None


@pytest.mark.parametrize("text, expected", [
    ("groq=1000, gemini=250", {"groq": 1000, "gemini": 250}),
    ("GROQ = 10", {"groq": 10}),
    ("groq=-1,gemini=abc,=5,openrouter", {}),
    ("", {}),
])
def test_daily_limits_are_parsed_leniently(text, expected):
    assert parse_daily_limits(text) == expected


def test_odd_values_are_never_recorded_as_something_impossible():
    stats = ProviderStats()
    stats.record(True, -5, T0, tokens=-3)
    stats.record(False, 5, T0, kind="made-up kind")

    s = stats.snapshot(T0)
    assert s["last_latency_ms"] == 5 and s["avg_latency_ms"] == 0 and s["tokens_total"] == 0
    assert s["failures_by_kind"] == {"other": 1}


# --- recorded by the manager, for real attempts only ------------------------------------------------------------------------


def test_every_real_attempt_is_recorded_with_its_latency_kind_and_tokens():
    m = chain(Scripted(ProviderError("Groq returned 401: invalid api key")), Scripted("fine", tokens=42), timer=Timer(0.05, 0.8))

    assert m.generate_reply("hi") == "fine"

    groq, openrouter = snap(m, "groq"), snap(m, "openrouter")
    assert groq["failures_by_kind"] == {"auth": 1} and groq["last_latency_ms"] == 50 and groq["avg_latency_ms"] is None
    assert openrouter["successes"] == 1 and openrouter["avg_latency_ms"] == 800 and openrouter["tokens_total"] == 42
    assert openrouter["last_success_at"] == T0.isoformat() and openrouter["reliability_sample"] == 1


def test_a_retried_5xx_is_two_attempts():
    m = chain(Scripted(ProviderError("Groq returned 503: busy"), "fine"), retry_count=1)

    m.generate_reply("hi")

    s = snap(m, "groq")
    assert (s["attempts"], s["failures"], s["successes"]) == (2, 1, 1) and s["failures_by_kind"] == {"transient": 1}


def test_a_rate_limit_is_counted_and_a_resting_provider_is_neither_asked_nor_recorded():
    clock = {"now": 0.0}
    m = chain(Scripted(ProviderError("Groq returned 429: Please try again in 2m")), Scripted("a", "b"), clock=lambda: clock["now"])

    m.generate_reply("1")
    m.generate_reply("2")  # groq rests: skipped

    groq = snap(m, "groq")
    assert groq["attempts"] == 1 and groq["rate_limited"] == 1 and groq["today"]["rate_limited"] == 1
    assert snap(m, "openrouter")["today"]["requests"] == 2


def test_a_disabled_provider_is_not_recorded():
    m = chain(Scripted(), Scripted("fine"))
    m.set_enabled("groq", False)

    m.generate_reply("hi")

    assert snap(m, "groq")["attempts"] == 0


def test_an_adapter_bug_is_recorded_as_unexpected():
    m = chain(Scripted(RuntimeError("boom")), Scripted("fine"))

    m.generate_reply("hi")

    assert snap(m, "groq")["failures_by_kind"] == {"unexpected": 1}


def test_a_reply_that_fails_validation_is_still_an_answered_attempt():
    m = chain(Scripted("not json"), Scripted('{"ok": 1}'))

    def parse(reply):
        if not reply.startswith("{"):
            raise ValueError("bad")
        return reply

    call_structured(m, "hi", parse)

    assert snap(m, "groq")["successes"] == 1  # the provider answered; the caller rejected the content


def test_when_every_provider_fails_each_failure_is_recorded_and_the_order_is_unchanged():
    m = chain(*(Scripted(ProviderError(f"Failed to reach {n}: down")) for n in FIXED_ORDER))

    with pytest.raises(AllProvidersFailedError):
        m.generate_reply("hi")

    assert [h.name for h in m.get_health()] == FIXED_ORDER
    assert [snap(m, n)["failures_by_kind"] for n in FIXED_ORDER] == [{"network": 1}] * 5


def test_tokens_from_one_attempt_never_leak_into_the_next():
    m = chain(Scripted("a", tokens=10), Scripted())
    m.generate_reply("1")
    usage.note(total_tokens=999)  # something noted outside any attempt on this thread

    m.entries[0].provider.tokens = None
    m.generate_reply("2")

    assert snap(m, "groq")["tokens_total"] == 10


def test_a_statistics_failure_never_changes_the_answer():
    class BrokenStore:
        def load_all(self):
            return {}

        def save(self, provider, state):
            raise sqlite3.OperationalError("disk I/O error")

    m = chain(Scripted("fine"), store=BrokenStore())

    assert m.generate_reply("hi") == "fine"
    assert m.generate_reply("again") == "fine"
    assert snap(m, "groq")["successes"] == 2  # still counted in memory


# --- persistence: statistics survive a restart --------------------------------------------------------------------------------


def test_statistics_survive_a_restart(tmp_path):
    path = str(tmp_path / "jarvis_provider_stats.db")
    first = chain(Scripted(ProviderError("Groq returned 429: slow down"), "ok", tokens=7), Scripted("ok"),
                  store=SQLiteProviderStatsStore(path), timer=Timer(0.2, 0.3, 0.4))
    first.generate_reply("1")  # groq answers 429 and rests (60 s) -> openrouter answers
    first.generate_reply("2")  # groq still resting: openrouter answers again

    before = {name: snap(first, name) for name in FIXED_ORDER[:2]}
    restarted = chain(Scripted(), Scripted(), store=SQLiteProviderStatsStore(path))

    after = {name: snap(restarted, name) for name in FIXED_ORDER[:2]}
    assert after == before
    assert after["groq"]["attempts"] == 1 and after["groq"]["rate_limited"] == 1
    assert after["openrouter"]["attempts"] == 2 and after["openrouter"]["reliability_sample"] == 2


def test_the_recent_window_and_daily_counts_are_restored_exactly(tmp_path):
    path = str(tmp_path / "stats.db")
    store = SQLiteProviderStatsStore(path)
    stats = ProviderStats()
    for i in range(60):
        stats.record(i % 3 != 0, 100 + i, T0 + timedelta(minutes=i), kind=None if i % 3 else "network", tokens=i)
    store.save("groq", stats.to_state())

    restored = load_stats(SQLiteProviderStatsStore(path), ["groq", "gemini"])

    assert restored["groq"].snapshot(T0) == stats.snapshot(T0)
    assert len(restored["groq"].recent) == 50 and restored["gemini"].attempts == 0


def test_the_stored_rows_hold_numbers_and_timestamps_never_text(tmp_path):
    path = tmp_path / "stats.db"
    secret_prompt = "my bank PIN is 4321 please remember"
    secret_error = "Groq returned 401: Invalid API key gsk_SECRETSECRETSECRET for user bassim"
    m = chain(Scripted(ProviderError(secret_error)), Scripted("the reply text that must not be stored", tokens=9),
              store=SQLiteProviderStatsStore(str(path)))

    m.generate_reply(secret_prompt)

    rows = sqlite3.connect(path).execute("SELECT provider, state FROM provider_stats").fetchall()
    blob = json.dumps(rows) + path.read_bytes().decode("latin-1")
    for forbidden in ("4321", "PIN", "gsk_", "SECRET", "Invalid API key", "bassim", "reply text"):
        assert forbidden not in blob, forbidden
    assert {row[0] for row in rows} == {"groq", "openrouter"}


@pytest.mark.parametrize("bad_state", [
    "not json at all",
    json.dumps([1, 2, 3]),
    json.dumps({"version": 99}),
    json.dumps({**ProviderStats().to_state(), "attempts": -1}),
    json.dumps({**ProviderStats().to_state(), "attempts": 5}),  # 0 successes + 0 failures != 5
    json.dumps({**ProviderStats().to_state(), "failures_by_kind": {"made_up": 1}}),
    json.dumps({**ProviderStats().to_state(), "attempts": 1, "successes": 1, "recent": [{"at": "x", "ok": "yes", "latency_ms": 1}]}),
    json.dumps({**ProviderStats().to_state(), "days": {"not-a-date": {"requests": 1, "failures": 0, "rate_limited": 0, "tokens": 0}}}),
    json.dumps({**ProviderStats().to_state(), "avg_latency_ms": "fast"}),
])
def test_a_stored_state_that_cannot_be_trusted_starts_again_from_empty(tmp_path, bad_state):
    path = str(tmp_path / "stats.db")
    SQLiteProviderStatsStore(path)  # creates the table
    connection = sqlite3.connect(path)
    connection.execute("INSERT INTO provider_stats VALUES ('groq', 1, ?, datetime('now'))", (bad_state,))
    connection.commit()
    connection.close()

    m = chain(Scripted("fine"), store=SQLiteProviderStatsStore(path))

    assert snap(m, "groq")["attempts"] == 0  # not a made-up number
    m.generate_reply("hi")
    assert snap(m, "groq")["attempts"] == 1  # and recording carries on, replacing the bad row
    assert load_stats(SQLiteProviderStatsStore(path), ["groq"])["groq"].attempts == 1


def test_from_state_rejects_inconsistent_counts():
    state = ProviderStats().to_state()
    state.update({"attempts": 2, "successes": 1, "failures": 1, "failures_by_kind": {"network": 2}})

    with pytest.raises(InvalidStatsState):
        ProviderStats.from_state(state)


def test_an_unopenable_store_does_not_stop_the_manager():
    class Unreadable:
        def load_all(self):
            raise sqlite3.DatabaseError("file is not a database")

        def save(self, provider, state):
            pass

    m = chain(Scripted("fine"), store=Unreadable())

    assert m.generate_reply("hi") == "fine" and snap(m, "groq")["attempts"] == 1


# --- concurrency -------------------------------------------------------------------------------------------------------------


def test_concurrent_requests_never_lose_a_count(tmp_path):
    store = SQLiteProviderStatsStore(str(tmp_path / "stats.db"))
    m = AIProviderManager([AIProviderEntry("groq", Scripted(*(["fine"] * 400), tokens=1))], stats_store=store,
                          wall_clock=lambda: T0, sleep=lambda s: None)
    errors = []

    def worker():
        try:
            for _ in range(50):
                m.generate_reply("hi")
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    s = snap(m, "groq")
    assert errors == [] and s["attempts"] == 400 and s["successes"] == 400 and s["tokens_total"] == 400
    assert s["today"]["requests"] == 400 and s["reliability_sample"] == 50
    assert load_stats(SQLiteProviderStatsStore(str(tmp_path / "stats.db")), ["groq"])["groq"].attempts == 400


def test_concurrent_recording_keeps_the_counts_consistent():
    stats_manager = chain(Scripted())
    entry = stats_manager.entries[0]

    def worker(ok):
        for _ in range(200):
            stats_manager._record(entry, ok, stats_manager._timer(), None if ok else "network", 1)

    threads = [threading.Thread(target=worker, args=(i % 2 == 0,)) for i in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    s = snap(stats_manager, "groq")
    assert s["attempts"] == 1200 and s["successes"] == 600 and s["failures"] == 600 and s["tokens_total"] == 1200
    assert ProviderStats.from_state(entry.health.stats.to_state()).attempts == 1200  # still internally consistent


# --- token counts from the real adapters (fake HTTP) --------------------------------------------------------------------------


def test_the_openai_compatible_adapter_hands_over_its_usage(monkeypatch):
    from app.providers.groq import GroqProvider

    monkeypatch.setattr(_openai_compatible.settings, "groq_api_key", "test-key")

    def fake_post(url, **kwargs):
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}],
                                         "usage": {"prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17}},
                              request=httpx.Request("POST", url))

    monkeypatch.setattr(_openai_compatible.httpx, "post", fake_post)
    m = AIProviderManager([AIProviderEntry("groq", GroqProvider())], wall_clock=lambda: T0)

    assert m.generate_reply("hi") == "ok"
    assert snap(m, "groq")["tokens_total"] == 17


def test_the_gemini_adapter_hands_over_its_usage(monkeypatch):
    from app.providers.gemini import GeminiProvider

    monkeypatch.setattr(gemini.settings, "gemini_api_key", "test-key")

    def fake_post(url, **kwargs):
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": "ok"}]}}],
                                         "usageMetadata": {"promptTokenCount": 8, "candidatesTokenCount": 3}},
                              request=httpx.Request("POST", url))

    monkeypatch.setattr(gemini.httpx, "post", fake_post)
    m = AIProviderManager([AIProviderEntry("gemini", GeminiProvider())], wall_clock=lambda: T0)

    m.generate_reply("hi")
    assert snap(m, "gemini")["tokens_total"] == 11


def test_a_response_without_usage_records_no_tokens(monkeypatch):
    from app.providers.groq import GroqProvider

    monkeypatch.setattr(_openai_compatible.settings, "groq_api_key", "test-key")
    monkeypatch.setattr(_openai_compatible.httpx, "post", lambda url, **kw: httpx.Response(
        200, json={"choices": [{"message": {"content": "ok"}}]}, request=httpx.Request("POST", url)))
    m = AIProviderManager([AIProviderEntry("groq", GroqProvider())], wall_clock=lambda: T0)

    m.generate_reply("hi")
    assert snap(m, "groq")["tokens_total"] == 0 and snap(m, "groq")["successes"] == 1


# --- shown by /providers/status and the dashboard --------------------------------------------------------------------------


@pytest.fixture
def served(monkeypatch):
    m = chain(Scripted("fine", tokens=3), Scripted(), Scripted(), Scripted(), Scripted(), timer=Timer(0.25),
              wall=lambda: datetime.now().astimezone(), daily_request_limits={"groq": 100})
    monkeypatch.setattr(main.core, "provider", m)
    m.generate_reply("hi")
    return TestClient(main.app, base_url="http://127.0.0.1:8000"), m


def test_the_status_endpoint_shows_each_providers_statistics_in_the_fixed_order(served):
    http, _ = served

    body = http.get("/providers/status").json()
    providers = {p["name"]: p for p in body["providers"]}

    assert body["order"] == FIXED_ORDER
    groq = providers["groq"]["stats"]
    assert groq["successes"] == 1 and groq["avg_latency_ms"] == 250 and groq["reliability_sample"] == 1
    assert groq["today"]["requests"] == 1 and groq["today"]["request_limit"] == 100 and groq["today"]["requests_left"] == 99
    assert groq["tokens_total"] == 3
    assert providers["openrouter"]["stats"]["reliability"] is None and providers["openrouter"]["stats"]["reliability_sample"] == 0


def test_the_dashboard_overview_carries_the_statistics(served):
    http, _ = served

    groq = http.get("/api/overview").json()["providers"][0]

    assert groq["name"] == "groq" and groq["stats"]["reliability_answered"] == 1 and groq["stats"]["reliability_sample"] == 1


def test_the_real_application_keeps_its_statistics_in_the_data_folder_setting():
    from app.config import DATA_FILE_SETTINGS, Settings

    assert "provider_stats_db_path" in DATA_FILE_SETTINGS
    assert Settings.model_fields["provider_stats_db_path"].default == "jarvis_provider_stats.db"
    placed = Settings(data_dir="some_folder", _env_file=None)
    assert placed.provider_stats_db_path.replace("\\", "/").endswith("some_folder/jarvis_provider_stats.db")
