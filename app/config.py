"""
Central configuration for Jarvis.

All settings are read from environment variables (or a local .env file).
Nothing sensitive is hard-coded here — see .env.example for the list of
variables Jarvis understands.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Basic app info
    app_name: str = "Jarvis"
    environment: str = "development"  # development | production

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    # Gemini (AI provider)
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.8-flash"

    # Additional AI providers (M11 foundation - see app/providers/factory.py
    # for the registry these are selected through). Each is independently
    # optional: leaving a provider's API key blank just means that
    # provider raises a clear config error if it's ever actually invoked -
    # it never affects importing, testing, or starting Jarvis. FreeLLMAPI
    # (below, after Cloudflare) extends this same set as the final
    # fallback - it's a self-hosted gateway rather than a cloud vendor, so
    # its base URL is user-configurable instead of fixed like the others.
    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-120b"

    openrouter_api_key: str = ""
    openrouter_model: str = "openrouter/auto"

    # Cloudflare Workers AI needs both a token and an account id, since its
    # API URL is per-account (.../accounts/{account_id}/ai/run/{model}).
    cloudflare_api_key: str = ""
    cloudflare_account_id: str = ""
    cloudflare_model: str = "@cf/meta/llama-3.1-8b-instruct"

    # FreeLLMAPI - a self-hosted, OpenAI-compatible AI gateway
    # (https://github.com/tashfeenahmed/freellmapi) the user runs locally
    # (e.g. via Docker Compose); it performs its OWN provider routing/
    # fallback across whatever free-tier backends it's configured with, so
    # Jarvis always requests model="auto" (see app/providers/freellmapi.py)
    # and never reproduces that routing itself - deliberately no
    # JARVIS_FREELLMAPI_MODEL setting. The key is FreeLLMAPI's own
    # "unified" key (its Dashboard -> Keys page) - never one of the
    # individual provider keys FreeLLMAPI may hold internally (e.g. its
    # own Gemini key), which Jarvis never sees. The base URL is genuinely
    # configurable, unlike the fixed vendor URLs above, since this is a
    # locally-hosted address the user controls.
    freellmapi_api_key: str = ""
    freellmapi_base_url: str = "http://localhost:3001/v1"

    # AI provider manager (M12 - see app/providers/manager.py and
    # app/providers/factory.py:create_default_ai_provider_manager) - ordered
    # fallback with retry/cooldown across the providers above. Local AI is
    # planned as a future addition to this same order, not built yet.
    # Ordered, comma-separated list of provider names to try, with automatic
    # fallback to the next on failure; a name left out of this list is
    # effectively disabled. Same convention as JARVIS_SEARCH_PROVIDER_ORDER.
    # Deliberately still the original four - FreeLLMAPI is NOT listed here.
    # factory.py:build_ai_provider_entries() derives the actual effective
    # order from this value, automatically appending "freellmapi" as the
    # final entry whenever JARVIS_FREELLMAPI_API_KEY is configured - so
    # this setting (and any pre-existing JARVIS_AI_PROVIDER_ORDER override
    # in .env) never needs to be touched to enable FreeLLMAPI.
    ai_provider_order: str = "gemini,groq,openrouter,cloudflare"
    # Extra attempts against the SAME provider for a transient (5xx) error
    # before falling back to the next provider - 2 means up to 3 attempts
    # total. Rate-limit/network/auth errors never retry the same provider.
    ai_provider_retry_count: int = 2
    ai_provider_retry_delay_seconds: float = 0.5
    # How long a provider that returned a rate-limit (429) error is skipped
    # for, in seconds, before the manager will try it again.
    ai_provider_cooldown_seconds: float = 60.0

    # Memory (local SQLite persistence for conversations and user facts)
    memory_db_path: str = "jarvis_memory.db"

    # Identity context (small, local, human-editable Markdown files - see
    # app/identity.py) - Jarvis's own personality and what it knows about
    # the user, loaded once as static prompt context for the DIRECT path.
    # Plain relative paths, same convention as every other *_db_path
    # setting above; defaults point at the repo root, next to README.md.
    personality_file_path: str = "personality.md"
    user_profile_file_path: str = "user_profile.md"

    # SearXNG (web search provider) - base URL of a running SearXNG instance
    searxng_url: str = ""

    # Ordered, comma-separated list of search provider names to try, with
    # automatic fallback to the next on failure (see app/search/manager.py
    # and app/search/factory.py). Only "searxng" is implemented so far.
    search_provider_order: str = "searxng"

    # Search result cache (SQLite-backed, see app/search/sqlite_cache.py) -
    # how long a cached web_search result stays valid, in seconds, and
    # where the cache file lives. Expired entries are treated as misses.
    search_cache_ttl: int = 600
    search_cache_db_path: str = "jarvis_search_cache.db"

    # Reminders (persistent tasks/reminders + background scheduler,
    # see app/reminders/) - local SQLite file, and how often the
    # background scheduler polls for due reminders, in seconds.
    reminders_db_path: str = "jarvis_reminders.db"
    reminder_poll_interval_seconds: float = 30.0

    # Automations (persistent automation engine, see app/automation/) -
    # local SQLite file, and how often the background engine polls for
    # due automations, in seconds.
    automations_db_path: str = "jarvis_automations.db"
    automation_poll_interval_seconds: float = 30.0

    # Scheduling foundation (see app/scheduling/): the IANA time zone NEW schedules are created in (JARVIS_TIMEZONE, e.g.
    # "Europe/London"). Empty -> the system's zone is used only if it can be determined reliably; if not, creating a
    # time-based schedule fails with a clear message instead of guessing. An existing schedule keeps the zone it was
    # created in. Also: how late a due run may be and still simply run (the grace window), and the shortest gap between
    # runs of a NEW repeating schedule. None of these change any existing reminder or automation.
    timezone: str = ""
    schedule_grace_seconds: int = 600
    schedule_min_interval_minutes: int = 15

    # Scheduled Jobs (see app/scheduled_jobs/): a Job created from a Schedule, once per occurrence, while Jarvis is running.
    # `scheduled_jobs_enabled` is the kill switch - OFF by default, and nothing in this foundation consults it yet. The rest are
    # the V1 limits: the shortest gap between runs of a NEW repeating Scheduled Job, how long a recurring one stays active after
    # the user approves it (renewal needs fresh approval), how late a missed occurrence may still be caught up (ONCE, never a
    # replay), how many may exist / run per day / run at the same time, and how many failed runs in a row pause one.
    scheduled_jobs_enabled: bool = False
    scheduled_job_min_interval_minutes: int = 60
    scheduled_job_lifetime_days: int = 90
    scheduled_job_catch_up_window_seconds: int = 21600
    scheduled_job_max_active: int = 10
    scheduled_job_max_runs_per_day: int = 10
    scheduled_job_max_concurrent: int = 2
    scheduled_job_breaker_failures: int = 3

    # Google Calendar (OAuth 2.0 desktop/localhost flow, see
    # app/calendar/) - client_id/client_secret come from a Google Cloud
    # "Desktop app" OAuth client; no default and no key is hard-coded
    # anywhere. A single primary calendar only (see app/calendar/base.py).
    google_client_id: str = ""
    google_client_secret: str = ""
    google_oauth_redirect_port: int = 8765
    calendar_id: str = "primary"
    calendar_token_db_path: str = "jarvis_calendar_token.db"

    # Jarvis Tasks (local-only todo items, see app/tasks/) - independent
    # of Reminders; no scheduling, no notifications, no external sync.
    tasks_db_path: str = "jarvis_tasks.db"

    # Notes (findings Jarvis saves for the user, see app/notes/) - their own
    # local SQLite file, deliberately separate from memory/facts and the
    # World Model.
    notes_db_path: str = "jarvis_notes.db"

    # Orchestration (M15 - goal-driven Jobs: plan -> validate -> execute
    # through the existing tool system, see app/orchestration/) - local
    # SQLite file, how often the background JobEngine polls active Jobs,
    # and the safety limits Plan Validator/JobEngine enforce so a plan or
    # a chain of replans can never grow or loop without bound. A Job's
    # default autonomy (see app/orchestration/base.py:AutonomyLevel) when
    # a caller doesn't specify one - ASSISTED per this milestone's own
    # default.
    orchestration_db_path: str = "jarvis_orchestration.db"
    orchestration_poll_interval_seconds: float = 15.0
    orchestration_max_steps: int = 20
    orchestration_max_total_steps: int = 40
    orchestration_max_retries_per_step: int = 2
    orchestration_retry_delay_seconds: float = 1.0
    orchestration_max_replans: int = 2
    orchestration_default_autonomy: str = "assisted"
    # How long an interaction (a question or confirmation waiting on the
    # user, see app/orchestration/interactions.py) stays open: 24 hours for a
    # CONFIRMATION, 7 days for an INPUT. Seconds; configurable per
    # deployment via JARVIS_ORCHESTRATION_*_TTL_SECONDS.
    orchestration_confirmation_ttl_seconds: float = 86400.0
    orchestration_input_ttl_seconds: float = 604800.0
    # How recently an interaction must have been SURFACED to a conversation
    # for a bare yes/no or a free-text answer there to resolve it (default 30
    # minutes). Older ones need the explicit `answer <code>: ...` form.
    orchestration_interaction_surface_window_seconds: float = 1800.0
    # Show a desktop notification when a Job finishes (completed or failed).
    # Optional and best-effort: it never affects the Job (see JobEngine).
    orchestration_notify_on_completion: bool = True

    # Notifications about waiting interactions (see app/notifications/ and
    # app/orchestration/interaction_lifecycle.py). The first notification is
    # immediate; reminders come `..._reminder_offsets_seconds` after the
    # interaction was opened (comma-separated), then ONE final warning
    # `..._final_warning_lead_seconds` before it expires, then a single
    # expiry notice. Defaults: a confirmation (24h) at once, +2h, +6h, a
    # warning ~20h in, expiry; an input (7d) at once, +1d, +3d, a warning
    # ~6d in, expiry. `notifications_max_reminders_per_interaction` is a hard
    # cap on reminders + the final warning, so nobody is notified forever.
    orchestration_confirmation_reminder_offsets_seconds: str = "7200,21600"
    orchestration_confirmation_final_warning_lead_seconds: float = 14400.0
    orchestration_input_reminder_offsets_seconds: str = "86400,259200"
    orchestration_input_final_warning_lead_seconds: float = 86400.0
    notifications_max_reminders_per_interaction: int = 4
    # Desktop toast channel: on/off, and the lowest priority it will show
    # ("low", "normal" or "high"). Failures are HIGH, waiting/reminders NORMAL,
    # a background Job finishing LOW.
    notifications_desktop_enabled: bool = True
    notifications_desktop_min_priority: str = "low"

    # Attention policy (app/notifications/attention*.py): when may Jarvis interrupt?
    attention_quiet_hours: str = ""  # "22:00-07:00"; empty = none
    attention_suppressing_contexts: str = "class,studying,meeting,sleeping,focused_work"
    attention_urgent_window_minutes: float = 15.0
    attention_enabled_channels: str = "desktop"
    attention_urgent_channels: str = "desktop"  # earbuds / watch / glasses are never urgent channels unless listed

    # Morning Brief (app/morning_brief/): an on-demand, read-only summary of today. News is off by
    # default (it needs the network and shows third-party headlines). The deterministic brief is always
    # built first and is complete; the existing AI provider then rewords it, and any failure or
    # validation problem falls back to the deterministic text (set false to never call a model).
    morning_brief_news_enabled: bool = False
    morning_brief_news_query: str = "top news headlines today"
    morning_brief_use_llm: bool = True

    # Voice (hands-free wake-word + STT + TTS, see app/voice/) - $0/local
    # only: openWakeWord for wake-word detection, Vosk for STT, Windows
    # SAPI for TTS. Model paths have no default - they point at files the
    # user downloads themselves (see .env.example / README's Voice
    # Interface section); voice_chat_url is the existing /chat endpoint,
    # never Core directly.
    voice_chat_url: str = "http://127.0.0.1:8000"
    voice_chat_timeout_seconds: float = 60.0
    # The PC's device credential file (written by `python -m app.devices.admin register-pc`; gitignored).
    # The voice client authenticates with it and talks to /device/chat (see app/voice/chat_client.py).
    voice_device_credential_file: str = "jarvis_pc_device.json"
    voice_wakeword_model_path: str = ""
    voice_wakeword_threshold: float = 0.5
    voice_vosk_model_path: str = ""
    voice_input_device: str = ""  # "" = system default input device
    voice_tts_voice_name: str = ""  # "" = system default SAPI voice
    voice_tts_rate: int = 0  # SAPI rate, -10 (slowest) to 10 (fastest)
    voice_silence_seconds: float = 1.2  # pause length that ends an utterance
    voice_max_utterance_seconds: float = 15.0  # hard cap on one utterance
    voice_speech_amplitude_threshold: int = 500  # int16 mean-abs "is this speech" cutoff

    # Self-Development Engine (V1, see app/self_development/) - lets
    # Jarvis propose and, only with explicit approval at each gate, apply
    # bounded changes to its own codebase. `self_dev_project_root` is the
    # ONE directory this whole subsystem is ever allowed to touch (see
    # app/self_development/safety.py) - configurable here rather than
    # hardcoded, but every path operation still re-validates against it at
    # runtime regardless of what this setting says.
    self_dev_project_root: str = (
        r"C:\Users\bassim.ms\Documents\claude-access\jarvis-foundation\jarvis"
    )
    # A small, fixed ceiling on bounded self-repair iterations after a
    # failed test/check - never unlimited (V1's own explicit requirement).
    self_dev_max_repair_iterations: int = 2
    # Bounds every single command this engine ever runs (git/pytest/
    # python -m ...) - "no infinite processes" (V1's own requirement).
    self_dev_command_timeout_seconds: float = 120.0
    # The fixed, non-AI-supplied command TEST_RUN always uses - never an
    # arbitrary command (see app/self_development/command_runner.py:
    # RunTestsTool). Tokenized with shlex.split, same as COMMAND_RUN.
    self_dev_test_command: str = "pytest -q"

    model_config = SettingsConfigDict(env_file=".env", env_prefix="JARVIS_")


settings = Settings()
