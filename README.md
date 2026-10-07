# Jarvis

Personal AI assistant project. This is **Milestone 11: Multi-Provider AI
Foundation** — `app/providers/` now has three more `AIProvider`
implementations alongside Gemini (Groq, OpenRouter, Cloudflare Workers
AI - deliberately exactly these four cloud providers for now) and a
small registry (`app/providers/factory.py`) to construct one by stable
name. This milestone is **foundation only**: `JarvisCore` still builds
its own default provider directly (`GeminiProvider()`, unchanged) and
has no notion of provider selection, fallback, retries, quota tracking,
or health-based rotation - that's the next milestone. See
[Multi-Provider AI Foundation](#multi-provider-ai-foundation-m11) for
the full provider table (base URL/required env/model config per
provider) and what's still deliberately missing. `SearchRouter`,
`CapabilityRouter`, `ToolCallBuilder`, Core's DIRECT/SEARCH/TOOL
dispatch, Hands-Free Voice + Wake Word (M10), Google Calendar + Local
Jarvis Tasks (M9), Persistent Automation Engine (M8), Persistent Tasks,
Reminders & Background Scheduling (M7), natural-language tool selection
(M6.16), capability routing (M6.13-M6.15), and the search pipeline
(M6.5-6.12) are all unchanged. No provider fallback/rotation, no quota
tracking, no agents/sub-agents, no new capabilities, and no API key is
required for any of the three new providers yet.

## Stack

- **Python 3.11+** with **FastAPI** (web framework) and **uvicorn** (server
  that runs it). Both are lightweight and run comfortably on a 4 GB RAM
  machine.
- **pydantic-settings** for configuration — reads settings from environment
  variables / a `.env` file, so no secrets are ever hard-coded.
- **httpx** talks directly to the Gemini REST API and to a SearXNG
  instance's JSON search API — no extra AI or search SDK needed for a
  single HTTP call.
- Python's stdlib **sqlite3** backs both conversation memory and the search
  result cache — no separate cache service (e.g. Redis) is used.
- **pytest** for tests.

## Setup

Run these from inside the `jarvis` folder.

1. Create a virtual environment (keeps Jarvis's packages separate from the
   rest of your system):
   ```
   python3 -m venv venv
   source venv/bin/activate
   ```
   On Windows: `venv\Scripts\activate`

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Copy the example environment file:
   ```
   cp .env.example .env
   ```
   Then open `.env` and set `JARVIS_GEMINI_API_KEY` to your key from
   https://aistudio.google.com/apikey (`.env` is git-ignored and never
   committed). To use the `web_search` tool, also set `JARVIS_SEARXNG_URL`
   to the base URL of a SearXNG instance you run or trust, and optionally
   `JARVIS_SEARCH_PROVIDER_ORDER`, `JARVIS_SEARCH_CACHE_TTL`, and
   `JARVIS_SEARCH_CACHE_DB_PATH` (see [Web Search](#web-search) below).
   Reminders, automations, and Jarvis Tasks need no setup at all - they
   work with the defaults in `.env.example` (see
   [Persistent Tasks, Reminders & Background Scheduling](#persistent-tasks-reminders--background-scheduling-m7),
   [Persistent Automation Engine](#persistent-automation-engine-m8), and
   [Jarvis Tasks](#jarvis-tasks-m9) below). To use the calendar tools,
   set `JARVIS_GOOGLE_CLIENT_ID`/`JARVIS_GOOGLE_CLIENT_SECRET` (from a
   Google Cloud "Desktop app" OAuth client) and run
   `python -m app.calendar.authorize` once (see
   [Google Calendar](#google-calendar-m9) below) - without that, the
   calendar tools just fail gracefully with a clear error. Voice
   (`python -m app.voice.run`) needs a few things downloaded/configured
   first - see [Hands-Free Voice + Wake Word](#hands-free-voice--wake-word-m10)'s
   own Setup section below for the exact TODOs.

## Run the server

```
uvicorn app.main:app --reload
```

Then open http://127.0.0.1:8000/health in a browser, or run:
```
curl http://127.0.0.1:8000/health
```

Expected output:
```json
{"status": "ok", "app": "Jarvis", "environment": "development"}
```

Try the chat endpoint (requires `JARVIS_GEMINI_API_KEY` to be set):
```
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d "{\"message\": \"Say hello in one sentence.\"}"
```

Expected output (actual text will vary):
```json
{"reply": "Hello there!"}
```

## Run the tests

```
pytest
```

Expected output ends with something like:
```
707 passed in 0.XXs
```

Tests never call the real Gemini or SearXNG APIs — `httpx.post` / `httpx.get`
are monkeypatched, so they run without a network connection or real
credentials. Fallback behavior is proven with lightweight fake
`SearchProvider`s, not real network calls to multiple backends. The cache
is tested against real SQLite (`:memory:` or a temp file), not mocked.

## Folder structure

```
jarvis/
  app/
    __init__.py       - marks app/ as a Python package
    main.py            - the FastAPI app, /health and /chat endpoints
    config.py          - all configuration, read from environment variables
    core.py            - Jarvis Core: dispatches a chat request DIRECT (AIProvider) or SEARCH (WebSearchTool + SearchAnswerGenerator)
    capability.py       - Capability/CapabilityDecision/CapabilityRouter: the DIRECT/SEARCH decision layer Core dispatches on
    providers/
      base.py           - AIProvider interface + ProviderError, so Core doesn't depend on any vendor; ProviderInfo (metadata type for describe_providers)
      gemini.py         - Gemini implementation of AIProvider (Core's default, unchanged since M11)
      _openai_compatible.py - OpenAICompatibleProvider: shared /chat/completions HTTP+JSON handling for Groq/OpenRouter
      groq.py            - Groq implementation of AIProvider (M11)
      openrouter.py       - OpenRouter implementation of AIProvider (M11)
      cloudflare.py         - Cloudflare Workers AI implementation of AIProvider - own request/response shape (M11)
      factory.py              - PROVIDER_FACTORIES registry + create_provider/describe_providers by stable name (M11)
    memory/
      base.py           - MemoryStore interface and the Message type
      sqlite_store.py   - SQLite implementation of MemoryStore
    tools/
      base.py           - Tool interface, PermissionLevel, ToolResult, ToolRequirements
      registry.py       - ToolRegistry: register/lookup/list/execute, permission enforcement
      decision.py         - ToolCall/ToolCallBuilder/GeminiToolCallProvider: natural-language tool selection
      web_search.py      - WebSearchTool: adapts a SearchProvider into the Tool interface
      defaults.py        - register_default_tools: wires the app's default tools (web_search, reminders, automations) into a registry
    reminders/
      base.py             - Reminder/ReminderStatus/Recurrence/ReminderStore interface, ReminderExecution, parse_recurrence/advance_by_recurrence (shared with automation)
      sqlite_store.py      - SQLiteReminderStore: SQLite implementation of ReminderStore
      notifier.py          - Notifier interface, WindowsToastNotifier (powershell.exe toast), ConsoleNotifier
      scheduler.py         - ReminderScheduler: background poller that triggers due reminders (V2 reminders: one catch-up for missed runs, via app/scheduling; pre-V2 reminders unchanged)
      service.py           - ReminderService (V2): the one place a structured request becomes a stored reminder; pause / resume / delete
      language.py          - (V2) pure grammar: clock rule + whole-message reminder phrasing (no model, no clock, no storage)
      operations.py        - (V2) ReminderOperations: the chat door ("remind me ...", "what reminders do I have?", "pause my ... reminder")
      occurrences.py       - (V2) which Schedule governs a stored reminder (its own, or one derived from a legacy recurrence)
      timefmt.py           - (V2) plain-words local times
      tools.py             - CreateReminderTool/ListRemindersTool/CompleteReminderTool/CancelReminderTool
      factory.py           - builds the default ReminderStore/Notifier/ReminderScheduler from config
    scheduling/
      schedule.py / runs.py / timezones.py - the pure Schedule value, run planning (`plan_run`), occurrence identity, time zones
      builder.py           - (Scheduled Jobs Slice 1) build_schedule: the request -> Schedule rules, moved out of ReminderService unchanged and shared
    scheduled_jobs/        - (Scheduled Jobs V1, Slice 1 - FOUNDATION ONLY; nothing runs, nothing is wired in)
      base.py              - ScheduledJob, the state machine (ALLOWED_TRANSITIONS), the occurrence ledger types, the ScheduledJobStore contract
      policy.py            - pure decisions: lifetime, new-schedule limits, missed-run (run-once catch-up), admission caps, circuit breaker
      sqlite_store.py      - SQLiteScheduledJobStore: the record + ledger; `record_run` writes the ledger row and the Job in ONE transaction
    scheduled_execution/   - (Scheduled Jobs V1, Slices 2-3 - the execution adapter, and the scheduler that decides what is due; hung on the JobEngine's tick)
      occurrence.py        - ScheduledOccurrence: one due occurrence, built from the Slice 1 policy's RUN / RUN_LATE decision
      context.py           - FreshContextBuilder: the run's context, read anew every time from the existing Context Engine + memory facts
      adapter.py           - ScheduledExecutionAdapter: occurrence -> ONE ordinary PENDING Job, atomically with its ledger row
    orchestration/profiles.py - (Slice 2) run profiles: `unattended_v1`, a READ-only REDUCTION of what a Job may use (grants nothing)
    action_proposal/       - (Productivity Action Loop, Slices 2-5) a temporary, in-memory plan the user can approve: honest 'Your part / My part' steps (My part: a reminder, or - if asked - saving the plan as a note); no Job until approval, no model; the verified one-time result report
    focus/                 - (Productivity Action Loop, Slice 1) read-only Focus: recognition / service (matching + deterministic ranking) / render; the last chat door, no state, no model
    scheduled_operations/  - (Scheduled Jobs V1, Slice 4 - creating, approving and managing them from chat; wired through main.py alone)
      language.py          - the fixed grammar: "every morning at 8, research X" / "what scheduled jobs do I have?" (no model; timing words are the reminders')
      service.py           - ScheduledJobService: propose -> (approval) -> ACTIVE, and pause / resume / cancel / look, all on the Slice 1 policy and store's CAS
      operations.py        - the chat door (`command_reply`), the plain-words approval question, and the confirmer that ASKS through the existing confirmation
      presenter.py         - (Slice 5) `present_run` (pure words from recorded facts) and `ScheduledJobReporter` (tells the user once, via the existing notifications)
      tools.py             - `commit_scheduled_job_approval`: CONFIRM-level, non-plannable, non-automatable; activates ONE confirmed proposal
    automation/
      base.py             - Automation/AutomationStatus/Trigger/Condition/AutomationStore interface, AutomationExecution
      sqlite_store.py      - SQLiteAutomationStore: SQLite implementation of AutomationStore
      engine.py             - AutomationEngine: background poller that runs due automations' actions
      tools.py              - CreateAutomationTool/ListAutomationsTool/Enable-/Disable-/Cancel-/RunAutomationTool
      factory.py            - builds the default AutomationStore/AutomationEngine from config
    calendar/
      base.py             - CalendarEvent/CalendarProvider interface, OAuthToken/TokenStore interface
      token_store.py        - SQLiteTokenStore: SQLite implementation of TokenStore
      oauth.py               - GoogleOAuthClient/OAuthCallbackServer/CalendarAuthFlow: desktop/localhost OAuth 2.0
      google_provider.py      - GoogleCalendarProvider: talks to the Calendar REST v3 API over httpx
      tools.py                 - ListEventsTool/CreateEventTool/UpdateEventTool/DeleteEventTool
      factory.py                - builds the default TokenStore/GoogleOAuthClient/CalendarProvider from config
      authorize.py               - `python -m app.calendar.authorize`: one-time interactive OAuth setup
    tasks/
      base.py             - Task/TaskStatus/TaskPriority/TaskStore interface
      sqlite_store.py      - SQLiteTaskStore: SQLite implementation of TaskStore
      tools.py              - CreateTaskTool/ListTasksTool/CompleteTaskTool/CancelTaskTool
      factory.py             - builds the default TaskStore from config
    voice/
      base.py             - Microphone/WakeWordProvider/SpeechToTextProvider/TextToSpeechProvider interfaces + *Error types
      microphone.py         - SoundDeviceMicrophone: sounddevice-backed Microphone
      wake_word.py           - OpenWakeWordProvider: local "hey jarvis" wake-word detection
      stt.py                  - VoskSTTProvider: local, offline speech-to-text
      tts.py                   - WindowsSapiTTSProvider: powershell.exe + System.Speech.Synthesis
      chat_client.py            - ChatClient: the only way this module ever reaches Jarvis (POSTs to /chat)
      session.py                 - VoiceState/VoiceSession: the wake-word -> listen -> STT -> /chat -> TTS loop
      factory.py                  - builds the default Microphone/providers/ChatClient from config
      run.py                       - `python -m app.voice.run`: the application entry point
      download_wakeword_model.py    - `python -m app.voice.download_wakeword_model`: one-time model download
    search/
      base.py           - SearchProvider interface and the SearchResult type
      searxng.py         - SearXNG implementation of SearchProvider
      manager.py          - SearchProviderManager: ordered fallback across SearchProviders
      cache.py            - SearchCache interface: provider-independent get/set
      sqlite_cache.py      - SQLiteSearchCache: SQLite-backed, TTL-aware SearchCache
      cached.py            - CachedSearchProvider: caches a wrapped SearchProvider's results
      modes.py             - SearchMode/SearchModeConfig: deterministic result limit per query
      planner.py            - SearchPlan/SearchPlanner/DeterministicSearchPlanner: which queries to run
      url_utils.py          - normalize_url: conservative comparison-key normalization
      quality.py            - deduplicate_results/ResultQualityProvider: URL-based dedup
      response.py            - SearchResponse: structured query/mode/queries_used/results context
      answer.py              - SearchAnswerGenerator/GeminiSearchAnswerGenerator: grounded answers
      router.py              - SearchRouter/SearchDecision: deterministic DIRECT vs SEARCH routing
      factory.py          - builds the default SearchProvider/SearchCache from config
  tests/
    test_health.py         - proves the server starts and /health works
    test_chat.py            - proves /chat works, with Jarvis Core mocked out
    test_core.py            - unit tests for Jarvis Core's provider delegation and DIRECT/SEARCH dispatch
    test_capability.py      - unit tests for Capability/CapabilityDecision/DeterministicCapabilityRouter
    test_gemini_provider.py - unit tests for the Gemini provider, with HTTP mocked out
    test_openai_compatible_provider.py - unit tests for the shared Groq/OpenRouter HTTP+JSON base class, via a dummy subclass
    test_groq_provider.py    - wiring tests for Groq (URL/model/key), shared JSON logic covered above
    test_openrouter_provider.py - wiring tests for OpenRouter (URL/model/key), shared JSON logic covered above
    test_cloudflare_provider.py - unit tests for Cloudflare Workers AI (account id, success:false handling), with HTTP mocked out
    test_provider_factory.py    - unit tests for the provider registry/factory: selection by name, unknown names, describe_providers metadata
    test_memory.py          - unit tests for the SQLite memory store
    test_tools.py           - unit tests for the Tool Registry and permission model
    test_tool_decision.py   - unit tests for ToolCall/ToolCallBuilder/GeminiToolCallProvider
    test_searxng_provider.py - unit tests for the SearXNG provider, with HTTP mocked out
    test_web_search_tool.py  - unit tests for WebSearchTool, through the Tool Registry
    test_default_tools.py    - proves web_search is in the default/application tool registry
    test_search_provider_manager.py - fallback ordering/short-circuiting, with fake providers
    test_search_factory.py   - proves the provider/cache config wiring
    test_search_cache.py     - unit tests for SQLiteSearchCache (hit/miss/TTL/persistence)
    test_cached_search_provider.py - unit tests for CachedSearchProvider's cache/fallback logic
    test_search_modes.py     - unit tests for SearchMode limits and fallback/caching under a mode
    test_search_quality.py   - unit tests for URL normalization, deduplication, and limit-after-dedup
    test_search_planner.py   - unit tests for DeterministicSearchPlanner's per-mode query plans
    test_search_response.py  - unit tests for SearchResponse and provenance through the whole chain
    test_search_answer.py    - unit tests for GeminiSearchAnswerGenerator and source validation
    test_search_router.py    - unit tests for SearchRouter's DIRECT/SEARCH decisions
    test_reminders_store.py       - unit tests for SQLiteReminderStore (persistence, due queries, lifecycle)
    test_reminders_notifier.py    - unit tests for WindowsToastNotifier/ConsoleNotifier, subprocess mocked out
    test_reminders_scheduler.py   - unit tests for ReminderScheduler (ticking, recurrence, restart recovery)
    test_reminders_tools.py       - unit tests for the reminder Tools, through the Tool Registry
    test_reminders_integration.py - default-registry wiring + end-to-end Core TOOL dispatch onto reminders
    test_reminders_v2_language.py  - (V2) the clock rule and the reminder grammar
    test_reminders_v2_service.py   - (V2) one-time / daily / weekday / weekly / interval, zones, DST, pause/resume/delete, duplicates, persistence, old databases
    test_reminders_v2_scheduler.py - (V2) missed-run collapsing, restart, duplicate prevention, attention/notification path, hostile text
    test_reminders_v2_door.py      - (V2) asking in plain words through JarvisCore, the questions it may ask, the boundaries
    test_reminders_v2_tools.py     - (V2) the reminder tools and the Planner / Handle This boundary
    test_scheduling_builder.py     - the shared request -> Schedule builder (same rules as Reminder V2; only the noun in a refusal varies)
    test_scheduled_jobs_base.py    - (Scheduled Jobs) the state machine, exactly which moves exist
    test_scheduled_jobs_policy.py  - (Scheduled Jobs) lifetime, new-schedule limits, missed runs, caps, circuit breaker, DST identity
    test_scheduled_jobs_store.py   - (Scheduled Jobs) store + ledger: one occurrence -> at most one Job, atomicity, races, restart, schema CHECKs
    test_scheduled_jobs_boundaries.py - (Scheduled Jobs) the foundation cannot run/plan/notify/call tools and is not wired in
    test_run_profiles.py           - (Slice 2) the READ-only unattended profile and per-run deadline, as the existing JobEngine enforces them
    test_scheduled_execution_adapter.py - (Slice 2) occurrence -> Job: fresh context, atomic one-occurrence invariant (races, restart), failure, memory
    test_scheduled_execution_boundaries.py - (Slices 2-3) the package has no loop/thread/timer/notification/memory-write/tool-execution; wired through main.py alone
    test_scheduled_job_scheduler.py - (Slice 3) the scheduler: due discovery, kill switch, overlap, breaker, caps, device trust, boundary, settlement, races, paused period
    test_scheduled_jobs_migration.py - (Slice 3) an OLD-schema ledger upgraded in place: rows, ids, constraints, new outcomes, rollback
    test_scheduled_operations_language.py - (Slice 4) the grammar: what it claims and, above all, what it leaves to ordinary conversation
    test_scheduled_operations.py   - (Slice 4) request -> proposal -> existing confirmation -> atomic approval -> scheduler -> ordinary Job; management; security
    test_scheduled_operations_boundaries.py - (Slices 4-5) the layer cannot plan/run/launch/settle, imports no Handle This, adds no scheduler or store
    test_focus_recognition.py / test_focus_ranking.py / test_focus_service.py / test_focus_door.py / test_focus_interaction_precedence.py / test_focus_presentation.py - (Focus, Slice 1) the grammar boundary, the pure ranking, Tests A-D + untrusted text + bounds, door gating / precedence + package boundaries, a waiting question or confirmation always wins
    test_action_proposal_recognition.py / _service.py / _door.py / _handoff.py / _execution.py / _result.py / _notes.py - (Action Proposal, Slices 2-5) the grammars, building / follow-ups / approval / revalidation, precedence in the real core, the store handoff and package boundary, the exact approved plan run by the real JobEngine (allow-list, refusals, failure, restart, pause/resume, no re-planning), the verified one-time result report (truth, conversation binding, exactly-once, bounds, hook safety), and the optional note step (grammar, shown in full, exact handoff, mixed outcomes, interruption)
    test_scheduled_jobs_final.py   - (Slice 5) Run Now, result presentation, renewal, the full lifecycle, and the final security review
    test_automation_store.py       - unit tests for SQLiteAutomationStore (persistence, trigger/condition/action round-trip)
    test_automation_conditions.py  - unit tests for FactEqualsCondition/TimeWindowCondition/condition_from_dict
    test_automation_engine.py      - unit tests for AutomationEngine (triggers, conditions, permissions, restart recovery, failure isolation)
    test_automation_tools.py       - unit tests for the automation Tools, through the Tool Registry
    test_automation_integration.py - default-registry wiring + end-to-end Core TOOL dispatch onto automations
    test_calendar_oauth.py         - unit tests for GoogleOAuthClient/OAuthCallbackServer/CalendarAuthFlow
    test_calendar_token_store.py   - unit tests for SQLiteTokenStore (persistence, restart behavior)
    test_calendar_provider.py      - unit tests for GoogleCalendarProvider (CRUD, auto-refresh, error mapping)
    test_calendar_tools.py         - unit tests for the calendar Tools, through the Tool Registry (permissions)
    test_calendar_integration.py   - default-registry wiring, automation-guard regression, end-to-end Core TOOL dispatch
    test_tasks_store.py            - unit tests for SQLiteTaskStore (persistence, lifecycle, restart behavior)
    test_tasks_tools.py            - unit tests for the Task tools, through the Tool Registry
    test_tasks_integration.py      - default-registry wiring + end-to-end Core TOOL dispatch onto tasks
    test_voice_wake_word.py        - unit tests for OpenWakeWordProvider (threshold, debounce/reset, error mapping)
    test_voice_stt.py               - unit tests for VoskSTTProvider (missing model, transcription, error mapping)
    test_voice_tts.py               - unit tests for WindowsSapiTTSProvider (subprocess mocked out, interruption)
    test_voice_microphone.py        - unit tests for SoundDeviceMicrophone (device open/read/close failures)
    test_voice_chat_client.py       - unit tests for ChatClient (the /chat HTTP boundary, mocked out)
    test_voice_session.py           - unit tests for VoiceSession's full state machine, with fakes for every provider
  requirements.txt    - exact dependency versions
  .env.example         - template for your local .env (never commit .env itself)
  .gitignore
```

As Jarvis grows, new capabilities (dashboard, memory, agents, voice,
hardware control, etc.) will each get their own module under `app/`,
keeping this structure from turning into a single giant file.

## Tool System

Jarvis is meant to grow to support many integrations (web, email, calendar,
files, WhatsApp, smart home, robotics, printers, ...). To avoid rewriting
Core every time one is added, tools live behind a small, generic
architecture in `app/tools/`:

- **`Tool`** (`app/tools/base.py`) is the interface every tool implements:
  a unique `name`, a `description`, a `permission` level, an optional list
  of `parameters`, optional `requirements` metadata (dependencies, env
  vars, whether it needs network), and an `execute(**kwargs)` method that
  returns a `ToolResult`.
- **`ToolRegistry`** (`app/tools/registry.py`) holds registered tools by
  name. It supports `register`, `unregister`, `get`, `has`, `list_tools`,
  and `execute`. It never contains tool-specific logic (no
  `if name == "weather"` branches anywhere) - it only calls the generic
  `Tool` interface, so it scales from a couple of tools to thousands
  without changes.
- **Permissions** are one of four `PermissionLevel`s, enforced inside the
  Registry (not by Gemini or any other AI provider):
  - `READ` / `SAFE_ACTION` run immediately.
  - `CONFIRM` / `DANGEROUS` require the caller to pass `confirmed=True`
    first; calling without it returns a `CONFIRMATION_REQUIRED` result
    (carrying the permission level, so a caller can show a stronger
    warning for `DANGEROUS` than for `CONFIRM`), and passing
    `confirmed=False` returns `DENIED`.
- **`ToolResult`** is a typed result, not a string: `status` is one of
  `SUCCESS`, `ERROR`, `DENIED`, or `CONFIRMATION_REQUIRED`, plus `data`,
  `message`, and (when relevant) `permission`.
- `JarvisCore` holds a `ToolRegistry` (`core.tools`). `JarvisCore()` on its
  own always starts with an *empty* registry - the class has no idea which
  tools exist, SearXNG included. The application populates it: `app/main.py`
  calls `register_default_tools(core.tools)` (`app/tools/defaults.py`) at
  startup, so the actual running app's registry ends up with `web_search`
  even though `JarvisCore`'s own default stays empty. AI tool-calling
  (letting Gemini decide which tool to invoke) is not wired up yet - that's
  a later milestone.
- Since M6.14, a chat message can also reach a registered tool through the
  `TOOL` capability - see TOOL Capability below. Since M6.16, once `TOOL`
  is decided, a separate Tool Decision Layer can construct its parameters
  - deterministically when possible, or via a small, strictly-validated
  Gemini call otherwise - see Natural-Language Tool Selection below.

### Adding a new tool

1. Create a class implementing `Tool` (see `EchoTool` / `DeleteEverythingTool`
   in `tests/test_tools.py` for minimal examples) with `name`,
   `description`, `permission`, optionally `parameters`/`requirements`,
   and `execute`.
2. Register an instance with a registry: `registry.register(MyTool())`.
   To make it part of the application's default set, register it in
   `app/tools/defaults.py::register_default_tools` instead.
3. That's it - no changes needed to `app/tools/base.py`,
   `app/tools/registry.py`, or `app/core.py`.

## Web Search

`web_search` is the first real tool, implementing the pattern above. It
follows the same "interface + swappable backend" shape as `AIProvider` /
`GeminiProvider`.

**General/stable knowledge vs. web search:** Since Milestone 6.9, this is
decided automatically and deterministically for every `/chat` message -
since M6.13, by the Capability Decision Layer (below), not by `JarvisCore`
itself. Questions about stable facts ("what is quantum entanglement?", "how
does a transistor work?") are answered by Gemini directly, with no tool
call. Questions needing current/external information ("what happened with
NASA today?", "latest robotics news") go through `web_search`. `web_search`
itself still has no idea which category a query falls into; it just
searches whatever query it's given - the decision lives one layer up.

### Capability Decision Layer (M6.13)

`JarvisCore` no longer decides DIRECT vs SEARCH vs TOOL itself - it asks an
injected **`CapabilityRouter`** (`app/capability.py`) for a
`CapabilityDecision` and dispatches on that:

```
message -> JarvisCore.handle_chat
        -> capability_router.route(message) -> CapabilityDecision(capability=..., tool_name=...)
        -> dispatch to DIRECT, SEARCH, or TOOL
```

- **`Capability`** is a small enum - only `DIRECT`, `SEARCH`, and `TOOL`
  (M6.14) exist today. Further capabilities (MEMORY, EMAIL, CALENDAR,
  PRINTER, ROBOTIC_ARM, ...) are deliberately **not** added as unused
  members yet - see Postponed decisions.
- **`CapabilityDecision`** is a small dataclass: `capability` plus
  `tool_name: str | None = None`, used only for `TOOL` (M6.14) - enough for
  Core to dispatch every capability implemented so far, and a place to grow
  if a future capability needs to carry different dispatch data, without
  changing `CapabilityRouter`'s return type.
- **`CapabilityRouter`** is the ABC; **`DeterministicCapabilityRouter`** is
  the only implementation, and it does not duplicate M6.9's keyword rules -
  it **delegates to `SearchRouter`** (still `app/search/router.py`,
  unchanged) for DIRECT vs SEARCH, then translates the `SearchDecision` it
  gets back into a `Capability`. `SearchRouter` is no longer the top-level
  thing `JarvisCore` talks to for that decision, but it's still the single
  source of truth for the matching rules, and its own tests
  (`tests/test_search_router.py`) are unchanged and still authoritative.
  TOOL detection (M6.14, see below) is checked first and is entirely
  separate, narrow, textual logic - it never touches `SearchRouter`.
- `JarvisCore` constructor-injects `capability_router` (defaulting to
  `DeterministicCapabilityRouter(search_router=self.router)`) - so
  injecting a custom `router` (a `SearchRouter`), the same parameter that
  existed before M6.13, still drives the DIRECT/SEARCH decision unchanged;
  injecting `capability_router` directly is a separate, independent
  override for swapping the whole decision layer.
- No Gemini/LLM call anywhere in this layer, same as `SearchRouter` itself.

### TOOL Capability (M6.14)

`TOOL` connects the Capability Decision Layer to the existing Tool System
(M5, below) - capability *routing* stays fully separate from tool
*execution*:

```
message -> DeterministicCapabilityRouter (TOOL check, first)
        -> CapabilityDecision(capability=TOOL, tool_name=...)
        -> JarvisCore._handle_tool -> ToolRegistry.execute(tool_name)
        -> ToolResult -> reply text
```

- **Deliberately conservative, deterministic detection**: only an explicit,
  command-like phrasing at the very start of the message counts, in either
  word order - `use`/`run`/`call`/`invoke` + `tool` + an optional name
  (`"use tool echo"`, `"run tool: echo"`), or (M6.16) the name given
  *before* the literal word "tool" (`"use the echo tool"`, `"call echo
  tool"`). A message that merely *mentions* a tool ("what tool should I
  use for this job?") stays `DIRECT`, and so does a nameless generic
  mention ("use a tool to check the weather" - "tool" itself, not a real
  name, is what would follow "a"). No Gemini/AI call, no NLP, no fuzzy or
  semantic matching - the tool name, when given, is captured exactly as
  typed, never guessed, normalized, or matched against registered tools by
  the router itself.
- **`CapabilityRouter` decides only "TOOL, name X" - it never touches
  `ToolRegistry`.** `JarvisCore._handle_tool` is what calls
  `self.tools.execute(decision.tool_name)` - the same `ToolRegistry.execute`
  API every other caller uses, unchanged. Core inspects only the generic
  `ToolResult` (`status`/`message`/`data`) it gets back, never which tool
  produced it - no `if tool_name == "echo": ...` branching anywhere.
- **No concrete tool identified is a real, typed outcome, not a guess**:
  an explicit invocation with nothing named (`"use tool"`) produces
  `CapabilityDecision(capability=TOOL, tool_name=None)`. With the plain
  `DeterministicCapabilityRouter`, `JarvisCore._handle_tool` returns a
  plain "couldn't tell which tool" reply for it - it never falls back to
  some default tool. (With `HybridCapabilityRouter`, M6.15 below, this
  exact case is instead what triggers the Gemini fallback.) An invocation
  naming a tool that isn't registered needs no special handling at all:
  `ToolRegistry.execute()` already returns a typed "Unknown tool" error for
  that, exactly as it always has.
- **Permission enforcement is untouched**: a `CONFIRM`/`DANGEROUS` tool
  still comes back as `CONFIRMATION_REQUIRED`/`DENIED` from the Registry,
  which simply becomes the reply text - `/chat` has no confirmation
  round-trip yet, so TOOL can't be used to bypass a tool's permission
  level, only to trigger the same outcome a direct `registry.execute(...)`
  call would.
- **`web_search` is intentionally excluded from generic TOOL dispatch** -
  it keeps its own first-class `SEARCH` capability and orchestration path
  (`SearchQueryBuilder` -> `WebSearchTool` -> `SearchAnswerGenerator`,
  M6.9-M6.12), unchanged. TOOL and SEARCH are deliberately not merged, even
  though both ultimately touch tool-shaped things.
- No AI tool selection, no automatic tool-chaining/multi-step execution,
  and no new tools were added - `web_search` is still the only tool
  registered by default (see Tool System below); TOOL currently has
  nothing safe and general-purpose to dispatch to in the running app
  beyond whatever a caller registers itself (as tests do with fakes).

### Hybrid Capability Routing (M6.15)

Capability routing is now deterministic-first, AI-fallback-only, not
purely deterministic:

```
message -> HybridCapabilityRouter
        -> deterministic_router.route(message)  [DeterministicCapabilityRouter, unchanged]
        -> confident? -> that CapabilityDecision, zero AI calls
        -> ambiguous? -> decision_provider.decide(message)  [GeminiCapabilityDecisionProvider]
        -> strictly validated CapabilityDecision -> JarvisCore (unchanged dispatch)
```

- **The deterministic fast path is untouched and still handles almost
  every message**: obvious `DIRECT` questions, explicit `SEARCH` triggers
  (M6.9), and explicit TOOL invocations with a name (M6.14) are all
  returned exactly as before, with **zero** Gemini calls - `HybridCapabilityRouter`
  wraps `DeterministicCapabilityRouter` rather than replacing or
  duplicating any of its logic.
- **The only ambiguity signal is `CapabilityDecision(TOOL, tool_name=None)`** -
  an explicit tool-invocation phrasing ("use tool") with nothing actually
  named. This reuses M6.14's own existing typed representation of "TOOL,
  but nothing to dispatch to" rather than inventing a new confidence
  score or a fuzzy ambiguity heuristic; every other deterministic outcome
  is treated as fully confident. Gemini is consulted **at most once per
  message**, never for the deterministic fast path.
- **`GeminiCapabilityDecisionProvider`** (`CapabilityDecisionProvider`'s
  only implementation) reuses the existing `AIProvider`/`GeminiProvider`
  infrastructure exactly like `GeminiSearchAnswerGenerator` (M6.8) does -
  no second Gemini client, no new API key/config, no paid API/billing/
  subscription of any kind. The prompt is small and fixed: the raw user
  message plus the three capability names (`DIRECT`/`SEARCH`/`TOOL`) -
  no tool descriptions, parameters, or registry contents are ever sent,
  so Gemini can only ever pick a capability, never attempt real tool
  selection.
- **Strict, fail-safe validation before any `CapabilityDecision` is
  constructed**: the response must parse as a JSON object (an optional
  markdown code fence is tolerantly stripped first) containing only
  `capability`/`tool_name`; `capability` must be exactly one of
  `DIRECT`/`SEARCH`/`TOOL` (case-insensitive); `tool_name` must be a
  non-empty string when `capability` is `TOOL` and must be `null`/absent
  otherwise. Any violation - invalid JSON, a non-object, an unknown
  capability, a missing capability, an unexpected extra field, `TOOL`
  with no usable `tool_name`, or a non-null `tool_name` on `DIRECT`/
  `SEARCH` - raises `CapabilityDecisionError` instead of guessing a
  decision or defaulting to some capability.
- **Gemini never executes anything.** A validated `TOOL` decision from
  Gemini flows through the exact same `JarvisCore._handle_tool` ->
  `ToolRegistry.execute(tool_name)` path a deterministic `TOOL` decision
  already uses (M6.14) - same permission enforcement, same "Unknown tool"
  handling for a name Gemini invented, same lack of any
  `if tool_name == "..."` branch in Core.
- **Provider failures follow the existing convention**: a
  `CapabilityDecisionError` (a failed Gemini call, or output that fails
  validation) is caught right where `capability_router.route(...)` is
  called in `JarvisCore.handle_chat` and re-raised as `ProviderError` -
  the same translation `_handle_search` already does for
  `SearchAnswerError`, so `/chat`'s existing error handling (a clean 502)
  needs no changes.
- **`JarvisCore`'s default `capability_router` is unchanged**: it's still
  the fully deterministic `DeterministicCapabilityRouter` (M6.13/M6.14),
  not `HybridCapabilityRouter` - using the hybrid router (and therefore
  ever calling Gemini for a capability decision) is an explicit,
  constructor-injected opt-in, not the running app's default behavior.

### Natural-Language Tool Selection (M6.16)

Once capability routing has already decided `TOOL`, a separate **Tool
Decision Layer** (`app/tools/decision.py`) turns it into a concrete,
validated tool call - a deliberately different question from "which
capability" (M6.13-M6.15), answered by a deliberately different
abstraction:

```
message + CapabilityDecision.tool_name -> ToolCallBuilder
        -> deterministic? -> ToolCall, zero AI calls
        -> ambiguous? -> call_provider.decide(message, candidate_tools)  [GeminiToolCallProvider]
        -> strictly validated ToolCall -> JarvisCore._handle_tool -> ToolRegistry.execute(name, params=...)
```

- **No parallel tool-schema system**: `Tool.parameters` (a `list[
  ToolParameter]`, each with `name`/`type`/`description`/`required` -
  `app/tools/base.py`, unchanged) already *is* the schema. M6.16 only adds
  a small serializer that sends a candidate tool's `name`/`description`/
  parameter list to Gemini - never its `permission`, its `requirements`
  (dependencies/env vars/network), its Python implementation, unrelated
  tools outside the candidate set, or any conversation history.
- **`ToolCall(tool_name, parameters)`** (`app/tools/decision.py`) is the
  small, separate result type - it does not replace or extend
  `CapabilityDecision`; `CapabilityRouter` still only ever decides
  DIRECT/SEARCH/TOOL(+ an optional known name), exactly as in M6.13-M6.15.
- **Deterministic-first, same philosophy as `SearchQueryBuilder`**:
  `ToolCallBuilder` (defaulting to a bare `GeminiToolCallProvider()` for
  its AI fallback) never re-asks which tool when `decision.tool_name` is
  already known from M6.14's explicit invocation syntax - it only
  constructs *parameters*, and even that with **zero AI calls** when the
  tool needs none, or needs exactly one required `"string"` parameter
  that the message already spells out as trailing text (`"use tool echo
  hello world"` -> `{tool_name: "echo", parameters: {message: "hello
  world"}}`, deterministically). Gemini is consulted only when a tool has
  multiple required parameters, a non-`"string"` required parameter, or
  the tool itself isn't already known - and even then, only the
  already-known tool (or the full registry, if the name truly isn't
  known) is offered as candidates, never a second capability-classification
  call.
- **Strict, fail-safe validation before any `ToolCall` is constructed**:
  the response must be a JSON object with only `tool_name`/`parameters`;
  `tool_name` must exactly match one of the *offered* candidate tools -
  Gemini can never select or invent a tool outside that list; `parameters`
  must be a JSON object whose keys are all declared for that tool, with
  every required parameter present and each value's type checked against
  the tool's own declared `type`. Any violation raises `ToolCallError`
  (`app/tools/decision.py`) - never a silently repaired guess.
- **Missing required parameters is its own typed outcome, not a hard
  failure**: `ToolCallIncompleteError` (a `ToolCallError` subtype) covers
  both "a required parameter's value isn't in the message" and "no tools
  are registered to choose from at all" - `JarvisCore` turns this into a
  plain reply (the same graceful-outcome shape M6.14 already used for an
  unnamed tool request), not a 502, since it's about what the user said,
  not a system failure. M6.16 doesn't build a conversational clarification
  workflow on top of this; it only preserves the typed boundary for a
  later milestone to use. Any other `ToolCallError` (a failed AI call, or
  output that fails validation) is translated to `ProviderError` right
  where `tool_call_builder.build(...)` is called in
  `JarvisCore._handle_tool` - the same 502-producing convention
  `CapabilityDecisionError`/`SearchAnswerError` already use.
- **Gemini never executes anything, and `ToolRegistry` is untouched.** A
  validated `ToolCall` flows straight into the exact same
  `ToolRegistry.execute(tool_name, params=...)` API every other caller
  already uses - same "Unknown tool" handling for an unregistered name
  (deterministic or AI-selected alike), same `CONFIRM`/`DANGEROUS`
  permission enforcement, same lack of any `if tool_name == "..."` branch
  in Core.
- **No multi-tool chaining, no agent loop, no retries/reflection that
  re-ask Gemini to reconsider** - `ToolCallBuilder` calls
  `call_provider.decide(...)` at most once per message, and that single
  call only ever proposes one `ToolCall` for `JarvisCore` to dispatch,
  never a sequence of them.

### Explicit Tool Commands (`app/tools/command.py`)

A message that itself says `use tool X ...` / `use the X tool ...` is
understood by a small, deterministic, schema-aware layer - never by
"give the tool the rest of the sentence":

    "use tool open_url https://luniti.app, then use tool scroll -600"
      -> tokenize (words / quoted strings / commas, with source offsets)
      -> split into steps on `then` / `and then` / `followed by`
         (optional leading comma; never inside a quoted string)
      -> resolve every tool in the registry
      -> bind every argument against THAT tool's own `ToolParameter`s
      -> validate every step (types, the tool's own `validate_arguments`,
         `ToolRegistry.check_permission`)
      -> ToolSequence            # immutable; the source of truth
      -> execute step 1, 2, ...  # via ToolRegistry.execute, in order

If **any** step is unknown, malformed, unbindable, invalid or unauthorized,
the whole command is rejected and **nothing executes**. Execution only starts
once the entire sequence has validated; if a step then fails at execution,
the sequence stops there, earlier real results are reported, and later steps
are stated as not run.

- **Argument boundaries come from the schema.** `number` = one numeric token
  (`-600` needs no AI); `boolean` = one true/false/yes/no token; `string` =
  one token or one quoted string. A string parameter declared
  `ToolParameter(..., free_text=True)` (a task's text, a search query) may
  span the rest of its step. Anything a tool's parameters don't consume is a
  rejection - text is never silently ignored, and values are never guessed
  (`use tool move_mouse somewhere` fails closed). To pass text containing
  `then`, a comma-clause or several words to a single-token parameter, quote
  it.
- **No wait tool exists**, so `..., wait for 5 seconds, ...` is rejected as
  an unavailable tool rather than absorbed into the previous step's URL.
- **AI is opt-in.** Only a tool that declares `natural_language_arguments =
  True` (reminders, calendar events, automations) may have its arguments
  extracted from prose by the `ToolCallProvider`; the result goes through the
  same validation. Computer tools are always deterministic.
- **Isolation is unchanged.** The planner resolves tools through
  `JarvisCore._resolve_tool_registry`; computer tools live only in
  `core.computer_tools`, which M15's JobEngine and AutomationEngine are never
  given.

Adding a tool needs no parser change: declare its `ToolParameter`s (and,
optionally, a side-effect-free `validate_arguments`).

### Goal-driven Jobs: data flow, `analyze` and Notes

A Job's plan (M15) can pass data between steps **deterministically** and
without an AI in the loop:

- **Step references.** An argument may be
  `{"from_step": "s1", "path": "data.results.0.title"}`. The referenced step
  must be in the step's own `depends_on`; the path (`status` / `message` /
  `data`, then keys or list indexes, no expressions) is validated at plan
  time and walked exactly at run time. A missing key, bad index, or a source
  that didn't succeed **fails the step** - nothing is guessed or defaulted -
  and goes through the existing bounded retry/replan path (a reference
  failure itself is never retried). The resolved value is re-validated
  against the target tool's schema and `validate_arguments`. Values are
  bounded (`MAX_REFERENCE_CHARS`, 12000): a larger string, list or object is
  **never truncated or pruned** (that could silently change what research
  data means) - the reference fails with a clear size error and the step goes
  through the normal failure/replan path. A gated step's confirmation
  message shows the resolved values.
- **`analyze`** (READ): analysis/synthesis of material an earlier step
  collected, via the existing AI provider. It holds only the provider - no
  registry, stores or OS access - so it can't execute tools or change
  anything; the material is presented as untrusted data inside per-call
  random markers, and the reply is plain text returned in the step's
  `message`. Its `material` may be an object or a list (a search result
  list, say).
- **Notes** (`save_note` SAFE_ACTION, `list_notes` READ): a separate SQLite
  store (`settings.notes_db_path`), never memory facts or the World Model,
  and never promoted into either automatically. When a Job step saves a
  note, the engine records `source_job_id`/`source_step_id`; a plan cannot
  supply them.

Together these express e.g. `web_search -> analyze -> save_note ->
create_reminder` as plan data (see `tests/test_job_data_flow.py`); nothing
about that workflow is hardcoded.

**Observation-driven replanning.** A plan step may carry an optional
`"expectation"` (read-only tools only): a short statement of what its result
must contain. Only such a step is evaluated - ordinary steps cost no AI call.

    step runs -> result -> has an expectation?
        no  -> continue
        yes -> ResultEvaluator: sufficient?
                 yes -> continue
                 no / unavailable -> step FAILED -> the existing bounded
                                     retry/replan path (max_replans,
                                     max_total_steps) -> the Planner decides

The evaluator answers one question - "is this result sufficient for this
step's stated expectation?" - and returns only `{sufficient, reason}`; any
extra key makes its answer invalid. It is given the expectation, the tool
name and the stored result, holds only an AI provider, and cannot plan,
name or run tools, or change permissions/autonomy. A provider outage,
malformed answer or missing evaluator is recorded as verdict `unavailable`
and fails the step - never treated as "sufficient". A result with no message
and no data is judged insufficient without any AI call. The verdict is
persisted on the step, so a restart never re-runs the tool, never evaluates
twice, and never lets a dependent run on an unvetted result. Tool results
remain untrusted data throughout.

**One provider path for every AI call in a Job.** Planning, replanning,
result evaluation and `analyze` all run through `core.provider` (the
AIProviderManager), so they share its ordered fallback and cooldown: one
rate-limited provider no longer fails a Job while another is healthy.

**Final report.** A completed Job's `final_result` is built from the plan's
*terminal* steps (the ones nothing depends on) - not a transcript of every
tool message: the synthesis of a step whose tool declares `produces_report`
(`analyze`) leads, followed by one bounded `tool: output` line per other
terminal step ("save_note: Saved note ...", "create_reminder: Reminder set
..."). It is deterministic (no AI call), persisted on the Job, and is what
"is that finished?" replies with.

**Completion notification (optional).** When a Job completes or fails, a
desktop toast can announce it (`JARVIS_ORCHESTRATION_NOTIFY_ON_COMPLETION`,
default on; the same `Notifier` reminders use). It is best-effort and never
part of the execution path: it is sent only after the Job's final state is
persisted, on its own daemon thread, its text is sanitised (single line, no
control characters, bounded, always starting with "Goal:"), and any failure -
or a notifier that hangs - cannot change the Job. The persisted Job is the
source of truth; a notification is never sent twice for one Job.

### Search Routing (M6.9)

The deterministic DIRECT/SEARCH matching rules themselves still live in
**`SearchRouter`** (`app/search/router.py`) - **deterministic keyword/
phrase matching only, never an extra Gemini call**. Since M6.13,
`DeterministicCapabilityRouter` (above) is what actually calls it; the two
paths it distinguishes are unchanged:

- **DIRECT** - the original, unchanged path: `_build_prompt` (history +
  facts + message) -> `provider.generate_reply(...)`.
- **SEARCH** - `WebSearchTool.search(message)` -> `SearchResponse` ->
  `SearchAnswerGenerator.generate(response)` -> `SearchAnswer.answer` is
  used as the reply. `WebSearchTool.execute()` (the `Tool`/`ToolResult`
  path used by the Registry) is untouched; Core instead calls the new
  `WebSearchTool.search(...)` method directly - the "minimal typed seam"
  that returns a real `SearchResponse` instead of a dict, so Core never
  reconstructs one from `ToolResult.data`.

`SearchRouter.route(message)` returns a `SearchDecision` (`DIRECT` /
`SEARCH`) by regex-matching three fixed categories - explicit search
requests ("search for", "search the web", "please search", "look up",
"google it", ...), freshness words ("latest", "current", "today",
"recent", "now", "this week/month/year"), and live/external-information
words ("weather", "scores", "results", "prices", "availability").
**Explicit search intent always overrides the subject** - e.g. "search the
web and explain quantum entanglement" is SEARCH - which falls out for free
from doing one unconditional match over every pattern, not from any
special override logic.

The explicit-search trigger deliberately requires an actual request
phrasing rather than the bare word "search": a bare `\bsearch\b` would
match "search" used as a plain noun/algorithm name ("explain binary
search", "how does depth-first search work") - a very plausible question
for a programming-context assistant - and wrongly send a stable question
down the SEARCH path. This was caught and fixed during verification (see
`tests/test_search_router.py`).

This routing is intentionally coarse (plain keyword matching, not intent
understanding), and some trigger words are still genuinely ambiguous in
isolation - "current" ("current events" vs. "electric current"),
"results" ("latest results" vs. "results of a calculation"), "available"/
"price"/"score" used in a stable, non-live sense. These are **accepted,
documented trade-offs** of deterministic keyword matching (verified and
pinned down by tests, not silently left unhandled), not something this
milestone attempts to fix with added heuristics - see Postponed decisions.

Both `router`/`answer_generator` are constructor-injectable on
`JarvisCore`, same DI pattern as `provider`/`memory`/`tools`. A
`SearchProviderError` (search failed) or `SearchAnswerError` (answer
generation failed) on the SEARCH path is re-raised as `ProviderError` -
the same error type the DIRECT path already raises on a Gemini failure -
so `/chat`'s existing error handling (a 502) needed **no changes** to
cover the new path. `SearchAnswer.sources` is exposed through
`/chat`/`ChatResponse` as `sources: list[ChatSource]` (M6.11): each
`ChatSource` carries `title`/`url`/`query` copied verbatim from the
underlying `SearchSource`, so only citations already deterministically
validated against real search results (see Search Answer Layer below) are
ever returned - never a snippet, and never an invented or modified URL. A
DIRECT reply always returns `sources: []`, since it never produces a
`SearchAnswer` at all.

- **`SearchProvider`** (`app/search/base.py`) is the interface a search
  backend implements: `search(query, max_results) -> list[SearchResult]`.
  `SearchResult` is a standardized `title` / `url` / `snippet` triple, the
  same regardless of which backend produced it, plus `source` and `query`
  metadata (see Search Result Quality and Search Context + Provenance
  below).
- **`SearXNGProvider`** (`app/search/searxng.py`) is the first
  implementation, calling a SearXNG instance's JSON search API
  (`GET {JARVIS_SEARXNG_URL}/search?q=...&format=json`) over `httpx`.
  Network errors and unexpected response shapes both raise `SearXNGError`
  (a `SearchProviderError`), never a bare exception.
- **`WebSearchTool`** (`app/tools/web_search.py`) implements `Tool`,
  taking a `SearchProvider` in its constructor (defaulting to a bare
  `SearXNGProvider` when none is given). It has
  `permission = PermissionLevel.READ`, so the Registry runs it immediately
  with no confirmation step. `search(query, max_results, mode) ->
  SearchResponse` runs the actual pipeline (planning, multi-query
  execution, dedup) and is the one place that logic lives; `execute(...)
  -> ToolResult` is a thin wrapper that calls `search()` and either
  flattens the `SearchResponse` into `ToolResult.data` or turns a
  `SearchProviderError` into `ToolResult.error(...)` - `execute()`'s
  contract (shape and errors) is unchanged from before M6.9.
- **`SearchProviderManager`** (`app/search/manager.py`) is itself a
  `SearchProvider` that wraps an *ordered list* of other `SearchProvider`s
  and adds fallback: it tries each in turn, moving to the next only when
  one *raises* `SearchProviderError` (a network/provider failure). A
  provider returning zero results is a valid answer, not a failure, so it's
  returned as-is with no fallback. If every provider raises, the manager
  raises one combined `SearchProviderError` rather than the caller seeing a
  pile of separate exceptions. Because it implements `SearchProvider`
  itself, `WebSearchTool(provider=SearchProviderManager([...]))` is
  indistinguishable from `WebSearchTool(provider=SomeSingleProvider())` to
  both `WebSearchTool` and `JarvisCore` - neither needed to change.
- **`app/search/factory.py`** turns `JARVIS_SEARCH_PROVIDER_ORDER` (a
  comma-separated list of provider names) into the actual ordered list of
  `SearchProvider` instances, via one small name→class mapping
  (`_PROVIDER_FACTORIES`), and wraps them in a `SearchProviderManager`.
  This is the one place a provider's name is ever checked as a string.
  `app/tools/defaults.py` uses it to build the application's default
  `web_search` tool, so `JARVIS_SEARCH_PROVIDER_ORDER` controls fallback
  order without touching `WebSearchTool` or `JarvisCore`.
- Adding a real `BraveProvider` or `TavilyProvider` later means: implement
  `SearchProvider`, add one entry to `_PROVIDER_FACTORIES`, and list it in
  `JARVIS_SEARCH_PROVIDER_ORDER` - no other file changes.

The full default chain, wired in `app/tools/defaults.py`, is:

```
WebSearchTool -> ResultQualityProvider -> CachedSearchProvider -> SearchProviderManager -> SearXNGProvider (, ...)
```

- **`SearchCache`** (`app/search/cache.py`) is the caching interface:
  `get(key) -> list[SearchResult] | None` and
  `set(key, results) -> None`. It knows nothing about queries, providers,
  or normalization - it's a plain key/value store with expiry, so the
  backend can be swapped later without touching `CachedSearchProvider`.
- **`SQLiteSearchCache`** (`app/search/sqlite_cache.py`) is the first
  implementation, following the same pattern as `SQLiteMemoryStore`: one
  held `sqlite3` connection guarded by a lock, a single `search_cache`
  table (`key`, `value`, `expires_at`), and a local file so cached results
  survive Jarvis restarts. No Redis or other external service. An entry
  past its `expires_at` is returned as a plain miss (`None`), not an
  error.
- **`CachedSearchProvider`** (`app/search/cached.py`) is itself a
  `SearchProvider` that wraps another one plus a `SearchCache`. The cache
  key is the normalized query (`strip().lower()`) plus `max_results`, e.g.
  `"python::5"` - no embeddings or semantic similarity, only exact
  normalized matches hit the cache. On a hit it returns the cached results
  without calling the wrapped provider; on a miss it calls the wrapped
  provider and, only if that call *succeeds*, caches the result - an empty
  result list (`[]`) is cached as a valid answer, but a
  `SearchProviderError` is never cached and propagates unchanged, so
  `SearchProviderManager`'s fallback behavior underneath is unaffected by
  caching. Because it implements `SearchProvider` itself, it composes with
  `SearchProviderManager` exactly like any other provider - neither
  `WebSearchTool` nor `JarvisCore` know caching or SQLite are involved.
- **`app/search/factory.py`** also builds the default `SearchCache` via
  `create_default_search_cache()`, reading `JARVIS_SEARCH_CACHE_TTL` and
  `JARVIS_SEARCH_CACHE_DB_PATH`. `app/tools/defaults.py` composes
  `CachedSearchProvider(create_default_search_provider(),
  create_default_search_cache())` as the application's default `web_search`
  provider.

### Search Modes

`web_search` takes an optional `mode` argument (`fast`, `normal`,
`research`, or `critical`; `normal` is the default, matching the tool's
pre-mode behavior exactly, so existing callers that only pass `query` are
unaffected). `SearchMode` and `SearchModeConfig` (`app/search/modes.py`)
are a small, fully deterministic strategy layer *above* the provider
chain: a mode picks the `result_limit` requested per query (overridden if
the caller passes an explicit `max_results`). Modes never know about
SearXNG, never touch caching or fallback, and never call Gemini to decide
anything - `SearchMode` doesn't even import `SearchProvider`.

Invalid mode values aren't specially validated: `WebSearchTool.execute`
constructs `SearchMode(mode)` directly, so an unknown string raises
`ValueError`, the same way any other unexpected input in this codebase
isn't pre-validated - through the Tool Registry that becomes a normal
`ToolResult` error via its existing generic exception handling (see
`ToolRegistry.execute`), not any special-cased code in `WebSearchTool`.

### Search Query Planning (M6.6)

Originally, `research`/`critical` ran multiple passes of the *identical*
query - which `CachedSearchProvider` then mostly absorbed, so extra passes
added little real coverage. `SearchPlanner` (`app/search/planner.py`) fixes
this: it decides *which distinct queries* to run for a mode, so multi-query
modes actually diversify what's searched.

- **`SearchPlan`** is just an ordered `list[str]` of queries to run.
- **`SearchPlanner`** is the interface (`plan(query, mode) -> SearchPlan`).
  It knows nothing about `SearchProvider`, caching, or result quality -
  only `SearchMode` and plain strings.
- **`DeterministicSearchPlanner`** is the only implementation: fixed,
  non-AI suffixes, never a generic filler word like "research":

  | Mode | Queries planned |
  |------|------------------|
  | `fast` | original |
  | `normal` (default) | original |
  | `research` | original, `{query} {current_year}`, `{query} student` |
  | `critical` | original, `{query} {current_year}`, `{query} student`, `{query} official` |

  `current_year` defaults to the real current year but is injectable
  (`DeterministicSearchPlanner(current_year=2026)`) for deterministic
  tests.
- `WebSearchTool` takes an injected `planner` (defaulting to
  `DeterministicSearchPlanner()`, same pattern as `provider`), calls the
  wrapped `SearchProvider` once per planned query at the mode's result
  limit, then runs the existing M6.5 deduplication over *all* combined
  results before applying the final limit - so a mode's limit still
  reflects unique results across every planned query, not just one.
- Each planned query is still just a normal call into
  `ResultQualityProvider -> CachedSearchProvider -> SearchProviderManager`,
  so it's cached, deduplicated, and falls back exactly like any other
  query - the planner adds no new caching or fallback logic of its own,
  and nothing below `WebSearchTool` knows planning exists. Because planned
  queries are now genuinely distinct strings, caching pays off across
  repeated `web_search` calls with the same query+mode (identical plan,
  every planned query already cached), not within a single call the way
  identical repeated passes used to be absorbed.
- `SearchModeConfig.passes` (`app/search/modes.py`) is unchanged and still
  present for compatibility, but no longer drives how many queries
  `WebSearchTool` runs - `SearchPlanner` is now the source of truth for
  that.

### Search Result Quality (M6.5)

`SearchResult` (`app/search/base.py`) carries a fourth field alongside
`title` / `url` / `snippet`: a generic `source` identifier (e.g.
`"searxng"`), set by each provider. `url` is always left exactly as the
provider returned it - it, together with `title` and `source`, is
preserved specifically so a future text-interface presentation layer can
render clickable sources (e.g. "Source 1 · Wikipedia") without needing to
know which provider produced a result. **No such UI exists yet** - this
milestone only makes sure the data is there for it.

- **`normalize_url`** (`app/search/url_utils.py`) builds a conservative
  comparison key for a URL - never the URL itself - by stripping
  whitespace, lowercasing scheme/host, collapsing a trailing slash, and
  removing obvious tracking parameters (`utm_*`, `fbclid`, `gclid`).
- **`deduplicate_results`** (`app/search/quality.py`) removes results whose
  normalized URLs match, keeping the *first* occurrence and preserving
  order - no ranking or credibility scoring.
- **`ResultQualityProvider`** (`app/search/quality.py`) is itself a
  `SearchProvider` that wraps another one and deduplicates its results,
  the same "transparent decorator" shape as `CachedSearchProvider` and
  `SearchProviderManager`. It sits directly above `CachedSearchProvider` in
  the default chain.
- `WebSearchTool` deduplicates the *combined* results from every search
  pass (see Search Modes above) before applying the result limit, so
  limits are always applied **after** deduplication, not before.

### Search Context + Provenance (M6.7)

A successful `web_search` no longer returns a flat list of results - it
returns structured search context, so an answer-generation layer can tell
not just *what* came back but *which query found it* and *under what
mode*. `SearchAnswerGenerator` (Search Answer Layer, below) is the first
consumer of this.

- **`SearchResult.query`** (`app/search/base.py`) records the exact
  planned query that produced that specific result - stamped on by the
  concrete provider that created it (`SearXNGProvider` sets it to the
  query it was actually asked), then carried unchanged through every
  wrapping layer above it (`ResultQualityProvider`, `CachedSearchProvider`,
  `SearchProviderManager`) since none of them construct new `SearchResult`
  objects, only filter/cache/forward the ones they're given.
  `SQLiteSearchCache` serializes/deserializes it like every other field,
  so it survives cache round-trips too.
- **`SearchResponse`** (`app/search/response.py`) is a small, provider-
  agnostic dataclass - `query` (the original user query), `mode` (the
  resolved `SearchMode`), `queries_used` (every query the `SearchPlanner`
  actually ran), and `results` (the final deduplicated, limit-applied
  list). It has no Gemini or provider logic of its own.
- `WebSearchTool.execute` builds one `SearchResponse` per call and puts a
  plain-dict rendering of it in the successful `ToolResult.data`:
  `{"query", "mode", "queries_used", "results"}`, where each entry in
  `results` is `{"title", "url", "snippet", "source", "query"}`. This is
  the one place the domain object is flattened to JSON-friendly types
  (`mode` becomes its string value) - `ToolResult` itself is untouched.
- Deduplication already kept the first occurrence's object completely
  unchanged (see Search Result Quality above), so provenance falls out of
  that for free: if the same URL is found by two different planned
  queries, the kept result's `query` is whichever query found it *first*.
- None of this required changing `SearchProviderManager`,
  `CachedSearchProvider`, `ResultQualityProvider`, `SearchPlanner`, or
  `JarvisCore` - provenance flows through the existing chain because
  every layer already just passes `SearchResult` objects through
  unmodified; only the source (`SearXNGProvider`) and the final assembly
  point (`WebSearchTool`) needed to change.

### Search Answer Layer (M6.8)

A provider-agnostic layer that turns a `SearchResponse` into a grounded,
free-text answer. Since Milestone 6.9 it's the last step of `JarvisCore`'s
SEARCH path (see Search Routing above); it was built and tested standalone
first, before being wired in.

- **`SearchAnswerGenerator`** (`app/search/answer.py`) is the interface:
  `generate(response: SearchResponse, conversation_context: str = "") ->
  SearchAnswer`. It only imports `SearchResponse`/`SearchResult` and the
  existing `AIProvider` interface - never Gemini's HTTP details directly.
- **`GeminiSearchAnswerGenerator`** is the only implementation, built
  entirely on the existing `AIProvider` abstraction (defaulting to
  `GeminiProvider()`, injectable like every other provider in this
  codebase) - no separate API key, HTTP client, or AI integration of its
  own. It prompts Gemini with the original query, resolved mode, every
  query actually searched, and each result's title/URL/snippet as
  evidence, with explicit instructions to answer only from that evidence,
  never invent facts/sources/URLs/citations, and say so plainly if the
  evidence is insufficient rather than guessing.
- **`SearchAnswer`** (`answer: str`, `sources: list[SearchSource]`) and
  **`SearchSource`** (`title`, `url`, `query`) are plain dataclasses.
  Crucially, **Gemini's text is never trusted for source data**: a source
  is only included if a real `SearchResult.url` from the `SearchResponse`
  appears as an exact token in Gemini's answer text (trailing punctuation
  like a sentence-ending period is stripped first, but a URL that merely
  *extends* a real one, e.g. an extra path segment, does **not** match).
  `title`/`url`/`query` on the returned `SearchSource` always come from
  the real `SearchResult`, never from anything Gemini wrote, so invented
  or modified URLs can never appear, and citing the same URL twice
  produces one source, not two.
- A `ProviderError` from the underlying `AIProvider` (e.g. a missing key
  or network failure) is caught and re-raised as `SearchAnswerError`, the
  answer layer's own domain error - matching how `SearXNGError` wraps
  `httpx` errors at the search-provider boundary.

### Conversation-Aware Search (M6.10)

A SEARCH follow-up like "what's the latest application deadline?" right
after "tell me about TUM" is ambiguous on its own. M6.10 lets the *answer*
step understand it, without changing what gets searched:

```
history/facts (already loaded by handle_chat)
    -> bounded conversation context (JarvisCore._format_context)
    -> answer_generator.generate(response, conversation_context=...)
```

- `SearchAnswerGenerator.generate()` gained one optional parameter,
  `conversation_context: str = ""` (both the abstract interface and
  `GeminiSearchAnswerGenerator`). When empty (the default), the generated
  prompt is byte-for-byte identical to before M6.10 - fully backward
  compatible. When non-empty, the prompt gains a clearly labeled "Recent
  conversation context (for understanding references only)" section, with
  explicit instructions: use it *only* to resolve what the question refers
  to, never as evidence, and never a source of facts/sources/URLs to
  answer from - the existing M6.8 grounding guarantees (answer only from
  search evidence, never invent citations) are unchanged and unweakened.
- **The search query itself was untouched in M6.10**: `SearchRouter.
  route(message)` and `WebSearchTool.search(message)` both received
  exactly the raw current message, nothing appended, no AI rewriting.
  `JarvisCore._handle_search` only built a context string and passed it to
  the answer generator, *after* the search had already run. M6.12 (below)
  later gave `WebSearchTool.search()` a narrow, deterministic exception to
  this for context-dependent follow-ups - `SearchRouter.route()` still
  never sees anything but the raw message.
- `JarvisCore._format_context(history, facts, max_messages=None)` is a new
  shared static helper factoring out the facts/history formatting that
  `_build_prompt` (DIRECT) already did, so DIRECT and SEARCH don't
  duplicate that logic. DIRECT still calls it unbounded
  (`max_messages=None`), so its behavior is **byte-for-byte unchanged**;
  SEARCH calls it bounded to `JarvisCore._SEARCH_CONTEXT_MAX_MESSAGES`
  (6 recent messages) and then caps the result to
  `_SEARCH_CONTEXT_MAX_CHARS` (800 characters) as a defensive backstop -
  so a search-answer prompt never grows with conversation length the way
  DIRECT's (still-unbounded) prompt can.
- Routing (`SearchRouter`, Milestone 6.9) is completely unchanged - it
  still decides DIRECT vs SEARCH from the current message alone, before
  any context is built. M6.10 only affects what happens *after* SEARCH has
  already been chosen; it cannot cause more messages to route to SEARCH.
- Using context to change *what* gets searched (not just how the answer is
  worded) was intentionally deferred here - see Search Query Understanding
  (M6.12) below for the deterministic version that was later built, and
  Postponed decisions for what's still left (an AI-driven version).

### Search Query Understanding (M6.12)

A SEARCH follow-up like "is there anything more recent on that?" right
after "what's the latest news about SpaceX?" is a valid SEARCH message
(it matches `SearchRouter`'s freshness pattern), but searching it
literally throws away the one thing that makes it answerable - the
subject ("SpaceX") lives only in the previous turn. M6.12 lets
`WebSearchTool.search()` see that, without touching *whether* something
gets searched:

```
message (already routed to SEARCH by SearchRouter)
    -> bounded conversation context (JarvisCore._format_context, same M6.10 bounds)
    -> query_builder.build(message, context) -> the actual query
    -> web_search.search(query)
```

- **`SearchQueryBuilder`** (`app/search/query_builder.py`) is a new
  interface with one deterministic implementation,
  `DeterministicSearchQueryBuilder` - same ABC-plus-one-impl shape as
  `SearchRouter` and `SearchPlanner`. **This is deterministic contextual
  enrichment, not AI-generated query rewriting**: no Gemini call, no
  network call, no embeddings/semantic similarity, no ranking, no
  autonomous research. `JarvisCore` constructor-injects it
  (`query_builder`, defaulting to `DeterministicSearchQueryBuilder()`),
  the same DI pattern already used for `router` and `answer_generator`.
- The trigger set is a small, explicit, fixed list of context-dependent
  phrasings ("on that/this/it", "about that/this/it/them", "the second
  one", "what about", "how about", "and that/this/them") - deliberately
  narrow, the same accepted-trade-off philosophy as `SearchRouter`'s own
  keyword list. It is **not** a general coreference resolver: a follow-up
  like "and Blue Origin?" won't be caught, on purpose - see Postponed
  decisions.
- Only when a message matches that trigger set **and** the bounded context
  contains a prior user message does enrichment happen, by prepending that
  prior user message to the current one. Every other message - including
  every self-contained SEARCH query such as "what's the latest news about
  NASA?" - is sent to `WebSearchTool.search()` **byte-for-byte unchanged**.
- **`SearchRouter.route(message)` is completely unaffected**: it still
  runs first, on the raw current message alone, before any context is
  built or any query is enriched, and the enriched query is never fed back
  into it. Only `WebSearchTool.search()` - called *after* routing has
  already chosen SEARCH - ever receives anything other than the raw
  message.
- Reuses the exact same bounded context `JarvisCore._handle_search` was
  already building for the M6.10 answer generator
  (`_SEARCH_CONTEXT_MAX_MESSAGES` / `_SEARCH_CONTEXT_MAX_CHARS`) - computed
  once and passed to both `query_builder.build()` and
  `answer_generator.generate()`. No new or wider context system.
- `SearchResponse.query` and each `SearchResult.query` (M6.7 provenance)
  now record the *enriched* query actually sent to the search pipeline,
  not the raw follow-up message - consistent with them always describing
  what was actually searched. Neither dataclass's shape changed.
- Zero added latency or cost: the builder is plain string/regex matching,
  no network or model call, so it adds nothing to the per-request cost of
  a SEARCH turn - matching Jarvis's zero-dollar/minimal-cost stance. A
  misfire (over- or under-enriching) degrades to today's existing failure
  mode (a possibly-weak search), never a crash, and never touches the
  M6.8 grounding guarantee - answers still only ever cite real, validated
  `SearchResult`s regardless of query quality.
- No new persisted state: there is no "last search query" stored anywhere;
  the prior user message is recovered each time from the same
  already-loaded, already-bounded conversation history.

Configuration:
- `JARVIS_SEARXNG_URL` - base URL of a SearXNG instance (self-hosted or a
  trusted public one). No default URL and no key is hard-coded anywhere.
- `JARVIS_SEARCH_PROVIDER_ORDER` - comma-separated, ordered list of
  provider names to try, e.g. `searxng` (default) or, once more providers
  exist, `searxng,brave`. Unknown names raise `UnknownSearchProviderError`
  at startup rather than failing silently.
- `JARVIS_SEARCH_CACHE_TTL` - how long a cached `web_search` result stays
  valid, in seconds. Defaults to `600` (10 minutes). Expired entries are
  treated as cache misses.
- `JARVIS_SEARCH_CACHE_DB_PATH` - local SQLite file the cache is stored in.
  Defaults to `jarvis_search_cache.db`.

See `.env.example` for all four.

## Persistent Tasks, Reminders & Background Scheduling (M7)

A new `app/reminders/` module adds Jarvis's first form of scheduled state:
persistent reminders, a background poller that fires them, and a free
local notification mechanism - all deliberately narrow, so "something can
be scheduled" never becomes "something can be automated":

```
create_reminder(text, due_in_minutes, recurrence_unit?, recurrence_interval?)
    -> ReminderStore.create -> Reminder(status=PENDING) persisted to SQLite

ReminderScheduler (background thread, polls every JARVIS_REMINDER_POLL_INTERVAL_SECONDS)
    -> ReminderStore.list_due(now) -> due Reminders
    -> Notifier.notify(title, reminder.text)          # the ONLY action ever taken
    -> ReminderStore.record_trigger(...)              # lifecycle/execution log
```

- **`Reminder`/`ReminderStatus`/`Recurrence`/`ReminderStore`**
  (`app/reminders/base.py`) are the domain types, the same
  "interface + swappable backend" shape as `MemoryStore`/`SearchCache`.
  `ReminderStatus` is `PENDING` -> (`TRIGGERED` for a one-time reminder, or
  back to `PENDING` with an advanced `due_at` for a recurring one) ->
  `COMPLETED`/`CANCELLED` (terminal). `Recurrence` is deliberately just a
  fixed interval (every N minutes/hours/days/weeks) - no cron expressions,
  no calendar-aware scheduling - the same accepted-trade-off philosophy as
  `SearchRouter`'s keyword list.
- **The security boundary this milestone adds**: a `Reminder` only ever
  carries free text plus a due time/recurrence - never a tool name or
  parameters. `ReminderScheduler` (`app/reminders/scheduler.py`) only ever
  calls `Notifier.notify(...)` for a due reminder - it never executes a
  registered `Tool`, never runs arbitrary code, and has no way to schedule
  anything other than a notification, no matter what's persisted.
- **`SQLiteReminderStore`** (`app/reminders/sqlite_store.py`) follows the
  exact pattern `SQLiteMemoryStore`/`SQLiteSearchCache` already established
  - one held `sqlite3` connection guarded by a lock, a local file so
  reminders (and whether they've already fired) survive a restart.
  `record_trigger` is the one place lifecycle state changes: on a
  *successful* notification it advances `due_at` and stays `PENDING`
  (recurring) or becomes `TRIGGERED` (one-time, so it's never picked up
  again); on a *failed* one it leaves the reminder `PENDING`/due so the
  next poll retries it. Either way it appends a `ReminderExecution` row to
  a separate `reminder_executions` table - the persisted lifecycle/
  execution log this milestone requires, independent of the reminder's own
  current state.
- **`ReminderScheduler`** (`app/reminders/scheduler.py`) is a plain daemon
  thread that calls `tick()` (`list_due(now)` -> notify each -> record the
  outcome) on a fixed interval. **Restart recovery needs no special code**:
  `list_due` is just `status = 'pending' AND due_at <= now` against
  persistent SQLite, so a reminder that came due while Jarvis was offline
  is simply found and triggered on the very first poll after startup,
  exactly like any other due reminder.
- **`Notifier`/`WindowsToastNotifier`/`ConsoleNotifier`**
  (`app/reminders/notifier.py`) is the free, fully local notification
  mechanism: `WindowsToastNotifier` shells out to `powershell.exe` (ships
  with every supported Windows version) to raise a native Windows Runtime
  toast via `Windows.UI.Notifications` - no new pip dependency, no
  external service, no API key, matching this codebase's "no extra SDK
  when a standard tool already does the job" stance (see
  [Stack](#stack)). Arguments reach PowerShell as a plain argv list
  (`-File <script> -AppId ... -Title ... -Message ...`), never
  shell-interpolated text, so reminder text can never inject script code.
  `ConsoleNotifier` is the non-Windows/dev fallback, used automatically by
  `app/reminders/factory.py::create_default_notifier` when
  `platform.system() != "Windows"`.
- **`create_reminder`/`list_reminders`/`complete_reminder`/
  `cancel_reminder`** (`app/reminders/tools.py`) are ordinary `Tool`s
  registered in `app/tools/defaults.py` alongside `web_search` - reached
  through the exact same `TOOL` capability and Tool Decision Layer
  (M6.13-M6.16) every other tool already uses, with no reminder-specific
  branching anywhere in `JarvisCore`. `complete_reminder`/`cancel_reminder`
  each take one required string parameter (`reminder_id`), so
  `ToolCallBuilder` fills it deterministically from trailing message text,
  zero AI calls - `create_reminder` takes prose arguments
  (`natural_language_arguments`), so it goes through `ToolCallBuilder`'s
  existing Gemini-assisted fallback, same as any other prose-argument tool. `create_reminder`/
  `complete_reminder`/`cancel_reminder` are `SAFE_ACTION` (each only ever
  adds a new reminder or changes that one reminder's own status);
  `list_reminders` is `READ`.
- **`app/main.py`** builds one shared `ReminderStore` at startup, passes
  it into both `register_default_tools` (so the tools operate on it) and
  `create_default_scheduler` (so the scheduler's polling sees the exact
  same persisted rows, not a second independent SQLite connection to the
  same file), and starts/stops the scheduler from a FastAPI `lifespan`
  handler.

Configuration:
- `JARVIS_REMINDERS_DB_PATH` - local SQLite file reminders are stored in.
  Defaults to `jarvis_reminders.db`.
- `JARVIS_REMINDER_POLL_INTERVAL_SECONDS` - how often the background
  scheduler polls for due reminders, in seconds. Defaults to `30`.

### Reminder V2 (everyday reminders)

Reminders now work the way a person says them, built on the pure `app/scheduling/` foundation (no second
scheduler, no new thread, no new notification path):

```
"Remind me tomorrow at 7 PM to study."          one time, local date and time
"Remind me every day at 7 PM to study."         daily
"Remind me every weekday at 6 PM to ..."        Monday to Friday
"Remind me every Monday at 5 PM about ..."      chosen weekdays
"Remind me in 10 minutes ..." / "every 2 hours" the existing interval forms
"What reminders do I have?"   "Pause / Resume / Delete my study reminder"
```

- **Time zone.** A new reminder is created in `JARVIS_TIMEZONE`, else the system zone when it can be determined
  reliably, else it fails with a clear message (nothing is created that might fire at the wrong time). The zone is
  stored with the reminder and never changes; "7 PM" stays 7 PM across daylight saving (a time inside the
  spring-forward gap becomes the first valid minute after it; a time that happens twice is its first occurrence).
- **Asking, not guessing.** "Remind me to study" asks *when*. A bare "at 7" alone means the next time the clock reads 7
  (and the reply says which); with an explicit day or a repetition it asks *AM or PM*. A pause / resume / delete that
  matches several reminders lists them and asks which ONE; "delete all my reminders" is declined. One question at a
  time, held in memory for a few minutes; anything that is not an answer simply drops it.
- **Missed reminders.** What came due while Jarvis was off is NOT replayed. A run up to `JARVIS_SCHEDULE_GRACE_SECONDS`
  (10 minutes) late is announced normally; later than that, ONE catch-up announcement stands for every missed
  occurrence ("...I missed while I was offline (it came due 3 times; the last was ...)") and the reminder jumps to its
  next future occurrence. Pausing never replays either: resuming goes to the next future occurrence. A NEW
  "every N minutes/hours/days/weeks" reminder (chat, or `create_reminder` with `due_in_minutes` + a recurrence) is stored as a
  V2 interval schedule, so it gets this rule too. Reminders that already exist from before V2 (stored with a legacy
  recurrence and no schedule) are NOT changed, migrated or rewritten: they keep their original behaviour, including replaying
  what was missed one occurrence per poll.
- **Duplicates.** An occurrence is identified by the UTC instant it resolved to; the update that records "handled" is a
  compare-and-set, and an occurrence is recorded as delivered at most once, so two polls (or two processes) cannot
  announce it twice. Creating an identical reminder twice returns the existing one.
- **Notification.** A due reminder is a normal `REMINDER` event (never urgent, never important from its words) handed to
  the existing attention-gated router, exactly as before; the event key is unchanged, so the Morning Brief still
  recognises it.
- **Boundary.** Reminders remain reminders: a reminder carries only text and a schedule, and becoming due only ever
  produces a notification - no Job, no Planner, no tool, no automation, no memory or world-model write. The chat door
  reads a fixed grammar (no model), acts only through the registered reminder tools, and only on a whole-message request
  on the ordinary conversation path; a multi-step goal that merely contains a reminder still goes to Handle This.
  `pause_reminder` / `resume_reminder` are not plannable and not automatable. The reminder's own words are always data.
- **Storage.** The `reminders` table gains a nullable `schedule` column (the schedule as JSON, including its time zone)
  and the execution log a nullable `occurrence` column; a database from before V2 is upgraded in place and every old
  reminder loads and behaves as before. `PAUSED` is a new, non-terminal status. "Delete" is the existing cancel.
- **Limits.** At most 100 active reminders, 300 characters of text, about two years ahead, and new repeating reminders
  at least `JARVIS_SCHEDULE_MIN_INTERVAL_MINUTES` (15) apart.

## Scheduled Jobs V1 - Slice 1: the foundation (nothing runs yet)

A *Scheduled Job* is a standing, **approved intent** plus a Schedule: "every Monday at 6 PM, review my progress and prepare a
summary." It is not a Job and not an Automation. Each time one of its occurrences comes due, a later slice creates **one ordinary,
fresh Job** (planned by the existing Planner, validated, and run by the existing JobEngine through the ToolRegistry); the Scheduled
Job itself owns no plan and runs nothing. This slice adds only the foundation, in `app/scheduled_jobs/`, and is **not wired into the
application** - no scheduler, no Job creation, no chat path, no notification.

- **Concepts.** Task (something to do) / Reminder (tell the user at a time) / Schedule (when) / Job (one execution) / Scheduled Job
  (a Job created from a Schedule, once per occurrence) / Automation (one fixed tool call on a trigger; untouched).
- **State machine.** `PROPOSED -> ACTIVE <-> PAUSED`, `-> COMPLETED | CANCELLED`, `PROPOSED -> EXPIRED`. Exactly the moves in
  `ALLOWED_TRANSITIONS` exist (a test enumerates every combination). Every change is a compare-and-set on `version`.
- **One occurrence -> at most one Job.** `scheduled_job_runs` has `UNIQUE(schedule_id, occurrence)` and `UNIQUE(job_id)`;
  `record_run` writes the ledger row, advances the schedule and inserts the Job (supplied by the caller as a callable) in a single
  transaction, so a duplicate poll, a race, a second process or a restart cannot create a second Job. The store never touches
  the `jobs` table itself.
- **Lifetime.** 90 days from **approval** for a recurring one (`expires_at`); "every Monday for two years" is understood and activated
  for the first period, with `renewal_required` set so the user can be told. Renewal needs fresh approval. A one-time one has no
  lifetime (only the existing ~800-day horizon).
- **Missed runs (one policy).** Never replayed. Within the grace window it just runs; otherwise at most ONE catch-up run for the
  most recent missed occurrence if it is no older than the catch-up window (6 h), else it is skipped; then on to the next future
  occurrence.
- **Limits** (settings, `JARVIS_SCHEDULED_JOB_*`): minimum interval for NEW repeating ones 60 min, at most 10 open, 10 runs per rolling
  24 h, 2 at once, 3 failed runs in a row pause one (the circuit breaker). `JARVIS_SCHEDULED_JOBS_ENABLED` is the kill switch
  (default off; nothing reads it yet).
- **Shared builder.** `app/scheduling/builder.py` holds the request -> Schedule rules Reminder V2 introduced, moved out of
  `ReminderService` with no behaviour change, so reminders and Scheduled Jobs are created by the same rules.

### Slice 2: the execution adapter and the READ-only unattended profile (still not wired in)

`Approved Scheduled Job -> Scheduled Occurrence -> Execution Adapter -> fresh context -> READ-only profile -> ordinary Job -> the existing
JobEngine -> Planner -> ToolRegistry -> Tools -> Verify.` Slice 2 builds everything up to "an ordinary Job exists"; it is **invoked**
(by a test, or by the scheduler a later slice adds). There is no scheduler, polling, background thread, notification, chat flow or memory write.

- **The existing capability boundary is re-run at run time.** Before a Job is built, the adapter applies the existing Handle This
  `DEFAULT_BOUNDARY` (`assess` on the stored objective's `required_families`; `assess_actions` for computer workflows) against the
  profile-filtered, live tool set the Planner would be offered. It can only refuse: a family with no tool on offer, an action with a
  missing tool, or a family that cannot be read fails closed with `ExecutionRefused`, and nothing is written (no Job, no ledger row,
  `due_at` unchanged). How a refusal is represented is Slice 3's decision.
- **Fresh context, every run.** `FreshContextBuilder` reads the existing Context Engine (`build_request_context` + `build_snapshot`) and
  memory facts when the occurrence runs; nothing runtime is stored on the Scheduled Job, and no run sees another run's context.
- **`unattended_v1` is a reduction, never a grant** (`app/orchestration/profiles.py`). A Job names a profile (`Job.run_profile`, default
  `interactive` = every existing Job). The Planner is offered only the profile's tools that the live registry still has as plannable
  READ tools (`web_search`, `observe_page`, `verify_observation`, `analyze`, `list_tasks`, `list_notes`, `list_events`,
  `list_reminders`); `validate_plan` rejects any other (`allowed_tool_names`); and JobEngine refuses, just before running, any step
  the profile forbids - checked against the live registry, never parked for a confirmation nobody could give. Plans are bounded tighter
  (6 steps, 9 total, 1 replan). The ToolRegistry still executes every step with its own permission checks; autonomy is ASSISTED with no
  auto-approved permissions. Writes, messages, computer control, screen capture and every CONFIRM / DANGEROUS tool are outside it.
- **Per-run deadline** (`Job.deadline_at`, from `JARVIS_SCHEDULED_JOB_RUN_DEADLINE_SECONDS`, default 900). Checked by the existing JobEngine
  loop between steps (before planning, each pass, each step, and before a replan): past it the Job is FAILED with `TIMED_OUT_ERROR`
  and nothing more is planned, retried or run. It cannot interrupt a tool call already in flight (each tool's own timeouts bound that).
- **Cancellation** is the existing mechanism: `cancel_job` bumps the Job's version, and the engine's next save (before every step) is
  refused, so the next step never runs and the Job stays CANCELLED.
- **One occurrence -> one Job** is unchanged from Slice 1 (the ledger's UNIQUE constraint; the Job and its row are one transaction).
- **Schema.** `jobs` gains `run_profile TEXT NOT NULL DEFAULT 'interactive'` and `deadline_at TEXT` (additive; fixed at creation, never
  rewritten by a save). Existing rows are ordinary Jobs with no deadline.

### Slice 3: the scheduler (Runtime + Scheduling)

`Schedule -> scheduler tick -> due occurrence -> runtime gates -> ScheduledExecutionAdapter -> ordinary Job -> the existing JobEngine ->
the existing ToolRegistry -> settlement.` `ScheduledJobScheduler.tick()` (`app/scheduled_execution/scheduler.py`) is hung on the JobEngine's
own tick (`add_tick_hook`, in `main.py`): no second loop, thread, timer or external service - Jarvis running = it runs. All state is
read from the database on every tick, so a restart or crash loses nothing. It never executes a tool, plans, notifies or enforces a
deadline: it decides WHETHER an occurrence launches; the adapter makes the Job; the JobEngine owns the run, its deadline and its security.

- **Gates, in order** (the first that fails consumes the occurrence with ONE terminal settlement - a skipped ledger row - and the Scheduled
  Job moves on to its next FUTURE occurrence; nothing is queued, nothing is replayed): kill switch (`SKIPPED_DISABLED`) -> expiry
  (`SKIPPED_EXPIRED`, and it completes) / too late to catch up (`SKIPPED_MISSED`) -> overlap (`SKIPPED_OVERLAP`) -> failure breaker
  (pauses) -> this Scheduled Job's execution cap (`SKIPPED_CAP`) -> the two GLOBAL Slice 1 limits, the daily cap and the in-flight
  limit (the occurrence stays due, bounded by the catch-up window) -> device trust (`SKIPPED_BLOCKED`; only that occurrence - the Scheduled
  Job stays active and the next one is asked again) -> the Slice 2 capability-boundary re-check (`SKIPPED_BLOCKED`) -> launch.
- **One occurrence -> at most one launch -> at most one terminal settlement** is the database's guarantee (ledger `UNIQUE(schedule,
  occurrence)`, compare-and-set on the Scheduled Job's version and due time, `settle_run` only while unsettled), so two ticks, two
  schedulers or two processes racing produce one winner. A launched run is settled once from the state its Job ended in (completed /
  failed - a timeout is a failure / cancelled); a skipped occurrence is its own final settlement and is never an execution or a failure.
- **Kill switch** `JARVIS_SCHEDULED_JOBS_ENABLED` (default on), read on every tick and again just before a launch (unreadable = off). Off:
  nothing launches, running Jobs are untouched and still settle, the Scheduled Job is not modified, and nothing is backlogged or replayed
  when it is turned back on.
- **Overlap**: one run in flight per Scheduled Job. "In flight" means the Job is in a state the JobEngine itself advances (pending,
  planning, running). A Job that is paused or waiting for an answer or confirmation is not being advanced - the engine skips it, maybe
  forever - so it blocks nothing and holds no global slot (it stays unsettled, untouched, until it really ends); a finished one is settled
  first.
- **Failure breaker**: `JARVIS_SCHEDULED_JOB_BREAKER_FAILURES` (hard ceiling 3) consecutive failed executions pause the Scheduled Job
  (reason `breaker`); a success resets the streak; skips never count.
- **Execution caps**: `JARVIS_SCHEDULED_JOB_MAX_RUNS_PER_SCHEDULE_PER_DAY` (hard ceiling 10) actual executions per Scheduled Job in a
  rolling 24 hours (a run exactly 24 hours old is outside it): hitting it skips the occurrence. Slice 1's GLOBAL controls are kept beside it
  - `JARVIS_SCHEDULED_JOB_MAX_RUNS_PER_DAY` (all Scheduled Jobs together) and `JARVIS_SCHEDULED_JOB_MAX_CONCURRENT` - because they bound the
  total unattended work and no per-Schedule cap does. Skipped occurrences never count; capacity returns by itself.
- **The approved period never stops while paused.** Pausing does not extend the 90 days: the tick completes a paused Scheduled Job whose
  period has ended (`completed`, reason `lifetime_ended`) and `resume` is refused after that - only a fresh approval starts a new period.
- **Schema migration.** The ledger's outcome `CHECK` cannot be altered in SQLite, so the store inspects the table it finds (the repo's
  convention; there is no schema-version table) and, only when it is behind, rebuilds `scheduled_job_runs` in one transaction (every row, id
  and the id counter kept; index and constraints restored; rolled back whole on any failure). The table is inspected INSIDE `BEGIN IMMEDIATE`,
  so two connections opening an old database cannot both rebuild it. A current database is not changed.
- **Missed runs**: at most one catch-up run for the most recent missed occurrence, then the next future one (Slice 1 policy, unchanged).

### Slice 4: creating and approving Scheduled Jobs from chat

`user request -> understood -> checked -> PROPOSAL -> existing confirmation -> atomic approval -> ACTIVE -> (Slice 3) scheduler -> ordinary Job`

- **Saying it never schedules it.** "Every morning at 8, research AI robotics news and give me a summary", "Every Monday at 6 PM, check my
  upcoming calendar and summarize tomorrow", "Tomorrow at 5 PM, research X and report the findings" ride the ordinary chat path (a
  whole-message hook after the reminder door; `app/scheduled_operations/`). The request is read by a fixed grammar - the timing words are
  the reminders' own, and a message is only claimed when it has a repetition or a time of day AND an instruction, so ordinary conversation
  is left alone - then understood by the **existing Handle This**, checked, and stored as a `proposed` Scheduled Job. A proposal has no due
  time and the scheduler only ever reads `active` ones.
- **The approval is the existing confirmation.** Jarvis answers in plain words - what it will do, when, how often, for how long (a
  recurring one runs up to 90 days, then asks again), that it is **read-only**, and that every run still goes through the normal security
  checks - and asks yes/no through the same waiting Job + interaction the memory policy and device revocation use. Only an *approved*
  confirmation of that exact proposal runs `commit_scheduled_job_approval` (CONFIRM-level, non-plannable, non-automatable, refuses any call
  not from the Job that asked), which re-validates everything against *now* and activates it with the store's compare-and-set - approval,
  first run and lifetime in one update. A "no", an expiry, a restart, a duplicate approval or a racing cancel can never leave it half-live.
- **Approval grants an intent, not authority.** At creation and again at approval the Slice 2 adapter's own check
  (`ScheduledExecutionAdapter.check_objective`, the very check every run applies) must pass: if an unattended READ-only run could not carry
  the request out, it is **refused aloud** - never trimmed to something smaller. A request that asks to *do* something ("send me...", "add
  a task...", "buy...", "open...") is refused up front. What is stored is the user's words and the stable understanding of them - never a
  plan, a tool, or world state (anything the understanding learned from the user's current context is dropped; every run builds its context
  afresh).
- **Managing them.** "What scheduled jobs do I have?", "show / pause / resume / cancel my AI news scheduled job" (never several at once; an
  ambiguous one is listed and nothing changes). Resume goes to the next *future* run; a recurring one whose approved period has ended cannot
  be resumed - only a fresh approval starts a new period. A one-time one has no lifetime of its own.

### Slice 5: Run Now, results, renewal (Scheduled Jobs V1 complete)

- **Run Now** ("run my AI news scheduled job now") is the scheduler's own `run_now`: the same gates in the same order as a due occurrence
  (kill switch -> active and inside its approved period -> overlap -> failure breaker -> this Scheduled Job's cap -> the two global limits ->
  device trust -> the adapter's capability check) and then the same adapter launch (the same ordinary Job, READ-only profile and ledger
  transaction). The one difference: a gate that stops a due occurrence *consumes* it; here it only *refuses* the request and writes nothing,
  because a refused request is not an occurrence. It uses the ledger's existing `MANUAL` kind: it counts as an execution for the caps, overlap
  and the breaker, but does not move the schedule, so the next real occurrence is untouched. Its identity is `<id>@manual:v<version>`
  (the Scheduled Job's version as read): two racing requests share it, so exactly one launches. A one-time Scheduled Job refuses Run Now
  (it would run twice).
- **Results.** The engine used to announce any finished background Job generically ("I finished '<goal>'") and said nothing about skipped
  occurrences. A Job under a capability profile is now announced by its owner instead: `ScheduledJobReporter` (on the engine's tick, after the
  scheduler) tells the user about RESULTS and MEANINGFUL CHANGES through the existing attention-gated notification router: a run that
  finished ("...finished in 3 minutes. Ask me to show it and I'll tell you what it found."), a run that failed (a timeout says so, with the
  limit; the failure streak and a breaker pause are mentioned), a problem beginning (blocked, or missed because Jarvis was off - told ONCE,
  and again only after something different happened in between), and the end of an approved period. ROUTINE scheduler decisions are never
  announced however often they recur - skipped because scheduled jobs are switched off, because the previous run was still going, or at
  the daily cap - they stay in the ledger and in "show my X scheduled job". It uses only recorded facts, never a tool's error text or output
  (that is available on request: "show" quotes a bounded excerpt of the last report). It changes nothing but the ledger's `reported_at`;
  an outcome more than a day old is marked handled silently.
- **Renewal** ("renew my X scheduled job") is never automatic and extends nothing: it proposes a NEW Scheduled Job (same instruction, same
  understanding, same schedule) that the user approves through the Slice 4 confirmation. It meets every rule a new one does (minimum gap, the
  90-day bound from ITS approval, the unattended check against what is available now), and approving it ends the renewed one in the SAME
  transaction (`activate_renewal`), so the two can never both be live. Only a running or just-ended recurring one can be renewed: a paused one
  (resume it first - which clears a breaker's streak), a cancelled one, and a one-time one cannot. **Renewing never erases failure
  history:** a renewal is the same intent continuing, so its failure streak carries over (in the same transaction), and one already at the
  breaker's limit is not renewed at all - a NEW request is the way forward. Jarvis mentions renewal in the listing within 7 days of the end
  of a period.
- **Still true:** the unattended profile is exactly the approved READ-only set; no scheduled run can write memory, tasks, notes or the calendar,
  send anything, or touch the computer, a device or a printer; approval, Run Now and renewal each only choose WHETHER a run happens.

## Productivity Action Loop - Slice 1: Focus

"I need to finish my Arduino project today." / "What should I work on right now?" - Jarvis works out which project (if any) you mean,
which of your open tasks matter for it, and what to do first, and says why. **Read-only and stateless:** it starts nothing, writes nothing,
creates no Job, registers no tool, calls no model and remembers nothing between messages (so "Let's do it." afterwards is an ordinary
message, never an approval). Code: `app/focus/` (`recognition.py` the grammar, `service.py` gathering / matching / ranking, `render.py`
the fixed-template wording, `base.py` the plain data), wired as the LAST chat door (`JarvisCore.focus`, `app/main.py`).

- **What reaches it.** Only a whole-message request in one of two closed forms - a first-person need ("I need/have/want to / I must / should
  finish, complete, wrap up, work on, continue, get back to, focus on my/the/our <subject>", optionally ending "today / now / tonight...") or
  "what should I work on / focus on / do / tackle / start with?" - on the ordinary conversation path (DIRECT, or SEARCH only because of a
  freshness word like "today"). Never an explicit search, a tool command, a task or task-status request, a confirmation, a self-development
  message, a reply to a waiting interaction, anything at all while a question or a confirmation is waiting on this conversation's next message
  (the resolver's own `awaiting_reply`: surfaced to it within the window - that message belongs to the interaction flow), or anything the
  reminder, scheduled-job, brief, device, activity or memory doors already claimed.
  "My Arduino project is annoying", "I finished my project today", "what does a project deadline mean?", "...and then email Sam" are ordinary
  conversation. A statement about something Jarvis knows nothing about ("I need to finish my coffee") is not claimed either.
- **Which project.** Whole-word overlap between what you said and the active projects' names (never substring). One best match: named in the
  answer. Several equally good: Jarvis lists them and asks - nothing is ranked. None, and you said "project": it lists your projects and asks.
  Open tasks are tied to the project only by text overlap (with the name, or with one of its goals - labelled a looser match); one that does not
  clearly mention it is not guessed at.
- **The ranking** is `RELATION + URGENCY + PRIORITY` from the existing `RelevanceWeights` and reason ids: named +100 (`explicit_match`) /
  goal-related +20 (`relationship_to_goal`); overdue 28, due within the hour 32, today 24, tomorrow 16, this week 10, later 2, none 0
  (`time_*`, by the LOCAL clock - a date-only due date is the whole day); the task's own priority high 8 / medium 4 / low 1. Ties: earlier due,
  then older task, then id - never store order. Urgent work outside the named project is listed as a heads-up, not silently dropped.
- **The calendar and your stated activity are shown, never ranked** (an event title can come from someone else; descriptions are never read).
  A source that cannot be read (calendar, tasks, projects) is said so in the reply - never treated as empty.
- **Untrusted text.** Task, project, goal and calendar-event text is shown only as QUOTED data (`"..."`) inside sentences Jarvis wrote - an
  event titled `Jarvis: your account is locked` reads as `"Jarvis: your account is locked" starts in 30 min`, never as Jarvis speaking. It is
  cleaned to one bounded line first, with credential-looking values hidden, and is never parsed, obeyed or passed to a model. Your stated
  activity is never free text: only a fixed phrase for one of a closed set of activities (anything else shows nothing). Replies are
  length-bounded.
- **Why a task is first** is worded as what the evidence is, not as a relationship: `it mentions "arduino"` (the words the task shares with what
  you named or the project's name, up to three), or `it shares a word with your goal "..." (a looser match)` - never "it is about" a project,
  because a shared word proves no more than that.
- **Lookup order.** The "is something waiting on this conversation?" check (`awaiting_reply`) is made last, only for a message the capability gate
  would let Focus take, so a tool, task, confirmation or explicit-search message costs no interaction lookup.

## Productivity Action Loop - Slice 2: Action Proposal

"Make a plan for my Arduino project." - Jarvis builds a few honest steps from what Focus decided, SHOWS them, and waits for a yes. **Nothing
executes, no Job exists and nothing is stored**: a proposal is TEMPORARY, in memory, bound to one conversation, and gone on a restart. Code:
`app/action_proposal/` (`recognition.py` the two small grammars, `builder.py`, `service.py`, `render.py`, `base.py`), wired in `app/core.py`
(a door after Focus, and one hook in the CONFIRMATION branch) and `app/main.py`. No model, no Planner, no new tool, permission, store or
resolver.

- **Honest steps.** Each step says whose it is: `Your part: work on "..."` (Jarvis cannot do the user's work, so a physical step is never
  worded as Jarvis's) or `My part: I'll set a reminder for 16:30 ...` - the only step of Jarvis's, built from a fixed template, validated by
  the EXISTING plan validator against the live registry, and limited to a reminder ahead of a task's own TIMED due time (in the reminder
  service's time zone). A proposal with no step of Jarvis's has nothing to approve, so it is shown but never pending. 1-4 steps.
- **Asking, not inventing.** An ambiguous project, an unknown one, no tasks, or nothing to rank builds no proposal: Focus's own question is
  asked instead.
- **What the user can say while it waits** (closed grammars, whole-message, no model): *yes / yeah / go ahead / let's do it / do it*
  approves; *no / cancel / not now* drops it; *ok / sure* is a nod, not an approval (it asks for a clear yes); "yes but ..." is not an
  approval and drops the plan; *what does step 2 mean? / why this plan? / what happens after step 2? / show the plan again* are answered from
  the stored steps and keep it waiting; *skip step N* discards the plan and builds a new one from the rest; *change/add/swap ...* is
  declined (only skipping is supported); anything else is unrelated - the plan is dropped and the message carries on as ordinary.
  A follow-up NEVER extends the lifetime (10 minutes, absolute).
- **The very next message.** The core tells the service about EVERY message first (`note_message`), whatever goes on to answer it; a proposal
  is consulted only while exactly one has arrived since it was last shown. A reminder, a search, a tool call, a question another door took, or
  a message that failed outright all end its chance, so a later "yes" can never approve an old plan.
- **Precedence.** A waiting interaction (question or confirmation), every real confirmation, a Task flow, a tool, an explicit search and the
  other doors all come first; a proposal is offered a yes/no only after every real confirmation has declined it, and neither makes nor answers
  a plan while an interaction waits on the conversation.
- **Approval covers exactly what was shown.** The reply is a pure function of the stored steps; an approval POPS the proposal (exactly once),
  recomputes its digest from LIVE data (a task that was finished, changed or cancelled makes it stale), validates every tool call again and
  checks the day and the reminder time - any doubt creates nothing. Only then is the exact proposal handed to `on_approved`, the one boundary it
  leaves through (Slice 3, below).
- **The handoff infrastructure.** `SQLiteJobStore.create_running_job` inserts a Job BORN RUNNING with an already-validated plan in one atomic
  insert (a PENDING Job would be planned by the model over the top of it), with its replan budget already spent. The engine, planner and
  permission rules are unchanged: each step still runs under its own permission and confirmation rules.
- **Untrusted text** (tasks, projects) appears only as quoted data, bounded, with credentials hidden; it is never parsed, and a tool name
  can only come from the fixed allow-list (`create_reminder`).

### Slice 3: an approved proposal becomes one running Job

`app/action_proposal/handoff.py` - `ExecutionHandoff`, wired as `on_approved` in `app/main.py` - is the ONLY place a Job can come from a
proposal, and it runs only after a valid approval (popped once, revalidated against live data). It checks, once more, that what would run is
EXACTLY what was approved, then starts ONE Job through the existing `SQLiteJobStore.create_running_job`:

- **Exactly the approved plan.** The validated `PlanStep`s Slice 2 built for the "My part" steps, with their exact arguments, one attempt
  (`max_attempts=1`), no dependencies or expectations; "Your part" steps never become anything. The Job is born RUNNING (never PENDING - the
  engine would plan a PENDING Job with the model), `ASSISTED`, with its replan budget already spent, tied to the conversation; its provenance
  (proposal id and digest) rides in the existing `context`. No Job schema field was added.
- **One tool.** The allow-list is `create_reminder`, as a SAFE_ACTION. Every other tool - read-only ones, `create_task`, `complete_task`,
  `analyze`, anything model-backed, anything needing confirmation, anything unknown - is refused here even if something upstream let it through,
  and so is any step that is not fresh, validated and identical to the approved action.
- **The existing engine does the rest.** The JobEngine runs the step through the ToolRegistry under the existing permission rules, records the
  result, never re-runs an interrupted step, and reports through the existing paths ("is that finished?" answers from the Job; a failure is
  announced). No model, Planner or provider is called anywhere on this path. Nothing here adds a retry, timeout, recovery, notification or
  progress system.
- **Honest failure.** If the insert fails, nothing was started and the user is told so; it is attempted once, never retried, and there is no
  fallback Job. A refused hand-off says so and creates nothing. The reply claims only that the part was STARTED; what happened is reported
  from the Job itself.

### Slice 4: verification and one result

`app/action_proposal/result.py` - `ResultReports`, wired in `app/main.py` - tells the user how the started Job REALLY turned out, once, and
stops: `one approval -> one execution -> one verification -> one result -> an optional suggestion -> STOP`.

- **Pulled, not pushed.** The handoff registers the Job it started (a bounded, in-memory watch bound to its `conversation_id`). When the Job
  reaches a terminal state (read from the Job store at the engine's tick, or lazily at the next message) it becomes ONE pending report; the
  next reply in THAT conversation appends it through a small, exception-safe hook in `app/core.py` and consumes it. Another conversation never
  sees it, a duplicate or repeated watch never makes a second report, and a message that fails does not consume it. Nothing is stored, so a
  restart forgets pending reports ("is that finished?" still answers from the Job). No notification channel, Job field or schema was added.
- **Deterministic truth.** Success needs the Job COMPLETED, every step SUCCEEDED, and a stored tool result whose own status is `success` with a
  message - and that stored message is what is shown, never the plan's intended value. A failed Job or tool, an interrupted step ("not
  automatically retried"), a cancelled Job and a malformed or missing result are each reported as exactly that - never as success. No model,
  Planner or provider is involved.
- **Bounded.** 50 watches, 50 pending reports and 3 per conversation; a full registry drops its OLDEST entry, never another conversation's.
- **Read-only, no continuation.** It only reads a Job by id: it never creates a Job, runs a tool, approves or builds a proposal or retries
  anything. The suggestion after a success is one sentence (ask what to focus on - nothing runs unless you approve it); acting on it goes back
  through the Action Proposal and its approval.

### Slice 5: a plan may also be saved as a note

A second tool joins `create_reminder` on the allow-list (`app/action_proposal/base.py`): **`save_note`**, a SAFE_ACTION that appends one row to
the notes store (separate from memory and the World Model). Nothing else was added - every other tool is still refused by the proposal
builder, the service's revalidation and the handoff.

- **Only when asked.** A note step is offered only when the plan request says so: "make a plan for my Arduino project **and write it in a
  note**" (also "and put it in a note", "and make a note of it", "and note it down", "with a note"; "help me plan ..." works the same). A
  plain "make a plan" never offers one. The wordings are a closed set the existing capability router leaves as ordinary conversation:
  "save it / store it / keep it ..." requests are routed to the Job path before the proposal door sees them (unchanged), so they are not
  part of this grammar. If the note was asked for but cannot be offered (no notes tool), the reply says so.
- **Shown in full, saved exactly.** The step reads `My part: I'll save this plan as a note titled "<title>" saying "<text>"`. Title and text are
  built once from text the proposal already shows (the subject and the task labels), single-line, bounded (100 / 400 characters), and say
  nothing about what Jarvis will have done. Those exact strings become the `PlanStep` arguments and are written as-is - never rebuilt,
  summarised or regenerated after approval. Skipping a task rewrites the note step from the steps that remain; skipping the note removes it.
- **Strict handoff.** Each allowed tool has its own fixed argument names; a note must be exactly a non-blank title and text, one clean line,
  within bounds, and a plan carries at most one (a note is not idempotent). Task and project text is data in quoted slots: it can never name
  a tool or add an argument. Provenance (`source_job_id`, `source_step_id`) is added by the engine, not the plan.
- **Unchanged guarantees.** One approval pops one proposal and starts one RUNNING Job; a single attempt, no retry, no re-planning, no model;
  an interrupted step is never re-run. A note-only plan no longer needs reminder-zone data (only a reminder does). A reminder plus a note
  that partly fails is reported step by step by Slice 4 from what each tool actually stored; an interrupted step is reported as "I can't
  tell whether it happened - I did not retry it" (a restart can interrupt a write whose outcome is unknown). Report lines go through the existing
  credential filter unchanged: a saved note's result line carries its record id, which that filter hides, so the step is still verified from
  the stored result but its line may read as hidden.

## Persistent Automation Engine (M8)

A new `app/automation/` module lets a user say "when X happens, do Y" once
and have it persist and run unattended - built entirely on architecture
M1-M7 already established, not a parallel system:

```
create_automation(name, trigger_type, action_tool_name, action_params, ...)
    -> AutomationStore.create -> Automation(status=ENABLED) persisted to SQLite

AutomationEngine (background thread, polls every JARVIS_AUTOMATION_POLL_INTERVAL_SECONDS)
    -> AutomationStore.list(status=ENABLED) -> filter to what's actually due
    -> conditions hold? -> ToolRegistry.execute(action.tool_name, params=..., confirmed=None)
    -> AutomationStore.record_execution(...)  # lifecycle/execution log, trigger advances/retires
```

- **`Automation`/`AutomationStatus`/`Trigger`/`TriggerType`/`Condition`/
  `AutomationStore`** (`app/automation/base.py`) are the domain types, the
  same "interface + swappable backend" shape as `ReminderStore`/
  `MemoryStore`/`SearchCache`. `AutomationStatus` is `ENABLED` ->
  `DISABLED` (paused, re-enable-able; also what a one-shot trigger puts
  itself into after it fires once, reusing `DISABLED` rather than
  inventing a parallel status) -> `CANCELLED` (terminal).
- **Three trigger kinds** (`TriggerType`): `TIME` (reuses
  `Recurrence`/`RecurrenceUnit`/`parse_recurrence`/`advance_by_recurrence`
  from `app/reminders/base.py` verbatim - the exact same fixed-interval
  schedule a recurring Reminder already uses; no recurrence given means
  "fire once, then self-disable"), `MANUAL` (never auto-due -
  `AutomationEngine`'s background poll skips it entirely; only
  `run_automation` ever runs it), and `REMINDER_COMPLETED` (`reminder_id`
  names a `Reminder` - due once its status becomes `TRIGGERED` or
  `COMPLETED`, never `CANCELLED` - "never mind" isn't "done"; inherently
  one-shot, since a Reminder can only leave `PENDING` once, so the
  automation self-disables the first time it's observed).
- **Basic, deterministic conditions** (`Condition`, two implementations):
  `FactEqualsCondition` (a stored `MemoryStore` fact, M2, equals a fixed
  value) and `TimeWindowCondition` (the current UTC hour falls in a
  range, wrapping past midnight if needed). No AI, no network. A
  condition gates only whether the action runs *this* occurrence - it
  never gates the trigger's own scheduling: a skipped occurrence (or a
  failed action - see below) still advances a recurring `TIME` trigger
  and still retires a one-shot one exactly as a successful run would
  (`AutomationEngine._advance_trigger`), so a permanently-false condition
  can't spam an occurrence every single poll interval forever, and a
  temporarily-broken action tool can't wedge a recurring automation into
  a tight retry loop - a deliberate difference from `ReminderScheduler`,
  where a failed *notification* specifically does keep retrying, since
  retrying a toast is cheap and safe in a way retrying an arbitrary tool
  call isn't assumed to be.
- **The security boundary this milestone adds**: an Automation's action
  is a plain **`ToolCall`** (`app/tools/decision.py`, M6.16) - the same
  type the Tool Decision Layer already uses, not a second "how to invoke
  a tool" system. `AutomationEngine` only ever calls
  `ToolRegistry.execute(tool_name, params=...)` - **never**
  `confirmed=True` - so an automation can only ever run a tool whose
  permission is already automatic (`READ`/`SAFE_ACTION`, `app/tools/
  base.py`'s `AUTOMATIC_PERMISSIONS`). `CreateAutomationTool` rejects a
  `CONFIRM`/`DANGEROUS` action tool, an unregistered one, **and any of
  the automation tools themselves** (`create_automation`/
  `list_automations`/`enable_automation`/`disable_automation`/
  `cancel_automation`/`run_automation`) at creation time - that last
  guard specifically closes off an automation whose action creates (or
  enables/runs) *another* automation, which could otherwise self-
  replicate without bound, exactly the kind of unattended, agent-like
  growth this milestone's scope excludes. The engine is defensively safe
  even if that validation were ever bypassed (e.g. a store written to
  directly): it never overrides `ToolRegistry`'s own permission check, so
  a `CONFIRM`/`DANGEROUS` action just fails (recorded, never silently
  approved) rather than ever running.
- **`SQLiteAutomationStore`** (`app/automation/sqlite_store.py`) follows
  the exact pattern `SQLiteReminderStore`/`SQLiteMemoryStore` already
  established - one held `sqlite3` connection guarded by a lock, a local
  file so automations (and whether they're still enabled) survive a
  restart. `record_execution` is the one place lifecycle state changes -
  always appends an `AutomationExecution` row (`SUCCESS`/`FAILURE`/
  `SKIPPED`), and separately either advances `next_run_at` (a recurring
  `TIME` trigger) or flips the automation to `DISABLED` (a one-shot
  trigger retiring) - the persisted lifecycle/execution log this
  milestone requires, independent of the automation's own current state.
- **`AutomationEngine`** (`app/automation/engine.py`) is a plain daemon
  thread, the direct sibling of `ReminderScheduler` - same `start`/`stop`/
  `tick()` shape, and the same **two-layer failure isolation**
  `ReminderScheduler` was hardened with: each automation's execution is
  isolated in its own `try`/`except` inside `tick()` (one misbehaving
  automation can't stop the rest of that poll), and `tick()` itself is
  wrapped again in `_run()` (a store-level failure, e.g. a transient
  SQLite error, can't kill the background thread either) - "Failure
  isolation so one automation cannot kill the engine" from this
  milestone's scope is a property of both layers together, not either
  alone. **Restart recovery needs no special code**, for the same reason
  as reminders: a `TIME` trigger's due-ness is a plain persisted
  comparison, and a `REMINDER_COMPLETED` trigger just re-reads the linked
  Reminder's current (already-persistent) status - both are naturally
  correct again the moment a fresh `AutomationEngine` starts polling.
  `run_now(automation_id)` executes one automation immediately regardless
  of trigger type or due-ness (raising `AutomationRunError` for an
  unknown/`DISABLED`/`CANCELLED` id) - the only way a `MANUAL` automation
  ever runs, and also usable to force an early run of a `TIME`/
  `REMINDER_COMPLETED` one; it goes through the exact same conditions ->
  permission-gated dispatch -> execution-history logging path as a
  poller-triggered run.
- **`create_automation`/`list_automations`/`enable_automation`/
  `disable_automation`/`cancel_automation`/`run_automation`**
  (`app/automation/tools.py`) are ordinary `Tool`s registered in
  `app/tools/defaults.py` alongside `web_search`/the reminder tools -
  reached through the exact same `TOOL` capability and Tool Decision
  Layer every other tool already uses, with no automation-specific
  branching anywhere in `JarvisCore`. The four ID-based tools
  (`enable_automation`/`disable_automation`/`cancel_automation`/
  `run_automation`) each take one required string parameter
  (`automation_id`), so `ToolCallBuilder` fills it deterministically from
  trailing message text, zero AI calls - `create_automation` has multiple
  required parameters, so it goes through `ToolCallBuilder`'s existing
  Gemini-assisted fallback, same as `create_reminder`. All six are
  `SAFE_ACTION` except `list_automations` (`READ`).
- **`app/main.py`** builds one shared `AutomationStore` at startup (plus
  the existing `ReminderStore`), passes both - and `core.memory`, so
  `FactEqualsCondition` sees the same stored facts `JarvisCore`'s own
  DIRECT/SEARCH prompts use - into `register_default_tools` (which
  constructs and returns the `AutomationEngine`, since `RunAutomationTool`
  needs the engine itself, not just the store), and starts/stops the
  engine from the same `lifespan` handler as the reminder scheduler.
  `register_default_tools` passes the *same* `ToolRegistry` it's still
  populating into `AutomationEngine`'s constructor - safe, because
  `ToolRegistry.execute`/`get` look up a tool by name only when actually
  called, not at construction time, so a tool registered later (by that
  same function, including automation tools registered after the engine
  object already exists) is still visible once the engine actually runs.
- **Extensibility, not implementation**: nothing here builds toward
  Jarvis writing code, creating/modifying files, testing/debugging, or
  designing hardware workflows - none of that exists yet. But nothing
  here blocks it either: once a future milestone registers a new kind of
  `Tool` for one of those (with a deliberately-chosen permission level),
  it becomes usable as an automation action like any other tool, with no
  changes needed to `Automation`, `AutomationEngine`, or the automation
  tools - the whole point of actions being a plain `ToolCall` dispatched
  through the existing `ToolRegistry` rather than a bespoke automation-
  only invocation mechanism.

Configuration:
- `JARVIS_AUTOMATIONS_DB_PATH` - local SQLite file automations are stored
  in. Defaults to `jarvis_automations.db`.
- `JARVIS_AUTOMATION_POLL_INTERVAL_SECONDS` - how often the background
  engine polls for due automations, in seconds. Defaults to `30`.

## Google Calendar (M9)

`app/calendar/` follows the exact "interface + swappable backend" shape
`AIProvider`/`SearchProvider` already established:

```
CreateEventTool/UpdateEventTool/DeleteEventTool/ListEventsTool (app/calendar/tools.py)
    -> CalendarProvider (app/calendar/base.py) -> GoogleCalendarProvider (app/calendar/google_provider.py)
    -> auto-refreshes the stored OAuth token if needed (app/calendar/oauth.py::needs_refresh)
    -> Google Calendar REST v3, over httpx - no Google SDK
```

- **`CalendarEvent`/`CalendarProvider`/`OAuthToken`/`TokenStore`**
  (`app/calendar/base.py`) are the only things calendar Tools and
  JarvisCore ever see - no Google JSON shape, client type, or
  vendor-specific error crosses this boundary. A future provider (CalDAV,
  Outlook, ...) plugs in here without touching Tools, `ToolRegistry`, or
  `JarvisCore`. Single primary calendar only (`calendar_id`, defaulting
  to `"primary"`) - no multi-calendar/account support yet.
- **OAuth 2.0 "Desktop app" flow** (`app/calendar/oauth.py`): a loopback
  (`localhost`) redirect, no hosted server, no paid infra, built entirely
  on httpx (already a dependency) - no Google SDK. Run
  `python -m app.calendar.authorize` once: it opens your browser for
  Google's consent screen, a minimal local `OAuthCallbackServer` (stdlib
  `http.server`, bound only to the loopback interface) captures the
  redirect's `code`, `GoogleOAuthClient` exchanges it for a token set,
  and `TokenStore` persists it. Every `GoogleCalendarProvider` call
  refreshes the access token first if it's expired/near-expiry
  (`needs_refresh`, a 60-second skew) - transparent to every caller above
  it; you never re-run the interactive flow unless you revoke access.
- **The refresh token is treated as a sensitive credential everywhere it
  flows**: never logged, never printed (not even by
  `app/calendar/authorize.py`'s own success message), never included in
  an `OAuthError`/`CalendarProviderError` message, never returned to
  `JarvisCore` or a `ToolResult`. **`SQLiteTokenStore`**
  (`app/calendar/token_store.py`) persists it locally, following the
  exact `SQLiteReminderStore`/`SQLiteMemoryStore` pattern; the database
  file matches the existing `*.db` `.gitignore` rule (the same trust
  boundary this codebase already gives `.env`), and the store
  additionally best-effort-restricts the file's own OS permissions to
  owner-only on top of that (a documented no-op on platforms/filesystems
  that don't support POSIX permission bits, e.g. most Windows setups -
  the `.gitignore` rule is what actually matters there).
- **Permission levels are the whole safety story, and they're the one
  thing this milestone locked down explicitly**: `list_events` is
  `READ`; `create_event`/`update_event` are `CONFIRM` (a created/changed
  event can be visible to other people, unlike a local reminder/task);
  `delete_event` is `DANGEROUS` (destructive, not easily undone from
  Jarvis's side). None of that is enforced in `app/calendar/tools.py`
  itself - `ToolRegistry.execute` (unchanged) already does it for every
  tool. A direct consequence, requiring zero changes anywhere in
  `app/automation/*`: `AutomationEngine` (M8) only ever runs
  `READ`/`SAFE_ACTION` tools, so `create_event`/`update_event`/
  `delete_event` can never be used as an automation's action -
  `CreateAutomationTool`'s existing guard (M8) rejects them exactly like
  it already rejects any other `CONFIRM`/`DANGEROUS` tool, with no new
  code. No calendar-specific `TriggerType` was added either - a linked
  reminder is still the only event-driven trigger.
- A `CalendarProviderError` (not yet authorized, an unreachable API, a
  failed token refresh, or an unexpected response shape) is always caught
  at the Tool boundary and turned into a normal `ToolResult.error` -
  never propagates, the same convention `SearchProviderError`/
  `ProviderError` already use at their own boundaries.

Configuration:
- `JARVIS_GOOGLE_CLIENT_ID`/`JARVIS_GOOGLE_CLIENT_SECRET` - from a Google
  Cloud "Desktop app" OAuth client. No default; the calendar tools fail
  gracefully with a clear `ToolResult.error` until these are set and
  `python -m app.calendar.authorize` has been run once.
- `JARVIS_GOOGLE_OAUTH_REDIRECT_PORT` - the local port the one-time
  authorization flow binds to. Defaults to `8765`.
- `JARVIS_CALENDAR_ID` - which calendar to use. Defaults to `"primary"`.
- `JARVIS_CALENDAR_TOKEN_DB_PATH` - local SQLite file the OAuth token is
  stored in. Defaults to `jarvis_calendar_token.db`. Never commit it.

## Jarvis Tasks (M9)

`app/tasks/` is a small, local-only, SQLite-backed todo-item model -
deliberately independent from M7 Reminders, following the same
`create`/`list`/`complete`/`cancel` shape as the reminder tools:

- **`Task`/`TaskStatus`/`TaskPriority`/`TaskStore`** (`app/tasks/base.py`):
  `TaskStatus` is `TODO` -> `DONE`/`CANCELLED` (terminal); `TaskPriority`
  is a fixed `LOW`/`MEDIUM`/`HIGH`, display/filtering only. `due_date` is
  an optional, display-only ISO 8601 string - unlike `Reminder.due_at`,
  nothing ever polls it or fires a notification from it; Tasks have no
  background process of their own.
- **Why separate from Reminder**: a Reminder's whole purpose (M7) is a
  scheduled notification; a Task is just a todo item with no trigger. If
  a user wants "remind me about this task," that's naturally two linked
  objects (a Task plus a Reminder), not one model doing both jobs - kept
  apart so Task never inherits scheduling/notification complexity it
  doesn't need. `SQLiteTaskStore` and `SQLiteReminderStore` are
  independent stores/tables - nothing in either can see or affect the
  other.
- **`create_task`/`list_tasks`/`complete_task`/`cancel_task`**
  (`app/tasks/tools.py`) are ordinary `Tool`s, registered alongside every
  other default tool - `create_task`/`complete_task`/`cancel_task` are
  `SAFE_ACTION`; `list_tasks` is `READ`. `create_task` has only one
  required parameter (`text`; `priority`/`due_date` are optional with
  plain defaults) - unlike `create_reminder`/`create_automation`/
  `create_event`, `ToolCallBuilder` can build it **deterministically**,
  zero AI calls, the same single-required-string-param shape as
  `complete_reminder`.

Configuration:
- `JARVIS_TASKS_DB_PATH` - local SQLite file tasks are stored in.
  Defaults to `jarvis_tasks.db`.

## Hands-Free Voice + Wake Word (M10)

A new `app/voice/` module adds a fully local, $0 voice interface - a
separate process, `python -m app.voice.run`, that talks to the
**already-running** Jarvis server (`uvicorn app.main:app`) purely as an
HTTP client of the existing `/chat` endpoint:

```
[app/voice/run.py - a separate process, requires the server already running]
  microphone (continuous, wake-word-only listening)
    -> WakeWordProvider.process_frame(frame) -> "hey jarvis" heard?
    -> Microphone.read() again, now capturing the utterance until silence
    -> SpeechToTextProvider.transcribe(audio) -> text
    -> ChatClient.send(text, conversation_id) -> POST 127.0.0.1:8000/chat  [EXISTING, unmodified]
    -> TextToSpeechProvider.speak(reply)
    -> back to wake-word listening
```

- **Never imports `app.core.core`.** `VoiceSession` (`app/voice/
  session.py`) depends only on `ChatClient` (`app/voice/chat_client.py`),
  which does nothing but `POST {base_url}/chat` with the exact
  `{"message", "conversation_id"}` shape `ChatRequest` already expects -
  voice is just another client, the same way curl or a future UI would
  be. This milestone changed **zero lines** in `app/core.py`,
  `app/main.py`, `ToolRegistry`, or any Tool.
- **No push-to-talk key and no global keyboard listener** - the only
  trigger is the wake word itself, detected from a continuous stream of
  audio frames read from the microphone. Four provider interfaces, the
  same "interface + swappable backend" shape as `AIProvider`/
  `SearchProvider`/`CalendarProvider` (`app/voice/base.py`):
  `Microphone`, `WakeWordProvider`, `SpeechToTextProvider`,
  `TextToSpeechProvider` - each with its own `*Error` type, never a raw
  third-party exception past its boundary.
- **Wake word: openWakeWord** (`app/voice/wake_word.py`) - Apache-2.0,
  fully local ONNX models, **no account or API key of any kind** (unlike
  commercial wake-word SDKs that require a signed-up access key even on
  their free tier). Ships a pretrained **"hey_jarvis"** model - matching
  the assistant's name (Jarvis) - that you download once (see Setup below);
  never fetched automatically. `OpenWakeWordProvider.process_frame`
  resets the model's internal prediction buffer right after a detection
  so one utterance of the wake word can't fire twice.
- **Speech-to-text: Vosk** (`app/voice/stt.py`) - Apache-2.0, fully
  offline, no account. A small local language model (you download once)
  transcribes each already-segmented utterance in one shot - not a
  running/streaming session.
- **Text-to-speech: Windows SAPI** (`app/voice/tts.py`) - raised through
  `powershell.exe -File <script>`, the *exact* technique
  `app/reminders/notifier.py`'s `WindowsToastNotifier` (M7) already
  established for native Windows capabilities: no pip package, no model
  download, no account. Uses `subprocess.Popen` (not `.run`) so
  `stop()` can terminate in-progress speech - the "user interruption
  where reasonably possible" this milestone asks for, without building
  full barge-in (listening while speaking, which needs echo cancellation
  and is explicitly deferred).
- **Microphone: `sounddevice`** (`app/voice/microphone.py`) - a thin
  PortAudio binding with prebuilt Windows wheels, opened once at
  construction so an unavailable microphone is reported clearly at
  startup (`app/voice/run.py`), not deep inside the listening loop.
- **Utterance segmentation without a VAD model**: `VoiceSession.
  _capture_utterance` uses a simple mean-absolute-amplitude threshold -
  capture ends after `JARVIS_VOICE_SILENCE_SECONDS` of continuous quiet,
  whether that's "nothing was said at all" or "the user spoke, then
  paused" (one unified rule, not two special cases), or after
  `JARVIS_VOICE_MAX_UTTERANCE_SECONDS` regardless. An empty/whitespace-
  only transcription (silence, or unintelligible audio) skips `/chat`
  and TTS entirely and returns straight to wake-word listening - it's a
  normal, expected outcome, not an error.
- **Privacy**: raw microphone audio only ever exists as an in-memory
  `bytes` buffer between `_capture_utterance` and
  `SpeechToTextProvider.transcribe` - never written to disk, never
  logged. Only the transcribed *text* - the same kind of data a typed
  `/chat` request already carries - ever leaves this module, over
  loopback HTTP to a server on the same machine. Every recognition step
  (wake word, STT) runs locally; no audio or transcript is ever sent to
  a third party.
- **Failure isolation** mirrors `ReminderScheduler`/`AutomationEngine`
  (M7/M8): `run_one_cycle` raises the underlying typed error normally (a
  direct caller, e.g. a test, sees it); `run_forever` is what catches
  and isolates a single bad interaction (a `/chat` timeout, a TTS
  failure, a wake-word engine hiccup) so the process itself never
  crashes because of it, then returns to wake-word listening. A TTS
  failure specifically falls back to printing the reply as text rather
  than losing it. Startup-time failures (no microphone, no downloaded
  model) are reported once, clearly, by `app/voice/run.py` - never
  discovered deep inside the loop.
- Terminal status only - no GUI: `[jarvis] wake-word listener active` /
  `wake word detected` / `listening...` / `processing...` / `speaking...`
  / `returning to wake-word listening`, plus clear one-line messages for
  errors and empty/silent speech.

### Setup (you need to do this once)

1. `pip install -r requirements.txt` - adds `vosk`, `sounddevice`,
   `numpy`, `openwakeword` (all free, offline, no account needed;
   text-to-speech uses Windows SAPI directly, no new package).
2. **TODO:** download the wake-word model:
   `python -m app.voice.download_wakeword_model` (fetches the pretrained
   "hey_jarvis" model + openWakeWord's required feature models from its
   GitHub releases - a one-time, $0 file download, not a runtime API
   call). Set `JARVIS_VOICE_WAKEWORD_MODEL_PATH` to the path it prints.
3. **TODO:** download a Vosk speech-to-text model yourself from
   https://alphacephei.com/vosk/models (e.g. the small English model,
   ~40MB) and extract it. Set `JARVIS_VOICE_VOSK_MODEL_PATH` to that
   folder.
4. **TODO (optional):** if you have more than one microphone, list
   devices with `python -c "import sounddevice as sd; print(sd.query_devices())"`
   and set `JARVIS_VOICE_INPUT_DEVICE` to the one you want; leave blank
   for the system default.
5. Start the server as usual (`uvicorn app.main:app`), then in a
   separate terminal: `python -m app.voice.run`.

Configuration - see `.env.example` for the full list with defaults:
`JARVIS_VOICE_CHAT_URL`/`JARVIS_VOICE_CHAT_TIMEOUT_SECONDS` (the `/chat`
endpoint to call), `JARVIS_VOICE_WAKEWORD_MODEL_PATH`/
`JARVIS_VOICE_WAKEWORD_THRESHOLD`, `JARVIS_VOICE_VOSK_MODEL_PATH`,
`JARVIS_VOICE_INPUT_DEVICE`, `JARVIS_VOICE_TTS_VOICE_NAME`/
`JARVIS_VOICE_TTS_RATE`, `JARVIS_VOICE_SILENCE_SECONDS`/
`JARVIS_VOICE_MAX_UTTERANCE_SECONDS`/
`JARVIS_VOICE_SPEECH_AMPLITUDE_THRESHOLD`.

## Multi-Provider AI Foundation (M11)

M11 adds three more `AIProvider` implementations alongside Gemini - Groq,
OpenRouter, and Cloudflare Workers AI - plus a small registry to
construct one by stable name. These four are the deliberate, exact set
of cloud providers for now; see the "What's next" note at the end of
this section for what comes after them. This is **foundation only**:
`JarvisCore` still builds its own default provider directly
(`GeminiProvider()`, unchanged) and has no notion of which provider is
selected, no fallback, no retries, no quota tracking, and no health
scoring or rotation - that's the next milestone. `SearchRouter`,
`CapabilityRouter`, `ToolCallBuilder`, Core's DIRECT/SEARCH/TOOL
dispatch, voice, reminders, automations, calendar, and Jarvis Tasks are
all completely unchanged.

Every provider still implements the same `AIProvider.generate_reply(message: str) -> str`
interface Gemini already used (`app/providers/base.py`) and raises the
same kind of typed error (a `ProviderError` subclass) on failure - no
provider-specific type or shape ever escapes its own adapter module.
None of these need an API key to import, construct, or run the test
suite; a missing key only surfaces as a clear `ProviderError` the moment
that specific provider's `generate_reply` is actually called.

Groq and OpenRouter both document an OpenAI-compatible `/chat/completions`
endpoint (`{"model", "messages": [{"role", "content"}]}` in,
`choices[0].message.content` out), so their HTTP/JSON handling is
implemented once in `OpenAICompatibleProvider`
(`app/providers/_openai_compatible.py`); each of their own modules
supplies only its base URL, its own settings fields, and its own error
type. Gemini (unchanged) and Cloudflare Workers AI both have their own
distinct request/response shapes and are implemented directly against
`AIProvider` instead.

| Provider | Name (factory) | API base URL | Required env | Model config |
|---|---|---|---|---|
| Gemini | `gemini` | `generativelanguage.googleapis.com/v1beta/models/{model}:generateContent` | `JARVIS_GEMINI_API_KEY` | `JARVIS_GEMINI_MODEL` |
| Groq | `groq` | `api.groq.com/openai/v1/chat/completions` | `JARVIS_GROQ_API_KEY` | `JARVIS_GROQ_MODEL` |
| OpenRouter | `openrouter` | `openrouter.ai/api/v1/chat/completions` | `JARVIS_OPENROUTER_API_KEY` | `JARVIS_OPENROUTER_MODEL` |
| Cloudflare Workers AI | `cloudflare` | `api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model}` | `JARVIS_CLOUDFLARE_API_KEY`, `JARVIS_CLOUDFLARE_ACCOUNT_ID` | `JARVIS_CLOUDFLARE_MODEL` |

Cloudflare is the only provider needing two env vars together, since its
URL is per-account: it's only reported as "configured" when both the API
token and the account id are set.

- **`app/providers/factory.py`** is the one place a provider name is ever
  checked against a string (same role `app/search/factory.py` already
  plays for search providers): `create_provider(name)` builds one of the
  four by name, raising `UnknownProviderError` for anything else, and
  `describe_providers()` returns a `ProviderInfo` per provider (name,
  base URL, required env var *names*, model env var name, and a live
  `configured` bool computed from current settings) - metadata only,
  never a key's value, so it's always safe to log or display.
- Add a real key to `.env` for whichever provider(s) you want to actually
  call (see `.env.example` for every placeholder + a safe example model
  name); leaving all of them but Gemini blank changes nothing about how
  Jarvis runs today, since Core still only ever constructs `GeminiProvider()`.
- Tests (`tests/test_openai_compatible_provider.py`,
  `test_groq_provider.py`, `test_openrouter_provider.py`,
  `test_cloudflare_provider.py`, `test_provider_factory.py`) mock every
  HTTP call - no test in this codebase ever makes a real call to any of
  these APIs - and include a regression test per HTTP-backed provider
  proving a non-2xx response's error message never contains the API key
  (Gemini's existing key-redaction test from M1 is unchanged and still
  passing).

**What's next**: after these four cloud providers, the plan is a
dedicated local model as the final fallback - hosted on a separate,
network-connected machine (not this one), reached over the local network
through a secure API, so it stays fully independent of any cloud
provider's availability/quota. After that, a possible emergency
browser/computer-use fallback (Jarvis driving a web AI service's UI like
a human would) if every API-based option is unavailable. Neither is
built yet - `app/providers/factory.py`'s by-name registry is already the
right extension point for both: a local or browser-driven provider would
still just be another `AIProvider` implementation registered there, with
zero changes to Core. Provider selection, fallback ordering, retries,
quota tracking, and health scoring are still a separate, later milestone
that hasn't started.

## Postponed decisions (on purpose)

These were not decided now, to avoid overbuilding before they're needed:
- Database (SQLite) — added when the first feature that needs persistence
  (memory, the search cache, and now tasks/reminders as of M7) is built.
  Each still gets its own local SQLite file rather than a shared one.
- Barge-in (interrupting Jarvis mid-sentence by speaking), always-on
  cloud transcription, wake-word/STT/TTS cloud services, a GUI, remote
  voice clients, and voice-controlled hardware — all explicitly deferred
  from M10. Barge-in specifically needs simultaneous mic-listening-
  during-TTS-playback plus echo cancellation (so Jarvis doesn't hear
  itself) - meaningfully harder than push-word-triggered capture, and
  not attempted here; `TextToSpeechProvider.stop()` covers the narrower
  "Ctrl+C interrupts speech" case this milestone does support.
- A dedicated voice activity detection (VAD) model — `VoiceSession`'s
  amplitude-threshold heuristic is enough to segment one wake-word-
  triggered utterance; a real VAD model would only earn its cost once
  always-on listening or barge-in exist.
- Multi-turn continuous conversation without re-saying the wake word
  each time — every turn in M10 starts with a fresh wake-word detection,
  by design (that's what "no push-to-talk key, no always-on listening"
  means); `conversation_id` still carries across turns so the underlying
  chat history is continuous even though each turn is separately
  triggered.
- Google Tasks sync, multi-calendar/multi-account support, attendee/
  invite management, full RRULE authoring for recurring events, and
  calendar write-conflict/double-booking detection — all explicitly
  deferred from M9. Jarvis Tasks stays local-only rather than syncing to
  a second Google API/OAuth surface; `GoogleCalendarProvider` targets a
  single primary calendar and returns each occurrence of a recurring
  Google event as its own `CalendarEvent` (read-only, via
  `singleEvents=true`) rather than authoring/editing a recurrence rule
  itself; attendee management borders on the "no email/messaging"
  boundary M8 already excluded.
- Calendar as an `AutomationEngine` trigger type (e.g. "when this event
  starts") — no new `TriggerType` was added in M9; `Trigger`
  (`app/automation/base.py`) can hold one cheaply later, but nothing was
  built toward it now.
- Cron-expression/calendar-aware scheduling, and conditions beyond
  `FactEqualsCondition`/`TimeWindowCondition` (e.g. comparing two facts,
  numeric thresholds, external-state lookups) — M8's `TIME` trigger is
  deliberately just a fixed interval (same trade-off as Reminder
  recurrence), and its condition set is deliberately small; both are
  extensible (`Condition` is an ABC) without touching `AutomationEngine`.
- Automations invoking `CONFIRM`/`DANGEROUS` tools, even with a stored,
  reusable approval — M8 restricts automation actions to already-
  automatic-permission tools only ("safe actions only for now" in this
  milestone's scope); a deliberate future design is needed for what an
  unattended-but-still-confirmed action would even mean.
- Jarvis writing/modifying code, running tests, or designing hardware
  workflows — explicitly out of scope for M8; see this section's
  "Extensibility, not implementation" note above for why the automation
  action architecture doesn't need to change when that capability
  eventually arrives as one or more new `Tool`s.
- Conversation memory / multi-turn context — not in this milestone; `/chat`
  is stateless, one message in, one reply out.
- Dashboard framework — decided when we build the actual dashboard.
- Future `Capability` values (MEMORY, EMAIL, CALENDAR, PRINTER,
  ROBOTIC_ARM, ...) — the Capability Decision Layer (M6.13) is built to
  hold more than DIRECT/SEARCH/TOOL, but no enum member or decision logic
  for any of these exists yet; each is added only when a milestone
  actually implements it.
- Multi-tool chaining, agents, planning, or any autonomous execution —
  M6.16's Tool Decision Layer proposes exactly one `ToolCall` per message,
  once, for `JarvisCore` to dispatch through `ToolRegistry`; there is no
  loop that lets an AI decide to call several tools in sequence, reflect
  on a result, or retry with a different tool. `web_search` stays excluded
  from generic TOOL dispatch and keeps its own SEARCH path.
- Conversational clarification for a missing tool parameter —
  `ToolCallIncompleteError` (M6.16) preserves the typed boundary (a
  missing required parameter becomes a plain "not enough information"
  reply), but there's no follow-up flow yet that asks the user for the
  missing value and retries the same tool call once they answer.
- Further widening what counts as an explicit `TOOL` request at the
  `CapabilityRouter` level — M6.16 added a second narrow phrasing (the
  tool name before the literal word "tool", e.g. `"use the echo tool"`)
  alongside M6.14's original one, but both are still fixed, anchored
  patterns requiring one of `use`/`run`/`call`/`invoke` at the very start
  of the message; a message that matches neither (e.g. `"echo this
  message: hello"`, with no verb/`"tool"` phrasing at all) never reaches
  the Tool Decision Layer. Recognizing arbitrary free-form tool requests
  without any such anchor, or `HybridCapabilityRouter`'s ambiguity
  boundary, is a separate, deliberate future decision.
- Autonomous AI tool-calling (an AI provider deciding *when* to invoke a
  tool and executing it itself) — Gemini still never decides when to call
  `web_search`; `HybridCapabilityRouter`'s capability fallback (M6.15) and
  `GeminiToolCallProvider`'s tool-call construction (M6.16) each only ever
  return a typed decision (`CapabilityDecision`/`ToolCall`) for
  `JarvisCore` to dispatch - neither calls `ToolRegistry` itself, neither
  is wired as `JarvisCore`'s default, and each runs at most once per
  message, only when the deterministic layer above it couldn't already
  resolve things. `SearchRouter` (M6.9) is still a fixed keyword matcher
  for its own decision, and tools are still only ever invoked through the
  Registry - not through any autonomous, multi-step, or AI-driven
  execution loop.
- Smarter routing — mode selection (FAST/RESEARCH/CRITICAL) for a SEARCH
  message, and precision improvements beyond fixed keywords (e.g. "results"
  as a live-info trigger is intentionally broad and can over-match) -
  `SearchRouter` is deliberately simple today, and M6.10 didn't change it.
- AI-driven (Gemini-generated) search-query rewriting/expansion — M6.12
  added *deterministic* contextual query enrichment (fixed-pattern
  matching, no model call - see Search Query Understanding above) for a
  narrow, explicit set of context-dependent follow-ups. General
  coreference resolution, or using an LLM to construct/expand a query, is
  still a deliberate, separate future milestone - not attempted here to
  avoid the extra latency/cost of another model call on every SEARCH turn.
- Real Brave/Tavily providers and autonomous research/agents — left for
  later milestones now that `SearchProvider`, `SearchProviderManager`, and
  `CachedSearchProvider` give them a place to plug in.
- Result ranking, source credibility scoring, and any source-button/link
  UI — `ResultQualityProvider` only deduplicates by URL today; `/chat`
  exposes the validated `title`/`url`/`query` per source (M6.11), but
  ranking, credibility scoring, and rendering it in a UI are still left for
  later.
- Cache eviction/cleanup of expired rows, background cache refresh, cache
  warming, and semantic/embedding-based cache lookups — `SearchCache` is a
  plain TTL key/value store for now; expired rows are treated as misses but
  not deleted.
- Remote access / security design — deferred until that milestone, since it
  needs deliberate design rather than a quick default.
- `/chat` hardening that is app-wide, not Scheduled-Jobs-specific: `/chat` is the anonymous LOCAL interface and relies on the documented
  start (`uvicorn app.main:app`, which binds the loopback address; no CORS, and a cross-site non-JSON POST is refused). The server does not
  validate the `Host` header, so a DNS-rebinding page could in principle reach it, and `app/config.py` carries a `host = "0.0.0.0"` default
  that nothing reads (a test pins that nothing binds from it) - it would expose `/chat` to the network if it were ever wired to the server.
  Both belong to the remote-access design above (host validation, an explicit bind address), not to any one feature.
