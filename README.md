# Jarvis

Personal AI assistant project. This is **Milestone 6.2: Search Provider
Manager + Fallback** — the `web_search` tool from Milestone 6.1 now runs its
`SearchProvider` through a `SearchProviderManager`, which tries an ordered,
configurable list of providers and falls back to the next on failure
(SearXNG is still the only real provider; Brave, Tavily, etc. can be added
later as entries in that list, with no changes to `WebSearchTool` or
Jarvis Core). No AI tool-calling, dashboard, agents, voice, or hardware
control yet.

## Stack

- **Python 3.11+** with **FastAPI** (web framework) and **uvicorn** (server
  that runs it). Both are lightweight and run comfortably on a 4 GB RAM
  machine.
- **pydantic-settings** for configuration — reads settings from environment
  variables / a `.env` file, so no secrets are ever hard-coded.
- **httpx** talks directly to the Gemini REST API and to a SearXNG
  instance's JSON search API — no extra AI or search SDK needed for a
  single HTTP call.
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
   `JARVIS_SEARCH_PROVIDER_ORDER` (see [Web Search](#web-search) below).

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
80 passed in 0.XXs
```

Tests never call the real Gemini or SearXNG APIs — `httpx.post` / `httpx.get`
are monkeypatched, so they run without a network connection or real
credentials. Fallback behavior is proven with lightweight fake
`SearchProvider`s, not real network calls to multiple backends.

## Folder structure

```
jarvis/
  app/
    __init__.py       - marks app/ as a Python package
    main.py            - the FastAPI app, /health and /chat endpoints
    config.py          - all configuration, read from environment variables
    core.py            - Jarvis Core: orchestrates a chat request via an AIProvider
    providers/
      base.py           - AIProvider interface, so Core doesn't depend on any vendor
      gemini.py         - Gemini implementation of AIProvider
    memory/
      base.py           - MemoryStore interface and the Message type
      sqlite_store.py   - SQLite implementation of MemoryStore
    tools/
      base.py           - Tool interface, PermissionLevel, ToolResult, ToolRequirements
      registry.py       - ToolRegistry: register/lookup/list/execute, permission enforcement
      web_search.py      - WebSearchTool: adapts a SearchProvider into the Tool interface
      defaults.py        - register_default_tools: wires the app's default tools (web_search) into a registry
    search/
      base.py           - SearchProvider interface and the SearchResult type
      searxng.py         - SearXNG implementation of SearchProvider
      manager.py          - SearchProviderManager: ordered fallback across SearchProviders
      factory.py          - builds the default SearchProvider from JARVIS_SEARCH_PROVIDER_ORDER
  tests/
    test_health.py         - proves the server starts and /health works
    test_chat.py            - proves /chat works, with Jarvis Core mocked out
    test_core.py            - unit tests for Jarvis Core's provider delegation
    test_gemini_provider.py - unit tests for the Gemini provider, with HTTP mocked out
    test_memory.py          - unit tests for the SQLite memory store
    test_tools.py           - unit tests for the Tool Registry and permission model
    test_searxng_provider.py - unit tests for the SearXNG provider, with HTTP mocked out
    test_web_search_tool.py  - unit tests for WebSearchTool, through the Tool Registry
    test_default_tools.py    - proves web_search is in the default/application tool registry
    test_search_provider_manager.py - fallback ordering/short-circuiting, with fake providers
    test_search_factory.py   - proves the provider list comes from configuration
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
`GeminiProvider`:

- **`SearchProvider`** (`app/search/base.py`) is the interface a search
  backend implements: `search(query, max_results) -> list[SearchResult]`.
  `SearchResult` is a standardized `title` / `url` / `snippet` triple, the
  same regardless of which backend produced it.
- **`SearXNGProvider`** (`app/search/searxng.py`) is the first
  implementation, calling a SearXNG instance's JSON search API
  (`GET {JARVIS_SEARXNG_URL}/search?q=...&format=json`) over `httpx`.
  Network errors and unexpected response shapes both raise `SearXNGError`
  (a `SearchProviderError`), never a bare exception.
- **`WebSearchTool`** (`app/tools/web_search.py`) implements `Tool`,
  taking a `SearchProvider` in its constructor (defaulting to a bare
  `SearXNGProvider` when none is given). It has
  `permission = PermissionLevel.READ`, so the Registry runs it immediately
  with no confirmation step. `execute` turns a `SearchProviderError` into a
  `ToolResult.error(...)` rather than letting it propagate.
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

Configuration:
- `JARVIS_SEARXNG_URL` - base URL of a SearXNG instance (self-hosted or a
  trusted public one). No default URL and no key is hard-coded anywhere.
- `JARVIS_SEARCH_PROVIDER_ORDER` - comma-separated, ordered list of
  provider names to try, e.g. `searxng` (default) or, once more providers
  exist, `searxng,brave`. Unknown names raise `UnknownSearchProviderError`
  at startup rather than failing silently.

See `.env.example` for both.

## Postponed decisions (on purpose)

These were not decided now, to avoid overbuilding before they're needed:
- Database (SQLite) — added when the first feature that needs persistence
  (tasks, reminders, memory) is built.
- Conversation memory / multi-turn context — not in this milestone; `/chat`
  is stateless, one message in, one reply out.
- Dashboard framework — decided when we build the actual dashboard.
- AI tool-calling (an AI provider choosing which tool to invoke) — Gemini
  doesn't decide when to call `web_search` yet; tools are invoked directly
  through the Registry for now.
- Real Brave/Tavily providers, result caching, search modes, and autonomous
  research/agents — left for later milestones now that `SearchProvider` and
  `SearchProviderManager` give them a place to plug in.
- Remote access / security design — deferred until that milestone, since it
  needs deliberate design rather than a quick default.
