# Jarvis

Personal AI assistant project. This is **Milestone 6.8: Search Answer
Layer** — a standalone `SearchAnswerGenerator` can now turn the structured
`SearchResponse` context from Milestone 6.7 into a grounded, free-text
answer via the existing Gemini/`AIProvider` integration, with a
deterministic, validated `sources` list (never AI-invented URLs). Nothing
in Jarvis calls it yet - it's data preparation and a ready-to-use layer,
not a wired-up feature. No automatic search routing, autonomous research,
citations UI, dashboard, agents, voice, or hardware control yet - see
[Web Search](#web-search) for the general knowledge vs. web search
distinction and what modes/planning/result quality/provenance/answers
actually do.

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
190 passed in 0.XXs
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
      cache.py            - SearchCache interface: provider-independent get/set
      sqlite_cache.py      - SQLiteSearchCache: SQLite-backed, TTL-aware SearchCache
      cached.py            - CachedSearchProvider: caches a wrapped SearchProvider's results
      modes.py             - SearchMode/SearchModeConfig: deterministic result limit per query
      planner.py            - SearchPlan/SearchPlanner/DeterministicSearchPlanner: which queries to run
      url_utils.py          - normalize_url: conservative comparison-key normalization
      quality.py            - deduplicate_results/ResultQualityProvider: URL-based dedup
      response.py            - SearchResponse: structured query/mode/queries_used/results context
      answer.py              - SearchAnswerGenerator/GeminiSearchAnswerGenerator: grounded answers
      factory.py          - builds the default SearchProvider/SearchCache from config
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
    test_search_factory.py   - proves the provider/cache config wiring
    test_search_cache.py     - unit tests for SQLiteSearchCache (hit/miss/TTL/persistence)
    test_cached_search_provider.py - unit tests for CachedSearchProvider's cache/fallback logic
    test_search_modes.py     - unit tests for SearchMode limits and fallback/caching under a mode
    test_search_quality.py   - unit tests for URL normalization, deduplication, and limit-after-dedup
    test_search_planner.py   - unit tests for DeterministicSearchPlanner's per-mode query plans
    test_search_response.py  - unit tests for SearchResponse and provenance through the whole chain
    test_search_answer.py    - unit tests for GeminiSearchAnswerGenerator and source validation
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
`GeminiProvider`.

**General/stable knowledge vs. web search:** Jarvis does not automatically
decide when to search - that's a deliberately postponed later milestone
(see below). Questions about stable facts ("what is quantum entanglement?",
"how does a transistor work?") are answered by Gemini directly, with no
tool call. Questions needing current/external information ("what happened
with NASA today?", "latest robotics news") go through `web_search`
explicitly. `web_search` itself has no idea which category a query falls
into; it just searches whatever query it's given.

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

A standalone, provider-agnostic layer that turns a `SearchResponse` into a
grounded, free-text answer - **not wired into `WebSearchTool`, `JarvisCore`,
or `/chat` yet**; it's a ready-to-use capability, tested on its own.

- **`SearchAnswerGenerator`** (`app/search/answer.py`) is the interface:
  `generate(response: SearchResponse) -> SearchAnswer`. It only imports
  `SearchResponse`/`SearchResult` and the existing `AIProvider` interface -
  never Gemini's HTTP details directly.
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

## Postponed decisions (on purpose)

These were not decided now, to avoid overbuilding before they're needed:
- Database (SQLite) — added when the first feature that needs persistence
  (tasks, reminders, memory) is built.
- Conversation memory / multi-turn context — not in this milestone; `/chat`
  is stateless, one message in, one reply out.
- Dashboard framework — decided when we build the actual dashboard.
- AI tool-calling (an AI provider choosing which tool to invoke) — Gemini
  doesn't decide when to call `web_search` yet; tools are invoked directly
  through the Registry for now. This includes automatic intent routing
  (deciding *whether* a query needs `web_search` at all, or which `mode`
  to use) - no AI classifier call is made just to make that decision.
- Real Brave/Tavily providers and autonomous research/agents — left for
  later milestones now that `SearchProvider`, `SearchProviderManager`, and
  `CachedSearchProvider` give them a place to plug in.
- Result ranking, source credibility scoring, and any source-button/link
  UI — `ResultQualityProvider` only deduplicates by URL today; the
  `source`/`query` metadata is preserved for a future presentation layer,
  but no UI renders it yet.
- Wiring `SearchAnswerGenerator` into `WebSearchTool`, `JarvisCore`, or
  `/chat` — M6.8 built the standalone layer; nothing in the running app
  calls it yet, and *whether/when* to generate an answer is still a
  routing decision for a later milestone.
- Cache eviction/cleanup of expired rows, background cache refresh, cache
  warming, and semantic/embedding-based cache lookups — `SearchCache` is a
  plain TTL key/value store for now; expired rows are treated as misses but
  not deleted.
- Remote access / security design — deferred until that milestone, since it
  needs deliberate design rather than a quick default.
