# Jarvis

Personal AI assistant project. This is **Milestone 2: Gemini integration** —
a running server with a health check and a `/chat` endpoint backed by
Google's Gemini API. No dashboard, memory, agents, voice, or hardware
control yet.

## Stack

- **Python 3.11+** with **FastAPI** (web framework) and **uvicorn** (server
  that runs it). Both are lightweight and run comfortably on a 4 GB RAM
  machine.
- **pydantic-settings** for configuration — reads settings from environment
  variables / a `.env` file, so no secrets are ever hard-coded.
- **httpx** talks directly to the Gemini REST API — no extra AI SDK needed
  for a single HTTP call.
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
   committed).

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
8 passed in 0.XXs
```

Tests never call the real Gemini API — `ask_gemini` / `httpx.post` are
monkeypatched, so they run without a network connection or a real API key.

## Folder structure

```
jarvis/
  app/
    __init__.py     - marks app/ as a Python package
    main.py          - the FastAPI app, /health and /chat endpoints
    config.py        - all configuration, read from environment variables
    ai_service.py    - minimal Gemini API client (ask_gemini)
  tests/
    test_health.py     - proves the server starts and /health works
    test_chat.py        - proves /chat works, with Gemini mocked out
    test_ai_service.py  - unit tests for the Gemini client, with HTTP mocked out
  requirements.txt    - exact dependency versions
  .env.example         - template for your local .env (never commit .env itself)
  .gitignore
```

As Jarvis grows, new capabilities (dashboard, memory, agents, voice,
hardware control, etc.) will each get their own module under `app/`,
keeping this structure from turning into a single giant file.

## Postponed decisions (on purpose)

These were not decided now, to avoid overbuilding before they're needed:
- Database (SQLite) — added when the first feature that needs persistence
  (tasks, reminders, memory) is built.
- Conversation memory / multi-turn context — not in this milestone; `/chat`
  is stateless, one message in, one reply out.
- Dashboard framework — decided when we build the actual dashboard.
- Remote access / security design — deferred until that milestone, since it
  needs deliberate design rather than a quick default.
