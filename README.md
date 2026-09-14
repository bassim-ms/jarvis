# Jarvis

Personal AI assistant project. This is **Milestone 1: project foundation** —
just a running server with a health check. No AI, dashboard, or hardware
control yet.

## Stack

- **Python 3.11+** with **FastAPI** (web framework) and **uvicorn** (server
  that runs it). Both are lightweight and run comfortably on a 4 GB RAM
  machine.
- **pydantic-settings** for configuration — reads settings from environment
  variables / a `.env` file, so no secrets are ever hard-coded.
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
   (Nothing needs to be filled in yet for this milestone.)

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

## Run the test

```
pytest
```

Expected output ends with something like:
```
1 passed in 0.XXs
```

## Folder structure

```
jarvis/
  app/
    __init__.py     - marks app/ as a Python package
    main.py          - the FastAPI app and the /health endpoint
    config.py        - all configuration, read from environment variables
  tests/
    test_health.py   - proves the server starts and /health works
  requirements.txt    - exact dependency versions
  .env.example         - template for your local .env (never commit .env itself)
  .gitignore
```

As Jarvis grows, new capabilities (AI integration, dashboard, hardware
control, etc.) will each get their own module under `app/`, keeping this
structure from turning into a single giant file.

## Postponed decisions (on purpose)

These were not decided now, to avoid overbuilding before they're needed:
- Database (SQLite) — added when the first feature that needs persistence
  (tasks, reminders, memory) is built.
- AI provider integration (Gemini) — next milestone.
- Dashboard framework — decided when we build the actual dashboard.
- Remote access / security design — deferred until that milestone, since it
  needs deliberate design rather than a quick default.
