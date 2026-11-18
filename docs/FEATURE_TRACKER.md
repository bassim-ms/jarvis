# Jarvis feature tracker

Evidence of what has been built and verified, against the master checklist **docs/MASTER_SPEC.md** ("JARVIS: THE COMPLETE
900-FEATURE MASTER SPECIFICATION", numbering as provided) and its addenda (**docs/MASTER_SPEC_ADDENDA.md**, 901+). The spec's
own [BUILT]/[PARTIAL]/[PLANNED] labels describe the project state when it was written; this tracker records what each batch
since then actually delivered, with its evidence. Earlier milestones (M0-M17) are described in the README.

Status: **Done** = implemented, tested and pushed. **Partial** = a working part is pushed; the rest is still to do.
"This laptop" = verified on the development laptop only; a second laptop has not been tested.

## Batch 1 - provider reliability and the portable installer (2026-11-18, commit 5ca989d)

| Spec # | Feature | Status | Evidence |
|---|---|---|---|
| 34, 67 | Provider fallback (strengthened: one fixed order on every workflow, strict-JSON replies validated and passed to the next provider) | Done | `tests/test_provider_fixed_order.py`, `tests/test_provider_router.py` |
| 70 | Provider health monitoring | Partial | `GET /providers/status`: order, key status from real requests, failures, rest time - `tests/test_providers_status_endpoint.py`. No latency/reliability history yet (71, 72) |
| 886 | Intelligent rate-limit handling | Partial | Provider-stated wait read in ms/s/m/h, bounded 0-24 h, provider rests in place - `tests/test_provider_fixed_order.py`. No quota tracking yet (74, 885) |
| 889 | Efficient token usage | Partial | Chat prompt carries the last 40 messages - `tests/test_chat_history_bound.py` |
| 421 | Portable Jarvis project setup | Partial | ZIP package + setup on this laptop - `tests/test_installer.py` (90) and a fresh-folder ZIP install |
| 422 | One-command installation | Done (this laptop) | `Setup-Jarvis.bat` (init, keys, check; stops at the failing step) - real cmd.exe runs in `tests/test_installer.py` |
| 423 | One-command startup | Done (this laptop) | `Start-Jarvis.bat` (check, then start on 127.0.0.1) |
| 424 | Environment detection | Partial | `check`: Python, disk, files, venv, dependencies, .env, keys. Not yet hardware (425) or Windows version (426) |
| 427 | Python version detection | Done | `check` and `init` refuse < 3.11 |
| 428 | Dependency verification | Done | every pinned package and version checked; `init` verifies after pip |
| 429 | Automatic virtual-environment setup | Done (this laptop) | `init` |
| 431 | Configuration migration | Partial | `migrate --include-env`; keys re-entry guided by `keys` |
| 432-438 | Database, memory, conversation, automation, task, user-profile and personality migration | Done (this laptop) | `migrate`: SQLite backup API, integrity check, row counts, never overwrites without `--replace` (backed up first) |
| 440 | Migration verification | Done (this laptop) | integrity + row-count comparison per database; byte-identical source |
| 901 | Portable installer package (addendum) | Partial | all of the above; not yet tested on a second laptop |

## Batch 2 - the web dashboard, Slice 1 (2026-11-18, commit 4648663)

| Spec # | Feature | Status | Evidence |
|---|---|---|---|
| 461 | Main Jarvis dashboard | Partial | `/dashboard` (app/dashboard/) - `tests/test_dashboard.py` (52); real browser check against the real app in a sandbox with a stub AI provider |
| 462 | Conversation interface | Done | chat panel on `POST /chat`, conversation restored after reload (browser-checked) |
| 464 | Task dashboard | Done | list/filter, add, done, cancel through the existing task tools (browser-checked) |
| 465 | Reminder dashboard | Partial | active reminders, soonest first, repeating marked (browser-checked: a reminder set in chat appears). Pause/resume/cancel: next batch |
| 468 | Job-status dashboard | Partial | recent Jobs with status and outcome. Pause/resume/cancel and step detail: next batch |
| 473 | Provider-status display | Done | fixed order, model and key status per provider |
| 208 | Human approval checkpoints | Partial | waiting questions/confirmations shown with Approve / Decline / Answer, sent as `answer <code>: ...` to the existing resolver |
| 480 | Unified control center | Partial | one page for chat, approvals, tasks, reminders, Jobs, notes, providers |

Security for every dashboard route: loopback-only Host guard; changes need the `X-Jarvis-Dashboard` header and a local origin
(cross-site requests refused - also checked against the live server); strict Content-Security-Policy; values rendered as text.

## Batch 3 - the web dashboard, Slice 2 (2026-11-18)

| Spec # | Feature | Status | Evidence |
|---|---|---|---|
| 465 | Reminder dashboard | Done | + Pause / Resume / Done / Cancel through the existing reminder tools - `tests/test_dashboard.py`; browser-checked (pause, resume) |
| 468 | Job-status dashboard | Done | + Pause / Resume / Cancel through the existing Job tools, plan steps with status, attempts, result/error, last event - tests; browser-checked (pause, plan stays open across refreshes) |
| 207 | Safe task cancellation | Partial | cancelling a Job from the dashboard goes through `cancel_job`, which also withdraws its waiting question (tested); asks before cancelling |
| 201 | Execution progress tracking | Partial | per-step status and attempts visible for every Job |
| 466 | Calendar dashboard | Partial | next 7 days from Google Calendar, cached 60 s, fixed error sentences (tested with a fake provider). Not checked against a live Google account: calendar is not connected on this laptop |
| 467 | Automation dashboard | Done | schedule, action, next/last run; Enable / Disable / Cancel through the existing tools - tests; browser-checked (disable, enable) |
| 469 | Memory browser | Done | stored facts, preferences and unconfirmed candidates (read-only) - tests; browser-checked. Editing (104) not yet |
| 470 | World-model browser | Partial | projects, goals, commitments, blockers, upcoming events (read-only) - tests; browser-checked. Resources and relationships not shown yet |
| 471 | Project dashboard | Done | projects with status, priority, description and goals (read-only) |
| 472 | Goal dashboard | Done | goals per project with status, target date and linked tasks (read-only) |

## Next batches

1. **Provider observability** (70-74, 885): per-provider latency, success/failure history, reliability score and quota
   hints from real requests, shown on the dashboard.
2. **Remote access** (secure, no open router ports), then the hardware sections (25 Bambu Lab P1S, 27 robotic arm, 29 lights)
   as the devices become available for real testing - simulated tests will be labelled as simulated.
