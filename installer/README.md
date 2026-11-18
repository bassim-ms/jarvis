# Portable Jarvis installer (foundation)

**Requirement 901 - addendum to the 900-feature master specification.** Approved by the user on 2026-10-31:

> A portable Jarvis installer package, ideally one ZIP that I can extract and launch on any laptop. It should automatically
> configure dependencies, initialize Jarvis, securely guide API-key setup, and support safe migration of my data.

It is recorded here as a separate addendum so the original 900 numbered entries stay exactly as written. (The specification
document itself was not on this laptop when this was built; add 901 to it as an addendum, not by editing an existing entry.)

## Using it

| Step | Command (from the Jarvis folder) | What it does |
|---|---|---|
| Build the ZIP | `python installer\jarvis_setup.py package --out jarvis.zip` | Packs the program only - **no API keys, databases, device credential or `user_profile.md`** - and verifies the ZIP by reading it back. Refuses to build if any shipped file holds what looks like a real key. |
| On the new laptop | extract the ZIP, double-click **`Setup-Jarvis.bat`** | Finds Python 3.11+, creates `venv\`, installs `requirements.txt` and then checks every pinned package really is installed, creates `.env` from the template (placeholders emptied), then asks for each API key in the fixed provider order. What you type is not shown. **Stops at the first step that fails** and says which one. |
| Bring your data | `python installer\jarvis_setup.py migrate --from "D:\old\jarvis"` | Copies the `jarvis_*.db` databases, `jarvis_pc_device.json`, `personality.md` and `user_profile.md`. Each database is copied as a consistent snapshot (SQLite's backup API), integrity-checked, and its tables and row counts are compared with the original. The old folder is never modified. A file that already exists is **skipped**; with `--replace` it is first copied into `backups\<timestamp>\`. `--include-env` also brings the old `.env` (your keys). |
| Check | `python installer\jarvis_setup.py check` | Python version, disk space, files, virtual environment, every pinned dependency and its version, `.env`, and each AI provider's **key status** (below). Changes nothing. |
| Start | double-click **`Start-Jarvis.bat`** | Runs the check, then starts Jarvis on `127.0.0.1:8000` - this computer only. |

### Key status in `check`

A key is never reported as working because it is non-empty, and `check` never sends a test request to an AI provider. If
Jarvis is running, `check` asks it - on `127.0.0.1` only, `--port` if not 8000 - what each provider's most recent real
answer said (`GET /providers/status`):

| Shown | Meaning |
|---|---|
| `[no key]` | no key in `.env` - run `keys` |
| `[unverified]` | key present, not verified: Jarvis is not running, has not used this provider since it started, or started before the key was added (restart it) |
| `[ok]` | authenticated: the provider accepted the key in a real request |
| `[REJECTED]` | authentication rejected: the provider refused the key (401/403) - get a current key, run `keys`, restart Jarvis |
| `[unavailable]` | the last attempt failed for a reason that says nothing about the key (network, rate limit, provider error) |

Readiness ("At least one AI provider") needs one key that has not been refused.

### When something fails

Every failure is one message on screen - `Setup failed: <what went wrong>` and `What to do: <the fix>` - never a Python
traceback, and never a key. For example, a failed dependency install says which step failed and pip's exit code, and on
Windows gives the folder's path length: some packages install files deep inside `venv\Lib\site-packages`, and Windows refuses
paths over 260 characters unless long paths are turned on, so a deeply nested folder can make pip fail (seen on this laptop:
the same ZIP installed fine from a shorter folder). A missing or wrong `--from` folder for `migrate`, a folder
with nothing to bring over, an incomplete package, a Python older than 3.11, and no keyboard for `keys` each get their own
message.

Exit codes, the same for every command:

| Code | Meaning |
|---|---|
| 0 | done (and, for `check` / `launch`, ready) |
| 1 | done, but not ready yet (`check` / `launch`: something is MISSING, e.g. no usable API key) |
| 2 | the command failed, or was used wrongly |
| 130 | stopped with Ctrl+C |

`Setup-Jarvis.bat` stops at the first non-zero step (init, then keys), prints `Setup stopped at step N of 3 - ...`, and
exits with that code; the final check's 1 is reported as "finished, but not ready yet". Steps already done are kept, so
running it again carries on. The `.bat` files keep Windows (CRLF) line endings, which `.gitattributes` enforces: with LF-only
endings cmd.exe can miss the `goto` that stops it.

## What has been verified, and what has not

* Verified by tests (`tests/test_installer.py`, temporary folders only): every rule above, including that keys are never
  printed, an existing `.env` or database is never overwritten, a migrated database matches its source, a broken database
  is reported without stopping the rest, every failure path gives its message and exit code, packaging **this repository**
  produces a ZIP with no secrets or personal data, and both `.bat` files - really run by cmd.exe against a stand-in
  `jarvis_setup.py` - stop at the failing step and pass its exit code on.
* Verified end to end on this laptop only: the ZIP extracted into a fresh folder, `init` created a new environment and
  installed the dependencies, and the offline smoke check passed inside it (see the session handover for the exact output).
  A deep folder failed at pip because of the Windows path-length limit; a short one worked.
* **Not yet verified:** a second, clean Windows laptop. Python itself is detected, not installed. FreeLLMAPI (the desktop app),
  SearXNG and the voice models are reported by `check`, not set up. There is no signed `.exe` installer yet.

## Next steps toward the full requirement

1. Test on a clean Windows machine (or a fresh Windows user account) and fix what that finds.
2. Offer to install Python when it is missing (official python.org installer, with the user's confirmation).
3. Detect the FreeLLMAPI desktop app (it keeps its port in `%APPDATA%\FreeLLMAPI\config.json`, 31415 by default; the code
   default `localhost:3001` is the Docker address - see README "FreeLLMAPI addresses") and SearXNG, and guide their setup.
4. An optional encrypted data bundle for moving databases and keys together.
