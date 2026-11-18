"""
Jarvis setup: the portable-installer foundation (approved requirement, feature 901 addendum - see installer/README.md).

    python installer/jarvis_setup.py check                      what this laptop has and what is missing (changes nothing)
    python installer/jarvis_setup.py init                       first run: virtual environment, dependencies, .env, data folder
    python installer/jarvis_setup.py keys                       guided API-key entry (typed keys are never shown or printed)
    python installer/jarvis_setup.py migrate --from OLD_FOLDER  bring databases and personal files from another Jarvis folder
    python installer/jarvis_setup.py package --out jarvis.zip   build the portable ZIP (no keys, no databases, no personal data)
    python installer/jarvis_setup.py launch                     check, then start Jarvis on this computer only (127.0.0.1)

Standard library only, on purpose: it has to run on a laptop where nothing of Jarvis's is installed yet.

Rules this file keeps, each one tested in tests/test_installer.py:
  * An API key is never printed, logged, echoed, put in a ZIP or written anywhere but the install's own .env.
  * Nothing existing is overwritten. `init` never touches an existing .env; `migrate` skips a file that already exists unless
    --replace is given, and then copies the existing one into backups/<timestamp>/ first; .env changes are preceded by a
    .env.bak.<timestamp> copy.
  * A migrated database is a consistent snapshot (SQLite's own backup API), integrity-checked, and its row counts are
    compared with the source before the result is reported as copied.
  * The ZIP is built from an allow-list walk that refuses secrets and data by pattern, and is re-opened and verified.
  * A key is never called valid because it is non-empty. `check` reports "key present, not verified" unless the running
    Jarvis (asked on 127.0.0.1 only) has seen the provider accept or refuse it in a real request; nothing here ever sends a
    test request to an AI provider.
  * A failure is one plain-language message and what to do about it - never a Python traceback - and the exit code says
    what happened (see EXIT_* below), so Setup-Jarvis.bat can stop at the step that failed.

Exit codes: 0 = done (and, for check/launch, ready); 1 = done, but not ready yet (check/launch: something is MISSING);
2 = the command failed (or was used wrongly); 130 = stopped with Ctrl+C.

What it does NOT claim: it has only been exercised on this laptop and in temporary folders. A clean second laptop has not
been tested, Python itself is detected but not installed, and FreeLLMAPI / SearXNG / voice models are reported, not set up.
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

INSTALL_ROOT = Path(__file__).resolve().parent.parent
MIN_PYTHON = (3, 11)
MIN_FREE_BYTES = 2 * 1024**3  # dependencies + voice/vision libraries need room; 2 GB is a floor, not a recommendation
DEFAULT_PORT = 8000

EXIT_OK = 0
EXIT_NOT_READY = 1
EXIT_FAILED = 2
EXIT_CANCELLED = 130

#: The five AI providers in the user's FIXED order, with the .env names each needs. `name` is the provider's name in
#: Jarvis (GET /providers/status). `prefix` is the public, documented shape of a key (a typo check only - never a claim
#: that the key works).
PROVIDER_KEYS: tuple[dict, ...] = (
    {"name": "groq", "provider": "Groq", "where": "https://console.groq.com/keys",
     "settings": (("JARVIS_GROQ_API_KEY", "gsk_"),)},
    {"name": "openrouter", "provider": "OpenRouter", "where": "https://openrouter.ai/keys (may cost money - set a spending limit)",
     "settings": (("JARVIS_OPENROUTER_API_KEY", "sk-or-"),)},
    {"name": "gemini", "provider": "Gemini", "where": "https://aistudio.google.com/apikey",
     "settings": (("JARVIS_GEMINI_API_KEY", ""),)},
    {"name": "cloudflare", "provider": "Cloudflare Workers AI", "where": "https://dash.cloudflare.com (API token + account ID)",
     "settings": (("JARVIS_CLOUDFLARE_API_KEY", ""), ("JARVIS_CLOUDFLARE_ACCOUNT_ID", ""))},
    {"name": "freellmapi", "provider": "FreeLLMAPI",
     "where": "the FreeLLMAPI app -> Keys page (only if you run FreeLLMAPI; see README: FreeLLMAPI addresses)",
     "settings": (("JARVIS_FREELLMAPI_API_KEY", ""),)},
)


class SetupError(Exception):
    """A failure the person running setup can act on: `str()` says what went wrong in plain words, `hint` what to do.
    Never carries a key value."""

    def __init__(self, message: str, hint: str = ""):
        super().__init__(message)
        self.hint = hint

#: What `migrate` brings over: the databases, the device credential and the two identity files. `.env` only with --include-env.
MIGRATE_PATTERNS: tuple[str, ...] = ("jarvis_*.db", "jarvis_pc_device.json", "personality.md", "user_profile.md")

#: Never in the ZIP, whatever else matches: secrets, data, personal files, environments, caches, logs, version control.
PACKAGE_EXCLUDED_DIRS = frozenset({".git", "venv", ".venv", "sandbox", "backups", "__pycache__", ".pytest_cache", ".idea", ".vscode",
                                   "node_modules", "dist"})
_PACKAGE_EXCLUDED_FILE = re.compile(
    r"(^\.env$|^\.env\.(?!example$).*|\.env\.bak.*|.*\.db$|.*\.db-(wal|shm|journal)$|.*\.sqlite3$|.*\.log(\.\d+)?$|"
    r"^jarvis_.*device.*\.json$|.*\.pyc$|^user_profile\.md$|.*\.zip$)",
    re.IGNORECASE,
)
#: The top-level entries a Jarvis package is made of (an allow-list: a stray file in the folder is not shipped by accident).
PACKAGE_INCLUDED_TOP_LEVEL: tuple[str, ...] = (
    "app", "installer", "scripts", "tests", "README.md", "requirements.txt", "pytest.ini", ".env.example", ".gitignore",
    "personality.md", "Setup-Jarvis.bat", "Start-Jarvis.bat",
)


# ---------------------------------------------------------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------------------------------------------------------


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def timestamp() -> str:
    return time.strftime("%Y%m%d%H%M%S")


def venv_python(root: Path) -> Path:
    return root / "venv" / ("Scripts" if os.name == "nt" else "bin") / ("python.exe" if os.name == "nt" else "python")


_PLACEHOLDER = re.compile(r"(YOUR_[A-Z0-9_]*|<[^>]*>|x+|changeme|\.\.\.)", re.IGNORECASE)


def is_placeholder(value: str) -> bool:
    """A template value such as YOUR_GROQ_API_KEY_HERE - not a key."""
    return bool(_PLACEHOLDER.fullmatch(value.strip().strip("\"'")))


def read_env_values(env_path: Path) -> dict[str, str]:
    """NAME -> value for every `NAME=value` line. Values stay inside this process; nothing here prints them."""
    values: dict[str, str] = {}
    if not env_path.is_file():
        return values
    for line in env_path.read_text(encoding="utf-8-sig").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        name, value = stripped.split("=", 1)
        value = value.strip().strip('"').strip("'")
        values[name.strip()] = "" if is_placeholder(value) else value  # a template placeholder is not a setting
    return values


def write_env_values(env_path: Path, updates: dict[str, str], *, now: str | None = None) -> Path | None:
    """Set `updates` in the .env at `env_path`: an existing `NAME=` line is replaced in place, a new name is appended, every
    other line is kept byte for byte. An existing file is copied to `.env.bak.<timestamp>` first (returned). Refuses a value
    with a line break, which would smuggle a second setting in."""
    for name, value in updates.items():
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", name):
            raise ValueError(f"not a setting name: {name!r}")
        if "\n" in value or "\r" in value:
            raise ValueError(f"the value for {name} contains a line break")
    backup = None
    lines: list[str] = []
    if env_path.is_file():
        backup = env_path.with_name(f"{env_path.name}.bak.{now or timestamp()}")
        shutil.copy2(env_path, backup)
        lines = env_path.read_text(encoding="utf-8-sig").splitlines()
    remaining = dict(updates)
    for index, line in enumerate(lines):
        name = line.split("=", 1)[0].strip() if "=" in line and not line.lstrip().startswith("#") else None
        if name in remaining:
            lines[index] = f"{name}={remaining.pop(name)}"
    lines.extend(f"{name}={value}" for name, value in remaining.items())
    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return backup


# ---------------------------------------------------------------------------------------------------------------------------
# check
# ---------------------------------------------------------------------------------------------------------------------------


@dataclass
class Finding:
    name: str
    ok: bool
    detail: str
    #: A failed REQUIRED finding means Jarvis cannot start; a failed optional one is something to know about.
    required: bool = True
    #: The word shown in brackets, when "ok" / "MISSING" / "note" would say the wrong thing (an AI provider's key status).
    mark: str | None = None


#: An AI provider's key status (the same names Jarvis's GET /providers/status uses): the bracketed word `check` shows, and
#: its explanation. Only a real, successful answer from the provider makes a key "authenticated".
KEY_STATUS_MARK = {
    "no_key": "no key",
    "present_not_verified": "unverified",
    "authenticated": "ok",
    "rejected": "REJECTED",
    "unavailable": "unavailable",
}
_KEY_STATUS_DETAIL = {
    "no_key": "no key configured - run: keys",
    "authenticated": "authenticated - the provider accepted the key in a real request (as seen by the running Jarvis)",
    "rejected": "authentication rejected - the provider refused the key in a real request; get a current key, run: keys, "
                "then restart Jarvis",
    "unavailable": "unavailable - the running Jarvis's last attempt failed before the key could be checked (network, rate "
                   "limit or a provider error), so the key is not verified",
}
_UNVERIFIED_BECAUSE = {
    "not_running": "Jarvis is not running, so no request has checked it yet",
    "not_used": "the running Jarvis has not used this provider since it started",
    "restart": "the running Jarvis started without this key - restart Jarvis to use it",
}


def fetch_running_status(port: int = DEFAULT_PORT, timeout: float = 2.0) -> dict[str, str] | None:
    """provider name -> key status, from the running Jarvis's own GET /providers/status on THIS computer (127.0.0.1, no
    proxy), or None when Jarvis is not running there. This asks Jarvis what it has seen in real requests: nothing is sent
    to any AI provider, and no key is sent or received."""
    import json
    import urllib.request

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(f"http://127.0.0.1:{port}/providers/status", timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except (OSError, ValueError):  # not running, something else on the port, or not JSON
        return None
    providers = body.get("providers") if isinstance(body, dict) else None
    if not isinstance(providers, list):
        return None
    return {entry["name"]: entry.get("key_status") for entry in providers
            if isinstance(entry, dict) and isinstance(entry.get("name"), str)}


def provider_key_status(key_present: bool, running: dict[str, str] | None, name: str) -> tuple[str, str]:
    """(status, why) for one provider: the running Jarvis's verdict when it has one for a key that is present, else
    "present_not_verified" (with the reason) or "no_key". A non-empty key alone is never "authenticated"."""
    if not key_present:
        return "no_key", ""
    if running is None:
        return "present_not_verified", "not_running"
    seen = running.get(name)
    if seen in ("authenticated", "rejected", "unavailable"):
        return seen, ""
    return "present_not_verified", "restart" if seen in (None, "no_key") else "not_used"


_ASK_JARVIS = object()


def parse_requirements(text: str) -> dict[str, str]:
    """package -> pinned version, from `name==version` lines (extras like uvicorn[standard] reduced to the package)."""
    pins: dict[str, str] = {}
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        match = re.fullmatch(r"([A-Za-z0-9_.\-]+)(?:\[[^\]]*\])?==([^\s;]+)", line)
        if match:
            pins[match.group(1).lower().replace("_", "-")] = match.group(2)
    return pins


def installed_versions(python: Path, packages: list[str]) -> dict[str, str | None]:
    """What `python`'s environment has of `packages` (None = not installed), asked from that interpreter itself."""
    code = (
        "import json,sys\nfrom importlib import metadata\nout={}\n"
        "for name in sys.argv[1:]:\n"
        "    try: out[name]=metadata.version(name)\n"
        "    except metadata.PackageNotFoundError: out[name]=None\n"
        "print(json.dumps(out))"
    )
    import json

    result = subprocess.run([str(python), "-c", code, *packages], capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        return {name: None for name in packages}
    return json.loads(result.stdout)


def check_system(root: Path = INSTALL_ROOT, *, python_version: tuple[int, ...] | None = None, free_bytes: int | None = None,
                 versions_of=installed_versions, running_status=_ASK_JARVIS, port: int = DEFAULT_PORT) -> list[Finding]:
    """Everything `check` reports, as data. Reads only, and sends nothing to any AI provider. `running_status` is what the
    running Jarvis reports (see fetch_running_status; None = not running) - asked for on 127.0.0.1:`port` unless given.
    `python_version`, `free_bytes`, `versions_of` and `running_status` exist for tests."""
    findings: list[Finding] = []
    version = python_version or sys.version_info[:3]
    findings.append(Finding("Python", tuple(version[:2]) >= MIN_PYTHON,
                            f"{'.'.join(map(str, version[:3]))} (needs {MIN_PYTHON[0]}.{MIN_PYTHON[1]} or newer)"))
    free = shutil.disk_usage(root).free if free_bytes is None else free_bytes
    findings.append(Finding("Disk space", free >= MIN_FREE_BYTES, f"{free / 1024**3:.1f} GB free (needs 2 GB or more)"))

    missing_parts = [name for name in ("app", "requirements.txt", ".env.example") if not (root / name).exists()]
    findings.append(Finding("Jarvis files", not missing_parts,
                            "complete" if not missing_parts else f"missing: {', '.join(missing_parts)}"))

    python = venv_python(root)
    findings.append(Finding("Virtual environment", python.is_file(), str(python.parent.parent) if python.is_file()
                            else "not created yet - run: init"))

    pins = parse_requirements((root / "requirements.txt").read_text(encoding="utf-8")) if (root / "requirements.txt").is_file() else {}
    if python.is_file() and pins:
        have = versions_of(python, sorted(pins))
        absent = sorted(name for name in pins if not have.get(name))
        different = sorted(f"{name} {have[name]} (wants {pins[name]})" for name in pins if have.get(name) and have[name] != pins[name])
        findings.append(Finding("Dependencies", not absent, f"{len(pins) - len(absent)}/{len(pins)} installed"
                                + (f"; missing: {', '.join(absent)}" if absent else "")))
        if different:
            findings.append(Finding("Dependency versions", False, "; ".join(different), required=False))
    else:
        findings.append(Finding("Dependencies", False, "cannot be checked until the virtual environment exists"))

    env_path = root / ".env"
    findings.append(Finding("Configuration (.env)", env_path.is_file(), "present" if env_path.is_file()
                            else "not created yet - run: init"))
    values = read_env_values(env_path)
    running = fetch_running_status(port) if running_status is _ASK_JARVIS else running_status
    statuses = []
    for entry in PROVIDER_KEYS:
        key_present = all(values.get(name) for name, _ in entry["settings"])
        status, why = provider_key_status(key_present, running, entry["name"])
        statuses.append(status)
        detail = f"key present, not verified ({_UNVERIFIED_BECAUSE[why]})" if why else _KEY_STATUS_DETAIL[status]
        findings.append(Finding(f"AI provider: {entry['provider']}", status == "authenticated", detail, required=False,
                                mark=KEY_STATUS_MARK[status]))
    with_key = [status for status in statuses if status != "no_key"]
    usable = [status for status in with_key if status != "rejected"]  # not known to be refused
    labels = {"authenticated": "authenticated", "present_not_verified": "not verified", "rejected": "rejected",
              "unavailable": "unavailable"}
    counts = ", ".join(f"{with_key.count(status)} {label}" for status, label in labels.items() if status in with_key)
    findings.append(Finding(
        "At least one AI provider", bool(usable),
        f"{len(with_key)} of {len(PROVIDER_KEYS)} have a key" + (f" ({counts})" if counts else "")
        + ("" if usable or not with_key else " - every key was refused; run: keys"),
    ))

    databases = sorted(path.name for path in root.glob("jarvis_*.db"))
    findings.append(Finding("Data", True, f"{len(databases)} database file(s) here" if databases
                            else "no databases yet - they are created on first start (or run: migrate)", required=False))
    return findings


def print_findings(findings: list[Finding]) -> bool:
    for finding in findings:
        mark = finding.mark or ("ok" if finding.ok else ("MISSING" if finding.required else "note"))
        print(f"  [{mark:>11}] {finding.name}: {finding.detail}")
    ready = all(finding.ok for finding in findings if finding.required)
    print("Ready to start." if ready else "Not ready yet - see MISSING above.")
    return ready


# ---------------------------------------------------------------------------------------------------------------------------
# init
# ---------------------------------------------------------------------------------------------------------------------------


_INCOMPLETE_PACKAGE_HINT = ("Extract the whole Jarvis ZIP again, then run Setup-Jarvis.bat from inside the extracted jarvis "
                            "folder.")


def dependency_failure_hint(root: Path) -> str:
    """What to do when installing the dependencies failed - including the Windows path-length limit, the one cause seen
    on this laptop (a deep folder made a package file's path too long for pip)."""
    hint = "pip's own error is printed above. Check the internet connection, then run setup again."
    if os.name == "nt":
        hint += (f" This folder's path is {len(str(root))} characters long, and Windows limits a file path to 260 characters"
                 " unless long paths are turned on; some packages put files deep inside venv\\Lib\\site-packages. If pip's"
                 " error mentions a missing file or a path that is too long, move the Jarvis folder somewhere short (for"
                 " example C:\\Jarvis) and run setup again.")
    return hint


def _run_step(run, command: list[str], *, failure: str, hint: str) -> None:
    """One external step (venv creation, pip): its failure becomes a SetupError naming the step and the exit code."""
    try:
        run(command, check=True)
    except subprocess.CalledProcessError as exc:
        raise SetupError(f"{failure} ({Path(command[0]).name} stopped with exit code {exc.returncode}).", hint) from None
    except OSError as exc:
        raise SetupError(f"{failure}: {exc.strerror or exc}.", hint) from None


def initialize(root: Path = INSTALL_ROOT, *, install_dependencies: bool = True, run=subprocess.run, report=print,
               python_version: tuple[int, ...] | None = None, versions_of=installed_versions) -> list[str]:
    """First run. Creates only what is missing; an existing .env, database or virtual environment is left exactly as it is.
    Returns what it did, in order. Raises SetupError - before changing anything when it can - when Python is too old, the
    package is incomplete, or creating the environment or installing the dependencies fails (then verified: pip saying
    "done" is not taken on trust). `run`, `python_version` and `versions_of` exist for tests."""
    version = tuple(python_version or sys.version_info[:3])
    if version[:2] < MIN_PYTHON:
        raise SetupError(
            f"This Python is {'.'.join(map(str, version))}; Jarvis needs {MIN_PYTHON[0]}.{MIN_PYTHON[1]} or newer.",
            "Install a current Python from https://www.python.org/downloads/ (tick \"Add python.exe to PATH\"), then run "
            "setup again.",
        )
    if not root.is_dir():
        raise SetupError(f"The Jarvis folder was not found: {root}", _INCOMPLETE_PACKAGE_HINT)
    done: list[str] = []
    env_path, example, requirements = root / ".env", root / ".env.example", root / "requirements.txt"
    if install_dependencies and not requirements.is_file():
        raise SetupError(f"This folder is not a complete Jarvis package: requirements.txt is missing ({root}).",
                         _INCOMPLETE_PACKAGE_HINT)
    if env_path.exists():
        done.append("kept the existing .env")
    elif example.is_file():
        # The template, with every placeholder (YOUR_..._HERE) emptied: a placeholder sent as a key only earns a 401.
        lines = []
        for line in example.read_text(encoding="utf-8").splitlines():
            name, separator, value = line.partition("=")
            if separator and not line.lstrip().startswith("#") and is_placeholder(value):
                line = f"{name}="
            lines.append(line)
        env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        done.append("created .env from .env.example (no keys in it yet)")
    else:
        raise SetupError(f"This folder is not a complete Jarvis package: .env.example is missing ({root}).",
                         _INCOMPLETE_PACKAGE_HINT)

    python = venv_python(root)
    if python.is_file():
        done.append("kept the existing virtual environment")
    elif install_dependencies:
        report("Creating the virtual environment...")
        _run_step(run, [sys.executable, "-m", "venv", str(root / "venv")],
                  failure="Creating the virtual environment failed",
                  hint="Python's own error is printed above. Check there is free disk space and that this folder is not "
                       "read-only, then run setup again.")
        done.append("created the virtual environment")
    if install_dependencies:
        report("Installing dependencies (this downloads packages and can take a few minutes)...")
        _run_step(run, [str(python), "-m", "pip", "install", "--disable-pip-version-check", "-r", str(requirements)],
                  failure="Installing the dependencies failed", hint=dependency_failure_hint(root))
        pins = parse_requirements(requirements.read_text(encoding="utf-8"))
        have = versions_of(python, sorted(pins)) if pins else {}
        absent = sorted(name for name in pins if not have.get(name))
        if absent:
            raise SetupError(f"pip finished, but these dependencies are still not installed: {', '.join(absent)}.",
                             dependency_failure_hint(root))
        done.append(f"installed and verified {len(pins)} dependencies from requirements.txt")
    else:
        done.append("skipped dependencies (asked to)")
    (root / "backups").mkdir(exist_ok=True)
    done.append("backups folder ready")
    return done


# ---------------------------------------------------------------------------------------------------------------------------
# keys
# ---------------------------------------------------------------------------------------------------------------------------


def looks_wrong(name: str, value: str, prefix: str) -> str | None:
    """A typo check on a typed key - the reason it looks wrong, or None. Never includes the value."""
    if any(character.isspace() for character in value):
        return "it contains a space or line break"
    if len(value) < 8:
        return "it is too short to be a key"
    if prefix and not value.startswith(prefix):
        return f"keys for this provider normally start with {prefix}"
    if name.endswith("ACCOUNT_ID") and not re.fullmatch(r"[0-9a-fA-F]{32}", value):
        return "an account ID is 32 letters a-f and digits"
    return None


def collect_keys(existing: dict[str, str], *, ask=getpass.getpass, report=print) -> dict[str, str]:
    """Walk the five providers in order and return the settings to write. `ask` reads one hidden line (getpass: nothing typed
    is shown). Enter keeps what is there. A value that looks wrong is asked for once more, then accepted as typed - the
    check is a typo guard, not a gatekeeper. Nothing typed is ever passed to `report`."""
    updates: dict[str, str] = {}
    for entry in PROVIDER_KEYS:
        report(f"\n{entry['provider']} - get a key at: {entry['where']}")
        for name, prefix in entry["settings"]:
            state = "already set - press Enter to keep it" if existing.get(name) else "not set - press Enter to skip"
            for attempt in (1, 2):
                value = ask(f"  {name} ({state}): ").strip()
                if not value:
                    break
                problem = looks_wrong(name, value, prefix)
                if problem and attempt == 1:
                    report(f"  That does not look right: {problem}. Type it again, or press Enter to leave it unchanged.")
                    continue
                updates[name] = value
                report(f"  {name}: saved ({len(value)} characters).")
                break
    return updates


# ---------------------------------------------------------------------------------------------------------------------------
# migrate
# ---------------------------------------------------------------------------------------------------------------------------


@dataclass
class MigrationReport:
    copied: list[str] = field(default_factory=list)
    skipped_existing: list[str] = field(default_factory=list)
    replaced: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)
    backup_folder: Path | None = None


def _table_counts(path: Path) -> dict[str, int]:
    connection = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    try:
        tables = [row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
        return {table: connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0] for table in tables}
    finally:
        connection.close()


def copy_database(source: Path, destination: Path) -> None:
    """A consistent snapshot of `source` at `destination` (SQLite's backup API reads through any write-ahead log), then
    proof: the copy passes an integrity check and has the same tables with the same row counts. The source is opened
    read-only and is never written. Raises on any difference, leaving no half-written destination."""
    partial = destination.with_name(destination.name + ".partial")
    partial.unlink(missing_ok=True)
    try:
        source_connection = sqlite3.connect(f"{source.resolve().as_uri()}?mode=ro", uri=True)
        try:
            target_connection = sqlite3.connect(partial)
            try:
                source_connection.backup(target_connection)
            finally:
                target_connection.close()
        finally:
            source_connection.close()
    except Exception:
        partial.unlink(missing_ok=True)  # e.g. the "database" is not one: no half-written file is left behind
        raise
    try:
        check = sqlite3.connect(f"{partial.resolve().as_uri()}?mode=ro", uri=True)
        try:
            verdict = check.execute("PRAGMA integrity_check").fetchone()[0]
        finally:
            check.close()
        if verdict != "ok":
            raise ValueError(f"the copy failed its integrity check: {verdict}")
        if _table_counts(partial) != _table_counts(source):
            raise ValueError("the copy's tables or row counts differ from the source")
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    os.replace(partial, destination)


def copy_plain_file(source: Path, destination: Path) -> None:
    partial = destination.with_name(destination.name + ".partial")
    shutil.copy2(source, partial)
    if sha256(partial) != sha256(source):
        partial.unlink(missing_ok=True)
        raise ValueError("the copy is not identical to the source")
    os.replace(partial, destination)


def migrate(source: Path, root: Path = INSTALL_ROOT, *, replace: bool = False, include_env: bool = False,
            now: str | None = None) -> MigrationReport:
    """Bring the data of another Jarvis folder into this one. Never deletes or modifies anything in `source`. A file that
    already exists here is skipped - or, with `replace`, first copied into backups/<timestamp>/ and only then replaced."""
    source, root = source.resolve(), root.resolve()
    if not source.is_dir():
        raise SetupError(
            f"The old Jarvis folder was not found: {source}" if not source.exists()
            else f"This is a file, not a folder: {source}",
            "Give the folder that holds the old jarvis_*.db files, in quotes if it has spaces, for example: "
            "migrate --from \"C:\\Users\\you\\old jarvis\"",
        )
    if source == root:
        raise SetupError("The old folder and this folder are the same.",
                         "Give --from the OTHER Jarvis folder, the one you are bringing data from.")
    report = MigrationReport()
    names = sorted({path.name for pattern in MIGRATE_PATTERNS for path in source.glob(pattern) if path.is_file()})
    if include_env and (source / ".env").is_file():
        names.append(".env")
    if not names:
        raise SetupError(f"There is nothing to bring over from {source}: it has no jarvis_*.db, jarvis_pc_device.json, "
                         "personality.md or user_profile.md.",
                         "Check that this is the old Jarvis folder itself (the one with the jarvis_*.db files), not a "
                         "folder above or below it. Nothing was changed.")
    for name in names:
        origin, target = source / name, root / name
        try:
            if target.exists():
                if not replace:
                    report.skipped_existing.append(name)
                    continue
                if report.backup_folder is None:
                    report.backup_folder = root / "backups" / (now or timestamp())
                    report.backup_folder.mkdir(parents=True, exist_ok=False)
                kept = report.backup_folder / name
                (copy_database if name.endswith(".db") else copy_plain_file)(target, kept)
                report.replaced.append(name)
            (copy_database if name.endswith(".db") else copy_plain_file)(origin, target)
            report.copied.append(name)
        except Exception as exc:  # noqa: BLE001 - one bad file must not stop the rest; it is reported by name
            report.failed[name] = f"{type(exc).__name__}: {exc}"
    return report


# ---------------------------------------------------------------------------------------------------------------------------
# package
# ---------------------------------------------------------------------------------------------------------------------------


def package_files(root: Path = INSTALL_ROOT) -> list[Path]:
    """The files a portable ZIP contains: the allow-listed top-level entries, minus everything the exclusion rules name."""
    chosen: list[Path] = []
    for top in PACKAGE_INCLUDED_TOP_LEVEL:
        start = root / top
        if start.is_file():
            if not _PACKAGE_EXCLUDED_FILE.match(start.name):
                chosen.append(start)
        elif start.is_dir():
            for folder, folders, files in os.walk(start):
                folders[:] = sorted(name for name in folders if name not in PACKAGE_EXCLUDED_DIRS)
                for name in sorted(files):
                    if not _PACKAGE_EXCLUDED_FILE.match(name):
                        chosen.append(Path(folder) / name)
    return chosen


def looks_like_a_secret(text: str) -> bool:
    """A filled-in key assignment in a file about to be shipped (the .env.example template's placeholders are not)."""
    for match in re.finditer(r"^JARVIS_[A-Z0-9_]*(?:API_KEY|TOKEN|SECRET)[ \t]*=[ \t]*([^\r\n]*)", text, re.MULTILINE):
        value = match.group(1).split(" #", 1)[0].strip().strip("\"'")
        if value and not is_placeholder(value):
            return True
    return False


def build_package(out: Path, root: Path = INSTALL_ROOT) -> dict:
    """Write the portable ZIP and verify it by reading it back. Returns {'files': n, 'bytes': n, 'sha256': ...}."""
    out = out.resolve()
    files = [path for path in package_files(root) if path.resolve() != out]
    for path in files:
        if path.suffix.lower() in (".example", ".md", ".txt", ".bat", ".ini", ".py") and path.stat().st_size < 2_000_000:
            if looks_like_a_secret(path.read_text(encoding="utf-8", errors="ignore")):
                raise ValueError(f"refusing to package: {path.relative_to(root)} contains what looks like a real key")
    manifest_lines = []
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_name(out.name + ".partial")
    with zipfile.ZipFile(partial, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            relative = path.relative_to(root).as_posix()
            archive.write(path, f"jarvis/{relative}")
            manifest_lines.append(f"{sha256(path)}  {relative}")
        archive.writestr("jarvis/PACKAGE_MANIFEST.txt", "\n".join(manifest_lines) + "\n")
    with zipfile.ZipFile(partial) as archive:
        if archive.testzip() is not None:
            raise ValueError("the ZIP failed its own integrity test")
        for name in archive.namelist():
            parts = name.split("/")
            if _PACKAGE_EXCLUDED_FILE.match(parts[-1]) or PACKAGE_EXCLUDED_DIRS.intersection(parts[:-1]):
                raise ValueError(f"refusing: {name} must never be packaged")
    os.replace(partial, out)
    return {"files": len(files), "bytes": out.stat().st_size, "sha256": sha256(out)}


# ---------------------------------------------------------------------------------------------------------------------------
# launch + command line
# ---------------------------------------------------------------------------------------------------------------------------


def launch_command(root: Path = INSTALL_ROOT, port: int = 8000) -> list[str]:
    """Jarvis on this computer only: bound to 127.0.0.1, never to the network."""
    return [str(venv_python(root)), "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(port)]


def report_failure(message: str, hint: str = "") -> None:
    """The one way a failure is shown: what went wrong, then what to do. On stderr, never a traceback, never a key."""
    print(f"\nSetup failed: {message}", file=sys.stderr)
    if hint:
        print(f"What to do: {hint}", file=sys.stderr)


def _describe(exc: BaseException) -> str:
    if isinstance(exc, OSError) and exc.strerror:
        return f"{exc.strerror}: {exc.filename}" if exc.filename else exc.strerror
    return str(exc) or type(exc).__name__


def _run(args: argparse.Namespace, root: Path) -> int:
    if args.command == "check":
        return EXIT_OK if print_findings(check_system(root, port=args.port)) else EXIT_NOT_READY
    if args.command == "init":
        for line in initialize(root, install_dependencies=not args.skip_dependencies):
            print(f"  - {line}")
        # Setup itself succeeded. What is still missing (normally: API keys - the next step) is reported, not a failure.
        print_findings(check_system(root))
        return EXIT_OK
    if args.command == "keys":
        env_path = root / ".env"
        if not env_path.is_file():
            raise SetupError(f"There is no .env in {root} yet.", "Run: init (or Setup-Jarvis.bat) first.")
        print("Type each key and press Enter. What you type is not shown. Press Enter alone to keep or skip.")
        try:
            updates = collect_keys(read_env_values(env_path))
        except EOFError:
            raise SetupError("No keyboard input is available here, so keys cannot be typed (nothing was saved).",
                             "Run Setup-Jarvis.bat by double-clicking it, or run this command in a terminal window.") from None
        if not updates:
            print("\nNothing changed.")
            return EXIT_OK
        backup = write_env_values(env_path, updates)
        print(f"\nSaved {len(updates)} setting(s) to .env. The previous .env is kept as {backup.name}.")
        print("A saved key is not verified yet: Jarvis learns whether it works the first time it uses it (see: check).")
        return EXIT_OK
    if args.command == "migrate":
        report = migrate(Path(args.source), root, replace=args.replace, include_env=args.include_env)
        for label, names in (("copied", report.copied), ("already here, left alone", report.skipped_existing),
                             ("replaced (old one backed up)", report.replaced)):
            if names:
                print(f"  {label}: {', '.join(names)}")
        for name, reason in report.failed.items():
            print(f"  FAILED {name}: {reason}")
        if report.backup_folder:
            print(f"  backups: {report.backup_folder}")
        if report.skipped_existing and not args.replace:
            print("  (run again with --replace to replace the files that are already here; they are backed up first)")
        if report.failed:
            raise SetupError(f"{len(report.failed)} file(s) could not be brought over - see FAILED above.",
                             "The other files were copied and checked. Close Jarvis in the old folder if it is running, "
                             "then run migrate again: files already here are left alone.")
        return EXIT_OK
    if args.command == "package":
        result = build_package(Path(args.out), root)
        print(f"  wrote {args.out}: {result['files']} files, {result['bytes'] / 1024**2:.1f} MB, SHA-256 {result['sha256']}")
        print("  No API keys, databases, device credentials or user_profile.md are inside. Bring those with: migrate")
        return EXIT_OK
    if args.command == "launch":
        if not print_findings(check_system(root, port=args.port)):
            return EXIT_NOT_READY
        print(f"Starting Jarvis at http://127.0.0.1:{args.port} (this computer only). Press Ctrl+C to stop.")
        return subprocess.call(launch_command(root, args.port), cwd=str(root))
    raise SetupError(f"Unknown command: {args.command}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    check_parser = commands.add_parser("check")
    check_parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                              help="where a running Jarvis is asked (on 127.0.0.1) what it has seen of each key")
    init_parser = commands.add_parser("init")
    init_parser.add_argument("--skip-dependencies", action="store_true", help="do not create the environment or install packages")
    commands.add_parser("keys")
    migrate_parser = commands.add_parser("migrate")
    migrate_parser.add_argument("--from", dest="source", required=True, help="the other Jarvis folder")
    migrate_parser.add_argument("--replace", action="store_true", help="replace files that already exist here (after backing them up)")
    migrate_parser.add_argument("--include-env", action="store_true", help="also bring the old .env (it holds your API keys)")
    package_parser = commands.add_parser("package")
    package_parser.add_argument("--out", required=True)
    launch_parser = commands.add_parser("launch")
    launch_parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args(argv)  # a usage mistake exits with argparse's own message and code 2 (EXIT_FAILED)

    try:
        return _run(args, INSTALL_ROOT)
    except SetupError as exc:
        report_failure(str(exc), exc.hint)
    except KeyboardInterrupt:
        print("\nStopped (Ctrl+C). Run the same command again to carry on: steps already done are kept.", file=sys.stderr)
        return EXIT_CANCELLED
    except (OSError, ValueError) as exc:  # e.g. a folder that cannot be written; packaging refusing a key-like file
        report_failure(_describe(exc), "Fix what the message names, then run the same command again.")
    except Exception as exc:  # noqa: BLE001 - a bug must still end in a message and a failure code, never look like "not ready"
        report_failure(f"an unexpected {type(exc).__name__}: {_describe(exc)}",
                       "Nothing more was done. Run the same command again; if it repeats, report this message.")
    return EXIT_FAILED


if __name__ == "__main__":
    sys.exit(main())
