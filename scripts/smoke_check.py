"""
Smoke check: start the real Jarvis application in a SANDBOX and prove the basic path works, without touching your real data.

    python scripts/smoke_check.py                  # offline: no network, no AI call
    python scripts/smoke_check.py --live           # also one real /chat round-trip through the configured provider chain
    python scripts/smoke_check.py --providers      # also one tiny real call to EACH configured provider, one at a time

Run it from the repository root with the project's venv. What it does:

  1. Sets, BEFORE the app is imported: JARVIS_DATA_DIR (default "sandbox"), the Workspace Files folder inside it, and the background workers,
     self-development, computer input and desktop notifications OFF - so every database, the device credential, the log and the workspace are
     in the sandbox, and nothing runs on its own, edits code, drives the desktop or pops a toast.
  2. FAILS CLOSED before the application is imported (importing it opens every database) if isolation is not guaranteed: any data file or
     the workspace resolving outside the sandbox (an absolute path in .env, or a relative one climbing out with ".."), a sandbox that contains
     the repository, or any of the switches above not off.
  3. Records a SHA-256 of each real data file in the repository root (jarvis_*.db, .env, jarvis_pc_device.json).
  4. Starts the app (lifespan included, so logging is configured) and checks /health and one /chat request. Offline, the chat goes through
     the whole real dispatch path with a stub AI provider and real network transports blocked; with --live it uses the real provider chain.
  5. Re-hashes the real data files and FAILS if any changed.

It prints no key, no reply text beyond a short preview, and no database content. Exit code 0 = every check passed, 1 = a check failed,
2 = refused to start because isolation could not be guaranteed. The sandbox folder is kept so you can read its log; delete it whenever you
like (it is git-ignored).
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LIVE_PROMPT = "Reply with exactly one word: ready"


def sandbox_environment(sandbox: Path) -> dict[str, str]:
    """The environment that confines the app to `sandbox` (set before the app is imported; it takes precedence over .env)."""
    return {
        "JARVIS_DATA_DIR": str(sandbox),
        "JARVIS_WORKSPACE_DIR": str(sandbox / "workspace"),
        "JARVIS_BACKGROUND_SCHEDULERS_ENABLED": "false",
        "JARVIS_SELF_DEV_ENABLED": "false",
        "JARVIS_COMPUTER_INPUT_ENABLED": "false",
        "JARVIS_NOTIFICATIONS_DESKTOP_ENABLED": "false",
    }


def _inside(path: Path, root: Path) -> bool:
    try:
        return path.resolve().is_relative_to(root.resolve())
    except (OSError, ValueError):
        return False


def isolation_problems(settings, sandbox: Path, world_model_db_path: str, repo_root: Path = REPO_ROOT) -> list[str]:
    """Every reason the app would NOT be confined to `sandbox` with these settings - empty means isolated. Pure: reads settings only."""
    problems = []
    if _inside(repo_root, sandbox):
        problems.append(f"the sandbox {sandbox} is (or contains) the repository, where the real databases live")
    if not settings.data_dir:
        problems.append("no data folder is configured")
    elif Path(settings.data_dir).resolve() != sandbox.resolve():
        problems.append(f"the configured data folder {settings.data_dir} is not the sandbox {sandbox}")
    extra = {"world_model_db_path": world_model_db_path, "workspace_dir": settings.workspace_dir}
    for name in settings.data_files_outside_data_dir(extra):
        problems.append(f"{name} resolves outside the sandbox")
    for name in ("background_schedulers_enabled", "self_dev_enabled", "computer_input_enabled", "notifications_desktop_enabled"):
        if getattr(settings, name):
            problems.append(f"{name} is on")
    return problems


def real_data_files() -> list[Path]:
    names = sorted(REPO_ROOT.glob("jarvis_*.db")) + [REPO_ROOT / ".env", REPO_ROOT / "jarvis_pc_device.json"]
    return [p for p in names if p.is_file()]


def digests(paths: list[Path]) -> dict[str, str]:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def _go_offline(core) -> None:
    """Offline mode: the shared provider chain answers a fixed word, and any other HTTP attempt fails loudly.

    The chain is patched on its INSTANCE, not replaced: the reasoner, planner and search answerer were built at import holding that same
    object, so they get the stub too. httpx's real-network transports are blocked as a second line, so nothing reaches the network even
    through a component that built a provider of its own."""
    import httpx

    def stub_reply(message: str, requirements=None) -> str:
        return "ready"

    def no_network(*args, **kwargs):
        raise RuntimeError("smoke check is offline: network disabled (use --live for a real call)")

    core.provider.generate_reply = stub_reply
    # The real-network transports only: the TestClient talks to the app through its own in-process transport, which stays untouched.
    httpx.HTTPTransport.handle_request = no_network
    httpx.AsyncHTTPTransport.handle_async_request = no_network


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="sandbox", help="sandbox folder (relative to the repository root); default: sandbox")
    parser.add_argument("--live", action="store_true", help="one real /chat round-trip through the configured provider chain")
    parser.add_argument("--providers", action="store_true", help="one tiny real call to each configured provider")
    args = parser.parse_args(argv)

    os.chdir(REPO_ROOT)
    sandbox = Path(args.data_dir)
    if _inside(REPO_ROOT, sandbox):
        print(f"refusing: the sandbox {sandbox} is (or contains) the repository, where the real databases live")
        return 2
    os.environ.update(sandbox_environment(sandbox))
    sys.path.insert(0, str(REPO_ROOT))

    # Settings and one constant only - NOT the application, whose import opens every database.
    from app.config import settings
    from app.world_model.sqlite_store import DEFAULT_WORLD_MODEL_DB_PATH

    problems = isolation_problems(settings, sandbox, settings.in_data_dir(DEFAULT_WORLD_MODEL_DB_PATH))
    if problems:
        print("refusing: isolation from your real data is not guaranteed, so the application was NOT started:")
        for problem in problems:
            print(f"  - {problem}")
        return 2

    failures: list[str] = []

    def check(ok: bool, label: str) -> None:
        print(f"  [{'ok' if ok else 'FAIL'}] {label}")
        if not ok:
            failures.append(label)

    protected = real_data_files()
    before = digests(protected)
    print(f"Isolation verified: every data file, the log and the workspace resolve inside {sandbox.resolve()}")
    print(f"Protecting {len(protected)} real data file(s) in {REPO_ROOT}")

    from fastapi.testclient import TestClient

    from app import main as jarvis

    print("Start-up")
    check(jarvis.core.self_development_engine is None, "self-development is not wired")

    if not args.live and not args.providers:
        _go_offline(jarvis.core)
    elif not args.live:
        # --providers without --live: the /chat check still uses the stub; the per-provider calls below are the only real ones.
        jarvis.core.provider.generate_reply = lambda message, requirements=None: "ready"

    # Addressed to 127.0.0.1: the loopback Host guard (app/host_guard.py) refuses the TestClient's default "testserver".
    with TestClient(jarvis.app, base_url="http://127.0.0.1") as client:
        print("Endpoints")
        health = client.get("/health")
        check(health.status_code == 200 and health.json().get("status") == "ok", "/health answers ok")

        started = time.monotonic()
        chat = client.post("/chat", json={"message": LIVE_PROMPT})
        elapsed = time.monotonic() - started
        body = chat.json() if chat.headers.get("content-type", "").startswith("application/json") else {}
        reply = str(body.get("reply", ""))
        mode = "real provider chain" if args.live else "stub provider"
        check(chat.status_code == 200 and bool(reply) and bool(body.get("conversation_id")),
              f"/chat round-trip via {mode}: HTTP {chat.status_code}, {elapsed:.1f}s, reply preview {reply[:40]!r}")

    if args.providers:
        from app.providers.factory import create_provider, describe_providers

        print("Providers (one tiny real call each)")
        for info in describe_providers():
            if not info.configured:
                print(f"  [skip] {info.name}: not configured")
                continue
            started = time.monotonic()
            try:
                answer = create_provider(info.name).generate_reply(LIVE_PROMPT)
                check(bool(answer.strip()), f"{info.name}: answered in {time.monotonic() - started:.1f}s")
            except Exception as exc:  # noqa: BLE001 - a provider failure is a result to report, not a crash
                check(False, f"{info.name}: {type(exc).__name__} after {time.monotonic() - started:.1f}s")

    print("Real data")
    after = digests(real_data_files())
    check(after == before, "every real data file is byte-identical to before")
    log_file = Path(settings.log_file) if settings.log_file else None
    check(log_file is not None and log_file.is_file() and log_file.stat().st_size > 0, f"log written to {log_file}")

    print("PASSED" if not failures else f"FAILED: {len(failures)} check(s)")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
