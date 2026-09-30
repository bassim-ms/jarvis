"""
The admin command for devices - the ONLY place a device is registered or trusted:

    python -m app.devices.admin list
    python -m app.devices.admin register --name "My phone" --type phone [--credential-file FILE]
    python -m app.devices.admin trust "My phone"
    python -m app.devices.admin revoke "My phone"
    python -m app.devices.admin register-pc [--name PC] [--credential-file FILE]

It runs on this machine, against the application's own database, with the same access as whoever runs it - there
is no network path and no chat path to it, so a device (or a message) cannot register or trust itself. Registering
issues the device's credential ONCE: it is written to a file readable only by the current user (never printed, never
logged) or, without --credential-file, printed once. Only the credential's hash is stored.

`register-pc` is the one explicit, deliberate admin act that registers THIS computer and trusts it. Nothing is ever
trusted because a request came from localhost.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable

from app.devices.base import Device, DeviceError, DeviceStatus, DeviceType, TrustState
from app.devices.registry import DeviceRegistry
from app.devices.store import SQLiteDeviceStore

DEFAULT_PC_CREDENTIAL_FILE = "jarvis_pc_device.json"


def build_registry() -> DeviceRegistry:
    from app.config import settings
    from app.orchestration.factory import create_default_job_store

    return DeviceRegistry(SQLiteDeviceStore.for_job_store(create_default_job_store(settings.orchestration_db_path)))


def write_credential_file(path: str, device: Device, credential: str, *, force: bool = False) -> None:
    """Write {device_id, credential} for the device's own client; owner-only where the OS supports it."""
    flags = os.O_WRONLY | os.O_CREAT | (os.O_TRUNC if force else os.O_EXCL)
    fd = os.open(path, flags, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump({"device_id": device.device_id, "credential": credential}, handle)


def describe(device: Device) -> str:
    return f"{device.device_id}  {device.name}  [{device.device_type.value}]  {device.trust_state.value}  {device.status.value}"


def _register(registry, args, out, *, trust: bool, default_name: str | None = None, default_type: str | None = None) -> int:
    name = args.name or default_name
    kind = getattr(args, "type", None) or default_type
    caps = [c.strip() for c in args.capabilities.split(",") if c.strip()] if getattr(args, "capabilities", None) else None
    status = DeviceStatus.OFFLINE if getattr(args, "offline", False) else DeviceStatus.UNKNOWN
    device, credential = registry.register(name, kind, caps, status=status)
    if trust:
        device = registry.trust(device.device_id)
    target = args.credential_file or (DEFAULT_PC_CREDENTIAL_FILE if trust else None)
    if target:
        write_credential_file(target, device, credential, force=args.force)
        out(f"Registered: {describe(device)}\nCredential written to {target} (keep it private; it is shown nowhere else).")
    else:
        out(f"Registered: {describe(device)}\nCredential (shown ONCE - store it now): {credential}")
    if not trust:
        out("The device is PENDING: it cannot authenticate until you run: trust <name or id>")
    return 0


def main(argv: list[str] | None = None, *, registry: DeviceRegistry | None = None, out: Callable[[str], object] = print) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.devices.admin", description="Manage Jarvis devices (local admin only).")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="list registered devices")
    reg = sub.add_parser("register", help="register a new (untrusted) device")
    reg.add_argument("--name", required=True)
    reg.add_argument("--type", required=True, choices=[t.value for t in DeviceType])
    reg.add_argument("--capabilities", help="comma-separated declarations; default depends on the type")
    reg.add_argument("--credential-file")
    reg.add_argument("--force", action="store_true", help="overwrite an existing credential file")
    reg.add_argument("--offline", action="store_true", help="register as OFFLINE (for test devices)")
    for name in ("trust", "revoke"):
        p = sub.add_parser(name, help=f"{name} a device by name or id")
        p.add_argument("device")
    pc = sub.add_parser("register-pc", help="register THIS computer and trust it (one explicit admin act)")
    pc.add_argument("--name", default="PC")
    pc.add_argument("--credential-file", default=DEFAULT_PC_CREDENTIAL_FILE)
    pc.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    registry = registry or build_registry()
    try:
        if args.command == "list":
            devices = registry.list_devices()
            out("\n".join(describe(d) for d in devices) if devices else "No devices registered.")
            return 0
        if args.command == "register":
            return _register(registry, args, out, trust=False)
        if args.command == "register-pc":
            return _register(registry, args, out, trust=True, default_type=DeviceType.COMPUTER.value)
        device = registry.find(args.device)
        if device is None:
            raise DeviceError("unknown device")
        changed = registry.trust(device.device_id) if args.command == "trust" else registry.revoke(device.device_id)
        out(f"{args.command.capitalize()}ed: {describe(changed)}" if args.command == "revoke" else f"Trusted: {describe(changed)}")
        return 0
    except (DeviceError, FileExistsError) as exc:
        out(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
