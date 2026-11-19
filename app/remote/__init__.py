"""
Secure remote access to this Jarvis from your own, explicitly approved devices - without opening a router port.

TRANSPORT (no port forwarding, nothing public): Jarvis keeps listening on 127.0.0.1 only. Reaching it from elsewhere goes
through an encrypted, outbound-only private network such as Tailscale (free personal plan; WireGuard): `tailscale serve`
forwards HTTPS on the computer's private tailnet name (e.g. jarvis-pc.tail1234.ts.net) to 127.0.0.1:8000. Only devices signed
in to YOUR tailnet can reach that name; nothing is published to the internet (do NOT use `tailscale funnel`). Jarvis answers a
request addressed to that name only when remote access is ON and only on `/remote/...` - the dashboard, the anonymous /chat,
/providers/status, /dev and the API docs stay loopback-only (app/host_guard.py).

DEVICES (reusing app/devices): a remote device is an ordinary registered device. It is registered and trusted ONLY with the
local admin command (`python -m app.devices.admin register ... --credential-file` then `trust`) - the app itself never
registers or trusts anything, so approval needs your hands on this computer. Its credential goes to the device's own
credential file; only its hash is stored here.

SESSIONS AND SIGNATURES (app/remote/sessions.py, signing.py): a device proves its credential once to open a session (12 h by
default, in memory only); after that every request is signed with the session key, carries a fresh timestamp and a never-reused
nonce, and is checked for replay, tampering and expiry. The device's trust is re-checked on every request, so revoking it on
the dashboard (or with the admin command) ends its access at once.

PERMISSIONS: a remote chat message goes through exactly the same path as /device/chat - the same ToolRegistry, permission
levels (READ / SAFE_ACTION run; CONFIRM / DANGEROUS are never executed from chat - they come back as "confirmation required"),
capability boundary and confirmation machinery. Remote access adds no permission.

OFF BY DEFAULT: `JARVIS_REMOTE_ACCESS_ENABLED=false`; with it off every /remote route answers 403 and a non-loopback Host is
refused for every route, exactly as before.
"""
