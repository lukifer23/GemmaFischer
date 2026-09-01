# Local Security Model

The supported server binds to `127.0.0.1` or `localhost`. It rejects non-loopback Host values, foreign Origin headers, and mutations without the per-launch capability token. The browser receives the token through an HttpOnly, SameSite=Strict cookie.

The player surface contains no training, evaluation, adapter, checkpoint,
settings, filesystem-path, or arbitrary process routes. Health reports engine
availability without returning the configured or resolved binary path. Request
models reject extra fields, and the HTTP boundary rejects bodies above 64 KiB
before JSON/model parsing even when `Content-Length` is absent or false. Static
assets are self-hosted under a restrictive Content Security Policy.

Completed analyses, server-owned sessions, and redacted tutor state are retained
in a local SQLite WAL ledger. Internal tutor records also contain copied engine
evidence and hidden answer keys; only the redacted view crosses the API. The
browser stores only the server session ID and view preferences in
same-origin local storage. Neither path is synchronized or intentionally exposed
beyond the loopback API. History contains FENs, move ledgers, engine evidence,
and coaching output; the Mac account remains the privacy boundary because the
database is not encrypted at rest.

SQLite is the commit authority for durable mutations. Migrations run inside a
transaction after an integrity check and a backup of populated older schemas;
future or corrupt databases are preserved and rejected rather than reset.
Client-safe errors never contain raw exception text, and the protected recovery
route performs only a bounded write probe and integrity check.

The player path prohibits `trust_remote_code`, unsafe pickle loading, external
CDNs, “latest checkpoint” discovery, and network model resolution. The full
profile resolves one pinned revision with local-files-only mode. Lifecycle
commands act only on the PID recorded by the app's atomic process lock and verify
ownership before signaling it. See [SECURITY.md](../SECURITY.md) for reporting.
