# Local Security Model

The supported server binds to `127.0.0.1` or `localhost`. It rejects non-loopback Host values, foreign Origin headers, and mutations without the per-launch capability token. The browser receives the token through an HttpOnly, SameSite=Strict cookie.

The player surface contains no training, evaluation, adapter, checkpoint, settings, filesystem-path, or arbitrary process routes. Request models reject extra fields and cap inputs. Static assets are self-hosted under a restrictive Content Security Policy.

Completed analyses and server-owned sessions are retained in a local SQLite WAL
ledger. The browser stores only the server session ID and view preferences in
same-origin local storage. Neither path is synchronized or intentionally exposed
beyond the loopback API. History contains FENs, move ledgers, engine evidence,
and coaching output; the Mac account remains the privacy boundary because the
database is not encrypted at rest.

The player path prohibits `trust_remote_code`, unsafe pickle loading, external
CDNs, “latest checkpoint” discovery, and network model resolution. The full
profile resolves one pinned revision with local-files-only mode. Lifecycle
commands act only on the PID recorded by the app's atomic process lock and verify
ownership before signaling it. See [SECURITY.md](../SECURITY.md) for reporting.
