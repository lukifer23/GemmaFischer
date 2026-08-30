# Local Security Model

The supported server binds to `127.0.0.1` or `localhost`. It rejects non-loopback Host values, foreign Origin headers, and mutations without the per-launch capability token. The browser receives the token through an HttpOnly, SameSite=Strict cookie.

The player surface contains no training, evaluation, adapter, checkpoint, settings, filesystem-path, or arbitrary process routes. Request models reject extra fields and cap inputs. Static assets are self-hosted under a restrictive Content Security Policy.

Completed analyses are retained locally in a bounded SQLite ledger, and the browser stores its current game in same-origin local storage. Neither path is synchronized or exposed beyond the loopback API. Analysis history contains FENs, engine evidence, and coaching output; users should treat the Mac account as the privacy boundary until encrypted-at-rest storage and a visible clear-history control are implemented.

The player path prohibits `trust_remote_code`, unsafe pickle loading, external CDNs, and “latest checkpoint” discovery. See [SECURITY.md](../SECURITY.md) for reporting.
