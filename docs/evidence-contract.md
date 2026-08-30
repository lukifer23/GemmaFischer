# Evidence and API Contract

`EngineEvidence` schema `1.0` preserves normalized six-field FEN, side to move, exact engine identity/hash/options/node budget, terminal reason, zero to three candidates, and deterministic board facts.

Scores, mate values, and WDL use side-to-move perspective. Positive mate means the side to move can force mate. WDL totals 1000. Terminal positions contain zero candidates; non-terminal positions contain one to three.

Candidate and position IDs are SHA-256 hashes over RFC 8785 canonical JSON. Principal variations retain at most 16 plies and the player initially displays eight.

The optional model may return only the closed `CoachingClaim` union. Every position-specific claim must cite known evidence. Unknown IDs and out-of-range PV references are removed with reason codes. Visible prose is rendered from validated payloads.

HTTP routes are:

- `POST /api/v1/analyses` and `GET /api/v1/analyses` for analysis creation and local history;
- `GET` and `DELETE /api/v1/analyses/{analysis_id}` for polling and cancellation;
- `POST /api/v1/board/legal-moves`, `/board/moves`, and `/board/engine-turn` for authoritative play;
- `GET /api/v1/health` for local capability status.

The generated contract is committed as [openapi.json](openapi.json); CI fails on drift. Mutations require the per-launch capability token.
