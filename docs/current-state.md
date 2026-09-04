# Current State

Status date: 2026-09-04. Version: 0.3.0 release candidate on `main`.

## Implemented

- One-PGN import with exact player-side selection, legal-mainline validation, 400-ply and
  256 KiB limits, source hashing, and sanitized headers.
- One persisted study state machine: parse, screen, deep analysis, ready, cancelled, failed,
  interrupted, and storage-paused.
- One shared Stockfish provider and worker. Screening uses at most 25,000 nodes per compared
  decision; shortlisted moments are re-run at the configured full budget.
- At most three public learning moments, with private engine answers and evidence stored in
  normalized SQLite tables. Deleting a study cascades through moments, attempts, and cards.
- Server-authoritative legal moves, answer-hidden first retry, equivalent-move acceptance,
  evidence-linked feedback, concept-matched near transfer, and spaced review.
- Learn, Review, Progress, and Position Lab navigation. The previous live-play path remains
  operational and unchanged in purpose.
- Adapter-aware optional runtime with an explicit directory, exactly one safetensors file,
  required adapter config, and optional SHA-256 pin.
- Training preparation keeps final-test bytes outside the trainer directory, uses 4096-token
  runs only, supports an explicit one-checkpoint resume, and requires adjudicated human
  pedagogy targets under the production manifest.

## Verified in this worktree

- Ruff and strict mypy pass.
- All model-free Python tests pass, including real Stockfish study generation.
- A real TestClient, SQLite, and Stockfish flow imported Fool's Mate and reached `ready` with
  ranked learning moments.
- Real Chromium completed import, hidden first miss, retry reveal, progress,
  reload restoration, Position Lab tutor restoration, and 390x844 layout with
  zero console errors.
- The optional pinned Gemma qualification test is failing honestly: the model
  returned one claim where the schema requires two to five. Runtime fallback remains safe.

Portable verification passes with 133 tests. Local-alpha verification passes
with 153 model-free tests, 70.59% whole-package coverage, real Stockfish, package
and isolated-wheel checks, and the expanded real Chromium flow. Release-ledger
and hosted checks still require a final commit.

## Open acceptance gates

- Physical desktop and narrow-screen observation on supported target hardware.
- VoiceOver keyboard and spoken-order acceptance.
- Long-running import/cancel/restart/storage-pressure endurance.
- Blinded human comparison of actual learning usefulness.
- A fully qualified adapter, if Gemma earns a role. Deterministic behavior remains the release
  baseline until then.
