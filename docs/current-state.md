# Current State

Status date: 2026-09-07. Version: 0.3.0 on `main`, with the game-to-mastery
loop, Position Lab practice integrity, and deterministic lesson copy repaired
after the 2026-09-04 endurance ledger.

The 0.3 release receipt still binds candidate
`a95396eae5888bdff7347ac79c8a1940f8b1e306`. This work is subsequent product
code on `main`; `verify --tier release` remains a ledger-only check against that
candidate until a new qualification receipt is recorded.

## Implemented

- One-PGN import with exact player-side selection, legal-mainline validation, 400-ply and
  256 KiB limits, source hashing, and sanitized headers.
- One persisted study state machine: parse, screen, deep analysis, ready, cancelled, failed,
  interrupted, and storage-paused.
- One shared Stockfish provider and worker. Screening uses at most 25,000 nodes per compared
  decision; shortlisted moments are re-run at the configured full budget.
- At most three public learning moments, with private engine answers and evidence stored in
  normalized SQLite tables. Deleting a study cascades through moments, attempts, and cards.
- Server-authoritative legal moves, answer-hidden first miss on both Learn and Position Lab
  tutor, Retry to reveal, equivalent-move acceptance, evidence-linked feedback, concept-matched
  near transfer, and spaced review.
- Learn Review tab opens the due moment on the Learn board. Moment cards show
  `new` / `in_progress` / `scheduled` / `mastered`. Promotion is chosen in the shared
  dialog, never auto-queened. The import form collapses while a job is active.
- Position Lab plays the FEN side to move, does not create a session or resume analysis
  until the Lab tab is opened, and disables live-game controls while tutor practice is
  active. Restoring an unanswered tutor turn does not paint the source analysis answer key.
- Deterministic coaching renders principal variations and comparisons in SAN, diagnoses
  a considered miss from matched-budget evidence, and keeps raw concept dumps behind
  cited-evidence disclosure. Learning moments prefer teaching ideas (missed mate, hanging
  piece, fork, back rank, missed capture, failed check evasion) over generic geometry.
  Transfer is offered only for the same teaching idea. Learn has a cited hint that does
  not reveal the move.
- Analysis history reads, study/review/progress reads, and every mutation require the
  per-launch capability token. Follow-up options are shuffled and are never a fixed
  first-option key.
- Adapter-aware optional runtime with an explicit directory, exactly one safetensors file,
  required adapter config, and optional SHA-256 pin.
- Training preparation keeps final-test bytes outside the trainer directory, uses 4096-token
  runs only, supports an explicit one-checkpoint resume, and requires adjudicated human
  pedagogy targets under the production manifest.

## Verified in this worktree

- Ruff and strict mypy pass.
- All model-free Python tests pass, including real Stockfish study generation.
- Real Chromium completed Fool's Mate import, locked hidden first miss, retry reveal,
  due-review navigation onto the Learn board, underpromotion then queen on a mate-in-one
  promotion miss, Progress, reload restoration, Position Lab tutor restore/dismiss,
  cited practice, and 390x844 layout with zero console errors.
- The optional pinned Gemma qualification test is failing honestly: the model
  returned one claim where the schema requires two to five. Runtime fallback remains safe.

## Open acceptance gates

- Physical desktop and VoiceOver acceptance on supported target hardware.
- Study screening checkpointing across gameplay preempt and restart.
- PGN stage timings, cold starts, natural exhibition, reduced-motion/keyboard/CLS.
- Blinded human comparison of actual learning usefulness on the deterministic lesson.
- A fully qualified adapter, if Gemma earns a role. Deterministic behavior remains the
  release baseline until then. Do not train on Stockfish-imitation labels.
