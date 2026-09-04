# GemmaFischer 0.3 Architecture

The supported application lives under `src/gemmafischer`. The legacy Flask,
inference, MoE, LC0, checkpoint, and training trees were removed from `main` and
are preserved by `archive/pre-recovery-2026-08-30` at commit
`ddff9f2d4ccb0d1d3aacb7f90c385266164c0e87`.

```text
Browser / CLI
  -> typed study, practice, session, analysis, or tutor command
  -> AnalysisService (one study/analysis worker plus gameplay-priority operations)
  -> one token-owned, gameplay-priority StockfishProvider/process
  -> CandidateSet plus optional matched-budget MoveComparisonEvidence
  -> immutable EngineEvidence 2.0 and deterministic concept extraction
  -> deterministic lesson spine / validated optional Gemma ID selection
  -> deterministic rendering
  -> bounded local SQLite studies, moments, attempts, reviews, sessions, and analyses
```

The default Learn path accepts exactly one PGN and persists a `StudyJobView`.
`study.py` validates the complete mainline, extracts only the selected player's
decision positions, screens them at a bounded node budget, and deeply re-analyzes
the shortlist. Public `LearningMomentView` objects omit answers. Private moment
records hold the preferred move, immutable evidence, and an optional concept-matched
transfer position. Practice is graded by a fresh Stockfish comparison, then one
transaction writes the attempt and its review card.

The loopback-only FastAPI server also owns Position Lab games. A `Session` contains the canonical
FEN, mode, status, revision, move ledger, difficulty, and links from plies to
their reviews. Every command carries `expected_revision`; stale mutations fail
with a 409 conflict. The same screen supports player-v-Stockfish play, automatic
move review, position explanation, and reviewed Stockfish-v-Stockfish
exhibition. The browser stores only a session identifier and view preferences.
Training, arbitrary filesystem, model switching, and
arbitrary process-control HTTP routes are absent. Post-training is an explicit
offline CLI workflow with separate manifests, receipts, and preflight gates.
The optional runtime can load only an explicitly configured, locally verified
adapter directory containing one safetensors file and its adapter config.

A tutor interaction copies the completed source evidence and freezes its FEN.
It has its own optimistic revision and moves through `awaiting_answer`,
`awaiting_follow_up`, and a terminal state. Hints cite copied evidence. Answers
are legal-move checked and graded by a fresh equal-budget Stockfish comparison.
The public view omits the answer key, and tutor commands never mutate session
FEN, plies, or revision.

One analysis runs at a time. Only a pending interactive analysis may supersede
another pending interactive analysis; automatic ply reviews are durable queue
entries. Every request receives a monotonic generation. The provider admits one
token-owned operation at a time, gives FIFO gameplay waiters priority, and
preempts/retries analysis when a move is waiting. Targeted cancellation can
close only the exact active analysis token and can never close gameplay.
Blocking session commands run in FastAPI's worker pool, keeping health and
polling responsive. Shutdown alone performs an untargeted provider close.

Analyses (250 unreferenced rows by default), sessions, bounded tutor
interactions are stored in SQLite WAL under macOS Application Support or the
Linux XDG data directory. `GEMMAFISCHER_DATA_DIR` is the explicit override.
Interrupted nonterminal analyses become explicit `ANALYSIS_INTERRUPTED`
failures after restart. Backup-first migrations reject corrupt or future
schemas instead of resetting history. Durable writes use revision/state
compare-and-swap, and create requests can atomically retain idempotency receipts.
`analysis_reservations` protects a durable review during the short interval before
its owning ply is committed. `session_analysis_refs` then protects every retained
ply review from independent pruning and releases it when its session is deleted.
Study jobs, learning moments, attempts, and review cards use normalized tables
with foreign-key cascades. Active study work is recovered as
`paused_interrupted`, never silently restarted. Exhibition pause and resume are revisioned, persisted session mutations rather
than browser-only flags.

Python 3.12 and `uv.lock` are the executable baseline. Stockfish 18 is the only
chess authority. One MultiPV search creates the ordered candidate set. A reviewed
move is compared with the engine choice by independent, equal-node constrained
searches and a 15-centipawn equality tolerance; those searches never rewrite
candidate ranks. Deterministic coaching and typed lesson templates are the
baseline. The optional pinned, offline Gemma runtime may select only supplied
claim, concept, question-template, and hint-template IDs and cannot author
factual chess prose. Mandatory deterministic claims remain present regardless
of the selection. Invalid or unavailable model output degrades to the
deterministic result.
