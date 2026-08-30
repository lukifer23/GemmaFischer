# GemmaFischer 0.2 Architecture

The supported application lives under `src/gemmafischer`. The legacy Flask,
inference, MoE, LC0, checkpoint, and training trees were removed from `main` and
are preserved by `archive/pre-recovery-2026-08-30` at commit
`ddff9f2d4ccb0d1d3aacb7f90c385266164c0e87`.

```text
Browser / CLI
  -> typed session command or ad hoc analysis request
  -> AnalysisService (one newest-pending analysis worker)
  -> one locked, persistent StockfishProvider
  -> CandidateSet plus optional matched-budget MoveComparisonEvidence
  -> immutable EngineEvidence 2.0 and deterministic concept extraction
  -> deterministic LessonPlan / validated optional Gemma claim selection
  -> deterministic rendering
  -> bounded local SQLite analysis and session history
```

The loopback-only FastAPI server owns games. A `Session` contains the canonical
FEN, mode, status, revision, move ledger, difficulty, and links from plies to
their reviews. Every command carries `expected_revision`; stale mutations fail
with a 409 conflict. The same screen supports player-v-Stockfish play, automatic
move review, position explanation, and reviewed Stockfish-v-Stockfish
exhibition. The browser stores only a session identifier and view preferences.
Training, adapter, checkpoint, arbitrary filesystem, model switching, and
arbitrary process-control routes are absent.

One ad hoc analysis runs at a time and only the newest pending analysis is kept.
Every request receives a monotonic generation. Cancellation invalidates late
results and interrupts an active UCI command; the provider restarts cleanly on
the next request. The service serializes Stockfish access behind one provider,
so analyses and session play do not create one engine per request. Shutdown
closes the worker, engine, store, and process lock. Analyses (250 by default) and
sessions are stored in SQLite WAL under
`~/Library/Application Support/GemmaFischer/`; interrupted nonterminal analyses
become explicit `ANALYSIS_INTERRUPTED` failures after restart.

Python 3.12 and `uv.lock` are the executable baseline. Stockfish 18 is the only
chess authority. One MultiPV search creates the ordered candidate set. A reviewed
move is compared with the engine choice by independent, equal-node constrained
searches and a 15-centipawn equality tolerance; those searches never rewrite
candidate ranks. Deterministic coaching and typed lesson templates are the
baseline. The optional pinned, offline Gemma runtime may select only validated
claim objects and cannot author factual chess prose. Invalid or unavailable
model output degrades to the deterministic result.
