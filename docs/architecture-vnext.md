# GemmaFischer vNext Architecture

The supported application lives entirely under `src/gemmafischer`. It must not import the legacy Flask application, inference monolith, MoE router, adapters, checkpoints, datasets, or reports.

```text
Browser / CLI
  -> typed application request
  -> single-worker AnalysisService
  -> fresh StockfishProvider
  -> immutable EngineEvidence
  -> deterministic coach or optional Gemma selector
  -> claim validation
  -> deterministic player rendering
  -> bounded local SQLite snapshot history
```

The loopback-only FastAPI server exposes analysis create/list/poll/cancel, legal-move lookup, player turns, engine turns, and health routes. Training, evaluation, settings, arbitrary filesystem, model-switching, and process-control routes are absent. The browser uses the same board for player games, position explanations, move tutoring, and reviewed engine exhibitions.

One analysis may run and only the newest pending analysis is retained. Every request receives a monotonic generation. Cancelling or replacing a request prevents its late result from becoming authoritative. Snapshots are written to a 250-entry SQLite WAL ledger in `~/Library/Application Support/GemmaFischer/`; interrupted nonterminal records become explicit `ANALYSIS_INTERRUPTED` failures on restart. Browser game state is kept in same-origin local storage. Separate spawned worker supervision that can terminate an active engine/model call remains a public-preview gate.

Python 3.12 and `uv.lock` are the executable baseline. Stockfish 18 is the only vNext chess authority. Deterministic coaching is the release baseline; Gemma may select only typed, evidence-citing claims and must fail closed when its output is invalid.
