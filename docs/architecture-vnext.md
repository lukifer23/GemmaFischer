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
```

The loopback-only FastAPI server exposes only create, poll, cancel, and health routes. Training, evaluation, settings, filesystem, model-switching, and process controls are absent.

One analysis may run and only the newest pending analysis is retained. Every request receives a monotonic generation. Cancelling or replacing a request prevents its late result from becoming authoritative. Separate spawned worker supervision remains a public-preview hardware gate.

Python 3.12 and `uv.lock` are the executable baseline. Stockfish 18 is the only MVP chess authority. Both player workflows are first class. Deterministic coaching is the release baseline; Gemma must beat it.

