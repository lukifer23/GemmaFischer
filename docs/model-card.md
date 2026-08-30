# Model Card: GemmaFischer Full Profile Candidate

Status: **target-host smoke passed; quality qualification open**.

The lead candidate is `mlx-community/gemma-4-e2b-it-4bit`, derived from `google/gemma-4-E2B-it`, using MLX-LM. The implementation feeds only validated engine evidence and expects 2–5 typed coaching claims.

The first target-host load on 2026-08-30 exposed a stale MLX-LM 0.28.4 pin, which rejected model type `gemma4`. MLX-LM 0.31.3 added the required architecture. The pinned model revision `238767527555cb75a05732a84dff5d6ba0dd6809` now loads on the M3 Pro/18 GB host and completed a real position-analysis smoke in about 11 seconds. The result state was `complete`, coaching source was `gemma`, and two typed claims passed evidence validation.

Gemma 4 thinking is deliberately disabled for claim selection. With thinking enabled, the E2B model spent the bounded generation on a thought channel. Final JSON may be fenced, so the runtime extracts exactly the first bounded JSON array and then applies the strict discriminated claim schema. Invalid or unsupported claims fail closed to deterministic coaching.

The revision, runtime version, cached size, tokenizer, chat template, and runtime asset hashes are recorded in `assets/model-manifest.json`. Qualification still measures multi-position schema compliance, correctness, unsupported claims, cold/warm latency, RSS, pressure behavior, cancellation, restart, offline reload, and human coaching quality on the named M3 Pro/18 GB host.

The model is excluded if it ties or loses to deterministic coaching, introduces correctness regressions, exceeds resource gates, or lacks reproducible assets.
