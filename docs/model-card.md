# Model Card: GemmaFischer Full Profile Candidate

Status: **target-host profile passed; schema reliability gate failed**.

The lead candidate is `mlx-community/gemma-4-e2b-it-4bit`, derived from `google/gemma-4-E2B-it`, using MLX-LM. The implementation feeds only validated engine evidence and expects 2–5 typed coaching claims.

The first target-host load on 2026-08-30 exposed a stale MLX-LM 0.28.4 pin, which rejected model type `gemma4`. MLX-LM 0.31.3 added the required architecture. The pinned model revision `238767527555cb75a05732a84dff5d6ba0dd6809` now loads on the M3 Pro/18 GB host.

The clean five-position profile at commit `b24be02` produced valid Gemma claims for 2 of 4 nonterminal positions, deterministic fallback for 2, and deterministic terminal handling for 1. One failure omitted required line-range fields; another returned fewer than two claims. Nonterminal end-to-end latency ranged from 1.52 to 8.50 seconds in that run. MLX active memory peaked at 2.60 GB, MLX allocation peak was 3.47 GB, and total system memory reached 79.2%. See [the full evidence](../artifacts/qualification/full-2026-08-30.json).

Gemma 4 thinking is deliberately disabled for claim selection. With thinking enabled, the E2B model spent the bounded generation on a thought channel. Final JSON may be fenced, so the runtime extracts exactly the first bounded JSON array and then applies the strict discriminated claim schema. Invalid or unsupported claims fail closed to deterministic coaching.

The revision, runtime version, cached size, tokenizer, chat template, and runtime asset hashes are recorded in `assets/model-manifest.json`. The current profile measures schema compliance, cold/warm latency, Stockfish time, process RSS, MLX memory, and system pressure. Broader correctness, cancellation latency, long-run pressure, restart, offline reload, and human coaching quality remain open.

The model is excluded if it ties or loses to deterministic coaching, introduces correctness regressions, exceeds resource gates, or lacks reproducible assets.
