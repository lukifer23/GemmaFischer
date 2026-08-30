# Model Card: GemmaFischer Full Profile Candidate

Status: **blocked: runtime incompatible**.

The lead candidate is `mlx-community/gemma-4-e2b-it-4bit`, derived from `google/gemma-4-E2B-it`, using MLX-LM. The implementation feeds only validated engine evidence and expects 2–5 typed coaching claims.

The first real target-host load on 2026-08-30 downloaded the candidate successfully, but MLX-LM 0.28.4 rejected model type `gemma4`. The application correctly returned an explicit engine-only result. This combination is not qualified and must not be advertised as working.

Before release, pin repository revision, every file hash, tokenizer, chat template, MLX-LM revision, generation settings, installed size, license, and quantization recipe. Qualification measures schema compliance, correctness, unsupported claims, cold/warm latency, RSS, pressure behavior, cancellation, restart, and offline reload on the named M3 Pro/18 GB host.

The model is excluded if it ties or loses to deterministic coaching, introduces correctness regressions, exceeds resource gates, or lacks reproducible assets.
