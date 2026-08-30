# Measured vNext Performance

These numbers are observations from the named local target, not universal performance claims.

## Target

- Apple M3 Pro, 18 GB unified memory, macOS 27.0 arm64
- Python 3.12, Stockfish 18
- MLX-LM 0.31.3 and pinned `mlx-community/gemma-4-e2b-it-4bit`
- Stockfish node budget: 250,000 per analysis

## Stockfish baseline

The clean 20-request run at commit `a3c6cc5` measured mean 0.663 s, p50 0.903 s, p95 1.062 s, and max 1.095 s. Fixture and engine hashes, per-request moves/scores, and raw timings are in [deterministic evidence](../artifacts/qualification/deterministic-2026-08-30.json).

## Full profile

The clean five-position run at commit `7e3f0d2` contained four nonterminal positions and one checkmate:

- valid Gemma output: 4/4 nonterminal positions;
- safe deterministic fallback: 0/4;
- terminal position: 1/1 handled without model invocation;
- total latency: mean 6.355 s across all five, p50 6.967 s, max 10.740 s;
- MLX active memory: 2.60 GB maximum;
- MLX allocation peak: 3.37 GB;
- system memory: 81.4% maximum, 3.60 GB minimum available.

The aggregate includes the near-zero terminal request, so use the raw per-position values for user-facing latency decisions. Full evidence is in [the full-profile artifact](../artifacts/qualification/full-2026-08-30.json).

## Open gates

Five positions are enough to expose failures, not enough to estimate production quality. Release still needs a larger licensed held-out suite, repeated cold/warm runs, token throughput, first-token latency, long engine exhibitions, cancellation-to-resource-release timing, offline restart, memory-pressure behavior, and browser performance on target viewport/device classes.
