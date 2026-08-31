# Performance Evidence

This page separates the current 0.2 local measurements from historical
baselines. The current run is a development-worktree qualification, not a
release claim.

## Target

- Apple M3 Pro, 18 GB unified memory, macOS 27.0 arm64
- Python 3.12, Stockfish 18
- MLX-LM 0.31.3 and pinned `mlx-community/gemma-4-e2b-it-4bit`
- Stockfish node budget: 250,000 per analysis

## Current 0.2 development-worktree run

Measured locally on 2026-08-30 with the target above:

- 100/100 warm deterministic analyses completed: mean 345.99 ms, p50 335.99
  ms, p95 394.14 ms, max 470.20 ms for the position-only API soak;
- the checked-in 100-request mixed diagnostic run (80 compare, 20 terminal)
  completed with mean 668.40 ms, p50 742.92 ms, p95 1,478.39 ms, and max
  1,696.89 ms. Compare requests intentionally add two equal-node constrained
  searches after MultiPV; terminal requests return immediately;
- page load 47.8 ms, FCP 40 ms, four resources and 31,325 transferred bytes;
- health, session-read, and legal-move endpoints all remained below 2.6 ms;
- engine move commands in a five-ply exhibition averaged 90.17 ms;
- player reply latency was 899-1,120 ms before priority repair and 289.77 ms
  after review submission was moved behind the immediate reply;
- warm-process growth after another 100 analyses was 1,328 KiB;
- Stockfish RSS was stable but high at roughly 485 MiB on this binary even
  after reducing its configured hash from 256 to 64 MiB;
- 375x812, 768x1024, 900x600, and 1280x720 all had no horizontal overflow;
- piece selection, legal/capture markers, player moves, replies, automatic
  review, exhibition play, and pause were exercised in a real browser;
- a post-input layout-shift score of 0.04666 exposed review-panel reflow. The
  empty-state guide was subsequently moved below stable game/review content;
  this source fix still needs a repeated instrumented CLS capture.

The pinned full profile completed two real offline requests in 11.64 and 14.07
seconds. In the final run Gemma supplied the accepted claim selection
(`source=gemma`), Stockfish 18 supplied `Bb5` and the chess evidence, and the
deterministic concept layer produced a cited development lesson. Observed peak
process RSS ranged from 1.35 to 2.13 GiB, and no child process remained after
either exit. See
[the current evidence artifact](../artifacts/qualification/performance-0.2-local.json).

The newer 21-request Gemma selector profile measured warm p95 TTFT 0.505 seconds,
warm p95 total latency 3.995 seconds, and warm decode throughput of at least
70.81 tokens/second, with 3.47 GB peak MLX allocation. The LFM2.5-2.6B LM Studio
candidate decoded hidden reasoning at 69.35-72.61 tokens/second but produced no
visible answer in five of five 768-token prompts; warm p95 total latency was
12.109 seconds. Raw decode TPS is therefore not treated as product success.
See the [candidate bakeoff](../artifacts/qualification/model-bakeoff-0.2-local.json).

## Historical Stockfish baseline

The 20-request run at commit `a3c6cc5` measured mean 0.663 s, p50 0.903 s,
p95 1.062 s, and max 1.095 s. Fixture and engine hashes, per-request
moves/scores, and raw timings are in
[deterministic evidence](../artifacts/qualification/deterministic-2026-08-30.json).
It predates the persistent provider and evidence schema 2.0.

## Historical full profile

The five-position run at commit `7e3f0d2` contained four nonterminal positions
and one checkmate:

- valid Gemma output: 4/4 nonterminal positions;
- safe deterministic fallback: 0/4;
- terminal position: 1/1 handled without model invocation;
- total latency: mean 6.355 s across all five, p50 6.967 s, max 10.740 s;
- MLX active memory: 2.60 GB maximum;
- MLX allocation peak: 3.37 GB;
- system memory: 81.4% maximum, 3.60 GB minimum available.

The aggregate includes the near-zero terminal request. Full raw evidence is in
[the full-profile artifact](../artifacts/qualification/full-2026-08-30.json).
Do not use it as a user-facing 0.2 latency claim.

## Remaining release-scale gates

Release qualification must still measure five cold starts, a 100-ply
exhibition, 1,000 analyses, and a 200-ply persisted session.
Record p50/p95/max wall time, engine time, model first-token and generation time,
RSS/MLX/system pressure, process counts, cancellation-to-resource-release, and
post-warm memory growth. Browser runs cover 375, 768, 900x600, and 1280-pixel
viewports with interaction latency, overflow, layout shift, keyboard behavior,
and accessibility checks. The four target widths and core mouse interactions
are covered; keyboard traversal and post-fix instrumented CLS remain open.
Every release artifact must record commit, lock, fixture,
engine binary, model revision, and configuration hashes.
