# Measurement and Qualification Execution

Status: implemented development gate, release-scale endurance and human review remain open.

This pass converts performance, model output, question grading, and training
readiness from assumptions into executable gates. Every result uses real
Stockfish, a real loopback Uvicorn server, the pinned local Gemma snapshot, or
the pinned local Lichess archive. No chess endpoint was intercepted and no
training was started.

## Runtime and browser

`gemmafischer profile-runtime` starts an isolated real server and temporary
SQLite store, sends 80 HTTP requests across 20 gameplay cycles at 250,000 nodes,
observes descendant Stockfish PIDs, and verifies graceful shutdown. The target
host passed: engine-move p95 310.1 ms, health p95 3.2 ms, legal-move p95 2.1 ms,
maximum one simultaneous Stockfish child, and zero orphans.

The stress run also exposed 19 process restarts. Rapid new gameplay preempted
the preceding review and restarted the shared engine. This preserves correctness
and the one-process contract, but the churn is an explicit endurance and CPU
optimization target rather than a closed performance claim.

A real Chromium navigation measured 44 ms FCP, 33.4 ms full load, four requests,
39,163 transferred bytes, 28,103 JavaScript bytes, and 9,836 CSS bytes. LCP was
unavailable and is recorded as null.

Evidence:

- [runtime qualification](../artifacts/qualification/runtime-2026-08-30.json)
- [browser performance](../artifacts/qualification/browser-performance-2026-08-30.json)

## Chess, tutoring, and questions

The constructed suite now includes rules-sensitive castling, en-passant,
promotion, forced-move, terminal, mate, and material cases. Eleven cases at
250,000 nodes and three repeats passed 24/24 nonterminal top-1 and top-3 checks,
100% legality, terminal correctness, mate consistency, and repeatability.

The deterministic tutoring suite passed 15/15 executions. A separate exact
question-answer contract grades UCI, SAN, and terminal-reason answers without
substring or prose guessing; 33 positive/adversarial examples across eight
frozen questions passed 33/33. These fixture-defined questions validate grading.
The public-alpha pass added the persisted, redacted tutor state machine and board UI.

Evidence:

- [constructed accuracy](../artifacts/qualification/accuracy-constructed.json)
- [question grading](../artifacts/qualification/test-question-grading-2026-08-30.json)
- [deterministic tutoring](../artifacts/qualification/tutoring-deterministic-2026-08-30.json)

## Gemma model contract

The profiler now uses the production 768-token ceiling and sends every response
through the real claim parser and evidence validator. Visible text alone cannot
pass. On the M3 Pro/18 GB host, all 21 requests were transport-successful and
contract-valid. Warm p95 visible TTFT was 654 ms, warm p95 total latency 8.83 s,
minimum warm decode throughput 27.66 tokens/s, and peak MLX allocation 3.47 GB.

The full tutoring profile separately passed 10/10 executions: eight nonterminal
Gemma selections were accepted and merged with required deterministic claims;
two terminal positions correctly bypassed the model. Human usefulness remains
open, so Gemma remains optional and deterministic coaching remains the product
baseline.

Evidence:

- [contract-valid model profile](../artifacts/qualification/model-profile-phase2-current.json)
- [full tutoring qualification](../artifacts/qualification/tutoring-full-phase2-current.json)

## Data and post-training

The builder now emits the exact runtime `lesson-selection-2.0` prompt/target contract
and isolates train, validation, and untouched final-test partitions by lineage
and semantic position. The audit blocks malformed records, illegal moves,
missing provenance, contract failure, duplicates, transpositions, conflicts,
semantic overlap, and lineage overlap.

A real 64-row v2 build from the hash-verified local archive scanned all
6,100,960 source rows, produced 56 train, four validation, and four final-test
rows, and passed the threshold-adjusted smoke audit. Production requires
12,000 / 1,500 / 1,500 rows.

The M3 Pro/18 GB machine is eligible for a bounded LoRA smoke. MLX 0.32.2 and
MLX-LM 0.31.3 are selected. Training remains blocked until the production
corpus passes, native base files are verified locally, and the error taxonomy,
untuned baseline, and two-reviewer adjudicated human gold are frozen. The
installed 4-bit inference quant is not a training source.

Evidence:

- [data audit](../artifacts/data-audit/latest.json)
- [v2 selector smoke](../artifacts/data-audit/selector-v2-smoke-2026-09-01.json)
- [training readiness](../artifacts/training/readiness-latest.json)

## Remaining ordered gates

1. Expand the checked-in real-browser acceptance from the public-alpha core flow
   to the full fixture, accessibility, and viewport matrix.
2. Reduce or justify Stockfish restart churn under rapid gameplay/review load,
   then run 100-ply and 1,000-request endurance/resource gates.
3. Build the full leakage-free 12,000/1,500/1,500 selector corpus and 1,000-case
   held-out chess suite.
4. Complete blinded human usefulness review before promoting Gemma or running
   the authorized 7-20-step MLX-LM smoke.

These are explicit unpassed gates, not placeholders or claimed functionality.
