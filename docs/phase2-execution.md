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
frozen questions passed 33/33. These fixture-defined questions validate grading,
not yet the future persisted tutor-question generator.

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

The builder now emits the exact runtime claim-selection prompt/target contract
and isolates train, validation, and untouched final-test partitions by lineage
and semantic position. The audit blocks malformed records, illegal moves,
missing provenance, contract failure, duplicates, transpositions, conflicts,
semantic overlap, and lineage overlap.

A real 100-row build from the hash-verified local archive produced 84 train, 10
validation, and 6 final-test rows with zero rejected rows and passed the
threshold-adjusted smoke audit. The production gate remains blocked at 10,000 /
1,000 / 1,000 rows.

The M3 Pro/18 GB machine is eligible for a bounded LoRA smoke. Unsloth and
unsloth-zoo are not installed. Training is blocked until the corpus reaches
scale, exact toolchain versions and native base weights are pinned and hashed,
and the error taxonomy, untuned baseline, and blind human evidence are frozen.
The installed 4-bit inference quant is not a training source.

Evidence:

- [data audit](../artifacts/data-audit/current-main.json)
- [selector smoke](../artifacts/data-audit/selector-smoke-2026-08-30.json)
- [training readiness](../artifacts/training/readiness-current.json)

## Remaining ordered gates

1. Check in a durable real-browser E2E harness for the full fixture/viewport
   matrix; this pass measured the browser but did not add the Playwright package.
2. Reduce or justify Stockfish restart churn under rapid gameplay/review load,
   then run 100-ply and 1,000-request endurance/resource gates.
3. Implement the persisted deterministic tutor interaction vertical slice;
   current question fixtures qualify grading but are not the product tutor.
4. Build the full leakage-free 10,000/1,000/1,000 selector corpus and 1,000-case
   held-out chess suite.
5. Complete blinded human usefulness review before promoting Gemma or authorizing
   any 7-20-step Unsloth/MLX-LM smoke comparison.

These are explicit unpassed gates, not placeholders or claimed functionality.
