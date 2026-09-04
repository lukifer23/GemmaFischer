# Qualification Plan

GemmaFischer is qualified in five separate layers. A green runtime test cannot
substitute for chess accuracy, and automated grounding cannot substitute for a
useful lesson.

## 0. Game-to-mastery behavior

Use real PGNs and Stockfish to prove exact side selection, legal replay, stable
ranking, answer redaction, first-miss secrecy, retry reveal, equivalent-move
acceptance, concept-matched transfer, delayed intervals, restart recovery, and
cascading deletion. Browser acceptance must complete the same loop at desktop
and narrow widths with keyboard-only operation and zero console errors.

The dependency order, implementation slices, and release definition are tracked
in the [execution roadmap](execution-roadmap.md). This document defines the
qualification gates; it is not an implementation backlog.

## 1. Runtime and performance

Record exact commit, worktree state, hardware, engine/model hashes, configuration,
and per-request raw measurements.

| Measure | Development budget | Release evidence |
|---|---:|---|
| Player engine reply p95 | <= 500 ms | 100 sequential turns |
| Position analysis p95 | <= 800 ms | 1,000 requests |
| Compare analysis p95 | <= 1,800 ms | 1,000 requests |
| Warm visible model TTFT p95 | <= 3,000 ms | 20 requests after one warm-up |
| Model generation | >= 20 tokens/s | 20 warm requests |
| Full lesson latency p95 | <= 10,000 ms | 20 warm requests |
| Post-warm memory growth | <= 50 MiB | 1,000 requests |
| Process leaks | 0 | cancellation, failure, and clean exit |

Model profiling must separate model load, prompt tokens, output tokens, time to
first generated token, time to first visible answer token, generation time,
total time, model-reported finish state, process RSS, and system pressure. Direct
MLX runs also record active/peak memory and prompt throughput. A server backend
must mark unavailable telemetry as null rather than estimate it. A 100% visible
output rate is required; hidden reasoning does not satisfy TTFT.

## 2. Chess-position accuracy

Two suites serve different purposes:

1. A checked-in edge suite covers terminal positions, forced moves, mate,
   promotions, castling, en passant, checks, and simple material outcomes with
   explicit expected sets.
2. A deterministic held-out sample from the pinned CC0 Lichess puzzle archive
   measures agreement with independently published solution moves. The first
   archive move is setup; the second is the expected solver move.

Required metrics are legal-move rate, terminal classification, expected top-1
and top-3 hit rate, mate-sign consistency, category breakdown, repeated-run
stability, source hash, selection hash, and rejected-row reasons.

Development gates are 100% legality and terminal correctness, 100% forced/only
move accuracy, at least 80% Lichess top-1 agreement, and at least 95% top-3
agreement. These are product gates, not claims that finite-node Stockfish must
match every puzzle label.

## 3. Automated tutoring correctness

Every served lesson is scored for:

- evidence IDs existing in the same position-scoped evidence package;
- move, score, line, comparison, and concept claims matching their evidence;
- required best-move and considered-move coverage;
- lesson steps citing concepts attached to the recommended candidate;
- rating bucket preservation and schema validity;
- deterministic repeatability or an explicitly recorded model variation;
- safe fallback when model output is invalid or unavailable.

Factual grounding, evidence integrity, schema validity, and required comparison
coverage are zero-tolerance gates: 100% or failure. This suite can prove factual
safety and coverage. It cannot prove that a human learned anything.

## 4. Human tutoring value

Use a frozen, randomized blind comparison of deterministic and model-selected
lessons. At least two reviewers score each lesson from 1 to 5 for correctness,
clarity, relevance, rating fit, actionability, and harmful omission. Reviewers
must not see the producing profile. Disagreements and comments remain in the
artifact.

The candidate is promoted only if it has no correctness regression, mean usefulness is
not worse than deterministic coaching, and no lesson receives a correctness or
harmful-omission score below 3. Until human results exist, status is
`automated-qualified-human-open` at best.

## Commands and artifacts

The qualification commands write JSON atomically under
`artifacts/qualification/`. Raw records are retained, not only aggregates.
Artifacts use `passed`, `failed`, or `blocked` and list every unmet gate. A
blocked missing-model or missing-data run is not rewritten as a pass.

`gemmafischer verify --tier portable` is the cross-platform correctness and
packaging gate. `--tier local-alpha` adds real Stockfish and Chromium acceptance;
`--tier release` also checks the explicit release-status ledger.
Hardware, model, endurance, and human qualification are explicit additional
commands because they consume local resources or require independent judgment.

LM Studio qualification accepts only a literal loopback endpoint and requires
the exact local weight path. The artifact records its byte size and SHA-256.
Reviewer packets and unblinding keys are written to separate files.
