# Data Provenance and Evaluation

Historical training and evaluation datasets were removed from `main` and remain
recoverable only through `archive/pre-recovery-2026-08-30`. They must not be used
for 0.2 training, model selection, or release claims.

The reproducible 2026-08-30 audit scanned 155,797 records (172 MB on disk) and found 152,235 unique records, 3,562 exact duplicates, 730 invalid FENs, 304 illegal best-move labels, 48,546 positions with conflicting best-move labels, 14 train/evaluation FEN overlaps, and no recorded license field on any training record. The machine-readable result is [the corpus audit](../artifacts/data-audit/2026-08-30.json). `gemmafischer audit-data` enforces these gates and currently exits with status 4.

A supported imported record must contain source and license, source item/game
IDs, full move sequence, normalized FEN, setup move, solution move, split role,
evidence and model-contract versions, engine configuration, and transformation
history. Audit schema 2.0 freezes `train`, `validation`, and untouched
`final_test` partitions. It compares semantic positions using the first four
FEN fields, so clock-only changes cannot evade deduplication, and separately
blocks source-lineage overlap across every partition pair.

The five diagnostic positions are smoke fixtures, not a quality set. A real
evaluation must be independently sourced, licensed, versioned, immutable, and
lineage-isolated, with position and compare-move tasks scored separately. Freeze
the evaluation set before building training data. Training begins only after a
frozen base model shows a repeated, trainable error pattern and the corpus gate
passes with zero blocking findings. No training recipe or adapter on `main` is
currently supported.

`data/sources.json` is the only acquisition authority. It pins the official Lichess
puzzle and evaluation exports published 2026-08-02, their CC0-1.0 license,
canonical URLs, and published SHA-256 checksums. `gemmafischer acquire-data`
streams to a partial file, verifies the complete digest, and only then
atomically publishes the archive. `gemmafischer build-dataset` validates the
archive again, applies the source-documented first setup move, validates the
solution move, rejects repeated normalized positions, creates current
Stockfish evidence and typed lesson targets, and deterministically assigns
complete puzzle lineages to train or evaluation. Generated raw and derived data
are ignored by Git; the source manifest and audit results are reviewable.

The local archive is 304,384,407 bytes and its SHA-256 is
`a0ea9129c6b6434dfb34a9ac4ec660c9cfff22b2de465e01854f018fc847f073`,
matching the pinned manifest. A new real 100-row, 5,000-node Stockfish smoke
build produced 84 train, 10 validation, and 6 final-test rows with zero rejected
source rows. Every row passed the exact prompt/target contract, legality,
metadata, deduplication, conflict, semantic-position isolation, and lineage
isolation checks. Its threshold-adjusted proof is
[selector-smoke-2026-08-30.json](../artifacts/data-audit/selector-smoke-2026-08-30.json).
The production [current-main.json](../artifacts/data-audit/current-main.json)
remains blocked because 84/10/6 is below the required 10,000/1,000/1,000. A
full build is not authorized merely because the archive is present.

The builder now emits only `claim-selection-1.0` examples. Each example stores
the exact production system/user prompt, complete typed `EngineEvidence`, a
target that is round-tripped through `parse_claim_selection`, the engine binary
hash and node budget, and complete source/transformation lineage. Puzzle
lineages are assigned deterministically to 80% train, 10% validation, and 10%
final test before publishing. The production gate requires at least
10,000/1,000/1,000 rows respectively and zero contract, legality, duplication,
conflict, semantic-position leakage, or lineage-leakage failures.

The obsolete `data/create_*finetune_dataset.py`, `data/prepare_dataset.py`, and
historical summary/validation JSON files were removed. They accepted unlicensed
free-form inputs, emitted a different task, and included a stale
`READY_FOR_TRAINING` statement. Git history remains the recovery mechanism;
none is a supported data authority.

This preparation path is not authorization to train. No adapter is promoted
until all three derived splits exist, `audit-data` passes, a base-vs-adapter
evaluation is frozen, and the adapter wins without correctness or latency
regressions.
