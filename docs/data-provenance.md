# Data Provenance and Evaluation

Historical training and evaluation datasets were removed from `main` and remain
recoverable only through `archive/pre-recovery-2026-08-30`. They must not be used
for 0.2 training, model selection, or release claims.

The reproducible 2026-08-30 audit scanned 155,797 records (172 MB on disk) and found 152,235 unique records, 3,562 exact duplicates, 730 invalid FENs, 304 illegal best-move labels, 48,546 positions with conflicting best-move labels, 14 train/evaluation FEN overlaps, and no recorded license field on any training record. The machine-readable result is [the corpus audit](../artifacts/data-audit/2026-08-30.json). `gemmafischer audit-data` enforces these gates and currently exits with status 4.

A future imported record must contain source and license, source item/game IDs,
full move sequence, normalized FEN, setup move, solution move, role semantics,
evidence-contract version, and transformation history. Splits are frozen by
normalized position plus source-game or puzzle lineage, then checked for exact
duplication, near duplication, transpositions, and source-lineage leakage.

The five diagnostic positions are smoke fixtures, not a quality set. A real
evaluation must be independently sourced, licensed, versioned, immutable, and
lineage-isolated, with position and compare-move tasks scored separately. Freeze
the evaluation set before building training data. Training begins only after a
frozen base model shows a repeated, trainable error pattern and the corpus gate
passes with zero blocking findings. No training recipe or adapter on `main` is
currently supported.

`data/sources.json` is the acquisition authority. It pins the official Lichess
puzzle and evaluation exports published 2026-08-02, their CC0-1.0 license,
canonical URLs, and published SHA-256 checksums. `gemmafischer acquire-data`
streams to a partial file, verifies the complete digest, and only then
atomically publishes the archive. `gemmafischer build-dataset` validates the
archive again, applies the source-documented first setup move, validates the
solution move, rejects repeated normalized positions, creates current
Stockfish evidence and typed lesson targets, and deterministically assigns
complete puzzle lineages to train or evaluation. Generated raw and derived data
are ignored by Git; the source manifest and audit results are reviewable.

The implementation was exercised against the real 304,384,407-byte pinned
archive. A 100-record, 50,000-node proof build produced 94 training and six
evaluation records with zero rejected rows, illegal moves, duplicates,
conflicts, or split overlap. This proves the vertical data path, but the current
audit remains blocked: the production gate requires at least 10,000 training
and 1,000 evaluation records. The current machine-readable state is
[current-main.json](../artifacts/data-audit/current-main.json).

This preparation path is not authorization to train. No adapter is promoted
until both derived splits exist, `audit-data` passes, a base-vs-adapter
evaluation is frozen, and the adapter wins without correctness or latency
regressions.
