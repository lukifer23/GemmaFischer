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
matching the pinned manifest. The builder scans all 6,100,960 source rows and
uses a bounded content-hash reservoir, so source-file order cannot silently
select the corpus. Exact position quotas produce 12,000 train, 1,500 validation,
and 1,500 final-test records from 3,000/375/375 distinct semantic positions.
Each selected position is rendered once for every product rating bucket and all
variants remain in the same partition.

The real 64-row, 5,000-node v2 smoke produced 56/4/4 rows from 14/1/1 semantic
positions with zero rejected rows. It passed the exact contract, legality,
metadata, duplicate, conflict, semantic-position, and lineage gates in
[selector-v2-smoke-2026-09-01.json](../artifacts/data-audit/selector-v2-smoke-2026-09-01.json).
The production [latest.json](../artifacts/data-audit/latest.json) records the
completed 250,000-node build: 12,000/1,500/1,500 rows from 3,000/375/375 unique
semantic positions, zero rejected source rows, and zero blocking contract,
legality, duplicate, conflict, semantic-overlap, or lineage-overlap findings.

The builder emits only `lesson-selection-2.0` examples. Every example stores
the exact production system/user prompt, complete typed `EngineEvidence`, and a
strict ID-only target round-tripped through `parse_lesson_selection`, plus the
engine binary hash, node budget, archive selection method, source item/game/
position lineage, and transformation record. Production requires zero contract,
legality, duplicate, conflict, semantic-position leakage, or lineage-leakage
findings.

The untouched final-test partition is also the only input allowed to
`freeze-question-eval`. The command freezes 1,000 engine-grounded best-move
questions with exact UCI/SAN grading examples and evidence hashes. The frozen
set and deterministic grader passed 1,000/1,000 cases. It never reads training
rows. Optional human gold is separate: 2,500 train records require two
complete independent reviews, the full rubric, an exact-selection agreement of
at least 0.67, and independent adjudication of every selection or material
rubric disagreement. `label-apply` replaces only those reviewed train targets
in a new derived corpus; validation and final-test bytes remain unchanged. When
present, that evidence authorizes pedagogy claims. The technical MLX receipt
instead binds the Stockfish/deterministic corpus to the exact audited source
hashes; preflight rejects stale or unaudited prepared data.

The obsolete `data/create_*finetune_dataset.py`, `data/prepare_dataset.py`, and
historical summary/validation JSON files were removed. They accepted unlicensed
free-form inputs, emitted a different task, and included a stale
`READY_FOR_TRAINING` statement. Git history remains the recovery mechanism;
none is a supported data authority.

This preparation path is not authorization to train. The data audit, frozen
error taxonomy, untuned baseline, exact native-weight hashes, prepared-data
receipt, disk floor, and explicit stage authorization all
pass through a separate preflight. No adapter is promoted until it beats both
deterministic selection and untuned Gemma without correctness, grounding,
latency, memory, or reliability regressions.
