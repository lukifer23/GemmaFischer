# Data Provenance and Evaluation

All historical training datasets remain quarantined. They must not be used for vNext training, model selection, or release claims.

The reproducible 2026-08-30 audit scanned 155,797 records (172 MB on disk) and found 152,235 unique records, 3,562 exact duplicates, 730 invalid FENs, 304 illegal best-move labels, 48,546 positions with conflicting best-move labels, 14 train/evaluation FEN overlaps, and no recorded license field on any training record. The machine-readable result is [the corpus audit](../artifacts/data-audit/2026-08-30.json). `gemmafischer audit-data` enforces these gates and currently exits with status 4.

A future imported record must contain source and license, source item/game IDs, full move sequence, normalized FEN, setup move, solution move, role semantics, and transformation history. Splits are frozen by normalized position plus source-game or puzzle lineage, then checked for exact duplication, near duplication, and transpositions.

Evaluation begins with five diagnostic positions, then must expand to independently sourced and lineage-isolated suites. The existing 200-position file is not held out until its provenance is documented and every overlap is removed. Position and compare-move workflows pass separately. Training begins only after a frozen base model shows a repeated, trainable error pattern and the corpus gate passes with zero blocking findings.
