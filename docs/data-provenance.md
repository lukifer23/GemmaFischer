# Data Provenance and Evaluation

All current datasets and evaluation files are quarantined. They must not be used for vNext training, model selection, or release claims.

A future imported record must contain source and license, source item/game IDs, full move sequence, normalized FEN, setup move, solution move, role semantics, and transformation history. Splits are frozen by normalized position plus source-game or puzzle lineage, then checked for exact duplication, near duplication, and transpositions.

Evaluation proceeds through five diagnostic positions, five target players, 20 independently created anchors per rating bucket, and an independently held 200-position suite. Position and compare-move workflows pass separately. Training begins only after a frozen model shows a repeated, trainable error pattern.

