# Compatibility and Migration

GemmaFischer's supported line is application `0.2.0`, experimental API `v1`, and
evidence schema `2.0`. The 0.2 API replaces browser-owned games with server-owned
`Session` resources and optimistic revisions. The old stateless `/board/*`
routes remain temporarily deprecated for compatibility and are not used by the
player. API or evidence breaks require a compatibility-table entry and explicit
migration note.

Stored schema-1 analysis snapshots are read through the narrow Pydantic migration
path and surfaced as schema 2 objects. They are historical records and cannot
qualify current engine correctness because they lack scoped CandidateSet and
matched-budget comparison evidence.

SQLite store schema version 2 adds transactionally maintained
`session_analysis_refs` and short-lived `analysis_reservations`. Opening an older
0.2 database creates both relations and backfills every still-existing ply review
without changing session or analysis IDs. A durable review is reserved when it is
queued and the reservation is released only after the owning ply reference is
committed. A review already removed by older independent retention cannot be
reconstructed and remains an honest missing historical record.

Legacy MoE settings, adapters, checkpoints, datasets, caches, and reports are not
migrated. Their recovery point is annotated tag
`archive/pre-recovery-2026-08-30`, which resolves to commit
`ddff9f2d4ccb0d1d3aacb7f90c385266164c0e87` and records 51 Git LFS paths in its
tag message. Restore that tag only in a separate checkout. Source rollback uses
Git tags plus the matching lock and asset manifests; there is no self-updater.
