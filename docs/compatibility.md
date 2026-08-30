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

Legacy MoE settings, adapters, checkpoints, datasets, caches, and reports are not
migrated. Their recovery point is annotated tag
`archive/pre-recovery-2026-08-30`, which resolves to commit
`ddff9f2d4ccb0d1d3aacb7f90c385266164c0e87` and records 51 Git LFS paths in its
tag message. Restore that tag only in a separate checkout. Source rollback uses
Git tags plus the matching lock and asset manifests; there is no self-updater.
