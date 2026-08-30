# Compatibility and Migration

GemmaFischer vNext starts at application `0.1.0`, API `v1`, and evidence schema `1.0`. API or evidence breaks require a compatibility-table entry and explicit migration note.

Legacy MoE settings, adapters, checkpoints, datasets, caches, and reports are not automatically migrated. Each receives an evidence-ledger decision before reuse or later deletion.

Source rollback uses Git release tags plus matching lock and asset manifests. A self-updating command is deferred until a packaged installer owns versioned directories and can atomically restore the prior bundle.

