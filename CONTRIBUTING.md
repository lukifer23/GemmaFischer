# Contributing

Use Python 3.12 and `uv`. Portable contributions require no model, engine, credentials, or historical LFS assets.

```bash
uv sync --frozen --group dev
uv run gemmafischer doctor --profile dev
uv run ruff check src/gemmafischer tests
uv run mypy
uv run pytest -m "not model" tests
```

The supported code is under `src/gemmafischer`; its tests are under `tests`. The
pre-recovery implementation is preserved by the annotated
`archive/pre-recovery-2026-08-30` tag, not by duplicate runtime trees on `main`.
Do not restore or import the archived Flask, MoE, LC0, checkpoint, or training
stacks into the supported package.

Changes to session persistence, engine evidence, comparison budgets, coaching
validation, model runtime/prompts, lifecycle locking, data gates, or benchmarks
require proportionate target-host requalification. Pull requests must state the
user outcome, affected contracts, tests run, evidence artifact and commit, and
open hardware, browser, data, or human-acceptance gates. A green portable suite
is not a hardware or coaching-quality claim.
