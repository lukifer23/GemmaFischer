# Contributing

Use Python 3.12 and `uv`. Portable contributions require no model, engine, credentials, or historical LFS assets.

```bash
uv sync --frozen --group dev
uv run gemmafischer doctor --profile dev
uv run ruff check src/gemmafischer tests_vnext
uv run mypy
uv run pytest -m "not hardware and not model"
```

Keep vNext changes under `src/gemmafischer`, `tests_vnext`, current lowercase documentation, and manifests unless a migration-ledger task explicitly covers legacy code. Do not import legacy runtime modules into vNext.

Engine, evidence, validator, model runtime/prompt, lock, or benchmark changes require target-host requalification. Pull requests must state the user outcome, affected contracts, tests, evidence status, and open hardware/human gates.
