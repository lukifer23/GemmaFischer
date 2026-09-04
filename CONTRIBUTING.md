# Contributing

Use Python 3.12 and `uv`. Portable contributions require no model, engine, credentials, or historical LFS assets.

```bash
uv sync --frozen --group dev
uv run gemmafischer doctor --profile dev
uv run gemmafischer verify --tier portable
```

Before changing the player, install Chromium once with `uv run playwright install chromium`,
install Stockfish, and run `uv run gemmafischer verify --tier local-alpha`. The browser test
uses the real local API, engine, and SQLite store. It does not intercept chess requests.

The supported code is under `src/gemmafischer`; its tests are under `tests`. The
pre-recovery implementation is preserved by the annotated
`archive/pre-recovery-2026-08-30` tag, not by duplicate runtime trees on `main`.
Do not restore or import the archived Flask, MoE, LC0, checkpoint, or training
stacks into the supported package.

The primary product path spans `study_domain.py`, `study.py`, `service.py`,
`storage.py`, `web.py`, and `static/study.js`. A change is incomplete if it updates
only the browser or only the API. Preserve the redacted public moment contract,
single Stockfish ownership, exact-revision mutations, and database cascades.

Changes to session persistence, engine evidence, comparison budgets, coaching
validation, model runtime/prompts, lifecycle locking, data gates, or benchmarks
require proportionate target-host requalification. Pull requests must state the
user outcome, affected contracts, tests run, evidence artifact and commit, and
open hardware, browser, data, or human-acceptance gates. A green portable suite
is not a hardware or coaching-quality claim.

The local-alpha model-free suite ratchets coverage at 70% across the whole Python
package. Raise the threshold when tests increase it; do not narrow the measured
source set to make the number green. Training changes must also prove final-test
is absent from the trainer directory and keep the production authorization gate closed.
