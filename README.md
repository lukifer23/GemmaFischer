# GemmaFischer

GemmaFischer 0.2 is an experimental, local-first chess learning session. Play, Stockfish replies, move review, position explanation, and engine-v-engine study all stay on one board.

The supported product is intentionally narrow and real:

- the server owns the session FEN, revision, move history, undo state, and review-to-ply link;
- Stockfish is the only chess authority and runs as one supervised, reusable local process;
- every comparison uses separate, equal-budget searches and never rewrites MultiPV ranking;
- evidence IDs include position, engine binary, and analysis configuration;
- visible factual coaching is rendered from typed evidence and deterministic concept facts;
- an optional qualified text model may select typed lesson claims, but never invent moves, scores, or prose;
- sessions and bounded analysis history persist in local SQLite;
- mutations require a per-launch capability token and the server accepts loopback traffic only.

The pre-recovery application, datasets, adapters, models, and reports were removed from `main` after being preserved in the remote tag `archive/pre-recovery-2026-08-30` (commit `ddff9f2d4ccb0d1d3aacb7f90c385266164c0e87`, tree `6c522a0938165c8d5631b8010fce7071cd8f5a8f`, 51 LFS paths).

## Quickstart

Requirements: macOS or Linux, Python 3.12 through [`uv`](https://docs.astral.sh/uv/), and Stockfish 18.

```bash
git clone https://github.com/lukifer23/GemmaFischer.git
cd GemmaFischer
uv sync --frozen --group dev
brew install stockfish                 # macOS; use your package manager on Linux
export GEMMAFISCHER_STOCKFISH="$(command -v stockfish)"
uv run gemmafischer doctor --profile deterministic
uv run gemmafischer launch
```

`launch` starts exactly one background instance, waits for health, and opens the browser. Lifecycle commands are explicit:

```bash
uv run gemmafischer status
uv run gemmafischer stop
```

For a foreground server, use `uv run gemmafischer serve --open`. The default URL is `http://127.0.0.1:8765`.

Analyze without the browser:

```bash
uv run gemmafischer analyze --example
uv run gemmafischer analyze --offline --mode compare \
  --fen 'r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3' \
  --consider f1b5 --rating 1400-1599
```

## Profiles

- `deterministic` is the baseline: FastAPI, python-chess, persistent Stockfish, evidence schema 2.0, and typed deterministic lessons.
- `full` adds MLX-LM 0.31.3 and the revision-pinned `mlx-community/gemma-4-e2b-it-4bit`. Runtime loading is offline-only; missing or corrupt assets degrade explicitly to deterministic coaching.
- LM Studio models are qualification candidates, not silently selected player runtimes. The local LFM2.5-2.6B 4-bit candidate is smaller, but its current always-thinking checkpoint failed the visible-output and tutoring gates described in the model card.
- `dev` checks portable contributor prerequisites without requiring Stockfish or model assets.

No training command exists in the player. Training is blocked until a newly acquired corpus has complete license and lineage metadata, zero invalid or conflicting labels, and frozen leakage-free evaluation splits. Historical data was audited, found unsafe for training, and archived rather than silently repaired.

The repository pins the official Lichess puzzle and evaluation exports by URL, publication date, CC0 license, and SHA-256 in `data/sources.json`. Acquisition and lesson-record construction are executable but deliberately separate from training:

```bash
uv run gemmafischer acquire-data
uv sync --all-extras
uv run gemmafischer build-dataset --limit 1000 --nodes 50000
uv run gemmafischer audit-data
```

The builder verifies the archive hash, applies the documented Lichess setup move, rejects illegal or repeated positions, creates Stockfish-backed concept evidence and typed lesson targets, and assigns whole puzzle lineages deterministically to train or evaluation. `audit-data` also blocks duplicates, incomplete chess records, corpora below 10,000 training and 1,000 evaluation records, or any license, provenance, legality, conflict, or leakage failure.

## Verification

```bash
uv run gemmafischer verify
uv run gemmafischer repo-audit
uv run gemmafischer benchmark --profile deterministic --requests 100 \
  --output artifacts/qualification/deterministic-local.json
uv run gemmafischer profile-model --requests 21 \
  --output artifacts/qualification/model-profile-local.json
uv run gemmafischer evaluate-accuracy --suite all
uv run gemmafischer evaluate-tutoring --profile full --repetitions 2 \
  --output artifacts/qualification/tutoring-full-local.json
```

An installed LM Studio candidate can be tested through the same real prompt and
tutoring contracts. The weight file is mandatory so the artifact records its
SHA-256 rather than trusting only a mutable model alias:

```bash
uv run gemmafischer profile-model --backend lmstudio \
  --model lfm2.5-2.6b-mlx \
  --model-artifact /absolute/path/to/model.safetensors \
  --output artifacts/qualification/model-profile-lfm-local.json
```

`verify` runs Ruff, strict mypy, and all non-model tests. The suite includes real Stockfish qualification, API security, session revision conflicts, SQLite restart persistence, evidence migration, and data-gate behavior. The generated OpenAPI contract is checked in CI.

The latest implementation pass also ran the actual browser at 900×600: no horizontal overflow, the complete 336 px board remained in the first viewport, only one square was tabbable, f3 highlighted e5/g5/d4/h4/g1, Stockfish completed the reply, and the console stayed clean. Treat that as local evidence, not universal device acceptance.

## Documentation

- [Dependency-ordered execution roadmap](docs/execution-roadmap.md)
- [Architecture and data flow](docs/architecture-vnext.md)
- [Evidence and HTTP contracts](docs/evidence-contract.md)
- [Model runtime and qualification](docs/model-card.md)
- [Training and data policy](docs/data-provenance.md)
- [Performance targets and evidence](docs/performance-vnext.md)
- [Qualification plan](docs/qualification-plan.md)
- [Tutoring review rubric](docs/tutoring-review-rubric.md)
- [Security model](docs/security-model.md)
- [Compatibility and archive recovery](docs/compatibility.md)
- [Third-party notices](docs/third-party-notices.md)

GemmaFischer source is MIT licensed. Stockfish, Gemma, and any future dataset retain their own terms; see the third-party notices before redistribution.
