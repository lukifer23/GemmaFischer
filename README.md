# GemmaFischer

GemmaFischer 0.2 is a public-alpha, local-first chess coach. Play Stockfish, inspect a cited move review, practice the reviewed position on the same board, get deterministic feedback, and return to the untouched game. Engine-v-engine study uses the same session and review path.

The supported product is intentionally narrow and real:

- the server owns the session FEN, revision, move history, undo state, and review-to-ply link;
- Stockfish is the only chess authority and runs as one supervised, reusable local process;
- every comparison uses separate, equal-budget searches and never rewrites MultiPV ranking;
- evidence IDs include position, engine binary, and analysis configuration;
- visible factual coaching is rendered from typed evidence and deterministic concept facts;
- an optional qualified text model may select typed lesson claims, but never invent moves, scores, or prose;
- sessions and bounded analysis history persist in local SQLite;
- guided practice is a persisted, revisioned state machine with hidden answer keys and real Stockfish grading;
- mutations require a per-launch capability token and the server accepts loopback traffic only.

The pre-recovery application, datasets, adapters, models, and reports were removed from `main` after being preserved in the remote tag `archive/pre-recovery-2026-08-30` (commit `ddff9f2d4ccb0d1d3aacb7f90c385266164c0e87`, tree `6c522a0938165c8d5631b8010fce7071cd8f5a8f`, 51 LFS paths).

## Quickstart

Requirements: macOS or Linux, Python 3.12 through [`uv`](https://docs.astral.sh/uv/), and Stockfish 18.

```bash
git clone https://github.com/lukifer23/GemmaFischer.git
cd GemmaFischer
uv sync --frozen --group dev
uv run gemmafischer setup --plan --profile deterministic
uv run gemmafischer setup --repair --yes --profile deterministic
uv run gemmafischer doctor --profile deterministic
uv run gemmafischer launch
```

`setup --plan` is read-only. Confirmed repair supports Homebrew on macOS and
apt-based Debian/Ubuntu, then re-verifies the installed engine. The full profile
adds one exact Gemma revision; it never discovers or installs a second checkpoint.

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

Post-training is not active product work. There is no training command, adapter,
training checkpoint, or training environment on `main`. Historical data and the
old readiness gate remain inspectable only as fail-closed governance records.
Model work stops unless the one pinned Gemma inference candidate first beats the
deterministic tutor in a frozen blinded human evaluation. A tie is a loss.

## Verification

```bash
uv run gemmafischer verify --tier portable
uv run gemmafischer verify --tier local-alpha
uv run gemmafischer verify --tier release
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

`portable` runs Ruff, strict mypy, engine-free/model-free tests, JavaScript syntax, repository and OpenAPI drift audits, dependency compatibility, distribution builds, and an isolated installed-wheel smoke test. `local-alpha` adds the 70% model-free whole-package coverage ratchet, real Stockfish tests, and the real Chromium flow. `release` also enforces the checked-in release-status ledger. Optional model tests remain separate because missing Gemma assets must not block the deterministic product.

The durable browser gate launches a real FastAPI server, real Stockfish,
temporary SQLite, and Chromium. It proves position analysis, persisted tutor
restore and dismissal across reload, cited hint display, legal board answer,
evidence-based grading, follow-up completion, return to an unchanged live FEN,
desktop column order, zero console errors, and no horizontal overflow at
390×844. Physical-device, VoiceOver, endurance, human-usefulness, and
optional-model release gates remain open and are listed in
[release status](docs/release-status.md).

## Documentation

- [Dependency-ordered execution roadmap](docs/execution-roadmap.md)
- [Phase 1 execution contract](docs/phase1-execution.md)
- [Measurement and qualification execution](docs/phase2-execution.md)
- [Current public-alpha release status](docs/release-status.md)
- [Historical clean stop and resume point](docs/resume-2026-08-30.md)
- [Architecture and data flow](docs/architecture-vnext.md)
- [Evidence and HTTP contracts](docs/evidence-contract.md)
- [Model runtime and qualification](docs/model-card.md)
- [Training and data policy](docs/data-provenance.md)
- [Archived post-training readiness record](docs/training-readiness.md)
- [Performance targets and evidence](docs/performance-vnext.md)
- [Runtime qualification](docs/runtime-qualification.md)
- [Qualification plan](docs/qualification-plan.md)
- [Tutoring review rubric](docs/tutoring-review-rubric.md)
- [Security model](docs/security-model.md)
- [Compatibility and archive recovery](docs/compatibility.md)
- [Third-party notices](docs/third-party-notices.md)

GemmaFischer source is MIT licensed. Stockfish, Gemma, and any future dataset retain their own terms; see the third-party notices before redistribution.
