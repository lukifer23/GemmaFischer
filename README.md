# GemmaFischer

GemmaFischer is an experimental, local-first chess coaching research preview. Play, analysis, tutoring, and reviewed engine exhibitions now share one board instead of separate screens. The current session supports:

- click-to-move play against Stockfish with legal-destination highlighting;
- automatic comparison of every player move with verified Stockfish evidence;
- position explanations without leaving the game;
- Stockfish-vs-Stockfish exhibitions with a review after every move;
- local board/session restore and bounded SQLite analysis history.

The supported vNext path is `src/gemmafischer`. Stockfish owns legality, candidate moves, evaluation, WDL, and principal variations. The deterministic coach turns that evidence into a concise lesson. An optional Gemma 4 E2B Q4 path may select typed, evidence-citing claims, but it is not required and silently substituted models are forbidden.

> **Legacy quarantine:** The old Flask/MoE/LC0 application, checkpoints, datasets, reports, performance claims, and uppercase guides remain historical research inputs. They are not verified vNext release evidence. See [the evidence ledger](assets/evidence-status.json).

## Status

- The board, Stockfish replies, engine exhibitions, move review, secure loopback API, and local persistence are implemented and exercised in a real browser.
- Stockfish 18 and the pinned Gemma 4 E2B Q4 model load on the M3 Pro/18 GB target. The latest five-position schema profile served valid grounded Gemma claims for all 4 nonterminal positions and handled mate deterministically. This is a passing schema smoke, not a broad correctness or tutoring-quality gate.
- The real 155,797-record historical corpus is blocked from training: the current audit found invalid positions, illegal labels, duplicate records, conflicting labels, train/eval overlap, and no per-record license fields.
- Portable lint, strict typing, API/security tests, a model smoke, runtime profiles, and machine-readable evidence are available. Broader held-out, cancellation, long-run, device, accessibility, and human coaching gates remain open. The project is not production-ready.

## Quickstart

Requirements: macOS or Linux, Git, [`uv`](https://docs.astral.sh/uv/), and Stockfish 18. The current hardware qualification target is Apple Silicon with Python 3.12.

```bash
git clone https://github.com/lukifer23/GemmaFischer.git
cd GemmaFischer
uv sync --frozen --extra deterministic
brew install stockfish
export GEMMAFISCHER_STOCKFISH="$(command -v stockfish)"
uv run gemmafischer setup --profile deterministic --plan
uv run gemmafischer doctor --profile deterministic
uv run gemmafischer analyze --example
uv run gemmafischer serve --open
```

The server binds only `127.0.0.1` by default. Open `http://127.0.0.1:8765` if the browser does not open automatically.

Compare a considered move:

```bash
uv run gemmafischer analyze \
  --mode compare \
  --fen 'r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3' \
  --consider f1b5 \
  --rating 1400-1599
```

JSON automation:

```bash
uv run gemmafischer analyze --example --format json
uv run gemmafischer version --json
```

## Profiles

- `deterministic`: FastAPI, Pydantic, python-chess, Stockfish evidence, deterministic coaching. This is the default and release baseline.
- `full`: Adds MLX-LM 0.31.3 and pinned Gemma 4 E2B Q4 claims. It fits the target host and passed the five-position schema smoke; deterministic fallback remains mandatory until larger correctness and human-quality gates pass.
- `dev`: Portable contributor tooling. It requires no engine, model, credentials, or LFS asset.

No training command exists in the player CLI. Research/training work must consume frozen, licensed, leakage-checked datasets through a separate future boundary.

Reproduce the current gates:

```bash
uv run gemmafischer benchmark --profile deterministic --requests 20 \
  --output artifacts/qualification/deterministic-local.json
uv run gemmafischer benchmark --profile full --requests 5 \
  --output artifacts/qualification/full-local.json
uv run gemmafischer audit-data --output artifacts/data-audit/local.json
```

`audit-data` currently exits nonzero by design because the quarantined corpus fails the training gate.

## Verify

```bash
uv sync --frozen --group dev
uv run gemmafischer doctor --profile dev
uv run ruff check src/gemmafischer tests_vnext
uv run mypy
uv run pytest -m "not hardware and not model"
```

Tests under `tests_vnext/` are the supported portable gate. Tests under `tests/` exercise quarantined v2 behavior and are not release evidence until individually migrated.

## Documentation

- [Architecture](docs/architecture-vnext.md)
- [Evidence and API contracts](docs/evidence-contract.md)
- [Security model](docs/security-model.md)
- [Data provenance and evaluation](docs/data-provenance.md)
- [Model card and qualification](docs/model-card.md)
- [Measured target-host performance](docs/performance-vnext.md)
- [Compatibility and migration](docs/compatibility.md)
- [Contributing](CONTRIBUTING.md)
- [Security reporting](SECURITY.md)

## License

GemmaFischer source code is MIT licensed. Stockfish, model, dataset, font, and future packaged-asset obligations are tracked separately in [third-party notices](docs/third-party-notices.md). Bundling or distributing an asset is blocked until its manifest and license fields are complete.
