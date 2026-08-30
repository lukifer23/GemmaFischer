# GemmaFischer

GemmaFischer is an experimental, local-first chess coaching research preview. It offers two equal workflows:

- explain a position using verified Stockfish evidence;
- compare a legal move the player considered against the engine recommendation.

The supported vNext path is `src/gemmafischer`. Stockfish owns legality, candidate moves, evaluation, WDL, and principal variations. The deterministic coach turns that evidence into a concise lesson. An optional Gemma 4 E2B Q4 path may select typed, evidence-citing claims, but it is not required and silently substituted models are forbidden.

> **Legacy quarantine:** The old Flask/MoE/LC0 application, checkpoints, datasets, reports, performance claims, and uppercase guides remain historical research inputs. They are not verified vNext release evidence. See [the evidence ledger](assets/evidence-status.json).

## Status

- vNext deterministic domain, Stockfish adapter, CLI, secure loopback API, and responsive player UI are implemented.
- Portable lint, type, contract, security, and API tests are available.
- This checkout does not currently contain a verified Stockfish binary or a qualified Gemma asset. `setup` and `doctor` report these honestly.
- Hardware benchmarks, clean-machine reproduction, held-out evaluation, external-player testing, and Gemma qualification remain release gates. The project is not production-ready.

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
- `full`: Adds MLX-LM and the pinned Gemma 4 E2B Q4 candidate. This profile remains unqualified until the hardware and human gates pass.
- `dev`: Portable contributor tooling. It requires no engine, model, credentials, or LFS asset.

No training command exists in the player CLI. Research/training work must consume frozen, licensed, leakage-checked datasets through a separate future boundary.

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
- [Compatibility and migration](docs/compatibility.md)
- [Contributing](CONTRIBUTING.md)
- [Security reporting](SECURITY.md)

## License

GemmaFischer source code is MIT licensed. Stockfish, model, dataset, font, and future packaged-asset obligations are tracked separately in [third-party notices](docs/third-party-notices.md). Bundling or distributing an asset is blocked until its manifest and license fields are complete.
