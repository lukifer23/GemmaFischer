# Release Status

GemmaFischer 0.3 is an active development worktree. The 0.2 deterministic release
baseline remains the last fully recorded release candidate. New game-to-mastery
code must not inherit those old release claims.

## 0.3 worktree gates

| Gate | Status | Evidence boundary |
|---|---|---|
| Ruff and strict mypy | Passed locally | Current worktree |
| Model-free Python suite | Passed locally | Current worktree, including real Stockfish study test |
| One-PGN study pipeline | Passed locally | Real Stockfish, SQLite, and TestClient smoke |
| Generated OpenAPI | Passed locally | Current worktree matches generated contract |
| Package and isolated-wheel smoke | Passed locally | 0.3.0 sdist and wheel |
| 70% whole-package coverage | Passed locally | 70.48%, 152 model-free tests |
| Real Chromium Learn and Position Lab | Passed locally | Desktop and 390x844, zero console errors |
| Hosted CI and CodeQL | Pending final commit | Exact pushed SHA only |
| Physical device and VoiceOver | Open | Human/device acceptance |
| Study cancel/restart/storage/endurance | Open | Target-host long-running qualification |
| Blind human learning usefulness | Open | At least two chess-literate reviewers |
| Optional adapter | Blocked by design | Human labels and explicit authorization absent |

The machine-readable [`assets/release-status.json`](../assets/release-status.json)
still describes the last 0.2 release candidate. It is historical until regenerated
for an exact 0.3 commit. `verify --tier release` must fail if that ledger claims a
different code candidate or stale artifact hashes.

## Required local commands

```bash
uv sync --frozen --group dev
uv run playwright install chromium
uv run gemmafischer verify --tier portable
uv run gemmafischer verify --tier local-alpha
uv run gemmafischer verify --tier release
```

GitHub Actions must run portable verification on Linux and macOS. The integration
job must install real Stockfish and Chromium. CodeQL and open dependency, secret,
and code-scanning alerts must be checked for the exact final SHA.

## Remaining release order

1. Run runtime, cancellation, restart, storage-pressure, and
   endurance gates. Record raw timings and process counts.
2. Complete physical desktop, narrow viewport, keyboard, and VoiceOver acceptance.
3. Complete blinded human usefulness review.
4. Only then regenerate the machine release ledger for one exact commit and verify
   hosted CI/security state. Adapter work remains a separate, fail-closed track.
