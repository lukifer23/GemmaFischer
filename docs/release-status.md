# Release Status

GemmaFischer 0.3 is committed on `main`. The machine ledger binds the verified
code candidate `dd8836df208999ace30fcfc2dfdea48fb86129fd`; this documentation and
ledger update is its allowed ledger-only child.

## 0.3 worktree gates

| Gate | Status | Evidence boundary |
|---|---|---|
| Ruff and strict mypy | Passed locally | Current worktree |
| Model-free Python suite | Passed locally | Current worktree, including real Stockfish study test |
| One-PGN study pipeline | Passed locally | Real Stockfish, SQLite, and TestClient smoke |
| Generated OpenAPI | Passed locally | Current worktree matches generated contract |
| Package and isolated-wheel smoke | Passed locally | 0.3.0 sdist and wheel |
| 70% whole-package coverage | Passed locally | 70.59%, 153 model-free tests |
| Real Chromium Learn and Position Lab | Passed locally | Desktop and 390x844, zero console errors |
| Hosted CI and CodeQL | Passed | Verification `33896792787` and CodeQL `33896792895` on the exact candidate SHA |
| Physical device and VoiceOver | Open | Human/device acceptance |
| Study cancel/restart/storage/endurance | Open | Target-host long-running qualification |
| Blind human learning usefulness | Open | At least two chess-literate reviewers |
| Optional adapter | Blocked by design | Human labels and explicit authorization absent |

The machine-readable [`assets/release-status.json`](../assets/release-status.json)
binds command receipts, committed artifact hashes, hosted run IDs, security-alert
observations, and typed external gates to that exact candidate. The only current
claim is `local_release_gates_passed`; open human and endurance gates remain open.

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
4. Update the ledger again only after new code or new external evidence. Adapter
   work remains a separate, fail-closed track.
