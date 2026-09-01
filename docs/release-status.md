# Public Alpha Release Status

GemmaFischer's deterministic product path is implemented and locally release-gated.
The supported user loop is one workspace: play, receive a cited review, practice
the frozen position, get Stockfish-graded feedback, answer one closed follow-up,
and return to the unchanged game.

## Current gates

| Gate | Status | Evidence boundary |
|---|---|---|
| Portable code, engine-free/model-free tests | Passed locally | Python 3.12 checkout |
| Local-alpha 70% model-free coverage ratchet | Passed locally | Real Stockfish path |
| Repository, OpenAPI, package resources, isolated wheel | Passed locally | Source and built distributions |
| Real Stockfish session and tutor integration | Passed locally | Installed Stockfish 18 |
| Real Chromium core flow at desktop and 390x844 | Passed locally | Headless local Chromium |
| Hosted GitHub Actions | Commit-specific | Verify the exact release SHA in GitHub Actions |
| Physical device and VoiceOver | Open | Human/device acceptance |
| 100-ply and 1,000-request endurance | Open | Target-host release qualification |
| Blind human tutoring usefulness | Open | At least two chess-literate reviewers |
| Optional Gemma release candidate | Open | Deterministic mode remains the product baseline |
| Training | Blocked | Data, toolchain, weights, baseline, and human gates remain unmet |

The machine-readable companion is
[`assets/release-status.json`](../assets/release-status.json). Its
`automated_blockers` field is reserved for failures in locally executable
release checks. External and human acceptance stays explicit in
`open_acceptance_gates`; it is never converted into an automated pass.

## Exact local commands

```bash
uv sync --frozen --group dev
uv run playwright install chromium
uv run gemmafischer verify --tier portable
uv run gemmafischer verify --tier local-alpha
uv run gemmafischer verify --tier release
```

GitHub Actions runs portable verification on Linux and macOS. A separate Linux
job installs Stockfish and Chromium and runs the local-alpha tier. CodeQL scans
Python and JavaScript on pushes to `main`, pull requests, and a weekly schedule.

## Next work, in order

1. Confirm the pushed `main` SHA and its hosted CI/CodeQL results.
2. Expand browser fixtures for promotions, castling, en passant, terminal games,
   restart recovery, keyboard paths, and the full viewport matrix.
3. Reduce or justify rapid-preemption Stockfish restarts, then run endurance,
   memory-growth, persistence, and shutdown gates.
4. Complete physical-device, VoiceOver, and blind human tutoring review.
5. Requalify Gemma only if it improves teaching value without a correctness,
   latency, or memory regression. Do not train until the separate data and
   post-training gates authorize a bounded experiment.
