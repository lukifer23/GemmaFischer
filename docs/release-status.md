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
| Hosted GitHub Actions | Passed | Verification `33580143137` and CodeQL `33580143098` passed on candidate `43c09df8140ce4bf5d9182d75c4763279487ada4` |
| Physical device and VoiceOver | Open | Human/device acceptance |
| 100-ply and 1,000-request endurance | Open | Target-host release qualification |
| Blind human tutoring usefulness | Open | At least two chess-literate reviewers |
| Optional Gemma release candidate | Open | Deterministic mode remains the product baseline |
| Post-training | Ready, unauthorized | The audited v2 corpus, frozen question set, real untuned baseline/error taxonomy, machine-supervision MLX data, seven native-file hashes, and technical preflight pass; smoke and production authorization are revoked, no trainer is running, and no adapter exists |

The machine-readable v2 companion is
[`assets/release-status.json`](../assets/release-status.json). It binds local
command and environment receipts, committed artifact hashes, hosted run IDs and
head SHAs, security-alert observations, and typed external gates to one exact
candidate. The only allowed current claim is `local_release_gates_passed`.
External and human acceptance is never converted into an automated pass.

The receipt is committed as a ledger-only child of the verified code candidate.
This avoids an impossible self-referential commit SHA while ensuring any code
change after the candidate invalidates release verification.

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

1. Expand browser fixtures for promotions, castling, en passant, terminal games,
   restart recovery, keyboard paths, and the full viewport matrix.
2. Measure Stockfish recovery and preemption, then run endurance,
   memory-growth, persistence, and shutdown gates.
3. Complete physical-desktop, VoiceOver, and blind human tutoring review.
4. Measure and eliminate unacceptable 1,024-token training truncation, then
   rerun preflight. The native revision and seven required files are verified.
5. Reauthorize the bounded 7-20-step smoke only on explicit request; production
   authorization remains separate. No training process or adapter exists now.
6. Promote only if the single selected adapter beats the deterministic product
   and untuned Gemma without a correctness, latency, memory, reliability, or
   teaching-value regression. A tie is a loss.
7. Run two-reviewer human validation only before making claims about improved
   clarity, pedagogy, or human usefulness.
