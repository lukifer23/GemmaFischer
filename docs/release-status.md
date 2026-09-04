# Release Status

GemmaFischer 0.3 is committed on `main`. The machine ledger binds the verified
evidence candidate `a95396eae5888bdff7347ac79c8a1940f8b1e306`; this documentation and
ledger update is its allowed ledger-only child.

## 0.3 worktree gates

| Gate | Status | Evidence boundary |
|---|---|---|
| Ruff and strict mypy | Passed locally | Current worktree |
| Model-free Python suite | Passed locally | Current worktree, including real Stockfish study test |
| One-PGN study pipeline | Passed locally | Real Stockfish, SQLite, and TestClient smoke |
| Generated OpenAPI | Passed locally | Current worktree matches generated contract |
| Package and isolated-wheel smoke | Passed locally | 0.3.0 sdist and wheel |
| 70% whole-package coverage | Passed locally | 70.28%, 161 local-alpha tests; 140 portable tests |
| Real Chromium Learn and Position Lab | Passed locally | Desktop and 390x844, zero console errors |
| Runtime endurance and recovery core | Passed locally | 1,000 real-engine cycles, 200-ply interruption/recovery, storage lock/retry, SQLite integrity, bounded memory, zero orphans |
| Hosted CI and CodeQL | Passed | Verification `33911187141` and CodeQL `33911187191` on the exact candidate SHA; zero open code, dependency, or secret alerts |
| Physical device and VoiceOver | Open | Human/device acceptance |
| Process-restart churn | Passed locally | One Stockfish PID across 1,000 cycles; zero replacements and an explicit zero-restart gate |
| Performance completion | Open | Cold starts, natural exhibition, stage timings, reduced-motion/keyboard/CLS, and cross-device latency remain |
| Blind human learning usefulness | Open | At least two chess-literate reviewers |
| Optional adapter | Blocked by design | Human labels and explicit authorization absent |

The machine-readable [`assets/release-status.json`](../assets/release-status.json)
binds command receipts, committed artifact hashes, hosted run IDs, security-alert
observations, and typed external gates to that exact candidate. The only current
claim is `local_release_gates_passed`; open human, physical/accessibility, and
remaining performance gates remain open.

## Endurance evidence

The clean code candidate `2bfd1aeda3336cb18c56fab5c7b46ed24e6497f7`
produced the committed evidence recorded by `a95396eae5888bdff7347ac79c8a1940f8b1e306`:

- 1,000 cycles and 4,000 loopback HTTP requests at 25,000 nodes, with 11.0 ms
  engine p95, 12,075,008 bytes post-warm RSS growth against a 52,428,800-byte
  ceiling, SQLite `quick_check=ok`, one Stockfish PID, zero restarts, and zero
  orphans;
- 20 cycles at the 250,000-node release budget, with 99.8 ms engine p95 and
  293.5 ms maximum;
- one legal 200-ply study with 2.2 ms active cancellation, 177.1 ms engine
  reuse, exact persisted-game restoration as `paused_interrupted`, successful
  resume, typed `503 STORAGE_UNAVAILABLE` under a real SQLite lock, successful
  retry, and two clean shutdowns;
- five isolated real Chromium loads at 1280x720 had median 28 ms FCP, 25.5 ms
  full load, and 22,537 transferred bytes; 1280x720 and 320x720 had zero
  horizontal overflow and zero console errors. Gzip reduced transfer 42.5%
  below the 0.2 baseline despite the added study workflow.

Gameplay preemption now stops the active UCI analysis through python-chess and
reuses the same engine process. The qualification rejects any future process
replacement, so the previous 999-restart defect cannot silently return.

Evidence: [runtime endurance](../artifacts/qualification/runtime-endurance-2026-09-04.json),
[release latency](../artifacts/qualification/runtime-release-2026-09-04.json),
[study recovery](../artifacts/qualification/study-recovery-2026-09-04.json), and
[browser performance](../artifacts/qualification/browser-performance-2026-09-04.json).

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

1. Complete cold-start, natural-exhibition, PGN stage-timing, reduced-motion,
   keyboard, CLS, and cross-device performance evidence.
2. Complete physical desktop and VoiceOver acceptance.
3. Complete blinded human usefulness review.
4. Update the ledger again only after new code or new external evidence. Adapter
   work remains a separate, fail-closed track.
