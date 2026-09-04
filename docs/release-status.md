# Release Status

GemmaFischer 0.3 is committed on `main`. The machine ledger binds the verified
evidence candidate `1986a70f2d3662c2b3e4bc4b8d0dc54a3082aac8`; this documentation and
ledger update is its allowed ledger-only child.

## 0.3 worktree gates

| Gate | Status | Evidence boundary |
|---|---|---|
| Ruff and strict mypy | Passed locally | Current worktree |
| Model-free Python suite | Passed locally | Current worktree, including real Stockfish study test |
| One-PGN study pipeline | Passed locally | Real Stockfish, SQLite, and TestClient smoke |
| Generated OpenAPI | Passed locally | Current worktree matches generated contract |
| Package and isolated-wheel smoke | Passed locally | 0.3.0 sdist and wheel |
| 70% whole-package coverage | Passed locally | 70.20%, 160 local-alpha tests; 139 portable tests |
| Real Chromium Learn and Position Lab | Passed locally | Desktop and 390x844, zero console errors |
| Runtime endurance and recovery core | Passed locally | 1,000 real-engine cycles, 200-ply interruption/recovery, storage lock/retry, SQLite integrity, bounded memory, zero orphans |
| Hosted CI and CodeQL | Passed | Verification `33908093108` and CodeQL `33908093217` on the exact candidate SHA; zero open code, dependency, or secret alerts |
| Physical device and VoiceOver | Open | Human/device acceptance |
| Process-restart churn and performance completion | Open | 999 safe Stockfish restarts in the zero-think-time stress; cold starts, natural exhibition, stage timings, reduced-motion/keyboard/CLS, bundle growth, and cross-device latency remain |
| Blind human learning usefulness | Open | At least two chess-literate reviewers |
| Optional adapter | Blocked by design | Human labels and explicit authorization absent |

The machine-readable [`assets/release-status.json`](../assets/release-status.json)
binds command receipts, committed artifact hashes, hosted run IDs, security-alert
observations, and typed external gates to that exact candidate. The only current
claim is `local_release_gates_passed`; open human, physical/accessibility, and
remaining performance gates remain open.

## Endurance evidence

The clean code candidate `d67062a08af5c13ffa622fc93713e96225fd95d5`
produced the committed evidence recorded by `1986a70f2d3662c2b3e4bc4b8d0dc54a3082aac8`:

- 1,000 cycles and 4,000 loopback HTTP requests at 25,000 nodes, with 187.5 ms
  engine p95, 20,873,216 bytes post-warm RSS growth against a 52,428,800-byte
  ceiling, SQLite `quick_check=ok`, one Stockfish child maximum, and zero
  orphans;
- 20 cycles at the 250,000-node release budget, with 271.3 ms engine p95 and
  312.9 ms maximum;
- one legal 200-ply study with 2.2 ms active cancellation, 177.1 ms engine
  reuse, exact persisted-game restoration as `paused_interrupted`, successful
  resume, typed `503 STORAGE_UNAVAILABLE` under a real SQLite lock, successful
  retry, and two clean shutdowns;
- real Chrome Headless Shell at 1280x720 and 320x720 with zero horizontal
  overflow and zero console errors. Absolute performance budgets passed; the
  application shell's relative bundle growth remains open.

The endurance stress restarted Stockfish 999 times because each immediate next
gameplay command preempted the prior background review. Concurrency, cleanup,
memory, integrity, and latency remained within their gates, but the restart
churn remains open as an efficiency defect rather than being called complete.

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

1. Eliminate or justify background-review preemption churn; complete cold-start,
   natural-exhibition, PGN stage-timing, reduced-motion, keyboard, CLS, bundle,
   and cross-device performance evidence.
2. Complete physical desktop and VoiceOver acceptance.
3. Complete blinded human usefulness review.
4. Update the ledger again only after new code or new external evidence. Adapter
   work remains a separate, fail-closed track.
