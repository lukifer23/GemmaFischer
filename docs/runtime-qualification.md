# Runtime Qualification

`profile-runtime` measures the production FastAPI/Uvicorn stack through a real
IPv4 loopback TCP socket. It does not use `TestClient`, replace Stockfish, or
estimate process state.

```console
uv run gemmafischer profile-runtime --requests 20 --nodes 250000 \
  --output artifacts/qualification/runtime-2026-08-30.json
```

Each cycle performs a health request, creates a persisted exhibition session,
requests legal moves, and executes a server-owned engine move. The artifact
retains every HTTP latency and response size in addition to per-operation p50,
p95, mean, and maximum latency.

The runner starts an isolated Uvicorn process with a temporary SQLite history,
observes its real descendant process tree with `ps`, and records Stockfish PIDs
after every engine request. It then requests Uvicorn's graceful shutdown,
requires a zero exit status, and verifies that every observed Stockfish PID is
gone. Maximum concurrent children and process restarts are separate measures:
safe gameplay preemption can restart Stockfish without ever allowing two engine
children or leaking one after shutdown.

The 2026-08-30 target-host run completed 20 cycles and 80 HTTP requests at the
250,000-node budget. Engine-move latency was 284.4 ms mean, 275.0 ms p50,
310.1 ms p95, and 413.1 ms maximum. Health p95 was 3.2 ms, session creation p95
was 3.2 ms, and legal-move p95 was 2.1 ms. Exactly one Stockfish child was seen
after every engine request and no tracked child remained after shutdown.

The run observed 20 distinct Stockfish PIDs. In this stress shape, each new
gameplay request preempted the preceding queued review, and the provider restarted
the interrupted engine while preserving the one-process invariant. The artifact
exposes this churn rather than normalizing it: reducing or justifying its CPU,
memory, and startup cost is still an endurance optimization gate.

This artifact is local target-host evidence. It does not establish cross-device
latency, model TTFT/TPS, chess accuracy, tutoring correctness, or long-duration
memory stability.
