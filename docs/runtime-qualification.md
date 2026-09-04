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
p95, mean, and maximum latency. It also samples full process-tree RSS after
every cycle, records SQLite/WAL/SHM footprint, and requires `PRAGMA quick_check`
to return `ok`.

The runner starts an isolated Uvicorn process with a temporary SQLite history,
observes its real descendant process tree with `ps`, and records Stockfish PIDs
after every engine request. Any child-process replacement now fails an explicit
zero-restart gate. It then requests Uvicorn's graceful shutdown,
requires a zero exit status, and verifies that every observed Stockfish PID is
gone. Maximum concurrent children and process restarts are separate measures.
Gameplay preemption stops the active UCI analysis and reuses the same engine
child; a process replacement is reserved for actual engine failure.

`profile-study-recovery` exercises the durable PGN path separately:

```console
uv run gemmafischer profile-study-recovery --nodes 25000 --timeout 120 \
  --output artifacts/qualification/study-recovery-2026-09-04.json
```

It uses one legal 200-ply game to cancel active work, prove immediate engine
reuse, stop Uvicorn during an in-flight study, restore the exact persisted game
as `paused_interrupted`, resume it to `ready`, force a real SQLite write lock,
recover storage, and check child-process cleanup after both shutdowns.

## 2026-09-04 clean-candidate results

Candidate `2bfd1aeda3336cb18c56fab5c7b46ed24e6497f7` passed the 1,000-cycle
endurance shape at 25,000 nodes: 4,000 HTTP requests, 11.0 ms engine-move p95,
12,075,008 bytes post-warm process-tree RSS growth, SQLite `quick_check=ok`, one
Stockfish child for the entire run, zero restarts, zero orphans, and a clean
Uvicorn shutdown. The 20-cycle 250,000-node release-budget sample passed at
99.8 ms engine-move p95 and 293.5 ms maximum.

The previous candidate restarted Stockfish 999 times because preemption closed
the engine transport. The provider now uses python-chess's supported analysis
stop operation, requeues the durable review, and preserves the same engine PID.
The qualification fails if even one replacement is observed.

The 200-ply recovery run passed every functional gate. Active cancellation took
2.2 ms, engine reuse after cancellation took 177.1 ms, interrupted state and the
exact game were restored, resume reached `ready`, a real SQLite lock returned a
typed `503 STORAGE_UNAVAILABLE`, retry restored `ready`, and both shutdowns left
zero Stockfish orphans.

See the [endurance](../artifacts/qualification/runtime-endurance-2026-09-04.json),
[release-budget](../artifacts/qualification/runtime-release-2026-09-04.json), and
[study recovery](../artifacts/qualification/study-recovery-2026-09-04.json)
artifacts for raw timings, hashes, process samples, and gate results.

## 2026-08-30 baseline

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
