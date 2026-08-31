# Phase 1 Execution Contract

Status: implemented; repository and browser verification recorded on completion.

This phase fixes execution ownership, persistent review linkage, stale browser
intent, exhibition state, and the complete promotion contract before interactive
tutoring is added.

## Build order

```text
1. one-process Stockfish ownership
   -> 2. targeted cancellation + gameplay priority
      -> 3. ASGI thread-pool boundary
         -> 4. durable review queue + SQLite references
            -> 5. server-owned exhibition pause/resume
               -> 6. latest-intent browser guards
                  -> 7. exact Q/R/B/N promotion UI
                     -> 8. real Stockfish/browser regression pass
```

## Engine ownership

`StockfishProvider` is a condition-based arbiter around one lazy Stockfish
process. Every operation has an opaque token and kind.

```text
idle -> analysis(A)
          | gameplay waits: preempt A, close A's command, retry A later
          | cancel(A): close A's command, persist cancelled
          | cancel(any other token): no effect
          v
        idle -> gameplay(G) -> idle
                  cancel(any analysis): no effect
```

Gameplay waiters are FIFO and have admission priority. Only the exact active
analysis token may close an analysis command. Shutdown is the only untargeted
provider close. A preempted durable ply review returns to the front of the queue;
a newer interactive request may supersede only another pending interactive
request.

The blocking session-command route is synchronous so FastAPI executes it in its
worker pool. Health, capability, polling, and cancellation handlers remain on
the event loop and responsive.

## Persistence

SQLite schema version 2 adds `session_analysis_refs(session_id, ply,
analysis_id)` and `analysis_reservations(analysis_id)`. A durable review is
reserved atomically with its first queued snapshot; `save_session` synchronizes
the permanent reference and releases that reservation in the same transaction.
Analysis retention prunes only unreferenced, unreserved rows; session deletion
releases its references through foreign-key cascade and makes those rows eligible
for normal pruning.
Existing session rows are backfilled when the store opens if their referenced
analysis still exists. Already-pruned historical evidence cannot be recreated.

Exhibition `pause` and `resume` are revisioned server commands. Browser pause
stops the local loop, waits for any uncertain in-flight engine mutation to
settle, sends `pause` with the canonical revision, and does not cancel the ply
review. Reload displays persisted `paused`, `active/ready`, or `complete` truth.

## Browser intent and promotion

Session replacement, legal-move lookup, session command, explanation, and review
polling responses are scoped to captured session IDs and monotonic intent epochs.
Stale reads are aborted; stale mutations are ignored and reconciled rather than
assumed not to have executed. Superseded session creations are deleted instead
of polluting bounded history.

The browser retains exact `moves_uci` alongside deduplicated target squares. A
single exact move is submitted directly. Four promotion candidates open a native
modal dialog in Q/R/B/N order; Escape/cancel retains the selected pawn and legal
highlight, while selection submits the exact five-character UCI. The server no
longer defaults a four-character promotion to queen and rejects it explicitly.

## Acceptance

- Real Stockfish gameplay preempts a long analysis, the analysis reports a typed
  preemption, and the provider recovers with one process.
- Exact active and gameplay-blocked analysis cancellation completes within the
  bounded tests and cannot target gameplay.
- Health remains responsive during a real engine session command.
- Durable reviews are not superseded by interactive requests.
- Reserved and session-linked analyses survive retention and restart;
  unreferenced analyses still prune; deleting the session releases them.
- Paused exhibition state survives restart and blocks moves until resumed.
- Legal move output exposes four promotion UCIs and one target; white, black,
  capture, and all Q/R/B/N choices preserve exact pieces and SAN/FEN.
- A real browser demonstrates selection/highlighting, an accessible promotion
  chooser, knight underpromotion, reviewed exhibition pause, reload restoration,
  and a clean console.

Release-scale subprocess, viewport, accessibility, layout-shift, endurance, and
resource measurements remain the explicitly ordered Phase 2 and Phase 6 gates
in the [execution roadmap](execution-roadmap.md); they are not claimed by this
functional phase.
