# Dependency-Ordered Execution Roadmap

Status: active  
Baseline: `f682ce1` on `main` (2026-08-30)

This is the controlling order for the next GemmaFischer pass. It supersedes
older backlog ordering, but not the repository's evidence, security,
provenance, or qualification contracts. A later phase starts only after the
previous phase's result is recorded as `passed`, `failed`, or `blocked`.

## Outcome

A user can play, receive the engine reply and cited review, practice the
reviewed position on the same board, request a hint, answer, receive
deterministically graded feedback, and return to the live game. The same flow
works for engine-v-engine study and survives restart.

```text
server-owned live session                  browser-owned display context
FEN + revision + plies                     live | historical review | tutor
          |                                           |
          +--> Stockfish analysis --> typed evidence -+
                                      |               |
                         deterministic lesson         |
                                      |               |
                            persisted tutor turn <----+
                                      |
                         deterministic grading
```

The displayed review FEN and the authoritative live-game FEN are separate.
Studying a position must never mutate the live game.

## Non-negotiable contracts

- Stockfish is the only chess authority. A model cannot create moves, scores,
  lines, concepts, correct answers, grading, or factual chess prose.
- Questions, options, hints, and feedback are typed and cite the same immutable,
  position-scoped evidence package.
- The deterministic profile remains complete when model assets are missing,
  corrupt, slow, cancelled, or disqualified.
- Persistence is atomic, restartable, bounded, and tested with real SQLite.
- Product code contains no fake providers, demo answers, placeholder metrics, or
  silent success. Unit test doubles do not replace real Stockfish, browser,
  persistence, and model qualification.
- A newer checkpoint or training tool is a candidate, not a promotion decision.
- Quantized inference weights are not fine-tuning source weights.

## What already exists

| Capability | Current implementation |
|---|---|
| Evidence and typed lessons | `domain.py`, `evidence.py`, `coach.py` |
| Persistent Stockfish and sessions | `engine.py`, `service.py` |
| SQLite WAL and recovery | `storage.py` |
| Local HTTP contracts | `web.py`, `docs/openapi.json` |
| Unified play/review/exhibition UI | `static/index.html`, `app.js`, `app.css` |
| Accuracy/tutor/model evaluation | `accuracy_eval.py`, `tutor_eval.py`, `model_profile.py` |
| Governed data acquisition | `data/sources.json`, build/audit commands |

The present evidence is useful but incomplete: 73 tests exist; the constructed
suite and a 100-position held-out sample passed; Gemma produced visible output
in 21/21 profiled requests; and ten automated tutoring cases passed factual
gates. Human usefulness remains open, there is no question/answer contract, the
current corpus is 94 training and 6 evaluation records, and long-run/browser
qualification is not complete.

## Phase 0 - Lock truth, security, and reproducibility

Do first:

1. Remove local filesystem paths from health and error responses.
2. Enforce and test a request-body limit before JSON parsing.
3. Repair documentation/command drift and verify checked-in OpenAPI.
4. Run the clean fast gate and record the exact commit, worktree state,
   hardware, Stockfish version/hash, commands, and artifacts.
5. Verify clean launch/status/stop, one managed instance, and zero lingering
   model/server/Stockfish processes when stopped.

Exit: Ruff, strict mypy, non-model tests, repository audit, OpenAPI drift,
oversized-body, path-disclosure, and lifecycle checks pass.

## Phase 1 - Fix execution ownership and persistence integrity

This precedes tutoring and visual polish because current engine work can block
the FastAPI event loop and gameplay/review share cancellation-sensitive engine
ownership.

Work:

- Move blocking session commands off the event loop.
- Implement one explicit engine schedule: either a measured separate gameplay
  provider or one owner-thread priority queue that guarantees gameplay
  precedence. Cancellation must never close unrelated gameplay work.
- Define queued/running engine, model, shutdown, and retry semantics with stable
  errors and stage timings.
- Make exhibition pause/resume server-owned and persistent.
- Prevent stale browser requests from committing after a newer session, legal
  move, command, or review request.
- Preserve review linkage when analysis retention prunes old rows: a persisted
  ply must retain a resolvable immutable review summary or owned snapshot.
- Preserve exact promotion UCI and support Q/R/B/N instead of silently forcing a
  queen.

Exit:

- Health/polling remain responsive during a 250,000-node compare.
- Gameplay meets its latency budget under review load; cancel cannot terminate
  gameplay; latest browser intent wins.
- Reload restores mode, paused state, FEN, revision, plies, review links, and
  preferences exactly.
- All four legal promotion choices work.

## Phase 2 - Establish durable real-browser acceptance

Add Playwright to the checked-in development/test toolchain and launch the real
FastAPI app with real Stockfish and a temporary SQLite database. Functional E2E
may use a low node budget; performance qualification uses the release budget.
Do not intercept chess endpoints.

Required fixtures cover start, castling, en passant, white/black promotion,
checkmate, stalemate, invalid FEN, long history, and a frozen 200-ply legal
session. Required flows cover selection/highlights, legal/illegal moves, engine
reply, automatic review, citations, undo, explain/compare, cancellation,
exhibition pause/reload/resume, rapid mode/FEN changes, engine/model failure,
terminal positions, and restart recovery.

Exit: the real browser harness produces traces/screenshots on failure and proves
the core session before tutoring changes begin.

## Phase 3 - Ship the deterministic tutor vertical slice

Do not ask Gemma to author questions or grade answers.

### Contract and state

Add `tutor.py` as pure logic and closed domain types for:

- `TutorQuestion`: interaction/source IDs, anchored FEN and position ID, prompt
  template, hint evidence IDs;
- `TutorFeedback`: submitted and engine moves, `matched_engine`, `equivalent`, or
  `engine_preferred`, comparison evidence IDs;
- `TutorFollowUp`: closed concept/evaluation option IDs;
- internal `TutorInteractionRecord` with hidden answer keys and source evidence;
- redacted public `TutorInteractionView`;
- `TutorCommandRequest` with an independent expected tutor revision.

```text
create -> awaiting_answer -> awaiting_follow_up -> complete
              |                    |
              +------ dismiss -----+-----------> dismissed

hint: awaiting_answer -> awaiting_answer (revision + 1)
illegal/failed grading: typed error, state unchanged and retryable
terminal-state mutation: 409
```

Use an independent tutor revision; revealing a hint must not conflict with live
game revision. Store tutor interactions in a dedicated SQLite table keyed by
session/source analysis, with immutable copied evidence, transactional session
deletion, and bounded per-session retention.

The first question is “find a move” on the frozen review FEN. The user answers
by moving on the same board. Grade the arbitrary legal move with a real,
equal-budget Stockfish comparison, then generate cited deterministic feedback
and a closed concept/evaluation follow-up. No guessed grading and no extra model
call.

API resources live under the session and support create/list/get, tutor-FEN
legal moves, and revisioned commands. Foreign, missing, unfinished, terminal,
stale, illegal, and engine-failure cases receive stable typed errors. Public
responses never expose hidden answers before submission.

The browser adds:

```text
boardContext = live | { tutor: interaction_id, frozen_fen }
```

“Practice this position” opens inline from a completed current or historical
review. The existing board shows a clear review banner, stops exhibition
autoplay, disables live-game commands, and offers “Return to game.” Returning
restores the newest server FEN immediately. Play, review, question, hint,
feedback, and history remain one workspace with no tabs.

Implementation slices:

1. domain + pure question/feedback logic + property tests;
2. SQLite + service locking/revisions + restart/concurrency tests;
3. typed HTTP + OpenAPI + real Stockfish integration;
4. board context + tutor UI + real-browser acceptance;
5. automated and human qualification fields.

Exit:

- 100% source-position match, legal answers, hidden-answer redaction, grading
  agreement, hint evidence validity, follow-up validity, and repeatability.
- Tutor work never changes live FEN/revision and survives restart.
- Player and exhibition reviews both enter the same tutor flow.

## Phase 4 - Correct and scale the data/evaluation system

Do not scale the current builder unchanged. Its FEN-to-UCI prompt and
deterministic `LessonPlan` target do not match the runtime model contract, which
is an exact claim-selection prompt to bounded claim/concept JSON.

Work in order:

1. Freeze separate schemas for chess-authority evaluation, deterministic tutor
   evaluation, model-selection training, question evaluation, and human review.
   A legitimate selector-training row must contain the exact runtime system
   prompt, `claim_selection_prompt` user payload, a target accepted by
   `parse_claim_selection`, evidence/configuration hashes, source/transformation
   provenance, and a model-contract version. A FEN-to-UCI row is not selector
   training data.
2. Make the audit require source/game/puzzle lineage, full provenance, evidence
   schema/configuration, exact target contract, rejection reasons, and category/
   rating distributions.
3. Detect lineage leakage, normalized/semantic position duplicates,
   transpositions, conflicting labels, and evaluation overlap—not only exact
   JSON/FEN duplicates.
4. Remove or quarantine stale unpinned data scripts and obsolete
   `READY_FOR_TRAINING` metadata that do not satisfy the current contract.
5. Profile a 1,000-row build for time, rejection rate, output size, and RSS;
   make it resumable/content-addressed if a full run is not safely bounded.
6. Split by source-game lineage and semantic position, ignoring FEN clocks for
   duplicate detection. Freeze three partitions before target construction: at
   least 10,000 training, 1,000 validation, and 1,000 untouched final-test rows.
   Build by quota across the complete archive rather than accepting the first
   qualifying rows.
7. Expand held-out chess agreement to at least 1,000 positions and report legal,
   top-1/top-3, terminal, category, stability, rejected-row, and uncertainty
   measures.
8. Freeze a question evaluation set that training never consumes. Measure
   availability, legality, unique answers, evidence integrity, rating fit, and
   error taxonomy.
9. Extend human packets and validated ingestion/adjudication to include every
   rubric field, including harmful omission.

Exit: 10,000/1,000/1,000 task-aligned, leakage-free train/validation/final-test
records and the frozen chess, question, and human-review artifacts pass every
zero-tolerance gate. A training command still does not exist.

## Phase 5 - Finish interaction quality and accessibility

- Replace the body-level piece ghost with a board-local, cancellable FLIP-style
  overlay; lock the board rectangle and commit final DOM state once per move.
- Separate session, displayed board, selection, request, and animation state.
- Preserve selection and highlights across safe renders and prevent stale
  review/engine responses from replacing the user-selected review.
- Provide selected/legal/capture/last-move/check/review/turn states with non-color
  cues, correct grid semantics, focus restoration, and reduced motion.
- Verify click-click, drag, keyboard, touch, promotions, castling, en passant,
  undo, rapid input, tutor/live switching, and exhibition.

Viewport matrix: 320x568 at 200% reflow, 375x812, 768x1024, 900x600,
1280x720, and 1440x900. No horizontal overflow; board position/size changes by
at most 1px during core interactions; full-flow CLS <= 0.02; no serious/critical
automated accessibility findings; manual VoiceOver covers a complete lesson.

## Phase 6 - Run real performance, endurance, and recovery gates

Required recorded runs:

| Run | Minimum |
|---|---:|
| Clean cold starts | 5 deterministic and 5 full-profile |
| Player loop | 100 sequential turns |
| Position analysis | 1,000 stratified requests |
| Compare analysis | 1,000 equal-budget requests |
| Exhibition | 100 plies or documented natural completion plus continuation |
| Persistence | 200 plies plus tutor turns, restart/read/append/undo/restart |
| Full lessons | 20 warm visible outputs |
| Cancellation | queued, engine, comparison, model, shutdown |

Record validation, queue, engine, comparison, evidence, deterministic lesson,
model selection, persistence, HTTP, and browser-render stages separately.

Hard gates include the published p95 latency budgets, question creation/hint
mutation <= 25ms p95, answer-to-feedback <= 1.8s p95, selection paint <= 50ms,
legal highlights <= 100ms, INP <= 200ms, CLS <= 0.02, persistence write <= 25ms,
post-warm growth <= 50MiB, SQLite integrity `ok`, exact restart equality, zero
lost/duplicate plies, zero orphan processes, and resource release <= 2s.

## Phase 7 - Decide whether Gemma earns a production role

Blindly compare deterministic ordering with Gemma-selected ordering from the
identical frozen eligible set. At least two reviewers score correctness,
clarity, relevance, rating fit, actionability, harmful omission, question/hint/
feedback usefulness, and whether the model adds value instead of rearranging the
baseline.

- If Gemma does not improve usefulness without correctness/latency regression,
  deterministic remains default and model work stops.
- If it wins, isolate model inference from engine/session execution, enforce hard
  cancellation/restart/memory ceilings, and rerun Phase 6 full-profile gates.
- Try a larger text-only model only for a documented E2B capability shortfall.
  Vision is out of scope because FEN and typed evidence already represent every
  required board fact.

LFM2.5-2.6B remains a recorded failed candidate for the current frozen harness;
that is not a claim about the whole model family and is not repaired by grading
it differently.

## Phase 8 - Decide on post-training and Unsloth

Fine-tuning is considered only after Phases 1-7 pass, the task-aligned corpus is
audited, repeated model errors form a stable taxonomy, and harness/schema errors
have been ruled out on frozen evaluation.

Unsloth is a real candidate toolchain, not a commitment. As of this roadmap,
its upstream project has an Apple Silicon MLX training path with real LoRA smoke
tests and MLX export, so it may fit this M3 Pro/18GB machine. It is also rapidly
changing and has recent model-specific MLX issues. Before a real run:

1. pin exact Unsloth, unsloth-zoo, MLX, MLX-LM, model revision, and native base
   weight hashes in an isolated environment;
2. run a tiny seven-to-twenty-step LoRA smoke test and prove resume, merge/export,
   reload in the production inference harness, memory ceiling, and deterministic
   seed/config capture;
3. compare Unsloth MLX against a minimal MLX-LM LoRA baseline on the same rows;
4. use native license-compatible source weights, not the 4-bit inference artifact;
5. proceed to SFT/adapter experiments only if the smoke artifact is complete.

Train model selection and pedagogical ordering, never chess facts or free-form
grading. Track hashes, seeds, train/eval loss, exact-schema rate, grounding,
question selection, human usefulness, latency, and memory. Ship an adapter only
if it beats deterministic and untuned Gemma on frozen automated and blind human
gates; otherwise preserve the negative result and keep the simpler profile.

## Phase 9 - Release proof and documentation closure

From one clean release commit, rerun fast, browser, endurance, persistence,
accuracy, data, tutoring, human, and selected-model gates. Update README,
architecture, evidence/API, security, provenance, model card, performance,
qualification, OpenAPI, and changelog from shipped behavior. Every claim links
to a raw artifact; local evidence is not universal device acceptance.

## Failure-mode acceptance

| Failure | Required behavior |
|---|---|
| Stockfish missing/crashes | explicit unavailable/retryable state; no lesson fabrication |
| Model missing/corrupt/hangs | deterministic path; bounded cancellation and resource release |
| Invalid model selection | reject and use deterministic eligible ordering |
| Stale session/tutor action | `409`; no duplicate move or grading against another FEN |
| No valid question | typed unavailable state; normal lesson remains |
| Rapid/double browser input | latest intent, one accepted mutation, restored focus |
| Interrupted animation | cancel overlay; render authority once |
| Interrupted DB write | previous committed state recovers with integrity `ok` |
| Oversized/malformed input | bounded typed error before expensive work |
| Changed data source/config on resume | refuse a mixed build |

## Test map

```text
pure domain/property tests
  -> SQLite/service concurrency and restart integration
    -> real Stockfish and HTTP integration
      -> real browser session/tutor/accessibility flows
        -> 1,000+ accuracy, data, performance and endurance qualification
          -> blind human usefulness decision
```

## Parallel execution

After contracts freeze, independent lanes may handle pure tutor logic, data
builder/audit work, browser baseline harness, and documentation/artifact schemas.
Storage/API follows the domain contract; UI follows the API; model comparison
follows the deterministic tutor and frozen eval; training follows the model
decision. Agents do not edit shared contract files concurrently.

## Not in scope

- Vision/OCR/camera board input; multiplayer, cloud hosting/sync, accounts, or
  internet-required play; a second chess authority; free-form LLM chess analysis
  or grading; dynamic model roulette/downloads; a frontend-framework migration;
  distributed queues or external databases without profiling evidence.

These are exclusions, not placeholders. There are no separate speculative TODO
items: required work is ordered above and deferred work is explicit.

## Complete means complete

One clean release commit proves the full tutor vertical slice, browser matrix,
scaled data/accuracy, endurance/resource budgets, human usefulness, model and
training decisions, and documentation closure. A green unit suite, attractive
screenshot, fast single request, or successful generation proves only its own
layer.
