# Evidence and API Contract

`EngineEvidence` schema `2.0` preserves normalized six-field FEN, a
position-scoped ID, side to move, actual engine identity/hash/applied
options/node budget/start time, terminal reason, deterministic board and concept
facts, and an optional `CandidateSet` and `MoveComparisonEvidence`.

Scores, mate values, and WDL use side-to-move perspective. Positive mate means the side to move can force mate. WDL totals 1000. Terminal positions contain zero candidates; non-terminal positions contain one to three.

Candidate, set, comparison, fact, concept, and position IDs are SHA-256 hashes
over RFC 8785 canonical JSON. IDs include the position and relevant engine
configuration so evidence from another position or budget cannot be substituted.
Principal variations retain at most 16 plies and the player displays at most
eight.

Candidate rank comes from one ordered MultiPV search. Compare mode additionally
runs the engine choice and considered move under the same node budget. The
comparison records `equal`, `engine_better`, or `considered_better`; centipawn
scores within 15 cp are equal and mate scores are compared explicitly. These
constrained searches do not reorder the CandidateSet.

The coach returns a typed `CoachingResult` with zero to five claims, a closed
question-template ID, a closed hint-template ID, and an optional typed
`LessonPlan`. Lesson steps cite deterministic concept evidence and use closed
templates. Under `lesson-selection-2.0`, the optional model may return only
supplied claim, concept, question-template, and hint-template IDs. The parser
rejects partial objects, unknown IDs, duplicates, and out-of-range selections;
mandatory deterministic claims are restored by the merge. User-visible factual
prose is rendered from validated payloads, and terminal positions need no filler
claims.

HTTP routes are:

- `POST /api/v1/analyses` and `GET /api/v1/analyses` for analysis creation and local history;
- `GET` and `DELETE /api/v1/analyses/{analysis_id}` for polling and cancellation;
- `POST` and `GET /api/v1/sessions` to create and list persistent games;
- `GET` and `DELETE /api/v1/sessions/{session_id}` to resume or delete one;
- `GET /api/v1/sessions/{session_id}/legal-moves` for authoritative selection;
- `POST /api/v1/sessions/{session_id}/commands` for player move, engine move,
  undo, pause, and resume with `expected_revision`;
- `POST` and `GET /api/v1/sessions/{session_id}/tutor` to create and list
  evidence-anchored practice interactions;
- `GET /api/v1/sessions/{session_id}/tutor/{interaction_id}` and its
  `/legal-moves` and `/commands` resources for frozen-position play, hint,
  answer, follow-up, and dismiss mutations with an independent revision;
- `GET /api/v1/health` and `/api/v1/capabilities` for local runtime status.
- `POST /api/v1/storage/retry` for a token-protected integrity/write probe that
  resumes work paused by a transient storage failure.

Tutor responses expose the frozen question, cited hint, submitted answer,
comparison-backed feedback, and closed follow-up options. They never expose the
hidden best move or correct follow-up key before submission. Tutor state copies
its immutable source evidence and cannot alter the live session.

The health contract reports capability/status only, including storage and worker
state, and never returns resolved filesystem paths. Durable create routes accept
`Idempotency-Key`; the receipt and resource commit together, while reuse with a
different payload returns `409 IDEMPOTENCY_CONFLICT`. Public failures carry a
generated request ID in the header and typed body plus safe retry/remediation
fields. Request bodies are limited to 64 KiB at the HTTP boundary; oversized
bodies receive `413 REQUEST_TOO_LARGE` before schema parsing.

The old stateless `POST /api/v1/board/*` routes do not exist. Clients use
server-owned sessions.

The generated contract is committed as [openapi.json](openapi.json); CI fails on drift. Mutations require the per-launch capability token.
