# Model Card: GemmaFischer Full Profile Candidate

Status: **implemented but not qualified for the current 0.2 commit**.

The optional full-profile candidate is
`mlx-community/gemma-4-e2b-it-4bit`, derived from
`google/gemma-4-E2B-it`, using MLX-LM 0.31.3. The revision is pinned and resolved
with Hugging Face local-files-only mode; the player never silently downloads a
model. Asset absence, corruption, or load failure degrades to deterministic
coaching.

The first target-host load on 2026-08-30 exposed a stale MLX-LM 0.28.4 pin, which rejected model type `gemma4`. MLX-LM 0.31.3 added the required architecture. The pinned model revision `238767527555cb75a05732a84dff5d6ba0dd6809` now loads on the M3 Pro/18 GB host.

The five-position run in
[full-2026-08-30.json](../artifacts/qualification/full-2026-08-30.json) belongs
to commit `7e3f0d2` and the old evidence contract. It showed that the pinned model
could load and emit schema-shaped claims on that host; it is retained as
historical feasibility evidence only. It does not qualify current 0.2
correctness, latency, cancellation, memory behavior, or teaching value.

Gemma 4 thinking is disabled for claim selection. The runtime extracts one
bounded JSON array and applies the strict discriminated claim schema. It receives
candidate IDs, scores, and deterministic concept labels. It may select and order
claims, but it may not invent moves, evaluations, facts, evidence IDs, or
user-visible factual prose. The deterministic coach owns the typed `LessonPlan`
and its text templates.

The revision, runtime version, cached size, tokenizer, chat template, and sampled
asset hashes are recorded in `assets/model-manifest.json`. That manifest records
the old smoke commit and is deliberately marked stale for current qualification.
A new full-profile run must bind every result to the current commit, lock,
runtime manifest, model revision, and engine/evidence hashes.

Promotion requires a licensed, lineage-isolated held-out suite; repeated cold and
warm measurements; offline restart; cancellation and memory-pressure tests; and
blind human comparison against deterministic coaching. The model remains
optional if it ties or loses, introduces a correctness regression, exceeds the
resource gates, or lacks reproducible assets. No fine-tuned adapter is supported
on `main` today.
