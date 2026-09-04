# Model Card: GemmaFischer Model Candidates

Status: **The deterministic coach is the public-alpha product. Gemma 4 E2B remains an optional development candidate; LFM2.5-2.6B is rejected out of the box for the bounded selector workload**.

The optional full-profile candidate is
`mlx-community/gemma-4-e2b-it-4bit`, derived from
`google/gemma-4-E2B-it`, using MLX-LM 0.31.3. The revision is pinned and resolved
with Hugging Face local-files-only mode; the player never silently downloads a
model. Asset absence, corruption, or load failure degrades to deterministic
coaching. An optional adapter is loaded only from
`GEMMAFISCHER_ADAPTER_PATH`; the directory must contain exactly one safetensors
file and `adapter_config.json`. `GEMMAFISCHER_ADAPTER_SHA256` can pin the weight
bytes.

## Current target-host evidence

The current Gemma E2B run used 21 real Stockfish-derived prompts at the same
768-token ceiling as production. Every captured response passed the production
claim parser and evidence validator. Warm p95 visible TTFT was 0.654 seconds,
warm p95 total generation was 8.830 seconds, minimum warm generation throughput
was 27.66 tokens/second, and MLX peak allocation was 3.47 GB. The five-case
tutoring suite ran twice and passed all 10 automated grounding executions after
required deterministic claims were preserved. This does not close the blinded
human-usefulness gate.

The exact local LM Studio challenger was
`lmstudio-community/LFM2.5-2.6B-MLX-4bit`, alias
`lfm2.5-2.6b-mlx`. Its `model.safetensors` is 1,517,616,892 bytes with SHA-256
`8a31b38d9d7c008bad771cb32331d47e7ff09b184adf3f2443a08b691c738574`.
It is text-only and substantially smaller than the 3,583,086,498-byte Gemma
snapshot.

That LFM checkpoint is a pure reasoning model. Across five exact application
prompts at a 768-token ceiling, it generated 767 hidden reasoning tokens every
time, returned no visible answer, and finished by token limit. Warm hidden-token
TTFT p95 was 1.079 seconds and generation throughput was 69.35 to 72.61
tokens/second, but visible-output rate was 0% and warm total p95 was 12.109
seconds. In the full tutoring suite, all eight nonterminal executions failed to
produce a parseable JSON array; the two terminal positions correctly bypassed
the model. A single 2,048-token probe eventually produced parseable JSON after
26.04 seconds and 1,412 hidden reasoning tokens. It therefore fails the current
latency and output-completeness contracts despite good raw decode speed.

Evidence:

- [Gemma model profile](../artifacts/qualification/model-profile-phase2-current.json)
- [Gemma tutoring qualification](../artifacts/qualification/tutoring-full-phase2-current.json)
- [LFM model profile](../artifacts/qualification/model-profile-lfm2.5-2.6b-local.json)
- [LFM tutoring qualification](../artifacts/qualification/tutoring-lfm2.5-2.6b-local.json)
- [Candidate decision](../artifacts/qualification/model-bakeoff-0.2-local.json)

A 2026-09-04 qualification rerun against the installed pinned Gemma snapshot
returned only one `claim_id` for the real test position. The contract requires
two to five, so the strict parser rejected it. The application correctly degrades
to deterministic coaching, but the optional model test is currently failing and
Gemma is not a 0.3 release candidate. The invalid response is not padded or
reported as a model success.

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
asset hashes are recorded in `assets/model-manifest.json`. The current artifacts
bind the model and engine revisions to the development commit, but were generated
from a dirty integration worktree. Release evidence must be regenerated from the
final clean commit and bind the lock and every configuration hash.

Promotion requires a licensed, lineage-isolated held-out suite; repeated cold and
warm measurements; offline restart; cancellation and memory-pressure tests; and
blind human comparison against deterministic coaching. The model remains
optional if it ties or loses, introduces a correctness regression, exceeds the
resource gates, or lacks reproducible assets. No fine-tuned adapter is supported
on `main` today.

## Fine-tuning decision

Do not fine-tune the installed 4-bit inference artifact. The active fail-closed
path selects the native `google/gemma-4-E2B-it` revision and pins every required
file hash. MLX-LM is the Mac-only LoRA baseline; Unsloth is not another supported
path unless a later authorized comparison earns it. Stockfish defines the valid
choice catalog, but deterministic imitation is not accepted as a claim of
improved teaching. Production selection targets require two independent human
reviewers and complete adjudication. Final-test rows stay outside the trainer
directory, and training is fixed at 4096 tokens to remove the known 1024-token
truncation path. Smoke and production authorization are currently revoked.
Exactly one adapter may be selected for qualification; no adapter is supported
today.
