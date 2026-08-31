# Model Card: GemmaFischer Model Candidates

Status: **Gemma 4 E2B is the current development candidate; LFM2.5-2.6B is rejected out of the box for the bounded selector workload**.

The optional full-profile candidate is
`mlx-community/gemma-4-e2b-it-4bit`, derived from
`google/gemma-4-E2B-it`, using MLX-LM 0.31.3. The revision is pinned and resolved
with Hugging Face local-files-only mode; the player never silently downloads a
model. Asset absence, corruption, or load failure degrades to deterministic
coaching.

## Current target-host evidence

The current Gemma E2B run used 21 real Stockfish-derived prompts. Warm p95 first
token was 0.505 seconds, warm p95 total generation was 3.995 seconds, warm
generation throughput was at least 70.81 tokens/second, and MLX peak allocation
was 3.47 GB. The five-case tutoring suite ran twice and passed all 10 automated
grounding executions after required deterministic claims were preserved. This
does not close the blinded human-usefulness gate.

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

- [Gemma model profile](../artifacts/qualification/model-profile-0.2-local.json)
- [Gemma tutoring qualification](../artifacts/qualification/tutoring-full-0.2-local.json)
- [LFM model profile](../artifacts/qualification/model-profile-lfm2.5-2.6b-local.json)
- [LFM tutoring qualification](../artifacts/qualification/tutoring-lfm2.5-2.6b-local.json)
- [Candidate decision](../artifacts/qualification/model-bakeoff-0.2-local.json)

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

## Fine-tuning decision

Do not fine-tune the installed 4-bit inference artifact. If later evidence shows
a stable, repeated selector error worth training, use Liquid AI's native Base or
native post-trained checkpoint for LoRA/SFT and export a new pinned MLX quant
afterward. Its smaller text-only footprint may reduce training and deployment
cost, but the current data gate and the always-thinking behavior remain more
important than parameter count. Training stays blocked until the licensed,
lineage-isolated corpus meets the documented minimums and leakage gates.
