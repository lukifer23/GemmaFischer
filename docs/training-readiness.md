# Post-training Operator Contract

Post-training is active infrastructure and remains fail-closed. The supported
path is Apple-Silicon MLX-LM LoRA over the native, revision-pinned
`google/gemma-4-E2B-it` base. The quantized runtime snapshot is never a training
source. No adapter is currently qualified or shipped.

The model learns only `lesson-selection-2.0`: it selects supplied claim,
concept, question-template, and hint-template IDs. Stockfish remains the chess
authority; deterministic code still owns factual prose, mandatory claims,
questions, hints, answer keys, and grading.

## Ordered gate

1. Acquire the CC0 Lichess archive through `data/sources.json` and verify its
   pinned digest.
2. Build 12,000 train, 1,500 validation, and 1,500 untouched final-test rows.
   Rows are selected across the complete archive, split by source-game lineage,
   and expanded across rating buckets only after the source position is split.
3. Audit the exact prompt/target contract, legality, provenance, duplicates,
   semantic-position overlap, and lineage overlap.
4. Freeze 1,000 engine-grounded questions from final-test only and prove their
   exact UCI/SAN grader.
5. Export 2,500 training rows for two independent chess-literate reviewers.
   Every response includes the exact ID selection and complete usefulness
   rubric. Validate full two-reviewer coverage, then have one independent
   adjudicator resolve every disagreement.
6. Run at least 250 validation rows through the real pinned untuned inference
   model, then derive the frozen error taxonomy from that exact receipt. The
   committed 250-row baseline is contract-valid on 43.6% of rows and exactly
   matches none of the deterministic targets. Its observed taxonomy contains
   141 contract failures, 109 claim-selection mismatches, 91 hint-selection
   mismatches, 65 concept-selection mismatches, and 4 question-selection
   mismatches. This clears the repeated-error rationale; it does not qualify a
   model. Point the three `evidence` fields in `training/post-training.json` at
   passing JSON receipts.
7. Download the one native base revision into a local cache and verify every
   declared file hash. Convert the human-gold-applied canonical data to MLX chat
   JSONL. The hash-pinned `training/mlx-lora.yaml` fixes rank 16, dropout 0.05,
   and scale 2.0 (alpha 32 divided by rank 16); the command fixes AdamW, batch
   size 1, gradient accumulation 16, completion-only loss, and seed 3407.
8. Run the preflight. Only a passing preflight may authorize the explicit 7-20
   step smoke. Production SFT additionally requires the manifest's separate
   production authorization.
9. Package exactly one selected adapter plus its receipts as a GitHub Release
   asset. Weights, native model files, prepared data, and rolling checkpoints do
   not enter Git.

## Commands

```bash
uv run gemmafischer build-dataset --limit 15000 --nodes 250000
uv run gemmafischer audit-data --output artifacts/data-audit/latest.json
uv run gemmafischer freeze-question-eval --limit 1000
uv run gemmafischer evaluate-questions
uv run gemmafischer evaluate-training-baseline --limit 250
uv run gemmafischer freeze-error-taxonomy

uv run gemmafischer label-export \
  --dataset data/derived/v2/train.jsonl \
  --output artifacts/training/labels/packet.json \
  --limit 2500
uv run gemmafischer label-validate \
  --dataset data/derived/v2/train.jsonl \
  --responses /path/to/two-reviewer-responses.jsonl \
  --output artifacts/training/labels/validated.json
uv run gemmafischer label-adjudicate \
  --dataset data/derived/v2/train.jsonl \
  --validation artifacts/training/labels/validated.json \
  --adjudications /path/to/adjudications.jsonl \
  --output artifacts/training/human-gold.json
uv run gemmafischer label-apply \
  --human-gold artifacts/training/human-gold.json

uv run gemmafischer prepare-training-data \
  --human-gold artifacts/training/human-gold.json
uv run gemmafischer training-readiness
uv run gemmafischer training-preflight \
  --model /path/to/google-gemma-4-E2B-it-native
```

After the preflight reports `smoke_ready: true`, the bounded real run is:

```bash
uv run gemmafischer train-smoke \
  --preflight artifacts/training/preflight-latest.json \
  --model /path/to/google-gemma-4-E2B-it-native \
  --data artifacts/training/mlx-data \
  --adapter artifacts/training/adapters/smoke \
  --receipt artifacts/training/smoke-receipt.json \
  --iterations 7 --max-seq-length 1024
```

`train-sft` is a separate production-authorized command. Both commands refuse
non-Apple-Silicon hosts, nonempty adapter destinations, unsupported sequence
lengths, and any result other than exactly one adapter `.safetensors` file.
`package-adapter` likewise refuses zero or multiple adapters.

## Promotion boundary

The adapter must beat deterministic selection and untuned Gemma on frozen
schema, grounding, legal-move, top-1/top-3, question, human-usefulness, latency,
memory, restart, and endurance gates. A tie is a loss. Failure retains the
negative receipt and the deterministic product; it does not create another
checkpoint or fallback model path.
