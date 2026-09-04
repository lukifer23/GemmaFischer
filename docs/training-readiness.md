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
5. Run at least 250 validation rows through the real pinned untuned inference
   model, then derive the frozen error taxonomy from that exact receipt. The
   committed 250-row baseline is contract-valid on 43.6% of rows and exactly
   matches none of the deterministic targets. Its observed taxonomy contains
   141 contract failures, 109 claim-selection mismatches, 91 hint-selection
   mismatches, 65 concept-selection mismatches, and 4 question-selection
   mismatches. This clears the repeated-error rationale; it does not qualify a
   model. Point the baseline and taxonomy evidence fields in
   `training/post-training.json` at passing JSON receipts.
6. Collect two complete independent reviews of at least 2,500 train records and
   adjudicate every material disagreement. Apply that human gold to a new derived
   train corpus. Stockfish still constrains the valid ID catalog; humans choose
   the teaching target. Deterministic imitation alone is not a production objective.
7. Prepare MLX `train.jsonl` and `valid.jsonl` directly from that audited,
   adjudicated corpus. The receipt binds both trainer files and the untouched
   final-test source hash, but never copies final-test rows into the trainer directory.
8. Download the one native base revision into the Hugging Face cache and verify
   all seven declared file hashes, including the official chat template. The
   hash-pinned `training/mlx-lora.yaml` fixes rank 16, dropout 0.05,
   and scale 2.0 (alpha 32 divided by rank 16); the command fixes AdamW, batch
   size 1, gradient accumulation 16, completion-only loss, and seed 3407.
9. Run the preflight. Only a passing preflight may authorize the explicit 7-20
   step smoke. Production SFT additionally requires the manifest's separate
   production authorization.
10. Package exactly one selected adapter plus its receipts as a GitHub Release
   asset. Weights, native model files, prepared data, and rolling checkpoints do
   not enter Git.

Independent human review is both the production selection-target authority and
a separate product-acceptance gate. Training labels do not prove that the final
experience teaches better; blinded post-training comparison still must.

## Commands

```bash
uv run gemmafischer build-dataset --limit 15000 --nodes 250000
uv run gemmafischer audit-data --output artifacts/data-audit/latest.json
uv run gemmafischer freeze-question-eval --limit 1000
uv run gemmafischer evaluate-questions
uv run gemmafischer evaluate-training-baseline --limit 250
uv run gemmafischer freeze-error-taxonomy

uv run gemmafischer label-export --dataset data/derived/v2/train.jsonl \
  --output artifacts/training/label-packet.jsonl
uv run gemmafischer label-validate --dataset data/derived/v2/train.jsonl \
  --responses /path/to/reviewer-responses.jsonl \
  --output artifacts/training/label-validation.json
uv run gemmafischer label-adjudicate --dataset data/derived/v2/train.jsonl \
  --validation artifacts/training/label-validation.json \
  --adjudications /path/to/adjudications.jsonl \
  --output artifacts/training/human-gold.json
uv run gemmafischer label-apply \
  --human-gold artifacts/training/human-gold.json
uv run gemmafischer prepare-training-data \
  --source-dir data/derived/v2-reviewed \
  --human-gold artifacts/training/human-gold.json
uv run gemmafischer acquire-training-model
uv run gemmafischer training-readiness
uv run gemmafischer training-preflight \
  --model /path/to/google-gemma-4-E2B-it-native
```

After the manifest contains explicit smoke authorization and the preflight
reports `smoke_ready: true`, the bounded real run is:

```bash
uv run gemmafischer train-smoke \
  --preflight artifacts/training/preflight-latest.json \
  --model /path/to/google-gemma-4-E2B-it-native \
  --data artifacts/training/mlx-data \
  --adapter artifacts/training/adapters/smoke \
  --receipt artifacts/training/smoke-receipt.json \
  --iterations 7 --max-seq-length 4096
```

`train-sft` is a separate production-authorized command. Both commands refuse
non-Apple-Silicon hosts, any sequence length other than 4096, and any result
other than exactly one adapter `.safetensors` file. A fresh run requires an empty
destination. `--resume` requires exactly one existing checkpoint and passes that
exact file to MLX-LM. `package-adapter` likewise refuses zero or multiple adapters.

Current state: the audited machine corpus, untuned baseline, error taxonomy, and
native model hashes exist. The previous machine-target preparation is not valid
for the corrected production objective. Adjudicated human targets, a fresh
prepared-data receipt, and a new preflight are required. Both smoke and production
authorization are revoked. No adapter weights or training process exist.

## Promotion boundary

The adapter must beat deterministic selection and untuned Gemma on frozen
schema, grounding, legal-move, top-1/top-3, question, latency, memory, restart,
and endurance gates. A tie is a loss. Failure retains the negative receipt and
the deterministic product; it does not create another checkpoint or model path.
Human usefulness remains a separate required gate for any claim that the adapter
improves teaching quality.
