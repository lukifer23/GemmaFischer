# Post-training and Unsloth Readiness

No training command exists, and the current state is **blocked**. Run the
inspectable gates without downloading weights or installing a toolchain:

```bash
uv run gemmafischer audit-data --output artifacts/data-audit/current-main.json
uv run gemmafischer training-readiness \
  --audit artifacts/data-audit/current-main.json \
  --manifest training/post-training.json \
  --output artifacts/training/readiness-current.json
```

Both commands intentionally return exit status 4 while blocked. The readiness
artifact reports `authorized_to_train: false` even after all prerequisites pass;
a green result authorizes asking for a bounded smoke run, not silently starting
one.

## Current evidence, 2026-08-30

The target machine is an Apple M3 Pro (`arm64`) with 18 GiB unified memory. The
project environment contains MLX 0.32.2 and MLX-LM 0.31.3; `unsloth` and
`unsloth-zoo` are not installed. Hardware therefore passes only the minimum
smoke-eligibility check. It does not prove that a chosen base checkpoint,
optimizer, sequence length, or batch configuration fits.

The current readiness artifact has four blockers:

- data contract and partition isolation are not green;
- an exact Unsloth/MLX toolchain is neither pinned nor installed;
- native, license-compatible base weights and hashes are not selected;
- a stable error taxonomy, frozen baseline, and frozen human review are absent.

The installed Gemma 4 E2B 4-bit MLX snapshot and LFM2.5 LM Studio quant are
inference artifacts, not accepted training sources. Selecting Gemma or LFM for
post-training requires native upstream weights, an immutable revision, every
weight hash, license review, and a repeated target-workload failure that cannot
be fixed in the harness or prompt.

## Why Unsloth remains a candidate

Unsloth's current repository documents macOS MLX training and contains a real
Apple-Silicon CI smoke that trains a small Gemma model for seven deterministic
LoRA steps and exercises export. That makes it credible enough to evaluate, not
safe enough to preselect. Upstream also has current Apple-Silicon loader and
startup issues, so model-specific load/export/reload evidence is mandatory:

- [Unsloth repository and macOS support](https://github.com/unslothai/unsloth)
- [real Apple-Silicon MLX CI](https://github.com/unslothai/unsloth/blob/main/.github/workflows/mlx-ci.yml)
- [current MLX model-config compatibility issue](https://github.com/unslothai/unsloth/issues/8126)
- [current Studio MLX capability race](https://github.com/unslothai/unsloth/issues/9120)

These links were reviewed on 2026-08-30. Versions must be pinned from the exact
tested environment; `latest`, an installer script, or a mutable branch is not a
reproducible training manifest.

## Allowed order

1. Build and pass 10,000/1,000/1,000 contract-valid train, validation, and
   untouched final-test rows.
2. Freeze the error taxonomy, untuned baseline, automated evaluation, and blind
   human packet before adapter training.
3. Select and hash native base weights. Do not train the production inference
   quant.
4. Pin Unsloth, unsloth-zoo, MLX, MLX-LM, tokenizer, chat template, seed, LoRA
   parameters, optimizer, sequence length, and memory ceiling in
   `training/post-training.json`.
5. Rerun `training-readiness`. After explicit authorization, execute only a
   7-20 step isolated smoke. Capture peak memory, loss, wall time, resume,
   adapter save, merge/export, and production-harness reload.
6. Compare the same rows and configuration against a minimal MLX-LM LoRA
   baseline. Continue only if the toolchain and model contract are reliable.
7. Promote nothing unless the adapter beats untuned Gemma and deterministic
   selection on frozen correctness, schema, grounding, usefulness, latency,
   memory, and regression gates.

Chess facts, legal-move authority, engine scoring, and answer grading remain
deterministic. Any adapter may select or order already validated coaching claims;
it may not invent chess evidence or grade the learner.
