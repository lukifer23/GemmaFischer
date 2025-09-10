# Gemma-3 (270M) LoRA fine-tuning for Chess — Reworked Plan

## One-line goal
Fine-tune Gemma-3 (270M) with LoRA on the ChessInstruct dataset to solve "find the missing move" and related chess instruction tasks, using Apple Silicon (M-series) with Unsloth.

## High-level phases
- Phase A: Environment + scaffold (venv, requirements, scripts, safe train/infer stubs). (status: scaffold created)
- Phase B: Data preparation and smoke training (10 steps) to validate pipeline.
- Phase C: Full training, checkpoints, evaluation, and adapter export.
- Phase D: Optional quantization/merge/export and deployment.

## Checklist (actionable)
- [ ] Create venv and install dependencies
- [ ] Validate PyTorch + MPS availability
- [ ] Implement data conversion (`data/prepare_dataset.py`)
- [ ] Implement `train.py` (LoRA attach, SFTTrainer) — safe by default
- [ ] Implement `inference.py` to load adapter and generate
- [ ] Add unit tests and run smoke training (10 steps)
- [ ] Save adapter and verify inference
- [ ] Document and finalize `README.md`

## Risk & notes
- `bitsandbytes` may not build on Apple Silicon — fallback to `adamw_hf` is recommended.
- Avoid running heavy downloads or training while other models run on the same machine (coordinate timing).

## Acceptance criteria
- Smoke run (10 steps) completes without exceptions and adapter artifacts are created.
- Tests pass (data conversion + inference stub).

