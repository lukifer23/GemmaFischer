#!/usr/bin/env python3
"""Minimal LoRA fine-tune script using Hugging Face Transformers + PEFT.
This script is intentionally conservative for macOS MPS: small batch, FP32, and periodic checkpointing.
It logs metrics, system resource usage, and writes a small training log.
"""
import argparse
import time
import json
import os
import sys
from pathlib import Path
from datetime import datetime

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    TrainerCallback,
    TrainerState,
    TrainerControl
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import numpy as np
from typing import Dict, Any, List

# resource monitoring
import psutil

# Ensure project root is on sys.path for absolute imports like 'src.*'
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.config_validation import (
    ConfigValidationError,
    validate_lora_config,
)


def log_system_stats(prefix=""):
    vm = psutil.virtual_memory()
    sm = psutil.swap_memory()
    cpu = psutil.cpu_percent(interval=None)
    proc = psutil.Process()
    rss = getattr(proc.memory_info(), 'rss', 0)
    return {
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'prefix': prefix,
        'cpu_percent': cpu,
        # system memory
        'mem_total': vm.total,
        'mem_available': vm.available,
        'mem_used_calc': max(0, vm.total - vm.available),
        # swap
        'swap_total': getattr(sm, 'total', 0),
        'swap_used': getattr(sm, 'used', 0),
        # process rss
        'proc_rss': rss,
    }


class CustomCallback(TrainerCallback):
    """Enhanced callback for better training monitoring and logging."""

    def __init__(self):
        self.best_eval_loss = float('inf')
        self.patience_counter = 0
        self.max_patience = 5  # Early stopping patience

    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        """Enhanced logging with system stats and training progress."""
        if logs is not None:
            # Add system stats to logs
            system_stats = log_system_stats("training")
            enhanced_logs = {**logs, **system_stats}

            # Calculate training progress
            progress = (state.global_step / state.max_steps) * 100 if state.max_steps else 0
            enhanced_logs['progress_percent'] = progress
            enhanced_logs['time_elapsed'] = time.time() - state.start_time if hasattr(state, 'start_time') else 0

            # Log enhanced metrics
            loss_val = logs.get('loss', 'N/A')
            loss_str = f"{loss_val:.4f}" if isinstance(loss_val, (int, float)) else str(loss_val)
            cpu_val = system_stats['cpu_percent']
            cpu_str = f"{cpu_val:.1f}" if isinstance(cpu_val, (int, float)) else str(cpu_val)
            mem_used_gb = system_stats['mem_used_calc'] / (1024**3)
            proc_rss_gb = system_stats['proc_rss'] / (1024**3)
            swap_used_gb = system_stats['swap_used'] / (1024**3)
            print(
                f"Step {state.global_step}/{state.max_steps} ({progress:.1f}%) | Loss: {loss_str} | "
                f"CPU: {cpu_str}% | SystemUsed: {mem_used_gb:.1f}GB | ProcRSS: {proc_rss_gb:.2f}GB | SwapUsed: {swap_used_gb:.1f}GB"
            )

            # Save enhanced logs
            log_file = Path(args.output_dir) / 'enhanced_train_log.jsonl'
            with open(log_file, 'a', encoding='utf-8') as f:
                json.dump(enhanced_logs, f)
                f.write('\n')

    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, metrics=None, **kwargs):
        """Handle evaluation results with early stopping."""
        if metrics is not None:
            eval_loss = metrics.get('eval_loss', float('inf'))

            # Check for best model
            if eval_loss < self.best_eval_loss:
                self.best_eval_loss = eval_loss
                self.patience_counter = 0
                print(f"🎯 New best eval loss: {eval_loss:.4f}")
            else:
                self.patience_counter += 1
                print(f"📈 Eval loss: {eval_loss:.4f} (patience: {self.patience_counter}/{self.max_patience})")

            # Early stopping
            if self.patience_counter >= self.max_patience:
                print("🛑 Early stopping triggered!")
                control.should_training_stop = True

    def on_save(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """Log checkpoint saves."""
        print(f"💾 Checkpoint saved at step {state.global_step}")

    def on_train_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """Final training summary."""
        final_stats = log_system_stats("training_end")
        final_stats.update({
            'total_steps': state.global_step,
            'best_eval_loss': self.best_eval_loss,
            'training_time_seconds': time.time() - state.start_time if hasattr(state, 'start_time') else 0
        })

        print("🏁 Training completed!")
        print(f"   Best eval loss: {self.best_eval_loss:.4f}")
        # Save final summary
        summary_file = Path(args.output_dir) / 'training_summary.json'
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(final_stats, f, indent=2)


class InstructionDataCollator:
    """Data collator for instruction-style datasets with prompt/response columns.

    Masks prompt tokens (label = -100) and supervises only response tokens.
    Tokenizes on the fly to avoid pre-tokenization complexity.
    """

    def __init__(self, tokenizer: AutoTokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        prompts: List[str] = []
        responses: List[str] = []
        for ex in examples:
            p = ex.get('prompt') or ''
            r = ex.get('response') or ''
            prompts.append(p)
            responses.append(r)

        # Tokenize separately to compute prompt lengths
        tok_prompts = self.tokenizer(prompts, add_special_tokens=False, truncation=True, max_length=self.max_length)
        tok_responses = self.tokenizer(responses, add_special_tokens=False, truncation=True, max_length=self.max_length)

        input_ids_batch: List[List[int]] = []
        attention_masks: List[List[int]] = []
        labels_batch: List[List[int]] = []

        eos_id = self.tokenizer.eos_token_id

        for p_ids, r_ids in zip(tok_prompts['input_ids'], tok_responses['input_ids']):
            # Compose input: [prompt] + [response] + [eos]
            input_ids = p_ids + r_ids + ([eos_id] if eos_id is not None else [])
            input_ids = input_ids[: self.max_length]

            # Attention mask
            attn = [1] * len(input_ids)

            # Labels: mask prompt, supervise response + eos
            labels = ([-100] * min(len(p_ids), len(input_ids)))
            resp_len = len(input_ids) - len(labels)
            labels += (r_ids + ([eos_id] if eos_id is not None else []))[:resp_len]
            labels = labels[: len(input_ids)]

            input_ids_batch.append(input_ids)
            attention_masks.append(attn)
            labels_batch.append(labels)

        # Pad to max in batch
        max_len = max(len(x) for x in input_ids_batch)
        def pad(seq, pad_id):
            return seq + [pad_id] * (max_len - len(seq))

        input_ids_batch = [pad(x, self.tokenizer.pad_token_id or eos_id or 0) for x in input_ids_batch]
        attention_masks = [pad(x, 0) for x in attention_masks]
        labels_batch = [pad(x, -100) for x in labels_batch]

        return {
            'input_ids': torch.tensor(input_ids_batch, dtype=torch.long),
            'attention_mask': torch.tensor(attention_masks, dtype=torch.long),
            'labels': torch.tensor(labels_batch, dtype=torch.long),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='auto', help='Path to YAML config or "auto" to choose per --expert')
    parser.add_argument('--expert', type=str, default='all', choices=['all', 'uci', 'tutor', 'director'], help='Filter dataset by task for MoE expert training')
    parser.add_argument('--use_instruction_collator', action='store_true', help='Use instruction-style collator (prompt masked, supervise response only)')
    parser.add_argument('--disable_eval', action='store_true', help='Disable periodic evaluation to speed up smoke runs')
    parser.add_argument('--eval_steps', type=int, default=None, help='Override evaluation frequency in steps (when eval enabled)')
    parser.add_argument('--max_steps_override', type=int, default=0, help='Override max_steps from config for quick smoke runs')
    args = parser.parse_args()

    # Resolve config path (support "auto" and robust relative paths)
    cfg_path = args.config
    cfg_dir = Path(__file__).parent / 'configs'
    if cfg_path == 'auto':
        auto_map = {
            'uci': cfg_dir / 'lora_uci.yaml',
            'tutor': cfg_dir / 'lora_tutor.yaml',
            'director': cfg_dir / 'lora_director_expert.yaml',
            'all': cfg_dir / 'lora_finetune.yaml',
        }
        cfg_path = str(auto_map.get(args.expert, auto_map['all']))
    else:
        # If not an absolute path and not found relative to CWD, try relative to this script
        p = Path(cfg_path)
        if not p.exists():
            alt = cfg_dir / p.name
            if alt.exists():
                cfg_path = str(alt)

    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg = json.loads(json.dumps(__import__('yaml').safe_load(f)))

    # Validate LoRA configuration
    try:
        cfg["lora"] = validate_lora_config(cfg.get("lora", {}))
    except ConfigValidationError as e:
        print(f"Invalid LoRA configuration: {e}")
        sys.exit(1)

    # Cap CPU threads to 2 by default if not set
    os.environ.setdefault('OMP_NUM_THREADS', '2')
    os.environ.setdefault('MKL_NUM_THREADS', '2')
    os.environ.setdefault('NUMEXPR_NUM_THREADS', '2')

    model_path = cfg['model']['pretrained_model_path']
    out_dir = Path(cfg['training']['output_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Detect curriculum early. If present, we'll build per-phase datasets below
    use_curriculum = bool(isinstance(cfg, dict) and cfg.get('curriculum'))

    # load dataset(s) only if NOT using curriculum
    ds = None
    if not use_curriculum:
        if 'datasets' in cfg and isinstance(cfg['datasets'], list) and cfg['datasets']:
            # Weighted mixture using dataset_mixer
            try:
                from src.training.dataset_mixer import build_mixture
                specs = []
                for item in cfg['datasets']:
                    p = item.get('path')
                    w = float(item.get('weight', 1.0))
                    if p and Path(p).exists():
                        specs.append({'path': p, 'weight': w})
                if not specs:
                    print('No valid dataset specs found in config.datasets.')
                    return
                ds = build_mixture(specs)
                print(f"Loaded mixed dataset from {len(specs)} sources")
            except Exception as e:
                print(f"Failed to build mixed dataset: {e}")
                return
        else:
            ds_path = cfg.get('dataset', {}).get('path') if isinstance(cfg.get('dataset'), dict) else None
            if not ds_path:
                print('Config must include either curriculum, datasets list, or dataset.path')
                return
            if not Path(ds_path).exists():
                print(f"Dataset {ds_path} not found. Run create_finetune_dataset.py or refine_dataset.py first.")
                return
            from src.training.dataset_mixer import _load_single_jsonl
            ds = _load_single_jsonl(ds_path)

    # Optional expert filtering by task field
    if ds is not None and args.expert != 'all':
        task_map = {'uci': 'engine_uci', 'tutor': 'tutor_explain', 'director': 'director_qa'}
        target_task = task_map.get(args.expert)
        if target_task:
            try:
                ds = ds.filter(lambda ex: (ex.get('task') or '') == target_task)
                print(f"Filtered dataset for task={target_task}, size={len(ds)}")
                if len(ds) == 0:
                    print("No samples matched the requested expert task. Please use --expert all or provide a dataset with task tags.")
                    return
            except Exception as e:
                print(f"Warning: task filter failed ({e}); proceeding without filtering")

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    # load model with the recommended eager attention implementation for Gemma3
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        device_map='auto',
        attn_implementation='eager'
    )

    # prepare peft lora
    lcfg = cfg['lora']
    peft_config = LoraConfig(
        r=lcfg['r'],
        lora_alpha=lcfg['lora_alpha'],
        target_modules=lcfg['target_modules'],
        lora_dropout=lcfg.get('lora_dropout', 0.0),
        bias='none',
        task_type='CAUSAL_LM'
    )
    model = get_peft_model(model, peft_config)

    # tokenization / collator
    # allow shorter sequences on memory-constrained devices
    tokenizer_max_length = 512
    if torch.backends.mps.is_available():
        print('MPS backend detected — applying memory-safe defaults: batch_size=1, max_length=256')
        tokenizer_max_length = 256

    # Default collator preference: enable for tutor/director when not explicitly requested
    use_instruction = bool(args.use_instruction_collator)
    if not use_instruction and args.expert in ('tutor', 'director'):
        use_instruction = True

    if not use_instruction:
        def preprocess(example):
            if example.get('prompt') is not None and example.get('response') is not None:
                text = f"{example['prompt']}{example['response']}"
            else:
                text = example.get('text', '')
            return tokenizer(text, truncation=True, max_length=tokenizer_max_length)

        # Only map here if a single/mixed dataset was already built (no curriculum)
        if 'curriculum' not in cfg or not cfg.get('curriculum'):
            ds = ds.map(preprocess, batched=False)
        data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    else:
        data_collator = InstructionDataCollator(tokenizer, max_length=tokenizer_max_length)

    # Coerce numeric config values to expected types
    tcfg = cfg['training']
    per_device_train_batch_size = int(tcfg.get('per_device_train_batch_size', 1))
    gradient_accumulation_steps = int(tcfg.get('gradient_accumulation_steps', 1))
    num_train_epochs = int(tcfg.get('num_train_epochs', 1))
    max_steps = int(tcfg.get('max_steps', 0))
    if int(args.max_steps_override or 0) > 0:
        max_steps = int(args.max_steps_override)
    learning_rate = float(tcfg.get('learning_rate', 1e-4))
    logging_steps = int(tcfg.get('logging_steps', 10))
    save_steps = int(tcfg.get('save_steps', 50))

    # If MPS is available, enforce tiny per-device batch size to reduce memory usage
    if torch.backends.mps.is_available():
        per_device_train_batch_size = min(per_device_train_batch_size, 1)

    # Evaluation configuration
    eval_strategy_value = 'no' if args.disable_eval else 'steps'
    eval_steps_value = int(args.eval_steps) if (args.eval_steps is not None) else save_steps

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        max_steps=max_steps,
        learning_rate=learning_rate,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=3,  # Keep only last 3 checkpoints to save disk space
        eval_strategy=eval_strategy_value,
        eval_steps=eval_steps_value,
        save_strategy="steps",
        load_best_model_at_end=(not args.disable_eval),
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=tcfg.get('fp16', False),
        bf16=tcfg.get('bf16', False),
        lr_scheduler_type="cosine",  # Cosine annealing for better convergence
        warmup_steps=int(max_steps * 0.1),  # 10% warmup steps
        weight_decay=0.01,
        adam_beta1=0.9,
        adam_beta2=0.999,
        max_grad_norm=1.0,
        report_to=[],  # We'll add MLflow later
        logging_first_step=True,
        logging_nan_inf_filter=False,
        remove_unused_columns=not use_instruction,
    )

    # Create evaluation dataset (10% split)
    def split_train_eval(dataset):
        if hasattr(dataset, 'train_test_split'):
            s = dataset.train_test_split(test_size=0.1, seed=3407)
            return s['train'], s['test']
        size = len(dataset)
        trn = int(0.9 * size)
        return dataset.select(range(trn)), dataset.select(range(trn, size))

    # Curriculum support: if cfg['curriculum'] present, iterate phases
    curriculum = cfg.get('curriculum') if isinstance(cfg, dict) else None
    if curriculum and isinstance(curriculum, list) and len(curriculum) > 0:
        print('Using curriculum with', len(curriculum), 'phases')
        from src.training.dataset_mixer import build_mixture
        total_phases = len(curriculum)
        for idx, phase in enumerate(curriculum, 1):
            phase_steps = int(phase.get('steps', 0))
            phase_specs = phase.get('datasets', [])
            print(f"Phase {idx}/{total_phases}: steps={phase_steps}")
            specs = []
            for item in phase_specs:
                p = item.get('path')
                w = float(item.get('weight', 1.0))
                if p and Path(p).exists():
                    specs.append({'path': p, 'weight': w})
            if not specs:
                print('  Skipping phase; no valid datasets')
                continue
            mixed = build_mixture(specs)
            if len(mixed) == 0:
                print('  Skipping phase; built dataset is empty')
                continue
            # Tokenize per-phase dataset only if not using instruction collator
            if not use_instruction:
                mixed = mixed.map(preprocess, batched=False)
            train_dataset, eval_dataset = split_train_eval(mixed)
            # Override steps for this phase if provided
            phase_args = TrainingArguments(
                **{**training_args.to_dict(), 'max_steps': (int(args.max_steps_override) or phase_steps or max_steps)}
            )

            trainer = Trainer(
                model=model,
                args=phase_args,
                train_dataset=train_dataset,
                eval_dataset=None if args.disable_eval else eval_dataset,
                data_collator=data_collator,
                callbacks=[CustomCallback()]
            )
            trainer.state.start_time = time.time()
            trainer.train()
            print(f"Completed curriculum phase {idx}")
    else:
        # Single-phase training
        train_dataset, eval_dataset = split_train_eval(ds)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=None if args.disable_eval else eval_dataset,
            data_collator=data_collator,
            callbacks=[CustomCallback()]
        )
        trainer.state.start_time = time.time()
        trainer.train()

    print('=' * 60)
    print('Training complete. Enhanced logs and checkpoints in', out_dir)

if __name__ == '__main__':
    main()
