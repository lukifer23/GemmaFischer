#!/usr/bin/env python3
"""Minimal LoRA fine-tune script using Hugging Face Transformers + PEFT.
This script is intentionally conservative for macOS MPS: small batch, FP32, and periodic checkpointing.
It logs metrics, system resource usage, and writes a small training log.
"""
import argparse
import time
import json
import os
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
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import numpy as np
from typing import Dict, Any

# resource monitoring
import psutil


def log_system_stats(prefix=""):
    vm = psutil.virtual_memory()
    cpu = psutil.cpu_percent(interval=None)
    return {
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'prefix': prefix,
        'cpu_percent': cpu,
        'mem_total': vm.total,
        'mem_available': vm.available,
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
            print(f"Step {state.global_step}/{state.max_steps} ({progress:.1f}%) | Loss: {loss_str} | CPU: {cpu_str}%")

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/lora_finetune.yaml')
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = json.loads(json.dumps(__import__('yaml').safe_load(f)))

    model_path = cfg['model']['pretrained_model_path']
    out_dir = Path(cfg['training']['output_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)

    # load dataset
    ds_path = cfg['dataset']['path']
    if not Path(ds_path).exists():
        print(f"Dataset {ds_path} not found. Run create_finetune_dataset.py first.")
        return
    ds = load_dataset('json', data_files=ds_path, split='train')

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
    lora_dropout=lcfg.get('dropout', 0.0),
        bias='none',
        task_type='CAUSAL_LM'
    )
    model = get_peft_model(model, peft_config)

    # tokenization
    # allow shorter sequences on memory-constrained devices
    tokenizer_max_length = 512

    # If running on MPS, clamp batch sizes and shorten sequences to avoid OOM
    if torch.backends.mps.is_available():
        print('MPS backend detected — applying memory-safe defaults: batch_size=1, max_length=256')
        tokenizer_max_length = 256
    def preprocess(example):
        text = example['text']
        return tokenizer(text, truncation=True, max_length=tokenizer_max_length)

    ds = ds.map(preprocess, batched=False)

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # Coerce numeric config values to expected types
    tcfg = cfg['training']
    per_device_train_batch_size = int(tcfg.get('per_device_train_batch_size', 1))
    gradient_accumulation_steps = int(tcfg.get('gradient_accumulation_steps', 1))
    num_train_epochs = int(tcfg.get('num_train_epochs', 1))
    max_steps = int(tcfg.get('max_steps', 0))
    learning_rate = float(tcfg.get('learning_rate', 1e-4))
    logging_steps = int(tcfg.get('logging_steps', 10))
    save_steps = int(tcfg.get('save_steps', 50))

    # If MPS is available, enforce tiny per-device batch size to reduce memory usage
    if torch.backends.mps.is_available():
        per_device_train_batch_size = min(per_device_train_batch_size, 1)

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
        eval_strategy="steps",
        eval_steps=save_steps,  # Evaluate at same frequency as saving
        save_strategy="steps",
        load_best_model_at_end=True,
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
    )

    # Create evaluation dataset (10% of training data for validation)
    train_size = int(0.9 * len(ds))
    eval_size = len(ds) - train_size
    train_dataset = ds.select(range(train_size))
    eval_dataset = ds.select(range(train_size, len(ds)))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[
            # Custom callback for enhanced logging
            CustomCallback()
        ]
    )

    # Set start time for tracking
    trainer.state.start_time = time.time()

    # training loop
    resume_chk = None
    if isinstance(cfg, dict) and cfg.get('resume'):
        resume_chk = cfg['resume'].get('from_checkpoint')
        if resume_chk:
            print(f'Resuming training from checkpoint: {resume_chk}')

    print(f"🚀 Starting training with enhanced monitoring...")
    print(f"   Max steps: {max_steps}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Save steps: {save_steps}")
    print(f"   Evaluation steps: {save_steps}")
    print(f"   Early stopping patience: 5")
    print("=" * 60)

    trainer.train(resume_from_checkpoint=resume_chk)

    print('=' * 60)
    print('Training complete. Enhanced logs and checkpoints in', out_dir)

if __name__ == '__main__':
    main()
