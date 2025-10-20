#!/usr/bin/env python3
"""
Unified trainer entrypoint consolidating training for all experts on MPS-only.

Usage:
  python -m src.training.unified_trainer --expert uci --config configs/default.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from ..config.config_manager import ConfigManager


def _resolve_config_path(path: Optional[str]) -> Path:
    if path:
        p = Path(path)
        if p.exists():
            return p
        raise FileNotFoundError(f"Config file not found: {path}")
    project_root = Path(__file__).resolve().parents[2]
    default_yaml = project_root / "configs" / "default.yaml"
    if not default_yaml.exists():
        raise FileNotFoundError("Default config not found at configs/default.yaml")
    return default_yaml


def main() -> int:
    parser = argparse.ArgumentParser(description="Unified ChessGemma trainer (MPS-only)")
    parser.add_argument("--expert", choices=["uci", "tutor", "director"], required=True)
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    parser.add_argument("--max_steps", type=int, default=None, help="Override max steps")
    parser.add_argument("--no_validate", action="store_true", help="Skip config validation")
    args = parser.parse_args()

    cfg_path = _resolve_config_path(args.config)

    # Load configuration
    manager = ConfigManager()
    config = manager.load_from_file(cfg_path)

    # Select expert-specific training config
    expert_cfg = config.get_training_config(args.expert)

    if args.max_steps is not None:
        expert_cfg["max_steps"] = int(args.max_steps)

    # Basic validation of essential fields
    if not args.no_validate:
        errors = []
        if expert_cfg.get("per_device_train_batch_size", 0) <= 0:
            errors.append("per_device_train_batch_size must be positive")
        if expert_cfg.get("max_steps", 0) <= 0 and expert_cfg.get("num_train_epochs", 0) <= 0:
            errors.append("Either max_steps or num_train_epochs must be positive")
        if errors:
            raise ValueError("Invalid training configuration: " + "; ".join(errors))

    # Defer heavy imports after validation
    from datasets import load_dataset
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    from trl import SFTTrainer, SFTConfig

    # MPS-only
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    # Resolve model
    model_ref = config.model.local_model_path or config.model.pretrained_model_path
    tokenizer = AutoTokenizer.from_pretrained(model_ref, local_files_only=bool(config.model.local_model_path), trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    torch_dtype = torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_ref,
        local_files_only=bool(config.model.local_model_path),
        device_map=None,
        attn_implementation="eager",
        trust_remote_code=True,
        torch_dtype=torch_dtype,
    )

    # LoRA
    lora_cfg = config.get_lora_config()
    lora = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        bias=lora_cfg["bias"],
        task_type=lora_cfg["task_type"],
    )
    model = get_peft_model(model, lora)
    model.to(device)
    model.train()

    # Dataset (simple supervised fine-tune on standardized data)
    datasets = config.datasets or []
    if not datasets:
        raise ValueError("No datasets configured in config file")
    # Use a small sample to keep memory in check by default
    ds_path = Path(datasets[0]["path"]).as_posix()
    dataset = load_dataset("json", data_files=ds_path, split="train[:200]")

    def convert(ex):
        return {
            "conversations": [
                {"role": "system", "content": ex.get("task", "")},
                {"role": "user", "content": str(ex.get("input", ""))},
                {"role": "assistant", "content": ex.get("expected_output", "")},
            ]
        }

    dataset = dataset.map(convert)

    # Trainer
    sft_args = SFTConfig(
        max_seq_length=512,
        per_device_train_batch_size=int(expert_cfg["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(expert_cfg["gradient_accumulation_steps"]),
        max_steps=int(expert_cfg["max_steps"]),
        learning_rate=float(expert_cfg["learning_rate"]),
        fp16=False,
        bf16=False,
        logging_steps=int(expert_cfg["logging_steps"]),
        optim="adamw_hf",
        weight_decay=float(expert_cfg["weight_decay"]),
        seed=config.system.seed,
        output_dir=expert_cfg["output_dir"],
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=sft_args,
    )

    trainer.train()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


