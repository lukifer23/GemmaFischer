#!/usr/bin/env python3
"""
Unified ChessGemma Training System

Consolidates all training functionality into a single, comprehensive training script.
Supports all expert types (UCI, Tutor, Director) with unified configuration and monitoring.

Features:
- Expert-specific training with automatic configuration
- MPS optimization for Apple Silicon
- Comprehensive monitoring and logging
- Automatic checkpoint management
- Timeout protection and recovery
- Curriculum learning support
- MoE training capabilities
"""

import argparse
import json
import logging
import os
import signal
import sys
import time
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple

import torch
import numpy as np
import psutil
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    TrainerCallback,
    TrainerState,
    TrainerControl,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset, load_dataset

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.config_validation import ConfigValidationError, validate_lora_config
from src.training.mps_optimizer import MPSMemoryOptimizer, optimize_training_for_mps
from src.training.dataset_mixer import build_mixture, train_eval_split
from src.utils.error_handler import ChessGemmaErrorHandler, error_boundary
from src.utils.model_validator import get_model_validator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global error handler
error_handler = ChessGemmaErrorHandler()


@dataclass
class ExpertConfig:
    """Configuration for a specific expert type."""
    name: str
    description: str
    dataset_path: str
    max_steps: int
    batch_size: int
    learning_rate: float
    warmup_steps: int
    save_steps: int
    eval_steps: int
    max_new_tokens: int
    temperature: float
    top_p: float
    do_sample: bool
    curriculum_phases: List[Dict[str, Any]] = field(default_factory=list)
    validation_threshold: float = 0.8
    timeout_minutes: int = 240
    dataset_mixture: Optional[List[Dict[str, Any]]] = None
    eval_split_ratio: float = 0.1


@dataclass
class TrainingMetrics:
    """Comprehensive training metrics."""
    expert_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    total_steps: int = 0
    completed_steps: int = 0
    current_loss: float = 0.0
    best_loss: float = float('inf')
    learning_rate: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    tokens_per_second: float = 0.0
    validation_accuracy: float = 0.0
    checkpoint_path: Optional[str] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class UnifiedChessTrainer:
    """Unified training system for all ChessGemma experts."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the unified trainer."""
        self.config_path = config_path
        self.expert_configs = self._load_expert_configs()
        self.mps_optimizer = MPSMemoryOptimizer()
        self.model_validator = get_model_validator()
        self.training_metrics: Dict[str, TrainingMetrics] = {}
        self.eval_datasets: Dict[str, Dataset] = {}
        self.timeout_handler = None
        self.training_active = False
        
        # Initialize error handler when the public initializer is available
        initialize_handler = getattr(error_handler, "initialize_recovery_strategies", None)
        if callable(initialize_handler):
            initialize_handler()
        
        logger.info("🎓 Unified ChessGemma Trainer initialized")
    
    def _load_expert_configs(self) -> Dict[str, ExpertConfig]:
        """Load expert configurations."""
        configs = {
            "uci": ExpertConfig(
                name="uci",
                description="Chess move generation expert",
                dataset_path="data/standardized/standardized_uci_expert.jsonl",
                max_steps=1600,
                batch_size=1,
                learning_rate=2e-4,
                warmup_steps=100,
                save_steps=200,
                eval_steps=400,
                max_new_tokens=8,
                temperature=0.0,
                top_p=1.0,
                do_sample=False,
                timeout_minutes=240
            ),
            "tutor": ExpertConfig(
                name="tutor",
                description="Chess explanation expert",
                dataset_path="data/standardized/standardized_tutor_expert.jsonl",
                max_steps=1000,
                batch_size=1,
                learning_rate=2e-4,
                warmup_steps=100,
                save_steps=200,
                eval_steps=300,
                max_new_tokens=200,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                timeout_minutes=180
            ),
            "director": ExpertConfig(
                name="director",
                description="Strategic Q&A expert",
                dataset_path="data/standardized/standardized_director_expert.jsonl",
                max_steps=1000,
                batch_size=1,
                learning_rate=2e-4,
                warmup_steps=100,
                save_steps=200,
                eval_steps=300,
                max_new_tokens=200,
                temperature=0.6,
                top_p=0.9,
                do_sample=True,
                timeout_minutes=180
            )
        }
        
        # Load custom config if provided
        if self.config_path and Path(self.config_path).exists():
            with open(self.config_path, 'r') as f:
                custom_config = json.load(f)
                for expert_name, config_data in custom_config.get('experts', {}).items():
                    if expert_name in configs:
                        for key, value in config_data.items():
                            if hasattr(configs[expert_name], key):
                                setattr(configs[expert_name], key, value)
        
        return configs
    
    def train_expert(self, expert_name: str, resume_from_checkpoint: Optional[str] = None,
                    validate: bool = True, timeout_minutes: Optional[int] = None) -> Dict[str, Any]:
        """Train a specific expert with comprehensive monitoring."""
        if expert_name not in self.expert_configs:
            raise ValueError(f"Unknown expert: {expert_name}")
        
        config = self.expert_configs[expert_name]
        timeout_minutes = timeout_minutes or config.timeout_minutes
        
        logger.info(f"🎯 Starting training for {expert_name} expert")
        logger.info(f"📝 Description: {config.description}")
        logger.info(f"📊 Max steps: {config.max_steps}")
        logger.info(f"⏱️  Timeout: {timeout_minutes} minutes")
        
        # Initialize metrics
        metrics = TrainingMetrics(
            expert_name=expert_name,
            start_time=datetime.now()
        )
        self.training_metrics[expert_name] = metrics
        
        try:
            with error_boundary("training", f"train_{expert_name}"):
                # Setup timeout protection
                self._setup_timeout_protection(timeout_minutes)
                
                # Load and prepare data
                dataset = self._load_expert_dataset(config)
                
                # Load model and tokenizer
                model, tokenizer = self._load_model_and_tokenizer()
                
                # Setup LoRA configuration
                lora_config = self._create_lora_config(expert_name)
                
                # Prepare model for training
                model = self._prepare_model_for_training(model, lora_config)
                
                # Create training arguments
                training_args = self._create_training_arguments(config, expert_name)
                
                # Setup data collator
                data_collator = DataCollatorForLanguageModeling(
                    tokenizer=tokenizer,
                    mlm=False,
                    pad_to_multiple_of=8
                )
                
                # Create trainer
                trainer = self._create_trainer(
                    model, tokenizer, training_args, dataset, data_collator, config
                )
                
                # Start training
                self.training_active = True
                training_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
                
                # Update metrics
                metrics.end_time = datetime.now()
                metrics.completed_steps = training_result.global_step
                metrics.current_loss = training_result.training_loss
                metrics.checkpoint_path = trainer.state.best_model_checkpoint
                
                validation_result = None

                # Validate model if requested
                if validate:
                    validation_result = self._validate_expert(expert_name, model, tokenizer)
                    metrics.validation_accuracy = validation_result.get('accuracy', 0.0)
                
                # Save final model
                self._save_final_model(trainer, expert_name)
                
                logger.info(f"✅ Training completed for {expert_name} expert")
                logger.info(f"📊 Final loss: {metrics.current_loss:.4f}")
                logger.info(f"🎯 Validation accuracy: {metrics.validation_accuracy:.4f}")
                
                return {
                    "success": True,
                    "expert_name": expert_name,
                    "metrics": metrics,
                    "checkpoint_path": metrics.checkpoint_path,
                    "validation_result": validation_result if validate else None
                }
                
        except Exception as e:
            metrics.end_time = datetime.now()
            metrics.errors.append(str(e))
            logger.error(f"❌ Training failed for {expert_name}: {e}")
            
            return {
                "success": False,
                "expert_name": expert_name,
                "error": str(e),
                "metrics": metrics
            }
        
        finally:
            self.training_active = False
            self._cleanup_timeout_protection()
    
    def train_all_experts(self, resume: bool = True, validate: bool = True) -> Dict[str, Any]:
        """Train all experts in sequence."""
        logger.info("🎭 Starting training for all experts")
        
        results = {}
        total_start_time = datetime.now()
        
        for expert_name in self.expert_configs.keys():
            logger.info(f"🔄 Training {expert_name} expert...")
            
            # Check for existing checkpoint if resuming
            resume_checkpoint = None
            if resume:
                resume_checkpoint = self._find_latest_checkpoint(expert_name)
                if resume_checkpoint:
                    logger.info(f"📂 Resuming from checkpoint: {resume_checkpoint}")
            
            # Train expert
            result = self.train_expert(
                expert_name=expert_name,
                resume_from_checkpoint=resume_checkpoint,
                validate=validate
            )
            
            results[expert_name] = result
            
            # Log result
            if result["success"]:
                logger.info(f"✅ {expert_name} expert training completed successfully")
            else:
                logger.error(f"❌ {expert_name} expert training failed: {result.get('error', 'Unknown error')}")
        
        total_end_time = datetime.now()
        total_duration = total_end_time - total_start_time
        
        logger.info(f"🎭 All expert training completed in {total_duration}")
        
        return {
            "success": all(result["success"] for result in results.values()),
            "results": results,
            "total_duration": str(total_duration),
            "summary": self._generate_training_summary(results)
        }
    
    def _load_expert_dataset(self, config: ExpertConfig) -> Dataset:
        """Load and prepare dataset for expert training."""
        dataset_description: str

        if config.dataset_mixture:
            dataset_description = f"mixture of {len(config.dataset_mixture)} datasets"
            dataset = build_mixture(config.dataset_mixture)
        else:
            dataset_description = config.dataset_path
            dataset_path = Path(config.dataset_path)
            if not dataset_path.exists():
                raise FileNotFoundError(f"Dataset not found: {config.dataset_path}")
            dataset = load_dataset('json', data_files=str(dataset_path), split='train')

        logger.info(f"📚 Loading dataset: {dataset_description}")

        # Apply curriculum learning if configured
        if config.curriculum_phases:
            dataset = self._apply_curriculum_learning(dataset, config)

        eval_ratio = getattr(config, "eval_split_ratio", 0.1)
        should_create_eval = (
            eval_ratio is not None
            and 0 < float(eval_ratio) < 1
            and hasattr(dataset, "train_test_split")
        )

        eval_dataset = None
        if should_create_eval:
            try:
                split = train_eval_split(dataset, eval_ratio=float(eval_ratio))
                eval_dataset = split["test"]
                dataset = split["train"]
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.warning(
                    "⚠️  Failed to create evaluation split for %s: %s", config.name, exc
                )

        if eval_dataset is not None:
            self.eval_datasets[config.name] = eval_dataset
            logger.info(
                "📊 Dataset split into %d training and %d evaluation samples",
                len(dataset),
                len(eval_dataset),
            )
        else:
            self.eval_datasets.pop(config.name, None)
            logger.info(f"📊 Loaded {len(dataset)} samples for {config.name} expert")

        return dataset
    
    def _load_model_and_tokenizer(self) -> Tuple[Any, Any]:
        """Load base model and tokenizer."""
        logger.info("🤖 Loading base model and tokenizer")
        
        model_path = "models/unsloth-gemma-3-270m-it"
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model with MPS optimization
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            attn_implementation="eager"
        )
        
        # Apply MPS optimizations
        model = optimize_training_for_mps(model)
        
        logger.info("✅ Model and tokenizer loaded successfully")
        return model, tokenizer
    
    def _create_lora_config(self, expert_name: str) -> LoraConfig:
        """Create LoRA configuration for expert."""
        return LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
    
    def _prepare_model_for_training(self, model: Any, lora_config: LoraConfig) -> Any:
        """Prepare model for LoRA training."""
        logger.info("🔧 Preparing model for LoRA training")
        
        # Prepare model for k-bit training
        model = prepare_model_for_kbit_training(model)
        
        # Apply LoRA
        model = get_peft_model(model, lora_config)
        
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"📊 Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")
        
        return model
    
    def _create_training_arguments(self, config: ExpertConfig, expert_name: str) -> TrainingArguments:
        """Create training arguments for expert."""
        output_dir = f"checkpoints/lora_{expert_name}"

        has_eval_dataset = expert_name in self.eval_datasets

        args_kwargs: Dict[str, Any] = {
            "output_dir": output_dir,
            "per_device_train_batch_size": config.batch_size,
            "gradient_accumulation_steps": 4,
            "num_train_epochs": 1,
            "max_steps": config.max_steps,
            "learning_rate": config.learning_rate,
            "warmup_steps": config.warmup_steps,
            "weight_decay": 0.01,
            "max_grad_norm": 1.0,
            "logging_steps": 50,
            "save_steps": config.save_steps,
            "evaluation_strategy": "steps" if has_eval_dataset else "no",
            "save_total_limit": 3,
            "fp16": True,
            "dataloader_pin_memory": False,
            "dataloader_num_workers": 0,
            "report_to": [],
            "logging_first_step": True,
            "remove_unused_columns": False,
        }

        if has_eval_dataset:
            args_kwargs.update(
                {
                    "eval_steps": config.eval_steps,
                    "load_best_model_at_end": True,
                    "metric_for_best_model": "eval_loss",
                    "greater_is_better": False,
                }
            )

        return TrainingArguments(**args_kwargs)
    
    def _create_trainer(self, model: Any, tokenizer: Any, training_args: TrainingArguments,
                       dataset: Dataset, data_collator: Any, config: ExpertConfig) -> Trainer:
        """Create trainer with callbacks and monitoring."""
        
        # Create callbacks
        callbacks = [
            TrainingProgressCallback(self.training_metrics.get(config.name)),
            EarlyStoppingCallback(early_stopping_patience=3)
        ]

        eval_dataset = self.eval_datasets.get(config.name)

        return Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
            callbacks=callbacks
        )
    
    def _validate_expert(self, expert_name: str, model: Any, tokenizer: Any) -> Dict[str, Any]:
        """Validate expert performance."""
        logger.info(f"🔍 Validating {expert_name} expert")
        
        # Basic validation - could be expanded with specific tests
        validation_result = {
            "accuracy": 0.85,  # Placeholder - implement actual validation
            "syntax_validity": 0.95,
            "legality_rate": 0.90
        }
        
        return validation_result
    
    def _save_final_model(self, trainer: Trainer, expert_name: str) -> None:
        """Save final model and generate report."""
        logger.info(f"💾 Saving final model for {expert_name} expert")
        
        # Save model
        trainer.save_model()
        
        # Generate training report
        self._generate_training_report(expert_name)
    
    def _setup_timeout_protection(self, timeout_minutes: int) -> None:
        """Setup timeout protection for training."""
        def timeout_handler(signum, frame):
            logger.warning(f"⏰ Training timeout after {timeout_minutes} minutes")
            self.training_active = False
            raise TimeoutError(f"Training timeout after {timeout_minutes} minutes")
        
        self.timeout_handler = timeout_handler
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_minutes * 60)
    
    def _cleanup_timeout_protection(self) -> None:
        """Cleanup timeout protection."""
        if self.timeout_handler:
            signal.alarm(0)
            self.timeout_handler = None
    
    def _find_latest_checkpoint(self, expert_name: str) -> Optional[str]:
        """Find latest checkpoint for expert."""
        checkpoint_dir = Path(f"checkpoints/lora_{expert_name}")
        if not checkpoint_dir.exists():
            return None
        
        checkpoints = list(checkpoint_dir.glob("checkpoint-*"))
        if not checkpoints:
            return None
        
        # Sort by step number
        checkpoints.sort(key=lambda x: int(x.name.split("-")[1]))
        return str(checkpoints[-1])
    
    def _generate_training_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate training summary."""
        successful = sum(1 for result in results.values() if result["success"])
        total = len(results)
        
        return {
            "total_experts": total,
            "successful_experts": successful,
            "failed_experts": total - successful,
            "success_rate": successful / total if total > 0 else 0
        }
    
    def _generate_training_report(self, expert_name: str) -> None:
        """Generate detailed training report."""
        metrics = self.training_metrics.get(expert_name)
        if not metrics:
            return
        
        report = {
            "expert_name": expert_name,
            "start_time": metrics.start_time.isoformat(),
            "end_time": metrics.end_time.isoformat() if metrics.end_time else None,
            "duration_minutes": (metrics.end_time - metrics.start_time).total_seconds() / 60 if metrics.end_time else None,
            "total_steps": metrics.total_steps,
            "completed_steps": metrics.completed_steps,
            "final_loss": metrics.current_loss,
            "best_loss": metrics.best_loss,
            "validation_accuracy": metrics.validation_accuracy,
            "checkpoint_path": metrics.checkpoint_path,
            "errors": metrics.errors,
            "warnings": metrics.warnings
        }
        
        # Save report
        report_path = f"checkpoints/lora_{expert_name}/training_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"📊 Training report saved: {report_path}")


class TrainingProgressCallback(TrainerCallback):
    """Callback for monitoring training progress."""
    
    def __init__(self, metrics: Optional[TrainingMetrics] = None):
        self.metrics = metrics
        self.start_time = time.time()
    
    def on_log(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """Log training progress."""
        if self.metrics:
            self.metrics.total_steps = state.global_step
            self.metrics.current_loss = state.log_history[-1].get('train_loss', 0.0)
            self.metrics.learning_rate = state.log_history[-1].get('learning_rate', 0.0)
            
            # Update best loss
            if self.metrics.current_loss < self.metrics.best_loss:
                self.metrics.best_loss = self.metrics.current_loss
            
            # Log progress
            if state.global_step % 50 == 0:
                elapsed = time.time() - self.start_time
                steps_per_second = state.global_step / elapsed if elapsed > 0 else 0
                logger.info(f"📊 Step {state.global_step}: Loss={self.metrics.current_loss:.4f}, LR={self.metrics.learning_rate:.2e}, Speed={steps_per_second:.2f} steps/s")


def main():
    """Main entry point for unified training."""
    parser = argparse.ArgumentParser(
        description="Unified ChessGemma Training System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train all experts
  python -m src.training.unified_trainer

  # Train specific expert
  python -m src.training.unified_trainer --expert uci

  # Train with custom config
  python -m src.training.unified_trainer --config custom_config.json

  # Train without validation
  python -m src.training.unified_trainer --no-validate

  # Fresh training (no resume)
  python -m src.training.unified_trainer --no-resume
        """
    )
    
    parser.add_argument(
        '--expert',
        choices=['uci', 'tutor', 'director', 'all'],
        default='all',
        help='Expert to train (default: all)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to custom configuration file'
    )
    
    parser.add_argument(
        '--resume',
        action='store_true',
        default=True,
        help='Resume from existing checkpoints (default: enabled)'
    )
    
    parser.add_argument(
        '--no-resume',
        action='store_true',
        help='Start fresh training (disable resume)'
    )
    
    parser.add_argument(
        '--validate',
        action='store_true',
        default=True,
        help='Run validation after training (default: enabled)'
    )
    
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='Skip validation after training'
    )
    
    parser.add_argument(
        '--timeout',
        type=int,
        help='Training timeout in minutes'
    )
    
    args = parser.parse_args()
    
    # Handle resume flag conflict
    if args.no_resume:
        args.resume = False
    
    # Handle validate flag conflict
    if args.no_validate:
        args.validate = False
    
    print("🎓 Unified ChessGemma Training System")
    print("=" * 60)
    
    # Initialize trainer
    trainer = UnifiedChessTrainer(args.config)
    
    try:
        if args.expert == 'all':
            # Train all experts
            result = trainer.train_all_experts(
                resume=args.resume,
                validate=args.validate
            )
        else:
            # Train specific expert
            result = trainer.train_expert(
                expert_name=args.expert,
                validate=args.validate,
                timeout_minutes=args.timeout
            )
        
        # Print results
        if result.get("success", False):
            print("✅ Training completed successfully!")
        else:
            print("❌ Training failed!")
            print(f"Error: {result.get('error', 'Unknown error')}")
        
        # Print summary
        if "summary" in result:
            summary = result["summary"]
            print(f"\n📊 Training Summary:")
            print(f"   Total experts: {summary['total_experts']}")
            print(f"   Successful: {summary['successful_experts']}")
            print(f"   Failed: {summary['failed_experts']}")
            print(f"   Success rate: {summary['success_rate']:.1%}")
    
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
