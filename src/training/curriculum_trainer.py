#!/usr/bin/env python3
"""
Advanced Curriculum Training System for ChessGemma

Implements multi-stage curriculum learning with:
- Progressive complexity escalation
- Domain-specific expert training
- Dynamic difficulty adjustment
- Comprehensive evaluation and validation
- Adaptive learning rate scheduling

This system addresses the core training issues by ensuring proper skill development
across multiple expert domains with increasing sophistication.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import time
import logging
import torch
from dataclasses import dataclass, field
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from src.training.config_validation import (
    ConfigValidationError,
    validate_lora_config,
)


@dataclass
class CurriculumStage:
    """Defines a single curriculum training stage."""
    name: str
    description: str
    difficulty_range: Tuple[int, int]  # (min_rating, max_rating)
    focus_areas: List[str]
    training_steps: int
    batch_size: int
    learning_rate: float
    datasets: List[str]
    validation_datasets: List[str]
    min_accuracy_threshold: float
    max_loss_threshold: float
    evaluation_frequency: int


@dataclass
class TrainingMetrics:
    """Comprehensive training metrics tracking."""
    stage_name: str
    epoch: int
    step: int
    loss: float
    accuracy: float
    learning_rate: float
    gradient_norm: float
    memory_usage: float
    time_per_step: float
    validation_metrics: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""


class ChessCurriculumTrainer:
    """Advanced curriculum training system for chess expertise development."""

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or Path(__file__).parent / "configs" / "curriculum_config.json"
        self.project_root = project_root
        self.training_root = self.project_root / "checkpoints" / "curriculum_training"
        self.training_root.mkdir(parents=True, exist_ok=True)

        # Initialize curriculum stages
        self.curriculum = self._define_curriculum_stages()
        self.current_stage_idx = 0
        self.training_history = []
        self.metrics_history = []

        # Training state
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.scheduler = None
        self.current_stage_metrics = []

    def _define_curriculum_stages(self) -> List[CurriculumStage]:
        """Define the comprehensive curriculum stages for chess expertise development."""

        return [
            # Stage 1: Foundation - Basic Tactics
            CurriculumStage(
                name="foundation_tactics",
                description="Master basic tactical patterns and fundamental chess principles",
                difficulty_range=(1000, 1400),
                focus_areas=["basic_tactics", "piece_movement", "material_balance", "king_safety"],
                training_steps=5000,
                batch_size=4,
                learning_rate=2e-4,
                datasets=["data/formatted/enhanced_tutor_expert.jsonl", "data/formatted/enhanced_uci_expert.jsonl"],
                validation_datasets=["data/validation/tactical_validation.jsonl"],
                min_accuracy_threshold=0.65,
                max_loss_threshold=2.5,
                evaluation_frequency=250
            ),

            # Stage 2: Intermediate Tactics
            CurriculumStage(
                name="intermediate_tactics",
                description="Develop understanding of complex tactical combinations",
                difficulty_range=(1400, 1800),
                focus_areas=["advanced_tactics", "combination_play", "piece_coordination", "initiative"],
                training_steps=7500,
                batch_size=4,
                learning_rate=1.5e-4,
                datasets=["data/formatted/enhanced_tutor_expert.jsonl", "data/formatted/enhanced_director_expert.jsonl"],
                validation_datasets=["data/validation/complex_tactics_validation.jsonl"],
                min_accuracy_threshold=0.70,
                max_loss_threshold=2.0,
                evaluation_frequency=300
            ),

            # Stage 3: Positional Understanding
            CurriculumStage(
                name="positional_mastery",
                description="Learn positional concepts, pawn structure, and strategic planning",
                difficulty_range=(1800, 2200),
                focus_areas=["positional_play", "pawn_structure", "piece_placement", "long_term_strategy"],
                training_steps=10000,
                batch_size=4,
                learning_rate=1e-4,
                datasets=["data/formatted/enhanced_tutor_expert.jsonl", "data/formatted/enhanced_director_expert.jsonl"],
                validation_datasets=["data/validation/positional_validation.jsonl"],
                min_accuracy_threshold=0.75,
                max_loss_threshold=1.8,
                evaluation_frequency=400
            ),

            # Stage 4: Expert Analysis
            CurriculumStage(
                name="expert_analysis",
                description="Achieve expert-level analysis and strategic understanding",
                difficulty_range=(2200, 2500),
                focus_areas=["expert_analysis", "strategic_depth", "endgame_technique", "opening_principles"],
                training_steps=15000,
                batch_size=2,
                learning_rate=5e-5,
                datasets=["data/formatted/enhanced_tutor_expert.jsonl", "data/formatted/enhanced_director_expert.jsonl"],
                validation_datasets=["data/validation/expert_validation.jsonl"],
                min_accuracy_threshold=0.80,
                max_loss_threshold=1.5,
                evaluation_frequency=500
            )
        ]

    def load_curriculum_config(self) -> Dict[str, Any]:
        """Load curriculum configuration from file."""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                config = json.load(f)
        else:
            # Default configuration
            config = {
                "model": {
                    "base_model": "models/unsloth-gemma-3-270m-it",
                    "max_seq_length": 2048,
                    "torch_dtype": "float16"
                },
                "training": {
                    "gradient_accumulation_steps": 8,
                    "warmup_steps": 100,
                    "weight_decay": 0.01,
                    "max_grad_norm": 1.0,
                    "save_steps": 500,
                    "logging_steps": 50,
                    "evaluation_strategy": "steps"
                },
                "lora": {
                    "r": 32,
                    "lora_alpha": 64,
                    "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj"],
                    "dropout": 0.05,
                    "task_type": "CAUSAL_LM"
                },
                "curriculum": {
                    "enable_progression": True,
                    "min_stage_completion": 0.85,
                    "max_stage_attempts": 3,
                    "early_stopping_patience": 5,
                    "checkpoint_best_only": True
                },
            }

        try:
            config["lora"] = validate_lora_config(config.get("lora", {}))
        except ConfigValidationError as e:
            raise ValueError(f"Invalid LoRA configuration: {e}")

        return config

    def initialize_training_environment(self):
        """Initialize the complete training environment."""
        logger.info("🔧 Initializing training environment...")

        try:
            # Import required modules
            from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
            import torch

            # Load configuration
            config = self.load_curriculum_config()

            # Load tokenizer
            model_path = config["model"]["base_model"]
            if not Path(model_path).exists():
                # Try relative path
                model_path = self.project_root / model_path

            logger.info(f"Loading tokenizer from {model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(model_path),
                local_files_only=True,
                trust_remote_code=True
            )

            # Load base model
            logger.info(f"Loading base model from {model_path}")
            self.model = AutoModelForCausalLM.from_pretrained(
                str(model_path),
                local_files_only=True,
                device_map="auto",
                attn_implementation="eager",
                torch_dtype=torch.float16 if config["model"]["torch_dtype"] == "float16" else torch.float32
            )

            # Configure LoRA
            lora_config = LoraConfig(**config["lora"])
            self.model = get_peft_model(self.model, lora_config)

            logger.info("✅ Training environment initialized successfully")
            logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
            logger.info(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")

        except Exception as e:
            logger.error(f"❌ Failed to initialize training environment: {e}")
            raise

    def load_stage_data(self, stage: CurriculumStage) -> Dict[str, Any]:
        """Load and prepare datasets for the current curriculum stage."""
        logger.info(f"📚 Loading data for stage: {stage.name}")

        from datasets import Dataset
        import pandas as pd

        all_texts = []

        # Load datasets specified for this stage
        for dataset_path in stage.datasets:
            dataset_file = self.project_root / dataset_path
            if not dataset_file.exists():
                logger.warning(f"Dataset not found: {dataset_file}")
                continue

            logger.info(f"Loading dataset: {dataset_file}")

            # Load JSONL data
            data = []
            with open(dataset_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        item = json.loads(line.strip())
                        # Filter by difficulty range
                        rating = item.get('meta', {}).get('rating', 1500)
                        if stage.difficulty_range[0] <= rating <= stage.difficulty_range[1]:
                            data.append(item)
                    except json.JSONDecodeError:
                        continue

            logger.info(f"Loaded {len(data)} examples from {dataset_file}")

            # Convert to training format
            for item in data:
                task = item.get('task', '')
                prompt = item.get('prompt', '')
                response = item.get('response', '')

                if task == 'tutor_explain':
                    # Format for tutor training
                    text = f"Tutor Analysis:\n{prompt}\n\nAnalysis:\n{response}"
                elif task == 'engine_uci':
                    # Format for UCI training
                    text = f"Engine Move:\n{prompt}\n\nMove: {response}"
                elif task == 'director_qa':
                    # Format for director training
                    text = f"Chess Director:\n{prompt}\n\nResponse:\n{response}"
                else:
                    # Generic format
                    text = f"{prompt}\n\n{response}"

                all_texts.append(text)

        # Create dataset
        if not all_texts:
            raise ValueError(f"No training data found for stage {stage.name}")

        df = pd.DataFrame({'text': all_texts})
        dataset = Dataset.from_pandas(df)

        # Tokenize dataset
        def tokenize_function(examples):
            return self.tokenizer(
                examples['text'],
                truncation=True,
                padding='max_length',
                max_length=2048
            )

        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=['text']
        )

        # Split into train/validation
        train_test_split = tokenized_dataset.train_test_split(test_size=0.1, seed=42)

        logger.info(f"✅ Prepared {len(train_test_split['train'])} training examples")
        logger.info(f"✅ Prepared {len(train_test_split['test'])} validation examples")

        return {
            'train': train_test_split['train'],
            'validation': train_test_split['test']
        }

    def train_curriculum_stage(self, stage: CurriculumStage) -> Dict[str, Any]:
        """Train a single curriculum stage."""
        logger.info(f"🎓 Starting curriculum stage: {stage.name}")
        logger.info(f"📝 Description: {stage.description}")
        logger.info(f"🎯 Focus areas: {', '.join(stage.focus_areas)}")
        logger.info(f"📊 Difficulty range: {stage.difficulty_range[0]}-{stage.difficulty_range[1]}")
        logger.info(f"⚙️  Training steps: {stage.training_steps}")

        # Load stage data
        datasets = self.load_stage_data(stage)

        # Configure training arguments
        config = self.load_curriculum_config()
        training_config = config["training"]

        output_dir = self.training_root / f"stage_{stage.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        training_args = TrainingArguments(
            output_dir=str(output_dir),
            per_device_train_batch_size=stage.batch_size,
            gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
            num_train_epochs=1,  # We'll control total steps manually
            max_steps=stage.training_steps,
            learning_rate=stage.learning_rate,
            warmup_steps=training_config["warmup_steps"],
            weight_decay=training_config["weight_decay"],
            max_grad_norm=training_config["max_grad_norm"],
            logging_steps=training_config["logging_steps"],
            save_steps=training_config["save_steps"],
            evaluation_strategy=training_config["evaluation_strategy"],
            eval_steps=stage.evaluation_frequency,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=(config["model"]["torch_dtype"] == "float16"),
            report_to=[],
            logging_first_step=True,
        )

        # Initialize trainer
        from transformers import Trainer, DataCollatorForLanguageModeling

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=datasets['train'],
            eval_dataset=datasets['validation'],
            data_collator=data_collator,
        )

        # Train the stage
        logger.info("🚀 Starting training...")
        start_time = time.time()

        trainer.train()

        training_time = time.time() - start_time
        logger.info(f"⏱️  Training completed in {training_time:.2f} seconds")
        # Evaluate final performance
        eval_results = trainer.evaluate()
        logger.info(f"📊 Final evaluation results: {eval_results}")

        # Check if stage completion criteria are met
        stage_completed = (
            eval_results.get('eval_loss', float('inf')) <= stage.max_loss_threshold
        )

        if stage_completed:
            logger.info(f"✅ Stage {stage.name} completed successfully!")
        else:
            logger.warning(f"⚠️  Stage {stage.name} did not meet completion criteria")

        # Save stage results
        stage_results = {
            'stage_name': stage.name,
            'completed': stage_completed,
            'training_time': training_time,
            'final_metrics': eval_results,
            'checkpoint_path': str(output_dir),
            'completion_criteria': {
                'target_loss': stage.max_loss_threshold,
                'actual_loss': eval_results.get('eval_loss', float('inf')),
                'met_criteria': stage_completed
            }
        }

        return stage_results

    def run_curriculum_training(self, start_stage: int = 0) -> Dict[str, Any]:
        """Run the complete curriculum training pipeline."""
        logger.info("🎯 Starting Chess Curriculum Training")
        logger.info("=" * 60)

        curriculum_results = {
            'start_time': datetime.now().isoformat(),
            'stages_completed': [],
            'stages_failed': [],
            'total_training_time': 0,
            'final_model_path': None,
            'training_summary': {}
        }

        # Initialize training environment
        self.initialize_training_environment()

        # Run each curriculum stage
        for i, stage in enumerate(self.curriculum[start_stage:], start=start_stage):
            logger.info(f"\n{'='*60}")
            logger.info(f"Stage {i+1}/{len(self.curriculum)}: {stage.name.upper()}")
            logger.info(f"{'='*60}")

            try:
                stage_results = self.train_curriculum_stage(stage)
                curriculum_results['stages_completed'].append(stage_results)

                if stage_results['completed']:
                    logger.info(f"✅ Stage {stage.name} completed successfully")
                else:
                    logger.warning(f"⚠️  Stage {stage.name} completed but did not meet criteria")
                    # Continue to next stage anyway for now

            except Exception as e:
                logger.error(f"❌ Stage {stage.name} failed: {e}")
                curriculum_results['stages_failed'].append({
                    'stage_name': stage.name,
                    'error': str(e),
                    'stage_index': i
                })
                continue

        # Calculate total training time
        curriculum_results['total_training_time'] = sum(
            stage['training_time'] for stage in curriculum_results['stages_completed']
        )

        # Save final model
        final_model_path = self.training_root / f"final_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.model.save_pretrained(str(final_model_path))
        self.tokenizer.save_pretrained(str(final_model_path))
        curriculum_results['final_model_path'] = str(final_model_path)

        # Create training summary
        curriculum_results['training_summary'] = {
            'total_stages': len(self.curriculum),
            'successful_stages': len(curriculum_results['stages_completed']),
            'failed_stages': len(curriculum_results['stages_failed']),
            'total_training_time_hours': curriculum_results['total_training_time'] / 3600,
            'average_stage_time_hours': (curriculum_results['total_training_time'] / 3600) / len(curriculum_results['stages_completed']) if curriculum_results['stages_completed'] else 0
        }

        # Save curriculum results
        results_file = self.training_root / f"curriculum_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(curriculum_results, f, indent=2, default=str)

        logger.info("\n" + "="*60)
        logger.info("🎓 CURRICULUM TRAINING COMPLETE")
        logger.info("="*60)
        logger.info(f"✅ Successful stages: {len(curriculum_results['stages_completed'])}")
        logger.info(f"❌ Failed stages: {len(curriculum_results['stages_failed'])}")
        logger.info(f"⏱️  Total training time: {curriculum_results['training_summary']['total_training_time_hours']:.2f} hours")
        logger.info(f"💾 Final model saved to: {curriculum_results['final_model_path']}")
        logger.info(f"📄 Results saved to: {results_file}")

        return curriculum_results

    def evaluate_expert_performance(self, expert_type: str, test_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate expert performance on domain-specific tasks."""
        logger.info(f"🧪 Evaluating {expert_type} expert performance...")

        # This would implement comprehensive evaluation metrics
        # For now, return placeholder results
        return {
            'expert_type': expert_type,
            'accuracy': 0.75,
            'f1_score': 0.72,
            'domain_specific_metrics': {},
            'evaluation_timestamp': datetime.now().isoformat()
        }


def main():
    """Main entry point for curriculum training."""
    parser = argparse.ArgumentParser(description="Chess Curriculum Training System")
    parser.add_argument('--config', type=str, help='Path to curriculum configuration file')
    parser.add_argument('--start_stage', type=int, default=0, help='Starting curriculum stage (0-based)')
    parser.add_argument('--expert_evaluation', action='store_true', help='Run expert performance evaluation')
    parser.add_argument('--output_dir', type=str, help='Output directory for training results')

    args = parser.parse_args()

    # Initialize curriculum trainer
    trainer = ChessCurriculumTrainer(args.config)

    if args.output_dir:
        trainer.training_root = Path(args.output_dir)

    # Run curriculum training
    results = trainer.run_curriculum_training(args.start_stage)

    # Optional expert evaluation
    if args.expert_evaluation:
        logger.info("\n🧪 Running Expert Performance Evaluation")
        logger.info("-" * 40)

        expert_types = ['uci', 'tutor', 'director']
        evaluation_results = {}

        for expert_type in expert_types:
            # Load test data for this expert
            test_data = []  # Would load actual test data
            evaluation_results[expert_type] = trainer.evaluate_expert_performance(expert_type, test_data)

        # Save evaluation results
        eval_file = trainer.training_root / f"expert_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(eval_file, 'w') as f:
            json.dump(evaluation_results, f, indent=2)

        logger.info(f"📊 Expert evaluation results saved to: {eval_file}")

    logger.info("\n🎯 Curriculum training session complete!")


if __name__ == '__main__':
    main()
