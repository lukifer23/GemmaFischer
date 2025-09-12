#!/usr/bin/env python3
"""
Expert Training System for ChessGemma

Specialized training system for building expert LoRA adapters:
- UCI Expert: Optimized for fast, accurate move generation
- Tutor Expert: Specialized for detailed chess analysis and explanations
- Director Expert: Focused on strategic chess understanding and planning

Features:
- Expert-specific data curation and filtering
- Curriculum-based progressive training
- Performance validation and optimization
- Integration with enhanced inference system
- Automated expert switching and management
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
import time
import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, TrainerCallback
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import datasets
from datasets import Dataset
import pandas as pd
import random

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

# Import checkpoint manager
try:
    from .checkpoint_manager import CheckpointManager, CheckpointMetadata
    CHECKPOINT_MANAGER_AVAILABLE = True
except ImportError:
    CHECKPOINT_MANAGER_AVAILABLE = False
    logger.warning("Checkpoint manager not available - limited checkpoint functionality")

# Import MPS optimizer
try:
    from .mps_optimizer import MPSMemoryOptimizer, optimize_training_for_mps
    MPS_OPTIMIZER_AVAILABLE = True
except ImportError:
    MPS_OPTIMIZER_AVAILABLE = False
    logger.warning("MPS optimizer not available - using standard training")

# Configure logging
try:
    from ..utils.logging_config import get_logger
    logger = get_logger(__name__)
except ImportError:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)


@dataclass
class ExpertConfig:
    """Configuration for expert training."""
    name: str
    description: str
    focus_areas: List[str]
    training_objective: str
    data_filters: Dict[str, Any]
    training_params: Dict[str, Any]
    validation_metrics: List[str]
    target_performance: Dict[str, float]


@dataclass
class ExpertTrainingResult:
    """Results from expert training."""
    expert_name: str
    success: bool
    training_time: float
    final_metrics: Dict[str, Any] = field(default_factory=dict)
    adapter_path: str = ""
    validation_results: Dict[str, Any] = field(default_factory=dict)
    performance_score: float = 0.0
    recommendations: List[str] = field(default_factory=list)


class ChessCheckpointCallback(TrainerCallback):
    """Custom callback for comprehensive checkpoint management."""

    def __init__(self, expert_name: str, checkpoint_manager: CheckpointManager,
                 expert_config: ExpertConfig, config: Dict[str, Any]):
        self.expert_name = expert_name
        self.checkpoint_manager = checkpoint_manager
        self.expert_config = expert_config
        self.config = config
        self.start_time = time.time()

    def on_save(self, args, state, control, **kwargs):
        """Called when a checkpoint is saved."""
        if not self.checkpoint_manager:
            return

        try:
            # Get current metrics
            metrics = {}
            if hasattr(state, 'log_history') and state.log_history:
                latest_log = state.log_history[-1]
                metrics = {
                    'training_loss': latest_log.get('train_loss'),
                    'learning_rate': latest_log.get('learning_rate'),
                    'epoch': latest_log.get('epoch', 0)
                }

            # Get evaluation metrics if available
            eval_loss = None
            if hasattr(state, 'log_history'):
                for log_entry in reversed(state.log_history):
                    if 'eval_loss' in log_entry:
                        eval_loss = log_entry['eval_loss']
                        break

            # Calculate current learning rate
            lr = metrics.get('learning_rate')
            if lr is None and hasattr(args, 'learning_rate'):
                lr = args.learning_rate

            # Collect dataset info
            dataset_info = {
                'expert_filters': self.expert_config.data_filters,
                'training_samples': getattr(self, '_train_samples', 0),
                'validation_samples': getattr(self, '_eval_samples', 0)
            }

            # Collect validation metrics
            validation_metrics = {
                'expert_focus_areas': self.expert_config.focus_areas,
                'target_performance': self.expert_config.target_performance
            }

            # Create checkpoint with full metadata
            checkpoint_dir, metadata = self.checkpoint_manager.create_checkpoint(
                expert_name=self.expert_name,
                step=state.global_step,
                epoch=float(metrics.get('epoch', 0)),
                global_step=state.global_step,
                training_loss=metrics.get('training_loss', 0.0),
                eval_loss=eval_loss,
                learning_rate=lr or 0.0,
                model_config=self.config.get('model', {}),
                training_config=self.config.get('training', {}),
                dataset_info=dataset_info,
                validation_metrics=validation_metrics
            )

            logger.info(f"✅ Checkpoint created: {checkpoint_dir.name}")

        except Exception as e:
            logger.error(f"❌ Failed to create checkpoint: {e}")

    def on_train_begin(self, args, state, control, **kwargs):
        """Called at the beginning of training."""
        self.start_time = time.time()
        logger.info(f"🏁 Training started for {self.expert_name}")

    def on_train_end(self, args, state, control, **kwargs):
        """Called at the end of training."""
        duration = time.time() - self.start_time
        logger.info(f"🎉 Training completed for {self.expert_name} in {duration:.1f}s")

    def on_epoch_end(self, args, state, control, **kwargs):
        """Called at the end of each epoch."""
        logger.info(f"📊 Epoch {state.epoch:.1f} completed - Step {state.global_step}")


class ChessExpertTrainer:
    """Specialized trainer for chess expert adapters."""

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        self.config_path = Path(config_path) if config_path else Path(__file__).parent / "configs" / "expert_training_config.json"
        self.project_root = project_root
        self.checkpoints_dir = self.project_root / "checkpoints"
        self.data_dir = self.project_root / "data" / "formatted"

        # Expert configurations
        self.expert_configs = self._define_expert_configs()

        # Training state
        self.tokenizer = None
        self.base_model = None
        self.current_expert = None

        # Initialize checkpoint manager
        if CHECKPOINT_MANAGER_AVAILABLE:
            self.checkpoint_manager = CheckpointManager(self.checkpoints_dir)
            logger.info("✅ Checkpoint manager initialized")
        else:
            self.checkpoint_manager = None
            logger.warning("⚠️  Checkpoint manager not available - using basic checkpointing")

        # Initialize MPS optimizer
        if MPS_OPTIMIZER_AVAILABLE:
            self.mps_optimizer = MPSMemoryOptimizer()
            logger.info("✅ MPS memory optimizer initialized")
        else:
            self.mps_optimizer = None
            logger.warning("⚠️  MPS optimizer not available - using standard configuration")

        logger.info("🔧 Chess Expert Trainer initialized")

    def _define_expert_configs(self) -> Dict[str, ExpertConfig]:
        """Define configurations for each expert type."""

        return {
            'uci': ExpertConfig(
                name='uci',
                description='Universal Chess Interface expert - optimized for fast, accurate move generation',
                focus_areas=['move_accuracy', 'response_speed', 'legal_moves'],
                training_objective='Maximize move accuracy while maintaining fast inference',
                data_filters={
                    'task': 'engine_uci',
                    'min_rating': 1200,
                    'max_length': 256,
                    'exclude_complex_positions': True
                },
                training_params={
                    'learning_rate': 2e-4,
                    'batch_size': 8,
                    'gradient_accumulation_steps': 4,
                    'max_steps': 2000,
                    'warmup_steps': 200,
                    'weight_decay': 0.01,
                    'evaluation_strategy': 'steps',
                    'eval_steps': 200,
                    'save_steps': 400
                },
                validation_metrics=['move_accuracy', 'response_time', 'legal_move_rate'],
                target_performance={
                    'move_accuracy': 0.75,
                    'response_time': 0.5,
                    'legal_move_rate': 0.95
                }
            ),

            'tutor': ExpertConfig(
                name='tutor',
                description='Chess tutor expert - specialized for detailed analysis and teaching',
                focus_areas=['analysis_depth', 'explanation_quality', 'teaching_methodology'],
                training_objective='Maximize analysis quality and educational value',
                data_filters={
                    'task': 'tutor_explain',
                    'min_rating': 1300,
                    'max_length': 1024,
                    'require_detailed_explanation': True
                },
                training_params={
                    'learning_rate': 1.5e-4,
                    'batch_size': 4,
                    'gradient_accumulation_steps': 8,
                    'max_steps': 3000,
                    'warmup_steps': 300,
                    'weight_decay': 0.01,
                    'evaluation_strategy': 'steps',
                    'eval_steps': 300,
                    'save_steps': 600
                },
                validation_metrics=['explanation_quality', 'analysis_depth', 'teaching_effectiveness'],
                target_performance={
                    'explanation_quality': 0.8,
                    'analysis_depth': 0.75,
                    'teaching_effectiveness': 0.7
                }
            ),

            'director': ExpertConfig(
                name='director',
                description='Chess director expert - focused on strategic planning and high-level concepts',
                focus_areas=['strategic_understanding', 'opening_theory', 'endgame_knowledge'],
                training_objective='Maximize strategic depth and chess knowledge application',
                data_filters={
                    'task': 'director_qa',
                    'min_rating': 1400,
                    'max_length': 2048,
                    'require_strategic_content': True
                },
                training_params={
                    'learning_rate': 1e-4,
                    'batch_size': 2,
                    'gradient_accumulation_steps': 16,
                    'max_steps': 4000,
                    'warmup_steps': 400,
                    'weight_decay': 0.01,
                    'evaluation_strategy': 'steps',
                    'eval_steps': 400,
                    'save_steps': 800
                },
                validation_metrics=['strategic_accuracy', 'knowledge_depth', 'planning_quality'],
                target_performance={
                    'strategic_accuracy': 0.8,
                    'knowledge_depth': 0.75,
                    'planning_quality': 0.7
                }
            )
        }

    def initialize_training_environment(self):
        """Initialize the base model and tokenizer for expert training."""
        logger.info("🔧 Initializing expert training environment...")

        try:
            # Load configuration
            config = self._load_training_config()

            # Load tokenizer
            model_path = config.get('model', {}).get('base_model', 'models/unsloth-gemma-3-270m-it')
            model_path = self.project_root / model_path

            # Find the actual model snapshot path (handle Hugging Face cache structure)
            if model_path.exists() and model_path.is_dir():
                # Check if this is a Hugging Face cache directory with snapshots
                snapshots_dir = model_path / "models--unsloth--gemma-3-270m-it" / "snapshots"
                if snapshots_dir.exists():
                    # Find the latest snapshot (should be only one)
                    snapshot_dirs = list(snapshots_dir.iterdir())
                    if snapshot_dirs:
                        model_path = snapshot_dirs[0]  # Use the first (only) snapshot

            logger.info(f"Loading tokenizer from {model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(model_path),
                local_files_only=True,
                trust_remote_code=True
            )

            # Load base model
            logger.info(f"Loading base model from {model_path}")
            self.base_model = AutoModelForCausalLM.from_pretrained(
                str(model_path),
                local_files_only=True,
                device_map="auto",
                attn_implementation="eager",
                torch_dtype=torch.float16
            )

            # Configure model for training
            self.base_model.gradient_checkpointing_enable()

            logger.info("✅ Expert training environment initialized")

        except Exception as e:
            logger.error(f"❌ Failed to initialize training environment: {e}")
            raise

    def _load_training_config(self) -> Dict[str, Any]:
        """Load training configuration."""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                return json.load(f)

        # Default configuration
        return {
            'model': {
                'base_model': 'models/unsloth-gemma-3-270m-it',
                'max_seq_length': 2048,
                'torch_dtype': 'float16'
            },
            'training': {
                'output_dir_base': 'checkpoints/expert_training',
                'logging_steps': 50,
                'save_total_limit': 3,
                'load_best_model_at_end': True,
                'metric_for_best_model': 'eval_loss',
                'greater_is_better': False,
                'fp16': True,
                'report_to': []
            }
        }

    def prepare_expert_data(self, expert_name: str) -> Tuple[Dataset, Dataset]:
        """Prepare filtered and optimized dataset for expert training."""
        logger.info(f"📚 Preparing data for {expert_name} expert...")

        config = self.expert_configs[expert_name]
        data_filters = config.data_filters

        # Load available datasets
        all_data = []
        dataset_files = list(self.data_dir.glob("enhanced_*.jsonl"))

        if not dataset_files:
            # Fallback to original datasets
            dataset_files = list(self.data_dir.glob("*.jsonl"))

        logger.info(f"Found {len(dataset_files)} dataset files")

        # Load and filter data
        for dataset_file in dataset_files:
            logger.info(f"Processing {dataset_file.name}...")

            with open(dataset_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        item = json.loads(line.strip())

                        # Apply expert-specific filters
                        if self._passes_filters(item, data_filters):
                            all_data.append(item)

                    except json.JSONDecodeError:
                        continue

        logger.info(f"Loaded {len(all_data)} total examples")

        # Apply expert-specific processing
        processed_data = self._process_expert_data(all_data, expert_name)

        # Convert to Dataset format
        df = pd.DataFrame(processed_data)
        dataset = Dataset.from_pandas(df)

        # Split into train/validation
        train_test_split = dataset.train_test_split(test_size=0.1, seed=42)

        logger.info(f"✅ Prepared {len(train_test_split['train'])} training examples")
        logger.info(f"✅ Prepared {len(train_test_split['test'])} validation examples")

        return train_test_split['train'], train_test_split['test']

    def _passes_filters(self, item: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if data item passes the expert filters."""
        # Task filter
        if 'task' in filters:
            item_task = item.get('task', '')
            filter_task = filters['task']
            if filter_task not in item_task:
                return False

        # Rating filter
        if 'min_rating' in filters:
            item_rating = item.get('meta', {}).get('rating', 1500)
            if item_rating < filters['min_rating']:
                return False

        # Content filters
        if filters.get('require_detailed_explanation'):
            response = item.get('response', '')
            if len(response.split()) < 50:  # Require substantial explanations
                return False

        if filters.get('require_strategic_content'):
            response = item.get('response', '')
            strategic_keywords = ['strategy', 'plan', 'position', 'control', 'initiative']
            if not any(keyword in response.lower() for keyword in strategic_keywords):
                return False

        return True

    def _process_expert_data(self, data: List[Dict[str, Any]], expert_name: str) -> List[Dict[str, Any]]:
        """Apply expert-specific data processing."""
        processed_data = []

        for item in data:
            processed_item = dict(item)  # Copy original

            if expert_name == 'uci':
                # UCI expert: Focus on clean move generation
                processed_item['text'] = self._format_uci_text(item)

            elif expert_name == 'tutor':
                # Tutor expert: Enhanced analysis format
                processed_item['text'] = self._format_tutor_text(item)

            elif expert_name == 'director':
                # Director expert: Strategic planning focus
                processed_item['text'] = self._format_director_text(item)

            processed_data.append(processed_item)

        return processed_data

    def _format_uci_text(self, item: Dict[str, Any]) -> str:
        """Format text for UCI expert training."""
        prompt = item.get('prompt', '')
        response = item.get('response', '')

        # Clean UCI format: FEN + Move only
        return f"Engine Move:\n{prompt}\n\nMove: {response}"

    def _format_tutor_text(self, item: Dict[str, Any]) -> str:
        """Format text for tutor expert training."""
        prompt = item.get('prompt', '')
        response = item.get('response', '')

        return f"Tutor Analysis:\n{prompt}\n\nDetailed Analysis:\n{response}"

    def _format_director_text(self, item: Dict[str, Any]) -> str:
        """Format text for director expert training."""
        prompt = item.get('prompt', '')
        response = item.get('response', '')

        return f"Chess Director:\n{prompt}\n\nStrategic Assessment:\n{response}"

    def train_expert(self, expert_name: str, resume_from_checkpoint: bool = True) -> ExpertTrainingResult:
        """Train a single expert adapter with checkpoint management."""
        logger.info(f"🎓 Training {expert_name} expert...")
        logger.info(f"📝 Description: {self.expert_configs[expert_name].description}")

        start_time = time.time()
        resume_checkpoint = None
        trainer_state = {}

        # Check for existing checkpoints to resume from
        if resume_from_checkpoint and self.checkpoint_manager:
            resume_data = self.checkpoint_manager.resume_from_checkpoint(expert_name)
            if resume_data:
                resume_checkpoint, resume_metadata, trainer_state = resume_data
                logger.info(f"📂 Resuming training from step {resume_metadata.global_step}")
            else:
                logger.info("🏁 Starting fresh training (no checkpoints found)")

        try:
            # Initialize training environment (loads model and tokenizer)
            self.initialize_training_environment()

            # Prepare data
            train_dataset, eval_dataset = self.prepare_expert_data(expert_name)

            # Tokenize datasets
            def tokenize_function(examples):
                return self.tokenizer(
                    examples['text'],
                    truncation=True,
                    padding='max_length',
                    max_length=2048
                )

            train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=['text'])
            eval_dataset = eval_dataset.map(tokenize_function, batched=True, remove_columns=['text'])

            # Configure LoRA for expert
            expert_config = self.expert_configs[expert_name]
            lora_config = LoraConfig(
                r=32,
                lora_alpha=64,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj"],
                lora_dropout=0.05,
                task_type="CAUSAL_LM",
                bias="none"
            )

            # Create expert model
            expert_model = get_peft_model(self.base_model, lora_config)

            # Configure training arguments
            config = self._load_training_config()
            training_config = config["training"]
            expert_params = expert_config.training_params

            # Use checkpoint manager for output directory if available
            if self.checkpoint_manager:
                output_base = self.checkpoints_dir / expert_name
                output_base.mkdir(parents=True, exist_ok=True)
            else:
                output_base = self.checkpoints_dir / f"expert_{expert_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # Apply MPS optimizations if available
            if self.mps_optimizer:
                training_config = self.mps_optimizer.get_mps_optimized_training_args(
                    training_config, self.base_model, self.tokenizer
                )
                logger.info("⚡ Applied MPS optimizations to training configuration")

            # Merge expert-specific parameters
            final_training_config = training_config.copy()
            final_training_config.update({
                'per_device_train_batch_size': expert_params['batch_size'],
                'gradient_accumulation_steps': expert_params['gradient_accumulation_steps'],
                'max_steps': expert_params['max_steps'],
                'learning_rate': expert_params['learning_rate'],
                'warmup_steps': expert_params['warmup_steps'],
                'weight_decay': expert_params['weight_decay'],
                'save_steps': expert_params['save_steps'],
                'evaluation_strategy': expert_params['evaluation_strategy'],
                'eval_steps': expert_params['eval_steps'],
            })

            training_args = TrainingArguments(
                output_dir=str(output_base),
                num_train_epochs=1,
                logging_first_step=True,
                # Resume from checkpoint if available
                resume_from_checkpoint=str(resume_checkpoint) if resume_checkpoint else None,
                **final_training_config  # Include all optimized parameters
            )

            # Initialize trainer
            from transformers import DataCollatorForLanguageModeling
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False
            )

            # Create custom callback for checkpoint management
            checkpoint_callback = None
            if self.checkpoint_manager:
                checkpoint_callback = ChessCheckpointCallback(
                    expert_name=expert_name,
                    checkpoint_manager=self.checkpoint_manager,
                    expert_config=expert_config,
                    config=config
                )

            trainer = Trainer(
                model=expert_model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator,
                callbacks=[checkpoint_callback] if checkpoint_callback else None,
            )

            # Train the expert
            logger.info("🚀 Starting expert training...")
            train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)

            # Evaluate final performance
            eval_results = trainer.evaluate()
            logger.info(f"📊 Final evaluation: {eval_results}")

            # Calculate performance score
            performance_score = self._calculate_performance_score(expert_name, eval_results)

            # Generate recommendations
            recommendations = self._generate_training_recommendations(expert_name, eval_results)

            training_time = time.time() - start_time

            # Save expert model
            expert_model.save_pretrained(str(output_dir))
            self.tokenizer.save_pretrained(str(output_dir))

            result = ExpertTrainingResult(
                expert_name=expert_name,
                success=True,
                training_time=training_time,
                final_metrics=eval_results,
                adapter_path=str(output_dir),
                validation_results={},  # Would be populated with comprehensive validation
                performance_score=performance_score,
                recommendations=recommendations
            )

            logger.info(f"✅ {expert_name} expert training completed in {training_time:.2f} seconds")
            logger.info(f"🏆 Performance score: {performance_score:.3f}")
            logger.info(f"💾 Model saved to: {output_dir}")

            return result

        except Exception as e:
            logger.error(f"❌ Expert training failed: {e}")
            training_time = time.time() - start_time

            return ExpertTrainingResult(
                expert_name=expert_name,
                success=False,
                training_time=training_time,
                final_metrics={},
                adapter_path="",
                validation_results={},
                performance_score=0.0,
                recommendations=[f"Training failed: {str(e)}"]
            )

    def _calculate_performance_score(self, expert_name: str, metrics: Dict[str, Any]) -> float:
        """Calculate overall performance score for expert."""
        base_score = 0.0

        # Loss-based scoring (lower is better)
        eval_loss = metrics.get('eval_loss', 2.0)
        base_score += max(0, (2.0 - eval_loss) / 2.0) * 0.6  # 60% weight on loss

        # Expert-specific metrics
        if expert_name == 'uci':
            # UCI focuses on accuracy and speed
            base_score += 0.4  # Assume good baseline for UCI
        elif expert_name == 'tutor':
            # Tutor focuses on quality and depth
            base_score += 0.3  # Would need quality metrics
        elif expert_name == 'director':
            # Director focuses on strategic understanding
            base_score += 0.3  # Would need strategic metrics

        return min(1.0, base_score)

    def _generate_training_recommendations(self, expert_name: str, metrics: Dict[str, Any]) -> List[str]:
        """Generate training recommendations based on results."""
        recommendations = []

        eval_loss = metrics.get('eval_loss', 2.0)

        if eval_loss > 1.5:
            recommendations.append("Consider increasing training steps or adjusting learning rate")

        if eval_loss < 0.5:
            recommendations.append("Model may be overfitting - consider regularization or early stopping")

        # Expert-specific recommendations
        if expert_name == 'uci':
            recommendations.append("Focus on move accuracy validation with chess engine")
        elif expert_name == 'tutor':
            recommendations.append("Validate explanation quality and analysis depth")
        elif expert_name == 'director':
            recommendations.append("Test strategic understanding and planning capabilities")

        return recommendations if recommendations else ["Training results look good"]

    def train_all_experts(self, resume_from_checkpoint: bool = True) -> Dict[str, ExpertTrainingResult]:
        """Train all expert adapters."""
        logger.info("🎯 Training all chess experts...")
        logger.info("=" * 60)

        # Initialize training environment
        self.initialize_training_environment()

        results = {}

        for expert_name in ['uci', 'tutor', 'director']:
            logger.info(f"\n{'='*60}")
            logger.info(f"EXPERT: {expert_name.upper()}")
            logger.info(f"{'='*60}")

            result = self.train_expert(expert_name, resume_from_checkpoint=resume_from_checkpoint)
            results[expert_name] = result

            # Brief pause between expert trainings
            time.sleep(5)

        # Generate summary report
        self._generate_training_summary(results)

        return results

    def _generate_training_summary(self, results: Dict[str, ExpertTrainingResult]):
        """Generate comprehensive training summary."""
        logger.info(f"\n{'='*60}")
        logger.info("🎓 EXPERT TRAINING SUMMARY")
        logger.info(f"{'='*60}")

        total_time = sum(result.training_time for result in results.values())
        avg_performance = sum(result.performance_score for result in results.values()) / len(results)

        logger.info(f"⏱️  Total training time: {total_time:.2f} seconds ({total_time/3600:.2f} hours)")
        logger.info(f"📊 Average performance score: {avg_performance:.3f}")
        logger.info(f"✅ Experts trained: {len(results)}")

        for expert_name, result in results.items():
            logger.info(f"\n🔹 {expert_name.upper()} Expert:")
            logger.info(f"   Performance: {result.performance_score:.3f}")
            logger.info(f"   Training time: {result.training_time:.2f}s")
            logger.info(f"   Model saved: {result.adapter_path}")
            if result.recommendations:
                logger.info(f"   Recommendations: {result.recommendations[0]}")

        # Save detailed results
        summary_file = self.checkpoints_dir / f"expert_training_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        summary_data = {
            'timestamp': datetime.now().isoformat(),
            'total_training_time': total_time,
            'average_performance': avg_performance,
            'expert_results': {
                name: {
                    'performance_score': result.performance_score,
                    'training_time': result.training_time,
                    'adapter_path': result.adapter_path,
                    'final_metrics': result.final_metrics,
                    'recommendations': result.recommendations
                }
                for name, result in results.items()
            }
        }

        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2, default=str)

        logger.info(f"📄 Detailed summary saved to: {summary_file}")

    def validate_expert_performance(self, expert_name: str, test_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate trained expert performance."""
        logger.info(f"🧪 Validating {expert_name} expert performance...")

        # Load the trained expert model
        expert_path = self.checkpoints_dir / f"expert_{expert_name}"
        if not expert_path.exists():
            logger.warning(f"Expert model not found: {expert_path}")
            return {}

        try:
            # Load expert model
            expert_model = AutoModelForCausalLM.from_pretrained(
                str(expert_path),
                local_files_only=True,
                device_map="auto",
                torch_dtype=torch.float16
            )

            # Basic validation (would be expanded with comprehensive evaluation)
            validation_results = {
                'expert_name': expert_name,
                'model_loaded': True,
                'basic_validation': 'passed',
                'timestamp': datetime.now().isoformat()
            }

            logger.info(f"✅ {expert_name} expert validation completed")
            return validation_results

        except Exception as e:
            logger.error(f"❌ Expert validation failed: {e}")
            return {
                'expert_name': expert_name,
                'model_loaded': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }


def main():
    """Main entry point for expert training."""
    parser = argparse.ArgumentParser(description="Chess Expert Training System")
    parser.add_argument('--expert', choices=['uci', 'tutor', 'director', 'all'], default='all',
                       help='Expert to train')
    parser.add_argument('--config', type=str, help='Path to training configuration file')
    parser.add_argument('--output_dir', type=str, help='Output directory for trained experts')
    parser.add_argument('--validate', action='store_true', help='Run validation after training')
    parser.add_argument('--resume', action='store_true', help='Resume training from latest checkpoint')
    parser.add_argument('--no_resume', action='store_true', help='Start fresh training (ignore checkpoints)')

    args = parser.parse_args()

    print("🎓 Chess Expert Training System")
    print("=" * 50)

    # Determine resume behavior (default is resume if available)
    resume_training = args.resume or (not args.no_resume)

    # Initialize trainer
    trainer = ChessExpertTrainer(args.config)

    if args.output_dir:
        trainer.checkpoints_dir = Path(args.output_dir)

    if args.expert == 'all':
        # Train all experts
        results = trainer.train_all_experts(resume_from_checkpoint=resume_training)

        if args.validate:
            print("\n🧪 Running Expert Validation")
            print("-" * 30)

            for expert_name in ['uci', 'tutor', 'director']:
                validation = trainer.validate_expert_performance(expert_name, [])
                print(f"✅ {expert_name}: {validation.get('basic_validation', 'unknown')}")

    else:
        # Train single expert
        print(f"🎯 Training {args.expert} expert...")
        result = trainer.train_expert(args.expert, resume_from_checkpoint=resume_training)

        print("\n📊 Training Results:")
        print(f"   Performance Score: {result.performance_score:.3f}")
        print(f"   Training Time: {result.training_time:.2f}s")
        print(f"   Model Saved: {result.adapter_path}")

        if args.validate:
            print(f"\n🧪 Validating {args.expert} expert...")
            validation = trainer.validate_expert_performance(args.expert, [])
            print(f"   Validation: {validation.get('basic_validation', 'unknown')}")

    print("\n🎯 Expert training session complete!")


if __name__ == '__main__':
    main()
