#!/usr/bin/env python3
"""
MPS Memory Optimization for ChessGemma Training

Advanced memory management and performance optimizations for Apple Silicon MPS:
- Dynamic batch sizing based on available memory
- Gradient accumulation optimization
- Memory-efficient attention mechanisms
- MPS-specific data loading optimizations
- Automatic memory monitoring and adjustment
"""

from __future__ import annotations

import os
import torch
import psutil
import gc
from typing import Dict, Any, Optional, Tuple, List
from contextlib import contextmanager
import logging

# Add project root to path
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

try:
    from ..utils.logging_config import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class MPSMemoryOptimizer:
    """Memory optimizer for Apple Silicon MPS training."""

    def __init__(self, target_memory_usage: float = 0.85):
        """
        Initialize MPS memory optimizer.

        Args:
            target_memory_usage: Target memory usage (0.0-1.0)
        """
        self.target_memory_usage = target_memory_usage
        self.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        self.is_mps = self.device.type == "mps"

        # System memory info
        self.system_memory = psutil.virtual_memory().total
        self.available_memory = self._get_available_memory()

        # MPS memory tracking
        self.peak_memory_used = 0
        self.current_memory_used = 0

        logger.info(f"🔧 MPS Memory Optimizer initialized")
        logger.info(f"   Device: {self.device}")
        logger.info(f"   System Memory: {self.system_memory / (1024**3):.1f}GB")
        logger.info(f"   Available Memory: {self.available_memory / (1024**3):.1f}GB")
        logger.info(f"   Target Usage: {self.target_memory_usage:.1%}")

    def _get_available_memory(self) -> int:
        """Get available memory for training."""
        if self.is_mps:
            # MPS has access to unified memory
            # Use 80% of available system memory as a conservative estimate
            available = int(psutil.virtual_memory().available * 0.8)
        else:
            # CPU training - use 60% of available memory
            available = int(psutil.virtual_memory().available * 0.6)

        return available

    def calculate_optimal_batch_size(self, model, tokenizer, sequence_length: int = 2048,
                                   safety_margin: float = 0.15) -> Dict[str, Any]:
        """
        Calculate optimal batch size for given model and sequence length.
        Uses adaptive memory profiling with MPS-specific optimizations.

        Args:
            model: The model to profile
            tokenizer: Tokenizer for encoding
            sequence_length: Maximum sequence length
            safety_margin: Additional safety margin for memory calculations

        Returns:
            Dict with batch size recommendations and memory estimates
        """

        logger.info("🔧 Calculating optimal MPS batch size with memory profiling")

        # More aggressive but safe settings for MPS
        if self.is_mps:
            # For Gemma 3 270M model on MPS with 18GB system memory
            # Use memory profiling for accurate batch sizing
            recommended_batch_size = self._profile_mps_batch_size(model, tokenizer, sequence_length, safety_margin)
            gradient_accumulation_steps = max(2, min(8, int(16 / max(recommended_batch_size, 1))))  # Adaptive accumulation
            effective_batch_size = recommended_batch_size * gradient_accumulation_steps

            # More accurate memory estimation based on profiling
            estimated_memory_per_sample = self._estimate_mps_memory_per_sample(model, sequence_length)

        else:
            # CPU settings (more liberal)
            recommended_batch_size = 4
            gradient_accumulation_steps = 8
            effective_batch_size = 32
            estimated_memory_per_sample = int(150 * 1024 * 1024)  # 150MB per sample estimate

        result = {
            'recommended_batch_size': recommended_batch_size,
            'gradient_accumulation_steps': gradient_accumulation_steps,
            'effective_batch_size': effective_batch_size,
            'estimated_memory_per_sample': estimated_memory_per_sample,
            'available_memory': int(self.available_memory * 0.8),  # Use 80% for better utilization
            'memory_utilization': self.target_memory_usage,
            'device': self.device.type
        }

        logger.info("📊 Optimized batch size settings:")
        logger.info(f"   Recommended batch size: {recommended_batch_size}")
        logger.info(f"   Gradient accumulation: {gradient_accumulation_steps}")
        logger.info(f"   Effective batch size: {result['effective_batch_size']}")
        logger.info(f"   Memory per sample: {estimated_memory_per_sample / (1024*1024):.1f}MB")

        return result

    def _profile_mps_batch_size(self, model, tokenizer, sequence_length: int, safety_margin: float) -> int:
        """Profile memory usage to determine optimal batch size for MPS."""
        if not self.is_mps:
            return 2

        # Start with conservative batch size and work up
        test_batch_sizes = [1, 2, 3, 4]
        optimal_batch_size = 1

        for batch_size in test_batch_sizes:
            try:
                # Test memory usage with this batch size
                test_memory = self._test_batch_memory_usage(model, tokenizer, sequence_length, batch_size)

                # If memory usage is acceptable, this could be our batch size
                memory_utilization = test_memory / self.available_memory
                if memory_utilization < (self.target_memory_usage - safety_margin):
                    optimal_batch_size = batch_size
                else:
                    # If this batch size exceeds limits, the previous one was optimal
                    break

            except Exception as e:
                logger.warning(f"Batch size profiling failed for size {batch_size}: {e}")
                break

        logger.info(f"📏 Profiled optimal batch size: {optimal_batch_size}")
        return optimal_batch_size

    def _test_batch_memory_usage(self, model, tokenizer, sequence_length: int, batch_size: int) -> int:
        """Test memory usage for a specific batch size."""
        if not self.is_mps:
            return 0

        # Clear MPS cache before testing
        torch.mps.empty_cache()

        try:
            # Create a small test batch
            test_text = "This is a test input for memory profiling." * 10
            inputs = tokenizer(test_text, return_tensors="pt", max_length=sequence_length, truncation=True)

            # Replicate to batch size
            input_ids = inputs['input_ids'].repeat(batch_size, 1)
            attention_mask = inputs['attention_mask'].repeat(batch_size, 1)

            # Move to MPS
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)

            # Force MPS to allocate memory
            with torch.no_grad():
                _ = model(input_ids, attention_mask=attention_mask)

            # Get memory usage
            if hasattr(torch.mps, 'current_allocated_memory'):
                memory_used = torch.mps.current_allocated_memory()
            else:
                # Fallback estimation
                memory_used = batch_size * 50 * 1024 * 1024  # 50MB per sample estimate

            return memory_used

        except Exception as e:
            logger.warning(f"Memory test failed: {e}")
            return batch_size * 100 * 1024 * 1024  # Conservative estimate
        finally:
            # Clean up
            torch.mps.empty_cache()

    def _estimate_mps_memory_per_sample(self, model, sequence_length: int) -> int:
        """Estimate memory usage per sample for MPS."""
        if not self.is_mps:
            return 100 * 1024 * 1024

        # Base estimates for Gemma 3 270M
        model_params = sum(p.numel() for p in model.parameters()) if model else 270_000_000
        param_memory = model_params * 4  # FP32

        # Activation memory (rough estimate)
        activation_memory = sequence_length * 4096 * 4 * 32  # Conservative estimate

        # Optimizer states (AdamW)
        optimizer_memory = model_params * 8  # AdamW states

        total_per_sample = param_memory + activation_memory + optimizer_memory

        logger.info(f"💾 Memory estimate per sample: {total_per_sample / (1024*1024):.1f}MB")
        logger.info(f"   Model params: {param_memory / (1024*1024):.1f}MB")
        logger.info(f"   Activations: {activation_memory / (1024*1024):.1f}MB")
        logger.info(f"   Optimizer: {optimizer_memory / (1024*1024):.1f}MB")

        return int(total_per_sample)

    def get_mps_optimized_training_args(self, base_config: Dict[str, Any],
                                      model=None, tokenizer=None) -> Dict[str, Any]:
        """
        Get MPS-optimized training arguments with improved stability.

        Args:
            base_config: Base training configuration
            model: Model for memory profiling (optional)
            tokenizer: Tokenizer for memory profiling (optional)

        Returns:
            Optimized training arguments
        """

        # Start with base configuration
        optimized_config = base_config.copy()

        # MPS-specific optimizations with improved stability
        if self.is_mps:
            # Get optimal batch sizing if model/tokenizer available
            batch_config = {}
            if model and tokenizer:
                try:
                    batch_recommendations = self.calculate_optimal_batch_size(
                        model, tokenizer, sequence_length=base_config.get('max_seq_length', 2048)
                    )
                    batch_config = {
                        'per_device_train_batch_size': batch_recommendations['recommended_batch_size'],
                        'gradient_accumulation_steps': batch_recommendations['gradient_accumulation_steps'],
                    }
                    logger.info(f"📊 Applied profiled batch config: {batch_config}")
                except Exception as e:
                    logger.warning(f"Batch profiling failed, using defaults: {e}")
                    batch_config = {
                        'per_device_train_batch_size': 2,  # Safe fallback
                        'gradient_accumulation_steps': 4,   # Effective batch size of 8
                    }
            else:
                # Use conservative defaults when profiling not available
                batch_config = {
                    'per_device_train_batch_size': 2,
                    'gradient_accumulation_steps': 4,
                }

            optimized_config.update({
                # MPS doesn't support bf16/fp16 mixed precision - use fp32
                'bf16': False,
                'fp16': False,
                'optim': 'adamw_torch',  # MPS-optimized optimizer
                'gradient_checkpointing': True,  # ENABLED: Critical for memory efficiency on MPS
                **batch_config
            })

        # Memory monitoring with more frequent logging
        optimized_config.update({
            'logging_steps': min(optimized_config.get('logging_steps', 50), 5),  # More frequent logging (every 5 steps)
            'save_steps': optimized_config.get('save_steps', 500),
            'save_total_limit': 3,  # Keep only last 3 checkpoints
            'logging_first_step': True,  # Ensure first step is logged
        })

        # Only add evaluation-related parameters if evaluation is enabled
        eval_strategy = optimized_config.get('eval_strategy')
        if eval_strategy and eval_strategy != 'no':
            optimized_config.update({
                'eval_steps': optimized_config.get('save_steps', 500),  # Evaluate at save points
                'load_best_model_at_end': True,
                'metric_for_best_model': 'eval_loss',
                'greater_is_better': False,
            })

        # Learning rate adjustments for MPS stability
        if self.is_mps:
            current_lr = optimized_config.get('learning_rate', 2e-4)
            # Conservative learning rate for stability
            optimized_config['learning_rate'] = min(current_lr, 1e-4)

            # Add warmup and cosine annealing for better convergence
            max_steps = optimized_config.get('max_steps', 1000)
            optimized_config['warmup_steps'] = max(10, int(max_steps * 0.1))  # 10% warmup
            optimized_config['lr_scheduler_type'] = 'cosine'

        # Timeout and stability settings - only add valid parameters
        optimized_config.update({
            'report_to': [],  # Disable external reporting for stability
        })

        # Note: remove_unused_columns and dataloader_timeout are handled in base config
        # to avoid conflicts with TrainingArguments validation

        logger.info("⚡ Enhanced MPS-optimized training configuration:")
        for key, value in optimized_config.items():
            if key in ['learning_rate', 'per_device_train_batch_size', 'gradient_accumulation_steps',
                      'bf16', 'fp16', 'gradient_checkpointing', 'warmup_steps']:
                logger.info(f"   {key}: {value}")

        return optimized_config

    def monitor_memory_usage(self) -> Dict[str, Any]:
        """Monitor current memory usage."""
        try:
            if self.is_mps:
                # MPS memory monitoring - use conservative estimates
                # Avoid using torch.mps memory functions that may not be available
                system_memory = psutil.virtual_memory()
                current_memory = system_memory.total - system_memory.available
                peak_memory = current_memory  # Approximation for MPS
            else:
                # CPU memory monitoring
                process = psutil.Process()
                current_memory = process.memory_info().rss
                peak_memory = current_memory  # Approximation

            memory_info = {
                'current_memory': current_memory,
                'peak_memory': peak_memory,
                'available_memory': self.available_memory,
                'memory_utilization': min(current_memory / self.available_memory, 1.0),
                'device': self.device.type
            }

            # Update tracking
            self.current_memory_used = current_memory
            self.peak_memory_used = max(self.peak_memory_used, peak_memory)

            return memory_info

        except Exception as e:
            logger.warning(f"Memory monitoring failed: {e}")
            return {
                'current_memory': 0,
                'peak_memory': 0,
                'available_memory': self.available_memory,
                'memory_utilization': 0.0,
                'device': self.device.type,
                'error': str(e)
            }

    @contextmanager
    def memory_efficient_context(self):
        """Context manager for memory-efficient operations."""
        if self.is_mps:
            torch.mps.empty_cache()

        try:
            yield
        finally:
            if self.is_mps:
                torch.mps.empty_cache()
            gc.collect()

    def optimize_model_for_mps(self, model):
        """Apply MPS-specific model optimizations with improved stability."""
        if not self.is_mps:
            return model

        try:
            # Safe gradient checkpointing for MPS
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()
                logger.info("✅ Gradient checkpointing enabled for memory efficiency")
        except Exception as e:
            logger.warning(f"⚠️  Gradient checkpointing failed: {e}")
            # Continue without gradient checkpointing

        # MPS-specific optimizations
        try:
            for module in model.modules():
                if hasattr(module, 'to'):
                    # Ensure all parameters are on MPS
                    module.to(self.device)
        except Exception as e:
            logger.warning(f"⚠️  Module device placement failed: {e}")

        # Additional MPS optimizations
        try:
            # Set model to evaluation mode initially to avoid issues
            model.eval()
            # Enable training mode when needed
            model.train()
        except Exception as e:
            logger.warning(f"⚠️  Model mode switching failed: {e}")

        logger.info("🔧 Model optimized for MPS training with enhanced stability")
        return model

    def apply_safe_gradient_checkpointing(self, model):
        """Apply gradient checkpointing safely for MPS."""
        if not self.is_mps:
            return False

        try:
            # Check if gradient checkpointing is already enabled
            if hasattr(model, 'is_gradient_checkpointing') and model.is_gradient_checkpointing:
                logger.info("ℹ️  Gradient checkpointing already enabled")
                return True

            # Enable gradient checkpointing with error handling
            model.gradient_checkpointing_enable()
            logger.info("✅ Gradient checkpointing enabled successfully")
            return True
        except Exception as e:
            logger.warning(f"⚠️  Failed to enable gradient checkpointing: {e}")
            return False

    def get_memory_optimization_tips(self) -> List[str]:
        """Get memory optimization tips for MPS training."""
        tips = [
            "Use gradient_checkpointing=True to reduce memory usage",
            "Enable fp16 training for better MPS performance",
            "Use gradient_accumulation_steps to achieve larger effective batch sizes",
            "Monitor memory usage with torch.mps.current_allocated_memory()",
            "Call torch.mps.empty_cache() periodically to free unused memory",
            "Use dataloader_num_workers=0 to avoid multiprocessing issues",
            "Consider using smaller sequence lengths for memory-constrained scenarios"
        ]

        if self.is_mps:
            tips.extend([
                "MPS benefits from unified memory - monitor system memory usage",
                "Use batch sizes that keep memory utilization below 85%",
                "Consider using torch.compile() for additional performance gains (PyTorch 2.0+)",
                "Profile memory usage with torch.mps.memory_stats() for detailed analysis"
            ])

        return tips


class MPSTrainingMonitor:
    """Real-time training monitor for MPS memory and performance."""

    def __init__(self, check_interval: int = 10):
        self.check_interval = check_interval
        self.step_count = 0
        self.last_memory_check = 0
        self.memory_warnings = 0
        self.max_memory_warnings = 5

    def check_training_health(self) -> Dict[str, Any]:
        """Check training health and return status."""
        if not torch.backends.mps.is_available():
            return {'status': 'ok', 'device': 'cpu'}

        try:
            current_memory = torch.mps.current_allocated_memory() / (1024**3)
            peak_memory = torch.mps.driver_allocated_memory() / (1024**3)

            # Check for memory issues
            warnings = []
            if current_memory > 14.0:  # Over 14GB
                warnings.append(f"High memory usage: {current_memory:.1f}GB")
                self.memory_warnings += 1
            elif current_memory > 12.0:  # Over 12GB
                warnings.append(f"Elevated memory usage: {current_memory:.1f}GB")

            # Check for memory leaks (peak much higher than current)
            if peak_memory > current_memory * 1.5 and peak_memory > 10.0:
                warnings.append(f"Potential memory leak detected: peak {peak_memory:.1f}GB")

            status = 'warning' if warnings else 'ok'

            return {
                'status': status,
                'device': 'mps',
                'current_memory_gb': current_memory,
                'peak_memory_gb': peak_memory,
                'warnings': warnings,
                'memory_warnings_count': self.memory_warnings
            }

        except Exception as e:
            return {
                'status': 'error',
                'device': 'mps',
                'error': str(e)
            }

    def should_clear_cache(self) -> bool:
        """Check if cache should be cleared based on memory usage."""
        if not torch.backends.mps.is_available():
            return False

        try:
            current_memory = torch.mps.current_allocated_memory() / (1024**3)
            return current_memory > 13.0  # Clear cache if over 13GB
        except:
            return False

    def log_training_status(self, step: int, loss: float = None):
        """Log training status periodically."""
        self.step_count += 1

        if self.step_count - self.last_memory_check >= self.check_interval:
            health = self.check_training_health()
            self.last_memory_check = self.step_count

            if health['status'] == 'warning':
                print(f"⚠️  Step {step}: {', '.join(health['warnings'])}")
                if self.should_clear_cache():
                    print("🧹 Clearing MPS cache to free memory...")
                    torch.mps.empty_cache()

            elif health['status'] == 'ok' and loss is not None:
                print(f"✅ Step {step}: Loss={loss:.4f}, Memory={health['current_memory_gb']:.1f}GB")


class MPSDataLoaderOptimizer:
    """DataLoader optimizations for MPS training."""

    def __init__(self):
        self.is_mps = torch.backends.mps.is_available()

    def get_optimized_dataloader_config(self) -> Dict[str, Any]:
        """Get MPS-optimized DataLoader configuration."""
        if self.is_mps:
            return {
                'pin_memory': False,  # MPS doesn't benefit from pinned memory
                'num_workers': 0,     # Avoid multiprocessing issues
                'persistent_workers': False,
                'prefetch_factor': None,
            }
        else:
            return {
                'pin_memory': True,
                'num_workers': min(4, os.cpu_count() or 1),
                'persistent_workers': True,
                'prefetch_factor': 2,
            }


def optimize_training_for_mps(training_config: Dict[str, Any],
                            model=None, tokenizer=None) -> Dict[str, Any]:
    """
    Convenience function to optimize training configuration for MPS.

    Args:
        training_config: Base training configuration
        model: Model for memory profiling (optional)
        tokenizer: Tokenizer for memory profiling (optional)

    Returns:
        Optimized training configuration
    """
    optimizer = MPSMemoryOptimizer()
    return optimizer.get_mps_optimized_training_args(training_config, model, tokenizer)


# Utility functions for MPS training
def get_mps_memory_stats() -> Dict[str, Any]:
    """Get detailed MPS memory statistics."""
    if not torch.backends.mps.is_available():
        return {'error': 'MPS not available'}

    try:
        current = torch.mps.current_allocated_memory()
        peak = torch.mps.driver_allocated_memory()  # Use driver_allocated as peak estimate

        return {
            'current_allocated': current,
            'peak_allocated': peak,
            'memory_stats': {'current_gb': current / (1024**3), 'peak_gb': peak / (1024**3)},
            'utilization': current / torch.mps.recommended_max_memory() if torch.mps.recommended_max_memory() > 0 else 0
        }
    except Exception as e:
        return {'error': str(e)}


def clear_mps_cache():
    """Clear MPS cache to free memory."""
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
        gc.collect()


def setup_mps_environment():
    """Setup environment variables for optimal MPS performance."""
    os.environ.setdefault('PYTORCH_MPS_HIGH_WATERMARK_RATIO', '0.0')  # Disable high watermark
    os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')  # Enable CPU fallback for unsupported ops

    if torch.backends.mps.is_available():
        logger.info("✅ MPS environment configured for optimal performance")
    else:
        logger.warning("⚠️  MPS not available, using CPU fallback")


# Initialize MPS environment on import
setup_mps_environment()
