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
                                   safety_margin: float = 0.1) -> Dict[str, Any]:
        """
        Calculate optimal batch size for given model and sequence length.
        Uses conservative, static estimation to avoid memory profiling issues.

        Args:
            model: The model to profile
            tokenizer: Tokenizer for encoding
            sequence_length: Maximum sequence length
            safety_margin: Additional safety margin for memory calculations

        Returns:
            Dict with batch size recommendations and memory estimates
        """

        logger.info("🔧 Using ultra-conservative MPS batch sizing (no memory profiling)")

        # Optimized settings for MPS performance with stability
        if self.is_mps:
            # For Gemma 3 270M model on MPS with 18GB system memory
            # Balance performance with stability - slightly increased batch size
            recommended_batch_size = 2  # Increased from 1 for better performance
            gradient_accumulation_steps = 4  # Moderate accumulation for effective batch size
            effective_batch_size = 8  # Better training dynamics

            # Estimate memory usage (rough approximation)
            # Model: ~270M params * 4 bytes (fp32) = ~1GB
            # LoRA: ~10MB
            # Optimizer states: ~2-3GB
            # Activations per sample: ~50MB
            estimated_memory_per_sample = int(50 * 1024 * 1024)  # 50MB per sample estimate

        else:
            # CPU settings (more liberal)
            recommended_batch_size = 2
            gradient_accumulation_steps = 8
            effective_batch_size = 16
            estimated_memory_per_sample = int(100 * 1024 * 1024)  # 100MB per sample estimate

        result = {
            'recommended_batch_size': recommended_batch_size,
            'gradient_accumulation_steps': gradient_accumulation_steps,
            'effective_batch_size': effective_batch_size,
            'estimated_memory_per_sample': estimated_memory_per_sample,
            'available_memory': int(self.available_memory * 0.7),  # Conservative 70% usage
            'memory_utilization': 0.3,  # Conservative target
            'device': self.device.type
        }

        logger.info("📊 Conservative batch size settings:")
        logger.info(f"   Recommended batch size: {recommended_batch_size}")
        logger.info(f"   Gradient accumulation: {gradient_accumulation_steps}")
        logger.info(f"   Effective batch size: {result['effective_batch_size']}")
        logger.info("   Memory profiling disabled for MPS stability")

        return result

    def get_mps_optimized_training_args(self, base_config: Dict[str, Any],
                                      model=None, tokenizer=None) -> Dict[str, Any]:
        """
        Get MPS-optimized training arguments.

        Args:
            base_config: Base training configuration
            model: Model for memory profiling (optional)
            tokenizer: Tokenizer for memory profiling (optional)

        Returns:
            Optimized training arguments
        """

        # Start with base configuration
        optimized_config = base_config.copy()

        # MPS-specific optimizations
        if self.is_mps:
            optimized_config.update({
                # MPS doesn't support bf16/fp16 mixed precision - use fp32
                'bf16': False,
                'fp16': False,
                'dataloader_pin_memory': False,  # MPS doesn't benefit from pinned memory
                'dataloader_num_workers': 0,  # Avoid multiprocessing issues on MPS
                'gradient_checkpointing': False,  # DISABLED: Causes buffer allocation issues on MPS
                'optim': 'adamw_torch',  # MPS-optimized optimizer
                # Basic MPS settings - let transformers handle optimization
                'per_device_train_batch_size': 1,  # Minimal batch size
                'gradient_accumulation_steps': 1   # No accumulation
            })

            # Memory-efficient attention if available
            try:
                from transformers.models.gemma.modeling_gemma import GemmaAttention
                # Enable flash attention if available (memory efficient)
                if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
                    optimized_config['attn_implementation'] = 'flash_attention_2'
            except ImportError:
                pass

        # Skip memory profiling for MPS to prevent buffer allocation errors
        # Use fixed ultra-conservative settings for MPS stability
        if self.is_mps:
            # MPS conservative settings for maximum stability
            optimized_config.update({
                'per_device_train_batch_size': 1,  # Conservative batch size for MPS stability
                'gradient_accumulation_steps': 2,   # Minimal accumulation for testing
            })
            logger.info("🔧 Applied conservative MPS batch sizing for stability")
        else:
            # Calculate optimal batch size if model provided for CPU training
            if model and tokenizer:
                batch_info = self.calculate_optimal_batch_size(model, tokenizer)
                optimized_config.update({
                    'per_device_train_batch_size': batch_info['recommended_batch_size'],
                    'gradient_accumulation_steps': batch_info['gradient_accumulation_steps'],
                })

        # Learning rate adjustments for MPS
        if self.is_mps:
            # MPS with ultra-conservative settings - very conservative learning rate
            current_lr = optimized_config.get('learning_rate', 2e-4)
            # Much more conservative learning rate for stability
            optimized_config['learning_rate'] = min(current_lr * 0.5, 1e-4)

        # Memory monitoring
        optimized_config.update({
            'logging_steps': min(optimized_config.get('logging_steps', 50), 25),
            'save_steps': optimized_config.get('save_steps', 500),
        })

        logger.info("⚡ MPS-optimized training configuration:")
        for key, value in optimized_config.items():
            if key in ['learning_rate', 'per_device_train_batch_size', 'gradient_accumulation_steps',
                      'bf16', 'fp16', 'gradient_checkpointing']:
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
        """Apply MPS-specific model optimizations."""
        if not self.is_mps:
            return model

        # Enable gradient checkpointing for memory efficiency
        model.gradient_checkpointing_enable()

        # MPS-specific optimizations
        for module in model.modules():
            if hasattr(module, 'to'):
                # Ensure all parameters are on MPS
                module.to(self.device)

        logger.info("🔧 Model optimized for MPS training")
        return model

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
        stats = torch.mps.memory_stats()
        current = torch.mps.current_allocated_memory()
        peak = torch.mps.peak_allocated_memory()

        return {
            'current_allocated': current,
            'peak_allocated': peak,
            'memory_stats': stats,
            'utilization': current / torch.mps.get_memory_info()[0] if torch.mps.get_memory_info()[0] > 0 else 0
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
