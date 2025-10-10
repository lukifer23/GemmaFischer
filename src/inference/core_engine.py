#!/usr/bin/env python3
"""
Core Inference Engine for ChessGemma

Provides the fundamental model loading, text generation, and basic inference capabilities.
This module handles the core interaction with the underlying language model.
"""

from __future__ import annotations

import os
import torch
import re
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from contextlib import nullcontext

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import threading

# Import common utilities
from ..utils.common import (
    get_logger, get_error_handler, get_config_manager,
    find_latest_dir, resolve_model_path, resolve_adapter_path,
    safe_import, conditional_import
)

# Get utility functions
logger = get_logger(__name__)
error_handler = get_error_handler()
config_manager = get_config_manager()

# Import optional dependencies with fallbacks
log_performance = safe_import('src.utils.logging_config.log_performance', lambda func: func)
model_validator = conditional_import('src.utils.model_validator', 'get_model_validator')()
if model_validator:
    validate_model = model_validator.validate_model_integrity
else:
    validate_model = lambda *args, **kwargs: None

# Environment hygiene and resource constraints
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# Use common utilities for path resolution


class ChessGemmaCoreEngine:
    """Core inference engine handling model loading and basic text generation."""

    def __init__(self, model_path: Optional[str] = None, adapter_path: Optional[str] = None):
        self.project_root = Path(__file__).resolve().parents[2]
        # Preserve provided strings for tests; otherwise resolve defaults
        self.model_path = model_path if model_path else resolve_model_path()
        self.adapter_path = adapter_path if adapter_path else str(resolve_adapter_path(self.project_root)) if resolve_adapter_path(self.project_root) else None

        self.tokenizer = None
        self.model = None
        self.is_loaded = False
        self._device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

        # Performance-optimized debug mode - only enable when explicitly requested
        self.debug = os.environ.get('CHESSGEMMA_DEBUG', '0') not in ('0', 'false', 'False')

        # Thread safety for model loading - use faster Lock for better performance
        self._model_load_lock = threading.Lock()
        self._model_loading = False

        # Performance monitoring - reduced overhead
        self._generation_stats = {
            'total_tokens_generated': 0,
            'average_generation_time': 0.0,
            'cache_hit_rate': 0.0,
            'memory_peak_usage': 0
        }

    def load_model(self) -> bool:
        """Lazily load tokenizer and model (MPS/Auto device) - optimized."""
        from pathlib import Path

        if self.is_loaded and self.model is not None and self.tokenizer is not None:
            return True

        # Thread-safe model loading - optimized double-check
        if self._model_loading:
            # Simple busy wait for better performance
            while self._model_loading and not self.is_loaded:
                pass
            return self.is_loaded

        with self._model_load_lock:
            # Double-check after acquiring lock (more efficiently)
            if self.is_loaded and self.model is not None and self.tokenizer is not None:
                return True

            self._model_loading = True

        try:
            if not self.model_path:
                print("Model path not configured. Set CHESSGEMMA_MODEL_ID or CHESSGEMMA_MODEL_PATH.")
                return False

            model_path_obj = Path(self.model_path)
            using_local_weights = model_path_obj.exists()
            model_ref = str(model_path_obj) if using_local_weights else self.model_path

            if using_local_weights:
                logger.info(f"Loading Gemma weights from local snapshot at {model_ref}")
            else:
                logger.info(f"Loading Gemma weights from Hugging Face repo {model_ref}")

            self.tokenizer = AutoTokenizer.from_pretrained(
                model_ref,
                local_files_only=using_local_weights,
                trust_remote_code=True,
            )

            torch_dtype = torch.float16
            device_map = "auto"
            if torch.backends.mps.is_available():
                torch_dtype = torch.float32
                device_map = None
            elif not torch.cuda.is_available():
                torch_dtype = torch.float32

            base_model = AutoModelForCausalLM.from_pretrained(
                model_ref,
                local_files_only=using_local_weights,
                device_map=device_map,
                attn_implementation="eager",
                trust_remote_code=True,
                torch_dtype=torch_dtype,
            )

            # Apply adapter if available
            applied_adapter = False
            if self.adapter_path:
                candidate = Path(self.adapter_path)
                if candidate.exists():
                    try:
                        self.model = PeftModel.from_pretrained(
                            base_model,
                            str(candidate),
                            is_trainable=False,
                        )
                        applied_adapter = True
                        self.adapter_path = str(candidate)
                    except Exception as peft_err:
                        logger.warning(
                            "Failed to initialize PeftModel from %s: %s", candidate, peft_err
                        )
                        self.model = base_model

            if not applied_adapter:
                self.model = base_model

            # Move model to the configured device
            if self.model is not None:
                try:
                    self.model.to(self._device)
                    try:
                        self._device = next(self.model.parameters()).device
                    except StopIteration:
                        pass
                except Exception as move_err:
                    logger.warning(f"Failed to move model to {self._device}: {move_err}")

            # Set model to eval mode only if model exists
            if self.model is not None:
                self.model.eval()

            # Model validation - use resolved local model path if available
            if model_validator:
                try:
                    # Use the resolved local path for validation
                    model_path_for_validation = str(self.model_path)
                    if model_path_for_validation == "google/gemma-3-270m":
                        # If using HF identifier, try to find local model
                        project_root = Path(__file__).resolve().parents[2]
                        local_model_path = project_root / "models" / "google-gemma-3-270m"
                        if local_model_path.exists():
                            model_path_for_validation = str(local_model_path)

                    validation_result = model_validator.validate_model_integrity(
                        model_path_for_validation, str(self.adapter_path) if self.adapter_path else None
                    )
                    if not validation_result.is_valid:
                        print(f"⚠️  Model validation failed: {', '.join(validation_result.errors)}")
                        for warning in validation_result.warnings:
                            print(f"⚠️  {warning}")
                    else:
                        print("✅ Model validation passed")
                except Exception as val_e:
                    print(f"⚠️  Model validation error: {val_e}")

            self.is_loaded = True
            self._model_loading = False  # Reset loading flag
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.is_loaded = False
            self._model_loading = False  # Reset loading flag
            return False
        finally:
            # Ensure loading flag is always reset
            self._model_loading = False

    def unload_model(self) -> None:
        """Free model resources and reset state."""
        try:
            if self.model is not None:
                del self.model
            if self.tokenizer is not None:
                del self.tokenizer
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
        finally:
            self.model = None
            self.tokenizer = None
            self.is_loaded = False

    def generate_text(
        self,
        prompt: str,
        max_new_tokens: int = 200,
        do_sample: bool = True,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
    ) -> str:
        """Generate raw text from a direct prompt string (no chat template) - optimized."""
        if not self.load_model():
            return ""

        try:
            # Optimized tokenization and device placement
            inputs = self.tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            # Optimized generation parameters for better performance
            generation_kwargs = {
                'max_new_tokens': max_new_tokens,
                'do_sample': do_sample,
                'pad_token_id': self.tokenizer.eos_token_id,
                'use_cache': True,  # Enable KV cache for better performance
            }

            # Only add optional parameters if they differ from defaults
            if temperature != 0.7:
                generation_kwargs['temperature'] = temperature
            if top_p != 0.9:
                generation_kwargs['top_p'] = top_p
            if repetition_penalty != 1.0:
                generation_kwargs['repetition_penalty'] = repetition_penalty

            with torch.no_grad():
                outputs = self.model.generate(**inputs, **generation_kwargs)

            # More efficient decoding
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        except Exception as e:
            logger.warning(f"Text generation failed: {e}")
            return ""

    def get_model_info(self) -> Dict[str, Any]:
        """Get basic model information."""
        device = str(next(self.model.parameters()).device) if (self.model is not None) else "unknown"
        return {
            "base_model": str(self.model_path) if self.model_path else None,
            "adapter_path": str(self.adapter_path) if self.adapter_path else None,
            "is_loaded": self.is_loaded,
            "device": device,
        }

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return self._generation_stats.copy()
