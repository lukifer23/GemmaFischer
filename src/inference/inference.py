"""
Enhanced Unified inference interface for ChessGemma.

Provides optimized inference with:
- Enhanced Chess Inference: Advanced caching, expert switching, performance monitoring
- Legacy ChessGemmaInference: Original interface for backward compatibility
- ChessModelInterface: Thin wrapper for UCI bridge compatibility
- Convenience module functions: All original functions plus enhanced versions
"""

from __future__ import annotations

import os
import glob
import re
import traceback
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from contextlib import nullcontext

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from transformers.generation.logits_process import LogitsProcessor
from transformers.generation.utils import GenerationConfig
from collections import OrderedDict
import logging
import threading
from functools import lru_cache
import hashlib

# Import MoE components
try:
    from .moe_router import ChessMoERouter, MoEInferenceManager
    MOE_AVAILABLE = True
except ImportError:
    MOE_AVAILABLE = False
    ChessMoERouter = None
    MoEInferenceManager = None

__all__ = [
    'ChessGemmaInference',
    'ChessModelInterface',
    'get_inference_instance',
    'run_inference',
    'load_model',
    'unload_model',
    'get_model_info'
]


# Import logging
try:
    from ..utils.logging_config import get_logger, log_performance
    logger = get_logger(__name__)
except ImportError:
    # Fallback to basic logging
    import logging
    logger = logging.getLogger(__name__)
    log_performance = lambda func: func

# Import error handling
try:
    from ..utils.error_handler import get_error_handler, error_boundary, handle_error
    error_handler = get_error_handler()
except ImportError:
    # Fallback if error handler not available
    error_handler = None

    def error_boundary(*args, **kwargs):
        return nullcontext()

    def handle_error(*args, **kwargs):
        return None

# Import model validation
try:
    from ..utils.model_validator import get_model_validator, validate_model
    model_validator = get_model_validator("models", "checkpoints")
except ImportError:
    # Fallback if model validator not available
    model_validator = None
    validate_model = lambda *args, **kwargs: None

# Environment hygiene and resource constraints
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _find_latest_dir(patterns: List[str]) -> Optional[Path]:
    """Find the most recently modified directory matching any of the glob patterns."""
    candidates: List[Path] = []
    for pattern in patterns:
        for path in glob.glob(pattern):
            p = Path(path)
            if p.is_dir():
                candidates.append(p)
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _find_latest_file(patterns: List[str]) -> Optional[Path]:
    """Find the most recently modified file matching any of the glob patterns."""
    candidates: List[Path] = []
    for pattern in patterns:
        for path in glob.glob(pattern):
            p = Path(path)
            if p.is_file():
                candidates.append(p)
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _resolve_default_model_path(project_root: Path) -> str:
    """Resolve the default model identifier or local snapshot directory."""
    env_override = os.environ.get("CHESSGEMMA_MODEL_PATH")
    if env_override:
        override_path = Path(env_override).expanduser()
        return str(override_path) if override_path.exists() else env_override

    env_model_id = os.environ.get("CHESSGEMMA_MODEL_ID")
    if env_model_id:
        return env_model_id

    # Prefer local snapshots when they exist so we can operate fully offline.
    base = project_root / "models" / "google-gemma-3-270m"
    if not base.exists():
        # Fall back to the public Hugging Face identifier.
        return "google/gemma-3-270m"
    return str(base)


def _resolve_latest_adapter_path(project_root: Path) -> Optional[Path]:
    """Find the latest LoRA checkpoint directory under checkpoints/."""
    patterns = [
        str(project_root / "checkpoints" / "lora_full" / "checkpoint-*"),
        str(project_root / "checkpoints" / "lora_expanded" / "checkpoint-*"),
        str(project_root / "checkpoints" / "lora_poc" / "checkpoint-*"),
        str(project_root / "checkpoints" / "lora_full_resume_smoke" / "checkpoint-*"),
        str(project_root / "checkpoints" / "lora_curriculum" / "checkpoint-*"),
    ]
    latest = _find_latest_dir(patterns)
    if latest is not None:
        return latest

    # Fall back to expert-specific directories when global adapters are absent.
    expert_patterns = [
        str(project_root / "checkpoints" / "lora_uci" / "checkpoint-*"),
        str(project_root / "checkpoints" / "lora_tutor" / "checkpoint-*"),
        str(project_root / "checkpoints" / "lora_director" / "checkpoint-*"),
    ]
    return _find_latest_dir(expert_patterns)


class ChessGemmaInference:
    """Unified inference with optional adapter and dual-mode prompting.

    This class now uses the new modular architecture internally for better maintainability.
    """

    def __init__(self, model_path: Optional[str] = None, adapter_path: Optional[str] = None):
        self.project_root = Path(__file__).resolve().parents[2]

        # Use new modular architecture internally
        try:
            from .core_engine import ChessGemmaCoreEngine
            from .caching import ChessInferenceCache
            from .expert_manager import ChessExpertManager

            self._core_engine = ChessGemmaCoreEngine(model_path, adapter_path)
            self._cache = ChessInferenceCache()
            self._expert_manager = ChessExpertManager(self._core_engine)

            # Maintain backward compatibility
            self.model_path = self._core_engine.model_path
            self.adapter_path = self._core_engine.adapter_path
            self.tokenizer = self._core_engine.tokenizer
            self.model = self._core_engine.model
            self.is_loaded = self._core_engine.is_loaded
            self.debug = self._core_engine.debug

        except ImportError:
            # Fallback to legacy implementation if new modules aren't available
            self.model_path = model_path if model_path else _resolve_default_model_path(self.project_root)
            self.adapter_path = adapter_path if adapter_path else _resolve_latest_adapter_path(self.project_root)
            if isinstance(self.model_path, Path):
                self.model_path = str(self.model_path)

            self.tokenizer = None
            self.model = None
            self.is_loaded = False
            self.debug = os.environ.get('CHESSGEMMA_DEBUG', '0') not in ('0', 'false', 'False')
            if self.debug:
                try:
                    logger.logger.setLevel(logging.DEBUG)
                except AttributeError:
                    logger.setLevel(logging.DEBUG)

        # Prompt templates cache
        self._engine_template: Optional[str] = None
        self._tutor_template: Optional[str] = None

        # Adapter management
        self._adapter_paths: Dict[str, Path] = {}
        # Map physical adapter names -> loaded flag
        self._loaded_adapters: Dict[str, bool] = {}
        # Map logical expert name (uci/tutor/director) -> physical adapter name (e.g., uci@checkpoint-600)
        self._logical_to_physical: Dict[str, str] = {}
        # Track which path a logical expert was loaded from
        self._adapter_loaded_from: Dict[str, Path] = {}
        self._active_adapter: Optional[str] = None

        # Performance optimization caches
        self._kv_cache = {}  # KV cache for repeated positions
        self._response_cache = OrderedDict()  # Response cache for identical queries
        self._cache_max_size = 256
        self._cache_hits = 0
        self._total_requests = 0

        # Simple memoization for deterministic engine prompts (FEN -> move)
        self._engine_cache_max = 512
        self._engine_cache: "OrderedDict[str, str]" = OrderedDict()
        self._cache_lock = threading.RLock()

        # Feature flags
        self._engine_rerank_enabled = (os.environ.get('CHESSGEMMA_ENGINE_RERANK', '1') not in ('0', 'false', 'False'))
        self._engine_policy = os.environ.get('CHESSGEMMA_ENGINE_POLICY', 'sample').strip().lower()  # sample | logprob
        self._engine_constrain_enabled = (os.environ.get('CHESSGEMMA_ENGINE_CONSTRAIN', '0') not in ('0','false','False'))
        self._engine_constrain_mode = os.environ.get('CHESSGEMMA_ENGINE_CONSTRAIN_MODE', 'simple').strip().lower()
        self._allowed_token_ids_cache: Optional[set] = None
        self._uci_token_info: Optional[Dict[int, str]] = None

        # Performance monitoring
        self._generation_stats = {
            'total_tokens_generated': 0,
            'average_generation_time': 0.0,
            'cache_hit_rate': 0.0,
            'memory_peak_usage': 0
        }

        # Thread safety for model loading
        self._model_load_lock = threading.RLock()
        self._model_loading = False

        # Pre-warming and lazy initialization
        self._prewarm_enabled = os.environ.get('CHESSGEMMA_PREWARM_ENABLED', '1') not in ('0', 'false', 'False')
        self._prewarm_thread: Optional[threading.Thread] = None
        self._prewarm_complete = False
        self._lazy_loading = True  # Enable lazy loading by default

        # Batch processing optimization
        self._batch_processing_enabled = os.environ.get('CHESSGEMMA_BATCH_PROCESSING', '1') not in ('0', 'false', 'False')
        self._max_batch_size = int(os.environ.get('CHESSGEMMA_MAX_BATCH_SIZE', '8'))
        self._batch_queue: List[Dict[str, Any]] = []
        self._batch_lock = threading.Lock()
        self._batch_processor_thread: Optional[threading.Thread] = None
        self._batch_shutdown = False

        # Enable debug logging to troubleshoot empty responses
        self.debug = True

        # MoE Router integration
        self.moe_router: Optional[ChessMoERouter] = None
        self.moe_manager: Optional[MoEInferenceManager] = None
        self.moe_enabled = MOE_AVAILABLE and (os.environ.get('CHESSGEMMA_MOE_ENABLED', '1') not in ('0', 'false', 'False'))
        self._expert_paths: Dict[str, str] = {}
        self._moe_dispatch_depth = 0

        # Initialize MoE if available and enabled
        if self.moe_enabled and MOE_AVAILABLE:
            self._initialize_moe_system()

        # Start pre-warming if enabled
        if self._prewarm_enabled and not self._lazy_loading:
            self._start_prewarm()

    def _initialize_moe_system(self) -> None:
        """Initialize the Mixture of Experts system.

        Router checkpoints are expected under ``checkpoints/moe_router/`` and
        expert adapters under ``checkpoints/lora_<expert>/checkpoint-*/``.
        """

        def _disable_moe(message: str, level: str = "warning") -> None:
            log_fn = getattr(logger, level, logger.warning)
            log_fn(message)
            self.moe_enabled = False
            self.moe_router = None
            self.moe_manager = None

        try:
            checkpoints_root = self.project_root / "checkpoints"
            if not checkpoints_root.exists():
                _disable_moe(
                    f"MoE checkpoints directory not found at {checkpoints_root}. "
                    "Falling back to single-expert mode.",
                    level="info",
                )
                return

            expert_search_patterns = {
                "uci": [str(checkpoints_root / "lora_uci" / "checkpoint-*")],
                "tutor": [str(checkpoints_root / "lora_tutor" / "checkpoint-*")],
                "director": [str(checkpoints_root / "lora_director" / "checkpoint-*")],
            }

            expert_paths: Dict[str, str] = {}
            for expert, patterns in expert_search_patterns.items():
                latest_dir = _find_latest_dir(patterns)
                if latest_dir is not None:
                    expert_paths[expert] = str(latest_dir)
                    logger.debug(
                        f"Discovered {expert} expert adapter at {latest_dir}"
                    )
                else:
                    logger.info(
                        f"No adapter checkpoint found for {expert} expert (searched {patterns})"
                    )

            self._expert_paths = expert_paths

            if len(self._expert_paths) < 2:
                _disable_moe(
                    "Insufficient expert checkpoints for MoE (need at least 2). "
                    "Falling back to single-expert mode.",
                    level="info",
                )
                return

            router_override = os.environ.get("CHESSGEMMA_MOE_ROUTER_CKPT")
            router_checkpoint: Optional[Path] = None
            if router_override:
                candidate = Path(router_override).expanduser()
                if candidate.is_file():
                    router_checkpoint = candidate
                    logger.info(
                        "Using MoE router checkpoint override from %s", candidate
                    )
                else:
                    logger.warning(
                        "MoE router checkpoint override %s not found; falling back to default search",
                        router_override=router_override,
                    )

            if router_checkpoint is None:
                router_dir = checkpoints_root / "moe_router"
            router_patterns = [
                str(router_dir / "router*.pt"),
                str(router_dir / "router*.bin"),
                str(router_dir / "router*.safetensors"),
                str(router_dir / "checkpoint-*" / "router*.pt"),
                str(router_dir / "checkpoint-*" / "*.pt"),
                str(router_dir / "checkpoint-*" / "*.bin"),
                str(router_dir / "checkpoint-*" / "*.safetensors"),
                str(router_dir / "final_checkpoint.pth"),
                str(router_dir / "best_checkpoint.pth"),
                str(router_dir / "*.pt"),
                str(router_dir / "*.bin"),
                str(router_dir / "*.safetensors"),
            ]
            router_checkpoint = _find_latest_file(router_patterns)

            if router_checkpoint is None:
                _disable_moe(
                    "MoE router checkpoint not found (expected under checkpoints/moe_router/). "
                    "Falling back to single-expert mode.",
                    level="info",
                )
                return

            self.moe_router = ChessMoERouter(
                num_experts=len(self._expert_paths),
                expert_names=list(self._expert_paths.keys())
            )
            try:
                self.moe_router.load_router(str(router_checkpoint))
            except Exception as exc:
                _disable_moe(
                    f"Failed to load MoE router checkpoint at {router_checkpoint}: {exc}",
                    level="error",
                )
                return

            self.moe_manager = MoEInferenceManager(
                self.moe_router, self._expert_paths, self
            )
            logger.info(
                f"MoE System initialized with experts: {list(self._expert_paths.keys())}"
            )

        except Exception as e:
            _disable_moe(
                f"Unexpected error while initializing MoE system: {e}",
                level="error",
            )

    def _start_prewarm(self):
        """Start background model pre-warming."""
        if self._prewarm_thread and self._prewarm_thread.is_alive():
            logger.debug("Pre-warm already in progress")
            return

        self._prewarm_thread = threading.Thread(
            target=self._prewarm_models,
            name="model-prewarm",
            daemon=True
        )
        self._prewarm_thread.start()
        logger.info("🚀 Started model pre-warming in background")

    def _prewarm_models(self):
        """Pre-warm models in the background to reduce first-request latency."""
        try:
            logger.debug("🔄 Pre-warming models...")

            # Load the base model first
            if not self.is_loaded:
                self.load_model()

            # Pre-warm tokenization with common chess prompts
            if self.is_loaded and self.tokenizer:
                prewarm_prompts = [
                    "FEN: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1\nWhat is the best move?",
                    "FEN: rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1\nAnalyze this position.",
                    "What are the main principles of chess openings?",
                ]

                # Tokenize prompts to warm up tokenizer cache
                for prompt in prewarm_prompts:
                    try:
                        self.tokenizer(prompt, return_tensors="pt")
                    except Exception as e:
                        logger.debug(f"Pre-warm tokenization failed for prompt: {e}")

            # Pre-warm MoE router if available
            if self.moe_enabled and self.moe_router:
                try:
                    # Pre-compute features for common positions
                    common_fens = [
                        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",  # Starting position
                        "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",  # Italian game
                    ]

                    for fen in common_fens:
                        try:
                            self.moe_router._extract_position_features_cached(fen, "auto")
                        except Exception as e:
                            logger.debug(f"MoE pre-warm failed for FEN {fen}: {e}")
                except Exception as e:
                    logger.debug(f"MoE router pre-warm failed: {e}")

            self._prewarm_complete = True
            logger.info("✅ Model pre-warming completed")

        except Exception as e:
            logger.error(f"❌ Model pre-warming failed: {e}")
            self._prewarm_complete = False

    def wait_for_prewarm(self, timeout: float = 30.0) -> bool:
        """Wait for pre-warming to complete.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            True if pre-warming completed, False if timed out
        """
        if not self._prewarm_enabled or self._lazy_loading:
            return True  # No pre-warming needed

        if self._prewarm_complete:
            return True

        if self._prewarm_thread and self._prewarm_thread.is_alive():
            self._prewarm_thread.join(timeout=timeout)
            return self._prewarm_complete

        return False

    def ensure_model_ready(self) -> bool:
        """Ensure model is loaded and pre-warmed before use.

        Returns:
            True if model is ready, False otherwise
        """
        # Load model if using lazy loading
        if self._lazy_loading and not self.is_loaded:
            if not self.load_model():
                return False

        # Wait for pre-warming if enabled
        if self._prewarm_enabled and not self._prewarm_complete:
            return self.wait_for_prewarm(timeout=10.0)

        return self.is_loaded

    def batch_tokenize(self, texts: List[str], **kwargs) -> Dict[str, Any]:
        """Batch tokenize multiple texts efficiently.

        Args:
            texts: List of texts to tokenize
            **kwargs: Additional tokenization arguments

        Returns:
            Tokenized batch
        """
        if not self.ensure_model_ready():
            return {}

        if not texts:
            return {}

        try:
            # Use the tokenizer's batch processing capabilities
            batch_result = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                **kwargs
            )

            # Move to device if needed
            if hasattr(batch_result, 'to') and hasattr(self.model, 'device'):
                batch_result = {k: v.to(self.model.device) for k, v in batch_result.items()}

            return batch_result

        except Exception as e:
            logger.warning(f"Batch tokenization failed: {e}")
            # Fallback to individual tokenization
            return self._fallback_batch_tokenize(texts, **kwargs)

    def _fallback_batch_tokenize(self, texts: List[str], **kwargs) -> Dict[str, Any]:
        """Fallback batch tokenization for when standard batching fails."""
        individual_results = []
        max_length = 0

        # Tokenize individually first
        for text in texts:
            try:
                result = self.tokenizer(text, return_tensors="pt", **kwargs)
                individual_results.append(result)
                max_length = max(max_length, result['input_ids'].shape[1])
            except Exception as e:
                logger.debug(f"Individual tokenization failed for text: {e}")
                # Add empty tensor as placeholder
                empty_tensor = torch.zeros((1, 1), dtype=torch.long)
                individual_results.append({'input_ids': empty_tensor, 'attention_mask': empty_tensor})

        if not individual_results:
            return {}

        # Pad to max length
        batch_input_ids = []
        batch_attention_masks = []

        for result in individual_results:
            input_ids = result['input_ids'].squeeze(0)
            attention_mask = result['attention_mask'].squeeze(0)

            # Pad if necessary
            if input_ids.shape[0] < max_length:
                padding_length = max_length - input_ids.shape[0]
                input_ids = torch.cat([input_ids, torch.zeros(padding_length, dtype=input_ids.dtype)])
                attention_mask = torch.cat([attention_mask, torch.zeros(padding_length, dtype=attention_mask.dtype)])

            batch_input_ids.append(input_ids)
            batch_attention_masks.append(attention_mask)

        # Stack into batch
        batch_result = {
            'input_ids': torch.stack(batch_input_ids),
            'attention_mask': torch.stack(batch_attention_masks)
        }

        # Move to device
        if hasattr(self.model, 'device'):
            batch_result = {k: v.to(self.model.device) for k, v in batch_result.items()}

        return batch_result

    def batch_generate(self, prompts: List[str], **generation_kwargs) -> List[str]:
        """Generate responses for multiple prompts in batch.

        Args:
            prompts: List of prompts to generate for
            **generation_kwargs: Generation parameters

        Returns:
            List of generated responses
        """
        if not self.ensure_model_ready() or not prompts:
            return [""] * len(prompts)

        try:
            # Batch tokenize
            batch_inputs = self.batch_tokenize(prompts)

            if not batch_inputs:
                return [""] * len(prompts)

            # Generate batch
            with torch.no_grad():
                batch_outputs = self.model.generate(
                    **batch_inputs,
                    **generation_kwargs
                )

            # Decode batch results
            results = []
            for i in range(len(prompts)):
                try:
                    # Extract individual sequence (handling padding)
                    output_ids = batch_outputs[i]
                    # Remove padding tokens if present
                    eos_token_id = generation_kwargs.get('eos_token_id', self.tokenizer.eos_token_id)
                    if eos_token_id is not None:
                        eos_positions = (output_ids == eos_token_id).nonzero(as_tuple=True)[0]
                        if len(eos_positions) > 0:
                            output_ids = output_ids[:eos_positions[0] + 1]

                    # Decode
                    response = self.tokenizer.decode(output_ids, skip_special_tokens=True)

                    # Remove original prompt if echoed
                    original_prompt = prompts[i]
                    if response.startswith(original_prompt):
                        response = response[len(original_prompt):].strip()

                    results.append(response)

                except Exception as e:
                    logger.debug(f"Batch decode failed for prompt {i}: {e}")
                    results.append("")

            return results

        except Exception as e:
            logger.warning(f"Batch generation failed: {e}")
            # Fallback to individual generation
            return self._fallback_batch_generate(prompts, **generation_kwargs)

    def _fallback_batch_generate(self, prompts: List[str], **generation_kwargs) -> List[str]:
        """Fallback to individual generation when batch processing fails."""
        results = []
        for prompt in prompts:
            try:
                result = self.generate_text(prompt, **generation_kwargs)
                results.append(result)
            except Exception as e:
                logger.debug(f"Individual generation failed: {e}")
                results.append("")
        return results

    def shutdown(self):
        """Shutdown the inference system and clean up resources."""
        self._batch_shutdown = True

        # Shutdown pre-warm thread
        if self._prewarm_thread and self._prewarm_thread.is_alive():
            self._prewarm_thread.join(timeout=5.0)

        # Shutdown MoE manager
        if self.moe_manager:
            self.moe_manager.shutdown()

        logger.debug("Inference system shutdown complete")

    def __del__(self):
        """Cleanup on deletion."""
        self.shutdown()

    def load_model(self) -> bool:
        """Lazily load tokenizer and model (MPS/Auto device)."""
        # Try to use new modular architecture first
        if hasattr(self, '_core_engine'):
            result = self._core_engine.load_model()
            # Update backward compatibility attributes
            self.tokenizer = self._core_engine.tokenizer
            self.model = self._core_engine.model
            self.is_loaded = self._core_engine.is_loaded
            # Set up adapter management for backward compatibility
            if result:
                self.refresh_adapters()
            return result

        if self.is_loaded and self.model is not None and self.tokenizer is not None:
            return True

        # Thread-safe model loading
        with self._model_load_lock:
            if self.is_loaded and self.model is not None and self.tokenizer is not None:  # Double-check after acquiring lock
                return True

            if self._model_loading:  # Another thread is already loading
                logger.info("Model loading already in progress, waiting...")
                # Wait for the other thread to finish loading
                while self._model_loading and not self.is_loaded:
                    import time
                    time.sleep(0.1)
                return self.is_loaded

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

            import torch

            torch_dtype = torch.float16
            device_map = "auto"
            if torch.backends.mps.is_available():
                torch_dtype = torch.float32
                device_map = None

            base_model = AutoModelForCausalLM.from_pretrained(
                model_ref,
                local_files_only=using_local_weights,
                device_map=device_map,
                attn_implementation="eager",
                trust_remote_code=True,
                torch_dtype=torch_dtype,
            )

            # Discover available adapters before wrapping the base model so we can
            # initialize a PEFT container with at least one expert adapter.
            self._discover_adapter_paths()

            initial_adapter_path: Optional[Path] = None
            initial_adapter_logical: Optional[str] = None

            if self.adapter_path:
                candidate = Path(self.adapter_path)
                if candidate.exists():
                    initial_adapter_path = candidate
                    for logical_name, path in self._adapter_paths.items():
                        if path == candidate:
                            initial_adapter_logical = logical_name
                            break

            if initial_adapter_path is None:
                if "uci" in self._adapter_paths:
                    initial_adapter_path = self._adapter_paths["uci"]
                    initial_adapter_logical = "uci"
                elif self._adapter_paths:
                    # Fall back to whichever adapter we discovered first
                    initial_adapter_logical, initial_adapter_path = next(iter(self._adapter_paths.items()))

            applied_adapter = False
            if initial_adapter_path is not None:
                try:
                    self.model = PeftModel.from_pretrained(
                        base_model,
                        str(initial_adapter_path),
                        is_trainable=False,
                    )
                    applied_adapter = True
                    if initial_adapter_logical:
                        self._active_adapter = initial_adapter_logical
                    self.adapter_path = str(initial_adapter_path)
                except Exception as peft_err:
                    logger.warning(
                        "Failed to initialize PeftModel from %s: %s", initial_adapter_path, peft_err
                    )
                    self.model = base_model

            if not applied_adapter:
                self.model = base_model

            self.model.eval()
            # Discover known expert adapters on disk for quick switching now that
            # the model exposes PEFT adapter management APIs.
            self.refresh_adapters()
            # Model validation - use resolved local model path if available
            if model_validator:
                try:
                    # Use the resolved local path for validation
                    model_path_for_validation = str(self.model_path)
                    logger.debug(f"Model validation: initial model_path = {model_path_for_validation}")
                    if model_path_for_validation == "google/gemma-3-270m":
                        # If using HF identifier, try to find local model
                        local_model_path = self.project_root / "models" / "google-gemma-3-270m"
                        logger.debug(f"Model validation: checking local path {local_model_path} exists = {local_model_path.exists()}")
                        if local_model_path.exists():
                            model_path_for_validation = str(local_model_path)
                            logger.debug(f"Model validation: using local path {model_path_for_validation}")

                    adapter_path_for_validation = str(self.adapter_path) if self.adapter_path else None
                    logger.debug(f"Model validation: validating model_path={model_path_for_validation}, adapter_path={adapter_path_for_validation}")

                    validation_result = model_validator.validate_model_integrity(
                        model_path_for_validation, adapter_path_for_validation
                    )
                    if not validation_result.is_valid:
                        logger.warning(f"Model validation failed: {', '.join(validation_result.errors)}")
                        # Continue anyway but log warnings
                        for warning in validation_result.warnings:
                            logger.warning(f"Model validation warning: {warning}")
                    else:
                        logger.info("✅ Model validation passed")
                except Exception as val_e:
                    logger.error(f"Model validation error: {val_e}")
                    logger.error(f"Model validation traceback: {traceback.format_exc()}")

            self.is_loaded = True
            self._model_loading = False  # Reset loading flag
            if self.moe_enabled and self.moe_manager:
                try:
                    self.moe_manager.prime_available_experts()
                except Exception as moe_err:
                    logger.warning("MoE expert priming failed: %s", error=moe_err)
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
            self._active_adapter = None
            self._loaded_adapters.clear()
            self._logical_to_physical.clear()
            self._adapter_loaded_from.clear()
            self._allowed_token_ids_cache = None
            self._uci_token_info = None
            with self._cache_lock:
                self._engine_cache.clear()

    def _discover_adapter_paths(self) -> None:
        """Populate mapping of expert names to latest adapter checkpoint paths.

        Primary targets are expert-specific directories. If none found for an expert,
        fall back to generic LoRA runs so the system still functions, with visibility
        into which checkpoint is actually used.
        """
        checkpoints_root = self.project_root / "checkpoints"

        # Primary expert-specific locations - prioritize better trained models
        primary = {
            "uci": [checkpoints_root / "lora_uci"],  # Use UCI-specific training
            "tutor": [checkpoints_root / "lora_tutor"],
            "director": [checkpoints_root / "lora_director"],
        }
        # Fallback generic runs if an expert-specific adapter is missing
        fallback_common = [
            checkpoints_root / "lora_full",
            checkpoints_root / "lora_poc",
            checkpoints_root / "lora_curriculum",
            checkpoints_root / "lora_formatted",  # may contain director-oriented runs
        ]

        self._adapter_paths.clear()
        for expert, primary_dirs in primary.items():
            latest = _find_latest_dir([str(d / "checkpoint-*") for d in primary_dirs if d.exists()])
            if latest is None:
                # try fallbacks
                latest = _find_latest_dir([str(d / "checkpoint-*") for d in fallback_common if d.exists()])
            if latest is not None:
                self._adapter_paths[expert] = latest

    def _ensure_adapter_loaded(self, logical_name: str, path: Path) -> None:
        """Load adapter weights if not already loaded from this path.

        Uses physical adapter names of the form '<logical>@<checkpoint-dir-name>' to allow
        reloading newer checkpoints while the process is running.
        """
        if not hasattr(self.model, "load_adapter"):
            return
        # If already loaded from this exact path, nothing to do
        loaded_from = self._adapter_loaded_from.get(logical_name)
        if loaded_from and loaded_from == path:
            return
        # Create a physical name tied to checkpoint directory
        physical_name = f"{logical_name}@{path.name}"
        if self._loaded_adapters.get(physical_name):
            # Already loaded this exact physical adapter; just (re)map logical -> physical
            self._logical_to_physical[logical_name] = physical_name
            self._adapter_loaded_from[logical_name] = path
            return
        try:
            self.model.load_adapter(str(path), adapter_name=physical_name)
            self._loaded_adapters[physical_name] = True
            self._logical_to_physical[logical_name] = physical_name
            self._adapter_loaded_from[logical_name] = path
        except Exception:
            pass

    def refresh_adapters(self) -> None:
        """Re-discover latest checkpoints and ensure corresponding adapters are loaded.

        Safe to call frequently (e.g., before switching adapters) to pick up freshly saved checkpoints
        without restarting the server process.
        """
        self._discover_adapter_paths()
        for logical_name, path in self._adapter_paths.items():
            self._ensure_adapter_loaded(logical_name, path)

    def set_active_adapter(self, name: Optional[str]) -> None:
        """Switch the active LoRA adapter by logical name (uci/tutor/director).

        This method re-discovers latest checkpoints and loads them on demand.
        """
        if not name:
            return
        # Refresh to capture newly saved checkpoints
        self.refresh_adapters()
        # Ensure requested logical adapter is loaded (loads latest if newer exists)
        path = self._adapter_paths.get(name)
        if path is not None:
            self._ensure_adapter_loaded(name, path)
        # Resolve to physical adapter name
        physical = self._logical_to_physical.get(name)
        if hasattr(self.model, "set_adapter") and physical and self._loaded_adapters.get(physical):
            try:
                self.model.set_adapter(physical)
                self._active_adapter = name
            except Exception:
                pass
        else:
            # Provide visibility if requested adapter is unavailable
            try:
                avail = ", ".join(sorted(self._adapter_paths.keys())) or "none"
                print(f"[Router] Requested adapter '{name}' not available. Available: {avail}")
            except Exception:
                pass

    def _load_prompt_template(self, mode: str) -> str:
        """Load prompt template from prompts directory, fallback to defaults."""
        def _load_from_file(filename: str, matcher: Optional[str] = None) -> Optional[str]:
            try:
                path = self.project_root / "prompts" / filename
                if not path.exists():
                    return None
                text = path.read_text(encoding='utf-8')
                code_blocks = re.findall(r"```(?:[^\n]*\n)?(.*?)```", text, re.DOTALL)
                if matcher and code_blocks:
                    for block in code_blocks:
                        if matcher in block:
                            return block.strip()
                if code_blocks:
                    return code_blocks[0].strip()
                return text.strip()
            except Exception:
                return None

        if mode == "engine":
            if self._engine_template is None:
                file_template = _load_from_file("engine_mode.txt", matcher="Mode: Engine")
                self._engine_template = file_template or "Find the best chess move in UCI format."
            return self._engine_template
        elif mode == "tutor":
            if self._tutor_template is None:
                file_template = _load_from_file("tutor_mode.txt", matcher="Mode: Tutor")
                if file_template:
                    self._tutor_template = file_template
                else:
                    # Simplified fallback template for better model response
                    self._tutor_template = "Analyze this chess position and explain the best move."
            return self._tutor_template
        else:
            file_template = _load_from_file("director_mode.txt", matcher="Mode: Director")
            if file_template:
                return file_template
            # Director default - clean Q&A prompt
            return "You are a chess expert. Answer this question:"

    def _build_messages(self, question: str, context: Optional[str], mode: str) -> List[Dict[str, str]]:
        # TEMPORARY: Use ultra-simple prompts to test if model can respond at all
        if mode == "tutor":
            # Simple, direct prompt
            prompt = f"Question: {question}\n\nYou are a chess tutor. Explain this clearly:"
            return [{"role": "user", "content": prompt}]
        elif mode == "director":
            # Simple Q&A format
            prompt = f"Question: {question}\n\nAnswer as a chess expert:"
            return [{"role": "user", "content": prompt}]
        else:
            # Engine mode - keep simple
            prompt = f"Find the best chess move: {question}"
            return [{"role": "user", "content": prompt}]

    def _create_cache_key(self, question: str, context: Optional[str], mode: str,
                         max_new_tokens: int, temperature: float, top_p: float) -> str:
        """Create a unique cache key for the request."""
        key_components = [
            question,
            context or "",
            mode,
            str(max_new_tokens),
            f"{temperature:.3f}",
            f"{top_p:.3f}"
        ]
        key_string = "|".join(key_components)
        return hashlib.md5(key_string.encode()).hexdigest()

    def _check_response_cache(self, cache_key: str) -> Tuple[Optional[Dict[str, Any]], Optional[float]]:
        """Check if response is cached and return a copy with current hit rate."""
        with self._cache_lock:
            cached = self._response_cache.get(cache_key)
            if cached is None:
                return None, None
            self._cache_hits += 1
            self._response_cache.move_to_end(cache_key)
            hit_rate = self._cache_hits / max(self._total_requests, 1)
            # Return a shallow copy to avoid mutating the cached entry outside the lock
            return dict(cached), hit_rate

    def _cache_response(self, cache_key: str, response: Dict[str, Any]):
        """Cache response for future use."""
        with self._cache_lock:
            self._response_cache[cache_key] = dict(response)
            # Maintain cache size
            while len(self._response_cache) > self._cache_max_size:
                self._response_cache.popitem(last=False)

    @property
    def cache_lock(self) -> threading.RLock:
        """Expose cache lock for coordinated access in helper interfaces."""
        return self._cache_lock

    def generate_response(
        self,
        question: str,
        context: Optional[str] = None,
        mode: str = "tutor",
        max_new_tokens: int = 200,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        do_sample: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Generate a response dict with performance optimizations and error handling."""
        import time

        with error_boundary("inference", "generate_response",
                          question=question[:100], mode=mode, max_new_tokens=max_new_tokens):
            start_time = time.time()
            with self._cache_lock:
                self._total_requests += 1

            requested_mode = mode
            if mode == "uci":
                mode = "engine"

            # Ensure model is ready (lazy loading + pre-warming)
            if not self.ensure_model_ready():
                return {
                    "error": "Model not ready",
                    "response": "",
                    "confidence": 0.0,
                    "model_loaded": False,
                    "generation_time": time.time() - start_time,
                    "cached": False,
                    "cache_hit_rate": 0.0,
                    "mode": mode,
                    "requested_mode": requested_mode,
                }

            # Apply expert-specific decoding parameters
            temperature, top_p, do_sample = self._get_expert_decoding_params(mode, temperature, top_p, do_sample)

            # Check response cache for identical requests
            cache_key = self._create_cache_key(question, context, mode, max_new_tokens, temperature, top_p)
            cached_response, cache_hit_rate = self._check_response_cache(cache_key)
            if cached_response:
                cached_response["cached"] = True
                if cache_hit_rate is None:
                    with self._cache_lock:
                        cache_hit_rate = self._cache_hits / max(self._total_requests, 1)
                cached_response["cache_hit_rate"] = cache_hit_rate
                if requested_mode != mode:
                    cached_response.setdefault("requested_mode", requested_mode)
                return cached_response

            # Use MoE routing if available and enabled
            if (
                self.moe_enabled
                and self.moe_manager
                and self._moe_dispatch_depth == 0
                and mode in ['tutor', 'engine', 'director']
            ):
                try:
                    # Extract FEN for MoE routing
                    from .uci_utils import extract_fen
                    fen = extract_fen(question) or extract_fen(context or "")

                    if fen:
                        # Determine query type for MoE
                        query_type = "auto"
                        if mode == "engine":
                            query_type = "engine"
                        elif mode == "tutor":
                            query_type = "tutor"
                        elif mode == "director":
                            query_type = "director"

                        # Use MoE for intelligent routing
                        moe_result = self.moe_manager.analyze_position(fen, query_type)
                        response = moe_result.get('response', '')

                        # Add MoE metadata to response
                        moe_info = moe_result.get('routing_info', {})
                        payload = {
                            "response": response,
                            "confidence": moe_info.get('confidence_score', 0.5),
                            "model_loaded": True,
                            "mode": mode,
                            "moe_used": True,
                            "primary_expert": moe_info.get('primary_expert'),
                            "ensemble_mode": moe_info.get('ensemble_mode'),
                            "routing_reasoning": moe_info.get('reasoning'),
                            "expert_weights": moe_info.get('expert_weights', {}),
                        }
                        if requested_mode != mode:
                            payload["requested_mode"] = requested_mode
                        return payload
                except Exception as e:
                    logger.info(f"MoE routing failed, falling back to standard inference: {e}")
                    # Fall through to standard inference

        try:
            messages = self._build_messages(question, context, mode)
            prompt_text: str

            # Use the simple prompt format we built
            prompt_text = messages[0]['content']

            # Debug logging
            if self.debug:
                logger.debug("INFERENCE DEBUG:")
                logger.debug(f"Mode: {mode}")
                logger.debug(f"Question: {question[:100]}{'...' if len(question) > 100 else ''}")
                logger.debug(f"System Prompt: {messages[0]['content'][:100]}{'...' if len(messages[0]['content']) > 100 else ''}")
                logger.debug(f"Prompt Length: {len(prompt_text)} chars")

            inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)

            # Debug: Check input tensor
            if self.debug:
                logger.debug(f"Input tensor shape: {inputs['input_ids'].shape}")
                logger.debug(f"Input tensor device: {inputs['input_ids'].device}")
                logger.debug(f"Model device: {self.model.device}")

            with torch.no_grad():
                if mode == "engine":
                    # Engine mode: try cached + policy/rerank constrained decoding
                    answer = self._generate_engine_move(question, messages[0]['content'], max_new_tokens)
                    if answer:
                        decoded = messages[0]['content'] + answer
                        outputs = None  # bypass default path
                    else:
                        # Fallback to deterministic single-shot decoding (optionally constrained)
                        logits_processors = None
                        if self._engine_constrain_enabled and self._engine_policy == 'sample':
                            if self._engine_constrain_mode == 'strict':
                                logits_processors = [self._build_stateful_uci_processor(prompt_len=inputs['input_ids'].shape[1])]
                            else:
                                logits_processors = [self._build_uci_logits_processor(prompt_len=inputs['input_ids'].shape[1])]
                        outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=max_new_tokens,
                            do_sample=False,
                            temperature=0.0,
                            top_p=1.0,
                            pad_token_id=self.tokenizer.eos_token_id,
                            eos_token_id=self.tokenizer.eos_token_id,
                            use_cache=True,
                            logits_processor=logits_processors,
                        )
                else:
                    # Debug: Log generation parameters
                    if self.debug:
                        logger.debug(f"Generation params: max_new_tokens={max_new_tokens}, do_sample={do_sample}, temperature={temperature}, top_p={top_p}")

                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=do_sample,
                        top_p=top_p,
                        temperature=temperature,
                        pad_token_id=self.tokenizer.eos_token_id,
                        repetition_penalty=1.1,
                        eos_token_id=self.tokenizer.eos_token_id,
                        use_cache=True,
                    )

                    # Debug: Check outputs immediately
                    if self.debug:
                        logger.debug(f"Raw outputs type: {type(outputs)}")
                        if hasattr(outputs, 'shape'):
                            logger.debug(f"Raw outputs shape: {outputs.shape}")
                        logger.debug(f"Raw outputs preview: {outputs[0][:10] if outputs is not None else 'None'}")

            if mode == "engine" and outputs is None:
                decoded = decoded  # already built above (prompt + answer)
            else:
                decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Debug logging for response
            if self.debug:
                logger.debug(f"Raw Response Length: {len(decoded)} chars")
                logger.debug(f"Raw Response Preview: {decoded[:300]}{'...' if len(decoded) > 300 else ''}")
                logger.debug(f"Raw Response Full: {decoded}")
            
            # Try to strip prompt prefix if echoed
            if decoded.startswith(prompt_text):
                answer = decoded[len(prompt_text):].strip()
                if self.debug:
                    logger.debug(f"Stripped prompt prefix, answer length: {len(answer)}")
                    logger.debug(f"Stripped answer preview: '{answer[:100]}'")
            else:
                answer = decoded.strip()
                if self.debug:
                    logger.debug("No prompt prefix found, using full response")
                    logger.debug(f"Full response preview: '{answer[:100]}'")

            # TEMPORARILY DISABLE PROMPT STRIPPING FOR DEBUGGING
            # If answer is too short after stripping, use full decoded response
            if len(answer) < 20 and len(decoded.strip()) > len(answer) + 50:
                if self.debug:
                    logger.debug(f"Answer too short after stripping ({len(answer)}), using full response")
                answer = decoded.strip()
            
            # Clean up common artifacts
            if answer.startswith("Answer:"):
                answer = answer[7:].strip()
            elif answer.startswith("Move:"):
                answer = answer[5:].strip()
            
            # Remove any remaining prompt fragments
            lines = answer.split('\n')
            content_lines = []
            if self.debug:
                logger.debug(f"Processing {len(lines)} lines from model response")
            for i, line in enumerate(lines):
                line = line.strip()
                if self.debug:
                    logger.debug(f"  Line {i}: '{line[:50]}{'...' if len(line) > 50 else ''}'")
                if line and not line.startswith(('Chess Tutor:', 'Chess Engine:', 'Question:', 'Position:', 'Answer:', 'Move:')):
                    content_lines.append(line)
                    if self.debug:
                        logger.debug(f"    Kept line {i}")
                else:
                    if self.debug:
                        logger.debug(f"    Filtered out line {i}")
            
            if content_lines:
                answer = '\n'.join(content_lines).strip()
                if self.debug:
                    logger.debug(f"Final processed answer: '{answer[:100]}{'...' if len(answer) > 100 else ''}'")
            else:
                if self.debug:
                    logger.debug("All lines were filtered out!")
            
            # Fallback if we still don't have a good answer
            if not answer or len(answer) < 5:
                if mode == "tutor":
                    answer = "I'm having trouble generating a response. Please try rephrasing your question or ask about a specific chess position."
                else:
                    answer = ""  # Defer to engine fallback
                if self.debug:
                    logger.info(f"Using fallback response due to poor model output (answer length: {len(answer) if answer else 0})")
                    logger.info(f"Final answer before fallback: '{answer[:200] if answer else 'None'}'")

            if self.debug:
                logger.debug(f"Final Answer Preview: {answer[:200]}{'...' if len(answer) > 200 else ''}")

            # Post-process for engine mode: extract legal UCI move or fallback to engine
            postprocessed = False
            if mode == "engine":
                import chess
                from .uci_utils import extract_fen, extract_first_legal_move_uci

                fen = extract_fen(question) or extract_fen(prompt_text)
                board = chess.Board(fen) if fen else None
                mv: Optional[str] = None
                if board is not None:
                    mv = extract_first_legal_move_uci(answer, board)
                    if not mv:
                        try:
                            from .chess_engine import ChessEngineManager
                            with ChessEngineManager() as ce:
                                best = ce.get_best_move(board)
                                mv = best.uci() if best else None
                        except Exception:
                            mv = None
                if mv:
                    # Cache the result for this FEN
                    try:
                        if fen:
                            self._engine_cache_store(fen, mv)
                    except Exception:
                        pass
                    answer = mv
                    postprocessed = True

            # Simple heuristic confidence
            if mode == "engine" and answer and len(answer) in (4, 5):
                confidence = 0.9  # high confidence for a valid UCI token
            else:
                word_count = len(answer.split())
                confidence = max(0.1, min(0.95, word_count / 60.0))

            # Update performance stats
            generation_time = time.time() - start_time
            token_count = len(answer.split())
            with self._cache_lock:
                self._generation_stats['total_tokens_generated'] += token_count
                total_requests = max(self._total_requests, 1)
                prev_requests = max(self._total_requests - 1, 0)
                prev_avg = self._generation_stats['average_generation_time']
                self._generation_stats['average_generation_time'] = (
                    (prev_avg * prev_requests) + generation_time
                ) / total_requests
                hit_rate = self._cache_hits / total_requests
                self._generation_stats['cache_hit_rate'] = hit_rate

            response_dict = {
                "response": answer,
                "confidence": confidence,
                "model_loaded": True,
                "mode": mode,
                "postprocessed": postprocessed,
                "prompt_len_chars": len(prompt_text),
                "answer_len_chars": len(answer),
                "generation_time": generation_time,
                "cached": False,
                "cache_hit_rate": hit_rate,
                "tokens_per_second": token_count / max(generation_time, 0.001)
            }

            if requested_mode != mode:
                response_dict["requested_mode"] = requested_mode

            # Cache the response for future use
            self._cache_response(cache_key, response_dict)

            return response_dict
        except Exception as e:
            generation_time = time.time() - start_time
            return {
                "error": str(e),
                "response": "",
                "confidence": 0.0,
                "model_loaded": True,
                "mode": mode,
                "requested_mode": requested_mode,
                "generation_time": generation_time,
                "cached": False,
                "cache_hit_rate": self._generation_stats['cache_hit_rate']
            }

    def generate_expert_response(
        self,
        question: str,
        context: Optional[str] = None,
        expert_mode: str = "tutor",
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        do_sample: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Generate response using a specific expert mode.

        This is a convenience method that maps expert modes to the appropriate
        inference parameters and calls generate_response.
        """
        # Map expert mode to inference mode
        mode_mapping = {
            'uci': 'engine',
            'tutor': 'tutor',
            'director': 'director'
        }

        mode = mode_mapping.get(expert_mode, 'tutor')

        # Use expert-specific defaults if not provided
        if max_new_tokens is None:
            if expert_mode == 'uci':
                max_new_tokens = 8
            elif expert_mode == 'tutor':
                max_new_tokens = 150
            elif expert_mode == 'director':
                max_new_tokens = 200
            else:
                max_new_tokens = 150

        if temperature is None:
            if expert_mode == 'uci':
                temperature = 0.0
            elif expert_mode == 'tutor':
                temperature = 0.7
            elif expert_mode == 'director':
                temperature = 0.6
            else:
                temperature = 0.7

        if top_p is None:
            top_p = 0.9

        if do_sample is None:
            do_sample = temperature > 0.0

        # Call the main generate_response method
        return self.generate_response(
            question=question,
            context=context,
            mode=mode,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample
        )

    def generate_parallel_responses(
        self,
        question: str,
        context: Optional[str] = None,
        experts: List[str] = None,
        max_new_tokens: int = 200,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        do_sample: Optional[bool] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Generate responses from multiple experts in parallel.

        Args:
            question: The chess question to answer
            context: Additional context (FEN, position description, etc.)
            experts: List of expert names to query (default: ['uci', 'tutor', 'director'])
            max_new_tokens: Maximum tokens per response
            temperature: Generation temperature (expert-specific defaults if None)
            top_p: Top-p sampling parameter
            do_sample: Whether to use sampling

        Returns:
            Dict mapping expert names to their response dictionaries
        """
        import time
        import threading

        if experts is None or len(experts) == 0:
            experts = ['uci', 'tutor', 'director']

        start_time = time.time()
        results = {}
        errors = []

        def run_single_expert(expert_name: str):
            """Run inference for a single expert in a thread."""
            try:
                # Map expert name to mode
                mode = 'engine' if expert_name == 'uci' else expert_name

                # Generate response for this expert
                response = self.generate_response(
                    question=question,
                    context=context,
                    mode=mode,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample
                )

                # Store result with expert name
                results[expert_name] = response

            except Exception as e:
                error_msg = f"Error generating {expert_name} response: {str(e)}"
                logger.error(error_msg)
                errors.append(error_msg)

                # Store error response
                results[expert_name] = {
                    "error": str(e),
                    "response": "",
                    "confidence": 0.0,
                    "model_loaded": self.is_loaded,
                    "mode": expert_name,
                    "generation_time": 0.0,
                    "cached": False,
                    "cache_hit_rate": 0.0
                }

        # Create and start threads for each expert
        threads = []
        for expert in experts:
            thread = threading.Thread(
                target=run_single_expert,
                args=(expert,),
                name=f"expert-{expert}"
            )
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=30.0)  # 30 second timeout per expert

        total_time = time.time() - start_time

        # Log summary
        successful_experts = [exp for exp in experts if exp in results and isinstance(results[exp], dict) and 'error' not in results[exp]]
        logger.info(
            f"Parallel inference completed in {total_time:.2f}s: "
            f"{len(successful_experts)}/{len(experts)} experts successful"
        )

        if errors:
            logger.warning(f"Parallel inference errors: {errors}")

        return results

    # ----------------------
    # Module-level convenience functions
    # ----------------------


    # ----------------------
    # Expert-specific decoding parameters
    # ----------------------
    def _get_expert_decoding_params(self, mode: str, temperature: Optional[float], 
                                  top_p: Optional[float], do_sample: Optional[bool]) -> Tuple[float, float, bool]:
        """Get expert-specific decoding parameters based on mode.
        
        Args:
            mode: Expert mode (engine, tutor, director)
            temperature: Override temperature (None to use expert default)
            top_p: Override top_p (None to use expert default)
            do_sample: Override do_sample (None to use expert default)
            
        Returns:
            Tuple of (temperature, top_p, do_sample)
        """
        # Expert-specific defaults
        expert_params = {
            "engine": {
                "temperature": 0.0,
                "top_p": 1.0,
                "do_sample": False
            },
            "tutor": {
                "temperature": 0.7,
                "top_p": 0.9,
                "do_sample": True
            },
            "director": {
                "temperature": 0.6,
                "top_p": 0.9,
                "do_sample": True
            }
        }
        
        # Get expert defaults
        normalized_mode = "engine" if mode == "uci" else mode
        expert_defaults = expert_params.get(normalized_mode, expert_params["tutor"])
        
        # Use provided values or expert defaults
        final_temperature = temperature if temperature is not None else expert_defaults["temperature"]
        final_top_p = top_p if top_p is not None else expert_defaults["top_p"]
        final_do_sample = do_sample if do_sample is not None else expert_defaults["do_sample"]
        
        return final_temperature, final_top_p, final_do_sample

    # ----------------------
    # Engine helpers
    # ----------------------
    def _engine_cache_store(self, fen: Optional[str], move: str) -> None:
        try:
            if not fen:
                return
            with self._cache_lock:
                # simple LRU behavior with OrderedDict
                if fen in self._engine_cache:
                    self._engine_cache.pop(fen, None)
                self._engine_cache[fen] = move
                # evict oldest
                while len(self._engine_cache) > self._engine_cache_max:
                    self._engine_cache.popitem(last=False)
        except Exception:
            pass

    def _engine_cache_lookup(self, fen: Optional[str]) -> Optional[str]:
        try:
            if not fen:
                return None
            with self._cache_lock:
                mv = self._engine_cache.get(fen)
                if mv is None:
                    return None
                # refresh LRU order
                self._engine_cache.pop(fen, None)
                self._engine_cache[fen] = mv
                return mv
        except Exception:
            return None

    def _generate_engine_move(self, question: str, prompt_text: str, max_new_tokens: int) -> Optional[str]:
        """Generate an engine-style UCI move using N-best sampling and legality + optional SF re-ranking.

        Returns a move string (uci) when successful, or None to signal fallback.
        """
        try:
            from .uci_utils import extract_fen, extract_first_legal_move_uci
            import chess

            fen = extract_fen(question) or extract_fen(prompt_text)
            if fen:
                cached = self._engine_cache_lookup(fen)
                if cached:
                    return cached

            board = chess.Board(fen) if fen else None

            # Policy: direct scoring of legal moves by log-prob (no sampling)
            if self._engine_policy == 'logprob' and board is not None:
                best = self._engine_policy_logprob(prompt_text, board)
                if best:
                    if fen:
                        self._engine_cache_store(fen, best)
                    return best

            if not self._engine_rerank_enabled:
                return None

            inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)
            logits_processors = None
            if self._engine_constrain_enabled and self._engine_policy == 'sample':
                if self._engine_constrain_mode == 'strict':
                    logits_processors = [self._build_stateful_uci_processor(prompt_len=inputs['input_ids'].shape[1])]
                else:
                    logits_processors = [self._build_uci_logits_processor(prompt_len=inputs['input_ids'].shape[1])]
            # Multi-sample candidates
            n_best = 5
            gen = self.model.generate(
                **inputs,
                max_new_tokens=min(max_new_tokens, 8),
                do_sample=True,
                temperature=0.3,
                top_p=0.95,
                num_return_sequences=n_best,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
                logits_processor=logits_processors,
            )

            # Decode candidates and parse potential moves
            cands: List[str] = []
            if gen is not None and getattr(gen, "shape", None) is not None and gen.shape[0] >= 1:
                for i in range(gen.shape[0]):
                    cand = self.tokenizer.decode(gen[i], skip_special_tokens=True)
                    if cand.startswith(prompt_text):
                        cand = cand[len(prompt_text):].strip()
                    cands.append(cand)

            legal: List[str] = []
            if board is not None:
                for s in cands:
                    mv = extract_first_legal_move_uci(s, board)
                    if mv:
                        legal.append(mv)

            # If we have at least one legal candidate, optionally score with engine
            if legal:
                best = legal[0]
                try:
                    from .chess_engine import ChessEngineManager
                    side = 1 if board.turn == chess.WHITE else -1
                    scores: List[float] = []
                    with ChessEngineManager() as ce:
                        for mv in legal:
                            # validate_move analyses resulting position
                            res = ce.validate_move(board.fen(), mv)
                            sc = res.centipawn_score if res.centipawn_score is not None else 0
                            scores.append(side * float(sc))
                    # pick argmax
                    if scores:
                        best = legal[int(max(range(len(scores)), key=lambda i: scores[i]))]
                except Exception:
                    # Stockfish unavailable; keep first legal
                    pass

                if fen:
                    self._engine_cache_store(fen, best)
                return best

            return None
        except Exception:
            return None

    # ----------------------
    # UCI constrained logits
    # ----------------------
    class _UCILogitsProcessor(LogitsProcessor):
        def __init__(self, tokenizer, allowed_token_ids: set, prompt_len: int):
            self.tokenizer = tokenizer
            self.allowed = allowed_token_ids
            self.prompt_len = prompt_len

        def __call__(self, input_ids, scores):
            try:
                # Mask all tokens not in the whitelist of UCI-friendly pieces
                import torch
                mask = torch.full_like(scores, fill_value=float('-inf'))
                # Set scores for allowed token ids to original values, others to -inf
                # scores shape: [batch, vocab]
                batch, vocab = scores.shape
                idxs = list(self.allowed)
                if idxs:
                    # gather original values
                    keep = scores[:, idxs]
                    mask[:, idxs] = keep
                return mask
            except Exception:
                return scores

    def _build_uci_logits_processor(self, prompt_len: int) -> 'LogitsProcessor':
        # Build a whitelist of tokens whose decoded text is composed only of [a-h1-8qrbn] and length <= 2
        if self._allowed_token_ids_cache is None:
            allowed_chars = set(list('abcdefgh12345678qrbn'))
            ids = set()
            try:
                for tok, tok_id in self.tokenizer.get_vocab().items():
                    # decode single token id
                    try:
                        s = self.tokenizer.convert_tokens_to_string([tok]) if hasattr(self.tokenizer, 'convert_tokens_to_string') else tok
                    except Exception:
                        s = tok
                    s = s.strip()
                    if not s:
                        continue
                    if len(s) <= 2 and all((c in allowed_chars) for c in s.lower()):
                        ids.add(int(tok_id))
            except Exception:
                ids = set()
            self._allowed_token_ids_cache = ids
        return self._UCILogitsProcessor(self.tokenizer, self._allowed_token_ids_cache or set(), prompt_len)

    class _UCIStatefulLogitsProcessor(LogitsProcessor):
        def __init__(self, tokenizer, token_info: Dict[int, str], prompt_len: int, eos_id: Optional[int]):
            self.tok = tokenizer
            self.info = token_info
            self.prompt_len = prompt_len
            self.eos_id = eos_id
            self.allowed_chars = set('abcdefgh12345678qrbn')

        def __call__(self, input_ids, scores):
            try:
                import torch
                # Assume batch size 1; if larger, operate element-wise defaulting to pass-through
                if input_ids.shape[0] != 1:
                    return scores
                gen_ids = input_ids[0, self.prompt_len:]
                text = self.tok.decode(gen_ids, skip_special_tokens=True).lower()
                # Extract only allowed chars from generated tail
                tail = ''.join([c for c in text if c in self.allowed_chars])
                L = len(tail)
                mask = torch.full_like(scores, float('-inf'))
                # Allow EOS when length in [4,5]
                if self.eos_id is not None and 4 <= L <= 5:
                    mask[:, self.eos_id] = scores[:, self.eos_id]
                # Allow tokens whose clean text keeps length <=5 and only allowed chars
                for tid, s in self.info.items():
                    if not s:
                        continue
                    nL = L + len(s)
                    if nL <= 5:
                        mask[:, tid] = scores[:, tid]
                return mask
            except Exception:
                return scores

    def _build_stateful_uci_processor(self, prompt_len: int) -> 'LogitsProcessor':
        # Build token_info mapping id -> cleaned char sequence for UCI characters
        if self._uci_token_info is None:
            info: Dict[int, str] = {}
            allowed_chars = set('abcdefgh12345678qrbn')
            try:
                vocab = self.tokenizer.get_vocab()
                for tok, tok_id in vocab.items():
                    try:
                        s = self.tokenizer.convert_tokens_to_string([tok]) if hasattr(self.tokenizer, 'convert_tokens_to_string') else tok
                    except Exception:
                        s = tok
                    s = ''.join([c for c in s.lower() if c in allowed_chars])
                    # Keep only 1-2 length fragments to avoid large jumps
                    if 0 < len(s) <= 2:
                        info[int(tok_id)] = s
            except Exception:
                info = {}
            self._uci_token_info = info
        eos_id = self.tokenizer.eos_token_id
        return self._UCIStatefulLogitsProcessor(self.tokenizer, self._uci_token_info or {}, prompt_len, eos_id)

    # Public helper for activating adapter from an explicit checkpoint path
    def activate_adapter_from_path(self, logical_name: str, adapter_path: str) -> bool:
        try:
            p = Path(adapter_path)
            if not p.exists() or not p.is_dir():
                return False
            self._ensure_adapter_loaded(logical_name, p)
            self.set_active_adapter(logical_name)
            return True
        except Exception:
            return False

    def _engine_policy_logprob(self, prompt_text: str, board: 'chess.Board') -> Optional[str]:
        """Score each legal UCI move by conditional log-prob under the model and return argmax.

        Batched implementation for efficiency. Computes average NLL over target tokens.
        """
        import torch
        import torch.nn.functional as F
        import chess
        legal_moves = [m.uci() for m in board.legal_moves]
        if not legal_moves:
            return None
        device = self.model.device
        # Tokenize prompt once
        prompt_ids = self.tokenizer(prompt_text, return_tensors='pt').to(device)

        # Tokenize all targets and batch
        tgt_tok = [self.tokenizer(mv, add_special_tokens=False, return_tensors='pt') for mv in legal_moves]
        tgt_lens = [t['input_ids'].shape[1] for t in tgt_tok]
        max_len = max(tgt_lens)

        # Build batched inputs by left-padding targets to align the ends
        def left_pad(t: 'dict'):
            pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id or 0
            cur = t['input_ids']
            pad = max_len - cur.shape[1]
            if pad > 0:
                pad_tensor = torch.full((1, pad), pad_id, dtype=cur.dtype)
                t_ids = torch.cat([pad_tensor, cur], dim=1)
                attn = torch.cat([torch.zeros((1, pad), dtype=torch.long), torch.ones_like(cur)], dim=1)
            else:
                t_ids = cur
                attn = torch.ones_like(cur)
            return {'input_ids': t_ids.to(device), 'attention_mask': attn.to(device)}

        tgt_batch = [left_pad(t) for t in tgt_tok]
        batch_input_ids = torch.cat([torch.cat([prompt_ids['input_ids'], tb['input_ids']], dim=1) for tb in tgt_batch], dim=0)
        batch_attn = torch.cat([torch.cat([prompt_ids['attention_mask'], tb['attention_mask']], dim=1) for tb in tgt_batch], dim=0)

        with torch.no_grad():
            outputs = self.model(input_ids=batch_input_ids, attention_mask=batch_attn)
            logits = outputs.logits  # [B, T, V]
            # Extract logits aligned to last max_len positions
            last_logits = logits[:, -max_len:, :]
            # Build target ids batch aligned to end
            tgt_ids_batch = torch.stack([tb['input_ids'].squeeze(0) for tb in tgt_batch], dim=0)
            # Compute token-wise negative log-prob only for valid (non-pad) positions
            log_probs = F.log_softmax(last_logits, dim=-1)
            # Gather log-probs at target tokens
            gathered = torch.gather(log_probs, dim=-1, index=tgt_ids_batch.unsqueeze(-1)).squeeze(-1)
            # Mask out padded positions
            masks = torch.stack([tb['attention_mask'].squeeze(0) for tb in tgt_batch], dim=0).to(gathered.dtype)
            masks = masks[:, -max_len:]
            # Sum log-probs over valid tokens and normalize by length
            sum_lp = (gathered * masks).sum(dim=1)
            lengths = masks.sum(dim=1).clamp(min=1)
            avg_lp = sum_lp / lengths
            # Select argmax
            best_idx = int(torch.argmax(avg_lp).item())
            return legal_moves[best_idx] if 0 <= best_idx < len(legal_moves) else None

    def generate_text(
        self,
        prompt: str,
        max_new_tokens: int = 200,
        do_sample: bool = True,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
    ) -> str:
        """Generate raw text from a direct prompt string (no chat template).

        Decoding parameters are explicitly configurable for expert-specific needs.
        """
        # Try to use new modular architecture first
        if hasattr(self, '_core_engine'):
            return self._core_engine.generate_text(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty
            )

        if not self.load_model():
            return ""
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    top_p=top_p,
                    temperature=temperature,
                    repetition_penalty=repetition_penalty,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception:
            return ""

    def get_model_info(self) -> Dict[str, Any]:
        device = str(next(self.model.parameters()).device) if (self.model is not None) else "unknown"
        info = {
            "base_model": str(self.model_path) if self.model_path else None,
            "adapter_path": str(self.adapter_path) if self.adapter_path else None,
            "is_loaded": self.is_loaded,
            "device": device,
            "active_adapter": self._active_adapter,
            "available_adapters": {
                k: str(v) for k, v in self._adapter_paths.items()
            },
            "moe_enabled": self.moe_enabled,
            "moe_available": MOE_AVAILABLE,
        }

        # Add MoE information if available
        if self.moe_enabled and self.moe_router:
            info["moe_info"] = self.moe_router.get_routing_stats()
            info["moe_experts"] = list(self._expert_paths.keys())

        return info

    def clear_caches(self):
        """Clear all performance caches."""
        with self._cache_lock:
            self._response_cache.clear()
            self._cache_hits = 0
        self._kv_cache.clear()
        logger.info("🧹 Inference caches cleared")

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        with self._cache_lock:
            total_requests = self._total_requests
            cache_hits = self._cache_hits
            response_cache_size = len(self._response_cache)
            engine_cache_size = len(self._engine_cache)
        return {
            'total_requests': total_requests,
            'cache_hits': cache_hits,
            'cache_hit_rate': cache_hits / max(total_requests, 1),
            'response_cache_size': response_cache_size,
            'engine_cache_size': engine_cache_size,
            'generation_stats': self._generation_stats.copy(),
            'cache_max_size': self._cache_max_size,
            'memory_efficiency': self._estimate_cache_memory_usage()
        }

    def _estimate_cache_memory_usage(self) -> float:
        """Estimate memory usage of caches in MB."""
        # Rough estimation: each cached response is ~2KB
        with self._cache_lock:
            cache_entries = len(self._response_cache) + len(self._engine_cache)
        return cache_entries * 2048 / (1024 * 1024)  # Convert to MB

    def optimize_generation_config(self, mode: str) -> Dict[str, Any]:
        """Get optimized generation configuration for different modes."""
        base_config = {
            'do_sample': True,
            'pad_token_id': self.tokenizer.eos_token_id,
            'eos_token_id': self.tokenizer.eos_token_id,
            'use_cache': True,  # Enable KV caching
        }

        if mode == "engine":
            # Fast, deterministic generation for moves
            return {
                **base_config,
                'max_new_tokens': 8,  # UCI moves are short
                'temperature': 0.1,   # Low temperature for consistency
                'top_p': 0.9,
                'do_sample': False,   # Deterministic for engine mode
                'repetition_penalty': 1.0
            }
        elif mode == "tutor":
            # Balanced generation for explanations
            return {
                **base_config,
                'max_new_tokens': 150,
                'temperature': 0.7,
                'top_p': 0.9,
                'repetition_penalty': 1.1
            }
        else:  # director
            # Creative generation for Q&A
            return {
                **base_config,
                'max_new_tokens': 200,
                'temperature': 0.8,
                'top_p': 0.95,
                'repetition_penalty': 1.2
            }


_INFERENCE_SINGLETON: Optional[ChessGemmaInference] = None


def get_inference_instance() -> ChessGemmaInference:
    global _INFERENCE_SINGLETON
    if _INFERENCE_SINGLETON is None:
        _INFERENCE_SINGLETON = ChessGemmaInference()
    return _INFERENCE_SINGLETON


def run_inference(question: str) -> Dict[str, Any]:
    instance = get_inference_instance()
    return instance.generate_response(question)


def load_model() -> bool:
    instance = get_inference_instance()
    return instance.load_model()


def unload_model() -> None:
    instance = get_inference_instance()
    return instance.unload_model()


def get_model_info() -> Dict[str, Any]:
    instance = get_inference_instance()
    return instance.get_model_info()


# Enhanced Inference Integration
# ==============================

# Enhanced inference with MoE support
def get_enhanced_inference_manager():
    """Get enhanced inference manager instance (now uses MoE when available)."""
    return get_inference_instance()

def initialize_enhanced_inference() -> bool:
    """Initialize enhanced inference system with MoE support."""
    instance = get_inference_instance()
    return instance.load_model()

def analyze_chess_position(fen: str, mode: str = "tutor") -> Dict[str, Any]:
    """Enhanced position analysis using MoE routing when available."""
    instance = get_inference_instance()
    question = f"FEN: {fen}\nAnalyze this position."
    return instance.generate_response(question, mode=mode)

def generate_best_move(fen: str) -> Dict[str, Any]:
    """Enhanced best move generation with MoE routing."""
    instance = get_inference_instance()
    question = f"FEN: {fen}\nWhat is the best move?"
    return instance.generate_response(question, mode="engine")

def switch_inference_expert(expert_name: str) -> bool:
    """Switch to a different expert adapter (legacy function, now uses MoE routing)."""
    instance = get_inference_instance()
    if instance.moe_enabled:
        # MoE handles expert switching automatically
        return True
    else:
        # Fall back to manual adapter switching
        instance.set_active_adapter(expert_name)
        return True

def get_inference_stats() -> Dict[str, Any]:
    """Get inference performance statistics including MoE metrics."""
    instance = get_inference_instance()
    info = instance.get_model_info()

    stats = {
        "model_loaded": info.get("is_loaded", False),
        "moe_enabled": info.get("moe_enabled", False),
        "device": info.get("device", "unknown"),
        "active_adapter": info.get("active_adapter"),
    }

    # Add MoE stats if available
    if info.get("moe_enabled") and "moe_info" in info:
        moe_info = info["moe_info"]
        stats.update({
            "moe_experts": info.get("moe_experts", []),
            "moe_routing_stats": moe_info.get("routing_parameters", {}),
            "expert_performance": moe_info.get("expert_performance", {}),
        })

    return stats

# Make enhanced functions available at module level
__all__.extend([
    'get_enhanced_inference_manager',
    'initialize_enhanced_inference',
    'analyze_chess_position',
    'generate_best_move',
    'switch_inference_expert',
    'get_inference_stats'
])


class ChessModelInterface:
    """Thin wrapper used by the UCI bridge to get raw text from a prompt."""

    def __init__(self, model_path: Optional[str] = None, adapter_path: Optional[str] = None):
        self._inference = ChessGemmaInference(model_path, adapter_path)

    def generate_response(self, prompt: str) -> str:
        cache_lock = getattr(self._inference, "cache_lock", None)
        lock_context = cache_lock if cache_lock is not None else nullcontext()
        with lock_context:
            return self._inference.generate_text(prompt)
