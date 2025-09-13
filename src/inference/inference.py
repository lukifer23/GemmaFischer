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
from pathlib import Path
from typing import Optional, Dict, Any, List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from transformers.generation.logits_process import LogitsProcessor
from collections import OrderedDict

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


def _resolve_default_model_path(project_root: Path) -> Optional[Path]:
    """Resolve the local base model snapshot directory under models/."""
    base = project_root / "models" / "unsloth-gemma-3-270m-it" / "models--unsloth--gemma-3-270m-it" / "snapshots"
    if not base.exists():
        return None
    # Pick latest snapshot directory
    subdirs = [p for p in base.iterdir() if p.is_dir()]
    if not subdirs:
        return None
    subdirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return subdirs[0]


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
    return latest


class ChessGemmaInference:
    """Unified inference with optional adapter and dual-mode prompting."""

    def __init__(self, model_path: Optional[str] = None, adapter_path: Optional[str] = None):
        self.project_root = Path(__file__).resolve().parents[2]
        # Preserve provided strings for tests; otherwise resolve defaults
        self.model_path = model_path if model_path else _resolve_default_model_path(self.project_root)
        self.adapter_path = adapter_path if adapter_path else _resolve_latest_adapter_path(self.project_root)

        self.tokenizer = None
        self.model = None
        self.is_loaded = False

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

        # Simple memoization for deterministic engine prompts (FEN -> move)
        self._engine_cache_max = 512
        self._engine_cache: "OrderedDict[str, str]" = OrderedDict()
        # Feature flags
        self._engine_rerank_enabled = (os.environ.get('CHESSGEMMA_ENGINE_RERANK', '1') not in ('0', 'false', 'False'))
        self._engine_policy = os.environ.get('CHESSGEMMA_ENGINE_POLICY', 'sample').strip().lower()  # sample | logprob
        self._engine_constrain_enabled = (os.environ.get('CHESSGEMMA_ENGINE_CONSTRAIN', '0') not in ('0','false','False'))
        self._engine_constrain_mode = os.environ.get('CHESSGEMMA_ENGINE_CONSTRAIN_MODE', 'simple').strip().lower()
        self._allowed_token_ids_cache: Optional[set] = None
        self._uci_token_info: Optional[Dict[int, str]] = None

        # MoE Router integration
        self.moe_router: Optional[ChessMoERouter] = None
        self.moe_manager: Optional[MoEInferenceManager] = None
        self.moe_enabled = MOE_AVAILABLE and (os.environ.get('CHESSGEMMA_MOE_ENABLED', '1') not in ('0', 'false', 'False'))
        self._expert_paths: Dict[str, str] = {}

        # Initialize MoE if available and enabled
        if self.moe_enabled and MOE_AVAILABLE:
            self._initialize_moe_system()

    def _initialize_moe_system(self) -> None:
        """Initialize the Mixture of Experts system."""
        try:
            # Set up expert paths for MoE
            checkpoints_root = self.project_root / "checkpoints"
            self._expert_paths = {
                'uci': str(_find_latest_dir([str(checkpoints_root / "lora_full" / "checkpoint-*")])),  # Use full model for UCI
                'tutor': str(_find_latest_dir([str(checkpoints_root / "lora_tutor" / "checkpoint-*")])),
                'director': str(_find_latest_dir([str(checkpoints_root / "lora_director" / "checkpoint-*")]))
            }

            # Filter out None values
            self._expert_paths = {k: v for k, v in self._expert_paths.items() if v is not None}

            if len(self._expert_paths) >= 2:  # Need at least 2 experts for MoE
                self.moe_router = ChessMoERouter(num_experts=len(self._expert_paths))
                self.moe_manager = MoEInferenceManager(self.moe_router, self._expert_paths, self)
                print(f"🧠 MoE System initialized with {len(self._expert_paths)} experts: {list(self._expert_paths.keys())}")
            else:
                print("⚠️  Insufficient expert checkpoints for MoE (need at least 2), falling back to single-expert mode")
                self.moe_enabled = False

        except Exception as e:
            print(f"⚠️  Failed to initialize MoE system: {e}")
            self.moe_enabled = False

    def load_model(self) -> bool:
        """Lazily load tokenizer and model (MPS/Auto device)."""
        if self.is_loaded and self.model is not None and self.tokenizer is not None:
            return True

        try:
            if not self.model_path or not Path(self.model_path).exists():
                print("Model snapshot not found under models/. Ensure weights are downloaded locally.")
                return False

            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path), local_files_only=True)
            base_model = AutoModelForCausalLM.from_pretrained(
                str(self.model_path),
                local_files_only=True,
                device_map="auto",
                attn_implementation="eager",
            )

            # Try to apply adapter if available; on failure, fall back to base model
            applied_adapter = False
            try:
                if self.adapter_path and Path(self.adapter_path).exists():
                    self.model = PeftModel.from_pretrained(base_model, str(self.adapter_path), is_trainable=False)
                    applied_adapter = True
                    self._active_adapter = "default"
                    self._loaded_adapters["default"] = True
            except Exception:
                self.model = base_model
            if not applied_adapter:
                self.model = self.model or base_model

            self.model.eval()
            # Discover known expert adapters on disk for quick switching
            self.refresh_adapters()
            self.is_loaded = True
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            self.is_loaded = False
            return False

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
            "uci": [checkpoints_root / "lora_full"],  # Use ONLY full model for UCI (2000 steps vs 1000)
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
        if mode == "engine":
            if self._engine_template is None:
                # Clean, minimal prompt for move generation
                self._engine_template = "Find the best chess move in UCI format."
            return self._engine_template
        elif mode == "tutor":
            if self._tutor_template is None:
                # Clean tutor prompt focused on analysis
                self._tutor_template = (
                    "You are a chess tutor. Analyze the position and explain your reasoning. "
                    "End with: Best move: <uci>"
                )
            return self._tutor_template
        else:
            # Director default - clean Q&A prompt
            return "You are a chess expert. Answer questions accurately and concisely."

    def _build_messages(self, question: str, context: Optional[str], mode: str) -> List[Dict[str, str]]:
        system_prompt = self._load_prompt_template(mode)

        if mode == "tutor":
            full_question = f"{context}\n\n{question}" if context else question
            prompt = f"{system_prompt}\n\n{full_question}"
            return [{"role": "user", "content": prompt}]
        if mode == "director":
            full_question = f"{context}\n\n{question}" if context else question
            prompt = f"{system_prompt}\n\n{full_question}"
            return [{"role": "user", "content": prompt}]

        # Engine mode: clean FEN + minimal instruction
        from .uci_utils import extract_fen
        fen = extract_fen(question) or (context or "").strip()
        prompt = f"FEN: {fen}\n{system_prompt}" if fen else system_prompt
        return [{"role": "user", "content": prompt}]

    def generate_response(
        self,
        question: str,
        context: Optional[str] = None,
        mode: str = "tutor",
        max_new_tokens: int = 200,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> Dict[str, Any]:
        """Generate a response dict (text + simple confidence)."""
        if not self.is_loaded:
            return {
                "error": "Model not loaded",
                "response": "",
                "confidence": 0.0,
                "model_loaded": False,
            }

        # Use MoE routing if available and enabled
        if self.moe_enabled and self.moe_manager and mode in ['tutor', 'engine', 'director']:
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
                    return {
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
            except Exception as e:
                print(f"MoE routing failed, falling back to standard inference: {e}")
                # Fall through to standard inference

        try:
            messages = self._build_messages(question, context, mode)
            prompt_text: str

            # Use the simple prompt format we built
            prompt_text = messages[0]['content']

            # Debug logging
            print(f"\n🔍 INFERENCE DEBUG:")
            print(f"Mode: {mode}")
            print(f"Question: {question[:100]}{'...' if len(question) > 100 else ''}")
            print(f"System Prompt: {messages[0]['content'][:100]}{'...' if len(messages[0]['content']) > 100 else ''}")
            print(f"Prompt Length: {len(prompt_text)} chars")

            inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)

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
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        top_p=top_p,
                        temperature=temperature,
                        pad_token_id=self.tokenizer.eos_token_id,
                        repetition_penalty=1.1,
                        eos_token_id=self.tokenizer.eos_token_id,
                        use_cache=True,
                    )

            if mode == "engine" and outputs is None:
                decoded = decoded  # already built above (prompt + answer)
            else:
                decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Debug logging for response
            print(f"Raw Response Length: {len(decoded)} chars")
            print(f"Raw Response Preview: {decoded[:300]}{'...' if len(decoded) > 300 else ''}")
            print(f"Raw Response Full: {decoded}")
            
            # Try to strip prompt prefix if echoed
            if decoded.startswith(prompt_text):
                answer = decoded[len(prompt_text):].strip()
                print(f"Stripped prompt prefix, answer length: {len(answer)}")
            else:
                answer = decoded.strip()
                print(f"No prompt prefix found, using full response")
            
            # Clean up common artifacts
            if answer.startswith("Answer:"):
                answer = answer[7:].strip()
            elif answer.startswith("Move:"):
                answer = answer[5:].strip()
            
            # Remove any remaining prompt fragments
            lines = answer.split('\n')
            content_lines = []
            print(f"🔍 Processing {len(lines)} lines from model response")
            for i, line in enumerate(lines):
                line = line.strip()
                print(f"  Line {i}: '{line[:50]}{'...' if len(line) > 50 else ''}'")
                if line and not line.startswith(('Chess Tutor:', 'Chess Engine:', 'Question:', 'Position:', 'Answer:', 'Move:')):
                    content_lines.append(line)
                    print(f"    ✅ Kept line {i}")
                else:
                    print(f"    ❌ Filtered out line {i}")
            
            if content_lines:
                answer = '\n'.join(content_lines).strip()
                print(f"Final processed answer: '{answer[:100]}{'...' if len(answer) > 100 else ''}'")
            else:
                print("⚠️  All lines were filtered out!")
            
            # Fallback if we still don't have a good answer
            if not answer or len(answer) < 10:
                if mode == "tutor":
                    answer = "I'm having trouble generating a response. Please try rephrasing your question or ask about a specific chess position."
                else:
                    answer = ""  # Defer to engine fallback
                print("Using fallback response due to poor model output")
            
            print(f"Final Answer Preview: {answer[:200]}{'...' if len(answer) > 200 else ''}")

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

            return {
                "response": answer,
                "confidence": confidence,
                "model_loaded": True,
                "mode": mode,
                "postprocessed": postprocessed,
                "prompt_len_chars": len(prompt_text),
                "answer_len_chars": len(answer),
            }
        except Exception as e:
            return {
                "error": str(e),
                "response": "",
                "confidence": 0.0,
                "model_loaded": True,
                "mode": mode,
            }

    # ----------------------
    # Engine helpers
    # ----------------------
    def _engine_cache_store(self, fen: Optional[str], move: str) -> None:
        try:
            if not fen:
                return
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
        return self._inference.generate_text(prompt)
