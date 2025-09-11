"""
Unified inference interface for ChessGemma.

Provides a single entry-point used by tests, UCI bridge, and the web app:
- ChessGemmaInference: lazy local-only model loading (MPS), adapter application, and generation
- ChessModelInterface: thin wrapper returning raw text (for UCI bridge)
- Convenience module functions: get_inference_instance, run_inference, load_model, get_model_info
"""

from __future__ import annotations

import os
import glob
from pathlib import Path
from typing import Optional, Dict, Any, List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


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
        from collections import OrderedDict  # local import to avoid top-level pollution
        self._engine_cache_max = 512
        self._engine_cache: "OrderedDict[str, str]" = OrderedDict()
        # Feature flags
        self._engine_rerank_enabled = (os.environ.get('CHESSGEMMA_ENGINE_RERANK', '1') not in ('0', 'false', 'False'))

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

    def _discover_adapter_paths(self) -> None:
        """Populate mapping of expert names to latest adapter checkpoint paths.

        Primary targets are expert-specific directories. If none found for an expert,
        fall back to generic LoRA runs so the system still functions, with visibility
        into which checkpoint is actually used.
        """
        checkpoints_root = self.project_root / "checkpoints"

        # Primary expert-specific locations
        primary = {
            "uci": [checkpoints_root / "lora_uci"],
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
                # Use a proper system prompt instead of the documentation file
                self._engine_template = (
                    "You are a chess engine. Analyze the given position and respond with only the best move in UCI format (e.g., e2e4). "
                    "Do not provide explanations, just the move."
                )
            return self._engine_template
        elif mode == "tutor":
            if self._tutor_template is None:
                # Use a proper system prompt instead of the documentation file
                self._tutor_template = (
                    "You are a chess tutor and analyst playing a teaching game with a student. "
                    "You MUST be aware of the current board position, material balance, and tactical situation. "
                    "When making moves, explain your reasoning based on the ACTUAL position, comment on the opponent's previous move, "
                    "and suggest what they should consider next. Be conversational and educational, "
                    "like a chess teacher playing a teaching game. Always provide clear explanations based on the real position. "
                    "CRITICAL: End your reply with a single line of the form: Best move: <uci> (e.g., Best move: e2e4)."
                )
            return self._tutor_template
        else:
            # Director default
            return (
                "You are a concise, accurate chess teacher. Answer clearly, avoid fabrications, cite rules or standard principles when relevant."
            )

    def _build_messages(self, question: str, context: Optional[str], mode: str) -> List[Dict[str, str]]:
        system_prompt = self._load_prompt_template(mode)

        if mode == "tutor":
            full_question = f"{context}\n\n{question}" if context else question
            prompt = f"Chess Tutor: {system_prompt}\n\nQuestion: {full_question}\n\nAnswer:"
            return [{"role": "user", "content": prompt}]
        if mode == "director":
            full_question = f"{context}\n\n{question}" if context else question
            sys_prompt = self._load_prompt_template("director")
            prompt = (
                f"Chess Director: {sys_prompt}\n\n"
                f"Question: {full_question}\n\n"
                "Answer:"
            )
            return [{"role": "user", "content": prompt}]

        # Engine mode expects strict minimal prompt: FEN then "Move:"
        from .uci_utils import build_engine_prompt, extract_fen
        fen = extract_fen(question) or (context or "").strip()
        prompt = build_engine_prompt(fen) if fen else "Move:"
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
                    # Engine mode: try cached + N-best sampling with legality + Stockfish re-ranking
                    answer = self._generate_engine_move(question, messages[0]['content'], max_new_tokens)
                    if answer:
                        decoded = messages[0]['content'] + answer
                        outputs = None  # bypass default path
                    else:
                        # Fallback to deterministic single-shot decoding
                        outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=max_new_tokens,
                            do_sample=False,
                            temperature=0.0,
                            top_p=1.0,
                            pad_token_id=self.tokenizer.eos_token_id,
                            eos_token_id=self.tokenizer.eos_token_id,
                            use_cache=True,
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
            if not self._engine_rerank_enabled:
                return None
            from .uci_utils import extract_fen, extract_first_legal_move_uci
            import chess

            fen = extract_fen(question) or extract_fen(prompt_text)
            if fen:
                cached = self._engine_cache_lookup(fen)
                if cached:
                    return cached

            board = chess.Board(fen) if fen else None

            inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)
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
        return {
            "base_model": str(self.model_path) if self.model_path else None,
            "adapter_path": str(self.adapter_path) if self.adapter_path else None,
            "is_loaded": self.is_loaded,
            "device": device,
            "active_adapter": self._active_adapter,
            "available_adapters": {
                k: str(v) for k, v in self._adapter_paths.items()
            },
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


def get_model_info() -> Dict[str, Any]:
    instance = get_inference_instance()
    return instance.get_model_info()


class ChessModelInterface:
    """Thin wrapper used by the UCI bridge to get raw text from a prompt."""

    def __init__(self, model_path: Optional[str] = None, adapter_path: Optional[str] = None):
        self._inference = ChessGemmaInference(model_path, adapter_path)

    def generate_response(self, prompt: str) -> str:
        return self._inference.generate_text(prompt)
