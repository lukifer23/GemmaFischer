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
            except Exception:
                self.model = base_model
            if not applied_adapter:
                self.model = self.model or base_model

            self.model.eval()
            self.is_loaded = True
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            self.is_loaded = False
            return False

    def _load_prompt_template(self, mode: str) -> str:
        """Load prompt template from prompts directory, fallback to defaults."""
        if mode == "engine":
            if self._engine_template is None:
                path = self.project_root / "prompts" / "engine_mode.txt"
                self._engine_template = path.read_text(encoding="utf-8") if path.exists() else (
                    "You are a chess engine. Output only the best move in UCI format (e2e4)."
                )
            return self._engine_template
        else:
            if self._tutor_template is None:
                path = self.project_root / "prompts" / "tutor_mode.txt"
                self._tutor_template = path.read_text(encoding="utf-8") if path.exists() else (
                    "You are a chess tutor. Analyze step-by-step and conclude with the best move in UCI."
                )
            return self._tutor_template

    def _build_messages(self, question: str, context: Optional[str], mode: str) -> List[Dict[str, str]]:
        system_prompt = self._load_prompt_template(mode)
        msgs = [
            {"role": "system", "content": system_prompt},
        ]
        if context:
            msgs.append({"role": "user", "content": f"Context: {context}"})
        msgs.append({"role": "user", "content": question})
        return msgs

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

            if hasattr(self.tokenizer, "apply_chat_template"):
                prompt_text = self.tokenizer.apply_chat_template(messages, tokenize=False)
            else:
                # Basic fallback: concatenate
                prompt_text = "\n".join([f"{m['role']}: {m['content']}" for m in messages])

            inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    top_p=top_p,
                    temperature=temperature,
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                )

            decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Try to strip prompt prefix if echoed
            answer = decoded[len(prompt_text):].strip() if decoded.startswith(prompt_text) else decoded.strip()

            # Post-process for engine mode: enforce a single legal UCI move if possible
            postprocessed = False
            if mode == 'engine':
                import re, chess
                try:
                    # Try to extract first UCI-looking token
                    pattern = r"\b([a-h][1-8][a-h][1-8][qrbn]?)\b"
                    m = re.findall(pattern, answer.lower())
                    if m:
                        # Validate against a board if question contains a FEN line
                        fen_match = re.search(r"position:\s*([^\n]+)", question, re.IGNORECASE)
                        mv = m[0]
                        if fen_match:
                            fen = fen_match.group(1).strip()
                            try:
                                board = chess.Board(fen)
                                move_obj = chess.Move.from_uci(mv)
                                if move_obj not in board.legal_moves:
                                    mv = None
                            except Exception:
                                pass
                        if mv:
                            answer = mv
                            postprocessed = True
                except Exception:
                    pass

            # Simple heuristic confidence
            word_count = len(answer.split())
            confidence = max(0.1, min(0.95, word_count / 60.0))

            return {
                "response": answer,
                "confidence": confidence,
                "model_loaded": True,
                "mode": mode,
                "postprocessed": postprocessed,
            }
        except Exception as e:
            return {
                "error": str(e),
                "response": "",
                "confidence": 0.0,
                "model_loaded": True,
                "mode": mode,
            }

    def generate_text(self, prompt: str, max_new_tokens: int = 200) -> str:
        """Generate raw text from a direct prompt string (no chat template)."""
        if not self.load_model():
            return ""
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    top_p=0.9,
                    temperature=0.7,
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

