#!/usr/bin/env python3
"""
Expert Management System for ChessGemma

Handles UCI, Tutor, and Director expert adapters with dynamic loading and switching.
Provides clean interface for expert-specific inference operations.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import threading

# Import common utilities
from src.utils.common import (
    get_logger, get_config_manager, find_latest_dir,
    safe_import, conditional_import
)

# Get utility functions
logger = get_logger(__name__)
config_manager = get_config_manager()

# Import core engine
try:
    from .core_engine import ChessGemmaCoreEngine
except ImportError:
    # Fallback for standalone testing
    ChessGemmaCoreEngine = None


class ChessExpertManager:
    """Manages UCI, Tutor, and Director expert adapters."""

    def __init__(self, core_engine: Optional[ChessGemmaCoreEngine] = None, chess_engine: Optional[ChessEngineManager] = None):
        self.core_engine = core_engine or ChessGemmaCoreEngine()
        self.chess_engine = chess_engine  # For UCI mode hybrid engine usage
        self.project_root = Path(__file__).resolve().parents[2]

        # Adapter management
        self._adapter_paths: Dict[str, Path] = {}
        self._loaded_adapters: Dict[str, bool] = {}
        self._logical_to_physical: Dict[str, str] = {}
        self._adapter_loaded_from: Dict[str, Path] = {}
        self._active_adapter: Optional[str] = None

        # Expert-specific configurations - load from unified config if available
        if config_manager:
            try:
                config = config_manager()
                inference_config = config.get_inference_config()

                self._expert_configs = {
                    "uci": {
                        "temperature": inference_config.get("engine_temperature", 0.0),
                        "top_p": inference_config.get("engine_top_p", 1.0),
                        "do_sample": inference_config.get("engine_do_sample", False),
                        "max_new_tokens": inference_config.get("engine_max_new_tokens", 8)
                    },
                    "tutor": {
                        "temperature": inference_config.get("tutor_temperature", 0.7),
                        "top_p": inference_config.get("tutor_top_p", 0.9),
                        "do_sample": inference_config.get("tutor_do_sample", True),
                        "max_new_tokens": inference_config.get("tutor_max_new_tokens", 150)
                    },
                    "director": {
                        "temperature": inference_config.get("director_temperature", 0.6),
                        "top_p": inference_config.get("director_top_p", 0.9),
                        "do_sample": inference_config.get("director_do_sample", True),
                        "max_new_tokens": inference_config.get("director_max_new_tokens", 200)
                    }
                }
            except:
                # Fall back to hardcoded defaults
                self._expert_configs = {
                    "uci": {
                        "temperature": 0.0,
                        "top_p": 1.0,
                        "do_sample": False,
                        "max_new_tokens": 8
                    },
                    "tutor": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "do_sample": True,
                        "max_new_tokens": 150
                    },
                    "director": {
                        "temperature": 0.6,
                        "top_p": 0.9,
                        "do_sample": True,
                        "max_new_tokens": 200
                    }
                }
        else:
            # Use hardcoded defaults
            self._expert_configs = {
                "uci": {
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "do_sample": False,
                    "max_new_tokens": 8
                },
                "tutor": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "do_sample": True,
                    "max_new_tokens": 150
                },
                "director": {
                    "temperature": 0.6,
                    "top_p": 0.9,
                    "do_sample": True,
                    "max_new_tokens": 200
                }
            }

        # Initialize adapter discovery
        self.refresh_adapters()

    def _discover_adapter_paths(self) -> None:
        """Populate mapping of expert names to latest adapter checkpoint paths."""
        checkpoints_root = self.project_root / "checkpoints"

        # Primary expert-specific locations - prioritize better trained models
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
            checkpoints_root / "lora_formatted",
        ]

        self._adapter_paths.clear()
        for expert, primary_dirs in primary.items():
            latest = find_latest_dir([str(d / "checkpoint-*") for d in primary_dirs if d.exists()])
            if latest is None:
                # Try fallbacks
                latest = find_latest_dir([str(d / "checkpoint-*") for d in fallback_common if d.exists()])
            if latest is not None:
                self._adapter_paths[expert] = latest

    def _ensure_adapter_loaded(self, logical_name: str, path: Path) -> None:
        """Load adapter weights if not already loaded from this path."""
        if not hasattr(self.core_engine.model, "load_adapter"):
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
            self.core_engine.model.load_adapter(str(path), adapter_name=physical_name)
            self._loaded_adapters[physical_name] = True
            self._logical_to_physical[logical_name] = physical_name
            self._adapter_loaded_from[logical_name] = path
        except Exception:
            pass

    def refresh_adapters(self) -> None:
        """Re-discover latest checkpoints and ensure corresponding adapters are loaded."""
        self._discover_adapter_paths()
        for logical_name, path in self._adapter_paths.items():
            self._ensure_adapter_loaded(logical_name, path)

    def set_active_adapter(self, name: Optional[str]) -> None:
        """Switch the active LoRA adapter by logical name (uci/tutor/director)."""
        if not name:
            return

        # Refresh to capture newly saved checkpoints
        self.refresh_adapters()

        # Ensure requested logical adapter is loaded
        path = self._adapter_paths.get(name)
        if path is not None:
            self._ensure_adapter_loaded(name, path)

        # Resolve to physical adapter name
        physical = self._logical_to_physical.get(name)
        if hasattr(self.core_engine.model, "set_adapter") and physical and self._loaded_adapters.get(physical):
            try:
                self.core_engine.model.set_adapter(physical)
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

    def activate_adapter_from_path(self, logical_name: str, adapter_path: str) -> bool:
        """Activate adapter from an explicit checkpoint path."""
        try:
            p = Path(adapter_path)
            if not p.exists() or not p.is_dir():
                return False
            self._ensure_adapter_loaded(logical_name, p)
            self.set_active_adapter(logical_name)
            return True
        except Exception:
            return False

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
            file_template = _load_from_file("engine_mode.txt", matcher="Mode: Engine")
            return file_template or "Find the best chess move in UCI format."
        elif mode == "tutor":
            file_template = _load_from_file("tutor_mode.txt", matcher="Mode: Tutor")
            if file_template:
                return file_template
            else:
                # Simplified fallback template for better model response
                return "Analyze this chess position and explain the best move."
        else:
            file_template = _load_from_file("director_mode.txt", matcher="Mode: Director")
            if file_template:
                return file_template
            # Director default - clean Q&A prompt
            return "You are a chess expert. Answer this question:"

    def _build_messages(self, question: str, context: Optional[str], mode: str) -> List[Dict[str, str]]:
        """Build messages for the given mode."""
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
        """Generate response using the specified expert."""
        # Map UCI to engine mode
        mode = "engine" if expert_mode == "uci" else expert_mode

        # For UCI mode, try hybrid engine first if available
        if expert_mode == "uci" and self.chess_engine:
            try:
                # Use hybrid engine for UCI moves
                import chess
                # Extract FEN from context or question if possible
                fen = context or question
                if "FEN:" in fen:
                    fen = fen.split("FEN:")[1].split("\n")[0].strip()
                board = chess.Board(fen)
                result = self.chess_engine.get_best_move(board, depth=12, time_limit_ms=5000)
                if result:
                    return {
                        "response": result.uci(),
                        "expert": expert_mode,
                        "mode": mode,
                        "engine_used": True,
                        "confidence": 1.0
                    }
            except Exception as e:
                # Log error but continue to LLM fallback
                import logging
                logging.warning(f"Hybrid engine failed in expert manager: {e}")
                pass

        # Set active adapter for this expert
        self.set_active_adapter(expert_mode)

        # Get expert-specific parameters
        expert_config = self._expert_configs.get(expert_mode, self._expert_configs["tutor"])

        # Use provided values or expert defaults
        final_max_tokens = max_new_tokens if max_new_tokens is not None else expert_config["max_new_tokens"]
        final_temperature = temperature if temperature is not None else expert_config["temperature"]
        final_top_p = top_p if top_p is not None else expert_config["top_p"]
        final_do_sample = do_sample if do_sample is not None else expert_config["do_sample"]

        # Build prompt
        messages = self._build_messages(question, context, mode)
        prompt_text = messages[0]['content']

        # Generate response using core engine
        response_text = self.core_engine.generate_text(
            prompt=prompt_text,
            max_new_tokens=final_max_tokens,
            do_sample=final_do_sample,
            temperature=final_temperature,
            top_p=final_top_p,
        )

        # Calculate confidence based on mode and response quality
        confidence = self._calculate_confidence(response_text, mode)

        return {
            "response": response_text,
            "confidence": confidence,
            "model_loaded": self.core_engine.is_loaded,
            "mode": mode,
            "expert": expert_mode,
            "active_adapter": self._active_adapter,
        }

    def _calculate_confidence(self, response: str, mode: str) -> float:
        """Calculate response confidence based on mode and content quality."""
        if not response:
            return 0.0

        if mode == "engine" and len(response.strip()) in (4, 5):
            # High confidence for valid UCI tokens
            return 0.9
        else:
            # For other modes, use word count as a proxy for quality
            word_count = len(response.split())
            return max(0.1, min(0.95, word_count / 60.0))

    def get_available_experts(self) -> List[str]:
        """Get list of available expert adapters."""
        return list(self._adapter_paths.keys())

    def get_expert_info(self) -> Dict[str, Any]:
        """Get information about all experts and their status."""
        return {
            "available_experts": self.get_available_experts(),
            "active_adapter": self._active_adapter,
            "loaded_adapters": list(self._loaded_adapters.keys()),
            "adapter_paths": {k: str(v) for k, v in self._adapter_paths.items()},
        }

    def get_expert_config(self, expert_name: str) -> Dict[str, Any]:
        """Get configuration for a specific expert."""
        return self._expert_configs.get(expert_name, self._expert_configs["tutor"])
