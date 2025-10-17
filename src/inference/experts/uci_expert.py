from __future__ import annotations

from typing import Optional

import chess

from src.inference.inference import ChessGemmaInference
from src.inference.hybrid_engine import HybridEngineResult
from src.inference.uci_utils import extract_first_uci, to_move_if_legal


class UCIExpert:
    """Expert specialized in producing a single legal UCI move quickly."""

    def __init__(self, inference: ChessGemmaInference, engine: Optional[object] = None):
        self.inference = inference
        # Engine is no longer used directly; HybridEngine is called via inference

    def suggest_move(
        self,
        board: chess.Board,
        style: str = "balanced",
        depth: int = 12,
        time_limit_ms: int = 5000,
    ) -> Optional[chess.Move]:
        """Return a legal move using HybridEngine (LC0 primary) or LLM fallback."""
        try:
            result = self.inference.analyze_with_engine(board.fen(), explanation_mode="tutor")
            move_uci = result.get("best_move") if isinstance(result, dict) else None
            if move_uci:
                mv = to_move_if_legal(board, move_uci)
                if mv:
                    return mv
        except Exception:
            pass

    def _fallback_engine(self, board: chess.Board, depth: int, time_limit_ms: int) -> Optional[chess.Move]:
        return None


