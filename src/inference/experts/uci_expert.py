from __future__ import annotations

from typing import Optional

import chess

from src.inference.inference import ChessGemmaInference
from src.inference.chess_engine import ChessEngineManager
from src.inference.uci_utils import extract_first_uci, to_move_if_legal


class UCIExpert:
    """Expert specialized in producing a single legal UCI move quickly."""

    def __init__(self, inference: ChessGemmaInference, engine: Optional[ChessEngineManager] = None):
        self.inference = inference
        self.engine = engine

    def suggest_move(
        self,
        board: chess.Board,
        style: str = "balanced",
        depth: int = 12,
        time_limit_ms: int = 5000,
    ) -> Optional[chess.Move]:
        """Return a legal move using LC0 engine as primary; LLM as strategic guidance only."""
        # Primary: Use LC0 engine for precise move calculation
        if self.engine:
            try:
                return self.engine.get_best_move(board, depth=depth, time_limit_ms=time_limit_ms)
            except Exception:
                pass

        # Fallback: Use LLM model if engine fails (should be rare)
        if not self.inference.load_model():
            return None

        prompt = (
            f"FEN: {board.fen()}\n"
            f"Move:\n"
            f"Style: {style}\n"
            f"Mode: Engine\n"
            "Generate the best move in UCI format (e.g., e2e4). Respond with only the move."
        )

        try:
            text = self.inference.generate_text(
                prompt,
                max_new_tokens=5,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                repetition_penalty=1.0,
            )
        except Exception:
            return None

        move_str = extract_first_uci(text)
        move_obj = to_move_if_legal(board, move_str) if move_str else None
        return move_obj

    def _fallback_engine(self, board: chess.Board, depth: int, time_limit_ms: int) -> Optional[chess.Move]:
        if not self.engine:
            return None
        try:
            return self.engine.get_best_move(board, depth=depth, time_limit_ms=time_limit_ms)
        except Exception:
            return None


