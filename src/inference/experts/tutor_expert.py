from __future__ import annotations

from typing import Optional, Tuple

import chess

from src.inference.inference import ChessGemmaInference
from src.inference.chess_engine import ChessEngineManager
from src.inference.uci_utils import extract_first_uci, to_move_if_legal


class TutorExpert:
    """Expert specialized in explanatory analysis ending with a legal UCI move."""

    def __init__(self, inference: ChessGemmaInference, engine: Optional[ChessEngineManager] = None):
        self.inference = inference
        self.engine = engine

    def analyze_and_suggest(
        self,
        board: chess.Board,
        style: str = "balanced",
        depth: int = 12,
        time_limit_ms: int = 5000,
    ) -> Tuple[str, Optional[chess.Move]]:
        """Return explanatory text and a legal move; fallback to engine if needed."""
        if not self.inference.load_model():
            return ("Unable to load model; using engine fallback.", self._fallback_engine(board, depth, time_limit_ms))

        prompt = (
            f"FEN: {board.fen()}\n"
            "Question: Analyze this position step by step.\n"
            f"Style: {style}\n"
            "Mode: Tutor\n\n"
            "1. Evaluate the current position\n"
            "2. Identify key threats and opportunities\n"
            "3. Consider candidate moves\n"
            "4. Choose the best move with reasoning\n\n"
            "Respond with the best move in UCI format at the end."
        )

        try:
            analysis_text = self.inference.generate_text(
                prompt,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
            )
        except Exception:
            return ("Generation error; using engine fallback.", self._fallback_engine(board, depth, time_limit_ms))

        # Prefer the last UCI token in an explanation as the final choice
        move_obj = None
        last_match = None
        import re
        for match in re.finditer(r"\b([a-h][1-8][a-h][1-8][qrbn]?)\b", analysis_text.lower()):
            last_match = match.group(1)
        if last_match:
            move_obj = to_move_if_legal(board, last_match)

        if move_obj is None:
            engine_move = self._fallback_engine(board, depth, time_limit_ms)
            return (analysis_text + "\n\n(Note: Model move was invalid or missing; using engine best.)", engine_move)

        return (analysis_text, move_obj)

    def _fallback_engine(self, board: chess.Board, depth: int, time_limit_ms: int) -> Optional[chess.Move]:
        if not self.engine:
            return None
        try:
            return self.engine.get_best_move(board, depth=depth, time_limit_ms=time_limit_ms)
        except Exception:
            return None


