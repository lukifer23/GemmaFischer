from __future__ import annotations

from typing import Optional, Tuple

import chess

from src.inference.inference import ChessGemmaInference
from src.inference.chess_engine import ChessEngineManager
from src.inference.experts.uci_expert import UCIExpert
from src.inference.experts.tutor_expert import TutorExpert
from src.inference.experts.director_expert import DirectorExpert


class Router:
    """Deterministic router for expert selection (v1)."""

    def __init__(self, inference: ChessGemmaInference, engine: Optional[ChessEngineManager] = None):
        self.inference = inference
        self.engine = engine
        self._uci = UCIExpert(inference, engine)
        self._tutor = TutorExpert(inference, engine)
        self._director = DirectorExpert(inference)

    def suggest_move(
        self,
        board: chess.Board,
        mode: str = "engine",
        style: str = "balanced",
        depth: int = 12,
        time_limit_ms: int = 5000,
    ) -> Optional[chess.Move]:
        mode_lc = (mode or "").strip().lower()
        if mode_lc == "tutor":
            _, move = self._tutor.analyze_and_suggest(board, style=style, depth=depth, time_limit_ms=time_limit_ms)
            return move
        # default to engine for UCI use cases
        return self._uci.suggest_move(board, style=style, depth=depth, time_limit_ms=time_limit_ms)

    def analyze_with_move(
        self,
        board: chess.Board,
        style: str = "balanced",
        depth: int = 12,
        time_limit_ms: int = 5000,
    ) -> Tuple[str, Optional[chess.Move]]:
        return self._tutor.analyze_and_suggest(board, style=style, depth=depth, time_limit_ms=time_limit_ms)

    def answer(self, question: str) -> str:
        return self._director.answer(question)


