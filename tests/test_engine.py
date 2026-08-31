from __future__ import annotations

import threading
from collections import deque
from pathlib import Path
from typing import Any

import chess
import pytest

from gemmafischer.domain import GameDifficulty
from gemmafischer.engine import ANALYSIS_SKILL_LEVEL, StockfishProvider, legal_moves_for_square


class _FailAfterConfigureEngine:
    options = {"Skill Level": object()}
    id = {"name": "Fakefish", "author": "tests"}

    def __init__(self) -> None:
        self.configurations: list[dict[str, Any]] = []

    def configure(self, options: dict[str, Any]) -> None:
        self.configurations.append(options)

    def analyse(self, *_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("stop after configuration")

    def quit(self) -> None:
        return None

    def close(self) -> None:
        return None


def test_analysis_restores_full_skill_after_gameplay_configuration() -> None:
    engine = _FailAfterConfigureEngine()
    provider = StockfishProvider.__new__(StockfishProvider)
    provider.path = Path("/fake/stockfish")
    provider.node_budget = 1
    provider.binary_sha256 = "fake"
    provider._engine = engine  # type: ignore[assignment]
    provider._condition = threading.Condition()
    provider._active_operation = None
    provider._analysis_waiters = set()
    provider._gameplay_waiters = deque()
    provider._interrupt_reasons = {}
    provider._closed = False
    provider._started_at = None
    provider._applied_options = {"Skill Level": 4}

    with pytest.raises(RuntimeError, match="stop after configuration"):
        provider.analyze("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")

    assert engine.configurations[0]["Skill Level"] == ANALYSIS_SKILL_LEVEL
    assert provider._applied_options["Skill Level"] == ANALYSIS_SKILL_LEVEL


@pytest.mark.hardware
def test_exact_underpromotion_moves_are_preserved_and_missing_suffix_is_rejected() -> None:
    fen = "7k/P7/8/8/8/8/8/7K w - - 0 1"
    legal = legal_moves_for_square(fen, "a7")
    assert set(legal.moves_uci) == {"a7a8q", "a7a8r", "a7a8b", "a7a8n"}
    assert legal.destinations == ("a8",)

    with StockfishProvider(node_budget=1) as provider:
        with pytest.raises(ValueError, match="requires q, r, b, or n suffix"):
            provider.play_move(
                fen, "a7a8", engine_reply=False, difficulty=GameDifficulty.CLUB
            )
        for suffix, piece_type in {
            "q": chess.QUEEN,
            "r": chess.ROOK,
            "b": chess.BISHOP,
            "n": chess.KNIGHT,
        }.items():
            result = provider.play_move(
                fen, f"a7a8{suffix}", engine_reply=False, difficulty=GameDifficulty.CLUB
            )
            board = chess.Board(result.fen)
            assert board.piece_at(chess.A8) == chess.Piece(piece_type, chess.WHITE)


@pytest.mark.hardware
def test_black_and_capture_underpromotions_are_preserved() -> None:
    with StockfishProvider(node_budget=1) as provider:
        black = provider.play_move(
            "7k/8/8/8/8/8/p7/7K b - - 0 1",
            "a2a1n",
            engine_reply=False,
            difficulty=GameDifficulty.CLUB,
        )
        capture = provider.play_move(
            "r6k/1P6/8/8/8/8/8/7K w - - 0 1",
            "b7a8r",
            engine_reply=False,
            difficulty=GameDifficulty.CLUB,
        )

    assert chess.Board(black.fen).piece_at(chess.A1) == chess.Piece(
        chess.KNIGHT, chess.BLACK
    )
    assert chess.Board(capture.fen).piece_at(chess.A8) == chess.Piece(
        chess.ROOK, chess.WHITE
    )
