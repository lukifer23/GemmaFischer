"""Tests for advanced benchmark move quality evaluation."""

from typing import List
from unittest.mock import patch
from pathlib import Path
import sys

import chess
import chess.engine

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.advanced_benchmark import ChessGemmaBenchmarker


class FakeEngine:
    """Minimal engine stub returning predetermined scores."""

    def __init__(self, scores: List[chess.engine.PovScore]):
        self._scores = scores
        self._call_count = 0

    def analyse(self, board: chess.Board, limit: chess.engine.Limit):  # type: ignore[override]
        index = min(self._call_count, len(self._scores) - 1)
        self._call_count += 1
        return {'score': self._scores[index]}


class FakeManager:
    """Context manager stub that records cleanup usage."""

    instances: List["FakeManager"] = []

    def __init__(self, scores: List[chess.engine.PovScore]):
        self.engine = FakeEngine(scores)
        self.cleaned = False
        FakeManager.instances.append(self)

    def __enter__(self) -> "FakeManager":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.cleanup()

    def cleanup(self) -> None:
        self.cleaned = True


def _make_score(cp: int, color: chess.Color = chess.WHITE) -> chess.engine.PovScore:
    return chess.engine.PovScore(chess.engine.Cp(cp), color)


def test_move_quality_prefers_best_move() -> None:
    benchmark = ChessGemmaBenchmarker()
    fen = chess.STARTING_BOARD_FEN

    best_score = _make_score(80)
    equal_candidate = _make_score(80)
    poor_candidate = _make_score(-400)

    FakeManager.instances = []
    with patch(
        "src.evaluation.advanced_benchmark.ChessEngineManager",
        side_effect=lambda: FakeManager([best_score, equal_candidate]),
    ):
        good_quality = benchmark._evaluate_move_quality({'fen': fen}, {'move': 'e2e4'})

    FakeManager.instances = []
    with patch(
        "src.evaluation.advanced_benchmark.ChessEngineManager",
        side_effect=lambda: FakeManager([best_score, poor_candidate]),
    ):
        bad_quality = benchmark._evaluate_move_quality({'fen': fen}, {'move': 'a2a3'})

    assert 0.0 <= good_quality <= 1.0
    assert 0.0 <= bad_quality <= 1.0
    assert good_quality > bad_quality


def test_move_quality_uses_response_text_and_cleans_up() -> None:
    benchmark = ChessGemmaBenchmarker()
    fen = chess.STARTING_BOARD_FEN

    best_score = _make_score(0)
    candidate_score = _make_score(0)

    FakeManager.instances = []
    with patch(
        "src.evaluation.advanced_benchmark.ChessEngineManager",
        side_effect=lambda: FakeManager([best_score, candidate_score]),
    ):
        quality = benchmark._evaluate_move_quality({'fen': fen}, {'response': 'I choose e2e4'})

    assert quality > 0.0
    assert FakeManager.instances, "Engine manager should have been instantiated"
    assert all(instance.cleaned for instance in FakeManager.instances)


def test_move_quality_handles_missing_data() -> None:
    benchmark = ChessGemmaBenchmarker()

    with patch("src.evaluation.advanced_benchmark.ChessEngineManager") as manager_cls:
        score = benchmark._evaluate_move_quality({}, {'response': 'e2e4'})

    assert score == 0.0
    manager_cls.assert_not_called()
