#!/usr/bin/env python3
"""Unit tests for the HybridEngine LC0 orchestration."""

from __future__ import annotations

from unittest.mock import Mock, patch
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config.config_manager import ChessEngineConfig, LC0EngineSettings, FallbackEngineSettings
from src.inference.chess_engine import PositionAnalysis, MoveAnalysis
from src.inference.hybrid_engine import HybridEngine, HybridEngineResult
from src.inference.moe_router import RoutingDecision, MoEInferenceManager

FEN_START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def _make_config() -> ChessEngineConfig:
    """Helper to build a deterministic engine configuration for tests."""

    return ChessEngineConfig(
        primary="lc0",
        lc0=LC0EngineSettings(enabled=True, time_limit=0.05, depth=12, use_pool=False),
        fallback=FallbackEngineSettings(enabled=True, time_limit=0.05, depth=18),
    )


def _analysis(best_move: str | None, depth: int, nodes: int, pv: list[str] | None = None) -> PositionAnalysis:
    """Create a PositionAnalysis structure for mocks."""

    return PositionAnalysis(
        fen=FEN_START,
        best_move=best_move,
        best_score=34 if best_move else None,
        mate_in=None,
        evaluation={"depth": depth, "nodes": nodes},
        principal_variation=pv or [],
        top_moves=[
            MoveAnalysis(move=move, is_legal=True, is_best=index == 0)
            for index, move in enumerate(pv or [])
        ],
        threats=["fork"],
        opportunities=["center"],
        position_type="opening",
    )


@patch("src.inference.hybrid_engine.create_stockfish_manager")
@patch("src.inference.hybrid_engine.create_lc0_manager")
def test_hybrid_engine_primary_success(mock_lc0, mock_stockfish):
    """LC0 primary results should propagate without fallback."""

    lc0_manager = Mock()
    lc0_manager.analyze_position.side_effect = lambda fen, depth, time_limit: _analysis(
        best_move="e2e4", depth=20, nodes=2048, pv=["e2e4", "e7e5"]
    )
    mock_lc0.return_value = lc0_manager

    stockfish_manager = Mock()
    stockfish_manager.analyze_position.return_value = _analysis(
        best_move="d2d4", depth=18, nodes=1024, pv=["d2d4", "d7d5"]
    )
    mock_stockfish.return_value = stockfish_manager

    engine = HybridEngine(_make_config())
    result = engine.analyze(FEN_START)

    assert isinstance(result, HybridEngineResult)
    assert result.engine_name == "LC0"
    assert result.best_move == "e2e4"
    assert result.principal_variation == ["e2e4", "e7e5"]
    assert result.fallback_used is False
    assert result.raw_analysis.keys() == {"evaluation", "threats", "opportunities", "position_type", "error"}
    assert result.raw_analysis["evaluation"]["depth"] == 20
    lc0_manager.analyze_position.assert_called_once()
    # Fallback engine should not be invoked when LC0 succeeds.
    stockfish_manager.analyze_position.assert_not_called()


@patch("src.inference.hybrid_engine.create_stockfish_manager")
@patch("src.inference.hybrid_engine.create_lc0_manager")
def test_hybrid_engine_fallback_when_primary_missing_move(mock_lc0, mock_stockfish):
    """Fallback engine should activate when LC0 returns no best move."""

    lc0_manager = Mock()
    lc0_manager.analyze_position.side_effect = lambda fen, depth, time_limit: _analysis(
        best_move=None, depth=12, nodes=512
    )
    mock_lc0.return_value = lc0_manager

    stockfish_manager = Mock()
    stockfish_manager.analyze_position.side_effect = lambda fen, depth, time_limit: _analysis(
        best_move="d2d4", depth=22, nodes=4096, pv=["d2d4", "d7d5"]
    )
    mock_stockfish.return_value = stockfish_manager

    engine = HybridEngine(_make_config())
    result = engine.analyze(FEN_START)

    assert result.engine_name == "Stockfish"
    assert result.best_move == "d2d4"
    assert result.principal_variation == ["d2d4", "d7d5"]
    assert result.fallback_used is True
    assert result.raw_analysis["error"] is None
    stockfish_manager.analyze_position.assert_called_once()


def test_moe_manager_routes_with_fallback_flag():
    """Router metadata should include fallback usage for downstream consumers."""

    class _StubRouter:
        def route_query(self, fen, query_type="auto", complexity_score=None, question_text=""):
            return RoutingDecision(
                primary_expert="uci",
                expert_weights={"uci": 1.0},
                confidence_score=0.85,
                reasoning="engine fallback",
                ensemble_mode=False,
                fallback_used=True,
            )

    inference_stub = Mock()
    inference_stub.generate_response.return_value = {"response": "e2e4", "confidence": 0.9}
    inference_stub.set_active_adapter = Mock()
    inference_stub.refresh_adapters = Mock()

    manager = MoEInferenceManager(_StubRouter(), {"uci": str(project_root)}, inference_stub)
    result = manager.analyze_position(FEN_START, query_type="engine")

    assert result["routing_info"]["fallback_used"] is True
    assert result["routing_info"]["primary_expert"] == "uci"
    inference_stub.set_active_adapter.assert_called()
