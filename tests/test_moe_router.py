#!/usr/bin/env python3
"""Tests for ChessMoERouter deterministic routing."""

import sys
from pathlib import Path

import logging

import pytest

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.inference.moe_router import ChessMoERouter
from src.inference.inference import ChessGemmaInference
from src.inference import inference as inference_module


def test_route_query_deterministic_expert_selection():
    """Repeated routing with identical input should select the same expert."""
    router = ChessMoERouter()

    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

    # Even if train(True) is called, router should stay in eval mode
    router.train(True)

    experts = [router.route_query(fen).primary_expert for _ in range(5)]

    assert len(set(experts)) == 1, "Expert selection should be deterministic"


def _king_safety_score(router: ChessMoERouter, fen: str, color: str) -> float:
    board = router._fen_to_board(fen)
    kings = router._find_kings(board)
    return router._calculate_king_safety_score(board, kings[color], color)


def test_king_safety_improves_with_white_defenders():
    router = ChessMoERouter()

    exposed = "4k3/8/8/8/8/8/8/4K3 w - - 0 1"
    defended = "4k3/8/8/8/8/8/3PPP2/4K3 w - - 0 1"
    under_attack = "4k3/8/8/8/8/8/3PPP2/3rKr2 w - - 0 1"

    exposed_score = _king_safety_score(router, exposed, 'white')
    defended_score = _king_safety_score(router, defended, 'white')
    attacked_score = _king_safety_score(router, under_attack, 'white')

    assert defended_score > exposed_score
    assert attacked_score < defended_score


def test_king_safety_improves_with_black_defenders():
    router = ChessMoERouter()

    exposed = "4k3/8/8/8/8/8/8/4K3 w - - 0 1"
    defended = "4k3/3ppp2/8/8/8/8/8/4K3 w - - 0 1"
    under_attack = "3RkR2/3ppp2/8/8/8/8/8/4K3 w - - 0 1"

    exposed_score = _king_safety_score(router, exposed, 'black')
    defended_score = _king_safety_score(router, defended, 'black')
    attacked_score = _king_safety_score(router, under_attack, 'black')

    assert defended_score > exposed_score
    assert attacked_score < defended_score


def test_moe_router_env_override(monkeypatch, tmp_path):
    """MoE initialization should honor the router checkpoint override."""

    checkpoints_root = tmp_path / "checkpoints"
    (checkpoints_root / "lora_uci" / "checkpoint-1").mkdir(parents=True)
    (checkpoints_root / "lora_tutor" / "checkpoint-1").mkdir(parents=True)

    override_ckpt = tmp_path / "custom_router.pt"
    override_ckpt.write_text("stub")

    loaded_path = {}

    def fake_load(self, path):
        loaded_path["value"] = path

    test_logger = logging.getLogger("test-moe-router")
    monkeypatch.setattr(inference_module, "logger", test_logger, raising=False)
    monkeypatch.setattr(ChessMoERouter, "load_router", fake_load, raising=True)
    monkeypatch.setenv("CHESSGEMMA_MOE_ROUTER_CKPT", str(override_ckpt))

    inference = ChessGemmaInference.__new__(ChessGemmaInference)
    inference.project_root = tmp_path
    inference.moe_enabled = True
    inference.moe_router = None
    inference.moe_manager = None
    inference._expert_paths = {}
    inference._moe_dispatch_depth = 0
    inference.debug = False
    inference.is_loaded = False
    inference._prewarm_enabled = False
    inference._prewarm_thread = None
    inference.refresh_adapters = lambda: None
    inference.set_active_adapter = lambda name: None

    inference._initialize_moe_system()

    assert loaded_path.get("value") == str(override_ckpt)
