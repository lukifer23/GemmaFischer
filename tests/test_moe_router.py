#!/usr/bin/env python3
"""Tests for ChessMoERouter deterministic routing."""

import sys
from pathlib import Path

import pytest

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.inference.moe_router import ChessMoERouter


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
