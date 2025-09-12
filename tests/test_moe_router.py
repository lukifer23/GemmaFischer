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
