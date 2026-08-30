from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import pytest

from gemmafischer.engine import ANALYSIS_SKILL_LEVEL, StockfishProvider


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
    provider._lock = threading.RLock()
    provider._started_at = None
    provider._applied_options = {"Skill Level": 4}

    with pytest.raises(RuntimeError, match="stop after configuration"):
        provider.analyze("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")

    assert engine.configurations[0]["Skill Level"] == ANALYSIS_SKILL_LEVEL
    assert provider._applied_options["Skill Level"] == ANALYSIS_SKILL_LEVEL
