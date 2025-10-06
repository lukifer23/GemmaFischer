"""Tests for the chess engine manager concurrency protections."""

import sys
import threading
import time
from pathlib import Path

import pytest

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.chess_engine import ChessEngineManager  # noqa: E402


class _FakeEngine:
    """Fake engine that raises when used concurrently."""

    def __init__(self):
        self._lock = threading.Lock()

    def use(self):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Engine is already in use")
        try:
            time.sleep(0.01)
        finally:
            self._lock.release()

    def quit(self):  # pragma: no cover - simple cleanup hook
        pass


@pytest.mark.parametrize("batch_size", [4, 8])
def test_batch_validate_moves_avoids_concurrent_engine_access(monkeypatch, batch_size):
    """batch_validate_moves should not trigger concurrent engine usage errors."""

    fake_engine = _FakeEngine()

    def fake_initialize(self):
        self.engine = fake_engine

    monkeypatch.setattr(ChessEngineManager, "_initialize_engine", fake_initialize, raising=False)

    manager = ChessEngineManager()

    def fake_validate_dataset_entry(self, question, answer):
        fake_engine.use()
        return {"question": question, "answer": answer}

    monkeypatch.setattr(ChessEngineManager, "validate_dataset_entry", fake_validate_dataset_entry, raising=False)

    moves_data = [
        {"question": f"fen {idx}", "answer": f"move {idx}"}
        for idx in range(batch_size)
    ]

    results = manager.batch_validate_moves(moves_data)

    assert len(results) == batch_size
    assert [result["question"] for result in results] == [item["question"] for item in moves_data]

