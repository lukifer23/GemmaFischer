from __future__ import annotations

import time
from concurrent.futures import Future, ThreadPoolExecutor

import pytest

from gemmafischer.domain import GameDifficulty
from gemmafischer.engine import (
    EngineOperationCancelled,
    EngineOperationPreempted,
    StockfishProvider,
)

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def _wait_for_operation(provider: StockfishProvider, token: str, timeout: float = 3) -> None:
    deadline = time.monotonic() + timeout
    while True:
        active = provider.operation_status()
        if active is not None and active.token == token:
            return
        if time.monotonic() >= deadline:
            raise AssertionError(f"Operation {token} did not become active")
        time.sleep(0.005)


@pytest.mark.hardware
def test_real_gameplay_preempts_analysis_and_provider_recovers() -> None:
    provider = StockfishProvider(node_budget=5_000_000)
    token = "analysis-to-preempt"
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            analysis: Future[object] = executor.submit(
                provider.analyze, START_FEN, operation_id=token
            )
            _wait_for_operation(provider, token)
            gameplay = executor.submit(
                provider.play_engine_turn, START_FEN, difficulty=GameDifficulty.CLUB
            )

            with pytest.raises(EngineOperationPreempted):
                analysis.result(timeout=5)
            result = gameplay.result(timeout=5)

        assert result.move_uci
        assert provider.operation_status() is None
        follow_up = provider.play_engine_turn(START_FEN, difficulty=GameDifficulty.CASUAL)
        assert follow_up.move_uci
    finally:
        provider.close()


@pytest.mark.hardware
def test_exact_analysis_cancellation_is_bounded_and_cannot_target_gameplay() -> None:
    provider = StockfishProvider(node_budget=5_000_000)
    token = "analysis-to-cancel"
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            analysis: Future[object] = executor.submit(
                provider.analyze, START_FEN, operation_id=token
            )
            _wait_for_operation(provider, token)
            started = time.monotonic()
            assert provider.interrupt_analysis(token)
            with pytest.raises(EngineOperationCancelled):
                analysis.result(timeout=3)
            assert time.monotonic() - started < 2

        result = provider.play_engine_turn(START_FEN, difficulty=GameDifficulty.CASUAL)
        assert result.move_uci
        assert not provider.interrupt_analysis(token)
    finally:
        provider.close()


@pytest.mark.hardware
def test_waiting_analysis_can_be_cancelled_without_interrupting_gameplay() -> None:
    provider = StockfishProvider(node_budget=500_000)
    analysis_token = "waiting-analysis"
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            gameplay = executor.submit(
                provider.play_engine_turn,
                START_FEN,
                difficulty=GameDifficulty.STRONG,
            )
            deadline = time.monotonic() + 3
            while (active := provider.operation_status()) is None or active.kind != "gameplay":
                if time.monotonic() >= deadline:
                    raise AssertionError("Gameplay did not become active")
                time.sleep(0.005)
            analysis = executor.submit(
                provider.analyze, START_FEN, operation_id=analysis_token
            )
            deadline = time.monotonic() + 2
            while not provider.interrupt_analysis(analysis_token):
                if time.monotonic() >= deadline:
                    raise AssertionError("Waiting analysis did not register for cancellation")
                time.sleep(0.005)

            with pytest.raises(EngineOperationCancelled):
                analysis.result(timeout=3)
            result = gameplay.result(timeout=5)

        assert result.move_uci
        assert provider.operation_status() is None
        assert provider.play_engine_turn(START_FEN, difficulty=GameDifficulty.CASUAL).move_uci
    finally:
        provider.close()
