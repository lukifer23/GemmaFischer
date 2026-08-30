from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest

import gemmafischer.service as service_module
from gemmafischer.domain import (
    AnalysisRequest,
    AnalysisState,
    CreateSessionRequest,
    Ply,
    SessionCommandRequest,
    Workflow,
)
from gemmafischer.service import AnalysisService, SessionConflict

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def test_same_revision_commands_are_atomic_per_session() -> None:
    service = AnalysisService()
    try:
        session = service.create_session(CreateSessionRequest(fen=START_FEN))
        command = SessionCommandRequest(expected_revision=0, action="pause")
        barrier = threading.Barrier(2)

        def pause() -> object:
            barrier.wait()
            return service.command_session(session.session_id, command)

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(pause) for _ in range(2)]
        results, conflicts = [], []
        for future in futures:
            try:
                results.append(future.result())
            except SessionConflict as exc:
                conflicts.append(exc)

        assert len(results) == 1
        assert len(conflicts) == 1
        assert service.get_session(session.session_id).revision == 1  # type: ignore[union-attr]
    finally:
        service.close()


class _BlockingProvider:
    instances = 0

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        type(self).instances += 1
        self.started = threading.Event()
        self.interrupted = threading.Event()
        self.closed = False

    def analyze(self, *_args: Any, **_kwargs: Any) -> Any:
        self.started.set()
        self.interrupted.wait(5)
        raise RuntimeError("interrupted")

    def interrupt(self) -> None:
        self.interrupted.set()

    def close(self) -> None:
        self.closed = True
        self.interrupted.set()


def test_close_interrupts_active_work_and_rejects_new_work(monkeypatch: pytest.MonkeyPatch) -> None:
    _BlockingProvider.instances = 0
    monkeypatch.setattr(service_module, "StockfishProvider", _BlockingProvider)
    service = AnalysisService()
    snapshot = service.submit(AnalysisRequest(mode=Workflow.POSITION, fen=START_FEN))
    deadline = time.monotonic() + 1
    while service._provider is None or not service._provider.started.is_set():
        assert time.monotonic() < deadline
        time.sleep(0.005)

    started = time.monotonic()
    service.close()

    assert time.monotonic() - started < 1
    assert service.get(snapshot.analysis_id).state is AnalysisState.CANCELLED  # type: ignore[union-attr]
    assert service._provider.closed
    with pytest.raises(RuntimeError, match="closed"):
        service.submit(AnalysisRequest(mode=Workflow.POSITION, fen=START_FEN))


def test_engine_provider_is_created_once_per_service(monkeypatch: pytest.MonkeyPatch) -> None:
    class CountingProvider:
        instances = 0

        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            type(self).instances += 1

        def interrupt(self) -> None:
            return None

        def close(self) -> None:
            return None

    monkeypatch.setattr(service_module, "StockfishProvider", CountingProvider)
    service = AnalysisService()
    try:
        assert service._engine() is service._engine()
        assert CountingProvider.instances == 1
    finally:
        service.close()


def test_repeated_cancel_of_terminal_job_does_not_interrupt_engine() -> None:
    class InterruptCounter:
        def __init__(self) -> None:
            self.interrupts = 0

        def interrupt(self) -> None:
            self.interrupts += 1

        def close(self) -> None:
            return None

    service = AnalysisService()
    try:
        with service._condition:
            snapshot = service.submit(AnalysisRequest(mode=Workflow.POSITION, fen=START_FEN))
            first = service.cancel(snapshot.analysis_id)
        assert first is not None and first.state is AnalysisState.CANCELLED
        counter = InterruptCounter()
        service._provider = counter  # type: ignore[assignment]

        second = service.cancel(snapshot.analysis_id)

        assert second is not None and second.state is AnalysisState.CANCELLED
        assert counter.interrupts == 0
    finally:
        service.close()


def test_undo_returns_player_to_previous_decision_point() -> None:
    service = AnalysisService()
    try:
        session = service.create_session(CreateSessionRequest(fen=START_FEN))
        after_e4 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
        after_e5 = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"
        session = service._replace_session(
            session,
            fen=after_e5,
            turn="white",
            plies=(
                Ply(
                    ply=1,
                    move_uci="e2e4",
                    move_san="e4",
                    fen_before=START_FEN,
                    fen_after=after_e4,
                    actor="player",
                ),
                Ply(
                    ply=2,
                    move_uci="e7e5",
                    move_san="e5",
                    fen_before=after_e4,
                    fen_after=after_e5,
                    actor="engine_black",
                ),
            ),
        )

        undone = service.command_session(
            session.session_id,
            SessionCommandRequest(expected_revision=1, action="undo"),
        )

        assert undone.plies == ()
        assert undone.fen == START_FEN
        assert undone.turn == "white"
    finally:
        service.close()
