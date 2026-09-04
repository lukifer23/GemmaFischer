from __future__ import annotations

import io
import time
from pathlib import Path

import chess.pgn
import pytest

import gemmafischer.study_qualification as qualification
from gemmafischer.service import AnalysisService
from gemmafischer.study import decision_positions, parse_import
from gemmafischer.study_domain import PGNImportRequest, StudyJobState
from gemmafischer.study_qualification import _long_game_pgn, run_study_recovery_qualification


def test_long_qualification_game_is_real_legal_and_exact() -> None:
    pgn = _long_game_pgn(200)
    game = parse_import(PGNImportRequest(pgn=pgn, perspective="white"))

    assert len(game.moves_uci) == 200
    assert len(decision_positions(game)) == 100
    assert chess.pgn.read_game(io.StringIO(pgn)) is not None


def test_study_qualification_rejects_invalid_budgets_before_engine_resolution(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="node_budget must be at least 1"):
        run_study_recovery_qualification(tmp_path / "nodes.json", node_budget=0)
    with pytest.raises(ValueError, match="timeout must be positive"):
        run_study_recovery_qualification(tmp_path / "timeout.json", timeout=0)
    with pytest.raises(ValueError, match="plies must be between 1 and 400"):
        _long_game_pgn(0)


def test_wait_for_study_returns_matching_live_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        qualification,
        "_request_json",
        lambda *_args, **_kwargs: ({"state": "screening", "progress": 4}, 200),
    )

    result = qualification._wait_for_study(
        "http://127.0.0.1:1", "token", "job", 1, lambda item: item["progress"] == 4
    )

    assert result["state"] == "screening"


def test_wait_for_study_reports_http_and_terminal_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        qualification, "_request_json", lambda *_args, **_kwargs: ({}, 503)
    )
    with pytest.raises(RuntimeError, match="Study poll returned HTTP 503"):
        qualification._wait_for_study(
            "http://127.0.0.1:1", "token", "job", 1, lambda _item: False
        )

    monkeypatch.setattr(
        qualification,
        "_request_json",
        lambda *_args, **_kwargs: ({"state": "failed", "error": "expected"}, 200),
    )
    with pytest.raises(RuntimeError, match="did not reach the expected state"):
        qualification._wait_for_study(
            "http://127.0.0.1:1", "token", "job", 1, lambda _item: False
        )


def test_timed_json_enforces_expected_status(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        qualification,
        "_request_json",
        lambda *_args, **_kwargs: ({"status": "ready"}, 200),
    )
    body, elapsed = qualification._timed_json(
        "GET", "http://127.0.0.1:1", "token", expected_status=200
    )
    assert body == {"status": "ready"}
    assert elapsed >= 0

    with pytest.raises(RuntimeError, match="returned HTTP 200, expected 202"):
        qualification._timed_json(
            "POST", "http://127.0.0.1:1", "token", expected_status=202
        )


@pytest.mark.hardware
def test_shutdown_cannot_overwrite_interrupted_study_with_failure(tmp_path: Path) -> None:
    database = tmp_path / "history.sqlite3"
    request = PGNImportRequest(pgn=_long_game_pgn(200), perspective="white")
    service = AnalysisService(node_budget=25_000, history_path=database)
    job = service.submit_study(request)
    deadline = time.monotonic() + 20
    while True:
        current = service.get_study(job.job_id)
        assert current is not None
        if (
            current.state in {StudyJobState.SCREENING, StudyJobState.DEEP_ANALYSIS}
            and current.progress.completed_units >= 4
        ):
            break
        assert time.monotonic() < deadline
        time.sleep(0.01)
    service.close()

    restored_service = AnalysisService(node_budget=1_000, history_path=database)
    try:
        restored = restored_service.get_study(job.job_id)
        assert restored is not None
        assert restored.state is StudyJobState.PAUSED_INTERRUPTED
        assert restored.game == current.game
    finally:
        restored_service.close()
