from pathlib import Path

from gemmafischer.domain import (
    AnalysisRequest,
    AnalysisSnapshot,
    AnalysisState,
    Session,
    SessionMode,
    SessionStatus,
    Workflow,
    now_utc,
)
from gemmafischer.storage import AnalysisStore


def snapshot(analysis_id: str, generation: int, state: AnalysisState) -> AnalysisSnapshot:
    timestamp = now_utc()
    return AnalysisSnapshot(
        analysis_id=analysis_id,
        generation=generation,
        state=state,
        created_at=timestamp,
        updated_at=timestamp,
        request=AnalysisRequest(
            mode=Workflow.POSITION,
            fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        ),
    )


def test_store_round_trips_and_prunes_snapshots(tmp_path: Path) -> None:
    store = AnalysisStore(tmp_path / "history.sqlite3", retention=2)
    for generation in range(1, 4):
        store.save(snapshot(f"analysis-{generation}", generation, AnalysisState.COMPLETE))

    assert store.get("analysis-1") is None
    assert [item.analysis_id for item in store.recent()] == ["analysis-3", "analysis-2"]


def test_store_marks_interrupted_work_failed_on_restart(tmp_path: Path) -> None:
    path = tmp_path / "history.sqlite3"
    AnalysisStore(path).save(snapshot("interrupted", 1, AnalysisState.MODEL_RUNNING))

    recovered = AnalysisStore(path).get("interrupted")

    assert recovered is not None
    assert recovered.state is AnalysisState.FAILED
    assert recovered.error is not None
    assert recovered.error.code == "ANALYSIS_INTERRUPTED"


def test_store_prunes_session_history_to_configured_retention(tmp_path: Path) -> None:
    store = AnalysisStore(tmp_path / "history.sqlite3", retention=2)
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    for generation in range(3):
        timestamp = now_utc()
        store.save_session(
            Session(
                session_id=f"session-{generation}",
                revision=0,
                mode=SessionMode.PLAYER,
                status=SessionStatus.ACTIVE,
                initial_fen=fen,
                fen=fen,
                turn="white",
                player_color="white",
                created_at=timestamp,
                updated_at=timestamp,
            )
        )

    assert [item.session_id for item in store.recent_sessions()] == ["session-2", "session-1"]
    assert store.get_session("session-0") is None
