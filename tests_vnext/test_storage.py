from pathlib import Path

from gemmafischer.domain import AnalysisRequest, AnalysisSnapshot, AnalysisState, Workflow, now_utc
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
