import sqlite3
from pathlib import Path

from gemmafischer.domain import (
    AnalysisRequest,
    AnalysisSnapshot,
    AnalysisState,
    Ply,
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


def test_session_referenced_analyses_survive_retention_and_restart(tmp_path: Path) -> None:
    path = tmp_path / "history.sqlite3"
    store = AnalysisStore(path, retention=1)
    first = snapshot("review-1", 1, AnalysisState.COMPLETE)
    second = snapshot("review-2", 2, AnalysisState.COMPLETE)
    store.save(first)
    store.save(second)
    timestamp = now_utc()
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    after_e4 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
    store.save_session(
        Session(
            session_id="session-with-review",
            revision=1,
            mode=SessionMode.PLAYER,
            status=SessionStatus.ACTIVE,
            initial_fen=fen,
            fen=after_e4,
            turn="black",
            player_color="white",
            plies=(
                Ply(
                    ply=1,
                    move_uci="e2e4",
                    move_san="e4",
                    fen_before=fen,
                    fen_after=after_e4,
                    actor="player",
                    analysis_id=second.analysis_id,
                ),
            ),
            created_at=timestamp,
            updated_at=timestamp,
        )
    )
    for generation in range(3, 8):
        store.save(snapshot(f"new-{generation}", generation, AnalysisState.COMPLETE))

    reopened = AnalysisStore(path, retention=1)

    assert reopened.get(second.analysis_id) == second
    assert reopened.get("new-7") is not None
    assert reopened.get("new-6") is None


def test_deleting_session_releases_analysis_for_retention(tmp_path: Path) -> None:
    path = tmp_path / "history.sqlite3"
    store = AnalysisStore(path, retention=1)
    review = snapshot("review", 1, AnalysisState.COMPLETE)
    store.save(review)
    timestamp = now_utc()
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    after_e4 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
    store.save_session(
        Session(
            session_id="session",
            revision=1,
            mode=SessionMode.PLAYER,
            status=SessionStatus.ACTIVE,
            initial_fen=fen,
            fen=after_e4,
            turn="black",
            player_color="white",
            plies=(
                Ply(
                    ply=1,
                    move_uci="e2e4",
                    move_san="e4",
                    fen_before=fen,
                    fen_after=after_e4,
                    actor="player",
                    analysis_id=review.analysis_id,
                ),
            ),
            created_at=timestamp,
            updated_at=timestamp,
        )
    )
    store.save(snapshot("new", 2, AnalysisState.COMPLETE))
    assert store.get("review") is not None

    assert store.delete_session("session")

    assert store.get("review") is None
    assert store.get("new") is not None


def test_reserved_review_survives_until_session_link_under_retention(tmp_path: Path) -> None:
    store = AnalysisStore(tmp_path / "history.sqlite3", retention=1)
    review = snapshot("reserved-review", 1, AnalysisState.QUEUED)
    store.save(review, reserve=True)
    store.save(review.model_copy(update={"state": AnalysisState.COMPLETE}))
    for generation in range(2, 6):
        store.save(snapshot(f"unrelated-{generation}", generation, AnalysisState.COMPLETE))

    assert store.get(review.analysis_id) is not None

    timestamp = now_utc()
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    after_e4 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
    store.save_session(
        Session(
            session_id="reservation-owner",
            revision=1,
            mode=SessionMode.PLAYER,
            status=SessionStatus.ACTIVE,
            initial_fen=fen,
            fen=after_e4,
            turn="black",
            player_color="white",
            plies=(
                Ply(
                    ply=1,
                    move_uci="e2e4",
                    move_san="e4",
                    fen_before=fen,
                    fen_after=after_e4,
                    actor="player",
                    analysis_id=review.analysis_id,
                ),
            ),
            created_at=timestamp,
            updated_at=timestamp,
        )
    )
    store.save(snapshot("newest", 6, AnalysisState.COMPLETE))

    assert store.get(review.analysis_id) is not None


def test_v1_database_migrates_and_backfills_review_references(tmp_path: Path) -> None:
    path = tmp_path / "history.sqlite3"
    review = snapshot("legacy-review", 1, AnalysisState.COMPLETE)
    timestamp = now_utc()
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    after_e4 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
    session = Session(
        session_id="legacy-session",
        revision=1,
        mode=SessionMode.PLAYER,
        status=SessionStatus.ACTIVE,
        initial_fen=fen,
        fen=after_e4,
        turn="black",
        player_color="white",
        plies=(
            Ply(
                ply=1,
                move_uci="e2e4",
                move_san="e4",
                fen_before=fen,
                fen_after=after_e4,
                actor="player",
                analysis_id=review.analysis_id,
            ),
        ),
        created_at=timestamp,
        updated_at=timestamp,
    )
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE analyses (analysis_id TEXT PRIMARY KEY, generation INTEGER NOT NULL, "
            "state TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL, "
            "snapshot_json TEXT NOT NULL)"
        )
        connection.execute(
            "CREATE TABLE sessions (session_id TEXT PRIMARY KEY, updated_at TEXT NOT NULL, "
            "session_json TEXT NOT NULL)"
        )
        connection.execute(
            "INSERT INTO analyses VALUES (?, ?, ?, ?, ?, ?)",
            (
                review.analysis_id,
                review.generation,
                review.state.value,
                review.created_at.isoformat(),
                review.updated_at.isoformat(),
                review.model_dump_json(),
            ),
        )
        connection.execute(
            "INSERT INTO sessions VALUES (?, ?, ?)",
            (session.session_id, session.updated_at.isoformat(), session.model_dump_json()),
        )

    migrated = AnalysisStore(path, retention=1)
    migrated.save(snapshot("post-migration", 2, AnalysisState.COMPLETE))

    assert migrated.get(review.analysis_id) == review
    assert migrated.get_session(session.session_id) == session
