from __future__ import annotations

import sqlite3
from pathlib import Path

from .domain import AnalysisSnapshot, AnalysisState, ErrorDetail, Session, now_utc

ACTIVE_STATES = (
    AnalysisState.QUEUED,
    AnalysisState.VALIDATING,
    AnalysisState.ENGINE_RUNNING,
    AnalysisState.COMPARISON_RUNNING,
    AnalysisState.MODEL_RUNNING,
)


def default_history_path() -> Path:
    return Path.home() / "Library" / "Application Support" / "GemmaFischer" / "history.sqlite3"


class AnalysisStore:
    """Small local SQLite ledger for analysis snapshots."""

    def __init__(self, path: Path, retention: int = 250) -> None:
        self.path = path
        self.retention = max(1, retention)
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS analyses (
                    analysis_id TEXT PRIMARY KEY,
                    generation INTEGER NOT NULL,
                    state TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    snapshot_json TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    updated_at TEXT NOT NULL,
                    session_json TEXT NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS sessions_updated_idx ON sessions(updated_at DESC)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS analyses_updated_idx ON analyses(updated_at DESC)"
            )
        self._recover_interrupted()

    def save(self, snapshot: AnalysisSnapshot) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO analyses (
                    analysis_id, generation, state, created_at, updated_at, snapshot_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(analysis_id) DO UPDATE SET
                    generation=excluded.generation,
                    state=excluded.state,
                    updated_at=excluded.updated_at,
                    snapshot_json=excluded.snapshot_json
                """,
                (
                    snapshot.analysis_id,
                    snapshot.generation,
                    snapshot.state.value,
                    snapshot.created_at.isoformat(),
                    snapshot.updated_at.isoformat(),
                    snapshot.model_dump_json(),
                ),
            )
            connection.execute(
                """
                DELETE FROM analyses WHERE analysis_id IN (
                    SELECT analysis_id FROM analyses
                    ORDER BY updated_at DESC LIMIT -1 OFFSET ?
                )
                """,
                (self.retention,),
            )

    def get(self, analysis_id: str) -> AnalysisSnapshot | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT snapshot_json FROM analyses WHERE analysis_id = ?", (analysis_id,)
            ).fetchone()
        return AnalysisSnapshot.model_validate_json(row[0]) if row else None

    def recent(self, limit: int = 20) -> tuple[AnalysisSnapshot, ...]:
        bounded_limit = max(1, min(limit, 100))
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT snapshot_json FROM analyses ORDER BY updated_at DESC LIMIT ?",
                (bounded_limit,),
            ).fetchall()
        return tuple(AnalysisSnapshot.model_validate_json(row[0]) for row in rows)

    def save_session(self, session: Session) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO sessions (session_id, updated_at, session_json) VALUES (?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    updated_at=excluded.updated_at,
                    session_json=excluded.session_json
                """,
                (session.session_id, session.updated_at.isoformat(), session.model_dump_json()),
            )
            connection.execute(
                """
                DELETE FROM sessions WHERE session_id IN (
                    SELECT session_id FROM sessions
                    ORDER BY updated_at DESC LIMIT -1 OFFSET ?
                )
                """,
                (self.retention,),
            )

    def get_session(self, session_id: str) -> Session | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT session_json FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
        return Session.model_validate_json(row[0]) if row else None

    def recent_sessions(self, limit: int = 20) -> tuple[Session, ...]:
        bounded_limit = max(1, min(limit, 100))
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT session_json FROM sessions ORDER BY updated_at DESC LIMIT ?",
                (bounded_limit,),
            ).fetchall()
        return tuple(Session.model_validate_json(row[0]) for row in rows)

    def delete_session(self, session_id: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM sessions WHERE session_id = ?", (session_id,)
            )
        return cursor.rowcount > 0

    def _recover_interrupted(self) -> None:
        placeholders = ",".join("?" for _ in ACTIVE_STATES)
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT snapshot_json FROM analyses WHERE state IN ({placeholders})",  # noqa: S608
                tuple(state.value for state in ACTIVE_STATES),
            ).fetchall()
        for row in rows:
            snapshot = AnalysisSnapshot.model_validate_json(row[0])
            error = ErrorDetail(
                code="ANALYSIS_INTERRUPTED",
                message="The local process stopped before this analysis completed.",
                stage="lifecycle",
                retryable=True,
                remediation=("Run the analysis again.",),
                request_id=snapshot.analysis_id,
            )
            self.save(
                snapshot.model_copy(
                    update={"state": AnalysisState.FAILED, "updated_at": now_utc(), "error": error}
                )
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=5)
