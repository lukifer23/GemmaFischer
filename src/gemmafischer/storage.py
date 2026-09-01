from __future__ import annotations

import hashlib
import json
import os
import platform
import sqlite3
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

from .domain import AnalysisSnapshot, AnalysisState, ErrorDetail, Session, now_utc
from .tutor import TutorInteractionRecord, deserialize_record, serialize_record

ACTIVE_STATES = (
    AnalysisState.QUEUED,
    AnalysisState.VALIDATING,
    AnalysisState.ENGINE_RUNNING,
    AnalysisState.COMPARISON_RUNNING,
    AnalysisState.MODEL_RUNNING,
)

SCHEMA_VERSION = 4


class StorageError(RuntimeError):
    """Base class for safe storage failures exposed by the service layer."""


class StorageUnavailable(StorageError):
    """A transient filesystem or SQLite failure that may be retried."""


class StorageCorrupt(StorageError):
    """A non-retryable database integrity or schema failure."""


class StorageConflict(StorageError):
    """The durable resource no longer matches the caller's expected state."""


class IdempotencyConflict(StorageConflict):
    """An idempotency key was reused with a different payload."""


def default_history_path() -> Path:
    override = os.environ.get("GEMMAFISCHER_DATA_DIR")
    if override:
        root = Path(override).expanduser()
    elif platform.system() == "Darwin":
        root = Path.home() / "Library" / "Application Support" / "GemmaFischer"
    else:
        root = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
        root /= "gemmafischer"
    return root / "history.sqlite3"


class AnalysisStore:
    """Small local SQLite ledger for analysis snapshots."""

    def __init__(self, path: Path, retention: int = 250) -> None:
        self.path = path
        self.retention = max(1, retention)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._migrate()
            self._recover_interrupted()
        except StorageError:
            raise
        except sqlite3.DatabaseError as exc:
            raise StorageCorrupt("The history database could not be opened safely.") from exc
        except OSError as exc:
            raise StorageUnavailable("The history directory is not writable.") from exc

    def _migrate(self) -> None:
        with self._connect_raw() as connection:
            integrity = connection.execute("PRAGMA quick_check").fetchone()
            if integrity is None or integrity[0] != "ok":
                raise StorageCorrupt("The history database failed its integrity check.")
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if version > SCHEMA_VERSION:
                raise StorageCorrupt(
                    f"History schema {version} is newer than supported schema {SCHEMA_VERSION}."
                )
            user_tables = connection.execute(
                "SELECT 1 FROM sqlite_master "
                "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' LIMIT 1"
            ).fetchone()
            # Older builds did not set user_version, so version zero can still
            # represent a real user database. Back up any populated schema
            # before the first migration statement touches it.
            if version < SCHEMA_VERSION and user_tables is not None:
                self._backup_before_migration(connection, version)
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("BEGIN IMMEDIATE")
            try:
                self._create_base_schema(connection)
                self._add_v4_columns(connection)
                self._backfill_session_analysis_refs(connection)
                connection.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
                connection.commit()
            except Exception:
                connection.rollback()
                raise

    def _create_base_schema(self, connection: sqlite3.Connection) -> None:
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
                    session_json TEXT NOT NULL,
                    revision INTEGER NOT NULL DEFAULT 0
                )
                """
            )
        connection.execute(
                "CREATE INDEX IF NOT EXISTS sessions_updated_idx ON sessions(updated_at DESC)"
            )
        connection.execute(
                """
                CREATE TABLE IF NOT EXISTS tutor_interactions (
                    interaction_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    record_json TEXT NOT NULL,
                    revision INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL DEFAULT 'awaiting_answer',
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
                )
                """
            )
        connection.execute(
                "CREATE INDEX IF NOT EXISTS tutor_session_updated_idx "
                "ON tutor_interactions(session_id, updated_at DESC)"
            )
        connection.execute(
                "CREATE INDEX IF NOT EXISTS analyses_updated_idx ON analyses(updated_at DESC)"
            )
        connection.execute(
                """
                CREATE TABLE IF NOT EXISTS session_analysis_refs (
                    session_id TEXT NOT NULL,
                    ply INTEGER NOT NULL,
                    analysis_id TEXT NOT NULL UNIQUE,
                    PRIMARY KEY (session_id, ply),
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE,
                    FOREIGN KEY (analysis_id) REFERENCES analyses(analysis_id) ON DELETE RESTRICT
                )
                """
            )
        connection.execute(
                "CREATE INDEX IF NOT EXISTS session_analysis_refs_analysis_idx "
                "ON session_analysis_refs(analysis_id)"
            )
        connection.execute(
                """
                CREATE TABLE IF NOT EXISTS analysis_reservations (
                    analysis_id TEXT PRIMARY KEY,
                    FOREIGN KEY (analysis_id) REFERENCES analyses(analysis_id) ON DELETE CASCADE
                )
                """
            )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS idempotency_receipts (
                scope TEXT NOT NULL,
                key TEXT NOT NULL,
                payload_hash TEXT NOT NULL,
                resource_type TEXT NOT NULL,
                resource_id TEXT NOT NULL,
                response_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (scope, key)
            )
            """
        )

    def _add_v4_columns(self, connection: sqlite3.Connection) -> None:
        session_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(sessions)").fetchall()
        }
        if "revision" not in session_columns:
            connection.execute(
                "ALTER TABLE sessions ADD COLUMN revision INTEGER NOT NULL DEFAULT 0"
            )
        tutor_columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(tutor_interactions)").fetchall()
        }
        if "revision" not in tutor_columns:
            connection.execute(
                "ALTER TABLE tutor_interactions ADD COLUMN revision INTEGER NOT NULL DEFAULT 0"
            )
        if "status" not in tutor_columns:
            connection.execute(
                "ALTER TABLE tutor_interactions ADD COLUMN status TEXT NOT NULL "
                "DEFAULT 'awaiting_answer'"
            )
        for row in connection.execute("SELECT session_id, session_json FROM sessions"):
            session = Session.model_validate_json(row[1])
            connection.execute(
                "UPDATE sessions SET revision = ? WHERE session_id = ?",
                (session.revision, row[0]),
            )
        for row in connection.execute(
            "SELECT interaction_id, record_json FROM tutor_interactions"
        ):
            record = deserialize_record(row[1])
            connection.execute(
                "UPDATE tutor_interactions SET revision = ?, status = ? "
                "WHERE interaction_id = ?",
                (record.view.revision, record.view.status.value, row[0]),
            )

    def _backup_before_migration(
        self, connection: sqlite3.Connection, version: int
    ) -> None:
        temporary = self.path.with_suffix(f".backup-{uuid.uuid4().hex}.tmp")
        try:
            with sqlite3.connect(temporary) as destination:
                connection.backup(destination)
            digest = hashlib.sha256(temporary.read_bytes()).hexdigest()[:12]
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        backup = self.path.with_suffix(f".schema{version}.{digest}.bak")
        if backup.exists():
            temporary.unlink()
        else:
            temporary.replace(backup)

    def save(
        self,
        snapshot: AnalysisSnapshot,
        *,
        reserve: bool = False,
        expected_state: AnalysisState | None = None,
        create: bool = False,
        receipt: tuple[str, str, str, str, str] | None = None,
    ) -> None:
        with self._transaction() as connection:
            if create:
                connection.execute(
                    """
                    INSERT INTO analyses (
                        analysis_id, generation, state, created_at, updated_at, snapshot_json
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    self._snapshot_values(snapshot),
                )
            elif expected_state is not None:
                cursor = connection.execute(
                    """
                    UPDATE analyses SET generation=?, state=?, updated_at=?, snapshot_json=?
                    WHERE analysis_id=? AND state=?
                    """,
                    (
                        snapshot.generation,
                        snapshot.state.value,
                        snapshot.updated_at.isoformat(),
                        snapshot.model_dump_json(),
                        snapshot.analysis_id,
                        expected_state.value,
                    ),
                )
                if cursor.rowcount != 1:
                    raise StorageConflict("The analysis changed before it could be saved.")
            else:
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
                    self._snapshot_values(snapshot),
                )
            if reserve:
                connection.execute(
                    "INSERT OR IGNORE INTO analysis_reservations (analysis_id) VALUES (?)",
                    (snapshot.analysis_id,),
                )
            if receipt:
                self._save_receipt(connection, *receipt)
            self._prune_analyses(connection)

    def create_with_cancellations(
        self,
        snapshot: AnalysisSnapshot,
        cancellations: Sequence[tuple[AnalysisSnapshot, AnalysisState]],
        *,
        reserve: bool,
        receipt: tuple[str, str, str, str, str] | None = None,
    ) -> None:
        with self._transaction() as connection:
            for cancelled, expected_state in cancellations:
                cursor = connection.execute(
                    """
                    UPDATE analyses SET generation=?, state=?, updated_at=?, snapshot_json=?
                    WHERE analysis_id=? AND state=?
                    """,
                    (
                        cancelled.generation,
                        cancelled.state.value,
                        cancelled.updated_at.isoformat(),
                        cancelled.model_dump_json(),
                        cancelled.analysis_id,
                        expected_state.value,
                    ),
                )
                if cursor.rowcount != 1:
                    raise StorageConflict("A pending analysis changed before replacement.")
            connection.execute(
                """
                INSERT INTO analyses (
                    analysis_id, generation, state, created_at, updated_at, snapshot_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                self._snapshot_values(snapshot),
            )
            if reserve:
                connection.execute(
                    "INSERT INTO analysis_reservations (analysis_id) VALUES (?)",
                    (snapshot.analysis_id,),
                )
            if receipt:
                self._save_receipt(connection, *receipt)
            self._prune_analyses(connection)

    @staticmethod
    def _snapshot_values(snapshot: AnalysisSnapshot) -> tuple[object, ...]:
        return (
            snapshot.analysis_id,
            snapshot.generation,
            snapshot.state.value,
            snapshot.created_at.isoformat(),
            snapshot.updated_at.isoformat(),
            snapshot.model_dump_json(),
        )

    def get(self, analysis_id: str) -> AnalysisSnapshot | None:
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT snapshot_json FROM analyses WHERE analysis_id = ?", (analysis_id,)
            ).fetchone()
        return AnalysisSnapshot.model_validate_json(row[0]) if row else None

    def recent(self, limit: int = 20) -> tuple[AnalysisSnapshot, ...]:
        bounded_limit = max(1, min(limit, 100))
        with self._transaction() as connection:
            rows = connection.execute(
                "SELECT snapshot_json FROM analyses ORDER BY updated_at DESC LIMIT ?",
                (bounded_limit,),
            ).fetchall()
        return tuple(AnalysisSnapshot.model_validate_json(row[0]) for row in rows)

    def save_session(
        self,
        session: Session,
        *,
        expected_revision: int | None = None,
        receipt: tuple[str, str, str, str, str] | None = None,
    ) -> None:
        with self._transaction() as connection:
            if expected_revision is None:
                connection.execute(
                    "INSERT INTO sessions "
                    "(session_id, updated_at, session_json, revision) VALUES (?, ?, ?, ?)",
                    (
                        session.session_id,
                        session.updated_at.isoformat(),
                        session.model_dump_json(),
                        session.revision,
                    ),
                )
            else:
                cursor = connection.execute(
                    "UPDATE sessions SET updated_at=?, session_json=?, revision=? "
                    "WHERE session_id=? AND revision=?",
                    (
                        session.updated_at.isoformat(),
                        session.model_dump_json(),
                        session.revision,
                        session.session_id,
                        expected_revision,
                    ),
                )
                if cursor.rowcount != 1:
                    raise StorageConflict("The session revision changed before it could be saved.")
            if receipt:
                self._save_receipt(connection, *receipt)
            connection.execute(
                "DELETE FROM session_analysis_refs WHERE session_id = ?",
                (session.session_id,),
            )
            for ply in session.plies:
                if ply.analysis_id is None:
                    continue
                connection.execute(
                    """
                    INSERT INTO session_analysis_refs (session_id, ply, analysis_id)
                    SELECT ?, ?, ?
                    WHERE EXISTS (
                        SELECT 1 FROM analyses WHERE analysis_id = ?
                    )
                    """,
                    (session.session_id, ply.ply, ply.analysis_id, ply.analysis_id),
                )
                connection.execute(
                    "DELETE FROM analysis_reservations WHERE analysis_id = ?",
                    (ply.analysis_id,),
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
            self._prune_analyses(connection)

    def get_session(self, session_id: str) -> Session | None:
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT session_json FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
        return Session.model_validate_json(row[0]) if row else None

    def recent_sessions(self, limit: int = 20) -> tuple[Session, ...]:
        bounded_limit = max(1, min(limit, 100))
        with self._transaction() as connection:
            rows = connection.execute(
                "SELECT session_json FROM sessions ORDER BY updated_at DESC LIMIT ?",
                (bounded_limit,),
            ).fetchall()
        return tuple(Session.model_validate_json(row[0]) for row in rows)

    def delete_session(self, session_id: str) -> bool:
        with self._transaction() as connection:
            cursor = connection.execute(
                "DELETE FROM sessions WHERE session_id = ?", (session_id,)
            )
            self._prune_analyses(connection)
        return cursor.rowcount > 0

    def save_tutor(
        self,
        record: TutorInteractionRecord,
        *,
        expected_revision: int | None = None,
        receipt: tuple[str, str, str, str, str] | None = None,
    ) -> None:
        with self._transaction() as connection:
            values = (
                record.view.interaction_id,
                record.view.session_id,
                record.view.updated_at.isoformat(),
                serialize_record(record),
                record.view.revision,
                record.view.status.value,
            )
            if expected_revision is None:
                connection.execute(
                    "INSERT INTO tutor_interactions "
                    "(interaction_id, session_id, updated_at, record_json, revision, status) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    values,
                )
            else:
                cursor = connection.execute(
                    "UPDATE tutor_interactions "
                    "SET updated_at=?, record_json=?, revision=?, status=? "
                    "WHERE interaction_id=? AND revision=?",
                    (
                        record.view.updated_at.isoformat(),
                        serialize_record(record),
                        record.view.revision,
                        record.view.status.value,
                        record.view.interaction_id,
                        expected_revision,
                    ),
                )
                if cursor.rowcount != 1:
                    raise StorageConflict("The tutor revision changed before it could be saved.")
            if receipt:
                self._save_receipt(connection, *receipt)
            connection.execute(
                """
                DELETE FROM tutor_interactions WHERE interaction_id IN (
                    SELECT interaction_id FROM tutor_interactions
                    WHERE session_id = ?
                    ORDER BY updated_at DESC LIMIT -1 OFFSET ?
                )
                """,
                (record.view.session_id, self.retention),
            )

    def get_tutor(self, interaction_id: str) -> TutorInteractionRecord | None:
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT record_json FROM tutor_interactions WHERE interaction_id = ?",
                (interaction_id,),
            ).fetchone()
        return deserialize_record(row[0]) if row else None

    def recent_tutors(
        self, session_id: str, limit: int = 20
    ) -> tuple[TutorInteractionRecord, ...]:
        bounded_limit = max(1, min(limit, 100))
        with self._transaction() as connection:
            rows = connection.execute(
                """
                SELECT record_json FROM tutor_interactions
                WHERE session_id = ? ORDER BY updated_at DESC LIMIT ?
                """,
                (session_id, bounded_limit),
            ).fetchall()
        return tuple(deserialize_record(row[0]) for row in rows)

    def _backfill_session_analysis_refs(self, connection: sqlite3.Connection) -> None:
        rows = connection.execute("SELECT session_json FROM sessions").fetchall()
        for row in rows:
            session = Session.model_validate_json(row[0])
            for ply in session.plies:
                if ply.analysis_id is None:
                    continue
                connection.execute(
                    """
                    INSERT OR IGNORE INTO session_analysis_refs (session_id, ply, analysis_id)
                    SELECT ?, ?, ?
                    WHERE EXISTS (
                        SELECT 1 FROM analyses WHERE analysis_id = ?
                    )
                    """,
                    (session.session_id, ply.ply, ply.analysis_id, ply.analysis_id),
                )

    def _prune_analyses(self, connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            DELETE FROM analyses WHERE analysis_id IN (
                SELECT analyses.analysis_id
                FROM analyses
                LEFT JOIN session_analysis_refs
                    ON session_analysis_refs.analysis_id = analyses.analysis_id
                LEFT JOIN analysis_reservations
                    ON analysis_reservations.analysis_id = analyses.analysis_id
                WHERE session_analysis_refs.analysis_id IS NULL
                    AND analysis_reservations.analysis_id IS NULL
                ORDER BY analyses.updated_at DESC LIMIT -1 OFFSET ?
            )
            """,
            (self.retention,),
        )

    def _recover_interrupted(self) -> None:
        placeholders = ",".join("?" for _ in ACTIVE_STATES)
        with self._transaction() as connection:
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

    def probe(self) -> Literal["ready"]:
        try:
            with self._connect_raw() as connection:
                connection.execute("BEGIN IMMEDIATE")
                result = connection.execute("PRAGMA quick_check").fetchone()
                connection.rollback()
        except sqlite3.DatabaseError as exc:
            if self._is_corruption(exc):
                raise StorageCorrupt("The history database is corrupt.") from exc
            raise StorageUnavailable("The history database is unavailable.") from exc
        except OSError as exc:
            raise StorageUnavailable("The history database is unavailable.") from exc
        if result is None or result[0] != "ok":
            raise StorageCorrupt("The history database failed its integrity check.")
        return "ready"

    def get_receipt(self, scope: str, key: str, payload_hash: str) -> str | None:
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT payload_hash, response_json FROM idempotency_receipts "
                "WHERE scope = ? AND key = ?",
                (scope, key),
            ).fetchone()
        if row is None:
            return None
        if row[0] != payload_hash:
            raise IdempotencyConflict("The idempotency key was used for another payload.")
        return str(row[1])

    @staticmethod
    def _save_receipt(
        connection: sqlite3.Connection,
        scope: str,
        key: str,
        payload_hash: str,
        resource_type: str,
        response_json: str,
    ) -> None:
        existing = connection.execute(
            "SELECT payload_hash FROM idempotency_receipts WHERE scope = ? AND key = ?",
            (scope, key),
        ).fetchone()
        if existing is not None:
            if existing[0] != payload_hash:
                raise IdempotencyConflict("The idempotency key was used for another payload.")
            return
        resource = AnalysisStore._resource_id(resource_type, response_json)
        connection.execute(
            "INSERT INTO idempotency_receipts "
            "(scope, key, payload_hash, resource_type, resource_id, response_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                scope,
                key,
                payload_hash,
                resource_type,
                resource,
                response_json,
                now_utc().isoformat(),
            ),
        )

    @staticmethod
    def _resource_id(resource_type: str, response_json: str) -> str:
        payload = json.loads(response_json)
        field = {
            "analysis": "analysis_id",
            "session": "session_id",
            "tutor": "interaction_id",
        }[resource_type]
        return str(payload[field])

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        try:
            with self._connect_raw() as connection:
                yield connection
        except StorageError:
            raise
        except sqlite3.DatabaseError as exc:
            if self._is_corruption(exc):
                raise StorageCorrupt("The history database is corrupt.") from exc
            raise StorageUnavailable("The history database is unavailable.") from exc
        except OSError as exc:
            raise StorageUnavailable("The history database is unavailable.") from exc

    def _connect_raw(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=0.25)
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    @staticmethod
    def _is_corruption(exc: sqlite3.DatabaseError) -> bool:
        message = str(exc).lower()
        return any(
            marker in message
            for marker in ("malformed", "not a database", "file is encrypted", "disk image")
        )
