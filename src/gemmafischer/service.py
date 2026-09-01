from __future__ import annotations

import threading
import uuid
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from .coach import (
    deterministic_coach,
    merge_model_claims,
    order_lesson_plan,
    validate_model_claims,
)
from .domain import (
    TERMINAL_STATES,
    AnalysisRequest,
    AnalysisSnapshot,
    AnalysisState,
    BoardMoveRequest,
    BoardMoveResult,
    CoachingResult,
    CreateSessionRequest,
    CreateTutorRequest,
    EngineTurnRequest,
    EngineTurnResult,
    ErrorDetail,
    Ply,
    RatingBucket,
    Session,
    SessionCommandRequest,
    SessionMode,
    SessionStatus,
    TutorCommandRequest,
    TutorInteractionView,
    TutorStatus,
    Workflow,
    normalize_fen,
    now_utc,
)
from .engine import (
    EngineOperationCancelled,
    EngineOperationPreempted,
    EngineUnavailable,
    StockfishProvider,
)
from .runtime import GemmaRuntime
from .storage import AnalysisStore
from .tutor import (
    TutorInteractionRecord,
    answer_follow_up,
    create_interaction,
    dismiss,
    grade_answer,
    reveal_hint,
)


@dataclass
class _Job:
    snapshot: AnalysisSnapshot
    durable: bool = False
    cancelled: bool = False
    operation_id: str | None = None


class SessionConflict(RuntimeError):
    pass


class TutorStateConflict(SessionConflict):
    pass


class AnalysisService:
    """Single-worker local orchestrator with one active and latest-pending semantics."""

    def __init__(
        self,
        engine_path: str | None = None,
        node_budget: int = 250_000,
        full_profile: bool = False,
        history_path: Path | None = None,
        history_retention: int = 250,
    ) -> None:
        self.engine_path = engine_path
        self.node_budget = node_budget
        self.full_profile = full_profile
        self._store = (
            AnalysisStore(history_path, retention=history_retention) if history_path else None
        )
        self._model_runtime: GemmaRuntime | None = None
        self.model_status = "disabled" if not full_profile else "loading"
        self._provider: StockfishProvider | None = None
        self._jobs: dict[str, _Job] = {}
        self._sessions: dict[str, Session] = {}
        self._tutors: dict[str, TutorInteractionRecord] = {}
        self._session_locks: dict[str, threading.RLock] = {}
        self._pending_ids: deque[str] = deque()
        self._active_id: str | None = None
        self._generation = 0
        self._job_retention = max(1, history_retention)
        self._condition = threading.Condition()
        self._closed = False
        self._worker = threading.Thread(target=self._run, name="gemmafischer-engine", daemon=True)
        self._worker.start()

    def submit(self, request: AnalysisRequest, *, durable: bool = False) -> AnalysisSnapshot:
        with self._condition:
            self._ensure_open_locked()
            self._generation += 1
            analysis_id = uuid.uuid4().hex
            timestamp = now_utc()
            snapshot = AnalysisSnapshot(
                analysis_id=analysis_id,
                generation=self._generation,
                state=AnalysisState.QUEUED,
                created_at=timestamp,
                updated_at=timestamp,
                request=request,
            )
            if not durable:
                for pending_id in tuple(self._pending_ids):
                    if not self._jobs[pending_id].durable:
                        self._cancel_locked(pending_id)
            self._jobs[analysis_id] = _Job(snapshot=snapshot, durable=durable)
            self._persist(snapshot, reserve=durable)
            self._pending_ids.append(analysis_id)
            self._condition.notify()
            return snapshot

    @property
    def history_enabled(self) -> bool:
        return self._store is not None

    def get(self, analysis_id: str) -> AnalysisSnapshot | None:
        with self._condition:
            job = self._jobs.get(analysis_id)
            if job:
                return job.snapshot
            return self._store.get(analysis_id) if self._store else None

    def recent(self, limit: int = 20) -> tuple[AnalysisSnapshot, ...]:
        if self._store:
            return self._store.recent(limit)
        with self._condition:
            snapshots = (job.snapshot for job in self._jobs.values())
            return tuple(
                sorted(snapshots, key=lambda item: item.updated_at, reverse=True)[:limit]
            )

    def create_session(self, request: CreateSessionRequest) -> Session:
        with self._condition:
            self._ensure_open_locked()
        board, fen = normalize_fen(request.fen)
        timestamp = now_utc()
        outcome = board.outcome()
        session = Session(
            session_id=uuid.uuid4().hex,
            revision=0,
            mode=request.mode,
            status=SessionStatus.COMPLETE if board.is_game_over() else SessionStatus.ACTIVE,
            initial_fen=fen,
            fen=fen,
            turn="white" if board.turn else "black",
            player_color=request.player_color,
            white_difficulty=request.white_difficulty,
            black_difficulty=request.black_difficulty,
            rating_bucket=request.rating_bucket,
            created_at=timestamp,
            updated_at=timestamp,
            outcome=outcome.result() if outcome else None,
        )
        self._save_session(session)
        return session

    def get_session(self, session_id: str) -> Session | None:
        with self._condition:
            return self._sessions.get(session_id) or (
                self._store.get_session(session_id) if self._store else None
            )

    def recent_sessions(self, limit: int = 20) -> tuple[Session, ...]:
        if self._store:
            return self._store.recent_sessions(limit)
        with self._condition:
            return tuple(
                sorted(self._sessions.values(), key=lambda item: item.updated_at, reverse=True)[
                    :limit
                ]
            )

    def delete_session(self, session_id: str) -> bool:
        lock = self._session_lock(session_id)
        with lock, self._condition:
            existed = self._sessions.pop(session_id, None) is not None
            self._tutors = {
                key: value
                for key, value in self._tutors.items()
                if value.view.session_id != session_id
            }
            deleted = self._store.delete_session(session_id) if self._store else existed
            self._session_locks.pop(session_id, None)
            return deleted or existed

    def create_tutor(
        self, session_id: str, request: CreateTutorRequest
    ) -> TutorInteractionView:
        lock = self._session_lock(session_id)
        with lock:
            session = self.get_session(session_id)
            if session is None:
                raise KeyError(session_id)
            analysis = self.get(request.source_analysis_id)
            if analysis is None or analysis.evidence is None or analysis.coaching is None:
                raise ValueError("The source analysis is not complete")
            session_positions = {session.initial_fen, session.fen}
            for ply in session.plies:
                session_positions.update((ply.fen_before, ply.fen_after))
            linked = request.source_analysis_id in {
                ply.analysis_id for ply in session.plies if ply.analysis_id is not None
            }
            if not linked and analysis.request.fen not in session_positions:
                raise ValueError("The source analysis does not belong to a session position")
            record = create_interaction(
                uuid.uuid4().hex,
                session_id,
                request.source_analysis_id,
                analysis.evidence,
            )
            self._save_tutor(record)
            return record.view

    def get_tutor(self, interaction_id: str) -> TutorInteractionView | None:
        record = self._tutors.get(interaction_id) or (
            self._store.get_tutor(interaction_id) if self._store else None
        )
        return record.view if record else None

    def recent_tutors(
        self, session_id: str, limit: int = 20
    ) -> tuple[TutorInteractionView, ...]:
        if self._store:
            return tuple(record.view for record in self._store.recent_tutors(session_id, limit))
        return tuple(
            sorted(
                (
                    record.view
                    for record in self._tutors.values()
                    if record.view.session_id == session_id
                ),
                key=lambda item: item.updated_at,
                reverse=True,
            )[:limit]
        )

    def command_tutor(
        self,
        session_id: str,
        interaction_id: str,
        command: TutorCommandRequest,
    ) -> TutorInteractionView:
        lock = self._session_lock(session_id)
        with lock:
            record = self._tutors.get(interaction_id) or (
                self._store.get_tutor(interaction_id) if self._store else None
            )
            if record is None or record.view.session_id != session_id:
                raise KeyError(interaction_id)
            if command.expected_revision != record.view.revision:
                raise SessionConflict(
                    f"Expected tutor revision {command.expected_revision}; "
                    f"current revision is {record.view.revision}."
                )
            if record.view.status in {TutorStatus.COMPLETE, TutorStatus.DISMISSED}:
                raise TutorStateConflict("This tutor interaction is already terminal")
            if command.action == "hint":
                updated = reveal_hint(record)
            elif command.action == "answer":
                assert command.move_uci is not None
                evidence = self._engine().analyze(record.view.question.fen, command.move_uci)
                updated = grade_answer(record, command.move_uci, evidence)
            elif command.action == "follow_up":
                assert command.option_id is not None
                updated = answer_follow_up(record, command.option_id)
            else:
                updated = dismiss(record)
            self._save_tutor(updated)
            return updated.view

    def _save_tutor(self, record: TutorInteractionRecord) -> None:
        self._tutors[record.view.interaction_id] = record
        if self._store:
            self._store.save_tutor(record)

    def command_session(self, session_id: str, command: SessionCommandRequest) -> Session:
        # A revision check and its mutation are one transaction from the API's
        # perspective. Engine work may be slow, so serialize per session rather
        # than blocking unrelated games.
        lock = self._session_lock(session_id)
        with lock:
            return self._command_session_locked(session_id, command)

    def _command_session_locked(
        self, session_id: str, command: SessionCommandRequest
    ) -> Session:
        with self._condition:
            self._ensure_open_locked()
        session = self.get_session(session_id)
        if session is None:
            raise KeyError(session_id)
        if command.expected_revision != session.revision:
            raise SessionConflict(
                f"Expected revision {command.expected_revision}; "
                f"current revision is {session.revision}."
            )
        if command.action == "pause":
            return self._replace_session(session, status=SessionStatus.PAUSED)
        if command.action == "resume":
            if session.outcome is not None:
                raise ValueError("A completed game cannot be resumed")
            return self._replace_session(session, status=SessionStatus.ACTIVE)
        if command.action == "undo":
            if not session.plies:
                raise ValueError("There is no move to undo")
            remaining = session.plies[:-1]
            if session.mode is SessionMode.PLAYER:
                # Return to the human's previous decision point. For a White
                # player this normally removes the engine reply and their move;
                # for a Black player it removes their move but keeps White's.
                while remaining:
                    previous_board, _ = normalize_fen(remaining[-1].fen_after)
                    previous_turn = "white" if previous_board.turn else "black"
                    if previous_turn == session.player_color:
                        break
                    remaining = remaining[:-1]
            fen = remaining[-1].fen_after if remaining else session.initial_fen
            board, fen = normalize_fen(fen)
            return self._replace_session(
                session,
                fen=fen,
                turn="white" if board.turn else "black",
                plies=remaining,
                outcome=None,
                status=SessionStatus.ACTIVE,
            )
        if session.status is not SessionStatus.ACTIVE:
            raise ValueError("The session is not active")
        if command.action == "player_move":
            if session.mode is not SessionMode.PLAYER or session.turn != session.player_color:
                raise ValueError("It is not the player's turn")
            assert command.move_uci is not None
            result = self._engine().play_move(
                session.fen,
                command.move_uci,
                engine_reply=False,
                difficulty=(
                    session.white_difficulty
                    if session.turn == "white"
                    else session.black_difficulty
                ),
            )
            # Do not enqueue the full review ahead of the immediate engine
            # reply. Both operations share one persistent Stockfish process;
            # queuing here made the reply wait behind a 250k-node analysis.
            # A terminal human move is the only case with no reply to protect.
            analysis = (
                self._submit_ply_review(
                    result.fen_before,
                    session.rating_bucket,
                    result.human_move_uci,
                )
                if result.outcome
                else None
            )
            ply = Ply(
                ply=len(session.plies) + 1,
                move_uci=result.human_move_uci,
                move_san=result.human_move_san,
                fen_before=result.fen_before,
                fen_after=result.fen,
                actor="player",
                analysis_id=analysis.analysis_id if analysis else None,
            )
            return self._advance_session(session, ply, result.outcome)
        if command.action == "engine_move":
            if session.mode is SessionMode.PLAYER and session.turn == session.player_color:
                raise ValueError("It is the player's turn")
            difficulty = (
                session.white_difficulty if session.turn == "white" else session.black_difficulty
            )
            engine_result = self._engine().play_engine_turn(session.fen, difficulty=difficulty)
            plies = session.plies
            if session.mode is SessionMode.PLAYER and plies:
                player_ply = plies[-1]
                if player_ply.actor == "player" and player_ply.analysis_id is None:
                    player_analysis = self._submit_ply_review(
                        player_ply.fen_before,
                        session.rating_bucket,
                        player_ply.move_uci,
                    )
                    plies = (
                        *plies[:-1],
                        player_ply.model_copy(
                            update={"analysis_id": player_analysis.analysis_id}
                        ),
                    )
            engine_analysis = (
                self._submit_ply_review(
                    engine_result.fen_before,
                    session.rating_bucket,
                    engine_result.move_uci,
                )
                if session.mode is SessionMode.EXHIBITION
                else None
            )
            ply = Ply(
                ply=len(session.plies) + 1,
                move_uci=engine_result.move_uci,
                move_san=engine_result.move_san,
                fen_before=engine_result.fen_before,
                fen_after=engine_result.fen,
                actor="engine_white" if session.turn == "white" else "engine_black",
                analysis_id=engine_analysis.analysis_id if engine_analysis else None,
            )
            if plies is not session.plies:
                session = session.model_copy(update={"plies": plies})
            return self._advance_session(session, ply, engine_result.outcome)
        raise ValueError(f"Unsupported session action: {command.action}")

    def _submit_ply_review(
        self, fen: str, rating_bucket: RatingBucket, move_uci: str
    ) -> AnalysisSnapshot:
        return self.submit(
            AnalysisRequest(
                mode=Workflow.COMPARE,
                fen=fen,
                rating_bucket=rating_bucket,
                considered_move_uci=move_uci,
            ),
            durable=True,
        )

    def _advance_session(self, session: Session, ply: Ply, outcome: str | None) -> Session:
        board, fen = normalize_fen(ply.fen_after)
        return self._replace_session(
            session,
            fen=fen,
            turn="white" if board.turn else "black",
            plies=(*session.plies, ply),
            outcome=outcome,
            status=SessionStatus.COMPLETE if outcome else SessionStatus.ACTIVE,
        )

    def _replace_session(self, session: Session, **values: object) -> Session:
        updated = session.model_copy(
            update={"revision": session.revision + 1, "updated_at": now_utc(), **values}
        )
        self._save_session(updated)
        return updated

    def _save_session(self, session: Session) -> None:
        with self._condition:
            self._ensure_open_locked()
            self._sessions[session.session_id] = session
            if self._store:
                self._store.save_session(session)

    def cancel(self, analysis_id: str) -> AnalysisSnapshot | None:
        with self._condition:
            if analysis_id not in self._jobs:
                return None
            self._cancel_locked(analysis_id)
            snapshot = self._jobs[analysis_id].snapshot
            self._evict_jobs()
            return snapshot

    def close(self) -> None:
        with self._condition:
            if self._closed:
                return
            self._closed = True
            for analysis_id, job in tuple(self._jobs.items()):
                if job.snapshot.state not in TERMINAL_STATES:
                    self._cancel_locked(analysis_id, interrupt_active=False)
            self._condition.notify_all()
        if self._provider is not None:
            self._provider.close()
        self._worker.join(timeout=2)
        if self._worker.is_alive():
            self._worker.join(timeout=2)

    def play_move(self, request: BoardMoveRequest) -> BoardMoveResult:
        return self._engine().play_move(
            request.fen,
            request.move_uci,
            engine_reply=request.engine_reply,
            difficulty=request.difficulty,
        )

    def play_engine_turn(self, request: EngineTurnRequest) -> EngineTurnResult:
        return self._engine().play_engine_turn(
            request.fen,
            difficulty=request.difficulty,
        )

    def _cancel_locked(self, analysis_id: str, *, interrupt_active: bool = True) -> None:
        job = self._jobs[analysis_id]
        was_terminal = job.snapshot.state in TERMINAL_STATES
        operation_id = job.operation_id
        job.cancelled = True
        if not was_terminal:
            job.snapshot = job.snapshot.model_copy(
                update={"state": AnalysisState.CANCELLED, "updated_at": now_utc()}
            )
            self._persist(job.snapshot)
        if analysis_id in self._pending_ids:
            self._pending_ids.remove(analysis_id)
        elif (
            interrupt_active
            and self._active_id == analysis_id
            and operation_id
            and self._provider is not None
        ):
            self._provider.interrupt_analysis(operation_id, "cancelled")

    def _ensure_open_locked(self) -> None:
        if self._closed:
            raise RuntimeError("The analysis service is closed")

    def _session_lock(self, session_id: str) -> threading.RLock:
        with self._condition:
            return self._session_locks.setdefault(session_id, threading.RLock())

    def _engine(self) -> StockfishProvider:
        with self._condition:
            self._ensure_open_locked()
            if self._provider is None:
                self._provider = StockfishProvider(self.engine_path, self.node_budget)
            return self._provider

    def _update(self, job: _Job, state: AnalysisState, **values: object) -> None:
        with self._condition:
            if job.cancelled:
                return
            job.snapshot = job.snapshot.model_copy(
                update={"state": state, "updated_at": now_utc(), **values}
            )
            self._persist(job.snapshot)

    def _persist(self, snapshot: AnalysisSnapshot, *, reserve: bool = False) -> None:
        if self._store:
            self._store.save(snapshot, reserve=reserve)

    def _run(self) -> None:
        while True:
            with self._condition:
                while not self._closed and not self._pending_ids:
                    self._condition.wait()
                if self._closed:
                    return
                analysis_id = self._pending_ids.popleft()
                self._active_id = analysis_id
                job = self._jobs[analysis_id]
            if job.cancelled:
                with self._condition:
                    if self._active_id == analysis_id:
                        self._active_id = None
                continue
            request = job.snapshot.request
            try:
                self._update(job, AnalysisState.VALIDATING)
                self._update(job, AnalysisState.ENGINE_RUNNING)
                operation_id = uuid.uuid4().hex
                with self._condition:
                    if job.cancelled:
                        continue
                    job.operation_id = operation_id
                evidence = self._engine().analyze(
                    request.fen,
                    request.considered_move_uci,
                    operation_id=operation_id,
                )
                job.operation_id = None
                if job.cancelled:
                    continue
                if request.considered_move_uci:
                    self._update(job, AnalysisState.COMPARISON_RUNNING, evidence=evidence)
                baseline = deterministic_coach(
                    evidence, request.rating_bucket, request.considered_move_uci
                )
                coaching = baseline
                state = AnalysisState.COMPLETE
                if self.full_profile:
                    self._update(job, AnalysisState.MODEL_RUNNING, evidence=evidence)
                    try:
                        if self._model_runtime is None:
                            self._model_runtime = GemmaRuntime()
                            self.model_status = "ready"
                        selection = self._model_runtime.select_claims(
                            evidence, request.rating_bucket
                        )
                        valid, removed = validate_model_claims(evidence, selection.claims)
                        if valid or selection.concept_ids:
                            merged = merge_model_claims(valid, baseline.claims)
                            coaching = CoachingResult(
                                summary=baseline.summary,
                                claims=merged,
                                removed_claim_codes=(
                                    selection.removed_claim_codes
                                    + removed
                                    + ("MODEL_SELECTION_MERGED_WITH_REQUIRED_BASELINE",)
                                ),
                                source="gemma",
                                lesson_plan=order_lesson_plan(
                                    baseline.lesson_plan, selection.concept_ids
                                ),
                            )
                        else:
                            state = AnalysisState.ENGINE_ONLY
                    except Exception:
                        self.model_status = "degraded"
                        state = AnalysisState.ENGINE_ONLY
                self._update(
                    job,
                    state,
                    evidence=evidence,
                    coaching=coaching,
                )
                self._evict_jobs()
            except EngineOperationCancelled:
                job.operation_id = None
            except EngineOperationPreempted:
                job.operation_id = None
                with self._condition:
                    if job.cancelled:
                        pass
                    elif job.durable or not self._pending_ids:
                        self._update(job, AnalysisState.QUEUED)
                        self._pending_ids.appendleft(analysis_id)
                        self._condition.notify()
                    else:
                        self._cancel_locked(analysis_id, interrupt_active=False)
            except (ValueError, EngineUnavailable) as exc:
                code = "INVALID_INPUT" if isinstance(exc, ValueError) else "ENGINE_UNAVAILABLE"
                self._fail(job, code, str(exc), retryable=not isinstance(exc, ValueError))
            except Exception as exc:
                self._fail(job, "ENGINE_FAILURE", str(exc), retryable=True)
            finally:
                with self._condition:
                    if self._active_id == analysis_id:
                        self._active_id = None
                self._evict_jobs()

    def _evict_jobs(self) -> None:
        with self._condition:
            terminal = sorted(
                (
                    item for item in self._jobs.values()
                    if item.snapshot.state in TERMINAL_STATES
                ),
                key=lambda item: item.snapshot.updated_at,
                reverse=True,
            )
            for item in terminal[self._job_retention :]:
                self._jobs.pop(item.snapshot.analysis_id, None)

    def _fail(self, job: _Job, code: str, message: str, retryable: bool) -> None:
        error = ErrorDetail(
            code=code,
            message=message,
            stage="engine",
            retryable=retryable,
            remediation=("gemmafischer doctor --profile deterministic",),
            request_id=job.snapshot.analysis_id,
        )
        self._update(job, AnalysisState.FAILED, error=error)
