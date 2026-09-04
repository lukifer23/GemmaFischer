from __future__ import annotations

import logging
import threading
import uuid
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
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
    canonical_hash,
    normalize_fen,
    now_utc,
)
from .engine import (
    EngineOperationCancelled,
    EngineOperationPreempted,
    EngineUnavailable,
    StockfishProvider,
    validate_player_move,
)
from .runtime import GemmaRuntime
from .storage import (
    AnalysisStore,
    StorageConflict,
    StorageCorrupt,
    StorageError,
    StorageUnavailable,
)
from .study import (
    MAX_MOMENTS,
    SCREENING_NODE_BUDGET,
    StudyWork,
    build_moment,
    decision_positions,
    evidence_ids,
    failed_study,
    new_study_work,
    parse_import,
    screening_candidate,
    select_shortlist,
)
from .study_domain import (
    AttemptOutcome,
    LearningMomentPrivate,
    PGNImportRequest,
    PracticeAttemptRequest,
    PracticeAttemptView,
    PracticeFeedback,
    PracticePhase,
    ProgressSummary,
    ReviewCard,
    StudyJobState,
    StudyJobView,
    StudyProgress,
)
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


LOGGER = logging.getLogger(__name__)


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
        self._study_jobs: dict[str, StudyWork] = {}
        self._sessions: dict[str, Session] = {}
        self._tutors: dict[str, TutorInteractionRecord] = {}
        self._session_locks: dict[str, threading.RLock] = {}
        self._creation_lock = threading.RLock()
        self._pending_ids: deque[str] = deque()
        self._study_pending_ids: deque[str] = deque()
        self._active_id: str | None = None
        self._active_study_id: str | None = None
        self._generation = 0
        self._job_retention = max(1, history_retention)
        self._condition = threading.Condition()
        self._closed = False
        self._storage_status = "ready" if self._store else "disabled"
        self._worker_status = "ready"
        self._worker = threading.Thread(target=self._run, name="gemmafischer-engine", daemon=True)
        self._worker.start()

    def submit_study(
        self, request: PGNImportRequest, *, idempotency_key: str | None = None
    ) -> StudyJobView:
        with self._condition:
            self._ensure_open_locked()
            payload_hash = canonical_hash(request.model_dump(mode="json"))
            if idempotency_key:
                replay = self._get_receipt("study:create", idempotency_key, payload_hash)
                if replay is not None:
                    return StudyJobView.model_validate_json(replay)
            work = new_study_work(request)
            if self._store:
                try:
                    self._store.save_study_job(
                        work.view,
                        create=True,
                        receipt=(
                            "study:create",
                            idempotency_key,
                            payload_hash,
                            "study",
                            work.view.model_dump_json(),
                        )
                        if idempotency_key
                        else None,
                    )
                except StorageError as exc:
                    self._note_storage_error(exc)
                    raise
            self._study_jobs[work.view.job_id] = work
            self._study_pending_ids.append(work.view.job_id)
            self._condition.notify_all()
            return work.view

    def get_study(self, job_id: str) -> StudyJobView | None:
        with self._condition:
            work = self._study_jobs.get(job_id)
            if work:
                return self._with_practice_statuses(work.view)
        try:
            stored = self._store.get_study_job(job_id) if self._store else None
        except StorageError as exc:
            self._note_storage_error(exc)
            raise
        return self._with_practice_statuses(stored) if stored else None

    def recent_studies(self, limit: int = 20) -> tuple[StudyJobView, ...]:
        if self._store:
            try:
                return tuple(
                    self._with_practice_statuses(item)
                    for item in self._store.recent_study_jobs(limit)
                )
            except StorageError as exc:
                self._note_storage_error(exc)
                raise
        with self._condition:
            values = sorted(
                (work.view for work in self._study_jobs.values()),
                key=lambda item: item.updated_at,
                reverse=True,
            )
            return tuple(values[: max(1, min(limit, 100))])

    def _with_practice_statuses(self, view: StudyJobView) -> StudyJobView:
        if not self._store or not view.moments:
            return view
        try:
            statuses = self._store.practice_statuses(view.job_id)
        except StorageError as exc:
            self._note_storage_error(exc)
            raise
        return view.model_copy(
            update={
                "moments": tuple(
                    moment.model_copy(
                        update={"practice_status": statuses.get(moment.moment_id, "new")}
                    )
                    for moment in view.moments
                )
            }
        )

    def cancel_study(self, job_id: str) -> StudyJobView | None:
        with self._condition:
            work = self._study_jobs.get(job_id)
            if work is None:
                stored = self._store.get_study_job(job_id) if self._store else None
                if stored is None:
                    return None
                work = StudyWork(request=None, view=stored)
                self._study_jobs[job_id] = work
            if work.view.state in {StudyJobState.READY, StudyJobState.CANCELLED}:
                return work.view
            work.cancelled = True
            if job_id in self._study_pending_ids:
                self._study_pending_ids.remove(job_id)
            if self._active_study_id == job_id and work.operation_id and self._provider:
                self._provider.interrupt_analysis(work.operation_id, "cancelled")
            self._update_study(work, StudyJobState.CANCELLED)
            return work.view

    def resume_study(self, job_id: str, expected_revision: int) -> StudyJobView:
        with self._condition:
            view = self.get_study(job_id)
            if view is None:
                raise KeyError(job_id)
            if view.revision != expected_revision:
                raise SessionConflict("The study job revision changed.")
            if view.state not in {
                StudyJobState.PAUSED_INTERRUPTED,
                StudyJobState.PAUSED_STORAGE,
                StudyJobState.FAILED,
                StudyJobState.CANCELLED,
            }:
                raise ValueError("This study job cannot be resumed")
            if view.game is None:
                raise ValueError("Re-submit a job interrupted before PGN parsing completed")
            work = StudyWork(request=None, view=view)
            self._study_jobs[job_id] = work
            self._update_study(work, StudyJobState.QUEUED)
            self._study_pending_ids.append(job_id)
            self._condition.notify_all()
            return work.view

    def delete_study(self, job_id: str) -> bool:
        if self.cancel_study(job_id) is None:
            return False
        with self._condition:
            self._study_jobs.pop(job_id, None)
        return self._store.delete_study_job(job_id) if self._store else True

    def submit_practice_attempt(
        self,
        job_id: str,
        moment_id: str,
        request: PracticeAttemptRequest,
        *,
        idempotency_key: str | None = None,
    ) -> PracticeAttemptView:
        with self._creation_lock:
            return self._submit_practice_attempt_locked(
                job_id,
                moment_id,
                request,
                idempotency_key=idempotency_key,
            )

    def _submit_practice_attempt_locked(
        self,
        job_id: str,
        moment_id: str,
        request: PracticeAttemptRequest,
        *,
        idempotency_key: str | None = None,
    ) -> PracticeAttemptView:
        payload_hash = canonical_hash(
            {
                "job_id": job_id,
                "moment_id": moment_id,
                "request": request.model_dump(mode="json"),
            }
        )
        receipt_scope = f"practice-attempt:{moment_id}"
        if idempotency_key:
            replay = self._get_receipt(receipt_scope, idempotency_key, payload_hash)
            if replay is not None:
                return PracticeAttemptView.model_validate_json(replay)
        job = self.get_study(job_id)
        if job is None:
            raise KeyError(job_id)
        if job.revision != request.expected_revision:
            raise SessionConflict("The study job revision changed.")
        private = self._private_moment(moment_id)
        if private is None or not any(
            moment.moment_id == private.view.moment_id for moment in job.moments
        ):
            raise KeyError(moment_id)
        previous = self._store.attempts_for_moment(moment_id) if self._store else ()
        self._validate_practice_phase(private, previous, request)
        attempt_number = len(previous) + 1
        transfer = request.phase is PracticePhase.TRANSFER
        if transfer and (
            private.transfer_fen is None
            or private.transfer_move_uci is None
            or private.transfer_move_san is None
        ):
            raise ValueError("No near-transfer challenge is available for this moment")
        challenge_fen = private.transfer_fen if transfer else private.view.fen
        preferred_move = private.transfer_move_uci if transfer else private.preferred_move_uci
        preferred_san = (
            private.transfer_move_san if transfer else private.preferred_move_san
        )
        assert challenge_fen is not None and preferred_move is not None
        assert preferred_san is not None
        evidence = self._engine().analyze(
            challenge_fen,
            request.move_uci,
            node_budget=self.node_budget,
        )
        comparison = evidence.move_comparison
        if comparison is None:
            raise ValueError("The submitted move could not be compared")
        if comparison.outcome == "engine_better":
            outcome = AttemptOutcome.INCORRECT
        elif request.move_uci == preferred_move:
            outcome = AttemptOutcome.CORRECT
        else:
            outcome = AttemptOutcome.EQUIVALENT
        reveal = (
            outcome is not AttemptOutcome.INCORRECT
            or request.phase is not PracticePhase.ORIGINAL
        )
        feedback = (
            PracticeFeedback(
                preferred_move_uci=preferred_move,
                preferred_move_san=preferred_san,
                message=(
                    "That move preserves the engine result."
                    if outcome is not AttemptOutcome.INCORRECT
                    else "Review the preferred move, then try the idea again."
                ),
                evidence_ids=(
                    (evidence.move_comparison.evidence_id,)
                    if transfer and evidence.move_comparison
                    else evidence_ids(private)
                ),
                next_phase=(
                    PracticePhase.TRANSFER
                    if not transfer and private.transfer_fen is not None
                    else None
                ),
                next_fen=private.transfer_fen if not transfer else None,
            )
            if reveal
            else None
        )
        attempt = PracticeAttemptView(
            attempt_id=uuid.uuid4().hex,
            moment_id=moment_id,
            phase=request.phase,
            attempt_number=attempt_number,
            submitted_move_uci=request.move_uci,
            outcome=outcome,
            hint_used=request.hint_used,
            feedback=feedback,
            created_at=now_utc(),
        )
        card = self._next_review_card(job_id, private, attempt) if feedback else None
        if self._store:
            try:
                self._store.save_attempt_and_card(
                    attempt,
                    card,
                    idempotency_key,
                    receipt=(
                        receipt_scope,
                        idempotency_key,
                        payload_hash,
                        "attempt",
                        attempt.model_dump_json(),
                    )
                    if idempotency_key
                    else None,
                )
            except StorageError as exc:
                self._note_storage_error(exc)
                raise
        return attempt

    def _validate_practice_phase(
        self,
        moment: LearningMomentPrivate,
        previous: tuple[PracticeAttemptView, ...],
        request: PracticeAttemptRequest,
    ) -> None:
        if request.phase is PracticePhase.ORIGINAL and previous:
            raise SessionConflict("The original attempt has already been submitted.")
        if request.phase is PracticePhase.RETRY and (
            not previous
            or previous[-1].phase is not PracticePhase.ORIGINAL
            or previous[-1].outcome is not AttemptOutcome.INCORRECT
        ):
            raise SessionConflict("Retry is available only after a missed original attempt.")
        if request.phase is PracticePhase.TRANSFER:
            if (
                moment.transfer_fen is None
                or moment.transfer_move_uci is None
                or moment.transfer_move_san is None
            ):
                raise ValueError("No near-transfer challenge is available for this moment")
            if not previous or previous[-1].phase not in {
                PracticePhase.ORIGINAL,
                PracticePhase.RETRY,
            } or previous[-1].feedback is None:
                raise SessionConflict("Complete the original challenge before transfer practice.")
        if request.phase is PracticePhase.DELAYED_REVIEW:
            card = self._store.get_review_card(moment.view.moment_id) if self._store else None
            if (
                not previous
                or previous[-1].feedback is None
                or card is None
                or card.mastered
                or card.due_at > now_utc()
            ):
                raise SessionConflict("This learning moment is not due for review.")

    def due_reviews(self, limit: int = 50) -> tuple[ReviewCard, ...]:
        if not self._store:
            return ()
        try:
            return self._store.due_reviews(now_utc().isoformat(), limit)
        except StorageError as exc:
            self._note_storage_error(exc)
            raise

    def progress(self) -> ProgressSummary:
        if self._store:
            try:
                return self._store.progress_summary(now_utc().isoformat())
            except StorageError as exc:
                self._note_storage_error(exc)
                raise
        return ProgressSummary(
            due=0,
            learning=0,
            retaining=0,
            mastered=0,
            attempts=0,
            original_accuracy=0,
            retry_accuracy=0,
            transfer_accuracy=0,
            delayed_accuracy=0,
        )

    def delete_progress(self) -> int:
        if not self._store:
            return 0
        try:
            return self._store.delete_progress()
        except StorageError as exc:
            self._note_storage_error(exc)
            raise

    def _private_moment(self, moment_id: str) -> LearningMomentPrivate | None:
        for work in self._study_jobs.values():
            for moment in work.private_moments:
                if moment.view.moment_id == moment_id:
                    return moment
        return self._store.get_learning_moment(moment_id) if self._store else None

    def _next_review_card(
        self, job_id: str, moment: LearningMomentPrivate, attempt: PracticeAttemptView
    ) -> ReviewCard:
        existing = self._store.get_review_card(moment.view.moment_id) if self._store else None
        successful = attempt.outcome is not AttemptOutcome.INCORRECT and not attempt.hint_used
        delayed = attempt.phase is PracticePhase.DELAYED_REVIEW
        successful_delayed = existing.successful_delayed_reviews if existing else 0
        lapses = existing.lapses if existing else 0
        interval = existing.interval_days if existing else 1
        if delayed and successful:
            successful_delayed += 1
            interval = (3, 7, 14, 30)[min(successful_delayed - 1, 3)]
        elif not successful:
            lapses += 1
            interval = 1
            if delayed:
                successful_delayed = 0
        due_at = datetime.now(UTC) + timedelta(days=interval)
        return ReviewCard(
            job_id=job_id,
            moment_id=moment.view.moment_id,
            moment=moment.view,
            concept_key=(
                moment.view.concept_keys[0]
                if moment.view.concept_keys
                else "calculation"
            ),
            due_at=due_at,
            interval_days=interval,
            successful_delayed_reviews=successful_delayed,
            lapses=lapses,
            mastered=successful_delayed >= 2,
        )

    def _update_study(
        self, work: StudyWork, state: StudyJobState, **values: object
    ) -> StudyJobView:
        previous_revision = work.view.revision
        candidate = work.view.model_copy(
            update={
                "revision": previous_revision + 1,
                "state": state,
                "updated_at": now_utc(),
                "error": None,
                **values,
            }
        )
        if self._store:
            try:
                self._store.save_study_job(candidate, expected_revision=previous_revision)
            except StorageError as exc:
                self._note_storage_error(exc)
                raise
        work.view = candidate
        return candidate

    def _process_study(self, work: StudyWork) -> None:
        try:
            if work.cancelled:
                return
            if work.view.game is None:
                if work.request is None:
                    raise ValueError("The original PGN is unavailable")
                self._update_study(work, StudyJobState.PARSING)
                game = parse_import(work.request)
                self._update_study(work, StudyJobState.SCREENING, game=game)
            else:
                game = work.view.game
                self._update_study(work, StudyJobState.SCREENING)

            decisions = decision_positions(game)
            candidates = []
            for completed, (ply, fen, move_uci, move_san) in enumerate(decisions, 1):
                if work.cancelled:
                    return
                operation_id = uuid.uuid4().hex
                work.operation_id = operation_id
                evidence = self._engine().analyze(
                    fen,
                    move_uci,
                    operation_id=operation_id,
                    node_budget=min(SCREENING_NODE_BUDGET, self.node_budget),
                    clear_hash=False,
                )
                candidate = screening_candidate(ply, fen, move_uci, move_san, evidence)
                if candidate is not None:
                    candidates.append(candidate)
                if completed % 4 == 0 or completed == len(decisions):
                    self._update_study(
                        work,
                        StudyJobState.SCREENING,
                        progress=StudyProgress(
                            completed_units=completed,
                            total_units=len(decisions),
                            current_ply=ply,
                        ),
                    )

            shortlist = select_shortlist(candidates)
            total = len(decisions) + len(shortlist)
            analyzed: list[LearningMomentPrivate] = []
            for offset, candidate in enumerate(shortlist, 1):
                if work.cancelled:
                    return
                self._update_study(
                    work,
                    StudyJobState.DEEP_ANALYSIS,
                    progress=StudyProgress(
                        completed_units=len(decisions) + offset - 1,
                        total_units=total,
                        current_ply=candidate.source_ply,
                    ),
                )
                operation_id = uuid.uuid4().hex
                work.operation_id = operation_id
                evidence = self._engine().analyze(
                    candidate.fen,
                    candidate.played_move_uci,
                    operation_id=operation_id,
                    node_budget=self.node_budget,
                )
                confirmed = screening_candidate(
                    candidate.source_ply,
                    candidate.fen,
                    candidate.played_move_uci,
                    candidate.played_move_san,
                    evidence,
                )
                if confirmed is not None and len(analyzed) < MAX_MOMENTS:
                    analyzed.append(build_moment(confirmed, evidence, len(analyzed) + 1))

            work.operation_id = None
            moments = analyzed[:MAX_MOMENTS]
            for index, moment in enumerate(moments):
                alternatives = [
                    item
                    for item in analyzed
                    if item.view.moment_id != moment.view.moment_id
                    and set(item.view.concept_keys) & set(moment.view.concept_keys)
                ]
                if not alternatives:
                    continue
                transfer_source = alternatives[index % len(alternatives)]
                moments[index] = moment.model_copy(
                    update={
                        "transfer_fen": transfer_source.view.fen,
                        "transfer_move_uci": transfer_source.preferred_move_uci,
                        "transfer_move_san": transfer_source.preferred_move_san,
                    }
                )
            work.private_moments = moments
            if self._store:
                self._store.replace_learning_moments(work.view.job_id, moments)
            self._update_study(
                work,
                StudyJobState.READY,
                moments=tuple(moment.view for moment in moments),
                progress=StudyProgress(completed_units=total, total_units=total),
            )
        except EngineOperationCancelled:
            work.operation_id = None
        except EngineOperationPreempted:
            work.operation_id = None
            if not work.cancelled:
                self._update_study(work, StudyJobState.QUEUED)
                with self._condition:
                    if work.view.job_id not in self._study_pending_ids:
                        self._study_pending_ids.appendleft(work.view.job_id)
                        self._condition.notify_all()
        except StorageError as exc:
            if work.cancelled:
                return
            self._note_storage_error(exc)
            work.view = work.view.model_copy(
                update={"state": StudyJobState.PAUSED_STORAGE, "updated_at": now_utc()}
            )
            with self._condition:
                if (
                    not isinstance(exc, StorageCorrupt)
                    and not work.cancelled
                    and work.view.job_id not in self._study_pending_ids
                ):
                    self._study_pending_ids.appendleft(work.view.job_id)
        except ValueError as exc:
            if work.cancelled:
                return
            work.view = failed_study(work, "INVALID_STUDY_INPUT", str(exc), False)
            if self._store:
                self._store.save_study_job(work.view)
        except EngineUnavailable:
            if work.cancelled:
                return
            work.view = failed_study(
                work, "ENGINE_UNAVAILABLE", "The chess engine is unavailable.", True
            )
            if self._store:
                self._store.save_study_job(work.view)
        except Exception:
            if work.cancelled:
                return
            LOGGER.exception("Study worker failed")
            work.view = failed_study(
                work, "STUDY_ENGINE_FAILURE", "The game study could not be completed.", True
            )
            if self._store:
                self._store.save_study_job(work.view)

    def submit(
        self,
        request: AnalysisRequest,
        *,
        durable: bool = False,
        idempotency_key: str | None = None,
    ) -> AnalysisSnapshot:
        with self._condition:
            self._ensure_open_locked()
            payload_hash = canonical_hash(request.model_dump(mode="json"))
            if idempotency_key:
                replay = self._get_receipt("analysis:create", idempotency_key, payload_hash)
                if replay is not None:
                    return AnalysisSnapshot.model_validate_json(replay)
            generation = self._generation + 1
            analysis_id = uuid.uuid4().hex
            timestamp = now_utc()
            snapshot = AnalysisSnapshot(
                analysis_id=analysis_id,
                generation=generation,
                state=AnalysisState.QUEUED,
                created_at=timestamp,
                updated_at=timestamp,
                request=request,
            )
            cancelled: list[tuple[_Job, AnalysisSnapshot]] = []
            if not durable:
                for pending_id in tuple(self._pending_ids):
                    job = self._jobs[pending_id]
                    if not job.durable:
                        cancelled.append(
                            (
                                job,
                                job.snapshot.model_copy(
                                    update={
                                        "state": AnalysisState.CANCELLED,
                                        "updated_at": now_utc(),
                                    }
                                ),
                            )
                        )
            if self._store:
                try:
                    self._store.create_with_cancellations(
                        snapshot,
                        tuple((candidate, job.snapshot.state) for job, candidate in cancelled),
                        reserve=durable,
                        receipt=(
                            "analysis:create",
                            idempotency_key,
                            payload_hash,
                            "analysis",
                            snapshot.model_dump_json(),
                        )
                        if idempotency_key
                        else None,
                    )
                except StorageError as exc:
                    self._note_storage_error(exc)
                    raise
            for job, candidate in cancelled:
                job.cancelled = True
                job.snapshot = candidate
                if candidate.analysis_id in self._pending_ids:
                    self._pending_ids.remove(candidate.analysis_id)
            self._generation = generation
            self._jobs[analysis_id] = _Job(snapshot=snapshot, durable=durable)
            self._pending_ids.append(analysis_id)
            self._condition.notify()
            return snapshot

    @property
    def history_enabled(self) -> bool:
        return self._store is not None

    @property
    def storage_status(self) -> str:
        with self._condition:
            return self._storage_status

    @property
    def worker_status(self) -> str:
        with self._condition:
            return self._worker_status

    def retry_storage(self) -> str:
        if self._store is None:
            return "disabled"
        try:
            self._store.probe()
        except StorageCorrupt:
            with self._condition:
                self._storage_status = "corrupt"
                self._worker_status = "failed"
            raise
        except StorageUnavailable:
            with self._condition:
                self._storage_status = "degraded"
                self._worker_status = "paused_storage"
            raise
        with self._condition:
            self._storage_status = "ready"
            self._worker_status = "ready"
            self._condition.notify_all()
        return "ready"

    def get(self, analysis_id: str) -> AnalysisSnapshot | None:
        with self._condition:
            job = self._jobs.get(analysis_id)
            if job:
                return job.snapshot
            if not self._store:
                return None
            try:
                return self._store.get(analysis_id)
            except StorageError as exc:
                self._note_storage_error(exc)
                raise

    def recent(self, limit: int = 20) -> tuple[AnalysisSnapshot, ...]:
        if self._store:
            try:
                return self._store.recent(limit)
            except StorageError as exc:
                self._note_storage_error(exc)
                raise
        with self._condition:
            snapshots = (job.snapshot for job in self._jobs.values())
            return tuple(
                sorted(snapshots, key=lambda item: item.updated_at, reverse=True)[:limit]
            )

    def create_session(
        self, request: CreateSessionRequest, *, idempotency_key: str | None = None
    ) -> Session:
        with self._creation_lock:
            return self._create_session(request, idempotency_key=idempotency_key)

    def _create_session(
        self, request: CreateSessionRequest, *, idempotency_key: str | None
    ) -> Session:
        with self._condition:
            self._ensure_open_locked()
            payload_hash = canonical_hash(request.model_dump(mode="json"))
            if idempotency_key:
                replay = self._get_receipt("session:create", idempotency_key, payload_hash)
                if replay is not None:
                    return Session.model_validate_json(replay)
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
        self._save_session(
            session,
            receipt=(
                "session:create",
                idempotency_key,
                payload_hash,
                "session",
                session.model_dump_json(),
            )
            if idempotency_key
            else None,
        )
        return session

    def get_session(self, session_id: str) -> Session | None:
        with self._condition:
            cached = self._sessions.get(session_id)
            if cached is not None or not self._store:
                return cached
            try:
                return self._store.get_session(session_id)
            except StorageError as exc:
                self._note_storage_error(exc)
                raise

    def recent_sessions(self, limit: int = 20) -> tuple[Session, ...]:
        if self._store:
            try:
                return self._store.recent_sessions(limit)
            except StorageError as exc:
                self._note_storage_error(exc)
                raise
        with self._condition:
            return tuple(
                sorted(self._sessions.values(), key=lambda item: item.updated_at, reverse=True)[
                    :limit
                ]
            )

    def delete_session(self, session_id: str) -> bool:
        lock = self._session_lock(session_id)
        with lock, self._condition:
            existed = self._sessions.get(session_id) is not None
            try:
                deleted = self._store.delete_session(session_id) if self._store else existed
            except StorageError as exc:
                self._note_storage_error(exc)
                raise
            if not (deleted or existed):
                return False
            self._sessions.pop(session_id, None)
            self._tutors = {
                key: value
                for key, value in self._tutors.items()
                if value.view.session_id != session_id
            }
            self._session_locks.pop(session_id, None)
            return True

    def create_tutor(
        self,
        session_id: str,
        request: CreateTutorRequest,
        *,
        idempotency_key: str | None = None,
    ) -> TutorInteractionView:
        lock = self._session_lock(session_id)
        with lock:
            scope = f"session:{session_id}:tutor:create"
            payload_hash = canonical_hash(request.model_dump(mode="json"))
            if idempotency_key:
                replay = self._get_receipt(scope, idempotency_key, payload_hash)
                if replay is not None:
                    return TutorInteractionView.model_validate_json(replay)
            active = next(
                (
                    item
                    for item in self.recent_tutors(session_id, 20)
                    if item.status not in {TutorStatus.COMPLETE, TutorStatus.DISMISSED}
                ),
                None,
            )
            if active is not None:
                if active.question.source_analysis_id == request.source_analysis_id:
                    return active
                raise TutorStateConflict(
                    "Dismiss the active tutor interaction before starting another."
                )
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
                question_template_id=analysis.coaching.question_template_id,
                hint_template_id=analysis.coaching.hint_template_id,
            )
            self._save_tutor(
                record,
                expected_revision=None,
                receipt=(
                    scope,
                    idempotency_key,
                    payload_hash,
                    "tutor",
                    record.view.model_dump_json(),
                )
                if idempotency_key
                else None,
            )
            return record.view

    def get_tutor(self, interaction_id: str) -> TutorInteractionView | None:
        record = self._tutors.get(interaction_id)
        if record is None and self._store:
            try:
                record = self._store.get_tutor(interaction_id)
            except StorageError as exc:
                self._note_storage_error(exc)
                raise
        return record.view if record else None

    def recent_tutors(
        self, session_id: str, limit: int = 20
    ) -> tuple[TutorInteractionView, ...]:
        if self._store:
            try:
                return tuple(
                    record.view for record in self._store.recent_tutors(session_id, limit)
                )
            except StorageError as exc:
                self._note_storage_error(exc)
                raise
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
            record = self._tutors.get(interaction_id)
            if record is None and self._store:
                try:
                    record = self._store.get_tutor(interaction_id)
                except StorageError as exc:
                    self._note_storage_error(exc)
                    raise
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
            self._save_tutor(updated, expected_revision=record.view.revision)
            return updated.view

    def _save_tutor(
        self,
        record: TutorInteractionRecord,
        *,
        expected_revision: int | None,
        receipt: tuple[str, str, str, str, str] | None = None,
    ) -> None:
        if self._store:
            try:
                self._store.save_tutor(
                    record, expected_revision=expected_revision, receipt=receipt
                )
            except StorageConflict as exc:
                raise SessionConflict("The tutor changed before it could be saved.") from exc
            except StorageError as exc:
                self._note_storage_error(exc)
                raise
        self._tutors[record.view.interaction_id] = record

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
        self._save_session(updated, expected_revision=session.revision)
        return updated

    def _save_session(
        self,
        session: Session,
        *,
        expected_revision: int | None = None,
        receipt: tuple[str, str, str, str, str] | None = None,
    ) -> None:
        with self._condition:
            self._ensure_open_locked()
            if self._store:
                try:
                    self._store.save_session(
                        session, expected_revision=expected_revision, receipt=receipt
                    )
                except StorageConflict as exc:
                    raise SessionConflict("The session changed before it could be saved.") from exc
                except StorageError as exc:
                    self._note_storage_error(exc)
                    raise
            self._sessions[session.session_id] = session

    def cancel(self, analysis_id: str) -> AnalysisSnapshot | None:
        with self._condition:
            if analysis_id not in self._jobs:
                return None
            self._cancel_locked(analysis_id)
            snapshot = self._jobs[analysis_id].snapshot
            self._evict_jobs()
            return snapshot

    def close(self) -> None:
        provider = None
        with self._condition:
            if self._closed:
                return
            self._closed = True
            for analysis_id, job in tuple(self._jobs.items()):
                if job.snapshot.state not in TERMINAL_STATES:
                    try:
                        self._cancel_locked(analysis_id, interrupt_active=False)
                    except StorageError:
                        LOGGER.exception("Failed to persist cancellation during shutdown")
            for work in self._study_jobs.values():
                if work.view.state not in {
                    StudyJobState.READY,
                    StudyJobState.CANCELLED,
                    StudyJobState.FAILED,
                }:
                    work.cancelled = True
                    try:
                        self._update_study(work, StudyJobState.PAUSED_INTERRUPTED)
                    except StorageError:
                        LOGGER.exception("Failed to persist study interruption during shutdown")
            self._condition.notify_all()
            provider = self._provider
        try:
            if provider is not None:
                provider.close()
        finally:
            self._worker.join(timeout=2)
            if self._worker.is_alive():
                self._worker.join(timeout=2)

    def play_move(self, request: BoardMoveRequest) -> BoardMoveResult:
        validate_player_move(request.fen, request.move_uci)
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
        if not was_terminal:
            candidate = job.snapshot.model_copy(
                update={"state": AnalysisState.CANCELLED, "updated_at": now_utc()}
            )
            self._persist(candidate, expected_state=job.snapshot.state)
            job.snapshot = candidate
        job.cancelled = True
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
            previous_state = job.snapshot.state
            candidate = job.snapshot.model_copy(
                update={"state": state, "updated_at": now_utc(), **values}
            )
            self._persist(candidate, expected_state=previous_state)
            job.snapshot = candidate

    def _persist(
        self,
        snapshot: AnalysisSnapshot,
        *,
        reserve: bool = False,
        expected_state: AnalysisState | None = None,
    ) -> None:
        if self._store:
            try:
                self._store.save(snapshot, reserve=reserve, expected_state=expected_state)
            except StorageError as exc:
                self._note_storage_error(exc)
                raise

    def _get_receipt(self, scope: str, key: str, payload_hash: str) -> str | None:
        if not self._store:
            return None
        try:
            return self._store.get_receipt(scope, key, payload_hash)
        except StorageError as exc:
            self._note_storage_error(exc)
            raise

    def _run(self) -> None:
        while True:
            with self._condition:
                while not self._closed and (
                    (not self._pending_ids and not self._study_pending_ids)
                    or self._storage_status not in {"ready", "disabled"}
                ):
                    self._condition.wait()
                if self._closed:
                    return
                if self._pending_ids:
                    pulled_id = self._pending_ids.popleft()
                    analysis_id: str | None = pulled_id
                    self._active_id = pulled_id
                    job: _Job | None = self._jobs[pulled_id]
                    study_work: StudyWork | None = None
                else:
                    study_id = self._study_pending_ids.popleft()
                    self._active_study_id = study_id
                    study_work = self._study_jobs[study_id]
                    analysis_id = None
                    job = None
            if study_work is not None:
                try:
                    self._process_study(study_work)
                finally:
                    with self._condition:
                        if self._active_study_id == study_work.view.job_id:
                            self._active_study_id = None
                continue
            assert analysis_id is not None and job is not None
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
                                question_template_id=selection.question_template_id,
                                hint_template_id=selection.hint_template_id,
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
                try:
                    with self._condition:
                        if job.cancelled:
                            pass
                        elif job.durable or not self._pending_ids:
                            self._update(job, AnalysisState.QUEUED)
                            self._pending_ids.appendleft(analysis_id)
                            self._condition.notify()
                        else:
                            self._cancel_locked(analysis_id, interrupt_active=False)
                except StorageError as exc:
                    self._pause_for_storage(job, analysis_id, exc)
            except StorageError as exc:
                self._pause_for_storage(job, analysis_id, exc)
            except (ValueError, EngineUnavailable) as exc:
                code = "INVALID_INPUT" if isinstance(exc, ValueError) else "ENGINE_UNAVAILABLE"
                message = (
                    "The analysis request is invalid."
                    if isinstance(exc, ValueError)
                    else "The chess engine is unavailable."
                )
                self._record_failure(
                    job,
                    analysis_id,
                    code,
                    message,
                    not isinstance(exc, ValueError),
                )
            except Exception:
                LOGGER.exception("Analysis worker failed")
                self._record_failure(
                    job,
                    analysis_id,
                    "ENGINE_FAILURE",
                    "The chess engine could not complete the analysis.",
                    True,
                )
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

    def _record_failure(
        self,
        job: _Job,
        analysis_id: str,
        code: str,
        message: str,
        retryable: bool,
    ) -> None:
        try:
            self._fail(job, code, message, retryable)
        except StorageError as exc:
            self._pause_for_storage(job, analysis_id, exc)

    def _pause_for_storage(
        self, job: _Job, analysis_id: str, exc: StorageError
    ) -> None:
        LOGGER.exception("Storage failure paused durable analysis work")
        self._note_storage_error(exc)
        with self._condition:
            if (
                not isinstance(exc, StorageCorrupt)
                and not job.cancelled
                and analysis_id not in self._pending_ids
            ):
                self._pending_ids.appendleft(analysis_id)

    def _note_storage_error(self, exc: StorageError) -> None:
        with self._condition:
            if isinstance(exc, StorageCorrupt):
                self._storage_status = "corrupt"
                self._worker_status = "failed"
            else:
                self._storage_status = "degraded"
                self._worker_status = "paused_storage"
