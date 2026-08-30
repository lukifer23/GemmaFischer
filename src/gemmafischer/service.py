from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass
from pathlib import Path

from .coach import deterministic_coach, merge_model_claims, validate_model_claims
from .domain import (
    TERMINAL_STATES,
    AnalysisRequest,
    AnalysisSnapshot,
    AnalysisState,
    BoardMoveRequest,
    BoardMoveResult,
    CoachingResult,
    EngineTurnRequest,
    EngineTurnResult,
    ErrorDetail,
    now_utc,
)
from .engine import EngineUnavailable, StockfishProvider
from .runtime import GemmaRuntime, ModelUnavailable
from .storage import AnalysisStore


@dataclass
class _Job:
    snapshot: AnalysisSnapshot
    cancelled: bool = False


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
        self._jobs: dict[str, _Job] = {}
        self._pending_id: str | None = None
        self._generation = 0
        self._condition = threading.Condition()
        self._closed = False
        self._worker = threading.Thread(target=self._run, name="gemmafischer-engine", daemon=True)
        self._worker.start()

    def submit(self, request: AnalysisRequest) -> AnalysisSnapshot:
        with self._condition:
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
            if self._pending_id:
                self._cancel_locked(self._pending_id)
            self._jobs[analysis_id] = _Job(snapshot=snapshot)
            self._persist(snapshot)
            self._pending_id = analysis_id
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

    def cancel(self, analysis_id: str) -> AnalysisSnapshot | None:
        with self._condition:
            if analysis_id not in self._jobs:
                return None
            self._cancel_locked(analysis_id)
            return self._jobs[analysis_id].snapshot

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        self._worker.join(timeout=2)

    def play_move(self, request: BoardMoveRequest) -> BoardMoveResult:
        provider = StockfishProvider(self.engine_path, self.node_budget)
        return provider.play_move(
            request.fen,
            request.move_uci,
            engine_reply=request.engine_reply,
            difficulty=request.difficulty,
        )

    def play_engine_turn(self, request: EngineTurnRequest) -> EngineTurnResult:
        provider = StockfishProvider(self.engine_path, self.node_budget)
        return provider.play_engine_turn(
            request.fen,
            difficulty=request.difficulty,
        )

    def _cancel_locked(self, analysis_id: str) -> None:
        job = self._jobs[analysis_id]
        job.cancelled = True
        if job.snapshot.state not in TERMINAL_STATES:
            job.snapshot = job.snapshot.model_copy(
                update={"state": AnalysisState.CANCELLED, "updated_at": now_utc()}
            )
            self._persist(job.snapshot)
        if self._pending_id == analysis_id:
            self._pending_id = None

    def _update(self, job: _Job, state: AnalysisState, **values: object) -> None:
        with self._condition:
            if job.cancelled:
                return
            job.snapshot = job.snapshot.model_copy(
                update={"state": state, "updated_at": now_utc(), **values}
            )
            self._persist(job.snapshot)

    def _persist(self, snapshot: AnalysisSnapshot) -> None:
        if self._store:
            self._store.save(snapshot)

    def _run(self) -> None:
        while True:
            with self._condition:
                while not self._closed and self._pending_id is None:
                    self._condition.wait()
                if self._closed:
                    return
                analysis_id = self._pending_id
                self._pending_id = None
                assert analysis_id is not None
                job = self._jobs[analysis_id]
            if job.cancelled:
                continue
            request = job.snapshot.request
            try:
                self._update(job, AnalysisState.VALIDATING)
                self._update(job, AnalysisState.ENGINE_RUNNING)
                provider = StockfishProvider(self.engine_path, self.node_budget)
                evidence = provider.analyze(request.fen, request.considered_move_uci)
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
                        selection = self._model_runtime.select_claims(
                            evidence, request.rating_bucket
                        )
                        valid, removed = validate_model_claims(evidence, selection.claims)
                        if len(valid) >= 2:
                            merged = merge_model_claims(valid, baseline.claims)
                            coaching = CoachingResult(
                                summary=baseline.summary,
                                claims=merged,
                                removed_claim_codes=selection.removed_claim_codes + removed,
                                source="gemma",
                            )
                        else:
                            state = AnalysisState.ENGINE_ONLY
                    except (ModelUnavailable, ValueError, RuntimeError):
                        state = AnalysisState.ENGINE_ONLY
                self._update(
                    job,
                    state,
                    evidence=evidence,
                    coaching=coaching,
                )
            except (ValueError, EngineUnavailable) as exc:
                code = "INVALID_INPUT" if isinstance(exc, ValueError) else "ENGINE_UNAVAILABLE"
                self._fail(job, code, str(exc), retryable=not isinstance(exc, ValueError))
            except Exception as exc:
                self._fail(job, "ENGINE_FAILURE", str(exc), retryable=True)

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
