from __future__ import annotations

import secrets
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Header, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from starlette.middleware.gzip import GZipMiddleware

from . import __version__
from .domain import (
    AnalysisAccepted,
    AnalysisList,
    AnalysisRequest,
    AnalysisSnapshot,
    BoardMoveRequest,
    CreateSessionRequest,
    CreateTutorRequest,
    DeleteResult,
    EngineTurnRequest,
    ErrorEnvelope,
    LegalMovesRequest,
    LegalMovesResult,
    RuntimeCapabilities,
    Session,
    SessionCommandRequest,
    SessionList,
    StorageRecoveryResult,
    TutorCommandRequest,
    TutorInteractionList,
    TutorInteractionView,
)
from .engine import EngineUnavailable, legal_moves_for_square, resolve_stockfish
from .service import AnalysisService, SessionConflict, TutorStateConflict
from .storage import (
    IdempotencyConflict,
    StorageConflict,
    StorageCorrupt,
    StorageError,
    StorageUnavailable,
)
from .study_domain import (
    LearningMomentView,
    PGNImportRequest,
    PracticeAttemptRequest,
    PracticeAttemptView,
    ProgressSummary,
    ReviewCardList,
    StudyJobAccepted,
    StudyJobCommand,
    StudyJobList,
    StudyJobView,
)

STATIC_DIR = Path(__file__).with_name("static")
ALLOWED_HOSTS = {"127.0.0.1", "localhost", "testserver"}
MAX_REQUEST_BODY_BYTES = 300 * 1024


class RequestBodyTooLarge(Exception):
    """Raised before request parsing when the local API body exceeds its limit."""


def create_app(
    *,
    engine_path: str | None = None,
    full_profile: bool = False,
    node_budget: int = 250_000,
    capability_token: str | None = None,
    history_path: Path | None = None,
) -> FastAPI:
    token = capability_token or secrets.token_urlsafe(32)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.service = AnalysisService(
            engine_path, node_budget, full_profile, history_path=history_path
        )
        yield
        app.state.service.close()

    app = FastAPI(
        title="GemmaFischer local API",
        version=__version__,
        docs_url=None,
        redoc_url=None,
        lifespan=lifespan,
    )
    app.add_middleware(GZipMiddleware, minimum_size=500, compresslevel=6)
    app.state.capability_token = token

    @app.middleware("http")
    async def local_security(request: Request, call_next):  # type: ignore[no-untyped-def]
        host = (request.url.hostname or "").lower()
        if host not in ALLOWED_HOSTS:
            return _error("INVALID_HOST", "Only loopback requests are accepted.", "security", 403)
        origin = request.headers.get("origin")
        if origin and origin not in {
            f"http://{host}",
            f"http://{host}:{request.url.port}",
        }:
            return _error("INVALID_ORIGIN", "The request origin is not local.", "security", 403)
        protected_history_read = request.method == "GET" and request.url.path.startswith(
            ("/api/v1/study", "/api/v1/reviews", "/api/v1/progress")
        )
        if request.method in {"POST", "PUT", "PATCH", "DELETE"} or protected_history_read:
            supplied = request.headers.get("x-gemmafischer-token") or request.cookies.get(
                "gemmafischer_token"
            )
            if not secrets.compare_digest(supplied or "", token):
                return _error(
                    "CAPABILITY_TOKEN_REQUIRED",
                    "A valid per-launch capability token is required.",
                    "security",
                    403,
                )
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                declared_bytes = int(content_length)
            except ValueError:
                return _error(
                    "INVALID_CONTENT_LENGTH",
                    "Content-Length must be a non-negative integer.",
                    "validation",
                    400,
                    field="content-length",
                )
            if declared_bytes < 0:
                return _error(
                    "INVALID_CONTENT_LENGTH",
                    "Content-Length must be a non-negative integer.",
                    "validation",
                    400,
                    field="content-length",
                )
            if declared_bytes > MAX_REQUEST_BODY_BYTES:
                return _request_too_large()

        received_bytes = 0
        original_receive = request._receive

        async def limited_receive():  # type: ignore[no-untyped-def]
            nonlocal received_bytes
            message = await original_receive()
            if message["type"] == "http.request":
                received_bytes += len(message.get("body", b""))
                if received_bytes > MAX_REQUEST_BODY_BYTES:
                    raise RequestBodyTooLarge
            return message

        request._receive = limited_receive
        try:
            # Buffer the bounded body here so chunked or falsely declared input
            # is rejected before FastAPI attempts JSON/model parsing.
            await request.body()
            response = await call_next(request)
        except RequestBodyTooLarge:
            return _request_too_large()
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
            "connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'"
        )
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        first = exc.errors()[0]
        field = str(first.get("loc", ["request"])[-1])
        return _error(
            "INVALID_REQUEST",
            "The request field is invalid.",
            "validation",
            422,
            field=field,
        )

    @app.exception_handler(StorageUnavailable)
    async def storage_unavailable(_request: Request, _exc: StorageUnavailable) -> JSONResponse:
        return _error(
            "STORAGE_UNAVAILABLE",
            "Local history is temporarily unavailable; the last committed state is unchanged.",
            "storage",
            503,
            retryable=True,
            remediation=("Retry storage recovery, then repeat the action.",),
        )

    @app.exception_handler(StorageCorrupt)
    async def storage_corrupt(_request: Request, _exc: StorageCorrupt) -> JSONResponse:
        return _error(
            "STORAGE_CORRUPT",
            "Local history failed its integrity check and has been left untouched.",
            "storage",
            503,
            remediation=("Stop the app, back up the database, and run doctor.",),
        )

    @app.exception_handler(StorageConflict)
    async def storage_conflict(_request: Request, _exc: StorageConflict) -> JSONResponse:
        return _error(
            "STORAGE_CONFLICT",
            "The durable state changed; refresh it before retrying.",
            "storage",
            409,
            remediation=("Refresh the resource before retrying.",),
        )

    @app.exception_handler(IdempotencyConflict)
    async def idempotency_conflict(
        _request: Request, _exc: IdempotencyConflict
    ) -> JSONResponse:
        return _error(
            "IDEMPOTENCY_CONFLICT",
            "This idempotency key was already used with a different request.",
            "idempotency",
            409,
            remediation=("Generate a new idempotency key for a different request.",),
        )

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        response = FileResponse(STATIC_DIR / "index.html")
        response.headers["Cache-Control"] = "no-store"
        response.set_cookie(
            "gemmafischer_token",
            token,
            httponly=True,
            samesite="strict",
            max_age=12 * 60 * 60,
        )
        return response

    @app.get("/app.css", include_in_schema=False)
    async def css() -> FileResponse:
        return FileResponse(
            STATIC_DIR / "app.css", media_type="text/css", headers={"Cache-Control": "no-store"}
        )

    @app.get("/app.js", include_in_schema=False)
    async def js() -> FileResponse:
        return FileResponse(
            STATIC_DIR / "app.js",
            media_type="text/javascript",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/study.js", include_in_schema=False)
    async def study_js() -> FileResponse:
        return FileResponse(
            STATIC_DIR / "study.js",
            media_type="text/javascript",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/api/v1/health")
    async def health(request: Request) -> dict[str, object]:
        try:
            resolve_stockfish(request.app.state.service.engine_path)
            engine_available = True
        except EngineUnavailable:
            engine_available = False
        return {
            "ok": True,
            "version": __version__,
            "engine_configured": request.app.state.service.engine_path is not None,
            "engine_available": engine_available,
            "model_profile": "full" if full_profile else "deterministic",
            "history_enabled": request.app.state.service.history_enabled,
            "storage_status": request.app.state.service.storage_status,
            "worker_status": request.app.state.service.worker_status,
        }

    @app.get("/api/v1/capabilities", response_model=RuntimeCapabilities)
    async def capabilities(request: Request) -> RuntimeCapabilities:
        try:
            resolve_stockfish(request.app.state.service.engine_path)
            engine_status = "ready"
        except EngineUnavailable:
            engine_status = "missing"
        return RuntimeCapabilities(
            engine_status=engine_status,  # type: ignore[arg-type]
            model_status=request.app.state.service.model_status,
            history_enabled=request.app.state.service.history_enabled,
            storage_status=request.app.state.service.storage_status,
            worker_status=request.app.state.service.worker_status,
        )

    @app.post("/api/v1/storage/retry", response_model=StorageRecoveryResult)
    def retry_storage(request: Request) -> StorageRecoveryResult:
        service: AnalysisService = request.app.state.service
        service.retry_storage()
        return StorageRecoveryResult(
            storage_status=service.storage_status,  # type: ignore[arg-type]
            worker_status=service.worker_status,  # type: ignore[arg-type]
        )

    @app.post(
        "/api/v1/analyses",
        status_code=202,
        response_model=AnalysisAccepted,
        responses={422: {"model": ErrorEnvelope}},
    )
    def create_analysis(
        payload: AnalysisRequest,
        request: Request,
        idempotency_key: str | None = Header(
            default=None, alias="Idempotency-Key", min_length=8, max_length=128
        ),
    ) -> JSONResponse:
        snapshot = request.app.state.service.submit(
            payload, idempotency_key=idempotency_key
        )
        location = f"/api/v1/analyses/{snapshot.analysis_id}"
        return JSONResponse(
            status_code=202,
            content={
                "analysis_id": snapshot.analysis_id,
                "state": snapshot.state,
                "generation": snapshot.generation,
            },
            headers={"Location": location, "Retry-After": "1"},
        )

    @app.get("/api/v1/analyses", response_model=AnalysisList)
    def list_analyses(
        request: Request, limit: int = Query(default=20, ge=1, le=100)
    ) -> JSONResponse:
        snapshots = request.app.state.service.recent(limit)
        return JSONResponse(
            content={
                "items": [snapshot.model_dump(mode="json") for snapshot in snapshots],
                "count": len(snapshots),
            }
        )

    @app.get(
        "/api/v1/analyses/{analysis_id}",
        response_model=AnalysisSnapshot,
        responses={404: {"model": ErrorEnvelope}},
    )
    def get_analysis(analysis_id: str, request: Request) -> JSONResponse:
        snapshot = request.app.state.service.get(analysis_id)
        if snapshot is None:
            return _error("ANALYSIS_NOT_FOUND", "No analysis has this ID.", "lookup", 404)
        return JSONResponse(content=snapshot.model_dump(mode="json"))

    @app.delete(
        "/api/v1/analyses/{analysis_id}",
        response_model=AnalysisSnapshot,
        responses={404: {"model": ErrorEnvelope}},
    )
    def cancel_analysis(analysis_id: str, request: Request) -> JSONResponse:
        snapshot = request.app.state.service.cancel(analysis_id)
        if snapshot is None:
            return _error("ANALYSIS_NOT_FOUND", "No analysis has this ID.", "lookup", 404)
        return JSONResponse(content=snapshot.model_dump(mode="json"))

    @app.post("/api/v1/board/moves", deprecated=True)
    def play_board_move(payload: BoardMoveRequest, request: Request) -> JSONResponse:
        try:
            result = request.app.state.service.play_move(payload)
        except ValueError:
            return _error(
                "ILLEGAL_MOVE",
                "The submitted move is not legal in this position.",
                "move_validation",
                422,
                field="move_uci",
            )
        except Exception:
            return _error(
                "ENGINE_FAILURE", "The chess engine could not complete the move.", "engine", 503
            )
        return JSONResponse(content=result.model_dump(mode="json"))

    @app.post("/api/v1/board/legal-moves", deprecated=True)
    def board_legal_moves(payload: LegalMovesRequest) -> JSONResponse:
        try:
            result = legal_moves_for_square(payload.fen, payload.from_square)
        except ValueError:
            return _error(
                "INVALID_POSITION",
                "The position or source square is invalid.",
                "move_validation",
                422,
                field="fen",
            )
        return JSONResponse(content=result.model_dump(mode="json"))

    @app.post("/api/v1/board/engine-turn", deprecated=True)
    def board_engine_turn(payload: EngineTurnRequest, request: Request) -> JSONResponse:
        try:
            result = request.app.state.service.play_engine_turn(payload)
        except ValueError:
            return _error(
                "INVALID_POSITION",
                "The position is invalid.",
                "move_validation",
                422,
                field="fen",
            )
        except Exception:
            return _error(
                "ENGINE_FAILURE", "The chess engine could not complete its turn.", "engine", 503
            )
        return JSONResponse(content=result.model_dump(mode="json"))

    @app.post(
        "/api/v1/sessions",
        response_model=Session,
        status_code=201,
        responses={422: {"model": ErrorEnvelope}},
    )
    def create_session(
        payload: CreateSessionRequest,
        request: Request,
        idempotency_key: str | None = Header(
            default=None, alias="Idempotency-Key", min_length=8, max_length=128
        ),
    ) -> Session:
        service: AnalysisService = request.app.state.service
        return service.create_session(payload, idempotency_key=idempotency_key)

    @app.get("/api/v1/sessions", response_model=SessionList)
    def list_sessions(
        request: Request, limit: int = Query(default=20, ge=1, le=100)
    ) -> SessionList:
        sessions = request.app.state.service.recent_sessions(limit)
        return SessionList(items=sessions, count=len(sessions))

    @app.get(
        "/api/v1/sessions/{session_id}",
        response_model=Session,
        responses={404: {"model": ErrorEnvelope}},
    )
    def get_session(session_id: str, request: Request) -> Session | JSONResponse:
        service: AnalysisService = request.app.state.service
        session = service.get_session(session_id)
        if session is None:
            return _error("SESSION_NOT_FOUND", "No session has this ID.", "lookup", 404)
        return session

    @app.get(
        "/api/v1/sessions/{session_id}/legal-moves",
        response_model=LegalMovesResult,
        responses={404: {"model": ErrorEnvelope}},
    )
    def session_legal_moves(
        session_id: str,
        from_square: str,
        request: Request,
    ) -> JSONResponse:
        service: AnalysisService = request.app.state.service
        session = service.get_session(session_id)
        if session is None:
            return _error("SESSION_NOT_FOUND", "No session has this ID.", "lookup", 404)
        try:
            result = legal_moves_for_square(session.fen, from_square)
        except ValueError:
            return _error(
                "INVALID_POSITION",
                "The stored session position is invalid.",
                "move_validation",
                422,
            )
        return JSONResponse(content=result.model_dump(mode="json"))

    @app.post(
        "/api/v1/sessions/{session_id}/commands",
        response_model=Session,
        responses={404: {"model": ErrorEnvelope}, 409: {"model": ErrorEnvelope}},
    )
    def command_session(
        session_id: str, payload: SessionCommandRequest, request: Request
    ) -> Session | JSONResponse:
        service: AnalysisService = request.app.state.service
        try:
            return service.command_session(session_id, payload)
        except KeyError:
            return _error("SESSION_NOT_FOUND", "No session has this ID.", "lookup", 404)
        except SessionConflict:
            return _error(
                "REVISION_CONFLICT",
                "The session changed; refresh it and retry the command.",
                "session",
                409,
            )
        except ValueError:
            return _error(
                "INVALID_SESSION_COMMAND",
                "The session command is invalid for the current position or state.",
                "session",
                422,
            )
        except StorageError:
            raise
        except Exception:
            return _error(
                "ENGINE_FAILURE",
                "The chess engine could not complete the session command.",
                "engine",
                503,
                retryable=True,
                remediation=("Retry the command or run doctor if the failure persists.",),
            )

    @app.delete(
        "/api/v1/sessions/{session_id}",
        response_model=DeleteResult,
        responses={404: {"model": ErrorEnvelope}},
    )
    def delete_session(session_id: str, request: Request) -> DeleteResult | JSONResponse:
        if not request.app.state.service.delete_session(session_id):
            return _error("SESSION_NOT_FOUND", "No session has this ID.", "lookup", 404)
        return DeleteResult()

    @app.post(
        "/api/v1/sessions/{session_id}/tutor",
        response_model=TutorInteractionView,
        status_code=201,
    )
    def create_tutor(
        session_id: str,
        payload: CreateTutorRequest,
        request: Request,
        idempotency_key: str | None = Header(
            default=None, alias="Idempotency-Key", min_length=8, max_length=128
        ),
    ) -> TutorInteractionView | JSONResponse:
        service: AnalysisService = request.app.state.service
        try:
            return service.create_tutor(
                session_id, payload, idempotency_key=idempotency_key
            )
        except KeyError:
            return _error("SESSION_NOT_FOUND", "No session has this ID.", "lookup", 404)
        except TutorStateConflict:
            return _error(
                "TUTOR_STATE_CONFLICT",
                "Finish or dismiss the active practice before starting another.",
                "tutor",
                409,
            )
        except ValueError:
            return _error(
                "TUTOR_UNAVAILABLE",
                "Practice requires a completed review from this session.",
                "tutor",
                422,
            )

    @app.get(
        "/api/v1/sessions/{session_id}/tutor",
        response_model=TutorInteractionList,
    )
    def list_tutors(
        session_id: str, request: Request, limit: int = Query(default=20, ge=1, le=100)
    ) -> TutorInteractionList | JSONResponse:
        service: AnalysisService = request.app.state.service
        if service.get_session(session_id) is None:
            return _error("SESSION_NOT_FOUND", "No session has this ID.", "lookup", 404)
        items = service.recent_tutors(session_id, limit)
        return TutorInteractionList(items=items, count=len(items))

    @app.get(
        "/api/v1/sessions/{session_id}/tutor/{interaction_id}",
        response_model=TutorInteractionView,
    )
    def get_tutor(
        session_id: str, interaction_id: str, request: Request
    ) -> TutorInteractionView | JSONResponse:
        service: AnalysisService = request.app.state.service
        interaction = service.get_tutor(interaction_id)
        if interaction is None or interaction.session_id != session_id:
            return _error("TUTOR_NOT_FOUND", "No tutor interaction has this ID.", "lookup", 404)
        return interaction

    @app.get(
        "/api/v1/sessions/{session_id}/tutor/{interaction_id}/legal-moves",
        response_model=LegalMovesResult,
    )
    def tutor_legal_moves(
        session_id: str,
        interaction_id: str,
        from_square: str,
        request: Request,
    ) -> LegalMovesResult | JSONResponse:
        service: AnalysisService = request.app.state.service
        interaction = service.get_tutor(interaction_id)
        if interaction is None or interaction.session_id != session_id:
            return _error("TUTOR_NOT_FOUND", "No tutor interaction has this ID.", "lookup", 404)
        try:
            return legal_moves_for_square(interaction.question.fen, from_square)
        except ValueError:
            return _error(
                "INVALID_POSITION",
                "The stored practice position is invalid.",
                "tutor",
                422,
            )

    @app.post(
        "/api/v1/sessions/{session_id}/tutor/{interaction_id}/commands",
        response_model=TutorInteractionView,
    )
    def command_tutor(
        session_id: str,
        interaction_id: str,
        payload: TutorCommandRequest,
        request: Request,
    ) -> TutorInteractionView | JSONResponse:
        service: AnalysisService = request.app.state.service
        try:
            return service.command_tutor(session_id, interaction_id, payload)
        except KeyError:
            return _error("TUTOR_NOT_FOUND", "No tutor interaction has this ID.", "lookup", 404)
        except TutorStateConflict:
            return _error(
                "TUTOR_STATE_CONFLICT",
                "This practice interaction is already complete or dismissed.",
                "tutor",
                409,
            )
        except SessionConflict:
            return _error(
                "TUTOR_REVISION_CONFLICT",
                "The practice interaction changed; refresh it and retry.",
                "tutor",
                409,
            )
        except ValueError:
            return _error(
                "INVALID_TUTOR_COMMAND",
                "The practice command is invalid for the current state.",
                "tutor",
                422,
            )
        except StorageError:
            raise
        except Exception:
            return _error(
                "ENGINE_FAILURE",
                "The chess engine could not grade this tutor answer.",
                "engine",
                503,
                retryable=True,
                remediation=("Retry the answer or run doctor if the failure persists.",),
            )

    @app.post(
        "/api/v1/study-jobs",
        status_code=202,
        response_model=StudyJobAccepted,
        responses={422: {"model": ErrorEnvelope}},
    )
    def create_study_job(
        payload: PGNImportRequest,
        request: Request,
        idempotency_key: str | None = Header(
            default=None, alias="Idempotency-Key", min_length=8, max_length=128
        ),
    ) -> JSONResponse:
        service: AnalysisService = request.app.state.service
        job = service.submit_study(payload, idempotency_key=idempotency_key)
        location = f"/api/v1/study-jobs/{job.job_id}"
        accepted = StudyJobAccepted(job_id=job.job_id, state=job.state, revision=job.revision)
        return JSONResponse(
            status_code=202,
            content=accepted.model_dump(mode="json"),
            headers={"Location": location, "Retry-After": "1"},
        )

    @app.get("/api/v1/study-jobs", response_model=StudyJobList)
    def list_study_jobs(
        request: Request, limit: int = Query(default=20, ge=1, le=100)
    ) -> StudyJobList:
        service: AnalysisService = request.app.state.service
        items = service.recent_studies(limit)
        return StudyJobList(items=items, count=len(items))

    @app.get(
        "/api/v1/study-jobs/{job_id}",
        response_model=StudyJobView,
        responses={404: {"model": ErrorEnvelope}},
    )
    def get_study_job(job_id: str, request: Request) -> StudyJobView | JSONResponse:
        service: AnalysisService = request.app.state.service
        job = service.get_study(job_id)
        if job is None:
            return _error("STUDY_NOT_FOUND", "No study job has this ID.", "lookup", 404)
        return job

    @app.delete(
        "/api/v1/study-jobs/{job_id}",
        response_model=StudyJobView,
        responses={404: {"model": ErrorEnvelope}},
    )
    def cancel_study_job(job_id: str, request: Request) -> StudyJobView | JSONResponse:
        service: AnalysisService = request.app.state.service
        job = service.cancel_study(job_id)
        if job is None:
            return _error("STUDY_NOT_FOUND", "No study job has this ID.", "lookup", 404)
        return job

    @app.post(
        "/api/v1/study-jobs/{job_id}/commands",
        response_model=StudyJobView,
        responses={404: {"model": ErrorEnvelope}, 409: {"model": ErrorEnvelope}},
    )
    def command_study_job(
        job_id: str, payload: StudyJobCommand, request: Request
    ) -> StudyJobView | JSONResponse:
        service: AnalysisService = request.app.state.service
        try:
            return service.resume_study(job_id, payload.expected_revision)
        except KeyError:
            return _error("STUDY_NOT_FOUND", "No study job has this ID.", "lookup", 404)
        except SessionConflict:
            return _error(
                "REVISION_CONFLICT",
                "The study changed; refresh it before resuming.",
                "study",
                409,
            )
        except ValueError:
            return _error(
                "INVALID_STUDY_COMMAND",
                "The requested study transition is not available.",
                "study",
                422,
            )

    @app.delete(
        "/api/v1/studies/{job_id}",
        response_model=DeleteResult,
        responses={404: {"model": ErrorEnvelope}},
    )
    def delete_study(job_id: str, request: Request) -> DeleteResult | JSONResponse:
        service: AnalysisService = request.app.state.service
        if not service.delete_study(job_id):
            return _error("STUDY_NOT_FOUND", "No study has this ID.", "lookup", 404)
        return DeleteResult()

    @app.get(
        "/api/v1/studies/{job_id}/moments/{moment_id}",
        response_model=LearningMomentView,
        responses={404: {"model": ErrorEnvelope}},
    )
    def get_learning_moment(
        job_id: str, moment_id: str, request: Request
    ) -> LearningMomentView | JSONResponse:
        service: AnalysisService = request.app.state.service
        job = service.get_study(job_id)
        if job is None:
            return _error("STUDY_NOT_FOUND", "No study has this ID.", "lookup", 404)
        moment = next((item for item in job.moments if item.moment_id == moment_id), None)
        if moment is None:
            return _error("MOMENT_NOT_FOUND", "No learning moment has this ID.", "lookup", 404)
        return moment

    @app.get(
        "/api/v1/studies/{job_id}/moments/{moment_id}/legal-moves",
        response_model=LegalMovesResult,
        responses={404: {"model": ErrorEnvelope}},
    )
    def learning_moment_legal_moves(
        job_id: str, moment_id: str, from_square: str, request: Request
    ) -> LegalMovesResult | JSONResponse:
        service: AnalysisService = request.app.state.service
        job = service.get_study(job_id)
        moment = (
            next((item for item in job.moments if item.moment_id == moment_id), None)
            if job
            else None
        )
        if moment is None:
            return _error("MOMENT_NOT_FOUND", "No learning moment has this ID.", "lookup", 404)
        try:
            return legal_moves_for_square(moment.fen, from_square)
        except ValueError:
            return _error("INVALID_POSITION", "The study position is invalid.", "study", 422)

    @app.post(
        "/api/v1/studies/{job_id}/moments/{moment_id}/attempts",
        response_model=PracticeAttemptView,
        responses={404: {"model": ErrorEnvelope}, 409: {"model": ErrorEnvelope}},
    )
    def create_practice_attempt(
        job_id: str,
        moment_id: str,
        payload: PracticeAttemptRequest,
        request: Request,
        idempotency_key: str | None = Header(
            default=None, alias="Idempotency-Key", min_length=8, max_length=128
        ),
    ) -> PracticeAttemptView | JSONResponse:
        service: AnalysisService = request.app.state.service
        try:
            return service.submit_practice_attempt(
                job_id, moment_id, payload, idempotency_key=idempotency_key
            )
        except KeyError:
            return _error("MOMENT_NOT_FOUND", "No learning moment has this ID.", "lookup", 404)
        except SessionConflict:
            return _error(
                "REVISION_CONFLICT",
                "The study changed; refresh it before submitting.",
                "study",
                409,
            )
        except ValueError:
            return _error("ILLEGAL_MOVE", "The submitted move is not legal.", "study", 422)
        except EngineUnavailable:
            return _error(
                "ENGINE_UNAVAILABLE", "Stockfish is unavailable.", "engine", 503, retryable=True
            )

    @app.get("/api/v1/reviews/due", response_model=ReviewCardList)
    def list_due_reviews(
        request: Request, limit: int = Query(default=50, ge=1, le=100)
    ) -> ReviewCardList:
        service: AnalysisService = request.app.state.service
        items = service.due_reviews(limit)
        return ReviewCardList(items=items, count=len(items))

    @app.get("/api/v1/progress", response_model=ProgressSummary)
    def get_progress(request: Request) -> ProgressSummary:
        service: AnalysisService = request.app.state.service
        return service.progress()

    @app.delete("/api/v1/progress", response_model=DeleteResult)
    def delete_progress(request: Request) -> DeleteResult:
        service: AnalysisService = request.app.state.service
        service.delete_progress()
        return DeleteResult()

    return app


def _error(
    code: str,
    message: str,
    stage: str,
    status: int,
    *,
    field: str | None = None,
    retryable: bool = False,
    remediation: tuple[str, ...] = (),
) -> JSONResponse:
    request_id = secrets.token_hex(16)
    return JSONResponse(
        status_code=status,
        content={
            "error": {
                "code": code,
                "message": message,
                "stage": stage,
                "retryable": retryable,
                "field": field,
                "remediation": list(remediation),
                "request_id": request_id,
            }
        },
        headers={"X-Request-ID": request_id},
    )


def _request_too_large() -> JSONResponse:
    return _error(
        "REQUEST_TOO_LARGE",
        f"Request bodies are limited to {MAX_REQUEST_BODY_BYTES} bytes.",
        "validation",
        413,
    )
