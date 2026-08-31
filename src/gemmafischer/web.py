from __future__ import annotations

import secrets
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse

from . import __version__
from .domain import (
    AnalysisAccepted,
    AnalysisList,
    AnalysisRequest,
    AnalysisSnapshot,
    BoardMoveRequest,
    CreateSessionRequest,
    DeleteResult,
    EngineTurnRequest,
    ErrorEnvelope,
    LegalMovesRequest,
    LegalMovesResult,
    RuntimeCapabilities,
    Session,
    SessionCommandRequest,
    SessionList,
)
from .engine import EngineUnavailable, legal_moves_for_square, resolve_stockfish
from .service import AnalysisService, SessionConflict

STATIC_DIR = Path(__file__).with_name("static")
ALLOWED_HOSTS = {"127.0.0.1", "localhost", "testserver"}
MAX_REQUEST_BODY_BYTES = 64 * 1024


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
        if request.method in {"POST", "PUT", "PATCH", "DELETE"}:
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
            str(first.get("msg", "The request is invalid.")),
            "validation",
            422,
            field=field,
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
        )

    @app.post(
        "/api/v1/analyses",
        status_code=202,
        response_model=AnalysisAccepted,
        responses={422: {"model": ErrorEnvelope}},
    )
    async def create_analysis(payload: AnalysisRequest, request: Request) -> JSONResponse:
        snapshot = request.app.state.service.submit(payload)
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
    async def list_analyses(
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
    async def get_analysis(analysis_id: str, request: Request) -> JSONResponse:
        snapshot = request.app.state.service.get(analysis_id)
        if snapshot is None:
            return _error("ANALYSIS_NOT_FOUND", "No analysis has this ID.", "lookup", 404)
        return JSONResponse(content=snapshot.model_dump(mode="json"))

    @app.delete(
        "/api/v1/analyses/{analysis_id}",
        response_model=AnalysisSnapshot,
        responses={404: {"model": ErrorEnvelope}},
    )
    async def cancel_analysis(analysis_id: str, request: Request) -> JSONResponse:
        snapshot = request.app.state.service.cancel(analysis_id)
        if snapshot is None:
            return _error("ANALYSIS_NOT_FOUND", "No analysis has this ID.", "lookup", 404)
        return JSONResponse(content=snapshot.model_dump(mode="json"))

    @app.post("/api/v1/board/moves", deprecated=True)
    def play_board_move(payload: BoardMoveRequest, request: Request) -> JSONResponse:
        try:
            result = request.app.state.service.play_move(payload)
        except ValueError as exc:
            return _error("ILLEGAL_MOVE", str(exc), "move_validation", 422, field="move_uci")
        except Exception:
            return _error(
                "ENGINE_FAILURE", "The chess engine could not complete the move.", "engine", 503
            )
        return JSONResponse(content=result.model_dump(mode="json"))

    @app.post("/api/v1/board/legal-moves", deprecated=True)
    def board_legal_moves(payload: LegalMovesRequest) -> JSONResponse:
        try:
            result = legal_moves_for_square(payload.fen, payload.from_square)
        except ValueError as exc:
            return _error("INVALID_POSITION", str(exc), "move_validation", 422, field="fen")
        return JSONResponse(content=result.model_dump(mode="json"))

    @app.post("/api/v1/board/engine-turn", deprecated=True)
    def board_engine_turn(payload: EngineTurnRequest, request: Request) -> JSONResponse:
        try:
            result = request.app.state.service.play_engine_turn(payload)
        except ValueError as exc:
            return _error("INVALID_POSITION", str(exc), "move_validation", 422, field="fen")
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
    async def create_session(payload: CreateSessionRequest, request: Request) -> Session:
        service: AnalysisService = request.app.state.service
        return service.create_session(payload)

    @app.get("/api/v1/sessions", response_model=SessionList)
    async def list_sessions(
        request: Request, limit: int = Query(default=20, ge=1, le=100)
    ) -> SessionList:
        sessions = request.app.state.service.recent_sessions(limit)
        return SessionList(items=sessions, count=len(sessions))

    @app.get(
        "/api/v1/sessions/{session_id}",
        response_model=Session,
        responses={404: {"model": ErrorEnvelope}},
    )
    async def get_session(session_id: str, request: Request) -> Session | JSONResponse:
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
    async def session_legal_moves(
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
        except ValueError as exc:
            return _error("INVALID_POSITION", str(exc), "move_validation", 422)
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
        except SessionConflict as exc:
            return _error("REVISION_CONFLICT", str(exc), "session", 409)
        except ValueError as exc:
            return _error("INVALID_SESSION_COMMAND", str(exc), "session", 422)
        except Exception:
            return _error(
                "ENGINE_FAILURE",
                "The chess engine could not complete the session command.",
                "engine",
                503,
            )

    @app.delete(
        "/api/v1/sessions/{session_id}",
        response_model=DeleteResult,
        responses={404: {"model": ErrorEnvelope}},
    )
    async def delete_session(session_id: str, request: Request) -> DeleteResult | JSONResponse:
        if not request.app.state.service.delete_session(session_id):
            return _error("SESSION_NOT_FOUND", "No session has this ID.", "lookup", 404)
        return DeleteResult()

    return app


def _error(
    code: str,
    message: str,
    stage: str,
    status: int,
    *,
    field: str | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={
            "error": {
                "code": code,
                "message": message,
                "stage": stage,
                "retryable": False,
                "field": field,
                "remediation": [],
                "request_id": "request-rejected",
            }
        },
    )


def _request_too_large() -> JSONResponse:
    return _error(
        "REQUEST_TOO_LARGE",
        f"Request bodies are limited to {MAX_REQUEST_BODY_BYTES} bytes.",
        "validation",
        413,
    )
