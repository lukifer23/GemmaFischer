from __future__ import annotations

import secrets
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse

from . import __version__
from .domain import AnalysisRequest, BoardMoveRequest, EngineTurnRequest, LegalMovesRequest
from .engine import legal_moves_for_square
from .service import AnalysisService

STATIC_DIR = Path(__file__).with_name("static")
ALLOWED_HOSTS = {"127.0.0.1", "localhost", "testserver"}


def create_app(
    *,
    engine_path: str | None = None,
    full_profile: bool = False,
    node_budget: int = 250_000,
    capability_token: str | None = None,
) -> FastAPI:
    token = capability_token or secrets.token_urlsafe(32)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.service = AnalysisService(engine_path, node_budget, full_profile)
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
        response = await call_next(request)
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
        return FileResponse(STATIC_DIR / "app.css", media_type="text/css")

    @app.get("/app.js", include_in_schema=False)
    async def js() -> FileResponse:
        return FileResponse(STATIC_DIR / "app.js", media_type="text/javascript")

    @app.get("/api/v1/health")
    async def health(request: Request) -> dict[str, object]:
        return {
            "ok": True,
            "version": __version__,
            "engine_configured": request.app.state.service.engine_path is not None,
            "model_profile": "full" if full_profile else "deterministic",
        }

    @app.post("/api/v1/analyses", status_code=202)
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

    @app.get("/api/v1/analyses/{analysis_id}")
    async def get_analysis(analysis_id: str, request: Request) -> JSONResponse:
        snapshot = request.app.state.service.get(analysis_id)
        if snapshot is None:
            return _error("ANALYSIS_NOT_FOUND", "No analysis has this ID.", "lookup", 404)
        return JSONResponse(content=snapshot.model_dump(mode="json"))

    @app.delete("/api/v1/analyses/{analysis_id}")
    async def cancel_analysis(analysis_id: str, request: Request) -> JSONResponse:
        snapshot = request.app.state.service.cancel(analysis_id)
        if snapshot is None:
            return _error("ANALYSIS_NOT_FOUND", "No analysis has this ID.", "lookup", 404)
        return JSONResponse(content=snapshot.model_dump(mode="json"))

    @app.post("/api/v1/board/moves")
    def play_board_move(payload: BoardMoveRequest, request: Request) -> JSONResponse:
        try:
            result = request.app.state.service.play_move(payload)
        except ValueError as exc:
            return _error("ILLEGAL_MOVE", str(exc), "move_validation", 422, field="move_uci")
        except Exception as exc:
            return _error("ENGINE_FAILURE", str(exc), "engine", 503)
        return JSONResponse(content=result.model_dump(mode="json"))

    @app.post("/api/v1/board/legal-moves")
    def board_legal_moves(payload: LegalMovesRequest) -> JSONResponse:
        try:
            result = legal_moves_for_square(payload.fen, payload.from_square)
        except ValueError as exc:
            return _error("INVALID_POSITION", str(exc), "move_validation", 422, field="fen")
        return JSONResponse(content=result.model_dump(mode="json"))

    @app.post("/api/v1/board/engine-turn")
    def board_engine_turn(payload: EngineTurnRequest, request: Request) -> JSONResponse:
        try:
            result = request.app.state.service.play_engine_turn(payload)
        except ValueError as exc:
            return _error("INVALID_POSITION", str(exc), "move_validation", 422, field="fen")
        except Exception as exc:
            return _error("ENGINE_FAILURE", str(exc), "engine", 503)
        return JSONResponse(content=result.model_dump(mode="json"))

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
