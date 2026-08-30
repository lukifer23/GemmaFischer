from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Any, Literal, cast

import chess
import rfc8785
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Workflow(StrEnum):
    POSITION = "position"
    COMPARE = "compare"


class RatingBucket(StrEnum):
    BEGINNER = "1000-1199"
    DEVELOPING = "1200-1399"
    CLUB = "1400-1599"
    ADVANCED = "1600-1800"


class GameDifficulty(StrEnum):
    CASUAL = "casual"
    CLUB = "club"
    STRONG = "strong"


class AnalysisState(StrEnum):
    QUEUED = "queued"
    VALIDATING = "validating"
    ENGINE_RUNNING = "engine_running"
    COMPARISON_RUNNING = "comparison_running"
    MODEL_RUNNING = "model_running"
    COMPLETE = "complete"
    ENGINE_ONLY = "engine_only"
    CANCELLED = "cancelled"
    FAILED = "failed"


TERMINAL_STATES = {
    AnalysisState.COMPLETE,
    AnalysisState.ENGINE_ONLY,
    AnalysisState.CANCELLED,
    AnalysisState.FAILED,
}


class AnalysisRequest(StrictModel):
    mode: Workflow
    fen: str = Field(min_length=1, max_length=256)
    rating_bucket: RatingBucket = RatingBucket.CLUB
    considered_move_uci: str | None = Field(default=None, min_length=4, max_length=5)

    @model_validator(mode="after")
    def validate_workflow(self) -> AnalysisRequest:
        if self.mode is Workflow.COMPARE and not self.considered_move_uci:
            raise ValueError("considered_move_uci is required in compare mode")
        if self.mode is Workflow.POSITION and self.considered_move_uci is not None:
            raise ValueError("considered_move_uci is only accepted in compare mode")
        return self


class BoardMoveRequest(StrictModel):
    fen: str = Field(min_length=1, max_length=256)
    move_uci: str = Field(min_length=4, max_length=5)
    engine_reply: bool = False
    difficulty: GameDifficulty = GameDifficulty.CLUB


class LegalMovesRequest(StrictModel):
    fen: str = Field(min_length=1, max_length=256)
    from_square: str = Field(pattern=r"^[a-h][1-8]$")


class LegalMovesResult(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    from_square: str
    moves_uci: tuple[str, ...]
    destinations: tuple[str, ...]


class BoardMoveResult(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    fen_before: str
    fen_after_human: str
    fen: str
    human_move_uci: str
    human_move_san: str
    engine_move_uci: str | None = None
    engine_move_san: str | None = None
    engine_name: str | None = None
    engine_nodes: int = 0
    game_over: bool
    outcome: str | None = None
    turn: Literal["white", "black"]


class EngineTurnRequest(StrictModel):
    fen: str = Field(min_length=1, max_length=256)
    difficulty: GameDifficulty = GameDifficulty.CLUB


class EngineTurnResult(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    fen_before: str
    fen: str
    move_uci: str
    move_san: str
    engine_name: str
    engine_nodes: int
    game_over: bool
    outcome: str | None = None
    turn: Literal["white", "black"]


class EngineMetadata(StrictModel):
    name: str
    author: str | None = None
    binary_sha256: str
    options: dict[str, int | str | bool | None]
    node_budget: int


class WDL(StrictModel):
    win: int = Field(ge=0, le=1000)
    draw: int = Field(ge=0, le=1000)
    loss: int = Field(ge=0, le=1000)

    @model_validator(mode="after")
    def totals_one_thousand(self) -> WDL:
        if self.win + self.draw + self.loss != 1000:
            raise ValueError("WDL must total 1000")
        return self


class CandidateEvidence(StrictModel):
    evidence_id: str
    rank: int = Field(ge=1, le=3)
    move_uci: str
    move_san: str
    score_cp: int | None = None
    mate_in: int | None = None
    score_perspective: Literal["side_to_move"] = "side_to_move"
    wdl_permille: WDL | None = None
    depth: int | None = None
    seldepth: int | None = None
    nodes: int
    pv_uci: tuple[str, ...] = Field(max_length=16)

    @model_validator(mode="after")
    def one_score(self) -> CandidateEvidence:
        if (self.score_cp is None) == (self.mate_in is None):
            raise ValueError("exactly one of score_cp or mate_in is required")
        return self


class BoardFact(StrictModel):
    evidence_id: str
    fact_type: Literal[
        "side_to_move",
        "in_check",
        "legal_move_count",
        "material_balance_cp",
        "castling_rights",
    ]
    value: bool | int | str


class EngineEvidence(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    position_id: str
    fen: str
    side_to_move: Literal["white", "black"]
    engine: EngineMetadata
    terminal_reason: str | None = None
    candidates: tuple[CandidateEvidence, ...] = Field(max_length=3)
    board_facts: tuple[BoardFact, ...]

    @model_validator(mode="after")
    def candidate_count_matches_terminal_state(self) -> EngineEvidence:
        if self.terminal_reason is None and not self.candidates:
            raise ValueError("non-terminal evidence requires at least one candidate")
        if self.terminal_reason is not None and self.candidates:
            raise ValueError("terminal evidence cannot contain candidates")
        return self


class MoveClaim(StrictModel):
    kind: Literal["move"] = "move"
    evidence_ids: tuple[str, ...]
    candidate_id: str


class LineClaim(StrictModel):
    kind: Literal["line"] = "line"
    evidence_ids: tuple[str, ...]
    candidate_id: str
    start_ply: int = Field(ge=0)
    end_ply: int = Field(gt=0, le=8)


class ScoreClaim(StrictModel):
    kind: Literal["score"] = "score"
    evidence_ids: tuple[str, ...]
    candidate_id: str


class ComparisonClaim(StrictModel):
    kind: Literal["comparison"] = "comparison"
    evidence_ids: tuple[str, ...]
    better_candidate_id: str
    considered_candidate_id: str


class GuidanceClaim(StrictModel):
    kind: Literal["guidance"] = "guidance"
    evidence_ids: tuple[str, ...] = ()
    template_id: Literal["calculate_forcing_moves", "compare_candidate_moves"]


CoachingClaim = Annotated[
    MoveClaim | LineClaim | ScoreClaim | ComparisonClaim | GuidanceClaim,
    Field(discriminator="kind"),
]


class CoachingResult(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    summary: str
    claims: tuple[CoachingClaim, ...] = Field(min_length=2, max_length=5)
    removed_claim_codes: tuple[str, ...] = ()
    source: Literal["deterministic", "gemma"]


class ErrorDetail(StrictModel):
    code: str
    message: str
    stage: str
    retryable: bool
    field: str | None = None
    remediation: tuple[str, ...] = ()
    request_id: str


class AnalysisSnapshot(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    analysis_id: str
    generation: int
    state: AnalysisState
    created_at: datetime
    updated_at: datetime
    request: AnalysisRequest
    evidence: EngineEvidence | None = None
    coaching: CoachingResult | None = None
    error: ErrorDetail | None = None


def canonical_hash(payload: object) -> str:
    return hashlib.sha256(rfc8785.dumps(cast(Any, payload))).hexdigest()


def normalize_fen(fen: str) -> tuple[chess.Board, str]:
    try:
        board = chess.Board(fen)
    except ValueError as exc:
        raise ValueError(f"Invalid FEN: {exc}") from exc
    if not board.is_valid():
        raise ValueError(f"Invalid position: status={board.status()}")
    return board, board.fen(en_passant="fen")


def now_utc() -> datetime:
    return datetime.now(UTC)
