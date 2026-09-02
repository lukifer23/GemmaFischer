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


class SessionMode(StrEnum):
    PLAYER = "player"
    EXHIBITION = "exhibition"


class SessionStatus(StrEnum):
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETE = "complete"


class TutorStatus(StrEnum):
    AWAITING_ANSWER = "awaiting_answer"
    AWAITING_FOLLOW_UP = "awaiting_follow_up"
    COMPLETE = "complete"
    DISMISSED = "dismissed"


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
    started_at: datetime | None = None


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


class CandidateSet(StrictModel):
    evidence_id: str
    position_id: str
    candidates: tuple[CandidateEvidence, ...] = Field(min_length=1, max_length=3)

    @model_validator(mode="after")
    def ranks_are_ordered(self) -> CandidateSet:
        if tuple(item.rank for item in self.candidates) != tuple(
            range(1, len(self.candidates) + 1)
        ):
            raise ValueError("candidate ranks must be ordered and contiguous")
        if len({item.move_uci for item in self.candidates}) != len(self.candidates):
            raise ValueError("candidate moves must be unique")
        return self


class MoveComparisonEvidence(StrictModel):
    evidence_id: str
    position_id: str
    engine_move_uci: str
    considered_move_uci: str
    engine_score_cp: int | None = None
    engine_mate_in: int | None = None
    considered_score_cp: int | None = None
    considered_mate_in: int | None = None
    outcome: Literal["equal", "engine_better", "considered_better"]
    tolerance_cp: int = Field(default=15, ge=0)
    node_budget_each: int = Field(gt=0)


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


class ConceptEvidence(StrictModel):
    evidence_id: str
    position_id: str
    candidate_id: str
    concept: Literal[
        "check",
        "capture",
        "promotion",
        "castling",
        "material_change",
        "opponent_check",
        "development",
    ]
    value: bool | int


class EngineEvidence(StrictModel):
    schema_version: Literal["2.0"] = "2.0"
    position_id: str
    fen: str
    side_to_move: Literal["white", "black"]
    engine: EngineMetadata
    terminal_reason: str | None = None
    candidate_set: CandidateSet | None = None
    move_comparison: MoveComparisonEvidence | None = None
    board_facts: tuple[BoardFact, ...]
    concepts: tuple[ConceptEvidence, ...] = ()

    @model_validator(mode="before")
    @classmethod
    def migrate_schema_one(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        migrated = dict(value)
        migrated["schema_version"] = "2.0"
        candidates = migrated.pop("candidates", None)
        if candidates and not migrated.get("candidate_set"):
            position_id = str(migrated.get("position_id", "legacy-position"))
            candidate_ids = [
                str(
                    item.get("evidence_id")
                    if isinstance(item, dict)
                    else item.evidence_id
                )
                for item in candidates
            ]
            migrated["candidate_set"] = {
                "evidence_id": canonical_hash(
                    {
                        "schema_version": "legacy-1.0",
                        "position_id": position_id,
                        "candidate_ids": candidate_ids,
                    }
                ),
                "position_id": position_id,
                "candidates": candidates,
            }
        return migrated

    @property
    def candidates(self) -> tuple[CandidateEvidence, ...]:
        return self.candidate_set.candidates if self.candidate_set else ()

    @model_validator(mode="after")
    def candidate_count_matches_terminal_state(self) -> EngineEvidence:
        if self.terminal_reason is None and self.candidate_set is None:
            raise ValueError("non-terminal evidence requires at least one candidate")
        if self.terminal_reason is not None and self.candidate_set is not None:
            raise ValueError("terminal evidence cannot contain candidates")
        if self.move_comparison and self.candidate_set is None:
            raise ValueError("move comparison requires a candidate set")
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
    comparison_id: str


class GuidanceClaim(StrictModel):
    kind: Literal["guidance"] = "guidance"
    evidence_ids: tuple[str, ...] = ()
    template_id: Literal["calculate_forcing_moves", "compare_candidate_moves"]


CoachingClaim = Annotated[
    MoveClaim | LineClaim | ScoreClaim | ComparisonClaim | GuidanceClaim,
    Field(discriminator="kind"),
]


class LessonStep(StrictModel):
    concept_id: str
    template_id: Literal[
        "notice_check",
        "notice_capture",
        "notice_promotion",
        "notice_castling",
        "notice_material_change",
        "notice_opponent_check",
        "notice_development",
    ]
    text: str


class LessonPlan(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    title: str
    steps: tuple[LessonStep, ...] = Field(max_length=4)


class CoachingResult(StrictModel):
    schema_version: Literal["2.0"] = "2.0"
    summary: str
    claims: tuple[CoachingClaim, ...] = Field(max_length=5)
    removed_claim_codes: tuple[str, ...] = ()
    source: Literal["deterministic", "gemma", "lfm"]
    lesson_plan: LessonPlan | None = None
    question_template_id: Literal[
        "find-strongest-move", "explain-engine-choice", "compare-candidates"
    ] = "find-strongest-move"
    hint_template_id: str = "forcing-moves"

    @model_validator(mode="before")
    @classmethod
    def migrate_schema_one(cls, value: Any) -> Any:
        if isinstance(value, dict) and value.get("schema_version") == "1.0":
            return {**value, "schema_version": "2.0"}
        return value


class ErrorDetail(StrictModel):
    code: str
    message: str
    stage: str
    retryable: bool
    field: str | None = None
    remediation: tuple[str, ...] = ()
    request_id: str


class ErrorEnvelope(StrictModel):
    error: ErrorDetail


class AnalysisAccepted(StrictModel):
    analysis_id: str
    state: AnalysisState
    generation: int


class AnalysisList(StrictModel):
    items: tuple[AnalysisSnapshot, ...]
    count: int = Field(ge=0)


class AnalysisSnapshot(StrictModel):
    schema_version: Literal["2.0"] = "2.0"
    analysis_id: str
    generation: int
    state: AnalysisState
    created_at: datetime
    updated_at: datetime
    request: AnalysisRequest
    evidence: EngineEvidence | None = None
    coaching: CoachingResult | None = None
    error: ErrorDetail | None = None

    @model_validator(mode="before")
    @classmethod
    def migrate_schema_one(cls, value: Any) -> Any:
        if isinstance(value, dict) and value.get("schema_version") == "1.0":
            return {**value, "schema_version": "2.0"}
        return value


class Ply(StrictModel):
    ply: int = Field(ge=1)
    move_uci: str
    move_san: str
    fen_before: str
    fen_after: str
    actor: Literal["player", "engine_white", "engine_black"]
    analysis_id: str | None = None


class Session(StrictModel):
    schema_version: Literal["2.0"] = "2.0"
    session_id: str
    revision: int = Field(ge=0)
    mode: SessionMode
    status: SessionStatus
    initial_fen: str
    fen: str
    turn: Literal["white", "black"]
    player_color: Literal["white", "black"] | None = None
    white_difficulty: GameDifficulty = GameDifficulty.CLUB
    black_difficulty: GameDifficulty = GameDifficulty.CLUB
    rating_bucket: RatingBucket = RatingBucket.CLUB
    plies: tuple[Ply, ...] = ()
    created_at: datetime
    updated_at: datetime
    outcome: str | None = None


class CreateSessionRequest(StrictModel):
    mode: SessionMode = SessionMode.PLAYER
    fen: str = chess.STARTING_FEN
    player_color: Literal["white", "black"] | None = "white"
    white_difficulty: GameDifficulty = GameDifficulty.CLUB
    black_difficulty: GameDifficulty = GameDifficulty.CLUB
    rating_bucket: RatingBucket = RatingBucket.CLUB

    @model_validator(mode="after")
    def mode_matches_player(self) -> CreateSessionRequest:
        if self.mode is SessionMode.PLAYER and self.player_color is None:
            raise ValueError("player_color is required in player mode")
        if self.mode is SessionMode.EXHIBITION and self.player_color is not None:
            raise ValueError("player_color must be null in exhibition mode")
        return self


class SessionCommandRequest(StrictModel):
    expected_revision: int = Field(ge=0)
    action: Literal["player_move", "engine_move", "undo", "pause", "resume"]
    move_uci: str | None = Field(default=None, min_length=4, max_length=5)

    @model_validator(mode="after")
    def move_required(self) -> SessionCommandRequest:
        if self.action == "player_move" and self.move_uci is None:
            raise ValueError("move_uci is required for player_move")
        if self.action != "player_move" and self.move_uci is not None:
            raise ValueError("move_uci is only accepted for player_move")
        return self


class TutorOption(StrictModel):
    option_id: str
    label: str


class TutorQuestion(StrictModel):
    prompt: str
    fen: str
    position_id: str
    source_analysis_id: str
    hint_available: bool = True


class TutorFeedback(StrictModel):
    submitted_move_uci: str
    submitted_move_san: str
    preferred_move_uci: str
    preferred_move_san: str
    outcome: Literal["matched_engine", "equivalent", "engine_preferred"]
    message: str
    evidence_ids: tuple[str, ...]


class TutorFollowUp(StrictModel):
    prompt: str
    options: tuple[TutorOption, ...] = Field(min_length=2, max_length=4)
    selected_option_id: str | None = None
    correct: bool | None = None


class TutorInteractionView(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    interaction_id: str
    session_id: str
    revision: int = Field(ge=0)
    status: TutorStatus
    question: TutorQuestion
    hint: str | None = None
    hint_evidence_ids: tuple[str, ...] = ()
    feedback: TutorFeedback | None = None
    follow_up: TutorFollowUp | None = None
    created_at: datetime
    updated_at: datetime


class TutorInteractionList(StrictModel):
    items: tuple[TutorInteractionView, ...]
    count: int = Field(ge=0)


class CreateTutorRequest(StrictModel):
    source_analysis_id: str


class TutorCommandRequest(StrictModel):
    expected_revision: int = Field(ge=0)
    action: Literal["hint", "answer", "follow_up", "dismiss"]
    move_uci: str | None = Field(default=None, min_length=4, max_length=5)
    option_id: str | None = None

    @model_validator(mode="after")
    def payload_matches_action(self) -> TutorCommandRequest:
        if self.action == "answer" and self.move_uci is None:
            raise ValueError("move_uci is required for answer")
        if self.action != "answer" and self.move_uci is not None:
            raise ValueError("move_uci is only accepted for answer")
        if self.action == "follow_up" and self.option_id is None:
            raise ValueError("option_id is required for follow_up")
        if self.action != "follow_up" and self.option_id is not None:
            raise ValueError("option_id is only accepted for follow_up")
        return self


class RuntimeCapabilities(StrictModel):
    schema_version: Literal["2.0"] = "2.0"
    engine_status: Literal["ready", "missing", "failed"]
    model_status: Literal["disabled", "loading", "ready", "missing", "corrupt", "degraded"]
    history_enabled: bool
    storage_status: Literal["disabled", "ready", "degraded", "corrupt"]
    worker_status: Literal["ready", "paused_storage", "recovering", "failed"]
    evidence_schema: Literal["2.0"] = "2.0"


class StorageRecoveryResult(StrictModel):
    storage_status: Literal["disabled", "ready", "degraded", "corrupt"]
    worker_status: Literal["ready", "paused_storage", "recovering", "failed"]


class SessionList(StrictModel):
    items: tuple[Session, ...]
    count: int = Field(ge=0)


class DeleteResult(StrictModel):
    deleted: Literal[True] = True


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
