from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from .domain import ErrorDetail, RatingBucket, StrictModel


class StudyJobState(StrEnum):
    QUEUED = "queued"
    PARSING = "parsing"
    SCREENING = "screening"
    DEEP_ANALYSIS = "deep_analysis"
    BUILDING_TRANSFER = "building_transfer"
    READY = "ready"
    PAUSED_INTERRUPTED = "paused_interrupted"
    PAUSED_STORAGE = "paused_storage"
    CANCELLED = "cancelled"
    FAILED = "failed"


class PracticePhase(StrEnum):
    ORIGINAL = "original"
    RETRY = "retry"
    TRANSFER = "transfer"
    DELAYED_REVIEW = "delayed_review"


class AttemptOutcome(StrEnum):
    CORRECT = "correct"
    EQUIVALENT = "equivalent"
    INCORRECT = "incorrect"


class PGNImportRequest(StrictModel):
    pgn: str = Field(min_length=1, max_length=256 * 1024)
    perspective: Literal["auto", "white", "black"] = "auto"
    player_name: str | None = Field(default=None, max_length=120)
    rating_bucket: RatingBucket = RatingBucket.CLUB

    @model_validator(mode="after")
    def auto_requires_player(self) -> PGNImportRequest:
        if self.perspective == "auto" and not (self.player_name or "").strip():
            raise ValueError("player_name is required when perspective is auto")
        return self


class ImportedGame(StrictModel):
    game_id: str
    source_sha256: str
    initial_fen: str
    moves_uci: tuple[str, ...] = Field(min_length=1, max_length=400)
    moves_san: tuple[str, ...] = Field(min_length=1, max_length=400)
    perspective: Literal["white", "black"]
    rating_bucket: RatingBucket
    white: str | None = None
    black: str | None = None
    date: str | None = None
    result: str | None = None

    @model_validator(mode="after")
    def move_ledgers_match(self) -> ImportedGame:
        if len(self.moves_uci) != len(self.moves_san):
            raise ValueError("UCI and SAN move ledgers must have equal length")
        return self


class StudyProgress(StrictModel):
    completed_units: int = Field(ge=0)
    total_units: int = Field(ge=0)
    current_ply: int | None = Field(default=None, ge=1)


class LearningMomentView(StrictModel):
    moment_id: str
    rank: int = Field(ge=1, le=3)
    source_ply: int = Field(ge=1)
    fen: str
    played_move_uci: str
    played_move_san: str
    severity_cp: int | None = Field(default=None, ge=0)
    mate_loss: bool = False
    concept_keys: tuple[str, ...] = ()
    reason_codes: tuple[str, ...]
    practice_status: Literal["new", "in_progress", "scheduled", "mastered"] = "new"


class LearningMomentPrivate(StrictModel):
    view: LearningMomentView
    preferred_move_uci: str
    preferred_move_san: str
    evidence_json: str
    transfer_fen: str | None = None
    transfer_move_uci: str | None = None
    transfer_move_san: str | None = None


class StudyJobView(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    job_id: str
    revision: int = Field(ge=0)
    state: StudyJobState
    created_at: datetime
    updated_at: datetime
    progress: StudyProgress
    game: ImportedGame | None = None
    moments: tuple[LearningMomentView, ...] = Field(default=(), max_length=3)
    error: ErrorDetail | None = None


class StudyJobAccepted(StrictModel):
    job_id: str
    state: StudyJobState
    revision: int


class StudyJobList(StrictModel):
    items: tuple[StudyJobView, ...]
    count: int = Field(ge=0)


class StudyJobCommand(StrictModel):
    expected_revision: int = Field(ge=0)
    action: Literal["resume"]


class PracticeAttemptRequest(StrictModel):
    expected_revision: int = Field(ge=0)
    phase: PracticePhase
    move_uci: str = Field(min_length=4, max_length=5)
    hint_used: bool = False


class PracticeFeedback(StrictModel):
    preferred_move_uci: str
    preferred_move_san: str
    message: str
    evidence_ids: tuple[str, ...]
    next_phase: PracticePhase | None = None
    next_fen: str | None = None


class PracticeAttemptView(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    attempt_id: str
    moment_id: str
    phase: PracticePhase
    attempt_number: int = Field(ge=1)
    submitted_move_uci: str
    outcome: AttemptOutcome
    hint_used: bool
    feedback: PracticeFeedback | None = None
    created_at: datetime


class ReviewCard(StrictModel):
    job_id: str
    moment_id: str
    moment: LearningMomentView
    concept_key: str
    due_at: datetime
    interval_days: int = Field(ge=1, le=30)
    successful_delayed_reviews: int = Field(ge=0)
    lapses: int = Field(ge=0)
    mastered: bool = False


class ReviewCardList(StrictModel):
    items: tuple[ReviewCard, ...]
    count: int = Field(ge=0)


class ProgressSummary(StrictModel):
    due: int = Field(ge=0)
    learning: int = Field(ge=0)
    retaining: int = Field(ge=0)
    mastered: int = Field(ge=0)
    attempts: int = Field(ge=0)
    original_accuracy: float = Field(ge=0, le=1)
    retry_accuracy: float = Field(ge=0, le=1)
    transfer_accuracy: float = Field(ge=0, le=1)
    delayed_accuracy: float = Field(ge=0, le=1)
