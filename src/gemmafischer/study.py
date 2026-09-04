from __future__ import annotations

import hashlib
import io
import json
import uuid
from dataclasses import dataclass, field

import chess
import chess.pgn

from .domain import EngineEvidence, ErrorDetail, canonical_hash, now_utc
from .study_domain import (
    ImportedGame,
    LearningMomentPrivate,
    LearningMomentView,
    PGNImportRequest,
    StudyJobState,
    StudyJobView,
    StudyProgress,
)

SCREENING_NODE_BUDGET = 25_000
MINIMUM_LOSS_CP = 50
MAX_SHORTLIST = 6
MAX_MOMENTS = 3


@dataclass
class StudyWork:
    request: PGNImportRequest | None
    view: StudyJobView
    private_moments: list[LearningMomentPrivate] = field(default_factory=list)
    cancelled: bool = False
    operation_id: str | None = None


@dataclass(frozen=True)
class ScreeningCandidate:
    source_ply: int
    fen: str
    played_move_uci: str
    played_move_san: str
    severity_cp: int | None
    mate_loss: bool
    evidence: EngineEvidence

    @property
    def sort_key(self) -> tuple[int, int, int]:
        return (1 if self.mate_loss else 0, self.severity_cp or 0, -self.source_ply)


def parse_import(request: PGNImportRequest) -> ImportedGame:
    source = request.pgn.strip()
    handle = io.StringIO(source)
    game = chess.pgn.read_game(handle)
    if game is None:
        raise ValueError("The PGN does not contain a game")
    if game.errors:
        raise ValueError("The PGN mainline is malformed or illegal")
    if chess.pgn.read_game(handle) is not None:
        raise ValueError("Import exactly one game at a time")
    variant = str(game.headers.get("Variant", "Standard")).lower()
    if variant not in {"standard", "from position"}:
        raise ValueError("Only standard chess PGNs are supported")
    board = game.board()
    initial_fen = board.fen(en_passant="fen")
    moves_uci: list[str] = []
    moves_san: list[str] = []
    for move in game.mainline_moves():
        if move not in board.legal_moves:
            raise ValueError("The PGN contains an illegal mainline move")
        moves_san.append(board.san(move))
        moves_uci.append(move.uci())
        board.push(move)
        if len(moves_uci) > 400:
            raise ValueError("PGNs are limited to 400 plies")
    if not moves_uci:
        raise ValueError("The PGN has no mainline moves")
    white = _clean_header(game.headers.get("White"))
    black = _clean_header(game.headers.get("Black"))
    perspective = request.perspective
    if perspective == "auto":
        needle = (request.player_name or "").strip().casefold()
        white_match = bool(white and white.casefold() == needle)
        black_match = bool(black and black.casefold() == needle)
        if white_match == black_match:
            raise ValueError("The player name does not identify exactly one PGN side")
        perspective = "white" if white_match else "black"
    source_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
    return ImportedGame(
        game_id=canonical_hash({"schema_version": "1.0", "pgn_sha256": source_hash}),
        source_sha256=source_hash,
        initial_fen=initial_fen,
        moves_uci=tuple(moves_uci),
        moves_san=tuple(moves_san),
        perspective=perspective,
        rating_bucket=request.rating_bucket,
        white=white,
        black=black,
        date=_clean_header(game.headers.get("Date")),
        result=_clean_header(game.headers.get("Result")),
    )


def decision_positions(game: ImportedGame) -> tuple[tuple[int, str, str, str], ...]:
    board = chess.Board(game.initial_fen)
    wanted_turn = chess.WHITE if game.perspective == "white" else chess.BLACK
    result: list[tuple[int, str, str, str]] = []
    for ply, (move_uci, move_san) in enumerate(
        zip(game.moves_uci, game.moves_san, strict=True), 1
    ):
        if board.turn == wanted_turn:
            result.append((ply, board.fen(en_passant="fen"), move_uci, move_san))
        board.push_uci(move_uci)
    return tuple(result)


def screening_candidate(
    ply: int, fen: str, move_uci: str, move_san: str, evidence: EngineEvidence
) -> ScreeningCandidate | None:
    comparison = evidence.move_comparison
    if comparison is None or comparison.outcome != "engine_better":
        return None
    engine_mate = comparison.engine_mate_in
    considered_mate = comparison.considered_mate_in
    mate_loss = (
        engine_mate is not None
        and engine_mate > 0
        and (considered_mate is None or considered_mate <= 0)
    ) or (
        considered_mate is not None
        and considered_mate < 0
        and (engine_mate is None or engine_mate > 0)
    )
    severity: int | None = None
    if comparison.engine_score_cp is not None and comparison.considered_score_cp is not None:
        severity = max(0, comparison.engine_score_cp - comparison.considered_score_cp)
    if not mate_loss and (severity or 0) < MINIMUM_LOSS_CP:
        return None
    return ScreeningCandidate(ply, fen, move_uci, move_san, severity, mate_loss, evidence)


def select_shortlist(candidates: list[ScreeningCandidate]) -> tuple[ScreeningCandidate, ...]:
    return tuple(sorted(candidates, key=lambda item: item.sort_key, reverse=True)[:MAX_SHORTLIST])


def build_moment(
    candidate: ScreeningCandidate, evidence: EngineEvidence, rank: int
) -> LearningMomentPrivate:
    comparison = evidence.move_comparison
    if comparison is None or not evidence.candidates:
        raise ValueError("Deep evidence lacks a move comparison")
    best = evidence.candidates[0]
    concepts = tuple(
        dict.fromkeys(
            item.concept
            for item in evidence.concepts
            if item.candidate_id == best.evidence_id
        )
    )
    reasons = ["mate_loss" if candidate.mate_loss else "avoidable_loss"]
    if concepts:
        reasons.append("teachable_concept")
    moment_id = canonical_hash(
        {
            "schema_version": "1.0",
            "position_id": evidence.position_id,
            "played_move": candidate.played_move_uci,
            "engine_evidence": comparison.evidence_id,
        }
    )
    view = LearningMomentView(
        moment_id=moment_id,
        rank=rank,
        source_ply=candidate.source_ply,
        fen=candidate.fen,
        played_move_uci=candidate.played_move_uci,
        played_move_san=candidate.played_move_san,
        severity_cp=candidate.severity_cp,
        mate_loss=candidate.mate_loss,
        concept_keys=concepts,
        reason_codes=tuple(reasons),
    )
    return LearningMomentPrivate(
        view=view,
        preferred_move_uci=best.move_uci,
        preferred_move_san=best.move_san,
        evidence_json=evidence.model_dump_json(),
    )


def new_study_work(request: PGNImportRequest) -> StudyWork:
    timestamp = now_utc()
    job_id = uuid.uuid4().hex
    return StudyWork(
        request=request,
        view=StudyJobView(
            job_id=job_id,
            revision=0,
            state=StudyJobState.QUEUED,
            created_at=timestamp,
            updated_at=timestamp,
            progress=StudyProgress(completed_units=0, total_units=0),
        ),
    )


def failed_study(work: StudyWork, code: str, message: str, retryable: bool) -> StudyJobView:
    return work.view.model_copy(
        update={
            "revision": work.view.revision + 1,
            "state": StudyJobState.FAILED,
            "updated_at": now_utc(),
            "error": ErrorDetail(
                code=code,
                message=message,
                stage="study",
                retryable=retryable,
                remediation=(
                    ("Resume the study job.",)
                    if retryable
                    else ("Correct the PGN and submit it again.",)
                ),
                request_id=work.view.job_id,
            ),
        }
    )


def _clean_header(value: str | None) -> str | None:
    if not value or not value.replace("?", "").replace(".", "").strip():
        return None
    cleaned = " ".join(value.replace("<", "").replace(">", "").split())[:120]
    return cleaned or None


def evidence_ids(private: LearningMomentPrivate) -> tuple[str, ...]:
    evidence = json.loads(private.evidence_json)
    comparison = evidence.get("move_comparison") or {}
    value = comparison.get("evidence_id")
    return (str(value),) if value else ()
