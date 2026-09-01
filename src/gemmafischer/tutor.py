from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import Literal

import chess

from .domain import (
    EngineEvidence,
    TutorFeedback,
    TutorFollowUp,
    TutorInteractionView,
    TutorOption,
    TutorQuestion,
    TutorStatus,
    now_utc,
)

CONCEPT_LABELS = {
    "check": "Give check",
    "capture": "Win or exchange material",
    "promotion": "Promote a pawn",
    "castling": "Secure the king by castling",
    "material_change": "Change the material balance",
    "opponent_check": "Prevent the opponent's check",
    "development": "Develop a piece",
}


@dataclass(frozen=True)
class TutorInteractionRecord:
    view: TutorInteractionView
    evidence: EngineEvidence
    answer_move_uci: str
    follow_up_answer_id: str


def serialize_record(record: TutorInteractionRecord) -> str:
    return json.dumps(
        {
            "view": record.view.model_dump(mode="json"),
            "evidence": record.evidence.model_dump(mode="json"),
            "answer_move_uci": record.answer_move_uci,
            "follow_up_answer_id": record.follow_up_answer_id,
        },
        separators=(",", ":"),
    )


def deserialize_record(payload: str) -> TutorInteractionRecord:
    value = json.loads(payload)
    return TutorInteractionRecord(
        view=TutorInteractionView.model_validate(value["view"]),
        evidence=EngineEvidence.model_validate(value["evidence"]),
        answer_move_uci=str(value["answer_move_uci"]),
        follow_up_answer_id=str(value["follow_up_answer_id"]),
    )


def create_interaction(
    interaction_id: str,
    session_id: str,
    source_analysis_id: str,
    evidence: EngineEvidence,
) -> TutorInteractionRecord:
    if not evidence.candidates:
        raise ValueError("A terminal analysis cannot create a move question")
    best = evidence.candidates[0]
    concepts = tuple(
        concept
        for concept in evidence.concepts
        if concept.candidate_id == best.evidence_id and bool(concept.value)
    )
    correct_id = concepts[0].concept if concepts else "calculate_forcing_moves"
    correct_label = CONCEPT_LABELS.get(correct_id, "Compare checks, captures, and threats")
    distractors = [
        TutorOption(option_id="improve_worst_piece", label="Improve the least active piece"),
        TutorOption(option_id="reduce_counterplay", label="Reduce the opponent's counterplay"),
    ]
    options = (TutorOption(option_id=correct_id, label=correct_label), *distractors)
    timestamp = now_utc()
    view = TutorInteractionView(
        interaction_id=interaction_id,
        session_id=session_id,
        revision=0,
        status=TutorStatus.AWAITING_ANSWER,
        question=TutorQuestion(
            prompt=f"Find the strongest move for {evidence.side_to_move}.",
            fen=evidence.fen,
            position_id=evidence.position_id,
            source_analysis_id=source_analysis_id,
        ),
        follow_up=TutorFollowUp(
            prompt="Which idea best explains the engine's preferred move?",
            options=options,
        ),
        created_at=timestamp,
        updated_at=timestamp,
    )
    return TutorInteractionRecord(
        view=view,
        evidence=evidence,
        answer_move_uci=best.move_uci,
        follow_up_answer_id=correct_id,
    )


def reveal_hint(record: TutorInteractionRecord) -> TutorInteractionRecord:
    if record.view.status is not TutorStatus.AWAITING_ANSWER:
        raise ValueError("A hint is only available before answering")
    best = record.evidence.candidates[0]
    concepts = tuple(
        concept
        for concept in record.evidence.concepts
        if concept.candidate_id == best.evidence_id and bool(concept.value)
    )
    hint = (
        f"Look for a move that can: {CONCEPT_LABELS[concepts[0].concept].lower()}."
        if concepts
        else "Compare forcing checks, captures, and threats before choosing."
    )
    evidence_ids = (concepts[0].evidence_id,) if concepts else (best.evidence_id,)
    return _update(record, hint=hint, hint_evidence_ids=evidence_ids)


def grade_answer(
    record: TutorInteractionRecord, submitted_move_uci: str, evidence: EngineEvidence
) -> TutorInteractionRecord:
    if record.view.status is not TutorStatus.AWAITING_ANSWER:
        raise ValueError("This interaction is not awaiting a move")
    board = chess.Board(record.view.question.fen)
    try:
        move = chess.Move.from_uci(submitted_move_uci)
    except ValueError as exc:
        raise ValueError("The answer must be a legal UCI move") from exc
    if move not in board.legal_moves:
        raise ValueError("The answer move is illegal in the practice position")
    comparison = evidence.move_comparison
    if comparison is None or comparison.considered_move_uci != submitted_move_uci:
        raise ValueError("The grading evidence does not match the submitted move")
    preferred = evidence.candidates[0]
    outcome: Literal["matched_engine", "equivalent", "engine_preferred"]
    if submitted_move_uci == preferred.move_uci:
        outcome = "matched_engine"
        message = "You found the engine's preferred move."
    elif comparison.outcome == "equal":
        outcome = "equivalent"
        message = "Your move is within the engine's equality tolerance."
    else:
        outcome = "engine_preferred"
        message = f"The matched-budget comparison prefers {preferred.move_san}."
    feedback = TutorFeedback(
        submitted_move_uci=submitted_move_uci,
        submitted_move_san=board.san(move),
        preferred_move_uci=preferred.move_uci,
        preferred_move_san=preferred.move_san,
        outcome=outcome,
        message=message,
        evidence_ids=(preferred.evidence_id, comparison.evidence_id),
    )
    updated = _update(record, status=TutorStatus.AWAITING_FOLLOW_UP, feedback=feedback)
    return replace(updated, evidence=evidence)


def answer_follow_up(record: TutorInteractionRecord, option_id: str) -> TutorInteractionRecord:
    if record.view.status is not TutorStatus.AWAITING_FOLLOW_UP or record.view.follow_up is None:
        raise ValueError("This interaction is not awaiting a follow-up")
    allowed = {option.option_id for option in record.view.follow_up.options}
    if option_id not in allowed:
        raise ValueError("Unknown follow-up option")
    follow_up = record.view.follow_up.model_copy(
        update={
            "selected_option_id": option_id,
            "correct": option_id == record.follow_up_answer_id,
        }
    )
    return _update(record, status=TutorStatus.COMPLETE, follow_up=follow_up)


def dismiss(record: TutorInteractionRecord) -> TutorInteractionRecord:
    if record.view.status in {TutorStatus.COMPLETE, TutorStatus.DISMISSED}:
        raise ValueError("This interaction is already terminal")
    return _update(record, status=TutorStatus.DISMISSED)


def _update(record: TutorInteractionRecord, **values: object) -> TutorInteractionRecord:
    view = record.view.model_copy(
        update={
            "revision": record.view.revision + 1,
            "updated_at": now_utc(),
            **values,
        }
    )
    return replace(record, view=view)
