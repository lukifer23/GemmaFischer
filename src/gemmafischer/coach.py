from __future__ import annotations

from .domain import (
    CoachingClaim,
    CoachingResult,
    ComparisonClaim,
    EngineEvidence,
    GuidanceClaim,
    LineClaim,
    MoveClaim,
    RatingBucket,
    ScoreClaim,
)


def _score_text(evidence: EngineEvidence, candidate_id: str) -> str:
    candidate = next(item for item in evidence.candidates if item.evidence_id == candidate_id)
    side = "White" if evidence.side_to_move == "white" else "Black"
    if candidate.mate_in is not None:
        if candidate.mate_in > 0:
            return f"{side} can force mate in {candidate.mate_in}."
        return f"{side} is being mated in {abs(candidate.mate_in)}."
    assert candidate.score_cp is not None
    if candidate.score_cp == 0:
        return "The engine evaluates the position as equal."
    favored = side if candidate.score_cp > 0 else ("Black" if side == "White" else "White")
    return f"{favored} is better by {abs(candidate.score_cp) / 100:.2f} pawns."


def deterministic_coach(
    evidence: EngineEvidence,
    rating: RatingBucket,
    considered_move_uci: str | None,
) -> CoachingResult:
    if not evidence.candidates:
        reason = (evidence.terminal_reason or "game over").replace("_", " ")
        return CoachingResult(
            summary=f"This position is terminal: {reason}.",
            claims=(
                GuidanceClaim(template_id="calculate_forcing_moves"),
                GuidanceClaim(template_id="compare_candidate_moves"),
            ),
            source="deterministic",
        )

    best = evidence.candidates[0]
    claims: list[object] = [
        MoveClaim(evidence_ids=(best.evidence_id,), candidate_id=best.evidence_id),
        ScoreClaim(evidence_ids=(best.evidence_id,), candidate_id=best.evidence_id),
    ]
    if len(best.pv_uci) > 1:
        line_length = 4 if rating in {RatingBucket.BEGINNER, RatingBucket.DEVELOPING} else 8
        claims.append(
            LineClaim(
                evidence_ids=(best.evidence_id,),
                candidate_id=best.evidence_id,
                start_ply=0,
                end_ply=min(line_length, len(best.pv_uci)),
            )
        )

    compared = None
    if considered_move_uci:
        compared = next(
            (item for item in evidence.candidates if item.move_uci == considered_move_uci), None
        )
        if compared:
            claims.append(
                ComparisonClaim(
                    evidence_ids=(best.evidence_id, compared.evidence_id),
                    better_candidate_id=best.evidence_id,
                    considered_candidate_id=compared.evidence_id,
                )
            )
    claims.append(
        GuidanceClaim(
            template_id=(
                "compare_candidate_moves" if considered_move_uci else "calculate_forcing_moves"
            )
        )
    )
    summary = f"Start with {best.move_san}. {_score_text(evidence, best.evidence_id)}"
    return CoachingResult(summary=summary, claims=tuple(claims[:5]), source="deterministic")  # type: ignore[arg-type]


def validate_model_claims(
    evidence: EngineEvidence, claims: tuple[CoachingClaim, ...]
) -> tuple[tuple[CoachingClaim, ...], tuple[str, ...]]:
    evidence_ids = {
        *(item.evidence_id for item in evidence.candidates),
        *(item.evidence_id for item in evidence.board_facts),
    }
    candidate_ids = {item.evidence_id for item in evidence.candidates}
    valid: list[CoachingClaim] = []
    removed: list[str] = []
    for claim in claims:
        if any(item not in evidence_ids for item in claim.evidence_ids):
            removed.append("UNKNOWN_EVIDENCE_ID")
            continue
        referenced = [
            getattr(claim, key)
            for key in ("candidate_id", "better_candidate_id", "considered_candidate_id")
            if hasattr(claim, key)
        ]
        if any(item not in candidate_ids for item in referenced):
            removed.append("UNKNOWN_CANDIDATE_ID")
            continue
        if isinstance(claim, LineClaim):
            candidate = next(
                item for item in evidence.candidates if item.evidence_id == claim.candidate_id
            )
            if claim.end_ply > len(candidate.pv_uci):
                removed.append("PV_RANGE_OUT_OF_BOUNDS")
                continue
        valid.append(claim)
    return tuple(valid), tuple(removed)


def render_claim(evidence: EngineEvidence, claim: object) -> str:
    candidates = {item.evidence_id: item for item in evidence.candidates}
    if isinstance(claim, MoveClaim):
        item = candidates[claim.candidate_id]
        return f"The engine recommends {item.move_san} ({item.move_uci})."
    if isinstance(claim, ScoreClaim):
        return _score_text(evidence, claim.candidate_id)
    if isinstance(claim, LineClaim):
        item = candidates[claim.candidate_id]
        return "Principal variation: " + " ".join(item.pv_uci[claim.start_ply : claim.end_ply])
    if isinstance(claim, ComparisonClaim):
        best = candidates[claim.better_candidate_id]
        considered = candidates[claim.considered_candidate_id]
        if best.evidence_id == considered.evidence_id:
            return f"Your considered move {considered.move_san} matches the engine's first choice."
        if best.score_cp is not None and considered.score_cp is not None:
            delta = best.score_cp - considered.score_cp
            return (
                f"{best.move_san} evaluates {delta / 100:.2f} pawns better than "
                f"the considered move {considered.move_san}."
            )
        return f"The engine prefers {best.move_san} to {considered.move_san} by mate outcome."
    if isinstance(claim, GuidanceClaim):
        return {
            "calculate_forcing_moves": "Before choosing, calculate checks, captures, and threats.",
            "compare_candidate_moves": "Compare the forcing replies to both candidate moves.",
        }[claim.template_id]
    raise TypeError(f"Unsupported claim type: {type(claim)!r}")
