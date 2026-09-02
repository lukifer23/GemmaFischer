from __future__ import annotations

from .domain import (
    CoachingClaim,
    CoachingResult,
    ComparisonClaim,
    EngineEvidence,
    GuidanceClaim,
    LessonPlan,
    LessonStep,
    LineClaim,
    MoveClaim,
    RatingBucket,
    ScoreClaim,
)


def _lesson_plan(evidence: EngineEvidence) -> LessonPlan:
    if not evidence.candidates:
        return LessonPlan(title="Game complete", steps=())
    best = evidence.candidates[0]
    templates = {
        "check": ("notice_check", f"Notice that {best.move_san} gives check."),
        "capture": ("notice_capture", f"Calculate the capture after {best.move_san}."),
        "promotion": ("notice_promotion", f"{best.move_san} promotes a pawn."),
        "castling": ("notice_castling", f"{best.move_san} castles the king."),
        "material_change": (
            "notice_material_change",
            f"Track the material change created by {best.move_san}.",
        ),
        "opponent_check": (
            "notice_opponent_check",
            "The principal variation includes a checking reply; calculate it first.",
        ),
        "development": (
            "notice_development",
            f"{best.move_san} develops a minor piece from its starting square.",
        ),
    }
    steps: list[LessonStep] = []
    for concept in evidence.concepts:
        if concept.candidate_id != best.evidence_id:
            continue
        template, text = templates[concept.concept]
        steps.append(
            LessonStep(
                concept_id=concept.evidence_id,
                template_id=template,  # type: ignore[arg-type]
                text=text,
            )
        )
    return LessonPlan(title=f"Why {best.move_san}?", steps=tuple(steps[:4]))


def order_lesson_plan(plan: LessonPlan | None, concept_ids: tuple[str, ...]) -> LessonPlan | None:
    if plan is None or not concept_ids:
        return plan
    order = {concept_id: index for index, concept_id in enumerate(concept_ids)}
    steps = tuple(
        sorted(plan.steps, key=lambda step: order.get(step.concept_id, len(order)))
    )
    return plan.model_copy(update={"steps": steps})


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
            claims=(),
            source="deterministic",
            lesson_plan=_lesson_plan(evidence),
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

    if (
        considered_move_uci
        and evidence.move_comparison
        and evidence.move_comparison.considered_move_uci == considered_move_uci
    ):
        comparison = evidence.move_comparison
        claims.append(
            ComparisonClaim(
                evidence_ids=(comparison.evidence_id,),
                comparison_id=comparison.evidence_id,
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
    return CoachingResult(
        summary=summary,
        claims=tuple(claims[:5]),  # type: ignore[arg-type]
        source="deterministic",
        lesson_plan=_lesson_plan(evidence),
    )


def validate_model_claims(
    evidence: EngineEvidence, claims: tuple[CoachingClaim, ...]
) -> tuple[tuple[CoachingClaim, ...], tuple[str, ...]]:
    evidence_ids = {
        *(item.evidence_id for item in evidence.candidates),
        *(item.evidence_id for item in evidence.board_facts),
        *((evidence.move_comparison.evidence_id,) if evidence.move_comparison else ()),
    }
    candidate_ids = {item.evidence_id for item in evidence.candidates}
    valid: list[CoachingClaim] = []
    removed: list[str] = []
    for claim in claims:
        if any(item not in evidence_ids for item in claim.evidence_ids):
            removed.append("UNKNOWN_EVIDENCE_ID")
            continue
        referenced = (
            [claim.candidate_id]
            if isinstance(claim, MoveClaim | LineClaim | ScoreClaim)
            else []
        )
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


def merge_model_claims(
    model_claims: tuple[CoachingClaim, ...], baseline_claims: tuple[CoachingClaim, ...]
) -> tuple[CoachingClaim, ...]:
    # The deterministic layer owns the required factual spine. Gemma may add a
    # bounded line or select concept ordering, but incomplete model output can
    # never remove the best move, score, guidance, or move comparison.
    required: list[CoachingClaim] = []
    for claim_type in (ComparisonClaim, MoveClaim, ScoreClaim, GuidanceClaim):
        required.extend(claim for claim in baseline_claims if isinstance(claim, claim_type))
    merged: list[CoachingClaim] = []
    seen: set[str] = set()
    for claim in (*required, *model_claims):
        identity = claim.model_dump_json()
        if identity in seen:
            continue
        seen.add(identity)
        merged.append(claim)
    return tuple(merged[:5])


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
        comparison = evidence.move_comparison
        if comparison is None or comparison.evidence_id != claim.comparison_id:
            raise ValueError("Comparison claim does not match the evidence bundle")
        if comparison.outcome == "equal":
            return (
                f"{comparison.considered_move_uci} is effectively equal to "
                f"{comparison.engine_move_uci} within {comparison.tolerance_cp} centipawns."
            )
        favored = (
            comparison.engine_move_uci
            if comparison.outcome == "engine_better"
            else comparison.considered_move_uci
        )
        return f"The matched-budget comparison favors {favored}."
    if isinstance(claim, GuidanceClaim):
        return {
            "calculate_forcing_moves": "Before choosing, calculate checks, captures, and threats.",
            "compare_candidate_moves": "Compare the forcing replies to both candidate moves.",
        }[claim.template_id]
    raise TypeError(f"Unsupported claim type: {type(claim)!r}")
