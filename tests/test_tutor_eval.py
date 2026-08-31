from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from gemmafischer.coach import deterministic_coach
from gemmafischer.domain import (
    WDL,
    BoardFact,
    CandidateEvidence,
    CandidateSet,
    ConceptEvidence,
    EngineEvidence,
    EngineMetadata,
    LessonPlan,
    LessonStep,
    MoveComparisonEvidence,
    RatingBucket,
)
from gemmafischer.tutor_eval import (
    TutoringCase,
    evaluate_tutoring_result,
    load_tutoring_cases,
)


def _evidence() -> EngineEvidence:
    candidate = CandidateEvidence(
        evidence_id="best",
        rank=1,
        move_uci="f1b5",
        move_san="Bb5",
        score_cp=42,
        wdl_permille=WDL(win=320, draw=600, loss=80),
        nodes=250_000,
        pv_uci=("f1b5", "a7a6", "b5a4", "g8f6", "e1g1"),
    )
    return EngineEvidence(
        position_id="position",
        fen="r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
        side_to_move="white",
        engine=EngineMetadata(
            name="Stockfish 18",
            binary_sha256="a" * 64,
            options={"Threads": 1},
            node_budget=250_000,
        ),
        candidate_set=CandidateSet(
            evidence_id="set", position_id="position", candidates=(candidate,)
        ),
        move_comparison=MoveComparisonEvidence(
            evidence_id="comparison",
            position_id="position",
            engine_move_uci="f1b5",
            considered_move_uci="f1b5",
            engine_score_cp=42,
            considered_score_cp=42,
            outcome="equal",
            node_budget_each=250_000,
        ),
        board_facts=(BoardFact(evidence_id="fact", fact_type="in_check", value=False),),
        concepts=(
            ConceptEvidence(
                evidence_id="development",
                position_id="position",
                candidate_id="best",
                concept="development",
                value=True,
            ),
        ),
    )


def _case() -> TutoringCase:
    return TutoringCase(
        id="opening",
        source="constructed",
        license="CC0-1.0",
        fen="r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
        rating_bucket=RatingBucket.BEGINNER,
        considered_move_uci="f1b5",
        acceptable_best_moves_uci=("f1b5",),
        acceptable_comparison_outcomes=("equal",),
        required_lesson_concepts=("development",),
    )


def test_frozen_case_corpus_loads_with_provenance_and_adversarial_coverage() -> None:
    cases = load_tutoring_cases(Path("data/evaluation/tutoring_cases.jsonl"))
    assert len(cases) == 5
    assert all(case.source and case.license for case in cases)
    tags = {tag for case in cases for tag in case.adversarial_tags}
    assert {"terminal-position", "side-in-check", "beginner-line-limit"} <= tags
    assert {bucket for bucket in RatingBucket} == {case.rating_bucket for case in cases}


def test_grounded_deterministic_lesson_passes_all_contract_checks() -> None:
    evidence = _evidence()
    coaching = deterministic_coach(evidence, RatingBucket.BEGINNER, "f1b5")
    result = evaluate_tutoring_result(_case(), evidence, coaching)
    assert result.passed
    assert result.failure_codes == ()
    assert result.evidence_fingerprint
    assert result.coaching_fingerprint


def test_evaluator_reports_accuracy_grounding_and_relevance_failures() -> None:
    evidence = _evidence()
    coaching = deterministic_coach(evidence, RatingBucket.BEGINNER, "f1b5")
    wrong_case = _case().model_copy(
        update={
            "acceptable_best_moves_uci": ("d2d4",),
            "acceptable_comparison_outcomes": ("engine_better",),
            "required_lesson_concepts": ("check",),
        }
    )
    result = evaluate_tutoring_result(wrong_case, evidence, coaching)
    assert not result.passed
    assert {
        "BEST_MOVE_UNEXPECTED",
        "COMPARISON_OUTCOME_UNEXPECTED",
        "REQUIRED_LESSON_CONCEPT_MISSING",
    } <= set(result.failure_codes)


def test_evaluator_rejects_mismatched_lesson_template_and_rating_length() -> None:
    evidence = _evidence()
    coaching = deterministic_coach(evidence, RatingBucket.BEGINNER, "f1b5")
    assert coaching.lesson_plan is not None
    bad_plan = LessonPlan(
        title=coaching.lesson_plan.title,
        steps=(
            LessonStep(
                concept_id="development",
                template_id="notice_check",
                text="Incorrect template.",
            ),
        ),
    )
    long_line = next(claim for claim in coaching.claims if claim.kind == "line").model_copy(
        update={"end_ply": 5}
    )
    bad_coaching = coaching.model_copy(
        update={
            "claims": tuple(
                long_line if claim.kind == "line" else claim for claim in coaching.claims
            ),
            "lesson_plan": bad_plan,
        }
    )
    result = evaluate_tutoring_result(_case(), evidence, bad_coaching)
    assert {"RATING_LINE_TOO_LONG", "LESSON_TEMPLATE_MISMATCH"} <= set(
        result.failure_codes
    )


def test_fingerprints_are_stable_and_ignore_engine_start_time() -> None:
    evidence = _evidence()
    coaching = deterministic_coach(evidence, RatingBucket.BEGINNER, "f1b5")
    first = evaluate_tutoring_result(_case(), evidence, coaching)
    changed_metadata = evidence.engine.model_copy(
        update={"started_at": datetime(2026, 8, 30, 12, tzinfo=UTC)}
    )
    changed = evidence.model_copy(update={"engine": changed_metadata})
    second = evaluate_tutoring_result(_case(), changed, coaching, repetition=2)
    assert first.evidence_fingerprint == second.evidence_fingerprint
    assert first.coaching_fingerprint == second.coaching_fingerprint


def test_full_profile_does_not_pass_a_silent_deterministic_fallback() -> None:
    evidence = _evidence()
    coaching = deterministic_coach(evidence, RatingBucket.BEGINNER, "f1b5")
    result = evaluate_tutoring_result(_case(), evidence, coaching, profile="full")
    assert not result.passed
    assert "MODEL_FALLBACK" in result.failure_codes
