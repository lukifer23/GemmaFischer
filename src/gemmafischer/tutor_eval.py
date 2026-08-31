from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import chess
from pydantic import Field, ValidationError, model_validator

from .coach import (
    deterministic_coach,
    merge_model_claims,
    order_lesson_plan,
    render_claim,
    validate_model_claims,
)
from .domain import (
    CoachingResult,
    ComparisonClaim,
    EngineEvidence,
    GuidanceClaim,
    LineClaim,
    MoveClaim,
    RatingBucket,
    ScoreClaim,
    StrictModel,
    normalize_fen,
)
from .engine import StockfishProvider
from .lmstudio import DEFAULT_LFM_MODEL, DEFAULT_LM_STUDIO_URL, LMStudioRuntime
from .runtime import DEFAULT_MODEL, DEFAULT_MODEL_REVISION, ClaimSelector, GemmaRuntime


class TutoringCase(StrictModel):
    """A frozen, independently asserted tutoring qualification case."""

    id: str = Field(min_length=1)
    source: str = Field(min_length=1)
    license: str = Field(min_length=1)
    fen: str = Field(min_length=1, max_length=256)
    rating_bucket: RatingBucket
    considered_move_uci: str | None = Field(default=None, min_length=4, max_length=5)
    acceptable_best_moves_uci: tuple[str, ...] = ()
    expected_terminal_reason: str | None = None
    acceptable_comparison_outcomes: tuple[
        Literal["equal", "engine_better", "considered_better"], ...
    ] = ()
    required_lesson_concepts: tuple[
        Literal[
            "check",
            "capture",
            "promotion",
            "castling",
            "material_change",
            "opponent_check",
            "development",
        ],
        ...,
    ] = ()
    adversarial_tags: tuple[str, ...] = ()

    @model_validator(mode="after")
    def coherent_expectations(self) -> TutoringCase:
        board = chess.Board(self.fen)
        if self.considered_move_uci is not None:
            move = chess.Move.from_uci(self.considered_move_uci)
            if move not in board.legal_moves:
                raise ValueError("considered_move_uci must be legal in the case position")
        if self.expected_terminal_reason and self.acceptable_best_moves_uci:
            raise ValueError("terminal cases cannot assert acceptable best moves")
        if self.acceptable_comparison_outcomes and self.considered_move_uci is None:
            raise ValueError("comparison outcomes require considered_move_uci")
        return self


class TutoringCaseResult(StrictModel):
    case_id: str
    repetition: int = Field(ge=1)
    profile: Literal["deterministic", "full"]
    passed: bool
    failure_codes: tuple[str, ...]
    position_id: str | None = None
    coaching_source: Literal["deterministic", "gemma", "lfm"] | None = None
    evidence_fingerprint: str | None = None
    coaching_fingerprint: str | None = None
    coaching_note_codes: tuple[str, ...] = ()
    error: str | None = None


_TEMPLATE_FOR_CONCEPT = {
    "check": "notice_check",
    "capture": "notice_capture",
    "promotion": "notice_promotion",
    "castling": "notice_castling",
    "material_change": "notice_material_change",
    "opponent_check": "notice_opponent_check",
    "development": "notice_development",
}


def load_tutoring_cases(path: Path) -> tuple[TutoringCase, ...]:
    cases: list[TutoringCase] = []
    seen: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            case = TutoringCase.model_validate_json(line)
        except (ValidationError, ValueError) as exc:
            raise ValueError(f"Invalid tutoring case at {path}:{line_number}: {exc}") from exc
        if case.id in seen:
            raise ValueError(f"Duplicate tutoring case id at {path}:{line_number}: {case.id}")
        seen.add(case.id)
        cases.append(case)
    if not cases:
        raise ValueError(f"Tutoring case file is empty: {path}")
    return tuple(cases)


def evaluate_tutoring_result(
    case: TutoringCase,
    evidence: EngineEvidence,
    coaching: CoachingResult,
    *,
    profile: Literal["deterministic", "full"] = "deterministic",
    repetition: int = 1,
) -> TutoringCaseResult:
    """Evaluate machine-checkable correctness; this does not rate human usefulness."""

    failures: list[str] = []
    _check_evidence(case, evidence, failures)
    _check_coaching(case, evidence, coaching, failures)
    if profile == "full" and evidence.candidates and coaching.source != "gemma":
        failures.append("MODEL_FALLBACK")
    evidence_fingerprint = _fingerprint(_stable_evidence(evidence))
    coaching_fingerprint = _fingerprint(coaching.model_dump(mode="json"))
    unique_failures = tuple(dict.fromkeys(failures))
    return TutoringCaseResult(
        case_id=case.id,
        repetition=repetition,
        profile=profile,
        passed=not unique_failures,
        failure_codes=unique_failures,
        position_id=evidence.position_id,
        coaching_source=coaching.source,
        evidence_fingerprint=evidence_fingerprint,
        coaching_fingerprint=coaching_fingerprint,
        coaching_note_codes=coaching.removed_claim_codes,
    )


def _check_evidence(
    case: TutoringCase, evidence: EngineEvidence, failures: list[str]
) -> None:
    try:
        EngineEvidence.model_validate(evidence.model_dump(mode="json"))
    except ValidationError:
        failures.append("EVIDENCE_SCHEMA_INVALID")
    _, normalized_case_fen = normalize_fen(case.fen)
    if evidence.fen != normalized_case_fen:
        failures.append("EVIDENCE_FEN_MISMATCH")
    if evidence.candidate_set and evidence.candidate_set.position_id != evidence.position_id:
        failures.append("CANDIDATE_SET_POSITION_MISMATCH")

    board = chess.Board(case.fen)
    candidate_ids = {candidate.evidence_id for candidate in evidence.candidates}
    for candidate in evidence.candidates:
        try:
            root = chess.Move.from_uci(candidate.move_uci)
        except ValueError:
            failures.append("CANDIDATE_UCI_INVALID")
            continue
        if root not in board.legal_moves:
            failures.append("CANDIDATE_MOVE_ILLEGAL")
        if not candidate.pv_uci or candidate.pv_uci[0] != candidate.move_uci:
            failures.append("PV_ROOT_MISMATCH")
        pv_board = board.copy()
        for move_uci in candidate.pv_uci:
            try:
                move = chess.Move.from_uci(move_uci)
            except ValueError:
                failures.append("PV_UCI_INVALID")
                break
            if move not in pv_board.legal_moves:
                failures.append("PV_MOVE_ILLEGAL")
                break
            pv_board.push(move)

    if case.expected_terminal_reason != evidence.terminal_reason:
        failures.append("TERMINAL_REASON_MISMATCH")
    if case.acceptable_best_moves_uci:
        if not evidence.candidates:
            failures.append("BEST_MOVE_MISSING")
        elif evidence.candidates[0].move_uci not in case.acceptable_best_moves_uci:
            failures.append("BEST_MOVE_UNEXPECTED")

    comparison = evidence.move_comparison
    if case.considered_move_uci is not None:
        if comparison is None:
            failures.append("COMPARISON_MISSING")
        else:
            if comparison.position_id != evidence.position_id:
                failures.append("COMPARISON_POSITION_MISMATCH")
            if comparison.considered_move_uci != case.considered_move_uci:
                failures.append("COMPARISON_MOVE_MISMATCH")
            if (
                evidence.candidates
                and comparison.engine_move_uci != evidence.candidates[0].move_uci
            ):
                failures.append("COMPARISON_ENGINE_MOVE_MISMATCH")
            if (
                case.acceptable_comparison_outcomes
                and comparison.outcome not in case.acceptable_comparison_outcomes
            ):
                failures.append("COMPARISON_OUTCOME_UNEXPECTED")
    elif comparison is not None:
        failures.append("UNEXPECTED_COMPARISON")

    concept_ids: set[str] = set()
    for concept in evidence.concepts:
        if concept.evidence_id in concept_ids:
            failures.append("CONCEPT_ID_DUPLICATE")
        concept_ids.add(concept.evidence_id)
        if concept.candidate_id not in candidate_ids:
            failures.append("CONCEPT_CANDIDATE_UNKNOWN")


def _check_coaching(
    case: TutoringCase,
    evidence: EngineEvidence,
    coaching: CoachingResult,
    failures: list[str],
) -> None:
    try:
        CoachingResult.model_validate(coaching.model_dump(mode="json"))
    except ValidationError:
        failures.append("COACHING_SCHEMA_INVALID")

    valid, removed = validate_model_claims(evidence, coaching.claims)
    if valid != coaching.claims or removed:
        failures.append("CLAIM_EVIDENCE_INVALID")
    candidate_ids = {item.evidence_id for item in evidence.candidates}
    comparison = evidence.move_comparison
    best_id = evidence.candidates[0].evidence_id if evidence.candidates else None
    move_claims = [claim for claim in coaching.claims if isinstance(claim, MoveClaim)]
    score_claims = [claim for claim in coaching.claims if isinstance(claim, ScoreClaim)]
    comparison_claims = [
        claim for claim in coaching.claims if isinstance(claim, ComparisonClaim)
    ]

    for claim in coaching.claims:
        if isinstance(claim, ComparisonClaim) and (
            comparison is None or claim.comparison_id != comparison.evidence_id
        ):
            failures.append("COMPARISON_CLAIM_INVALID")
        if isinstance(claim, LineClaim):
            maximum = (
                4
                if case.rating_bucket in {RatingBucket.BEGINNER, RatingBucket.DEVELOPING}
                else 8
            )
            if claim.end_ply - claim.start_ply > maximum:
                failures.append("RATING_LINE_TOO_LONG")

    if evidence.candidates:
        if not any(claim.candidate_id == best_id for claim in move_claims):
            failures.append("BEST_MOVE_CLAIM_MISSING")
        if not any(claim.candidate_id == best_id for claim in score_claims):
            failures.append("BEST_SCORE_CLAIM_MISSING")
        if not any(isinstance(claim, GuidanceClaim) for claim in coaching.claims):
            failures.append("GUIDANCE_CLAIM_MISSING")
        if best_id not in candidate_ids:
            failures.append("BEST_CANDIDATE_UNKNOWN")
    elif coaching.claims:
        failures.append("TERMINAL_CLAIMS_PRESENT")

    if case.considered_move_uci is not None and (
        comparison is None
        or not any(
            claim.comparison_id == comparison.evidence_id for claim in comparison_claims
        )
    ):
        failures.append("COMPARISON_CLAIM_MISSING")

    plan = coaching.lesson_plan
    if plan is None:
        failures.append("LESSON_PLAN_MISSING")
        return
    concepts = {item.evidence_id: item for item in evidence.concepts}
    lesson_concepts: set[str] = set()
    for step in plan.steps:
        concept = concepts.get(step.concept_id)
        if concept is None:
            failures.append("LESSON_CONCEPT_UNKNOWN")
            continue
        lesson_concepts.add(concept.concept)
        if best_id is not None and concept.candidate_id != best_id:
            failures.append("LESSON_CONCEPT_NOT_BEST_MOVE")
        if step.template_id != _TEMPLATE_FOR_CONCEPT[concept.concept]:
            failures.append("LESSON_TEMPLATE_MISMATCH")
    for required in case.required_lesson_concepts:
        if required not in lesson_concepts:
            failures.append("REQUIRED_LESSON_CONCEPT_MISSING")


def run_tutoring_qualification(
    case_path: Path,
    output_path: Path,
    *,
    profile: Literal["deterministic", "full"] = "deterministic",
    repetitions: int = 2,
    model_id: str = DEFAULT_MODEL,
    model_revision: str | None = DEFAULT_MODEL_REVISION,
    model_manifest_path: Path | None = None,
    model_backend: Literal["mlx", "lmstudio"] = "mlx",
    model_base_url: str = DEFAULT_LM_STUDIO_URL,
    model_artifact_path: Path | None = None,
) -> dict[str, Any]:
    """Run real Stockfish and optionally one verified model runtime over frozen cases."""

    if repetitions < 2:
        raise ValueError("repetitions must be at least 2 to measure stability")
    cases = load_tutoring_cases(case_path)
    provider = StockfishProvider()
    runtime: ClaimSelector | None = None
    if profile == "full":
        if model_backend == "lmstudio":
            if model_artifact_path is None:
                raise ValueError("model_artifact_path is required for LM Studio qualification")
            runtime = LMStudioRuntime(
                model_id or DEFAULT_LFM_MODEL,
                base_url=model_base_url,
                model_artifact=model_artifact_path,
            )
        else:
            runtime = GemmaRuntime(model_id, model_revision, model_manifest_path)
    results: list[TutoringCaseResult] = []
    human_rows: list[dict[str, Any]] = []
    unblinding_key: dict[str, str] = {}
    try:
        for repetition in range(1, repetitions + 1):
            for case in cases:
                try:
                    evidence = provider.analyze(case.fen, case.considered_move_uci)
                    baseline = deterministic_coach(
                        evidence, case.rating_bucket, case.considered_move_uci
                    )
                    coaching = baseline
                    if runtime is not None and evidence.candidates:
                        selection = runtime.select_claims(evidence, case.rating_bucket)
                        valid, removed = validate_model_claims(evidence, selection.claims)
                        if valid or selection.concept_ids:
                            coaching = CoachingResult(
                                summary=baseline.summary,
                                claims=merge_model_claims(valid, baseline.claims),
                                removed_claim_codes=(
                                    selection.removed_claim_codes
                                    + removed
                                    + ("MODEL_SELECTION_MERGED_WITH_REQUIRED_BASELINE",)
                                ),
                                source=runtime.source,
                                lesson_plan=order_lesson_plan(
                                    baseline.lesson_plan, selection.concept_ids
                                ),
                            )
                    if repetition == 1:
                        variants = [("deterministic", baseline)]
                        if coaching.source != "deterministic":
                            variants.append((coaching.source, coaching))
                        blinded: list[dict[str, Any]] = []
                        for source, variant in variants:
                            variant_id = _fingerprint(
                                {
                                    "case_id": case.id,
                                    "source": source,
                                    "case_sha256": _sha256_file(case_path),
                                }
                            )[:12]
                            unblinding_key[variant_id] = source
                            blinded.append(
                                {
                                    "variant_id": variant_id,
                                    "summary": variant.summary,
                                    "claims": [
                                        render_claim(evidence, claim)
                                        for claim in variant.claims
                                    ],
                                    "lesson_title": (
                                        variant.lesson_plan.title
                                        if variant.lesson_plan is not None
                                        else None
                                    ),
                                    "lesson_steps": (
                                        [step.text for step in variant.lesson_plan.steps]
                                        if variant.lesson_plan is not None
                                        else []
                                    ),
                                }
                            )
                        blinded.sort(key=lambda item: str(item["variant_id"]))
                        human_rows.append(
                            {
                                "case_id": case.id,
                                "fen": case.fen,
                                "rating_bucket": case.rating_bucket.value,
                                "considered_move_uci": case.considered_move_uci,
                                "variants": blinded,
                                "rubric_fields": {
                                    "factual_correctness_1_to_5": None,
                                    "relevance_1_to_5": None,
                                    "rating_appropriateness_1_to_5": None,
                                    "clarity_1_to_5": None,
                                    "actionability_1_to_5": None,
                                    "preferred_variant_id": None,
                                    "reviewer_notes": None,
                                },
                            }
                        )
                    results.append(
                        evaluate_tutoring_result(
                            case,
                            evidence,
                            coaching,
                            profile=profile,
                            repetition=repetition,
                        )
                    )
                except Exception as exc:
                    results.append(
                        TutoringCaseResult(
                            case_id=case.id,
                            repetition=repetition,
                            profile=profile,
                            passed=False,
                            failure_codes=("EXECUTION_ERROR",),
                            error=f"{type(exc).__name__}: {str(exc)[:500]}",
                        )
                    )
    finally:
        provider.close()

    by_case: dict[str, list[TutoringCaseResult]] = {}
    for result in results:
        by_case.setdefault(result.case_id, []).append(result)
    unstable_cases: list[str] = []
    for case_id, case_results in by_case.items():
        fingerprints = {
            (item.evidence_fingerprint, item.coaching_fingerprint)
            for item in case_results
            if item.error is None
        }
        if len(fingerprints) > 1:
            unstable_cases.append(case_id)
    raw = [item.model_dump(mode="json") for item in results]
    failure_counts: dict[str, int] = {}
    note_counts: dict[str, int] = {}
    for item in results:
        for code in item.failure_codes:
            failure_counts[code] = failure_counts.get(code, 0) + 1
        for code in item.coaching_note_codes:
            note_counts[code] = note_counts.get(code, 0) + 1
    if unstable_cases:
        failure_counts["OUTPUT_UNSTABLE"] = len(unstable_cases)
    passed = all(item.passed for item in results) and not unstable_cases
    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "status": "passed" if passed else "failed",
        "scope": "automated-grounding-and-contract-qualification",
        "human_usefulness_status": "human_open",
        "test_question_status": "not-assessed-no-question-output-contract",
        "generated_at": datetime.now(UTC).isoformat(),
        "commit": _git_revision(),
        "profile": profile,
        "model": (
            {
                "backend": model_backend,
                "model_id": model_id,
                "revision": model_revision if model_backend == "mlx" else None,
                "identity": (
                    runtime.identity.as_dict()
                    if isinstance(runtime, LMStudioRuntime)
                    else None
                ),
            }
            if profile == "full"
            else None
        ),
        "engine_sha256": provider.binary_sha256,
        "engine_node_budget": provider.node_budget,
        "case_path": str(case_path),
        "case_sha256": _sha256_file(case_path),
        "case_count": len(cases),
        "repetitions": repetitions,
        "execution_count": len(results),
        "passed_execution_count": sum(item.passed for item in results),
        "failed_execution_count": sum(not item.passed for item in results),
        "unstable_case_ids": unstable_cases,
        "failure_counts": failure_counts,
        "coaching_note_counts": note_counts,
        "results": raw,
        "human_review_packet_path": str(output_path.with_suffix(".human-review.json")),
        "human_review_unblinding_path": str(output_path.with_suffix(".unblinding.json")),
    }
    review_payload = {
        "schema_version": "1.0",
        "status": "human_open",
        "blinded": True,
        "instructions": (
            "Score each variant independently. Automated grounding results do not "
            "establish pedagogical usefulness."
        ),
        "rows": human_rows,
    }
    unblinding_payload = {
        "schema_version": "1.0",
        "warning": "Keep separate from reviewers until scoring is complete.",
        "variants": unblinding_key,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(output_path, payload)
    _write_json_atomic(output_path.with_suffix(".human-review.json"), review_payload)
    _write_json_atomic(output_path.with_suffix(".unblinding.json"), unblinding_payload)
    return payload


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _stable_evidence(evidence: EngineEvidence) -> dict[str, Any]:
    payload = evidence.model_dump(mode="json")
    engine = payload.get("engine")
    if isinstance(engine, dict):
        engine.pop("started_at", None)
    return payload


def _fingerprint(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision() -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return completed.stdout.strip() or None
