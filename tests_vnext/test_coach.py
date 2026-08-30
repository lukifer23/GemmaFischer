from gemmafischer.coach import deterministic_coach, render_claim, validate_model_claims
from gemmafischer.domain import (
    WDL,
    BoardFact,
    CandidateEvidence,
    EngineEvidence,
    EngineMetadata,
    GuidanceClaim,
    LineClaim,
    RatingBucket,
)


def evidence() -> EngineEvidence:
    metadata = EngineMetadata(
        name="Stockfish 18",
        binary_sha256="a" * 64,
        options={"Threads": 1},
        node_budget=250_000,
    )
    candidates = (
        CandidateEvidence(
            evidence_id="best",
            rank=1,
            move_uci="f1b5",
            move_san="Bb5",
            score_cp=42,
            wdl_permille=WDL(win=320, draw=600, loss=80),
            nodes=250_000,
            pv_uci=("f1b5", "a7a6", "b5a4", "g8f6"),
        ),
        CandidateEvidence(
            evidence_id="considered",
            rank=2,
            move_uci="d2d4",
            move_san="d4",
            score_cp=10,
            nodes=250_000,
            pv_uci=("d2d4", "e5d4"),
        ),
    )
    return EngineEvidence(
        position_id="position",
        fen="r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
        side_to_move="white",
        engine=metadata,
        candidates=candidates,
        board_facts=(BoardFact(evidence_id="fact", fact_type="in_check", value=False),),
    )


def test_deterministic_position_coach_is_evidence_grounded() -> None:
    result = deterministic_coach(evidence(), RatingBucket.CLUB, None)
    assert result.source == "deterministic"
    assert 2 <= len(result.claims) <= 5
    assert "Bb5" in result.summary
    assert all(
        evidence_id in {"best", "considered", "fact"}
        for claim in result.claims
        for evidence_id in claim.evidence_ids
    )


def test_compare_workflow_mentions_considered_move() -> None:
    result = deterministic_coach(evidence(), RatingBucket.CLUB, "d2d4")
    rendered = [render_claim(evidence(), claim) for claim in result.claims]
    assert any("considered move d4" in line for line in rendered)


def test_compare_workflow_confirms_best_move_match() -> None:
    result = deterministic_coach(evidence(), RatingBucket.CLUB, "f1b5")
    rendered = [render_claim(evidence(), claim) for claim in result.claims]
    assert any("matches the engine's first choice" in line for line in rendered)


def test_model_claim_validator_drops_unknown_and_invalid_pv() -> None:
    claims = (
        GuidanceClaim(template_id="calculate_forcing_moves"),
        LineClaim(
            evidence_ids=("best",),
            candidate_id="best",
            start_ply=0,
            end_ply=8,
        ),
        LineClaim(
            evidence_ids=("missing",),
            candidate_id="missing",
            start_ply=0,
            end_ply=1,
        ),
    )
    valid, removed = validate_model_claims(evidence(), claims)
    assert valid == (claims[0],)
    assert removed == ("PV_RANGE_OUT_OF_BOUNDS", "UNKNOWN_EVIDENCE_ID")
