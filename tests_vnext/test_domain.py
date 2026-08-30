import pytest
from pydantic import ValidationError

from gemmafischer.domain import (
    WDL,
    AnalysisRequest,
    RatingBucket,
    Workflow,
    canonical_hash,
    normalize_fen,
)


def test_normalize_fen_preserves_six_fields() -> None:
    _, normalized = normalize_fen(
        "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"
    )
    assert len(normalized.split()) == 6
    assert normalized.endswith("2 3")


def test_invalid_position_is_rejected() -> None:
    with pytest.raises(ValueError, match="Invalid"):
        normalize_fen("8/8/8/8/8/8/8/8 w - - 0 1")


def test_compare_mode_requires_considered_move() -> None:
    with pytest.raises(ValidationError, match="considered_move_uci"):
        AnalysisRequest(
            mode=Workflow.COMPARE,
            fen="8/8/8/8/8/8/K6k/8 w - - 0 1",
            rating_bucket=RatingBucket.CLUB,
        )


def test_position_mode_rejects_considered_move() -> None:
    with pytest.raises(ValidationError, match="only accepted"):
        AnalysisRequest(
            mode=Workflow.POSITION,
            fen="8/8/8/8/8/8/K6k/8 w - - 0 1",
            considered_move_uci="a2a3",
        )


def test_wdl_must_total_one_thousand() -> None:
    with pytest.raises(ValidationError, match="total 1000"):
        WDL(win=300, draw=300, loss=300)


def test_canonical_hash_is_order_independent() -> None:
    assert canonical_hash({"a": 1, "b": 2}) == canonical_hash({"b": 2, "a": 1})

