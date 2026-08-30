import pytest

from gemmafischer.domain import RatingBucket
from gemmafischer.engine import StockfishProvider
from gemmafischer.runtime import DEFAULT_MODEL_REVISION, GemmaRuntime

FEN = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"


@pytest.mark.hardware
@pytest.mark.model
def test_pinned_gemma_returns_valid_grounded_claims() -> None:
    with StockfishProvider(node_budget=10_000) as provider:
        evidence = provider.analyze(FEN)
    runtime = GemmaRuntime()
    claims = runtime.claims(evidence, RatingBucket.CLUB)
    assert runtime.revision == DEFAULT_MODEL_REVISION
    assert 2 <= len(claims) <= 5
    candidate_ids = {item.evidence_id for item in evidence.candidates}
    assert all(set(claim.evidence_ids) <= candidate_ids for claim in claims)
