import pytest

from gemmafischer.engine import StockfishProvider

FEN = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"


@pytest.mark.hardware
def test_stockfish_repeatability_at_release_budget() -> None:
    provider = StockfishProvider()
    first = provider.analyze(FEN)
    second = provider.analyze(FEN)
    assert [item.move_uci for item in first.candidates] == [
        item.move_uci for item in second.candidates
    ]
    for left, right in zip(first.candidates, second.candidates, strict=True):
        assert left.mate_in == right.mate_in
        if left.score_cp is not None and right.score_cp is not None:
            assert abs(left.score_cp - right.score_cp) <= 15
