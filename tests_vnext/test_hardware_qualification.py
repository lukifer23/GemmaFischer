import pytest

from gemmafischer.domain import GameDifficulty
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


@pytest.mark.hardware
def test_stockfish_plays_a_real_legal_reply() -> None:
    result = StockfishProvider(node_budget=1_000).play_move(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "e2e4",
        engine_reply=True,
        difficulty=GameDifficulty.CLUB,
    )
    assert result.human_move_san == "e4"
    assert result.engine_move_uci is not None
    assert result.engine_move_san is not None
    assert result.engine_name is not None and "Stockfish" in result.engine_name
    assert result.turn == "white"
