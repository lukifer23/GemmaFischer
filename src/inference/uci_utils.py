import re
from typing import Optional

import chess

UCI_PATTERN = re.compile(r"\b([a-h][1-8][a-h][1-8][qrbn]?)\b")

def extract_first_uci(text: str) -> Optional[str]:
    """Extract the first UCI move string from text."""
    matches = UCI_PATTERN.findall(text.lower())
    return matches[0] if matches else None

def is_legal_uci(fen: str, uci: str) -> bool:
    """Return True if the UCI move is legal in the position described by fen."""
    try:
        board = chess.Board(fen)
        move = chess.Move.from_uci(uci)
        return move in board.legal_moves
    except Exception:
        return False
