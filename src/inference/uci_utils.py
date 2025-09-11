"""UCI utilities for move extraction and legality validation.

Centralized helpers to keep parsing and validation consistent across
inference, UCI bridge, evaluation, and web layers.
"""

from __future__ import annotations

import re
from typing import Optional

import chess


# Canonical UCI move pattern: e2e4, a7a8q, etc.
UCI_MOVE_PATTERN = re.compile(r"\b([a-h][1-8][a-h][1-8][qrbn]?)\b", re.IGNORECASE)


def extract_first_uci(text: str) -> Optional[str]:
    """Extract the first UCI move-like token from text.

    Returns lowercased move if found; otherwise None.
    """
    if not text:
        return None
    match = UCI_MOVE_PATTERN.search(text)
    if not match:
        return None
    return match.group(1).lower()


def is_legal_uci(fen: str, move: str) -> bool:
    """Return True if move is a legal UCI move in the given FEN position."""
    try:
        board = chess.Board(fen)
        move_obj = chess.Move.from_uci(move)
        return move_obj in board.legal_moves
    except Exception:
        return False


def to_move_if_legal(board: chess.Board, move_str: str) -> Optional[chess.Move]:
    """Return chess.Move if the given UCI string is legal on board, else None."""
    try:
        move_obj = chess.Move.from_uci(move_str)
        return move_obj if move_obj in board.legal_moves else None
    except Exception:
        return None


def extract_fen(text: str) -> Optional[str]:
    """Extract a FEN string from text. Prefers lines starting with 'FEN:'.

    Falls back to a permissive FEN-like regex when no explicit prefix is found.
    """
    if not text:
        return None
    # Preferred: explicit FEN: <fen>
    for line in text.splitlines():
        line = line.strip()
        if line.lower().startswith("fen:"):
            candidate = line.split(":", 1)[1].strip()
            try:
                chess.Board(candidate)
                return candidate
            except Exception:
                continue
    # Fallback: permissive FEN regex (piece placement and fields)
    fen_like = re.compile(r"(?:[rnbqkpRNBQKP1-8]+/){7}[rnbqkpRNBQKP1-8]+\s[wb]\s(?:K?Q?k?q?|-)\s(?:[a-h][36]|-)\s\d+\s\d+")
    m = fen_like.search(text)
    if m:
        candidate = m.group(0)
        try:
            chess.Board(candidate)
            return candidate
        except Exception:
            return None
    return None


def build_engine_prompt(fen: str) -> str:
    """Construct minimal deterministic engine prompt for UCI move generation."""
    fen = fen.strip()
    return (
        f"FEN: {fen}\n"
        "Move:\n"
        "Style: balanced\n"
        "Mode: Engine\n"
        "Generate the best move in UCI format (e.g., e2e4). Respond with only the move."
    )


def extract_first_legal_move_uci(text: str, board: chess.Board) -> Optional[str]:
    """Extract the first legal UCI move token from text for the given position."""
    if not text:
        return None
    for m in UCI_MOVE_PATTERN.finditer(text.lower()):
        token = m.group(1)
        try:
            mv = chess.Move.from_uci(token)
            if mv in board.legal_moves:
                return token
        except Exception:
            continue
    return None


def extract_first_legal_move(text: str, board: chess.Board) -> Optional[chess.Move]:
    """Extract the first legal move as a chess.Move from text for the given position."""
    token = extract_first_legal_move_uci(text, board)
    if not token:
        return None
    try:
        mv = chess.Move.from_uci(token)
        return mv if mv in board.legal_moves else None
    except Exception:
        return None

"""Utilities for UCI prompt building and parsing.

Centralizes:
- Engine-mode prompt normalization
- UCI move extraction and legality validation
- FEN extraction from freeform text
"""

from __future__ import annotations

import re
from typing import List, Optional

import chess


UCI_MOVE_REGEX = r"\b([a-h][1-8][a-h][1-8][qrbn]?)\b"


def build_engine_prompt(fen: str) -> str:
    """Return the canonical engine-mode prompt.

    Keep it minimal to bias the model toward emitting only a UCI move.
    """
    return f"FEN: {fen}\nMove:"


def extract_fen(text: str) -> Optional[str]:
    """Extract the first FEN after a 'FEN:' prefix from text, if present."""
    m = re.search(r"FEN:\s*([^\n]+)", text, flags=re.IGNORECASE)
    return m.group(1).strip() if m else None


def parse_uci_candidates(text: str) -> List[str]:
    """Return all UCI substrings found in text (lowercased)."""
    return [m for m in re.findall(UCI_MOVE_REGEX, text.lower())]


def extract_first_legal_move(text: str, board: chess.Board) -> Optional[chess.Move]:
    """Extract and validate the first legal move present in text for the given board."""
    for token in parse_uci_candidates(text):
        try:
            mv = chess.Move.from_uci(token)
        except Exception:
            continue
        if mv in board.legal_moves:
            return mv
    return None


def extract_first_legal_move_uci(text: str, board: chess.Board) -> Optional[str]:
    """Extract first legal move as UCI string for the given board."""
    mv = extract_first_legal_move(text, board)
    return mv.uci() if mv else None


