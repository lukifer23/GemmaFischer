"""UCI utilities for move extraction and legality validation.

Centralized helpers to keep parsing and validation consistent across
inference, UCI bridge, evaluation, and web layers.

Enhanced with strict UCI validation and post-processing for reliability.
"""

from __future__ import annotations

import re
import logging
from typing import Optional, List, Tuple
import chess
import chess.engine

# Configure logging
logger = logging.getLogger(__name__)

# Canonical UCI move pattern: e2e4, a7a8q, etc.
UCI_MOVE_PATTERN = re.compile(r"\b([a-h][1-8][a-h][1-8][qrbn]?)\b", re.IGNORECASE)

# Strict UCI validation pattern - must be exactly 4-5 characters
STRICT_UCI_PATTERN = re.compile(r"^([a-h][1-8][a-h][1-8][qrbn]?)$", re.IGNORECASE)


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


def validate_uci_syntax(move: str) -> bool:
    """Validate UCI move syntax strictly.
    
    Args:
        move: UCI move string to validate
        
    Returns:
        True if move matches strict UCI pattern, False otherwise
    """
    if not move or not isinstance(move, str):
        return False
    
    move = move.strip().lower()
    return bool(STRICT_UCI_PATTERN.match(move))


def extract_and_validate_uci(text: str, board: chess.Board) -> Optional[str]:
    """Extract first UCI move from text and validate both syntax and legality.
    
    Args:
        text: Text to extract UCI move from
        board: Chess board for legality validation
        
    Returns:
        Valid UCI move string or None if no valid move found
    """
    if not text or not board:
        return None
    
    # Extract first UCI-like token
    uci_candidate = extract_first_uci(text)
    if not uci_candidate:
        return None
    
    # Validate syntax
    if not validate_uci_syntax(uci_candidate):
        logger.warning(f"Invalid UCI syntax: {uci_candidate}")
        return None
    
    # Validate legality
    if not is_legal_uci(board.fen(), uci_candidate):
        logger.warning(f"Illegal UCI move: {uci_candidate} in position {board.fen()}")
        return None
    
    return uci_candidate


def post_process_uci_response(
    response: Optional[object],
    board: chess.Board,
    fallback_to_stockfish: bool = True,
) -> Optional[str]:
    """Validate LC0 move output and fall back to Stockfish only when necessary."""

    if board is None:
        return None

    candidate_move: Optional[str] = None
    raw_text: Optional[str] = None

    if hasattr(response, "best_move"):
        candidate_move = getattr(response, "best_move", None)
        raw_text = candidate_move or ""
    elif isinstance(response, dict):
        candidate_move = response.get("best_move")
        raw_text = response.get("response") or candidate_move or ""
    elif isinstance(response, str):
        raw_text = response
    else:
        raw_text = str(response) if response else ""

    if candidate_move:
        move = candidate_move.strip().lower()
        if validate_uci_syntax(move) and is_legal_uci(board.fen(), move):
            logger.info(f"Valid LC0 move returned: {move}")
            return move
        logger.warning(f"LC0 candidate move invalid: {move}")

    if raw_text:
        parsed_move = extract_and_validate_uci(raw_text, board)
        if parsed_move:
            logger.info(f"Valid UCI move extracted: {parsed_move}")
            return parsed_move

    if fallback_to_stockfish:
        try:
            stockfish_move = get_stockfish_best_move(board)
            if stockfish_move:
                logger.info(f"Using Stockfish fallback move: {stockfish_move}")
                return stockfish_move
        except Exception as e:
            logger.warning(f"Stockfish fallback failed: {e}")

    snippet = raw_text[:100] + "..." if raw_text and len(raw_text) > 100 else raw_text
    logger.warning(f"No valid UCI move found in response: {snippet}")
    return None


def get_stockfish_best_move(board: chess.Board, depth: int = 12) -> Optional[str]:
    """Get best move from Stockfish engine.
    
    Args:
        board: Chess board position
        depth: Search depth
        
    Returns:
        UCI move string or None
    """
    try:
        # Try to find Stockfish executable
        import subprocess
        import shutil
        
        stockfish_path = shutil.which("stockfish")
        if not stockfish_path:
            # Try common paths
            for path in ["/usr/local/bin/stockfish", "/opt/homebrew/bin/stockfish", 
                        "/usr/bin/stockfish", "stockfish"]:
                if shutil.which(path):
                    stockfish_path = path
                    break
        
        if not stockfish_path:
            logger.warning("Stockfish not found in PATH")
            return None
        
        # Use chess.engine for reliable Stockfish communication
        with chess.engine.SimpleEngine.popen_uci(stockfish_path) as engine:
            result = engine.play(board, chess.engine.Limit(depth=depth))
            if result.move:
                return result.move.uci()
    
    except Exception as e:
        logger.warning(f"Stockfish engine error: {e}")
    
    return None


def extract_all_uci_moves(text: str) -> List[str]:
    """Extract all UCI move-like tokens from text.
    
    Args:
        text: Text to search for UCI moves
        
    Returns:
        List of UCI move strings found
    """
    if not text:
        return []
    
    matches = UCI_MOVE_PATTERN.findall(text)
    return [match.lower() for match in matches]


def get_legal_moves_from_text(text: str, board: chess.Board) -> List[str]:
    """Extract all legal UCI moves from text for given position.
    
    Args:
        text: Text to search for UCI moves
        board: Chess board for legality validation
        
    Returns:
        List of legal UCI move strings
    """
    if not text or not board:
        return []
    
    all_moves = extract_all_uci_moves(text)
    legal_moves = []
    
    for move in all_moves:
        if is_legal_uci(board.fen(), move):
            legal_moves.append(move)
    
    return legal_moves


def create_engine_prompt_strict(fen: str) -> str:
    """Create strict engine prompt for UCI move generation.
    
    Args:
        fen: FEN position string
        
    Returns:
        Formatted prompt for UCI move generation
    """
    fen = fen.strip()
    return (
        f"FEN: {fen}\n"
        "Mode: Engine\n"
        "Generate the best move in UCI format (e.g., e2e4).\n"
        "Respond with only the UCI move, no other text."
    )


def create_tutor_prompt_with_uci(fen: str, question: str = "Analyze this position") -> str:
    """Create tutor prompt that requires UCI move at the end.
    
    Args:
        fen: FEN position string
        question: Analysis question
        
    Returns:
        Formatted prompt for tutor mode with UCI requirement
    """
    fen = fen.strip()
    return (
        f"FEN: {fen}\n"
        f"Question: {question}\n"
        "Mode: Tutor\n"
        "Analyze this position step by step and provide your reasoning.\n"
        "End your response with: Best move: <UCI_MOVE>"
    )
