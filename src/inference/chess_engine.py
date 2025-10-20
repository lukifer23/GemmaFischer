#!/usr/bin/env python3
"""
Comprehensive Chess Engine Integration Module

Provides full integration with Stockfish chess engine for:
- Move validation and analysis
- Position evaluation
- Best move calculation
- Chess-specific metrics and feedback
- Dataset validation and enhancement
"""

import os
import shutil
import chess
import chess.engine
import chess.pgn
import chess.svg
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field, asdict
from pathlib import Path
import time
import logging
import re
import threading
import json
from datetime import datetime
from ..utils.common import get_config_manager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_STOCKFISH_OPTIONS = {
    'Threads': 2,
    'Hash': 128,
    'Skill Level': 20,
    'UCI_LimitStrength': False,
    'UCI_ShowWDL': True,
}

DEFAULT_STOCKFISH_PATHS = [
    "/opt/homebrew/bin/stockfish",
    "/usr/local/bin/stockfish",
    "/usr/bin/stockfish",
    "/usr/games/stockfish",
    "stockfish",
]


@dataclass
class MoveAnalysis:
    """Comprehensive move analysis result."""
    move: str
    is_legal: bool
    is_best: bool
    centipawn_score: Optional[int] = None
    mate_in: Optional[int] = None
    principal_variation: List[str] = field(default_factory=list)
    depth: int = 0
    time_taken: float = 0.0
    nodes_searched: int = 0
    engine_score: Optional[float] = None
    move_quality: str = "unknown"  # excellent, good, ok, poor, blunder
    explanation: str = ""


@dataclass
class PositionAnalysis:
    """Complete position analysis."""
    fen: str
    best_move: Optional[str] = None
    best_score: Optional[int] = None
    mate_in: Optional[int] = None
    evaluation: Dict[str, Any] = field(default_factory=dict)
    principal_variation: List[str] = field(default_factory=list)
    top_moves: List[MoveAnalysis] = field(default_factory=list)
    threats: List[str] = field(default_factory=list)
    opportunities: List[str] = field(default_factory=list)
    position_type: str = "middle_game"  # opening, middle_game, endgame


class ChessEngineManager:
    """High-level chess engine management with Stockfish integration."""

    def __init__(
        self,
        engine_path: str = "/opt/homebrew/bin/stockfish",
        engine_options: Optional[Dict[str, Any]] = None,
        debug: bool = False,
        name: str = "Stockfish",
        search_paths: Optional[List[str]] = None,
    ):
        """Initialize chess engine with comprehensive error handling."""
        self.engine_path = engine_path
        self.debug = debug
        self.engine = None
        self._engine_lock = threading.RLock()
        self.name = name

        # Engine configuration
        if engine_options is None:
            self.engine_options = DEFAULT_STOCKFISH_OPTIONS.copy()
        else:
            self.engine_options = dict(engine_options)

        self.search_paths = search_paths or DEFAULT_STOCKFISH_PATHS

        self._initialize_engine()

    def _initialize_engine(self) -> None:
        """Initialize Stockfish engine with retry logic."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Try provided path first; if it fails, try discovery
                try:
                    engine_instance = chess.engine.SimpleEngine.popen_uci(self.engine_path)
                except Exception:
                    discovered = self._discover_engine()
                    if not discovered:
                        raise
                    engine_instance = chess.engine.SimpleEngine.popen_uci(discovered)

                with self._engine_lock:
                    self.engine = engine_instance

                    # Configure engine with supported options only
                    supported_options = self.engine.options
                    valid_options = {}

                    for option_name, option_value in self.engine_options.items():
                        if option_name in supported_options:
                            valid_options[option_name] = option_value
                        else:
                            logger.warning(f"Option '{option_name}' not supported by engine, skipping")

                    if valid_options:
                        self.engine.configure(valid_options)
                        logger.info(f"[{self.name}] Configured engine with options: {list(valid_options.keys())}")

                    # Verify engine is responsive
                    try:
                        self.engine.ping()
                    except Exception:
                        # Fallback: issue a very quick analyse to ensure readiness
                        _ = self.engine.analyse(chess.Board(), chess.engine.Limit(depth=1, time=0.01))
                    logger.info(f"[{self.name}] UCI engine initialized successfully")

                    # Lightweight readiness probe (keep extremely short to avoid startup stalls)
                    board = chess.Board()
                    try:
                        info = self.engine.analyse(board, chess.engine.Limit(depth=2, time=0.05))
                        logger.info(f"[{self.name}] Engine test successful, score: {info.get('score')}")
                    except Exception:
                        # Non-fatal: engine responded to ping above; proceed
                        logger.info(f"[{self.name}] Engine quick test skipped due to error; continuing")

                return

            except Exception as e:
                logger.warning(f"[{self.name}] Engine initialization attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    raise RuntimeError(f"Failed to initialize {self.name} engine after {max_retries} attempts")
                time.sleep(1)

    def _discover_engine(self) -> Optional[str]:
        """Find engine binary in configured search paths."""
        for path in self.search_paths:
            if os.path.isabs(path) and os.path.exists(path):
                return path
            resolved = shutil.which(path)
            if resolved:
                return resolved
        return None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

    def cleanup(self) -> None:
        """Clean up engine resources."""
        with self._engine_lock:
            if self.engine:
                try:
                    self.engine.quit()
                    logger.info(f"[{self.name}] Engine cleaned up successfully")
                except Exception as e:
                    logger.warning(f"[{self.name}] Error during engine cleanup: {e}")
                finally:
                    self.engine = None

    def validate_move(self, fen: str, move: str) -> MoveAnalysis:
        """Validate a move and provide comprehensive analysis."""
        start_time = time.time()

        try:
            board = chess.Board(fen)
            move_obj = chess.Move.from_uci(move)

            if not board.is_legal(move_obj):
                return MoveAnalysis(
                    move=move,
                    is_legal=False,
                    is_best=False,
                    time_taken=time.time() - start_time,
                    explanation="Illegal move"
                )

            # Apply the move to analyze the resulting position
            board.push(move_obj)

            # Get engine analysis
            limit = chess.engine.Limit(depth=15, time=0.5)
            with self._engine_lock:
                engine = self.engine
                if engine is None:
                    raise RuntimeError("Chess engine is not initialized")
                info = engine.analyse(board, limit)

                # Get the best move for comparison
                best_move_result = engine.play(board, limit)
            best_move = best_move_result.move.uci() if best_move_result.move else None

            # Analyze move quality
            move_quality = self._assess_move_quality(board, move_obj, info, best_move_result.move)

            return MoveAnalysis(
                move=move,
                is_legal=True,
                is_best=(move_obj == best_move_result.move),
                centipawn_score=info['score'].white().score() if info['score'] else None,
                mate_in=info['score'].white().mate() if info['score'] and info['score'].is_mate() else None,
                principal_variation=[m.uci() for m in info.get('pv', [])],
                depth=info.get('depth', 0),
                time_taken=time.time() - start_time,
                nodes_searched=info.get('nodes', 0),
                engine_score=info['score'].white().score(mate_score=10000) / 100.0 if info['score'] else None,
                move_quality=move_quality,
                explanation=self._generate_move_explanation(board, move_obj, move_quality)
            )

        except Exception as e:
            logger.error(f"Error validating move {move}: {e}")
            return MoveAnalysis(
                move=move,
                is_legal=False,
                is_best=False,
                time_taken=time.time() - start_time,
                explanation=f"Error: {str(e)}"
            )

    def _assess_move_quality(self, board: chess.Board, move: chess.Move, info, best_move: chess.Move) -> str:
        """Assess the quality of a move based on engine analysis."""
        if not info.get('score'):
            return "unclear"

        score_diff = abs((info['score'].white().score() or 0) - (self._get_best_score(board, best_move) or 0))

        if move == best_move:
            return "excellent"
        elif score_diff < 50:  # Less than 0.5 pawns
            return "good"
        elif score_diff < 150:  # Less than 1.5 pawns
            return "ok"
        elif score_diff < 300:  # Less than 3 pawns
            return "poor"
        else:
            return "blunder"

    def _get_best_score(self, board: chess.Board, best_move: chess.Move) -> Optional[int]:
        """Get the score of the best move."""
        if not best_move:
            return None

        temp_board = board.copy()
        temp_board.push(best_move)

        limit = chess.engine.Limit(depth=10, time=0.2)
        with self._engine_lock:
            engine = self.engine
            if engine is None:
                raise RuntimeError("Chess engine is not initialized")
            info = engine.analyse(temp_board, limit)
        return info['score'].white().score() if info.get('score') else None

    def _generate_move_explanation(self, board: chess.Board, move: chess.Move, quality: str) -> str:
        """Generate a human-readable explanation for the move."""
        explanations = {
            "excellent": "This is the best move according to the engine analysis.",
            "good": "This is a solid move with minimal drawbacks.",
            "ok": "This move is acceptable but not optimal.",
            "poor": "This move has significant disadvantages.",
            "blunder": "This move loses significant material or position.",
            "unclear": "The position is complex and the move quality cannot be clearly determined."
        }

        base_explanation = explanations.get(quality, "Move quality assessment not available.")

        # Add tactical insights
        if board.is_capture(move):
            base_explanation += " This move captures an enemy piece."
        if board.is_check():
            base_explanation += " This move puts the opponent in check."
        if board.is_checkmate():
            base_explanation += " This move delivers checkmate!"

        return base_explanation

    def analyze_position(self, fen: str, depth: int = 15, time_limit: float = 1.0) -> PositionAnalysis:
        """Provide comprehensive position analysis."""
        start_time = time.time()

        try:
            board = chess.Board(fen)

            with self._engine_lock:
                engine = self.engine
                if engine is None:
                    raise RuntimeError("Chess engine is not initialized")

                # Allocate time budget to avoid starving best-move search
                if self.name == "LC0":
                    # For LC0, get the move first, then a very quick eval if desired
                    play_limit = chess.engine.Limit(time=max(0.1, float(time_limit or 1.0)))
                    best_move_result = engine.play(board, play_limit)
                    best_move = best_move_result.move.uci() if (best_move_result and best_move_result.move) else None

                    # Quick, lightweight eval (do not block)
                    try:
                        info = engine.analyse(board, chess.engine.Limit(time=0.05))
                    except Exception:
                        info = {}
                else:
                    # For Stockfish, balanced analyse + play using shared budget
                    limit = chess.engine.Limit(depth=depth, time=time_limit)
                    info = engine.analyse(board, limit)
                    best_move_result = engine.play(board, limit)
                    best_move = best_move_result.move.uci() if (best_move_result and best_move_result.move) else None

            principal_variation = []
            if isinstance(info, dict):
                pv_moves = info.get('pv') or []
                principal_variation = [m.uci() for m in pv_moves if hasattr(m, "uci")]

            # Get top moves only for Stockfish; skip for LC0 to reduce latency
            if self.name == "LC0":
                top_moves = []
            else:
                top_moves = self._get_top_moves(board, chess.engine.Limit(depth=10, time=0.3), count=5)

            # Analyze position characteristics
            position_type = self._classify_position(board)
            threats = self._identify_threats(board)
            opportunities = self._identify_opportunities(board)

            return PositionAnalysis(
                fen=fen,
                best_move=best_move,
                best_score=info['score'].white().score() if info.get('score') else None,
                mate_in=info['score'].white().mate() if info.get('score') and info['score'].is_mate() else None,
                evaluation={
                    'depth': info.get('depth', 0),
                    'nodes': info.get('nodes', 0),
                    'time': info.get('time', 0),
                    'score_type': 'mate' if info.get('score') and info['score'].is_mate() else 'centipawns'
                },
                principal_variation=principal_variation,
                top_moves=top_moves,
                threats=threats,
                opportunities=opportunities,
                position_type=position_type
            )

        except Exception as e:
            logger.error(f"Error analyzing position {fen}: {e}")
            return PositionAnalysis(fen=fen)

    def _get_top_moves(self, board: chess.Board, limit: chess.engine.Limit, count: int = 5) -> List[MoveAnalysis]:
        """Get top N moves with analysis."""
        moves = []

        for move in board.legal_moves:
            temp_board = board.copy()
            temp_board.push(move)

            with self._engine_lock:
                engine = self.engine
                if engine is None:
                    raise RuntimeError("Chess engine is not initialized")
                info = engine.analyse(temp_board, chess.engine.Limit(depth=10, time=0.3))

            moves.append(MoveAnalysis(
                move=move.uci(),
                is_legal=True,
                is_best=False,  # Will be set later
                centipawn_score=info['score'].white().score() if info.get('score') else None,
                depth=info.get('depth', 0),
                time_taken=info.get('time', 0)
            ))

        # Sort by score (descending) and take top N
        moves.sort(key=lambda x: x.centipawn_score or -9999, reverse=True)
        top_moves = moves[:count]

        # Mark the best move
        if top_moves:
            top_moves[0].is_best = True

        return top_moves

    def _classify_position(self, board: chess.Board) -> str:
        """Classify the position type (opening, middle game, endgame)."""
        total_pieces = chess.popcount(board.occupied)

        if total_pieces > 28:  # Most pieces still on board
            return "opening"
        elif total_pieces > 12:  # Significant pieces remain
            return "middle_game"
        else:
            return "endgame"

    def _identify_threats(self, board: chess.Board) -> List[str]:
        """Identify tactical threats in the position."""
        threats = []

        # Check for basic threats
        if board.is_check():
            threats.append("King is in check")

        # Look for hanging pieces
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color == board.turn:
                if board.is_attacked_by(not board.turn, square):
                    threats.append(f"{piece.symbol().upper()} on {chess.square_name(square)} is under attack")

        return threats

    def _identify_opportunities(self, board: chess.Board) -> List[str]:
        """Identify tactical opportunities."""
        opportunities = []

        # Check for captures
        for move in board.legal_moves:
            if board.is_capture(move):
                captured_piece = board.piece_at(move.to_square)
                if captured_piece:
                    opportunities.append(f"Can capture {captured_piece.symbol().upper()} on {chess.square_name(move.to_square)}")

        # Check for checks
        for move in board.legal_moves:
            temp_board = board.copy()
            temp_board.push(move)
            if temp_board.is_check():
                opportunities.append(f"Move {move.uci()} puts opponent in check")

        return opportunities

    def get_best_move(self, board: chess.Board, depth: int = 12, time_limit_ms: int = 5000) -> Optional[chess.Move]:
        """Return the engine's best move for the given board.

        Uses Stockfish with the provided depth and time limit (milliseconds).
        """
        try:
            limit = chess.engine.Limit(depth=depth, time=max(0.0, float(time_limit_ms) / 1000.0))
            with self._engine_lock:
                engine = self.engine
                if engine is None:
                    raise RuntimeError("Chess engine is not initialized")
                result = engine.play(board, limit)
            return result.move if result and result.move else None
        except Exception as e:
            logger.error(f"Error getting best move from engine: {e}")
            return None

    def get_top_moves(self, board: chess.Board, depth: int = 8, top_k: int = 3) -> List[chess.Move]:
        """Return up to top_k moves ranked by Stockfish evaluation."""
        try:
            limit = chess.engine.Limit(depth=depth)
            with self._engine_lock:
                engine = self.engine
                if engine is None:
                    raise RuntimeError("Chess engine is not initialized")
                info = engine.analyse(board, limit, multipv=top_k)

            moves: List[chess.Move] = []
            if isinstance(info, list):
                for entry in info:
                    pv = entry.get('pv')
                    if pv:
                        moves.append(pv[0])
            elif isinstance(info, dict):
                pv = info.get('pv')
                if pv:
                    moves.append(pv[0])

            return moves
        except Exception as e:
            logger.error(f"Error getting top moves from engine: {e}")
            return []

    def get_top_moves_info(
        self,
        board: chess.Board,
        depth: int = 8,
        top_k: int = 3,
        time_limit_ms: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Return detailed information about the top Stockfish moves.

        Includes UCI move, centipawn score (relative to side to move), mate score, and principal variation.
        """
        try:
            limit_kwargs: Dict[str, Any] = {}
            if depth is not None:
                limit_kwargs["depth"] = depth
            if time_limit_ms is not None:
                limit_kwargs["time"] = max(0.0, float(time_limit_ms) / 1000.0)
            if not limit_kwargs:
                # Fallback to a lightweight depth restriction
                limit_kwargs["depth"] = 6

            limit = chess.engine.Limit(**limit_kwargs)
            with self._engine_lock:
                engine = self.engine
                if engine is None:
                    raise RuntimeError("Chess engine is not initialized")
                info = engine.analyse(board, limit, multipv=max(1, top_k))

            info_list = info if isinstance(info, list) else [info]
            entries: List[Dict[str, Any]] = []
            for entry in info_list:
                pv = entry.get("pv") or []
                move_obj = pv[0] if pv else entry.get("move")
                if move_obj is None:
                    continue
                move_str = move_obj.uci() if isinstance(move_obj, chess.Move) else str(move_obj)

                cp_score: Optional[int] = None
                mate_score: Optional[int] = None
                raw_score = entry.get("score")
                if raw_score is not None:
                    try:
                        pov = raw_score.pov(board.turn)
                        mate_score = pov.mate()
                        if mate_score is None:
                            cp_score = pov.score(mate_score=100000)
                    except Exception:
                        cp_score = None
                        mate_score = None

                entries.append({
                    "move": move_str,
                    "score_cp": int(cp_score) if cp_score is not None else None,
                    "mate": mate_score,
                    "depth": entry.get("depth"),
                    "seldepth": entry.get("seldepth"),
                    "nodes": entry.get("nodes"),
                    "nps": entry.get("nps"),
                    "multipv": entry.get("multipv"),
                    "pv": [mv.uci() for mv in pv if isinstance(mv, chess.Move)],
                })

            # Sort by multipv index to preserve engine ordering
            entries.sort(key=lambda item: item.get("multipv") or 0)
            return entries[:top_k]
        except Exception as e:
            logger.error(f"Error getting detailed top moves from engine: {e}")
            return []

    def validate_dataset_entry(self, question: str, answer: str) -> Dict[str, Any]:
        """Validate a dataset entry using chess engine analysis."""
        validation_result = {
            'question': question,
            'answer': answer,
            'moves_found': [],
            'validation_score': 0.0,
            'issues': [],
            'recommendations': []
        }

        # Extract FEN positions from question
        fen_pattern = r'([rnbqkbnrpppp/pppp/8/8/8/8/PPPP/RNBQKBNR w KQkq - 0 1]|[rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1]|[1rnbqkbnr/1ppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1]|[rnbqkbnr/1ppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1]|(?:[rnbqkpRNBQKP1-8]+/){7}[rnbqkpRNBQKP1-8]+ [wb] [KQkq-]+ [a-h1-8-]+ \d+ \d+)'
        fens = re.findall(fen_pattern, question)

        # Extract moves from answer
        move_pattern = r'\b([KQRBN]?[a-h]?[1-8]?x?[a-h][1-8][+#]?|O-O(?:-O)?)\b'
        moves = re.findall(move_pattern, answer)

        validation_result['moves_found'] = moves

        for fen in fens:
            for move in moves:
                analysis = self.validate_move(fen, move)
                if analysis.is_legal:
                    validation_result['validation_score'] += 1.0
                    if analysis.move_quality in ['excellent', 'good']:
                        validation_result['validation_score'] += 0.5
                else:
                    validation_result['issues'].append(f"Move {move} is illegal in position {fen}")

        if moves:
            validation_result['validation_score'] /= len(moves)

        # Generate recommendations
        if validation_result['validation_score'] < 0.5:
            validation_result['recommendations'].append("Consider revising the answer - many moves appear invalid")
        if not fens:
            validation_result['recommendations'].append("Consider adding FEN positions to questions for better context")

        return validation_result

    def batch_validate_moves(self, moves_data: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Validate multiple moves sequentially with thread-safe engine access."""
        results = []

        for item in moves_data:
            try:
                results.append(self.validate_dataset_entry(item['question'], item['answer']))
            except Exception as e:
                logger.error(f"Error in batch validation: {e}")

        return results

    def generate_position_analysis(self, fen: str) -> str:
        """Generate a natural language analysis of a chess position."""
        analysis = self.analyze_position(fen)

        if not analysis.best_move:
            return "Unable to analyze this position."

        response = f"Position Analysis ({analysis.position_type.title()}):\n\n"

        if analysis.mate_in:
            response += f"Checkmate in {abs(analysis.mate_in)} moves!\n"
        elif analysis.best_score:
            score = analysis.best_score / 100.0
            response += f"Evaluation: {score:+.2f} pawns (White advantage)\n"

        response += f"Best move: {analysis.best_move}\n\n"

        if analysis.threats:
            response += "Threats:\n"
            for threat in analysis.threats:
                response += f"- {threat}\n"

        if analysis.opportunities:
            response += "\nOpportunities:\n"
            for opp in analysis.opportunities:
                response += f"- {opp}\n"

        if analysis.top_moves:
            response += f"\nTop {len(analysis.top_moves)} moves:\n"
            for i, move in enumerate(analysis.top_moves[:3], 1):
                score = move.centipawn_score / 100.0 if move.centipawn_score else "?"
                response += f"{i}. {move.move} ({score:+.1f})\n"

        return response


def create_stockfish_manager(config: Dict[str, Any]) -> ChessEngineManager:
    """Create a configured Stockfish engine manager from configuration data."""

    engine_path = config.get('engine_path', "/opt/homebrew/bin/stockfish")
    threads = config.get('threads', 2)
    hash_mb = config.get('hash', 128)
    skill_level = config.get('skill_level', 20)
    show_wdl = config.get('show_wdl', True)
    search_paths = config.get('search_paths') or DEFAULT_STOCKFISH_PATHS

    engine_options = {
        'Threads': threads,
        'Hash': hash_mb,
        'Skill Level': skill_level,
        'UCI_LimitStrength': False,
        'UCI_ShowWDL': bool(show_wdl),
    }

    return ChessEngineManager(
        engine_path=engine_path,
        engine_options=engine_options,
        name="Stockfish",
        search_paths=search_paths,
        debug=config.get('debug', False),
    )


def create_lc0_manager(config: Dict[str, Any]) -> ChessEngineManager:
    """Create a configured LC0 engine manager from configuration data."""

    engine_path = config.get('engine_path', 'lc0')
    weights_file = config.get('weights_file') or ""
    backend = config.get('backend', 'metal')
    threads = config.get('threads', 2)
    nn_cache_size = config.get('nn_cache_size')
    search_paths = config.get('search_paths') or [
        "/opt/homebrew/bin/lc0",
        "/usr/local/bin/lc0",
        "/usr/bin/lc0",
        "lc0",
    ]

    engine_options: Dict[str, Any] = {
        'Threads': threads,
    }

    # Always set weights file if provided to ensure custom weights are used
    if weights_file:
        # Use absolute path to ensure correct file is loaded
        import os
        weights_path = os.path.abspath(weights_file)
        engine_options['WeightsFile'] = weights_path

        # Verify the weights file exists
        if os.path.exists(weights_path):
            logger.info(f"[LC0] Using custom weights file: {weights_path}")
            # Ensure this takes precedence over any default weights
            logger.info(f"[LC0] Weights file priority set to override defaults")
        else:
            logger.warning(f"[LC0] Custom weights file not found at {weights_path}, engine may use default weights")
            # Remove the option if file doesn't exist to avoid LC0 errors
            engine_options.pop('WeightsFile', None)
    if backend:
        engine_options['Backend'] = backend
    if nn_cache_size is not None:
        # Only pass NNCacheSize if provided by config; lc0 will ignore if unsupported
        engine_options['NNCacheSize'] = nn_cache_size

    manager = ChessEngineManager(
        engine_path=engine_path,
        engine_options=engine_options,
        name="LC0",
        search_paths=search_paths,
        debug=config.get('debug', False),
    )

    if weights_file and not Path(weights_file).expanduser().exists():
        logger.warning(f"[LC0] Weights file not found at {weights_file}. Engine may fail to load network.")

    return manager


class LC0EnginePool:
    """
    Singleton pool for managing LC0 engine instances.

    Prevents multiple LC0 processes from spawning and provides proper
    lifecycle management with comprehensive logging.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialize_pool()
        return cls._instance

    def _initialize_pool(self):
        """Initialize the engine pool."""
        self._engines: Dict[str, ChessEngineManager] = {}
        self._engine_usage: Dict[str, int] = {}
        self._pool_lock = threading.RLock()
        self._creation_times: Dict[str, float] = {}

        # Register cleanup on exit
        import atexit
        atexit.register(self.cleanup_all)

    def get_engine(self, config_key: str = "default",
                   config: Optional[Dict[str, Any]] = None) -> ChessEngineManager:
        """
        Get or create an LC0 engine instance for the given configuration.

        Args:
            config_key: Unique identifier for engine configuration
            config: Optional engine configuration override

        Returns:
            ChessEngineManager instance for the configuration
        """
        with self._pool_lock:
            if config_key not in self._engines:
                logger.info(f"Creating new LC0 engine instance: {config_key}")

                # Use provided config or default LC0 config
                if config is None:
                    config_manager = get_config_manager()
                    if config_manager:
                        try:
                            config = asdict(config_manager().chess_engine.lc0)
                        except Exception:
                            config = {
                                'engine_path': '/opt/homebrew/bin/lc0',
                                'weights_file': 'models/lc0_weights/network.pb.gz',
                                'backend': 'metal',
                                'threads': 4,
                                'nn_cache_size': 262144,
                                'time_limit': 1.5
                            }

                # Create the engine instance
                engine = create_lc0_manager(config)
                self._engines[config_key] = engine
                self._engine_usage[config_key] = 0
                self._creation_times[config_key] = time.time()

                logger.info(f"LC0 engine '{config_key}' created successfully")

            # Increment usage counter
            self._engine_usage[config_key] += 1

            engine = self._engines[config_key]
            logger.debug(f"LC0 engine '{config_key}' usage count: {self._engine_usage[config_key]}")

            return engine

    def release_engine(self, config_key: str = "default") -> None:
        """
        Release an engine instance (decrement usage counter).

        Args:
            config_key: Engine configuration key to release
        """
        with self._pool_lock:
            if config_key in self._engine_usage:
                self._engine_usage[config_key] -= 1
                logger.debug(f"LC0 engine '{config_key}' usage count: {self._engine_usage[config_key]}")

    def cleanup_unused_engines(self, max_idle_time: float = 300.0) -> int:
        """
        Clean up engine instances that haven't been used recently.

        Args:
            max_idle_time: Maximum idle time in seconds before cleanup

        Returns:
            Number of engines cleaned up
        """
        with self._pool_lock:
            current_time = time.time()
            cleaned_count = 0

            keys_to_remove = []
            for config_key, last_used in self._engine_usage.items():
                # Check if engine hasn't been used and is old enough
                if (last_used == 0 and
                    config_key in self._creation_times and
                    current_time - self._creation_times[config_key] > max_idle_time):

                    logger.info(f"Cleaning up unused LC0 engine: {config_key}")

                    # Clean up the engine
                    try:
                        engine = self._engines[config_key]
                        engine.cleanup()
                    except Exception as e:
                        logger.warning(f"Error cleaning up engine {config_key}: {e}")

                    keys_to_remove.append(config_key)
                    cleaned_count += 1

            # Remove from tracking dictionaries
            for key in keys_to_remove:
                self._engines.pop(key, None)
                self._engine_usage.pop(key, None)
                self._creation_times.pop(key, None)

            if cleaned_count > 0:
                logger.info(f"Cleaned up {cleaned_count} unused LC0 engine(s)")

            return cleaned_count

    def cleanup_all(self) -> None:
        """
        Clean up all engine instances in the pool.
        Called automatically on program exit.
        """
        self._safe_log("info", "Cleaning up all LC0 engine instances")

        with self._pool_lock:
            for config_key, engine in self._engines.items():
                try:
                    self._safe_log("debug", f"Cleaning up LC0 engine: {config_key}")
                    engine.cleanup()
                except Exception as e:
                    self._safe_log("warning", f"Error cleaning up engine {config_key}: {e}")

            self._engines.clear()
            self._engine_usage.clear()
            self._creation_times.clear()

        self._safe_log("info", "All LC0 engine instances cleaned up")

    def get_pool_status(self) -> Dict[str, Any]:
        """
        Get comprehensive status of the engine pool.

        Returns:
            Dictionary with pool statistics and engine details
        """
        with self._pool_lock:
            current_time = time.time()

            status = {
                'total_engines': len(self._engines),
                'active_engines': sum(1 for usage in self._engine_usage.values() if usage > 0),
                'engines': {}
            }

            for config_key in self._engines.keys():
                engine = self._engines[config_key]
                usage = self._engine_usage.get(config_key, 0)
                creation_time = self._creation_times.get(config_key, 0)
                age = current_time - creation_time if creation_time > 0 else 0

                status['engines'][config_key] = {
                    'usage_count': usage,
                    'creation_time': creation_time,
                    'age_seconds': age,
                    'is_active': usage > 0,
                    'engine_name': getattr(engine, 'name', 'Unknown')
                }

            return status

    @staticmethod
    def _safe_log(level: str, message: str) -> None:
        """Emit log messages defensively, even during interpreter shutdown."""
        try:
            handlers = getattr(logger, "handlers", [])
        except Exception:
            pass
        else:
            has_open_handler = False
            for handler in handlers:
                stream = getattr(handler, "stream", None)
                if stream is None or not getattr(stream, "closed", False):
                    has_open_handler = True
                    break
            if not has_open_handler:
                return
        try:
            getattr(logger, level)(message)
        except Exception:
            pass


# Global instance for easy access
lc0_pool = LC0EnginePool()


# Convenience functions for easy integration
def validate_chess_move(fen: str, move: str) -> MoveAnalysis:
    """Convenience function for single move validation."""
    with ChessEngineManager() as engine:
        return engine.validate_move(fen, move)


def analyze_chess_position(fen: str) -> PositionAnalysis:
    """Convenience function for position analysis."""
    with ChessEngineManager() as engine:
        return engine.analyze_position(fen)


def generate_position_explanation(fen: str) -> str:
    """Convenience function for natural language analysis."""
    with ChessEngineManager() as engine:
        return engine.generate_position_analysis(fen)


# LC0 Pool test function for debugging
def test_lc0_pool() -> Dict[str, Any]:
    """
    Test the LC0 engine pool functionality.

    Returns:
        Dictionary with test results and pool status
    """
    results = {
        'pool_created': False,
        'engine_created': False,
        'move_generated': False,
        'pool_status': {},
        'errors': []
    }

    try:
        # Test pool creation
        pool = lc0_pool
        results['pool_created'] = True
        logger.info("LC0 engine pool created successfully")

        # Test engine creation
        engine = pool.get_engine("test")
        results['engine_created'] = True
        logger.info("LC0 engine instance created from pool")

        # Test move generation
        import chess
        board = chess.Board()
        move = engine.get_best_move(board, depth=8, time_limit_ms=2000)
        if move:
            results['move_generated'] = True
            logger.info(f"LC0 generated move: {move.uci()}")

        # Get pool status
        results['pool_status'] = pool.get_pool_status()
        logger.info(f"LC0 pool status: {results['pool_status']}")

    except Exception as e:
        results['errors'].append(str(e))
        logger.error(f"LC0 pool test failed: {e}")

    return results


if __name__ == "__main__":
    # Test the engine integration and LC0 pool
    print("Testing Chess Engine Integration & LC0 Pool...")
    print("=" * 60)

    # Test LC0 pool
    print("\nTesting LC0 Engine Pool...")
    try:
        pool_test = test_lc0_pool()
        print(f"Pool created: {'OK' if pool_test['pool_created'] else 'FAIL'}")
        print(f"Engine created: {'OK' if pool_test['engine_created'] else 'FAIL'}")
        print(f"Move generated: {'OK' if pool_test['move_generated'] else 'FAIL'}")

        if pool_test['errors']:
            print(f"Errors: {pool_test['errors']}")

        # Show pool status
        status = pool_test['pool_status']
        print(f"Total engines: {status.get('total_engines', 0)}")
        print(f"Active engines: {status.get('active_engines', 0)}")

        for engine_key, engine_info in status.get('engines', {}).items():
            print(f"  {engine_key}: usage={engine_info['usage_count']}, active={engine_info['is_active']}")

    except Exception as e:
        print(f"LC0 pool test failed: {e}")

    # Test basic engine creation (backward compatibility)
    print("\nTesting Basic Engine Creation...")
    try:
        print("Creating engine...")
        with ChessEngineManager() as engine:
            print(f"Engine created: {engine.name}")

            # Test move validation
            starting_position = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
            test_move = "e2e4"

            print(f"Testing move {test_move} in starting position...")
            with ChessEngineManager() as engine:
                analysis = engine.validate_move(starting_position, test_move)
                print(f"Move: {analysis.move}")
                print(f"Legal: {analysis.is_legal}")
                print(f"Best move: {analysis.is_best}")
                print(f"Quality: {analysis.move_quality}")
                print(f"Explanation: {analysis.explanation}")

    except Exception as e:
        print(f"Engine test failed: {e}")

    print("=" * 60)
    print("Engine integration test complete.")
