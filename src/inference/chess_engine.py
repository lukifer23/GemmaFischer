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

import chess
import chess.engine
import chess.pgn
import chess.svg
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import time
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    top_moves: List[MoveAnalysis] = field(default_factory=list)
    threats: List[str] = field(default_factory=list)
    opportunities: List[str] = field(default_factory=list)
    position_type: str = "middle_game"  # opening, middle_game, endgame


class ChessEngineManager:
    """High-level chess engine management with Stockfish integration."""

    def __init__(self, engine_path: str = "/opt/homebrew/bin/stockfish", debug: bool = False):
        """Initialize chess engine with comprehensive error handling."""
        self.engine_path = engine_path
        self.debug = debug
        self.engine = None
        self.executor = ThreadPoolExecutor(max_workers=4)

        # Engine configuration
        self.engine_options = {
            'Threads': 2,  # Use 2 threads on Apple Silicon
            'Hash': 128,   # 128MB hash table
            'Skill Level': 20,  # Maximum skill
            'UCI_LimitStrength': False,
            'UCI_ShowWDL': True,  # Show win/draw/loss probabilities
        }

        self._initialize_engine()

    def _initialize_engine(self) -> None:
        """Initialize Stockfish engine with retry logic."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Try provided path first; if it fails, try discovery
                try:
                    self.engine = chess.engine.SimpleEngine.popen_uci(self.engine_path)
                except Exception:
                    discovered = self._find_stockfish()
                    if not discovered:
                        raise
                    self.engine = chess.engine.SimpleEngine.popen_uci(discovered)

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
                    logger.info(f"Configured engine with options: {list(valid_options.keys())}")

                # Verify engine is responsive
                try:
                    self.engine.ping()
                except Exception:
                    # Fallback: issue a very quick analyse to ensure readiness
                    _ = self.engine.analyse(chess.Board(), chess.engine.Limit(depth=1, time=0.01))
                logger.info("Stockfish engine initialized successfully")

                # Test with a simple position
                board = chess.Board()
                info = self.engine.analyse(board, chess.engine.Limit(depth=10))
                logger.info(f"Engine test successful, score: {info['score']}")

                return

            except Exception as e:
                logger.warning(f"Engine initialization attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    raise RuntimeError(f"Failed to initialize Stockfish engine after {max_retries} attempts")
                time.sleep(1)

    def _find_stockfish(self) -> Optional[str]:
        """Find Stockfish binary in common locations."""
        import os
        common_paths = [
            "/opt/homebrew/bin/stockfish",
            "/usr/local/bin/stockfish",
            "/usr/bin/stockfish",
            "stockfish",
        ]
        for path in common_paths:
            if os.path.exists(path) or (os.system(f"which {path} > /dev/null 2>&1") == 0):
                return path
        return None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

    def cleanup(self) -> None:
        """Clean up engine resources."""
        if self.engine:
            try:
                self.engine.quit()
                logger.info("Chess engine cleaned up successfully")
            except Exception as e:
                logger.warning(f"Error during engine cleanup: {e}")
        self.executor.shutdown(wait=True)

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
            info = self.engine.analyse(board, limit)

            # Get the best move for comparison
            best_move_result = self.engine.play(board, limit)
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
        info = self.engine.analyse(temp_board, limit)
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

            # Get main analysis
            limit = chess.engine.Limit(depth=depth, time=time_limit)
            info = self.engine.analyse(board, limit)

            # Get best move
            best_move_result = self.engine.play(board, limit)
            best_move = best_move_result.move.uci() if best_move_result.move else None

            # Get top 5 moves for variety
            top_moves = self._get_top_moves(board, limit, count=5)

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

            info = self.engine.analyse(temp_board, chess.engine.Limit(depth=10, time=0.3))

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
            result = self.engine.play(board, limit)
            return result.move if result and result.move else None
        except Exception as e:
            logger.error(f"Error getting best move from engine: {e}")
            return None

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
        """Validate multiple moves concurrently."""
        results = []

        def validate_single(item):
            return self.validate_dataset_entry(item['question'], item['answer'])

        futures = [self.executor.submit(validate_single, item) for item in moves_data]

        for future in as_completed(futures):
            try:
                results.append(future.result())
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
            response += f"♔ Checkmate in {abs(analysis.mate_in)} moves!\n"
        elif analysis.best_score:
            score = analysis.best_score / 100.0
            response += f"Evaluation: {score:+.2f} pawns (White advantage)\n"

        response += f"Best move: {analysis.best_move}\n\n"

        if analysis.threats:
            response += "Threats:\n"
            for threat in analysis.threats:
                response += f"• {threat}\n"

        if analysis.opportunities:
            response += "\nOpportunities:\n"
            for opp in analysis.opportunities:
                response += f"• {opp}\n"

        if analysis.top_moves:
            response += f"\nTop {len(analysis.top_moves)} moves:\n"
            for i, move in enumerate(analysis.top_moves[:3], 1):
                score = move.centipawn_score / 100.0 if move.centipawn_score else "?"
                response += f"{i}. {move.move} ({score:+.1f})\n"

        return response


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


if __name__ == "__main__":
    # Test the engine integration
    print("Testing Chess Engine Integration...")

    # Test basic move validation
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

        # Test position analysis
        print("\nAnalyzing starting position...")
        pos_analysis = engine.analyze_position(starting_position)
        print(f"Best move: {pos_analysis.best_move}")
        print(f"Position type: {pos_analysis.position_type}")
        print(f"Top moves: {[m.move for m in pos_analysis.top_moves[:3]]}")

    print("Chess engine integration test completed!")
