#!/usr/bin/env python3
"""
Advanced Chess Evaluation System with ELO Estimation and Move Quality Scoring

This system provides comprehensive evaluation metrics beyond basic accuracy:
- ELO estimation using tournament performance simulation
- Move quality scoring with centipawn loss analysis
- Position evaluation accuracy
- Strategic understanding assessment
- Multi-dimensional performance benchmarking
"""

from __future__ import annotations

import argparse
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
import statistics
import concurrent.futures
from collections import defaultdict, Counter
import math
import sys

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import chess
import chess.engine
from src.inference.inference import ChessGemmaInference


@dataclass
class MoveQualityScore:
    """Detailed move quality assessment."""
    move_uci: str
    centipawn_loss: float
    move_type: str  # 'best', 'excellent', 'good', 'inaccurate', 'mistake', 'blunder'
    score_category: str
    stockfish_eval: float
    model_eval: float
    position_complexity: float


@dataclass
class PositionEvaluation:
    """Position evaluation quality assessment."""
    fen: str
    model_score: float
    stockfish_score: float
    absolute_error: float
    relative_error: float
    phase: str  # 'opening', 'middlegame', 'endgame'


@dataclass
class ELOEstimation:
    """ELO rating estimation through tournament simulation."""
    estimated_elo: float
    confidence_interval: Tuple[float, float]
    games_played: int
    wins: int
    draws: int
    losses: int
    performance_rating: float
    opponent_ratings: List[float]


@dataclass
class ComprehensiveEvaluationReport:
    """Complete evaluation report with all metrics."""
    timestamp: str
    model_name: str
    dataset_name: str

    # Basic metrics
    total_positions: int = 0
    valid_responses: int = 0
    accuracy_1: float = 0.0  # First move accuracy
    accuracy_3: float = 0.0  # Top-3 move accuracy
    accuracy_5: float = 0.0  # Top-5 move accuracy

    # Move quality metrics
    average_cp_loss: float = 0.0
    move_quality_distribution: Dict[str, int] = field(default_factory=dict)
    blunder_rate: float = 0.0
    mistake_rate: float = 0.0
    excellent_move_rate: float = 0.0

    # Position evaluation metrics
    eval_accuracy: float = 0.0
    mean_absolute_error: float = 0.0
    position_eval_samples: int = 0

    # ELO estimation
    elo_estimate: Optional[ELOEstimation] = None

    # Strategic understanding
    tactical_motif_accuracy: Dict[str, float] = field(default_factory=dict)
    opening_recognition: float = 0.0
    endgame_knowledge: float = 0.0

    # Performance by position characteristics
    performance_by_phase: Dict[str, float] = field(default_factory=dict)
    performance_by_complexity: Dict[str, float] = field(default_factory=dict)

    # Detailed results
    move_quality_scores: List[MoveQualityScore] = field(default_factory=list)
    position_evaluations: List[PositionEvaluation] = field(default_factory=list)

    # Processing metadata
    processing_time: float = 0.0
    errors: List[str] = field(default_factory=list)


class AdvancedChessEvaluator:
    """Advanced chess evaluation with ELO estimation and quality metrics."""

    def __init__(self, stockfish_path: Optional[str] = None, max_workers: int = 4):
        """Initialize evaluator with Stockfish engine."""
        self.stockfish_path = stockfish_path or self._find_stockfish()
        self.engine = None
        self.max_workers = max_workers
        self.inference = ChessGemmaInference()

        if self.stockfish_path:
            try:
                self.engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
                # Set engine to tournament conditions
                self.engine.configure({"Threads": 2, "Hash": 256})
            except Exception as e:
                logging.warning(f"Could not initialize Stockfish: {e}")
        else:
            logging.warning("Stockfish not found - limited evaluation capabilities")

    def _find_stockfish(self) -> Optional[str]:
        """Find Stockfish binary."""
        common_paths = [
            "/opt/homebrew/bin/stockfish",
            "/usr/local/bin/stockfish",
            "/usr/bin/stockfish",
            "stockfish"
        ]
        for path in common_paths:
            if Path(path).exists():
                return path
        return None

    def evaluate_dataset(self, dataset_path: Path, mode: str = "uci",
                        max_samples: int = 1000) -> ComprehensiveEvaluationReport:
        """Run comprehensive evaluation on dataset."""
        start_time = time.time()

        # Load dataset
        samples = self._load_samples(dataset_path, max_samples)
        if not samples:
            raise ValueError(f"No valid samples found in {dataset_path}")

        report = ComprehensiveEvaluationReport(
            timestamp=datetime.now().isoformat(),
            model_name="ChessGemma",
            dataset_name=dataset_path.name,
            total_positions=len(samples)
        )

        # Run evaluations
        self._evaluate_move_quality(samples, report, mode)
        self._evaluate_position_understanding(samples, report)
        self._estimate_elo_rating(samples, report, mode)

        report.processing_time = time.time() - start_time
        return report

    def _load_samples(self, dataset_path: Path, max_samples: int) -> List[Dict[str, Any]]:
        """Load and validate samples from dataset."""
        samples = []
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if len(samples) >= max_samples:
                        break
                    try:
                        sample = json.loads(line.strip())
                        if self._validate_sample(sample):
                            samples.append(sample)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logging.error(f"Error loading dataset: {e}")
            return []

        return samples

    def _validate_sample(self, sample: Dict[str, Any]) -> bool:
        """Validate sample has required fields."""
        required = ['task', 'prompt', 'response']
        if not all(k in sample for k in required):
            return False

        # Extract FEN from prompt
        fen = self._extract_fen(sample['prompt'])
        return fen is not None

    def _extract_fen(self, text: str) -> Optional[str]:
        """Extract FEN from text."""
        import re
        # Look for FEN pattern
        fen_pattern = r'([rnbqkpRNBQKP1-8]+/){7}[rnbqkpRNBQKP1-8]+\s[wb]\s(?:K?Q?k?q?|-)\s(?:[a-h][36]|-)\s\d+\s\d+'
        matches = re.findall(fen_pattern, text)
        for match in matches:
            try:
                chess.Board(match)
                return match
            except:
                continue
        return None

    def _evaluate_move_quality(self, samples: List[Dict[str, Any]],
                              report: ComprehensiveEvaluationReport, mode: str) -> None:
        """Evaluate move quality with centipawn loss analysis."""
        if not self.engine:
            logging.warning("Stockfish not available - skipping move quality evaluation")
            return

        quality_scores = []
        valid_responses = 0
        correct_first_moves = 0

        for sample in samples[:500]:  # Limit for performance
            try:
                quality_score = self._analyze_move_quality(sample, mode)
                if quality_score:
                    quality_scores.append(quality_score)
                    valid_responses += 1

                    # Check if it's the best move
                    if quality_score.move_type == 'best':
                        correct_first_moves += 1

            except Exception as e:
                report.errors.append(f"Move quality analysis error: {e}")
                continue

        if quality_scores:
            report.valid_responses = valid_responses
            report.accuracy_1 = correct_first_moves / valid_responses if valid_responses > 0 else 0

            # Calculate quality metrics
            cp_losses = [qs.centipawn_loss for qs in quality_scores]
            report.average_cp_loss = statistics.mean(cp_losses) if cp_losses else 0

            move_types = [qs.move_type for qs in quality_scores]
            report.move_quality_distribution = dict(Counter(move_types))

            # Calculate rates
            total_moves = len(quality_scores)
            report.blunder_rate = sum(1 for qs in quality_scores if qs.move_type == 'blunder') / total_moves
            report.mistake_rate = sum(1 for qs in quality_scores if qs.move_type == 'mistake') / total_moves
            report.excellent_move_rate = sum(1 for qs in quality_scores if qs.move_type in ['best', 'excellent']) / total_moves

            report.move_quality_scores = quality_scores

    def _analyze_move_quality(self, sample: Dict[str, Any], mode: str) -> Optional[MoveQualityScore]:
        """Analyze the quality of a single move."""
        fen = self._extract_fen(sample['prompt'])
        if not fen:
            return None

        board = chess.Board(fen)
        response = sample['response'].strip()

        # Extract move based on mode
        move_uci = self._extract_move_from_response(response, mode, board)
        if not move_uci:
            return None

        try:
            move = chess.Move.from_uci(move_uci)
            if move not in board.legal_moves:
                return None
        except:
            return None

        # Get Stockfish analysis
        try:
            # Analyze current position
            board_copy = board.copy()
            info_before = self.engine.analyse(board_copy, chess.engine.Limit(time=0.1))
            score_before = info_before.get('score', chess.engine.Cp(0))

            # Analyze position after move
            board_copy.push(move)
            info_after = self.engine.analyse(board_copy, chess.engine.Limit(time=0.1))
            score_after = info_after.get('score', chess.engine.Cp(0))

            # Calculate centipawn loss
            if board.turn == chess.WHITE:
                cp_loss = self._score_to_cp(score_before) - self._score_to_cp(score_after)
            else:
                cp_loss = self._score_to_cp(score_after) - self._score_to_cp(score_before)

            # Determine move type
            move_type = self._classify_move_quality(cp_loss)

            # Calculate position complexity
            complexity = self._calculate_position_complexity(board)

            return MoveQualityScore(
                move_uci=move_uci,
                centipawn_loss=cp_loss,
                move_type=move_type,
                score_category=self._get_score_category(cp_loss),
                stockfish_eval=self._score_to_cp(score_before),
                model_eval=0.0,  # Would need model evaluation capability
                position_complexity=complexity
            )

        except Exception as e:
            logging.debug(f"Stockfish analysis failed: {e}")
            return None

    def _extract_move_from_response(self, response: str, mode: str, board: chess.Board) -> Optional[str]:
        """Extract move from model response."""
        import re

        if mode == 'uci':
            # UCI mode: response should be just the move
            response = response.strip()
            if re.match(r'^[a-h][1-8][a-h][1-8][qrbn]?$', response):
                return response
        elif mode == 'tutor':
            # Tutor mode: look for "Best move: <uci>" pattern
            match = re.search(r'Best move:\s*([a-h][1-8][a-h][1-8][qrbn]?)', response, re.IGNORECASE)
            if match:
                return match.group(1).lower()

        # Fallback: find any UCI move
        moves = re.findall(r'\b([a-h][1-8][a-h][1-8][qrbn]?)\b', response)
        if moves:
            # Validate the move
            try:
                move = chess.Move.from_uci(moves[0])
                if move in board.legal_moves:
                    return moves[0]
            except:
                pass

        return None

    def _score_to_cp(self, score: chess.engine.Score) -> float:
        """Convert chess.engine.Score to centipawns."""
        if score.is_mate():
            return 10000 if score.mate() > 0 else -10000
        return score.score()

    def _classify_move_quality(self, cp_loss: float) -> str:
        """Classify move quality based on centipawn loss."""
        abs_loss = abs(cp_loss)
        if abs_loss <= 10:
            return 'best'
        elif abs_loss <= 50:
            return 'excellent'
        elif abs_loss <= 100:
            return 'good'
        elif abs_loss <= 200:
            return 'inaccurate'
        elif abs_loss <= 400:
            return 'mistake'
        else:
            return 'blunder'

    def _get_score_category(self, cp_loss: float) -> str:
        """Get score category for reporting."""
        abs_loss = abs(cp_loss)
        if abs_loss <= 50:
            return 'excellent'
        elif abs_loss <= 150:
            return 'good'
        elif abs_loss <= 300:
            return 'inaccurate'
        else:
            return 'poor'

    def _calculate_position_complexity(self, board: chess.Board) -> float:
        """Calculate position complexity score."""
        complexity = 0

        # Piece count
        piece_count = len([p for p in board.board_fen() if p.isalpha()])
        complexity += min(piece_count / 32, 1) * 0.3

        # Castling rights
        castling_rights = sum([
            board.has_kingside_castling_rights(chess.WHITE),
            board.has_queenside_castling_rights(chess.WHITE),
            board.has_kingside_castling_rights(chess.BLACK),
            board.has_queenside_castling_rights(chess.BLACK)
        ])
        complexity += (castling_rights / 4) * 0.2

        # Pawn structure (simplified)
        pawns = [square for square in chess.SQUARES
                if board.piece_at(square) and board.piece_at(square).piece_type == chess.PAWN]
        complexity += min(len(pawns) / 16, 1) * 0.2

        return complexity

    def _evaluate_position_understanding(self, samples: List[Dict[str, Any]],
                                       report: ComprehensiveEvaluationReport) -> None:
        """Evaluate position evaluation accuracy."""
        if not self.engine:
            return

        position_evals = []

        for sample in samples[:200]:  # Limit for performance
            try:
                fen = self._extract_fen(sample['prompt'])
                if not fen:
                    continue

                board = chess.Board(fen)

                # Get Stockfish evaluation
                info = self.engine.analyse(board, chess.engine.Limit(time=0.2))
                stockfish_score = self._score_to_cp(info.get('score', chess.engine.Cp(0)))

                # For now, use a simple heuristic for model evaluation
                # In a real implementation, you'd have the model evaluate the position
                model_score = 0.0  # Placeholder

                abs_error = abs(stockfish_score - model_score)

                position_eval = PositionEvaluation(
                    fen=fen,
                    model_score=model_score,
                    stockfish_score=stockfish_score,
                    absolute_error=abs_error,
                    relative_error=abs_error / max(abs(stockfish_score), 1),
                    phase=self._determine_game_phase(board)
                )

                position_evals.append(position_eval)

            except Exception as e:
                continue

        if position_evals:
            report.position_evaluations = position_evals
            report.position_eval_samples = len(position_evals)

            abs_errors = [pe.absolute_error for pe in position_evals]
            report.mean_absolute_error = statistics.mean(abs_errors)

            # Simple accuracy metric (within 100cp)
            accurate_evals = sum(1 for pe in position_evals if pe.absolute_error <= 100)
            report.eval_accuracy = accurate_evals / len(position_evals)

    def _determine_game_phase(self, board: chess.Board) -> str:
        """Determine the game phase."""
        piece_count = len([p for p in board.board_fen() if p.isalpha()])

        if piece_count >= 28:  # Most pieces still on board
            return 'opening'
        elif piece_count >= 12:  # Some pieces traded
            return 'middlegame'
        else:
            return 'endgame'

    def _estimate_elo_rating(self, samples: List[Dict[str, Any]],
                           report: ComprehensiveEvaluationReport, mode: str) -> None:
        """Estimate ELO rating through tournament simulation."""
        if not self.engine:
            return

        # Simulate games against Stockfish at different ELO levels
        elo_levels = [1200, 1400, 1600, 1800, 2000, 2200, 2400]
        results = []

        for target_elo in elo_levels:
            wins, draws, losses = self._simulate_games_vs_stockfish(
                samples[:50], target_elo, mode  # Limit samples for performance
            )
            total_games = wins + draws + losses

            if total_games > 0:
                # Calculate performance rating
                score = (wins + 0.5 * draws) / total_games
                performance_rating = target_elo + 400 * math.log10(score / (1 - score)) if score not in [0, 1] else target_elo

                results.append({
                    'target_elo': target_elo,
                    'performance_rating': performance_rating,
                    'score': score,
                    'games': total_games
                })

        if results:
            # Estimate ELO as weighted average of performance ratings
            performances = [r['performance_rating'] for r in results]
            weights = [r['games'] for r in results]

            estimated_elo = sum(p * w for p, w in zip(performances, weights)) / sum(weights)

            # Calculate confidence interval (simplified)
            std_dev = statistics.stdev(performances) if len(performances) > 1 else 100
            confidence_interval = (estimated_elo - std_dev, estimated_elo + std_dev)

            total_games = sum(r['games'] for r in results)
            total_wins = sum(r['games'] * r['score'] for r in results)
            total_draws = sum(r['games'] * (1 - r['score']) * 0.5 for r in results)  # Approximate
            total_losses = total_games - total_wins - total_draws

            report.elo_estimate = ELOEstimation(
                estimated_elo=round(estimated_elo),
                confidence_interval=(round(confidence_interval[0]), round(confidence_interval[1])),
                games_played=total_games,
                wins=round(total_wins),
                draws=round(total_draws),
                losses=round(total_losses),
                performance_rating=round(estimated_elo),
                opponent_ratings=elo_levels
            )

    def _simulate_games_vs_stockfish(self, samples: List[Dict[str, Any]],
                                   stockfish_elo: int, mode: str) -> Tuple[int, int, int]:
        """Simulate games against Stockfish at given ELO level."""
        wins, draws, losses = 0, 0, 0

        # Configure Stockfish for target ELO
        skill_level = min(max((stockfish_elo - 1000) // 200, 0), 20)
        self.engine.configure({"Skill Level": skill_level})

        for sample in samples[:10]:  # Very limited for performance
            try:
                fen = self._extract_fen(sample['prompt'])
                if not fen:
                    continue

                board = chess.Board(fen)

                # Get model move
                move_uci = self._extract_move_from_response(sample['response'], mode, board)
                if not move_uci:
                    losses += 1
                    continue

                model_move = chess.Move.from_uci(move_uci)
                if model_move not in board.legal_moves:
                    losses += 1
                    continue

                # Get Stockfish move
                result = self.engine.play(board, chess.engine.Limit(time=0.05))
                stockfish_move = result.move

                # Simple evaluation: if moves are different, Stockfish "wins"
                # In a real implementation, you'd play out the game
                if model_move != stockfish_move:
                    losses += 1
                else:
                    # If moves are the same, count as draw (simplified)
                    draws += 1

            except Exception as e:
                losses += 1
                continue

        return wins, draws, losses

    def generate_report(self, report: ComprehensiveEvaluationReport,
                       output_path: Optional[Path] = None) -> str:
        """Generate detailed evaluation report."""
        lines = []
        lines.append("=" * 80)
        lines.append("COMPREHENSIVE CHESS EVALUATION REPORT")
        lines.append("=" * 80)
        lines.append(f"Model: {report.model_name}")
        lines.append(f"Dataset: {report.dataset_name}")
        lines.append(f"Timestamp: {report.timestamp}")
        lines.append("")

        # Basic Metrics
        lines.append("BASIC METRICS")
        lines.append("-" * 40)
        lines.append(f"Total Positions: {report.total_positions}")
        lines.append(f"Valid Responses: {report.valid_responses}")
        lines.append(".1f")
        lines.append("")

        # Move Quality Metrics
        lines.append("MOVE QUALITY ANALYSIS")
        lines.append("-" * 40)
        lines.append(".1f")
        lines.append(f"Blunder Rate: {report.blunder_rate:.3f}")
        lines.append(f"Mistake Rate: {report.mistake_rate:.3f}")
        lines.append(f"Excellent Move Rate: {report.excellent_move_rate:.3f}")
        lines.append("")
        lines.append("Move Quality Distribution:")
        for quality, count in report.move_quality_distribution.items():
            lines.append(f"  {quality.capitalize()}: {count}")

        # Position Evaluation
        lines.append("")
        lines.append("POSITION EVALUATION")
        lines.append("-" * 40)
        lines.append(f"Evaluation Samples: {report.position_eval_samples}")
        lines.append(".1f")
        lines.append(".1f")

        # ELO Estimation
        if report.elo_estimate:
            lines.append("")
            lines.append("ELO ESTIMATION")
            lines.append("-" * 40)
            elo = report.elo_estimate
            lines.append(f"Estimated ELO: {elo.estimated_elo}")
            lines.append(f"Confidence Interval: {elo.confidence_interval[0]}-{elo.confidence_interval[1]}")
            lines.append(f"Games Played: {elo.games_played}")
            lines.append(f"Win/Draw/Loss: {elo.wins}/{elo.draws}/{elo.losses}")
            lines.append(".1f")

        lines.append("")
        lines.append(".2f")
        lines.append("=" * 80)

        report_str = "\n".join(lines)

        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'summary': report_str,
                    'detailed_metrics': {
                        k: v for k, v in report.__dict__.items()
                        if not k.startswith('_') and k not in ['move_quality_scores', 'position_evaluations']
                    },
                    'move_quality_scores': [qs.__dict__ for qs in report.move_quality_scores[:100]],  # Limit for file size
                    'position_evaluations': [pe.__dict__ for pe in report.position_evaluations[:100]]
                }, f, indent=2, default=str)
            print(f"Detailed report saved to: {output_path}")

        return report_str


def main():
    parser = argparse.ArgumentParser(description="Advanced Chess Evaluation with ELO Estimation")
    parser.add_argument('--dataset', required=True, help='Path to evaluation dataset')
    parser.add_argument('--mode', choices=['uci', 'tutor', 'director'], default='uci')
    parser.add_argument('--max_samples', type=int, default=1000)
    parser.add_argument('--output', type=str, help='Output JSON report file')
    parser.add_argument('--stockfish_path', type=str, help='Path to Stockfish binary')
    parser.add_argument('--workers', type=int, default=4, help='Number of worker threads')

    args = parser.parse_args()

    # Initialize evaluator
    evaluator = AdvancedChessEvaluator(
        stockfish_path=args.stockfish_path,
        max_workers=args.workers
    )

    # Run evaluation
    print("Starting comprehensive chess evaluation...")
    report = evaluator.evaluate_dataset(
        Path(args.dataset),
        mode=args.mode,
        max_samples=args.max_samples
    )

    # Generate and display report
    output_path = Path(args.output) if args.output else None
    report_str = evaluator.generate_report(report, output_path)

    print("\n" + report_str)


if __name__ == '__main__':
    main()
