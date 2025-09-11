#!/usr/bin/env python3
"""
Comprehensive Chess Evaluation System

Advanced evaluation suite with:
- Stockfish analysis integration for move quality assessment
- Multi-dimensional performance metrics
- Position evaluation and strategic understanding
- Explanation quality assessment
- Automated benchmarking and reporting
- Real-time performance monitoring

Addresses the critical evaluation gaps identified in the audit.
"""

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

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from ..inference.enhanced_inference import get_inference_manager, analyze_chess_position, generate_best_move
    from ..inference.chess_engine import ChessEngineManager
    import chess
except ImportError:
    logger.warning("Enhanced inference not available, using fallback")
    get_inference_manager = None


@dataclass
class EvaluationMetrics:
    """Comprehensive evaluation metrics."""
    total_positions: int = 0
    correct_moves: int = 0
    move_accuracy: float = 0.0
    position_understanding_score: float = 0.0
    explanation_quality_score: float = 0.0
    strategic_accuracy: float = 0.0
    tactical_accuracy: float = 0.0
    average_response_time: float = 0.0
    stockfish_agreement_rate: float = 0.0
    move_quality_distribution: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    error_categories: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    performance_by_difficulty: Dict[str, Dict[str, float]] = field(default_factory=dict)


@dataclass
class PositionEvaluation:
    """Detailed evaluation of a single position."""
    fen: str
    predicted_move: str
    actual_move: str
    is_correct: bool
    confidence_score: float
    response_time: float
    explanation_quality: float
    strategic_understanding: float
    stockfish_evaluation: Dict[str, Any] = field(default_factory=dict)
    error_type: Optional[str] = None
    analysis_depth: str = "shallow"


class ComprehensiveChessEvaluator:
    """Comprehensive chess evaluation system."""

    def __init__(self, stockfish_path: Optional[str] = None, max_workers: int = 4):
        self.stockfish_path = stockfish_path
        self.max_workers = max_workers
        self.engine_manager = None
        self.inference_manager = None

        # Evaluation results
        self.metrics = EvaluationMetrics()
        self.position_evaluations: List[PositionEvaluation] = []

        # Chess knowledge base for quality assessment
        self.chess_concepts = self._load_chess_knowledge()

        logger.info("🔧 Comprehensive Chess Evaluator initialized")

    def _load_chess_knowledge(self) -> Dict[str, List[str]]:
        """Load chess knowledge base for quality assessment."""
        return {
            'opening_principles': [
                'develop pieces quickly',
                'castle early',
                'control the center',
                'don\'t move the same piece twice',
                'bring queen out too early'
            ],
            'tactical_themes': [
                'pin', 'fork', 'skewer', 'discovered attack',
                'double attack', 'zwischenzug', 'deflection',
                'overloading', 'interference', 'clearance'
            ],
            'strategic_concepts': [
                'pawn structure', 'piece activity', 'king safety',
                'space advantage', 'weak squares', 'open files',
                'outposts', 'initiative', 'tempo'
            ],
            'endgame_principles': [
                'king activity', 'pawn promotion', 'zugzwang',
                'opposition', 'triangulation'
            ]
        }

    def initialize_evaluation_system(self) -> bool:
        """Initialize all evaluation components."""
        try:
            # Initialize Stockfish engine
            if self.stockfish_path:
                self.engine_manager = ChessEngineManager(self.stockfish_path)
                logger.info("✅ Stockfish engine initialized")
            else:
                logger.warning("⚠️  No Stockfish path provided, engine analysis disabled")

            # Initialize inference system
            if get_inference_manager:
                self.inference_manager = get_inference_manager()
                if not self.inference_manager.initialize():
                    logger.warning("⚠️  Enhanced inference initialization failed")
                else:
                    logger.info("✅ Enhanced inference initialized")
            else:
                logger.warning("⚠️  Enhanced inference not available")

            return True

        except Exception as e:
            logger.error(f"❌ Failed to initialize evaluation system: {e}")
            return False

    def evaluate_position(self, fen: str, expected_move: str, difficulty: str = "medium") -> PositionEvaluation:
        """Evaluate a single chess position comprehensively."""
        start_time = time.time()

        try:
            # Get model predictions
            move_result = self._get_model_move(fen)
            analysis_result = self._get_model_analysis(fen)

            predicted_move = move_result.get('response', '').strip()
            explanation = analysis_result.get('response', '')

            # Assess correctness
            is_correct = self._assess_move_correctness(predicted_move, expected_move)

            # Get confidence and quality scores
            confidence = move_result.get('confidence', 0.5)
            explanation_quality = self._assess_explanation_quality(explanation, fen)
            strategic_understanding = self._assess_strategic_understanding(explanation, fen)

            # Stockfish analysis
            stockfish_eval = self._get_stockfish_evaluation(fen, predicted_move, expected_move)

            # Determine error type if incorrect
            error_type = None
            if not is_correct:
                error_type = self._classify_error(predicted_move, expected_move, explanation, fen)

            response_time = time.time() - start_time

            return PositionEvaluation(
                fen=fen,
                predicted_move=predicted_move,
                actual_move=expected_move,
                is_correct=is_correct,
                confidence_score=confidence,
                response_time=response_time,
                explanation_quality=explanation_quality,
                strategic_understanding=strategic_understanding,
                stockfish_evaluation=stockfish_eval,
                error_type=error_type,
                analysis_depth=self._assess_analysis_depth(explanation)
            )

        except Exception as e:
            logger.error(f"❌ Position evaluation failed: {e}")
            return PositionEvaluation(
                fen=fen,
                predicted_move="",
                actual_move=expected_move,
                is_correct=False,
                confidence_score=0.0,
                response_time=time.time() - start_time,
                explanation_quality=0.0,
                strategic_understanding=0.0,
                error_type="evaluation_error"
            )

    def _get_model_move(self, fen: str) -> Dict[str, Any]:
        """Get move prediction from model."""
        if self.inference_manager:
            return generate_best_move(fen)
        else:
            # Fallback to basic inference
            return {"response": "", "confidence": 0.0}

    def _get_model_analysis(self, fen: str) -> Dict[str, Any]:
        """Get position analysis from model."""
        if self.inference_manager:
            return analyze_chess_position(fen, "tutor")
        else:
            # Fallback to basic inference
            return {"response": "", "confidence": 0.0}

    def _assess_move_correctness(self, predicted: str, expected: str) -> bool:
        """Assess if predicted move is correct."""
        if not predicted or not expected:
            return False

        # Normalize moves (remove check/checkmate symbols)
        predicted_clean = predicted.strip().lower()[:4]  # Take first 4 chars (uci format)
        expected_clean = expected.strip().lower()[:4]

        return predicted_clean == expected_clean

    def _assess_explanation_quality(self, explanation: str, fen: str) -> float:
        """Assess the quality of the explanation (0.0 to 1.0)."""
        if not explanation:
            return 0.0

        explanation_lower = explanation.lower()
        quality_score = 0.0

        # Check for chess concept coverage
        concept_coverage = 0
        total_concepts = 0

        for category, concepts in self.chess_concepts.items():
            total_concepts += len(concepts)
            for concept in concepts:
                if concept.lower() in explanation_lower:
                    concept_coverage += 1

        if total_concepts > 0:
            quality_score += (concept_coverage / total_concepts) * 0.4

        # Check for structural elements
        structural_elements = [
            'material', 'position', 'king', 'pawn', 'piece',
            'development', 'center', 'attack', 'defense'
        ]

        structural_coverage = sum(1 for elem in structural_elements if elem in explanation_lower)
        quality_score += (structural_coverage / len(structural_elements)) * 0.3

        # Check for analysis depth (word count, specificity)
        word_count = len(explanation.split())
        if word_count > 50:
            quality_score += 0.2
        elif word_count > 20:
            quality_score += 0.1

        # Check for specific references (coordinates, piece names)
        has_coordinates = bool(any(coord in explanation_lower for coord in 'abcdefgh12345678'))
        has_pieces = bool(any(piece in explanation_lower for piece in ['pawn', 'knight', 'bishop', 'rook', 'queen', 'king']))

        if has_coordinates:
            quality_score += 0.05
        if has_pieces:
            quality_score += 0.05

        return min(1.0, quality_score)

    def _assess_strategic_understanding(self, explanation: str, fen: str) -> float:
        """Assess strategic understanding (0.0 to 1.0)."""
        if not explanation:
            return 0.0

        explanation_lower = explanation.lower()
        strategic_score = 0.0

        # Strategic concept coverage
        strategic_concepts = self.chess_concepts['strategic_concepts']
        concept_coverage = sum(1 for concept in strategic_concepts if concept in explanation_lower)
        strategic_score += (concept_coverage / len(strategic_concepts)) * 0.5

        # Long-term thinking indicators
        long_term_indicators = [
            'long term', 'future', 'plan', 'strategy', 'control',
            'initiative', 'tempo', 'advantage', 'weakness'
        ]

        long_term_coverage = sum(1 for indicator in long_term_indicators if indicator in explanation_lower)
        strategic_score += (long_term_coverage / len(long_term_indicators)) * 0.3

        # Position-specific analysis
        position_indicators = [
            'center', 'development', 'safety', 'structure', 'activity'
        ]

        position_coverage = sum(1 for indicator in position_indicators if indicator in explanation_lower)
        strategic_score += (position_coverage / len(position_indicators)) * 0.2

        return min(1.0, strategic_score)

    def _get_stockfish_evaluation(self, fen: str, predicted_move: str, expected_move: str) -> Dict[str, Any]:
        """Get Stockfish evaluation of moves."""
        if not self.engine_manager:
            return {}

        try:
            with self.engine_manager as engine:
                # Evaluate the position
                analysis = engine.analyze_position(fen, depth=12)

                return {
                    'position_evaluation': analysis.get('evaluation', 0.0),
                    'best_moves': analysis.get('best_moves', [])[:3],
                    'predicted_move_quality': self._assess_move_quality(fen, predicted_move),
                    'expected_move_quality': self._assess_move_quality(fen, expected_move)
                }
        except Exception as e:
            logger.warning(f"Stockfish evaluation failed: {e}")
            return {}

    def _assess_move_quality(self, fen: str, move: str) -> str:
        """Assess move quality using Stockfish."""
        if not self.engine_manager or not move:
            return "unknown"

        try:
            board = chess.Board(fen)
            move_obj = chess.Move.from_uci(move)
            if move_obj not in board.legal_moves:
                return "illegal"

            board.push(move_obj)

            with self.engine_manager as engine:
                analysis = engine.analyze_position(board.fen(), depth=8)
                eval_score = analysis.get('evaluation', 0.0)

                if eval_score > 1.0:
                    return "excellent"
                elif eval_score > 0.3:
                    return "good"
                elif eval_score > -0.3:
                    return "ok"
                elif eval_score > -1.0:
                    return "poor"
                else:
                    return "blunder"

        except Exception:
            return "unknown"

    def _classify_error(self, predicted: str, expected: str, explanation: str, fen: str) -> str:
        """Classify the type of error made."""
        if not predicted:
            return "no_move_predicted"

        if len(predicted.strip()) < 4:
            return "incomplete_move"

        # Check if move is legal
        try:
            board = chess.Board(fen)
            move_obj = chess.Move.from_uci(predicted.strip()[:4])
            if move_obj not in board.legal_moves:
                return "illegal_move"
        except:
            return "invalid_format"

        # Tactical vs strategic errors
        explanation_lower = (explanation or "").lower()

        if any(tactic in explanation_lower for tactic in self.chess_concepts['tactical_themes']):
            return "tactical_miss"
        elif any(strategy in explanation_lower for strategy in self.chess_concepts['strategic_concepts']):
            return "strategic_miss"
        else:
            return "general_error"

    def _assess_analysis_depth(self, explanation: str) -> str:
        """Assess the depth of analysis."""
        if not explanation:
            return "none"

        word_count = len(explanation.split())

        if word_count > 200:
            return "deep"
        elif word_count > 100:
            return "moderate"
        elif word_count > 50:
            return "shallow"
        else:
            return "minimal"

    def run_comprehensive_evaluation(self, test_dataset: List[Dict[str, Any]],
                                   max_positions: int = 100) -> Dict[str, Any]:
        """Run comprehensive evaluation on a test dataset."""
        logger.info(f"🎯 Running comprehensive evaluation on {min(len(test_dataset), max_positions)} positions")

        # Initialize evaluation system
        if not self.initialize_evaluation_system():
            return {"error": "Failed to initialize evaluation system"}

        # Select test positions
        test_positions = test_dataset[:max_positions]

        # Run evaluations
        start_time = time.time()

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_position = {
                executor.submit(self.evaluate_position,
                              pos['fen'], pos.get('label_move', ''),
                              pos.get('difficulty', 'medium')): pos
                for pos in test_positions
            }

            for future in concurrent.futures.as_completed(future_to_position):
                try:
                    evaluation = future.result()
                    self.position_evaluations.append(evaluation)
                except Exception as e:
                    logger.error(f"Position evaluation failed: {e}")

        # Calculate comprehensive metrics
        self._calculate_metrics()

        evaluation_time = time.time() - start_time

        logger.info(f"✅ Evaluation completed in {evaluation_time:.2f} seconds")
        logger.info(f"📊 Evaluated {len(self.position_evaluations)} positions")

        return self._generate_evaluation_report(evaluation_time)

    def _calculate_metrics(self):
        """Calculate comprehensive evaluation metrics."""
        if not self.position_evaluations:
            return

        total_positions = len(self.position_evaluations)

        # Basic accuracy metrics
        correct_moves = sum(1 for eval in self.position_evaluations if eval.is_correct)
        move_accuracy = correct_moves / total_positions

        # Quality scores
        explanation_qualities = [eval.explanation_quality for eval in self.position_evaluations]
        strategic_understandings = [eval.strategic_understanding for eval in self.position_evaluations]
        response_times = [eval.response_time for eval in self.position_evaluations]

        # Stockfish agreement
        stockfish_agreements = []
        for eval in self.position_evaluations:
            if eval.stockfish_evaluation:
                pred_quality = eval.stockfish_evaluation.get('predicted_move_quality', 'unknown')
                if pred_quality in ['excellent', 'good']:
                    stockfish_agreements.append(1)
                else:
                    stockfish_agreements.append(0)

        stockfish_agreement_rate = (
            sum(stockfish_agreements) / len(stockfish_agreements)
            if stockfish_agreements else 0.0
        )

        # Move quality distribution
        quality_counts = Counter()
        for eval in self.position_evaluations:
            if eval.stockfish_evaluation:
                quality = eval.stockfish_evaluation.get('predicted_move_quality', 'unknown')
                quality_counts[quality] += 1

        # Error analysis
        error_counts = Counter(eval.error_type for eval in self.position_evaluations if eval.error_type)

        # Update metrics
        self.metrics.total_positions = total_positions
        self.metrics.correct_moves = correct_moves
        self.metrics.move_accuracy = move_accuracy
        self.metrics.position_understanding_score = statistics.mean(explanation_qualities) if explanation_qualities else 0.0
        self.metrics.explanation_quality_score = statistics.mean(explanation_qualities) if explanation_qualities else 0.0
        self.metrics.strategic_accuracy = statistics.mean(strategic_understandings) if strategic_understandings else 0.0
        self.metrics.average_response_time = statistics.mean(response_times) if response_times else 0.0
        self.metrics.stockfish_agreement_rate = stockfish_agreement_rate
        self.metrics.move_quality_distribution = dict(quality_counts)
        self.metrics.error_categories = dict(error_counts)

    def _generate_evaluation_report(self, evaluation_time: float) -> Dict[str, Any]:
        """Generate comprehensive evaluation report."""
        report = {
            'evaluation_timestamp': datetime.now().isoformat(),
            'evaluation_duration': evaluation_time,
            'total_positions_evaluated': len(self.position_evaluations),

            'performance_metrics': {
                'move_accuracy': round(self.metrics.move_accuracy * 100, 2),
                'explanation_quality_score': round(self.metrics.explanation_quality_score, 3),
                'strategic_understanding_score': round(self.metrics.strategic_accuracy, 3),
                'average_response_time': round(self.metrics.average_response_time, 3),
                'stockfish_agreement_rate': round(self.metrics.stockfish_agreement_rate * 100, 2)
            },

            'move_quality_distribution': self.metrics.move_quality_distribution,
            'error_analysis': self.metrics.error_categories,

            'detailed_results': [
                {
                    'fen': eval.fen,
                    'predicted_move': eval.predicted_move,
                    'actual_move': eval.actual_move,
                    'is_correct': eval.is_correct,
                    'confidence': eval.confidence_score,
                    'explanation_quality': eval.explanation_quality,
                    'strategic_understanding': eval.strategic_understanding,
                    'response_time': eval.response_time,
                    'error_type': eval.error_type,
                    'analysis_depth': eval.analysis_depth
                }
                for eval in self.position_evaluations[:50]  # Limit detailed results
            ],

            'recommendations': self._generate_recommendations()
        }

        return report

    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on evaluation results."""
        recommendations = []

        if self.metrics.move_accuracy < 0.6:
            recommendations.append("Focus on improving basic move accuracy through enhanced training data")

        if self.metrics.explanation_quality_score < 0.5:
            recommendations.append("Improve explanation quality by training on more comprehensive analysis examples")

        if self.metrics.strategic_accuracy < 0.4:
            recommendations.append("Enhance strategic understanding through curriculum training on positional concepts")

        if self.metrics.average_response_time > 2.0:
            recommendations.append("Optimize inference performance for faster response times")

        if self.metrics.stockfish_agreement_rate < 0.7:
            recommendations.append("Align move quality with Stockfish evaluations through better training objectives")

        # Error-specific recommendations
        if self.metrics.error_categories.get('illegal_move', 0) > 10:
            recommendations.append("Add more emphasis on legal move generation in training")

        if self.metrics.error_categories.get('tactical_miss', 0) > 15:
            recommendations.append("Improve tactical pattern recognition through specialized training")

        return recommendations if recommendations else ["Performance is strong across all metrics"]

    def save_evaluation_report(self, report: Dict[str, Any], output_path: Path):
        """Save evaluation report to file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"📄 Evaluation report saved to: {output_path}")

    def run_benchmark_comparison(self, models_to_compare: List[str],
                                test_dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run benchmark comparison between different models."""
        logger.info(f"🏁 Running benchmark comparison for {len(models_to_compare)} models")

        benchmark_results = {}

        for model_name in models_to_compare:
            logger.info(f"Evaluating {model_name}...")

            # Switch to different model/expert (implementation would depend on your setup)
            if hasattr(self, 'switch_model'):
                self.switch_model(model_name)

            # Run evaluation
            model_results = self.run_comprehensive_evaluation(test_dataset, max_positions=50)
            benchmark_results[model_name] = model_results

        # Generate comparison report
        comparison_report = {
            'benchmark_timestamp': datetime.now().isoformat(),
            'models_compared': models_to_compare,
            'results': benchmark_results,
            'comparison_summary': self._generate_comparison_summary(benchmark_results)
        }

        return comparison_report

    def _generate_comparison_summary(self, benchmark_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of model comparisons."""
        if not benchmark_results:
            return {}

        summary = {
            'best_performing_model': None,
            'accuracy_leaderboard': [],
            'quality_leaderboard': [],
            'speed_leaderboard': []
        }

        # Sort models by different metrics
        accuracy_sorted = sorted(
            benchmark_results.items(),
            key=lambda x: x[1].get('performance_metrics', {}).get('move_accuracy', 0),
            reverse=True
        )

        quality_sorted = sorted(
            benchmark_results.items(),
            key=lambda x: x[1].get('performance_metrics', {}).get('explanation_quality_score', 0),
            reverse=True
        )

        speed_sorted = sorted(
            benchmark_results.items(),
            key=lambda x: x[1].get('performance_metrics', {}).get('average_response_time', float('inf'))
        )

        summary['accuracy_leaderboard'] = [model for model, _ in accuracy_sorted]
        summary['quality_leaderboard'] = [model for model, _ in quality_sorted]
        summary['speed_leaderboard'] = [model for model, _ in speed_sorted]
        summary['best_performing_model'] = accuracy_sorted[0][0] if accuracy_sorted else None

        return summary


def load_test_dataset(dataset_path: Path) -> List[Dict[str, Any]]:
    """Load test dataset from file."""
    test_data = []

    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    if 'fen' in item and ('label_move' in item or 'solution' in item):
                        test_data.append(item)
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        logger.error(f"Failed to load test dataset: {e}")

    return test_data


def main():
    """Main entry point for comprehensive evaluation."""
    parser = argparse.ArgumentParser(description="Comprehensive Chess Evaluation System")
    parser.add_argument('--dataset', type=Path, required=True, help='Path to test dataset')
    parser.add_argument('--output', type=Path, default=Path('evaluation_results.json'), help='Output file path')
    parser.add_argument('--max_positions', type=int, default=100, help='Maximum positions to evaluate')
    parser.add_argument('--stockfish_path', type=str, help='Path to Stockfish engine')
    parser.add_argument('--benchmark', nargs='+', help='Models to benchmark compare')

    args = parser.parse_args()

    print("🎯 Comprehensive Chess Evaluation System")
    print("=" * 50)

    # Load test dataset
    test_dataset = load_test_dataset(args.dataset)
    if not test_dataset:
        print(f"❌ Failed to load test dataset from {args.dataset}")
        return

    print(f"📚 Loaded {len(test_dataset)} test positions")

    # Initialize evaluator
    evaluator = ComprehensiveChessEvaluator(
        stockfish_path=args.stockfish_path,
        max_workers=4
    )

    if args.benchmark:
        # Run benchmark comparison
        print(f"🏁 Running benchmark comparison: {args.benchmark}")
        results = evaluator.run_benchmark_comparison(args.benchmark, test_dataset)
    else:
        # Run single comprehensive evaluation
        print("🔍 Running comprehensive evaluation...")
        results = evaluator.run_comprehensive_evaluation(test_dataset, args.max_positions)

    # Save results
    evaluator.save_evaluation_report(results, args.output)

    # Print summary
    if 'performance_metrics' in results:
        metrics = results['performance_metrics']
        print("\n📊 Performance Summary:")
        print(f"  Move Accuracy: {metrics['move_accuracy']}%")
        print(f"  Explanation Quality: {metrics['explanation_quality_score']:.3f}")
        print(f"  Strategic Understanding: {metrics['strategic_understanding_score']:.3f}")
        print(f"  Average Response Time: {metrics['average_response_time']:.3f}s")
        print(f"  Stockfish Agreement: {metrics['stockfish_agreement_rate']}%")

    print(f"\n✅ Evaluation complete! Results saved to: {args.output}")


if __name__ == '__main__':
    main()
