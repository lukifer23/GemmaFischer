#!/usr/bin/env python3
"""
Simplified Chess Evaluation Framework

Clear, interpretable evaluation metrics that focus on the most important
aspects of chess AI performance. Replaces overly complex evaluation system
with straightforward, actionable metrics.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import statistics
import chess
import re

logger = logging.getLogger(__name__)


@dataclass
class SimpleEvaluationMetrics:
    """Clear, interpretable evaluation metrics."""

    # Core performance metrics
    total_positions: int = 0
    correct_moves: int = 0
    move_accuracy: float = 0.0

    # Response quality metrics
    average_response_time: float = 0.0
    average_response_length: float = 0.0

    # Stockfish agreement metrics
    stockfish_agreement_rate: float = 0.0
    stockfish_agreement_top3: float = 0.0

    # Error analysis
    error_rate: float = 0.0
    timeout_rate: float = 0.0

    # Difficulty-based performance
    easy_accuracy: float = 0.0
    medium_accuracy: float = 0.0
    hard_accuracy: float = 0.0

    # Position type analysis
    tactical_accuracy: float = 0.0
    strategic_accuracy: float = 0.0
    endgame_accuracy: float = 0.0

    # Response quality indicators
    responses_with_explanation: float = 0.0
    responses_with_reasoning: float = 0.0

    # Overall score (0-100)
    overall_score: float = 0.0


@dataclass
class PositionResult:
    """Result for a single position evaluation."""
    fen: str
    expected_move: str
    predicted_move: str
    is_correct: bool
    response_time: float
    response_text: str
    position_difficulty: str = "medium"
    position_type: str = "mixed"
    stockfish_agreement: bool = False
    has_explanation: bool = False
    has_reasoning: bool = False


class SimpleChessEvaluator:
    """Simplified chess evaluation system with clear metrics."""

    def __init__(self, stockfish_path: Optional[str] = None):
        self.stockfish_path = stockfish_path
        self.stockfish_available = self._check_stockfish()

        # Difficulty thresholds based on rating
        self.difficulty_thresholds = {
            "easy": (800, 1400),
            "medium": (1400, 2000),
            "hard": (2000, 3000)
        }

        logger.info(f"Simple Chess Evaluator initialized (Stockfish: {self.stockfish_available})")

    def _check_stockfish(self) -> bool:
        """Check if Stockfish is available."""
        try:
            import subprocess
            result = subprocess.run(
                [self.stockfish_path or "stockfish"],
                input="uci\nquit\n",
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0
        except Exception:
            return False

    def evaluate_positions(self, positions_data: List[Dict[str, Any]],
                          expert_responses: List[str]) -> SimpleEvaluationMetrics:
        """Evaluate expert responses against expected moves."""
        logger.info(f"Evaluating {len(positions_data)} positions")

        results = []
        start_time = time.time()

        for i, (position, response) in enumerate(zip(positions_data, expert_responses)):
            try:
                result = self._evaluate_single_position(position, response, i)
                results.append(result)
            except Exception as e:
                logger.warning(f"Failed to evaluate position {i}: {e}")
                # Create a failed result
                result = PositionResult(
                    fen=position.get("fen", ""),
                    expected_move=position.get("best_move", ""),
                    predicted_move="ERROR",
                    is_correct=False,
                    response_time=0.0,
                    response_text=response,
                    position_difficulty="unknown",
                    position_type="unknown",
                    stockfish_agreement=False,
                    has_explanation=False,
                    has_reasoning=False
                )
                results.append(result)

        # Calculate metrics from results
        metrics = self._calculate_metrics(results)

        total_time = time.time() - start_time
        logger.info(f"Evaluation complete in {total_time:.2f}s. Overall score: {metrics.overall_score:.1f}/100")

        return metrics

    def _evaluate_single_position(self, position: Dict[str, Any], response: str, index: int) -> PositionResult:
        """Evaluate a single position."""
        fen = position.get("fen", "")
        expected_move = position.get("best_move", "")
        rating = position.get("rating", 1500)

        # Extract predicted move from response
        predicted_move = self._extract_move_from_response(response)

        # Determine position difficulty
        difficulty = self._classify_difficulty(rating)

        # Determine position type
        position_type = self._classify_position_type(position)

        # Check if move is correct
        is_correct = predicted_move == expected_move

        # Check Stockfish agreement if available
        stockfish_agreement = False
        if self.stockfish_available and fen:
            stockfish_agreement = self._check_stockfish_agreement(fen, predicted_move, expected_move)

        # Analyze response quality
        has_explanation = self._has_explanation(response)
        has_reasoning = self._has_reasoning(response)

        return PositionResult(
            fen=fen,
            expected_move=expected_move,
            predicted_move=predicted_move,
            is_correct=is_correct,
            response_time=0.0,  # Would need timing data
            response_text=response,
            position_difficulty=difficulty,
            position_type=position_type,
            stockfish_agreement=stockfish_agreement,
            has_explanation=has_explanation,
            has_reasoning=has_reasoning
        )

    def _extract_move_from_response(self, response: str) -> str:
        """Extract UCI move from response text."""
        # Look for UCI move pattern
        uci_pattern = r'\b([a-h][1-8][a-h][1-8][qrbn]?)\b'
        match = re.search(uci_pattern, response.lower())

        if match:
            return match.group(1)

        # Fallback: look for any chess move notation
        move_patterns = [
            r'[a-h][1-8][a-h][1-8]',  # Basic UCI
            r'[a-h][1-8][a-h][1-8][qrbn]',  # With promotion
            r'[NBRQ][a-h][1-8]',  # SAN notation
        ]

        for pattern in move_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(0).lower()

        return "NO_MOVE_FOUND"

    def _classify_difficulty(self, rating: int) -> str:
        """Classify position difficulty based on rating."""
        for difficulty, (min_rating, max_rating) in self.difficulty_thresholds.items():
            if min_rating <= rating <= max_rating:
                return difficulty
        return "medium"  # Default

    def _classify_position_type(self, position: Dict[str, Any]) -> str:
        """Classify position type based on metadata."""
        topic = position.get("topic", "").lower()

        if "tactic" in topic:
            return "tactical"
        elif "endgame" in topic:
            return "endgame"
        elif "opening" in topic:
            return "opening"
        else:
            return "mixed"

    def _check_stockfish_agreement(self, fen: str, predicted_move: str, expected_move: str) -> bool:
        """Check if predicted move agrees with Stockfish."""
        try:
            import subprocess

            # Run Stockfish analysis
            process = subprocess.run(
                [self.stockfish_path or "stockfish"],
                input=f"position fen {fen}\ngo depth 6\nquit\n",
                capture_output=True,
                text=True,
                timeout=10
            )

            if process.returncode != 0:
                return False

            # Parse Stockfish output for best move
            output = process.stdout
            best_move_line = None
            for line in output.split('\n'):
                if line.startswith('bestmove'):
                    best_move_line = line
                    break

            if best_move_line:
                stockfish_move = best_move_line.split()[1]
                return stockfish_move == predicted_move

        except Exception as e:
            logger.warning(f"Stockfish check failed: {e}")

        return False

    def _has_explanation(self, response: str) -> bool:
        """Check if response contains explanation."""
        explanation_indicators = [
            "because", "therefore", "since", "as a result",
            "explains", "reason", "why", "how", "this move"
        ]

        response_lower = response.lower()
        return any(indicator in response_lower for indicator in explanation_indicators)

    def _has_reasoning(self, response: str) -> bool:
        """Check if response contains reasoning."""
        reasoning_indicators = [
            "consider", "evaluate", "threat", "opportunity",
            "advantage", "disadvantage", "attack", "defense",
            "development", "king safety", "material"
        ]

        response_lower = response.lower()
        return any(indicator in response_lower for indicator in reasoning_indicators)

    def _calculate_metrics(self, results: List[PositionResult]) -> SimpleEvaluationMetrics:
        """Calculate evaluation metrics from results."""
        if not results:
            return SimpleEvaluationMetrics()

        metrics = SimpleEvaluationMetrics()

        # Basic counts
        metrics.total_positions = len(results)
        metrics.correct_moves = sum(1 for r in results if r.is_correct)

        if metrics.total_positions > 0:
            metrics.move_accuracy = metrics.correct_moves / metrics.total_positions

        # Response quality
        response_lengths = [len(r.response_text) for r in results if r.response_text]
        if response_lengths:
            metrics.average_response_length = statistics.mean(response_lengths)

        # Stockfish agreement
        stockfish_results = [r for r in results if r.stockfish_agreement is not None]
        if stockfish_results:
            stockfish_correct = sum(1 for r in stockfish_results if r.stockfish_agreement)
            metrics.stockfish_agreement_rate = stockfish_correct / len(stockfish_results)

        # Difficulty-based accuracy
        difficulty_results = {
            "easy": [r for r in results if r.position_difficulty == "easy"],
            "medium": [r for r in results if r.position_difficulty == "medium"],
            "hard": [r for r in results if r.position_difficulty == "hard"]
        }

        for difficulty, diff_results in difficulty_results.items():
            if diff_results:
                correct = sum(1 for r in diff_results if r.is_correct)
                accuracy = correct / len(diff_results)
                setattr(metrics, f"{difficulty}_accuracy", accuracy)

        # Position type accuracy
        type_results = {
            "tactical": [r for r in results if r.position_type == "tactical"],
            "strategic": [r for r in results if r.position_type == "strategic"],
            "endgame": [r for r in results if r.position_type == "endgame"]
        }

        for pos_type, type_results_list in type_results.items():
            if type_results_list:
                correct = sum(1 for r in type_results_list if r.is_correct)
                accuracy = correct / len(type_results_list)
                setattr(metrics, f"{pos_type}_accuracy", accuracy)

        # Response quality indicators
        total_responses = len(results)
        if total_responses > 0:
            metrics.responses_with_explanation = sum(1 for r in results if r.has_explanation) / total_responses
            metrics.responses_with_reasoning = sum(1 for r in results if r.has_reasoning) / total_responses

        # Calculate overall score (0-100)
        metrics.overall_score = self._calculate_overall_score(metrics)

        return metrics

    def _calculate_overall_score(self, metrics: SimpleEvaluationMetrics) -> float:
        """Calculate overall performance score (0-100)."""
        # Weighted scoring system
        weights = {
            "move_accuracy": 0.4,
            "stockfish_agreement": 0.3,
            "response_quality": 0.2,
            "consistency": 0.1
        }

        # Move accuracy score (0-40 points)
        move_score = metrics.move_accuracy * 40

        # Stockfish agreement score (0-30 points)
        stockfish_score = metrics.stockfish_agreement_rate * 30

        # Response quality score (0-20 points)
        quality_score = (metrics.responses_with_explanation + metrics.responses_with_reasoning) * 10

        # Consistency score (0-10 points) - based on variation across difficulties
        if all([metrics.easy_accuracy, metrics.medium_accuracy, metrics.hard_accuracy]):
            accuracy_variance = statistics.stdev([metrics.easy_accuracy, metrics.medium_accuracy, metrics.hard_accuracy])
            consistency_score = max(0, 10 - (accuracy_variance * 20))  # Penalize high variance
        else:
            consistency_score = 5  # Neutral score if insufficient data

        total_score = move_score + stockfish_score + quality_score + consistency_score
        return min(total_score, 100.0)

    def generate_evaluation_report(self, metrics: SimpleEvaluationMetrics,
                                 positions_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive evaluation report."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_positions": metrics.total_positions,
                "overall_score": round(metrics.overall_score, 1),
                "move_accuracy": round(metrics.move_accuracy * 100, 1),
                "stockfish_agreement": round(metrics.stockfish_agreement_rate * 100, 1),
                "average_response_time": round(metrics.average_response_time, 2),
                "response_quality": round((metrics.responses_with_explanation + metrics.responses_with_reasoning) * 50, 1)
            },
            "detailed_metrics": {
                "move_accuracy_percent": round(metrics.move_accuracy * 100, 1),
                "stockfish_agreement_percent": round(metrics.stockfish_agreement_rate * 100, 1),
                "stockfish_top3_agreement_percent": round(metrics.stockfish_agreement_top3 * 100, 1),
                "error_rate_percent": round(metrics.error_rate * 100, 1),
                "timeout_rate_percent": round(metrics.timeout_rate * 100, 1),
                "average_response_length": round(metrics.average_response_length, 0)
            },
            "difficulty_performance": {
                "easy_accuracy_percent": round(metrics.easy_accuracy * 100, 1) if metrics.easy_accuracy else 0,
                "medium_accuracy_percent": round(metrics.medium_accuracy * 100, 1) if metrics.medium_accuracy else 0,
                "hard_accuracy_percent": round(metrics.hard_accuracy * 100, 1) if metrics.hard_accuracy else 0
            },
            "position_type_performance": {
                "tactical_accuracy_percent": round(metrics.tactical_accuracy * 100, 1) if metrics.tactical_accuracy else 0,
                "strategic_accuracy_percent": round(metrics.strategic_accuracy * 100, 1) if metrics.strategic_accuracy else 0,
                "endgame_accuracy_percent": round(metrics.endgame_accuracy * 100, 1) if metrics.endgame_accuracy else 0
            },
            "response_quality_indicators": {
                "responses_with_explanation_percent": round(metrics.responses_with_explanation * 100, 1),
                "responses_with_reasoning_percent": round(metrics.responses_with_reasoning * 100, 1)
            },
            "recommendations": self._generate_recommendations(metrics)
        }

        return report

    def _generate_recommendations(self, metrics: SimpleEvaluationMetrics) -> List[str]:
        """Generate actionable recommendations based on metrics."""
        recommendations = []

        # Move accuracy recommendations
        if metrics.move_accuracy < 0.6:
            recommendations.append("Low move accuracy - consider improving move generation or data quality")
        elif metrics.move_accuracy < 0.8:
            recommendations.append("Moderate move accuracy - review tactical training data")

        # Stockfish agreement recommendations
        if metrics.stockfish_agreement_rate < 0.5:
            recommendations.append("Low Stockfish agreement - verify move quality and engine configuration")
        elif metrics.stockfish_agreement_rate < 0.7:
            recommendations.append("Moderate Stockfish agreement - consider position difficulty calibration")

        # Response quality recommendations
        if metrics.responses_with_explanation < 0.5:
            recommendations.append("Low explanation rate - improve prompt engineering for explanations")

        if metrics.responses_with_reasoning < 0.5:
            recommendations.append("Low reasoning rate - enhance training data for strategic thinking")

        # Difficulty-based recommendations
        if metrics.hard_accuracy < metrics.medium_accuracy - 0.2:
            recommendations.append("Performance drops significantly on hard positions - focus training on complex positions")

        if metrics.easy_accuracy < 0.9:
            recommendations.append("Even easy positions have issues - review basic move generation")

        # Overall recommendations
        if metrics.overall_score < 60:
            recommendations.append("Overall performance below threshold - comprehensive review recommended")
        elif metrics.overall_score < 80:
            recommendations.append("Good performance with room for improvement - targeted optimization suggested")

        return recommendations


def create_simple_evaluator(stockfish_path: Optional[str] = None) -> SimpleChessEvaluator:
    """Factory function to create a simple evaluator."""
    return SimpleChessEvaluator(stockfish_path)


def evaluate_chess_expert(positions_file: str, responses_file: str,
                         output_file: str, stockfish_path: Optional[str] = None) -> Dict[str, Any]:
    """Convenient function to evaluate chess expert performance."""
    logger.info(f"Evaluating chess expert: {positions_file} vs {responses_file}")

    # Load positions data
    try:
        with open(positions_file, 'r') as f:
            positions_data = [json.loads(line) for line in f if line.strip()]
    except Exception as e:
        logger.error(f"Failed to load positions: {e}")
        return {"error": f"Failed to load positions: {e}"}

    # Load expert responses
    try:
        with open(responses_file, 'r') as f:
            expert_responses = [line.strip() for line in f if line.strip()]
    except Exception as e:
        logger.error(f"Failed to load responses: {e}")
        return {"error": f"Failed to load responses: {e}"}

    if len(positions_data) != len(expert_responses):
        logger.warning(f"Mismatch: {len(positions_data)} positions vs {len(expert_responses)} responses")

    # Create evaluator and run evaluation
    evaluator = create_simple_evaluator(stockfish_path)
    metrics = evaluator.evaluate_positions(positions_data, expert_responses)

    # Generate report
    report = evaluator.generate_evaluation_report(metrics, positions_data)

    # Save report
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)

    logger.info(f"Evaluation report saved to {output_file}")
    logger.info(f"Overall score: {metrics.overall_score:.1f}/100")

    return report
