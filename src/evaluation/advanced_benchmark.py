#!/usr/bin/env python3
"""
Advanced Benchmarking and Performance Regression Detection System

Comprehensive evaluation suite with automated benchmarking, performance regression
detection, statistical analysis, and continuous monitoring capabilities.
"""

import json
import time
import statistics
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import threading
import logging
from collections import defaultdict
import scipy.stats as stats

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics."""
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    response_time_mean: float = 0.0
    response_time_std: float = 0.0
    response_time_p95: float = 0.0
    throughput: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    quality_score: float = 0.0
    consistency_score: float = 0.0


@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""
    model_name: str
    timestamp: datetime
    metrics: PerformanceMetrics
    test_cases: List[Dict[str, Any]] = field(default_factory=list)
    raw_results: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RegressionAnalysis:
    """Analysis of performance regression."""
    is_regression: bool
    confidence_level: float
    regression_magnitude: float
    affected_metrics: List[str]
    statistical_significance: float
    recommended_actions: List[str]


class ChessGemmaBenchmarker:
    """Advanced benchmarking system with regression detection."""

    def __init__(self, results_dir: str = "benchmark_results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.baseline_results: Dict[str, BenchmarkResult] = {}
        self.historical_results: List[BenchmarkResult] = []
        self.lock = threading.Lock()

        # Statistical thresholds
        self.regression_threshold = 0.05  # 5% degradation
        self.significance_level = 0.05    # 95% confidence
        self.min_samples = 10             # Minimum samples for statistical analysis

        logger.info("📊 Advanced ChessGemma Benchmarker initialized")

    def run_comprehensive_benchmark(self, model_name: str, inference_func: Callable,
                                  test_dataset: List[Dict[str, Any]],
                                  metadata: Optional[Dict[str, Any]] = None) -> BenchmarkResult:
        """Run comprehensive benchmark with statistical analysis."""

        logger.info(f"🏃 Running comprehensive benchmark for {model_name}")

        start_time = time.time()
        results = []
        response_times = []

        # Run benchmark on test dataset
        for i, test_case in enumerate(test_dataset):
            try:
                case_start = time.time()

                # Run inference
                result = inference_func(test_case)

                case_time = time.time() - case_start
                response_times.append(case_time)

                # Evaluate result
                evaluation = self._evaluate_result(test_case, result)

                results.append({
                    'test_case': test_case,
                    'result': result,
                    'response_time': case_time,
                    'evaluation': evaluation,
                    'case_number': i
                })

                if (i + 1) % 50 == 0:
                    logger.info(f"   Processed {i + 1}/{len(test_dataset)} test cases")

            except Exception as e:
                logger.error(f"Benchmark error on case {i}: {e}")
                results.append({
                    'test_case': test_case,
                    'error': str(e),
                    'response_time': 0.0,
                    'evaluation': {'accuracy': 0.0, 'quality': 0.0},
                    'case_number': i
                })

        # Calculate comprehensive metrics
        metrics = self._calculate_metrics(results, response_times)

        # Create benchmark result
        benchmark_result = BenchmarkResult(
            model_name=model_name,
            timestamp=datetime.now(),
            metrics=metrics,
            test_cases=results,
            raw_results={
                'total_cases': len(test_dataset),
                'successful_cases': len([r for r in results if 'error' not in r]),
                'failed_cases': len([r for r in results if 'error' in r]),
                'response_times': response_times
            },
            metadata=metadata or {}
        )

        # Save results
        self._save_benchmark_result(benchmark_result)

        # Check for regressions
        regression_analysis = self._analyze_regression(benchmark_result)
        if regression_analysis.is_regression:
            logger.warning("⚠️  Performance regression detected!"            self._log_regression_alert(benchmark_result, regression_analysis)

        total_time = time.time() - start_time
        logger.info(".2f"
        return benchmark_result

    def _evaluate_result(self, test_case: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate the quality and accuracy of a result."""
        evaluation = {
            'accuracy': 0.0,
            'quality': 0.0,
            'correctness': False,
            'reasoning_quality': 0.0,
            'move_quality': 0.0
        }

        try:
            # Extract expected and actual results
            expected = test_case.get('expected', test_case.get('response', ''))
            actual = result.get('response', '')

            # Basic accuracy check
            if expected and actual:
                evaluation['accuracy'] = self._calculate_text_similarity(expected, actual)
                evaluation['correctness'] = evaluation['accuracy'] > 0.8

            # Quality assessment
            evaluation['quality'] = self._assess_response_quality(actual, test_case)

            # Chess-specific evaluation
            if 'fen' in test_case:
                evaluation['move_quality'] = self._evaluate_move_quality(test_case, result)

            # Reasoning quality for CoT examples
            if test_case.get('task') == 'director_qa':
                evaluation['reasoning_quality'] = self._evaluate_reasoning_quality(actual)

        except Exception as e:
            logger.warning(f"Evaluation error: {e}")

        return evaluation

    def _calculate_metrics(self, results: List[Dict[str, Any]],
                          response_times: List[float]) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics."""
        metrics = PerformanceMetrics()

        if not results:
            return metrics

        # Accuracy metrics
        accuracies = [r['evaluation']['accuracy'] for r in results if 'evaluation' in r]
        if accuracies:
            metrics.accuracy = statistics.mean(accuracies)
            metrics.precision = self._calculate_precision_recall(accuracies)[0]
            metrics.recall = self._calculate_precision_recall(accuracies)[1]
            metrics.f1_score = 2 * (metrics.precision * metrics.recall) / (metrics.precision + metrics.recall) if (metrics.precision + metrics.recall) > 0 else 0

        # Response time metrics
        if response_times:
            metrics.response_time_mean = statistics.mean(response_times)
            metrics.response_time_std = statistics.stdev(response_times) if len(response_times) > 1 else 0
            metrics.response_time_p95 = np.percentile(response_times, 95)

        # Quality metrics
        qualities = [r['evaluation']['quality'] for r in results if 'evaluation' in r]
        if qualities:
            metrics.quality_score = statistics.mean(qualities)
            metrics.consistency_score = 1.0 - (statistics.stdev(qualities) if len(qualities) > 1 else 0)

        # Throughput
        total_time = sum(response_times)
        metrics.throughput = len(results) / total_time if total_time > 0 else 0

        return metrics

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity score."""
        if not text1 or not text2:
            return 0.0

        # Simple word-based similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union) if union else 0.0

    def _assess_response_quality(self, response: str, test_case: Dict[str, Any]) -> float:
        """Assess overall response quality."""
        if not response:
            return 0.0

        quality_score = 0.0
        max_score = 0.0

        # Length appropriateness
        expected_length = len(test_case.get('response', ''))
        actual_length = len(response)

        if expected_length > 0:
            length_ratio = min(actual_length / expected_length, expected_length / actual_length)
            quality_score += length_ratio * 0.3
            max_score += 0.3

        # Content richness
        word_count = len(response.split())
        if word_count > 10:
            quality_score += 0.2
        max_score += 0.2

        # Chess-specific content
        chess_terms = ['pawn', 'knight', 'bishop', 'rook', 'queen', 'king', 'check', 'mate', 'move', 'position']
        chess_term_count = sum(1 for term in chess_terms if term in response.lower())
        quality_score += min(chess_term_count / 3, 1.0) * 0.5
        max_score += 0.5

        return quality_score / max_score if max_score > 0 else 0.0

    def _evaluate_move_quality(self, test_case: Dict[str, Any], result: Dict[str, Any]) -> float:
        """Evaluate chess move quality."""
        # Placeholder for chess engine integration
        # This would use Stockfish or similar to evaluate move quality
        return 0.5  # Neutral score for now

    def _evaluate_reasoning_quality(self, response: str) -> float:
        """Evaluate reasoning quality in CoT responses."""
        if not response:
            return 0.0

        reasoning_score = 0.0

        # Check for step-by-step reasoning
        step_indicators = ['first', 'next', 'then', 'after', 'finally', 'therefore', 'because', 'so']
        step_count = sum(1 for indicator in step_indicators if indicator in response.lower())
        reasoning_score += min(step_count / 3, 1.0) * 0.4

        # Check for chess concepts
        chess_concepts = ['center', 'development', 'safety', 'structure', 'advantage', 'control']
        concept_count = sum(1 for concept in chess_concepts if concept in response.lower())
        reasoning_score += min(concept_count / 4, 1.0) * 0.4

        # Length and coherence
        word_count = len(response.split())
        if word_count > 50:
            reasoning_score += 0.2

        return reasoning_score

    def _calculate_precision_recall(self, accuracies: List[float]) -> Tuple[float, float]:
        """Calculate precision and recall from accuracies."""
        if not accuracies:
            return 0.0, 0.0

        # Simple approximation using accuracy distribution
        threshold = 0.8  # Correctness threshold
        correct_predictions = sum(1 for acc in accuracies if acc >= threshold)
        total_predictions = len(accuracies)

        precision = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        recall = precision  # Simplified for this context

        return precision, recall

    def _analyze_regression(self, current_result: BenchmarkResult) -> RegressionAnalysis:
        """Analyze if current results show performance regression."""
        analysis = RegressionAnalysis(
            is_regression=False,
            confidence_level=0.0,
            regression_magnitude=0.0,
            affected_metrics=[],
            statistical_significance=1.0,
            recommended_actions=[]
        )

        if not self.baseline_results or current_result.model_name not in self.baseline_results:
            return analysis

        baseline = self.baseline_results[current_result.model_name]

        # Compare key metrics
        metrics_to_check = ['accuracy', 'response_time_mean', 'quality_score']

        for metric_name in metrics_to_check:
            current_value = getattr(current_result.metrics, metric_name)
            baseline_value = getattr(baseline.metrics, metric_name)

            if baseline_value > 0:
                change = (current_value - baseline_value) / baseline_value

                # Check for degradation
                if metric_name in ['accuracy', 'quality_score'] and change < -self.regression_threshold:
                    analysis.is_regression = True
                    analysis.affected_metrics.append(metric_name)
                    analysis.regression_magnitude = max(analysis.regression_magnitude, abs(change))

                # Check for performance degradation
                elif metric_name == 'response_time_mean' and change > self.regression_threshold:
                    analysis.is_regression = True
                    analysis.affected_metrics.append(metric_name)
                    analysis.regression_magnitude = max(analysis.regression_magnitude, change)

        # Statistical significance test
        if len(self.historical_results) >= self.min_samples:
            # Perform t-test against historical results
            historical_values = [getattr(r.metrics, 'accuracy') for r in self.historical_results[-self.min_samples:]]
            if historical_values:
                t_stat, p_value = stats.ttest_1samp(historical_values, current_result.metrics.accuracy)
                analysis.statistical_significance = p_value
                analysis.confidence_level = 1.0 - p_value

        # Generate recommended actions
        if analysis.is_regression:
            analysis.recommended_actions = self._generate_regression_actions(analysis)

        return analysis

    def _generate_regression_actions(self, analysis: RegressionAnalysis) -> List[str]:
        """Generate recommended actions for regression mitigation."""
        actions = []

        if 'accuracy' in analysis.affected_metrics:
            actions.extend([
                "Review recent model changes and roll back if necessary",
                "Check training data quality and distribution",
                "Validate model architecture consistency"
            ])

        if 'response_time_mean' in analysis.affected_metrics:
            actions.extend([
                "Profile model inference performance",
                "Check for memory leaks or resource contention",
                "Consider model optimization or quantization"
            ])

        if 'quality_score' in analysis.affected_metrics:
            actions.extend([
                "Review response generation parameters",
                "Check prompt engineering and template consistency",
                "Validate evaluation metrics calibration"
            ])

        actions.append("Run additional benchmarks to confirm regression")
        actions.append("Consider updating baseline if regression is expected")

        return actions

    def _save_benchmark_result(self, result: BenchmarkResult):
        """Save benchmark result to disk."""
        filename = f"benchmark_{result.model_name}_{result.timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.results_dir / filename

        data = {
            'model_name': result.model_name,
            'timestamp': result.timestamp.isoformat(),
            'metrics': {
                'accuracy': result.metrics.accuracy,
                'precision': result.metrics.precision,
                'recall': result.metrics.recall,
                'f1_score': result.metrics.f1_score,
                'response_time_mean': result.metrics.response_time_mean,
                'response_time_std': result.metrics.response_time_std,
                'response_time_p95': result.metrics.response_time_p95,
                'throughput': result.metrics.throughput,
                'quality_score': result.metrics.quality_score,
                'consistency_score': result.metrics.consistency_score
            },
            'raw_results': result.raw_results,
            'metadata': result.metadata
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)

        logger.info(f"💾 Benchmark result saved to {filepath}")

    def _log_regression_alert(self, result: BenchmarkResult, analysis: RegressionAnalysis):
        """Log regression alert with details."""
        alert_msg = ".1f"".2f"f"""
⚠️  PERFORMANCE REGRESSION ALERT ⚠️

Model: {result.model_name}
Timestamp: {result.timestamp}
Confidence: {analysis.confidence_level:.1%}
Magnitude: {analysis.regression_magnitude:.1%}
Affected Metrics: {', '.join(analysis.affected_metrics)}

Recommended Actions:
{chr(10).join(f"- {action}" for action in analysis.recommended_actions)}
"""
        logger.warning(alert_msg)

        # Save regression report
        regression_file = self.results_dir / f"regression_{result.model_name}_{result.timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        regression_data = {
            'alert_time': datetime.now().isoformat(),
            'model_name': result.model_name,
            'is_regression': analysis.is_regression,
            'confidence_level': analysis.confidence_level,
            'regression_magnitude': analysis.regression_magnitude,
            'affected_metrics': analysis.affected_metrics,
            'statistical_significance': analysis.statistical_significance,
            'recommended_actions': analysis.recommended_actions,
            'current_metrics': {
                'accuracy': result.metrics.accuracy,
                'response_time': result.metrics.response_time_mean,
                'quality_score': result.metrics.quality_score
            }
        }

        with open(regression_file, 'w') as f:
            json.dump(regression_data, f, indent=2)

    def load_baseline_results(self, baseline_file: Optional[str] = None):
        """Load baseline results for regression comparison."""
        if baseline_file:
            baseline_path = Path(baseline_file)
        else:
            # Find latest baseline file
            baseline_files = list(self.results_dir.glob("baseline_*.json"))
            baseline_path = max(baseline_files, key=lambda x: x.stat().st_mtime) if baseline_files else None

        if baseline_path and baseline_path.exists():
            with open(baseline_path, 'r') as f:
                baseline_data = json.load(f)

            # Convert to BenchmarkResult objects
            for model_name, result_data in baseline_data.items():
                # Recreate metrics object
                metrics_data = result_data['metrics']
                metrics = PerformanceMetrics(**metrics_data)

                result = BenchmarkResult(
                    model_name=model_name,
                    timestamp=datetime.fromisoformat(result_data['timestamp']),
                    metrics=metrics,
                    raw_results=result_data.get('raw_results', {}),
                    metadata=result_data.get('metadata', {})
                )

                self.baseline_results[model_name] = result

            logger.info(f"📊 Loaded baseline results for {len(self.baseline_results)} models")
        else:
            logger.warning("⚠️  No baseline results found")

    def save_baseline_results(self, filename: Optional[str] = None):
        """Save current results as baseline."""
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"baseline_{timestamp}.json"

        baseline_path = self.results_dir / filename

        baseline_data = {}
        for model_name, result in self.baseline_results.items():
            baseline_data[model_name] = {
                'timestamp': result.timestamp.isoformat(),
                'metrics': {
                    'accuracy': result.metrics.accuracy,
                    'precision': result.metrics.precision,
                    'recall': result.metrics.recall,
                    'f1_score': result.metrics.f1_score,
                    'response_time_mean': result.metrics.response_time_mean,
                    'response_time_std': result.metrics.response_time_std,
                    'response_time_p95': result.metrics.response_time_p95,
                    'throughput': result.metrics.throughput,
                    'quality_score': result.metrics.quality_score,
                    'consistency_score': result.metrics.consistency_score
                },
                'raw_results': result.raw_results,
                'metadata': result.metadata
            }

        with open(baseline_path, 'w') as f:
            json.dump(baseline_data, f, indent=2)

        logger.info(f"💾 Baseline results saved to {baseline_path}")

    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        report = {
            'report_time': datetime.now().isoformat(),
            'models_evaluated': list(self.baseline_results.keys()),
            'total_benchmarks': len(self.historical_results),
            'regression_alerts': sum(1 for r in self.historical_results if hasattr(r, 'regression_analysis') and r.regression_analysis.is_regression),
            'performance_summary': {}
        }

        # Summarize performance by model
        for model_name, baseline in self.baseline_results.items():
            recent_results = [r for r in self.historical_results if r.model_name == model_name][-10:]  # Last 10 results

            if recent_results:
                avg_accuracy = statistics.mean([r.metrics.accuracy for r in recent_results])
                avg_response_time = statistics.mean([r.metrics.response_time_mean for r in recent_results])

                report['performance_summary'][model_name] = {
                    'baseline_accuracy': baseline.metrics.accuracy,
                    'current_avg_accuracy': avg_accuracy,
                    'accuracy_trend': avg_accuracy - baseline.metrics.accuracy,
                    'baseline_response_time': baseline.metrics.response_time_mean,
                    'current_avg_response_time': avg_response_time,
                    'response_time_trend': avg_response_time - baseline.metrics.response_time_mean,
                    'samples': len(recent_results)
                }

        return report


# Convenience functions
def create_benchmark_dataset(size: int = 100) -> List[Dict[str, Any]]:
    """Create a benchmark dataset for testing."""
    dataset = []

    # Sample test cases
    test_cases = [
        {
            "prompt": "FEN: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1\nWhat is the best move for white?",
            "expected": "e2e4",
            "task": "engine_uci"
        },
        {
            "prompt": "FEN: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1\nExplain the opening principles.",
            "expected": "Control the center",
            "task": "tutor_explain"
        },
        {
            "prompt": "FEN: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1\nWhat should white do strategically?",
            "expected": "Develop pieces and control center",
            "task": "director_qa"
        }
    ]

    # Generate dataset
    for i in range(size):
        case = test_cases[i % len(test_cases)].copy()
        case['id'] = i
        dataset.append(case)

    return dataset


def run_quick_benchmark(model_name: str, inference_function: Callable, dataset_size: int = 50) -> BenchmarkResult:
    """Run a quick benchmark with minimal setup."""
    benchmarker = ChessGemmaBenchmarker()
    dataset = create_benchmark_dataset(dataset_size)

    return benchmarker.run_comprehensive_benchmark(model_name, inference_function, dataset)
