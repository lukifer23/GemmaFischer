#!/usr/bin/env python3
"""
Comprehensive evaluation suite runner

Runs curated evaluation suite with MoE routing enabled/disabled
Provides detailed metrics by category and expert performance
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import re
import statistics
import argparse

# Add src to path for imports
import sys
sys.path.append('src')

from inference.inference import ChessGemmaInference

@dataclass
class EvaluationResult:
    """Result for a single evaluation test case."""
    test_id: str
    category: str
    question: str
    expected_expert: str
    expected_format: str
    response: str = ""
    confidence: float = 0.0
    generation_time: float = 0.0
    routed_expert: str = ""
    moe_used: bool = False
    format_score: float = 0.0
    expert_score: float = 0.0

@dataclass
class EvaluationSuiteResults:
    """Aggregated results for the entire evaluation suite."""
    total_tests: int = 0
    categories: Dict[str, Dict[str, Any]] = field(default_factory=lambda: defaultdict(dict))
    experts: Dict[str, Dict[str, Any]] = field(default_factory=lambda: defaultdict(dict))
    format_accuracy: Dict[str, float] = field(default_factory=dict)
    routing_accuracy: float = 0.0
    total_time: float = 0.0
    avg_generation_time: float = 0.0
    avg_confidence: float = 0.0

class EvaluationSuiteRunner:
    """Comprehensive evaluation suite runner with MoE support."""

    def __init__(self, model_path: str = "models/google-gemma-3-270m"):
        self.model_path = model_path
        self.inference = None

    def initialize_inference(self) -> bool:
        """Initialize the inference system."""
        try:
            self.inference = ChessGemmaInference(model_path=self.model_path)
            return self.inference.load_model()
        except Exception as e:
            print(f"❌ Failed to initialize inference: {e}")
            return False

    def validate_uci_move(self, text: str) -> bool:
        """Validate if text contains a valid UCI move."""
        # Basic UCI move pattern: e2e4, Nf3, O-O, e7e8=Q, etc.
        uci_pattern = r'\b[a-h][1-8][a-h][1-8][qrbn]?\b'
        matches = re.findall(uci_pattern, text.lower())
        return len(matches) > 0 and any(self._is_valid_uci_move(move) for move in matches)

    def _is_valid_uci_move(self, move: str) -> bool:
        """Check if a string is a valid UCI move format."""
        if not move or len(move) < 4:
            return False

        # Basic format check
        if not re.match(r'^[a-h][1-8][a-h][1-8][qrbn]?/?$', move.lower()):
            return False

        # Check if from/to squares are different
        from_square = move[:2]
        to_square = move[2:4]
        return from_square != to_square

    def validate_response_format(self, response: str, expected_format: str) -> float:
        """Validate response matches expected format."""
        if not response or not response.strip():
            return 0.0

        response_lower = response.lower().strip()

        if expected_format == "uci_move_only":
            # Must contain exactly one valid UCI move and little else
            moves = re.findall(r'\b[a-h][1-8][a-h][1-8][qrbn]?\b', response_lower)
            valid_moves = [m for m in moves if self._is_valid_uci_move(m)]
            # Should have exactly one valid move and response should be short
            return 1.0 if len(valid_moves) == 1 and len(response.strip()) < 20 else 0.0

        elif expected_format in ["step_by_step_analysis", "tactical_analysis", "analysis_with_move"]:
            # Must contain analysis and end with "Best move:" for analysis_with_move
            has_analysis = len(response.strip()) > 50  # Substantial analysis

            if expected_format == "analysis_with_move":
                has_best_move = "best move:" in response_lower
                has_uci = self.validate_uci_move(response) > 0
                return 1.0 if has_analysis and has_best_move and has_uci else 0.0
            else:
                return 1.0 if has_analysis else 0.0

        elif expected_format in ["strategic_explanation", "rules_explanation"]:
            # Must be explanatory text of reasonable length
            return 1.0 if len(response.strip()) > 100 else 0.0

        return 0.0

    def evaluate_single_test(self, test_case: Dict[str, Any], use_moe: bool = True) -> EvaluationResult:
        """Evaluate a single test case."""
        test_id = test_case["id"]
        category = test_case["category"]
        question = test_case["question"]
        expected_expert = test_case["expert"]
        expected_format = test_case["expected_format"]

        result = EvaluationResult(
            test_id=test_id,
            category=category,
            question=question,
            expected_expert=expected_expert,
            expected_format=expected_format
        )

        try:
            start_time = time.time()

            # Generate response
            if use_moe:
                # Use MoE routing (auto mode)
                response_data = self.inference.generate_response(
                    question=question,
                    mode="tutor"  # Use tutor as base for MoE routing
                )
                result.moe_used = True
            else:
                # Use specific expert
                response_data = self.inference.generate_response(
                    question=question,
                    mode=expected_expert
                )
                result.moe_used = False

            result.response = response_data.get("response", "")
            result.confidence = response_data.get("confidence", 0.0)
            result.generation_time = time.time() - start_time

            # Check which expert was actually used
            active_adapter = response_data.get("active_adapter")
            if active_adapter:
                if "uci" in active_adapter:
                    result.routed_expert = "uci"
                elif "tutor" in active_adapter:
                    result.routed_expert = "tutor"
                elif "director" in active_adapter:
                    result.routed_expert = "director"

            # Validate response format
            result.format_score = self.validate_response_format(result.response, expected_format)

            # Check expert routing accuracy
            result.expert_score = 1.0 if result.routed_expert == expected_expert else 0.0

        except Exception as e:
            print(f"❌ Error evaluating {test_id}: {e}")
            result.response = f"ERROR: {str(e)}"
            result.format_score = 0.0
            result.expert_score = 0.0

        return result

    def run_evaluation_suite(self, eval_file_path: str, use_moe: bool = True) -> EvaluationSuiteResults:
        """Run the complete evaluation suite."""
        print("🎯 Running Evaluation Suite")
        print(f"   File: {eval_file_path}")
        print(f"   MoE: {'Enabled' if use_moe else 'Disabled'}")
        print("=" * 60)

        # Load test cases
        test_cases = []
        with open(eval_file_path, 'r') as f:
            for line in f:
                if line.strip():
                    test_cases.append(json.loads(line))

        print(f"📋 Loaded {len(test_cases)} test cases")

        # Initialize results
        results = EvaluationSuiteResults()
        results.total_tests = len(test_cases)

        # Run evaluations
        individual_results = []
        for i, test_case in enumerate(test_cases, 1):
            if i % 10 == 0:
                print(f"   Progress: {i}/{len(test_cases)} tests completed")

            result = self.evaluate_single_test(test_case, use_moe)
            individual_results.append(result)

            # Update running totals
            results.total_time += result.generation_time
            results.avg_confidence += result.confidence

        # Calculate aggregate metrics
        results.avg_generation_time = results.total_time / len(test_cases)
        results.avg_confidence = results.avg_confidence / len(test_cases)

        # Calculate category metrics
        category_results = defaultdict(list)
        expert_results = defaultdict(list)

        for result in individual_results:
            category_results[result.category].append(result)
            expert_results[result.expected_expert].append(result)

        # Process category results
        for category, cat_results in category_results.items():
            format_scores = [r.format_score for r in cat_results]
            expert_scores = [r.expert_score for r in cat_results]

            results.categories[category] = {
                "count": len(cat_results),
                "format_accuracy": statistics.mean(format_scores),
                "expert_accuracy": statistics.mean(expert_scores),
                "avg_confidence": statistics.mean([r.confidence for r in cat_results]),
                "avg_time": statistics.mean([r.generation_time for r in cat_results])
            }

        # Process expert results
        for expert, exp_results in expert_results.items():
            format_scores = [r.format_score for r in exp_results]

            results.experts[expert] = {
                "count": len(exp_results),
                "format_accuracy": statistics.mean(format_scores),
                "avg_confidence": statistics.mean([r.confidence for r in exp_results]),
                "avg_time": statistics.mean([r.generation_time for r in exp_results])
            }

        # Calculate overall format accuracy
        all_format_scores = [r.format_score for r in individual_results]
        results.format_accuracy["overall"] = statistics.mean(all_format_scores)

        # Calculate routing accuracy (if MoE was used)
        if use_moe:
            routing_scores = [r.expert_score for r in individual_results]
            results.routing_accuracy = statistics.mean(routing_scores)

        return results

    def print_results(self, results: EvaluationSuiteResults, use_moe: bool):
        """Print formatted evaluation results."""
        print("\n" + "=" * 80)
        print("📊 EVALUATION SUITE RESULTS")
        print("=" * 80)

        print("\n🎯 Configuration:")
        print(f"   MoE Routing: {'Enabled' if use_moe else 'Disabled'}")
        print(f"   Total Tests: {results.total_tests}")

        print("\n⚡ Performance Metrics:")
        print(f"   Average Response Time: {results.avg_response_time:.3f}s")
        print(f"   Average Confidence: {results.avg_confidence:.3f}")
        print(f"   Cache Hit Rate: {results.cache_hit_rate:.2f}%")

        if use_moe:
            print(f"   MoE Ensemble Rate: {results.moe_ensemble_rate:.1f}%")

        print("\n📈 Category Performance:")
        print("Category".ljust(20) + "Count".ljust(8) + "Format Acc".ljust(12) + "Expert Acc".ljust(12) + "Avg Conf".ljust(10) + "Avg Time")
        print("-" * 90)

        for category, metrics in results.categories.items():
            print(f"{category:<20}{metrics['count']:<8}{metrics['format_accuracy']:.1%}{metrics['expert_accuracy']:.1%}{metrics['avg_confidence']:.2f}{metrics['avg_time']:.3f}")

        print("\n👥 Expert Performance:")
        print("Expert".ljust(12) + "Count".ljust(8) + "Format Acc".ljust(12) + "Avg Conf".ljust(10) + "Avg Time")
        print("-" * 65)

        for expert, metrics in results.experts.items():
            print(f"{expert:<12}{metrics['count']:<8}{metrics['format_accuracy']:.1%}{metrics['avg_confidence']:.2f}{metrics['avg_time']:.3f}")

        print(f"\n🎖️  Overall Format Accuracy:".ljust(30) + f"{results.overall_format_accuracy:.1%}")
        if use_moe:
            print(f"🎯 MoE Routing Accuracy:".ljust(30) + f"{results.moe_routing_accuracy:.1%}")

def main():
    parser = argparse.ArgumentParser(description="Run chess evaluation suite")
    parser.add_argument("--eval-file", default="data/validation/eval_suite.jsonl",
                       help="Path to evaluation suite JSONL file")
    parser.add_argument("--model-path", default="models/google-gemma-3-270m",
                       help="Path to model directory")
    parser.add_argument("--no-moe", action="store_true",
                       help="Disable MoE routing, use expert-specific mode")
    parser.add_argument("--output", help="Save results to JSON file")

    args = parser.parse_args()

    # Initialize runner
    runner = EvaluationSuiteRunner(model_path=args.model_path)

    if not runner.initialize_inference():
        print("❌ Failed to initialize inference system")
        sys.exit(1)

    # Run evaluation
    results = runner.run_evaluation_suite(args.eval_file, use_moe=not args.no_moe)

    # Print results
    runner.print_results(results, use_moe=not args.no_moe)

    # Save results if requested
    if args.output:
        output_data = {
            "configuration": {
                "moe_enabled": not args.no_moe,
                "eval_file": args.eval_file,
                "model_path": args.model_path
            },
            "results": {
                "total_tests": results.total_tests,
                "format_accuracy": results.format_accuracy,
                "routing_accuracy": results.routing_accuracy,
                "avg_generation_time": results.avg_generation_time,
                "avg_confidence": results.avg_confidence,
                "categories": dict(results.categories),
                "experts": dict(results.experts)
            }
        }

        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\n💾 Results saved to: {args.output}")

if __name__ == "__main__":
    main()
