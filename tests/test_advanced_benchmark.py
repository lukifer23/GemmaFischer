"""Regression tests for the advanced benchmark module."""

from pathlib import Path
import sys


# Ensure the repository root (containing the ``src`` package) is importable.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.advanced_benchmark import (  # noqa: E402  (import after path setup)
    BenchmarkResult,
    ChessGemmaBenchmarker,
)


def test_run_comprehensive_benchmark_returns_result(tmp_path):
    """Ensure running the benchmark on a small dataset succeeds."""

    benchmarker = ChessGemmaBenchmarker(results_dir=tmp_path)

    dataset = [
        {
            "prompt": "FEN: start position\nWhat is the best move for white?",
            "expected": "e2e4",
            "task": "engine_uci",
        },
        {
            "prompt": "Explain basic opening principles.",
            "expected": "Control the center",
            "task": "tutor_explain",
        },
        {
            "prompt": "What strategic plan should white follow?",
            "expected": "Develop pieces",
            "task": "director_qa",
        },
    ]

    def inference_func(test_case):
        return {"response": test_case.get("expected", "")}

    result = benchmarker.run_comprehensive_benchmark(
        "test-model",
        inference_func,
        dataset,
    )

    assert isinstance(result, BenchmarkResult)
    assert result.model_name == "test-model"
    assert len(result.test_cases) == len(dataset)
