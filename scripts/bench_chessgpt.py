#!/usr/bin/env python3
"""
Benchmark the ChessGPT base model for latency, memory footprint, and move quality.

The script loads the smallest public ChessGPT checkpoint and runs it against a
subset of our evaluation positions.  Results are written to
`reports/chessgpt_benchmark.json` and printed to stdout.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import psutil
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import chess


DEFAULT_MODEL_PATH = Path("models/chessgpt-base-v1")
DEFAULT_POSITIONS = Path("data/validation/eval_mixed_positions_200.jsonl")
REPORT_PATH = Path("reports/chessgpt_benchmark.json")

UCI_PATTERN = re.compile(r"\b([a-h][1-8][a-h][1-8][qrbn]?)\b", re.IGNORECASE)


@dataclass
class SampleResult:
    fen: str
    expected_move: str
    predicted_move: Optional[str]
    legal: bool
    matched: bool
    latency_ms: float
    raw_output: str


@dataclass
class BenchmarkSummary:
    model_path: str
    device: str
    dtype: str
    total_samples: int
    evaluated_samples: int
    legal_predictions: int
    correct_predictions: int
    load_time_sec: float
    average_latency_ms: float
    latency_p95_ms: float
    latency_std_ms: float
    max_rss_gb: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark ChessGPT performance and accuracy.")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Local path to the ChessGPT checkpoint (default: models/chessgpt-base-v1)",
    )
    parser.add_argument(
        "--positions",
        type=Path,
        default=DEFAULT_POSITIONS,
        help="JSONL file containing FEN positions with expected moves.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=25,
        help="Number of positions to evaluate (default: 25).",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "mps"],
        default="cpu",
        help="Computation device. MPS support is experimental. (default: cpu)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=6,
        help="Maximum number of tokens to generate per query (default: 6).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPORT_PATH,
        help="Path to save benchmark results (default: reports/chessgpt_benchmark.json).",
    )
    return parser.parse_args()


def load_positions(path: Path, limit: int) -> List[Dict[str, Any]]:
    data: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            record = json.loads(line)
            if "fen" not in record or "expected_move" not in record:
                continue
            data.append(record)
            if len(data) >= limit:
                break
    return data


def extract_uci_move(text: str) -> Optional[str]:
    """Extract the first UCI-like token from generated text."""
    match = UCI_PATTERN.search(text)
    if match:
        return match.group(1).lower()
    return None


def format_prompt(fen: str) -> str:
    return (
        "You are a chess engine. Given the position in Forsyth-Edwards Notation, "
        "output the best move in UCI notation.\n"
        f"FEN: {fen}\n"
        "Best move (UCI):"
    )


def main() -> None:
    args = parse_args()
    if not args.model_path.exists():
        raise FileNotFoundError(f"Model path not found: {args.model_path}")
    if not args.positions.exists():
        raise FileNotFoundError(f"Positions file not found: {args.positions}")

    samples = load_positions(args.positions, args.num_samples)
    if not samples:
        raise RuntimeError(f"No valid samples found in {args.positions}")

    process = psutil.Process()
    start_mem = process.memory_info().rss

    load_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_mps = args.device == "mps" and torch.backends.mps.is_available()
    dtype = torch.float16 if use_mps else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=dtype)
    device = torch.device("mps" if use_mps else "cpu")
    model = model.to(device)

    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.eos_token_id = tokenizer.eos_token_id

    load_time = time.perf_counter() - load_start

    load_mem = process.memory_info().rss
    max_rss = max(start_mem, load_mem)

    results: List[SampleResult] = []
    latencies: List[float] = []

    for entry in samples:
        fen = entry["fen"]
        expected = entry["expected_move"].lower()
        board = chess.Board(fen)

        prompt = format_prompt(fen)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        gen_start = time.perf_counter()
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
            )
        latency = (time.perf_counter() - gen_start) * 1000.0
        latencies.append(latency)

        decoded = tokenizer.decode(output[0], skip_special_tokens=True)
        predicted = extract_uci_move(decoded)

        legal = False
        matched = False
        if predicted:
            try:
                move_obj = chess.Move.from_uci(predicted)
                legal = move_obj in board.legal_moves
                matched = predicted == expected
            except ValueError:
                legal = False
        else:
            predicted = None

        results.append(
            SampleResult(
                fen=fen,
                expected_move=expected,
                predicted_move=predicted,
                legal=legal,
                matched=matched,
                latency_ms=latency,
                raw_output=decoded.strip(),
            )
        )
        max_rss = max(max_rss, process.memory_info().rss)

    evaluated = len(results)
    legal_predictions = sum(1 for r in results if r.legal)
    correct_predictions = sum(1 for r in results if r.matched)
    avg_latency = statistics.mean(latencies) if latencies else math.nan
    latency_std = statistics.stdev(latencies) if len(latencies) > 1 else 0.0
    latency_p95 = statistics.quantiles(latencies, n=100)[94] if len(latencies) >= 20 else max(latencies, default=0.0)

    summary = BenchmarkSummary(
        model_path=str(args.model_path),
        device=str(device),
        dtype=str(dtype),
        total_samples=len(samples),
        evaluated_samples=evaluated,
        legal_predictions=legal_predictions,
        correct_predictions=correct_predictions,
        load_time_sec=load_time,
        average_latency_ms=avg_latency,
        latency_p95_ms=latency_p95,
        latency_std_ms=latency_std,
        max_rss_gb=max_rss / (1024 ** 3),
    )

    report_payload = {
        "summary": asdict(summary),
        "samples": [asdict(r) for r in results],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump(report_payload, fh, indent=2)

    accuracy = (correct_predictions / evaluated * 100.0) if evaluated else 0.0
    legality_rate = (legal_predictions / evaluated * 100.0) if evaluated else 0.0

    print("=== ChessGPT Benchmark ===")
    print(f"Model path        : {summary.model_path}")
    print(f"Device / dtype    : {summary.device} / {summary.dtype}")
    print(f"Samples evaluated : {evaluated}/{summary.total_samples}")
    print(f"Accuracy          : {accuracy:.1f}%")
    print(f"Legal move rate   : {legality_rate:.1f}%")
    print(f"Avg latency       : {summary.average_latency_ms:.1f} ms (p95 {summary.latency_p95_ms:.1f} ms)")
    print(f"Load time         : {summary.load_time_sec:.1f} s")
    print(f"Peak RSS          : {summary.max_rss_gb:.2f} GB")
    print(f"\nDetailed report saved to {args.output}")


if __name__ == "__main__":
    main()
