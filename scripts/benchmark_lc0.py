#!/usr/bin/env python3
"""Benchmark hybrid LC0 + LLM analysis pipeline."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from statistics import mean

import chess

import sys
# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.inference.inference import get_inference_instance
from src.inference.hybrid_engine import HybridEngine


DEFAULT_POSITIONS = Path('data/validation/eval_mixed_positions_200.jsonl')
REPORT_PATH = Path('reports/hybrid_benchmark.json')


def load_positions(path: Path, limit: int) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Evaluation positions file not found: {path}")
    positions: list[dict[str, str]] = []
    with path.open('r', encoding='utf-8') as fh:
        for line in fh:
            if not line.strip():
                continue
            record = json.loads(line)
            if 'fen' in record:
                positions.append(record)
            if len(positions) >= limit:
                break
    return positions


def benchmark_hybrid_engine(num_samples: int, output: Path) -> None:
    inference = get_inference_instance()
    inference.load_model()
    hybrid = HybridEngine()

    positions = load_positions(DEFAULT_POSITIONS, num_samples)

    samples = []
    for entry in positions:
        fen = entry['fen']
        description = entry.get('id') or entry.get('name') or entry.get('category') or 'position'

        print(f"Analyzing {description}: {fen}")
        start = time.perf_counter()
        result = inference.analyze_with_engine(fen)
        total_time = time.perf_counter() - start

        board = chess.Board(fen)
        legal = False
        if result.get('best_move'):
            try:
                move = chess.Move.from_uci(result['best_move'])
                legal = move in board.legal_moves
            except ValueError:
                legal = False

        samples.append({
            'fen': fen,
            'best_move': result.get('best_move'),
            'engine': result.get('engine'),
            'engine_time': result.get('engine_time'),
            'total_time': total_time,
            'fallback_used': result.get('fallback_used'),
            'evaluation_cp': result.get('evaluation_cp'),
            'mate_in': result.get('mate_in'),
            'principal_variation': result.get('principal_variation'),
            'legal': legal,
        })

    engine_times = [s['engine_time'] for s in samples if s['engine_time'] is not None]
    total_times = [s['total_time'] for s in samples]
    legal_rate = sum(1 for s in samples if s['legal']) / max(len(samples), 1)

    summary = {
        'samples': len(samples),
        'avg_engine_time': mean(engine_times) if engine_times else None,
        'avg_total_time': mean(total_times) if total_times else None,
        'legal_move_rate': legal_rate,
        'fallback_count': sum(1 for s in samples if s['fallback_used']),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', encoding='utf-8') as fh:
        json.dump({'summary': summary, 'samples': samples}, fh, indent=2)

    print('\nHybrid Benchmark Summary')
    print('-------------------------')
    print(f"Positions evaluated : {summary['samples']}")
    if summary['avg_engine_time'] is not None:
        print(f"Average engine time: {summary['avg_engine_time']:.3f}s")
    print(f"Average total time : {summary['avg_total_time']:.3f}s")
    print(f"Legal move rate   : {legal_rate * 100:.1f}%")
    print(f"Fallback usages   : {summary['fallback_count']}")
    print(f"Report written to : {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description='Benchmark LC0 hybrid analysis pipeline.')
    parser.add_argument('--num-samples', type=int, default=10, help='Number of positions to evaluate (default: 10)')
    parser.add_argument('--output', type=Path, default=REPORT_PATH, help='Where to write the JSON report.')
    args = parser.parse_args()

    benchmark_hybrid_engine(args.num_samples, args.output)


if __name__ == '__main__':
    main()
