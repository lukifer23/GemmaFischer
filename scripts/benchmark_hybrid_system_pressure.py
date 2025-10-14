#!/usr/bin/env python3
"""
Hybrid LLM+LC0 System Pressure Benchmark (Apple Silicon / MPS)

Measures end-to-end latency and system pressure while the LLM (Gemma) and LC0
are both active. Collects per-request timings and periodic system metrics
including CPU and memory. Optionally samples MPS memory if available.

Outputs a JSON report with summary stats and a metrics time series.
"""

from __future__ import annotations

import argparse
import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from dataclasses import dataclass, asdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

import psutil
import chess
import signal

# Ensure project src is importable
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.inference import get_inference_instance


@dataclass
class RequestResult:
    kind: str  # 'engine' or 'tutor'
    fen: str
    total_time: float
    engine_time: Optional[float] = None
    best_move: Optional[str] = None
    legal: Optional[bool] = None
    fallback_used: Optional[bool] = None
    evaluation_cp: Optional[int] = None


def load_positions(path: Path, limit: int) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Positions file not found: {path}")
    fens: List[str] = []
    with path.open('r', encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                fen = rec.get('fen')
                if fen:
                    fens.append(fen)
            except json.JSONDecodeError:
                # tolerate plain FEN lines if present
                if ' ' in line and '/' in line:
                    fens.append(line)
            if len(fens) >= limit:
                break
    if not fens:
        raise ValueError(f"No FEN positions found in {path}")
    return fens


def sample_mps_memory_mb() -> Dict[str, Optional[float]]:
    try:
        import torch
        if not torch.backends.mps.is_available():
            return {'mps_current_mb': None, 'mps_driver_mb': None}
        current = None
        driver = None
        try:
            current = float(torch.mps.current_allocated_memory()) / (1024 * 1024)
        except Exception:
            current = None
        try:
            driver = float(torch.mps.driver_allocated_memory()) / (1024 * 1024)
        except Exception:
            driver = None
        return {'mps_current_mb': current, 'mps_driver_mb': driver}
    except Exception:
        return {'mps_current_mb': None, 'mps_driver_mb': None}


def start_metrics_sampler(interval_sec: float, stop_event: threading.Event) -> Tuple[threading.Thread, List[Dict[str, Any]]]:
    process = psutil.Process()
    # Prime CPU percent
    try:
        process.cpu_percent(None)
    except Exception:
        pass

    samples: List[Dict[str, Any]] = []

    def _loop() -> None:
        while not stop_event.is_set():
            ts = time.time()
            try:
                mem = process.memory_info().rss / (1024 * 1024)
                cpu = process.cpu_percent(interval=None)
                mps = sample_mps_memory_mb()
                samples.append({
                    't': ts,
                    'rss_mb': mem,
                    'cpu_percent': cpu,
                    'mps_current_mb': mps['mps_current_mb'],
                    'mps_driver_mb': mps['mps_driver_mb'],
                })
            except Exception:
                samples.append({'t': ts, 'error': 'sampling_failed'})
            time.sleep(max(0.01, interval_sec))

    th = threading.Thread(target=_loop, daemon=True)
    th.start()
    return th, samples


def run_engine_request(inf, fen: str) -> RequestResult:
    start = time.perf_counter()
    payload = inf.analyze_with_engine(fen)
    total = time.perf_counter() - start
    best = payload.get('best_move')
    legal = False
    if best:
        try:
            board = chess.Board(fen)
            legal = chess.Move.from_uci(best) in board.legal_moves
        except Exception:
            legal = False
    return RequestResult(
        kind='engine',
        fen=fen,
        total_time=total,
        engine_time=payload.get('engine_time'),
        best_move=best,
        legal=legal,
        fallback_used=payload.get('fallback_used'),
        evaluation_cp=payload.get('evaluation_cp'),
    )


def run_tutor_request(inf, fen: str) -> RequestResult:
    start = time.perf_counter()
    res = inf.generate_response(
        question=f"FEN: {fen}\nExplain briefly and end with a best move.",
        context=f"Current position: {fen}",
        mode='tutor',
        max_new_tokens=160,
    )
    total = time.perf_counter() - start
    return RequestResult(kind='tutor', fen=fen, total_time=total)


def percentile(values: List[float], p: float) -> Optional[float]:
    if not values:
        return None
    values_sorted = sorted(values)
    k = (len(values_sorted) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(values_sorted) - 1)
    if f == c:
        return values_sorted[int(k)]
    d0 = values_sorted[f] * (c - k)
    d1 = values_sorted[c] * (k - f)
    return d0 + d1


def summarize(results: List[RequestResult]) -> Dict[str, Any]:
    engine = [r for r in results if r.kind == 'engine']
    tutor = [r for r in results if r.kind == 'tutor']

    def _lat(ns: List[RequestResult]) -> Dict[str, Any]:
        vals = [r.total_time for r in ns]
        return {
            'count': len(vals),
            'avg': mean(vals) if vals else None,
            'p50': percentile(vals, 50) if vals else None,
            'p95': percentile(vals, 95) if vals else None,
            'p99': percentile(vals, 99) if vals else None,
        }

    engine_times = [r.engine_time for r in engine if isinstance(r.engine_time, (int, float))]
    fallback_count = sum(1 for r in engine if r.fallback_used)
    legal_rate = sum(1 for r in engine if r.legal) / max(1, len(engine))

    return {
        'overall_latency': _lat(results),
        'engine_latency': _lat(engine),
        'tutor_latency': _lat(tutor),
        'engine_internal_time_avg': mean(engine_times) if engine_times else None,
        'engine_fallback_count': fallback_count,
        'engine_legal_rate': legal_rate,
    }


def run_benchmark(
    fen_file: Path,
    num_requests: int,
    concurrency: int,
    engine_share: float,
    sample_interval: float,
    output_path: Path,
) -> Dict[str, Any]:
    fens = load_positions(fen_file, limit=max(10, num_requests))
    inf = get_inference_instance()

    # Warm-up: load model and LC0
    inf.load_model()
    _ = inf.analyze_with_engine(random.choice(fens))

    stop_event = threading.Event()
    sampler, samples = start_metrics_sampler(sample_interval, stop_event)

    results: List[RequestResult] = []
    start_ts = time.time()

    # Allow graceful interruption: on SIGINT, stop sampling and continue to summary
    interrupted = {'flag': False}
    def _sigint_handler(signum, frame):
        interrupted['flag'] = True
        print("\n[Signal] Interrupt received, finalizing...", flush=True)
    prev_handler = signal.signal(signal.SIGINT, _sigint_handler)

    def _task(fen: str, use_engine: bool) -> RequestResult:
        return run_engine_request(inf, fen) if use_engine else run_tutor_request(inf, fen)

    completed = 0
    last_print = time.time()
    smoothed_rate: Optional[float] = None  # EMA to stabilize ETA
    def maybe_progress(force: bool = False) -> None:
        nonlocal last_print, smoothed_rate
        now = time.time()
        if force or (now - last_print) >= 0.5:
            elapsed = now - start_ts
            inst_rate = completed / max(1e-6, elapsed)
            if smoothed_rate is None:
                smoothed_rate = inst_rate
            else:
                alpha = 0.3
                smoothed_rate = alpha * inst_rate + (1 - alpha) * smoothed_rate
            remaining = max(0, num_requests - completed)
            eta_sec = remaining / max(1e-6, smoothed_rate) if smoothed_rate > 0 else None
            if eta_sec is not None and eta_sec < 3e6:
                finish_ts = datetime.fromtimestamp(now + eta_sec).strftime('%H:%M:%S')
                eta_text = f" ~{eta_sec:.1f}s (finish {finish_ts})"
            else:
                eta_text = " (warming up…)"
            print(f"[Progress] {completed}/{num_requests} done | rate {smoothed_rate:.2f}/s | elapsed {elapsed:.1f}s{eta_text}", flush=True)
            last_print = now

    try:
        with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
            futures = []
            for i in range(num_requests):
                if interrupted['flag']:
                    break
                fen = fens[i % len(fens)]
                use_engine = (random.random() < engine_share)
                futures.append(pool.submit(_task, fen, use_engine))
            for fut in as_completed(futures):
                try:
                    results.append(fut.result())
                except Exception:
                    results.append(RequestResult(kind='error', fen='', total_time=0.0))
                finally:
                    completed += 1
                    maybe_progress(False)
                if interrupted['flag']:
                    break
    finally:
        stop_event.set()
        sampler.join(timeout=2.0)
        signal.signal(signal.SIGINT, prev_handler)

    maybe_progress(True)
    end_ts = time.time()

    summary = summarize(results)

    report = {
        'started_at': start_ts,
        'ended_at': end_ts,
        'duration_sec': end_ts - start_ts,
        'interrupted': interrupted['flag'],
        'config': {
            'num_requests': num_requests,
            'concurrency': concurrency,
            'engine_share': engine_share,
            'sample_interval_sec': sample_interval,
            'fen_file': str(fen_file),
        },
        'summary': summary,
        'system_samples': samples,
        'results': [asdict(r) for r in results],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as fh:
        json.dump(report, fh, indent=2)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description='Hybrid LLM+LC0 system pressure benchmark (MPS).')
    parser.add_argument('--fen-file', type=Path, default=Path('data/validation/eval_mixed_positions_200.jsonl'))
    parser.add_argument('--num-requests', type=int, default=40)
    parser.add_argument('--concurrency', type=int, default=8)
    parser.add_argument('--engine-share', type=float, default=0.6, help='Fraction [0,1] of engine (LC0) requests')
    parser.add_argument('--sample-interval', type=float, default=0.2, help='Seconds between system metric samples')
    parser.add_argument('--output', type=Path, default=Path('reports/system_pressure_benchmark.json'))
    args = parser.parse_args()

    report = run_benchmark(
        fen_file=args.fen_file,
        num_requests=args.num_requests,
        concurrency=args.concurrency,
        engine_share=max(0.0, min(1.0, args.engine_share)),
        sample_interval=max(0.02, args.sample_interval),
        output_path=args.output,
    )

    print('\nHybrid System Pressure Benchmark Summary')
    print('--------------------------------------')
    print(f"Requests         : {report['config']['num_requests']}")
    print(f"Concurrency      : {report['config']['concurrency']}")
    s = report['summary']
    eng = s['engine_latency']; tut = s['tutor_latency']
    print(f"Engine p50/p95   : {eng['p50']:.3f}s / {eng['p95']:.3f}s" if eng['p50'] is not None else "Engine p50/p95   : n/a")
    print(f"Tutor p50/p95    : {tut['p50']:.3f}s / {tut['p95']:.3f}s" if tut['p50'] is not None else "Tutor p50/p95    : n/a")
    print(f"Engine legal rate: {s['engine_legal_rate']*100:.1f}%")
    print(f"Fallback used    : {s['engine_fallback_count']}")
    print(f"Report written to: {args.output}")


if __name__ == '__main__':
    main()


