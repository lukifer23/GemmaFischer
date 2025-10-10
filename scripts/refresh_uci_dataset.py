#!/usr/bin/env python3
"""Re-generate the standardized UCI dataset with deeper Stockfish labels.

For each entry in the source JSONL file we:
  * Recompute the best move using Stockfish at the requested depth/time limit
  * Record the top-k moves (if available)
  * Normalize the response to the single UCI move string

The refreshed dataset is written to ``data/standardized/standardized_uci_expert_v2.jsonl``.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

import chess

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = PROJECT_ROOT / "data" / "standardized" / "standardized_uci_expert.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "data" / "standardized" / "standardized_uci_expert_v2.jsonl"


def iter_jsonl(path: Path) -> Iterator[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def save_jsonl(records: Iterable[Dict[str, object]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False))
            fh.write("\n")


def refresh_dataset(depth: int, top_k: int, time_limit_ms: int, limit: Optional[int]) -> int:
    from src.inference.chess_engine import ChessEngineManager

    total = 0
    refreshed: List[Dict[str, object]] = []

    with ChessEngineManager(debug=False) as engine:
        for idx, row in enumerate(iter_jsonl(SOURCE_PATH), start=1):
            if limit is not None and idx > limit:
                break

            meta = dict(row.get("meta") or {})
            fen = meta.get("fen")
            if not isinstance(fen, str):
                continue

            board = chess.Board(fen)
            best_move = engine.get_best_move(board, depth=depth, time_limit_ms=time_limit_ms)
            if best_move is None:
                continue

            top_moves = engine.get_top_moves(board, depth=depth, top_k=top_k)
            top_moves_uci = [mv.uci() for mv in top_moves if mv is not None]

            refreshed_meta = {
                **meta,
                "best_move": best_move.uci(),
                "stockfish_depth": depth,
                "stockfish_time_limit_ms": time_limit_ms,
                "top_moves": top_moves_uci,
                "source": meta.get("source", "standardized_uci_expert_v2"),
            }

            refreshed.append({
                "task": row.get("task", "engine_uci"),
                "prompt": row.get("prompt"),
                "response": best_move.uci(),
                "meta": refreshed_meta,
            })

            total = len(refreshed)
            if idx % 100 == 0:
                print(f"Processed {idx} entries (kept {total})...")

    save_jsonl(refreshed, OUTPUT_PATH)
    return total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh UCI dataset with new Stockfish labels")
    parser.add_argument("--depth", type=int, default=14, help="Stockfish search depth")
    parser.add_argument("--top-k", type=int, default=3, help="Number of principal variations to store")
    parser.add_argument("--time-limit-ms", type=int, default=1000, help="Time limit per position (milliseconds)")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap for quick runs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not SOURCE_PATH.exists():
        raise FileNotFoundError(f"Source dataset not found: {SOURCE_PATH}")

    start = time.time()
    total = refresh_dataset(
        depth=args.depth,
        top_k=args.top_k,
        time_limit_ms=args.time_limit_ms,
        limit=args.limit,
    )
    elapsed = time.time() - start

    print(f"✅ Refreshed dataset written to {OUTPUT_PATH} ({total} samples)")
    print(f"⏱️  Elapsed time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
