#!/usr/bin/env python3
"""Evaluate model move agreement with Stockfish on a set of FEN positions.

Inputs: JSONL with objects containing { "fen": <FEN> }
Outputs: prints summary and optional JSON report when --out is provided.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Optional, Dict, Any, List
import sys

import chess

# Ensure project root on sys.path when running directly
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.inference import ChessGemmaInference
from src.inference.chess_engine import ChessEngineManager
from src.inference.uci_utils import extract_first_legal_move


def parse_uci_from_text(text: str, board: chess.Board) -> Optional[chess.Move]:
    return extract_first_legal_move(text, board)


def load_fens(path: Path, limit: Optional[int]) -> List[str]:
    fens: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                fen = obj.get("fen") or obj.get("FEN")
                if fen:
                    fens.append(fen)
                    if limit and len(fens) >= limit:
                        break
            except json.JSONDecodeError:
                continue
    return fens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="JSONL file containing {fen}")
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    src = Path(args.file)
    if not src.exists():
        print(f"Input not found: {src}")
        return

    fens = load_fens(src, args.limit)
    if not fens:
        print("No FENs found in input file.")
        return

    inference = ChessGemmaInference()
    if not inference.load_model():
        print("Could not load model.")
        return

    results: List[Dict[str, Any]] = []
    match = 0
    legal = 0
    invalid = 0
    latencies: List[float] = []

    with ChessEngineManager() as engine:
        for i, fen in enumerate(fens, 1):
            board = chess.Board(fen)
            # Model move (engine mode)
            q = f"FEN: {fen}\nMove:\nMode: Engine\nGenerate the best move in UCI format (e.g., e2e4). Respond with only the move."
            t0 = time.time()
            gen = inference.generate_response(q, mode="engine", max_new_tokens=12)
            latencies.append(time.time() - t0)
            model_text = gen.get("response", "")
            model_move = parse_uci_from_text(model_text, board)

            # Stockfish best move
            sf_move = engine.get_best_move(board, depth=args.depth, time_limit_ms=0)

            if model_move is not None:
                legal += 1
            else:
                invalid += 1

            agree = (model_move == sf_move) if (model_move and sf_move) else False
            if agree:
                match += 1

            results.append({
                "fen": fen,
                "model": model_move.uci() if model_move else None,
                "stockfish": sf_move.uci() if sf_move else None,
                "agree": agree,
            })

            if i % 10 == 0:
                print(f"Processed {i}/{len(fens)}")

    total = len(results)
    rate = match / total if total else 0.0
    legal_rate = legal / total if total else 0.0
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    print(f"\nStockfish match (top-1) on {total} positions: {rate:.3f}")
    print(f"Legal output rate: {legal_rate:.3f} | Invalid: {invalid}")
    print(f"Avg latency: {avg_latency:.3f}s")

    if args.out:
        outp = Path(args.out)
        with outp.open("w", encoding="utf-8") as f:
            json.dump({
                "rate": rate,
                "legal_rate": legal_rate,
                "avg_latency_sec": avg_latency,
                "results": results
            }, f, indent=2)
        print(f"Saved report to {outp}")


if __name__ == "__main__":
    main()
