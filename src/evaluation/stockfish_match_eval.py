#!/usr/bin/env python3
"""Evaluate model move agreement with Stockfish on a set of FEN positions.

Inputs: JSONL or JSON containing either a ``fen`` field or a question string
with ``FEN: ...`` embedded. Outputs a console summary and (optionally) a JSON
report when ``--out`` is provided.
"""

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Optional, Dict, Any, List
import sys

import chess
import chess.engine

# Ensure project root on sys.path when running directly
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.inference import ChessGemmaInference
from src.inference.chess_engine import ChessEngineManager
from src.inference.uci_utils import extract_first_legal_move


def parse_uci_from_text(text: str, board: chess.Board) -> Optional[chess.Move]:
    return extract_first_legal_move(text, board)


FEN_PATTERN = re.compile(r"FEN:\s*([^\s]+)", re.IGNORECASE)


def extract_fen(record: Dict[str, Any]) -> Optional[str]:
    fen = record.get("fen") or record.get("FEN")
    if fen:
        return fen
    for key in ("question", "prompt", "text", "context"):
        value = record.get(key)
        if isinstance(value, str):
            match = FEN_PATTERN.search(value)
            if match:
                return match.group(1)
    return None


def load_fens(path: Path, limit: Optional[int]) -> List[str]:
    fens: List[str] = []
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                fen = extract_fen(obj)
                if fen:
                    fens.append(fen)
                    if limit and len(fens) >= limit:
                        break
    else:
        payload = json.load(path.open("r", encoding="utf-8"))
        if isinstance(payload, dict) and "queries" in payload:
            records = payload["queries"]
        elif isinstance(payload, list):
            records = payload
        else:
            raise ValueError("Unsupported JSON structure; expected list or {\"queries\": ...}")
        for obj in records:
            if not isinstance(obj, dict):
                continue
            fen = extract_fen(obj)
            if fen:
                fens.append(fen)
                if limit and len(fens) >= limit:
                    break
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
    model_loaded = inference.load_model()
    if not model_loaded:
        if os.environ.get("CHESSGEMMA_SKIP_MODEL_LOAD", "0") not in ("0", "false", "False"):
            print("⚠️  Model loading skipped (CHESSGEMMA_SKIP_MODEL_LOAD). Continuing with offline evaluation.")
        else:
            print("Could not load model.")
            return

    # Warm-up only when the model is actually available to avoid repeated
    # warnings in offline benchmarking scenarios.
    if model_loaded:
        print("⚙️  Priming inference pipeline (warm-up)...")
        warmup_prompt = (
            "FEN: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1\n"
            "Move:\nMode: Engine\nGenerate the best move in UCI format (e.g., e2e4). Respond with only the move."
        )
        inference.set_active_adapter("uci")
        warm_start = time.time()
        inference.generate_response(warmup_prompt, mode="engine", max_new_tokens=6)
        print(f"   Warm-up completed in {time.time() - warm_start:.2f}s (excluded from measurements)")

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

            # Stockfish best move (fallback if helper is unavailable)
            if hasattr(engine, "get_best_move"):
                sf_move = engine.get_best_move(board, depth=args.depth, time_limit_ms=0)
            else:
                limit = chess.engine.Limit(depth=args.depth, time=0.0)
                with engine._engine_lock:
                    backend = getattr(engine, "engine", None)
                    if backend is None:
                        raise RuntimeError("Stockfish backend is not initialized")
                    result = backend.play(board, limit)
                sf_move = result.move if result else None

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
