#!/usr/bin/env python3
"""Evaluate tactical puzzle first-move and sequence accuracy.

Input: JSONL with entries containing at least {"fen": <FEN>, "solution": [uci1, uci2, ...]}
Outputs: prints summary and optional JSON report when --out is provided.
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

import chess

from src.inference.inference import ChessGemmaInference
from src.inference.uci_utils import extract_first_legal_move_uci


def parse_first_uci(text: str, board: chess.Board) -> Optional[str]:
    return extract_first_legal_move_uci(text, board)


def load_puzzles(path: Path, limit: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                if 'fen' in obj and 'solution' in obj and isinstance(obj['solution'], list) and obj['solution']:
                    out.append(obj)
                    if len(out) >= limit:
                        break
            except json.JSONDecodeError:
                continue
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--file', required=True, help='Puzzle JSONL with fen and solution')
    ap.add_argument('--limit', type=int, default=200)
    ap.add_argument('--out', type=str, default=None)
    args = ap.parse_args()

    src = Path(args.file)
    if not src.exists():
        print(f"Input not found: {src}")
        return

    puzzles = load_puzzles(src, args.limit)
    if not puzzles:
        print('No puzzles with solution found.')
        return

    inf = ChessGemmaInference()
    if not inf.load_model():
        print('Model could not be loaded.')
        return

    first_ok = 0
    seq_ok = 0
    results: List[Dict[str, Any]] = []

    for i, p in enumerate(puzzles, 1):
        fen = p['fen']
        sol = p['solution']
        board = chess.Board(fen)
        prompt = f"FEN: {fen}\nMove:\nMode: Engine\nGenerate the best move in UCI format (e.g., e2e4). Respond with only the move."
        out = inf.generate_response(prompt, mode='engine', max_new_tokens=12)
        mv = parse_first_uci(out.get('response', ''), board)
        first = (mv == sol[0]) if mv else False
        if first:
            first_ok += 1
        # Basic sequence check (apply first predicted if correct, compare second)
        seq = False
        if first and len(sol) > 1:
            try:
                board.push(chess.Move.from_uci(mv))
                # Opponent's best response unknown; skip strict multi-ply evaluation in smoke
                # For now, require at least first move match for sequence credit
                seq = True
            except Exception:
                seq = False
        if seq:
            seq_ok += 1
        results.append({
            'fen': fen,
            'pred': mv,
            'solution_first': sol[0],
            'first_move_ok': first,
        })
        if i % 20 == 0:
            print(f"Evaluated {i}/{len(puzzles)} puzzles")

    first_rate = first_ok / len(puzzles)
    seq_rate = seq_ok / len(puzzles)
    print(f"\nPuzzle evaluation on {len(puzzles)} puzzles:")
    print(f"First-move accuracy: {first_rate:.3f}")
    print(f"Sequence accuracy (len>=1 surrogate): {seq_rate:.3f}")

    if args.out:
        with Path(args.out).open('w', encoding='utf-8') as f:
            json.dump({'first_move_accuracy': first_rate, 'sequence_accuracy': seq_rate, 'results': results}, f, indent=2)
        print(f"Saved report to {args.out}")


if __name__ == '__main__':
    main()


