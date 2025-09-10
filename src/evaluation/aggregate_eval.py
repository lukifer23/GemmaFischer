#!/usr/bin/env python3
"""Aggregate evaluation runner.

Runs:
- Move legality/syntax (sampled via chess_evaluation.py style metrics)
- Stockfish top-1 match on a mixed-position set
- Puzzle first-move accuracy (if puzzle file provided)

Outputs a JSON report and optional Markdown summary.
"""

import argparse
import json
from pathlib import Path

from src.evaluation.stockfish_match_eval import load_fens as _load_fens
from src.evaluation.puzzle_eval import load_puzzles as _load_puzzles
from src.evaluation.puzzle_eval import ChessGemmaInference, parse_first_uci
import chess


def eval_legality_syntax(sample_questions):
    # Simple relevance proxy: count chess terms
    terms = ['pawn', 'knight', 'bishop', 'rook', 'queen', 'king', 'check', 'mate', 'castle']
    from src.inference.inference import ChessGemmaInference
    inf = ChessGemmaInference()
    if not inf.load_model():
        return {'chess_relevance': 0.0}
    score = 0
    for q in sample_questions:
        out = inf.generate_response(q, max_new_tokens=128)
        text = out.get('response', '').lower()
        score += sum(1 for t in terms if t in text) > 0
    return {'chess_relevance': score / len(sample_questions) if sample_questions else 0.0}


def eval_stockfish_match(fens_file: Path, limit: int, depth: int):
    from src.evaluation.stockfish_match_eval import ChessGemmaInference, ChessEngineManager, parse_uci_from_text
    fens = _load_fens(fens_file, limit)
    if not fens:
        return {'top1_match': 0.0, 'count': 0}
    inf = ChessGemmaInference()
    if not inf.load_model():
        return {'top1_match': 0.0, 'count': 0}
    match = 0
    with ChessEngineManager() as engine:
        for fen in fens:
            board = chess.Board(fen)
            q = f"Position: {fen}\nMode: Engine\nGenerate the best move in UCI format (e.g., e2e4). Respond with only the move."
            gen = inf.generate_response(q, mode='engine', max_new_tokens=12)
            mv = parse_uci_from_text(gen.get('response', ''), board)
            sf = engine.get_best_move(board, depth=depth, time_limit_ms=0)
            if mv and sf and mv == sf.uci():
                match += 1
    return {'top1_match': match / len(fens), 'count': len(fens)}


def eval_puzzles(puzzles_file: Path, limit: int):
    puzzles = _load_puzzles(puzzles_file, limit)
    if not puzzles:
        return {'first_move_accuracy': 0.0, 'count': 0}
    inf = ChessGemmaInference()
    if not inf.load_model():
        return {'first_move_accuracy': 0.0, 'count': 0}
    ok = 0
    for p in puzzles:
        fen = p['fen']
        sol = p['solution']
        board = chess.Board(fen)
        q = f"Position: {fen}\nMode: Engine\nGenerate the best move in UCI format (e.g., e2e4). Respond with only the move."
        gen = inf.generate_response(q, mode='engine', max_new_tokens=12)
        mv = parse_first_uci(gen.get('response', ''), board)
        if mv == sol[0]:
            ok += 1
    return {'first_move_accuracy': ok / len(puzzles), 'count': len(puzzles)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fens', type=str, default='data/datasets/eval_mixed_positions_200.jsonl')
    ap.add_argument('--puzzles', type=str, default='data/datasets/lichess_puzzles_1000_2000.jsonl')
    ap.add_argument('--limit_fens', type=int, default=100)
    ap.add_argument('--limit_puzzles', type=int, default=200)
    ap.add_argument('--depth', type=int, default=8)
    ap.add_argument('--out', type=str, default='eval_report.json')
    args = ap.parse_args()

    # Legality/relevance quick proxy on a few questions
    sample_q = [
        'What is the best opening move for White?',
        'Explain castling in chess.',
        'What is a fork and how to create one?'
    ]
    legality = eval_legality_syntax(sample_q)
    sf_match = eval_stockfish_match(Path(args.fens), args.limit_fens, args.depth)
    puzzle = eval_puzzles(Path(args.puzzles), args.limit_puzzles)

    report = {
        'legality_proxy': legality,
        'stockfish_match': sf_match,
        'puzzle_accuracy': puzzle,
    }

    with Path(args.out).open('w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    print(f"Saved report to {args.out}")


if __name__ == '__main__':
    main()


