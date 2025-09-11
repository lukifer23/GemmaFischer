#!/usr/bin/env python3
"""
Build instruction-style expert datasets from existing sources.

Outputs (JSONL, instruction schema):
- data/formatted/uci_expert.jsonl        (task=engine_uci)
- data/formatted/tutor_expert.jsonl      (task=tutor_explain)
- data/formatted/director_expert.jsonl   (task=director_qa)

Input defaults:
- Puzzles for UCI/Tutor: data/datasets/lichess_puzzles_1000_2000.jsonl
- Director Q&A: data/datasets/cot_reasoning_examples.jsonl (if present)

Schema per line:
{ "task": str, "prompt": str, "response": str, "meta": {...} }
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any


def ensure_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def build_uci_from_puzzles(puzzles_path: Path, out_path: Path) -> int:
    ensure_dir(out_path)
    count = 0
    with puzzles_path.open('r', encoding='utf-8') as fin, out_path.open('w', encoding='utf-8') as fout:
        for line in fin:
            try:
                ex = json.loads(line)
                fen = ex.get('fen') or ''
                label = (ex.get('label_move') or '').strip().lower()
                if not fen or not label:
                    continue
                prompt = (
                    f"FEN: {fen}\n"
                    "Move:\n"
                    "Style: balanced\n"
                    "Mode: Engine\n"
                    "Generate the best move in UCI format (e.g., e2e4). Respond with only the move."
                )
                item = {
                    'task': 'engine_uci',
                    'prompt': prompt,
                    'response': label,
                    'meta': {
                        'fen': fen,
                        'source': puzzles_path.name,
                        'rating': ex.get('rating'),
                        'topic': ex.get('topic', 'tactics')
                    }
                }
                fout.write(json.dumps(item, ensure_ascii=False) + '\n')
                count += 1
            except Exception:
                continue
    return count


def build_tutor_from_puzzles(puzzles_path: Path, out_path: Path) -> int:
    ensure_dir(out_path)
    count = 0
    with puzzles_path.open('r', encoding='utf-8') as fin, out_path.open('w', encoding='utf-8') as fout:
        for line in fin:
            try:
                ex = json.loads(line)
                fen = ex.get('fen') or ''
                label = (ex.get('label_move') or '').strip().lower()
                sol = ex.get('solution') or []
                if not fen or not label:
                    continue
                prompt = (
                    f"FEN: {fen}\n"
                    "Question: Analyze this position step by step.\n"
                    "Style: balanced\n"
                    "Mode: Tutor\n\n"
                    "1. Evaluate the current position\n"
                    "2. Identify key threats and opportunities\n"
                    "3. Consider candidate moves\n"
                    "4. Choose the best move with reasoning\n\n"
                    "Respond with the best move in UCI format at the end."
                )
                explanation_lines = []
                if isinstance(sol, list) and sol:
                    explanation_lines.append("Tactical line: " + ' '.join(sol))
                response = "\n".join(explanation_lines + [f"Best move: {label}"]) if explanation_lines else f"Best move: {label}"
                item = {
                    'task': 'tutor_explain',
                    'prompt': prompt,
                    'response': response,
                    'meta': {
                        'fen': fen,
                        'source': puzzles_path.name,
                        'rating': ex.get('rating'),
                        'topic': ex.get('topic', 'tactics')
                    }
                }
                fout.write(json.dumps(item, ensure_ascii=False) + '\n')
                count += 1
            except Exception:
                continue
    return count


def build_director_from_cot(cot_path: Path, out_path: Path) -> int:
    ensure_dir(out_path)
    count = 0
    with cot_path.open('r', encoding='utf-8') as fin, out_path.open('w', encoding='utf-8') as fout:
        for line in fin:
            try:
                ex = json.loads(line)
                # Heuristics: prefer conversations; else split text by Question/Answer
                question = None
                answer = None
                if isinstance(ex.get('conversations'), list) and ex['conversations']:
                    user_msgs = [m.get('content','') for m in ex['conversations'] if m.get('role') == 'user']
                    asst_msgs = [m.get('content','') for m in ex['conversations'] if m.get('role') in ('assistant','model')]
                    if user_msgs and asst_msgs:
                        question = user_msgs[0]
                        answer = asst_msgs[0]
                if not question and isinstance(ex.get('text'), str):
                    txt = ex['text']
                    if 'Question:' in txt and 'Answer:' in txt:
                        q_part, a_part = txt.split('Answer:', 1)
                        question = q_part.replace('Question:', '').strip()
                        answer = a_part.strip()
                    else:
                        # Fallback: treat entire text as answer to a generic prompt
                        question = "Explain this chess concept clearly."
                        answer = txt.strip()
                if not question or not answer:
                    continue
                item = {
                    'task': 'director_qa',
                    'prompt': question,
                    'response': answer,
                    'meta': {'source': cot_path.name}
                }
                fout.write(json.dumps(item, ensure_ascii=False) + '\n')
                count += 1
            except Exception:
                continue
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--puzzles', type=Path, default=Path('data/datasets/lichess_puzzles_1000_2000.jsonl'))
    parser.add_argument('--cot', type=Path, default=Path('data/datasets/cot_reasoning_examples.jsonl'))
    parser.add_argument('--out_dir', type=Path, default=Path('data/formatted'))
    args = parser.parse_args()

    uci_out = args.out_dir / 'uci_expert.jsonl'
    tutor_out = args.out_dir / 'tutor_expert.jsonl'
    director_out = args.out_dir / 'director_expert.jsonl'

    uci_n = build_uci_from_puzzles(args.puzzles, uci_out)
    tutor_n = build_tutor_from_puzzles(args.puzzles, tutor_out)
    director_n = 0
    if args.cot.exists():
        director_n = build_director_from_cot(args.cot, director_out)

    print(f"engine_uci: {uci_n} -> {uci_out}")
    print(f"tutor_explain: {tutor_n} -> {tutor_out}")
    print(f"director_qa: {director_n} -> {director_out}")


if __name__ == '__main__':
    main()


