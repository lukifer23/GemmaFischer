#!/usr/bin/env python3
"""Validate and optionally repair chess instruction datasets.

Reads JSONL with either legacy {text} or instruct schema {task, prompt, response, meta}.
Performs:
 - FEN parsing and side-to-move sanity checks when present
 - UCI legality checks for engine/tutor samples
 - Tutor: enforce trailing 'Best move: <uci>' and legal against FEN
 - Optional Stockfish relabel when illegal/missing moves

Outputs a cleaned JSONL. Invalid samples can be dropped or repaired.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import chess


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='inp', required=True, help='Input JSONL')
    ap.add_argument('--out', dest='out', required=True, help='Output JSONL (cleaned)')
    ap.add_argument('--mode', choices=['uci', 'tutor', 'director'], default='uci')
    ap.add_argument('--relabel_with_stockfish', action='store_true', help='Use Stockfish to repair missing/illegal moves')
    ap.add_argument('--max', dest='limit', type=int, default=0)
    return ap.parse_args()


def extract_fen_from_text(txt: str) -> Optional[str]:
    if not txt:
        return None
    import re
    for line in txt.splitlines():
        if line.lower().startswith('fen:'):
            return line.split(':', 1)[1].strip()
    fen_like = re.compile(r"(?:[rnbqkpRNBQKP1-8]+/){7}[rnbqkpRNBQKP1-8]+\s[wb]\s(?:K?Q?k?q?|-)\s(?:[a-h][36]|-)\s\d+\s\d+")
    m = fen_like.search(txt)
    return m.group(0) if m else None


def extract_final_uci_from_tutor(resp: str) -> Optional[str]:
    if not resp:
        return None
    import re
    # Look for explicit "Best move: <uci>"
    m = re.search(r"Best move:\s*([a-h][1-8][a-h][1-8][qrbn]?)", resp, flags=re.IGNORECASE)
    if m:
        return m.group(1).lower()
    # Fallback: last UCI token in response
    tokens = re.findall(r"\b([a-h][1-8][a-h][1-8][qrbn]?)\b", resp.lower())
    return tokens[-1] if tokens else None


def get_stockfish_best(board: chess.Board) -> Optional[str]:
    try:
        from src.inference.chess_engine import ChessEngineManager
        with ChessEngineManager() as ce:
            mv = ce.get_best_move(board, depth=12, time_limit_ms=200)
            return mv.uci() if mv else None
    except Exception:
        return None


def process_line(obj: Dict[str, Any], mode: str, use_sf: bool) -> Optional[Dict[str, Any]]:
    task = (obj.get('task') or '').strip()
    prompt = obj.get('prompt')
    response = obj.get('response')

    # Quick skip for director
    if mode == 'director':
        return obj

    # Determine FEN context
    fen = None
    for src in (prompt, response):
        fen = extract_fen_from_text(src or '')
        if fen:
            break
    try:
        board = chess.Board(fen) if fen else None
    except Exception:
        return None

    # Skip positions that are already checkmate or stalemate
    if board is not None and (board.is_checkmate() or board.is_stalemate()):
        return None

    # Extract/validate move
    mv: Optional[str] = None
    if mode == 'uci':
        # Expect response to be a single UCI move
        mv = extract_final_uci_from_tutor(response or '')
    elif mode == 'tutor':
        mv = extract_final_uci_from_tutor(response or '')

    if board is not None and mv:
        try:
            move_obj = chess.Move.from_uci(mv)
            if move_obj not in board.legal_moves:
                mv = None
        except Exception:
            mv = None

    if (board is not None) and (mv is None) and use_sf:
        best = get_stockfish_best(board)
        if best:
            # repair response: ensure explicit final line for tutor, set response for uci
            if mode == 'tutor':
                fixed = (response or '').rstrip() + f"\n\nBest move: {best}"
                obj['response'] = fixed
            else:
                obj['response'] = best
            mv = best

    # If still no valid move but mode requires it, drop sample
    if (mode in ('uci', 'tutor')) and (board is not None) and (mv is None):
        return None

    return obj


def main():
    args = parse_args()
    src = Path(args.inp)
    outp = Path(args.out)
    count_in = 0
    count_out = 0

    mode = args.mode

    with src.open('r', encoding='utf-8') as fin, outp.open('w', encoding='utf-8') as fout:
        for line in fin:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            count_in += 1
            out = process_line(obj, mode, args.relabel_with_stockfish)
            if out is None:
                continue
            fout.write(json.dumps(out, ensure_ascii=False) + '\n')
            count_out += 1
            if args.limit and count_out >= args.limit:
                break

    print(f"Processed: {count_in} | Kept: {count_out} | Dropped: {count_in - count_out}")


if __name__ == '__main__':
    main()

