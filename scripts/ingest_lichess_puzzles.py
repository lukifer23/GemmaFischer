#!/usr/bin/env python3
"""Ingest Lichess puzzles with rating filter and convert to Q&A JSONL.

Input: CSV at data/raw/lichess_puzzles.csv (columns include FEN, Moves, Rating, Themes)
Output: data/datasets/lichess_puzzles_1000_2000.jsonl

For each puzzle (rating in [1000, 2000]):
- Build a tutor-style question with FEN
- Provide a brief explanation derived from themes if available
- Include the final UCI move as the last line of the assistant answer
"""

import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Optional


OUT = Path("data/datasets/lichess_puzzles_1000_2000.jsonl")


def _motifs_from_themes(themes: str) -> List[str]:
    if not themes:
        return []
    parts = [t.strip().lower() for t in themes.split()] if ' ' in themes and ',' not in themes else [t.strip().lower() for t in themes.split(',')]
    # Normalize common motifs
    norm = []
    for t in parts:
        t = t.replace('-', '_')
        if t in ("pin", "fork", "skewer", "discovered_attack", "double_attack", "deflection", "decoy", "clearance", "zwischenzug", "sacrifice", "back_rank_mate"):
            norm.append(t)
    return list(dict.fromkeys(norm))


def _explanation_from_motifs(motifs: List[str]) -> str:
    if not motifs:
        return "Find the best tactical continuation."
    primary = motifs[0]
    templates = {
        "pin": "It exploits a pin to win material.",
        "fork": "It creates a fork to attack two targets at once.",
        "skewer": "It uses a skewer to expose a more valuable piece.",
        "discovered_attack": "It unleashes a discovered attack for gain.",
        "double_attack": "It creates a double attack to overload defense.",
        "deflection": "It deflects a defender from a key square.",
        "decoy": "It decoys a piece onto a vulnerable square.",
        "clearance": "It clears a line for a decisive tactic.",
        "zwischenzug": "It inserts an intermediate move to change the outcome.",
        "sacrifice": "A temporary sacrifice opens lines to win material or mate.",
        "back_rank_mate": "It threatens back-rank mate, forcing material gain.",
    }
    return templates.get(primary, "It converts a tactical motif into advantage.")


def convert_row(row: Dict[str, str]) -> Dict[str, str]:
    fen = row.get("FEN") or row.get("fen") or ""
    moves_str = (row.get("Moves") or row.get("moves") or "").strip()
    rating_str = row.get("Rating") or row.get("rating") or "0"
    themes = row.get("Themes") or row.get("themes") or ""
    try:
        rating = int(rating_str)
    except ValueError:
        return None
    if not fen or not moves_str:
        return None
    moves = moves_str.split()
    best = moves[0]
    motifs = _motifs_from_themes(themes)
    rationale = _explanation_from_motifs(motifs)

    question = (
        f"FEN: {fen}\n"
        "Side to move is given by the FEN. Find the best tactical move."
    )
    # Ensure last line is the UCI move for strict extraction
    answer = (
        f"{rationale}\n"
        f"Best move: {best}"
    )

    return {
        "text": f"Question: {question}\nAnswer: {answer}",
        "conversations": [
            {"role": "system", "content": "You are a chess tactics tutor."},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
        "topic": "tactics",
        "difficulty": "intermediate" if rating < 1600 else "advanced",
        "rating": rating,
        "fen": fen,
        "label_move": best,
        "solution": moves,
        "num_moves": len(moves),
        "motifs": motifs,
    }


def _find_puzzles_path() -> Optional[Path]:
    # Allow override
    env = os.environ.get("PUZZLES_PATH")
    if env:
        p = Path(env)
        return p if p.exists() else None
    # Common locations, CSV first then ZST
    # Prefer nested .zst, then nested .csv, then root .zst, then root .csv, and skip zero-byte files
    candidates = [
        Path("data/raw/lichess/puzzles/lichess_puzzles.csv.zst"),
        Path("data/raw/lichess/puzzles/lichess_puzzles.csv"),
        Path("data/raw/lichess_puzzles.csv.zst"),
        Path("data/raw/lichess_puzzles.csv"),
    ]
    for c in candidates:
        if c.exists():
            try:
                if c.stat().st_size > 0:
                    return c
            except Exception:
                return c
    return None


def _ingest_from_csv(csv_path: Path, out_path: Path, limit: int) -> None:
    kept = 0
    total = 0
    with csv_path.open("r", encoding="utf-8", errors="ignore") as fin, out_path.open("w", encoding="utf-8") as fout:
        reader = csv.DictReader(fin)
        for row in reader:
            total += 1
            try:
                rating = int(row.get("Rating") or row.get("rating") or 0)
            except ValueError:
                continue
            if rating < 1000 or rating > 2000:
                continue
            obj = convert_row(row)
            if obj is None:
                continue
            kept += 1
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            if limit and kept >= limit:
                break
    print(f"Saved {kept}/{total} filtered puzzles to {out_path}")


def _ingest_from_zst(zst_path: Path, out_path: Path, limit: int) -> None:
    try:
        import zstandard as zstd
    except ImportError:
        print("zstandard not installed. Run: pip install zstandard")
        return
    kept = 0
    total = 0
    dctx = zstd.ZstdDecompressor()
    import io
    with zst_path.open("rb") as fh, out_path.open("w", encoding="utf-8") as fout:
        with dctx.stream_reader(fh) as reader:
            text_stream = io.TextIOWrapper(reader, encoding="utf-8", errors="ignore")
            csv_reader = csv.DictReader(text_stream)
            for row in csv_reader:
                total += 1
                try:
                    rating = int(row.get("Rating") or row.get("rating") or 0)
                except ValueError:
                    continue
                if rating < 1000 or rating > 2000:
                    continue
                obj = convert_row(row)
                if obj is None:
                    continue
                kept += 1
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                if limit and kept >= limit:
                    break
    print(f"Saved {kept}/{total} filtered puzzles to {out_path}")


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    limit = int(os.environ.get("PUZZLES_LIMIT", "50000"))

    p = _find_puzzles_path()
    if not p:
        print("Puzzle CSV/ZST not found. Set PUZZLES_PATH or place file under data/raw/.")
        return
    if p.suffix == ".zst":
        _ingest_from_zst(p, OUT, limit)
    else:
        _ingest_from_csv(p, OUT, limit)

    print("Ingestion completed.")


if __name__ == "__main__":
    main()


