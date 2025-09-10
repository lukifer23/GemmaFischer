#!/usr/bin/env python3
"""Refine dataset with filters, tagging, and standardized formats.

Features:
- Filters: max length, chess-term presence
- Tagging: difficulty buckets (heuristic), topic labels (tactics/strategy/endgame/openings)
- Output: JSONL with both `text` and `conversations`
"""

import json
import re
from pathlib import Path
from typing import Dict, Any


INPUT_FILE = Path("data/finetune/chess_finetune_full.jsonl")
OUTPUT_FILE = Path("data/finetune/chess_finetune_refined.jsonl")

CHESS_TERMS = {
    "pawn", "knight", "bishop", "rook", "queen", "king",
    "check", "checkmate", "castle", "castling", "fork", "pin", "skewer",
    "e4", "d4", "Nf3", "O-O", "position", "advantage", "endgame", "opening",
}


def _has_chess_terms(text: str) -> bool:
    tl = text.lower()
    return any(term.lower() in tl for term in CHESS_TERMS)


def _topic_label(question: str) -> str:
    ql = question.lower()
    if any(k in ql for k in ["e4", "d4", "opening", "sicilian", "french", "italian"]):
        return "openings"
    if any(k in ql for k in ["fork", "pin", "skewer", "tactic", "mate", "checkmate"]):
        return "tactics"
    if any(k in ql for k in ["endgame", "rook endgame", "pawn endgame"]):
        return "endgames"
    return "strategy"


def _difficulty_bucket(question: str) -> str:
    # Simple heuristic based on length/terms
    qlen = len(question)
    if qlen < 60:
        return "beginner"
    if qlen < 140:
        return "intermediate"
    return "advanced"


def refine_line(obj: Dict[str, Any]) -> Dict[str, Any]:
    text = obj.get("text", "")
    if not text:
        return None

    # Expect format: "Question: <q>\nAnswer: <a>"
    parts = text.split("\n", 1)
    if len(parts) != 2:
        return None
    q_raw = parts[0].replace("Question:", "").strip()
    a_raw = parts[1].replace("Answer:", "").strip()

    # Filters
    if len(q_raw) > 500:
        return None
    if not _has_chess_terms(q_raw + " " + a_raw):
        return None

    topic = _topic_label(q_raw)
    difficulty = _difficulty_bucket(q_raw)

    conversations = [
        {"role": "system", "content": "You are a chess tutor and engine."},
        {"role": "user", "content": q_raw},
        {"role": "assistant", "content": a_raw},
    ]

    return {
        "text": text,
        "conversations": conversations,
        "topic": topic,
        "difficulty": difficulty,
    }


def main():
    if not INPUT_FILE.exists():
        print(f"Input file not found: {INPUT_FILE}")
        return

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    total = 0
    with INPUT_FILE.open("r", encoding="utf-8") as fin, OUTPUT_FILE.open("w", encoding="utf-8") as fout:
        for line in fin:
            total += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            out = refine_line(obj)
            if out is None:
                continue
            kept += 1
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f"Refined dataset saved to {OUTPUT_FILE} (kept {kept}/{total})")


if __name__ == "__main__":
    main()
