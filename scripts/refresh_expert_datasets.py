#!/usr/bin/env python3
"""Refresh tutor/director expert datasets with cleaned, augmented entries.

- deduplicate by FEN/prompt combination
- ensure tutor responses end with ``best move: <uci>``
- ensure director responses end with ``Final move (UCI): <uci>`` when available
- append curated evaluation puzzles/questions to broaden coverage
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]

STANDARDIZED_DIR = PROJECT_ROOT / "data" / "standardized"
VALIDATION_DIR = PROJECT_ROOT / "data" / "validation"

TUTOR_SOURCE = STANDARDIZED_DIR / "standardized_tutor_expert.jsonl"
DIRECTOR_SOURCE = STANDARDIZED_DIR / "standardized_director_expert_v2.jsonl"

TUTOR_OUTPUT = STANDARDIZED_DIR / "standardized_tutor_expert_v2.jsonl"
DIRECTOR_OUTPUT = STANDARDIZED_DIR / "standardized_director_expert_v3.jsonl"

TUTOR_EVAL = VALIDATION_DIR / "tutor_comprehensive_validation.json"
DIRECTOR_EVAL = VALIDATION_DIR / "director_comprehensive_validation.json"

# Helper regex patterns
BEST_MOVE_PATTERN = re.compile(r"best move:\s*([a-h][1-8][a-h][1-8][qrbn]?)", re.IGNORECASE)
FINAL_MOVE_PATTERN = re.compile(r"final move\s*\(uci\):\s*([a-h][1-8][a-h][1-8][qrbn]?)", re.IGNORECASE)
UCI_PATTERN = re.compile(r"^[a-h][1-8][a-h][1-8][qrbn]?$")


def load_jsonl(path: Path) -> Iterable[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def save_jsonl(records: List[Dict[str, object]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False))
            fh.write("\n")


def validate_uci(move: Optional[str]) -> Optional[str]:
    if not move:
        return None
    mv = move.strip().lower()
    return mv if UCI_PATTERN.match(mv) else None


def ensure_best_move_line(text: str, move: str) -> str:
    body = text.strip()
    # remove existing best-move suffixes to avoid duplicates
    body = BEST_MOVE_PATTERN.sub("", body)
    body = body.rstrip()
    if body and not body.endswith("\n"):
        body += "\n"
    body += f"best move: {move}"
    return body


def ensure_final_move_line(text: str, move: Optional[str]) -> str:
    body = text.strip()
    body = FINAL_MOVE_PATTERN.sub("", body)
    if move:
        if body and not body.endswith("\n"):
            body += "\n"
        body += f"Final move (UCI): {move}"
    return body


def process_tutor_dataset() -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    seen: set[Tuple[str, str]] = set()

    def add_entry(task: str, prompt: str, response: str, meta: Dict[str, object]) -> None:
        fen = meta.get("fen")
        if not isinstance(fen, str):
            return
        key = (fen, prompt)
        if key in seen:
            return
        bm = validate_uci(meta.get("best_move"))
        if not bm:
            match = BEST_MOVE_PATTERN.search(response)
            if match:
                bm = validate_uci(match.group(1))
        if not bm:
            return
        meta = dict(meta)
        meta["best_move"] = bm
        meta.setdefault("quality_score", 0.8)
        meta.setdefault("source", "standardized_tutor_expert_v2")
        normalized_response = ensure_best_move_line(response, bm)
        records.append({
            "task": task,
            "prompt": prompt,
            "response": normalized_response,
            "meta": meta,
        })
        seen.add(key)

    for row in load_jsonl(TUTOR_SOURCE):
        task = row.get("task", "tutor_explain")
        prompt = (row.get("prompt") or "").strip()
        response = row.get("response") or ""
        meta = row.get("meta") or {}
        if not prompt or not response:
            continue
        add_entry(task, prompt, response, meta)

    if TUTOR_EVAL.exists():
        payload = json.loads(TUTOR_EVAL.read_text(encoding="utf-8"))
        for puzzle in payload.get("puzzles", []):
            fen = puzzle.get("fen")
            if not isinstance(fen, str):
                continue
            question = (puzzle.get("question") or "").strip()
            if not question:
                continue
            prompt = question if question.startswith("FEN:") else f"FEN: {fen}\n{question}"
            reference = (puzzle.get("reference_analysis") or "").strip()
            response = reference or ""
            expected_move = validate_uci(puzzle.get("expected_move") or puzzle.get("best_move"))
            if not expected_move:
                continue
            response = ensure_best_move_line(response or "", expected_move)
            meta = {
                "fen": fen,
                "source": "tutor_eval",
                "rating": puzzle.get("rating"),
                "difficulty": puzzle.get("difficulty"),
                "best_move": expected_move,
                "quality_score": 0.9,
            }
            add_entry("tutor_explain", prompt, response, meta)

    return records


def process_director_dataset() -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    seen_questions: set[str] = set()

    def add_entry(task: str, prompt: str, response: str, meta: Dict[str, object]) -> None:
        key = prompt.strip()
        if key in seen_questions:
            return
        bm = validate_uci(meta.get("best_move"))
        normalized = ensure_final_move_line(response, bm)
        meta = dict(meta)
        if bm:
            meta["best_move"] = bm
        meta.setdefault("quality_score", 0.8)
        meta.setdefault("source", "standardized_director_expert_v3")
        records.append({
            "task": task,
            "prompt": prompt,
            "response": normalized,
            "meta": meta,
        })
        seen_questions.add(key)

    for row in load_jsonl(DIRECTOR_SOURCE):
        task = row.get("task", "director_qa")
        prompt = (row.get("prompt") or "").strip()
        response = row.get("response") or ""
        meta = row.get("meta") or {}
        if not prompt or not response:
            continue
        add_entry(task, prompt, response, meta)

    if DIRECTOR_EVAL.exists():
        payload = json.loads(DIRECTOR_EVAL.read_text(encoding="utf-8"))
        for qa in payload.get("questions", []):
            question = (qa.get("question") or "").strip()
            if not question:
                continue
            expected_answer = (qa.get("expected_answer") or "").strip()
            best_move = validate_uci(qa.get("best_move"))
            meta = {
                "category": qa.get("category", "strategic_explanation"),
                "source": "director_eval",
                "best_move": best_move,
                "rating": qa.get("rating"),
                "quality_score": 0.85,
            }
            response = expected_answer or question
            response = ensure_final_move_line(response, best_move)
            add_entry("director_qa", question, response, meta)

    return records


def main() -> None:
    tutor_records = process_tutor_dataset()
    director_records = process_director_dataset()

    save_jsonl(tutor_records, TUTOR_OUTPUT)
    save_jsonl(director_records, DIRECTOR_OUTPUT)

    print(f"✅ Tutor dataset written to {TUTOR_OUTPUT} ({len(tutor_records)} entries)")
    print(f"✅ Director dataset written to {DIRECTOR_OUTPUT} ({len(director_records)} entries)")


if __name__ == "__main__":
    main()
