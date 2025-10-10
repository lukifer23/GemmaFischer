#!/usr/bin/env python3
"""
Generate evaluation datasets for expert scorecards and router testing.

This script samples high-quality records from the standardized training corpora
and produces curated evaluation files with consistent schemas:

  - data/validation/eval_mixed_positions_200.jsonl
  - data/validation/tutor_comprehensive_validation.json
  - data/validation/director_comprehensive_validation.json

No synthetic placeholders are introduced; all samples come directly from the
cleaned training data. Existing files will be overwritten.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable, Dict, Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
STANDARDIZED_DIR = PROJECT_ROOT / "data" / "standardized"
VALIDATION_DIR = PROJECT_ROOT / "data" / "validation"

UCI_SOURCE = STANDARDIZED_DIR / "standardized_uci_expert_v2.jsonl"
TUTOR_SOURCE = STANDARDIZED_DIR / "standardized_tutor_expert_v2.jsonl"
DIRECTOR_SOURCE = STANDARDIZED_DIR / "standardized_director_expert_v3.jsonl"

UCI_OUTPUT = VALIDATION_DIR / "eval_mixed_positions_200.jsonl"
TUTOR_OUTPUT = VALIDATION_DIR / "tutor_comprehensive_validation.json"
DIRECTOR_OUTPUT = VALIDATION_DIR / "director_comprehensive_validation.json"

SAMPLE_SEED = 3407
RANDOM = random.Random(SAMPLE_SEED)


def load_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def difficulty_from_rating(rating: Optional[int]) -> str:
    if rating is None:
        return "unknown"
    if rating < 1400:
        return "beginner"
    if rating < 1800:
        return "intermediate"
    if rating < 2100:
        return "advanced"
    return "expert"


def pick_uci_positions(limit: int = 200) -> None:
    from src.inference.uci_utils import validate_uci_syntax  # Local import

    seen_fens = set()
    samples = []

    candidate_rows = list(load_jsonl(UCI_SOURCE))
    RANDOM.shuffle(candidate_rows)

    for row in candidate_rows:
        meta = row.get("meta") or {}
        fen = meta.get("fen")
        move = (row.get("response") or "").strip()

        if not fen or fen in seen_fens:
            continue
        if not validate_uci_syntax(move):
            continue

        rating = meta.get("rating")
        entry = {
            "id": f"uci-{len(samples)+1:04d}",
            "fen": fen,
            "expected_move": move,
            "difficulty": difficulty_from_rating(rating),
            "rating": rating,
            "category": meta.get("topic", "mixed"),
            "source": meta.get("source", "standardized_uci_expert_v2"),
        }
        samples.append(entry)
        seen_fens.add(fen)

        if len(samples) >= limit:
            break

    if len(samples) < limit:
        raise RuntimeError(f"Only gathered {len(samples)} UCI positions; need {limit}.")

    with UCI_OUTPUT.open("w", encoding="utf-8") as fh:
        for sample in samples:
            fh.write(json.dumps(sample))
            fh.write("\n")


def pick_tutor_puzzles(limit: int = 80) -> None:
    puzzles = []
    candidate_rows = list(load_jsonl(TUTOR_SOURCE))
    RANDOM.shuffle(candidate_rows)

    for row in candidate_rows:
        meta = row.get("meta") or {}
        fen = meta.get("fen")
        prompt = row.get("prompt") or ""
        response = row.get("response") or ""
        best_move = meta.get("best_move")

        if not fen or "[tactical_move]" in response:
            continue

        if not best_move:
            # Attempt to extract from response tail
            for line in response.splitlines()[::-1]:
                line = line.strip()
                if line.lower().startswith("best move:"):
                    best_move = line.split(":", 1)[-1].strip()
                    break
        if not best_move:
            continue

        rating = meta.get("rating")
        puzzles.append({
            "id": f"tutor-{len(puzzles)+1:04d}",
            "fen": fen,
            "expected_move": best_move,
            "rating": rating,
            "difficulty": difficulty_from_rating(rating),
            "question": prompt.strip(),
            "reference_analysis": response.strip(),
            "source": meta.get("source", "standardized_tutor_expert"),
        })

        if len(puzzles) >= limit:
            break

    if len(puzzles) < limit:
        raise RuntimeError(f"Only gathered {len(puzzles)} tutor puzzles; need {limit}.")

    payload = {
        "metadata": {
            "generated_from": str(TUTOR_SOURCE),
            "sample_size": len(puzzles),
            "seed": SAMPLE_SEED,
        },
        "puzzles": puzzles,
    }

    with TUTOR_OUTPUT.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def pick_director_questions(limit: int = 80) -> None:
    questions = []
    candidate_rows = list(load_jsonl(DIRECTOR_SOURCE))
    RANDOM.shuffle(candidate_rows)

    for row in candidate_rows:
        prompt = row.get("prompt") or ""
        response = row.get("response") or ""
        meta = row.get("meta") or {}

        if "[tactical_move]" in response:
            continue

        category = meta.get("topic", "general")
        rating = meta.get("rating")

        questions.append({
            "id": f"director-{len(questions)+1:04d}",
            "question": prompt.strip(),
            "expected_answer": response.strip(),
            "category": category,
            "rating": rating,
            "best_move": meta.get("best_move"),
            "source": meta.get("source", "standardized_director_expert_v2"),
        })

        if len(questions) >= limit:
            break

    if len(questions) < limit:
        raise RuntimeError(f"Only gathered {len(questions)} director questions; need {limit}.")

    payload = {
        "metadata": {
            "generated_from": str(DIRECTOR_SOURCE),
            "sample_size": len(questions),
            "seed": SAMPLE_SEED,
        },
        "questions": questions,
    }

    with DIRECTOR_OUTPUT.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def main() -> None:
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)

    pick_uci_positions()
    pick_tutor_puzzles()
    pick_director_questions()

    print("✅ Evaluation datasets generated:")
    print(f"  - {UCI_OUTPUT}")
    print(f"  - {TUTOR_OUTPUT}")
    print(f"  - {DIRECTOR_OUTPUT}")


if __name__ == "__main__":
    main()
