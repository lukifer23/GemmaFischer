#!/usr/bin/env python3
"""Refresh tutor/director expert datasets with cleaned, augmented entries.

- Deduplicate by FEN/prompt combination
- Ensure tutor responses end with ``best move: <uci>``
- Ensure director responses end with ``Final move (UCI): <uci>`` when available
- Optionally generate an LC0-labelled tutor split with hybrid explanations
  saved to ``data/standardized/standardized_tutor_lc0_v1.jsonl``
"""
from __future__ import annotations

import argparse
import json
import logging
import random
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
TUTOR_LC0_OUTPUT = STANDARDIZED_DIR / "standardized_tutor_lc0_v1.jsonl"

TUTOR_EVAL = VALIDATION_DIR / "tutor_comprehensive_validation.json"
DIRECTOR_EVAL = VALIDATION_DIR / "director_comprehensive_validation.json"

# Helper regex patterns
BEST_MOVE_PATTERN = re.compile(r"best move:\s*([a-h][1-8][a-h][1-8][qrbn]?)", re.IGNORECASE)
FINAL_MOVE_PATTERN = re.compile(r"final move\s*\(uci\):\s*([a-h][1-8][a-h][1-8][qrbn]?)", re.IGNORECASE)
UCI_PATTERN = re.compile(r"^[a-h][1-8][a-h][1-8][qrbn]?$")

RNG = random.Random(3407)

try:  # Lazy import so offline environments can still refresh baselines
    from src.inference.inference import get_inference_instance
except Exception:  # pragma: no cover - optional dependency
    get_inference_instance = None


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


def _normalize_hybrid_response(text: str, engine_move: str) -> str:
    """Ensure the hybrid explanation ends with a best-move line."""

    body = text.strip()
    if body and not body.endswith("\n"):
        body += "\n"
    if f"best move: {engine_move}".lower() not in body.lower():
        body += f"Best move: {engine_move}"
    return body


def generate_lc0_tutor_dataset(
    base_records: List[Dict[str, object]],
    *,
    limit: Optional[int] = None,
) -> List[Dict[str, object]]:
    """Run LC0 hybrid analysis over tutor prompts to build labelled samples."""

    if get_inference_instance is None:
        raise RuntimeError(
            "LC0 inference stack is unavailable. Install dependencies and ensure "
            "the inference package can be imported."
        )

    inference = get_inference_instance()
    inference.load_model()

    candidates = [rec for rec in base_records if rec.get("meta", {}).get("fen")]
    RNG.shuffle(candidates)

    seen_fens: set[str] = set()
    enriched: List[Dict[str, object]] = []

    for record in candidates:
        if limit is not None and len(enriched) >= limit:
            break

        meta = record.get("meta") or {}
        fen = meta.get("fen")
        if not fen or fen in seen_fens:
            continue

        prompt = record.get("prompt") or ""

        try:
            engine_payload = inference.analyze_with_engine(fen, explanation_mode="tutor")
        except Exception as exc:  # pragma: no cover - engine failures are rare
            logging.warning("LC0 analysis failed for %s: %s", fen, exc)
            continue

        engine_move = engine_payload.get("best_move")
        if not engine_move:
            continue

        explanation = engine_payload.get("explanation") or record.get("response") or ""
        normalized_response = _normalize_hybrid_response(explanation, engine_move)

        principal_variation = engine_payload.get("principal_variation") or []

        enriched_meta = dict(meta)
        enriched_meta.update(
            {
                "engine": engine_payload.get("engine"),
                "engine_move": engine_move,
                "engine_evaluation_cp": engine_payload.get("evaluation_cp"),
                "engine_evaluation_pawns": engine_payload.get("evaluation_pawns"),
                "engine_mate_in": engine_payload.get("mate_in"),
                "engine_depth": engine_payload.get("depth"),
                "engine_nodes": engine_payload.get("nodes"),
                "engine_time": engine_payload.get("engine_time"),
                "principal_variation": principal_variation,
                "hybrid_key_points": engine_payload.get("key_points", []),
                "explanation_adapter": engine_payload.get("explanation_adapter"),
                "engine_source": "lc0_hybrid_v1",
                "source": "standardized_tutor_lc0_v1",
            }
        )
        enriched_meta.setdefault("quality_score", meta.get("quality_score", 0.9))

        enriched.append(
            {
                "task": "tutor_explain",
                "prompt": prompt,
                "response": normalized_response,
                "meta": enriched_meta,
            }
        )
        seen_fens.add(fen)

    return enriched


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
    parser = argparse.ArgumentParser(description="Refresh standardized expert datasets.")
    parser.add_argument(
        "--skip-lc0",
        action="store_true",
        help="Skip generating the LC0-labelled tutor dataset.",
    )
    parser.add_argument(
        "--lc0-limit",
        type=int,
        default=None,
        help="Cap the number of LC0 tutor samples (defaults to all available prompts).",
    )
    args = parser.parse_args()

    tutor_records = process_tutor_dataset()
    director_records = process_director_dataset()

    save_jsonl(tutor_records, TUTOR_OUTPUT)
    save_jsonl(director_records, DIRECTOR_OUTPUT)

    print(f"✅ Tutor dataset written to {TUTOR_OUTPUT} ({len(tutor_records)} entries)")
    print(f"✅ Director dataset written to {DIRECTOR_OUTPUT} ({len(director_records)} entries)")

    if not args.skip_lc0:
        try:
            lc0_records = generate_lc0_tutor_dataset(tutor_records, limit=args.lc0_limit)
        except RuntimeError as exc:
            logging.warning("Skipping LC0 tutor generation: %s", exc)
        else:
            save_jsonl(lc0_records, TUTOR_LC0_OUTPUT)
            print(
                f"✅ LC0 tutor dataset written to {TUTOR_LC0_OUTPUT} "
                f"({len(lc0_records)} entries)"
            )


if __name__ == "__main__":
    main()
