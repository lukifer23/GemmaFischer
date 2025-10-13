#!/usr/bin/env python3
"""
Train the ChessGemma MoE router using curated, labeled queries.

This refreshes the router with real evaluation data drawn from the cleaned
datasets (UCI, Tutor, Director) and the expanded evaluation suites. The
resulting checkpoint is saved under ``checkpoints/moe_router``.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

# Repository paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.moe_router import ChessMoERouter, RouterTrainingExample  # noqa: E402

FEN_PATTERN = re.compile(r"FEN:\s*([^\n]+)", re.IGNORECASE)
RNG = random.Random(3407)

# Data sources
EVAL_SUITE = PROJECT_ROOT / "data" / "validation" / "eval_suite.jsonl"
EXPANDED_EVAL_SUITE = PROJECT_ROOT / "data" / "validation" / "expanded_eval_suite.jsonl"
ROUTER_QUERIES = PROJECT_ROOT / "data" / "validation" / "router_evaluation_queries.json"
UCI_EVAL = PROJECT_ROOT / "data" / "validation" / "eval_mixed_positions_200.jsonl"
TUTOR_EVAL = PROJECT_ROOT / "data" / "validation" / "tutor_comprehensive_validation.json"
DIRECTOR_EVAL = PROJECT_ROOT / "data" / "validation" / "director_comprehensive_validation.json"

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "moe_router"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


def load_jsonl(path: Path) -> List[Dict[str, any]]:
    records: List[Dict[str, any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def extract_fen(question: str) -> Optional[str]:
    match = FEN_PATTERN.search(question)
    if match:
        return match.group(1).strip()
    return None


@dataclass
class LabeledQuery:
    question: str
    expected_expert: str
    category: str = "general"
    fen: Optional[str] = None


def load_labeled_queries() -> List[LabeledQuery]:
    queries: Dict[str, LabeledQuery] = {}

    def register(question: str, expert: str, category: str = "general", fen: Optional[str] = None) -> None:
        key = question.strip()
        if key in queries:
            return
        queries[key] = LabeledQuery(
            question=question.strip(),
            expected_expert=expert,
            category=category,
            fen=fen or extract_fen(question),
        )

    # Core evaluation suite
    for row in load_jsonl(EVAL_SUITE):
        register(
            question=row["question"],
            expert=row["expert"],
            category=row.get("category", "general"),
        )

    # Expanded evaluation suite
    for row in load_jsonl(EXPANDED_EVAL_SUITE):
        register(
            question=row["question"],
            expert=row["expert"],
            category=row.get("category", "general"),
        )

    # Router health queries
    if ROUTER_QUERIES.exists():
        payload = json.loads(ROUTER_QUERIES.read_text(encoding="utf-8"))
        for row in payload.get("queries", []):
            register(
                question=row["question"],
                expert=row["expected_expert"],
                category=row.get("category", "general"),
            )

    # High-quality UCI positions
    for row in load_jsonl(UCI_EVAL):
        fen = row["fen"]
        prompt = f"FEN: {fen}\nFind the strongest move in UCI notation. Respond with the move only."
        register(
            question=prompt,
            expert="uci",
            category=row.get("category", "pure_move"),
            fen=fen,
        )

    # Tutor analysis puzzles
    tutor_payload = json.loads(TUTOR_EVAL.read_text(encoding="utf-8"))
    seen_tutor: set[str] = set()
    for puzzle in tutor_payload.get("puzzles", []):
        question = puzzle.get("question", "").strip()
        if not question or question in seen_tutor:
            continue
        prompt = question if question.startswith("FEN:") else f"FEN: {puzzle['fen']}\n{question}"
        register(
            question=prompt,
            expert="tutor",
            category=puzzle.get("difficulty", "analysis"),
            fen=puzzle.get("fen"),
        )
        seen_tutor.add(question)

    # Director strategic questions
    director_payload = json.loads(DIRECTOR_EVAL.read_text(encoding="utf-8"))
    seen_director: set[str] = set()
    for qa in director_payload.get("questions", []):
        question = (qa.get("question") or "").strip()
        if not question or question in seen_director:
            continue
        register(
            question=question,
            expert="director",
            category=qa.get("category", "strategic_explanation"),
            fen=qa.get("best_move") and extract_fen(question),
        )
        seen_director.add(question)

    return list(queries.values())


def balance_queries(queries: List[LabeledQuery], max_per_expert: int) -> List[LabeledQuery]:
    buckets: Dict[str, List[LabeledQuery]] = {"uci": [], "tutor": [], "director": []}
    for q in queries:
        if q.expected_expert in buckets:
            buckets[q.expected_expert].append(q)

    balanced: List[LabeledQuery] = []
    for expert, items in buckets.items():
        RNG.shuffle(items)
        balanced.extend(items[:max_per_expert])
    RNG.shuffle(balanced)
    return balanced


def load_logged_queries(log_path: Path, min_confidence: float, include_overrides: bool) -> List[LabeledQuery]:
    """Load high-confidence router decisions for self-training."""
    if not log_path.exists():
        return []

    logged: List[LabeledQuery] = []
    with log_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue

            confidence = float(payload.get("final_confidence", 0.0))
            if confidence < min_confidence:
                continue
            if not include_overrides and payload.get("keyword_override"):
                continue

            question = payload.get("question") or payload.get("question_preview") or ""
            expert = payload.get("primary_expert")
            if not question or expert not in {"uci", "tutor", "director"}:
                continue
            fen = payload.get("fen") or extract_fen(question)
            logged.append(
                LabeledQuery(
                    question=question,
                    expected_expert=expert,
                    category="logged_routing",
                    fen=fen,
                )
            )
    return logged


def build_training_examples(router: ChessMoERouter, queries: List[LabeledQuery]) -> List[RouterTrainingExample]:
    training_examples: List[RouterTrainingExample] = []
    for idx, query in enumerate(queries, 1):
        fen = query.fen or ""
        features = router._extract_position_features(fen, query.question)
        training_examples.append(
            RouterTrainingExample(
                question=query.question,
                question_embedding=np.array(features.cpu().numpy(), dtype=np.float32),
                expected_expert=query.expected_expert,
                fen=query.fen,
                category=query.category,
            )
        )
        if idx % 100 == 0:
            print(f"   Prepared {idx} examples...")
    return training_examples


def split_examples(
    examples: List[RouterTrainingExample], train_ratio: float = 0.85
) -> Tuple[List[RouterTrainingExample], List[RouterTrainingExample]]:
    RNG.shuffle(examples)
    split_idx = int(len(examples) * train_ratio)
    return examples[:split_idx], examples[split_idx:]


def train_router(
    epochs: int,
    batch_size: int,
    learning_rate: float,
    train_examples: List[RouterTrainingExample],
    val_examples: List[RouterTrainingExample],
) -> Tuple[float, float]:
    router = ChessMoERouter(num_experts=3, expert_names=["uci", "tutor", "director"])
    print(f"📦 Training set: {len(train_examples)} examples | Validation: {len(val_examples)} examples")

    best_train_acc = router.train_router(
        training_examples=train_examples,
        num_epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        validate_every=max(1, epochs // 5),
        validation_examples=val_examples or None,
    )

    val_acc = router.evaluate_routing_accuracy(val_examples) if val_examples else 0.0

    best_path = CHECKPOINT_DIR / "best_checkpoint.pth"
    final_path = CHECKPOINT_DIR / "router_final.pth"
    router.save_router(str(final_path))
    print(f"💾 Final router checkpoint saved to: {final_path}")
    if best_path.exists():
        print(f"⭐ Best checkpoint available at: {best_path}")

    return best_train_acc, val_acc


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the MoE router with curated labels")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=2e-3, help="Learning rate")
    parser.add_argument("--max-per-expert", type=int, default=400, help="Max samples per expert")
    parser.add_argument("--decision-log", type=str, default=str(Path("reports") / "moe" / "routing_decisions.jsonl"),
                        help="Optional router decision log to augment training data")
    parser.add_argument("--min-log-confidence", type=float, default=0.78,
                        help="Minimum confidence to accept a logged routing decision")
    parser.add_argument("--include-overrides", action="store_true",
                        help="Include keyword-override decisions from the log")
    parser.add_argument("--max-log-samples", type=int, default=300,
                        help="Maximum number of logged routing samples to use (0 = unlimited)")
    args = parser.parse_args()

    print("🎯 Building labeled query set...")
    queries = load_labeled_queries()
    print(f"   Total unique queries collected: {len(queries)}")

    # Optionally extend with logged production decisions
    decision_log_path = Path(args.decision_log)
    logged_queries = load_logged_queries(decision_log_path, args.min_log_confidence, args.include_overrides)
    if args.max_log_samples > 0:
        logged_queries = logged_queries[:args.max_log_samples]
    if logged_queries:
        existing_questions = {q.question for q in queries}
        fresh = [q for q in logged_queries if q.question not in existing_questions]
        queries.extend(fresh)
        print(f"   Added {len(fresh)} logged routing samples (from {decision_log_path})")

    balanced_queries = balance_queries(queries, max_per_expert=args.max_per_expert)
    counts = {}
    for q in balanced_queries:
        counts[q.expected_expert] = counts.get(q.expected_expert, 0) + 1
    print(f"   Balanced counts: {counts}")

    router = ChessMoERouter(num_experts=3, expert_names=["uci", "tutor", "director"])
    print("⚙️  Extracting feature representations...")
    all_examples = build_training_examples(router, balanced_queries)
    train_examples, val_examples = split_examples(all_examples, train_ratio=0.85)

    print("🚀 Training router...")
    best_train_acc, val_acc = train_router(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        train_examples=train_examples,
        val_examples=val_examples,
    )

    print("\n📊 Training summary:")
    print(f"   Train accuracy (best epoch): {best_train_acc:.1%}")
    print(f"   Validation accuracy: {val_acc:.1%}")
    print("✅ Router training complete.")


if __name__ == "__main__":
    main()
