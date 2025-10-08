#!/usr/bin/env python3
"""
Train the MoE router on routing decisions

This script trains the ChessMoERouter to intelligently route queries
to the appropriate expert (UCI, Tutor, Director) based on question content.
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any
import random

# Add src to path
sys.path.append('src')

from inference.moe_router import ChessMoERouter, RouterTrainingExample

def load_evaluation_queries(eval_file: str) -> List[Dict[str, Any]]:
    """Load evaluation queries for training data."""
    queries = []
    with open(eval_file, 'r') as f:
        for line in f:
            if line.strip():
                queries.append(json.loads(line))
    return queries

def split_training_data(queries: List[Dict[str, Any]], train_ratio: float = 0.8):
    """Split queries into training and validation sets."""
    random.shuffle(queries)
    split_idx = int(len(queries) * train_ratio)

    train_queries = queries[:split_idx]
    val_queries = queries[split_idx:]

    return train_queries, val_queries

def main():
    """Main training function."""
    print("🎯 MoE Router Training Script")
    print("=" * 50)

    # Configuration
    eval_file = "data/validation/eval_suite.jsonl"
    num_epochs = 20
    batch_size = 16
    learning_rate = 1e-3

    # Check if evaluation file exists
    if not Path(eval_file).exists():
        print(f"❌ Evaluation file not found: {eval_file}")
        print("Please run the evaluation suite creation first.")
        sys.exit(1)

    # Load evaluation queries
    print("📋 Loading evaluation queries...")
    queries = load_evaluation_queries(eval_file)
    print(f"   Found {len(queries)} evaluation queries")

    # Split into train/validation
    train_queries, val_queries = split_training_data(queries, train_ratio=0.8)
    print(f"   Training: {len(train_queries)} queries")
    print(f"   Validation: {len(val_queries)} queries")

    # Initialize router
    print("\n🔧 Initializing MoE router...")
    router = ChessMoERouter(
        num_experts=3,
        feature_dim=30,  # Match our embedding size
        expert_names=["uci", "tutor", "director"]
    )

    # Prepare training data
    print("\n🎯 Preparing training data...")
    training_examples = router.prepare_training_data(train_queries, inference_system=None)

    # Prepare validation data
    val_examples = router.prepare_training_data(val_queries, inference_system=None)

    # Train the router
    print("\n🚀 Training MoE router...")
    best_accuracy = router.train_router(
        training_examples=training_examples,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate
    )

    # Evaluate on validation set
    print("\n📊 Evaluating on validation set...")
    val_accuracy = router.evaluate_routing_accuracy(val_examples)

    print("\n✅ Training Complete!")
    print(f"Final training accuracy: {best_accuracy:.1%}")
    print(f"Final validation accuracy: {val_accuracy:.1%}")
    # Save final model
    final_path = "checkpoints/moe_router/final_checkpoint.pth"
    router.save_router(final_path)
    print(f"\n💾 Final model saved to: {final_path}")

    # Print expert distribution in training data
    print("\n📈 Training Data Expert Distribution:")
    expert_counts = {}
    for example in training_examples:
        expert_counts[example.expected_expert] = expert_counts.get(example.expected_expert, 0) + 1

    for expert, count in expert_counts.items():
        percentage = count / len(training_examples) * 100
        print("6")

    print("\n🎯 Next Steps:")
    print("1. Test the trained router with: python scripts/run_evaluation_suite.py")
    print("2. Enable MoE in inference by setting CHESSGEMMA_MOE_ENABLED=1")
    print("3. Monitor routing accuracy in web interface logs")

if __name__ == "__main__":
    main()
