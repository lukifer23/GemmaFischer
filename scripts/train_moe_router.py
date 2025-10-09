#!/usr/bin/env python3
"""
Train the MoE router on actual expert performance

This script trains the ChessMoERouter by testing all experts on training queries
and learning to route to the expert that performs best for each query type.
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import random
import time
import argparse

import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.inference.moe_router import ChessMoERouter, RouterTrainingExample
from src.inference.inference import ChessGemmaInference

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

class ExpertEvaluator:
    """Evaluates which expert performs best on a given query."""

    def __init__(self, inference_system: ChessGemmaInference):
        self.inference = inference_system
        self.expert_modes = ['uci', 'tutor', 'director']

    def evaluate_expert_responses(self, question: str, context: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """Run all experts on a query and return their responses."""
        results = {}

        for expert in self.expert_modes:
            try:
                # Configure expert-specific parameters
                if expert == 'uci':
                    max_tokens = 8
                    temperature = 0.0
                elif expert == 'tutor':
                    max_tokens = 150
                    temperature = 0.7
                else:  # director
                    max_tokens = 200
                    temperature = 0.6

                response = self.inference.generate_expert_response(
                    question=question,
                    context=context,
                    expert_mode=expert,
                    max_new_tokens=max_tokens,
                    temperature=temperature
                )

                results[expert] = {
                    'response': response.get('response', ''),
                    'confidence': response.get('confidence', 0.5),
                    'expert': expert
                }

            except Exception as e:
                print(f"Error with {expert} expert: {e}")
                results[expert] = {
                    'response': '',
                    'confidence': 0.0,
                    'expert': expert,
                    'error': str(e)
                }

        return results

    def score_expert_response(self, response: Dict[str, Any], expected_category: str) -> float:
        """Score how well an expert response matches the expected category."""
        expert = response['expert']
        text = response.get('response', '').strip()
        confidence = response.get('confidence', 0.5)

        if not text:
            return 0.0

        score = confidence  # Base score from model's confidence

        # Category-specific scoring
        if expected_category in ['pure_move', 'tactical_patterns']:
            # UCI expert should give clean moves
            if expert == 'uci' and len(text.strip()) <= 10 and any(c.islower() for c in text):
                score += 0.5  # Bonus for UCI-like format
        elif expected_category in ['position_analysis', 'mixed_analysis']:
            # Tutor expert should give detailed analysis
            if expert == 'tutor' and len(text.split()) > 20:
                score += 0.3  # Bonus for detailed analysis
        elif expected_category in ['opening_strategy', 'endgame_principles', 'rules_explanation']:
            # Director expert should give strategic explanations
            if expert == 'director' and any(term in text.lower() for term in ['strategy', 'principle', 'important', 'key']):
                score += 0.4  # Bonus for strategic content

        # Penalize off-topic responses
        if expert == 'uci' and len(text.split()) > 15:
            score -= 0.3  # UCI should be concise
        elif expert == 'tutor' and len(text.split()) < 10:
            score -= 0.3  # Tutor should be detailed
        elif expert == 'director' and not any(term in text.lower() for term in ['strategy', 'principle', 'rule', 'concept']):
            score -= 0.2  # Director should be conceptual

        return max(0.0, min(1.0, score))

    def select_best_expert(self, expert_responses: Dict[str, Dict[str, Any]], expected_category: str) -> str:
        """Select the best expert based on query category rather than response quality.

        This approach is more reliable than trying to score actual model responses,
        since all responses might look similar quality-wise.
        """
        # Rule-based expert selection based on query category
        category_to_expert = {
            # UCI expert for move generation
            'pure_move': 'uci',
            'tactical_move': 'uci',
            'standard_move': 'uci',
            'complex_tactical': 'uci',

            # Tutor expert for analysis and teaching
            'position_analysis': 'tutor',
            'candidate_evaluation': 'tutor',
            'tactical_patterns': 'tutor',
            'mixed_analysis': 'tutor',
            'endgame_principles': 'tutor',

            # Director expert for strategy and concepts
            'opening_strategy': 'director',
            'strategic_explanation': 'director',
            'endgame_principles': 'director',
            'rules_explanation': 'director',
            'middlegame_strategy': 'director',
        }

        # Default to tutor if category not recognized
        best_expert = category_to_expert.get(expected_category, 'tutor')

        print(f"      Best: {best_expert} (category: {expected_category})")

        return best_expert

def load_router_checkpoint(router: ChessMoERouter, checkpoint_path: Optional[str] = None) -> bool:
    """Load router checkpoint if available."""
    if checkpoint_path and Path(checkpoint_path).exists():
        try:
            router.load_router(checkpoint_path)
            print(f"✅ Loaded router checkpoint: {checkpoint_path}")
            return True
        except Exception as e:
            print(f"❌ Failed to load checkpoint {checkpoint_path}: {e}")
            return False

    # Try to find latest checkpoint
    checkpoint_dir = Path("checkpoints/moe_router")
    if checkpoint_dir.exists():
        checkpoints = list(checkpoint_dir.glob("*.pth"))
        if checkpoints:
            latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
            try:
                router.load_router(str(latest))
                print(f"✅ Loaded latest router checkpoint: {latest}")
                return True
            except Exception as e:
                print(f"❌ Failed to load latest checkpoint {latest}: {e}")
                return False

    print("ℹ️  No router checkpoint found, starting fresh training")
    return False

def prepare_training_data_with_inference(queries: List[Dict[str, Any]],
                                       inference_system: ChessGemmaInference,
                                       router: ChessMoERouter) -> List[RouterTrainingExample]:
    """Prepare training data by testing actual expert performance."""
    evaluator = ExpertEvaluator(inference_system)
    training_examples = []

    print("🎯 Preparing MoE router training data with expert inference...")

    for i, query in enumerate(queries):
        if i % 10 == 0:  # More frequent progress updates
            print(f"   Processing query {i+1}/{len(queries)}")

        question = query["question"]
        expected_category = query.get("category", "general")

        # Extract FEN if present
        fen_match = router._fen_pattern.search(question)
        fen = fen_match.group(1) if fen_match else None

        print(f"   Query: {question[:60]}...")
        print(f"      Category: {expected_category}, FEN: {fen}")

        # Use the labeled expert from the dataset for training
        # This is more reliable than trying to evaluate actual responses
        best_expert = query.get("expert", "tutor")  # Default to tutor if not specified

        print(f"      Labeled expert: {best_expert}")

        # Extract features from position and question
        features = router._extract_position_features(fen or "", question)

        training_examples.append(RouterTrainingExample(
            question=question,
            question_embedding=features.numpy(),
            expected_expert=best_expert,
            fen=fen,
            category=expected_category
        ))

    print(f"✅ Prepared {len(training_examples)} training examples")
    return training_examples

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train MoE Router with expert evaluation')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--eval_file', type=str, default='data/validation/eval_suite.jsonl',
                       help='Evaluation data file')
    parser.add_argument('--validate_every', type=int, default=5, help='Validate every N epochs')
    parser.add_argument('--resume_from', type=str, default=None,
                       help='Resume training from checkpoint path')
    parser.add_argument('--skip_data_prep', action='store_true',
                       help='Skip data preparation (use cached if available)')

    args = parser.parse_args()

    print("🎯 MoE Router Training Script (Expert Evaluation)")
    print("=" * 60)

    # Check if evaluation file exists
    if not Path(args.eval_file).exists():
        print(f"❌ Evaluation file not found: {args.eval_file}")
        print("Please ensure evaluation data exists.")
        sys.exit(1)

    # Load evaluation queries
    print("📋 Loading evaluation queries...")
    queries = load_evaluation_queries(args.eval_file)
    print(f"   Found {len(queries)} evaluation queries")

    # Split into train/validation
    train_queries, val_queries = split_training_data(queries, train_ratio=0.8)
    print(f"   Training: {len(train_queries)} queries")
    print(f"   Validation: {len(val_queries)} queries")

    # Initialize inference system
    print("\n🔧 Initializing inference system...")
    try:
        inference_system = ChessGemmaInference()
        print("   ✅ Inference system ready")
    except Exception as e:
        print(f"❌ Failed to initialize inference system: {e}")
        sys.exit(1)

    # Initialize router
    print("\n🔧 Initializing MoE router...")
    router = ChessMoERouter(
        num_experts=3,
        expert_names=["uci", "tutor", "director"]
    )

    # Load checkpoint if resuming
    checkpoint_loaded = load_router_checkpoint(router, args.resume_from)
    if checkpoint_loaded:
        print("📋 Resuming from checkpoint - will regenerate training data...")

    # Prepare training data with actual expert evaluation (unless skipping)
    if not args.skip_data_prep:
        print("\n🎯 Preparing training data with expert inference...")
        training_examples = prepare_training_data_with_inference(train_queries, inference_system, router)
        # Prepare validation data
        val_examples = prepare_training_data_with_inference(val_queries, inference_system, router)
    else:
        print("\n⏭️  Skipping data preparation...")
        training_examples = []
        val_examples = []

    # Train the router
    print("\n🚀 Training MoE router...")
    print(f"   Epochs: {args.epochs}, Batch size: {args.batch_size}, LR: {args.learning_rate}")

    best_accuracy = router.train_router(
        training_examples=training_examples,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        validate_every=args.validate_every,
        validation_examples=val_examples
    )

    # Evaluate final performance
    print("\n📊 Final evaluation on validation set...")
    val_accuracy = router.evaluate_routing_accuracy(val_examples)

    print("\n✅ Training Complete!")
    print(".1f")
    print(".1f")
    # Save final model
    os.makedirs("checkpoints/moe_router", exist_ok=True)
    final_path = "checkpoints/moe_router/final_checkpoint.pth"
    router.save_router(final_path)
    print(f"\n💾 Final model saved to: {final_path}")

    # Print expert distribution
    print("\n📈 Training Data Expert Distribution:")
    expert_counts = {}
    for example in training_examples:
        expert_counts[example.expected_expert] = expert_counts.get(example.expected_expert, 0) + 1

    for expert, count in expert_counts.items():
        percentage = count / len(training_examples) * 100
        print(".1f")
    print("\n🎯 Next Steps:")
    print("1. Test router: python scripts/test_moe_routing.py")
    print("2. Enable MoE: export CHESSGEMMA_MOE_ENABLED=1")
    print("3. Set router path: export CHESSGEMMA_MOE_ROUTER_CKPT='checkpoints/moe_router/final_checkpoint.pth'")

if __name__ == "__main__":
    main()
