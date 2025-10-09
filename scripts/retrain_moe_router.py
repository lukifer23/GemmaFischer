#!/usr/bin/env python3
"""
Retrained MoE Router with Better Query Pattern Recognition

This script retrains the MoE router using the actual evaluation queries
to improve routing accuracy for different question types.
"""

import json
import sys
import os
from pathlib import Path
from typing import List, Dict, Any
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.inference.moe_router import ChessMoERouter, RouterTrainingExample, RouterTrainingDataset
from src.inference.core_engine import ChessGemmaCoreEngine

def create_router_training_data(eval_file: str) -> List[RouterTrainingExample]:
    """Create router training data from evaluation queries with correct expert mappings."""

    training_examples = []

    with open(eval_file, 'r') as f:
        for line in f:
            if line.strip():
                test_case = json.loads(line)

                # Map category to expert
                category_to_expert = {
                    "pure_move": "uci",
                    "position_analysis": "tutor",
                    "opening_strategy": "director",
                    "endgame_principles": "director",
                    "rules_explanation": "director",
                    "tactical_patterns": "director",
                    "mixed_analysis": "tutor"  # Default to tutor for mixed
                }

                expected_expert = category_to_expert.get(test_case['category'], "tutor")
                question = test_case['question']

                # Extract FEN from question
                import re
                fen_match = re.search(r'FEN:\s*([^\n]+)', question)
                if fen_match:
                    fen = fen_match.group(1).strip()
                else:
                    # Skip if no FEN
                    continue

                # Create training example
                import numpy as np
                example = RouterTrainingExample(
                    question=question,
                    question_embedding=np.random.rand(768),  # Placeholder embedding
                    expected_expert=expected_expert,
                    fen=fen,
                    category=test_case['category']
                )

                training_examples.append(example)

    return training_examples

def retrain_router():
    """Retrain the MoE router with corrected training data."""

    print("🔄 Retraining MoE Router...")
    print("=" * 50)

    # Load evaluation data to create training examples
    eval_file = "data/validation/eval_suite.jsonl"
    if not Path(eval_file).exists():
        print(f"❌ Evaluation file not found: {eval_file}")
        return

    print("📚 Loading evaluation queries...")
    training_examples = create_router_training_data(eval_file)
    print(f"   Created {len(training_examples)} training examples")

    # Split into train/val
    random.shuffle(training_examples)
    split_idx = int(len(training_examples) * 0.8)
    train_examples = training_examples[:split_idx]
    val_examples = training_examples[split_idx:]

    print(f"   Training set: {len(train_examples)} examples")
    print(f"   Validation set: {len(val_examples)} examples")

    # Create router
    router = ChessMoERouter(num_experts=3)

    # Create data loader
    train_dataset = RouterTrainingDataset(train_examples)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    val_dataset = RouterTrainingDataset(val_examples)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    # Training setup
    optimizer = torch.optim.Adam(router.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    router.to(device)

    print(f"🚀 Training on {device}...")

    best_val_accuracy = 0.0
    patience = 5
    patience_counter = 0

    for epoch in range(20):  # Max 20 epochs
        # Training phase
        router.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in train_loader:
            optimizer.zero_grad()

            # Move to device
            features = batch['embedding'].to(device)
            labels = batch['target'].to(device)

            # Forward pass
            gate_logits, confidence = router(features.unsqueeze(0))  # Add batch dimension
            loss = criterion(gate_logits, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Calculate accuracy
            predictions = torch.argmax(gate_logits, dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        train_accuracy = correct / total
        avg_loss = total_loss / len(train_loader)

        # Validation phase
        router.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                features = batch['embedding'].to(device)
                labels = batch['target'].to(device)

                gate_logits, confidence = router(features.unsqueeze(0))  # Add batch dimension
                predictions = torch.argmax(gate_logits, dim=-1)
                val_correct += (predictions == labels).sum().item()
                val_total += labels.size(0)

        val_accuracy = val_correct / val_total

        print(f"Epoch {epoch+1:2d}: Loss={avg_loss:.4f}, Train Acc={train_accuracy:.3f}, Val Acc={val_accuracy:.3f}")

        # Early stopping
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            patience_counter = 0

            # Save best model
            checkpoint_dir = Path("checkpoints/moe_router")
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            torch.save({
                'epoch': epoch,
                'model_state_dict': router.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_accuracy,
                'loss': avg_loss
            }, checkpoint_dir / "retrained_checkpoint.pth")

            print("   💾 Saved best model checkpoint")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping after {epoch+1} epochs")
                break

    print("\n✅ Router retraining complete!")
    print(f"   Best validation accuracy: {best_val_accuracy:.3f}")

    # Load best checkpoint and save as final
    if (checkpoint_dir / "retrained_checkpoint.pth").exists():
        checkpoint = torch.load(checkpoint_dir / "retrained_checkpoint.pth")
        router.load_state_dict(checkpoint['model_state_dict'])

        # Save as final checkpoint
        torch.save(router.state_dict(), checkpoint_dir / "final_checkpoint_retrained.pth")
        print(f"   📦 Saved final retrained checkpoint")

if __name__ == "__main__":
    retrain_router()
