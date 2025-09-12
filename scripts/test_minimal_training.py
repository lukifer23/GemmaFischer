#!/usr/bin/env python3
"""
Minimal Training Test - Debug the hanging issue

This script tests the absolute minimal training setup to identify
where exactly the hang is occurring.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

def test_minimal_training():
    """Test with minimal settings to isolate the hang."""
    print("🔬 Minimal Training Diagnostic Test")
    print("=" * 50)

    try:
        from src.training.expert_trainer import ChessExpertTrainer
        import torch

        # Initialize with minimal settings
        trainer = ChessExpertTrainer()

        print("✅ Expert trainer initialized")

        # Test data loading with very small dataset
        print("📚 Loading minimal dataset...")
        train_dataset, eval_dataset = trainer.prepare_expert_data('uci')
        print(f"✅ Loaded {len(train_dataset)} train, {len(eval_dataset)} eval examples")

        # Test model loading
        print("🤖 Loading model...")
        trainer.initialize_training_environment()
        print("✅ Model and tokenizer loaded")

        # Test tokenization with very small batch
        print("🔄 Testing tokenization...")
        def tokenize_function(examples):
            return trainer.tokenizer(
                examples['text'],
                truncation=True,
                padding='max_length',
                max_length=128  # Very short for testing
            )

        # Only tokenize first 5 examples
        small_train = train_dataset.select(range(min(5, len(train_dataset))))
        tokenized = small_train.map(tokenize_function, batched=True)
        print("✅ Tokenization successful")

        # Test forward pass only (no training)
        print("🔥 Testing forward pass...")
        with torch.no_grad():
            # Get first example and convert to tensors
            first_example = tokenized[0]
            sample_batch = {}
            for k, v in first_example.items():
                if isinstance(v, (int, float)):  # Convert scalars to tensors
                    sample_batch[k] = torch.tensor([v]).to(trainer.base_model.device)
                elif hasattr(v, 'to'):  # Tensor-like objects
                    sample_batch[k] = v.to(trainer.base_model.device)
                else:  # Skip non-tensor data
                    continue

            outputs = trainer.base_model(**sample_batch)
            print("✅ Forward pass successful")
            print(f"   Output shape: {outputs.logits.shape}")

        print("\n🎉 All minimal tests passed!")
        print("The issue is likely in the training loop, not model/data loading.")

        return True

    except Exception as e:
        print(f"❌ Test failed at: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_minimal_training()
    if success:
        print("\n💡 Next: Try running the actual training with minimal settings")
        print("     The hang is likely in the training loop itself")
    else:
        print("\n🚨 The issue is in the basic setup, not training loop")
    sys.exit(0 if success else 1)
