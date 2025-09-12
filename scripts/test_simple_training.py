#!/usr/bin/env python3
"""
Simple Training Loop Test - Isolate the hanging issue

This script tests the training loop directly without the full orchestrator
to isolate exactly where the hang is occurring.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

def test_simple_training_loop():
    """Test just the training loop with minimal setup."""
    print("🔬 Simple Training Loop Test")
    print("=" * 50)

    try:
        from src.training.expert_trainer import ChessExpertTrainer
        from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
        import torch
        import time

        # Initialize trainer
        trainer = ChessExpertTrainer()

        print("✅ Expert trainer initialized")

        # Load minimal data
        print("📚 Loading minimal dataset...")
        train_dataset, eval_dataset = trainer.prepare_expert_data('uci')
        print(f"✅ Loaded {len(train_dataset)} train examples")

        # Initialize model
        print("🤖 Loading model...")
        trainer.initialize_training_environment()
        print("✅ Model loaded")

        # Tokenize minimal dataset
        print("🔄 Tokenizing data...")
        def tokenize_function(examples):
            return trainer.tokenizer(
                examples['text'],
                truncation=True,
                padding='max_length',
                max_length=256
            )

        # Only use first 10 examples for testing
        small_train = train_dataset.select(range(min(10, len(train_dataset))))
        tokenized_train = small_train.map(tokenize_function, batched=True, remove_columns=['text'])
        print("✅ Data tokenized")

        # Create minimal training arguments
        training_args = TrainingArguments(
            output_dir="./test_output",
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,  # No accumulation for testing
            max_steps=5,  # Just 5 steps
            logging_steps=1,
            save_steps=10,  # Won't save during test
            evaluation_strategy="no",  # Disable evaluation
            fp16=False,
            dataloader_num_workers=0,
            remove_unused_columns=True,
        )

        # Create data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=trainer.tokenizer,
            mlm=False
        )

        # Create simple trainer
        simple_trainer = Trainer(
            model=trainer.base_model,
            args=training_args,
            train_dataset=tokenized_train,
            data_collator=data_collator,
        )

        print("🚀 Starting minimal training...")
        start_time = time.time()

        # Test just the training step
        print("⚡ Testing first training step...")
        try:
            train_result = simple_trainer.train()
            end_time = time.time()
            print(f"✅ Training completed in {end_time - start_time:.2f}s")
            print("🎉 No hang detected!")
            return True

        except Exception as e:
            end_time = time.time()
            print(f"❌ Training failed after {end_time - start_time:.2f}s: {e}")
            import traceback
            traceback.print_exc()
            return False

    except Exception as e:
        print(f"❌ Setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_simple_training_loop()
    if success:
        print("\n💡 The training loop works! The hang might be in:")
        print("   - Full orchestrator setup")
        print("   - Memory management between steps")
        print("   - Progress callbacks")
        print("   - Dataset size or complexity")
    else:
        print("\n🚨 The issue is in the basic training loop")
    sys.exit(0 if success else 1)
