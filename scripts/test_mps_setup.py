#!/usr/bin/env python3
"""
MPS Setup and Training Component Test

This script tests the MPS setup and training components to ensure they work
before running the full training pipeline.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

def test_mps_availability():
    """Test MPS availability and basic functionality."""
    print("🔧 Testing MPS availability...")

    try:
        import torch

        print(f"PyTorch version: {torch.__version__}")
        print(f"MPS available: {torch.backends.mps.is_available()}")
        print(f"CUDA available: {torch.cuda.is_available()}")

        if torch.backends.mps.is_available():
            device = torch.device("mps")
            print(f"MPS device: {device}")

            # Test basic tensor operations
            x = torch.randn(10, 10).to(device)
            y = torch.randn(10, 10).to(device)
            z = x + y
            print(f"✅ Basic MPS operations work: {z.shape}")

            # Test memory operations
            torch.mps.empty_cache()
            print("✅ MPS cache operations work")

            return True
        else:
            print("⚠️  MPS not available")
            return False

    except Exception as e:
        print(f"❌ MPS test failed: {e}")
        return False

def test_mps_optimizer():
    """Test MPS optimizer functionality."""
    print("\n🔧 Testing MPS optimizer...")

    try:
        from src.training.mps_optimizer import MPSMemoryOptimizer

        optimizer = MPSMemoryOptimizer()
        print("✅ MPS optimizer initialized")

        # Test memory monitoring (should not fail)
        memory_info = optimizer.monitor_memory_usage()
        print(f"✅ Memory monitoring works: {memory_info['memory_utilization']:.1%}")

        # Test conservative batch sizing
        batch_info = optimizer.calculate_optimal_batch_size(None, None)
        print(f"✅ Batch sizing works: {batch_info['recommended_batch_size']}")

        return True

    except Exception as e:
        print(f"❌ MPS optimizer test failed: {e}")
        return False

def test_expert_trainer_initialization():
    """Test expert trainer initialization without full training."""
    print("\n🔧 Testing expert trainer initialization...")

    try:
        from src.training.expert_trainer import ChessExpertTrainer

        # Test basic initialization without model loading
        trainer = ChessExpertTrainer()

        # Check expert configurations
        configs = trainer.expert_configs
        print(f"✅ Expert configurations loaded: {list(configs.keys())}")

        # Verify conservative batch sizes
        for expert_name, config in configs.items():
            batch_size = config.training_params['batch_size']
            grad_accum = config.training_params['gradient_accumulation_steps']
            print(f"✅ {expert_name}: batch_size={batch_size}, grad_accum={grad_accum}")

        return True

    except Exception as e:
        print(f"❌ Expert trainer test failed: {e}")
        return False

def test_data_loading():
    """Test data loading with memory limits."""
    print("\n🔧 Testing data loading...")

    try:
        from src.training.expert_trainer import ChessExpertTrainer

        trainer = ChessExpertTrainer()

        # Test data preparation for UCI expert (should be memory efficient)
        print("Testing UCI data preparation...")
        train_dataset, eval_dataset = trainer.prepare_expert_data('uci')

        print(f"✅ UCI data loaded: {len(train_dataset)} train, {len(eval_dataset)} eval")

        # Verify reasonable dataset sizes
        if len(train_dataset) > 0 and len(train_dataset) <= 200:
            print("✅ Dataset size is reasonable for memory efficiency")
        else:
            print(f"⚠️  Dataset size may be too large: {len(train_dataset)}")

        return True

    except Exception as e:
        print(f"❌ Data loading test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🎯 MPS Setup and Training Component Test")
    print("=" * 50)

    tests = [
        ("MPS Availability", test_mps_availability),
        ("MPS Optimizer", test_mps_optimizer),
        ("Expert Trainer", test_expert_trainer_initialization),
        ("Data Loading", test_data_loading)
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name} test...")
        if test_func():
            passed += 1
            print(f"✅ {test_name} test PASSED")
        else:
            print(f"❌ {test_name} test FAILED")

    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Ready for training.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
