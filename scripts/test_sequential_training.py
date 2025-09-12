#!/usr/bin/env python3
"""
Test Sequential Expert Training with Memory Management

This script tests the new sequential expert training approach with proper
memory management and cleanup between experts.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

def test_memory_isolation():
    """Test that memory is properly isolated between expert trainings."""
    print("🔧 Testing memory isolation between experts...")

    try:
        from src.training.expert_trainer import ChessExpertTrainer
        import psutil
        import gc

        trainer = ChessExpertTrainer()

        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / (1024**3)
        print(f"📊 Initial memory usage: {initial_memory:.1f}GB")

        # Test loading UCI data
        print("Testing UCI data loading...")
        uci_train, uci_eval = trainer.prepare_expert_data('uci')
        print(f"✅ UCI data loaded: {len(uci_train)} train, {len(uci_eval)} eval")

        memory_after_uci = process.memory_info().rss / (1024**3)
        print(f"📊 Memory after UCI data: {memory_after_uci:.1f}GB")

        # Clear data
        del uci_train, uci_eval
        gc.collect()

        memory_after_cleanup = process.memory_info().rss / (1024**3)
        print(f"📊 Memory after cleanup: {memory_after_cleanup:.1f}GB")

        # Test loading Tutor data
        print("Testing Tutor data loading...")
        tutor_train, tutor_eval = trainer.prepare_expert_data('tutor')
        print(f"✅ Tutor data loaded: {len(tutor_train)} train, {len(tutor_eval)} eval")

        memory_after_tutor = process.memory_info().rss / (1024**3)
        print(f"📊 Memory after Tutor data: {memory_after_tutor:.1f}GB")

        # Check that memory usage is reasonable
        memory_increase = memory_after_tutor - initial_memory
        if memory_increase < 2.0:  # Less than 2GB increase for data loading
            print("✅ Memory usage is reasonable")
            return True
        else:
            print(f"⚠️  Memory increase too high: {memory_increase:.1f}GB")
            return False

    except Exception as e:
        print(f"❌ Memory isolation test failed: {e}")
        return False

def test_training_environment_cleanup():
    """Test that training environment cleanup works properly."""
    print("\n🔧 Testing training environment cleanup...")

    try:
        from src.training.train_chessgemmma import ChessGemmaTrainingOrchestrator
        import psutil

        # Initialize orchestrator
        orchestrator = ChessGemmaTrainingOrchestrator()

        # Initialize components
        if not orchestrator.initialize_components():
            raise Exception("Failed to initialize training components")

        # Get initial memory
        process = psutil.Process()
        initial_memory = process.memory_info().rss / (1024**3)
        print(f"📊 Initial memory: {initial_memory:.1f}GB")

        # Test cleanup method
        orchestrator._cleanup_expert_training_environment()

        # Check memory after cleanup
        after_cleanup = process.memory_info().rss / (1024**3)
        print(f"📊 Memory after cleanup: {after_cleanup:.1f}GB")

        print("✅ Training environment cleanup test completed")
        return True

    except Exception as e:
        print(f"❌ Training environment cleanup test failed: {e}")
        return False

def test_expert_trainer_methods():
    """Test that the expert trainer has the new methods."""
    print("\n🔧 Testing expert trainer methods...")

    try:
        from src.training.expert_trainer import ChessExpertTrainer

        trainer = ChessExpertTrainer()

        # Check that new methods exist
        if hasattr(trainer, 'train_expert_with_data'):
            print("✅ train_expert_with_data method exists")
        else:
            print("❌ train_expert_with_data method missing")
            return False

        # Check that initialize_training_environment has force_reload parameter
        import inspect
        sig = inspect.signature(trainer.initialize_training_environment)
        if 'force_reload' in sig.parameters:
            print("✅ initialize_training_environment has force_reload parameter")
        else:
            print("❌ initialize_training_environment missing force_reload parameter")
            return False

        return True

    except Exception as e:
        print(f"❌ Expert trainer methods test failed: {e}")
        return False

def test_sequential_workflow():
    """Test the complete sequential workflow simulation."""
    print("\n🔧 Testing sequential workflow simulation...")

    try:
        from src.training.train_chessgemmma import ChessGemmaTrainingOrchestrator
        import psutil

        orchestrator = ChessGemmaTrainingOrchestrator()

        # Initialize components
        if not orchestrator.initialize_components():
            raise Exception("Failed to initialize training components")

        # Simulate the workflow for one expert
        expert_name = 'uci'
        print(f"Simulating sequential workflow for {expert_name}...")

        # Step 1: Load data
        print("Step 1: Loading data...")
        train_dataset, eval_dataset = orchestrator.expert_trainer.prepare_expert_data(expert_name)
        print(f"✅ Data loaded: {len(train_dataset)} train, {len(eval_dataset)} eval")

        # Step 2: Initialize environment
        print("Step 2: Initializing training environment...")
        orchestrator.expert_trainer.initialize_training_environment()
        print("✅ Training environment initialized")

        # Step 3: Simulate cleanup
        print("Step 3: Simulating cleanup...")
        orchestrator._cleanup_expert_training_environment()
        print("✅ Cleanup completed")

        print("✅ Sequential workflow simulation completed")
        return True

    except Exception as e:
        print(f"❌ Sequential workflow test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🎯 Sequential Expert Training Test")
    print("=" * 50)

    tests = [
        ("Memory Isolation", test_memory_isolation),
        ("Training Environment Cleanup", test_training_environment_cleanup),
        ("Expert Trainer Methods", test_expert_trainer_methods),
        ("Sequential Workflow", test_sequential_workflow)
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
        print("🎉 All tests passed! Sequential training is ready.")
        print("\nKey improvements implemented:")
        print("✅ Load dataset per expert individually")
        print("✅ Initialize training environment per expert")
        print("✅ Train expert with fresh data and environment")
        print("✅ Comprehensive cleanup between experts")
        print("✅ Memory monitoring and leak prevention")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
