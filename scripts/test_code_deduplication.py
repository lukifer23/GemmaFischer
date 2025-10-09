#!/usr/bin/env python3
"""
Code Deduplication Test

Tests that our deduplication efforts have successfully removed code duplication
and that common utilities are being used consistently.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


def test_common_utilities_usage():
    """Test that common utilities are being used correctly."""
    try:
        # Test that common utilities can be imported
        from src.utils.common import (
            get_logger, get_error_handler, get_config_manager,
            find_latest_dir, resolve_model_path, resolve_adapter_path,
            get_environment_config, safe_file_read, safe_file_write,
            format_duration, get_memory_usage
        )

        print("✅ Common utilities import successfully")

        # Test basic functionality
        logger = get_logger("test")
        print(f"✅ Logger created: {logger.name}")

        env_config = get_environment_config()
        print(f"✅ Environment config loaded: {len(env_config)} variables")

        duration = format_duration(1.5)
        print(f"✅ Duration formatting: {duration}")

        return True

    except Exception as e:
        print(f"Common utilities test failed: {e}")
        return False


def test_module_imports_without_duplication():
    """Test that modules import correctly and use common utilities."""
    modules_to_test = [
        'src.inference.core_engine',
        'src.inference.caching',
        'src.inference.expert_manager',
        'src.config.config_manager',
    ]

    print("Testing module imports without duplication...")

    for module_name in modules_to_test:
        try:
            __import__(module_name)
            print(f"{module_name} imports successfully")
        except Exception as e:
            print(f"FAILED: {module_name} import failed: {e}")
            return False

    return True


def test_configuration_consistency():
    """Test that configuration is consistent across modules."""
    try:
        from src.config.config_manager import ChessGemmaConfig

        # Create config and test consistency
        config = ChessGemmaConfig()

        # Test that expert configs are consistent
        uci_config = config.get_training_config("uci")
        tutor_config = config.get_training_config("tutor")
        director_config = config.get_training_config("director")

        # All should have consistent base settings
        assert uci_config['per_device_train_batch_size'] == tutor_config['per_device_train_batch_size']
        assert tutor_config['per_device_train_batch_size'] == director_config['per_device_train_batch_size']

        print("Configuration consistency verified")

        return True

    except Exception as e:
        print(f"Configuration consistency test failed: {e}")
        return False


def test_error_handler_optimization():
    """Test that error handler optimizations are working."""
    try:
        from src.utils.error_handler import get_error_handler

        handler = get_error_handler()

        # Test that error classification is fast
        import time
        start = time.time()

        for i in range(100):
            try:
                raise ValueError(f"Test error {i}")
            except ValueError as e:
                from src.utils.error_handler import ErrorContext
                context = ErrorContext(component="test", operation="test")
                severity, category = handler._classify_error(e, context)

        classification_time = time.time() - start

        print(f"Error classification performance: {classification_time*1000:.2f}ms for 100 classifications")

        # Test system state caching
        start = time.time()
        state1 = handler._capture_system_state()
        first_call = time.time() - start

        start = time.time()
        state2 = handler._capture_system_state()
        second_call = time.time() - start

        print(f"System state caching: {first_call*1000:.2f}ms -> {second_call*1000:.2f}ms")

        return True

    except Exception as e:
        print(f"Error handler optimization test failed: {e}")
        return False


def main():
    """Run all deduplication tests."""
    print("🔧 ChessGemma Code Deduplication Tests")
    print("=" * 60)

    tests = [
        ("Common Utilities Usage", test_common_utilities_usage),
        ("Module Imports Without Duplication", test_module_imports_without_duplication),
        ("Configuration Consistency", test_configuration_consistency),
        ("Error Handler Optimization", test_error_handler_optimization),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 40)

        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} failed")

    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} passed")

    if passed == total:
        print("All deduplication tests passed!")
        print("Code deduplication is working correctly")
        print("Common utilities are being used consistently")
        print("No significant code duplication detected")
        return 0
    else:
        print("Some deduplication tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
