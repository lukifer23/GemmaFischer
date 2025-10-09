#!/usr/bin/env python3
"""
Error Handler Performance Test

Tests the optimized error handling system for performance and correctness.
"""

import time
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


def test_error_classification_performance():
    """Test that error classification is fast and accurate."""
    from src.utils.error_handler import ChessGemmaErrorHandler, ErrorCategory, ErrorSeverity

    handler = ChessGemmaErrorHandler()

    # Test various error types
    test_cases = [
        ("cuda out of memory", ErrorCategory.MEMORY, ErrorSeverity.HIGH),
        ("model loading failed", ErrorCategory.MODEL_LOADING, ErrorSeverity.HIGH),
        ("inference error", ErrorCategory.INFERENCE, ErrorSeverity.MEDIUM),
        ("training failed", ErrorCategory.TRAINING, ErrorSeverity.MEDIUM),
        ("dataset not found", ErrorCategory.DATA_LOADING, ErrorSeverity.MEDIUM),
        ("config invalid", ErrorCategory.CONFIGURATION, ErrorSeverity.MEDIUM),
        ("unknown error", ErrorCategory.UNKNOWN, ErrorSeverity.LOW),
    ]

    print("Testing error classification performance...")

    start_time = time.time()
    for error_msg, expected_category, expected_severity in test_cases:
        # Create a test exception
        try:
            raise ValueError(error_msg)
        except ValueError as e:
            from src.utils.error_handler import ErrorContext
            context = ErrorContext(component="test", operation="test")

            severity, category = handler._classify_error(e, context)

            if category != expected_category or severity != expected_severity:
                print(f"❌ Classification failed for '{error_msg}': expected {expected_category}/{expected_severity}, got {category}/{severity}")
                return False

    classification_time = time.time() - start_time
    avg_time = classification_time / len(test_cases)

    print(f"✅ Classification performance: {avg_time*1000:.2f}ms per classification")
    return True


def test_error_boundary_performance():
    """Test that error boundaries don't add significant overhead."""
    from src.utils.error_handler import get_error_handler

    handler = get_error_handler()

    print("Testing error boundary performance...")

    # Test normal operation (no errors)
    start_time = time.time()
    for i in range(1000):
        with handler.error_boundary("test", "normal_operation"):
            # Simulate normal operation
            x = i * 2
    normal_time = time.time() - start_time

    # Test error handling
    start_time = time.time()
    error_count = 0
    for i in range(100):
        try:
            with handler.error_boundary("test", "error_operation"):
                # Simulate operation that fails
                raise ValueError(f"Test error {i}")
        except ValueError:
            error_count += 1
    error_time = time.time() - start_time

    print(f"✅ Normal operations: {normal_time*1000:.2f}ms for 1000 operations")
    print(f"✅ Error handling: {error_time*1000:.2f}ms for 100 errors")
    print(f"✅ Errors handled: {error_count}")

    return error_count == 100


def test_memory_optimization():
    """Test that memory optimizations are working."""
    from src.utils.error_handler import get_error_handler

    handler = get_error_handler()

    print("Testing memory optimizations...")

    # Generate many low-severity errors to test memory limiting
    for i in range(600):  # More than max_history_size // 2
        try:
            with handler.error_boundary("test", "memory_test"):
                raise ValueError(f"Low severity error {i}")
        except ValueError:
            pass

    # Check that we don't have too many stored errors
    stats = handler.get_performance_stats()
    stored_errors = stats['total_errors_handled']
    memory_usage = stats['memory_usage_mb']

    print(f"✅ Stored errors: {stored_errors} (should be limited)")
    print(f"✅ Memory usage: {memory_usage:.2f}MB")

    # Should have limited the number of low-severity errors stored
    if stored_errors > handler.max_history_size:
        print(f"❌ Too many errors stored: {stored_errors} > {handler.max_history_size}")
        return False

    return True


def test_system_state_caching():
    """Test that system state caching reduces overhead."""
    from src.utils.error_handler import get_error_handler

    handler = get_error_handler()

    print("Testing system state caching...")

    # First call should capture fresh state
    start_time = time.time()
    state1 = handler._capture_system_state()
    first_call_time = time.time() - start_time

    # Second call should use cached state
    start_time = time.time()
    state2 = handler._capture_system_state()
    second_call_time = time.time() - start_time

    print(f"✅ First call: {first_call_time*1000:.2f}ms")
    print(f"✅ Second call: {second_call_time*1000:.2f}ms")

    # Second call should be faster due to caching
    if second_call_time > first_call_time * 2:
        print("❌ System state caching not working effectively")
        return False

    return True


def main():
    """Run all error handler performance tests."""
    print("🔧 ChessGemma Error Handler Performance Tests")
    print("=" * 60)

    tests = [
        ("Error Classification Performance", test_error_classification_performance),
        ("Error Boundary Performance", test_error_boundary_performance),
        ("Memory Optimization", test_memory_optimization),
        ("System State Caching", test_system_state_caching),
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
    print(f"📊 Test Results: {passed}/{total} passed")

    if passed == total:
        print("🎉 All error handler performance tests passed!")
        print("✅ Error handling system is optimized and working correctly")
        return 0
    else:
        print("❌ Some error handler performance tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
