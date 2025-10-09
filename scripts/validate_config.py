#!/usr/bin/env python3
"""
Configuration Validation Script

Validates the unified configuration system and ensures all settings are correct.
Tests configuration loading, validation, and expert-specific configurations.
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


def test_config_loading():
    """Test configuration loading from file."""
    try:
        from src.config.config_manager import ChessGemmaConfig, get_config

        print("Testing configuration loading...")

        # Test loading default config
        config = get_config("chessgemma_unified")
        print(f"✓ Loaded unified configuration: {config}")

        # Test configuration validation
        errors = config.validate()
        if errors:
            print(f"✗ Configuration validation errors: {errors}")
            return False
        else:
            print("✓ Configuration validation passed")

        # Test expert configurations
        for expert in ["uci", "tutor", "director"]:
            training_config = config.get_training_config(expert)
            lora_config = config.get_lora_config()
            inference_config = config.get_inference_config(expert)

            print(f"✓ {expert.capitalize()} expert config loaded:")
            print(f"  - Training: max_steps={training_config.get('max_steps', 'N/A')}")
            print(f"  - LoRA: r={lora_config.get('r', 'N/A')}")
            print(f"  - Inference: temperature={inference_config.get('temperature', 'N/A')}")

        return True

    except Exception as e:
        print(f"✗ Configuration loading failed: {e}")
        return False


def test_environment_overrides():
    """Test environment variable overrides."""
    try:
        import os
        from src.config.config_manager import ChessGemmaConfig

        print("Testing environment overrides...")

        # Set test environment variables
        test_env = {
            "CHESSGEMMA_MODEL_ID": "test-model-id",
            "CHESSGEMMA_DEBUG": "1",
            "CHESSGEMMA_TIMEOUT_MINUTES": "120",
            "CHESSGEMMA_CACHE_SIZE": "256"
        }

        # Save original env values
        original_env = {}
        for key in test_env:
            original_env[key] = os.environ.get(key)

        # Set test values
        for key, value in test_env.items():
            os.environ[key] = value

        try:
            # Create new config to test overrides
            config = ChessGemmaConfig()

            # Check if overrides were applied
            assert config.model.pretrained_model_path == "test-model-id"
            assert config.system.debug_mode == True
            assert config.system.timeout_minutes == 120
            assert config.cache.max_cache_size == 256

            print("✓ Environment overrides applied correctly")

        finally:
            # Restore original env values
            for key, original_value in original_env.items():
                if original_value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = original_value

        return True

    except Exception as e:
        print(f"✗ Environment override test failed: {e}")
        return False


def test_config_serialization():
    """Test configuration serialization and deserialization."""
    try:
        from src.config.config_manager import ChessGemmaConfig

        print("Testing configuration serialization...")

        # Create a config
        config = ChessGemmaConfig()

        # Modify some values
        config.training.max_steps = 500
        config.lora.r = 8
        config.system.debug_mode = True

        # Save to file
        test_file = project_root / "test_config.yaml"
        config.save_to_file(test_file)

        # Load from file
        loaded_config = ChessGemmaConfig.load_from_file(test_file)

        # Verify values match
        assert loaded_config.training.max_steps == 500
        assert loaded_config.lora.r == 8
        assert loaded_config.system.debug_mode == True

        print("✓ Configuration serialization/deserialization works")

        # Clean up
        if test_file.exists():
            test_file.unlink()

        return True

    except Exception as e:
        print(f"✗ Configuration serialization test failed: {e}")
        return False


def main():
    """Run all configuration validation tests."""
    print("🔧 ChessGemma Configuration Validation")
    print("=" * 50)

    tests = [
        ("Configuration Loading", test_config_loading),
        ("Environment Overrides", test_environment_overrides),
        ("Configuration Serialization", test_config_serialization),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 30)

        if test_func():
            passed += 1
        else:
            print(f"FAILED: {test_name} failed")

    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} passed")

    if passed == total:
        print("All configuration tests passed!")
        return 0
    else:
        print("❌ Some configuration tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
