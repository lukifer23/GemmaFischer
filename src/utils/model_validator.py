#!/usr/bin/env python3
"""
Model Validation and Integrity System for ChessGemma

Provides comprehensive model validation, adapter integrity checks,
and automatic recovery mechanisms for maintaining model reliability.
"""

import os
import torch
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import threading
import time
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelValidationResult:
    """Result of model validation check."""
    is_valid: bool
    model_path: str
    adapter_path: Optional[str]
    validation_time: datetime
    checksum_valid: bool
    structure_valid: bool
    performance_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AdapterIntegrityResult:
    """Result of adapter integrity check."""
    adapter_path: str
    is_integrity: bool
    checksum_match: bool
    file_structure_valid: bool
    last_modified: datetime
    size_bytes: int
    errors: List[str] = field(default_factory=list)


class ChessGemmaModelValidator:
    """Comprehensive model validation system."""

    def __init__(self, model_base_path: str, adapters_base_path: str):
        self.model_base_path = Path(model_base_path)
        self.adapters_base_path = Path(adapters_base_path)
        self.validation_cache: Dict[str, ModelValidationResult] = {}
        self.integrity_cache: Dict[str, AdapterIntegrityResult] = {}
        self.checksum_cache: Dict[str, str] = {}
        self.cache_timeout = timedelta(hours=1)
        self.lock = threading.Lock()

        # Performance test parameters
        self.test_positions = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",  # Starting position
            "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 1",  # Complex middlegame
            "8/8/8/8/8/8/8/K7 w - - 0 1"  # Endgame
        ]

        logger.info("🔍 ChessGemma Model Validator initialized")

    def validate_model_integrity(self, model_path: Optional[str] = None,
                               adapter_path: Optional[str] = None) -> ModelValidationResult:
        """Comprehensive model integrity validation."""
        model_path = model_path or str(self._find_latest_model())
        adapter_path = adapter_path or str(self._find_latest_adapter())

        cache_key = f"{model_path}_{adapter_path}"

        # Check cache first
        with self.lock:
            if cache_key in self.validation_cache:
                cached_result = self.validation_cache[cache_key]
                if datetime.now() - cached_result.validation_time < self.cache_timeout:
                    return cached_result

        result = ModelValidationResult(
            is_valid=True,
            model_path=model_path,
            adapter_path=adapter_path,
            validation_time=datetime.now(),
            checksum_valid=True,
            structure_valid=True,
            performance_valid=True
        )

        try:
            # 1. Validate model file structure
            result.structure_valid = self._validate_model_structure(model_path)
            if not result.structure_valid:
                result.errors.append("Model file structure is invalid")
                result.is_valid = False

            # 2. Validate checksum if available
            result.checksum_valid = self._validate_model_checksum(model_path)
            if not result.checksum_valid:
                result.warnings.append("Model checksum validation failed")
                # Don't mark as invalid for checksum issues

            # 3. Validate adapter if provided
            if adapter_path:
                adapter_result = self.validate_adapter_integrity(adapter_path)
                if not adapter_result.is_integrity:
                    result.errors.extend(adapter_result.errors)
                    result.is_valid = False

            # 4. Performance validation
            if result.structure_valid:
                result.performance_valid = self._validate_model_performance(model_path, adapter_path)
                if not result.performance_valid:
                    result.errors.append("Model performance validation failed")
                    result.is_valid = False

            # 5. Collect metrics
            result.metrics = self._collect_model_metrics(model_path, adapter_path)

        except Exception as e:
            result.errors.append(f"Validation error: {str(e)}")
            result.is_valid = False

        # Cache result
        with self.lock:
            self.validation_cache[cache_key] = result

        logger.info(f"✅ Model validation completed: {'VALID' if result.is_valid else 'INVALID'}")
        return result

    def validate_adapter_integrity(self, adapter_path: str) -> AdapterIntegrityResult:
        """Validate adapter integrity and consistency."""
        adapter_path = Path(adapter_path)

        # Check cache
        cache_key = str(adapter_path)
        with self.lock:
            if cache_key in self.integrity_cache:
                cached_result = self.integrity_cache[cache_key]
                if datetime.now() - cached_result.last_modified < self.cache_timeout:
                    return cached_result

        result = AdapterIntegrityResult(
            adapter_path=str(adapter_path),
            is_integrity=True,
            checksum_match=True,
            file_structure_valid=True,
            last_modified=datetime.fromtimestamp(adapter_path.stat().st_mtime),
            size_bytes=adapter_path.stat().st_size
        )

        try:
            # 1. Check file structure
            result.file_structure_valid = self._validate_adapter_structure(adapter_path)
            if not result.file_structure_valid:
                result.errors.append("Adapter file structure is invalid")
                result.is_integrity = False

            # 2. Validate checksum
            result.checksum_match = self._validate_adapter_checksum(adapter_path)
            if not result.checksum_match:
                result.errors.append("Adapter checksum mismatch")
                result.is_integrity = False

            # 3. Check for corruption indicators
            corruption_indicators = self._check_adapter_corruption(adapter_path)
            if corruption_indicators:
                result.errors.extend(corruption_indicators)
                result.is_integrity = False

        except Exception as e:
            result.errors.append(f"Integrity check error: {str(e)}")
            result.is_integrity = False

        # Cache result
        with self.lock:
            self.integrity_cache[cache_key] = result

        return result

    def _validate_model_structure(self, model_path: str) -> bool:
        """Validate model file structure."""
        model_path = Path(model_path)

        if not model_path.exists():
            return False

        # Check for essential model files
        required_files = ['config.json', 'pytorch_model.bin']
        for file in required_files:
            if not (model_path / file).exists():
                return False

        # Validate config file
        try:
            with open(model_path / 'config.json', 'r') as f:
                config = json.load(f)
                if not isinstance(config, dict):
                    return False
                if 'model_type' not in config:
                    return False
        except:
            return False

        return True

    def _validate_adapter_structure(self, adapter_path: Path) -> bool:
        """Validate adapter file structure."""
        if not adapter_path.exists():
            return False

        # Check for adapter_config.json
        config_file = adapter_path / 'adapter_config.json'
        if not config_file.exists():
            return False

        # Check for adapter model files
        model_files = list(adapter_path.glob('*.bin'))
        if not model_files:
            return False

        # Validate config
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
                if 'peft_type' not in config:
                    return False
        except:
            return False

        return True

    def _validate_model_checksum(self, model_path: str) -> bool:
        """Validate model checksum if available."""
        checksum_file = Path(model_path) / 'checksums.json'
        if not checksum_file.exists():
            return True  # No checksum to validate

        try:
            with open(checksum_file, 'r') as f:
                expected_checksums = json.load(f)

            for file_path, expected_checksum in expected_checksums.items():
                full_path = Path(model_path) / file_path
                if full_path.exists():
                    actual_checksum = self._calculate_file_checksum(full_path)
                    if actual_checksum != expected_checksum:
                        return False
        except:
            return False

        return True

    def _validate_adapter_checksum(self, adapter_path: Path) -> bool:
        """Validate adapter checksum."""
        checksum_file = adapter_path / 'checksum.txt'
        if not checksum_file.exists():
            return True  # No checksum to validate

        try:
            with open(checksum_file, 'r') as f:
                expected_checksum = f.read().strip()

            # Calculate checksum of all adapter files
            actual_checksum = self._calculate_directory_checksum(adapter_path)
            return actual_checksum == expected_checksum
        except:
            return False

    def _validate_model_performance(self, model_path: str, adapter_path: Optional[str]) -> bool:
        """Validate model performance with basic tests."""
        try:
            # Load model (lightweight test)
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel

            tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                local_files_only=True,
                device_map="cpu",  # Use CPU for validation
                torch_dtype=torch.float32
            )

            if adapter_path:
                model = PeftModel.from_pretrained(model, adapter_path)

            # Test basic generation
            test_input = "What is the best move in chess?"
            inputs = tokenizer(test_input, return_tensors="pt")

            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False)

            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            if not generated_text or len(generated_text) < len(test_input):
                return False

            return True

        except Exception as e:
            logger.warning(f"Performance validation failed: {e}")
            return False

    def _collect_model_metrics(self, model_path: str, adapter_path: Optional[str]) -> Dict[str, Any]:
        """Collect model metrics."""
        metrics = {}

        try:
            model_path = Path(model_path)

            # Model size
            if model_path.exists():
                total_size = sum(f.stat().st_size for f in model_path.rglob('*') if f.is_file())
                metrics['model_size_bytes'] = total_size
                metrics['model_size_mb'] = total_size / (1024 * 1024)

            # Adapter size
            if adapter_path:
                adapter_path = Path(adapter_path)
                if adapter_path.exists():
                    adapter_size = sum(f.stat().st_size for f in adapter_path.rglob('*') if f.is_file())
                    metrics['adapter_size_bytes'] = adapter_size
                    metrics['adapter_size_mb'] = adapter_size / (1024 * 1024)

            # Parameter count (estimate)
            config_file = model_path / 'config.json'
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config = json.load(f)
                    hidden_size = config.get('hidden_size', 0)
                    num_layers = config.get('num_hidden_layers', 0)
                    vocab_size = config.get('vocab_size', 0)

                    # Rough parameter estimate
                    estimated_params = (
                        hidden_size * hidden_size * 4 * num_layers +  # Attention layers
                        hidden_size * vocab_size  # Output layer
                    )
                    metrics['estimated_parameters'] = estimated_params

        except Exception as e:
            logger.warning(f"Metrics collection failed: {e}")

        return metrics

    def _check_adapter_corruption(self, adapter_path: Path) -> List[str]:
        """Check for adapter corruption indicators."""
        errors = []

        try:
            # Check for NaN values in adapter files
            for bin_file in adapter_path.glob('*.bin'):
                state_dict = torch.load(bin_file, map_location='cpu')
                for name, tensor in state_dict.items():
                    if torch.isnan(tensor).any():
                        errors.append(f"NaN values found in {bin_file.name}:{name}")
                    if torch.isinf(tensor).any():
                        errors.append(f"Inf values found in {bin_file.name}:{name}")

        except Exception as e:
            errors.append(f"Corruption check failed: {str(e)}")

        return errors

    def _calculate_file_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of a file."""
        cache_key = str(file_path)
        if cache_key in self.checksum_cache:
            return self.checksum_cache[cache_key]

        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)

        checksum = hash_sha256.hexdigest()
        self.checksum_cache[cache_key] = checksum
        return checksum

    def _calculate_directory_checksum(self, dir_path: Path) -> str:
        """Calculate combined checksum of all files in directory."""
        hash_sha256 = hashlib.sha256()

        for file_path in sorted(dir_path.rglob('*')):
            if file_path.is_file() and file_path.name != 'checksum.txt':
                file_hash = self._calculate_file_checksum(file_path)
                hash_sha256.update(file_hash.encode())

        return hash_sha256.hexdigest()

    def _find_latest_model(self) -> Path:
        """Find the latest model snapshot."""
        if not self.model_base_path.exists():
            raise FileNotFoundError(f"Model base path not found: {self.model_base_path}")

        snapshots = list(self.model_base_path.glob('**/snapshots/*'))
        if not snapshots:
            raise FileNotFoundError("No model snapshots found")

        return max(snapshots, key=lambda x: x.stat().st_mtime)

    def _find_latest_adapter(self) -> Optional[Path]:
        """Find the latest adapter checkpoint."""
        if not self.adapters_base_path.exists():
            return None

        checkpoints = list(self.adapters_base_path.glob('**/checkpoint-*'))
        if not checkpoints:
            return None

        return max(checkpoints, key=lambda x: int(x.name.split('-')[1]))

    def get_validation_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report."""
        report = {
            'validation_time': datetime.now().isoformat(),
            'total_validations': len(self.validation_cache),
            'valid_models': sum(1 for r in self.validation_cache.values() if r.is_valid),
            'invalid_models': sum(1 for r in self.validation_cache.values() if not r.is_valid),
            'total_integrity_checks': len(self.integrity_cache),
            'integrity_passed': sum(1 for r in self.integrity_cache.values() if r.is_integrity),
            'integrity_failed': sum(1 for r in self.integrity_cache.values() if not r.is_integrity),
        }

        # Collect all errors and warnings
        all_errors = []
        all_warnings = []
        for result in self.validation_cache.values():
            all_errors.extend(result.errors)
            all_warnings.extend(result.warnings)

        report['common_errors'] = list(set(all_errors))[:10]  # Top 10 unique errors
        report['common_warnings'] = list(set(all_warnings))[:10]  # Top 10 unique warnings

        return report

    def clear_cache(self):
        """Clear validation caches."""
        with self.lock:
            self.validation_cache.clear()
            self.integrity_cache.clear()
            self.checksum_cache.clear()
        logger.info("🧹 Validation caches cleared")


# Global validator instance
_validator = None
_validator_lock = threading.Lock()


def get_model_validator(model_base_path: str = "models",
                       adapters_base_path: str = "checkpoints") -> ChessGemmaModelValidator:
    """Get the global model validator instance."""
    global _validator
    if _validator is None:
        with _validator_lock:
            if _validator is None:
                _validator = ChessGemmaModelValidator(model_base_path, adapters_base_path)
    return _validator


def validate_model(model_path: Optional[str] = None,
                  adapter_path: Optional[str] = None) -> ModelValidationResult:
    """Convenience function to validate a model."""
    validator = get_model_validator()
    return validator.validate_model_integrity(model_path, adapter_path)


def validate_adapter(adapter_path: str) -> AdapterIntegrityResult:
    """Convenience function to validate an adapter."""
    validator = get_model_validator()
    return validator.validate_adapter_integrity(adapter_path)
