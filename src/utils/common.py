#!/usr/bin/env python3
"""
Common Utilities for ChessGemma

Consolidates frequently used patterns and utilities across the codebase to reduce
duplication and improve maintainability.
"""

from __future__ import annotations

import os
import glob
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Union
from contextlib import contextmanager
import logging


# Project root for relative path resolution
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def safe_import(module_name: str, fallback_value=None):
    """Safely import a module with fallback."""
    try:
        return __import__(module_name)
    except ImportError:
        return fallback_value


def conditional_import(module_name: str, attr_name: str = None):
    """Conditionally import a module and return specific attribute."""
    try:
        module = __import__(module_name, fromlist=[attr_name] if attr_name else [])
        if attr_name:
            return getattr(module, attr_name)
        return module
    except (ImportError, AttributeError):
        return None


def get_logger(name: str):
    """Get logger with fallback to basic logging."""
    try:
        from .logging_config import get_logger as get_configured_logger
        return get_configured_logger(name)
    except ImportError:
        return logging.getLogger(name)


def get_error_handler():
    """Get error handler with fallback."""
    try:
        from .error_handler import get_error_handler as get_configured_handler
        return get_configured_handler()
    except ImportError:
        return None


def get_config_manager():
    """Get configuration manager with fallback."""
    try:
        from ..config.config_manager import get_config
        return get_config
    except ImportError:
        return None


def find_latest_dir(patterns: List[str]) -> Optional[Path]:
    """Find the most recently modified directory matching any of the glob patterns."""
    candidates: List[Path] = []
    for pattern in patterns:
        for path in glob.glob(pattern):
            p = Path(path)
            if p.is_dir():
                candidates.append(p)
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def find_latest_file(patterns: List[str]) -> Optional[Path]:
    """Find the most recently modified file matching any of the glob patterns."""
    candidates: List[Path] = []
    for pattern in patterns:
        for path in glob.glob(pattern):
            p = Path(path)
            if p.is_file():
                candidates.append(p)
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def resolve_model_path(model_path: Optional[str] = None) -> str:
    """Resolve model path with environment variable and local path support."""
    if not model_path:
        # Try unified configuration first
        config_manager = get_config_manager()
        if config_manager:
            try:
                config = config_manager()
                if config.model.local_model_path:
                    return config.model.local_model_path
                return config.model.pretrained_model_path
            except:
                pass

        # Fall back to environment variables
        env_override = os.environ.get("CHESSGEMMA_MODEL_PATH")
        if env_override:
            override_path = Path(env_override).expanduser()
            return str(override_path) if override_path.exists() else env_override

        env_model_id = os.environ.get("CHESSGEMMA_MODEL_ID")
        if env_model_id:
            return env_model_id

        # Default to local snapshot or HuggingFace
        base = PROJECT_ROOT / "models" / "google-gemma-3-270m"
        if not base.exists():
            return "google/gemma-3-270m"
        return str(base)

    return model_path


def resolve_adapter_path(project_root: Optional[Path] = None) -> Optional[Path]:
    """Find the latest LoRA checkpoint directory."""
    if project_root is None:
        project_root = PROJECT_ROOT

    patterns = [
        str(project_root / "checkpoints" / "lora_full" / "checkpoint-*"),
        str(project_root / "checkpoints" / "lora_expanded" / "checkpoint-*"),
        str(project_root / "checkpoints" / "lora_poc" / "checkpoint-*"),
        str(project_root / "checkpoints" / "lora_full_resume_smoke" / "checkpoint-*"),
        str(project_root / "checkpoints" / "lora_curriculum" / "checkpoint-*"),
    ]
    latest = find_latest_dir(patterns)
    if latest is not None:
        return latest

    # Fall back to expert-specific directories
    expert_patterns = [
        str(project_root / "checkpoints" / "lora_uci" / "checkpoint-*"),
        str(project_root / "checkpoints" / "lora_tutor" / "checkpoint-*"),
        str(project_root / "checkpoints" / "lora_director" / "checkpoint-*"),
    ]
    return find_latest_dir(expert_patterns)


def get_environment_config() -> Dict[str, Any]:
    """Get configuration from environment variables."""
    config = {}

    # Model configuration
    if model_id := os.environ.get("CHESSGEMMA_MODEL_ID"):
        config['model_id'] = model_id
    if model_path := os.environ.get("CHESSGEMMA_MODEL_PATH"):
        config['model_path'] = model_path

    # System configuration
    if debug := os.environ.get('CHESSGEMMA_DEBUG'):
        config['debug_mode'] = debug.lower() not in ('0', 'false', 'False')

    if timeout := os.environ.get('CHESSGEMMA_TIMEOUT_MINUTES'):
        try:
            config['timeout_minutes'] = int(timeout)
        except ValueError:
            pass

    if cache_size := os.environ.get('CHESSGEMMA_CACHE_SIZE'):
        try:
            config['cache_size'] = int(cache_size)
        except ValueError:
            pass

    # Engine configuration
    if engine_primary := os.environ.get("CHESSGEMMA_ENGINE_PRIMARY"):
        config['engine_primary'] = engine_primary.lower()

    if lc0_path := os.environ.get("CHESSGEMMA_LC0_PATH"):
        config['lc0_path'] = lc0_path

    if lc0_weights := os.environ.get("CHESSGEMMA_LC0_WEIGHTS"):
        config['lc0_weights'] = lc0_weights

    if lc0_backend := os.environ.get("CHESSGEMMA_LC0_BACKEND"):
        config['lc0_backend'] = lc0_backend

    if lc0_threads := os.environ.get("CHESSGEMMA_LC0_THREADS"):
        config['lc0_threads'] = lc0_threads

    if lc0_use_pool := os.environ.get("CHESSGEMMA_LC0_USE_POOL"):
        config['lc0_use_pool'] = lc0_use_pool

    # LC0 optional time limit (seconds)
    if lc0_time_limit := os.environ.get("CHESSGEMMA_LC0_TIME_LIMIT"):
        config['lc0_time_limit'] = lc0_time_limit

    if fallback_engine := os.environ.get("CHESSGEMMA_FALLBACK_ENGINE_PATH"):
        config['fallback_engine_path'] = fallback_engine

    return config


def safe_file_read(filepath: Union[str, Path], default_content: str = "") -> str:
    """Safely read a file with fallback to default content."""
    try:
        path = Path(filepath)
        if path.exists():
            return path.read_text(encoding='utf-8')
        return default_content
    except Exception:
        return default_content


def safe_file_write(filepath: Union[str, Path], content: str) -> bool:
    """Safely write content to a file."""
    try:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding='utf-8')
        return True
    except Exception:
        return False


@contextmanager
def suppress_errors(*error_types):
    """Context manager to suppress specific error types."""
    try:
        yield
    except error_types:
        pass


def batch_process(items: List[Any], batch_size: int = 10, processor: callable = None):
    """Process items in batches for memory efficiency."""
    if not processor:
        return items

    results = []
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        batch_results = processor(batch)
        if batch_results:
            results.extend(batch_results if isinstance(batch_results, list) else [batch_results])

    return results


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string."""
    if seconds < 1:
        return f"{seconds*1000:.1f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        seconds = seconds % 60
        return f"{minutes}m {seconds:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage information."""
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()

        return {
            'rss_mb': memory_info.rss / (1024 * 1024),
            'vms_mb': memory_info.vms / (1024 * 1024),
            'system_percent': psutil.virtual_memory().percent,
        }
    except ImportError:
        return {'error': 'psutil not available'}
    except Exception as e:
        return {'error': str(e)}


def validate_path_exists(path: Union[str, Path], create_if_missing: bool = False) -> bool:
    """Validate that a path exists, optionally creating it."""
    path = Path(path)

    if path.exists():
        return True

    if create_if_missing:
        try:
            path.mkdir(parents=True, exist_ok=True)
            return True
        except Exception:
            return False

    return False


def merge_dicts(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    """Merge multiple dictionaries with later values overriding earlier ones."""
    result = {}
    for d in dicts:
        if d:
            result.update(d)
    return result


def flatten_dict(d: Dict[str, Any], prefix: str = "", separator: str = ".") -> Dict[str, Any]:
    """Flatten a nested dictionary."""
    items = []
    for k, v in d.items():
        new_key = f"{prefix}{separator}{k}" if prefix else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, separator=separator).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten_dict(d: Dict[str, Any], separator: str = ".") -> Dict[str, Any]:
    """Unflatten a flattened dictionary."""
    result = {}
    for key, value in d.items():
        parts = key.split(separator)
        current = result
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value
    return result
