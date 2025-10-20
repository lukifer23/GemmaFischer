#!/usr/bin/env python3
"""
ChessGemma Source Package

This package provides a unified interface for chess analysis using
large language models enhanced with traditional chess engines.
"""

import sys
from pathlib import Path

# Ensure the src package is properly set up in Python path
_current_dir = Path(__file__).parent
_project_root = _current_dir.parent

# Add project root to path if not already there
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Package version
__version__ = "1.0.0"

# Package metadata
__description__ = "Chess Analysis with Hybrid LLM + Engine Integration"
__author__ = "ChessGemma Team"
__license__ = "MIT"

# Import key components for easy access
try:
    from .inference.inference import ChessGemmaInference, get_inference_instance
    from .inference.hybrid_engine import HybridEngine, HybridEngineResult
    from .web.app import app as web_app
    from .config.config_manager import get_config

    # Mark successful initialization
    __initialized__ = True

except ImportError as e:
    # Package not fully initialized
    __initialized__ = False
    __init_error__ = str(e)

    # Provide helpful error message
    import warnings
    warnings.warn(
        f"ChessGemma package initialization failed: {e}. "
        "Some features may not be available.",
        ImportWarning,
        stacklevel=2
    )
