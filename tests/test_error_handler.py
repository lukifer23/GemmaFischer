#!/usr/bin/env python3
"""Unit tests for the error handler utilities."""

import logging
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest

# Ensure project root is on the Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils import error_handler as error_handler_module


@pytest.fixture
def error_handler(monkeypatch):
    """Provide a fresh error handler instance for each test."""
    # Default to no torch for predictable fixture behaviour
    monkeypatch.setattr(error_handler_module, "torch", None)
    return error_handler_module.ChessGemmaErrorHandler()


def test_clear_model_cache_without_torch(caplog, error_handler):
    """Ensure cache clearing works gracefully without PyTorch installed."""
    caplog.set_level(logging.DEBUG, logger=error_handler_module.logger.name)

    with patch("gc.collect") as mock_collect:
        mock_collect.return_value = 0
        result = error_handler._clear_model_cache(None)

    assert result["success"] is True
    mock_collect.assert_called_once()
    assert "PyTorch not available" in caplog.text


def test_clear_model_cache_with_torch_mps(monkeypatch):
    """Ensure MPS cache clearing is attempted when MPS is available."""
    mps_calls = {"empty_cache": 0}

    class _DummyMPS:
        @staticmethod
        def empty_cache():
            mps_calls["empty_cache"] += 1

    class _DummyBackendsMPS:
        @staticmethod
        def is_available():
            return True

    dummy_torch = types.SimpleNamespace(
        mps=_DummyMPS,
        backends=types.SimpleNamespace(mps=_DummyBackendsMPS),
    )

    monkeypatch.setattr(error_handler_module, "torch", dummy_torch)
    handler = error_handler_module.ChessGemmaErrorHandler()

    with patch("gc.collect") as mock_collect:
        mock_collect.return_value = 0
        result = handler._clear_model_cache(None)

    assert result["success"] is True
    assert mps_calls["empty_cache"] == 1
