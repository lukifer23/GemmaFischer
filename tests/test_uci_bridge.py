#!/usr/bin/env python3
"""Tests for the UCIBridge command handlers."""

import importlib.machinery
import sys
import types
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

# Provide lightweight stand-ins for optional heavy dependencies before importing the bridge
peft_stub = types.ModuleType("peft")
peft_stub.PeftModel = object  # Minimal placeholder used only for type references
peft_stub.__spec__ = importlib.machinery.ModuleSpec("peft", loader=None)
sys.modules["peft"] = peft_stub

# Ensure project root is on sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.inference.uci_bridge import UCIBridge


@pytest.mark.parametrize("inference_behavior", ["missing", "unloaded"])
def test_handle_isready_returns_readyok(inference_behavior):
    """`isready` should always respond with readyok without raising."""
    with patch("src.inference.uci_bridge.ChessEngineManager"):
        if inference_behavior == "missing":
            with patch(
                "src.inference.uci_bridge.ChessGemmaInference",
                side_effect=RuntimeError("inference init failed"),
            ):
                bridge = UCIBridge()
                response = bridge.handle_uci_command("isready")
                assert bridge.inference is None
        else:
            mock_inference = Mock()
            mock_inference.is_loaded = False
            with patch(
                "src.inference.uci_bridge.ChessGemmaInference", return_value=mock_inference
            ):
                bridge = UCIBridge()
                response = bridge.handle_uci_command("isready")
                assert bridge.inference is mock_inference

    assert response == "readyok"
