"""Tests for UCIBridge mode routing logic."""

import importlib.machinery
import sys
import types
from pathlib import Path
from unittest.mock import Mock, patch

import chess

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Provide lightweight stub for optional peft dependency used during imports
peft_stub = types.ModuleType("peft")
peft_stub.__spec__ = importlib.machinery.ModuleSpec("peft", loader=None)


class _DummyPeftModel:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):  # pragma: no cover - simple stub
        return None


peft_stub.PeftModel = _DummyPeftModel
sys.modules.setdefault("peft", peft_stub)

from src.inference.uci_bridge import UCIBridge


def _setup_bridge(mock_inference_cls, mock_engine_manager):
    mock_inference = Mock()
    mock_inference.generate_response.return_value = {"response": "e2e4"}
    mock_inference_cls.return_value = mock_inference
    mock_engine_manager.return_value = Mock()

    bridge = UCIBridge()
    return bridge, mock_inference


@patch("src.inference.uci_bridge.ChessEngineManager")
@patch("src.inference.uci_bridge.ChessGemmaInference")
def test_uci_mode_uses_engine_generation(mock_inference_cls, mock_engine_manager):
    bridge, mock_inference = _setup_bridge(mock_inference_cls, mock_engine_manager)
    bridge.options.mode = "uci"

    board = chess.Board()

    with patch("src.inference.uci_bridge.post_process_uci_response", return_value="e2e4"):
        move = bridge._generate_chessgemmma_move(board, depth=12, time_limit=1000)

    assert move == "e2e4"
    _, kwargs = mock_inference.generate_response.call_args
    assert kwargs["mode"] == "engine"
    assert bridge._last_generation_metadata == {
        "requested_mode": "uci",
        "routed_mode": "uci",
        "generation_mode": "engine",
    }


@patch("src.inference.uci_bridge.ChessEngineManager")
@patch("src.inference.uci_bridge.ChessGemmaInference")
def test_auto_mode_routes_to_engine_and_records_request(mock_inference_cls, mock_engine_manager):
    bridge, mock_inference = _setup_bridge(mock_inference_cls, mock_engine_manager)
    bridge.options.mode = "auto"
    bridge.options.moe_enabled = True

    board = chess.Board()

    with patch("src.inference.uci_bridge.post_process_uci_response", return_value="e2e4"):
        move = bridge._generate_chessgemmma_move(board, depth=12, time_limit=1000)

    assert move == "e2e4"
    _, kwargs = mock_inference.generate_response.call_args
    assert kwargs["mode"] == "engine"
    assert bridge._last_generation_metadata == {
        "requested_mode": "auto",
        "routed_mode": "auto",
        "generation_mode": "engine",
    }


@patch("src.inference.uci_bridge.ChessEngineManager")
@patch("src.inference.uci_bridge.ChessGemmaInference")
def test_auto_mode_without_moe_behaves_like_uci(mock_inference_cls, mock_engine_manager):
    bridge, mock_inference = _setup_bridge(mock_inference_cls, mock_engine_manager)
    bridge.options.mode = "auto"
    bridge.options.moe_enabled = False

    board = chess.Board()

    with patch("src.inference.uci_bridge.post_process_uci_response", return_value="e2e4"):
        move = bridge._generate_chessgemmma_move(board, depth=12, time_limit=1000)

    assert move == "e2e4"
    _, kwargs = mock_inference.generate_response.call_args
    assert kwargs["mode"] == "engine"
    assert bridge._last_generation_metadata == {
        "requested_mode": "auto",
        "routed_mode": "uci",
        "generation_mode": "engine",
    }
