#!/usr/bin/env python3
"""
Comprehensive tests for ChessGemma inference functionality.
"""

import pytest
import sys
import logging
import types
import importlib.machinery
from pathlib import Path
from unittest.mock import Mock, patch

import chess

# Add project root to path
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

from src.inference.inference import (
    ChessGemmaInference,
    run_inference,
    load_model,
    unload_model,
    get_model_info,
)
from src.inference.moe_router import RoutingDecision, MoEInferenceManager


class TestChessGemmaInference:
    """Test cases for ChessGemmaInference class."""
    
    def test_initialization(self):
        """Test inference class initialization."""
        inference = ChessGemmaInference()
        assert inference.model is None
        assert inference.tokenizer is None
        assert inference.is_loaded is False
    
    def test_initialization_with_paths(self):
        """Test initialization with custom paths."""
        model_path = "test_model_path"
        adapter_path = "test_adapter_path"
        inference = ChessGemmaInference(model_path, adapter_path)
        assert inference.model_path == model_path
        assert inference.adapter_path == adapter_path
    
    @patch('src.inference.inference.AutoTokenizer')
    @patch('src.inference.inference.AutoModelForCausalLM')
    def test_load_model_success(self, mock_model, mock_tokenizer, tmp_path):
        """Test successful model loading."""
        # Mock the model and tokenizer
        mock_tokenizer_instance = Mock()
        mock_model_instance = Mock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        mock_model.from_pretrained.return_value = mock_model_instance
        
        inference = ChessGemmaInference()
        inference.model_path = tmp_path
        result = inference.load_model()
        
        assert result is True
        assert inference.is_loaded is True
        assert inference.model is not None
        assert inference.tokenizer is not None
    
    @patch('src.inference.inference.AutoTokenizer')
    def test_load_model_failure(self, mock_tokenizer):
        """Test model loading failure."""
        mock_tokenizer.from_pretrained.side_effect = Exception("Load failed")
        
        inference = ChessGemmaInference()
        result = inference.load_model()
        
        assert result is False
        assert inference.is_loaded is False

    @patch('src.inference.inference.torch')
    def test_unload_model(self, mock_torch):
        """Test unloading frees resources and resets state."""
        inference = ChessGemmaInference()
        inference.model = Mock()
        inference.tokenizer = Mock()
        inference.is_loaded = True
        inference._loaded_adapters = {'a': True}
        inference._logical_to_physical = {'a': 'a@1'}
        inference._adapter_loaded_from = {'a': Path('path')}
        inference._engine_cache['x'] = 'y'
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.empty_cache = Mock()

        inference.unload_model()

        assert inference.model is None
        assert inference.tokenizer is None
        assert inference.is_loaded is False
        assert inference._loaded_adapters == {}
        assert inference._logical_to_physical == {}
        assert inference._adapter_loaded_from == {}
        assert inference._engine_cache == {}
        mock_torch.cuda.empty_cache.assert_called_once()
    
    @patch('src.inference.inference.AutoTokenizer')
    @patch('src.inference.inference.AutoModelForCausalLM')
    def test_generate_response_success(self, mock_model, mock_tokenizer):
        """Test successful response generation."""
        # Mock the model and tokenizer
        mock_tokenizer_instance = Mock()
        mock_model_instance = Mock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        mock_model.from_pretrained.return_value = mock_model_instance
        
        # Mock tokenizer methods
        mock_tokenizer_instance.apply_chat_template.return_value = "test prompt"
        mock_tokenizer_instance.return_value = {"input_ids": Mock()}
        mock_tokenizer_instance.decode.return_value = "test response"
        
        # Mock model generation
        mock_model_instance.generate.return_value = Mock()
        mock_model_instance.device = "cpu"
        
        inference = ChessGemmaInference()
        inference.is_loaded = True
        inference.model = mock_model_instance
        inference.tokenizer = mock_tokenizer_instance
        
        result = inference.generate_response("Test question")
        
        assert "response" in result
        assert "confidence" in result
        assert result["model_loaded"] is True

    @patch('src.inference.inference.AutoTokenizer')
    @patch('src.inference.inference.AutoModelForCausalLM')
    def test_generate_response_debug_logging(self, mock_model, mock_tokenizer, caplog, monkeypatch):
        """Ensure debug logs are emitted when CHESSGEMMA_DEBUG is set."""
        monkeypatch.setenv('CHESSGEMMA_DEBUG', '1')

        mock_tokenizer_instance = Mock()
        mock_model_instance = Mock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        mock_model.from_pretrained.return_value = mock_model_instance

        mock_tokenizer_instance.return_value = {"input_ids": Mock()}
        mock_tokenizer_instance.decode.return_value = "test response"
        mock_model_instance.generate.return_value = Mock()
        mock_model_instance.device = "cpu"

        inference = ChessGemmaInference()
        inference.is_loaded = True
        inference.model = mock_model_instance
        inference.tokenizer = mock_tokenizer_instance

        with caplog.at_level(logging.DEBUG):
            inference.generate_response("Test question")

        assert any("INFERENCE DEBUG" in record.message for record in caplog.records)
    
    def test_generate_response_not_loaded(self):
        """Test response generation when model not loaded."""
        inference = ChessGemmaInference()
        result = inference.generate_response("Test question")

        assert "error" in result
        assert result["confidence"] == 0.0

    def test_generate_response_handles_missing_utils_import(self):
        """generate_response should continue working when utils imports fail."""
        import importlib
        import builtins

        import src.inference.inference as inference_module

        real_import = builtins.__import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "src.utils.error_handler":
                raise ImportError("Simulated missing error handler")
            return real_import(name, globals, locals, fromlist, level)

        try:
            with patch("builtins.__import__", side_effect=fake_import):
                inference_module = importlib.reload(inference_module)

                assert callable(inference_module.handle_error)

                class _DummyInputs(dict):
                    def __init__(self):
                        super().__init__({"input_ids": Mock()})

                    def to(self, device):  # pragma: no cover - trivial passthrough
                        return self

                class _DummyTokenizer:
                    eos_token_id = 0

                    def __call__(self, *args, **kwargs):  # pragma: no cover - simple stub
                        return _DummyInputs()

                    def decode(self, *args, **kwargs):
                        return "Test response"

                inference = inference_module.ChessGemmaInference()
                inference.is_loaded = True
                inference.model = Mock()
                inference.model.device = "cpu"
                inference.model.generate.return_value = [Mock()]
                inference.tokenizer = _DummyTokenizer()

                with inference_module.error_boundary("inference", "generate_response"):
                    pass

                result = inference.generate_response("What is the best move?")

                assert result["response"]
                assert result["model_loaded"] is True
        finally:
            importlib.reload(inference_module)

    def test_moe_uci_route_produces_legal_move(self, monkeypatch):
        """MoE routing to the UCI expert should yield a legal engine move."""

        # Disable automatic MoE initialization so the test can inject a stub manager
        monkeypatch.setenv('CHESSGEMMA_MOE_ENABLED', '0')

        inference = ChessGemmaInference()
        inference.is_loaded = True

        # Provide lightweight model/tokenizer stubs for engine generation
        class _DummyTensor:
            shape = (1, 1)

        class _DummyInputs(dict):
            def __init__(self):
                super().__init__({'input_ids': _DummyTensor()})

            def to(self, device):
                return self

        class _DummyTokenizer:
            eos_token_id = 0

            def __call__(self, *args, **kwargs):  # pragma: no cover - simple stub
                return _DummyInputs()

        inference.tokenizer = _DummyTokenizer()
        mock_model = Mock()
        mock_model.device = 'cpu'
        inference.model = mock_model
        inference.set_active_adapter = Mock()

        # Force the engine helper to return a deterministic move
        inference._generate_engine_move = Mock(return_value='e2e4')

        class _StubRouter:
            def __init__(self):
                self.last_query_type = None

            def route_query(self, fen, query_type="auto", complexity_score=None):
                self.last_query_type = query_type
                return RoutingDecision(
                    primary_expert='uci',
                    expert_weights={'uci': 1.0},
                    confidence_score=0.9,
                    reasoning='test route',
                    ensemble_mode=False,
                    fallback_used=False,
                )

        router = _StubRouter()
        moe_manager = MoEInferenceManager(router, {}, inference)
        inference.moe_manager = moe_manager
        inference.moe_enabled = True

        starting_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        question = f"FEN: {starting_fen}"

        with patch('src.inference.uci_utils.extract_first_legal_move_uci', return_value='e2e4') as mock_extract:
            result = inference.generate_response(question, mode='engine')

        inference._generate_engine_move.assert_called_once()
        mock_extract.assert_called()
        assert result["moe_used"] is True
        move_str = result["response"]
        assert len(move_str) in (4, 5)

        board = chess.Board(starting_fen)
        move = chess.Move.from_uci(move_str)
        assert move in board.legal_moves
        assert router.last_query_type == 'engine'

    def test_get_model_info(self):
        """Test model info retrieval."""
        inference = ChessGemmaInference()
        info = inference.get_model_info()
        
        assert "base_model" in info
        assert "adapter_path" in info
        assert "is_loaded" in info
        assert "device" in info


class TestConvenienceFunctions:
    """Test cases for convenience functions."""
    
    @patch('src.inference.inference.get_inference_instance')
    def test_run_inference(self, mock_get_instance):
        """Test run_inference convenience function."""
        mock_inference = Mock()
        mock_inference.generate_response.return_value = {"response": "test", "confidence": 0.8}
        mock_get_instance.return_value = mock_inference
        
        result = run_inference("Test question")
        
        assert result["response"] == "test"
        assert result["confidence"] == 0.8
        mock_inference.generate_response.assert_called_once_with("Test question")
    
    @patch('src.inference.inference.get_inference_instance')
    def test_load_model_function(self, mock_get_instance):
        """Test load_model convenience function."""
        mock_inference = Mock()
        mock_inference.load_model.return_value = True
        mock_get_instance.return_value = mock_inference
        
        result = load_model()
        
        assert result is True
        mock_inference.load_model.assert_called_once()

    @patch('src.inference.inference.get_inference_instance')
    def test_unload_model_function(self, mock_get_instance):
        """Test unload_model convenience function."""
        mock_inference = Mock()
        mock_get_instance.return_value = mock_inference

        unload_model()

        mock_inference.unload_model.assert_called_once()
    
    @patch('src.inference.inference.get_inference_instance')
    def test_get_model_info_function(self, mock_get_instance):
        """Test get_model_info convenience function."""
        mock_inference = Mock()
        mock_inference.get_model_info.return_value = {"test": "info"}
        mock_get_instance.return_value = mock_inference
        
        result = get_model_info()
        
        assert result["test"] == "info"
        mock_inference.get_model_info.assert_called_once()


class TestChessEngineIntegration:
    """Test cases for chess engine integration."""
    
    @patch('src.inference.chess_engine.ChessEngineManager')
    def test_chess_engine_validation(self, mock_engine_manager):
        """Test chess engine move validation."""
        from src.inference.chess_engine import validate_chess_move
        
        # Mock the engine manager
        mock_engine = Mock()
        mock_analysis = Mock()
        mock_analysis.move = "e2e4"
        mock_analysis.is_legal = True
        mock_analysis.move_quality = "good"
        mock_engine.validate_move.return_value = mock_analysis
        mock_engine_manager.return_value.__enter__.return_value = mock_engine
        
        result = validate_chess_move("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "e2e4")
        
        assert result.move == "e2e4"
        assert result.is_legal is True
        assert result.move_quality == "good"


class TestErrorHandling:
    """Test cases for error handling."""
    
    def test_inference_error_handling(self):
        """Test error handling in inference."""
        inference = ChessGemmaInference()
        
        # Test with invalid model path
        inference.model_path = "invalid/path"
        result = inference.load_model()
        
        assert result is False
    
    @patch('src.inference.inference.AutoTokenizer')
    @patch('src.inference.inference.AutoModelForCausalLM')
    def test_generation_error_handling(self, mock_model, mock_tokenizer):
        """Test error handling during generation."""
        # Mock successful model loading
        mock_tokenizer_instance = Mock()
        mock_model_instance = Mock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        mock_model.from_pretrained.return_value = mock_model_instance
        
        inference = ChessGemmaInference()
        inference.is_loaded = True
        inference.model = mock_model_instance
        inference.tokenizer = mock_tokenizer_instance
        
        # Mock generation error
        mock_model_instance.generate.side_effect = Exception("Generation failed")
        
        result = inference.generate_response("Test question")
        
        assert "error" in result
        assert result["confidence"] == 0.0


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
