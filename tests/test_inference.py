#!/usr/bin/env python3
"""
Comprehensive tests for ChessGemma inference functionality.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.inference.inference import ChessGemmaInference, run_inference, load_model, get_model_info


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
    def test_load_model_success(self, mock_model, mock_tokenizer):
        """Test successful model loading."""
        # Mock the model and tokenizer
        mock_tokenizer_instance = Mock()
        mock_model_instance = Mock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        mock_model.from_pretrained.return_value = mock_model_instance
        
        inference = ChessGemmaInference()
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
    
    def test_generate_response_not_loaded(self):
        """Test response generation when model not loaded."""
        inference = ChessGemmaInference()
        result = inference.generate_response("Test question")
        
        assert "error" in result
        assert result["confidence"] == 0.0
    
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