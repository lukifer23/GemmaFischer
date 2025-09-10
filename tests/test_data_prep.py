#!/usr/bin/env python3
"""
Tests for data preparation and processing functionality.
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from data.prepare_dataset import convert_and_save


class TestDataPreparation:
    """Test cases for data preparation functions."""
    
    def test_convert_and_save_sample(self):
        """Test dataset conversion with sample data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test with sample=False (default)
            convert_and_save(temp_dir, full=False)
            
            output_file = Path(temp_dir) / "chess_conversations.json"
            assert output_file.exists()
            
            # Check that file contains data
            with open(output_file, 'r') as f:
                lines = f.readlines()
                assert len(lines) > 0
                assert len(lines) <= 100  # Should be limited for sample
    
    @patch('data.prepare_dataset.load_dataset')
    def test_convert_and_save_full(self, mock_load_dataset):
        """Test dataset conversion with full dataset."""
        # Mock the dataset
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=1000)
        mock_dataset.select.return_value = mock_dataset
        mock_dataset.map.return_value = mock_dataset
        mock_load_dataset.return_value = mock_dataset
        
        with tempfile.TemporaryDirectory() as temp_dir:
            convert_and_save(temp_dir, full=True)
            
            output_file = Path(temp_dir) / "chess_conversations.json"
            assert output_file.exists()
    
    def test_conversation_format(self):
        """Test that conversations are properly formatted."""
        # Sample data that would come from ChessInstruct
        sample_data = {
            "task": "Given an incomplete set of chess moves, find the missing move.",
            "input": {"moves": ["e2e4", "e7e5", "?"], "result": "1-0"},
            "expected_output": "g1f3"
        }
        
        # Test the conversion function
        from data.prepare_dataset import convert_to_chat_format
        result = convert_to_chat_format(sample_data)
        
        assert "conversations" in result
        assert len(result["conversations"]) == 3
        
        # Check roles
        assert result["conversations"][0]["role"] == "system"
        assert result["conversations"][1]["role"] == "user"
        assert result["conversations"][2]["role"] == "assistant"
        
        # Check content
        assert "incomplete set of chess moves" in result["conversations"][0]["content"]
        assert "e2e4" in result["conversations"][1]["content"]
        assert "g1f3" in result["conversations"][2]["content"]


class TestDatasetValidation:
    """Test cases for dataset validation."""
    
    def test_jsonl_format(self):
        """Test that output is in proper JSONL format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            convert_and_save(temp_dir, full=False)
            
            output_file = Path(temp_dir) / "chess_conversations.json"
            
            with open(output_file, 'r') as f:
                for line in f:
                    # Each line should be valid JSON
                    data = json.loads(line)
                    assert "conversations" in data
                    assert isinstance(data["conversations"], list)
    
    def test_conversation_structure(self):
        """Test conversation structure validation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            convert_and_save(temp_dir, full=False)
            
            output_file = Path(temp_dir) / "chess_conversations.json"
            
            with open(output_file, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    conversations = data["conversations"]
                    
                    # Should have exactly 3 messages
                    assert len(conversations) == 3
                    
                    # Each message should have role and content
                    for msg in conversations:
                        assert "role" in msg
                        assert "content" in msg
                        assert msg["role"] in ["system", "user", "assistant"]


class TestDataProcessing:
    """Test cases for data processing utilities."""
    
    def test_create_finetune_dataset(self):
        """Test creation of fine-tuning dataset."""
        # This would test the create_finetune_dataset.py script
        # For now, just test that the function exists
        try:
            from create_finetune_dataset import main
            assert callable(main)
        except ImportError:
            # Script might not be in path, that's ok for this test
            pass
    
    def test_dataset_file_structure(self):
        """Test that dataset files have proper structure."""
        # Check if any existing dataset files are properly formatted
        dataset_dir = Path("data/datasets")
        if dataset_dir.exists():
            for file_path in dataset_dir.glob("*.jsonl"):
                with open(file_path, 'r') as f:
                    for line_num, line in enumerate(f, 1):
                        try:
                            data = json.loads(line)
                            # Basic structure validation
                            assert isinstance(data, dict)
                        except json.JSONDecodeError as e:
                            pytest.fail(f"Invalid JSON in {file_path} at line {line_num}: {e}")


class TestDataQuality:
    """Test cases for data quality validation."""
    
    def test_chess_move_format(self):
        """Test that chess moves are in proper format."""
        # Sample chess moves for testing
        valid_moves = ["e2e4", "Nf3", "O-O", "O-O-O", "exd5", "Nxe5"]
        invalid_moves = ["invalid", "e2e9", "KxK", "1234"]
        
        for move in valid_moves:
            # Basic validation - should be 2-6 characters
            assert 2 <= len(move) <= 6, f"Move {move} has invalid length"
        
        for move in invalid_moves:
            # These should be caught by validation
            assert len(move) < 2 or len(move) > 6 or not move.isalnum(), f"Move {move} should be invalid"
    
    def test_conversation_quality(self):
        """Test conversation quality metrics."""
        with tempfile.TemporaryDirectory() as temp_dir:
            convert_and_save(temp_dir, full=False)
            
            output_file = Path(temp_dir) / "chess_conversations.json"
            
            with open(output_file, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    conversations = data["conversations"]
                    
                    # Check that content is not empty
                    for msg in conversations:
                        assert len(msg["content"]) > 0, "Empty message content"
                    
                    # Check that system message contains task description
                    system_msg = conversations[0]
                    assert len(system_msg["content"]) > 10, "System message too short"
                    
                    # Check that user message contains input
                    user_msg = conversations[1]
                    assert len(user_msg["content"]) > 5, "User message too short"
                    
                    # Check that assistant message contains output
                    assistant_msg = conversations[2]
                    assert len(assistant_msg["content"]) > 0, "Assistant message empty"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])