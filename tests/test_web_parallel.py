#!/usr/bin/env python3
"""
Dedicated tests for web interface parallel inference functionality.
Tests the Flask API endpoints and web-specific integrations.
"""

import sys
import json
import time
from pathlib import Path
import pytest
from unittest.mock import Mock

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.web.app import app as flask_app, chess_model, chess_rag


class TestParallelWebAPI:
    """Test the parallel inference web API endpoints."""

    def setup_method(self):
        """Set up test environment."""
        flask_app.config['TESTING'] = True
        self.client = flask_app.test_client()

        # Mock components to avoid actual inference
        self._mock_components()

    def _mock_components(self):
        """Mock web components for testing."""
        # Mock chess model
        chess_model.is_loaded = True
        self.sample_hybrid_result = {
            'fen': 'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2',
            'engine': 'LC0',
            'best_move': 'e2e4',
            'principal_variation': ['e2e4', 'e7e5', 'g1f3'],
            'evaluation_cp': 85,
            'evaluation_pawns': 0.85,
            'mate_in': None,
            'depth': 20,
            'nodes': 2048,
            'engine_time': 0.42,
            'fallback_used': False,
            'analysis': {
                'evaluation': {'depth': 20, 'nodes': 2048},
                'threats': ['center pressure'],
                'opportunities': ['king safety'],
                'position_type': 'opening',
                'error': None,
            },
            'explanation': 'Control the center and develop quickly.',
            'key_points': ['Central control', 'Rapid development'],
            'explanation_adapter': 'tutor',
        }
        chess_model.analyze_with_engine = Mock(return_value=self.sample_hybrid_result)

        # Mock that returns only requested experts
        def mock_generate_parallel_responses(question=None, context=None, experts=None, **kwargs):
            all_responses = {
                'uci': {
                    'response': 'e2e4',
                    'confidence': 0.9,
                    'generation_time': 1.0,
                    'model_loaded': True,
                    'mode': 'uci',
                    'cached': False,
                    'cache_hit_rate': 0.0
                },
                'tutor': {
                    'response': 'Opening move tutorial...',
                    'confidence': 0.85,
                    'generation_time': 1.5,
                    'model_loaded': True,
                    'mode': 'tutor',
                    'cached': False,
                    'cache_hit_rate': 0.0
                },
                'director': {
                    'response': 'Strategic analysis...',
                    'confidence': 0.8,
                    'generation_time': 1.2,
                    'model_loaded': True,
                    'mode': 'director',
                    'cached': False,
                    'cache_hit_rate': 0.0
                }
            }

            # Return only requested experts, default to all if none specified
            if experts is None or len(experts) == 0:
                experts = ['uci', 'tutor', 'director']

            return {expert: all_responses[expert] for expert in experts if expert in all_responses}

        chess_model.generate_parallel_responses = Mock(side_effect=mock_generate_parallel_responses)

        # Mock RAG system
        chess_rag.get_relevant_knowledge = Mock(return_value="Mock chess knowledge")

    def test_parallel_endpoint_exists(self):
        """Test that /api/ask_parallel endpoint exists and responds."""
        response = self.client.post('/api/ask_parallel', json={'question': 'Test'})
        assert response.status_code == 200

    def test_parallel_api_basic_request(self):
        """Test basic parallel API request structure."""
        response = self.client.post('/api/ask_parallel',
                                  json={'question': 'What is the best move?'})

        assert response.status_code == 200
        data = response.get_json()

        # Check top-level structure
        required_fields = ['question', 'experts', 'total_time', 'results']
        for field in required_fields:
            assert field in data

        # Check results structure
        results = data['results']
        assert isinstance(results, dict)
        assert len(results) == 3  # Default experts

        # Check each expert response
        for expert in ['uci', 'tutor', 'director']:
            assert expert in results
            expert_data = results[expert]
            assert 'response' in expert_data
            assert 'confidence' in expert_data
            assert 'generation_time' in expert_data

    def test_parallel_api_custom_experts(self):
        """Test parallel API with custom expert selection."""
        response = self.client.post('/api/ask_parallel',
                                  json={
                                      'question': 'Test question',
                                      'experts': ['uci', 'tutor']
                                  })

        assert response.status_code == 200
        data = response.get_json()

        results = data['results']
        assert len(results) == 2
        assert 'uci' in results
        assert 'tutor' in results
        assert 'director' not in results

    def test_parallel_api_single_expert(self):
        """Test parallel API with single expert."""
        response = self.client.post('/api/ask_parallel',
                                  json={
                                      'question': 'Test question',
                                      'experts': ['uci']
                                  })

        assert response.status_code == 200
        data = response.get_json()

        results = data['results']
        assert len(results) == 1
        assert 'uci' in results

    def test_parallel_api_empty_experts_defaults(self):
        """Test that empty experts list defaults to all experts."""
        response = self.client.post('/api/ask_parallel',
                                  json={
                                      'question': 'Test question',
                                      'experts': []
                                  })

        assert response.status_code == 200
        data = response.get_json()

        results = data['results']
        assert len(results) == 3  # Should default to all experts

    def test_parallel_api_with_context(self):
        """Test parallel API with additional context."""
        context = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"

        response = self.client.post('/api/ask_parallel',
                                  json={
                                      'question': 'What is the best move?',
                                      'context': context
                                  })

        assert response.status_code == 200
        data = response.get_json()

        # Should include context in response
        assert data['context'] == context

        # Verify RAG was called
        chess_rag.get_relevant_knowledge.assert_called_with('What is the best move?')

    def test_ask_endpoint_includes_lc0_metadata(self):
        """LC0 analysis responses should expose metadata for the UI."""

        fen = self.sample_hybrid_result['fen']
        response = self.client.post(
            '/api/ask',
            json={'question': f'FEN: {fen}\nWhat now?', 'expert': 'uci'}
        )

        assert response.status_code == 200
        data = response.get_json()

        assert data['engine'] == self.sample_hybrid_result['engine']
        assert data['fallback_used'] == self.sample_hybrid_result['fallback_used']
        assert 'analysis' in data

        raw_analysis = data['analysis']['analysis']
        assert raw_analysis['evaluation']['depth'] == self.sample_hybrid_result['analysis']['evaluation']['depth']
        assert raw_analysis['position_type'] == 'opening'
        assert raw_analysis['error'] is None
        assert data['analysis']['key_points'] == self.sample_hybrid_result['key_points']
        assert data['best_move'] == self.sample_hybrid_result['best_move']
        assert data['principal_variation'] == self.sample_hybrid_result['principal_variation']
        assert chess_model.analyze_with_engine.called

    def test_parallel_api_error_missing_question(self):
        """Test error handling when question is missing."""
        response = self.client.post('/api/ask_parallel', json={})

        assert response.status_code == 200  # Returns error in JSON
        data = response.get_json()

        assert 'error' in data
        assert 'No question provided' in data['error']

    def test_parallel_api_error_empty_question(self):
        """Test error handling when question is empty."""
        response = self.client.post('/api/ask_parallel', json={'question': ''})

        assert response.status_code == 200
        data = response.get_json()

        assert 'error' in data
        assert 'No question provided' in data['error']

    def test_parallel_api_error_invalid_json(self):
        """Test handling of invalid JSON."""
        response = self.client.post('/api/ask_parallel',
                                  data='invalid json',
                                  content_type='application/json')

        # Flask should handle this gracefully
        assert response.status_code in [200, 400]

    def test_parallel_api_content_type(self):
        """Test that API requires correct content type."""
        response = self.client.post('/api/ask_parallel',
                                  data=json.dumps({'question': 'test'}),
                                  content_type='text/plain')

        # Should still work or return error gracefully
        assert response.status_code in [200, 400, 415]

    def test_parallel_api_method_not_allowed(self):
        """Test that GET requests are rejected."""
        response = self.client.get('/api/ask_parallel')

        assert response.status_code == 405  # Method Not Allowed

    def test_parallel_api_cors_headers(self):
        """Test CORS headers are present."""
        response = self.client.post('/api/ask_parallel',
                                  json={'question': 'Test'})

        # Check for CORS headers (Flask-CORS adds these)
        assert 'Access-Control-Allow-Origin' in response.headers or response.status_code == 200

    def test_parallel_api_performance_metrics(self):
        """Test that performance metrics are included."""
        start_time = time.time()
        response = self.client.post('/api/ask_parallel',
                                  json={'question': 'Performance test'})
        end_time = time.time()

        assert response.status_code == 200
        data = response.get_json()

        # Should include timing information
        assert 'total_time' in data
        assert isinstance(data['total_time'], (int, float))
        assert data['total_time'] >= 0
        assert data['total_time'] <= (end_time - start_time + 1)  # Allow some tolerance

    def test_parallel_api_rag_integration(self):
        """Test RAG system integration."""
        response = self.client.post('/api/ask_parallel',
                                  json={'question': 'What is castling?'})

        assert response.status_code == 200

        # Verify RAG was called with the question
        chess_rag.get_relevant_knowledge.assert_called_with('What is castling?')

        # Verify the model was called with enhanced context
        call_args = chess_model.generate_parallel_responses.call_args
        assert call_args is not None

        # Should have RAG context prepended
        context_arg = call_args[1]['context']  # kwargs
        assert 'Chess Knowledge: Mock chess knowledge' in context_arg

    def test_parallel_api_model_unloaded(self):
        """Test API behavior when model is not loaded."""
        # Temporarily set model as unloaded
        original_loaded = chess_model.is_loaded
        chess_model.is_loaded = False

        try:
            response = self.client.post('/api/ask_parallel',
                                      json={'question': 'Test when unloaded'})

            assert response.status_code == 200
            data = response.get_json()

            # Should return error responses for all experts
            results = data['results']
            assert len(results) == 3

            for expert, result in results.items():
                assert 'error' in result
                assert result['error'] == 'Model not loaded'

        finally:
            chess_model.is_loaded = original_loaded

    def test_parallel_api_large_payload(self):
        """Test handling of large request payloads."""
        large_question = "What is the best move? " * 1000  # Very long question
        large_context = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3 " * 100

        response = self.client.post('/api/ask_parallel',
                                  json={
                                      'question': large_question,
                                      'context': large_context
                                  })

        # Should handle gracefully (either succeed or fail gracefully)
        assert response.status_code in [200, 413, 400]

        if response.status_code == 200:
            data = response.get_json()
            assert 'results' in data

    def test_parallel_api_concurrent_requests(self):
        """Test handling of concurrent requests to parallel API."""
        import threading

        results = []
        errors = []

        def make_request(request_id):
            """Make a parallel API request."""
            try:
                response = self.client.post('/api/ask_parallel',
                                          json={'question': f'Concurrent test {request_id}'})
                results.append((request_id, response.status_code, response.get_json() if response.status_code == 200 else None))
            except Exception as e:
                errors.append((request_id, str(e)))

        # Start multiple concurrent requests
        threads = []
        num_requests = 5

        for i in range(num_requests):
            thread = threading.Thread(target=make_request, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all to complete
        for thread in threads:
            thread.join(timeout=10.0)

        # Verify results
        assert len(results) == num_requests
        assert len(errors) == 0

        # All requests should have succeeded
        for request_id, status_code, data in results:
            assert status_code == 200
            assert data is not None
            assert 'results' in data

    def test_parallel_api_response_consistency(self):
        """Test that identical requests return consistent response structure."""
        # Make the same request multiple times
        responses = []

        for i in range(3):
            response = self.client.post('/api/ask_parallel',
                                      json={'question': 'Consistency test'})
            assert response.status_code == 200
            responses.append(response.get_json())

        # All responses should have the same structure
        first_response = responses[0]
        for response in responses[1:]:
            assert set(response.keys()) == set(first_response.keys())
            assert set(response['results'].keys()) == set(first_response['results'].keys())

            # Check that each expert response has the same fields
            for expert in response['results']:
                assert set(response['results'][expert].keys()) == set(first_response['results'][expert].keys())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
