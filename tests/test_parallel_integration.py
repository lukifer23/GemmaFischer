#!/usr/bin/env python3
"""
Integration tests for parallel inference functionality.
Tests the complete end-to-end flow including MoE router integration.
"""

import sys
import time
import json
from pathlib import Path
import pytest
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.inference.inference import ChessGemmaInference, get_inference_instance
from src.inference.moe_router import ChessMoERouter, MoEInferenceManager
from src.web.app import app as flask_app, chess_model


class TestParallelIntegration:
    """Integration tests for parallel inference with full system components."""

    def setup_method(self):
        """Set up integration test environment."""
        # Create a real inference instance for integration testing
        self.inference = ChessGemmaInference()
        self.inference.is_loaded = True  # Mock as loaded for testing

    def test_end_to_end_parallel_inference_flow(self):
        """Test complete parallel inference workflow."""
        # This test validates that all components work together
        # without actually running inference (since model isn't loaded)

        question = "What is the best move for white?"
        context = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"

        # Test with unloaded model (should return error responses)
        results = self.inference.generate_parallel_responses(
            question=question,
            context=context,
            experts=['uci', 'tutor', 'director']
        )

        # Verify structure and error handling
        assert isinstance(results, dict)
        assert len(results) == 3

        for expert, response in results.items():
            assert isinstance(response, dict)
            assert 'error' in response
            assert response['error'] == "Model not loaded"
            assert response['response'] == ""
            assert response['confidence'] == 0.0
            assert 'model_loaded' in response
            assert 'generation_time' in response

    def test_moe_router_parallel_integration(self):
        """Test MoE router integration with parallel inference."""
        # Create mock MoE router
        mock_router = Mock(spec=ChessMoERouter)
        mock_routing_decision = Mock()
        mock_routing_decision.primary_expert = 'tutor'
        mock_routing_decision.expert_weights = {'uci': 0.3, 'tutor': 0.5, 'director': 0.2}
        mock_routing_decision.confidence_score = 0.85
        mock_router.route_query.return_value = mock_routing_decision

        # Create MoE manager
        moe_manager = MoEInferenceManager(mock_router, {}, self.inference)

        # Test that parallel inference works with MoE components
        results = self.inference.generate_parallel_responses(
            "Test MoE integration",
            experts=['uci', 'tutor']
        )

        # Should return results for requested experts
        assert len(results) == 2
        assert 'uci' in results
        assert 'tutor' in results

    def test_adapter_switching_parallel_safety(self):
        """Test that adapter switching is thread-safe in parallel execution."""
        # Mock the adapter switching mechanism
        self.inference.set_active_adapter = Mock()
        self.inference._logical_to_physical = {'uci': 'uci@checkpoint-100'}

        # Mock generate_response to track adapter calls
        adapter_calls = []
        def mock_generate_response(question, context=None, mode=None, **kwargs):
            adapter_calls.append(mode)
            return {
                'response': f'Response for {mode}',
                'confidence': 0.8,
                'generation_time': 1.0,
                'model_loaded': True,
                'mode': mode,
                'cached': False,
                'cache_hit_rate': 0.0
            }

        self.inference.generate_response = mock_generate_response

        # Run parallel inference
        results = self.inference.generate_parallel_responses(
            "Thread safety test",
            experts=['uci', 'tutor']
        )

        # Verify results structure
        assert len(results) == 2
        assert len(adapter_calls) == 2

        # Verify each expert got called
        assert 'uci' in adapter_calls
        assert 'tutor' in adapter_calls

    def test_cache_integration_parallel(self):
        """Test caching integration with parallel inference."""
        # Mock cache functionality
        self.inference._response_cache = {}
        self.inference._cache_hits = 0
        self.inference._total_requests = 0

        # Mock generate_response
        call_count = 0
        def mock_generate_response(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return {
                'response': f'Mock response {call_count}',
                'confidence': 0.8,
                'generation_time': 1.0,
                'model_loaded': True,
                'mode': 'test',
                'cached': False,
                'cache_hit_rate': 0.0
            }

        self.inference.generate_response = mock_generate_response

        # First parallel call
        results1 = self.inference.generate_parallel_responses(
            "Cache test question",
            experts=['uci']
        )

        # Second identical call
        results2 = self.inference.generate_parallel_responses(
            "Cache test question",
            experts=['uci']
        )

        # Should have made 2 calls total (cache doesn't work across parallel calls
        # since each expert has different cache keys)
        assert call_count == 2

    def test_error_propagation_parallel(self):
        """Test error propagation and handling in parallel execution."""
        # Mock generate_response to simulate various error conditions
        call_count = 0
        def mock_generate_response(question, context=None, mode=None, **kwargs):
            nonlocal call_count
            call_count += 1

            if mode == 'uci':
                # Normal response
                return {
                    'response': 'e2e4',
                    'confidence': 0.9,
                    'generation_time': 1.0,
                    'model_loaded': True,
                    'mode': 'uci',
                    'cached': False,
                    'cache_hit_rate': 0.0
                }
            elif mode == 'tutor':
                # Simulate failure
                raise RuntimeError("Tutor expert network error")
            elif mode == 'director':
                # Different type of response
                return {
                    'response': '',
                    'confidence': 0.0,
                    'generation_time': 0.5,
                    'model_loaded': True,
                    'mode': 'director',
                    'cached': False,
                    'cache_hit_rate': 0.0
                }

        self.inference.generate_response = mock_generate_response

        # Run parallel inference with mixed success/failure
        results = self.inference.generate_parallel_responses(
            "Error propagation test",
            experts=['uci', 'tutor', 'director']
        )

        # Verify structure
        assert len(results) == 3

        # UCI should succeed
        assert results['uci']['response'] == 'e2e4'
        assert 'error' not in results['uci']

        # Tutor should fail
        assert 'error' in results['tutor']
        assert 'Tutor expert network error' in results['tutor']['error']

        # Director should succeed but with empty response
        assert results['director']['response'] == ''
        assert 'error' not in results['director']

    def test_performance_monitoring_parallel(self):
        """Test performance monitoring in parallel execution."""
        import time

        # Mock generate_response with timing
        def mock_generate_response(question, context=None, mode=None, **kwargs):
            # Simulate different response times per expert
            if mode == 'uci':
                time.sleep(0.1)
                generation_time = 0.1
            elif mode == 'tutor':
                time.sleep(0.2)
                generation_time = 0.2
            else:  # director
                time.sleep(0.05)
                generation_time = 0.05

            return {
                'response': f'Response from {mode}',
                'confidence': 0.8,
                'generation_time': generation_time,
                'model_loaded': True,
                'mode': mode,
                'cached': False,
                'cache_hit_rate': 0.0
            }

        self.inference.generate_response = mock_generate_response

        # Measure total execution time
        start_time = time.time()
        results = self.inference.generate_parallel_responses(
            "Performance test",
            experts=['uci', 'tutor', 'director']
        )
        total_time = time.time() - start_time

        # Parallel execution should be faster than sequential
        # (sequential would be ~0.35s, parallel should be ~0.2s)
        assert total_time < 0.3  # Allow some overhead

        # Each expert should report its individual timing
        assert abs(results['uci']['generation_time'] - 0.1) < 0.05
        assert abs(results['tutor']['generation_time'] - 0.2) < 0.05
        assert abs(results['director']['generation_time'] - 0.05) < 0.05

    def test_resource_cleanup_parallel(self):
        """Test that resources are properly cleaned up after parallel execution."""
        # Track thread creation and cleanup
        import threading

        initial_thread_count = threading.active_count()

        # Mock generate_response
        def mock_generate_response(*args, **kwargs):
            return {
                'response': 'Test response',
                'confidence': 0.8,
                'generation_time': 0.1,
                'model_loaded': True,
                'mode': 'test',
                'cached': False,
                'cache_hit_rate': 0.0
            }

        self.inference.generate_response = mock_generate_response

        # Run parallel inference
        results = self.inference.generate_parallel_responses(
            "Resource cleanup test",
            experts=['uci', 'tutor', 'director']
        )

        # Give threads time to clean up
        time.sleep(0.1)

        # Thread count should return to near initial (allowing for test runner threads)
        final_thread_count = threading.active_count()
        assert final_thread_count <= initial_thread_count + 2  # Allow some tolerance

    def test_large_scale_parallel_stress(self):
        """Stress test with many parallel requests."""
        # Mock lightweight response
        def mock_generate_response(*args, **kwargs):
            return {
                'response': 'Stress test response',
                'confidence': 0.8,
                'generation_time': 0.01,
                'model_loaded': True,
                'mode': 'stress',
                'cached': False,
                'cache_hit_rate': 0.0
            }

        self.inference.generate_response = mock_generate_response

        # Run multiple parallel inferences
        start_time = time.time()
        all_results = []

        for i in range(10):
            results = self.inference.generate_parallel_responses(
                f"Stress test question {i}",
                experts=['uci', 'tutor']
            )
            all_results.append(results)

        total_time = time.time() - start_time

        # Should handle 10 parallel inferences reasonably quickly
        assert total_time < 2.0  # Should complete in under 2 seconds

        # All should have proper structure
        assert len(all_results) == 10
        for results in all_results:
            assert len(results) == 2
            assert 'uci' in results
            assert 'tutor' in results

    def test_mixed_expert_configurations(self):
        """Test parallel inference with various expert configurations."""
        def mock_generate_response(question, context=None, mode=None, **kwargs):
            return {
                'response': f'{mode} response',
                'confidence': 0.8,
                'generation_time': 0.1,
                'model_loaded': True,
                'mode': mode,
                'cached': False,
                'cache_hit_rate': 0.0
            }

        self.inference.generate_response = mock_generate_response

        # Test different expert combinations
        test_cases = [
            ['uci'],
            ['tutor'],
            ['uci', 'tutor'],
            ['uci', 'director'],
            ['tutor', 'director'],
            ['uci', 'tutor', 'director']
        ]

        for experts in test_cases:
            results = self.inference.generate_parallel_responses(
                f"Test with {experts}",
                experts=experts
            )

            # Should return exactly the requested experts
            assert len(results) == len(experts)
            for expert in experts:
                assert expert in results
                assert results[expert]['response'] == f'{expert} response'


class TestSystemIntegration:
    """System-level integration tests."""

    def test_full_pipeline_integration(self):
        """Test the complete pipeline from web request to parallel inference."""
        flask_app.config['TESTING'] = True

        # Mock all components
        chess_model.is_loaded = True

        def mock_parallel_responses(*args, **kwargs):
            return {
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
                    'response': 'Opening tutorial...',
                    'confidence': 0.85,
                    'generation_time': 1.5,
                    'model_loaded': True,
                    'mode': 'tutor',
                    'cached': False,
                    'cache_hit_rate': 0.0
                }
            }

        chess_model.generate_parallel_responses = mock_parallel_responses

        # Test the full web pipeline
        with flask_app.test_client() as client:
            response = client.post('/api/ask_parallel',
                                  json={
                                      'question': 'Integration test',
                                      'experts': ['uci', 'tutor']
                                  })

            assert response.status_code == 200
            data = response.get_json()

            # Verify complete pipeline
            assert 'question' in data
            assert 'experts' in data
            assert 'total_time' in data
            assert 'results' in data

            results = data['results']
            assert len(results) == 2
            assert 'uci' in results
            assert 'tutor' in results

    def test_cross_component_compatibility(self):
        """Test compatibility between different system components."""
        # Test that inference, MoE, and web components can work together

        # Create minimal working system
        inference = ChessGemmaInference()
        inference.is_loaded = True

        # Mock MoE components
        mock_router = Mock()
        moe_manager = MoEInferenceManager(mock_router, {}, inference)

        # Mock web model
        original_model = chess_model
        chess_model._inference = inference

        try:
            # Test that components can coexist
            assert hasattr(inference, 'generate_parallel_responses')
            assert hasattr(moe_manager, 'prime_available_experts')
            assert hasattr(chess_model, 'generate_parallel_responses')

            # Test basic functionality
            results = inference.generate_parallel_responses("Compatibility test")
            assert isinstance(results, dict)

        finally:
            # Restore original model
            import src.web.app as web_app
            web_app.chess_model = original_model


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
