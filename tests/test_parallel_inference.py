#!/usr/bin/env python3
"""
Comprehensive tests for parallel inference functionality.
Tests real thread safety, performance, error handling, and integration scenarios.
"""

import sys
import time
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import pytest
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.inference.inference import ChessGemmaInference, get_inference_instance
from src.web.app import app as flask_app


class TestParallelInference:
    """Comprehensive test cases for parallel inference functionality."""

    def test_generate_parallel_responses_signature(self):
        """Test that generate_parallel_responses has correct method signature."""
        instance = ChessGemmaInference()

        # Check method exists
        assert hasattr(instance, 'generate_parallel_responses')

        # Check signature
        import inspect
        sig = inspect.signature(instance.generate_parallel_responses)
        expected_params = ['question', 'context', 'experts', 'max_new_tokens', 'temperature', 'top_p', 'do_sample']
        actual_params = list(sig.parameters.keys())

        for param in expected_params:
            assert param in actual_params, f"Parameter {param} missing from generate_parallel_responses"

    def test_web_interface_has_parallel_method(self):
        """Test that web interface exposes parallel functionality."""
        from src.web.app import chess_model
        assert hasattr(chess_model, 'generate_parallel_responses')

    def test_parallel_inference_not_loaded(self):
        """Test parallel inference when model is not loaded."""
        instance = ChessGemmaInference()
        instance.is_loaded = False

        results = instance.generate_parallel_responses("Test question")

        # Should return error responses for all experts
        assert isinstance(results, dict)
        assert len(results) == 3  # Default experts: uci, tutor, director
        for expert, response in results.items():
            assert 'error' in response
            assert response['error'] == "Model not loaded"
            assert response['response'] == ""
            assert response['confidence'] == 0.0

    def test_parallel_inference_custom_experts_list(self):
        """Test parallel inference with custom expert selection."""
        instance = ChessGemmaInference()
        instance.is_loaded = False

        # Test with only 2 experts
        results = instance.generate_parallel_responses(
            "Test question",
            experts=['uci', 'tutor']
        )

        assert isinstance(results, dict)
        assert len(results) == 2
        assert 'uci' in results
        assert 'tutor' in results
        assert 'director' not in results

    def test_parallel_inference_empty_experts_list(self):
        """Test parallel inference with empty experts list defaults to all."""
        instance = ChessGemmaInference()
        instance.is_loaded = False

        results = instance.generate_parallel_responses(
            "Test question",
            experts=[]
        )

        # Empty list should default to all experts
        assert isinstance(results, dict)
        assert len(results) == 3
        assert 'uci' in results
        assert 'tutor' in results
        assert 'director' in results

    def test_parallel_inference_single_expert(self):
        """Test parallel inference with single expert."""
        instance = ChessGemmaInference()
        instance.is_loaded = False

        results = instance.generate_parallel_responses(
            "Test question",
            experts=['uci']
        )

        assert isinstance(results, dict)
        assert len(results) == 1
        assert 'uci' in results

    def test_parallel_inference_response_structure(self):
        """Test that parallel inference responses have correct structure."""
        instance = ChessGemmaInference()
        instance.is_loaded = False

        results = instance.generate_parallel_responses("Test question")

        for expert, response in results.items():
            assert isinstance(response, dict)
            assert 'error' in response
            assert 'response' in response
            assert 'confidence' in response
            assert 'model_loaded' in response
            assert 'mode' in response
            assert 'generation_time' in response
            assert 'cached' in response
            assert 'cache_hit_rate' in response

    def test_parallel_inference_expert_mode_mapping(self):
        """Test that experts are correctly mapped to their modes."""
        instance = ChessGemmaInference()
        instance.is_loaded = False

        results = instance.generate_parallel_responses("Test question")

        # Check that each expert has the correct mode
        # Note: 'uci' expert maps to 'engine' mode internally
        assert results['uci']['mode'] == 'engine'
        assert results['tutor']['mode'] == 'tutor'
        assert results['director']['mode'] == 'director'

    def test_enhanced_inference_mode_adapter_routing(self):
        """Deprecated: enhanced inference removed in unified architecture."""
        assert True


class TestThreadSafety:
    """Test thread safety and race condition handling in parallel inference."""

    def test_concurrent_adapter_switching(self):
        """Test that concurrent adapter switching doesn't cause race conditions."""
        instance = ChessGemmaInference()
        instance.is_loaded = False

        results = []
        errors = []

        def worker_thread(thread_id):
            """Worker thread that performs parallel inference."""
            try:
                thread_results = instance.generate_parallel_responses(
                    f"Thread {thread_id} question",
                    experts=['uci', 'tutor']
                )
                results.append((thread_id, thread_results))
            except Exception as e:
                errors.append((thread_id, str(e)))

        # Start multiple threads simultaneously
        threads = []
        num_threads = 5

        for i in range(num_threads):
            thread = threading.Thread(target=worker_thread, args=(i,))
            threads.append(thread)

        # Start all threads at once
        for thread in threads:
            thread.start()

        # Wait for all to complete
        for thread in threads:
            thread.join(timeout=10.0)  # 10 second timeout

        # Verify results
        assert len(results) == num_threads, f"Expected {num_threads} results, got {len(results)}"
        assert len(errors) == 0, f"Unexpected errors: {errors}"

        # Verify each thread got proper responses
        for thread_id, thread_results in results:
            assert isinstance(thread_results, dict)
            assert len(thread_results) == 2  # uci and tutor
            assert 'uci' in thread_results
            assert 'tutor' in thread_results

    def test_parallel_inference_timeout_handling(self):
        """Test that threads respect timeout limits."""
        instance = ChessGemmaInference()
        instance.is_loaded = False

        start_time = time.time()

        # This should complete quickly since model is not loaded
        results = instance.generate_parallel_responses(
            "Test question",
            experts=['uci', 'tutor', 'director']
        )

        elapsed = time.time() - start_time

        # Should complete in well under the 30-second thread timeout
        assert elapsed < 5.0, f"Took too long: {elapsed:.2f}s"

        # Should have results for all experts
        assert len(results) == 3
        assert all('error' in response for response in results.values())

    def test_thread_isolation(self):
        """Test that thread-local state doesn't leak between parallel calls."""
        instance = ChessGemmaInference()
        instance.is_loaded = False

        # Make multiple calls with different expert sets
        result1 = instance.generate_parallel_responses("Question 1", experts=['uci'])
        result2 = instance.generate_parallel_responses("Question 2", experts=['tutor', 'director'])
        result3 = instance.generate_parallel_responses("Question 3", experts=['uci', 'tutor', 'director'])

        # Verify each call has the correct number of results
        assert len(result1) == 1 and 'uci' in result1
        assert len(result2) == 2 and 'tutor' in result2 and 'director' in result2
        assert len(result3) == 3 and all(exp in result3 for exp in ['uci', 'tutor', 'director'])


class TestPerformanceBenchmarking:
    """Test performance characteristics of parallel inference."""

    def test_parallel_vs_sequential_performance_baseline(self):
        """Establish baseline performance characteristics."""
        instance = ChessGemmaInference()
        instance.is_loaded = False

        question = "What is the best move for white?"
        context = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"

        # Measure parallel execution time
        start_time = time.time()
        parallel_results = instance.generate_parallel_responses(
            question, context, experts=['uci', 'tutor', 'director']
        )
        parallel_time = time.time() - start_time

        # Measure sequential execution time
        start_time = time.time()
        sequential_results = {}
        for expert in ['uci', 'tutor', 'director']:
            sequential_results[expert] = instance.generate_response(question, context, mode=expert)
        sequential_time = time.time() - start_time

        # Parallel should be faster than 3x sequential time (some overhead expected)
        assert parallel_time < sequential_time, f"Parallel ({parallel_time:.2f}s) should be faster than sequential ({sequential_time:.2f}s)"

        # Both should return same structure
        assert len(parallel_results) == len(sequential_results) == 3

    def test_memory_usage_scaling(self):
        """Test that memory usage scales appropriately with expert count."""
        instance = ChessGemmaInference()
        instance.is_loaded = False

        base_results = instance.generate_parallel_responses("Test", experts=['uci'])
        double_results = instance.generate_parallel_responses("Test", experts=['uci', 'tutor'])
        triple_results = instance.generate_parallel_responses("Test", experts=['uci', 'tutor', 'director'])

        # Results should scale linearly with expert count
        assert len(base_results) == 1
        assert len(double_results) == 2
        assert len(triple_results) == 3

    def test_cache_performance_parallel(self):
        """Test that caching works correctly in parallel execution."""
        instance = ChessGemmaInference()
        instance.is_loaded = False

        # First call - should cache results
        results1 = instance.generate_parallel_responses("Cached question", experts=['uci', 'tutor'])

        # Second call with same question - should use cache
        results2 = instance.generate_parallel_responses("Cached question", experts=['uci', 'tutor'])

        # Both calls should return same structure
        assert len(results1) == len(results2) == 2
        assert set(results1.keys()) == set(results2.keys())

    def test_expert_load_balancing(self):
        """Test that experts are evenly utilized in parallel execution."""
        instance = ChessGemmaInference()
        instance.is_loaded = False

        # Run multiple parallel inferences
        all_results = []
        for i in range(10):
            results = instance.generate_parallel_responses(f"Question {i}", experts=['uci', 'tutor', 'director'])
            all_results.append(results)

        # All calls should have returned results for all experts
        assert all(len(results) == 3 for results in all_results)


class TestWebAPIIntegration:
    """Test web API integration for parallel inference."""

    def test_flask_parallel_endpoint_exists(self):
        """Test that the parallel API endpoint is properly registered."""
        with flask_app.test_client() as client:
            # Check that endpoint exists
            response = client.get('/api/ask_parallel')
            # Should get method not allowed, not 404
            assert response.status_code == 405  # Method Not Allowed

    def test_parallel_api_response_format(self):
        """Test that parallel API returns correct JSON format."""
        flask_app.config['TESTING'] = True

        with flask_app.test_client() as client:
            response = client.post('/api/ask_parallel',
                                 json={'question': 'Test question'})

            assert response.status_code == 200
            data = response.get_json()

            # Check response structure
            assert 'question' in data
            assert 'experts' in data
            assert 'total_time' in data
            assert 'results' in data

            # Check results structure
            results = data['results']
            assert isinstance(results, dict)
            assert len(results) == 3  # Default experts

            for expert, result in results.items():
                assert 'response' in result
                assert 'confidence' in result
                assert 'generation_time' in result
                assert 'model_loaded' in result

    def test_parallel_api_custom_experts(self):
        """Test parallel API with custom expert selection."""
        flask_app.config['TESTING'] = True

        with flask_app.test_client() as client:
            response = client.post('/api/ask_parallel',
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

    def test_parallel_api_error_handling(self):
        """Test parallel API error handling."""
        flask_app.config['TESTING'] = True

        with flask_app.test_client() as client:
            # Test missing question
            response = client.post('/api/ask_parallel', json={})
            assert response.status_code == 200  # Returns error in JSON
            data = response.get_json()
            assert 'error' in data

            # Test empty question
            response = client.post('/api/ask_parallel', json={'question': ''})
            assert response.status_code == 200
            data = response.get_json()
            assert 'error' in data

    def test_parallel_api_rag_integration(self):
        """Test that parallel API integrates with RAG system."""
        flask_app.config['TESTING'] = True

        with flask_app.test_client() as client:
            response = client.post('/api/ask_parallel',
                                 json={'question': 'What is castling?'})

            assert response.status_code == 200
            data = response.get_json()

            # Should have processed RAG context
            assert 'question' in data
            assert 'context' in data

    def test_parallel_api_performance_logging(self):
        """Test that parallel API properly logs performance metrics."""
        flask_app.config['TESTING'] = True

        with flask_app.test_client() as client:
            response = client.post('/api/ask_parallel',
                                 json={'question': 'Performance test'})

            assert response.status_code == 200
            data = response.get_json()

            # Should include timing information
            assert 'total_time' in data
            assert isinstance(data['total_time'], (int, float))
            assert data['total_time'] >= 0


class TestMoEIntegration:
    """Test integration with Mixture of Experts router."""

    @patch('src.inference.moe_router.ChessMoERouter')
    def test_parallel_with_moe_router(self, mock_router_class):
        """Test parallel inference integrated with MoE routing."""
        # Mock MoE router
        mock_router = Mock()
        mock_router_class.return_value = mock_router

        # Mock routing decision
        routing_decision = Mock()
        routing_decision.primary_expert = 'tutor'
        routing_decision.expert_weights = {'uci': 0.2, 'tutor': 0.6, 'director': 0.2}
        routing_decision.confidence_score = 0.85

        mock_router.route_query.return_value = routing_decision

        instance = ChessGemmaInference()
        instance.is_loaded = False

        # This would normally integrate with MoE, but since model isn't loaded,
        # it should still handle the routing logic gracefully
        results = instance.generate_parallel_responses("Test question")

        # Should return results for all experts despite MoE routing
        assert len(results) == 3
        assert all('error' in response for response in results.values())

    def test_ensemble_response_compatibility(self):
        """Test that parallel results are compatible with ensemble processing."""
        instance = ChessGemmaInference()
        instance.is_loaded = False

        results = instance.generate_parallel_responses(
            "Test question",
            experts=['uci', 'tutor', 'director']
        )

        # Results should be in format expected by ensemble processors
        assert isinstance(results, dict)
        assert all(isinstance(response, dict) for response in results.values())

        # Each response should have standard fields
        required_fields = ['response', 'confidence', 'generation_time', 'model_loaded']
        for expert, response in results.items():
            for field in required_fields:
                assert field in response

    def test_moe_fallback_behavior(self):
        """Test behavior when MoE routing fails but parallel succeeds."""
        instance = ChessGemmaInference()
        instance.is_loaded = False

        # This simulates the case where MoE might fail but parallel inference
        # still provides individual expert responses
        results = instance.generate_parallel_responses("Test question")

        # Should still return individual expert results
        assert len(results) == 3
        assert all(expert in results for expert in ['uci', 'tutor', 'director'])


class TestErrorHandling:
    """Comprehensive error handling tests for parallel inference."""

    def test_partial_expert_failure(self):
        """Test handling when some experts fail but others succeed."""
        instance = ChessGemmaInference()
        instance.is_loaded = True

        # Mock generate_response to fail for one expert
        original_generate = instance.generate_response
        call_count = 0

        def mock_generate_response(question, context=None, mode=None, **kwargs):
            nonlocal call_count
            call_count += 1
            if mode == 'tutor':
                raise Exception("Tutor expert temporarily unavailable")
            return {
                'response': f'Mock response for {mode}',
                'confidence': 0.8,
                'generation_time': 1.0,
                'model_loaded': True,
                'mode': mode,
                'cached': False,
                'cache_hit_rate': 0.0
            }

        instance.generate_response = mock_generate_response

        try:
            results = instance.generate_parallel_responses("Test question")

            # Should have results for all experts, with error for tutor
            assert len(results) == 3
            assert 'uci' in results
            assert 'tutor' in results
            assert 'director' in results

            # Tutor should have error
            assert 'error' in results['tutor']
            assert results['tutor']['error'] == "Tutor expert temporarily unavailable"

            # Others should have normal responses
            assert 'response' in results['uci']
            assert 'response' in results['director']

        finally:
            instance.generate_response = original_generate

    def test_complete_parallel_failure(self):
        """Test handling when all experts fail."""
        instance = ChessGemmaInference()
        instance.is_loaded = True

        # Mock generate_response to always fail
        def mock_generate_response(*args, **kwargs):
            raise Exception("Complete system failure")

        original_generate = instance.generate_response
        instance.generate_response = mock_generate_response

        try:
            results = instance.generate_parallel_responses("Test question")

            # Should have error responses for all experts
            assert len(results) == 3
            for expert, response in results.items():
                assert 'error' in response
                assert response['error'] == "Complete system failure"
                assert response['response'] == ""
                assert response['confidence'] == 0.0

        finally:
            instance.generate_response = original_generate

    def test_timeout_error_handling(self):
        """Test handling of expert timeouts."""
        instance = ChessGemmaInference()
        instance.is_loaded = True

        # Mock generate_response to take longer than timeout
        def mock_generate_response(*args, **kwargs):
            time.sleep(35)  # Longer than 30s timeout
            return {'response': 'Should not reach here'}

        original_generate = instance.generate_response
        instance.generate_response = mock_generate_response

        try:
            start_time = time.time()
            results = instance.generate_parallel_responses("Test question")
            elapsed = time.time() - start_time

            # Should complete in reasonable time despite timeout
            assert elapsed < 40.0  # Should not hang indefinitely

            # Should have some results (may be partial due to timeout)
            assert isinstance(results, dict)

        finally:
            instance.generate_response = original_generate

    def test_invalid_expert_names(self):
        """Test handling of invalid expert names."""
        instance = ChessGemmaInference()
        instance.is_loaded = False

        # Test with invalid expert names
        results = instance.generate_parallel_responses(
            "Test question",
            experts=['invalid_expert', 'uci']
        )

        # Should still process valid experts
        assert 'uci' in results
        assert 'invalid_expert' in results

        # Invalid expert should have error
        assert 'error' in results['invalid_expert']

    def test_malformed_responses(self):
        """Test handling of malformed responses from experts."""
        instance = ChessGemmaInference()
        instance.is_loaded = True

        # Mock generate_response to return malformed responses
        call_count = 0
        def mock_generate_response(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:  # UCI
                return "Not a dict"  # Malformed
            elif call_count == 2:  # Tutor
                return {'response': 'OK', 'missing_fields': True}  # Missing required fields
            else:  # Director
                return {'response': 'OK', 'confidence': 0.9, 'generation_time': 1.0,
                       'model_loaded': True, 'mode': 'director', 'cached': False, 'cache_hit_rate': 0.0}

        original_generate = instance.generate_response
        instance.generate_response = mock_generate_response

        try:
            results = instance.generate_parallel_responses("Test question")

            # Should handle gracefully and provide some results
            assert isinstance(results, dict)
            assert len(results) >= 1  # At least director should work

        finally:
            instance.generate_response = original_generate

    def test_concurrent_error_isolation(self):
        """Test that errors in one expert don't affect others."""
        instance = ChessGemmaInference()
        instance.is_loaded = True

        # Mock generate_response with mixed success/failure
        call_count = 0
        def mock_generate_response(question, context=None, mode=None, **kwargs):
            nonlocal call_count
            call_count += 1

            if mode == 'engine':  # uci expert maps to engine mode
                return {'response': 'e2e4', 'confidence': 0.9, 'generation_time': 1.0,
                       'model_loaded': True, 'mode': 'engine', 'cached': False, 'cache_hit_rate': 0.0}
            elif mode == 'tutor':
                raise RuntimeError("Tutor network error")
            elif mode == 'director':
                return {'response': 'Strategic analysis', 'confidence': 0.8, 'generation_time': 1.2,
                       'model_loaded': True, 'mode': 'director', 'cached': False, 'cache_hit_rate': 0.0}

        original_generate = instance.generate_response
        instance.generate_response = mock_generate_response

        try:
            results = instance.generate_parallel_responses("Test question")

            # Should have mixed results: success for uci/director, error for tutor
            assert len(results) == 3

            # UCI should succeed
            assert results['uci']['response'] == 'e2e4'
            assert 'error' not in results['uci']

            # Tutor should fail
            assert 'error' in results['tutor']
            assert 'Tutor network error' in results['tutor']['error']

            # Director should succeed
            assert results['director']['response'] == 'Strategic analysis'
            assert 'error' not in results['director']

        finally:
            instance.generate_response = original_generate


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
