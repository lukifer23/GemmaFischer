#!/usr/bin/env python3
"""
Inference Performance Benchmark for ChessGemma

Measures and compares inference performance across different configurations
to validate optimization improvements.
"""

import time
import statistics
from typing import Dict, List, Any
import json
import argparse
import sys
from pathlib import Path

# Add project root to Python path for imports
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


class InferencePerformanceBenchmark:
    """Benchmark inference performance across different scenarios."""

    def __init__(self):
        self.results = {}

    def benchmark_module_imports(self) -> Dict[str, Any]:
        """Benchmark module import times."""
        results = {}

        modules_to_test = [
            'src.inference.core_engine',
            'src.inference.caching',
            'src.inference.expert_manager',
        ]

        for module_name in modules_to_test:
            start_time = time.time()
            try:
                __import__(module_name)
                import_time = time.time() - start_time
                results[module_name] = {
                    'import_time': import_time,
                    'success': True,
                    'error': None
                }
            except Exception as e:
                import_time = time.time() - start_time
                results[module_name] = {
                    'import_time': import_time,
                    'success': False,
                    'error': str(e)
                }

        return results

    def benchmark_cache_operations(self, num_operations: int = 1000) -> Dict[str, Any]:
        """Benchmark cache operations performance."""
        try:
            from src.inference.caching import ChessInferenceCache

            cache = ChessInferenceCache(max_cache_size=1000)

            # Benchmark cache key creation
            start_time = time.time()
            for i in range(num_operations):
                key = cache._create_cache_key(f"test question {i}", None, "tutor", 200, 0.7, 0.9)
            key_creation_time = time.time() - start_time

            # Benchmark cache storage
            test_response = {"response": "test", "confidence": 0.8}
            start_time = time.time()
            for i in range(num_operations):
                cache._cache_response(f"key_{i}", test_response)
            storage_time = time.time() - start_time

            # Benchmark cache lookup
            start_time = time.time()
            for i in range(num_operations):
                cache._check_response_cache(f"key_{i % 100}")  # Mix of hits and misses
            lookup_time = time.time() - start_time

            return {
                'key_creation_avg_time': key_creation_time / num_operations,
                'storage_avg_time': storage_time / num_operations,
                'lookup_avg_time': lookup_time / num_operations,
                'cache_size': len(cache._response_cache),
                'success': True
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def benchmark_text_generation(self, num_samples: int = 10) -> Dict[str, Any]:
        """Benchmark text generation performance (without actual model)."""
        try:
            from src.inference.core_engine import ChessGemmaCoreEngine

            # Test with a mock core engine (won't load actual model)
            core_engine = ChessGemmaCoreEngine()

            # Generate some test prompts
            test_prompts = [
                "What is the best move for white?",
                "Analyze this chess position for black.",
                "Explain the Sicilian Defense opening.",
            ] * (num_samples // 3 + 1)

            generation_times = []

            for prompt in test_prompts[:num_samples]:
                start_time = time.time()
                # This will fail since no model is loaded, but we can measure the overhead
                result = core_engine.generate_text(prompt, max_new_tokens=50)
                generation_time = time.time() - start_time
                generation_times.append(generation_time)

            if generation_times:
                return {
                    'avg_generation_time': statistics.mean(generation_times),
                    'min_generation_time': min(generation_times),
                    'max_generation_time': max(generation_times),
                    'std_generation_time': statistics.stdev(generation_times) if len(generation_times) > 1 else 0,
                    'num_samples': len(generation_times),
                    'success': True
                }
            else:
                return {'success': False, 'error': 'No timing data collected'}

        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def run_full_benchmark(self) -> Dict[str, Any]:
        """Run the complete performance benchmark suite."""
        print("Running ChessGemma Inference Performance Benchmark")
        print("=" * 60)

        results = {
            'timestamp': time.time(),
            'benchmark_version': '1.0',
            'results': {}
        }

        # Benchmark 1: Module imports
        print("📦 Benchmarking module imports...")
        import_results = self.benchmark_module_imports()
        results['results']['module_imports'] = import_results
        print(f"   ✅ Import times: {import_results}")

        # Benchmark 2: Cache operations
        print("💾 Benchmarking cache operations...")
        cache_results = self.benchmark_cache_operations(num_operations=1000)
        results['results']['cache_operations'] = cache_results
        print(f"   ✅ Cache performance: {cache_results}")

        # Benchmark 3: Text generation overhead
        print("🤖 Benchmarking text generation overhead...")
        generation_results = self.benchmark_text_generation(num_samples=10)
        results['results']['text_generation'] = generation_results
        print(f"   ✅ Generation overhead: {generation_results}")

        # Summary
        total_time = time.time() - results['timestamp']
        results['total_benchmark_time'] = total_time

        print("\n" + "=" * 60)
        print("📊 BENCHMARK SUMMARY")
        print("=" * 60)
        print(f"Total benchmark time: {total_time:.2f}s")

        # Save results
        output_file = Path("inference_performance_benchmark.json")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"📄 Results saved to: {output_file}")

        return results


def main():
    """Command-line interface for performance benchmarking."""
    parser = argparse.ArgumentParser(description="ChessGemma Inference Performance Benchmark")
    parser.add_argument('--output', type=str, default='inference_performance_benchmark.json',
                       help='Output file for benchmark results')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick benchmark with fewer operations')

    args = parser.parse_args()

    benchmark = InferencePerformanceBenchmark()

    if args.quick:
        # Quick benchmark with reduced operations
        print("🏃 Running quick benchmark...")
        results = benchmark.run_full_benchmark()
    else:
        # Full benchmark
        results = benchmark.run_full_benchmark()

    # Print key metrics
    print("\n🎯 KEY METRICS:")
    print(f"Module import performance: {'✅' if results['results']['module_imports'].get('success', False) else '❌'}")
    print(f"Cache operations: {'✅' if results['results']['cache_operations'].get('success', False) else '❌'}")
    print(f"Text generation overhead: {'✅' if results['results']['text_generation'].get('success', False) else '❌'}")


if __name__ == '__main__':
    main()
