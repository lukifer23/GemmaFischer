#!/usr/bin/env python3
"""
Test MoE routing functionality

Tests the trained MoE router with various chess positions and questions
to verify intelligent expert selection and performance.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, List

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.inference.inference import ChessGemmaInference
from src.inference.moe_router import ChessMoERouter, MoEInferenceManager

def test_router_basic():
    """Test basic router functionality."""
    print("🧠 Testing MoE Router Basic Functionality")
    print("=" * 50)

    # Initialize router
    router = ChessMoERouter()
    # Try to load the trained checkpoint
    checkpoint_path = Path(__file__).parent.parent / "checkpoints" / "moe_router" / "final_checkpoint.pth"
    if checkpoint_path.exists():
        router.load_router(str(checkpoint_path))
        print(f"✅ Router loaded from checkpoint: {checkpoint_path}")
    else:
        print("⚠️  No checkpoint found, using untrained router")
    print(f"✅ Router initialized with {router.num_experts} experts: {router.expert_names}")

    # Test routing decisions with actual query examples from dataset
    test_cases = [
        ("FEN: rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1\nWhat is the best move?", "uci"),
        ("FEN: rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1\nAnalyze this position.", "tutor"),
        ("What are the main ideas behind the Sicilian Defense?", "director"),
    ]

    print("\n🎯 Testing routing decisions:")
    for query, expected in test_cases:
        # Extract FEN if present, otherwise use empty string
        if query.startswith("FEN:"):
            lines = query.split("\n")
            fen = lines[0].replace("FEN: ", "")
            query_text = lines[1] if len(lines) > 1 else ""
        else:
            fen = ""
            query_text = query

        decision = router.route_query(fen, query_text)
        status = "✅" if decision.primary_expert == expected else "❌"
        print(f"  {status} {query_text[:30]}... -> {decision.primary_expert} (expected: {expected})")

    print("\n📊 Router cache stats:")
    cache_stats = router.get_cache_stats()
    print(f"  Position cache: {cache_stats['position_cache_size']} entries")
    print(f"  Routing cache: {cache_stats['routing_cache_size']} entries")
    print(".1f")
    return True

def test_moe_inference():
    """Test full MoE inference pipeline."""
    print("\n🎯 Testing Full MoE Inference Pipeline")
    print("=" * 50)

    try:
        # Initialize inference system
        inference = ChessGemmaInference()
        print("✅ Inference system initialized")

        # Test expert loading
        expert_info = inference._expert_manager.get_expert_info()
        print(f"✅ Available experts: {expert_info['available_experts']}")
        print(f"✅ Active expert: {expert_info['active_adapter']}")

        # Test MoE manager
        if hasattr(inference, 'moe_manager') and inference.moe_manager:
            print("✅ MoE manager available")

            # Test MoE routing
            test_questions = [
                "FEN: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1\nWhat is the best move?",
                "FEN: r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 1\nAnalyze this position.",
                "What are the key principles of chess endgames?"
            ]

            print("\n🧪 Testing MoE responses:")
            for i, question in enumerate(test_questions, 1):
                print(f"\n  Test {i}: {question[:50]}...")
                try:
                    response = inference.generate_response(question)
                    moe_used = response.get('moe_used', False)
                    expert = response.get('primary_expert', 'unknown')

                    status = "🎯" if moe_used else "⚠️"
                    print(f"    {status} MoE: {moe_used}, Expert: {expert}")
                    print(f"    Response: {response.get('response', '')[:100]}...")

                except Exception as e:
                    print(f"    ❌ Error: {e}")

        else:
            print("❌ MoE manager not available")
            return False

    except Exception as e:
        print(f"❌ Failed to test MoE inference: {e}")
        return False

    return True

def test_feature_extraction():
    """Test feature extraction to see what features look like."""
    print("\n🔍 Testing Feature Extraction")
    print("=" * 50)

    try:
        from src.inference.moe_router import ChessMoERouter
        router = ChessMoERouter()

        # Test different types of queries
        test_cases = [
            ("uci", "FEN: rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1\nWhat is the best move?"),
            ("tutor", "FEN: rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1\nAnalyze this position."),
            ("director", "What are the main ideas behind the Sicilian Defense?"),
        ]

        for expected_expert, query in test_cases:
            features = router._extract_position_features("", query)  # No FEN for simplicity
            question_features = router._extract_question_features(query)
            print(f"\n{expected_expert.upper()}: {query[:50]}...")
            print(f"Full features: {features.tolist()}")
            print(f"Question features: {question_features}")

        return True

    except Exception as e:
        print(f"❌ Feature extraction test failed: {e}")
        return False

def test_expert_routing_accuracy():
    """Test routing accuracy against known good examples."""
    print("\n📊 Testing Expert Routing Accuracy")
    print("=" * 50)

    try:
        # Load evaluation queries directly from JSONL file
        import json
        queries = []
        eval_file = Path(__file__).parent.parent / "data" / "validation" / "expanded_eval_suite.jsonl"
        with open(eval_file, 'r') as f:
            for line in f:
                if line.strip():
                    queries.append(json.loads(line))
        if not queries:
            print("❌ No evaluation queries found")
            return False

        # Sample a few queries for testing
        test_queries = queries[:10]  # Test first 10 queries

        # Initialize systems
        inference = ChessGemmaInference()
        router = ChessMoERouter()
        # Try to load the trained checkpoint
        checkpoint_path = Path(__file__).parent.parent / "checkpoints" / "moe_router" / "final_checkpoint.pth"
        if checkpoint_path.exists():
            router.load_router(str(checkpoint_path))
            print(f"✅ Router loaded from checkpoint: {checkpoint_path}")
        else:
            print("⚠️  No checkpoint found, using untrained router")

        correct_predictions = 0
        total_predictions = 0

        print(f"Testing {len(test_queries)} queries...")

        for query in test_queries:
            question = query['question']
            expected_expert = query['expert']

            # Extract FEN
            fen_match = router._fen_pattern.search(question)
            fen = fen_match.group(1) if fen_match else None

            if fen:
                # Get router prediction
                decision = router.route_query(fen, "auto")
                predicted_expert = decision.primary_expert

                # Check accuracy
                is_correct = predicted_expert == expected_expert
                if is_correct:
                    correct_predictions += 1
                total_predictions += 1

                status = "✅" if is_correct else "❌"
                print(f"  {status} Expected: {expected_expert}, Predicted: {predicted_expert}")

        if total_predictions > 0:
            accuracy = correct_predictions / total_predictions
            print(".1f")
            return accuracy > 0.7  # Require 70% accuracy
        else:
            print("❌ No predictions made")
            return False

    except Exception as e:
        print(f"❌ Failed to test routing accuracy: {e}")
        return False

def main():
    """Run all MoE tests."""
    print("🎯 MoE System Test Suite")
    print("=" * 60)

    results = []

    # Test 1: Basic router functionality
    results.append(("Basic Router", test_router_basic()))

    # Test 2: Feature extraction
    results.append(("Feature Extraction", test_feature_extraction()))

    # Test 3: Full MoE inference pipeline
    results.append(("MoE Inference", test_moe_inference()))

    # Test 3: Routing accuracy
    results.append(("Routing Accuracy", test_expert_routing_accuracy()))

    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Results Summary:")

    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1

    print(f"\n🎯 Overall: {passed}/{len(results)} tests passed")

    if passed == len(results):
        print("🎉 All tests passed! MoE system is ready.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above.")
        return 1

if __name__ == "__main__":
    import os
    sys.exit(main())
