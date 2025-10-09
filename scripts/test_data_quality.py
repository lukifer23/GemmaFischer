#!/usr/bin/env python3
"""
Test data quality and basic functionality without requiring full inference dependencies.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, List

def test_expert_data_quality():
    """Test that expert training data is properly formatted."""
    print("🧪 Testing Expert Data Quality")
    print("=" * 50)

    expert_files = {
        'uci': 'data/standardized/standardized_uci_expert.jsonl',
        'tutor': 'data/standardized/standardized_tutor_expert.jsonl',
        'director': 'data/standardized/standardized_director_expert_v2.jsonl'
    }

    results = {}

    for expert, filepath in expert_files.items():
        print(f"\n📋 Checking {expert.upper()} expert data...")

        if not Path(filepath).exists():
            print(f"❌ File not found: {filepath}")
            results[expert] = False
            continue

        try:
            count = 0
            valid_count = 0
            sample_entries = []

            with open(filepath, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    if line.strip():
                        count += 1
                        try:
                            entry = json.loads(line)

                            # Check required fields
                            required_fields = ['task', 'prompt', 'response', 'meta']
                            if not all(field in entry for field in required_fields):
                                print(f"  ❌ Line {line_num}: Missing required fields")
                                continue

                            # Check meta has FEN
                            if 'fen' not in entry.get('meta', {}):
                                print(f"  ❌ Line {line_num}: Missing FEN in meta")
                                continue

                            valid_count += 1

                            # Keep a few samples
                            if len(sample_entries) < 3:
                                sample_entries.append(entry)

                        except json.JSONDecodeError as e:
                            print(f"  ❌ Line {line_num}: Invalid JSON - {e}")
                            continue

                    if count >= 100:  # Test first 100 entries
                        break

            print(f"  ✅ Found {count} total entries, {valid_count} valid")
            print(".1f")
            # Show sample entries
            for i, entry in enumerate(sample_entries[:2]):
                prompt_preview = entry['prompt'][:80] + "..." if len(entry['prompt']) > 80 else entry['prompt']
                response_preview = entry['response'][:40] + "..." if len(entry['response']) > 40 else entry['response']
                print(f"    Sample {i+1}: {prompt_preview}")
                print(f"              -> {response_preview}")

            results[expert] = valid_count > 0

        except Exception as e:
            print(f"❌ Error reading {filepath}: {e}")
            results[expert] = False

    return results

def test_evaluation_data():
    """Test evaluation data for MoE router training."""
    print("\n🎯 Testing Evaluation Data for MoE Router")
    print("=" * 50)

    eval_file = 'data/validation/eval_suite.jsonl'

    if not Path(eval_file).exists():
        print(f"❌ Evaluation file not found: {eval_file}")
        return False

    try:
        queries = []
        expert_counts = {}

        with open(eval_file, 'r') as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    queries.append(entry)

                    expert = entry.get('expert', 'unknown')
                    expert_counts[expert] = expert_counts.get(expert, 0) + 1

        print(f"✅ Found {len(queries)} evaluation queries")
        print("📊 Expert distribution:")
        for expert, count in expert_counts.items():
            print(f"   {expert}: {count}")

        # Check for required fields
        valid_queries = 0
        for query in queries:
            if all(field in query for field in ['question', 'expert', 'category']):
                valid_queries += 1

        print(".1f")
        # Show sample queries
        print("\n📝 Sample queries:")
        for i, query in enumerate(queries[:3]):
            question_preview = query['question'][:60] + "..." if len(query['question']) > 60 else query['question']
            print(f"   {i+1}. [{query['expert']}] {question_preview}")

        return len(queries) >= 20 and valid_queries == len(queries)

    except Exception as e:
        print(f"❌ Error reading evaluation data: {e}")
        return False

def test_router_initialization():
    """Test basic router initialization without dependencies."""
    print("\n🧠 Testing Router Initialization")
    print("=" * 50)

    try:
        # Add src to path
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

        # Import only what we need
        from src.inference.moe_router import ChessMoERouter

        router = ChessMoERouter()
        print(f"✅ Router initialized with {router.num_experts} experts: {router.expert_names}")
        print(f"✅ Feature dimension: {router.feature_dim}")

        # Test basic functionality
        test_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        test_question = "What is the best move?"

        features = router._extract_position_features(test_fen, test_question)
        print(f"✅ Feature extraction works: {len(features)} features generated")

        # Test routing decision
        decision = router.route_query(test_fen, "engine")
        print(f"✅ Routing works: {decision.primary_expert} (confidence: {decision.confidence_score:.3f})")

        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   This is expected without full dependencies installed")
        return False
    except Exception as e:
        print(f"❌ Router initialization failed: {e}")
        return False

def main():
    """Run all data quality tests."""
    print("🎯 Data Quality Test Suite")
    print("=" * 60)

    results = []

    # Test 1: Expert data quality
    expert_results = test_expert_data_quality()
    results.append(("Expert Data Quality", all(expert_results.values())))

    # Test 2: Evaluation data
    results.append(("Evaluation Data", test_evaluation_data()))

    # Test 3: Router initialization
    results.append(("Router Initialization", test_router_initialization()))

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
        print("🎉 All data quality tests passed! Ready for training.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
