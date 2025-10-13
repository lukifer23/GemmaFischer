#!/usr/bin/env python3
"""
Test data quality and basic functionality without requiring full inference dependencies.
"""

import os
import sys
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import chess

PLACEHOLDER_TOKENS = {
    "[tactical_move]",
    "[discovered_attack]",
    "[double_attack]",
    "[skewer]",
    "[fork]",
}

MOVE_PATTERN = re.compile(r"\b([a-h][1-8][a-h][1-8][qrbn]?)\b", re.IGNORECASE)


def _find_best_move(expert: str, response: str) -> Optional[str]:
    """Extract the best-move token from the response text."""
    lower = response.lower()

    if expert == "uci":
        match = MOVE_PATTERN.search(lower)
        return match.group(1) if match else None

    for marker in (
        r"best move[:\s]+([a-h][1-8][a-h][1-8][qrbn]?)",
        r"final move[:\s]+([a-h][1-8][a-h][1-8][qrbn]?)",
    ):
        match = re.search(marker, lower)
        if match:
            return match.group(1)

    matches = MOVE_PATTERN.findall(lower)
    if matches:
        return matches[-1]

    return None


def _validate_move(board: chess.Board, move_str: str) -> Tuple[bool, Optional[str]]:
    """Check whether the provided move is legal for the given board."""
    try:
        move = chess.Move.from_uci(move_str)
    except ValueError:
        return False, "invalid_format"

    if move not in board.legal_moves:
        return False, "illegal"

    return True, None


def test_expert_data_quality():
    """Test that expert training data is properly formatted."""
    print("🧪 Testing Expert Data Quality")
    print("=" * 50)

    expert_files = {
        'uci': 'data/standardized/standardized_uci_expert_v2.jsonl',
        'tutor': 'data/standardized/standardized_tutor_expert_v2.jsonl',
        'director': 'data/standardized/standardized_director_expert_v3.jsonl'
    }

    results = {}
    stats: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    seen_fingerprints: Dict[str, set] = defaultdict(set)
    sample_limit = 250

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

            placeholder_hits = 0

            with open(filepath, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if count >= sample_limit:
                        break

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

                            response: str = entry.get("response", "")
                            stats[expert]["total"] += 1

                            if any(token in response for token in PLACEHOLDER_TOKENS):
                                print(f"  ❌ Line {line_num}: Placeholder token detected in response")
                                placeholder_hits += 1
                                stats[expert]["placeholder"] += 1
                                continue

                            if expert == 'tutor' and 'best move:' not in response.lower():
                                print(f"  ❌ Line {line_num}: Tutor response missing 'best move:' marker")
                                stats[expert]["missing_best_marker"] += 1
                                continue

                            if expert == 'director' and (('final move' not in response.lower()) and entry.get('meta', {}).get('best_move')):
                                print(f"  ⚠️  Line {line_num}: Director response missing 'Final move (UCI):' marker")
                                stats[expert]["missing_final_marker"] += 1

                            fen = entry["meta"]["fen"]
                            try:
                                board = chess.Board(fen)
                            except ValueError as fen_err:
                                print(f"  ❌ Line {line_num}: Invalid FEN ({fen_err})")
                                stats[expert]["invalid_fen"] += 1
                                continue

                            move_token = _find_best_move(expert, response)
                            if not move_token:
                                print(f"  ❌ Line {line_num}: Unable to extract move token")
                                stats[expert]["missing_move_token"] += 1
                                continue

                            is_legal, reason = _validate_move(board, move_token)
                            if not is_legal:
                                label = "illegal move" if reason == "illegal" else "invalid move format"
                                print(f"  ❌ Line {line_num}: {label} '{move_token}'")
                                stats[expert]["illegal_moves"] += 1
                                continue

                            meta_move = entry.get("meta", {}).get("best_move")
                            if meta_move and meta_move.lower() != move_token.lower():
                                print(
                                    f"  ❌ Line {line_num}: Best move mismatch "
                                    f"(response: {move_token}, meta: {meta_move})"
                                )
                                stats[expert]["meta_mismatch"] += 1
                                continue

                            fingerprint = f"{fen}|{move_token}|{entry['prompt'][:50]}"
                            if fingerprint in seen_fingerprints[expert]:
                                stats[expert]["duplicates"] += 1
                            else:
                                seen_fingerprints[expert].add(fingerprint)

                            valid_count += 1

                            # Keep a few samples
                            if len(sample_entries) < 3:
                                sample_entries.append(entry)

                        except json.JSONDecodeError as e:
                            print(f"  ❌ Line {line_num}: Invalid JSON - {e}")
                            continue

            print(f"  ✅ Found {count} total entries, {valid_count} valid")
            if placeholder_hits:
                print(f"  ⚠️  Placeholder responses skipped: {placeholder_hits}")

            # Show sample entries
            for i, entry in enumerate(sample_entries[:2]):
                prompt_preview = entry['prompt'][:80] + "..." if len(entry['prompt']) > 80 else entry['prompt']
                response_preview = entry['response'][:40] + "..." if len(entry['response']) > 40 else entry['response']
                print(f"    Sample {i+1}: {prompt_preview}")
                print(f"              -> {response_preview}")

            stats[expert]["valid"] = valid_count
            results[expert] = (
                valid_count > 0
                and stats[expert]["invalid_fen"] == 0
                and stats[expert]["illegal_moves"] == 0
                and stats[expert]["meta_mismatch"] == 0
                and stats[expert]["missing_move_token"] == 0
            )

            metrics = stats[expert]
            if any(metrics.get(key, 0) > 0 for key in ("duplicates", "missing_best_marker", "missing_final_marker")):
                print("  ℹ️  Additional observations:")
                if metrics.get("duplicates"):
                    print(f"     • Potential duplicates spotted: {metrics['duplicates']}")
                if metrics.get("missing_best_marker"):
                    print(f"     • Missing tutor markers: {metrics['missing_best_marker']}")
                if metrics.get("missing_final_marker"):
                    print(f"     • Missing director final-move markers: {metrics['missing_final_marker']}")

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

        if valid_queries != len(queries):
            print(f"  ❌ Schema errors detected in {len(queries) - valid_queries} queries")
        else:
            print(f"  ✅ Schema integrity: {valid_queries}/{len(queries)} queries valid")
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
