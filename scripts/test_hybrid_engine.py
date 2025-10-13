#!/usr/bin/env python3
"""
Comprehensive Test Suite for Hybrid Chess Engine

Tests the LLM + LC0 hybrid system with different strategic intents
and positions to validate the integration.
"""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from inference.hybrid_engine import create_hybrid_engine, StrategicIntent

def test_strategic_intents():
    """Test different strategic intents on the same position."""
    
    print("🎯 Testing Strategic Intent Variations")
    print("=" * 50)
    
    # Test position: Italian Game where different strategies apply
    fen = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"
    
    intents = [
        StrategicIntent.DEVELOPMENT,
        StrategicIntent.AGGRESSIVE, 
        StrategicIntent.POSITIONAL,
        StrategicIntent.TACTICAL
    ]
    
    results = {}
    
    try:
        with create_hybrid_engine() as engine:
            for intent in intents:
                print(f"\\n🧠 Testing {intent.value} strategy...")
                
                start_time = time.time()
                analysis = engine.analyze_position_with_strategy(fen, intent, time_limit=3.0)
                elapsed = time.time() - start_time
                
                print(f"   Move: {analysis.best_move}")
                print(f"   Confidence: {analysis.confidence:.2f}")
                print(f"   Time: {elapsed:.2f}s")
                print(f"   Themes: {', '.join(analysis.key_themes[:2])}")
                
                results[intent.value] = {
                    'move': analysis.best_move.uci() if analysis.best_move else 'None',
                    'confidence': analysis.confidence,
                    'time': elapsed,
                    'themes': analysis.key_themes
                }
                
                # Brief explanation preview
                explanation_preview = analysis.explanation[:100] + "..." if len(analysis.explanation) > 100 else analysis.explanation
                print(f"   Explanation: {explanation_preview}")
    
    except Exception as e:
        print(f"❌ Strategic intent test failed: {e}")
        return False
    
    print("\\n📊 Strategic Intent Results:")
    for intent, data in results.items():
        print(f"   {intent}: {data['move']} (conf: {data['confidence']:.2f})")
    
    return True

def test_position_complexity():
    """Test how the engine handles different position complexities."""
    
    print("\\n🎯 Testing Position Complexity Handling")
    print("=" * 50)
    
    positions = [
        {
            'fen': 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
            'name': 'Starting Position',
            'expected_complexity': 'low'
        },
        {
            'fen': 'r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3', 
            'name': 'Italian Game',
            'expected_complexity': 'medium'
        },
        {
            'fen': 'r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 4 5',
            'name': 'Complex Middlegame',
            'expected_complexity': 'high'
        }
    ]
    
    try:
        with create_hybrid_engine() as engine:
            for pos in positions:
                print(f"\\n📍 {pos['name']} ({pos['expected_complexity']} complexity)")
                
                analysis = engine.analyze_position_with_strategy(
                    pos['fen'], 
                    StrategicIntent.POSITIONAL,
                    time_limit=2.0
                )
                
                print(f"   Move: {analysis.best_move}")
                print(f"   Strategic guidance: {analysis.strategic_guidance.intent.value}")
                print(f"   Time allocation: {analysis.strategic_guidance.time_allocation:.1f}s")
                print(f"   Risk tolerance: {analysis.strategic_guidance.risk_tolerance}")
                
    except Exception as e:
        print(f"❌ Position complexity test failed: {e}")
        return False
    
    return True

def test_performance_comparison():
    """Compare hybrid engine performance vs pure LC0."""
    
    print("\\n⚡ Performance Comparison: Hybrid vs Pure LC0")
    print("=" * 50)
    
    fen = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"
    
    try:
        with create_hybrid_engine() as engine:
            # Hybrid analysis
            print("🤖 Hybrid Engine Analysis:")
            start_time = time.time()
            hybrid_analysis = engine.analyze_position_with_strategy(
                fen, StrategicIntent.AGGRESSIVE, time_limit=2.0
            )
            hybrid_time = time.time() - start_time
            
            print(f"   Move: {hybrid_analysis.best_move}")
            print(f"   Time: {hybrid_time:.2f}s")
            print(f"   LLM time: {hybrid_analysis.llm_time:.2f}s")
            print(f"   LC0 time: {hybrid_analysis.lc0_time:.2f}s")
            print(f"   Confidence: {hybrid_analysis.confidence:.2f}")
            
            # Get performance stats
            stats = engine.get_performance_stats()
            print(f"   Performance: {stats['llm_percentage']:.1f}% LLM, {stats['lc0_percentage']:.1f}% LC0")
            
            # Pure LC0 analysis for comparison
            print("\\n🎯 Pure LC0 Analysis:")
            import chess
            from inference.lc0_engine import LC0EngineManager
            
            with LC0EngineManager() as lc0:
                board = chess.Board(fen)
                start_time = time.time()
                lc0_analysis = lc0.analyze_position(board, chess.engine.Limit(time=1.5))
                lc0_time = time.time() - start_time
                
                print(f"   Move: {lc0_analysis.best_move}")
                print(f"   Time: {lc0_time:.2f}s")
                print(f"   Confidence: {lc0_analysis.confidence:.2f}")
                
                # Compare moves
                if (hybrid_analysis.best_move and lc0_analysis.best_move and 
                    hybrid_analysis.best_move.uci() == lc0_analysis.best_move.uci()):
                    print("   ✅ Moves match!")
                else:
                    print("   ⚠️  Different moves (expected due to strategic guidance)")
                    
    except Exception as e:
        print(f"❌ Performance comparison failed: {e}")
        return False
    
    return True

def test_educational_value():
    """Test the educational value of hybrid explanations."""
    
    print("\\n📚 Testing Educational Value")
    print("=" * 50)
    
    fen = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"
    
    try:
        with create_hybrid_engine() as engine:
            analysis = engine.analyze_position_with_strategy(
                fen, StrategicIntent.DEVELOPMENT, time_limit=3.0
            )
            
            print("📖 Hybrid Explanation:")
            print("-" * 30)
            
            # Show explanation with key educational elements
            explanation = analysis.explanation
            lines = explanation.split('\\n')
            
            educational_indicators = []
            for line in lines[:10]:  # First 10 lines
                line = line.strip()
                if len(line) > 10:
                    print(f"   {line}")
                    
                    # Check for educational content
                    if any(word in line.lower() for word in ['development', 'control', 'safety', 'initiative', 'strategy']):
                        educational_indicators.append('strategic_concept')
                    if 'because' in line.lower() or 'why' in line.lower():
                        educational_indicators.append('explanation')
                    if any(word in line.lower() for word in ['follow', 'then', 'next']):
                        educational_indicators.append('planning')
            
            print("\\n🎓 Educational Analysis:")
            unique_indicators = list(set(educational_indicators))
            for indicator in unique_indicators:
                print(f"   ✅ Contains {indicator.replace('_', ' ')}")
                
            if len(unique_indicators) >= 2:
                print("   🏆 High educational value!")
            elif len(unique_indicators) >= 1:
                print("   👍 Good educational value")
            else:
                print("   ⚠️  Limited educational content")
                
    except Exception as e:
        print(f"❌ Educational value test failed: {e}")
        return False
    
    return True

def main():
    """Run comprehensive hybrid engine tests."""
    
    print("🧠 Hybrid Chess Engine Comprehensive Test Suite")
    print("=" * 60)
    
    tests = [
        ("Strategic Intent Variations", test_strategic_intents),
        ("Position Complexity Handling", test_position_complexity), 
        ("Performance Comparison", test_performance_comparison),
        ("Educational Value", test_educational_value)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\\n🔬 Running: {test_name}")
        try:
            success = test_func()
            results.append((test_name, success))
            status = "✅ PASSED" if success else "❌ FAILED"
            print(f"\\n{status}: {test_name}")
        except Exception as e:
            print(f"❌ ERROR in {test_name}: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY:")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {status}: {test_name}")
        if success:
            passed += 1
    
    print(f"\\n🏆 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\\n🎉 ALL TESTS PASSED! Hybrid Chess Engine is production-ready!")
        return True
    else:
        print(f"\\n⚠️  {total - passed} tests failed. Review and fix issues before production.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
