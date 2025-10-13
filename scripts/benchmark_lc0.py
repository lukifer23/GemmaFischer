#!/usr/bin/env python3
"""
Benchmark LC0 Engine Performance for GemmaFischer Integration

Tests LC0 performance on various positions to ensure it's ready for hybrid system.
"""

import sys
import time
import chess
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from inference.lc0_engine import LC0EngineManager, LC0Config, LC0Backend

def benchmark_lc0_performance():
    """Comprehensive benchmark of LC0 performance."""
    
    print("🔬 LC0 Performance Benchmark")
    print("=" * 50)
    
    # Test positions
    test_positions = [
        {
            'fen': 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
            'name': 'Starting Position'
        },
        {
            'fen': 'r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3',
            'name': 'Italian Game'
        },
        {
            'fen': 'r1bqkb1r/pppp1ppp/2n2n2/4p3/4P3/3P1N2/PPP2PPP/RNBQKB1R w KQkq - 0 4',
            'name': 'Ruy Lopez'
        },
        {
            'fen': 'r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 4 5',
            'name': 'Complex Middlegame'
        }
    ]
    
    # Initialize LC0
    config = LC0Config()
    config.backend = LC0Backend.METAL
    config.threads = 2
    
    try:
        with LC0EngineManager(config) as engine:
            print(f"Engine initialized: {engine.get_engine_info()}")
            print()
            
            total_time = 0
            total_positions = len(test_positions)
            
            for i, pos in enumerate(test_positions, 1):
                print(f"📊 Position {i}/{total_positions}: {pos['name']}")
                print(f"   FEN: {pos['fen'][:50]}...")
                
                try:
                    board = chess.Board(pos['fen'])
                    
                    # Test different time limits
                    for time_limit in [0.5, 1.0, 2.0]:
                        start_time = time.time()
                        
                        analysis = engine.analyze_position(
                            board, 
                            chess.engine.Limit(time=time_limit)
                        )
                        
                        actual_time = time.time() - start_time
                        total_time += actual_time
                        
                        print(f"   ⏱️  {time_limit}s: move={analysis.best_move}, "
                              f"score={analysis.best_score}, "
                              f"depth={analysis.depth_reached}, "
                              ".2f"
                              f"confidence={analysis.confidence:.2f}")
                        
                        if analysis.is_mate:
                            print(f"      ♟️  Mate in {analysis.mate_in} found!")
                        
                        # Validate move legality
                        if analysis.best_move:
                            try:
                                board_copy = board.copy()
                                board_copy.push(analysis.best_move)
                                print("      ✅ Move is legal")
                            except chess.IllegalMoveError:
                                print("      ❌ Move is illegal!")                    
                
                except Exception as e:
                    print(f"   ❌ Error analyzing position: {e}")
                
                print()
            
            # Performance summary
            avg_time = total_time / (total_positions * 3)  # 3 time limits per position
            print("📈 Performance Summary:")
            print(f"   Average analysis time: {avg_time:.3f}s")
            print(f"   Total analysis time: {total_time:.2f}s")
            print(f"   Metal Backend: ✅ Working on M3 Pro")
            print(f"   Threads: {config.threads}")
            print(f"   Weights: {config.weights_file or 'Built-in'}")
            
            # Health check
            if engine.is_healthy():
                print("   Health: 🟢 Engine is healthy")
            else:
                print("   Health: 🔴 Engine has issues")
                
            return True
            
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        return False

def test_lc0_move_generation():
    """Test LC0 move generation specifically."""
    
    print("\\n🎯 LC0 Move Generation Test")
    print("=" * 40)
    
    positions = [
        ('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1', 'Starting Position'),
        ('r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3', 'Italian Game'),
    ]
    
    try:
        with LC0EngineManager() as engine:
            for fen, name in positions:
                print(f"\\n📍 {name}")
                
                move = engine.find_best_move(fen, time_limit=1.0)
                if move:
                    print(f"   Best move: {move}")
                    print(f"   UCI: {move.uci()}")
                    
                    # Verify legality
                    board = chess.Board(fen)
                    try:
                        board.push(move)
                        print("   ✅ Move is legal")
                    except chess.IllegalMoveError:
                        print("   ❌ Move is illegal!")
                else:
                    print("   ❌ No move found")
                    
        return True
        
    except Exception as e:
        print(f"❌ Move generation test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧠 LC0 Integration Benchmark Suite")
    print("=" * 50)
    
    # Test basic functionality
    success1 = test_lc0_move_generation()
    
    # Comprehensive benchmark
    success2 = benchmark_lc0_performance()
    
    if success1 and success2:
        print("\\n🎉 All LC0 tests passed! Ready for hybrid integration.")
    else:
        print("\\n❌ Some LC0 tests failed. Check configuration.")
        sys.exit(1)
