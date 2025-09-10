#!/usr/bin/env python3
"""Debug chess engine initialization issues."""

import chess
import chess.engine
import subprocess
import sys
from pathlib import Path

def debug_stockfish():
    """Comprehensive debugging of Stockfish integration."""
    print("=" * 60)
    print("CHESS ENGINE DEBUG INFORMATION")
    print("=" * 60)

    # 1. Check Stockfish binary
    stockfish_path = "/opt/homebrew/bin/stockfish"
    print(f"1. Stockfish path: {stockfish_path}")
    print(f"   Path exists: {Path(stockfish_path).exists()}")

    if Path(stockfish_path).exists():
        try:
            result = subprocess.run([stockfish_path, "--help"], capture_output=True, text=True, timeout=5)
            print(f"   Stockfish version info: {result.stdout[:200]}...")
        except Exception as e:
            print(f"   Error running stockfish --help: {e}")

    # 2. Test basic chess library
    print(f"\n2. Testing python-chess library:")
    try:
        board = chess.Board()
        print(f"   Board created successfully: {board.fen()}")
        moves = list(board.legal_moves)[:5]
        print(f"   Legal moves: {[move.uci() for move in moves]}")
    except Exception as e:
        print(f"   Chess library error: {e}")
        return

    # 3. Test engine initialization step by step
    print("
3. Testing engine initialization:")

    # Step 3.1: Basic engine startup
    try:
        print("   Step 3.1: Basic engine startup...")
        engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        print("   ✓ Engine started successfully")
    except Exception as e:
        print(f"   ✗ Engine startup failed: {e}")
        print(f"   Exception type: {type(e)}")
        return

    # Step 3.2: Check supported options
    try:
        print("   Step 3.2: Checking supported options...")
        options = engine.options
        print(f"   ✓ Found {len(options)} supported options")
        print(f"   Sample options: {list(options.keys())[:10]}...")

        # Check for key options
        key_options = ['Threads', 'Hash', 'Skill Level', 'UCI_LimitStrength']
        print("   Key options support:")
        for opt in key_options:
            supported = opt in options
            print(f"   - {opt}: {'✓' if supported else '✗'}")

    except Exception as e:
        print(f"   ✗ Options check failed: {e}")
        engine.quit()
        return

    # Step 3.3: Test configuration
    try:
        print("   Step 3.3: Testing configuration...")
        # Only configure supported options
        config_options = {}
        if 'Threads' in options:
            config_options['Threads'] = 2
        if 'Hash' in options:
            config_options['Hash'] = 128
        if 'Skill Level' in options:
            config_options['Skill Level'] = 20

        if config_options:
            engine.configure(config_options)
            print(f"   ✓ Configured options: {list(config_options.keys())}")
        else:
            print("   ⚠ No supported configuration options found")

    except Exception as e:
        print(f"   ✗ Configuration failed: {e}")
        engine.quit()
        return

    # Step 3.4: Test basic analysis
    try:
        print("   Step 3.4: Testing basic analysis...")
        board = chess.Board()
        limit = chess.engine.Limit(depth=5, time=0.5)
        info = engine.analyse(board, limit)

        print("   ✓ Analysis successful")
        print(f"   Score: {info.get('score', 'N/A')}")
        print(f"   Depth: {info.get('depth', 'N/A')}")
        print(f"   Time: {info.get('time', 'N/A')}")

    except Exception as e:
        print(f"   ✗ Analysis failed: {e}")
        print(f"   Exception type: {type(e)}")
        engine.quit()
        return

    # Step 3.5: Test move generation
    try:
        print("   Step 3.5: Testing move generation...")
        board = chess.Board()
        result = engine.play(board, chess.engine.Limit(time=0.5))
        if result.move:
            print(f"   ✓ Best move found: {result.move.uci()}")
        else:
            print("   ⚠ No best move returned")

    except Exception as e:
        print(f"   ✗ Move generation failed: {e}")

    # Cleanup
    try:
        engine.quit()
        print("   ✓ Engine cleanup successful")
    except Exception as e:
        print(f"   ⚠ Engine cleanup warning: {e}")

    print("\n" + "=" * 60)
    print("DEBUG COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    debug_stockfish()
