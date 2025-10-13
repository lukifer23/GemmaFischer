#!/usr/bin/env python3
"""
Quick test to verify the MoE system works without full evaluation suite
"""

import sys
import os
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

from inference.inference import ChessGemmaInference

def main():
    print("Testing MoE inference...")

    # Initialize inference
    inference = ChessGemmaInference()
    inference.load_model()

    # Test simple MoE routing
    test_question = "What is the best move for white? FEN: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

    print(f"Testing question: {test_question[:50]}...")

    try:
        response = inference.generate_response(test_question)
        print(f"Response received: {len(response['response'])} chars")
        print(f"Expert used: {response.get('expert', 'unknown')}")
        print("✅ MoE inference working!")
    except Exception as e:
        print(f"❌ Error: {e}")
        return

    # Test engine analysis (this should trigger lazy initialization)
    print("\nTesting engine analysis...")
    try:
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        analysis = inference._ensure_hybrid_engine().analyze(fen)
        print(f"Engine analysis: {analysis.best_move}")
        print("✅ Engine analysis working!")
    except Exception as e:
        print(f"❌ Engine analysis error: {e}")

if __name__ == '__main__':
    main()
