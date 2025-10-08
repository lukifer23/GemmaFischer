#!/usr/bin/env python3
"""Debug script to test basic model functionality"""

import sys
import os
sys.path.append('src')

from inference.inference import ChessGemmaInference

def test_basic_model():
    """Test if the model can generate anything at all"""
    print("🔍 Testing basic model functionality...")

    # Initialize inference
    inf = ChessGemmaInference()

    # Try to load model
    print("📦 Loading model...")
    if not inf.load_model():
        print("❌ Failed to load model")
        return False

    print("✅ Model loaded successfully")

    # Test basic generation
    test_prompt = "You are a chess tutor. Explain what a rook is."
    print(f"🧪 Testing generation with prompt: {test_prompt[:50]}...")

    try:
        result = inf.generate_response(
            question="Explain what a rook is in chess.",
            mode="tutor",
            max_new_tokens=50
        )

        print("📊 Generation result:")
        print(f"  Response length: {len(result.get('response', ''))}")
        print(f"  Confidence: {result.get('confidence', 0)}")
        print(f"  Error: {result.get('error', 'None')}")
        print(f"  Response preview: {result.get('response', '')[:100]}")

        if len(result.get('response', '')) > 0:
            print("✅ Model generated response successfully!")
            return True
        else:
            print("❌ Model returned empty response")
            return False

    except Exception as e:
        print(f"❌ Generation failed with error: {e}")
        return False

if __name__ == "__main__":
    success = test_basic_model()
    sys.exit(0 if success else 1)
