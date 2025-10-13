#!/usr/bin/env python3
"""
Run the ChessGemma web application.

Usage:
    python run_web_app.py

The web application will be available at http://localhost:5000
"""

import sys
import os
import socket
from pathlib import Path

def main():
    """Start the ChessGemma web application."""
    # Add the project root to Python path
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))

    try:
        # Import and run the web application
        from src.web.app import app, chess_model

        print("=" * 60)
        print("🚀 Starting ChessGemma Web Application")
        print("=" * 60)
        # Determine port with fallback if busy
        preferred_port = int(os.environ.get("CHESSGEMMA_PORT", "5000"))
        port = preferred_port
        for _ in range(50):
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                s.bind(("0.0.0.0", port))
                s.close()
                break
            except OSError:
                s.close()
                port += 1

        print(f"📍 URL: http://localhost:{port}")
        print("🎯 Features:")
        print("   • AI-powered chess Q&A")
        print("   • Interactive chess board")
        print("   • Real-time model responses")
        print("   • Example questions")
        print("=" * 60)

        # Warm the model at startup so first request is responsive
        try:
            print("⏳ Loading ChessGemma model (one-time startup cost)...")
            if chess_model.load_model():
                print("✅ Model loaded and adapters initialized.")
            else:
                print("⚠️  Model failed to load during startup; requests will retry on demand.")
        except Exception as preload_err:
            print(f"⚠️  Model preload encountered an error: {preload_err}")
            print("   The server will continue and attempt to load on first request.")

        # Start the Flask development server
        app.run(
            host='0.0.0.0',
            port=port,
            debug=False,  # Set to False for production
            threaded=True,
            use_reloader=False  # Disable reloader to avoid issues with model loading
        )

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you have activated the virtual environment:")
        print("  source .venv/bin/activate")
        print("And installed dependencies:")
        print("  pip install -r requirements.txt")
        sys.exit(1)

    except Exception as e:
        print(f"❌ Error starting web application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
