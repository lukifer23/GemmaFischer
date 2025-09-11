#!/usr/bin/env python3
"""
Run the ChessGemma web application.

Usage:
    python run_web_app.py

The web application will be available at http://localhost:5000
"""

import sys
import os
from pathlib import Path

def main():
    """Start the ChessGemma web application."""
    # Add the project root to Python path
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))

    try:
        # Import and run the web application
        from src.web.app import app

        print("=" * 60)
        print("🚀 Starting ChessGemma Web Application")
        print("=" * 60)
        print("📍 URL: http://localhost:5000")
        print("🎯 Features:")
        print("   • AI-powered chess Q&A")
        print("   • Interactive chess board")
        print("   • Real-time model responses")
        print("   • Example questions")
        print("=" * 60)

        # Start the Flask development server
        app.run(
            host='0.0.0.0',
            port=5000,
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
