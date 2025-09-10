#!/usr/bin/env python3
"""
ChessGemma Web Interface

A Flask web application for chess Q&A with the fine-tuned Gemma model.
Features:
- Chess board visualization
- Real-time Q&A with the model
- Move validation and suggestions
- Interactive chess analysis
"""

from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
import json
import os
from pathlib import Path
import sys
import time
from typing import Dict, List, Any, Optional
import traceback

# Add the project root to the Python path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

# Import required modules
try:
    import torch
    from src.inference.inference import get_inference_instance
except ImportError as e:
    print(f"Warning: Could not import required module: {e}")
    torch = None

app = Flask(__name__,
            template_folder=str(Path(__file__).parent / 'templates'),
            static_folder=str(Path(__file__).parent / 'static'))
CORS(app)

# Configure Flask
app.config['SECRET_KEY'] = 'chess-gemma-secret-key'
app.config['SESSION_TYPE'] = 'filesystem'

# Global model cache
model_cache = {
    'model': None,
    'tokenizer': None,
    'last_used': None
}


class ChessModelInterface:
    """Web adapter that reuses the unified inference singleton."""

    def __init__(self):
        self._inference = get_inference_instance()
        self.is_loaded = False

    def load_model(self):
        ok = self._inference.load_model()
        self.is_loaded = ok
        return ok

    def generate_response(self, question: str, context: Optional[str] = None, max_length: int = 200) -> Dict[str, Any]:
        mode = 'tutor'
        return self._inference.generate_response(question, context=context, mode=mode, max_new_tokens=max_length)


# Initialize the model interface
chess_model = ChessModelInterface()


@app.route('/')
def index():
    """Main page with chess Q&A interface."""
    return render_template('index.html')


@app.route('/api/ask', methods=['POST'])
def ask_question():
    """API endpoint for chess questions."""
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        context = data.get('context', '').strip()

        if not question:
            return jsonify({
                'error': 'No question provided',
                'response': 'Please ask a chess-related question.',
                'confidence': 0.0
            })

        # Generate response
        result = chess_model.generate_response(question, context)

        # Add question to response for frontend
        result['question'] = question

        return jsonify(result)

    except Exception as e:
        print(f"API Error: {e}")
        traceback.print_exc()
        return jsonify({
            'error': str(e),
            'response': 'Sorry, there was an error processing your request.',
            'confidence': 0.0
        })


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    model_status = "loaded" if chess_model.is_loaded else "not_loaded"

    return jsonify({
        'status': 'healthy',
        'model_status': model_status,
        'timestamp': time.time()
    })


@app.route('/api/examples', methods=['GET'])
def get_examples():
    """Get example chess questions."""
    examples = [
        "What is the best opening move for White and why?",
        "Explain the concept of castling in chess.",
        "How should I evaluate material versus initiative in the middlegame?",
        "What are common mating patterns when the opponent's king is in the center?",
        "Give three practical tips for rook and pawn endgames.",
        "What is a fork in chess and how can I create one?",
        "Why is controlling the center important?",
        "How do I castle safely in chess?"
    ]

    return jsonify({'examples': examples})


@app.route('/api/debug/compare', methods=['POST'])
def debug_compare():
    """Compare engine/tutor/Stockfish suggestions for a FEN."""
    try:
        data = request.get_json()
        fen = data.get('fen', '').strip()
        depth = int(data.get('depth', 8))
        if not fen:
            return jsonify({'error': 'Missing fen'}), 400

        from src.inference.inference import get_inference_instance
        from src.inference.chess_engine import ChessEngineManager
        import chess
        import re

        inf = get_inference_instance()
        if not inf.load_model():
            return jsonify({'error': 'Model not loaded'}), 500

        board = chess.Board(fen)

        def parse_uci(text: str):
            m = re.findall(r'\b([a-h][1-8][a-h][1-8][qrbn]?)\b', text.lower())
            for u in m:
                try:
                    mv = chess.Move.from_uci(u)
                    if mv in board.legal_moves:
                        return u
                except Exception:
                    continue
            return None

        # Engine mode
        eng = inf.generate_response(
            f"Position: {fen}\nMode: Engine\nGenerate the best move in UCI format (e.g., e2e4). Respond with only the move.",
            mode='engine', max_new_tokens=12
        )
        eng_move = parse_uci(eng.get('response', ''))

        # Tutor mode
        tut = inf.generate_response(
            f"Position: {fen}\nMode: Tutor\nAnalyze step-by-step and end with a single UCI move line.",
            mode='tutor', max_new_tokens=160
        )
        tut_move = parse_uci(tut.get('response', ''))

        # Stockfish
        with ChessEngineManager() as ce:
            sf_mv = ce.get_best_move(board, depth=depth, time_limit_ms=0)
        sf_move = sf_mv.uci() if sf_mv else None

        return jsonify({
            'fen': fen,
            'engine_mode': {'text': eng.get('response'), 'move': eng_move},
            'tutor_mode': {'text': tut.get('response'), 'move': tut_move},
            'stockfish': sf_move
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/model_info', methods=['GET'])
def get_model_info():
    """Get information about the loaded model."""
    info = {
        'model_type': 'ChessGemma (Gemma-3 270M fine-tuned)',
        'fine_tuned_for': 'Chess Q&A and analysis',
        'capabilities': [
            'Opening analysis',
            'Tactical explanations',
            'Strategic concepts',
            'Endgame principles',
            'Move recommendations'
        ],
        'limitations': [
            'No real-time engine analysis',
            'Limited to text-based responses',
            'May not detect complex tactical combinations'
        ]
    }

    return jsonify(info)


if __name__ == '__main__':
    print("Starting ChessGemma Web Interface...")
    print("Visit http://localhost:5000 to use the application")

    # Try to preload the model
    print("Preloading model...")
    preload_success = chess_model.load_model()
    if preload_success:
        print("✓ Model preloaded successfully")
    else:
        print("⚠ Model preloading failed - will load on first request")

    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,  # Set to False for production
        threaded=True
    )
