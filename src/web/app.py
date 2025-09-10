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
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import required modules
try:
    import torch
    from scripts.inference import run_inference
except ImportError as e:
    print(f"Warning: Could not import required module: {e}")
    torch = None

app = Flask(__name__,
            template_folder=str(project_root / 'web_app' / 'templates'),
            static_folder=str(project_root / 'web_app' / 'static'))
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
    """Interface to the ChessGemma model with caching and error handling."""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.is_loaded = False

    def load_model(self):
        """Load the model on demand with caching."""
        if self.is_loaded and self.model is not None:
            return True

        try:
            # Import here to avoid loading at startup
            from transformers import AutoTokenizer, AutoModelForCausalLM
            from peft import PeftModel

            print("Loading ChessGemma model...")

            # Model paths
            model_path = project_root / 'models' / 'unsloth-gemma-3-270m-it' / 'models--unsloth--gemma-3-270m-it' / 'snapshots' / '23cf460f6bb16954176b3ddcc8d4f250501458a9'
            adapter_path = project_root / 'checkpoints' / 'lora_expanded' / 'checkpoint-50'

            # Load base model
            self.tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
            base_model = AutoModelForCausalLM.from_pretrained(
                str(model_path),
                local_files_only=True,
                device_map='auto',
                attn_implementation='eager'
            )

            # Load adapter if available
            if adapter_path.exists():
                print(f"Loading adapter from {adapter_path}")
                self.model = PeftModel.from_pretrained(base_model, str(adapter_path), is_trainable=False)
            else:
                print("Warning: No adapter found, using base model")
                self.model = base_model

            self.is_loaded = True
            print("Model loaded successfully!")
            return True

        except Exception as e:
            print(f"Error loading model: {e}")
            traceback.print_exc()
            return False

    def generate_response(self, question: str, context: Optional[str] = None, max_length: int = 200) -> Dict[str, Any]:
        """Generate a response to a chess question."""
        if not self.load_model():
            return {
                'error': 'Model failed to load',
                'response': 'Sorry, the chess model is currently unavailable.',
                'confidence': 0.0
            }

        try:
            # Prepare the prompt
            if context:
                prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
            else:
                prompt = f"Question: {question}\nAnswer:"

            # Tokenize
            inputs = self.tokenizer(prompt, return_tensors='pt').to(self.model.device)

            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    do_sample=True,
                    top_p=0.9,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )

            # Decode response
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract the answer part
            if 'Answer:' in full_response:
                answer = full_response.split('Answer:', 1)[1].strip()
            else:
                answer = full_response.replace(prompt, '').strip()

            # Basic confidence estimation (can be improved)
            confidence = min(0.9, len(answer.split()) / 50)  # Rough heuristic

            return {
                'response': answer,
                'confidence': confidence,
                'model_loaded': True,
                'generation_time': time.time()
            }

        except Exception as e:
            print(f"Error generating response: {e}")
            traceback.print_exc()
            return {
                'error': str(e),
                'response': 'Sorry, I encountered an error while processing your question.',
                'confidence': 0.0
            }


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
