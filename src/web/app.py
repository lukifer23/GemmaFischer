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
import torch
from typing import Dict, List, Any, Optional
import traceback
import psutil
import threading
from datetime import datetime
import subprocess
import threading as _threading
import queue as _queue

# Add the project root to the Python path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

# Import required modules
try:
    import torch
    import chess
    from src.inference.inference import get_inference_instance
    from src.inference.chess_engine import ChessEngineManager
    from src.inference.uci_utils import extract_first_legal_move, extract_first_legal_move_uci
    from src.web.chess_game import ChessGame, ChessRAG
    from src.web.stockfish_match import StockfishMatch
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

# Performance monitoring
performance_stats = {
    'request_count': 0,
    'total_response_time': 0.0,
    'avg_response_time': 0.0,
    'max_response_time': 0.0,
    'min_response_time': float('inf'),
    'memory_usage_mb': 0.0,
    'cpu_usage_percent': 0.0,
    'tokens_per_second': 0.0,
    'context_length': 0,
    'last_request_time': None
}

# Thread lock for stats
stats_lock = threading.Lock()


def get_system_stats():
    """Get current system resource usage."""
    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        cpu_percent = process.cpu_percent()
        
        return {
            'memory_mb': memory_info.rss / 1024 / 1024,
            'cpu_percent': cpu_percent,
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        return {
            'memory_mb': 0.0,
            'cpu_percent': 0.0,
            'timestamp': datetime.now().isoformat(),
            'error': str(e)
        }


def log_performance_stats(question, response_time, response_length, context_length=0):
    """Log performance statistics for a request."""
    global performance_stats
    
    with stats_lock:
        performance_stats['request_count'] += 1
        performance_stats['total_response_time'] += response_time
        performance_stats['avg_response_time'] = performance_stats['total_response_time'] / performance_stats['request_count']
        performance_stats['max_response_time'] = max(performance_stats['max_response_time'], response_time)
        performance_stats['min_response_time'] = min(performance_stats['min_response_time'], response_time)
        performance_stats['context_length'] = context_length
        performance_stats['last_request_time'] = datetime.now().isoformat()
        
        # Calculate tokens per second (rough estimate)
        if response_time > 0:
            estimated_tokens = len(response_length.split()) * 1.3  # rough token estimation
            performance_stats['tokens_per_second'] = estimated_tokens / response_time
        
        # Get current system stats
        sys_stats = get_system_stats()
        performance_stats['memory_usage_mb'] = sys_stats['memory_mb']
        performance_stats['cpu_usage_percent'] = sys_stats['cpu_percent']
        
        # Log to terminal
        print(f"\n{'='*60}")
        print(f"📊 PERFORMANCE METRICS - Request #{performance_stats['request_count']}")
        print(f"{'='*60}")
        print(f"⏱️  Response Time: {response_time:.3f}s")
        print(f"📈 Avg Response Time: {performance_stats['avg_response_time']:.3f}s")
        print(f"⚡ Min/Max Response Time: {performance_stats['min_response_time']:.3f}s / {performance_stats['max_response_time']:.3f}s")
        print(f"🧠 Memory Usage: {performance_stats['memory_usage_mb']:.1f} MB")
        print(f"💻 CPU Usage: {performance_stats['cpu_usage_percent']:.1f}%")
        print(f"🚀 Tokens/Second: {performance_stats['tokens_per_second']:.1f}")
        print(f"📝 Context Length: {context_length} chars")
        print(f"📏 Response Length: {len(response_length)} chars")
        print(f"❓ Question: {question[:100]}{'...' if len(question) > 100 else ''}")
        print(f"⏰ Timestamp: {performance_stats['last_request_time']}")
        print(f"{'='*60}\n")


class ChessModelInterface:
    """Web adapter that reuses the unified inference singleton."""

    def __init__(self):
        self._inference = get_inference_instance()
        self.is_loaded = False

    def load_model(self):
        ok = self._inference.load_model()
        self.is_loaded = ok
        return ok

    def generate_response(self, question: str, context: Optional[str] = None, mode: str = 'tutor', max_length: int = 200) -> Dict[str, Any]:
        print(f"🎯 ChessModel.generate_response called with mode: {mode}")
        # Ensure model is loaded on first request
        if not self.is_loaded:
            print("🔄 Loading model on-demand for web request...")
            if not self.load_model():
                return {
                    'error': 'Model not loaded',
                    'response': '',
                    'confidence': 0.0
                }
        # Minimal MoE adapter switching for web paths
        try:
            if mode == 'engine':
                self._inference.set_active_adapter('uci')
            elif mode == 'tutor':
                self._inference.set_active_adapter('tutor')
            elif mode == 'director':
                self._inference.set_active_adapter('director')
        except Exception:
            pass
        return self._inference.generate_response(question, context=context, mode=mode, max_new_tokens=max_length)


# Initialize the model interface
chess_model = ChessModelInterface()

# Initialize chess game and RAG
chess_game = ChessGame()
chess_rag = ChessRAG()
stockfish_match = None


# ---------------------------
# Training Job Manager
# ---------------------------

class TrainingJob:
    def __init__(self):
        self.proc: Optional[subprocess.Popen] = None
        self.running: bool = False
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.args: Dict[str, Any] = {}
        self._log_q: _queue.Queue[str] = _queue.Queue(maxsize=10000)
        self._log_tail: list[str] = []
        self._lock = _threading.Lock()

    def _reader(self, stream):
        try:
            for line in iter(stream.readline, ''):
                with self._lock:
                    self._log_tail.append(line.rstrip())
                    if len(self._log_tail) > 500:
                        self._log_tail = self._log_tail[-500:]
        except Exception:
            pass

    def start(self, expert: str, steps: int, use_instruction: bool, disable_eval: bool, dataset_path: Optional[str] = None) -> bool:
        if self.running:
            return False
        cmd = [
            sys.executable,
            str(project_root / 'src' / 'training' / 'train_lora_poc.py'),
            '--expert', expert,
            '--config', 'auto',
            '--max_steps_override', str(int(steps)),
        ]
        if use_instruction:
            cmd.append('--use_instruction_collator')
        if disable_eval:
            cmd.append('--disable_eval')
        env = os.environ.copy()
        cwd = str(project_root)
        try:
            self.proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        except Exception:
            return False
        self.running = True
        self.start_time = time.time()
        self.end_time = None
        self.args = {
            'expert': expert,
            'steps': steps,
            'use_instruction': use_instruction,
            'disable_eval': disable_eval,
        }
        # reader thread
        t = _threading.Thread(target=self._reader, args=(self.proc.stdout,), daemon=True)
        t.start()
        # watcher thread
        def _watch():
            try:
                rc = self.proc.wait()
            finally:
                self.running = False
                self.end_time = time.time()
        _threading.Thread(target=_watch, daemon=True).start()
        return True

    def stop(self) -> bool:
        if not self.running or not self.proc:
            return False
        try:
            self.proc.terminate()
        except Exception:
            return False
        return True

    def status(self) -> Dict[str, Any]:
        with self._lock:
            logs = '\n'.join(self._log_tail[-200:])
        return {
            'running': self.running,
            'args': self.args,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'elapsed_sec': (time.time() - self.start_time) if (self.start_time and self.running) else None,
            'logs_tail': logs,
        }


TRAINING_JOB = TrainingJob()


@app.route('/')
def index():
    """Main page with chess Q&A interface."""
    return render_template('index.html')


@app.route('/api/ask', methods=['POST'])
def ask_question():
    """API endpoint for chess questions."""
    start_time = time.time()
    question = ""
    context = ""
    
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        context = data.get('context', '').strip()
        expert = data.get('expert', 'auto').strip().lower()

        if not question:
            return jsonify({
                'error': 'No question provided',
                'response': 'Please ask a chess-related question.',
                'confidence': 0.0
            })

        print(f"\n🎯 NEW REQUEST RECEIVED")
        print(f"📝 Question: {question}")
        print(f"📋 Context: {context if context else 'None'}")
        print(f"⏰ Start Time: {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")

        # Get RAG knowledge for the question
        rag_knowledge = chess_rag.get_relevant_knowledge(question)
        rag_context = f"Chess Knowledge: {rag_knowledge}\n\n" if rag_knowledge else ""
        enhanced_context = f"{rag_context}{context}" if context else rag_context
        
        print(f"🧠 RAG Knowledge: {rag_knowledge}")

        # Decide mode based on expert selector
        mode = 'tutor'
        if expert == 'uci':
            mode = 'engine'
        elif expert == 'tutor':
            mode = 'tutor'
        elif expert == 'director':
            mode = 'director'

        # Switch adapter explicitly by expert
        try:
            if expert in ('uci', 'tutor', 'director'):
                chess_model._inference.set_active_adapter(expert)
        except Exception:
            pass

        # If question contains a FEN, include it explicitly in context so tutor has state
        try:
            import re
            m = re.search(r"FEN:\s*([^\n]+)", question, flags=re.IGNORECASE)
            fen_from_q = m.group(1).strip() if m else None
            if fen_from_q:
                enhanced_context = f"Current position: {fen_from_q}\n\n{enhanced_context}" if enhanced_context else f"Current position: {fen_from_q}"
        except Exception:
            pass

        # Generate response with RAG context
        result = chess_model.generate_response(question, enhanced_context, mode=mode)
        # Detailed routing + context diagnostics
        try:
            info = chess_model._inference.get_model_info()
            print(f"🔧 Active adapter: {info.get('active_adapter')} | Available: {list(info.get('available_adapters', {}).keys())}")
            print(f"🧵 Prompt chars: {result.get('prompt_len_chars')} | Answer chars: {result.get('answer_len_chars')}")
        except Exception:
            pass
        
        # Log detailed performance metrics
        response_time = time.time() - start_time
        response_text = result.get('response', '')
        tokens_per_second = len(response_text.split()) / response_time if response_time > 0 else 0
        context_length = len(enhanced_context) if enhanced_context else 0
        
        print(f"⏱️  Response Time: {response_time:.2f}s")
        print(f"🚀 Tokens/Second: {tokens_per_second:.1f}")
        print(f"📊 Response Length: {len(response_text)} chars")
        print(f"📋 Context Length: {context_length} chars")
        print(f"🎯 Confidence: {result.get('confidence', 0.0):.2f}")
        
        # Log performance stats
        response_text = result.get('response', '')
        log_performance_stats(question, response_time, response_text, len(context))

        # Add question to response for frontend
        result['question'] = question
        result['expert'] = expert

        return jsonify(result)

    except Exception as e:
        response_time = time.time() - start_time
        print(f"\n❌ API ERROR after {response_time:.3f}s")
        print(f"Error: {e}")
        traceback.print_exc()
        
        # Log error stats
        log_performance_stats(question, response_time, f"ERROR: {str(e)}", len(context))
        
        return jsonify({
            'error': str(e),
            'response': 'Sorry, there was an error processing your request.',
            'confidence': 0.0
        })


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    # Also query underlying inference status to avoid stale flag
    try:
        inf_loaded = getattr(chess_model._inference, 'is_loaded', False)
    except Exception:
        inf_loaded = False
    model_status = "loaded" if (chess_model.is_loaded or inf_loaded) else "not_loaded"

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
            mv = extract_first_legal_move(text, board)
            return mv.uci() if mv else None

        # Engine mode
        eng = inf.generate_response(
            f"FEN: {fen}\nMove:\nMode: Engine\nGenerate the best move in UCI format (e.g., e2e4). Respond with only the move.",
            mode='engine', max_new_tokens=12
        )
        eng_move = parse_uci(eng.get('response', ''))

        # Tutor mode
        tut = inf.generate_response(
            f"FEN: {fen}\nQuestion: Analyze step-by-step and end with a single UCI move line.\nMode: Tutor",
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
    try:
        inf = chess_model._inference
        loaded = inf.is_loaded
        model_info = inf.get_model_info() if loaded else {}
    except Exception:
        loaded = False
        model_info = {}
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
        ],
        'loaded': loaded,
        'device': model_info.get('device') if loaded else None,
        'active_adapter': model_info.get('active_adapter') if loaded else None,
        'available_adapters': model_info.get('available_adapters') if loaded else {},
    }

    return jsonify(info)


@app.route('/api/stats', methods=['GET'])
def get_performance_stats():
    """Get current performance statistics."""
    with stats_lock:
        return jsonify(performance_stats)


@app.route('/api/game/state', methods=['GET'])
def get_game_state():
    """Get current game state."""
    return jsonify(chess_game.get_game_summary())


@app.route('/api/game/move', methods=['POST'])
def make_move():
    """Make a move in the chess game."""
    try:
        data = request.get_json()
        move_uci = data.get('move', '').strip()
        
        if not move_uci:
            return jsonify({'error': 'No move provided'}), 400
        
        result = chess_game.make_move(move_uci)
        
        # Log the move
        print(f"\n🎯 CHESS MOVE: {move_uci}")
        print(f"Success: {result['success']}")
        if result['success']:
            print(f"Game State: {result['game_state']}")
            print(f"Current Player: {result['current_player']}")
        else:
            print(f"Error: {result.get('error', 'Unknown error')}")
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Move error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/game/analyze', methods=['POST'])
def analyze_position():
    """Analyze a specific square or position."""
    try:
        data = request.get_json()
        square = data.get('square', '').strip()
        
        if not square:
            return jsonify({'error': 'No square provided'}), 400
        
        analysis = chess_game.get_position_analysis(square)
        
        # Get RAG knowledge for this position
        fen = chess_game.get_fen()
        rag_advice = chess_rag.get_position_specific_advice(fen, square)
        analysis['rag_advice'] = rag_advice
        
        print(f"\n🔍 POSITION ANALYSIS: {square}")
        print(f"Piece: {analysis['piece_name']}")
        print(f"Legal Moves: {analysis['legal_moves']}")
        print(f"RAG Advice: {rag_advice}")
        
        return jsonify(analysis)
        
    except Exception as e:
        print(f"Analysis error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/match/test', methods=['GET'])
def test_stockfish():
    """Test if Stockfish is available and working."""
    try:
        print(f"\n🔍 TESTING STOCKFISH AVAILABILITY")
        
        # Try to find Stockfish
        match = StockfishMatch()
        print(f"📍 Stockfish path: {match.stockfish_path}")
        
        # Try to start engine
        if match.start_engine():
            # Test a simple move
            test_board = chess.Board()
            result = match.engine.play(test_board, chess.engine.Limit(time=1.0))
            match.stop_engine()
            
            print(f"✅ Stockfish test successful - played: {result.move}")
            return jsonify({
                'success': True,
                'message': 'Stockfish is working correctly',
                'path': match.stockfish_path,
                'test_move': str(result.move)
            })
        else:
            return jsonify({'error': 'Failed to start Stockfish engine'}), 500
            
    except Exception as e:
        print(f"Stockfish test error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/match/start', methods=['POST'])
def start_stockfish_match():
    """Start a Stockfish vs Model match."""
    try:
        global stockfish_match
        
        data = request.get_json() or {}
        model_plays_white = data.get('model_plays_white', True)
        time_control = data.get('time_control', '10+0.1')  # 10 seconds + 0.1s increment
        
        print(f"\n🎮 STARTING STOCKFISH MATCH")
        print(f"📋 Model plays: {'White' if model_plays_white else 'Black'}")
        print(f"⏰ Time control: {time_control}")
        
        # Initialize match
        stockfish_match = StockfishMatch(time_control=time_control)
        
        if not stockfish_match.start_engine():
            return jsonify({'error': 'Failed to start Stockfish engine'}), 500
        
        return jsonify({
            'success': True,
            'message': f'Match started - Model plays {"White" if model_plays_white else "Black"}',
            'time_control': time_control,
            'model_plays_white': model_plays_white,
            'stockfish_path': stockfish_match.stockfish_path
        })
        
    except Exception as e:
        print(f"Match start error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/match/play', methods=['POST'])
def play_match_move():
    """Play one move in the Stockfish match."""
    try:
        global stockfish_match
        
        if not stockfish_match:
            return jsonify({'error': 'No active match'}), 400
        
        data = request.get_json() or {}
        model_plays_white = data.get('model_plays_white', True)
        
        legal_moves = [move.uci() for move in stockfish_match.board.legal_moves]
        
        if not legal_moves:
            return jsonify({'error': 'No legal moves available'}), 400
        
        is_model_turn = (stockfish_match.board.turn == chess.WHITE) == model_plays_white
        
        if is_model_turn:
            # Model's turn - no time limit
            def model_generator(question, context, mode="engine"):
                # Ensure UCI adapter for engine mode; tutor/director otherwise
                try:
                    if mode == 'engine':
                        chess_model._inference.set_active_adapter('uci')
                    elif mode == 'tutor':
                        chess_model._inference.set_active_adapter('tutor')
                    elif mode == 'director':
                        chess_model._inference.set_active_adapter('director')
                except Exception:
                    pass
                return chess_model.generate_response(question, context, mode)
            
            move_result = stockfish_match.get_model_move(model_generator, legal_moves, chess_rag)
            player = "Model"
        else:
            # Stockfish's turn
            move_result = stockfish_match.get_stockfish_move()
            player = "Stockfish"
        
        payload = {
            'success': True,
            'move': move_result.move,
            'san': move_result.san,
            'fen': move_result.fen,
            'player': player,
            'time_taken': move_result.time_taken,
            'evaluation': move_result.evaluation,
            'depth': move_result.depth,
            'is_game_over': stockfish_match.board.is_game_over(),
            'game_result': stockfish_match._determine_result() if stockfish_match.board.is_game_over() else None
        }
        if payload['is_game_over']:
            print("\n🏁 GAME OVER DETECTED")
            print(f"Winner/Reason: {payload['game_result']}")
        return jsonify(payload)
        
    except Exception as e:
        print(f"Match move error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/match/status', methods=['GET'])
def get_match_status():
    """Get current match status."""
    try:
        global stockfish_match
        
        if not stockfish_match:
            return jsonify({'active': False})
        
        return jsonify({
            'active': True,
            'fen': stockfish_match.board.fen(),
            'turn': 'white' if stockfish_match.board.turn == chess.WHITE else 'black',
            'is_game_over': stockfish_match.board.is_game_over(),
            'move_count': len(stockfish_match.moves),
            'legal_moves': [move.uci() for move in stockfish_match.board.legal_moves]
        })
        
    except Exception as e:
        print(f"Match status error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/train/start', methods=['POST'])
def api_train_start():
    try:
        data = request.get_json() or {}
        expert = (data.get('expert') or 'uci').strip().lower()
        steps = int(data.get('steps') or 1000)
        use_instruction = bool(data.get('use_instruction') or (expert in ('tutor', 'director')))
        disable_eval = bool(data.get('disable_eval') or True)
        if expert not in ('uci', 'tutor', 'director'):
            return jsonify({'error': 'Invalid expert'}), 400
        ok = TRAINING_JOB.start(expert, steps, use_instruction, disable_eval)
        if not ok:
            return jsonify({'error': 'Training already running or failed to start'}), 409
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/train/status', methods=['GET'])
def api_train_status():
    try:
        return jsonify(TRAINING_JOB.status())
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/train/stop', methods=['POST'])
def api_train_stop():
    try:
        ok = TRAINING_JOB.stop()
        return jsonify({'success': bool(ok)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/match/stop', methods=['POST'])
def stop_match():
    """Stop the current match."""
    try:
        global stockfish_match
        
        if stockfish_match:
            stockfish_match.stop_engine()
            stockfish_match = None
            print("🛑 Match stopped")
        
        return jsonify({'success': True, 'message': 'Match stopped'})
        
    except Exception as e:
        print(f"Match stop error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/game/reset', methods=['POST'])
def reset_game():
    """Reset the chess game to starting position."""
    try:
        chess_game.reset_game()
        print("\n🔄 GAME RESET")
        return jsonify({'success': True, 'message': 'Game reset to starting position'})
        
    except Exception as e:
        print(f"Reset error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/game/ai_move', methods=['POST'])
def get_ai_move():
    """Get AI's recommended move for the current position."""
    try:
        fen = chess_game.get_fen()
        current_player = chess_game.current_player
        legal_moves = chess_game.get_legal_moves()
        data = request.get_json() or {}
        expert = (data.get('expert') or 'tutor').strip().lower()
        
        start_time = time.time()
        print(f"\n🤖 AI MOVE REQUEST")
        print(f"FEN: {fen}")
        print(f"Player: {current_player}")
        print(f"Legal moves: {legal_moves}")
        
        if not legal_moves:
            return jsonify({
                'success': False,
                'error': 'No legal moves available',
                'game_state': chess_game.game_state
            })
        
        # Guided Play pipeline
        # 1) Use UCI expert to pick a precise move
        # 2) If Tutor selected, generate a concise explanation after making the move
        engine_question = (
            f"FEN: {fen}\n"
            "Move:\n"
            "Mode: Engine\n"
            "Generate the best move in UCI format (e.g., e2e4). Respond with only the move."
        )
        rag_knowledge = chess_rag.get_relevant_knowledge(engine_question, fen)
        rag_context = f"Chess Knowledge: {rag_knowledge}\n\n" if rag_knowledge else ""
        result_engine = chess_model.generate_response(
            engine_question,
            context=f"Current position: {fen}",
            mode='engine',
            max_length=16
        )
        
        # Try to extract a move from the engine response
        response_text = result_engine.get('response', '')
        try:
            b = chess.Board(fen)
            strict_mv = extract_first_legal_move_uci(response_text, b)
        except Exception:
            strict_mv = None
        move_uci = strict_mv or extract_move_from_response(response_text, legal_moves)
        
        print(f"Engine text: {response_text[:200]}...")
        print(f"Extracted move: {move_uci}")
        
        # Log performance metrics for AI move
        response_time = time.time() - start_time
        tokens_per_second = len((response_text or '').split()) / response_time if response_time > 0 else 0
        print(f"⏱️  AI Response Time: {response_time:.2f}s")
        print(f"🚀 AI Tokens/Second: {tokens_per_second:.1f}")
        print(f"📊 AI Response Length: {len(response_text or '')} chars")
        
        if move_uci and move_uci in legal_moves:
            # Make the AI move
            move_result = chess_game.make_move(move_uci)
            if expert == 'uci':
                # Engine-only path, return raw engine text
                move_result['ai_response'] = response_text
                move_result['ai_confidence'] = result_engine.get('confidence', 0.0)
            else:
                # Tutor explanation (concise)
                pre_fen = fen
                post_fen = move_result.get('fen', chess_game.get_fen())
                move_san = move_result.get('san', move_uci)
                tutor_question = (
                    f"FEN before: {pre_fen}\n"
                    f"FEN after: {post_fen}\n"
                    f"We played {move_san} ({move_uci}).\n\n"
                    "In 3 short bullets, explain: \n"
                    "- Why this move is good now (threats/ideas)\n"
                    "- Opponent's best reply and our follow-up\n"
                    "- One practical tip for the user in this position\n\n"
                    "Keep it under 120 words."
                )
                result_tutor = chess_model.generate_response(
                    tutor_question,
                    context=f"Current position: {post_fen}",
                    mode='tutor',
                    max_length=180
                )
                tutor_text = result_tutor.get('response', '')
                move_result['ai_response'] = tutor_text
                move_result['ai_confidence'] = result_tutor.get('confidence', 0.0)

            print(f"AI Move: {move_uci}")
            print(f"Success: {move_result['success']}")

            return jsonify(move_result)
        else:
            # Fallback: use ChessEngineManager to find a legal move
            fallback_move = None
            try:
                with ChessEngineManager() as ce:
                    board = chess.Board(fen)
                    engine_move = ce.get_best_move(board, depth=12, time_limit_ms=5000)
                    if engine_move:
                        fallback_move = engine_move.uci()
            except Exception as e:
                print(f"Engine fallback error: {e}")

            if not fallback_move:
                import random
                fallback_move = random.choice(legal_moves)
                fallback_text = f"AI chose: {fallback_move} (random fallback)"
            else:
                fallback_text = f"AI chose: {fallback_move} (engine fallback)"

            print(f"Using fallback move: {fallback_move}")

            move_result = chess_game.make_move(fallback_move)
            move_result['ai_response'] = fallback_text
            move_result['ai_confidence'] = 0.5

            return jsonify(move_result)
            
    except Exception as e:
        print(f"AI move error: {e}")
        return jsonify({'error': str(e)}), 500


def extract_move_from_response(response: str, legal_moves: List[str]) -> Optional[str]:
    """Extract a legal move from AI response text."""
    import re
    
    print(f"Extracting move from: {response[:200]}...")
    print(f"Legal moves: {legal_moves}")
    
    # Look for UCI format moves (e.g., e2e4, g1f3)
    uci_pattern = r'\b([a-h][1-8][a-h][1-8][qrbn]?)\b'
    matches = re.findall(uci_pattern, response.lower())
    
    print(f"Found UCI matches: {matches}")
    
    for match in matches:
        if match in legal_moves:
            print(f"✅ Found legal UCI move: {match}")
            return match
    
    # Look for partial matches (e.g., if AI says "e2e4" but we have "e2e4" in legal moves)
    for move in legal_moves:
        if move.lower() in response.lower():
            print(f"✅ Found partial UCI match: {move}")
            return move
    
    # Look for SAN format moves and try to convert (simplified)
    san_pattern = r'\b([KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[+#]?)\b'
    san_matches = re.findall(san_pattern, response)
    
    print(f"Found SAN matches: {san_matches}")
    
    # Try to find moves mentioned in explanations
    explanation_pattern = r'(?:move|play|choose|select).*?([a-h][1-8][a-h][1-8])'
    explanation_matches = re.findall(explanation_pattern, response.lower())
    print(f"Found explanation matches: {explanation_matches}")
    
    # Check explanation matches
    for move in explanation_matches:
        if move in legal_moves:
            print(f"✅ Found move in explanation: {move}")
            return move
    
    print("❌ No valid move found in response")
    return None


def find_free_port(start_port=5000, max_attempts=10):
    """Find a free port starting from start_port."""
    import socket
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"Could not find a free port in range {start_port}-{start_port + max_attempts - 1}")


if __name__ == '__main__':
    print("🚀 Starting ChessGemma Web Interface...")
    print("="*60)
    
    # Show initial system stats
    initial_stats = get_system_stats()
    print(f"💻 Initial System Stats:")
    print(f"   Memory: {initial_stats['memory_mb']:.1f} MB")
    print(f"   CPU: {initial_stats['cpu_percent']:.1f}%")
    print(f"   Time: {initial_stats['timestamp']}")
    print("="*60)

    # Try to preload the model
    print("🔄 Preloading model...")
    model_start_time = time.time()
    preload_success = chess_model.load_model()
    model_load_time = time.time() - model_start_time
    
    if preload_success:
        print(f"✅ Model preloaded successfully in {model_load_time:.3f}s")
        
        # Show model info
        model_info = chess_model._inference.get_model_info()
        print(f"📊 Model Info:")
        print(f"   Device: {model_info.get('device', 'unknown')}")
        print(f"   Base Model: {model_info.get('base_model', 'unknown')}")
        print(f"   Adapter: {model_info.get('adapter_path', 'none')}")
        print(f"   Loaded: {model_info.get('is_loaded', False)}")
        print(f"   MPS Available: {torch.backends.mps.is_available()}")
        print(f"   MPS Built: {torch.backends.mps.is_built()}")
        if hasattr(chess_model._inference.model, 'device'):
            print(f"   Model Device: {chess_model._inference.model.device}")
    else:
        print(f"⚠️  Model preloading failed after {model_load_time:.3f}s - will load on first request")

    # Find an available port
    try:
        port = find_free_port()
        print("="*60)
        print(f"🌐 Web Interface Ready!")
        print(f"📍 URL: http://localhost:{port}")
        print(f"📊 Performance Stats: http://localhost:{port}/api/stats")
        print(f"🔍 Health Check: http://localhost:{port}/api/health")
        print("="*60)
        print("🎯 Ready to accept chess questions!")
        print("="*60)
        
        app.run(
            host='0.0.0.0',
            port=port,
            debug=False,  # Set to False for production
            threaded=True
        )
    except RuntimeError as e:
        print(f"❌ Error: {e}")
        print("Please free up some ports or try again later.")
