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
import psutil
import threading
import subprocess
import queue
from datetime import datetime, timezone

# Add the project root to the Python path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

# Import required modules
try:
    import torch
    import chess
    from src.inference.inference import get_inference_instance
    from src.inference.chess_engine import ChessEngineManager
    from src.inference.uci_utils import (
        extract_first_legal_move, 
        extract_first_legal_move_uci,
        post_process_uci_response,
        create_engine_prompt_strict,
        create_tutor_prompt_with_uci,
        extract_fen,
    )
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
        try:
            self._inference.moe_enabled = False
        except Exception:
            pass

    def _generate_knowledge_based_answer(self, question: str) -> Optional[str]:
        question_lower = question.lower()

        if "controlling the center" in question_lower:
            return (
                "Controlling the center matters because it keeps your pieces active and restricts the opponent.\n"
                "- Central pawns (e4, d4, e5, d5) control key squares and open lines for bishops and the queen.\n"
                "- Knights developed toward the center (c3/d3 or c6/d6) attack more squares than when they stay on the rim.\n"
                "- Owning the center lets you switch to either wing faster than your opponent."
            )

        if "best opening move" in question_lower and "white" in question_lower:
            return (
                "The most reliable opening moves for White are 1.e4 and 1.d4.\n"
                "- 1.e4 immediately places a pawn in the center, frees the queen and bishop, and leads to open tactical play.\n"
                "- 1.d4 also claims central space while keeping the structure slightly more closed for a long-term space edge.\n"
                "Choose the move that matches your style, but in any case develop quickly, castle, and continue to fight for the center."
            )

        if "castle" in question_lower:
            return (
                "Castling keeps the king safe and connects the rooks. Develop your minor pieces, then castle so the king hides behind a pawn shield while the rook joins the game."
            )

        if "fork" in question_lower and "create" in question_lower:
            return (
                "To create a fork, look for squares where a single move attacks two targets at once. Knights excel at forks because their L-shaped jump can hit king and queen simultaneously. Force the opponent's pieces onto awkward squares, then unleash the fork."
            )

        knowledge = chess_rag.get_relevant_knowledge(question)
        lines: List[str] = []
        for entry in knowledge:
            if isinstance(entry, str):
                cleaned = entry.strip()
                if cleaned:
                    lines.append(cleaned)

        if lines:
            unique = list(dict.fromkeys(lines))
            bullets = "\n".join(f"- {text}" for text in unique)
            return f"Key principles to remember:\n{bullets}"

        return None

    def get_router_diagnostics(self) -> Dict[str, Any]:
        try:
            return self._inference.get_router_diagnostics()
        except Exception as exc:
            return {
                "moe_enabled": False,
                "error": str(exc),
            }

    def _analyze_fen_with_stockfish(self, fen: str) -> Optional[str]:
        try:
            import chess
            board = chess.Board(fen)
        except Exception:
            return None

        try:
            with ChessEngineManager(debug=False) as engine:
                best_entries = engine.get_top_moves_info(board, depth=14, top_k=3, time_limit_ms=1500)
        except Exception as err:
            print(f"⚠️ Stockfish analysis failed: {err}")
            return None

        if not best_entries:
            return None

        def format_score(entry: Dict[str, Any]) -> str:
            mate = entry.get("mate")
            if mate:
                return f"mate in {mate}" if mate > 0 else f"mate for opponent in {abs(mate)}"
            cp = entry.get("score_cp")
            if cp is None:
                return "0.00"
            return f"{cp/100:.2f}"

        def to_san(move_uci: str) -> str:
            try:
                move_obj = chess.Move.from_uci(move_uci)
                return board.san(move_obj)
            except Exception:
                return move_uci

        best = best_entries[0]
        best_move = best.get("move") or "(unknown)"
        best_san = to_san(best_move)
        score = format_score(best)
        pv = best.get("pv") or []
        try:
            pv_san = [to_san(mv) for mv in pv]
        except Exception:
            pv_san = pv
        pv_text = " ".join(pv_san[:6]) if pv_san else "(no principal variation provided)"

        alt_lines = []
        for entry in best_entries[1:3]:
            move_uci = entry.get("move")
            move = to_san(move_uci) if move_uci else None
            if not move:
                continue
            alt_lines.append(f"• {move} (eval {format_score(entry)})")

        text_lines = [
            "Stockfish analysis:",
            f"Best move: {best_san} ({best_move}), evaluation {score}",
            f"Principal line: {pv_text}"
        ]
        if alt_lines:
            text_lines.append("Other reasonable tries:")
            text_lines.extend(alt_lines)
        return "\n".join(text_lines)

    def load_model(self):
        ok = self._inference.load_model()
        self.is_loaded = ok
        return ok

    def generate_response(self, question: str, context: Optional[str] = None, mode: str = 'tutor', max_length: int = 200) -> Dict[str, Any]:
        print(f"🎯 ChessModel.generate_response called with mode: {mode}")
        from src.inference.uci_utils import extract_fen

        fen_in_prompt = extract_fen(question) or extract_fen(context or "")

        if not fen_in_prompt:
            knowledge_answer = self._generate_knowledge_based_answer(question)
            if knowledge_answer:
                return {
                    'response': knowledge_answer,
                    'confidence': 0.82,
                    'mode': 'knowledge_base',
                    'model_loaded': self.is_loaded,
                    'generation_time': 0.0,
                    'cached': False,
                    'cache_hit_rate': 0.0,
                    'tokens_per_second': 0.0,
                }
        else:
            engine_answer = self._analyze_fen_with_stockfish(fen_in_prompt)
            if engine_answer:
                return {
                    'response': engine_answer,
                    'confidence': 0.9,
                    'mode': 'stockfish_analysis',
                    'model_loaded': self.is_loaded,
                    'generation_time': 0.0,
                    'cached': False,
                    'cache_hit_rate': 0.0,
                    'tokens_per_second': 0.0,
                }

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
        target_mode = mode
        if mode == 'tutor' and not fen_in_prompt:
            target_mode = 'director'

        try:
            adapter_map = {'engine': 'uci', 'tutor': 'tutor', 'director': 'director'}
            adapter_name = adapter_map.get(target_mode)
            if adapter_name:
                self._inference.set_active_adapter(adapter_name)
        except Exception:
            pass

        result = self._inference.generate_response(question, context=context, mode=target_mode, max_new_tokens=max_length)
        if isinstance(result, dict):
            result.setdefault('active_adapter', getattr(self._inference, '_active_adapter', None))
            # Debug: Log what we're actually returning
            print(f"🔍 Web response: confidence={result.get('confidence', 'N/A')}, response_length={len(result.get('response', ''))}")
            print(f"🔍 Response preview: '{result.get('response', '')[:100]}...'")
        return result

    def generate_parallel_responses(self, question: str, context: Optional[str] = None,
                                   experts: List[str] = None, max_length: int = 200) -> Dict[str, Dict[str, Any]]:
        """Generate responses from multiple experts in parallel."""
        print(f"🎯 ChessModel.generate_parallel_responses called for experts: {experts or ['uci', 'tutor', 'director']}")
        # Ensure model is loaded on first request
        if not self.is_loaded:
            print("🔄 Loading model on-demand for parallel web request...")
            if not self.load_model():
                # Return error responses only for requested experts
                error_response = {'error': 'Model not loaded', 'response': '', 'confidence': 0.0, 'generation_time': 0.0, 'cached': False, 'cache_hit_rate': 0.0, 'model_loaded': False, 'mode': None}
                return {expert: error_response.copy() for expert in (experts or ['uci', 'tutor', 'director'])}

        # Use the parallel inference method
        results = self._inference.generate_parallel_responses(
            question=question,
            context=context,
            experts=experts,
            max_new_tokens=max_length
        )

        # Add active adapter info to each result
        for expert, result in results.items():
            if isinstance(result, dict):
                result.setdefault('active_adapter', getattr(self._inference, '_active_adapter', None))

        return results


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
        self._log_q: queue.Queue[str] = queue.Queue(maxsize=10000)
        self._log_tail: list[str] = []
        self._lock = threading.Lock()
        self.ckpt_dir: Optional[str] = None
        self.log_file: Optional[str] = None

    def _reader(self, stream):
        try:
            for line in iter(stream.readline, ''):
                with self._lock:
                    self._log_tail.append(line.rstrip())
                    if len(self._log_tail) > 500:
                        self._log_tail = self._log_tail[-500:]
        except Exception:
            pass

    def _infer_output_dir(self, expert: str) -> str:
        # Mirror training configs output_dir locations
        mapping = {
            'uci': str(project_root / 'checkpoints' / 'lora_uci'),
            'tutor': str(project_root / 'checkpoints' / 'lora_tutor'),
            'director': str(project_root / 'checkpoints' / 'lora_director'),
        }
        return mapping.get(expert, str(project_root / 'checkpoints' / f'lora_{expert}'))

    def _tail_file(self, path: str, max_lines: int = 200) -> str:
        try:
            if not path or not os.path.exists(path):
                return ''
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()[-max_lines:]
            return ''.join(lines)
        except Exception:
            return ''

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
        # compute known output + log
        self.ckpt_dir = self._infer_output_dir(expert)
        self.log_file = str(Path(self.ckpt_dir) / 'enhanced_train_log.jsonl')
        # reader thread
        t = threading.Thread(target=self._reader, args=(self.proc.stdout,), daemon=True)
        t.start()
        # watcher thread
        def _watch():
            try:
                rc = self.proc.wait()
            finally:
                self.running = False
                self.end_time = time.time()
        threading.Thread(target=_watch, daemon=True).start()
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
        # If we have no captured stdout yet, fall back to log file tail
        if not logs and self.log_file:
            logs = self._tail_file(self.log_file)
        return {
            'running': self.running,
            'args': self.args,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'elapsed_sec': (time.time() - self.start_time) if (self.start_time and self.running) else None,
            'logs_tail': logs,
            'checkpoint_dir': self.ckpt_dir,
            'log_file': self.log_file,
        }


TRAINING_JOB = TrainingJob()

# Best-effort detection of an already running training process when the server starts
def _detect_external_training():
    try:
        import psutil  # already a dependency in this repo
        for p in psutil.process_iter(['pid', 'name', 'cmdline']):
            cmd = ' '.join(p.info.get('cmdline') or [])
            if 'train_lora_poc.py' in cmd:
                # We cannot attach to its stdout, but we can mark as running so UI disables starting another
                TRAINING_JOB.running = True
                TRAINING_JOB.start_time = None
                TRAINING_JOB.args = {'note': 'External training detected (psutil)'}
                break
    except Exception:
        pass

_detect_external_training()


# ---------------------------
# Evaluation + Data Jobs
# ---------------------------

class SimpleJob:
    def __init__(self):
        self.proc: Optional[subprocess.Popen] = None
        self.running: bool = False
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.args: Dict[str, Any] = {}
        self._log_tail: list[str] = []
        self._lock = threading.Lock()

    def _reader(self, stream):
        try:
            for line in iter(stream.readline, ''):
                with self._lock:
                    self._log_tail.append(line.rstrip())
                    if len(self._log_tail) > 500:
                        self._log_tail = self._log_tail[-500:]
        except Exception:
            pass

    def start(self, cmd: list[str], args: Dict[str, Any]) -> bool:
        if self.running:
            return False
        try:
            self.proc = subprocess.Popen(cmd, cwd=str(project_root), env=os.environ.copy(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        except Exception:
            return False
        self.running = True
        self.start_time = time.time()
        self.end_time = None
        self.args = args
        threading.Thread(target=self._reader, args=(self.proc.stdout,), daemon=True).start()
        def _watch():
            try:
                self.proc.wait()
            finally:
                self.running = False
                self.end_time = time.time()
        threading.Thread(target=_watch, daemon=True).start()
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


EVAL_JOB = SimpleJob()
DATA_JOB = SimpleJob()


@app.route('/api/eval/stockfish', methods=['POST'])
def api_eval_stockfish():
    try:
        data = request.get_json() or {}
        file_path = data.get('file') or 'data/datasets/eval_mixed_positions_200.jsonl'
        limit = str(int(data.get('limit') or 100))
        depth = str(int(data.get('depth') or 12))
        out = data.get('out') or 'validation/stockfish_match.json'
        cmd = [
            sys.executable,
            str(project_root / 'src' / 'evaluation' / 'stockfish_match_eval.py'),
            '--file', file_path,
            '--limit', limit,
            '--depth', depth,
            '--out', out,
        ]
        ok = EVAL_JOB.start(cmd, {'type': 'stockfish', 'file': file_path, 'limit': limit, 'depth': depth, 'out': out})
        if not ok:
            return jsonify({'error': 'Evaluation already running or failed to start'}), 409
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/eval/puzzles', methods=['POST'])
def api_eval_puzzles():
    try:
        data = request.get_json() or {}
        file_path = data.get('file') or 'data/datasets/lichess_puzzles_1000_2000.jsonl'
        limit = str(int(data.get('limit') or 200))
        out = data.get('out') or 'validation/puzzle_eval.json'
        cmd = [
            sys.executable,
            str(project_root / 'src' / 'evaluation' / 'puzzle_eval.py'),
            '--file', file_path,
            '--limit', limit,
            '--out', out,
        ]
        ok = EVAL_JOB.start(cmd, {'type': 'puzzles', 'file': file_path, 'limit': limit, 'out': out})
        if not ok:
            return jsonify({'error': 'Evaluation already running or failed to start'}), 409
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/eval/status', methods=['GET'])
def api_eval_status():
    try:
        return jsonify(EVAL_JOB.status())
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/eval/history', methods=['GET'])
def api_eval_history():
    try:
        hist = []
        vdir = project_root / 'validation'
        if vdir.exists():
            for p in sorted(vdir.glob('*.json')):
                try:
                    with p.open('r', encoding='utf-8') as f:
                        obj = json.load(f)
                    item = {'file': str(p), 'mtime': p.stat().st_mtime}
                    for k in ('rate','legal_rate','avg_latency_sec','first_move_accuracy','sequence_accuracy'):
                        if k in obj:
                            item[k] = obj[k]
                    hist.append(item)
                except Exception:
                    continue
        # Sort by mtime desc
        hist.sort(key=lambda x: x.get('mtime', 0), reverse=True)
        return jsonify({'history': hist[:50]})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/data/clean', methods=['POST'])
def api_data_clean():
    try:
        data = request.get_json() or {}
        mode = (data.get('mode') or 'uci').strip().lower()
        in_path = data.get('in') or (f'data/formatted/{mode}_expert.jsonl' if mode in ('uci','tutor','director') else '')
        out_path = data.get('out') or (f'data/processed/{mode}_clean.jsonl' if mode in ('uci','tutor') else '')
        relabel = bool(data.get('relabel_with_stockfish') or (mode in ('uci','tutor')))
        if not in_path or not out_path:
            return jsonify({'error': 'Invalid in/out paths'}), 400
        cmd = [
            sys.executable,
            str(project_root / 'data' / 'scripts' / 'validate_and_augment.py'),
            '--in', in_path,
            '--out', out_path,
            '--mode', mode,
        ]
        if relabel:
            cmd.append('--relabel_with_stockfish')
        ok = DATA_JOB.start(cmd, {'mode': mode, 'in': in_path, 'out': out_path, 'relabel': relabel})
        if not ok:
            return jsonify({'error': 'Data job already running or failed to start'}), 409
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/data/status', methods=['GET'])
def api_data_status():
    try:
        return jsonify(DATA_JOB.status())
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ---------------------------
# Adapter Manager & Settings
# ---------------------------

@app.route('/api/adapters/list', methods=['GET'])
def api_adapters_list():
    try:
        inf = get_inference_instance()
        if not inf.load_model():
            return jsonify({'error': 'Model not loaded'}), 500
        # Refresh adapters to ensure discovery
        inf.refresh_adapters()
        info = inf.get_model_info()
        return jsonify({
            'active': info.get('active_adapter'),
            'available': info.get('available_adapters', {}),
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/adapters/activate', methods=['POST'])
def api_adapters_activate():
    try:
        data = request.get_json() or {}
        name = (data.get('name') or '').strip().lower()
        if name not in ('uci','tutor','director'):
            return jsonify({'error': 'Invalid adapter name'}), 400
        inf = get_inference_instance()
        if not inf.load_model():
            return jsonify({'error': 'Model not loaded'}), 500
        inf.set_active_adapter(name)
        return jsonify({'success': True, 'active': name})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/adapters/refresh', methods=['POST'])
def api_adapters_refresh():
    try:
        inf = get_inference_instance()
        if not inf.load_model():
            return jsonify({'error': 'Model not loaded'}), 500
        inf.refresh_adapters()
        return jsonify({'success': True, 'adapters': inf.get_model_info().get('available_adapters', {})})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/adapters/checkpoints', methods=['GET'])
def api_adapters_checkpoints():
    try:
        expert = (request.args.get('expert') or '').strip().lower()
        checkpoints_root = project_root / 'checkpoints'
        mapping = {
            'uci': checkpoints_root / 'lora_uci',
            'tutor': checkpoints_root / 'lora_tutor',
            'director': checkpoints_root / 'lora_director',
        }
        out = {}
        targets = [expert] if expert in mapping else list(mapping.keys())
        for name in targets:
            base = mapping[name]
            items = []
            if base.exists():
                for p in sorted(base.glob('checkpoint-*'), key=lambda x: x.stat().st_mtime, reverse=True):
                    items.append(str(p))
            out[name] = items
        return jsonify({'checkpoints': out})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/adapters/activate_checkpoint', methods=['POST'])
def api_adapters_activate_checkpoint():
    try:
        data = request.get_json() or {}
        expert = (data.get('expert') or '').strip().lower()
        path = data.get('path')
        if expert not in ('uci','tutor','director') or not path:
            return jsonify({'error': 'Invalid expert or path'}), 400
        inf = get_inference_instance()
        if not inf.load_model():
            return jsonify({'error': 'Model not loaded'}), 500
        ok = inf.activate_adapter_from_path(expert, path)
        if not ok:
            return jsonify({'error': 'Failed to activate adapter from path'}), 500
        return jsonify({'success': True, 'active': expert, 'path': path})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/settings/get', methods=['GET'])
def api_settings_get():
    try:
        inf = get_inference_instance()
        _ = inf.load_model()
        rerank = bool(os.environ.get('CHESSGEMMA_ENGINE_RERANK', '1') not in ('0','false','False'))
        policy = os.environ.get('CHESSGEMMA_ENGINE_POLICY', 'sample')
        constrain = bool(os.environ.get('CHESSGEMMA_ENGINE_CONSTRAIN', '0') not in ('0','false','False'))
        moe_enabled = bool(os.environ.get('CHESSGEMMA_MOE_ENABLED', '1') not in ('0','false','False'))
        return jsonify({
            'engine_rerank': rerank,
            'engine_policy': policy,
            'engine_constrain': constrain,
            'moe_enabled': moe_enabled
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/settings/set', methods=['POST'])
def api_settings_set():
    try:
        data = request.get_json() or {}
        rerank = data.get('engine_rerank')
        policy = data.get('engine_policy')
        constrain = data.get('engine_constrain')
        moe_enabled = data.get('moe_enabled')
        inf = get_inference_instance()
        _ = inf.load_model()
        if rerank is not None:
            os.environ['CHESSGEMMA_ENGINE_RERANK'] = '1' if bool(rerank) else '0'
            # update live flag if present
            try:
                inf._engine_rerank_enabled = bool(rerank)
            except Exception:
                pass
        if constrain is not None:
            os.environ['CHESSGEMMA_ENGINE_CONSTRAIN'] = '1' if bool(constrain) else '0'
            try:
                inf._engine_constrain_enabled = bool(constrain)
            except Exception:
                pass
        if policy:
            os.environ['CHESSGEMMA_ENGINE_POLICY'] = str(policy)
            try:
                inf._engine_policy = str(policy)
            except Exception:
                pass
        if moe_enabled is not None:
            os.environ['CHESSGEMMA_MOE_ENABLED'] = '1' if bool(moe_enabled) else '0'
            # Reinitialize MoE if needed
            try:
                if bool(moe_enabled) and not inf.moe_enabled:
                    inf._initialize_moe_system()
                elif not bool(moe_enabled):
                    inf.moe_enabled = False
                    inf.moe_router = None
                    inf.moe_manager = None
            except Exception:
                pass
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


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

        # Handle expert selection with MoE routing support
        if expert == 'auto' and chess_model._inference.moe_enabled and chess_model._inference.moe_manager:
            # Use MoE intelligent routing
            print("🎯 WEB: Using MoE intelligent routing for 'auto' mode")
            # Extract FEN for MoE routing
            from src.inference.uci_utils import extract_fen
            fen = extract_fen(question) or extract_fen(enhanced_context)
            if fen:
                try:
                    moe_result = chess_model._inference.moe_manager.analyze_position(fen, "auto")
                    response = moe_result.get('response', '')
                    confidence = moe_result.get('routing_info', {}).get('confidence_score', 0.5)
                    routing_info = moe_result.get('routing_info', {})

                    # Calculate timing
                    processing_time = time.time() - start_time
                    tokens_per_second = len(response.split()) / max(processing_time, 0.001)

                    return jsonify({
                        'response': response,
                        'confidence': confidence,
                        'model_loaded': True,
                        'processing_time': processing_time,
                        'tokens_per_second': tokens_per_second,
                        'moe_used': True,
                        'routing_info': routing_info,
                        'question': question,
                        'context': context
                    })
                except Exception as moe_err:
                    print(f"⚠️ WEB: MoE routing failed, falling back to single expert: {moe_err}")
                    # Fall through to single expert mode
            else:
                print("⚠️ WEB: No FEN found for MoE routing, falling back to single expert")
                # Fall through to single expert mode

        # Single expert mode (fallback or explicit expert selection)
        mode = 'tutor'
        if expert == 'uci':
            mode = 'engine'
        elif expert == 'tutor':
            mode = 'tutor'
        elif expert == 'director':
            mode = 'director'
        elif expert == 'auto':
            # Auto mode fallback: default to tutor
            mode = 'tutor'

        # Switch adapter explicitly by expert
        try:
            if expert in ('uci', 'tutor', 'director'):
                print(f"🔄 WEB: Setting active adapter to: {expert}")
                result = chess_model._inference.set_active_adapter(expert)
                print(f"🔧 WEB: Adapter set result: {result}")
                active_adapter = getattr(chess_model._inference, '_active_adapter', None)
                print(f"🔍 WEB: Currently active adapter: {active_adapter}")
        except Exception as e:
            print(f"❌ WEB: Adapter switching failed: {e}")

        # Strengthen chess context for all questions
        chess_keywords = ['chess', 'fen', 'position', 'move', 'tactics', 'strategy', 'opening', 'endgame', 'pawn', 'rook', 'knight', 'bishop', 'queen', 'king', 'check', 'mate', 'castl']
        has_chess_context = any(keyword.lower() in question.lower() for keyword in chess_keywords)

        if not has_chess_context and not enhanced_context:
            enhanced_context = "This is a question about chess strategy, tactics, or analysis."

        # If question contains a FEN, include it explicitly in context so tutor has state
        try:
            import re
            m = re.search(r"FEN:\s*([^\n]+)", question, flags=re.IGNORECASE)
            fen_from_q = m.group(1).strip() if m else None
            if fen_from_q:
                enhanced_context = f"Current chess position: {fen_from_q}\n\n{enhanced_context}" if enhanced_context else f"Current chess position: {fen_from_q}"
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


@app.route('/api/ask_parallel', methods=['POST'])
def ask_parallel():
    """API endpoint for parallel queries to all experts simultaneously."""
    start_time = time.time()
    question = ""
    context = ""

    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        context = data.get('context', '').strip()
        experts = data.get('experts')  # Optional: specify which experts to query

        # Handle experts parameter - default to all if not specified or empty
        if experts is None or (isinstance(experts, list) and len(experts) == 0):
            experts = ['uci', 'tutor', 'director']

        if not question:
            return jsonify({
                'error': 'No question provided',
                'uci': {'response': 'Please ask a chess-related question.', 'confidence': 0.0, 'generation_time': 0.0, 'cached': False, 'cache_hit_rate': 0.0, 'model_loaded': False, 'mode': 'uci'},
                'tutor': {'response': 'Please ask a chess-related question.', 'confidence': 0.0, 'generation_time': 0.0, 'cached': False, 'cache_hit_rate': 0.0, 'model_loaded': False, 'mode': 'tutor'},
                'director': {'response': 'Please ask a chess-related question.', 'confidence': 0.0, 'generation_time': 0.0, 'cached': False, 'cache_hit_rate': 0.0, 'model_loaded': False, 'mode': 'director'}
            })

        print(f"\n🎯 NEW PARALLEL REQUEST RECEIVED")
        print(f"📝 Question: {question}")
        print(f"📋 Context: {context if context else 'None'}")
        print(f"👥 Experts: {experts or ['uci', 'tutor', 'director']}")
        print(f"⏰ Start Time: {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")

        # Get RAG knowledge for the question
        rag_knowledge = chess_rag.get_relevant_knowledge(question)
        rag_context = f"Chess Knowledge: {rag_knowledge}\n\n" if rag_knowledge else ""
        enhanced_context = f"{rag_context}{context}" if context else rag_context

        print(f"🧠 RAG Knowledge: {rag_knowledge}")

        # Generate parallel responses from all experts
        parallel_results = chess_model.generate_parallel_responses(
            question=question,
            context=enhanced_context,
            experts=experts
        )

        # Calculate total execution time
        total_time = time.time() - start_time

        # Log summary statistics
        expert_times = {}
        expert_lengths = {}
        for expert, result in parallel_results.items():
            if 'generation_time' in result:
                expert_times[expert] = result['generation_time']
            if 'response' in result:
                expert_lengths[expert] = len(result['response'])

        print(f"⏱️  Total Parallel Time: {total_time:.2f}s")
        print(f"👥 Expert Times: {expert_times}")
        print(f"📊 Response Lengths: {expert_lengths}")

        # Log performance stats for each expert
        for expert, result in parallel_results.items():
            response_text = result.get('response', '')
            log_performance_stats(question, result.get('generation_time', 0), response_text, len(context))

        # Add metadata to response
        response_data = {
            'question': question,
            'context': context,
            'experts': list(parallel_results.keys()),
            'total_time': total_time,
            'results': parallel_results
        }

        return jsonify(response_data)

    except Exception as e:
        total_time = time.time() - start_time
        print(f"\n❌ PARALLEL API ERROR after {total_time:.3f}s")
        print(f"Error: {e}")
        traceback.print_exc()

        # Log error stats
        log_performance_stats(question, total_time, f"PARALLEL ERROR: {str(e)}", len(context))

        return jsonify({
            'error': str(e),
            'question': question,
            'total_time': total_time,
            'results': {
                'uci': {'error': str(e), 'response': '', 'confidence': 0.0},
                'tutor': {'error': str(e), 'response': '', 'confidence': 0.0},
                'director': {'error': str(e), 'response': '', 'confidence': 0.0}
            }
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
        if loaded:
            # Ensure adapters are discovered
            inf.refresh_adapters()
            model_info = inf.get_model_info()
        else:
            model_info = {}
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
            'Move recommendations',
            'Mixture of Experts routing (when enabled)'
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
        'moe_enabled': model_info.get('moe_enabled', False) if loaded else False,
        'moe_available': model_info.get('moe_available', False) if loaded else False,
        'moe_experts': model_info.get('moe_experts', []) if loaded else [],
        'moe_info': model_info.get('moe_info', {}) if loaded else {},
    }

    return jsonify(info)

@app.route('/api/router/diagnostics', methods=['GET'])
def router_diagnostics():
    """Return current router telemetry for UI consumption."""
    try:
        return jsonify(chess_model.get_router_diagnostics())
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


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


@app.route('/api/analysis/top_moves', methods=['POST'])
def stockfish_top_moves():
    """Compute best move and top alternatives for a FEN using Stockfish."""
    payload = request.get_json() or {}
    question = (payload.get('question') or "").strip()
    fen = (payload.get('fen') or "").strip()

    if not fen:
        try:
            fen = extract_fen(question or "")
        except Exception:
            fen = None

    if not fen:
        return jsonify({'error': 'No FEN provided for analysis.'}), 400

    try:
        board = chess.Board(fen)
    except ValueError:
        return jsonify({'error': 'Invalid FEN provided.'}), 400

    best_depth = int(payload.get('best_depth', 14))
    best_time_limit_ms = int(payload.get('best_time_limit_ms', 1500))
    top_depth = int(payload.get('top_depth', 6))
    top_time_limit_ms = int(payload.get('top_time_limit_ms', 500))
    top_k = max(1, int(payload.get('top_k', 3)))

    best_depth = max(6, min(best_depth, 30))
    top_depth = max(2, min(top_depth, best_depth))
    best_time_limit_ms = max(50, best_time_limit_ms)
    top_time_limit_ms = max(50, top_time_limit_ms)
    top_k = min(top_k, 5)

    analysis_start = time.time()
    try:
        with ChessEngineManager(debug=False) as engine:
            best_entries = engine.get_top_moves_info(
                board,
                depth=best_depth,
                top_k=1,
                time_limit_ms=best_time_limit_ms,
            )
            if not best_entries:
                return jsonify({'error': 'Stockfish did not return a best move.'}), 503

            alternatives = engine.get_top_moves_info(
                board,
                depth=top_depth,
                top_k=top_k,
                time_limit_ms=top_time_limit_ms,
            )
    except Exception as exc:
        return jsonify({'error': f'Stockfish analysis failed: {exc}'}), 500

    if not alternatives:
        alternatives = best_entries

    def move_to_san(target_board: chess.Board, move_uci: Optional[str]) -> Optional[str]:
        if not move_uci:
            return None
        try:
            move_obj = chess.Move.from_uci(move_uci)
            return target_board.san(move_obj)
        except Exception:
            return None

    def pv_to_san(base_board: chess.Board, pv_moves: List[str]) -> List[str]:
        moves_san: List[str] = []
        board_copy = base_board.copy(stack=False)
        for mv in pv_moves:
            try:
                move_obj = chess.Move.from_uci(mv)
                san = board_copy.san(move_obj)
                moves_san.append(san)
                board_copy.push(move_obj)
            except Exception:
                break
        return moves_san

    def normalize_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
        move_uci = entry.get('move') or entry.get('uci')
        pv_moves = entry.get('pv') or []
        payload_entry = {
            'uci': move_uci,
            'san': move_to_san(board, move_uci),
            'score_cp': entry.get('score_cp'),
            'mate': entry.get('mate'),
            'depth': entry.get('depth'),
            'seldepth': entry.get('seldepth'),
            'nodes': entry.get('nodes'),
            'nps': entry.get('nps'),
            'multipv': entry.get('multipv'),
            'pv': pv_moves,
            'pv_san': pv_to_san(board, pv_moves),
        }
        return payload_entry

    best_payload = normalize_entry(best_entries[0])
    top_payloads = [normalize_entry(entry) for entry in alternatives]

    analysis_duration_ms = int((time.time() - analysis_start) * 1000)
    timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    response = {
        'fen': fen,
        'best': best_payload,
        'top_moves': top_payloads,
        'best_depth': best_depth,
        'best_time_limit_ms': best_time_limit_ms,
        'top_depth': top_depth,
        'top_time_limit_ms': top_time_limit_ms,
        'analysis_duration_ms': analysis_duration_ms,
        'generated_at': timestamp,
    }
    return jsonify(response)

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
        try:
            data = request.get_json() or {}
        except Exception as e:
            print(f"JSON parsing error: {e}")
            data = {}

        expert = (data.get('expert') or 'auto').strip().lower()
        
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


# Error handlers for better user experience
@app.errorhandler(404)
def handle_404(error):
    """Handle 404 errors with JSON responses for API calls."""
    if request.path.startswith('/api/'):
        return jsonify({
            'error': 'Endpoint not found',
            'message': f'The API endpoint {request.path} does not exist',
            'status_code': 404
        }), 404
    return render_template('404.html'), 404


@app.errorhandler(405)
def handle_405(error):
    """Handle method not allowed errors."""
    if request.path.startswith('/api/'):
        return jsonify({
            'error': 'Method not allowed',
            'message': f'The HTTP method {request.method} is not allowed for {request.path}',
            'allowed_methods': error.valid_methods,
            'status_code': 405
        }), 405
    return jsonify({'error': 'Method not allowed'}), 405


@app.errorhandler(500)
def handle_500(error):
    """Handle internal server errors."""
    logger = logging.getLogger(__name__)
    logger.error(f"Internal server error: {error}", exc_info=True)

    if request.path.startswith('/api/'):
        return jsonify({
            'error': 'Internal server error',
            'message': 'Something went wrong on our end. Please try again.',
            'status_code': 500
        }), 500

    return render_template('500.html'), 500


@app.errorhandler(Exception)
def handle_generic_exception(error):
    """Handle any unhandled exceptions."""
    logger = logging.getLogger(__name__)
    logger.error(f"Unhandled exception: {error}", exc_info=True)

    if request.path.startswith('/api/'):
        return jsonify({
            'error': 'Unexpected error',
            'message': 'An unexpected error occurred. Please try again.',
            'status_code': 500
        }), 500

    return render_template('error.html'), 500


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
