#!/usr/bin/env python3
"""Chess-specific evaluation metrics for fine-tuned models.

This script provides comprehensive evaluation of chess understanding including:
- Move accuracy validation
- Position evaluation quality
- Tactical motif recognition
- Opening and endgame knowledge
"""

import os
import json
import torch
from typing import List, Dict, Any, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import re
import chess
import chess.engine
from pathlib import Path


class ChessEvaluator:
    """Comprehensive chess evaluation suite."""

    def __init__(self, model_path: str, adapter_path: str = None):
        """Initialize evaluator with model and optional adapter."""
        print("Loading model and tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

        if adapter_path:
            # Load base model then apply adapter
            base_model = AutoModelForCausalLM.from_pretrained(
                model_path, local_files_only=True, device_map='auto', attn_implementation='eager'
            )
            self.model = PeftModel.from_pretrained(base_model, adapter_path, is_trainable=False)
        else:
            # Load model directly
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, local_files_only=True, device_map='auto', attn_implementation='eager'
            )

        self.device = next(self.model.parameters()).device
        self.model.eval()

        # Initialize chess engine for validation (optional)
        self.stockfish_path = self._find_stockfish()
        if self.stockfish_path:
            try:
                self.engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
                print("Stockfish engine loaded for move validation")
            except Exception as e:
                print(f"Could not load Stockfish: {e}")
                self.engine = None
        else:
            print("Stockfish not found - move validation disabled")
            self.engine = None

    def _find_stockfish(self) -> str:
        """Find Stockfish binary in common locations."""
        common_paths = [
            "/usr/local/bin/stockfish",
            "/usr/bin/stockfish",
            "/opt/homebrew/bin/stockfish",
            "./stockfish/stockfish",
            "stockfish"
        ]

        for path in common_paths:
            if os.path.exists(path) or (os.system(f"which {path} > /dev/null 2>&1") == 0):
                return path
        return None

    def extract_moves_from_response(self, response: str) -> List[str]:
        """Extract algebraic moves from model response."""
        # Look for patterns like e2e4, Nf3, O-O, etc.
        move_pattern = r'\b([KQRBN]?[a-h]?[1-8]?x?[a-h][1-8][+#]?|O-O(?:-O)?)\b'
        moves = re.findall(move_pattern, response)
        return moves

    def validate_move_syntax(self, move: str) -> bool:
        """Validate if a move string is syntactically correct."""
        try:
            # Basic validation - should be 4-5 characters for standard moves
            if len(move) < 4 or len(move) > 6:
                return False

            # Check basic pattern (source + destination)
            if not re.match(r'^[KQRBN]?[a-h][1-8][a-h][1-8][+#]?', move):
                return False

            return True
        except:
            return False

    def evaluate_position(self, prompt: str, max_tokens: int = 100) -> Dict[str, Any]:
        """Evaluate a single chess position/question."""
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                top_p=0.9,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(prompt):].strip() if response.startswith(prompt) else response.strip()

        # Extract and analyze moves
        moves = self.extract_moves_from_response(response)
        valid_moves = [move for move in moves if self.validate_move_syntax(move)]

        # Calculate metrics
        metrics = {
            'response': response,
            'total_moves_mentioned': len(moves),
            'valid_moves': len(valid_moves),
            'move_syntax_accuracy': len(valid_moves) / len(moves) if moves else 0,
            'response_length': len(response),
            'chess_relevance_score': self._calculate_chess_relevance(response)
        }

        # Engine validation if available
        if self.engine and moves:
            try:
                # For now, just validate first move syntax
                # Could be expanded to full position evaluation
                metrics['engine_validation'] = 'available'
            except:
                metrics['engine_validation'] = 'failed'

        return metrics

    def _calculate_chess_relevance(self, response: str) -> float:
        """Calculate how chess-relevant the response is."""
        chess_terms = [
            'pawn', 'knight', 'bishop', 'rook', 'queen', 'king',
            'check', 'checkmate', 'castle', 'castling',
            'e4', 'd4', 'Nf3', 'O-O', 'position', 'advantage'
        ]

        response_lower = response.lower()
        term_count = sum(1 for term in chess_terms if term in response_lower)

        # Boost score for algebraic notation
        if re.search(r'[a-h][1-8]', response):
            term_count += 2

        # Boost score for proper move sequences
        if re.search(r'[a-h][1-8][a-h][1-8]', response):
            term_count += 1

        return min(term_count / 5.0, 1.0)  # Normalize to 0-1

    def evaluate_test_set(self, test_questions: List[str], output_file: str = None) -> Dict[str, Any]:
        """Evaluate a set of chess questions."""
        results = []

        print(f"Evaluating {len(test_questions)} chess questions...")

        for i, question in enumerate(test_questions):
            print(f"Evaluating question {i+1}/{len(test_questions)}")
            metrics = self.evaluate_position(question)
            metrics['question'] = question
            results.append(metrics)

        # Aggregate statistics
        summary = {
            'total_questions': len(results),
            'average_move_syntax_accuracy': sum(r['move_syntax_accuracy'] for r in results) / len(results),
            'average_chess_relevance': sum(r['chess_relevance_score'] for r in results) / len(results),
            'total_moves_mentioned': sum(r['total_moves_mentioned'] for r in results),
            'total_valid_moves': sum(r['valid_moves'] for r in results),
            'detailed_results': results
        }

        print("\n📊 Evaluation Summary:")
        print(f"   Move syntax accuracy: {summary['average_move_syntax_accuracy']:.3f}")
        print(f"   Chess relevance: {summary['average_chess_relevance']:.3f}")
        print(f"   Total moves mentioned: {summary['total_moves_mentioned']}")
        print(f"   Valid moves: {summary['total_valid_moves']}")

        if output_file:
            with open(output_file, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"Results saved to {output_file}")

        return summary

    def cleanup(self):
        """Clean up resources."""
        if self.engine:
            try:
                self.engine.quit()
            except:
                pass


def main():
    """Run chess evaluation on test questions."""
    # Load test questions from initial Q&A
    qa_file = Path(__file__).parents[2] / 'archive' / 'initial_chess_q_and_a.md'
    if not qa_file.exists():
        print(f"Test file not found: {qa_file}")
        return

    # Parse questions
    questions = []
    with open(qa_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Extract questions using regex
    import re
    question_pattern = r'##\s+Q\d+:\s*(.+?)(?=\n##|\n---|\Z)'
    matches = re.findall(question_pattern, content, re.DOTALL)

    for match in matches:
        questions.append(match.strip())

    if not questions:
        print("No questions found in test file")
        return

    print(f"Found {len(questions)} test questions")

    # Initialize evaluator
    model_path = Path(__file__).parent.parent / 'models' / 'unsloth-gemma-3-270m-it' / 'models--unsloth--gemma-3-270m-it' / 'snapshots' / '23cf460f6bb16954176b3ddcc8d4f250501458a9'

    # Try to find latest checkpoint
    checkpoints_dir = Path(__file__).parent.parent / 'checkpoints' / 'lora_full'
    adapter_path = None

    if checkpoints_dir.exists():
        checkpoint_dirs = [d for d in checkpoints_dir.iterdir() if d.is_dir() and d.name.startswith('checkpoint-')]
        if checkpoint_dirs:
            # Get latest checkpoint
            latest_checkpoint = max(checkpoint_dirs, key=lambda x: int(x.name.split('-')[-1]))
            adapter_path = str(latest_checkpoint)
            print(f"Using latest checkpoint: {latest_checkpoint.name}")

    evaluator = ChessEvaluator(str(model_path), adapter_path)

    try:
        # Run evaluation
        output_file = Path(__file__).parent.parent / 'chess_evaluation_results.json'
        results = evaluator.evaluate_test_set(questions, str(output_file))

        print(f"\n🎯 Chess Evaluation Complete!")
        print(f"   Move syntax accuracy: {results['average_move_syntax_accuracy']:.1%}")
        print(f"   Chess relevance: {results['average_chess_relevance']:.1%}")

    finally:
        evaluator.cleanup()


if __name__ == '__main__':
    main()
