#!/usr/bin/env python3
"""
Stockfish vs ChessGemma Match System
Allows the model to play against Stockfish with no time limits for the model.
"""

import chess
import chess.engine
import time
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

@dataclass
class MoveResult:
    move: str
    san: str
    fen: str
    time_taken: float
    evaluation: Optional[float] = None
    depth: Optional[int] = None

@dataclass
class GameResult:
    moves: List[MoveResult]
    winner: str  # 'white', 'black', 'draw'
    reason: str  # 'checkmate', 'stalemate', 'resignation', 'timeout', 'draw'
    final_fen: str
    total_moves: int
    game_duration: float

class StockfishMatch:
    def __init__(self, stockfish_path: str = None, time_control: str = "10+0.1"):
        """
        Initialize Stockfish match system.
        
        Args:
            stockfish_path: Path to Stockfish executable (auto-detect if None)
            time_control: Time control for Stockfish (e.g., "10+0.1" = 10 seconds + 0.1s increment)
        """
        self.stockfish_path = stockfish_path or self._find_stockfish()
        self.time_control = time_control
        self.engine = None
        self.board = chess.Board()
        self.moves = []
        self.start_time = None
        
    def _find_stockfish(self) -> str:
        """Find Stockfish executable path."""
        import shutil
        import os
        
        # Common paths to check
        possible_paths = [
            "/opt/homebrew/bin/stockfish",  # Homebrew on Apple Silicon
            "/usr/local/bin/stockfish",     # Homebrew on Intel Mac
            "/usr/bin/stockfish",           # System installation
            "stockfish"                     # In PATH
        ]
        
        for path in possible_paths:
            if os.path.exists(path) or shutil.which(path):
                print(f"✅ Found Stockfish at: {path}")
                return path
        
        # If not found, try to find it in PATH
        stockfish_path = shutil.which("stockfish")
        if stockfish_path:
            print(f"✅ Found Stockfish in PATH: {stockfish_path}")
            return stockfish_path
        
        raise RuntimeError("Stockfish not found. Please install it with: brew install stockfish")
        
    def start_engine(self) -> bool:
        """Start Stockfish engine."""
        try:
            self.engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
            print(f"✅ Stockfish engine started successfully")
            return True
        except Exception as e:
            print(f"❌ Failed to start Stockfish: {e}")
            return False
    
    def stop_engine(self):
        """Stop Stockfish engine."""
        if self.engine:
            self.engine.quit()
            self.engine = None
            print("🛑 Stockfish engine stopped")
    
    def reset_game(self):
        """Reset the game to starting position."""
        self.board = chess.Board()
        self.moves = []
        self.start_time = None
        print("🔄 Game reset to starting position")
    
    def get_stockfish_move(self) -> MoveResult:
        """Get Stockfish's move with time control."""
        if not self.engine:
            raise RuntimeError("Stockfish engine not started")
        
        start_time = time.time()
        
        # Parse time control (e.g., "10+0.1" -> 10.0 seconds + 0.1s increment)
        if '+' in self.time_control:
            base_time, increment = self.time_control.split('+')
            time_limit = float(base_time) + (len(self.moves) * float(increment))
        else:
            time_limit = float(self.time_control)
        
        # Get Stockfish's move
        result = self.engine.play(
            self.board, 
            chess.engine.Limit(time=time_limit),
            info=chess.engine.INFO_ALL
        )
        
        move_time = time.time() - start_time
        move_uci = result.move.uci()
        san = self.board.san(result.move)
        
        # Get evaluation info
        evaluation = None
        depth = None
        if hasattr(result, 'info') and 'score' in result.info:
            score = result.info['score']
            if score.is_mate():
                evaluation = float('inf') if score.mate() > 0 else float('-inf')
            else:
                evaluation = score.relative.score(mate_score=10000) / 100.0
        
        if hasattr(result, 'info') and 'depth' in result.info:
            depth = result.info['depth']
        
        # Make the move
        self.board.push(result.move)
        
        move_result = MoveResult(
            move=move_uci,
            san=san,
            fen=self.board.fen(),
            time_taken=move_time,
            evaluation=evaluation,
            depth=depth
        )
        
        self.moves.append(move_result)
        
        print(f"🏆 Stockfish played: {san} ({move_uci})")
        print(f"⏱️  Time: {move_time:.2f}s")
        if evaluation is not None:
            print(f"📊 Evaluation: {evaluation:+.2f}")
        if depth is not None:
            print(f"🔍 Depth: {depth}")
        
        return move_result
    
    def get_model_move(self, model_generator, legal_moves: List[str]) -> MoveResult:
        """Get model's move (no time limit)."""
        start_time = time.time()
        
        print(f"\n🤖 MODEL MOVE REQUEST")
        print(f"Position: {self.board.fen()}")
        print(f"Legal moves: {legal_moves}")
        
        # Create educational prompt for the model
        question = f"""You are playing as {'white' if self.board.turn else 'black'} in this chess position: {self.board.fen()}

The legal moves available are: {', '.join(legal_moves[:10])}

Please:
1. Choose the best move from the legal moves
2. Explain why this move is good
3. Comment on what the opponent just played
4. Suggest what the opponent should consider next

Respond with your chosen move first, then your analysis."""
        
        # Get model response
        result = model_generator(question, f"Current position: {self.board.fen()}")
        response_text = result.get('response', '')
        
        # Extract move from response
        move_uci = self._extract_move_from_response(response_text, legal_moves)
        
        if not move_uci:
            # Fallback to random move
            import random
            move_uci = random.choice(legal_moves)
            print(f"⚠️  Using fallback move: {move_uci}")
        
        # Make the move
        move = chess.Move.from_uci(move_uci)
        san = self.board.san(move)
        self.board.push(move)
        
        move_time = time.time() - start_time
        
        move_result = MoveResult(
            move=move_uci,
            san=san,
            fen=self.board.fen(),
            time_taken=move_time
        )
        
        self.moves.append(move_result)
        
        print(f"🤖 Model played: {san} ({move_uci})")
        print(f"⏱️  Time: {move_time:.2f}s")
        print(f"💭 Response: {response_text[:200]}...")
        
        return move_result
    
    def _extract_move_from_response(self, response: str, legal_moves: List[str]) -> Optional[str]:
        """Extract a legal move from model response."""
        import re
        
        # Look for UCI format moves
        uci_pattern = r'\b([a-h][1-8][a-h][1-8][qrbn]?)\b'
        matches = re.findall(uci_pattern, response.lower())
        
        for match in matches:
            if match in legal_moves:
                return match
        
        # Look for partial matches
        for move in legal_moves:
            if move.lower() in response.lower():
                return move
        
        return None
    
    def play_game(self, model_generator, model_plays_white: bool = True, max_moves: int = 200) -> GameResult:
        """Play a complete game between Stockfish and the model."""
        if not self.start_engine():
            raise RuntimeError("Could not start Stockfish engine")
        
        self.reset_game()
        self.start_time = time.time()
        
        print(f"\n🎮 STARTING STOCKFISH vs MODEL MATCH")
        print(f"📋 Model plays: {'White' if model_plays_white else 'Black'}")
        print(f"⏰ Stockfish time control: {self.time_control}")
        print(f"🎯 Max moves: {max_moves}")
        print("=" * 60)
        
        move_count = 0
        
        while not self.board.is_game_over() and move_count < max_moves:
            legal_moves = [move.uci() for move in self.board.legal_moves]
            
            if not legal_moves:
                break
            
            is_model_turn = (self.board.turn == chess.WHITE) == model_plays_white
            
            if is_model_turn:
                move_result = self.get_model_move(model_generator, legal_moves)
            else:
                move_result = self.get_stockfish_move()
            
            move_count += 1
            print(f"Move {move_count}: {move_result.san}")
            print("-" * 40)
        
        # Determine game result
        game_duration = time.time() - self.start_time
        winner, reason = self._determine_result()
        
        game_result = GameResult(
            moves=self.moves,
            winner=winner,
            reason=reason,
            final_fen=self.board.fen(),
            total_moves=move_count,
            game_duration=game_duration
        )
        
        self._print_game_summary(game_result)
        return game_result
    
    def _determine_result(self) -> Tuple[str, str]:
        """Determine the game result."""
        if self.board.is_checkmate():
            winner = 'white' if self.board.turn == chess.BLACK else 'black'
            reason = 'checkmate'
        elif self.board.is_stalemate():
            winner = 'draw'
            reason = 'stalemate'
        elif self.board.is_insufficient_material():
            winner = 'draw'
            reason = 'insufficient_material'
        elif self.board.is_seventyfive_moves():
            winner = 'draw'
            reason = 'seventyfive_moves'
        elif self.board.is_fivefold_repetition():
            winner = 'draw'
            reason = 'fivefold_repetition'
        else:
            winner = 'draw'
            reason = 'unknown'
        
        return winner, reason
    
    def _print_game_summary(self, result: GameResult):
        """Print game summary."""
        print("\n" + "=" * 60)
        print("🏁 GAME SUMMARY")
        print("=" * 60)
        print(f"🏆 Winner: {result.winner.upper()}")
        print(f"📋 Reason: {result.reason}")
        print(f"🎯 Total moves: {result.total_moves}")
        print(f"⏱️  Game duration: {result.game_duration:.1f}s")
        print(f"📊 Final position: {result.final_fen}")
        
        # Print move list
        print("\n📝 Move List:")
        for i, move in enumerate(result.moves, 1):
            player = "Model" if i % 2 == 1 else "Stockfish"
            print(f"{i:2d}. {player:8s}: {move.san:6s} ({move.time_taken:.2f}s)")
        
        print("=" * 60)
    
    def save_game(self, filename: str, result: GameResult):
        """Save game to PGN file."""
        try:
            import chess.pgn
            
            game = chess.pgn.Game()
            game.headers["Event"] = "Stockfish vs ChessGemma"
            game.headers["Site"] = "ChessGemma Match"
            game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
            game.headers["Round"] = "1"
            game.headers["White"] = "ChessGemma Model"
            game.headers["Black"] = "Stockfish"
            game.headers["Result"] = result.winner.upper() if result.winner != 'draw' else "1/2-1/2"
            
            node = game
            for move in result.moves:
                node = node.add_variation(chess.Move.from_uci(move.move))
            
            with open(filename, 'w') as f:
                print(game, file=f)
            
            print(f"💾 Game saved to {filename}")
        except Exception as e:
            print(f"❌ Failed to save game: {e}")

def main():
    """Example usage of StockfishMatch."""
    # This would be called from the web interface
    pass

if __name__ == "__main__":
    main()
