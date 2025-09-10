"""
UCI Bridge Module for GemmaFischer

This module provides UCI (Universal Chess Interface) compatibility for the GemmaFischer
chess engine, allowing it to interface with standard chess software.

Platform: Mac-only (M3 Pro) with MPS acceleration - no CUDA/CPU fallbacks.
"""

import sys
import json
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum
import chess
import chess.engine
from pathlib import Path

# Add the project root to the path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.inference.inference import ChessModelInterface
from src.inference.chess_engine import ChessEngineManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UCICommand(Enum):
    """UCI command types"""
    UCI = "uci"
    DEBUG = "debug"
    ISREADY = "isready"
    SETOPTION = "setoption"
    UCINEWGAME = "ucinewgame"
    POSITION = "position"
    GO = "go"
    STOP = "stop"
    QUIT = "quit"

@dataclass
class UCIPosition:
    """Represents a chess position for UCI"""
    fen: str
    moves: List[str] = None
    
    def __post_init__(self):
        if self.moves is None:
            self.moves = []

@dataclass
class UCIOptions:
    """UCI engine options"""
    mode: str = "engine"  # "engine" or "tutor"
    style: str = "balanced"  # "fischer", "aggressive", "defensive", "balanced"
    depth: int = 12
    time_limit: int = 5000  # milliseconds
    use_stockfish_fallback: bool = True

class UCIBridge:
    """
    UCI Bridge for GemmaFischer
    
    Provides UCI protocol compatibility for chess software integration.
    Supports both engine mode (fast moves) and tutor mode (explanations).
    """
    
    def __init__(self, model_path: Optional[str] = None, adapter_path: Optional[str] = None):
        """
        Initialize UCI Bridge
        
        Args:
            model_path: Path to the base model
            adapter_path: Path to the LoRA adapter
        """
        self.model_interface = None
        self.chess_engine = None
        self.options = UCIOptions()
        self.current_position = None
        self.engine_name = "GemmaFischer"
        self.author = "ChessGemma Team"
        self.version = "1.0.0"
        
        # Initialize model interface
        try:
            self.model_interface = ChessModelInterface(model_path, adapter_path)
            logger.info("Model interface initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize model interface: {e}")
            self.model_interface = None
        
        # Initialize chess engine for fallback
        try:
            self.chess_engine = ChessEngineManager()
            logger.info("Chess engine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize chess engine: {e}")
            self.chess_engine = None
    
    def handle_uci_command(self, command: str) -> str:
        """
        Handle UCI command and return response
        
        Args:
            command: UCI command string
            
        Returns:
            UCI response string
        """
        try:
            parts = command.strip().split()
            if not parts:
                return ""
            
            cmd = parts[0].lower()
            
            if cmd == UCICommand.UCI.value:
                return self._handle_uci()
            elif cmd == UCICommand.DEBUG.value:
                return self._handle_debug(parts[1:])
            elif cmd == UCICommand.ISREADY.value:
                return self._handle_isready()
            elif cmd == UCICommand.SETOPTION.value:
                return self._handle_setoption(parts[1:])
            elif cmd == UCICommand.UCINEWGAME.value:
                return self._handle_ucinewgame()
            elif cmd == UCICommand.POSITION.value:
                return self._handle_position(parts[1:])
            elif cmd == UCICommand.GO.value:
                return self._handle_go(parts[1:])
            elif cmd == UCICommand.STOP.value:
                return self._handle_stop()
            elif cmd == UCICommand.QUIT.value:
                return self._handle_quit()
            else:
                logger.warning(f"Unknown UCI command: {command}")
                return ""
                
        except Exception as e:
            logger.error(f"Error handling UCI command '{command}': {e}")
            return ""
    
    def _handle_uci(self) -> str:
        """Handle 'uci' command"""
        response = [
            f"id name {self.engine_name}",
            f"id author {self.author}",
            "option name Mode type combo default engine var engine var tutor",
            "option name Style type combo default balanced var fischer var aggressive var defensive var balanced",
            "option name Depth type spin default 12 min 1 max 20",
            "option name TimeLimit type spin default 5000 min 100 max 300000",
            "option name UseStockfishFallback type check default true",
            "uciok"
        ]
        return "\n".join(response)
    
    def _handle_debug(self, args: List[str]) -> str:
        """Handle 'debug' command"""
        if args and args[0].lower() == "on":
            logging.getLogger().setLevel(logging.DEBUG)
        else:
            logging.getLogger().setLevel(logging.INFO)
        return ""
    
    def _handle_isready(self) -> str:
        """Handle 'isready' command"""
        if self.model_interface is None:
            return "readyok"
        return "readyok"
    
    def _handle_setoption(self, args: List[str]) -> str:
        """Handle 'setoption' command"""
        if len(args) < 4 or args[0] != "name" or args[2] != "value":
            return ""
        
        option_name = args[1]
        option_value = args[3]
        
        if option_name == "Mode":
            if option_value in ["engine", "tutor"]:
                self.options.mode = option_value
        elif option_name == "Style":
            if option_value in ["fischer", "aggressive", "defensive", "balanced"]:
                self.options.style = option_value
        elif option_name == "Depth":
            try:
                self.options.depth = int(option_value)
            except ValueError:
                pass
        elif option_name == "TimeLimit":
            try:
                self.options.time_limit = int(option_value)
            except ValueError:
                pass
        elif option_name == "UseStockfishFallback":
            self.options.use_stockfish_fallback = option_value.lower() == "true"
        
        return ""
    
    def _handle_ucinewgame(self) -> str:
        """Handle 'ucinewgame' command"""
        self.current_position = None
        return ""
    
    def _handle_position(self, args: List[str]) -> str:
        """Handle 'position' command"""
        if not args:
            return ""
        
        if args[0] == "startpos":
            fen = chess.STARTING_FEN
            moves = args[2:] if len(args) > 2 and args[1] == "moves" else []
        elif args[0] == "fen":
            # Find the moves part
            moves_start = -1
            for i, arg in enumerate(args[1:], 1):
                if arg == "moves":
                    moves_start = i + 1
                    break
            
            if moves_start > 0:
                fen_parts = args[1:moves_start-1]
                moves = args[moves_start:]
            else:
                fen_parts = args[1:]
                moves = []
            
            fen = " ".join(fen_parts)
        else:
            return ""
        
        self.current_position = UCIPosition(fen=fen, moves=moves)
        return ""
    
    def _handle_go(self, args: List[str]) -> str:
        """Handle 'go' command"""
        if self.current_position is None:
            return ""
        
        # Parse go command arguments
        depth = self.options.depth
        time_limit = self.options.time_limit
        
        for i in range(0, len(args), 2):
            if i + 1 < len(args):
                if args[i] == "depth":
                    depth = int(args[i + 1])
                elif args[i] == "movetime":
                    time_limit = int(args[i + 1])
        
        # Generate move using the model
        move = self._generate_move(depth, time_limit)
        
        if move:
            return f"bestmove {move}"
        else:
            return "bestmove (none)"
    
    def _handle_stop(self) -> str:
        """Handle 'stop' command"""
        # In a real implementation, this would stop the current search
        return ""
    
    def _handle_quit(self) -> str:
        """Handle 'quit' command"""
        if self.chess_engine:
            self.chess_engine.close()
        return ""
    
    def _generate_move(self, depth: int, time_limit: int) -> Optional[str]:
        """
        Generate a move using the model or fallback to Stockfish
        
        Args:
            depth: Search depth
            time_limit: Time limit in milliseconds
            
        Returns:
            UCI move string or None
        """
        try:
            # Create board from current position
            board = chess.Board(self.current_position.fen)
            
            # Apply moves
            for move_str in self.current_position.moves:
                try:
                    move = chess.Move.from_uci(move_str)
                    board.push(move)
                except ValueError:
                    logger.warning(f"Invalid move: {move_str}")
                    continue
            
            # Check if game is over
            if board.is_game_over():
                return None
            
            # Generate move based on mode
            if self.options.mode == "tutor":
                move = self._generate_tutor_move(board, depth, time_limit)
            else:
                move = self._generate_engine_move(board, depth, time_limit)
            
            return move.uci() if move else None
            
        except Exception as e:
            logger.error(f"Error generating move: {e}")
            return None
    
    def _generate_engine_move(self, board: chess.Board, depth: int, time_limit: int) -> Optional[chess.Move]:
        """Generate a move in engine mode (fast, minimal output)"""
        try:
            if self.model_interface and not self.options.use_stockfish_fallback:
                # Use the model for move generation
                prompt = self._create_engine_prompt(board)
                response = self.model_interface.generate_response(prompt)
                move = self._parse_move_from_response(response, board)
                if move:
                    return move
            
            # Fallback to Stockfish
            if self.chess_engine:
                return self.chess_engine.get_best_move(board, depth, time_limit)
            
            return None
            
        except Exception as e:
            logger.error(f"Error in engine mode: {e}")
            return None
    
    def _generate_tutor_move(self, board: chess.Board, depth: int, time_limit: int) -> Optional[chess.Move]:
        """Generate a move in tutor mode (with explanations)"""
        try:
            if self.model_interface:
                # Use the model for move generation with explanations
                prompt = self._create_tutor_prompt(board)
                response = self.model_interface.generate_response(prompt)
                move = self._parse_move_from_response(response, board)
                if move:
                    return move
            
            # Fallback to Stockfish
            if self.chess_engine:
                return self.chess_engine.get_best_move(board, depth, time_limit)
            
            return None
            
        except Exception as e:
            logger.error(f"Error in tutor mode: {e}")
            return None
    
    def _create_engine_prompt(self, board: chess.Board) -> str:
        """Create a prompt for engine mode (minimal, fast)"""
        fen = board.fen()
        style = self.options.style
        
        prompt = f"""Position: {fen}
Style: {style}
Mode: Engine
Generate the best move in UCI format (e.g., e2e4). Respond with only the move."""
        
        return prompt
    
    def _create_tutor_prompt(self, board: chess.Board) -> str:
        """Create a prompt for tutor mode (with explanations)"""
        fen = board.fen()
        style = self.options.style
        
        prompt = f"""Position: {fen}
Style: {style}
Mode: Tutor

Analyze this position step by step:
1. Evaluate the current position
2. Identify key threats and opportunities
3. Consider candidate moves
4. Choose the best move with reasoning

Respond with the best move in UCI format at the end."""
        
        return prompt
    
    def _parse_move_from_response(self, response: str, board: chess.Board) -> Optional[chess.Move]:
        """Parse a move from the model response"""
        try:
            # Look for UCI move format in the response
            import re
            uci_pattern = r'\b([a-h][1-8][a-h][1-8][qrbn]?)\b'
            matches = re.findall(uci_pattern, response.lower())
            
            for match in matches:
                try:
                    move = chess.Move.from_uci(match)
                    if move in board.legal_moves:
                        return move
                except ValueError:
                    continue
            
            return None
            
        except Exception as e:
            logger.error(f"Error parsing move from response: {e}")
            return None

def main():
    """Main UCI loop"""
    bridge = UCIBridge()
    
    try:
        while True:
            try:
                command = input().strip()
                if not command:
                    continue
                
                response = bridge.handle_uci_command(command)
                if response:
                    print(response, flush=True)
                
                if command.lower() == "quit":
                    break
                    
            except EOFError:
                break
            except KeyboardInterrupt:
                break
                
    finally:
        if bridge.chess_engine:
            bridge.chess_engine.close()

if __name__ == "__main__":
    main()
