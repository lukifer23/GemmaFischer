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

from src.inference.inference import ChessGemmaInference
from src.inference.chess_engine import ChessEngineManager
from src.inference.uci_utils import (
    post_process_uci_response, 
    create_engine_prompt_strict,
    create_tutor_prompt_with_uci,
    extract_and_validate_uci
)

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
    mode: str = "auto"  # "auto", "uci", "tutor", "director"
    style: str = "balanced"  # "fischer", "aggressive", "defensive", "balanced"
    depth: int = 12
    time_limit: int = 5000  # milliseconds
    moe_enabled: bool = True  # Use MoE routing
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
            adapter_path: Path to the LoRA adapter (optional with MoE)
        """
        self.inference = None
        self.chess_engine = None
        self.options = UCIOptions()
        self.current_position = None
        self.engine_name = "ChessGemma"
        self.author = "ChessGemma Team"
        self.version = "2.0.0"
        self._last_generation_metadata: Dict[str, Optional[str]] = {
            "requested_mode": None,
            "routed_mode": None,
            "generation_mode": None,
        }

        # Initialize inference with MoE support
        try:
            self.inference = ChessGemmaInference(model_path, adapter_path)
            logger.info("ChessGemma inference initialized with MoE support")
        except Exception as e:
            logger.error(f"Failed to create inference: {e}")
            self.inference = None

        # Initialize chess engine for fallback and validation
        try:
            self.chess_engine = ChessEngineManager()
            logger.info("Chess engine initialized for validation and fallback")
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
            f"id name {self.engine_name} {self.version}",
            f"id author {self.author}",
            "option name Mode type combo default auto var auto var uci var tutor var director",
            "option name MoE_Enabled type check default true",
            "option name Style type combo default balanced var fischer var aggressive var defensive var balanced",
            "option name Depth type spin default 12 min 1 max 20",
            "option name TimeLimit type spin default 5000 min 100 max 300000",
            "option name Movetime type spin default 5000 min 100 max 300000",
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
        if self.inference is None:
            logger.debug("Inference not initialized; responding with readyok")
            return "readyok"

        if not getattr(self.inference, "is_loaded", False):
            logger.debug("Inference not yet loaded; responding with readyok")

        return "readyok"
    
    def _handle_setoption(self, args: List[str]) -> str:
        """Handle 'setoption' command"""
        if len(args) < 4 or args[0] != "name" or args[2] != "value":
            return ""
        
        option_name = args[1]
        option_value = args[3]
        
        if option_name == "Mode":
            if option_value in ["auto", "uci", "tutor", "director"]:
                self.options.mode = option_value
        elif option_name == "UCI_Mode":  # Legacy support
            if option_value in ["engine", "tutor"]:
                self.options.mode = "uci" if option_value == "engine" else option_value
        elif option_name == "MoE_Enabled":
            self.options.moe_enabled = option_value.lower() in ("true", "1", "yes", "on")
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
        elif option_name == "Movetime":
            try:
                self.options.time_limit = int(option_value)
            except ValueError:
                pass
        elif option_name == "UseStockfishFallback":
            self.options.use_stockfish_fallback = option_value.lower() in ("true", "1", "yes", "on")
        
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
            try:
                self.chess_engine.cleanup()
            except Exception:
                pass
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
            
            # Generate move using ChessGemma with MoE routing
            move_uci = self._generate_chessgemmma_move(board, depth, time_limit)

            if move_uci:
                try:
                    move = chess.Move.from_uci(move_uci)
                    if move in board.legal_moves:
                        return move
                except ValueError:
                    pass

            # Fallback to Stockfish if available and enabled
            if self.options.use_stockfish_fallback and self.chess_engine:
                return self._generate_stockfish_move(board, depth, time_limit)

            return None
            
        except Exception as e:
            logger.error(f"Error generating move: {e}")
            return None

    def _generate_chessgemmma_move(self, board: chess.Board, depth: int, time_limit: int) -> Optional[str]:
        """Generate a move using LC0 hybrid engine as primary, LLM as fallback"""
        if self.inference is None:
            self._last_generation_metadata = {
                "requested_mode": self.options.mode,
                "routed_mode": None,
                "generation_mode": None,
            }
            return None

        try:
            # Convert board to FEN
            fen = board.fen()

            # Determine expert mode based on UCI options
            requested_mode = self.options.mode
            routed_mode = requested_mode
            if routed_mode == "auto" and not self.options.moe_enabled:
                routed_mode = "uci"

            generation_mode = "engine" if routed_mode in ("uci", "auto") else routed_mode
            self._last_generation_metadata = {
                "requested_mode": requested_mode,
                "routed_mode": routed_mode,
                "generation_mode": generation_mode,
            }

            # Primary: Use LC0 hybrid engine for UCI moves (much more reliable than LLM)
            if routed_mode in ("uci", "auto"):
                try:
                    engine_result = self.inference.analyze_with_engine(fen)
                    uci_move = engine_result.get('best_move')
                    if uci_move:
                        # Validate the move is legal
                        try:
                            move_obj = chess.Move.from_uci(uci_move)
                            if board.is_legal(move_obj):
                                logger.info(f"LC0 hybrid engine generated valid UCI move: {uci_move}")
                                return uci_move
                        except (ValueError, chess.InvalidMoveError):
                            logger.warning(f"LC0 generated invalid UCI move: {uci_move}")
                except Exception as e:
                    logger.warning(f"LC0 hybrid engine failed, falling back to LLM: {e}")

            # Fallback: Use LLM model for tutor/director modes or when engine fails
            if routed_mode != "uci":
                prompt = create_tutor_prompt_with_uci(fen, "Analyze this position step by step")
            else:
                prompt = create_engine_prompt_strict(fen)

            # Generate response using ChessGemma LLM
            response_dict = self.inference.generate_response(
                prompt,
                mode=generation_mode,
                max_new_tokens=8,  # Limit for UCI moves
                temperature=0.0,   # Deterministic for engine mode
                top_p=1.0
            )

            response = response_dict.get('response', '')
            if not response:
                logger.warning("Empty response from ChessGemma LLM")
                return None

            # Use enhanced post-processing with strict validation
            uci_move = post_process_uci_response(
                response,
                board,
                fallback_to_stockfish=self.options.use_stockfish_fallback
            )

            if uci_move:
                logger.info(f"ChessGemma LLM generated valid UCI move: {uci_move}")
                return uci_move
            else:
                logger.warning(f"ChessGemma LLM failed to generate valid UCI move from: {response[:100]}...")
                return None

        except Exception as e:
            logger.error(f"Error generating ChessGemma move: {e}")
            return None

    def _generate_stockfish_move(self, board: chess.Board, depth: int, time_limit: int) -> Optional[chess.Move]:
        """Generate a move using Stockfish as fallback"""
        if not self.chess_engine:
            return None

        try:
            # Use Stockfish to find best move
            result = self.chess_engine.get_best_move(
                board,
                depth=min(depth, 15),  # Limit depth for UCI compatibility
                time_limit_ms=int(time_limit * 1000)  # Convert seconds to milliseconds
            )
            return result
        except Exception as e:
            logger.error(f"Error generating Stockfish move: {e}")
            return None

    
    def _create_engine_prompt(self, board: chess.Board) -> str:
        """Deprecated: use build_engine_prompt instead."""
        return build_engine_prompt(board.fen())
    
    def _create_tutor_prompt(self, board: chess.Board) -> str:
        """Create a prompt for tutor mode (with explanations)"""
        fen = board.fen()
        style = self.options.style
        
        prompt = f"""FEN: {fen}
Question: Analyze this position step by step.
Style: {style}
Mode: Tutor

1. Evaluate the current position
2. Identify key threats and opportunities
3. Consider candidate moves
4. Choose the best move with reasoning

Respond with the best move in UCI format at the end."""
        
        return prompt
    
    def _parse_move_from_response(self, response: str, board: chess.Board) -> Optional[chess.Move]:
        """Deprecated: use extract_first_legal_move instead."""
        return extract_first_legal_move(response, board)

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
            try:
                bridge.chess_engine.cleanup()
            except Exception:
                pass

if __name__ == "__main__":
    main()
