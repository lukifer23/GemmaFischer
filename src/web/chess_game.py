#!/usr/bin/env python3
"""
Chess Game Engine for Interactive Play

Handles:
- Board state management
- Move validation
- Legal move generation
- Game state tracking
- FEN position management
"""

import chess
import chess.engine
from typing import List, Dict, Any, Optional, Tuple
import json
from datetime import datetime


class ChessGame:
    """Interactive chess game with move validation and state tracking."""
    
    def __init__(self):
        self.board = chess.Board()
        self.move_history = []
        self.game_state = "active"  # active, checkmate, stalemate, draw
        self.current_player = "white"  # white, black
        self.last_move = None
        self.game_id = f"game_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
    def get_fen(self) -> str:
        """Get current board position as FEN string."""
        return self.board.fen()
    
    def get_board_visualization(self) -> str:
        """Get ASCII representation of the board."""
        return str(self.board)
    
    def get_legal_moves(self, square: Optional[str] = None) -> List[str]:
        """Get all legal moves, optionally filtered by square."""
        moves = []
        for move in self.board.legal_moves:
            move_uci = move.uci()
            if square is None or move_uci.startswith(square):
                moves.append(move_uci)
        return moves
    
    def get_piece_moves(self, square: str) -> List[str]:
        """Get legal moves for a specific piece on a square."""
        try:
            square_obj = chess.parse_square(square)
            moves = []
            for move in self.board.legal_moves:
                if move.from_square == square_obj:
                    moves.append(move.uci())
            return moves
        except ValueError:
            return []
    
    def is_legal_move(self, move_uci: str) -> bool:
        """Check if a move is legal."""
        try:
            move = chess.Move.from_uci(move_uci)
            return move in self.board.legal_moves
        except ValueError:
            return False
    
    def make_move(self, move_uci: str) -> Dict[str, Any]:
        """Make a move and return game state."""
        print(f"Attempting move: {move_uci}")
        print(f"Current FEN: {self.board.fen()}")
        print(f"Current player: {self.current_player}")
        print(f"Legal moves: {self.get_legal_moves()}")
        
        if not self.is_legal_move(move_uci):
            return {
                "success": False,
                "error": f"Invalid move: {move_uci}",
                "legal_moves": self.get_legal_moves(),
                "current_fen": self.board.fen()
            }
        
        try:
            move = chess.Move.from_uci(move_uci)
            
            # Get SAN before pushing the move (while it's still legal)
            san_before = self.board.san(move)
            
            self.board.push(move)
            
            # Use the SAN we got before the move
            san_after = san_before
            
            self.move_history.append({
                "move": move_uci,
                "san": san_after,
                "fen": self.board.fen(),
                "player": self.current_player,
                "timestamp": datetime.now().isoformat()
            })
            
            # Update game state
            self.current_player = "black" if self.current_player == "white" else "white"
            self.last_move = move_uci
            
            # Check for game end conditions
            if self.board.is_checkmate():
                self.game_state = "checkmate"
                winner = "black" if self.current_player == "white" else "white"
            elif self.board.is_stalemate():
                self.game_state = "stalemate"
                winner = None
            elif self.board.is_insufficient_material():
                self.game_state = "draw"
                winner = None
            else:
                winner = None
            
            print(f"Move successful: {move_uci} -> {san_after}")
            print(f"New FEN: {self.board.fen()}")
            print(f"New player: {self.current_player}")
            
            return {
                "success": True,
                "move": move_uci,
                "san": san_after,
                "fen": self.board.fen(),
                "current_player": self.current_player,
                "game_state": self.game_state,
                "winner": winner,
                "is_check": self.board.is_check(),
                "legal_moves": self.get_legal_moves(),
                "move_count": len(self.move_history)
            }
            
        except Exception as e:
            print(f"Move error: {str(e)}")
            return {
                "success": False,
                "error": f"Error making move: {str(e)}",
                "legal_moves": self.get_legal_moves(),
                "current_fen": self.board.fen()
            }
    
    def get_position_analysis(self, square: str) -> Dict[str, Any]:
        """Analyze a specific square and its piece."""
        try:
            square_obj = chess.parse_square(square)
            piece = self.board.piece_at(square_obj)
            
            if piece is None:
                return {
                    "square": square,
                    "piece": None,
                    "piece_name": "Empty",
                    "legal_moves": [],
                    "is_attacked": False,
                    "is_defended": False
                }
            
            piece_name = chess.piece_name(piece.piece_type)
            piece_color = "White" if piece.color else "Black"
            
            return {
                "square": square,
                "piece": piece.symbol(),
                "piece_name": f"{piece_color} {piece_name.title()}",
                "legal_moves": self.get_piece_moves(square),
                "is_attacked": self.board.is_attacked_by(not piece.color, square_obj),
                "is_defended": self.board.is_attacked_by(piece.color, square_obj)
            }
            
        except ValueError:
            return {
                "square": square,
                "piece": None,
                "piece_name": "Invalid square",
                "legal_moves": [],
                "is_attacked": False,
                "is_defended": False
            }
    
    def reset_game(self):
        """Reset the game to starting position."""
        self.board = chess.Board()
        self.move_history = []
        self.game_state = "active"
        self.current_player = "white"
        self.last_move = None
        self.game_id = f"game_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    def get_game_summary(self) -> Dict[str, Any]:
        """Get complete game state summary."""
        return {
            "game_id": self.game_id,
            "fen": self.get_fen(),
            "current_player": self.current_player,
            "game_state": self.game_state,
            "move_count": len(self.move_history),
            "last_move": self.last_move,
            "is_check": self.board.is_check(),
            "legal_moves": self.get_legal_moves(),
            "move_history": self.move_history[-10:]  # Last 10 moves
        }


class ChessRAG:
    """Retrieval Augmented Generation for chess knowledge."""
    
    def __init__(self):
        self.knowledge_base = self._load_chess_knowledge()
    
    def _load_chess_knowledge(self) -> Dict[str, List[Dict]]:
        """Load chess knowledge base."""
        return {
            "pawn_moves": [
                {
                    "position": "starting",
                    "moves": ["forward one square", "forward two squares"],
                    "captures": ["diagonally forward"],
                    "special": ["en passant", "promotion"]
                },
                {
                    "position": "middle_game",
                    "moves": ["forward one square"],
                    "captures": ["diagonally forward"],
                    "special": ["en passant", "promotion"]
                }
            ],
            "opening_principles": [
                "Control the center",
                "Develop pieces quickly",
                "Castle early",
                "Don't move the same piece twice",
                "Don't bring out the queen too early"
            ],
            "tactical_patterns": [
                "Fork: Attack two pieces simultaneously",
                "Pin: Prevent a piece from moving",
                "Skewer: Force a valuable piece to move",
                "Discovered attack: Move a piece to reveal an attack",
                "Double attack: Attack two targets at once"
            ],
            "endgame_principles": [
                "King activity is crucial",
                "Pawn promotion is the goal",
                "Opposition in king and pawn endgames",
                "Rook behind passed pawns",
                "Bishop vs Knight in endgames"
            ]
        }
    
    def get_relevant_knowledge(self, question: str, position: str = None) -> List[str]:
        """Retrieve relevant chess knowledge for a question."""
        relevant = []
        question_lower = question.lower()
        
        # Check for specific patterns
        if "pawn" in question_lower:
            relevant.extend(self.knowledge_base["pawn_moves"])
        if "opening" in question_lower or "start" in question_lower:
            relevant.extend(self.knowledge_base["opening_principles"])
        if "tactic" in question_lower or "attack" in question_lower:
            relevant.extend(self.knowledge_base["tactical_patterns"])
        if "endgame" in question_lower or "end" in question_lower:
            relevant.extend(self.knowledge_base["endgame_principles"])
        
        return relevant
    
    def get_position_specific_advice(self, fen: str, square: str) -> List[str]:
        """Get advice specific to a position and square."""
        try:
            board = chess.Board(fen)
            square_obj = chess.parse_square(square)
            piece = board.piece_at(square_obj)
            
            if piece is None:
                return ["This square is empty. Consider what pieces could move here."]
            
            advice = []
            piece_name = chess.piece_name(piece.piece_type)
            piece_color = "White" if piece.color else "Black"
            
            if piece_name == "pawn":
                advice.append(f"The {piece_color} pawn on {square} can move forward")
                if piece.color and square[1] == '2':  # White pawn on 2nd rank
                    advice.append("This pawn can move one or two squares forward")
                elif not piece.color and square[1] == '7':  # Black pawn on 7th rank
                    advice.append("This pawn can move one or two squares forward")
                else:
                    advice.append("This pawn can move one square forward")
                advice.append("Pawns capture diagonally forward")
            
            elif piece_name == "knight":
                advice.append(f"The {piece_color} knight on {square} moves in an L-shape")
                advice.append("Knights can jump over other pieces")
                advice.append("Knights are most effective in the center")
            
            elif piece_name == "bishop":
                advice.append(f"The {piece_color} bishop on {square} moves diagonally")
                advice.append("Bishops are most effective on long diagonals")
                advice.append("Bishops cannot change color squares")
            
            elif piece_name == "rook":
                advice.append(f"The {piece_color} rook on {square} moves horizontally and vertically")
                advice.append("Rooks are most effective on open files and ranks")
                advice.append("Rooks work well together")
            
            elif piece_name == "queen":
                advice.append(f"The {piece_color} queen on {square} combines rook and bishop moves")
                advice.append("The queen is the most powerful piece")
                advice.append("Be careful not to bring the queen out too early")
            
            elif piece_name == "king":
                advice.append(f"The {piece_color} king on {square} moves one square in any direction")
                advice.append("The king should be kept safe, especially in the opening")
                advice.append("Castling is an important king safety move")
            
            return advice
            
        except Exception as e:
            return [f"Error analyzing position: {str(e)}"]
