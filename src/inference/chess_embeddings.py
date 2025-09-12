#!/usr/bin/env python3
"""
Chess Position Embeddings and Similarity Search

Provides vector embeddings for chess positions and efficient similarity search:
- FEN-to-embedding conversion using chess-specific features
- Vector similarity search for finding related positions
- Retrieval-augmented generation for enhanced chess analysis
- Position clustering and pattern recognition
"""

from __future__ import annotations

import numpy as np
import chess
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
import hashlib
import pickle
from pathlib import Path
import time
from collections import defaultdict

# Add project root to path
import sys
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

try:
    from ..utils.logging_config import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


@dataclass
class PositionEmbedding:
    """Embedding data for a chess position."""
    fen: str
    embedding: np.ndarray
    metadata: Dict[str, Any]
    position_hash: str


@dataclass
class SimilarityResult:
    """Result of a similarity search."""
    fen: str
    similarity_score: float
    metadata: Dict[str, Any]
    common_features: List[str]


class ChessPositionEmbedder:
    """
    Chess-specific position embedder using domain features.

    Creates embeddings based on:
    - Piece positions and types
    - Material balance
    - King safety
    - Pawn structure
    - Piece mobility
    - Tactical motifs
    """

    def __init__(self, embedding_dim: int = 256):
        self.embedding_dim = embedding_dim
        self.feature_weights = {
            'material': 0.25,
            'king_safety': 0.20,
            'pawn_structure': 0.15,
            'piece_mobility': 0.15,
            'tactical_motifs': 0.15,
            'positional': 0.10
        }

        # Initialize piece value mapping
        self.piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0  # King has no material value in counting
        }

        logger.info(f"🔧 Chess Position Embedder initialized (dim={embedding_dim})")

    def embed_position(self, fen: str) -> PositionEmbedding:
        """
        Create embedding for a chess position.

        Args:
            fen: FEN string of the position

        Returns:
            PositionEmbedding with vector representation
        """
        try:
            board = chess.Board(fen)
        except ValueError:
            raise ValueError(f"Invalid FEN: {fen}")

        # Extract features
        features = self._extract_position_features(board)

        # Create embedding vector
        embedding = self._features_to_embedding(features)

        # Generate position hash for deduplication
        position_hash = hashlib.md5(fen.encode()).hexdigest()

        # Extract metadata
        metadata = self._extract_position_metadata(board, features)

        return PositionEmbedding(
            fen=fen,
            embedding=embedding,
            metadata=metadata,
            position_hash=position_hash
        )

    def _extract_position_features(self, board: chess.Board) -> Dict[str, Any]:
        """Extract chess-specific features from position."""
        features = {}

        # Material balance
        features['material'] = self._calculate_material_balance(board)

        # King safety
        features['king_safety'] = self._calculate_king_safety(board)

        # Pawn structure
        features['pawn_structure'] = self._calculate_pawn_structure(board)

        # Piece mobility
        features['piece_mobility'] = self._calculate_piece_mobility(board)

        # Tactical motifs
        features['tactical_motifs'] = self._detect_tactical_motifs(board)

        # Positional features
        features['positional'] = self._calculate_positional_features(board)

        return features

    def _calculate_material_balance(self, board: chess.Board) -> Dict[str, float]:
        """Calculate material balance and distribution."""
        white_material = 0
        black_material = 0
        piece_counts = defaultdict(int)

        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                piece_type = piece.piece_type
                piece_counts[f"{'white' if piece.color else 'black'}_{chess.piece_name(piece_type)}"] += 1

                if piece.color == chess.WHITE:
                    white_material += self.piece_values[piece_type]
                else:
                    black_material += self.piece_values[piece_type]

        return {
            'balance': white_material - black_material,
            'total_material': white_material + black_material,
            'white_material': white_material,
            'black_material': black_material,
            'piece_counts': dict(piece_counts),
            'material_ratio': white_material / max(black_material, 1)
        }

    def _calculate_king_safety(self, board: chess.Board) -> Dict[str, Any]:
        """Calculate king safety metrics."""
        white_king_square = board.king(chess.WHITE)
        black_king_square = board.king(chess.BLACK)

        # King zone definition (3x3 area around king)
        def king_zone_squares(king_square):
            rank, file = chess.square_rank(king_square), chess.square_file(king_square)
            zone_squares = []
            for dr in [-1, 0, 1]:
                for df in [-1, 0, 1]:
                    r, f = rank + dr, file + df
                    if 0 <= r <= 7 and 0 <= f <= 7:
                        zone_squares.append(chess.square(f, r))
            return zone_squares

        white_king_zone = king_zone_squares(white_king_square)
        black_king_zone = king_zone_squares(black_king_square)

        # Count attackers in king zones
        white_attackers = 0
        black_attackers = 0

        for square in white_king_zone:
            if board.is_attacked_by(chess.BLACK, square):
                white_attackers += 1

        for square in black_king_zone:
            if board.is_attacked_by(chess.WHITE, square):
                black_attackers += 1

        return {
            'white_king_zone_attackers': white_attackers,
            'black_king_zone_attackers': black_attackers,
            'king_zone_balance': white_attackers - black_attackers,
            'white_king_open_files': self._count_open_files_near_king(board, chess.WHITE),
            'black_king_open_files': self._count_open_files_near_king(board, chess.BLACK)
        }

    def _count_open_files_near_king(self, board: chess.Board, color: chess.Color) -> int:
        """Count open files near the king."""
        king_square = board.king(color)
        king_file = chess.square_file(king_square)

        open_files = 0
        # Check files adjacent to king
        for df in [-1, 0, 1]:
            f = king_file + df
            if 0 <= f <= 7:
                if self._is_open_file(board, f):
                    open_files += 1

        return open_files

    def _is_open_file(self, board: chess.Board, file: int) -> bool:
        """Check if a file is open (no pawns)."""
        for rank in range(8):
            square = chess.square(file, rank)
            piece = board.piece_at(square)
            if piece and piece.piece_type == chess.PAWN:
                return False
        return True

    def _calculate_pawn_structure(self, board: chess.Board) -> Dict[str, Any]:
        """Calculate pawn structure features."""
        white_pawns = []
        black_pawns = []

        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.piece_type == chess.PAWN:
                if piece.color == chess.WHITE:
                    white_pawns.append(chess.square_file(square))
                else:
                    black_pawns.append(chess.square_file(square))

        # Calculate pawn islands (groups of pawns separated by open files)
        white_islands = self._count_pawn_islands(white_pawns)
        black_islands = self._count_pawn_islands(black_pawns)

        # Calculate doubled pawns
        white_doubled = len(white_pawns) - len(set(white_pawns))
        black_doubled = len(black_pawns) - len(set(black_pawns))

        return {
            'white_pawn_count': len(white_pawns),
            'black_pawn_count': len(black_pawns),
            'pawn_balance': len(white_pawns) - len(black_pawns),
            'white_pawn_islands': white_islands,
            'black_pawn_islands': black_islands,
            'white_doubled_pawns': white_doubled,
            'black_doubled_pawns': black_doubled,
            'pawn_structure_balance': white_islands - black_islands
        }

    def _count_pawn_islands(self, pawn_files: List[int]) -> int:
        """Count pawn islands in a list of pawn files."""
        if not pawn_files:
            return 0

        sorted_files = sorted(set(pawn_files))
        islands = 1

        for i in range(1, len(sorted_files)):
            if sorted_files[i] > sorted_files[i-1] + 1:
                islands += 1

        return islands

    def _calculate_piece_mobility(self, board: chess.Board) -> Dict[str, Any]:
        """Calculate piece mobility metrics."""
        white_moves = len(list(board.legal_moves))
        board.turn = chess.BLACK
        black_moves = len(list(board.legal_moves))
        board.turn = chess.WHITE  # Reset

        # Count piece-specific mobility
        white_piece_mobility = self._count_piece_mobility(board, chess.WHITE)
        black_piece_mobility = self._count_piece_mobility(board, chess.BLACK)

        return {
            'white_total_moves': white_moves,
            'black_total_moves': black_moves,
            'move_balance': white_moves - black_moves,
            'white_piece_mobility': white_piece_mobility,
            'black_piece_mobility': black_piece_mobility
        }

    def _count_piece_mobility(self, board: chess.Board, color: chess.Color) -> Dict[str, int]:
        """Count mobility for each piece type."""
        mobility = defaultdict(int)

        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color == color:
                piece_name = chess.piece_name(piece.piece_type)
                # Count legal moves for this piece (simplified)
                board_copy = board.copy()
                board_copy.turn = color
                piece_moves = 0
                for move in board_copy.legal_moves:
                    if move.from_square == square:
                        piece_moves += 1
                mobility[piece_name] += piece_moves

        return dict(mobility)

    def _detect_tactical_motifs(self, board: chess.Board) -> Dict[str, int]:
        """Detect tactical motifs in the position."""
        motifs = defaultdict(int)

        # Check for pins
        for square in chess.SQUARES:
            if board.is_pinned(chess.WHITE, square) or board.is_pinned(chess.BLACK, square):
                motifs['pins'] += 1

        # Check for attacks on king
        if board.is_check():
            motifs['checks'] += 1

        # Count hanging pieces (simplified)
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                attackers = len(list(board.attackers(not piece.color, square)))
                defenders = len(list(board.attackers(piece.color, square)))
                if attackers > defenders:
                    motifs['hanging_pieces'] += 1

        # Check for castling rights
        if board.has_kingside_castling_rights(chess.WHITE) or board.has_queenside_castling_rights(chess.WHITE):
            motifs['white_castling_rights'] += 1
        if board.has_kingside_castling_rights(chess.BLACK) or board.has_queenside_castling_rights(chess.BLACK):
            motifs['black_castling_rights'] += 1

        return dict(motifs)

    def _calculate_positional_features(self, board: chess.Board) -> Dict[str, Any]:
        """Calculate general positional features."""
        # Center control
        center_squares = [chess.D4, chess.D5, chess.E4, chess.E5]
        white_center_control = sum(1 for sq in center_squares if board.is_attacked_by(chess.WHITE, sq))
        black_center_control = sum(1 for sq in center_squares if board.is_attacked_by(chess.BLACK, sq))

        # Development (pieces moved from starting squares)
        white_developed_pieces = 0
        black_developed_pieces = 0

        # Starting positions for knights and bishops
        starting_squares = {
            chess.WHITE: [chess.B1, chess.G1, chess.C1, chess.F1],  # knights and bishops
            chess.BLACK: [chess.B8, chess.G8, chess.C8, chess.F8]
        }

        for color in [chess.WHITE, chess.BLACK]:
            for square in starting_squares[color]:
                piece = board.piece_at(square)
                if not piece or piece.piece_type not in [chess.KNIGHT, chess.BISHOP]:
                    if color == chess.WHITE:
                        white_developed_pieces += 1
                    else:
                        black_developed_pieces += 1

        return {
            'white_center_control': white_center_control,
            'black_center_control': black_center_control,
            'center_control_balance': white_center_control - black_center_control,
            'white_developed_pieces': white_developed_pieces,
            'black_developed_pieces': black_developed_pieces,
            'development_balance': white_developed_pieces - black_developed_pieces
        }

    def _features_to_embedding(self, features: Dict[str, Any]) -> np.ndarray:
        """Convert extracted features to vector embedding."""
        # Flatten features into a vector
        feature_vector = []

        # Material features
        material = features['material']
        feature_vector.extend([
            material['balance'] / 39,  # Max material imbalance
            material['total_material'] / 78,  # Max total material
            np.log1p(material['material_ratio']) / 5  # Log ratio, normalized
        ])

        # King safety features
        king_safety = features['king_safety']
        feature_vector.extend([
            king_safety['white_king_zone_attackers'] / 9,  # Max 9 attackers
            king_safety['black_king_zone_attackers'] / 9,
            king_safety['king_zone_balance'] / 9,
            king_safety['white_king_open_files'] / 3,
            king_safety['black_king_open_files'] / 3
        ])

        # Pawn structure features
        pawn_structure = features['pawn_structure']
        feature_vector.extend([
            pawn_structure['pawn_balance'] / 16,  # Max 16 pawns
            pawn_structure['white_pawn_islands'] / 8,
            pawn_structure['black_pawn_islands'] / 8,
            pawn_structure['pawn_structure_balance'] / 8
        ])

        # Piece mobility features
        piece_mobility = features['piece_mobility']
        feature_vector.extend([
            piece_mobility['move_balance'] / 100,  # Normalize move count difference
            piece_mobility['white_total_moves'] / 200,
            piece_mobility['black_total_moves'] / 200
        ])

        # Tactical motifs (normalized counts)
        tactical_motifs = features['tactical_motifs']
        feature_vector.extend([
            min(tactical_motifs.get('pins', 0) / 10, 1),
            min(tactical_motifs.get('checks', 0) / 5, 1),
            min(tactical_motifs.get('hanging_pieces', 0) / 16, 1)
        ])

        # Positional features
        positional = features['positional']
        feature_vector.extend([
            positional['center_control_balance'] / 4,
            positional['development_balance'] / 4
        ])

        # Convert to numpy array and pad/truncate to embedding dimension
        embedding = np.array(feature_vector, dtype=np.float32)

        if len(embedding) < self.embedding_dim:
            # Pad with zeros
            padding = np.zeros(self.embedding_dim - len(embedding), dtype=np.float32)
            embedding = np.concatenate([embedding, padding])
        elif len(embedding) > self.embedding_dim:
            # Truncate
            embedding = embedding[:self.embedding_dim]

        # Normalize to unit vector
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding /= norm

        return embedding

    def _extract_position_metadata(self, board: chess.Board, features: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metadata for the position."""
        return {
            'phase': self._determine_game_phase(board),
            'material_balance': features['material']['balance'],
            'total_pieces': len([p for p in board.board_fen() if p.isalpha()]),
            'castling_rights': {
                'white_kingside': board.has_kingside_castling_rights(chess.WHITE),
                'white_queenside': board.has_queenside_castling_rights(chess.WHITE),
                'black_kingside': board.has_kingside_castling_rights(chess.BLACK),
                'black_queenside': board.has_queenside_castling_rights(chess.BLACK)
            },
            'is_check': board.is_check(),
            'is_checkmate': board.is_checkmate(),
            'is_stalemate': board.is_stalemate(),
            'halfmove_clock': board.halfmove_clock,
            'fullmove_number': board.fullmove_number
        }

    def _determine_game_phase(self, board: chess.Board) -> str:
        """Determine the game phase."""
        total_material = sum(1 for square in chess.SQUARES
                           if board.piece_at(square) and
                           board.piece_at(square).piece_type != chess.KING)

        if total_material >= 28:  # Most pieces still on board
            return 'opening'
        elif total_material >= 12:  # Some pieces traded
            return 'middlegame'
        else:
            return 'endgame'


class ChessPositionRetriever:
    """Efficient similarity search for chess positions."""

    def __init__(self, embedder: ChessPositionEmbedder):
        self.embedder = embedder
        self.index: Dict[str, PositionEmbedding] = {}
        self.position_hashes: Set[str] = set()

        logger.info("🔧 Chess Position Retriever initialized")

    def add_position(self, fen: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Add a position to the index."""
        try:
            embedding = self.embedder.embed_position(fen)
            if metadata:
                embedding.metadata.update(metadata)

            # Avoid duplicates
            if embedding.position_hash in self.position_hashes:
                return False

            self.index[embedding.position_hash] = embedding
            self.position_hashes.add(embedding.position_hash)
            return True

        except Exception as e:
            logger.warning(f"Failed to add position {fen}: {e}")
            return False

    def find_similar_positions(self, query_fen: str, top_k: int = 5,
                             min_similarity: float = 0.7) -> List[SimilarityResult]:
        """Find similar positions to the query."""
        try:
            query_embedding = self.embedder.embed_position(query_fen)
        except Exception as e:
            logger.warning(f"Failed to embed query position: {e}")
            return []

        similarities = []

        for position_hash, embedding in self.index.items():
            if position_hash == query_embedding.position_hash:
                continue  # Skip identical position

            # Calculate cosine similarity
            similarity = np.dot(query_embedding.embedding, embedding.embedding)

            if similarity >= min_similarity:
                # Find common features
                common_features = self._find_common_features(
                    query_embedding.metadata, embedding.metadata
                )

                similarities.append(SimilarityResult(
                    fen=embedding.fen,
                    similarity_score=similarity,
                    metadata=embedding.metadata,
                    common_features=common_features
                ))

        # Sort by similarity and return top-k
        similarities.sort(key=lambda x: x.similarity_score, reverse=True)
        return similarities[:top_k]

    def _find_common_features(self, metadata1: Dict[str, Any],
                            metadata2: Dict[str, Any]) -> List[str]:
        """Find common features between two positions."""
        common = []

        # Same game phase
        if metadata1.get('phase') == metadata2.get('phase'):
            common.append(f"phase:{metadata1['phase']}")

        # Similar material balance
        balance1 = metadata1.get('material_balance', 0)
        balance2 = metadata2.get('material_balance', 0)
        if abs(balance1 - balance2) <= 2:
            common.append(f"material_balance:{balance1}")

        # Same castling rights
        castling1 = metadata1.get('castling_rights', {})
        castling2 = metadata2.get('castling_rights', {})
        if castling1 == castling2:
            common.append("castling_rights")

        # Check status
        if metadata1.get('is_check') == metadata2.get('is_check'):
            if metadata1.get('is_check'):
                common.append("check")

        return common

    def save_index(self, filepath: Path) -> None:
        """Save the position index to disk."""
        filepath.parent.mkdir(parents=True, exist_ok=True)

        index_data = {
            'embedder_config': {'embedding_dim': self.embedder.embedding_dim},
            'positions': [emb.__dict__ for emb in self.index.values()]
        }

        with open(filepath, 'wb') as f:
            pickle.dump(index_data, f)

        logger.info(f"💾 Saved position index with {len(self.index)} positions to {filepath}")

    def load_index(self, filepath: Path) -> bool:
        """Load the position index from disk."""
        try:
            with open(filepath, 'rb') as f:
                index_data = pickle.load(f)

            # Recreate embedder
            embedder_config = index_data.get('embedder_config', {})
            self.embedder = ChessPositionEmbedder(**embedder_config)

            # Load positions
            self.index.clear()
            self.position_hashes.clear()

            for pos_data in index_data.get('positions', []):
                embedding_array = np.array(pos_data['embedding'])
                pos_data['embedding'] = embedding_array
                embedding = PositionEmbedding(**pos_data)
                self.index[embedding.position_hash] = embedding
                self.position_hashes.add(embedding.position_hash)

            logger.info(f"📂 Loaded position index with {len(self.index)} positions from {filepath}")
            return True

        except Exception as e:
            logger.error(f"Failed to load index from {filepath}: {e}")
            return False

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the position index."""
        if not self.index:
            return {'total_positions': 0}

        phases = defaultdict(int)
        material_balances = []

        for embedding in self.index.values():
            metadata = embedding.metadata
            phases[metadata.get('phase', 'unknown')] += 1
            material_balances.append(metadata.get('material_balance', 0))

        return {
            'total_positions': len(self.index),
            'phase_distribution': dict(phases),
            'avg_material_balance': np.mean(material_balances) if material_balances else 0,
            'material_balance_std': np.std(material_balances) if material_balances else 0
        }


def create_enhanced_response_with_context(inference_response: str, similar_positions: List[SimilarityResult]) -> str:
    """Enhance a response with context from similar positions."""
    if not similar_positions:
        return inference_response

    # Add context about similar positions
    context_lines = ["\n**Similar Positions Found:**"]

    for i, result in enumerate(similar_positions[:3]):  # Top 3
        context_lines.append(f"{i+1}. Position with {result.similarity_score:.2f} similarity")
        if result.common_features:
            context_lines.append(f"   Common features: {', '.join(result.common_features)}")

    enhanced_response = inference_response + "\n\n".join(context_lines)
    return enhanced_response


# Example usage and testing
if __name__ == "__main__":
    # Initialize embedder and retriever
    embedder = ChessPositionEmbedder(embedding_dim=256)
    retriever = ChessPositionRetriever(embedder)

    # Add some example positions
    example_positions = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",  # Starting position
        "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",  # After e4 e5
        "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",  # After Nf3 Nc6
        "r1bqkb1r/pppp1ppp/2n2n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 4 4",  # After Nf6
    ]

    print("Adding example positions to index...")
    for fen in example_positions:
        retriever.add_position(fen)

    # Test similarity search
    query_fen = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"
    print(f"\nSearching for positions similar to: {query_fen}")

    similar = retriever.find_similar_positions(query_fen, top_k=3)
    for result in similar:
        print(f"Similarity: {result.similarity_score:.3f}")
        print(f"FEN: {result.fen}")
        print(f"Common features: {result.common_features}")
        print()

    print("Chess embedding system test completed!")
