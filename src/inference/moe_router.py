#!/usr/bin/env python3
"""
Mixture of Experts Router for ChessGemma

True MoE implementation with automatic expert routing:
- Dynamic expert selection based on input characteristics
- Ensemble capabilities for complex queries
- Performance-aware routing
- Adaptive routing based on confidence scores

Features:
- Automatic gating mechanism using position analysis
- Multi-expert ensemble for comprehensive analysis
- Confidence-based expert selection
- Performance monitoring and optimization
- Fallback mechanisms for robustness
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple, Union, Set
import numpy as np
import logging
from pathlib import Path
import json
import re
from datetime import datetime
from dataclasses import dataclass, field
import hashlib
from functools import lru_cache
from collections import OrderedDict
import threading
import random
from torch.utils.data import Dataset, DataLoader

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Thread-local storage for caching
_thread_local = threading.local()

@dataclass
class RouterTrainingExample:
    """Training example for MoE router."""
    question: str
    question_embedding: np.ndarray
    expected_expert: str
    fen: Optional[str] = None
    category: str = "general"

class RouterTrainingDataset(Dataset):
    """Dataset for training the MoE router."""

    def __init__(self, examples: List[RouterTrainingExample]):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]

        # Convert expert name to index
        expert_to_idx = {"uci": 0, "tutor": 1, "director": 2}
        target_idx = expert_to_idx.get(example.expected_expert, 1)  # Default to tutor

        return {
            "embedding": torch.tensor(example.question_embedding, dtype=torch.float32),
            "target": torch.tensor(target_idx, dtype=torch.long),
            "expert_name": example.expected_expert
        }


@dataclass
class RoutingDecision:
    """Decision made by the MoE router."""
    primary_expert: str
    expert_weights: Dict[str, float]
    confidence_score: float
    reasoning: str
    ensemble_mode: bool = False
    fallback_used: bool = False


@dataclass
class MoERoutingMetrics:
    """Metrics for MoE routing performance."""
    total_requests: int = 0
    routing_accuracy: float = 0.0
    average_confidence: float = 0.0
    ensemble_usage_rate: float = 0.0
    fallback_rate: float = 0.0
    expert_usage_stats: Dict[str, int] = field(default_factory=dict)


class ChessMoERouter(nn.Module):
    """Optimized Mixture of Experts Router for Chess Analysis.

    Automatically routes chess queries to the most appropriate expert(s)
    based on position characteristics and query requirements.
    Features advanced caching and performance optimizations.
    """

    def __init__(
        self,
        num_experts: int = 3,
        feature_dim: Optional[int] = None,
        hidden_dim: int = 128,
        expert_names: Optional[List[str]] = None,
    ):
        super().__init__()
        default_names = ['uci', 'tutor', 'director']

        if expert_names:
            self.expert_names = expert_names
        else:
            if num_experts <= len(default_names):
                self.expert_names = default_names[:num_experts]
            else:
                extra = [f"expert_{i}" for i in range(len(default_names), num_experts)]
                self.expert_names = default_names + extra

        self.num_experts = len(self.expert_names)
        if self.num_experts == 0:
            raise ValueError("ChessMoERouter requires at least one expert to operate.")

        if feature_dim is None:
            feature_dim = self._determine_feature_dim()

        self.feature_dim = feature_dim

        # Feature extraction layers
        self.position_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Gating network
        self.gate_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, self.num_experts)
        )

        # Confidence estimation
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

        # Expert performance tracking
        self.expert_performance = {name: {'accuracy': 0.7, 'speed': 0.8, 'quality': 0.75}
                                  for name in self.expert_names}

        # Performance optimization caches
        self._position_cache = OrderedDict()  # LRU cache for position features
        self._routing_cache = OrderedDict()   # LRU cache for routing decisions
        self._cache_max_size = 1000
        self._feature_cache_hits = 0
        self._routing_cache_hits = 0
        self._total_requests = 0

        logger.info(f"🧠 Optimized MoE Router initialized with {num_experts} experts")

        # The router is used purely for inference; ensure dropout layers are disabled
        self.eval()

    def _determine_feature_dim(self) -> int:
        """Return the fixed training embedding dimensionality."""
        # Fixed feature dimension: 20 chess terms + 4 additional features
        return 24

    def train(self, mode: bool = False):
        """Override to keep the router in evaluation mode.

        Dropout in the gating networks would introduce nondeterminism during
        routing.  Regardless of the requested mode, the router remains in
        evaluation mode.
        """
        if mode:
            logger.warning(
                "ChessMoERouter.train(True) called, but router stays in eval mode"
            )
        return super().train(False)

    def forward(self, position_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the gating network."""
        # Extract position features
        features = self.position_encoder(position_features)

        # Generate gating logits
        gate_logits = self.gate_network(features)

        # Generate confidence score
        confidence = self.confidence_head(features)

        return gate_logits, confidence

    def route_query(self, position_fen: str, query_type: str = "auto",
                   complexity_score: Optional[float] = None) -> RoutingDecision:
        """Optimized routing with advanced caching and performance improvements."""

        self._total_requests += 1

        # Ensure evaluation mode during routing to keep dropout disabled
        self.eval()

        # Create cache key for this query
        cache_key = self._create_cache_key(position_fen, query_type, complexity_score)

        # Check routing cache first
        if cache_key in self._routing_cache:
            self._routing_cache_hits += 1
            cached_decision = self._routing_cache[cache_key]
            # Update LRU order
            self._routing_cache.move_to_end(cache_key)
            logger.info(f"🎯 Cached routing: {cached_decision.primary_expert} (confidence: {cached_decision.confidence_score:.3f})")
            return cached_decision

        # Extract features from position (with caching)
        position_features = self._extract_position_features_cached(position_fen, query_type)

        # Get routing decision
        with torch.no_grad():
            gate_logits, confidence = self.forward(position_features.unsqueeze(0))
            gate_probs = F.softmax(gate_logits, dim=-1).squeeze()

        # Apply expert performance weighting
        weighted_probs = self._apply_performance_weighting(gate_probs)

        # Make routing decision
        decision = self._make_routing_decision(weighted_probs, confidence.item(), position_fen, query_type)

        # Cache the decision
        self._cache_routing_decision(cache_key, decision)

        logger.info(f"🎯 Computed routing: {decision.primary_expert} (confidence: {decision.confidence_score:.3f})")
        return decision

    def _extract_position_features(self, fen: str, query_type: str) -> torch.Tensor:
        """Extract features from chess position for routing."""
        features = []

        # Basic position features
        board = self._fen_to_board(fen)

        # Material balance (-1 to 1)
        material_balance = self._calculate_material_balance(board)
        features.append(material_balance)

        # King safety (0 to 1)
        king_safety = self._assess_king_safety(board)
        features.append(king_safety)

        # Piece activity (0 to 1)
        piece_activity = self._calculate_piece_activity(board)
        features.append(piece_activity)

        # Pawn structure complexity (0 to 1)
        pawn_complexity = self._assess_pawn_structure(board)
        features.append(pawn_complexity)

        # Tactical opportunities (0 to 1)
        tactical_potential = self._detect_tactical_motifs(board)
        features.append(tactical_potential)

        # Query type encoding
        query_features = self._encode_query_type(query_type)
        features.extend(query_features)

        # Convert to tensor and pad to feature_dim
        feature_tensor = torch.tensor(features, dtype=torch.float32)
        if len(feature_tensor) < self.feature_dim:
            padding = torch.zeros(self.feature_dim - len(feature_tensor))
            feature_tensor = torch.cat([feature_tensor, padding])

        return feature_tensor[:self.feature_dim]

    def _create_cache_key(self, fen: str, query_type: str, complexity_score: Optional[float]) -> str:
        """Create a unique cache key for position and query combination."""
        key_components = [fen, query_type]
        if complexity_score is not None:
            key_components.append(f"{complexity_score:.3f}")
        key_string = "|".join(key_components)
        return hashlib.md5(key_string.encode()).hexdigest()

    def _extract_position_features_cached(self, fen: str, query_type: str) -> torch.Tensor:
        """Extract position features with caching for improved performance."""
        cache_key = f"features_{fen}_{query_type}"

        # Check cache first
        if cache_key in self._position_cache:
            self._feature_cache_hits += 1
            cached_features = self._position_cache[cache_key]
            # Update LRU order
            self._position_cache.move_to_end(cache_key)
            return cached_features

        # Compute features
        features = self._extract_position_features(fen, query_type)

        # Cache the result
        self._position_cache[cache_key] = features.clone()
        self._maintain_cache_size()

        return features

    def _cache_routing_decision(self, cache_key: str, decision: RoutingDecision):
        """Cache routing decision for future use."""
        self._routing_cache[cache_key] = decision
        self._maintain_cache_size()

    def _maintain_cache_size(self):
        """Maintain cache size limits using LRU eviction."""
        # Maintain position feature cache
        while len(self._position_cache) > self._cache_max_size:
            self._position_cache.popitem(last=False)

        # Maintain routing decision cache
        while len(self._routing_cache) > self._cache_max_size:
            self._routing_cache.popitem(last=False)

    def clear_caches(self):
        """Clear all caches."""
        self._position_cache.clear()
        self._routing_cache.clear()
        self._feature_cache_hits = 0
        self._routing_cache_hits = 0
        logger.info("🧹 MoE Router caches cleared")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache performance statistics."""
        total_requests = max(self._total_requests, 1)  # Avoid division by zero

        return {
            'position_cache_size': len(self._position_cache),
            'routing_cache_size': len(self._routing_cache),
            'cache_max_size': self._cache_max_size,
            'feature_cache_hit_rate': self._feature_cache_hits / total_requests,
            'routing_cache_hit_rate': self._routing_cache_hits / total_requests,
            'total_requests': self._total_requests,
            'cache_memory_usage_mb': self._estimate_cache_memory_usage()
        }

    def _estimate_cache_memory_usage(self) -> float:
        """Estimate memory usage of caches in MB."""
        # Rough estimation: each cached tensor/feature is ~1KB
        cache_entries = len(self._position_cache) + len(self._routing_cache)
        return cache_entries * 1024 / (1024 * 1024)  # Convert to MB

    def _fen_to_board(self, fen: str) -> List[List[str]]:
        """Convert FEN to board representation."""
        board = []
        rows = fen.split()[0].split('/')

        for row in rows:
            board_row = []
            for char in row:
                if char.isdigit():
                    board_row.extend([''] * int(char))
                else:
                    board_row.append(char)
            board.append(board_row)

        return board

    def _calculate_material_balance(self, board: List[List[str]]) -> float:
        """Calculate material balance (-1 to 1, negative favors black)."""
        piece_values = {'p': 1, 'n': 3, 'b': 3, 'r': 5, 'q': 9, 'k': 0}
        white_material = 0
        black_material = 0

        for row in board:
            for piece in row:
                if piece and piece != '':
                    if piece.isupper():
                        white_material += piece_values.get(piece.lower(), 0)
                    else:
                        black_material += piece_values.get(piece, 0)

        total_material = white_material + black_material
        if total_material == 0:
            return 0.0

        return (white_material - black_material) / total_material

    def _assess_king_safety(self, board: List[List[str]]) -> float:
        """Assess king safety (0=exposed, 1=safe)."""
        # Simplified king safety assessment
        king_positions = self._find_kings(board)

        white_safety = self._calculate_king_safety_score(
            board, king_positions['white'], 'white'
        )
        black_safety = self._calculate_king_safety_score(
            board, king_positions['black'], 'black'
        )

        return (white_safety + black_safety) / 2

    def _find_kings(self, board: List[List[str]]) -> Dict[str, Tuple[int, int]]:
        """Find positions of both kings."""
        positions = {'white': None, 'black': None}

        for i, row in enumerate(board):
            for j, piece in enumerate(row):
                if piece == 'K':
                    positions['white'] = (i, j)
                elif piece == 'k':
                    positions['black'] = (i, j)

        return positions

    def _calculate_king_safety_score(
        self,
        board: List[List[str]],
        king_pos: Tuple[int, int],
        king_color: Optional[str] = None
    ) -> float:
        """Calculate safety score for a king position."""
        if not king_pos:
            return 0.5

        i, j = king_pos
        defenders = 0
        attackers = 0

        # Determine king color if not provided
        square_piece = board[i][j] if 0 <= i < 8 and 0 <= j < 8 else ''
        is_white = None
        if king_color is not None:
            if king_color in ('K', 'k'):
                is_white = king_color == 'K'
            else:
                lowered = king_color.lower()
                if lowered in ('white', 'w'):
                    is_white = True
                elif lowered in ('black', 'b'):
                    is_white = False

        if is_white is None and square_piece:
            is_white = square_piece.isupper()

        if is_white is None:
            return 0.5

        # Check adjacent squares for defenders/attackers
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue

                ni, nj = i + di, j + dj
                if 0 <= ni < 8 and 0 <= nj < 8:
                    piece = board[ni][nj]
                    if piece:
                        if is_white:
                            if piece.isupper():
                                defenders += 1
                            elif piece.islower():
                                attackers += 1
                        else:
                            if piece.islower():
                                defenders += 1
                            elif piece.isupper():
                                attackers += 1

        total_pieces = defenders + attackers
        return defenders / max(total_pieces, 1)

    def _calculate_piece_activity(self, board: List[List[str]]) -> float:
        """Calculate piece activity score."""
        active_squares = 0
        total_squares = 0

        for i, row in enumerate(board):
            for j, piece in enumerate(row):
                if piece and piece != '':
                    total_squares += 1
                    # Simplified activity: pieces not on edges are more active
                    if 1 <= i <= 6 and 1 <= j <= 6:
                        active_squares += 1

        return active_squares / max(total_squares, 1)

    def _assess_pawn_structure(self, board: List[List[str]]) -> float:
        """Assess pawn structure complexity."""
        pawn_positions = []

        for i, row in enumerate(board):
            for j, piece in enumerate(row):
                if piece.lower() == 'p':
                    pawn_positions.append((i, j))

        # Calculate pawn structure complexity based on isolation, backwardness, etc.
        isolated_pawns = 0
        for pawn in pawn_positions:
            i, j = pawn
            has_neighbor = False
            for dj in [-1, 1]:
                nj = j + dj
                if 0 <= nj < 8 and any(board[ni][nj].lower() == 'p' for ni in range(8)):
                    has_neighbor = True
                    break
            if not has_neighbor:
                isolated_pawns += 1

        return min(isolated_pawns / max(len(pawn_positions), 1), 1.0)

    def _detect_tactical_motifs(self, board: List[List[str]]) -> float:
        """Detect tactical opportunities."""
        # Simplified tactical detection
        motifs = 0

        # Check for pins, forks, etc. (simplified)
        for i, row in enumerate(board):
            for j, piece in enumerate(row):
                if piece and piece != '':
                    # Check for potential attacks
                    if self._has_attackers(board, i, j):
                        motifs += 1

        return min(motifs / 16, 1.0)  # Normalize by board size

    def _has_attackers(self, board: List[List[str]], i: int, j: int) -> bool:
        """Check if a square has attackers."""
        # Simplified attack detection
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                ni, nj = i + di, j + dj
                if 0 <= ni < 8 and 0 <= nj < 8:
                    attacker = board[ni][nj]
                    if attacker and attacker != '':
                        return True
        return False

    def _encode_query_type(self, query_type: str) -> List[float]:
        """Encode query type into features."""
        if query_type == "engine" or query_type == "uci":
            return [1.0, 0.0, 0.0]  # Move-focused
        elif query_type == "tutor" or query_type == "explain":
            return [0.0, 1.0, 0.0]  # Analysis-focused
        elif query_type == "director" or query_type == "strategy":
            return [0.0, 0.0, 1.0]  # Strategic-focused
        else:
            return [0.33, 0.33, 0.34]  # Balanced

    def _apply_performance_weighting(self, gate_probs: torch.Tensor) -> torch.Tensor:
        """Apply expert performance weighting to routing probabilities."""
        weights = torch.tensor(
            [self.expert_performance.get(name, {}).get('accuracy', 1.0)
             for name in self.expert_names],
            dtype=torch.float32
        )

        # Weight the probabilities by expert performance
        weighted_probs = gate_probs * weights
        total = weighted_probs.sum()
        if total.item() == 0:
            return torch.full_like(weighted_probs, 1.0 / self.num_experts)
        return weighted_probs / total

    def _make_routing_decision(self, probs: torch.Tensor, confidence: float,
                              fen: str, query_type: str) -> RoutingDecision:
        """Make the final routing decision."""

        if not self.expert_names:
            raise RuntimeError("MoE router has no experts configured")

        # Get expert probabilities
        expert_probs = {name: prob.item() for name, prob in zip(self.expert_names, probs)}

        # Determine primary expert
        primary_expert = max(expert_probs.keys(), key=lambda x: expert_probs[x])

        # Check if ensemble mode is beneficial
        ensemble_mode = self._should_use_ensemble(expert_probs, confidence, query_type)

        # Adjust weights for ensemble mode
        if ensemble_mode:
            # Keep top 2 experts with significant weights
            sorted_experts = sorted(expert_probs.items(), key=lambda x: x[1], reverse=True)
            expert_weights = {name: prob for name, prob in sorted_experts[:2]}
            # Normalize weights
            total_weight = sum(expert_weights.values())
            expert_weights = {name: weight/total_weight for name, weight in expert_weights.items()}
        else:
            expert_weights = {primary_expert: 1.0}

        # Generate reasoning
        reasoning = self._generate_routing_reasoning(primary_expert, expert_probs, confidence, query_type)

        return RoutingDecision(
            primary_expert=primary_expert,
            expert_weights=expert_weights,
            confidence_score=confidence,
            reasoning=reasoning,
            ensemble_mode=ensemble_mode
        )

    def _should_use_ensemble(self, expert_probs: Dict[str, float],
                           confidence: float, query_type: str) -> bool:
        """Determine if ensemble mode should be used."""
        # Use ensemble for complex queries with low confidence
        max_prob = max(expert_probs.values())
        second_max_prob = sorted(expert_probs.values(), reverse=True)[1]

        # Ensemble conditions:
        # 1. Low confidence in primary expert
        # 2. Close competition between top 2 experts
        # 3. Complex query types
        ensemble_conditions = [
            confidence < 0.7,  # Low confidence
            max_prob - second_max_prob < 0.2,  # Close competition
            query_type in ['complex', 'analysis', 'strategy']  # Complex queries
        ]

        return any(ensemble_conditions)

    def _generate_routing_reasoning(self, primary_expert: str,
                                   expert_probs: Dict[str, float],
                                   confidence: float, query_type: str) -> str:
        """Generate human-readable reasoning for routing decision."""
        reasons = []

        if confidence > 0.8:
            reasons.append(f"High confidence ({confidence:.2f}) in {primary_expert} expert")
        elif confidence > 0.6:
            reasons.append(f"Moderate confidence ({confidence:.2f}) in {primary_expert} expert")
        else:
            reasons.append(f"Low confidence ({confidence:.2f}), using ensemble mode")

        # Add query-specific reasoning
        if query_type == "engine" or query_type == "uci":
            reasons.append("Query focuses on move generation")
        elif query_type == "tutor":
            reasons.append("Query requires detailed analysis")
        elif query_type == "director":
            reasons.append("Query involves strategic planning")

        return "; ".join(reasons)

    def update_expert_performance(self, expert_name: str, performance: Dict[str, float]):
        """Update expert performance metrics for better routing."""
        if expert_name in self.expert_performance:
            # Exponential moving average for smooth updates
            alpha = 0.1
            for metric in ['accuracy', 'speed', 'quality']:
                if metric in performance:
                    old_value = self.expert_performance[expert_name][metric]
                    new_value = performance[metric]
                    self.expert_performance[expert_name][metric] = (
                        alpha * new_value + (1 - alpha) * old_value
                    )

        logger.info(f"📊 Updated performance for {expert_name}: {self.expert_performance[expert_name]}")

    def get_routing_stats(self) -> Dict[str, Any]:
        """Get comprehensive routing statistics including cache performance."""
        cache_stats = self.get_cache_stats()

        return {
            'expert_performance': self.expert_performance,
            'routing_parameters': {
                'num_experts': self.num_experts,
                'feature_dim': self.feature_dim,
                'expert_names': self.expert_names
            },
            'model_info': {
                'total_parameters': sum(p.numel() for p in self.parameters()),
                'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad)
            },
            'cache_performance': cache_stats,
            'performance_metrics': {
                'cache_hit_rate': (cache_stats['feature_cache_hit_rate'] + cache_stats['routing_cache_hit_rate']) / 2,
                'cache_memory_efficiency': cache_stats['cache_memory_usage_mb'],
                'routing_speedup': 1.0 / (1.0 - cache_stats['routing_cache_hit_rate']) if cache_stats['routing_cache_hit_rate'] < 1.0 else 10.0
            }
        }

    def save_router(self, path: str):
        """Save the router model and configuration."""
        save_dict = {
            'model_state_dict': self.state_dict(),
            'expert_performance': self.expert_performance,
            'config': {
                'num_experts': self.num_experts,
                'feature_dim': self.feature_dim,
                'expert_names': self.expert_names
            },
            'timestamp': datetime.now().isoformat()
        }

        torch.save(save_dict, path)
        logger.info(f"💾 Router saved to {path}")

    def load_router(self, path: str):
        """Load the router model and configuration."""
        save_dict = torch.load(path, map_location='cpu')

        self.load_state_dict(save_dict['model_state_dict'])
        self.expert_performance = save_dict.get('expert_performance', self.expert_performance)

        logger.info(f"📂 Router loaded from {path}")

    def prepare_training_data(self, evaluation_queries: List[Dict[str, Any]],
                            inference_system) -> List[RouterTrainingExample]:
        """Prepare training data from evaluation queries and expert responses."""
        training_examples = []

        print("🎯 Preparing MoE router training data...")

        for i, query in enumerate(evaluation_queries):
            if i % 50 == 0:
                print(f"   Processing query {i+1}/{len(evaluation_queries)}")

            question = query["question"]
            expected_expert = query["expert"]  # Use "expert" key from eval suite
            category = query.get("category", "general")

            # Extract FEN if present
            fen_match = re.search(r'FEN:\s*([^\s\n]+)', question)
            fen = fen_match.group(1) if fen_match else None

            # Generate question embedding (simplified - use basic features)
            question_embedding = self._embed_question_for_training(question)

            training_examples.append(RouterTrainingExample(
                question=question,
                question_embedding=question_embedding,
                expected_expert=expected_expert,
                fen=fen,
                category=category
            ))

        print(f"✅ Prepared {len(training_examples)} training examples")
        return training_examples

    def _embed_question_for_training(self, question: str) -> np.ndarray:
        """Create a simple embedding for training (can be replaced with better embedding)."""
        # Simple bag-of-words style features for chess terms
        chess_terms = {
            'fen': 1.0, 'position': 0.8, 'move': 0.9, 'best': 0.7,
            'analyze': 0.9, 'tactics': 0.8, 'strategy': 0.8,
            'opening': 0.6, 'endgame': 0.6, 'middlegame': 0.6,
            'pawn': 0.4, 'rook': 0.4, 'knight': 0.4, 'bishop': 0.4, 'queen': 0.4, 'king': 0.4,
            'check': 0.5, 'mate': 0.5, 'castl': 0.5, 'attack': 0.6, 'defense': 0.6
        }

        # Create feature vector
        features = []
        question_lower = question.lower()

        for term, weight in chess_terms.items():
            if term in question_lower:
                features.append(weight)
            else:
                features.append(0.0)

        # Add question length features
        features.extend([
            len(question) / 500.0,  # Normalized length
            len(re.findall(r'\b\w+\b', question)) / 50.0,  # Word count
            1.0 if '?' in question else 0.0,  # Question mark
            1.0 if 'FEN:' in question.upper() else 0.0,  # Has FEN
        ])

        return np.array(features, dtype=np.float32)

    def train_router(self, training_examples: List[RouterTrainingExample],
                    num_epochs: int = 10, batch_size: int = 32, learning_rate: float = 1e-3):
        """Train the MoE router on routing decisions."""

        print("🚀 Starting MoE router training...")
        print(f"   Examples: {len(training_examples)}")
        print(f"   Epochs: {num_epochs}")
        print(f"   Batch size: {batch_size}")
        print(f"   Learning rate: {learning_rate}")

        # Create dataset and dataloader
        dataset = RouterTrainingDataset(training_examples)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Set up optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()

        # Training loop
        self.train()
        best_accuracy = 0.0

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0

            for batch in dataloader:
                embeddings = batch["embedding"]
                targets = batch["target"]

                # Forward pass
                self.optimizer.zero_grad()
                gate_logits, _ = self.forward(embeddings)
                loss = self.criterion(gate_logits, targets)

                # Backward pass
                loss.backward()
                self.optimizer.step()

                # Statistics
                epoch_loss += loss.item()
                _, predicted = torch.max(gate_logits.data, 1)
                epoch_correct += (predicted == targets).sum().item()
                epoch_total += targets.size(0)

            # Calculate epoch metrics
            epoch_accuracy = epoch_correct / epoch_total
            epoch_avg_loss = epoch_loss / len(dataloader)

            print(f"Epoch {epoch+1:2d}: Loss={epoch_avg_loss:.4f}, Accuracy={epoch_accuracy:.1%}")
            # Save best model
            if epoch_accuracy > best_accuracy:
                best_accuracy = epoch_accuracy
                self.save_router("checkpoints/moe_router/best_checkpoint.pth")

        print(f"Best accuracy achieved: {best_accuracy:.1f}")
        # Switch back to eval mode
        self.eval()

        return best_accuracy

    def evaluate_routing_accuracy(self, test_examples: List[RouterTrainingExample]) -> float:
        """Evaluate routing accuracy on test data."""
        self.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for example in test_examples:
                embedding = torch.tensor(example.question_embedding, dtype=torch.float32).unsqueeze(0)
                gate_logits, _ = self.forward(embedding)
                _, predicted = torch.max(gate_logits, 1)

                expert_idx_to_name = {0: "uci", 1: "tutor", 2: "director"}
                predicted_expert = expert_idx_to_name[predicted.item()]

                if predicted_expert == example.expected_expert:
                    correct += 1
                total += 1

        return correct / total if total > 0 else 0.0


class MoEInferenceManager:
    """Manages MoE inference with automatic routing and ensemble capabilities."""

    def __init__(self, router: ChessMoERouter, expert_models: Dict[str, Any], inference_system=None):
        self.router = router
        self.expert_models = expert_models
        self.inference_system = inference_system
        self.metrics = MoERoutingMetrics()
        self._expert_adapter_paths: Dict[str, Path] = {}
        self._expert_ready: Dict[str, bool] = {}
        self._missing_expert_logged: Set[str] = set()
        self._priming_lock = threading.Lock()
        self._deferred_prime_logged = False

        if self.inference_system and hasattr(self.inference_system, "refresh_adapters"):
            try:
                self.inference_system.refresh_adapters()
            except Exception as refresh_err:
                logger.warning(
                    "Unable to refresh adapters during MoE initialization: %s",
                    error=refresh_err,
                )

        for expert_name, model_path in expert_models.items():
            path_obj = Path(model_path) if model_path else None
            if path_obj and path_obj.exists():
                self._expert_adapter_paths[expert_name] = path_obj
                self._expert_ready[expert_name] = False
                logger.info("Registered %s expert adapter at %s", expert_name, path_obj)
            else:
                logger.warning(
                    "Expert model not found for %s: %s", expert_name=expert_name, model_path=model_path
                )

        self.prime_available_experts()

    def prime_available_experts(self) -> None:
        """Ensure all known experts are loaded into the shared inference model."""
        if not self.inference_system or not hasattr(
            self.inference_system, "set_active_adapter"
        ):
            logger.debug("No inference system available to prime MoE experts")
            return

        if not getattr(self.inference_system, "is_loaded", False):
            if not self._deferred_prime_logged:
                logger.info(
                    "MoE expert priming deferred until base model weights are loaded"
                )
                self._deferred_prime_logged = True
            return

        with self._priming_lock:
            self._deferred_prime_logged = False
            for expert_name in list(self._expert_adapter_paths.keys()):
                self._prime_single_expert(expert_name)

    def _prime_single_expert(self, expert_name: str) -> None:
        if self._expert_ready.get(expert_name):
            return

        adapter_path = self._expert_adapter_paths.get(expert_name)
        if adapter_path is None:
            return

        active_adapter = getattr(self.inference_system, "_active_adapter", None)
        try:
            self.inference_system.set_active_adapter(expert_name)
            self._expert_ready[expert_name] = True
            logger.info(
                "Primed %s expert using adapter at %s", expert_name, adapter_path
            )
        except Exception as exc:
            self._expert_ready[expert_name] = False
            logger.error(
                "Failed to prime %s expert from %s: %s",
                expert_name,
                adapter_path,
                exc,
            )
        finally:
            if active_adapter and active_adapter != expert_name:
                try:
                    self.inference_system.set_active_adapter(active_adapter)
                except Exception:
                    logger.debug(
                        "Unable to restore previous adapter %s after priming %s",
                        active_adapter,
                        expert_name,
                    )

    def _ensure_expert_ready(self, expert_name: str) -> None:
        if expert_name not in self._expert_adapter_paths:
            return

        if not self._expert_ready.get(expert_name):
            self.prime_available_experts()
            if (
                not self._expert_ready.get(expert_name)
                and expert_name not in self._missing_expert_logged
            ):
                logger.warning(
                    "Expert %s is not primed; using fallback behaviour", expert_name=expert_name
                )
                self._missing_expert_logged.add(expert_name)

    def analyze_position(self, fen: str, query_type: str = "auto",
                        complexity_score: Optional[float] = None) -> Dict[str, Any]:
        """Analyze a chess position using optimized MoE routing with caching."""

        # Get routing decision (with caching)
        routing_decision = self.router.route_query(fen, query_type, complexity_score)

        # Execute routing decision
        if routing_decision.ensemble_mode:
            response = self._execute_ensemble_inference(fen, routing_decision)
        else:
            response = self._execute_single_expert_inference(fen, routing_decision.primary_expert)

        # Update metrics
        self._update_metrics(routing_decision)

        # Add routing metadata
        response['routing_info'] = {
            'primary_expert': routing_decision.primary_expert,
            'expert_weights': routing_decision.expert_weights,
            'confidence_score': routing_decision.confidence_score,
            'reasoning': routing_decision.reasoning,
            'ensemble_mode': routing_decision.ensemble_mode
        }

        return response

    def _execute_single_expert_inference(self, fen: str, expert_name: str) -> Dict[str, Any]:
        """Execute inference with a single expert."""
        self._ensure_expert_ready(expert_name)
        if self.inference_system:
            try:
                # Switch to the correct expert
                self.inference_system.set_active_adapter(expert_name)

                # Generate question based on expert type
                if expert_name == 'uci':
                    question = f"FEN: {fen}\nGenerate the best move in UCI format (e.g., e2e4). Respond with only the move."
                elif expert_name == 'tutor':
                    question = f"FEN: {fen}\nExplain the position and suggest the best move."
                else:  # director
                    question = f"FEN: {fen}\nAnalyze this chess position strategically."

                inference_mode = 'engine' if expert_name == 'uci' else expert_name

                # Prevent recursive MoE dispatch when delegating back to inference
                depth_attr = '_moe_dispatch_depth'
                previous_depth = getattr(self.inference_system, depth_attr, None)
                if previous_depth is not None:
                    setattr(self.inference_system, depth_attr, previous_depth + 1)

                try:
                    # Get response from the actual inference system
                    result = self.inference_system.generate_response(
                        question,
                        context=f"Current position: {fen}",
                        mode=inference_mode
                    )
                finally:
                    if previous_depth is not None:
                        setattr(self.inference_system, depth_attr, previous_depth)

                return {
                    'response': result.get('response', f'Analysis from {expert_name} expert'),
                    'expert_used': expert_name,
                    'analysis_type': 'single_expert',
                    'confidence': result.get('confidence', 0.5)
                }
            except Exception as e:
                logger.error(f"Error in single expert inference: {e}")

        # Fallback placeholder response
        return {
            'response': f"Analysis from {expert_name} expert",
            'expert_used': expert_name,
            'analysis_type': 'single_expert'
        }

    def _execute_ensemble_inference(self, fen: str, routing_decision: RoutingDecision) -> Dict[str, Any]:
        """Execute inference with expert ensemble."""
        responses = []
        weights = []

        for expert_name, weight in routing_decision.expert_weights.items():
            expert_response = self._execute_single_expert_inference(fen, expert_name)
            responses.append(expert_response)
            weights.append(weight)

        # Combine responses (simplified ensemble logic)
        combined_response = self._combine_expert_responses(responses, weights)

        return {
            'response': combined_response,
            'ensemble_used': list(routing_decision.expert_weights.keys()),
            'weights': routing_decision.expert_weights,
            'analysis_type': 'ensemble'
        }

    def _combine_expert_responses(self, responses: List[Dict[str, Any]],
                                weights: List[float]) -> str:
        """Combine responses from multiple experts."""
        # Simplified response combination
        combined_parts = []

        for response, weight in zip(responses, weights):
            response_text = response.get('response', '')
            if weight > 0.3:  # Only include significant contributors
                combined_parts.append(f"[{weight:.2f}] {response_text}")

        return "Ensemble Analysis:\n" + "\n".join(combined_parts)

    def _update_metrics(self, decision: RoutingDecision):
        """Update routing metrics."""
        self.metrics.total_requests += 1

        # Update expert usage stats
        for expert_name in decision.expert_weights.keys():
            self.metrics.expert_usage_stats[expert_name] = (
                self.metrics.expert_usage_stats.get(expert_name, 0) + 1
            )

        # Update ensemble rate
        if decision.ensemble_mode:
            self.metrics.ensemble_usage_rate = (
                (self.metrics.ensemble_usage_rate * (self.metrics.total_requests - 1) + 1)
                / self.metrics.total_requests
            )

        # Update confidence
        self.metrics.average_confidence = (
            (self.metrics.average_confidence * (self.metrics.total_requests - 1) + decision.confidence_score)
            / self.metrics.total_requests
        )

    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report with cache metrics."""
        router_stats = self.router.get_routing_stats()

        return {
            'routing_metrics': {
                'total_requests': self.metrics.total_requests,
                'average_confidence': self.metrics.average_confidence,
                'ensemble_usage_rate': self.metrics.ensemble_usage_rate,
                'expert_usage_distribution': self.metrics.expert_usage_stats
            },
            'cache_performance': router_stats.get('cache_performance', {}),
            'performance_optimization': {
                'cache_hit_rate': router_stats.get('performance_metrics', {}).get('cache_hit_rate', 0.0),
                'routing_speedup': router_stats.get('performance_metrics', {}).get('routing_speedup', 1.0),
                'memory_efficiency': router_stats.get('performance_metrics', {}).get('cache_memory_efficiency', 0.0)
            },
            'router_stats': router_stats,
            'expert_models': list(self.expert_models.keys())
        }

    def optimize_performance(self):
        """Apply performance optimizations."""
        # Clear caches periodically for optimal memory usage
        cache_stats = self.router.get_cache_stats()
        if cache_stats['cache_memory_usage_mb'] > 50:  # Clear if cache > 50MB
            self.router.clear_caches()
            logger.info("🧹 Auto-cleared MoE caches for memory optimization")

        # Log performance metrics
        perf_report = self.get_performance_report()
        logger.info(f"⚡ MoE Performance: Cache Hit Rate: {perf_report['performance_optimization']['cache_hit_rate']:.1%}, "
                   f"Speedup: {perf_report['performance_optimization']['routing_speedup']:.1f}x")

    def clear_all_caches(self):
        """Clear all caches in the MoE system."""
        self.router.clear_caches()
        logger.info("🧹 All MoE system caches cleared")


def create_moe_system(expert_paths: Dict[str, str], inference_system=None) -> Tuple[ChessMoERouter, MoEInferenceManager]:
    """Create a complete MoE system with router and inference manager."""
    # Initialize router
    router = ChessMoERouter()

    # Initialize inference manager
    inference_manager = MoEInferenceManager(router, expert_paths, inference_system)

    logger.info("🎯 MoE System created successfully")
    return router, inference_manager


# Example usage and testing
if __name__ == "__main__":
    print("🧠 Chess Mixture of Experts System Demo")
    print("=" * 50)

    # Create router
    router = ChessMoERouter()

    # Example routing decisions
    test_positions = [
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "engine"),  # Opening
        ("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 1", "tutor"),  # Complex middlegame
        ("8/8/8/8/8/8/8/K7 w - - 0 1", "director")  # Endgame
    ]

    for fen, query_type in test_positions:
        print(f"\n📝 Testing position: {fen[:30]}...")
        decision = router.route_query(fen, query_type)
        print(f"   🎯 Primary Expert: {decision.primary_expert}")
        print(f"   🎚️  Confidence: {decision.confidence_score:.3f}")
        print(f"   🧠 Ensemble Mode: {decision.ensemble_mode}")
        print(f"   💭 Reasoning: {decision.reasoning}")

    print("\n✅ MoE Router Demo Complete!")
    print("🔧 To use in production:")
    print("   1. Train router on chess position data")
    print("   2. Integrate with expert models")
    print("   3. Use MoEInferenceManager for automatic routing")
