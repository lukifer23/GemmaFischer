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
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
import logging
from pathlib import Path
import json
from datetime import datetime
from dataclasses import dataclass, field
import hashlib
from functools import lru_cache
from collections import OrderedDict
import threading

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Thread-local storage for caching
_thread_local = threading.local()


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

    def __init__(self, num_experts: int = 3, feature_dim: int = 256, hidden_dim: int = 128):
        super().__init__()
        self.num_experts = num_experts
        self.feature_dim = feature_dim

        # Expert mapping
        self.expert_names = ['uci', 'tutor', 'director']

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
            nn.Linear(hidden_dim, num_experts)
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

        white_safety = self._calculate_king_safety_score(board, king_positions['white'])
        black_safety = self._calculate_king_safety_score(board, king_positions['black'])

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

    def _calculate_king_safety_score(self, board: List[List[str]], king_pos: Tuple[int, int]) -> float:
        """Calculate safety score for a king position."""
        if not king_pos:
            return 0.5

        i, j = king_pos
        defenders = 0
        attackers = 0

        # Check adjacent squares for defenders/attackers
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue

                ni, nj = i + di, j + dj
                if 0 <= ni < 8 and 0 <= nj < 8:
                    piece = board[ni][nj]
                    if piece:
                        if piece.isupper():  # White piece near white king
                            defenders += 1
                        else:  # Black piece near white king
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
        weights = torch.tensor([self.expert_performance[name]['accuracy']
                               for name in self.expert_names], dtype=torch.float32)

        # Weight the probabilities by expert performance
        weighted_probs = gate_probs * weights
        return weighted_probs / weighted_probs.sum()

    def _make_routing_decision(self, probs: torch.Tensor, confidence: float,
                              fen: str, query_type: str) -> RoutingDecision:
        """Make the final routing decision."""

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


class MoEInferenceManager:
    """Manages MoE inference with automatic routing and ensemble capabilities."""

    def __init__(self, router: ChessMoERouter, expert_models: Dict[str, Any], inference_system=None):
        self.router = router
        self.expert_models = expert_models
        self.inference_system = inference_system
        self.metrics = MoERoutingMetrics()

        # Load expert models
        for expert_name, model_path in expert_models.items():
            if Path(model_path).exists():
                logger.info(f"Loading {expert_name} expert from {model_path}")
                # Load logic would go here
            else:
                logger.warning(f"Expert model not found: {model_path}")

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

                # Get response from the actual inference system
                result = self.inference_system.generate_response(
                    question,
                    context=f"Current position: {fen}",
                    mode=expert_name
                )

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

