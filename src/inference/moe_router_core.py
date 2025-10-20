#!/usr/bin/env python3
"""Core MoE router implementation with neural network routing logic.

Extracted from the monolithic moe_router.py to improve maintainability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple, Union, Set
import numpy as np
import logging
import re
import hashlib
from functools import lru_cache
from collections import OrderedDict
import threading
import os
import chess

from .moe_types import (
    RoutingDecision,
    AdaptiveRoutingState,
    MoETrainingConfig,
    EXPERT_NAMES,
    EXPERT_TO_INDEX,
    QUERY_CATEGORIES
)

logger = logging.getLogger(__name__)


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

        # FEN pattern for position extraction
        self._fen_pattern = re.compile(r'FEN:\s*([^\s\n]+)')

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

        # Expert performance tracking (default baseline values)
        self.expert_performance = {name: {'accuracy': 0.7, 'speed': 0.8, 'quality': 0.75}
                                  for name in self.expert_names}

        # Adaptive routing state
        self.adaptive_state = AdaptiveRoutingState()

        # Initialize expert performance metrics
        for expert_name in self.expert_names:
            if expert_name not in self.adaptive_state.expert_metrics:
                from .moe_types import ExpertPerformanceMetrics
                self.adaptive_state.expert_metrics[expert_name] = ExpertPerformanceMetrics(
                    baseline_performance=self.expert_performance[expert_name]['quality']
                )

        # Performance optimization caches
        self._position_cache = OrderedDict()  # LRU cache for position features
        self._routing_cache = OrderedDict()   # LRU cache for routing decisions
        self._cache_max_size = 1000
        self._feature_cache_hits = 0
        self._routing_cache_hits = 0
        self._total_requests = 0

        # Decision logging for offline routing analysis
        self.log_decisions_enabled = os.environ.get("CHESSGEMMA_ROUTER_LOGGING", "1") not in ("0", "false", "False")
        default_log_path = "reports/moe/routing_decisions.jsonl"
        self.decision_log_path = os.environ.get("CHESSGEMMA_ROUTER_LOG", default_log_path)
        self._decision_log_lock = threading.Lock()
        if self.log_decisions_enabled:
            try:
                from pathlib import Path
                Path(self.decision_log_path).parent.mkdir(parents=True, exist_ok=True)
            except Exception as exc:
                logger.warning(f"Unable to create router log directory {Path(self.decision_log_path).parent}: {exc}")
                self.log_decisions_enabled = False

        logger.info(f"Optimized MoE Router initialized with {num_experts} experts")

        # The router is used purely for inference; ensure dropout layers are disabled
        self.eval()

    def _fen_to_board(self, fen: str) -> chess.Board:
        """Convert a FEN string to a Board with defensive fallback."""
        try:
            return chess.Board(fen)
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.warning("Invalid FEN provided to router: %s", exc)
            return chess.Board()

    def _find_kings(self, board: chess.Board) -> Dict[str, Optional[chess.Square]]:
        """Locate both kings on the board."""
        return {
            "white": board.king(chess.WHITE),
            "black": board.king(chess.BLACK),
        }

    def _calculate_king_safety_score(
        self,
        board: chess.Board,
        king_square: Optional[chess.Square],
        color: str,
    ) -> float:
        """Heuristic king-safety estimate used for deterministic routing rules."""
        if king_square is None:
            return 0.0

        friendly_color = chess.WHITE if color == "white" else chess.BLACK
        enemy_color = chess.BLACK if friendly_color == chess.WHITE else chess.WHITE

        king_zone = chess.SquareSet(chess.BB_KING_ATTACKS[king_square])

        friendly_guards = len(board.attackers(friendly_color, king_square))
        friendly_support = sum(
            1 for sq in king_zone if board.color_at(sq) == friendly_color
        )

        enemy_pressure = len(board.attackers(enemy_color, king_square))
        enemy_control = sum(
            1 for sq in king_zone if board.is_attacked_by(enemy_color, sq)
        )

        score = 0.5
        score += 0.12 * friendly_guards
        score += 0.08 * friendly_support
        score -= 0.18 * enemy_pressure
        score -= 0.08 * enemy_control

        if board.is_attacked_by(enemy_color, king_square):
            score -= 0.2
        if board.is_check():
            score -= 0.25

        return max(0.0, min(1.0, score))

    def _determine_feature_dim(self) -> int:
        """Return the fixed training embedding dimensionality."""
        # Fixed feature dimension: 16 position features + 16 question features
        return 32

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

    def _keyword_based_routing(self, query_text: str, fen_present: bool = False) -> Optional[str]:
        """Enhanced keyword-based routing as fallback for ML router."""
        query_lower = query_text.lower()

        # Tutor expert: Analysis and evaluation (highest priority for analysis questions)
        tutor_keywords = [
            "analyze", "analysis", "evaluate", "assessment", "position",
            "step by step", "examine", "review", "look at", "consider",
            "what tactical pattern", "tactical pattern"
        ]
        if any(keyword in query_lower for keyword in tutor_keywords):
            return "tutor"

        # FEN-aware bias:
        # If a FEN is present, prefer Tutor unless user explicitly asks for a single move-only output
        if fen_present:
            move_only_tokens = ["uci", "respond with only", "only the move", "just the move", "best move"]
            analysis_tokens = ["analyze", "analysis", "evaluate", "explain", "step by step", "why", "how"]
            if any(tok in query_lower for tok in move_only_tokens) and not any(tok in query_lower for tok in analysis_tokens):
                return "uci"
            return "tutor"

        # UCI expert: Pure move questions (high priority) when no FEN bias
        move_keywords = ["what is the best move", "what move", "best move", "what should", "play", "move"]
        if any(keyword in query_lower for keyword in move_keywords):
            if not any(word in query_lower for word in ["analyze", "analysis", "evaluate", "explain", "step by step", "why", "how"]):
                return "uci"

        # Director expert: Strategy, principles, rules, explanations (medium priority)
        director_keywords = [
            "strategy", "opening", "endgame", "principle", "rules", "explain",
            "tactical", "concept", "idea", "theory", "understanding",
            "how to", "what is", "why", "chess rules", "castling", "pawn structure",
            "defense", "attack", "main ideas", "key concepts", "fundamental",
            "basic", "important", "essential", "behind", "development"
        ]
        if any(keyword in query_lower for keyword in director_keywords):
            return "director"

        return None  # No clear keyword match

    def route_query(self, position_fen: str, query_type: str = "auto",
                   complexity_score: Optional[float] = None, question_text: str = "") -> RoutingDecision:
        """Optimized routing with advanced caching and performance improvements."""

        self._total_requests += 1

        # Ensure evaluation mode during routing to keep dropout disabled
        self.eval()

        # Create cache key for this query
        cache_key = self._create_cache_key(position_fen, query_type, complexity_score, question_text)

        # Check routing cache first
        if cache_key in self._routing_cache:
            self._routing_cache_hits += 1
            cached_decision = self._routing_cache[cache_key]
            # Update LRU order
            self._routing_cache.move_to_end(cache_key)
            logger.info(f"Cached routing: {cached_decision.primary_expert} (confidence: {cached_decision.confidence_score:.3f})")
            return cached_decision

        # Extract features from position (with caching)
        position_features = self._extract_position_features_cached(position_fen, question_text)

        # Get routing decision
        routing_decision = self._make_routing_decision(position_features, question_text, complexity_score)

        # Cache the decision
        self._routing_cache[cache_key] = routing_decision
        if len(self._routing_cache) > self._cache_max_size:
            self._routing_cache.popitem(last=False)  # Remove oldest

        # Log decision if enabled
        if self.log_decisions_enabled:
            self._log_routing_decision(routing_decision, position_fen, question_text)

        return routing_decision

    def _create_cache_key(self, position_fen: str, query_type: str, complexity_score: Optional[float], question_text: str) -> str:
        """Create a deterministic cache key for routing decisions."""
        # Normalize and hash the inputs for consistent caching
        normalized_fen = position_fen.strip() if position_fen else ""
        normalized_question = question_text.strip().lower() if question_text else ""

        # Create a hash of the key components
        key_components = f"{normalized_fen}|{query_type}|{complexity_score or 0:.2f}|{normalized_question}"
        return hashlib.md5(key_components.encode()).hexdigest()

    @lru_cache(maxsize=512)
    def _extract_position_features_cached(self, position_fen: str, question_text: str) -> torch.Tensor:
        """Cached position feature extraction with LRU cache."""
        self._feature_cache_hits += 1
        return self._extract_position_features(position_fen, question_text)

    def _extract_position_features(self, position_fen: str, question_text: str) -> torch.Tensor:
        """Extract features from chess position and question for routing."""
        # Initialize feature vector (32 dimensions as per training)
        features = np.zeros(32, dtype=np.float32)

        # Position-based features (first 16 dimensions)
        if position_fen:
            position_features = self._extract_fen_features(position_fen)
            features[:16] = position_features

        # Question-based features (last 16 dimensions)
        if question_text:
            question_features = self._extract_question_features(question_text)
            features[16:] = question_features

        return torch.tensor(features, dtype=torch.float32)

    def _extract_fen_features(self, fen: str) -> np.ndarray:
        """Extract numerical features from FEN string."""
        features = np.zeros(16, dtype=np.float32)

        try:
            # Basic position analysis
            parts = fen.split()
            if len(parts) >= 1:
                board = parts[0]

                # Count pieces by type
                piece_counts = {'p': 0, 'n': 0, 'b': 0, 'r': 0, 'q': 0, 'k': 0,
                              'P': 0, 'N': 0, 'B': 0, 'R': 0, 'Q': 0, 'K': 0}

                for char in board:
                    if char in piece_counts:
                        piece_counts[char] += 1

                # Material balance (first 12 features)
                features[0] = piece_counts['P'] - piece_counts['p']  # Pawn advantage
                features[1] = piece_counts['N'] - piece_counts['n']  # Knight advantage
                features[2] = piece_counts['B'] - piece_counts['b']  # Bishop advantage
                features[3] = piece_counts['R'] - piece_counts['r']  # Rook advantage
                features[4] = piece_counts['Q'] - piece_counts['q']  # Queen advantage

                # Total material
                white_material = (piece_counts['P'] + 3*piece_counts['N'] + 3*piece_counts['B'] +
                                5*piece_counts['R'] + 9*piece_counts['Q'])
                black_material = (piece_counts['p'] + 3*piece_counts['n'] + 3*piece_counts['b'] +
                                5*piece_counts['r'] + 9*piece_counts['q'])
                features[5] = (white_material - black_material) / 100.0  # Normalized material advantage

                # Game phase (based on piece count)
                total_pieces = sum(piece_counts.values())
                features[6] = total_pieces / 32.0  # Normalized piece count

                # Castling rights and move number
                if len(parts) >= 3:
                    features[7] = 1.0 if 'K' in parts[2] else 0.0  # White kingside castling
                    features[8] = 1.0 if 'Q' in parts[2] else 0.0  # White queenside castling
                    features[9] = 1.0 if 'k' in parts[2] else 0.0  # Black kingside castling
                    features[10] = 1.0 if 'q' in parts[2] else 0.0  # Black queenside castling

                if len(parts) >= 5:
                    try:
                        features[11] = int(parts[4]) / 100.0  # Half-move clock (normalized)
                        features[12] = int(parts[5]) / 200.0  # Full move number (normalized)
                    except (ValueError, IndexError):
                        pass

                # En passant target square
                if len(parts) >= 4 and parts[3] != '-':
                    features[13] = 1.0  # En passant possible
                else:
                    features[13] = 0.0

                # Side to move
                if len(parts) >= 2:
                    features[14] = 1.0 if parts[1] == 'w' else -1.0  # White to move = 1, Black = -1

                # Check/checkmate detection (simplified)
                features[15] = 0.0  # Placeholder for check status

        except Exception as e:
            logger.warning(f"Error extracting FEN features: {e}")

        return features

    def _extract_question_features(self, question_text: str) -> np.ndarray:
        """Extract features from question text."""
        features = np.zeros(16, dtype=np.float32)
        question_lower = question_text.lower()

        # Question type indicators
        question_words = question_lower.split()

        # Move-related questions
        move_indicators = ["move", "best", "suggest", "recommend", "play", "what", "uci"]
        features[0] = 1.0 if any(word in question_words for word in move_indicators) else 0.0

        # Analysis-related questions
        analysis_indicators = ["analyze", "analysis", "evaluate", "assessment", "examine", "step"]
        features[1] = 1.0 if any(word in question_lower for word in analysis_indicators) else 0.0

        # Strategy-related questions
        strategy_indicators = ["strategy", "opening", "endgame", "principle", "plan", "idea"]
        features[2] = 1.0 if any(word in question_lower for word in strategy_indicators) else 0.0

        # Tactical questions
        tactical_indicators = ["tactical", "tactics", "attack", "defense", "sacrifice", "combination"]
        features[3] = 1.0 if any(word in question_lower for word in tactical_indicators) else 0.0

        # Explanation requests
        explanation_indicators = ["explain", "why", "how", "because", "reason", "understand"]
        features[4] = 1.0 if any(word in question_lower for word in explanation_indicators) else 0.0

        # Question length and complexity
        features[5] = len(question_text) / 200.0  # Normalized length
        features[6] = len(question_words) / 20.0  # Normalized word count

        # Question mark presence (indicates question)
        features[7] = 1.0 if '?' in question_text else 0.0

        # Imperative mood (commands)
        imperative_indicators = ["suggest", "recommend", "show", "tell", "give", "find"]
        features[8] = 1.0 if any(word in question_words for word in imperative_indicators) else 0.0

        # Uncertainty indicators
        uncertainty_indicators = ["maybe", "perhaps", "could", "might", "possible", "option"]
        features[9] = 1.0 if any(word in question_lower for word in uncertainty_indicators) else 0.0

        # Position-specific terms
        position_indicators = ["position", "board", "square", "piece", "pawn", "knight", "bishop", "rook", "queen", "king"]
        features[10] = 1.0 if any(word in question_lower for word in position_indicators) else 0.0

        # Time-related terms
        time_indicators = ["time", "tempo", "initiative", "development", "clock", "moves"]
        features[11] = 1.0 if any(word in question_lower for word in time_indicators) else 0.0

        # Complexity indicators
        complexity_indicators = ["complex", "difficult", "advanced", "deep", "complicate"]
        features[12] = 1.0 if any(word in question_lower for word in complexity_indicators) else 0.0

        # Educational terms
        educational_indicators = ["learn", "teach", "understand", "concept", "principle", "rule"]
        features[13] = 1.0 if any(word in question_lower for word in educational_indicators) else 0.0

        # Technical chess terms
        technical_indicators = ["castling", "en passant", "promotion", "fork", "pin", "skewer", "discovered"]
        features[14] = 1.0 if any(word in question_lower for word in technical_indicators) else 0.0

        # FEN presence indicator
        features[15] = 1.0 if 'fen:' in question_lower else 0.0

        return features

    def _make_routing_decision(self, position_features: torch.Tensor, question_text: str, complexity_score: Optional[float]) -> RoutingDecision:
        """Make a routing decision using the neural network and fallback logic."""
        try:
            # Use neural network for routing
            with torch.no_grad():
                gate_logits, confidence = self.forward(position_features)

                # Apply softmax to get probabilities
                gate_probs = F.softmax(gate_logits, dim=-1)

                # Get expert index and confidence
                expert_idx = torch.argmax(gate_probs, dim=-1).item()
                confidence_score = confidence.item()

            # Map index back to expert name
            primary_expert = self.expert_names[expert_idx] if expert_idx < len(self.expert_names) else self.expert_names[0]

            # Apply adaptive routing adjustments
            primary_expert = self._apply_adaptive_routing(primary_expert, gate_probs.numpy(), confidence_score, question_text)

            # Calculate expert weights
            expert_weights = {self.expert_names[i]: prob for i, prob in enumerate(gate_probs.numpy())}

            # Determine ensemble mode
            ensemble_mode = confidence_score < 0.7 and len([w for w in expert_weights.values() if w > 0.3]) > 1

            # Generate reasoning
            reasoning = self._generate_routing_reasoning(primary_expert, confidence_score, question_text, gate_probs.numpy())

            return RoutingDecision(
                primary_expert=primary_expert,
                expert_weights=expert_weights,
                confidence_score=confidence_score,
                reasoning=reasoning,
                ensemble_mode=ensemble_mode,
                fallback_used=False
            )

        except Exception as e:
            logger.warning(f"Neural routing failed, falling back to keyword-based routing: {e}")

            # Fallback to keyword-based routing
            fen_present = 'FEN:' in question_text.upper()
            fallback_expert = self._keyword_based_routing(question_text, fen_present)

            if fallback_expert:
                return RoutingDecision(
                    primary_expert=fallback_expert,
                    expert_weights={fallback_expert: 1.0},
                    confidence_score=0.5,  # Lower confidence for fallback
                    reasoning=f"Fallback to keyword-based routing: {fallback_expert}",
                    ensemble_mode=False,
                    fallback_used=True
                )
            else:
                # Ultimate fallback to tutor expert
                return RoutingDecision(
                    primary_expert="tutor",
                    expert_weights={"tutor": 1.0},
                    confidence_score=0.3,
                    reasoning="Ultimate fallback to tutor expert",
                    ensemble_mode=False,
                    fallback_used=True
                )

    def _apply_adaptive_routing(self, primary_expert: str, gate_probs: np.ndarray, confidence: float, question_text: str) -> str:
        """Apply adaptive routing adjustments based on expert performance."""
        # If confidence is low, consider performance-based adjustments
        if confidence < 0.6:
            # Check if there's a better performing expert
            best_expert = primary_expert
            best_score = self.adaptive_state.expert_performance_scores.get(primary_expert, 0.7)

            for expert_name in self.expert_names:
                if expert_name != primary_expert:
                    score = self.adaptive_state.expert_performance_scores.get(expert_name, 0.7)
                    if score > best_score + 0.1:  # 10% improvement threshold
                        best_expert = expert_name
                        best_score = score

            if best_expert != primary_expert:
                logger.info(f"Adaptive routing: switching from {primary_expert} to {best_expert} based on performance")
                return best_expert

        return primary_expert

    def _generate_routing_reasoning(self, primary_expert: str, confidence: float, question_text: str, gate_probs: np.ndarray) -> str:
        """Generate human-readable reasoning for the routing decision."""
        reasoning_parts = []

        # Confidence-based reasoning
        if confidence > 0.8:
            reasoning_parts.append("High confidence routing")
        elif confidence > 0.6:
            reasoning_parts.append("Moderate confidence routing")
        else:
            reasoning_parts.append("Low confidence routing with fallback logic")

        # Question type analysis
        question_lower = question_text.lower()
        if any(word in question_lower for word in ["move", "best", "play"]):
            reasoning_parts.append("Move-focused query")
        elif any(word in question_lower for word in ["analyze", "analysis", "evaluate"]):
            reasoning_parts.append("Analysis-focused query")
        elif any(word in question_lower for word in ["strategy", "opening", "endgame"]):
            reasoning_parts.append("Strategic query")
        else:
            reasoning_parts.append("General chess query")

        # Expert-specific reasoning
        if primary_expert == "uci":
            reasoning_parts.append("Routed to UCI expert for move generation")
        elif primary_expert == "tutor":
            reasoning_parts.append("Routed to Tutor expert for educational analysis")
        elif primary_expert == "director":
            reasoning_parts.append("Routed to Director expert for strategic guidance")

        return " | ".join(reasoning_parts)

    def _log_routing_decision(self, decision: RoutingDecision, position_fen: str, question_text: str):
        """Log routing decision for offline analysis."""
        try:
            log_entry = {
                "timestamp": time.time(),
                "position_fen": position_fen[:100] + "..." if len(position_fen) > 100 else position_fen,
                "question": question_text[:200] + "..." if len(question_text) > 200 else question_text,
                "primary_expert": decision.primary_expert,
                "confidence": decision.confidence_score,
                "ensemble_mode": decision.ensemble_mode,
                "fallback_used": decision.fallback_used,
                "expert_weights": decision.expert_weights
            }

            with self._decision_log_lock:
                with open(self.decision_log_path, 'a') as f:
                    f.write(json.dumps(log_entry) + '\n')

        except Exception as e:
            logger.warning(f"Failed to log routing decision: {e}")

    def update_expert_performance(self, expert_name: str, accuracy: float, response_time: float, user_satisfaction: float):
        """Update performance metrics for an expert."""
        if expert_name in self.adaptive_state.expert_metrics:
            metrics = self.adaptive_state.expert_metrics[expert_name]

            # Update moving averages
            if metrics.sample_count == 0:
                metrics.accuracy = accuracy
                metrics.response_time = response_time
                metrics.user_satisfaction = user_satisfaction
            else:
                # Exponential moving average with 0.1 learning rate
                alpha = 0.1
                metrics.accuracy = (1 - alpha) * metrics.accuracy + alpha * accuracy
                metrics.response_time = (1 - alpha) * metrics.response_time + alpha * response_time
                metrics.user_satisfaction = (1 - alpha) * metrics.user_satisfaction + alpha * user_satisfaction

            metrics.sample_count += 1
            metrics.last_updated = time.time()

            # Update performance scores
            self.adaptive_state.expert_performance_scores[expert_name] = metrics.user_satisfaction

    def get_routing_stats(self) -> Dict[str, Any]:
        """Get routing statistics for monitoring."""
        return {
            "total_requests": self._total_requests,
            "cache_hit_rate": self._routing_cache_hits / max(1, self._total_requests),
            "feature_cache_hit_rate": self._feature_cache_hits / max(1, self._total_requests),
            "cache_size": len(self._routing_cache),
            "expert_performance": self.adaptive_state.expert_performance_scores.copy(),
            "expert_metrics": {
                name: {
                    "accuracy": metrics.accuracy,
                    "response_time": metrics.response_time,
                    "user_satisfaction": metrics.user_satisfaction,
                    "sample_count": metrics.sample_count
                }
                for name, metrics in self.adaptive_state.expert_metrics.items()
            }
        }

    def save_router_state(self, filepath: str):
        """Save router state for persistence."""
        import json

        state = {
            "expert_performance": self.expert_performance,
            "adaptive_state": {
                "expert_performance_scores": self.adaptive_state.expert_performance_scores,
                "phase_expert_preferences": self.adaptive_state.phase_expert_preferences,
                "learning_rate": self.adaptive_state.learning_rate,
                "performance_monitoring_enabled": self.adaptive_state.performance_monitoring_enabled
            },
            "routing_stats": self.get_routing_stats()
        }

        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)

        logger.info(f"Router state saved to {filepath}")

    def load_router_state(self, filepath: str) -> bool:
        """Load router state from file."""
        try:
            import json

            with open(filepath, 'r') as f:
                state = json.load(f)

            # Restore expert performance
            self.expert_performance.update(state.get("expert_performance", {}))

            # Restore adaptive state
            adaptive_data = state.get("adaptive_state", {})
            self.adaptive_state.expert_performance_scores.update(adaptive_data.get("expert_performance_scores", {}))
            self.adaptive_state.phase_expert_preferences.update(adaptive_data.get("phase_expert_preferences", {}))
            self.adaptive_state.learning_rate = adaptive_data.get("learning_rate", 0.01)
            self.adaptive_state.performance_monitoring_enabled = adaptive_data.get("performance_monitoring_enabled", True)

            logger.info(f"Router state loaded from {filepath}")
            return True

        except Exception as e:
            logger.error(f"Failed to load router state from {filepath}: {e}")
            return False


# Import here to avoid circular imports
import json
import time
