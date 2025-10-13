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
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import os

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
class UserFeedback:
    """User feedback for adaptive routing."""
    query_hash: str
    expert_used: str
    rating: float  # 0.0 to 1.0 (user satisfaction)
    response_quality: float  # 0.0 to 1.0 (objective quality metric)
    timestamp: float
    query_type: str
    game_phase: Optional[List[float]] = None


@dataclass
class ExpertPerformanceMetrics:
    """Performance metrics for individual experts."""
    accuracy: float = 0.0
    response_time: float = 0.0
    user_satisfaction: float = 0.0
    response_quality: float = 0.0
    last_updated: float = 0.0
    sample_count: int = 0
    retraining_triggered: bool = False
    baseline_performance: float = 0.7


@dataclass
class RetrainingTrigger:
    """Trigger conditions for expert retraining."""
    expert_name: str
    trigger_reason: str
    current_performance: float
    threshold: float
    timestamp: float
    recommended_action: str


@dataclass
class AdaptiveRoutingState:
    """State for adaptive routing learning."""
    expert_performance_scores: Dict[str, float] = field(default_factory=lambda: {'uci': 0.7, 'tutor': 0.75, 'director': 0.65})
    expert_metrics: Dict[str, ExpertPerformanceMetrics] = field(default_factory=dict)
    phase_expert_preferences: Dict[str, Dict[str, float]] = field(default_factory=dict)
    feedback_history: List[UserFeedback] = field(default_factory=list)
    retraining_triggers: List[RetrainingTrigger] = field(default_factory=list)
    learning_rate: float = 0.01
    max_feedback_history: int = 1000
    performance_monitoring_enabled: bool = True


@dataclass
class MoERoutingMetrics:
    """Metrics for MoE routing performance."""
    total_requests: int = 0
    routing_accuracy: float = 0.0
    average_confidence: float = 0.0
    ensemble_usage_rate: float = 0.0
    fallback_rate: float = 0.0
    expert_usage_stats: Dict[str, int] = field(default_factory=dict)
    user_satisfaction_score: float = 0.0
    adaptive_improvement_rate: float = 0.0


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

        # Expert performance tracking
        self.expert_performance = {name: {'accuracy': 0.7, 'speed': 0.8, 'quality': 0.75}
                                  for name in self.expert_names}

        # Adaptive routing state
        self.adaptive_state = AdaptiveRoutingState()

        # Initialize expert performance metrics
        for expert_name in self.expert_names:
            if expert_name not in self.adaptive_state.expert_metrics:
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
        default_log_path = Path("reports") / "moe" / "routing_decisions.jsonl"
        self.decision_log_path = Path(os.environ.get("CHESSGEMMA_ROUTER_LOG", default_log_path))
        self._decision_log_lock = threading.Lock()
        if self.log_decisions_enabled:
            try:
                self.decision_log_path.parent.mkdir(parents=True, exist_ok=True)
            except Exception as exc:
                logger.warning(f"Unable to create router log directory {self.decision_log_path.parent}: {exc}")
                self.log_decisions_enabled = False

        logger.info(f"🧠 Optimized MoE Router initialized with {num_experts} experts")

        # The router is used purely for inference; ensure dropout layers are disabled
        self.eval()

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
            logger.info(f"🎯 Cached routing: {cached_decision.primary_expert} (confidence: {cached_decision.confidence_score:.3f})")
            return cached_decision

        # Extract features from position (with caching)
        position_features = self._extract_position_features_cached(position_fen, question_text)

        # Get routing decision
        with torch.no_grad():
            gate_logits, confidence = self.forward(position_features.unsqueeze(0))
            gate_probs = F.softmax(gate_logits, dim=-1).squeeze()

        # Apply expert performance weighting
        weighted_probs = self._apply_performance_weighting(gate_probs)
        raw_confidence = float(confidence.item())
        weighted_prob_dict = {
            name: float(weighted_probs[idx].item()) if idx < len(weighted_probs) else 0.0
            for idx, name in enumerate(self.expert_names)
        }

        # Extract game phase information for context-aware routing
        game_phase = None
        if position_fen:
            try:
                board = self._fen_to_board(position_fen)
                game_phase = self._detect_game_phase(board)
            except Exception:
                game_phase = [0.33, 0.34, 0.33]  # Default equal distribution

        # Make routing decision with context awareness
        decision = self._make_routing_decision(weighted_probs, raw_confidence, position_fen, query_type, game_phase)

        # Keyword-based routing fallback for low confidence decisions
        keyword_override = False
        if decision.confidence_score < 0.7:  # Balanced threshold
            # Try keyword-based routing on question text if available, otherwise query_type
            routing_text = question_text if question_text else (query_type if query_type != "auto" else "")
            keyword_expert = self._keyword_based_routing(routing_text, fen_present=bool(position_fen))
            if keyword_expert:
                # Override with keyword-based decision
                keyword_override = True
                decision.primary_expert = keyword_expert
                decision.confidence_score = 0.9  # Higher confidence for keyword matches
                decision.expert_weights = {"uci": 0.1, "tutor": 0.1, "director": 0.1}  # Reset weights as dict
                decision.expert_weights[keyword_expert] = 0.8   # Boost primary expert
                decision.ensemble_mode = False
                decision.reasoning = f"Keyword-based routing override (ML confidence {decision.confidence_score:.2f})"
                logger.info(f"🔄 Keyword routing override: {keyword_expert} (original confidence: {decision.confidence_score:.3f})")

        # Cache the decision
        self._cache_routing_decision(cache_key, decision)

        # Persist routing metadata for offline analysis
        question_feature_vector = self._extract_question_features(question_text) if question_text else [0.0] * 16
        self._log_routing_event(
            fen=position_fen,
            question_text=question_text,
            query_type=query_type,
            raw_confidence=raw_confidence,
            weighted_probs=weighted_prob_dict,
            decision=decision,
            keyword_override=keyword_override,
            question_features=question_feature_vector
        )

        logger.info(f"🎯 Computed routing: {decision.primary_expert} (confidence: {decision.confidence_score:.3f})")
        return decision

    def _extract_position_features(self, fen: str, question: str) -> torch.Tensor:
        """Extract comprehensive features from chess position and question for routing."""
        features = []

        # Position-based features (16 features)
        if fen:
            try:
                board = self._fen_to_board(fen)
                features.extend(self._extract_position_features_only(board))
            except Exception:
                # Fallback for invalid FEN
                features.extend([0.0] * 16)
        else:
            features.extend([0.0] * 16)

        # Question-based features (16 features)
        features.extend(self._extract_question_features(question))

        # Ensure we have exactly feature_dim features
        feature_tensor = torch.tensor(features, dtype=torch.float32)
        if len(feature_tensor) != self.feature_dim:
            if len(feature_tensor) < self.feature_dim:
                padding = torch.zeros(self.feature_dim - len(feature_tensor))
                feature_tensor = torch.cat([feature_tensor, padding])
            else:
                feature_tensor = feature_tensor[:self.feature_dim]

        return feature_tensor

    def _extract_position_features_only(self, board: List[List[str]]) -> List[float]:
        """Extract features from chess position only."""
        features = []

        # Material analysis (4 features)
        material_balance = self._calculate_material_balance(board)
        features.append(material_balance)  # -1 to 1

        total_material = sum(self._count_pieces(board, color) for color in ['white', 'black'])
        features.append(min(total_material / 78, 1.0))  # Normalized total material 0-1

        white_material = self._count_pieces(board, 'white')
        black_material = self._count_pieces(board, 'black')
        features.append(white_material / max(total_material, 1))  # White material ratio
        features.append(black_material / max(total_material, 1))  # Black material ratio

        # King safety and positioning (4 features)
        king_safety_white = self._assess_king_safety(board)
        king_safety_black = self._assess_king_safety(board)
        features.append(king_safety_white)
        features.append(king_safety_black)

        king_distance = self._calculate_king_distance(board)
        features.append(min(king_distance / 14, 1.0))  # Normalized king distance

        # Check for castling rights (simplified from FEN)
        features.append(1.0)  # Placeholder for castling rights

        # Piece activity and development (4 features)
        piece_activity = self._calculate_piece_activity(board)
        features.append(piece_activity)

        developed_pieces = self._count_developed_pieces(board)
        features.append(developed_pieces / 16)  # Max 16 developed pieces

        central_control = self._assess_central_control(board)
        features.append(central_control)

        mobility_ratio = self._calculate_mobility_ratio(board)
        features.append(mobility_ratio)

        # Pawn structure and tactics (4 features)
        pawn_complexity = self._assess_pawn_structure(board)
        features.append(pawn_complexity)

        tactical_potential = self._detect_tactical_motifs(board)
        features.append(tactical_potential)

        open_files = self._count_open_files(board)
        features.append(open_files / 8)  # Max 8 open files

        doubled_pawns = self._count_doubled_pawns(board)
        features.append(min(doubled_pawns / 8, 1.0))  # Max 8 doubled pawn files

        # Game phase detection (3 features)
        game_phase = self._detect_game_phase(board)
        features.extend(game_phase)  # [opening_score, middlegame_score, endgame_score]

        return features

    def _detect_game_phase(self, board: List[List[str]]) -> List[float]:
        """Detect the game phase (opening/middlegame/endgame) and return normalized scores.

        Returns [opening_score, middlegame_score, endgame_score] where scores sum to 1.
        """
        # Calculate total material (excluding kings)
        total_material = sum(self._count_pieces(board, color) for color in ['white', 'black'])

        # Phase detection based on material and piece development
        developed_pieces = self._count_developed_pieces(board)
        castling_possible = self._check_castling_potential(board)

        # Opening characteristics: High material, low development, castling available
        opening_score = 0.0
        if total_material >= 60:  # Most pieces still on board
            opening_score = 0.8
            if developed_pieces < 8:  # Few pieces developed
                opening_score += 0.2
        opening_score = min(opening_score, 1.0)

        # Middlegame characteristics: Moderate material, good development, kings safer
        middlegame_score = 0.0
        if 25 <= total_material <= 60:  # Some pieces traded
            middlegame_score = 0.6
            if 6 <= developed_pieces <= 12:  # Moderate development
                middlegame_score += 0.3
            if not castling_possible:  # Castling likely completed or not possible
                middlegame_score += 0.1
        middlegame_score = min(middlegame_score, 1.0)

        # Endgame characteristics: Low material, high development, simplified position
        endgame_score = 0.0
        if total_material <= 25:  # Few pieces remaining
            endgame_score = 0.7
            if developed_pieces >= 10:  # Most pieces highly active
                endgame_score += 0.3
        endgame_score = min(endgame_score, 1.0)

        # Normalize to ensure they sum to approximately 1
        total_score = opening_score + middlegame_score + endgame_score
        if total_score > 0:
            opening_score /= total_score
            middlegame_score /= total_score
            endgame_score /= total_score

        return [opening_score, middlegame_score, endgame_score]

    def _apply_game_phase_routing(self, expert_probs: Dict[str, float],
                                game_phase: List[float], query_type: str) -> Dict[str, float]:
        """Apply context-aware routing adjustments based on game phase.

        Args:
            expert_probs: Base expert probabilities from the model
            game_phase: [opening_score, middlegame_score, endgame_score]
            query_type: Type of query being made

        Returns:
            Adjusted expert probabilities
        """
        opening_score, middlegame_score, endgame_score = game_phase
        adjusted_probs = expert_probs.copy()

        # Phase-specific expert preferences
        phase_preferences = {
            'opening': {
                'uci': 1.0,      # Openings benefit from precise move generation
                'tutor': 1.2,    # Educational analysis helps with development principles
                'director': 1.1  # Strategic understanding of opening plans
            },
            'middlegame': {
                'uci': 1.1,      # Tactical moves are crucial in middlegame
                'tutor': 1.3,    # Complex analysis needed for middlegame positions
                'director': 1.0  # Strategic planning is key
            },
            'endgame': {
                'uci': 1.2,      # Precise endgame moves are critical
                'tutor': 1.1,    # Technical endgame analysis
                'director': 1.0  # Endgame principles and techniques
            }
        }

        # Determine dominant phase
        if opening_score > 0.5:
            phase = 'opening'
        elif endgame_score > 0.5:
            phase = 'endgame'
        else:
            phase = 'middlegame'

        # Apply phase-specific adjustments
        preferences = phase_preferences[phase]
        for expert, preference in preferences.items():
            if expert in adjusted_probs:
                # Apply preference multiplier (subtle adjustment to not override model completely)
                adjustment_factor = 0.1 * (preference - 1.0)  # Scale down for subtle influence
                adjusted_probs[expert] *= (1.0 + adjustment_factor)

        # Normalize probabilities
        total = sum(adjusted_probs.values())
        if total > 0:
            adjusted_probs = {expert: prob / total for expert, prob in adjusted_probs.items()}

        logger.debug(f"Phase-aware routing: {phase} phase -> {adjusted_probs}")
        return adjusted_probs

    def track_expert_performance(self, expert_name: str, response_time: float,
                               response_quality: float, user_satisfaction: Optional[float] = None):
        """Track performance metrics for an expert.

        Args:
            expert_name: Name of the expert
            response_time: Time taken to generate response (seconds)
            response_quality: Objective quality score (0.0 to 1.0)
            user_satisfaction: User satisfaction rating (0.0 to 1.0), uses response_quality if None
        """
        if not self.adaptive_state.performance_monitoring_enabled:
            return

        if user_satisfaction is None:
            user_satisfaction = response_quality

        metrics = self.adaptive_state.expert_metrics.get(expert_name)
        if not metrics:
            metrics = ExpertPerformanceMetrics()
            self.adaptive_state.expert_metrics[expert_name] = metrics

        # Update metrics using exponential moving average
        alpha = 0.1  # Smoothing factor
        metrics.response_time = metrics.response_time * (1 - alpha) + response_time * alpha
        metrics.response_quality = metrics.response_quality * (1 - alpha) + response_quality * alpha
        metrics.user_satisfaction = metrics.user_satisfaction * (1 - alpha) + user_satisfaction * alpha
        metrics.last_updated = time.time()
        metrics.sample_count += 1

        # Calculate overall performance score
        # Weight: 40% quality, 30% satisfaction, 30% speed (inverse normalized)
        speed_score = max(0, 1.0 - (response_time / 5.0))  # Penalize responses > 5 seconds
        overall_score = (response_quality * 0.4 + user_satisfaction * 0.3 + speed_score * 0.3)
        metrics.accuracy = overall_score

        # Check for retraining triggers
        self._check_retraining_triggers(expert_name, metrics)

        logger.debug(f"Tracked performance for {expert_name}: quality={response_quality:.2f}, satisfaction={user_satisfaction:.2f}, time={response_time:.2f}s")

    def _check_retraining_triggers(self, expert_name: str, metrics: ExpertPerformanceMetrics):
        """Check if retraining should be triggered for an expert."""
        current_performance = metrics.accuracy
        baseline = metrics.baseline_performance
        threshold = self.moe_config.retraining_threshold if hasattr(self, 'moe_config') else 0.1

        # Trigger conditions
        triggers = []

        # Performance degradation
        if current_performance < baseline - threshold:
            triggers.append(RetrainingTrigger(
                expert_name=expert_name,
                trigger_reason="performance_degradation",
                current_performance=current_performance,
                threshold=baseline - threshold,
                timestamp=time.time(),
                recommended_action=f"Retrain {expert_name} expert due to {threshold:.0%} performance drop"
            ))

        # Low response quality
        if metrics.response_quality < 0.6:
            triggers.append(RetrainingTrigger(
                expert_name=expert_name,
                trigger_reason="low_quality",
                current_performance=metrics.response_quality,
                threshold=0.6,
                timestamp=time.time(),
                recommended_action=f"Retrain {expert_name} expert due to consistently low response quality"
            ))

        # High response time
        if metrics.response_time > 3.0:  # More than 3 seconds average
            triggers.append(RetrainingTrigger(
                expert_name=expert_name,
                trigger_reason="slow_responses",
                current_performance=metrics.response_time,
                threshold=3.0,
                timestamp=time.time(),
                recommended_action=f"Retrain {expert_name} expert due to slow response times"
            ))

        # Add triggers to the list
        for trigger in triggers:
            self.adaptive_state.retraining_triggers.append(trigger)
            metrics.retraining_triggered = True
            logger.warning(f"🚨 Retraining trigger for {expert_name}: {trigger.trigger_reason} "
                          f"(current: {trigger.current_performance:.2f}, threshold: {trigger.threshold:.2f})")

    def get_expert_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive expert performance report."""
        report = {
            'performance_monitoring_enabled': self.adaptive_state.performance_monitoring_enabled,
            'expert_metrics': {},
            'retraining_triggers': [],
            'overall_health_score': 0.0
        }

        total_score = 0.0
        expert_count = 0

        for expert_name, metrics in self.adaptive_state.expert_metrics.items():
            report['expert_metrics'][expert_name] = {
                'accuracy': metrics.accuracy,
                'response_time': metrics.response_time,
                'response_quality': metrics.response_quality,
                'user_satisfaction': metrics.user_satisfaction,
                'sample_count': metrics.sample_count,
                'last_updated': metrics.last_updated,
                'baseline_performance': metrics.baseline_performance,
                'performance_trend': metrics.accuracy - metrics.baseline_performance,
                'retraining_needed': metrics.retraining_triggered
            }
            total_score += metrics.accuracy
            expert_count += 1

        if expert_count > 0:
            report['overall_health_score'] = total_score / expert_count

        # Add retraining triggers
        for trigger in self.adaptive_state.retraining_triggers[-10:]:  # Last 10 triggers
            report['retraining_triggers'].append({
                'expert': trigger.expert_name,
                'reason': trigger.trigger_reason,
                'current_performance': trigger.current_performance,
                'threshold': trigger.threshold,
                'timestamp': trigger.timestamp,
                'recommended_action': trigger.recommended_action
            })

        return report

    def reset_retraining_triggers(self, expert_name: Optional[str] = None):
        """Reset retraining triggers for an expert or all experts.

        Args:
            expert_name: Specific expert to reset, or None for all experts
        """
        if expert_name:
            # Remove triggers for specific expert
            self.adaptive_state.retraining_triggers = [
                t for t in self.adaptive_state.retraining_triggers
                if t.expert_name != expert_name
            ]
            # Reset expert's retraining flag
            if expert_name in self.adaptive_state.expert_metrics:
                self.adaptive_state.expert_metrics[expert_name].retraining_triggered = False
        else:
            # Reset all triggers
            self.adaptive_state.retraining_triggers.clear()
            for metrics in self.adaptive_state.expert_metrics.values():
                metrics.retraining_triggered = False

        logger.info(f"Reset retraining triggers for {expert_name or 'all experts'}")

    def collect_user_feedback(self, query_hash: str, expert_used: str,
                            user_rating: float, response_quality: float = None,
                            query_type: str = "unknown", game_phase: List[float] = None):
        """Collect user feedback to improve routing decisions.

        Args:
            query_hash: Unique identifier for the query
            expert_used: Which expert was used to generate the response
            user_rating: User satisfaction rating (0.0 to 1.0)
            response_quality: Objective quality metric (0.0 to 1.0), uses user_rating if None
            query_type: Type of query that was made
            game_phase: Game phase scores [opening, middlegame, endgame]
        """
        if response_quality is None:
            response_quality = user_rating

        feedback = UserFeedback(
            query_hash=query_hash,
            expert_used=expert_used,
            rating=user_rating,
            response_quality=response_quality,
            timestamp=time.time(),
            query_type=query_type,
            game_phase=game_phase
        )

        # Add to feedback history
        self.adaptive_state.feedback_history.append(feedback)

        # Maintain history size limit
        if len(self.adaptive_state.feedback_history) > self.adaptive_state.max_feedback_history:
            self.adaptive_state.feedback_history.pop(0)

        # Update adaptive routing based on feedback
        self._update_adaptive_routing(feedback)

        # Track expert performance metrics
        # Note: response_time would need to be passed from the calling context
        # For now, we'll use a placeholder response time
        self.track_expert_performance(
            expert_name=expert_used,
            response_time=1.0,  # Placeholder - should be passed from inference system
            response_quality=response_quality,
            user_satisfaction=user_rating
        )

        logger.debug(f"Collected feedback for {expert_used}: rating={user_rating:.2f}, quality={response_quality:.2f}")

    def _update_adaptive_routing(self, feedback: UserFeedback):
        """Update adaptive routing state based on user feedback."""
        expert = feedback.expert_used
        combined_score = (feedback.rating + feedback.response_quality) / 2.0

        # Update expert performance scores using exponential moving average
        learning_rate = self.adaptive_state.learning_rate
        current_score = self.adaptive_state.expert_performance_scores.get(expert, 0.7)
        new_score = current_score * (1 - learning_rate) + combined_score * learning_rate
        self.adaptive_state.expert_performance_scores[expert] = new_score

        # Update phase-specific preferences if game phase is available
        if feedback.game_phase:
            opening_score, middlegame_score, endgame_score = feedback.game_phase

            # Determine dominant phase
            if opening_score > 0.5:
                phase = 'opening'
            elif endgame_score > 0.5:
                phase = 'endgame'
            else:
                phase = 'middlegame'

            # Update phase-specific expert preferences
            if phase not in self.adaptive_state.phase_expert_preferences:
                self.adaptive_state.phase_expert_preferences[phase] = {}

            phase_prefs = self.adaptive_state.phase_expert_preferences[phase]
            current_pref = phase_prefs.get(expert, 1.0)
            # Adjust preference based on feedback
            adjustment = (combined_score - 0.5) * learning_rate * 0.5  # Smaller adjustment for preferences
            phase_prefs[expert] = current_pref + adjustment

        logger.debug(f"Updated adaptive routing: {expert} score -> {new_score:.3f}")

    def _apply_adaptive_routing(self, expert_probs: Dict[str, float],
                               game_phase: Optional[List[float]] = None) -> Dict[str, float]:
        """Apply adaptive routing adjustments based on learned performance."""
        adjusted_probs = expert_probs.copy()

        # Apply expert performance scores
        for expert, score in self.adaptive_state.expert_performance_scores.items():
            if expert in adjusted_probs:
                # Boost probability for high-performing experts
                performance_boost = (score - 0.7) * 0.2  # Scale performance difference
                adjusted_probs[expert] *= (1.0 + performance_boost)

        # Apply phase-specific preferences
        if game_phase:
            opening_score, middlegame_score, endgame_score = game_phase

            if opening_score > 0.5:
                phase = 'opening'
            elif endgame_score > 0.5:
                phase = 'endgame'
            else:
                phase = 'middlegame'

            phase_prefs = self.adaptive_state.phase_expert_preferences.get(phase, {})
            for expert, preference in phase_prefs.items():
                if expert in adjusted_probs:
                    # Apply learned preference adjustment
                    pref_adjustment = (preference - 1.0) * 0.1  # Scale down learned preferences
                    adjusted_probs[expert] *= (1.0 + pref_adjustment)

        # Normalize probabilities
        total = sum(adjusted_probs.values())
        if total > 0:
            adjusted_probs = {expert: prob / total for expert, prob in adjusted_probs.items()}

        return adjusted_probs

    def get_adaptive_routing_stats(self) -> Dict[str, Any]:
        """Get statistics about adaptive routing performance."""
        feedback_count = len(self.adaptive_state.feedback_history)

        if feedback_count == 0:
            return {
                'feedback_collected': 0,
                'expert_performance_scores': self.adaptive_state.expert_performance_scores.copy(),
                'phase_preferences': self.adaptive_state.phase_expert_preferences.copy(),
                'adaptive_learning_active': False
            }

        # Calculate average satisfaction
        avg_rating = sum(f.rating for f in self.adaptive_state.feedback_history) / feedback_count
        avg_quality = sum(f.response_quality for f in self.adaptive_state.feedback_history) / feedback_count

        # Calculate expert-specific stats
        expert_stats = {}
        for expert in self.expert_names:
            expert_feedback = [f for f in self.adaptive_state.feedback_history if f.expert_used == expert]
            if expert_feedback:
                expert_stats[expert] = {
                    'feedback_count': len(expert_feedback),
                    'avg_rating': sum(f.rating for f in expert_feedback) / len(expert_feedback),
                    'avg_quality': sum(f.response_quality for f in expert_feedback) / len(expert_feedback)
                }

        return {
            'feedback_collected': feedback_count,
            'average_user_rating': avg_rating,
            'average_response_quality': avg_quality,
            'expert_performance_scores': self.adaptive_state.expert_performance_scores.copy(),
            'phase_preferences': self.adaptive_state.phase_expert_preferences.copy(),
            'expert_feedback_stats': expert_stats,
            'adaptive_learning_active': True
        }

    def _check_castling_potential(self, board: List[List[str]]) -> bool:
        """Check if castling is still possible or likely."""
        # Simplified check: look for kings and rooks in starting positions
        # This is a basic heuristic since we don't have full FEN parsing
        white_king_home = 'K' in board[7]  # White king on back rank
        black_king_home = 'k' in board[0]  # Black king on back rank
        white_rooks_home = 'R' in board[7]  # White rooks on back rank
        black_rooks_home = 'r' in board[0]  # Black rooks on back rank

        return (white_king_home and white_rooks_home) or (black_king_home and black_rooks_home)

    def _extract_question_features(self, question: str) -> List[float]:
        """Extract a fixed-width feature vector from the raw question text."""
        question_lower = question.lower()

        # Core intent buckets (6 features)
        move_only_tokens = ['best move', 'move only', 'uci', 'play', 'respond with only', 'just give']
        analysis_tokens = ['analyze', 'analysis', 'evaluate', 'explain', 'step by step', 'why', 'because']
        strategy_tokens = ['strategy', 'plan', 'concept', 'idea', 'principle', 'theme', 'long term']
        rules_tokens = ['rule', 'legal', 'allowed', 'can i', 'is it legal', 'en passant', 'castle']
        opening_tokens = ['opening', 'theory', 'variation', 'line', 'sicilian', 'french', 'e4 e5']
        endgame_tokens = ['endgame', 'king and pawn', 'tablebase', 'opposition', 'zugzwang', 'promotion']

        def bucket_score(tokens: List[str]) -> float:
            hits = sum(1 for token in tokens if token in question_lower)
            return min(hits / 3.0, 1.0)

        features: List[float] = [
            bucket_score(move_only_tokens),
            bucket_score(analysis_tokens),
            bucket_score(strategy_tokens),
            bucket_score(rules_tokens),
            bucket_score(opening_tokens),
            bucket_score(endgame_tokens),
        ]

        # Structural cues (5 features)
        features.extend([
            1.0 if 'fen:' in question_lower else 0.0,
            1.0 if '?' in question else 0.0,
            min(len(question.split()) / 60.0, 1.0),
            min(len(question) / 400.0, 1.0),
            1.0 if any(str(i) + '.' in question for i in range(1, 4)) else 0.0,  # enumerated steps
        ])

        # Mode and persona hints (3 features)
        features.append(1.0 if 'mode: engine' in question_lower or 'engine mode' in question_lower else 0.0)
        features.append(1.0 if 'mode: tutor' in question_lower or 'tutor mode' in question_lower else 0.0)
        features.append(1.0 if 'mode: director' in question_lower or 'director mode' in question_lower else 0.0)

        # Output-format expectations (2 features)
        features.append(1.0 if any(token in question_lower for token in ['respond with only', 'just the move', 'uci format']) else 0.0)
        features.append(1.0 if any(token in question_lower for token in ['explain', 'why', 'because', 'steps']) else 0.0)

        # Domain references (2 features)
        history_tokens = ['fischer', 'kasparov', 'capablanca', 'history', 'classic', 'famous game']
        tutoring_tokens = ['learning', 'teach', 'lesson', 'tips', 'improve']
        features.append(bucket_score(history_tokens))
        features.append(bucket_score(tutoring_tokens))

        # Ensure deterministic width
        if len(features) != 16:
            if len(features) < 16:
                features.extend([0.0] * (16 - len(features)))
            else:
                features = features[:16]

        return features

    def _create_cache_key(self, fen: str, query_type: str, complexity_score: Optional[float], question_text: str = "") -> str:
        """Create a unique cache key for position and query combination."""
        key_components = [fen, query_type]
        if complexity_score is not None:
            key_components.append(f"{complexity_score:.3f}")
        if question_text:
            hashed_question = hashlib.md5(question_text.encode('utf-8')).hexdigest()
            key_components.append(hashed_question)
        key_string = "|".join(key_components)
        return hashlib.md5(key_string.encode()).hexdigest()

    def _extract_position_features_cached(self, fen: str, question_text: str) -> torch.Tensor:
        """Extract position features with caching for improved performance."""
        cache_basis = f"features_{fen}|{question_text}" if question_text else f"features_{fen}"
        cache_key = hashlib.md5(cache_basis.encode('utf-8')).hexdigest()

        # Check cache first
        if cache_key in self._position_cache:
            self._feature_cache_hits += 1
            cached_features = self._position_cache[cache_key]
            # Update LRU order
            self._position_cache.move_to_end(cache_key)
            return cached_features

        # Compute features
        features = self._extract_position_features(fen, question_text)

        # Cache the result
        self._position_cache[cache_key] = features.clone()
        self._maintain_cache_size()

        return features

    def _cache_routing_decision(self, cache_key: str, decision: RoutingDecision):
        """Cache routing decision for future use."""
        self._routing_cache[cache_key] = decision
        self._maintain_cache_size()

    def _log_routing_event(
        self,
        fen: str,
        question_text: str,
        query_type: str,
        raw_confidence: float,
        weighted_probs: Dict[str, float],
        decision: RoutingDecision,
        keyword_override: bool,
        question_features: List[float]
    ) -> None:
        """Persist routing metadata for offline analysis and retraining."""
        if not self.log_decisions_enabled:
            return

        try:
            ranked_experts = sorted(
                weighted_probs.items(),
                key=lambda item: item[1],
                reverse=True
            )
            entry = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "fen": fen,
                "query_type": query_type,
                "question": question_text,
                "question_preview": question_text[:200],
                "raw_confidence": raw_confidence,
                "final_confidence": decision.confidence_score,
                "primary_expert": decision.primary_expert,
                "expert_weights": ranked_experts,
                "keyword_override": keyword_override,
                "ensemble_mode": decision.ensemble_mode,
                "fallback_used": decision.fallback_used,
                "question_features": [round(val, 3) for val in question_features[:16]],
            }
            with self._decision_log_lock:
                with self.decision_log_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(entry) + "\n")
        except Exception as exc:
            logger.debug(f"Router decision logging failed: {exc}")

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

    def _count_pieces(self, board: List[List[str]], color: str) -> int:
        """Count total material value for a color."""
        piece_values = {'p': 1, 'n': 3, 'b': 3, 'r': 5, 'q': 9, 'k': 0}
        total = 0

        for row in board:
            for piece in row:
                if piece and piece != '':
                    if color == 'white' and piece.isupper():
                        total += piece_values.get(piece.lower(), 0)
                    elif color == 'black' and piece.islower():
                        total += piece_values.get(piece.lower(), 0)

        return total

    def _calculate_king_distance(self, board: List[List[str]]) -> float:
        """Calculate distance between kings."""
        kings = self._find_kings(board)
        if not kings['white'] or not kings['black']:
            return 7.0  # Default distance

        w_i, w_j = kings['white']
        b_i, b_j = kings['black']
        return ((w_i - b_i) ** 2 + (w_j - b_j) ** 2) ** 0.5

    def _count_developed_pieces(self, board: List[List[str]]) -> int:
        """Count pieces that have moved from starting squares."""
        developed = 0

        # Check knights
        knight_squares = [(0, 1), (0, 6), (7, 1), (7, 6)]
        for i, j in knight_squares:
            if i < len(board) and j < len(board[i]):
                piece = board[i][j]
                if piece != 'N' and piece != 'n':  # Knight has moved
                    developed += 1

        # Check bishops
        bishop_squares = [(0, 2), (0, 5), (7, 2), (7, 5)]
        for i, j in bishop_squares:
            if i < len(board) and j < len(board[i]):
                piece = board[i][j]
                if piece != 'B' and piece != 'b':  # Bishop has moved
                    developed += 1

        # Check center pawns (simple development indicator)
        center_pawn_squares = [(1, 3), (1, 4), (6, 3), (6, 4)]
        for i, j in center_pawn_squares:
            if i < len(board) and j < len(board[i]):
                piece = board[i][j]
                if piece != 'P' and piece != 'p':  # Pawn has moved
                    developed += 1

        return developed

    def _assess_central_control(self, board: List[List[str]]) -> float:
        """Assess control of central squares."""
        center_squares = [(3, 3), (3, 4), (4, 3), (4, 4)]
        controlled = 0

        for i, j in center_squares:
            if i < len(board) and j < len(board[i]):
                piece = board[i][j]
                if piece and piece != '':
                    controlled += 1

        return controlled / 4  # Max 4 central squares

    def _calculate_mobility_ratio(self, board: List[List[str]]) -> float:
        """Calculate piece mobility ratio between sides."""
        # Simplified mobility calculation
        white_pieces = sum(1 for row in board for piece in row if piece and piece.isupper())
        black_pieces = sum(1 for row in board for piece in row if piece and piece.islower())

        if black_pieces == 0:
            return 1.0
        return white_pieces / black_pieces

    def _count_open_files(self, board: List[List[str]]) -> int:
        """Count open files (no pawns)."""
        open_files = 0

        for j in range(8):
            has_white_pawn = any(board[i][j] == 'P' for i in range(8))
            has_black_pawn = any(board[i][j] == 'p' for i in range(8))

            if not has_white_pawn and not has_black_pawn:
                open_files += 1

        return open_files

    def _count_doubled_pawns(self, board: List[List[str]]) -> int:
        """Count files with doubled pawns."""
        doubled = 0

        for j in range(8):
            white_pawns = sum(1 for i in range(8) if board[i][j] == 'P')
            black_pawns = sum(1 for i in range(8) if board[i][j] == 'p')

            if white_pawns > 1:
                doubled += 1
            if black_pawns > 1:
                doubled += 1

        return doubled

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
                              fen: str, query_type: str, game_phase: Optional[List[float]] = None) -> RoutingDecision:
        """Make the final routing decision."""

        if not self.expert_names:
            raise RuntimeError("MoE router has no experts configured")

        # Get expert probabilities
        expert_probs = {name: prob.item() for name, prob in zip(self.expert_names, probs)}

        # Apply context-aware routing adjustments based on game phase
        if game_phase:
            expert_probs = self._apply_game_phase_routing(expert_probs, game_phase, query_type)

        # Apply adaptive routing based on learned user feedback
        expert_probs = self._apply_adaptive_routing(expert_probs, game_phase)

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
        # Ensure the directory exists
        Path(path).parent.mkdir(parents=True, exist_ok=True)

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
                    num_epochs: int = 10, batch_size: int = 32, learning_rate: float = 1e-3,
                    validate_every: int = 5, validation_examples: Optional[List[RouterTrainingExample]] = None):
        """Train the MoE router on routing decisions with validation."""

        print("🚀 Starting MoE router training...")
        print(f"   Examples: {len(training_examples)}")
        print(f"   Epochs: {num_epochs}")
        print(f"   Batch size: {batch_size}")
        print(f"   Learning rate: {learning_rate}")
        if validation_examples:
            print(f"   Validation examples: {len(validation_examples)}")

        # Create dataset and dataloader
        dataset = RouterTrainingDataset(training_examples)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Set up optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()

        # Training loop
        self.train()
        best_accuracy = 0.0
        best_val_accuracy = 0.0

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

            # Validation
            val_accuracy = None
            if validation_examples and (epoch + 1) % validate_every == 0:
                val_accuracy = self.evaluate_routing_accuracy(validation_examples)

            # Print metrics
            if val_accuracy is not None:
                print(f"Epoch {epoch+1:2d}: Loss={epoch_avg_loss:.4f}, TrainAcc={epoch_accuracy:.1%}, ValAcc={val_accuracy:.1%}")
            else:
                print(f"Epoch {epoch+1:2d}: Loss={epoch_avg_loss:.4f}, TrainAcc={epoch_accuracy:.1%}")

            # Save best model
            if epoch_accuracy > best_accuracy:
                best_accuracy = epoch_accuracy
                self.save_router("checkpoints/moe_router/best_checkpoint.pth")

            if val_accuracy and val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy

        print(f"🏁 Training complete. Best train accuracy: {best_accuracy:.1%}")
        if validation_examples:
            print(f"   Best validation accuracy: {best_val_accuracy:.1%}")
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
                logger.info(f"Registered {expert_name} expert adapter at {path_obj}")
            else:
                logger.warning(
                    "Expert model not found for %s: %s", expert_name=expert_name, model_path=model_path
                )

        self.prime_available_experts()

        # Parallel processing configuration
        self._parallel_enabled = os.environ.get('CHESSGEMMA_PARALLEL_INFERENCE', '1') not in ('0', 'false', 'False')
        self._max_parallel_experts = min(len(self.expert_models), int(os.environ.get('CHESSGEMMA_MAX_PARALLEL_EXPERTS', '3')))
        self._parallel_executor = ThreadPoolExecutor(max_workers=self._max_parallel_experts, thread_name_prefix="moe-expert")

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
                        complexity_score: Optional[float] = None, question_text: str = "") -> Dict[str, Any]:
        """Analyze a chess position using optimized MoE routing with caching."""

        # Get routing decision (with caching)
        routing_decision = self.router.route_query(fen, query_type, complexity_score, question_text=question_text)

        # Execute routing decision
        if routing_decision.ensemble_mode:
            # Use parallel processing if enabled and beneficial
            if self._parallel_enabled and len(routing_decision.expert_weights) > 1:
                response = self._execute_parallel_ensemble_inference(fen, routing_decision)
            else:
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
            'ensemble_mode': routing_decision.ensemble_mode,
            'fallback_used': routing_decision.fallback_used,
        }

        return response

    def shutdown(self):
        """Shutdown the MoE inference manager and clean up resources."""
        if hasattr(self, '_parallel_executor') and self._parallel_executor:
            self._parallel_executor.shutdown(wait=True)
            logger.debug("MoE parallel executor shutdown")

    def __del__(self):
        """Cleanup on deletion."""
        self.shutdown()

    def _execute_parallel_ensemble_inference(self, fen: str, routing_decision: RoutingDecision) -> Dict[str, Any]:
        """Execute ensemble inference with parallel expert processing."""
        if not self._parallel_enabled or len(routing_decision.expert_weights) <= 1:
            # Fall back to sequential processing
            return self._execute_ensemble_inference(fen, routing_decision)

        expert_results = []
        confidence_weights = []

        # Submit parallel inference tasks
        future_to_expert = {}
        for expert_name in routing_decision.expert_weights.keys():
            future = self._parallel_executor.submit(self._execute_single_expert_inference, fen, expert_name)
            future_to_expert[future] = expert_name

        # Collect results as they complete
        for future in as_completed(future_to_expert, timeout=30.0):  # 30 second timeout
            expert_name = future_to_expert[future]
            try:
                base_weight = routing_decision.expert_weights[expert_name]
                expert_response = future.result()

                # Extract confidence
                confidence = expert_response.get('confidence', 0.5)
                confidence_weight = base_weight * confidence

                expert_results.append({
                    'expert': expert_name,
                    'response': expert_response,
                    'base_weight': base_weight,
                    'confidence': confidence,
                    'confidence_weight': confidence_weight
                })
                confidence_weights.append(confidence_weight)

            except Exception as exc:
                logger.warning(f"Parallel inference failed for {expert_name}: {exc}")
                # Add fallback result
                base_weight = routing_decision.expert_weights[expert_name]
                expert_results.append({
                    'expert': expert_name,
                    'response': {'response': f'Analysis from {expert_name} expert', 'confidence': 0.1},
                    'base_weight': base_weight,
                    'confidence': 0.1,
                    'confidence_weight': base_weight * 0.1
                })
                confidence_weights.append(base_weight * 0.1)

        # Normalize confidence weights
        total_weight = sum(confidence_weights)
        if total_weight > 0:
            normalized_weights = [w / total_weight for w in confidence_weights]
        else:
            normalized_weights = [1.0 / len(confidence_weights)] * len(confidence_weights)

        # Reassign normalized weights
        for result, norm_weight in zip(expert_results, normalized_weights):
            result['normalized_weight'] = norm_weight

        # Combine responses using confidence-weighted ensemble
        combined_response = self._combine_confidence_weighted_responses(expert_results)

        # Calculate ensemble confidence
        ensemble_confidence = sum(r['confidence'] * r['normalized_weight'] for r in expert_results)

        return {
            'response': combined_response,
            'ensemble_used': [r['expert'] for r in expert_results],
            'weights': {r['expert']: r['normalized_weight'] for r in expert_results},
            'base_weights': {r['expert']: r['base_weight'] for r in expert_results},
            'confidences': {r['expert']: r['confidence'] for r in expert_results},
            'analysis_type': 'parallel_confidence_weighted_ensemble',
            'ensemble_confidence': ensemble_confidence,
            'expert_details': expert_results,
            'parallel_processing': True
        }

    def _execute_ensemble_inference(self, fen: str, routing_decision: RoutingDecision) -> Dict[str, Any]:
        """Execute inference with expert ensemble (sequential fallback)."""
        expert_results = []
        confidence_weights = []

        for expert_name, base_weight in routing_decision.expert_weights.items():
            expert_response = self._execute_single_expert_inference(fen, expert_name)

            # Extract confidence
            confidence = expert_response.get('confidence', 0.5)
            confidence_weight = base_weight * confidence

            expert_results.append({
                'expert': expert_name,
                'response': expert_response,
                'base_weight': base_weight,
                'confidence': confidence,
                'confidence_weight': confidence_weight
            })
            confidence_weights.append(confidence_weight)

        # Normalize confidence weights
        total_weight = sum(confidence_weights)
        if total_weight > 0:
            normalized_weights = [w / total_weight for w in confidence_weights]
        else:
            normalized_weights = [1.0 / len(confidence_weights)] * len(confidence_weights)

        # Reassign normalized weights
        for result, norm_weight in zip(expert_results, normalized_weights):
            result['normalized_weight'] = norm_weight

        # Combine responses using confidence-weighted ensemble
        combined_response = self._combine_confidence_weighted_responses(expert_results)

        # Calculate ensemble confidence
        ensemble_confidence = sum(r['confidence'] * r['normalized_weight'] for r in expert_results)

        return {
            'response': combined_response,
            'ensemble_used': [r['expert'] for r in expert_results],
            'weights': {r['expert']: r['normalized_weight'] for r in expert_results},
            'base_weights': {r['expert']: r['base_weight'] for r in expert_results},
            'confidences': {r['expert']: r['confidence'] for r in expert_results},
            'analysis_type': 'confidence_weighted_ensemble',
            'ensemble_confidence': ensemble_confidence,
            'expert_details': expert_results,
            'parallel_processing': False
        }

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
        """Execute inference with confidence-weighted expert ensemble."""
        expert_results = []
        confidence_weights = []

        # Execute inference for each expert
        for expert_name, base_weight in routing_decision.expert_weights.items():
            expert_response = self._execute_single_expert_inference(fen, expert_name)

            # Extract confidence from response (default to 0.5 if not available)
            confidence = expert_response.get('confidence', 0.5)

            # Apply confidence weighting: combine base routing weight with response confidence
            # This creates a more nuanced weighting that considers both routing decision and response quality
            confidence_weight = base_weight * confidence

            expert_results.append({
                'expert': expert_name,
                'response': expert_response,
                'base_weight': base_weight,
                'confidence': confidence,
                'confidence_weight': confidence_weight
            })
            confidence_weights.append(confidence_weight)

        # Normalize confidence weights
        total_weight = sum(confidence_weights)
        if total_weight > 0:
            normalized_weights = [w / total_weight for w in confidence_weights]
        else:
            # Fallback to equal weights if all confidences are 0
            normalized_weights = [1.0 / len(confidence_weights)] * len(confidence_weights)

        # Reassign normalized weights to results
        for result, norm_weight in zip(expert_results, normalized_weights):
            result['normalized_weight'] = norm_weight

        # Combine responses using confidence-weighted ensemble
        combined_response = self._combine_confidence_weighted_responses(expert_results)

        # Calculate ensemble confidence as weighted average of individual confidences
        ensemble_confidence = sum(r['confidence'] * r['normalized_weight'] for r in expert_results)

        return {
            'response': combined_response,
            'ensemble_used': [r['expert'] for r in expert_results],
            'weights': {r['expert']: r['normalized_weight'] for r in expert_results},
            'base_weights': {r['expert']: r['base_weight'] for r in expert_results},
            'confidences': {r['expert']: r['confidence'] for r in expert_results},
            'analysis_type': 'confidence_weighted_ensemble',
            'ensemble_confidence': ensemble_confidence,
            'expert_details': expert_results
        }

    def _combine_confidence_weighted_responses(self, expert_results: List[Dict[str, Any]]) -> str:
        """Combine responses using confidence-weighted ensemble logic."""
        if not expert_results:
            return "No expert responses available for ensemble."

        # Separate UCI moves from textual analysis
        uci_moves = []
        analysis_parts = []

        for result in expert_results:
            expert_name = result['expert']
            response_data = result['response']
            weight = result['normalized_weight']
            confidence = result['confidence']

            response_text = response_data.get('response', '')

            # Extract UCI moves from UCI expert if present
            if expert_name == 'uci' and weight > 0.2:  # Only consider significant UCI contributions
                # Try to extract UCI move from response
                import re
                uci_pattern = r'\b[a-h][1-8][a-h][1-8][qrbn]?\b'
                matches = re.findall(uci_pattern, response_text.lower())
                if matches:
                    uci_moves.append((matches[0], weight, confidence))

            # Collect analysis parts for non-UCI experts or low-confidence UCI
            if expert_name != 'uci' or weight <= 0.2:
                if weight > 0.15:  # Only include reasonably weighted analyses
                    analysis_parts.append({
                        'expert': expert_name,
                        'text': response_text,
                        'weight': weight,
                        'confidence': confidence
                    })

        # Build combined response
        response_parts = []

        # Add UCI move recommendation if available
        if uci_moves:
            # Select the highest confidence UCI move
            best_move = max(uci_moves, key=lambda x: x[2])  # Sort by confidence
            move, weight, confidence = best_move
            response_parts.append(f"**Recommended Move:** {move} (confidence: {confidence:.2f})")

        # Add expert analyses
        if analysis_parts:
            response_parts.append("\n**Expert Analysis:**")
            for analysis in sorted(analysis_parts, key=lambda x: x['weight'], reverse=True):
                expert_title = analysis['expert'].title()
                weight_indicator = f"[{analysis['weight']:.2f}]"
                response_parts.append(f"\n**{expert_title} Expert** {weight_indicator}:")
                response_parts.append(analysis['text'])

        # Add ensemble metadata
        ensemble_info = []
        for result in expert_results:
            ensemble_info.append(f"{result['expert']}: weight={result['normalized_weight']:.2f}, confidence={result['confidence']:.2f}")

        response_parts.append(f"\n**Ensemble Details:** {' | '.join(ensemble_info)}")

        return "\n".join(response_parts)

    def _combine_expert_responses(self, responses: List[Dict[str, Any]],
                                weights: List[float]) -> str:
        """Legacy method for backward compatibility - combines responses from multiple experts."""
        # Simplified response combination for legacy usage
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
