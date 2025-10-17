#!/usr/bin/env python3
"""Type definitions and data classes for MoE router system.

Extracted from the monolithic moe_router.py to improve maintainability.
"""

from typing import Dict, List, Optional, Any
import numpy as np
from dataclasses import dataclass, field
import torch
from torch.utils.data import Dataset


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
    expert_usage_distribution: Dict[str, int] = field(default_factory=dict)
    response_time_distribution: Dict[str, float] = field(default_factory=dict)
    cache_hit_rate: float = 0.0
    adaptive_routing_improvements: float = 0.0


@dataclass
class MoEPerformanceSnapshot:
    """Performance snapshot for MoE system."""
    timestamp: float
    metrics: MoERoutingMetrics
    expert_performance: Dict[str, ExpertPerformanceMetrics]
    system_health: Dict[str, Any]
    recommendations: List[str] = field(default_factory=list)


@dataclass
class MoETrainingConfig:
    """Configuration for MoE router training."""
    learning_rate: float = 1e-3
    hidden_dim: int = 128
    training_steps: int = 1000
    batch_size: int = 64
    eval_interval: int = 100
    save_interval: int = 500
    early_stopping_patience: int = 10
    embedding_dim: int = 32
    dropout_rate: float = 0.1
    weight_decay: float = 1e-4
    warmup_steps: int = 100


# Expert name constants
EXPERT_NAMES = ["uci", "tutor", "director"]
EXPERT_TO_INDEX = {name: i for i, name in enumerate(EXPERT_NAMES)}

# Query categories for routing
QUERY_CATEGORIES = [
    "move_analysis",
    "position_evaluation",
    "strategic_guidance",
    "tactical_analysis",
    "opening_principles",
    "endgame_strategy",
    "general_chess"
]

# Performance thresholds
PERFORMANCE_THRESHOLDS = {
    "min_confidence": 0.6,
    "retraining_threshold": 0.1,
    "ensemble_threshold": 0.8,
    "fallback_threshold": 0.3
}
