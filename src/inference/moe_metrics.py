#!/usr/bin/env python3
"""Performance metrics and monitoring for MoE router system.

Extracted from the monolithic moe_router.py to improve maintainability.
"""

import time
import logging
from typing import Dict, List, Optional, Any
from collections import defaultdict, deque
import json
from pathlib import Path
import threading

from .moe_types import (
    ExpertPerformanceMetrics,
    MoERoutingMetrics,
    MoEPerformanceSnapshot,
    UserFeedback,
    RetrainingTrigger,
    AdaptiveRoutingState,
    PERFORMANCE_THRESHOLDS
)

logger = logging.getLogger(__name__)


class MoEMetricsCollector:
    """Collects and manages MoE system performance metrics."""

    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.metrics = MoERoutingMetrics()
        self.expert_metrics: Dict[str, ExpertPerformanceMetrics] = {}
        self.feedback_history: List[UserFeedback] = []
        self.retraining_triggers: List[RetrainingTrigger] = []
        self.response_times: deque = deque(maxlen=max_history)
        self.routing_decisions: deque = deque(maxlen=max_history)
        self._lock = threading.Lock()

    def record_routing_decision(self, decision: 'RoutingDecision', response_time: float, success: bool = True):
        """Record a routing decision and its outcome."""
        with self._lock:
            self.metrics.total_requests += 1
            self.response_times.append(response_time)

            if decision.ensemble_mode:
                self.metrics.ensemble_usage_rate = (
                    (self.metrics.ensemble_usage_rate * (self.metrics.total_requests - 1) + 1) /
                    self.metrics.total_requests
                )
            else:
                self.metrics.ensemble_usage_rate = (
                    (self.metrics.ensemble_usage_rate * (self.metrics.total_requests - 1)) /
                    self.metrics.total_requests
                )

            if decision.fallback_used:
                self.metrics.fallback_rate = (
                    (self.metrics.fallback_rate * (self.metrics.total_requests - 1) + 1) /
                    self.metrics.total_requests
                )
            else:
                self.metrics.fallback_rate = (
                    (self.metrics.fallback_rate * (self.metrics.total_requests - 1)) /
                    self.metrics.total_requests
                )

            # Update expert usage distribution
            if decision.primary_expert not in self.metrics.expert_usage_distribution:
                self.metrics.expert_usage_distribution[decision.primary_expert] = 0
            self.metrics.expert_usage_distribution[decision.primary_expert] += 1

            # Update response time distribution
            rt_bucket = self._get_response_time_bucket(response_time)
            if rt_bucket not in self.metrics.response_time_distribution:
                self.metrics.response_time_distribution[rt_bucket] = 0
            self.metrics.response_time_distribution[rt_bucket] += 1

            # Calculate average confidence
            if self.routing_decisions:
                confidences = [d.confidence_score for d in self.routing_decisions if hasattr(d, 'confidence_score')]
                if confidences:
                    self.metrics.average_confidence = sum(confidences) / len(confidences)

            # Store decision for history
            self.routing_decisions.append(decision)

    def record_feedback(self, feedback: UserFeedback):
        """Record user feedback for adaptive routing."""
        with self._lock:
            self.feedback_history.append(feedback)

            # Keep only recent feedback
            if len(self.feedback_history) > self.max_history:
                self.feedback_history = self.feedback_history[-self.max_history:]

            # Update expert metrics
            expert_name = feedback.expert_used
            if expert_name not in self.expert_metrics:
                self.expert_metrics[expert_name] = ExpertPerformanceMetrics()

            metrics = self.expert_metrics[expert_name]
            metrics.user_satisfaction = (
                (metrics.user_satisfaction * metrics.sample_count + feedback.rating) /
                (metrics.sample_count + 1)
            )
            metrics.response_quality = (
                (metrics.response_quality * metrics.sample_count + feedback.response_quality) /
                (metrics.sample_count + 1)
            )
            metrics.sample_count += 1
            metrics.last_updated = time.time()

    def update_expert_performance(self, expert_name: str, accuracy: float, response_time: float):
        """Update performance metrics for an expert."""
        with self._lock:
            if expert_name not in self.expert_metrics:
                self.expert_metrics[expert_name] = ExpertPerformanceMetrics()

            metrics = self.expert_metrics[expert_name]
            metrics.accuracy = (
                (metrics.accuracy * metrics.sample_count + accuracy) /
                (metrics.sample_count + 1)
            )
            metrics.response_time = (
                (metrics.response_time * metrics.sample_count + response_time) /
                (metrics.sample_count + 1)
            )
            metrics.sample_count += 1
            metrics.last_updated = time.time()

    def check_retraining_needed(self, expert_name: str) -> Optional[RetrainingTrigger]:
        """Check if an expert needs retraining based on performance."""
        with self._lock:
            if expert_name not in self.expert_metrics:
                return None

            metrics = self.expert_metrics[expert_name]

            # Check if performance has dropped significantly
            if metrics.accuracy < metrics.baseline_performance - PERFORMANCE_THRESHOLDS["retraining_threshold"]:
                return RetrainingTrigger(
                    expert_name=expert_name,
                    trigger_reason="accuracy_degradation",
                    current_performance=metrics.accuracy,
                    threshold=metrics.baseline_performance - PERFORMANCE_THRESHOLDS["retraining_threshold"],
                    timestamp=time.time(),
                    recommended_action="full_retrain"
                )

            # Check if user satisfaction is too low
            if metrics.user_satisfaction < 0.5 and metrics.sample_count > 10:
                return RetrainingTrigger(
                    expert_name=expert_name,
                    trigger_reason="low_user_satisfaction",
                    current_performance=metrics.user_satisfaction,
                    threshold=0.5,
                    timestamp=time.time(),
                    recommended_action="targeted_retrain"
                )

            return None

    def record_retraining_trigger(self, trigger: RetrainingTrigger):
        """Record a retraining trigger."""
        with self._lock:
            self.retraining_triggers.append(trigger)

            # Update expert metrics
            if trigger.expert_name in self.expert_metrics:
                self.expert_metrics[trigger.expert_name].retraining_triggered = True

    def get_performance_snapshot(self) -> MoEPerformanceSnapshot:
        """Get a comprehensive performance snapshot."""
        with self._lock:
            # Calculate cache hit rate
            cache_hits = sum(1 for decision in self.routing_decisions if getattr(decision, 'cached', False))
            total_decisions = len(self.routing_decisions)
            cache_hit_rate = cache_hits / total_decisions if total_decisions > 0 else 0.0

            # Calculate adaptive routing improvements
            recent_feedback = [f for f in self.feedback_history if f.timestamp > time.time() - 3600]  # Last hour
            improvements = 0.0
            if recent_feedback:
                avg_satisfaction = sum(f.rating for f in recent_feedback) / len(recent_feedback)
                improvements = avg_satisfaction - 0.7  # Baseline satisfaction

            return MoEPerformanceSnapshot(
                timestamp=time.time(),
                metrics=MoERoutingMetrics(
                    total_requests=self.metrics.total_requests,
                    routing_accuracy=self.metrics.routing_accuracy,
                    average_confidence=self.metrics.average_confidence,
                    ensemble_usage_rate=self.metrics.ensemble_usage_rate,
                    fallback_rate=self.metrics.fallback_rate,
                    expert_usage_distribution=self.metrics.expert_usage_distribution.copy(),
                    response_time_distribution=self.metrics.response_time_distribution.copy(),
                    cache_hit_rate=cache_hit_rate,
                    adaptive_routing_improvements=improvements
                ),
                expert_performance=self.expert_metrics.copy(),
                system_health={
                    "memory_usage_mb": self._get_memory_usage(),
                    "active_threads": threading.active_count(),
                    "cache_size": len(self.routing_decisions),
                    "feedback_count": len(self.feedback_history)
                },
                recommendations=self._generate_recommendations()
            )

    def _get_response_time_bucket(self, response_time: float) -> str:
        """Categorize response time into buckets."""
        if response_time < 0.5:
            return "fast"
        elif response_time < 1.0:
            return "medium"
        elif response_time < 2.0:
            return "slow"
        else:
            return "very_slow"

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0

    def _generate_recommendations(self) -> List[str]:
        """Generate performance-based recommendations."""
        recommendations = []

        # Check for slow response times
        if self.response_times:
            avg_response_time = sum(self.response_times) / len(self.response_times)
            if avg_response_time > 2.0:
                recommendations.append("Consider optimizing model loading or reducing batch size")

        # Check for high fallback rate
        if self.metrics.fallback_rate > 0.2:
            recommendations.append("High fallback rate detected - check expert availability")

        # Check for low cache hit rate
        cache_hits = sum(1 for decision in self.routing_decisions if getattr(decision, 'cached', False))
        total_decisions = len(self.routing_decisions)
        if total_decisions > 0 and cache_hits / total_decisions < 0.3:
            recommendations.append("Low cache hit rate - consider increasing cache size or improving cache keys")

        # Check for expert performance issues
        for expert_name, metrics in self.expert_metrics.items():
            if metrics.accuracy < 0.6 and metrics.sample_count > 5:
                recommendations.append(f"Expert {expert_name} showing low accuracy - consider retraining")

        return recommendations

    def save_metrics(self, filepath: str):
        """Save metrics to JSON file."""
        snapshot = self.get_performance_snapshot()
        filepath = Path(filepath)

        # Create directory if it doesn't exist
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Convert to JSON-serializable format
        data = {
            "timestamp": snapshot.timestamp,
            "metrics": {
                "total_requests": snapshot.metrics.total_requests,
                "routing_accuracy": snapshot.metrics.routing_accuracy,
                "average_confidence": snapshot.metrics.average_confidence,
                "ensemble_usage_rate": snapshot.metrics.ensemble_usage_rate,
                "fallback_rate": snapshot.metrics.fallback_rate,
                "expert_usage_distribution": snapshot.metrics.expert_usage_distribution,
                "response_time_distribution": snapshot.metrics.response_time_distribution,
                "cache_hit_rate": snapshot.metrics.cache_hit_rate,
                "adaptive_routing_improvements": snapshot.metrics.adaptive_routing_improvements
            },
            "expert_performance": {
                name: {
                    "accuracy": metrics.accuracy,
                    "response_time": metrics.response_time,
                    "user_satisfaction": metrics.user_satisfaction,
                    "response_quality": metrics.response_quality,
                    "sample_count": metrics.sample_count,
                    "last_updated": metrics.last_updated
                }
                for name, metrics in snapshot.expert_performance.items()
            },
            "system_health": snapshot.system_health,
            "recommendations": snapshot.recommendations
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Metrics saved to {filepath}")

    def load_metrics(self, filepath: str) -> bool:
        """Load metrics from JSON file."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            # Restore metrics
            self.metrics.total_requests = data["metrics"]["total_requests"]
            self.metrics.routing_accuracy = data["metrics"]["routing_accuracy"]
            self.metrics.average_confidence = data["metrics"]["average_confidence"]
            self.metrics.ensemble_usage_rate = data["metrics"]["ensemble_usage_rate"]
            self.metrics.fallback_rate = data["metrics"]["fallback_rate"]
            self.metrics.expert_usage_distribution = data["metrics"]["expert_usage_distribution"]
            self.metrics.response_time_distribution = data["metrics"]["response_time_distribution"]
            self.metrics.cache_hit_rate = data["metrics"]["cache_hit_rate"]
            self.metrics.adaptive_routing_improvements = data["metrics"]["adaptive_routing_improvements"]

            # Restore expert metrics
            for name, expert_data in data["expert_performance"].items():
                self.expert_metrics[name] = ExpertPerformanceMetrics(**expert_data)

            logger.info(f"Metrics loaded from {filepath}")
            return True

        except Exception as e:
            logger.error(f"Failed to load metrics from {filepath}: {e}")
            return False
