#!/usr/bin/env python3
"""
Mixture of Experts Router for ChessGemma

True MoE implementation with automatic expert routing using modular design.
This file now imports from focused modules for better maintainability.

Features:
- Dynamic expert selection based on input characteristics
- Ensemble capabilities for complex queries
- Performance-aware routing
- Adaptive routing based on confidence scores
- Comprehensive monitoring and metrics
"""

import logging
import os
import time
from typing import Dict, List, Any, Optional, Tuple, Union, Callable, Set
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor

# Import from modular components
from .moe_types import (
    RouterTrainingExample,
    RouterTrainingDataset,
    RoutingDecision,
    UserFeedback,
    ExpertPerformanceMetrics,
    RetrainingTrigger,
    AdaptiveRoutingState,
    MoERoutingMetrics,
    MoEPerformanceSnapshot,
    MoETrainingConfig,
    EXPERT_NAMES,
    PERFORMANCE_THRESHOLDS
)
from .moe_metrics import MoEMetricsCollector
from .moe_router_core import ChessMoERouter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Legacy compatibility functions for backward compatibility
def _determine_feature_dim() -> int:
    """Return the fixed training embedding dimensionality."""
    return 32


def create_moe_system(expert_paths: Dict[str, str], inference_system=None) -> Tuple[ChessMoERouter, 'MoEInferenceManager']:
    """Create a complete MoE system with router and inference manager."""
    # Create router
    router = ChessMoERouter()

    # Create inference manager
    inference_manager = MoEInferenceManager(router, expert_paths, inference_system)

    return router, inference_manager


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
                    refresh_err,
                )

        for expert_name, model_path in expert_models.items():
            path_obj = Path(model_path) if model_path else None
            if path_obj and path_obj.exists():
                self._expert_adapter_paths[expert_name] = path_obj
                self._expert_ready[expert_name] = False
                logger.info(f"Registered {expert_name} expert adapter at {path_obj}")
            else:
                logger.warning("Expert model not found for %s: %s", expert_name, model_path)

        self.prime_available_experts()

        # Parallel processing configuration
        self._parallel_enabled = os.environ.get('CHESSGEMMA_PARALLEL_INFERENCE', '1') not in ('0', 'false', 'False')
        self._max_parallel_experts = max(1, min(len(self.expert_models), int(os.environ.get('CHESSGEMMA_MAX_PARALLEL_EXPERTS', '3'))))
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
        except Exception as e:
            logger.error(
                "Failed to prime %s expert: %s", expert_name, e
            )
        finally:
            # Restore previous adapter if there was one
            if active_adapter:
                try:
                    self.inference_system.set_active_adapter(active_adapter)
                except Exception:
                    pass

    def route_and_generate(self, question: str, position_fen: str = "", context: str = "") -> Dict[str, Any]:
        """Route query to appropriate expert and generate response."""
        start_time = time.time()

        # Route the query
        routing_decision = self.router.route_query(
            position_fen=position_fen,
            question_text=question,
            query_type="auto"
        )

        response_time = time.time() - start_time
        self._record_metrics(routing_decision, response_time)

        # Generate response using selected expert
        response = self._generate_expert_response(
            routing_decision.primary_expert,
            question,
            position_fen,
            context
        )

        return {
            "response": response,
            "routing_decision": routing_decision,
            "response_time": response_time,
            "expert_used": routing_decision.primary_expert
        }

    def analyze_position(self, fen: str, query_type: str = "auto") -> Dict[str, Any]:
        """Route a position-focused query and expose routing metadata."""
        start_time = time.time()

        routing_decision = self.router.route_query(
            fen,
            query_type=query_type,
            question_text="",
        )

        response_time = time.time() - start_time
        self._record_metrics(routing_decision, response_time)

        routing_info = {
            "primary_expert": routing_decision.primary_expert,
            "confidence": routing_decision.confidence_score,
            "fallback_used": getattr(routing_decision, "fallback_used", False),
            "ensemble_mode": routing_decision.ensemble_mode,
            "weights": routing_decision.expert_weights,
            "reasoning": routing_decision.reasoning,
        }

        payload: Dict[str, Any] = {
            "routing_info": routing_info,
            "response_time": response_time,
        }

        if self.inference_system:
            try:
                payload["response"] = self._generate_expert_response(
                    routing_decision.primary_expert,
                    question="",
                    position_fen=fen,
                    context="",
                )
            except Exception as exc:
                payload["response_error"] = str(exc)

        return payload

    def _record_metrics(self, decision: RoutingDecision, response_time: float) -> None:
        """Lightweight metrics tracking without requiring the full collector."""
        if not self.metrics:
            return

        metrics = self.metrics
        previous_total = metrics.total_requests
        metrics.total_requests += 1

        confidence = getattr(decision, "confidence_score", 0.0) or 0.0
        fallback = 1.0 if getattr(decision, "fallback_used", False) else 0.0
        ensemble = 1.0 if getattr(decision, "ensemble_mode", False) else 0.0

        if metrics.total_requests:
            metrics.average_confidence = (
                (metrics.average_confidence * previous_total + confidence) / metrics.total_requests
            )
            metrics.fallback_rate = (
                (metrics.fallback_rate * previous_total + fallback) / metrics.total_requests
            )
            metrics.ensemble_usage_rate = (
                (metrics.ensemble_usage_rate * previous_total + ensemble) / metrics.total_requests
            )

        metrics.expert_usage_distribution.setdefault(decision.primary_expert, 0)
        metrics.expert_usage_distribution[decision.primary_expert] += 1

        bucket = (
            "under_0.5s" if response_time < 0.5
            else "under_1s" if response_time < 1.0
            else "over_1s"
        )
        metrics.response_time_distribution[bucket] = metrics.response_time_distribution.get(bucket, 0) + 1

    def _generate_expert_response(self, expert_name: str, question: str, position_fen: str, context: str) -> str:
        """Generate response using the specified expert."""
        if expert_name not in self._expert_ready:
            logger.warning(f"Expert {expert_name} not ready for inference")
            return f"Expert {expert_name} is not available for inference."

        # Set active adapter for this expert
        try:
            self.inference_system.set_active_adapter(expert_name)
        except Exception as e:
            logger.error(f"Failed to set active adapter {expert_name}: {e}")
            return f"Failed to activate {expert_name} expert: {e}"

        # Generate response using the inference system
        try:
            if hasattr(self.inference_system, "generate_response"):
                return self.inference_system.generate_response(
                    question,
                    position_fen=position_fen,
                    context=context
                )
            else:
                return f"Expert {expert_name} response generation not implemented."
        except Exception as e:
            logger.error(f"Failed to generate response with {expert_name}: {e}")
            return f"Error generating response with {expert_name}: {e}"

    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status."""
        health = {
            "experts_ready": sum(1 for ready in self._expert_ready.values() if ready),
            "total_experts": len(self._expert_ready),
            "router_stats": self.router.get_routing_stats(),
            "metrics": {
                "total_requests": self.metrics.total_requests,
                "average_confidence": self.metrics.average_confidence,
                "ensemble_usage_rate": self.metrics.ensemble_usage_rate,
                "fallback_rate": self.metrics.fallback_rate
            }
        }

        # Add expert-specific health
        for expert_name, ready in self._expert_ready.items():
            health[f"expert_{expert_name}_ready"] = ready

        return health

    def shutdown(self):
        """Shutdown the inference manager and clean up resources."""
        if hasattr(self, '_parallel_executor'):
            self._parallel_executor.shutdown(wait=True)
        logger.info("MoE Inference Manager shutdown complete")
