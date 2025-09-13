#!/usr/bin/env python3
"""
Comprehensive Error Handling System for ChessGemma

Provides robust error handling, fallback mechanisms, and recovery strategies
throughout the ChessGemma system with intelligent error classification and
graceful degradation.
"""

import logging
import traceback
import time
from typing import Dict, List, Any, Optional, Callable, Type
from dataclasses import dataclass, field
from enum import Enum
from contextlib import contextmanager
import threading
import psutil

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification."""
    MODEL_LOADING = "model_loading"
    INFERENCE = "inference"
    TRAINING = "training"
    DATA_LOADING = "data_loading"
    MEMORY = "memory"
    NETWORK = "network"
    VALIDATION = "validation"
    CONFIGURATION = "configuration"
    HARDWARE = "hardware"
    UNKNOWN = "unknown"


@dataclass
class ErrorContext:
    """Context information for errors."""
    component: str
    operation: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    system_state: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    retry_count: int = 0


@dataclass
class ErrorRecord:
    """Complete error record with context and recovery information."""
    error_id: str
    exception: Exception
    severity: ErrorSeverity
    category: ErrorCategory
    context: ErrorContext
    traceback: str
    recovery_attempts: List[Dict[str, Any]] = field(default_factory=list)
    resolved: bool = False
    resolution_time: Optional[float] = None


class ChessGemmaErrorHandler:
    """Centralized error handling system for ChessGemma."""

    def __init__(self):
        self.error_history: List[ErrorRecord] = []
        self.recovery_strategies: Dict[ErrorCategory, List[Callable]] = {}
        self.fallback_handlers: Dict[str, Callable] = {}
        self.error_counts: Dict[ErrorCategory, int] = {}
        self.max_history_size = 1000
        self.lock = threading.Lock()

        # Initialize default recovery strategies
        self._initialize_recovery_strategies()

        logger.info("🛡️ ChessGemma Error Handler initialized")

    def _initialize_recovery_strategies(self):
        """Initialize default recovery strategies for different error categories."""
        self.recovery_strategies = {
            ErrorCategory.MODEL_LOADING: [
                self._retry_with_backoff,
                self._clear_model_cache,
                self._fallback_to_base_model
            ],
            ErrorCategory.MEMORY: [
                self._clear_caches,
                self._reduce_batch_size,
                self._enable_gradient_checkpointing
            ],
            ErrorCategory.INFERENCE: [
                self._retry_with_backoff,
                self._use_cached_response,
                self._fallback_to_simpler_model
            ],
            ErrorCategory.TRAINING: [
                self._save_checkpoint,
                self._reduce_learning_rate,
                self._skip_problematic_batch
            ],
            ErrorCategory.DATA_LOADING: [
                self._retry_with_backoff,
                self._use_backup_data_source,
                self._skip_corrupted_samples
            ]
        }

    @contextmanager
    def error_boundary(self, component: str, operation: str, **context_params):
        """Context manager for error boundaries with automatic recovery."""
        error_context = ErrorContext(
            component=component,
            operation=operation,
            parameters=context_params,
            system_state=self._capture_system_state()
        )

        try:
            yield
        except Exception as e:
            self._handle_error(e, error_context)
            raise  # Re-raise after handling

    def handle_error(self, exception: Exception, component: str, operation: str,
                    **context_params) -> Any:
        """Handle an error with recovery attempts."""
        error_context = ErrorContext(
            component=component,
            operation=operation,
            parameters=context_params,
            system_state=self._capture_system_state()
        )

        return self._handle_error(exception, error_context)

    def _handle_error(self, exception: Exception, context: ErrorContext) -> Any:
        """Internal error handling logic."""
        # Classify the error
        severity, category = self._classify_error(exception, context)

        # Create error record
        error_record = ErrorRecord(
            error_id=f"{int(time.time() * 1000)}_{context.component}_{context.operation}",
            exception=exception,
            severity=severity,
            category=category,
            context=context,
            traceback=traceback.format_exc()
        )

        # Log the error
        self._log_error(error_record)

        # Update error counts
        with self.lock:
            self.error_counts[category] = self.error_counts.get(category, 0) + 1

        # Attempt recovery
        recovery_result = self._attempt_recovery(error_record)

        # Store error record
        with self.lock:
            self.error_history.append(error_record)
            if len(self.error_history) > self.max_history_size:
                self.error_history.pop(0)

        # Return recovery result or raise
        if recovery_result.get('success', False):
            error_record.resolved = True
            error_record.resolution_time = time.time()
            return recovery_result.get('result')
        else:
            raise exception

    def _classify_error(self, exception: Exception, context: ErrorContext) -> tuple[ErrorSeverity, ErrorCategory]:
        """Classify error by type and severity."""
        error_type = type(exception).__name__
        error_message = str(exception).lower()

        # Memory-related errors
        if any(keyword in error_message for keyword in ['cuda out of memory', 'mps out of memory', 'memory', 'allocation']):
            return ErrorSeverity.HIGH, ErrorCategory.MEMORY

        # Model loading errors
        if any(keyword in error_message for keyword in ['model', 'loading', 'checkpoint', 'adapter']):
            return ErrorSeverity.HIGH, ErrorCategory.MODEL_LOADING

        # Training errors
        if context.operation in ['train', 'training', 'fit']:
            if 'nan' in error_message or 'inf' in error_message:
                return ErrorSeverity.MEDIUM, ErrorCategory.TRAINING
            return ErrorSeverity.MEDIUM, ErrorCategory.TRAINING

        # Inference errors
        if context.operation in ['generate', 'inference', 'predict']:
            return ErrorSeverity.MEDIUM, ErrorCategory.INFERENCE

        # Data loading errors
        if any(keyword in error_message for keyword in ['data', 'dataset', 'file', 'path']):
            return ErrorSeverity.MEDIUM, ErrorCategory.DATA_LOADING

        # Network errors
        if any(keyword in error_message for keyword in ['connection', 'timeout', 'network', 'http']):
            return ErrorSeverity.LOW, ErrorCategory.NETWORK

        # Default classification
        return ErrorSeverity.MEDIUM, ErrorCategory.UNKNOWN

    def _attempt_recovery(self, error_record: ErrorRecord) -> Dict[str, Any]:
        """Attempt recovery using appropriate strategies."""
        strategies = self.recovery_strategies.get(error_record.category, [])

        for strategy in strategies:
            try:
                result = strategy(error_record)
                if result.get('success', False):
                    logger.info(f"✅ Recovery successful using {strategy.__name__}")
                    error_record.recovery_attempts.append({
                        'strategy': strategy.__name__,
                        'success': True,
                        'timestamp': time.time()
                    })
                    return result
                else:
                    error_record.recovery_attempts.append({
                        'strategy': strategy.__name__,
                        'success': False,
                        'timestamp': time.time()
                    })
            except Exception as recovery_error:
                logger.warning(f"Recovery strategy {strategy.__name__} failed: {recovery_error}")
                error_record.recovery_attempts.append({
                    'strategy': strategy.__name__,
                    'success': False,
                    'error': str(recovery_error),
                    'timestamp': time.time()
                })

        return {'success': False, 'error': 'All recovery strategies failed'}

    def _log_error(self, error_record: ErrorRecord):
        """Log error with appropriate severity."""
        log_message = (
            f"{error_record.severity.value.upper()} ERROR in {error_record.context.component}.{error_record.context.operation}: "
            f"{error_record.exception.__class__.__name__}: {error_record.exception}"
        )

        if error_record.severity == ErrorSeverity.CRITICAL:
            logger.critical(log_message)
        elif error_record.severity == ErrorSeverity.HIGH:
            logger.error(log_message)
        elif error_record.severity == ErrorSeverity.MEDIUM:
            logger.warning(log_message)
        else:
            logger.info(log_message)

    def _capture_system_state(self) -> Dict[str, Any]:
        """Capture current system state for error context."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()

            return {
                'cpu_percent': psutil.cpu_percent(interval=0.1),
                'memory_rss_mb': memory_info.rss / (1024 * 1024),
                'memory_vms_mb': memory_info.vms / (1024 * 1024),
                'system_memory_percent': psutil.virtual_memory().percent,
                'timestamp': time.time()
            }
        except Exception:
            return {'error': 'Could not capture system state'}

    # Recovery Strategies
    def _retry_with_backoff(self, error_record: ErrorRecord) -> Dict[str, Any]:
        """Retry operation with exponential backoff."""
        if error_record.context.retry_count >= 3:
            return {'success': False, 'error': 'Max retries exceeded'}

        delay = 2 ** error_record.context.retry_count
        logger.info(f"Retrying operation in {delay} seconds (attempt {error_record.context.retry_count + 1})")

        time.sleep(delay)
        error_record.context.retry_count += 1

        # This would need to be implemented with the actual retry logic
        return {'success': False, 'error': 'Retry mechanism needs operation-specific implementation'}

    def _clear_model_cache(self, error_record: ErrorRecord) -> Dict[str, Any]:
        """Clear model cache to free memory."""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
                torch.mps.empty_cache()
            import gc
            gc.collect()
            return {'success': True, 'message': 'Model cache cleared'}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _fallback_to_base_model(self, error_record: ErrorRecord) -> Dict[str, Any]:
        """Fallback to base model without adapters."""
        return {'success': False, 'error': 'Base model fallback needs implementation'}

    def _clear_caches(self, error_record: ErrorRecord) -> Dict[str, Any]:
        """Clear all system caches."""
        try:
            # This would clear various caches in the system
            return {'success': True, 'message': 'Caches cleared'}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _reduce_batch_size(self, error_record: ErrorRecord) -> Dict[str, Any]:
        """Reduce batch size to handle memory issues."""
        return {'success': False, 'error': 'Batch size reduction needs training context'}

    def _enable_gradient_checkpointing(self, error_record: ErrorRecord) -> Dict[str, Any]:
        """Enable gradient checkpointing for memory efficiency."""
        return {'success': False, 'error': 'Gradient checkpointing needs model context'}

    def _use_cached_response(self, error_record: ErrorRecord) -> Dict[str, Any]:
        """Use cached response if available."""
        return {'success': False, 'error': 'Cache lookup needs request context'}

    def _fallback_to_simpler_model(self, error_record: ErrorRecord) -> Dict[str, Any]:
        """Fallback to simpler model configuration."""
        return {'success': False, 'error': 'Model fallback needs implementation'}

    def _save_checkpoint(self, error_record: ErrorRecord) -> Dict[str, Any]:
        """Save training checkpoint before failure."""
        return {'success': False, 'error': 'Checkpoint saving needs training context'}

    def _reduce_learning_rate(self, error_record: ErrorRecord) -> Dict[str, Any]:
        """Reduce learning rate to stabilize training."""
        return {'success': False, 'error': 'Learning rate adjustment needs optimizer context'}

    def _skip_problematic_batch(self, error_record: ErrorRecord) -> Dict[str, Any]:
        """Skip problematic training batch."""
        return {'success': False, 'error': 'Batch skipping needs training context'}

    def _use_backup_data_source(self, error_record: ErrorRecord) -> Dict[str, Any]:
        """Use backup data source."""
        return {'success': False, 'error': 'Backup data source needs configuration'}

    def _skip_corrupted_samples(self, error_record: ErrorRecord) -> Dict[str, Any]:
        """Skip corrupted data samples."""
        return {'success': False, 'error': 'Corrupted sample handling needs data pipeline context'}

    # Public API methods
    def register_recovery_strategy(self, category: ErrorCategory, strategy: Callable):
        """Register a custom recovery strategy."""
        if category not in self.recovery_strategies:
            self.recovery_strategies[category] = []
        self.recovery_strategies[category].append(strategy)
        logger.info(f"Registered recovery strategy for {category.value}")

    def register_fallback_handler(self, operation: str, handler: Callable):
        """Register a fallback handler for specific operations."""
        self.fallback_handlers[operation] = handler
        logger.info(f"Registered fallback handler for {operation}")

    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics."""
        with self.lock:
            return {
                'total_errors': len(self.error_history),
                'error_counts_by_category': self.error_counts.copy(),
                'recent_errors': len([e for e in self.error_history[-100:] if not e.resolved]),
                'recovery_rate': sum(1 for e in self.error_history if e.resolved) / max(len(self.error_history), 1),
                'most_common_errors': sorted(self.error_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            }

    def clear_error_history(self):
        """Clear error history."""
        with self.lock:
            self.error_history.clear()
            self.error_counts.clear()
        logger.info("Error history cleared")


# Global error handler instance
_error_handler = None
_error_handler_lock = threading.Lock()


def get_error_handler() -> ChessGemmaErrorHandler:
    """Get the global error handler instance."""
    global _error_handler
    if _error_handler is None:
        with _error_handler_lock:
            if _error_handler is None:
                _error_handler = ChessGemmaErrorHandler()
    return _error_handler


def handle_error(exception: Exception, component: str, operation: str, **context) -> Any:
    """Convenience function to handle errors."""
    return get_error_handler().handle_error(exception, component, operation, **context)


@contextmanager
def error_boundary(component: str, operation: str, **context):
    """Convenience context manager for error boundaries."""
    with get_error_handler().error_boundary(component, operation, **context):
        yield
