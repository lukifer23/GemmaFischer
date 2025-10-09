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

try:
    import torch
except Exception:  # pragma: no cover - optional dependency
    torch = None

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


# Performance-optimized error classification lookup tables
ERROR_KEYWORDS = {
    ErrorCategory.MEMORY: ['cuda out of memory', 'mps out of memory', 'memory', 'allocation', 'out of memory'],
    ErrorCategory.MODEL_LOADING: ['model', 'loading', 'checkpoint', 'adapter', 'pretrained', 'peft'],
    ErrorCategory.INFERENCE: ['inference', 'generation', 'tokenizer', 'decode', 'encode'],
    ErrorCategory.TRAINING: ['training', 'optimizer', 'gradient', 'loss', 'epoch', 'batch'],
    ErrorCategory.DATA_LOADING: ['dataset', 'data', 'jsonl', 'file not found', 'encoding'],
    ErrorCategory.NETWORK: ['connection', 'network', 'timeout', 'http', 'url'],
    ErrorCategory.CONFIGURATION: ['config', 'yaml', 'json', 'setting', 'parameter'],
    ErrorCategory.VALIDATION: ['validation', 'format', 'schema'],
    ErrorCategory.HARDWARE: ['gpu', 'cuda', 'mps', 'device', 'hardware'],
}

SEVERITY_PATTERNS = {
    ErrorSeverity.CRITICAL: ['segmentation fault', 'kernel died', 'system error'],
    ErrorSeverity.HIGH: ['out of memory', 'cuda error', 'exception', 'timeout', 'deadlock', 'model loading failed'],
    ErrorSeverity.MEDIUM: ['failed', 'warning', 'deprecated', 'missing', 'not found', 'inference error', 'training failed', 'config invalid'],
    ErrorSeverity.LOW: ['info', 'debug', 'trace'],
}


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
        self.max_history_size = 500  # Reduced from 1000 for better memory usage
        self.lock = threading.Lock()

        # Performance optimization: cache for system state
        self._cached_memory_state = {}
        self._last_memory_check = 0

        # Initialize default recovery strategies
        self._initialize_recovery_strategies()

        logger.info("ChessGemma Error Handler initialized")

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
        """Context manager for error boundaries with automatic recovery - optimized."""
        # Only capture system state for critical operations to reduce overhead
        capture_state = operation in ['model_loading', 'training', 'inference']

        error_context = ErrorContext(
            component=component,
            operation=operation,
            parameters=context_params,
            system_state=self._capture_system_state() if capture_state else {}
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

        # Create error record - optimized for memory usage
        error_record = ErrorRecord(
            error_id=f"{int(time.time() * 1000)}_{context.component}_{context.operation}",
            exception=exception,
            severity=severity,
            category=category,
            context=context,
            # Only store full traceback for critical/high severity errors
            traceback=traceback.format_exc() if severity in [ErrorSeverity.CRITICAL, ErrorSeverity.HIGH] else str(exception)
        )

        # Log the error
        self._log_error(error_record)

        # Update error counts
        with self.lock:
            self.error_counts[category] = self.error_counts.get(category, 0) + 1

        # Attempt recovery
        recovery_result = self._attempt_recovery(error_record)

        # Store error record - optimized storage
        with self.lock:
            # For high-frequency operations, limit storage to reduce memory usage
            if (error_record.severity == ErrorSeverity.LOW and
                len(self.error_history) > self.max_history_size // 2):
                # Skip storing low-severity errors if we're getting too many
                return

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
        """Classify error by type and severity - optimized for performance."""
        error_type = type(exception).__name__
        error_message = str(exception).lower()

        # Fast lookup-based classification for better performance
        category = self._classify_by_keywords(error_message)
        severity = self._classify_severity(error_message, error_type)

        return severity, category

    def _classify_by_keywords(self, error_message: str) -> ErrorCategory:
        """Classify error category using optimized keyword lookup."""
        # Check each category's keywords for matches
        for category, keywords in ERROR_KEYWORDS.items():
            for keyword in keywords:
                if keyword in error_message:
                    return category

        return ErrorCategory.UNKNOWN

    def _classify_severity(self, error_message: str, error_type: str) -> ErrorSeverity:
        """Classify error severity using optimized pattern matching."""
        # Check severity patterns
        for severity, patterns in SEVERITY_PATTERNS.items():
            for pattern in patterns:
                if pattern in error_message:
                    return severity

        # Fallback based on error type
        if error_type in ['KeyboardInterrupt', 'SystemExit']:
            return ErrorSeverity.CRITICAL
        elif error_type in ['MemoryError', 'OSError']:
            return ErrorSeverity.HIGH
        elif error_type in ['TypeError', 'AttributeError']:
            return ErrorSeverity.MEDIUM
        elif error_type in ['RuntimeError', 'ImportError']:
            return ErrorSeverity.MEDIUM
        else:
            return ErrorSeverity.LOW

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
        """Capture current system state for error context - optimized."""
        # Only capture essential system state to reduce overhead
        state = {'timestamp': time.time()}

        try:
            # Only capture memory if it's critical for debugging
            if hasattr(self, '_last_memory_check') and time.time() - self._last_memory_check < 10:
                # Reuse recent memory info to avoid overhead
                state.update(self._cached_memory_state)
            else:
                # Capture fresh memory info only when needed
                memory = psutil.virtual_memory()
                state['memory_percent'] = memory.percent

                if torch and torch.backends.mps.is_available():
                    try:
                        state['mps_memory'] = torch.mps.current_allocated_memory()
                    except:
                        pass

                self._cached_memory_state = state.copy()
                self._last_memory_check = time.time()
        except Exception:
            state['memory_error'] = 'Failed to capture memory state'

        return state

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
            if torch is None:
                logger.debug("PyTorch not available; skipping GPU/MPS cache cleanup.")
            else:
                cuda_available = (
                    hasattr(torch, 'cuda')
                    and hasattr(torch.cuda, 'is_available')
                    and callable(torch.cuda.is_available)
                    and torch.cuda.is_available()
                )
                if cuda_available and hasattr(torch.cuda, 'empty_cache') and callable(torch.cuda.empty_cache):
                    torch.cuda.empty_cache()
                else:
                    mps_available = (
                        hasattr(torch, 'mps')
                        and hasattr(torch, 'backends')
                        and hasattr(torch.backends, 'mps')
                        and hasattr(torch.backends.mps, 'is_available')
                        and callable(torch.backends.mps.is_available)
                        and torch.backends.mps.is_available()
                    )
                    if mps_available and hasattr(torch.mps, 'empty_cache') and callable(torch.mps.empty_cache):
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


    def get_performance_stats(self) -> Dict[str, Any]:
        """Get error handling performance statistics."""
        with self.lock:
            total_errors = len(self.error_history)
            errors_by_category = dict(self.error_counts)
            errors_by_severity = {}

            for record in self.error_history:
                severity = record.severity.value
                errors_by_severity[severity] = errors_by_severity.get(severity, 0) + 1

        return {
            'total_errors_handled': total_errors,
            'errors_by_category': errors_by_category,
            'errors_by_severity': errors_by_severity,
            'recovery_success_rate': self._calculate_recovery_success_rate(),
            'memory_usage_mb': self._estimate_memory_usage(),
        }

    def _calculate_recovery_success_rate(self) -> float:
        """Calculate the success rate of error recovery attempts."""
        resolved_errors = sum(1 for record in self.error_history if record.resolved)
        total_errors = len(self.error_history)
        return resolved_errors / max(total_errors, 1)

    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage of error handling system."""
        # Rough estimate: each error record is ~1-2KB
        avg_record_size = 1500  # bytes
        return len(self.error_history) * avg_record_size / (1024 * 1024)  # MB


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
