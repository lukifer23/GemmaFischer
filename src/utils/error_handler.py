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
import inspect
import gc
from typing import Dict, List, Any, Optional, Callable, Iterable
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
        self.fallback_handlers: Dict[str, List[Callable]] = {}
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

    def _extract_callables(self, candidate: Any) -> List[Callable]:
        """Normalize callable definitions to a flat list."""
        if candidate is None:
            return []
        if callable(candidate):
            return [candidate]
        if isinstance(candidate, (list, tuple, set)):
            callables: List[Callable] = []
            for item in candidate:
                callables.extend(self._extract_callables(item))
            return callables
        if isinstance(candidate, dict):
            callables: List[Callable] = []
            for item in candidate.values():
                callables.extend(self._extract_callables(item))
            return callables
        return []

    def _resolve_handler_list(self, keys: Iterable[str]) -> List[Callable]:
        """Resolve registered fallback handlers for the provided keys."""
        handlers: List[Callable] = []
        for key in keys:
            if not key:
                continue
            handlers.extend(self._extract_callables(self.fallback_handlers.get(key)))
        return handlers

    def _deduplicate_callables(self, handlers: Iterable[Callable]) -> List[Callable]:
        """Remove duplicate callables while preserving order."""
        seen: set[str] = set()
        unique_handlers: List[Callable] = []
        for handler in handlers:
            if not callable(handler):
                continue
            identifier = getattr(handler, '__qualname__', getattr(handler, '__name__', repr(handler)))
            if identifier in seen:
                continue
            seen.add(identifier)
            unique_handlers.append(handler)
        return unique_handlers

    def _call_handler(self, handler: Callable, error_record: ErrorRecord, *args, **kwargs) -> Any:
        """Safely invoke a handler with optional error record context."""
        try:
            signature = inspect.signature(handler)
        except (TypeError, ValueError):
            signature = None

        if signature is None:
            try:
                return handler(error_record, *args, **kwargs)
            except TypeError:
                return handler(*args, **kwargs)

        parameters = signature.parameters
        if parameters:
            first_param = next(iter(parameters.values()))
            if first_param.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.VAR_POSITIONAL
            ):
                return handler(error_record, *args, **kwargs)
        return handler(*args, **kwargs)

    def _normalize_success_result(self, result: Any, *, message: Optional[str] = None, **extra: Any) -> Dict[str, Any]:
        """Normalize handler results to the expected dictionary format."""
        if isinstance(result, dict):
            normalized = result.copy()
            normalized.setdefault('success', True)
            if message and 'message' not in normalized:
                normalized['message'] = message
            if extra:
                normalized.update(extra)
            return normalized

        payload: Dict[str, Any] = {'success': True}
        if result is not None:
            payload['result'] = result
        if message:
            payload['message'] = message
        if extra:
            payload.update(extra)
        return payload

    def _format_failure_result(self, error_message: str, **extra: Any) -> Dict[str, Any]:
        """Normalize failure payloads."""
        payload: Dict[str, Any] = {'success': False, 'error': error_message}
        if extra:
            payload.update(extra)
        return payload

    def _update_training_state(self, params: Dict[str, Any], key: str, value: Any, error_record: ErrorRecord):
        """Update training state dictionaries and notify any callback."""
        training_state = params.get('training_state')
        if isinstance(training_state, dict):
            training_state[key] = value

        update_fn = params.get('update_training_state')
        if not callable(update_fn):
            return

        try:
            update_fn(key, value, error_record=error_record)
        except TypeError:
            try:
                update_fn(key, value)
            except TypeError:
                update_fn({key: value})

    # Recovery Strategies
    def _retry_with_backoff(self, error_record: ErrorRecord) -> Dict[str, Any]:
        """Retry operation with configurable exponential backoff and fallbacks."""
        params = error_record.context.parameters

        max_retries = params.get('max_retries', params.get('retry_max_retries', 3))
        try:
            max_retries = int(max_retries)
        except (TypeError, ValueError):
            max_retries = 3
        max_retries = max(0, max_retries)

        base_delay = params.get('retry_base_delay', params.get('retry_delay', 0.1))
        try:
            base_delay = float(base_delay)
        except (TypeError, ValueError):
            base_delay = 0.0

        max_delay = params.get('retry_max_delay', 2.0)
        try:
            max_delay = float(max_delay)
        except (TypeError, ValueError):
            max_delay = 2.0

        disable_sleep = bool(params.get('retry_disable_sleep', False))
        sleep_fn = params.get('retry_sleep_fn', time.sleep)
        if not callable(sleep_fn):
            sleep_fn = time.sleep

        primary_candidates: List[Callable] = []
        for key in ('retry_callback', 'operation_callable', 'retry_operation'):
            primary_candidates.extend(self._extract_callables(params.get(key)))
        primary_candidates.extend(self._extract_callables(params.get('retry_callables')))

        handler_keys: List[str] = []
        if 'retry_handler_keys' in params:
            custom_keys = params['retry_handler_keys']
            if isinstance(custom_keys, str):
                handler_keys.append(custom_keys)
            else:
                handler_keys.extend(list(custom_keys))

        operation_name = error_record.context.operation
        component_name = error_record.context.component
        handler_keys.extend([
            f"{component_name}.{operation_name}",
            operation_name,
            f"{operation_name}.retry",
            f"{component_name}.retry",
            'default_retry'
        ])
        fallback_candidates = self._resolve_handler_list(handler_keys)

        primary_candidates = self._deduplicate_callables(primary_candidates)
        fallback_candidates = self._deduplicate_callables(fallback_candidates)

        if not primary_candidates and not fallback_candidates:
            return self._format_failure_result('No retry operations registered')

        args_candidate = params.get('retry_args', params.get('operation_args', ()))
        if isinstance(args_candidate, (list, tuple)):
            operation_args = tuple(args_candidate)
        elif args_candidate is None:
            operation_args = ()
        else:
            operation_args = (args_candidate,)

        operation_kwargs: Dict[str, Any] = {}
        for key in ('operation_kwargs', 'retry_kwargs'):
            candidate_kwargs = params.get(key)
            if isinstance(candidate_kwargs, dict):
                operation_kwargs.update(candidate_kwargs)

        last_exception: Optional[Exception] = None
        current_retry_count = error_record.context.retry_count
        attempts_allowed = max(0, max_retries - current_retry_count)

        if primary_candidates and attempts_allowed > 0:
            primary = primary_candidates[0]
            logger.info(
                "Retrying %s.%s with backoff (remaining attempts: %s)",
                component_name,
                operation_name,
                attempts_allowed
            )
            for attempt_index in range(current_retry_count, current_retry_count + attempts_allowed):
                if attempt_index > current_retry_count and not disable_sleep:
                    delay = min(max_delay, base_delay * (2 ** (attempt_index - current_retry_count)))
                    if delay > 0:
                        try:
                            sleep_fn(delay)
                        except Exception:
                            pass

                try:
                    result = primary(*operation_args, **operation_kwargs)
                    error_record.context.retry_count = attempt_index + 1
                    logger.info(
                        "✅ Retry succeeded for %s.%s on attempt %s",
                        component_name,
                        operation_name,
                        error_record.context.retry_count
                    )
                    return self._normalize_success_result(
                        result,
                        message='Operation retried successfully',
                        retry_attempts=error_record.context.retry_count
                    )
                except Exception as exc:
                    last_exception = exc
                    error_record.context.parameters['last_retry_error'] = str(exc)
                    error_record.context.retry_count = attempt_index + 1
                    logger.debug(
                        "Retry attempt %s for %s.%s failed: %s",
                        error_record.context.retry_count,
                        component_name,
                        operation_name,
                        exc
                    )

            primary_candidates = primary_candidates[1:]

        combined_fallbacks = primary_candidates + fallback_candidates

        for handler in combined_fallbacks:
            try:
                result = self._call_handler(handler, error_record)
                logger.info(
                    "Fallback handler %s resolved %s.%s",
                    getattr(handler, '__name__', repr(handler)),
                    component_name,
                    operation_name
                )
                return self._normalize_success_result(
                    result,
                    message='Fallback handler executed after retry failure'
                )
            except Exception as exc:
                last_exception = exc
                error_record.context.parameters['last_retry_error'] = str(exc)
                logger.debug(
                    "Fallback handler %s failed for %s.%s: %s",
                    getattr(handler, '__name__', repr(handler)),
                    component_name,
                    operation_name,
                    exc
                )

        if last_exception:
            logger.error(
                "Retry strategy exhausted for %s.%s: %s",
                component_name,
                operation_name,
                last_exception
            )
        else:
            logger.error(
                "Retry strategy exhausted for %s.%s with no handlers succeeding",
                component_name,
                operation_name
            )

        error_message = str(last_exception) if last_exception else 'Retry operation failed'
        return self._format_failure_result(
            error_message,
            retry_attempts=error_record.context.retry_count,
            operation=operation_name
        )

    def _clear_model_cache(self, error_record: ErrorRecord) -> Dict[str, Any]:
        """Clear model cache to free memory using available hooks."""
        params = error_record.context.parameters
        operations_performed = 0
        warnings: List[str] = []

        torch_module = None
        try:
            import torch as torch_module  # type: ignore
        except Exception as exc:
            torch_module = None
            if params.get('require_torch_for_cache', False):
                warnings.append(str(exc))

        if torch_module is not None:
            try:
                if torch_module.cuda.is_available():
                    torch_module.cuda.empty_cache()
                    operations_performed += 1
            except Exception as exc:
                warnings.append(str(exc))

            try:
                if hasattr(torch_module, 'mps') and torch_module.backends.mps.is_available():
                    torch_module.mps.empty_cache()
                    operations_performed += 1
            except Exception as exc:
                warnings.append(str(exc))

        try:
            gc.collect()
            operations_performed += 1
        except Exception as exc:
            warnings.append(str(exc))

        for hook in self._extract_callables(params.get('model_cache_hooks')):
            try:
                self._call_handler(hook, error_record)
                operations_performed += 1
            except Exception as exc:
                warnings.append(str(exc))

        handler_keys = [
            f"{error_record.context.component}.clear_model_cache",
            f"{error_record.context.operation}.clear_model_cache",
            'clear_model_cache'
        ]
        for handler in self._resolve_handler_list(handler_keys):
            try:
                self._call_handler(handler, error_record)
                operations_performed += 1
            except Exception as exc:
                warnings.append(str(exc))

        if operations_performed:
            payload = {
                'success': True,
                'message': 'Model cache cleared',
                'operations': operations_performed
            }
            if warnings:
                payload['warnings'] = warnings
            return payload

        if warnings:
            return self._format_failure_result('No model cache operations executed', warnings=warnings)
        return self._format_failure_result('No model cache operations executed')

    def _fallback_to_base_model(self, error_record: ErrorRecord) -> Dict[str, Any]:
        """Fallback to base model without adapters."""
        params = error_record.context.parameters

        handlers = []
        handlers.extend(self._extract_callables(params.get('base_model_loader')))
        handlers.extend(self._extract_callables(params.get('load_base_model')))

        handler_keys = [
            f"{error_record.context.component}.base_model",
            f"{error_record.context.operation}.base_model",
            'base_model',
            'fallback_to_base_model'
        ]
        handlers.extend(self._resolve_handler_list(handler_keys))
        handlers = self._deduplicate_callables(handlers)

        if not handlers:
            return self._format_failure_result('No base model fallback registered')

        last_error: Optional[Exception] = None
        for handler in handlers:
            try:
                result = self._call_handler(handler, error_record)
                variant_callback = params.get('set_model_variant')
                if callable(variant_callback):
                    try:
                        variant_callback('base', error_record=error_record)
                    except TypeError:
                        variant_callback('base')
                params['active_model_variant'] = 'base'
                return self._normalize_success_result(result, message='Base model fallback executed')
            except Exception as exc:
                last_error = exc
                logger.debug(
                    "Base model fallback %s failed: %s",
                    getattr(handler, '__name__', repr(handler)),
                    exc
                )

        error_message = str(last_error) if last_error else 'Base model fallback failed'
        return self._format_failure_result(error_message)

    def _clear_caches(self, error_record: ErrorRecord) -> Dict[str, Any]:
        """Clear all system caches."""
        params = error_record.context.parameters
        operations_performed = 0
        warnings: List[str] = []

        hooks = self._extract_callables(params.get('cache_clear_hooks'))
        cache_manager = params.get('cache_manager')
        if cache_manager is not None:
            if hasattr(cache_manager, 'clear') and callable(cache_manager.clear):
                hooks.append(cache_manager.clear)  # type: ignore[arg-type]
            if hasattr(cache_manager, 'reset') and callable(cache_manager.reset):
                hooks.append(cache_manager.reset)  # type: ignore[arg-type]

        cache_objects = params.get('cache_objects', params.get('caches'))
        if cache_objects is not None:
            if isinstance(cache_objects, dict):
                cache_iterable = cache_objects.values()
            elif isinstance(cache_objects, (list, tuple, set)):
                cache_iterable = cache_objects
            else:
                cache_iterable = [cache_objects]

            for cache_obj in cache_iterable:
                if hasattr(cache_obj, 'clear') and callable(getattr(cache_obj, 'clear')):
                    hooks.append(getattr(cache_obj, 'clear'))  # type: ignore[arg-type]
                elif hasattr(cache_obj, 'reset') and callable(getattr(cache_obj, 'reset')):
                    hooks.append(getattr(cache_obj, 'reset'))  # type: ignore[arg-type]

        for hook in hooks:
            try:
                self._call_handler(hook, error_record)
                operations_performed += 1
            except Exception as exc:
                warnings.append(str(exc))

        try:
            gc.collect()
            operations_performed += 1
        except Exception as exc:
            warnings.append(str(exc))

        handler_keys = [
            'clear_cache',
            'clear_caches',
            f"{error_record.context.component}.clear_cache",
            f"{error_record.context.component}.clear_caches",
            f"{error_record.context.operation}.clear_cache"
        ]
        for handler in self._resolve_handler_list(handler_keys):
            try:
                self._call_handler(handler, error_record)
                operations_performed += 1
            except Exception as exc:
                warnings.append(str(exc))

        if operations_performed:
            payload = {
                'success': True,
                'message': 'Caches cleared',
                'operations': operations_performed
            }
            if warnings:
                payload['warnings'] = warnings
            return payload

        if warnings:
            return self._format_failure_result('Failed to clear caches', warnings=warnings)
        return self._format_failure_result('No cache clearing hooks executed')

    def _reduce_batch_size(self, error_record: ErrorRecord) -> Dict[str, Any]:
        """Reduce batch size to handle memory issues."""
        params = error_record.context.parameters
        training_state = params.get('training_state')

        reduction_factor = params.get('batch_size_reduction_factor', 0.5)
        try:
            reduction_factor = float(reduction_factor)
        except (TypeError, ValueError):
            reduction_factor = 0.5
        reduction_factor = min(max(reduction_factor, 0.1), 1.0)

        min_batch_size = params.get('min_batch_size', 1)
        try:
            min_batch_size = max(1, int(min_batch_size))
        except (TypeError, ValueError):
            min_batch_size = 1

        updated = False
        previous_batch_size: Optional[int] = None
        new_batch_size: Optional[int] = None

        if isinstance(training_state, dict) and 'batch_size' in training_state:
            try:
                previous_batch_size = int(training_state['batch_size'])
            except (TypeError, ValueError):
                previous_batch_size = None

            if previous_batch_size and previous_batch_size > min_batch_size:
                new_batch_size = max(min_batch_size, int(max(1, previous_batch_size * reduction_factor)))
                if new_batch_size < previous_batch_size:
                    training_state['batch_size'] = new_batch_size
                    params['batch_size_previous'] = previous_batch_size
                    params['batch_size_reduced'] = True
                    updated = True
                    self._update_training_state(params, 'batch_size', new_batch_size, error_record)

        if not updated and params.get('target_batch_size') is not None:
            try:
                new_batch_size = int(params['target_batch_size'])
                self._update_training_state(params, 'batch_size', new_batch_size, error_record)
                params['batch_size_reduced'] = True
                updated = True
            except (TypeError, ValueError):
                pass

        if updated:
            return self._normalize_success_result(
                new_batch_size,
                message=f'Batch size reduced to {new_batch_size}',
                previous_batch_size=previous_batch_size
            )

        last_error: Optional[Exception] = None

        for handler in self._extract_callables(params.get('reduce_batch_size_callback')):
            try:
                result = self._call_handler(handler, error_record)
                return self._normalize_success_result(result, message='Batch size reduced via callback')
            except Exception as exc:
                last_error = exc

        handler_keys = [
            'reduce_batch_size',
            'adjust_batch_size',
            f"{error_record.context.component}.reduce_batch_size",
            f"{error_record.context.component}.adjust_batch_size"
        ]
        for handler in self._resolve_handler_list(handler_keys):
            try:
                result = self._call_handler(handler, error_record)
                return self._normalize_success_result(result, message='Batch size reduced via fallback handler')
            except Exception as exc:
                last_error = exc

        if last_error:
            return self._format_failure_result(str(last_error))
        return self._format_failure_result('Batch size reduction unavailable')

    def _enable_gradient_checkpointing(self, error_record: ErrorRecord) -> Dict[str, Any]:
        """Enable gradient checkpointing for memory efficiency."""
        params = error_record.context.parameters
        model = params.get('model')
        enabled = False
        warnings: List[str] = []

        if model is not None:
            for method_name in ('gradient_checkpointing_enable', 'enable_gradient_checkpointing'):
                method = getattr(model, method_name, None)
                if callable(method):
                    try:
                        method()
                        enabled = True
                        break
                    except Exception as exc:
                        warnings.append(str(exc))
            if not enabled and hasattr(model, 'gradient_checkpointing'):
                try:
                    setattr(model, 'gradient_checkpointing', True)
                    enabled = True
                except Exception as exc:
                    warnings.append(str(exc))

        if not enabled:
            callback_candidates = [
                params.get('enable_gradient_checkpointing'),
                params.get('gradient_checkpointing_hook'),
                params.get('gradient_checkpointing_callback'),
                params.get('enable_gradient_checkpointing_callback')
            ]
            for candidate in callback_candidates:
                for hook in self._extract_callables(candidate):
                    try:
                        self._call_handler(hook, error_record)
                        enabled = True
                        break
                    except Exception as exc:
                        warnings.append(str(exc))
                if enabled:
                    break

        if not enabled:
            handler_keys = [
                'enable_gradient_checkpointing',
                'gradient_checkpointing',
                f"{error_record.context.component}.enable_gradient_checkpointing",
                f"{error_record.context.component}.gradient_checkpointing"
            ]
            for handler in self._resolve_handler_list(handler_keys):
                try:
                    self._call_handler(handler, error_record)
                    enabled = True
                    break
                except Exception as exc:
                    warnings.append(str(exc))

        if enabled:
            params['gradient_checkpointing_enabled'] = True
            self._update_training_state(params, 'gradient_checkpointing', True, error_record)
            payload = {'success': True, 'message': 'Gradient checkpointing enabled'}
            if warnings:
                payload['warnings'] = warnings
            return payload

        if warnings:
            return self._format_failure_result('Unable to enable gradient checkpointing', warnings=warnings)
        return self._format_failure_result('Gradient checkpointing unavailable')

    def _use_cached_response(self, error_record: ErrorRecord) -> Dict[str, Any]:
        """Use cached response if available."""
        params = error_record.context.parameters

        if 'cached_response' in params:
            return self._normalize_success_result(
                params['cached_response'],
                message='Using cached response from context',
                source='context'
            )

        cache_store = params.get('cache_store') or params.get('response_cache')
        cache_key = params.get('cache_key') or params.get('request_id') or params.get('request_hash')
        last_error: Optional[Exception] = None

        if cache_store is not None and cache_key is not None:
            try:
                if hasattr(cache_store, 'get') and callable(getattr(cache_store, 'get')):
                    cached_value = cache_store.get(cache_key)
                else:
                    cached_value = cache_store[cache_key]  # type: ignore[index]
                if cached_value is not None:
                    return self._normalize_success_result(
                        cached_value,
                        message='Using cached response from store',
                        source='cache_store'
                    )
            except Exception as exc:
                last_error = exc

        for hook in self._extract_callables(params.get('cache_lookup')):
            try:
                cached_value = self._call_handler(hook, error_record)
                if cached_value is not None:
                    return self._normalize_success_result(
                        cached_value,
                        message='Using cached response via callback',
                        source='callback'
                    )
            except Exception as exc:
                last_error = exc

        handler_keys = [
            'cached_response',
            'use_cached_response',
            f"{error_record.context.component}.use_cached_response",
            f"{error_record.context.operation}.use_cached_response"
        ]
        for handler in self._resolve_handler_list(handler_keys):
            try:
                cached_value = self._call_handler(handler, error_record)
                if cached_value is not None:
                    return self._normalize_success_result(
                        cached_value,
                        message='Using cached response via fallback handler',
                        source='fallback_handler'
                    )
            except Exception as exc:
                last_error = exc

        if last_error:
            return self._format_failure_result(str(last_error))
        return self._format_failure_result('No cached response available')

    def _fallback_to_simpler_model(self, error_record: ErrorRecord) -> Dict[str, Any]:
        """Fallback to simpler model configuration."""
        params = error_record.context.parameters

        handlers = []
        handlers.extend(self._extract_callables(params.get('simpler_model_loader')))
        handlers.extend(self._extract_callables(params.get('fallback_model_loader')))

        handler_keys = [
            f"{error_record.context.component}.simpler_model",
            f"{error_record.context.operation}.simpler_model",
            'simpler_model',
            'fallback_to_simpler_model',
            f"{error_record.context.operation}.fallback_model"
        ]
        handlers.extend(self._resolve_handler_list(handler_keys))
        handlers = self._deduplicate_callables(handlers)

        if not handlers:
            return self._format_failure_result('No simpler model fallback registered')

        last_error: Optional[Exception] = None
        for handler in handlers:
            try:
                result = self._call_handler(handler, error_record)
                params['active_model_variant'] = 'simpler'
                variant_callback = params.get('set_model_variant')
                if callable(variant_callback):
                    try:
                        variant_callback('simpler', error_record=error_record)
                    except TypeError:
                        variant_callback('simpler')
                return self._normalize_success_result(result, message='Switched to simpler model configuration')
            except Exception as exc:
                last_error = exc
                logger.debug(
                    "Simpler model fallback %s failed: %s",
                    getattr(handler, '__name__', repr(handler)),
                    exc
                )

        error_message = str(last_error) if last_error else 'Simpler model fallback failed'
        return self._format_failure_result(error_message)

    def _save_checkpoint(self, error_record: ErrorRecord) -> Dict[str, Any]:
        """Save training checkpoint before failure."""
        params = error_record.context.parameters
        last_error: Optional[Exception] = None

        for saver in self._extract_callables(params.get('checkpoint_saver')):
            try:
                checkpoint_path = self._call_handler(saver, error_record)
                if checkpoint_path is not None:
                    params['last_checkpoint_path'] = checkpoint_path
                self._update_training_state(params, 'checkpoint_path', checkpoint_path, error_record)
                return self._normalize_success_result(
                    checkpoint_path,
                    message='Checkpoint saved before failure'
                )
            except Exception as exc:
                last_error = exc

        trainer = params.get('trainer')
        if trainer is not None:
            for method_name in ('save_checkpoint', 'save_state', 'save'):
                method = getattr(trainer, method_name, None)
                if callable(method):
                    try:
                        checkpoint_path = method()
                        params['last_checkpoint_path'] = checkpoint_path
                        self._update_training_state(params, 'checkpoint_path', checkpoint_path, error_record)
                        return self._normalize_success_result(
                            checkpoint_path,
                            message='Trainer checkpoint saved'
                        )
                    except Exception as exc:
                        last_error = exc

        handler_keys = [
            'save_checkpoint',
            'checkpoint',
            f"{error_record.context.component}.save_checkpoint"
        ]
        for handler in self._resolve_handler_list(handler_keys):
            try:
                checkpoint_path = self._call_handler(handler, error_record)
                if checkpoint_path is not None:
                    params['last_checkpoint_path'] = checkpoint_path
                self._update_training_state(params, 'checkpoint_path', checkpoint_path, error_record)
                return self._normalize_success_result(
                    checkpoint_path,
                    message='Checkpoint saved via fallback handler'
                )
            except Exception as exc:
                last_error = exc

        if last_error:
            return self._format_failure_result(str(last_error))
        return self._format_failure_result('Checkpoint saving unavailable')

    def _reduce_learning_rate(self, error_record: ErrorRecord) -> Dict[str, Any]:
        """Reduce learning rate to stabilize training."""
        params = error_record.context.parameters
        training_state = params.get('training_state')
        optimizer = params.get('optimizer')

        reduction_factor = params.get('learning_rate_reduction_factor', 0.5)
        try:
            reduction_factor = float(reduction_factor)
        except (TypeError, ValueError):
            reduction_factor = 0.5
        reduction_factor = min(max(reduction_factor, 0.0), 1.0)

        min_learning_rate = params.get('min_learning_rate')
        try:
            min_learning_rate = float(min_learning_rate) if min_learning_rate is not None else None
        except (TypeError, ValueError):
            min_learning_rate = None

        updated = False
        previous_lr: Optional[float] = None
        new_lr: Optional[float] = None

        if optimizer is not None and hasattr(optimizer, 'param_groups'):
            for group in getattr(optimizer, 'param_groups', []):
                if 'lr' not in group:
                    continue
                try:
                    original_lr = float(group['lr'])
                except (TypeError, ValueError):
                    continue
                previous_lr = original_lr if previous_lr is None else previous_lr
                if reduction_factor == 0.0:
                    new_lr_candidate = min_learning_rate or 0.0
                else:
                    new_lr_candidate = original_lr * reduction_factor
                if min_learning_rate is not None:
                    new_lr_candidate = max(min_learning_rate, new_lr_candidate)
                if new_lr_candidate < original_lr:
                    group['lr'] = new_lr_candidate
                    new_lr = new_lr_candidate
                    updated = True
            if updated and new_lr is not None:
                params['learning_rate_previous'] = previous_lr
                self._update_training_state(params, 'learning_rate', new_lr, error_record)
                return self._normalize_success_result(
                    new_lr,
                    message=f'Learning rate reduced to {new_lr}',
                    previous_learning_rate=previous_lr
                )

        if not updated and isinstance(training_state, dict) and 'learning_rate' in training_state:
            try:
                original_lr = float(training_state['learning_rate'])
                previous_lr = original_lr
                if reduction_factor == 0.0:
                    new_lr = min_learning_rate or 0.0
                else:
                    new_lr = original_lr * reduction_factor
                if min_learning_rate is not None:
                    new_lr = max(min_learning_rate, new_lr)
                if new_lr < original_lr:
                    training_state['learning_rate'] = new_lr
                    updated = True
                    self._update_training_state(params, 'learning_rate', new_lr, error_record)
            except (TypeError, ValueError):
                new_lr = None

            if updated and new_lr is not None:
                return self._normalize_success_result(
                    new_lr,
                    message=f'Learning rate reduced to {new_lr}',
                    previous_learning_rate=previous_lr
                )

        last_error: Optional[Exception] = None

        scheduler = params.get('scheduler')
        if scheduler is not None and hasattr(scheduler, 'step') and callable(getattr(scheduler, 'step')):
            try:
                scheduler.step()
                updated = True
                params['scheduler_stepped'] = True
                return self._normalize_success_result(
                    None,
                    message='Learning rate scheduler stepped'
                )
            except Exception as exc:
                last_error = exc

        for handler in self._extract_callables(params.get('reduce_learning_rate_callback')):
            try:
                result = self._call_handler(handler, error_record)
                return self._normalize_success_result(result, message='Learning rate reduced via callback')
            except Exception as exc:
                last_error = exc

        handler_keys = [
            'reduce_learning_rate',
            'adjust_learning_rate',
            f"{error_record.context.component}.reduce_learning_rate"
        ]
        for handler in self._resolve_handler_list(handler_keys):
            try:
                result = self._call_handler(handler, error_record)
                return self._normalize_success_result(result, message='Learning rate reduced via fallback handler')
            except Exception as exc:
                last_error = exc

        if last_error:
            return self._format_failure_result(str(last_error))
        return self._format_failure_result('Learning rate adjustment unavailable')

    def _skip_problematic_batch(self, error_record: ErrorRecord) -> Dict[str, Any]:
        """Skip problematic training batch."""
        params = error_record.context.parameters
        training_state = params.get('training_state')
        skipped = False
        last_error: Optional[Exception] = None

        if isinstance(training_state, dict):
            training_state['skip_batch'] = True
            params['batch_skipped'] = True
            skipped = True
            self._update_training_state(params, 'skip_batch', True, error_record)

        data_loader = params.get('data_loader')
        if data_loader is not None:
            for method_name in ('skip_current_batch', 'skip_batch', 'advance'):
                method = getattr(data_loader, method_name, None)
                if callable(method):
                    try:
                        method()
                        skipped = True
                        break
                    except Exception as exc:
                        last_error = exc

        for hook in self._extract_callables(params.get('skip_batch_callback')):
            try:
                self._call_handler(hook, error_record)
                skipped = True
            except Exception as exc:
                last_error = exc

        handler_keys = [
            'skip_problematic_batch',
            'skip_batch',
            f"{error_record.context.component}.skip_batch"
        ]
        for handler in self._resolve_handler_list(handler_keys):
            try:
                self._call_handler(handler, error_record)
                skipped = True
                break
            except Exception as exc:
                last_error = exc

        if skipped:
            return self._normalize_success_result(
                None,
                message='Problematic batch skipped'
            )

        if last_error:
            return self._format_failure_result(str(last_error))
        return self._format_failure_result('No mechanism to skip batch available')

    def _use_backup_data_source(self, error_record: ErrorRecord) -> Dict[str, Any]:
        """Use backup data source."""
        params = error_record.context.parameters
        switched = False
        new_source = None
        last_error: Optional[Exception] = None

        backup_handlers = []
        backup_handlers.extend(self._extract_callables(params.get('use_backup_data')))
        backup_handlers.extend(self._extract_callables(params.get('backup_data_loader')))
        backup_handlers.extend(self._extract_callables(params.get('backup_data_provider')))

        dataset_manager = params.get('dataset_manager')
        if dataset_manager is not None:
            for method_name in ('switch_to_backup', 'use_backup', 'load_backup'):
                method = getattr(dataset_manager, method_name, None)
                if callable(method):
                    backup_handlers.append(lambda record, m=method: m())

        for handler in backup_handlers:
            try:
                new_source = self._call_handler(handler, error_record)
                switched = True
                break
            except Exception as exc:
                last_error = exc

        if not switched and 'backup_data' in params:
            new_source = params['backup_data']
            switched = True

        if not switched:
            handler_keys = [
                'use_backup_data_source',
                'backup_data',
                f"{error_record.context.component}.use_backup_data_source"
            ]
            for handler in self._resolve_handler_list(handler_keys):
                try:
                    new_source = self._call_handler(handler, error_record)
                    switched = True
                    break
                except Exception as exc:
                    last_error = exc

        if switched:
            params['active_data_source'] = 'backup'
            if new_source is not None:
                params['backup_data_reference'] = new_source
            self._update_training_state(params, 'data_source', 'backup', error_record)
            return self._normalize_success_result(
                new_source,
                message='Switched to backup data source'
            )

        if last_error:
            return self._format_failure_result(str(last_error))
        return self._format_failure_result('No backup data source configured')

    def _skip_corrupted_samples(self, error_record: ErrorRecord) -> Dict[str, Any]:
        """Skip corrupted data samples."""
        params = error_record.context.parameters
        dataset = params.get('dataset')
        corrupted_indices = params.get('corrupted_indices') or []
        corrupted_ids = params.get('corrupted_sample_ids') or []
        skipped_count = 0
        last_error: Optional[Exception] = None

        def _ensure_iterable(value: Any) -> List[Any]:
            if value is None:
                return []
            if isinstance(value, (list, tuple, set)):
                return list(value)
            return [value]

        indices_list = _ensure_iterable(corrupted_indices)
        id_list = _ensure_iterable(corrupted_ids)

        if dataset is not None and indices_list:
            for index in sorted(indices_list, reverse=True):
                try:
                    if hasattr(dataset, 'pop') and callable(getattr(dataset, 'pop')):
                        dataset.pop(index)
                    elif hasattr(dataset, '__delitem__'):
                        del dataset[index]
                    skipped_count += 1
                except Exception as exc:
                    last_error = exc

        params.setdefault('skipped_samples', [])
        initial_skipped = len(params['skipped_samples'])
        params['skipped_samples'].extend(id_list or indices_list)
        skipped_count = max(skipped_count, len(params['skipped_samples']) - initial_skipped)

        for hook in self._extract_callables(params.get('skip_corrupted_callback')):
            try:
                self._call_handler(hook, error_record)
                skipped_count = max(skipped_count, 1)
            except Exception as exc:
                last_error = exc

        handler_keys = [
            'skip_corrupted_samples',
            'handle_corrupted_samples',
            f"{error_record.context.component}.skip_corrupted_samples"
        ]
        for handler in self._resolve_handler_list(handler_keys):
            try:
                self._call_handler(handler, error_record)
                skipped_count = max(skipped_count, 1)
                break
            except Exception as exc:
                last_error = exc

        if skipped_count > 0:
            self._update_training_state(params, 'skipped_samples', params.get('skipped_samples'), error_record)
            return self._normalize_success_result(
                skipped_count,
                message='Corrupted samples skipped',
                skipped_samples=params.get('skipped_samples')
            )

        if last_error:
            return self._format_failure_result(str(last_error))
        return self._format_failure_result('No corrupted samples were skipped')

    # Public API methods
    def register_recovery_strategy(self, category: ErrorCategory, strategy: Callable):
        """Register a custom recovery strategy."""
        if category not in self.recovery_strategies:
            self.recovery_strategies[category] = []
        self.recovery_strategies[category].append(strategy)
        logger.info(f"Registered recovery strategy for {category.value}")

    def register_fallback_handler(self, operation: str, handler: Callable):
        """Register a fallback handler for specific operations."""
        if not callable(handler):
            raise ValueError('Fallback handler must be callable')

        handlers = self.fallback_handlers.setdefault(operation, [])
        if not any(existing is handler for existing in handlers):
            handlers.append(handler)
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
