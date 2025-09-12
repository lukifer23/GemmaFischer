#!/usr/bin/env python3
"""
Comprehensive Logging Configuration for ChessGemma

Provides structured logging with:
- Multiple log levels and formats
- File and console handlers
- Performance timing
- Error tracking and reporting
- JSON structured logs for analysis
"""

import logging
import logging.handlers
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import threading
from functools import wraps


class StructuredLogger:
    """Structured logger with performance timing and error tracking."""

    def __init__(self, name: str, log_level: str = "INFO", log_file: Optional[Path] = None):
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level.upper()))

        # Remove existing handlers to avoid duplicates
        self.logger.handlers.clear()

        # Create formatters
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        json_formatter = logging.Formatter(
            '{"timestamp": "%(asctime)s", "name": "%(name)s", "level": "%(levelname)s", '
            '"message": "%(message)s", "module": "%(module)s", "func": "%(funcName)s", '
            '"line": %(lineno)d}'
        )

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

        # File handler (JSON format for analysis)
        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.handlers.RotatingFileHandler(
                log_file, maxBytes=10*1024*1024, backupCount=5
            )
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(json_formatter)
            self.logger.addHandler(file_handler)

        self._timers: Dict[str, float] = {}
        self._counters: Dict[str, int] = {}
        self._lock = threading.Lock()

    def debug(self, message: str, **kwargs):
        """Log debug message with optional structured data."""
        self._log('debug', message, **kwargs)

    def info(self, message: str, **kwargs):
        """Log info message with optional structured data."""
        self._log('info', message, **kwargs)

    def warning(self, message: str, **kwargs):
        """Log warning message with optional structured data."""
        self._log('warning', message, **kwargs)

    def error(self, message: str, exc_info: bool = False, **kwargs):
        """Log error message with optional exception info."""
        self._log('error', message, exc_info=exc_info, **kwargs)

    def critical(self, message: str, exc_info: bool = True, **kwargs):
        """Log critical message with exception info."""
        self._log('critical', message, exc_info=exc_info, **kwargs)

    def _log(self, level: str, message: str, exc_info: bool = False, **kwargs):
        """Internal logging method."""
        if kwargs:
            # Add structured data to message
            extra_data = json.dumps(kwargs, default=str)
            message = f"{message} | {extra_data}"

        getattr(self.logger, level)(message, exc_info=exc_info)

    def start_timer(self, name: str):
        """Start a performance timer."""
        with self._lock:
            self._timers[name] = time.time()
        self.debug(f"Started timer: {name}")

    def end_timer(self, name: str) -> float:
        """End a performance timer and return elapsed time."""
        with self._lock:
            if name not in self._timers:
                self.warning(f"Timer '{name}' was not started")
                return 0.0

            elapsed = time.time() - self._timers[name]
            del self._timers[name]

        self.info(f"Timer '{name}' completed", elapsed_seconds=elapsed)
        return elapsed

    def time_function(self, func_name: Optional[str] = None):
        """Decorator to time function execution."""
        def decorator(func):
            name = func_name or f"{func.__module__}.{func.__qualname__}"

            @wraps(func)
            def wrapper(*args, **kwargs):
                self.start_timer(name)
                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    self.error(f"Function '{name}' failed", error=str(e), exc_info=True)
                    raise
                finally:
                    elapsed = self.end_timer(name)
                    self.debug(f"Function '{name}' took {elapsed:.3f}s")

            return wrapper
        return decorator

    def increment_counter(self, name: str, amount: int = 1):
        """Increment a named counter."""
        with self._lock:
            self._counters[name] = self._counters.get(name, 0) + amount

    def get_counter(self, name: str) -> int:
        """Get current value of a counter."""
        with self._lock:
            return self._counters.get(name, 0)

    def log_metrics(self, metrics: Dict[str, Any]):
        """Log a collection of metrics."""
        self.info("Metrics update", **metrics)

    def log_model_info(self, model_info: Dict[str, Any]):
        """Log model loading/initialization info."""
        self.info("Model info", **model_info)

    def log_inference_request(self, request_data: Dict[str, Any]):
        """Log inference request details."""
        # Sanitize sensitive data
        safe_data = {k: v for k, v in request_data.items()
                    if k not in ['api_key', 'token']}
        self.info("Inference request", **safe_data)

    def log_inference_response(self, response_data: Dict[str, Any], elapsed: float):
        """Log inference response details."""
        self.info("Inference response", response_time=elapsed, **response_data)

    def log_training_progress(self, epoch: int, step: int, loss: float,
                            learning_rate: float, **metrics):
        """Log training progress."""
        self.info("Training progress",
                 epoch=epoch,
                 step=step,
                 loss=loss,
                 learning_rate=learning_rate,
                 **metrics)

    def log_evaluation_results(self, dataset_name: str, results: Dict[str, Any]):
        """Log evaluation results."""
        self.info("Evaluation results", dataset=dataset_name, **results)


# Global logger instances
_loggers: Dict[str, StructuredLogger] = {}
_logger_lock = threading.Lock()


def get_logger(name: str, log_level: str = "INFO", log_file: Optional[Path] = None) -> StructuredLogger:
    """Get or create a structured logger instance."""
    with _logger_lock:
        if name not in _loggers:
            _loggers[name] = StructuredLogger(name, log_level, log_file)
        return _loggers[name]


def setup_global_logging(log_level: str = "INFO", log_file: Optional[Path] = None):
    """Setup global logging configuration."""
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))

    # Clear existing handlers
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=10*1024*1024, backupCount=5
        )
        json_formatter = logging.Formatter(
            '{"timestamp": "%(asctime)s", "name": "%(name)s", "level": "%(levelname)s", '
            '"message": "%(message)s", "module": "%(module)s", "func": "%(funcName)s", '
            '"line": %(lineno)d}'
        )
        file_handler.setFormatter(json_formatter)
        root_logger.addHandler(file_handler)


def log_performance(func):
    """Decorator to log function performance."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        start_time = time.time()

        logger.debug(f"Starting {func.__qualname__}")

        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            logger.info(f"Completed {func.__qualname__}", elapsed_seconds=elapsed)
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"Failed {func.__qualname__}",
                        elapsed_seconds=elapsed,
                        error=str(e),
                        exc_info=True)
            raise

    return wrapper


# Convenience functions for common logging patterns
def log_model_load(model_name: str, model_path: Path, success: bool, **kwargs):
    """Log model loading events."""
    logger = get_logger("model_loading")
    level = "info" if success else "error"
    message = f"{'Loaded' if success else 'Failed to load'} model {model_name}"
    getattr(logger, level)(message, model_path=str(model_path), **kwargs)


def log_inference_error(error: Exception, request_data: Dict[str, Any]):
    """Log inference errors with context."""
    logger = get_logger("inference_errors")
    logger.error("Inference failed",
                error_type=type(error).__name__,
                error_message=str(error),
                request_data=request_data,
                exc_info=True)


def log_training_checkpoint(epoch: int, step: int, checkpoint_path: Path, metrics: Dict[str, Any]):
    """Log training checkpoint saves."""
    logger = get_logger("training")
    logger.info("Saved training checkpoint",
               epoch=epoch,
               step=step,
               checkpoint_path=str(checkpoint_path),
               **metrics)


# Initialize default loggers
default_log_file = Path("logs/chessgemma.log")
setup_global_logging(log_file=default_log_file)

# Create module-level logger
logger = get_logger(__name__)
