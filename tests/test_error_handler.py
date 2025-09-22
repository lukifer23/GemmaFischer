import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.error_handler import ChessGemmaErrorHandler


def test_memory_error_recovers_with_cache_clear_hook():
    handler = ChessGemmaErrorHandler()
    cleared = []

    def cache_hook(record):
        cleared.append(record.context.operation)

    result = handler.handle_error(
        RuntimeError("CUDA out of memory"),
        component="trainer",
        operation="train_step",
        cache_clear_hooks=[cache_hook],
        training_state={'batch_size': 8}
    )

    assert result is None
    assert cleared == ["train_step"]

    history = handler.error_history[-1]
    assert history.resolved is True
    assert history.category.name == "MEMORY"
    assert history.recovery_attempts[0]['strategy'] == '_clear_caches'
    assert history.recovery_attempts[0]['success'] is True


def test_training_error_reduces_learning_rate():
    handler = ChessGemmaErrorHandler()

    class OptimizerMock:
        def __init__(self):
            self.param_groups = [{'lr': 0.01}]

    optimizer = OptimizerMock()
    training_state = {'learning_rate': 0.01}

    result = handler.handle_error(
        ValueError("nan loss encountered"),
        component="trainer",
        operation="train",
        optimizer=optimizer,
        training_state=training_state,
        learning_rate_reduction_factor=0.5
    )

    assert result == pytest.approx(0.005)
    history = handler.error_history[-1]
    assert history.resolved is True
    assert history.category.name == "TRAINING"
    assert history.recovery_attempts[-1]['strategy'] == '_reduce_learning_rate'
    assert history.recovery_attempts[-1]['success'] is True
    assert optimizer.param_groups[0]['lr'] == pytest.approx(0.005)
    assert training_state['learning_rate'] == pytest.approx(0.005)


def test_inference_error_recovers_via_retry():
    handler = ChessGemmaErrorHandler()
    call_counter = {'count': 0}

    def flaky_operation():
        call_counter['count'] += 1
        if call_counter['count'] < 2:
            raise RuntimeError("temporary failure")
        return "ok"

    result = handler.handle_error(
        RuntimeError("generation failure"),
        component="inference",
        operation="generate",
        retry_callback=flaky_operation,
        retry_max_retries=3,
        retry_base_delay=0,
        retry_disable_sleep=True
    )

    assert result == "ok"
    assert call_counter['count'] == 2

    history = handler.error_history[-1]
    assert history.resolved is True
    assert history.category.name == "INFERENCE"
    assert history.recovery_attempts[0]['strategy'] == '_retry_with_backoff'
    assert history.recovery_attempts[0]['success'] is True
