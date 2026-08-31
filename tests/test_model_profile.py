from __future__ import annotations

import pytest

from gemmafischer.model_profile import (
    ModelProfile,
    ModelRequestMetrics,
    summarize_model_requests,
    validate_profile_outputs,
)


def _metrics(
    index: int, *, ttft: float, total: float, generation_tps: float
) -> ModelRequestMetrics:
    return ModelRequestMetrics(
        request_index=index,
        cold_request=index == 0,
        prompt_tokens=100,
        output_tokens=20,
        time_to_first_token_seconds=ttft,
        prompt_duration_seconds=0.5,
        generation_duration_seconds=20 / generation_tps,
        total_latency_seconds=total,
        prompt_tokens_per_second=200.0,
        generation_tokens_per_second=generation_tps,
        peak_mlx_memory_bytes=1_000 + index,
        finish_reason="stop",
        output_text="grounded output",
    )


def test_summary_separates_cold_request_from_warm_distribution() -> None:
    requests = (
        _metrics(0, ttft=3.0, total=4.0, generation_tps=10.0),
        _metrics(1, ttft=1.0, total=2.0, generation_tps=20.0),
        _metrics(2, ttft=1.2, total=2.2, generation_tps=25.0),
    )

    result = summarize_model_requests(requests)

    assert result["request_count"] == 3
    assert result["total_prompt_tokens"] == 300
    assert result["total_output_tokens"] == 60
    assert result["peak_mlx_memory_bytes"] == 1_002
    assert result["all_requests"]["time_to_first_token_seconds"]["max"] == 3.0
    assert result["warm_requests"]["time_to_first_token_seconds"] == {
        "mean": 1.1,
        "p50": 1.0,
        "p95": 1.2,
        "min": 1.0,
        "max": 1.2,
    }
    assert result["warm_requests"]["generation_tokens_per_second"]["mean"] == 22.5


def test_single_request_is_also_the_warm_summary() -> None:
    request = _metrics(0, ttft=3.0, total=4.0, generation_tps=10.0)

    result = summarize_model_requests((request,))

    assert result["warm_requests"] == result["all_requests"]


def test_summary_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="At least one"):
        summarize_model_requests(())


def test_profile_output_validation_records_exact_contract_rate() -> None:
    requests = (
        _metrics(0, ttft=1.0, total=2.0, generation_tps=20.0),
        _metrics(1, ttft=1.0, total=2.0, generation_tps=20.0),
    )
    profile = ModelProfile("model", "revision", 0.1, 0.2, requests, {})

    def validator(index: int, output: str) -> None:
        assert output == "grounded output"
        if index == 1:
            raise ValueError("schema mismatch")

    result = validate_profile_outputs(profile, validator)

    assert result.requests[0].contract_valid is True
    assert result.requests[1].contract_valid is False
    assert result.requests[1].contract_error == "schema mismatch"
    assert result.summary["contract_checked_request_count"] == 2
    assert result.summary["contract_valid_request_count"] == 1
    assert result.summary["contract_valid_request_rate"] == 0.5
