from __future__ import annotations

import json
import statistics
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from .lmstudio import LMStudioIdentity, verify_lmstudio_identity
from .runtime import DEFAULT_MODEL, DEFAULT_MODEL_REVISION, inspect_model_assets


@dataclass(frozen=True)
class ModelRequestMetrics:
    """Timing reported by one real MLX-LM streaming generation."""

    request_index: int
    cold_request: bool
    prompt_tokens: int
    output_tokens: int
    time_to_first_token_seconds: float
    prompt_duration_seconds: float | None
    generation_duration_seconds: float
    total_latency_seconds: float
    prompt_tokens_per_second: float | None
    generation_tokens_per_second: float
    peak_mlx_memory_bytes: int | None
    finish_reason: str
    output_text: str
    time_to_first_visible_token_seconds: float | None = None
    reasoning_tokens: int | None = None
    reasoning_text: str = ""
    succeeded: bool = True
    error_code: str | None = None
    contract_valid: bool | None = None
    contract_error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ModelProfile:
    model_id: str
    revision: str | None
    asset_verification_seconds: float
    model_load_seconds: float | None
    requests: tuple[ModelRequestMetrics, ...]
    summary: dict[str, Any]
    backend: str = "mlx"
    identity: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "model_id": self.model_id,
            "revision": self.revision,
            "asset_verification_seconds": self.asset_verification_seconds,
            "model_load_seconds": self.model_load_seconds,
            "requests": [request.as_dict() for request in self.requests],
            "summary": self.summary,
            "backend": self.backend,
            "identity": self.identity,
        }


def summarize_model_requests(requests: Sequence[ModelRequestMetrics]) -> dict[str, Any]:
    """Build a serialization-safe summary without importing or loading MLX."""

    if not requests:
        raise ValueError("At least one model request is required")
    warm = requests[1:] if len(requests) > 1 else requests
    contract_checked = [item for item in requests if item.contract_valid is not None]
    return {
        "request_count": len(requests),
        "total_prompt_tokens": sum(item.prompt_tokens for item in requests),
        "total_output_tokens": sum(item.output_tokens for item in requests),
        "successful_request_count": sum(item.succeeded for item in requests),
        "failed_request_count": sum(not item.succeeded for item in requests),
        "successful_request_rate": sum(item.succeeded for item in requests) / len(requests),
        "contract_checked_request_count": len(contract_checked),
        "contract_valid_request_count": sum(item.contract_valid is True for item in requests),
        "contract_valid_request_rate": (
            sum(item.contract_valid is True for item in contract_checked) / len(contract_checked)
            if contract_checked
            else None
        ),
        "peak_mlx_memory_bytes": _optional_max(
            [item.peak_mlx_memory_bytes for item in requests]
        ),
        "all_requests": _request_summary(requests),
        "warm_requests": _request_summary(warm),
    }


def validate_profile_outputs(
    profile: ModelProfile,
    validator: Callable[[int, str], None],
) -> ModelProfile:
    """Apply the production output contract to every captured model response."""

    checked: list[ModelRequestMetrics] = []
    for request in profile.requests:
        try:
            validator(request.request_index, request.output_text)
        except ValueError as exc:
            checked.append(
                replace(
                    request,
                    contract_valid=False,
                    contract_error=str(exc)[:500],
                )
            )
        else:
            checked.append(replace(request, contract_valid=True, contract_error=None))
    frozen = tuple(checked)
    return replace(profile, requests=frozen, summary=summarize_model_requests(frozen))


def profile_mlx_generation(
    prompts: Iterable[str],
    *,
    model_id: str = DEFAULT_MODEL,
    revision: str | None = DEFAULT_MODEL_REVISION,
    max_tokens: int = 256,
    system_prompt: str | None = None,
    manifest_path: Path | None = None,
) -> ModelProfile:
    """Load the pinned model once and profile real streaming generation requests.

    ``time_to_first_token_seconds`` is measured at the caller and includes prompt
    prefill plus first-token decoding. MLX-LM's own counters supply prompt and
    generation throughput, avoiding estimates based on Python string lengths.
    """

    prompt_list = tuple(prompts)
    if not prompt_list or any(not prompt.strip() for prompt in prompt_list):
        raise ValueError("One or more non-empty prompts are required")
    if max_tokens < 1:
        raise ValueError("max_tokens must be positive")

    try:
        import mlx.core as mx
        from mlx_lm import load, stream_generate
    except ImportError as exc:
        raise RuntimeError("Install the full profile with: uv sync --extra full") from exc

    verification_started = time.perf_counter()
    verified = inspect_model_assets(model_id, revision, manifest_path)
    verification_seconds = time.perf_counter() - verification_started

    load_started = time.perf_counter()
    loaded = load(str(verified["snapshot"]))
    model, tokenizer = loaded[0], loaded[1]
    load_seconds = time.perf_counter() - load_started

    requests: list[ModelRequestMetrics] = []
    for index, prompt in enumerate(prompt_list):
        formatted_prompt = (
            tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            if system_prompt is not None
            else prompt
        )
        mx.reset_peak_memory()
        started = time.perf_counter()
        first_response_at: float | None = None
        output_parts: list[str] = []
        final_response: Any | None = None
        for response in stream_generate(
            model,
            tokenizer,
            prompt=formatted_prompt,
            max_tokens=max_tokens,
        ):
            if first_response_at is None:
                first_response_at = time.perf_counter()
            output_parts.append(response.text)
            final_response = response
        ended = time.perf_counter()
        if final_response is None or first_response_at is None:
            raise RuntimeError("MLX-LM generation returned no token telemetry")

        prompt_tps = float(final_response.prompt_tps)
        generation_tps = float(final_response.generation_tps)
        prompt_tokens = int(final_response.prompt_tokens)
        output_tokens = int(final_response.generation_tokens)
        if prompt_tps <= 0 or generation_tps <= 0:
            raise RuntimeError("MLX-LM returned non-positive throughput telemetry")
        response_peak_bytes = int(float(final_response.peak_memory) * 1_000_000_000)
        requests.append(
            ModelRequestMetrics(
                request_index=index,
                cold_request=index == 0,
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
                time_to_first_token_seconds=first_response_at - started,
                prompt_duration_seconds=prompt_tokens / prompt_tps,
                generation_duration_seconds=output_tokens / generation_tps,
                total_latency_seconds=ended - started,
                prompt_tokens_per_second=prompt_tps,
                generation_tokens_per_second=generation_tps,
                peak_mlx_memory_bytes=max(response_peak_bytes, int(mx.get_peak_memory())),
                finish_reason=str(final_response.finish_reason or "unknown"),
                output_text="".join(output_parts),
                time_to_first_visible_token_seconds=first_response_at - started,
            )
        )

    frozen_requests = tuple(requests)
    return ModelProfile(
        model_id=model_id,
        revision=revision,
        asset_verification_seconds=verification_seconds,
        model_load_seconds=load_seconds,
        requests=frozen_requests,
        summary=summarize_model_requests(frozen_requests),
    )


def profile_lmstudio_generation(
    prompts: Iterable[str],
    *,
    model_id: str,
    base_url: str,
    model_artifact: Path,
    max_tokens: int = 256,
    system_prompt: str | None = None,
    timeout_seconds: float = 30.0,
) -> ModelProfile:
    """Profile a real loopback LM Studio stream without inventing unavailable metrics."""

    prompt_list = tuple(prompts)
    if not prompt_list or any(not prompt.strip() for prompt in prompt_list):
        raise ValueError("One or more non-empty prompts are required")
    if max_tokens < 1:
        raise ValueError("max_tokens must be positive")
    verification_started = time.perf_counter()
    identity = verify_lmstudio_identity(base_url, model_id, model_artifact, timeout_seconds)
    verification_seconds = time.perf_counter() - verification_started
    requests = tuple(
        _profile_lmstudio_request(
            prompt,
            index=index,
            identity=identity,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
        )
        for index, prompt in enumerate(prompt_list)
    )
    return ModelProfile(
        model_id=model_id,
        revision=None,
        asset_verification_seconds=verification_seconds,
        model_load_seconds=None,
        requests=requests,
        summary=summarize_model_requests(requests),
        backend="lmstudio-openai-compatible",
        identity=identity.as_dict(),
    )


def _profile_lmstudio_request(
    prompt: str,
    *,
    index: int,
    identity: LMStudioIdentity,
    system_prompt: str | None,
    max_tokens: int,
    timeout_seconds: float,
) -> ModelRequestMetrics:
    messages: list[dict[str, str]] = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    payload = {
        "model": identity.model_id,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    request = urllib.request.Request(
        f"{identity.base_url}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    started = time.perf_counter()
    first_token_at: float | None = None
    first_visible_at: float | None = None
    output_parts: list[str] = []
    reasoning_parts: list[str] = []
    usage: dict[str, Any] | None = None
    finish_reason = "unknown"
    returned_model: str | None = None
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            for raw_line in response:
                if len(raw_line) > 1024 * 1024:
                    raise RuntimeError("LM Studio stream chunk exceeded 1 MiB")
                line = raw_line.decode("utf-8").strip()
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                returned_model = str(chunk.get("model") or returned_model or "")
                if isinstance(chunk.get("usage"), dict):
                    usage = chunk["usage"]
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                finish_reason = str(choice.get("finish_reason") or finish_reason)
                delta = choice.get("delta") or {}
                reasoning = str(delta.get("reasoning_content") or "")
                content = str(delta.get("content") or "")
                if (reasoning or content) and first_token_at is None:
                    first_token_at = time.perf_counter()
                if content and first_visible_at is None:
                    first_visible_at = time.perf_counter()
                reasoning_parts.append(reasoning)
                output_parts.append(content)
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"LM Studio streaming request failed: {exc}") from exc
    ended = time.perf_counter()
    if returned_model != identity.model_id:
        raise RuntimeError(
            f"LM Studio returned model {returned_model!r}, expected {identity.model_id!r}"
        )
    if first_token_at is None:
        raise RuntimeError("LM Studio stream did not return generated tokens")
    if usage is None:
        raise RuntimeError("LM Studio stream did not return required usage telemetry")
    prompt_tokens = int(usage.get("prompt_tokens", 0))
    output_tokens = int(usage.get("completion_tokens", 0))
    if prompt_tokens <= 0 or output_tokens <= 0:
        raise RuntimeError("LM Studio returned non-positive token usage")
    generation_seconds = ended - first_token_at
    reasoning_details = usage.get("completion_tokens_details") or {}
    reasoning_tokens = reasoning_details.get("reasoning_tokens")
    return ModelRequestMetrics(
        request_index=index,
        cold_request=index == 0,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        time_to_first_token_seconds=first_token_at - started,
        prompt_duration_seconds=None,
        generation_duration_seconds=generation_seconds,
        total_latency_seconds=ended - started,
        prompt_tokens_per_second=None,
        generation_tokens_per_second=output_tokens / generation_seconds,
        peak_mlx_memory_bytes=None,
        finish_reason=finish_reason,
        output_text="".join(output_parts),
        time_to_first_visible_token_seconds=(
            first_visible_at - started if first_visible_at is not None else None
        ),
        reasoning_tokens=int(reasoning_tokens) if reasoning_tokens is not None else None,
        reasoning_text="".join(reasoning_parts),
        succeeded=first_visible_at is not None and finish_reason != "length",
        error_code=(
            "NO_VISIBLE_OUTPUT"
            if first_visible_at is None
            else "TOKEN_LIMIT_REACHED"
            if finish_reason == "length"
            else None
        ),
    )


def _request_summary(requests: Sequence[ModelRequestMetrics]) -> dict[str, Any]:
    return {
        "time_to_first_token_seconds": _distribution(
            [item.time_to_first_token_seconds for item in requests]
        ),
        "time_to_first_visible_token_seconds": _optional_distribution(
            [item.time_to_first_visible_token_seconds for item in requests]
        ),
        "total_latency_seconds": _distribution(
            [item.total_latency_seconds for item in requests]
        ),
        "prompt_tokens_per_second": _optional_distribution(
            [item.prompt_tokens_per_second for item in requests]
        ),
        "generation_tokens_per_second": _distribution(
            [item.generation_tokens_per_second for item in requests]
        ),
    }


def _distribution(values: Sequence[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "mean": statistics.fmean(ordered),
        "p50": _percentile(ordered, 0.50),
        "p95": _percentile(ordered, 0.95),
        "min": ordered[0],
        "max": ordered[-1],
    }


def _optional_distribution(values: Sequence[float | None]) -> dict[str, float] | None:
    present = [item for item in values if item is not None]
    return _distribution(present) if present else None


def _optional_max(values: Sequence[int | None]) -> int | None:
    present = [item for item in values if item is not None]
    return max(present) if present else None


def _percentile(ordered: Sequence[float], fraction: float) -> float:
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return ordered[index]
