from __future__ import annotations

import hashlib
import json
import os
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .domain import EngineEvidence, RatingBucket
from .model_profile import profile_mlx_generation
from .runtime import (
    DEFAULT_MODEL,
    DEFAULT_MODEL_REVISION,
    LESSON_SELECTION_CONTRACT_VERSION,
    LESSON_SELECTION_SYSTEM_PROMPT,
    lesson_selection_prompt,
    parse_lesson_selection,
)


def evaluate_untuned_training_baseline(
    dataset_path: Path,
    output_path: Path,
    *,
    limit: int = 250,
    model_id: str = DEFAULT_MODEL,
    revision: str = DEFAULT_MODEL_REVISION,
) -> dict[str, Any]:
    """Run the real pinned untuned model over frozen validation records."""
    records = _load_validation_records(dataset_path, limit)
    profile = profile_mlx_generation(
        (str(record["prompt"]) for record in records),
        model_id=model_id,
        revision=revision,
        max_tokens=192,
        system_prompt=LESSON_SELECTION_SYSTEM_PROMPT,
    )
    results: list[dict[str, Any]] = []
    errors: Counter[str] = Counter()
    for record, request in zip(records, profile.requests, strict=True):
        evidence = EngineEvidence.model_validate(record["input"])
        rating = RatingBucket(record["meta"]["rating_bucket"])
        codes: list[str] = []
        parsed: dict[str, Any] | None = None
        try:
            parse_lesson_selection(request.output_text, evidence, rating)
            parsed = _extract_object(request.output_text)
        except ValueError:
            codes.append("contract_invalid")
        if parsed is not None:
            expected = record["target"]
            for field, code in (
                ("claim_ids", "claim_selection_mismatch"),
                ("concept_ids", "concept_selection_mismatch"),
                ("question_template_id", "question_selection_mismatch"),
                ("hint_template_id", "hint_selection_mismatch"),
            ):
                if parsed.get(field) != expected.get(field):
                    codes.append(code)
        errors.update(codes)
        results.append(
            {
                "record_id": record["record_id"],
                "rating_bucket": rating.value,
                "contract_valid": "contract_invalid" not in codes,
                "exact_target_match": not codes,
                "error_codes": codes,
                "output": request.output_text,
                "latency_seconds": request.total_latency_seconds,
                "peak_mlx_memory_bytes": request.peak_mlx_memory_bytes,
            }
        )
    record_count = len(results)
    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "status": "passed",
        "task": LESSON_SELECTION_CONTRACT_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "dataset": str(dataset_path),
        "dataset_sha256": _sha256_file(dataset_path),
        "dataset_split": "validation",
        "record_count": record_count,
        "model_id": model_id,
        "revision": revision,
        "contract_valid_rate": sum(row["contract_valid"] for row in results) / record_count,
        "exact_target_match_rate": sum(row["exact_target_match"] for row in results)
        / record_count,
        "error_counts": dict(errors),
        "profile_summary": profile.summary,
        "results": results,
    }
    _write_json_atomic(output_path, payload)
    return payload


def freeze_error_taxonomy(baseline_path: Path, output_path: Path) -> dict[str, Any]:
    """Freeze observed selector failures from a completed untuned baseline."""
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    if (
        not isinstance(baseline, dict)
        or baseline.get("status") != "passed"
        or baseline.get("task") != LESSON_SELECTION_CONTRACT_VERSION
        or not isinstance(baseline.get("results"), list)
    ):
        raise ValueError("Baseline is not a completed lesson-selection-2.0 evaluation")
    examples: defaultdict[str, list[str]] = defaultdict(list)
    counts: Counter[str] = Counter()
    for result in baseline["results"]:
        if not isinstance(result, dict):
            raise ValueError("Baseline result rows must be objects")
        record_id = str(result.get("record_id", ""))
        codes = result.get("error_codes")
        if not record_id or not isinstance(codes, list):
            raise ValueError("Baseline result row is incomplete")
        for code in codes:
            if code not in {
                "contract_invalid",
                "claim_selection_mismatch",
                "concept_selection_mismatch",
                "question_selection_mismatch",
                "hint_selection_mismatch",
            }:
                raise ValueError(f"Unknown baseline error code: {code}")
            counts[code] += 1
            if len(examples[code]) < 10:
                examples[code].append(record_id)
    record_count = int(baseline.get("record_count", 0))
    if record_count < 1 or record_count != len(baseline["results"]):
        raise ValueError("Baseline record count is invalid")
    payload = {
        "schema_version": "1.0",
        "status": "passed",
        "task": LESSON_SELECTION_CONTRACT_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "baseline": str(baseline_path),
        "baseline_sha256": _sha256_file(baseline_path),
        "record_count": record_count,
        "total_error_assignments": sum(counts.values()),
        "categories": [
            {
                "code": code,
                "count": count,
                "rate": count / record_count,
                "example_record_ids": examples[code],
            }
            for code, count in sorted(counts.items())
        ],
    }
    _write_json_atomic(output_path, payload)
    return payload


def _load_validation_records(path: Path, limit: int) -> list[dict[str, Any]]:
    if limit < 1:
        raise ValueError("Baseline limit must be positive")
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                evidence = EngineEvidence.model_validate(record["input"])
                rating = RatingBucket(record["meta"]["rating_bucket"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid baseline row at line {line_number}") from exc
            if record.get("meta", {}).get("split") != "validation":
                raise ValueError("Untuned baseline can only consume validation rows")
            if record.get("task") != LESSON_SELECTION_CONTRACT_VERSION:
                raise ValueError("Untuned baseline task contract is invalid")
            if record.get("prompt") != lesson_selection_prompt(evidence, rating):
                raise ValueError("Untuned baseline prompt contract is invalid")
            parse_lesson_selection(str(record["response"]), evidence, rating)
            records.append(record)
            if len(records) == limit:
                break
    if len(records) != limit:
        raise ValueError(f"Validation data has {len(records)} rows; {limit} required")
    return records


def _extract_object(output: str) -> dict[str, Any]:
    start, end = output.find("{"), output.rfind("}")
    value = json.loads(output[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("Model selection is not an object")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)
