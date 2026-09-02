from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def inspect_training_hardware() -> dict[str, object]:
    """Return non-sensitive facts needed to decide whether a local smoke run is viable."""
    memory_bytes: int | None = None
    chip: str | None = None
    if platform.system() == "Darwin":
        memory = subprocess.run(
            ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, check=False
        )
        if memory.returncode == 0 and memory.stdout.strip().isdigit():
            memory_bytes = int(memory.stdout.strip())
        processor = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            check=False,
        )
        chip = processor.stdout.strip() or None
    return {
        "system": platform.system(),
        "machine": platform.machine(),
        "chip": chip,
        "memory_bytes": memory_bytes,
    }


def evaluate_training_readiness(
    audit_path: Path,
    manifest_path: Path,
    output_path: Path,
    *,
    hardware: dict[str, object] | None = None,
    installed_versions: dict[str, str | None] | None = None,
) -> dict[str, Any]:
    """Evaluate the fail-closed gate before any post-training command may exist."""
    audit = _read_json(audit_path, "data audit")
    manifest = _read_json(manifest_path, "post-training manifest")
    hardware = hardware or inspect_training_hardware()
    toolchain = manifest.get("toolchain", {})
    if not isinstance(toolchain, dict):
        raise ValueError("post-training manifest toolchain must be an object")
    raw_pins = toolchain.get("packages", toolchain)
    if not isinstance(raw_pins, dict):
        raise ValueError("post-training manifest package pins must be an object")
    pins = {str(name): str(version) for name, version in raw_pins.items() if version}
    versions = installed_versions or {name: _installed_version(name) for name in pins}
    minimum_memory = int(manifest.get("hardware", {}).get("minimum_memory_bytes", 0))
    memory = hardware.get("memory_bytes")
    hardware_passed = (
        hardware.get("system") == "Darwin"
        and hardware.get("machine") == "arm64"
        and isinstance(memory, int)
        and memory >= minimum_memory
    )
    data_config = manifest.get("data", {})
    if not isinstance(data_config, dict):
        raise ValueError("post-training manifest data requirements must be an object")
    actual_counts = {
        "train": int(audit.get("training", {}).get("totals", {}).get("records", 0)),
        "validation": int(audit.get("validation", {}).get("totals", {}).get("records", 0)),
        "final_test": int(audit.get("evaluation", {}).get("totals", {}).get("records", 0)),
    }
    required_counts = {
        "train": int(data_config.get("minimum_train", 0)),
        "validation": int(data_config.get("minimum_validation", 0)),
        "final_test": int(data_config.get("minimum_final_test", 0)),
    }
    data_passed = (
        audit.get("schema_version") == "2.0"
        and audit.get("status") == "passed"
        and audit.get("gate", {}).get("ready_for_training") is True
        and all(actual_counts[name] >= required_counts[name] for name in actual_counts)
    )
    pins_complete = bool(pins) and all(pins.get(name) for name in versions)
    toolchain_passed = pins_complete and all(versions[name] == pins.get(name) for name in versions)
    model = manifest.get("model", {})
    model_passed = (
        isinstance(model, dict)
        and isinstance(model.get("native_base_model"), str)
        and isinstance(model.get("revision"), str)
        and isinstance(model.get("weight_sha256"), dict)
        and bool(model.get("weight_sha256"))
        and model.get("inference_quant_is_training_source") is False
    )
    evidence = manifest.get("evidence", {})
    evidence_passed = isinstance(evidence, dict) and _baseline_and_taxonomy_passed(
        evidence.get("frozen_baseline"),
        evidence.get("error_taxonomy"),
        int(data_config.get("baseline_minimum", 0)),
        int(data_config.get("minimum_repeated_error_count", 1)),
    )
    human_evidence_passed = isinstance(evidence, dict) and _human_evidence_passed(
        evidence.get("frozen_human_review"), data_config
    )
    checks = {
        "hardware_smoke_eligible": hardware_passed,
        "data_contract_and_isolation": data_passed,
        "toolchain_exactly_pinned_and_installed": toolchain_passed,
        "native_base_weights_pinned": model_passed,
        "error_and_baseline_evidence_frozen": evidence_passed,
        "two_reviewer_human_gold_adjudicated": human_evidence_passed,
    }
    ready = all(checks.values())
    authorization = manifest.get("authorization", {})
    smoke_authorized = (
        isinstance(authorization, dict) and authorization.get("smoke") is True
    )
    payload = {
        "schema_version": "1.0",
        "generated_at": datetime.now(UTC).isoformat(),
        "status": "ready_for_smoke" if ready else "blocked",
        "authorized_to_train": ready and smoke_authorized,
        "checks": checks,
        "hardware": hardware,
        "installed_versions": versions,
        "manifest": str(manifest_path),
        "data_audit": str(audit_path),
        "data_counts": {"actual": actual_counts, "required": required_counts},
        "blockers": [name for name, passed in checks.items() if not passed],
        "next_allowed_action": (
            "execute only the authorized 7-20 step isolated LoRA smoke run"
            if ready
            else "resolve blockers and rerun this gate; do not train"
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output_path)
    return payload


def _declared_evidence_passed(value: object) -> bool:
    if not isinstance(value, str) or not value or not Path(value).is_file():
        return False
    try:
        payload = _read_json(Path(value), "training evidence")
    except ValueError:
        return False
    return payload.get("status") == "passed"


def _baseline_and_taxonomy_passed(
    baseline_value: object,
    taxonomy_value: object,
    minimum: int,
    minimum_repeated_error_count: int,
) -> bool:
    if not all(
        isinstance(value, str) and value and Path(value).is_file()
        for value in (baseline_value, taxonomy_value)
    ):
        return False
    assert isinstance(baseline_value, str)
    assert isinstance(taxonomy_value, str)
    try:
        baseline = _read_json(Path(baseline_value), "untuned baseline evidence")
        taxonomy = _read_json(Path(taxonomy_value), "error taxonomy evidence")
    except ValueError:
        return False
    digest = hashlib.sha256(Path(baseline_value).read_bytes()).hexdigest()
    categories = taxonomy.get("categories")
    repeated_error = bool(
        isinstance(categories, list)
        and any(
            isinstance(category, dict)
            and category.get("count", 0) >= minimum_repeated_error_count
            for category in categories
        )
    )
    return bool(
        baseline.get("status") == "passed"
        and baseline.get("task") == "lesson-selection-2.0"
        and baseline.get("dataset_split") == "validation"
        and baseline.get("record_count", 0) >= minimum
        and taxonomy.get("status") == "passed"
        and taxonomy.get("task") == "lesson-selection-2.0"
        and taxonomy.get("record_count") == baseline.get("record_count")
        and taxonomy.get("baseline_sha256") == digest
        and repeated_error
    )


def _human_evidence_passed(value: object, data_config: dict[str, Any]) -> bool:
    if not isinstance(value, str) or not value or not Path(value).is_file():
        return False
    try:
        payload = _read_json(Path(value), "human training evidence")
    except ValueError:
        return False
    agreement = payload.get("exact_selection_agreement")
    rubric = payload.get("rubric_summary")
    means = rubric.get("mean_scores") if isinstance(rubric, dict) else None
    harmful_omissions = (
        rubric.get("harmful_omission_count") if isinstance(rubric, dict) else None
    )
    minimum_mean = float(data_config.get("human_rubric_minimum_mean", 0.0))
    return bool(
        payload.get("status") == "passed"
        and payload.get("reviewer_count") == int(data_config.get("reviewers_required", 2))
        and payload.get("record_count", 0) >= int(data_config.get("human_gold_minimum", 0))
        and isinstance(agreement, (int, float))
        and not isinstance(agreement, bool)
        and agreement >= float(data_config.get("agreement_minimum", 0.0))
        and payload.get("adjudication_complete") is True
        and isinstance(means, dict)
        and means
        and all(
            isinstance(score, (int, float))
            and not isinstance(score, bool)
            and score >= minimum_mean
            for score in means.values()
        )
        and isinstance(harmful_omissions, int)
        and not isinstance(harmful_omissions, bool)
        and harmful_omissions
        <= int(data_config.get("human_harmful_omission_maximum", 0))
    )


def _installed_version(package: str) -> str | None:
    normalized = package.replace("_", "-")
    try:
        return importlib.metadata.version(normalized)
    except importlib.metadata.PackageNotFoundError:
        return None


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is missing or invalid: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return value
