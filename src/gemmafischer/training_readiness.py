from __future__ import annotations

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
    pins = manifest.get("toolchain", {})
    if not isinstance(pins, dict):
        raise ValueError("post-training manifest toolchain must be an object")
    versions = installed_versions or {
        name: _installed_version(name) for name in ("unsloth", "unsloth_zoo", "mlx", "mlx_lm")
    }
    minimum_memory = int(manifest.get("hardware", {}).get("minimum_memory_bytes", 0))
    memory = hardware.get("memory_bytes")
    hardware_passed = (
        hardware.get("system") == "Darwin"
        and hardware.get("machine") == "arm64"
        and isinstance(memory, int)
        and memory >= minimum_memory
    )
    data_passed = (
        audit.get("schema_version") == "2.0"
        and audit.get("status") == "passed"
        and audit.get("gate", {}).get("ready_for_training") is True
    )
    pins_complete = all(isinstance(pins.get(name), str) and pins.get(name) for name in versions)
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
    evidence_passed = isinstance(evidence, dict) and all(
        _declared_file_exists(evidence.get(name))
        for name in ("error_taxonomy", "frozen_baseline", "frozen_human_review")
    )
    checks = {
        "hardware_smoke_eligible": hardware_passed,
        "data_contract_and_isolation": data_passed,
        "toolchain_exactly_pinned_and_installed": toolchain_passed,
        "native_base_weights_pinned": model_passed,
        "error_and_baseline_evidence_frozen": evidence_passed,
    }
    ready = all(checks.values())
    payload = {
        "schema_version": "1.0",
        "generated_at": datetime.now(UTC).isoformat(),
        "status": "ready_for_smoke" if ready else "blocked",
        "authorized_to_train": False,
        "checks": checks,
        "hardware": hardware,
        "installed_versions": versions,
        "manifest": str(manifest_path),
        "data_audit": str(audit_path),
        "blockers": [name for name, passed in checks.items() if not passed],
        "next_allowed_action": (
            "request explicit authorization for a 7-20 step isolated LoRA smoke run"
            if ready
            else "resolve blockers and rerun this gate; do not train"
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output_path)
    return payload


def _declared_file_exists(value: object) -> bool:
    return isinstance(value, str) and bool(value) and Path(value).is_file()


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
