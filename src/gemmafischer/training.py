from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .domain import EngineEvidence, RatingBucket
from .runtime import (
    LESSON_SELECTION_CONTRACT_VERSION,
    LESSON_SELECTION_SYSTEM_PROMPT,
    lesson_selection_prompt,
    parse_lesson_selection,
)
from .training_readiness import evaluate_training_readiness

MACHINE_SUPERVISION_AUTHORITY = "stockfish-deterministic-v2"


def prepare_mlx_dataset(
    source_dir: Path, output_dir: Path, *, human_gold_path: Path | None = None
) -> dict[str, Any]:
    """Validate canonical records and atomically emit MLX chat partitions."""
    human_gold_sha256: str | None = None
    if human_gold_path is not None:
        human_gold = _read_json(human_gold_path)
        application = _read_json(source_dir / "human-gold-application.json")
        human_gold_sha256 = sha256_file(human_gold_path)
        if (
            human_gold.get("status") != "passed"
            or human_gold.get("adjudication_complete") is not True
            or application.get("status") != "passed"
            or application.get("human_gold_sha256") != human_gold_sha256
        ):
            raise ValueError("Prepared source is not bound to passing human gold")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(
            "Prepared-data destination must be empty; duplicate datasets are forbidden"
        )
    mapping = {"train": "train", "validation": "valid", "final_test": "test"}
    output_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    hashes: dict[str, str] = {}
    source_hashes: dict[str, str] = {}
    for source_name, output_name in mapping.items():
        source = source_dir / f"{source_name}.jsonl"
        if not source.is_file():
            raise ValueError(f"Missing canonical partition: {source}")
        source_hashes[source.name] = sha256_file(source)
        destination = output_dir / f"{output_name}.jsonl"
        temporary = destination.with_suffix(".jsonl.tmp")
        count = 0
        with (
            source.open(encoding="utf-8") as reader,
            temporary.open("w", encoding="utf-8") as writer,
        ):
            for line_number, line in enumerate(reader, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    evidence = EngineEvidence.model_validate(record["input"])
                    rating = RatingBucket(record["meta"]["rating_bucket"])
                    response = str(record["response"])
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise ValueError(f"Invalid record at {source}:{line_number}") from exc
                if record.get("task") != LESSON_SELECTION_CONTRACT_VERSION:
                    raise ValueError(f"Unsupported task at {source}:{line_number}")
                if record.get("system_prompt") != LESSON_SELECTION_SYSTEM_PROMPT:
                    raise ValueError(f"System prompt drift at {source}:{line_number}")
                if record.get("prompt") != lesson_selection_prompt(evidence, rating):
                    raise ValueError(f"User prompt drift at {source}:{line_number}")
                parse_lesson_selection(response, evidence, rating)
                writer.write(
                    json.dumps(
                        {
                            "messages": [
                                {"role": "system", "content": LESSON_SELECTION_SYSTEM_PROMPT},
                                {"role": "user", "content": record["prompt"]},
                                {"role": "assistant", "content": response},
                            ],
                            "record_id": record["record_id"],
                        },
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                count += 1
        if count == 0:
            temporary.unlink(missing_ok=True)
            raise ValueError(f"Canonical partition is empty: {source}")
        temporary.replace(destination)
        counts[output_name] = count
        hashes[destination.name] = sha256_file(destination)
    receipt = {
        "schema_version": "1.0",
        "task": LESSON_SELECTION_CONTRACT_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "counts": counts,
        "sha256": hashes,
        "source_sha256": source_hashes,
        "supervision_authority": (
            "two-reviewer-adjudicated-human-gold"
            if human_gold_sha256
            else MACHINE_SUPERVISION_AUTHORITY
        ),
        "human_gold_sha256": human_gold_sha256,
    }
    _write_json_atomic(output_dir / "dataset-receipt.json", receipt)
    return receipt


def training_preflight(
    manifest_path: Path,
    audit_path: Path,
    model_path: Path,
    data_path: Path,
    output_path: Path,
    config_path: Path,
    *,
    hardware: dict[str, object] | None = None,
    installed_versions: dict[str, str | None] | None = None,
) -> dict[str, Any]:
    readiness = evaluate_training_readiness(
        audit_path,
        manifest_path,
        output_path,
        hardware=hardware,
        installed_versions=installed_versions,
    )
    manifest = _read_json(manifest_path)
    model = manifest.get("model", {})
    expected = model.get("weight_sha256", {}) if isinstance(model, dict) else {}
    model_checks: dict[str, dict[str, object]] = {}
    if isinstance(expected, dict):
        for relative, digest in expected.items():
            path = model_path / str(relative)
            actual = sha256_file(path) if path.is_file() else None
            model_checks[str(relative)] = {
                "expected": digest,
                "actual": actual,
                "passed": actual == digest,
            }
    data_receipt = data_path / "dataset-receipt.json"
    prepared_receipt = _read_json(data_receipt) if data_receipt.is_file() else {}
    expected_data_hashes = prepared_receipt.get("sha256", {})
    data_file_checks: dict[str, dict[str, object]] = {}
    if isinstance(expected_data_hashes, dict):
        for name in ("train.jsonl", "valid.jsonl", "test.jsonl"):
            path = data_path / name
            expected_hash = expected_data_hashes.get(name)
            actual_hash = sha256_file(path) if path.is_file() else None
            data_file_checks[name] = {
                "expected": expected_hash,
                "actual": actual_hash,
                "passed": bool(expected_hash and actual_hash == expected_hash),
            }
    evidence = manifest.get("evidence", {})
    human_path = evidence.get("frozen_human_review") if isinstance(evidence, dict) else None
    human_gold_matches: bool | None = (
        bool(
            isinstance(human_path, str)
            and Path(human_path).is_file()
            and prepared_receipt.get("human_gold_sha256") == sha256_file(Path(human_path))
        )
        if human_path is not None
        else None
    )
    audit_source_hashes = _audit_source_hashes(_read_json(audit_path))
    prepared_source_hashes = prepared_receipt.get("source_sha256")
    source_hashes_match = bool(
        isinstance(prepared_source_hashes, dict) and prepared_source_hashes == audit_source_hashes
    )
    data_config = manifest.get("data", {})
    required_supervision = (
        data_config.get("training_supervision_authority") if isinstance(data_config, dict) else None
    )
    supervision_matches = bool(
        required_supervision == MACHINE_SUPERVISION_AUTHORITY
        and prepared_receipt.get("supervision_authority") == required_supervision
        and prepared_receipt.get("human_gold_sha256") is None
    )
    data_ready = bool(
        data_receipt.is_file()
        and len(data_file_checks) == 3
        and all(check["passed"] for check in data_file_checks.values())
        and prepared_receipt.get("task") == LESSON_SELECTION_CONTRACT_VERSION
        and source_hashes_match
        and supervision_matches
    )
    disk_root = model_path.parent if model_path.parent.exists() else Path.cwd()
    free_bytes = shutil.disk_usage(disk_root).free
    expected_config_hash = str(manifest.get("training", {}).get("recipe_sha256", ""))
    config_hash = sha256_file(config_path) if config_path.is_file() else None
    config_ready = bool(expected_config_hash and config_hash == expected_config_hash)
    preflight = {
        **readiness,
        "schema_version": "2.0",
        "model_path": str(model_path.resolve()),
        "model_files": model_checks,
        "model_hashes_passed": bool(model_checks)
        and all(bool(item["passed"]) for item in model_checks.values()),
        "prepared_data": str(data_path.resolve()),
        "prepared_data_passed": data_ready,
        "prepared_data_files": data_file_checks,
        "prepared_human_gold_matches": human_gold_matches,
        "prepared_source_hashes_match_audit": source_hashes_match,
        "prepared_supervision_authority": prepared_receipt.get("supervision_authority"),
        "prepared_data_receipt_sha256": (
            sha256_file(data_receipt) if data_receipt.is_file() else None
        ),
        "free_bytes": free_bytes,
        "production_authorized": bool(
            isinstance(manifest.get("authorization"), dict)
            and manifest["authorization"].get("production") is True
        ),
        "training_config": str(config_path.resolve()),
        "training_config_sha256": config_hash,
        "training_config_passed": config_ready,
    }
    preflight["smoke_ready"] = bool(
        readiness["status"] == "ready_for_smoke"
        and readiness["authorized_to_train"] is True
        and preflight["model_hashes_passed"]
        and data_ready
        and config_ready
        and free_bytes >= int(manifest.get("hardware", {}).get("minimum_free_bytes", 0))
    )
    preflight_blockers = list(readiness.get("blockers", []))
    if readiness.get("authorized_to_train") is not True:
        preflight_blockers.append("smoke_training_not_authorized")
    if not preflight["model_hashes_passed"]:
        preflight_blockers.append("native_model_files_hash_verified")
    if not data_ready:
        preflight_blockers.append("prepared_machine_supervision_data_hash_verified")
    if not config_ready:
        preflight_blockers.append("training_recipe_hash_verified")
    if free_bytes < int(manifest.get("hardware", {}).get("minimum_free_bytes", 0)):
        preflight_blockers.append("minimum_free_disk")
    preflight["blockers"] = preflight_blockers
    _write_json_atomic(output_path, preflight)
    return preflight


def validate_training_preflight(
    preflight_path: Path,
    model_path: Path,
    data_path: Path,
    config_path: Path,
    *,
    production: bool,
) -> dict[str, Any]:
    """Re-bind a passing preflight to unchanged model and prepared-data files."""
    preflight = _read_json(preflight_path)
    if preflight.get("smoke_ready") is not True:
        raise ValueError("Training preflight is not smoke-ready")
    if production and preflight.get("production_authorized") is not True:
        raise ValueError("Production training is not explicitly authorized")
    if preflight.get("model_path") != str(model_path.resolve()):
        raise ValueError("Training model path does not match the preflight")
    if preflight.get("prepared_data") != str(data_path.resolve()):
        raise ValueError("Training data path does not match the preflight")
    if (
        preflight.get("training_config") != str(config_path.resolve())
        or not config_path.is_file()
        or sha256_file(config_path) != preflight.get("training_config_sha256")
    ):
        raise ValueError("Training recipe changed after preflight")
    model_files = preflight.get("model_files")
    if not isinstance(model_files, dict) or not model_files:
        raise ValueError("Training preflight contains no model file authority")
    for relative, check in model_files.items():
        if not isinstance(check, dict) or not isinstance(check.get("expected"), str):
            raise ValueError("Training preflight model file authority is invalid")
        path = model_path / str(relative)
        if not path.is_file() or sha256_file(path) != check["expected"]:
            raise ValueError(f"Training model file changed after preflight: {relative}")
    receipt = data_path / "dataset-receipt.json"
    if not receipt.is_file() or sha256_file(receipt) != preflight.get(
        "prepared_data_receipt_sha256"
    ):
        raise ValueError("Prepared training data changed after preflight")
    data_files = preflight.get("prepared_data_files")
    if not isinstance(data_files, dict) or len(data_files) != 3:
        raise ValueError("Training preflight contains no prepared-data file authority")
    for relative, check in data_files.items():
        if not isinstance(check, dict) or not isinstance(check.get("expected"), str):
            raise ValueError("Training preflight prepared-data authority is invalid")
        path = data_path / str(relative)
        if not path.is_file() or sha256_file(path) != check["expected"]:
            raise ValueError(f"Prepared training file changed after preflight: {relative}")
    return preflight


def run_mlx_sft(
    *,
    model_path: Path,
    data_path: Path,
    adapter_path: Path,
    receipt_path: Path,
    iterations: int,
    max_seq_length: int,
    smoke: bool,
    config_path: Path,
) -> dict[str, Any]:
    """Run one real MLX-LM LoRA job and retain exactly one adapter checkpoint."""
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise ValueError("MLX training requires Apple Silicon macOS")
    if iterations < 7 if smoke else iterations < 1:
        raise ValueError("Smoke training requires at least 7 iterations")
    if smoke and iterations > 20:
        raise ValueError("Smoke training cannot exceed 20 iterations")
    if max_seq_length not in {1024, 2048, 4096}:
        raise ValueError("max sequence length must be 1024, 2048, or 4096")
    if adapter_path.exists() and any(adapter_path.iterdir()):
        raise ValueError("Adapter destination must be empty; duplicate checkpoints are forbidden")
    adapter_path.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "gemmafischer.mlx_training",
        "--model",
        str(model_path.resolve()),
        "--train",
        "--data",
        str(data_path.resolve()),
        "--fine-tune-type",
        "lora",
        "--config",
        str(config_path.resolve()),
        "--optimizer",
        "adamw",
        "--num-layers",
        "4" if smoke else "16",
        "--batch-size",
        "1",
        "--grad-accumulation-steps",
        "16",
        "--iters",
        str(iterations),
        "--learning-rate",
        "0.0002",
        "--adapter-path",
        str(adapter_path.resolve()),
        "--save-every",
        str(iterations),
        "--steps-per-eval",
        str(iterations),
        "--val-batches",
        "25",
        "--max-seq-length",
        str(max_seq_length),
        "--mask-prompt",
        "--seed",
        "3407",
        "--grad-checkpoint",
    ]
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = receipt_path.with_suffix(".log")
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
    duration = time.monotonic() - started
    adapters = sorted(adapter_path.glob("*.safetensors"))
    if result.returncode or len(adapters) != 1:
        raise RuntimeError(
            f"MLX training failed or produced {len(adapters)} adapters; inspect {log_path}"
        )
    receipt = {
        "schema_version": "1.0",
        "status": "passed",
        "stage": "smoke" if smoke else "sft",
        "iterations": iterations,
        "duration_seconds": duration,
        "model_path": str(model_path.resolve()),
        "data_receipt_sha256": sha256_file(data_path / "dataset-receipt.json"),
        "adapter": str(adapters[0].resolve()),
        "adapter_sha256": sha256_file(adapters[0]),
        "log": str(log_path.resolve()),
        "log_sha256": sha256_file(log_path),
        "command": command,
    }
    _write_json_atomic(receipt_path, receipt)
    return receipt


def package_training_artifact(
    adapter_path: Path, receipts: list[Path], output_path: Path
) -> dict[str, Any]:
    adapters = sorted(adapter_path.glob("*.safetensors"))
    if len(adapters) != 1:
        raise ValueError("Exactly one adapter must exist before packaging")
    members = [adapters[0], *receipts]
    if any(not item.is_file() for item in members):
        raise ValueError("Every declared training receipt must exist")
    if len({item.name for item in members}) != len(members):
        raise ValueError("Packaged adapter and receipts must have unique filenames")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with tarfile.open(temporary, "w:gz") as archive:
        for item in members:
            archive.add(item, arcname=item.name, recursive=False)
    temporary.replace(output_path)
    result = {
        "path": str(output_path.resolve()),
        "sha256": sha256_file(output_path),
        "bytes": output_path.stat().st_size,
        "members": {item.name: sha256_file(item) for item in members},
    }
    _write_json_atomic(output_path.with_suffix(output_path.suffix + ".json"), result)
    return result


def acquire_native_training_model(manifest_path: Path, receipt_path: Path) -> dict[str, Any]:
    """Acquire and hash-verify the one revision-pinned native training base."""
    from huggingface_hub import snapshot_download

    manifest = _read_json(manifest_path)
    model = manifest.get("model")
    if not isinstance(model, dict):
        raise ValueError("Post-training manifest model authority is missing")
    model_id = model.get("native_base_model")
    revision = model.get("revision")
    expected = model.get("weight_sha256")
    if (
        not isinstance(model_id, str)
        or not model_id
        or not isinstance(revision, str)
        or not revision
        or not isinstance(expected, dict)
        or not expected
        or not all(
            isinstance(name, str) and isinstance(digest, str) for name, digest in expected.items()
        )
    ):
        raise ValueError("Native model ID, revision, and file hashes must be pinned")
    snapshot = Path(
        snapshot_download(
            repo_id=model_id,
            revision=revision,
            allow_patterns=sorted(expected),
        )
    ).resolve()
    snapshots = sorted(path for path in snapshot.parent.iterdir() if path.is_dir())
    if snapshots != [snapshot]:
        raise ValueError(
            f"Native base cache contains {len(snapshots)} snapshots; exactly one is permitted"
        )
    checks: dict[str, dict[str, object]] = {}
    for relative, digest in expected.items():
        path = snapshot / relative
        actual = sha256_file(path) if path.is_file() else None
        checks[relative] = {"expected": digest, "actual": actual, "passed": actual == digest}
    if not checks or not all(bool(check["passed"]) for check in checks.values()):
        raise ValueError("Native base download failed exact file-hash verification")
    receipt = {
        "schema_version": "1.0",
        "status": "passed",
        "model_id": model_id,
        "revision": revision,
        "snapshot_path": str(snapshot),
        "snapshot_count": 1,
        "files": checks,
        "generated_at": datetime.now(UTC).isoformat(),
    }
    _write_json_atomic(receipt_path, receipt)
    return receipt


def _audit_source_hashes(audit: dict[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for section in ("training", "validation", "evaluation"):
        payload = audit.get(section)
        files = payload.get("files") if isinstance(payload, dict) else None
        if not isinstance(files, list):
            return {}
        for item in files:
            if not isinstance(item, dict):
                return {}
            path = item.get("path")
            digest = item.get("sha256")
            if not isinstance(path, str) or not isinstance(digest, str):
                return {}
            result[Path(path).name] = digest
    return result


def sha256_file(path: Path) -> str:
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


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value
