import hashlib
import json
from pathlib import Path

from gemmafischer.training_readiness import evaluate_training_readiness


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_readiness_fails_closed_with_real_blocker_names(tmp_path: Path) -> None:
    audit = tmp_path / "audit.json"
    manifest = tmp_path / "manifest.json"
    output = tmp_path / "readiness.json"
    write_json(
        audit,
        {"schema_version": "2.0", "status": "blocked", "gate": {"ready_for_training": False}},
    )
    write_json(
        manifest,
        {
            "hardware": {"minimum_memory_bytes": 16},
            "toolchain": {"unsloth": None, "unsloth_zoo": None, "mlx": None, "mlx_lm": None},
            "model": {"inference_quant_is_training_source": False, "weight_sha256": {}},
            "evidence": {},
        },
    )

    result = evaluate_training_readiness(
        audit,
        manifest,
        output,
        hardware={"system": "Darwin", "machine": "arm64", "memory_bytes": 18},
        installed_versions={"unsloth": None, "unsloth_zoo": None, "mlx": "1", "mlx_lm": "1"},
    )

    assert result["status"] == "blocked"
    assert result["authorized_to_train"] is False
    assert "data_contract_and_isolation" in result["blockers"]
    assert "toolchain_exactly_pinned_and_installed" in result["blockers"]
    assert output.exists()


def test_readiness_reports_smoke_eligible_but_never_authorizes_training(tmp_path: Path) -> None:
    write_json(
        tmp_path / "baseline.json",
        {
            "status": "passed",
            "task": "lesson-selection-2.0",
            "dataset_split": "validation",
            "record_count": 1,
        },
    )
    baseline_hash = hashlib.sha256((tmp_path / "baseline.json").read_bytes()).hexdigest()
    write_json(
        tmp_path / "taxonomy.json",
        {
            "status": "passed",
            "task": "lesson-selection-2.0",
            "record_count": 1,
            "baseline_sha256": baseline_hash,
            "categories": [{"code": "contract_invalid", "count": 1}],
        },
    )
    write_json(
        tmp_path / "human.json",
        {
            "status": "passed",
            "record_count": 1,
            "reviewer_count": 2,
            "exact_selection_agreement": 1.0,
            "adjudication_complete": True,
            "rubric_summary": {
                "mean_scores": {"correctness": 5.0, "clarity": 4.0},
                "harmful_omission_count": 0,
            },
        },
    )
    audit = tmp_path / "audit.json"
    manifest = tmp_path / "manifest.json"
    write_json(
        audit,
        {"schema_version": "2.0", "status": "passed", "gate": {"ready_for_training": True}},
    )
    versions = {"unsloth": "1", "unsloth_zoo": "2", "mlx": "3", "mlx_lm": "4"}
    write_json(
        manifest,
        {
            "hardware": {"minimum_memory_bytes": 16},
            "data": {
                "human_gold_minimum": 1,
                "baseline_minimum": 1,
                "minimum_repeated_error_count": 1,
                "reviewers_required": 2,
                "agreement_minimum": 0.67,
                "human_rubric_minimum_mean": 3.0,
                "human_harmful_omission_maximum": 0,
            },
            "toolchain": versions,
            "model": {
                "native_base_model": "provider/base",
                "revision": "abc123",
                "weight_sha256": {"model.safetensors": "f" * 64},
                "inference_quant_is_training_source": False,
            },
            "evidence": {
                "error_taxonomy": str(tmp_path / "taxonomy.json"),
                "frozen_baseline": str(tmp_path / "baseline.json"),
                "frozen_human_review": str(tmp_path / "human.json"),
            },
        },
    )

    result = evaluate_training_readiness(
        audit,
        manifest,
        tmp_path / "result.json",
        hardware={"system": "Darwin", "machine": "arm64", "memory_bytes": 18},
        installed_versions=versions,
    )

    assert result["status"] == "ready_for_smoke"
    assert result["authorized_to_train"] is False
