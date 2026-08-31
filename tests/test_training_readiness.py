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
    for name in ("taxonomy.json", "baseline.json", "human.json"):
        write_json(tmp_path / name, {})
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
