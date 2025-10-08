#!/usr/bin/env python3
"""Quick health check for MoE adapters.

Scans the expected checkpoint directories and reports whether a usable
adapter is present for each expert. Outputs a JSON summary to
``reports/moe_health.json`` and prints a short status table.
"""
import json
import os
from pathlib import Path
from typing import Dict, Optional

ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_ROOT = ROOT / "checkpoints"
REPORT_DIR = ROOT / "reports"
REPORT_PATH = REPORT_DIR / "moe_health.json"

EXPECTED_EXPERTS = {
    "uci": CHECKPOINT_ROOT / "lora_uci",
    "tutor": CHECKPOINT_ROOT / "lora_tutor",
    "director": CHECKPOINT_ROOT / "lora_director",
}


def find_latest_adapter(dir_path: Path) -> Optional[Path]:
    if not dir_path.exists():
        return None
    checkpoint_dirs = [p for p in dir_path.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")]
    if checkpoint_dirs:
        def checkpoint_step(path: Path) -> int:
            try:
                return int(path.name.split("-")[-1])
            except ValueError:
                return -1
        checkpoint_dirs.sort(key=checkpoint_step, reverse=True)
        for candidate in checkpoint_dirs:
            if (candidate / "adapter_model.safetensors").exists() or (candidate / "adapter_config.json").exists():
                return candidate
    # fallback: directory itself might contain adapter files
    if (dir_path / "adapter_model.safetensors").exists() or (dir_path / "adapter_config.json").exists():
        return dir_path
    return None


def check_adapters() -> Dict[str, Dict[str, Optional[str]]]:
    results: Dict[str, Dict[str, Optional[str]]] = {}
    for expert, path in EXPECTED_EXPERTS.items():
        latest = find_latest_adapter(path)
        results[expert] = {
            "expected_path": str(path),
            "status": "available" if latest else "missing",
            "adapter_path": str(latest) if latest else None,
        }
    return results


def main() -> None:
    results = check_adapters()
    REPORT_DIR.mkdir(exist_ok=True)
    REPORT_PATH.write_text(json.dumps({"adapters": results}, indent=2))

    print("MoE adapter health:")
    for expert, data in results.items():
        status = data["status"].upper()
        location = data["adapter_path"] or "(not found)"
        print(f" - {expert:8s}: {status:9s} -> {location}")
    print(f"\nSummary saved to {REPORT_PATH}")


if __name__ == "__main__":
    main()
