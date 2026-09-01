from __future__ import annotations

import json
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import Any

from .repo_audit import audit_repository
from .web import create_app

PORTABLE_COMMANDS = (
    ["uv", "run", "ruff", "check", "src", "tests", "scripts", "data"],
    ["uv", "run", "mypy"],
    ["uv", "pip", "check"],
    [
        "uv",
        "run",
        "pytest",
        "--cov=gemmafischer",
        "--cov-report=term-missing",
        "--cov-fail-under=70",
        "-m",
        "not model",
        "tests",
    ],
    ["node", "--check", "src/gemmafischer/static/app.js"],
)


def run_command(command: list[str]) -> int:
    print("+ " + " ".join(command), flush=True)
    return subprocess.run(command, check=False).returncode


def verify_openapi(root: Path) -> list[str]:
    expected_path = root / "docs" / "openapi.json"
    generated = json.dumps(create_app().openapi(), sort_keys=True, indent=2) + "\n"
    try:
        expected = expected_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ["docs/openapi.json is missing"]
    return [] if expected == generated else ["docs/openapi.json differs from the generated API"]


def verify_package(root: Path) -> list[str]:
    required = {
        "gemmafischer/static/index.html",
        "gemmafischer/resources/assets/model-manifest.json",
        "gemmafischer/resources/data/evaluation/diagnostic_positions.jsonl",
    }
    with tempfile.TemporaryDirectory(prefix="gemmafischer-build-") as directory:
        command = ["uv", "build", "--out-dir", directory]
        if run_command(command):
            return ["wheel and source distribution build failed"]
        wheels = tuple(Path(directory).glob("*.whl"))
        if len(wheels) != 1:
            return [f"expected one wheel, found {len(wheels)}"]
        with zipfile.ZipFile(wheels[0]) as archive:
            missing = sorted(required.difference(archive.namelist()))
        if missing:
            return ["wheel is missing: " + ", ".join(missing)]
        if run_command(
            ["uv", "run", "--isolated", "--with", str(wheels[0]), "gemmafischer", "version"]
        ):
            return ["isolated wheel smoke test failed"]
    return []


def verify_release_status(root: Path) -> list[str]:
    path = root / "assets" / "release-status.json"
    try:
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return ["assets/release-status.json is missing or invalid"]
    if not isinstance(payload, dict) or payload.get("schema_version") != "1.0":
        return ["release status schema must be 1.0"]
    blockers = payload.get("automated_blockers")
    if not isinstance(blockers, list):
        return ["release status must list automated_blockers"]
    return [str(blocker) for blocker in blockers]


def portable_findings(root: Path) -> list[str]:
    findings = []
    audit = audit_repository(root)
    if audit["status"] != "passed":
        findings.append("repository audit failed: " + json.dumps(audit["findings"], sort_keys=True))
    findings.extend(verify_openapi(root))
    findings.extend(verify_package(root))
    return findings
