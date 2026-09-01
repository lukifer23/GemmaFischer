from __future__ import annotations

import hashlib
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
    ["uv", "run", "pytest", "-m", "not model and not hardware", "tests"],
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
            [
                "uv",
                "run",
                "--isolated",
                "--no-project",
                "--with",
                str(wheels[0]),
                "gemmafischer",
                "version",
            ]
        ):
            return ["isolated wheel smoke test failed"]
    return []


def verify_release_status(root: Path) -> list[str]:
    path = root / "assets" / "release-status.json"
    try:
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return ["assets/release-status.json is missing or invalid"]
    if not isinstance(payload, dict) or payload.get("schema_version") != "2.0":
        return ["release status schema must be 2.0"]
    findings: list[str] = []
    candidate = payload.get("candidate")
    if not isinstance(candidate, dict):
        return ["release status must contain a candidate receipt"]
    sha = candidate.get("sha")
    if not isinstance(sha, str) or len(sha) != 40:
        return ["candidate SHA must be a full 40-character commit"]
    head = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if head != sha:
        parent = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD^"],
            check=False,
            capture_output=True,
            text=True,
        ).stdout.strip()
        receipt_files = set(
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(root),
                    "diff-tree",
                    "--no-commit-id",
                    "--name-only",
                    "-r",
                    "HEAD",
                ],
                check=False,
                capture_output=True,
                text=True,
            ).stdout.splitlines()
        )
        allowed_receipt_files = {
            "assets/release-status.json",
            "docs/release-status.md",
        }
        if parent != sha or not receipt_files or not receipt_files <= allowed_receipt_files:
            findings.append(
                f"release receipt targets {sha}, but HEAD {head} is not its ledger-only child"
            )
    if candidate.get("clean_tree") is not True:
        findings.append("candidate receipt must assert a clean tree")
    dirty = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain"],
        check=False,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        findings.append("release verification requires a clean working tree")
    if payload.get("claim") != "local_release_gates_passed":
        findings.append("release claim must be local_release_gates_passed")
    receipts = payload.get("receipts")
    if not isinstance(receipts, list) or not receipts:
        findings.append("release status must contain command receipts")
    else:
        for index, receipt in enumerate(receipts):
            if not isinstance(receipt, dict):
                findings.append(f"receipt {index} is invalid")
                continue
            if receipt.get("conclusion") != "success" or not receipt.get("command"):
                findings.append(f"receipt {index} is not a successful command")
            environment = receipt.get("environment")
            if not isinstance(environment, dict) or not environment.get("python"):
                findings.append(f"receipt {index} lacks environment identity")
            artifacts = receipt.get("artifacts", [])
            if not isinstance(artifacts, list):
                findings.append(f"receipt {index} artifacts must be a list")
                continue
            for artifact in artifacts:
                if not isinstance(artifact, dict):
                    findings.append(f"receipt {index} contains an invalid artifact")
                    continue
                artifact_path = artifact.get("path")
                expected_hash = artifact.get("sha256")
                if not isinstance(artifact_path, str) or not isinstance(expected_hash, str):
                    findings.append(f"receipt {index} has an incomplete artifact hash")
                    continue
                content = subprocess.run(
                    ["git", "-C", str(root), "show", f"{sha}:{artifact_path}"],
                    check=False,
                    capture_output=True,
                )
                if content.returncode:
                    findings.append(f"receipt artifact is absent from candidate: {artifact_path}")
                elif hashlib.sha256(content.stdout).hexdigest() != expected_hash:
                    findings.append(f"receipt artifact hash mismatch: {artifact_path}")
    hosted = payload.get("hosted")
    required_hosted = {"verification", "codeql"}
    if not isinstance(hosted, list):
        findings.append("release status must contain hosted receipts")
    else:
        seen: set[str] = set()
        for receipt in hosted:
            if not isinstance(receipt, dict):
                continue
            kind = receipt.get("kind")
            if isinstance(kind, str):
                seen.add(kind)
            if (
                receipt.get("head_sha") != sha
                or receipt.get("conclusion") != "success"
                or not isinstance(receipt.get("run_id"), int)
            ):
                findings.append(f"hosted {kind or 'unknown'} receipt is stale or unsuccessful")
        missing = sorted(required_hosted - seen)
        if missing:
            findings.append("missing hosted receipts: " + ", ".join(missing))
    alerts = payload.get("security_alerts")
    if not isinstance(alerts, dict) or any(
        alerts.get(key) != 0 for key in ("code_scanning", "dependabot", "secret_scanning")
    ):
        findings.append("security alert receipt must report zero open alerts")
    external = payload.get("external_gates")
    if not isinstance(external, list) or not external:
        findings.append("release status must preserve typed external gates")
    elif any(
        not isinstance(gate, dict) or gate.get("status") not in {"open", "passed", "blocked"}
        for gate in external
    ):
        findings.append("external gates contain an invalid status")
    return findings


def portable_findings(root: Path) -> list[str]:
    findings = []
    audit = audit_repository(root)
    if audit["status"] != "passed":
        findings.append("repository audit failed: " + json.dumps(audit["findings"], sort_keys=True))
    findings.extend(verify_openapi(root))
    findings.extend(verify_package(root))
    return findings
