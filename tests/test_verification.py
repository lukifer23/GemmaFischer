import hashlib
import json
import subprocess
from pathlib import Path

from gemmafischer.verification import verify_release_status


def test_release_receipt_accepts_only_ledger_child_of_candidate(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "release@example.invalid"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Release Test"], cwd=tmp_path, check=True
    )
    artifact = tmp_path / "artifact.txt"
    artifact.write_text("candidate evidence\n", encoding="utf-8")
    subprocess.run(["git", "add", "artifact.txt"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "candidate"], cwd=tmp_path, check=True)
    candidate = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assets = tmp_path / "assets"
    assets.mkdir()
    payload = {
        "schema_version": "2.0",
        "claim": "local_release_gates_passed",
        "candidate": {"sha": candidate, "clean_tree": True},
        "receipts": [
            {
                "command": "verify",
                "conclusion": "success",
                "environment": {"python": "3.12"},
                "artifacts": [
                    {
                        "path": "artifact.txt",
                        "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                    }
                ],
            }
        ],
        "hosted": [
            {
                "kind": kind,
                "run_id": index,
                "head_sha": candidate,
                "conclusion": "success",
            }
            for index, kind in enumerate(("verification", "codeql"), 1)
        ],
        "security_alerts": {
            "code_scanning": 0,
            "dependabot": 0,
            "secret_scanning": 0,
        },
        "external_gates": [{"code": "HUMAN", "status": "open"}],
    }
    (assets / "release-status.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    subprocess.run(["git", "add", "assets/release-status.json"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "release receipt"], cwd=tmp_path, check=True)

    assert verify_release_status(tmp_path) == []


def test_release_receipt_rejects_legacy_schema(tmp_path: Path) -> None:
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "release-status.json").write_text(
        json.dumps({"schema_version": "1.0"}), encoding="utf-8"
    )

    assert verify_release_status(tmp_path) == ["release status schema must be 2.0"]
