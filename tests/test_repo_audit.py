from pathlib import Path

from gemmafischer.repo_audit import audit_repository


def test_repository_has_no_archived_runtime_or_exact_duplicates() -> None:
    result = audit_repository(Path(__file__).parents[1])

    assert result["status"] == "passed", result["findings"]
