from pathlib import Path

from gemmafischer.repo_audit import audit_repository


def test_repository_has_no_archived_runtime_or_exact_duplicates() -> None:
    result = audit_repository(Path(__file__).parents[1])

    assert result["status"] == "passed", result["findings"]


def test_repository_audit_rejects_legacy_root_package_and_broken_docs(
    tmp_path: Path, monkeypatch
) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "__init__.py").write_text("# legacy\n")
    (tmp_path / "README.md").write_text("[missing](docs/nope.md)\n")
    monkeypatch.setattr(
        "gemmafischer.repo_audit.subprocess.run",
        lambda *args, **kwargs: type(
            "Result", (), {"stdout": "src/__init__.py\nREADME.md\n"}
        )(),
    )

    result = audit_repository(tmp_path)

    assert result["status"] == "blocked"
    assert result["findings"]["forbidden_paths"] == ["src/__init__.py"]
    assert result["findings"]["broken_local_links"] == [
        {"path": "README.md", "target": "docs/nope.md"}
    ]
