import hashlib
from pathlib import Path

import pytest

from gemmafischer.dataset import acquire_source, load_source


def test_pinned_lichess_source_has_license_and_hash() -> None:
    source = load_source(Path("data/sources.json"), "lichess-puzzles-2026-08-02")

    assert source["license"] == "CC0-1.0"
    assert len(source["sha256"]) == 64
    assert source["url"].startswith("https://database.lichess.org/")


def test_acquisition_verifies_content_before_atomic_publish(tmp_path: Path) -> None:
    content = b"licensed-source\n"
    source_path = tmp_path / "source.bin"
    source_path.write_bytes(content)
    source = {
        "id": "fixture",
        "url": source_path.as_uri(),
        "sha256": hashlib.sha256(content).hexdigest(),
    }
    output = tmp_path / "raw" / "source.bin"

    result = acquire_source(source, output)

    assert result["sha256"] == source["sha256"]
    assert output.read_bytes() == content


def test_acquisition_rejects_hash_mismatch_without_publishing(tmp_path: Path) -> None:
    source_path = tmp_path / "source.bin"
    source_path.write_bytes(b"unexpected")
    output = tmp_path / "published.bin"

    with pytest.raises(ValueError, match="hash mismatch"):
        acquire_source(
            {"id": "fixture", "url": source_path.as_uri(), "sha256": "0" * 64},
            output,
        )

    assert not output.exists()
