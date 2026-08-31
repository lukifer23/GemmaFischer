from __future__ import annotations

import hashlib

import pytest

from gemmafischer.lmstudio import sha256_file, validate_loopback_base_url


@pytest.mark.parametrize(
    "url",
    (
        "http://127.0.0.1:1234/v1",
        "http://localhost:1234/v1/",
        "http://[::1]:1234/v1",
    ),
)
def test_loopback_model_endpoint_is_accepted(url: str) -> None:
    assert validate_loopback_base_url(url).endswith("/v1")


@pytest.mark.parametrize(
    "url",
    (
        "https://127.0.0.1:1234/v1",
        "http://192.168.1.10:1234/v1",
        "http://example.com/v1",
        "http://user:secret@127.0.0.1:1234/v1",
    ),
)
def test_non_loopback_or_credentialed_model_endpoint_is_rejected(url: str) -> None:
    with pytest.raises(ValueError, match="loopback|credentials"):
        validate_loopback_base_url(url)


def test_model_artifact_hash_reads_real_file_bytes(tmp_path) -> None:
    artifact = tmp_path / "weights.safetensors"
    artifact.write_bytes(b"exact-local-model-bytes")
    assert sha256_file(artifact) == hashlib.sha256(b"exact-local-model-bytes").hexdigest()
