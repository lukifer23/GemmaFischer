from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

from .domain import EngineEvidence, RatingBucket
from .runtime import (
    LESSON_SELECTION_SYSTEM_PROMPT,
    ModelClaimSelection,
    lesson_selection_prompt,
    parse_lesson_selection,
)

DEFAULT_LM_STUDIO_URL = "http://127.0.0.1:1234/v1"
DEFAULT_LFM_MODEL = "lfm2.5-2.6b-mlx"


class LMStudioUnavailable(RuntimeError):
    pass


def validate_loopback_base_url(base_url: str) -> str:
    parsed = urlparse(base_url.rstrip("/"))
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "::1", "localhost"}:
        raise ValueError("LM Studio endpoint must be a literal loopback HTTP URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("LM Studio endpoint must not contain credentials, query, or fragment")
    return base_url.rstrip("/")


def sha256_file(path: Path) -> str:
    from hashlib import sha256

    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class LMStudioIdentity:
    base_url: str
    model_id: str
    model_artifact: str
    model_artifact_bytes: int
    model_artifact_sha256: str
    catalog_model_ids: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "base_url": self.base_url,
            "model_id": self.model_id,
            "model_artifact": self.model_artifact,
            "model_artifact_bytes": self.model_artifact_bytes,
            "model_artifact_sha256": self.model_artifact_sha256,
            "catalog_model_ids": list(self.catalog_model_ids),
        }


class LMStudioRuntime:
    """Loopback-only OpenAI-compatible runtime for a verified local model file."""

    source: Literal["lfm"] = "lfm"

    def __init__(
        self,
        model_id: str = DEFAULT_LFM_MODEL,
        *,
        base_url: str = DEFAULT_LM_STUDIO_URL,
        model_artifact: Path,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.base_url = validate_loopback_base_url(base_url)
        self.model_id = model_id
        self.timeout_seconds = timeout_seconds
        self.identity = verify_lmstudio_identity(
            self.base_url, self.model_id, model_artifact, timeout_seconds
        )

    def select_claims(
        self, evidence: EngineEvidence, rating: RatingBucket
    ) -> ModelClaimSelection:
        payload = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": LESSON_SELECTION_SYSTEM_PROMPT},
                {"role": "user", "content": lesson_selection_prompt(evidence, rating)},
            ],
            "temperature": 0,
            "max_tokens": 768,
            "stream": False,
        }
        response = _json_request(
            f"{self.base_url}/chat/completions", payload, self.timeout_seconds
        )
        try:
            returned_model = str(response["model"])
            output = str(response["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as exc:
            raise LMStudioUnavailable("LM Studio returned an invalid chat response") from exc
        if returned_model != self.model_id:
            raise LMStudioUnavailable(
                f"LM Studio returned model {returned_model!r}, expected {self.model_id!r}"
            )
        return parse_lesson_selection(output, evidence, rating)


def verify_lmstudio_identity(
    base_url: str,
    model_id: str,
    model_artifact: Path,
    timeout_seconds: float = 10.0,
) -> LMStudioIdentity:
    base_url = validate_loopback_base_url(base_url)
    artifact = model_artifact.resolve()
    if not artifact.is_file():
        raise LMStudioUnavailable(f"Local model artifact is missing: {artifact}")
    catalog = _json_request(f"{base_url}/models", None, timeout_seconds)
    try:
        catalog_ids = tuple(sorted(str(item["id"]) for item in catalog["data"]))
    except (KeyError, TypeError) as exc:
        raise LMStudioUnavailable("LM Studio returned an invalid model catalog") from exc
    if model_id not in catalog_ids:
        raise LMStudioUnavailable(f"LM Studio model is not available: {model_id}")
    return LMStudioIdentity(
        base_url=base_url,
        model_id=model_id,
        model_artifact=str(artifact),
        model_artifact_bytes=artifact.stat().st_size,
        model_artifact_sha256=sha256_file(artifact),
        catalog_model_ids=catalog_ids,
    )


def _json_request(
    url: str, payload: dict[str, Any] | None, timeout_seconds: float
) -> dict[str, Any]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="GET" if payload is None else "POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            raw = response.read(4 * 1024 * 1024 + 1)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise LMStudioUnavailable(f"LM Studio request failed: {exc}") from exc
    if len(raw) > 4 * 1024 * 1024:
        raise LMStudioUnavailable("LM Studio response exceeded 4 MiB")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LMStudioUnavailable("LM Studio returned invalid JSON") from exc
    if not isinstance(parsed, dict):
        raise LMStudioUnavailable("LM Studio response must be a JSON object")
    return parsed
