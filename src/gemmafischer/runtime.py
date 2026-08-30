from __future__ import annotations

import json
import os
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter, ValidationError

from .domain import CoachingClaim, EngineEvidence, RatingBucket

DEFAULT_MODEL = "mlx-community/gemma-4-e2b-it-4bit"
DEFAULT_MODEL_REVISION = "238767527555cb75a05732a84dff5d6ba0dd6809"


class ModelUnavailable(RuntimeError):
    pass


def resolve_model_snapshot(
    model_id: str = DEFAULT_MODEL,
    revision: str | None = DEFAULT_MODEL_REVISION,
) -> Path:
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        from huggingface_hub import constants, snapshot_download
        from huggingface_hub.file_download import repo_folder_name

        try:
            snapshot = Path(
                snapshot_download(repo_id=model_id, revision=revision, local_files_only=True)
            ).resolve()
        except OSError:
            snapshot = (
                Path(constants.HF_HUB_CACHE)
                / repo_folder_name(repo_id=model_id, repo_type="model")
                / "snapshots"
                / str(revision)
            ).resolve()
    except (ImportError, OSError) as exc:
        raise ModelUnavailable(
            f"Pinned model assets are not available locally for {model_id}@{revision}"
        ) from exc
    required = ("config.json", "tokenizer.json")
    missing = [name for name in required if not (snapshot / name).is_file()]
    weights = tuple(snapshot.glob("*.safetensors"))
    if missing or not weights:
        missing_assets = [*missing]
        if not weights:
            missing_assets.append("*.safetensors")
        detail = ", ".join(missing_assets)
        raise ModelUnavailable(f"Pinned model snapshot is incomplete: {detail}")
    return snapshot


def inspect_model_assets(
    model_id: str = DEFAULT_MODEL,
    revision: str | None = DEFAULT_MODEL_REVISION,
) -> dict[str, object]:
    snapshot = resolve_model_snapshot(model_id, revision)
    manifest_path = Path(__file__).parents[2] / "assets" / "model-manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise ModelUnavailable("The pinned model asset manifest is missing or invalid") from exc
    if manifest.get("model_id") != model_id or manifest.get("revision") != revision:
        raise ModelUnavailable("The model request does not match the pinned asset manifest")
    expected: dict[str, str] = dict(manifest.get("file_hashes", {}))
    expected["tokenizer.json"] = str(manifest.get("tokenizer_hash", ""))
    expected["chat_template.jinja"] = str(manifest.get("chat_template_hash", ""))
    mismatches: list[str] = []
    for name, expected_hash in expected.items():
        path = snapshot / name
        actual_hash = _sha256_file(path) if path.is_file() else "missing"
        if not expected_hash or actual_hash != expected_hash:
            mismatches.append(name)
    if mismatches:
        raise ModelUnavailable(
            "Pinned model asset hash mismatch: " + ", ".join(sorted(mismatches))
        )
    files = sorted(path for path in snapshot.rglob("*") if path.is_file())
    manifest_rows = [
        f"{path.relative_to(snapshot)}:{path.stat().st_size}:{path.resolve().name}"
        for path in files
    ]
    return {
        "status": "verified-local",
        "model_id": model_id,
        "revision": revision,
        "snapshot": str(snapshot),
        "file_count": len(files),
        "bytes": sum(path.stat().st_size for path in files),
        "manifest_sha256": sha256("\n".join(manifest_rows).encode()).hexdigest(),
    }


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ModelClaimSelection:
    claims: tuple[CoachingClaim, ...]
    removed_claim_codes: tuple[str, ...] = ()
    concept_ids: tuple[str, ...] = ()


def extract_json_array(output: str) -> Any:
    start = output.find("[")
    end = output.rfind("]")
    if start < 0 or end < start:
        raise ValueError("Gemma response did not contain a JSON array")
    try:
        return json.loads(output[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError(f"Gemma returned malformed JSON: {exc}") from exc


class GemmaRuntime:
    """Real optional MLX-LM runtime. Import and assets are required only for the full profile."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        revision: str | None = DEFAULT_MODEL_REVISION,
    ) -> None:
        try:
            from mlx_lm import generate, load
        except ImportError as exc:
            raise ModelUnavailable("Install the full profile with: uv sync --extra full") from exc
        self._generate = generate
        verified = inspect_model_assets(model_id, revision)
        snapshot = Path(str(verified["snapshot"]))
        try:
            loaded = load(str(snapshot))
        except (OSError, RuntimeError, ValueError) as exc:
            raise ModelUnavailable(f"Pinned model assets could not be loaded: {exc}") from exc
        self._model, self._tokenizer = loaded[0], loaded[1]
        self.model_id = model_id
        self.revision = revision

    def claims(self, evidence: EngineEvidence, rating: RatingBucket) -> tuple[CoachingClaim, ...]:
        selection = self.select_claims(evidence, rating)
        if not 2 <= len(selection.claims) <= 5:
            raise ValueError("Gemma must return 2 to 5 valid coaching claims")
        return selection.claims

    def select_claims(
        self, evidence: EngineEvidence, rating: RatingBucket
    ) -> ModelClaimSelection:
        if not evidence.candidates:
            raise ValueError("Gemma coaching requires at least one engine candidate")
        best = evidence.candidates[0]
        candidates = [
            {
                "candidate_id": item.evidence_id,
                "move_san": item.move_san,
                "score_cp": item.score_cp,
                "mate_in": item.mate_in,
                "pv_length": len(item.pv_uci),
                "concepts": [
                    {"concept_id": concept.evidence_id, "concept": concept.concept}
                    for concept in evidence.concepts
                    if concept.candidate_id == item.evidence_id
                ],
            }
            for item in evidence.candidates
        ]
        best_id = best.evidence_id
        prompt = f"""Select a grounded lesson for a {rating.value} chess player.
Return only one JSON array with 2 to 5 claim objects and up to 4 optional concept objects.
Do not return evidence rows or prose.
Candidate data: {json.dumps(candidates, separators=(",", ":"))}

Allowed object shapes, using only candidate_id values listed above:
{{"kind":"move","evidence_ids":["ID"],"candidate_id":"ID"}}
{{"kind":"score","evidence_ids":["ID"],"candidate_id":"ID"}}
{{"kind":"guidance","evidence_ids":[],"template_id":"calculate_forcing_moves"}}
Optional lesson ordering objects, using only concept_id values listed above:
{{"kind":"concept","concept_id":"ID"}}

Use this valid structure as your starting point, then select the most useful 2 to 5 claims:
[
{{"kind":"move","evidence_ids":["{best_id}"],"candidate_id":"{best_id}"}},
{{"kind":"score","evidence_ids":["{best_id}"],"candidate_id":"{best_id}"}},
{{"kind":"guidance","evidence_ids":[],"template_id":"calculate_forcing_moves"}}
]"""
        messages = [
            {"role": "system", "content": "You select grounded chess coaching claims."},
            {"role": "user", "content": prompt},
        ]
        formatted = self._tokenizer.apply_chat_template(  # type: ignore[no-untyped-call]
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        output = self._generate(
            self._model,
            self._tokenizer,
            prompt=formatted,
            max_tokens=768,
            verbose=False,
        )
        try:
            raw = extract_json_array(output)
        except ValueError as exc:
            raise ValueError(f"Gemma returned invalid coaching claims: {exc}") from exc
        if not isinstance(raw, list):
            raise ValueError("Gemma coaching payload must be a JSON array")
        adapter: TypeAdapter[CoachingClaim] = TypeAdapter(CoachingClaim)
        claims: list[CoachingClaim] = []
        concept_ids: list[str] = []
        known_concepts = {item.evidence_id for item in evidence.concepts}
        removed: list[str] = []
        for item in raw[:9]:
            if isinstance(item, dict) and item.get("kind") == "concept":
                concept_id = item.get("concept_id")
                if isinstance(concept_id, str) and concept_id in known_concepts:
                    if concept_id not in concept_ids:
                        concept_ids.append(concept_id)
                else:
                    removed.append("MODEL_CONCEPT_ID_INVALID")
                continue
            try:
                claims.append(adapter.validate_python(item))
            except ValidationError:
                removed.append("MODEL_CLAIM_SCHEMA_INVALID")
        return ModelClaimSelection(tuple(claims[:5]), tuple(removed), tuple(concept_ids[:4]))
