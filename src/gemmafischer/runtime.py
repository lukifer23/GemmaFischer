from __future__ import annotations

import json
import os
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal, Protocol, cast

from pydantic import TypeAdapter, ValidationError

from .domain import CoachingClaim, EngineEvidence, LineClaim, RatingBucket, canonical_hash
from .resources import bundled_path

DEFAULT_MODEL = "mlx-community/gemma-4-e2b-it-4bit"
DEFAULT_MODEL_REVISION = "238767527555cb75a05732a84dff5d6ba0dd6809"
CLAIM_SELECTION_SYSTEM_PROMPT = "You select grounded chess coaching claims."
CLAIM_SELECTION_CONTRACT_VERSION = "claim-selection-1.0"
LESSON_SELECTION_SYSTEM_PROMPT = (
    "You select a grounded chess lesson using only the supplied IDs."
)
LESSON_SELECTION_CONTRACT_VERSION = "lesson-selection-2.0"


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
    manifest_path: Path | None = None,
) -> dict[str, object]:
    snapshot = resolve_model_snapshot(model_id, revision)
    manifest_path = manifest_path or bundled_path("assets/model-manifest.json")
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


def inspect_adapter_assets(
    adapter_path: Path, expected_sha256: str | None = None
) -> dict[str, object]:
    resolved = adapter_path.expanduser().resolve()
    if not resolved.is_dir():
        raise ModelUnavailable(f"Adapter directory does not exist: {resolved}")
    weights = sorted(resolved.glob("*.safetensors"))
    if len(weights) != 1:
        raise ModelUnavailable(
            "A runtime adapter directory must contain exactly one safetensors file"
        )
    config = resolved / "adapter_config.json"
    if not config.is_file():
        raise ModelUnavailable("The runtime adapter is missing adapter_config.json")
    digest = _sha256_file(weights[0])
    if expected_sha256 and digest != expected_sha256:
        raise ModelUnavailable(
            "The runtime adapter hash does not match GEMMAFISCHER_ADAPTER_SHA256"
        )
    return {
        "status": "verified-local",
        "path": str(resolved),
        "weights": weights[0].name,
        "sha256": digest,
    }


@dataclass(frozen=True)
class ModelClaimSelection:
    claims: tuple[CoachingClaim, ...]
    removed_claim_codes: tuple[str, ...] = ()
    concept_ids: tuple[str, ...] = ()
    question_template_id: Literal[
        "find-strongest-move", "explain-engine-choice", "compare-candidates"
    ] = "find-strongest-move"
    hint_template_id: str = "forcing-moves"


@dataclass(frozen=True)
class LessonSelectionCatalog:
    claims: dict[str, CoachingClaim]
    concept_ids: tuple[str, ...]
    question_template_ids: tuple[str, ...]
    hint_template_ids: tuple[str, ...]


class ClaimSelector(Protocol):
    """Provider-neutral boundary for bounded, evidence-backed model selection."""

    @property
    def source(self) -> Literal["gemma", "lfm"]: ...

    @property
    def model_id(self) -> str: ...

    def select_claims(
        self, evidence: EngineEvidence, rating: RatingBucket
    ) -> ModelClaimSelection: ...


def claim_selection_prompt(evidence: EngineEvidence, rating: RatingBucket) -> str:
    """Return the exact user prompt used by the bounded claim selector."""
    if not evidence.candidates:
        raise ValueError("Model coaching requires at least one engine candidate")
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
    return f"""Select a grounded lesson for a {rating.value} chess player.
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


def lesson_selection_catalog(
    evidence: EngineEvidence, rating: RatingBucket
) -> LessonSelectionCatalog:
    """Build the complete deterministic choice set exposed to a selector model."""
    from .coach import deterministic_coach

    considered = (
        evidence.move_comparison.considered_move_uci if evidence.move_comparison else None
    )
    baseline = deterministic_coach(evidence, rating, considered)
    candidates: list[CoachingClaim] = list(baseline.claims)
    line_length = 4 if rating in {RatingBucket.BEGINNER, RatingBucket.DEVELOPING} else 8
    for candidate in evidence.candidates:
        if len(candidate.pv_uci) > 1:
            candidates.append(
                LineClaim(
                    evidence_ids=(candidate.evidence_id,),
                    candidate_id=candidate.evidence_id,
                    start_ply=0,
                    end_ply=min(line_length, len(candidate.pv_uci)),
                )
            )
    claims: dict[str, CoachingClaim] = {}
    for claim in candidates:
        claim_id = "claim-" + canonical_hash(claim.model_dump(mode="json"))[:16]
        claims.setdefault(claim_id, claim)
    concepts = tuple(
        item.evidence_id
        for item in evidence.concepts
        if bool(item.value) and item.evidence_id not in ()
    )
    questions: tuple[str, ...] = ("find-strongest-move", "explain-engine-choice")
    if len(evidence.candidates) > 1:
        questions += ("compare-candidates",)
    hints = ("forcing-moves", *(f"concept:{item}" for item in concepts))
    return LessonSelectionCatalog(claims, concepts, questions, hints)


def deterministic_lesson_selection(
    evidence: EngineEvidence, rating: RatingBucket
) -> ModelClaimSelection:
    """Return the contract-valid baseline target used for fallback and data generation."""
    from .coach import deterministic_coach

    catalog = lesson_selection_catalog(evidence, rating)
    considered = (
        evidence.move_comparison.considered_move_uci if evidence.move_comparison else None
    )
    baseline = deterministic_coach(evidence, rating, considered)
    selected = tuple(
        claim_id
        for claim_id, claim in catalog.claims.items()
        if claim in baseline.claims
    )
    concepts = tuple(
        item.evidence_id
        for item in evidence.concepts
        if item.candidate_id == evidence.candidates[0].evidence_id and bool(item.value)
    )[:4]
    hint = f"concept:{concepts[0]}" if concepts else "forcing-moves"
    return ModelClaimSelection(
        claims=tuple(catalog.claims[item] for item in selected[:5]),
        concept_ids=concepts,
        question_template_id="find-strongest-move",
        hint_template_id=hint,
    )


def lesson_selection_target(
    evidence: EngineEvidence, rating: RatingBucket
) -> dict[str, object]:
    catalog = lesson_selection_catalog(evidence, rating)
    selection = deterministic_lesson_selection(evidence, rating)
    reverse = {claim.model_dump_json(): claim_id for claim_id, claim in catalog.claims.items()}
    return {
        "claim_ids": [reverse[item.model_dump_json()] for item in selection.claims],
        "concept_ids": list(selection.concept_ids),
        "question_template_id": selection.question_template_id,
        "hint_template_id": selection.hint_template_id,
    }


def lesson_selection_prompt(evidence: EngineEvidence, rating: RatingBucket) -> str:
    catalog = lesson_selection_catalog(evidence, rating)
    claim_rows = [
        {"claim_id": claim_id, "claim": claim.model_dump(mode="json")}
        for claim_id, claim in catalog.claims.items()
    ]
    payload = {
        "rating_bucket": rating.value,
        "claims": claim_rows,
        "concept_ids": list(catalog.concept_ids),
        "question_template_ids": list(catalog.question_template_ids),
        "hint_template_ids": list(catalog.hint_template_ids),
    }
    return (
        "Select the most useful grounded lesson for this learner. Return exactly one JSON "
        "object with claim_ids (2 to 5 unique IDs), concept_ids (up to 4 unique IDs), "
        "question_template_id, and hint_template_id. Use only supplied IDs.\n"
        + json.dumps(payload, separators=(",", ":"))
    )


def parse_lesson_selection(
    output: str, evidence: EngineEvidence, rating: RatingBucket
) -> ModelClaimSelection:
    """Strictly parse v2 selector output; partial or invented selections never survive."""
    start = output.find("{")
    end = output.rfind("}")
    if start < 0 or end < start:
        raise ValueError("Model response did not contain a JSON object")
    try:
        raw = json.loads(output[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError(f"Model returned malformed JSON: {exc}") from exc
    if not isinstance(raw, dict) or set(raw) != {
        "claim_ids", "concept_ids", "question_template_id", "hint_template_id"
    }:
        raise ValueError("Model lesson selection has an invalid object shape")
    claim_ids = raw["claim_ids"]
    concept_ids = raw["concept_ids"]
    if not isinstance(claim_ids, list) or not 2 <= len(claim_ids) <= 5:
        raise ValueError("Model lesson selection requires 2 to 5 claim IDs")
    if not isinstance(concept_ids, list) or len(concept_ids) > 4:
        raise ValueError("Model lesson selection accepts up to 4 concept IDs")
    values = (*claim_ids, *concept_ids, raw["question_template_id"], raw["hint_template_id"])
    if not all(isinstance(item, str) for item in values):
        raise ValueError("Model lesson selection IDs must be strings")
    if len(set(claim_ids)) != len(claim_ids) or len(set(concept_ids)) != len(concept_ids):
        raise ValueError("Model lesson selection IDs must be unique")
    catalog = lesson_selection_catalog(evidence, rating)
    if any(item not in catalog.claims for item in claim_ids):
        raise ValueError("Model lesson selection contains an unknown claim ID")
    if any(item not in catalog.concept_ids for item in concept_ids):
        raise ValueError("Model lesson selection contains an unknown concept ID")
    question = str(raw["question_template_id"])
    hint = str(raw["hint_template_id"])
    if question not in catalog.question_template_ids:
        raise ValueError("Model lesson selection contains an unknown question template")
    if hint not in catalog.hint_template_ids:
        raise ValueError("Model lesson selection contains an unknown hint template")
    return ModelClaimSelection(
        claims=tuple(catalog.claims[item] for item in claim_ids),
        concept_ids=tuple(str(item) for item in concept_ids),
        question_template_id=cast(
            Literal["find-strongest-move", "explain-engine-choice", "compare-candidates"],
            question,
        ),
        hint_template_id=hint,
    )


def extract_json_array(output: str) -> Any:
    start = output.find("[")
    end = output.rfind("]")
    if start < 0 or end < start:
        raise ValueError("Model response did not contain a JSON array")
    try:
        return json.loads(output[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError(f"Model returned malformed JSON: {exc}") from exc


def parse_claim_selection(output: str, evidence: EngineEvidence) -> ModelClaimSelection:
    """Parse and constrain one provider response against known engine evidence."""

    try:
        raw = extract_json_array(output)
    except ValueError as exc:
        raise ValueError(f"Model returned invalid coaching claims: {exc}") from exc
    if not isinstance(raw, list):
        raise ValueError("Model coaching payload must be a JSON array")
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


class GemmaRuntime:
    """Real optional MLX-LM runtime. Import and assets are required only for the full profile."""

    source: Literal["gemma"] = "gemma"

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        revision: str | None = DEFAULT_MODEL_REVISION,
        manifest_path: Path | None = None,
        adapter_path: Path | None = None,
        adapter_sha256: str | None = None,
    ) -> None:
        try:
            from mlx_lm import generate, load
        except ImportError as exc:
            raise ModelUnavailable("Install the full profile with: uv sync --extra full") from exc
        self._generate = generate
        verified = inspect_model_assets(model_id, revision, manifest_path)
        snapshot = Path(str(verified["snapshot"]))
        configured_adapter = adapter_path
        if configured_adapter is None and os.environ.get("GEMMAFISCHER_ADAPTER_PATH"):
            configured_adapter = Path(os.environ["GEMMAFISCHER_ADAPTER_PATH"])
        expected_adapter_hash = adapter_sha256 or os.environ.get(
            "GEMMAFISCHER_ADAPTER_SHA256"
        )
        self.adapter: dict[str, object] | None
        try:
            if configured_adapter is not None:
                adapter = inspect_adapter_assets(configured_adapter, expected_adapter_hash)
                loaded = load(str(snapshot), adapter_path=str(adapter["path"]))
                self.adapter = adapter
            else:
                loaded = load(str(snapshot))
                self.adapter = None
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
        prompt = lesson_selection_prompt(evidence, rating)
        messages = [
            {"role": "system", "content": LESSON_SELECTION_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        formatted = self._tokenizer.apply_chat_template(
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
        return parse_lesson_selection(output, evidence, rating)
