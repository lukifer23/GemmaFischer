from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from pydantic import TypeAdapter, ValidationError

from .domain import CoachingClaim, EngineEvidence, RatingBucket

DEFAULT_MODEL = "mlx-community/gemma-4-e2b-it-4bit"
DEFAULT_MODEL_REVISION = "238767527555cb75a05732a84dff5d6ba0dd6809"


class ModelUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class ModelClaimSelection:
    claims: tuple[CoachingClaim, ...]
    removed_claim_codes: tuple[str, ...] = ()


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
        loaded = load(model_id, revision=revision)
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
            }
            for item in evidence.candidates
        ]
        best_id = best.evidence_id
        prompt = f"""Select a grounded lesson for a {rating.value} chess player.
Return only one JSON array with 2 to 5 objects. Do not return evidence rows or prose.
Candidate data: {json.dumps(candidates, separators=(",", ":"))}

Allowed object shapes, using only candidate_id values listed above:
{{"kind":"move","evidence_ids":["ID"],"candidate_id":"ID"}}
{{"kind":"score","evidence_ids":["ID"],"candidate_id":"ID"}}
{{"kind":"guidance","evidence_ids":[],"template_id":"calculate_forcing_moves"}}

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
        removed: list[str] = []
        for item in raw[:5]:
            try:
                claims.append(adapter.validate_python(item))
            except ValidationError:
                removed.append("MODEL_CLAIM_SCHEMA_INVALID")
        return ModelClaimSelection(tuple(claims), tuple(removed))
