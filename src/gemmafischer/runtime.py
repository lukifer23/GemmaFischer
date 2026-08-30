from __future__ import annotations

import json
from typing import Any

from pydantic import TypeAdapter, ValidationError

from .domain import CoachingClaim, EngineEvidence, RatingBucket

DEFAULT_MODEL = "mlx-community/gemma-4-e2b-it-4bit"


class ModelUnavailable(RuntimeError):
    pass


class GemmaRuntime:
    """Real optional MLX-LM runtime. Import and assets are required only for the full profile."""

    def __init__(self, model_id: str = DEFAULT_MODEL) -> None:
        try:
            from mlx_lm import generate, load
        except ImportError as exc:
            raise ModelUnavailable("Install the full profile with: uv sync --extra full") from exc
        self._generate = generate
        loaded = load(model_id)
        self._model, self._tokenizer = loaded[0], loaded[1]
        self.model_id = model_id

    def claims(self, evidence: EngineEvidence, rating: RatingBucket) -> tuple[CoachingClaim, ...]:
        prompt = (
            "Return only a JSON array of 2 to 5 coaching claim objects. Use only the supplied "
            "evidence IDs and values. Do not add chess facts. Schema kinds are move, line, score, "
            "comparison, and guidance. Evidence:\n"
            + evidence.model_dump_json()
            + f"\nRating bucket: {rating.value}"
        )
        messages = [
            {"role": "system", "content": "You select grounded chess coaching claims."},
            {"role": "user", "content": prompt},
        ]
        formatted = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        output = self._generate(
            self._model,
            self._tokenizer,
            prompt=formatted,
            max_tokens=768,
            verbose=False,
        )
        try:
            raw: Any = json.loads(output)
            claims = TypeAdapter(list[CoachingClaim]).validate_python(raw)
        except (json.JSONDecodeError, ValidationError) as exc:
            raise ValueError(f"Gemma returned invalid coaching claims: {exc}") from exc
        if not 2 <= len(claims) <= 5:
            raise ValueError("Gemma must return 2 to 5 coaching claims")
        return tuple(claims)
