from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def sanitize_gemma4_shared_kv_weights(
    weights: Mapping[str, Any], *, num_hidden_layers: int, num_kv_shared_layers: int
) -> dict[str, Any]:
    """Drop only checkpoint tensors unused by Gemma 4 KV-sharing layers."""
    first_shared_layer = num_hidden_layers - num_kv_shared_layers
    if num_hidden_layers < 1 or not 0 <= num_kv_shared_layers < num_hidden_layers:
        raise ValueError("Gemma 4 shared-KV layer configuration is invalid")
    unused_suffixes = (
        ".self_attn.k_norm.weight",
        ".self_attn.k_proj.weight",
        ".self_attn.v_proj.weight",
    )
    sanitized: dict[str, Any] = {}
    for name, value in weights.items():
        unused = any(
            name.startswith(f"language_model.model.layers.{layer}.")
            and name.endswith(unused_suffixes)
            for layer in range(first_shared_layer, num_hidden_layers)
        )
        if not unused:
            sanitized[name] = value
    return sanitized


def main() -> None:
    """Run MLX-LM LoRA with strict Gemma 4 shared-KV checkpoint sanitation."""
    from mlx_lm import lora
    from mlx_lm.models.gemma4_text import Model

    original_sanitize: Any = Model.sanitize

    def sanitize(self: Any, weights: Mapping[str, Any]) -> dict[str, Any]:
        upstream = original_sanitize(self, weights)
        return sanitize_gemma4_shared_kv_weights(
            upstream,
            num_hidden_layers=int(self.args.num_hidden_layers),
            num_kv_shared_layers=int(self.args.num_kv_shared_layers),
        )

    model_class: Any = Model
    model_class.sanitize = sanitize
    lora_main: Any = lora.main
    lora_main()


if __name__ == "__main__":
    main()
