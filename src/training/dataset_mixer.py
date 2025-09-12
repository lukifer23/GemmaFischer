#!/usr/bin/env python3
"""Utilities to build mixed training datasets from multiple JSONL sources.

Each source can follow either the legacy schema with a single ``text`` field or
the newer instruction tuning schema ``{task, prompt, response, meta}``. This
module loads and normalizes each dataset so downstream code can consume a
uniform structure before constructing a weighted mixture using
``datasets.interleave_datasets``.
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional

from datasets import (
    load_dataset,
    Dataset,
    IterableDataset,
    interleave_datasets,
)


def _load_single_jsonl(
    path: str,
    *,
    streaming: bool = False,
    cache_dir: Optional[str] = None,
) -> Dataset | IterableDataset:
    """Load a JSONL dataset and normalize its columns.

    Supports two layouts:
      1. Legacy ``{"text": ...}``
      2. Instruction schema ``{"task", "prompt", "response", "meta"}``

    The returned dataset always contains the columns ``text``, ``prompt``,
    ``response``, ``task`` and ``meta`` (missing fields are filled with ``None``).
    When ``text`` is absent but ``prompt``/``response`` are present, a ``text``
    field is synthesized by concatenating them.
    """

    ds = load_dataset(
        "json",
        data_files=path,
        split="train",
        streaming=streaming,
        cache_dir=cache_dir,
    )

    expected_cols = ["text", "prompt", "response", "task", "meta"]

    def _normalize(example: Dict[str, Any]) -> Dict[str, Any]:
        prompt = example.get("prompt")
        response = example.get("response")
        text = example.get("text")
        if text is None and prompt is not None and response is not None:
            text = f"{prompt}{response}"
        return {
            "text": text,
            "prompt": prompt,
            "response": response,
            "task": example.get("task"),
            "meta": example.get("meta"),
        }

    # Map lazily and drop any unexpected columns
    extra_cols = [c for c in ds.column_names if c not in expected_cols]
    ds = ds.map(_normalize, remove_columns=extra_cols)
    return ds


def build_mixture(
    dataset_specs: List[Dict[str, Any]],
    seed: int = 3407,
    *,
    streaming: bool = False,
    cache_dir: Optional[str] = None,
) -> Dataset | IterableDataset:
    """Build an interleaved mixture from dataset specs.

    dataset_specs: list of { 'path': str, 'weight': float }
    Weights are normalized automatically.
    """
    if not dataset_specs:
        raise ValueError('No dataset specs provided for mixture.')

    datasets_list: List[Dataset | IterableDataset] = []
    weights: List[float] = []

    for spec in dataset_specs:
        path = spec.get('path')
        weight = float(spec.get('weight', 1.0))
        if not path:
            raise ValueError('Each dataset spec must include a path.')
        if weight <= 0:
            # Skip zero/negative weights
            continue
        ds = _load_single_jsonl(path, streaming=streaming, cache_dir=cache_dir)
        datasets_list.append(ds)
        weights.append(weight)

    if not datasets_list:
        raise ValueError('No valid datasets after parsing specs.')

    # Normalize weights
    total_w = sum(weights)
    probs = [w / total_w for w in weights]

    mixed = interleave_datasets(datasets_list, probabilities=probs, seed=seed)
    return mixed


def train_eval_split(ds: Dataset, eval_ratio: float = 0.1, seed: int = 3407):
    """Create a small evaluation split from the mixed dataset."""
    return ds.train_test_split(test_size=eval_ratio, seed=seed)


