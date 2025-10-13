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
    drop_tasks: Optional[List[str]] = None,
    keep_tasks: Optional[List[str]] = None,
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
        task = example.get("task", "")

        # Strengthen chess context in prompts
        if prompt and not prompt.startswith("You are") and not prompt.startswith("FEN:"):
            if "tutor" in task:
                chess_prefix = "You are a chess tutor. "
            elif "director" in task:
                chess_prefix = "You are a chess grandmaster. "
            elif "engine" in task:
                chess_prefix = "You are a chess engine. "
            else:
                chess_prefix = "This is about chess. "

            prompt = chess_prefix + prompt

        text = example.get("text")
        if text is None and prompt is not None and response is not None:
            text = f"{prompt}{response}"

        return {
            "text": text,
            "prompt": prompt,
            "response": response,
            "task": task,
            "meta": example.get("meta"),
        }

    # Map lazily and drop any unexpected columns
    extra_cols = [c for c in ds.column_names if c not in expected_cols]
    ds = ds.map(_normalize, remove_columns=extra_cols)

    if (drop_tasks or keep_tasks) and streaming:
        raise ValueError("Task filtering is not supported in streaming mode.")

    if drop_tasks or keep_tasks:
        drop_set = {t.lower() for t in drop_tasks or []}
        keep_set = {t.lower() for t in keep_tasks or []} or None

        def _task_filter(example: Dict[str, Any]) -> bool:
            task_name = (example.get("task") or "").lower()
            if keep_set is not None and task_name not in keep_set:
                return False
            if drop_set and (task_name in drop_set or any(task_name.startswith(prefix) for prefix in drop_set)):
                return False
            return True

        ds = ds.filter(_task_filter)
    return ds


def build_mixture(
    dataset_specs: List[Dict[str, Any]],
    seed: int = 3407,
    *,
    streaming: bool = False,
    cache_dir: Optional[str] = None,
) -> Dataset | IterableDataset:
    """Build an interleaved mixture from dataset specs.

    ``dataset_specs`` accepts dictionaries with the following keys::

        {
            'path': str,
            'weight': float,
            'drop_tasks': List[str],      # optional task blacklist
            'keep_tasks': List[str],      # optional task whitelist
            'drop_engine_uci': bool,      # convenience flag
        }

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
        drop_tasks = spec.get('drop_tasks')
        keep_tasks = spec.get('keep_tasks')
        if spec.get('drop_engine_uci'):
            drop_tasks = list(drop_tasks or []) + ['engine_uci']

        ds = _load_single_jsonl(
            path,
            streaming=streaming,
            cache_dir=cache_dir,
            drop_tasks=drop_tasks,
            keep_tasks=keep_tasks,
        )
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


