#!/usr/bin/env python3
"""Utilities to build mixed training datasets from multiple JSONL sources.

Each source is a JSONL file with at least a `text` field (as produced by our
refinement and ingestion scripts). This module constructs a weighted mixture
using Hugging Face `datasets.interleave_datasets`.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any, Optional

from datasets import load_dataset, Dataset, interleave_datasets


def _load_single_jsonl(path: str) -> Dataset:
    ds = load_dataset('json', data_files=path, split='train')
    # Ensure expected field
    if 'text' not in ds.column_names:
        raise ValueError(f"Dataset {path} must contain a 'text' field.")
    return ds


def build_mixture(dataset_specs: List[Dict[str, Any]], seed: int = 3407) -> Dataset:
    """Build an interleaved mixture from dataset specs.

    dataset_specs: list of { 'path': str, 'weight': float }
    Weights are normalized automatically.
    """
    if not dataset_specs:
        raise ValueError('No dataset specs provided for mixture.')

    datasets_list: List[Dataset] = []
    weights: List[float] = []

    for spec in dataset_specs:
        path = spec.get('path')
        weight = float(spec.get('weight', 1.0))
        if not path:
            raise ValueError('Each dataset spec must include a path.')
        if weight <= 0:
            # Skip zero/negative weights
            continue
        ds = _load_single_jsonl(path)
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


