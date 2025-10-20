#!/usr/bin/env python3
"""
Dataset management for ChessGemma training.
"""

from typing import List, Dict, Any


class ChessDataset:
    """Dataset class for chess training data."""

    def __init__(self, data: List[Dict[str, Any]]):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.data[idx]
