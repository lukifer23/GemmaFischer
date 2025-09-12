#!/usr/bin/env python3
"""Curate and improve chess training datasets for better model performance.

This script performs advanced data curation including:
- Difficulty filtering (remove too easy/hard positions)
- Quality scoring based on move uniqueness
- Duplicate removal
- Length normalization
- Stockfish validation and enhancement
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Any, List, Optional, Set
from collections import defaultdict
import hashlib

import chess


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True, help='Input JSONL file')
    ap.add_argument('--output', required=True, help='Output curated JSONL file')
    ap.add_argument('--mode', choices=['uci', 'tutor', 'director'], default='uci')
    ap.add_argument('--min_rating', type=int, default=1200, help='Minimum puzzle rating')
    ap.add_argument('--max_rating', type=int, default=2500, help='Maximum puzzle rating')
    ap.add_argument('--max_samples', type=int, default=0, help='Limit output samples (0 = no limit)')
    ap.add_argument('--remove_duplicates', action='store_true', help='Remove duplicate positions')
    ap.add_argument('--balance_difficulty', action='store_true', help='Balance difficulty distribution')
    return ap.parse_args()


def extract_fen(text: str) -> Optional[str]:
    """Extract FEN from text."""
    import re
    # Look for FEN pattern
    fen_pattern = r'([rnbqkpRNBQKP1-8]+/){7}[rnbqkpRNBQKP1-8]+\s[wb]\s(?:K?Q?k?q?|-)\s(?:[a-h][36]|-)\s\d+\s\d+'
    m = re.search(fen_pattern, text)
    return m.group(0) if m else None


def calculate_position_complexity(fen: str) -> float:
    """Calculate position complexity score (0-1)."""
    try:
        board = chess.Board(fen)

        # Factors contributing to complexity
        factors = []

        # 1. Piece count (more pieces = more complex)
        piece_count = len([p for p in board.board_fen() if p not in '12345678/'])
        factors.append(min(piece_count / 32, 1.0))  # Normalize to 32 max

        # 2. King safety (castling rights, open king)
        king_safety = 0
        if board.has_kingside_castling_rights(chess.WHITE) or board.has_queenside_castling_rights(chess.WHITE):
            king_safety += 0.3
        if board.has_kingside_castling_rights(chess.BLACK) or board.has_queenside_castling_rights(chess.BLACK):
            king_safety += 0.3
        if board.is_check():
            king_safety += 0.4
        factors.append(king_safety)

        # 3. Tactical motifs (simplified detection)
        tactical_score = 0
        # Check for pins, discovered attacks, etc.
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.piece_type in [chess.QUEEN, chess.ROOK, chess.BISHOP]:
                # Count attacked squares
                attacks = len([s for s in board.attacks(square) if board.piece_at(s)])
                tactical_score += min(attacks / 8, 0.5)
        factors.append(min(tactical_score, 1.0))

        return sum(factors) / len(factors)

    except Exception:
        return 0.5


def score_sample_quality(obj: Dict[str, Any], mode: str) -> float:
    """Score sample quality (0-1, higher is better)."""
    try:
        # Base quality factors
        quality_factors = []

        # 1. Response length appropriateness
        response = obj.get('response', '')
        if mode == 'uci':
            # UCI moves should be short
            quality_factors.append(1.0 if 3 <= len(response) <= 5 else 0.5)
        elif mode == 'tutor':
            # Tutor responses should be informative but not too long
            word_count = len(response.split())
            quality_factors.append(1.0 if 10 <= word_count <= 100 else 0.3)
        else:
            quality_factors.append(0.8)  # Director mode

        # 2. Contains FEN (important for context)
        prompt = obj.get('prompt', '')
        has_fen = 'fen' in prompt.lower() or extract_fen(prompt) is not None
        quality_factors.append(1.0 if has_fen else 0.3)

        # 3. Position complexity (prefer neither too simple nor too complex)
        fen = extract_fen(prompt) or extract_fen(response)
        if fen:
            complexity = calculate_position_complexity(fen)
            # Prefer medium complexity (0.3-0.7)
            if 0.3 <= complexity <= 0.7:
                quality_factors.append(1.0)
            elif complexity < 0.3 or complexity > 0.7:
                quality_factors.append(0.7)
            else:
                quality_factors.append(0.5)

        # 4. Rating appropriateness
        meta = obj.get('meta', {})
        rating = meta.get('rating', 1500)
        if 1200 <= rating <= 2200:
            quality_factors.append(1.0)
        elif rating < 1000 or rating > 2500:
            quality_factors.append(0.5)
        else:
            quality_factors.append(0.8)

        return sum(quality_factors) / len(quality_factors)

    except Exception:
        return 0.5


def remove_duplicates(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove duplicate samples based on FEN."""
    seen_fens: Set[str] = set()
    unique_samples = []

    for sample in samples:
        fen = None
        for field in ['prompt', 'response']:
            text = sample.get(field, '')
            fen = extract_fen(text)
            if fen:
                break

        if fen and fen not in seen_fens:
            seen_fens.add(fen)
            unique_samples.append(sample)
        elif not fen:
            # Keep samples without FEN
            unique_samples.append(sample)

    return unique_samples


def balance_difficulty_distribution(samples: List[Dict[str, Any]], target_bins: int = 5) -> List[Dict[str, Any]]:
    """Balance samples across difficulty levels."""
    # Group by rating ranges
    bins = defaultdict(list)
    for sample in samples:
        rating = sample.get('meta', {}).get('rating', 1500)
        bin_idx = min(target_bins - 1, rating // 400)  # 400-point bins
        bins[bin_idx].append(sample)

    # Find smallest bin size
    min_size = min(len(bins[b]) for b in bins.keys()) if bins else 0

    # Sample equally from each bin
    balanced = []
    for bin_samples in bins.values():
        balanced.extend(random.sample(bin_samples, min_size))

    random.shuffle(balanced)
    return balanced


def main():
    args = parse_args()

    # Read all samples
    samples = []
    with open(args.input, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                obj = json.loads(line.strip())
                samples.append(obj)
            except json.JSONDecodeError:
                continue

    print(f"Loaded {len(samples)} samples from {args.input}")

    # Apply filters
    filtered_samples = []

    for sample in samples:
        # Rating filter
        rating = sample.get('meta', {}).get('rating', 1500)
        if not (args.min_rating <= rating <= args.max_rating):
            continue

        # Quality filter
        quality = score_sample_quality(sample, args.mode)
        if quality < 0.6:  # Only keep reasonably good samples
            continue

        filtered_samples.append(sample)

    print(f"After filtering: {len(filtered_samples)} samples")

    # Remove duplicates if requested
    if args.remove_duplicates:
        filtered_samples = remove_duplicates(filtered_samples)
        print(f"After deduplication: {len(filtered_samples)} samples")

    # Balance difficulty if requested
    if args.balance_difficulty:
        filtered_samples = balance_difficulty_distribution(filtered_samples)
        print(f"After difficulty balancing: {len(filtered_samples)} samples")

    # Limit samples if requested
    if args.max_samples > 0 and len(filtered_samples) > args.max_samples:
        filtered_samples = random.sample(filtered_samples, args.max_samples)
        print(f"Limited to {len(filtered_samples)} samples")

    # Write output
    with open(args.output, 'w', encoding='utf-8') as f:
        for sample in filtered_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    print(f"Wrote {len(filtered_samples)} curated samples to {args.output}")


if __name__ == '__main__':
    main()
