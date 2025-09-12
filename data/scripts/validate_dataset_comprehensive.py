#!/usr/bin/env python3
"""Comprehensive dataset validation pipeline for ChessGemma training data.

Performs extensive validation including:
- Move legality and correctness verification
- Position quality assessment
- Data consistency checks
- Quality metrics and scoring
- Detailed validation reports
- Statistical analysis
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set
from collections import defaultdict, Counter
import time
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

import chess


@dataclass
class ValidationResult:
    """Result of validating a single sample."""
    sample_id: int
    is_valid: bool = False
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    quality_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetValidationReport:
    """Comprehensive validation report for an entire dataset."""
    total_samples: int = 0
    valid_samples: int = 0
    invalid_samples: int = 0
    quality_distribution: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    error_summary: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    warning_summary: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    position_stats: Dict[str, Any] = field(default_factory=dict)
    move_stats: Dict[str, Any] = field(default_factory=dict)
    quality_metrics: Dict[str, Any] = field(default_factory=dict)
    processing_time: float = 0.0
    validation_errors: List[str] = field(default_factory=list)


class ChessDatasetValidator:
    """Comprehensive validator for chess training datasets."""

    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.stockfish_available = self._check_stockfish_available()

    def _check_stockfish_available(self) -> bool:
        """Check if Stockfish is available for validation."""
        try:
            import chess.engine
            return True
        except ImportError:
            return False

    def validate_dataset(self, input_file: Path, mode: str) -> DatasetValidationReport:
        """Validate an entire dataset and return comprehensive report."""
        start_time = time.time()

        # Load samples
        samples = self._load_samples(input_file)
        report = DatasetValidationReport(total_samples=len(samples))

        if not samples:
            report.validation_errors.append("No samples found in input file")
            report.processing_time = time.time() - start_time
            return report

        # Validate samples (parallel processing)
        validation_results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self._validate_sample, i, sample, mode)
                      for i, sample in enumerate(samples)]
            for future in as_completed(futures):
                validation_results.append(future.result())

        # Sort by sample_id for consistent reporting
        validation_results.sort(key=lambda x: x.sample_id)

        # Process results
        self._process_validation_results(validation_results, samples, report)

        report.processing_time = time.time() - start_time
        return report

    def _load_samples(self, input_file: Path) -> List[Dict[str, Any]]:
        """Load samples from JSONL file."""
        samples = []
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        sample = json.loads(line.strip())
                        samples.append(sample)
                    except json.JSONDecodeError as e:
                        print(f"Warning: Invalid JSON on line {line_num}: {e}")
                        continue
        except Exception as e:
            print(f"Error loading file {input_file}: {e}")
            return []

        return samples

    def _validate_sample(self, sample_id: int, sample: Dict[str, Any], mode: str) -> ValidationResult:
        """Validate a single sample comprehensively."""
        result = ValidationResult(sample_id=sample_id)

        try:
            # Basic structure validation
            if not self._validate_basic_structure(sample, result):
                return result

            # Mode-specific validation
            if mode in ['uci', 'tutor']:
                self._validate_chess_sample(sample, mode, result)
            elif mode == 'director':
                self._validate_director_sample(sample, result)

            # Quality assessment
            result.quality_score = self._assess_sample_quality(sample, mode, result)

            # Determine if sample is valid
            result.is_valid = len(result.errors) == 0

        except Exception as e:
            result.errors.append(f"Validation error: {str(e)}")
            result.is_valid = False

        return result

    def _validate_basic_structure(self, sample: Dict[str, Any], result: ValidationResult) -> bool:
        """Validate basic sample structure."""
        required_fields = ['task', 'prompt', 'response']
        missing_fields = [field for field in required_fields if field not in sample]

        if missing_fields:
            result.errors.append(f"Missing required fields: {missing_fields}")
            return False

        # Validate field types
        if not isinstance(sample.get('prompt'), str) or not sample['prompt'].strip():
            result.errors.append("Prompt must be non-empty string")

        if not isinstance(sample.get('response'), str) or not sample['response'].strip():
            result.errors.append("Response must be non-empty string")

        return len(result.errors) == 0

    def _validate_chess_sample(self, sample: Dict[str, Any], mode: str, result: ValidationResult) -> None:
        """Validate chess-specific samples (UCI/Tutor modes)."""
        prompt = sample['prompt']
        response = sample['response']

        # Extract FEN
        fen = self._extract_fen(prompt)
        if not fen:
            result.errors.append("No valid FEN found in prompt")
            return

        # Validate FEN
        try:
            board = chess.Board(fen)
        except Exception as e:
            result.errors.append(f"Invalid FEN: {e}")
            return

        # Check for terminal positions
        if board.is_checkmate():
            result.errors.append("Position is already checkmate")
            return
        elif board.is_stalemate():
            result.errors.append("Position is stalemate")
            return

        # Extract and validate move
        move = self._extract_move_from_response(response, mode)
        if not move:
            result.errors.append("No valid move found in response")
            return

        # Validate move legality
        try:
            chess_move = chess.Move.from_uci(move)
            if chess_move not in board.legal_moves:
                result.errors.append(f"Move {move} is not legal in position")
                return
        except Exception as e:
            result.errors.append(f"Invalid UCI move format: {e}")
            return

        # Additional quality checks
        self._check_move_quality(board, chess_move, result)

        # Store metadata
        result.metadata.update({
            'fen': fen,
            'move': move,
            'position_complexity': self._calculate_position_complexity(board),
            'move_type': self._classify_move(board, chess_move)
        })

    def _validate_director_sample(self, sample: Dict[str, Any], result: ValidationResult) -> None:
        """Validate director/Q&A samples."""
        prompt = sample['prompt']
        response = sample['response']

        # Basic length checks
        if len(prompt) < 10:
            result.warnings.append("Prompt seems too short for meaningful question")

        if len(response) < 20:
            result.warnings.append("Response seems too short for comprehensive answer")

        # Check for chess-related content
        chess_keywords = ['pawn', 'rook', 'bishop', 'knight', 'queen', 'king',
                         'check', 'mate', 'castle', 'en passant', 'fen', 'uci']
        content_has_chess = any(keyword in (prompt + response).lower() for keyword in chess_keywords)

        if not content_has_chess:
            result.warnings.append("Content may not be chess-related")

    def _check_move_quality(self, board: chess.Board, move: chess.Move, result: ValidationResult) -> None:
        """Check the quality of a chess move."""
        # Check if move captures
        if board.is_capture(move):
            result.metadata['is_capture'] = True

        # Check if move gives check
        board.push(move)
        try:
            if board.is_check():
                result.metadata['gives_check'] = True

            # Check if move is checkmate
            if board.is_checkmate():
                result.metadata['is_checkmate'] = True
        finally:
            board.pop()

    def _extract_fen(self, text: str) -> Optional[str]:
        """Extract FEN from text."""
        import re

        # Look for explicit FEN: prefix
        for line in text.splitlines():
            if line.lower().startswith('fen:'):
                fen = line.split(':', 1)[1].strip()
                if self._is_valid_fen(fen):
                    return fen

        # Look for FEN pattern anywhere in text
        fen_pattern = r'([rnbqkpRNBQKP1-8]+/){7}[rnbqkpRNBQKP1-8]+\s[wb]\s(?:K?Q?k?q?|-)\s(?:[a-h][36]|-)\s\d+\s\d+'
        matches = re.findall(fen_pattern, text)
        for match in matches:
            if self._is_valid_fen(match):
                return match

        return None

    def _is_valid_fen(self, fen: str) -> bool:
        """Check if FEN string is valid."""
        try:
            chess.Board(fen)
            return True
        except:
            return False

    def _extract_move_from_response(self, response: str, mode: str) -> Optional[str]:
        """Extract UCI move from response."""
        import re

        if mode == 'uci':
            # UCI mode: response should be just the move
            response = response.strip()
            if re.match(r'^[a-h][1-8][a-h][1-8][qrbn]?$', response):
                return response
        elif mode == 'tutor':
            # Tutor mode: look for "Best move: <uci>" pattern
            match = re.search(r'Best move:\s*([a-h][1-8][a-h][1-8][qrbn]?)', response, re.IGNORECASE)
            if match:
                return match.group(1).lower()

        # Fallback: find any UCI move
        moves = re.findall(r'\b([a-h][1-8][a-h][1-8][qrbn]?)\b', response)
        return moves[-1] if moves else None

    def _calculate_position_complexity(self, board: chess.Board) -> float:
        """Calculate position complexity score."""
        complexity = 0

        # Piece count
        piece_count = len([p for p in board.board_fen() if p.isalpha()])
        complexity += min(piece_count / 32, 1) * 0.3

        # Castling rights
        castling_rights = sum([
            board.has_kingside_castling_rights(chess.WHITE),
            board.has_queenside_castling_rights(chess.WHITE),
            board.has_kingside_castling_rights(chess.BLACK),
            board.has_queenside_castling_rights(chess.BLACK)
        ])
        complexity += (castling_rights / 4) * 0.2

        # Check status
        if board.is_check():
            complexity += 0.3

        # Pawn structure (simplified)
        pawns = [square for square in chess.SQUARES
                if board.piece_at(square) and board.piece_at(square).piece_type == chess.PAWN]
        complexity += min(len(pawns) / 16, 1) * 0.2

        return complexity

    def _classify_move(self, board: chess.Board, move: chess.Move) -> str:
        """Classify the type of move."""
        if board.is_capture(move):
            return "capture"
        elif board.gives_check(move):
            return "check"
        elif move.promotion:
            return "promotion"
        elif abs(chess.square_file(move.from_square) - chess.square_file(move.to_square)) > 1:
            return "long_range"
        else:
            return "quiet"

    def _assess_sample_quality(self, sample: Dict[str, Any], mode: str, result: ValidationResult) -> float:
        """Assess overall sample quality score (0-1)."""
        # Note: Quality assessment happens before is_valid is set, so we assess quality regardless of validation status

        quality_factors = []

        # Basic content quality
        prompt_len = len(sample['prompt'])
        response_len = len(sample['response'])

        if mode == 'uci':
            # UCI should be concise
            response = sample['response'].strip()
            quality_factors.append(1.0 if 3 <= len(response) <= 6 else 0.5)
        elif mode == 'tutor':
            # Tutor should be informative
            word_count = len(sample['response'].split())
            quality_factors.append(1.0 if 20 <= word_count <= 150 else 0.7)
        else:
            # Director should be substantial
            quality_factors.append(1.0 if len(sample['response']) > 50 else 0.6)

        # Always include a base quality factor
        quality_factors.append(0.8)  # Base quality for valid samples

        # Metadata quality
        meta = sample.get('meta', {})
        if 'rating' in meta:
            rating = meta['rating']
            quality_factors.append(1.0 if 1000 <= rating <= 2500 else 0.7)
        else:
            # No rating available, neutral score
            quality_factors.append(0.5)

        # Chess-specific quality
        if 'position_complexity' in result.metadata:
            complexity = result.metadata['position_complexity']
            # Prefer medium complexity
            quality_factors.append(1.0 if 0.3 <= complexity <= 0.8 else 0.8)
        else:
            # No complexity data, neutral score
            quality_factors.append(0.5)

        return statistics.mean(quality_factors) if quality_factors else 0.5

    def _process_validation_results(self, validation_results: List[ValidationResult],
                                 samples: List[Dict[str, Any]], report: DatasetValidationReport) -> None:
        """Process validation results into comprehensive report."""
        quality_scores = []
        position_complexities = []
        move_types = Counter()

        for result in validation_results:
            if result.is_valid:
                report.valid_samples += 1
                quality_scores.append(result.quality_score)

                # Categorize quality
                if result.quality_score >= 0.9:
                    report.quality_distribution['excellent'] += 1
                elif result.quality_score >= 0.7:
                    report.quality_distribution['good'] += 1
                elif result.quality_score >= 0.5:
                    report.quality_distribution['fair'] += 1
                else:
                    report.quality_distribution['poor'] += 1

                # Collect metadata stats
                if 'position_complexity' in result.metadata:
                    position_complexities.append(result.metadata['position_complexity'])

                if 'move_type' in result.metadata:
                    move_types[result.metadata['move_type']] += 1

            else:
                report.invalid_samples += 1

            # Aggregate errors and warnings
            for error in result.errors:
                report.error_summary[error] += 1

            for warning in result.warnings:
                report.warning_summary[warning] += 1

        # Calculate quality metrics
        if quality_scores:
            report.quality_metrics = {
                'mean_quality': statistics.mean(quality_scores),
                'median_quality': statistics.median(quality_scores),
                'quality_std': statistics.stdev(quality_scores) if len(quality_scores) > 1 else 0,
                'min_quality': min(quality_scores),
                'max_quality': max(quality_scores)
            }

        # Position stats
        if position_complexities:
            report.position_stats = {
                'mean_complexity': statistics.mean(position_complexities),
                'complexity_distribution': self._create_histogram(position_complexities, 5)
            }

        # Move stats
        report.move_stats = dict(move_types.most_common(10))


    def _create_histogram(self, values: List[float], bins: int) -> Dict[str, int]:
        """Create a simple histogram of values."""
        if not values:
            return {}

        min_val, max_val = min(values), max(values)
        if min_val == max_val:
            return {f"{min_val:.2f}": len(values)}

        bin_size = (max_val - min_val) / bins
        histogram = defaultdict(int)

        for value in values:
            bin_idx = int((value - min_val) / bin_size)
            if bin_idx == bins:
                bin_idx = bins - 1
            bin_start = min_val + bin_idx * bin_size
            bin_end = min_val + (bin_idx + 1) * bin_size
            histogram[f"{bin_start:.2f}-{bin_end:.2f}"] += 1

        return dict(histogram)


def main():
    parser = argparse.ArgumentParser(description="Comprehensive dataset validation for ChessGemma")
    parser.add_argument('--input', required=True, help='Input JSONL file')
    parser.add_argument('--mode', choices=['uci', 'tutor', 'director'], default='uci')
    parser.add_argument('--output_report', help='Output JSON report file')
    parser.add_argument('--workers', type=int, default=4, help='Number of worker threads')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')

    args = parser.parse_args()

    input_file = Path(args.input)
    if not input_file.exists():
        print(f"Error: Input file {input_file} does not exist")
        return

    print(f"Validating dataset: {input_file}")
    print(f"Mode: {args.mode}")
    print(f"Workers: {args.workers}")
    print("-" * 50)

    validator = ChessDatasetValidator(max_workers=args.workers)
    report = validator.validate_dataset(input_file, args.mode)

    # Print summary
    print("VALIDATION SUMMARY")
    print("=" * 50)
    print(f"Total samples: {report.total_samples}")
    print(f"Valid samples: {report.valid_samples} ({report.valid_samples/report.total_samples*100:.1f}%)")
    print(f"Invalid samples: {report.invalid_samples} ({report.invalid_samples/report.total_samples*100:.1f}%)")
    print()

    if report.quality_metrics:
        print("QUALITY METRICS")
        print("-" * 30)
        for key, value in report.quality_metrics.items():
            if isinstance(value, float):
                print(f"{key}: {value:.3f}")
            else:
                print(f"{key}: {value}")
        print()

    if report.quality_distribution:
        print("QUALITY DISTRIBUTION")
        print("-" * 30)
        for quality, count in sorted(report.quality_distribution.items()):
            pct = count / report.valid_samples * 100
            print(f"{quality}: {count} ({pct:.1f}%)")
        print()

    if report.error_summary:
        print("TOP ERRORS")
        print("-" * 30)
        for error, count in sorted(report.error_summary.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"{count:4d}: {error}")
        print()

    if report.warning_summary:
        print("TOP WARNINGS")
        print("-" * 30)
        for warning, count in sorted(report.warning_summary.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"{count:4d}: {warning}")
        print()

    print(f"Processing time: {report.processing_time:.2f} seconds")
    print(f"Samples/second: {report.total_samples/report.processing_time:.1f}")

    # Save detailed report
    if args.output_report:
        output_file = Path(args.output_report)
        report_dict = {
            'summary': {
                'total_samples': report.total_samples,
                'valid_samples': report.valid_samples,
                'invalid_samples': report.invalid_samples,
                'valid_percentage': report.valid_samples / report.total_samples * 100 if report.total_samples > 0 else 0,
            },
            'quality_metrics': report.quality_metrics,
            'quality_distribution': dict(report.quality_distribution),
            'error_summary': dict(report.error_summary),
            'warning_summary': dict(report.warning_summary),
            'position_stats': report.position_stats,
            'move_stats': report.move_stats,
            'processing_time': report.processing_time,
            'validation_errors': report.validation_errors,
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report_dict, f, indent=2, ensure_ascii=False)

        print(f"\nDetailed report saved to: {output_file}")


if __name__ == '__main__':
    main()
