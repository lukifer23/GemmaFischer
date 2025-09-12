#!/usr/bin/env python3
"""
Data Format Standardization Script

Standardizes all chess training data to a consistent format:
{
    "task": "engine_uci" | "tutor_explain" | "director_qa",
    "prompt": "string with FEN and instructions",
    "response": "model response",
    "meta": {
        "fen": "FEN string",
        "source": "data source",
        "rating": int or null,
        "topic": "tactics|strategy|endgame|opening|general",
        "difficulty": "beginner|intermediate|advanced",
        "quality_score": float (0-1)
    }
}
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import re
import sys

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class StandardizedSample:
    """Standardized training sample."""
    task: str
    prompt: str
    response: str
    meta: Dict[str, Any]


class DataFormatStandardizer:
    """Standardizes various chess data formats to a consistent schema."""

    def __init__(self):
        self.conversion_stats = {
            'processed': 0,
            'converted': 0,
            'skipped': 0,
            'errors': 0
        }

    def standardize_dataset(self, input_path: Path, output_path: Path,
                          target_format: str = 'expert') -> Dict[str, Any]:
        """Standardize a dataset to the target format."""
        logger.info(f"Standardizing {input_path} to {target_format} format")

        samples = []
        with open(input_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                self.conversion_stats['processed'] += 1
                try:
                    sample = json.loads(line.strip())
                    standardized = self._standardize_sample(sample, target_format)
                    if standardized:
                        samples.append(standardized.__dict__)
                        self.conversion_stats['converted'] += 1
                    else:
                        self.conversion_stats['skipped'] += 1
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON on line {line_num}: {e}")
                    self.conversion_stats['errors'] += 1
                    continue

        # Write standardized dataset
        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')

        logger.info(f"Standardized {len(samples)} samples to {output_path}")
        return self.conversion_stats

    def _standardize_sample(self, sample: Dict[str, Any], target_format: str) -> Optional[StandardizedSample]:
        """Convert a sample to standardized format."""
        # Detect input format
        input_format = self._detect_format(sample)

        if input_format == 'expert':
            # Already in expert format, just validate and clean
            return self._standardize_expert_format(sample)
        elif input_format == 'legacy_text':
            # Convert from legacy text format
            return self._convert_legacy_text_format(sample)
        elif input_format == 'raw_fen':
            # Convert from raw FEN format
            return self._convert_raw_fen_format(sample)
        else:
            logger.warning(f"Unknown format for sample: {sample.keys()}")
            return None

    def _detect_format(self, sample: Dict[str, Any]) -> str:
        """Detect the format of a sample."""
        if 'task' in sample and 'prompt' in sample and 'response' in sample:
            return 'expert'
        elif 'text' in sample:
            return 'legacy_text'
        elif 'fen' in sample and 'solution' in sample:
            return 'raw_fen'
        else:
            return 'unknown'

    def _standardize_expert_format(self, sample: Dict[str, Any]) -> Optional[StandardizedSample]:
        """Validate and standardize expert format."""
        try:
            # Validate required fields
            required = ['task', 'prompt', 'response']
            if not all(k in sample for k in required):
                return None

            # Validate task type
            valid_tasks = ['engine_uci', 'tutor_explain', 'director_qa']
            if sample['task'] not in valid_tasks:
                # Try to infer task from content
                sample['task'] = self._infer_task_from_content(sample)

            # Ensure meta field exists
            if 'meta' not in sample:
                sample['meta'] = {}

            # Extract and validate FEN
            fen = self._extract_fen_from_sample(sample)
            if fen:
                sample['meta']['fen'] = fen

            # Add quality score if missing
            if 'quality_score' not in sample['meta']:
                sample['meta']['quality_score'] = 0.8  # Default quality

            # Clean and validate response
            sample['response'] = self._clean_response(sample['response'], sample['task'])

            return StandardizedSample(
                task=sample['task'],
                prompt=sample['prompt'],
                response=sample['response'],
                meta=sample.get('meta', {})
            )

        except Exception as e:
            logger.warning(f"Error standardizing expert format: {e}")
            return None

    def _convert_legacy_text_format(self, sample: Dict[str, Any]) -> Optional[StandardizedSample]:
        """Convert from legacy text format to expert format."""
        try:
            text = sample.get('text', '')

            # Try to extract FEN from text
            fen = self._extract_fen_from_text(text)
            if not fen:
                # Skip samples without FEN (general Q&A)
                return None

            # Determine task type based on content
            task = self._infer_task_from_text(text)

            # Extract prompt and response from text
            prompt, response = self._split_text_conversation(text)

            # Create meta information
            meta = {
                'fen': fen,
                'source': sample.get('source', 'unknown'),
                'rating': None,
                'topic': sample.get('category', 'general'),
                'difficulty': sample.get('difficulty', 'intermediate'),
                'quality_score': 0.6  # Lower quality for converted legacy data
            }

            return StandardizedSample(
                task=task,
                prompt=prompt,
                response=response,
                meta=meta
            )

        except Exception as e:
            logger.warning(f"Error converting legacy text format: {e}")
            return None

    def _convert_raw_fen_format(self, sample: Dict[str, Any]) -> Optional[StandardizedSample]:
        """Convert from raw FEN format to expert format."""
        try:
            fen = sample.get('fen', '')
            solution = sample.get('solution', [])

            if not fen or not solution:
                return None

            # Create prompt
            prompt = f"FEN: {fen}\nMove:\nStyle: balanced\nMode: Engine\nGenerate the best move in UCI format (e.g., e2e4). Respond with only the move."

            # Create response (first move in solution)
            response = solution[0] if isinstance(solution, list) and solution else str(solution)

            # Create meta
            meta = {
                'fen': fen,
                'source': sample.get('source', 'raw_fen'),
                'rating': sample.get('rating'),
                'topic': sample.get('topic', 'tactics'),
                'difficulty': 'intermediate',
                'quality_score': 0.9
            }

            return StandardizedSample(
                task='engine_uci',
                prompt=prompt,
                response=response,
                meta=meta
            )

        except Exception as e:
            logger.warning(f"Error converting raw FEN format: {e}")
            return None

    def _infer_task_from_content(self, sample: Dict[str, Any]) -> str:
        """Infer task type from sample content."""
        prompt = sample.get('prompt', '').lower()
        response = sample.get('response', '').lower()

        if 'uci format' in prompt or 'only the move' in prompt:
            return 'engine_uci'
        elif 'analyze' in prompt or 'explain' in prompt or 'best move:' in response:
            return 'tutor_explain'
        else:
            return 'director_qa'

    def _infer_task_from_text(self, text: str) -> str:
        """Infer task type from legacy text format."""
        text_lower = text.lower()

        if 'move' in text_lower and ('uci' in text_lower or 'e2e4' in text_lower):
            return 'engine_uci'
        elif 'analyze' in text_lower or 'explain' in text_lower:
            return 'tutor_explain'
        else:
            return 'director_qa'

    def _extract_fen_from_sample(self, sample: Dict[str, Any]) -> Optional[str]:
        """Extract FEN from sample."""
        # Check meta first
        meta = sample.get('meta', {})
        if 'fen' in meta:
            return meta['fen']

        # Extract from prompt
        prompt = sample.get('prompt', '')
        return self._extract_fen_from_text(prompt)

    def _extract_fen_from_text(self, text: str) -> Optional[str]:
        """Extract FEN from text."""
        import re

        # Look for FEN pattern
        fen_pattern = r'([rnbqkpRNBQKP1-8]+/){7}[rnbqkpRNBQKP1-8]+\s[wb]\s(?:K?Q?k?q?|-)\s(?:[a-h][36]|-)\s\d+\s\d+'
        matches = re.findall(fen_pattern, text)

        for match in matches:
            try:
                # Validate FEN
                import chess
                chess.Board(match)
                return match
            except:
                continue

        return None

    def _clean_response(self, response: str, task: str) -> str:
        """Clean and standardize response format."""
        response = response.strip()

        if task == 'engine_uci':
            # Extract just the UCI move
            import re
            moves = re.findall(r'\b([a-h][1-8][a-h][1-8][qrbn]?)\b', response)
            if moves:
                return moves[0].lower()
            return response.lower()

        elif task == 'tutor_explain':
            # Ensure it ends with "Best move: <uci>"
            if not response.lower().endswith('best move:'):
                # Try to find existing "Best move:" pattern
                import re
                best_move_match = re.search(r'best move:\s*([a-h][1-8][a-h][1-8][qrbn]?)', response, re.IGNORECASE)
                if best_move_match:
                    move = best_move_match.group(1).lower()
                    # Remove any existing "Best move:" and add it at the end
                    response = re.sub(r'best move:.*', '', response, flags=re.IGNORECASE).strip()
                    response = f"{response}\nBest move: {move}"
                else:
                    # Extract move and add "Best move:" format
                    moves = re.findall(r'\b([a-h][1-8][a-h][1-8][qrbn]?)\b', response)
                    if moves:
                        move = moves[-1].lower()
                        response = f"{response}\nBest move: {move}"

        return response

    def _split_text_conversation(self, text: str) -> Tuple[str, str]:
        """Split legacy text format into prompt and response."""
        # Split by model response marker
        if '<start_of_turn>model\n' in text:
            parts = text.split('<start_of_turn>model\n', 1)
            if len(parts) == 2:
                prompt = parts[0].replace('<start_of_turn>system\n', '').replace('<start_of_turn>user\n', '').strip()
                response = parts[1].replace('<end_of_turn>', '').strip()
                return prompt, response

        # Fallback: return whole text as response
        return "Analyze this chess position.", text

    def batch_standardize_directory(self, input_dir: Path, output_dir: Path,
                                  target_format: str = 'expert') -> Dict[str, Any]:
        """Standardize all datasets in a directory."""
        output_dir.mkdir(parents=True, exist_ok=True)

        total_stats = {
            'files_processed': 0,
            'total_processed': 0,
            'total_converted': 0,
            'total_skipped': 0,
            'total_errors': 0
        }

        # Find all JSONL files
        jsonl_files = list(input_dir.glob('*.jsonl'))

        for input_file in jsonl_files:
            if input_file.name.startswith('standardized_'):
                continue  # Skip already standardized files

            output_file = output_dir / f"standardized_{input_file.name}"

            try:
                stats = self.standardize_dataset(input_file, output_file, target_format)
                total_stats['files_processed'] += 1
                total_stats['total_processed'] += stats['processed']
                total_stats['total_converted'] += stats['converted']
                total_stats['total_skipped'] += stats['skipped']
                total_stats['total_errors'] += stats['errors']

            except Exception as e:
                logger.error(f"Failed to standardize {input_file}: {e}")
                total_stats['total_errors'] += 1

        # Write summary report
        summary_file = output_dir / "standardization_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(total_stats, f, indent=2)

        logger.info(f"Standardization complete. Summary: {total_stats}")
        return total_stats


def main():
    parser = argparse.ArgumentParser(description="Standardize chess training data formats")
    parser.add_argument('--input', required=True, help='Input file or directory')
    parser.add_argument('--output', required=True, help='Output file or directory')
    parser.add_argument('--format', default='expert', choices=['expert'],
                       help='Target format (currently only expert supported)')

    args = parser.parse_args()

    standardizer = DataFormatStandardizer()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if input_path.is_file():
        # Standardize single file
        stats = standardizer.standardize_dataset(input_path, output_path, args.format)
        print(f"Standardization complete: {stats}")

    elif input_path.is_dir():
        # Standardize directory
        stats = standardizer.batch_standardize_directory(input_path, output_path, args.format)
        print(f"Batch standardization complete: {stats}")

    else:
        print(f"Input path does not exist: {input_path}")
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
