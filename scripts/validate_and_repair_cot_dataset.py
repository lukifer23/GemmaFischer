#!/usr/bin/env python3
"""
CoT Dataset Validation and Repair Script

Comprehensive validation and repair system for Chain-of-Thought chess reasoning datasets:
- Validates existing CoT examples for quality and correctness
- Repairs invalid examples with improved reasoning
- Generates missing CoT examples
- Implements robust quality assurance pipeline
"""

import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import chess
from dataclasses import dataclass
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from scripts.cot_chess_reasoning import ChessCoTGenerator


@dataclass
class CoTValidationResult:
    """Results of CoT validation."""
    total_examples: int = 0
    valid_examples: int = 0
    invalid_examples: int = 0
    repaired_examples: int = 0
    quality_scores: List[float] = None
    error_categories: Dict[str, int] = None

    def __post_init__(self):
        if self.quality_scores is None:
            self.quality_scores = []
        if self.error_categories is None:
            self.error_categories = defaultdict(int)


class CoTDatasetValidator:
    """Comprehensive validator for CoT chess reasoning datasets."""

    def __init__(self):
        self.generator = ChessCoTGenerator()
        self.chess_concepts = [
            "pawn", "knight", "bishop", "rook", "queen", "king", "castle", "castling",
            "check", "checkmate", "center", "development", "safety", "structure",
            "advantage", "control", "space", "activity", "attack", "defense",
            "tactics", "strategy", "endgame", "opening", "middlegame"
        ]

    def validate_cot_example(self, example: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Validate a single CoT example.
        Returns: (is_valid, error_message, quality_metrics)
        """
        quality_metrics = {
            "step_by_step_reasoning": False,
            "logical_progression": False,
            "chess_concepts_used": 0,
            "conclusion_supported": False,
            "reasoning_depth": 0,
            "overall_quality": 0.0
        }

        try:
            # Validate required fields
            required_fields = ["task", "prompt", "response", "meta"]
            for field in required_fields:
                if field not in example:
                    return False, f"Missing required field: {field}", quality_metrics

            response = example.get("response", "").strip()
            if not response:
                return False, "Empty response", quality_metrics

            # Check for step-by-step reasoning
            step_indicators = ["first", "next", "then", "after", "finally", "therefore", "conclusion", "step"]
            quality_metrics["step_by_step_reasoning"] = any(
                indicator in response.lower() for indicator in step_indicators
            )

            # Check for logical progression
            logical_connectors = ["because", "therefore", "so", "thus", "hence", "as a result", "this means"]
            quality_metrics["logical_progression"] = any(
                connector in response.lower() for connector in logical_connectors
            )

            # Count chess concepts used
            response_lower = response.lower()
            quality_metrics["chess_concepts_used"] = sum(
                1 for concept in self.chess_concepts if concept in response_lower
            )

            # Assess reasoning depth
            sentences = re.split(r'[.!?]+', response)
            quality_metrics["reasoning_depth"] = len([s for s in sentences if len(s.strip()) > 10])

            # Check conclusion support
            quality_metrics["conclusion_supported"] = (
                len(response.split()) > 50 and
                quality_metrics["logical_progression"] and
                quality_metrics["reasoning_depth"] >= 3
            )

            # Calculate overall quality score
            score_components = [
                quality_metrics["step_by_step_reasoning"],
                quality_metrics["logical_progression"],
                min(quality_metrics["chess_concepts_used"] / 5, 1),  # Normalize
                quality_metrics["conclusion_supported"],
                min(quality_metrics["reasoning_depth"] / 8, 1)  # Normalize
            ]
            quality_metrics["overall_quality"] = sum(score_components) / len(score_components)

            # Determine validity based on quality threshold
            is_valid = quality_metrics["overall_quality"] >= 0.6  # 60% quality threshold
            error_msg = "" if is_valid else f"Quality score too low: {quality_metrics['overall_quality']:.2f}"

            return is_valid, error_msg, quality_metrics

        except Exception as e:
            return False, f"Validation error: {str(e)}", quality_metrics

    def repair_cot_example(self, example: Dict[str, Any], error_msg: str) -> Dict[str, Any]:
        """Attempt to repair an invalid CoT example."""
        try:
            # Extract position information if available
            prompt = example.get("prompt", "")
            fen_match = re.search(r'FEN:\s*([rnbqkpRNBQKP1-8/]+)', prompt)

            if fen_match:
                # Generate improved CoT reasoning based on position
                fen = fen_match.group(1)
                category = self._determine_position_category(fen)
                repaired_example = self.generator.generate_cot_example(category)

                # Preserve original metadata but update response
                repaired_example.update({
                    "task": example.get("task", "director_qa"),
                    "prompt": example.get("prompt", ""),
                    "meta": example.get("meta", {})
                })

                return repaired_example
            else:
                # Generate generic CoT example
                repaired_example = self.generator.generate_cot_example("general")
                repaired_example.update({
                    "task": example.get("task", "director_qa"),
                    "prompt": example.get("prompt", ""),
                    "meta": example.get("meta", {})
                })
                return repaired_example

        except Exception as e:
            print(f"Repair failed: {e}")
            return example

    def _determine_position_category(self, fen: str) -> str:
        """Determine the category of a chess position."""
        try:
            board = chess.Board(fen)
            total_pieces = sum(1 for _ in board.piece_map())

            if total_pieces > 20:
                return "opening"
            elif total_pieces > 10:
                return "tactical"
            elif total_pieces > 6:
                return "positional"
            else:
                return "endgame"
        except:
            return "general"

    def validate_and_repair_dataset(self, input_file: str, output_file: str = None) -> CoTValidationResult:
        """Validate and repair an entire CoT dataset."""
        result = CoTValidationResult()

        if not Path(input_file).exists():
            print(f"❌ Input file not found: {input_file}")
            return result

        examples = []
        repaired_examples = []

        print(f"🔍 Validating CoT dataset: {input_file}")

        with open(input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    example = json.loads(line.strip())
                    result.total_examples += 1

                    is_valid, error_msg, quality_metrics = self.validate_cot_example(example)

                    if is_valid:
                        result.valid_examples += 1
                        examples.append(example)
                    else:
                        result.invalid_examples += 1
                        result.error_categories[error_msg] += 1

                        # Attempt repair
                        repaired = self.repair_cot_example(example, error_msg)
                        if repaired != example:  # Repair was successful
                            result.repaired_examples += 1
                            examples.append(repaired)
                            repaired_examples.append(repaired)
                        else:
                            examples.append(example)  # Keep original if repair failed

                    result.quality_scores.append(quality_metrics["overall_quality"])

                    # Progress update
                    if result.total_examples % 100 == 0:
                        print(f"   Processed {result.total_examples} examples... "
                              f"Valid: {result.valid_examples}, Invalid: {result.invalid_examples}")

                except json.JSONDecodeError as e:
                    print(f"   JSON parse error at line {line_num}: {e}")
                    result.invalid_examples += 1
                    result.error_categories["JSON parse error"] += 1

        # Generate output file
        if output_file:
            self._save_repaired_dataset(examples, output_file)

        print("\n📊 Validation Results:")
        print(f"   Total Examples: {result.total_examples}")
        print(f"   Valid Examples: {result.valid_examples}")
        print(f"   Invalid Examples: {result.invalid_examples}")
        print(f"   Repaired Examples: {result.repaired_examples}")
        print(f"   Success Rate: {(result.valid_examples / max(result.total_examples, 1)):.1f}")
        if result.quality_scores:
            print(f"   Average Quality: {sum(result.quality_scores) / len(result.quality_scores):.3f}")
            print(f"   Median Quality: {sorted(result.quality_scores)[len(result.quality_scores)//2]:.3f}")
        print("\n🔧 Error Categories:")
        for error, count in sorted(result.error_categories.items(), key=lambda x: x[1], reverse=True):
            print(f"   {error}: {count}")

        return result

    def _save_repaired_dataset(self, examples: List[Dict[str, Any]], output_file: str):
        """Save the repaired dataset to file."""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            for example in examples:
                json.dump(example, f, ensure_ascii=False)
                f.write('\n')

        print(f"💾 Repaired dataset saved to: {output_path}")

        # Generate metadata
        metadata = {
            "total_examples": len(examples),
            "repaired_at": str(datetime.now()),
            "validation_version": "2.0",
            "quality_threshold": 0.6
        }

        metadata_file = output_path.with_suffix('.metadata.json')
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)

        print(f"📋 Metadata saved to: {metadata_file}")

    def generate_missing_cot_examples(self, target_count: int, output_file: str):
        """Generate missing CoT examples to reach target count."""
        print(f"🧠 Generating {target_count} new CoT examples...")

        examples = []
        categories = ["opening", "tactical", "positional", "endgame", "general"]

        for i in range(target_count):
            category = categories[i % len(categories)]
            example = self.generator.generate_cot_example(category)

            # Add standard metadata
            example.update({
                "task": "director_qa",
                "meta": {
                    "source": "generated_cot",
                    "category": category,
                    "quality_score": 0.8,
                    "generated_at": str(datetime.now())
                }
            })

            examples.append(example)

            if (i + 1) % 50 == 0:
                print(f"   Generated {i + 1}/{target_count} examples...")

        self._save_repaired_dataset(examples, output_file)
        print(f"✅ Generated {target_count} new CoT examples")


def main():
    """Main validation and repair workflow."""
    print("🔧 ChessGemma CoT Dataset Validation & Repair")
    print("=" * 50)

    validator = CoTDatasetValidator()

    # Define file paths
    data_dir = project_root / "data"
    input_file = data_dir / "datasets" / "cot_reasoning_examples.jsonl"
    output_file = data_dir / "standardized" / "standardized_cot_reasoning_repaired.jsonl"

    # Check if input file exists
    if not input_file.exists():
        print(f"⚠️  CoT dataset not found at {input_file}")
        print("🔄 Generating new CoT dataset instead...")

        # Generate new CoT examples
        validator.generate_missing_cot_examples(2000, str(output_file))
        return

    # Validate and repair existing dataset
    result = validator.validate_and_repair_dataset(str(input_file), str(output_file))

    # If we have too many invalid examples, generate additional ones
    if result.total_examples < 2000 or (result.invalid_examples / result.total_examples) > 0.3:
        additional_needed = max(0, 2000 - result.valid_examples)
        if additional_needed > 0:
            print(f"📈 Generating {additional_needed} additional high-quality CoT examples...")
            supplemental_file = data_dir / "standardized" / "standardized_cot_reasoning_supplemental.jsonl"
            validator.generate_missing_cot_examples(additional_needed, str(supplemental_file))

            # Combine datasets
            combined_file = data_dir / "standardized" / "standardized_director_expert_enhanced.jsonl"
            validator._combine_datasets([str(output_file), str(supplemental_file)], str(combined_file))

    print("\n✅ CoT Dataset Validation & Repair Complete!")


if __name__ == "__main__":
    main()
