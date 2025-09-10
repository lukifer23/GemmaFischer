#!/usr/bin/env python3
"""
Chess Data Validation and Quality Control

Validates chess training datasets for:
- Format correctness (JSON structure)
- Chess validity (FEN, moves, positions)
- Content quality (relevance, length, clarity)
- Data consistency and completeness

Generates quality reports and filtered datasets.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import chess
from datetime import datetime
import statistics


class ChessDataValidator:
    """Validate and quality-check chess training datasets."""
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.validation_dir = data_dir / "validation"
        self.validation_dir.mkdir(parents=True, exist_ok=True)
        
        # Quality thresholds
        self.quality_thresholds = {
            'min_question_length': 10,
            'max_question_length': 500,
            'min_answer_length': 10,
            'max_answer_length': 1000,
            'min_quality_score': 3.0,  # Reasonable threshold
            'max_duplicate_ratio': 0.1
        }
        
        # Chess-related keywords for relevance checking
        self.chess_keywords = [
            'pawn', 'knight', 'bishop', 'rook', 'queen', 'king',
            'check', 'mate', 'castling', 'en passant', 'promotion',
            'opening', 'middlegame', 'endgame', 'tactics', 'strategy',
            'fork', 'pin', 'skewer', 'discovered', 'double attack',
            'position', 'move', 'square', 'rank', 'file', 'diagonal',
            'attack', 'defend', 'capture', 'threat', 'combination',
            'sacrifice', 'exchange', 'material', 'advantage', 'disadvantage',
            'fen', 'uci', 'algebraic', 'notation', 'board', 'piece',
            'tactical', 'positional', 'endgame', 'opening', 'analysis',
            'evaluate', 'calculate', 'variation', 'line', 'sequence',
            'win', 'lose', 'draw', 'result', 'game', 'play', 'player'
        ]
    
    def validate_dataset(self, input_file: Path, output_file: Optional[Path] = None) -> Dict[str, Any]:
        """Validate a chess training dataset."""
        print(f"🎯 Validating dataset: {input_file}")
        
        if not input_file.exists():
            print(f"❌ Input file not found: {input_file}")
            return {}
        
        validation_results = {
            'input_file': str(input_file),
            'validation_date': datetime.now().isoformat(),
            'total_examples': 0,
            'valid_examples': 0,
            'invalid_examples': 0,
            'quality_scores': [],
            'errors': [],
            'warnings': [],
            'statistics': {}
        }
        
        valid_examples = []
        invalid_examples = []
        
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                
                try:
                    example = json.loads(line)
                    validation_results['total_examples'] += 1
                    
                    # Validate example
                    is_valid, errors, warnings, quality_score = self._validate_example(example, line_num)
                    
                    if is_valid:
                        valid_examples.append(example)
                        validation_results['valid_examples'] += 1
                    
                    # Always add quality score for analysis
                    validation_results['quality_scores'].append(quality_score)
                    
                    if not is_valid:
                        invalid_examples.append({
                            'line_number': line_num,
                            'example': example,
                            'errors': errors,
                            'warnings': warnings
                        })
                        validation_results['invalid_examples'] += 1
                    
                    # Collect errors and warnings
                    validation_results['errors'].extend([f"Line {line_num}: {error}" for error in errors])
                    validation_results['warnings'].extend([f"Line {line_num}: {warning}" for warning in warnings])
                
                except json.JSONDecodeError as e:
                    validation_results['invalid_examples'] += 1
                    validation_results['errors'].append(f"Line {line_num}: JSON decode error - {e}")
        
        # Calculate statistics
        validation_results['statistics'] = self._calculate_statistics(valid_examples)
        
        # Save filtered dataset if output file specified
        if output_file and valid_examples:
            self._save_filtered_dataset(valid_examples, output_file)
            validation_results['filtered_output'] = str(output_file)
        
        # Save validation report
        report_file = self.validation_dir / f"validation_report_{input_file.stem}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(validation_results, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Validation complete!")
        print(f"   📊 Total examples: {validation_results['total_examples']:,}")
        print(f"   ✅ Valid examples: {validation_results['valid_examples']:,}")
        print(f"   ❌ Invalid examples: {validation_results['invalid_examples']:,}")
        print(f"   📁 Report: {report_file}")
        
        return validation_results
    
    def _validate_example(self, example: Dict[str, Any], line_num: int = 0) -> Tuple[bool, List[str], List[str], float]:
        """Validate a single training example."""
        errors = []
        warnings = []
        quality_score = 0.0
        
        # Check required fields
        required_fields = ['text', 'conversations']
        for field in required_fields:
            if field not in example:
                errors.append(f"Missing required field: {field}")
                return False, errors, warnings, 0.0
        
        # Validate text field
        if not isinstance(example['text'], str) or not example['text'].strip():
            errors.append("Text field is empty or invalid")
            return False, errors, warnings, 0.0
        
        # Validate conversations
        if not isinstance(example['conversations'], list) or len(example['conversations']) < 2:
            errors.append("Conversations field must be a list with at least 2 items")
            return False, errors, warnings, 0.0
        
        # Validate conversation structure
        for i, conv in enumerate(example['conversations']):
            if not isinstance(conv, dict):
                errors.append(f"Conversation {i} is not a dictionary")
                continue
            
            if 'role' not in conv or 'content' not in conv:
                errors.append(f"Conversation {i} missing role or content")
                continue
            
            if conv['role'] not in ['system', 'user', 'assistant']:
                errors.append(f"Conversation {i} has invalid role: {conv['role']}")
                continue
            
            if not isinstance(conv['content'], str) or not conv['content'].strip():
                errors.append(f"Conversation {i} has empty content")
                continue
        
        # Extract question and answer
        question = ""
        answer = ""
        
        for conv in example['conversations']:
            if conv['role'] == 'user':
                question = conv['content']
            elif conv['role'] == 'assistant':
                answer = conv['content']
        
        if not question or not answer:
            errors.append("Missing user question or assistant answer")
            return False, errors, warnings, 0.0
        
        # Validate lengths
        if len(question) < self.quality_thresholds['min_question_length']:
            errors.append(f"Question too short: {len(question)} chars (min: {self.quality_thresholds['min_question_length']})")
        elif len(question) > self.quality_thresholds['max_question_length']:
            warnings.append(f"Question very long: {len(question)} chars (max: {self.quality_thresholds['max_question_length']})")
        
        if len(answer) < self.quality_thresholds['min_answer_length']:
            errors.append(f"Answer too short: {len(answer)} chars (min: {self.quality_thresholds['min_answer_length']})")
        elif len(answer) > self.quality_thresholds['max_answer_length']:
            warnings.append(f"Answer very long: {len(answer)} chars (max: {self.quality_thresholds['max_answer_length']})")
        
        # Check chess relevance
        chess_relevance = self._check_chess_relevance(question + " " + answer)
        if chess_relevance < 0.1:  # Lowered threshold
            errors.append(f"Low chess relevance: {chess_relevance:.2f} (min: 0.1)")
        elif chess_relevance < 0.3:
            warnings.append(f"Moderate chess relevance: {chess_relevance:.2f}")
        
        # Validate FEN if present
        fen_errors = self._validate_fen(example)
        errors.extend(fen_errors)
        
        # Validate moves if present
        move_errors = self._validate_moves(example)
        errors.extend(move_errors)
        
        # Calculate quality score
        quality_score = self._calculate_quality_score(example, chess_relevance)
        
        # Check if example passes validation
        is_valid = len(errors) == 0 and quality_score >= self.quality_thresholds['min_quality_score']
        
        # Debug: print validation details for first few examples (commented out for production)
        # if line_num <= 3:  # Only debug first 3 examples
        #     print(f"   Debug - Errors: {len(errors)}, Quality: {quality_score:.2f}, Threshold: {self.quality_thresholds['min_quality_score']}, Valid: {is_valid}")
        
        return is_valid, errors, warnings, quality_score
    
    def _check_chess_relevance(self, text: str) -> float:
        """Check how relevant the text is to chess."""
        text_lower = text.lower()
        
        # Count chess keywords
        chess_count = sum(1 for keyword in self.chess_keywords if keyword in text_lower)
        
        # Calculate relevance score
        total_words = len(text.split())
        if total_words == 0:
            return 0.0
        
        relevance = chess_count / total_words
        return min(relevance, 1.0)
    
    def _validate_fen(self, example: Dict[str, Any]) -> List[str]:
        """Validate FEN string if present."""
        errors = []
        
        # Check for FEN in various fields
        fen_fields = ['fen', 'position', 'board']
        fen_value = None
        
        for field in fen_fields:
            if field in example and example[field]:
                fen_value = example[field]
                break
        
        # Also check in text content
        if not fen_value:
            text = example.get('text', '')
            fen_match = re.search(r'FEN:\s*([^\s\n]+)', text)
            if fen_match:
                fen_value = fen_match.group(1)
        
        if fen_value:
            try:
                board = chess.Board(fen_value)
                # Additional FEN validation could go here
            except Exception as e:
                errors.append(f"Invalid FEN: {fen_value} - {e}")
        
        return errors
    
    def _validate_moves(self, example: Dict[str, Any]) -> List[str]:
        """Validate chess moves if present."""
        errors = []
        
        # Check for moves in various fields
        move_fields = ['move', 'best_move', 'moves', 'solution']
        moves_value = None
        
        for field in move_fields:
            if field in example and example[field]:
                moves_value = example[field]
                break
        
        # Also check in text content
        if not moves_value:
            text = example.get('text', '')
            move_match = re.search(r'move:\s*([a-h][1-8][a-h][1-8][qrbn]?)', text, re.IGNORECASE)
            if move_match:
                moves_value = move_match.group(1)
        
        if moves_value:
            try:
                if isinstance(moves_value, str):
                    moves = moves_value.split()
                else:
                    moves = [moves_value]
                
                for move in moves:
                    chess.Move.from_uci(move)
            except Exception as e:
                errors.append(f"Invalid move: {moves_value} - {e}")
        
        return errors
    
    def _calculate_quality_score(self, example: Dict[str, Any], chess_relevance: float) -> float:
        """Calculate quality score for an example."""
        score = 0.0
        
        # Base score from chess relevance
        score += chess_relevance * 4.0
        
        # Length score (prefer medium length)
        text = example.get('text', '')
        length = len(text)
        if 50 <= length <= 300:
            score += 2.0
        elif 20 <= length <= 500:
            score += 1.0
        
        # Structure score
        if 'conversations' in example and len(example['conversations']) >= 2:
            score += 1.0
        
        # Category score
        if 'category' in example and example['category']:
            score += 0.5
        
        # Difficulty score
        if 'difficulty' in example and example['difficulty'] in ['beginner', 'intermediate', 'advanced']:
            score += 0.5
        
        # Source score
        if 'source' in example and example['source']:
            score += 0.5
        
        return min(score, 10.0)
    
    def _calculate_statistics(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate statistics for valid examples."""
        if not examples:
            return {}
        
        stats = {
            'total_examples': len(examples),
            'categories': {},
            'difficulties': {},
            'sources': {},
            'length_distribution': {
                'questions': [],
                'answers': [],
                'total_text': []
            }
        }
        
        for example in examples:
            # Category distribution
            category = example.get('category', 'unknown')
            stats['categories'][category] = stats['categories'].get(category, 0) + 1
            
            # Difficulty distribution
            difficulty = example.get('difficulty', 'unknown')
            stats['difficulties'][difficulty] = stats['difficulties'].get(difficulty, 0) + 1
            
            # Source distribution
            source = example.get('source', 'unknown')
            stats['sources'][source] = stats['sources'].get(source, 0) + 1
            
            # Length distribution
            text = example.get('text', '')
            stats['length_distribution']['total_text'].append(len(text))
            
            # Extract question and answer lengths
            for conv in example.get('conversations', []):
                if conv.get('role') == 'user':
                    stats['length_distribution']['questions'].append(len(conv.get('content', '')))
                elif conv.get('role') == 'assistant':
                    stats['length_distribution']['answers'].append(len(conv.get('content', '')))
        
        # Calculate averages
        length_types = list(stats['length_distribution'].keys())
        for length_type in length_types:
            lengths = stats['length_distribution'][length_type]
            if lengths:
                stats['length_distribution'][f'{length_type}_avg'] = statistics.mean(lengths)
                stats['length_distribution'][f'{length_type}_min'] = min(lengths)
                stats['length_distribution'][f'{length_type}_max'] = max(lengths)
        
        return stats
    
    def _save_filtered_dataset(self, examples: List[Dict[str, Any]], output_file: Path):
        """Save filtered dataset to file."""
        print(f"💾 Saving {len(examples):,} valid examples to {output_file}")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for example in examples:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    def validate_multiple_datasets(self, dataset_files: List[Path]) -> Dict[str, Any]:
        """Validate multiple datasets and create comparison report."""
        print(f"🎯 Validating {len(dataset_files)} datasets...")
        
        all_results = {}
        
        for dataset_file in dataset_files:
            if dataset_file.exists():
                print(f"   📊 Validating: {dataset_file.name}")
                results = self.validate_dataset(dataset_file)
                all_results[dataset_file.name] = results
            else:
                print(f"   ❌ File not found: {dataset_file}")
        
        # Create comparison report
        comparison_report = {
            'validation_date': datetime.now().isoformat(),
            'datasets': all_results,
            'summary': self._create_comparison_summary(all_results)
        }
        
        # Save comparison report
        report_file = self.validation_dir / "dataset_comparison_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(comparison_report, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Multi-dataset validation complete!")
        print(f"   📁 Comparison report: {report_file}")
        
        return comparison_report
    
    def _create_comparison_summary(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary comparison of all datasets."""
        summary = {
            'total_datasets': len(all_results),
            'best_quality_score': 0.0,
            'best_dataset': None,
            'total_examples': 0,
            'total_valid_examples': 0,
            'average_quality': 0.0
        }
        
        quality_scores = []
        
        for dataset_name, results in all_results.items():
            if results:
                summary['total_examples'] += results.get('total_examples', 0)
                summary['total_valid_examples'] += results.get('valid_examples', 0)
                
                quality_scores_list = results.get('quality_scores', [])
                if quality_scores_list:
                    avg_quality = statistics.mean(quality_scores_list)
                    quality_scores.append(avg_quality)
                    
                    if avg_quality > summary['best_quality_score']:
                        summary['best_quality_score'] = avg_quality
                        summary['best_dataset'] = dataset_name
        
        if quality_scores:
            summary['average_quality'] = statistics.mean(quality_scores)
        
        return summary


def main():
    """Main entry point for data validation."""
    parser = argparse.ArgumentParser(description="Validate chess training datasets")
    parser.add_argument("--data_dir", default="data", help="Data directory path")
    parser.add_argument("--input", help="Input dataset file to validate")
    parser.add_argument("--output", help="Output file for filtered dataset")
    parser.add_argument("--validate_all", action="store_true", help="Validate all datasets in datasets/ directory")
    parser.add_argument("--min_quality", type=float, default=5.0, help="Minimum quality score threshold")
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    validator = ChessDataValidator(data_dir)
    
    # Update quality threshold
    validator.quality_thresholds['min_quality_score'] = args.min_quality
    
    if args.validate_all:
        # Validate all datasets
        datasets_dir = data_dir / "datasets"
        dataset_files = list(datasets_dir.glob("*.jsonl"))
        
        if dataset_files:
            validator.validate_multiple_datasets(dataset_files)
        else:
            print("❌ No .jsonl files found in datasets directory")
    
    elif args.input:
        # Validate single dataset
        input_file = Path(args.input)
        output_file = Path(args.output) if args.output else None
        
        validator.validate_dataset(input_file, output_file)
    
    else:
        print("❌ Please specify --input or --validate_all")
        parser.print_help()


if __name__ == "__main__":
    main()
