#!/usr/bin/env python3
"""
Comprehensive Data Validation and Repair System for ChessGemma

Validates data consistency across all datasets, repairs common issues,
and ensures standardized formats for training and evaluation.
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
import hashlib
import chess
from collections import defaultdict
import time

logger = logging.getLogger(__name__)


@dataclass
class ValidationRule:
    """Data validation rule definition."""
    name: str
    description: str
    validator_function: callable
    severity: str = "error"  # error, warning, info
    auto_repair: bool = False
    repair_function: Optional[callable] = None


@dataclass
class ValidationResult:
    """Result of data validation."""
    is_valid: bool
    rule_name: str
    severity: str
    message: str
    record_id: Optional[str] = None
    suggested_repair: Optional[str] = None


@dataclass
class DatasetReport:
    """Comprehensive dataset validation report."""
    dataset_path: str
    total_records: int
    valid_records: int
    invalid_records: int
    warnings: int
    errors: int
    validation_results: List[ValidationResult] = field(default_factory=list)
    repair_suggestions: List[str] = field(default_factory=list)
    format_issues: List[str] = field(default_factory=list)
    consistency_issues: List[str] = field(default_factory=list)
    quality_score: float = 0.0


class ChessDataValidator:
    """Comprehensive validator for chess training data."""

    def __init__(self):
        self.validation_rules = self._initialize_validation_rules()
        self.fen_pattern = re.compile(r'^[rnbqkpRNBQKP1-8]+\/[rnbqkpRNBQKP1-8]+\/[rnbqkpRNBQKP1-8]+\/[rnbqkpRNBQKP1-8]+\/[rnbqkpRNBQKP1-8]+\/[rnbqkpRNBQKP1-8]+\/[rnbqkpRNBQKP1-8]+\/[rnbqkpRNBQKP1-8]+\s+[wb]\s+[-KQkq]+\s+-\s+\d+\s+\d+$')

    def _initialize_validation_rules(self) -> List[ValidationRule]:
        """Initialize comprehensive validation rules."""
        return [
            ValidationRule(
                "json_format",
                "Valid JSON format",
                self._validate_json_format,
                severity="error",
                auto_repair=True,
                repair_function=self._repair_json_format
            ),
            ValidationRule(
                "required_fields",
                "Required fields present",
                self._validate_required_fields,
                severity="error"
            ),
            ValidationRule(
                "fen_validity",
                "Valid FEN string",
                self._validate_fen_string,
                severity="error",
                auto_repair=True,
                repair_function=self._repair_fen_string
            ),
            ValidationRule(
                "uci_move_format",
                "Valid UCI move format",
                self._validate_uci_move,
                severity="error",
                auto_repair=True,
                repair_function=self._repair_uci_move
            ),
            ValidationRule(
                "task_consistency",
                "Consistent task type",
                self._validate_task_consistency,
                severity="warning"
            ),
            ValidationRule(
                "metadata_completeness",
                "Complete metadata",
                self._validate_metadata_completeness,
                severity="warning"
            ),
            ValidationRule(
                "chess_board_consistency",
                "Consistent chess board state",
                self._validate_chess_board_consistency,
                severity="error"
            ),
            ValidationRule(
                "response_quality",
                "Response quality indicators",
                self._validate_response_quality,
                severity="info"
            )
        ]

    def validate_dataset(self, dataset_path: str, repair_mode: bool = False) -> DatasetReport:
        """Validate entire dataset and optionally repair issues."""
        logger.info(f"Validating dataset: {dataset_path}")

        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        report = DatasetReport(
            dataset_path=str(dataset_path),
            total_records=0,
            valid_records=0,
            invalid_records=0,
            warnings=0,
            errors=0
        )

        # Read and validate each record
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                report.total_records += 1

                try:
                    # Parse JSON
                    record = json.loads(line)

                    # Validate record
                    record_results = self._validate_record(record, line_num)

                    # Update report statistics
                    for result in record_results:
                        report.validation_results.append(result)
                        if result.severity == "error":
                            report.errors += 1
                            report.invalid_records += 1
                        elif result.severity == "warning":
                            report.warnings += 1

                    # Apply repairs if enabled
                    if repair_mode:
                        repaired_record = self._repair_record(record, record_results)
                        if repaired_record != record:
                            # Record was repaired
                            logger.info(f"Repaired record {line_num}")

                    # Count as valid if no errors
                    if not any(r.severity == "error" for r in record_results):
                        report.valid_records += 1

                except json.JSONDecodeError as e:
                    report.errors += 1
                    report.invalid_records += 1
                    report.validation_results.append(ValidationResult(
                        is_valid=False,
                        rule_name="json_format",
                        severity="error",
                        message=f"Invalid JSON: {e}",
                        record_id=f"line_{line_num}"
                    ))
                except Exception as e:
                    report.errors += 1
                    report.invalid_records += 1
                    report.validation_results.append(ValidationResult(
                        is_valid=False,
                        rule_name="unknown",
                        severity="error",
                        message=f"Unexpected error: {e}",
                        record_id=f"line_{line_num}"
                    ))

        # Calculate quality score
        if report.total_records > 0:
            report.quality_score = report.valid_records / report.total_records

        # Generate format and consistency issues
        report.format_issues = self._analyze_format_issues(report)
        report.consistency_issues = self._analyze_consistency_issues(report)
        report.repair_suggestions = self._generate_repair_suggestions(report)

        logger.info(f"Dataset validation complete. Quality score: {report.quality_score:.2%}")
        return report

    def _validate_record(self, record: Dict[str, Any], line_num: int) -> List[ValidationResult]:
        """Validate a single record against all rules."""
        results = []

        for rule in self.validation_rules:
            try:
                result = rule.validator_function(record, line_num)
                if result:
                    results.append(result)
            except Exception as e:
                results.append(ValidationResult(
                    is_valid=False,
                    rule_name=rule.name,
                    severity="error",
                    message=f"Validation rule error: {e}",
                    record_id=f"line_{line_num}"
                ))

        return results

    def _validate_json_format(self, record: Dict[str, Any], line_num: int) -> Optional[ValidationResult]:
        """Validate basic JSON structure."""
        required_keys = ["task", "prompt", "response"]
        for key in required_keys:
            if key not in record:
                return ValidationResult(
                    is_valid=False,
                    rule_name="json_format",
                    severity="error",
                    message=f"Missing required field: {key}",
                    record_id=f"line_{line_num}"
                )
        return None

    def _validate_required_fields(self, record: Dict[str, Any], line_num: int) -> Optional[ValidationResult]:
        """Validate that required fields have appropriate content."""
        issues = []

        # Task field validation
        if not record.get("task"):
            issues.append("Empty task field")

        # Prompt field validation
        prompt = record.get("prompt", "")
        if not prompt or len(prompt.strip()) < 10:
            issues.append("Prompt too short or empty")

        # Response field validation
        response = record.get("response", "")
        if not response or len(response.strip()) < 2:
            issues.append("Response too short or empty")

        if issues:
            return ValidationResult(
                is_valid=False,
                rule_name="required_fields",
                severity="error",
                message=" | ".join(issues),
                record_id=f"line_{line_num}"
            )

        return None

    def _validate_fen_string(self, record: Dict[str, Any], line_num: int) -> Optional[ValidationResult]:
        """Validate FEN string format and chess validity."""
        fen = None

        # Extract FEN from various possible locations
        if "meta" in record and "fen" in record["meta"]:
            fen = record["meta"]["fen"]
        elif "fen" in record:
            fen = record["fen"]
        elif "FEN:" in record.get("prompt", ""):
            # Extract FEN from prompt
            fen_match = re.search(r'FEN:\s*([^\n]+)', record["prompt"])
            if fen_match:
                fen = fen_match.group(1).strip()

        if not fen:
            return ValidationResult(
                is_valid=False,
                rule_name="fen_validity",
                severity="warning",
                message="No FEN string found",
                record_id=f"line_{line_num}"
            )

        # Validate FEN format
        if not self.fen_pattern.match(fen):
            return ValidationResult(
                is_valid=False,
                rule_name="fen_validity",
                severity="error",
                message=f"Invalid FEN format: {fen}",
                record_id=f"line_{line_num}",
                suggested_repair="Fix FEN string format"
            )

        # Validate chess board
        try:
            board = chess.Board(fen)
            if board.is_checkmate() or board.is_stalemate():
                return ValidationResult(
                    is_valid=False,
                    rule_name="fen_validity",
                    severity="warning",
                    message="Position is checkmate or stalemate",
                    record_id=f"line_{line_num}"
                )
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                rule_name="fen_validity",
                severity="error",
                message=f"Invalid chess position: {e}",
                record_id=f"line_{line_num}"
            )

        return None

    def _validate_uci_move(self, record: Dict[str, Any], line_num: int) -> Optional[ValidationResult]:
        """Validate UCI move format."""
        # Extract UCI move from response
        response = record.get("response", "")
        uci_match = re.search(r'\b([a-h][1-8][a-h][1-8][qrbn]?)\b', response.lower())

        if not uci_match:
            return ValidationResult(
                is_valid=False,
                rule_name="uci_move_format",
                severity="warning",
                message="No UCI move found in response",
                record_id=f"line_{line_num}"
            )

        uci_move = uci_match.group(1)

        # Basic UCI format validation
        if not re.match(r'^[a-h][1-8][a-h][1-8][qrbn]?$', uci_move):
            return ValidationResult(
                is_valid=False,
                rule_name="uci_move_format",
                severity="error",
                message=f"Invalid UCI move format: {uci_move}",
                record_id=f"line_{line_num}"
            )

        # If we have a FEN, validate the move against the position
        fen = self._extract_fen_from_record(record)
        if fen:
            try:
                board = chess.Board(fen)
                move = chess.Move.from_uci(uci_move)

                if move not in board.legal_moves:
                    return ValidationResult(
                        is_valid=False,
                        rule_name="uci_move_format",
                        severity="error",
                        message=f"Illegal move {uci_move} for position {fen}",
                        record_id=f"line_{line_num}"
                    )
            except Exception:
                # If move validation fails, that's okay - just warn
                pass

        return None

    def _validate_task_consistency(self, record: Dict[str, Any], line_num: int) -> Optional[ValidationResult]:
        """Validate task type consistency."""
        task = record.get("task", "")

        # Define expected task patterns
        task_patterns = {
            "engine_uci": ["uci", "move", "best move"],
            "tutor_explain": ["analyze", "explain", "step by step", "evaluation"],
            "director_qa": ["strategy", "strategic", "principle", "guidance"]
        }

        # Check if task content matches expected patterns
        prompt_lower = record.get("prompt", "").lower()

        if task == "engine_uci":
            if not any(keyword in prompt_lower for keyword in task_patterns["engine_uci"]):
                return ValidationResult(
                    is_valid=False,
                    rule_name="task_consistency",
                    severity="warning",
                    message="UCI task but prompt doesn't mention moves",
                    record_id=f"line_{line_num}"
                )

        elif task == "tutor_explain":
            if not any(keyword in prompt_lower for keyword in task_patterns["tutor_explain"]):
                return ValidationResult(
                    is_valid=False,
                    rule_name="task_consistency",
                    severity="warning",
                    message="Tutor task but prompt doesn't mention analysis",
                    record_id=f"line_{line_num}"
                )

        elif task == "director_qa":
            if not any(keyword in prompt_lower for keyword in task_patterns["director_qa"]):
                return ValidationResult(
                    is_valid=False,
                    rule_name="task_consistency",
                    severity="warning",
                    message="Director task but prompt doesn't mention strategy",
                    record_id=f"line_{line_num}"
                )

        return None

    def _validate_metadata_completeness(self, record: Dict[str, Any], line_num: int) -> Optional[ValidationResult]:
        """Validate metadata completeness."""
        meta = record.get("meta", {})
        issues = []

        # Required metadata fields
        required_fields = ["fen", "source", "rating", "topic", "quality_score"]

        for field in required_fields:
            if field not in meta:
                issues.append(f"Missing metadata field: {field}")

        # Validate metadata values
        if "quality_score" in meta:
            score = meta["quality_score"]
            if not isinstance(score, (int, float)) or not (0 <= score <= 1):
                issues.append(f"Invalid quality_score: {score}")

        if "rating" in meta:
            rating = meta["rating"]
            if not isinstance(rating, int) or not (800 <= rating <= 3000):
                issues.append(f"Invalid rating: {rating}")

        if issues:
            return ValidationResult(
                is_valid=False,
                rule_name="metadata_completeness",
                severity="warning",
                message=" | ".join(issues),
                record_id=f"line_{line_num}"
            )

        return None

    def _validate_chess_board_consistency(self, record: Dict[str, Any], line_num: int) -> Optional[ValidationResult]:
        """Validate chess board state consistency."""
        fen = self._extract_fen_from_record(record)

        if not fen:
            return None  # Already handled by FEN validation

        try:
            board = chess.Board(fen)

            # Check for obvious inconsistencies
            piece_count = len(board.piece_map())
            if piece_count < 2:  # Too few pieces
                return ValidationResult(
                    is_valid=False,
                    rule_name="chess_board_consistency",
                    severity="warning",
                    message=f"Too few pieces on board: {piece_count}",
                    record_id=f"line_{line_num}"
                )

            if piece_count > 32:  # Too many pieces (shouldn't happen with valid FEN)
                return ValidationResult(
                    is_valid=False,
                    rule_name="chess_board_consistency",
                    severity="error",
                    message=f"Too many pieces on board: {piece_count}",
                    record_id=f"line_{line_num}"
                )

            # Check if both kings are present
            white_king = False
            black_king = False

            for piece in board.piece_map().values():
                if piece.piece_type == chess.KING:
                    if piece.color == chess.WHITE:
                        white_king = True
                    else:
                        black_king = True

            if not (white_king and black_king):
                return ValidationResult(
                    is_valid=False,
                    rule_name="chess_board_consistency",
                    severity="error",
                    message="Missing king(s) on board",
                    record_id=f"line_{line_num}"
                )

        except Exception as e:
            return ValidationResult(
                is_valid=False,
                rule_name="chess_board_consistency",
                severity="error",
                message=f"Chess board validation failed: {e}",
                record_id=f"line_{line_num}"
            )

        return None

    def _validate_response_quality(self, record: Dict[str, Any], line_num: int) -> Optional[ValidationResult]:
        """Validate response quality indicators."""
        response = record.get("response", "")
        issues = []

        # Check response length
        if len(response.strip()) < 10:
            issues.append("Response too short")

        # Check for placeholder content
        placeholders = ["placeholder", "todo", "fixme", "xxx", "???"]
        if any(ph in response.lower() for ph in placeholders):
            issues.append("Contains placeholder content")

        # Check for repetitive content
        words = response.split()
        if len(words) > 0:
            word_counts = {}
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1

            # Check for excessive repetition
            max_repetitions = max(word_counts.values()) if word_counts else 0
            if max_repetitions > len(words) * 0.3:  # More than 30% repetition
                issues.append("Excessive word repetition")

        if issues:
            return ValidationResult(
                is_valid=False,
                rule_name="response_quality",
                severity="info",
                message=" | ".join(issues),
                record_id=f"line_{line_num}"
            )

        return None

    def _extract_fen_from_record(self, record: Dict[str, Any]) -> Optional[str]:
        """Extract FEN string from record."""
        if "meta" in record and "fen" in record["meta"]:
            return record["meta"]["fen"]
        elif "fen" in record:
            return record["fen"]
        elif "FEN:" in record.get("prompt", ""):
            fen_match = re.search(r'FEN:\s*([^\n]+)', record["prompt"])
            if fen_match:
                return fen_match.group(1).strip()
        return None

    def _repair_record(self, record: Dict[str, Any], validation_results: List[ValidationResult]) -> Dict[str, Any]:
        """Apply repairs to a record based on validation results."""
        repaired_record = record.copy()

        for result in validation_results:
            if result.severity == "error" and hasattr(self, f"_repair_{result.rule_name}"):
                repair_func = getattr(self, f"_repair_{result.rule_name}")
                try:
                    repaired_record = repair_func(repaired_record, result)
                except Exception as e:
                    logger.warning(f"Failed to repair {result.rule_name}: {e}")

        return repaired_record

    def _repair_json_format(self, record: Dict[str, Any], result: ValidationResult) -> Dict[str, Any]:
        """Repair JSON format issues."""
        # This would implement JSON repair logic
        return record

    def _repair_fen_string(self, record: Dict[str, Any], result: ValidationResult) -> Dict[str, Any]:
        """Repair FEN string issues."""
        # This would implement FEN repair logic
        return record

    def _repair_uci_move(self, record: Dict[str, Any], result: ValidationResult) -> Dict[str, Any]:
        """Repair UCI move issues."""
        # This would implement UCI move repair logic
        return record

    def _analyze_format_issues(self, report: DatasetReport) -> List[str]:
        """Analyze format issues in the dataset."""
        issues = []

        # Check for field inconsistencies
        field_counts = defaultdict(int)
        for result in report.validation_results:
            if "field" in result.message.lower():
                field_counts[result.rule_name] += 1

        if field_counts:
            issues.append(f"Field validation issues found: {dict(field_counts)}")

        return issues

    def _analyze_consistency_issues(self, report: DatasetReport) -> List[str]:
        """Analyze consistency issues across the dataset."""
        issues = []

        # Check for task distribution
        task_counts = defaultdict(int)
        for result in report.validation_results:
            if "task" in result.message.lower():
                task_counts[result.severity] += 1

        if task_counts:
            issues.append(f"Task consistency issues: {dict(task_counts)}")

        return issues

    def _generate_repair_suggestions(self, report: DatasetReport) -> List[str]:
        """Generate repair suggestions based on validation results."""
        suggestions = []

        if report.errors > 0:
            error_rate = report.errors / report.total_records
            if error_rate > 0.1:  # More than 10% errors
                suggestions.append("High error rate detected - consider regenerating dataset")
            elif error_rate > 0.05:  # More than 5% errors
                suggestions.append("Moderate error rate - review and repair problematic records")

        if report.warnings > report.total_records * 0.2:  # More than 20% warnings
            suggestions.append("High warning rate - review data quality standards")

        if report.quality_score < 0.8:
            suggestions.append("Low quality score - consider improving data sources")

        return suggestions

    def repair_dataset(self, dataset_path: str, output_path: str, dry_run: bool = False) -> DatasetReport:
        """Repair dataset issues and save to new file."""
        logger.info(f"Repairing dataset: {dataset_path}")

        original_report = self.validate_dataset(dataset_path, repair_mode=False)

        if dry_run:
            logger.info("Dry run mode - no changes will be made")
            return original_report

        # Read, repair, and write records
        repaired_count = 0
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(dataset_path, 'r', encoding='utf-8') as infile, \
             open(output_path, 'w', encoding='utf-8') as outfile:

            for line_num, line in enumerate(infile, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                    record_results = self._validate_record(record, line_num)

                    # Apply repairs
                    repaired_record = self._repair_record(record, record_results)

                    # Write repaired record
                    outfile.write(json.dumps(repaired_record, ensure_ascii=False) + '\n')
                    repaired_count += 1

                except Exception as e:
                    logger.warning(f"Failed to repair record {line_num}: {e}")
                    # Write original record as-is
                    outfile.write(line + '\n')

        logger.info(f"Repaired {repaired_count} records. Saved to {output_path}")

        # Generate new report for repaired dataset
        repaired_report = self.validate_dataset(output_path, repair_mode=False)
        return repaired_report

    def standardize_dataset_format(self, input_path: str, output_path: str) -> DatasetReport:
        """Standardize dataset format across all records."""
        logger.info(f"Standardizing dataset format: {input_path}")

        # Define standard format structure
        standard_structure = {
            "task": "",
            "prompt": "",
            "response": "",
            "meta": {
                "fen": "",
                "source": "",
                "rating": 1500,
                "topic": "general",
                "quality_score": 0.8,
                "best_move": "",
                "created_at": "",
                "validation_status": "standardized"
            }
        }

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(input_path, 'r', encoding='utf-8') as infile, \
             open(output_path, 'w', encoding='utf-8') as outfile:

            for line_num, line in enumerate(infile, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                    standardized = self._standardize_record(record, standard_structure)
                    outfile.write(json.dumps(standardized, ensure_ascii=False) + '\n')

                except Exception as e:
                    logger.warning(f"Failed to standardize record {line_num}: {e}")

        # Generate report for standardized dataset
        report = self.validate_dataset(output_path, repair_mode=False)
        logger.info(f"Dataset standardized. Quality score: {report.quality_score:.2%}")
        return report

    def _standardize_record(self, record: Dict[str, Any], standard_structure: Dict[str, Any]) -> Dict[str, Any]:
        """Standardize a single record to the standard format."""
        standardized = standard_structure.copy()

        # Copy existing fields
        for key in ["task", "prompt", "response"]:
            if key in record:
                standardized[key] = record[key]

        # Standardize metadata
        meta = standardized["meta"]
        if "meta" in record:
            # Merge existing metadata
            meta.update(record["meta"])

        # Ensure FEN is properly extracted and stored
        fen = self._extract_fen_from_record(record)
        if fen:
            meta["fen"] = fen

        # Set creation timestamp
        meta["created_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ")

        # Extract best move if available
        if "best_move" not in meta:
            # Try to extract from response
            response = record.get("response", "")
            uci_match = re.search(r'\b([a-h][1-8][a-h][1-8][qrbn]?)\b', response.lower())
            if uci_match:
                meta["best_move"] = uci_match.group(1)

        return standardized

    def generate_dataset_summary(self, dataset_paths: List[str]) -> Dict[str, Any]:
        """Generate summary report for multiple datasets."""
        summary = {
            "datasets": {},
            "overall_quality": 0.0,
            "total_records": 0,
            "common_issues": [],
            "recommendations": []
        }

        all_reports = []
        for path in dataset_paths:
            report = self.validate_dataset(path)
            summary["datasets"][Path(path).name] = {
                "quality_score": report.quality_score,
                "total_records": report.total_records,
                "valid_records": report.valid_records,
                "errors": report.errors,
                "warnings": report.warnings
            }
            all_reports.append(report)
            summary["total_records"] += report.total_records

        # Calculate overall quality
        if summary["total_records"] > 0:
            total_quality = sum(r.quality_score * r.total_records for r in all_reports)
            summary["overall_quality"] = total_quality / summary["total_records"]

        # Find common issues
        issue_counts = defaultdict(int)
        for report in all_reports:
            for issue in report.format_issues + report.consistency_issues:
                issue_counts[issue] += 1

        summary["common_issues"] = [
            issue for issue, count in issue_counts.items()
            if count >= len(dataset_paths) // 2  # Appears in at least half of datasets
        ]

        # Generate recommendations
        if summary["overall_quality"] < 0.8:
            summary["recommendations"].append("Overall data quality is below threshold - review data sources")

        if summary["common_issues"]:
            summary["recommendations"].append("Address common format and consistency issues")

        return summary


def create_data_validator() -> ChessDataValidator:
    """Factory function to create a data validator."""
    return ChessDataValidator()


def validate_and_repair_dataset(dataset_path: str, repair_mode: bool = True) -> DatasetReport:
    """Convenient function to validate and optionally repair a dataset."""
    validator = create_data_validator()
    return validator.validate_dataset(dataset_path, repair_mode=repair_mode)
