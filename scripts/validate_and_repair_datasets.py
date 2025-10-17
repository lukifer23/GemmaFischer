#!/usr/bin/env python3
"""
Dataset Validation and Repair Script

Validates all chess datasets, identifies issues, and repairs common problems.
Generates comprehensive reports on data quality and consistency.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import our data validator
try:
    from src.data.data_validator import create_data_validator, ChessDataValidator
except ImportError:
    # Fallback if the module structure is different
    import sys
    sys.path.append(str(Path(__file__).parent.parent / "src"))
    from data.data_validator import create_data_validator, ChessDataValidator


def validate_single_dataset(dataset_path: str, repair_mode: bool = False, output_dir: str = "reports/validation") -> Dict[str, Any]:
    """Validate a single dataset and optionally repair it."""
    validator = create_data_validator()

    logger.info(f"Processing dataset: {dataset_path}")

    # Validate the dataset
    report = validator.validate_dataset(dataset_path, repair_mode=repair_mode)

    # Save detailed report
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_name = Path(dataset_path).stem
    report_file = output_dir / f"{dataset_name}_validation_report.json"

    report_data = {
        "dataset_path": report.dataset_path,
        "total_records": report.total_records,
        "valid_records": report.valid_records,
        "invalid_records": report.invalid_records,
        "warnings": report.warnings,
        "errors": report.errors,
        "quality_score": report.quality_score,
        "format_issues": report.format_issues,
        "consistency_issues": report.consistency_issues,
        "repair_suggestions": report.repair_suggestions,
        "validation_results_sample": [
            {
                "rule_name": r.rule_name,
                "severity": r.severity,
                "message": r.message,
                "record_id": r.record_id
            }
            for r in report.validation_results[:50]  # First 50 results
        ]
    }

    with open(report_file, 'w') as f:
        json.dump(report_data, f, indent=2)

    logger.info(f"Validation report saved to {report_file}")
    logger.info(f"Quality score: {report.quality_score:.2%} ({report.valid_records}/{report.total_records} valid)")

    return report_data


def validate_all_datasets(datasets_dir: str = "data/standardized", repair_mode: bool = False,
                         output_dir: str = "reports/validation") -> Dict[str, Any]:
    """Validate all datasets in a directory."""
    datasets_dir = Path(datasets_dir)

    if not datasets_dir.exists():
        logger.error(f"Datasets directory not found: {datasets_dir}")
        return {"error": "Datasets directory not found"}

    # Find all JSONL files
    jsonl_files = list(datasets_dir.glob("*.jsonl"))

    if not jsonl_files:
        logger.warning(f"No JSONL files found in {datasets_dir}")
        return {"error": "No datasets found"}

    logger.info(f"Found {len(jsonl_files)} datasets to validate")

    all_reports = {}
    summary = {
        "total_datasets": len(jsonl_files),
        "total_records": 0,
        "total_valid_records": 0,
        "total_errors": 0,
        "total_warnings": 0,
        "average_quality_score": 0.0,
        "datasets": {},
        "common_issues": [],
        "recommendations": []
    }

    for dataset_path in jsonl_files:
        dataset_name = dataset_path.stem

        try:
            report_data = validate_single_dataset(str(dataset_path), repair_mode, output_dir)
            all_reports[dataset_name] = report_data

            # Update summary
            summary["total_records"] += report_data["total_records"]
            summary["total_valid_records"] += report_data["valid_records"]
            summary["total_errors"] += report_data["errors"]
            summary["total_warnings"] += report_data["warnings"]
            summary["datasets"][dataset_name] = {
                "quality_score": report_data["quality_score"],
                "total_records": report_data["total_records"],
                "errors": report_data["errors"],
                "warnings": report_data["warnings"]
            }

        except Exception as e:
            logger.error(f"Failed to validate {dataset_name}: {e}")
            all_reports[dataset_name] = {"error": str(e)}

    # Calculate average quality score
    if summary["total_records"] > 0:
        summary["average_quality_score"] = summary["total_valid_records"] / summary["total_records"]

    # Generate overall recommendations
    if summary["average_quality_score"] < 0.8:
        summary["recommendations"].append("Overall data quality is below threshold - review data sources")

    if summary["total_errors"] > summary["total_records"] * 0.1:  # More than 10% errors
        summary["recommendations"].append("High error rate across datasets - consider regenerating data")

    # Find common issues
    issue_counts = {}
    for report in all_reports.values():
        if "format_issues" in report:
            for issue in report["format_issues"] + report["consistency_issues"]:
                issue_counts[issue] = issue_counts.get(issue, 0) + 1

    summary["common_issues"] = [
        {"issue": issue, "count": count}
        for issue, count in issue_counts.items()
        if count >= len(jsonl_files) // 2  # Appears in at least half of datasets
    ]

    # Save overall summary
    summary_file = Path(output_dir) / "dataset_validation_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Overall validation complete. Average quality: {summary['average_quality_score']:.2%}")
    logger.info(f"Summary saved to {summary_file}")

    return summary


def repair_datasets(datasets_dir: str = "data/standardized", output_dir: str = "data/validated",
                   dry_run: bool = False) -> Dict[str, Any]:
    """Repair all datasets and save to new location."""
    datasets_dir = Path(datasets_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not datasets_dir.exists():
        logger.error(f"Datasets directory not found: {datasets_dir}")
        return {"error": "Datasets directory not found"}

    jsonl_files = list(datasets_dir.glob("*.jsonl"))

    logger.info(f"Repairing {len(jsonl_files)} datasets{' (dry run)' if dry_run else ''}")

    repair_results = {}

    for dataset_path in jsonl_files:
        dataset_name = dataset_path.stem
        output_path = output_dir / dataset_path.name

        try:
            validator = create_data_validator()
            if dry_run:
                # Just validate without repair
                report = validator.validate_dataset(str(dataset_path), repair_mode=False)
                repair_results[dataset_name] = {
                    "status": "dry_run",
                    "quality_score": report.quality_score,
                    "would_repair": report.errors > 0 or report.warnings > 0
                }
            else:
                # Actually repair the dataset
                report = validator.repair_dataset(str(dataset_path), str(output_path), dry_run=False)
                repair_results[dataset_name] = {
                    "status": "repaired",
                    "original_quality": 0.0,  # Would need to calculate this
                    "repaired_quality": report.quality_score,
                    "output_path": str(output_path)
                }

            logger.info(f"{'Would repair' if dry_run else 'Repaired'} {dataset_name}: "
                       f"Quality {repair_results[dataset_name].get('repaired_quality', repair_results[dataset_name].get('quality_score', 0)):.2%}")

        except Exception as e:
            logger.error(f"Failed to repair {dataset_name}: {e}")
            repair_results[dataset_name] = {"error": str(e)}

    # Save repair summary
    repair_summary_file = output_dir / "repair_summary.json"
    with open(repair_summary_file, 'w') as f:
        json.dump(repair_results, f, indent=2)

    logger.info(f"Repair operation complete. Summary saved to {repair_summary_file}")

    return repair_results


def standardize_dataset_formats(datasets_dir: str = "data/standardized",
                              output_dir: str = "data/standardized_v2") -> Dict[str, Any]:
    """Standardize all dataset formats to ensure consistency."""
    datasets_dir = Path(datasets_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not datasets_dir.exists():
        logger.error(f"Datasets directory not found: {datasets_dir}")
        return {"error": "Datasets directory not found"}

    jsonl_files = list(datasets_dir.glob("*.jsonl"))

    logger.info(f"Standardizing {len(jsonl_files)} datasets")

    standardization_results = {}

    for dataset_path in jsonl_files:
        dataset_name = dataset_path.stem
        output_path = output_dir / dataset_path.name

        try:
            validator = create_data_validator()
            report = validator.standardize_dataset_format(str(dataset_path), str(output_path))

            standardization_results[dataset_name] = {
                "status": "standardized",
                "output_path": str(output_path),
                "quality_score": report.quality_score,
                "total_records": report.total_records,
                "valid_records": report.valid_records
            }

            logger.info(f"Standardized {dataset_name}: Quality {report.quality_score:.2%}")

        except Exception as e:
            logger.error(f"Failed to standardize {dataset_name}: {e}")
            standardization_results[dataset_name] = {"error": str(e)}

    # Save standardization summary
    summary_file = output_dir / "standardization_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(standardization_results, f, indent=2)

    logger.info(f"Standardization complete. Summary saved to {summary_file}")

    return standardization_results


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(description="Validate and repair chess datasets")
    parser.add_argument("--datasets-dir", default="data/standardized",
                       help="Directory containing datasets to validate")
    parser.add_argument("--output-dir", default="reports/validation",
                       help="Directory for validation reports")
    parser.add_argument("--repair", action="store_true",
                       help="Enable repair mode")
    parser.add_argument("--repair-output-dir", default="data/validated",
                       help="Directory for repaired datasets")
    parser.add_argument("--standardize", action="store_true",
                       help="Standardize dataset formats")
    parser.add_argument("--standardize-output-dir", default="data/standardized_v2",
                       help="Directory for standardized datasets")
    parser.add_argument("--dry-run", action="store_true",
                       help="Dry run for repair operations")
    parser.add_argument("--single-dataset", help="Validate only a single dataset")
    parser.add_argument("--generate", action="store_true", help="Generate evaluation dataset (consolidated)")
    parser.add_argument("--gen-out", default="data/validation/eval_suite_auto.jsonl", help="Output path for generated eval data")

    args = parser.parse_args()

    if args.single_dataset:
        # Validate single dataset
        validate_single_dataset(args.single_dataset, args.repair, args.output_dir)
    else:
        if args.generate:
            try:
                from src.data.create_finetune_dataset import main as gen_main  # consolidated generator
                gen_main()
                print(f"Generated evaluation dataset to {args.gen_out}")
            except Exception as e:
                print(f"Dataset generation failed: {e}")
        # Validate all datasets
        summary = validate_all_datasets(args.datasets_dir, args.repair, args.output_dir)
        print(f"\nValidation Summary:")
        print(f"Total datasets: {summary['total_datasets']}")
        print(f"Total records: {summary['total_records']}")
        print(f"Average quality: {summary['average_quality_score']:.2%}")
        print(f"Total errors: {summary['total_errors']}")
        print(f"Total warnings: {summary['total_warnings']}")

        if summary['recommendations']:
            print(f"\nRecommendations:")
            for rec in summary['recommendations']:
                print(f"  - {rec}")

        if args.repair:
            repair_results = repair_datasets(args.datasets_dir, args.repair_output_dir, args.dry_run)
            print(f"\nRepair Summary:")
            successful_repairs = sum(1 for r in repair_results.values()
                                  if r.get('status') in ['repaired', 'dry_run'])
            print(f"Successfully processed: {successful_repairs}/{len(repair_results)} datasets")

        if args.standardize:
            std_results = standardize_dataset_formats(args.datasets_dir, args.standardize_output_dir)
            print(f"\nStandardization Summary:")
            successful_std = sum(1 for r in std_results.values()
                               if r.get('status') == 'standardized')
            print(f"Successfully standardized: {successful_std}/{len(std_results)} datasets")


if __name__ == "__main__":
    main()
