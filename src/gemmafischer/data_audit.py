from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import chess

from .domain import EngineEvidence, RatingBucket
from .runtime import (
    CLAIM_SELECTION_CONTRACT_VERSION,
    CLAIM_SELECTION_SYSTEM_PROMPT,
    claim_selection_prompt,
    parse_claim_selection,
)

UCI_PATTERN = re.compile(r"^[a-h][1-8][a-h][1-8][qrbn]?$", re.IGNORECASE)
MINIMUM_TRAINING_RECORDS = 10_000
MINIMUM_VALIDATION_RECORDS = 1_000
MINIMUM_EVALUATION_RECORDS = 1_000
FEN_PATTERN = re.compile(r"(?:FEN|Position):\s*([^\n]+)", re.IGNORECASE)
REQUIRED_META_FIELDS = (
    "source", "source_item_id", "lineage", "license", "split", "transformation",
    "setup_move", "solution_move", "move_sequence", "evidence_contract_version",
    "model_contract_version", "engine_binary_sha256", "engine_node_budget",
)
BLOCKING_TOTALS = (
    "malformed_records", "invalid_fens", "illegal_best_moves", "missing_fens",
    "missing_best_moves", "missing_provenance", "missing_license", "missing_lineage",
    "missing_required_metadata", "split_mismatches", "unsupported_tasks",
    "invalid_model_contracts", "duplicates_within_file", "duplicates_across_files",
    "duplicate_semantic_positions",
)


def audit_data(
    training_paths: list[Path],
    evaluation_paths: list[Path],
    output_path: Path,
    *,
    validation_paths: list[Path] | None = None,
    minimum_training_records: int = MINIMUM_TRAINING_RECORDS,
    minimum_validation_records: int = MINIMUM_VALIDATION_RECORDS,
    minimum_evaluation_records: int = MINIMUM_EVALUATION_RECORDS,
    enforce_model_contract: bool = True,
) -> dict[str, Any]:
    """Audit immutable train/validation/final-test partitions and fail closed."""
    validation_paths = validation_paths or []
    training = _scan(training_paths, "train", enforce_model_contract)
    validation = _scan(validation_paths, "validation", enforce_model_contract)
    evaluation = _scan(evaluation_paths, "final_test", enforce_model_contract)
    internals = {
        "train": _pop_internals(training),
        "validation": _pop_internals(validation),
        "final_test": _pop_internals(evaluation),
    }
    position_overlaps = _pairwise_overlaps(internals, "semantic_positions")
    lineage_overlaps = _pairwise_overlaps(internals, "lineages")
    labels: defaultdict[str, set[str]] = defaultdict(set)
    for values in internals.values():
        for position, moves in values["labels"].items():
            labels[position].update(moves)
    conflicts: list[dict[str, Any]] = [
        {"semantic_position": position, "best_moves": sorted(moves)}
        for position, moves in labels.items()
        if len(moves) > 1
    ]
    conflicts.sort(key=lambda item: str(item["semantic_position"]))
    blocked = _blocked(
        training, validation, evaluation, conflicts, position_overlaps, lineage_overlaps,
        minimum_training_records, minimum_validation_records, minimum_evaluation_records,
    )
    payload = {
        "schema_version": "2.0",
        "status": "blocked" if blocked else "passed",
        "generated_at": datetime.now(UTC).isoformat(),
        "training": training,
        "validation": validation,
        "evaluation": evaluation,
        "cross_dataset": {
            "conflicting_best_move_positions": len(conflicts),
            "conflict_samples": conflicts[:25],
            "semantic_position_overlap": sum(item["count"] for item in position_overlaps),
            "semantic_position_overlap_by_pair": position_overlaps,
            "lineage_overlap": sum(item["count"] for item in lineage_overlaps),
            "lineage_overlap_by_pair": lineage_overlaps,
            "conflicting_best_move_fens": len(conflicts),
            "train_evaluation_fen_overlap": next(
                (item["count"] for item in position_overlaps
                 if item["partitions"] == ["train", "final_test"]), 0,
            ),
        },
        "gate": {
            "ready_for_training": not blocked,
            "requirements": {
                "minimum_training_records": minimum_training_records,
                "minimum_validation_records": minimum_validation_records,
                "minimum_evaluation_records": minimum_evaluation_records,
                **{name: 0 for name in BLOCKING_TOTALS},
                "conflicting_best_move_positions": 0,
                "semantic_position_overlap": 0,
                "lineage_overlap": 0,
            },
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output_path)
    return payload


def _blocked(
    training: dict[str, Any], validation: dict[str, Any], evaluation: dict[str, Any],
    conflicts: list[dict[str, Any]], position_overlaps: list[dict[str, Any]],
    lineage_overlaps: list[dict[str, Any]], minimum_training_records: int,
    minimum_validation_records: int, minimum_evaluation_records: int,
) -> bool:
    partitions = (training, validation, evaluation)
    return any((
        training["totals"].get("records", 0) < minimum_training_records,
        validation["totals"].get("records", 0) < minimum_validation_records,
        evaluation["totals"].get("records", 0) < minimum_evaluation_records,
        any(partition["totals"].get(name, 0)
            for partition in partitions for name in BLOCKING_TOTALS),
        len(conflicts), sum(item["count"] for item in position_overlaps),
        sum(item["count"] for item in lineage_overlaps),
    ))


def _scan(paths: list[Path], expected_split: str, enforce_model_contract: bool) -> dict[str, Any]:
    totals: Counter[str] = Counter()
    files: list[dict[str, Any]] = []
    all_hashes: set[str] = set()
    semantic_positions: set[str] = set()
    lineages: set[str] = set()
    labels: defaultdict[str, set[str]] = defaultdict(set)
    issue_samples: list[dict[str, Any]] = []
    for path in paths:
        file_counts: Counter[str] = Counter()
        file_hashes: set[str] = set()
        prior_file_hashes = set(all_hashes)
        sources: Counter[str] = Counter()
        tasks: Counter[str] = Counter()
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                file_counts["records"] += 1
                record = _decode_record(line, path, line_number, file_counts, issue_samples)
                if record is None:
                    continue
                digest = hashlib.sha256(
                    json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest()
                if digest in file_hashes:
                    file_counts["duplicates_within_file"] += 1
                if digest in prior_file_hashes:
                    file_counts["duplicates_across_files"] += 1
                file_hashes.add(digest)
                meta = _metadata(record)
                task = str(record.get("task") or "unknown")
                tasks[task] += 1
                source = str(meta.get("source") or "")
                if source:
                    sources[source] += 1
                else:
                    file_counts["missing_provenance"] += 1
                if not meta.get("license") and not record.get("license"):
                    file_counts["missing_license"] += 1
                lineage = str(meta.get("lineage") or "")
                if lineage:
                    lineages.add(lineage)
                else:
                    file_counts["missing_lineage"] += 1
                if enforce_model_contract:
                    missing = [field for field in REQUIRED_META_FIELDS if not meta.get(field)]
                    if missing:
                        file_counts["missing_required_metadata"] += 1
                        _sample(
                            issue_samples,
                            path,
                            line_number,
                            "missing_metadata",
                            ",".join(missing),
                        )
                    if meta.get("split") != expected_split:
                        file_counts["split_mismatches"] += 1
                    if task != CLAIM_SELECTION_CONTRACT_VERSION:
                        file_counts["unsupported_tasks"] += 1
                fen = _extract_fen(record, meta)
                if not fen:
                    file_counts["missing_fens"] += 1
                    continue
                file_counts["records_with_fen"] += 1
                try:
                    board = chess.Board(fen)
                    if not board.is_valid():
                        raise ValueError(f"status={board.status()}")
                except ValueError as exc:
                    file_counts["invalid_fens"] += 1
                    _sample(issue_samples, path, line_number, "invalid_fen", str(exc))
                    continue
                semantic = _semantic_position(board)
                if semantic in semantic_positions:
                    file_counts["duplicate_semantic_positions"] += 1
                semantic_positions.add(semantic)
                move = _extract_best_move(record, meta)
                if not move:
                    file_counts["missing_best_moves"] += 1
                else:
                    file_counts["records_with_best_move"] += 1
                    try:
                        parsed_move = chess.Move.from_uci(move.lower())
                    except ValueError:
                        parsed_move = None
                    if parsed_move is None or parsed_move not in board.legal_moves:
                        file_counts["illegal_best_moves"] += 1
                        _sample(issue_samples, path, line_number, "illegal_best_move", move)
                    else:
                        labels[semantic].add(move.lower())
                if enforce_model_contract and not _valid_model_contract(record):
                    file_counts["invalid_model_contracts"] += 1
                    _sample(issue_samples, path, line_number, "invalid_model_contract", task)
        all_hashes.update(file_hashes)
        totals.update(file_counts)
        files.append({
            "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            **dict(file_counts), "tasks": dict(tasks.most_common()),
            "sources": dict(sources.most_common(10)),
        })
    return {
        "files": files, "totals": dict(totals), "unique_records": len(all_hashes),
        "unique_semantic_positions": len(semantic_positions),
        "unique_lineages": len(lineages), "issue_samples": issue_samples,
        "_semantic_positions": semantic_positions, "_lineages": lineages, "_labels": labels,
    }


def _decode_record(
    line: str, path: Path, line_number: int, counts: Counter[str],
    samples: list[dict[str, Any]],
) -> dict[str, Any] | None:
    try:
        value = json.loads(line)
    except json.JSONDecodeError as exc:
        counts["malformed_records"] += 1
        _sample(samples, path, line_number, "malformed_json", str(exc))
        return None
    if not isinstance(value, dict):
        counts["malformed_records"] += 1
        _sample(samples, path, line_number, "record_not_object", type(value).__name__)
        return None
    return value


def _valid_model_contract(record: dict[str, Any]) -> bool:
    if record.get("task") != CLAIM_SELECTION_CONTRACT_VERSION:
        return False
    if record.get("system_prompt") != CLAIM_SELECTION_SYSTEM_PROMPT:
        return False
    raw_input, response, meta = record.get("input"), record.get("response"), _metadata(record)
    if not isinstance(raw_input, dict) or not isinstance(response, str):
        return False
    try:
        evidence = EngineEvidence.model_validate(raw_input)
        rating = RatingBucket(str(meta.get("rating_bucket")))
        if record.get("prompt") != claim_selection_prompt(evidence, rating):
            return False
        parsed = parse_claim_selection(response, evidence)
    except (TypeError, ValueError):
        return False
    expected = [claim.model_dump(mode="json") for claim in parsed.claims]
    return (
        2 <= len(parsed.claims) <= 5
        and not parsed.removed_claim_codes
        and record.get("target") == expected
    )


def _semantic_position(board: chess.Board) -> str:
    return " ".join(board.fen(en_passant="legal").split()[:4])


def _pairwise_overlaps(internals: dict[str, dict[str, Any]], key: str) -> list[dict[str, Any]]:
    names = list(internals)
    overlaps: list[dict[str, Any]] = []
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            shared = sorted(internals[left][key] & internals[right][key])
            overlaps.append(
                {"partitions": [left, right], "count": len(shared), "samples": shared[:25]}
            )
    return overlaps


def _pop_internals(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "semantic_positions": payload.pop("_semantic_positions"),
        "lineages": payload.pop("_lineages"),
        "labels": payload.pop("_labels"),
    }


def _metadata(record: dict[str, Any]) -> dict[str, Any]:
    raw = record.get("meta")
    return {str(key): value for key, value in raw.items()} if isinstance(raw, dict) else {}


def _extract_fen(record: dict[str, Any], meta: dict[str, Any]) -> str | None:
    if isinstance(meta.get("fen"), str):
        return str(meta["fen"]).strip()
    raw_input = record.get("input")
    if isinstance(raw_input, dict) and isinstance(raw_input.get("fen"), str):
        return str(raw_input["fen"]).strip()
    for key in ("prompt", "question"):
        value = record.get(key)
        if isinstance(value, str) and (match := FEN_PATTERN.search(value)):
            return match.group(1).strip()
    return None


def _extract_best_move(record: dict[str, Any], meta: dict[str, Any]) -> str | None:
    for key in ("solution_move", "best_move"):
        value = meta.get(key)
        if isinstance(value, str) and UCI_PATTERN.fullmatch(value.strip()):
            return value.strip()
    response = record.get("response")
    if isinstance(response, str) and UCI_PATTERN.fullmatch(response.strip()):
        return response.strip()
    return None


def _sample(
    samples: list[dict[str, Any]], path: Path, line_number: int, code: str, detail: str,
) -> None:
    if len(samples) < 50:
        samples.append(
            {"path": str(path), "line": line_number, "code": code, "detail": detail[:240]}
        )
