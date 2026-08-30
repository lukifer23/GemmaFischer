from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import chess

UCI_PATTERN = re.compile(r"^[a-h][1-8][a-h][1-8][qrbn]?$", re.IGNORECASE)
MINIMUM_TRAINING_RECORDS = 10_000
MINIMUM_EVALUATION_RECORDS = 1_000
FEN_PATTERN = re.compile(
    r"(?:FEN|Position):\s*([^\n]+)",
    re.IGNORECASE,
)


def audit_data(
    training_paths: list[Path],
    evaluation_paths: list[Path],
    output_path: Path,
    *,
    minimum_training_records: int = MINIMUM_TRAINING_RECORDS,
    minimum_evaluation_records: int = MINIMUM_EVALUATION_RECORDS,
) -> dict[str, Any]:
    training = _scan(training_paths)
    evaluation = _scan(evaluation_paths)
    train_fens = set(training.pop("_fens"))
    evaluation_fens = set(evaluation.pop("_fens"))
    train_labels = training.pop("_labels")
    evaluation.pop("_labels")
    conflicts = [
        {"fen": fen, "best_moves": sorted(moves)}
        for fen, moves in train_labels.items()
        if len(moves) > 1
    ]
    conflicts.sort(key=lambda item: item["fen"])
    leaked_fens = sorted(train_fens & evaluation_fens)
    payload = {
        "schema_version": "1.0",
        "status": (
            "blocked"
            if _blocked(
                training,
                evaluation,
                conflicts,
                leaked_fens,
                minimum_training_records,
                minimum_evaluation_records,
            )
            else "passed"
        ),
        "generated_at": datetime.now(UTC).isoformat(),
        "training": training,
        "evaluation": evaluation,
        "cross_dataset": {
            "conflicting_best_move_fens": len(conflicts),
            "conflict_samples": conflicts[:25],
            "train_evaluation_fen_overlap": len(leaked_fens),
            "leakage_samples": leaked_fens[:25],
        },
        "gate": {
            "ready_for_training": not _blocked(
                training,
                evaluation,
                conflicts,
                leaked_fens,
                minimum_training_records,
                minimum_evaluation_records,
            ),
            "requirements": {
                "minimum_training_records": minimum_training_records,
                "minimum_evaluation_records": minimum_evaluation_records,
                "malformed_records": 0,
                "invalid_fens": 0,
                "illegal_best_moves": 0,
                "missing_fens": 0,
                "missing_best_moves": 0,
                "missing_provenance": 0,
                "missing_license": 0,
                "duplicates_within_file": 0,
                "duplicates_across_files": 0,
                "conflicting_best_move_fens": 0,
                "train_evaluation_fen_overlap": 0,
            },
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output_path)
    return payload


def _blocked(
    training: dict[str, Any],
    evaluation: dict[str, Any],
    conflicts: list[dict[str, Any]],
    leaks: list[str],
    minimum_training_records: int,
    minimum_evaluation_records: int,
) -> bool:
    training_totals = training["totals"]
    evaluation_totals = evaluation["totals"]
    return any(
        (
            training_totals.get("records", 0) < minimum_training_records,
            evaluation_totals.get("records", 0) < minimum_evaluation_records,
            training_totals.get("malformed_records", 0),
            training_totals.get("invalid_fens", 0),
            training_totals.get("illegal_best_moves", 0),
            training_totals.get("missing_fens", 0),
            training_totals.get("missing_best_moves", 0),
            training_totals.get("missing_provenance", 0),
            training_totals.get("missing_license", 0),
            training_totals.get("duplicates_within_file", 0),
            training_totals.get("duplicates_across_files", 0),
            evaluation_totals.get("malformed_records", 0),
            evaluation_totals.get("invalid_fens", 0),
            evaluation_totals.get("illegal_best_moves", 0),
            evaluation_totals.get("missing_fens", 0),
            evaluation_totals.get("missing_best_moves", 0),
            evaluation_totals.get("missing_provenance", 0),
            evaluation_totals.get("missing_license", 0),
            evaluation_totals.get("duplicates_within_file", 0),
            evaluation_totals.get("duplicates_across_files", 0),
            len(conflicts),
            len(leaks),
        )
    )


def _scan(paths: list[Path]) -> dict[str, Any]:
    totals: Counter[str] = Counter()
    files: list[dict[str, Any]] = []
    all_hashes: set[str] = set()
    all_fens: set[str] = set()
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
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    file_counts["malformed_records"] += 1
                    _sample(issue_samples, path, line_number, "malformed_json", str(exc))
                    continue
                if not isinstance(record, dict):
                    file_counts["malformed_records"] += 1
                    _sample(
                        issue_samples,
                        path,
                        line_number,
                        "record_not_object",
                        type(record).__name__,
                    )
                    continue
                digest = hashlib.sha256(
                    json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest()
                if digest in file_hashes:
                    file_counts["duplicates_within_file"] += 1
                if digest in prior_file_hashes:
                    file_counts["duplicates_across_files"] += 1
                file_hashes.add(digest)
                raw_meta: object = record.get("meta")
                meta: dict[str, Any] = (
                    {str(key): value for key, value in raw_meta.items()}
                    if isinstance(raw_meta, dict)
                    else {}
                )
                task = str(record.get("task") or record.get("expert") or "unknown")
                tasks[task] += 1
                source = str(meta.get("source") or "")
                if source:
                    sources[source] += 1
                else:
                    file_counts["missing_provenance"] += 1
                if not meta.get("license") and not record.get("license"):
                    file_counts["missing_license"] += 1
                fen = _extract_fen(record, meta)
                if fen:
                    file_counts["records_with_fen"] += 1
                    try:
                        board = chess.Board(fen)
                        if not board.is_valid():
                            raise ValueError(f"status={board.status()}")
                        normalized = board.fen(en_passant="fen")
                        all_fens.add(normalized)
                        move = _extract_best_move(record, meta)
                        if move:
                            file_counts["records_with_best_move"] += 1
                            try:
                                parsed_move = chess.Move.from_uci(move.lower())
                            except ValueError:
                                parsed_move = None
                            if parsed_move is None or parsed_move not in board.legal_moves:
                                file_counts["illegal_best_moves"] += 1
                                _sample(issue_samples, path, line_number, "illegal_best_move", move)
                            else:
                                labels[normalized].add(move.lower())
                        else:
                            file_counts["missing_best_moves"] += 1
                    except ValueError as exc:
                        file_counts["invalid_fens"] += 1
                        _sample(issue_samples, path, line_number, "invalid_fen", str(exc))
                else:
                    file_counts["missing_fens"] += 1
        all_hashes.update(file_hashes)
        totals.update(file_counts)
        files.append(
            {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                **dict(file_counts),
                "tasks": dict(tasks.most_common()),
                "sources": dict(sources.most_common(10)),
            }
        )
    return {
        "files": files,
        "totals": dict(totals),
        "unique_records": len(all_hashes),
        "unique_fens": len(all_fens),
        "issue_samples": issue_samples,
        "_fens": all_fens,
        "_labels": labels,
    }


def _extract_fen(record: dict[str, Any], meta: dict[str, Any]) -> str | None:
    if isinstance(meta.get("fen"), str):
        return str(meta["fen"]).strip()
    for key in ("prompt", "question"):
        value = record.get(key)
        if isinstance(value, str) and (match := FEN_PATTERN.search(value)):
            return match.group(1).strip()
    return None


def _extract_best_move(record: dict[str, Any], meta: dict[str, Any]) -> str | None:
    value = meta.get("best_move")
    if isinstance(value, str) and UCI_PATTERN.fullmatch(value.strip()):
        return value.strip()
    response = record.get("response")
    if isinstance(response, str) and UCI_PATTERN.fullmatch(response.strip()):
        return response.strip()
    return None


def _sample(
    samples: list[dict[str, Any]], path: Path, line_number: int, code: str, detail: str
) -> None:
    if len(samples) < 50:
        samples.append(
            {"path": str(path), "line": line_number, "code": code, "detail": detail[:240]}
        )
