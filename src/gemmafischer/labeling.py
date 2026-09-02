from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .domain import EngineEvidence, RatingBucket
from .runtime import lesson_selection_catalog, parse_lesson_selection

RUBRIC_SCORE_FIELDS = (
    "correctness",
    "clarity",
    "relevance",
    "rating_fit",
    "actionability",
    "question_usefulness",
    "hint_usefulness",
)


def export_label_packet(
    dataset_path: Path, output_path: Path, *, limit: int = 2_500
) -> dict[str, Any]:
    """Export a blind, contract-complete packet for two independent reviewers."""
    if limit < 1:
        raise ValueError("Label packet limit must be positive")
    records: list[dict[str, Any]] = []
    with dataset_path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                evidence = EngineEvidence.model_validate(record["input"])
                rating = RatingBucket(record["meta"]["rating_bucket"])
                catalog = lesson_selection_catalog(evidence, rating)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid label source at line {line_number}") from exc
            records.append(
                {
                    "record_id": record["record_id"],
                    "rating_bucket": rating.value,
                    "fen": evidence.fen,
                    "claims": {
                        key: value.model_dump(mode="json") for key, value in catalog.claims.items()
                    },
                    "concept_ids": list(catalog.concept_ids),
                    "question_template_ids": list(catalog.question_template_ids),
                    "hint_template_ids": list(catalog.hint_template_ids),
                }
            )
            if len(records) == limit:
                break
    packet = {
        "schema_version": "1.0",
        "generated_at": datetime.now(UTC).isoformat(),
        "source": str(dataset_path),
        "required_reviewers_per_record": 2,
        "rubric": {
            "score_fields": list(RUBRIC_SCORE_FIELDS),
            "score_range": [1, 5],
            "harmful_omission": "boolean",
        },
        "records": records,
    }
    _write_json_atomic(output_path, packet)
    return {"records": len(records), "output": str(output_path)}


def validate_label_responses(
    dataset_path: Path, response_path: Path, output_path: Path
) -> dict[str, Any]:
    source: dict[str, tuple[EngineEvidence, RatingBucket]] = {}
    with dataset_path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                source[str(record["record_id"])] = (
                    EngineEvidence.model_validate(record["input"]),
                    RatingBucket(record["meta"]["rating_bucket"]),
                )
    accepted: list[dict[str, Any]] = []
    reviewers: set[str] = set()
    seen_pairs: set[tuple[str, str]] = set()
    with response_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                response = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid response JSON at line {line_number}") from exc
            if not isinstance(response, dict):
                raise ValueError(f"Response must be an object at line {line_number}")
            record_id = str(response.get("record_id", ""))
            reviewer_id = str(response.get("reviewer_id", ""))
            if not reviewer_id or record_id not in source:
                raise ValueError(f"Unknown reviewer or record at line {line_number}")
            pair = (record_id, reviewer_id)
            if pair in seen_pairs:
                raise ValueError(f"Duplicate reviewer response at line {line_number}")
            evidence, rating = source[record_id]
            selection = response.get("selection")
            encoded = json.dumps(selection, separators=(",", ":"))
            parse_lesson_selection(encoded, evidence, rating)
            _validate_rubric(response.get("rubric"), line_number)
            seen_pairs.add(pair)
            reviewers.add(reviewer_id)
            accepted.append(response)
    if len(reviewers) != 2:
        raise ValueError(
            f"Exactly two independent reviewers are required; received {len(reviewers)}"
        )
    missing = sorted(
        (record_id, reviewer)
        for record_id in source
        for reviewer in reviewers
        if (record_id, reviewer) not in seen_pairs
    )
    if missing:
        raise ValueError(f"Every record needs both reviews; missing {len(missing)} responses")
    by_record: dict[str, list[dict[str, Any]]] = {record_id: [] for record_id in source}
    for response in accepted:
        by_record[str(response["record_id"])].append(response)
    agreed: list[dict[str, Any]] = []
    disagreements: list[dict[str, Any]] = []
    selection_agreements = 0
    for record_id, responses in by_record.items():
        first, second = sorted(responses, key=lambda item: str(item["reviewer_id"]))
        selection_agrees = _canonical_selection(first["selection"]) == _canonical_selection(
            second["selection"]
        )
        selection_agreements += int(selection_agrees)
        rubric_disagrees = _rubric_disagrees(first["rubric"], second["rubric"])
        if selection_agrees and not rubric_disagrees:
            agreed.append(
                {
                    "record_id": record_id,
                    "selection": first["selection"],
                    "rubric": _merge_agreed_rubric(first["rubric"], second["rubric"]),
                }
            )
        else:
            disagreements.append(
                {
                    "record_id": record_id,
                    "reasons": [
                        reason
                        for reason, present in (
                            ("selection", not selection_agrees),
                            ("rubric", rubric_disagrees),
                        )
                        if present
                    ],
                    "reviewer_selections": [
                        {
                            "reviewer_id": response["reviewer_id"],
                            "selection": response["selection"],
                            "rubric": response["rubric"],
                        }
                        for response in (first, second)
                    ],
                }
            )
    agreement = selection_agreements / len(source) if source else 0.0
    result = {
        "schema_version": "1.0",
        "status": "passed" if not disagreements else "needs_adjudication",
        "record_count": len(source),
        "response_count": len(accepted),
        "reviewer_count": len(reviewers),
        "reviewer_ids": sorted(reviewers),
        "exact_selection_agreement": agreement,
        "agreement_count": selection_agreements,
        "disagreement_count": len(disagreements),
        "adjudication_complete": not disagreements,
        "reviewer_rubric_summary": _rubric_summary(
            [response["rubric"] for response in accepted]
        ),
        "agreed": agreed,
        "disagreements": disagreements,
    }
    _write_json_atomic(output_path, result)
    return result


def adjudicate_label_responses(
    dataset_path: Path,
    validation_path: Path,
    adjudication_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Resolve every two-reviewer disagreement with one independent adjudicator."""
    source: dict[str, tuple[EngineEvidence, RatingBucket]] = {}
    with dataset_path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                source[str(record["record_id"])] = (
                    EngineEvidence.model_validate(record["input"]),
                    RatingBucket(record["meta"]["rating_bucket"]),
                )
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    reviewer_ids = {str(value) for value in validation.get("reviewer_ids", [])}
    if len(reviewer_ids) != 2 or validation.get("record_count") != len(source):
        raise ValueError("Validation artifact does not cover this dataset with two reviewers")
    disputes = {
        str(item["record_id"]): item for item in validation.get("disagreements", [])
    }
    resolved: dict[str, dict[str, Any]] = {}
    adjudicator_ids: set[str] = set()
    with adjudication_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                response = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid adjudication JSON at line {line_number}") from exc
            if not isinstance(response, dict):
                raise ValueError(f"Adjudication must be an object at line {line_number}")
            record_id = str(response.get("record_id", ""))
            adjudicator_id = str(response.get("adjudicator_id", ""))
            if record_id not in disputes or record_id in resolved:
                raise ValueError(f"Unknown or duplicate dispute at line {line_number}")
            if not adjudicator_id or adjudicator_id in reviewer_ids:
                raise ValueError(f"Adjudicator must be independent at line {line_number}")
            evidence, rating = source[record_id]
            selection = response.get("selection")
            parse_lesson_selection(
                json.dumps(selection, separators=(",", ":")), evidence, rating
            )
            _validate_rubric(response.get("rubric"), line_number)
            adjudicator_ids.add(adjudicator_id)
            resolved[record_id] = {
                "record_id": record_id,
                "selection": selection,
                "adjudicator_id": adjudicator_id,
                "rubric": response["rubric"],
            }
    if set(resolved) != set(disputes):
        raise ValueError(
            f"Every disagreement must be adjudicated; resolved {len(resolved)} of {len(disputes)}"
        )
    if len(adjudicator_ids) > 1:
        raise ValueError("One consistent independent adjudicator is required")
    final = [*validation.get("agreed", []), *resolved.values()]
    final.sort(key=lambda item: str(item["record_id"]))
    result = {
        "schema_version": "1.0",
        "status": "passed",
        "record_count": len(source),
        "reviewer_count": 2,
        "reviewer_ids": sorted(reviewer_ids),
        "adjudicator_ids": sorted(adjudicator_ids),
        "exact_selection_agreement": validation.get("exact_selection_agreement", 0.0),
        "disagreement_count": len(disputes),
        "adjudication_complete": True,
        "rubric_summary": _rubric_summary([item["rubric"] for item in final]),
        "selections": final,
    }
    _write_json_atomic(output_path, result)
    return result


def apply_human_gold(
    source_dir: Path, human_gold_path: Path, output_dir: Path
) -> dict[str, Any]:
    """Apply adjudicated selections to train rows and preserve held-out partitions."""
    if source_dir.resolve() == output_dir.resolve():
        raise ValueError("Human gold output must not overwrite the canonical source")
    if output_dir.exists():
        raise ValueError("Human gold output directory must not already exist")
    temporary_dir = output_dir.with_name(output_dir.name + ".tmp")
    if temporary_dir.exists():
        raise ValueError("Stale human gold temporary directory must be inspected")
    gold = json.loads(human_gold_path.read_text(encoding="utf-8"))
    if (
        not isinstance(gold, dict)
        or gold.get("status") != "passed"
        or gold.get("adjudication_complete") is not True
        or not isinstance(gold.get("selections"), list)
    ):
        raise ValueError("Human gold is incomplete or not adjudicated")
    selections = {
        str(item["record_id"]): item["selection"]
        for item in gold["selections"]
        if isinstance(item, dict) and item.get("record_id")
    }
    if len(selections) != len(gold["selections"]):
        raise ValueError("Human gold contains duplicate or invalid record IDs")
    temporary_dir.mkdir(parents=True, exist_ok=False)
    applied: set[str] = set()
    counts: dict[str, int] = {}
    for partition in ("train", "validation", "final_test"):
        source = source_dir / f"{partition}.jsonl"
        destination = temporary_dir / source.name
        if not source.is_file():
            raise ValueError(f"Missing canonical partition: {source}")
        if partition != "train":
            shutil.copyfile(source, destination)
            counts[partition] = sum(1 for line in source.open(encoding="utf-8") if line.strip())
            continue
        count = 0
        with source.open(encoding="utf-8") as reader, destination.open(
            "w", encoding="utf-8"
        ) as writer:
            for line in reader:
                if not line.strip():
                    continue
                record = json.loads(line)
                record_id = str(record.get("record_id", ""))
                if record_id in selections:
                    evidence = EngineEvidence.model_validate(record["input"])
                    rating = RatingBucket(record["meta"]["rating_bucket"])
                    selection = selections[record_id]
                    parse_lesson_selection(
                        json.dumps(selection, separators=(",", ":")), evidence, rating
                    )
                    record["target"] = selection
                    record["response"] = json.dumps(selection, separators=(",", ":"))
                    record["meta"] = {
                        **record["meta"],
                        "label_authority": "two-reviewer-adjudicated-human-gold",
                    }
                    applied.add(record_id)
                writer.write(json.dumps(record, separators=(",", ":")) + "\n")
                count += 1
        counts[partition] = count
    missing = sorted(set(selections) - applied)
    if missing:
        raise ValueError(f"Human gold references {len(missing)} records outside train")
    receipt = {
        "schema_version": "1.0",
        "status": "passed",
        "human_gold": str(human_gold_path),
        "human_gold_sha256": _sha256_file(human_gold_path),
        "human_labeled_records": len(applied),
        "counts": counts,
    }
    _write_json_atomic(temporary_dir / "human-gold-application.json", receipt)
    os.replace(temporary_dir, output_dir)
    return receipt


def _validate_rubric(value: object, line_number: int) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"Missing rubric at line {line_number}")
    for field in RUBRIC_SCORE_FIELDS:
        score = value.get(field)
        if not isinstance(score, int) or isinstance(score, bool) or not 1 <= score <= 5:
            raise ValueError(f"Rubric {field} must be an integer from 1 to 5 at line {line_number}")
    if not isinstance(value.get("harmful_omission"), bool):
        raise ValueError(f"Rubric harmful_omission must be boolean at line {line_number}")


def _canonical_selection(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rubric_disagrees(first: dict[str, Any], second: dict[str, Any]) -> bool:
    return bool(
        first["harmful_omission"] != second["harmful_omission"]
        or any(abs(first[field] - second[field]) >= 2 for field in RUBRIC_SCORE_FIELDS)
    )


def _merge_agreed_rubric(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    return {
        **{field: (first[field] + second[field]) / 2 for field in RUBRIC_SCORE_FIELDS},
        "harmful_omission": first["harmful_omission"] or second["harmful_omission"],
    }


def _rubric_summary(values: list[dict[str, Any]]) -> dict[str, Any]:
    if not values:
        return {
            "mean_scores": {field: 0.0 for field in RUBRIC_SCORE_FIELDS},
            "harmful_omission_count": 0,
        }
    return {
        "mean_scores": {
            field: sum(float(value[field]) for value in values) / len(values)
            for field in RUBRIC_SCORE_FIELDS
        },
        "harmful_omission_count": sum(bool(value["harmful_omission"]) for value in values),
    }


def _write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)
