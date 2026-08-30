from __future__ import annotations

import csv
import hashlib
import io
import json
import urllib.request
from pathlib import Path
from typing import Any

import chess

from .coach import deterministic_coach
from .domain import RatingBucket, canonical_hash
from .engine import StockfishProvider


def load_source(manifest_path: Path, source_id: str) -> dict[str, str]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    for source in payload.get("sources", []):
        if source.get("id") == source_id:
            return {str(key): str(value) for key, value in source.items()}
    raise ValueError(f"Unknown source ID: {source_id}")


def acquire_source(source: dict[str, str], output_path: Path) -> dict[str, object]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".partial")
    digest = hashlib.sha256()
    size = 0
    try:
        with urllib.request.urlopen(source["url"], timeout=60) as response, temporary.open(
            "wb"
        ) as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
                digest.update(chunk)
                size += len(chunk)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    actual = digest.hexdigest()
    if actual != source["sha256"]:
        temporary.unlink(missing_ok=True)
        raise ValueError(f"Source hash mismatch: expected {source['sha256']}, received {actual}")
    temporary.replace(output_path)
    return {"source_id": source["id"], "path": str(output_path), "sha256": actual, "bytes": size}


def build_puzzle_dataset(
    archive_path: Path,
    output_dir: Path,
    source: dict[str, str],
    *,
    limit: int,
    node_budget: int,
) -> dict[str, object]:
    if limit < 1:
        raise ValueError("limit must be at least 1")
    try:
        import zstandard
    except ImportError as exc:
        raise RuntimeError("Install the data profile with: uv sync --extra data") from exc
    digest = hashlib.sha256()
    with archive_path.open("rb") as archive:
        for chunk in iter(lambda: archive.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != source["sha256"]:
        raise ValueError("The puzzle archive does not match the pinned source manifest")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {split: output_dir / f"{split}.jsonl" for split in ("train", "evaluation")}
    handles = {
        split: path.with_suffix(".jsonl.tmp").open("w", encoding="utf-8")
        for split, path in paths.items()
    }
    counts = {"train": 0, "evaluation": 0, "rejected": 0}
    seen: set[str] = set()
    try:
        with StockfishProvider(node_budget=node_budget) as provider, archive_path.open("rb") as raw:
            reader = csv.DictReader(
                io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(raw), encoding="utf-8")
            )
            for row in reader:
                if counts["train"] + counts["evaluation"] >= limit:
                    break
                try:
                    board = chess.Board(row["FEN"])
                    moves = tuple(chess.Move.from_uci(value) for value in row["Moves"].split())
                    if len(moves) < 2 or moves[0] not in board.legal_moves:
                        raise ValueError("invalid setup move")
                    board.push(moves[0])
                    solution = moves[1]
                    if solution not in board.legal_moves:
                        raise ValueError("invalid solution move")
                    fen = board.fen(en_passant="fen")
                    lineage = f"lichess-puzzle:{row['PuzzleId']}"
                    if fen in seen:
                        counts["rejected"] += 1
                        continue
                    seen.add(fen)
                    evidence = provider.analyze(fen, solution.uci())
                    lesson = deterministic_coach(evidence, RatingBucket.CLUB, solution.uci())
                    split = (
                        "evaluation"
                        if int(hashlib.sha256(lineage.encode()).hexdigest()[:8], 16) % 10 == 0
                        else "train"
                    )
                    record: dict[str, Any] = {
                        "record_id": canonical_hash(
                            {"source_id": source["id"], "lineage": lineage, "fen": fen}
                        ),
                        "task": "grounded_lesson_plan",
                        "prompt": f"FEN: {fen}",
                        "response": solution.uci(),
                        "license": source["license"],
                        "meta": {
                            "fen": fen,
                            "best_move": solution.uci(),
                            "source": source["id"],
                            "source_item_id": row["PuzzleId"],
                            "lineage": lineage,
                            "license": source["license"],
                            "split": split,
                            "themes": row.get("Themes", "").split(),
                            "rating": int(row["Rating"]),
                            "transformation": (
                                "apply first UCI setup move; analyze solution position"
                            ),
                        },
                        "input": {
                            "position_id": evidence.position_id,
                            "candidate_set_id": (
                                evidence.candidate_set.evidence_id
                                if evidence.candidate_set
                                else None
                            ),
                            "concepts": [
                                item.model_dump(mode="json") for item in evidence.concepts
                            ],
                        },
                        "target": (
                            lesson.lesson_plan.model_dump(mode="json")
                            if lesson.lesson_plan
                            else None
                        ),
                    }
                    handles[split].write(json.dumps(record, separators=(",", ":")) + "\n")
                    counts[split] += 1
                # Bad source rows are rejected. Runtime/engine failures must
                # abort the build so a broken pipeline cannot publish a
                # deceptively small or empty dataset.
                except (KeyError, TypeError, ValueError):
                    counts["rejected"] += 1
    finally:
        for handle in handles.values():
            handle.close()
    for path in paths.values():
        path.with_suffix(".jsonl.tmp").replace(path)
    return {
        "source_id": source["id"],
        "archive_sha256": source["sha256"],
        "node_budget": node_budget,
        "counts": counts,
        "outputs": {split: str(path) for split, path in paths.items()},
    }
