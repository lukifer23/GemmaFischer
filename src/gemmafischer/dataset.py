from __future__ import annotations

import csv
import hashlib
import heapq
import io
import json
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import chess

from .domain import RatingBucket, canonical_hash
from .engine import StockfishProvider
from .runtime import (
    LESSON_SELECTION_CONTRACT_VERSION,
    LESSON_SELECTION_SYSTEM_PROMPT,
    lesson_selection_prompt,
    lesson_selection_target,
    parse_lesson_selection,
)


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
    splits = ("train", "validation", "final_test")
    paths = {split: output_dir / f"{split}.jsonl" for split in splits}
    handles = {
        split: path.with_suffix(".jsonl.tmp").open("w", encoding="utf-8")
        for split, path in paths.items()
    }
    counts = {"train": 0, "validation": 0, "final_test": 0, "rejected": 0}
    rejection_reasons: Counter[str] = Counter()
    seen: set[str] = set()
    unique_position_goal = (limit + len(RatingBucket) - 1) // len(RatingBucket)
    held_out_positions = 1 if unique_position_goal >= 3 else 0
    if unique_position_goal >= 20:
        held_out_positions = max(1, unique_position_goal // 10)
    position_targets = {
        "train": unique_position_goal - 2 * held_out_positions,
        "validation": held_out_positions,
        "final_test": held_out_positions,
    }
    position_counts = {split: 0 for split in splits}
    try:
        selected, scanned = _select_puzzle_rows(
            archive_path,
            zstandard,
            candidate_count=max(unique_position_goal * 4, unique_position_goal + 32),
        )
        with StockfishProvider(node_budget=node_budget) as provider:
            for row in selected:
                if sum(counts[split] for split in splits) >= limit:
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
                    game_reference = row.get("GameId") or row.get("GameUrl")
                    if not game_reference:
                        raise ValueError("missing game lineage")
                    parsed_game = urlparse(game_reference)
                    game_path = parsed_game.path if parsed_game.scheme else game_reference
                    game_id = game_path.strip("/").split("/", 1)[0].split("#", 1)[0]
                    if not game_id:
                        raise ValueError("invalid game lineage")
                    lineage = f"lichess-game:{game_id}"
                    semantic = " ".join(chess.Board(fen).fen(en_passant="legal").split()[:4])
                    if semantic in seen:
                        counts["rejected"] += 1
                        continue
                    seen.add(semantic)
                    evidence = provider.analyze(fen, solution.uci())
                    bucket = int(hashlib.sha256(lineage.encode()).hexdigest()[:8], 16) % 10
                    if bucket == 0:
                        split = "final_test"
                    elif bucket == 1:
                        split = "validation"
                    else:
                        split = "train"
                    if position_counts[split] >= position_targets[split]:
                        continue
                    position_counts[split] += 1
                    for rating in RatingBucket:
                        if sum(counts[item] for item in splits) >= limit:
                            break
                        target = lesson_selection_target(evidence, rating)
                        response = json.dumps(target, separators=(",", ":"))
                        parse_lesson_selection(response, evidence, rating)
                        record: dict[str, Any] = {
                            "record_id": canonical_hash({
                                "source_id": source["id"], "lineage": lineage,
                                "fen": fen, "rating_bucket": rating.value,
                            }),
                            "task": LESSON_SELECTION_CONTRACT_VERSION,
                            "system_prompt": LESSON_SELECTION_SYSTEM_PROMPT,
                            "prompt": lesson_selection_prompt(evidence, rating),
                            "response": response,
                            "license": source["license"],
                            "meta": {
                                "fen": fen,
                                "best_move": solution.uci(),
                                "source": source["id"],
                                "source_item_id": row["PuzzleId"],
                                "source_game_id": game_id,
                                "source_position_id": semantic,
                                "lineage": lineage,
                                "license": source["license"],
                                "split": split,
                                "themes": row.get("Themes", "").split(),
                                "puzzle_rating": int(row["Rating"]),
                                "rating_bucket": rating.value,
                                "setup_move": moves[0].uci(),
                                "solution_move": solution.uci(),
                                "move_sequence": [move.uci() for move in moves],
                                "evidence_contract_version": evidence.schema_version,
                                "model_contract_version": LESSON_SELECTION_CONTRACT_VERSION,
                                "engine_binary_sha256": provider.binary_sha256,
                                "engine_node_budget": node_budget,
                                "selection_method": "full-archive-content-hash-reservoir",
                                "transformation": (
                                    "apply first UCI setup move; analyze solution position; "
                                    "render every product rating bucket"
                                ),
                            },
                            "input": evidence.model_dump(mode="json"),
                            "target": target,
                        }
                        handles[split].write(
                            json.dumps(record, separators=(",", ":")) + "\n"
                        )
                        counts[split] += 1
                # Bad source rows are rejected. Runtime/engine failures must
                # abort the build so a broken pipeline cannot publish a
                # deceptively small or empty dataset.
                except KeyError as exc:
                    counts["rejected"] += 1
                    rejection_reasons[f"missing-field:{exc.args[0]}"] += 1
                except (TypeError, ValueError) as exc:
                    counts["rejected"] += 1
                    rejection_reasons[f"invalid:{str(exc)[:120]}"] += 1
    finally:
        for handle in handles.values():
            handle.close()
    if sum(counts[split] for split in splits) != limit:
        for path in paths.values():
            path.with_suffix(".jsonl.tmp").unlink(missing_ok=True)
        raise RuntimeError(
            f"Dataset sampling could not satisfy split quotas: {counts}; "
            f"positions={position_counts}; targets={position_targets}"
        )
    for path in paths.values():
        path.with_suffix(".jsonl.tmp").replace(path)
    return {
        "source_id": source["id"],
        "archive_sha256": source["sha256"],
        "node_budget": node_budget,
        "source_rows_scanned": scanned,
        "selection_method": "full-archive-content-hash-reservoir",
        "counts": counts,
        "position_counts": position_counts,
        "position_targets": position_targets,
        "rejection_reasons": dict(rejection_reasons.most_common()),
        "outputs": {split: str(path) for split, path in paths.items()},
    }


def _select_puzzle_rows(
    archive_path: Path, zstandard: Any, *, candidate_count: int
) -> tuple[list[dict[str, str]], int]:
    """Select an order-independent bounded sample while streaming the complete archive."""
    heap: list[tuple[int, str, dict[str, str]]] = []
    scanned = 0
    with archive_path.open("rb") as raw:
        reader = csv.DictReader(
            io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(raw), encoding="utf-8")
        )
        for row in reader:
            scanned += 1
            puzzle_id = row.get("PuzzleId", "")
            if not puzzle_id:
                continue
            priority = int(hashlib.sha256(puzzle_id.encode()).hexdigest(), 16)
            item = (-priority, puzzle_id, {str(key): str(value) for key, value in row.items()})
            if len(heap) < candidate_count:
                heapq.heappush(heap, item)
            elif priority < -heap[0][0]:
                heapq.heapreplace(heap, item)
    return [item[2] for item in sorted(heap, key=lambda value: (-value[0], value[1]))], scanned
