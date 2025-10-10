#!/usr/bin/env python3
"""Refresh the standardized UCI dataset with deeper Stockfish analysis.

For every source record we:
  * Recompute the best move at a configurable depth/time limit (default depth 14)
  * Capture a separate depth-6 ``top-3`` move list for tactical diversity
  * Deduplicate positions by FEN and normalize the prompt/response fields
  * Persist detailed engine metadata, including centipawn/mate scores and PVs

The refreshed dataset is written to ``data/standardized/standardized_uci_expert_v2.jsonl``.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence
import sys

import chess

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_SOURCE_FILES: Sequence[Path] = (
    PROJECT_ROOT / "data" / "standardized" / "standardized_uci_expert.jsonl",
    PROJECT_ROOT / "data" / "standardized" / "standardized_enhanced_uci_expert.jsonl",
)
OUTPUT_PATH = PROJECT_ROOT / "data" / "standardized" / "standardized_uci_expert_v2.jsonl"


def iter_jsonl(path: Path) -> Iterator[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def save_jsonl(records: Iterable[Dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False))
            fh.write("\n")


def resolve_source_paths(explicit: Optional[Sequence[Path]]) -> List[Path]:
    """Resolve source dataset paths, ensuring they exist and removing duplicates."""
    candidates: List[Path] = []
    if explicit:
        for supplied in explicit:
            candidate = supplied.expanduser()
            if not candidate.is_absolute():
                candidate = (PROJECT_ROOT / candidate).resolve()
            if not candidate.exists():
                raise FileNotFoundError(f"Source dataset not found: {candidate}")
            candidates.append(candidate)
    else:
        for candidate in DEFAULT_SOURCE_FILES:
            if candidate.exists():
                candidates.append(candidate.resolve())

    if not candidates:
        raise FileNotFoundError(
            "No UCI source datasets detected. Provide --sources or populate data/standardized."
        )

    seen: set[Path] = set()
    unique_paths: List[Path] = []
    for path in candidates:
        resolved = path.resolve()
        if resolved == OUTPUT_PATH.resolve():
            # Avoid reading the file we are about to overwrite
            continue
        if resolved in seen:
            continue
        unique_paths.append(resolved)
        seen.add(resolved)
    if not unique_paths:
        raise FileNotFoundError("All provided sources matched the output file; nothing to refresh.")
    return unique_paths


def default_prompt_for_fen(fen: str) -> str:
    return (
        "Task: Find the best chess move\n"
        f"Position: {fen}\n"
        "Instruction: Analyze the position and respond with the best move in UCI format only.\n\n"
        "Response:"
    )


def refresh_dataset(
    sources: Sequence[Path],
    best_depth: int,
    best_time_limit_ms: int,
    top_depth: int,
    top_time_limit_ms: int,
    top_k: int,
    limit: Optional[int],
    output_path: Path,
) -> Dict[str, int]:
    from src.inference.chess_engine import ChessEngineManager

    stats = {
        "processed": 0,
        "written": 0,
        "deduplicated": 0,
        "skipped_missing_fen": 0,
        "skipped_invalid_fen": 0,
        "skipped_engine_failure": 0,
    }
    refreshed: List[Dict[str, object]] = []
    seen_fens: set[str] = set()

    with ChessEngineManager(debug=False) as engine:
        for source in sources:
            print(f"📥 Loading {source} ...")
            for row in iter_jsonl(source):
                if limit is not None and stats["written"] >= limit:
                    break
                stats["processed"] += 1

                meta = dict(row.get("meta") or {})
                fen = (meta.get("fen") or "").strip()
                if not fen:
                    stats["skipped_missing_fen"] += 1
                    continue

                if fen in seen_fens:
                    stats["deduplicated"] += 1
                    continue

                try:
                    board = chess.Board(fen)
                except ValueError:
                    stats["skipped_invalid_fen"] += 1
                    continue

                analysis_start = time.time()
                best_infos = engine.get_top_moves_info(
                    board,
                    depth=best_depth,
                    top_k=1,
                    time_limit_ms=best_time_limit_ms,
                )
                if not best_infos:
                    stats["skipped_engine_failure"] += 1
                    continue
                best_info = best_infos[0]
                best_move = best_info.get("move")
                if not best_move:
                    stats["skipped_engine_failure"] += 1
                    continue

                candidate_infos = engine.get_top_moves_info(
                    board,
                    depth=top_depth,
                    top_k=top_k,
                    time_limit_ms=top_time_limit_ms,
                )
                if not candidate_infos:
                    candidate_infos = best_infos

                seen_moves: set[str] = set()
                structured_top_moves: List[Dict[str, object]] = []
                for entry in candidate_infos:
                    move_uci = entry.get("move")
                    if not move_uci or move_uci in seen_moves:
                        continue
                    seen_moves.add(move_uci)
                    structured_top_moves.append({
                        "uci": move_uci,
                        "score_cp": entry.get("score_cp"),
                        "mate": entry.get("mate"),
                        "depth": entry.get("depth"),
                        "seldepth": entry.get("seldepth"),
                        "nodes": entry.get("nodes"),
                        "nps": entry.get("nps"),
                        "multipv": entry.get("multipv"),
                        "pv": entry.get("pv"),
                    })

                if not structured_top_moves:
                    structured_top_moves.append({
                        "uci": best_move,
                        "score_cp": best_info.get("score_cp"),
                        "mate": best_info.get("mate"),
                        "depth": best_info.get("depth"),
                        "seldepth": best_info.get("seldepth"),
                        "nodes": best_info.get("nodes"),
                        "nps": best_info.get("nps"),
                        "multipv": best_info.get("multipv", 1),
                        "pv": best_info.get("pv"),
                    })

                top_moves_uci = [entry["uci"] for entry in structured_top_moves]
                analysis_duration_ms = int((time.time() - analysis_start) * 1000)
                refreshed_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

                best_entry = {
                    "uci": best_move,
                    "score_cp": best_info.get("score_cp"),
                    "mate": best_info.get("mate"),
                    "depth": best_info.get("depth") or best_depth,
                    "seldepth": best_info.get("seldepth"),
                    "nodes": best_info.get("nodes"),
                    "nps": best_info.get("nps"),
                    "multipv": best_info.get("multipv", 1),
                    "pv": best_info.get("pv"),
                    "time_limit_ms": best_time_limit_ms,
                }

                top_moves_key = f"top_moves_depth{top_depth}"
                refreshed_meta = {
                    **meta,
                    "source": meta.get("source") or source.stem,
                    "input_source": source.name,
                    "best_move": best_move,
                    "stockfish_depth": best_depth,
                    "stockfish_time_limit_ms": best_time_limit_ms,
                    "analysis_generated_at": refreshed_at,
                    "analysis_duration_ms": analysis_duration_ms,
                    top_moves_key: top_moves_uci,
                    "top_moves": top_moves_uci,  # backward compatibility
                    "stockfish_analysis": {
                        "best_move": best_entry,
                        "top_moves": {
                            "depth": top_depth,
                            "time_limit_ms": top_time_limit_ms,
                            "entries": structured_top_moves,
                        },
                    },
                }
                refreshed_meta.setdefault("quality_score", meta.get("quality_score", 0.8))

                prompt = row.get("prompt") or default_prompt_for_fen(fen)

                refreshed.append({
                    "task": row.get("task", "engine_uci"),
                    "prompt": prompt,
                    "response": best_move,
                    "meta": refreshed_meta,
                })

                seen_fens.add(fen)
                stats["written"] += 1

                if stats["written"] % 100 == 0:
                    print(
                        f"  • Processed {stats['processed']} rows "
                        f"(written {stats['written']}, deduped {stats['deduplicated']})"
                    )

            if limit is not None and stats["written"] >= limit:
                break

    save_jsonl(refreshed, output_path)
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh UCI dataset with new Stockfish labels")
    parser.add_argument(
        "--sources",
        nargs="+",
        type=Path,
        default=None,
        help="Optional explicit list of source JSONL files to refresh.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help="Destination for the refreshed dataset (default: standardized_uci_expert_v2.jsonl).",
    )
    parser.add_argument(
        "--best-depth",
        type=int,
        default=14,
        help="Search depth for best-move annotation.",
    )
    parser.add_argument(
        "--best-time-limit-ms",
        type=int,
        default=1500,
        help="Time limit in milliseconds for best-move annotation.",
    )
    parser.add_argument(
        "--top-depth",
        type=int,
        default=6,
        help="Stockfish depth for the top-k move list.",
    )
    parser.add_argument(
        "--top-time-limit-ms",
        type=int,
        default=500,
        help="Time limit in milliseconds for the top-k move list.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of alternative moves to retain for tactical coverage.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of refreshed samples (useful for smoke tests).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        sources = resolve_source_paths(args.sources)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc))

    output_path = args.output if args.output else OUTPUT_PATH
    if not output_path.is_absolute():
        output_path = (PROJECT_ROOT / output_path).resolve()

    start = time.time()
    stats = refresh_dataset(
        sources=sources,
        best_depth=args.best_depth,
        best_time_limit_ms=args.best_time_limit_ms,
        top_depth=args.top_depth,
        top_time_limit_ms=args.top_time_limit_ms,
        top_k=args.top_k,
        limit=args.limit,
        output_path=output_path,
    )
    elapsed = time.time() - start

    print(f"\n✅ Refreshed dataset written to {output_path} ({stats['written']} samples)")
    print(
        f"   Processed={stats['processed']} | Deduped={stats['deduplicated']} | "
        f"Missing FEN={stats['skipped_missing_fen']} | Invalid FEN={stats['skipped_invalid_fen']} | "
        f"Engine skips={stats['skipped_engine_failure']}"
    )
    print(f"⏱️  Elapsed time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
