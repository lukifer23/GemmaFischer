from __future__ import annotations

import csv
import hashlib
import heapq
import io
import json
import platform
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import chess

from .dataset import load_source
from .engine import StockfishProvider, sha256_file

ScoreExpectation = Literal["any", "mate_for_side", "mate_against_side", "terminal"]


@dataclass(frozen=True)
class AccuracyPosition:
    id: str
    fen: str
    category: str
    expected_top_moves: tuple[str, ...]
    expected_terminal_reason: str | None
    score_expectation: ScoreExpectation
    source: str
    license: str
    reference_method: str


def load_accuracy_positions(path: Path) -> tuple[AccuracyPosition, ...]:
    positions: list[AccuracyPosition] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
            position = AccuracyPosition(
                id=str(raw["id"]),
                fen=str(raw["fen"]),
                category=str(raw["category"]),
                expected_top_moves=tuple(str(move) for move in raw["expected_top_moves"]),
                expected_terminal_reason=(
                    str(raw["expected_terminal_reason"])
                    if raw.get("expected_terminal_reason") is not None
                    else None
                ),
                score_expectation=raw["score_expectation"],
                source=str(raw["source"]),
                license=str(raw["license"]),
                reference_method=str(raw["reference_method"]),
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid accuracy fixture at line {line_number}") from exc
        _validate_position(position, line_number)
        if position.id in seen_ids:
            raise ValueError(f"Duplicate accuracy fixture ID: {position.id}")
        seen_ids.add(position.id)
        positions.append(position)
    if not positions:
        raise ValueError("The accuracy fixture file is empty")
    return tuple(positions)


def run_constructed_accuracy_benchmark(
    fixture_path: Path,
    output_path: Path,
    *,
    repeats: int = 3,
    node_budget: int = 250_000,
    stockfish_path: str | None = None,
    score_tolerance_cp: int = 15,
) -> dict[str, Any]:
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    positions = load_accuracy_positions(fixture_path)
    raw: list[dict[str, Any]] = []
    with StockfishProvider(path=stockfish_path, node_budget=node_budget) as provider:
        for position in positions:
            runs = [_evaluate_position(provider, position) for _ in range(repeats)]
            baseline = runs[0]
            for run in runs:
                run["repeatable"] = _is_repeatable(baseline, run, score_tolerance_cp)
            raw.append({"fixture": _position_payload(position), "runs": runs})
        engine = _engine_payload(provider)

    flat = [run for item in raw for run in item["runs"]]
    nonterminal = [run for run in flat if run["top1_hit"] is not None]
    summary = {
        "top1_hits": sum(run["top1_hit"] is True for run in nonterminal),
        "top1_total": len(nonterminal),
        "top1_rate": _rate(nonterminal, "top1_hit"),
        "top3_hits": sum(run["top3_hit"] is True for run in nonterminal),
        "top3_total": len(nonterminal),
        "top3_rate": _rate(nonterminal, "top3_hit"),
        "legality_rate": _rate(flat, "legal"),
        "terminal_correctness_rate": _rate(flat, "terminal_correct"),
        "score_mate_consistency_rate": _rate(flat, "score_mate_consistent"),
        "repeatability_rate": _rate(flat, "repeatable"),
    }
    gates = {
        key: {"required": 1.0, "actual": summary[key], "passed": summary[key] == 1.0}
        for key in (
            "top1_rate",
            "top3_rate",
            "legality_rate",
            "terminal_correctness_rate",
            "score_mate_consistency_rate",
            "repeatability_rate",
        )
    }
    payload: dict[str, Any] = {
        **_environment_payload(),
        "schema_version": "1.0",
        "benchmark": "constructed-chess-accuracy",
        "status": "passed" if all(item["passed"] for item in gates.values()) else "failed",
        "evidence_scope": "rules-derived-edge-suite",
        "fixture_path": str(fixture_path),
        "fixture_sha256": sha256_file(fixture_path),
        "fixture_count": len(positions),
        "repeats": repeats,
        "score_tolerance_cp": score_tolerance_cp,
        "engine": engine,
        "summary": summary,
        "gates": gates,
        "cases": raw,
    }
    _write_payload(output_path, payload)
    return payload


def run_lichess_puzzle_accuracy_benchmark(
    archive_path: Path,
    source_manifest_path: Path,
    output_path: Path,
    *,
    sample_size: int = 100,
    node_budget: int = 250_000,
    stockfish_path: str | None = None,
    selection_seed: str = "gemmafischer-accuracy-v1",
) -> dict[str, Any]:
    if sample_size < 1:
        raise ValueError("sample_size must be at least 1")
    if not archive_path.is_file():
        raise FileNotFoundError(f"Lichess puzzle archive not found: {archive_path}")
    try:
        import zstandard
    except ImportError as exc:
        raise RuntimeError("Install the data profile with: uv sync --extra data") from exc
    source = load_source(source_manifest_path, "lichess-puzzles-2026-08-02")
    archive_sha256 = sha256_file(archive_path)
    if archive_sha256 != source["sha256"]:
        raise ValueError("The puzzle archive does not match the pinned source manifest")

    # The evaluation split is isolated from training by the same lineage rule as
    # dataset construction. The N lowest seeded PuzzleId hashes make selection
    # independent of archive order and exactly reproducible.
    selected: list[tuple[int, str, dict[str, str]]] = []
    rows_scanned = 0
    evaluation_rows = 0
    invalid_rows = 0
    with archive_path.open("rb") as raw:
        stream = zstandard.ZstdDecompressor().stream_reader(raw)
        reader = csv.DictReader(io.TextIOWrapper(stream, encoding="utf-8"))
        for row in reader:
            rows_scanned += 1
            try:
                puzzle_id = row["PuzzleId"]
                lineage = f"lichess-puzzle:{puzzle_id}"
                if int(hashlib.sha256(lineage.encode()).hexdigest()[:8], 16) % 10 != 0:
                    continue
                evaluation_rows += 1
                rank = int(hashlib.sha256(f"{selection_seed}:{puzzle_id}".encode()).hexdigest(), 16)
                entry = (-rank, puzzle_id, {str(key): str(value) for key, value in row.items()})
                if len(selected) < sample_size:
                    heapq.heappush(selected, entry)
                elif rank < -selected[0][0]:
                    heapq.heapreplace(selected, entry)
            except (KeyError, TypeError, ValueError):
                invalid_rows += 1
    if len(selected) < sample_size:
        raise ValueError(
            f"Only {len(selected)} valid held-out puzzles were available; requested {sample_size}"
        )

    ordered = [entry[2] for entry in sorted(selected, key=lambda item: (-item[0], item[1]))]
    cases: list[dict[str, Any]] = []
    with StockfishProvider(path=stockfish_path, node_budget=node_budget) as provider:
        for row in ordered:
            cases.append(_evaluate_lichess_row(provider, row))
        engine = _engine_payload(provider)

    summary = {
        "top1_hits": sum(case["top1_hit"] for case in cases),
        "top1_total": len(cases),
        "top1_rate": _rate(cases, "top1_hit"),
        "top3_hits": sum(case["top3_hit"] for case in cases),
        "top3_total": len(cases),
        "top3_rate": _rate(cases, "top3_hit"),
        "legality_rate": _rate(cases, "legal"),
        "categories": _category_summary(cases),
    }
    gates = {
        "top1_rate": {
            "required": 0.80,
            "actual": summary["top1_rate"],
            "passed": summary["top1_rate"] >= 0.80,
        },
        "top3_rate": {
            "required": 0.95,
            "actual": summary["top3_rate"],
            "passed": summary["top3_rate"] >= 0.95,
        },
        "legality_rate": {
            "required": 1.0,
            "actual": summary["legality_rate"],
            "passed": summary["legality_rate"] == 1.0,
        },
    }
    payload = {
        **_environment_payload(),
        "schema_version": "1.0",
        "benchmark": "lichess-held-out-puzzle-accuracy",
        "status": "passed" if all(item["passed"] for item in gates.values()) else "failed",
        "evidence_scope": "external-cc0-label-agreement",
        "source": {
            "id": source["id"],
            "license": source["license"],
            "archive_path": str(archive_path),
            "archive_sha256": archive_sha256,
            "manifest_path": str(source_manifest_path),
            "manifest_sha256": sha256_file(source_manifest_path),
        },
        "selection": {
            "seed": selection_seed,
            "rule": "evaluation lineage split; lowest sha256(seed:PuzzleId)",
            "sample_size": sample_size,
            "rows_scanned": rows_scanned,
            "eligible_evaluation_rows": evaluation_rows,
            "invalid_rows": invalid_rows,
        },
        "engine": engine,
        "summary": summary,
        "gates": gates,
        "cases": cases,
    }
    _write_payload(output_path, payload)
    return payload


def _validate_position(position: AccuracyPosition, line_number: int) -> None:
    if position.score_expectation not in {
        "any",
        "mate_for_side",
        "mate_against_side",
        "terminal",
    }:
        raise ValueError(f"Invalid score expectation at line {line_number}")
    try:
        board = chess.Board(position.fen)
    except ValueError as exc:
        raise ValueError(f"Invalid FEN at line {line_number}") from exc
    if not board.is_valid():
        raise ValueError(f"Invalid board state at line {line_number}")
    outcome = board.outcome(claim_draw=False)
    actual_terminal = outcome.termination.name.lower() if outcome else None
    if actual_terminal != position.expected_terminal_reason:
        raise ValueError(f"Terminal expectation does not match chess rules at line {line_number}")
    if outcome is not None:
        if position.expected_top_moves or position.score_expectation != "terminal":
            raise ValueError(f"Terminal fixture has move expectations at line {line_number}")
        return
    if not position.expected_top_moves or position.score_expectation == "terminal":
        raise ValueError(f"Nonterminal fixture lacks move expectations at line {line_number}")
    for value in position.expected_top_moves:
        try:
            move = chess.Move.from_uci(value)
        except ValueError as exc:
            raise ValueError(f"Invalid expected move at line {line_number}: {value}") from exc
        if move not in board.legal_moves:
            raise ValueError(f"Illegal expected move at line {line_number}: {value}")


def _evaluate_position(
    provider: StockfishProvider, position: AccuracyPosition
) -> dict[str, Any]:
    evidence = provider.analyze(position.fen)
    candidates = list(evidence.candidates)
    moves = [candidate.move_uci for candidate in candidates]
    expected = set(position.expected_top_moves)
    board = chess.Board(position.fen)
    legal = _candidates_are_legal(board, candidates)
    terminal_correct = evidence.terminal_reason == position.expected_terminal_reason
    if position.expected_terminal_reason is None:
        terminal_correct = terminal_correct and evidence.candidate_set is not None
    else:
        terminal_correct = terminal_correct and evidence.candidate_set is None
    return {
        "candidate_moves": moves,
        "candidate_scores": [
            {"score_cp": candidate.score_cp, "mate_in": candidate.mate_in}
            for candidate in candidates
        ],
        "top1_hit": (moves[0] in expected) if moves else None,
        "top3_hit": bool(expected.intersection(moves[:3])) if moves else None,
        "legal": legal,
        "terminal_reason": evidence.terminal_reason,
        "terminal_correct": terminal_correct,
        "score_mate_consistent": _score_is_consistent(candidates, position.score_expectation),
        "repeatable": True,
    }


def _evaluate_lichess_row(provider: StockfishProvider, row: dict[str, str]) -> dict[str, Any]:
    board = chess.Board(row["FEN"])
    sequence = tuple(chess.Move.from_uci(value) for value in row["Moves"].split())
    if len(sequence) < 2 or sequence[0] not in board.legal_moves:
        raise ValueError(f"Invalid setup move in Lichess puzzle {row['PuzzleId']}")
    board.push(sequence[0])
    expected = sequence[1]
    if expected not in board.legal_moves:
        raise ValueError(f"Invalid solution move in Lichess puzzle {row['PuzzleId']}")
    fen = board.fen(en_passant="fen")
    evidence = provider.analyze(fen)
    candidates = list(evidence.candidates)
    moves = [candidate.move_uci for candidate in candidates]
    themes = sorted(set(row.get("Themes", "").split()))
    return {
        "puzzle_id": row["PuzzleId"],
        "fen": fen,
        "expected_move": expected.uci(),
        "candidate_moves": moves,
        "top1_hit": bool(moves and moves[0] == expected.uci()),
        "top3_hit": expected.uci() in moves[:3],
        "legal": _candidates_are_legal(board, candidates),
        "rating": int(row["Rating"]),
        "themes": themes,
    }


def _candidates_are_legal(board: chess.Board, candidates: list[Any]) -> bool:
    for candidate in candidates:
        current = board.copy(stack=False)
        for value in candidate.pv_uci:
            try:
                move = chess.Move.from_uci(value)
            except ValueError:
                return False
            if move not in current.legal_moves:
                return False
            current.push(move)
    return True


def _score_is_consistent(candidates: list[Any], expectation: ScoreExpectation) -> bool:
    if expectation == "terminal":
        return not candidates
    if not candidates:
        return False
    if any((item.score_cp is None) == (item.mate_in is None) for item in candidates):
        return False
    top_mate = candidates[0].mate_in
    if expectation == "mate_for_side":
        return top_mate is not None and top_mate > 0
    if expectation == "mate_against_side":
        return top_mate is not None and top_mate < 0
    return True


def _is_repeatable(baseline: dict[str, Any], run: dict[str, Any], tolerance: int) -> bool:
    if baseline["candidate_moves"] != run["candidate_moves"]:
        return False
    for left, right in zip(baseline["candidate_scores"], run["candidate_scores"], strict=True):
        if left["mate_in"] != right["mate_in"]:
            return False
        if (
            left["score_cp"] is not None
            and right["score_cp"] is not None
            and abs(left["score_cp"] - right["score_cp"]) > tolerance
        ):
            return False
    return bool(baseline["terminal_reason"] == run["terminal_reason"])


def _category_summary(cases: list[dict[str, Any]]) -> dict[str, dict[str, int | float]]:
    categories: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        for theme in case["themes"] or ["untagged"]:
            categories.setdefault(theme, []).append(case)
    return {
        theme: {
            "count": len(items),
            "top1_rate": _rate(items, "top1_hit"),
            "top3_rate": _rate(items, "top3_hit"),
        }
        for theme, items in sorted(categories.items())
    }


def _position_payload(position: AccuracyPosition) -> dict[str, Any]:
    return {
        "id": position.id,
        "fen": position.fen,
        "category": position.category,
        "expected_top_moves": list(position.expected_top_moves),
        "expected_terminal_reason": position.expected_terminal_reason,
        "score_expectation": position.score_expectation,
        "source": position.source,
        "license": position.license,
        "reference_method": position.reference_method,
    }


def _engine_payload(provider: StockfishProvider) -> dict[str, Any]:
    return {
        "path": str(provider.path),
        "binary_sha256": provider.binary_sha256,
        "node_budget": provider.node_budget,
        "options": {"Threads": 1, "Hash": 64, "UCI_ShowWDL": True, "Skill Level": 20},
    }


def _environment_payload() -> dict[str, Any]:
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "commit": _git("rev-parse", "HEAD"),
        "working_tree_clean": not bool(_git("status", "--porcelain")),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
    }


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def _rate(items: list[dict[str, Any]], key: str) -> float:
    return sum(item[key] is True for item in items) / len(items) if items else 0.0


def _write_payload(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
