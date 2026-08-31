from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path

import pytest

from gemmafischer.accuracy_eval import (
    load_accuracy_positions,
    run_constructed_accuracy_benchmark,
    run_lichess_puzzle_accuracy_benchmark,
)


def test_constructed_fixture_references_are_rule_checkable() -> None:
    positions = load_accuracy_positions(Path("data/evaluation/accuracy_positions.jsonl"))
    assert len(positions) == 8
    assert {position.license for position in positions} == {"CC0-1.0"}
    assert sum(position.expected_terminal_reason is not None for position in positions) == 3
    assert (
        sum(
            position.reference_method == "single legal move enumeration"
            for position in positions
        )
        == 2
    )


def test_loader_rejects_an_illegal_expected_move(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "bad",
                "fen": "7k/8/5KQ1/8/8/8/8/8 w - - 0 1",
                "category": "mate",
                "expected_top_moves": ["a1a8"],
                "expected_terminal_reason": None,
                "score_expectation": "mate_for_side",
                "source": "constructed",
                "license": "CC0-1.0",
                "reference_method": "enumeration",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Illegal expected move"):
        load_accuracy_positions(path)


@pytest.mark.hardware
def test_constructed_benchmark_uses_real_stockfish_and_writes_evidence(tmp_path: Path) -> None:
    output = tmp_path / "accuracy.json"
    result = run_constructed_accuracy_benchmark(
        Path("data/evaluation/accuracy_positions.jsonl"),
        output,
        repeats=2,
        node_budget=10_000,
    )
    assert result["summary"] == {
        "top1_hits": 10,
        "top1_total": 10,
        "top1_rate": 1.0,
        "top3_hits": 10,
        "top3_total": 10,
        "top3_rate": 1.0,
        "legality_rate": 1.0,
        "terminal_correctness_rate": 1.0,
        "score_mate_consistency_rate": 1.0,
        "repeatability_rate": 1.0,
    }
    assert len(result["engine"]["binary_sha256"]) == 64
    assert len(result["fixture_sha256"]) == 64
    assert json.loads(output.read_text(encoding="utf-8"))["benchmark"] == (
        "constructed-chess-accuracy"
    )


@pytest.mark.hardware
def test_lichess_benchmark_streams_a_deterministic_held_out_sample(tmp_path: Path) -> None:
    zstandard = pytest.importorskip("zstandard")
    eligible_ids: list[str] = []
    index = 0
    while len(eligible_ids) < 3:
        puzzle_id = f"fixture-{index}"
        lineage = f"lichess-puzzle:{puzzle_id}"
        if int(hashlib.sha256(lineage.encode()).hexdigest()[:8], 16) % 10 == 0:
            eligible_ids.append(puzzle_id)
        index += 1
    rows = [
        {
            "PuzzleId": puzzle_id,
            "FEN": "7k/p7/5KQ1/8/8/8/8/8 b - - 0 1",
            "Moves": "a7a6 g6g7",
            "Rating": str(1200 + offset),
            "Themes": "mate mateIn1",
        }
        for offset, puzzle_id in enumerate(eligible_ids)
    ]
    csv_buffer = io.StringIO()
    writer = csv.DictWriter(csv_buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    archive = tmp_path / "puzzles.csv.zst"
    archive.write_bytes(zstandard.ZstdCompressor().compress(csv_buffer.getvalue().encode()))
    archive_hash = hashlib.sha256(archive.read_bytes()).hexdigest()
    manifest = tmp_path / "sources.json"
    manifest.write_text(
        json.dumps(
            {
                "sources": [
                    {
                        "id": "lichess-puzzles-2026-08-02",
                        "sha256": archive_hash,
                        "license": "CC0-1.0",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "result.json"
    result = run_lichess_puzzle_accuracy_benchmark(
        archive,
        manifest,
        output,
        sample_size=2,
        node_budget=10_000,
        selection_seed="test-seed",
    )
    assert result["summary"]["top1_rate"] == 1.0
    assert result["summary"]["top3_rate"] == 1.0
    assert result["summary"]["legality_rate"] == 1.0
    assert result["summary"]["categories"]["mateIn1"]["count"] == 2
    assert result["selection"]["eligible_evaluation_rows"] == 3
    assert result["source"]["archive_sha256"] == archive_hash
    assert output.is_file()


def test_lichess_benchmark_fails_clearly_when_archive_is_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="archive not found"):
        run_lichess_puzzle_accuracy_benchmark(
            tmp_path / "missing.zst",
            Path("data/sources.json"),
            tmp_path / "result.json",
        )
