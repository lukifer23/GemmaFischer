import json
from pathlib import Path

from gemmafischer.data_audit import audit_data

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def test_audit_blocks_illegal_unlicensed_and_leaked_training_data(tmp_path: Path) -> None:
    training = tmp_path / "training.jsonl"
    evaluation = tmp_path / "evaluation.jsonl"
    output = tmp_path / "audit.json"
    write_jsonl(
        training,
        [
            {
                "task": "engine_uci",
                "prompt": f"FEN: {START_FEN}",
                "response": "e2e5",
                "meta": {"fen": START_FEN, "source": "fixture", "best_move": "e2e5"},
            }
        ],
    )
    write_jsonl(
        evaluation,
        [{"question": f"FEN: {START_FEN}\nWhat is best?", "license": "CC0-1.0"}],
    )

    result = audit_data([training], [evaluation], output)

    assert result["status"] == "blocked"
    assert result["training"]["totals"]["illegal_best_moves"] == 1
    assert result["training"]["totals"]["missing_license"] == 1
    assert result["cross_dataset"]["train_evaluation_fen_overlap"] == 1
    assert output.exists()


def test_audit_passes_clean_isolated_fixture(tmp_path: Path) -> None:
    training = tmp_path / "training.jsonl"
    evaluation = tmp_path / "evaluation.jsonl"
    write_jsonl(
        training,
        [
            {
                "task": "engine_uci",
                "prompt": f"FEN: {START_FEN}",
                "response": "e2e4",
                "license": "CC0-1.0",
                "meta": {"fen": START_FEN, "source": "fixture", "best_move": "e2e4"},
            }
        ],
    )
    write_jsonl(
        evaluation,
        [
            {
                "question": "How should a beginner think about development?",
                "license": "CC0-1.0",
                "meta": {"source": "fixture"},
            }
        ],
    )

    result = audit_data([training], [evaluation], tmp_path / "audit.json")

    assert result["status"] == "passed"
    assert result["gate"]["ready_for_training"] is True
