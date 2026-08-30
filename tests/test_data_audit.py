import json
from pathlib import Path

from gemmafischer.data_audit import audit_data

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
EVAL_FEN = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"


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
                "question": f"FEN: {EVAL_FEN}\nWhat is best?",
                "response": "e7e5",
                "license": "CC0-1.0",
                "meta": {
                    "fen": EVAL_FEN,
                    "source": "fixture",
                    "best_move": "e7e5",
                },
            }
        ],
    )

    result = audit_data(
        [training],
        [evaluation],
        tmp_path / "audit.json",
        minimum_training_records=1,
        minimum_evaluation_records=1,
    )

    assert result["status"] == "passed"
    assert result["gate"]["ready_for_training"] is True


def test_audit_blocks_empty_inputs(tmp_path: Path) -> None:
    result = audit_data([], [], tmp_path / "audit.json")

    assert result["status"] == "blocked"
    assert result["gate"]["ready_for_training"] is False


def test_duplicate_categories_do_not_double_count_within_file(tmp_path: Path) -> None:
    training = tmp_path / "training.jsonl"
    evaluation = tmp_path / "evaluation.jsonl"
    record = {
        "task": "engine_uci",
        "prompt": f"FEN: {START_FEN}",
        "response": "e2e4",
        "license": "CC0-1.0",
        "meta": {"fen": START_FEN, "source": "fixture", "best_move": "e2e4"},
    }
    write_jsonl(training, [record, record])
    write_jsonl(
        evaluation,
        [{"question": "isolated", "license": "CC0-1.0", "meta": {"source": "fixture"}}],
    )

    result = audit_data([training], [evaluation], tmp_path / "audit.json")

    totals = result["training"]["totals"]
    assert totals["duplicates_within_file"] == 1
    assert totals.get("duplicates_across_files", 0) == 0
