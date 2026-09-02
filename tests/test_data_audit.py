import json
from pathlib import Path

from gemmafischer.data_audit import audit_data
from gemmafischer.domain import EngineEvidence, RatingBucket
from gemmafischer.runtime import (
    LESSON_SELECTION_CONTRACT_VERSION,
    LESSON_SELECTION_SYSTEM_PROMPT,
    lesson_selection_prompt,
    lesson_selection_target,
)

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
                "meta": {
                    "fen": START_FEN,
                    "source": "fixture",
                    "lineage": "fixture:start",
                    "best_move": "e2e4",
                },
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
                    "lineage": "fixture:eval",
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
        minimum_validation_records=0,
        minimum_evaluation_records=1,
        enforce_model_contract=False,
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


def test_audit_detects_clock_normalized_position_and_lineage_leakage(tmp_path: Path) -> None:
    training = tmp_path / "train.jsonl"
    final_test = tmp_path / "final_test.jsonl"
    later_clock = START_FEN.rsplit(" ", 2)[0] + " 19 44"
    write_jsonl(
        training,
        [{
            "task": "engine_uci",
            "prompt": f"FEN: {START_FEN}",
            "response": "e2e4",
            "license": "CC0-1.0",
            "meta": {
                "fen": START_FEN,
                "source": "fixture",
                "lineage": "same-game",
                "best_move": "e2e4",
            },
        }],
    )
    write_jsonl(
        final_test,
        [{
            "task": "engine_uci",
            "prompt": f"FEN: {later_clock}",
            "response": "e2e4",
            "license": "CC0-1.0",
            "meta": {
                "fen": later_clock,
                "source": "fixture",
                "lineage": "same-game",
                "best_move": "e2e4",
            },
        }],
    )

    result = audit_data(
        [training],
        [final_test],
        tmp_path / "audit.json",
        minimum_training_records=1,
        minimum_validation_records=0,
        minimum_evaluation_records=1,
        enforce_model_contract=False,
    )

    assert result["cross_dataset"]["semantic_position_overlap"] == 1
    assert result["cross_dataset"]["lineage_overlap"] == 1
    assert result["gate"]["ready_for_training"] is False


def test_audit_rejects_wrong_model_target_contract(tmp_path: Path) -> None:
    training = tmp_path / "train.jsonl"
    write_jsonl(
        training,
        [{
            "task": "grounded_lesson_plan",
            "prompt": f"FEN: {START_FEN}",
            "response": "e2e4",
            "license": "CC0-1.0",
            "meta": {
                "fen": START_FEN,
                "source": "fixture",
                "lineage": "fixture:start",
                "best_move": "e2e4",
            },
        }],
    )

    result = audit_data([training], [], tmp_path / "audit.json")

    assert result["training"]["totals"]["unsupported_tasks"] == 1
    assert result["training"]["totals"]["invalid_model_contracts"] == 1


def test_audit_accepts_exact_lesson_selection_contract(tmp_path: Path) -> None:
    fen = "8/8/8/8/8/8/4K3/6k1 w - - 0 1"
    evidence = EngineEvidence.model_validate({
        "position_id": "position",
        "fen": fen,
        "side_to_move": "white",
        "engine": {
            "name": "fixture",
            "binary_sha256": "a" * 64,
            "options": {},
            "node_budget": 1,
        },
        "candidates": [{
            "evidence_id": "candidate",
            "rank": 1,
            "move_uci": "e2e3",
            "move_san": "Ke3",
            "score_cp": 0,
            "nodes": 1,
            "pv_uci": ["e2e3"],
        }],
        "board_facts": [],
    })
    target = lesson_selection_target(evidence, RatingBucket.CLUB)
    training = tmp_path / "train.jsonl"
    write_jsonl(training, [{
        "record_id": "record",
        "task": LESSON_SELECTION_CONTRACT_VERSION,
        "system_prompt": LESSON_SELECTION_SYSTEM_PROMPT,
        "prompt": lesson_selection_prompt(evidence, RatingBucket.CLUB),
        "response": json.dumps(target, separators=(",", ":")),
        "license": "CC0-1.0",
        "input": evidence.model_dump(mode="json"),
        "target": target,
        "meta": {
            "fen": fen,
            "best_move": "e2e3",
            "solution_move": "e2e3",
            "setup_move": "g1g2",
            "move_sequence": ["g1g2", "e2e3"],
            "source": "fixture",
            "source_item_id": "one",
            "source_game_id": "game-one",
            "source_position_id": "position-one",
            "lineage": "fixture:one",
            "license": "CC0-1.0",
            "split": "train",
            "transformation": "fixture",
            "rating_bucket": RatingBucket.CLUB.value,
            "evidence_contract_version": "2.0",
            "model_contract_version": LESSON_SELECTION_CONTRACT_VERSION,
            "engine_binary_sha256": "a" * 64,
            "engine_node_budget": 1,
            "selection_method": "fixture",
        },
    }])

    result = audit_data(
        [training],
        [],
        tmp_path / "audit.json",
        minimum_training_records=1,
        minimum_validation_records=0,
        minimum_evaluation_records=0,
    )

    assert result["status"] == "passed"
    assert result["training"]["totals"].get("invalid_model_contracts", 0) == 0
