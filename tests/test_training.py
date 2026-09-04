import json
from pathlib import Path

import pytest

from gemmafischer.domain import EngineEvidence, RatingBucket
from gemmafischer.labeling import (
    adjudicate_label_responses,
    apply_human_gold,
    export_label_packet,
    validate_label_responses,
)
from gemmafischer.mlx_training import sanitize_gemma4_shared_kv_weights
from gemmafischer.question_eval import freeze_question_cases, load_question_cases
from gemmafischer.runtime import (
    LESSON_SELECTION_CONTRACT_VERSION,
    LESSON_SELECTION_SYSTEM_PROMPT,
    lesson_selection_prompt,
    lesson_selection_target,
)
from gemmafischer.training import (
    package_training_artifact,
    prepare_mlx_dataset,
    run_mlx_sft,
    sha256_file,
    training_preflight,
    validate_training_preflight,
)
from gemmafischer.training_eval import freeze_error_taxonomy


def _record(record_id: str, split: str) -> dict[str, object]:
    evidence = EngineEvidence.model_validate(
        {
            "position_id": f"position-{record_id}",
            "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "side_to_move": "white",
            "engine": {
                "name": "Stockfish 18",
                "binary_sha256": "a" * 64,
                "options": {},
                "node_budget": 1,
            },
            "candidates": [
                {
                    "evidence_id": f"candidate-{record_id}",
                    "rank": 1,
                    "move_uci": "e2e4",
                    "move_san": "e4",
                    "score_cp": 0,
                    "nodes": 1,
                    "pv_uci": ["e2e4", "e7e5"],
                }
            ],
            "board_facts": [],
        }
    )
    rating = RatingBucket.CLUB
    target = lesson_selection_target(evidence, rating)
    return {
        "record_id": record_id,
        "task": LESSON_SELECTION_CONTRACT_VERSION,
        "system_prompt": LESSON_SELECTION_SYSTEM_PROMPT,
        "prompt": lesson_selection_prompt(evidence, rating),
        "response": json.dumps(target, separators=(",", ":")),
        "input": evidence.model_dump(mode="json"),
        "target": target,
        "license": "CC0-1.0",
        "meta": {"rating_bucket": rating.value, "split": split, "source": "test-source"},
    }


def _write_partitions(root: Path) -> None:
    for name in ("train", "validation", "final_test"):
        (root / f"{name}.jsonl").write_text(
            json.dumps(_record(name, name)) + "\n", encoding="utf-8"
        )


def test_prepare_mlx_dataset_preserves_valid_chat_contract(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "prepared"
    source.mkdir()
    _write_partitions(source)

    receipt = prepare_mlx_dataset(source, output)

    assert receipt["counts"] == {"train": 1, "valid": 1}
    assert receipt["trainer_files"] == ["train.jsonl", "valid.jsonl"]
    assert not (output / "test.jsonl").exists()
    assert receipt["supervision_authority"] == "stockfish-deterministic-v2"
    assert receipt["human_gold_sha256"] is None
    row = json.loads((output / "train.jsonl").read_text(encoding="utf-8"))
    assert [message["role"] for message in row["messages"]] == [
        "system",
        "user",
        "assistant",
    ]


def test_training_requires_full_context_and_exact_resume_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("gemmafischer.training.platform.system", lambda: "Darwin")
    monkeypatch.setattr("gemmafischer.training.platform.machine", lambda: "arm64")
    arguments = {
        "model_path": tmp_path / "model",
        "data_path": tmp_path / "data",
        "adapter_path": tmp_path / "adapter",
        "receipt_path": tmp_path / "receipt.json",
        "iterations": 7,
        "smoke": True,
        "config_path": tmp_path / "config.yaml",
    }
    with pytest.raises(ValueError, match="4096-token"):
        run_mlx_sft(max_seq_length=1024, **arguments)
    with pytest.raises(ValueError, match="exactly one existing"):
        run_mlx_sft(max_seq_length=4096, resume=True, **arguments)


def test_gemma4_shared_kv_sanitizer_drops_only_architecture_unused_weights() -> None:
    weights = {
        "language_model.model.layers.14.self_attn.k_proj.weight": "active-k",
        "language_model.model.layers.15.self_attn.k_proj.weight": "unused-k",
        "language_model.model.layers.15.self_attn.v_proj.weight": "unused-v",
        "language_model.model.layers.15.self_attn.k_norm.weight": "unused-norm",
        "language_model.model.layers.15.self_attn.q_proj.weight": "active-q",
        "language_model.model.layers.34.mlp.down_proj.weight": "active-mlp",
    }

    sanitized = sanitize_gemma4_shared_kv_weights(
        weights, num_hidden_layers=35, num_kv_shared_layers=20
    )

    assert sanitized == {
        "language_model.model.layers.14.self_attn.k_proj.weight": "active-k",
        "language_model.model.layers.15.self_attn.q_proj.weight": "active-q",
        "language_model.model.layers.34.mlp.down_proj.weight": "active-mlp",
    }


def test_prepare_mlx_dataset_rejects_prompt_drift(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    _write_partitions(source)
    row = _record("train", "train")
    row["prompt"] = "drifted"
    (source / "train.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="prompt drift"):
        prepare_mlx_dataset(source, tmp_path / "prepared")


def test_training_preflight_is_bound_to_unchanged_model_and_data(tmp_path: Path) -> None:
    model = tmp_path / "model"
    data = tmp_path / "data"
    model.mkdir()
    data.mkdir()
    weight = model / "model.safetensors"
    receipt = data / "dataset-receipt.json"
    config = tmp_path / "mlx-lora.yaml"
    weight.write_bytes(b"native-weight")
    receipt.write_text('{"status":"passed"}\n', encoding="utf-8")
    config.write_text("lora_parameters: {}\n", encoding="utf-8")
    training_file = data / "train.jsonl"
    valid_file = data / "valid.jsonl"
    for path in (training_file, valid_file):
        path.write_text('{"messages":[]}\n', encoding="utf-8")
    preflight = tmp_path / "preflight.json"
    preflight.write_text(
        json.dumps(
            {
                "smoke_ready": True,
                "production_authorized": False,
                "model_path": str(model.resolve()),
                "prepared_data": str(data.resolve()),
                "prepared_data_receipt_sha256": sha256_file(receipt),
                "model_files": {
                    "model.safetensors": {
                        "expected": sha256_file(weight),
                        "actual": sha256_file(weight),
                        "passed": True,
                    }
                },
                "training_config": str(config.resolve()),
                "training_config_sha256": sha256_file(config),
                "prepared_data_files": {
                    path.name: {"expected": sha256_file(path)}
                    for path in (training_file, valid_file)
                },
            }
        ),
        encoding="utf-8",
    )

    validate_training_preflight(preflight, model, data, config, production=False)
    with pytest.raises(ValueError, match="not explicitly authorized"):
        validate_training_preflight(preflight, model, data, config, production=True)
    with pytest.raises(ValueError, match="model path does not match"):
        validate_training_preflight(
            preflight, tmp_path / "other-model", data, config, production=False
        )
    weight.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed after preflight"):
        validate_training_preflight(preflight, model, data, config, production=False)
    weight.write_bytes(b"native-weight")
    training_file.write_text('{"messages":["changed"]}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="Prepared training file changed"):
        validate_training_preflight(preflight, model, data, config, production=False)
    training_file.write_text('{"messages":[]}\n', encoding="utf-8")
    receipt.write_text('{"status":"changed"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="data changed after preflight"):
        validate_training_preflight(preflight, model, data, config, production=False)


def test_error_taxonomy_is_bound_to_observed_baseline(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps(
            {
                "status": "passed",
                "task": "lesson-selection-2.0",
                "record_count": 2,
                "results": [
                    {"record_id": "one", "error_codes": ["contract_invalid"]},
                    {"record_id": "two", "error_codes": []},
                ],
            }
        ),
        encoding="utf-8",
    )

    result = freeze_error_taxonomy(baseline, tmp_path / "taxonomy.json")

    assert result["status"] == "passed"
    assert result["total_error_assignments"] == 1
    assert result["categories"][0]["code"] == "contract_invalid"


def test_package_training_artifact_retains_exactly_one_adapter_and_receipt(
    tmp_path: Path,
) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    weight = adapter / "adapter.safetensors"
    weight.write_bytes(b"adapter-package-contract-fixture")
    receipt = tmp_path / "smoke-receipt.json"
    receipt.write_text('{"status":"passed"}\n', encoding="utf-8")

    result = package_training_artifact(adapter, [receipt], tmp_path / "release" / "adapter.tar.gz")

    assert result["members"] == {
        "adapter.safetensors": sha256_file(weight),
        "smoke-receipt.json": sha256_file(receipt),
    }
    assert Path(result["path"]).is_file()


def test_training_preflight_verifies_every_bound_input(tmp_path: Path) -> None:
    audit = tmp_path / "audit.json"
    model = tmp_path / "model"
    data = tmp_path / "data"
    model.mkdir()
    data.mkdir()
    weight = model / "model.safetensors"
    weight.write_bytes(b"native")
    config = tmp_path / "recipe.yaml"
    config.write_text("lora_parameters: {}\n", encoding="utf-8")
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps(
            {
                "status": "passed",
                "task": "lesson-selection-2.0",
                "dataset_split": "validation",
                "record_count": 1,
            }
        ),
        encoding="utf-8",
    )
    taxonomy = tmp_path / "taxonomy.json"
    taxonomy.write_text(
        json.dumps(
            {
                "status": "passed",
                "task": "lesson-selection-2.0",
                "record_count": 1,
                "baseline_sha256": sha256_file(baseline),
                "categories": [{"code": "contract_invalid", "count": 1}],
            }
        ),
        encoding="utf-8",
    )
    source_hashes = {
        "train.jsonl": "a" * 64,
        "validation.jsonl": "b" * 64,
        "final_test.jsonl": "c" * 64,
    }
    files = {}
    for name in ("train.jsonl", "valid.jsonl"):
        path = data / name
        path.write_text('{"messages":[]}\n', encoding="utf-8")
        files[name] = sha256_file(path)
    (data / "dataset-receipt.json").write_text(
        json.dumps(
            {
                "task": "lesson-selection-2.0",
                "sha256": files,
                "source_sha256": source_hashes,
                "supervision_authority": "stockfish-deterministic-v2",
                "human_gold_sha256": None,
            }
        ),
        encoding="utf-8",
    )
    audit.write_text(
        json.dumps(
            {
                "schema_version": "2.0",
                "status": "passed",
                "gate": {"ready_for_training": True},
                "training": {
                    "totals": {"records": 1},
                    "files": [
                        {
                            "path": "data/derived/v2/train.jsonl",
                            "sha256": source_hashes["train.jsonl"],
                        }
                    ],
                },
                "validation": {
                    "totals": {"records": 1},
                    "files": [
                        {
                            "path": "data/derived/v2/validation.jsonl",
                            "sha256": source_hashes["validation.jsonl"],
                        }
                    ],
                },
                "evaluation": {
                    "totals": {"records": 1},
                    "files": [
                        {
                            "path": "data/derived/v2/final_test.jsonl",
                            "sha256": source_hashes["final_test.jsonl"],
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "hardware": {"minimum_memory_bytes": 1, "minimum_free_bytes": 1},
                "toolchain": {"packages": {"test_tool": "1.0"}},
                "model": {
                    "native_base_model": "provider/model",
                    "revision": "revision",
                    "weight_sha256": {weight.name: sha256_file(weight)},
                    "inference_quant_is_training_source": False,
                },
                "training": {"recipe_sha256": sha256_file(config)},
                "data": {
                    "minimum_train": 1,
                    "minimum_validation": 1,
                    "minimum_final_test": 1,
                    "baseline_minimum": 1,
                    "minimum_repeated_error_count": 1,
                    "training_supervision_authority": "stockfish-deterministic-v2",
                    "human_review_policy": "optional_pedagogy_claim_only",
                    "human_gold_minimum": 1,
                    "reviewers_required": 2,
                    "agreement_minimum": 0.67,
                    "human_rubric_minimum_mean": 3.0,
                    "human_harmful_omission_maximum": 0,
                },
                "evidence": {
                    "frozen_baseline": str(baseline),
                    "error_taxonomy": str(taxonomy),
                    "frozen_human_review": None,
                },
                "authorization": {"smoke": True, "production": False},
            }
        ),
        encoding="utf-8",
    )

    result = training_preflight(
        manifest,
        audit,
        model,
        data,
        tmp_path / "preflight.json",
        config,
        hardware={"system": "Darwin", "machine": "arm64", "memory_bytes": 18},
        installed_versions={"test_tool": "1.0"},
    )

    assert result["smoke_ready"] is True
    assert result["blockers"] == []
    assert result["prepared_data_passed"] is True
    assert result["prepared_human_gold_matches"] is None
    assert result["prepared_source_hashes_match_audit"] is True


def test_freeze_question_eval_uses_only_valid_final_test_records(tmp_path: Path) -> None:
    dataset = tmp_path / "final_test.jsonl"
    dataset.write_text(json.dumps(_record("final", "final_test")) + "\n", encoding="utf-8")
    output = tmp_path / "questions.jsonl"

    receipt = freeze_question_cases(dataset, output, limit=1)
    cases = load_question_cases(output)

    assert receipt["question_count"] == 1
    assert cases[0].kind == "best_move"
    assert cases[0].accepted_moves_uci == ("e2e4",)


def test_freeze_question_eval_rejects_training_partition(tmp_path: Path) -> None:
    dataset = tmp_path / "train.jsonl"
    dataset.write_text(json.dumps(_record("train", "train")) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="only consume final_test"):
        freeze_question_cases(dataset, tmp_path / "questions.jsonl", limit=1)


def test_human_label_packet_and_response_use_exact_catalog(tmp_path: Path) -> None:
    dataset = tmp_path / "train.jsonl"
    record = _record("one", "train")
    dataset.write_text(json.dumps(record) + "\n", encoding="utf-8")
    packet = tmp_path / "packet.json"
    export_label_packet(dataset, packet)
    response = tmp_path / "responses.jsonl"
    rubric = {
        "correctness": 5,
        "clarity": 5,
        "relevance": 5,
        "rating_fit": 5,
        "actionability": 5,
        "question_usefulness": 5,
        "hint_usefulness": 5,
        "harmful_omission": False,
    }
    response.write_text(
        "\n".join(
            json.dumps(
                {
                    "record_id": "one",
                    "reviewer_id": reviewer,
                    "selection": record["target"],
                    "rubric": rubric,
                }
            )
            for reviewer in ("reviewer-a", "reviewer-b")
        )
        + "\n",
        encoding="utf-8",
    )

    result = validate_label_responses(dataset, response, tmp_path / "validated.json")

    assert result["status"] == "passed"
    assert result["reviewer_ids"] == ["reviewer-a", "reviewer-b"]
    assert result["adjudication_complete"] is True


def test_human_label_validation_refuses_incomplete_reviewer_coverage(tmp_path: Path) -> None:
    dataset = tmp_path / "train.jsonl"
    record = _record("one", "train")
    dataset.write_text(json.dumps(record) + "\n", encoding="utf-8")
    rubric = {
        **{
            field: 5
            for field in (
                "correctness",
                "clarity",
                "relevance",
                "rating_fit",
                "actionability",
                "question_usefulness",
                "hint_usefulness",
            )
        },
        "harmful_omission": False,
    }
    responses = tmp_path / "responses.jsonl"
    responses.write_text(
        json.dumps(
            {
                "record_id": "one",
                "reviewer_id": "only-reviewer",
                "selection": record["target"],
                "rubric": rubric,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Exactly two independent reviewers"):
        validate_label_responses(dataset, responses, tmp_path / "invalid.json")


def test_human_label_disagreement_requires_independent_adjudication(tmp_path: Path) -> None:
    dataset = tmp_path / "train.jsonl"
    record = _record("one", "train")
    dataset.write_text(json.dumps(record) + "\n", encoding="utf-8")
    rubric = {
        **{
            field: 4
            for field in (
                "correctness",
                "clarity",
                "relevance",
                "rating_fit",
                "actionability",
                "question_usefulness",
                "hint_usefulness",
            )
        },
        "harmful_omission": False,
    }
    alternate = dict(record["target"])
    alternate["question_template_id"] = "explain-engine-choice"
    responses = tmp_path / "responses.jsonl"
    responses.write_text(
        "\n".join(
            (
                json.dumps(
                    {
                        "record_id": "one",
                        "reviewer_id": "a",
                        "selection": record["target"],
                        "rubric": rubric,
                    }
                ),
                json.dumps(
                    {
                        "record_id": "one",
                        "reviewer_id": "b",
                        "selection": alternate,
                        "rubric": rubric,
                    }
                ),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    validation_path = tmp_path / "validation.json"
    validation = validate_label_responses(dataset, responses, validation_path)
    assert validation["status"] == "needs_adjudication"

    adjudications = tmp_path / "adjudications.jsonl"
    adjudications.write_text(
        json.dumps(
            {
                "record_id": "one",
                "adjudicator_id": "c",
                "selection": record["target"],
                "rubric": rubric,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    human_gold = tmp_path / "human-gold.json"
    result = adjudicate_label_responses(dataset, validation_path, adjudications, human_gold)
    assert result["status"] == "passed"
    assert result["adjudication_complete"] is True
    assert result["adjudicator_ids"] == ["c"]

    source = tmp_path / "source"
    source.mkdir()
    (source / "train.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    for partition in ("validation", "final_test"):
        (source / f"{partition}.jsonl").write_text(
            json.dumps(_record(partition, partition)) + "\n", encoding="utf-8"
        )
    reviewed = tmp_path / "reviewed"
    application = apply_human_gold(source, human_gold, reviewed)
    prepared = tmp_path / "prepared-reviewed"
    receipt = prepare_mlx_dataset(reviewed, prepared, human_gold_path=human_gold)

    assert application["human_labeled_records"] == 1
    assert receipt["human_gold_sha256"] == sha256_file(human_gold)
