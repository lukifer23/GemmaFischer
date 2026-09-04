import json
from pathlib import Path

import pytest

from gemmafischer.resources import bundled_path
from gemmafischer.runtime import (
    GemmaRuntime,
    ModelUnavailable,
    extract_json_array,
    inspect_adapter_assets,
    lesson_selection_target,
    parse_lesson_selection,
)


def test_extract_json_array_accepts_markdown_fence() -> None:
    output = '<|channel>final\n```json\n[{"kind":"guidance"}]\n```'
    assert extract_json_array(output) == [{"kind": "guidance"}]


def test_bundled_runtime_resources_resolve_from_checkout() -> None:
    assert bundled_path("assets/model-manifest.json").is_file()
    assert bundled_path("data/evaluation/diagnostic_positions.jsonl").is_file()


def test_runtime_adapter_requires_one_configured_hash_pinned_checkpoint(
    tmp_path: Path,
) -> None:
    (tmp_path / "adapter_config.json").write_text("{}\n", encoding="utf-8")
    weights = tmp_path / "adapters.safetensors"
    weights.write_bytes(b"adapter")

    inspected = inspect_adapter_assets(tmp_path)

    assert inspected["weights"] == "adapters.safetensors"
    assert len(str(inspected["sha256"])) == 64
    with pytest.raises(ModelUnavailable, match="hash does not match"):
        inspect_adapter_assets(tmp_path, "0" * 64)
    (tmp_path / "duplicate.safetensors").write_bytes(b"duplicate")
    with pytest.raises(ModelUnavailable, match="exactly one"):
        inspect_adapter_assets(tmp_path)


def test_extract_json_array_rejects_missing_payload() -> None:
    with pytest.raises(ValueError, match="did not contain"):
        extract_json_array("no structured result")


def test_runtime_accepts_only_exact_lesson_selection_ids() -> None:
    runtime = object.__new__(GemmaRuntime)
    runtime._model = object()

    class Tokenizer:
        def apply_chat_template(self, *args: object, **kwargs: object) -> str:
            return "prompt"

    runtime._tokenizer = Tokenizer()
    from gemmafischer.domain import EngineEvidence, RatingBucket

    evidence = EngineEvidence.model_validate(
        {
            "position_id": "position",
            "fen": "8/8/8/8/8/8/4K3/6k1 w - - 0 1",
            "side_to_move": "white",
            "engine": {
                "name": "fixture",
                "binary_sha256": "hash",
                "options": {},
                "node_budget": 1,
            },
            "candidates": [
                {
                    "evidence_id": "candidate",
                    "rank": 1,
                    "move_uci": "e2e3",
                    "move_san": "Ke3",
                    "score_cp": 0,
                    "nodes": 1,
                    "pv_uci": ["e2e3"],
                }
            ],
            "board_facts": [],
        }
    )
    target = lesson_selection_target(evidence, RatingBucket.CLUB)
    runtime._generate = lambda *args, **kwargs: json.dumps(target)

    selection = runtime.select_claims(evidence, RatingBucket.CLUB)

    assert len(selection.claims) >= 2
    assert selection.question_template_id == "find-strongest-move"

    target["claim_ids"][0] = "invented"
    with pytest.raises(ValueError, match="unknown claim"):
        parse_lesson_selection(json.dumps(target), evidence, RatingBucket.CLUB)
