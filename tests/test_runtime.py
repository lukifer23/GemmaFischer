import pytest

from gemmafischer.domain import GuidanceClaim
from gemmafischer.resources import bundled_path
from gemmafischer.runtime import GemmaRuntime, extract_json_array


def test_extract_json_array_accepts_markdown_fence() -> None:
    output = '<|channel>final\n```json\n[{"kind":"guidance"}]\n```'
    assert extract_json_array(output) == [{"kind": "guidance"}]


def test_bundled_runtime_resources_resolve_from_checkout() -> None:
    assert bundled_path("assets/model-manifest.json").is_file()
    assert bundled_path("data/evaluation/diagnostic_positions.jsonl").is_file()


def test_extract_json_array_rejects_missing_payload() -> None:
    with pytest.raises(ValueError, match="did not contain"):
        extract_json_array("no structured result")


def test_claim_selection_keeps_valid_items_and_reports_invalid_ones() -> None:
    runtime = object.__new__(GemmaRuntime)
    runtime._generate = lambda *args, **kwargs: """[
      {"kind":"guidance","evidence_ids":[],"template_id":"calculate_forcing_moves"},
      {"kind":"line","evidence_ids":["candidate"],"candidate_id":"candidate"}
    ]"""
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

    selection = runtime.select_claims(evidence, RatingBucket.CLUB)

    assert selection.claims == (GuidanceClaim(template_id="calculate_forcing_moves"),)
    assert selection.removed_claim_codes == ("MODEL_CLAIM_SCHEMA_INVALID",)
