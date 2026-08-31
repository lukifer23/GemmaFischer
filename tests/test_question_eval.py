from __future__ import annotations

import json
from pathlib import Path

import pytest

from gemmafischer.question_eval import (
    grade_question_answer,
    load_question_cases,
    run_question_grading_qualification,
)

FIXTURES = Path("data/evaluation/test_questions.jsonl")


def test_frozen_questions_cover_rules_and_adversarial_grading() -> None:
    cases = load_question_cases(FIXTURES)
    assert len(cases) == 8
    assert {case.kind for case in cases} == {
        "mate_move",
        "only_legal_move",
        "terminal_reason",
    }
    assert all(case.source and case.license for case in cases)
    assert all(
        {example.expected_correct for example in case.grading_examples} == {False, True}
        for case in cases
    )


def test_every_frozen_grading_example_matches_its_independent_label() -> None:
    for case in load_question_cases(FIXTURES):
        for example in case.grading_examples:
            assert grade_question_answer(case, example.answer).correct is example.expected_correct


def test_grader_rejects_prose_and_incomplete_promotion_notation() -> None:
    cases = {case.id: case for case in load_question_cases(FIXTURES)}
    prose = grade_question_answer(cases["queen-mate-uci-san"], "I choose Qg7#")
    incomplete = grade_question_answer(cases["promotion-mates"], "a7a8")
    assert (prose.correct, prose.reason) == (False, "invalid_notation")
    assert (incomplete.correct, incomplete.reason) == (False, "invalid_notation")


def test_loader_rejects_an_incomplete_mate_answer_set(tmp_path: Path) -> None:
    source = next(
        json.loads(line)
        for line in FIXTURES.read_text(encoding="utf-8").splitlines()
        if '"id":"promotion-mates"' in line
    )
    source["accepted_moves_uci"] = ["a7a8q"]
    path = tmp_path / "incomplete.jsonl"
    path.write_text(json.dumps(source) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Mate-move label is incomplete"):
        load_question_cases(path)


def test_qualification_writes_raw_agreement_evidence(tmp_path: Path) -> None:
    output = tmp_path / "questions.json"
    payload = run_question_grading_qualification(FIXTURES, output)
    assert payload["status"] == "passed"
    assert payload["summary"] == {
        "case_count": 8,
        "grading_example_count": 33,
        "grading_agreement_rate": 1.0,
        "correct_examples": 18,
        "incorrect_examples": 15,
    }
    assert payload["question_generation_status"] == "fixture-defined"
    assert payload["answer_contract"] == "exact-uci-san-or-terminal-reason"
    assert len(payload["case_sha256"]) == 64
    assert json.loads(output.read_text(encoding="utf-8"))["results"] == payload["results"]
