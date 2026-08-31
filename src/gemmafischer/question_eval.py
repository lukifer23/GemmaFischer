from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import chess

QuestionKind = Literal["mate_move", "only_legal_move", "terminal_reason"]


@dataclass(frozen=True)
class GradingExample:
    answer: str
    expected_correct: bool


@dataclass(frozen=True)
class QuestionCase:
    id: str
    source: str
    license: str
    fen: str
    prompt: str
    kind: QuestionKind
    accepted_moves_uci: tuple[str, ...]
    expected_terminal_reason: str | None
    grading_examples: tuple[GradingExample, ...]


@dataclass(frozen=True)
class GradeResult:
    correct: bool
    normalized_answer: str | None
    reason: Literal["accepted", "incorrect", "invalid_notation"]


def load_question_cases(path: Path) -> tuple[QuestionCase, ...]:
    cases: list[QuestionCase] = []
    seen: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
            case = QuestionCase(
                id=str(raw["id"]),
                source=str(raw["source"]),
                license=str(raw["license"]),
                fen=str(raw["fen"]),
                prompt=str(raw["prompt"]),
                kind=raw["kind"],
                accepted_moves_uci=tuple(str(value) for value in raw["accepted_moves_uci"]),
                expected_terminal_reason=(
                    str(raw["expected_terminal_reason"])
                    if raw.get("expected_terminal_reason") is not None
                    else None
                ),
                grading_examples=tuple(
                    GradingExample(
                        answer=str(example["answer"]),
                        expected_correct=bool(example["expected_correct"]),
                    )
                    for example in raw["grading_examples"]
                ),
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid question fixture at {path}:{line_number}") from exc
        _validate_case(case, path, line_number)
        if case.id in seen:
            raise ValueError(f"Duplicate question fixture ID: {case.id}")
        seen.add(case.id)
        cases.append(case)
    if not cases:
        raise ValueError(f"Question fixture file is empty: {path}")
    return tuple(cases)


def grade_question_answer(case: QuestionCase, answer: str) -> GradeResult:
    """Grade the documented exact-notation answer contract without fuzzy matching."""

    if case.kind == "terminal_reason":
        normalized = _normalize_terminal_answer(answer)
        if normalized is None:
            return GradeResult(False, None, "invalid_notation")
        correct = normalized == case.expected_terminal_reason
        return GradeResult(correct, normalized, "accepted" if correct else "incorrect")

    board = chess.Board(case.fen)
    normalized_move = _parse_move_answer(board, answer)
    if normalized_move is None:
        return GradeResult(False, None, "invalid_notation")
    correct = normalized_move in case.accepted_moves_uci
    return GradeResult(correct, normalized_move, "accepted" if correct else "incorrect")


def run_question_grading_qualification(
    case_path: Path, output_path: Path
) -> dict[str, Any]:
    """Prove the deterministic question labels and grader against frozen examples."""

    cases = load_question_cases(case_path)
    rows: list[dict[str, Any]] = []
    for case in cases:
        for index, example in enumerate(case.grading_examples, 1):
            result = grade_question_answer(case, example.answer)
            rows.append(
                {
                    "case_id": case.id,
                    "example": index,
                    "answer": example.answer,
                    "expected_correct": example.expected_correct,
                    "actual_correct": result.correct,
                    "matched": result.correct == example.expected_correct,
                    "normalized_answer": result.normalized_answer,
                    "reason": result.reason,
                }
            )
    agreement = sum(row["matched"] for row in rows) / len(rows)
    summary = {
        "case_count": len(cases),
        "grading_example_count": len(rows),
        "grading_agreement_rate": agreement,
        "correct_examples": sum(row["expected_correct"] for row in rows),
        "incorrect_examples": sum(not row["expected_correct"] for row in rows),
    }
    gate = {"required": 1.0, "actual": agreement, "passed": agreement == 1.0}
    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "benchmark": "deterministic-test-question-grading",
        "status": "passed" if gate["passed"] else "failed",
        "evidence_scope": "rules-derived-label-and-exact-notation-grader",
        "generated_at": datetime.now(UTC).isoformat(),
        "commit": _git_revision(),
        "working_tree_clean": not bool(_git_status()),
        "case_path": str(case_path),
        "case_sha256": hashlib.sha256(case_path.read_bytes()).hexdigest(),
        "question_generation_status": "fixture-defined",
        "answer_contract": "exact-uci-san-or-terminal-reason",
        "summary": summary,
        "gates": {"grading_agreement_rate": gate},
        "results": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output_path)
    return payload


def _validate_case(case: QuestionCase, path: Path, line_number: int) -> None:
    if not case.id or not case.source or not case.license or not case.prompt:
        raise ValueError(f"Question metadata cannot be empty at {path}:{line_number}")
    if case.kind not in {"mate_move", "only_legal_move", "terminal_reason"}:
        raise ValueError(f"Unknown question kind at {path}:{line_number}")
    try:
        board = chess.Board(case.fen)
    except ValueError as exc:
        raise ValueError(f"Invalid question FEN at {path}:{line_number}") from exc
    if not board.is_valid():
        raise ValueError(f"Invalid question board at {path}:{line_number}")
    outcome = board.outcome(claim_draw=False)
    actual_terminal = outcome.termination.name.lower() if outcome else None
    expected_moves = set(case.accepted_moves_uci)
    if case.kind == "terminal_reason":
        if expected_moves or case.expected_terminal_reason != actual_terminal:
            raise ValueError(f"Terminal question label mismatch at {path}:{line_number}")
    else:
        if case.expected_terminal_reason is not None or not expected_moves or outcome is not None:
            raise ValueError(f"Move question label mismatch at {path}:{line_number}")
        legal = set(board.legal_moves)
        accepted: set[chess.Move] = set()
        for value in expected_moves:
            try:
                move = chess.Move.from_uci(value)
            except ValueError as exc:
                raise ValueError(f"Invalid accepted move at {path}:{line_number}") from exc
            if move not in legal:
                raise ValueError(f"Illegal accepted move at {path}:{line_number}: {value}")
            accepted.add(move)
        if case.kind == "only_legal_move" and accepted != legal:
            raise ValueError(f"Only-legal-move label is incomplete at {path}:{line_number}")
        if case.kind == "mate_move":
            mating = set()
            for move in legal:
                child = board.copy(stack=False)
                child.push(move)
                if child.is_checkmate():
                    mating.add(move)
            if accepted != mating:
                raise ValueError(f"Mate-move label is incomplete at {path}:{line_number}")
    if not case.grading_examples:
        raise ValueError(f"Question needs grading examples at {path}:{line_number}")
    expected_results = {example.expected_correct for example in case.grading_examples}
    if expected_results != {False, True}:
        raise ValueError(f"Question needs positive and negative examples at {path}:{line_number}")


def _parse_move_answer(board: chess.Board, answer: str) -> str | None:
    value = answer.strip()
    if not value or any(character.isspace() for character in value):
        return None
    try:
        move = chess.Move.from_uci(value.lower())
        if move in board.legal_moves:
            return move.uci()
    except ValueError:
        pass
    san = value.replace("0-0-0", "O-O-O").replace("0-0", "O-O")
    try:
        return board.parse_san(san).uci()
    except ValueError:
        return None


def _normalize_terminal_answer(answer: str) -> str | None:
    value = answer.strip().lower().rstrip(".!?").replace("-", "_").replace(" ", "_")
    aliases = {
        "mate": "checkmate",
        "checkmate": "checkmate",
        "stalemate": "stalemate",
        "insufficient_material": "insufficient_material",
        "draw_by_insufficient_material": "insufficient_material",
    }
    return aliases.get(value)


def _git_revision() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def _git_status() -> str:
    return subprocess.run(
        ["git", "status", "--porcelain"], check=True, capture_output=True, text=True
    ).stdout.strip()
