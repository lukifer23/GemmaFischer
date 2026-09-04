from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path

import pytest

from gemmafischer.service import AnalysisService
from gemmafischer.storage import AnalysisStore
from gemmafischer.study import decision_positions, parse_import
from gemmafischer.study_domain import (
    AttemptOutcome,
    LearningMomentPrivate,
    LearningMomentView,
    PGNImportRequest,
    PracticeAttemptRequest,
    PracticeAttemptView,
    PracticePhase,
    ReviewCard,
    StudyJobState,
)

FOOLS_MATE = """[Event "Club game"]
[White "Alice"]
[Black "Bob"]
[Result "0-1"]

1. f3 e5 2. g4 Qh4# 0-1
"""


def test_pgn_import_builds_exact_player_decision_ledger() -> None:
    game = parse_import(PGNImportRequest(pgn=FOOLS_MATE, player_name="alice"))

    assert game.perspective == "white"
    assert game.moves_uci == ("f2f3", "e7e5", "g2g4", "d8h4")
    assert [item[0] for item in decision_positions(game)] == [1, 3]
    assert game.date is None


@pytest.mark.parametrize(
    ("import_request", "message"),
    [
        (PGNImportRequest(pgn=FOOLS_MATE, player_name="Nobody"), "exactly one"),
        (
            PGNImportRequest(pgn=f"{FOOLS_MATE}\n{FOOLS_MATE}", player_name="Alice"),
            "exactly one game",
        ),
    ],
)
def test_pgn_import_rejects_ambiguous_or_multiple_games(
    import_request: PGNImportRequest, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        parse_import(import_request)


def test_study_records_cascade_with_attempts_and_review_cards(tmp_path: Path) -> None:
    store = AnalysisStore(tmp_path / "history.sqlite3")
    game = parse_import(PGNImportRequest(pgn=FOOLS_MATE, player_name="Alice"))
    service = AnalysisService(history_path=tmp_path / "service.sqlite3")
    try:
        job = service.submit_study(PGNImportRequest(pgn=FOOLS_MATE, player_name="Alice"))
        service.cancel_study(job.job_id)
        job = service.get_study(job.job_id)
        assert job is not None
    finally:
        service.close()
    job = job.model_copy(update={"game": game, "state": StudyJobState.READY})
    store.save_study_job(job, create=True)
    view = LearningMomentView(
        moment_id="moment-1",
        rank=1,
        source_ply=3,
        fen=game.initial_fen,
        played_move_uci="g2g4",
        played_move_san="g4",
        mate_loss=True,
        reason_codes=("mate_loss",),
    )
    store.replace_learning_moments(
        job.job_id,
        [
            LearningMomentPrivate(
                view=view,
                preferred_move_uci="e2e4",
                preferred_move_san="e4",
                evidence_json="{}",
            )
        ],
    )
    now = datetime.now(UTC)
    attempt = PracticeAttemptView(
        attempt_id="attempt-1",
        moment_id=view.moment_id,
        phase=PracticePhase.ORIGINAL,
        attempt_number=1,
        submitted_move_uci="e2e4",
        outcome=AttemptOutcome.CORRECT,
        hint_used=False,
        created_at=now,
    )
    card = ReviewCard(
        job_id=job.job_id,
        moment_id=view.moment_id,
        moment=view,
        concept_key="calculation",
        due_at=now,
        interval_days=1,
        successful_delayed_reviews=0,
        lapses=0,
    )
    store.save_attempt_and_card(attempt, card, None)

    assert store.progress_summary(now.isoformat()).attempts == 1
    assert store.practice_statuses(job.job_id) == {view.moment_id: "scheduled"}
    assert store.delete_study_job(job.job_id)
    assert store.get_learning_moment(view.moment_id) is None
    assert store.attempts_for_moment(view.moment_id) == ()


@pytest.mark.hardware
def test_real_stockfish_study_reaches_ready_with_ranked_moment(tmp_path: Path) -> None:
    service = AnalysisService(node_budget=1_000, history_path=tmp_path / "history.sqlite3")
    try:
        job = service.submit_study(PGNImportRequest(pgn=FOOLS_MATE, player_name="Alice"))
        deadline = time.monotonic() + 10
        while job.state not in {StudyJobState.READY, StudyJobState.FAILED}:
            assert time.monotonic() < deadline
            time.sleep(0.02)
            refreshed = service.get_study(job.job_id)
            assert refreshed is not None
            job = refreshed

        assert job.state is StudyJobState.READY
        assert job.moments
        assert [moment.rank for moment in job.moments] == list(
            range(1, len(job.moments) + 1)
        )
        assert all(moment.played_move_uci != "d8h4" for moment in job.moments)
    finally:
        service.close()


@pytest.mark.hardware
def test_first_miss_stays_hidden_then_retry_reveals_and_schedules(tmp_path: Path) -> None:
    service = AnalysisService(node_budget=2_000, history_path=tmp_path / "history.sqlite3")
    try:
        job = service.submit_study(PGNImportRequest(pgn=FOOLS_MATE, player_name="Alice"))
        deadline = time.monotonic() + 10
        while job.state not in {StudyJobState.READY, StudyJobState.FAILED}:
            assert time.monotonic() < deadline
            time.sleep(0.02)
            refreshed = service.get_study(job.job_id)
            assert refreshed is not None
            job = refreshed
        moment = next(item for item in job.moments if item.source_ply == 3)
        original_request = PracticeAttemptRequest(
            expected_revision=job.revision,
            phase=PracticePhase.ORIGINAL,
            move_uci=moment.played_move_uci,
        )
        first = service.submit_practice_attempt(
            job.job_id,
            moment.moment_id,
            original_request,
            idempotency_key="first-miss-0001",
        )
        replay = service.submit_practice_attempt(
            job.job_id,
            moment.moment_id,
            original_request,
            idempotency_key="first-miss-0001",
        )
        after_miss = service.get_study(job.job_id)
        assert after_miss is not None
        missed_moment = next(
            item for item in after_miss.moments if item.moment_id == moment.moment_id
        )
        assert missed_moment.practice_status == "in_progress"
        assert service.due_reviews() == ()
        assert service.progress().learning == 0
        retry = service.submit_practice_attempt(
            job.job_id,
            moment.moment_id,
            PracticeAttemptRequest(
                expected_revision=job.revision,
                phase=PracticePhase.RETRY,
                move_uci=moment.played_move_uci,
            ),
            idempotency_key="retry-miss-0001",
        )

        assert first.outcome is AttemptOutcome.INCORRECT
        assert first.feedback is None
        assert replay.attempt_id == first.attempt_id
        assert retry.feedback is not None
        after_retry = service.get_study(job.job_id)
        assert after_retry is not None
        retried_moment = next(
            item for item in after_retry.moments if item.moment_id == moment.moment_id
        )
        assert retried_moment.practice_status == "scheduled"
        assert service.progress().attempts == 2
        assert service.progress().learning == 1
    finally:
        service.close()
