import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from gemmafischer.web import MAX_REQUEST_BODY_BYTES, create_app

TOKEN = "test-capability-token"
START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def test_health_and_player_are_local_and_self_hosted() -> None:
    with TestClient(create_app(capability_token=TOKEN, node_budget=1)) as client:
        health = client.get("/api/v1/health")
        assert health.status_code == 200
        assert isinstance(health.json()["engine_available"], bool)
        assert health.json()["history_enabled"] is False
        assert "engine_path" not in health.json()
        page = client.get("/")
        assert page.status_code == 200
        assert "Explain current position" in page.text
        assert "Play the position. Understand the decision." in page.text
        assert "Engine vs engine" in page.text
        assert "https://" not in page.text


def test_health_does_not_disclose_configured_engine_path() -> None:
    private_path = "/Users/private/bin/stockfish"
    with TestClient(
        create_app(engine_path=private_path, capability_token=TOKEN, node_budget=1)
    ) as client:
        response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert private_path not in response.text
    assert "engine_path" not in response.json()


def test_request_body_limit_is_enforced_before_json_parsing() -> None:
    with TestClient(create_app(capability_token=TOKEN, node_budget=1)) as client:
        response = client.post(
            "/api/v1/sessions",
            headers={"X-GemmaFischer-Token": TOKEN, "Content-Type": "application/json"},
            content=b"x" * (MAX_REQUEST_BODY_BYTES + 1),
        )

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "REQUEST_TOO_LARGE"


def test_request_body_limit_does_not_trust_a_false_content_length() -> None:
    with TestClient(create_app(capability_token=TOKEN, node_budget=1)) as client:
        response = client.post(
            "/api/v1/sessions",
            headers={
                "X-GemmaFischer-Token": TOKEN,
                "Content-Type": "application/json",
                "Content-Length": "1",
            },
            content=b"x" * (MAX_REQUEST_BODY_BYTES + 1),
        )

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "REQUEST_TOO_LARGE"


def test_mutation_requires_capability_token() -> None:
    with TestClient(create_app(capability_token=TOKEN, node_budget=1)) as client:
        client.cookies.clear()
        response = client.post(
            "/api/v1/analyses",
            json={
                "mode": "position",
                "fen": "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
                "rating_bucket": "1400-1599",
            },
        )
        assert response.status_code == 403
        assert response.json()["error"]["code"] == "CAPABILITY_TOKEN_REQUIRED"


def test_invalid_origin_is_rejected() -> None:
    with TestClient(create_app(capability_token=TOKEN, node_budget=1)) as client:
        response = client.get("/api/v1/health", headers={"Origin": "https://attacker.example"})
        assert response.status_code == 403
        assert response.json()["error"]["code"] == "INVALID_ORIGIN"


def test_create_poll_and_idempotent_cancel() -> None:
    with TestClient(create_app(capability_token=TOKEN, node_budget=1)) as client:
        response = client.post(
            "/api/v1/analyses",
            headers={"X-GemmaFischer-Token": TOKEN},
            json={
                "mode": "position",
                "fen": "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
                "rating_bucket": "1400-1599",
            },
        )
        assert response.status_code == 202
        assert response.headers["location"].startswith("/api/v1/analyses/")
        location = response.headers["location"]
        first = client.delete(location, headers={"X-GemmaFischer-Token": TOKEN})
        second = client.delete(location, headers={"X-GemmaFischer-Token": TOKEN})
        assert first.json()["state"] == "cancelled"
        assert second.json()["state"] == "cancelled"


def test_analysis_history_survives_app_restart(tmp_path: Path) -> None:
    history_path = tmp_path / "history.sqlite3"
    app = create_app(capability_token=TOKEN, node_budget=1, history_path=history_path)
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/analyses",
            headers={"X-GemmaFischer-Token": TOKEN},
            json={
                "mode": "position",
                "fen": "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
                "rating_bucket": "1400-1599",
            },
        )
        analysis_id = response.json()["analysis_id"]
        client.delete(
            f"/api/v1/analyses/{analysis_id}",
            headers={"X-GemmaFischer-Token": TOKEN},
        )

    with TestClient(
        create_app(capability_token=TOKEN, node_budget=1, history_path=history_path)
    ) as client:
        history = client.get("/api/v1/analyses").json()
        restored = client.get(f"/api/v1/analyses/{analysis_id}")

    assert history["count"] == 1
    assert history["items"][0]["analysis_id"] == analysis_id
    assert restored.json()["state"] == "cancelled"


def test_legacy_control_plane_routes_do_not_exist() -> None:
    with TestClient(create_app(capability_token=TOKEN, node_budget=1)) as client:
        for path in (
            "/api/train/start",
            "/api/eval/stockfish",
            "/api/data/clean",
            "/api/adapters/activate",
            "/api/settings/set",
        ):
            assert client.post(path, headers={"X-GemmaFischer-Token": TOKEN}).status_code == 404


def test_board_move_rejects_illegal_move_without_starting_engine() -> None:
    with TestClient(create_app(capability_token=TOKEN, node_budget=1)) as client:
        response = client.post(
            "/api/v1/board/moves",
            headers={"X-GemmaFischer-Token": TOKEN},
            json={
                "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                "move_uci": "e2e5",
                "engine_reply": False,
                "difficulty": "club",
            },
        )
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "ILLEGAL_MOVE"


def test_legal_move_destinations_are_available_without_engine() -> None:
    with TestClient(create_app(capability_token=TOKEN, node_budget=1)) as client:
        response = client.post(
            "/api/v1/board/legal-moves",
            headers={"X-GemmaFischer-Token": TOKEN},
            json={
                "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                "from_square": "e2",
            },
        )
        assert response.status_code == 200
        assert set(response.json()["destinations"]) == {"e3", "e4"}


def test_server_owned_session_enforces_revision_and_persists(tmp_path: Path) -> None:
    history_path = tmp_path / "history.sqlite3"
    headers = {"X-GemmaFischer-Token": TOKEN}
    with TestClient(
        create_app(capability_token=TOKEN, node_budget=1, history_path=history_path)
    ) as client:
        created = client.post(
            "/api/v1/sessions",
            headers=headers,
            json={"mode": "player", "player_color": "white", "fen": START_FEN},
        )
        assert created.status_code == 201
        session = created.json()
        moved = client.post(
            f"/api/v1/sessions/{session['session_id']}/commands",
            headers=headers,
            json={"expected_revision": 0, "action": "player_move", "move_uci": "e2e4"},
        )
        assert moved.status_code == 200
        assert moved.json()["revision"] == 1
        assert moved.json()["plies"][0]["analysis_id"] is None
        replied = client.post(
            f"/api/v1/sessions/{session['session_id']}/commands",
            headers=headers,
            json={"expected_revision": 1, "action": "engine_move"},
        )
        assert replied.status_code == 200
        assert replied.json()["plies"][0]["analysis_id"]
        stale = client.post(
            f"/api/v1/sessions/{session['session_id']}/commands",
            headers=headers,
            json={"expected_revision": 0, "action": "undo"},
        )
        assert stale.status_code == 409
        assert stale.json()["error"]["code"] == "REVISION_CONFLICT"

    with TestClient(
        create_app(capability_token=TOKEN, node_budget=1, history_path=history_path)
    ) as client:
        restored = client.get(f"/api/v1/sessions/{session['session_id']}")
        assert restored.status_code == 200
        assert restored.json()["plies"][0]["move_uci"] == "e2e4"


def test_exhibition_pause_and_resume_survive_restart(tmp_path: Path) -> None:
    history_path = tmp_path / "history.sqlite3"
    headers = {"X-GemmaFischer-Token": TOKEN}
    with TestClient(
        create_app(capability_token=TOKEN, node_budget=1, history_path=history_path)
    ) as client:
        created = client.post(
            "/api/v1/sessions",
            headers=headers,
            json={"mode": "exhibition", "player_color": None, "fen": START_FEN},
        ).json()
        paused = client.post(
            f"/api/v1/sessions/{created['session_id']}/commands",
            headers=headers,
            json={"expected_revision": 0, "action": "pause"},
        )
        assert paused.status_code == 200
        assert paused.json()["status"] == "paused"

    with TestClient(
        create_app(capability_token=TOKEN, node_budget=1, history_path=history_path)
    ) as client:
        restored = client.get(f"/api/v1/sessions/{created['session_id']}")
        assert restored.json()["status"] == "paused"
        blocked = client.post(
            f"/api/v1/sessions/{created['session_id']}/commands",
            headers=headers,
            json={"expected_revision": 1, "action": "engine_move"},
        )
        assert blocked.status_code == 422
        resumed = client.post(
            f"/api/v1/sessions/{created['session_id']}/commands",
            headers=headers,
            json={"expected_revision": 1, "action": "resume"},
        )
        assert resumed.status_code == 200
        assert resumed.json()["status"] == "active"


def test_tutor_practice_is_evidence_graded_redacted_and_persistent(tmp_path: Path) -> None:
    history_path = tmp_path / "history.sqlite3"
    headers = {"X-GemmaFischer-Token": TOKEN}
    with TestClient(
        create_app(capability_token=TOKEN, node_budget=1, history_path=history_path)
    ) as client:
        session = client.post(
            "/api/v1/sessions",
            headers=headers,
            json={"mode": "player", "player_color": "white", "fen": START_FEN},
        ).json()
        moved = client.post(
            f"/api/v1/sessions/{session['session_id']}/commands",
            headers=headers,
            json={"expected_revision": 0, "action": "player_move", "move_uci": "e2e4"},
        ).json()
        replied = client.post(
            f"/api/v1/sessions/{session['session_id']}/commands",
            headers=headers,
            json={"expected_revision": moved["revision"], "action": "engine_move"},
        ).json()
        analysis_id = replied["plies"][0]["analysis_id"]
        deadline = time.monotonic() + 20
        while True:
            analysis = client.get(f"/api/v1/analyses/{analysis_id}").json()
            if analysis["state"] == "complete":
                break
            assert time.monotonic() < deadline
            time.sleep(0.01)

        created = client.post(
            f"/api/v1/sessions/{session['session_id']}/tutor",
            headers=headers,
            json={"source_analysis_id": analysis_id},
        )
        assert created.status_code == 201
        interaction = created.json()
        assert interaction["status"] == "awaiting_answer"
        assert "answer_move_uci" not in created.text
        hint = client.post(
            f"/api/v1/sessions/{session['session_id']}/tutor/"
            f"{interaction['interaction_id']}/commands",
            headers=headers,
            json={"expected_revision": 0, "action": "hint"},
        ).json()
        assert hint["hint"]
        preferred = analysis["evidence"]["candidate_set"]["candidates"][0]["move_uci"]
        answered = client.post(
            f"/api/v1/sessions/{session['session_id']}/tutor/"
            f"{interaction['interaction_id']}/commands",
            headers=headers,
            json={"expected_revision": 1, "action": "answer", "move_uci": preferred},
        )
        assert answered.status_code == 200
        answer = answered.json()
        assert answer["feedback"]["outcome"] == "matched_engine"
        option_id = answer["follow_up"]["options"][0]["option_id"]
        completed = client.post(
            f"/api/v1/sessions/{session['session_id']}/tutor/"
            f"{interaction['interaction_id']}/commands",
            headers=headers,
            json={"expected_revision": 2, "action": "follow_up", "option_id": option_id},
        ).json()
        assert completed["status"] == "complete"
        assert completed["follow_up"]["correct"] is True
        terminal = client.post(
            f"/api/v1/sessions/{session['session_id']}/tutor/"
            f"{interaction['interaction_id']}/commands",
            headers=headers,
            json={"expected_revision": 3, "action": "dismiss"},
        )
        assert terminal.status_code == 409
        assert terminal.json()["error"]["code"] == "TUTOR_STATE_CONFLICT"
        live_session = client.get(f"/api/v1/sessions/{session['session_id']}").json()
        assert live_session["fen"] == replied["fen"]

    with TestClient(
        create_app(capability_token=TOKEN, node_budget=1, history_path=history_path)
    ) as client:
        restored = client.get(
            f"/api/v1/sessions/{session['session_id']}/tutor/{interaction['interaction_id']}"
        )
        assert restored.status_code == 200
        assert restored.json()["status"] == "complete"


def test_tutor_accepts_completed_analysis_of_current_session_position(tmp_path: Path) -> None:
    headers = {"X-GemmaFischer-Token": TOKEN}
    with TestClient(
        create_app(capability_token=TOKEN, node_budget=1, history_path=tmp_path / "history.sqlite3")
    ) as client:
        session = client.post(
            "/api/v1/sessions",
            headers=headers,
            json={"mode": "player", "player_color": "white", "fen": START_FEN},
        ).json()
        analysis = client.post(
            "/api/v1/analyses",
            headers=headers,
            json={"mode": "position", "fen": session["fen"], "rating_bucket": "1400-1599"},
        ).json()
        deadline = time.monotonic() + 20
        while True:
            completed = client.get(f"/api/v1/analyses/{analysis['analysis_id']}").json()
            if completed["state"] == "complete":
                break
            assert time.monotonic() < deadline
            time.sleep(0.01)
        tutor = client.post(
            f"/api/v1/sessions/{session['session_id']}/tutor",
            headers=headers,
            json={"source_analysis_id": analysis["analysis_id"]},
        )
        assert tutor.status_code == 201
        assert tutor.json()["question"]["fen"] == session["fen"]


def test_session_preserves_exact_underpromotion_and_revision() -> None:
    headers = {"X-GemmaFischer-Token": TOKEN}
    fen = "7k/P7/8/8/8/8/8/7K w - - 0 1"
    with TestClient(create_app(capability_token=TOKEN, node_budget=1)) as client:
        created = client.post(
            "/api/v1/sessions",
            headers=headers,
            json={"mode": "player", "player_color": "white", "fen": fen},
        ).json()
        promoted = client.post(
            f"/api/v1/sessions/{created['session_id']}/commands",
            headers=headers,
            json={"expected_revision": 0, "action": "player_move", "move_uci": "a7a8n"},
        )

    assert promoted.status_code == 200
    assert promoted.json()["revision"] == 1
    assert promoted.json()["plies"][0]["move_uci"] == "a7a8n"
    assert promoted.json()["fen"].startswith("N6k/")


@pytest.mark.hardware
def test_health_stays_responsive_during_real_engine_session_command() -> None:
    headers = {"X-GemmaFischer-Token": TOKEN}
    app = create_app(capability_token=TOKEN, node_budget=1_000_000)
    with TestClient(app) as client:
        created = client.post(
            "/api/v1/sessions",
            headers=headers,
            json={
                "mode": "exhibition",
                "player_color": None,
                "fen": START_FEN,
                "white_difficulty": "strong",
                "black_difficulty": "strong",
            },
        ).json()
        with ThreadPoolExecutor(max_workers=1) as executor:
            command = executor.submit(
                client.post,
                f"/api/v1/sessions/{created['session_id']}/commands",
                headers=headers,
                json={"expected_revision": 0, "action": "engine_move"},
            )
            deadline = time.monotonic() + 5
            while True:
                provider = app.state.service._provider
                active = provider.operation_status() if provider else None
                if active is not None and active.kind == "gameplay":
                    break
                assert time.monotonic() < deadline
                time.sleep(0.005)

            latencies = []
            for _ in range(20):
                started = time.monotonic()
                health = client.get("/api/v1/health")
                latencies.append(time.monotonic() - started)
                assert health.status_code == 200

            result = command.result(timeout=10)

    assert result.status_code == 200
    assert sorted(latencies)[18] < 0.2
