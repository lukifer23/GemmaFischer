from pathlib import Path

from fastapi.testclient import TestClient

from gemmafischer.web import create_app

TOKEN = "test-capability-token"


def test_health_and_player_are_local_and_self_hosted() -> None:
    with TestClient(create_app(capability_token=TOKEN, node_budget=1)) as client:
        health = client.get("/api/v1/health")
        assert health.status_code == 200
        assert isinstance(health.json()["engine_available"], bool)
        assert health.json()["history_enabled"] is False
        page = client.get("/")
        assert page.status_code == 200
        assert "Explain this position" in page.text
        assert "Play, analyze, and learn in one session" in page.text
        assert "Engine vs engine" in page.text
        assert "https://" not in page.text


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
