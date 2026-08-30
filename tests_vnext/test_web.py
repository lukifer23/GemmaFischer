from fastapi.testclient import TestClient

from gemmafischer.web import create_app

TOKEN = "test-capability-token"


def test_health_and_player_are_local_and_self_hosted() -> None:
    with TestClient(create_app(capability_token=TOKEN, node_budget=1)) as client:
        assert client.get("/api/v1/health").status_code == 200
        page = client.get("/")
        assert page.status_code == 200
        assert "Explain this position" in page.text
        assert "Compare a move I considered" in page.text
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

