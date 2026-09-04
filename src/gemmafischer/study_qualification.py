from __future__ import annotations

import json
import os
import platform
import secrets
import signal
import sqlite3
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import chess
import chess.pgn

from .engine import resolve_stockfish, sha256_file
from .runtime_qualification import (
    _available_loopback_port,
    _git,
    _pid_exists,
    _process_tree_rss_bytes,
    _request_json,
    _sqlite_footprint_bytes,
    _sqlite_quick_check,
    _stockfish_descendants,
    _wait_until_ready,
    _write_json_atomic,
)

TERMINAL_STUDY_STATES = {"ready", "cancelled", "failed", "paused_storage"}


def run_study_recovery_qualification(
    output_path: Path,
    *,
    node_budget: int = 25_000,
    timeout: float = 120.0,
    shutdown_timeout: float = 10.0,
    engine_path: str | None = None,
) -> dict[str, Any]:
    """Exercise real long-study cancellation, restart, and SQLite recovery over HTTP."""
    if node_budget < 1:
        raise ValueError("node_budget must be at least 1")
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    engine = resolve_stockfish(engine_path)
    token = secrets.token_urlsafe(32)
    pgn = _long_game_pgn(200)
    observed_pids: set[int] = set()
    maximum_children = 0
    graceful_shutdowns: list[bool] = []
    raw: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix="gemmafischer-study-qualification-") as temporary:
        root = Path(temporary)
        database = root / "qualification.sqlite3"
        server, base_url = _start_server(
            root, database, engine, node_budget, token, timeout, attempt=1
        )
        rss_warm = _process_tree_rss_bytes(server.pid)
        try:
            cancel_job, create_cancel_seconds = _timed_json(
                "POST",
                f"{base_url}/api/v1/study-jobs",
                token,
                {"pgn": pgn, "perspective": "white", "rating_bucket": "1400-1599"},
                expected_status=202,
            )
            cancel_active = _wait_for_study(
                base_url,
                token,
                str(cancel_job["job_id"]),
                timeout,
                lambda item: item["state"] in {"screening", "deep_analysis"}
                and int(item["progress"]["completed_units"]) >= 4,
            )
            pids = _stockfish_descendants(server.pid, engine)
            observed_pids.update(pids)
            maximum_children = max(maximum_children, len(pids))
            cancelled, cancel_seconds = _timed_json(
                "DELETE",
                f"{base_url}/api/v1/study-jobs/{cancel_job['job_id']}",
                token,
                expected_status=200,
            )
            recovery_session, _ = _timed_json(
                "POST",
                f"{base_url}/api/v1/sessions",
                token,
                {"mode": "exhibition", "player_color": None},
                expected_status=201,
            )
            _recovered_move, engine_recovery_seconds = _timed_json(
                "POST",
                f"{base_url}/api/v1/sessions/{recovery_session['session_id']}/commands",
                token,
                {"expected_revision": 0, "action": "engine_move"},
                expected_status=200,
            )

            restart_job, create_restart_seconds = _timed_json(
                "POST",
                f"{base_url}/api/v1/study-jobs",
                token,
                {"pgn": pgn, "perspective": "white", "rating_bucket": "1400-1599"},
                expected_status=202,
            )
            before_restart = _wait_for_study(
                base_url,
                token,
                str(restart_job["job_id"]),
                timeout,
                lambda item: item["state"] in {"screening", "deep_analysis"}
                and int(item["progress"]["completed_units"]) >= 4,
            )
            pids = _stockfish_descendants(server.pid, engine)
            observed_pids.update(pids)
            maximum_children = max(maximum_children, len(pids))
            rss_before_restart = _process_tree_rss_bytes(server.pid)
        finally:
            graceful_shutdowns.append(_stop_server(server, shutdown_timeout))

        server, base_url = _start_server(
            root, database, engine, node_budget, token, timeout, attempt=2
        )
        try:
            after_restart, _ = _timed_json(
                "GET",
                f"{base_url}/api/v1/study-jobs/{restart_job['job_id']}",
                token,
                expected_status=200,
            )
            resumed, resume_seconds = _timed_json(
                "POST",
                f"{base_url}/api/v1/study-jobs/{restart_job['job_id']}/commands",
                token,
                {"expected_revision": after_restart["revision"], "action": "resume"},
                expected_status=200,
            )
            ready = _wait_for_study(
                base_url,
                token,
                str(restart_job["job_id"]),
                timeout,
                lambda item: item["state"] in TERMINAL_STUDY_STATES,
            )
            pids = _stockfish_descendants(server.pid, engine)
            observed_pids.update(pids)
            maximum_children = max(maximum_children, len(pids))
            rss_after_resume = _process_tree_rss_bytes(server.pid)

            lock = sqlite3.connect(database, timeout=0.1)
            try:
                lock.execute("BEGIN IMMEDIATE")
                _locked_body, locked_status, storage_failure_seconds = _timed_json_any_status(
                    "POST",
                    f"{base_url}/api/v1/study-jobs",
                    token,
                    {"pgn": pgn, "perspective": "white", "rating_bucket": "1400-1599"},
                )
            finally:
                lock.rollback()
                lock.close()
            recovered, storage_recovery_seconds = _timed_json(
                "POST",
                f"{base_url}/api/v1/storage/retry",
                token,
                expected_status=200,
            )
            proof_job, _ = _timed_json(
                "POST",
                f"{base_url}/api/v1/study-jobs",
                token,
                {"pgn": pgn, "perspective": "white", "rating_bucket": "1400-1599"},
                expected_status=202,
            )
            _timed_json(
                "DELETE",
                f"{base_url}/api/v1/study-jobs/{proof_job['job_id']}",
                token,
                expected_status=200,
            )
            database_bytes = _sqlite_footprint_bytes(database)
        finally:
            graceful_shutdowns.append(_stop_server(server, shutdown_timeout))

        integrity = _sqlite_quick_check(database)
        logs = "\n".join(
            item.read_text(encoding="utf-8", errors="replace")[-2000:]
            for item in sorted(root.glob("server-*.log"))
        )[-4000:]

    orphaned = sorted(pid for pid in observed_pids if _pid_exists(pid))
    exact_game = before_restart.get("game") == after_restart.get("game") == ready.get("game")
    gates: dict[str, dict[str, object]] = {
        "long_game_plies": {
            "required": 200,
            "actual": len((ready.get("game") or {}).get("moves_uci", [])),
            "passed": len((ready.get("game") or {}).get("moves_uci", [])) == 200,
        },
        "active_engine_cancellation_seconds": {
            "required_max": 2.0,
            "actual": cancel_seconds,
            "passed": cancelled.get("state") == "cancelled" and cancel_seconds <= 2.0,
        },
        "engine_available_after_cancellation_seconds": {
            "required_max": 2.0,
            "actual": engine_recovery_seconds,
            "passed": engine_recovery_seconds <= 2.0,
        },
        "restart_state": {
            "required": "paused_interrupted",
            "actual": after_restart.get("state"),
            "passed": after_restart.get("state") == "paused_interrupted",
        },
        "restart_exact_game": {
            "required": True,
            "actual": exact_game,
            "passed": exact_game,
        },
        "resumed_completion": {
            "required": "ready",
            "actual": ready.get("state"),
            "passed": resumed.get("state") == "queued" and ready.get("state") == "ready",
        },
        "storage_failure_is_typed": {
            "required": {"status": 503, "code": "STORAGE_UNAVAILABLE"},
            "actual": {
                "status": locked_status,
                "code": (_locked_body.get("error") or {}).get("code"),
            },
            "passed": locked_status == 503
            and (_locked_body.get("error") or {}).get("code") == "STORAGE_UNAVAILABLE",
        },
        "storage_recovery": {
            "required": "ready",
            "actual": recovered.get("storage_status"),
            "passed": recovered.get("storage_status") == "ready",
        },
        "sqlite_integrity": {
            "required": "ok",
            "actual": integrity,
            "passed": integrity == "ok",
        },
        "maximum_stockfish_children": {
            "required_max": 1,
            "actual": maximum_children,
            "passed": maximum_children == 1,
        },
        "orphaned_stockfish_processes": {
            "required_max": 0,
            "actual": len(orphaned),
            "passed": not orphaned,
        },
        "graceful_shutdowns": {
            "required": [True, True],
            "actual": graceful_shutdowns,
            "passed": graceful_shutdowns == [True, True],
        },
    }
    raw.extend(
        [
            {"operation": "create_cancel_study", "seconds": create_cancel_seconds},
            {"operation": "cancel_active_study", "seconds": cancel_seconds},
            {"operation": "engine_after_cancel", "seconds": engine_recovery_seconds},
            {"operation": "create_restart_study", "seconds": create_restart_seconds},
            {"operation": "resume_interrupted_study", "seconds": resume_seconds},
            {"operation": "storage_locked_write", "seconds": storage_failure_seconds},
            {"operation": "storage_recovery", "seconds": storage_recovery_seconds},
        ]
    )
    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "status": "passed" if all(bool(gate["passed"]) for gate in gates.values()) else "failed",
        "generated_at": datetime.now(UTC).isoformat(),
        "commit": _git("rev-parse", "HEAD"),
        "working_tree_clean": not bool(_git("status", "--porcelain")),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "transport": "tcp-loopback",
        "server": "uvicorn",
        "node_budget": node_budget,
        "fixture": {"kind": "legal_repetition_game", "plies": 200},
        "engine_path": str(engine),
        "engine_sha256": sha256_file(engine),
        "memory": {
            "warm_rss_bytes": rss_warm,
            "before_restart_rss_bytes": rss_before_restart,
            "after_resume_rss_bytes": rss_after_resume,
        },
        "storage": {"sqlite_bytes": database_bytes, "quick_check": integrity},
        "stockfish_lifecycle": {
            "observed_pids": sorted(observed_pids),
            "maximum_children": maximum_children,
            "orphaned_pids_after_shutdown": orphaned,
        },
        "gates": gates,
        "raw": raw,
        "server_log_tail": logs,
        "cancelled_job_progress": cancel_active["progress"],
    }
    _write_json_atomic(output_path, payload)
    return payload


def _long_game_pgn(plies: int) -> str:
    if plies < 1 or plies > 400:
        raise ValueError("plies must be between 1 and 400")
    game = chess.pgn.Game()
    game.headers.update({"Event": "Qualification", "White": "Player", "Black": "Engine"})
    board = game.board()
    node: chess.pgn.GameNode = game
    cycle = ("g1f3", "g8f6", "f3g1", "f6g8")
    for index in range(plies):
        move = chess.Move.from_uci(cycle[index % len(cycle)])
        if move not in board.legal_moves:
            raise RuntimeError("qualification fixture generation produced an illegal move")
        node = node.add_variation(move)
        board.push(move)
    return game.accept(chess.pgn.StringExporter(headers=True, variations=False, comments=False))


def _start_server(
    root: Path,
    database: Path,
    engine: Path,
    node_budget: int,
    token: str,
    timeout: float,
    *,
    attempt: int,
) -> tuple[subprocess.Popen[bytes], str]:
    port = _available_loopback_port()
    base_url = f"http://127.0.0.1:{port}"
    environment = os.environ.copy()
    environment.update(
        {
            "GEMMAFISCHER_QUALIFICATION_PORT": str(port),
            "GEMMAFISCHER_QUALIFICATION_TOKEN": token,
            "GEMMAFISCHER_QUALIFICATION_HISTORY": str(database),
            "GEMMAFISCHER_QUALIFICATION_NODES": str(node_budget),
            "GEMMAFISCHER_QUALIFICATION_ENGINE": str(engine),
        }
    )
    log_path = root / f"server-{attempt}.log"
    started = time.perf_counter()
    with log_path.open("ab") as log:
        server = subprocess.Popen(
            [sys.executable, "-m", "gemmafischer._qualification_server"],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=environment,
            close_fds=True,
        )
    _wait_until_ready(base_url, server, started, min(timeout, 15.0), log_path)
    return server, base_url


def _stop_server(server: subprocess.Popen[bytes], timeout: float) -> bool:
    if server.poll() is not None:
        return server.returncode == 0
    server.send_signal(signal.SIGINT)
    try:
        server.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        server.kill()
        server.wait(timeout=2)
        return False
    return server.returncode == 0


def _wait_for_study(
    base_url: str,
    token: str,
    job_id: str,
    timeout: float,
    predicate: Any,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    latest: dict[str, Any] = {}
    while time.monotonic() < deadline:
        latest, status = _request_json(
            "GET", f"{base_url}/api/v1/study-jobs/{job_id}", token=token
        )
        if status != 200:
            raise RuntimeError(f"Study poll returned HTTP {status}")
        if predicate(latest):
            return latest
        if latest.get("state") in TERMINAL_STUDY_STATES:
            break
        time.sleep(0.01)
    raise RuntimeError(f"Study {job_id} did not reach the expected state; latest={latest}")


def _timed_json(
    method: str,
    url: str,
    token: str,
    payload: dict[str, object] | None = None,
    *,
    expected_status: int,
) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    body, status = _request_json(method, url, token=token, payload=payload)
    elapsed = time.perf_counter() - started
    if status != expected_status:
        raise RuntimeError(f"{method} {url} returned HTTP {status}, expected {expected_status}")
    return body, elapsed


def _timed_json_any_status(
    method: str,
    url: str,
    token: str,
    payload: dict[str, object] | None = None,
) -> tuple[dict[str, Any], int, float]:
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {"X-GemmaFischer-Token": token}
    if data is not None:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read()), response.status, time.perf_counter() - started
    except urllib.error.HTTPError as exc:
        return json.loads(exc.read()), exc.code, time.perf_counter() - started
