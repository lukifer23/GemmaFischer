from __future__ import annotations

import json
import os
import platform
import secrets
import signal
import socket
import sqlite3
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .engine import resolve_stockfish, sha256_file
from .qualification import percentile

RUNTIME_GATES_SECONDS = {
    "health": 0.100,
    "session_create": 0.100,
    "legal_moves": 0.100,
    "engine_move": 0.500,
}
MAX_POST_WARM_RSS_GROWTH_BYTES = 50 * 1024 * 1024


def run_runtime_qualification(
    output_path: Path,
    *,
    request_count: int = 20,
    node_budget: int = 250_000,
    startup_timeout: float = 15.0,
    shutdown_timeout: float = 10.0,
    engine_path: str | None = None,
) -> dict[str, Any]:
    """Measure the production ASGI stack over a real loopback TCP socket."""
    if request_count < 1:
        raise ValueError("request_count must be at least 1")
    if node_budget < 1:
        raise ValueError("node_budget must be at least 1")
    engine = resolve_stockfish(engine_path)
    token = secrets.token_urlsafe(32)
    port = _available_loopback_port()
    base_url = f"http://127.0.0.1:{port}"
    raw: list[dict[str, Any]] = []
    observed_stockfish_pids: set[int] = set()
    stockfish_counts: list[int] = []
    rss_samples: list[int] = []
    started_at = time.perf_counter()
    graceful_shutdown = False
    server_returncode: int | None = None

    with tempfile.TemporaryDirectory(prefix="gemmafischer-runtime-") as temporary:
        temporary_path = Path(temporary)
        environment = os.environ.copy()
        environment.update(
            {
                "GEMMAFISCHER_QUALIFICATION_PORT": str(port),
                "GEMMAFISCHER_QUALIFICATION_TOKEN": token,
                "GEMMAFISCHER_QUALIFICATION_HISTORY": str(
                    temporary_path / "qualification.sqlite3"
                ),
                "GEMMAFISCHER_QUALIFICATION_NODES": str(node_budget),
                "GEMMAFISCHER_QUALIFICATION_ENGINE": str(engine),
            }
        )
        log_path = temporary_path / "server.log"
        with log_path.open("wb") as server_log:
            server = subprocess.Popen(
                [sys.executable, "-m", "gemmafischer._qualification_server"],
                stdin=subprocess.DEVNULL,
                stdout=server_log,
                stderr=subprocess.STDOUT,
                env=environment,
                close_fds=True,
            )
            try:
                startup_seconds = _wait_until_ready(
                    base_url, server, started_at, startup_timeout, log_path
                )
                for index in range(request_count):
                    _record_request(raw, index, "health", "GET", f"{base_url}/api/v1/health")
                    session = _record_request(
                        raw,
                        index,
                        "session_create",
                        "POST",
                        f"{base_url}/api/v1/sessions",
                        token=token,
                        payload={"mode": "exhibition", "player_color": None},
                        expected_status=201,
                    )
                    session_id = str(session["session_id"])
                    _record_request(
                        raw,
                        index,
                        "legal_moves",
                        "GET",
                        f"{base_url}/api/v1/sessions/{session_id}/legal-moves?from_square=e2",
                    )
                    _record_request(
                        raw,
                        index,
                        "engine_move",
                        "POST",
                        f"{base_url}/api/v1/sessions/{session_id}/commands",
                        token=token,
                        payload={"expected_revision": 0, "action": "engine_move"},
                    )
                    stockfish_pids = _stockfish_descendants(server.pid, engine)
                    stockfish_counts.append(len(stockfish_pids))
                    observed_stockfish_pids.update(stockfish_pids)
                    rss_samples.append(_process_tree_rss_bytes(server.pid))
                database_bytes = _sqlite_footprint_bytes(
                    temporary_path / "qualification.sqlite3"
                )
            finally:
                if server.poll() is None:
                    # SIGINT follows Uvicorn's supported graceful-shutdown path and
                    # therefore exercises FastAPI lifespan cleanup and provider.close().
                    server.send_signal(signal.SIGINT)
                    try:
                        server.wait(timeout=shutdown_timeout)
                        graceful_shutdown = True
                    except subprocess.TimeoutExpired:
                        server.kill()
                        server.wait(timeout=2)
                server_returncode = server.returncode
        server_log_text = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
        integrity = _sqlite_quick_check(temporary_path / "qualification.sqlite3")

    orphaned_pids = sorted(pid for pid in observed_stockfish_pids if _pid_exists(pid))
    summaries = {
        name: _latency_summary(
            [float(item["latency_seconds"]) for item in raw if item["operation"] == name]
        )
        for name in RUNTIME_GATES_SECONDS
    }
    gates: dict[str, dict[str, object]] = {
        f"{name}_p95_seconds": {
            "required_max": limit,
            "actual": summaries[name]["p95"],
            "passed": summaries[name]["p95"] <= limit,
        }
        for name, limit in RUNTIME_GATES_SECONDS.items()
    }
    gates.update(
        {
            "stockfish_process_max": {
                "required_max": 1,
                "actual": max(stockfish_counts, default=0),
                "passed": bool(stockfish_counts) and max(stockfish_counts) == 1,
            },
            "stockfish_process_min_after_engine_start": {
                "required_min": 1,
                "actual": min(stockfish_counts, default=0),
                "passed": bool(stockfish_counts) and min(stockfish_counts) == 1,
            },
            "orphaned_stockfish_processes": {
                "required_max": 0,
                "actual": len(orphaned_pids),
                "passed": not orphaned_pids,
            },
            "graceful_server_shutdown": {
                "required": True,
                "actual": graceful_shutdown,
                "passed": graceful_shutdown and server_returncode == 0,
            },
            "post_warm_rss_growth_bytes": {
                "required_max": MAX_POST_WARM_RSS_GROWTH_BYTES,
                "actual": max(rss_samples[1:], default=rss_samples[0] if rss_samples else 0)
                - (rss_samples[0] if rss_samples else 0),
                "passed": bool(rss_samples)
                and max(rss_samples[1:], default=rss_samples[0]) - rss_samples[0]
                <= MAX_POST_WARM_RSS_GROWTH_BYTES,
            },
            "sqlite_integrity": {
                "required": "ok",
                "actual": integrity,
                "passed": integrity == "ok",
            },
        }
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
        "request_cycles": request_count,
        "total_http_requests": len(raw),
        "node_budget": node_budget,
        "engine_path": str(engine),
        "engine_sha256": sha256_file(engine),
        "startup_seconds": startup_seconds,
        "storage": {
            "sqlite_bytes": database_bytes,
            "quick_check": integrity,
        },
        "memory": {
            "process_tree_rss_bytes_after_each_cycle": rss_samples,
            "warm_baseline_bytes": rss_samples[0] if rss_samples else None,
            "peak_bytes": max(rss_samples, default=0),
            "post_warm_growth_bytes": (
                max(rss_samples[1:], default=rss_samples[0]) - rss_samples[0]
                if rss_samples
                else None
            ),
        },
        "latency_seconds": summaries,
        "stockfish_lifecycle": {
            "counts_after_engine_request": stockfish_counts,
            "observed_pids": sorted(observed_stockfish_pids),
            "unique_observed_processes": len(observed_stockfish_pids),
            "process_restarts_observed": max(0, len(observed_stockfish_pids) - 1),
            "maximum_processes": max(stockfish_counts, default=0),
            "orphaned_pids_after_shutdown": orphaned_pids,
            "server_returncode": server_returncode,
            "graceful_shutdown": graceful_shutdown,
        },
        "gates": gates,
        "server_log_tail": server_log_text,
        "raw": raw,
    }
    _write_json_atomic(output_path, payload)
    return payload


def _available_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _wait_until_ready(
    base_url: str,
    server: subprocess.Popen[bytes],
    started_at: float,
    timeout: float,
    log_path: Path,
) -> float:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if server.poll() is not None:
            log = log_path.read_text(encoding="utf-8", errors="replace")
            raise RuntimeError(
                f"Qualification server exited with {server.returncode}: {log[-1000:]}"
            )
        try:
            _request_json("GET", f"{base_url}/api/v1/health")
            return time.perf_counter() - started_at
        except (OSError, RuntimeError, urllib.error.URLError):
            time.sleep(0.025)
    raise RuntimeError(f"Qualification server did not start within {timeout:.1f} seconds")


def _record_request(
    raw: list[dict[str, Any]],
    cycle: int,
    operation: str,
    method: str,
    url: str,
    *,
    token: str | None = None,
    payload: dict[str, object] | None = None,
    expected_status: int = 200,
) -> dict[str, Any]:
    started = time.perf_counter()
    response, status = _request_json(method, url, token=token, payload=payload)
    elapsed = time.perf_counter() - started
    raw.append(
        {
            "cycle": cycle,
            "operation": operation,
            "method": method,
            "status": status,
            "response_bytes": len(json.dumps(response, separators=(",", ":")).encode()),
            "latency_seconds": elapsed,
        }
    )
    if status != expected_status:
        raise RuntimeError(f"{operation} returned HTTP {status}, expected {expected_status}")
    return response


def _request_json(
    method: str,
    url: str,
    *,
    token: str | None = None,
    payload: dict[str, object] | None = None,
) -> tuple[dict[str, Any], int]:
    body = json.dumps(payload).encode() if payload is not None else None
    headers = {"Content-Type": "application/json"} if body is not None else {}
    if token is not None:
        headers["X-GemmaFischer-Token"] = token
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read()), response.status
    except urllib.error.HTTPError as exc:
        body_text = exc.read().decode(errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {body_text[:500]}") from exc


def _process_rows() -> list[tuple[int, int, str]]:
    output = subprocess.run(
        ["ps", "-axo", "pid=,ppid=,command="],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    rows: list[tuple[int, int, str]] = []
    for line in output.splitlines():
        fields = line.strip().split(maxsplit=2)
        if len(fields) == 3:
            try:
                rows.append((int(fields[0]), int(fields[1]), fields[2]))
            except ValueError:
                continue
    return rows


def _stockfish_descendants(server_pid: int, engine_path: Path) -> set[int]:
    rows = _process_rows()
    descendants = {server_pid}
    changed = True
    while changed:
        changed = False
        for pid, parent_pid, _command in rows:
            if parent_pid in descendants and pid not in descendants:
                descendants.add(pid)
                changed = True
    engine_resolved = str(engine_path.resolve())
    return {
        pid
        for pid, _parent, command in rows
        if pid in descendants
        and (engine_resolved in command or Path(command.split()[0]).name == engine_path.name)
    }


def _pid_exists(pid: int) -> bool:
    return any(row_pid == pid for row_pid, _parent, _command in _process_rows())


def _process_tree_rss_bytes(root_pid: int) -> int:
    rows = _process_rows()
    descendants = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, parent_pid, _command in rows:
            if parent_pid in descendants and pid not in descendants:
                descendants.add(pid)
                changed = True
    output = subprocess.run(
        ["ps", "-o", "pid=,rss=", "-p", ",".join(str(pid) for pid in descendants)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return sum(int(line.split()[1]) * 1024 for line in output.splitlines() if line.split())


def _sqlite_footprint_bytes(database: Path) -> int:
    return sum(
        candidate.stat().st_size
        for candidate in (database, Path(f"{database}-wal"), Path(f"{database}-shm"))
        if candidate.exists()
    )


def _sqlite_quick_check(database: Path) -> str:
    with sqlite3.connect(database) as connection:
        row = connection.execute("PRAGMA quick_check").fetchone()
    return str(row[0]) if row else "missing-result"


def _latency_summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "max": max(values),
    }


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], check=True, capture_output=True, text=True
    ).stdout.strip()
