from __future__ import annotations

import json
from pathlib import Path

import pytest

from gemmafischer.runtime_qualification import run_runtime_qualification


@pytest.mark.hardware
def test_real_loopback_runtime_and_stockfish_lifecycle(tmp_path: Path) -> None:
    output = tmp_path / "runtime.json"

    result = run_runtime_qualification(output, request_count=2, node_budget=1_000)

    assert result["transport"] == "tcp-loopback"
    assert result["total_http_requests"] == 8
    assert result["stockfish_lifecycle"]["counts_after_engine_request"] == [1, 1]
    assert result["stockfish_lifecycle"]["maximum_processes"] == 1
    assert result["stockfish_lifecycle"]["unique_observed_processes"] == 1
    assert result["stockfish_lifecycle"]["process_restarts_observed"] == 0
    assert result["stockfish_lifecycle"]["orphaned_pids_after_shutdown"] == []
    assert result["stockfish_lifecycle"]["graceful_shutdown"] is True
    assert result["stockfish_lifecycle"]["server_returncode"] == 0
    assert result["storage"]["quick_check"] == "ok"
    assert result["memory"]["warm_baseline_bytes"] > 0
    assert result["gates"]["sqlite_integrity"]["passed"] is True
    assert result["gates"]["stockfish_process_restarts"]["passed"] is True
    assert len(result["raw"]) == 8
    assert json.loads(output.read_text(encoding="utf-8"))["engine_sha256"]
