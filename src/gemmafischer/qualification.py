from __future__ import annotations

import json
import platform
import resource
import statistics
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .engine import StockfishProvider


def percentile(values: list[float], percent: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * percent)))
    return ordered[index]


def run_deterministic_benchmark(
    fixture_path: Path,
    output_path: Path,
    request_count: int = 100,
) -> dict[str, Any]:
    fixtures = [json.loads(line) for line in fixture_path.read_text().splitlines() if line]
    if not fixtures:
        raise ValueError("The benchmark fixture file is empty")
    provider = StockfishProvider()
    raw: list[dict[str, Any]] = []
    for index in range(request_count):
        fixture = fixtures[index % len(fixtures)]
        considered = fixture.get("considered_move_uci")
        started = time.perf_counter()
        evidence = provider.analyze(fixture["fen"], considered)
        elapsed = time.perf_counter() - started
        raw.append(
            {
                "index": index,
                "fixture_id": fixture["id"],
                "workflow": "compare" if considered else "position",
                "latency_seconds": elapsed,
                "position_id": evidence.position_id,
                "candidate_moves": [item.move_uci for item in evidence.candidates],
                "candidate_scores_cp": [item.score_cp for item in evidence.candidates],
                "terminal_reason": evidence.terminal_reason,
            }
        )
    latencies = [item["latency_seconds"] for item in raw]
    payload = {
        "schema_version": "1.0",
        "status": "target-host-repeatability-only",
        "generated_at": datetime.now(UTC).isoformat(),
        "commit": _git("rev-parse", "HEAD"),
        "working_tree_clean": not bool(_git("status", "--porcelain")),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "fixture_path": str(fixture_path),
        "fixture_sha256": _sha256(fixture_path),
        "engine_path": str(provider.path),
        "engine_sha256": provider.binary_sha256,
        "node_budget": provider.node_budget,
        "request_count": request_count,
        "latency_seconds": {
            "mean": statistics.fmean(latencies),
            "p50": percentile(latencies, 0.50),
            "p95": percentile(latencies, 0.95),
            "max": max(latencies),
        },
        "process_max_rss_raw": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "raw": raw,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output_path)
    return payload


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()

