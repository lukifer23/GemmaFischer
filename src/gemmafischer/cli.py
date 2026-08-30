from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

import uvicorn

from . import __version__
from .coach import render_claim
from .data_audit import audit_data
from .dataset import acquire_source, build_puzzle_dataset, load_source
from .domain import AnalysisRequest, AnalysisState, RatingBucket, Workflow
from .engine import EngineUnavailable, StockfishProvider, resolve_stockfish
from .lifecycle import (
    InstanceAlreadyRunning,
    InstanceLock,
    instance_status,
    runtime_dir,
    stop_instance,
)
from .qualification import run_deterministic_benchmark, run_full_profile_benchmark
from .repo_audit import audit_repository
from .runtime import ModelUnavailable, inspect_model_assets
from .service import AnalysisService
from .storage import default_history_path
from .web import create_app

EXAMPLE_FEN = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="gemmafischer", description="Grounded local chess coach")
    root.add_argument("--version", action="version", version=__version__)
    commands = root.add_subparsers(dest="command", required=True)

    setup = commands.add_parser("setup", help="Inspect and verify runtime assets")
    setup.add_argument("--profile", choices=("deterministic", "full"), default="deterministic")
    setup.add_argument("--plan", action="store_true")
    setup.add_argument("--yes", action="store_true")
    setup.add_argument("--repair", action="store_true")
    setup.add_argument("--format", choices=("human", "json"), default="human")

    doctor = commands.add_parser("doctor", help="Check prerequisites and asset integrity")
    doctor.add_argument(
        "--profile", choices=("dev", "deterministic", "full"), default="deterministic"
    )
    doctor.add_argument("--format", choices=("human", "json"), default="human")

    analyze = commands.add_parser("analyze", help="Analyze one position without the browser")
    source = analyze.add_mutually_exclusive_group(required=True)
    source.add_argument("--example", action="store_true")
    source.add_argument("--fen")
    source.add_argument("--stdin", action="store_true")
    analyze.add_argument("--mode", choices=("position", "compare"), default="position")
    analyze.add_argument("--consider")
    analyze.add_argument(
        "--rating",
        choices=tuple(item.value for item in RatingBucket),
        default="1400-1599",
    )
    analyze.add_argument("--profile", choices=("deterministic", "full"), default="deterministic")
    analyze.add_argument("--timeout", type=float, default=60)
    analyze.add_argument("--format", choices=("human", "json"), default="human")
    analyze.add_argument("--output", type=Path)
    analyze.add_argument("--offline", action="store_true")
    analyze.add_argument("--no-color", action="store_true")

    serve = commands.add_parser("serve", help="Start the loopback-only player server")
    serve.add_argument("--host", default="127.0.0.1", choices=("127.0.0.1", "localhost"))
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--profile", choices=("deterministic", "full"), default="deterministic")
    serve.add_argument("--open", action="store_true")
    serve.add_argument("--print-token", action="store_true")

    launch = commands.add_parser("launch", help="Start one local instance and open the player")
    launch.add_argument("--port", type=int, default=8765)
    launch.add_argument("--profile", choices=("deterministic", "full"), default="deterministic")
    launch.add_argument("--no-open", action="store_true")
    launch.add_argument("--timeout", type=float, default=15.0)

    commands.add_parser("status", help="Show the verified local server process")
    stop = commands.add_parser("stop", help="Stop the verified local server process")
    stop.add_argument("--timeout", type=float, default=10.0)

    commands.add_parser("version", help="Print version metadata").add_argument(
        "--json", action="store_true"
    )
    commands.add_parser("audit-legacy", help="Print the legacy evidence ledger status")
    data_audit = commands.add_parser("audit-data", help="Audit training data and eval isolation")
    data_audit.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/data-audit/latest.json"),
    )
    acquire = commands.add_parser("acquire-data", help="Download and verify a pinned source")
    acquire.add_argument("--manifest", type=Path, default=Path("data/sources.json"))
    acquire.add_argument("--source", default="lichess-puzzles-2026-08-02")
    acquire.add_argument(
        "--output", type=Path, default=Path("data/raw/lichess_db_puzzle.csv.zst")
    )
    build_data = commands.add_parser(
        "build-dataset", help="Build typed, lineage-isolated lesson records"
    )
    build_data.add_argument("--manifest", type=Path, default=Path("data/sources.json"))
    build_data.add_argument("--source", default="lichess-puzzles-2026-08-02")
    build_data.add_argument(
        "--archive", type=Path, default=Path("data/raw/lichess_db_puzzle.csv.zst")
    )
    build_data.add_argument("--output-dir", type=Path, default=Path("data/derived"))
    build_data.add_argument("--limit", type=int, default=1000)
    build_data.add_argument("--nodes", type=int, default=50_000)
    benchmark = commands.add_parser(
        "benchmark", help="Run the deterministic qualification benchmark"
    )
    benchmark.add_argument(
        "--fixtures",
        type=Path,
        default=Path("data/evaluation/diagnostic_positions.jsonl"),
    )
    benchmark.add_argument(
        "--profile", choices=("deterministic", "full"), default="deterministic"
    )
    benchmark.add_argument("--requests", type=int, default=100)
    benchmark.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/qualification/deterministic-latest.json"),
    )
    commands.add_parser("verify", help="Run the documented portable verification command")
    commands.add_parser("repo-audit", help="Detect duplicate and unsupported repository content")
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        return int(globals()[f"cmd_{args.command.replace('-', '_')}"](args))
    except KeyboardInterrupt:
        return 130
    except ValueError as exc:
        print(f"INVALID_INPUT: {exc}", file=sys.stderr)
        return 2


def _emit(payload: object, output_format: str) -> None:
    if output_format == "json":
        print(json.dumps(payload, indent=2, default=str))
    elif isinstance(payload, dict):
        for key, value in payload.items():
            print(f"{key}: {value}")
    else:
        print(payload)


def cmd_setup(args: argparse.Namespace) -> int:
    try:
        engine = resolve_stockfish()
        provider = StockfishProvider(str(engine), node_budget=1)
        engine_info = {
            "path": str(engine),
            "sha256": provider.binary_sha256,
            "status": "verified-local",
        }
    except EngineUnavailable:
        engine_info = {
            "status": "missing",
            "repair": (
                "brew install stockfish; export "
                "GEMMAFISCHER_STOCKFISH=$(command -v stockfish)"
            ),
        }
    payload = {
        "profile": args.profile,
        "mutating": False,
        "engine": engine_info,
        "model": (
            {"id": "mlx-community/gemma-4-e2b-it-4bit", "download": "opt-in"}
            if args.profile == "full"
            else {"status": "not required"}
        ),
    }
    _emit(payload, args.format)
    return 0 if engine_info["status"] != "missing" else 3


def cmd_doctor(args: argparse.Namespace) -> int:
    checks: list[dict[str, object]] = []
    checks.append(
        {
            "code": "PYTHON_VERSION",
            "ok": sys.version_info[:2] == (3, 12),
            "actual": platform.python_version(),
        }
    )
    if args.profile != "dev":
        try:
            path = resolve_stockfish()
            checks.append({"code": "STOCKFISH", "ok": True, "path": str(path)})
        except EngineUnavailable as exc:
            checks.append({"code": "STOCKFISH", "ok": False, "message": str(exc)})
    if args.profile == "full":
        try:
            import mlx_lm  # noqa: F401
            checks.append({"code": "MLX_LM", "ok": True})
            checks.append({"code": "MODEL_ASSETS", "ok": True, **inspect_model_assets()})
        except ImportError:
            checks.append({"code": "MLX_LM", "ok": False, "message": "Run uv sync --extra full"})
        except ModelUnavailable as exc:
            checks.append({"code": "MODEL_ASSETS", "ok": False, "message": str(exc)})
    payload = {
        "profile": args.profile,
        "ok": all(bool(item["ok"]) for item in checks),
        "checks": checks,
    }
    _emit(payload, args.format)
    return 0 if payload["ok"] else 3


def cmd_analyze(args: argparse.Namespace) -> int:
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
    fen = EXAMPLE_FEN if args.example else (sys.stdin.read().strip() if args.stdin else args.fen)
    assert fen is not None
    request = AnalysisRequest(
        mode=Workflow(args.mode),
        fen=fen,
        rating_bucket=RatingBucket(args.rating),
        considered_move_uci=args.consider,
    )
    service = AnalysisService(full_profile=args.profile == "full")
    try:
        snapshot = service.submit(request)
        deadline = time.monotonic() + args.timeout
        while snapshot.state not in {
            AnalysisState.COMPLETE,
            AnalysisState.ENGINE_ONLY,
            AnalysisState.FAILED,
            AnalysisState.CANCELLED,
        }:
            if time.monotonic() >= deadline:
                service.cancel(snapshot.analysis_id)
                print("ANALYSIS_TIMEOUT: analysis exceeded --timeout", file=sys.stderr)
                return 5
            time.sleep(0.05)
            snapshot = service.get(snapshot.analysis_id) or snapshot
    finally:
        service.close()
    if snapshot.state is AnalysisState.FAILED:
        assert snapshot.error
        print(f"{snapshot.error.code}: {snapshot.error.message}", file=sys.stderr)
        return 3 if snapshot.error.code == "ENGINE_UNAVAILABLE" else 5
    if args.format == "json":
        text = snapshot.model_dump_json(indent=2)
    else:
        assert snapshot.evidence and snapshot.coaching
        lines = [snapshot.coaching.summary]
        lines.extend(render_claim(snapshot.evidence, claim) for claim in snapshot.coaching.claims)
        lines.append(f"Evidence: {snapshot.evidence.position_id}")
        text = "\n".join(lines)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    token = os.urandom(32).hex()
    if args.print_token:
        print(token, file=sys.stderr)
    url = f"http://{args.host}:{args.port}"
    if args.open:
        webbrowser.open(url)
    try:
        with InstanceLock(args.host, args.port, args.profile):
            uvicorn.run(
                create_app(
                    full_profile=args.profile == "full",
                    capability_token=token,
                    history_path=default_history_path(),
                ),
                host=args.host,
                port=args.port,
                log_level="info",
            )
    except InstanceAlreadyRunning as exc:
        print(f"INSTANCE_ALREADY_RUNNING: {exc}", file=sys.stderr)
        return 6
    return 0


def cmd_launch(args: argparse.Namespace) -> int:
    current = instance_status()
    if current.get("running"):
        url = f"http://{current['host']}:{current['port']}"
        if not args.no_open:
            webbrowser.open(url)
        _emit({"status": "already_running", "url": url, "pid": current["pid"]}, "human")
        return 0
    runtime_dir().mkdir(parents=True, exist_ok=True)
    log_path = runtime_dir() / "server.log"
    with log_path.open("ab") as log:
        subprocess.Popen(
            [
                sys.executable,
                "-m",
                "gemmafischer.cli",
                "serve",
                "--profile",
                args.profile,
                "--port",
                str(args.port),
            ],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    url = f"http://127.0.0.1:{args.port}"
    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{url}/api/v1/health", timeout=0.5) as response:
                if response.status == 200:
                    if not args.no_open:
                        webbrowser.open(url)
                    status = instance_status()
                    _emit({"status": "running", "url": url, "pid": status.get("pid")}, "human")
                    return 0
        except (urllib.error.URLError, TimeoutError):
            time.sleep(0.1)
    print(f"LAUNCH_TIMEOUT: inspect {log_path}", file=sys.stderr)
    return 5


def cmd_status(args: argparse.Namespace) -> int:
    status = instance_status()
    _emit(status, "human")
    return 0 if status.get("running") else 3


def cmd_stop(args: argparse.Namespace) -> int:
    result = stop_instance(args.timeout)
    _emit(result, "human")
    return 0 if result.get("stopped") or not result.get("running") else 5


def cmd_version(args: argparse.Namespace) -> int:
    payload = {"application": __version__, "api": "v1", "evidence_schema": "2.0"}
    _emit(payload, "json" if args.json else "human")
    return 0


def cmd_audit_legacy(args: argparse.Namespace) -> int:
    ledger = Path("assets/evidence-status.json")
    print(ledger.read_text(encoding="utf-8") if ledger.exists() else "Legacy ledger missing")
    return 0 if ledger.exists() else 3


def cmd_audit_data(args: argparse.Namespace) -> int:
    training_paths = sorted(Path("data/derived").glob("train*.jsonl"))
    evaluation_paths = sorted(Path("data/derived").glob("evaluation*.jsonl"))
    payload = audit_data(training_paths, evaluation_paths, args.output)
    _emit(
        {
            "status": payload["status"],
            "ready_for_training": payload["gate"]["ready_for_training"],
            "training_records": payload["training"]["totals"].get("records", 0),
            "conflicting_best_move_fens": payload["cross_dataset"][
                "conflicting_best_move_fens"
            ],
            "train_evaluation_fen_overlap": payload["cross_dataset"][
                "train_evaluation_fen_overlap"
            ],
            "output": str(args.output),
        },
        "human",
    )
    return 0 if payload["gate"]["ready_for_training"] else 4


def cmd_acquire_data(args: argparse.Namespace) -> int:
    source = load_source(args.manifest, args.source)
    payload = acquire_source(source, args.output)
    print(json.dumps(payload, indent=2))
    return 0


def cmd_build_dataset(args: argparse.Namespace) -> int:
    source = load_source(args.manifest, args.source)
    payload = build_puzzle_dataset(
        args.archive,
        args.output_dir,
        source,
        limit=args.limit,
        node_budget=args.nodes,
    )
    print(json.dumps(payload, indent=2))
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    if args.requests < 1:
        raise ValueError("--requests must be at least 1")
    payload = (
        run_full_profile_benchmark(args.fixtures, args.output, args.requests)
        if args.profile == "full"
        else run_deterministic_benchmark(args.fixtures, args.output, args.requests)
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "requests": payload["request_count"],
                "latency_seconds": payload.get(
                    "latency_seconds", payload.get("total_latency_seconds")
                ),
                "engine_sha256": payload["engine_sha256"],
            },
            indent=2,
        )
    )
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    commands = (
        ["uv", "run", "ruff", "check", "src/gemmafischer", "tests"],
        ["uv", "run", "mypy"],
        ["uv", "run", "pytest", "-m", "not model", "tests"],
    )
    for command in commands:
        print("+ " + " ".join(command), flush=True)
        completed = subprocess.run(command, check=False)
        if completed.returncode:
            return completed.returncode
    return 0


def cmd_repo_audit(args: argparse.Namespace) -> int:
    payload = audit_repository(Path.cwd())
    print(json.dumps(payload, indent=2))
    return 0 if payload["status"] == "passed" else 4


if __name__ == "__main__":
    raise SystemExit(main())
