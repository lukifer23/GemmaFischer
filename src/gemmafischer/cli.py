from __future__ import annotations

import argparse
import hashlib
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
from typing import Any

import uvicorn

from . import __version__
from .accuracy_eval import (
    run_constructed_accuracy_benchmark,
    run_lichess_puzzle_accuracy_benchmark,
)
from .coach import render_claim, validate_model_claims
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
from .lmstudio import DEFAULT_LFM_MODEL, DEFAULT_LM_STUDIO_URL
from .model_profile import (
    profile_lmstudio_generation,
    profile_mlx_generation,
    validate_profile_outputs,
)
from .qualification import run_deterministic_benchmark, run_full_profile_benchmark
from .repo_audit import audit_repository
from .resources import bundled_path
from .runtime import (
    DEFAULT_MODEL,
    DEFAULT_MODEL_REVISION,
    ModelUnavailable,
    claim_selection_prompt,
    inspect_model_assets,
    parse_claim_selection,
)
from .runtime_qualification import run_runtime_qualification
from .service import AnalysisService
from .storage import default_history_path
from .training_readiness import evaluate_training_readiness
from .tutor_eval import run_tutoring_qualification
from .verification import (
    PORTABLE_COMMANDS,
    portable_findings,
    run_command,
    verify_release_status,
)
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
    readiness = commands.add_parser(
        "training-readiness", help="Fail-closed data, hardware, and toolchain gate"
    )
    readiness.add_argument(
        "--audit", type=Path, default=Path("artifacts/data-audit/latest.json")
    )
    readiness.add_argument(
        "--manifest", type=Path, default=Path("training/post-training.json")
    )
    readiness.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/training/readiness-latest.json"),
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
        default=bundled_path("data/evaluation/diagnostic_positions.jsonl"),
    )
    model_profile = commands.add_parser(
        "profile-model", help="Measure real pinned-model TTFT, tokens, TPS, and memory"
    )
    model_profile.add_argument(
        "--fixtures",
        type=Path,
        default=bundled_path("data/evaluation/diagnostic_positions.jsonl"),
    )
    model_profile.add_argument("--requests", type=int, default=21)
    model_profile.add_argument("--max-tokens", type=int, default=768)
    model_profile.add_argument("--nodes", type=int, default=250_000)
    model_profile.add_argument("--backend", choices=("mlx", "lmstudio"), default="mlx")
    model_profile.add_argument("--model")
    model_profile.add_argument(
        "--revision", default=DEFAULT_MODEL_REVISION
    )
    model_profile.add_argument(
        "--manifest", type=Path, default=bundled_path("assets/model-manifest.json")
    )
    model_profile.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/qualification/model-profile-latest.json"),
    )
    model_profile.add_argument("--base-url", default=DEFAULT_LM_STUDIO_URL)
    model_profile.add_argument("--model-artifact", type=Path)
    model_profile.add_argument("--timeout", type=float, default=30.0)
    accuracy = commands.add_parser(
        "evaluate-accuracy", help="Run constructed and held-out Lichess accuracy suites"
    )
    accuracy.add_argument("--suite", choices=("constructed", "lichess", "all"), default="all")
    accuracy.add_argument(
        "--fixtures",
        type=Path,
        default=bundled_path("data/evaluation/accuracy_positions.jsonl"),
    )
    accuracy.add_argument(
        "--archive", type=Path, default=Path("data/raw/lichess_db_puzzle.csv.zst")
    )
    accuracy.add_argument("--manifest", type=Path, default=bundled_path("data/sources.json"))
    accuracy.add_argument("--sample-size", type=int, default=100)
    accuracy.add_argument("--repeats", type=int, default=3)
    accuracy.add_argument("--nodes", type=int, default=250_000)
    accuracy.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/qualification")
    )
    tutor = commands.add_parser(
        "evaluate-tutoring", help="Run automated tutoring and blinded-review qualification"
    )
    tutor.add_argument(
        "--cases",
        type=Path,
        default=bundled_path("data/evaluation/tutoring_cases.jsonl"),
    )
    tutor.add_argument("--profile", choices=("deterministic", "full"), default="deterministic")
    tutor.add_argument("--repetitions", type=int, default=2)
    tutor.add_argument("--backend", choices=("mlx", "lmstudio"), default="mlx")
    tutor.add_argument("--model")
    tutor.add_argument(
        "--revision", default="238767527555cb75a05732a84dff5d6ba0dd6809"
    )
    tutor.add_argument(
        "--manifest", type=Path, default=bundled_path("assets/model-manifest.json")
    )
    tutor.add_argument("--base-url", default=DEFAULT_LM_STUDIO_URL)
    tutor.add_argument("--model-artifact", type=Path)
    tutor.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/qualification/tutoring-latest.json"),
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
    runtime_profile = commands.add_parser(
        "profile-runtime",
        help="Measure real loopback HTTP latency and Stockfish process lifecycle",
    )
    runtime_profile.add_argument("--requests", type=int, default=20)
    runtime_profile.add_argument("--nodes", type=int, default=250_000)
    runtime_profile.add_argument("--startup-timeout", type=float, default=15.0)
    runtime_profile.add_argument("--shutdown-timeout", type=float, default=10.0)
    runtime_profile.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/qualification/runtime-latest.json"),
    )
    verify = commands.add_parser("verify", help="Run portable, local-alpha, or release gates")
    verify.add_argument(
        "--tier", choices=("portable", "local-alpha", "release"), default="portable"
    )
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
    validation_paths = sorted(Path("data/derived").glob("validation*.jsonl"))
    evaluation_paths = sorted(Path("data/derived").glob("final_test*.jsonl"))
    payload = audit_data(
        training_paths,
        evaluation_paths,
        args.output,
        validation_paths=validation_paths,
    )
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


def cmd_training_readiness(args: argparse.Namespace) -> int:
    payload = evaluate_training_readiness(args.audit, args.manifest, args.output)
    _emit(
        {
            "status": payload["status"],
            "authorized_to_train": payload["authorized_to_train"],
            "blockers": payload["blockers"],
            "output": str(args.output),
        },
        "human",
    )
    return 0 if payload["status"] == "ready_for_smoke" else 4


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


def cmd_profile_runtime(args: argparse.Namespace) -> int:
    payload = run_runtime_qualification(
        args.output,
        request_count=args.requests,
        node_budget=args.nodes,
        startup_timeout=args.startup_timeout,
        shutdown_timeout=args.shutdown_timeout,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "status": payload["status"],
                "latency_seconds": payload["latency_seconds"],
                "stockfish_lifecycle": payload["stockfish_lifecycle"],
            },
            indent=2,
        )
    )
    return 0 if payload["status"] == "passed" else 4


def cmd_profile_model(args: argparse.Namespace) -> int:
    if args.requests < 2:
        raise ValueError("--requests must include one cold and at least one warm request")
    fixtures = [
        json.loads(line)
        for line in args.fixtures.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not fixtures:
        raise ValueError("The model profile fixture file is empty")
    prompts: list[str] = []
    prompt_evidence = []
    with StockfishProvider(node_budget=args.nodes) as provider:
        for fixture in fixtures:
            evidence = provider.analyze(fixture["fen"], fixture.get("considered_move_uci"))
            if evidence.candidates:
                prompts.append(claim_selection_prompt(evidence, RatingBucket.CLUB))
                prompt_evidence.append(evidence)
        engine_sha256 = provider.binary_sha256
    if not prompts:
        raise ValueError("The model profile fixtures contain no nonterminal positions")
    request_prompts = tuple(prompts[index % len(prompts)] for index in range(args.requests))
    model_id = args.model or (DEFAULT_LFM_MODEL if args.backend == "lmstudio" else DEFAULT_MODEL)
    if args.backend == "lmstudio":
        if args.model_artifact is None:
            raise ValueError("--model-artifact is required for LM Studio profiling")
        profile = profile_lmstudio_generation(
            request_prompts,
            model_id=model_id,
            base_url=args.base_url,
            model_artifact=args.model_artifact,
            max_tokens=args.max_tokens,
            system_prompt="You select grounded chess coaching claims.",
            timeout_seconds=args.timeout,
        )
    else:
        profile = profile_mlx_generation(
            request_prompts,
            model_id=model_id,
            revision=args.revision,
            max_tokens=args.max_tokens,
            system_prompt="You select grounded chess coaching claims.",
            manifest_path=args.manifest,
        )
    request_evidence = tuple(
        prompt_evidence[index % len(prompt_evidence)] for index in range(args.requests)
    )

    def validate_contract(index: int, output: str) -> None:
        selection = parse_claim_selection(output, request_evidence[index])
        valid, _removed = validate_model_claims(request_evidence[index], selection.claims)
        if not valid and not selection.concept_ids:
            raise ValueError("Output contained no production-eligible claim or concept")

    profile = validate_profile_outputs(profile, validate_contract)
    payload = profile.as_dict()
    warm = payload["summary"]["warm_requests"]
    visible_ttft = warm["time_to_first_visible_token_seconds"]
    gates = {
        "warm_visible_ttft_p95_seconds": {
            "required_max": 3.0,
            "actual": visible_ttft["p95"] if visible_ttft else None,
        },
        "warm_total_p95_seconds": {
            "required_max": 10.0,
            "actual": warm["total_latency_seconds"]["p95"],
        },
        "warm_generation_tps_min": {
            "required_min": 20.0,
            "actual": warm["generation_tokens_per_second"]["min"],
        },
        "successful_request_rate": {
            "required_min": 1.0,
            "actual": payload["summary"]["successful_request_rate"],
        },
        "contract_valid_request_rate": {
            "required_min": 1.0,
            "actual": payload["summary"]["contract_valid_request_rate"],
        },
    }
    for gate in gates.values():
        gate["passed"] = gate["actual"] is not None and (
            gate["actual"] <= gate["required_max"]
            if "required_max" in gate
            else gate["actual"] >= gate["required_min"]
        )
    payload.update(
        {
            "status": "passed" if all(item["passed"] for item in gates.values()) else "failed",
            "commit": _git_revision(),
            "working_tree_clean": not bool(subprocess.run(
                ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
            ).stdout.strip()),
            "fixture_path": str(args.fixtures),
            "fixture_sha256": _sha256_path(args.fixtures),
            "engine_sha256": engine_sha256,
            "engine_node_budget": args.nodes,
            "max_tokens": args.max_tokens,
            "gates": gates,
        }
    )
    _write_json_atomic(args.output, payload)
    print(
        json.dumps(
            {"output": str(args.output), "status": payload["status"], "gates": gates},
            indent=2,
        )
    )
    return 0 if payload["status"] == "passed" else 4


def cmd_evaluate_accuracy(args: argparse.Namespace) -> int:
    results: dict[str, Any] = {}
    if args.suite in {"constructed", "all"}:
        path = args.output_dir / "accuracy-constructed.json"
        results["constructed"] = run_constructed_accuracy_benchmark(
            args.fixtures, path, repeats=args.repeats, node_budget=args.nodes
        )
    if args.suite in {"lichess", "all"}:
        path = args.output_dir / "accuracy-lichess.json"
        results["lichess"] = run_lichess_puzzle_accuracy_benchmark(
            args.archive,
            args.manifest,
            path,
            sample_size=args.sample_size,
            node_budget=args.nodes,
        )
    statuses = {name: result["status"] for name, result in results.items()}
    print(json.dumps({"status": statuses, "output_dir": str(args.output_dir)}, indent=2))
    return 0 if all(status == "passed" for status in statuses.values()) else 4


def cmd_evaluate_tutoring(args: argparse.Namespace) -> int:
    payload = run_tutoring_qualification(
        args.cases,
        args.output,
        profile=args.profile,
        repetitions=args.repetitions,
        model_id=args.model
        or (DEFAULT_LFM_MODEL if args.backend == "lmstudio" else DEFAULT_MODEL),
        model_revision=args.revision,
        model_manifest_path=args.manifest,
        model_backend=args.backend,
        model_base_url=args.base_url,
        model_artifact_path=args.model_artifact,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "status": payload["status"],
                "human_usefulness_status": payload["human_usefulness_status"],
                "failure_counts": payload["failure_counts"],
            },
            indent=2,
        )
    )
    return 0 if payload["status"] == "passed" else 4


def _git_revision() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def cmd_verify(args: argparse.Namespace) -> int:
    root = Path.cwd()
    for command in PORTABLE_COMMANDS:
        if run_command(list(command)):
            return 4
    findings = portable_findings(root)
    if args.tier in {"local-alpha", "release"}:
        if run_command(["uv", "run", "pytest", "-m", "hardware", "tests"]):
            return 4
        browser_gate = root / "scripts" / "run-browser-acceptance.sh"
        if not browser_gate.is_file():
            findings.append("scripts/run-browser-acceptance.sh is missing")
        elif run_command([str(browser_gate)]):
            return 4
    if args.tier == "release":
        findings.extend(verify_release_status(root))
    if findings:
        print(json.dumps({"status": "blocked", "findings": findings}, indent=2))
        return 4
    return 0


def cmd_repo_audit(args: argparse.Namespace) -> int:
    payload = audit_repository(Path.cwd())
    print(json.dumps(payload, indent=2))
    return 0 if payload["status"] == "passed" else 4


if __name__ == "__main__":
    raise SystemExit(main())
