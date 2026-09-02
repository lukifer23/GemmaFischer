from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
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
from .engine import (
    EngineUnavailable,
    StockfishProvider,
    inspect_stockfish_binary,
    resolve_stockfish,
)
from .labeling import (
    adjudicate_label_responses,
    apply_human_gold,
    export_label_packet,
    validate_label_responses,
)
from .lifecycle import (
    InstanceAlreadyRunning,
    InstanceLock,
    instance_is_compatible,
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
from .question_eval import freeze_question_cases, run_question_grading_qualification
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
from .training import (
    acquire_native_training_model,
    package_training_artifact,
    prepare_mlx_dataset,
    run_mlx_sft,
    training_preflight,
    validate_training_preflight,
)
from .training_eval import evaluate_untuned_training_baseline, freeze_error_taxonomy
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
    data_audit.add_argument("--data-dir", type=Path, default=Path("data/derived/v2"))
    readiness = commands.add_parser(
        "training-readiness", help="Fail-closed data, hardware, and toolchain gate"
    )
    readiness.add_argument("--audit", type=Path, default=Path("artifacts/data-audit/latest.json"))
    prepare_training = commands.add_parser(
        "prepare-training-data", help="Validate and convert canonical v2 data for MLX"
    )
    prepare_training.add_argument("--source-dir", type=Path, default=Path("data/derived/v2"))
    prepare_training.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/training/mlx-data")
    )
    prepare_training.add_argument("--human-gold", type=Path)
    acquire_training_model = commands.add_parser(
        "acquire-training-model",
        help="Download and hash-verify the one pinned native training base",
    )
    acquire_training_model.add_argument(
        "--manifest", type=Path, default=bundled_path("training/post-training.json")
    )
    acquire_training_model.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/training/native-model.json"),
    )
    label_export = commands.add_parser(
        "label-export", help="Export a blind human-review packet from canonical data"
    )
    label_export.add_argument("--dataset", type=Path, required=True)
    label_export.add_argument("--output", type=Path, required=True)
    label_export.add_argument("--limit", type=int, default=2_500)
    label_validate = commands.add_parser(
        "label-validate", help="Validate human selections against the exact v2 catalog"
    )
    label_validate.add_argument("--dataset", type=Path, required=True)
    label_validate.add_argument("--responses", type=Path, required=True)
    label_validate.add_argument("--output", type=Path, required=True)
    label_adjudicate = commands.add_parser(
        "label-adjudicate", help="Resolve every material two-reviewer disagreement"
    )
    label_adjudicate.add_argument("--dataset", type=Path, required=True)
    label_adjudicate.add_argument("--validation", type=Path, required=True)
    label_adjudicate.add_argument("--adjudications", type=Path, required=True)
    label_adjudicate.add_argument("--output", type=Path, required=True)
    label_apply = commands.add_parser(
        "label-apply", help="Apply adjudicated human selections to canonical train rows"
    )
    label_apply.add_argument("--source-dir", type=Path, default=Path("data/derived/v2"))
    label_apply.add_argument("--human-gold", type=Path, required=True)
    label_apply.add_argument("--output-dir", type=Path, default=Path("data/derived/v2-reviewed"))
    question_freeze = commands.add_parser(
        "freeze-question-eval", help="Freeze an engine-grounded final-test question set"
    )
    question_freeze.add_argument(
        "--dataset", type=Path, default=Path("data/derived/v2/final_test.jsonl")
    )
    question_freeze.add_argument(
        "--output", type=Path, default=Path("artifacts/evaluation/questions-v2.jsonl")
    )
    question_freeze.add_argument("--limit", type=int, default=1_000)
    question_grade = commands.add_parser(
        "evaluate-questions", help="Validate the frozen deterministic question grader"
    )
    question_grade.add_argument(
        "--questions", type=Path, default=Path("artifacts/evaluation/questions-v2.jsonl")
    )
    question_grade.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/qualification/questions-v2.json"),
    )
    baseline = commands.add_parser(
        "evaluate-training-baseline", help="Run the real untuned model on validation rows"
    )
    baseline.add_argument("--dataset", type=Path, default=Path("data/derived/v2/validation.jsonl"))
    baseline.add_argument(
        "--output", type=Path, default=Path("artifacts/training/untuned-baseline.json")
    )
    baseline.add_argument("--limit", type=int, default=250)
    taxonomy = commands.add_parser(
        "freeze-error-taxonomy", help="Freeze observed errors from an untuned baseline"
    )
    taxonomy.add_argument(
        "--baseline", type=Path, default=Path("artifacts/training/untuned-baseline.json")
    )
    taxonomy.add_argument(
        "--output", type=Path, default=Path("artifacts/training/error-taxonomy.json")
    )
    preflight = commands.add_parser(
        "training-preflight", help="Verify data, native weights, toolchain, and evidence"
    )
    preflight.add_argument(
        "--manifest", type=Path, default=bundled_path("training/post-training.json")
    )
    preflight.add_argument("--audit", type=Path, default=Path("artifacts/data-audit/latest.json"))
    preflight.add_argument("--model", type=Path, required=True)
    preflight.add_argument("--data", type=Path, default=Path("artifacts/training/mlx-data"))
    preflight.add_argument(
        "--output", type=Path, default=Path("artifacts/training/preflight-latest.json")
    )
    preflight.add_argument("--config", type=Path, default=bundled_path("training/mlx-lora.yaml"))
    for name, smoke in (("train-smoke", True), ("train-sft", False)):
        train = commands.add_parser(name, help=f"Run real MLX {'smoke' if smoke else 'SFT'}")
        train.set_defaults(training_smoke=smoke)
        train.add_argument("--preflight", type=Path, required=True)
        train.add_argument("--model", type=Path, required=True)
        train.add_argument("--data", type=Path, required=True)
        train.add_argument("--adapter", type=Path, required=True)
        train.add_argument("--receipt", type=Path, required=True)
        train.add_argument("--iterations", type=int, default=7 if smoke else 1000)
        train.add_argument("--max-seq-length", type=int, default=1024)
        train.add_argument("--config", type=Path, default=bundled_path("training/mlx-lora.yaml"))
    package = commands.add_parser(
        "package-adapter", help="Package exactly one adapter and its receipts"
    )
    package.add_argument("--adapter", type=Path, required=True)
    package.add_argument("--receipt", type=Path, action="append", required=True)
    package.add_argument("--output", type=Path, required=True)
    readiness.add_argument(
        "--manifest", type=Path, default=bundled_path("training/post-training.json")
    )
    readiness.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/training/readiness-latest.json"),
    )
    acquire = commands.add_parser("acquire-data", help="Download and verify a pinned source")
    acquire.add_argument("--manifest", type=Path, default=Path("data/sources.json"))
    acquire.add_argument("--source", default="lichess-puzzles-2026-08-02")
    acquire.add_argument("--output", type=Path, default=Path("data/raw/lichess_db_puzzle.csv.zst"))
    build_data = commands.add_parser(
        "build-dataset", help="Build typed, lineage-isolated lesson records"
    )
    build_data.add_argument("--manifest", type=Path, default=Path("data/sources.json"))
    build_data.add_argument("--source", default="lichess-puzzles-2026-08-02")
    build_data.add_argument(
        "--archive", type=Path, default=Path("data/raw/lichess_db_puzzle.csv.zst")
    )
    build_data.add_argument("--output-dir", type=Path, default=Path("data/derived/v2"))
    build_data.add_argument("--limit", type=int, default=15_000)
    build_data.add_argument("--nodes", type=int, default=250_000)
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
    model_profile.add_argument("--revision", default=DEFAULT_MODEL_REVISION)
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
    accuracy.add_argument("--output-dir", type=Path, default=Path("artifacts/qualification"))
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
    tutor.add_argument("--revision", default="238767527555cb75a05732a84dff5d6ba0dd6809")
    tutor.add_argument("--manifest", type=Path, default=bundled_path("assets/model-manifest.json"))
    tutor.add_argument("--base-url", default=DEFAULT_LM_STUDIO_URL)
    tutor.add_argument("--model-artifact", type=Path)
    tutor.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/qualification/tutoring-latest.json"),
    )
    benchmark.add_argument("--profile", choices=("deterministic", "full"), default="deterministic")
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
    if args.yes and not args.repair:
        raise ValueError("--yes is only valid with --repair")
    state = _setup_state(args.profile)
    actions, blockers = _setup_actions(args.profile, state)
    payload: dict[str, object] = {
        "profile": args.profile,
        "mode": "repair" if args.repair else ("plan" if args.plan else "inspect"),
        "mutating": bool(args.repair and args.yes),
        "ready": not actions and not blockers,
        "state": state,
        "actions": [
            {"code": code, "description": description, "command": list(command)}
            for code, description, command in actions
        ],
        "blockers": blockers,
    }
    if args.repair and not args.yes:
        payload["confirmation_required"] = "Repeat with --repair --yes to execute this plan."
        _emit(payload, args.format)
        return 2
    if args.repair and args.yes:
        if blockers:
            _emit(payload, args.format)
            return 3
        completed: list[str] = []
        for code, _description, command in actions:
            result = subprocess.run(command, check=False)
            if result.returncode:
                payload["completed"] = completed
                payload["failed_action"] = code
                _emit(payload, args.format)
                return 4
            completed.append(code)
        state = _setup_state(args.profile)
        remaining, blockers = _setup_actions(args.profile, state)
        payload.update(
            {
                "state": state,
                "completed": completed,
                "actions": [
                    {"code": code, "description": description, "command": list(command)}
                    for code, description, command in remaining
                ],
                "blockers": blockers,
                "ready": not remaining and not blockers,
            }
        )
    _emit(payload, args.format)
    if args.plan:
        return 0
    return 0 if payload["ready"] else 3


def _setup_state(profile: str) -> dict[str, object]:
    try:
        engine = resolve_stockfish()
        identity = inspect_stockfish_binary(engine)
        engine_info: dict[str, object] = {
            **identity,
            "status": "verified-local",
        }
    except EngineUnavailable:
        engine_info = {"status": "missing"}
    state: dict[str, object] = {"engine": engine_info}
    if profile == "full":
        try:
            import mlx_lm  # noqa: F401

            state["full_dependencies"] = {"status": "installed"}
        except ImportError:
            state["full_dependencies"] = {"status": "missing"}
        try:
            state["model"] = inspect_model_assets(DEFAULT_MODEL, DEFAULT_MODEL_REVISION)
        except ModelUnavailable as exc:
            state["model"] = {"status": "missing_or_invalid", "message": str(exc)}
    else:
        state["full_dependencies"] = {"status": "not_required"}
        state["model"] = {"status": "not_required"}
    return state


def _setup_actions(
    profile: str, state: dict[str, object]
) -> tuple[list[tuple[str, str, tuple[str, ...]]], list[str]]:
    actions: list[tuple[str, str, tuple[str, ...]]] = []
    blockers: list[str] = []
    engine = state["engine"]
    assert isinstance(engine, dict)
    if engine.get("status") != "verified-local":
        if platform.system() == "Darwin" and shutil.which("brew"):
            actions.append(
                (
                    "INSTALL_STOCKFISH",
                    "Install Stockfish with Homebrew.",
                    ("brew", "install", "stockfish"),
                )
            )
        elif platform.system() == "Linux" and shutil.which("apt-get"):
            prefix = () if os.geteuid() == 0 else (("sudo",) if shutil.which("sudo") else ())
            if os.geteuid() != 0 and not prefix:
                blockers.append("Stockfish repair needs root access or sudo for apt-get.")
            else:
                actions.extend(
                    (
                        (
                            "APT_UPDATE",
                            "Refresh apt package metadata.",
                            (*prefix, "apt-get", "update"),
                        ),
                        (
                            "INSTALL_STOCKFISH",
                            "Install Stockfish with apt.",
                            (*prefix, "apt-get", "install", "-y", "stockfish"),
                        ),
                    )
                )
        else:
            blockers.append("Install Homebrew on macOS or use an apt-based Linux system.")
    if profile == "full":
        if platform.system() != "Darwin" or platform.machine() not in {"arm64", "aarch64"}:
            blockers.append("The full MLX profile requires Apple Silicon macOS.")
            return actions, blockers
        dependencies = state["full_dependencies"]
        assert isinstance(dependencies, dict)
        if dependencies.get("status") != "installed":
            command: tuple[str, ...]
            if shutil.which("uv"):
                command = (
                    "uv",
                    "pip",
                    "install",
                    "--python",
                    sys.executable,
                    "mlx-lm==0.31.3",
                    "psutil==7.2.2",
                )
            else:
                command = (
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "mlx-lm==0.31.3",
                    "psutil==7.2.2",
                )
            actions.append(
                (
                    "INSTALL_FULL_DEPENDENCIES",
                    "Install the pinned full-profile runtime.",
                    command,
                )
            )
        model = state["model"]
        assert isinstance(model, dict)
        if model.get("status") != "verified-local":
            script = (
                "from huggingface_hub import snapshot_download; "
                f"snapshot_download(repo_id={DEFAULT_MODEL!r}, revision={DEFAULT_MODEL_REVISION!r})"
            )
            actions.append(
                (
                    "DOWNLOAD_PINNED_MODEL",
                    "Download the exact model revision; verification runs after download.",
                    (sys.executable, "-c", script),
                )
            )
    return actions, blockers


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
            checks.append({"code": "STOCKFISH", "ok": True, **inspect_stockfish_binary(path)})
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
        if not instance_is_compatible(
            current,
            host="127.0.0.1",
            port=args.port,
            profile=args.profile,
        ):
            print(
                "INSTANCE_CONFIG_CONFLICT: running instance uses "
                f"profile={current.get('profile')} port={current.get('port')}; "
                "stop it before launching a different configuration",
                file=sys.stderr,
            )
            return 6
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
                    status = instance_status()
                    if not instance_is_compatible(
                        status,
                        host="127.0.0.1",
                        port=args.port,
                        profile=args.profile,
                    ):
                        if status.get("running"):
                            print(
                                "INSTANCE_CONFIG_CONFLICT: health endpoint belongs to an "
                                "incompatible managed instance",
                                file=sys.stderr,
                            )
                            return 6
                        time.sleep(0.1)
                        continue
                    if not args.no_open:
                        webbrowser.open(url)
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
    training_paths = sorted(args.data_dir.glob("train*.jsonl"))
    validation_paths = sorted(args.data_dir.glob("validation*.jsonl"))
    evaluation_paths = sorted(args.data_dir.glob("final_test*.jsonl"))
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
            "conflicting_best_move_fens": payload["cross_dataset"]["conflicting_best_move_fens"],
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


def cmd_prepare_training_data(args: argparse.Namespace) -> int:
    payload = prepare_mlx_dataset(args.source_dir, args.output_dir, human_gold_path=args.human_gold)
    _emit({**payload, "output": str(args.output_dir)}, "human")
    return 0


def cmd_acquire_training_model(args: argparse.Namespace) -> int:
    payload = acquire_native_training_model(args.manifest, args.output)
    _emit(
        {
            "status": payload["status"],
            "model_id": payload["model_id"],
            "revision": payload["revision"],
            "snapshot_path": payload["snapshot_path"],
            "snapshot_count": payload["snapshot_count"],
            "output": str(args.output),
        },
        "human",
    )
    return 0


def cmd_label_export(args: argparse.Namespace) -> int:
    payload = export_label_packet(args.dataset, args.output, limit=args.limit)
    _emit(payload, "human")
    return 0


def cmd_label_validate(args: argparse.Namespace) -> int:
    payload = validate_label_responses(args.dataset, args.responses, args.output)
    _emit(
        {
            "status": payload["status"],
            "responses": payload["response_count"],
            "reviewers": payload["reviewer_count"],
            "output": str(args.output),
        },
        "human",
    )
    return 0 if payload["status"] == "passed" else 4


def cmd_label_adjudicate(args: argparse.Namespace) -> int:
    payload = adjudicate_label_responses(
        args.dataset, args.validation, args.adjudications, args.output
    )
    _emit(payload, "human")
    return 0


def cmd_label_apply(args: argparse.Namespace) -> int:
    payload = apply_human_gold(args.source_dir, args.human_gold, args.output_dir)
    _emit(payload, "human")
    return 0


def cmd_freeze_question_eval(args: argparse.Namespace) -> int:
    payload = freeze_question_cases(args.dataset, args.output, limit=args.limit)
    _emit(payload, "human")
    return 0


def cmd_evaluate_questions(args: argparse.Namespace) -> int:
    payload = run_question_grading_qualification(args.questions, args.output)
    _emit(
        {
            "status": payload["status"],
            "questions": payload["summary"]["case_count"],
            "agreement": payload["summary"]["grading_agreement_rate"],
            "output": str(args.output),
        },
        "human",
    )
    return 0 if payload["status"] == "passed" else 4


def cmd_evaluate_training_baseline(args: argparse.Namespace) -> int:
    payload = evaluate_untuned_training_baseline(args.dataset, args.output, limit=args.limit)
    _emit(
        {
            "status": payload["status"],
            "records": payload["record_count"],
            "contract_valid_rate": payload["contract_valid_rate"],
            "exact_target_match_rate": payload["exact_target_match_rate"],
            "output": str(args.output),
        },
        "human",
    )
    return 0


def cmd_freeze_error_taxonomy(args: argparse.Namespace) -> int:
    payload = freeze_error_taxonomy(args.baseline, args.output)
    _emit(payload, "human")
    return 0


def cmd_training_preflight(args: argparse.Namespace) -> int:
    payload = training_preflight(
        args.manifest, args.audit, args.model, args.data, args.output, args.config
    )
    _emit(
        {
            "status": payload["status"],
            "smoke_ready": payload["smoke_ready"],
            "blockers": payload["blockers"],
            "output": str(args.output),
        },
        "human",
    )
    return 0 if payload["smoke_ready"] else 4


def _run_training_command(args: argparse.Namespace) -> int:
    validate_training_preflight(
        args.preflight,
        args.model,
        args.data,
        args.config,
        production=not args.training_smoke,
    )
    payload = run_mlx_sft(
        model_path=args.model,
        data_path=args.data,
        adapter_path=args.adapter,
        receipt_path=args.receipt,
        iterations=args.iterations,
        max_seq_length=args.max_seq_length,
        smoke=args.training_smoke,
        config_path=args.config,
    )
    _emit(payload, "human")
    return 0


def cmd_train_smoke(args: argparse.Namespace) -> int:
    return _run_training_command(args)


def cmd_train_sft(args: argparse.Namespace) -> int:
    return _run_training_command(args)


def cmd_package_adapter(args: argparse.Namespace) -> int:
    payload = package_training_artifact(args.adapter, args.receipt, args.output)
    _emit(payload, "human")
    return 0


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
            "working_tree_clean": not bool(
                subprocess.run(
                    ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
                ).stdout.strip()
            ),
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
        model_id=args.model or (DEFAULT_LFM_MODEL if args.backend == "lmstudio" else DEFAULT_MODEL),
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
        if run_command(
            [
                "uv",
                "run",
                "pytest",
                "--cov=gemmafischer",
                "--cov-report=term-missing",
                "--cov-fail-under=70",
                "-m",
                "not model",
                "tests",
            ]
        ):
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
