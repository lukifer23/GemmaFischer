import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

import gemmafischer.cli as cli
from gemmafischer.cli import main, parser


def test_dev_doctor_needs_no_engine(capsys) -> None:
    assert main(["doctor", "--profile", "dev", "--format", "json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert [item["code"] for item in payload["checks"]] == ["PYTHON_VERSION"]


def test_version_contract(capsys) -> None:
    assert main(["version", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == {"application": "0.2.0", "api": "v1", "evidence_schema": "2.0"}


def test_lmstudio_qualification_options_are_explicit() -> None:
    args = parser().parse_args(
        [
            "profile-model",
            "--backend",
            "lmstudio",
            "--model",
            "lfm2.5-2.6b-mlx",
            "--model-artifact",
            "/tmp/model.safetensors",
        ]
    )
    assert args.backend == "lmstudio"
    assert args.model == "lfm2.5-2.6b-mlx"
    assert str(args.model_artifact) == "/tmp/model.safetensors"


def test_runtime_qualification_options_are_explicit() -> None:
    args = parser().parse_args(
        ["profile-runtime", "--requests", "7", "--nodes", "1234", "--output", "/tmp/run.json"]
    )
    assert args.requests == 7
    assert args.nodes == 1234
    assert str(args.output) == "/tmp/run.json"


def test_verify_tiers_are_explicit() -> None:
    assert parser().parse_args(["verify"]).tier == "portable"
    assert parser().parse_args(["verify", "--tier", "release"]).tier == "release"


def test_setup_plan_is_non_mutating_and_lists_homebrew_repair(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "_setup_state", lambda _profile: {
        "engine": {"status": "missing"},
        "full_dependencies": {"status": "not_required"},
        "model": {"status": "not_required"},
    })
    monkeypatch.setattr(cli.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(
        cli.shutil,
        "which",
        lambda name: "/opt/homebrew/bin/brew" if name == "brew" else None,
    )
    calls: list[tuple[str, ...]] = []
    monkeypatch.setattr(
        cli.subprocess,
        "run",
        lambda command, **_kwargs: calls.append(tuple(command)),
    )

    assert main(["setup", "--plan", "--format", "json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["mutating"] is False
    assert payload["actions"][0]["command"] == ["brew", "install", "stockfish"]
    assert calls == []


def test_setup_state_verifies_real_engine_identity_for_deterministic_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binary = tmp_path / "stockfish"
    monkeypatch.setattr(cli, "resolve_stockfish", lambda: binary)
    monkeypatch.setattr(
        cli,
        "inspect_stockfish_binary",
        lambda path: {"name": "Stockfish 18", "path": str(path), "sha256": "abc"},
    )

    state = cli._setup_state("deterministic")

    assert state == {
        "engine": {
            "name": "Stockfish 18",
            "path": str(binary),
            "sha256": "abc",
            "status": "verified-local",
        },
        "full_dependencies": {"status": "not_required"},
        "model": {"status": "not_required"},
    }


def test_setup_repair_requires_explicit_yes(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "_setup_state", lambda _profile: {
        "engine": {"status": "missing"},
        "full_dependencies": {"status": "not_required"},
        "model": {"status": "not_required"},
    })
    monkeypatch.setattr(cli.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(cli.shutil, "which", lambda _name: "/opt/homebrew/bin/brew")

    assert main(["setup", "--repair", "--format", "json"]) == 2
    assert "confirmation_required" in json.loads(capsys.readouterr().out)


def test_setup_confirmed_repair_executes_plan_and_rechecks(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    missing = {
        "engine": {"status": "missing"},
        "full_dependencies": {"status": "not_required"},
        "model": {"status": "not_required"},
    }
    ready = {
        "engine": {"status": "verified-local"},
        "full_dependencies": {"status": "not_required"},
        "model": {"status": "not_required"},
    }
    states = iter((missing, ready))
    monkeypatch.setattr(cli, "_setup_state", lambda _profile: next(states))
    monkeypatch.setattr(cli.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(cli.shutil, "which", lambda _name: "/opt/homebrew/bin/brew")
    calls: list[tuple[str, ...]] = []

    def run(command: tuple[str, ...], **_kwargs: object) -> SimpleNamespace:
        calls.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(cli.subprocess, "run", run)

    assert main(["setup", "--repair", "--yes", "--format", "json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ready"] is True
    assert payload["completed"] == ["INSTALL_STOCKFISH"]
    assert calls == [("brew", "install", "stockfish")]


def test_full_setup_plan_has_one_pinned_model_download(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = {
        "engine": {"status": "verified-local"},
        "full_dependencies": {"status": "missing"},
        "model": {"status": "missing_or_invalid"},
    }
    monkeypatch.setattr(cli.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(cli.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(cli.shutil, "which", lambda name: "/usr/bin/uv" if name == "uv" else None)

    actions, blockers = cli._setup_actions("full", state)

    assert blockers == []
    assert [action[0] for action in actions] == [
        "INSTALL_FULL_DEPENDENCIES",
        "DOWNLOAD_PINNED_MODEL",
    ]
    dependency_install = actions[0][2]
    assert "mlx-lm==0.31.3" in dependency_install
    assert "psutil==7.2.2" in dependency_install
    download = actions[1][2]
    assert download[:2] == (cli.sys.executable, "-c")
    assert download[2].count(cli.DEFAULT_MODEL_REVISION) == 1


def test_linux_setup_plan_uses_apt_without_duplicate_install(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = {
        "engine": {"status": "missing"},
        "full_dependencies": {"status": "not_required"},
        "model": {"status": "not_required"},
    }
    monkeypatch.setattr(cli.platform, "system", lambda: "Linux")
    monkeypatch.setattr(
        cli.shutil,
        "which",
        lambda name: "/usr/bin/apt-get" if name == "apt-get" else None,
    )
    monkeypatch.setattr(cli.os, "geteuid", lambda: 0)

    actions, blockers = cli._setup_actions("deterministic", state)

    assert blockers == []
    assert [action[0] for action in actions] == ["APT_UPDATE", "INSTALL_STOCKFISH"]
    assert sum(action[2].count("stockfish") for action in actions) == 1


def test_setup_fails_closed_without_supported_package_manager(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = {
        "engine": {"status": "missing"},
        "full_dependencies": {"status": "not_required"},
        "model": {"status": "not_required"},
    }
    monkeypatch.setattr(cli.platform, "system", lambda: "Windows")
    monkeypatch.setattr(cli.shutil, "which", lambda _name: None)

    actions, blockers = cli._setup_actions("deterministic", state)

    assert actions == []
    assert blockers == ["Install Homebrew on macOS or use an apt-based Linux system."]


def test_full_profile_fails_closed_off_apple_silicon(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = {
        "engine": {"status": "verified-local"},
        "full_dependencies": {"status": "missing"},
        "model": {"status": "missing_or_invalid"},
    }
    monkeypatch.setattr(cli.platform, "system", lambda: "Linux")
    monkeypatch.setattr(cli.platform, "machine", lambda: "x86_64")

    actions, blockers = cli._setup_actions("full", state)

    assert actions == []
    assert blockers == ["The full MLX profile requires Apple Silicon macOS."]


def test_launch_rejects_running_profile_or_port_mismatch(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        cli,
        "instance_status",
        lambda: {
            "running": True,
            "host": "127.0.0.1",
            "port": 8765,
            "profile": "deterministic",
            "pid": 42,
        },
    )
    args = Namespace(port=9000, profile="full", no_open=True, timeout=1.0)

    assert cli.cmd_launch(args) == 6
    assert "INSTANCE_CONFIG_CONFLICT" in capsys.readouterr().err
