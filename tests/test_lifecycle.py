from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import gemmafischer.lifecycle as lifecycle


def test_runtime_dir_is_platform_aware(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("GEMMAFISCHER_DATA_DIR", raising=False)
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    monkeypatch.setattr(lifecycle.platform, "system", lambda: "Linux")

    assert lifecycle.runtime_dir() == tmp_path / "gemmafischer"


def test_runtime_dir_honors_explicit_data_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GEMMAFISCHER_DATA_DIR", str(tmp_path))

    assert lifecycle.runtime_dir() == tmp_path


def test_instance_compatibility_requires_complete_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GEMMAFISCHER_DATA_DIR", str(tmp_path))
    status = {
        "running": True,
        "host": "127.0.0.1",
        "port": 8765,
        "profile": "deterministic",
        "application_version": lifecycle.__version__,
        "executable": str(Path(lifecycle.sys.executable).resolve()),
        "data_dir": str(tmp_path.resolve()),
    }

    assert lifecycle.instance_is_compatible(
        status, host="127.0.0.1", port=8765, profile="deterministic"
    )
    assert not lifecycle.instance_is_compatible(
        {**status, "application_version": "0.1.0"},
        host="127.0.0.1",
        port=8765,
        profile="deterministic",
    )


def test_status_rejects_reused_pid_even_when_command_looks_owned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = tmp_path / "server.json"
    record.write_text(
        json.dumps({"pid": 42, "process_started": "old-start"}), encoding="utf-8"
    )
    monkeypatch.setattr(lifecycle, "pid_path", lambda: record)
    monkeypatch.setattr(lifecycle.os, "kill", lambda *_args: None)

    def fake_run(command: list[str], **_kwargs: object) -> SimpleNamespace:
        output = "new-start" if "lstart=" in command else "uv run gemmafischer serve"
        return SimpleNamespace(stdout=output)

    monkeypatch.setattr(lifecycle.subprocess, "run", fake_run)

    status = lifecycle.instance_status()

    assert status["owned"] is True
    assert status["identity_matches"] is False
    assert status["running"] is False


def test_stop_does_not_signal_unverified_process(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        lifecycle,
        "instance_status",
        lambda: {"pid": 42, "running": False, "owned": True, "identity_matches": False},
    )
    signals: list[tuple[int, int]] = []
    monkeypatch.setattr(lifecycle.os, "kill", lambda pid, sig: signals.append((pid, sig)))

    result = lifecycle.stop_instance()

    assert result["stopped"] is False
    assert signals == []


def test_status_recognizes_python_module_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = tmp_path / "server.json"
    record.write_text(
        json.dumps({"pid": 42, "process_started": "same-start"}), encoding="utf-8"
    )
    monkeypatch.setattr(lifecycle, "pid_path", lambda: record)
    monkeypatch.setattr(lifecycle.os, "kill", lambda *_args: None)

    def fake_run(command: list[str], **_kwargs: object) -> SimpleNamespace:
        output = (
            "same-start"
            if "lstart=" in command
            else "/venv/bin/python -m gemmafischer.cli serve --profile deterministic"
        )
        return SimpleNamespace(stdout=output)

    monkeypatch.setattr(lifecycle.subprocess, "run", fake_run)

    status = lifecycle.instance_status()

    assert status["owned"] is True
    assert status["identity_matches"] is True
    assert status["running"] is True
