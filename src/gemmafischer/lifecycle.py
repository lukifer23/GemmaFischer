from __future__ import annotations

import fcntl
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class InstanceAlreadyRunning(RuntimeError):
    pass


def runtime_dir() -> Path:
    return Path.home() / "Library" / "Application Support" / "GemmaFischer"


def pid_path() -> Path:
    return runtime_dir() / "server.json"


def lock_path() -> Path:
    return runtime_dir() / "server.lock"


class InstanceLock:
    def __init__(self, host: str, port: int, profile: str) -> None:
        self.host = host
        self.port = port
        self.profile = profile
        self._handle: Any = None

    def __enter__(self) -> InstanceLock:
        runtime_dir().mkdir(parents=True, exist_ok=True)
        self._handle = lock_path().open("a+")
        try:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise InstanceAlreadyRunning("A GemmaFischer server is already running") from exc
        payload = {
            "pid": os.getpid(),
            "host": self.host,
            "port": self.port,
            "profile": self.profile,
            "cwd": str(Path.cwd().resolve()),
            "executable": str(Path(sys.executable).resolve()),
            "started_at": datetime.now(UTC).isoformat(),
            "process_started": _process_started(os.getpid()),
        }
        temporary = pid_path().with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        temporary.replace(pid_path())
        return self

    def __exit__(self, *_args: object) -> None:
        current = instance_status()
        if current.get("pid") == os.getpid():
            pid_path().unlink(missing_ok=True)
        if self._handle is not None:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
            self._handle.close()


def instance_status() -> dict[str, Any]:
    try:
        payload = json.loads(pid_path().read_text(encoding="utf-8"))
        pid = int(payload["pid"])
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return {"running": False}
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return {**payload, "running": False, "stale": True}
    except PermissionError:
        return {**payload, "running": False, "permission_denied": True}
    command = subprocess.run(
        ["ps", "-p", str(pid), "-o", "command="],
        check=False,
        capture_output=True,
        text=True,
    ).stdout.strip()
    try:
        arguments = shlex.split(command)
    except ValueError:
        arguments = []
    direct_index = next(
        (
            index
            for index, argument in enumerate(arguments)
            if Path(argument).name == "gemmafischer"
        ),
        None,
    )
    direct = direct_index is not None and "serve" in arguments[direct_index + 1 :]
    module = any(
        arguments[index : index + 3] == ["-m", "gemmafischer.cli", "serve"]
        for index in range(max(0, len(arguments) - 2))
    )
    owned = direct or module
    expected_start = payload.get("process_started")
    actual_start = _process_started(pid)
    identity_matches = bool(expected_start) and expected_start == actual_start
    return {
        **payload,
        "running": owned and identity_matches,
        "command": command,
        "owned": owned,
        "identity_matches": identity_matches,
    }


def _process_started(pid: int) -> str:
    """Return the OS process-start fingerprint used to reject PID reuse."""
    return subprocess.run(
        ["ps", "-p", str(pid), "-o", "lstart="],
        check=False,
        capture_output=True,
        text=True,
    ).stdout.strip()


def stop_instance(timeout: float = 10.0) -> dict[str, Any]:
    status = instance_status()
    if not status.get("running") or not status.get("owned"):
        return {**status, "stopped": False}
    pid = int(status["pid"])
    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            pid_path().unlink(missing_ok=True)
            return {**status, "running": False, "stopped": True}
        time.sleep(0.1)
    return {**status, "running": True, "stopped": False, "timeout": True}
