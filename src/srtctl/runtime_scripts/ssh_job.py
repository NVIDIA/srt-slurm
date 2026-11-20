# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free SSH job transport and detached supervisor (Python 3.10+).

The client sends this source to the remote Python over OpenSSH and a request on
stdin. Submit stages files; stop writes a request marker. The supervisor outlives
the SSH connection and owns the Popen it signals; stop requests never signal a PID
read from a stale file.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, TypedDict, cast

POLL_SECONDS = 0.2
STALE_SECONDS = 10
STOP_TIMEOUT_SECONDS = 180
TERMINAL_STATES = ("succeeded", "failed", "stopped")


class Request(TypedDict):
    action: str
    workdir: str
    job_id: str
    uv: str
    venv: str | None
    recipe: str
    selector: str | None
    cluster_config: dict[str, Any]
    args: list[str]
    follow: bool
    source: str


class JobState(TypedDict):
    job_id: str
    state: str
    pid: int | None
    output_dir: str
    log_file: str
    started_at: float
    updated_at: float
    exit_code: int | None
    stop_requested: bool
    error: str | None


def _path(value: str) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(value))).resolve()


def job_directory(workdir: str, job_id: str) -> Path:
    """One private directory per remote submission, independent of the recipe's output setting."""
    if re.fullmatch(r"ssh-[0-9a-f]{32}", job_id) is None:
        raise ValueError("job ID must be the ssh-<32 hex digits> printed by apply")
    return _path(workdir) / "outputs" / "ssh" / job_id


def _write_state(directory: Path, state: JobState) -> None:
    state["updated_at"] = time.time()
    temporary = directory / "status.json.tmp"
    temporary.write_text(json.dumps(state) + "\n", encoding="utf-8")
    temporary.replace(directory / "status.json")


def read_state(directory: Path) -> JobState:
    """Report a stale supervisor as unknown, rather than claiming its engine is still running."""
    state = cast(JobState, json.loads((directory / "status.json").read_text(encoding="utf-8")))
    if state["state"] not in TERMINAL_STATES and time.time() - state["updated_at"] > STALE_SECONDS:
        state["state"] = "unknown"
        state["error"] = "Supervisor heartbeat is stale; inspect apply.log and the remote machine"
    return state


def _environment(request: Request, directory: Path) -> dict[str, str]:
    # This is the REMOTE login environment, never the submitting laptop's environment.
    env = {key: value for key, value in os.environ.items() if not key.startswith("SLURM_")}
    for key in ("SRTCTL_JOB_ID", "SRTCTL_HOST_PATH", "SRTCTL_HOST_VIRTUAL_ENV"):
        env.pop(key, None)
    env["SRTSLURM_CONFIG"] = str(directory / "srtslurm.yaml")
    env["UV_PROJECT_ENVIRONMENT"] = str(_path(request["workdir"]) / ".venv")
    if request["venv"] is not None:
        venv = _path(request["venv"])
        if not (venv / "bin" / "python3").is_file():
            raise ValueError(f"ssh.venv has no bin/python3: {venv}")
        env["VIRTUAL_ENV"] = str(venv)
        env["PATH"] = str(venv / "bin") + os.pathsep + env.get("PATH", os.defpath)
    return env


def _command(request: Request, directory: Path) -> list[str]:
    uv = os.path.expanduser(os.path.expandvars(request["uv"]))
    recipe = str(directory / "recipe.yaml")
    if request["selector"]:
        recipe += ":" + request["selector"]
    args = request["args"]
    output = (
        []
        if "--output" in args or request["cluster_config"].get("output_dir")
        else ["--output", str(directory / "runs")]
    )
    return [
        uv,
        "run",
        "--python",
        "3.12",
        "--no-dev",
        "-m",
        "srtctl.cli.submit",
        "apply",
        "--yes",
        "--json",
        "--file",
        recipe,
        *output,
        *args,
    ]


def submit(request: Request) -> JobState:
    """Stage the recipe and cluster config, then hand the complete apply to a detached supervisor."""
    workdir = _path(request["workdir"])
    if not (workdir / "src" / "srtctl" / "core" / "host.py").is_file():
        raise ValueError(f"ssh.workdir must contain an srt-slurm checkout with launcher: host support: {workdir}")
    directory = job_directory(request["workdir"], request["job_id"])
    env = _environment(request, directory)
    command = _command(request, directory)
    if shutil.which(command[0], path=env.get("PATH")) is None:
        raise ValueError(f"ssh.uv executable is unavailable on the remote machine: {command[0]}")
    directory.mkdir(parents=True, mode=0o700)
    directory.chmod(0o700)
    (directory / "recipe.yaml").write_text(request["recipe"], encoding="utf-8")
    # JSON is valid YAML. Resolve only remote-owned paths here, on the target.
    cluster = {**request["cluster_config"], "srtctl_root": str(workdir)}
    # The target only needs the underlying host launcher, not the SSH client feature.
    # Older cluster schemas reject even an unknown key whose value is null.
    cluster.pop("ssh", None)
    (directory / "srtslurm.yaml").write_text(json.dumps(cluster), encoding="utf-8")
    (directory / "request.json").write_text(json.dumps(request), encoding="utf-8")
    helper = directory / "supervisor.py"
    helper.write_text(request["source"], encoding="utf-8")
    now = time.time()
    state: JobState = {
        "job_id": request["job_id"],
        "state": "starting",
        "pid": None,
        "output_dir": str(directory),
        "log_file": str(directory / "apply.log"),
        "started_at": now,
        "updated_at": now,
        "exit_code": None,
        "stop_requested": False,
        "error": None,
    }
    _write_state(directory, state)
    with (directory / "apply.log").open("a", encoding="utf-8") as log:
        supervisor = subprocess.Popen(
            [sys.executable, str(helper), "--supervise", str(directory)],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    # A bounded startup acknowledgement catches an unlaunchable helper before apply reports success.
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        state = read_state(directory)
        if state["state"] != "starting":
            return state
        if supervisor.poll() is not None:
            raise RuntimeError(f"Remote supervisor failed to start; inspect {directory / 'apply.log'}")
        time.sleep(POLL_SECONDS)
    raise RuntimeError(f"Remote job startup unconfirmed; inspect {directory} before retrying")


def supervise(directory: Path) -> None:
    """Keep job state and a stop channel alive after the submitting SSH connection closes."""
    request = cast(Request, json.loads((directory / "request.json").read_text(encoding="utf-8")))
    state = read_state(directory)
    state["pid"] = os.getpid()
    stopping = False

    def stop_requested(_signum: int, _frame: Any) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, stop_requested)
    signal.signal(signal.SIGINT, stop_requested)
    child: subprocess.Popen | None = None
    try:
        child = subprocess.Popen(
            _command(request, directory),
            cwd=_path(request["workdir"]),
            env=_environment(request, directory),
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
        state["state"] = "running"
        _write_state(directory, state)
        stop_at: float | None = None
        while child.poll() is None:
            stopping = stopping or (directory / "stop.requested").exists()
            if stopping and stop_at is None:
                # Reuse Ctrl+C handling: apply keeps streaming while the orchestrator
                # cleans up its engine groups. TERM would kill apply before that wait.
                stop_at = time.monotonic()
                state["state"] = "stopping"
                state["stop_requested"] = True
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(child.pid, signal.SIGINT)
            if stop_at is not None and time.monotonic() - stop_at > STOP_TIMEOUT_SECONDS:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(child.pid, signal.SIGKILL)
                state["error"] = "Apply exceeded the stop timeout; inspect the remote machine for surviving processes"
            _write_state(directory, state)
            time.sleep(POLL_SECONDS)
        state["exit_code"] = child.wait()
        if state["error"]:
            state["state"] = "failed"
        elif stopping:
            state["state"] = "stopped"
        else:
            state["state"] = "succeeded" if state["exit_code"] == 0 else "failed"
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        state["state"] = "failed"
        state["error"] = str(exc)
        state["exit_code"] = 1
        print(f"SSH job failed: {exc}", file=sys.stderr, flush=True)
    finally:
        # A supervisor failure (for example, a failed heartbeat write) must also
        # stop its apply, rather than abandon a live engine behind stale state.
        if child is not None and child.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(child.pid, signal.SIGINT)
            try:
                child.wait(timeout=STOP_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=5)
        _write_state(directory, state)


def logs(directory: Path, follow: bool) -> None:
    """Stream saved output without coupling job lifetime to this reader."""
    with (directory / "apply.log").open(encoding="utf-8", errors="replace") as log:
        while True:
            chunk = log.read(65536)
            if chunk:
                sys.stdout.write(chunk)
                sys.stdout.flush()
            elif not follow or read_state(directory)["state"] in (*TERMINAL_STATES, "unknown"):
                return
            else:
                time.sleep(POLL_SECONDS)


def handle(request: Request) -> JobState | list[JobState] | None:
    if request["action"] == "submit":
        return submit(request)
    if request["action"] == "list":
        root = _path(request["workdir"]) / "outputs" / "ssh"
        return [read_state(path) for path in sorted(root.glob("ssh-*/")) if (path / "status.json").is_file()]
    directory = job_directory(request["workdir"], request["job_id"])
    state = read_state(directory)
    if request["action"] == "status":
        return state
    if request["action"] == "stop":
        if state["state"] == "unknown":
            raise RuntimeError(state["error"])
        if state["state"] not in TERMINAL_STATES:
            (directory / "stop.requested").touch()
            state["stop_requested"] = True
        return state
    if request["action"] == "logs":
        logs(directory, request["follow"])
        return None
    raise ValueError(f"Unknown SSH job action: {request['action']}")


def main() -> int:
    if len(sys.argv) == 3 and sys.argv[1] == "--supervise":
        supervise(Path(sys.argv[2]))
        return 0
    try:
        request = cast(Request, json.load(sys.stdin))
        result = handle(request)
        if result is not None:
            print(json.dumps(result), flush=True)
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"SSH job: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
