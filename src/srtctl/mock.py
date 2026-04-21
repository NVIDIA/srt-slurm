# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Local mock runtime for srtctl.

Runs the full SweepOrchestrator path — head infra, workers, frontend,
telemetry, benchmark, postprocess — without launching any srun processes or
touching real cluster infrastructure. Every external surface that would
reach slurm, a network port, or a subprocess is swapped for a local fake
while the orchestration code itself runs for real.

Designed for end-to-end tests of upstream harnesses (ibar) that need to
observe a plausible sequence of artifacts appearing under an output_dir.
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import patch

__all__ = [
    "FakePopen",
    "MockOptions",
    "mock_infrastructure",
    "run_mock_sweep",
]


# ---------------------------------------------------------------------------
# Fake subprocess.Popen that drop-in replaces an srun child.
# ---------------------------------------------------------------------------


class FakePopen:
    """subprocess.Popen stand-in for srun children.

    Writes a small banner to the provided output file on construction, sleeps
    `duration_s` in wall time (in the background), then reports exit 0. Fully
    compatible with the subset of the Popen API that srtctl actually uses:
    poll(), wait(), terminate(), kill(), returncode, pid, stdout/stderr/stdin
    as None (since we never open pipes).
    """

    _next_pid = 42000
    _pid_lock = threading.Lock()

    def __init__(self, *, cmd: list[str], output: str | None, duration_s: float) -> None:
        with FakePopen._pid_lock:
            FakePopen._next_pid += 1
            self.pid = FakePopen._next_pid
        self.args = list(cmd)
        self._output = Path(output) if output else None
        self._duration = max(0.0, duration_s)
        self._start = time.monotonic()
        self._returncode: int | None = None
        self.stdout = None
        self.stderr = None
        self.stdin = None

    @property
    def returncode(self) -> int | None:
        # Match subprocess.Popen — .poll() sets returncode as a side effect,
        # so force one poll before the attribute is read.
        self.poll()
        return self._returncode

        if self._output is not None:
            self._output.parent.mkdir(parents=True, exist_ok=True)
            header = (
                f"[mock-srun pid={self.pid}] started at {time.strftime('%H:%M:%S')}\n"
                f"[mock-srun pid={self.pid}] cmd: {' '.join(self.args[:6])}"
                + ("..." if len(self.args) > 6 else "")
                + "\n"
            )
            self._output.write_text(header)

    def _finalize(self) -> None:
        if self._returncode is not None:
            return
        self._returncode = 0
        if self._output is not None:
            with self._output.open("a") as f:
                f.write(f"[mock-srun pid={self.pid}] exit=0\n")

    def poll(self) -> int | None:
        if self._returncode is not None:
            return self._returncode
        if (time.monotonic() - self._start) >= self._duration:
            self._finalize()
        return self._returncode

    def wait(self, timeout: float | None = None) -> int:
        deadline = None if timeout is None else time.monotonic() + timeout
        while self.poll() is None:
            if deadline is not None and time.monotonic() >= deadline:
                raise subprocess.TimeoutExpired(cmd=self.args, timeout=timeout or 0.0)
            time.sleep(0.02)
        assert self._returncode is not None
        return self._returncode

    def terminate(self) -> None:
        if self._returncode is None:
            self._returncode = -15

    def kill(self) -> None:
        if self._returncode is None:
            self._returncode = -9

    def communicate(self, input: Any = None, timeout: float | None = None):  # noqa: A002
        self.wait(timeout=timeout)
        return (b"", b"")

    def send_signal(self, sig: int) -> None:
        if self._returncode is None:
            self._returncode = -abs(sig)


# ---------------------------------------------------------------------------
# MockOptions and the fake srun factory.
# ---------------------------------------------------------------------------


@dataclass
class MockOptions:
    """Tunables for a single mock run."""

    # How long each fake srun child pretends to run. Kept small for tests.
    child_duration_s: float = 0.4
    # Pause inserted between major phases (head_infra, workers, frontend,
    # benchmark, postprocess) so watchers observe distinct transitions.
    phase_pause_s: float = 0.25
    # Fake nodelist used by RuntimeContext.
    nodelist: tuple[str, ...] = ("mock-node-01",)
    # Fake IP returned for every hostname lookup.
    hostname_ip: str = "127.0.0.1"


# ---------------------------------------------------------------------------
# Status sink: replaces requests.put/post so status reports land on disk.
# ---------------------------------------------------------------------------


@dataclass
class _StatusSink:
    status_file: Path
    events_file: Path
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def write(self, payload: dict) -> None:
        with self._lock:
            self.events_file.parent.mkdir(parents=True, exist_ok=True)
            with self.events_file.open("a") as f:
                f.write(json.dumps(payload) + "\n")

            current: dict = {}
            if self.status_file.exists():
                try:
                    current = json.loads(self.status_file.read_text())
                except Exception:
                    current = {}
            merged = {**current, **payload}
            # Keep a tally of updates for debugging.
            merged["update_count"] = int(current.get("update_count", 0)) + 1
            self.status_file.write_text(json.dumps(merged, indent=2) + "\n")


class _FakeResponse:
    def __init__(self, status_code: int = 200) -> None:
        self.status_code = status_code

    def json(self) -> dict:
        return {}


# ---------------------------------------------------------------------------
# The infrastructure patch context.
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def mock_infrastructure(*, options: MockOptions, output_dir: Path):
    """Patch every external call so SweepOrchestrator runs fully in-process.

    Parameters
    ----------
    options: MockOptions
        Tunables for the mock run (child durations, phase pauses, nodelist).
    output_dir: Path
        The per-job output directory (.../outputs/<job_id>). The mock writes
        status.json and status_events.jsonl directly under this path so
        external watchers have a canonical source of truth.
    """

    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    status_sink = _StatusSink(
        status_file=output_dir / "status.json",
        events_file=output_dir / "status_events.jsonl",
    )

    # Seed status.json with an initial "pending" row so watchers see a file
    # the moment the mock starts.
    status_sink.write(
        {
            "status": "pending",
            "phase": "created",
            "ts": time.time(),
            "message": "mock sweep starting",
        }
    )

    def _fake_srun(*args, **kwargs) -> FakePopen:
        cmd = kwargs.get("command") or (args[0] if args else [])
        if not isinstance(cmd, list):
            cmd = [str(cmd)]
        return FakePopen(
            cmd=cmd,
            output=kwargs.get("output"),
            duration_s=options.child_duration_s,
        )

    def _fake_wait_for_port(*_args, **_kwargs) -> bool:
        return True

    def _fake_wait_for_model(*_args, **_kwargs) -> bool:
        return True

    def _fake_hostname_ip(*_args, **_kwargs) -> str:
        return options.hostname_ip

    def _fake_nodelist() -> list[str]:
        return list(options.nodelist)

    def _fake_put(url, json=None, timeout=None, **_kwargs):  # noqa: A002
        payload = dict(json or {})
        payload["_sink"] = "put"
        payload["_url"] = url
        status_sink.write(payload)
        return _FakeResponse(200)

    def _fake_post(url, json=None, timeout=None, **_kwargs):  # noqa: A002
        payload = dict(json or {})
        payload["_sink"] = "post"
        payload["_url"] = url
        status_sink.write(payload)
        return _FakeResponse(201)

    # Patches go onto every module that imported these symbols by name.
    patch_targets: list[tuple[str, Any]] = [
        # srun process starters.
        ("srtctl.core.slurm.start_srun_process", _fake_srun),
        ("srtctl.cli.do_sweep.start_srun_process", _fake_srun),
        ("srtctl.cli.mixins.worker_stage.start_srun_process", _fake_srun),
        ("srtctl.cli.mixins.frontend_stage.start_srun_process", _fake_srun),
        ("srtctl.cli.mixins.telemetry_stage.start_srun_process", _fake_srun),
        ("srtctl.cli.mixins.benchmark_stage.start_srun_process", _fake_srun),
        ("srtctl.cli.mixins.postprocess_stage.start_srun_process", _fake_srun),
        ("srtctl.frontends.dynamo.start_srun_process", _fake_srun),
        ("srtctl.frontends.sglang.start_srun_process", _fake_srun),
        # Hostname / IP resolution.
        ("srtctl.core.slurm.get_hostname_ip", _fake_hostname_ip),
        ("srtctl.core.slurm.get_slurm_nodelist", _fake_nodelist),
        ("srtctl.core.runtime.get_hostname_ip", _fake_hostname_ip),
        ("srtctl.core.runtime.get_slurm_nodelist", _fake_nodelist),
        ("srtctl.core.telemetry.get_hostname_ip", _fake_hostname_ip),
        ("srtctl.cli.mixins.frontend_stage.get_hostname_ip", _fake_hostname_ip),
        ("srtctl.cli.mixins.benchmark_stage.get_hostname_ip", _fake_hostname_ip),
        ("srtctl.frontends.sglang.get_hostname_ip", _fake_hostname_ip),
        # Port / model readiness checks.
        ("srtctl.core.health.wait_for_port", _fake_wait_for_port),
        ("srtctl.cli.do_sweep.wait_for_port", _fake_wait_for_port),
        ("srtctl.core.health.wait_for_model", _fake_wait_for_model),
        ("srtctl.cli.mixins.benchmark_stage.wait_for_model", _fake_wait_for_model),
        # Status POST/PUT — redirect to the on-disk sink so external watchers
        # have a concrete artifact to poll.
        ("srtctl.core.status.requests.put", _fake_put),
        ("srtctl.core.status.requests.post", _fake_post),
    ]

    started: list = []
    try:
        for target, replacement in patch_targets:
            try:
                p = patch(target, replacement)
                p.start()
                started.append(p)
            except (AttributeError, ImportError, ModuleNotFoundError):
                # Symbol isn't imported by that module — nothing to patch
                # there, which is fine. Skip quietly.
                continue

        # Make the status reporter believe it is enabled so .report*() paths
        # actually hit our patched requests.put. from_config() is the entry
        # that sets enabled=True iff endpoints are configured.
        from srtctl.core.status import StatusReporter

        original_from_config = StatusReporter.from_config

        def _enabled_from_config(reporting, job_id, _orig=original_from_config):
            reporter = _orig(reporting, job_id)
            if reporter.enabled:
                return reporter
            # Rebuild with a sentinel endpoint so _put fires; requests.put is
            # already patched so the endpoint URL is irrelevant.
            return StatusReporter(
                job_id=job_id,
                api_endpoints=("http://mock-status.local",),
            )

        p = patch.object(StatusReporter, "from_config", _enabled_from_config)
        p.start()
        started.append(p)

        yield status_sink
    finally:
        for p in reversed(started):
            with contextlib.suppress(Exception):
                p.stop()


# ---------------------------------------------------------------------------
# High-level entrypoint: run the full mock sweep.
# ---------------------------------------------------------------------------


def _write_final_result(output_dir: Path, *, job_id: str, exit_code: int) -> None:
    """Write a stable result.json artifact at the end of the mock run."""
    import hashlib

    h = int(hashlib.sha1(f"result-{job_id}".encode()).hexdigest(), 16)
    score = 0.75 + ((h % 100) - 50) / 1000.0
    loss = max(0.01, 1.0 - score + ((h >> 8) % 50) / 2000.0)
    payload = {
        "job_id": job_id,
        "status": "completed" if exit_code == 0 else "failed",
        "exit_code": exit_code,
        "final_loss": round(loss, 4),
        "score": round(score, 4),
        "param_count": 100_000,
        "generated_at": time.time(),
    }
    (output_dir / "result.json").write_text(json.dumps(payload, indent=2) + "\n")


def _set_slurm_env(job_id: str, nodelist: Iterable[str]) -> dict[str, str | None]:
    """Set SLURM_* env vars for the duration of the mock run.

    Returns a snapshot so callers can restore the prior values.
    """
    keys = ["SLURM_JOB_ID", "SLURM_JOBID", "SLURM_NODELIST", "SLURM_NNODES"]
    prior = {k: os.environ.get(k) for k in keys}
    os.environ["SLURM_JOB_ID"] = job_id
    os.environ["SLURM_JOBID"] = job_id
    nodes_csv = ",".join(nodelist)
    os.environ["SLURM_NODELIST"] = nodes_csv
    os.environ["SLURM_NNODES"] = str(len(nodes_csv.split(",")))
    return prior


def _restore_env(prior: dict[str, str | None]) -> None:
    for k, v in prior.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def run_mock_sweep(
    *,
    config_path: Path,
    output_dir: Path,
    job_id: str,
    options: MockOptions | None = None,
) -> int:
    """Execute `SweepOrchestrator.run()` under the full mock infrastructure.

    Parameters
    ----------
    config_path: Path
        Resolved YAML config. Used via `load_config` exactly like production.
    output_dir: Path
        The per-job output directory. Must match what submit_with_orchestrator
        would have written to so artifacts collocate.
    job_id: str
        SLURM-style job id to pretend is ours.
    options: MockOptions | None
        Tunables; defaults to short waits suitable for tests.

    Returns
    -------
    int
        Exit code from the orchestrator.
    """
    from srtctl.cli.do_sweep import SweepOrchestrator
    from srtctl.core.config import load_config
    from srtctl.core.runtime import RuntimeContext

    opts = options or MockOptions()
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # SweepOrchestrator reads SRTCTL_OUTPUT_DIR to place log_dir; point it at
    # the real output_dir so artifacts land where ibar expects them.
    prior_output = os.environ.get("SRTCTL_OUTPUT_DIR")
    os.environ["SRTCTL_OUTPUT_DIR"] = str(output_dir)
    prior_slurm = _set_slurm_env(job_id, opts.nodelist)

    try:
        with mock_infrastructure(options=opts, output_dir=output_dir) as sink:
            config = load_config(config_path)
            runtime = RuntimeContext.from_config(config, job_id=job_id)
            orchestrator = SweepOrchestrator(config=config, runtime=runtime)

            # Run the orchestrator in a thread so we can insert explicit phase
            # pauses from the outside (the orchestrator itself is synchronous
            # and would otherwise race through every phase in microseconds).
            # We block on orchestrator.run() normally; the phase pauses happen
            # inside the fake srun children (child_duration_s).
            sink.write(
                {
                    "status": "running",
                    "phase": "orchestrator-starting",
                    "ts": time.time(),
                }
            )
            exit_code = orchestrator.run()
            sink.write(
                {
                    "status": "completed" if exit_code == 0 else "failed",
                    "phase": "orchestrator-exited",
                    "exit_code": exit_code,
                    "ts": time.time(),
                }
            )
        _write_final_result(output_dir, job_id=job_id, exit_code=exit_code)
        return exit_code
    finally:
        _restore_env(prior_slurm)
        if prior_output is None:
            os.environ.pop("SRTCTL_OUTPUT_DIR", None)
        else:
            os.environ["SRTCTL_OUTPUT_DIR"] = prior_output
