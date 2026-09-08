# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Job lifecycle tools for the srtctl MCP server: submit, watch, read logs, cancel.

Unlike the schema tools in :mod:`srtctl.mcp.spec_tools`, these run where Slurm
is: the MCP server has to be started on a login node of the target cluster, in
a checkout with its ``srtslurm.yaml``. Each tool is a thin layer over the same
commands an operator would type (``srtctl apply --json``, ``sacct``, ``squeue``,
``scancel``) plus the job's output directory, so what an agent sees is what the
CLI and the log directory say.

Output directories follow ``srtctl apply``: an explicit ``output_dir`` argument,
else ``output_dir`` from ``srtslurm.yaml``, else ``<srtctl_root>/outputs``.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from srtctl.core.config import get_srtslurm_setting

SLURM_TIMEOUT_SECONDS = 30
SUBMIT_TIMEOUT_SECONDS = 600
SACCT_FIELDS = ("JobID", "JobName", "State", "Elapsed", "ExitCode", "NodeList", "Start", "End")
SQUEUE_FORMAT = "%i|%j|%T|%M|%N|%r"


def _srtctl_command() -> list[str]:
    """The srtctl CLI of this interpreter's environment (not whatever is first on PATH)."""
    sibling = Path(sys.executable).with_name("srtctl")
    if sibling.exists():
        return [str(sibling)]
    found = shutil.which("srtctl")
    if found:
        return [found]
    return [sys.executable, "-m", "srtctl.cli.submit"]


def _run(command: list[str], *, timeout: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)


def outputs_dir(output_dir: str | None = None) -> Path:
    """Where ``srtctl apply`` writes ``<job_id>/`` directories, resolved the way the CLI does."""
    if output_dir:
        return Path(os.path.expandvars(output_dir)).expanduser().resolve()
    configured = get_srtslurm_setting("output_dir")
    if configured:
        return Path(os.path.expandvars(str(configured))).expanduser().resolve()
    root = get_srtslurm_setting("srtctl_root")
    base = Path(str(root)) if root else Path(__file__).resolve().parents[3]
    return (base / "outputs").resolve()


def _job_dir(job_id: str, output_dir: str | None) -> Path:
    if not str(job_id).isdigit():
        raise ValueError(f"job_id must be a Slurm job id, got {job_id!r}")
    return outputs_dir(output_dir) / str(job_id)


def submit_job(
    config_path: str,
    *,
    set_overrides: list[str] | None = None,
    unset: list[str] | None = None,
    tags: list[str] | None = None,
    serve_only: bool = False,
    output_dir: str | None = None,
) -> dict[str, Any]:
    """Submit a recipe with ``srtctl apply -y --json`` and return the submission record.

    ``set_overrides`` are ``KEY=VALUE`` entries (the CLI's ``--set``), ``unset``
    the keys to drop (``--unset``). The record carries ``slurm_job_id``,
    ``output_dir``, ``metadata_path``, and ``applied_overrides``.
    """
    path = Path(config_path).expanduser()
    if not path.exists():
        return {"ok": False, "error": f"recipe not found: {path}"}
    command = [*_srtctl_command(), "apply", "-f", str(path), "-y", "--json"]
    for item in set_overrides or []:
        command += ["--set", item]
    for key in unset or []:
        command += ["--unset", key]
    if tags:
        command += ["--tags", ",".join(tags)]
    if serve_only:
        command.append("--serve-only")
    if output_dir:
        command += ["-o", output_dir]
    try:
        result = _run(command, timeout=SUBMIT_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "error": f"srtctl apply did not return within {SUBMIT_TIMEOUT_SECONDS}s",
            "command": command,
        }
    records = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if result.returncode != 0 or not records:
        return {
            "ok": False,
            "error": "srtctl apply failed" if result.returncode != 0 else "srtctl apply printed no submission record",
            "exit_code": result.returncode,
            "stdout": result.stdout[-4000:],
            "stderr": result.stderr[-4000:],
            "command": command,
        }
    return {"ok": True, "submissions": records, "command": command}


def dry_run(
    config_path: str,
    *,
    set_overrides: list[str] | None = None,
    unset: list[str] | None = None,
) -> dict[str, Any]:
    """Render a recipe with ``srtctl dry-run`` (sbatch script, services, mounts, env) without submitting."""
    path = Path(config_path).expanduser()
    if not path.exists():
        return {"ok": False, "error": f"recipe not found: {path}"}
    command = [*_srtctl_command(), "dry-run", "-f", str(path)]
    for item in set_overrides or []:
        command += ["--set", item]
    for key in unset or []:
        command += ["--unset", key]
    try:
        result = _run(command, timeout=SUBMIT_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"srtctl dry-run did not return within {SUBMIT_TIMEOUT_SECONDS}s"}
    return {
        "ok": result.returncode == 0,
        "exit_code": result.returncode,
        "output": result.stdout[-20000:],
        "stderr": result.stderr[-4000:],
        "command": command,
    }


def _sacct(job_id: str) -> dict[str, Any] | None:
    if shutil.which("sacct") is None:
        return None
    result = _run(
        ["sacct", "-j", job_id, "-X", "-n", "-P", f"--format={','.join(SACCT_FIELDS)}"],
        timeout=SLURM_TIMEOUT_SECONDS,
    )
    if result.returncode != 0:
        return {"error": result.stderr.strip() or f"sacct exited {result.returncode}"}
    for line in result.stdout.splitlines():
        parts = line.split("|")
        if len(parts) >= len(SACCT_FIELDS) and parts[0] == job_id:
            return dict(zip((f.lower() for f in SACCT_FIELDS), parts[: len(SACCT_FIELDS)], strict=True))
    return None


def _tail(path: Path, lines: int) -> list[str]:
    try:
        content = path.read_text(errors="replace").splitlines()
    except OSError:
        return []
    return content[-lines:] if lines > 0 else content


def job_status(job_id: str, *, output_dir: str | None = None, tail: int = 20) -> dict[str, Any]:
    """Slurm accounting for a job plus what its output directory says: metadata, current stage, log tail.

    ``stage`` is the last ``[INFO]`` line of the sweep log, which names the phase
    the orchestrator is in (infra services, workers, frontend, benchmark, cleanup).
    """
    job_id = str(job_id)
    job_dir = _job_dir(job_id, output_dir)
    status: dict[str, Any] = {"job_id": job_id, "output_dir": str(job_dir), "slurm": _sacct(job_id)}
    metadata_path = job_dir / f"{job_id}.json"
    if metadata_path.exists():
        try:
            status["metadata"] = json.loads(metadata_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            status["metadata_error"] = str(exc)
    sweep_log = job_dir / "logs" / f"sweep_{job_id}.log"
    if sweep_log.exists():
        lines = _tail(sweep_log, 0)
        status["sweep_log"] = str(sweep_log)
        status["sweep_log_tail"] = lines[-tail:]
        info = [line for line in lines if "[INFO]" in line]
        status["stage"] = info[-1] if info else None
        status["errors"] = [line for line in lines if "[ERROR]" in line][-10:]
    rollup = job_dir / "logs" / "benchmark-rollup.json"
    if rollup.exists():
        try:
            status["benchmark_rollup"] = json.loads(rollup.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            status["benchmark_rollup_error"] = str(exc)
    dashboard = job_dir / "logs" / "perf_dashboard.html"
    if dashboard.exists():
        status["perf_dashboard"] = str(dashboard)
    return status


def job_logs(job_id: str, *, name: str | None = None, tail: int = 200, output_dir: str | None = None) -> dict[str, Any]:
    """List a job's log files, or return the tail of one of them.

    Names are relative to ``<output_dir>/<job_id>/logs``: ``sweep_<id>.log`` (the
    orchestrator), ``<node>_<mode>_w<i>.out`` (workers), ``<node>_frontend_<i>.out``,
    ``service_<name>.out`` (etcd, nats, exporters, declared services),
    ``tachometer.out``, ``benchmark.out``.
    """
    job_id = str(job_id)
    logs_dir = _job_dir(job_id, output_dir) / "logs"
    if not logs_dir.is_dir():
        return {"ok": False, "error": f"no log directory at {logs_dir}"}
    if name is None:
        files = sorted(p for p in logs_dir.iterdir() if p.is_file())
        return {
            "ok": True,
            "logs_dir": str(logs_dir),
            "files": [{"name": p.name, "bytes": p.stat().st_size} for p in files],
        }
    target = (logs_dir / name).resolve()
    if logs_dir.resolve() not in target.parents:
        return {"ok": False, "error": f"{name!r} is outside the job's log directory"}
    if not target.is_file():
        return {"ok": False, "error": f"no such log: {target}"}
    return {"ok": True, "path": str(target), "tail": _tail(target, tail)}


def cancel_job(job_id: str) -> dict[str, Any]:
    """``scancel`` a job; srtctl's orchestrator handles the SIGTERM and stops every step it launched."""
    job_id = str(job_id)
    if not job_id.isdigit():
        return {"ok": False, "error": f"job_id must be a Slurm job id, got {job_id!r}"}
    if shutil.which("scancel") is None:
        return {"ok": False, "error": "scancel is not on PATH; run the MCP server on a Slurm login node"}
    result = _run(["scancel", job_id], timeout=SLURM_TIMEOUT_SECONDS)
    return {"ok": result.returncode == 0, "job_id": job_id, "stderr": result.stderr.strip()}


def list_jobs(user: str | None = None) -> dict[str, Any]:
    """Pending and running Slurm jobs of ``user`` (default: the current user) with state, elapsed, nodes."""
    if shutil.which("squeue") is None:
        return {"ok": False, "error": "squeue is not on PATH; run the MCP server on a Slurm login node"}
    command = ["squeue", "-h", "-o", SQUEUE_FORMAT]
    command += ["-u", user] if user else ["--me"]
    result = _run(command, timeout=SLURM_TIMEOUT_SECONDS)
    if result.returncode != 0:
        return {"ok": False, "error": result.stderr.strip() or f"squeue exited {result.returncode}"}
    jobs = []
    for line in result.stdout.splitlines():
        parts = line.split("|")
        if len(parts) >= 6:
            jobs.append(
                {
                    "job_id": parts[0],
                    "name": parts[1],
                    "state": parts[2],
                    "elapsed": parts[3],
                    "nodes": parts[4],
                    "reason": parts[5],
                }
            )
    return {"ok": True, "jobs": jobs}
