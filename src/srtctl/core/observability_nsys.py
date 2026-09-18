# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Automatic observability profiling during measured work or from process startup."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from srtctl.core.nsys_keepalive import keepalive_command
from srtctl.runtime_scripts.nsys_window import write_json

if TYPE_CHECKING:
    from srtctl.core.schema import SrtConfig


def wrap_observability_nsys(
    command: list[str],
    *,
    config: SrtConfig,
    log_dir: Path,
    report_name: str,
    ranks: int = 1,
    frontend: bool = False,
) -> tuple[list[str], dict[str, str]]:
    """Profile a launch; all MPI tasks share a fresh report-finalization barrier.

    ``report_name`` is relative to ``profiles/`` and includes the Slurm rank
    substitution for MPI launches. No PROFILE_TYPE or traffic duration is set:
    custom, time-limited and manual benchmarks retain their own lifecycle.
    """
    settings = config.observability.nsys
    (log_dir / "profiles" / report_name).parent.mkdir(parents=True, exist_ok=True)
    capture_args = [
        "--sample=process-tree",
        "--cpuctxsw=process-tree",
        "--gpu-metrics-devices=none",
        "--sampling-period=26000000",
        "--samples-per-backtrace=32",
    ]
    prefix = [
        config.profiling.nsys_binary,
        "profile",
        "--force-overwrite=true",
        "--trace=nvtx",
        *capture_args,
        "--kill",
        "none",
        "--wait",
        "all",
    ]
    prefix += ["-o", f"/logs/profiles/{report_name}"]
    environment = {
        "DYN_ENABLE_RUST_NVTX": "1",
        "SRT_NSYS_REPORT_BARRIER_DIR": f"/logs/profiles/.stopped/{uuid.uuid4().hex}",
        "SRT_NSYS_REPORT_EXPECTED": str(ranks),
        "SRT_NSYS_REPORT_STOP_TIMEOUT": str(settings.report_timeout_secs),
    }
    if not frontend and config.backend_type == "trtllm":
        environment.update(TLLM_LLMAPI_ENABLE_NVTX="1", TLLM_PROFILE_LOG_RANKS="all")
    if settings.nvtx_injection_path:
        environment["NVTX_INJECTION64_PATH"] = settings.nvtx_injection_path
    if settings.capture_window == "measured_workload":
        step = uuid.uuid4().hex
        write_json(log_dir / "profiles" / ".control" / "steps" / f"{step}.json", {"ranks": ranks})
        spec = {
            "control_dir": "/logs/profiles/.control",
            "step": step,
            "ranks": ranks,
            "nsys": config.profiling.nsys_binary,
            "start_args": capture_args,
            "output": f"/logs/profiles/{report_name}",
            "timeout": settings.report_timeout_secs,
        }
        return [
            "python3",
            "/srtctl-runtime/nsys_window.py",
            "worker",
            "--spec",
            json.dumps(spec),
            "--",
            *command,
        ], environment
    return keepalive_command(prefix + command), environment


def benchmark_nsys_env(config: SrtConfig) -> dict[str, str]:
    """Make the same boundary API available to bundled and custom clients."""
    if not config.observability_nsys_enabled or config.observability.nsys.capture_window != "measured_workload":
        return {}
    return {
        "SRT_NSYS_CONTROL_DIR": "/logs/profiles/.control",
        "SRT_NSYS_CONTROL_SCRIPT": "/srtctl-runtime/nsys_window.py",
        "SRT_NSYS_CONTROL_TIMEOUT": str(config.observability.nsys.report_timeout_secs),
    }
