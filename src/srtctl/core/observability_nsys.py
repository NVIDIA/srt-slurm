# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The automatic observability profiler, independent of benchmark control."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from srtctl.core.nsys_keepalive import keepalive_command

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
    sample_cpu = frontend and settings.frontend_cpu_sampling
    prefix = [
        config.profiling.nsys_binary, "profile", "--force-overwrite=true",
        "--trace=nvtx", "--sample=system-wide" if sample_cpu else "--sample=none",
        "--cpuctxsw=none", "--gpu-metrics-devices=none",
        "--delay", str(settings.delay_secs), "--kill", "none", "--wait", "all",
    ]
    if sample_cpu:
        prefix += ["--sampling-period=26000000", "--samples-per-backtrace=32"]
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
    return keepalive_command(prefix + command), environment
