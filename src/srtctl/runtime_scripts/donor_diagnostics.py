# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run a donor with memory diagnostics in its own cgroup and mount namespace."""

from __future__ import annotations

import json
import re
import signal
import subprocess
import sys
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path

CGROUP_FILES = ("memory.current", "memory.max", "memory.high", "memory.peak", "memory.events", "cpuset.mems.effective")
KERNEL_PATTERN = re.compile(r"mlx5|mkey|umem|allocation failure|oom|out of memory", re.IGNORECASE)


def read_file(path: Path) -> str:
    try:
        return path.read_text().strip()
    except OSError as error:
        return f"unavailable: {error}"


def memory_snapshot(phase: str, proc_root: Path = Path("/proc"), cgroup_root: Path = Path("/sys/fs/cgroup")) -> None:
    """The child inherits this launcher's cgroup and process resource limits."""
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "phase": phase,
        "cgroup": read_file(proc_root / "self/cgroup"),
        "limits": read_file(proc_root / "self/limits"),
        "memory": {name: read_file(cgroup_root / name) for name in CGROUP_FILES},
    }
    print("[donor-memory] " + json.dumps(record), flush=True)


def kernel_snapshot(phase: str) -> None:
    """Read, never clear, the kernel ring buffer; lack of privilege is nonfatal."""
    print(f"[donor-kernel] phase={phase} timestamp={datetime.now(timezone.utc).isoformat()}", flush=True)
    try:
        result = subprocess.run(["dmesg", "-T"], capture_output=True, text=True, timeout=5, check=False)
    except (OSError, subprocess.TimeoutExpired) as error:
        print(f"[donor-kernel] unavailable: {error}", flush=True)
        return
    if result.returncode:
        print(f"[donor-kernel] unavailable: {result.stderr.strip() or result.stdout.strip()}", flush=True)
        return
    matches = [line for line in result.stdout.splitlines() if KERNEL_PATTERN.search(line)]
    for line in matches[-100:]:
        print("[donor-kernel] " + line, flush=True)
    if not matches:
        print("[donor-kernel] no matching messages", flush=True)


def main(command: list[str]) -> int:
    if not command:
        print("usage: donor_diagnostics.py COMMAND [ARGS...]", file=sys.stderr)
        return 2
    memory_snapshot("before_start")
    kernel_snapshot("before_start")
    child: subprocess.Popen | None = None
    pending_signal: int | None = None

    def forward_signal(signum: int, _frame: object) -> None:
        nonlocal pending_signal
        pending_signal = signum
        if child is not None:
            with suppress(ProcessLookupError):
                child.send_signal(signum)

    signal.signal(signal.SIGTERM, forward_signal)
    signal.signal(signal.SIGINT, forward_signal)
    try:
        child = subprocess.Popen(command)
        if pending_signal is not None:
            forward_signal(pending_signal, None)
        while True:
            try:
                code = child.wait(timeout=5)
                break
            except subprocess.TimeoutExpired:
                memory_snapshot("running")
        print(f"[donor-memory] child_exit_code={code}", flush=True)
    except OSError as error:
        print(f"[donor-memory] could not start donor: {error}", file=sys.stderr, flush=True)
        code = 127
    finally:
        memory_snapshot("after_exit")
        kernel_snapshot("after_exit")
    return code if code >= 0 else 128 - code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
