# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sample Linux memory diagnostics to stdout as JSONL; no extra dependencies."""

from __future__ import annotations

import argparse
import json
import re
import signal
import socket
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def read_text(path: Path) -> str | None:
    """Files can disappear as processes exit, or be hidden by permissions."""
    try:
        return path.read_text(errors="replace")
    except OSError:
        return None


def memory_fields(text: str) -> dict[str, int]:
    """Normalize Linux kB memory counters to bytes, retaining count fields."""
    fields = {}
    for line in text.splitlines():
        match = re.search(r"([\w()]+):\s+(\d+)(?:\s+(kB))?\s*$", line)
        if match:
            key, value, unit = match.groups()
            fields[key + ("_bytes" if unit else "")] = int(value) * (1024 if unit else 1)
    return fields


def process_memory(path: Path) -> dict[str, Any] | None:
    status = read_text(path / "status")
    if status is None:
        return None
    values = dict(line.split(":", 1) for line in status.splitlines() if ":" in line)
    command = read_text(path / "cmdline") or ""
    limits = read_text(path / "limits") or ""
    memory = memory_fields("\n".join(line for line in status.splitlines() if line.startswith("Vm")))
    return {
        "pid": int(path.name),
        "name": values.get("Name", "").strip(),
        "memory": memory,
        "mems_allowed_list": values.get("Mems_allowed_list", "").strip(),
        "locked_memory_limit": next(
            (s.strip() for s in limits.splitlines() if s.startswith("Max locked memory")), None
        ),
        "cgroup": (read_text(path / "cgroup") or "").splitlines(),
        "watched": any(name in command.lower() for name in ("mooncake", "trtllm", "tensorrt_llm", "dynamo")),
    }


def cgroup_memory(sys_root: Path, processes: list[dict[str, Any]], own_cgroup: str) -> dict[str, Any]:
    """Include ancestor limits: a parent cgroup can constrain a child."""
    root = sys_root / "fs/cgroup"
    paths = {root}
    entries = own_cgroup.splitlines() + [entry for process in processes for entry in process["cgroup"]]
    for entry in entries:
        parts = entry.split(":", 2)
        if len(parts) != 3 or ".." in Path(parts[2]).parts:
            continue
        if parts[1] == "":
            base = root  # cgroup v2
        elif "memory" in parts[1].split(","):
            base = root / "memory"  # conventional cgroup v1 mount
        else:
            continue
        path = base / parts[2].lstrip("/")
        while True:
            paths.add(path)
            if path == base:
                break
            path = path.parent
    files = (
        "memory.current",
        "memory.max",
        "memory.high",
        "memory.peak",
        "memory.events",
        "memory.stat",
        "cpuset.mems.effective",
        "memory.usage_in_bytes",
        "memory.limit_in_bytes",
        "memory.max_usage_in_bytes",
        "memory.failcnt",
    )
    result = {}
    for path in sorted(paths):
        counters = {name: value.strip() for name in files if (value := read_text(path / name)) is not None}
        if counters:
            result[str(path.relative_to(root))] = counters
    return result


def collect(proc_root: Path, sys_root: Path, top: int) -> dict[str, Any]:
    processes = []
    for path in proc_root.iterdir():
        if path.name.isdecimal() and (process := process_memory(path)) is not None:
            processes.append(process)
    processes.sort(key=lambda p: p["memory"].get("VmRSS_bytes", 0), reverse=True)
    selected = [p for index, p in enumerate(processes) if index < top or p["watched"]]
    numa = {}
    for path in sorted((sys_root / "devices/system/node").glob("node[0-9]*/meminfo")):
        if (content := read_text(path)) is not None:
            numa[path.parent.name] = memory_fields(content)
    own_cgroup = read_text(proc_root / "self/cgroup") or ""
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "meminfo": memory_fields(read_text(proc_root / "meminfo") or ""),
        "numa": numa,
        "process_count": len(processes),
        "processes": selected,
        "cgroups": cgroup_memory(sys_root, selected, own_cgroup),
        "collector_cgroup": own_cgroup.splitlines(),
        "collector_locked_memory_limit": next(
            (
                s.strip()
                for s in (read_text(proc_root / "self/limits") or "").splitlines()
                if s.startswith("Max locked memory")
            ),
            None,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval", type=float, default=5, help="Seconds between samples (default: 5)")
    parser.add_argument(
        "--top", type=int, default=20, help="Top RSS processes, plus all Mooncake/TRT-LLM/Dynamo processes"
    )
    parser.add_argument("--samples", type=int, default=0, help="Stop after this many samples; 0 runs until terminated")
    parser.add_argument("--proc-root", type=Path, default=Path("/proc"))
    parser.add_argument("--sys-root", type=Path, default=Path("/sys"))
    args = parser.parse_args()
    if args.interval <= 0 or args.top < 0 or args.samples < 0:
        parser.error("interval must be positive; top and samples must be nonnegative")
    stopped = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stopped.set())
    signal.signal(signal.SIGINT, lambda *_: stopped.set())
    count = 0
    while not stopped.is_set():
        print(json.dumps(collect(args.proc_root, args.sys_root, args.top)), flush=True)
        count += 1
        if args.samples and count >= args.samples:
            break
        stopped.wait(args.interval)


if __name__ == "__main__":
    main()
