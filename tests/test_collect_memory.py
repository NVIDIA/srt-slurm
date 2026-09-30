# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Memory diagnostics retain units, watched processes, and ancestor limits."""

import json
import runpy
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "configs/collect_memory.py"
COLLECT = runpy.run_path(str(SCRIPT))["collect"]


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_collect_memory_and_limits(tmp_path: Path) -> None:
    proc, sys_root = tmp_path / "proc", tmp_path / "sys"
    write(proc / "meminfo", "MemFree: 100 kB\nMemAvailable: 200 kB\nHugePages_Free: 3\n")
    write(sys_root / "devices/system/node/node1/meminfo", "Node 1 MemFree: 50 kB\n")
    for pid, name, rss in ((10, "other", 1000), (11, "mooncake_donor", 1)):
        write(proc / str(pid) / "status", f"Name:\t{name}\nVmRSS:\t{rss} kB\nMems_allowed_list:\t0-1\n")
        write(proc / str(pid) / "cmdline", name + "\x00")
        write(proc / str(pid) / "cgroup", "0::/job/donor\n")
        write(proc / str(pid) / "limits", "Max locked memory         unlimited            unlimited            bytes\n")
    write(proc / "self/cgroup", "0::/job/monitor\n")
    write(sys_root / "fs/cgroup/job/memory.max", "687194767360\n")
    write(sys_root / "fs/cgroup/job/donor/memory.current", "123456\n")
    write(sys_root / "fs/cgroup/job/donor/memory.events", "oom 1\noom_kill 0\n")
    sample = COLLECT(proc, sys_root, 1)
    assert sample["meminfo"] == {"MemFree_bytes": 102400, "MemAvailable_bytes": 204800, "HugePages_Free": 3}
    assert sample["numa"]["node1"]["MemFree_bytes"] == 51200
    assert [p["pid"] for p in sample["processes"]] == [10, 11]
    assert sample["processes"][1]["mems_allowed_list"] == "0-1"
    assert "unlimited" in sample["processes"][1]["locked_memory_limit"]
    assert sample["cgroups"]["job"]["memory.max"] == "687194767360"
    assert sample["cgroups"]["job/donor"]["memory.events"] == "oom 1\noom_kill 0"


def test_single_sample_with_missing_optional_files(tmp_path: Path) -> None:
    proc = tmp_path / "proc"
    write(proc / "meminfo", "MemFree: 1 kB\n")
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--samples", "1", "--proc-root", str(proc), "--sys-root", str(tmp_path / "sys")],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    sample = json.loads(result.stdout)
    assert sample["meminfo"]["MemFree_bytes"] == 1024
    assert sample["processes"] == []
    assert sample["cgroups"] == {}
