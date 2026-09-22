# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check the worker's effective NUMA policy without changing host placement."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    ("node", "prefer_memory", "expected_memory", "expected_cpus"),
    [
        ("0", True, "preferred:0", "0-3"),
        ("1", True, "preferred:1", "4-7"),
        ("-1", True, "bind:0,1", None),
        ("1", False, None, "4-7"),
    ],
)
def test_worker_inherits_resolved_numa_policy(
    tmp_path: Path, node: str, prefer_memory: bool, expected_memory: str | None, expected_cpus: str | None
) -> None:
    # Execute each wrapper in order: a later numactl would overwrite the
    # observed policy, as it does in a real launch. No host NUMA calls run.
    mock = (
        f"#!{sys.executable}\n"
        + """
import os
from pathlib import Path
import sys

name, args = Path(sys.argv[0]).name, sys.argv[1:]
if name == "nvidia-smi":
    assert args == ["--query-gpu=pci.bus_id", "--format=csv,noheader", "-i", "3"]
    print("00000000:AB:00.0")
elif name == "cat":
    node = os.environ["TEST_NUMA_NODE"]
    if args == ["/sys/bus/pci/devices/0000:ab:00.0/numa_node"]:
        print(node)
    else:
        assert args == [f"/sys/devices/system/node/node{node}/cpulist"]
        print({"0": "0-3", "1": "4-7"}[node])
elif name == "numactl":
    if args[0].startswith("--preferred="):
        os.environ["TEST_MEMORY_POLICY"] = "preferred:" + args.pop(0).split("=", 1)[1]
    else:
        assert args[:2] == ["-m", "0,1"]
        os.environ["TEST_MEMORY_POLICY"] = "bind:" + args[1]
        args = args[2:]
    os.execvp(args[0], args)
elif name == "taskset":
    assert args[0] == "-c"
    os.environ["TEST_CPU_MASK"] = args[1]
    os.execvp(args[2], args[2:])
else:
    raise AssertionError(name)
"""
    )
    for name in ("nvidia-smi", "cat", "numactl", "taskset"):
        executable = tmp_path / name
        executable.write_text(mock)
        executable.chmod(0o755)

    script = Path(__file__).resolve().parents[1] / "configs" / "numa_cpu_bind.sh"
    arguments = ["value with spaces", "", "literal $value; `command`"]
    worker = [
        sys.executable,
        "-c",
        (
            "import json, os, sys; print(json.dumps(["
            "os.getenv('TEST_MEMORY_POLICY'), os.getenv('TEST_CPU_MASK'), sys.argv[1:]]))"
        ),
        *arguments,
    ]
    env = {
        **os.environ,
        "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}",
        "SLURM_LOCALID": "1",
        "CUDA_VISIBLE_DEVICES": "2,3",
        "TEST_NUMA_NODE": node,
    }
    env.pop("TEST_MEMORY_POLICY", None)
    env.pop("TEST_CPU_MASK", None)
    result = subprocess.run(
        ["bash", str(script), *(["--preferred-memory"] if prefer_memory else []), *worker],
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    assert json.loads(result.stdout) == [expected_memory, expected_cpus, arguments]
