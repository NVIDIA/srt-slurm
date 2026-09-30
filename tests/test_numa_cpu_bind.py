# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check the worker's effective NUMA policy without changing host placement."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from srtctl.backends import TRTLLMProtocol


@pytest.mark.parametrize("memory_bind", [None, False, True, "local"])
@pytest.mark.parametrize("bind_cpu", [False, True])
def test_memory_policy_round_trip(memory_bind, bind_cpu) -> None:
    schema = TRTLLMProtocol.Schema()
    settings = {"numa_cpu_bind": bind_cpu, "numa_memory_bind": memory_bind}
    backend = schema.load(settings)
    assert backend.numa_memory_bind == memory_bind
    assert schema.dump(backend)["numa_memory_bind"] == memory_bind
    assert backend.numa_cpu_bind is bind_cpu


def test_local_memory_leaves_cpu_binding_disabled_by_default() -> None:
    backend = TRTLLMProtocol.Schema().load({"numa_memory_bind": "local"})
    assert backend.numa_cpu_bind is False


@pytest.mark.parametrize(
    ("node", "bind_memory", "expected_memory", "expected_cpus", "expected_exit"),
    [
        ("0", True, "bind:0", "0-3", 0),
        ("1", True, "bind:1", "4-7", 0),
        ("-1", True, None, None, 2),
        ("missing", True, None, None, 2),
        ("invalid", True, None, None, 2),
        ("empty-cpus", True, None, None, 2),
        ("missing-cpus", True, None, None, 2),
        ("denied", True, None, None, 42),
        ("1", False, None, "4-7", 0),
        ("-1", False, None, None, 0),
    ],
)
@pytest.mark.parametrize("bind_cpu", [False, True])
def test_worker_inherits_resolved_numa_policy(
    tmp_path: Path,
    node: str,
    bind_memory: bool,
    expected_memory: str | None,
    expected_cpus: str | None,
    expected_exit: int,
    bind_cpu: bool,
) -> None:
    if not bind_cpu:
        expected_cpus = None
        if node in ("empty-cpus", "missing-cpus"):
            expected_memory, expected_exit = "bind:1", 0
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
    if node == "missing":
        sys.exit(1)
    if node in ("denied", "empty-cpus", "missing-cpus"):
        node = "1"
    if args == ["/sys/bus/pci/devices/0000:ab:00.0/numa_node"]:
        print(node)
    else:
        assert args == [f"/sys/devices/system/node/node{node}/cpulist"]
        if os.environ["TEST_NUMA_NODE"] == "missing-cpus":
            sys.exit(1)
        if os.environ["TEST_NUMA_NODE"] == "empty-cpus":
            sys.exit(0)
        print({"0": "0-3", "1": "4-7"}[node])
elif name == "numactl":
    assert args[0].startswith("--membind=")
    if os.environ["TEST_NUMA_NODE"] == "denied":
        sys.exit(42)
    os.environ["TEST_MEMORY_POLICY"] = "bind:" + args.pop(0).split("=", 1)[1]
    os.execvp(args[0], args)
elif name == "taskset":
    assert os.environ["TEST_BIND_CPU"] == "1", "memory-only binding must not invoke taskset"
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
        "TEST_BIND_CPU": "1" if bind_cpu else "0",
    }
    env.pop("TEST_MEMORY_POLICY", None)
    env.pop("TEST_CPU_MASK", None)
    result = subprocess.run(
        [
            "bash",
            str(script),
            *(["--bind-memory"] if bind_memory else []),
            *(["--no-bind-cpu"] if not bind_cpu else []),
            *worker,
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert result.returncode == expected_exit, result.stderr
    if expected_exit:
        assert result.stdout == ""  # The worker must not run after a binding failure.
        if expected_exit == 2:
            expected_error = "cannot bind CPUs" if node in ("empty-cpus", "missing-cpus") else "cannot bind memory"
            assert expected_error in result.stderr
    else:
        assert json.loads(result.stdout) == [expected_memory, expected_cpus, arguments]
