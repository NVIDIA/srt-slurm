# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check the worker's effective NUMA policy without changing host placement."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from srtctl.backends import TRTLLMBackend


@pytest.mark.parametrize("memory_bind", [None, False, True, "local", "preferred"])
def test_memory_policy_round_trip(memory_bind) -> None:
    schema = TRTLLMBackend.Schema()
    settings = {"numa_cpu_bind": True, "numa_memory_bind": memory_bind}
    backend = schema.load(settings)
    assert backend.numa_memory_bind == memory_bind
    assert schema.dump(backend)["numa_memory_bind"] == memory_bind


@pytest.mark.parametrize("memory_bind", ["local", "preferred"])
def test_wrapper_memory_policy_requires_cpu_binding(memory_bind) -> None:
    with pytest.raises(ValueError, match=f"numa_memory_bind: {memory_bind} requires numa_cpu_bind: true"):
        TRTLLMBackend.Schema().load({"numa_memory_bind": memory_bind})


@pytest.fixture
def wrapper_env(tmp_path: Path) -> dict[str, str]:
    # Mock numactl executes the real worker subprocess with its selected policy.
    # A taskset invocation fails, including one hidden behind numactl.
    mock = (
        f"#!{sys.executable}\n"
        + """
import json
import os
from pathlib import Path
import sys

name, args = Path(sys.argv[0]).name, sys.argv[1:]
if name == "nvidia-smi":
    assert args == ["--query-gpu=pci.bus_id", "--format=csv,noheader", "-i", os.environ["TEST_GPU"]]
    if code := int(os.environ.get("TEST_NVIDIA_SMI_EXIT", "0")):
        sys.exit(code)
    print("00000000:AB:00.0")
elif name == "cat":
    node = os.environ["TEST_NUMA_NODE"]
    if args == ["/sys/bus/pci/devices/0000:ab:00.0/numa_node"]:
        if node == "missing":
            sys.exit(1)
        print(node)
    else:
        assert args == [f"/sys/devices/system/node/node{node}/cpulist"]
        if os.environ.get("TEST_CPULIST_MISSING") == "1":
            sys.exit(1)
        print(os.environ.get("TEST_NODE_CPUS", {"0": "0-3", "1": "4-7"}[node]))
elif name == "numactl":
    with open(os.environ["TEST_NUMACTL_TRACE"], "a") as trace:
        trace.write(json.dumps(args) + "\\n")
    if code := int(os.environ.get("TEST_NUMACTL_EXIT", "0")):
        sys.exit(code)
    node = os.environ["TEST_NUMA_NODE"]
    assert args.pop(0) == f"--cpunodebind={node}"
    if args[0] == f"--membind={node}":
        os.environ["TEST_MEMORY_POLICY"] = "bind:" + node
        args.pop(0)
    elif args[0] == f"--preferred={node}":
        os.environ["TEST_MEMORY_POLICY"] = "preferred:" + node
        args.pop(0)
    os.environ["TEST_CPU_MASK"] = json.dumps({"0": [0, 1, 2, 3], "1": [4, 5, 6, 7]}[node])
    os.execvp(args[0], args)
elif name == "taskset":
    raise AssertionError("taskset must not be invoked")
else:
    raise AssertionError(name)
"""
    )
    for name in ("nvidia-smi", "cat", "numactl", "taskset"):
        executable = tmp_path / name
        executable.write_text(mock)
        executable.chmod(0o755)

    worker = tmp_path / "worker.py"
    worker.write_text(
        "import json, os, sys\n"
        "print(json.dumps([os.getenv('TEST_MEMORY_POLICY'), json.loads(os.environ['TEST_CPU_MASK']), sys.argv[1:]]))\n"
        "sys.exit(int(os.environ.get('TEST_WORKER_EXIT', '0')))\n"
    )
    return {
        **os.environ,
        "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}",
        "SLURM_LOCALID": "1",
        "CUDA_VISIBLE_DEVICES": "2,3",
        "TEST_GPU": "3",
        "TEST_NUMA_NODE": "1",
        "TEST_MEMORY_POLICY": "interleave:0,1",
        "TEST_CPU_MASK": "[0, 1, 2, 3, 4, 5, 6, 7]",
        "TEST_NUMACTL_TRACE": str(tmp_path / "numactl.jsonl"),
        "TEST_WORKER": str(worker),
    }


def _run_wrapper(
    env: dict[str, str], memory_flag: str | None = None, arguments: tuple[str, ...] = ()
) -> subprocess.CompletedProcess[str]:
    script = Path(__file__).resolve().parents[1] / "configs" / "numa_cpu_bind.sh"
    worker = [sys.executable, env["TEST_WORKER"], *arguments]
    return subprocess.run(
        ["bash", str(script), *([memory_flag] if memory_flag else []), *worker],
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )


@pytest.mark.parametrize("node", ["0", "1"])
@pytest.mark.parametrize(
    ("memory_flag", "numactl_memory_flag", "memory_policy"),
    [
        (None, None, "interleave:0,1"),
        ("--bind-memory", "--membind", "bind"),
        ("--prefer-memory", "--preferred", "preferred"),
    ],
)
def test_worker_inherits_resolved_numa_policy(
    wrapper_env: dict[str, str],
    node: str,
    memory_flag: str | None,
    numactl_memory_flag: str | None,
    memory_policy: str,
) -> None:
    wrapper_env["TEST_NUMA_NODE"] = node
    arguments = ("value with spaces", "", "literal $value; `command`", "--option")
    result = _run_wrapper(wrapper_env, memory_flag, arguments)
    assert result.returncode == 0, result.stderr
    expected_memory = f"{memory_policy}:{node}" if memory_flag else memory_policy
    cpus = list(range(0, 4)) if node == "0" else list(range(4, 8))
    assert json.loads(result.stdout) == [expected_memory, cpus, list(arguments)]
    expected_flags = [f"--cpunodebind={node}"]
    if numactl_memory_flag:
        expected_flags.append(f"{numactl_memory_flag}={node}")
    calls = Path(wrapper_env["TEST_NUMACTL_TRACE"]).read_text().splitlines()
    assert [json.loads(call) for call in calls] == [
        [*expected_flags, sys.executable, wrapper_env["TEST_WORKER"], *arguments]
    ]


@pytest.mark.parametrize("memory_flag", [None, "--bind-memory", "--prefer-memory"])
@pytest.mark.parametrize("node", ["-1", "missing", "invalid"])
def test_unknown_node_falls_back_only_without_memory_policy(
    wrapper_env: dict[str, str], memory_flag: str | None, node: str
) -> None:
    wrapper_env["TEST_NUMA_NODE"] = node
    result = _run_wrapper(wrapper_env, memory_flag)
    assert not Path(wrapper_env["TEST_NUMACTL_TRACE"]).exists()
    if memory_flag:
        assert result.returncode == 2
        assert "cannot apply memory policy" in result.stderr
        assert result.stdout == ""
    else:
        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout) == ["interleave:0,1", list(range(8)), []]


@pytest.mark.parametrize("memory_flag", [None, "--bind-memory", "--prefer-memory"])
@pytest.mark.parametrize("cpu_env", [{"TEST_NODE_CPUS": ""}, {"TEST_CPULIST_MISSING": "1"}])
def test_unusable_node_cpu_list_stops_worker(
    wrapper_env: dict[str, str], memory_flag: str | None, cpu_env: dict[str, str]
) -> None:
    wrapper_env.update(cpu_env)
    result = _run_wrapper(wrapper_env, memory_flag)
    assert result.returncode == 2
    assert "cannot bind CPUs" in result.stderr
    assert result.stdout == ""
    assert not Path(wrapper_env["TEST_NUMACTL_TRACE"]).exists()


@pytest.mark.parametrize("memory_flag", [None, "--bind-memory", "--prefer-memory"])
def test_numactl_failure_stops_worker(wrapper_env: dict[str, str], memory_flag: str | None) -> None:
    wrapper_env["TEST_NUMACTL_EXIT"] = "42"
    result = _run_wrapper(wrapper_env, memory_flag)
    assert result.returncode == 42
    assert result.stdout == ""


@pytest.mark.parametrize("memory_flag", [None, "--bind-memory", "--prefer-memory"])
def test_worker_failure_propagates(wrapper_env: dict[str, str], memory_flag: str | None) -> None:
    wrapper_env["TEST_WORKER_EXIT"] = "37"
    result = _run_wrapper(wrapper_env, memory_flag)
    assert result.returncode == 37
    assert json.loads(result.stdout)[1] == [4, 5, 6, 7]


@pytest.mark.parametrize("visible_gpus", [None, "", "2,3", "GPU-first,GPU-second"])
def test_gpu_resolution_uses_visible_devices_and_local_rank(
    wrapper_env: dict[str, str], visible_gpus: str | None
) -> None:
    if visible_gpus is None:
        wrapper_env.pop("CUDA_VISIBLE_DEVICES")
    else:
        wrapper_env["CUDA_VISIBLE_DEVICES"] = visible_gpus
    wrapper_env["TEST_GPU"] = visible_gpus.split(",")[1] if visible_gpus else "1"
    result = _run_wrapper(wrapper_env)
    assert result.returncode == 0, result.stderr


def test_unresolved_gpu_stops_worker(wrapper_env: dict[str, str]) -> None:
    wrapper_env["CUDA_VISIBLE_DEVICES"] = "2"
    result = _run_wrapper(wrapper_env)
    assert result.returncode == 2
    assert "no GPU resolved" in result.stderr
    assert result.stdout == ""


def test_gpu_query_failure_propagates(wrapper_env: dict[str, str]) -> None:
    wrapper_env["TEST_NVIDIA_SMI_EXIT"] = "19"
    result = _run_wrapper(wrapper_env)
    assert result.returncode == 19
    assert result.stdout == ""
