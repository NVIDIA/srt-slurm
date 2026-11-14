# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM workers bound to their own GPUs' NUMA node(s) (``numa_cpu_bind`` / ``numa_memory_bind``).

The commands come from ``VLLMBackend``; the binding is what configs/numa_cpu_bind.sh
resolves for them on a mocked two-socket, four-GPU node (GPUs 0-1 on NUMA node 0,
GPUs 2-3 on node 1), so no Slurm, GPU or numactl is needed.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import yaml
from marshmallow import ValidationError

from srtctl.backends import VLLMBackend
from srtctl.core.schema import DynamoConfig, RoleConfig, SrtConfig
from srtctl.core.topology import Endpoint, Process

REPO = Path(__file__).resolve().parents[1]
WRAPPER = REPO / "configs" / "numa_cpu_bind.sh"
EXAMPLE = REPO / "examples" / "vllm" / "dynamo-disagg-numa.yaml"

# Two sockets, two GPUs each, as on a GB200/VR200 compute tray.
GPU_NODES = {"0": "0", "1": "0", "2": "1", "3": "1"}
NODE_CPUS = {"0": "0-3", "1": "4-7"}


@pytest.fixture
def two_socket_node(tmp_path: Path) -> dict[str, str]:
    """PATH mocks of nvidia-smi, cat and numactl for a two-socket, four-GPU node.

    numactl records its options (everything before the worker command) and exits
    without running the worker.
    """
    mock = (
        f"#!{sys.executable}\n"
        + """
import json
import os
from pathlib import Path
import sys

gpu_nodes = json.loads(os.environ["TEST_GPU_NODES"])
name, args = Path(sys.argv[0]).name, sys.argv[1:]
if name == "nvidia-smi":
    assert args[:3] == ["--query-gpu=pci.bus_id", "--format=csv,noheader", "-i"], args
    print(f"00000000:{0x10 + int(args[3]):02X}:00.0")
elif name == "cat":
    (path,) = args
    if path.startswith("/sys/bus/pci/devices/"):
        gpu = str(int(path.split("/")[5].split(":")[1], 16) - 0x10)
        print(gpu_nodes[gpu])
    else:
        node = path.removeprefix("/sys/devices/system/node/node").removesuffix("/cpulist")
        print(json.loads(os.environ["TEST_NODE_CPUS"])[node])
elif name == "numactl":
    options = []
    while args and args[0].startswith("--"):
        options.append(args.pop(0))
    with open(os.environ["TEST_NUMACTL_TRACE"], "a") as trace:
        trace.write(json.dumps({"options": options, "command": args}) + "\\n")
else:
    raise AssertionError(name)
"""
    )
    for name in ("nvidia-smi", "cat", "numactl"):
        executable = tmp_path / name
        executable.write_text(mock)
        executable.chmod(0o755)
    env = {key: value for key, value in os.environ.items() if key not in ("SLURM_LOCALID", "CUDA_VISIBLE_DEVICES")}
    return {
        **env,
        "PATH": f"{tmp_path}{os.pathsep}{env['PATH']}",
        "TEST_GPU_NODES": json.dumps(GPU_NODES),
        "TEST_NODE_CPUS": json.dumps(NODE_CPUS),
        "TEST_NUMACTL_TRACE": str(tmp_path / "numactl.jsonl"),
    }


def _launch(env: dict[str, str], command: list[str]) -> subprocess.CompletedProcess[str]:
    """Run a worker command's wrapper with the repository script in place of /configs."""
    assert command[:2] == ["bash", "/configs/numa_cpu_bind.sh"], command
    return subprocess.run(
        ["bash", str(WRAPPER), *command[2:]], env=env, capture_output=True, text=True, check=False, timeout=10
    )


def _numactl_options(env: dict[str, str], command: list[str]) -> list[str]:
    """The numactl options the wrapper starts ``command`` with on the mocked node."""
    trace = Path(env["TEST_NUMACTL_TRACE"])
    trace.unlink(missing_ok=True)
    result = _launch(env, command)
    assert result.returncode == 0, result.stderr
    (call,) = [json.loads(line) for line in trace.read_text().splitlines()]
    wrapped = command[command.index("--gpus") + 2 :]
    wrapped = [arg for arg in wrapped if arg not in ("--bind-memory", "--prefer-memory")]
    assert call["command"] == wrapped
    return call["options"]


def _runtime(sidecar: bool = False) -> MagicMock:
    runtime = MagicMock(
        model_path=Path("/model"), is_hf_model=False, worker_model_arg="/model", frontend_port=9000, request_plane="tcp"
    )
    runtime.dynamo = DynamoConfig(sidecar=sidecar)
    return runtime


def _commands(
    backend: VLLMBackend, endpoints: list[Endpoint], frontend: str = "dynamo", sidecar: bool = False
) -> list[tuple[Process, list[str]]]:
    """Every process of ``endpoints`` with its worker command."""
    launched = []
    with patch("srtctl.core.slurm.get_hostname_ip", return_value="127.0.0.1"):
        for endpoint in endpoints:
            processes = backend.endpoints_to_processes([endpoint], frontend_type=frontend, dynamo_sidecar=sidecar)
            for process in processes:
                command = backend.build_worker_command(process, processes, _runtime(sidecar), frontend_type=frontend)
                launched.append((process, command))
    return launched


def _backend(role: dict[str, Any], args: dict[str, Any], mode: str = "prefill", **engine: Any) -> VLLMBackend:
    return VLLMBackend(**engine, roles={mode: RoleConfig(**role, args=args)})


# Two workers packed on one node: GPUs 0-1 (socket 0) and GPUs 2-3 (socket 1).
PACKED = [
    Endpoint("prefill", 0, ("node0",), frozenset({0, 1})),
    Endpoint("prefill", 1, ("node0",), frozenset({2, 3})),
]


@pytest.mark.parametrize("where", ["engine", "role"])
@pytest.mark.parametrize(
    "args",
    [
        {"tensor-parallel-size": 2},
        # One DEP2 worker per socket, launched as one per-node process each.
        {"data-parallel-size": 2, "enable-expert-parallel": True},
    ],
    ids=["tp2", "dep2"],
)
@pytest.mark.parametrize("set_visible_devices", [False, True])
@pytest.mark.parametrize(
    ("memory_bind", "memory_flag", "socket_options"),
    [
        (None, [], lambda node: [f"--cpunodebind={node}"]),
        ("local", ["--bind-memory"], lambda node: [f"--cpunodebind={node}", f"--membind={node}"]),
        ("preferred", ["--prefer-memory"], lambda node: [f"--cpunodebind={node}", f"--preferred={node}"]),
    ],
)
def test_packed_workers_bind_to_their_own_socket(
    two_socket_node, where, args, set_visible_devices, memory_bind, memory_flag, socket_options
) -> None:
    """One shared role config binds the worker on GPUs 0-1 to socket 0 and the one on GPUs 2-3 to socket 1."""
    settings = {"numa_cpu_bind": True, "numa_memory_bind": memory_bind}
    if where == "engine":
        backend = _backend({}, args, set_visible_devices=set_visible_devices, **settings)
    else:
        backend = _backend(settings, args, set_visible_devices=set_visible_devices)

    launched = _commands(backend, PACKED)

    assert [sorted(process.gpu_indices) for process, _ in launched] == [[0, 1], [2, 3]]
    for (_, command), gpus, node in zip(launched, ("0,1", "2,3"), ("0", "1"), strict=True):
        assert command[: 5 + len(memory_flag)] == [
            "bash",
            "/configs/numa_cpu_bind.sh",
            "--gpus",
            gpus,
            *memory_flag,
            "python3",
        ]
        # Physical GPUs reach the wrapper whether the engine gets a mask or --device-ids.
        assert ("--device-ids" in command) is not set_visible_devices
        assert _numactl_options(two_socket_node, command) == socket_options(node)


@pytest.mark.parametrize(
    ("memory_bind", "options"),
    [
        (None, ["--cpunodebind=0,1"]),
        ("local", ["--cpunodebind=0,1", "--membind=0,1"]),
        # The kernel's default local allocation already prefers the CPU's own node.
        ("preferred", ["--cpunodebind=0,1"]),
    ],
)
def test_tp8_worker_across_two_nodes_binds_each_node_to_its_gpus_nodes(two_socket_node, memory_bind, options) -> None:
    """A TP8 worker on two four-GPU nodes binds each node's process to the NUMA nodes of its four GPUs."""
    backend = _backend(
        {"numa_cpu_bind": True, "numa_memory_bind": memory_bind}, {"tensor-parallel-size": 8}, mode="decode"
    )
    endpoint = Endpoint("decode", 0, ("node0", "node1"), frozenset(range(4)))

    launched = _commands(backend, [endpoint])

    assert [process.node for process, _ in launched] == ["node0", "node1"]
    for process, command in launched:
        assert command[2:4] == ["--gpus", "0,1,2,3"]
        assert ("--headless" in command) is (process.node == "node1")
        assert _numactl_options(two_socket_node, command) == options


@pytest.mark.parametrize(
    ("engine", "role"),
    [
        ({}, {}),
        ({}, {"numa_cpu_bind": False}),
        # The engine binds every role; this role opts out, so the engine's memory policy does not apply either.
        ({"numa_cpu_bind": True, "numa_memory_bind": "local"}, {"numa_cpu_bind": False}),
    ],
    ids=["default", "role-off", "engine-on-role-off"],
)
@pytest.mark.parametrize(("frontend", "sidecar"), [("dynamo", False), ("dynamo", True), ("vllm", False)])
def test_option_off_leaves_commands_unchanged(engine, role, frontend, sidecar) -> None:
    """Unbound workers start exactly as a backend without the options starts them."""
    mode = "agg" if frontend == "vllm" else "prefill"
    args = {"tensor-parallel-size": 2}
    endpoints = [Endpoint(mode, endpoint.index, endpoint.nodes, endpoint.gpu_indices) for endpoint in PACKED]
    unbound = _backend(role, args, mode=mode, **engine)

    commands = [command for _, command in _commands(unbound, endpoints, frontend, sidecar)]

    assert commands == [
        command for _, command in _commands(_backend({}, args, mode=mode), endpoints, frontend, sidecar)
    ]
    assert not any("numa_cpu_bind.sh" in " ".join(command) for command in commands)
    assert unbound.numa_cpu_bind_for_mode(mode) is False
    assert unbound.numa_wrapper_policy_for_mode(mode) is None


@pytest.mark.parametrize(("frontend", "sidecar"), [("vllm", False), ("dynamo", True)])
def test_direct_and_sidecar_engines_are_bound(two_socket_node, frontend, sidecar) -> None:
    """The direct `vllm serve` worker is wrapped whole; in sidecar mode only the engine is, not the sidecar."""
    backend = _backend({"numa_cpu_bind": True, "numa_memory_bind": "local"}, {"tensor-parallel-size": 2}, mode="agg")

    ((_, command),) = _commands(backend, [Endpoint("agg", 0, ("node0",), frozenset({2, 3}))], frontend, sidecar)

    if sidecar:
        script = command[2].splitlines()
        (engine,) = [line.removesuffix(" &") for line in script if "vllm-rs serve" in line]
        command = shlex.split(engine)
        assert not any("numa_cpu_bind.sh" in line for line in script if "dynamo.vllm.sidecar" in line)
    assert command[:5] == ["bash", "/configs/numa_cpu_bind.sh", "--gpus", "2,3", "--bind-memory"]
    assert _numactl_options(two_socket_node, command) == ["--cpunodebind=1", "--membind=1"]


def test_wrapper_gpus_ignore_slurm_localid_and_visible_devices(two_socket_node) -> None:
    """With --gpus the wrapper reads neither SLURM_LOCALID nor CUDA_VISIBLE_DEVICES."""
    env = {**two_socket_node, "SLURM_LOCALID": "1", "CUDA_VISIBLE_DEVICES": "2,3"}
    assert _numactl_options(env, ["bash", "/configs/numa_cpu_bind.sh", "--gpus", "0,1", "true"]) == ["--cpunodebind=0"]


@pytest.mark.parametrize("gpus", ["", "0,", "a", "0 1", "-1"])
def test_wrapper_rejects_malformed_gpu_list(two_socket_node, gpus) -> None:
    result = _launch(two_socket_node, ["bash", "/configs/numa_cpu_bind.sh", "--gpus", gpus, "true"])
    assert result.returncode == 2
    assert "--gpus needs a comma-separated list" in result.stderr


@pytest.mark.parametrize("memory_flag", [None, "--bind-memory", "--prefer-memory"])
def test_wrapper_unknown_gpu_node_fails_only_with_a_memory_policy(two_socket_node, memory_flag) -> None:
    """One GPU without NUMA affinity runs the worker unbound, or stops it when a memory policy needs the node."""
    two_socket_node["TEST_GPU_NODES"] = json.dumps({**GPU_NODES, "3": "-1"})
    command = ["bash", "/configs/numa_cpu_bind.sh", "--gpus", "2,3", *([memory_flag] if memory_flag else [])]
    result = _launch(two_socket_node, [*command, sys.executable, "-c", "print('worker ran')"])
    assert not Path(two_socket_node["TEST_NUMACTL_TRACE"]).exists()
    if memory_flag:
        assert result.returncode == 2
        assert "GPU 3" in result.stderr and "cannot apply memory policy" in result.stderr
        assert result.stdout == ""
    else:
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "worker ran"


def test_wrapper_stops_on_a_node_without_cpus(two_socket_node) -> None:
    two_socket_node["TEST_NODE_CPUS"] = json.dumps({"0": "0-3", "1": ""})
    result = _launch(two_socket_node, ["bash", "/configs/numa_cpu_bind.sh", "--gpus", "1,2", "true"])
    assert result.returncode == 2
    assert "NUMA node 1 has no readable, nonempty CPU list" in result.stderr


@pytest.mark.parametrize("memory_bind", [None, "local", "preferred"])
def test_engine_options_round_trip(memory_bind) -> None:
    schema = VLLMBackend.Schema()
    backend = schema.load({"numa_cpu_bind": True, "numa_memory_bind": memory_bind})
    assert backend.numa_memory_bind == memory_bind
    assert schema.dump(backend)["numa_memory_bind"] == memory_bind


@pytest.mark.parametrize("memory_bind", [True, False, "interleave"])
def test_engine_memory_bind_takes_local_or_preferred(memory_bind) -> None:
    """vLLM has no default two-node policy to switch, so TRT-LLM's boolean values are not accepted."""
    with pytest.raises(ValidationError, match="numa_memory_bind"):
        VLLMBackend.Schema().load({"numa_cpu_bind": True, "numa_memory_bind": memory_bind})


@pytest.mark.parametrize(
    ("engine", "role", "args", "error"),
    [
        ({"numa_memory_bind": "local"}, {}, {}, "engine.numa_memory_bind: local requires numa_cpu_bind: true"),
        (
            {},
            {"numa_memory_bind": "preferred"},
            {},
            "roles.prefill.numa_memory_bind: preferred requires numa_cpu_bind: true",
        ),
        (
            {"numa_cpu_bind": True},
            {"numa_cpu_bind": False, "numa_memory_bind": "local"},
            {},
            "roles.prefill.numa_memory_bind: local requires numa_cpu_bind: true",
        ),
        ({}, {"numa_cpu_bind": True}, {"numa-bind": True}, r"roles.prefill.args.numa-bind: srtctl already binds"),
        (
            {"numa_cpu_bind": True},
            {},
            {"numa_bind_nodes": [0, 0], "numa-bind-cpus": ["0-3", "4-7"]},
            r"roles.prefill.args.numa-bind-cpus, roles.prefill.args.numa-bind-nodes: srtctl already binds",
        ),
    ],
)
def test_invalid_combinations_fail_at_load(engine, role, args, error) -> None:
    with pytest.raises(ValidationError, match=error):
        _backend(role, {"tensor-parallel-size": 2, **args}, **engine)


@pytest.mark.parametrize(
    ("role", "args"),
    [
        # vLLM's own binding stays available to a role srtctl does not bind.
        ({}, {"numa-bind": True, "numa-bind-nodes": [0, 0, 1, 1]}),
        ({"numa_cpu_bind": False}, {"numa-bind": True}),
        # Turning vLLM's best-effort default off explicitly does not conflict.
        ({"numa_cpu_bind": True}, {"numa-bind": False}),
    ],
)
def test_vllm_numa_args_are_accepted_where_srtctl_does_not_bind_them(role, args) -> None:
    _backend(role, {"tensor-parallel-size": 2, **args})


def _example() -> dict[str, Any]:
    recipe = yaml.safe_load(EXAMPLE.read_text())
    recipe["model"].update(path="hf:Qwen/Qwen3-0.6B", container="vllm.sqsh")
    return recipe


def _load(recipe: dict[str, Any], tmp_path: Path) -> SrtConfig:
    path = tmp_path / "recipe.yaml"
    path.write_text(yaml.safe_dump(recipe))
    return SrtConfig.from_yaml(path)


def test_example_binds_prefill_and_leaves_decode_alone(tmp_path: Path) -> None:
    backend = _load(_example(), tmp_path).backend
    assert isinstance(backend, VLLMBackend)
    assert backend.numa_cpu_bind_for_mode("prefill") is True
    assert backend.numa_wrapper_policy_for_mode("prefill") is not None
    assert backend.numa_cpu_bind_for_mode("decode") is False
    assert backend.numa_wrapper_policy_for_mode("decode") is None


@pytest.mark.parametrize("engine", [{"type": "trtllm"}, {"type": "sglang"}])
@pytest.mark.parametrize("key", [{"numa_cpu_bind": True}, {"numa_memory_bind": "local"}])
def test_role_keys_are_refused_on_other_engines(tmp_path: Path, engine, key) -> None:
    recipe = _example()
    recipe["engine"] = engine
    for spec in recipe["roles"].values():
        spec.pop("numa_cpu_bind", None)
        spec.pop("numa_memory_bind", None)
        spec["args"] = {}
    recipe["roles"]["prefill"].update(key)
    with pytest.raises(ValidationError, match="read by the vllm engine only"):
        _load(recipe, tmp_path)
