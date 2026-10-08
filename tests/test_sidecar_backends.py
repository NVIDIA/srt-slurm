# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Observable command and topology contracts for native-gRPC sidecars."""

import json
import shlex
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from srtctl.backends import SGLangBackend, TRTLLMBackend, VLLMBackend
from srtctl.cli.do_sweep import SweepOrchestrator
from srtctl.core.config import load_config
from srtctl.core.runtime import Nodes, RuntimeContext
from srtctl.core.schema import DynamoConfig, RoleConfig
from srtctl.core.topology import Endpoint, NodePortAllocator, Process


def _process(
    *,
    node: str = "node0",
    node_rank: int = 0,
    mode: str = "agg",
    sys_port: int = 7500,
    kv_events_port: int | None = None,
) -> Process:
    # Stand-in for what endpoints_to_processes(dynamo_sidecar=True) allocates:
    # one sidecar gRPC port and one NCCL port per process, in process order.
    ordinal = sys_port - 7500
    return Process(
        node=node,
        gpu_indices=frozenset(range(4)),
        sys_port=sys_port,
        http_port=6100,
        endpoint_mode=mode,
        endpoint_index=0,
        node_rank=node_rank,
        bootstrap_port=7200 if mode == "prefill" else None,
        kv_events_port=kv_events_port,
        sidecar_grpc_port=50051 + ordinal,
        nccl_port=17500 + ordinal,
        dist_init_port=8300,
    )


def _runtime(tmp_path: Path | None = None) -> MagicMock:
    runtime = MagicMock()
    runtime.model_path = Path("/models/example-model")
    runtime.worker_model_arg = "/model"
    runtime.is_hf_model = False
    runtime.gpu_type = "h100"
    runtime.log_dir = tmp_path or Path("/tmp")
    runtime.network_interface = None
    runtime.dynamo = DynamoConfig(sidecar=True)
    return runtime


def _sglang_launch_commands(command: list[str]) -> tuple[list[str], list[str] | None]:
    """Read the executable commands, preserving quoted JSON and argument boundaries."""
    if command[:3] == ["python3", "-m", "sglang.launch_server"]:
        return command, None
    assert command[:2] == ["bash", "-lc"]
    commands = [shlex.split(line.removesuffix(" &")) for line in command[2].splitlines() if line.startswith("python3 ")]
    engine = next(tokens for tokens in commands if tokens[:3] == ["python3", "-m", "sglang.launch_server"])
    sidecar = next(tokens for tokens in commands if tokens[:3] == ["python3", "-m", "dynamo.sglang.sidecar"])
    return engine, sidecar


def _sglang_endpoint(mode: str, node_count: int, gpus_per_node: int, index: int = 0) -> Endpoint:
    return Endpoint(
        mode=mode,
        index=index,
        nodes=tuple(f"node{rank}" for rank in range(node_count)),
        gpu_indices=frozenset(range(gpus_per_node)),
        gpus_per_node=gpus_per_node,
    )


def test_sglang_sidecar_owns_leader_and_couples_lifecycle() -> None:
    leader = _process(mode="prefill")
    follower = _process(node="node1", node_rank=1, mode="prefill", sys_port=7501)
    backend = SGLangBackend(roles={"prefill": RoleConfig(args={"tensor-parallel-size": 8})})

    with patch("srtctl.core.slurm.get_hostname_ip", return_value="10.0.0.1"):
        leader_command = backend.build_worker_command(leader, [leader, follower], _runtime())
        follower_command = backend.build_worker_command(follower, [leader, follower], _runtime())

    leader_script = leader_command[2]
    assert "python3 -m sglang.launch_server" in leader_script
    assert "--grpc-port 50051" in leader_script
    assert "python3 -m dynamo.sglang.sidecar --grpc-endpoint 127.0.0.1:50051" in leader_script
    assert 'wait -n "${ENGINE_PID}" "${SIDECAR_PID}"' in leader_script
    assert follower_command[:3] == ["python3", "-m", "sglang.launch_server"]
    assert "--grpc-port" not in follower_command
    assert "dynamo.sglang.sidecar" not in follower_command
    # The sidecar consumes deltas; the engine must stream disjoint segments on every rank.
    assert "--incremental-streaming-output" in leader_script
    assert "--incremental-streaming-output" in follower_command
    assert "--nccl-port" in leader_script
    assert "--nccl-port" in follower_command


@pytest.mark.parametrize("mode", ["agg", "prefill", "decode"])
@pytest.mark.parametrize(
    "args",
    [
        {"tensor-parallel-size": 1, "data-parallel-size": 2},
        {"tensor_parallel_size": 1, "data_parallel_size": 2},
        {"tp-size": 1, "dp-size": 2},
        {"tp_size": 1, "dp_size": 2},
        {"tp-size": 1, "dp-size": 2, "enable-dp-attention": False},
        {"tp_size": 1, "dp_size": 2, "enable_dp_attention": False},
    ],
)
def test_sglang_regular_dp_sidecar_leaves_nccl_ports_to_engine(mode: str, args: dict) -> None:
    """A fixed NCCL port makes regular DP's independent TP groups collide."""
    backend = SGLangBackend(roles={mode: RoleConfig(args=args)})
    processes = backend.endpoints_to_processes([_sglang_endpoint(mode, 1, 2)], dynamo_sidecar=True)
    with patch("srtctl.core.slurm.get_hostname_ip", return_value="10.0.0.1"):
        command = backend.build_worker_command(processes[0], processes, _runtime())
    engine, sidecar = _sglang_launch_commands(command)
    assert "--nccl-port" not in engine
    assert "--enable-dp-attention" not in engine
    assert "--grpc-port" in engine
    assert sidecar is not None
    assert ("--bootstrap-host" in sidecar) is (mode == "prefill")


def test_sglang_sidecar_respects_an_explicit_incremental_streaming_setting() -> None:
    process = _process(mode="agg")
    backend = SGLangBackend(
        roles={"agg": RoleConfig(args={"tensor-parallel-size": 4, "incremental-streaming-output": False})}
    )
    with patch("srtctl.core.slurm.get_hostname_ip", return_value="10.0.0.1"):
        command = backend.build_worker_command(process, [process], _runtime())
    leader_script = command[2]
    # An explicit false is honored: a false bool renders as no flag at all, and srtctl must not
    # add its own copy on top. An explicit true renders exactly once.
    assert "incremental-streaming-output" not in leader_script
    backend_true = SGLangBackend(
        roles={"agg": RoleConfig(args={"tensor-parallel-size": 4, "incremental-streaming-output": True})}
    )
    with patch("srtctl.core.slurm.get_hostname_ip", return_value="10.0.0.1"):
        command_true = backend_true.build_worker_command(process, [process], _runtime())
    assert command_true[2].count("--incremental-streaming-output") == 1


def test_sglang_sidecar_kv_events_config_true_covers_aggregated_mode() -> None:
    # Regression: the kv_events_config=True shortcut only matched prefill/decode, so an
    # aggregated topology never got --kv-events-config and the sidecar's
    # kv_event_sources stayed at 0 (every routed request scored 0.00 cache overlap).
    process = _process(mode="agg", kv_events_port=5557)
    backend = SGLangBackend(
        roles={
            "prefill": RoleConfig(kv_events=True),
            "decode": RoleConfig(kv_events=True),
            "agg": RoleConfig(args={"tensor-parallel-size": 8}, kv_events=True),
        }
    )

    with patch("srtctl.core.slurm.get_hostname_ip", return_value="10.0.0.1"):
        command = backend.build_worker_command(process, [process], _runtime())

    engine, _sidecar = _sglang_launch_commands(command)
    kv_config = json.loads(engine[engine.index("--kv-events-config") + 1])
    assert kv_config["endpoint"] == "tcp://*:5557"
    assert kv_config["publisher"] == "zmq"


@pytest.mark.parametrize("mode", ["agg", "prefill", "decode"])
@pytest.mark.parametrize(
    "args",
    [
        {"tensor-parallel-size": 8, "data-parallel-size": 2, "enable-dp-attention": True},
        {"tensor_parallel_size": 8, "data_parallel_size": 2, "enable_dp_attention": True},
        {"tp-size": 8, "dp-size": 2, "enable-dp-attention": True},
        {"tp_size": 8, "dp_size": 2, "enable_dp_attention": True},
    ],
)
def test_sglang_multinode_dp_sidecar_relays_follower_kv_events(mode: str, args: dict) -> None:
    backend = SGLangBackend(roles={mode: RoleConfig(args=args, kv_events={"topic": "cache events"})})
    processes = backend.endpoints_to_processes([_sglang_endpoint(mode, 2, 4)], dynamo_sidecar=True)

    with patch("srtctl.core.slurm.get_hostname_ip", return_value="10.0.0.1"):
        commands = [backend.build_worker_command(process, processes, _runtime()) for process in processes]

    for node_rank, command in enumerate(commands):
        engine, sidecar = _sglang_launch_commands(command)
        assert sidecar is not None
        grpc_port = engine[engine.index("--grpc-port") + 1]
        assert sidecar[sidecar.index("--grpc-endpoint") + 1] == f"127.0.0.1:{grpc_port}"
        assert engine[engine.index("--node-rank") + 1] == str(node_rank)
        assert engine[engine.index("--nnodes") + 1] == "2"
        assert engine[engine.index("--dist-init-addr") + 1].startswith("10.0.0.1:")
        assert "--incremental-streaming-output" in engine
        assert "--nccl-port" in engine
        kv_config = json.loads(engine[engine.index("--kv-events-config") + 1])
        assert kv_config["endpoint"] == f"tcp://*:{processes[node_rank].kv_events_port}"
        assert kv_config["topic"] == "cache events"
        assert "--telemetry-only" not in sidecar
        assert ("--bootstrap-host" in sidecar) is (mode == "prefill" and node_rank == 0)
        if mode != "agg":
            assert engine[engine.index("--disaggregation-mode") + 1] == mode
        assert 'wait -n "${ENGINE_PID}" "${SIDECAR_PID}"' in command[2]


def test_sglang_disagg_dp_example_launches_two_disjoint_worker_groups(tmp_path: Path) -> None:
    """Exercise recipe normalization, allocation, role grouping, and all four srun launches."""
    recipe = Path(__file__).parents[1] / "examples/features/sglang-sidecar-multinode-disagg-dp.yaml"
    with patch("srtctl.core.config.load_cluster_config", return_value={}):
        config = load_config(recipe)
    nodes = ("prefill0", "prefill1", "decode0", "decode1")
    node_ips = {node: f"10.0.0.{index + 1}" for index, node in enumerate(nodes)}
    runtime = RuntimeContext(
        job_id="12345",
        run_name=config.name,
        nodes=Nodes(head=nodes[0], bench=nodes[0], infra=nodes[0], worker=nodes),
        head_node_ip=node_ips[nodes[0]],
        infra_node_ip=node_ips[nodes[0]],
        log_dir=tmp_path,
        model_path=Path("Qwen/Qwen3-0.6B"),
        container_image=Path(config.model.container),
        gpus_per_node=config.resources.gpus_per_node,
        network_interface=None,
        is_hf_model=True,
        dynamo=config.dynamo,
        request_plane=config.dynamo.request_plane,
    )
    orchestrator = SweepOrchestrator(config, runtime)

    assert config.total_nodes == 4
    assert config.dynamo.sidecar
    assert config.frontend.args["router-mode"] == "kv"
    assert [(endpoint.mode, endpoint.nodes, endpoint.total_gpus) for endpoint in orchestrator.endpoints] == [
        ("prefill", nodes[:2], 2),
        ("decode", nodes[2:], 2),
    ]

    def node_ip(node: str, _interface: str | None = None) -> str:
        return node_ips[node]

    with (
        patch("srtctl.core.slurm.get_hostname_ip", side_effect=node_ip),
        patch("srtctl.cli.mixins.worker_stage.get_hostname_ip", side_effect=node_ip),
        patch("srtctl.cli.mixins.worker_stage.generate_capture_script", return_value="true"),
        patch("srtctl.cli.mixins.worker_stage.start_srun_process") as mock_srun,
    ):
        managed = orchestrator.start_all_workers()

    assert len(managed) == mock_srun.call_count == 4
    for process, launch in zip(orchestrator.backend_processes, mock_srun.call_args_list, strict=True):
        args = launch.kwargs
        engine, sidecar = _sglang_launch_commands(args["command"])
        prefill = process.endpoint_mode == "prefill"
        leader = process.node_rank == 0
        group_leader = nodes[0] if prefill else nodes[2]
        assert process.gpu_indices == frozenset({0})
        assert args["nodelist"] == [process.node]
        assert managed[args["step_name"]].critical
        assert args["env_to_set"]["DYN_REQUEST_PLANE"] == "nats"
        assert args["env_to_set"]["DYN_EVENT_PLANE"] == "nats"
        assert args["env_to_set"]["SGLANG_RUST_BUILD_MODE"] == "never"
        assert engine[engine.index("--model-path") + 1] == "Qwen/Qwen3-0.6B"
        assert engine[engine.index("--disaggregation-mode") + 1] == process.endpoint_mode
        assert engine[engine.index("--disaggregation-transfer-backend") + 1] == "nixl"
        assert engine[engine.index("--nnodes") + 1] == "2"
        assert engine[engine.index("--node-rank") + 1] == str(process.node_rank)
        assert engine[engine.index("--dist-init-addr") + 1].split(":")[0] == node_ips[group_leader]
        assert engine[engine.index("--tensor-parallel-size") + 1] == "2"
        assert engine[engine.index("--data-parallel-size") + 1] == "2"
        assert "--enable-dp-attention" in engine
        assert "--skip-server-warmup" in engine
        assert "--incremental-streaming-output" in engine

        assert ("--disaggregation-bootstrap-port" in engine) is prefill
        assert ("--kv-events-config" in engine) is prefill
        if prefill:
            assert engine[engine.index("--disaggregation-bootstrap-port") + 1] == str(process.bootstrap_port)
            kv_config = json.loads(engine[engine.index("--kv-events-config") + 1])
            assert kv_config["endpoint"] == f"tcp://*:{process.kv_events_port}"
            assert kv_config["publisher"] == "zmq"

        assert (sidecar is not None) is (prefill or leader)
        assert ("--grpc-port" in engine) is (prefill or leader)
        if sidecar is not None:
            grpc_port = engine[engine.index("--grpc-port") + 1]
            assert sidecar[sidecar.index("--grpc-endpoint") + 1] == f"127.0.0.1:{grpc_port}"
            assert "--telemetry-only" not in sidecar
            assert ("--bootstrap-host" in sidecar) is (prefill and leader)
            if prefill and leader:
                assert sidecar[sidecar.index("--bootstrap-host") + 1] == node_ips[group_leader]


@pytest.mark.parametrize(
    ("args", "node_count", "gpus_per_node", "publisher_nodes"),
    [
        ({"tp-size": 8, "dp-size": 4, "enable-dp-attention": True}, 2, 4, {0, 1}),
        # Each attention-DP group spans two nodes; only its first node publishes.
        ({"tp-size": 8, "dp-size": 2, "enable-dp-attention": True}, 4, 2, {0, 2}),
        # CP rank zero is the publisher even when all attention TP ranks are zero.
        ({"tp-size": 8, "dp-size": 2, "attn-cp-size": 4, "enable-dp-attention": True}, 4, 2, {0, 2}),
        # Later pipeline stages cannot publish the rank's KV events.
        ({"tp-size": 8, "dp-size": 2, "pp-size": 2, "enable-dp-attention": True}, 4, 4, {0, 1}),
        (
            {
                "tensor_parallel_size": 8,
                "data_parallel_size": 2,
                "pipeline_parallel_size": 2,
                "enable_dp_attention": True,
            },
            4,
            4,
            {0, 1},
        ),
        ({"tp-size": 8}, 2, 4, {0}),
        ({"tp-size": 4, "pp-size": 2}, 2, 4, {0}),
    ],
)
def test_sglang_sidecar_only_runs_on_nodes_with_publishers(
    args: dict, node_count: int, gpus_per_node: int, publisher_nodes: set[int]
) -> None:
    backend = SGLangBackend(roles={"agg": RoleConfig(args=args, kv_events=True)})
    processes = backend.endpoints_to_processes(
        [_sglang_endpoint("agg", node_count, gpus_per_node)], dynamo_sidecar=True
    )

    with patch("srtctl.core.slurm.get_hostname_ip", return_value="10.0.0.1"):
        commands = [backend.build_worker_command(process, processes, _runtime()) for process in processes]

    for node_rank, command in enumerate(commands):
        engine, sidecar = _sglang_launch_commands(command)
        assert (sidecar is not None) is (node_rank in publisher_nodes)
        assert ("--grpc-port" in engine) is (node_rank in publisher_nodes)
        if sidecar is not None:
            assert "--telemetry-only" not in sidecar


@pytest.mark.parametrize("kv_events_config", [None, False])
def test_sglang_multinode_dp_without_kv_events_has_no_follower_sidecar(kv_events_config: bool | None) -> None:
    backend = SGLangBackend(
        roles={
            "agg": RoleConfig(
                args={"tp-size": 8, "dp-size": 2, "enable-dp-attention": True}, kv_events=kv_events_config
            )
        }
    )
    processes = backend.endpoints_to_processes([_sglang_endpoint("agg", 2, 4)], dynamo_sidecar=True)
    with patch("srtctl.core.slurm.get_hostname_ip", return_value="10.0.0.1"):
        leader = backend.build_worker_command(processes[0], processes, _runtime())
        follower = backend.build_worker_command(processes[1], processes, _runtime())

    leader_engine, leader_sidecar = _sglang_launch_commands(leader)
    follower_engine, follower_sidecar = _sglang_launch_commands(follower)
    assert leader_sidecar is not None
    assert "--telemetry-only" not in leader_sidecar
    assert "--grpc-port" in leader_engine
    assert follower_sidecar is None
    assert "--grpc-port" not in follower_engine
    assert "--kv-events-config" not in leader_engine
    assert "--kv-events-config" not in follower_engine


def test_sglang_null_publisher_has_no_follower_sidecar() -> None:
    backend = SGLangBackend(
        roles={
            "agg": RoleConfig(
                args={"tp-size": 8, "dp-size": 2, "enable-dp-attention": True}, kv_events={"publisher": "null"}
            )
        }
    )
    processes = backend.endpoints_to_processes([_sglang_endpoint("agg", 2, 4)], dynamo_sidecar=True)
    with patch("srtctl.core.slurm.get_hostname_ip", return_value="10.0.0.1"):
        commands = [backend.build_worker_command(process, processes, _runtime()) for process in processes]

    leader_engine, leader_sidecar = _sglang_launch_commands(commands[0])
    follower_engine, follower_sidecar = _sglang_launch_commands(commands[1])
    assert leader_sidecar is not None
    assert "--telemetry-only" not in leader_sidecar
    assert follower_sidecar is None
    assert "--grpc-port" not in follower_engine
    assert json.loads(leader_engine[leader_engine.index("--kv-events-config") + 1])["publisher"] == "null"


def test_sglang_colocated_dp_workers_reserve_nonoverlapping_publisher_ports() -> None:
    backend = SGLangBackend(
        roles={"agg": RoleConfig(args={"tp-size": 4, "dp-size": 4, "enable-dp-attention": True}, kv_events=True)}
    )
    endpoints = [
        Endpoint(mode="agg", index=0, nodes=("node0",), gpu_indices=frozenset(range(4)), gpus_per_node=8),
        Endpoint(mode="agg", index=1, nodes=("node0",), gpu_indices=frozenset(range(4, 8)), gpus_per_node=8),
    ]
    processes = backend.endpoints_to_processes(endpoints, dynamo_sidecar=True)
    publisher_ports = []
    grpc_ports = []
    with patch("srtctl.core.slurm.get_hostname_ip", return_value="10.0.0.1"):
        for process in processes:
            command = backend.build_worker_command(process, [process], _runtime())
            engine, sidecar = _sglang_launch_commands(command)
            assert sidecar is not None
            assert "--telemetry-only" not in sidecar
            kv_config = json.loads(engine[engine.index("--kv-events-config") + 1])
            base_port = int(kv_config["endpoint"].rsplit(":", 1)[1])
            # SGLang offsets the configured port by the global DP rank.
            publisher_ports.append(set(range(base_port, base_port + 4)))
            grpc_ports.append(engine[engine.index("--grpc-port") + 1])

    assert publisher_ports[0].isdisjoint(publisher_ports[1])
    assert grpc_ports[0] != grpc_ports[1]


def test_sglang_follower_global_dp_port_offsets_do_not_collide_with_next_worker() -> None:
    allocator = NodePortAllocator()
    multinode_backend = SGLangBackend(
        roles={"agg": RoleConfig(args={"tp-size": 4, "dp-size": 4, "enable-dp-attention": True}, kv_events=True)}
    )
    local_backend = SGLangBackend(
        roles={"decode": RoleConfig(args={"tp-size": 2, "dp-size": 2, "enable-dp-attention": True}, kv_events=True)}
    )
    multinode_processes = multinode_backend.endpoints_to_processes(
        [_sglang_endpoint("agg", 2, 2)], port_allocator=allocator, dynamo_sidecar=True
    )
    local_processes = local_backend.endpoints_to_processes(
        [Endpoint(mode="decode", index=0, nodes=("node1",), gpu_indices=frozenset({2, 3}), gpus_per_node=4)],
        base_sys_port=7502,
        port_allocator=allocator,
        dynamo_sidecar=True,
    )

    with patch("srtctl.core.slurm.get_hostname_ip", return_value="10.0.0.1"):
        follower_command = multinode_backend.build_worker_command(
            multinode_processes[1], multinode_processes, _runtime()
        )
        local_command = local_backend.build_worker_command(local_processes[0], local_processes, _runtime())

    follower_engine, follower_sidecar = _sglang_launch_commands(follower_command)
    local_engine, local_sidecar = _sglang_launch_commands(local_command)
    assert follower_sidecar is not None and "--telemetry-only" not in follower_sidecar
    assert local_sidecar is not None and "--telemetry-only" not in local_sidecar
    follower_config = json.loads(follower_engine[follower_engine.index("--kv-events-config") + 1])
    local_config = json.loads(local_engine[local_engine.index("--kv-events-config") + 1])
    follower_base = int(follower_config["endpoint"].rsplit(":", 1)[1])
    local_base = int(local_config["endpoint"].rsplit(":", 1)[1])
    # node1 owns ranks 2/3 of the multinode group and ranks 0/1 of its local group.
    assert {follower_base + 2, follower_base + 3}.isdisjoint({local_base, local_base + 1})


@pytest.mark.parametrize("dp_size", [8, 12])
def test_vllm_sidecar_exposes_each_nodes_hybrid_dp_range(dp_size: int) -> None:
    # Regression: a headless follower has no local gRPC/sidecar endpoint, so
    # Dynamo cannot route to that node independently of the group leader.
    backend = VLLMBackend(
        connector=None,
        roles={
            "decode": RoleConfig(args={"data-parallel-size": dp_size, "enable-expert-parallel": True}, kv_events=True)
        },
    )
    endpoint = Endpoint(
        mode="decode",
        index=0,
        nodes=tuple(f"node{i}" for i in range(dp_size // 4)),
        gpu_indices=frozenset(range(4)),
        gpus_per_node=4,
    )
    processes = backend.endpoints_to_processes([endpoint], dynamo_sidecar=True)
    node_ips = {node: f"10.0.0.{i + 1}" for i, node in enumerate(endpoint.nodes)}

    with patch("srtctl.core.slurm.get_hostname_ip", side_effect=lambda node, _interface=None: node_ips[node]):
        commands = [backend.build_worker_command(process, processes, _runtime()) for process in processes]

    assert [process.node_rank for process in processes] == list(range(0, dp_size, 4))
    for i, (process, command) in enumerate(zip(processes, commands, strict=True)):
        assert command[:2] == ["bash", "-lc"]
        script = command[2]
        engine_line = next(line for line in script.splitlines() if "vllm.entrypoints.cli.main serve" in line)
        engine = shlex.split(engine_line)
        assert "VLLM_USE_RUST_FRONTEND=1" in engine
        assert engine[engine.index("--data-parallel-size") + 1] == str(dp_size)
        assert engine[engine.index("--data-parallel-size-local") + 1] == "4"
        assert engine[engine.index("--data-parallel-start-rank") + 1] == str(i * 4)
        assert "--data-parallel-hybrid-lb" in engine
        assert "--headless" not in engine
        assert engine[engine.index("--data-parallel-address") + 1] == node_ips["node0"]
        assert engine[engine.index("--data-parallel-rpc-port") + 1] == str(processes[0].dp_rpc_port)
        assert engine[engine.index("--grpc-port") + 1] == str(50051 + i)
        assert f"python3 -m dynamo.vllm.sidecar --grpc-endpoint 127.0.0.1:{50051 + i}" in script
        assert 'wait -n "${ENGINE_PID}" "${SIDECAR_PID}"' in script
        kv_config = json.loads(engine[engine.index("--kv-events-config") + 1])
        assert kv_config["endpoint"] == f"tcp://{node_ips[process.node]}:{process.kv_events_port}"


@pytest.mark.parametrize("override", [{"grpc": True}, {"data_parallel_external_lb": True}, {"api-server-count": 0}])
def test_vllm_sidecar_rejects_frontend_options_that_bypass_hybrid_lb(override: dict) -> None:
    # A valid recipe must not disable the local frontend or select Python gRPC
    # while srtctl waits for a Rust Control service on that node.
    backend = VLLMBackend(connector=None, roles={"decode": RoleConfig(args={"data-parallel-size": 8, **override})})
    endpoint = Endpoint(
        mode="decode",
        index=0,
        nodes=("node0", "node1"),
        gpu_indices=frozenset(range(4)),
        gpus_per_node=4,
    )
    processes = backend.endpoints_to_processes([endpoint], dynamo_sidecar=True)
    with (
        patch("srtctl.core.slurm.get_hostname_ip", return_value="10.0.0.1"),
        pytest.raises(ValueError, match="sidecar hybrid mode requires"),
    ):
        backend.build_worker_command(processes[0], processes, _runtime())


@pytest.mark.parametrize(
    "parallelism",
    [
        {"tensor-parallel-size": 8},
        {"tensor_parallel_size": 8, "enable-expert-parallel": True, "data-parallel-size": 1},
        {"tensor-parallel-size": 4, "pipeline-parallel-size": 2},
        {"tensor-parallel-size": 16, "enable-expert-parallel": True},
    ],
)
def test_vllm_sidecar_multi_node_replica_has_one_frontend(parallelism: dict) -> None:
    # Regression: exposing a sidecar on a TP follower either hangs startup or
    # registers an engine incapable of serving independent requests.
    backend = VLLMBackend(
        connector=None,
        roles={
            "agg": RoleConfig(
                args={
                    **parallelism,
                    "api-server-count": 1,
                    "master_addr": "stale-host",
                    "node_rank": 7,
                    "nnodes": 9,
                    "master_port": 1234,
                }
            )
        },
    )
    tp = parallelism.get("tensor-parallel-size", parallelism.get("tensor_parallel_size", 1))
    node_count = tp * parallelism.get("pipeline-parallel-size", 1) // 4
    endpoint = Endpoint(
        mode="agg",
        index=0,
        nodes=tuple(f"node{i}" for i in range(node_count)),
        gpu_indices=frozenset(range(4)),
        gpus_per_node=4,
    )
    processes = backend.endpoints_to_processes([endpoint], dynamo_sidecar=True)
    runtime = _runtime()
    runtime.network_interface = "ib0"

    def node_ip(node, interface=None):
        assert interface == "ib0"
        return f"10.0.0.{endpoint.nodes.index(node) + 1}"

    with patch("srtctl.core.slurm.get_hostname_ip", side_effect=node_ip):
        commands = [backend.build_worker_command(p, processes, runtime) for p in processes]

    engines = []
    for rank, command in enumerate(commands):
        subprocess.run(["bash", "-n", "-c", command[2]], check=True)
        engine = shlex.split(
            next(line for line in command[2].splitlines() if "vllm.entrypoints.cli.main serve" in line)
        )
        engines.append(engine)
        assert "VLLM_USE_RUST_FRONTEND=1" in engine
        for flag, value in {
            "--nnodes": str(node_count),
            "--node-rank": str(rank),
            "--master-addr": "10.0.0.1",
            "--distributed-executor-backend": "mp",
        }.items():
            assert engine.count(flag) == 1
            assert engine[engine.index(flag) + 1] == value
        assert ("--enable-expert-parallel" in engine) == bool(parallelism.get("enable-expert-parallel"))
    master_ports = [engine[engine.index("--master-port") + 1] for engine in engines]
    assert len(set(master_ports)) == 1 and master_ports[0] != "1234"
    assert master_ports[0] != engines[0][engines[0].index("--port") + 1]
    assert "dynamo.vllm.sidecar" in commands[0][2]
    assert "--grpc-port" in engines[0]
    assert "--headless" not in engines[0]
    for engine, command in zip(engines[1:], commands[1:], strict=True):
        assert "--headless" in engine
        assert not {"--grpc", "--grpc-port", "--port", "--api-server-count"}.intersection(engine)
        assert "dynamo.vllm.sidecar" not in command[2]
        assert "/dev/tcp" not in command[2]

    from srtctl.cli.mixins.benchmark_stage import _get_health_expectations

    config = SimpleNamespace(
        backend=backend,
        dynamo=runtime.dynamo,
        frontend=SimpleNamespace(type="dynamo"),
        topology=SimpleNamespace(num_agg=1, num_prefill=0, num_decode=0),
    )
    prefill, decode, _, total = _get_health_expectations(config, processes)
    assert (prefill, decode, total) == (0, 1, 1)


@pytest.mark.parametrize("exit_code", [0, 7])
def test_headless_follower_exit_is_reported_as_failure(exit_code: int) -> None:
    # A clean but unexpected follower exit must trigger ProcessRegistry's
    # nonzero-exit detection and therefore stop the rest of the job.
    from srtctl.backends.sidecar import build_sidecar_launch_command

    command = build_sidecar_launch_command(
        engine=["bash", "-c", f"exit {exit_code}"],
        sidecar=None,
        grpc_port=50051,
        engine_name="vLLM follower",
        startup_timeout=60,
    )
    # Exercise the wrapper without sourcing workstation login-shell hooks.
    command[1] = "-c"
    result = subprocess.run(command, capture_output=True, text=True, timeout=15, check=False)
    assert result.returncode == (exit_code or 1)


def test_headless_follower_termination_reaps_engine(tmp_path: Path) -> None:
    # Job-wide cleanup signals the wrapper; its engine must not survive it.
    from srtctl.backends.sidecar import build_sidecar_launch_command

    ready = tmp_path / "ready"
    stopped = tmp_path / "stopped"
    engine = (
        "import signal,time; from pathlib import Path; "
        f"signal.signal(signal.SIGTERM, lambda *_: (Path({str(stopped)!r}).touch(), exit(0))); "
        f"Path({str(ready)!r}).touch(); time.sleep(60)"
    )
    command = build_sidecar_launch_command(
        engine=[sys.executable, "-c", engine],
        sidecar=None,
        grpc_port=50051,
        engine_name="vLLM follower",
        startup_timeout=60,
    )
    command[1] = "-c"
    child = subprocess.Popen(command)
    try:
        deadline = time.monotonic() + 5
        while not ready.exists() and child.poll() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ready.exists()
        child.terminate()
        child.wait(timeout=15)
        assert stopped.exists()
    finally:
        if child.poll() is None:
            child.terminate()
            child.wait(timeout=15)


@pytest.mark.parametrize("memory_bind", [False, True, "local"])
@pytest.mark.parametrize("bind_cpu", [False, True])
def test_trtllm_sidecar_uses_native_grpc_on_rank_zero(tmp_path: Path, memory_bind, bind_cpu) -> None:
    process = _process()
    backend = TRTLLMBackend(
        roles={"agg": RoleConfig(args={"tensor_parallel_size": 4, "max_seq_len": 4096})},
        numa_cpu_bind=bind_cpu,
        numa_memory_bind=memory_bind,
    )

    command = backend.build_worker_command(process, [process], _runtime(tmp_path))

    script = command[2]
    assert "trtllm-llmapi-launch python3 -m tensorrt_llm.commands.serve /model" in script
    assert ("bash /configs/numa_cpu_bind.sh --bind-memory" in script) is (memory_bind == "local")
    assert ("--no-bind-cpu" in script) is (memory_bind == "local" and not bind_cpu)
    assert ("numactl -m 0,1" in script) is (memory_bind is True)
    assert "--grpc --host 127.0.0.1 --port 50051" in script
    assert "python3 -m dynamo.trtllm.sidecar --grpc-endpoint 127.0.0.1:50051 --model-path /model" in script
    assert "--context-length 4096" in script
    assert "${SLURM_PROCID:-0}" in script


def test_trtllm_sidecar_rejects_disaggregated_workers(tmp_path: Path) -> None:
    backend = TRTLLMBackend()

    with pytest.raises(ValueError, match="supports aggregated workers only"):
        backend.build_worker_command(_process(mode="prefill"), [], _runtime(tmp_path))
