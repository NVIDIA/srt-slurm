# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Observable command and topology contracts for native-gRPC sidecars."""

import json
import shlex
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from srtctl.backends import (
    SGLangProtocol,
    SGLangServerConfig,
    TRTLLMProtocol,
    TRTLLMServerConfig,
    VLLMProtocol,
    VLLMServerConfig,
)
from srtctl.core.schema import DynamoConfig
from srtctl.core.topology import Endpoint, NodePortAllocator, Process


def _process(
    *,
    node: str = "node0",
    node_rank: int = 0,
    mode: str = "agg",
    sys_port: int = 7500,
    kv_events_port: int | None = None,
) -> Process:
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
    backend = SGLangProtocol(sglang_config=SGLangServerConfig(prefill={"tensor-parallel-size": 8}))

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


def test_sglang_sidecar_respects_an_explicit_incremental_streaming_setting() -> None:
    process = _process(mode="agg")
    backend = SGLangProtocol(
        sglang_config=SGLangServerConfig(aggregated={"tensor-parallel-size": 4, "incremental-streaming-output": False})
    )
    with patch("srtctl.core.slurm.get_hostname_ip", return_value="10.0.0.1"):
        command = backend.build_worker_command(process, [process], _runtime())
    leader_script = command[2]
    # An explicit false is honored: a false bool renders as no flag at all, and srtctl must not
    # add its own copy on top. An explicit true renders exactly once.
    assert "incremental-streaming-output" not in leader_script
    backend_true = SGLangProtocol(
        sglang_config=SGLangServerConfig(aggregated={"tensor-parallel-size": 4, "incremental-streaming-output": True})
    )
    with patch("srtctl.core.slurm.get_hostname_ip", return_value="10.0.0.1"):
        command_true = backend_true.build_worker_command(process, [process], _runtime())
    assert command_true[2].count("--incremental-streaming-output") == 1


def test_sglang_sidecar_kv_events_config_true_covers_aggregated_mode() -> None:
    # Regression: the kv_events_config=True shortcut only matched prefill/decode, so an
    # aggregated topology never got --kv-events-config and the sidecar's
    # kv_event_sources stayed at 0 (every routed request scored 0.00 cache overlap).
    process = _process(mode="agg", kv_events_port=5557)
    backend = SGLangProtocol(
        kv_events_config=True,
        sglang_config=SGLangServerConfig(aggregated={"tensor-parallel-size": 8}),
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
    config_mode = "aggregated" if mode == "agg" else mode
    backend = SGLangProtocol(
        kv_events_config={config_mode: {"topic": "cache events"}},
        sglang_config=SGLangServerConfig(**{config_mode: args}),
    )
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
        kv_config = json.loads(engine[engine.index("--kv-events-config") + 1])
        assert kv_config["endpoint"] == f"tcp://*:{processes[node_rank].kv_events_port}"
        assert kv_config["topic"] == "cache events"
        assert ("--telemetry-only" in sidecar) is (node_rank != 0)
        assert ("--bootstrap-host" in sidecar) is (mode == "prefill" and node_rank == 0)
        if mode != "agg":
            assert engine[engine.index("--disaggregation-mode") + 1] == mode
        assert 'wait -n "${ENGINE_PID}" "${SIDECAR_PID}"' in command[2]


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
    backend = SGLangProtocol(kv_events_config=True, sglang_config=SGLangServerConfig(aggregated=args))
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
            assert ("--telemetry-only" in sidecar) is (node_rank != 0)


@pytest.mark.parametrize("kv_events_config", [None, False, {"aggregated": False}, {"decode": True}])
def test_sglang_multinode_dp_without_kv_events_has_no_follower_sidecar(kv_events_config: bool | dict | None) -> None:
    backend = SGLangProtocol(
        kv_events_config=kv_events_config,
        sglang_config=SGLangServerConfig(aggregated={"tp-size": 8, "dp-size": 2, "enable-dp-attention": True}),
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
    backend = SGLangProtocol(
        kv_events_config={"aggregated": {"publisher": "null"}},
        sglang_config=SGLangServerConfig(aggregated={"tp-size": 8, "dp-size": 2, "enable-dp-attention": True}),
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
    backend = SGLangProtocol(
        kv_events_config=True,
        sglang_config=SGLangServerConfig(aggregated={"tp-size": 4, "dp-size": 4, "enable-dp-attention": True}),
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
    multinode_backend = SGLangProtocol(
        kv_events_config=True,
        sglang_config=SGLangServerConfig(aggregated={"tp-size": 4, "dp-size": 4, "enable-dp-attention": True}),
    )
    local_backend = SGLangProtocol(
        kv_events_config={"decode": True},
        sglang_config=SGLangServerConfig(decode={"tp-size": 2, "dp-size": 2, "enable-dp-attention": True}),
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
    assert follower_sidecar is not None and "--telemetry-only" in follower_sidecar
    assert local_sidecar is not None and "--telemetry-only" not in local_sidecar
    follower_config = json.loads(follower_engine[follower_engine.index("--kv-events-config") + 1])
    local_config = json.loads(local_engine[local_engine.index("--kv-events-config") + 1])
    follower_base = int(follower_config["endpoint"].rsplit(":", 1)[1])
    local_base = int(local_config["endpoint"].rsplit(":", 1)[1])
    # node1 owns ranks 2/3 of the multinode group and ranks 0/1 of its local group.
    assert {follower_base + 2, follower_base + 3}.isdisjoint({local_base, local_base + 1})


def test_vllm_sidecar_exposes_one_complete_multi_node_dp_group() -> None:
    backend = VLLMProtocol(
        connector=None,
        kv_events_config={"decode": True},
        vllm_config=VLLMServerConfig(decode={"data-parallel-size": 8, "enable-expert-parallel": True}),
    )
    endpoint = Endpoint(
        mode="decode",
        index=0,
        nodes=("node0", "node1"),
        gpu_indices=frozenset(range(4)),
        gpus_per_node=4,
    )
    processes = backend.endpoints_to_processes([endpoint], dynamo_sidecar=True)
    node_ips = {"node0": "10.0.0.1", "node1": "10.0.0.2"}

    with patch("srtctl.core.slurm.get_hostname_ip", side_effect=lambda node, _interface=None: node_ips[node]):
        leader_command = backend.build_worker_command(processes[0], processes, _runtime())
        follower_command = backend.build_worker_command(processes[1], processes, _runtime())

    assert [(process.node, process.node_rank) for process in processes] == [("node0", 0), ("node1", 4)]
    leader_script = leader_command[2]
    assert "--data-parallel-size 8 --data-parallel-size-local 4" in leader_script
    assert "python3 -m dynamo.vllm.sidecar --grpc-endpoint 127.0.0.1:50051" in leader_script
    assert follower_command[:3] == ["vllm-rs", "serve", "/model"]
    assert "--headless" in follower_command
    follower_kv_config = json.loads(follower_command[follower_command.index("--kv-events-config") + 1])
    assert follower_kv_config["endpoint"] == "tcp://10.0.0.2:5204"
    assert follower_command[-2:] == ["--data-parallel-start-rank", "4"]
    assert "dynamo.vllm.sidecar" not in follower_command


def test_vllm_sidecar_rejects_unimplemented_multi_node_tp() -> None:
    leader = _process(node="node0")
    follower = _process(node="node1", node_rank=1, sys_port=7501)
    backend = VLLMProtocol(connector=None)

    with (
        patch("srtctl.core.slurm.get_hostname_ip", return_value="10.0.0.1"),
        pytest.raises(ValueError, match="does not support multi-node tensor-parallel"),
    ):
        backend.build_worker_command(leader, [leader, follower], _runtime())


def test_trtllm_sidecar_uses_native_grpc_on_rank_zero(tmp_path: Path) -> None:
    process = _process()
    backend = TRTLLMProtocol(
        trtllm_config=TRTLLMServerConfig(aggregated={"tensor_parallel_size": 4, "max_seq_len": 4096}),
    )

    command = backend.build_worker_command(process, [process], _runtime(tmp_path))

    script = command[2]
    assert "trtllm-llmapi-launch python3 -m tensorrt_llm.commands.serve /model" in script
    assert "--grpc --host 127.0.0.1 --port 50051" in script
    assert "python3 -m dynamo.trtllm.sidecar --grpc-endpoint 127.0.0.1:50051 --model-path /model" in script
    assert "--context-length 4096" in script
    assert "${SLURM_PROCID:-0}" in script


def test_trtllm_sidecar_rejects_disaggregated_workers(tmp_path: Path) -> None:
    backend = TRTLLMProtocol()

    with pytest.raises(ValueError, match="supports aggregated workers only"):
        backend.build_worker_command(_process(mode="prefill"), [], _runtime(tmp_path))
