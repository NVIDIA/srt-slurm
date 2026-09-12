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
from srtctl.core.topology import Endpoint, Process


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


@pytest.mark.parametrize("dp_size", [8, 12])
def test_vllm_sidecar_exposes_each_nodes_hybrid_dp_range(dp_size: int) -> None:
    # Regression: a headless follower has no local gRPC/sidecar endpoint, so
    # Dynamo cannot route to that node independently of the group leader.
    backend = VLLMProtocol(
        connector=None,
        kv_events_config={"decode": True},
        vllm_config=VLLMServerConfig(decode={"data-parallel-size": dp_size, "enable-expert-parallel": True}),
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
    backend = VLLMProtocol(
        connector=None,
        vllm_config=VLLMServerConfig(decode={"data-parallel-size": 8, **override}),
    )
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
