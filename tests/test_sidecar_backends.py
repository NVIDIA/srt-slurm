# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Observable command and topology contracts for native-gRPC sidecars."""

import json
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


def test_sglang_sidecar_kv_events_config_true_covers_aggregated_mode() -> None:
    # Regression test: kv_events_config=True's "global bool" shortcut only
    # checked mode in ("prefill", "decode"), silently dropping --kv-events-config
    # for aggregated-mode topologies (mode == "agg") -- confirmed live: an
    # aggregated ThunderAgent deployment with kv_events_config: true never
    # got --kv-events-config in its launched sglang.launch_server command,
    # leaving dynamo.sglang.sidecar's kv_event_sources permanently at 0 and
    # every routed request's cache-overlap score stuck at 0.00.
    process = _process(mode="agg", kv_events_port=5557)
    backend = SGLangProtocol(
        kv_events_config=True,
        sglang_config=SGLangServerConfig(aggregated={"tensor-parallel-size": 8}),
    )

    with patch("srtctl.core.slurm.get_hostname_ip", return_value="10.0.0.1"):
        command = backend.build_worker_command(process, [process], _runtime())

    leader_script = command[2]
    assert "--kv-events-config" in leader_script
    # The script quotes the JSON arg with single quotes (bash -lc style);
    # pull out what's between the first pair following the flag.
    after_flag = leader_script.split("--kv-events-config ", 1)[1]
    raw_json = after_flag.split("'", 2)[1]
    kv_config = json.loads(raw_json)
    assert kv_config["endpoint"] == "tcp://*:5557"
    assert kv_config["publisher"] == "zmq"


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
