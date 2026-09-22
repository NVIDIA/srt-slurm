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

    leader_script = command[2]
    assert "--kv-events-config" in leader_script
    after_flag = leader_script.split("--kv-events-config ", 1)[1]
    kv_config = json.loads(after_flag.split("'", 2)[1])
    assert kv_config["endpoint"] == "tcp://*:5557"
    assert kv_config["publisher"] == "zmq"


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
    backend = VLLMProtocol(
        connector=None,
        vllm_config=VLLMServerConfig(
            aggregated={
                **parallelism,
                "api-server-count": 1,
                "master_addr": "stale-host",
                "node_rank": 7,
                "nnodes": 9,
                "master_port": 1234,
            }
        ),
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
        resources=SimpleNamespace(num_agg=1, num_prefill=0, num_decode=0),
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
