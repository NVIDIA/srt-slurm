# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Where a Dynamo sidecar job's metrics are served and scraped, for every sidecar engine.

Each worker process has up to two Tachometer targets: the sidecar's Dynamo runtime
endpoint (``backend_{mode}{index}_rank{rank}``, on the system port, only where a
sidecar runs) and the native engine's own Prometheus endpoint
(``backend_{mode}{index}_rank{rank}_engine``). See docs/sidecars.md#metrics.
"""

from __future__ import annotations

import os
import re
import shlex
import tempfile
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from srtctl.backends import SGLangBackend, TRTLLMBackend, VLLMBackend
from srtctl.core.schema import DynamoConfig, RoleConfig
from srtctl.core.telemetry import sidecar_worker_metrics_targets
from srtctl.core.topology import Endpoint, NodePortAllocator, Process
from srtctl.frontends import get_frontend
from tests.launch_snapshots import EXAMPLES_DIR, _isolated_cluster

DYNAMO = get_frontend("dynamo")


def _endpoint(mode: str, node_count: int, gpus_per_node: int, index: int = 0) -> Endpoint:
    return Endpoint(
        mode=mode,
        index=index,
        nodes=tuple(f"node{rank}" for rank in range(node_count)),
        gpu_indices=frozenset(range(gpus_per_node)),
        gpus_per_node=gpus_per_node,
    )


def _runtime() -> MagicMock:
    runtime = MagicMock()
    runtime.model_path = Path("/models/example-model")
    runtime.worker_model_arg = "/model"
    runtime.is_hf_model = False
    runtime.hf_snapshot_path = None
    runtime.gpu_type = "h100"
    runtime.log_dir = Path(tempfile.gettempdir())
    runtime.network_interface = None
    runtime.dynamo = DynamoConfig(sidecar=True)
    return runtime


def _targets(backend, processes: list[Process]) -> list[tuple[str, str, int, str]]:
    return [
        (target.endpoint_name, target.process.node, target.port, target.path)
        for target in sidecar_worker_metrics_targets(processes, backend, DYNAMO)
    ]


def _engine_line(command: list[str], marker: str) -> list[str]:
    if command[:2] != ["bash", "-lc"]:
        return command
    return shlex.split(next(line for line in command[2].splitlines() if marker in line).removesuffix(" &"))


# --- vLLM -------------------------------------------------------------------------------------


def test_vllm_single_node_sidecar_serves_engine_metrics_on_every_interface() -> None:
    backend = VLLMBackend(connector=None, roles={"agg": RoleConfig(args={"tensor-parallel-size": 1})})
    processes = backend.endpoints_to_processes([_endpoint("agg", 1, 1)], dynamo_sidecar=True)
    (process,) = processes

    with patch("srtctl.core.slurm.get_hostname_ip", return_value="10.0.0.1"):
        command = backend.build_worker_command(process, processes, _runtime())

    engine = _engine_line(command, "vllm-rs serve")
    # Bound to every interface so Tachometer can scrape it; the port is the allocator's.
    assert engine[engine.index("--host") + 1] == "0.0.0.0"
    assert engine[engine.index("--port") + 1] == str(process.http_port)
    assert process.http_port > 0
    assert _targets(backend, processes) == [
        ("backend_agg0_rank0", "node0", process.sys_port, "/metrics"),
        ("backend_agg0_rank0_engine", "node0", process.http_port, "/metrics"),
    ]


def test_vllm_sidecar_without_an_allocated_http_port_fails_instead_of_deriving_one() -> None:
    backend = VLLMBackend(connector=None, roles={"agg": RoleConfig(args={"tensor-parallel-size": 1})})
    (process,) = backend.endpoints_to_processes([_endpoint("agg", 1, 1)], dynamo_sidecar=True)
    unallocated = replace(process, http_port=0)

    assert backend.sidecar_engine_metrics_port(unallocated, [unallocated]) is None
    with (
        patch("srtctl.core.slurm.get_hostname_ip", return_value="10.0.0.1"),
        pytest.raises(ValueError, match="no allocated HTTP port"),
    ):
        backend.build_worker_command(unallocated, [unallocated], _runtime())


def test_vllm_multi_node_replica_scrapes_only_the_leader() -> None:
    backend = VLLMBackend(connector=None, roles={"agg": RoleConfig(args={"tensor-parallel-size": 8})})
    leader, follower = backend.endpoints_to_processes([_endpoint("agg", 2, 4)], dynamo_sidecar=True)

    # Headless followers run neither a sidecar nor a frontend.
    assert backend.runs_dynamo_sidecar(follower, [leader, follower]) is False
    assert backend.sidecar_engine_metrics_port(follower, [leader, follower]) is None
    assert _targets(backend, [leader, follower]) == [
        ("backend_agg0_rank0", "node0", leader.sys_port, "/metrics"),
        ("backend_agg0_rank0_engine", "node0", leader.http_port, "/metrics"),
    ]


def test_vllm_hybrid_dp_scrapes_every_nodes_sidecar_and_frontend() -> None:
    backend = VLLMBackend(
        connector=None,
        roles={"decode": RoleConfig(args={"data-parallel-size": 8, "enable-expert-parallel": True})},
    )
    processes = backend.endpoints_to_processes([_endpoint("decode", 2, 4)], dynamo_sidecar=True)

    with patch("srtctl.core.slurm.get_hostname_ip", return_value="10.0.0.1"):
        commands = [backend.build_worker_command(process, processes, _runtime()) for process in processes]

    for process, command in zip(processes, commands, strict=True):
        engine = _engine_line(command, "vllm.entrypoints.cli.main serve")
        assert engine[engine.index("--host") + 1] == "0.0.0.0"
        assert engine[engine.index("--port") + 1] == str(process.http_port)
    first, second = processes
    # node_rank is the node's first global DP rank.
    assert _targets(backend, processes) == [
        ("backend_decode0_rank0", "node0", first.sys_port, "/metrics"),
        ("backend_decode0_rank0_engine", "node0", first.http_port, "/metrics"),
        ("backend_decode0_rank4", "node1", second.sys_port, "/metrics"),
        ("backend_decode0_rank4_engine", "node1", second.http_port, "/metrics"),
    ]


# --- SGLang -----------------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["agg", "prefill", "decode"])
def test_sglang_sidecar_engine_enables_metrics_on_every_node(mode: str) -> None:
    backend = SGLangBackend(roles={mode: RoleConfig(args={"tensor-parallel-size": 8})})
    processes = backend.endpoints_to_processes([_endpoint(mode, 2, 4)], dynamo_sidecar=True)

    with patch("srtctl.core.slurm.get_hostname_ip", return_value="10.0.0.1"):
        commands = [backend.build_worker_command(process, processes, _runtime()) for process in processes]

    leader, follower = processes
    # The follower's health server (/health, /metrics) binds an allocated port, not port 0.
    assert follower.http_port > 0
    for process, command in zip(processes, commands, strict=True):
        engine = _engine_line(command, "sglang.launch_server")
        assert engine[engine.index("--port") + 1] == str(process.http_port)
        assert engine.count("--enable-metrics") == 1
    # A TP-only follower runs no sidecar, so its system port is not a target.
    assert _targets(backend, processes) == [
        (f"backend_{mode}0_rank0", "node0", leader.sys_port, "/metrics"),
        (f"backend_{mode}0_rank0_engine", "node0", leader.http_port, "/metrics"),
        (f"backend_{mode}0_rank1_engine", "node1", follower.http_port, "/metrics"),
    ]


def test_sglang_sidecar_respects_an_explicit_enable_metrics_setting() -> None:
    backend = SGLangBackend(roles={"agg": RoleConfig(args={"tensor-parallel-size": 1, "enable-metrics": False})})
    processes = backend.endpoints_to_processes([_endpoint("agg", 1, 1)], dynamo_sidecar=True)

    with patch("srtctl.core.slurm.get_hostname_ip", return_value="10.0.0.1"):
        command = backend.build_worker_command(processes[0], processes, _runtime())

    assert "--enable-metrics" not in _engine_line(command, "sglang.launch_server")
    assert _targets(backend, processes) == [("backend_agg0_rank0", "node0", processes[0].sys_port, "/metrics")]


def test_sglang_attention_dp_relay_followers_expose_both_endpoints() -> None:
    backend = SGLangBackend(
        roles={"agg": RoleConfig(args={"tp-size": 8, "dp-size": 2, "enable-dp-attention": True}, kv_events=True)}
    )
    processes = backend.endpoints_to_processes([_endpoint("agg", 4, 2)], dynamo_sidecar=True)

    names = [name for name, *_ in _targets(backend, processes)]

    # Nodes 0 and 2 own an attention-DP group's first rank and publish KV events.
    assert names == [
        "backend_agg0_rank0",
        "backend_agg0_rank0_engine",
        "backend_agg0_rank1_engine",
        "backend_agg0_rank2",
        "backend_agg0_rank2_engine",
        "backend_agg0_rank3_engine",
    ]


def test_sglang_colocated_follower_http_ports_do_not_collide() -> None:
    backend = SGLangBackend(roles={"prefill": RoleConfig(args={"tensor-parallel-size": 8})})
    allocator = NodePortAllocator()
    processes = backend.endpoints_to_processes(
        [_endpoint("prefill", 2, 4, index=0), _endpoint("prefill", 2, 4, index=1)],
        port_allocator=allocator,
        dynamo_sidecar=True,
    )

    ports = [(process.node, process.http_port) for process in processes]
    assert len(set(ports)) == len(ports)


def test_sglang_without_sidecars_keeps_followers_off_the_http_allocator() -> None:
    backend = SGLangBackend(roles={"agg": RoleConfig(args={"tensor-parallel-size": 8})})
    _, follower = backend.endpoints_to_processes([_endpoint("agg", 2, 4)])

    assert follower.http_port == 0


# --- TRT-LLM ----------------------------------------------------------------------------------


@pytest.mark.parametrize("node_count", [1, 2])
def test_trtllm_sidecar_scrapes_only_the_rank_zero_runtime(node_count: int) -> None:
    backend = TRTLLMBackend(roles={"decode": RoleConfig(args={"tensor_parallel_size": 4 * node_count})})
    processes = backend.endpoints_to_processes([_endpoint("decode", node_count, 4)], dynamo_sidecar=True)

    # OpenEngine gRPC serves no Prometheus endpoint, so there is no engine target.
    assert _targets(backend, processes) == [("backend_decode0_rank0", "node0", processes[0].sys_port, "/metrics")]


# --- Example recipes through the mock orchestrator ----------------------------------------------


_ENDPOINT_FIELD = re.compile(r'^(name|url|"hostname") = "(.*)"$', re.MULTILINE)


def _mock_worker_targets(recipe: Path) -> list[tuple[str, str, str]]:
    """``(name, hostname, port+path)`` of every backend Tachometer target the mock orchestrator writes."""
    from srtctl.mock import MockOptions, run_mock_sweep

    with tempfile.TemporaryDirectory(prefix="srt-sidecar-metrics-") as tmp:
        tmp_path = Path(tmp).resolve()
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        container_file = tmp_path / "container.sqsh"
        container_file.touch()
        output_dir = tmp_path / "output"
        prior_cwd = Path.cwd()
        os.chdir(tmp_path)
        try:
            with (
                _isolated_cluster(model_dir, container_file),
                patch(
                    "srtctl.core.power.session.PowerTelemetrySession.start_and_wait_for_readiness", return_value=True
                ),
            ):
                options = MockOptions(
                    child_duration_s=0.0,
                    phase_pause_s=0.0,
                    nodelist=tuple(f"node-{i:02d}" for i in range(1, 8)),
                    end_manual_hold=True,
                )
                exit_code = run_mock_sweep(config_path=recipe, output_dir=output_dir, job_id="4242", options=options)
        finally:
            os.chdir(prior_cwd)
        assert exit_code == 0
        text = (output_dir / "logs" / "tachometer_config.toml").read_text()

    targets = []
    for block in text.split("[[endpoints]]")[1:]:
        fields = dict(_ENDPOINT_FIELD.findall(block))
        if fields["name"].startswith("backend_"):
            targets.append((fields["name"], fields['"hostname"'], fields["url"].split(":", 2)[2]))
    return targets


@pytest.mark.parametrize(
    ("recipe", "expected"),
    [
        (
            "vllm/dynamo-sidecar-agg.yaml",
            [
                ("backend_agg0_rank0", "node-01", "7500/metrics"),
                ("backend_agg0_rank0_engine", "node-01", "6100/metrics"),
            ],
        ),
        (
            "features/sglang-sidecar-multinode-dp.yaml",
            [
                ("backend_agg0_rank0", "node-01", "7500/metrics"),
                ("backend_agg0_rank0_engine", "node-01", "6100/metrics"),
                # Node 1 relays its KV events through a telemetry sidecar.
                ("backend_agg0_rank1", "node-02", "7501/metrics"),
                ("backend_agg0_rank1_engine", "node-02", "6100/metrics"),
            ],
        ),
        (
            "features/sglang-sidecar-multinode-disagg-dp.yaml",
            [
                ("backend_decode0_rank0", "node-03", "7502/metrics"),
                ("backend_decode0_rank0_engine", "node-03", "6100/metrics"),
                # Decode publishes no KV events: its follower runs the engine only.
                ("backend_decode0_rank1_engine", "node-04", "6100/metrics"),
                ("backend_prefill0_rank0", "node-01", "7500/metrics"),
                ("backend_prefill0_rank0_engine", "node-01", "6100/metrics"),
                ("backend_prefill0_rank1", "node-02", "7501/metrics"),
                ("backend_prefill0_rank1_engine", "node-02", "6100/metrics"),
            ],
        ),
        (
            "trtllm/dynamo-sidecar-disagg.yaml",
            [
                # One sidecar per endpoint, on rank zero; the decode follower node serves nothing.
                ("backend_decode0_rank0", "node-02", "7501/metrics"),
                ("backend_prefill0_rank0", "node-01", "7500/metrics"),
            ],
        ),
    ],
)
def test_example_sidecar_recipes_scrape_every_live_endpoint(recipe: str, expected: list[tuple[str, str, str]]) -> None:
    assert _mock_worker_targets(EXAMPLES_DIR / recipe) == expected
