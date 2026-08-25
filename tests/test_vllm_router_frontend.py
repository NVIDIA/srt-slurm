# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""High-signal tests for official vLLM Router orchestration."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from marshmallow import ValidationError

from srtctl.backends import VLLMProtocol, VLLMServerConfig
from srtctl.cli.mixins.benchmark_stage import _get_health_expectations
from srtctl.core.schema import FrontendConfig, ResourceConfig, SrtConfig
from srtctl.core.topology import Endpoint, NodePortAllocator, Process
from srtctl.frontends import VLLMRouterFrontend, get_frontend
from srtctl.frontends.static_router import RouterWorker
from srtctl.frontends.vllm_router import node_local_data_parallel_size, routed_process_dp_size


def _runtime() -> SimpleNamespace:
    return SimpleNamespace(
        model_path=Path("/model"),
        is_hf_model=False,
        frontend_port=8000,
        network_interface="ib0",
        request_plane="nats",
    )


def test_get_frontend_exposes_official_vllm_router() -> None:
    assert isinstance(get_frontend("vllm-router"), VLLMRouterFrontend)


@pytest.mark.parametrize("frontend_type", ["sglang", "vllm-router"])
def test_static_router_aggregate_command_advertises_all_bases(frontend_type: str) -> None:
    frontend = get_frontend(frontend_type)
    command = frontend.build_router_command(
        [
            RouterWorker("agg", "http://10.0.0.1:6100"),
            RouterWorker("agg", "http://10.0.0.2:6100"),
        ],
        "0.0.0.0",
        8000,
    )

    assert command[command.index("--worker-urls") + 1 : -4] == [
        "http://10.0.0.1:6100",
        "http://10.0.0.2:6100",
    ]
    assert command[-4:] == ["--host", "0.0.0.0", "--port", "8000"]


def test_vllm_router_pd_command_uses_nixl_bootstrap_ports() -> None:
    frontend = VLLMRouterFrontend()
    command = frontend.build_router_command(
        [
            RouterWorker("prefill", "http://10.0.0.1:6100", 5400),
            RouterWorker("decode", "http://10.0.0.2:6100"),
        ],
        "0.0.0.0",
        8000,
    )

    assert command[:2] == ["vllm-router", "--vllm-pd-disaggregation"]
    assert command[command.index("--prefill") + 1 : command.index("--decode")] == [
        "http://10.0.0.1:6100",
        "5400",
    ]


def test_collect_workers_uses_positive_http_ports_and_configured_interface() -> None:
    frontend = VLLMRouterFrontend()
    processes = [
        SimpleNamespace(endpoint_mode="prefill", node="p0", http_port=6100, nixl_port=5400),
        SimpleNamespace(endpoint_mode="prefill", node="p1", http_port=0, nixl_port=5400),
        SimpleNamespace(endpoint_mode="decode", node="d0", http_port=6100, nixl_port=5500),
    ]

    with patch("srtctl.frontends.static_router.get_hostname_ip", side_effect=["10.0.0.1", "10.0.0.2"]) as resolve:
        workers = frontend.collect_workers(MagicMock(), processes, "ib0")

    assert workers == [
        RouterWorker("prefill", "http://10.0.0.1:6100", 5400),
        RouterWorker("decode", "http://10.0.0.2:6100", 5500),
    ]
    assert [mock_call.args for mock_call in resolve.call_args_list] == [("p0", "ib0"), ("d0", "ib0")]


def test_dep4_expansion_and_health_counts_follow_upstream_per_node_topology() -> None:
    backend = VLLMProtocol(
        vllm_config=VLLMServerConfig(
            prefill={"data-parallel-size": 4},
            decode={"data-parallel-size": 4},
        )
    )
    processes = [
        Process("p0", frozenset(range(4)), 7500, 6100, "prefill", 0, nixl_port=5400),
        Process("d0", frozenset(range(4)), 7501, 6100, "decode", 0, nixl_port=5500),
        Process("d1", frozenset(range(4)), 7502, 6100, "decode", 1, nixl_port=5504),
    ]
    config = SimpleNamespace(
        frontend=SimpleNamespace(type="vllm-router"),
        backend=backend,
        resources=SimpleNamespace(num_prefill=1, num_decode=2, num_agg=0),
    )

    assert node_local_data_parallel_size(backend, processes) == 4
    assert _get_health_expectations(config, processes) == (
        4,
        8,
        "4P + 8D Router workers; logical workers: 1P + 2D",
        12,
    )


def test_cross_node_model_parallel_base_is_not_dp_expanded() -> None:
    backend = VLLMProtocol(
        vllm_config=VLLMServerConfig(aggregated={"data-parallel-size": 2, "tensor-parallel-size": 8})
    )
    process = Process("n0", frozenset(range(4)), 7500, 6100, "agg", 0)

    assert routed_process_dp_size(backend, process) == 1


def test_managed_args_derive_dp_and_startup_timeout_without_overriding_user_policy() -> None:
    frontend = VLLMRouterFrontend()
    backend = VLLMProtocol(vllm_config=VLLMServerConfig(aggregated={"data-parallel-size": 4}))
    processes = [Process("n0", frozenset(range(4)), 7500, 6100, "agg", 0)]
    config = SimpleNamespace(
        frontend=SimpleNamespace(args={"policy": "consistent_hash"}),
        health_check=SimpleNamespace(max_attempts=360, interval_seconds=10),
    )

    assert frontend.get_managed_frontend_args(config, backend, processes) == [
        "--intra-node-data-parallel-size",
        "4",
        "--worker-startup-timeout-secs",
        "3600",
    ]


def test_router_launch_uses_router_image_env_setup_and_captured_log() -> None:
    frontend = VLLMRouterFrontend()
    runtime = SimpleNamespace(
        log_dir=Path("/logs"),
        container_image=Path("/worker.sqsh"),
        container_mounts={"/host": "/container"},
        environment={"GLOBAL": "value", "ROUTER_LOG": "info"},
        network_interface="ib0",
        nodes=SimpleNamespace(het_group_for=lambda _node: 1),
    )
    config = SimpleNamespace(
        backend=SimpleNamespace(type="vllm"),
        health_check=SimpleNamespace(max_attempts=360, interval_seconds=10),
        frontend=SimpleNamespace(
            args={"policy": "consistent_hash"},
            env={"ROUTER_LOG": "debug"},
            container_image="docker://router:test",
        ),
        setup_script="router-deps.sh",
    )
    topology = SimpleNamespace(frontend_nodes=["router0"], frontend_port=8000)
    worker = Process("worker0", frozenset(range(4)), 7500, 6100, "agg", 0)
    backend = MagicMock()
    backend._get_dp_size.return_value = None

    with (
        patch("srtctl.frontends.static_router.get_hostname_ip", return_value="10.0.0.1"),
        patch.object(frontend, "start_process", return_value=MagicMock()) as start,
    ):
        managed = frontend.start_frontends(topology, runtime, config, backend, [worker])

    kwargs = start.call_args.kwargs
    assert kwargs["container_image"] == "docker://router:test"
    assert kwargs["env_to_set"] == {"GLOBAL": "value", "ROUTER_LOG": "debug"}
    assert kwargs["output"] == "/logs/router0_vllm-router_0.out"
    assert "/configs/${setup_script}" in kwargs["bash_preamble"]
    assert managed[0].log_file == Path("/logs/router0_vllm-router_0.out")


def test_schema_rejects_backend_mismatch_and_deprecated_per_gpu_dp() -> None:
    from srtctl.backends import SGLangProtocol

    common = {
        "name": "router",
        "model": {"path": "model", "container": "image", "precision": "fp8"},
        "resources": ResourceConfig(gpu_type="h100", gpus_per_node=8, agg_nodes=1, agg_workers=1),
        "frontend": FrontendConfig(type="vllm-router", enable_multiple_frontends=False),
    }
    with pytest.raises(ValidationError, match="requires backend.type: vllm"):
        SrtConfig(**common, backend=SGLangProtocol())
    with pytest.raises(ValidationError, match="requires backend.dp_launch_mode: per_node"):
        SrtConfig(
            **common,
            backend=VLLMProtocol(
                dp_launch_mode="per_gpu",
                vllm_config=VLLMServerConfig(aggregated={"data-parallel-size": 8}),
            ),
        )


def test_vllm_router_preserves_upstream_multinode_tp_serve() -> None:
    backend = VLLMProtocol(vllm_config=VLLMServerConfig(aggregated={"tensor-parallel-size": 8}))
    endpoint = Endpoint("agg", 0, ("n0", "n1"), frozenset(range(4)), gpus_per_node=4)
    processes = backend.endpoints_to_processes(
        [endpoint], port_allocator=NodePortAllocator(), frontend_type="vllm-router"
    )

    with patch("srtctl.core.slurm.get_hostname_ip", return_value="10.0.0.1"):
        leader = backend.build_worker_command(processes[0], processes, _runtime(), frontend_type="vllm-router")
        follower = backend.build_worker_command(processes[1], processes, _runtime(), frontend_type="vllm-router")

    assert leader[:3] == ["vllm", "serve", "/model"]
    assert leader[leader.index("--port") + 1] == "6100"
    assert leader[leader.index("--nnodes") + 1] == "2"
    assert "--headless" not in leader
    assert "--headless" in follower
    assert "--port" not in follower


def test_vllm_router_per_node_dep4_launches_one_api_per_node_pool() -> None:
    backend = VLLMProtocol(vllm_config=VLLMServerConfig(aggregated={"data-parallel-size": 8}))
    endpoint = Endpoint("agg", 0, ("n0", "n1"), frozenset(range(4)), gpus_per_node=4)
    processes = backend.endpoints_to_processes(
        [endpoint], port_allocator=NodePortAllocator(), frontend_type="vllm-router"
    )

    with patch("srtctl.core.slurm.get_hostname_ip", return_value="10.0.0.1"):
        commands = [
            backend.build_worker_command(process, processes, _runtime(), frontend_type="vllm-router")
            for process in processes
        ]

    assert [process.http_port for process in processes] == [6100, 6100]
    assert [command[command.index("--data-parallel-start-rank") + 1] for command in commands] == ["0", "4"]
    assert all("--data-parallel-size-local" in command for command in commands)
    assert all("--data-parallel-hybrid-lb" in command for command in commands)
    assert all("--headless" not in command for command in commands)


def test_single_node_dp_preserves_native_vllm_topology() -> None:
    """A single server owns local DP ranks; srt-slurm must not split it."""
    backend = VLLMProtocol(
        vllm_config=VLLMServerConfig(aggregated={"tensor-parallel-size": 2, "data-parallel-size": 2})
    )
    endpoint = Endpoint("agg", 0, ("n0",), frozenset(range(4)), gpus_per_node=4)
    processes = backend.endpoints_to_processes(
        [endpoint], port_allocator=NodePortAllocator(), frontend_type="vllm-router"
    )

    with patch("srtctl.core.slurm.get_hostname_ip", return_value="10.0.0.1"):
        command = backend.build_worker_command(processes[0], processes, _runtime(), frontend_type="vllm-router")

    assert len(processes) == 1
    assert "--data-parallel-size-local" not in command
    assert "--data-parallel-start-rank" not in command
    assert "--data-parallel-hybrid-lb" not in command
    assert command[command.index("--data-parallel-size") + 1] == "2"


def test_dp_size_one_is_not_a_distributed_launch_mode() -> None:
    backend = VLLMProtocol(vllm_config=VLLMServerConfig(aggregated={"data-parallel-size": 1}))

    assert backend.find_dp_modes() == []
    assert backend._is_dp_mode("agg") is False


def test_router_validates_pcp_as_part_of_the_vllm_world_size() -> None:
    common = {
        "name": "router-pcp",
        "model": {"path": "model", "container": "image", "precision": "fp8"},
        "resources": ResourceConfig(gpu_type="h100", gpus_per_node=8, agg_nodes=1, agg_workers=1),
        "frontend": FrontendConfig(type="vllm-router", enable_multiple_frontends=False),
    }

    with pytest.raises(ValidationError, match=r"DP\*TP\*PP\*PCP=2\*8=16 GPUs"):
        SrtConfig(
            **common,
            backend=VLLMProtocol(
                vllm_config=VLLMServerConfig(aggregated={"data-parallel-size": 2, "prefill-context-parallel-size": 8})
            ),
        )


def test_vllm_router_pd_worker_uses_direct_vllm_and_nixl_connector() -> None:
    backend = VLLMProtocol(vllm_config=VLLMServerConfig(prefill={"tensor-parallel-size": 4}))
    process = Process("p0", frozenset(range(4)), 7500, 6100, "prefill", 0, nixl_port=5400)

    with patch("srtctl.core.slurm.get_hostname_ip", return_value="10.0.0.1"):
        command = backend.build_worker_command(process, [process], _runtime(), frontend_type="vllm-router")

    assert command[:3] == ["vllm", "serve", "/model"]
    assert "--kv-transfer-config" in command
    assert "NixlConnector" in command[command.index("--kv-transfer-config") + 1]


def test_backend_readiness_requires_all_advertised_urls_in_one_pass() -> None:
    from srtctl.core.health import wait_for_http_endpoints

    ready = SimpleNamespace(status_code=200)
    unavailable = SimpleNamespace(status_code=503)
    with (
        patch("srtctl.core.health.requests.get", side_effect=[ready, unavailable, ready, ready]) as get,
        patch("srtctl.core.health.time.sleep"),
    ):
        assert wait_for_http_endpoints(["http://p/health", "http://d/health"], timeout=10)

    assert [mock_call.args[0] for mock_call in get.call_args_list] == [
        "http://p/health",
        "http://d/health",
        "http://p/health",
        "http://d/health",
    ]
