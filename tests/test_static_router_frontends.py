# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for native static-router frontend adapters."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from srtctl.frontends import SGLangFrontend, VLLMRouterFrontend, get_frontend
from srtctl.frontends.static_router import RouterWorker


def test_registry_exposes_native_router_names() -> None:
    assert isinstance(get_frontend("sglang"), SGLangFrontend)
    assert isinstance(get_frontend("vllm-router"), VLLMRouterFrontend)


@pytest.mark.parametrize("frontend", [SGLangFrontend(), VLLMRouterFrontend()])
def test_aggregate_command_advertises_all_logical_workers(frontend) -> None:
    command = frontend.build_router_command(
        [
            RouterWorker("agg", "http://10.0.0.1:30000"),
            RouterWorker("agg", "http://10.0.0.2:30000"),
        ],
        "0.0.0.0",
        8000,
    )

    assert command[-4:] == ["--host", "0.0.0.0", "--port", "8000"]
    worker_urls = command[command.index("--worker-urls") + 1 : -4]
    assert worker_urls == ["http://10.0.0.1:30000", "http://10.0.0.2:30000"]


@pytest.mark.parametrize(
    ("frontend", "pd_flag"),
    [
        (SGLangFrontend(), "--pd-disaggregation"),
        (VLLMRouterFrontend(), "--vllm-pd-disaggregation"),
    ],
)
def test_disaggregated_command_preserves_modes_and_bootstrap(frontend, pd_flag: str) -> None:
    command = frontend.build_router_command(
        [
            RouterWorker("prefill", "http://10.0.0.1:30000", 30001),
            RouterWorker("decode", "http://10.0.0.2:30000"),
        ],
        "0.0.0.0",
        8000,
    )

    assert pd_flag in command
    assert command[command.index("--prefill") + 1 : command.index("--decode")] == [
        "http://10.0.0.1:30000",
        "30001",
    ]
    assert command[command.index("--decode") + 1] == "http://10.0.0.2:30000"


def test_router_command_rejects_incomplete_or_mixed_topology() -> None:
    frontend = VLLMRouterFrontend()
    with pytest.raises(ValueError, match="requires prefill and decode"):
        frontend.build_router_command([RouterWorker("prefill", "http://p:1")], "0.0.0.0", 8000)
    with pytest.raises(ValueError, match="cannot mix"):
        frontend.build_router_command(
            [
                RouterWorker("agg", "http://a:1"),
                RouterWorker("prefill", "http://p:1"),
                RouterWorker("decode", "http://d:1"),
            ],
            "0.0.0.0",
            8000,
        )


def test_frontend_args_repeat_list_values() -> None:
    frontend = VLLMRouterFrontend()
    assert frontend.get_frontend_args_list({"routing-logic": ["round_robin", "session"]}) == [
        "--routing-logic",
        "round_robin",
        "--routing-logic",
        "session",
    ]


def test_vllm_router_advertises_nixl_side_channel_port() -> None:
    frontend = VLLMRouterFrontend()
    process = SimpleNamespace(
        is_leader=True,
        endpoint_mode="prefill",
        node="node1",
        http_port=30000,
        bootstrap_port=12000,
        nixl_port=13000,
    )

    with patch.object(frontend, "get_hostname_ip", return_value="10.0.0.1"):
        workers = frontend.collect_workers(MagicMock(), [process])

    assert workers == [RouterWorker("prefill", "http://10.0.0.1:30000", 13000)]


def test_vllm_router_health_gates_every_advertised_2p2d_logical_worker() -> None:
    frontend = VLLMRouterFrontend()
    backend = MagicMock()
    processes = [
        SimpleNamespace(endpoint_mode="prefill", node="p0", http_port=6100, nixl_port=5400),
        SimpleNamespace(endpoint_mode="prefill", node="p1", http_port=6100, nixl_port=5401),
        SimpleNamespace(endpoint_mode="decode", node="d0", http_port=6100, nixl_port=5500),
        SimpleNamespace(endpoint_mode="decode", node="d1", http_port=6100, nixl_port=5501),
        SimpleNamespace(endpoint_mode="decode", node="tp-follower", http_port=0, nixl_port=5502),
    ]

    with patch.object(frontend, "get_hostname_ip", side_effect=lambda node: f"ip-{node}"):
        urls = frontend.get_backend_health_urls(backend, processes)

    assert urls == [
        "http://ip-p0:6100/health",
        "http://ip-p1:6100/health",
        "http://ip-d0:6100/health",
        "http://ip-d1:6100/health",
    ]


def test_vllm_router_derives_dep4_expansion_for_1p2d() -> None:
    """One P URL and two D URLs are each expanded to four ranks by Router."""
    frontend = VLLMRouterFrontend()
    backend = MagicMock()
    backend._is_dp_mode.return_value = True
    backend._get_dp_size.return_value = 4
    processes = [
        SimpleNamespace(
            endpoint_mode="prefill",
            endpoint_index=0,
            node="prefill",
            gpu_indices=frozenset(range(4)),
            http_port=6100,
            nixl_port=5400,
            node_rank=0,
        ),
        SimpleNamespace(
            endpoint_mode="decode",
            endpoint_index=0,
            node="decode0",
            gpu_indices=frozenset(range(4)),
            http_port=6100,
            nixl_port=5500,
            node_rank=0,
        ),
        SimpleNamespace(
            endpoint_mode="decode",
            endpoint_index=1,
            node="decode1",
            gpu_indices=frozenset(range(4)),
            http_port=6100,
            nixl_port=5504,
            node_rank=0,
        ),
    ]
    config = SimpleNamespace(
        frontend=SimpleNamespace(args={}),
        health_check=SimpleNamespace(max_attempts=360, interval_seconds=10),
    )

    with patch.object(frontend, "get_hostname_ip", side_effect=lambda node: f"ip-{node}"):
        workers = frontend.collect_workers(backend, processes)
        command = frontend.build_router_command(workers, "0.0.0.0", 8000)

    assert len([worker for worker in workers if worker.mode == "prefill"]) == 1
    assert len([worker for worker in workers if worker.mode == "decode"]) == 2
    assert command.count("--prefill") == 1
    assert command.count("--decode") == 2
    assert frontend.get_managed_frontend_args(config, backend, processes) == [
        "--intra-node-data-parallel-size",
        "4",
        "--worker-startup-timeout-secs",
        "3600",
    ]


def test_vllm_router_launch_uses_router_container_env_and_only_leaders() -> None:
    frontend = VLLMRouterFrontend()
    runtime = SimpleNamespace(
        log_dir=Path("/logs"),
        container_image=Path("/worker.sqsh"),
        container_mounts={"/host": "/container"},
        environment={"GLOBAL": "value", "ROUTER_LOG": "info"},
        nodes=SimpleNamespace(het_group_for=lambda node: 1),
    )
    config = SimpleNamespace(
        backend=SimpleNamespace(type="vllm"),
        health_check=SimpleNamespace(max_attempts=360, interval_seconds=10),
        frontend=SimpleNamespace(
            args={"routing-logic": "session"},
            env={"ROUTER_LOG": "debug"},
            container_image="docker://router:test",
        ),
        setup_script="router-deps.sh",
    )
    topology = SimpleNamespace(frontend_nodes=["node0"], frontend_port=8180)
    workers = [
        SimpleNamespace(
            is_leader=True,
            endpoint_mode="agg",
            endpoint_index=0,
            node="node1",
            gpu_indices=frozenset(range(8)),
            http_port=30000,
            bootstrap_port=None,
            nixl_port=None,
        ),
        SimpleNamespace(
            is_leader=False,
            endpoint_mode="agg",
            endpoint_index=0,
            node="node2",
            gpu_indices=frozenset(range(8)),
            http_port=0,
            bootstrap_port=None,
            nixl_port=None,
        ),
    ]

    backend = MagicMock()
    backend._is_dp_mode.return_value = False
    backend._get_dp_size.return_value = None

    with (
        patch.object(frontend, "get_hostname_ip", return_value="10.0.0.1"),
        patch.object(frontend, "start_process", return_value=MagicMock()) as start,
    ):
        processes = frontend.start_frontends(topology, runtime, config, backend, workers)

    kwargs = start.call_args.kwargs
    assert kwargs["output"] == "/logs/node0_vllm-router_0.out"
    assert kwargs["container_image"] == "docker://router:test"
    assert kwargs["env_to_set"] == {"GLOBAL": "value", "ROUTER_LOG": "debug"}
    assert kwargs["het_group"] == 1
    assert "/configs/${setup_script}" in kwargs["bash_preamble"]
    assert kwargs["command"].count("http://10.0.0.1:30000") == 1
    assert "--routing-logic" in kwargs["command"]
    timeout_index = kwargs["command"].index("--worker-startup-timeout-secs")
    assert kwargs["command"][timeout_index + 1] == "3600"
    assert processes[0].log_file == Path("/logs/node0_vllm-router_0.out")


def test_vllm_router_explicit_worker_startup_timeout_overrides_managed_value() -> None:
    frontend = VLLMRouterFrontend()
    config = SimpleNamespace(
        health_check=SimpleNamespace(max_attempts=360, interval_seconds=10),
        frontend=SimpleNamespace(args={"worker-startup-timeout-secs": 7200}),
    )

    command = [
        *frontend.get_managed_frontend_args(config),
        *frontend.get_frontend_args_list(config.frontend.args),
    ]

    assert command == ["--worker-startup-timeout-secs", "7200"]


def test_router_rejects_backend_mismatch_before_launch() -> None:
    frontend = VLLMRouterFrontend()
    config = SimpleNamespace(
        backend=SimpleNamespace(type="sglang"),
        frontend=SimpleNamespace(args=None, env=None, container_image=None),
    )
    topology = SimpleNamespace(frontend_nodes=["node0"], frontend_port=8180)
    runtime = SimpleNamespace(log_dir=Path("/logs"), container_image=Path("/worker.sqsh"))

    with pytest.raises(ValueError, match="requires backend.type: vllm"):
        frontend.start_frontends(topology, runtime, config, MagicMock(), [])


def test_schema_rejects_router_backend_mismatch() -> None:
    from marshmallow import ValidationError

    from srtctl.backends import SGLangProtocol
    from srtctl.core.schema import FrontendConfig, ResourceConfig, SrtConfig

    with pytest.raises(ValidationError, match="vllm-router requires backend.type: vllm"):
        SrtConfig(
            name="bad-router-pair",
            model={"path": "model", "container": "image", "precision": "fp8"},
            resources=ResourceConfig(gpu_type="h100", gpus_per_node=8, agg_nodes=1, agg_workers=1),
            frontend=FrontendConfig(type="vllm-router", enable_multiple_frontends=False),
            backend=SGLangProtocol(),
        )


def test_vllm_router_accepts_many_single_node_endpoints() -> None:
    from srtctl.backends import VLLMProtocol, VLLMServerConfig
    from srtctl.core.schema import FrontendConfig, ResourceConfig, SrtConfig

    config = SrtConfig(
        name="multi-endpoint-router",
        model={"path": "model", "container": "image", "precision": "fp8"},
        resources=ResourceConfig(
            gpu_type="h100",
            gpus_per_node=8,
            agg_nodes=4,
            agg_workers=4,
        ),
        frontend=FrontendConfig(type="vllm-router", enable_multiple_frontends=False),
        backend=VLLMProtocol(vllm_config=VLLMServerConfig(aggregated={"tensor-parallel-size": 8})),
    )

    assert config.resources.gpus_per_agg == 8


def test_vllm_router_rejects_multinode_tp_only_endpoint() -> None:
    from marshmallow import ValidationError

    from srtctl.backends import VLLMProtocol
    from srtctl.core.schema import FrontendConfig, ResourceConfig, SrtConfig

    with pytest.raises(ValidationError, match="multi-node TP-only"):
        SrtConfig(
            name="multi-node-endpoint",
            model={"path": "model", "container": "image", "precision": "fp8"},
            resources=ResourceConfig(
                gpu_type="h100",
                gpus_per_node=8,
                prefill_nodes=2,
                prefill_workers=1,
                decode_nodes=1,
                decode_workers=1,
            ),
            frontend=FrontendConfig(type="vllm-router", enable_multiple_frontends=False),
            backend=VLLMProtocol(dp_launch_mode="per_node"),
        )


def test_vllm_router_accepts_multinode_dep8_endpoint() -> None:
    from srtctl.backends import VLLMProtocol, VLLMServerConfig
    from srtctl.core.schema import FrontendConfig, ResourceConfig, SrtConfig

    config = SrtConfig(
        name="multi-node-dep8",
        model={"path": "model", "container": "image", "precision": "fp8"},
        resources=ResourceConfig(
            gpu_type="gb200",
            gpus_per_node=4,
            agg_nodes=2,
            agg_workers=1,
        ),
        frontend=FrontendConfig(type="vllm-router", enable_multiple_frontends=False),
        backend=VLLMProtocol(
            dp_launch_mode="per_node",
            vllm_config=VLLMServerConfig(
                aggregated={"data-parallel-size": 8, "enable-expert-parallel": True},
            ),
        ),
    )

    assert config.resources.gpus_per_agg == 8


def test_vllm_router_accepts_native_single_node_tp2_dp2() -> None:
    """A single vllm serve owns its complete native TP x DP topology."""
    from srtctl.backends import VLLMProtocol, VLLMServerConfig
    from srtctl.core.schema import FrontendConfig, ResourceConfig, SrtConfig

    config = SrtConfig(
        name="native-tp2-dp2",
        model={"path": "model", "container": "image", "precision": "fp8"},
        resources=ResourceConfig(
            gpu_type="gb200",
            gpus_per_node=4,
            agg_nodes=1,
            agg_workers=1,
        ),
        frontend=FrontendConfig(type="vllm-router", enable_multiple_frontends=False),
        backend=VLLMProtocol(
            vllm_config=VLLMServerConfig(
                aggregated={
                    "tensor-parallel-size": 2,
                    "data-parallel-size": 2,
                    "enable-expert-parallel": True,
                }
            ),
        ),
    )

    assert config.backend._get_model_parallel_size("agg") == 2
    assert config.backend._get_local_dp_size("agg", 4) == 2


def test_vllm_router_rejects_parallelism_allocation_mismatch() -> None:
    """Reject a recipe before Slurm when vLLM cannot consume its GPU allocation."""
    from marshmallow import ValidationError

    from srtctl.backends import VLLMProtocol, VLLMServerConfig
    from srtctl.core.schema import FrontendConfig, ResourceConfig, SrtConfig

    with pytest.raises(ValidationError, match=r"DP\*TP\*PP\*PCP=2\*2=4 GPUs.*allocate 8 GPUs"):
        SrtConfig(
            name="invalid-native-world",
            model={"path": "model", "container": "image", "precision": "fp8"},
            resources=ResourceConfig(
                gpu_type="h100",
                gpus_per_node=8,
                agg_nodes=1,
                agg_workers=1,
            ),
            frontend=FrontendConfig(type="vllm-router", enable_multiple_frontends=False),
            backend=VLLMProtocol(
                vllm_config=VLLMServerConfig(aggregated={"tensor-parallel-size": 2, "data-parallel-size": 2})
            ),
        )


def test_direct_vllm_validates_native_parallelism_allocation() -> None:
    """Direct vLLM uses the same native world-size invariant as Router workers."""
    from marshmallow import ValidationError

    from srtctl.backends import VLLMProtocol, VLLMServerConfig
    from srtctl.core.schema import FrontendConfig, ResourceConfig, SrtConfig

    valid = SrtConfig(
        name="direct-native-tp2-dp2",
        model={"path": "model", "container": "image", "precision": "fp8"},
        resources=ResourceConfig(gpu_type="gb200", gpus_per_node=4, agg_nodes=1, agg_workers=1),
        frontend=FrontendConfig(type="vllm", enable_multiple_frontends=False),
        backend=VLLMProtocol(
            vllm_config=VLLMServerConfig(aggregated={"tensor-parallel-size": 2, "data-parallel-size": 2})
        ),
    )
    assert valid.resources.gpus_per_agg == 4

    with pytest.raises(ValidationError, match=r"direct vLLM parallelism requires.*4 GPUs.*allocate 8 GPUs"):
        SrtConfig(
            name="invalid-direct-native-world",
            model={"path": "model", "container": "image", "precision": "fp8"},
            resources=ResourceConfig(gpu_type="h100", gpus_per_node=8, agg_nodes=1, agg_workers=1),
            frontend=FrontendConfig(type="vllm", enable_multiple_frontends=False),
            backend=VLLMProtocol(
                vllm_config=VLLMServerConfig(aggregated={"tensor-parallel-size": 2, "data-parallel-size": 2})
            ),
        )


def test_vllm_router_accepts_multinode_tp2_dp4_endpoint() -> None:
    """Hybrid mode derives two local DP replicas per four-GPU node."""
    from srtctl.backends import VLLMProtocol, VLLMServerConfig
    from srtctl.core.schema import FrontendConfig, ResourceConfig, SrtConfig

    config = SrtConfig(
        name="multi-node-tp2-dp4",
        model={"path": "model", "container": "image", "precision": "fp8"},
        resources=ResourceConfig(
            gpu_type="gb200",
            gpus_per_node=4,
            agg_nodes=2,
            agg_workers=1,
        ),
        frontend=FrontendConfig(type="vllm-router", enable_multiple_frontends=False),
        backend=VLLMProtocol(
            dp_launch_mode="per_node",
            vllm_config=VLLMServerConfig(
                aggregated={
                    "tensor-parallel-size": 2,
                    "data-parallel-size": 4,
                    "enable-expert-parallel": True,
                }
            ),
        ),
    )

    assert config.backend._get_local_dp_size("agg", 4) == 2


def test_vllm_router_rejects_different_local_dp_sizes_across_pd_pools() -> None:
    """Router has one DP expansion factor, so every advertised pool must agree."""
    from marshmallow import ValidationError

    from srtctl.backends import VLLMProtocol, VLLMServerConfig
    from srtctl.core.schema import FrontendConfig, ResourceConfig, SrtConfig

    with pytest.raises(ValidationError, match="same node-local data-parallel size.*prefill=4, decode=2"):
        SrtConfig(
            name="mismatched-pd-local-dp",
            model={"path": "model", "container": "image", "precision": "fp8"},
            resources=ResourceConfig(
                gpu_type="gb200",
                gpus_per_node=4,
                prefill_nodes=1,
                prefill_workers=1,
                decode_nodes=1,
                decode_workers=1,
            ),
            frontend=FrontendConfig(type="vllm-router", enable_multiple_frontends=False),
            backend=VLLMProtocol(
                vllm_config=VLLMServerConfig(
                    prefill={"tensor-parallel-size": 1, "data-parallel-size": 4},
                    decode={"tensor-parallel-size": 2, "data-parallel-size": 2},
                )
            ),
        )


def test_sgl_router_rejects_non_divisible_tp_dp_layout() -> None:
    from marshmallow import ValidationError

    from srtctl.backends import SGLangProtocol, SGLangServerConfig
    from srtctl.core.schema import FrontendConfig, ResourceConfig, SrtConfig

    with pytest.raises(ValidationError, match="tp-size=1 must be divisible by dp-size=8"):
        SrtConfig(
            name="invalid-sglang-dpa",
            model={"path": "model", "container": "image", "precision": "fp8"},
            resources=ResourceConfig(gpu_type="h100", gpus_per_node=8, agg_nodes=1, agg_workers=1),
            frontend=FrontendConfig(type="sglang", enable_multiple_frontends=False),
            backend=SGLangProtocol(
                sglang_config=SGLangServerConfig(aggregated={"tp-size": 1, "dp-size": 8, "enable-dp-attention": True})
            ),
        )


def test_sgl_router_accepts_divisible_tp_dp_layout() -> None:
    from srtctl.backends import SGLangProtocol, SGLangServerConfig
    from srtctl.core.schema import FrontendConfig, ResourceConfig, SrtConfig

    config = SrtConfig(
        name="valid-sglang-dpa",
        model={"path": "model", "container": "image", "precision": "fp8"},
        resources=ResourceConfig(gpu_type="h100", gpus_per_node=8, agg_nodes=1, agg_workers=1),
        frontend=FrontendConfig(type="sglang", enable_multiple_frontends=False),
        backend=SGLangProtocol(
            sglang_config=SGLangServerConfig(aggregated={"tp-size": 8, "dp-size": 8, "enable-dp-attention": True})
        ),
    )

    assert config.backend.sglang_config.aggregated["tp-size"] == 8
