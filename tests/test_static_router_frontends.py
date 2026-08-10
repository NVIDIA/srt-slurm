# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for native static-router frontend adapters."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from srtctl.frontends import SGLRouterFrontend, VLLMRouterFrontend, get_frontend
from srtctl.frontends.static_router import RouterWorker


def test_registry_exposes_explicit_router_names_and_legacy_alias() -> None:
    assert isinstance(get_frontend("sgl-router"), SGLRouterFrontend)
    assert get_frontend("sglang").type == "sglang"
    assert isinstance(get_frontend("vllm-router"), VLLMRouterFrontend)


@pytest.mark.parametrize("frontend", [SGLRouterFrontend(), VLLMRouterFrontend()])
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
        (SGLRouterFrontend(), "--pd-disaggregation"),
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
        frontend=SimpleNamespace(
            args={"routing-logic": "session"},
            env={"ROUTER_LOG": "debug"},
            container_image="docker://router:test",
        ),
    )
    topology = SimpleNamespace(frontend_nodes=["node0"], frontend_port=8180)
    workers = [
        SimpleNamespace(
            is_leader=True,
            endpoint_mode="agg",
            node="node1",
            http_port=30000,
            bootstrap_port=None,
        ),
        SimpleNamespace(
            is_leader=False,
            endpoint_mode="agg",
            node="node2",
            http_port=0,
            bootstrap_port=None,
        ),
    ]

    with (
        patch.object(frontend, "get_hostname_ip", return_value="10.0.0.1"),
        patch.object(frontend, "start_process", return_value=MagicMock()) as start,
    ):
        frontend.start_frontends(topology, runtime, config, MagicMock(), workers)

    kwargs = start.call_args.kwargs
    assert kwargs["container_image"] == "docker://router:test"
    assert kwargs["env_to_set"] == {"GLOBAL": "value", "ROUTER_LOG": "debug"}
    assert kwargs["het_group"] == 1
    assert kwargs["command"].count("http://10.0.0.1:30000") == 1
    assert "--routing-logic" in kwargs["command"]


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
    from srtctl.backends import VLLMProtocol
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
        backend=VLLMProtocol(),
    )

    assert config.resources.gpus_per_agg == 8


def test_vllm_router_rejects_endpoint_spanning_nodes() -> None:
    from marshmallow import ValidationError

    from srtctl.backends import VLLMProtocol
    from srtctl.core.schema import FrontendConfig, ResourceConfig, SrtConfig

    with pytest.raises(ValidationError, match="each logical vLLM endpoint"):
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
            backend=VLLMProtocol(),
        )
