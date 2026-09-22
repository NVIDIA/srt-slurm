# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for frontend implementations."""

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import yaml

from srtctl.core.schema import ObservabilityConfig
from srtctl.frontends import (
    DynamoFrontend,
    SGLangFrontend,
    SGLangRouterFrontend,
    VLLMFrontend,
    get_frontend,
    list_frontend_types,
    register_frontend,
)

# ============================================================================
# get_frontend() Tests
# ============================================================================


class TestGetFrontend:
    """Tests for frontend factory function."""

    def test_get_dynamo_frontend(self):
        """get_frontend('dynamo') returns DynamoFrontend."""
        frontend = get_frontend("dynamo")
        assert isinstance(frontend, DynamoFrontend)
        assert frontend.type == "dynamo"

    def test_get_sglang_frontend(self):
        """get_frontend('sglang') is the direct frontend; 'sglang-router' is the Model Gateway."""
        frontend = get_frontend("sglang")
        assert isinstance(frontend, SGLangFrontend)
        assert frontend.type == "sglang"
        router = get_frontend("sglang-router")
        assert isinstance(router, SGLangRouterFrontend)
        assert router.type == "sglang-router"
        assert frontend.type == "sglang"

    def test_get_vllm_frontend(self):
        """get_frontend('vllm') returns VLLMFrontend."""
        frontend = get_frontend("vllm")
        assert isinstance(frontend, VLLMFrontend)
        assert frontend.type == "vllm"

    def test_get_unknown_frontend_raises(self):
        """get_frontend() with unknown type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown frontend type"):
            get_frontend("unknown")

        with pytest.raises(ValueError, match="Unknown frontend type"):
            get_frontend("invalid")


class TestFrontendRegistry:
    """frontend.type resolves through the registry and nowhere else."""

    def test_registry_lists_every_frontend_type(self):
        assert list_frontend_types() == [
            "dynamo",
            "none",
            "sglang",
            "sglang-router",
            "trtllm_serve",
            "vllm",
            "vllm-router",
        ]
        for name in list_frontend_types():
            if name == "none":
                continue
            frontend = get_frontend(name)
            assert frontend.type == name
            assert hasattr(frontend, "required_backend")
            assert callable(frontend.validate)

    @pytest.mark.parametrize(
        ("frontend_type", "launch", "agg_port", "pd_port", "expands"),
        [
            ("dynamo", "dynamo", "allocated", "allocated", False),
            ("sglang", "direct", "public", "public", False),
            ("sglang-router", "direct", "allocated", "allocated", False),
            ("trtllm_serve", "direct", "public", "allocated", False),
            ("vllm", "direct", "public", "public", False),
            ("vllm-router", "direct", "allocated", "allocated", True),
        ],
    )
    def test_worker_shape_contract(self, frontend_type, launch, agg_port, pd_port, expands):
        """Backends read these instead of comparing frontend names."""
        frontend = get_frontend(frontend_type)
        assert frontend.worker_launch == launch
        assert frontend.worker_api_port("agg") == agg_port
        assert frontend.worker_api_port("prefill") == pd_port
        assert frontend.worker_api_port("decode") == pd_port
        assert frontend.expands_node_local_dp is expands

    @pytest.mark.parametrize(
        ("frontend_type", "metrics_path", "metrics", "endpoint", "direct_nodes", "ready"),
        [
            # metrics/endpoint: ports for (agg leader, agg follower, routed decode pool, prefill leader)
            ("dynamo", "/metrics", (7500, 7501, 7501, 7502), (7500, None, None, 7502), [], 7500),
            ("vllm", "/metrics", (8000, None, None, None), (8000, None, None, 8000), ["n0"], 7500),
            ("sglang", "/metrics", (8000, None, None, None), (8000, None, None, 8000), ["n0"], 7500),
            ("sglang-router", "/metrics", (6100, None, None, 6100), (6100, None, None, 6100), [], 7500),
            ("vllm-router", "/metrics", (6100, None, 6132, 6100), (6100, None, 6132, 6100), [], 7500),
            ("trtllm_serve", "/prometheus/metrics", (None, None, 6132, 6100), (8000, None, None, 6100), [], 6100),
        ],
    )
    def test_worker_port_contract(self, frontend_type, metrics_path, metrics, endpoint, direct_nodes, ready):
        """Telemetry, the benchmark env, and sequential start read these instead of comparing names."""
        from srtctl.core.topology import Process

        agg_leader = Process("n0", frozenset({0}), 7500, 6100, "agg", 0, node_rank=0)
        agg_follower = Process("n1", frozenset({0}), 7501, 0, "agg", 0, node_rank=1)
        routed_pool = Process("n1", frozenset({0}), 7501, 6132, "decode", 0, node_rank=1)
        prefill_leader = Process("n2", frozenset({0}), 7502, 6100, "prefill", 0, node_rank=0)
        processes = [agg_leader, agg_follower, routed_pool, prefill_leader]
        runtime = SimpleNamespace(frontend_port=8000, network_interface=None)
        config = SimpleNamespace(dynamo=SimpleNamespace(sidecar=False))

        frontend = get_frontend(frontend_type)
        assert frontend.metrics_path == metrics_path
        assert tuple(frontend.worker_metrics_port(p, runtime) for p in processes) == metrics
        assert tuple(frontend.worker_endpoint_port(p, config, runtime) for p in processes) == endpoint
        assert frontend.direct_endpoint_nodes(processes) == direct_nodes
        assert frontend.worker_ready_port(agg_leader) == ready
        assert isinstance(frontend.profiling_control_is_leader_only(config), bool)

    def test_dynamo_sidecar_moves_the_endpoint_to_the_engine_port(self):
        from srtctl.core.topology import Process

        leader = Process("n0", frozenset({0}), 7500, 6100, "agg", 0, node_rank=0)
        runtime = SimpleNamespace(frontend_port=8000, network_interface=None)
        dynamo = get_frontend("dynamo")
        assert (
            dynamo.worker_endpoint_port(leader, SimpleNamespace(dynamo=SimpleNamespace(sidecar=True)), runtime) == 6100
        )
        assert dynamo.profiling_control_is_leader_only(SimpleNamespace(dynamo=SimpleNamespace(sidecar=True))) is True
        assert dynamo.profiling_control_is_leader_only(SimpleNamespace(dynamo=SimpleNamespace(sidecar=False))) is False

    def test_dynamic_frontend_base_carries_the_registration_defaults(self, monkeypatch):
        """Dynamo is a DynamicFrontend; a future registration-based frontend inherits the same defaults."""
        from srtctl.frontends import DynamicFrontend, base

        assert isinstance(get_frontend("dynamo"), DynamicFrontend)
        monkeypatch.setattr(base, "_FRONTENDS", dict(base._FRONTENDS))

        @register_frontend("toy-discovery")
        class ToyDiscovery(DynamicFrontend):
            type = "toy-discovery"
            worker_launch = "direct"

        toy = get_frontend("toy-discovery")
        assert isinstance(toy, ToyDiscovery)
        assert toy.required_backend is None
        assert toy.health_endpoint == "/health"
        assert toy.metrics_path == "/metrics"
        assert toy.expands_node_local_dp is False
        assert toy.worker_api_port("prefill") == "allocated"
        assert toy.get_backend_health_urls(None, [], None) == []
        assert toy.direct_endpoint_nodes([]) == []
        assert toy.get_frontend_args_list({"router_mode": "kv", "flag": True, "off": False}) == [
            "--router_mode",
            "kv",
            "--flag",
        ]
        toy.validate(SimpleNamespace())

    def test_register_frontend_makes_a_type_resolvable(self, monkeypatch):
        from srtctl.frontends import base

        monkeypatch.setattr(base, "_FRONTENDS", dict(base._FRONTENDS))

        @register_frontend("toy-router")
        class ToyRouter:
            required_backend = "vllm"
            worker_launch = "direct"
            expands_node_local_dp = False

            @property
            def type(self) -> str:
                return "toy-router"

            def validate(self, config) -> None:
                del config

            def worker_api_port(self, mode: str) -> str:
                del mode
                return "allocated"

        assert isinstance(get_frontend("toy-router"), ToyRouter)
        assert "toy-router" in list_frontend_types()

    def test_schema_rejects_unknown_type_at_load(self):
        from marshmallow import ValidationError

        from srtctl.backends import SGLangProtocol
        from srtctl.core.schema import FrontendConfig, ResourceConfig, SrtConfig

        with pytest.raises(ValidationError, match="Unknown frontend.type 'toy-router'.*Available: dynamo, none"):
            SrtConfig(
                name="toy",
                model={"path": "model", "container": "image", "precision": "fp8"},
                resources=ResourceConfig(gpu_type="h100", gpus_per_node=8, agg_nodes=1, agg_workers=1),
                frontend=FrontendConfig(type="toy-router", enable_multiple_frontends=False),
                backend=SGLangProtocol(),
            )

    @pytest.mark.parametrize(
        ("frontend_type", "required"),
        [("sglang", "sglang"), ("sglang-router", "sglang"), ("vllm", "vllm"), ("vllm-router", "vllm")],
    )
    def test_schema_enforces_required_backend_generically(self, frontend_type, required):
        from marshmallow import ValidationError

        from srtctl.backends import TRTLLMProtocol
        from srtctl.core.schema import FrontendConfig, ResourceConfig, SrtConfig

        assert get_frontend(frontend_type).required_backend == required
        with pytest.raises(ValidationError, match=f"frontend.type: {frontend_type} requires backend.type: {required}"):
            SrtConfig(
                name="pairing",
                model={"path": "model", "container": "image", "precision": "fp8"},
                resources=ResourceConfig(gpu_type="h100", gpus_per_node=8, agg_nodes=1, agg_workers=1),
                frontend=FrontendConfig(type=frontend_type, enable_multiple_frontends=False),
                backend=TRTLLMProtocol(),
            )


# ============================================================================
# Frontend Properties Tests
# ============================================================================


class TestFrontendProperties:
    """Tests for frontend properties."""

    def test_dynamo_type(self):
        """DynamoFrontend.type is 'dynamo'."""
        frontend = DynamoFrontend()
        assert frontend.type == "dynamo"

    def test_sglang_type(self):
        """SGLangRouterFrontend.type is 'sglang-router'."""
        frontend = SGLangRouterFrontend()
        assert frontend.type == "sglang-router"

    def test_vllm_type(self):
        """VLLMFrontend.type is 'vllm'."""
        frontend = VLLMFrontend()
        assert frontend.type == "vllm"

    def test_dynamo_health_endpoint(self):
        """DynamoFrontend uses /health endpoint."""
        frontend = DynamoFrontend()
        assert frontend.health_endpoint == "/health"

    def test_sglang_health_endpoint(self):
        """SGLangFrontend uses /workers endpoint."""
        frontend = SGLangRouterFrontend()
        assert frontend.health_endpoint == "/workers"

    def test_frontend_metrics_port_and_implied_services(self):
        """Only the SGLang gateway runs a separate metrics listener; only Dynamo brings a discovery plane."""
        from types import SimpleNamespace

        from srtctl.ports import SGLANG_ROUTER_METRICS_PORT

        gateway = SGLangRouterFrontend()
        assert gateway.frontend_metrics_port(None) == SGLANG_ROUTER_METRICS_PORT
        assert gateway.frontend_metrics_port({"prometheus-port": 31000}) == 31000
        for frontend_type in ("dynamo", "vllm", "sglang", "trtllm_serve", "vllm-router"):
            assert get_frontend(frontend_type).frontend_metrics_port({"prometheus-port": 31000}) is None

        dynamo_config = SimpleNamespace(
            frontend=SimpleNamespace(type="dynamo"),
            dynamo=SimpleNamespace(request_plane="nats", event_plane="zmq"),
            infra=SimpleNamespace(etcd_nats_dedicated_node=False, nats_max_payload_mb=None),
        )
        implied = get_frontend("dynamo").implied_services(dynamo_config)
        assert [(entry.service.name, entry.service.type, entry.reason) for entry in implied] == [
            ("etcd", "etcd", "frontend.type dynamo"),
            ("nats", "nats", "dynamo.request_plane nats"),
        ]
        for frontend_type in ("vllm", "sglang", "sglang-router", "trtllm_serve", "vllm-router"):
            assert get_frontend(frontend_type).implied_services(dynamo_config) == []


# ============================================================================
# Frontend Args List Tests
# ============================================================================


class TestGetFrontendArgsList:
    """Tests for get_frontend_args_list() method."""

    def test_empty_args_returns_empty_list(self):
        """None or empty args returns empty list."""
        frontend = SGLangRouterFrontend()

        assert frontend.get_frontend_args_list(None) == []
        assert frontend.get_frontend_args_list({}) == []

    def test_boolean_true_flag(self):
        """Boolean True generates flag without value."""
        frontend = SGLangRouterFrontend()

        result = frontend.get_frontend_args_list({"verbose": True})
        assert result == ["--verbose"]

    def test_boolean_false_flag_skipped(self):
        """Boolean False is skipped."""
        frontend = SGLangRouterFrontend()

        result = frontend.get_frontend_args_list({"verbose": False})
        assert result == []

    def test_none_value_skipped(self):
        """None values are skipped."""
        frontend = SGLangRouterFrontend()

        result = frontend.get_frontend_args_list({"some-arg": None})
        assert result == []

    def test_string_value(self):
        """String values become --key value pairs."""
        frontend = SGLangRouterFrontend()

        result = frontend.get_frontend_args_list({"policy": "cache_aware"})
        assert result == ["--policy", "cache_aware"]

    def test_numeric_value(self):
        """Numeric values are converted to strings."""
        frontend = SGLangRouterFrontend()

        result = frontend.get_frontend_args_list({"timeout": 120})
        assert result == ["--timeout", "120"]

    def test_float_value(self):
        """Float values are converted to strings."""
        frontend = SGLangRouterFrontend()

        result = frontend.get_frontend_args_list({"temperature": 0.5})
        assert result == ["--temperature", "0.5"]

    def test_mixed_args(self):
        """Mixed arg types are handled correctly."""
        frontend = SGLangRouterFrontend()

        result = frontend.get_frontend_args_list(
            {
                "policy": "round_robin",
                "verbose": True,
                "timeout": 60,
                "disabled": False,
                "optional": None,
            }
        )

        # Check all expected args are present
        assert "--policy" in result
        assert "round_robin" in result
        assert "--verbose" in result
        assert "--timeout" in result
        assert "60" in result
        # Disabled and None should not appear
        assert "--disabled" not in result
        assert "--optional" not in result

    def test_dynamo_frontend_args_list(self):
        """DynamoFrontend has same args list behavior."""
        frontend = DynamoFrontend()

        result = frontend.get_frontend_args_list(
            {
                "router-mode": "kv",
                "router-reset-states": True,
            }
        )

        assert "--router-mode" in result
        assert "kv" in result
        assert "--router-reset-states" in result


# ============================================================================
# SGLang gRPC Scheme Tests
# ============================================================================


@dataclass
class MockProcess:
    """Mock Process for testing."""

    node: str
    endpoint_mode: str
    http_port: int
    bootstrap_port: int | None = None
    is_leader: bool = True


@dataclass
class MockTopology:
    """Mock FrontendTopology for testing."""

    frontend_nodes: list[str]
    frontend_port: int = 8180


@dataclass
class MockFrontendConfig:
    """Mock FrontendConfig for testing."""

    type: str = "sglang"
    args: dict | None = None
    env: dict | None = None
    numa_bind: bool = False


@dataclass
class MockResourceConfig:
    """Mock ResourceConfig for testing."""

    num_prefill: int = 0
    num_decode: int = 0
    num_agg: int = 0


@dataclass
class MockObservabilityConfig:
    """Mock ObservabilityConfig for testing."""

    enable_otel: bool = False
    otel_endpoint: str | None = None


@dataclass
class MockConfig:
    """Mock SrtConfig for testing."""

    frontend: MockFrontendConfig
    resources: MockResourceConfig
    observability: MockObservabilityConfig = field(default_factory=MockObservabilityConfig)


class TestSGLangGrpcScheme:
    """Tests for gRPC/HTTP scheme selection in SGLang frontend."""

    @patch("srtctl.frontends.sglang.start_srun_process")
    @patch("srtctl.frontends.sglang.get_hostname_ip")
    def test_http_scheme_by_default(self, mock_get_ip, mock_srun):
        """Default scheme is http:// when gRPC not enabled."""
        mock_get_ip.return_value = "10.0.0.1"
        mock_srun.return_value = MagicMock()

        frontend = SGLangRouterFrontend()
        topology = MockTopology(frontend_nodes=["node0"])
        config = MockConfig(
            frontend=MockFrontendConfig(),
            resources=MockResourceConfig(num_agg=2),
        )

        # Mock backend without gRPC
        backend = MagicMock()
        backend.is_grpc_mode.return_value = False

        # Mock runtime
        runtime = MagicMock()
        runtime.log_dir = MagicMock()
        runtime.log_dir.__truediv__ = lambda self, x: f"/logs/{x}"
        runtime.container_image = "/container.sqsh"
        runtime.container_mounts = {}

        # Mock agg workers
        processes = [
            MockProcess(node="node1", endpoint_mode="agg", http_port=30000),
            MockProcess(node="node2", endpoint_mode="agg", http_port=30000),
        ]

        frontend.start_frontends(topology, runtime, config, backend, processes)

        # Check the command passed to start_srun_process
        call_args = mock_srun.call_args
        cmd = call_args.kwargs["command"]

        # Should use http:// scheme
        assert any("http://10.0.0.1:30000" in arg for arg in cmd)
        assert not any("grpc://" in arg for arg in cmd)

    @patch("srtctl.frontends.sglang.start_srun_process")
    @patch("srtctl.frontends.sglang.get_hostname_ip")
    def test_grpc_scheme_when_enabled(self, mock_get_ip, mock_srun):
        """gRPC scheme used when backend has grpc-mode enabled."""
        mock_get_ip.return_value = "10.0.0.1"
        mock_srun.return_value = MagicMock()

        frontend = SGLangRouterFrontend()
        topology = MockTopology(frontend_nodes=["node0"])
        config = MockConfig(
            frontend=MockFrontendConfig(),
            resources=MockResourceConfig(num_agg=1),
        )

        # Mock SGLangProtocol backend with gRPC enabled
        from srtctl.backends.sglang import SGLangProtocol

        backend = MagicMock(spec=SGLangProtocol)
        backend.is_grpc_mode.side_effect = lambda mode: mode == "agg"

        # Mock runtime
        runtime = MagicMock()
        runtime.log_dir = MagicMock()
        runtime.log_dir.__truediv__ = lambda self, x: f"/logs/{x}"
        runtime.container_image = "/container.sqsh"
        runtime.container_mounts = {}

        processes = [
            MockProcess(node="node1", endpoint_mode="agg", http_port=30000),
        ]

        frontend.start_frontends(topology, runtime, config, backend, processes)

        call_args = mock_srun.call_args
        cmd = call_args.kwargs["command"]

        # Should use grpc:// scheme for agg
        assert any("grpc://10.0.0.1:30000" in arg for arg in cmd)

    @patch("srtctl.frontends.sglang.start_srun_process")
    @patch("srtctl.frontends.sglang.get_hostname_ip")
    def test_disaggregated_mode_command(self, mock_get_ip, mock_srun):
        """Disaggregated mode uses --pd-disaggregation with --prefill and --decode."""
        mock_get_ip.side_effect = lambda node: f"10.0.0.{node[-1]}"
        mock_srun.return_value = MagicMock()

        frontend = SGLangRouterFrontend()
        topology = MockTopology(frontend_nodes=["node0"])
        config = MockConfig(
            frontend=MockFrontendConfig(),
            resources=MockResourceConfig(num_prefill=1, num_decode=2),
        )

        backend = MagicMock()
        backend.is_grpc_mode.return_value = False

        runtime = MagicMock()
        runtime.log_dir = MagicMock()
        runtime.log_dir.__truediv__ = lambda self, x: f"/logs/{x}"
        runtime.container_image = "/container.sqsh"
        runtime.container_mounts = {}

        processes = [
            MockProcess(node="node1", endpoint_mode="prefill", http_port=30000, bootstrap_port=30001),
            MockProcess(node="node2", endpoint_mode="decode", http_port=30000),
            MockProcess(node="node3", endpoint_mode="decode", http_port=30000),
        ]

        frontend.start_frontends(topology, runtime, config, backend, processes)

        call_args = mock_srun.call_args
        cmd = call_args.kwargs["command"]

        # Check disaggregated mode flags
        assert "--pd-disaggregation" in cmd
        assert "--prefill" in cmd
        assert "--decode" in cmd
        # Bootstrap port should be included
        assert "30001" in cmd

    @patch("srtctl.frontends.sglang.start_srun_process")
    @patch("srtctl.frontends.sglang.get_hostname_ip")
    def test_aggregated_mode_command(self, mock_get_ip, mock_srun):
        """Aggregated mode uses --worker-urls."""
        mock_get_ip.side_effect = lambda node: f"10.0.0.{node[-1]}"
        mock_srun.return_value = MagicMock()

        frontend = SGLangRouterFrontend()
        topology = MockTopology(frontend_nodes=["node0"])
        config = MockConfig(
            frontend=MockFrontendConfig(),
            resources=MockResourceConfig(num_agg=2),
        )

        backend = MagicMock()
        backend.is_grpc_mode.return_value = False

        runtime = MagicMock()
        runtime.log_dir = MagicMock()
        runtime.log_dir.__truediv__ = lambda self, x: f"/logs/{x}"
        runtime.container_image = "/container.sqsh"
        runtime.container_mounts = {}

        processes = [
            MockProcess(node="node1", endpoint_mode="agg", http_port=30000),
            MockProcess(node="node2", endpoint_mode="agg", http_port=30000),
        ]

        frontend.start_frontends(topology, runtime, config, backend, processes)

        call_args = mock_srun.call_args
        cmd = call_args.kwargs["command"]

        # Check aggregated mode flags
        assert "--worker-urls" in cmd
        assert "--pd-disaggregation" not in cmd


# ============================================================================
# Frontend Env Handling Tests
# ============================================================================


class TestFrontendEnvHandling:
    """Tests for frontend environment variable handling."""

    @patch("srtctl.frontends.sglang.start_srun_process")
    @patch("srtctl.frontends.sglang.get_hostname_ip")
    def test_sglang_env_passed_to_process(self, mock_get_ip, mock_srun):
        """SGLang frontend passes env dict to start_srun_process."""
        mock_get_ip.return_value = "10.0.0.1"
        mock_srun.return_value = MagicMock()

        frontend = SGLangRouterFrontend()
        topology = MockTopology(frontend_nodes=["node0"])
        config = MockConfig(
            frontend=MockFrontendConfig(env={"MY_VAR": "my_value", "ANOTHER": "123"}),
            resources=MockResourceConfig(num_agg=1),
        )

        backend = MagicMock()
        backend.is_grpc_mode.return_value = False

        runtime = MagicMock()
        runtime.log_dir = MagicMock()
        runtime.log_dir.__truediv__ = lambda self, x: f"/logs/{x}"
        runtime.container_image = "/container.sqsh"
        runtime.container_mounts = {}

        processes = [
            MockProcess(node="node1", endpoint_mode="agg", http_port=30000),
        ]

        frontend.start_frontends(topology, runtime, config, backend, processes)

        call_args = mock_srun.call_args
        env_to_set = call_args.kwargs.get("env_to_set")

        assert env_to_set is not None
        assert env_to_set["MY_VAR"] == "my_value"
        assert env_to_set["ANOTHER"] == "123"

    @patch("srtctl.frontends.sglang.start_srun_process")
    @patch("srtctl.frontends.sglang.get_hostname_ip")
    def test_sglang_no_env_when_empty(self, mock_get_ip, mock_srun):
        """SGLang frontend passes None for env when not configured."""
        mock_get_ip.return_value = "10.0.0.1"
        mock_srun.return_value = MagicMock()

        frontend = SGLangRouterFrontend()
        topology = MockTopology(frontend_nodes=["node0"])
        config = MockConfig(
            frontend=MockFrontendConfig(env=None),
            resources=MockResourceConfig(num_agg=1),
        )

        backend = MagicMock()
        backend.is_grpc_mode.return_value = False

        runtime = MagicMock()
        runtime.log_dir = MagicMock()
        runtime.log_dir.__truediv__ = lambda self, x: f"/logs/{x}"
        runtime.container_image = "/container.sqsh"
        runtime.container_mounts = {}

        processes = [
            MockProcess(node="node1", endpoint_mode="agg", http_port=30000),
        ]

        frontend.start_frontends(topology, runtime, config, backend, processes)

        call_args = mock_srun.call_args
        env_to_set = call_args.kwargs.get("env_to_set")

        # Should be None when no env configured
        assert env_to_set is None

    @patch("srtctl.frontends.sglang.start_srun_process")
    @patch("srtctl.frontends.sglang.get_hostname_ip")
    def test_sglang_frontend_args_in_command(self, mock_get_ip, mock_srun):
        """SGLang frontend includes args in command."""
        mock_get_ip.return_value = "10.0.0.1"
        mock_srun.return_value = MagicMock()

        frontend = SGLangRouterFrontend()
        topology = MockTopology(frontend_nodes=["node0"])
        config = MockConfig(
            frontend=MockFrontendConfig(args={"policy": "cache_aware", "verbose": True}),
            resources=MockResourceConfig(num_agg=1),
        )

        backend = MagicMock()
        backend.is_grpc_mode.return_value = False

        runtime = MagicMock()
        runtime.log_dir = MagicMock()
        runtime.log_dir.__truediv__ = lambda self, x: f"/logs/{x}"
        runtime.container_image = "/container.sqsh"
        runtime.container_mounts = {}

        processes = [
            MockProcess(node="node1", endpoint_mode="agg", http_port=30000),
        ]

        frontend.start_frontends(topology, runtime, config, backend, processes)

        call_args = mock_srun.call_args
        cmd = call_args.kwargs["command"]

        assert "--policy" in cmd
        assert "cache_aware" in cmd
        assert "--verbose" in cmd


class TestNumaBind:
    """frontend.numa_bind prefixes the frontend process command with numactl."""

    @patch("srtctl.frontends.sglang.start_srun_process")
    @patch("srtctl.frontends.sglang.get_hostname_ip")
    def test_static_router_prefixed_when_enabled(self, mock_get_ip, mock_srun):
        mock_get_ip.return_value = "10.0.0.1"
        mock_srun.return_value = MagicMock()

        frontend = SGLangRouterFrontend()
        topology = MockTopology(frontend_nodes=["node0"])
        config = MockConfig(
            frontend=MockFrontendConfig(numa_bind=True),
            resources=MockResourceConfig(num_agg=1),
        )
        backend = MagicMock()
        backend.is_grpc_mode.return_value = False
        runtime = MagicMock()
        runtime.log_dir = MagicMock()
        runtime.log_dir.__truediv__ = lambda self, x: f"/logs/{x}"
        runtime.container_image = "/container.sqsh"
        runtime.container_mounts = {}
        processes = [MockProcess(node="node1", endpoint_mode="agg", http_port=30000)]

        frontend.start_frontends(topology, runtime, config, backend, processes)

        cmd = mock_srun.call_args.kwargs["command"]
        assert cmd[:3] == ["numactl", "--cpunodebind=0", "--membind=0"]

    @patch("srtctl.frontends.sglang.start_srun_process")
    @patch("srtctl.frontends.sglang.get_hostname_ip")
    def test_static_router_not_prefixed_by_default(self, mock_get_ip, mock_srun):
        mock_get_ip.return_value = "10.0.0.1"
        mock_srun.return_value = MagicMock()

        frontend = SGLangRouterFrontend()
        topology = MockTopology(frontend_nodes=["node0"])
        config = MockConfig(
            frontend=MockFrontendConfig(),
            resources=MockResourceConfig(num_agg=1),
        )
        backend = MagicMock()
        backend.is_grpc_mode.return_value = False
        runtime = MagicMock()
        runtime.log_dir = MagicMock()
        runtime.log_dir.__truediv__ = lambda self, x: f"/logs/{x}"
        runtime.container_image = "/container.sqsh"
        runtime.container_mounts = {}
        processes = [MockProcess(node="node1", endpoint_mode="agg", http_port=30000)]

        frontend.start_frontends(topology, runtime, config, backend, processes)

        cmd = mock_srun.call_args.kwargs["command"]
        assert "numactl" not in cmd

    def test_dynamo_frontend_prefixed_when_enabled(self):
        frontend = DynamoFrontend()
        topology = SimpleNamespace(frontend_nodes=["node0"], frontend_port=8180)
        runtime = SimpleNamespace(
            log_dir=Path("/logs"),
            nodes=SimpleNamespace(infra="infra-node", het_group_for=lambda node: None),
            container_image=Path("/container.sqsh"),
            container_mounts={},
            environment={},
        )
        config = SimpleNamespace(
            frontend=SimpleNamespace(args=None, env=None, worker_selection=None, numa_bind=True),
            observability=ObservabilityConfig(),
            dynamo=SimpleNamespace(
                install=False, get_install_commands=lambda: "", request_plane="tcp", event_plane=None
            ),
            setup_script=None,
        )
        with patch("srtctl.frontends.dynamo.start_srun_process") as mock_srun:
            mock_srun.return_value = MagicMock()
            frontend.start_frontends(topology, runtime, config, MagicMock(), [])

        cmd = mock_srun.call_args.kwargs["command"]
        assert cmd[:3] == ["numactl", "--cpunodebind=0", "--membind=0"]
        assert "dynamo.frontend" in cmd


# ============================================================================
# Dynamo Frontend ENROOT_REMAP_ROOT injection Tests
# ============================================================================


def test_dynamo_frontend_is_a_named_step_with_a_drain_timeout():
    from srtctl.core.processes import FRONTEND_TERMINATE_TIMEOUT_SECONDS

    frontend = DynamoFrontend()
    topology = SimpleNamespace(frontend_nodes=["node0"], frontend_port=8180)
    runtime = SimpleNamespace(
        log_dir=Path("/logs"),
        nodes=SimpleNamespace(infra="infra-node", het_group_for=lambda node: None),
        container_image=Path("/container.sqsh"),
        container_mounts={},
        environment={},
    )
    config = SimpleNamespace(
        frontend=SimpleNamespace(args=None, env=None),
        observability=ObservabilityConfig(),
        dynamo=SimpleNamespace(install=False, get_install_commands=lambda: "", request_plane="tcp", event_plane=None),
        setup_script=None,
    )
    with patch("srtctl.frontends.dynamo.start_srun_process") as mock_srun:
        mock_srun.return_value = MagicMock()
        (proc,) = frontend.start_frontends(topology, runtime, config, MagicMock(), [])
    assert mock_srun.call_args.kwargs["step_name"] == "frontend_0"
    assert proc.step_name == "frontend_0"
    assert proc.terminate_timeout == FRONTEND_TERMINATE_TIMEOUT_SECONDS


def _dynamo_frontend_call(*, dynamo_install: bool, event_plane: str | None = "zmq"):
    """Invoke DynamoFrontend.start_frontends with a minimal config; return the mock srun call."""
    frontend = DynamoFrontend()
    topology = SimpleNamespace(frontend_nodes=["node0"], frontend_port=8180)
    runtime = SimpleNamespace(
        log_dir=Path("/logs"),
        nodes=SimpleNamespace(infra="infra-node", het_group_for=lambda node: None),
        container_image=Path("/container.sqsh"),
        container_mounts={},
        environment={},
    )
    config = SimpleNamespace(
        frontend=SimpleNamespace(args=None, env=None, worker_selection=None),
        observability=ObservabilityConfig(),
        dynamo=SimpleNamespace(
            install=dynamo_install,
            get_install_commands=lambda: "echo install-dynamo",
            request_plane="nats",
            event_plane=event_plane,
        ),
        setup_script=None,
    )
    with patch("srtctl.frontends.dynamo.start_srun_process") as mock_srun:
        mock_srun.return_value = MagicMock()
        frontend.start_frontends(topology, runtime, config, MagicMock(), [])
    return mock_srun


class TestDynamoFrontendRemapRoot:
    """Dynamo frontend injects ENROOT_REMAP_ROOT only when it installs dynamo."""

    def test_injects_remap_root_when_install(self):
        mock_srun = _dynamo_frontend_call(dynamo_install=True)
        assert mock_srun.call_args.kwargs["srun_export_env"] == {"ENROOT_REMAP_ROOT": "yes"}

    def test_no_remap_root_when_install_false(self):
        mock_srun = _dynamo_frontend_call(dynamo_install=False)
        assert mock_srun.call_args.kwargs["srun_export_env"] is None


class TestDynamoFrontendEventPlane:
    """DYN_EVENT_PLANE is injected only when dynamo.event_plane is set."""

    def test_default_not_injected(self):
        mock_srun = _dynamo_frontend_call(dynamo_install=False, event_plane=None)
        assert "DYN_EVENT_PLANE" not in mock_srun.call_args.kwargs["env_to_set"]

    @pytest.mark.parametrize("event_plane", ["zmq", "nats"])
    def test_explicit_injected(self, event_plane):
        mock_srun = _dynamo_frontend_call(dynamo_install=False, event_plane=event_plane)
        assert mock_srun.call_args.kwargs["env_to_set"]["DYN_EVENT_PLANE"] == event_plane


def test_dynamo_frontend_materializes_inline_worker_selection(tmp_path):
    """Inline policy config becomes a mounted YAML file and frontend CLI argument."""
    frontend = DynamoFrontend()
    topology = SimpleNamespace(frontend_nodes=["node0"], frontend_port=8180)
    runtime = SimpleNamespace(
        log_dir=tmp_path,
        nodes=SimpleNamespace(infra="infra-node", het_group_for=lambda node: None),
        container_image=Path("/container.sqsh"),
        container_mounts={tmp_path: Path("/logs")},
        environment={},
    )
    worker_selection = {
        "prefill": "max-kv-overlap",
        "decode": "default",
        "instances": [
            {
                "name": "max-kv-overlap",
                "type": "dynamo-two-tier-cost-fn",
                "parameters": {
                    "cache_threshold": 0.0,
                    "balance_abs_threshold": 1_000_000_000,
                    "balance_rel_threshold": 1_000_000_000.0,
                },
            }
        ],
    }
    config = SimpleNamespace(
        frontend=SimpleNamespace(args={"router-mode": "kv"}, env=None, worker_selection=worker_selection),
        observability=ObservabilityConfig(),
        dynamo=SimpleNamespace(
            install=False,
            get_install_commands=lambda: "",
            request_plane="nats",
            event_plane=None,
        ),
        setup_script=None,
    )

    with patch("srtctl.frontends.dynamo.start_srun_process") as mock_srun:
        mock_srun.return_value = MagicMock()
        frontend.start_frontends(topology, runtime, config, MagicMock(), [])

    policy_path = tmp_path / "router_policy_config.yaml"
    assert yaml.safe_load(policy_path.read_text()) == {"worker_selection": worker_selection}
    cmd = mock_srun.call_args.kwargs["command"]
    policy_arg = cmd.index("--router-policy-config")
    assert cmd[policy_arg + 1] == "/logs/router_policy_config.yaml"
    assert cmd[cmd.index("--router-mode") + 1] == "kv"
