# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for frontend implementations."""

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from srtctl.core.schema import ObservabilityConfig
from srtctl.frontends import DynamoFrontend, SGLangFrontend, SGLRouterFrontend, VLLMFrontend, get_frontend

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
        """get_frontend('sglang') returns SGLangFrontend."""
        frontend = get_frontend("sglang")
        assert isinstance(frontend, SGLangFrontend)
        assert frontend.type == "sglang"

    def test_get_sgl_router_frontend(self):
        """get_frontend('sgl-router') returns the experimental Rust router."""
        frontend = get_frontend("sgl-router")
        assert isinstance(frontend, SGLRouterFrontend)
        assert frontend.type == "sgl-router"

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
        """SGLangFrontend.type is 'sglang'."""
        frontend = SGLangFrontend()
        assert frontend.type == "sglang"

    def test_sgl_router_type(self):
        """SGLRouterFrontend.type is 'sgl-router'."""
        frontend = SGLRouterFrontend()
        assert frontend.type == "sgl-router"

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
        frontend = SGLangFrontend()
        assert frontend.health_endpoint == "/workers"

    def test_sgl_router_health_endpoint(self):
        """SGLRouterFrontend uses /metrics for static worker registration."""
        frontend = SGLRouterFrontend()
        assert frontend.health_endpoint == "/metrics"

    def test_vllm_health_endpoint(self):
        """VLLMFrontend uses /health endpoint."""
        frontend = VLLMFrontend()
        assert frontend.health_endpoint == "/health"


# ============================================================================
# Frontend Args List Tests
# ============================================================================


class TestGetFrontendArgsList:
    """Tests for get_frontend_args_list() method."""

    def test_empty_args_returns_empty_list(self):
        """None or empty args returns empty list."""
        frontend = SGLangFrontend()

        assert frontend.get_frontend_args_list(None) == []
        assert frontend.get_frontend_args_list({}) == []

    def test_boolean_true_flag(self):
        """Boolean True generates flag without value."""
        frontend = SGLangFrontend()

        result = frontend.get_frontend_args_list({"verbose": True})
        assert result == ["--verbose"]

    def test_boolean_false_flag_skipped(self):
        """Boolean False is skipped."""
        frontend = SGLangFrontend()

        result = frontend.get_frontend_args_list({"verbose": False})
        assert result == []

    def test_none_value_skipped(self):
        """None values are skipped."""
        frontend = SGLangFrontend()

        result = frontend.get_frontend_args_list({"some-arg": None})
        assert result == []

    def test_string_value(self):
        """String values become --key value pairs."""
        frontend = SGLangFrontend()

        result = frontend.get_frontend_args_list({"policy": "cache_aware"})
        assert result == ["--policy", "cache_aware"]

    def test_numeric_value(self):
        """Numeric values are converted to strings."""
        frontend = SGLangFrontend()

        result = frontend.get_frontend_args_list({"timeout": 120})
        assert result == ["--timeout", "120"]

    def test_float_value(self):
        """Float values are converted to strings."""
        frontend = SGLangFrontend()

        result = frontend.get_frontend_args_list({"temperature": 0.5})
        assert result == ["--temperature", "0.5"]

    def test_mixed_args(self):
        """Mixed arg types are handled correctly."""
        frontend = SGLangFrontend()

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

        frontend = SGLangFrontend()
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

        frontend = SGLangFrontend()
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

        frontend = SGLangFrontend()
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

        frontend = SGLangFrontend()
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


class TestSGLRouterFrontend:
    """Tests for the experimental Rust router's static worker launch path."""

    @patch("srtctl.frontends.sgl_router.start_srun_process")
    @patch("srtctl.frontends.sgl_router.get_hostname_ip")
    def test_builds_binary_and_registers_static_http_workers(self, mock_get_ip, mock_srun):
        mock_get_ip.side_effect = lambda node: {"node1": "10.0.0.1", "node2": "10.0.0.2"}[node]
        mock_srun.return_value = MagicMock()
        frontend = SGLRouterFrontend()
        topology = MockTopology(frontend_nodes=["router0"])
        config = SimpleNamespace(
            frontend=SimpleNamespace(
                sgl_router=SimpleNamespace(source="/opt/sglang", binary=None),
                args={"policy": "cache_aware_zmq"},
                env={"RUST_LOG": "info"},
            ),
            served_model_name="Qwen/Qwen3-32B-FP8",
        )
        runtime = SimpleNamespace(
            log_dir=Path("/logs"),
            container_image=Path("/containers/sglang.sqsh"),
            container_mounts={},
            worker_model_arg="/model",
            is_hf_model=False,
            nodes=SimpleNamespace(het_group_for=lambda node: None),
        )
        workers = [
            MockProcess(node="node1", endpoint_mode="agg", http_port=30000),
            MockProcess(node="node2", endpoint_mode="agg", http_port=30001),
        ]

        processes = frontend.start_frontends(topology, runtime, config, MagicMock(), workers)

        assert processes[0].name == "sgl_router_0"
        assert processes[0].log_file == Path("/logs/router0_sgl_router_0.out")
        call = mock_srun.call_args.kwargs
        assert call["command"][:4] == ["bash", "-lc", 'exec "$SRTCTL_SGL_ROUTER_BINARY" "$@"', "sgl-router"]
        assert "--tokenizer-path" in call["command"]
        assert "/model/tokenizer.json" in call["command"]
        assert "http://10.0.0.1:30000" in call["command"]
        assert "http://10.0.0.2:30001" in call["command"]
        assert (
            "cargo build --manifest-path /opt/sglang/experimental/sgl-router/Cargo.toml --release"
            in call["bash_preamble"]
        )
        assert call["env_to_set"] == {"RUST_LOG": "info"}

    def test_worker_command_uses_direct_sglang_server(self):
        """The experimental router needs HTTP SGLang workers, not dynamo.sglang."""
        from srtctl.backends import SGLangProtocol
        from srtctl.core.topology import Process

        process = Process(
            node="node0",
            gpu_indices=frozenset({0}),
            sys_port=7500,
            http_port=6100,
            endpoint_mode="agg",
            endpoint_index=0,
        )
        runtime = SimpleNamespace(model_path=Path("/model"), is_hf_model=False)

        with patch("srtctl.core.slurm.get_hostname_ip", return_value="10.0.0.1"):
            command = SGLangProtocol().build_worker_command(
                process,
                [process],
                runtime,
                frontend_type="sgl-router",
            )

        assert "sglang.launch_server" in command
        assert "dynamo.sglang" not in command
        assert "--request-plane" not in command


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

        frontend = SGLangFrontend()
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

        frontend = SGLangFrontend()
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

        frontend = SGLangFrontend()
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


# ============================================================================
# Dynamo Frontend ENROOT_REMAP_ROOT injection Tests
# ============================================================================


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
        frontend=SimpleNamespace(args=None, env=None),
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
