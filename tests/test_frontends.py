# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for frontend implementations (SGLang and Dynamo)."""

from dataclasses import dataclass, field
from unittest.mock import MagicMock, patch

import pytest

from srtctl.frontends import DynamoFrontend, SGLangFrontend, get_frontend

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

    def test_get_unknown_frontend_raises(self):
        """get_frontend() with unknown type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown frontend type"):
            get_frontend("unknown")

        with pytest.raises(ValueError, match="Unknown frontend type"):
            get_frontend("vllm")


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

    def test_dynamo_health_endpoint(self):
        """DynamoFrontend uses /health endpoint."""
        frontend = DynamoFrontend()
        assert frontend.health_endpoint == "/health"

    def test_sglang_health_endpoint(self):
        """SGLangFrontend uses /workers endpoint."""
        frontend = SGLangFrontend()
        assert frontend.health_endpoint == "/workers"


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
class MockInfraConfig:
    """Mock InfraConfig for testing."""

    enable_otel: bool = False
    otel_endpoint: str | None = None

    def otel_env(self, component: str) -> dict[str, str]:
        if not self.enable_otel or not self.otel_endpoint:
            return {}
        return {
            "DYN_LOGGING_JSONL": "1",
            "OTEL_EXPORT_ENABLED": "1",
            "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT": self.otel_endpoint,
            "OTEL_SERVICE_NAME": f"dynamo-{component}",
        }


@dataclass
class MockConfig:
    """Mock SrtConfig for testing."""

    frontend: MockFrontendConfig
    resources: MockResourceConfig
    infra: MockInfraConfig = field(default_factory=MockInfraConfig)


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
