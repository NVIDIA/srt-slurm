# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for telemetry configuration and startup."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from marshmallow import ValidationError

from srtctl.cli.mixins.frontend_stage import FrontendTopology
from srtctl.cli.mixins.telemetry_stage import TelemetryStageMixin
from srtctl.core.schema import (
    BenchmarkConfig,
    ModelConfig,
    ResourceConfig,
    SrtConfig,
    TelemetryConfig,
    TelemetryExporterConfig,
)
from srtctl.core.telemetry import generate_telemetry_config
from srtctl.core.topology import Process


def _make_config(*, telemetry: TelemetryConfig | None = None) -> SrtConfig:
    return SrtConfig(
        name="test",
        model=ModelConfig(path="/model", container="/image", precision="fp4"),
        resources=ResourceConfig(gpu_type="h100"),
        benchmark=BenchmarkConfig(type="manual"),
        telemetry=telemetry or TelemetryConfig(),
    )


class TestTelemetryConfig:
    """Telemetry schema validation."""

    def test_requires_dcgm_exporter_when_enabled(self):
        with pytest.raises(ValidationError, match="telemetry.dcgm_exporter"):
            _make_config(telemetry=TelemetryConfig(enabled=True))

    def test_node_exporter_requires_scraper(self):
        """Setting node_exporter without container_image is an error: nothing reads it."""
        with pytest.raises(ValidationError, match="container_image"):
            _make_config(
                telemetry=TelemetryConfig(
                    enabled=True,
                    dcgm_exporter=TelemetryExporterConfig(container_image="dcgm:latest", port=9401),
                    node_exporter=TelemetryExporterConfig(container_image="node:latest", port=9101),
                )
            )

    def test_dcgm_only_is_valid(self):
        """dcgm-exporter alone (no scraper, no node-exporter) is a valid config."""
        config = _make_config(
            telemetry=TelemetryConfig(
                enabled=True,
                dcgm_exporter=TelemetryExporterConfig(container_image="dcgm:latest", port=9401),
            )
        )
        assert config.telemetry.enabled is True
        assert config.telemetry.container_image is None
        assert config.telemetry.node_exporter is None


class TestTelemetryConfigGeneration:
    """Topology-to-config generation."""

    @patch("srtctl.core.telemetry.get_hostname_ip")
    def test_generate_telemetry_config(self, mock_get_hostname_ip):
        mock_get_hostname_ip.side_effect = lambda host, interface=None: {"node-a": "10.0.0.1", "node-b": "10.0.0.2"}[
            host
        ]

        telemetry = TelemetryConfig(
            enabled=True,
            container_image="telemetry:latest",
            extra_metadata={"cluster": "pdx"},
            dcgm_exporter=TelemetryExporterConfig(container_image="dcgm:latest", port=9401),
            node_exporter=TelemetryExporterConfig(container_image="node:latest", port=9101),
        )
        runtime = MagicMock()
        runtime.job_id = "12345"
        runtime.run_name = "test_12345"
        runtime.network_interface = "eth0"
        processes = [
            Process(
                node="node-a",
                gpu_indices=frozenset({0, 1}),
                sys_port=8081,
                http_port=30000,
                endpoint_mode="prefill",
                endpoint_index=0,
                node_rank=0,
            ),
            Process(
                node="node-b",
                gpu_indices=frozenset({0, 1}),
                sys_port=8082,
                http_port=30000,
                endpoint_mode="decode",
                endpoint_index=0,
                node_rank=0,
            ),
        ]
        topology = FrontendTopology(
            nginx_node=None,
            frontend_nodes=["node-a"],
            frontend_port=8000,
            public_port=8000,
        )

        config_text = generate_telemetry_config(
            processes=processes,
            frontend_topology=topology,
            runtime=runtime,
            telemetry=telemetry,
        )

        assert 'storage = "/logs/telemetry"' in config_text
        assert 'name = "dcgm_node-a"' in config_text
        assert 'url = "http://10.0.0.1:8081/metrics"' in config_text
        assert '"cluster" = "pdx"' in config_text
        assert 'name = "frontend0"' in config_text


class TestTelemetryStageMixin:
    """Telemetry stage startup."""

    @staticmethod
    def _make_harness(
        tmp_path: Path,
        *,
        worker_nodes: list[str],
        scraper: bool = True,
        node_exporter: bool = True,
    ) -> TelemetryStageMixin:
        telemetry_kwargs: dict[str, object] = {
            "enabled": True,
            "dcgm_exporter": TelemetryExporterConfig(container_image="dcgm:latest", port=9401),
        }
        if scraper:
            telemetry_kwargs["container_image"] = "telemetry:latest"
        if node_exporter:
            telemetry_kwargs["node_exporter"] = TelemetryExporterConfig(container_image="node:latest", port=9101)

        class Harness(TelemetryStageMixin):
            def __init__(self) -> None:
                self.config = _make_config(telemetry=TelemetryConfig(**telemetry_kwargs))
                self.runtime = MagicMock()
                self.runtime.job_id = "12345"
                self.runtime.log_dir = tmp_path
                self.runtime.nodes.head = "node-a"
                self.runtime.nodes.het = False
                self.runtime.srun_options = {}
                self.runtime.container_mounts = {Path(tmp_path): Path("/logs")}
                self._backend_processes = [
                    Process(
                        node=node,
                        gpu_indices=frozenset({0}),
                        sys_port=8081 + i,
                        http_port=30000,
                        endpoint_mode="agg",
                        endpoint_index=i,
                        node_rank=0,
                    )
                    for i, node in enumerate(worker_nodes)
                ]

            @property
            def backend_processes(self):
                return self._backend_processes

            def _compute_frontend_topology(self):
                return FrontendTopology(
                    nginx_node=None,
                    frontend_nodes=[worker_nodes[0]],
                    frontend_port=8000,
                    public_port=8000,
                )

        return Harness()

    @patch("srtctl.cli.mixins.telemetry_stage.start_srun_process")
    @patch(
        "srtctl.cli.mixins.telemetry_stage.generate_telemetry_config",
        return_value='storage = "/logs/telemetry"\n',
    )
    def test_start_telemetry_starts_exporters_and_scraper(self, _mock_config, mock_srun, tmp_path):
        mock_srun.return_value = MagicMock()
        harness = self._make_harness(tmp_path, worker_nodes=["node-a"])

        procs = harness.start_telemetry()

        assert len(procs) == 3
        config_path = tmp_path / "telemetry_config.toml"
        container_config_path = Path("/telemetry_config.toml")
        assert config_path.exists()
        assert (tmp_path / "telemetry" / "12345" / "local").exists()
        assert mock_srun.call_count == 3
        scraper_call = mock_srun.call_args_list[-1]
        assert scraper_call.kwargs["command"][2] == str(container_config_path)
        assert scraper_call.kwargs["command"][4] == "/logs/telemetry/12345/local"
        assert scraper_call.kwargs["container_mounts"][config_path] == container_config_path

    @patch("srtctl.cli.mixins.telemetry_stage.start_srun_process")
    def test_start_telemetry_dcgm_only(self, mock_srun, tmp_path):
        """With only dcgm_exporter configured, start dcgm and skip scraper/node-exporter."""
        mock_srun.return_value = MagicMock()
        harness = self._make_harness(tmp_path, worker_nodes=["node-a"], scraper=False, node_exporter=False)

        procs = harness.start_telemetry()

        assert len(procs) == 1
        assert procs[0].name == "telemetry_dcgm_exporter"
        assert not (tmp_path / "telemetry_config.toml").exists()
        assert mock_srun.call_count == 1

    @patch("srtctl.cli.mixins.telemetry_stage.start_srun_process")
    @patch(
        "srtctl.cli.mixins.telemetry_stage.generate_telemetry_config",
        return_value='storage = "/logs/telemetry"\n',
    )
    def test_exporter_srun_passes_nodes_arg_matching_nodelist(self, _mock_config, mock_srun, tmp_path):
        """Regression: exporter srun must pass nodes=N when nodelist has N entries.

        Without it, srun gets ``--nodes 1 --ntasks N --nodelist <N nodes>`` and
        SLURM rejects with 'Requested node configuration is not available'.
        """
        mock_srun.return_value = MagicMock()
        nodes = ["node-a", "node-b", "node-c", "node-d"]
        harness = self._make_harness(tmp_path, worker_nodes=nodes)

        harness.start_telemetry()

        exporter_calls = [call for call in mock_srun.call_args_list if call.kwargs.get("nodelist") == nodes]
        assert len(exporter_calls) == 2
        for call in exporter_calls:
            assert call.kwargs.get("nodes") == 4
            assert call.kwargs.get("ntasks") == 4
            # Distroless images (e.g. prom/node-exporter) have no bash, so we
            # must invoke the binary directly rather than via `bash -c`.
            assert call.kwargs.get("use_bash_wrapper") is False
