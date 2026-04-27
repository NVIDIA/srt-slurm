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

    def test_requires_container_image_when_enabled(self):
        with pytest.raises(ValidationError, match="telemetry.container_image"):
            _make_config(
                telemetry=TelemetryConfig(
                    enabled=True,
                    dcgm_exporter=TelemetryExporterConfig(container_image="dcgm:latest", port=9401),
                    node_exporter=TelemetryExporterConfig(container_image="node:latest", port=9101),
                )
            )


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
            storage_path="/logs/telemetry/2026-04-27/test_12345",
        )

        assert 'storage = "/logs/telemetry/2026-04-27/test_12345"' in config_text
        assert 'name = "dcgm_node-a"' in config_text
        assert 'url = "http://10.0.0.1:8081/metrics"' in config_text
        assert '"cluster" = "pdx"' in config_text
        assert 'name = "frontend0"' in config_text


class TestTelemetryStageMixin:
    """Telemetry stage startup."""

    @staticmethod
    def _make_harness(tmp_path: Path, *, worker_nodes: list[str]) -> TelemetryStageMixin:
        class Harness(TelemetryStageMixin):
            def __init__(self) -> None:
                self.config = _make_config(
                    telemetry=TelemetryConfig(
                        enabled=True,
                        container_image="telemetry:latest",
                        dcgm_exporter=TelemetryExporterConfig(container_image="dcgm:latest", port=9401),
                        node_exporter=TelemetryExporterConfig(container_image="node:latest", port=9101),
                    )
                )
                self.runtime = MagicMock()
                self.runtime.log_dir = tmp_path
                self.runtime.nodes.head = worker_nodes[0]
                self.runtime.run_name = "test_12345"
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
        return_value='storage = "/logs/telemetry/2026-04-27/test_12345"\n',
    )
    def test_start_telemetry_starts_exporters_and_scraper(self, _mock_config, mock_srun, tmp_path):
        mock_srun.return_value = MagicMock()
        harness = self._make_harness(tmp_path, worker_nodes=["node-a"])

        procs = harness.start_telemetry()

        assert len(procs) == 3
        assert (tmp_path / "telemetry_config.toml").exists()
        # Local working dir is pre-created; storage dir is NOT (scraper rejects existing).
        assert (tmp_path / "telemetry" / "local").exists()
        assert not (tmp_path / "telemetry" / "2026-04-27").exists()
        assert mock_srun.call_count == 3

    @patch("srtctl.cli.mixins.telemetry_stage.start_srun_process")
    @patch(
        "srtctl.cli.mixins.telemetry_stage.generate_telemetry_config",
        return_value='storage = "/logs/telemetry/2026-04-27/test_12345"\n',
    )
    def test_exporter_srun_passes_nodes_arg_matching_nodelist(self, _mock_config, mock_srun, tmp_path):
        """Regression: exporter srun must pass nodes=N when nodelist has N entries.

        Without it, srun gets ``--nodes 1 --ntasks N --nodelist <N nodes>`` and
        SLURM rejects with 'Requested node configuration is not available'.
        """
        mock_srun.return_value = MagicMock()
        harness = self._make_harness(tmp_path, worker_nodes=["node-a", "node-b", "node-c", "node-d"])

        harness.start_telemetry()

        # Two exporter srun calls (dcgm + node_exporter) span all 4 worker nodes.
        exporter_calls = [
            call for call in mock_srun.call_args_list if call.kwargs.get("nodelist") == ["node-a", "node-b", "node-c", "node-d"]
        ]
        assert len(exporter_calls) == 2
        for call in exporter_calls:
            assert call.kwargs.get("nodes") == 4
            assert call.kwargs.get("ntasks") == 4
            # Distroless images (e.g. prom/node-exporter) have no bash, so we
            # must invoke the binary directly rather than via `bash -c`.
            assert call.kwargs.get("use_bash_wrapper") is False

    @patch("srtctl.cli.mixins.telemetry_stage.start_srun_process")
    @patch(
        "srtctl.cli.mixins.telemetry_stage.generate_telemetry_config",
        return_value='storage = "/logs/telemetry/2026-04-27/test_12345"\n',
    )
    def test_storage_path_is_nested_with_date_and_run_name(self, mock_config, mock_srun, tmp_path):
        """Regression: storage path passed to TOML generator must nest under <base>/<date>/<run-name>.

        Newer tachometer-scraper rejects pre-existing storage dirs and demands
        the nested form.
        """
        mock_srun.return_value = MagicMock()
        harness = self._make_harness(tmp_path, worker_nodes=["node-a"])
        harness.start_telemetry()

        storage_path = mock_config.call_args.kwargs["storage_path"]
        # /logs/telemetry/YYYY-MM-DD/test_12345
        assert storage_path.startswith("/logs/telemetry/")
        assert storage_path.endswith("/test_12345")
        date_segment = storage_path.split("/")[3]
        assert len(date_segment) == 10 and date_segment[4] == "-" and date_segment[7] == "-"
