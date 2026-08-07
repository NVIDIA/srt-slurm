# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for telemetry configuration and startup."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from marshmallow import ValidationError

from srtctl.cli.mixins.frontend_stage import FrontendTopology
from srtctl.cli.mixins.telemetry_stage import TelemetryStageMixin
from srtctl.core.power.contract import Reason
from srtctl.core.processes import ProcessRegistry
from srtctl.core.schema import (
    BenchmarkConfig,
    InfraConfig,
    ModelConfig,
    ResourceConfig,
    SrtConfig,
    TelemetryConfig,
    TelemetryExporterConfig,
    TelemetryProvider,
)
from srtctl.core.telemetry import generate_telemetry_config
from srtctl.core.topology import Process


def _make_config(*, telemetry: TelemetryConfig | None = None, benchmark: BenchmarkConfig | None = None) -> SrtConfig:
    return SrtConfig(
        name="test",
        model=ModelConfig(path="/model", container="/image", precision="fp4"),
        resources=ResourceConfig(gpu_type="h100"),
        benchmark=benchmark or BenchmarkConfig(type="manual"),
        telemetry=telemetry or TelemetryConfig(),
    )


def _sa_bench(**overrides) -> BenchmarkConfig:
    return BenchmarkConfig(type="sa-bench", concurrencies=[4], client_placement="head", **overrides)


def _dcgm_power(**overrides) -> TelemetryConfig:
    fields: dict = {
        "enabled": True,
        "provider": TelemetryProvider.DCGM_POWER,
        "default_frequency": 1.0,
        "storage_subdir": "power",
        "required": True,
        "startup_timeout_seconds": 30.0,
        "request_timeout_seconds": 2.0,
        "collector_join_timeout_seconds": 10.0,
        "dcgm_exporter": TelemetryExporterConfig(container_image="dcgm-exporter", port=9401),
    }
    fields.update(overrides)
    return TelemetryConfig(**fields)


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


class TestDcgmPowerConfig:
    """dcgm-power provider validation is independent of the scraper provider."""

    def test_accepts_dcgm_exporter_only(self):
        config = _make_config(telemetry=_dcgm_power(), benchmark=_sa_bench())

        assert config.telemetry.provider == TelemetryProvider.DCGM_POWER
        assert config.telemetry.container_image is None
        assert config.telemetry.node_exporter is None

    def test_defaults_are_stable(self):
        defaults = TelemetryConfig()

        assert defaults.provider == TelemetryProvider.SCRAPER
        assert defaults.required is False
        assert defaults.startup_timeout_seconds == 30.0
        assert defaults.request_timeout_seconds == 2.0
        assert defaults.collector_join_timeout_seconds == 10.0

    def test_scraper_validation_is_unchanged(self):
        with pytest.raises(ValidationError, match="telemetry.node_exporter"):
            _make_config(
                telemetry=TelemetryConfig(
                    enabled=True,
                    container_image="scraper:latest",
                    dcgm_exporter=TelemetryExporterConfig(container_image="dcgm:latest", port=9401),
                )
            )

    @pytest.mark.parametrize(
        ("telemetry_overrides", "benchmark", "match"),
        [
            ({"dcgm_exporter": None}, None, "telemetry.dcgm_exporter"),
            (
                {"dcgm_exporter": TelemetryExporterConfig(container_image="", port=9401)},
                None,
                "container_image",
            ),
            (
                {"dcgm_exporter": TelemetryExporterConfig(container_image="dcgm", port=0)},
                None,
                "port",
            ),
            (
                {"dcgm_exporter": TelemetryExporterConfig(container_image="dcgm", port=70000)},
                None,
                "port",
            ),
            ({"default_frequency": 0.0}, None, "default_frequency"),
            ({"default_frequency": float("nan")}, None, "default_frequency"),
            ({"default_frequency": float("inf")}, None, "default_frequency"),
            ({"default_frequency": 3.5}, None, "sample_gap_exceeded"),
            ({"startup_timeout_seconds": 0.0}, None, "startup_timeout_seconds"),
            ({"request_timeout_seconds": -1.0}, None, "request_timeout_seconds"),
            ({"collector_join_timeout_seconds": 2.0}, None, "collector_join_timeout_seconds"),
            ({"storage_subdir": "/abs"}, None, "storage_subdir"),
            ({"storage_subdir": "../escape"}, None, "storage_subdir"),
            ({"storage_subdir": ""}, None, "storage_subdir"),
            ({}, BenchmarkConfig(type="manual"), "benchmark.type"),
            ({}, BenchmarkConfig(type="sa-bench", concurrencies=None), "benchmark.concurrencies"),
            ({}, BenchmarkConfig(type="sa-bench", concurrencies=[4, 4]), "benchmark.concurrencies"),
            ({}, BenchmarkConfig(type="sa-bench", concurrencies=[0]), "benchmark.concurrencies"),
            (
                {},
                BenchmarkConfig(type="sa-bench", concurrencies=[4], client_placement="last_decode"),
                "benchmark.client_placement",
            ),
        ],
    )
    def test_invalid_configurations_are_rejected(self, telemetry_overrides, benchmark, match):
        with pytest.raises(ValidationError, match=match):
            _make_config(
                telemetry=_dcgm_power(**telemetry_overrides),
                benchmark=benchmark or _sa_bench(),
            )

    def test_the_shared_default_frequency_is_rejected_for_dcgm_power(self):
        telemetry = TelemetryConfig(
            enabled=True,
            provider=TelemetryProvider.DCGM_POWER,
            storage_subdir="power",
            required=True,
            startup_timeout_seconds=30.0,
            request_timeout_seconds=2.0,
            collector_join_timeout_seconds=10.0,
            dcgm_exporter=TelemetryExporterConfig(container_image="dcgm-exporter", port=9401),
        )
        assert telemetry.default_frequency == 5.0
        with pytest.raises(ValidationError, match="sample_gap_exceeded"):
            _make_config(telemetry=telemetry, benchmark=_sa_bench())

    def test_schema_constant_copies_match_the_power_contract(self):
        from srtctl.core import schema as schema_module
        from srtctl.core.power import contract

        assert schema_module._BENCHMARK_TYPE_SA_BENCH == contract.BENCHMARK_TYPE_SA_BENCH
        assert schema_module._DCGM_POWER_MAX_SAMPLE_GAP_SECONDS == contract.MAX_SAMPLE_GAP_SECONDS

    @pytest.mark.parametrize(
        ("telemetry", "benchmark", "dedicated", "rejected"),
        [
            (_dcgm_power(), _sa_bench(), True, True),
            (_dcgm_power(), _sa_bench(), False, False),
            (
                TelemetryConfig(
                    enabled=True,
                    container_image="scraper:latest",
                    dcgm_exporter=TelemetryExporterConfig(container_image="dcgm:latest", port=9401),
                    node_exporter=TelemetryExporterConfig(container_image="node:latest", port=9101),
                ),
                BenchmarkConfig(type="manual"),
                True,
                False,
            ),
        ],
        ids=["dcgm-power-dedicated", "dcgm-power-shared", "scraper-dedicated"],
    )
    def test_a_dedicated_infra_node_is_rejected_only_for_dcgm_power(self, telemetry, benchmark, dedicated, rejected):
        def build():
            return SrtConfig(
                name="test",
                model=ModelConfig(path="/model", container="/image", precision="fp4"),
                resources=ResourceConfig(gpu_type="h100"),
                benchmark=benchmark,
                telemetry=telemetry,
                infra=InfraConfig(etcd_nats_dedicated_node=dedicated),
            )

        if rejected:
            with pytest.raises(ValidationError, match="etcd_nats_dedicated_node"):
                build()
            return

        assert build().infra.etcd_nats_dedicated_node is dedicated

    def test_unknown_provider_is_rejected(self):
        with pytest.raises(ValidationError, match="Unsupported telemetry provider"):
            _make_config(telemetry=TelemetryConfig(enabled=True, provider="mystery"))


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

    @patch("srtctl.cli.mixins.telemetry_stage.start_srun_process")
    @patch("srtctl.cli.mixins.telemetry_stage.generate_telemetry_config", return_value='storage = "/logs/telemetry"\n')
    def test_start_telemetry_starts_exporters_and_scraper(self, _mock_config, mock_srun, tmp_path):
        class Harness(TelemetryStageMixin):
            def __init__(self):
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
                self.runtime.nodes.head = "node-a"
                self.runtime.nodes.het = False
                self.runtime.srun_options = {}
                self.runtime.container_mounts = {Path(tmp_path): Path("/logs")}
                self._backend_processes = [
                    Process(
                        node="node-a",
                        gpu_indices=frozenset({0}),
                        sys_port=8081,
                        http_port=30000,
                        endpoint_mode="agg",
                        endpoint_index=0,
                        node_rank=0,
                    )
                ]

            @property
            def backend_processes(self):
                return self._backend_processes

            def _compute_frontend_topology(self):
                return FrontendTopology(
                    nginx_node=None,
                    frontend_nodes=["node-a"],
                    frontend_port=8000,
                    public_port=8000,
                )

        mock_srun.return_value = _running_exporter()
        harness = Harness()

        procs = harness.start_telemetry()

        assert len(procs) == 3
        assert (tmp_path / "telemetry_config.toml").exists()
        assert (tmp_path / "telemetry" / "local").exists()
        assert mock_srun.call_count == 3

    @patch("srtctl.cli.mixins.telemetry_stage.start_srun_process")
    @patch("srtctl.cli.mixins.telemetry_stage.generate_telemetry_config", return_value='storage = "/logs/telemetry"\n')
    def test_multinode_exporters_request_one_node_per_task(self, _mock_config, mock_srun, tmp_path):
        """srun rejects --nodes 1 with a longer --nodelist, so the exporter launch
        must size --nodes to the worker set."""

        class Harness(TelemetryStageMixin):
            def __init__(self):
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
                self.runtime.nodes.head = "node-a"
                self.runtime.nodes.het = False
                self.runtime.srun_options = {}
                self.runtime.container_mounts = {Path(tmp_path): Path("/logs")}
                self._backend_processes = [
                    Process(
                        node=node,
                        gpu_indices=frozenset({0}),
                        sys_port=8081,
                        http_port=30000,
                        endpoint_mode="agg",
                        endpoint_index=index,
                        node_rank=index,
                    )
                    for index, node in enumerate(["node-a", "node-b"])
                ]

            @property
            def backend_processes(self):
                return self._backend_processes

            def _compute_frontend_topology(self):
                return FrontendTopology(
                    nginx_node=None,
                    frontend_nodes=["node-a"],
                    frontend_port=8000,
                    public_port=8000,
                )

        mock_srun.return_value = _running_exporter()
        harness = Harness()

        harness.start_telemetry()

        exporter_calls = [call for call in mock_srun.call_args_list if call.kwargs.get("nodelist") == ["node-a", "node-b"]]
        assert len(exporter_calls) == 2
        for call in exporter_calls:
            assert call.kwargs["nodes"] == 2
            assert call.kwargs["ntasks"] == 2


def _running_exporter():
    """A just-launched srun process: still running, so poll() is None."""
    proc = MagicMock()
    proc.poll.return_value = None
    return proc


def _power_harness(tmp_path, processes, *, het=False, het_groups=None):
    class Harness(TelemetryStageMixin):
        def __init__(self):
            # NOTE: srun is mocked so no exporter answers; a short deadline avoids a 30s stall per test.
            self.config = _make_config(
                telemetry=_dcgm_power(
                    startup_timeout_seconds=0.2,
                    request_timeout_seconds=0.1,
                    collector_join_timeout_seconds=1.0,
                ),
                benchmark=_sa_bench(),
            )
            self.runtime = MagicMock()
            self.runtime.log_dir = tmp_path
            self.runtime.job_id = "12345"
            self.runtime.run_name = "recipe_12345"
            self.runtime.network_interface = "eth0"
            self.runtime.nodes.head = "node-a"
            self.runtime.nodes.het = het
            self.runtime.nodes.het_group_for.side_effect = lambda node: (het_groups or {}).get(node)
            self.runtime.srun_options = {}
            self.runtime.container_mounts = {Path(tmp_path): Path("/logs")}
            self._backend_processes = processes

        @property
        def backend_processes(self):
            return self._backend_processes

        def _compute_frontend_topology(self):
            raise AssertionError("dcgm-power must not build a scraper topology")

    return Harness()


def _worker(node, gpus, mode="agg", index=0, het_group=None):
    return Process(
        node=node,
        gpu_indices=frozenset(gpus),
        sys_port=8081,
        http_port=30000,
        endpoint_mode=mode,
        endpoint_index=index,
        node_rank=0,
        het_group=het_group,
    )


class TestDcgmPowerExporterLaunch:
    """One exporter task per allocated physical node, owned before the next launch."""

    @patch("srtctl.cli.mixins.telemetry_stage.start_srun_process")
    def test_single_node_launches_one_task_without_a_bash_wrapper(self, mock_srun, tmp_path):
        mock_srun.return_value = _running_exporter()
        harness = _power_harness(tmp_path, [_worker("node-a", range(4))])
        registry = ProcessRegistry(job_id="12345")

        session = harness.start_power_telemetry(registry)

        assert mock_srun.call_count == 1
        kwargs = mock_srun.call_args.kwargs
        assert kwargs["nodes"] == 1
        assert kwargs["ntasks"] == 1
        assert kwargs["nodelist"] == ["node-a"]
        assert kwargs["use_bash_wrapper"] is False
        assert kwargs["container_image"] == "dcgm-exporter"
        assert "--address :9401" in " ".join(kwargs["command"])
        assert registry.process_count == 1
        assert all(proc.critical is False for proc in registry.get_all_processes().values())
        session.stop_and_finalize()

    @patch("srtctl.cli.mixins.telemetry_stage.start_srun_process")
    def test_manifest_records_the_command_that_actually_ran(self, mock_srun, tmp_path):
        """A custom exporter command must not be misreported as the default."""
        mock_srun.return_value = _running_exporter()
        harness = _power_harness(tmp_path, [_worker("node-a", range(4))])
        harness.config = _make_config(
            telemetry=_dcgm_power(
                startup_timeout_seconds=0.2,
                request_timeout_seconds=0.1,
                collector_join_timeout_seconds=1.0,
                dcgm_exporter=TelemetryExporterConfig(
                    container_image="dcgm-exporter",
                    port=9401,
                    command="dcgm-exporter --collect-interval=50 --address :{port} --kubernetes=false",
                ),
            ),
            benchmark=_sa_bench(),
        )

        session = harness.start_power_telemetry(ProcessRegistry(job_id="12345"))
        session.stop_and_finalize()

        launched = " ".join(mock_srun.call_args.kwargs["command"])
        recorded = json.loads((tmp_path / "power" / "manifest.json").read_text())["dcgm_exporter"]["command"]
        assert launched == recorded
        assert "--collect-interval=50" in recorded
        assert "--address :9401" in recorded

    @patch("srtctl.cli.mixins.telemetry_stage.start_srun_process")
    def test_two_nodes_launch_two_tasks_in_one_srun(self, mock_srun, tmp_path):
        mock_srun.return_value = _running_exporter()
        harness = _power_harness(tmp_path, [_worker("node-a", range(4)), _worker("node-b", range(4), index=1)])

        session = harness.start_power_telemetry(ProcessRegistry(job_id="12345"))

        kwargs = mock_srun.call_args.kwargs
        assert mock_srun.call_count == 1
        assert kwargs["nodes"] == 2
        assert kwargs["ntasks"] == 2
        assert kwargs["nodelist"] == ["node-a", "node-b"]
        session.stop_and_finalize()

    @patch("srtctl.cli.mixins.telemetry_stage.start_srun_process")
    def test_duplicate_processes_on_one_node_launch_one_exporter(self, mock_srun, tmp_path):
        mock_srun.return_value = _running_exporter()
        harness = _power_harness(
            tmp_path,
            [_worker("node-a", [0, 1], index=0), _worker("node-a", [2, 3], index=1)],
        )

        session = harness.start_power_telemetry(ProcessRegistry(job_id="12345"))

        assert mock_srun.call_count == 1
        assert mock_srun.call_args.kwargs["nodelist"] == ["node-a"]
        session.stop_and_finalize()

    @patch("srtctl.cli.mixins.telemetry_stage.start_srun_process")
    def test_heterogeneous_groups_launch_once_per_group(self, mock_srun, tmp_path):
        mock_srun.return_value = _running_exporter()
        harness = _power_harness(
            tmp_path,
            [
                _worker("node-a", range(4), mode="prefill", het_group=0),
                _worker("node-b", range(4), mode="decode", het_group=1),
            ],
            het=True,
            het_groups={"node-a": 0, "node-b": 1},
        )
        registry = ProcessRegistry(job_id="12345")

        session = harness.start_power_telemetry(registry)

        assert mock_srun.call_count == 2
        launched = [(call.kwargs["nodelist"], call.kwargs["het_group"]) for call in mock_srun.call_args_list]
        assert launched == [(["node-a"], 0), (["node-b"], 1)]
        assert registry.process_count == 2
        session.stop_and_finalize()

    @patch("srtctl.cli.mixins.telemetry_stage.start_srun_process")
    def test_second_group_failure_leaves_the_first_group_owned(self, mock_srun, tmp_path):
        registry = ProcessRegistry(job_id="12345")
        owned_at_launch = []

        def record(*_args, **_kwargs):
            owned_at_launch.append(registry.process_count)
            if len(owned_at_launch) > 1:
                raise RuntimeError("srun refused")
            return _running_exporter()

        mock_srun.side_effect = record
        harness = _power_harness(
            tmp_path,
            [
                _worker("node-a", range(4), mode="prefill", het_group=0),
                _worker("node-b", range(4), mode="decode", het_group=1),
            ],
            het=True,
            het_groups={"node-a": 0, "node-b": 1},
        )

        session = harness.start_power_telemetry(registry)
        outcome = session.stop_and_finalize()

        assert owned_at_launch == [0, 1]
        assert registry.process_count == 1
        assert Reason.EXPORTER_LAUNCH_FAILED in outcome.reason_codes
        assert outcome.status == "failed"
        assert outcome.exit_nonzero is True
