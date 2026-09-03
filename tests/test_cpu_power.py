# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ACPI host CPU power leg: schema, lifecycle, and AIPerf wiring."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from marshmallow import ValidationError

from srtctl.cli.mixins.telemetry_stage import TelemetryStageMixin
from srtctl.core.power.contract import Reason
from srtctl.core.power.cpu_session import parse_cpu_power_scrape
from srtctl.core.processes import ProcessRegistry
from srtctl.core.schema import (
    BenchmarkConfig,
    CpuPowerConfig,
    ModelConfig,
    ResourceConfig,
    SrtConfig,
    TelemetryConfig,
    TelemetryExporterConfig,
)
from srtctl.core.topology import Process

REPO_ROOT = Path(__file__).resolve().parents[1]


def _config(
    cpu_power: CpuPowerConfig,
    *,
    telemetry_enabled: bool = True,
    benchmark: BenchmarkConfig | None = None,
    **telemetry_kwargs,
) -> SrtConfig:
    return SrtConfig(
        name="test",
        model=ModelConfig(path="/model", container="/image", precision="fp4"),
        resources=ResourceConfig(gpu_type="gb200"),
        benchmark=benchmark or BenchmarkConfig(type="manual"),
        telemetry=TelemetryConfig(enabled=telemetry_enabled, cpu_power=cpu_power, **telemetry_kwargs),
    )


class TestCpuPowerSchema:
    """The leg is configurable, defaulted off, and validated when it is on."""

    def test_defaults_are_off_and_best_effort(self):
        cpu_power = _config(CpuPowerConfig(), telemetry_enabled=False).telemetry.cpu_power

        assert cpu_power.enabled is False
        assert cpu_power.source == "auto"
        assert cpu_power.required is False
        assert cpu_power.acpi_mandatory is False

    @pytest.mark.parametrize(
        ("source", "required", "mandatory"),
        [("auto", False, False), ("auto", True, True), ("acpi", False, True), ("acpi", True, True)],
    )
    def test_acpi_is_mandatory_when_named_or_required(self, source, required, mandatory):
        """``auto`` alone is best-effort; naming acpi or asking for required is not."""
        cpu_power = CpuPowerConfig(enabled=True, source=source, required=required)

        assert _config(cpu_power).telemetry.cpu_power.acpi_mandatory is mandatory

    def test_schema_round_trip_preserves_every_field(self):
        payload = {
            "enabled": True,
            "source": "acpi",
            "sample_interval_seconds": 0.25,
            "startup_timeout_seconds": 15.0,
            "required": True,
            "prometheus_port": 9411,
            "storage_subdir": "cpu",
        }

        loaded = CpuPowerConfig.Schema().load(payload)

        assert CpuPowerConfig.Schema().dump(loaded) == payload

    def test_the_shipped_recipe_loads(self):
        """configs/cpu-power-test.yaml is the leg's end-to-end reference."""
        config = SrtConfig.from_yaml(REPO_ROOT / "configs" / "cpu-power-test.yaml")

        assert config.telemetry.cpu_power.enabled is True
        assert config.telemetry.cpu_power.acpi_mandatory is True
        assert config.benchmark.type == "custom"
        assert "AIPERF_SERVER_METRICS_URLS" in config.benchmark.command

    def test_enabling_the_leg_requires_telemetry(self):
        with pytest.raises(ValidationError, match="requires telemetry.enabled"):
            _config(CpuPowerConfig(enabled=True), telemetry_enabled=False)

    def test_required_without_enabled_is_rejected(self):
        with pytest.raises(ValidationError, match="no effect"):
            _config(CpuPowerConfig(required=True))

    @pytest.mark.parametrize("port", [0, 65536, -1])
    def test_port_must_be_in_range(self, port):
        with pytest.raises(ValidationError, match="prometheus_port"):
            _config(CpuPowerConfig(enabled=True, prometheus_port=port))

    def test_port_may_not_collide_with_the_dcgm_exporter(self):
        """Both exporters run on every worker node, so one port cannot serve both."""
        with pytest.raises(ValidationError, match="collides"):
            _config(
                CpuPowerConfig(enabled=True, prometheus_port=9400),
                benchmark=BenchmarkConfig(type="sa-bench", concurrencies=[4], client_placement="head"),
                dcgm_exporter=TelemetryExporterConfig(container_image="dcgm", port=9400),
            )

    @pytest.mark.parametrize("interval", [0.0, -1.0, float("inf"), float("nan")])
    def test_sample_interval_must_be_finite_and_positive(self, interval):
        with pytest.raises(ValidationError, match="sample_interval_seconds"):
            _config(CpuPowerConfig(enabled=True, sample_interval_seconds=interval))

    def test_sample_interval_may_not_exceed_the_max_sample_gap(self):
        with pytest.raises(ValidationError, match="max sample gap"):
            _config(CpuPowerConfig(enabled=True, sample_interval_seconds=5.0))

    def test_startup_timeout_must_be_finite_and_positive(self):
        with pytest.raises(ValidationError, match="startup_timeout_seconds"):
            _config(CpuPowerConfig(enabled=True, startup_timeout_seconds=0.0))

    @pytest.mark.parametrize("subdir", ["/abs", "../escape", "a/../../b", ""])
    def test_storage_subdir_must_stay_below_the_log_dir(self, subdir):
        with pytest.raises(ValidationError, match="storage_subdir"):
            _config(CpuPowerConfig(enabled=True, storage_subdir=subdir))

    def test_storage_subdir_may_not_shadow_the_dcgm_leg(self):
        with pytest.raises(ValidationError, match="must differ"):
            _config(CpuPowerConfig(enabled=True, storage_subdir="power"), storage_subdir="power")


def _scrape(*lines: str) -> str:
    header = "# HELP cpu_power_acpi_watts Host CPU rail power.\n# TYPE cpu_power_acpi_watts gauge\n"
    return header + "".join(f"{line}\n" for line in lines)


class TestCpuPowerScrapeParsing:
    """The sensor label is the rail identity; anything else is descriptive."""

    def test_distinct_sensors_are_kept(self):
        readings, reasons = parse_cpu_power_scrape(
            _scrape(
                'cpu_power_acpi_watts{sensor="hwmon0/power1",type="CPU",socket="0",oem_info=""} 42.5',
                'cpu_power_acpi_watts{sensor="hwmon1/power1",type="CPU",socket="1",oem_info=""} 43.5',
            )
        )

        assert reasons == ()
        assert [(r.sensor, r.socket, r.power_w) for r in readings] == [
            ("hwmon0/power1", "0", 42.5),
            ("hwmon1/power1", "1", 43.5),
        ]

    def test_duplicate_sensors_are_dropped_and_reported(self):
        """A repeated series is ambiguous: neither copy can be trusted."""
        readings, reasons = parse_cpu_power_scrape(
            _scrape(
                'cpu_power_acpi_watts{sensor="hwmon0/power1",type="CPU",socket="0"} 42.5',
                'cpu_power_acpi_watts{sensor="hwmon0/power1",type="CPU",socket="0"} 51.0',
                'cpu_power_acpi_watts{sensor="hwmon1/power1",type="CPU",socket="1"} 43.5',
            )
        )

        assert Reason.DUPLICATE_POWER_METRIC in reasons
        assert [r.sensor for r in readings] == ["hwmon1/power1"]

    def test_unlabelled_sample_is_reported(self):
        readings, reasons = parse_cpu_power_scrape(_scrape("cpu_power_acpi_watts 42.5"))

        assert readings == ()
        assert Reason.CPU_SENSOR_MISSING in reasons

    def test_negative_power_is_rejected(self):
        readings, reasons = parse_cpu_power_scrape(_scrape('cpu_power_acpi_watts{sensor="hwmon0/power1"} -1.0'))

        assert readings == ()
        assert Reason.INVALID_POWER_VALUE in reasons

    def test_an_exporter_with_no_rails_is_reported(self):
        readings, reasons = parse_cpu_power_scrape("# HELP other Other.\n# TYPE other gauge\nother 1\n")

        assert readings == ()
        assert Reason.CPU_POWER_METRIC_MISSING in reasons

    def test_malformed_exposition_is_reported(self):
        readings, reasons = parse_cpu_power_scrape("# TYPE cpu_power_acpi_watts gauge\ncpu_power_acpi_watts{ 1\n")

        assert readings == ()
        assert Reason.ENDPOINT_PARSE_ERROR in reasons


def _running_exporter():
    proc = MagicMock()
    proc.poll.return_value = None
    return proc


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


def _harness(tmp_path, processes, *, cpu_power=None, het=False, het_groups=None):
    cpu_power = cpu_power or CpuPowerConfig(enabled=True, source="acpi", required=True, startup_timeout_seconds=0.2)

    class Harness(TelemetryStageMixin):
        def __init__(self):
            # NOTE: srun is mocked so no exporter answers; a short deadline avoids a stall per test.
            self.config = _config(
                cpu_power,
                # A DCGM exporter keeps telemetry.enabled valid when the CPU leg is off.
                benchmark=BenchmarkConfig(type="sa-bench", concurrencies=[4], client_placement="head"),
                dcgm_exporter=TelemetryExporterConfig(container_image="dcgm", port=9400),
                request_timeout_seconds=0.1,
                collector_join_timeout_seconds=3.0,
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
            self._backend_processes = processes

        @property
        def backend_processes(self):
            return self._backend_processes

    return Harness()


class TestCpuPowerLifecycle:
    """Exporter launch, readiness gating, and finalization."""

    def test_disabled_leg_starts_nothing(self, tmp_path):
        harness = _harness(tmp_path, [_worker("node-a", range(4))], cpu_power=CpuPowerConfig())
        registry = ProcessRegistry(job_id="12345")

        assert harness.start_cpu_power_telemetry(registry) is None
        assert registry.process_count == 0
        assert harness.cpu_power_telemetry_blocks_benchmark() is False
        assert harness.finalize_cpu_power_telemetry(0) == 0

    @patch("srtctl.cli.mixins.telemetry_stage.start_srun_process")
    def test_one_containerless_task_per_worker_node(self, mock_srun, tmp_path):
        """The exporter reads the node's own /sys, so it runs outside the container."""
        mock_srun.return_value = _running_exporter()
        harness = _harness(
            tmp_path,
            [_worker("node-a", range(4)), _worker("node-a", range(4), index=1), _worker("node-b", range(4), index=2)],
        )
        registry = ProcessRegistry(job_id="12345")

        session = harness.start_cpu_power_telemetry(registry)

        assert mock_srun.call_count == 1
        kwargs = mock_srun.call_args.kwargs
        assert kwargs["nodelist"] == ["node-a", "node-b"]
        assert kwargs["nodes"] == 2
        assert kwargs["ntasks"] == 2
        assert kwargs["use_bash_wrapper"] is False
        assert "container_image" not in kwargs
        assert kwargs["command"][1:] == ["--port", "9401"]
        assert registry.process_count == 1
        assert all(proc.critical is False for proc in registry.get_all_processes().values())
        session.stop_and_finalize()

    @patch("srtctl.cli.mixins.telemetry_stage.start_srun_process")
    def test_heterogeneous_groups_launch_once_per_group(self, mock_srun, tmp_path):
        mock_srun.return_value = _running_exporter()
        harness = _harness(
            tmp_path,
            [
                _worker("node-a", range(4), mode="prefill", het_group=0),
                _worker("node-b", range(4), mode="decode", het_group=1),
            ],
            het=True,
            het_groups={"node-a": 0, "node-b": 1},
        )
        registry = ProcessRegistry(job_id="12345")

        session = harness.start_cpu_power_telemetry(registry)

        assert [(c.kwargs["nodelist"], c.kwargs["het_group"]) for c in mock_srun.call_args_list] == [
            (["node-a"], 0),
            (["node-b"], 1),
        ]
        assert registry.process_count == 2
        session.stop_and_finalize()

    @patch("srtctl.cli.mixins.telemetry_stage.start_srun_process")
    def test_launch_failure_is_session_state_not_an_exception(self, mock_srun, tmp_path):
        mock_srun.side_effect = RuntimeError("srun refused")
        harness = _harness(tmp_path, [_worker("node-a", range(4))])

        session = harness.start_cpu_power_telemetry(ProcessRegistry(job_id="12345"))
        outcome = session.stop_and_finalize()

        assert Reason.EXPORTER_LAUNCH_FAILED in outcome.reason_codes
        assert outcome.exit_nonzero is True

    @patch("srtctl.cli.mixins.telemetry_stage.start_srun_process")
    def test_required_mode_blocks_the_benchmark_when_startup_fails(self, mock_srun, tmp_path):
        """Running the workload without collection would burn the allocation."""
        mock_srun.return_value = _running_exporter()
        harness = _harness(tmp_path, [_worker("node-a", range(4))])

        session = harness.start_cpu_power_telemetry(ProcessRegistry(job_id="12345"))

        assert harness.cpu_power_telemetry_blocks_benchmark() is True
        assert harness.finalize_cpu_power_telemetry(0) == 1
        assert session is harness._cpu_power_session

    @patch("srtctl.cli.mixins.telemetry_stage.start_srun_process")
    def test_best_effort_mode_does_not_block_or_fail(self, mock_srun, tmp_path):
        """``source: auto`` on a cluster with no ACPI rails still runs the sweep."""
        mock_srun.return_value = _running_exporter()
        harness = _harness(
            tmp_path,
            [_worker("node-a", range(4))],
            cpu_power=CpuPowerConfig(enabled=True, startup_timeout_seconds=0.2),
        )

        harness.start_cpu_power_telemetry(ProcessRegistry(job_id="12345"))

        assert harness.cpu_power_telemetry_blocks_benchmark() is False
        assert harness.finalize_cpu_power_telemetry(0) == 0

    @patch("srtctl.core.power.cpu_session.requests.get")
    @patch("srtctl.core.power.cpu_session.get_hostname_ip", side_effect=lambda node, _iface: f"ip-{node}")
    @patch("srtctl.cli.mixins.telemetry_stage.start_srun_process")
    def test_a_serving_exporter_reaches_readiness_and_publishes(self, mock_srun, _mock_ip, mock_get, tmp_path):
        mock_srun.return_value = _running_exporter()
        mock_get.return_value = SimpleNamespace(
            text=_scrape('cpu_power_acpi_watts{sensor="hwmon0/power1",type="CPU",socket="0"} 42.5'),
            raise_for_status=lambda: None,
        )
        harness = _harness(tmp_path, [_worker("node-a", range(4))])

        harness.start_cpu_power_telemetry(ProcessRegistry(job_id="12345"))

        assert harness.cpu_power_telemetry_blocks_benchmark() is False
        assert harness.finalize_cpu_power_telemetry(0) == 0
        assert mock_get.call_args.args[0] == "http://ip-node-a:9401/metrics"

        manifest = json.loads((tmp_path / "cpu_power" / "manifest.json").read_text())
        assert manifest["status"] == "complete"
        assert manifest["publication_valid"] is True
        assert manifest["exporter"]["port"] == 9401
        assert manifest["exporter"]["command"].endswith("--port 9401")
        assert manifest["observed_sensors"] == {"node-a": ["hwmon0/power1"]}
        rows = (tmp_path / "cpu_power" / "samples.csv").read_text().splitlines()
        assert rows[0].startswith("schema_version,")
        assert any("hwmon0/power1" in row for row in rows[1:])


class TestCpuPowerServerMetricsUrls:
    """CPU exporters are advertised to AIPerf alongside the engine endpoints."""

    @staticmethod
    def _stage(processes, *, frontend_type="dynamo", cpu_power=None):
        from srtctl.cli.mixins.benchmark_stage import BenchmarkStageMixin

        class Stage(BenchmarkStageMixin):
            @property
            def backend_processes(self):
                return processes

        stage = Stage()
        stage.config = SimpleNamespace(
            benchmark=SimpleNamespace(type="custom", aiperf_package=None),
            backend=SimpleNamespace(type="sglang", prefill_environment={}, aggregated_environment={}),
            backend_type="sglang",
            frontend=SimpleNamespace(type=frontend_type),
            telemetry=SimpleNamespace(
                enabled=cpu_power is not None,
                cpu_power=cpu_power or CpuPowerConfig(),
            ),
        )
        stage.runtime = SimpleNamespace(network_interface="eth0")
        return stage

    def _urls(self, stage):
        with patch(
            "srtctl.cli.mixins.benchmark_stage.get_hostname_ip",
            side_effect=lambda node, _iface: f"ip-{node}",
        ):
            env = stage._get_aiperf_server_metrics_env()
        return env.get("AIPERF_SERVER_METRICS_URLS", "").split(",") if env else []

    def test_one_cpu_url_per_worker_node(self):
        stage = self._stage(
            [_worker("node-a", range(4)), _worker("node-a", range(4), index=1), _worker("node-b", range(4), index=2)],
            cpu_power=CpuPowerConfig(enabled=True),
        )

        assert "http://ip-node-a:9401/metrics" in self._urls(stage)
        assert "http://ip-node-b:9401/metrics" in self._urls(stage)

    def test_vllm_frontend_still_gets_the_cpu_urls(self):
        """The vLLM branch used to return early, dropping the CPU endpoints."""
        process = _worker("node-a", range(4))
        stage = self._stage([process], frontend_type="vllm", cpu_power=CpuPowerConfig(enabled=True))

        urls = self._urls(stage)

        assert "http://ip-node-a:9401/metrics" in urls
        assert any(url.endswith("8000/metrics") for url in urls)

    def test_disabled_leg_adds_no_cpu_urls(self):
        stage = self._stage([_worker("node-a", range(4))])

        assert all(":9401/" not in url for url in self._urls(stage))
