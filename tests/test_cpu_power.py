# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ACPI host CPU power leg: schema, lifecycle, and AIPerf wiring."""

import json
import os
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from marshmallow import ValidationError

from srtctl.backends import BackendConfig, SGLangProtocol, TRTLLMProtocol
from srtctl.cli.mixins.telemetry_stage import TelemetryStageMixin
from srtctl.core.power.contract import Reason
from srtctl.core.power.cpu_session import (
    MAX_SCRAPE_BYTES,
    CpuEndpoint,
    CpuPowerSessionSettings,
    CpuPowerTelemetrySession,
    cpu_endpoint_address,
    fetch_metrics,
    parse_cpu_power_scrape,
)
from srtctl.core.processes import ProcessRegistry
from srtctl.core.schema import (
    BenchmarkConfig,
    CpuPowerConfig,
    FrontendConfig,
    ModelConfig,
    ObservabilityConfig,
    ResourceConfig,
    SrtConfig,
    TachometerConfig,
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
    observability: ObservabilityConfig | None = None,
    backend: BackendConfig | None = None,
    frontend: FrontendConfig | None = None,
    resources: ResourceConfig | None = None,
    **telemetry_kwargs,
) -> SrtConfig:
    return SrtConfig(
        name="test",
        model=ModelConfig(path="/model", container="/image", precision="fp4"),
        resources=resources or ResourceConfig(gpu_type="gb200"),
        backend=backend or SGLangProtocol(),
        frontend=frontend or FrontendConfig(),
        benchmark=benchmark or BenchmarkConfig(type="manual"),
        telemetry=TelemetryConfig(enabled=telemetry_enabled, cpu_power=cpu_power, **telemetry_kwargs),
        observability=observability or ObservabilityConfig(),
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

    @pytest.mark.parametrize("cpu_power", [CpuPowerConfig(required=True), CpuPowerConfig(source="acpi")])
    def test_mandatory_cpu_power_under_disabled_telemetry_is_rejected(self, cpu_power):
        """Accepting it would collect nothing and still pass the run it was meant to guard."""
        with pytest.raises(ValidationError, match="no effect"):
            _config(cpu_power, telemetry_enabled=False)

    def test_naming_acpi_without_enabling_the_leg_is_rejected(self):
        with pytest.raises(ValidationError, match="no effect"):
            _config(CpuPowerConfig(source="acpi"))

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

    @pytest.mark.parametrize("exporter", ["dcgm_exporter", "node_exporter"])
    def test_port_may_not_collide_with_a_tachometer_exporter(self, exporter):
        """Tachometer's exporters run on the same worker nodes as this one."""
        tachometer = TachometerConfig(
            enabled=True, **{exporter: TelemetryExporterConfig(container_image="img", port=9401)}
        )
        with pytest.raises(ValidationError, match=f"observability.tachometer.{exporter}"):
            _config(
                CpuPowerConfig(enabled=True, prometheus_port=9401),
                observability=ObservabilityConfig(enabled=True, tachometer=tachometer),
            )

    def test_port_may_not_collide_with_a_dynamo_system_port(self):
        """A Dynamo worker binds its assigned system-status port on the exporter node."""
        with pytest.raises(ValidationError, match="Dynamo system port"):
            _config(
                CpuPowerConfig(enabled=True, prometheus_port=7500),
                resources=ResourceConfig(gpu_type="gb200", agg_nodes=1, agg_workers=1),
            )

    def test_each_per_process_dynamo_system_port_is_checked(self):
        """Packed workers receive consecutive system ports on their shared node."""
        with pytest.raises(ValidationError, match="7501"):
            _config(
                CpuPowerConfig(enabled=True, prometheus_port=7501),
                resources=ResourceConfig(gpu_type="gb200", agg_nodes=1, agg_workers=2),
            )

    def test_an_unallocated_dynamo_system_port_is_allowed(self):
        config = _config(
            CpuPowerConfig(enabled=True, prometheus_port=7501),
            resources=ResourceConfig(gpu_type="gb200", agg_nodes=1, agg_workers=1),
        )

        assert config.telemetry.cpu_power.prometheus_port == 7501

    def test_native_sglang_does_not_reserve_dynamo_system_ports(self):
        config = _config(
            CpuPowerConfig(enabled=True, prometheus_port=7500),
            frontend=FrontendConfig(type="sglang"),
            resources=ResourceConfig(gpu_type="gb200", agg_nodes=1, agg_workers=1),
        )

        assert config.telemetry.cpu_power.prometheus_port == 7500

    def test_trtllm_endpoint_launch_uses_only_each_leaders_system_port(self):
        """MPI ranks inherit the endpoint leader's DYN_SYSTEM_PORT, not every topology value."""
        resources = ResourceConfig(
            gpu_type="gb200",
            gpus_per_node=4,
            prefill_nodes=2,
            prefill_workers=1,
            decode_nodes=2,
            decode_workers=1,
        )

        allowed = _config(
            CpuPowerConfig(enabled=True, prometheus_port=7501),
            backend=TRTLLMProtocol(),
            resources=resources,
        )
        assert allowed.telemetry.cpu_power.prometheus_port == 7501

        with pytest.raises(ValidationError, match="7502"):
            _config(
                CpuPowerConfig(enabled=True, prometheus_port=7502),
                backend=TRTLLMProtocol(),
                resources=resources,
            )

    def test_cpu_only_telemetry_leaves_tachometers_dcgm_exporter_alone(self):
        """Nothing conflicts: the CPU leg configures no DCGM exporter of its own."""
        config = _config(
            CpuPowerConfig(enabled=True),
            observability=ObservabilityConfig(
                enabled=True,
                tachometer=TachometerConfig(
                    enabled=True, dcgm_exporter=TelemetryExporterConfig(container_image="dcgm", port=9400)
                ),
            ),
        )

        assert config.observability.tachometer.dcgm_exporter is not None

    @pytest.mark.parametrize("timeout", [0.0, -1.0, float("inf"), float("nan")])
    def test_cpu_only_telemetry_still_validates_the_scrape_timeout(self, timeout):
        """cpu_session polls with it, so a DCGM-less recipe needs it checked too."""
        with pytest.raises(ValidationError, match="request_timeout_seconds"):
            _config(CpuPowerConfig(enabled=True), request_timeout_seconds=timeout)

    def test_cpu_only_telemetry_still_validates_the_collector_join_budget(self):
        """A join shorter than two collector cycles abandons the writer."""
        with pytest.raises(ValidationError, match="collector_join_timeout_seconds"):
            _config(
                CpuPowerConfig(enabled=True),
                request_timeout_seconds=5.0,
                collector_join_timeout_seconds=1.0,
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

    @pytest.mark.parametrize("subdir", ["/abs", "../escape", "a/../../b", ""])
    def test_cpu_only_telemetry_storage_subdir_must_stay_below_the_log_dir(self, subdir):
        """CPU measurement windows are still read from the parent telemetry directory."""
        with pytest.raises(ValidationError, match="telemetry.storage_subdir"):
            _config(CpuPowerConfig(enabled=True), storage_subdir=subdir)


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


def _session(log_dir, nodes, **overrides):
    defaults = {
        "cpu_dir": log_dir / "cpu_power",
        "job_id": "12345",
        "run_name": "recipe_12345",
        "source": "acpi",
        "sample_interval_seconds": 0.1,
        "startup_timeout_seconds": 0.2,
        "request_timeout_seconds": 1.0,
        "collector_join_timeout_seconds": 3.0,
        "required": True,
        "acpi_mandatory": True,
        "exporter_port": 9401,
        "exporter_command": "cpu-power-exporter --port 9401",
    }
    settings = CpuPowerSessionSettings(**{**defaults, **overrides})
    return CpuPowerTelemetrySession(settings=settings, nodes=nodes)


def _node_ip(node: str) -> str:
    """Node names map to addresses; the session only polls literal addresses."""
    return f"10.0.0.{ord(node[-1]) - ord('a') + 1}"


def _serve_one_rail(handler):
    """Answer one ordinary scrape, exactly as the exporter does."""
    body = ONE_RAIL.encode()
    handler.send_response(200)
    handler.send_header("Content-Type", "text/plain")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


ONE_RAIL = _scrape('cpu_power_acpi_watts{sensor="hwmon0/power1",type="CPU",socket="0"} 42.5')


def _serves(body: str, *, delay: float = 0.0):
    """A ``fetch_metrics`` stand-in: a body plus the instant it finished arriving."""

    def fetch(_endpoint, _budget):
        if delay:
            time.sleep(delay)
        return body, time.time()

    return fetch


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
    @patch("srtctl.core.power.cpu_session.get_hostname_ip", side_effect=lambda node, _iface: _node_ip(node))
    def test_one_containerless_task_per_worker_node(self, _mock_ip, mock_srun, tmp_path):
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
        assert kwargs["command"][1:] == ["--port", "9401", "--bind", "0.0.0.0"]
        assert registry.process_count == 1
        assert all(proc.critical is False for proc in registry.get_all_processes().values())
        session.stop_and_finalize()

    @patch("srtctl.cli.mixins.telemetry_stage.start_srun_process")
    @patch("srtctl.core.power.cpu_session.get_hostname_ip", side_effect=lambda node, _iface: _node_ip(node))
    def test_heterogeneous_groups_launch_once_per_group(self, _mock_ip, mock_srun, tmp_path):
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
    @patch("srtctl.core.power.cpu_session.get_hostname_ip", side_effect=lambda node, _iface: _node_ip(node))
    @patch("srtctl.core.power.cpu_session.fetch_metrics", side_effect=OSError("no exporter listening"))
    def test_launch_failure_is_session_state_not_an_exception(self, _mock_fetch, _mock_ip, mock_srun, tmp_path):
        mock_srun.side_effect = RuntimeError("srun refused")
        harness = _harness(tmp_path, [_worker("node-a", range(4))])

        session = harness.start_cpu_power_telemetry(ProcessRegistry(job_id="12345"))
        outcome = session.stop_and_finalize()

        assert Reason.EXPORTER_LAUNCH_FAILED in outcome.reason_codes
        assert outcome.exit_nonzero is True

    @patch("srtctl.cli.mixins.telemetry_stage.start_srun_process")
    @patch("srtctl.core.power.cpu_session.get_hostname_ip", side_effect=lambda node, _iface: _node_ip(node))
    @patch("srtctl.core.power.cpu_session.fetch_metrics", side_effect=OSError("no exporter listening"))
    def test_required_mode_blocks_the_benchmark_when_startup_fails(self, _mock_fetch, _mock_ip, mock_srun, tmp_path):
        """Running the workload without collection would burn the allocation."""
        mock_srun.return_value = _running_exporter()
        harness = _harness(tmp_path, [_worker("node-a", range(4))])

        session = harness.start_cpu_power_telemetry(ProcessRegistry(job_id="12345"))

        assert harness.cpu_power_telemetry_blocks_benchmark() is True
        assert harness.finalize_cpu_power_telemetry(0) == 1
        assert session is harness._cpu_power_session

    @patch("srtctl.cli.mixins.telemetry_stage.start_srun_process")
    @patch("srtctl.core.power.cpu_session.get_hostname_ip", side_effect=lambda node, _iface: _node_ip(node))
    @patch("srtctl.core.power.cpu_session.fetch_metrics", side_effect=OSError("no exporter listening"))
    def test_best_effort_mode_does_not_block_or_fail(self, _mock_fetch, _mock_ip, mock_srun, tmp_path):
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

    @patch("srtctl.cli.mixins.telemetry_stage.start_srun_process")
    @patch("srtctl.core.power.cpu_session.get_hostname_ip", side_effect=lambda node, _iface: _node_ip(node))
    @patch("srtctl.core.power.cpu_session.fetch_metrics", side_effect=OSError("no exporter listening"))
    def test_naming_acpi_gates_and_fails_the_job_without_required(self, _mock_fetch, _mock_ip, mock_srun, tmp_path):
        """`source: acpi` names a mandatory provider; `required` only restates it."""
        mock_srun.return_value = _running_exporter()
        harness = _harness(
            tmp_path,
            [_worker("node-a", range(4))],
            cpu_power=CpuPowerConfig(enabled=True, source="acpi", required=False, startup_timeout_seconds=0.2),
        )

        harness.start_cpu_power_telemetry(ProcessRegistry(job_id="12345"))

        assert harness.cpu_power_telemetry_blocks_benchmark() is True
        assert harness.finalize_cpu_power_telemetry(0) == 1

    @patch("srtctl.cli.mixins.telemetry_stage.start_srun_process")
    @patch("srtctl.core.power.cpu_session.get_hostname_ip", side_effect=lambda _node, _iface: "2001:db8::1")
    def test_an_ipv6_node_binds_the_exporter_to_that_family(self, _mock_ip, mock_srun, tmp_path):
        """The default 0.0.0.0 bind would serve nobody at the bracketed scrape URL."""
        mock_srun.return_value = _running_exporter()
        harness = _harness(tmp_path, [_worker("node-a", range(4))])

        session = harness.start_cpu_power_telemetry(ProcessRegistry(job_id="12345"))

        command = mock_srun.call_args.kwargs["command"]
        assert command[-2:] == ["--bind", "::"]
        assert [endpoint.url for endpoint in session.endpoints] == ["http://[2001:db8::1]:9401/metrics"]
        manifest = json.loads((tmp_path / "cpu_power" / "manifest.json").read_text())
        assert manifest["exporter"]["command"].endswith("--bind ::")
        session.stop_and_finalize()

    @patch("srtctl.cli.mixins.telemetry_stage.start_srun_process")
    @patch("srtctl.core.power.cpu_session.get_hostname_ip", side_effect=lambda node, _iface: _node_ip(node))
    def test_an_ipv4_node_binds_the_exporter_to_that_family(self, _mock_ip, mock_srun, tmp_path):
        """An unconditional `::` would fail to bind where IPv6 is disabled."""
        mock_srun.return_value = _running_exporter()
        harness = _harness(tmp_path, [_worker("node-a", range(4))])

        session = harness.start_cpu_power_telemetry(ProcessRegistry(job_id="12345"))

        assert mock_srun.call_args.kwargs["command"][-2:] == ["--bind", "0.0.0.0"]
        session.stop_and_finalize()

    @patch("srtctl.cli.mixins.telemetry_stage.start_srun_process")
    @patch(
        "srtctl.core.power.cpu_session.get_hostname_ip",
        side_effect=lambda node, _iface: "2001:db8::2" if node == "node-b" else _node_ip(node),
    )
    def test_mixed_address_families_are_launched_separately(self, _mock_ip, mock_srun, tmp_path):
        """One bind for the whole allocation serves whichever family it is not."""
        mock_srun.return_value = _running_exporter()
        harness = _harness(tmp_path, [_worker("node-a", range(4)), _worker("node-b", range(4), index=1)])

        session = harness.start_cpu_power_telemetry(ProcessRegistry(job_id="12345"))

        launches = [(c.kwargs["nodelist"], c.kwargs["command"][-1]) for c in mock_srun.call_args_list]
        assert sorted(launches) == [(["node-a"], "0.0.0.0"), (["node-b"], "::")]
        assert all(c.kwargs["ntasks"] == 1 for c in mock_srun.call_args_list)
        session.stop_and_finalize()

    @patch("srtctl.cli.mixins.telemetry_stage.start_srun_process")
    @patch(
        "srtctl.core.power.cpu_session.get_hostname_ip",
        side_effect=lambda node, _iface: "2001:db8::2" if node == "node-b" else _node_ip(node),
    )
    def test_mixed_families_within_a_het_group_stay_in_that_group(self, _mock_ip, mock_srun, tmp_path):
        """Splitting by family must not move a task out of its het component."""
        mock_srun.return_value = _running_exporter()
        harness = _harness(
            tmp_path,
            [
                _worker("node-a", range(4), mode="prefill", het_group=0),
                _worker("node-b", range(4), mode="prefill", index=1, het_group=0),
                _worker("node-c", range(4), mode="decode", index=2, het_group=1),
            ],
            het=True,
            het_groups={"node-a": 0, "node-b": 0, "node-c": 1},
        )

        session = harness.start_cpu_power_telemetry(ProcessRegistry(job_id="12345"))

        launches = sorted(
            (c.kwargs["het_group"], c.kwargs["nodelist"], c.kwargs["command"][-1]) for c in mock_srun.call_args_list
        )
        assert launches == [
            (0, ["node-a"], "0.0.0.0"),
            (0, ["node-b"], "::"),
            (1, ["node-c"], "0.0.0.0"),
        ]
        session.stop_and_finalize()

    @patch("srtctl.core.power.cpu_session.fetch_metrics")
    @patch("srtctl.core.power.cpu_session.get_hostname_ip", side_effect=lambda node, _iface: _node_ip(node))
    @patch("srtctl.cli.mixins.telemetry_stage.start_srun_process")
    def test_a_serving_exporter_reaches_readiness_and_publishes(self, mock_srun, _mock_ip, mock_fetch, tmp_path):
        mock_srun.return_value = _running_exporter()
        mock_fetch.side_effect = _serves(ONE_RAIL)
        harness = _harness(tmp_path, [_worker("node-a", range(4))])

        harness.start_cpu_power_telemetry(ProcessRegistry(job_id="12345"))

        assert harness.cpu_power_telemetry_blocks_benchmark() is False
        start = time.time()
        time.sleep(0.3)
        harness.record_cpu_power_benchmark_spans(start, time.time())
        assert harness.finalize_cpu_power_telemetry(0) == 0
        assert mock_fetch.call_args.args[0].url == "http://10.0.0.1:9401/metrics"

        manifest = json.loads((tmp_path / "cpu_power" / "manifest.json").read_text())
        assert manifest["status"] == "complete"
        assert manifest["publication_valid"] is True
        assert manifest["exporter"]["port"] == 9401
        assert manifest["exporter"]["command"].endswith("--port 9401 --bind 0.0.0.0")
        assert manifest["observed_sensors"] == {"node-a": ["hwmon0/power1"]}
        rows = (tmp_path / "cpu_power" / "samples.csv").read_text().splitlines()
        assert rows[0].startswith("schema_version,")
        assert any("hwmon0/power1" in row for row in rows[1:])
        assert [
            (span["covered"], sorted(span["per_series_max_sample_gap_seconds"])) for span in manifest["benchmark_spans"]
        ] == [(True, ["node-a/hwmon0/power1"])]

    @patch("srtctl.core.power.cpu_session.fetch_metrics")
    @patch("srtctl.core.power.cpu_session.get_hostname_ip", side_effect=lambda node, _iface: _node_ip(node))
    @patch("srtctl.cli.mixins.telemetry_stage.start_srun_process")
    def test_readings_that_do_not_cover_the_benchmark_do_not_publish(self, mock_srun, _mock_ip, mock_fetch, tmp_path):
        """Readiness samples alone are not measurement: the run must be bracketed."""
        mock_srun.return_value = _running_exporter()
        mock_fetch.side_effect = _serves(ONE_RAIL)
        harness = _harness(tmp_path, [_worker("node-a", range(4))])

        harness.start_cpu_power_telemetry(ProcessRegistry(job_id="12345"))
        # A benchmark that ran long before this session started collecting.
        harness.record_cpu_power_benchmark_spans(time.time() - 600, time.time() - 500)

        assert harness.finalize_cpu_power_telemetry(0) == 1

        manifest = json.loads((tmp_path / "cpu_power" / "manifest.json").read_text())
        assert manifest["publication_valid"] is False
        assert Reason.MEASUREMENT_WINDOW_NOT_BRACKETED in manifest["reason_codes"]

    @patch("srtctl.core.power.cpu_session.fetch_metrics")
    @patch("srtctl.core.power.cpu_session.get_hostname_ip", side_effect=lambda node, _iface: _node_ip(node))
    @patch("srtctl.cli.mixins.telemetry_stage.start_srun_process")
    def test_a_session_that_never_saw_a_benchmark_never_passes_vacuously(
        self, mock_srun, _mock_ip, mock_fetch, tmp_path
    ):
        mock_srun.return_value = _running_exporter()
        mock_fetch.side_effect = _serves(ONE_RAIL)
        harness = _harness(tmp_path, [_worker("node-a", range(4))])

        harness.start_cpu_power_telemetry(ProcessRegistry(job_id="12345"))

        assert harness.finalize_cpu_power_telemetry(0) == 1
        manifest = json.loads((tmp_path / "cpu_power" / "manifest.json").read_text())
        assert manifest["publication_valid"] is False
        assert manifest["benchmark_spans"] == []

    @patch("srtctl.core.power.cpu_session.fetch_metrics")
    @patch("srtctl.core.power.cpu_session.get_hostname_ip", side_effect=lambda node, _iface: node)
    @patch("srtctl.cli.mixins.telemetry_stage.start_srun_process")
    def test_a_node_that_resolves_to_its_own_name_is_never_polled(self, mock_srun, _mock_ip, mock_fetch, tmp_path):
        """get_hostname_ip hands back the hostname when it cannot resolve it.

        A URL carrying that name would be re-resolved inside every requests.get,
        before any socket exists, where the watchdog cannot reach -- one stuck
        poll thread per cycle behind a wedged resolver.
        """
        mock_srun.return_value = _running_exporter()
        harness = _harness(tmp_path, [_worker("node-a", range(4))])

        session = harness.start_cpu_power_telemetry(ProcessRegistry(job_id="12345"))

        assert harness.cpu_power_telemetry_blocks_benchmark() is True
        outcome = session.stop_and_finalize()
        assert Reason.ENDPOINT_RESOLUTION_FAILED in outcome.reason_codes
        assert outcome.publication_valid is False
        assert mock_fetch.call_count == 0

    @patch("srtctl.core.power.cpu_session.fetch_metrics")
    @patch("srtctl.cli.mixins.telemetry_stage.start_srun_process")
    def test_a_wedged_resolver_is_bounded_and_leaves_no_growing_thread_pile(self, mock_srun, mock_fetch, tmp_path):
        """Resolution happens once, under the startup deadline -- not per scrape."""
        mock_srun.return_value = _running_exporter()
        wedged = threading.Event()

        def never_resolves(_node, _iface):
            wedged.wait()
            return "10.0.0.1"

        harness = _harness(tmp_path, [_worker("node-a", range(4))])
        before = threading.active_count()
        try:
            with patch("srtctl.core.power.cpu_session.get_hostname_ip", side_effect=never_resolves):
                started = time.monotonic()
                session = harness.start_cpu_power_telemetry(ProcessRegistry(job_id="12345"))
                assert harness.cpu_power_telemetry_blocks_benchmark() is True
                elapsed = time.monotonic() - started
                # One abandoned resolver, and no scrape thread piling up behind it.
                time.sleep(0.5)
                assert threading.active_count() <= before + 2
                outcome = session.stop_and_finalize()
        finally:
            wedged.set()

        assert elapsed < 5.0
        assert Reason.ENDPOINT_RESOLUTION_FAILED in outcome.reason_codes
        assert outcome.publication_valid is False
        assert mock_fetch.call_count == 0

    @patch("srtctl.core.power.cpu_session.fetch_metrics")
    @patch("srtctl.core.power.cpu_session.get_hostname_ip")
    @patch("srtctl.cli.mixins.telemetry_stage.start_srun_process")
    def test_an_unresolvable_node_fails_readiness_closed(self, mock_srun, mock_ip, mock_fetch, tmp_path):
        """Readiness on the survivors would silently shrink what the run covers."""
        mock_srun.return_value = _running_exporter()
        mock_ip.side_effect = lambda node, _iface: "" if node == "node-b" else _node_ip(node)
        mock_fetch.side_effect = _serves(ONE_RAIL)
        harness = _harness(tmp_path, [_worker("node-a", range(4)), _worker("node-b", range(4), index=1)])

        session = harness.start_cpu_power_telemetry(ProcessRegistry(job_id="12345"))

        assert harness.cpu_power_telemetry_blocks_benchmark() is True
        outcome = session.stop_and_finalize()
        assert Reason.ENDPOINT_RESOLUTION_FAILED in outcome.reason_codes
        assert outcome.publication_valid is False

    @patch("srtctl.core.power.cpu_session.fetch_metrics")
    @patch("srtctl.core.power.cpu_session.get_hostname_ip", side_effect=lambda node, _iface: _node_ip(node))
    @patch("srtctl.cli.mixins.telemetry_stage.start_srun_process")
    def test_a_rail_that_disappears_mid_benchmark_does_not_publish(self, mock_srun, _mock_ip, mock_fetch, tmp_path):
        """The surviving rails keep the host sampling, so per-host coverage would pass."""
        mock_srun.return_value = _running_exporter()
        both = _scrape(
            'cpu_power_acpi_watts{sensor="hwmon0/power1",type="CPU",socket="0"} 42.5',
            'cpu_power_acpi_watts{sensor="hwmon0/power2",type="SYSIO",socket="0"} 12.5',
        )
        sysio_only = _scrape('cpu_power_acpi_watts{sensor="hwmon0/power2",type="SYSIO",socket="0"} 12.5')
        answers = {"count": 0}

        def scrape(_endpoint, _budget):
            answers["count"] += 1
            # The CPU rail answers the readiness cycle, then goes quiet.
            return (both if answers["count"] <= 1 else sysio_only), time.time()

        mock_fetch.side_effect = scrape
        harness = _harness(tmp_path, [_worker("node-a", range(4))])

        harness.start_cpu_power_telemetry(ProcessRegistry(job_id="12345"))
        assert harness.cpu_power_telemetry_blocks_benchmark() is False
        start = time.time()
        time.sleep(0.3)
        harness.record_cpu_power_benchmark_spans(start, time.time())

        assert harness.finalize_cpu_power_telemetry(0) == 1
        manifest = json.loads((tmp_path / "cpu_power" / "manifest.json").read_text())
        assert manifest["publication_valid"] is False
        assert Reason.MEASUREMENT_WINDOW_NOT_BRACKETED in manifest["reason_codes"]
        gaps = manifest["benchmark_spans"][0]["per_series_max_sample_gap_seconds"]
        assert "node-a/hwmon0/power1" not in gaps, "the rail that vanished cannot report a gap"

    @patch("srtctl.core.power.cpu_session.fetch_metrics")
    @patch("srtctl.core.power.cpu_session.get_hostname_ip", side_effect=lambda node, _iface: _node_ip(node))
    @patch("srtctl.cli.mixins.telemetry_stage.start_srun_process")
    def test_a_dribbling_exporter_is_abandoned_and_never_publishes(self, mock_srun, _mock_ip, mock_fetch, tmp_path):
        """A benchmark that exits 0 is not evidence the CPU leg collected anything.

        The scrape ends on its own budget (proved against real sockets in
        TestCpuPowerScrapeBudget); the required leg still has to fail the job on
        its own coverage check.
        """
        mock_srun.return_value = _running_exporter()

        def abandoned(_endpoint, _budget):
            raise TimeoutError("scrape exceeded its budget")

        mock_fetch.side_effect = abandoned
        harness = _harness(tmp_path, [_worker("node-a", range(4))])

        started = time.monotonic()
        harness.start_cpu_power_telemetry(ProcessRegistry(job_id="12345"))
        assert harness.cpu_power_telemetry_blocks_benchmark() is True
        elapsed = time.monotonic() - started
        assert elapsed < 5.0, f"a trickling exporter held startup for {elapsed:.1f}s"

        start = time.time()
        harness.record_cpu_power_benchmark_spans(start, time.time())
        # AIPerf's own exit code says nothing about CPU telemetry validity.
        assert harness.finalize_cpu_power_telemetry(0) == 1
        manifest = json.loads((tmp_path / "cpu_power" / "manifest.json").read_text())
        assert manifest["publication_valid"] is False
        assert Reason.ENDPOINT_TIMEOUT in manifest["reason_codes"]

    @patch("srtctl.core.power.cpu_session.fetch_metrics")
    @patch("srtctl.core.power.cpu_session.get_hostname_ip", side_effect=lambda node, _iface: _node_ip(node))
    def test_readiness_reached_after_the_deadline_is_rejected(self, _mock_ip, mock_fetch, tmp_path):
        """A set event proves readiness happened, not that it happened in time."""

        slow_scrape = _serves(ONE_RAIL, delay=0.3)

        mock_fetch.side_effect = slow_scrape

        class BlockingEvent(threading.Event):
            """Stands in for losing the race between wait() timing out and set()."""

            def wait(self, timeout=None):
                return super().wait()

        session = _session(tmp_path, ["node-a"], startup_timeout_seconds=0.05)
        session._ready = BlockingEvent()
        session.initialize()

        assert session.start_and_wait_for_readiness() is False
        outcome = session.stop_and_finalize()
        assert Reason.EXPORTER_STARTUP_TIMEOUT in outcome.reason_codes


class TestCpuPowerShutdown:
    def test_lock_timeout_marks_sample_count_unknown_while_a_row_can_still_commit(self, tmp_path):
        """A terminal manifest must not freeze a count before its blocked writer finishes."""
        session = _session(tmp_path, ["node-a"], collector_join_timeout_seconds=0.05)
        session.initialize()
        with patch("srtctl.core.power.cpu_session.get_hostname_ip", side_effect=lambda node, _iface: _node_ip(node)):
            session.resolve_endpoints()

        entered_writer = threading.Event()
        release_writer = threading.Event()
        writer = session._writer

        class BlockingWriter:
            def writerows(self, rows):
                entered_writer.set()
                release_writer.wait()
                writer.writerows(rows)

        session._writer = BlockingWriter()
        with patch("srtctl.core.power.cpu_session.fetch_metrics", side_effect=_serves(ONE_RAIL)):
            collector = threading.Thread(target=session.collect_once, daemon=True)
            session._thread = collector
            collector.start()
            assert entered_writer.wait(1.0)
            try:
                outcome = session.stop_and_finalize()
            finally:
                release_writer.set()
                collector.join(1.0)

        manifest = json.loads(session.manifest_path.read_text())
        committed_rows = session.samples_path.read_text().splitlines()[1:]
        assert len(committed_rows) == 1
        assert manifest["sample_row_count"] is None
        assert manifest["publication_valid"] is False
        assert Reason.COLLECTOR_JOIN_TIMEOUT in outcome.reason_codes
        assert Reason.COLLECTOR_JOIN_TIMEOUT in manifest["reason_codes"]


class TestCpuPowerScrapeBudget:
    """Real sockets: the transport owns connect and read on one wall clock.

    A library client bounds each operation separately, so an exporter that
    trickles -- headers included -- resets that clock forever without ever
    tripping a per-operation timeout, and no deadline checked between chunks is
    ever reached.
    """

    @staticmethod
    def _endpoint(address) -> CpuEndpoint:
        host, port = address[0], address[1]
        return CpuEndpoint(hostname="node-a", address=host, port=port)

    @staticmethod
    def _raw_server(responder):
        """A bare listener: an HTTP server will not trickle a header block."""
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen(8)

        def serve():
            while True:
                try:
                    connection, _ = listener.accept()
                except OSError:
                    return
                threading.Thread(target=responder, args=(connection,), daemon=True).start()

        threading.Thread(target=serve, daemon=True).start()
        return listener

    @staticmethod
    def _http_server(handler_body):
        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_GET(self):
                handler_body(self)

            def log_message(self, *_args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        return server

    @classmethod
    def _trickling_server(cls, delay: float = 0.05):
        def body(handler):
            handler.send_response(200)
            handler.send_header("Content-Type", "text/plain")
            handler.send_header("Content-Length", "1000000")
            handler.end_headers()
            try:
                while True:
                    handler.wfile.write(b"x" * 20)
                    handler.wfile.flush()
                    time.sleep(delay)
            except OSError:
                pass

        return cls._http_server(body)

    @staticmethod
    def _trickle_headers(connection):
        """Answer with a status line, then one header byte at a time, forever."""
        try:
            connection.recv(4096)
            connection.sendall(b"HTTP/1.0 200 OK\r\n")
            while True:
                connection.sendall(b"X")
                time.sleep(0.05)
        except OSError:
            pass
        finally:
            connection.close()

    def test_send_uses_only_the_budget_remaining_after_connect(self):
        """A slow connect must not leave sendall with the original full timeout."""
        clock = SimpleNamespace(now=100.0)

        class FakeSocket:
            def __init__(self):
                self.timeout = None
                self.send_timeout = None
                self.responses = iter(
                    [b"HTTP/1.0 200 OK\r\nContent-Type: text/plain\r\n\r\nmetric 1\n", b""]
                )

            def settimeout(self, timeout):
                self.timeout = timeout

            def connect(self, _peer):
                clock.now += 0.75

            def sendall(self, _request):
                self.send_timeout = self.timeout

            def recv(self, _size):
                return next(self.responses)

            def close(self):
                pass

        sock = FakeSocket()
        endpoint = CpuEndpoint(hostname="node-a", address="127.0.0.1", port=9401)

        with (
            patch("srtctl.core.power.cpu_session.socket.socket", return_value=sock),
            patch("srtctl.core.power.cpu_session.time.monotonic", side_effect=lambda: clock.now),
        ):
            body, _completed_at = fetch_metrics(endpoint, 1.0)

        assert sock.send_timeout == pytest.approx(0.25)
        assert body == "metric 1\n"

    def test_a_trickling_body_is_cut_off_at_the_budget(self, tmp_path):
        server = self._trickling_server()
        try:
            session = _session(tmp_path, ["node-a"], request_timeout_seconds=0.2)

            started = time.monotonic()
            result = session._poll(self._endpoint(server.server_address))
            elapsed = time.monotonic() - started

            assert result.reason_codes == [Reason.ENDPOINT_TIMEOUT]
            assert result.readings == []
            # 2 x request_timeout, plus room for a loaded CI box.
            assert elapsed < 1.5, f"the scrape ran {elapsed:.2f}s past a 0.4s budget"
        finally:
            server.shutdown()
            server.server_close()

    def test_trickled_headers_are_cut_off_at_the_budget(self, tmp_path):
        """Nothing is a Response yet here: a client-side read timeout never fires."""
        listener = self._raw_server(self._trickle_headers)
        try:
            session = _session(tmp_path, ["node-a"], request_timeout_seconds=0.2)

            started = time.monotonic()
            result = session._poll(self._endpoint(listener.getsockname()))
            elapsed = time.monotonic() - started

            assert result.reason_codes == [Reason.ENDPOINT_TIMEOUT]
            assert elapsed < 1.5, f"the scrape ran {elapsed:.2f}s past a 0.4s budget"
        finally:
            listener.close()

    def test_a_cut_off_scrape_leaves_no_thread_behind(self, tmp_path):
        """One leaked reader per cycle would outlive the run it was collecting."""
        listener = self._raw_server(self._trickle_headers)
        try:
            session = _session(tmp_path, ["node-a"], request_timeout_seconds=0.2)
            endpoint = self._endpoint(listener.getsockname())
            before = threading.active_count()

            for _ in range(3):
                session._poll(endpoint)

            deadline = time.monotonic() + 2.0
            while threading.active_count() > before and time.monotonic() < deadline:
                time.sleep(0.05)
            assert threading.active_count() <= before
        finally:
            listener.close()

    def test_an_oversized_body_is_cut_off(self, tmp_path):
        """A runaway body would otherwise be read for the whole budget."""

        def flood(connection):
            try:
                connection.recv(4096)
                connection.sendall(b"HTTP/1.0 200 OK\r\nContent-Type: text/plain\r\n\r\n")
                while True:
                    connection.sendall(b"x" * 65536)
            except OSError:
                pass
            finally:
                connection.close()

        listener = self._raw_server(flood)
        try:
            session = _session(tmp_path, ["node-a"], request_timeout_seconds=5.0)

            started = time.monotonic()
            result = session._poll(self._endpoint(listener.getsockname()))

            assert result.reason_codes == [Reason.ENDPOINT_HTTP_ERROR]
            assert time.monotonic() - started < 5.0, "the size cap did not end the read"
            assert MAX_SCRAPE_BYTES > 0
        finally:
            listener.close()

    def test_a_prompt_exporter_is_read_in_full(self, tmp_path):
        """The budget must not truncate an ordinary scrape."""
        server = self._http_server(_serve_one_rail)
        try:
            session = _session(tmp_path, ["node-a"], request_timeout_seconds=0.2)

            result = session._poll(self._endpoint(server.server_address))

            assert result.reason_codes == []
            assert [reading.sensor for reading in result.readings] == ["hwmon0/power1"]
        finally:
            server.shutdown()
            server.server_close()

    def test_an_ambient_proxy_is_never_consulted(self, tmp_path):
        """The exporter is on the cluster fabric; a job-wide proxy must not be dialed.

        A proxy-aware client would send this scrape to a black hole, and a
        wedged proxy resolver would block before any socket existed.
        """
        server = self._http_server(_serve_one_rail)
        proxy = {var: "http://192.0.2.1:9" for var in ("http_proxy", "HTTP_PROXY", "all_proxy", "ALL_PROXY")}
        try:
            session = _session(tmp_path, ["node-a"], request_timeout_seconds=0.5)
            with patch.dict(os.environ, proxy):
                result = session._poll(self._endpoint(server.server_address))

            assert result.reason_codes == []
            assert [reading.sensor for reading in result.readings] == ["hwmon0/power1"]
        finally:
            server.shutdown()
            server.server_close()


class TestCpuPowerSampleTimestamps:
    """Every sample is dated when its own body arrived."""

    def test_endpoints_are_timestamped_independently(self, tmp_path):
        """One cycle timestamp would date a fast node's reading to the slow node's finish.

        That is what moves a reading across a benchmark boundary and lets
        coverage bracket an interval nothing actually sampled.
        """
        session = _session(tmp_path, ["node-a", "node-b"], request_timeout_seconds=1.0)
        session.initialize()
        slow = _serves(ONE_RAIL, delay=0.4)
        fast = _serves(ONE_RAIL)

        with patch("srtctl.core.power.cpu_session.get_hostname_ip", side_effect=lambda node, _iface: _node_ip(node)):
            session.resolve_endpoints()
        with patch(
            "srtctl.core.power.cpu_session.fetch_metrics",
            side_effect=lambda endpoint, budget: (slow if endpoint.hostname == "node-b" else fast)(endpoint, budget),
        ):
            session.collect_once()
        session.stop_and_finalize()

        rows = [row.split(",") for row in (tmp_path / "cpu_power" / "samples.csv").read_text().splitlines()[1:]]
        stamps = {row[3]: float(row[1]) for row in rows}
        assert set(stamps) == {"node-a", "node-b"}
        assert stamps["node-b"] - stamps["node-a"] > 0.3, "both nodes shared one cycle timestamp"


class TestCpuPowerMeasurementSpans:
    """The audited interval is what the runner measured, not the whole script."""

    @staticmethod
    def _window(windows_dir: Path, name: str, *, status: str, start: float, end: float | None) -> None:
        windows_dir.mkdir(parents=True, exist_ok=True)
        duration = None if end is None else end - start
        payload = {
            "schema_version": 1,
            "benchmark_type": "sa-bench",
            "result_path": f"results/{name}.json",
            "concurrency": int(name.removeprefix("c")),
            "benchmark_start_time_unix": start,
            "benchmark_end_time_unix": end,
            "duration": duration,
            "clock_source": "head_node_unix_clock",
            "status": status,
            "reason": None if status != "interrupted" else "job_cancelled",
        }
        (windows_dir / f"{name}.json").write_text(json.dumps(payload))
        if status == "completed":
            results_dir = windows_dir.parent.parent / "results"
            results_dir.mkdir(exist_ok=True)
            (results_dir / f"{name}.json").write_text(
                json.dumps(
                    {
                        "benchmark_start_time_unix": start,
                        "benchmark_end_time_unix": end,
                        "duration": duration,
                    }
                )
            )

    def _harness_with_session(self, tmp_path):
        harness = _harness(tmp_path, [_worker("node-a", range(4))])
        session = _session(tmp_path, ["node-a"])
        session.initialize()
        harness._cpu_power_session = session
        return harness, session

    def test_each_completed_series_is_its_own_span(self, tmp_path):
        """Setup, warmups, and the gaps between series were never measured."""
        harness, session = self._harness_with_session(tmp_path)
        base = time.time()
        windows = tmp_path / "power" / "windows"
        self._window(windows, "c4", status="completed", start=base + 10, end=base + 20)
        self._window(windows, "c8", status="completed", start=base + 40, end=base + 50)

        harness.record_cpu_power_benchmark_spans(base, base + 60)
        session.stop_and_finalize()

        recorded = json.loads((tmp_path / "cpu_power" / "manifest.json").read_text())["benchmark_spans"]

        assert [(span["start_unix"], span["end_unix"]) for span in recorded] == [
            (base + 10, base + 20),
            (base + 40, base + 50),
        ]

    @pytest.mark.parametrize(
        ("status", "end"),
        [("running", None), ("interrupted", None)],
    )
    def test_a_window_without_a_trustworthy_end_falls_back_to_the_script(self, tmp_path, status, end):
        """The orchestrator never invents the boundary the runner failed to publish."""
        harness, session = self._harness_with_session(tmp_path)
        base = time.time()
        self._window(tmp_path / "power" / "windows", "c4", status=status, start=base + 10, end=end)

        harness.record_cpu_power_benchmark_spans(base, base + 60)
        session.stop_and_finalize()

        recorded = json.loads((tmp_path / "cpu_power" / "manifest.json").read_text())["benchmark_spans"]
        assert [(span["start_unix"], span["end_unix"]) for span in recorded] == [(base, base + 60)]

    def test_a_benchmark_that_writes_no_window_still_marks_the_script(self, tmp_path):
        """The CPU leg serves benchmark types the window contract does not cover."""
        harness, session = self._harness_with_session(tmp_path)
        base = time.time()

        harness.record_cpu_power_benchmark_spans(base, base + 60)
        session.stop_and_finalize()

        recorded = json.loads((tmp_path / "cpu_power" / "manifest.json").read_text())["benchmark_spans"]
        assert [(span["start_unix"], span["end_unix"]) for span in recorded] == [(base, base + 60)]

    def test_an_equal_completed_span_falls_back_to_the_script(self, tmp_path):
        """A zero-duration completed artifact is malformed, not a measured interval."""
        harness, session = self._harness_with_session(tmp_path)
        base = time.time()
        self._window(tmp_path / "power" / "windows", "c4", status="completed", start=base + 10, end=base + 10)

        harness.record_cpu_power_benchmark_spans(base, base + 60)
        session.stop_and_finalize()

        recorded = json.loads((tmp_path / "cpu_power" / "manifest.json").read_text())["benchmark_spans"]
        assert [(span["start_unix"], span["end_unix"]) for span in recorded] == [(base, base + 60)]

    def test_a_structurally_malformed_completed_window_falls_back_to_the_script(self, tmp_path):
        """A status and two timestamps alone do not satisfy the window contract."""
        harness, session = self._harness_with_session(tmp_path)
        base = time.time()
        windows_dir = tmp_path / "power" / "windows"
        windows_dir.mkdir(parents=True)
        (windows_dir / "c4.json").write_text(
            json.dumps(
                {
                    "status": "completed",
                    "benchmark_start_time_unix": base + 10,
                    "benchmark_end_time_unix": base + 20,
                }
            )
        )

        harness.record_cpu_power_benchmark_spans(base, base + 60)
        session.stop_and_finalize()

        recorded = json.loads((tmp_path / "cpu_power" / "manifest.json").read_text())["benchmark_spans"]
        assert [(span["start_unix"], span["end_unix"]) for span in recorded] == [(base, base + 60)]

    def test_a_gap_between_series_invalidates_only_the_script_interval(self, tmp_path):
        """This is the regression: sampling paused between series, not during one."""
        base = time.time()
        # Two dense series with a long idle stretch between them.
        series = [base, base + 0.5, base + 1.0, base + 20.0, base + 20.5, base + 21.0]

        def audit(spans):
            session = _session(tmp_path / f"run{len(spans)}", ["node-a"])
            session.initialize()
            with patch(
                "srtctl.core.power.cpu_session.get_hostname_ip", side_effect=lambda node, _iface: _node_ip(node)
            ):
                session.resolve_endpoints()
            for stamp in series:
                with patch(
                    "srtctl.core.power.cpu_session.fetch_metrics",
                    side_effect=lambda _endpoint, _budget, at=stamp: (ONE_RAIL, at),
                ):
                    session.collect_once()
            for start, end in spans:
                session.record_benchmark_span(start, end)
            return session.stop_and_finalize()

        per_series = audit([(base + 0.1, base + 0.9), (base + 20.1, base + 20.9)])
        whole_script = audit([(base + 0.1, base + 20.9)])

        assert per_series.publication_valid is True
        assert whole_script.publication_valid is False
        assert Reason.SAMPLE_GAP_EXCEEDED in whole_script.reason_codes


class TestCpuPowerEndpointResolution:
    """A node is polled at a literal address, in whichever family it has one."""

    @staticmethod
    def _addrinfo(*addresses):
        """getaddrinfo's shape, carrying only what the resolver reads."""
        return [
            (socket.AF_INET6 if ":" in address else socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, 0, 0, 0))
            for address in addresses
        ]

    def test_an_interface_address_is_used_as_is(self):
        """The interface-aware path answers first; nothing else may override it."""
        with (
            patch("srtctl.core.power.cpu_session.get_hostname_ip", return_value="10.0.0.7") as resolve,
            patch("socket.getaddrinfo", side_effect=AssertionError("must not be consulted")),
        ):
            assert cpu_endpoint_address("node-a", "eth0") == "10.0.0.7"
        assert resolve.call_args.args == ("node-a", "eth0")

    def test_an_ipv6_only_node_resolves_to_its_ipv6_address(self):
        """get_hostname_ip is IPv4-only and hands such a node back its own name.

        Dropping it would omit the node under `auto` and fail the allocation
        under mandatory settings, on a cluster where the exporter is reachable.
        """
        with (
            patch("srtctl.core.power.cpu_session.get_hostname_ip", return_value="node-a"),
            patch("socket.getaddrinfo", return_value=self._addrinfo("2001:db8::1")),
        ):
            assert cpu_endpoint_address("node-a", "eth0") == "2001:db8::1"

    def test_an_ipv6_only_node_is_polled_and_advertised_bracketed(self, tmp_path):
        """The one address is a URL both the collector and the client can use."""
        session = _session(tmp_path, ["node-a"], exporter_port=9401)
        with (
            patch("srtctl.core.power.cpu_session.get_hostname_ip", return_value="node-a"),
            patch("socket.getaddrinfo", return_value=self._addrinfo("2001:db8::1")),
        ):
            session.resolve_endpoints()

        assert [endpoint.url for endpoint in session.endpoints] == ["http://[2001:db8::1]:9401/metrics"]

    def test_a_dual_stack_node_keeps_the_ipv4_answer(self):
        """The fallback cannot express an interface preference, so it defers to one."""
        with (
            patch("srtctl.core.power.cpu_session.get_hostname_ip", return_value="node-a"),
            patch("socket.getaddrinfo", return_value=self._addrinfo("2001:db8::1", "10.0.0.7")),
        ):
            assert cpu_endpoint_address("node-a", "eth0") == "10.0.0.7"

    def test_link_local_resolution_preserves_scope_in_the_address_and_url(self):
        """getaddrinfo carries the interface as sockaddr's scope-id field."""
        scoped = (socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("fe80::1", 0, 0, 7))
        with (
            patch("srtctl.core.power.cpu_session.get_hostname_ip", return_value="node-a"),
            patch("socket.getaddrinfo", return_value=[scoped]),
        ):
            address = cpu_endpoint_address("node-a", "eth0")

        assert address == "fe80::1%7"
        assert CpuEndpoint("node-a", address, 9401).url == "http://[fe80::1%257]:9401/metrics"

    def test_link_local_scrape_connects_with_the_resolved_scope_id(self):
        """No live link-local interface is needed: assert the sockaddr handed to the socket."""
        endpoint = CpuEndpoint("node-a", "fe80::1%7", 9401)
        sock = MagicMock()
        sock.recv.side_effect = [b"HTTP/1.0 200 OK\r\nContent-Type: text/plain\r\n\r\nmetric 1\n", b""]

        with patch("srtctl.core.power.cpu_session.socket.socket", return_value=sock) as make_socket:
            body, _completed_at = fetch_metrics(endpoint, 1.0)

        make_socket.assert_called_once_with(socket.AF_INET6, socket.SOCK_STREAM)
        sock.connect.assert_called_once_with(("fe80::1", 9401, 0, 7))
        assert b"Host: [fe80::1%257]:9401\r\n" in sock.sendall.call_args.args[0]
        assert body == "metric 1\n"

    def test_a_node_with_no_address_in_any_family_is_dropped(self):
        with (
            patch("srtctl.core.power.cpu_session.get_hostname_ip", return_value="node-a"),
            patch("socket.getaddrinfo", side_effect=socket.gaierror("Name or service not known")),
        ):
            assert cpu_endpoint_address("node-a", "eth0") is None


class TestServerMetricsUrlHosts:
    """Every server-metrics family spells an IPv6 host the same way."""

    @staticmethod
    def _urls(stage):
        with patch("srtctl.cli.mixins.benchmark_stage.get_hostname_ip", return_value="2001:db8::1"):
            env = stage._get_aiperf_server_metrics_env()
        return env.get("AIPERF_SERVER_METRICS_URLS", "").split(",")

    def test_a_vllm_frontend_brackets_its_ipv6_host(self):
        stage = TestCpuPowerServerMetricsUrls._stage([_worker("node-a", range(4))], frontend_type="vllm")

        assert "http://[2001:db8::1]:8000/metrics" in self._urls(stage)

    def test_a_vllm_router_brackets_its_ipv6_hosts(self):
        stage = TestCpuPowerServerMetricsUrls._stage([_worker("node-a", range(4))], frontend_type="vllm-router")

        assert "http://[2001:db8::1]:30000/metrics" in self._urls(stage)

    def test_worker_sys_ports_bracket_their_ipv6_hosts(self):
        stage = TestCpuPowerServerMetricsUrls._stage([_worker("node-a", range(4))])

        assert "http://[2001:db8::1]:8081/metrics" in self._urls(stage)

    def test_every_advertised_url_parses_back_to_the_address_and_port(self):
        """An unbracketed literal parses as host "2001" -- or not at all."""
        from urllib.parse import urlsplit

        for frontend_type in ("dynamo", "vllm", "vllm-router"):
            stage = TestCpuPowerServerMetricsUrls._stage([_worker("node-a", range(4))], frontend_type=frontend_type)
            for url in self._urls(stage):
                parts = urlsplit(url)
                assert parts.hostname == "2001:db8::1", url
                assert parts.port is not None, url


class TestCpuPowerServerMetricsUrls:
    """CPU exporters are advertised to AIPerf alongside the engine endpoints.

    The URLs are the telemetry session's own resolved endpoints -- never a
    second resolution -- so these build a real session and project it.
    """

    @staticmethod
    def _stage(processes, *, frontend_type="dynamo", cpu_power=None, session=None):
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
        stage._cpu_power_session = session
        return stage

    def _resolved_session(self, tmp_path, nodes, resolve, *, serving=None):
        """A session that resolved and polled its endpoints as the real one does.

        Only endpoints proven to serve rails are advertised, so a session that
        never collected is indistinguishable from one whose exporters are dead.
        """
        session = _session(tmp_path, nodes, exporter_port=9401)
        with patch("srtctl.core.power.cpu_session.get_hostname_ip", side_effect=resolve):
            session.resolve_endpoints()

        def scrape(endpoint, _budget):
            if serving is not None and endpoint.hostname not in serving:
                raise OSError("no exporter listening")
            return ONE_RAIL, time.time()

        with patch("srtctl.core.power.cpu_session.fetch_metrics", side_effect=scrape):
            session.collect_once()
        return session

    def _urls(self, stage, resolve=None):
        resolve = resolve or (lambda node, _iface: _node_ip(node))
        with patch("srtctl.cli.mixins.benchmark_stage.get_hostname_ip", side_effect=resolve):
            env = stage._get_aiperf_server_metrics_env()
        return env.get("AIPERF_SERVER_METRICS_URLS", "").split(",") if env else []

    def test_one_cpu_url_per_worker_node(self, tmp_path):
        processes = [
            _worker("node-a", range(4)),
            _worker("node-a", range(4), index=1),
            _worker("node-b", range(4), index=2),
        ]
        resolve = lambda node, _iface: _node_ip(node)  # noqa: E731
        stage = self._stage(
            processes,
            cpu_power=CpuPowerConfig(enabled=True),
            session=self._resolved_session(tmp_path, ["node-a", "node-b"], resolve),
        )

        assert "http://10.0.0.1:9401/metrics" in self._urls(stage)
        assert "http://10.0.0.2:9401/metrics" in self._urls(stage)

    def test_vllm_frontend_still_gets_the_cpu_urls(self, tmp_path):
        """The vLLM branch used to return early, dropping the CPU endpoints."""
        resolve = lambda node, _iface: _node_ip(node)  # noqa: E731
        stage = self._stage(
            [_worker("node-a", range(4))],
            frontend_type="vllm",
            cpu_power=CpuPowerConfig(enabled=True),
            session=self._resolved_session(tmp_path, ["node-a"], resolve),
        )

        urls = self._urls(stage)

        assert "http://10.0.0.1:9401/metrics" in urls
        assert any(url.endswith("8000/metrics") for url in urls)

    def test_disabled_leg_adds_no_cpu_urls(self):
        stage = self._stage([_worker("node-a", range(4))])

        assert all(":9401/" not in url for url in self._urls(stage))

    def test_a_node_that_never_resolves_is_not_advertised(self, tmp_path):
        """The session refuses such a node; handing it to AIPerf only buys DNS failures."""
        resolve = lambda node, _iface: node if node == "node-b" else _node_ip(node)  # noqa: E731
        stage = self._stage(
            [_worker("node-a", range(4)), _worker("node-b", range(4), index=1)],
            cpu_power=CpuPowerConfig(enabled=True),
            session=self._resolved_session(tmp_path, ["node-a", "node-b"], resolve),
        )

        urls = self._urls(stage, resolve=resolve)

        assert [url for url in urls if ":9401/" in url] == ["http://10.0.0.1:9401/metrics"]

    def test_an_ipv6_node_is_bracketed(self, tmp_path):
        """Unbracketed, the address's own colons read as the port separator."""
        resolve = lambda _node, _iface: "2001:db8::1"  # noqa: E731
        stage = self._stage(
            [_worker("node-a", range(4))],
            cpu_power=CpuPowerConfig(enabled=True),
            session=self._resolved_session(tmp_path, ["node-a"], resolve),
        )

        assert "http://[2001:db8::1]:9401/metrics" in self._urls(stage, resolve=resolve)

    def test_only_endpoints_that_served_rails_are_advertised(self, tmp_path):
        """A launched-but-dead exporter would otherwise be presented as coverage."""
        resolve = lambda node, _iface: _node_ip(node)  # noqa: E731
        stage = self._stage(
            [_worker("node-a", range(4)), _worker("node-b", range(4), index=1)],
            cpu_power=CpuPowerConfig(enabled=True),
            session=self._resolved_session(tmp_path, ["node-a", "node-b"], resolve, serving={"node-a"}),
        )

        assert [url for url in self._urls(stage) if ":9401/" in url] == ["http://10.0.0.1:9401/metrics"]

    def test_a_leg_where_nothing_served_advertises_nothing(self, tmp_path):
        resolve = lambda node, _iface: _node_ip(node)  # noqa: E731
        stage = self._stage(
            [_worker("node-a", range(4))],
            cpu_power=CpuPowerConfig(enabled=True),
            session=self._resolved_session(tmp_path, ["node-a"], resolve, serving=set()),
        )

        assert all(":9401/" not in url for url in self._urls(stage))

    def test_a_leg_that_never_started_advertises_nothing(self, tmp_path):
        """Exporters that were never launched must not be presented as coverage."""
        stage = self._stage([_worker("node-a", range(4))], cpu_power=CpuPowerConfig(enabled=True))

        assert all(":9401/" not in url for url in self._urls(stage))
