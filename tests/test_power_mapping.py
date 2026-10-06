# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU power metric mappings: the exporter config that makes the power collector exporter-agnostic.

The DCGM default must reproduce the pre-mapping contract byte for byte on the
wire keys it owns; an AMD (rocm/device-metrics-exporter) ``power`` block
exercises every place that used to hard-code a DCGM metric or label.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from marshmallow import ValidationError

from srtctl.cli.mixins.telemetry_stage import TelemetryStageMixin, resolve_exporter_command
from srtctl.core.config import resolve_config_with_defaults
from srtctl.core.power.contract import MANIFEST_FILENAME, UTILIZATION_METRICS, Reason
from srtctl.core.power.manifest import DcgmExporterIdentity, ExpectedWindow, PowerManifest
from srtctl.core.power.mapping import DCGM_EXPORTER_COMMAND_TEMPLATE, DCGM_POWER_MAPPING, TACHOMETER_SCRAPE
from srtctl.core.power.parser import parse_power_scrape
from srtctl.core.power.samples import read_samples
from srtctl.core.power.session import PowerEndpoint, PowerSessionSettings, PowerTelemetrySession
from srtctl.core.power.topology import build_expected_devices
from srtctl.core.power.validate_artifacts import validate_power_artifacts
from srtctl.core.processes import ProcessRegistry
from srtctl.core.schema import (
    DCGM_GPU_LABELS,
    DCGM_GPU_METRICS,
    BenchmarkConfig,
    ModelConfig,
    PlacementConfig,
    ResourceConfig,
    SrtConfig,
    TelemetryConfig,
    TelemetryExporterConfig,
)
from srtctl.core.topology import Process

AMD_IMAGE = "docker://rocm/device-metrics-exporter:v1.5.2"
AMD_SCOPE = "gpu_device_power_as_reported_by_amd_device_metrics_exporter"
AMD_GPU_CONFIG = {
    "kind": "custom",
    "gpu_labels": {"index": "gpu_id", "identity": "serial_number"},
    "gpu_metrics": {
        "power": {"metric": "gpu_power_usage", "scope": AMD_SCOPE},
        "gpu_util": {"metric": "gpu_gfx_activity"},
    },
}
AMD_EXPORTER = TelemetryExporterConfig.Schema().load(
    {"container_image": AMD_IMAGE, "port": 5000, "command": "/home/amd/tools/entrypoint.sh", **AMD_GPU_CONFIG}
)
AMD = AMD_EXPORTER.power_mapping


def _amd_body(count=4, watts=500.0, *, utilization=False, serial="SN", partition="NA"):
    """One rocm/device-metrics-exporter scrape: lowercase names, default label set, no gpu_uuid."""
    lines = ["# HELP gpu_power_usage GPU Power usage in Watts", "# TYPE gpu_power_usage gauge"]
    for index in range(count):
        lines.append(
            f'gpu_power_usage{{gpu_id="{index}",serial_number="{serial}{index}",card_model="MI355X",'
            f'gpu_partition_id="{partition}",gpu_compute_partition_type="SPX",hostname="exporter-lies"}} '
            f"{watts + index}"
        )
    if utilization:
        lines.append("# TYPE gpu_gfx_activity gauge")
        for index in range(count):
            lines.append(f'gpu_gfx_activity{{gpu_id="{index}",serial_number="{serial}{index}"}} {10 * index}')
    return "\n".join(lines) + "\n"


def _dcgm_body(count=4, watts=400.0):
    lines = ["# TYPE DCGM_FI_DEV_POWER_USAGE gauge"]
    for index in range(count):
        lines.append(
            f'DCGM_FI_DEV_POWER_USAGE{{gpu="{index}",UUID="GPU-{index}",device="nvidia{index}"}} {watts + index}'
        )
    return "\n".join(lines) + "\n"


def _worker(node="node-a", gpus=range(4), mode="agg", index=0):
    return Process(
        node=node,
        gpu_indices=frozenset(gpus),
        sys_port=8081,
        http_port=30000,
        endpoint_mode=mode,
        endpoint_index=index,
        node_rank=0,
        het_group=None,
    )


class TestMapping:
    def test_dcgm_default_pins_the_pre_mapping_contract(self):
        """The NVIDIA path must be unchanged: same metric, labels, riders and launch command."""
        assert TelemetryExporterConfig(container_image="dcgm-exporter", port=9401).power_mapping == DCGM_POWER_MAPPING
        assert DCGM_POWER_MAPPING.power_metric == "DCGM_FI_DEV_POWER_USAGE"
        assert DCGM_POWER_MAPPING.power_scope == "gpu_device_board_as_reported_by_dcgm"
        assert (DCGM_POWER_MAPPING.gpu_index_label, DCGM_POWER_MAPPING.gpu_identity_label) == ("gpu", "UUID")
        assert DCGM_POWER_MAPPING.utilization_metrics == UTILIZATION_METRICS
        assert DCGM_POWER_MAPPING.instance_labels == ("GPU_I_ID", "GPU_I_PROFILE")
        assert DCGM_EXPORTER_COMMAND_TEMPLATE == "dcgm-exporter --collect-interval=100 --address :{port}"

    def test_amd_power_block_resolves_with_contract_units(self):
        assert AMD.power_metric == "gpu_power_usage"
        assert AMD.power_scope == AMD_SCOPE
        assert (AMD.gpu_index_label, AMD.gpu_identity_label) == ("gpu_id", "serial_number")
        assert [(m.column, m.metric, m.unit, m.max_value) for m in AMD.utilization_metrics] == [
            ("gpu_util_pct", "gpu_gfx_activity", "percent", 100.0)
        ]
        assert AMD.instance_labels == ()


class TestParsingByMapping:
    def test_amd_scrape_yields_serial_identified_readings_with_gfx_activity(self):
        scrape = parse_power_scrape(_amd_body(count=2, utilization=True), AMD)

        assert scrape.reason_codes == ()
        assert [(r.gpu_index, r.gpu_uuid, r.power_w, r.gpu_util_pct, r.sm_active) for r in scrape.readings] == [
            (0, "SN0", 500.0, 0.0, None),
            (1, "SN1", 501.0, 10.0, None),
        ]

    def test_amd_scrape_without_utilization_leaves_the_riders_empty(self):
        scrape = parse_power_scrape(_amd_body(count=1), AMD)
        assert scrape.readings[0].gpu_util_pct is None and scrape.readings[0].sm_active is None

    @pytest.mark.parametrize(
        ("body", "mapping"),
        [(_dcgm_body(), AMD), (_amd_body(), DCGM_POWER_MAPPING)],
        ids=["dcgm-body-under-amd-mapping", "amd-body-under-dcgm-mapping"],
    )
    def test_a_body_from_another_exporter_is_a_missing_power_metric_not_a_reading(self, body, mapping):
        scrape = parse_power_scrape(body, mapping)
        assert scrape.readings == ()
        assert scrape.reason_codes == (Reason.POWER_METRIC_MISSING,)

    def test_default_mapping_still_parses_dcgm(self):
        scrape = parse_power_scrape(_dcgm_body(count=2))
        assert [(r.gpu_index, r.gpu_uuid) for r in scrape.readings] == [(0, "GPU-0"), (1, "GPU-1")]

    def test_missing_identity_label_is_reported_under_the_uuid_reason(self):
        body = '# TYPE gpu_power_usage gauge\ngpu_power_usage{gpu_id="0",serial_number=""} 500\n'
        scrape = parse_power_scrape(body, AMD)
        assert scrape.readings == () and Reason.GPU_UUID_MISSING in scrape.reason_codes

    def test_instance_labels_are_the_dcgm_mappings_concern(self):
        """A MIG label means nothing to a mapping that declares no instance labels."""
        body = '# TYPE gpu_power_usage gauge\ngpu_power_usage{gpu_id="0",serial_number="SN0",GPU_I_ID="1"} 500\n'
        assert parse_power_scrape(body, AMD).readings[0].gpu_uuid == "SN0"
        dcgm = '# TYPE DCGM_FI_DEV_POWER_USAGE gauge\nDCGM_FI_DEV_POWER_USAGE{gpu="0",UUID="GPU-0",GPU_I_ID="1"} 500\n'
        assert Reason.MIG_INSTANCE_UNSUPPORTED in parse_power_scrape(dcgm).reason_codes


def _manifest(mapping=None):
    fields = {
        "job_id": "12345",
        "run_name": "recipe_12345",
        "sample_interval_seconds": 1.0,
        "request_timeout_seconds": 2.0,
        "required": True,
        "started_at_unix": 1785168000.0,
        "dcgm_exporter": DcgmExporterIdentity(AMD_IMAGE, None, 5000, "/home/amd/tools/entrypoint.sh"),
        "expected_devices": build_expected_devices([_worker(gpus=[0])]),
        "expected_windows": [ExpectedWindow("sa-bench", 4)],
    }
    if mapping is not None:
        fields["mapping"] = mapping
    return PowerManifest(**fields)


class TestManifestProvenance:
    def test_default_manifest_keeps_the_dcgm_wire_values(self):
        payload = _manifest().to_dict()
        assert payload["source_metric"] == "DCGM_FI_DEV_POWER_USAGE"
        assert payload["power_scope"] == "gpu_device_board_as_reported_by_dcgm"
        assert [m["source_metric"] for m in payload["utilization_metrics"]] == [
            "DCGM_FI_DEV_GPU_UTIL",
            "DCGM_FI_PROF_SM_ACTIVE",
        ]

    def test_amd_manifest_describes_the_amd_measurement(self):
        payload = _manifest(AMD).to_dict()
        assert payload["producer"] == "srt-slurm.dcgm-power"  # the artifact producer, not the exporter
        assert payload["source_metric"] == "gpu_power_usage"
        assert payload["power_scope"] == AMD_SCOPE
        assert payload["utilization_metrics"] == [
            {"column": "gpu_util_pct", "source_metric": "gpu_gfx_activity", "unit": "percent"}
        ]
        assert payload["dcgm_exporter"]["container_image_resolved"] == AMD_IMAGE


class _FakeResponse:
    status_code = 200

    def __init__(self, body):
        self.text = body

    def raise_for_status(self):
        return None


class TestSessionWithAmdExporter:
    def _session(self, tmp_path, body):
        settings = PowerSessionSettings(
            power_dir=tmp_path / "logs" / "power",
            log_dir=tmp_path / "logs",
            job_id="12345",
            run_name="recipe_12345",
            sample_interval_seconds=0.05,
            startup_timeout_seconds=2.0,
            request_timeout_seconds=0.5,
            collector_join_timeout_seconds=5.0,
            required=True,
            exporter_port=5000,
            exporter_image=AMD_IMAGE,
            exporter_command="/home/amd/tools/entrypoint.sh",
            mapping=AMD,
        )
        session = PowerTelemetrySession(
            settings=settings,
            expected_devices=build_expected_devices([_worker(gpus=range(2))]),
            expected_windows=[ExpectedWindow("sa-bench", 4)],
            nodes=["node-a"],
            endpoints=[PowerEndpoint("node-a", "http://node-a:5000/metrics")],
        )
        session.initialize()
        with patch("srtctl.core.power.session.requests.get", return_value=_FakeResponse(body)):
            session.collect_once()
        return session

    def test_samples_carry_the_serial_as_identity_and_the_manifest_names_the_metric(self, tmp_path):
        session = self._session(tmp_path, _amd_body(count=2, utilization=True))
        session.stop_and_finalize()

        rows, reasons = read_samples(session.samples_path)
        assert reasons == ()
        assert [(r.hostname, r.gpu_index, r.gpu_uuid, r.power_w, r.gpu_util_pct, r.sm_active) for r in rows] == [
            ("node-a", 0, "SN0", 500.0, 0.0, None),
            ("node-a", 1, "SN1", 501.0, 10.0, None),
        ]
        manifest = json.loads((session.power_dir / MANIFEST_FILENAME).read_text())
        assert manifest["source_metric"] == "gpu_power_usage"
        assert [d["gpu_uuids"] for d in manifest["observed_devices"]] == [["SN0"], ["SN1"]]
        assert Reason.GPU_UUID_CHANGED not in manifest["reason_codes"]

    def test_partitioned_gpus_sharing_a_serial_fail_device_identity(self, tmp_path):
        """Compute partitions report the parent's serial, so identity is no longer 1:1 with the index."""
        session = self._session(tmp_path, _amd_body(count=2, serial="SN", partition="0").replace("SN1", "SN0"))
        outcome = session.stop_and_finalize()
        assert Reason.GPU_UUID_CHANGED in outcome.reason_codes
        assert outcome.publication_valid is False

    def test_offline_validator_accepts_the_recorded_metric_and_requires_one(self, tmp_path):
        session = self._session(tmp_path, _amd_body(count=2))
        session.stop_and_finalize()

        report = validate_power_artifacts(power_dir=session.power_dir, result_root=tmp_path / "logs")
        assert not [f for f in report.failures if f.startswith(("source_metric", "power_scope"))], report.failures

        manifest_path = session.power_dir / MANIFEST_FILENAME
        manifest = json.loads(manifest_path.read_text())
        manifest["source_metric"] = ""
        manifest_path.write_text(json.dumps(manifest))
        report = validate_power_artifacts(power_dir=session.power_dir, result_root=tmp_path / "logs")
        assert "source_metric is not a non-empty string" in report.failures


def _srt_config(exporter: TelemetryExporterConfig) -> SrtConfig:
    return SrtConfig(
        name="test",
        model=ModelConfig(path="/model", container="/image", precision="fp8"),
        resources=ResourceConfig(gpu_type="mi355x"),
        benchmark=BenchmarkConfig(type="sa-bench", concurrencies=[4], placement=PlacementConfig(node="head")),
        telemetry=TelemetryConfig(enabled=True, dcgm_exporter=exporter),
    )


class TestSchema:
    def test_exporter_block_accepts_gpu_labels_and_metrics(self):
        config = _srt_config(AMD_EXPORTER)
        assert config.telemetry.dcgm_exporter.power_mapping == AMD

    def test_unset_blocks_are_dcgm(self):
        exporter = TelemetryExporterConfig.Schema().load({"container_image": "dcgm-exporter", "port": 9401})
        assert (exporter.gpu_labels, exporter.gpu_metrics) == (None, None)
        assert exporter.power_mapping == DCGM_POWER_MAPPING

    def test_dcgm_written_out_resolves_to_the_default(self):
        explicit = TelemetryExporterConfig(
            container_image="dcgm-exporter", port=9401, gpu_labels=DCGM_GPU_LABELS, gpu_metrics=DCGM_GPU_METRICS
        )
        assert explicit.power_mapping == DCGM_POWER_MAPPING

    def test_unset_optional_metrics_leave_their_columns_empty(self):
        exporter = TelemetryExporterConfig.Schema().load(
            {
                **AMD_GPU_CONFIG,
                "container_image": AMD_IMAGE,
                "port": 5000,
                "command": "x",
                "gpu_metrics": {"power": {"metric": "gpu_power_usage", "scope": AMD_SCOPE}},
            }
        )
        assert exporter.power_mapping.utilization_metrics == ()

    def test_unknown_metric_is_rejected_at_load(self):
        with pytest.raises(ValidationError, match="gpu_temp"):
            TelemetryExporterConfig.Schema().load(
                {
                    **AMD_GPU_CONFIG,
                    "container_image": AMD_IMAGE,
                    "port": 5000,
                    "gpu_metrics": {
                        "power": {"metric": "gpu_power_usage", "scope": AMD_SCOPE},
                        "gpu_temp": {"metric": "gpu_edge_temperature"},
                    },
                }
            )

    @pytest.mark.parametrize("drop", ["command", "gpu_labels", "gpu_metrics"])
    def test_a_custom_exporter_states_command_labels_and_metrics(self, drop):
        fields = {"container_image": AMD_IMAGE, "port": 5000, "command": "x", **AMD_GPU_CONFIG}
        del fields[drop]
        with pytest.raises(ValidationError, match="`kind: custom` GPU exporter needs"):
            TelemetryExporterConfig.Schema().load(fields)

    def test_an_unknown_kind_is_rejected(self):
        with pytest.raises(ValidationError, match="kind"):
            TelemetryExporterConfig.Schema().load({"container_image": "x", "port": 1, "kind": "amd"})

    def test_kind_not_the_metric_name_decides_dcgm(self):
        """Another DCGM power field stays DCGM: the default command and DCGM labels still apply."""
        exporter = TelemetryExporterConfig.Schema().load(
            {
                "container_image": "dcgm-exporter",
                "port": 9401,
                "gpu_metrics": {"power": {"metric": "DCGM_FI_DEV_POWER_USAGE_INSTANT", "scope": "board"}},
            }
        )
        assert exporter.kind == "dcgm" and exporter.command is None
        assert exporter.power_mapping.gpu_index_label == "gpu"
        assert TACHOMETER_SCRAPE[exporter.kind] == TACHOMETER_SCRAPE["dcgm"]


AMD_CLUSTER_EXPORTER = {
    "container_image": "amd-exporter",
    "command": "/home/amd/tools/entrypoint.sh",
    "port": 5000,
    **AMD_GPU_CONFIG,
}


def _recipe(telemetry):
    return {
        "schema": 2,
        "name": "test",
        "model": {"path": "/model", "container": "/image", "precision": "bf16"},
        "resources": {"gpu_type": "mi355x"},
        "roles": {"agg": {"nodes": 1}},
        "benchmark": {"type": "sa-bench", "concurrencies": [4]},
        "telemetry": telemetry,
    }


class TestClusterDefaultFlowsIntoPowerTelemetry:
    def test_enabling_telemetry_alone_inherits_the_cluster_exporter_with_its_power_block(self):
        resolved = resolve_config_with_defaults(
            _recipe({"enabled": True}),
            {"default_gpu_exporter": AMD_CLUSTER_EXPORTER, "containers": {"amd-exporter": AMD_IMAGE}},
        )
        config = SrtConfig.Schema().load(resolved)
        exporter = config.telemetry.dcgm_exporter
        assert (exporter.container_image, exporter.port, exporter.command, exporter.power_mapping) == (
            AMD_IMAGE,  # the alias resolved, exactly like a recipe exporter
            5000,
            "/home/amd/tools/entrypoint.sh",
            AMD,
        )
        # The tachometer copy is the same block, untouched by the power copy.
        assert config.observability.tachometer.default_gpu_exporter.power_mapping == AMD

    def test_a_recipe_exporter_wins(self):
        recipe_exporter = {"container_image": "dcgm-exporter", "port": 9401}
        resolved = resolve_config_with_defaults(
            _recipe({"enabled": True, "dcgm_exporter": recipe_exporter}),
            {"default_gpu_exporter": AMD_CLUSTER_EXPORTER},
        )
        assert resolved["telemetry"]["dcgm_exporter"] == recipe_exporter

    @pytest.mark.parametrize(
        "telemetry",
        [
            {"enabled": True, "cpu_power_exporter": {"port": 9405}},
            {"enabled": True, "cpu_power": {"enabled": True}},
            {"enabled": False},
        ],
        ids=["cpu-exporter-leg", "cpu-host-leg", "disabled"],
    )
    def test_cpu_only_and_disabled_telemetry_are_untouched(self, telemetry):
        resolved = resolve_config_with_defaults(_recipe(telemetry), {"default_gpu_exporter": AMD_CLUSTER_EXPORTER})
        assert "dcgm_exporter" not in resolved["telemetry"]

    def test_a_disabled_cpu_leg_still_inherits_the_gpu_exporter(self):
        resolved = resolve_config_with_defaults(
            _recipe({"enabled": True, "cpu_power": {"enabled": False}}),
            {"default_gpu_exporter": AMD_CLUSTER_EXPORTER},
        )
        assert resolved["telemetry"]["dcgm_exporter"]["kind"] == "custom"

    def test_a_cluster_without_a_gpu_exporter_leaves_the_original_validation_error(self):
        resolved = resolve_config_with_defaults(_recipe({"enabled": True}), {"default_gpu_exporter": None})
        with pytest.raises(ValidationError, match="nothing to collect"):
            SrtConfig.Schema().load(resolved)


def _harness(tmp_path, exporter, processes):
    class Harness(TelemetryStageMixin):
        def __init__(self):
            self.config = SrtConfig(
                name="test",
                model=ModelConfig(path="/model", container="/image", precision="fp8"),
                resources=ResourceConfig(gpu_type="mi355x"),
                benchmark=BenchmarkConfig(type="sa-bench", concurrencies=[4], placement=PlacementConfig(node="head")),
                telemetry=TelemetryConfig(
                    enabled=True,
                    dcgm_exporter=exporter,
                    startup_timeout_seconds=0.2,
                    request_timeout_seconds=0.1,
                    collector_join_timeout_seconds=3.0,
                ),
            )
            self.runtime = MagicMock()
            self.runtime.log_dir = Path(tmp_path)
            self.runtime.job_id = "12345"
            self.runtime.run_name = "recipe_12345"
            self.runtime.network_interface = "eth0"
            self.runtime.nodes.head = "node-a"
            self.runtime.nodes.het = False
            self.runtime.nodes.compute = ()
            self.runtime.srun_options = {}
            self.runtime.container_mounts = {Path(tmp_path): Path("/logs")}

        @property
        def backend_processes(self):
            return processes

    return Harness()


class TestTelemetryStage:
    def test_dcgm_is_the_default_command_and_a_power_block_brings_its_own(self):
        dcgm = TelemetryExporterConfig(container_image="dcgm-exporter", port=9401)
        assert resolve_exporter_command(dcgm, DCGM_EXPORTER_COMMAND_TEMPLATE) == (
            "dcgm-exporter --collect-interval=100 --address :9401"
        )
        assert resolve_exporter_command(AMD_EXPORTER, DCGM_EXPORTER_COMMAND_TEMPLATE) == "/home/amd/tools/entrypoint.sh"

    @patch("srtctl.cli.mixins.telemetry_stage.start_srun_process")
    def test_amd_exporter_launches_with_its_command_and_the_session_parses_its_metric(self, mock_srun, tmp_path):
        popen = MagicMock()
        popen.poll.return_value = None
        mock_srun.return_value = popen
        harness = _harness(tmp_path, AMD_EXPORTER, [_worker(gpus=range(2))])

        with patch("srtctl.core.power.session.requests.get", return_value=_FakeResponse(_amd_body(count=2))):
            session = harness.start_power_telemetry(ProcessRegistry(job_id="12345"))
            assert session is not None
            ready = harness._power_telemetry_ready
            session.stop_and_finalize()

        kwargs = mock_srun.call_args.kwargs
        assert kwargs["command"] == ["/home/amd/tools/entrypoint.sh"]
        assert kwargs["container_image"] == AMD_IMAGE
        assert ready is True, "readiness needs every expected GPU parsed from the AMD metric"
        manifest = json.loads((tmp_path / "power" / MANIFEST_FILENAME).read_text())
        assert manifest["source_metric"] == "gpu_power_usage"
        assert manifest["dcgm_exporter"]["command"] == "/home/amd/tools/entrypoint.sh"
        rows, _ = read_samples(tmp_path / "power" / "samples.csv")
        assert {row.gpu_uuid for row in rows} == {"SN0", "SN1"}

    def test_tachometer_target_filter_follows_the_mapping(self, tmp_path):
        processes = [_worker(gpus=range(2))]
        amd = _harness(tmp_path, AMD_EXPORTER, processes)
        dcgm = _harness(tmp_path, TelemetryExporterConfig(container_image="dcgm-exporter", port=9401), processes)

        (amd_target,) = amd._power_exporter_targets()
        (dcgm_target,) = dcgm._power_exporter_targets()
        assert (amd_target.endpoint_name, amd_target.url, amd_target.filter, amd_target.gpu_metadata) == (
            "gpu-power_node-a",
            "http://node-a:5000/metrics",
            "passthrough",
            False,
        )
        assert (dcgm_target.endpoint_name, dcgm_target.filter, dcgm_target.gpu_metadata) == (
            "dcgm_node-a",
            "dcgm",
            True,
        )


def test_dry_run_names_the_power_metric(capsys):
    from srtctl.cli.submit import show_config_details

    show_config_details(_srt_config(AMD_EXPORTER))
    out = capsys.readouterr().out
    assert "power_metric" in out and "gpu_power_usage" in out
    show_config_details(_srt_config(TelemetryExporterConfig(container_image="dcgm-exporter", port=9401)))
    assert "DCGM_FI_DEV_POWER_USAGE" in capsys.readouterr().out


class TestTemperature:
    def test_dcgm_records_gpu_temp_by_default(self):
        assert DCGM_POWER_MAPPING.temperature_metric == "DCGM_FI_DEV_GPU_TEMP"
        payload = _manifest().to_dict()
        assert payload["temperature_metric"] == "DCGM_FI_DEV_GPU_TEMP"

    def test_configured_temperature_metric_fills_the_column_by_the_mapping_labels(self):
        exporter = TelemetryExporterConfig.Schema().load(
            {
                **AMD_GPU_CONFIG,
                "container_image": AMD_IMAGE,
                "port": 5000,
                "command": "x",
                "gpu_metrics": {
                    **AMD_GPU_CONFIG["gpu_metrics"],
                    "temperature": {"metric": "gpu_junction_temperature"},
                },
            }
        )
        mapping = exporter.power_mapping
        body = _amd_body(count=1) + (
            "# TYPE gpu_junction_temperature gauge\n"
            'gpu_junction_temperature{gpu_id="0",serial_number="SN0"} 61.0\n'
            # DCGM's temperature means nothing to this mapping.
            'DCGM_FI_DEV_GPU_TEMP{gpu_id="0",serial_number="SN0"} 99.0\n'
        )
        assert [r.temperature_c for r in parse_power_scrape(body, mapping).readings] == [61.0]
        assert _manifest(mapping).to_dict()["temperature_metric"] == "gpu_junction_temperature"

    def test_unset_temperature_leaves_the_column_empty(self):
        assert AMD.temperature_metric is None
        body = _amd_body(count=1) + 'gpu_junction_temperature{gpu_id="0",serial_number="SN0"} 61.0\n'
        assert [r.temperature_c for r in parse_power_scrape(body, AMD).readings] == [None]
