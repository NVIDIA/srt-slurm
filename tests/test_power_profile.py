# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU power exporter profiles: the table that makes the power collector exporter-agnostic.

The DCGM row must reproduce the pre-profile contract byte for byte on the wire
keys it owns; the AMD row (rocm/device-metrics-exporter) exercises every place
that used to hard-code a DCGM metric or label.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from marshmallow import ValidationError

from srtctl.cli.mixins.telemetry_stage import TelemetryStageMixin, resolve_exporter_command
from srtctl.core.config import resolve_config_with_defaults
from srtctl.core.power.contract import MANIFEST_FILENAME, UTILIZATION_METRICS, Reason
from srtctl.core.power.manifest import DcgmExporterIdentity, ExpectedWindow, PowerManifest
from srtctl.core.power.parser import parse_power_scrape
from srtctl.core.power.profile import (
    AMD_DEVICE_METRICS_POWER_PROFILE,
    DCGM_POWER_PROFILE,
    DEFAULT_POWER_PROFILE,
    POWER_PROFILES,
)
from srtctl.core.power.samples import SampleRow, read_samples
from srtctl.core.power.session import PowerEndpoint, PowerSessionSettings, PowerTelemetrySession
from srtctl.core.power.topology import build_expected_devices
from srtctl.core.power.validate_artifacts import validate_power_artifacts
from srtctl.core.processes import ProcessRegistry
from srtctl.core.schema import (
    BenchmarkConfig,
    ModelConfig,
    PlacementConfig,
    ResourceConfig,
    SrtConfig,
    TelemetryConfig,
    TelemetryExporterConfig,
)
from srtctl.core.topology import Process

AMD = AMD_DEVICE_METRICS_POWER_PROFILE
AMD_IMAGE = "docker://rocm/device-metrics-exporter:v1.5.2"


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


class TestProfileTable:
    def test_rows_are_keyed_by_their_own_name_and_dcgm_is_the_default(self):
        assert {name: profile.name for name, profile in POWER_PROFILES.items()} == {
            name: name for name in POWER_PROFILES
        }
        assert DEFAULT_POWER_PROFILE is DCGM_POWER_PROFILE

    def test_dcgm_row_pins_the_pre_profile_contract(self):
        """The NVIDIA path must be unchanged: same metric, labels, riders and launch command."""
        assert DCGM_POWER_PROFILE.power_metric == "DCGM_FI_DEV_POWER_USAGE"
        assert DCGM_POWER_PROFILE.power_scope == "gpu_device_board_as_reported_by_dcgm"
        assert (DCGM_POWER_PROFILE.gpu_index_label, DCGM_POWER_PROFILE.gpu_identity_label) == ("gpu", "UUID")
        assert DCGM_POWER_PROFILE.utilization_metrics == UTILIZATION_METRICS
        assert DCGM_POWER_PROFILE.instance_labels == ("GPU_I_ID", "GPU_I_PROFILE")
        assert DCGM_POWER_PROFILE.default_command_template == "dcgm-exporter --collect-interval=100 --address :{port}"
        assert (DCGM_POWER_PROFILE.tachometer_filter, DCGM_POWER_PROFILE.tachometer_gpu_metadata) == ("dcgm", True)

    def test_amd_row_matches_the_device_metrics_exporter_exposition(self):
        assert AMD.power_metric == "gpu_power_usage"
        assert (AMD.gpu_index_label, AMD.gpu_identity_label) == ("gpu_id", "serial_number")
        assert [(m.column, m.metric) for m in AMD.utilization_metrics] == [("gpu_util_pct", "gpu_gfx_activity")]
        assert AMD.instance_labels == ()
        assert AMD.default_command_template == "/home/amd/tools/entrypoint.sh"
        assert (AMD.tachometer_filter, AMD.tachometer_gpu_metadata) == ("passthrough", False)

    def test_every_row_fills_only_contract_columns_with_contract_semantics(self):
        contract = {metric.column: metric for metric in UTILIZATION_METRICS}
        row_fields = {field.name for field in dataclasses.fields(SampleRow)}
        for profile in POWER_PROFILES.values():
            for metric in profile.utilization_metrics:
                assert metric.column in contract and metric.column in row_fields
                assert (metric.unit, metric.max_value) == (
                    contract[metric.column].unit,
                    contract[metric.column].max_value,
                )


class TestParsingByProfile:
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
        ("body", "profile"),
        [(_dcgm_body(), AMD), (_amd_body(), DCGM_POWER_PROFILE)],
        ids=["dcgm-body-under-amd-row", "amd-body-under-dcgm-row"],
    )
    def test_a_body_from_another_exporter_is_a_missing_power_metric_not_a_reading(self, body, profile):
        scrape = parse_power_scrape(body, profile)
        assert scrape.readings == ()
        assert scrape.reason_codes == (Reason.POWER_METRIC_MISSING,)

    def test_default_profile_still_parses_dcgm(self):
        scrape = parse_power_scrape(_dcgm_body(count=2))
        assert [(r.gpu_index, r.gpu_uuid) for r in scrape.readings] == [(0, "GPU-0"), (1, "GPU-1")]

    def test_missing_identity_label_is_reported_under_the_uuid_reason(self):
        body = '# TYPE gpu_power_usage gauge\ngpu_power_usage{gpu_id="0",serial_number=""} 500\n'
        scrape = parse_power_scrape(body, AMD)
        assert scrape.readings == () and Reason.GPU_UUID_MISSING in scrape.reason_codes

    def test_instance_labels_are_the_dcgm_rows_concern(self):
        """A MIG label means nothing to a row that declares no instance labels."""
        body = '# TYPE gpu_power_usage gauge\ngpu_power_usage{gpu_id="0",serial_number="SN0",GPU_I_ID="1"} 500\n'
        assert parse_power_scrape(body, AMD).readings[0].gpu_uuid == "SN0"
        dcgm = '# TYPE DCGM_FI_DEV_POWER_USAGE gauge\nDCGM_FI_DEV_POWER_USAGE{gpu="0",UUID="GPU-0",GPU_I_ID="1"} 500\n'
        assert Reason.MIG_INSTANCE_UNSUPPORTED in parse_power_scrape(dcgm).reason_codes


def _manifest(profile=None):
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
    if profile is not None:
        fields["profile"] = profile
    return PowerManifest(**fields)


class TestManifestProvenance:
    def test_default_manifest_keeps_the_dcgm_wire_values_and_names_its_row(self):
        payload = _manifest().to_dict()
        assert payload["power_profile"] == "dcgm"
        assert payload["source_metric"] == "DCGM_FI_DEV_POWER_USAGE"
        assert payload["power_scope"] == "gpu_device_board_as_reported_by_dcgm"
        assert [m["source_metric"] for m in payload["utilization_metrics"]] == [
            "DCGM_FI_DEV_GPU_UTIL",
            "DCGM_FI_PROF_SM_ACTIVE",
        ]

    def test_amd_manifest_describes_the_amd_measurement(self):
        payload = _manifest(AMD).to_dict()
        assert payload["producer"] == "srt-slurm.dcgm-power"  # the artifact producer, not the exporter
        assert payload["power_profile"] == "amd-device-metrics"
        assert payload["source_metric"] == "gpu_power_usage"
        assert payload["power_scope"] == AMD.power_scope
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
            profile=AMD,
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

    def test_samples_carry_the_serial_as_identity_and_the_manifest_names_the_row(self, tmp_path):
        session = self._session(tmp_path, _amd_body(count=2, utilization=True))
        session.stop_and_finalize()

        rows, reasons = read_samples(session.samples_path)
        assert reasons == ()
        assert [(r.hostname, r.gpu_index, r.gpu_uuid, r.power_w, r.gpu_util_pct, r.sm_active) for r in rows] == [
            ("node-a", 0, "SN0", 500.0, 0.0, None),
            ("node-a", 1, "SN1", 501.0, 10.0, None),
        ]
        manifest = json.loads((session.power_dir / MANIFEST_FILENAME).read_text())
        assert manifest["power_profile"] == "amd-device-metrics"
        assert manifest["source_metric"] == "gpu_power_usage"
        assert [d["gpu_uuids"] for d in manifest["observed_devices"]] == [["SN0"], ["SN1"]]
        assert Reason.GPU_UUID_CHANGED not in manifest["reason_codes"]

    def test_partitioned_gpus_sharing_a_serial_fail_device_identity(self, tmp_path):
        """Compute partitions report the parent's serial, so identity is no longer 1:1 with the index."""
        session = self._session(tmp_path, _amd_body(count=2, serial="SN", partition="0").replace("SN1", "SN0"))
        outcome = session.stop_and_finalize()
        assert Reason.GPU_UUID_CHANGED in outcome.reason_codes
        assert outcome.publication_valid is False

    def test_offline_validator_checks_the_wire_keys_against_the_named_row(self, tmp_path):
        session = self._session(tmp_path, _amd_body(count=2))
        session.stop_and_finalize()

        report = validate_power_artifacts(power_dir=session.power_dir, result_root=tmp_path / "logs")
        wire_failures = [
            f for f in report.failures if f.split(" is ")[0] in ("source_metric", "power_scope", "power_profile")
        ]
        assert wire_failures == [], report.failures

        manifest_path = session.power_dir / MANIFEST_FILENAME
        manifest = json.loads(manifest_path.read_text())
        manifest["power_profile"] = "nvml"
        manifest_path.write_text(json.dumps(manifest))
        report = validate_power_artifacts(power_dir=session.power_dir, result_root=tmp_path / "logs")
        assert any(f.startswith("power_profile is 'nvml'") for f in report.failures)

    def test_a_manifest_written_before_profiles_is_read_as_dcgm(self, tmp_path):
        session = self._session(tmp_path, _amd_body(count=2))
        session.stop_and_finalize()
        manifest_path = session.power_dir / MANIFEST_FILENAME
        manifest = json.loads(manifest_path.read_text())
        del manifest["power_profile"]
        manifest_path.write_text(json.dumps(manifest))

        report = validate_power_artifacts(power_dir=session.power_dir, result_root=tmp_path / "logs")
        assert any(
            f.startswith("source_metric is 'gpu_power_usage', expected 'DCGM_FI_DEV_POWER_USAGE'")
            for f in report.failures
        )


def _srt_config(exporter: TelemetryExporterConfig) -> SrtConfig:
    return SrtConfig(
        name="test",
        model=ModelConfig(path="/model", container="/image", precision="fp8"),
        resources=ResourceConfig(gpu_type="mi355x"),
        benchmark=BenchmarkConfig(type="sa-bench", concurrencies=[4], placement=PlacementConfig(node="head")),
        telemetry=TelemetryConfig(enabled=True, dcgm_exporter=exporter),
    )


class TestSchema:
    def test_exporter_block_accepts_a_known_profile(self):
        exporter = TelemetryExporterConfig.Schema().load(
            {"container_image": AMD_IMAGE, "port": 5000, "power_profile": "amd-device-metrics"}
        )
        config = _srt_config(exporter)
        assert config.telemetry.dcgm_exporter.power_profile == "amd-device-metrics"

    def test_unset_profile_is_the_dcgm_row(self):
        exporter = TelemetryExporterConfig.Schema().load({"container_image": "dcgm-exporter", "port": 9401})
        assert POWER_PROFILES[exporter.power_profile] is DCGM_POWER_PROFILE

    def test_unknown_profile_is_rejected_with_the_known_rows(self):
        with pytest.raises(
            ValidationError, match="power_profile='nvml' is unknown; known profiles: amd-device-metrics, dcgm"
        ):
            _srt_config(TelemetryExporterConfig(container_image="x", port=5000, power_profile="nvml"))


AMD_CLUSTER_EXPORTER = {
    "container_image": "amd-exporter",
    "command": "/home/amd/tools/entrypoint.sh",
    "port": 5000,
    "power_profile": "amd-device-metrics",
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
    def test_enabling_telemetry_alone_inherits_the_cluster_exporter_with_its_profile(self):
        resolved = resolve_config_with_defaults(
            _recipe({"enabled": True}),
            {"default_gpu_exporter": AMD_CLUSTER_EXPORTER, "containers": {"amd-exporter": AMD_IMAGE}},
        )
        config = SrtConfig.Schema().load(resolved)
        exporter = config.telemetry.dcgm_exporter
        assert (exporter.container_image, exporter.port, exporter.command, exporter.power_profile) == (
            AMD_IMAGE,  # the alias resolved, exactly like a recipe exporter
            5000,
            "/home/amd/tools/entrypoint.sh",
            "amd-device-metrics",
        )
        # The tachometer copy is the same row, untouched by the power copy.
        assert config.observability.tachometer.default_gpu_exporter.power_profile == "amd-device-metrics"

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
    def test_default_command_comes_from_the_row(self):
        amd = TelemetryExporterConfig(container_image=AMD_IMAGE, port=5000, power_profile="amd-device-metrics")
        dcgm = TelemetryExporterConfig(container_image="dcgm-exporter", port=9401)
        assert resolve_exporter_command(amd, AMD.default_command_template) == "/home/amd/tools/entrypoint.sh"
        assert resolve_exporter_command(dcgm, POWER_PROFILES[dcgm.power_profile].default_command_template) == (
            "dcgm-exporter --collect-interval=100 --address :9401"
        )

    @patch("srtctl.cli.mixins.telemetry_stage.start_srun_process")
    def test_amd_exporter_launches_with_the_row_command_and_the_session_parses_its_metric(self, mock_srun, tmp_path):
        popen = MagicMock()
        popen.poll.return_value = None
        mock_srun.return_value = popen
        exporter = TelemetryExporterConfig(container_image=AMD_IMAGE, port=5000, power_profile="amd-device-metrics")
        harness = _harness(tmp_path, exporter, [_worker(gpus=range(2))])

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
        assert manifest["power_profile"] == "amd-device-metrics"
        assert manifest["dcgm_exporter"]["command"] == "/home/amd/tools/entrypoint.sh"
        rows, _ = read_samples(tmp_path / "power" / "samples.csv")
        assert {row.gpu_uuid for row in rows} == {"SN0", "SN1"}

    def test_tachometer_target_filter_follows_the_row(self, tmp_path):
        processes = [_worker(gpus=range(2))]
        amd = _harness(
            tmp_path,
            TelemetryExporterConfig(container_image=AMD_IMAGE, port=5000, power_profile="amd-device-metrics"),
            processes,
        )
        dcgm = _harness(tmp_path, TelemetryExporterConfig(container_image="dcgm-exporter", port=9401), processes)

        (amd_target,) = amd._power_exporter_targets()
        (dcgm_target,) = dcgm._power_exporter_targets()
        assert (amd_target.endpoint_name, amd_target.url, amd_target.filter, amd_target.gpu_metadata) == (
            "amd-device-metrics_node-a",
            "http://node-a:5000/metrics",
            "passthrough",
            False,
        )
        assert (dcgm_target.endpoint_name, dcgm_target.filter, dcgm_target.gpu_metadata) == (
            "dcgm_node-a",
            "dcgm",
            True,
        )


def test_dry_run_names_the_profile(capsys):
    from srtctl.cli.submit import show_config_details

    config = _srt_config(
        TelemetryExporterConfig(container_image=AMD_IMAGE, port=5000, power_profile="amd-device-metrics")
    )
    show_config_details(config)
    out = capsys.readouterr().out
    assert "power_profile" in out and "amd-device-metrics" in out
    config = _srt_config(TelemetryExporterConfig(container_image="dcgm-exporter", port=9401))
    show_config_details(config)
    assert "power_profile" in capsys.readouterr().out
