# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Raw artifact contract for the dcgm-power telemetry provider."""

import csv
import json
import os
from contextlib import suppress
from pathlib import Path

import pytest

import srtctl.core.power.parser as power_parser
from srtctl.core.power.contract import (
    MANIFEST_FILENAME,
    SAMPLES_FILENAME,
    SAMPLES_HEADER,
    SCHEMA_VERSION,
    WINDOWS_DIRNAME,
    Reason,
    atomic_write_json,
    is_safe_relative_subpath,
)
from srtctl.core.power.manifest import (
    DcgmExporterIdentity,
    ExpectedWindow,
    PowerManifest,
)
from srtctl.core.power.parser import parse_power_scrape
from srtctl.core.power.samples import (
    ObservedDevice,
    SampleRow,
    SampleWriter,
    derive_observed_devices,
    read_samples,
)
from srtctl.core.power.topology import (
    ExpectedDevice,
    build_expected_devices,
    resolve_het_groups,
    resolve_roles,
    validate_devices,
)
from srtctl.core.power.windows import convert_running_windows, validate_expected_windows
from srtctl.core.topology import Process


def _process(node, gpus, mode, index=0, node_rank=0, het_group=None):
    return Process(
        node=node,
        gpu_indices=frozenset(gpus),
        sys_port=8080,
        http_port=30000,
        endpoint_mode=mode,
        endpoint_index=index,
        node_rank=node_rank,
        het_group=het_group,
    )


def _metric(gpu, uuid, value, **labels):
    label_text = ",".join(
        [f'gpu="{gpu}"', f'UUID="{uuid}"', *[f'{k}="{v}"' for k, v in labels.items()]],
    )
    return f"DCGM_FI_DEV_POWER_USAGE{{{label_text}}} {value}"


PREAMBLE = "# HELP DCGM_FI_DEV_POWER_USAGE Power draw (in W).\n# TYPE DCGM_FI_DEV_POWER_USAGE gauge\n"


def _scrape(*metrics, preamble=PREAMBLE):
    return preamble + "\n".join(metrics) + "\n"


class TestDcgmParser:
    """Strict DCGM power parsing."""

    def test_parses_power_and_ignores_unrelated_metrics(self):
        text = _scrape(
            _metric(0, "GPU-aaa", 412.5, device="nvidia0", modelName="NVIDIA GB200"),
            _metric(1, "GPU-bbb", 98.25, device="nvidia1", modelName="NVIDIA GB200"),
        )
        text += '# TYPE DCGM_FI_DEV_GPU_TEMP gauge\nDCGM_FI_DEV_GPU_TEMP{gpu="0",UUID="GPU-aaa"} 42\n'

        scrape = parse_power_scrape(text)

        assert [(r.gpu_index, r.gpu_uuid, r.power_w) for r in scrape.readings] == [
            (0, "GPU-aaa", 412.5),
            (1, "GPU-bbb", 98.25),
        ]
        assert scrape.reason_codes == ()

    def test_label_order_and_escapes_do_not_change_identity(self):
        text = (
            PREAMBLE + 'DCGM_FI_DEV_POWER_USAGE{modelName="NVIDIA GB200 \\"Grace\\"",UUID="GPU-aaa",'
            'Hostname="",gpu="3",device="nvidia3"} 100.0\n'
        )

        scrape = parse_power_scrape(text)

        assert [(r.gpu_index, r.gpu_uuid) for r in scrape.readings] == [(3, "GPU-aaa")]

    def test_devices_are_ordered_by_index_regardless_of_emission_order(self):
        text = _scrape(
            _metric(2, "GPU-ccc", 102.0),
            _metric(0, "GPU-aaa", 100.0),
            _metric(1, "GPU-bbb", 101.0),
        )

        scrape = parse_power_scrape(text)

        assert [r.gpu_index for r in scrape.readings] == [0, 1, 2]
        assert [r.gpu_uuid for r in scrape.readings] == ["GPU-aaa", "GPU-bbb", "GPU-ccc"]

    def test_duplicate_device_metric_omits_row_and_reports(self):
        text = _scrape(
            _metric(0, "GPU-aaa", 100.0),
            _metric(0, "GPU-aaa", 101.0),
            _metric(1, "GPU-bbb", 102.0),
        )

        scrape = parse_power_scrape(text)

        assert [r.gpu_index for r in scrape.readings] == [1]
        assert Reason.DUPLICATE_POWER_METRIC in scrape.reason_codes

    def test_mig_instance_is_rejected_without_marking_metric_missing(self):
        text = _scrape(_metric(0, "GPU-aaa", 100.0, GPU_I_ID="3", GPU_I_PROFILE="1g.10gb"))

        scrape = parse_power_scrape(text)

        assert scrape.readings == ()
        assert scrape.reason_codes == (Reason.MIG_INSTANCE_UNSUPPORTED,)

    @pytest.mark.parametrize(
        ("value", "reason"),
        [
            ("NaN", Reason.INVALID_POWER_VALUE),
            ("-1.0", Reason.INVALID_POWER_VALUE),
        ],
    )
    def test_invalid_power_values_are_rejected(self, value, reason):
        scrape = parse_power_scrape(_scrape(_metric(0, "GPU-aaa", value)))

        assert scrape.readings == ()
        assert reason in scrape.reason_codes

    def test_missing_identity_labels_are_rejected(self):
        text = PREAMBLE + 'DCGM_FI_DEV_POWER_USAGE{UUID="GPU-aaa"} 100.0\n' + 'DCGM_FI_DEV_POWER_USAGE{gpu="1"} 100.0\n'

        scrape = parse_power_scrape(text)

        assert scrape.readings == ()
        assert Reason.GPU_INDEX_MISSING in scrape.reason_codes
        assert Reason.GPU_UUID_MISSING in scrape.reason_codes

    def test_absent_power_family_reports_missing_metric(self):
        scrape = parse_power_scrape('# TYPE DCGM_FI_DEV_GPU_TEMP gauge\nDCGM_FI_DEV_GPU_TEMP{gpu="0"} 42\n')

        assert scrape.readings == ()
        assert Reason.POWER_METRIC_MISSING in scrape.reason_codes

    @pytest.mark.parametrize(
        "error",
        [
            ValueError("malformed exposition"),
            KeyError("histogram field"),
            IndexError("short directive in prometheus-client 0.20"),
        ],
    )
    def test_exporter_parse_failures_are_classified_without_escaping(self, monkeypatch, error):
        def fail(_text):
            raise error

        monkeypatch.setattr(power_parser, "text_string_to_metric_families", fail)

        scrape = parse_power_scrape("malformed")

        assert scrape.readings == ()
        assert scrape.reason_codes == ("endpoint_parse_error",)

    def test_parser_control_flow_exceptions_are_not_hidden(self, monkeypatch):
        def fail(_text):
            raise KeyboardInterrupt

        monkeypatch.setattr(power_parser, "text_string_to_metric_families", fail)

        with pytest.raises(KeyboardInterrupt):
            parse_power_scrape("malformed")


class TestExpectedTopology:
    """Expected devices derived from srt-slurm backend processes."""

    def test_empty_device_assignments_are_rejected(self):
        with pytest.raises(ValueError, match="at least one assignment"):
            ExpectedDevice(hostname="node-a", gpu_index=0, assignments=())

    def test_aggregated_topology(self):
        processes = [_process("node-a", range(4), "agg")]

        devices = build_expected_devices(processes)

        assert [(d.hostname, d.gpu_index) for d in devices] == [("node-a", i) for i in range(4)]
        assert {a.worker_role for d in devices for a in d.assignments} == {"agg"}
        assert resolve_roles(devices) == ({("node-a", i): "agg" for i in range(4)}, ())
        assert resolve_het_groups(devices) == ({"node-a": None}, ())

    def test_disaggregated_1p1d_topology(self):
        processes = [
            _process("node-a", range(4), "prefill", het_group=0),
            _process("node-b", range(4), "decode", het_group=1),
        ]

        devices = build_expected_devices(processes)

        roles, role_conflicts = resolve_roles(devices)
        groups, group_conflicts = resolve_het_groups(devices)

        assert len(devices) == 8
        assert sorted(roles.values()).count("prefill") == 4
        assert sorted(roles.values()).count("decode") == 4
        assert groups == {"node-a": 0, "node-b": 1}
        assert role_conflicts == () and group_conflicts == ()

    def test_repeated_same_role_assignments_are_allowed(self):
        processes = [
            _process("node-a", [0, 1], "decode", index=0),
            _process("node-a", [0, 1], "decode", index=1),
        ]

        devices = build_expected_devices(processes)

        assert len(devices) == 2
        assert len(devices[0].assignments) == 2
        assert resolve_roles(devices)[1] == ()

    def test_conflicting_roles_on_one_device_are_invalid(self):
        processes = [
            _process("node-a", [0], "prefill"),
            _process("node-a", [0], "decode"),
        ]

        roles, conflicts = resolve_roles(build_expected_devices(processes))

        assert roles == {}
        assert Reason.CONFLICTING_WORKER_ROLES in conflicts

    def test_conflicting_het_groups_on_one_node_are_invalid(self):
        processes = [
            _process("node-a", [0], "prefill", het_group=0),
            _process("node-a", [1], "decode", het_group=1),
        ]

        groups, conflicts = resolve_het_groups(build_expected_devices(processes))

        assert groups == {}
        assert Reason.CONFLICTING_HET_GROUPS in conflicts


class TestSampleArtifact:
    """samples.csv round trip."""

    def test_header_constant_is_pinned(self):
        """Writer and reader both consume the constant, so pin it literally."""
        assert SAMPLES_HEADER == (
            "schema_version",
            "timestamp_unix",
            "scrape_seq",
            "hostname",
            "gpu_index",
            "gpu_uuid",
            "power_w",
        )

    def test_round_trip_preserves_rows_and_derives_devices(self, tmp_path):
        path = tmp_path / SAMPLES_FILENAME
        writer = SampleWriter(path)
        writer.append(
            [
                SampleRow(1000.0, 0, "node-a", 0, "GPU-aaa", 400.0),
                SampleRow(1000.1, 0, "node-b", 0, "GPU-bbb", 401.0),
            ]
        )
        writer.append([SampleRow(1001.0, 1, "node-a", 0, "GPU-aaa", 402.0)])
        writer.close()

        rows, reasons = read_samples(path)

        with open(path, newline="") as handle:
            assert next(csv.reader(handle)) == list(SAMPLES_HEADER)
        assert reasons == ()
        assert writer.row_count == 3
        assert [row.schema_version for row in rows] == [SCHEMA_VERSION] * 3
        observed = derive_observed_devices(rows)
        assert [(d.hostname, d.gpu_index, d.gpu_uuids) for d in observed] == [
            ("node-a", 0, ("GPU-aaa",)),
            ("node-b", 0, ("GPU-bbb",)),
        ]
        assert observed[0].first_sample_time_unix == 1000.0
        assert observed[0].last_sample_time_unix == 1001.0

    def test_writer_closes_handle_when_header_write_fails(self, tmp_path, monkeypatch):
        opened = []

        def tracking_open(*args, **kwargs):
            handle = open(*args, **kwargs)  # noqa: SIM115 - retain the handle to assert explicit cleanup
            opened.append(handle)
            return handle

        class FailingWriter:
            def writerow(self, _row):
                raise OSError("disk full")

        monkeypatch.setattr("srtctl.core.power.samples.open", tracking_open, raising=False)
        monkeypatch.setattr("srtctl.core.power.samples.csv.writer", lambda _handle: FailingWriter())

        with pytest.raises(OSError, match="disk full"):
            SampleWriter(tmp_path / SAMPLES_FILENAME)

        assert len(opened) == 1
        assert opened[0].closed is True

    def test_uuid_change_is_retained_and_flagged(self, tmp_path):
        path = tmp_path / SAMPLES_FILENAME
        writer = SampleWriter(path)
        writer.append(
            [
                SampleRow(1000.0, 0, "node-a", 0, "GPU-aaa", 400.0),
                SampleRow(1001.0, 1, "node-a", 0, "GPU-zzz", 400.0),
            ]
        )
        writer.close()

        observed = derive_observed_devices(read_samples(path)[0])

        assert observed[0].gpu_uuids == ("GPU-aaa", "GPU-zzz")

    @pytest.mark.parametrize(
        ("bad_row", "reason"),
        [
            ("1,1000.0,0,node-a,0,GPU-aaa,not-a-number", Reason.SAMPLES_CSV_MALFORMED),
            ("1,1000.0,0,node-a,0,GPU-aaa,NaN", Reason.SAMPLES_CSV_MALFORMED),
            ("1,1000.0,0,node-a,0,GPU-aaa,-5", Reason.SAMPLES_CSV_MALFORMED),
            ("1,1000.0,0,node-a,0,GPU-aaa", Reason.SAMPLES_CSV_MALFORMED),
        ],
    )
    def test_malformed_rows_are_reported(self, tmp_path, bad_row, reason):
        path = tmp_path / SAMPLES_FILENAME
        path.write_text(",".join(SAMPLES_HEADER) + "\n" + bad_row + "\n")

        rows, reasons = read_samples(path)

        assert rows == ()
        assert reason in reasons

    def test_duplicate_row_key_is_reported(self, tmp_path):
        path = tmp_path / SAMPLES_FILENAME
        path.write_text(
            ",".join(SAMPLES_HEADER) + "\n1,1000.0,0,node-a,0,GPU-aaa,400.0\n1,1000.5,0,node-a,0,GPU-aaa,401.0\n"
        )

        _, reasons = read_samples(path)

        assert Reason.DUPLICATE_SAMPLE_ROW in reasons

    def test_non_monotonic_device_timestamps_are_reported(self, tmp_path):
        path = tmp_path / SAMPLES_FILENAME
        path.write_text(
            ",".join(SAMPLES_HEADER) + "\n1,1001.0,0,node-a,0,GPU-aaa,400.0\n1,1000.0,1,node-a,0,GPU-aaa,401.0\n"
        )

        _, reasons = read_samples(path)

        assert Reason.TIMESTAMP_NON_MONOTONIC in reasons

    def test_equal_device_timestamps_are_allowed(self, tmp_path):
        path = tmp_path / SAMPLES_FILENAME
        path.write_text(
            ",".join(SAMPLES_HEADER) + "\n1,1000.0,0,node-a,0,GPU-aaa,400.0\n1,1000.0,1,node-a,0,GPU-aaa,401.0\n"
        )

        _, reasons = read_samples(path)

        assert Reason.TIMESTAMP_NON_MONOTONIC not in reasons

    def test_invalid_utf8_bytes_are_reported(self, tmp_path):
        path = tmp_path / SAMPLES_FILENAME
        path.write_bytes(",".join(SAMPLES_HEADER).encode() + b"\n1,1000.0,0,node-\xff\xfe,0,GPU-aaa,400.0\n")

        rows, reasons = read_samples(path)

        assert rows == ()
        assert Reason.SAMPLES_CSV_MALFORMED in reasons

    def test_oversized_field_is_reported(self, tmp_path):
        path = tmp_path / SAMPLES_FILENAME
        giant = "x" * (csv.field_size_limit() + 1)
        path.write_text(",".join(SAMPLES_HEADER) + f"\n1,1000.0,0,{giant},0,GPU-aaa,400.0\n")

        rows, reasons = read_samples(path)

        assert rows == ()
        assert Reason.SAMPLES_CSV_MALFORMED in reasons

    def test_header_mismatch_and_missing_file_are_reported(self, tmp_path):
        wrong = tmp_path / "wrong.csv"
        wrong.write_text("timestamp,power\n")

        assert Reason.SAMPLES_CSV_HEADER_MISMATCH in read_samples(wrong)[1]
        assert Reason.SAMPLES_CSV_MISSING in read_samples(tmp_path / "nope.csv")[1]


class TestDeviceValidation:
    """Identity and topology gates."""

    def _expected(self):
        return build_expected_devices(
            [
                _process("node-a", [0, 1], "prefill", het_group=0),
                _process("node-b", [0, 1], "decode", het_group=1),
            ]
        )

    def _rows(self):
        return [
            SampleRow(1000.0, 0, "node-a", 0, "GPU-a0", 400.0),
            SampleRow(1000.0, 0, "node-a", 1, "GPU-a1", 400.0),
            SampleRow(1000.0, 0, "node-b", 0, "GPU-b0", 400.0),
            SampleRow(1000.0, 0, "node-b", 1, "GPU-b1", 400.0),
        ]

    def test_matching_topology_is_valid(self):
        result = validate_devices(self._expected(), derive_observed_devices(self._rows()))

        assert result.valid is True
        assert result.reason_codes == ()

    def test_missing_device_invalidates(self):
        observed = derive_observed_devices(self._rows()[:3])

        result = validate_devices(self._expected(), observed)

        assert result.valid is False
        assert Reason.EXPECTED_DEVICE_MISSING in result.reason_codes

    def test_unexpected_device_invalidates(self):
        rows = [*self._rows(), SampleRow(1000.0, 0, "node-c", 0, "GPU-c0", 400.0)]

        result = validate_devices(self._expected(), derive_observed_devices(rows))

        assert result.valid is False
        assert Reason.UNEXPECTED_DEVICE in result.reason_codes

    def test_changed_uuid_invalidates(self):
        rows = [*self._rows(), SampleRow(1001.0, 1, "node-a", 0, "GPU-swapped", 400.0)]

        result = validate_devices(self._expected(), derive_observed_devices(rows))

        assert result.valid is False
        assert Reason.GPU_UUID_CHANGED in result.reason_codes

    def test_one_uuid_under_two_device_keys_invalidates(self):
        """An endpoint misroute must not let one physical GPU be counted twice."""
        rows = self._rows()[:3]
        rows.append(SampleRow(1000.0, 0, "node-b", 1, "GPU-a0", 400.0))

        result = validate_devices(self._expected(), derive_observed_devices(rows))

        assert result.valid is False
        assert Reason.GPU_UUID_CHANGED in result.reason_codes

    def test_conflicting_role_invalidates(self):
        expected = build_expected_devices([_process("node-a", [0], "prefill"), _process("node-a", [0], "decode")])
        observed = derive_observed_devices([SampleRow(1000.0, 0, "node-a", 0, "GPU-a0", 400.0)])

        result = validate_devices(expected, observed)

        assert result.valid is False
        assert Reason.CONFLICTING_WORKER_ROLES in result.reason_codes

    def test_empty_expected_set_is_invalid(self):
        result = validate_devices([], [])

        assert result.valid is False
        assert Reason.EXPECTED_DEVICE_MISSING in result.reason_codes


class TestManifest:
    """manifest.json shape."""

    def _manifest(self):
        return PowerManifest(
            job_id="12345",
            run_name="recipe_12345",
            sample_interval_seconds=1.0,
            request_timeout_seconds=2.0,
            required=True,
            started_at_unix=1785168000.0,
            producer_git_commit="abcdef0123456789abcdef0123456789abcdef01",
            dcgm_exporter=DcgmExporterIdentity(
                container_image_resolved="/containers/dcgm-exporter.sqsh",
                container_image_sha256="0" * 64,
                port=9401,
                command="dcgm-exporter --collect-interval=100 --address :9401",
            ),
            expected_devices=build_expected_devices([_process("node-a", [0], "agg")]),
            expected_windows=[ExpectedWindow(benchmark_type="sa-bench", concurrency=4)],
        )

    def test_starting_manifest_shape(self, tmp_path):
        payload = self._manifest().to_dict()

        assert payload["schema_version"] == SCHEMA_VERSION
        assert payload["producer"] == "srt-slurm.dcgm-power"
        assert payload["source_metric"] == "DCGM_FI_DEV_POWER_USAGE"
        assert payload["unit"] == "W"
        assert payload["timestamp_source"] == "head_node_unix_clock"
        assert payload["status"] == "starting"
        assert payload["publication_valid"] is None
        assert payload["stopped_at_unix"] is None
        assert payload["max_scrape_duration_seconds"] is None
        assert payload["scrape_count"] == 0
        assert payload["sample_row_count"] == 0
        assert payload["expected_windows"] == [{"benchmark_type": "sa-bench", "concurrency": 4}]
        assert payload["expected_devices"] == [
            {
                "hostname": "node-a",
                "gpu_index": 0,
                "assignments": [{"worker_role": "agg", "worker_index": 0, "worker_process": 0, "het_group": None}],
            }
        ]

        atomic_write_json(tmp_path / MANIFEST_FILENAME, payload)
        assert json.loads((tmp_path / MANIFEST_FILENAME).read_text()) == payload

    def test_failed_startup_manifest_is_never_publishable(self):
        manifest = self._manifest()
        manifest.mark_terminal(status="failed", stopped_at_unix=1785168010.0, publication_valid=True)

        payload = manifest.to_dict()

        assert payload["status"] == "failed"
        assert payload["publication_valid"] is False

    def test_terminal_manifest_cannot_be_changed_by_a_second_call(self):
        manifest = self._manifest()
        manifest.mark_terminal(status="complete", stopped_at_unix=1785168010.0, publication_valid=True)
        first_terminal_state = (manifest.status, manifest.stopped_at_unix, manifest.publication_valid)

        with pytest.raises(RuntimeError, match="already terminal"):
            manifest.mark_terminal(status="failed", stopped_at_unix=1785168020.0, publication_valid=False)

        assert (manifest.status, manifest.stopped_at_unix, manifest.publication_valid) == first_terminal_state

    def test_terminal_guard_survives_direct_status_reassignment(self):
        manifest = self._manifest()
        manifest.mark_terminal(status="complete", stopped_at_unix=1785168010.0, publication_valid=True)
        first_terminal_evidence = (manifest.stopped_at_unix, manifest.publication_valid)
        manifest.status = "running"

        with pytest.raises(RuntimeError, match="already terminal"):
            manifest.mark_terminal(status="failed", stopped_at_unix=1785168020.0, publication_valid=False)

        assert (manifest.stopped_at_unix, manifest.publication_valid) == first_terminal_evidence

    def test_atomic_write_leaves_no_partial_file(self, tmp_path):
        target = tmp_path / MANIFEST_FILENAME
        atomic_write_json(target, {"a": 1})
        atomic_write_json(target, {"a": 2})

        assert json.loads(target.read_text()) == {"a": 2}
        assert sorted(p.name for p in tmp_path.iterdir()) == [MANIFEST_FILENAME]

    def test_atomic_write_closes_raw_fd_when_fdopen_fails(self, tmp_path, monkeypatch):
        captured = {}

        def fail(fd, *_args, **_kwargs):
            captured["fd"] = fd
            assert _kwargs["closefd"] is False
            raise OSError("too many open files")

        monkeypatch.setattr("srtctl.core.power.contract.os.fdopen", fail)
        try:
            with pytest.raises(OSError, match="too many open files"):
                atomic_write_json(tmp_path / MANIFEST_FILENAME, {"a": 1})

            with pytest.raises(OSError):
                os.fstat(captured["fd"])
            assert list(tmp_path.iterdir()) == []
        finally:
            with suppress(OSError):
                os.close(captured["fd"])

    def test_atomic_write_does_not_close_a_reused_fd_after_wrapper_failure(self, tmp_path, monkeypatch):
        real_fdopen = os.fdopen
        replacement_fd = None

        def fail_after_wrapper_created(fd, *args, **kwargs):
            nonlocal replacement_fd
            handle = real_fdopen(fd, *args, **kwargs)
            handle.close()
            replacement_fd = os.open(tmp_path / "replacement", os.O_CREAT | os.O_WRONLY)
            raise OSError("wrapper setup failed")

        monkeypatch.setattr("srtctl.core.power.contract.os.fdopen", fail_after_wrapper_created)
        try:
            with pytest.raises(OSError, match="wrapper setup failed"):
                atomic_write_json(tmp_path / MANIFEST_FILENAME, {"a": 1})

            assert replacement_fd is not None
            os.fstat(replacement_fd)
        finally:
            if replacement_fd is not None:
                with suppress(OSError):
                    os.close(replacement_fd)


class TestMeasurementWindowArtifacts:
    @staticmethod
    def _write_completed_window(tmp_path, *, start, end, duration):
        windows_dir = tmp_path / WINDOWS_DIRNAME
        windows_dir.mkdir(exist_ok=True)
        payload = {
            "schema_version": SCHEMA_VERSION,
            "benchmark_type": "sa-bench",
            "result_path": "result.json",
            "concurrency": 4,
            "benchmark_start_time_unix": start,
            "benchmark_end_time_unix": end,
            "duration": duration,
            "clock_source": "head_node_unix_clock",
            "status": "completed",
            "reason": None,
        }
        (windows_dir / "result.json").write_text(json.dumps(payload))
        (tmp_path / "result.json").write_text(json.dumps(payload))

    @staticmethod
    def _validate(tmp_path, *, expected_device_keys=None, observed_devices=()):
        errors = []
        rows = validate_expected_windows(
            power_dir=tmp_path,
            result_root=tmp_path,
            expected_windows=[ExpectedWindow("sa-bench", 4)],
            expected_device_keys=set() if expected_device_keys is None else expected_device_keys,
            observed_devices=observed_devices,
            artifact_errors=errors,
        )
        return rows[0], errors

    def test_unreadable_windows_directory_is_reported(self, tmp_path, monkeypatch):
        windows_dir = tmp_path / WINDOWS_DIRNAME
        windows_dir.mkdir()
        original_iterdir = Path.iterdir

        def fail_for_windows(path):
            if path == windows_dir:
                raise PermissionError("permission denied")
            return original_iterdir(path)

        monkeypatch.setattr(Path, "iterdir", fail_for_windows)
        row, errors = self._validate(tmp_path)

        assert row.reason_codes == (Reason.MEASUREMENT_WINDOW_MISSING,)
        assert [(error.path, error.reason_codes) for error in errors] == [
            (WINDOWS_DIRNAME, (Reason.MEASUREMENT_WINDOW_ARTIFACT_PATH_INVALID,))
        ]

    def test_unreadable_windows_directory_does_not_break_interruption_cleanup(self, tmp_path, monkeypatch):
        windows_dir = tmp_path / WINDOWS_DIRNAME
        windows_dir.mkdir()
        original_glob = Path.glob

        def fail_for_windows(path, pattern):
            if path == windows_dir:
                raise PermissionError("permission denied")
            return original_glob(path, pattern)

        monkeypatch.setattr(Path, "glob", fail_for_windows)

        assert convert_running_windows(windows_dir, reason="interrupted") == 0

    def test_interruption_cleanup_rejects_windows_directory_symlink_escape(self, tmp_path):
        power_dir = tmp_path / "power"
        power_dir.mkdir()
        external_dir = tmp_path / "external"
        external_dir.mkdir()
        victim = external_dir / "victim.json"
        victim.write_text(json.dumps({"status": "running", "sentinel": "unchanged"}))
        windows_dir = power_dir / WINDOWS_DIRNAME
        windows_dir.symlink_to(external_dir, target_is_directory=True)

        original = victim.read_text()
        converted = convert_running_windows(windows_dir, reason="interrupted")

        assert converted == 0
        assert victim.read_text() == original

    def test_interruption_cleanup_converts_running_window(self, tmp_path):
        windows_dir = tmp_path / WINDOWS_DIRNAME
        windows_dir.mkdir()
        window_path = windows_dir / "result.json"
        window_path.write_text(
            json.dumps(
                {
                    "status": "running",
                    "benchmark_end_time_unix": 1002.0,
                    "duration": 2.0,
                    "reason": None,
                }
            )
        )

        converted = convert_running_windows(windows_dir, reason="benchmark child terminated")

        payload = json.loads(window_path.read_text())
        assert converted == 1
        assert payload["status"] == "interrupted"
        assert payload["benchmark_end_time_unix"] is None
        assert payload["duration"] is None
        assert payload["reason"] == "benchmark child terminated"

    def test_unresolvable_windows_directory_is_reported(self, tmp_path, monkeypatch):
        windows_dir = tmp_path / WINDOWS_DIRNAME
        windows_dir.mkdir()
        original_resolve = Path.resolve

        def fail_for_windows(path, *args, **kwargs):
            if path == windows_dir:
                raise RuntimeError("symlink loop")
            return original_resolve(path, *args, **kwargs)

        monkeypatch.setattr(Path, "resolve", fail_for_windows)
        row, errors = self._validate(tmp_path)

        assert row.reason_codes == (Reason.MEASUREMENT_WINDOW_MISSING,)
        assert [(error.path, error.reason_codes) for error in errors] == [
            (WINDOWS_DIRNAME, (Reason.MEASUREMENT_WINDOW_ARTIFACT_PATH_INVALID,))
        ]

    def test_non_json_window_file_is_rejected(self, tmp_path):
        self._write_completed_window(tmp_path, start=1000.0, end=1001.0, duration=1.0)
        window_path = tmp_path / WINDOWS_DIRNAME / "result.json"
        window_path.rename(window_path.with_suffix(".txt"))

        row, errors = self._validate(tmp_path)

        assert row.reason_codes == (Reason.MEASUREMENT_WINDOW_MISSING,)
        assert [(error.path, error.reason_codes) for error in errors] == [
            (f"{WINDOWS_DIRNAME}/result.txt", (Reason.MEASUREMENT_WINDOW_ARTIFACT_PATH_INVALID,))
        ]

    @pytest.mark.parametrize(
        ("start", "end"),
        [
            (1000.1, 1000.0),
            (1000.0, 1000.0),
        ],
    )
    def test_non_positive_wall_clock_interval_is_rejected(self, tmp_path, start, end):
        self._write_completed_window(tmp_path, start=start, end=end, duration=0.1)
        row, _ = self._validate(tmp_path)

        assert row.power_coverage_valid is False
        assert row.reason_codes == (Reason.MEASUREMENT_WINDOW_CLOCK_MISMATCH,)

    @pytest.mark.parametrize(
        ("field", "boolean_value"),
        [
            ("benchmark_start_time_unix", False),
            ("benchmark_end_time_unix", True),
            ("duration", True),
        ],
    )
    def test_result_timing_fields_reject_boolean_numbers(self, tmp_path, field, boolean_value):
        self._write_completed_window(tmp_path, start=0.0, end=1.0, duration=1.0)
        result_path = tmp_path / "result.json"
        result = json.loads(result_path.read_text())
        result[field] = boolean_value
        result_path.write_text(json.dumps(result))

        row, _ = self._validate(tmp_path)

        assert row.power_coverage_valid is False
        assert row.reason_codes == (Reason.MEASUREMENT_WINDOW_RESULT_MISMATCH,)

    def test_computed_device_gaps_are_retained_when_coverage_is_invalid(self, tmp_path):
        self._write_completed_window(tmp_path, start=1000.0, end=1004.0, duration=4.0)
        observed = derive_observed_devices(
            [
                SampleRow(999.0, 0, "node-a", 0, "GPU-a", 400.0),
                SampleRow(1005.0, 1, "node-a", 0, "GPU-a", 400.0),
            ]
        )

        row, _ = self._validate(
            tmp_path,
            expected_device_keys={("node-a", 0), ("node-b", 0)},
            observed_devices=observed,
        )

        assert row.power_coverage_valid is False
        assert row.reason_codes == (
            Reason.SAMPLE_GAP_EXCEEDED,
            Reason.MEASUREMENT_WINDOW_NOT_BRACKETED,
        )
        assert row.per_device_max_sample_gap_seconds == {"node-a/GPU-a": 6.0}

    def test_window_coverage_does_not_depend_on_sample_time_order(self, tmp_path):
        self._write_completed_window(tmp_path, start=1000.0, end=1002.0, duration=2.0)
        observed = [
            ObservedDevice(
                hostname="node-a",
                gpu_index=0,
                gpu_uuids=("GPU-a",),
                first_sample_time_unix=999.0,
                last_sample_time_unix=1003.0,
                sample_times=(1000.0, 1003.0, 999.0, 1002.0),
            )
        ]

        row, _ = self._validate(
            tmp_path,
            expected_device_keys={("node-a", 0)},
            observed_devices=observed,
        )

        assert row.power_coverage_valid is True
        assert row.reason_codes == ()
        assert row.per_device_max_sample_gap_seconds == {"node-a/GPU-a": 2.0}

    def test_three_duplicate_windows_are_each_recorded_once(self, tmp_path):
        windows_dir = tmp_path / WINDOWS_DIRNAME
        windows_dir.mkdir()
        for index in range(3):
            stem = f"result-{index}"
            (windows_dir / f"{stem}.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "benchmark_type": "sa-bench",
                        "result_path": f"{stem}.json",
                        "concurrency": 4,
                        "benchmark_start_time_unix": 1000.0,
                        "benchmark_end_time_unix": 1020.0,
                        "duration": 20.0,
                        "clock_source": "head_node_unix_clock",
                        "status": "completed",
                        "reason": None,
                    }
                )
            )
        errors = []

        rows = validate_expected_windows(
            power_dir=tmp_path,
            result_root=tmp_path,
            expected_windows=[ExpectedWindow("sa-bench", 4)],
            expected_device_keys=set(),
            observed_devices=[],
            artifact_errors=errors,
        )

        assert rows[0].reason_codes == (Reason.MEASUREMENT_WINDOW_DUPLICATE,)
        assert [error.path for error in errors] == [
            f"{WINDOWS_DIRNAME}/result-0.json",
            f"{WINDOWS_DIRNAME}/result-1.json",
            f"{WINDOWS_DIRNAME}/result-2.json",
        ]
        assert all(error.reason_codes == (Reason.MEASUREMENT_WINDOW_DUPLICATE,) for error in errors)


class TestStorageSubdirSafety:
    @pytest.mark.parametrize("value", ["power", "telemetry/power", "a/b/c"])
    def test_safe_paths_accepted(self, value):
        assert is_safe_relative_subpath(value) is True

    @pytest.mark.parametrize("value", ["", "/power", "../power", "power/../..", "a/../../b", "~/power"])
    def test_unsafe_paths_rejected(self, value):
        assert is_safe_relative_subpath(value) is False
