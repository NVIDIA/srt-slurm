# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline re-validation of a retained power artifact package."""

import hashlib
import json
from pathlib import Path

import pytest

import srtctl.core.power.validate_artifacts as power_validator
from srtctl.cli.validate_power_artifacts import main
from srtctl.core.power.contract import (
    ALL_REASON_CODES,
    MANIFEST_FILENAME,
    MAX_SAMPLE_GAP_SECONDS,
    SAMPLES_FILENAME,
    WINDOWS_DIRNAME,
    Reason,
    atomic_write_json,
)
from srtctl.core.power.manifest import (
    STATUS_COMPLETE,
    DcgmExporterIdentity,
    ExpectedWindow,
    PowerManifest,
)
from srtctl.core.power.samples import SampleRow, SampleWriter, derive_observed_devices
from srtctl.core.power.topology import build_expected_devices
from srtctl.core.power.validate_artifacts import validate_power_artifacts
from srtctl.core.power.windows import validate_expected_windows
from srtctl.core.topology import Process

START = 1785168100.0
END = 1785168120.0
RESULT_SUBDIR = "sa-bench_isl_8192_osl_1024"
RESULT_STEM = "results_concurrency_4_gpus_8_ctx_4_gen_4"


def _processes(*, decode_het_group=1):
    return [
        Process(
            node="node-a",
            gpu_indices=frozenset(range(4)),
            sys_port=8081,
            http_port=30000,
            endpoint_mode="prefill",
            endpoint_index=0,
            het_group=0,
        ),
        Process(
            node="node-b",
            gpu_indices=frozenset(range(4)),
            sys_port=8082,
            http_port=30000,
            endpoint_mode="decode",
            endpoint_index=0,
            het_group=decode_het_group,
        ),
    ]


@pytest.fixture
def package(tmp_path):
    """A retained, publishable 1P1D artifact package."""

    def build(*, processes=None, rows=None, publication_valid=True):
        log_dir = tmp_path / "logs"
        power_dir = log_dir / "power"
        (power_dir / WINDOWS_DIRNAME).mkdir(parents=True)
        (log_dir / RESULT_SUBDIR).mkdir(parents=True)

        expected = build_expected_devices(processes or _processes())

        written = rows if rows is not None else _rows(expected)
        writer = SampleWriter(power_dir / SAMPLES_FILENAME)
        writer.append(written)
        writer.close()

        (log_dir / RESULT_SUBDIR / f"{RESULT_STEM}.json").write_text(
            json.dumps(
                {
                    "duration": END - START,
                    "benchmark_start_time_unix": START,
                    "benchmark_end_time_unix": END,
                    "completed": 40,
                }
            )
        )
        atomic_write_json(
            power_dir / WINDOWS_DIRNAME / f"{RESULT_STEM}.json",
            {
                "schema_version": 1,
                "benchmark_type": "sa-bench",
                "result_path": f"{RESULT_SUBDIR}/{RESULT_STEM}.json",
                "concurrency": 4,
                "benchmark_start_time_unix": START,
                "benchmark_end_time_unix": END,
                "duration": END - START,
                "clock_source": "head_node_unix_clock",
                "status": "completed",
                "reason": None,
            },
        )

        manifest = PowerManifest(
            job_id="12345",
            run_name="canary_12345",
            sample_interval_seconds=1.0,
            request_timeout_seconds=2.0,
            required=True,
            started_at_unix=START - 30,
            producer_git_commit="a" * 40,
            dcgm_exporter=DcgmExporterIdentity(
                container_image_resolved="/containers/dcgm-exporter.sqsh",
                container_image_sha256="0" * 64,
                port=9401,
                command="dcgm-exporter --collect-interval=100 --address :9401",
            ),
            expected_devices=expected,
            expected_windows=[ExpectedWindow("sa-bench", 4)],
        )
        observed = derive_observed_devices(written)
        manifest.observed_devices = observed
        manifest.sample_row_count = len(written)
        manifest.samples_sha256 = hashlib.sha256((power_dir / SAMPLES_FILENAME).read_bytes()).hexdigest()
        manifest.scrape_count = (max(row.scrape_seq for row in written) + 1) if written else 0
        # The audit the producer stores is the one the validator recomputes, so it can never drift.
        manifest.window_validations = validate_expected_windows(
            power_dir=power_dir,
            result_root=log_dir,
            expected_windows=manifest.expected_windows,
            expected_device_keys={device.key for device in expected},
            observed_devices=observed,
            artifact_errors=manifest.artifact_errors,
        )
        manifest.mark_terminal(status=STATUS_COMPLETE, stopped_at_unix=END + 5, publication_valid=publication_valid)
        atomic_write_json(power_dir / MANIFEST_FILENAME, manifest.to_dict())
        return log_dir, power_dir

    return build


def _rows(expected, *, step=1.0, skip=()):
    rows = []
    seq = 0
    timestamp = START - 2.0
    while timestamp <= END + 2.0:
        for device in expected:
            if device.key in skip:
                continue
            rows.append(
                SampleRow(
                    timestamp, seq, device.hostname, device.gpu_index, f"GPU-{device.hostname}{device.gpu_index}", 400.0
                )
            )
        seq += 1
        timestamp += step
    return rows


def _validate(power_dir, log_dir, **kwargs):
    return validate_power_artifacts(power_dir=power_dir, result_root=log_dir, **kwargs)


class TestRetainedPackage:
    def test_core_module_documents_the_functional_console_entrypoint(self):
        assert "srtctl-validate-power" in power_validator.__doc__
        assert "python -m srtctl.core.power.validate_artifacts" not in power_validator.__doc__

    def test_valid_package_passes_every_canary_assertion(self, package):
        log_dir, power_dir = package()

        report = _validate(
            power_dir,
            log_dir,
            expected_roles={"prefill": 4, "decode": 4},
            require_distinct_het_groups=True,
        )

        assert report.ok is True
        assert report.failures == ()
        assert report.summary["expected_devices"] == 8
        assert report.summary["observed_devices"] == 8
        assert report.summary["stable_uuids"] == 8
        assert report.summary["producer_git_commit"] == "a" * 40
        assert report.summary["max_sample_gap_seconds"] == pytest.approx(1.0)

    def test_cli_exits_zero_on_a_valid_package(self, package, capsys):
        log_dir, power_dir = package()

        code = main(["--power-dir", str(power_dir), "--result-root", str(log_dir)])

        assert code == 0
        assert "publication_valid" in capsys.readouterr().out

    def test_render_separates_canary_failure_from_publication_validity(self, package):
        log_dir, power_dir = package()

        report = _validate(power_dir, log_dir, expected_roles={"prefill": 8, "decode": 4})

        assert report.ok is False
        assert report.publication_valid is True
        assert report.render().splitlines()[:2] == ["validation_ok: False", "publication_valid: True"]


class TestIndependenceFromTheManifestBooleans:
    def test_a_manifest_claiming_validity_cannot_rescue_missing_samples(self, package, capsys):
        expected = build_expected_devices(_processes())
        log_dir, power_dir = package(rows=_rows(expected, skip={("node-b", 0)}), publication_valid=True)

        assert json.loads((power_dir / MANIFEST_FILENAME).read_text())["publication_valid"] is True

        report = _validate(power_dir, log_dir)

        assert report.ok is False
        assert any("expected_device_missing" in failure for failure in report.failures)
        assert any("publication_valid" in failure and "recomputed" in failure for failure in report.failures)
        assert main(["--power-dir", str(power_dir), "--result-root", str(log_dir)]) == 1
        assert "expected_device_missing" in capsys.readouterr().out

    def test_tampered_samples_are_rejected(self, package):
        log_dir, power_dir = package()
        samples = power_dir / SAMPLES_FILENAME
        samples.write_text(samples.read_text().replace("400.0", "not-a-number", 1))

        report = _validate(power_dir, log_dir)

        assert report.ok is False
        assert any("samples_csv_malformed" in failure for failure in report.failures)

    def test_changed_power_values_are_rejected(self, package):
        log_dir, power_dir = package()
        samples = power_dir / SAMPLES_FILENAME
        samples.write_text(samples.read_text().replace("400.0", "1.0"))

        report = _validate(power_dir, log_dir)

        assert report.ok is False
        assert any("samples_sha256 mismatch" in failure for failure in report.failures)

    @pytest.mark.parametrize("digest", [None, "", "A" * 64, "0" * 63])
    def test_missing_or_malformed_samples_digest_is_rejected(self, package, digest):
        log_dir, power_dir = package()
        manifest_path = power_dir / MANIFEST_FILENAME
        manifest = json.loads(manifest_path.read_text())
        if digest is None:
            manifest.pop("samples_sha256", None)
        else:
            manifest["samples_sha256"] = digest
        atomic_write_json(manifest_path, manifest)

        report = _validate(power_dir, log_dir)

        assert report.ok is False
        assert any("samples_sha256 is not a lowercase SHA-256 digest" in failure for failure in report.failures)

    def test_gap_beyond_the_threshold_is_rejected(self, package):
        expected = build_expected_devices(_processes())
        log_dir, power_dir = package(rows=_rows(expected, step=4.0))

        report = _validate(power_dir, log_dir)

        assert report.ok is False
        assert any("sample_gap_exceeded" in failure for failure in report.failures)

    def test_reversed_short_window_is_rejected_end_to_end(self, package):
        log_dir, power_dir = package()
        window_path = power_dir / WINDOWS_DIRNAME / f"{RESULT_STEM}.json"
        result_path = log_dir / RESULT_SUBDIR / f"{RESULT_STEM}.json"
        for path in (window_path, result_path):
            payload = json.loads(path.read_text())
            payload["benchmark_start_time_unix"] = START
            payload["benchmark_end_time_unix"] = START - 0.1
            payload["duration"] = 0.1
            atomic_write_json(path, payload)

        report = _validate(power_dir, log_dir)

        assert report.ok is False
        assert any(Reason.MEASUREMENT_WINDOW_CLOCK_MISMATCH in failure for failure in report.failures)


class TestTopologyAssertions:
    def test_zero_count_asserts_that_a_known_role_is_absent(self, package):
        log_dir, power_dir = package()

        report = _validate(
            power_dir,
            log_dir,
            expected_roles={"prefill": 4, "decode": 4, "agg": 0},
        )

        assert report.ok is True
        assert report.failures == ()

    def test_zero_count_rejects_a_role_that_is_present(self, package):
        processes = [
            *_processes(),
            Process(
                node="node-c",
                gpu_indices=frozenset(range(4)),
                sys_port=8083,
                http_port=30000,
                endpoint_mode="agg",
                endpoint_index=0,
                het_group=2,
            ),
        ]
        log_dir, power_dir = package(processes=processes)

        report = _validate(
            power_dir,
            log_dir,
            expected_roles={"prefill": 4, "decode": 4, "agg": 0},
        )

        assert report.ok is False
        assert any("expected 0 agg GPUs" in failure for failure in report.failures)

    def test_unknown_zero_count_role_is_rejected_by_the_direct_api(self, package):
        log_dir, power_dir = package()

        report = _validate(power_dir, log_dir, expected_roles={"bogus": 0})

        assert report.ok is False
        assert any("unknown expected roles" in failure for failure in report.failures)

    def test_shared_het_group_fails_the_distinct_group_requirement(self, package):
        log_dir, power_dir = package(processes=_processes(decode_het_group=0))

        report = _validate(power_dir, log_dir, require_distinct_het_groups=True)

        assert report.ok is False
        assert any("het" in failure for failure in report.failures)

    @pytest.mark.parametrize(
        ("extra_agg_role", "expected_roles", "expected_failure"),
        [
            (True, {"prefill": 4, "decode": 4}, "expected roles"),
            (False, {"prefill": 8}, "expected 8 prefill GPUs, found 4"),
        ],
        ids=["extra-role", "wrong-count"],
    )
    def test_the_expected_role_set_must_match_exactly(self, package, extra_agg_role, expected_roles, expected_failure):
        """4P+4D must not pass beside an extra agg role, nor under a wrong count."""
        processes = _processes()
        if extra_agg_role:
            processes.append(
                Process(
                    node="node-c",
                    gpu_indices=frozenset(range(4)),
                    sys_port=8083,
                    http_port=30000,
                    endpoint_mode="agg",
                    endpoint_index=0,
                    het_group=2,
                )
            )
        log_dir, power_dir = package(processes=processes)

        report = _validate(power_dir, log_dir, expected_roles=expected_roles)

        assert report.ok is False
        assert any(expected_failure in failure for failure in report.failures)

    def test_a_role_spanning_two_het_groups_is_rejected(self, package):
        """Decode split across groups 1 and 2 must not pass the distinct check."""
        processes = [
            _processes()[0],
            Process(
                node="node-b",
                gpu_indices=frozenset(range(2)),
                sys_port=8082,
                http_port=30000,
                endpoint_mode="decode",
                endpoint_index=0,
                het_group=1,
            ),
            Process(
                node="node-c",
                gpu_indices=frozenset(range(2)),
                sys_port=8083,
                http_port=30000,
                endpoint_mode="decode",
                endpoint_index=1,
                het_group=2,
            ),
        ]
        log_dir, power_dir = package(processes=processes)

        report = _validate(power_dir, log_dir, require_distinct_het_groups=True)

        assert report.ok is False
        assert any("spans het groups" in failure for failure in report.failures)

    def test_a_null_het_group_is_not_a_distinct_group(self, package):
        """A non-heterogeneous job can never satisfy the canary's group split."""
        processes = [
            Process(
                node="node-a",
                gpu_indices=frozenset(range(4)),
                sys_port=8081,
                http_port=30000,
                endpoint_mode="prefill",
                endpoint_index=0,
                het_group=None,
            ),
            Process(
                node="node-b",
                gpu_indices=frozenset(range(4)),
                sys_port=8082,
                http_port=30000,
                endpoint_mode="decode",
                endpoint_index=0,
                het_group=1,
            ),
        ]
        log_dir, power_dir = package(processes=processes)

        report = _validate(power_dir, log_dir, require_distinct_het_groups=True)

        assert report.ok is False
        assert any("expected a non-negative integer" in failure for failure in report.failures)

    def test_missing_manifest_is_reported(self, tmp_path):
        report = validate_power_artifacts(power_dir=tmp_path, result_root=tmp_path)

        assert report.ok is False
        assert any("manifest" in failure for failure in report.failures)


class TestWireContract:
    """Internally invalid manifest metadata must be rejected on its own terms."""

    @pytest.mark.parametrize(
        ("key", "value"),
        [
            ("schema_version", 999),
            ("schema_version", True),
            ("producer", "someone-elses-producer"),
            ("source_metric", "DCGM_FI_DEV_GPU_TEMP"),
            ("unit", "mW"),
            ("power_scope", "whole_node"),
            ("timestamp_source", "worker_node_clock"),
            ("producer_version", ""),
            ("producer_version", {}),
            ("producer_git_commit", "abc123"),
            ("job_id", ""),
            ("run_name", []),
            ("started_at_unix", None),
            ("stopped_at_unix", None),
            ("sample_interval_seconds", 0),
            ("sample_interval_seconds", MAX_SAMPLE_GAP_SECONDS + 0.1),
            ("request_timeout_seconds", -1.0),
            ("max_scrape_duration_seconds", -0.1),
            ("max_scrape_duration_seconds", float("nan")),
            ("max_scrape_duration_seconds", {}),
            ("scrape_count", -1),
            ("sample_row_count", "many"),
            ("publication_valid", None),
            ("publication_valid", "true"),
            ("publication_valid", 1),
            ("reason_codes", "not-a-list"),
            ("reason_codes", ["not_a_v1_reason"]),
            ("reason_codes", ["endpoint_timeout", "endpoint_timeout"]),
            ("dcgm_exporter", None),
        ],
    )
    def test_invalid_wire_metadata_is_rejected(self, package, key, value):
        log_dir, power_dir = package()
        manifest = json.loads((power_dir / MANIFEST_FILENAME).read_text())
        manifest[key] = value
        atomic_write_json(power_dir / MANIFEST_FILENAME, manifest)

        report = _validate(power_dir, log_dir)

        assert report.ok is False
        assert any(key in failure for failure in report.failures)

    def test_a_start_after_stop_is_rejected(self, package):
        log_dir, power_dir = package()
        manifest = json.loads((power_dir / MANIFEST_FILENAME).read_text())
        manifest["started_at_unix"] = manifest["stopped_at_unix"] + 60.0
        atomic_write_json(power_dir / MANIFEST_FILENAME, manifest)

        report = _validate(power_dir, log_dir)

        assert report.ok is False
        assert any("precedes" in failure for failure in report.failures)

    @pytest.mark.parametrize(
        ("field", "value", "expected_failure"),
        [
            ("started_at_unix", START - 1.0, "samples.csv starts"),
            ("stopped_at_unix", END + 1.0, "samples.csv ends"),
        ],
    )
    def test_manifest_lifecycle_must_contain_all_samples(self, package, field, value, expected_failure):
        log_dir, power_dir = package()
        manifest = json.loads((power_dir / MANIFEST_FILENAME).read_text())
        manifest[field] = value
        atomic_write_json(power_dir / MANIFEST_FILENAME, manifest)

        report = _validate(power_dir, log_dir)

        assert report.ok is False
        assert any(expected_failure in failure for failure in report.failures)

    def test_samples_may_equal_the_manifest_lifecycle_boundaries(self, package):
        log_dir, power_dir = package()
        manifest = json.loads((power_dir / MANIFEST_FILENAME).read_text())
        manifest["started_at_unix"] = START - 2.0
        manifest["stopped_at_unix"] = END + 2.0
        atomic_write_json(power_dir / MANIFEST_FILENAME, manifest)

        assert _validate(power_dir, log_dir).ok is True

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("container_image_sha256", 123),
            ("container_image_sha256", "x"),
            ("container_image_sha256", "A" * 64),
            ("port", None),
            ("port", 0),
            ("port", True),
            ("command", ""),
        ],
    )
    def test_an_incomplete_exporter_identity_is_rejected(self, package, field, value):
        log_dir, power_dir = package()
        manifest = json.loads((power_dir / MANIFEST_FILENAME).read_text())
        manifest["dcgm_exporter"][field] = value
        atomic_write_json(power_dir / MANIFEST_FILENAME, manifest)

        report = _validate(power_dir, log_dir)

        assert report.ok is False
        assert any(f"dcgm_exporter.{field}" in failure for failure in report.failures)

    def test_an_empty_exporter_image_is_rejected(self, package):
        log_dir, power_dir = package()
        manifest = json.loads((power_dir / MANIFEST_FILENAME).read_text())
        manifest["dcgm_exporter"]["container_image_resolved"] = ""
        atomic_write_json(power_dir / MANIFEST_FILENAME, manifest)

        report = _validate(power_dir, log_dir)

        assert report.ok is False
        assert any("container_image_resolved" in failure for failure in report.failures)

    def test_compatible_older_producer_identity_is_accepted(self, package):
        log_dir, power_dir = package()
        manifest = json.loads((power_dir / MANIFEST_FILENAME).read_text())
        manifest["producer_version"] = "0.1.0-older"
        manifest["producer_git_commit"] = None
        manifest["dcgm_exporter"]["container_image_sha256"] = None
        atomic_write_json(power_dir / MANIFEST_FILENAME, manifest)

        assert _validate(power_dir, log_dir).ok is True

    def test_a_false_stored_verdict_vetoes_an_otherwise_valid_package(self, package):
        log_dir, power_dir = package(publication_valid=False)

        assert json.loads((power_dir / MANIFEST_FILENAME).read_text())["publication_valid"] is False
        report = _validate(power_dir, log_dir)

        assert report.ok is False
        assert any("publication_valid" in failure and "recomputed" in failure for failure in report.failures)


class TestEvidenceReconciliation:
    """Only the stored verdict is ignored; other derived claims must hold."""

    def _damaged(self, package, mutate):
        log_dir, power_dir = package()
        manifest = json.loads((power_dir / MANIFEST_FILENAME).read_text())
        mutate(manifest)
        atomic_write_json(power_dir / MANIFEST_FILENAME, manifest)
        return _validate(power_dir, log_dir)

    def test_sample_row_count_must_match_the_csv(self, package):
        report = self._damaged(package, lambda m: m.update(sample_row_count=999))

        assert report.ok is False
        assert any("sample_row_count" in failure for failure in report.failures)

    def test_empty_observed_devices_is_rejected(self, package):
        report = self._damaged(package, lambda m: m.update(observed_devices=[]))

        assert report.ok is False
        assert any("observed_devices" in failure for failure in report.failures)

    def test_empty_window_validations_is_rejected(self, package):
        report = self._damaged(package, lambda m: m.update(window_validations=[]))

        assert report.ok is False
        assert any("window_validations" in failure for failure in report.failures)

    @pytest.mark.parametrize("key", ["expected_windows", "expected_devices"])
    def test_duplicate_expected_keys_are_rejected(self, package, key):
        report = self._damaged(package, lambda m: m.update({key: [*m[key], m[key][0]]}))

        assert report.ok is False
        assert any(f"{key} contains duplicate keys" in failure for failure in report.failures)

    def test_forged_observed_device_uuids_are_rejected(self, package):
        def forge(m):
            m["observed_devices"][0]["gpu_uuids"] = ["GPU-not-what-was-sampled"]

        report = self._damaged(package, forge)

        assert report.ok is False
        assert any("observed_devices does not match" in failure for failure in report.failures)

    def test_boolean_observed_device_index_is_rejected(self, package):
        def forge(m):
            m["observed_devices"][0]["gpu_index"] = False

        report = self._damaged(package, forge)

        assert report.ok is False
        assert any("observed_devices does not match" in failure for failure in report.failures)

    def test_forged_window_validation_content_is_rejected(self, package):
        def forge(m):
            m["window_validations"][0]["per_device_max_sample_gap_seconds"] = {"node-a/GPU-fake": 0.1}

        report = self._damaged(package, forge)

        assert report.ok is False
        assert any("window_validations does not match" in failure for failure in report.failures)

    def test_integer_window_validity_is_rejected(self, package):
        def forge(m):
            m["window_validations"][0]["power_coverage_valid"] = 1

        report = self._damaged(package, forge)

        assert report.ok is False
        assert any("window_validations does not match" in failure for failure in report.failures)

    def test_fabricated_artifact_errors_are_rejected(self, package):
        report = self._damaged(
            package,
            lambda m: m.update(artifact_errors=[{"path": "windows/ghost.json", "reason_codes": ["nonsense"]}]),
        )

        assert report.ok is False
        assert any("artifact_errors does not match" in failure for failure in report.failures)

    def test_a_stale_disk_derived_reason_is_rejected(self, package):
        report = self._damaged(package, lambda m: m.update(reason_codes=[Reason.MEASUREMENT_WINDOW_MISSING]))

        assert report.ok is False
        assert any("disk-derived reason_codes mismatch" in failure for failure in report.failures)

    def test_a_missing_disk_derived_reason_is_rejected(self, package):
        expected = build_expected_devices(_processes())
        log_dir, power_dir = package(rows=_rows(expected, step=4.0), publication_valid=False)

        report = _validate(power_dir, log_dir)

        assert report.ok is False
        assert any("disk-derived reason_codes mismatch" in failure for failure in report.failures)

    def test_a_runtime_only_reason_does_not_need_disk_reproduction(self, package):
        report = self._damaged(package, lambda m: m.update(reason_codes=[Reason.ENDPOINT_TIMEOUT]))

        assert report.ok is True
        assert report.failures == ()

    def test_every_v1_reason_has_exactly_one_reconciliation_class(self):
        assert power_validator._DISK_DERIVED_REASON_CODES.isdisjoint(power_validator._RUNTIME_ONLY_REASON_CODES)
        assert (
            power_validator._DISK_DERIVED_REASON_CODES | power_validator._RUNTIME_ONLY_REASON_CODES
        ) == ALL_REASON_CODES

    def test_zero_scrape_count_with_samples_present_is_rejected(self, package):
        report = self._damaged(package, lambda m: m.update(scrape_count=0))

        assert report.ok is False
        assert any("scrape_count" in failure for failure in report.failures)

    def test_trailing_empty_cycles_may_exceed_the_sequence_high_water_mark(self, package):
        """A cycle that wrote no rows legitimately raises scrape_count."""
        log_dir, power_dir = package()
        manifest = json.loads((power_dir / MANIFEST_FILENAME).read_text())
        manifest["scrape_count"] += 3
        atomic_write_json(power_dir / MANIFEST_FILENAME, manifest)

        assert _validate(power_dir, log_dir).ok is True

    def test_non_string_reason_codes_do_not_crash(self, package):
        """A non-hashable entry must not raise out of the set operation."""
        report = self._damaged(package, lambda m: m.update(reason_codes=[{}]))

        assert report.ok is False
        assert any("reason_codes is not a list of strings" in failure for failure in report.failures)

    @pytest.mark.parametrize(
        "reason",
        ["exporter_startup_timeout", "exporter_launch_failed", "endpoint_resolution_failed"],
    )
    def test_required_mode_rejects_complete_with_a_startup_failure_reason(self, package, reason):
        report = self._damaged(package, lambda m: m.update(required=True, reason_codes=[reason]))

        assert report.ok is False
        assert any("lifecycle-failure reasons" in failure for failure in report.failures)

    def test_best_effort_accepts_a_recovered_exporter_startup_timeout(self, package):
        """A slow exporter can recover and produce publishable best-effort data.

        Mirrors PowerTelemetrySession._terminal_status: only required mode turns
        a startup failure into a lifecycle failure.
        """
        report = self._damaged(
            package,
            lambda m: m.update(required=False, reason_codes=[Reason.EXPORTER_STARTUP_TIMEOUT]),
        )

        assert report.ok is True
        assert report.failures == ()

    @pytest.mark.parametrize(
        "reason",
        [Reason.EXPORTER_LAUNCH_FAILED, Reason.ENDPOINT_RESOLUTION_FAILED],
    )
    def test_best_effort_rejects_an_unrecoverable_startup_failure(self, package, reason):
        """These paths cannot produce complete evidence without manifest forgery."""
        report = self._damaged(package, lambda m: m.update(required=False, reason_codes=[reason]))

        assert report.ok is False
        assert any("publication_valid" in failure and "recomputed False" in failure for failure in report.failures)

    @pytest.mark.parametrize("required", [True, False])
    @pytest.mark.parametrize(
        "reason",
        ["collector_exception", "collector_join_timeout", "exporter_exited", "benchmark_child_reap_timeout"],
    )
    def test_fatal_reasons_are_rejected_in_both_modes(self, package, reason, required):
        report = self._damaged(package, lambda m: m.update(required=required, reason_codes=[reason]))

        assert report.ok is False
        assert any("lifecycle-failure reasons" in failure for failure in report.failures)

    @pytest.mark.parametrize("value", [None, "true", 1])
    def test_a_non_boolean_required_flag_is_rejected(self, package, value):
        """The validator cannot reproduce mode-dependent semantics without it."""
        report = self._damaged(package, lambda m: m.update(required=value))

        assert report.ok is False
        assert any("required is" in failure for failure in report.failures)


class TestDamagedManifest:
    """A damaged manifest must report cleanly, never raise out of the CLI."""

    @pytest.mark.parametrize(
        "damage",
        [
            pytest.param(lambda m: m["expected_devices"][0].update(gpu_index=None), id="null-gpu-index"),
            pytest.param(lambda m: m["expected_devices"][0].pop("hostname"), id="missing-hostname"),
            pytest.param(lambda m: m.update(expected_devices="not-a-list"), id="device-list-is-a-string"),
            pytest.param(lambda m: m["expected_windows"][0].update(concurrency="four"), id="non-numeric-concurrency"),
            pytest.param(lambda m: m.update(expected_windows=[None]), id="null-window-entry"),
            pytest.param(
                lambda m: m["expected_devices"][0]["assignments"][0].update(worker_index="first"),
                id="non-numeric-worker-index",
            ),
            pytest.param(lambda m: m["expected_devices"][0]["assignments"][0].update(worker_role="bogus"), id="role"),
            pytest.param(lambda m: m["expected_windows"][0].update(concurrency=True), id="bool-concurrency"),
            pytest.param(lambda m: m["expected_windows"][0].update(concurrency=0), id="zero-concurrency"),
            pytest.param(lambda m: m["expected_devices"][0].update(gpu_index=-1), id="negative-index"),
            pytest.param(
                lambda m: m["expected_devices"][0]["assignments"][0].update(het_group=True), id="bool-het-group"
            ),
            pytest.param(lambda m: m["expected_devices"][0].update(assignments=[]), id="empty-assignments"),
            pytest.param(lambda m: m["expected_devices"][0].update(hostname=""), id="empty-hostname"),
        ],
    )
    def test_damage_becomes_a_clean_failure(self, package, damage):
        log_dir, power_dir = package()
        manifest = json.loads((power_dir / MANIFEST_FILENAME).read_text())
        damage(manifest)
        atomic_write_json(power_dir / MANIFEST_FILENAME, manifest)

        report = _validate(power_dir, log_dir)

        assert report.ok is False
        assert any("manifest malformed" in failure for failure in report.failures)

    def test_malformed_topology_preserves_wire_contract_failures(self, package):
        log_dir, power_dir = package()
        manifest_path = power_dir / MANIFEST_FILENAME
        manifest = json.loads(manifest_path.read_text())
        manifest["status"] = "running"
        manifest["dcgm_exporter"]["port"] = 0
        manifest["expected_devices"][0]["gpu_index"] = None
        atomic_write_json(manifest_path, manifest)

        report = _validate(power_dir, log_dir)

        assert report.ok is False
        assert any("status is 'running'" in failure for failure in report.failures)
        assert any("dcgm_exporter.port is not a valid port number" in failure for failure in report.failures)
        assert any("manifest malformed" in failure for failure in report.failures)

    @pytest.mark.parametrize(
        ("key", "expected_failure"),
        [
            ("expected_devices", "expected_device_missing"),
            ("expected_windows", "no expected measurement window"),
        ],
    )
    def test_a_null_list_fails_with_its_specific_reason(self, package, key, expected_failure):
        """Structurally valid but empty reads as the precise gate, not as malformed."""
        log_dir, power_dir = package()
        manifest = json.loads((power_dir / MANIFEST_FILENAME).read_text())
        manifest[key] = None
        atomic_write_json(power_dir / MANIFEST_FILENAME, manifest)

        report = _validate(power_dir, log_dir)

        assert report.ok is False
        assert any(expected_failure in failure for failure in report.failures)

    @pytest.mark.parametrize("value", ["prefill", "prefill=x", "=4", "prefill=-1", "bogus=0"])
    def test_a_malformed_expect_role_is_a_usage_error(self, package, value):
        log_dir, power_dir = package()

        with pytest.raises(SystemExit) as excinfo:
            main(["--power-dir", str(power_dir), "--result-root", str(log_dir), "--expect-role", value])

        assert excinfo.value.code == 2

    def test_a_duplicate_expect_role_is_a_usage_error(self, package):
        log_dir, power_dir = package()
        argv = ["--power-dir", str(power_dir), "--result-root", str(log_dir)]

        with pytest.raises(SystemExit) as excinfo:
            main(argv + ["--expect-role", "prefill=4", "--expect-role", "prefill=8"])

        assert excinfo.value.code == 2

    def test_cli_reports_a_damaged_manifest_instead_of_crashing(self, package, capsys):
        log_dir, power_dir = package()
        manifest = json.loads((power_dir / MANIFEST_FILENAME).read_text())
        manifest["expected_devices"][0]["gpu_index"] = None
        atomic_write_json(power_dir / MANIFEST_FILENAME, manifest)

        code = main(["--power-dir", str(power_dir), "--result-root", str(log_dir)])

        assert code == 1
        assert "manifest malformed" in capsys.readouterr().out

    @pytest.mark.parametrize("payload", ["[]", "null"], ids=["array", "null"])
    def test_a_non_object_manifest_is_reported(self, package, payload):
        """Valid JSON that is not an object must fail before any field access."""
        log_dir, power_dir = package()
        (power_dir / MANIFEST_FILENAME).write_text(payload)

        report = _validate(power_dir, log_dir)

        assert report.ok is False
        assert any("manifest malformed: not an object" in failure for failure in report.failures)

    def test_unreadable_json_is_reported(self, package):
        log_dir, power_dir = package()
        (power_dir / MANIFEST_FILENAME).write_text("{not json")

        report = _validate(power_dir, log_dir)

        assert report.ok is False
        assert any("manifest unreadable" in failure for failure in report.failures)

    def test_unreadable_windows_directory_is_a_clean_cli_failure(self, package, monkeypatch, capsys):
        log_dir, power_dir = package()
        windows_dir = power_dir / WINDOWS_DIRNAME
        original_iterdir = Path.iterdir

        def fail_for_windows(path):
            if path == windows_dir:
                raise PermissionError("permission denied")
            return original_iterdir(path)

        monkeypatch.setattr(Path, "iterdir", fail_for_windows)

        code = main(["--power-dir", str(power_dir), "--result-root", str(log_dir)])

        assert code == 1
        assert Reason.MEASUREMENT_WINDOW_ARTIFACT_PATH_INVALID in capsys.readouterr().out
