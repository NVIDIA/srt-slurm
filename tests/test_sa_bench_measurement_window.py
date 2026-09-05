# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SA-Bench formal measurement window and its power-coverage audit."""

import importlib
import importlib.util
import json
from contextlib import suppress
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from srtctl.cli.mixins.benchmark_stage import BenchmarkStageMixin
from srtctl.core.power.contract import (
    BENCHMARK_TYPE_SA_BENCH,
    CLOCK_SOURCE,
    CONTAINER_LOG_DIR,
    MAX_SAMPLE_GAP_SECONDS,
    MEASUREMENT_WINDOW_DIR_ENV,
    SCHEMA_VERSION,
    WINDOWS_DIRNAME,
    Reason,
)
from srtctl.core.power.manifest import ExpectedWindow
from srtctl.core.power.samples import SampleRow, derive_observed_devices
from srtctl.core.power.windows import (
    WINDOW_STATUS_COMPLETED,
    WINDOW_STATUS_FAILED,
    WINDOW_STATUS_INTERRUPTED,
    WINDOW_STATUS_RUNNING,
    convert_running_windows,
    validate_expected_windows,
)
from srtctl.core.schema import (
    BenchmarkConfig,
    FrontendConfig,
    ModelConfig,
    ProfilingConfig,
    ProfilingPhaseConfig,
    ResourceConfig,
    SrtConfig,
    TelemetryConfig,
    TelemetryExporterConfig,
)

SA_BENCH_DIR = Path(__file__).resolve().parents[1] / "src/srtctl/benchmarks/scripts/sa-bench"


def _benchmark_harness(tmp_path, *, enabled=True):
    telemetry = TelemetryConfig(
        enabled=enabled,
        default_frequency=1.0,
        storage_subdir="power",
        dcgm_exporter=TelemetryExporterConfig(container_image="dcgm-exporter", port=9401),
    )
    harness = BenchmarkStageMixin()
    harness.config = SrtConfig(
        name="test",
        model=ModelConfig(path="/model", container="/image", precision="fp8"),
        resources=ResourceConfig(gpu_type="gb200"),
        benchmark=BenchmarkConfig(type="sa-bench", concurrencies=[4], isl=8192, osl=1024),
        telemetry=telemetry,
    )
    harness.runtime = MagicMock()
    harness.runtime.log_dir = tmp_path
    harness.runtime.container_mounts = {tmp_path: Path("/logs")}
    return harness


def _load_measurement_window():
    spec = importlib.util.spec_from_file_location("sa_bench_measurement_window", SA_BENCH_DIR / "measurement_window.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


measurement_window = _load_measurement_window()
MeasurementWindow = measurement_window.MeasurementWindow

RESULT_STEM = "results_concurrency_4_gpus_8_ctx_4_gen_4"
RESULT_SUBDIR = "sa-bench_isl_8192_osl_1024"


@pytest.fixture
def logs(tmp_path):
    """A container ``/logs`` mount holding both results and power artifacts."""
    (tmp_path / RESULT_SUBDIR).mkdir()
    (tmp_path / "power" / WINDOWS_DIRNAME).mkdir(parents=True)
    return tmp_path


def _create(logs, *, save_result=True, window_dir=None, result_filename=f"{RESULT_STEM}.json"):
    return MeasurementWindow.create(
        save_result=save_result,
        window_dir=str(logs / "power" / WINDOWS_DIRNAME) if window_dir is None else window_dir,
        result_dir=str(logs / RESULT_SUBDIR),
        result_filename=result_filename,
        concurrency=4,
        log_root=str(logs),
    )


def _write_result(logs, *, start, end, duration, stem=RESULT_STEM):
    path = logs / RESULT_SUBDIR / f"{stem}.json"
    path.write_text(
        json.dumps(
            {
                "duration": duration,
                "benchmark_start_time_unix": start,
                "benchmark_end_time_unix": end,
                "completed": 40,
            }
        )
    )
    return path


def _window_json(logs, stem=RESULT_STEM):
    return json.loads((logs / "power" / WINDOWS_DIRNAME / f"{stem}.json").read_text())


def _samples(start, end, *, step=1.0, devices=(("node-a", 0, "GPU-a0"),), pad=None):
    rows = []
    seq = 0
    pad = max(2.0, step) if pad is None else pad
    timestamp = start - pad
    while timestamp <= end + pad:
        for hostname, gpu_index, uuid in devices:
            rows.append(SampleRow(timestamp, seq, hostname, gpu_index, uuid, 400.0))
        seq += 1
        timestamp += step
    return derive_observed_devices(rows)


def _validate(logs, observed, expected=(("sa-bench", 4),), errors=None):
    return validate_expected_windows(
        power_dir=logs / "power",
        result_root=logs,
        expected_windows=[ExpectedWindow(bt, c) for bt, c in expected],
        expected_device_keys={device.key for device in observed},
        observed_devices=observed,
        artifact_errors=errors if errors is not None else [],
    )


class TestWriterReaderContract:
    def test_standalone_writer_constants_match_the_power_contract(self):
        """The writer is mounted into the bench container and cannot import srtctl."""
        assert measurement_window.SCHEMA_VERSION == SCHEMA_VERSION
        assert measurement_window.BENCHMARK_TYPE == BENCHMARK_TYPE_SA_BENCH
        assert measurement_window.CLOCK_SOURCE == CLOCK_SOURCE
        assert measurement_window.WINDOW_DIR_ENV == MEASUREMENT_WINDOW_DIR_ENV
        assert measurement_window.CONTAINER_LOG_DIR == CONTAINER_LOG_DIR
        assert measurement_window.STATUS_RUNNING == WINDOW_STATUS_RUNNING
        assert measurement_window.STATUS_COMPLETED == WINDOW_STATUS_COMPLETED
        assert measurement_window.STATUS_FAILED == WINDOW_STATUS_FAILED
        assert measurement_window.STATUS_INTERRUPTED == WINDOW_STATUS_INTERRUPTED


class TestWindowWriterActivation:
    def test_warmup_without_save_result_writes_nothing(self, logs):
        assert _create(logs, save_result=False) is None
        assert _create(logs, result_filename=None) is None
        assert list((logs / "power" / WINDOWS_DIRNAME).iterdir()) == []

    def test_absent_window_dir_env_writes_nothing(self, logs):
        assert _create(logs, window_dir="") is None
        assert _create(logs, window_dir=str(logs / "nope")) is None

    def test_formal_run_creates_a_running_window_first(self, logs):
        window = _create(logs)

        window.mark_running(1785168100.0)

        payload = _window_json(logs)
        assert payload["schema_version"] == 1
        assert payload["benchmark_type"] == "sa-bench"
        assert payload["concurrency"] == 4
        assert payload["clock_source"] == "head_node_unix_clock"
        assert payload["status"] == "running"
        assert payload["benchmark_start_time_unix"] == 1785168100.0
        assert payload["benchmark_end_time_unix"] is None
        assert payload["duration"] is None
        assert payload["reason"] is None
        assert payload["result_path"] == f"{RESULT_SUBDIR}/{RESULT_STEM}.json"


class TestWindowStates:
    def test_completed_window_shape(self, logs):
        window = _create(logs)
        window.mark_running(1785168100.0)

        window.mark_completed(start_unix=1785168100.0, end_unix=1785168120.0, duration=20.0)

        payload = _window_json(logs)
        assert payload["status"] == "completed"
        assert payload["benchmark_end_time_unix"] == 1785168120.0
        assert payload["duration"] == 20.0
        assert payload["reason"] is None

    def test_failed_window_keeps_the_boundary_and_a_reason(self, logs):
        window = _create(logs)
        window.mark_running(1785168100.0)

        window.mark_failed(
            start_unix=1785168100.0, end_unix=1785168110.0, duration=10.0, reason="RuntimeError: upstream reset"
        )

        payload = _window_json(logs)
        assert payload["status"] == "failed"
        assert payload["duration"] == 10.0
        assert payload["reason"] == "RuntimeError: upstream reset"

    def test_atomic_writer_closes_raw_fd_when_fdopen_fails(self, logs, monkeypatch):
        captured = {}

        def fail(fd, *_args, **_kwargs):
            captured["fd"] = fd
            assert _kwargs["closefd"] is False
            raise OSError("too many open files")

        monkeypatch.setattr(measurement_window.os, "fdopen", fail)
        target = logs / "power" / WINDOWS_DIRNAME / "failed.json"
        try:
            with pytest.raises(OSError, match="too many open files"):
                measurement_window._atomic_write_json(str(target), {"status": "running"})

            with pytest.raises(OSError):
                measurement_window.os.fstat(captured["fd"])
            assert not target.exists()
            assert list(target.parent.glob(".failed.json.*")) == []
        finally:
            with suppress(OSError):
                measurement_window.os.close(captured["fd"])

    def test_atomic_writer_does_not_close_a_reused_fd_after_wrapper_failure(self, logs, monkeypatch):
        real_fdopen = measurement_window.os.fdopen
        replacement_fd = None
        replacement_path = logs / "replacement"

        def fail_after_wrapper_created(fd, *args, **kwargs):
            nonlocal replacement_fd
            handle = real_fdopen(fd, *args, **kwargs)
            handle.close()
            replacement_fd = measurement_window.os.open(
                replacement_path,
                measurement_window.os.O_CREAT | measurement_window.os.O_WRONLY,
            )
            raise OSError("wrapper setup failed")

        monkeypatch.setattr(measurement_window.os, "fdopen", fail_after_wrapper_created)
        target = logs / "power" / WINDOWS_DIRNAME / "failed.json"
        try:
            with pytest.raises(OSError, match="wrapper setup failed"):
                measurement_window._atomic_write_json(str(target), {"status": "running"})

            assert replacement_fd is not None
            measurement_window.os.fstat(replacement_fd)
        finally:
            if replacement_fd is not None:
                with suppress(OSError):
                    measurement_window.os.close(replacement_fd)

    def test_atomic_replacement_leaves_no_partial_file(self, logs):
        window = _create(logs)
        window.mark_running(1785168100.0)
        window.mark_completed(start_unix=1785168100.0, end_unix=1785168120.0, duration=20.0)

        assert [p.name for p in (logs / "power" / WINDOWS_DIRNAME).iterdir()] == [f"{RESULT_STEM}.json"]

    def test_multiple_concurrencies_do_not_overwrite_each_other(self, logs):
        for concurrency, stem in ((4, RESULT_STEM), (16, "results_concurrency_16_gpus_8_ctx_4_gen_4")):
            window = MeasurementWindow.create(
                save_result=True,
                window_dir=str(logs / "power" / WINDOWS_DIRNAME),
                result_dir=str(logs / RESULT_SUBDIR),
                result_filename=f"{stem}.json",
                concurrency=concurrency,
                log_root=str(logs),
            )
            window.mark_running(1785168100.0)
            window.mark_completed(start_unix=1785168100.0, end_unix=1785168120.0, duration=20.0)

        assert len(list((logs / "power" / WINDOWS_DIRNAME).iterdir())) == 2
        assert _window_json(logs)["concurrency"] == 4

    def test_orchestrator_converts_running_windows_to_interrupted(self, logs):
        window = _create(logs)
        window.mark_running(1785168100.0)

        convert_running_windows(logs / "power" / WINDOWS_DIRNAME, reason="benchmark child terminated")

        payload = _window_json(logs)
        assert payload["status"] == "interrupted"
        assert payload["benchmark_end_time_unix"] is None
        assert payload["duration"] is None
        assert payload["reason"] == "benchmark child terminated"


class TestCoverageValidation:
    def _completed(self, logs, *, start=1000.0, end=1020.0, duration=20.0, result_duration=None):
        window = _create(logs)
        window.mark_running(start)
        window.mark_completed(start_unix=start, end_unix=end, duration=duration)
        _write_result(logs, start=start, end=end, duration=result_duration if result_duration is not None else duration)
        return start, end

    def test_bracketed_window_with_small_gaps_is_valid(self, logs):
        start, end = self._completed(logs)

        rows = _validate(logs, _samples(start, end))

        assert len(rows) == 1
        assert rows[0].power_coverage_valid is True
        assert rows[0].reason_codes == ()
        assert rows[0].window_file == f"{WINDOWS_DIRNAME}/{RESULT_STEM}.json"
        assert rows[0].per_device_max_sample_gap_seconds["node-a/GPU-a0"] == pytest.approx(1.0)

    def test_missing_expected_window_is_never_vacuously_valid(self, logs):
        rows = _validate(logs, _samples(1000.0, 1020.0))

        assert rows[0].power_coverage_valid is False
        assert rows[0].window_file is None
        assert Reason.MEASUREMENT_WINDOW_MISSING in rows[0].reason_codes
        assert rows[0].per_device_max_sample_gap_seconds == {}

    def test_device_without_bracketing_samples_is_invalid(self, logs):
        start, end = self._completed(logs)
        late = _samples(start + 5, end)

        rows = _validate(logs, late)

        assert rows[0].power_coverage_valid is False
        assert Reason.MEASUREMENT_WINDOW_NOT_BRACKETED in rows[0].reason_codes

    def test_gap_exactly_at_the_threshold_passes(self, logs):
        start, end = self._completed(logs)

        rows = _validate(logs, _samples(start, end, step=MAX_SAMPLE_GAP_SECONDS))

        assert rows[0].power_coverage_valid is True

    def test_gap_above_the_threshold_fails(self, logs):
        start, end = self._completed(logs)

        rows = _validate(logs, _samples(start, end, step=MAX_SAMPLE_GAP_SECONDS + 0.5))

        assert rows[0].power_coverage_valid is False
        assert Reason.SAMPLE_GAP_EXCEEDED in rows[0].reason_codes

    def test_result_timing_mismatch_is_reported(self, logs):
        start, end = self._completed(logs, result_duration=19.0)

        rows = _validate(logs, _samples(start, end))

        assert rows[0].power_coverage_valid is False
        assert Reason.MEASUREMENT_WINDOW_RESULT_MISMATCH in rows[0].reason_codes

    @pytest.mark.parametrize(
        ("end", "duration", "valid"),
        [
            (1020.0, 19.5, True),  # exactly max(0.5s, 1% of 20s) = 0.5s of skew
            (1020.0, 19.49, False),
            (1200.0, 20.0, False),  # the wall clock disagrees with the monotonic duration outright
        ],
    )
    def test_clock_tolerance_boundary(self, logs, end, duration, valid):
        window = _create(logs)
        window.mark_running(1000.0)
        window.mark_completed(start_unix=1000.0, end_unix=end, duration=duration)
        _write_result(logs, start=1000.0, end=end, duration=duration)

        rows = _validate(logs, _samples(1000.0, end))

        assert rows[0].power_coverage_valid is valid
        if not valid:
            assert Reason.MEASUREMENT_WINDOW_CLOCK_MISMATCH in rows[0].reason_codes

    def test_missing_result_file_is_reported(self, logs):
        window = _create(logs)
        window.mark_running(1000.0)
        window.mark_completed(start_unix=1000.0, end_unix=1020.0, duration=20.0)

        rows = _validate(logs, _samples(1000.0, 1020.0))

        assert rows[0].power_coverage_valid is False
        assert Reason.MEASUREMENT_WINDOW_RESULT_MISSING in rows[0].reason_codes

    def test_non_object_result_json_is_reported(self, logs):
        """Valid JSON that is not an object must not reach ``result.get()``."""
        start, end = self._completed(logs)
        (logs / RESULT_SUBDIR / f"{RESULT_STEM}.json").write_text("[]")

        rows = _validate(logs, _samples(start, end))

        assert rows[0].power_coverage_valid is False
        assert Reason.MEASUREMENT_WINDOW_RESULT_MISMATCH in rows[0].reason_codes

    def test_uuid_change_invalidates_the_window_verdict(self, logs):
        start, end = self._completed(logs)
        swapped = derive_observed_devices(
            [
                SampleRow(start - 1.0, 0, "node-a", 0, "GPU-a0", 400.0),
                SampleRow(start + 1.0, 1, "node-a", 0, "GPU-swapped", 400.0),
                SampleRow(end + 1.0, 2, "node-a", 0, "GPU-swapped", 400.0),
            ]
        )

        rows = _validate(logs, swapped)

        assert rows[0].power_coverage_valid is False
        assert Reason.GPU_UUID_CHANGED in rows[0].reason_codes

    def test_running_window_is_incomplete(self, logs):
        window = _create(logs)
        window.mark_running(1000.0)

        rows = _validate(logs, _samples(1000.0, 1020.0))

        assert rows[0].power_coverage_valid is False
        assert Reason.MEASUREMENT_WINDOW_INCOMPLETE in rows[0].reason_codes

    def test_one_invalid_window_among_several_is_isolated(self, logs):
        self._completed(logs)
        stale_stem = "results_concurrency_16_gpus_8_ctx_4_gen_4"
        other = MeasurementWindow.create(
            save_result=True,
            window_dir=str(logs / "power" / WINDOWS_DIRNAME),
            result_dir=str(logs / RESULT_SUBDIR),
            result_filename=f"{stale_stem}.json",
            concurrency=16,
            log_root=str(logs),
        )
        other.mark_running(1000.0)

        rows = _validate(logs, _samples(1000.0, 1020.0), expected=(("sa-bench", 4), ("sa-bench", 16)))

        by_concurrency = {row.concurrency: row for row in rows}
        assert by_concurrency[4].power_coverage_valid is True
        assert by_concurrency[16].power_coverage_valid is False


class TestArtifactErrors:
    def test_unexpected_window_file_is_recorded(self, logs):
        stale = logs / "power" / WINDOWS_DIRNAME / "results_concurrency_99_gpus_8.json"
        stale.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "benchmark_type": "sa-bench",
                    "result_path": f"{RESULT_SUBDIR}/results_concurrency_99_gpus_8.json",
                    "concurrency": 99,
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

        _validate(logs, _samples(1000.0, 1020.0), errors=errors)

        assert [error.path for error in errors] == [f"{WINDOWS_DIRNAME}/results_concurrency_99_gpus_8.json"]
        assert Reason.MEASUREMENT_WINDOW_UNEXPECTED in errors[0].reason_codes

    def test_malformed_window_file_is_recorded(self, logs):
        (logs / "power" / WINDOWS_DIRNAME / "broken.json").write_text("{not json")
        errors = []

        rows = _validate(logs, _samples(1000.0, 1020.0), errors=errors)

        assert Reason.MEASUREMENT_WINDOW_MALFORMED in errors[0].reason_codes
        assert rows[0].power_coverage_valid is False

    def test_duplicate_windows_for_one_key_invalidate_it(self, logs):
        self._write_duplicate(logs)
        errors = []

        rows = _validate(logs, _samples(1000.0, 1020.0), errors=errors)

        assert rows[0].power_coverage_valid is False
        assert Reason.MEASUREMENT_WINDOW_DUPLICATE in rows[0].reason_codes
        assert len(errors) == 2

    def _write_duplicate(self, logs):
        """Two well-formed windows both claiming (sa-bench, 4)."""
        for stem in (RESULT_STEM, "results_concurrency_4_gpus_8"):
            _write_result(logs, start=1000.0, end=1020.0, duration=20.0, stem=stem)
            (logs / "power" / WINDOWS_DIRNAME / f"{stem}.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "benchmark_type": "sa-bench",
                        "result_path": f"{RESULT_SUBDIR}/{stem}.json",
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

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("status", "interrupted"),  # the orchestrator always records why
            ("reason", "premature"),  # nothing has gone wrong while still running
            ("clock_source", "node_local_clock"),
            ("concurrency", True),  # bool is an int subclass; must not key as concurrency 1
            ("schema_version", True),  # same trap: True == 1
        ],
    )
    def test_a_mutated_window_field_is_malformed(self, logs, field, value):
        window = _create(logs)
        window.mark_running(1000.0)
        path = logs / "power" / WINDOWS_DIRNAME / f"{RESULT_STEM}.json"
        payload = json.loads(path.read_text())
        payload[field] = value
        path.write_text(json.dumps(payload))
        errors = []

        rows = _validate(logs, _samples(1000.0, 1020.0), errors=errors)

        assert Reason.MEASUREMENT_WINDOW_MALFORMED in errors[0].reason_codes
        assert rows[0].power_coverage_valid is False

    @pytest.mark.parametrize("result_path", ["/etc/passwd", "../escape.json", "a/../../b.json"])
    def test_unsafe_result_paths_are_rejected(self, logs, result_path):
        path = logs / "power" / WINDOWS_DIRNAME / f"{RESULT_STEM}.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "benchmark_type": "sa-bench",
                    "result_path": result_path,
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

        rows = _validate(logs, _samples(1000.0, 1020.0), errors=errors)

        assert rows[0].power_coverage_valid is False
        assert Reason.MEASUREMENT_WINDOW_RESULT_PATH_INVALID in errors[0].reason_codes

    def test_benchmark_stage_injects_the_container_windows_dir(self, tmp_path):
        harness = _benchmark_harness(tmp_path)

        env = harness._get_measurement_window_env()

        assert env == {"SRT_MEASUREMENT_WINDOW_DIR": f"/logs/power/{WINDOWS_DIRNAME}"}

    def test_other_providers_get_no_window_dir(self, tmp_path):
        assert _benchmark_harness(tmp_path, enabled=False)._get_measurement_window_env() == {}

    def test_sa_bench_env_keeps_window_after_logical_endpoint_refactor(self, tmp_path):
        """One benchmark env must carry logical endpoints, slow_down, and the window dir together."""
        harness = _benchmark_harness(tmp_path)
        harness.config = SrtConfig(
            name="test",
            model=ModelConfig(path="/model", container="/image", precision="fp8"),
            resources=ResourceConfig(
                gpu_type="gb200",
                prefill_nodes=1,
                decode_nodes=1,
                prefill_workers=1,
                decode_workers=1,
            ),
            benchmark=BenchmarkConfig(
                type="sa-bench",
                concurrencies=[4],
                isl=8192,
                osl=1024,
                slow_down_sleep_time=1.0,
                slow_down_wait_time=1.0,
            ),
            telemetry=harness.config.telemetry,
            frontend=FrontendConfig(type="sglang"),
            profiling=ProfilingConfig(
                type="nsys",
                prefill=ProfilingPhaseConfig(start_step=1, stop_step=2),
                decode=ProfilingPhaseConfig(start_step=1, stop_step=2),
            ),
        )
        processes = [SimpleNamespace(is_leader=True, endpoint_mode="decode", node="node-d", http_port=1234, sys_port=0)]
        harness.runtime.environment = {}
        harness.runtime.network_interface = "eth0"
        runner = SimpleNamespace(name="SA-Bench")

        with (
            patch.object(BenchmarkStageMixin, "backend_processes", processes),
            patch(
                "srtctl.cli.mixins.benchmark_stage.get_hostname_ip",
                side_effect=lambda node, _interface: node if isinstance(node, str) else "head",
            ),
        ):
            env = harness._get_benchmark_env(runner)

        assert env["PROFILE_DECODE_ENDPOINTS"] == "node-d:1234"
        assert env["SA_BENCH_SLOW_DOWN_URLS"] == "http://node-d:1234"
        assert env["SRT_MEASUREMENT_WINDOW_DIR"] == "/logs/power/windows"

    def test_bench_script_saves_results_only_for_the_formal_run(self):
        script = (SA_BENCH_DIR / "bench.sh").read_text()
        warmup, _, formal = script.partition("num_prompts=$((concurrency * NUM_PROMPTS_MULT))")

        assert "--save-result" not in warmup
        assert "--save-result --result-dir" in formal

    def test_result_path_symlinked_outside_the_root_is_rejected(self, logs, tmp_path_factory):
        """The window file itself is safe; its result_path escapes via a symlink."""
        outside = tmp_path_factory.mktemp("outside_result_root")
        (outside / f"{RESULT_STEM}.json").write_text("{}")
        (logs / RESULT_SUBDIR / "escape").symlink_to(outside, target_is_directory=True)

        path = logs / "power" / WINDOWS_DIRNAME / f"{RESULT_STEM}.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "benchmark_type": "sa-bench",
                    "result_path": f"{RESULT_SUBDIR}/escape/{RESULT_STEM}.json",
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

        rows = _validate(logs, _samples(1000.0, 1020.0), errors=errors)

        assert rows[0].power_coverage_valid is False
        assert Reason.MEASUREMENT_WINDOW_RESULT_PATH_INVALID in errors[0].reason_codes

    def test_symlinked_windows_directory_is_rejected(self, logs, tmp_path_factory):
        """The whole directory moved outside and linked back must not pass."""
        outside = tmp_path_factory.mktemp("outside_windows")
        _write_result(logs, start=1000.0, end=1020.0, duration=20.0)
        (outside / f"{RESULT_STEM}.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "benchmark_type": "sa-bench",
                    "result_path": f"{RESULT_SUBDIR}/{RESULT_STEM}.json",
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
        windows = logs / "power" / WINDOWS_DIRNAME
        windows.rmdir()
        windows.symlink_to(outside, target_is_directory=True)
        errors = []

        rows = _validate(logs, _samples(1000.0, 1020.0), errors=errors)

        assert rows[0].power_coverage_valid is False
        assert [error.path for error in errors] == [WINDOWS_DIRNAME]
        assert Reason.MEASUREMENT_WINDOW_ARTIFACT_PATH_INVALID in errors[0].reason_codes

    def test_symlinked_window_is_rejected(self, logs):
        real = logs / "outside.json"
        real.write_text("{}")
        (logs / "power" / WINDOWS_DIRNAME / f"{RESULT_STEM}.json").symlink_to(real)
        errors = []

        rows = _validate(logs, _samples(1000.0, 1020.0), errors=errors)

        assert rows[0].power_coverage_valid is False
        assert Reason.MEASUREMENT_WINDOW_ARTIFACT_PATH_INVALID in errors[0].reason_codes
