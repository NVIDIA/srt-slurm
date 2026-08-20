# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Head-node power collector lifecycle against fake DCGM endpoints."""

import json
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from srtctl.cli.do_sweep import SweepOrchestrator
from srtctl.cli.mixins.benchmark_stage import BenchmarkStageMixin
from srtctl.cli.mixins.telemetry_stage import TelemetryStageMixin
from srtctl.core.power.contract import MANIFEST_FILENAME, SAMPLES_FILENAME, WINDOWS_DIRNAME, Reason
from srtctl.core.power.manifest import ExpectedWindow
from srtctl.core.power.samples import read_samples
from srtctl.core.power.session import PowerEndpoint, PowerSessionSettings, PowerTelemetrySession, _run_daemon_workers
from srtctl.core.power.topology import build_expected_devices
from srtctl.core.processes import ManagedProcess, ProcessRegistry
from srtctl.core.schema import TelemetryExporterConfig
from srtctl.core.topology import Process

GPUS_PER_NODE = 4


def _processes():
    return [
        Process(
            node="node-a",
            gpu_indices=frozenset(range(GPUS_PER_NODE)),
            sys_port=8081,
            http_port=30000,
            endpoint_mode="prefill",
            endpoint_index=0,
            het_group=0,
        ),
        Process(
            node="node-b",
            gpu_indices=frozenset(range(GPUS_PER_NODE)),
            sys_port=8082,
            http_port=30000,
            endpoint_mode="decode",
            endpoint_index=0,
            het_group=1,
        ),
    ]


def _body(prefix, count=GPUS_PER_NODE, watts=400.0):
    lines = ["# TYPE DCGM_FI_DEV_POWER_USAGE gauge"]
    for index in range(count):
        lines.append(
            f'DCGM_FI_DEV_POWER_USAGE{{gpu="{index}",UUID="GPU-{prefix}{index}",'
            f'device="nvidia{index}",Hostname="exporter-lies"}} {watts + index}'
        )
    return "\n".join(lines) + "\n"


class FakeExporter:
    """A localhost DCGM exporter whose body and latency are controllable."""

    def __init__(self, body, delay=0.0, barrier=None, fail_requests=0):
        self.body = body
        self.delay = delay
        self.barrier = barrier
        self.fail_requests = fail_requests
        self.request_count = 0
        exporter = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                exporter.request_count += 1
                if exporter.request_count <= exporter.fail_requests:
                    self.send_response(503)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                if exporter.barrier is not None:
                    try:
                        exporter.barrier.wait(timeout=5)
                    except threading.BrokenBarrierError:
                        self.send_response(500)
                        self.end_headers()
                        return
                if exporter.delay:
                    time.sleep(exporter.delay)
                payload = exporter.body.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *_args):
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=lambda: self._server.serve_forever(poll_interval=0.01), daemon=True)
        self._thread.start()

    @property
    def url(self):
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}/metrics"

    def stop(self):
        self._server.shutdown()
        self._server.server_close()


@pytest.fixture
def exporters():
    started = []

    def make(body, delay=0.0, barrier=None, fail_requests=0):
        exporter = FakeExporter(body, delay=delay, barrier=barrier, fail_requests=fail_requests)
        started.append(exporter)
        return exporter

    yield make
    for exporter in started:
        exporter.stop()


def _settings(tmp_path, **overrides):
    image = tmp_path / "dcgm-exporter.sqsh"
    image.write_bytes(b"squashfs")
    fields = {
        "power_dir": tmp_path / "logs" / "power",
        "job_id": "12345",
        "run_name": "recipe_12345",
        "sample_interval_seconds": 0.05,
        "startup_timeout_seconds": 2.0,
        "request_timeout_seconds": 0.5,
        "collector_join_timeout_seconds": 5.0,
        "required": True,
        "exporter_port": 9401,
        "exporter_image": str(image),
        "exporter_command": "dcgm-exporter --collect-interval=100 --address :9401",
    }
    fields.update(overrides)
    return PowerSessionSettings(**fields)


def _session(tmp_path, endpoints, *, processes=None, windows=None, **overrides):
    return PowerTelemetrySession(
        settings=_settings(tmp_path, **overrides),
        expected_devices=build_expected_devices(processes if processes is not None else _processes()),
        expected_windows=windows if windows is not None else [ExpectedWindow("sa-bench", 4)],
        nodes=["node-a", "node-b"],
        endpoints=endpoints,
    )


def _endpoints(*pairs):
    return [PowerEndpoint(hostname=hostname, url=url) for hostname, url in pairs]


def _running_exporter():
    """A just-launched srun process: still running, so poll() is None."""
    proc = MagicMock()
    proc.poll.return_value = None
    return proc


def _exited_exporter(name, returncode):
    popen = MagicMock(spec=subprocess.Popen)
    popen.poll.return_value = returncode
    popen.pid = 4242
    return ManagedProcess(name=name, popen=popen, log_file=None, node="node-a", critical=False)


def _manifest(session):
    return json.loads((session.power_dir / MANIFEST_FILENAME).read_text())


class TestDaemonWorkers:
    def test_completed_workers_return_values_and_failures(self):
        def succeed(value):
            return value * 2

        def fail(_value):
            raise RuntimeError("worker failed")

        results, failures = _run_daemon_workers(
            [("success", succeed, 2), ("failure", fail, None)],
            deadline=time.monotonic() + 1,
        )

        assert results == (4,)
        assert len(failures) == 1
        assert isinstance(failures[0], RuntimeError)

    @pytest.mark.parametrize("raises", [False, True])
    def test_late_worker_cannot_mutate_the_returned_snapshot(self, raises):
        release = threading.Event()
        finished = threading.Event()

        def finish_late(_argument):
            release.wait(5)
            finished.set()
            if raises:
                raise RuntimeError("late failure")
            return "late"

        results, failures = _run_daemon_workers(
            [("late-worker", finish_late, None)],
            deadline=time.monotonic() + 0.01,
        )

        assert results == ()
        assert failures == ()
        release.set()
        assert finished.wait(1)
        assert results == ()


class TestInitialize:
    def test_writes_header_and_starting_manifest_before_launch(self, tmp_path):
        session = _session(tmp_path, _endpoints(("node-a", "http://127.0.0.1:1/metrics")))

        session.initialize()

        manifest = _manifest(session)
        assert manifest["status"] == "starting"
        assert manifest["publication_valid"] is None
        assert manifest["stopped_at_unix"] is None
        assert len(manifest["expected_devices"]) == 2 * GPUS_PER_NODE
        assert manifest["dcgm_exporter"]["container_image_sha256"] is not None
        assert (session.power_dir / SAMPLES_FILENAME).read_text().startswith("schema_version,timestamp_unix")

        session.stop_and_finalize()


class TestCollection:
    def test_two_endpoints_yield_eight_unique_devices(self, tmp_path, exporters):
        a = exporters(_body("a"))
        b = exporters(_body("b"))
        session = _session(tmp_path, _endpoints(("node-a", a.url), ("node-b", b.url)))
        session.initialize()

        written = session.collect_once()
        session.stop_and_finalize()

        rows, reasons = read_samples(session.power_dir / SAMPLES_FILENAME)
        assert written == 2 * GPUS_PER_NODE
        assert reasons == ()
        assert len({(row.hostname, row.gpu_index) for row in rows}) == 2 * GPUS_PER_NODE
        assert len({row.gpu_uuid for row in rows}) == 2 * GPUS_PER_NODE
        assert {row.hostname for row in rows} == {"node-a", "node-b"}
        assert {row.scrape_seq for row in rows} == {0}

    def test_hostname_comes_from_the_endpoint_map(self, tmp_path, exporters):
        a = exporters(_body("a"))
        session = _session(
            tmp_path,
            _endpoints(("node-a", a.url)),
            processes=_processes()[:1],
        )
        session.initialize()
        session.collect_once()
        session.stop_and_finalize()

        rows, _ = read_samples(session.power_dir / SAMPLES_FILENAME)

        assert {row.hostname for row in rows} == {"node-a"}
        assert "exporter-lies" not in (session.power_dir / SAMPLES_FILENAME).read_text()

    def test_endpoints_are_polled_concurrently(self, tmp_path, exporters):
        barrier = threading.Barrier(2)
        a = exporters(_body("a"), barrier=barrier)
        b = exporters(_body("b"), barrier=barrier)
        session = _session(
            tmp_path,
            _endpoints(("node-a", a.url), ("node-b", b.url)),
            request_timeout_seconds=5.0,
        )
        session.initialize()

        written = session.collect_once()
        session.stop_and_finalize()

        assert written == 2 * GPUS_PER_NODE
        assert not barrier.broken

    def test_endpoint_timeout_does_not_reuse_stale_power(self, tmp_path, exporters):
        a = exporters(_body("a"))
        stalled = exporters(_body("b"), delay=5.0)
        session = _session(
            tmp_path,
            _endpoints(("node-a", a.url), ("node-b", stalled.url)),
            request_timeout_seconds=0.2,
        )
        session.initialize()

        session.collect_once()
        session.stop_and_finalize()

        rows, _ = read_samples(session.power_dir / SAMPLES_FILENAME)
        assert {row.hostname for row in rows} == {"node-a"}
        assert Reason.ENDPOINT_TIMEOUT in _manifest(session)["reason_codes"]

    def test_abandoned_poller_is_reported_as_endpoint_timeout(self, tmp_path, exporters, monkeypatch):
        """A worker still alive at the cycle deadline settles nothing but must still count as a miss."""
        a = exporters(_body("a"))
        session = _session(
            tmp_path,
            _endpoints(("node-a", a.url), ("node-b", a.url)),
            request_timeout_seconds=0.05,
        )
        session.initialize()
        real_poll = session._poll

        def poll(endpoint, scrape_seq):
            if endpoint.hostname == "node-b":
                time.sleep(3.0)
            return real_poll(endpoint, scrape_seq)

        monkeypatch.setattr(session, "_poll", poll)
        session.collect_once()
        session.stop_and_finalize()

        rows, _ = read_samples(session.power_dir / SAMPLES_FILENAME)
        assert {row.hostname for row in rows} == {"node-a"}
        assert Reason.ENDPOINT_TIMEOUT in _manifest(session)["reason_codes"]

    def test_endpoint_recovers_on_a_later_cycle(self, tmp_path, exporters):
        a = exporters(_body("a"))
        b = exporters(_body("b"), fail_requests=1)
        session = _session(tmp_path, _endpoints(("node-a", a.url), ("node-b", b.url)))
        session.initialize()

        first = session.collect_once()
        second = session.collect_once()
        session.stop_and_finalize()

        assert first == GPUS_PER_NODE
        assert second == 2 * GPUS_PER_NODE
        assert Reason.ENDPOINT_HTTP_ERROR in _manifest(session)["reason_codes"]


class TestReadiness:
    def test_readiness_uses_the_persisted_cycle_signal_without_rescanning_csv(self, tmp_path, exporters):
        a = exporters(_body("a"))
        b = exporters(_body("b"))
        session = _session(tmp_path, _endpoints(("node-a", a.url), ("node-b", b.url)))
        session.initialize()

        with patch(
            "srtctl.core.power.session.read_samples",
            side_effect=AssertionError("readiness must not rescan samples.csv"),
        ):
            ready = session.start_and_wait_for_readiness()

        assert ready is True
        session.stop_and_finalize()

    def test_complete_scrape_after_the_startup_deadline_is_not_ready(self, tmp_path, exporters):
        a = exporters(_body("a"))
        b = exporters(_body("b"))
        session = _session(tmp_path, _endpoints(("node-a", a.url), ("node-b", b.url)))
        session.initialize()
        session.collect_once()

        assert session._wait_for_readiness(time.monotonic() - 1.0) is False

        outcome = session.stop_and_finalize()
        assert Reason.EXPORTER_STARTUP_TIMEOUT in outcome.reason_codes

    def test_complete_scrape_before_deadline_survives_waiter_scheduling_delay(self, tmp_path, exporters):
        a = exporters(_body("a"))
        b = exporters(_body("b"))
        session = _session(tmp_path, _endpoints(("node-a", a.url), ("node-b", b.url)))
        session.initialize()
        session.collect_once()
        assert session._ready_at_monotonic is not None
        deadline = session._ready_at_monotonic + 1.0

        with patch("srtctl.core.power.session.time.monotonic", return_value=deadline + 1.0):
            assert session._wait_for_readiness(deadline) is True

        session.stop_and_finalize()

    def test_required_mode_reaches_readiness_when_every_device_reports(self, tmp_path, exporters):
        a = exporters(_body("a"))
        b = exporters(_body("b"))
        session = _session(tmp_path, _endpoints(("node-a", a.url), ("node-b", b.url)))
        session.initialize()

        assert session.start_and_wait_for_readiness() is True

        outcome = session.stop_and_finalize()
        assert outcome.status == "complete"
        assert Reason.EXPECTED_DEVICE_MISSING not in outcome.reason_codes
        assert Reason.EXPORTER_STARTUP_TIMEOUT not in outcome.reason_codes

    def test_missing_endpoint_fails_required_startup_within_one_deadline(self, tmp_path, exporters):
        a = exporters(_body("a"))
        session = _session(
            tmp_path,
            _endpoints(("node-a", a.url), ("node-b", "http://127.0.0.1:9/metrics")),
            startup_timeout_seconds=1.0,
        )
        session.initialize()

        started = time.perf_counter()
        ready = session.start_and_wait_for_readiness()
        elapsed = time.perf_counter() - started
        outcome = session.stop_and_finalize()

        assert ready is False
        assert elapsed < 4.0
        assert outcome.status == "failed"
        assert Reason.EXPORTER_STARTUP_TIMEOUT in outcome.reason_codes
        assert outcome.exit_nonzero is True

    def test_best_effort_mode_survives_a_missing_endpoint(self, tmp_path, exporters):
        a = exporters(_body("a"))
        session = _session(
            tmp_path,
            _endpoints(("node-a", a.url), ("node-b", "http://127.0.0.1:9/metrics")),
            startup_timeout_seconds=1.0,
            required=False,
        )
        session.initialize()

        session.start_and_wait_for_readiness()
        outcome = session.stop_and_finalize()

        assert outcome.exit_nonzero is False
        assert outcome.publication_valid is False
        assert (session.power_dir / SAMPLES_FILENAME).exists()

    def test_unresolvable_node_is_bounded_by_the_startup_deadline(self, tmp_path):
        session = PowerTelemetrySession(
            settings=_settings(tmp_path, startup_timeout_seconds=1.0),
            expected_devices=build_expected_devices(_processes()),
            expected_windows=[ExpectedWindow("sa-bench", 4)],
            nodes=["node-a", "node-b"],
        )
        session.initialize()

        released = threading.Event()

        def never_resolves(node, _interface=None):
            released.wait(30)
            return "10.0.0.1"

        try:
            with patch("srtctl.core.power.session.get_hostname_ip", side_effect=never_resolves):
                started = time.perf_counter()
                ready = session.start_and_wait_for_readiness()
                elapsed = time.perf_counter() - started

            outcome = session.stop_and_finalize()
        finally:
            released.set()

        assert ready is False
        assert elapsed < 5.0
        assert Reason.ENDPOINT_RESOLUTION_FAILED in outcome.reason_codes


class TestFailurePaths:
    def test_collector_exception_keeps_partial_artifacts_and_fails_strict(self, tmp_path, exporters):
        a = exporters(_body("a"))
        b = exporters(_body("b"))
        session = _session(tmp_path, _endpoints(("node-a", a.url), ("node-b", b.url)))
        session.initialize()
        assert session.start_and_wait_for_readiness() is True

        with patch("srtctl.core.power.session.parse_power_scrape", side_effect=RuntimeError("boom")):
            deadline = time.monotonic() + 5.0
            while session.collector_alive and time.monotonic() < deadline:
                time.sleep(0.02)
            outcome = session.stop_and_finalize()

        manifest = _manifest(session)

        assert Reason.COLLECTOR_EXCEPTION in outcome.reason_codes
        assert outcome.status == "incomplete"
        assert outcome.publication_valid is False
        assert outcome.exit_nonzero is True
        assert manifest["status"] == "incomplete"
        assert manifest["sample_row_count"] > 0

    def test_exporter_exit_is_recorded_as_fatal(self, tmp_path, exporters):
        a = exporters(_body("a"))
        b = exporters(_body("b"))
        session = _session(tmp_path, _endpoints(("node-a", a.url), ("node-b", b.url)))
        session.initialize()
        session.add_exporter(_exited_exporter("telemetry_dcgm_exporter_g0", 1))

        session.collect_once()
        outcome = session.stop_and_finalize()

        assert Reason.EXPORTER_EXITED in outcome.reason_codes
        assert outcome.status == "incomplete"

    def test_normal_shutdown_does_not_report_exporter_exit(self, tmp_path, exporters):
        a = exporters(_body("a"))
        b = exporters(_body("b"))
        session = _session(tmp_path, _endpoints(("node-a", a.url), ("node-b", b.url)))
        session.initialize()
        running = MagicMock(spec=subprocess.Popen)
        running.poll.return_value = None
        running.pid = 99
        session.add_exporter(
            ManagedProcess(name="telemetry_dcgm_exporter", popen=running, node="node-a", critical=False)
        )

        outcome = session.stop_and_finalize()

        assert Reason.EXPORTER_EXITED not in outcome.reason_codes

    def test_unexpected_device_invalidates_publication(self, tmp_path, exporters):
        a = exporters(_body("a", count=GPUS_PER_NODE + 1))
        b = exporters(_body("b"))
        session = _session(tmp_path, _endpoints(("node-a", a.url), ("node-b", b.url)))
        session.initialize()

        session.collect_once()
        outcome = session.stop_and_finalize()

        assert Reason.UNEXPECTED_DEVICE in outcome.reason_codes
        assert outcome.publication_valid is False


class TestPublication:
    """A complete session with a covered formal window is publishable."""

    def _write_window_and_result(self, session, start, end):
        result_dir = session.power_dir.parent / "sa-bench_isl_8192_osl_1024"
        result_dir.mkdir(parents=True, exist_ok=True)
        stem = "results_concurrency_4_gpus_8_ctx_4_gen_4"
        (result_dir / f"{stem}.json").write_text(
            json.dumps(
                {
                    "duration": end - start,
                    "benchmark_start_time_unix": start,
                    "benchmark_end_time_unix": end,
                    "completed": 40,
                }
            )
        )
        (session.windows_dir / f"{stem}.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "benchmark_type": "sa-bench",
                    "result_path": f"sa-bench_isl_8192_osl_1024/{stem}.json",
                    "concurrency": 4,
                    "benchmark_start_time_unix": start,
                    "benchmark_end_time_unix": end,
                    "duration": end - start,
                    "clock_source": "head_node_unix_clock",
                    "status": "completed",
                    "reason": None,
                }
            )
        )

    def test_covered_window_publishes(self, tmp_path, exporters):
        a = exporters(_body("a"))
        b = exporters(_body("b"))
        session = _session(tmp_path, _endpoints(("node-a", a.url), ("node-b", b.url)), sample_interval_seconds=0.2)
        session.initialize()
        assert session.start_and_wait_for_readiness() is True

        start = time.time()
        time.sleep(0.6)
        end = time.time()
        self._write_window_and_result(session, start, end)

        outcome = session.stop_and_finalize(allow_window_mutation=True)
        manifest = _manifest(session)

        assert outcome.status == "complete"
        assert outcome.publication_valid is True
        assert outcome.exit_nonzero is False
        assert outcome.reason_codes == ()
        assert manifest["window_validations"][0]["power_coverage_valid"] is True
        assert len(manifest["window_validations"][0]["per_device_max_sample_gap_seconds"]) == 2 * GPUS_PER_NODE
        assert manifest["artifact_errors"] == []

    def test_a_stray_artifact_file_blocks_publication(self, tmp_path, exporters):
        """A valid expected window must not publish beside an unusable file."""
        a = exporters(_body("a"))
        b = exporters(_body("b"))
        session = _session(tmp_path, _endpoints(("node-a", a.url), ("node-b", b.url)), sample_interval_seconds=0.2)
        session.initialize()
        assert session.start_and_wait_for_readiness() is True

        start = time.time()
        time.sleep(0.6)
        end = time.time()
        self._write_window_and_result(session, start, end)
        (session.windows_dir / "broken.json").write_text("{not json")

        outcome = session.stop_and_finalize(allow_window_mutation=True)
        manifest = _manifest(session)

        assert manifest["window_validations"][0]["power_coverage_valid"] is True
        assert outcome.publication_valid is False
        assert outcome.exit_nonzero is True
        assert Reason.MEASUREMENT_WINDOW_MALFORMED in outcome.reason_codes
        assert [error["path"] for error in manifest["artifact_errors"]] == [f"{WINDOWS_DIRNAME}/broken.json"]

    def test_exporter_dying_during_the_final_scrape_is_recorded(self, tmp_path, exporters):
        a = exporters(_body("a"))
        b = exporters(_body("b"))
        session = _session(tmp_path, _endpoints(("node-a", a.url), ("node-b", b.url)), sample_interval_seconds=0.2)
        session.initialize()

        popen = MagicMock(spec=subprocess.Popen)
        popen.poll.side_effect = lambda: 1 if session._stop.is_set() else None
        popen.pid = 4242
        session.add_exporter(ManagedProcess(name="telemetry_dcgm_exporter", popen=popen, node="node-a", critical=False))
        assert session.start_and_wait_for_readiness() is True

        start = time.time()
        time.sleep(0.5)
        end = time.time()
        self._write_window_and_result(session, start, end)

        outcome = session.stop_and_finalize(allow_window_mutation=True)

        assert Reason.EXPORTER_EXITED in outcome.reason_codes
        assert outcome.status == "incomplete"
        assert outcome.publication_valid is False
        assert _manifest(session)["publication_valid"] is False


class TestSessionOwnership:
    """The session must be finalizable even if startup blows up mid-flight."""

    def _harness(self, tmp_path, srun):
        class Harness(TelemetryStageMixin):
            def __init__(self):
                self.config = MagicMock()
                self.config.telemetry.enabled = True
                self.config.telemetry.storage_subdir = "power"
                self.config.telemetry.default_frequency = 0.05
                self.config.telemetry.startup_timeout_seconds = 0.2
                self.config.telemetry.request_timeout_seconds = 0.1
                self.config.telemetry.collector_join_timeout_seconds = 1.0
                self.config.telemetry.resolved_collector_join_timeout_seconds = 1.0
                self.config.telemetry.required = True
                self.config.telemetry.dcgm_exporter = TelemetryExporterConfig(
                    container_image="dcgm-exporter", port=9401
                )
                self.config.benchmark.type = "sa-bench"
                self.config.benchmark.get_concurrency_list.return_value = [4]
                self.runtime = MagicMock()
                self.runtime.log_dir = tmp_path
                self.runtime.job_id = "12345"
                self.runtime.run_name = "recipe_12345"
                self.runtime.network_interface = "eth0"
                self.runtime.nodes.het = False
                self.runtime.srun_options = {}
                self.runtime.container_mounts = {}

            @property
            def backend_processes(self):
                return _processes()[:1]

        return Harness()

    @pytest.mark.parametrize(
        ("method", "error"),
        [("initialize", OSError), ("start_and_wait_for_readiness", RuntimeError)],
    )
    def test_a_startup_failure_still_leaves_a_stored_session(self, tmp_path, method, error):
        """A run with no manifest at all is worse than one with a failed manifest."""
        harness = self._harness(tmp_path, None)

        with (
            patch("srtctl.cli.mixins.telemetry_stage.start_srun_process", return_value=_running_exporter()),
            patch.object(PowerTelemetrySession, method, side_effect=error("startup blew up")),
            pytest.raises(error),
        ):
            harness.start_power_telemetry(ProcessRegistry(job_id="12345"))

        assert harness._power_session is not None
        exit_code = harness.finalize_power_telemetry(0)
        manifest = json.loads((tmp_path / "power" / MANIFEST_FILENAME).read_text())
        assert manifest["status"] in ("complete", "incomplete", "failed")
        assert manifest["stopped_at_unix"] is not None
        assert exit_code == 1

    def test_exporter_launch_failure_blocks_the_benchmark(self, tmp_path):
        """Sibling of the readiness gate: a failed launch must not run the workload."""
        harness = self._harness(tmp_path, None)

        with patch(
            "srtctl.cli.mixins.telemetry_stage.start_srun_process",
            side_effect=RuntimeError("srun refused"),
        ):
            session = harness.start_power_telemetry(ProcessRegistry(job_id="12345"))

        assert harness.power_telemetry_blocks_benchmark() is True
        outcome = session.stop_and_finalize()
        assert Reason.EXPORTER_LAUNCH_FAILED in outcome.reason_codes
        assert outcome.exit_nonzero is True


class TestRequiredReadinessGate:
    """Required-mode startup failure must not burn the allocation."""

    def _orchestrator(self, tmp_path, *, required, ready):
        config = MagicMock()
        config.telemetry.enabled = True
        config.telemetry.required = required
        config.frontend.type = "dynamo"
        config.profiling.enabled = False
        runtime = MagicMock()
        runtime.log_dir = tmp_path
        runtime.job_id = "12345"
        orchestrator = SweepOrchestrator(config=config, runtime=runtime)
        orchestrator._power_session = MagicMock()
        orchestrator._power_telemetry_ready = ready
        return orchestrator

    @pytest.mark.parametrize(
        ("required", "ready", "blocks"),
        [(True, False, True), (False, False, False), (True, True, False)],
    )
    def test_gate_truth_table(self, tmp_path, required, ready, blocks):
        orchestrator = self._orchestrator(tmp_path, required=required, ready=ready)

        assert orchestrator.power_telemetry_blocks_benchmark() is blocks

    def test_unset_readiness_fails_closed(self, tmp_path):
        """A stored session whose readiness was never recorded must block."""
        orchestrator = self._orchestrator(tmp_path, required=True, ready=False)
        del orchestrator._power_telemetry_ready

        assert orchestrator.power_telemetry_blocks_benchmark() is True

    def test_run_skips_the_benchmark_when_required_telemetry_failed(self, tmp_path):
        orchestrator = self._orchestrator(tmp_path, required=True, ready=False)

        with (
            patch.object(SweepOrchestrator, "start_tachometer", return_value=[]) as start_tachometer,
            patch.object(SweepOrchestrator, "start_power_telemetry") as start_power,
            patch.object(SweepOrchestrator, "run_benchmark") as run_benchmark,
            patch.object(SweepOrchestrator, "start_all_workers", return_value={}),
            patch.object(SweepOrchestrator, "start_frontend", return_value=[]),
            patch.object(SweepOrchestrator, "start_head_infrastructure", return_value=MagicMock()),
            patch.object(SweepOrchestrator, "start_mooncake_master", return_value=None),
            patch.object(SweepOrchestrator, "_print_connection_info"),
            patch.object(SweepOrchestrator, "run_postprocess"),
            patch.object(SweepOrchestrator, "finalize_power_telemetry", side_effect=lambda code, **_: code),
            patch("srtctl.cli.do_sweep.record_resource_snapshot"),
            patch("srtctl.cli.do_sweep.write_lockfile"),
            patch("srtctl.cli.do_sweep.StatusReporter"),
            patch("srtctl.cli.do_sweep.setup_signal_handlers"),
            patch("srtctl.cli.do_sweep.start_process_monitor"),
        ):
            start_power.side_effect = lambda registry: orchestrator._power_session
            orchestrator.runtime.staged_model_path = None
            orchestrator.runtime.is_hf_model = False
            exit_code = orchestrator.run()

        run_benchmark.assert_not_called()
        start_tachometer.assert_called_once()
        assert exit_code == 1

    def test_eval_only_run_never_starts_power_telemetry(self, tmp_path):
        """Eval-only skips the benchmark, so required telemetry must not fail the run over missing windows."""
        orchestrator = self._orchestrator(tmp_path, required=True, ready=False)
        orchestrator._power_session = None

        with (
            patch.object(SweepOrchestrator, "start_tachometer", return_value=[]) as start_tachometer,
            patch.object(SweepOrchestrator, "start_power_telemetry") as start_power,
            patch.object(SweepOrchestrator, "run_benchmark") as run_benchmark,
            patch.object(SweepOrchestrator, "_run_post_eval", return_value=0),
            patch.object(SweepOrchestrator, "start_all_workers", return_value={}),
            patch.object(SweepOrchestrator, "start_frontend", return_value=[]),
            patch.object(SweepOrchestrator, "start_head_infrastructure", return_value=MagicMock()),
            patch.object(SweepOrchestrator, "start_mooncake_master", return_value=None),
            patch.object(SweepOrchestrator, "_print_connection_info"),
            patch.object(SweepOrchestrator, "run_postprocess"),
            patch("srtctl.cli.do_sweep.record_resource_snapshot"),
            patch("srtctl.cli.do_sweep.write_lockfile"),
            patch("srtctl.cli.do_sweep.StatusReporter"),
            patch("srtctl.cli.do_sweep.setup_signal_handlers"),
            patch("srtctl.cli.do_sweep.start_process_monitor"),
            patch.dict("os.environ", {"EVAL_ONLY": "true"}),
        ):
            orchestrator.runtime.staged_model_path = None
            orchestrator.runtime.is_hf_model = False
            exit_code = orchestrator.run()

        start_power.assert_not_called()
        start_tachometer.assert_called_once()
        run_benchmark.assert_not_called()
        assert exit_code == 0


class TestBenchmarkChildReaping:
    """An unreaped benchmark child is fatal and must never mutate a window."""

    class _Harness(TelemetryStageMixin):
        def __init__(self, session, reaped, *, allows_window_mutation=None):
            self._power_session = session
            self.benchmark_child_reaped = reaped
            self.benchmark_child_allows_window_mutation = allows_window_mutation

    def _session_with_running_window(self, tmp_path, exporters):
        a = exporters(_body("a"))
        b = exporters(_body("b"))
        session = _session(tmp_path, _endpoints(("node-a", a.url), ("node-b", b.url)))
        session.initialize()
        session.collect_once()
        (session.windows_dir / "results_concurrency_4_gpus_8_ctx_4_gen_4.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "benchmark_type": "sa-bench",
                    "result_path": "sa-bench_isl_8192_osl_1024/results_concurrency_4_gpus_8_ctx_4_gen_4.json",
                    "concurrency": 4,
                    "benchmark_start_time_unix": time.time(),
                    "benchmark_end_time_unix": None,
                    "duration": None,
                    "clock_source": "head_node_unix_clock",
                    "status": "running",
                    "reason": None,
                }
            )
        )
        return session

    def test_unreaped_child_is_recorded_and_leaves_the_window_alone(self, tmp_path, exporters):
        session = self._session_with_running_window(tmp_path, exporters)

        exit_code = self._Harness(session, False, allows_window_mutation=False).finalize_power_telemetry(0)

        window = json.loads((session.windows_dir / "results_concurrency_4_gpus_8_ctx_4_gen_4.json").read_text())
        manifest = _manifest(session)
        assert Reason.BENCHMARK_CHILD_REAP_TIMEOUT in manifest["reason_codes"]
        assert manifest["status"] == "incomplete"
        assert manifest["publication_valid"] is False
        assert window["status"] == "running"  # a surviving child must not be overwritten
        assert exit_code == 1

    def test_reaped_child_converts_the_window_and_records_nothing(self, tmp_path, exporters):
        session = self._session_with_running_window(tmp_path, exporters)

        self._Harness(session, True, allows_window_mutation=True).finalize_power_telemetry(0)

        window = json.loads((session.windows_dir / "results_concurrency_4_gpus_8_ctx_4_gen_4.json").read_text())
        manifest = _manifest(session)
        assert window["status"] == "interrupted"
        assert Reason.BENCHMARK_CHILD_REAP_TIMEOUT not in manifest["reason_codes"]
        assert Reason.MEASUREMENT_WINDOW_INCOMPLETE in manifest["reason_codes"]
        assert manifest["publication_valid"] is False

    def test_force_killed_local_child_leaves_the_window_alone(self, tmp_path, exporters):
        session = self._session_with_running_window(tmp_path, exporters)

        self._Harness(session, True, allows_window_mutation=False).finalize_power_telemetry(0)

        window = json.loads((session.windows_dir / "results_concurrency_4_gpus_8_ctx_4_gen_4.json").read_text())
        manifest = _manifest(session)
        assert window["status"] == "running"
        assert Reason.BENCHMARK_CHILD_REAP_TIMEOUT not in manifest["reason_codes"]
        assert Reason.MEASUREMENT_WINDOW_INCOMPLETE in manifest["reason_codes"]

    def test_no_benchmark_child_is_not_a_reap_failure(self, tmp_path, exporters):
        session = self._session_with_running_window(tmp_path, exporters)

        self._Harness(session, None, allows_window_mutation=None).finalize_power_telemetry(0)

        assert Reason.BENCHMARK_CHILD_REAP_TIMEOUT not in _manifest(session)["reason_codes"]

    def test_unexpected_finalizer_error_fails_even_in_best_effort_mode(self):
        """An operational finalizer crash is not ordinary measurement invalidity."""
        session = MagicMock()
        session.stop_and_finalize.side_effect = OSError("artifact write failed")
        harness = self._Harness(session, None)
        harness.config = MagicMock()
        harness.config.telemetry.required = False

        assert harness.finalize_power_telemetry(0) == 1

    @staticmethod
    def _benchmark_harness(tmp_path):
        class Harness(BenchmarkStageMixin):
            config = MagicMock()
            runtime = MagicMock()

            def _benchmark_node(self):
                return "node-a"

            def _get_benchmark_env(self, runner):
                return {}

        harness = Harness()
        harness.runtime.log_dir = tmp_path
        harness.runtime.srun_options = {}
        harness.runtime.nodes.het_group_for.return_value = None
        runner = MagicMock()
        runner.build_command.return_value = ["bench"]
        runner.get_environment.return_value = {}
        runner.get_container_image.return_value = "img"
        runner.get_container_mounts.return_value = {}
        runner.name = "SA-Bench"
        return harness, runner

    def test_sigterm_unwind_still_reaps_the_child(self, tmp_path):
        """The signal handler raises SystemExit, so only `finally` can reap."""
        proc = MagicMock(spec=subprocess.Popen)
        proc.poll.return_value = None
        proc.wait.return_value = -15
        harness, runner = self._benchmark_harness(tmp_path)

        with (
            patch("srtctl.cli.mixins.benchmark_stage.start_srun_process", return_value=proc),
            patch("srtctl.analysis.live_metrics.try_start_snapshotter", return_value=None),
            patch("srtctl.cli.mixins.benchmark_stage.time.sleep", side_effect=SystemExit(1)),
            pytest.raises(SystemExit),
        ):
            harness._run_benchmark_script(runner, tmp_path / "benchmark.out", threading.Event())

        proc.terminate.assert_called_once()
        proc.kill.assert_not_called()
        assert harness.benchmark_child_reaped is True

    def test_stop_request_reaps_the_child_before_returning(self, tmp_path):
        proc = MagicMock(spec=subprocess.Popen)
        proc.poll.return_value = None
        proc.wait.return_value = -15
        harness, runner = self._benchmark_harness(tmp_path)
        stop_event = threading.Event()
        stop_event.set()

        with (
            patch("srtctl.cli.mixins.benchmark_stage.start_srun_process", return_value=proc),
            patch("srtctl.analysis.live_metrics.try_start_snapshotter", return_value=None),
        ):
            exit_code = harness._run_benchmark_script(runner, tmp_path / "benchmark.out", stop_event)

        assert exit_code == 1
        proc.terminate.assert_called_once()
        proc.kill.assert_not_called()
        assert harness.benchmark_child_reaped is True

    def test_unreapable_child_on_unwind_stays_false(self, tmp_path):
        proc = MagicMock(spec=subprocess.Popen)
        proc.poll.return_value = None
        proc.wait.side_effect = subprocess.TimeoutExpired(cmd="bench", timeout=1)
        harness, runner = self._benchmark_harness(tmp_path)

        with (
            patch("srtctl.cli.mixins.benchmark_stage.start_srun_process", return_value=proc),
            patch("srtctl.analysis.live_metrics.try_start_snapshotter", return_value=None),
            patch("srtctl.cli.mixins.benchmark_stage.time.sleep", side_effect=SystemExit(1)),
            pytest.raises(SystemExit),
        ):
            harness._run_benchmark_script(runner, tmp_path / "benchmark.out", threading.Event())

        assert harness.benchmark_child_reaped is False

    def test_force_killed_child_on_unwind_does_not_allow_window_mutation(self, tmp_path):
        proc = MagicMock(spec=subprocess.Popen)
        proc.poll.return_value = None
        proc.wait.side_effect = [subprocess.TimeoutExpired(cmd="bench", timeout=1), -9]
        harness, runner = self._benchmark_harness(tmp_path)

        with (
            patch("srtctl.cli.mixins.benchmark_stage.start_srun_process", return_value=proc),
            patch("srtctl.analysis.live_metrics.try_start_snapshotter", return_value=None),
            patch("srtctl.cli.mixins.benchmark_stage.time.sleep", side_effect=SystemExit(1)),
            pytest.raises(SystemExit),
        ):
            harness._run_benchmark_script(runner, tmp_path / "benchmark.out", threading.Event())

        assert harness.benchmark_child_reaped is True
        assert harness.benchmark_child_allows_window_mutation is False


class TestExporterIdentity:
    """The manifest must record the image string srun actually received."""

    @pytest.mark.parametrize(
        "image",
        [
            "docker://nvcr.io/nvidia/k8s/dcgm-exporter:3.3.5-3.4.0-ubuntu22.04",
            "dcgm-exporter",
        ],
    )
    def test_registry_uris_are_recorded_verbatim(self, tmp_path, image):
        session = _session(tmp_path, _endpoints(), exporter_image=image)
        session.initialize()
        session.stop_and_finalize()

        exporter = _manifest(session)["dcgm_exporter"]
        assert exporter["container_image_resolved"] == image
        assert exporter["container_image_sha256"] is None

    def test_a_local_squashfs_is_still_hashed(self, tmp_path):
        session = _session(tmp_path, _endpoints())
        session.initialize()
        session.stop_and_finalize()

        exporter = _manifest(session)["dcgm_exporter"]
        assert exporter["container_image_resolved"].endswith("dcgm-exporter.sqsh")
        assert len(exporter["container_image_sha256"]) == 64


class TestTerminalStatusAndExit:
    """Lifecycle precedence and the required/best-effort exit table."""

    def _finalize(self, tmp_path, reasons, *, required):
        session = _session(tmp_path, _endpoints(), required=required)
        session.initialize()
        for reason in reasons:
            session.record_reason(reason)
        return session.stop_and_finalize()

    @pytest.mark.parametrize(
        ("reasons", "required", "status"),
        [
            ([Reason.EXPORTER_STARTUP_TIMEOUT, Reason.COLLECTOR_EXCEPTION], True, "incomplete"),
            ([Reason.COLLECTOR_EXCEPTION], True, "incomplete"),
            ([Reason.COLLECTOR_EXCEPTION], False, "incomplete"),
            ([Reason.EXPORTER_STARTUP_TIMEOUT], True, "failed"),
            ([Reason.EXPORTER_LAUNCH_FAILED], True, "failed"),
            ([Reason.ENDPOINT_RESOLUTION_FAILED], True, "failed"),
            ([Reason.EXPORTER_STARTUP_TIMEOUT], False, "complete"),
            ([], False, "complete"),
        ],
    )
    def test_terminal_status_precedence(self, tmp_path, reasons, required, status):
        outcome = self._finalize(tmp_path, reasons, required=required)

        assert outcome.status == status
        # No endpoint ever answered, so nothing here can be publishable, complete or not.
        assert outcome.publication_valid is False

    @pytest.mark.parametrize(
        ("reasons", "required", "exit_nonzero"),
        [
            ([], True, True),
            ([Reason.EXPORTER_STARTUP_TIMEOUT], True, True),
            ([], False, False),
            ([Reason.EXPORTER_STARTUP_TIMEOUT], False, False),
            ([Reason.COLLECTOR_EXCEPTION], False, False),
            ([Reason.BENCHMARK_CHILD_REAP_TIMEOUT], False, True),
            ([Reason.COLLECTOR_JOIN_TIMEOUT], False, True),
            ([Reason.BENCHMARK_CHILD_REAP_TIMEOUT], True, True),
        ],
    )
    def test_exit_code_table(self, tmp_path, reasons, required, exit_nonzero):
        assert self._finalize(tmp_path, reasons, required=required).exit_nonzero is exit_nonzero


class TestShutdown:
    def test_finalize_stops_the_collector_and_closes_the_writer(self, tmp_path, exporters):
        a = exporters(_body("a"))
        b = exporters(_body("b"))
        session = _session(tmp_path, _endpoints(("node-a", a.url), ("node-b", b.url)))
        session.initialize()
        session.start_and_wait_for_readiness()

        session.stop_and_finalize()

        assert session.collector_alive is False
        assert session.writer_closed is True
        assert not [thread for thread in threading.enumerate() if thread.name.startswith("PowerCollector")]

    def test_join_timeout_cannot_block_global_cleanup(self, tmp_path, exporters):
        """A wedged collector must not stop the orchestrator reaching registry cleanup."""
        a = exporters(_body("a"))
        b = exporters(_body("b"))
        session = _session(
            tmp_path,
            _endpoints(("node-a", a.url), ("node-b", b.url)),
            collector_join_timeout_seconds=1.0,
        )
        session.initialize()
        session.collect_once()

        release = threading.Event()

        def hold_writer():
            with session._writer_lock:
                release.wait(30)

        holder = threading.Thread(target=hold_writer, daemon=True)
        holder.start()
        deadline = time.monotonic() + 5.0
        while not session._writer_lock.locked() and time.monotonic() < deadline:
            time.sleep(0.01)

        started = time.perf_counter()
        outcome = session.stop_and_finalize()
        elapsed = time.perf_counter() - started
        release.set()

        assert elapsed < 4.0
        assert outcome.status == "incomplete"
        assert Reason.COLLECTOR_JOIN_TIMEOUT in outcome.reason_codes
        assert outcome.publication_valid is False
        assert outcome.exit_nonzero is True
        assert session.artifact_mutation_disabled is False
        assert session.writer_closed is False
        manifest = _manifest(session)
        assert manifest["status"] == "incomplete"
        assert manifest["publication_valid"] is False
        assert Reason.COLLECTOR_JOIN_TIMEOUT in manifest["reason_codes"]

    def test_finalize_is_idempotent(self, tmp_path, exporters):
        a = exporters(_body("a"))
        b = exporters(_body("b"))
        session = _session(tmp_path, _endpoints(("node-a", a.url), ("node-b", b.url)))
        session.initialize()
        session.start_and_wait_for_readiness()

        first = session.stop_and_finalize()
        manifest_after_first = _manifest(session)
        second = session.stop_and_finalize()

        assert second == first
        assert _manifest(session) == manifest_after_first

    def test_interrupted_shutdown_is_incomplete_but_preserves_rows(self, tmp_path, exporters):
        a = exporters(_body("a"))
        b = exporters(_body("b"))
        session = _session(tmp_path, _endpoints(("node-a", a.url), ("node-b", b.url)))
        session.initialize()
        session.collect_once()

        outcome = session.stop_and_finalize(interrupted=True)

        assert outcome.status == "incomplete"
        assert Reason.COLLECTOR_INTERRUPTED in outcome.reason_codes
        assert len(read_samples(session.power_dir / SAMPLES_FILENAME)[0]) == 2 * GPUS_PER_NODE

    def test_final_scrape_runs_on_normal_stop(self, tmp_path, exporters):
        a = exporters(_body("a"))
        b = exporters(_body("b"))
        session = _session(tmp_path, _endpoints(("node-a", a.url), ("node-b", b.url)), sample_interval_seconds=30.0)
        session.initialize()
        session.start_and_wait_for_readiness()

        session.stop_and_finalize()

        rows, _ = read_samples(session.power_dir / SAMPLES_FILENAME)
        assert max(row.scrape_seq for row in rows) >= 1

    def test_manifest_records_disk_derived_counts(self, tmp_path, exporters):
        a = exporters(_body("a"))
        b = exporters(_body("b"))
        session = _session(tmp_path, _endpoints(("node-a", a.url), ("node-b", b.url)))
        session.initialize()
        session.collect_once()
        session.collect_once()

        session.stop_and_finalize()
        manifest = _manifest(session)

        assert manifest["sample_row_count"] == 4 * GPUS_PER_NODE
        assert manifest["scrape_count"] == 2
        assert manifest["max_scrape_duration_seconds"] > 0
        assert len(manifest["observed_devices"]) == 2 * GPUS_PER_NODE
        assert Path(manifest["dcgm_exporter"]["container_image_resolved"]).name == "dcgm-exporter.sqsh"
