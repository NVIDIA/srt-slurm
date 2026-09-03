# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Head-node host CPU power collection session.

The shape mirrors :mod:`srtctl.core.power.session`: exporters run on the worker
nodes and are owned by ``ProcessRegistry``, the collector thread lives in the
orchestrator process, and every sample is timestamped on the head node's clock
so CPU and GPU power share one time base.

The provider is ACPI: ``cpu-power-exporter`` reads ``/sys/class/hwmon`` and
publishes ``cpu_power_acpi_watts`` per power rail. There is deliberately no GPU
topology and no device expectation here -- host rails are not allocated to
workers, so the artifact is a flat time series plus the sensors that were
actually observed. Coverage is per host: publication requires that every
rail discovered at readiness bracketed every benchmark the orchestrator ran, on
every configured node, with the same maximum sample gap the GPU leg's
measurement windows are held to. Spans come
from the orchestrator rather than ``windows/*.json`` because the CPU leg also
serves benchmark types that write no window artifact.
"""

from __future__ import annotations

import csv
import ipaddress
import logging
import math
import socket
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import requests
from prometheus_client.parser import text_string_to_metric_families

from srtctl import __version__ as PRODUCER_VERSION
from srtctl.core.power.contract import (
    CLOCK_SOURCE,
    COLLECT_CYCLE_TIMEOUT_GRACE_SECONDS,
    FATAL_LIFECYCLE_REASONS,
    MANIFEST_FILENAME,
    OPERATIONAL_FAILURE_REASONS,
    POWER_UNIT,
    SAMPLES_FILENAME,
    SCHEMA_VERSION,
    STARTUP_FAILURE_REASONS,
    Reason,
    atomic_write_json,
    dedupe,
)
from srtctl.core.power.manifest import (
    STATUS_COMPLETE,
    STATUS_FAILED,
    STATUS_INCOMPLETE,
    STATUS_RUNNING,
    STATUS_STARTING,
)
from srtctl.core.power.session import SessionOutcome, run_daemon_workers
from srtctl.core.power.windows import check_series_coverage
from srtctl.core.processes import ManagedProcess
from srtctl.core.slurm import get_hostname_ip

logger = logging.getLogger(__name__)

CPU_POWER_PRODUCER = "srt-slurm.cpu-power"
CPU_POWER_METRIC = "cpu_power_acpi_watts"
CPU_POWER_SCOPE = "host_cpu_power_rail_as_reported_by_acpi"

# A scrape of every rail on a dense host is a few KB; a megabyte is a broken or
# hostile exporter, and reading it to the end would burn the cycle budget.
MAX_SCRAPE_BYTES = 1 << 20

CPU_SAMPLES_HEADER = (
    "schema_version",
    "timestamp_unix",
    "scrape_seq",
    "hostname",
    "sensor",
    "rail_type",
    "socket",
    "power_w",
)


@dataclass(frozen=True)
class CpuPowerSessionSettings:
    """Everything the CPU session needs that comes from config and runtime."""

    cpu_dir: Path
    job_id: str
    run_name: str
    source: str
    sample_interval_seconds: float
    startup_timeout_seconds: float
    request_timeout_seconds: float
    collector_join_timeout_seconds: float
    required: bool
    # ``auto`` tolerates a cluster with no ACPI rails; ``acpi`` (or ``required``)
    # makes exporter readiness a startup failure.
    acpi_mandatory: bool
    exporter_port: int
    exporter_command: str
    network_interface: str | None = None
    producer_git_commit: str | None = None


@dataclass(frozen=True)
class CpuPowerReading:
    """One power rail within a single scrape."""

    sensor: str
    rail_type: str
    socket: str
    power_w: float


@dataclass(frozen=True)
class CpuEndpoint:
    """An allocated node and the resolved URL used to poll it."""

    hostname: str
    url: str


@dataclass(frozen=True)
class BenchmarkSpanCoverage:
    """Whether every expected rail sampled across one benchmark run."""

    start_unix: float
    end_unix: float
    covered: bool
    reason_codes: tuple[str, ...]
    per_series_max_sample_gap_seconds: dict[str, float]

    def to_dict(self) -> dict[str, object]:
        return {
            "start_unix": self.start_unix,
            "end_unix": self.end_unix,
            "covered": self.covered,
            "reason_codes": list(self.reason_codes),
            "per_series_max_sample_gap_seconds": dict(self.per_series_max_sample_gap_seconds),
        }


@dataclass
class _EndpointResult:
    hostname: str
    readings: list[CpuPowerReading]
    reason_codes: list[str]


def cpu_endpoint_host(node: str, network_interface: str | None = None) -> str | None:
    """The URL host of ``node``'s CPU power exporter, or ``None`` if it has no address.

    The single source of truth for both consumers: the telemetry session's own
    endpoints and the ``AIPERF_SERVER_METRICS_URLS`` projection handed to the
    benchmark client. Computing them separately let the two disagree about
    which nodes are reachable and how an address is spelled.

    ``get_hostname_ip`` hands back the hostname unchanged when it cannot resolve
    it, and a URL carrying a name is re-resolved on every scrape -- so a node
    without a literal address is omitted rather than handed to a client that
    would spend the run on DNS failures. IPv6 is bracketed, without which the
    address's own colons would be read as the port separator.
    """
    try:
        ip = get_hostname_ip(node, network_interface)
    except Exception as exc:  # noqa: BLE001 - an unresolvable node is a caller's reason code
        logger.warning("CPU power endpoint resolution failed for %s: %s", node, exc)
        return None
    if not ip:
        return None
    try:
        address = ipaddress.ip_address(ip)
    except ValueError:
        logger.warning("CPU power endpoint for %s resolved to %r, not an address", node, ip)
        return None
    return f"[{ip}]" if isinstance(address, ipaddress.IPv6Address) else ip


def parse_cpu_power_scrape(text: str) -> tuple[tuple[CpuPowerReading, ...], tuple[str, ...]]:
    """Parse one exporter ``/metrics`` body into publishable rail readings.

    The ``sensor`` label is the exporter's stable per-rail identity, so it is
    what distinguishes readings; ``type``/``socket`` are descriptive only and
    may repeat across rails.
    """
    reasons: list[str] = []
    try:
        families = list(text_string_to_metric_families(text))
    # Same third-party boundary as the DCGM parser: malformed exposition has
    # raised ValueError, KeyError and IndexError across supported versions.
    except Exception:  # noqa: BLE001
        return (), (Reason.ENDPOINT_PARSE_ERROR,)

    by_sensor: dict[str, CpuPowerReading] = {}
    duplicated: set[str] = set()
    saw_sample = False

    for family in families:
        for sample in family.samples:
            if sample.name != CPU_POWER_METRIC:
                continue
            saw_sample = True
            sensor = (sample.labels.get("sensor") or "").strip()
            if not sensor:
                reasons.append(Reason.CPU_SENSOR_MISSING)
                continue
            value = sample.value
            if not math.isfinite(value) or value < 0:
                reasons.append(Reason.INVALID_POWER_VALUE)
                continue
            if sensor in by_sensor:
                duplicated.add(sensor)
                continue
            by_sensor[sensor] = CpuPowerReading(
                sensor=sensor,
                rail_type=(sample.labels.get("type") or "").strip(),
                socket=(sample.labels.get("socket") or "").strip(),
                power_w=value,
            )

    if duplicated:
        reasons.append(Reason.DUPLICATE_POWER_METRIC)
        for sensor in duplicated:
            by_sensor.pop(sensor, None)
    if not saw_sample:
        reasons.append(Reason.CPU_POWER_METRIC_MISSING)

    return tuple(by_sensor[sensor] for sensor in sorted(by_sensor)), dedupe(reasons)


def _abort_scrape(response: requests.Response) -> None:
    """End a scrape that overran its budget, from another thread.

    Closing the response is not enough: a reader blocked in recv stays blocked
    until the peer sends something. Shutting the socket down fails the read
    immediately, which is the only way the budget is a wall-clock guarantee.
    """
    raw = getattr(response, "raw", None)
    sock = getattr(getattr(raw, "_connection", None), "sock", None)
    if sock is None:
        # urllib3 hands the socket to http.client, which wraps it twice.
        sock = getattr(getattr(getattr(raw, "_fp", None), "fp", None), "raw", None)
        sock = getattr(sock, "_sock", None)
    try:
        if sock is not None:
            sock.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass
    try:
        response.close()
    except Exception:  # the scrape is being abandoned either way
        logger.debug("closing an abandoned CPU power scrape failed", exc_info=True)


class CpuPowerTelemetrySession:
    """One idempotent host CPU-power collection session for one sweep."""

    def __init__(self, *, settings: CpuPowerSessionSettings, nodes: Sequence[str]):
        self._settings = settings
        self._nodes = list(nodes)
        self._endpoints: list[CpuEndpoint] = []

        self._writer_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._exporters_lock = threading.Lock()
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread: threading.Thread | None = None
        self._handle = None
        self._writer = None
        self._mutation_disabled = False

        self._exporters: list[ManagedProcess] = []
        self._scrape_seq = 0
        self._scrape_count = 0
        self._row_count = 0
        self._observed: dict[str, set[str]] = {}
        # One head-clock timestamp per rail per producing cycle, keyed
        # "<host>/<sensor>": the series a measurement window is bracketed
        # against. Per host would be too coarse -- a package rail can stop
        # reporting while the remaining rails keep the host's series dense.
        self._sample_times: dict[str, list[float]] = {}
        # The rails that answered in the readiness cycle. Publication holds the
        # run to what the session promised at readiness, so a rail that
        # disappears mid-benchmark is a coverage failure rather than an
        # expectation that quietly shrinks with it.
        self._expected_series: set[str] = set()
        self._benchmark_spans: list[tuple[float, float]] = []
        self._span_coverage: list[BenchmarkSpanCoverage] = []
        self._reasons: list[str] = []
        self._status = STATUS_STARTING
        self._started_at_unix = time.time()
        self._stopped_at_unix: float | None = None
        self._publication_valid: bool | None = None
        self._ready_at_monotonic: float | None = None
        self._outcome: SessionOutcome | None = None

    @property
    def cpu_dir(self) -> Path:
        return self._settings.cpu_dir

    @property
    def samples_path(self) -> Path:
        return self._settings.cpu_dir / SAMPLES_FILENAME

    @property
    def manifest_path(self) -> Path:
        return self._settings.cpu_dir / MANIFEST_FILENAME

    @property
    def collector_alive(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive()

    def initialize(self) -> None:
        """Create the exact CSV header and the ``starting`` manifest."""
        self.samples_path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = open(self.samples_path, "w", newline="", encoding="utf-8")  # noqa: SIM115
        try:
            self._writer = csv.writer(self._handle)
            self._writer.writerow(CPU_SAMPLES_HEADER)
            self._handle.flush()
        except BaseException:
            self._handle.close()
            self._handle = None
            raise
        self._write_manifest()

    def add_exporter(self, process: ManagedProcess) -> None:
        """Track an exporter the registry already owns."""
        with self._exporters_lock:
            self._exporters.append(process)

    def record_benchmark_span(self, start_unix: float, end_unix: float) -> None:
        """Mark the interval a benchmark actually ran, on the head node clock."""
        with self._state_lock:
            self._benchmark_spans.append((start_unix, end_unix))

    def record_reason(self, reason: str) -> None:
        """Record a provider-level failure without raising into the sweep."""
        with self._state_lock:
            self._reasons.append(reason)

    def start_and_wait_for_readiness(self) -> bool:
        """Resolve endpoints and start collecting under one absolute deadline."""
        deadline = time.monotonic() + self._settings.startup_timeout_seconds
        self._resolve_endpoints(deadline)
        if not self._endpoints:
            return False
        # A node that never resolved is a node this session cannot cover, so in
        # mandatory mode readiness fails closed rather than on the survivors.
        unresolved = len(self._nodes) - len(self._endpoints)

        self._status = STATUS_RUNNING
        self._write_manifest()
        self._thread = threading.Thread(target=self._run, name="CpuPowerCollector", daemon=True)
        self._thread.start()

        self._ready.wait(timeout=max(0.0, deadline - time.monotonic()))
        # NOTE: Event.wait returning true says only that readiness happened, not
        # that it happened in time; the timestamp is what enforces the deadline.
        ready_at = self._ready_at_monotonic
        if ready_at is None or ready_at > deadline:
            self.record_reason(Reason.EXPORTER_STARTUP_TIMEOUT)
            return False
        return not (unresolved and self._settings.acpi_mandatory)

    def _resolve_endpoints(self, deadline: float) -> None:
        """Resolve every allocated hostname once, concurrently, before sampling.

        Only literal addresses become endpoints. ``get_hostname_ip`` hands back
        the hostname unchanged when it cannot resolve it, and a URL carrying a
        name is re-resolved on every scrape -- inside ``requests.get``, before
        any socket exists, where neither the request timeout nor the scrape
        watchdog reaches. A wedged resolver would then strand one poll thread
        per cycle. Resolution happens once, here, under this deadline; a node
        that does not yield an address is a node the session cannot cover.
        """

        def resolve(node: str) -> tuple[str, str] | None:
            host = cpu_endpoint_host(node, self._settings.network_interface)
            return (node, host) if host else None

        results, _ = run_daemon_workers(
            [(f"CpuPowerResolve-{node}", resolve, node) for node in self._nodes],
            deadline=deadline,
        )
        resolved = dict(result for result in results if result is not None)
        for node in self._nodes:
            host = resolved.get(node)
            if host is None:
                self.record_reason(Reason.ENDPOINT_RESOLUTION_FAILED)
                continue
            self._endpoints.append(
                CpuEndpoint(hostname=node, url=f"http://{host}:{self._settings.exporter_port}/metrics")
            )

    def collect_once(self) -> int:
        """Run one logical cycle: poll every endpoint concurrently, append rows."""
        with self._writer_lock:
            if self._mutation_disabled:
                return 0
            endpoints = list(self._endpoints)
        with self._state_lock:
            scrape_seq = self._scrape_seq
            self._scrape_seq += 1
            self._scrape_count += 1

        # NOTE: requests applies its timeout to connect and read separately, so an endpoint can take 2x.
        deadline = time.monotonic() + 2 * self._settings.request_timeout_seconds + COLLECT_CYCLE_TIMEOUT_GRACE_SECONDS
        results, failures = run_daemon_workers(
            [(f"CpuPowerScrape-{endpoint.hostname}", self._poll, endpoint) for endpoint in endpoints],
            deadline=deadline,
        )
        if failures:
            raise failures[0]

        settled = sorted(results, key=lambda item: item.hostname)
        timestamp_unix = time.time()
        rows: list[list[object]] = []
        reasons: list[str] = []
        for result in settled:
            reasons.extend(result.reason_codes)
            for reading in result.readings:
                rows.append(
                    [
                        SCHEMA_VERSION,
                        repr(timestamp_unix),
                        scrape_seq,
                        result.hostname,
                        reading.sensor,
                        reading.rail_type,
                        reading.socket,
                        repr(reading.power_w),
                    ]
                )
        # A poller abandoned at the deadline settles nothing; it must still
        # account as a miss or the manifest under-reports scrape coverage.
        settled_hosts = {result.hostname for result in settled}
        reasons.extend(Reason.ENDPOINT_TIMEOUT for endpoint in endpoints if endpoint.hostname not in settled_hosts)

        with self._state_lock:
            self._reasons.extend(reasons)
            for result in settled:
                if result.readings:
                    self._observed.setdefault(result.hostname, set()).update(r.sensor for r in result.readings)
                    for reading in result.readings:
                        self._sample_times.setdefault(f"{result.hostname}/{reading.sensor}", []).append(timestamp_unix)

        with self._writer_lock:
            if self._mutation_disabled or self._writer is None:
                return 0
            self._writer.writerows(rows)
            self._handle.flush()
            self._row_count += len(rows)
            # Readiness means one cycle in which every polled node served rails:
            # the union across cycles would let a flapping exporter look healthy.
            covering = {result.hostname for result in settled if result.readings}
            if endpoints and covering == {endpoint.hostname for endpoint in endpoints}:
                if self._ready_at_monotonic is None:
                    self._ready_at_monotonic = time.monotonic()
                    with self._state_lock:
                        self._expected_series = {
                            f"{result.hostname}/{reading.sensor}" for result in settled for reading in result.readings
                        }
                self._ready.set()
        return len(rows)

    def _poll(self, endpoint: CpuEndpoint) -> _EndpointResult:
        # requests' timeout is per socket operation, and a chunk read blocks
        # until the chunk is full, so an exporter that trickles bytes trips
        # neither. collect_once can abandon such a poller but cannot end it: the
        # thread and its socket would survive every cycle for the whole run.
        # The watchdog is what actually ends the scrape -- it tears the socket
        # down under the reader, which fails the read for real.
        expired = threading.Event()
        lock = threading.Lock()
        open_response: list[requests.Response] = []

        def abort() -> None:
            with lock:
                expired.set()
                for response in open_response:
                    _abort_scrape(response)

        watchdog = threading.Timer(2 * self._settings.request_timeout_seconds, abort)
        watchdog.daemon = True
        watchdog.start()
        try:
            with requests.get(endpoint.url, timeout=self._settings.request_timeout_seconds, stream=True) as response:
                with lock:
                    # The watchdog may have fired between the request and here.
                    open_response.append(response)
                    if expired.is_set():
                        _abort_scrape(response)
                response.raise_for_status()
                chunks: list[bytes] = []
                size = 0
                for chunk in response.iter_content(8192):
                    chunks.append(chunk)
                    size += len(chunk)
                    if size > MAX_SCRAPE_BYTES:
                        return _EndpointResult(endpoint.hostname, [], [Reason.ENDPOINT_HTTP_ERROR])
                body = b"".join(chunks).decode("utf-8", errors="replace")
        except requests.Timeout:
            return _EndpointResult(endpoint.hostname, [], [Reason.ENDPOINT_TIMEOUT])
        except (requests.RequestException, OSError, ValueError):
            # A torn-down socket surfaces as a transport error, not a timeout.
            reason = Reason.ENDPOINT_TIMEOUT if expired.is_set() else Reason.ENDPOINT_HTTP_ERROR
            return _EndpointResult(endpoint.hostname, [], [reason])
        finally:
            watchdog.cancel()
        if expired.is_set():
            return _EndpointResult(endpoint.hostname, [], [Reason.ENDPOINT_TIMEOUT])

        readings, reasons = parse_cpu_power_scrape(body)
        return _EndpointResult(endpoint.hostname, list(readings), list(reasons))

    def _run(self) -> None:
        """Collector thread: fixed-cadence cycles that never overlap."""
        interval = self._settings.sample_interval_seconds
        try:
            next_cycle = time.monotonic()
            while not self._stop.is_set():
                self.collect_once()
                self._check_exporters()
                next_cycle += interval
                self._stop.wait(max(0.0, next_cycle - time.monotonic()))
            self.collect_once()
        except Exception:
            logger.exception("CPU power collector stopped")
            self.record_reason(Reason.COLLECTOR_EXCEPTION)

    def _any_exporter_exited(self) -> bool:
        with self._exporters_lock:
            return any(not process.is_running for process in self._exporters)

    def _check_exporters(self) -> None:
        if self._stop.is_set():
            return
        if self._any_exporter_exited():
            self.record_reason(Reason.EXPORTER_EXITED)

    def stop_and_finalize(self, *, interrupted: bool = False) -> SessionOutcome:
        """Stop collection, close the writer, and commit the terminal manifest.

        All shutdown work shares one absolute deadline. A wedged collector must
        never keep the orchestrator from reaching ``ProcessRegistry.cleanup()``,
        so the writer lock is only ever acquired with a timeout here.
        """
        if self._outcome is not None:
            return self._outcome

        deadline = time.monotonic() + self._settings.collector_join_timeout_seconds
        self._check_exporters()
        self._stop.set()

        thread = self._thread
        if thread is not None:
            thread.join(timeout=max(0.0, deadline - time.monotonic()))
            if thread.is_alive():
                self.record_reason(Reason.COLLECTOR_JOIN_TIMEOUT)
        if interrupted:
            self.record_reason(Reason.COLLECTOR_INTERRUPTED)
        # NOTE: the pre-stop poll cannot see an exporter that died during the final scrape.
        if self._any_exporter_exited():
            self.record_reason(Reason.EXPORTER_EXITED)

        if not self._writer_lock.acquire(timeout=max(0.0, deadline - time.monotonic())):
            self.record_reason(Reason.COLLECTOR_JOIN_TIMEOUT)
            self._outcome = self._terminal(STATUS_INCOMPLETE, publication_valid=False)
            logger.error("CPU power collector did not release its writer; wrote a minimal terminal manifest")
            return self._outcome
        try:
            self._mutation_disabled = True
            if self._handle is not None:
                self._handle.close()
                self._handle = None
        finally:
            self._writer_lock.release()

        self._span_coverage = self._audit_spans()
        with self._state_lock:
            self._reasons.extend(reason for span in self._span_coverage for reason in span.reason_codes)
            reasons = list(self._reasons)
        status = self._terminal_status(reasons)
        # Every configured node must be covered: an endpoint dropped at
        # resolution time cannot silently shrink what publication means, and a
        # session that only ever produced its readiness sample measured nothing.
        spans_covered = bool(self._span_coverage) and all(span.covered for span in self._span_coverage)
        publication_valid = (
            status == STATUS_COMPLETE
            and self._row_count > 0
            and bool(self._nodes)
            and set(self._observed) == set(self._nodes)
            and bool(self._expected_series)
            and spans_covered
        )
        self._outcome = self._terminal(status, publication_valid=publication_valid)
        return self._outcome

    def _audit_spans(self) -> list[BenchmarkSpanCoverage]:
        """Hold every benchmark span to the GPU leg's bracketing and gap rules."""
        with self._state_lock:
            spans = list(self._benchmark_spans)
            sample_times = {series: list(times) for series, times in self._sample_times.items()}
            expected = sorted(self._expected_series)
        if not spans:
            logger.warning("CPU power session finalized without a benchmark span; nothing to publish")

        coverage: list[BenchmarkSpanCoverage] = []
        for start, end in spans:
            gaps, reasons = check_series_coverage(start, end, expected, sample_times)
            coverage.append(
                BenchmarkSpanCoverage(
                    start_unix=start,
                    end_unix=end,
                    covered=not reasons,
                    reason_codes=dedupe(reasons),
                    per_series_max_sample_gap_seconds=gaps,
                )
            )
        return coverage

    def _terminal_status(self, reasons: Sequence[str]) -> str:
        """Lifecycle precedence: losing collection outranks failing to start it."""
        if any(reason in FATAL_LIFECYCLE_REASONS for reason in reasons):
            return STATUS_INCOMPLETE
        if self._settings.acpi_mandatory and any(reason in STARTUP_FAILURE_REASONS for reason in reasons):
            return STATUS_FAILED
        return STATUS_COMPLETE

    def _terminal(self, status: str, *, publication_valid: bool) -> SessionOutcome:
        self._status = status
        self._publication_valid = publication_valid
        self._stopped_at_unix = time.time()
        self._write_manifest()
        with self._state_lock:
            reasons = dedupe(self._reasons)
        # Best-effort telemetry never turns a passing benchmark into a failure,
        # but something left unreaped fails the job in either mode.
        exit_nonzero = any(reason in OPERATIONAL_FAILURE_REASONS for reason in reasons) or (
            self._settings.required and not publication_valid
        )
        return SessionOutcome(
            status=status,
            publication_valid=publication_valid,
            reason_codes=reasons,
            exit_nonzero=exit_nonzero,
        )

    def _write_manifest(self) -> None:
        settings = self._settings
        with self._state_lock:
            reasons = list(dedupe(self._reasons))
            observed = {host: sorted(sensors) for host, sensors in sorted(self._observed.items())}
            scrape_count = self._scrape_count
        atomic_write_json(
            self.manifest_path,
            {
                "schema_version": SCHEMA_VERSION,
                "producer": CPU_POWER_PRODUCER,
                "producer_version": PRODUCER_VERSION,
                "producer_git_commit": settings.producer_git_commit,
                "job_id": settings.job_id,
                "run_name": settings.run_name,
                "metric": CPU_POWER_METRIC,
                "unit": POWER_UNIT,
                "scope": CPU_POWER_SCOPE,
                "clock_source": CLOCK_SOURCE,
                "source": settings.source,
                "required": settings.required,
                "sample_interval_seconds": settings.sample_interval_seconds,
                "exporter": {"port": settings.exporter_port, "command": settings.exporter_command},
                "nodes": list(self._nodes),
                "endpoints": [endpoint.url for endpoint in self._endpoints],
                "status": self._status,
                "started_at_unix": self._started_at_unix,
                "stopped_at_unix": self._stopped_at_unix,
                "scrape_count": scrape_count,
                "sample_row_count": self._row_count,
                "observed_sensors": observed,
                "benchmark_spans": [span.to_dict() for span in self._span_coverage],
                "publication_valid": self._publication_valid,
                "reason_codes": reasons,
            },
        )
