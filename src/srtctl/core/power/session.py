# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Head-node power collection session.

The session owns the collector thread, the daemon request workers, and the
``samples.csv`` writer. Exporter processes stay owned by ``ProcessRegistry``;
the session only holds handles so it can notice a premature exit. Expected
telemetry invalidity is returned as an outcome rather than raised, so the
orchestrator can finalize artifacts before deciding the job's exit code.
"""

from __future__ import annotations

import hashlib
import logging
import queue
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from srtctl.core.power.contract import (
    COLLECT_CYCLE_TIMEOUT_GRACE_SECONDS,
    FATAL_LIFECYCLE_REASONS,
    MANIFEST_FILENAME,
    OPERATIONAL_FAILURE_REASONS,
    SAMPLES_FILENAME,
    STARTUP_FAILURE_REASONS,
    WINDOWS_DIRNAME,
    Reason,
    atomic_write_json,
    dedupe,
)
from srtctl.core.power.manifest import (
    STATUS_COMPLETE,
    STATUS_FAILED,
    STATUS_INCOMPLETE,
    STATUS_RUNNING,
    DcgmExporterIdentity,
    ExpectedWindow,
    PowerManifest,
)
from srtctl.core.power.parser import parse_power_scrape
from srtctl.core.power.samples import SampleRow, SampleWriter, derive_observed_devices, read_samples
from srtctl.core.power.topology import ExpectedDevice, validate_devices
from srtctl.core.power.windows import convert_running_windows, validate_expected_windows
from srtctl.core.processes import ManagedProcess
from srtctl.core.slurm import get_hostname_ip

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PowerEndpoint:
    """An allocated node and the resolved URL used to poll it."""

    hostname: str
    url: str


@dataclass(frozen=True)
class PowerSessionSettings:
    """Everything the session needs that comes from config and runtime."""

    power_dir: Path
    job_id: str
    run_name: str
    sample_interval_seconds: float
    startup_timeout_seconds: float
    request_timeout_seconds: float
    collector_join_timeout_seconds: float
    required: bool
    exporter_port: int
    exporter_image: str
    exporter_command: str
    network_interface: str | None = None
    producer_git_commit: str | None = None
    log_dir: Path | None = None

    @property
    def result_root(self) -> Path:
        """Root that measurement-window ``result_path`` values are relative to."""
        return self.log_dir if self.log_dir is not None else self.power_dir.parent


@dataclass(frozen=True)
class SessionOutcome:
    """Terminal state handed back to the orchestrator."""

    status: str
    publication_valid: bool
    reason_codes: tuple[str, ...]
    exit_nonzero: bool


@dataclass
class _EndpointResult:
    hostname: str
    rows: list[SampleRow]
    reason_codes: list[str]
    duration_seconds: float | None


class PowerTelemetrySession:
    """One idempotent power-collection session for one sweep."""

    def __init__(
        self,
        *,
        settings: PowerSessionSettings,
        expected_devices: Sequence[ExpectedDevice],
        expected_windows: Sequence[ExpectedWindow],
        nodes: Sequence[str],
        endpoints: Sequence[PowerEndpoint] | None = None,
    ):
        self._settings = settings
        self._nodes = list(nodes)
        self._endpoints: list[PowerEndpoint] = list(endpoints) if endpoints is not None else []
        self._endpoints_resolved = endpoints is not None

        # NOTE: only _writer_lock is held across I/O, so only it needs a timed acquire at shutdown.
        self._writer_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._exporters_lock = threading.Lock()
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._ready_at_monotonic: float | None = None
        self._thread: threading.Thread | None = None
        self._writer: SampleWriter | None = None
        self._exporters: list[ManagedProcess] = []

        self._scrape_seq = 0
        self._scrape_count = 0
        self._max_scrape_duration: float | None = None
        self._reasons: list[str] = []
        self._outcome: SessionOutcome | None = None
        self._mutation_disabled = False
        expected_device_list = list(expected_devices)
        self._expected_device_keys = frozenset(device.key for device in expected_device_list)

        self._manifest = PowerManifest(
            job_id=settings.job_id,
            run_name=settings.run_name,
            sample_interval_seconds=settings.sample_interval_seconds,
            request_timeout_seconds=settings.request_timeout_seconds,
            required=settings.required,
            started_at_unix=time.time(),
            producer_git_commit=settings.producer_git_commit,
            dcgm_exporter=_exporter_identity(settings),
            expected_devices=expected_device_list,
            expected_windows=list(expected_windows),
        )

    @property
    def power_dir(self) -> Path:
        return self._settings.power_dir

    @property
    def samples_path(self) -> Path:
        return self._settings.power_dir / SAMPLES_FILENAME

    @property
    def manifest_path(self) -> Path:
        return self._settings.power_dir / MANIFEST_FILENAME

    @property
    def windows_dir(self) -> Path:
        return self._settings.power_dir / WINDOWS_DIRNAME

    @property
    def collector_alive(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive()

    @property
    def writer_closed(self) -> bool:
        writer = self._writer
        return writer is None or writer.closed

    @property
    def artifact_mutation_disabled(self) -> bool:
        return self._mutation_disabled

    def initialize(self) -> None:
        """Create the exact CSV header and the ``starting`` manifest."""
        self.windows_dir.mkdir(parents=True, exist_ok=True)
        self._writer = SampleWriter(self.samples_path)
        self._write_manifest()

    def add_exporter(self, process: ManagedProcess) -> None:
        """Track an exporter the registry already owns."""
        with self._exporters_lock:
            self._exporters.append(process)

    def record_reason(self, reason: str) -> None:
        """Record a provider-level failure without raising into the sweep."""
        with self._state_lock:
            self._reasons.append(reason)

    def start_and_wait_for_readiness(self) -> bool:
        """Resolve endpoints and start collecting under one absolute deadline."""
        deadline = time.monotonic() + self._settings.startup_timeout_seconds

        if not self._endpoints_resolved:
            self._resolve_endpoints(deadline)
        if not self._endpoints:
            return False

        self._manifest.status = STATUS_RUNNING
        self._write_manifest()
        self._start_collector()
        return self._wait_for_readiness(deadline)

    def _resolve_endpoints(self, deadline: float) -> None:
        """Resolve every allocated hostname once, concurrently, before sampling."""

        def resolve(node: str) -> tuple[str, str] | None:
            try:
                ip = get_hostname_ip(node, self._settings.network_interface)
            except Exception as exc:  # noqa: BLE001 - an unresolvable node is a reason code
                logger.warning("Power endpoint resolution failed for %s: %s", node, exc)
                return None
            return (node, ip) if ip else None

        results, _ = _run_daemon_workers(
            [(f"PowerResolve-{node}", resolve, node) for node in self._nodes],
            deadline=deadline,
        )
        resolved = dict(result for result in results if result is not None)
        self._endpoints_resolved = True

        for node in self._nodes:
            ip = resolved.get(node)
            if ip is None:
                self.record_reason(Reason.ENDPOINT_RESOLUTION_FAILED)
                continue
            self._endpoints.append(
                PowerEndpoint(hostname=node, url=f"http://{ip}:{self._settings.exporter_port}/metrics")
            )

    def _start_collector(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="PowerCollector", daemon=True)
        self._thread.start()

    def _wait_for_readiness(self, deadline: float) -> bool:
        """Wait for one persisted *complete* scrape covering every expected device.

        The union of all scrapes is not enough: a flapping exporter could
        contribute one node per cycle and never have all devices live at once.
        """
        self._ready.wait(timeout=max(0.0, deadline - time.monotonic()))
        if self._ready_at_monotonic is not None and self._ready_at_monotonic < deadline:
            return True

        self.record_reason(Reason.EXPORTER_STARTUP_TIMEOUT)
        return False

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
        results, failures = _run_daemon_workers(
            [
                (f"PowerScrape-{endpoint.hostname}", lambda endpoint: self._poll(endpoint, scrape_seq), endpoint)
                for endpoint in endpoints
            ],
            deadline=deadline,
        )
        if failures:
            raise failures[0]

        rows: list[SampleRow] = []
        reasons: list[str] = []
        durations: list[float] = []
        settled = sorted(results, key=lambda item: item.hostname)
        for result in settled:
            rows.extend(result.rows)
            reasons.extend(result.reason_codes)
            if result.duration_seconds is not None:
                durations.append(result.duration_seconds)
        # A poller abandoned at the deadline settles nothing; it must still
        # account as a miss or the manifest under-reports scrape coverage.
        settled_hosts = {result.hostname for result in settled}
        reasons.extend(Reason.ENDPOINT_TIMEOUT for endpoint in endpoints if endpoint.hostname not in settled_hosts)

        with self._state_lock:
            self._reasons.extend(reasons)
            if durations:
                self._max_scrape_duration = max(durations + [self._max_scrape_duration or 0.0])

        with self._writer_lock:
            if self._mutation_disabled or self._writer is None:
                return 0
            self._writer.append(rows)
            self._writer.flush()
            observed_keys = {(row.hostname, row.gpu_index) for row in rows}
            if self._expected_device_keys and self._expected_device_keys <= observed_keys and not self._ready.is_set():
                self._ready_at_monotonic = time.monotonic()
                self._ready.set()
        return len(rows)

    def _poll(self, endpoint: PowerEndpoint, scrape_seq: int) -> _EndpointResult:
        """One endpoint request, timestamped adjacently on the head-node clock."""
        started_unix = time.time()
        started_monotonic = time.perf_counter()
        try:
            response = requests.get(endpoint.url, timeout=self._settings.request_timeout_seconds)
            response.raise_for_status()
            body = response.text
        except requests.Timeout:
            return _EndpointResult(endpoint.hostname, [], [Reason.ENDPOINT_TIMEOUT], None)
        except requests.RequestException:
            return _EndpointResult(endpoint.hostname, [], [Reason.ENDPOINT_HTTP_ERROR], None)
        settled_monotonic = time.perf_counter()
        settled_unix = time.time()

        scrape = parse_power_scrape(body)
        timestamp_unix = (started_unix + settled_unix) / 2
        rows = [
            SampleRow(
                timestamp_unix=timestamp_unix,
                scrape_seq=scrape_seq,
                hostname=endpoint.hostname,
                gpu_index=reading.gpu_index,
                gpu_uuid=reading.gpu_uuid,
                power_w=reading.power_w,
            )
            for reading in scrape.readings
        ]
        return _EndpointResult(
            hostname=endpoint.hostname,
            rows=rows,
            reason_codes=list(scrape.reason_codes),
            duration_seconds=settled_monotonic - started_monotonic,
        )

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
            logger.exception("Power collector stopped")
            self.record_reason(Reason.COLLECTOR_EXCEPTION)

    def _any_exporter_exited(self) -> bool:
        with self._exporters_lock:
            return any(not process.is_running for process in self._exporters)

    def _check_exporters(self) -> None:
        """A DCGM exporter exit during collection invalidates the run."""
        if self._stop.is_set():
            return
        if self._any_exporter_exited():
            self.record_reason(Reason.EXPORTER_EXITED)

    def stop_and_finalize(self, *, interrupted: bool = False, allow_window_mutation: bool = False) -> SessionOutcome:
        """Stop collection, close the writer, and commit the terminal manifest.

        All shutdown work shares one absolute deadline derived from
        ``collector_join_timeout_seconds``. A wedged collector must never keep
        the orchestrator from reaching ``ProcessRegistry.cleanup()``, so the
        writer lock is only ever acquired with a timeout here; if it cannot be
        taken, collector-owned state is left untouched and a minimal terminal
        manifest is written from the last committed snapshot instead.
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
            self._outcome = self._minimal_terminal_manifest()
            return self._outcome
        try:
            self._mutation_disabled = True
            if self._writer is not None:
                self._writer.close()
        finally:
            self._writer_lock.release()

        self._outcome = self._finalize_manifest(allow_window_mutation=allow_window_mutation)
        return self._outcome

    def _minimal_terminal_manifest(self) -> SessionOutcome:
        """Publish a terminal manifest without touching collector-owned state.

        ``samples.csv`` is deliberately not re-read: the collector may still be
        mid-write, and a torn tail would be indistinguishable from corruption.
        """
        with self._state_lock:
            reasons = list(self._reasons)
        self._manifest.reason_codes = list(dedupe(reasons))
        self._manifest.mark_terminal(
            status=STATUS_INCOMPLETE,
            stopped_at_unix=time.time(),
            publication_valid=False,
        )
        self._write_manifest()
        logger.error("Power collector did not release its writer; wrote a minimal terminal manifest")
        return SessionOutcome(
            status=self._manifest.status,
            publication_valid=False,
            reason_codes=tuple(self._manifest.reason_codes),
            exit_nonzero=self._exit_nonzero(self._manifest.reason_codes, publication_valid=False),
        )

    def _finalize_manifest(self, *, allow_window_mutation: bool) -> SessionOutcome:
        rows, sample_reasons = read_samples(self.samples_path)
        observed = derive_observed_devices(rows)
        devices = validate_devices(self._manifest.expected_devices, observed)

        with self._state_lock:
            reasons = [*self._reasons, *sample_reasons, *devices.reason_codes]
            self._manifest.scrape_count = self._scrape_count
            self._manifest.max_scrape_duration_seconds = self._max_scrape_duration

        if allow_window_mutation:
            convert_running_windows(self.windows_dir, reason="benchmark did not reach a formal end boundary")

        self._manifest.observed_devices = observed
        self._manifest.sample_row_count = len(rows)
        self._manifest.window_validations = validate_expected_windows(
            power_dir=self.power_dir,
            result_root=self._settings.result_root,
            expected_windows=self._manifest.expected_windows,
            expected_device_keys={device.key for device in self._manifest.expected_devices},
            observed_devices=observed,
            artifact_errors=self._manifest.artifact_errors,
        )
        reasons.extend(reason for validation in self._manifest.window_validations for reason in validation.reason_codes)
        # NOTE: an unusable artifact file is itself a publication gate, not something to ignore.
        reasons.extend(reason for error in self._manifest.artifact_errors for reason in error.reason_codes)

        status = self._terminal_status(reasons)
        windows_valid = bool(self._manifest.expected_windows) and all(
            validation.power_coverage_valid for validation in self._manifest.window_validations
        )
        publication_valid = (
            status == STATUS_COMPLETE
            and devices.valid
            and windows_valid
            and not sample_reasons
            and not self._manifest.artifact_errors
        )

        self._manifest.reason_codes = list(dedupe(reasons))
        self._manifest.mark_terminal(
            status=status,
            stopped_at_unix=time.time(),
            publication_valid=publication_valid,
        )
        self._write_manifest()

        return SessionOutcome(
            status=self._manifest.status,
            publication_valid=bool(self._manifest.publication_valid),
            reason_codes=tuple(self._manifest.reason_codes),
            exit_nonzero=self._exit_nonzero(self._manifest.reason_codes, bool(self._manifest.publication_valid)),
        )

    def _exit_nonzero(self, reasons: Sequence[str], publication_valid: bool = False) -> bool:
        """Measurement invalidity is mode-dependent; operational failure is not.

        Best-effort telemetry never turns a passing benchmark into a failure,
        but something left live or unreaped fails the job in either mode.
        """
        if any(reason in OPERATIONAL_FAILURE_REASONS for reason in reasons):
            return True
        return self._settings.required and not publication_valid

    def _terminal_status(self, reasons: Sequence[str]) -> str:
        """Lifecycle precedence: losing collection outranks failing to start it.

        ``failed`` means *required* startup could not establish collection. A
        best-effort run that keeps serving still reaches its normal finalizer,
        so it is ``complete`` — publication validity is a separate gate.
        """
        if any(reason in FATAL_LIFECYCLE_REASONS for reason in reasons):
            return STATUS_INCOMPLETE
        if self._settings.required and any(reason in STARTUP_FAILURE_REASONS for reason in reasons):
            return STATUS_FAILED
        return STATUS_COMPLETE

    def _write_manifest(self) -> None:
        atomic_write_json(self.manifest_path, self._manifest.to_dict())


def _exporter_identity(settings: PowerSessionSettings) -> DcgmExporterIdentity:
    """Record which exporter image produced the samples, hashed when it is a file.

    The image string is recorded verbatim. Routing a registry URI through
    ``Path`` would collapse ``docker://host/x`` to ``docker:/host/x``, so the
    manifest would disagree with what srun actually received.
    """
    image = settings.exporter_image
    digest: str | None = None
    candidate = Path(image)
    if "://" not in image and candidate.is_file():
        hasher = hashlib.sha256()
        with open(candidate, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                hasher.update(chunk)
        digest = hasher.hexdigest()
    return DcgmExporterIdentity(
        container_image_resolved=image,
        container_image_sha256=digest,
        port=settings.exporter_port,
        command=settings.exporter_command,
    )


def _run_daemon_workers(
    jobs: Sequence[tuple[str, Callable[[Any], Any], Any]],
    *,
    deadline: float,
) -> tuple[tuple[Any, ...], tuple[BaseException, ...]]:
    """Run each job on its own daemon thread and abandon stragglers at ``deadline``.

    Daemon threads mean neither a stuck HTTP request nor a slow resolver can
    keep interpreter exit alive past the caller's absolute deadline. Worker
    exceptions are returned so the caller decides whether they are fatal.
    """
    outcomes: queue.Queue[tuple[float, bool, Any]] = queue.Queue()

    def run(target: Callable[[Any], Any], argument: Any) -> None:
        try:
            value = target(argument)
        except Exception as exc:  # noqa: BLE001 - reported to the caller, never swallowed
            outcomes.put((time.monotonic(), False, exc))
        else:
            outcomes.put((time.monotonic(), True, value))

    threads = []
    for name, target, argument in jobs:
        thread = threading.Thread(target=run, name=name, args=(target, argument), daemon=True)
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join(timeout=max(0.0, deadline - time.monotonic()))

    values: list[Any] = []
    failures: list[BaseException] = []
    while True:
        try:
            completed_at, succeeded, value = outcomes.get_nowait()
        except queue.Empty:
            break
        if completed_at >= deadline:
            continue
        if succeeded:
            values.append(value)
        else:
            failures.append(value)
    return tuple(values), tuple(failures)
