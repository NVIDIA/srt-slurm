# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Fire-and-forget status reporter for external job tracking.

This module provides optional status reporting to one or more external API endpoints.
If no endpoints are configured or all are unreachable, operations silently continue.
The API contract is defined in srtctl.contract.

Configuration (in srtslurm.yaml or recipe YAML):
    # Single endpoint (backward-compatible)
    reporting:
      status:
        endpoint: "https://status.example.com"

    # Multiple endpoints
    reporting:
      status:
        endpoints:
          - "https://status.example.com"
          - "https://status2.example.com"

    # Both (merged, deduplicated)
    reporting:
      status:
        endpoint: "https://status.example.com"
        endpoints:
          - "https://status2.example.com"

    # Also push new log and metric output every 10 seconds
    reporting:
      status:
        endpoint: "https://status.example.com"
        logging-stream-interval: 10
"""

import logging
import os
import re
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO

import requests

from srtctl.contract import JobCreatePayload, JobStage, JobStatus, JobUpdatePayload

if TYPE_CHECKING:
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.schema import ReportingConfig, ReportingStatusConfig, SrtConfig

logger = logging.getLogger(__name__)

# Environment variable the reporter reads its bearer token from unless
# ``reporting.status.token_env`` names another one. The token itself never
# appears in a recipe or srtslurm.yaml: the resolved config is written to the
# lockfile and copied into the log directory that reporting.s3 uploads.
DEFAULT_TOKEN_ENV = "SRTCTL_STATUS_TOKEN"


def _token_env(status: "ReportingStatusConfig | None") -> str:
    return (status.token_env if status and status.token_env else None) or DEFAULT_TOKEN_ENV


def _auth_headers(token_env: str) -> dict[str, str]:
    """``Authorization: Bearer`` header when the token variable is set, else nothing."""
    token = os.environ.get(token_env)
    return {"Authorization": f"Bearer {token}"} if token else {}


def _log_rejection(action: str, endpoint: str, status_code: int, token_env: str) -> None:
    """Auth failures and redirects are configuration errors, so they warn; anything else stays at DEBUG.

    Redirects matter because a collector behind a login page (an SSO proxy, for
    example) answers every request with a 302 to the sign-in page. Following it
    would land on an HTML page with HTTP 200 and look like success.
    """
    if 300 <= status_code < 400:
        logger.warning(
            "%s to %s was redirected (HTTP %d); the endpoint is behind a login page or proxy, nothing was recorded",
            action,
            endpoint,
            status_code,
        )
    elif status_code in (401, 403):
        logger.warning(
            "%s to %s rejected (HTTP %d); check the bearer token in $%s", action, endpoint, status_code, token_env
        )
    else:
        logger.debug("%s to %s failed: HTTP %d", action, endpoint, status_code)


def _cluster_setting() -> str | None:
    """The cluster name from srtslurm.yaml, or None; never raises (reporting must not break a run)."""
    try:
        from srtctl.core.config import get_srtslurm_setting  # lazy: core.config is heavier than this module

        value = get_srtslurm_setting("cluster")
    except Exception:  # noqa: BLE001 - any config problem just means "unknown cluster"
        return None
    return str(value) if value else None


def _resolve_endpoints(status: "ReportingStatusConfig | None") -> tuple[str, ...]:
    """Merge endpoint + endpoints into a deduplicated tuple with trailing slashes stripped.

    Args:
        status: ReportingStatusConfig (may be None)

    Returns:
        Tuple of unique endpoint URLs
    """
    if not status:
        return ()

    seen: dict[str, None] = {}
    if status.endpoint:
        seen[status.endpoint.rstrip("/")] = None
    if status.endpoints:
        for ep in status.endpoints:
            seen[ep.rstrip("/")] = None
    return tuple(seen)


@dataclass(frozen=True)
class StatusReporter:
    """Fire-and-forget status reporter.

    Reports job status to one or more external APIs if reporting.status endpoints
    are configured. All operations are non-blocking and failures are silently logged.

    Usage:
        reporter = StatusReporter.from_config(config.reporting, job_id="12345")
        reporter.report(JobStatus.WORKERS, stage=JobStage.WORKERS)
    """

    job_id: str
    api_endpoints: tuple[str, ...] = ()
    timeout: float = 5.0
    # Environment variable holding the bearer token; see DEFAULT_TOKEN_ENV.
    token_env: str = DEFAULT_TOKEN_ENV
    # Connection attempts per endpoint for each report; see _put.
    attempts: int = 2

    @classmethod
    def from_config(cls, reporting: "ReportingConfig | None", job_id: str) -> "StatusReporter":
        """Create reporter from reporting config.

        Args:
            reporting: ReportingConfig from srtslurm.yaml or recipe
            job_id: SLURM job ID

        Returns:
            StatusReporter instance (disabled if no endpoints configured)
        """
        status = reporting.status if reporting else None
        endpoints = _resolve_endpoints(status)
        if endpoints:
            logger.info("Status reporting enabled: %s", ", ".join(endpoints))

        return cls(job_id=job_id, api_endpoints=endpoints, token_env=_token_env(status))

    @property
    def enabled(self) -> bool:
        """Check if reporting is enabled."""
        return len(self.api_endpoints) > 0

    def _now_iso(self) -> str:
        """Get current UTC time in ISO8601 format."""
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def _put(self, payload: dict) -> bool:
        """Send PUT to all endpoints, up to ``attempts`` tries each. Returns True if any succeeded.

        A lost PUT is a lost event in the collector, and a flaky egress path can
        eat one attempt's whole connect timeout on a dead address, so one retry
        buys a lot. The final failure is a WARNING in the sweep log; the run itself
        is never affected.
        """
        cluster = _cluster_setting()
        if cluster:
            payload = {**payload, "metadata": {**(payload.get("metadata") or {}), "cluster": cluster}}
        any_success = False
        headers = _auth_headers(self.token_env)
        for endpoint in self.api_endpoints:
            url = f"{endpoint}/api/jobs/{self.job_id}"
            for attempt in range(1, self.attempts + 1):
                try:
                    # allow_redirects=False: a 3xx is a failure, never something to follow (see _log_rejection).
                    response = requests.put(
                        url, json=payload, headers=headers, timeout=self.timeout, allow_redirects=False
                    )
                except requests.exceptions.RequestException as e:
                    if attempt < self.attempts:
                        logger.debug(
                            "Status report to %s failed (attempt %d/%d): %s", endpoint, attempt, self.attempts, e
                        )
                        continue
                    logger.warning("Status report to %s lost after %d attempts: %s", endpoint, self.attempts, e)
                    break
                if response.status_code == 200:
                    logger.debug("Status reported to %s", endpoint)
                    any_success = True
                else:
                    _log_rejection("Status report", endpoint, response.status_code, self.token_env)
                break
        return any_success

    def report(
        self,
        status: JobStatus,
        stage: JobStage | None = None,
        message: str | None = None,
    ) -> bool:
        """Report status update (fire-and-forget).

        Args:
            status: New job status
            stage: Current execution stage
            message: Optional human-readable message

        Returns:
            True if reported to at least one endpoint, False otherwise
        """
        if not self.enabled:
            return False

        payload = JobUpdatePayload(
            status=status.value,
            updated_at=self._now_iso(),
            stage=stage.value if stage else None,
            message=message,
        )

        return self._put(payload.model_dump(exclude_none=True))

    def report_started(
        self,
        config: "SrtConfig",
        runtime: "RuntimeContext",
        resource_snapshot: dict | None = None,
    ) -> bool:
        """Report job started with initial metadata.

        Args:
            config: Job configuration
            runtime: Runtime context with computed values

        Returns:
            True if reported to at least one endpoint, False otherwise
        """
        if not self.enabled:
            return False

        resource_snapshot = resource_snapshot or {}
        metadata = {
            "model": {
                "path": str(config.model.path),
                "precision": config.model.precision,
            },
            "resources": {
                "gpu_type": config.resources.gpu_type,
                "gpus_per_node": config.resources.gpus_per_node,
                "prefill_workers": config.topology.num_prefill,
                "decode_workers": config.topology.num_decode,
                "agg_workers": config.topology.num_agg,
                "cpu_allocation": resource_snapshot.get("cpus"),
                "cpu_check": resource_snapshot.get("cpu_check"),
            },
            "benchmark": {
                "type": config.benchmark.type,
            },
            "backend_type": config.backend_type,
            "frontend_type": config.frontend.type,
            "head_node": runtime.nodes.head,
            # Where the run writes its logs on the cluster filesystem. A collector
            # on the same filesystem (srtctl status-server on a login node) can
            # open them directly; logs_url only appears later if reporting.s3 is set.
            "log_dir": str(runtime.log_dir),
            # Identity, repeated from the submit-time POST. That POST is one attempt
            # from the login node; when it is lost (a flaky egress path), the
            # collector's placeholder row takes its name and cluster from here.
            "job_name": getattr(config, "name", None),
            "cluster": _cluster_setting(),
        }

        payload = JobUpdatePayload(
            status=JobStatus.STARTING.value,
            stage=JobStage.STARTING.value,
            message=f"Job started on {runtime.nodes.head}",
            started_at=self._now_iso(),
            updated_at=self._now_iso(),
            metadata=metadata,
        )

        return self._put(payload.model_dump(exclude_none=True))

    def report_completed(self, exit_code: int, logs_url: str | None = None) -> bool:
        """Report job completed with exit code and optional logs pointer.

        Benchmark results themselves are intentionally not carried in this PUT.
        S3 is the source of truth for artifacts; the collector stores the
        pointer (``logs_url``) and consumers fetch the full rollup from S3.

        Args:
            exit_code: Process exit code (0 = success)
            logs_url: URL where logs were uploaded (e.g. s3://bucket/prefix/job/)

        Returns:
            True if reported to at least one endpoint, False otherwise
        """
        if not self.enabled:
            return False

        status = JobStatus.COMPLETED if exit_code == 0 else JobStatus.FAILED
        message = "Benchmark completed successfully" if exit_code == 0 else f"Job failed with exit code {exit_code}"

        payload = JobUpdatePayload(
            status=status.value,
            stage=JobStage.CLEANUP.value,
            message=message,
            completed_at=self._now_iso(),
            updated_at=self._now_iso(),
            exit_code=exit_code,
            logs_url=logs_url,
        )

        return self._put(payload.model_dump(exclude_none=True))

    def report_artifacts(self, logs_url: str) -> bool:
        """Push an artifacts pointer (``logs_url``) eagerly, mid-run.

        Used after S3 sync completes but before later stages (AI analysis,
        cleanup) that can hang or fail. Keeps the status value at its current
        stage (``benchmark``) so this PUT is purely an artifact-pointer update,
        not a lifecycle transition. The collector merges ``logs_url`` in; a
        later ``report_completed`` will reassert it idempotently.

        Args:
            logs_url: URL where logs are being / have been uploaded

        Returns:
            True if reported to at least one endpoint, False otherwise
        """
        if not self.enabled or not logs_url:
            return False

        payload = JobUpdatePayload(
            status=JobStatus.BENCHMARK.value,
            stage=JobStage.CLEANUP.value,
            message="Artifacts uploaded",
            updated_at=self._now_iso(),
            logs_url=logs_url,
        )

        return self._put(payload.model_dump(exclude_none=True))


def create_job_record(
    reporting: "ReportingConfig | None",
    job_id: str,
    job_name: str,
    cluster: str | None = None,
    recipe: str | None = None,
    metadata: dict | None = None,
    attempts: int = 2,
    submitted_at: str | None = None,
) -> bool:
    """Create initial job record in status APIs (called at submission time).

    This is a standalone function used by submit.py before the job starts.
    Sends to all configured endpoints. Each endpoint gets up to ``attempts``
    tries: the login node's path to a collector can be flaky (a dead address
    behind a round-robin name eats the whole connect timeout), and this POST is
    the only message carrying the job name, cluster and recipe. A final failure
    is a WARNING because the person running ``srtctl apply`` is right there;
    the run still shows up once it starts, named from ``report_started``.

    Args:
        reporting: ReportingConfig from srtslurm.yaml or recipe
        job_id: SLURM job ID
        job_name: Job/config name
        cluster: Cluster name (optional)
        recipe: Path to recipe file (optional)
        metadata: Job metadata dict (may include "tags" list)
        attempts: Connection attempts per endpoint before giving up
        submitted_at: ISO 8601 submit time; defaults to now. Pass the real time when
            re-posting a job after the fact, the collector only ever moves it earlier.

    Returns:
        True if created on at least one endpoint, False otherwise
    """
    status = reporting.status if reporting else None
    endpoints = _resolve_endpoints(status)
    if not endpoints:
        return False
    token_env = _token_env(status)
    headers = _auth_headers(token_env)

    payload = JobCreatePayload(
        job_id=job_id,
        job_name=job_name,
        submitted_at=submitted_at or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        cluster=cluster,
        recipe=recipe,
        metadata=metadata,
    )
    payload_dict = payload.model_dump(exclude_none=True)

    any_success = False
    for endpoint in endpoints:
        url = f"{endpoint}/api/jobs"
        for attempt in range(1, attempts + 1):
            try:
                response = requests.post(url, json=payload_dict, headers=headers, timeout=5.0, allow_redirects=False)
            except requests.exceptions.RequestException as e:
                if attempt < attempts:
                    logger.debug("Job record creation on %s failed (attempt %d/%d): %s", endpoint, attempt, attempts, e)
                    continue
                logger.warning(
                    "Job record creation on %s failed after %d attempts: %s. The run still appears in the collector "
                    "once it starts; its name and cluster come from the started report.",
                    endpoint,
                    attempts,
                    e,
                )
                break
            if response.status_code == 201:
                logger.debug("Job record created on %s: %s", endpoint, job_id)
                any_success = True
            else:
                _log_rejection("Job record creation", endpoint, response.status_code, token_env)
            break

    return any_success


# Append-only logs and structured output; contents are interpreted by the API.
STREAM_SUFFIXES = (".out", ".err", ".log", ".csv", ".jsonl")
STREAM_REQUEST_BYTES = 1 << 20
STREAM_SHUTDOWN_SECONDS = 5.0
# Sibling of the log directory, so neither the S3 sync nor dsight's capture
# discovery sees segments that final.parquet already contains.
TACHOMETER_OUTBOX = "tachometer-outbox"
_SEGMENT_NAME = re.compile(r"out-(\d+)\.parquet")


def log_stream_interval(reporting: "ReportingConfig | None") -> float | None:
    """Seconds between live uploads, or None when streaming is off or has no endpoint."""
    status = reporting.status if reporting else None
    if status is None or not _resolve_endpoints(status):
        return None
    return status.logging_stream_interval


def tachometer_outbox(log_dir: Path) -> Path:
    """Where Tachometer links sealed segments for the streamer to upload and unlink."""
    return log_dir.parent / TACHOMETER_OUTBOX


def _segment_order(path: Path) -> tuple[int, str]:
    match = _SEGMENT_NAME.fullmatch(path.name)
    return (int(match[1]) if match else sys.maxsize, path.name)


@dataclass
class _SegmentUpload:
    source: BinaryIO
    total: int
    offset: int = 0
    pending: bytes | None = None


class LogStreamer:
    """Upload raw log deltas and sealed Tachometer segments; the collector does all decoding.

    Segments in ``outbox_dir`` are immutable, so each is sent exactly once per
    endpoint and unlinked after every endpoint has acknowledged it. Memory is
    bounded to one chunk per pending file.
    """

    def __init__(self, reporter: StatusReporter, log_dir: Path, interval: float, outbox_dir: Path | None = None):
        self.reporter = reporter
        self.log_dir = log_dir
        self.interval = interval
        self.outbox_dir = outbox_dir
        self._offsets: dict[tuple[str, str], int] = {}
        self._pending: dict[tuple[str, str], bytes] = {}
        self._finalized: set[tuple[str, str]] = set()
        self._segments: dict[tuple[str, str], _SegmentUpload] = {}
        self._delivered: dict[str, set[str]] = {}
        self._cluster = _cluster_setting()
        self._session = requests.Session()
        self._shutdown_deadline: float | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="status-log-stream", daemon=True)

    @classmethod
    def from_config(
        cls,
        reporting: "ReportingConfig | None",
        reporter: StatusReporter,
        log_dir: Path,
        outbox_dir: Path | None = None,
    ) -> "LogStreamer | None":
        interval = log_stream_interval(reporting)
        if not reporter.enabled or interval is None:
            return None
        return cls(reporter, log_dir, interval, outbox_dir)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        """Give the worker a bounded final flush; telemetry cannot delay completion indefinitely."""
        self._shutdown_deadline = time.monotonic() + STREAM_SHUTDOWN_SECONDS
        self._stop.set()
        if self._thread.ident is not None:
            self._thread.join(timeout=STREAM_SHUTDOWN_SECONDS)
            if self._thread.is_alive():
                logger.warning(
                    "Log streaming did not finish within %.1fs; local logs are retained", STREAM_SHUTDOWN_SECONDS
                )

    def _run(self) -> None:
        try:
            while not self._stop.wait(self.interval):
                self.flush()
            self.flush()
        finally:
            for upload in self._segments.values():
                upload.source.close()
            self._segments.clear()
            self._session.close()

    def _expired(self) -> bool:
        return self._shutdown_deadline is not None and time.monotonic() >= self._shutdown_deadline

    def flush(self) -> None:
        """Upload finite file snapshots, retaining failed chunks verbatim for retry."""
        try:
            if self._expired():
                return
            files = sorted(p for p in self.log_dir.rglob("*") if p.suffix in STREAM_SUFFIXES and p.is_file())
            segments = sorted(self.outbox_dir.glob("out-*.parquet"), key=_segment_order) if self.outbox_dir else []
            for endpoint in self.reporter.api_endpoints:
                if self._flush_logs(endpoint, files):
                    self._flush_segments(endpoint, segments)
            self._release_segments(segments)
        except Exception:
            logger.warning("Log streaming flush failed; will retry", exc_info=True)

    def _flush_logs(self, endpoint: str, files: list[Path]) -> bool:
        for path in files:
            if self._expired():
                return False
            rel = path.relative_to(self.log_dir).as_posix()
            key = (endpoint, rel)
            try:
                with path.open("rb") as source:
                    end = os.fstat(source.fileno()).st_size
                    while not self._expired():
                        offset = self._offsets.get(key, 0)
                        data = self._pending.get(key)
                        if data is None:
                            if offset >= end:
                                if self._stop.is_set() and key not in self._finalized:
                                    ack = self._post(
                                        endpoint, "logs", {"file": rel, "offset": str(offset), "final": "1"}, b""
                                    )
                                    if ack is None or ack.get("stored") not in (0, 1):
                                        return False
                                    self._finalized.add(key)
                                break
                            source.seek(offset)
                            data = source.read(min(STREAM_REQUEST_BYTES, end - offset))
                            if not data:
                                break
                            self._pending[key] = data
                        params = {"file": rel, "offset": str(offset)}
                        if self._stop.is_set() and offset + len(data) == end:
                            params["final"] = "1"
                        ack = self._post(endpoint, "logs", params, data)
                        if ack is None or ack.get("stored") not in (0, 1):
                            return False
                        self._offsets[key] = offset + len(data)
                        del self._pending[key]
                        if params.get("final") == "1":
                            self._finalized.add(key)
            except OSError as exc:
                logger.debug("Log stream skipped %s: %s", rel, exc)
        return True

    def _flush_segments(self, endpoint: str, segments: list[Path]) -> None:
        for path in segments:
            if endpoint in self._delivered.get(path.name, ()):
                continue
            key = (endpoint, path.name)
            try:
                upload = self._segments.get(key)
                if upload is None:
                    source = path.open("rb")
                    upload = _SegmentUpload(source, os.fstat(source.fileno()).st_size)
                    self._segments[key] = upload
                while upload.offset < upload.total:
                    if self._expired():
                        return
                    if upload.pending is None:
                        upload.pending = upload.source.read(min(STREAM_REQUEST_BYTES, upload.total - upload.offset))
                    if not upload.pending:
                        raise OSError(f"{path.name} shrank while uploading")
                    params = {"file": path.name, "offset": str(upload.offset), "total": str(upload.total)}
                    ack = self._post(endpoint, "captures", params, upload.pending)
                    next_offset = upload.offset + len(upload.pending)
                    if (
                        ack is None
                        or ack.get("next_offset") != next_offset
                        or ack.get("complete") is not (next_offset == upload.total)
                    ):
                        return
                    upload.offset, upload.pending = next_offset, None
                upload.source.close()
                del self._segments[key]
                self._delivered.setdefault(path.name, set()).add(endpoint)
            except OSError as exc:
                logger.debug("Segment upload skipped %s: %s", path.name, exc)
                if key in self._segments:
                    self._segments.pop(key).source.close()

    def _release_segments(self, segments: list[Path]) -> None:
        """Unlink segments every endpoint holds; the rest wait for the next flush."""
        endpoints = set(self.reporter.api_endpoints)
        if not endpoints:
            return
        for path in segments:
            if self._delivered.get(path.name, set()) >= endpoints:
                path.unlink(missing_ok=True)
                del self._delivered[path.name]

    def _post(self, endpoint: str, route: str, params: dict[str, str], data: bytes) -> dict | None:
        if self._cluster:
            params["cluster"] = self._cluster
        timeout = self.reporter.timeout
        if self._shutdown_deadline is not None:
            timeout = min(timeout, self._shutdown_deadline - time.monotonic())
            if timeout <= 0:
                return None
        try:
            response = self._session.post(
                f"{endpoint}/api/jobs/{self.reporter.job_id}/{route}",
                params=params,
                data=data,
                headers={**_auth_headers(self.reporter.token_env), "Content-Type": "application/octet-stream"},
                timeout=timeout,
                allow_redirects=False,
            )
            if response.status_code != 200:
                _log_rejection("Log stream", endpoint, response.status_code, self.reporter.token_env)
                return None
            ack = response.json()
            if isinstance(ack, dict) and ack.get("job_id") == self.reporter.job_id:
                return ack
        except (requests.exceptions.RequestException, ValueError) as exc:
            logger.debug("Log stream to %s failed: %s", endpoint, exc)
        return None
