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
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import requests

from srtctl.contract import JobCreatePayload, JobStage, JobStatus, JobUpdatePayload

if TYPE_CHECKING:
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.schema import ReportingConfig, ReportingStatusConfig, SrtConfig

logger = logging.getLogger(__name__)


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
        reporter.report(JobStatus.WORKERS_READY, stage=JobStage.WORKERS)
    """

    job_id: str
    api_endpoints: tuple[str, ...] = ()
    timeout: float = 5.0

    @classmethod
    def from_config(cls, reporting: "ReportingConfig | None", job_id: str) -> "StatusReporter":
        """Create reporter from reporting config.

        Args:
            reporting: ReportingConfig from srtslurm.yaml or recipe
            job_id: SLURM job ID

        Returns:
            StatusReporter instance (disabled if no endpoints configured)
        """
        endpoints = _resolve_endpoints(reporting.status if reporting else None)
        if endpoints:
            logger.info("Status reporting enabled: %s", ", ".join(endpoints))

        return cls(job_id=job_id, api_endpoints=endpoints)

    @property
    def enabled(self) -> bool:
        """Check if reporting is enabled."""
        return len(self.api_endpoints) > 0

    def _now_iso(self) -> str:
        """Get current UTC time in ISO8601 format."""
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def _put(self, payload: dict) -> bool:
        """Send PUT to all endpoints. Returns True if any succeeded."""
        any_success = False
        for endpoint in self.api_endpoints:
            try:
                url = f"{endpoint}/api/jobs/{self.job_id}"
                response = requests.put(url, json=payload, timeout=self.timeout)
                if response.status_code == 200:
                    logger.debug("Status reported to %s", endpoint)
                    any_success = True
                else:
                    logger.debug("Status report to %s failed: HTTP %d", endpoint, response.status_code)
            except requests.exceptions.RequestException as e:
                logger.debug("Status report to %s error (ignored): %s", endpoint, e)
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

    def report_started(self, config: "SrtConfig", runtime: "RuntimeContext") -> bool:
        """Report job started with initial metadata.

        Args:
            config: Job configuration
            runtime: Runtime context with computed values

        Returns:
            True if reported to at least one endpoint, False otherwise
        """
        if not self.enabled:
            return False

        metadata = {
            "model": {
                "path": str(config.model.path),
                "precision": config.model.precision,
            },
            "resources": {
                "gpu_type": config.resources.gpu_type,
                "gpus_per_node": config.resources.gpus_per_node,
                "prefill_workers": config.resources.num_prefill,
                "decode_workers": config.resources.num_decode,
                "agg_workers": config.resources.num_agg,
            },
            "benchmark": {
                "type": config.benchmark.type,
            },
            "backend_type": config.backend_type,
            "frontend_type": config.frontend.type,
            "head_node": runtime.nodes.head,
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
) -> bool:
    """Create initial job record in status APIs (called at submission time).

    This is a standalone function used by submit.py before the job starts.
    Sends to all configured endpoints.

    Args:
        reporting: ReportingConfig from srtslurm.yaml or recipe
        job_id: SLURM job ID
        job_name: Job/config name
        cluster: Cluster name (optional)
        recipe: Path to recipe file (optional)
        metadata: Job metadata dict (may include "tags" list)

    Returns:
        True if created on at least one endpoint, False otherwise
    """
    endpoints = _resolve_endpoints(reporting.status if reporting else None)
    if not endpoints:
        return False

    payload = JobCreatePayload(
        job_id=job_id,
        job_name=job_name,
        submitted_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        cluster=cluster,
        recipe=recipe,
        metadata=metadata,
    )
    payload_dict = payload.model_dump(exclude_none=True)

    any_success = False
    for endpoint in endpoints:
        try:
            url = f"{endpoint}/api/jobs"
            response = requests.post(url, json=payload_dict, timeout=5.0)

            if response.status_code == 201:
                logger.debug("Job record created on %s: %s", endpoint, job_id)
                any_success = True
            else:
                logger.debug("Job record creation on %s failed: HTTP %d", endpoint, response.status_code)

        except requests.exceptions.RequestException as e:
            logger.debug("Job record creation on %s error (ignored): %s", endpoint, e)

    return any_success
