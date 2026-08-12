# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``manifest.json``: producer identity, topology, lifecycle state, and validity."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from srtctl import __version__ as PRODUCER_VERSION
from srtctl.core.power.contract import (
    CLOCK_SOURCE,
    POWER_METRIC,
    POWER_SCOPE,
    POWER_UNIT,
    PRODUCER,
    SCHEMA_VERSION,
    dedupe,
)
from srtctl.core.power.samples import ObservedDevice
from srtctl.core.power.topology import ExpectedDevice

STATUS_STARTING = "starting"
STATUS_RUNNING = "running"
STATUS_COMPLETE = "complete"
STATUS_INCOMPLETE = "incomplete"
STATUS_FAILED = "failed"

TERMINAL_STATUSES = (STATUS_COMPLETE, STATUS_INCOMPLETE, STATUS_FAILED)


@dataclass(frozen=True)
class DcgmExporterIdentity:
    """Exactly which exporter image produced the samples.

    ``container_image_sha256`` is ``None`` when the resolved image is not a
    regular file (for example a registry URI pulled at srun time).
    """

    container_image_resolved: str
    container_image_sha256: str | None
    port: int
    command: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "container_image_resolved": self.container_image_resolved,
            "container_image_sha256": self.container_image_sha256,
            "port": self.port,
            "command": self.command,
        }


@dataclass(frozen=True)
class ExpectedWindow:
    """One measured concurrency point the benchmark is expected to record."""

    benchmark_type: str
    concurrency: int

    @property
    def key(self) -> tuple[str, int]:
        return (self.benchmark_type, self.concurrency)

    def to_dict(self) -> dict[str, Any]:
        return {"benchmark_type": self.benchmark_type, "concurrency": self.concurrency}


@dataclass(frozen=True)
class WindowValidation:
    """Structural coverage audit for one expected window."""

    benchmark_type: str
    concurrency: int
    window_file: str | None
    power_coverage_valid: bool
    reason_codes: tuple[str, ...] = ()
    per_device_max_sample_gap_seconds: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark_type": self.benchmark_type,
            "concurrency": self.concurrency,
            "window_file": self.window_file,
            "power_coverage_valid": self.power_coverage_valid,
            "reason_codes": list(self.reason_codes),
            "per_device_max_sample_gap_seconds": dict(self.per_device_max_sample_gap_seconds),
        }


@dataclass(frozen=True)
class ArtifactError:
    """A stale, malformed, duplicate, or unsafe artifact file."""

    path: str
    reason_codes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "reason_codes": list(self.reason_codes)}


@dataclass
class PowerManifest:
    """The orchestrator-owned manifest for one power session."""

    job_id: str
    run_name: str
    sample_interval_seconds: float
    request_timeout_seconds: float
    required: bool
    started_at_unix: float
    dcgm_exporter: DcgmExporterIdentity
    expected_devices: list[ExpectedDevice]
    expected_windows: list[ExpectedWindow]
    producer_git_commit: str | None = None
    status: str = STATUS_STARTING
    stopped_at_unix: float | None = None
    publication_valid: bool | None = None
    observed_devices: list[ObservedDevice] = field(default_factory=list)
    max_scrape_duration_seconds: float | None = None
    scrape_count: int = 0
    sample_row_count: int = 0
    window_validations: list[WindowValidation] = field(default_factory=list)
    artifact_errors: list[ArtifactError] = field(default_factory=list)
    reason_codes: list[str] = field(default_factory=list)
    _terminal_committed: bool = field(default=False, init=False, repr=False)

    def mark_terminal(self, *, status: str, stopped_at_unix: float, publication_valid: bool) -> None:
        """Freeze lifecycle state. Only ``complete`` may ever publish."""
        if self._terminal_committed or self.status in TERMINAL_STATUSES:
            raise RuntimeError(f"manifest is already terminal: {self.status!r}")
        if status not in TERMINAL_STATUSES:
            raise ValueError(f"not a terminal status: {status!r}")
        self.status = status
        self.stopped_at_unix = stopped_at_unix
        self.publication_valid = publication_valid and status == STATUS_COMPLETE
        self._terminal_committed = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "producer": PRODUCER,
            "producer_version": PRODUCER_VERSION,
            "producer_git_commit": self.producer_git_commit,
            "source_metric": POWER_METRIC,
            "unit": POWER_UNIT,
            "power_scope": POWER_SCOPE,
            "timestamp_source": CLOCK_SOURCE,
            "job_id": self.job_id,
            "run_name": self.run_name,
            "sample_interval_seconds": self.sample_interval_seconds,
            "request_timeout_seconds": self.request_timeout_seconds,
            "max_scrape_duration_seconds": self.max_scrape_duration_seconds,
            "required": self.required,
            "started_at_unix": self.started_at_unix,
            "stopped_at_unix": self.stopped_at_unix,
            "status": self.status,
            "publication_valid": self.publication_valid,
            "dcgm_exporter": self.dcgm_exporter.to_dict(),
            "expected_devices": [device.to_dict() for device in self.expected_devices],
            "observed_devices": [device.to_dict() for device in self.observed_devices],
            "expected_windows": [window.to_dict() for window in self.expected_windows],
            "scrape_count": self.scrape_count,
            "sample_row_count": self.sample_row_count,
            "window_validations": [validation.to_dict() for validation in self.window_validations],
            "artifact_errors": [error.to_dict() for error in self.artifact_errors],
            "reason_codes": list(dedupe(self.reason_codes)),
        }
