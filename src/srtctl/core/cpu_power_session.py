# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lifecycle and artifact aggregation for host CPU-power collectors."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from srtctl.core.cpu_power import SAMPLES_HEADER
from srtctl.core.node_samples import create_capture_dirs, read_node_markers, read_node_metadata, stop_and_merge
from srtctl.core.power.contract import atomic_write_json
from srtctl.core.power.session import SessionOutcome
from srtctl.core.processes import ManagedProcess


@dataclass(frozen=True)
class CpuPowerSessionSettings:
    """CPU telemetry settings resolved for one run."""

    cpu_dir: Path
    job_id: str
    run_name: str
    nodes: tuple[str, ...]
    source: str
    sample_interval_seconds: float
    startup_timeout_seconds: float
    required: bool
    producer_git_commit: str | None = None


class CpuPowerTelemetrySession:
    """Own readiness/finalization while ProcessRegistry retains process cleanup."""

    def __init__(self, settings: CpuPowerSessionSettings) -> None:
        self.settings = settings
        self.samples_dir = settings.cpu_dir / "nodes"
        self.ready_dir = settings.cpu_dir / "ready"
        self.samples_path = settings.cpu_dir / "samples.csv"
        self.manifest_path = settings.cpu_dir / "manifest.json"
        self._processes: list[ManagedProcess] = []
        self._ready_nodes: tuple[str, ...] = ()

    def initialize(self) -> None:
        create_capture_dirs(self.samples_dir, self.ready_dir)
        self._write_manifest(status="starting", reasons=(), row_count=0, observed_nodes=())

    def add_process(self, process: ManagedProcess) -> None:
        self._processes.append(process)

    def wait_for_readiness(self) -> bool:
        deadline = time.monotonic() + self.settings.startup_timeout_seconds
        expected = set(self.settings.nodes)
        while time.monotonic() < deadline:
            markers = read_node_markers(self.ready_dir)
            if expected <= markers.ready:
                self._ready_nodes = tuple(sorted(markers.ready))
                self._write_manifest(status="running", reasons=(), row_count=0, observed_nodes=self._ready_nodes)
                return True
            if markers.errors or any(process.exit_code not in (None, 0) for process in self._processes):
                break
            time.sleep(0.1)
        markers = read_node_markers(self.ready_dir)
        self._ready_nodes = tuple(sorted(markers.ready))
        reason = "cpu_power_source_unavailable" if markers.errors else "cpu_power_startup_timeout"
        self._write_manifest(status="failed", reasons=(reason,), row_count=0, observed_nodes=self._ready_nodes)
        return False

    def stop_and_finalize(self, *, interrupted: bool = False) -> SessionOutcome:
        capture = stop_and_merge(
            self._processes,
            self.settings.nodes,
            self.samples_dir,
            SAMPLES_HEADER,
            self.samples_path,
            sort_key=lambda row: (float(row[1]), row[3], row[5]),
        )
        merged = capture.merged
        reasons: list[str] = []
        if interrupted:
            reasons.append("interrupted")
        if merged.malformed:
            reasons.append("cpu_samples_malformed")
        missing = capture.missing_nodes
        if missing:
            reasons.append("cpu_node_samples_missing")
        if not merged.row_count:
            reasons.append("cpu_samples_empty")
        failed_processes = capture.exited_processes
        if failed_processes:
            reasons.append("cpu_collector_failed")
        publication_valid = not reasons
        status = "complete" if publication_valid else "incomplete"
        self._write_manifest(
            status=status,
            reasons=tuple(dict.fromkeys(reasons)),
            row_count=merged.row_count,
            observed_nodes=merged.observed_nodes,
            missing_nodes=missing,
            failed_processes=failed_processes,
        )
        return SessionOutcome(
            status=status,
            publication_valid=publication_valid,
            reason_codes=tuple(dict.fromkeys(reasons)),
            exit_nonzero=self.settings.required and not publication_valid,
        )

    def _write_manifest(
        self,
        *,
        status: str,
        reasons: tuple[str, ...],
        row_count: int,
        observed_nodes: tuple[str, ...],
        missing_nodes: tuple[str, ...] = (),
        failed_processes: tuple[str, ...] = (),
    ) -> None:
        atomic_write_json(
            self.manifest_path,
            {
                "schema_version": 1,
                "job_id": self.settings.job_id,
                "run_name": self.settings.run_name,
                "status": status,
                "required": self.settings.required,
                "requested_source": self.settings.source,
                "sample_interval_seconds": self.settings.sample_interval_seconds,
                "expected_nodes": list(self.settings.nodes),
                "observed_nodes": list(observed_nodes),
                "missing_nodes": list(missing_nodes),
                "sample_row_count": row_count,
                "reason_codes": list(reasons),
                "failed_processes": list(failed_processes),
                "producer_git_commit": self.settings.producer_git_commit,
                "node_metadata": read_node_metadata(self.samples_dir),
                "updated_at_unix": time.time(),
            },
        )
