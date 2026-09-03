# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lifecycle and artifact aggregation for host CPU-power collectors."""

from __future__ import annotations

import contextlib
import csv
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from srtctl.core.cpu_power import SAMPLES_HEADER
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
        os.umask(0o002)
        self.samples_dir.mkdir(parents=True, exist_ok=True)
        self.ready_dir.mkdir(parents=True, exist_ok=True)
        self._write_manifest(status="starting", reasons=(), row_count=0, observed_nodes=())

    def add_process(self, process: ManagedProcess) -> None:
        self._processes.append(process)

    def wait_for_readiness(self) -> bool:
        deadline = time.monotonic() + self.settings.startup_timeout_seconds
        expected = set(self.settings.nodes)
        while time.monotonic() < deadline:
            ready = {path.name.removesuffix(".ready.json") for path in self.ready_dir.glob("*.ready.json")}
            errors = list(self.ready_dir.glob("*.error.json"))
            if expected <= ready:
                self._ready_nodes = tuple(sorted(ready))
                self._write_manifest(status="running", reasons=(), row_count=0, observed_nodes=self._ready_nodes)
                return True
            if errors or any(process.exit_code not in (None, 0) for process in self._processes):
                break
            time.sleep(0.1)
        ready = {path.name.removesuffix(".ready.json") for path in self.ready_dir.glob("*.ready.json")}
        self._ready_nodes = tuple(sorted(ready))
        reason = (
            "cpu_power_source_unavailable" if list(self.ready_dir.glob("*.error.json")) else "cpu_power_startup_timeout"
        )
        self._write_manifest(status="failed", reasons=(reason,), row_count=0, observed_nodes=self._ready_nodes)
        return False

    def stop_and_finalize(self, *, interrupted: bool = False) -> SessionOutcome:
        prematurely_exited = [process.name for process in self._processes if not process.is_running]
        for process in self._processes:
            process.terminate()
        rows: list[list[str]] = []
        malformed = False
        observed_nodes: set[str] = set()
        for path in sorted(self.samples_dir.glob("*.csv")):
            try:
                with path.open(newline="", encoding="utf-8") as handle:
                    reader = csv.reader(handle)
                    if tuple(next(reader, ())) != SAMPLES_HEADER:
                        malformed = True
                        continue
                    for row in reader:
                        if len(row) != len(SAMPLES_HEADER):
                            malformed = True
                            continue
                        rows.append(row)
                        observed_nodes.add(row[2])
            except (OSError, csv.Error, UnicodeDecodeError):
                malformed = True
        rows.sort(key=lambda row: (float(row[1]), row[2], row[4]))
        with self.samples_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(SAMPLES_HEADER)
            writer.writerows(rows)
        reasons: list[str] = []
        if interrupted:
            reasons.append("interrupted")
        if malformed:
            reasons.append("cpu_samples_malformed")
        missing = sorted(set(self.settings.nodes) - observed_nodes)
        if missing:
            reasons.append("cpu_node_samples_missing")
        if not rows:
            reasons.append("cpu_samples_empty")
        failed_processes = prematurely_exited
        if failed_processes:
            reasons.append("cpu_collector_failed")
        publication_valid = not reasons
        status = "complete" if publication_valid else "incomplete"
        self._write_manifest(
            status=status,
            reasons=tuple(dict.fromkeys(reasons)),
            row_count=len(rows),
            observed_nodes=tuple(sorted(observed_nodes)),
            missing_nodes=tuple(missing),
            failed_processes=tuple(failed_processes),
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
        metadata: dict[str, Any] = {}
        for path in sorted(self.samples_dir.glob("*.metadata.json")):
            with contextlib.suppress(OSError, json.JSONDecodeError):
                metadata[path.name.removesuffix(".metadata.json")] = json.loads(path.read_text())
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
                "node_metadata": metadata,
                "updated_at_unix": time.time(),
            },
        )
