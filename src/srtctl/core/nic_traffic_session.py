# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Head-node side of the RDMA NIC traffic capture: merge node CSVs and write the manifest.

The artifact names are deliberately not ``samples.csv`` or ``manifest.json``:
the power report and downstream consumers treat any file with those names
below the log directory as power evidence.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

from srtctl.core.nic_traffic import SAMPLES_HEADER, SCHEMA_VERSION
from srtctl.core.node_samples import merge_node_csvs, read_node_markers, read_node_metadata
from srtctl.core.power.contract import atomic_write_json
from srtctl.core.processes import ManagedProcess

TRAFFIC_FILENAME = "nic_traffic.csv"
MANIFEST_FILENAME = "nic_manifest.json"


@dataclass(frozen=True)
class NicTrafficSessionSettings:
    nic_dir: Path
    job_id: str
    run_name: str
    nodes: tuple[str, ...]
    sample_interval_seconds: float
    producer_git_commit: str | None = None


@dataclass(frozen=True)
class NicTrafficOutcome:
    status: str
    reason_codes: tuple[str, ...]


class NicTrafficSession:
    """Owns the capture directory and its manifest; ``ProcessRegistry`` owns the sampler processes."""

    def __init__(self, settings: NicTrafficSessionSettings) -> None:
        self.settings = settings
        self.samples_dir = settings.nic_dir / "nodes"
        self.ready_dir = settings.nic_dir / "ready"
        self.traffic_path = settings.nic_dir / TRAFFIC_FILENAME
        self.manifest_path = settings.nic_dir / MANIFEST_FILENAME
        self._processes: list[ManagedProcess] = []

    def initialize(self) -> None:
        os.umask(0o002)
        self.samples_dir.mkdir(parents=True, exist_ok=True)
        self.ready_dir.mkdir(parents=True, exist_ok=True)
        self._write_manifest(status="running", reasons=(), row_count=0, observed_nodes=())

    def add_process(self, process: ManagedProcess) -> None:
        self._processes.append(process)

    def stop_and_finalize(self, *, interrupted: bool = False) -> NicTrafficOutcome:
        """Stop the samplers, merge their CSVs into ``nic_traffic.csv`` and write the final manifest."""
        exited_early = [process.name for process in self._processes if not process.is_running]
        for process in self._processes:
            process.terminate()
        merged = merge_node_csvs(
            self.samples_dir,
            SAMPLES_HEADER,
            self.traffic_path,
            sort_key=lambda row: (float(row[0]), row[2], row[3], row[4]),
        )
        markers = read_node_markers(self.ready_dir)
        missing = tuple(sorted(set(self.settings.nodes) - set(merged.observed_nodes)))
        reasons = [
            reason
            for reason, applies in (
                ("interrupted", interrupted),
                ("nic_counters_unavailable", bool(markers.errors)),
                ("nic_node_samples_missing", bool(missing)),
                ("nic_samples_malformed", merged.malformed),
                ("nic_sampler_exited", bool(exited_early)),
            )
            if applies
        ]
        status = "incomplete" if reasons else "complete"
        self._write_manifest(
            status=status,
            reasons=tuple(reasons),
            row_count=merged.row_count,
            observed_nodes=merged.observed_nodes,
            missing_nodes=missing,
            node_errors=markers.errors,
            exited_processes=tuple(exited_early),
        )
        return NicTrafficOutcome(status=status, reason_codes=tuple(reasons))

    def _write_manifest(
        self,
        *,
        status: str,
        reasons: tuple[str, ...],
        row_count: int,
        observed_nodes: tuple[str, ...],
        missing_nodes: tuple[str, ...] = (),
        node_errors: dict[str, str] | None = None,
        exited_processes: tuple[str, ...] = (),
    ) -> None:
        atomic_write_json(
            self.manifest_path,
            {
                "schema_version": SCHEMA_VERSION,
                "job_id": self.settings.job_id,
                "run_name": self.settings.run_name,
                "status": status,
                "traffic_csv": TRAFFIC_FILENAME,
                "sample_interval_seconds": self.settings.sample_interval_seconds,
                "expected_nodes": list(self.settings.nodes),
                "observed_nodes": list(observed_nodes),
                "missing_nodes": list(missing_nodes),
                "node_errors": node_errors or {},
                "sample_row_count": row_count,
                "reason_codes": list(reasons),
                "exited_processes": list(exited_processes),
                "producer_git_commit": self.settings.producer_git_commit,
                "nodes": read_node_metadata(self.samples_dir),
                "updated_at_unix": time.time(),
            },
        )
