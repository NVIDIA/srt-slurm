# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Head-node side of the RDMA NIC traffic capture: merge node CSVs and write the manifest.

The artifact names are deliberately not ``samples.csv`` or ``manifest.json``:
the power report and downstream consumers treat any file with those names
below the log directory as power evidence.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from srtctl.core.nic_traffic import SAMPLES_HEADER, SCHEMA_VERSION
from srtctl.core.node_samples import (
    MergedNodeCsvs,
    NodeCapture,
    create_capture_dirs,
    read_node_markers,
    read_node_metadata,
    stop_and_merge,
)
from srtctl.core.power.contract import atomic_write_json
from srtctl.core.processes import ManagedProcess

TRAFFIC_FILENAME = "nic_traffic.csv"
MANIFEST_FILENAME = "nic_manifest.json"

_NOTHING_MERGED = NodeCapture(
    merged=MergedNodeCsvs(row_count=0, observed_nodes=(), malformed=False), missing_nodes=(), exited_processes=()
)


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
        create_capture_dirs(self.samples_dir, self.ready_dir)
        self._write_manifest(status="running")

    def add_process(self, process: ManagedProcess) -> None:
        self._processes.append(process)

    def stop_and_finalize(self, *, interrupted: bool = False) -> NicTrafficOutcome:
        """Stop the samplers, merge their CSVs into ``nic_traffic.csv`` and write the final manifest."""
        capture = stop_and_merge(
            self._processes,
            self.settings.nodes,
            self.samples_dir,
            SAMPLES_HEADER,
            self.traffic_path,
            sort_key=lambda row: (float(row[0]), row[2], row[3], row[4]),
        )
        node_errors = read_node_markers(self.ready_dir).errors
        reasons = tuple(
            reason
            for reason, applies in (
                ("interrupted", interrupted),
                ("nic_counters_unavailable", bool(node_errors)),
                ("nic_node_samples_missing", bool(capture.missing_nodes)),
                ("nic_samples_malformed", capture.merged.malformed),
                ("nic_sampler_exited", bool(capture.exited_processes)),
            )
            if applies
        )
        status = "incomplete" if reasons else "complete"
        self._write_manifest(status=status, reasons=reasons, capture=capture, node_errors=node_errors)
        return NicTrafficOutcome(status=status, reason_codes=reasons)

    def _write_manifest(
        self,
        *,
        status: str,
        reasons: tuple[str, ...] = (),
        capture: NodeCapture = _NOTHING_MERGED,
        node_errors: dict[str, str] | None = None,
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
                "observed_nodes": list(capture.merged.observed_nodes),
                "missing_nodes": list(capture.missing_nodes),
                "node_errors": node_errors or {},
                "sample_row_count": capture.merged.row_count,
                "reason_codes": list(reasons),
                "exited_processes": list(capture.exited_processes),
                "producer_git_commit": self.settings.producer_git_commit,
                "nodes": read_node_metadata(self.samples_dir),
                "updated_at_unix": time.time(),
            },
        )
