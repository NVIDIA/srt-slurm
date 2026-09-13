# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Head-node CPU power collector.

Best-effort and fully decoupled from the GPU power pipeline's lifecycle: no
failure here — an unresolvable node, an unreachable exporter, a malformed
scrape, a wedged collector thread — is ever raised into the caller. The only
durable output is ``cpu/samples.csv`` and a non-authoritative
``cpu_manifest.json`` written at teardown.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import requests

from srtctl.core.power.contract import CPU_MANIFEST_FILENAME, CPU_SAMPLES_FILENAME, atomic_write_json
from srtctl.core.power.cpu_parser import parse_cpu_scrape
from srtctl.core.power.cpu_samples import CpuSampleRow, CpuSampleWriter
from srtctl.core.power.session import _run_daemon_workers
from srtctl.core.processes import ManagedProcess
from srtctl.core.slurm import get_hostname_ip

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CpuPowerEndpoint:
    hostname: str
    url: str


@dataclass(frozen=True)
class CpuPowerSessionSettings:
    power_dir: Path
    sample_interval_seconds: float
    request_timeout_seconds: float
    collector_join_timeout_seconds: float
    exporter_port: int
    network_interface: str | None = None
    producer_git_commit: str | None = None


@dataclass
class _NodeStats:
    resolved_mode: str = "unknown"
    scrape_count: int = 0
    error_count: int = 0


class CpuPowerCollector:
    """Best-effort head-node collector for ``cpu-power-exporter``."""

    def __init__(self, *, settings: CpuPowerSessionSettings, nodes: Sequence[str]):
        self._settings = settings
        self._nodes = list(nodes)
        self._endpoints: list[CpuPowerEndpoint] = []
        self._writer_lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._exporters_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._writer: CpuSampleWriter | None = None
        self._exporters: list[ManagedProcess] = []
        self._started_at_unix: float | None = None
        self._stats: dict[str, _NodeStats] = {node: _NodeStats() for node in self._nodes}

    @property
    def samples_path(self) -> Path:
        return self._settings.power_dir / CPU_SAMPLES_FILENAME

    @property
    def manifest_path(self) -> Path:
        return self._settings.power_dir / CPU_MANIFEST_FILENAME

    def add_exporter(self, process: ManagedProcess) -> None:
        """Track an exporter process the registry already owns (unused by finalize; kept for parity/logging)."""
        with self._exporters_lock:
            self._exporters.append(process)

    def start(self) -> None:
        """Resolve endpoints, open the writer, and start the collector thread."""
        try:
            self._writer = CpuSampleWriter(self.samples_path)
        except OSError:
            logger.exception("Failed to open cpu/samples.csv writer; CPU power collection disabled")
            return

        for node in self._nodes:
            try:
                ip = get_hostname_ip(node, self._settings.network_interface)
            except Exception:
                logger.warning("CPU power endpoint resolution failed for %s", node, exc_info=True)
                continue
            if not ip:
                continue
            self._endpoints.append(
                CpuPowerEndpoint(hostname=node, url=f"http://{ip}:{self._settings.exporter_port}/metrics")
            )

        self._started_at_unix = time.time()
        self._thread = threading.Thread(target=self._run, name="CpuPowerCollector", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        interval = self._settings.sample_interval_seconds
        next_cycle = time.monotonic()
        while not self._stop.is_set():
            try:
                self._collect_once()
            except Exception:
                logger.exception("CPU power collector cycle failed; continuing")
            next_cycle += interval
            self._stop.wait(max(0.0, next_cycle - time.monotonic()))

    def _collect_once(self) -> None:
        endpoints = list(self._endpoints)
        if not endpoints:
            return
        deadline = time.monotonic() + 2 * self._settings.request_timeout_seconds + 1.0
        results, failures = _run_daemon_workers(
            [(f"CpuPowerScrape-{endpoint.hostname}", self._poll, endpoint) for endpoint in endpoints],
            deadline=deadline,
        )
        for failure in failures:
            logger.warning("CPU power scrape worker raised: %s", failure)

        rows: list[CpuSampleRow] = []
        for hostname, mode, node_rows in results:
            rows.extend(node_rows)
            with self._stats_lock:
                stats = self._stats.setdefault(hostname, _NodeStats())
                stats.scrape_count += 1
                if mode != "unknown":
                    stats.resolved_mode = mode
                if not node_rows:
                    stats.error_count += 1

        with self._writer_lock:
            if self._writer is not None:
                self._writer.append(rows)
                self._writer.flush()

    def _poll(self, endpoint: CpuPowerEndpoint) -> tuple[str, str, list[CpuSampleRow]]:
        try:
            response = requests.get(endpoint.url, timeout=self._settings.request_timeout_seconds)
            response.raise_for_status()
            body = response.text
        except requests.RequestException as exc:
            logger.debug("CPU power scrape failed for %s: %s", endpoint.hostname, exc)
            return endpoint.hostname, "unknown", []

        timestamp_unix = time.time()
        scrape = parse_cpu_scrape(body)
        if not scrape.readings:
            return endpoint.hostname, "unknown", []

        rows = [
            CpuSampleRow(
                timestamp_unix=timestamp_unix,
                hostname=endpoint.hostname,
                source=reading.source,
                sensor=reading.sensor,
                socket_id=reading.socket_id,
                power_w=reading.power_w,
                total_power_w=scrape.total_power_w,
            )
            for reading in scrape.readings
        ]
        return endpoint.hostname, scrape.mode, rows

    def stop_and_finalize(self) -> None:
        """Stop the collector, close the writer, and write the provenance manifest."""
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=self._settings.collector_join_timeout_seconds)
            if thread.is_alive():
                logger.warning(
                    "CPU power collector thread did not stop within %.1fs",
                    self._settings.collector_join_timeout_seconds,
                )

        with self._writer_lock:
            if self._writer is not None:
                try:
                    self._writer.close()
                except OSError:
                    logger.exception("Failed to close cpu/samples.csv writer")

        self._write_manifest()

    def _write_manifest(self) -> None:
        with self._stats_lock:
            nodes = {
                hostname: {
                    "resolved_mode": stats.resolved_mode,
                    "scrape_count": stats.scrape_count,
                    "error_count": stats.error_count,
                }
                for hostname, stats in sorted(self._stats.items())
            }
        payload = {
            "schema_version": 1,
            "producer": "srt-slurm.cpu-power",
            "producer_git_commit": self._settings.producer_git_commit,
            "started_at_unix": self._started_at_unix,
            "stopped_at_unix": time.time(),
            "nodes": nodes,
        }
        try:
            atomic_write_json(self.manifest_path, payload)
        except OSError:
            logger.exception("Failed to write cpu_manifest.json")
