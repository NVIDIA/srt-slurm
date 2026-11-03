# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-concurrency-window RDMA NIC traffic from the merged ``nic_traffic.csv``.

For each benchmark window, each NIC's window is snapped to its nearest sample
on each side, as the power report does. Between consecutive samples the
counter delta over the elapsed time is that interval's rate. An interval where
either counter decreased spans a counter reset (driver reload, port reset) and
is dropped. Over the remaining intervals:

* mean Gbit/s = summed byte deltas / summed interval time;
* peak Gbit/s = the highest single-interval rate.

A node's mean is the sum of its NICs' means. A node's peak sums the rates of
the NIC intervals that end at the same sampler pass (``sample_seq``) and takes
the maximum. Timestamps cannot identify a pass, because each row is stamped
at its own counter reads and a NIC read late in a pass is stamped up to
hundreds of milliseconds after the first.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from srtctl.analysis.power_energy_report import ConcurrencyWindow, discover_concurrency_windows, nearest_sample_index
from srtctl.core.nic_traffic import SAMPLES_HEADER
from srtctl.core.nic_traffic_session import TRAFFIC_FILENAME
from srtctl.core.power.contract import MAX_SAMPLE_GAP_SECONDS

SUMMARY_FILENAME = "nic_traffic_summary.csv"
SUMMARY_HEADER = (
    "concurrency",
    "window_start_unix",
    "window_end_unix",
    "scope",
    "hostname",
    "device",
    "port",
    "netdev",
    "driver",
    "intervals",
    "dropped_intervals",
    "mean_tx_gbps",
    "mean_rx_gbps",
    "peak_tx_gbps",
    "peak_rx_gbps",
)

_BITS_PER_GBIT = 1e9


@dataclass(frozen=True)
class NicSeries:
    """One RDMA port's cumulative byte counters, sorted by sample time."""

    hostname: str
    device: str
    port: str
    netdev: str
    driver: str
    times: np.ndarray
    sample_seqs: np.ndarray
    tx_bytes: np.ndarray
    rx_bytes: np.ndarray


@dataclass(frozen=True)
class WindowTraffic:
    """One NIC's or one node's traffic over one window."""

    intervals: int
    dropped_intervals: int
    mean_tx_gbps: float
    mean_rx_gbps: float
    peak_tx_gbps: float
    peak_rx_gbps: float


@dataclass(frozen=True)
class _NicWindow:
    traffic: WindowTraffic
    # Each valid interval's ending sampler pass and its (tx, rx) rates, for summing across a node's NICs.
    end_seqs: np.ndarray
    tx_gbps: np.ndarray
    rx_gbps: np.ndarray


def load_nic_series(path: Path) -> list[NicSeries]:
    """Group the merged CSV's rows into one series per (host, device, port)."""
    rows: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != SAMPLES_HEADER:
            raise ValueError(f"{path}: header is not {','.join(SAMPLES_HEADER)}")
        for row in reader:
            rows[(row["hostname"], row["device"], row["port"])].append(row)
    series = []
    for (hostname, device, port), group in sorted(rows.items()):
        group.sort(key=lambda row: float(row["timestamp_unix"]))
        series.append(
            NicSeries(
                hostname=hostname,
                device=device,
                port=port,
                netdev=group[-1]["netdev"],
                driver=group[-1]["driver"],
                times=np.array([float(row["timestamp_unix"]) for row in group]),
                sample_seqs=np.array([int(row["sample_seq"]) for row in group], dtype=np.int64),
                tx_bytes=np.array([int(row["tx_bytes"]) for row in group], dtype=np.int64),
                rx_bytes=np.array([int(row["rx_bytes"]) for row in group], dtype=np.int64),
            )
        )
    return series


def _nic_window(series: NicSeries, start: float, end: float) -> _NicWindow | None:
    """The NIC's traffic over ``[start, end]``, or None when samples do not cover the window."""
    first = nearest_sample_index(series.times, start)
    last = nearest_sample_index(series.times, end)
    if (
        last <= first
        or abs(series.times[first] - start) > MAX_SAMPLE_GAP_SECONDS
        or abs(series.times[last] - end) > MAX_SAMPLE_GAP_SECONDS
    ):
        return None
    times = series.times[first : last + 1]
    seconds = np.diff(times)
    tx = np.diff(series.tx_bytes[first : last + 1])
    rx = np.diff(series.rx_bytes[first : last + 1])
    valid = (seconds > 0) & (tx >= 0) & (rx >= 0)
    if not valid.any():
        return None
    seconds, tx, rx = seconds[valid], tx[valid], rx[valid]
    tx_gbps = tx * 8 / seconds / _BITS_PER_GBIT
    rx_gbps = rx * 8 / seconds / _BITS_PER_GBIT
    elapsed = float(seconds.sum())
    return _NicWindow(
        traffic=WindowTraffic(
            intervals=int(valid.sum()),
            dropped_intervals=int((~valid).sum()),
            mean_tx_gbps=float(tx.sum()) * 8 / elapsed / _BITS_PER_GBIT,
            mean_rx_gbps=float(rx.sum()) * 8 / elapsed / _BITS_PER_GBIT,
            peak_tx_gbps=float(tx_gbps.max()),
            peak_rx_gbps=float(rx_gbps.max()),
        ),
        end_seqs=series.sample_seqs[first + 1 : last + 1][valid],
        tx_gbps=tx_gbps,
        rx_gbps=rx_gbps,
    )


def _node_window(nics: list[_NicWindow]) -> WindowTraffic:
    rates: dict[int, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for nic in nics:
        for end_seq, tx, rx in zip(nic.end_seqs.tolist(), nic.tx_gbps.tolist(), nic.rx_gbps.tolist(), strict=True):
            rates[end_seq][0] += tx
            rates[end_seq][1] += rx
    return WindowTraffic(
        intervals=len(rates),
        dropped_intervals=sum(nic.traffic.dropped_intervals for nic in nics),
        mean_tx_gbps=sum(nic.traffic.mean_tx_gbps for nic in nics),
        mean_rx_gbps=sum(nic.traffic.mean_rx_gbps for nic in nics),
        peak_tx_gbps=max(tx for tx, _ in rates.values()),
        peak_rx_gbps=max(rx for _, rx in rates.values()),
    )


@dataclass(frozen=True)
class NicTrafficSummary:
    rows: list[tuple[object, ...]]
    # "<host>/<device>/<port> @ concurrency <n>" for NICs whose samples do not cover a window.
    uncovered: list[str]


def summarize(series: list[NicSeries], windows: list[ConcurrencyWindow]) -> NicTrafficSummary:
    """One ``nic`` row per covered NIC and one ``node`` row per host, per window."""
    rows: list[tuple[object, ...]] = []
    uncovered: list[str] = []
    for window in windows:
        head = (window.concurrency, window.start_unix, window.end_unix)
        by_node: dict[str, list[_NicWindow]] = defaultdict(list)
        for nic in series:
            result = _nic_window(nic, window.start_unix, window.end_unix)
            if result is None:
                uncovered.append(f"{nic.hostname}/{nic.device}/{nic.port} @ concurrency {window.concurrency}")
                continue
            by_node[nic.hostname].append(result)
            rows.append(
                (*head, "nic", nic.hostname, nic.device, nic.port, nic.netdev, nic.driver, *_values(result.traffic))
            )
        for hostname, nics in sorted(by_node.items()):
            rows.append((*head, "node", hostname, "", "", "", "", *_values(_node_window(nics))))
    return NicTrafficSummary(rows=rows, uncovered=uncovered)


def _values(traffic: WindowTraffic) -> tuple[object, ...]:
    return (
        traffic.intervals,
        traffic.dropped_intervals,
        round(traffic.mean_tx_gbps, 6),
        round(traffic.mean_rx_gbps, 6),
        round(traffic.peak_tx_gbps, 6),
        round(traffic.peak_rx_gbps, 6),
    )


def write_run_summary(log_dir: Path, nic_dir: Path) -> NicTrafficSummary:
    """Summarize ``<nic_dir>/nic_traffic.csv`` over the run's windows into ``nic_traffic_summary.csv``.

    Raises:
        PowerReportError: The run has no benchmark windows to summarize over.
    """
    windows = discover_concurrency_windows(log_dir)
    summary = summarize(load_nic_series(nic_dir / TRAFFIC_FILENAME), windows)
    with (nic_dir / SUMMARY_FILENAME).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(SUMMARY_HEADER)
        writer.writerows(summary.rows)
    return summary
