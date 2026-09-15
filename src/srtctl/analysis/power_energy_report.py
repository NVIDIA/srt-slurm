# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Trapezoidal energy/J-per-token report for a completed srtslurm run.

Reads the CPU (``power/cpu/samples.csv`` from the head-node scraper, or
``cpu_power/samples.csv`` from the host-side collector) and GPU
(``power/samples.csv``) power-telemetry CSVs, joins them against the profiling
window and token counts of each concurrency point in a sa-bench or
aiperf/AgentX sweep, and integrates power into energy with ``numpy.trapz``.

Utilization columns are optional on both CSVs (``gpu_util_pct``/``sm_active``
per GPU; ``cpu_util_*`` per socket from the DCGM host collector). When present
and populated they are summarized as a windowed mean/max next to the joules;
when absent or blank they are simply not reported. Power stays authoritative:
missing utilization coverage is a warning, never an error.

Timestamps are never reconstructed from ``benchmark.out`` log text: aiperf's
``profile_export.jsonl`` already carries ``time.time_ns()`` wall-clock
timestamps per record, and sa-bench's result JSON already carries
``benchmark_start_time_unix``/``benchmark_end_time_unix`` directly (both are
the same epoch-seconds clock as the power CSVs' ``timestamp_unix``, so no
timezone or date-anchoring guesswork is needed). ``benchmark.out`` is only
used to detect which benchmark engine produced the run.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TextIO, TypeVar

import numpy as np

from srtctl.core.cpu_power import UTILIZATION_COLUMNS as CPU_UTILIZATION_COLUMNS
from srtctl.core.power.contract import MAX_SAMPLE_GAP_SECONDS, UTILIZATION_METRICS

GPU_UTILIZATION_COLUMNS = tuple(metric.column for metric in UTILIZATION_METRICS)

# Directory names whose samples.csv is CPU power: "cpu" is the head-node
# scraper's power/cpu/, "cpu_power" is the host-side collector's default
# telemetry.cpu_power.storage_subdir. Anything else is the GPU leg.
CPU_SAMPLES_DIRNAMES = ("cpu", "cpu_power")

_AIPERF_PHASE_RE = re.compile(r"Phase \w+ \(profiling\) (started|complete)")
# Full NOTICE line: time-of-day stamp, then the phase event. "sending complete"
# is deliberately excluded (it is not the phase end). ``elapsed=`` is optional.
_AIPERF_PHASE_LINE_RE = re.compile(
    r"^(?P<stamp>\d{2}:\d{2}:\d{2}\.\d{3}) NOTICE\s+Phase \w+ \(profiling\) (?P<event>started|complete)\b"
    r"(?P<rest>.*)$"
)
_AIPERF_ELAPSED_RE = re.compile(r"elapsed=(?P<seconds>\d+(?:\.\d+)?)s")
_SA_BENCH_MARKERS = ("Serving Benchmark Result", "Successful requests:")
_CONC_DIR_RE = re.compile(r"^conc_(\d+)$")
_RESULT_FILE_RE = re.compile(r"^results_concurrency_(\d+)_")

BENCHMARK_TYPE_AIPERF = "aiperf"
BENCHMARK_TYPE_SA_BENCH = "sa-bench"


class PowerReportError(RuntimeError):
    """Raised for any condition that would otherwise silently corrupt the report."""


# ---------------------------------------------------------------------------
# Benchmark-type detection
# ---------------------------------------------------------------------------


def detect_benchmark_type(benchmark_out: Path) -> str:
    """Classify a run as aiperf/AgentX or sa-bench from its console log.

    Detection only -- never a timestamp source. aiperf's non-TTY NOTICE lines
    carry no date or timezone, and sa-bench's own timing is never printed at
    all (its internal clock is ``time.perf_counter()``, never logged).
    """
    saw_aiperf = False
    saw_sa_bench = False
    with benchmark_out.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if _AIPERF_PHASE_RE.search(line):
                saw_aiperf = True
            elif any(marker in line for marker in _SA_BENCH_MARKERS):
                saw_sa_bench = True
    if saw_aiperf and saw_sa_bench:
        raise PowerReportError(f"{benchmark_out}: matched both aiperf and sa-bench markers")
    if saw_aiperf:
        return BENCHMARK_TYPE_AIPERF
    if saw_sa_bench:
        return BENCHMARK_TYPE_SA_BENCH
    raise PowerReportError(f"{benchmark_out}: matched neither aiperf nor sa-bench markers")


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunPaths:
    benchmark_out: Path
    cpu_samples_csv: Path | None
    gpu_samples_csv: Path | None
    gpu_manifest: Path | None
    concurrency_sources: tuple[tuple[int, Path], ...]


def discover_run(log_dir: Path, *, cpu_samples_csv: Path | None = None) -> RunPaths:
    """Locate the run's artifacts.

    ``cpu_samples_csv`` pins the CPU CSV explicitly; it is required when both CPU
    legs (scraper under ``power/cpu/``, host collector under ``cpu_power/``)
    wrote a ``samples.csv`` for the same run.
    """
    benchmark_out = log_dir / "benchmark.out"
    if not benchmark_out.is_file():
        raise PowerReportError(f"{benchmark_out}: not found")

    all_matches = sorted(log_dir.rglob("samples.csv"))
    cpu_matches = [p for p in all_matches if p.parent.name in CPU_SAMPLES_DIRNAMES]
    gpu_matches = [p for p in all_matches if p.parent.name not in CPU_SAMPLES_DIRNAMES]
    if cpu_samples_csv is not None:
        if not cpu_samples_csv.is_file():
            raise PowerReportError(f"{cpu_samples_csv}: not found")
    elif len(cpu_matches) > 1:
        listing = ", ".join(str(p) for p in cpu_matches)
        raise PowerReportError(
            f"multiple CPU power samples.csv found below {log_dir}: {listing}; pick one with --cpu-samples"
        )
    else:
        cpu_samples_csv = cpu_matches[0] if cpu_matches else None
    if len(gpu_matches) > 1:
        raise PowerReportError(f"multiple GPU power samples.csv found below {log_dir}: {gpu_matches}")
    gpu_samples_csv = gpu_matches[0] if gpu_matches else None
    if cpu_samples_csv is None and gpu_samples_csv is None:
        raise PowerReportError(f"no power samples.csv (CPU or GPU) found below {log_dir}")

    gpu_manifest = None
    if gpu_samples_csv is not None:
        candidate = gpu_samples_csv.with_name("manifest.json")
        gpu_manifest = candidate if candidate.is_file() else None

    benchmark_type = detect_benchmark_type(benchmark_out)
    if benchmark_type == BENCHMARK_TYPE_AIPERF:
        sources = _discover_aiperf_sources(log_dir)
    else:
        sources = _discover_sa_bench_sources(log_dir)
    if not sources:
        raise PowerReportError(f"no {benchmark_type} result artifacts found below {log_dir}")

    return RunPaths(
        benchmark_out=benchmark_out,
        cpu_samples_csv=cpu_samples_csv,
        gpu_samples_csv=gpu_samples_csv,
        gpu_manifest=gpu_manifest,
        concurrency_sources=tuple(sorted(sources)),
    )


def _discover_aiperf_sources(log_dir: Path) -> list[tuple[int, Path]]:
    sources: list[tuple[int, Path]] = []
    for jsonl_path in log_dir.rglob("conc_*/aiperf_artifacts/profile_export.jsonl"):
        conc_dir = jsonl_path.parent.parent
        match = _CONC_DIR_RE.match(conc_dir.name)
        if match is None:
            continue
        sources.append((int(match.group(1)), jsonl_path))
    return sources


def _discover_sa_bench_sources(log_dir: Path) -> list[tuple[int, Path]]:
    # Restricted to sa-bench_*/ result directories (bench.sh:185-189) so this
    # never matches power/windows/results_concurrency_*.json, which shares
    # the same filename but is a different artifact without token fields.
    sources: list[tuple[int, Path]] = []
    for result_path in log_dir.rglob("sa-bench_*/results_concurrency_*.json"):
        match = _RESULT_FILE_RE.match(result_path.name)
        if match is None:
            continue
        sources.append((int(match.group(1)), result_path))
    return sources


# ---------------------------------------------------------------------------
# Per-concurrency window + token extraction
# ---------------------------------------------------------------------------


REPORTED_UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class ReportedTiming:
    """The benchmark's own account of its measurement window.

    Comparison only, never a validation input: the report integrates power
    over the *computed* window (``ConcurrencyWindow.start_unix``/``end_unix``)
    and carries this alongside so a reader can see how far the benchmark's
    self-reported duration sits from the record-derived one.

    ``source`` names where it came from: ``sa-bench-json`` (the result JSON's
    ``duration`` plus its wall-clock start/end), ``aiperf-json`` (the aggregate
    ``benchmark_duration`` plus ``start_time``/``end_time``), ``aiperf-phase-log``
    (the profiling-phase NOTICE lines in ``benchmark.out``; time-of-day only, so
    no absolute start/end), or ``unavailable``.
    """

    source: str
    start_unix: float | None = None
    end_unix: float | None = None
    duration_seconds: float | None = None
    note: str | None = None


REPORTED_TIMING_UNAVAILABLE = ReportedTiming(source=REPORTED_UNAVAILABLE)


@dataclass(frozen=True)
class ConcurrencyWindow:
    benchmark_type: str
    concurrency: int
    start_unix: float
    end_unix: float
    output_tokens: float
    input_tokens: float
    source: Path
    reported: ReportedTiming = REPORTED_TIMING_UNAVAILABLE

    @property
    def duration_seconds(self) -> float:
        return self.end_unix - self.start_unix


def _local_iso_to_unix(stamp: object) -> float | None:
    """aiperf writes naive local-time ISO stamps; interpret them in this host's zone."""
    if not isinstance(stamp, str):
        return None
    try:
        return datetime.fromisoformat(stamp).astimezone().timestamp()
    except ValueError:
        return None


def reported_timing_from_aiperf_json(aggregate: dict) -> ReportedTiming:
    duration = aggregate.get("benchmark_duration")
    duration_seconds = duration.get("avg") if isinstance(duration, dict) else duration
    start_unix = _local_iso_to_unix(aggregate.get("start_time"))
    end_unix = _local_iso_to_unix(aggregate.get("end_time"))
    if duration_seconds is None and start_unix is None and end_unix is None:
        return REPORTED_TIMING_UNAVAILABLE
    return ReportedTiming(
        source="aiperf-json",
        start_unix=start_unix,
        end_unix=end_unix,
        duration_seconds=float(duration_seconds) if duration_seconds is not None else None,
        note="start/end are aiperf's naive local ISO stamps converted in this host's timezone",
    )


def _time_of_day_seconds(stamp: str) -> float:
    """``HH:MM:SS.mmm`` -> seconds since midnight (no date, no timezone)."""
    hours, minutes, seconds = stamp.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def reported_timing_from_phase_log(benchmark_out: Path) -> ReportedTiming:
    """Profiling-phase duration from aiperf's NOTICE lines, when exactly one phase ran.

    The stamps are time-of-day only, so absolute start/end stay None and the
    duration comes from ``elapsed=`` on the complete line, falling back to the
    difference of the two stamps (midnight wrap tolerated). With several
    profiling phases in one log there is no safe way to pair them to a
    concurrency, so the result is ``unavailable`` rather than a guess.
    """
    starts: list[str] = []
    completes: list[tuple[str, str]] = []
    with benchmark_out.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = _AIPERF_PHASE_LINE_RE.match(line.rstrip("\n"))
            if match is None:
                continue
            if match.group("event") == "started":
                starts.append(match.group("stamp"))
            else:
                completes.append((match.group("stamp"), match.group("rest")))
    if not starts or not completes:
        return REPORTED_TIMING_UNAVAILABLE
    if len(starts) != 1 or len(completes) != 1:
        count = max(len(starts), len(completes))
        return ReportedTiming(
            source=REPORTED_UNAVAILABLE,
            note=f"{count} profiling phases in {benchmark_out.name}; cannot pair them to one concurrency",
        )
    start_stamp, (end_stamp, rest) = starts[0], completes[0]
    elapsed = _AIPERF_ELAPSED_RE.search(rest)
    if elapsed is not None:
        duration_seconds = float(elapsed.group("seconds"))
    else:
        duration_seconds = _time_of_day_seconds(end_stamp) - _time_of_day_seconds(start_stamp)
        if duration_seconds < 0:
            duration_seconds += 24 * 3600.0  # the phase crossed midnight
    return ReportedTiming(
        source="aiperf-phase-log",
        duration_seconds=duration_seconds,
        note=f"phase started {start_stamp}, complete {end_stamp} (time-of-day only)",
    )


def aiperf_window(concurrency: int, profile_jsonl: Path, *, benchmark_out: Path | None = None) -> ConcurrencyWindow:
    """Real epoch-second window from ``time.time_ns()`` per-record timestamps.

    Uses only profiling-phase, non-error rows -- the same filter
    ``measurement_window.py`` uses -- so the window reflects actual request
    activity, not the phase's grace-period timeout deadline (which can run
    long after the last real response, as observed in practice).
    """
    starts_ns: list[int] = []
    ends_ns: list[int] = []
    with profile_jsonl.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("error"):
                continue
            metadata = record.get("metadata", {})
            if metadata.get("benchmark_phase") != "profiling":
                continue
            starts_ns.append(metadata["request_start_ns"])
            ends_ns.append(metadata["request_end_ns"])
    if not starts_ns:
        raise PowerReportError(f"{profile_jsonl}: no successful profiling-phase records")

    aggregate_path = profile_jsonl.with_name("profile_export_aiperf.json")
    if not aggregate_path.is_file():
        raise PowerReportError(f"{aggregate_path}: not found (expected alongside {profile_jsonl})")
    aggregate = json.loads(aggregate_path.read_text())
    output_tokens = aggregate["total_osl"]["avg"]
    input_tokens = aggregate["total_isl"]["avg"]

    reported = reported_timing_from_aiperf_json(aggregate)
    if reported.source == REPORTED_UNAVAILABLE and benchmark_out is not None and benchmark_out.is_file():
        reported = reported_timing_from_phase_log(benchmark_out)

    return ConcurrencyWindow(
        benchmark_type=BENCHMARK_TYPE_AIPERF,
        concurrency=concurrency,
        start_unix=min(starts_ns) / 1e9,
        end_unix=max(ends_ns) / 1e9,
        output_tokens=output_tokens,
        input_tokens=input_tokens,
        source=profile_jsonl,
        reported=reported,
    )


def sa_bench_window(concurrency: int, result_json: Path) -> ConcurrencyWindow:
    """Window and tokens read directly from sa-bench's own result JSON.

    Earlier designs derived the start time from the result file's mtime minus
    its reported duration; that heuristic is wrong whenever the run directory
    is copied/archived after the fact (mtime no longer reflects when the
    benchmark ran). The result JSON already carries the true
    ``benchmark_start_time_unix``/``benchmark_end_time_unix`` fields, so read
    those instead.
    """
    result = json.loads(result_json.read_text())
    for key in ("benchmark_start_time_unix", "benchmark_end_time_unix", "total_input_tokens", "total_output_tokens"):
        if key not in result:
            raise PowerReportError(f"{result_json}: missing required field {key!r}")
    duration = result.get("duration")
    return ConcurrencyWindow(
        benchmark_type=BENCHMARK_TYPE_SA_BENCH,
        concurrency=concurrency,
        start_unix=result["benchmark_start_time_unix"],
        end_unix=result["benchmark_end_time_unix"],
        output_tokens=result["total_output_tokens"],
        input_tokens=result["total_input_tokens"],
        source=result_json,
        reported=ReportedTiming(
            source="sa-bench-json",
            start_unix=float(result["benchmark_start_time_unix"]),
            end_unix=float(result["benchmark_end_time_unix"]),
            duration_seconds=float(duration) if duration is not None else None,
            note=None if duration is not None else "result JSON has no 'duration' field",
        ),
    )


def load_concurrency_windows(paths: RunPaths) -> list[ConcurrencyWindow]:
    windows = []
    for concurrency, source in paths.concurrency_sources:
        if source.name == "profile_export.jsonl":
            windows.append(aiperf_window(concurrency, source, benchmark_out=paths.benchmark_out))
        else:
            windows.append(sa_bench_window(concurrency, source))
    return windows


# ---------------------------------------------------------------------------
# Sample loading
# ---------------------------------------------------------------------------


K = TypeVar("K")


def _sorted_series(rows: dict[K, list[tuple[float, float]]]) -> dict[K, tuple[np.ndarray, np.ndarray]]:
    series = {}
    for key, points in rows.items():
        points.sort(key=lambda p: p[0])
        times = np.array([t for t, _ in points], dtype=float)
        watts = np.array([w for _, w in points], dtype=float)
        series[key] = (times, watts)
    return series


# column -> (times, values); only columns that had at least one populated cell appear.
UtilizationSeries = dict[str, tuple[np.ndarray, np.ndarray]]


def _collect_utilization(
    row: dict[str, str], columns: tuple[str, ...], timestamp: float, store: dict[str, list[tuple[float, float]]]
) -> None:
    """Append populated utilization cells by column name; absent columns and blanks are skipped."""
    for column in columns:
        raw = row.get(column)
        if raw is None or raw == "":
            continue
        store.setdefault(column, []).append((timestamp, float(raw)))


def _sorted_utilization(raw: dict[K, dict[str, list[tuple[float, float]]]]) -> dict[K, UtilizationSeries]:
    return {key: _sorted_series(columns) for key, columns in raw.items() if columns}


@dataclass(frozen=True)
class CpuSamples:
    per_socket: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]]
    per_node: dict[str, tuple[np.ndarray, np.ndarray]]
    per_socket_utilization: dict[tuple[str, int], UtilizationSeries] = field(default_factory=dict)


def load_cpu_samples(path: Path) -> CpuSamples:
    with path.open(newline="", encoding="utf-8") as handle:
        return load_cpu_samples_from(handle)


def load_cpu_samples_from(handle: TextIO) -> CpuSamples:
    per_socket: dict[tuple[str, int], list[tuple[float, float]]] = {}
    node_totals: dict[str, dict[float, float]] = {}
    utilization: dict[tuple[str, int], dict[str, list[tuple[float, float]]]] = {}
    for row in csv.DictReader(handle):
        timestamp = float(row["timestamp_unix"])
        hostname = row["hostname"]
        socket_raw = row["socket_id"]
        if socket_raw != "":
            key = (hostname, int(socket_raw))
            per_socket.setdefault(key, []).append((timestamp, float(row["power_w"])))
            _collect_utilization(row, CPU_UTILIZATION_COLUMNS, timestamp, utilization.setdefault(key, {}))
        # total_power_w is blank whenever an ACPI scrape has no `grace` channel
        # (see contract.CPU_SAMPLES_HEADER); skip rather than crash on float("").
        if row["total_power_w"] != "":
            node_totals.setdefault(hostname, {})[timestamp] = float(row["total_power_w"])

    per_node_rows = {host: list(values.items()) for host, values in node_totals.items()}
    return CpuSamples(
        per_socket=_sorted_series(per_socket),
        per_node=_sorted_series(per_node_rows),
        per_socket_utilization=_sorted_utilization(utilization),
    )


@dataclass(frozen=True)
class GpuSamples:
    per_device: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]]
    per_node: dict[str, tuple[np.ndarray, np.ndarray]]
    per_role: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]  # role -> hostname -> series
    per_device_utilization: dict[tuple[str, int], UtilizationSeries] = field(default_factory=dict)
    device_roles: dict[tuple[str, int], set[str]] = field(default_factory=dict)


def load_gpu_roles(manifest_path: Path) -> dict[tuple[str, int], set[str]]:
    manifest = json.loads(manifest_path.read_text())
    roles: dict[tuple[str, int], set[str]] = {}
    for device in manifest.get("expected_devices", []):
        key = (device["hostname"], device["gpu_index"])
        roles[key] = {assignment["worker_role"] for assignment in device["assignments"]}
    return roles


def load_gpu_samples(path: Path, roles: dict[tuple[str, int], set[str]] | None) -> GpuSamples:
    with path.open(newline="", encoding="utf-8") as handle:
        return load_gpu_samples_from(handle, roles)


def load_gpu_samples_from(handle: TextIO, roles: dict[tuple[str, int], set[str]] | None) -> GpuSamples:
    per_device: dict[tuple[str, int], list[tuple[float, float]]] = {}
    node_totals: dict[str, dict[float, float]] = {}
    role_totals: dict[str, dict[str, dict[float, float]]] = {}
    utilization: dict[tuple[str, int], dict[str, list[tuple[float, float]]]] = {}
    for row in csv.DictReader(handle):
        timestamp = float(row["timestamp_unix"])
        hostname = row["hostname"]
        gpu_index = int(row["gpu_index"])
        watts = float(row["power_w"])
        per_device.setdefault((hostname, gpu_index), []).append((timestamp, watts))
        _collect_utilization(row, GPU_UTILIZATION_COLUMNS, timestamp, utilization.setdefault((hostname, gpu_index), {}))
        node_totals.setdefault(hostname, {})
        node_totals[hostname][timestamp] = node_totals[hostname].get(timestamp, 0.0) + watts
        if roles is not None:
            for role in roles.get((hostname, gpu_index), ()):
                by_host = role_totals.setdefault(role, {}).setdefault(hostname, {})
                by_host[timestamp] = by_host.get(timestamp, 0.0) + watts

    per_node_rows = {host: list(values.items()) for host, values in node_totals.items()}
    per_role = {
        role: _sorted_series({host: list(values.items()) for host, values in by_host.items()})
        for role, by_host in role_totals.items()
    }
    return GpuSamples(
        per_device=_sorted_series(per_device),
        per_node=_sorted_series(per_node_rows),
        per_role=per_role,
        per_device_utilization=_sorted_utilization(utilization),
        device_roles=dict(roles) if roles is not None else {},
    )


# ---------------------------------------------------------------------------
# Windowed trapezoidal integration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EnergyBreakdown:
    label: str
    joules: float
    # Time-weighted: joules / window duration. Consistent with the energy figure
    # even when sampling is uneven.
    avg_power_w: float
    # The samples the trapezoid actually spanned, so a reader can compare the
    # power data's edges against the benchmark's own window.
    sample_start_unix: float | None = None
    sample_end_unix: float | None = None
    samples: int = 0
    # Sample-based distribution over those same samples. ``mean_w`` is the
    # plain sample mean; it differs from ``avg_power_w`` when sampling is uneven,
    # and both are kept so that difference is visible. Percentiles use numpy's
    # default linear interpolation. None only on rows built without samples.
    mean_w: float | None = None
    min_w: float | None = None
    p5_w: float | None = None
    p50_w: float | None = None
    p95_w: float | None = None
    p99_w: float | None = None
    max_w: float | None = None


POWER_STAT_FIELDS = ("mean_w", "min_w", "p5_w", "p50_w", "p95_w", "p99_w", "max_w")


def _nearest_index(times: np.ndarray, target: float) -> int:
    idx = int(np.searchsorted(times, target))
    if idx <= 0:
        return 0
    if idx >= len(times):
        return len(times) - 1
    before, after = times[idx - 1], times[idx]
    return idx - 1 if (target - before) <= (after - target) else idx


def windowed_energy(label: str, times: np.ndarray, watts: np.ndarray, start: float, end: float) -> EnergyBreakdown:
    """Snap the window to the nearest sample on each side and trapz between them.

    Never interpolates an exact boundary value -- at the CPU/GPU collectors'
    sub-second sample rate, the piecewise-linear error from snapping instead
    of interpolating is bounded by half a sample interval, negligible next to
    a run measured in minutes. What *does* matter is refusing to integrate
    over a window that isn't actually backed by samples.
    """
    if len(times) == 0:
        raise PowerReportError(f"{label}: no power samples available")
    start_i = _nearest_index(times, start)
    end_i = _nearest_index(times, end)
    start_gap = abs(times[start_i] - start)
    end_gap = abs(times[end_i] - end)
    if start_gap > MAX_SAMPLE_GAP_SECONDS:
        raise PowerReportError(f"{label}: nearest sample to window start is {start_gap:.3f}s away, no coverage")
    if end_gap > MAX_SAMPLE_GAP_SECONDS:
        raise PowerReportError(f"{label}: nearest sample to window end is {end_gap:.3f}s away, no coverage")
    if end_i <= start_i:
        raise PowerReportError(f"{label}: window narrower than the sample spacing")

    inside = watts[start_i : end_i + 1]
    joules = float(np.trapezoid(inside, x=times[start_i : end_i + 1]))
    duration = end - start
    p5, p50, p95, p99 = (float(v) for v in np.percentile(inside, (5, 50, 95, 99)))
    return EnergyBreakdown(
        label=label,
        joules=joules,
        avg_power_w=joules / duration if duration > 0 else 0.0,
        sample_start_unix=float(times[start_i]),
        sample_end_unix=float(times[end_i]),
        samples=int(end_i - start_i + 1),
        mean_w=float(inside.mean()),
        min_w=float(inside.min()),
        p5_w=p5,
        p50_w=p50,
        p95_w=p95,
        p99_w=p99,
        max_w=float(inside.max()),
    )


# ---------------------------------------------------------------------------
# Windowed utilization
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UtilizationSummary:
    label: str
    column: str
    mean: float
    max: float
    samples: int


def windowed_utilization(
    label: str, column: str, times: np.ndarray, values: np.ndarray, start: float, end: float
) -> UtilizationSummary | None:
    """Mean/max of the samples that fall inside ``[start, end]``.

    Utilization is a gauge, not a rate to integrate, so no trapezoid and no
    boundary snapping: only samples actually inside the window count. Returns
    None when the window holds no samples; callers turn that into a warning.
    """
    mask = (times >= start) & (times <= end)
    count = int(mask.sum())
    if count == 0:
        return None
    inside = values[mask]
    return UtilizationSummary(
        label=label, column=column, mean=float(inside.mean()), max=float(inside.max()), samples=count
    )


def _summarize_utilization(
    series_by_key: dict[tuple[str, int], UtilizationSeries],
    *,
    device_label: str,
    group_labels: dict[tuple[str, int], tuple[str, ...]],
    start: float,
    end: float,
    warnings: list[str],
) -> tuple[UtilizationSummary, ...]:
    """Per-device summaries, then equal-weight means of those per group label.

    ``device_label`` formats ``(hostname, index)`` into the per-device label;
    ``group_labels`` maps each device to the node/role labels it rolls up into.
    """
    summaries: list[UtilizationSummary] = []
    grouped: dict[tuple[str, str], list[UtilizationSummary]] = {}
    for key, columns in sorted(series_by_key.items()):
        label = device_label.format(host=key[0], index=key[1])
        for column, (times, values) in sorted(columns.items()):
            summary = windowed_utilization(label, column, times, values, start, end)
            if summary is None:
                warnings.append(f"{label} {column}: no samples inside the window, utilization not reported")
                continue
            summaries.append(summary)
            for group in group_labels.get(key, ()):
                grouped.setdefault((group, column), []).append(summary)
    for (group, column), members in sorted(grouped.items()):
        summaries.append(
            UtilizationSummary(
                label=group,
                column=column,
                mean=float(np.mean([m.mean for m in members])),
                max=max(m.max for m in members),
                samples=sum(m.samples for m in members),
            )
        )
    return tuple(summaries)


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConcurrencyReport:
    window: ConcurrencyWindow
    cpu_per_socket: tuple[EnergyBreakdown, ...] = ()
    cpu_per_node: tuple[EnergyBreakdown, ...] = ()
    cpu_total_joules: float = 0.0
    gpu_per_device: tuple[EnergyBreakdown, ...] = ()
    gpu_per_role: tuple[EnergyBreakdown, ...] = ()
    gpu_per_node: tuple[EnergyBreakdown, ...] = ()
    gpu_total_joules: float = 0.0
    cpu_utilization: tuple[UtilizationSummary, ...] = ()
    gpu_utilization: tuple[UtilizationSummary, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def combined_total_joules(self) -> float:
        return self.cpu_total_joules + self.gpu_total_joules

    def joules_per_output_token(self) -> float | None:
        return self.combined_total_joules / self.window.output_tokens if self.window.output_tokens else None

    def joules_per_total_token(self) -> float | None:
        total_tokens = self.window.output_tokens + self.window.input_tokens
        return self.combined_total_joules / total_tokens if total_tokens else None

    # -- perf/W ---------------------------------------------------------------
    # tokens/s divided by average watts is tokens per joule, i.e. the
    # reciprocal of the J/token figures above. Both are kept because readers
    # ask for both; they must never be computed from different windows.

    @property
    def has_gpu_power(self) -> bool:
        return bool(self.gpu_per_node)

    @property
    def has_cpu_power(self) -> bool:
        return bool(self.cpu_per_node)

    @property
    def duration_seconds(self) -> float:
        return self.window.duration_seconds

    @property
    def output_tokens_per_second(self) -> float | None:
        return self.window.output_tokens / self.duration_seconds if self.duration_seconds > 0 else None

    @property
    def total_tokens_per_second(self) -> float | None:
        if self.duration_seconds <= 0:
            return None
        return (self.window.output_tokens + self.window.input_tokens) / self.duration_seconds

    @property
    def gpu_avg_power_w(self) -> float | None:
        if not self.has_gpu_power or self.duration_seconds <= 0:
            return None
        return self.gpu_total_joules / self.duration_seconds

    @property
    def combined_avg_power_w(self) -> float | None:
        """CPU+GPU average watts; None unless *both* legs contributed, so it never silently equals GPU-only."""
        if not (self.has_gpu_power and self.has_cpu_power) or self.duration_seconds <= 0:
            return None
        return self.combined_total_joules / self.duration_seconds

    @staticmethod
    def _per_watt(rate: float | None, watts: float | None) -> float | None:
        return rate / watts if rate is not None and watts else None

    @property
    def output_tokens_per_second_per_gpu_watt(self) -> float | None:
        return self._per_watt(self.output_tokens_per_second, self.gpu_avg_power_w)

    @property
    def output_tokens_per_second_per_combined_watt(self) -> float | None:
        return self._per_watt(self.output_tokens_per_second, self.combined_avg_power_w)

    @property
    def total_tokens_per_second_per_gpu_watt(self) -> float | None:
        return self._per_watt(self.total_tokens_per_second, self.gpu_avg_power_w)

    @property
    def total_tokens_per_second_per_combined_watt(self) -> float | None:
        return self._per_watt(self.total_tokens_per_second, self.combined_avg_power_w)

    # -- sample coverage -------------------------------------------------------

    @property
    def coverage_start_unix(self) -> float | None:
        starts = [
            b.sample_start_unix for b in (*self.cpu_per_node, *self.gpu_per_node) if b.sample_start_unix is not None
        ]
        return min(starts) if starts else None

    @property
    def coverage_end_unix(self) -> float | None:
        ends = [b.sample_end_unix for b in (*self.cpu_per_node, *self.gpu_per_node) if b.sample_end_unix is not None]
        return max(ends) if ends else None

    @property
    def coverage_duration_seconds(self) -> float | None:
        start, end = self.coverage_start_unix, self.coverage_end_unix
        return end - start if start is not None and end is not None else None


def build_concurrency_report(
    window: ConcurrencyWindow,
    cpu_samples: CpuSamples | None,
    gpu_samples: GpuSamples | None,
) -> ConcurrencyReport:
    warnings: list[str] = []
    start, end = window.start_unix, window.end_unix

    cpu_per_socket: tuple[EnergyBreakdown, ...] = ()
    cpu_per_node: tuple[EnergyBreakdown, ...] = ()
    cpu_total = 0.0
    if cpu_samples is not None:
        cpu_per_socket = tuple(
            windowed_energy(f"cpu/{host}/socket{socket}", times, watts, start, end)
            for (host, socket), (times, watts) in sorted(cpu_samples.per_socket.items())
        )
        cpu_per_node = tuple(
            windowed_energy(f"cpu/{host}", times, watts, start, end)
            for host, (times, watts) in sorted(cpu_samples.per_node.items())
        )
        cpu_total = sum(node.joules for node in cpu_per_node)
        cpu_utilization = _summarize_utilization(
            cpu_samples.per_socket_utilization,
            device_label="cpu/{host}/socket{index}",
            group_labels={key: (f"cpu/{key[0]}",) for key in cpu_samples.per_socket_utilization},
            start=start,
            end=end,
            warnings=warnings,
        )
    else:
        cpu_utilization = ()

    gpu_per_device: tuple[EnergyBreakdown, ...] = ()
    gpu_per_role: tuple[EnergyBreakdown, ...] = ()
    gpu_per_node: tuple[EnergyBreakdown, ...] = ()
    gpu_total = 0.0
    if gpu_samples is not None:
        gpu_per_device = tuple(
            windowed_energy(f"gpu/{host}/gpu{index}", times, watts, start, end)
            for (host, index), (times, watts) in sorted(gpu_samples.per_device.items())
        )
        gpu_per_node = tuple(
            windowed_energy(f"gpu/{host}", times, watts, start, end)
            for host, (times, watts) in sorted(gpu_samples.per_node.items())
        )
        gpu_total = sum(node.joules for node in gpu_per_node)
        if gpu_samples.per_role:
            per_role = []
            for role, by_host in sorted(gpu_samples.per_role.items()):
                for host, (times, watts) in sorted(by_host.items()):
                    per_role.append(windowed_energy(f"gpu/{host}/{role}", times, watts, start, end))
            gpu_per_role = tuple(per_role)
        else:
            warnings.append("GPU role breakdown unavailable (no manifest.json / expected_devices)")
        gpu_utilization = _summarize_utilization(
            gpu_samples.per_device_utilization,
            device_label="gpu/{host}/gpu{index}",
            group_labels={
                key: (
                    f"gpu/{key[0]}",
                    *(f"gpu/{key[0]}/{role}" for role in sorted(gpu_samples.device_roles.get(key, ()))),
                )
                for key in gpu_samples.per_device_utilization
            },
            start=start,
            end=end,
            warnings=warnings,
        )
    else:
        gpu_utilization = ()

    if gpu_samples is not None and cpu_samples is None:
        warnings.append("combined perf/W unavailable: no CPU power leg in this run (GPU-only figures reported)")

    return ConcurrencyReport(
        window=window,
        cpu_per_socket=cpu_per_socket,
        cpu_per_node=cpu_per_node,
        cpu_total_joules=cpu_total,
        gpu_per_device=gpu_per_device,
        gpu_per_role=gpu_per_role,
        gpu_per_node=gpu_per_node,
        gpu_total_joules=gpu_total,
        cpu_utilization=cpu_utilization,
        gpu_utilization=gpu_utilization,
        warnings=tuple(warnings),
    )


def build_reports(log_dir: Path, *, cpu_samples_csv: Path | None = None) -> list[ConcurrencyReport]:
    paths = discover_run(log_dir, cpu_samples_csv=cpu_samples_csv)
    windows = load_concurrency_windows(paths)

    cpu_samples = load_cpu_samples(paths.cpu_samples_csv) if paths.cpu_samples_csv else None
    gpu_roles = load_gpu_roles(paths.gpu_manifest) if paths.gpu_manifest else None
    gpu_samples = load_gpu_samples(paths.gpu_samples_csv, gpu_roles) if paths.gpu_samples_csv else None

    return [build_concurrency_report(window, cpu_samples, gpu_samples) for window in windows]


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def _fmt(value: float | None, unit: str = "") -> str:
    return "n/a" if value is None else f"{value:,.2f}{unit}"


def render_table(reports: list[ConcurrencyReport]) -> str:
    lines = []
    total_energy = 0.0
    for report in reports:
        w = report.window
        lines.append(f"[{w.benchmark_type} concurrency={w.concurrency}] window={w.end_unix - w.start_unix:.2f}s")
        for warning in report.warnings:
            lines.append(f"  warning: {warning}")
        lines.append(f"  cpu total:      {_fmt(report.cpu_total_joules, ' J')}")
        lines.append(f"  gpu total:      {_fmt(report.gpu_total_joules, ' J')}")
        lines.append(f"  combined total: {_fmt(report.combined_total_joules, ' J')}")
        lines.append(f"  tokens: output={w.output_tokens:,.0f} input={w.input_tokens:,.0f}")
        lines.append(f"  J/output-token: {_fmt(report.joules_per_output_token())}")
        lines.append(f"  J/total-token:  {_fmt(report.joules_per_total_token())}")
        lines.append(f"  timing: {_render_timing(report)}")
        lines.append(f"  perf/W: {_render_perf_per_watt(report)}")
        for breakdown in (*report.cpu_per_socket, *report.gpu_per_device, *report.gpu_per_role):
            stats = ""
            if breakdown.p99_w is not None:
                stats = (
                    f"; p50={breakdown.p50_w:,.2f} p95={breakdown.p95_w:,.2f} "
                    f"p99={breakdown.p99_w:,.2f} max={breakdown.max_w:,.2f} W"
                )
            lines.append(
                f"    {breakdown.label}: {breakdown.joules:,.2f} J ({breakdown.avg_power_w:,.2f} W avg{stats})"
            )
        for label, columns in _utilization_by_label((*report.cpu_utilization, *report.gpu_utilization)).items():
            cells = "; ".join(f"{u.column} mean={u.mean:,.2f} max={u.max:,.2f}" for u in columns)
            lines.append(f"    {label} utilization: {cells}")
        total_energy += report.combined_total_joules
    lines.append(f"\ntotal energy across all concurrency points: {total_energy:,.2f} J")
    return "\n".join(lines)


def _fmt_unix(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _render_timing(report: ConcurrencyReport) -> str:
    w = report.window
    parts = [f"computed start={_fmt_unix(w.start_unix)} end={_fmt_unix(w.end_unix)} duration={w.duration_seconds:.2f}s"]
    reported = w.reported
    if reported.source == REPORTED_UNAVAILABLE:
        parts.append("reported: n/a" + (f" ({reported.note})" if reported.note else ""))
    else:
        cells = [f"reported[{reported.source}]"]
        if reported.start_unix is not None or reported.end_unix is not None:
            cells.append(f"start={_fmt_unix(reported.start_unix)} end={_fmt_unix(reported.end_unix)}")
        if reported.duration_seconds is not None:
            delta = w.duration_seconds - reported.duration_seconds
            cells.append(f"duration={reported.duration_seconds:.2f}s (computed-reported={delta:+.2f}s)")
        parts.append(" ".join(cells))
    if report.coverage_duration_seconds is not None:
        parts.append(
            f"samples cover {_fmt_unix(report.coverage_start_unix)}..{_fmt_unix(report.coverage_end_unix)} "
            f"({report.coverage_duration_seconds:.2f}s)"
        )
    return " | ".join(parts)


def _render_perf_per_watt(report: ConcurrencyReport) -> str:
    def per_watt(output: float | None, total: float | None) -> str:
        if output is None and total is None:
            return "n/a"
        return f"output {output:.4f} tok/s/W, total {total:.4f} tok/s/W"

    parts = [f"output={_fmt(report.output_tokens_per_second)} tok/s total={_fmt(report.total_tokens_per_second)} tok/s"]
    if report.gpu_avg_power_w is not None:
        parts.append(
            f"gpu avg={report.gpu_avg_power_w:,.2f} W -> "
            + per_watt(report.output_tokens_per_second_per_gpu_watt, report.total_tokens_per_second_per_gpu_watt)
        )
    else:
        parts.append("gpu: n/a")
    if report.combined_avg_power_w is not None:
        parts.append(
            f"combined avg={report.combined_avg_power_w:,.2f} W -> "
            + per_watt(
                report.output_tokens_per_second_per_combined_watt, report.total_tokens_per_second_per_combined_watt
            )
        )
    else:
        parts.append("combined: n/a")
    return " | ".join(parts)


def _utilization_by_label(summaries: tuple[UtilizationSummary, ...]) -> dict[str, list[UtilizationSummary]]:
    by_label: dict[str, list[UtilizationSummary]] = {}
    for summary in summaries:
        by_label.setdefault(summary.label, []).append(summary)
    return by_label


def report_to_dict(report: ConcurrencyReport) -> dict:
    w = report.window

    def dump(breakdowns: tuple[EnergyBreakdown, ...]) -> list[dict]:
        return [
            {
                "label": b.label,
                "joules": b.joules,
                "avg_power_w": b.avg_power_w,
                "sample_start_unix": b.sample_start_unix,
                "sample_end_unix": b.sample_end_unix,
                "samples": b.samples,
                **{name: getattr(b, name) for name in POWER_STAT_FIELDS},
            }
            for b in breakdowns
        ]

    def dump_utilization(summaries: tuple[UtilizationSummary, ...]) -> list[dict]:
        return [
            {"label": u.label, "column": u.column, "mean": u.mean, "max": u.max, "samples": u.samples}
            for u in summaries
        ]

    return {
        "benchmark_type": w.benchmark_type,
        "concurrency": w.concurrency,
        "start_unix": w.start_unix,
        "end_unix": w.end_unix,
        "output_tokens": w.output_tokens,
        "input_tokens": w.input_tokens,
        "cpu_per_socket": dump(report.cpu_per_socket),
        "cpu_per_node": dump(report.cpu_per_node),
        "cpu_total_joules": report.cpu_total_joules,
        "gpu_per_device": dump(report.gpu_per_device),
        "gpu_per_role": dump(report.gpu_per_role),
        "gpu_per_node": dump(report.gpu_per_node),
        "gpu_total_joules": report.gpu_total_joules,
        "combined_total_joules": report.combined_total_joules,
        "joules_per_output_token": report.joules_per_output_token(),
        "joules_per_total_token": report.joules_per_total_token(),
        "timing": {
            "computed": {"start_unix": w.start_unix, "end_unix": w.end_unix, "duration_seconds": w.duration_seconds},
            "reported": {
                "source": w.reported.source,
                "start_unix": w.reported.start_unix,
                "end_unix": w.reported.end_unix,
                "duration_seconds": w.reported.duration_seconds,
                "note": w.reported.note,
            },
            "coverage": {
                "sample_start_unix": report.coverage_start_unix,
                "sample_end_unix": report.coverage_end_unix,
                "duration_seconds": report.coverage_duration_seconds,
            },
        },
        "perf_per_watt": {
            "output_tokens_per_second": report.output_tokens_per_second,
            "total_tokens_per_second": report.total_tokens_per_second,
            "gpu_avg_power_w": report.gpu_avg_power_w,
            "combined_avg_power_w": report.combined_avg_power_w,
            "output_tokens_per_second_per_gpu_watt": report.output_tokens_per_second_per_gpu_watt,
            "output_tokens_per_second_per_combined_watt": report.output_tokens_per_second_per_combined_watt,
            "total_tokens_per_second_per_gpu_watt": report.total_tokens_per_second_per_gpu_watt,
            "total_tokens_per_second_per_combined_watt": report.total_tokens_per_second_per_combined_watt,
        },
        "cpu_utilization": dump_utilization(report.cpu_utilization),
        "gpu_utilization": dump_utilization(report.gpu_utilization),
        "warnings": list(report.warnings),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_dir", type=Path, help="A run's logs/ directory")
    parser.add_argument("--json-out", type=Path, help="Optional path to write the report as JSON")
    parser.add_argument(
        "--cpu-samples",
        type=Path,
        help="Explicit CPU samples.csv; required when both power/cpu/ (scraper) and cpu_power/ (host collector) exist",
    )
    args = parser.parse_args(argv)

    try:
        reports = build_reports(args.log_dir, cpu_samples_csv=args.cpu_samples)
    except PowerReportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(render_table(reports))
    if args.json_out:
        args.json_out.write_text(json.dumps([report_to_dict(r) for r in reports], indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
