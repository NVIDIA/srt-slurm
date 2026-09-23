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
import logging
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TextIO, TypeVar

import numpy as np

from srtctl.core.cpu_power import UTILIZATION_COLUMNS as CPU_UTILIZATION_COLUMNS
from srtctl.core.power.contract import MAX_SAMPLE_GAP_SECONDS, UTILIZATION_METRICS
from srtctl.core.power.cpu_rails import COMPONENT_RAIL_KINDS as CPU_COMPONENT_RAIL_KINDS
from srtctl.core.power.cpu_rails import RAIL_COLUMN_NAMES as CPU_RAIL_COLUMN_NAMES
from srtctl.core.power.cpu_rails import RAIL_COLUMNS as CPU_RAIL_COLUMNS
from srtctl.core.power.cpu_rails import classify_sensor, legacy_rail_rank

logger = logging.getLogger(__name__)

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
    """Per concurrency: ``profile_export.jsonl`` when the run exported per-record data
    (exact request-activity window + warmup span), else ``profile_export_aiperf.json``
    (phase-level start/end only -- see ``aiperf_aggregate_window``)."""
    sources: dict[int, Path] = {}
    for artifacts_dir in log_dir.rglob("conc_*/aiperf_artifacts"):
        match = _CONC_DIR_RE.match(artifacts_dir.parent.name)
        if match is None or not artifacts_dir.is_dir():
            continue
        concurrency = int(match.group(1))
        jsonl_path = artifacts_dir / "profile_export.jsonl"
        aggregate_path = artifacts_dir / "profile_export_aiperf.json"
        if jsonl_path.is_file():
            sources[concurrency] = jsonl_path
        elif aggregate_path.is_file() and concurrency not in sources:
            sources[concurrency] = aggregate_path
    return list(sources.items())


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
    # Time-per-output-token (TPOT, aka inter-token latency), milliseconds.
    # None when the benchmark's own result artifact didn't carry that percentile.
    tpot_p50_ms: float | None = None
    tpot_p90_ms: float | None = None
    # Warmup-phase span (epoch seconds) when the benchmark records one; None for
    # engines that don't expose it (sa-bench). Used to colour the run timeline.
    warmup_start_unix: float | None = None
    warmup_end_unix: float | None = None
    # Drain: from ``end_unix`` (aiperf stopped issuing requests) to the last
    # in-flight request's completion. Load falls from the concurrency target to a
    # handful of stragglers here, so it is excluded from the measured window and
    # shown as its own phase band. None when the engine does not expose it.
    drain_end_unix: float | None = None

    @property
    def duration_seconds(self) -> float:
        return self.end_unix - self.start_unix


TIMEZONE_OFFSET_FILENAME = "agentic_power_timezone_offset.txt"
_TZ_OFFSET_RE = re.compile(r"^([+-])(\d{2}):?(\d{2})$")


def _read_timezone_offset(near: Path) -> timezone | None:
    """The benchmark stage writes the cluster's UTC offset (``-0700`` / ``+05:30``)
    next to each concurrency's artifacts; find it in the artifact dir or up to two
    parents so aiperf's naive local stamps resolve in the *cluster's* zone, not the
    zone of whatever machine builds the report."""
    for base in (near, *near.parents[:2]):
        candidate = base / TIMEZONE_OFFSET_FILENAME
        if candidate.is_file():
            match = _TZ_OFFSET_RE.match(candidate.read_text().strip())
            if match:
                sign = 1 if match.group(1) == "+" else -1
                return timezone(sign * timedelta(hours=int(match.group(2)), minutes=int(match.group(3))))
    return None


def _local_iso_to_unix(stamp: object, tz: timezone | None = None) -> float | None:
    """aiperf writes naive local-time ISO stamps; interpret them in ``tz`` when the
    run recorded its offset, else in this host's zone."""
    if not isinstance(stamp, str):
        return None
    try:
        parsed = datetime.fromisoformat(stamp)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=tz) if tz is not None else parsed.astimezone()
    return parsed.timestamp()


def reported_timing_from_aiperf_json(aggregate: dict, *, tz: timezone | None = None) -> ReportedTiming:
    duration = aggregate.get("benchmark_duration")
    duration_seconds = duration.get("avg") if isinstance(duration, dict) else duration
    start_unix = _local_iso_to_unix(aggregate.get("start_time"), tz)
    end_unix = _local_iso_to_unix(aggregate.get("end_time"), tz)
    if duration_seconds is None and start_unix is None and end_unix is None:
        return REPORTED_TIMING_UNAVAILABLE
    return ReportedTiming(
        source="aiperf-json",
        start_unix=start_unix,
        end_unix=end_unix,
        duration_seconds=float(duration_seconds) if duration_seconds is not None else None,
        note=(
            "start/end are aiperf's naive local ISO stamps converted in the run's recorded timezone offset"
            if tz is not None
            else "start/end are aiperf's naive local ISO stamps converted in this host's timezone"
        ),
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


_AIPERF_WARMUP_LINE_RE = re.compile(
    r"^(?P<stamp>\d{2}:\d{2}:\d{2}\.\d{3}) NOTICE\s+Phase \w+ \(warmup\) (?P<event>started|complete)\b(?P<rest>.*)$"
)


def _warmup_span_by_stamp(
    benchmark_out: Path, profile_stamp: str, profile_start_unix: float
) -> tuple[float, float] | None:
    """Absolute warmup span for the profiling phase that starts at ``profile_start_unix``.

    aiperf's NOTICE stamps are time-of-day only, so the profiling phase's own
    absolute start (from the aggregate JSON) anchors them: the warmup whose
    ``complete`` line immediately precedes that phase's ``started`` line is its
    warmup, and its ``elapsed=`` gives the span ``[profile_start - elapsed,
    profile_start]``. Returns None when the log has no matching warmup.
    """
    last_warmup_elapsed: float | None = None
    last_warmup_start: str | None = None
    with benchmark_out.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.rstrip("\n")
            warm = _AIPERF_WARMUP_LINE_RE.match(line)
            if warm is not None:
                if warm.group("event") == "started":
                    last_warmup_start, last_warmup_elapsed = warm.group("stamp"), None
                else:
                    elapsed = _AIPERF_ELAPSED_RE.search(warm.group("rest"))
                    if elapsed is not None:
                        last_warmup_elapsed = float(elapsed.group("seconds"))
                    elif last_warmup_start is not None:
                        d = _time_of_day_seconds(warm.group("stamp")) - _time_of_day_seconds(last_warmup_start)
                        last_warmup_elapsed = d + (24 * 3600.0 if d < 0 else 0.0)
                continue
            prof = _AIPERF_PHASE_LINE_RE.match(line)
            if prof is not None and prof.group("event") == "started":
                # Stamps in the log are the cluster's local time-of-day; compare on HH:MM:SS
                # in the same zone the aggregate's start_time was resolved in.
                if prof.group("stamp")[:8] == profile_stamp and last_warmup_elapsed is not None:
                    return profile_start_unix - last_warmup_elapsed, profile_start_unix
                last_warmup_elapsed = None
    return None


def _tpot_from_aiperf_aggregate(aggregate: dict) -> tuple[float | None, float | None]:
    """p50/p90 TPOT (ms) from aiperf's ``inter_token_latency`` percentile block."""
    itl = aggregate.get("inter_token_latency")
    if not isinstance(itl, dict):
        return None, None
    return itl.get("p50"), itl.get("p90")


def _tpot_from_sa_bench_result(result: dict) -> tuple[float | None, float | None]:
    """p50/p90 TPOT (ms) from sa-bench's result JSON.

    ``median_tpot_ms`` is always the p50; p90 only exists when ``--percentile-metrics``
    requested it, as ``percentiles_tpot_ms``: a list of ``[percentile, value_ms]`` pairs.
    """
    p50 = result.get("median_tpot_ms")
    p90 = None
    percentiles = result.get("percentiles_tpot_ms")
    if isinstance(percentiles, list):
        for entry in percentiles:
            if isinstance(entry, list | tuple) and len(entry) == 2 and float(entry[0]) == 90.0:
                p90 = entry[1]
                break
    return p50, p90


def aiperf_window(concurrency: int, profile_jsonl: Path, *, benchmark_out: Path | None = None) -> ConcurrencyWindow:
    """Real epoch-second window from ``time.time_ns()`` per-record timestamps.

    Uses only profiling-phase, non-error rows -- the same filter
    ``measurement_window.py`` uses -- so the window reflects actual request
    activity, not the phase's grace-period timeout deadline (which can run
    long after the last real response, as observed in practice).

    The window starts at the first profiling request and ends when aiperf stopped
    *issuing* requests: ``start + benchmark_duration`` from the aggregate. Requests
    still in flight past that point are the drain -- concurrency collapses from the
    target to a few stragglers, so integrating over it would dilute the average.
    The drain is kept as ``drain_end_unix`` for the timeline. If the aggregate has
    no duration the window falls back to the last request's completion.
    """
    starts_ns: list[int] = []
    ends_ns: list[int] = []
    warmup_starts_ns: list[int] = []
    warmup_ends_ns: list[int] = []
    with profile_jsonl.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("error"):
                continue
            metadata = record.get("metadata", {})
            phase = metadata.get("benchmark_phase")
            if phase == "profiling":
                starts_ns.append(metadata["request_start_ns"])
                ends_ns.append(metadata["request_end_ns"])
            elif phase == "warmup" and "request_start_ns" in metadata:
                warmup_starts_ns.append(metadata["request_start_ns"])
                warmup_ends_ns.append(metadata.get("request_end_ns", metadata["request_start_ns"]))
    if not starts_ns:
        raise PowerReportError(f"{profile_jsonl}: no successful profiling-phase records")

    aggregate_path = profile_jsonl.with_name("profile_export_aiperf.json")
    if not aggregate_path.is_file():
        raise PowerReportError(f"{aggregate_path}: not found (expected alongside {profile_jsonl})")
    aggregate = json.loads(aggregate_path.read_text())
    output_tokens = aggregate["total_osl"]["avg"]
    input_tokens = aggregate["total_isl"]["avg"]

    reported = reported_timing_from_aiperf_json(aggregate, tz=_read_timezone_offset(profile_jsonl.parent))
    if reported.source == REPORTED_UNAVAILABLE and benchmark_out is not None and benchmark_out.is_file():
        reported = reported_timing_from_phase_log(benchmark_out)

    tpot_p50_ms, tpot_p90_ms = _tpot_from_aiperf_aggregate(aggregate)

    start_unix = min(starts_ns) / 1e9
    last_end_unix = max(ends_ns) / 1e9
    issue_end_unix = _issuing_end(start_unix, aggregate)
    end_unix = min(issue_end_unix, last_end_unix) if issue_end_unix is not None else last_end_unix
    return ConcurrencyWindow(
        benchmark_type=BENCHMARK_TYPE_AIPERF,
        concurrency=concurrency,
        start_unix=start_unix,
        end_unix=end_unix,
        output_tokens=output_tokens,
        input_tokens=input_tokens,
        source=profile_jsonl,
        reported=reported,
        tpot_p50_ms=tpot_p50_ms,
        tpot_p90_ms=tpot_p90_ms,
        warmup_start_unix=min(warmup_starts_ns) / 1e9 if warmup_starts_ns else None,
        warmup_end_unix=max(warmup_ends_ns) / 1e9 if warmup_ends_ns else None,
        drain_end_unix=last_end_unix if last_end_unix > end_unix else None,
    )


def _issuing_end(start_unix: float, aggregate: dict) -> float | None:
    """``start + benchmark_duration``: when aiperf stopped issuing requests. The
    aggregate's ``benchmark_duration`` is in seconds (``unit`` is checked)."""
    duration = aggregate.get("benchmark_duration")
    if not isinstance(duration, dict):
        return None
    value, unit = duration.get("avg"), duration.get("unit", "sec")
    if not isinstance(value, (int, float)):
        return None
    scale = {"sec": 1.0, "s": 1.0, "seconds": 1.0, "ms": 1e-3}.get(unit)
    if scale is None:
        return None
    return start_unix + float(value) * scale


def aiperf_aggregate_window(
    concurrency: int, aggregate_path: Path, *, benchmark_out: Path | None = None
) -> ConcurrencyWindow:
    """Window from ``profile_export_aiperf.json`` alone, for runs that did not export
    per-record ``profile_export.jsonl``.

    aiperf's ``start_time``/``end_time`` bracket the whole profiling phase including
    its drain, so the window is cut at ``start_time + benchmark_duration`` (end of
    request issuing) like the per-record path, with ``end_time`` kept as the drain
    end. Stamps are naive local time; the run's recorded timezone offset
    (``agentic_power_timezone_offset.txt``) is required to place them -- without it
    the host's zone is used, which is only right when the report is built where the
    benchmark ran.
    """
    aggregate = json.loads(aggregate_path.read_text())
    tz = _read_timezone_offset(aggregate_path.parent)
    reported = reported_timing_from_aiperf_json(aggregate, tz=tz)
    if reported.start_unix is None or reported.end_unix is None:
        raise PowerReportError(f"{aggregate_path}: no start_time/end_time to derive a window from")
    for key in ("total_osl", "total_isl"):
        if key not in aggregate:
            raise PowerReportError(f"{aggregate_path}: missing required field {key!r}")
    tpot_p50_ms, tpot_p90_ms = _tpot_from_aiperf_aggregate(aggregate)
    warmup = None
    if benchmark_out is not None and benchmark_out.is_file():
        warmup = _warmup_span(benchmark_out, aggregate.get("start_time"), reported.start_unix, tz)
    issue_end_unix = _issuing_end(reported.start_unix, aggregate)
    end_unix = min(issue_end_unix, reported.end_unix) if issue_end_unix is not None else reported.end_unix
    return ConcurrencyWindow(
        benchmark_type=BENCHMARK_TYPE_AIPERF,
        concurrency=concurrency,
        start_unix=reported.start_unix,
        end_unix=end_unix,
        output_tokens=aggregate["total_osl"]["avg"],
        input_tokens=aggregate["total_isl"]["avg"],
        source=aggregate_path,
        reported=reported,
        tpot_p50_ms=tpot_p50_ms,
        tpot_p90_ms=tpot_p90_ms,
        warmup_start_unix=warmup[0] if warmup else None,
        warmup_end_unix=warmup[1] if warmup else None,
        drain_end_unix=reported.end_unix if reported.end_unix > end_unix else None,
    )


def _warmup_span(
    benchmark_out: Path, start_stamp: object, profile_start_unix: float, tz: timezone | None
) -> tuple[float, float] | None:
    """Match the phase log's local time-of-day against the aggregate's own naive local
    ``start_time`` (same clock, no zone conversion needed), then anchor the warmup
    to the resolved ``profile_start_unix``."""
    if not isinstance(start_stamp, str):
        return None
    local_hms = start_stamp[11:19]
    return _warmup_span_by_stamp(benchmark_out, local_hms, profile_start_unix)


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
    tpot_p50_ms, tpot_p90_ms = _tpot_from_sa_bench_result(result)
    return ConcurrencyWindow(
        benchmark_type=BENCHMARK_TYPE_SA_BENCH,
        concurrency=concurrency,
        start_unix=result["benchmark_start_time_unix"],
        end_unix=result["benchmark_end_time_unix"],
        output_tokens=result["total_output_tokens"],
        input_tokens=result["total_input_tokens"],
        source=result_json,
        tpot_p50_ms=tpot_p50_ms,
        tpot_p90_ms=tpot_p90_ms,
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
        elif source.name == "profile_export_aiperf.json":
            windows.append(aiperf_aggregate_window(concurrency, source, benchmark_out=paths.benchmark_out))
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
class NodePower:
    """Window-average power for one node, split by source, for the per-node bar chart.

    ``gpu_w`` sums the node's GPUs; ``cpu_w`` sums its sockets' authoritative
    ``power_w`` (ACPI envelope / DCGM). ``cpu_rails_w`` holds the component rails
    (``cpu_rail`` / ``soc`` / ``dram``) summed across sockets when the file
    carried them -- a breakdown *of* ``cpu_w``, not an addition to it. Any of
    the three may be None when that leg wasn't collected on the node.
    """

    hostname: str
    gpu_w: float | None
    cpu_w: float | None
    cpu_rails_w: dict[str, float] = field(default_factory=dict)
    # Device counts behind the sums (for per-device averages) and the worker roles
    # the node's GPUs carry (from the GPU manifest), so nodes can be grouped by type.
    gpu_count: int = 0
    socket_count: int = 0
    roles: tuple[str, ...] = ()


@dataclass(frozen=True)
class CpuSensorProvenance:
    """Which channel a socket's power series came from, for the report."""

    label: str
    sensor: str
    other_sensors: tuple[str, ...] = ()


@dataclass(frozen=True)
class CpuSamples:
    per_socket: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]]
    per_node: dict[str, tuple[np.ndarray, np.ndarray]]
    per_socket_utilization: dict[tuple[str, int], UtilizationSeries] = field(default_factory=dict)
    # Which sensor channel each socket's series came from (legacy long-format files:
    # see _select_legacy_socket_series; wide files: the row's power_w column), plus
    # the channels that were present but not used -- provenance for the report.
    socket_sensor: dict[tuple[str, int], str] = field(default_factory=dict)
    socket_other_sensors: dict[tuple[str, int], tuple[str, ...]] = field(default_factory=dict)
    # Component rails (cpu_rail / soc / dram) per socket, when the file carries them.
    # Reference breakdowns of ``per_socket`` -- NOT additive to it (see cpu_rails).
    per_socket_rails: dict[tuple[str, int], dict[str, tuple[np.ndarray, np.ndarray]]] = field(default_factory=dict)


def load_cpu_samples(path: Path) -> CpuSamples:
    with path.open(newline="", encoding="utf-8") as handle:
        return load_cpu_samples_from(handle)


def _is_wide_cpu_csv(fieldnames: Sequence[str] | None) -> bool:
    """Wide layout (one row per socket, rails as columns) vs. legacy long layout (one row per rail)."""
    return fieldnames is not None and all(column in fieldnames for column in CPU_RAIL_COLUMN_NAMES)


def _select_legacy_socket_series(
    by_sensor: dict[tuple[str, int], dict[str, dict[float, float]]],
) -> tuple[dict[tuple[str, int], list[tuple[float, float]]], dict[tuple[str, int], str]]:
    """Legacy long-format CSVs: pick ONE sensor per socket to be its power series.

    Those files hold one row per rail per instant (total envelope plus the
    cpu_rail/soc/dram components) all under the same socket_id. Feeding them
    all to the trapezoid produced several "samples" at the same instant and
    integrated a jumble of rails instead of the socket's power. The socket
    envelope is authoritative when present; the ranking and sensor-name
    classification come from ``cpu_rails`` so this never drifts from what
    the writers emit. Ties within a kind resolve by sensor name.

    Returns ``(series per socket, chosen sensor name per socket)``.
    """
    selected: dict[tuple[str, int], list[tuple[float, float]]] = {}
    chosen: dict[tuple[str, int], str] = {}
    for key, sensors in by_sensor.items():
        best = min(sensors, key=lambda name: (legacy_rail_rank(name), name))
        selected[key] = list(sensors[best].items())
        chosen[key] = best
    return selected, chosen


def load_cpu_samples_from(handle: TextIO) -> CpuSamples:
    """Load either CPU CSV layout into per-socket / per-node power series.

    Wide layout (current writers): ``power_w`` *is* the socket's power, so the
    row feeds the socket series directly. Long layout (legacy): rows are
    grouped by sensor and one rail per socket is selected afterwards.
    """
    reader = csv.DictReader(handle)
    wide = _is_wide_cpu_csv(reader.fieldnames)
    # (host, socket) -> sensor -> timestamp -> watts. Keyed by timestamp so a
    # repeated row for the same sensor/instant overwrites rather than
    # double-counting. In the wide layout each socket has exactly one sensor.
    by_sensor: dict[tuple[str, int], dict[str, dict[float, float]]] = {}
    wide_sensor: dict[tuple[str, int], str] = {}  # wide layout: the row's own sensor name, for provenance
    # (host, socket) -> rail kind -> timestamp -> watts, for the component rails.
    rails: dict[tuple[str, int], dict[str, dict[float, float]]] = {}
    node_totals: dict[str, dict[float, float]] = {}
    utilization: dict[tuple[str, int], dict[str, list[tuple[float, float]]]] = {}
    for row in reader:
        timestamp = float(row["timestamp_unix"])
        hostname = row["hostname"]
        socket_raw = row["socket_id"]
        if socket_raw != "":
            key = (hostname, int(socket_raw))
            sensor = "power_w" if wide else row["sensor"]
            if wide:
                wide_sensor.setdefault(key, row.get("sensor") or "power_w")
                for kind in CPU_COMPONENT_RAIL_KINDS:
                    raw = row.get(CPU_RAIL_COLUMNS[kind])
                    if raw not in (None, ""):
                        rails.setdefault(key, {}).setdefault(kind, {})[timestamp] = float(raw)
            else:
                kind = classify_sensor(sensor)
                if kind in CPU_COMPONENT_RAIL_KINDS:
                    rails.setdefault(key, {}).setdefault(kind, {})[timestamp] = float(row["power_w"])
            by_sensor.setdefault(key, {}).setdefault(sensor, {})[timestamp] = float(row["power_w"])
            _collect_utilization(row, CPU_UTILIZATION_COLUMNS, timestamp, utilization.setdefault(key, {}))
        # total_power_w is blank whenever an ACPI scrape has no `grace` channel
        # (see contract.CPU_SAMPLES_HEADER); skip rather than crash on float("").
        if row["total_power_w"] != "":
            node_totals.setdefault(hostname, {})[timestamp] = float(row["total_power_w"])

    if wide:
        per_socket_rows = {key: list(next(iter(sensors.values())).items()) for key, sensors in by_sensor.items()}
        socket_sensor = wide_sensor
        socket_other: dict[tuple[str, int], tuple[str, ...]] = {}
    else:
        per_socket_rows, socket_sensor = _select_legacy_socket_series(by_sensor)
        socket_other = {
            key: tuple(sorted(name for name in sensors if name != socket_sensor[key]))
            for key, sensors in by_sensor.items()
        }
    per_node_rows = {host: list(values.items()) for host, values in node_totals.items()}
    return CpuSamples(
        per_socket=_sorted_series(per_socket_rows),
        per_node=_sorted_series(per_node_rows),
        per_socket_utilization=_sorted_utilization(utilization),
        socket_sensor=socket_sensor,
        socket_other_sensors=socket_other,
        per_socket_rails={
            key: _sorted_series({kind: list(values.items()) for kind, values in kinds.items()})
            for key, kinds in rails.items()
        },
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
    cpu_sensors: tuple[CpuSensorProvenance, ...] = ()
    node_power: tuple[NodePower, ...] = ()
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
    def cpu_avg_power_w(self) -> float | None:
        if not self.has_cpu_power or self.duration_seconds <= 0:
            return None
        return self.cpu_total_joules / self.duration_seconds

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
    def output_tokens_per_second_per_cpu_watt(self) -> float | None:
        return self._per_watt(self.output_tokens_per_second, self.cpu_avg_power_w)

    @property
    def output_tokens_per_second_per_combined_watt(self) -> float | None:
        return self._per_watt(self.output_tokens_per_second, self.combined_avg_power_w)

    @property
    def total_tokens_per_second_per_gpu_watt(self) -> float | None:
        return self._per_watt(self.total_tokens_per_second, self.gpu_avg_power_w)

    @property
    def total_tokens_per_second_per_cpu_watt(self) -> float | None:
        return self._per_watt(self.total_tokens_per_second, self.cpu_avg_power_w)

    @property
    def total_tokens_per_second_per_combined_watt(self) -> float | None:
        return self._per_watt(self.total_tokens_per_second, self.combined_avg_power_w)

    # -- per-GPU throughput -----------------------------------------------------

    @property
    def num_gpus(self) -> int:
        return len(self.gpu_per_device)

    @property
    def output_tokens_per_second_per_gpu(self) -> float | None:
        if not self.num_gpus or self.output_tokens_per_second is None:
            return None
        return self.output_tokens_per_second / self.num_gpus

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
    cpu_sensors: tuple[CpuSensorProvenance, ...] = ()
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
        _check_cpu_socket_sum(cpu_per_socket, cpu_total, warnings)
        cpu_sensors = tuple(
            CpuSensorProvenance(
                label=f"cpu/{host}/socket{socket}",
                sensor=cpu_samples.socket_sensor.get((host, socket), ""),
                other_sensors=cpu_samples.socket_other_sensors.get((host, socket), ()),
            )
            for (host, socket) in sorted(cpu_samples.per_socket)
        )
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

    node_power = _node_power(cpu_samples, gpu_samples, cpu_per_node, gpu_per_node, start, end)

    return ConcurrencyReport(
        window=window,
        cpu_per_socket=cpu_per_socket,
        cpu_per_node=cpu_per_node,
        cpu_total_joules=cpu_total,
        cpu_sensors=cpu_sensors,
        node_power=node_power,
        gpu_per_device=gpu_per_device,
        gpu_per_role=gpu_per_role,
        gpu_per_node=gpu_per_node,
        gpu_total_joules=gpu_total,
        cpu_utilization=cpu_utilization,
        gpu_utilization=gpu_utilization,
        warnings=tuple(warnings),
    )


def _window_avg_w(times: np.ndarray, watts: np.ndarray, start: float, end: float) -> float | None:
    """Window-average watts via the same snapped trapezoid as ``windowed_energy``;
    None when the series has no coverage of the window."""
    try:
        return windowed_energy("rail", times, watts, start, end).avg_power_w
    except PowerReportError:
        return None


def _node_power(
    cpu_samples: CpuSamples | None,
    gpu_samples: GpuSamples | None,
    cpu_per_node: tuple[EnergyBreakdown, ...],
    gpu_per_node: tuple[EnergyBreakdown, ...],
    start: float,
    end: float,
) -> tuple[NodePower, ...]:
    """One ``NodePower`` per host seen on either leg, sorted by hostname."""
    cpu_by_host = {b.label.removeprefix("cpu/"): b.avg_power_w for b in cpu_per_node}
    gpu_by_host = {b.label.removeprefix("gpu/"): b.avg_power_w for b in gpu_per_node}
    hosts = sorted(set(cpu_by_host) | set(gpu_by_host))

    gpu_count: dict[str, int] = {}
    roles_by_host: dict[str, set[str]] = {}
    if gpu_samples is not None:
        for host, index in gpu_samples.per_device:
            gpu_count[host] = gpu_count.get(host, 0) + 1
            roles_by_host.setdefault(host, set()).update(gpu_samples.device_roles.get((host, index), ()))
    socket_count: dict[str, int] = {}
    if cpu_samples is not None:
        for host, _socket in cpu_samples.per_socket:
            socket_count[host] = socket_count.get(host, 0) + 1

    rails_by_host: dict[str, dict[str, float]] = {}
    if cpu_samples is not None:
        for (host, _socket), kinds in cpu_samples.per_socket_rails.items():
            for kind, (times, watts) in kinds.items():
                avg = _window_avg_w(times, watts, start, end)
                if avg is not None:
                    rails_by_host.setdefault(host, {})[kind] = rails_by_host.get(host, {}).get(kind, 0.0) + avg

    return tuple(
        NodePower(
            hostname=host,
            gpu_w=gpu_by_host.get(host),
            cpu_w=cpu_by_host.get(host),
            cpu_rails_w={
                kind: rails_by_host[host][kind]
                for kind in CPU_COMPONENT_RAIL_KINDS
                if kind in rails_by_host.get(host, {})
            },
            gpu_count=gpu_count.get(host, 0),
            socket_count=socket_count.get(host, 0),
            roles=tuple(sorted(roles_by_host.get(host, ()))),
        )
        for host in hosts
    )


# Socket-vs-node energy agreement tolerance. ACPI ``total_power_w`` is the exporter's
# own sum of the per-socket envelope channels at scrape time, so the two integrals
# should match to rounding; anything beyond this means the socket series are being
# built from the wrong channel(s) again.
CPU_SOCKET_SUM_TOLERANCE = 0.02


def _check_cpu_socket_sum(per_socket: tuple[EnergyBreakdown, ...], node_total: float, warnings: list[str]) -> None:
    socket_total = sum(b.joules for b in per_socket)
    if node_total <= 0 or not per_socket:
        return
    rel = abs(socket_total - node_total) / node_total
    if rel > CPU_SOCKET_SUM_TOLERANCE:
        warnings.append(
            f"CPU energy mismatch: per-socket sum {socket_total:,.0f} J vs node total {node_total:,.0f} J "
            f"({rel:.1%} apart) -- socket series may be using the wrong sensor channel"
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
        lines.append(f"  TPOT: p50={_fmt(w.tpot_p50_ms, ' ms')} p90={_fmt(w.tpot_p90_ms, ' ms')}")
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

    parts = [
        (
            f"output={_fmt(report.output_tokens_per_second)} tok/s "
            f"({_fmt(report.output_tokens_per_second_per_gpu)} tok/s/gpu across {report.num_gpus} gpu(s)) "
            f"total={_fmt(report.total_tokens_per_second)} tok/s"
        )
    ]
    if report.gpu_avg_power_w is not None:
        parts.append(
            f"gpu avg={report.gpu_avg_power_w:,.2f} W -> "
            + per_watt(report.output_tokens_per_second_per_gpu_watt, report.total_tokens_per_second_per_gpu_watt)
        )
    else:
        parts.append("gpu: n/a")
    if report.cpu_avg_power_w is not None:
        parts.append(
            f"cpu avg={report.cpu_avg_power_w:,.2f} W -> "
            + per_watt(report.output_tokens_per_second_per_cpu_watt, report.total_tokens_per_second_per_cpu_watt)
        )
    else:
        parts.append("cpu: n/a")
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
        "source": str(w.source),
        "concurrency": w.concurrency,
        "start_unix": w.start_unix,
        "end_unix": w.end_unix,
        "output_tokens": w.output_tokens,
        "input_tokens": w.input_tokens,
        "tpot_p50_ms": w.tpot_p50_ms,
        "tpot_p90_ms": w.tpot_p90_ms,
        "warmup_start_unix": w.warmup_start_unix,
        "warmup_end_unix": w.warmup_end_unix,
        "drain_end_unix": w.drain_end_unix,
        "cpu_per_socket": dump(report.cpu_per_socket),
        "cpu_per_node": dump(report.cpu_per_node),
        "cpu_total_joules": report.cpu_total_joules,
        "node_power": [
            {
                "hostname": n.hostname,
                "gpu_w": n.gpu_w,
                "cpu_w": n.cpu_w,
                "cpu_rails_w": dict(n.cpu_rails_w),
                "gpu_count": n.gpu_count,
                "socket_count": n.socket_count,
                "roles": list(n.roles),
            }
            for n in report.node_power
        ],
        "cpu_sensors": [
            {"label": p.label, "sensor": p.sensor, "other_sensors": list(p.other_sensors)} for p in report.cpu_sensors
        ],
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
            "num_gpus": report.num_gpus,
            "output_tokens_per_second_per_gpu": report.output_tokens_per_second_per_gpu,
            "gpu_avg_power_w": report.gpu_avg_power_w,
            "cpu_avg_power_w": report.cpu_avg_power_w,
            "combined_avg_power_w": report.combined_avg_power_w,
            "output_tokens_per_second_per_gpu_watt": report.output_tokens_per_second_per_gpu_watt,
            "output_tokens_per_second_per_cpu_watt": report.output_tokens_per_second_per_cpu_watt,
            "output_tokens_per_second_per_combined_watt": report.output_tokens_per_second_per_combined_watt,
            "total_tokens_per_second_per_gpu_watt": report.total_tokens_per_second_per_gpu_watt,
            "total_tokens_per_second_per_cpu_watt": report.total_tokens_per_second_per_cpu_watt,
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
