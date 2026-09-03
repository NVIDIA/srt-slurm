# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Trapezoidal energy/J-per-token report for a completed srtslurm run.

Reads the CPU (``power/cpu/samples.csv``) and GPU (``power/samples.csv``)
power-telemetry CSVs already written by ``cpu_power.py`` / the dcgm-power
exporter, joins them against the profiling window and token counts of each
concurrency point in a sa-bench or aiperf/AgentX sweep, and integrates power
into energy with ``numpy.trapz``.

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
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from srtctl.core.power.contract import MAX_SAMPLE_GAP_SECONDS

_AIPERF_PHASE_RE = re.compile(r"Phase \w+ \(profiling\) (started|complete)")
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


def discover_run(log_dir: Path) -> RunPaths:
    benchmark_out = log_dir / "benchmark.out"
    if not benchmark_out.is_file():
        raise PowerReportError(f"{benchmark_out}: not found")

    cpu_matches = [p for p in log_dir.rglob("samples.csv") if p.parent.name == "cpu"]
    gpu_matches = [p for p in log_dir.rglob("samples.csv") if p.parent.name != "cpu"]
    if len(cpu_matches) > 1:
        raise PowerReportError(f"multiple CPU power samples.csv found below {log_dir}: {cpu_matches}")
    if len(gpu_matches) > 1:
        raise PowerReportError(f"multiple GPU power samples.csv found below {log_dir}: {gpu_matches}")
    cpu_samples_csv = cpu_matches[0] if cpu_matches else None
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


@dataclass(frozen=True)
class ConcurrencyWindow:
    benchmark_type: str
    concurrency: int
    start_unix: float
    end_unix: float
    output_tokens: float
    input_tokens: float
    source: Path


def aiperf_window(concurrency: int, profile_jsonl: Path) -> ConcurrencyWindow:
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

    return ConcurrencyWindow(
        benchmark_type=BENCHMARK_TYPE_AIPERF,
        concurrency=concurrency,
        start_unix=min(starts_ns) / 1e9,
        end_unix=max(ends_ns) / 1e9,
        output_tokens=output_tokens,
        input_tokens=input_tokens,
        source=profile_jsonl,
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
    return ConcurrencyWindow(
        benchmark_type=BENCHMARK_TYPE_SA_BENCH,
        concurrency=concurrency,
        start_unix=result["benchmark_start_time_unix"],
        end_unix=result["benchmark_end_time_unix"],
        output_tokens=result["total_output_tokens"],
        input_tokens=result["total_input_tokens"],
        source=result_json,
    )


def load_concurrency_windows(paths: RunPaths) -> list[ConcurrencyWindow]:
    windows = []
    for concurrency, source in paths.concurrency_sources:
        if source.name == "profile_export.jsonl":
            windows.append(aiperf_window(concurrency, source))
        else:
            windows.append(sa_bench_window(concurrency, source))
    return windows


# ---------------------------------------------------------------------------
# Sample loading
# ---------------------------------------------------------------------------


def _sorted_series(rows: dict[object, list[tuple[float, float]]]) -> dict[object, tuple[np.ndarray, np.ndarray]]:
    series = {}
    for key, points in rows.items():
        points.sort(key=lambda p: p[0])
        times = np.array([t for t, _ in points], dtype=float)
        watts = np.array([w for _, w in points], dtype=float)
        series[key] = (times, watts)
    return series


@dataclass(frozen=True)
class CpuSamples:
    per_socket: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]]
    per_node: dict[str, tuple[np.ndarray, np.ndarray]]


def load_cpu_samples(path: Path) -> CpuSamples:
    per_socket: dict[tuple[str, int], list[tuple[float, float]]] = {}
    node_totals: dict[str, dict[float, float]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            timestamp = float(row["timestamp_unix"])
            hostname = row["hostname"]
            socket_raw = row["socket_id"]
            if socket_raw != "":
                per_socket.setdefault((hostname, int(socket_raw)), []).append((timestamp, float(row["power_w"])))
            node_totals.setdefault(hostname, {})[timestamp] = float(row["total_power_w"])

    per_node_rows = {host: list(values.items()) for host, values in node_totals.items()}
    return CpuSamples(per_socket=_sorted_series(per_socket), per_node=_sorted_series(per_node_rows))


@dataclass(frozen=True)
class GpuSamples:
    per_device: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]]
    per_node: dict[str, tuple[np.ndarray, np.ndarray]]
    per_role: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]  # role -> hostname -> series


def load_gpu_roles(manifest_path: Path) -> dict[tuple[str, int], set[str]]:
    manifest = json.loads(manifest_path.read_text())
    roles: dict[tuple[str, int], set[str]] = {}
    for device in manifest.get("expected_devices", []):
        key = (device["hostname"], device["gpu_index"])
        roles[key] = {assignment["worker_role"] for assignment in device["assignments"]}
    return roles


def load_gpu_samples(path: Path, roles: dict[tuple[str, int], set[str]] | None) -> GpuSamples:
    per_device: dict[tuple[str, int], list[tuple[float, float]]] = {}
    node_totals: dict[str, dict[float, float]] = {}
    role_totals: dict[str, dict[str, dict[float, float]]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            timestamp = float(row["timestamp_unix"])
            hostname = row["hostname"]
            gpu_index = int(row["gpu_index"])
            watts = float(row["power_w"])
            per_device.setdefault((hostname, gpu_index), []).append((timestamp, watts))
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
    )


# ---------------------------------------------------------------------------
# Windowed trapezoidal integration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EnergyBreakdown:
    label: str
    joules: float
    avg_power_w: float


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

    joules = float(np.trapezoid(watts[start_i : end_i + 1], x=times[start_i : end_i + 1]))
    duration = end - start
    return EnergyBreakdown(label=label, joules=joules, avg_power_w=joules / duration if duration > 0 else 0.0)


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
    warnings: tuple[str, ...] = ()

    @property
    def combined_total_joules(self) -> float:
        return self.cpu_total_joules + self.gpu_total_joules

    def joules_per_output_token(self) -> float | None:
        return self.combined_total_joules / self.window.output_tokens if self.window.output_tokens else None

    def joules_per_total_token(self) -> float | None:
        total_tokens = self.window.output_tokens + self.window.input_tokens
        return self.combined_total_joules / total_tokens if total_tokens else None


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

    return ConcurrencyReport(
        window=window,
        cpu_per_socket=cpu_per_socket,
        cpu_per_node=cpu_per_node,
        cpu_total_joules=cpu_total,
        gpu_per_device=gpu_per_device,
        gpu_per_role=gpu_per_role,
        gpu_per_node=gpu_per_node,
        gpu_total_joules=gpu_total,
        warnings=tuple(warnings),
    )


def build_reports(log_dir: Path) -> list[ConcurrencyReport]:
    paths = discover_run(log_dir)
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
        for breakdown in (*report.cpu_per_socket, *report.gpu_per_device, *report.gpu_per_role):
            lines.append(f"    {breakdown.label}: {breakdown.joules:,.2f} J ({breakdown.avg_power_w:,.2f} W avg)")
        total_energy += report.combined_total_joules
    lines.append(f"\ntotal energy across all concurrency points: {total_energy:,.2f} J")
    return "\n".join(lines)


def report_to_dict(report: ConcurrencyReport) -> dict:
    w = report.window

    def dump(breakdowns: tuple[EnergyBreakdown, ...]) -> list[dict]:
        return [{"label": b.label, "joules": b.joules, "avg_power_w": b.avg_power_w} for b in breakdowns]

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
        "warnings": list(report.warnings),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_dir", type=Path, help="A run's logs/ directory")
    parser.add_argument("--json-out", type=Path, help="Optional path to write the report as JSON")
    args = parser.parse_args(argv)

    try:
        reports = build_reports(args.log_dir)
    except PowerReportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(render_table(reports))
    if args.json_out:
        args.json_out.write_text(json.dumps([report_to_dict(r) for r in reports], indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
