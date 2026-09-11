# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the trapezoidal CPU/GPU energy report."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from srtctl.analysis.power_energy_report import (
    ConcurrencyReport,
    ConcurrencyWindow,
    EnergyBreakdown,
    PowerReportError,
    aiperf_window,
    build_concurrency_report,
    build_reports,
    detect_benchmark_type,
    discover_run,
    load_cpu_samples,
    load_gpu_roles,
    load_gpu_samples,
    render_table,
    report_to_dict,
    reported_timing_from_phase_log,
    sa_bench_window,
    windowed_energy,
    windowed_utilization,
)

# ---------------------------------------------------------------------------
# detect_benchmark_type
# ---------------------------------------------------------------------------


def test_detects_aiperf_from_phase_notice_lines(tmp_path: Path) -> None:
    log = tmp_path / "benchmark.out"
    log.write_text(
        "17:59:31.680 NOTICE   Phase profiling (profiling) started | target: 3600.0s duration (runner.py:593)\n"
        "19:00:01.681 NOTICE   Phase profiling (profiling) complete | completed=1,342 (runner.py:1162)\n"
    )

    assert detect_benchmark_type(log) == "aiperf"


def test_detects_sa_bench_from_result_markers(tmp_path: Path) -> None:
    log = tmp_path / "benchmark.out"
    log.write_text(
        "============ Serving Benchmark Result ============\nSuccessful requests:                     10        \n"
    )

    assert detect_benchmark_type(log) == "sa-bench"


def test_ambiguous_markers_raise(tmp_path: Path) -> None:
    log = tmp_path / "benchmark.out"
    log.write_text(
        "17:59:31.680 NOTICE   Phase profiling (profiling) started (runner.py:593)\n"
        "Successful requests:                     10\n"
    )

    with pytest.raises(PowerReportError, match="both aiperf and sa-bench"):
        detect_benchmark_type(log)


def test_no_markers_raise(tmp_path: Path) -> None:
    log = tmp_path / "benchmark.out"
    log.write_text("nothing relevant here\n")

    with pytest.raises(PowerReportError, match="neither aiperf nor sa-bench"):
        detect_benchmark_type(log)


# ---------------------------------------------------------------------------
# windowed_energy / nearest-sample bracketing
# ---------------------------------------------------------------------------


def test_windowed_energy_matches_manual_trapezoid() -> None:
    times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    watts = np.array([10.0, 20.0, 20.0, 10.0, 10.0])

    result = windowed_energy("label", times, watts, start=1.0, end=3.0)

    assert result.joules == pytest.approx(np.trapezoid(watts[1:4], x=times[1:4]))
    assert result.avg_power_w == pytest.approx(result.joules / 2.0)


def test_windowed_energy_snaps_to_nearest_sample_on_each_side() -> None:
    times = np.array([0.0, 0.9, 2.1, 3.0])
    watts = np.array([10.0, 10.0, 10.0, 10.0])

    # start=1.0 is nearer 0.9 than 2.1; end=2.0 is nearer 2.1 than 0.9.
    result = windowed_energy("label", times, watts, start=1.0, end=2.0)

    assert result.joules == pytest.approx(np.trapezoid([10.0, 10.0], x=[0.9, 2.1]))


def test_windowed_energy_errors_when_start_gap_exceeds_threshold() -> None:
    times = np.array([100.0, 101.0, 102.0])
    watts = np.array([5.0, 5.0, 5.0])

    with pytest.raises(PowerReportError, match="window start"):
        windowed_energy("label", times, watts, start=0.0, end=101.0)


def test_windowed_energy_errors_when_end_gap_exceeds_threshold() -> None:
    times = np.array([100.0, 101.0, 102.0])
    watts = np.array([5.0, 5.0, 5.0])

    with pytest.raises(PowerReportError, match="window end"):
        windowed_energy("label", times, watts, start=100.0, end=500.0)


def test_windowed_energy_errors_on_sub_interval_window() -> None:
    times = np.array([0.0, 0.1, 0.2])
    watts = np.array([5.0, 5.0, 5.0])

    with pytest.raises(PowerReportError, match="narrower than the sample spacing"):
        windowed_energy("label", times, watts, start=0.04, end=0.05)


def test_windowed_energy_errors_on_empty_series() -> None:
    with pytest.raises(PowerReportError, match="no power samples"):
        windowed_energy("label", np.array([]), np.array([]), start=0.0, end=1.0)


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------


def _write_cpu_csv(path: Path, rows: list[tuple]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "schema_version",
                "timestamp_unix",
                "hostname",
                "source",
                "sensor",
                "socket_id",
                "power_w",
                "total_power_w",
            ]
        )
        writer.writerows(rows)


def test_load_cpu_samples_groups_per_socket_and_dedupes_node_total(tmp_path: Path) -> None:
    path = tmp_path / "cpu" / "samples.csv"
    _write_cpu_csv(
        path,
        [
            (1, 10.0, "node-a", "acpi", "CPU0:cpuPowerUsageW", 0, 40.0, 90.0),
            (1, 10.0, "node-a", "acpi", "CPU1:cpuPowerUsageW", 1, 50.0, 90.0),
            (1, 11.0, "node-a", "acpi", "CPU0:cpuPowerUsageW", 0, 42.0, 92.0),
            (1, 11.0, "node-a", "acpi", "CPU1:cpuPowerUsageW", 1, 50.0, 92.0),
        ],
    )

    samples = load_cpu_samples(path)

    times, watts = samples.per_socket[("node-a", 0)]
    assert list(times) == [10.0, 11.0]
    assert list(watts) == [40.0, 42.0]

    node_times, node_watts = samples.per_node["node-a"]
    assert list(node_times) == [10.0, 11.0]
    assert list(node_watts) == [90.0, 92.0]  # deduped, not summed across the two sensor rows


def test_load_cpu_samples_skips_blank_total_without_a_grace_channel(tmp_path: Path) -> None:
    """ACPI-only scrapes have no `grace` channel, so total_power_w is legitimately blank."""
    path = tmp_path / "cpu" / "samples.csv"
    _write_cpu_csv(
        path,
        [
            (1, 10.0, "node-a", "acpi", "CPU0:cpuPowerUsageW", 0, 40.0, ""),
            (1, 10.0, "node-a", "acpi", "CPU1:cpuPowerUsageW", 1, 50.0, ""),
            (1, 11.0, "node-a", "acpi", "CPU0:cpuPowerUsageW", 0, 42.0, 92.0),
            (1, 11.0, "node-a", "acpi", "CPU1:cpuPowerUsageW", 1, 50.0, 92.0),
        ],
    )

    samples = load_cpu_samples(path)

    times, watts = samples.per_socket[("node-a", 0)]
    assert list(times) == [10.0, 11.0]
    assert list(watts) == [40.0, 42.0]

    node_times, node_watts = samples.per_node["node-a"]
    assert list(node_times) == [11.0]
    assert list(node_watts) == [92.0]


# ---------------------------------------------------------------------------
# aiperf window/tokens
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def test_aiperf_window_filters_warmup_and_errors(tmp_path: Path) -> None:
    conc_dir = tmp_path / "conc_8" / "aiperf_artifacts"
    jsonl_path = conc_dir / "profile_export.jsonl"
    _write_jsonl(
        jsonl_path,
        [
            {"metadata": {"benchmark_phase": "warmup", "request_start_ns": 1, "request_end_ns": 2}},
            {
                "metadata": {
                    "benchmark_phase": "profiling",
                    "request_start_ns": 1_000_000_000,
                    "request_end_ns": 2_000_000_000,
                }
            },
            {
                "metadata": {
                    "benchmark_phase": "profiling",
                    "request_start_ns": 500_000_000,
                    "request_end_ns": 3_000_000_000,
                }
            },
            {
                "error": "boom",
                "metadata": {"benchmark_phase": "profiling", "request_start_ns": 0, "request_end_ns": 999_000_000_000},
            },
        ],
    )
    (conc_dir / "profile_export_aiperf.json").write_text(
        json.dumps({"total_osl": {"avg": 42.0}, "total_isl": {"avg": 7.0}})
    )

    window = aiperf_window(8, jsonl_path)

    assert window.start_unix == pytest.approx(0.5)
    assert window.end_unix == pytest.approx(3.0)
    assert window.output_tokens == 42.0
    assert window.input_tokens == 7.0


def test_aiperf_window_requires_aggregate_file(tmp_path: Path) -> None:
    conc_dir = tmp_path / "conc_8" / "aiperf_artifacts"
    jsonl_path = conc_dir / "profile_export.jsonl"
    _write_jsonl(
        jsonl_path,
        [{"metadata": {"benchmark_phase": "profiling", "request_start_ns": 1, "request_end_ns": 2}}],
    )

    with pytest.raises(PowerReportError, match="profile_export_aiperf.json"):
        aiperf_window(8, jsonl_path)


def test_sa_bench_window_reads_fields_directly(tmp_path: Path) -> None:
    result_path = tmp_path / "results_concurrency_1_gpus_4.json"
    result_path.write_text(
        json.dumps(
            {
                "benchmark_start_time_unix": 100.0,
                "benchmark_end_time_unix": 101.5,
                "total_input_tokens": 10,
                "total_output_tokens": 20,
            }
        )
    )

    window = sa_bench_window(1, result_path)

    assert window.start_unix == 100.0
    assert window.end_unix == 101.5
    assert window.input_tokens == 10
    assert window.output_tokens == 20


def test_sa_bench_window_rejects_missing_field(tmp_path: Path) -> None:
    result_path = tmp_path / "results_concurrency_1_gpus_4.json"
    result_path.write_text(json.dumps({"benchmark_start_time_unix": 100.0}))

    with pytest.raises(PowerReportError, match="benchmark_end_time_unix"):
        sa_bench_window(1, result_path)


# ---------------------------------------------------------------------------
# GPU roles
# ---------------------------------------------------------------------------


def test_load_gpu_roles_and_per_role_aggregation(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "expected_devices": [
                    {"hostname": "node-a", "gpu_index": 0, "assignments": [{"worker_role": "prefill"}]},
                    {"hostname": "node-a", "gpu_index": 1, "assignments": [{"worker_role": "decode"}]},
                ]
            }
        )
    )
    roles = load_gpu_roles(manifest_path)
    assert roles == {("node-a", 0): {"prefill"}, ("node-a", 1): {"decode"}}

    gpu_csv = tmp_path / "samples.csv"
    with gpu_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["schema_version", "timestamp_unix", "scrape_seq", "hostname", "gpu_index", "gpu_uuid", "power_w"]
        )
        writer.writerow([1, 10.0, 0, "node-a", 0, "GPU-a", 100.0])
        writer.writerow([1, 10.0, 0, "node-a", 1, "GPU-b", 50.0])
        writer.writerow([1, 11.0, 1, "node-a", 0, "GPU-a", 110.0])
        writer.writerow([1, 11.0, 1, "node-a", 1, "GPU-b", 55.0])

    samples = load_gpu_samples(gpu_csv, roles)

    _node_times, node_watts = samples.per_node["node-a"]
    assert list(node_watts) == [150.0, 165.0]  # summed across both GPUs at each shared timestamp

    _prefill_times, prefill_watts = samples.per_role["prefill"]["node-a"]
    assert list(prefill_watts) == [100.0, 110.0]  # only gpu0, which is solely "prefill"


# ---------------------------------------------------------------------------
# Discovery + end-to-end
# ---------------------------------------------------------------------------


def test_discover_run_disambiguates_cpu_and_gpu_by_parent_dir(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    (log_dir / "power" / "cpu").mkdir(parents=True)
    (log_dir / "power" / "cpu" / "samples.csv").write_text("h\n")
    (log_dir / "power" / "samples.csv").write_text("h\n")
    (log_dir / "benchmark.out").write_text("Successful requests: 1\n")
    conc_dir = log_dir / "sa-bench_isl_1_osl_1"
    conc_dir.mkdir()
    (conc_dir / "results_concurrency_1_gpus_1.json").write_text(
        json.dumps(
            {
                "benchmark_start_time_unix": 1.0,
                "benchmark_end_time_unix": 2.0,
                "total_input_tokens": 1,
                "total_output_tokens": 1,
            }
        )
    )

    paths = discover_run(log_dir)

    assert paths.cpu_samples_csv == log_dir / "power" / "cpu" / "samples.csv"
    assert paths.gpu_samples_csv == log_dir / "power" / "samples.csv"
    assert paths.concurrency_sources == ((1, conc_dir / "results_concurrency_1_gpus_1.json"),)


def test_sa_bench_glob_ignores_power_windows_directory(tmp_path: Path) -> None:
    """power/windows/results_concurrency_*.json shares a filename with the real
    result file but lacks token fields; it must never be picked up as a source."""
    log_dir = tmp_path / "logs"
    (log_dir / "power" / "windows").mkdir(parents=True)
    (log_dir / "power" / "windows" / "results_concurrency_1_gpus_1.json").write_text(
        json.dumps({"benchmark_type": "sa-bench", "concurrency": 1})
    )
    (log_dir / "power" / "cpu").mkdir(parents=True)
    (log_dir / "power" / "cpu" / "samples.csv").write_text("h\n")
    (log_dir / "benchmark.out").write_text("Successful requests: 1\n")
    conc_dir = log_dir / "sa-bench_isl_1_osl_1"
    conc_dir.mkdir()
    real_result = conc_dir / "results_concurrency_1_gpus_1.json"
    real_result.write_text(
        json.dumps(
            {
                "benchmark_start_time_unix": 1.0,
                "benchmark_end_time_unix": 2.0,
                "total_input_tokens": 1,
                "total_output_tokens": 1,
            }
        )
    )

    paths = discover_run(log_dir)

    assert paths.concurrency_sources == ((1, real_result),)


def test_build_reports_end_to_end_aiperf(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    (log_dir / "benchmark.out").write_text(
        "17:59:31.680 NOTICE   Phase profiling (profiling) started (runner.py:593)\n"
        "19:00:01.681 NOTICE   Phase profiling (profiling) complete (runner.py:1162)\n"
    )

    conc_dir = log_dir / "agentic" / "conc_4" / "aiperf_artifacts"
    conc_dir.mkdir(parents=True)
    _write_jsonl(
        conc_dir / "profile_export.jsonl",
        [
            {
                "metadata": {
                    "benchmark_phase": "profiling",
                    "request_start_ns": 10_000_000_000,
                    "request_end_ns": 20_000_000_000,
                }
            },
        ],
    )
    (conc_dir / "profile_export_aiperf.json").write_text(
        json.dumps({"total_osl": {"avg": 5.0}, "total_isl": {"avg": 2.0}})
    )

    _write_cpu_csv(
        log_dir / "power" / "cpu" / "samples.csv",
        [
            (2, 9.0, "node-a", "acpi", "CPU0:cpuPowerUsageW", 0, 40.0, 40.0),
            (2, 15.0, "node-a", "acpi", "CPU0:cpuPowerUsageW", 0, 44.0, 44.0),
            (2, 21.0, "node-a", "acpi", "CPU0:cpuPowerUsageW", 0, 42.0, 42.0),
        ],
    )

    reports = build_reports(log_dir)

    assert len(reports) == 1
    report = reports[0]
    assert report.window.concurrency == 4
    assert report.window.output_tokens == 5.0
    assert report.gpu_total_joules == 0.0
    assert report.cpu_total_joules > 0.0
    assert report.joules_per_output_token() == pytest.approx(report.cpu_total_joules / 5.0)


def _write_cpu_csv_v3(path: Path, rows: list[tuple]) -> None:
    """Host-collector layout: timestamp_local plus the five DCGM utilization columns."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "schema_version",
                "timestamp_unix",
                "timestamp_local",
                "hostname",
                "source",
                "sensor",
                "socket_id",
                "power_w",
                "total_power_w",
                "cpu_util_total",
                "cpu_util_user",
                "cpu_util_nice",
                "cpu_util_sys",
                "cpu_util_irq",
            ]
        )
        writer.writerows(rows)


def test_load_cpu_samples_reads_utilization_columns_by_name(tmp_path: Path) -> None:
    path = tmp_path / "cpu_power" / "samples.csv"
    _write_cpu_csv_v3(
        path,
        [
            (3, 10.0, "t", "node-a", "dcgm", "CPU0:cpuPowerUsageW", 0, 40.0, 90.0, 0.50, 0.40, "", 0.10, ""),
            (3, 10.0, "t", "node-a", "dcgm", "CPU1:cpuPowerUsageW", 1, 50.0, 90.0, 0.20, "", "", "", ""),
            (3, 11.0, "t", "node-a", "dcgm", "CPU0:cpuPowerUsageW", 0, 42.0, 92.0, 0.70, 0.60, "", 0.10, ""),
            (3, 11.0, "t", "node-a", "dcgm", "CPU1:cpuPowerUsageW", 1, 50.0, 92.0, "", "", "", "", ""),
        ],
    )

    samples = load_cpu_samples(path)

    util0 = samples.per_socket_utilization[("node-a", 0)]
    assert set(util0) == {"cpu_util_total", "cpu_util_user", "cpu_util_sys"}
    assert list(util0["cpu_util_total"][0]) == [10.0, 11.0]
    assert list(util0["cpu_util_total"][1]) == [0.5, 0.7]
    util1 = samples.per_socket_utilization[("node-a", 1)]
    assert list(util1["cpu_util_total"][1]) == [0.2]  # the blank second sample is skipped, not zero
    # power series are unaffected by the extra columns
    assert list(samples.per_socket[("node-a", 0)][1]) == [40.0, 42.0]


def test_load_cpu_samples_without_utilization_columns_yields_no_utilization(tmp_path: Path) -> None:
    path = tmp_path / "cpu" / "samples.csv"
    _write_cpu_csv(path, [(1, 10.0, "node-a", "acpi", "CPU0:cpuPowerUsageW", 0, 40.0, 40.0)])

    samples = load_cpu_samples(path)

    assert samples.per_socket_utilization == {}


def _write_gpu_csv(path: Path, rows: list[list], *, utilization: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ["schema_version", "timestamp_unix", "scrape_seq", "hostname", "gpu_index", "gpu_uuid", "power_w"]
    if utilization:
        header += ["gpu_util_pct", "sm_active"]
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def test_load_gpu_samples_reads_utilization_columns_and_keeps_roles(tmp_path: Path) -> None:
    gpu_csv = tmp_path / "samples.csv"
    _write_gpu_csv(
        gpu_csv,
        [
            [2, 10.0, 0, "node-a", 0, "GPU-a", 100.0, 80.0, 0.70],
            [2, 10.0, 0, "node-a", 1, "GPU-b", 50.0, 20.0, ""],
            [2, 11.0, 1, "node-a", 0, "GPU-a", 110.0, 90.0, 0.80],
            [2, 11.0, 1, "node-a", 1, "GPU-b", 55.0, "", ""],
        ],
        utilization=True,
    )
    roles = {("node-a", 0): {"prefill"}, ("node-a", 1): {"decode"}}

    samples = load_gpu_samples(gpu_csv, roles)

    util_a = samples.per_device_utilization[("node-a", 0)]
    assert list(util_a["gpu_util_pct"][1]) == [80.0, 90.0]
    assert list(util_a["sm_active"][1]) == [0.7, 0.8]
    util_b = samples.per_device_utilization[("node-a", 1)]
    assert list(util_b["gpu_util_pct"][1]) == [20.0]
    assert "sm_active" not in util_b  # never populated -> absent, not an empty series
    assert samples.device_roles == roles


def test_load_gpu_samples_v1_header_has_no_utilization(tmp_path: Path) -> None:
    gpu_csv = tmp_path / "samples.csv"
    _write_gpu_csv(gpu_csv, [[1, 10.0, 0, "node-a", 0, "GPU-a", 100.0]], utilization=False)

    samples = load_gpu_samples(gpu_csv, None)

    assert samples.per_device_utilization == {}


def test_windowed_utilization_averages_only_samples_inside_the_window() -> None:
    times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    values = np.array([0.1, 0.5, 0.7, 0.9, 0.2])

    summary = windowed_utilization("gpu/node-a/gpu0", "sm_active", times, values, start=1.0, end=3.0)

    assert summary is not None
    assert summary.mean == pytest.approx((0.5 + 0.7 + 0.9) / 3)
    assert summary.max == pytest.approx(0.9)
    assert summary.samples == 3


def test_windowed_utilization_returns_none_without_coverage() -> None:
    times = np.array([100.0, 101.0])
    values = np.array([0.5, 0.5])

    assert windowed_utilization("label", "sm_active", times, values, start=0.0, end=1.0) is None


def test_discover_run_accepts_host_collector_cpu_power_directory(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    (log_dir / "cpu_power" / "nodes").mkdir(parents=True)
    (log_dir / "cpu_power" / "samples.csv").write_text("h\n")
    (log_dir / "cpu_power" / "nodes" / "node-a.csv").write_text("h\n")
    (log_dir / "power").mkdir()
    (log_dir / "power" / "samples.csv").write_text("h\n")
    _write_sa_bench_run(log_dir)

    paths = discover_run(log_dir)

    assert paths.cpu_samples_csv == log_dir / "cpu_power" / "samples.csv"
    assert paths.gpu_samples_csv == log_dir / "power" / "samples.csv"


def test_discover_run_requires_a_choice_when_both_cpu_legs_wrote_samples(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    (log_dir / "cpu_power").mkdir(parents=True)
    (log_dir / "cpu_power" / "samples.csv").write_text("h\n")
    (log_dir / "power" / "cpu").mkdir(parents=True)
    (log_dir / "power" / "cpu" / "samples.csv").write_text("h\n")
    _write_sa_bench_run(log_dir)

    with pytest.raises(PowerReportError, match="--cpu-samples"):
        discover_run(log_dir)

    paths = discover_run(log_dir, cpu_samples_csv=log_dir / "cpu_power" / "samples.csv")
    assert paths.cpu_samples_csv == log_dir / "cpu_power" / "samples.csv"


def test_discover_run_rejects_missing_cpu_samples_override(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    (log_dir / "power").mkdir(parents=True)
    (log_dir / "power" / "samples.csv").write_text("h\n")
    _write_sa_bench_run(log_dir)

    with pytest.raises(PowerReportError, match="not found"):
        discover_run(log_dir, cpu_samples_csv=log_dir / "nope.csv")


def _write_sa_bench_run(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "benchmark.out").write_text("Successful requests: 1\n")
    conc_dir = log_dir / "sa-bench_isl_1_osl_1"
    conc_dir.mkdir(exist_ok=True)
    (conc_dir / "results_concurrency_1_gpus_1.json").write_text(
        json.dumps(
            {
                "benchmark_start_time_unix": 10.0,
                "benchmark_end_time_unix": 20.0,
                "total_input_tokens": 100,
                "total_output_tokens": 50,
            }
        )
    )


def test_build_reports_summarizes_cpu_and_gpu_utilization(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    _write_sa_bench_run(log_dir)
    _write_cpu_csv_v3(
        log_dir / "cpu_power" / "samples.csv",
        [
            (3, 9.0, "t", "node-a", "dcgm", "CPU0:cpuPowerUsageW", 0, 40.0, 40.0, 0.10, "", "", "", ""),
            (3, 15.0, "t", "node-a", "dcgm", "CPU0:cpuPowerUsageW", 0, 44.0, 44.0, 0.50, "", "", "", ""),
            (3, 21.0, "t", "node-a", "dcgm", "CPU0:cpuPowerUsageW", 0, 42.0, 42.0, 0.90, "", "", "", ""),
        ],
    )
    (log_dir / "power").mkdir()
    (log_dir / "power" / "manifest.json").write_text(
        json.dumps(
            {
                "expected_devices": [
                    {"hostname": "node-a", "gpu_index": 0, "assignments": [{"worker_role": "prefill"}]},
                    {"hostname": "node-a", "gpu_index": 1, "assignments": [{"worker_role": "prefill"}]},
                ]
            }
        )
    )
    _write_gpu_csv(
        log_dir / "power" / "samples.csv",
        [
            [2, 9.0, 0, "node-a", 0, "GPU-a", 100.0, 10.0, ""],
            [2, 9.0, 0, "node-a", 1, "GPU-b", 100.0, 30.0, ""],
            [2, 15.0, 1, "node-a", 0, "GPU-a", 100.0, 60.0, ""],
            [2, 15.0, 1, "node-a", 1, "GPU-b", 100.0, 80.0, ""],
            [2, 21.0, 2, "node-a", 0, "GPU-a", 100.0, 0.0, ""],
            [2, 21.0, 2, "node-a", 1, "GPU-b", 100.0, 0.0, ""],
        ],
        utilization=True,
    )

    report = build_reports(log_dir)[0]

    cpu = {(u.label, u.column): u for u in report.cpu_utilization}
    assert cpu[("cpu/node-a/socket0", "cpu_util_total")].mean == pytest.approx(0.5)  # only t=15 is inside [10, 20]
    assert cpu[("cpu/node-a", "cpu_util_total")].mean == pytest.approx(0.5)
    gpu = {(u.label, u.column): u for u in report.gpu_utilization}
    assert gpu[("gpu/node-a/gpu0", "gpu_util_pct")].mean == pytest.approx(60.0)
    assert gpu[("gpu/node-a/gpu1", "gpu_util_pct")].mean == pytest.approx(80.0)
    assert gpu[("gpu/node-a", "gpu_util_pct")].mean == pytest.approx(70.0)  # mean across the node's devices
    assert gpu[("gpu/node-a/prefill", "gpu_util_pct")].mean == pytest.approx(70.0)
    assert ("gpu/node-a/gpu0", "sm_active") not in gpu  # column present but never populated -> not reported
    assert report.gpu_total_joules > 0.0

    text = render_table([report])
    assert "gpu/node-a/gpu0 utilization: gpu_util_pct mean=60.00 max=60.00" in text
    assert "cpu/node-a/socket0 utilization: cpu_util_total mean=0.50 max=0.50" in text
    payload = report_to_dict(report)
    assert {"label": "gpu/node-a", "column": "gpu_util_pct", "mean": 70.0, "max": 80.0, "samples": 2} in payload[
        "gpu_utilization"
    ]
    assert any(u["column"] == "cpu_util_total" for u in payload["cpu_utilization"])


def test_build_reports_warns_when_utilization_has_no_window_coverage(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    _write_sa_bench_run(log_dir)
    _write_cpu_csv_v3(
        log_dir / "cpu_power" / "samples.csv",
        [
            (3, 9.0, "t", "node-a", "dcgm", "CPU0:cpuPowerUsageW", 0, 40.0, 40.0, 0.10, "", "", "", ""),
            (3, 15.0, "t", "node-a", "dcgm", "CPU0:cpuPowerUsageW", 0, 44.0, 44.0, "", "", "", "", ""),
            (3, 21.0, "t", "node-a", "dcgm", "CPU0:cpuPowerUsageW", 0, 42.0, 42.0, 0.90, "", "", "", ""),
        ],
    )

    report = build_reports(log_dir)[0]

    assert report.cpu_utilization == ()
    assert any(
        "cpu/node-a/socket0 cpu_util_total" in warning and "no samples" in warning for warning in report.warnings
    )


# ---------------------------------------------------------------------------
# Reported timing (comparison only) + perf/W
# ---------------------------------------------------------------------------


def _aiperf_case(tmp_path: Path, aggregate_extra: dict) -> Path:
    conc_dir = tmp_path / "conc_8" / "aiperf_artifacts"
    jsonl_path = conc_dir / "profile_export.jsonl"
    _write_jsonl(
        jsonl_path,
        [
            {
                "metadata": {
                    "benchmark_phase": "profiling",
                    "request_start_ns": 1_000_000_000_000,
                    "request_end_ns": 1_100_000_000_000,
                }
            },
        ],
    )
    (conc_dir / "profile_export_aiperf.json").write_text(
        json.dumps({"total_osl": {"avg": 42.0}, "total_isl": {"avg": 7.0}, **aggregate_extra})
    )
    return jsonl_path


def test_aiperf_window_reads_reported_timing_from_aggregate_json(tmp_path: Path) -> None:
    jsonl_path = _aiperf_case(
        tmp_path,
        {
            "benchmark_duration": {"unit": "sec", "avg": 99.5},
            "start_time": "2026-09-06T14:32:00.646759",
            "end_time": "2026-09-06T14:33:40.146759",
        },
    )

    window = aiperf_window(8, jsonl_path)

    assert window.duration_seconds == pytest.approx(100.0)
    reported = window.reported
    assert reported.source == "aiperf-json"
    assert reported.duration_seconds == pytest.approx(99.5)
    # naive ISO stamps are aiperf's local wall clock; the report keeps them as a
    # local-time conversion and never treats them as authoritative.
    assert reported.end_unix - reported.start_unix == pytest.approx(99.5)


def test_aiperf_window_falls_back_to_a_single_phase_log_pair(tmp_path: Path) -> None:
    jsonl_path = _aiperf_case(tmp_path, {})
    benchmark_out = tmp_path / "benchmark.out"
    benchmark_out.write_text(
        "22:09:07.432 NOTICE   Phase profiling (profiling) started | phase_index=0 | target: 1200.0s duration (runner.py:593)\n"
        "22:29:07.432 NOTICE   Phase profiling (profiling) sending complete | sent=63 (runner.py:1039)\n"
        "22:29:07.932 NOTICE   Phase profiling (profiling) complete | completed=63 | elapsed=1200.50s (runner.py:1162)\n"
    )

    window = aiperf_window(8, jsonl_path, benchmark_out=benchmark_out)

    assert window.reported.source == "aiperf-phase-log"
    assert window.reported.duration_seconds == pytest.approx(1200.5)
    assert window.reported.start_unix is None  # time-of-day stamps carry no date
    assert "22:09:07.432" in window.reported.note and "22:29:07.932" in window.reported.note


def test_phase_log_fallback_derives_duration_from_stamps_when_elapsed_is_absent(tmp_path: Path) -> None:
    benchmark_out = tmp_path / "benchmark.out"
    benchmark_out.write_text(
        "23:59:00.000 NOTICE   Phase profiling (profiling) started (runner.py:593)\n"
        "00:01:00.000 NOTICE   Phase profiling (profiling) complete | completed=1 (runner.py:1162)\n"
    )

    reported = reported_timing_from_phase_log(benchmark_out)

    assert reported.source == "aiperf-phase-log"
    assert reported.duration_seconds == pytest.approx(120.0)  # wraps midnight


def test_phase_log_fallback_is_unavailable_when_ambiguous(tmp_path: Path) -> None:
    benchmark_out = tmp_path / "benchmark.out"
    benchmark_out.write_text(
        "10:00:00.000 NOTICE   Phase profiling (profiling) started (runner.py:593)\n"
        "10:10:00.000 NOTICE   Phase profiling (profiling) complete | elapsed=600.00s (runner.py:1162)\n"
        "10:20:00.000 NOTICE   Phase profiling (profiling) started (runner.py:593)\n"
        "10:30:00.000 NOTICE   Phase profiling (profiling) complete | elapsed=600.00s (runner.py:1162)\n"
    )

    reported = reported_timing_from_phase_log(benchmark_out)

    assert reported.source == "unavailable"
    assert reported.duration_seconds is None
    assert "2 profiling phases" in reported.note


def test_aiperf_window_without_any_reported_source_is_unavailable(tmp_path: Path) -> None:
    jsonl_path = _aiperf_case(tmp_path, {})

    window = aiperf_window(8, jsonl_path)

    assert window.reported.source == "unavailable"
    assert window.reported.duration_seconds is None


def test_sa_bench_window_reports_its_own_duration(tmp_path: Path) -> None:
    result_json = tmp_path / "results_concurrency_4_gpus_8.json"
    result_json.write_text(
        json.dumps(
            {
                "benchmark_start_time_unix": 100.0,
                "benchmark_end_time_unix": 160.0,
                "duration": 59.7,
                "total_input_tokens": 1000,
                "total_output_tokens": 500,
            }
        )
    )

    window = sa_bench_window(4, result_json)

    assert window.duration_seconds == pytest.approx(60.0)
    assert window.reported.source == "sa-bench-json"
    assert window.reported.duration_seconds == pytest.approx(59.7)
    assert window.reported.start_unix == 100.0 and window.reported.end_unix == 160.0


def test_windowed_energy_records_the_samples_it_actually_integrated() -> None:
    times = np.array([0.0, 0.9, 2.1, 3.0])
    watts = np.array([10.0, 10.0, 10.0, 10.0])

    result = windowed_energy("label", times, watts, start=1.0, end=2.0)

    assert result.sample_start_unix == 0.9
    assert result.sample_end_unix == 2.1
    assert result.samples == 2


def _window(*, output_tokens: float = 500.0, input_tokens: float = 1500.0) -> ConcurrencyWindow:
    return ConcurrencyWindow(
        benchmark_type="sa-bench",
        concurrency=4,
        start_unix=100.0,
        end_unix=110.0,
        output_tokens=output_tokens,
        input_tokens=input_tokens,
        source=Path("results.json"),
    )


def _flat_series(watts: float) -> tuple[np.ndarray, np.ndarray]:
    times = np.arange(99.0, 112.0, 1.0)
    return times, np.full(len(times), watts)


def test_perf_per_watt_is_tokens_per_joule_and_needs_both_legs_for_combined() -> None:
    from srtctl.analysis.power_energy_report import CpuSamples, GpuSamples

    gpu = GpuSamples(
        per_device={("node-a", 0): _flat_series(400.0)},
        per_node={"node-a": _flat_series(400.0)},
        per_role={},
    )

    gpu_only = build_concurrency_report(_window(), None, gpu)

    assert gpu_only.duration_seconds == pytest.approx(10.0)
    assert gpu_only.output_tokens_per_second == pytest.approx(50.0)
    assert gpu_only.total_tokens_per_second == pytest.approx(200.0)
    assert gpu_only.gpu_avg_power_w == pytest.approx(400.0)
    assert gpu_only.output_tokens_per_second_per_gpu_watt == pytest.approx(50.0 / 400.0)
    assert gpu_only.output_tokens_per_second_per_gpu_watt == pytest.approx(1.0 / gpu_only.joules_per_output_token())
    assert gpu_only.total_tokens_per_second_per_gpu_watt == pytest.approx(200.0 / 400.0)
    assert gpu_only.combined_avg_power_w is None
    assert gpu_only.output_tokens_per_second_per_combined_watt is None
    assert any("combined perf/W unavailable" in w for w in gpu_only.warnings)

    cpu = CpuSamples(per_socket={("node-a", 0): _flat_series(100.0)}, per_node={"node-a": _flat_series(100.0)})
    both = build_concurrency_report(_window(), cpu, gpu)

    assert both.combined_avg_power_w == pytest.approx(500.0)
    assert both.output_tokens_per_second_per_combined_watt == pytest.approx(50.0 / 500.0)
    assert both.total_tokens_per_second_per_combined_watt == pytest.approx(200.0 / 500.0)
    assert not any("combined perf/W" in w for w in both.warnings)
    assert both.coverage_start_unix == 100.0 and both.coverage_end_unix == 110.0


def test_perf_per_watt_is_none_without_a_gpu_leg() -> None:
    from srtctl.analysis.power_energy_report import CpuSamples

    cpu = CpuSamples(per_socket={("node-a", 0): _flat_series(100.0)}, per_node={"node-a": _flat_series(100.0)})

    report = build_concurrency_report(_window(), cpu, None)

    assert report.gpu_avg_power_w is None
    assert report.output_tokens_per_second_per_gpu_watt is None
    assert report.total_tokens_per_second_per_gpu_watt is None


def test_render_and_json_carry_timing_and_perf_per_watt() -> None:
    from srtctl.analysis.power_energy_report import GpuSamples, ReportedTiming

    window = ConcurrencyWindow(
        benchmark_type="sa-bench",
        concurrency=4,
        start_unix=100.0,
        end_unix=110.0,
        output_tokens=500.0,
        input_tokens=1500.0,
        source=Path("results.json"),
        reported=ReportedTiming(source="sa-bench-json", start_unix=100.0, end_unix=110.0, duration_seconds=9.8),
    )
    gpu = GpuSamples(
        per_device={("node-a", 0): _flat_series(400.0)}, per_node={"node-a": _flat_series(400.0)}, per_role={}
    )
    report = build_concurrency_report(window, None, gpu)

    text = render_table([report])
    assert "timing: computed start=100.000 end=110.000 duration=10.00s" in text
    assert "reported[sa-bench-json] start=100.000 end=110.000 duration=9.80s (computed-reported=+0.20s)" in text
    assert "samples cover 100.000..110.000 (10.00s)" in text
    assert "perf/W: output=50.00 tok/s total=200.00 tok/s | gpu avg=400.00 W" in text
    assert "output 0.1250 tok/s/W" in text and "total 0.5000 tok/s/W" in text
    assert "combined: n/a" in text

    payload = report_to_dict(report)
    assert payload["timing"]["computed"] == {"start_unix": 100.0, "end_unix": 110.0, "duration_seconds": 10.0}
    assert payload["timing"]["reported"]["source"] == "sa-bench-json"
    assert payload["timing"]["reported"]["duration_seconds"] == 9.8
    assert payload["timing"]["coverage"]["duration_seconds"] == 10.0
    assert payload["perf_per_watt"]["output_tokens_per_second_per_gpu_watt"] == pytest.approx(0.125)
    assert payload["perf_per_watt"]["output_tokens_per_second_per_combined_watt"] is None
    assert payload["gpu_per_device"][0]["samples"] == 11
    assert payload["gpu_per_device"][0]["sample_start_unix"] == 100.0


def test_concurrency_report_defaults_keep_older_constructors_working() -> None:
    report = ConcurrencyReport(window=_window())

    assert report.gpu_avg_power_w is None
    assert report.combined_avg_power_w is None
    assert report.coverage_start_unix is None
    assert EnergyBreakdown(label="x", joules=1.0, avg_power_w=1.0).samples == 0


# ---------------------------------------------------------------------------
# Power percentile stats
# ---------------------------------------------------------------------------


def test_windowed_energy_reports_sample_based_power_stats() -> None:
    times = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    watts = np.array([100.0, 10.0, 20.0, 30.0, 40.0, 100.0])

    result = windowed_energy("label", times, watts, start=1.0, end=4.0)

    inside = watts[1:5]  # 10, 20, 30, 40 -- the same samples the trapezoid spanned
    assert result.samples == 4
    assert result.min_w == 10.0
    assert result.max_w == 40.0
    assert result.mean_w == pytest.approx(25.0)
    assert result.p50_w == pytest.approx(np.percentile(inside, 50))
    assert result.p95_w == pytest.approx(np.percentile(inside, 95))
    assert result.p99_w == pytest.approx(np.percentile(inside, 99))
    assert result.p5_w == pytest.approx(np.percentile(inside, 5))
    # time-weighted average (joules / duration) and sample mean coincide on a uniform grid
    assert result.avg_power_w == pytest.approx(result.mean_w)


def test_time_weighted_average_and_sample_mean_diverge_on_uneven_sampling() -> None:
    # Three quick low samples, then one long high ramp: the ramp dominates the
    # energy (trapezoid 10 + 10 + 440 = 460 J) but is a single sample point.
    times = np.array([0.0, 1.0, 2.0, 10.0])
    watts = np.array([10.0, 10.0, 10.0, 100.0])

    result = windowed_energy("label", times, watts, start=0.0, end=10.0)

    assert result.mean_w == pytest.approx(32.5)
    assert result.avg_power_w == pytest.approx(46.0)
    assert result.avg_power_w == pytest.approx(result.joules / 10.0)


def test_node_percentiles_come_from_the_summed_series() -> None:
    from srtctl.analysis.power_energy_report import GpuSamples

    times = np.arange(99.0, 112.0, 1.0)
    gpu0 = np.where(times % 2 == 0, 100.0, 300.0)
    gpu1 = np.where(times % 2 == 0, 300.0, 100.0)
    gpu = GpuSamples(
        per_device={("node-a", 0): (times, gpu0), ("node-a", 1): (times, gpu1)},
        per_node={"node-a": (times, gpu0 + gpu1)},
        per_role={},
    )

    report = build_concurrency_report(_window(), None, gpu)

    node = report.gpu_per_node[0]
    assert node.min_w == node.max_w == 400.0  # anti-phase devices sum flat; not 2 x per-device p99
    assert report.gpu_per_device[0].max_w == 300.0


def test_render_and_json_carry_power_percentiles() -> None:
    from srtctl.analysis.power_energy_report import GpuSamples

    gpu = GpuSamples(
        per_device={("node-a", 0): _flat_series(400.0)}, per_node={"node-a": _flat_series(400.0)}, per_role={}
    )
    report = build_concurrency_report(_window(), None, gpu)

    text = render_table([report])
    assert "gpu/node-a/gpu0: 4,000.00 J (400.00 W avg; p50=400.00 p95=400.00 p99=400.00 max=400.00 W)" in text

    entry = report_to_dict(report)["gpu_per_device"][0]
    for key in ("mean_w", "min_w", "p5_w", "p50_w", "p95_w", "p99_w", "max_w"):
        assert entry[key] == pytest.approx(400.0), key
    assert EnergyBreakdown(label="x", joules=1.0, avg_power_w=1.0).p99_w is None
