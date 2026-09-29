# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the self-contained HTML power/perf report."""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from srtctl.analysis.power_energy_report import PowerReportError
from srtctl.analysis.power_report_html import (
    _build_run_series,
    _dedupe_labels,
    _downsample_minmax,
    _family_label,
    _load_run_family,
    _load_run_topology,
    _model_select_html,
    _pareto_points,
    _phase_bands,
    build,
    build_combined,
    build_combined_report,
    build_report,
    main,
)

# ---------------------------------------------------------------------------
# Downsampling
# ---------------------------------------------------------------------------


def test_downsample_passes_through_short_series_unchanged() -> None:
    times = np.arange(10, dtype=float)
    watts = np.arange(10, dtype=float)

    out_t, out_w = _downsample_minmax(times, watts, max_buckets=100)

    assert list(out_t) == list(times)
    assert list(out_w) == list(watts)


def test_downsample_caps_point_count_and_keeps_the_peak() -> None:
    n = 10_000
    times = np.arange(n, dtype=float)
    watts = np.zeros(n)
    watts[n // 2] = 999.0  # a lone spike a stride/average sample would likely miss

    out_t, out_w = _downsample_minmax(times, watts, max_buckets=50)

    assert len(out_t) <= 100
    assert out_w.max() == pytest.approx(999.0)


# ---------------------------------------------------------------------------
# Run series building
# ---------------------------------------------------------------------------


def _series(watts: list[float]) -> tuple[np.ndarray, np.ndarray]:
    return np.arange(len(watts), dtype=float), np.array(watts, dtype=float)


def test_build_run_series_sorted_by_host_then_index() -> None:
    per_device = {
        ("node-b", 0): _series([10.0, 12.0]),
        ("node-a", 1): _series([20.0, 22.0]),
        ("node-a", 0): _series([30.0, 32.0]),
    }

    series = _build_run_series(per_device, label_fmt="{host}/gpu{index}")

    assert [s["label"] for s in series] == ["node-a/gpu0", "node-a/gpu1", "node-b/gpu0"]


def test_build_run_series_colors_by_host_and_shades_by_device() -> None:
    per_device = {(host, i): _series([float(i)] * 3) for host in ("node-a", "node-b") for i in range(2)}

    series = _build_run_series(per_device, label_fmt="{host}/gpu{index}")

    by_label = {s["label"]: s for s in series}
    a0, a1, b0 = by_label["node-a/gpu0"], by_label["node-a/gpu1"], by_label["node-b/gpu0"]
    # same host -> same hue, different lightness; different host -> different hue
    assert a0["color"].split()[0] == a1["color"].split()[0]
    assert a0["color"] != a1["color"]
    assert a0["color"].split()[0] != b0["color"].split()[0]
    assert all(s["pattern"] == 0 for s in series)


def test_build_run_series_cycles_stroke_pattern_past_four_devices_on_one_host() -> None:
    per_device = {("node-a", i): _series([float(i)] * 3) for i in range(5)}

    series = _build_run_series(per_device, label_fmt="{host}/gpu{index}")

    assert [s["pattern"] for s in series] == [0, 0, 0, 0, 1]


def test_build_run_series_window_slices_samples_and_shifts_to_window_start() -> None:
    per_device = {
        ("node-a", 0): (np.array([100.0, 105.0, 110.0, 115.0]), np.array([1.0, 2.0, 3.0, 4.0])),
        ("node-a", 1): (np.array([200.0, 201.0]), np.array([9.0, 9.0])),  # entirely outside the window
    }

    series = _build_run_series(per_device, label_fmt="{host}/gpu{index}", window=(104.0, 111.0))

    assert [s["label"] for s in series] == ["node-a/gpu0"]
    assert series[0]["t"] == [1.0, 6.0]
    assert series[0]["w"] == [2.0, 3.0]
    assert series[0]["stats"]["max"] == 3.0  # stats reflect the window, not the whole run


def test_build_run_series_time_shifts_to_the_earliest_sample_across_all_devices() -> None:
    per_device = {
        ("node-a", 0): (np.array([100.0, 101.0]), np.array([1.0, 2.0])),
        ("node-a", 1): (np.array([105.0, 106.0]), np.array([3.0, 4.0])),
    }

    series = _build_run_series(per_device, label_fmt="{host}/gpu{index}")

    assert series[0]["t"] == [0.0, 1.0]
    assert series[1]["t"] == [5.0, 6.0]


def test_build_run_series_tags_roles_from_manifest_and_host_fallback() -> None:
    per_device = {("node-a", 0): _series([1.0]), ("node-a", 1): _series([1.0]), ("node-b", 0): _series([1.0])}
    roles = {("node-a", 0): {"prefill"}, ("node-a", 1): {"decode"}}

    gpu = _build_run_series(per_device, label_fmt="{host}/gpu{index}", roles=roles)
    cpu = _build_run_series(
        {("node-a", 0): _series([1.0])}, label_fmt="{host}/socket{index}", host_roles={"node-a": {"prefill", "decode"}}
    )

    assert [s["roles"] for s in gpu] == [["prefill"], ["decode"], []]  # node-b has no manifest entry
    assert cpu[0]["roles"] == ["decode", "prefill"]  # socket inherits every role on its host


def test_build_run_series_origin_overrides_the_time_zero() -> None:
    per_device = {("node-a", 0): (np.array([105.0, 106.0]), np.array([1.0, 2.0]))}

    series = _build_run_series(per_device, label_fmt="{host}/gpu{index}", origin=100.0)

    assert series[0]["t"] == [5.0, 6.0]


def test_build_run_series_empty_input_returns_no_series() -> None:
    assert _build_run_series({}, label_fmt="{host}/gpu{index}") == []


# ---------------------------------------------------------------------------
# Pareto points
# ---------------------------------------------------------------------------


def _report_dict(
    *,
    concurrency: int,
    output_tps: float | None,
    tps_per_gpu: float | None,
    num_gpus: int = 2,
) -> dict:
    return {
        "benchmark_type": "aiperf",
        "concurrency": concurrency,
        "start_unix": 10.0,
        "end_unix": 70.0,
        "tpot_p50_ms": 8.0,
        "tpot_p90_ms": 12.0,
        "joules_per_output_token": 3.0,
        "timing": {"computed": {"duration_seconds": 60.0}},
        "perf_per_watt": {
            "output_tokens_per_second": output_tps,
            "total_tokens_per_second": None if output_tps is None else output_tps * 1.4,
            "output_tokens_per_second_per_gpu": tps_per_gpu,
            "num_gpus": num_gpus,
            "gpu_avg_power_w": 300.0,
            "cpu_avg_power_w": 40.0,
            "combined_avg_power_w": 340.0,
            "output_tokens_per_second_per_gpu_watt": 0.16,
            "output_tokens_per_second_per_combined_watt": 0.14,
        },
    }


def test_pareto_points_extracts_metrics_and_label() -> None:
    reports = [_report_dict(concurrency=4, output_tps=100.0, tps_per_gpu=50.0)]

    points = _pareto_points(reports, run_label="runA")

    assert len(points) == 1
    p = points[0]
    assert p["label"] == "runA · aiperf c=4"
    assert p["id"] == "runA::aiperf::c4"
    assert p["run"] == "runA"
    assert p["m"]["output_tps"] == 100.0
    assert p["m"]["tps_per_gpu"] == 50.0
    assert p["m"]["total_tps_per_gpu"] == pytest.approx(70.0)
    assert p["m"]["inv_tpot_p90"] == pytest.approx(1000.0 / 12.0)
    assert p["m"]["concurrency"] == 4


def test_pareto_points_color_follows_run_position() -> None:
    reports = [_report_dict(concurrency=4, output_tps=100.0, tps_per_gpu=50.0)]

    a = _pareto_points(reports, run_label="runA", run_position=0)[0]["color"]
    b = _pareto_points(reports, run_label="runB", run_position=1)[0]["color"]

    assert a != b


def test_pareto_points_skips_rows_with_no_gpu_count() -> None:
    reports = [
        _report_dict(concurrency=4, output_tps=100.0, tps_per_gpu=50.0),
        _report_dict(concurrency=8, output_tps=None, tps_per_gpu=None, num_gpus=0),
    ]

    points = _pareto_points(reports, run_label="runA")

    assert len(points) == 1
    assert points[0]["label"] == "runA · aiperf c=4"


def test_power_scatter_keeps_only_points_with_both_power_legs() -> None:
    from srtctl.analysis.power_report_html import _power_scatter_html

    reports = [_report_dict(concurrency=4, output_tps=100.0, tps_per_gpu=50.0)]
    points = _pareto_points(reports, run_label="runA")
    points[0]["m"]["cpu_w"] = None
    assert "No concurrency points have both" in _power_scatter_html(points)

    points = _pareto_points(reports, run_label="runA")
    content = _power_scatter_html(points)
    assert 'data-x="cpu_w" data-y="gpu_w" data-frontier="off"' in content
    assert points[0]["m"]["cpu_w"] == 40.0
    assert points[0]["m"]["gpu_w"] == 300.0


def test_baseline_view_needs_two_runs_and_lists_them_as_options() -> None:
    from srtctl.analysis.power_report_html import _baseline_view_html

    reports = [_report_dict(concurrency=4, output_tps=100.0, tps_per_gpu=50.0)]
    one_run = _pareto_points(reports, run_label="runA")
    assert "at least two runs" in _baseline_view_html(one_run)

    two_runs = one_run + _pareto_points(reports, run_label="runB", run_position=1)
    content = _baseline_view_html(two_runs)
    assert '<option value="runA" selected>' in content
    assert '<option value="runB">' in content
    assert "<select data-baseline>" in content
    assert all(p["bench"] == "aiperf" for p in two_runs)  # what the JS matches baseline points on


def test_pareto_points_omits_run_label_prefix_when_none() -> None:
    reports = [_report_dict(concurrency=4, output_tps=100.0, tps_per_gpu=50.0)]

    points = _pareto_points(reports)

    assert points[0]["label"] == "aiperf c=4"


def test_pareto_points_includes_panel_and_hover_fields() -> None:
    reports = [_report_dict(concurrency=4, output_tps=100.0, tps_per_gpu=50.0)]

    points = _pareto_points(reports, run_label="runA")

    field_names = [name for name, _ in points[0]["fields"]]
    assert "Concurrency / active GPUs" in field_names
    assert "Output tok/s / active GPU" in field_names
    assert "P90 TPOT" in field_names
    assert "Total GPU watts (all GPUs, avg)" in field_names
    assert "Watts per GPU (avg)" in field_names
    assert "Output tok/s / (GPU+CPU) W" in field_names
    assert "Measured window" in field_names
    hover_names = [name for name, _ in points[0]["hover"]]
    assert hover_names == [
        "GPU type / hosts",
        "Concurrency / GPUs",
        "P90 TPOT",
        "Total GPU watts",
        "Watts per GPU",
        "Output tok/s per GPU watt",
    ]
    assert points[0]["m"]["gpu_w_per_gpu"] == 150.0  # 300 W across 2 GPUs


# ---------------------------------------------------------------------------
# End-to-end report build
# ---------------------------------------------------------------------------


def _write_gpu_csv(path: Path, rows: list[tuple]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["schema_version", "timestamp_unix", "scrape_seq", "hostname", "gpu_index", "gpu_uuid", "power_w"]
        )
        writer.writerows(rows)


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


def _write_aiperf_conc(log_dir: Path, concurrency: int) -> None:
    conc_dir = log_dir / "agentic" / f"conc_{concurrency}" / "aiperf_artifacts"
    conc_dir.mkdir(parents=True)
    with (conc_dir / "profile_export.jsonl").open("w") as handle:
        handle.write(
            json.dumps(
                {
                    "metadata": {
                        "benchmark_phase": "profiling",
                        "request_start_ns": 10_000_000_000,
                        "request_end_ns": 20_000_000_000,
                    }
                }
            )
            + "\n"
        )
    (conc_dir / "profile_export_aiperf.json").write_text(
        json.dumps(
            {"total_osl": {"avg": 5.0}, "total_isl": {"avg": 2.0}, "inter_token_latency": {"p50": 8.0, "p90": 12.0}}
        )
    )


def _write_aiperf_run(log_dir: Path, *, concurrencies: tuple[int, ...] = (4,)) -> None:
    (log_dir / "benchmark.out").write_text(
        "17:59:31.680 NOTICE   Phase profiling (profiling) started (runner.py:593)\n"
        "19:00:01.681 NOTICE   Phase profiling (profiling) complete (runner.py:1162)\n"
    )
    for concurrency in concurrencies:
        _write_aiperf_conc(log_dir, concurrency)


def test_build_report_renders_summary_table_and_charts(tmp_path: Path) -> None:
    log_dir = tmp_path
    _write_aiperf_run(log_dir)
    _write_gpu_csv(
        log_dir / "power" / "samples.csv",
        [
            (1, 9.0, 1, "node-a", 0, "GPU-a", 100.0),
            (1, 15.0, 2, "node-a", 0, "GPU-a", 110.0),
            (1, 21.0, 3, "node-a", 0, "GPU-a", 105.0),
        ],
    )
    _write_cpu_csv(
        log_dir / "power" / "cpu" / "samples.csv",
        [
            (2, 9.0, "node-a", "acpi", "CPU0:cpuPowerUsageW", 0, 40.0, 40.0),
            (2, 15.0, "node-a", "acpi", "CPU0:cpuPowerUsageW", 0, 44.0, 44.0),
            (2, 21.0, "node-a", "acpi", "CPU0:cpuPowerUsageW", 0, 42.0, 42.0),
        ],
    )

    content = build_report(log_dir)

    assert "Throughput &amp; power by concurrency" in content
    assert "Power over time" in content
    assert "GPU power (W)" in content
    assert "CPU socket power (W)" in content
    assert "node-a/gpu0" in content
    assert "node-a/socket0" in content
    # GPU and CPU are stacked in one chart group so the crosshair/zoom span both.
    # Single run, single concurrency -> one per-concurrency card on the data table
    # (which is also the run's series source) plus the Pareto page's whole-run copy.
    assert content.count('class="conc-card"') == 1
    assert content.count('data-source-id="run-src-0"') == 1
    assert content.count('data-focus-bench="aiperf" data-focus-conc="4"') == 1
    # a single-host run gets no host legend (nothing to toggle between)
    assert 'class="legend host-legend"' not in content
    # KPI stat cards in the header
    assert "stat-card-num" in content
    assert "GPUs" in content
    assert "CPU sockets" in content


def test_host_legend_lists_each_host_once_across_both_legs() -> None:
    from srtctl.analysis.power_report_html import _host_legend_html

    gpu = _build_run_series(
        {(h, i): _series([1.0, 2.0]) for h in ("node-a", "node-b") for i in range(2)}, label_fmt="{host}/gpu{index}"
    )
    cpu = _build_run_series({("node-a", 0): _series([1.0, 2.0])}, label_fmt="{host}/socket{index}")

    content = _host_legend_html(gpu, cpu)

    assert content.count('class="legend-key host-key"') == 2
    assert 'data-host="node-a"' in content
    assert 'data-host="node-b"' in content


def test_build_report_omits_pareto_tab_with_a_single_concurrency_point(tmp_path: Path) -> None:
    log_dir = tmp_path
    _write_aiperf_run(log_dir)
    _write_gpu_csv(
        log_dir / "power" / "samples.csv",
        [(1, 9.0, 1, "node-a", 0, "GPU-a", 100.0), (1, 21.0, 3, "node-a", 0, "GPU-a", 105.0)],
    )

    content = build_report(log_dir)

    assert "Pareto view" not in content
    # ... but the per-node power bars still appear, self-rendered from the embedded point
    assert content.count('class="pareto-card node-power-card" data-point=') == 1
    assert "node_power" in content
    assert (
        content.index("<table>")
        < content.index('class="pareto-card node-power-card"')
        < content.index('class="conc-card"')
    )


def test_build_report_includes_pareto_tab_with_multiple_concurrency_points(tmp_path: Path) -> None:
    log_dir = tmp_path
    _write_aiperf_run(log_dir, concurrencies=(4, 8))
    _write_gpu_csv(
        log_dir / "power" / "samples.csv",
        [
            (1, 9.0, 1, "node-a", 0, "GPU-a", 100.0),
            (1, 15.0, 2, "node-a", 0, "GPU-a", 120.0),  # inside the 10..20 s measured window
            (1, 21.0, 3, "node-a", 0, "GPU-a", 105.0),
        ],
    )

    content = build_report(log_dir)

    assert "Pareto view" in content
    assert "Inspect a point" in content
    assert "aiperf c=4" in content
    assert "aiperf c=8" in content
    # single-run points have no run prefix in their id; each gets its own window charts
    assert 'data-point-id="::aiperf::c4"' in content
    assert 'data-point-id="::aiperf::c8"' in content


def test_build_report_charts_only_without_benchmark_windows(tmp_path: Path) -> None:
    """No benchmark.out at all: still renders the power charts, just no stats table."""
    log_dir = tmp_path
    _write_gpu_csv(
        log_dir / "power" / "samples.csv",
        [
            (1, 9.0, 1, "node-a", 0, "GPU-a", 100.0),
            (1, 15.0, 2, "node-a", 0, "GPU-a", 110.0),
        ],
    )

    content = build_report(log_dir)

    assert "No concurrency-level benchmark windows" in content
    assert "GPU power (W)" in content


def test_build_report_raises_when_nothing_applies(tmp_path: Path) -> None:
    with pytest.raises(PowerReportError):
        build_report(tmp_path)


def test_build_writes_html_file_next_to_the_run(tmp_path: Path) -> None:
    log_dir = tmp_path
    _write_gpu_csv(log_dir / "power" / "samples.csv", [(1, 9.0, 1, "node-a", 0, "GPU-a", 100.0)])

    out = build(log_dir)

    assert out == log_dir / "power_report.html"
    assert out.is_file()


def test_build_returns_none_when_nothing_applies(tmp_path: Path) -> None:
    assert build(tmp_path) is None


# ---------------------------------------------------------------------------
# Multi-directory rollup
# ---------------------------------------------------------------------------


def test_dedupe_labels_suffixes_repeats() -> None:
    assert _dedupe_labels(["a", "b", "a", "a", "c"]) == ["a", "b", "a (2)", "a (3)", "c"]


def _write_run(log_dir: Path, *, gpu_watts: float) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    _write_aiperf_run(log_dir)
    _write_gpu_csv(
        log_dir / "power" / "samples.csv",
        [
            (1, 9.0, 1, "node-a", 0, "GPU-a", gpu_watts),
            (1, 15.0, 2, "node-a", 0, "GPU-a", gpu_watts + 10.0),
            (1, 21.0, 3, "node-a", 0, "GPU-a", gpu_watts + 5.0),
        ],
    )


def _write_config_yaml(
    job_dir: Path, *, prefill_workers: int, gpus_per_prefill: int, decode_workers: int, gpus_per_decode: int
) -> None:
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "config.yaml").write_text(
        "resources:\n"
        f"  prefill_workers: {prefill_workers}\n"
        f"  gpus_per_prefill: {gpus_per_prefill}\n"
        f"  decode_workers: {decode_workers}\n"
        f"  gpus_per_decode: {gpus_per_decode}\n"
    )


# ---------------------------------------------------------------------------
# Run topology classification
# ---------------------------------------------------------------------------


def test_load_run_topology_reads_resources_block(tmp_path: Path) -> None:
    job_dir = tmp_path / "job"
    log_dir = job_dir / "logs"
    log_dir.mkdir(parents=True)
    _write_config_yaml(job_dir, prefill_workers=1, gpus_per_prefill=4, decode_workers=4, gpus_per_decode=8)

    result = _load_run_topology(log_dir)

    assert result == ("P1x4+D4x8", 36)


def test_load_run_topology_falls_back_to_the_copy_inside_logs(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    _write_config_yaml(log_dir, prefill_workers=1, gpus_per_prefill=4, decode_workers=1, gpus_per_decode=8)

    assert _load_run_topology(log_dir) == ("P1x4+D1x8", 12)


def test_load_run_topology_returns_none_without_a_config(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    log_dir.mkdir()

    assert _load_run_topology(log_dir) is None


def test_load_run_topology_returns_none_on_missing_resources_fields(tmp_path: Path) -> None:
    job_dir = tmp_path / "job"
    log_dir = job_dir / "logs"
    log_dir.mkdir(parents=True)
    (job_dir / "config.yaml").write_text("resources:\n  gpu_type: gb300\n")

    assert _load_run_topology(log_dir) is None


def test_build_combined_report_merges_rows_from_every_run(tmp_path: Path) -> None:
    run_a = tmp_path / "runA" / "logs"
    run_b = tmp_path / "runB" / "logs"
    _write_run(run_a, gpu_watts=100.0)
    _write_run(run_b, gpu_watts=200.0)

    content = build_combined_report([run_a, run_b])

    assert "2 runs: runA, runB" in content
    # (fixture runs have no CPU leg, so the Run cell carries a ⚠ coverage marker)
    assert content.count('<td>runA · aiperf c=4 <span class="row-warn"') == 1
    assert content.count('<td>runB · aiperf c=4 <span class="row-warn"') == 1
    assert "CPU power not collected for this run" in content
    assert "runA — aiperf c=4" in content  # per-concurrency card titles
    assert "runB — aiperf c=4" in content
    # data-table tab: one card per run x concurrency, a run/concurrency checkbox filter
    assert content.count('class="conc-card"') == 2
    assert 'data-run="runA" data-conc="4"' in content
    assert 'data-run="runB" data-conc="4"' in content
    assert content.count('data-filter="run"') >= 2
    # each card carries both a whole-run view and a profile-window view behind one global toggle
    assert content.count('class="view-window-only"') == 1
    assert content.count('<div class="view-window" hidden>') == 2
    assert "runA — aiperf c=4 (profile window)" in content
    assert '<input type="checkbox" data-filter="conc" value="4" checked>' in content
    # series data embedded once per run and shared by that run's cards
    assert content.count('data-source-id="run-src-0"') == 1
    assert content.count('data-source-id="run-src-1"') == 1
    # whole-run traces carry idle/warmup/profile phase shading
    assert "data-phases=" in content
    assert "Profile (measured)" in content
    # one shared table, not one table per run
    assert content.count("Throughput &amp; power by concurrency") == 1
    assert "Pareto view" in content
    assert "runA · aiperf c=4" in content
    assert "runB · aiperf c=4" in content
    # clicking a Pareto point swaps in that run's power charts
    assert 'data-pareto-run="runA"' in content
    assert 'data-pareto-run="runB"' in content
    # ... and its own measured-window charts, keyed by point id
    assert 'data-point-id="runA::aiperf::c4"' in content
    assert 'data-point-id="runB::aiperf::c4"' in content
    # both scopes live in one card behind a checkbox; the window is the default
    assert content.count('class="scope-whole-run"') == 1
    assert '<div class="scope-run" hidden>' in content
    # KPI stat cards in the header
    assert "Runs" in content
    assert "GPU devices tracked" in content
    # tab order: Pareto (front), Data table, CPU vs GPU, vs baseline
    assert (
        content.index('data-tab="pareto"')
        < content.index('data-tab="table"')
        < content.index('data-tab="power"')
        < content.index('data-tab="baseline"')
    )
    assert 'data-tab-panel="table" hidden' in content
    assert 'data-tab-panel="power" hidden' in content


def test_build_combined_report_sorts_by_gpu_count_then_concurrency(tmp_path: Path) -> None:
    big = tmp_path / "big" / "logs"
    small = tmp_path / "small" / "logs"
    _write_run(big, gpu_watts=100.0)
    _write_run(small, gpu_watts=200.0)
    _write_config_yaml(tmp_path / "big", prefill_workers=1, gpus_per_prefill=4, decode_workers=4, gpus_per_decode=8)
    _write_config_yaml(tmp_path / "small", prefill_workers=1, gpus_per_prefill=4, decode_workers=1, gpus_per_decode=8)

    # Passed in "big first" order; the smaller topology should still render first.
    content = build_combined_report([big, small])

    assert content.index("P1x4+D1x8") < content.index("P1x4+D4x8")


def test_build_combined_report_dedupes_identical_labels(tmp_path: Path) -> None:
    run_a = tmp_path / "same" / "logs"
    run_b = tmp_path / "same_copy" / "same" / "logs"
    _write_run(run_a, gpu_watts=100.0)
    _write_run(run_b, gpu_watts=150.0)

    content = build_combined_report([run_a, run_b])

    assert "same, same (2)" in content
    assert "<td>same · aiperf c=4 <span" in content
    assert "<td>same (2) · aiperf c=4 <span" in content


def test_build_combined_report_skips_empty_dirs_but_keeps_the_rest(tmp_path: Path) -> None:
    run_a = tmp_path / "runA" / "logs"
    empty_dir = tmp_path / "empty" / "logs"
    empty_dir.mkdir(parents=True)
    _write_run(run_a, gpu_watts=100.0)

    content = build_combined_report([run_a, empty_dir])

    assert "1 runs: runA" in content


def test_build_combined_report_raises_when_every_dir_is_empty(tmp_path: Path) -> None:
    empty_a = tmp_path / "a"
    empty_b = tmp_path / "b"
    empty_a.mkdir()
    empty_b.mkdir()

    with pytest.raises(PowerReportError):
        build_combined_report([empty_a, empty_b])


def test_build_combined_writes_to_the_given_output_path(tmp_path: Path) -> None:
    run_a = tmp_path / "runA" / "logs"
    run_b = tmp_path / "runB" / "logs"
    _write_run(run_a, gpu_watts=100.0)
    _write_run(run_b, gpu_watts=200.0)
    out_path = tmp_path / "combined.html"

    out = build_combined([run_a, run_b], output_path=out_path)

    assert out == out_path
    assert out_path.is_file()


def test_build_combined_returns_none_when_nothing_applies(tmp_path: Path) -> None:
    empty_a = tmp_path / "a"
    empty_b = tmp_path / "b"
    empty_a.mkdir()
    empty_b.mkdir()

    assert build_combined([empty_a, empty_b], output_path=tmp_path / "out.html") is None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_main_single_dir_writes_into_the_run_dir(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    log_dir = tmp_path
    _write_gpu_csv(log_dir / "power" / "samples.csv", [(1, 9.0, 1, "node-a", 0, "GPU-a", 100.0)])

    exit_code = main([str(log_dir)])

    assert exit_code == 0
    assert (log_dir / "power_report.html").is_file()
    assert str(log_dir / "power_report.html") in capsys.readouterr().out


def test_main_multi_dir_writes_combined_report_to_explicit_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_a = tmp_path / "runA" / "logs"
    run_b = tmp_path / "runB" / "logs"
    _write_run(run_a, gpu_watts=100.0)
    _write_run(run_b, gpu_watts=200.0)
    out_path = tmp_path / "combined.html"

    exit_code = main([str(run_a), str(run_b), "-o", str(out_path)])

    assert exit_code == 0
    assert out_path.is_file()
    assert "2 runs" in out_path.read_text()


def test_main_multi_dir_defaults_output_to_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    run_a = tmp_path / "runA" / "logs"
    run_b = tmp_path / "runB" / "logs"
    _write_run(run_a, gpu_watts=100.0)
    _write_run(run_b, gpu_watts=200.0)
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)

    exit_code = main([str(run_a), str(run_b)])

    assert exit_code == 0
    assert (cwd / "power_report_combined.html").is_file()


def test_main_returns_nonzero_when_nothing_applies(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main([str(tmp_path)])

    assert exit_code == 1
    assert "error" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Phase bands
# ---------------------------------------------------------------------------


def _window(concurrency: int, start: float, end: float, warmup: tuple[float, float] | None = None) -> dict:
    return {
        "benchmark_type": "aiperf",
        "concurrency": concurrency,
        "start_unix": start,
        "end_unix": end,
        "warmup_start_unix": warmup[0] if warmup else None,
        "warmup_end_unix": warmup[1] if warmup else None,
    }


def test_phase_bands_classify_idle_warmup_profile_in_order() -> None:
    reports = [_window(4, 200.0, 300.0, warmup=(150.0, 200.0)), _window(1, 20.0, 100.0)]

    bands = _phase_bands(reports, origin=0.0, run_end=400.0)

    assert [(b["kind"], b["t0"], b["t1"]) for b in bands] == [
        ("idle", 0.0, 20.0),
        ("profile", 20.0, 100.0),
        ("idle", 100.0, 150.0),
        ("warmup", 150.0, 200.0),
        ("profile", 200.0, 300.0),
        ("idle", 300.0, 400.0),
    ]
    assert bands[4]["label"] == "aiperf c=4 profile"
    assert (bands[4]["bench"], bands[4]["conc"]) == ("aiperf", 4)
    assert "conc" not in bands[0]  # idle bands belong to no concurrency


def test_phase_bands_are_relative_to_origin_and_clip_overlaps() -> None:
    # warmup recorded as running past the profile start is clipped to it
    reports = [_window(2, 1100.0, 1200.0, warmup=(1050.0, 1120.0))]

    bands = _phase_bands(reports, origin=1000.0, run_end=1200.0)

    assert [(b["kind"], b["t0"], b["t1"]) for b in bands] == [
        ("idle", 0.0, 50.0),
        ("warmup", 50.0, 100.0),
        ("profile", 100.0, 200.0),
    ]


def test_phase_bands_empty_without_reports() -> None:
    assert _phase_bands([], origin=0.0, run_end=10.0) == []


# ---------------------------------------------------------------------------
# Frontier families (GPU type x model)
# ---------------------------------------------------------------------------


def test_load_run_family_prefers_identity_repo_over_model_path() -> None:
    config = {
        "resources": {"gpu_type": "GB300"},
        "identity": {"model": {"repo": "deepseek-ai/DeepSeek-V4-Pro"}},
        "model": {"path": "/scratch/models/something-else"},
    }

    assert _load_run_family(config) == ("gb300", "deepseek-ai/DeepSeek-V4-Pro")


def test_load_run_family_falls_back_to_model_path_basename() -> None:
    config = {"resources": {"gpu_type": "gb300"}, "model": {"path": "hf:org/MiniMax-M3-NVFP4"}}
    assert _load_run_family(config) == ("gb300", "MiniMax-M3-NVFP4")

    config = {"resources": {"gpu_type": "gb300"}, "model": {"path": "/scratch/models/MiniMax-M3-NVFP4/"}}
    assert _load_run_family(config) == ("gb300", "MiniMax-M3-NVFP4")


def test_load_run_family_tolerates_missing_config_and_blocks() -> None:
    assert _load_run_family(None) == (None, None)
    assert _load_run_family({}) == (None, None)
    assert _load_run_family({"identity": {"model": {}}, "model": {}}) == (None, None)


def test_family_label_keeps_unknown_parts_visible() -> None:
    assert _family_label("gb300", "deepseek-ai/DeepSeek-V4-Pro") == "gb300 · deepseek-ai/DeepSeek-V4-Pro"
    assert _family_label(None, None) == "unknown gpu · unknown model"


def test_pareto_points_group_and_colour_follow_the_family_not_the_run() -> None:
    reports = [_report_dict(concurrency=4, output_tps=100.0, tps_per_gpu=50.0)]

    a = _pareto_points(reports, run_label="runA", run_position=0, group="gb300 · m", group_position=0)[0]
    b = _pareto_points(reports, run_label="runB", run_position=1, group="gb300 · m", group_position=0)[0]
    c = _pareto_points(reports, run_label="runC", run_position=2, group="b200 · m", group_position=1)[0]

    assert a["group"] == b["group"] and a["color"] == b["color"]  # same family -> same frontier + hue
    assert a["run"] != b["run"]  # but still distinct runs (baseline / chart swapping key on run)
    assert c["group"] != a["group"] and c["color"] != a["color"]


def test_model_select_always_rendered_and_defaults_to_first_of_several() -> None:
    reports = [_report_dict(concurrency=4, output_tps=100.0, tps_per_gpu=50.0)]
    assert _model_select_html(_pareto_points(reports, run_label="runA")) == ""  # no model known
    one = _pareto_points(reports, run_label="runA", model="M1")
    assert '<option value="M1" selected>M1</option>' in _model_select_html(one)

    two = one + _pareto_points(reports, run_label="runB", run_position=1, model="M2")
    content = _model_select_html(two)
    assert '<option value="">All models</option>' in content
    assert '<option value="M1" selected>M1</option>' in content
    assert '<option value="M2">M2</option>' in content
    assert two[0]["model"] == "M1" and two[1]["model"] == "M2"


# ---------------------------------------------------------------------------
# Embedded JavaScript
# ---------------------------------------------------------------------------


@pytest.mark.skipif(shutil.which("node") is None, reason="node not available for a JS syntax check")
def test_embedded_javascript_parses(tmp_path: Path) -> None:
    """The inlined script is built from Python string fragments; a stray brace or
    duplicated declaration breaks the whole page silently. Parse it with node."""
    from srtctl.analysis.power_report_html import _JS

    script = tmp_path / "report.js"
    script.write_text(_JS)
    result = subprocess.run(["node", "--check", str(script)], capture_output=True, text=True, check=False)

    assert result.returncode == 0, result.stderr


# ---------------------------------------------------------------------------
# Throughput over time (aiperf timeslices)
# ---------------------------------------------------------------------------


def test_load_tps_series_reads_aiperf_timeslices_next_to_the_result(tmp_path: Path) -> None:
    from srtctl.analysis.power_report_html import _load_tps_series

    artifacts = tmp_path / "aiperf_artifacts"
    artifacts.mkdir()
    (artifacts / "profile_export_aiperf_timeslices.json").write_text(
        json.dumps(
            {
                "timeslices": [
                    {
                        "start_ns": 1_000_000_000_000,
                        "end_ns": 1_001_000_000_000,
                        "output_token_throughput": {"avg": 100.0},
                        "input_token_throughput": {"avg": 20.0},
                    },
                    {
                        "start_ns": 1_001_000_000_000,
                        "end_ns": 1_002_000_000_000,
                        "output_token_throughput": {"avg": 120.0},
                        "input_token_throughput": {"avg": None},
                    },
                ]
            }
        )
    )
    report = {"source": str(artifacts / "profile_export.jsonl")}

    series = _load_tps_series(report, origin=1000.0)

    assert [s["label"] for s in series] == ["output tok/s"]  # input intentionally not plotted
    assert series[0]["t"] == [0.5, 1.5]  # slice midpoints, relative to the window origin
    assert series[0]["w"] == [100.0, 120.0]


def test_load_tps_series_is_empty_without_timeslices(tmp_path: Path) -> None:
    from srtctl.analysis.power_report_html import _load_tps_series

    assert _load_tps_series({"source": str(tmp_path / "profile_export.jsonl")}, origin=0.0) == []
    assert _load_tps_series({}, origin=0.0) == []


def test_derive_phase_tps_spreads_tokens_over_prefill_and_decode_spans(tmp_path: Path) -> None:
    from srtctl.analysis.power_report_html import _derive_phase_tps

    jsonl = tmp_path / "profile_export.jsonl"
    t0_ns = 1_000 * 10**9

    def rec(start_s: float, ttft_ms: float, decode_ms: float, isl: int, osl: int, phase: str = "profiling") -> str:
        return json.dumps(
            {
                "metadata": {"request_start_ns": t0_ns + int(start_s * 1e9), "benchmark_phase": phase},
                "metrics": {
                    "time_to_first_token": {"value": ttft_ms, "unit": "ms"},
                    "decode_duration": {"value": decode_ms, "unit": "ms"},
                    "input_sequence_length": {"value": isl, "unit": "tokens"},
                    "output_sequence_length": {"value": osl, "unit": "tokens"},
                },
            }
        )

    jsonl.write_text(
        "\n".join(
            [
                # prefill 0.0-1.0 s (1000 tok), decode 1.0-3.0 s (200 tok -> 100 tok/s)
                rec(0.0, 1000.0, 2000.0, 1000, 200),
                # prefill 2.5-3.0 s (500 tok -> 1000 tok/s, half a bin), decode 3.0-4.0 s (50 tok)
                rec(2.5, 500.0, 1000.0, 500, 50),
                rec(0.0, 1000.0, 1000.0, 99_999, 99_999, phase="warmup"),  # ignored
            ]
        )
    )
    report = {"source": str(jsonl), "start_unix": 1000.0, "end_unix": 1004.0}

    prefill, decode = _derive_phase_tps(report, origin=1000.0)

    assert prefill[0]["t"] == [0.5, 1.5, 2.5, 3.5]  # 1 s bins, midpoints relative to origin
    assert prefill[0]["w"] == [1000.0, 0.0, 500.0, 0.0]  # 500 tok at 1000 tok/s over half of bin 2
    assert decode[0]["w"] == [0.0, 100.0, 100.0, 50.0]
    assert sum(prefill[0]["w"]) == 1500 and sum(decode[0]["w"]) == 250  # integrates to token totals


def test_derive_phase_tps_is_none_without_per_request_export(tmp_path: Path) -> None:
    from srtctl.analysis.power_report_html import _derive_phase_tps

    agg = tmp_path / "profile_export_aiperf.json"
    agg.write_text("{}")
    assert _derive_phase_tps({"source": str(agg), "start_unix": 0.0, "end_unix": 10.0}, origin=0.0) is None
    assert _derive_phase_tps({"start_unix": 0.0, "end_unix": 10.0}, origin=0.0) is None


# ---------------------------------------------------------------------------
# Missing power legs are called out, not silently dropped
# ---------------------------------------------------------------------------


def test_coverage_warnings_explain_a_cancelled_cpu_exporter_step(tmp_path: Path) -> None:
    from srtctl.analysis.power_report_html import _power_coverage_warnings

    (tmp_path / "telemetry_cpu_power_exporter.node-a.out").write_text(
        "srun: error: node-a: task 0: Exited\nslurmstepd: error: *** STEP 1.2 CANCELLED DUE TO TASK FAILURE ***\n"
    )
    (tmp_path / "telemetry_cpu_power_exporter.node-b.out").write_text("listening addr=0.0.0.0:9405\nreceived SIGTERM\n")
    empty_csv = tmp_path / "power" / "cpu" / "samples.csv"
    empty_csv.parent.mkdir(parents=True)
    empty_csv.write_text("ts_unix\n")

    warnings = _power_coverage_warnings(
        tmp_path,
        cpu_csv=empty_csv,
        gpu_csv=tmp_path / "power" / "samples.csv",
        gpu_per_device={("node-a", 0): _series([1.0])},
        cpu_per_socket={},
    )

    assert len(warnings) == 1
    assert warnings[0].startswith("CPU power missing: exporter started on 2 host(s) but wrote no samples")
    assert "TASK FAILURE on node-a" in warnings[0]


def test_coverage_warnings_list_gpu_hosts_without_cpu_samples(tmp_path: Path) -> None:
    from srtctl.analysis.power_report_html import _power_coverage_warnings

    warnings = _power_coverage_warnings(
        tmp_path,
        cpu_csv=tmp_path / "cpu.csv",
        gpu_csv=tmp_path / "gpu.csv",
        gpu_per_device={("node-a", 0): _series([1.0]), ("node-b", 0): _series([1.0])},
        cpu_per_socket={("node-a", 0): _series([1.0])},
    )

    assert warnings == [
        "CPU power missing on 1 of 2 GPU host(s): node-b. CPU totals cover only the hosts that reported."
    ]


def test_coverage_warnings_are_silent_when_both_legs_are_present(tmp_path: Path) -> None:
    from srtctl.analysis.power_report_html import _power_coverage_warnings

    assert (
        _power_coverage_warnings(
            tmp_path,
            cpu_csv=tmp_path / "cpu.csv",
            gpu_csv=tmp_path / "gpu.csv",
            gpu_per_device={("node-a", 0): _series([1.0])},
            cpu_per_socket={("node-a", 0): _series([1.0])},
        )
        == []
    )


def test_power_charts_render_notices_strip() -> None:
    from srtctl.analysis.power_report_html import _power_charts_html

    gpu = _build_run_series({("node-a", 0): _series([1.0, 2.0])}, label_fmt="{host}/gpu{index}")
    content = _power_charts_html(gpu, [], notices=["CPU power missing: <test>"])
    assert 'class="chart-notices"' in content
    assert "CPU power missing: &lt;test&gt;" in content
    assert 'class="chart-notices"' not in _power_charts_html(gpu, [])


def test_node_power_card_has_a_notices_slot_for_coverage_warnings() -> None:
    from srtctl.analysis.power_report_html import _node_power_card_html

    content = _node_power_card_html()
    assert 'class="chart-notices node-power-notices" hidden' in content
    # the JS fills it from point.warnings matching "power missing" / "not collected"


def test_hosts_summary_collapses_numeric_ranges_for_the_tooltip() -> None:
    from srtctl.analysis.power_report_html import _hosts_summary

    hosts = [f"nvl72d090-T{n:02d}" for n in (10, 11, 12, 13, 14, 16, 18)]
    assert _hosts_summary(hosts) == "nvl72d090-T10…18 (7)"
    assert _hosts_summary(hosts, full=True) == ", ".join(hosts) + " (7)"
    assert _hosts_summary(["a", "b"]) == "a, b"
    assert _hosts_summary(None) == "—"
    assert _hosts_summary(["alpha1", "beta9", "gamma3", "delta2"]) == "alpha1 … gamma3 (4)"


def test_pareto_points_carry_gpu_type_and_hosts() -> None:
    from srtctl.analysis.power_report_html import _pareto_points

    reports = [_report_dict(concurrency=4, output_tps=100.0, tps_per_gpu=50.0)]
    points = _pareto_points(reports, run_label="runA", gpu_type="gb300", hosts=["n1", "n2"])
    hover = dict(points[0]["hover"])
    fields = dict(points[0]["fields"])
    assert hover["GPU type / hosts"] == "gb300 · n1, n2"
    assert fields["GPU type"] == "gb300"
    assert fields["Hosts"] == "n1, n2"


def test_power_variant_watts_measured_projected_static() -> None:
    from srtctl.analysis.power_report_html import GpuPowerBudget, _power_variant_watts

    b = GpuPowerBudget(static_node_w=8_000.0, gpus_per_node=4, overhead_w_per_gpu=500.0, cpu_estimate_w_per_gpu=50.0)
    w, est = _power_variant_watts(gpu_w=3_000.0, cpu_w=400.0, num_gpus=4, budget=b)
    assert est is False
    assert w["measured"] == 3_400.0
    assert w["projected"] == 3_400.0 + 500.0 * 4
    assert w["static"] == 8_000.0  # 4 GPUs = one node's static budget

    # No CPU leg: the per-GPU estimate stands in (50 W x 4 GPUs), flagged, and projected builds on it.
    w, est = _power_variant_watts(gpu_w=3_000.0, cpu_w=None, num_gpus=4, budget=b)
    assert est is True
    assert w["measured"] == 3_000.0 + 200.0
    assert w["projected"] == 3_200.0 + 2_000.0
    assert w["static"] == 8_000.0

    # Unknown GPU type: only what was measured; no CPU and no budget -> no measured value either.
    w, est = _power_variant_watts(gpu_w=3_000.0, cpu_w=400.0, num_gpus=4, budget=None)
    assert (w, est) == ({"measured": 3_400.0, "projected": None, "static": None}, False)
    w, est = _power_variant_watts(gpu_w=3_000.0, cpu_w=None, num_gpus=4, budget=None)
    assert (w["measured"], est) == (None, False)


def test_pareto_points_carry_basis_split_metrics_and_budget() -> None:
    from srtctl.analysis.power_report_html import GPU_POWER_BUDGETS

    reports = [_report_dict(concurrency=4, output_tps=100.0, tps_per_gpu=50.0)]
    p = _pareto_points(reports, run_label="runA", gpu_type="GB300")[0]
    b = GPU_POWER_BUDGETS["gb300"]
    assert p["power_basis_w"]["measured"] == 340.0
    assert p["power_basis_w"]["projected"] == pytest.approx(340.0 + 2 * b.overhead_w_per_gpu)
    assert p["power_basis_w"]["static"] == pytest.approx(2 * b.static_w_per_gpu)
    assert p["m"]["total_tps_per_mw__measured"] == pytest.approx(140.0 / 340.0 * 1e6)
    assert p["m"]["input_tps_per_mw__measured"] == pytest.approx(40.0 / 340.0 * 1e6)
    assert p["m"]["output_tps_per_mw__static"] == pytest.approx(100.0 / (2 * b.static_w_per_gpu) * 1e6)
    assert p["m"]["node_w_per_gpu__static"] == pytest.approx(b.static_w_per_gpu)
    assert p["cpu_estimated"] is False
    assert p["budget"]["static_w_per_gpu"] == b.static_w_per_gpu
    fields = dict(p["fields"])
    assert "Static power budget" in fields and "Projected avg-rack power (approx.)" in fields
    assert not any("No power budget" in w for w in p["warnings"])

    from srtctl.analysis.power_report_html import _power_budget_for

    assert _power_budget_for("VR") is GPU_POWER_BUDGETS["vr200"]  # alias, case-insensitive
    assert _power_budget_for("gb200") is None  # no budget agreed for GB200

    unknown = _pareto_points(reports, run_label="runA", gpu_type="h100")[0]
    assert unknown["budget"] is None
    assert unknown["m"]["total_tps_per_mw__static"] is None
    assert any("No power budget for GPU type 'h100'" in w for w in unknown["warnings"])


def test_pareto_points_estimate_cpu_when_leg_missing() -> None:
    r = _report_dict(concurrency=4, output_tps=100.0, tps_per_gpu=50.0)
    r["perf_per_watt"]["cpu_avg_power_w"] = None
    r["perf_per_watt"]["combined_avg_power_w"] = None
    p = _pareto_points([r], run_label="runA", gpu_type="gb300")[0]
    assert p["cpu_estimated"] is True
    assert p["power_basis_w"]["measured"] == pytest.approx(300.0 + 50.0 * 2)
    assert p["m"]["total_tps_per_mw__measured"] is not None
    assert any("CPU power not measured" in w for w in p["warnings"])
    assert "Measured GPU + estimated CPU (avg)" in dict(p["fields"])


def test_type_power_card_is_emitted_above_the_node_power_card() -> None:
    from srtctl.analysis.power_report_html import _node_power_card_html, _type_power_card_html

    content = _type_power_card_html() + _node_power_card_html()
    assert content.index('class="pareto-card type-power-card"') < content.index('class="pareto-card node-power-card"')
    assert 'class="chart-notices type-power-notices" hidden' in content
