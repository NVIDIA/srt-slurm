# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-window NIC traffic summary math and its post-processing step."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from srtctl.analysis.nic_traffic_report import SUMMARY_FILENAME, SUMMARY_HEADER, load_nic_series, summarize
from srtctl.analysis.power_energy_report import BENCHMARK_TYPE_SA_BENCH, ConcurrencyWindow
from srtctl.cli.mixins.postprocess_stage import PostProcessStageMixin
from srtctl.core.nic_traffic import SAMPLES_HEADER

GB = 125_000_000  # bytes in one gigabit

# One sampler pass per second; each pass reads node-a's rdma1 0.3 s after its rdma0.
# node-a rdma0 moves 1, 2, (counter reset), 3 Gbit in consecutive passes and receives 2 Gbit in the last one;
# node-a rdma1 moves 2, 1, 1, 1 Gbit; node-b starts sampling too late to cover the window.
SAMPLES = [
    *(
        ("node-a", "rdma0", seq, 100.0 + seq, tx, rx)
        for seq, (tx, rx) in enumerate([(0, 0), (1 * GB, 0), (3 * GB, 0), (10, 0), (3 * GB + 10, 2 * GB)])
    ),
    *(("node-a", "rdma1", seq, 100.3 + seq, tx * GB, 0) for seq, tx in enumerate([0, 2, 3, 4, 5])),
    ("node-b", "rdma0", 0, 103.5, 0, 0),
    ("node-b", "rdma0", 1, 104.5, GB, 0),
]


def _write_traffic(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(SAMPLES_HEADER)
        for host, device, seq, timestamp, tx, rx in sorted(SAMPLES, key=lambda sample: sample[3]):
            writer.writerow((repr(timestamp), seq, host, device, "1", f"net-{device}", "ionic", tx, rx))


def _window(start: float, end: float) -> ConcurrencyWindow:
    return ConcurrencyWindow(
        benchmark_type=BENCHMARK_TYPE_SA_BENCH,
        concurrency=4,
        start_unix=start,
        end_unix=end,
        output_tokens=1,
        input_tokens=1,
        source=Path("results.json"),
    )


def _rows(rows: list[tuple[object, ...]]) -> dict[tuple[object, ...], dict[str, object]]:
    keyed = {}
    for row in rows:
        record = dict(zip(SUMMARY_HEADER, row, strict=True))
        keyed[(record["scope"], record["hostname"], record["device"])] = record
    return keyed


def test_window_mean_and_peak_drop_the_counter_reset_interval(tmp_path: Path) -> None:
    _write_traffic(tmp_path / "nic_traffic.csv")

    summary = summarize(load_nic_series(tmp_path / "nic_traffic.csv"), [_window(100.2, 103.9)])

    rows = _rows(summary.rows)
    assert set(rows) == {("nic", "node-a", "rdma0"), ("nic", "node-a", "rdma1"), ("node", "node-a", "")}
    rdma0 = rows[("nic", "node-a", "rdma0")]
    assert (rdma0["intervals"], rdma0["dropped_intervals"]) == (3, 1)
    assert rdma0["mean_tx_gbps"] == pytest.approx(2.0)  # (1 + 2 + 3) Gbit over the 3 kept seconds
    assert rdma0["peak_tx_gbps"] == pytest.approx(3.0)
    assert rdma0["mean_rx_gbps"] == pytest.approx(2 / 3, abs=1e-6)
    assert rdma0["peak_rx_gbps"] == pytest.approx(2.0)
    assert (rdma0["netdev"], rdma0["driver"]) == ("net-rdma0", "ionic")
    rdma1 = rows[("nic", "node-a", "rdma1")]
    assert rdma1["intervals"] == 4
    assert (rdma1["mean_tx_gbps"], rdma1["peak_tx_gbps"]) == pytest.approx((1.25, 2.0))
    node = rows[("node", "node-a", "")]
    assert node["intervals"] == 4
    assert node["mean_tx_gbps"] == pytest.approx(3.25)
    # 3 + 1 Gbit/s in the last pass, though rdma1's row is stamped 0.3 s after rdma0's; the NIC peaks
    # (3 and 2 Gbit/s) fall in different passes, so their sum is not the node peak.
    assert node["peak_tx_gbps"] == pytest.approx(4.0)
    assert node["peak_rx_gbps"] == pytest.approx(2.0)
    assert node["dropped_intervals"] == 1
    assert summary.uncovered == ["node-b/rdma0/1 @ concurrency 4"]


def _sa_bench_run(log_dir: Path, start: float, end: float) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "benchmark.out").write_text("============ Serving Benchmark Result ============\n")
    result_dir = log_dir / "sa-bench_isl_8_osl_8"
    result_dir.mkdir()
    (result_dir / "results_concurrency_4_gpus_16.json").write_text(
        json.dumps(
            {
                "benchmark_start_time_unix": start,
                "benchmark_end_time_unix": end,
                "duration": end - start,
                "total_input_tokens": 1,
                "total_output_tokens": 1,
            }
        )
    )


class _PostProcess(PostProcessStageMixin):
    def __init__(self, log_dir: Path) -> None:
        self.runtime = SimpleNamespace(log_dir=log_dir)  # type: ignore[assignment]
        self.config = SimpleNamespace(  # type: ignore[assignment]
            observability=SimpleNamespace(nic_traffic=SimpleNamespace(storage_subdir="nic_traffic"))
        )


def test_postprocess_writes_the_summary_without_power_telemetry(tmp_path: Path) -> None:
    _sa_bench_run(tmp_path, 100.2, 103.9)
    _write_traffic(tmp_path / "nic_traffic" / "nic_traffic.csv")

    _PostProcess(tmp_path)._build_nic_traffic_summary()

    with (tmp_path / "nic_traffic" / SUMMARY_FILENAME).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [(row["scope"], row["hostname"], row["device"]) for row in rows] == [
        ("nic", "node-a", "rdma0"),
        ("nic", "node-a", "rdma1"),
        ("node", "node-a", ""),
    ]
    assert rows[0]["concurrency"] == "4"
    assert float(rows[2]["peak_tx_gbps"]) == pytest.approx(4.0)


def test_postprocess_skips_a_run_without_benchmark_windows(tmp_path: Path) -> None:
    _write_traffic(tmp_path / "nic_traffic" / "nic_traffic.csv")

    _PostProcess(tmp_path)._build_nic_traffic_summary()

    assert not (tmp_path / "nic_traffic" / SUMMARY_FILENAME).exists()
