# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Semantic tests for direct raw-capture ingestion, independent of dashboard layout."""

import gzip
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from srtctl.analysis.tachometer_dashboard.reader import discover_sources, parse_metric_name, reduce_capture


def capture(path: Path, rows: list[dict], *, relative: bool = False) -> Path:
    defaults = {
        "metric_name": "requests_total",
        "metric_value": 0.0,
        "scraper_endpoint": "frontend0",
        "histogram_bucket_upper": None,
        "histogram_bucket_lower": None,
        "histogram_sum": None,
        "histogram_count": None,
        "hostname": "",
    }
    complete = []
    for row in rows:
        item = defaults | row
        if not relative:
            item["timestamp_ns"] = 1_700_000_000_000_000_000 + int(item.pop("t", 0) * 1e9)
        else:
            item["time_since_start"] = float(item.pop("t", 0))
        complete.append(item)
    table = pa.Table.from_pylist(complete)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".arrow":
        with pa.OSFile(str(path), "wb") as target, pa.ipc.new_stream(target, table.schema) as writer:
            writer.write_table(table)
    else:
        pq.write_table(table, path)
    return path


def payload(catalog: dict, destination: Path, name: str) -> dict:
    item = next(metric for metric in catalog["metrics"] if metric["name"] == name)
    with gzip.open(destination / item["payload"], "rt") as source:
        return json.load(source)


def test_labels_quoted_commas_escapes_and_unquoted_values() -> None:
    name, labels = parse_metric_name(r'latency{group="a,b",thread="say\"hi",line="a\nb",path="a\\b",mode=user}')
    assert name == "latency"
    assert labels == {"group": "a,b", "thread": 'say"hi', "line": "a\nb", "path": "a\\b", "mode": "user"}
    with pytest.raises(ValueError, match="Duplicate label"):
        parse_metric_name('metric{a="1",a="2"}')
    with pytest.raises(ValueError, match="Unterminated"):
        parse_metric_name('metric{a="1}')


def test_full_identity_preserves_endpoint_and_conflicting_metadata(tmp_path: Path) -> None:
    rows = [
        {"metric_name": 'cpu_total{hostname="inline",group="a,b"}', "metric_value": 3.0, "hostname": "node1"},
        {"metric_name": 'cpu_total{hostname="inline",group="a,b"}', "metric_value": 3.0, "hostname": "node2"},
        {
            "metric_name": 'cpu_total{hostname="inline",group="a,b"}',
            "metric_value": 3.0,
            "hostname": "node1",
            "scraper_endpoint": "another",
        },
    ]
    source = capture(tmp_path / "input.parquet", rows)
    out = tmp_path / "out"
    catalog = reduce_capture([source], out)
    result = payload(catalog, out, "cpu_total")
    assert len(result["series"]) == 3
    assert {series["endpoint"] for series in result["series"]} == {"frontend0", "another"}
    assert {series["metadata"]["hostname"] for series in result["series"]} == {"node1", "node2"}
    assert all(series["labels"]["hostname"] == "inline" for series in result["series"])


def test_counter_rates_skip_reset_gap_and_first_sample(tmp_path: Path) -> None:
    source = capture(
        tmp_path / "input.parquet",
        [{"t": t, "metric_value": v} for t, v in [(0, 10), (1, 12), (2, 15), (3, 1), (4, 3), (20, 8), (21, 10)]],
    )
    out = tmp_path / "out"
    catalog = reduce_capture([source], out, resolution_s=1, is_counter=lambda name: name.endswith("_total"))
    points = payload(catalog, out, "requests_total")["series"][0]["points"]
    by_bin = {point[0]: point for point in points}
    assert by_bin[0][5:7] == [None, None]
    assert by_bin[1][5:7] == [2, 1]
    assert by_bin[2][5:7] == [3, 1]
    assert by_bin[3][5:7] == [None, None]
    assert by_bin[3][8] == 1
    assert by_bin[20][5:7] == [None, None]
    assert by_bin[20][9] == 1
    assert by_bin[21][5:7] == [2, 1]
    assert 10 not in by_bin


def test_counter_rate_uses_raw_intervals_inside_bin(tmp_path: Path) -> None:
    source = capture(
        tmp_path / "input.parquet", [{"t": t, "metric_value": v} for t, v in [(0, 0), (1, 8), (2, 1), (3, 5)]]
    )
    out = tmp_path / "out"
    catalog = reduce_capture([source], out, resolution_s=10, is_counter=lambda name: True)
    point = payload(catalog, out, "requests_total")["series"][0]["points"][0]
    assert point[5:7] == [12, 2]
    assert point[8] == 1


def test_gauge_decreases_are_not_counter_resets(tmp_path: Path) -> None:
    source = capture(tmp_path / "input.parquet", [{"t": 0, "metric_value": 10}, {"t": 1, "metric_value": 2}])
    out = tmp_path / "out"
    catalog = reduce_capture([source], out)
    point = payload(catalog, out, "requests_total")["series"][0]["points"][0]
    assert point[1:5] == [6, 2, 10, 2]
    assert point[5:7] == [None, None]
    assert point[8] == 0


def test_histogram_uses_bucket_identity_not_damaged_attached_stats(tmp_path: Path) -> None:
    rows = []
    for stage in ["tokenize", "dispatch"]:
        for le, bound, count in [("0.1", 0.1, 3), ("+Inf", None, 9)]:
            for t in [0, 1]:
                rows.append(
                    {
                        "metric_name": f'latency{{stage="{stage}",le="{le}"}}',
                        "t": t,
                        "metric_value": count * (t + 1),
                        "histogram_bucket_upper": bound,
                        "histogram_sum": 999999.0,
                        "histogram_count": 999999.0,
                        "histogram_bucket_lower": 999999.0,
                    }
                )
    source = capture(tmp_path / "input.parquet", rows)
    out = tmp_path / "out"
    catalog = reduce_capture([source], out, resolution_s=1)
    result = payload(catalog, out, "latency")
    assert result["kind"] == "histogram"
    assert len(result["series"]) == 4
    assert {series["labels"]["le"] for series in result["series"]} == {"0.1", "+Inf"}
    assert {series["labels"]["stage"] for series in result["series"]} == {"tokenize", "dispatch"}
    assert all(series["points"][1][5] in (3, 9) for series in result["series"])
    assert len(catalog["metrics"]) == 1


def test_summary_quantile_is_not_a_histogram(tmp_path: Path) -> None:
    source = capture(
        tmp_path / "input.parquet", [{"metric_name": 'go_gc_duration_seconds{quantile="0.5"}', "metric_value": 0.1}]
    )
    catalog = reduce_capture([source], tmp_path / "out")
    assert catalog["metrics"][0]["kind"] == "scalar"


def test_arrow_tail_extends_parquet_and_retains_counter_continuity(tmp_path: Path) -> None:
    parquet = capture(tmp_path / "incomplete-1.parquet", [{"t": 0, "metric_value": 0}, {"t": 1, "metric_value": 3}])
    arrow = capture(tmp_path / "current.arrow", [{"t": 2, "metric_value": 8}, {"t": 3, "metric_value": 12}])
    out = tmp_path / "out"
    catalog = reduce_capture([arrow, parquet], out, resolution_s=1, is_counter=lambda name: True)
    assert catalog["row_count"] == 4
    assert catalog["duration_s"] == 3
    assert [info["format"] for info in catalog["source_files"]] == ["parquet", "arrow"]
    assert payload(catalog, out, "requests_total")["series"][0]["points"][2][5] == 5


def test_overlap_rejected_before_payload_write(tmp_path: Path) -> None:
    first = capture(tmp_path / "first.parquet", [{"t": 0}, {"t": 2}])
    second = capture(tmp_path / "second.arrow", [{"t": 1}, {"t": 3}])
    with pytest.raises(ValueError, match="Overlapping raw captures"):
        reduce_capture([first, second], tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_relative_only_capture_and_mixed_clock_rejection(tmp_path: Path) -> None:
    source = capture(tmp_path / "relative.arrow", [{"t": 5}, {"t": 6}], relative=True)
    catalog = reduce_capture([source], tmp_path / "out")
    assert catalog["start_ns"] is None
    assert catalog["end_ns"] is None
    assert catalog["relative_start_s"] == 5
    assert catalog["duration_s"] == 1
    epoch = capture(tmp_path / "epoch.parquet", [{"t": 0}])
    with pytest.raises(ValueError, match="Mixed epoch and relative"):
        reduce_capture([source, epoch], tmp_path / "mixed")


def test_discovery_prefers_final_and_never_opens_sidecars(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    final = capture(tmp_path / "final.parquet", [{"t": 0}])
    capture(tmp_path / "incomplete-1.parquet", [{"t": 0}])
    tail = capture(tmp_path / "current.arrow", [{"t": 1}])
    for name in ["tachometer_config.toml", "run.log", "server_metrics_export.jsonl"]:
        (tmp_path / name).write_text("must never be read")
    original = Path.open

    def guarded(path: Path, *args, **kwargs):
        assert path.suffix not in {".toml", ".log", ".jsonl"}, f"Unexpected sidecar access: {path}"
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded)
    assert discover_sources(tmp_path) == [final, tail]
    assert reduce_capture(discover_sources(tmp_path), tmp_path / "out")["row_count"] == 2
    with pytest.raises(ValueError, match="raw .parquet"):
        discover_sources(tmp_path / "server_metrics_export.jsonl")


def test_nonfinite_values_reported_and_conflicting_duplicates_preserved(tmp_path: Path) -> None:
    source = capture(tmp_path / "bad.parquet", [{"t": 0, "metric_value": float("nan")}, {"t": 1, "metric_value": 3}])
    catalog = reduce_capture([source], tmp_path / "out")
    assert catalog["nonfinite_rows"] == 1
    conflict = capture(tmp_path / "conflict.parquet", [{"t": 1, "metric_value": 2}, {"t": 1, "metric_value": 3}])
    catalog = reduce_capture([conflict], tmp_path / "other", is_counter=lambda name: True)
    series = payload(catalog, tmp_path / "other", "requests_total")["series"][0]
    assert series["ambiguous"]
    assert series["points"][0][1:4] == [2.5, 2, 3]
    assert series["points"][0][5:7] == [None, None]
    assert catalog["metrics"][0]["conflicting_rows"] == 1


def test_equivalent_inline_label_order_groups_before_time_sort(tmp_path: Path) -> None:
    source = capture(
        tmp_path / "labels.parquet",
        [
            {"t": 0, "metric_value": 0, "metric_name": 'requests_total{model="m",status="ok"}'},
            {"t": 1, "metric_value": 1, "metric_name": 'requests_total{status="ok",model="m"}'},
            {"t": 2, "metric_value": 2, "metric_name": 'requests_total{model="m",status="ok"}'},
        ],
    )
    catalog = reduce_capture([source], tmp_path / "out", is_counter=lambda name: True)
    series = payload(catalog, tmp_path / "out", "requests_total")["series"]
    assert len(series) == 1
    assert series[0]["points"][0][5:7] == [2, 2]


def test_nonoverlapping_captures_with_different_raw_origins_rejected(tmp_path: Path) -> None:
    first = capture(tmp_path / "one.parquet", [{"t": 0, "time_since_start": 0.0}, {"t": 1, "time_since_start": 1.0}])
    second = capture(tmp_path / "two.arrow", [{"t": 10, "time_since_start": 0.0}, {"t": 11, "time_since_start": 1.0}])
    with pytest.raises(ValueError, match="Mixed raw capture origins"):
        reduce_capture([first, second], tmp_path / "out")


def test_raw_origin_verified_across_parquet_arrow_and_metadata_conflicts_visible(tmp_path: Path) -> None:
    first = capture(
        tmp_path / "one.parquet",
        [{"t": 0, "time_since_start": 0.0, "metric_name": 'metric{hostname="inline"}', "hostname": "node"}],
    )
    second = capture(
        tmp_path / "two.arrow",
        [{"t": 1, "time_since_start": 1.0, "metric_name": 'metric{hostname="inline"}', "hostname": "node"}],
    )
    catalog = reduce_capture([first, second], tmp_path / "out")
    assert catalog["source_files"][0]["origin_anchor_min_ns"] == "1700000000000000000"
    assert any("conflicts with raw metadata" in text for text in catalog["metrics"][0]["warnings"])
    assert catalog["label_values"]["hostname"] == ["inline", "node"]
    assert catalog["label_values"]["capture.hostname"] == ["node"]
    assert catalog["label_values"]["label.hostname"] == ["inline"]


def test_null_epoch_column_falls_back_to_raw_relative_time(tmp_path: Path) -> None:
    source = capture(
        tmp_path / "relative.arrow", [{"t": 5, "timestamp_ns": None}, {"t": 6, "timestamp_ns": None}], relative=True
    )
    catalog = reduce_capture([source], tmp_path / "out")
    assert catalog["start_ns"] is None
    assert catalog["relative_start_s"] == 5


def test_counter_continuity_across_reader_batches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from srtctl.analysis.tachometer_dashboard import reader

    monkeypatch.setattr(reader, "BATCH_SIZE", 2)
    source = capture(tmp_path / "batches.parquet", [{"t": t, "metric_value": t * 2} for t in range(5)])
    catalog = reduce_capture([source], tmp_path / "out", is_counter=lambda name: True)
    points = payload(catalog, tmp_path / "out", "requests_total")["series"][0]["points"]
    assert points[0][5:8] == [8, 4, 5]


def test_column_only_histogram_bounds_retain_float_precision(tmp_path: Path) -> None:
    bounds = (0.12345671, 0.12345672)
    source = capture(
        tmp_path / "bounds.parquet",
        [
            {"metric_name": "latency_bucket", "histogram_bucket_upper": bound, "t": t, "metric_value": t + 1.0}
            for bound in bounds
            for t in (0, 1)
        ],
    )
    out = tmp_path / "out"
    catalog = reduce_capture([source], out)
    series = payload(catalog, out, "latency")["series"]
    assert len(series) == 2
    assert {float(item["labels"]["le"]) for item in series} == set(bounds)
    assert all(not item["ambiguous"] for item in series)


def test_raw_source_changed_between_passes_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from srtctl.analysis.tachometer_dashboard import reader

    source = capture(tmp_path / "input.parquet", [{"t": 0}, {"t": 1}])
    original = reader._source_info

    def inspect_then_replace(path: Path) -> dict:
        info = original(path)
        capture(path, [{"t": 0}, {"t": 1}, {"t": 2}])
        return info

    monkeypatch.setattr(reader, "_source_info", inspect_then_replace)
    with pytest.raises(ValueError, match="changed.*immutable"):
        reduce_capture([source], tmp_path / "out")
    assert not list((tmp_path / "out").glob("*.json.gz"))


def test_source_changed_during_reduction_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import os

    from srtctl.analysis.tachometer_dashboard import reader

    source = capture(tmp_path / "input.parquet", [{"t": 0}, {"t": 1}])
    original = reader._process_batch

    def process_then_change(*args, **kwargs) -> int:
        rejected = original(*args, **kwargs)
        stat = source.stat()
        os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
        return rejected

    monkeypatch.setattr(reader, "_process_batch", process_then_change)
    with pytest.raises(ValueError, match="changed.*immutable"):
        reduce_capture([source], tmp_path / "out")
    assert not list((tmp_path / "out").glob("*.json.gz"))


@pytest.mark.parametrize("suffix", [".parquet", ".arrow"])
def test_corrupt_selected_raw_source_is_not_silently_skipped(tmp_path: Path, suffix: str) -> None:
    valid = capture(tmp_path / "valid.parquet", [{"t": 0}, {"t": 1}])
    broken = tmp_path / ("broken" + suffix)
    broken.write_bytes(b"not a valid capture")
    with pytest.raises((pa.ArrowInvalid, OSError)):
        reduce_capture([valid, broken], tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_inline_bucket_identity_ignores_absent_redundant_upper(tmp_path: Path) -> None:
    source = capture(
        tmp_path / "buckets.parquet",
        [
            {"metric_name": 'latency{le="0.1"}', "histogram_bucket_upper": upper, "t": t, "metric_value": t}
            for t, upper in ((0, 0.1), (1, None), (2, 0.1))
        ],
    )
    out = tmp_path / "out"
    catalog = reduce_capture([source], out, resolution_s=1)
    series = payload(catalog, out, "latency")["series"]
    assert len(series) == 1
    assert [point[5] for point in series[0]["points"]] == [None, 1, 1]


def test_conflicting_inline_and_column_bucket_bounds_are_rejected(tmp_path: Path) -> None:
    source = capture(
        tmp_path / "conflict.parquet",
        [{"metric_name": 'latency{le="0.1"}', "histogram_bucket_upper": 0.2, "metric_value": 1}],
    )
    with pytest.raises(ValueError, match="Conflicting histogram bounds"):
        reduce_capture([source], tmp_path / "out")


def test_current_writer_string_schemas_and_metadata_survive_compaction(tmp_path: Path) -> None:
    rows = [
        {
            "t": t,
            "time_since_start": float(t),
            "metric_value": t + 1.0,
            "metric_name": "memory_numa_MemFree{numa_node=0}",
            "metric_name_clean": "memory_numa_MemFree",
            "job_id": "example-job",
            "run_name": "example-run",
            "hostname": "node1",
            "worker_role": "prefill",
            "worker_index": "0",
            "worker_process": "1",
            "gpu": "",
            "frontend_index": "",
            "mode": "",
        }
        for t in (0, 1)
    ]
    source = capture(tmp_path / "final.parquet", rows)
    table = pq.read_table(source)
    table = table.cast(
        pa.schema(
            pa.field(item.name, pa.large_string() if pa.types.is_string(item.type) else item.type)
            for item in table.schema
        )
    )
    pq.write_table(table, source)
    out = tmp_path / "out"
    catalog = reduce_capture([source], out)
    series = payload(catalog, out, "memory_numa_MemFree")["series"][0]
    assert series["labels"] == {"numa_node": "0"}
    assert series["metadata"] == {
        "hostname": "node1",
        "job_id": "example-job",
        "run_name": "example-run",
        "worker_role": "prefill",
        "worker_index": "0",
        "worker_process": "1",
    }
    info = catalog["source_files"][0]
    assert info["size_bytes"] == source.stat().st_size
    assert info["mtime_ns"] == str(source.stat().st_mtime_ns)
    assert info["origin_anchor_min_ns"] == info["origin_anchor_max_ns"] == "1700000000000000000"
