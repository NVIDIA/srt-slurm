# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""All-family import contracts: raw values, source identities, and display semantics."""

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from srtctl.analysis.metric_catalog import COMPONENTS, GROUP_ORDER, describe_metric
from srtctl.dsight.importer import Importer
from srtctl.dsight.metrics import read_metrics

ORIGIN = 1_780_000_000_000_000_000


def sample(name, value=1.0, time=1, **labels):
    return {
        "timestamp_ns": ORIGIN + int(time * 1e9),
        "metric_name": name,
        "metric_value": value,
        "scraper_endpoint": "backend_decode0_rank0",
        "hostname": "node0",
        "worker_role": "decode",
        "worker_index": "0",
        **labels,
    }


def import_rows(tmp_path: Path, rows):
    root = tmp_path / "tachometer/local"
    root.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(rows), root / "final.parquet")
    run = Importer(tmp_path)
    run.origin, run.duration = ORIGIN, 10.0
    series = read_metrics(run)
    return run, series


def catalog(run):
    return {item["name"]: item for item in run.metric_catalog}


def test_all_names_are_cataloged_and_only_trace_window_values_imported(tmp_path):
    run, series = import_rows(
        tmp_path,
        [
            sample("new_unknown_metric", 0),
            sample("before_trace", 7, -1),
            sample("after_trace", 9, 11),
            sample("trtllm_num_requests_running", 3, 2),
        ],
    )
    assert set(catalog(run)) == {"new_unknown_metric", "before_trace", "after_trace", "trtllm_num_requests_running"}
    assert {item["name"] for item in series} == {"new_unknown_metric", "trtllm_num_requests_running"}
    unknown = next(item for item in series if item["name"] == "new_unknown_metric")
    assert unknown["points"] == [[1.0, 0, 0, 0]]
    assert unknown["value_kind"] == "stored" and unknown["unit"] == "stored value"
    for name in ("before_trace", "after_trace"):
        assert catalog(run)[name]["samples"] == catalog(run)[name]["series_count"] == 0
    assert run.audit["metric_rows_scanned"] == 4
    assert run.audit["metric_families"] == 4


def test_canonical_label_order_and_compaction_helper_do_not_split_series(tmp_path):
    run, initial = import_rows(tmp_path, [sample('novel{b="2",a="1"}', 3, metric_name_clean="novel")])
    assert len(initial) == 1
    tail = pa.Table.from_pylist([sample('novel{a="1",b="2"}', 4, 2)])
    with (
        (tmp_path / "tachometer/local/tail.arrow").open("wb") as stream,
        pa.ipc.new_stream(stream, tail.schema) as writer,
    ):
        writer.write_table(tail)
    fresh = Importer(tmp_path)
    fresh.origin, fresh.duration = ORIGIN, 10
    series = read_metrics(fresh)
    assert len(series) == 1
    assert [point[1] for point in series[0]["points"]] == [3, 4]
    assert "metric_name_clean" not in series[0]["labels"]
    assert series[0]["labels"]["metric.a"] == "1"
    assert series[0]["source_ids"] == [0, 1]


def test_histogram_bounds_are_identity_and_units_are_observations(tmp_path):
    rows = [
        sample(
            'request_seconds_bucket{le="1",stage="decode"}',
            4,
            1,
            histogram_bucket_upper=None,
            histogram_sum=999,
            histogram_count=999,
        ),
        sample(
            'request_seconds_bucket{stage="decode"}',
            5,
            2,
            histogram_bucket_upper=1.0,
            histogram_sum=999,
            histogram_count=999,
        ),
        sample(
            'request_seconds_bucket{le="+Inf",stage="decode"}',
            8,
            2,
            histogram_bucket_upper=float("inf"),
            histogram_sum=999,
            histogram_count=999,
        ),
    ]
    run, series = import_rows(tmp_path, rows)
    assert set(catalog(run)) == {"request_seconds"}
    assert len(series) == 2
    by_bound = {item["labels"]["metric.le"]: item for item in series}
    assert set(by_bound) == {"1.0", "+Inf"}
    assert [point[1] for point in by_bound["1.0"]["points"]] == [4, 5]
    assert all(item["value_kind"] == "histogram" and item["unit"] == "observations" for item in series)
    entry = catalog(run)["request_seconds"]
    assert entry["unit"] == "observations" and entry["observation_unit"] == "s"
    assert entry["samples"] == 3 and entry["series_count"] == 2
    assert not any("sum" in name or "count" in name for name in catalog(run))


def test_column_only_histogram_bounds_keep_float_precision(tmp_path):
    _, series = import_rows(
        tmp_path,
        [
            sample("latency_seconds", 2, histogram_bucket_upper=1.00000001),
            sample("latency_seconds", 3, histogram_bucket_upper=1.00000002),
        ],
    )
    assert len(series) == 2
    assert {item["labels"]["metric.le"] for item in series} == {"1.00000001", "1.00000002"}


@pytest.mark.parametrize("time", [1, 20])
def test_conflicting_bounds_fail_even_outside_selected_window(tmp_path, time):
    with pytest.raises(ValueError, match="Conflicting histogram bounds"):
        import_rows(tmp_path, [sample('latency_seconds{le="1"}', time=time, histogram_bucket_upper=2.0)])


@pytest.mark.parametrize("bound", ["NaN", "wrong"])
def test_invalid_inline_bounds_are_rejected(tmp_path, bound):
    with pytest.raises(ValueError, match="Invalid histogram bound"):
        import_rows(tmp_path, [sample(f'latency_seconds{{le="{bound}"}}')])


@pytest.mark.parametrize("name", [None, "", '{label="value"}'])
def test_missing_metric_names_fail_clearly(tmp_path, name):
    with pytest.raises(ValueError, match="Missing raw metric name"):
        import_rows(tmp_path, [sample(name)])


def test_mixed_histogram_and_scalar_units_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="mixes scalar values and histogram buckets"):
        import_rows(tmp_path, [sample('latency_seconds{le="1"}'), sample("latency_seconds", 2, 20)])


def test_summary_quantiles_remain_raw_scalar_samples(tmp_path):
    run, series = import_rows(
        tmp_path,
        [
            sample('go_gc_duration_seconds{quantile="0.99"}', 0.4),
            sample("go_gc_duration_seconds_sum", 17),
            sample("go_gc_duration_seconds_count", 23),
        ],
    )
    summary = next(item for item in series if item["name"] == "go_gc_duration_seconds")
    assert summary["value_kind"] == "stored"
    assert summary["labels"]["metric.quantile"] == "0.99"
    assert summary["points"][0][1] == 0.4
    assert catalog(run)["go_gc_duration_seconds_count"]["unit"] == "observations"
    assert len(series) == 3


def test_counter_reset_is_preserved_and_unknown_monotonic_values_are_not_typed(tmp_path):
    run, series = import_rows(
        tmp_path,
        [
            sample("work_total", 12, 1),
            sample("work_total", 2, 2),
            sample("new_unknown_metric", 1, 1),
            sample("new_unknown_metric", 2, 2),
        ],
    )
    counter = next(item for item in series if item["name"] == "work_total")
    assert [point[1] for point in counter["points"]] == [12, 2]
    assert counter["value_kind"] == "counter"
    assert catalog(run)["new_unknown_metric"]["value_kind"] == "stored"
    assert "raw cumulative" in catalog(run)["work_total"]["description"]


def test_conflicting_observations_keep_every_distinct_value_and_source_row(tmp_path):
    run, series = import_rows(
        tmp_path,
        [
            sample("work_total", 1, 2),
            sample("work_total", 3, 1),
            sample("work_total", 4, 1),
            sample("work_total", 4, 1),
            sample("work_total", 5, 3),
        ],
    )
    item = series[0]
    assert item["points"] == [[1.0, 3, 0, 1], [1.0, 4, 0, 2], [2.0, 1, 0, 0], [3.0, 5, 0, 4]]
    assert item["conflict_timestamps"] == [1.0]
    assert item["conflicting_samples"] == run.audit["conflicting_metric_samples"] == 2
    assert run.audit["duplicate_metric_points"] == 1
    assert catalog(run)["work_total"]["samples"] == 4
    assert "Conflicting raw values" in catalog(run)["work_total"]["quality"]


def test_nonfinite_only_family_stays_in_catalog(tmp_path):
    run, series = import_rows(tmp_path, [sample("unavailable", float("nan")), sample("unavailable", None)])
    assert series == []
    assert catalog(run)["unavailable"]["samples"] == 0
    assert run.audit["nonfinite_metric_points"] == 2


def test_source_ids_are_stable_across_reimports(tmp_path):
    run, first = import_rows(tmp_path, [sample(f"metric_{index % 3}", index, index % 9) for index in range(30)])
    second = read_metrics(run)
    assert [(item["name"], item["id"], item["points"]) for item in first] == [
        (item["name"], item["id"], item["points"]) for item in second
    ]


@pytest.mark.parametrize(
    ("name", "component", "group"),
    [
        ("dynamo_frontend_requests_total", "Frontend", "Requests and latency"),
        ("dynamo_frontend_tokenizer_cache_hits_total", "Frontend", "Tokenizer cache"),
        ("dynamo_frontend_router_queue_pending_requests", "Router", "Queue and backpressure"),
        ("trtllm_num_requests_running", "Workers", "Engine scheduling"),
        ("trtllm_kv_cache_utilization", "Workers", "Engine KV cache"),
        ("DCGM_FI_DEV_GPU_UTIL", "GPU", "Utilization"),
        ("FI_DEV_FB_USED", "GPU", "Framebuffer memory"),
        ("memory_numa_Active", "Host", "NUMA memory and locality"),
        ("namedprocess_namegroup_thread_cpu_seconds_total", "Host", "Process CPU and scheduling"),
        ("process_cpu_seconds_total", "Host", "Collection health"),
    ],
)
def test_taxonomy_retains_reference_categories(name, component, group):
    info = describe_metric(name)
    assert info["component"] == component and info["group"] == group
    assert component in COMPONENTS
    assert info["group_order"] == GROUP_ORDER[component].index(group)
    assert isinstance(info["order"], int)


def test_semantic_exceptions_and_unknown_category_are_preserved():
    assert describe_metric("cpu_time_user")["counter"] is False
    assert describe_metric("FI_DEV_NVLINK_BANDWIDTH_TOTAL")["counter"] is False
    assert describe_metric("gpu_mem_total")["unit"] == "stored value"
    assert describe_metric("new_family", ["frontend0"])["component"] == "Frontend"
    assert describe_metric("new_family", ["backend_decode0"])["component"] == "Workers"
    assert describe_metric("new_family", ["other"])["group"] == "Other captured metrics"
    assert "counter rate" not in describe_metric("dynamo_frontend_disconnected_clients")["quality"]


def test_vmstat_state_fields_are_not_classified_as_event_counters(tmp_path):
    run, _ = import_rows(
        tmp_path,
        [
            sample("node_vmstat_nr_free_pages", 42),
            sample("vmstat_pgfault", 123),
            sample("vmstat_unknown_future_field", 99),
        ],
    )
    assert catalog(run)["node_vmstat_nr_free_pages"]["value_kind"] == "stored"
    assert catalog(run)["vmstat_pgfault"]["value_kind"] == "counter"
    assert catalog(run)["vmstat_unknown_future_field"]["value_kind"] == "stored"
