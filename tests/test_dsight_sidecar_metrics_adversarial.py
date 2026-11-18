# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Edge cases of DSight's sidecar-mode worker metrics (docs/dsight-data-flow.md).

``tests/test_dsight_sidecar_metrics.py`` covers one capture per engine. These
tests probe the catalog rows for every native vLLM and SGLang family, the
precedence of the rank keys, the worker-only scope of the data-parallel keys,
and series identity across a worker's sidecar and engine endpoints.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from srtctl.analysis.metric_catalog import _ENGINE_FAMILIES, GROUP_ORDER, describe_metric, is_counter
from srtctl.dsight.importer import Importer

ORIGIN = 1_780_000_000_000_000_000
SIDECAR, ENGINE = "backend_decode0_rank0", "backend_decode0_rank0_engine"
WORKER = {"hostname": "n0", "worker_index": "0", "worker_process": "0", "worker_role": "decode"}

# How each catalog key is exported upstream, checked by hand against the pinned sources:
# vLLM b272cff368 rust/src/metrics/src/{scheduler,request}.rs registers counters without
# `_total` and prometheus-client appends it; SGLang 7591118794
# python/sglang/srt/observability/metrics_collector.py names its counters with `_total`.
# vLLM ac9126e5 v1/metrics/loggers.py, kv_connector/v1/nixl/stats.py and
# kv_connector/v1/mooncake/store/metrics.py declare the prompt-source and KV-transfer families.
# The catalog key is the sample name Tachometer records (the `_bucket` suffix and the
# `le` label are stripped from histogram rows; `_total` is kept).
UPSTREAM_KIND = {
    "vllm:num_requests_running": "gauge",
    "vllm:num_requests_waiting": "gauge",
    "vllm:num_requests_waiting_by_reason": "gauge",
    "vllm:num_preemptions_total": "counter",
    "vllm:iteration_tokens_total": "histogram",
    "vllm:kv_cache_usage_perc": "gauge",
    "vllm:prefix_cache_queries_total": "counter",
    "vllm:prefix_cache_hits_total": "counter",
    "vllm:external_prefix_cache_queries_total": "counter",
    "vllm:external_prefix_cache_hits_total": "counter",
    "vllm:prompt_tokens_cached_by_source_total": "counter",
    "vllm:spec_decode_num_drafts_total": "counter",
    "vllm:spec_decode_num_draft_tokens_total": "counter",
    "vllm:spec_decode_num_accepted_tokens_total": "counter",
    "vllm:request_success_total": "counter",
    "vllm:prompt_tokens_total": "counter",
    "vllm:generation_tokens_total": "counter",
    "vllm:time_to_first_token_seconds": "histogram",
    "vllm:inter_token_latency_seconds": "histogram",
    "vllm:e2e_request_latency_seconds": "histogram",
    "vllm:request_queue_time_seconds": "histogram",
    "vllm:request_prefill_time_seconds": "histogram",
    "vllm:request_decode_time_seconds": "histogram",
    "vllm:request_inference_time_seconds": "histogram",
    "vllm:request_time_per_output_token_seconds": "histogram",
    "vllm:prompt_tokens_by_source_total": "counter",
    "vllm:prompt_tokens_cached_total": "counter",
    "vllm:request_prefill_kv_computed_tokens": "histogram",
    "vllm:nixl_xfer_time_seconds": "histogram",
    "vllm:nixl_post_time_seconds": "histogram",
    "vllm:nixl_bytes_transferred": "histogram",
    "vllm:nixl_num_descriptors": "histogram",
    "vllm:nixl_num_failed_transfers_total": "counter",
    "vllm:nixl_num_failed_notifications_total": "counter",
    "vllm:nixl_num_kv_expired_reqs_total": "counter",
    "vllm:mooncake_store_operation_total": "counter",
    "vllm:mooncake_store_operation_keys_total": "counter",
    "vllm:mooncake_store_operation_failed_keys_total": "counter",
    "vllm:mooncake_store_operation_bytes_total": "counter",
    "vllm:mooncake_store_operation_time_seconds": "histogram",
    "sglang:num_running_reqs": "gauge",
    "sglang:num_queue_reqs": "gauge",
    "sglang:num_grammar_queue_reqs": "gauge",
    "sglang:gen_throughput": "gauge",
    "sglang:token_usage": "gauge",
    "sglang:num_used_tokens": "gauge",
    "sglang:max_total_num_tokens": "gauge",
    "sglang:cache_hit_rate": "gauge",
    "sglang:cached_tokens_total": "counter",
    "sglang:num_prefill_bootstrap_queue_reqs": "gauge",
    "sglang:num_prefill_inflight_queue_reqs": "gauge",
    "sglang:num_decode_prealloc_queue_reqs": "gauge",
    "sglang:num_decode_transfer_queue_reqs": "gauge",
    "sglang:kv_transfer_speed_gb_s": "histogram",
    "sglang:kv_transfer_latency_ms": "histogram",
    "sglang:num_requests_total": "counter",
    "sglang:prompt_tokens_total": "counter",
    "sglang:generation_tokens_total": "counter",
    "sglang:queue_time_seconds": "histogram",
    "sglang:time_to_first_token_seconds": "histogram",
    "sglang:inter_token_latency_seconds": "histogram",
    "sglang:e2e_request_latency_seconds": "histogram",
    "sglang:request_time_per_output_token_seconds": "histogram",
}


def write_capture(logs: Path, endpoints: dict[str, dict[str, str]], samples: list[tuple]) -> None:
    """A Tachometer ``backend``-filter capture: node metadata become columns, labels stay inline.

    ``endpoints`` maps endpoint name to its node metadata (any keys, e.g. extra_metadata);
    a sample is ``(endpoint, raw metric name, value[, histogram upper bound])``.
    """
    lines = ['storage = "tachometer/raw/scrape"', ""]
    for name, metadata in endpoints.items():
        lines += [
            "[[endpoints]]",
            f"name = {json.dumps(name)}",
            'url = "http://h:1/metrics"',
            "collect_interval_ms = 1000",
        ]
        lines += [
            'filter = "backend"' if name.startswith("backend") else 'filter = "frontend"',
            "[endpoints.node_metadata]",
        ]
        lines += [f"{json.dumps(k)} = {json.dumps(v)}" for k, v in sorted(metadata.items())]
        lines.append("")
    (logs / "tachometer_config.toml").write_text("\n".join(lines))
    columns = sorted({key for metadata in endpoints.values() for key in metadata})
    rows = []
    for endpoint, metric, value, *bound in samples:
        for time in (1, 2):
            rows.append(
                {
                    "scraper_endpoint": endpoint,
                    "metric_name": metric,
                    "metric_value": float(value * time),
                    "histogram_bucket_upper": bound[0] if bound else None,
                    "timestamp_ns": ORIGIN + time * 1_000_000_000,
                    **{key: endpoints[endpoint].get(key) for key in columns},
                }
            )
    schema = pa.schema(
        [
            ("scraper_endpoint", pa.string()),
            ("metric_name", pa.string()),
            ("metric_value", pa.float64()),
            ("histogram_bucket_upper", pa.float64()),
            ("timestamp_ns", pa.int64()),
            *((key, pa.string()) for key in columns),
        ]
    )
    root = logs / "tachometer/local"
    root.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), root / "final.parquet")


def imported(logs: Path) -> dict[str, Any]:
    return Importer(logs).run()


def one(data: dict[str, Any], name: str, **match: str) -> dict[str, Any]:
    (item,) = [
        s
        for s in data["metrics"]
        if s["name"] == name and all(s["labels"].get(k) == v or s.get(k) == v for k, v in match.items())
    ]
    return item


# --- Catalog rows -------------------------------------------------------------------------------


def test_the_kind_table_covers_every_engine_family() -> None:
    assert set(UPSTREAM_KIND) == set(_ENGINE_FAMILIES)


@pytest.mark.parametrize("name", sorted(_ENGINE_FAMILIES))
def test_every_engine_family_is_a_described_worker_engine_metric(name: str) -> None:
    group, unit, title = _ENGINE_FAMILIES[name]
    described = describe_metric(name, [ENGINE])

    assert (described["component"], described["group"]) == ("Workers", group)
    assert group in GROUP_ORDER["Workers"] and described["group_order"] < 100
    assert described["title"] == title and title.strip()
    assert described["unit"] == unit and unit not in ("", "stored value")
    engine = "vLLM" if name.startswith("vllm:") else "SGLang"
    assert described["description"].startswith(f"{engine} engine measurements")
    # Placement does not depend on which endpoint served the family.
    assert describe_metric(name, [SIDECAR, "frontend0"])["group"] == group
    if UPSTREAM_KIND[name] == "counter":
        assert name.endswith("_total") and is_counter(name)
    elif UPSTREAM_KIND[name] == "gauge":
        assert not is_counter(name)
        if group in {"Engine scheduling", "Engine iterations", "Engine KV cache"}:
            assert "latest published engine statistics" in described["description"]


def test_engine_created_series_are_timestamps_not_their_family() -> None:
    for name in ("sglang:num_requests_created", "vllm:prompt_tokens_created"):
        described = describe_metric(name, [ENGINE])
        assert described["component"] == "Workers"
        assert described["unit"] == "epoch s"
        assert "creation timestamp" in described["quality"]
        assert described["title"] not in {row[2] for row in _ENGINE_FAMILIES.values()}


# --- Import: histograms, identity, ranks -------------------------------------------------------


def test_engine_histograms_keep_their_observation_unit(tmp_path) -> None:
    labels = 'engine="0",model_name="m"'
    write_capture(
        tmp_path,
        {ENGINE: WORKER},
        [
            (ENGINE, f'vllm:iteration_tokens_total{{{labels},le="64"}}', 3, 64.0),
            (ENGINE, f'vllm:iteration_tokens_total{{{labels},le="+Inf"}}', 4),
            (ENGINE, 'sglang:kv_transfer_latency_ms{dp_rank="1",le="10"}', 1, 10.0),
            (ENGINE, 'sglang:kv_transfer_speed_gb_s{dp_rank="1",le="+Inf"}', 2),
        ],
    )
    data = imported(tmp_path)
    catalog = {item["name"]: item for item in data["metric_catalog"]}

    for name, unit, group in (
        ("vllm:iteration_tokens_total", "tokens", "Engine iterations"),
        ("sglang:kv_transfer_latency_ms", "ms", "Engine KV transfer"),
        ("sglang:kv_transfer_speed_gb_s", "GB/s", "Engine KV transfer"),
    ):
        entry = catalog[name]
        assert (entry["value_kind"], entry["unit"], entry["observation_unit"], entry["group"]) == (
            "histogram",
            "observations",
            unit,
            group,
        ), name
    # A `_total` histogram is still one histogram family, not a counter family plus buckets.
    assert not any(name.startswith("vllm:iteration_tokens_total_") for name in catalog)
    buckets = [s for s in data["metrics"] if s["name"] == "vllm:iteration_tokens_total"]
    assert sorted(s["labels"]["metric.le"] for s in buckets) == ["+Inf", "64.0"]
    assert {(s["rank"], s["rank_kind"]) for s in buckets} == {("0", "dp")}


def test_identical_samples_from_the_sidecar_and_engine_endpoints_stay_two_series(tmp_path) -> None:
    write_capture(
        tmp_path,
        {SIDECAR: WORKER, ENGINE: WORKER},
        [
            (SIDECAR, 'process_resident_memory_bytes{pid="1"}', 10),
            (ENGINE, 'process_resident_memory_bytes{pid="1"}', 20),
        ],
    )
    data = imported(tmp_path)

    found = [s for s in data["metrics"] if s["name"] == "process_resident_memory_bytes"]
    assert sorted((s["endpoint"], [p[1] for p in s["points"]]) for s in found) == [
        (SIDECAR, [10.0, 20.0]),
        (ENGINE, [20.0, 40.0]),
    ]
    assert {s["worker"] for s in found} == {"decode-0"}
    entry = next(item for item in data["metric_catalog"] if item["name"] == "process_resident_memory_bytes")
    assert entry["endpoints"] == [SIDECAR, ENGINE]


@pytest.mark.parametrize(
    ("labels", "expected"),
    [
        pytest.param('global_rank="3",dp_rank="1"', ("3", "global_rank"), id="global_rank-over-dp_rank"),
        pytest.param('rank="2",engine="1"', ("2", "rank"), id="rank-over-engine"),
        pytest.param('dp_rank="1",engine="5"', ("1", "dp"), id="dp_rank-over-engine"),
        pytest.param('engine="7"', ("7", "dp"), id="engine-alone"),
        pytest.param('tp_rank="4",pp_rank="0",moe_ep_rank="2"', (None, None), id="other-ranks-stay-labels"),
    ],
)
def test_worker_rank_keys_resolve_in_documented_precedence(tmp_path, labels: str, expected: tuple) -> None:
    write_capture(tmp_path, {ENGINE: WORKER}, [(ENGINE, f"vllm:num_requests_running{{{labels}}}", 1)])
    item = one(imported(tmp_path), "vllm:num_requests_running")
    assert (item["rank"], item["rank_kind"]) == expected


@pytest.mark.parametrize("labels", ['dp_rank="1"', 'engine="1"', 'dp_rank="1",engine="2"'])
def test_data_parallel_keys_never_scope_frontend_or_unattributed_series(tmp_path, labels: str) -> None:
    write_capture(
        tmp_path,
        {"frontend0": {"frontend_index": "0", "hostname": "n9"}, "kv-router_n8": {"hostname": "n8", "service": "r"}},
        [
            ("frontend0", f"dynamo_frontend_worker_active_decode_blocks{{{labels}}}", 3),
            ("kv-router_n8", f"dynamo_router_overhead_seconds{{{labels}}}", 1),
        ],
    )
    data = imported(tmp_path)
    for item in data["metrics"]:
        assert item["worker"] in ("frontend", None)
        assert (item["rank"], item["rank_kind"]) == (None, None), item["endpoint"]


def test_hybrid_dp_engines_on_two_nodes_are_distinct_ranks_of_one_worker(tmp_path) -> None:
    """vLLM hybrid DP: each node's frontend labels its local engines with their global DP ranks."""
    node1 = {**WORKER, "hostname": "n1", "worker_process": "4"}
    write_capture(
        tmp_path,
        {ENGINE: WORKER, "backend_decode0_rank4_engine": node1},
        [
            *((ENGINE, f'vllm:num_requests_running{{engine="{dp}",model_name="m"}}', 1) for dp in range(4)),
            *(
                ("backend_decode0_rank4_engine", f'vllm:num_requests_running{{engine="{dp}",model_name="m"}}', 1)
                for dp in range(4, 8)
            ),
        ],
    )
    data = imported(tmp_path)
    running = [s for s in data["metrics"] if s["name"] == "vllm:num_requests_running"]

    assert {s["worker"] for s in running} == {"decode-0"}
    assert sorted(int(s["rank"]) for s in running) == list(range(8))
    assert {s["host"] for s in running if int(s["rank"]) >= 4} == {"n1"}


def test_extra_metadata_named_engine_is_not_a_dp_rank(tmp_path) -> None:
    metadata = {**WORKER, "engine": "sglang"}
    write_capture(
        tmp_path,
        {SIDECAR: metadata, ENGINE: metadata},
        [
            (SIDECAR, 'dynamo_component_requests_total{dynamo_component="backend"}', 5),
            (ENGINE, 'sglang:num_running_reqs{dp_rank="1",model_name="m"}', 3),
        ],
    )
    data = imported(tmp_path)

    # The engine's own dp_rank label still scopes its series ...
    assert (one(data, "sglang:num_running_reqs")["rank"], one(data, "sglang:num_running_reqs")["rank_kind"]) == (
        "1",
        "dp",
    )
    # ... but a sidecar runtime series has no rank: static metadata is not one.
    sidecar = one(data, "dynamo_component_requests_total")
    assert (sidecar["rank"], sidecar["rank_kind"]) == (None, None)
