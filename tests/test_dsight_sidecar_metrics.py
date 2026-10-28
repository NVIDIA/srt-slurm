# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sidecar-mode Tachometer captures: one worker, a Dynamo sidecar endpoint and a native engine endpoint.

Rows follow the scraper's ``backend`` filter: the metric name keeps its Prometheus
labels (``le`` included for histogram buckets, whose bound is also in
``histogram_bucket_upper``; ``+Inf`` has no upper bound) and each endpoint's
``node_metadata`` becomes columns. Endpoint names follow the launch contract:
``backend_{mode}{index}_rank{rank}`` for the sidecar's system status server and
``..._engine`` for the engine's own Prometheus server.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from srtctl.analysis.metric_catalog import describe_metric
from srtctl.dsight.importer import Importer
from srtctl.dsight.query import TraceDataset

ORIGIN = 1_780_000_000_000_000_000
DYNAMO = 'dynamo_component="backend",dynamo_endpoint="generate",dynamo_namespace="dynamo"'


def endpoint(name: str, host: str, role: str, index: int, process: int) -> dict[str, Any]:
    metadata = {"hostname": host, "worker_index": str(index), "worker_process": str(process), "worker_role": role}
    return {"name": name, "url": f"http://{host}:9000/metrics", "filter": "backend", "node_metadata": metadata}


def write_capture(logs: Path, endpoints: list[dict[str, Any]], samples: list[tuple]) -> None:
    lines = ['storage = "tachometer/raw/scrape"', ""]
    for item in endpoints:
        lines += ["[[endpoints]]", f"name = {json.dumps(item['name'])}", f"url = {json.dumps(item['url'])}"]
        lines += ["collect_interval_ms = 1000", f"filter = {json.dumps(item['filter'])}", "[endpoints.node_metadata]"]
        lines += [f"{json.dumps(k)} = {json.dumps(v)}" for k, v in sorted(item["node_metadata"].items())]
        lines.append("")
    (logs / "tachometer_config.toml").write_text("\n".join(lines))
    by_name = {item["name"]: item for item in endpoints}
    rows = []
    for name, metric, value, *bound in samples:
        upper = bound[0] if bound else None
        for time in (1, 2):
            rows.append(
                {
                    "scraper_endpoint": name,
                    "metric_name": metric,
                    "metric_value": float(value * time),
                    "histogram_bucket_upper": upper,
                    "timestamp_ns": ORIGIN + time * 1_000_000_000,
                    **by_name[name]["node_metadata"],
                }
            )
    root = logs / "tachometer/local"
    root.mkdir(parents=True)
    schema = pa.schema(
        [
            ("scraper_endpoint", pa.string()),
            ("metric_name", pa.string()),
            ("metric_value", pa.float64()),
            ("histogram_bucket_upper", pa.float64()),
            ("timestamp_ns", pa.int64()),
            ("hostname", pa.string()),
            ("worker_index", pa.string()),
            ("worker_process", pa.string()),
            ("worker_role", pa.string()),
        ]
    )
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), root / "final.parquet")


def sidecar_samples(name: str, kv_relay: bool) -> list[tuple]:
    samples = [
        (name, f"dynamo_component_requests_total{{{DYNAMO}}}", 5),
        (name, f"dynamo_component_inflight_requests{{{DYNAMO}}}", 2),
        (name, f'dynamo_component_request_duration_seconds{{{DYNAMO},le="1"}}', 3, 1.0),
        (name, f'dynamo_component_request_duration_seconds{{{DYNAMO},le="+Inf"}}', 4),
        (name, f"dynamo_component_uptime_seconds{{{DYNAMO}}}", 60),
        (name, "dynamo_work_handler_queue_depth", 0),
    ]
    if kv_relay:
        samples.append(
            (name, f'dynamo_component_kv_publisher_zmq_events_total{{{DYNAMO},event_type="stored",stage="seen"}}', 9)
        )
    return samples


def run(logs: Path) -> tuple[dict[str, Any], TraceDataset]:
    data = Importer(logs).run()
    return data, TraceDataset(data)


def series(data: dict[str, Any], name: str) -> list[dict[str, Any]]:
    return [item for item in data["metrics"] if item["name"] == name]


def catalog(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {item["name"]: item for item in data["metric_catalog"]}


def placement(data: dict[str, Any], name: str) -> tuple[str, str]:
    entry = catalog(data)[name]
    return entry["component"], entry["group"]


def test_vllm_sidecar_and_engine_endpoints_share_one_worker_and_keep_dp_engines(tmp_path):
    sidecar, engine = "backend_decode0_rank0", "backend_decode0_rank0_engine"
    labels = 'model_name="m",engine="{}"'
    write_capture(
        tmp_path,
        [endpoint(sidecar, "n0", "decode", 0, 0), endpoint(engine, "n0", "decode", 0, 0)],
        [
            *sidecar_samples(sidecar, kv_relay=True),
            *((engine, f"vllm:num_requests_running{{{labels.format(dp)}}}", 3 + dp) for dp in (0, 1)),
            *((engine, f"vllm:num_requests_waiting{{{labels.format(dp)}}}", dp) for dp in (0, 1)),
            (engine, f"vllm:kv_cache_usage_perc{{{labels.format(0)}}}", 0.25),
            (engine, f"vllm:prompt_tokens_total{{{labels.format(0)}}}", 100),
            (engine, f'vllm:time_to_first_token_seconds{{{labels.format(0)},le="0.1"}}', 1, 0.1),
            (engine, f'vllm:time_to_first_token_seconds{{{labels.format(0)},le="+Inf"}}', 2),
            (engine, 'http_requests_total{handler="/v1/completions",method="POST",status="2xx"}', 7),
        ],
    )
    data, trace = run(tmp_path)

    assert data["capabilities"]["metrics"] and data["capabilities"]["worker_metrics"]
    assert [w["id"] for w in data["workers"]] == ["decode-0"]
    assert {item["worker"] for item in data["metrics"]} == {"decode-0"}
    # Two endpoints for one worker stay distinct series; neither family lists the other endpoint.
    assert {item["endpoint"] for item in series(data, "dynamo_component_requests_total")} == {sidecar}
    assert catalog(data)["vllm:num_requests_running"]["endpoints"] == [engine]
    assert catalog(data)["dynamo_component_requests_total"]["endpoints"] == [sidecar]

    running = {item["labels"]["metric.engine"]: item for item in series(data, "vllm:num_requests_running")}
    assert set(running) == {"0", "1"}
    assert sorted((item["rank"], item["rank_kind"]) for item in running.values()) == [("0", "dp"), ("1", "dp")]
    assert [point[1] for point in running["1"]["points"]] == [4.0, 8.0]
    dp1 = trace.query("metrics", worker="decode-0", name="vllm:num_requests_running", rank=1)
    assert dp1["total"] == 1 and dp1["items"][0]["labels"]["metric.engine"] == "1"
    assert all(item["rank"] is None for item in series(data, "dynamo_component_requests_total"))

    assert placement(data, "vllm:num_requests_running") == ("Workers", "Engine scheduling")
    assert placement(data, "vllm:kv_cache_usage_perc") == ("Workers", "Engine KV cache")
    assert placement(data, "vllm:prompt_tokens_total") == ("Workers", "Engine requests and latency")
    assert catalog(data)["vllm:prompt_tokens_total"]["value_kind"] == "counter"
    ttft = catalog(data)["vllm:time_to_first_token_seconds"]
    assert (ttft["component"], ttft["group"], ttft["value_kind"]) == (
        "Workers",
        "Engine requests and latency",
        "histogram",
    )
    assert ttft["observation_unit"] == "s"
    assert {item["labels"]["metric.le"] for item in series(data, "vllm:time_to_first_token_seconds")} == {
        "0.1",
        "+Inf",
    }
    assert placement(data, "dynamo_component_requests_total") == ("Workers", "Request lifecycle")
    assert placement(data, "dynamo_component_uptime_seconds") == ("Workers", "Component lifecycle")
    assert placement(data, "dynamo_work_handler_queue_depth") == ("Workers", "Admission and queues")
    relay = catalog(data)["dynamo_component_kv_publisher_zmq_events_total"]
    assert (relay["component"], relay["group"], relay["unit"]) == ("Workers", "Engine KV events", "events")
    # Unlisted engine-endpoint families remain worker evidence under their raw names.
    assert placement(data, "http_requests_total") == ("Workers", "Other captured metrics")


def test_sglang_leader_and_follower_engine_endpoints_scope_dp_ranks_to_their_hosts(tmp_path):
    leader, follower = "backend_prefill0_rank0", "backend_prefill0_rank1"
    labels = 'dp_rank="{}",engine_type="prefill",model_name="m",moe_ep_rank="0",pp_rank="0",tp_rank="{}"'
    write_capture(
        tmp_path,
        [
            endpoint(leader, "n0", "prefill", 0, 0),
            endpoint(f"{leader}_engine", "n0", "prefill", 0, 0),
            endpoint(f"{follower}_engine", "n1", "prefill", 0, 1),
        ],
        [
            *sidecar_samples(leader, kv_relay=True),
            *(
                (f"{name}_engine", f"{metric}{{{labels.format(dp, dp * 4)}}}", value)
                for name, dp in ((leader, 0), (follower, 1))
                for metric, value in (
                    ("sglang:num_running_reqs", 6),
                    ("sglang:num_queue_reqs", 2),
                    ("sglang:token_usage", 0.5),
                    ("sglang:num_prefill_bootstrap_queue_reqs", 1),
                )
            ),
            (f"{leader}_engine", 'sglang:time_to_first_token_seconds{is_streaming="true",model_name="m",le="1"}', 4, 1.0),
            (f"{leader}_engine", 'sglang:prompt_tokens_total{is_streaming="true",model_name="m"}', 50),
        ],
    )
    data, trace = run(tmp_path)

    assert data["capabilities"]["worker_metrics"]
    worker = next(w for w in data["workers"] if w["id"] == "prefill-0")
    assert [w["id"] for w in data["workers"]] == ["prefill-0"] and sorted(worker["hosts"]) == ["n0", "n1"]
    running = {item["host"]: item for item in series(data, "sglang:num_running_reqs")}
    assert set(running) == {"n0", "n1"}
    assert {host: (item["rank"], item["rank_kind"], item["worker_process"]) for host, item in running.items()} == {
        "n0": ("0", "dp", "0"),
        "n1": ("1", "dp", "1"),
    }
    assert running["n1"]["endpoint"] == f"{follower}_engine"
    assert running["n1"]["labels"]["metric.tp_rank"] == "4"
    follower_rank = trace.query("metrics", worker="prefill-0", name="sglang:num_running_reqs", rank=1)
    assert [item["host"] for item in follower_rank["items"]] == ["n1"]

    assert placement(data, "sglang:num_running_reqs") == ("Workers", "Engine scheduling")
    assert placement(data, "sglang:token_usage") == ("Workers", "Engine KV cache")
    assert placement(data, "sglang:num_prefill_bootstrap_queue_reqs") == ("Workers", "Engine KV transfer")
    assert placement(data, "sglang:time_to_first_token_seconds") == ("Workers", "Engine requests and latency")
    assert catalog(data)["sglang:time_to_first_token_seconds"]["observation_unit"] == "s"
    assert catalog(data)["sglang:prompt_tokens_total"]["unit"] == "tokens"
    assert placement(data, "dynamo_component_kv_publisher_zmq_events_total") == ("Workers", "Engine KV events")


def test_trtllm_sidecar_only_capture_supplies_worker_metrics_without_engine_families(tmp_path):
    sidecar = "backend_decode0_rank0"
    write_capture(tmp_path, [endpoint(sidecar, "n0", "decode", 0, 0)], sidecar_samples(sidecar, kv_relay=False))
    data, _ = run(tmp_path)

    assert data["capabilities"]["metrics"] and data["capabilities"]["worker_metrics"]
    assert [w["id"] for w in data["workers"]] == ["decode-0"]
    assert {item["endpoint"] for item in data["metrics"]} == {sidecar}
    assert not any(name.startswith("trtllm_") for name in catalog(data))
    assert {item["component"] for item in data["metric_catalog"]} == {"Workers"}
    assert catalog(data)["dynamo_component_inflight_requests"]["title"] == "Worker in flight"


def test_one_concept_shares_group_unit_and_title_across_engines():
    concepts = {
        "running": ("trtllm_num_requests_running", "vllm:num_requests_running", "sglang:num_running_reqs"),
        "waiting": ("trtllm_num_requests_waiting", "vllm:num_requests_waiting", "sglang:num_queue_reqs"),
        "kv usage": ("trtllm_kv_cache_utilization", "vllm:kv_cache_usage_perc", "sglang:token_usage"),
        "ttft": (
            "trtllm_time_to_first_token_seconds",
            "vllm:time_to_first_token_seconds",
            "sglang:time_to_first_token_seconds",
        ),
        "lifetime": (
            "trtllm_e2e_request_latency_seconds",
            "vllm:e2e_request_latency_seconds",
            "sglang:e2e_request_latency_seconds",
        ),
        "tpot": (
            "trtllm_time_per_output_token_seconds",
            "vllm:request_time_per_output_token_seconds",
            "sglang:request_time_per_output_token_seconds",
        ),
    }
    for concept, names in concepts.items():
        described = [describe_metric(name) for name in names]
        shape = {(d["component"], d["group"], d["unit"], d["title"]) for d in described}
        assert len(shape) == 1, (concept, shape)
    assert describe_metric("vllm:num_requests_running")["description"].startswith("vLLM engine measurements")
    assert len({describe_metric(name)["quality"] for name in concepts["tpot"]}) == 1
    future = describe_metric("sglang:some_future_gauge")
    assert (future["component"], future["group"], future["title"]) == (
        "Workers",
        "Other captured metrics",
        "Some future gauge",
    )


def test_router_dp_rank_labels_do_not_scope_frontend_series(tmp_path):
    write_capture(
        tmp_path,
        [{"name": "frontend0", "url": "http://n0:8000/metrics", "filter": "frontend", "node_metadata": {}}],
        [("frontend0", 'dynamo_frontend_worker_active_decode_blocks{dp_rank="1",worker_id="7"}', 3)],
    )
    data, _ = run(tmp_path)
    (item,) = data["metrics"]
    assert item["worker"] == "frontend" and item["rank"] is None
    assert item["labels"]["metric.dp_rank"] == "1"
