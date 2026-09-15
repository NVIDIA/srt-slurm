# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Coverage for the optional local ruter route-decision viewer."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from threading import Thread
from urllib.request import urlopen

import pyarrow as pa
import pyarrow.parquet as pq

from srtctl.ruter.normalize import normalize_run
from srtctl.ruter.view import _handler, _router_settings, load_view_data


def _write(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents)


def _json_line(value: dict[str, object]) -> str:
    return json.dumps(value, separators=(",", ":")) + "\n"


def _formula(worker_id: int, cached: float, score: float) -> str:
    return (
        f"Formula for worker_id={worker_id} dp_rank=0 with {cached:.3f} effective cached blocks: "
        f"{score:.3f} = prefill_load_scale * adjusted_prefill_blocks + decode_blocks + "
        "active_request_cost_blocks = 1.000 * 4.000 + 8.000 + 0.000 "
        f"(raw_prefill_blocks: 12.000, overlap_credit_blocks: {cached:.3f}, overlap_credit_decay: 1.000)"
    )


def _make_run(root: Path) -> Path:
    router = root / "logs" / "router.log"
    _write(
        router,
        "".join(
            [
                _json_line(
                    {
                        "time": "2026-08-20T00:00:02.000000Z",
                        "message": _formula(42, 8, 12),
                        "request_id": "internal-1",
                    }
                ),
                _json_line(
                    {
                        "time": "2026-08-20T00:00:02.010000Z",
                        "message": _formula(43, 12, 16),
                        "request_id": "internal-1",
                    }
                ),
                _json_line(
                    {
                        "time": "2026-08-20T00:00:02.020000Z",
                        "message": "[ROUTING] Best: worker_42 dp_rank=0 with 8/16 blocks overlap",
                        "request_id": "internal-1",
                        "worker_id": 42,
                        "dp_rank": 0,
                        "overlap_blocks": 8,
                        "total_blocks": 16,
                    }
                ),
            ]
        ),
    )
    _write(
        root / "logs" / "worker-0.log",
        _json_line(
            {
                "time": "2026-08-20T00:00:01.000000Z",
                "message": "request received",
                "instance_id": 42,
                "request_id": "internal-1",
            }
        ),
    )
    _write(
        root / "logs" / "worker-1.log",
        _json_line(
            {
                "time": "2026-08-20T00:00:01.000000Z",
                "message": "request received",
                "instance_id": 43,
                "request_id": "internal-1",
            }
        ),
    )
    _write(root / "logs" / "tachometer.log", "2026-08-20T00:00:00Z INFO tachometer started\n")

    trace = {
        "event": {
            "schema": "dynamo.request.trace.v1",
            "event_type": "request_end",
            "event_time_unix_ms": 1_787_068_803_000,
            "request": {
                "request_id": "internal-1",
                "x_request_id": "client-1",
                "request_received_ms": 1_787_068_802_000,
                "input_tokens": 256,
                "output_tokens": 32,
                "cached_tokens": 128,
                "kv_hit_rate": 0.5,
                "ttft_ms": 44,
                "total_time_ms": 120,
                "queue_depth": 1,
                "worker": {"prefill_worker_id": 42, "prefill_dp_rank": 0},
            },
        }
    }
    trace_path = root / "artifacts" / "dynamo-request-trace.000000.jsonl.gz"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(trace_path, "wt", encoding="utf-8") as handle:
        handle.write(_json_line(trace))

    _write(
        root / "artifacts" / "aiperf" / "trace.jsonl",
        _json_line(
            {
                "metadata": {"x_request_id": "client-1", "credit_issued_ns": 1_787_068_801_900_000_000},
                "metrics": {
                    "input_sequence_length": {"value": 256},
                    "output_sequence_length": {"value": 32},
                    "time_to_first_token": {"value": 44},
                    "request_latency": {"value": 120},
                },
            }
        ),
    )
    metrics = []
    for worker_id, endpoint, running, queued, kv_used in [
        (42, "worker_agg_0_0", 3, 1, 0.4),
        (43, "worker_agg_1_0", 4, 2, 0.8),
    ]:
        metrics.extend(
            [
                ("router", f'dynamo_frontend_worker_active_prefill_tokens{{worker_id="{worker_id}"}}', 128.0),
                ("router", f'dynamo_frontend_worker_last_time_to_first_token_seconds{{worker_id="{worker_id}"}}', 0.04),
                (
                    "router",
                    f'dynamo_frontend_worker_last_inter_token_latency_seconds{{worker_id="{worker_id}"}}',
                    0.002,
                ),
                (endpoint, "sglang:num_running_reqs", running),
                (endpoint, "sglang:num_queue_reqs", queued),
                (endpoint, "dynamo_component_gpu_cache_usage_percent", kv_used),
            ]
        )
    table = pa.table(
        {
            "scraper_endpoint": [row[0] for row in metrics],
            "metric_name": [row[1] for row in metrics],
            "metric_value": [row[2] for row in metrics],
            "time_since_start": [1.0] * len(metrics),
        }
    )
    parquet_path = root / "artifacts" / "tachometer" / "local" / "final.parquet"
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, parquet_path)
    return root


def test_view_preloads_route_decision_and_worker_state(tmp_path: Path) -> None:
    run = _make_run(tmp_path / "run")
    normalize_run(run)

    view = load_view_data(run)

    assert view.summary() == {
        "requestTraces": 1,
        "aiperfRequests": 1,
        "decisions": 1,
        "workers": 2,
        "workerAliases": ["A", "B"],
        "avgKvHitRate": 0.5,
        "avgTtftMs": 44.0,
        "routerSettings": None,
    }
    timeline = view.timeline()
    assert timeline["traces"][0]["lowerPrefixSelected"] is True
    assert timeline["traces"][0]["prefillWorkerAlias"] == "A"
    decision = view.decision("internal-1")
    assert decision["selectedWorkerAlias"] == "A"
    assert [candidate["workerAlias"] for candidate in decision["candidates"]] == ["A", "B"]
    assert decision["candidates"][0]["runningReqs"] == 3.0
    assert decision["candidates"][1]["queuedReqs"] == 2.0
    assert decision["candidates"][1]["gpuCacheUsageFraction"] == 0.8


def test_view_reads_extensionless_dynamo_request_trace(tmp_path: Path) -> None:
    run = _make_run(tmp_path / "run")
    compressed_trace = run / "artifacts" / "dynamo-request-trace.000000.jsonl.gz"
    with gzip.open(compressed_trace, "rt", encoding="utf-8") as handle:
        _write(run / "artifacts" / "dynamo-request-trace", handle.read())
    compressed_trace.unlink()

    normalize_run(run)

    assert load_view_data(run).summary()["requestTraces"] == 1


def test_view_reads_json_logged_router_config_dump(tmp_path: Path) -> None:
    run = _make_run(tmp_path / "run")
    config = {
        "router_mode": "kv",
        "overlap_score_credit": 1.0,
        "overlap_score_credit_decay": 0.0,
        "prefill_load_scale": 1.0,
        "decode_active_request_weight": 0.5,
        "router_temperature": 0.0,
    }
    _write(
        run / "logs" / "router.log",
        _json_line({"time": "2026-08-20T00:00:00Z", "message": f"CONFIG_DUMP: {json.dumps({'config': config})}"}),
    )

    assert _router_settings(run) == config


def test_view_splits_prefill_and_decode_score_batches(tmp_path: Path) -> None:
    run = _make_run(tmp_path / "run")
    router_rows = [
        {"time": "2026-08-20T00:00:02.000000Z", "message": _formula(42, 8, 12), "request_id": "internal-1"},
        {"time": "2026-08-20T00:00:02.001000Z", "message": _formula(43, 12, 16), "request_id": "internal-1"},
        {
            "time": "2026-08-20T00:00:02.002000Z",
            "message": "[ROUTING] Best: worker_42 dp_rank=0 with 8/16 blocks overlap",
            "request_id": "internal-1",
            "worker_id": 42,
            "dp_rank": 0,
            "overlap_blocks": 8,
            "total_blocks": 16,
        },
        {"time": "2026-08-20T00:00:02.003000Z", "message": _formula(44, 0, 4), "request_id": "internal-1"},
        {"time": "2026-08-20T00:00:02.004000Z", "message": _formula(45, 0, 8), "request_id": "internal-1"},
        {
            "time": "2026-08-20T00:00:02.005000Z",
            "message": "[ROUTING] Best: worker_44 dp_rank=0 with 0/16 blocks overlap",
            "request_id": "internal-1",
            "worker_id": 44,
            "dp_rank": 0,
            "overlap_blocks": 0,
            "total_blocks": 16,
        },
    ]
    _write(run / "logs" / "router.log", "".join(_json_line(row) for row in router_rows))
    trace = {
        "event": {
            "schema": "dynamo.request.trace.v1",
            "event_type": "request_end",
            "event_time_unix_ms": 1_787_068_803_000,
            "request": {
                "request_id": "internal-1",
                "request_received_ms": 1_787_068_802_000,
                "input_tokens": 256,
                "cached_tokens": 128,
                "kv_hit_rate": 0.5,
                "worker": {
                    "prefill_worker_id": 42,
                    "prefill_dp_rank": 0,
                    "decode_worker_id": 44,
                    "decode_dp_rank": 0,
                },
            },
        }
    }
    with gzip.open(run / "artifacts" / "dynamo-request-trace.000000.jsonl.gz", "wt", encoding="utf-8") as handle:
        handle.write(_json_line(trace))

    normalize_run(run)
    view = load_view_data(run)

    assert [(row["stage"], row["worker_id"]) for row in view.decisions] == [("prefill", "42"), ("decode", "44")]
    assert view.summary()["workerAliases"] == ["P-A", "P-B", "D-A", "D-B"]
    timeline = view.timeline()["traces"][0]
    assert timeline["prefillWorkerAlias"] == "P-A"
    assert timeline["decodeWorkerAlias"] == "D-A"
    assert timeline["lowerPrefixSelected"] is True
    assert [candidate["workerAlias"] for candidate in view.decision("internal-1:0")["candidates"]] == ["P-A", "P-B"]
    assert [candidate["workerAlias"] for candidate in view.decision("internal-1:1")["candidates"]] == ["D-A", "D-B"]
    assert view.decision("internal-1:1")["lowerPrefixSelected"] is False
    route_rows = view.decision_rows()
    assert len(route_rows) == 1
    assert {key: route_rows[0][key] for key in route_rows[0] if key != "benchS"} == {
        "requestId": "internal-1",
        "prefillDecisionId": "internal-1:0",
        "decodeDecisionId": "internal-1:1",
        "prefillWorkerAlias": "P-A",
        "decodeWorkerAlias": "D-A",
        "overlapBlocks": 8,
        "totalBlocks": 16,
        "prefillScoreBlocks": 12.0,
        "decodeScoreBlocks": 4.0,
        "lowerPrefixSelected": True,
    }


def test_view_serves_dashboard_and_preloaded_api(tmp_path: Path) -> None:
    run = _make_run(tmp_path / "run")
    normalize_run(run)
    view = load_view_data(run)

    from http.server import ThreadingHTTPServer

    server = ThreadingHTTPServer(("127.0.0.1", 0), _handler(view))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urlopen(f"http://127.0.0.1:{server.server_port}/api/decision?id=internal-1") as response:
            assert response.headers["Content-Type"] == "application/json"
            assert json.load(response)["found"] is True
        with urlopen(f"http://127.0.0.1:{server.server_port}/") as response:
            assert b"ruter" in response.read().lower()
    finally:
        server.shutdown()
        thread.join()
        server.server_close()
