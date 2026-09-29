# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SGLang batch evidence enters the shared DSight metric schema without rank pooling."""

from __future__ import annotations

from pathlib import Path

from test_dsight import write_run

from srtctl.dsight.importer import Importer
from srtctl.dsight.log_metrics.sglang import SGLangLogMetrics
from srtctl.dsight.sources import source_identity

# Representative prefill and decode log formats; request identifiers are synthetic.
# Timestamps change when placing them inside the existing importer fixture window.
PREFILL = (
    "[2026-09-24 01:47:50.615 DP3 TP3 EP3] Prefill batch [1], #new-seq: 1, #new-token: 64, "
    "#cached-token: 0, token usage: 0.00, #running-req: 0, #queue-req: 0, #pending-token: 0, "
    "#bootstrap-req: 0, #inflight-req: 1, #optimistic-req: 0, cuda graph: False, "
    "input throughput (token/s): 7.29"
)
DECODE = (
    "[2026-09-24 01:50:43.001 DP0 TP0 EP0] Decode batch [40], #running-req: 1, #token: 532160, "
    "token usage: 0.19, accept len: 4.40, accept rate: 0.68, pre-allocated usage: 0.18, "
    "#prealloc-req: 0, #transfer-req: 22, #retracted-req: 0, cuda graph: True, "
    "gen throughput (token/s): 11.87, #queue-req: 0"
)
PREFILL_REQUEST = (
    "[2026-09-24 01:47:50.616 DP3 TP3 EP3] "
    "ReqTimeStats(rid=request-one, bootstrap_room=3, input_len=4, "
    "cached_input_len=0, output_len=1, attempts=0, type=prefill): "
    "bootstrap_duration=0.32ms, queue_duration=0.41ms, forward_duration=8348.19ms, "
    "entry_time=1790239662.267, transfer_speed=0.00 GB/s, transfer_total=0.00 MB"
)
DECODE_REQUEST = (
    "[2026-09-24 01:50:40.600 DP2 TP2 EP2] "
    "ReqTimeStats(rid=request-decode, bootstrap_room=4, "
    "input_len=16956, cached_input_len=0, output_len=19, attempts=0, type=decode): "
    "bootstrap_duration=9.05ms, alloc_wait_duration=11.50ms, transfer_duration=22297.77ms, "
    "queue_duration=0.71ms, forward_duration=1321.16ms, entry_time=1790239816.960"
)


def test_real_batch_examples_preserve_each_recorded_value_and_scope():
    generator = SGLangLogMetrics()
    prefill_source = source_identity(Path("host_prefill_w0.out"))
    decode_source = source_identity(Path("host_decode_w0.out"))
    assert prefill_source and decode_source
    prefill = generator.parse_line(PREFILL, prefill_source)
    decode = generator.parse_line(DECODE, decode_source)
    assert prefill and decode
    assert prefill.time == "2026-09-24 01:47:50.615" and prefill.time_resolution_s == 0.001
    assert prefill.rank == 3 and prefill.rank_kind == "dp"
    assert dict(prefill.labels) == {"phase": "prefill", "tp": "3", "ep": "3"}
    assert dict(prefill.values) == {
        "log_sglang_new_sequences": 1,
        "log_sglang_new_tokens": 64,
        "log_sglang_cached_tokens": 0,
        "log_sglang_token_usage": 0.0,
        "log_sglang_running_requests": 0,
        "log_sglang_queued_requests": 0,
        "log_sglang_pending_tokens": 0,
        "log_sglang_bootstrap_requests": 0,
        "log_sglang_inflight_requests": 1,
        "log_sglang_optimistic_requests": 0,
        "log_sglang_cuda_graph_enabled": 0,
        "log_sglang_input_throughput_tokens_per_second": 7.29,
    }
    assert decode.time == "2026-09-24 01:50:43.001"
    assert decode.rank == 0 and dict(decode.labels) == {"phase": "decode", "tp": "0", "ep": "0"}
    assert dict(decode.values) == {
        "log_sglang_running_requests": 1,
        "log_sglang_decode_tokens": 532160,
        "log_sglang_token_usage": 0.19,
        "log_sglang_accept_length": 4.4,
        "log_sglang_accept_rate": 0.68,
        "log_sglang_preallocated_usage": 0.18,
        "log_sglang_preallocated_requests": 0,
        "log_sglang_transfer_requests": 22,
        "log_sglang_retracted_requests": 0,
        "log_sglang_cuda_graph_enabled": 1,
        "log_sglang_generation_throughput_tokens_per_second": 11.87,
        "log_sglang_queued_requests": 0,
    }


def test_partial_and_irrelevant_lines_do_not_invent_values():
    generator = SGLangLogMetrics()
    source = source_identity(Path("host_prefill_w0.out"))
    frontend = source_identity(Path("host_frontend_0.out"))
    assert source and frontend
    partial = PREFILL.split(", #cached-token:")[0] + ", #cached-token: bad, #queue-req: 0"
    event = generator.parse_line(partial, source)
    assert event and dict(event.values) == {
        "log_sglang_new_sequences": 1,
        "log_sglang_new_tokens": 64,
        "log_sglang_queued_requests": 0,
    }
    assert generator.parse_line("[2026-09-24 01:47:50.615 DP3 TP3 EP3] ReqTimeStats(...)", source) is None
    assert generator.parse_line("[2026-09-24 01:47:50.615] Prefill batch [1], #new-token: 64", source) is None
    assert generator.parse_line(PREFILL, frontend) is None
    invalid_count = generator.parse_line(PREFILL.replace("#new-token: 64", "#new-token: bad"), source)
    assert invalid_count and "log_sglang_new_tokens" not in dict(invalid_count.values)


def test_real_request_records_keep_completion_time_and_valid_same_request_fraction():
    generator = SGLangLogMetrics()
    prefill_source = source_identity(Path("host_prefill_w0.out"))
    decode_source = source_identity(Path("host_decode_w0.out"))
    assert prefill_source and decode_source
    prefill = generator.parse_line(PREFILL_REQUEST, prefill_source)
    decode = generator.parse_line(DECODE_REQUEST, decode_source)
    assert prefill and decode
    assert prefill.time == "2026-09-24 01:47:50.616" and prefill.rank == 3
    assert dict(prefill.labels) == {"phase": "prefill", "tp": "3", "ep": "3"}
    assert dict(prefill.values) == {
        "log_sglang_request_input_tokens": 4,
        "log_sglang_request_cached_input_tokens": 0,
        "log_sglang_request_uncached_input_tokens": 4,
        "log_sglang_request_cached_input_fraction": 0,
        "log_sglang_request_bootstrap_duration_ms": 0.32,
        "log_sglang_request_queue_duration_ms": 0.41,
        "log_sglang_request_forward_duration_ms": 8348.19,
        "log_sglang_request_transfer_speed_gib_per_second": 0,
        "log_sglang_request_transfer_total_mib": 0,
    }
    assert decode.time == "2026-09-24 01:50:40.600" and decode.rank == 2
    assert dict(decode.labels) == {"phase": "decode", "tp": "2", "ep": "2"}
    assert dict(decode.values) == {
        "log_sglang_request_input_tokens": 16956,
        "log_sglang_request_cached_input_tokens": 0,
        "log_sglang_request_uncached_input_tokens": 16956,
        "log_sglang_request_cached_input_fraction": 0,
        "log_sglang_request_bootstrap_duration_ms": 9.05,
        "log_sglang_request_allocation_wait_duration_ms": 11.50,
        "log_sglang_request_transfer_duration_ms": 22297.77,
        "log_sglang_request_queue_duration_ms": 0.71,
        "log_sglang_request_forward_duration_ms": 1321.16,
    }
    assert all("rid" not in dict(event.labels) for event in (prefill, decode))


def test_request_derivations_require_valid_counts_and_known_denominator():
    generator = SGLangLogMetrics()
    source = source_identity(Path("host_prefill_w0.out"))
    assert source
    cached = generator.parse_line(
        PREFILL_REQUEST.replace("input_len=4, cached_input_len=0", "input_len=4, cached_input_len=3"), source
    )
    assert cached and dict(cached.values)["log_sglang_request_cached_input_fraction"] == 0.75
    assert dict(cached.values)["log_sglang_request_uncached_input_tokens"] == 1
    for modified in (
        PREFILL_REQUEST.replace("input_len=4, cached_input_len=0", "input_len=0, cached_input_len=0"),
        PREFILL_REQUEST.replace("input_len=4, cached_input_len=0", "input_len=4, cached_input_len=5"),
        PREFILL_REQUEST.replace("input_len=4", "input_len=unknown"),
    ):
        event = generator.parse_line(modified, source)
        assert event
        assert "log_sglang_request_cached_input_fraction" not in dict(event.values)
    assert generator.parse_line(PREFILL_REQUEST.replace("type=prefill", "type=unknown"), source) is None


def test_importer_retains_rank_phase_time_and_source_line(tmp_path):
    logs, _ = write_run(tmp_path)
    prefill = PREFILL.replace("2026-09-24 01:47:50.615", "2026-09-17 10:58:33.230")
    same_file_other_rank = prefill.replace("DP3 TP3 EP3", "DP0 TP0 EP0").replace(
        "#cached-token: 0", "#cached-token: 576"
    )
    decode = DECODE.replace("2026-09-24 01:50:43.001", "2026-09-17 10:58:33.231")
    request = PREFILL_REQUEST.replace("2026-09-24 01:47:50.616", "2026-09-17 10:58:33.232")
    path = logs / "prefill-host_prefill_w0.out"
    path.write_text("startup\n" + prefill + "\n" + same_file_other_rank + "\n" + request + "\n")
    (logs / "decode-host_decode_w0.out").write_text(decode + "\n")
    data = Importer(logs, iteration_timezone="UTC").run()
    generated = [s for s in data["metrics"] if s.get("generator") == "sglang"]
    cached = [s for s in generated if s["name"] == "log_sglang_cached_tokens"]
    assert len(cached) == 2
    assert {(s["rank"], s["labels"]["tp"], s["points"][0][1]) for s in cached} == {(3, "3", 0), (0, "0", 576)}
    for series in cached:
        assert series["rank_kind"] == "dp" and series["labels"]["phase"] == "prefill"
        point = series["points"][0]
        assert point[0] == 2.23 and point[3] in {2, 3}
        assert Path(data["sources"][point[2]]["path"]).read_text().splitlines()[point[3] - 1] in {
            prefill,
            same_file_other_rank,
        }
    assert {s["labels"]["phase"] for s in generated if s["name"] == "log_sglang_running_requests"} == {
        "prefill",
        "decode",
    }
    assert not any("hit_rate" in s["name"] for s in generated)
    queue = next(s for s in generated if s["name"] == "log_sglang_request_queue_duration_ms")
    point = queue["points"][0]
    assert point[:2] == [2.232, 0.41] and point[3] == 4
    assert Path(data["sources"][point[2]]["path"]).read_text().splitlines()[point[3] - 1] == request
    assert "rid" not in queue["labels"]


def test_local_timestamp_requires_explicit_timezone(tmp_path):
    logs, _ = write_run(tmp_path)
    (logs / "prefill-host_prefill_w0.out").write_text(
        PREFILL.replace("2026-09-24 01:47:50.615", "2026-09-17 10:58:33.230") + "\n"
    )
    data = Importer(logs).run()
    assert not any(s.get("generator") == "sglang" for s in data["metrics"])
    assert data["audit"]["unaligned_log_metric_records"] == 1
    assert any("Log metrics with local timestamps omitted" in warning for warning in data["meta"]["warnings"])


def test_distinct_requests_at_one_timestamp_keep_all_values_and_line_evidence(tmp_path):
    logs, _ = write_run(tmp_path)
    base = PREFILL_REQUEST.replace("2026-09-24 01:47:50.616", "2026-09-17 10:58:33.232")
    lines = [
        base,
        base.replace("rid=request-one", "rid=request-two"),
        base.replace("rid=request-one", "rid=request-three").replace("queue_duration=0.41ms", "queue_duration=1.23ms"),
    ]
    (logs / "prefill-host_prefill_w0.out").write_text("\n".join(lines) + "\n")
    data = Importer(logs, iteration_timezone="UTC").run()
    queue = next(s for s in data["metrics"] if s["name"] == "log_sglang_request_queue_duration_ms")
    assert queue["temporal"] == "event"
    assert [point[1] for point in queue["points"]] == [0.41, 0.41, 1.23]
    assert [point[3] for point in queue["points"]] == [1, 2, 3]
    assert len({point[0] for point in queue["points"]}) == 1
    assert queue["conflict_timestamps"] == [] and queue["conflicting_samples"] == 0
    for point in queue["points"]:
        assert Path(data["sources"][point[2]]["path"]).read_text().splitlines()[point[3] - 1] == lines[point[3] - 1]


def test_synthetic_example_imports_all_families_without_client_or_telemetry():
    logs = Path(__file__).parents[1] / "examples" / "dsight" / "sglang"
    data = Importer(logs, iteration_timezone="UTC").run()
    assert not data["requests"]
    assert {s["name"] for s in data["metrics"]} == {d.name for d in SGLangLogMetrics.definitions}
    queue = next(
        s
        for s in data["metrics"]
        if s["name"] == "log_sglang_request_queue_duration_ms" and s["labels"]["phase"] == "prefill"
    )
    assert [p[1] for p in queue["points"]] == [10, 10, 40]
    worker = next(w for w in data["workers"] if w["id"] == "decode-0")
    assert worker["hosts"] == ["node-b", "node-c"] and worker["host"] is None
