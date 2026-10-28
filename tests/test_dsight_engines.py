# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Engine and partial-capture contracts, exercised through real source formats."""

from __future__ import annotations

import json
import shutil
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from test_dsight import CLIENT, ORIGIN, SERVER, trtllm_iteration_line, write_run

from srtctl.dsight.build import build_dashboard
from srtctl.dsight.engines import _trtllm_iteration_value, parse_engine_log, parse_trtllm_iteration
from srtctl.dsight.importer import Importer
from srtctl.dsight.query import TraceDataset


def tokenspeed_run(root: Path) -> tuple[Path, Path]:
    logs, sqlites = write_run(root)
    (logs / "front_frontend_0.out").write_text(
        json.dumps(
            {
                "time": "2026-09-17T10:58:31Z",
                "x_request_id": CLIENT,
                "dynamo.request.id": SERVER,
            }
        )
        + "\n"
    )
    for role in ("prefill", "decode"):
        binding = {
            "target": "dynamo.request_lifecycle",
            "dynamo.request.id": SERVER,
            "dynamo.instance.id": f"{role}-host",
            "dynamo.operation.role": role,
            "dynamo.process.epoch": f"epoch-{role}",
        }
        snapshot = (
            "[2026-09-17 10:58:33,230  ATTN TP RANK 0] - INFO - Decode batch. "
            "#running-req: 4, #pages(active/cached/total): 8/16/4096, page ratio: 0.01, "
            "gen throughput (token/s): 32.1, avg_accept_len: 1.00, accept_rate: 0.00, #queue-req: 12 (batch_log.py:151)"
        )
        (logs / f"{role}-host_{role}_w0.out").write_text(
            json.dumps(binding) + "\n" + json.dumps(binding) + "\n" + snapshot + "\n"
        )
    otel = next(logs.glob("otel/*/traces.jsonl"))
    doc = json.loads(otel.read_text())
    times = {
        "dselect": (0.14, 0.15),
        "droute": (0.151, 7.9),
        "dadmit": (0.16, 0.17),
        "dop": (0.171, 7.9),
        "ddispatch": (0.18, 0.19),
        "dpump": (0.2, 7.89),
    }
    for span in doc["resourceSpans"][0]["scopeSpans"][0]["spans"]:
        if span["spanId"] in times:
            a, b = times[span["spanId"]]
            span["startTimeUnixNano"] = str(ORIGIN + round(a * 1e9))
            span["endTimeUnixNano"] = str(ORIGIN + round(b * 1e9))
    # Both flat and per-collector OTLP layouts are supported.
    otel.rename(logs / "otel/traces.jsonl")
    (logs / "otel/traces.jsonl").write_text(json.dumps(doc) + "\n")
    profile = next(sqlites.iterdir())
    renamed = profile.with_name("decode-host_decode_w0_profile_gpu0-1_window001.sqlite")
    profile.rename(renamed)
    with sqlite3.connect(renamed) as conn:
        conn.execute("DELETE FROM NVTX_EVENTS")
        conn.executemany(
            "INSERT INTO NVTX_EVENTS VALUES (?,?,?,NULL,?)",
            [
                (200_000_000, 260_000_000, "forward_step ext=0 dec=4", (101 << 24) | 111),
                (210_000_000, 250_000_000, "graph_replay", (101 << 24) | 111),
                (212_000_000, 252_000_000, "graph_replay", (102 << 24) | 112),
                (200_000_000, 300_000_000, "operator_detail", (101 << 24) | 111),
            ],
        )
    metrics = logs / "tachometer/local"
    for path in metrics.iterdir():
        path.unlink()
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "timestamp_ns": ORIGIN + 1_000_000_000,
                    "metric_name": name,
                    "metric_value": value,
                    "scraper_endpoint": role,
                    "hostname": f"{role}-host",
                    "worker_role": role,
                    "worker_index": "0",
                }
                for name, value in (
                    ("tokenspeed:num_requests_running", 4.0),
                    ("tokenspeed:num_requests_waiting", 12.0),
                    ("tokenspeed:kv_cache_usage_perc", 0.75),
                )
                for role in ("prefill", "decode")
            ]
        ),
        metrics / "final.parquet",
    )
    return logs, sqlites


def test_tokenspeed_sources_preserve_parallel_activities_and_recorded_identities(tmp_path):
    data = Importer(*tokenspeed_run(tmp_path), iteration_timezone="UTC").run()
    request = data["requests"][0]
    assert request["server_ids"] == [SERVER]
    assert request["engine"] == []  # No engine-local request ID was recorded.
    assert request["workers"] == ["decode-0", "prefill-0"]
    assert len(request["worker_bindings"]) == 2  # Repeated JSON observations deduplicate.
    assert all(b["process"].startswith("epoch-") for b in request["worker_bindings"])
    assert all(b["evidence"][1] == 1 for b in request["worker_bindings"])
    model = request["lifecycle"]
    assert model["available"] and not model["issues"]
    milestones = {m["label"]: m["time"] for m in model["milestones"]}
    assert milestones["Decode route selected"] == pytest.approx(0.15)
    assert milestones["Prefill response handling complete"] == pytest.approx(2)
    assert [m["time"] for m in model["milestones"]] == sorted(m["time"] for m in model["milestones"])
    assert sum(s["end"] - s["start"] for s in model["stages"]) == pytest.approx(8)
    activity = {a["id"]: a for a in model["activities"]}
    assert activity["pop"]["end"] > activity["dop"]["start"]
    assert activity["dop"]["worker"] == "decode-0"
    assert activity["pop"]["worker"] == "prefill-0"
    assert data["capabilities"]["request_breakdown"]

    profile = data["profiles"][0]
    assert profile["worker"] == "decode-0" and profile["rank"] is None
    assert profile["gpus"] == "0-1" and profile["backends"] == ["tokenspeed"]
    assert len(profile["events"]) == 3
    assert {t["pid"] for t in profile["threads"]} == {101, 102}
    assert {t["tid"] for t in profile["threads"]} == {111, 112}
    assert profile["imported_range"] == [0.2, 0.26]
    assert data["capabilities"]["nvtx"] and not data["capabilities"]["cpu_samples"]

    snapshots = data["iterations"]
    assert len(snapshots) == 2
    assert all(r["kind"] == "batch_snapshot" and r["iteration"] is None for r in snapshots)
    assert all(r["global_rank"] is None and r["rank_kind"] == "attention_tp" for r in snapshots)
    assert all(r["host_step_ms"] is None and r["previous_device_step_ms"] is None for r in snapshots)
    assert snapshots[0]["start"] == pytest.approx(2.23)
    assert snapshots[0]["end"] - snapshots[0]["start"] == pytest.approx(0.001)
    assert snapshots[0]["batch_requests"] == 4 and snapshots[0]["queued_requests"] == 12
    assert snapshots[0]["total_pages"] == 4096
    kv = next(m for m in data["metrics"] if m["name"] == "tokenspeed:kv_cache_usage_perc")
    assert kv["unit"] == "ratio" and kv["points"][0][1] == 0.75
    assert all(m["component"] == "Workers" for m in data["metrics"])


@pytest.mark.parametrize("wrapped", [False, True])
def test_frontend_json_identity_does_not_require_rewriting_the_log(tmp_path, wrapped):
    logs, _ = write_run(tmp_path)
    fields = {"x_request_id": CLIENT, "dynamo.request.id": SERVER}
    obj = {"fields": fields} if wrapped else fields
    path = logs / "front_frontend_0.out"
    path.write_text("\n" + json.dumps(obj) + "\n")
    data = Importer(logs).run()
    request = data["requests"][0]
    assert request["server_ids"] == [SERVER]
    source, line = request["bridge_evidence"][0]
    assert data["sources"][source]["path"] == str(path.resolve()) and line == 2


@pytest.mark.parametrize("source", ["client", "metrics", "nsight", "otel", "batch"])
def test_sources_work_independently_without_synthetic_client_records(tmp_path, source):
    original, sqlites = write_run(tmp_path / "original")
    logs = tmp_path / "subset"
    logs.mkdir()
    options = {"iteration_timezone": "UTC"}
    if source == "client":
        shutil.copy(next(original.rglob("profile_export.jsonl")), logs / "profile_export.jsonl")
    elif source == "metrics":
        shutil.copytree(original / "tachometer/local", logs / "tachometer/local")
    elif source == "nsight":
        options["sqlites"] = sqlites
    elif source == "otel":
        shutil.copytree(original / "otel", logs / "otel")
    else:
        for path in original.glob("*_w*.out"):
            shutil.copy(path, logs / path.name)
    summary = build_dashboard(logs, tmp_path / "report", **options)
    dataset = TraceDataset.from_path(summary["data"])
    data, cap = dataset.data, summary["capabilities"]
    assert cap["requests"] is (source == "client")
    assert cap["nsight"] is (source == "nsight")
    assert cap["metrics"] is (source == "metrics")
    assert cap["server_activity"] is (source == "otel")
    assert cap["iterations"] is (source == "batch")
    assert cap["request_breakdown"] is False
    if source != "client":
        assert data["requests"] == [] and data["sessions"] == []
        assert "no client" in data["meta"]["time_basis"]
    if source == "otel":
        assert dataset.query("server_spans")["total"] >= 12
        assert not cap["request_breakdown"]
    if source == "metrics":
        assert data["meta"]["origin_ns"] == str(ORIGIN + 1_000_000_000)
        assert data["meta"]["duration"] == 7
        assert data["workers"][0]["id"] == "decode-0"


def test_no_recorded_time_is_an_explicit_error(tmp_path):
    with pytest.raises(ValueError, match="No positive recorded time range"):
        Importer(tmp_path).run()


def test_unknown_nvtx_does_not_enable_an_empty_nsight_section(tmp_path):
    logs, sqlites = write_run(tmp_path)
    with sqlite3.connect(next(sqlites.iterdir())) as conn:
        conn.execute("UPDATE NVTX_EVENTS SET text='unrecognized_annotation'")
    data = Importer(logs, sqlites).run()
    assert len(data["profiles"]) == 1
    assert data["profiles"][0]["timed_ranges_scanned"] == 2
    assert not data["capabilities"]["nsight"]
    assert not data["capabilities"]["cpu_samples"]


def test_disabled_otel_is_not_read_when_client_is_missing(tmp_path):
    logs, _ = write_run(tmp_path)
    next(logs.rglob("profile_export.jsonl")).unlink()
    next(logs.glob("otel/*/traces.jsonl")).write_text("malformed but disabled")
    data = Importer(logs, otel=False).run()
    assert data["capabilities"]["metrics"]
    assert not data["capabilities"]["server_activity"]
    assert not any(s["kind"] == "otel" for s in data["sources"])


def test_uncertain_worker_binding_stays_unassigned(tmp_path):
    logs, sqlites = tokenspeed_run(tmp_path)
    original = logs / "decode-host_decode_w0.out"
    (logs / "decode-host_decode_w1.out").write_text(original.read_text())
    data = Importer(logs, sqlites).run()
    decode = [s for s in data["requests"][0]["spans"] if s["role"] == "decode"]
    assert decode and all(s["worker"] is None for s in decode)
    assert data["audit"]["ambiguous_span_workers"] > 0
    request = data["requests"][0]
    assert request["workers"] == ["prefill-0"]
    assert all(b["ambiguous"] for b in request["worker_bindings"] if b["role"] == "decode")
    assert data["audit"]["ambiguous_worker_bindings"] == 2


@pytest.mark.parametrize("content", ["", "  \n\n"])
def test_empty_client_export_uses_other_recorded_sources(tmp_path, content):
    logs, sqlites = tokenspeed_run(tmp_path)
    next(logs.rglob("profile_export.jsonl")).write_text(content)
    data = Importer(logs, sqlites, iteration_timezone="UTC").run()
    assert data["requests"] == [] and not data["capabilities"]["requests"]
    assert data["capabilities"]["metrics"] and data["capabilities"]["nsight"]
    assert data["capabilities"]["server_activity"]


def test_filtered_client_export_never_falls_back_to_another_phase(tmp_path):
    logs, sqlites = tokenspeed_run(tmp_path)
    path = next(logs.rglob("profile_export.jsonl"))
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for row in rows:
        row["metadata"]["benchmark_phase"] = "warmup"
    path.write_text("\n".join(json.dumps(row) for row in rows))
    with pytest.raises(ValueError, match="warmup is never substituted"):
        Importer(logs, sqlites).run()


def test_no_client_window_unions_metrics_and_later_batch_observations(tmp_path):
    logs, _ = tokenspeed_run(tmp_path)
    next(logs.rglob("profile_export.jsonl")).unlink()
    shutil.rmtree(logs / "otel")
    for path in logs.glob("*_w*.out"):
        path.write_text(path.read_text().replace("10:58:33,230", "10:58:43,230"))
    data = Importer(logs, iteration_timezone="UTC").run()
    assert len(data["iterations"]) == 2
    assert data["meta"]["origin_ns"] == str(ORIGIN + 1_000_000_000)
    assert data["meta"]["duration"] == pytest.approx(11.231)
    assert all(r["start"] == pytest.approx(11.23) for r in data["iterations"])


def test_trt_global_rank_and_local_rank_remain_distinct(tmp_path):
    logs, _ = write_run(tmp_path)
    for path in logs.glob("*_w*.out"):
        path.write_text(path.read_text().replace("global_rank = 0", "global_rank = 4"))
    data = Importer(logs, iteration_timezone="UTC").run()
    dataset = TraceDataset(data)
    assert dataset.query("iterations", rank=4)["total"] == 2
    assert dataset.query("iterations", rank=0)["total"] == 0
    assert all(r["rank"] == r["global_rank"] == 4 and r["local_rank"] == 0 for r in data["iterations"])


def test_missing_first_token_preserves_activity_without_inventing_ttft(tmp_path):
    logs, _ = tokenspeed_run(tmp_path)
    path = next(logs.rglob("profile_export.jsonl"))
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for row in rows:
        row["metrics"].pop("time_to_first_token", None)
    path.write_text("\n".join(json.dumps(row) for row in rows))
    data = Importer(logs).run()
    model = data["requests"][0]["lifecycle"]
    assert model["activities"] and len(model["stages"]) == 1
    assert model["stages"][0]["label"] == "Client complete"
    assert any("first-token timing is unavailable" in issue for issue in model["issues"])
    assert not any("clocks" in issue for issue in model["issues"])


def add_empty_cpu_profile(sqlites: Path) -> None:
    path = sqlites / "front_frontend_0.sqlite"
    shutil.copy(next(sqlites.iterdir()), path)
    with sqlite3.connect(path) as conn:
        conn.execute("UPDATE NVTX_EVENTS SET text='route.generate'")
        conn.executescript("""
            CREATE TABLE COMPOSITE_EVENTS (id INTEGER, start INTEGER, globalTid INTEGER);
            CREATE TABLE SAMPLING_CALLCHAINS (id INTEGER, stackDepth INTEGER, symbol INTEGER, unresolved INTEGER);
        """)


def test_empty_cpu_sampling_tables_do_not_enable_hotspots(tmp_path):
    logs, sqlites = tokenspeed_run(tmp_path)
    add_empty_cpu_profile(sqlites)
    data = Importer(logs, sqlites).run()
    profile = next(p for p in data["profiles"] if p["worker"] == "frontend")
    assert profile["events"] and profile["cpu"]["samples"] == []
    assert data["capabilities"]["nsight"] and not data["capabilities"]["cpu_samples"]


def test_legacy_dataset_rank_filter_still_uses_global_rank(tmp_path):
    data = Importer(*write_run(tmp_path), iteration_timezone="UTC").run()
    for row in data["iterations"]:
        row.pop("rank_kind")
        row["global_rank"], row["rank"] = 4, 0
    trace = TraceDataset(data)
    assert trace.query("iterations", rank=4)["total"] == 2
    assert trace.query("iterations", rank=0)["total"] == 0


@pytest.mark.parametrize("fraction", ["23", "230123"])
def test_snapshot_precision_agrees_with_source_window(tmp_path, fraction):
    logs, _ = tokenspeed_run(tmp_path)
    next(logs.rglob("profile_export.jsonl")).unlink()
    shutil.rmtree(logs / "otel")
    shutil.rmtree(logs / "tachometer")
    for path in logs.glob("*_w*.out"):
        path.write_text(path.read_text().replace("10:58:33,230", f"10:58:33,{fraction}"))
    data = Importer(logs, iteration_timezone="UTC").run()
    resolution = 10 ** -len(fraction)
    assert data["meta"]["duration"] == pytest.approx(resolution)
    assert all(r["start"] == 0 and r["end"] == pytest.approx(resolution) for r in data["iterations"])
    assert data["meta"]["origin_ns"] == str(ORIGIN + 2_000_000_000 + int(fraction.ljust(9, "0")))


def test_trtllm_iteration_keeps_every_recorded_field_in_order():
    it = parse_trtllm_iteration(trtllm_iteration_line(12588, 3, "2026-09-17 10:58:33"))
    assert it is not None
    assert (it.iteration, it.global_rank, it.rank, it.local_time) == (12588, 3, 0, "2026-09-17 10:58:33")
    assert (it.batch_requests, it.kv_cache_util, it.host_step_ms, it.previous_device_step_ms) == (18, 0.25, 0.001, 25.0)
    fields = dict(it.fields)
    assert list(fields)[:4] == ["iter", "global_rank", "rank", "num_scheduled_requests"]
    assert fields["currank_total_requests"] == (32, 64)
    assert fields["states.cached_kv_tokens"] == 512
    assert fields["num_unscheduled"] == 2
    assert "states" not in fields  # The dict flattens; its parent key is not repeated.
    record = parse_engine_log(trtllm_iteration_line(1, 0, "2026-09-17 10:58:33"))
    assert record is not None and record.backend == "trtllm" and record.iteration is not None


def test_trtllm_iteration_passes_unknown_fields_through_by_value_shape():
    line = trtllm_iteration_line(7, 0, "2026-09-17 10:58:33").replace(
        "num_active = 20",
        "future_ratio = 0.5, future_ms = 3ms, future_pair = 1/2, future_tag = warm, "
        "future_list = [1, 2], num_active = 20",
    )
    line = line.replace("'cached_kv_tokens': 512}", "'cached_kv_tokens': 512, 'cached_kv_tokens_per_req': [7, 0, 9]}")
    fields = dict(parse_trtllm_iteration(line).fields)
    assert fields["future_ratio"] == 0.5
    assert fields["future_ms"] == 3.0
    assert fields["future_pair"] == (1, 2)
    assert fields["future_tag"] == "warm"
    assert fields["future_list"] == (1, 2)
    assert fields["states.cached_kv_tokens_per_req"] == (7, 0, 9)
    assert fields["num_active"] == 20


def test_trtllm_iteration_tolerates_removed_and_unavailable_fields():
    line = (
        "iter = 1, global_rank = 0, rank = 0, kv_cache_util = N/A, prev_device_step_time = N/A, "
        "timestamp = 2026-09-17 10:58:33"
    )
    it = parse_trtllm_iteration(line)
    assert it is not None
    assert it.kv_cache_util is it.previous_device_step_ms is it.batch_requests is it.host_step_ms is None
    assert dict(it.fields)["prev_device_step_time"] is None


@pytest.mark.parametrize(
    "line",
    [
        "Engine ID map only",
        "for iter = 3 the loop retried",  # No ranks or timestamp: not the iteration log.
        "iter = 1, global_rank = 0, rank = 0, timestamp = soon",
        "iter = x, global_rank = 0, rank = 0, timestamp = 2026-09-17 10:58:33",
        "iter = 1, global_rank = 0, timestamp = 2026-09-17 10:58:33",
        "iter = 1, rank = 0, timestamp = 2026-09-17 10:58:33",
        "iter = 1, global_rank = 0, rank = 0",
        "iter = 1, global_rank = 0, rank = N/A, timestamp = 2026-09-17 10:58:33",
        "iter = 1.5, global_rank = 0, rank = 0, timestamp = 2026-09-17 10:58:33",
        "iter = 1, global_rank = 0, rank = 0, timestamp = 2026-09-17 10:58:33 late",
        "iter = 1, global_rank = 0, rank = 0, host_step_time = nanms, timestamp = 2026-09-17 10:58:33",
    ],
)
def test_trtllm_iteration_requires_identity_and_time(line):
    it = parse_trtllm_iteration(line)
    if "nanms" in line:
        assert it is not None and dict(it.fields)["host_step_time"] == "nanms"  # Never a non-finite number.
    else:
        assert it is None


def trtllm_emitted_line(
    executor: SimpleNamespace,
    *,
    kv_util_str: str,
    host_step_time: float,
    prev_device_step_time_str: str,
    formatted_timestamp: str,
    num_fetched: int,
) -> str:
    """The iteration log exactly as TensorRT-LLM emits it, so the format contract is checked against the source.

    The message is the f-string from tensorrt_llm/_torch/pyexecutor/profiling.py:764-781
    (TensorRT-LLM 7695494b) copied verbatim; the prefix is Logger.log in tensorrt_llm/logger.py
    (`[TRT-LLM] [I] [<module>][RANK <n>]`) behind the stdlib handler's timestamp.
    """
    message = (
        f"iter = {executor.iter_counter}, "
        f"global_rank = {executor.global_rank}, "
        f"rank = {executor.dist.rank}, "
        f"num_scheduled_requests = {executor.num_scheduled_requests}, "
        f"adp_dummy_ctx_tokens = {executor._iter_adp_dummy_ctx_tokens}, "
        f"adp_dummy_gen_tokens = {executor._iter_adp_dummy_gen_tokens}, "
        f"kv_cache_util = {kv_util_str}, "
        f"currank_total_requests = {executor.num_fetch_requests_cur_rank}/"
        f"{executor.num_fetch_requests}, "
        f"host_step_time = {host_step_time}ms, "
        f"prev_device_step_time = {prev_device_step_time_str}, "
        f"timestamp = {formatted_timestamp}, "
        f"states = {executor.model_engine.iter_states}, "
        f"num_active = {len(executor.active_requests)}, "
        f"num_fetched = {num_fetched}, "
        f"num_dummy = {executor.num_dummy_requests}, "
        f"num_unscheduled = {executor.num_unscheduled_requests}"
    )
    return f"[10/19/2026-16:53:36] [TRT-LLM] [I] [_torch][RANK {executor.global_rank}] {message}"


def trtllm_executor(iter_states: dict, *, iteration: int = 1, global_rank: int = 0) -> SimpleNamespace:
    return SimpleNamespace(
        iter_counter=iteration,
        global_rank=global_rank,
        dist=SimpleNamespace(rank=global_rank % 4),
        num_scheduled_requests=3,
        _iter_adp_dummy_ctx_tokens=0,
        _iter_adp_dummy_gen_tokens=1,
        num_fetch_requests_cur_rank=0,
        num_fetch_requests=3,
        model_engine=SimpleNamespace(iter_states=iter_states),
        active_requests=[object()] * 3,
        num_dummy_requests=0,
        num_unscheduled_requests=0,
    )


# Every shape each emitted value can take: the first iteration has no previous device
# time, a missing KV manager prints N/A, iter_states starts empty (model_engine.py:581),
# float reprs may use exponents, and log_cached_kv_tokens_per_req adds a list
# (decoder/inputs.py:496-499).
@pytest.mark.parametrize(
    ("states", "kv_util_str", "host_step_time", "prev_device", "expected_tail"),
    [
        ({}, "N/A", 77.96812057495117, None, []),
        (
            {"num_ctx_requests": 3, "num_ctx_tokens": 8192, "num_generation_tokens": 0, "cached_kv_tokens": 0},
            f"{1.0 - 3 / 64:.3f}",
            5e-05,
            12.345678901234567,
            [
                ("states.num_ctx_requests", 3),
                ("states.num_ctx_tokens", 8192),
                ("states.num_generation_tokens", 0),
                ("states.cached_kv_tokens", 0),
            ],
        ),
        (
            {
                "num_ctx_requests": 1,
                "num_ctx_tokens": 17,
                "num_generation_tokens": 2,
                "cached_kv_tokens": 4224,
                "cached_kv_tokens_per_req": [0, 4096, 128],
                "cached_kv_tokens_cuda_graph_padding": 0,
            },
            "0.000",
            1.0,
            0.25,
            [
                ("states.num_ctx_requests", 1),
                ("states.num_ctx_tokens", 17),
                ("states.num_generation_tokens", 2),
                ("states.cached_kv_tokens", 4224),
                ("states.cached_kv_tokens_per_req", (0, 4096, 128)),
                ("states.cached_kv_tokens_cuda_graph_padding", 0),
            ],
        ),
    ],
)
def test_trtllm_iteration_decodes_the_emitted_format_in_order(
    states, kv_util_str, host_step_time, prev_device, expected_tail
):
    prev_str = "N/A" if prev_device is None else f"{prev_device}ms"
    line = trtllm_emitted_line(
        trtllm_executor(states, iteration=12588, global_rank=6),
        kv_util_str=kv_util_str,
        host_step_time=host_step_time,
        prev_device_step_time_str=prev_str,
        formatted_timestamp="2026-10-19 16:53:36",
        num_fetched=2,
    )
    record = parse_engine_log(line)
    assert record is not None and record.backend == "trtllm" and record.identity is None
    it = record.iteration
    kv_util = None if kv_util_str == "N/A" else float(kv_util_str)
    # Float reprs round-trip exactly, so the parsed durations equal the emitted floats.
    assert (it.iteration, it.global_rank, it.rank, it.local_time) == (12588, 6, 2, "2026-10-19 16:53:36")
    assert (it.batch_requests, it.kv_cache_util, it.host_step_ms, it.previous_device_step_ms) == (
        3,
        kv_util,
        host_step_time,
        prev_device,
    )
    assert list(it.fields) == [
        ("iter", 12588),
        ("global_rank", 6),
        ("rank", 2),
        ("num_scheduled_requests", 3),
        ("adp_dummy_ctx_tokens", 0),
        ("adp_dummy_gen_tokens", 1),
        ("kv_cache_util", kv_util),
        ("currank_total_requests", (0, 3)),
        ("host_step_time", host_step_time),
        ("prev_device_step_time", prev_device),
        ("timestamp", "2026-10-19 16:53:36"),
        *expected_tail,
        ("num_active", 3),
        ("num_fetched", 2),
        ("num_dummy", 0),
        ("num_unscheduled", 0),
    ]
    assert all(type(v) is float for k, v in it.fields if k.endswith("_time") and v is not None)


def test_trtllm_iteration_parses_the_recorded_cluster_line():
    line = (
        "[10/19/2026-16:53:36] [TRT-LLM] [I] [_torch] [RANK 0] iter = 1, global_rank = 0, rank = 0, "
        "num_scheduled_requests = 3, adp_dummy_ctx_tokens = 0, adp_dummy_gen_tokens = 0, kv_cache_util = 0.952, "
        "currank_total_requests = 0/3, host_step_time = 77.96812057495117ms, prev_device_step_time = N/A, "
        "timestamp = 2026-10-19 16:53:36, states = {'num_ctx_requests': 3, 'num_ctx_tokens': 8192, "
        "'num_generation_tokens': 0, 'cached_kv_tokens': 0}, num_active = 3, num_fetched = 0, num_dummy = 0, "
        "num_unscheduled = 0\n"
    )
    it = parse_trtllm_iteration(line)
    assert it is not None
    assert (it.batch_requests, it.kv_cache_util, it.host_step_ms, it.previous_device_step_ms) == (
        3,
        0.952,
        77.96812057495117,
        None,
    )
    assert len(it.fields) == 19 and dict(it.fields)["states.num_ctx_tokens"] == 8192
    assert dict(it.fields)["num_unscheduled"] == 0  # The trailing newline is not part of the value.


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("N/A", None),
        ("0", 0),
        ("-3", -3),
        ("0.952", 0.952),
        ("77.96812057495117ms", 77.96812057495117),
        ("1e-05ms", 1e-05),
        ("2.5E+1ms", 25.0),
        # Non-finite numbers stay text: JSON storage rejects NaN and Infinity.
        ("nan", "nan"),
        ("nanms", "nanms"),
        ("inf", "inf"),
        ("-infms", "-infms"),
        ("0/3", (0, 3)),
        ("1/2/3", "1/2/3"),
        ("-1/2", "-1/2"),
        ("[1, 2]", (1, 2)),
        ("[]", ()),
        ("{'a': 1, 'b': [2]}", {"a": 1, "b": [2]}),
        ("{}", {}),
        ("{1, 2}", "{1, 2}"),  # A set is not a recorded shape.
        ("{'a': 1", "{'a': 1"),  # Malformed literals stay text instead of raising.
        ("[1, 2", "[1, 2"),
        ("[tensor(1)]", "[tensor(1)]"),
        ("(1, 2)", "(1, 2)"),
        ("warm", "warm"),
        ("warm ms", "warm ms"),
        ("", ""),
    ],
)
def test_trtllm_iteration_value_converts_by_shape(text, expected):
    value = _trtllm_iteration_value(text)
    assert value == expected and type(value) is type(expected)


def test_trtllm_iteration_boundaries_ignore_commas_inside_values():
    line = (
        "iter = 1, global_rank = 0, rank = 0, timestamp = 2026-09-17 10:58:33, "
        "states = {'note': 'a, b', 'tags': ['x, y', 'z'], 'per_req': [1, 2, 3]}, num_active = 3"
    )
    assert list(parse_trtllm_iteration(line).fields)[4:] == [
        ("states.note", "a, b"),
        ("states.tags", ("x, y", "z")),
        ("states.per_req", (1, 2, 3)),
        ("num_active", 3),
    ]


def test_trtllm_iteration_empty_states_records_no_keys_and_keeps_later_fields():
    line = "iter = 0, global_rank = 0, rank = 0, timestamp = 2026-09-17 10:58:33, states = {}, num_active = 0"
    fields = dict(parse_trtllm_iteration(line).fields)
    assert not any(k == "states" or k.startswith("states.") for k in fields)
    assert fields["num_active"] == 0


def test_trtllm_iteration_malformed_states_stays_text():
    line = "iter = 1, global_rank = 0, rank = 0, timestamp = 2026-09-17 10:58:33, states = {'a': tensor(1)}"
    assert dict(parse_trtllm_iteration(line).fields)["states"] == "{'a': tensor(1)}"


@pytest.mark.xfail(
    strict=True,
    reason="The last field's value runs to end of line, so text written after the iteration log "
    "(here an Engine ID map interleaved on the same line) turns num_unscheduled into a string.",
)
def test_trtllm_iteration_followed_by_identity_keeps_last_field_numeric():
    """Text after the last field is not a value; the field keeps its shape or the line is rejected."""
    line = (
        trtllm_iteration_line(42, 4, "2026-09-17 10:58:34")
        + f" Engine ID map: request_id={SERVER} trtllm_client_id=8 disagg_request_id=101"
    )
    record = parse_engine_log(line)
    assert record is not None and record.identity is not None and record.iteration is not None
    assert dict(record.iteration.fields)["num_unscheduled"] == 2


@pytest.mark.xfail(
    strict=True,
    reason="'<n>ms' is documented as a float duration, but an integral value such as '3ms' becomes int. "
    "TensorRT-LLM always prints a float repr today, so this is unreachable from profiling.py.",
)
def test_trtllm_iteration_integral_milliseconds_are_floats():
    assert type(_trtllm_iteration_value("3ms")) is float


# The format contract: content may change freely (keys added, removed, reordered, any value
# shape), but the grammar `key = value` joined by ", " may not. A line whose grammar drifts is
# rejected rather than decoded into values that silently absorb their neighbours.
REQUIRED = [("iter", "1", 1), ("global_rank", "0", 0), ("rank", "0", 0), ("timestamp", "2026-09-17 10:58:33", None)]


def iteration_fields(pairs: list[tuple[str, str, object]]) -> str:
    return "[TRT-LLM] [I] [_torch][RANK 0] " + ", ".join(f"{key} = {text}" for key, text, _ in pairs)


@pytest.mark.parametrize(
    "extra",
    [
        pytest.param([], id="only-required-keys"),
        pytest.param([("num_active", "3", 3), ("num_unscheduled", "0", 0)], id="keys-kept"),
        pytest.param([("num_unscheduled", "0", 0), ("num_active", "3", 3)], id="keys-reordered"),
        pytest.param(
            [("new_ratio", "0.5", 0.5), ("new_ms", "1.5ms", 1.5), ("new_pair", "4/8", (4, 8))], id="new-numbers"
        ),
        pytest.param(
            [("new_na", "N/A", None), ("new_tag", "warm", "warm"), ("new_list", "[3, 1]", (3, 1))], id="new-shapes"
        ),
        pytest.param(
            [
                ("states", "{'cached_kv_tokens': 0, 'new_state': [1, 2], 'new_flag': True}", None),
                ("num_active", "3", 3),
            ],
            id="new-nested-keys",
        ),
    ],
)
@pytest.mark.parametrize("required_last", [False, True], ids=["required-first", "required-after-iter"])
def test_trtllm_iteration_tolerates_content_changes(extra, required_last):
    """Adding, removing or reordering fields after `iter` never needs a parser change."""
    pairs = [*REQUIRED, *extra] if not required_last else [REQUIRED[0], *extra, *REQUIRED[1:]]
    it = parse_trtllm_iteration(iteration_fields(pairs))
    assert it is not None
    fields = dict(it.fields)
    expected_states = {"states.cached_kv_tokens": 0, "states.new_state": (1, 2), "states.new_flag": True}
    for key, text, value in pairs:
        if key == "states":
            assert {k: v for k, v in fields.items() if k.startswith("states.")} == expected_states
        else:
            assert fields[key] == (text if key == "timestamp" else value)
    assert [k for k in fields if not k.startswith("states.")] == [k for k, _, _ in pairs if k != "states"]


GOOD = "iter = 1, global_rank = 0, rank = 0, timestamp = 2026-09-17 10:58:33, num_active = 3, num_fetched = 0"


@pytest.mark.parametrize(
    "line",
    [
        pytest.param(GOOD.replace(" = ", ": "), id="colon-pairs"),
        pytest.param(GOOD.replace(" = ", "="), id="unspaced-pairs"),
        pytest.param(GOOD.replace(", ", "; "), id="semicolon-separator"),
        pytest.param(GOOD.replace(", ", " | "), id="pipe-separator"),
        pytest.param(GOOD.replace(", ", "\t"), id="tab-separator"),
        pytest.param(GOOD.replace("2026-09-17 10:58:33", "2026-09-17T10:58:33"), id="iso-timestamp"),
        pytest.param(GOOD.replace("2026-09-17 10:58:33", "09/17/2026-10:58:33"), id="logger-timestamp"),
        pytest.param(GOOD.replace("2026-09-17 10:58:33", "2026-09-17 10:58:33.123"), id="fractional-timestamp"),
        pytest.param(GOOD.replace("iter = 1, ", "iteration = 1, "), id="iter-renamed"),
        pytest.param(GOOD.replace("global_rank = 0, ", ""), id="global-rank-removed"),
        pytest.param(GOOD.replace("rank = 0, timestamp", "timestamp"), id="rank-removed"),
        pytest.param(GOOD.replace("timestamp = 2026-09-17 10:58:33, ", ""), id="timestamp-removed"),
        pytest.param(GOOD.replace("num_active = 3", "num_active: 3"), id="colon-field-after-timestamp"),
    ],
)
def test_trtllm_iteration_rejects_format_changes(line):
    """A changed grammar, timestamp format or missing identity key rejects the whole line."""
    assert parse_trtllm_iteration(line) is None
    assert parse_engine_log(line) is None


@pytest.mark.xfail(
    strict=True,
    reason="Partial format drift is accepted: a segment that is not `identifier = value` is not split "
    "off, so it is glued onto the previous field's value as a string (e.g. num_active = "
    "'3, num_fetched: 0') instead of rejecting the line.",
)
@pytest.mark.parametrize(
    "line",
    [
        pytest.param(GOOD.replace("num_fetched = 0", "num_fetched: 0") + ", num_dummy = 0", id="one-colon-field"),
        pytest.param(GOOD.replace("num_fetched = 0", "num_fetched=0"), id="one-unspaced-field"),
        pytest.param(GOOD.replace(", num_fetched", "; num_fetched"), id="one-semicolon-separator"),
        pytest.param(GOOD.replace(", num_fetched", " | num_fetched"), id="one-pipe-separator"),
        pytest.param(GOOD.replace("num_fetched = 0", "num-fetched = 0"), id="non-identifier-key"),
        pytest.param(GOOD + ", ", id="trailing-separator"),
        pytest.param(GOOD + ", num_dummy = ", id="empty-last-value"),
    ],
)
def test_trtllm_iteration_rejects_partial_format_drift(line):
    """One drifted field or separator rejects the line instead of corrupting a neighbour."""
    assert parse_trtllm_iteration(line) is None


@pytest.mark.xfail(
    strict=True,
    reason="Fields start at the first `iter = `, so a key reordered ahead of iter is silently dropped "
    "(or, for a required key, rejects a valid line).",
)
@pytest.mark.parametrize(
    "line",
    [
        pytest.param("step = 9, " + GOOD, id="new-key-before-iter"),
        pytest.param(
            GOOD.replace("iter = 1, global_rank = 0, rank = 0", "rank = 0, iter = 1, global_rank = 0"), id="rank-first"
        ),
    ],
)
def test_trtllm_iteration_tolerates_keys_reordered_before_iter(line):
    """Reordering is a content change, including moving a key ahead of iter."""
    it = parse_trtllm_iteration("[TRT-LLM] [I] [_torch][RANK 0] " + line)
    assert it is not None
    assert {k for k, _ in it.fields} >= {k for k, _ in (seg.split(" = ") for seg in line.split(", "))}
