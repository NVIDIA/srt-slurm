# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Engine and partial-capture contracts, exercised through real source formats."""

from __future__ import annotations

import json
import shutil
import sqlite3
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from test_dsight import CLIENT, ORIGIN, SERVER, write_run

from srtctl.dsight.build import build_dashboard
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
                    "scraper_endpoint": "decode",
                    "hostname": "decode-host",
                    "worker_role": "decode",
                    "worker_index": "0",
                }
                for name, value in (
                    ("tokenspeed:num_requests_running", 4.0),
                    ("tokenspeed:num_requests_waiting", 12.0),
                    ("tokenspeed:kv_cache_usage_perc", 0.75),
                )
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
    assert all(m["group"] == "worker" for m in data["metrics"])


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
