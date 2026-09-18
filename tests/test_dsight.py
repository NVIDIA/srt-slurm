# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Synthetic source artifacts exercise joins, not a copy of normalized output."""

from __future__ import annotations

import copy
import gzip
import hashlib
import json
import sqlite3
import sys
from itertools import pairwise
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pyarrow import ipc

from srtctl.dsight.build import build_dashboard
from srtctl.dsight.importer import Importer
from srtctl.dsight.mcp import query_trace
from srtctl.dsight.model import lifecycle
from srtctl.dsight.query import TraceDataset

ORIGIN = 1789642711000000000
CLIENT = "11111111-1111-4111-8111-111111111111"
SERVER = "22222222-2222-4222-8222-222222222222"
TRACE = "1234567890abcdef1234567890abcdef"


def write_run(root: Path) -> tuple[Path, Path]:
    logs = root / "run" / "logs"
    logs.mkdir(parents=True)
    client = logs / "agentic" / "conc_2" / "aiperf_artifacts" / "profile_export.jsonl"
    client.parent.mkdir(parents=True)
    rows = [
        {
            "metadata": {
                "x_request_id": CLIENT,
                "request_start_ns": ORIGIN,
                "request_end_ns": ORIGIN + 8_000_000_000,
                "root_correlation_id": "session-a",
                "x_correlation_id": "agent-a",
                "benchmark_phase": "profiling",
            },
            "metrics": {"time_to_first_token": {"value": 3000}, "input_sequence_length": {"value": 100}},
        },
        {
            "metadata": {
                "request_id": "client-only",
                "session_id": "session-b",
                "request_start_ns": ORIGIN + 9_000_000_000,
                "request_end_ns": ORIGIN + 10_000_000_000,
                "benchmark_phase": "profiling",
                "was_cancelled": True,
            },
            "metrics": {},
        },
        {
            "metadata": {
                "request_id": "warmup",
                "request_start_ns": ORIGIN - 10_000_000_000,
                "request_end_ns": ORIGIN - 9_000_000_000,
                "benchmark_phase": "warmup",
            },
            "metrics": {},
        },
    ]
    client.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (logs / "front_frontend_0.out").write_text(
        f'2026-09-17T10:58:31Z x_request_id="{CLIENT}" dynamo.request.id={SERVER}\n'
    )
    spans = []

    def span(ident: str, name: str, start: float, end: float, role: str = "frontend", parent: str = "root", **extra):
        attributes = {
            "dynamo.request.id": SERVER,
            "dynamo.operation.role": role,
            "dynamo.instance.id": "front" if role == "frontend" else f"{role}-host",
            "dynamo.process.epoch": "epoch-" + role,
            **extra,
        }
        spans.append(
            {
                "spanId": ident,
                "traceId": TRACE,
                "parentSpanId": parent,
                "name": name,
                "startTimeUnixNano": str(ORIGIN + int(start * 1e9)),
                "endTimeUnixNano": str(ORIGIN + int(end * 1e9)),
                "attributes": [{"key": k, "value": {"stringValue": v}} for k, v in attributes.items()],
            }
        )

    span("preprocess", "request.preprocessing", 0.01, 0.1)
    span("pselect", "kv_router.select_worker", 0.11, 0.12)
    span("proute", "kv_router.route_request", 0.13, 2.05, phase="Prefill", dp_rank="4", request_id=SERVER)
    span("padmit", "worker.admission", 0.14, 0.2, "prefill")
    span("pop", "worker.operation.prefill", 0.2, 2, "prefill")
    span("pdispatch", "request.dispatch", 0.21, 0.22, "prefill", "pop")
    span("ppump", "response.streaming.prefill", 0.23, 1.99, "prefill", "pop")
    span("dselect", "kv_router.select_worker", 2.1, 2.11)
    span("droute", "kv_router.route_request", 2.12, 7.9, phase="Decode", dp_rank="1", request_id=SERVER)
    span("dadmit", "worker.admission", 2.13, 2.2, "decode")
    span("dop", "worker.operation.decode", 2.2, 7.9, "decode")
    span("ddispatch", "request.dispatch", 2.21, 2.22, "decode", "dop")
    span("dpump", "response.streaming.decode", 2.23, 7.89, "decode", "dop")
    span("fstream", "response.streaming", 2.99, 7.99)
    otel = logs / "otel" / "collector" / "traces.jsonl"
    otel.parent.mkdir(parents=True)
    doc = {
        "resourceSpans": [
            {
                "resource": {"attributes": [{"key": "service.name", "value": {"stringValue": "dynamo"}}]},
                "scopeSpans": [{"spans": spans}],
            }
        ]
    }
    otel.write_text(json.dumps(doc) + "\n")
    for role in ("prefill", "decode"):
        (logs / f"{role}-host_{role}_w0.out").write_text(
            f"2026-09-17T10:58:33.230Z Engine ID map: request_id={SERVER} trtllm_client_id=7 disagg_request_id=99\n"
            "[TRT-LLM] iter = 12588, global_rank = 0, rank = 0, num_scheduled_requests = 18, kv_cache_util = 0.08, "
            "host_step_time = 29.9ms, prev_device_step_time = 28.6ms, timestamp = 2026-09-17 10:58:33\n"
        )
    metrics = logs / "tachometer" / "local"
    metrics.mkdir(parents=True)
    points = [
        {
            "timestamp_ns": ORIGIN + t * 1_000_000_000,
            "metric_name": 'trtllm_num_requests_running{model="test"}',
            "metric_value": float(t),
            "scraper_endpoint": "decode",
            "hostname": "decode-host",
            "gpu": "0",
            "worker_role": "decode",
            "worker_index": "0",
            "worker_process": "epoch-decode",
            "rank": "0",
        }
        for t in (1, 3, 8)
    ]
    points.append({**points[1], "rank": "1", "metric_value": 99.0})
    pq.write_table(pa.Table.from_pylist(points), metrics / "final.parquet")
    pq.write_table(pa.Table.from_pylist([{**points[0], "metric_value": 900.0}]), metrics / "compacted.parquet")
    tail = pa.Table.from_pylist([points[-1]])
    with (metrics / "tail.arrow").open("wb") as stream, ipc.new_stream(stream, tail.schema) as writer:
        writer.write_table(tail)
    mirror = logs / "tachometer" / "upload"
    mirror.mkdir()
    pq.write_table(pa.Table.from_pylist([{**points[0], "metric_value": 999.0}]), mirror / "mirror.parquet")
    sqlites = root / "sqlites"
    sqlites.mkdir()
    with sqlite3.connect(sqlites / "decode-host_decode_w0_profile_rank0.sqlite") as conn:
        conn.executescript("""
            CREATE TABLE TARGET_INFO_SESSION_START_TIME (utcEpochNs INTEGER, systemClockNs INTEGER);
            CREATE TABLE ANALYSIS_DETAILS (startTime INTEGER, stopTime INTEGER);
            CREATE TABLE NVTX_EVENTS (start INTEGER, end INTEGER, text TEXT, textId INTEGER, globalTid INTEGER);
            CREATE TABLE StringIds (id INTEGER, value TEXT);
        """)
        conn.execute("INSERT INTO TARGET_INFO_SESSION_START_TIME VALUES (?, ?)", (ORIGIN, 1000))
        conn.execute("INSERT INTO ANALYSIS_DETAILS VALUES (?, ?)", (1000, 10_000_001_000))
        conn.executemany(
            "INSERT INTO NVTX_EVENTS VALUES (?, ?, ?, NULL, 17)",
            [
                (2_300_000_000, 2_400_000_000, "[Executor] _forward_step 12587: 18 gen reqs"),
                (2_500_000_000, 2_600_000_000, "[Executor] _forward_step 12588: 19 gen reqs"),
                (2_000_000_000, 1_000_000_000, "[Executor] invalid"),
            ],
        )
    return logs, sqlites


@pytest.fixture
def artifacts(tmp_path):
    return write_run(tmp_path)


def test_real_source_joins_and_inclusive_lifecycle(artifacts):
    logs, sqlites = artifacts
    data = Importer(logs, sqlites, iteration_timezone="UTC").run()
    request = data["requests"][0]
    assert data["audit"]["client_requests"] == 2
    assert data["audit"]["excluded_client_rows"] == 1
    assert request["server_ids"] == [SERVER]
    assert request["workers"] == ["decode-0", "prefill-0"]
    assert {e["process"] for e in request["engine"]} == {"epoch-prefill", "epoch-decode"}
    assert all(not e["identity_ambiguous"] for e in request["engine"])
    model = request["lifecycle"]
    assert model["available"]
    assert not model["issues"]
    stages = model["stages"]
    assert sum(s["end"] - s["start"] for s in stages) == pytest.approx(8)
    assert sum(s["end"] - s["start"] for s in stages[:-1]) == pytest.approx(3)
    assert all(a["end"] == b["start"] for a, b in pairwise(stages))
    assert len(model["rows"][-1]) == len(stages)
    raw = {s["id"]: s for s in model["activities"]}
    assert raw["dop"]["kind"] == "envelope"
    assert raw["ddispatch"]["parent_activity"] == "dop"
    assert raw["dpump"]["parent_activity"] == "dop"
    assert raw["fstream"]["kind"] == "concurrent"
    assert raw["fstream"]["start"] < raw["dop"]["end"]
    assert raw["pselect"]["routing_context"]["phase"] == "Prefill"
    assert raw["dselect"]["routing_context"]["phase"] == "Decode"
    # A worker-local engine ID reused by another worker must not collide.
    assert data["audit"]["ambiguous_engine_ids"] == 0
    assert data["requests"][1]["lifecycle"]["available"] is False
    assert data["requests"][1]["lifecycle"]["stages"] == []
    assert data["requests"][1]["lifecycle"]["activities"] == []
    assert data["requests"][1]["status"] == "cancelled"


def test_raw_metrics_preserve_rank_and_skip_compacted_shards_and_upload_mirror(artifacts):
    data = Importer(artifacts[0]).run()
    assert len(data["metrics"]) == 2
    assert sum(len(s["points"]) for s in data["metrics"]) == 4
    assert data["audit"]["duplicate_metric_points"] == 1
    assert {s["rank"] for s in data["metrics"]} == {"0", "1"}
    assert max(p[1] for s in data["metrics"] for p in s["points"]) == 99
    assert all("upload" not in s["path"] for s in data["sources"])


def test_nsys_is_read_only_and_preserves_window_and_rank(artifacts):
    logs, sqlites = artifacts
    path = next(sqlites.iterdir())
    before = hashlib.sha256(path.read_bytes()).digest()
    data = Importer(logs, sqlites, max_profile_events=1).run()
    assert hashlib.sha256(path.read_bytes()).digest() == before
    profile = data["profiles"][0]
    assert profile["worker"] == "decode-0" and profile["rank"] == 0
    assert profile["events"][0][:2] == [2.3, 2.4]
    assert profile["invalid_or_boundary_ranges"] == 1
    assert profile["truncated"]
    assert "limit reached" in " ".join(data["meta"]["warnings"])


def test_missing_optional_sources_still_build_a_client_dashboard(artifacts, tmp_path):
    logs, _ = artifacts
    bare = tmp_path / "bare"
    bare.mkdir()
    client = next(logs.rglob("profile_export.jsonl"))
    (bare / "profile_export.jsonl").write_bytes(client.read_bytes())
    result = build_dashboard(bare, tmp_path / "dashboard")
    assert result["counts"]["requests"] == 2
    assert result["counts"]["profiles"] == 0
    assert result["audit"]["clients_with_lifecycle"] == 0
    assert not any("lifecycle" in warning or "OTel" in warning for warning in result["meta"]["warnings"])
    assert query_trace(result["output"], "lifecycle", request_id=CLIENT)["available"] is False
    html = Path(result["html"]).read_text()
    assert "__TRACE_DATA_GZIP_BASE64__" not in html
    assert '<script src="' not in html


@pytest.mark.parametrize("mode", ["missing", "empty", "unjoined", "unsupported", "disabled"])
def test_optional_otel_preserves_independent_sources(artifacts, mode):
    logs, sqlites = artifacts
    path = next(logs.glob("otel/*/traces.jsonl"))
    if mode == "missing":
        path.unlink()
    elif mode == "empty":
        path.write_text("")
    elif mode == "unjoined":
        path.write_text(path.read_text().replace(SERVER, "33333333-3333-4333-8333-333333333333"))
    elif mode == "unsupported":
        doc = json.loads(path.read_text())
        for span in doc["resourceSpans"][0]["scopeSpans"][0]["spans"]:
            span["name"] = "request.lifecycle"
        path.write_text(json.dumps(doc) + "\n")
    else:
        # Explicit opt-out must not even parse an existing, malformed OTel file.
        path.write_text("not valid OTLP JSON\n")
    data = Importer(logs, sqlites, otel=mode != "disabled", iteration_timezone="UTC").run()
    assert data["meta"]["otel_enabled"] is (mode != "disabled")
    assert data["audit"]["clients_with_lifecycle"] == 0
    for request in data["requests"]:
        model = TraceDataset(data).query("lifecycle", request_id=request["id"])
        assert model["available"] is False
        assert all(model[field] == [] for field in ("stages", "activities", "milestones", "rows"))
    assert data["requests"][0]["ttft_ms"] == 3000
    assert data["requests"][0]["workers"] == ["decode-0", "prefill-0"]
    assert data["requests"][0]["bridge_evidence"]
    assert len(data["metrics"]) == 2
    assert len(data["profiles"]) == 1
    assert len(data["iterations"]) == 2
    if mode in ("missing", "empty", "disabled"):
        assert not any(source["kind"] == "otel" for source in data["sources"])
    if mode != "unsupported":
        assert data["audit"]["joined_spans"] == 0
        assert all(not request["spans"] for request in data["requests"])
        assert all(entry["process"] is None for entry in data["requests"][0]["engine"])


def test_queries_match_evidence_and_are_bounded(artifacts):
    data = Importer(*artifacts, iteration_timezone="UTC").run()
    dataset = TraceDataset(data)
    assert dataset.query("requests", start=0, end=4, worker="decode-0")["total"] == 1
    assert dataset.query("requests", start=0, end=4, offset=1)["items"] == []
    assert dataset.query("lifecycle", request_id=CLIENT) == data["requests"][0]["lifecycle"]
    assert dataset.query("metrics", start=2, end=4, rank=0, points=True)["items"][0]["points"][0][:2] == [3, 3]
    assert dataset.query("nsys", start=2.35, end=2.36, rank=0)["total"] == 1
    assert dataset.query("nsys", start=2, end=3, rank=1)["total"] == 0
    assert dataset.query("iterations", start=2.1, end=2.2, rank=0)["total"] == 2
    assert dataset.query("iterations")["items"][0]["iteration"] == 12588  # no universal shift
    for options in ({"start": -1}, {"end": float("nan")}, {"start": 1, "end": 1}, {"limit": 1001}, {"offset": -1}):
        with pytest.raises(ValueError):
            dataset.query("requests", **options)
    with pytest.raises(ValueError, match="Unknown client"):
        dataset.query("request", request_id="missing")


def test_unzoned_iterations_are_not_silently_aligned(artifacts):
    dataset = TraceDataset(Importer(artifacts[0]).run())
    result = dataset.query("iterations")
    assert result["total"] == 0 and result["unaligned_rows"] == 2
    assert "no timezone" in " ".join(dataset.data["meta"]["limitations"])


@pytest.mark.parametrize("mutation", ["skew", "retry", "duplicate_milestone", "invalid_ttft"])
def test_uncertain_lifecycle_does_not_invent_a_partition(artifacts, mutation):
    request = copy.deepcopy(Importer(artifacts[0]).run()["requests"][0])
    if mutation == "skew":
        next(s for s in request["spans"] if s["id"] == "ddispatch")["end"] = 1
    elif mutation == "retry":
        request["server_ids"].append("another-attempt")
    elif mutation == "duplicate_milestone":
        request["spans"].append({**request["spans"][0], "id": "again"})
    else:
        request["first"] = 99
    result = lifecycle(request)
    assert result["issues"]
    assert len(result["stages"]) <= 2
    assert len(result["activities"]) >= 12
    assert all(s["end"] >= s["start"] for s in result["stages"])


def test_multiple_exports_require_explicit_client_selection(artifacts):
    logs, _ = artifacts
    source = next(logs.rglob("profile_export.jsonl"))
    (logs / "profile_export.jsonl").write_bytes(source.read_bytes())
    with pytest.raises(ValueError, match="--client"):
        Importer(logs).run()
    assert Importer(logs, client=source).run()["audit"]["client_requests"] == 2


def test_rebuild_is_safe_and_mcp_reads_new_generation(artifacts, tmp_path, monkeypatch):
    logs, sqlites = artifacts
    out = tmp_path / "out"
    result = build_dashboard(logs, out, sqlites=sqlites)
    assert query_trace(str(out), "summary")["counts"]["requests"] == 2
    before = (out / "index.html").read_bytes()
    with pytest.raises(ValueError, match="Output exists"):
        build_dashboard(logs, logs / "otel")
    with monkeypatch.context() as patch:
        patch.setattr(Importer, "run", lambda _: (_ for _ in ()).throw(ValueError("corrupt source")))
        with pytest.raises(ValueError, match="corrupt source"):
            build_dashboard(logs, out)
    assert (out / "index.html").read_bytes() == before
    build_dashboard(logs, out, job="new-generation")
    assert query_trace(str(out), "summary")["meta"]["job"] == "new-generation"
    assert query_trace(str(out), "requests", limit=1001)["ok"] is False
    assert result["counts"]["profiles"] == 1


@pytest.mark.parametrize("no_otel", [False, True])
def test_cli_and_embedded_data_share_the_same_contract(artifacts, tmp_path, monkeypatch, capsys, no_otel):
    from srtctl.cli.submit import main

    logs, _ = artifacts
    out = tmp_path / "out"
    args = ["srtctl", "dsight", "build", str(logs), "--output", str(out)]
    if no_otel:
        args.append("--no-otel")
    monkeypatch.setattr(sys, "argv", args)
    with pytest.raises(SystemExit) as exit_info:
        main()
    assert exit_info.value.code == 0
    result = json.loads(capsys.readouterr().out)
    assert result["counts"]["requests"] == 2
    payload = json.loads(gzip.decompress((out / "trace-data.json.gz").read_bytes()))
    assert payload["schema"] == "srtctl-trace/1"
    assert payload["meta"]["otel_enabled"] is not no_otel
    assert payload["requests"][0]["lifecycle"]["available"] is not no_otel
    monkeypatch.setattr(sys, "argv", ["srtctl", "dsight", "query", str(out), "requests", "--from", "0", "--to", "4"])
    with pytest.raises(SystemExit) as exit_info:
        main()
    assert exit_info.value.code == 0
    assert json.loads(capsys.readouterr().out)["total"] == 1


def test_mcp_registration_exposes_read_only_query():
    import asyncio

    from srtctl.mcp.server import mcp

    tools = asyncio.run(mcp.list_tools())
    tool = next(t for t in tools if t.name == "query_trace")
    wire = tool.model_dump(by_alias=True)
    assert wire["annotations"]["readOnlyHint"] is True
    assert "dataset" in wire["inputSchema"]["properties"]


def test_conflicting_otel_duplicate_is_rejected(artifacts):
    logs, _ = artifacts
    path = next(logs.glob("otel/*/traces.jsonl"))
    raw = json.loads(path.read_text())
    spans = raw["resourceSpans"][0]["scopeSpans"][0]["spans"]
    spans.append({**spans[0], "endTimeUnixNano": str(ORIGIN + 900_000_000)})
    path.write_text(json.dumps(raw) + "\n")
    with pytest.raises(ValueError, match="conflicting duplicate OTel"):
        Importer(logs).run()


def test_colocated_worker_ambiguity_does_not_assign_shared_span(artifacts):
    logs, _ = artifacts
    old = logs / "prefill-host_prefill_w0.out"
    (logs / "prefill-host_prefill_w1.out").write_bytes(old.read_bytes())
    data = Importer(logs).run()
    request = data["requests"][0]
    assert data["audit"]["ambiguous_span_workers"] > 0
    assert all(s["worker"] is None for s in request["spans"] if s["role"] == "prefill")
    assert all(s["worker"] == "decode-0" for s in request["spans"] if s["role"] == "decode")


def test_exact_nsys_limit_is_not_reported_as_truncation(artifacts):
    data = Importer(*artifacts, max_profile_events=2).run()
    assert len(data["profiles"][0]["events"]) == 2
    assert not data["profiles"][0]["truncated"]


def test_rebuild_preserves_unmanaged_files(artifacts, tmp_path):
    logs, _ = artifacts
    out = tmp_path / "report"
    build_dashboard(logs, out)
    (out / "notes.md").write_text("my investigation")
    with pytest.raises(ValueError, match="did not generate"):
        build_dashboard(logs, out)
    assert (out / "notes.md").read_text() == "my investigation"


def test_metric_rank_labels_survive_null_columns(artifacts):
    logs, _ = artifacts
    path = logs / "tachometer/local/final.parquet"
    rows = pq.read_table(path).to_pylist()
    for row in rows:
        row["rank"] = None
        row["global_rank"] = None
        row["metric_name"] = 'trtllm_num_requests_running{rank="3",model="test"}'
    pq.write_table(pa.Table.from_pylist(rows), path)
    data = TraceDataset(Importer(logs).run())
    result = data.query("metrics", rank=3, points=True)
    assert result["total"] == 1
    series = result["items"][0]
    assert series["rank"] == "3" and series["rank_kind"] == "rank"
    assert series["labels"]["metric.rank"] == "3"
    assert series["samples"] == 4
