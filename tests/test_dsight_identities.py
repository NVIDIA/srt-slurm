# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Common Dynamo identity joins from captured log and OTLP formats."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_dsight import CLIENT, SERVER, write_run

from srtctl.dsight.importer import Importer
from srtctl.dsight.query import TraceDataset


def encode(fields: dict[str, str], style: str = "flat") -> str:
    if style == "text":
        return "2026-09-17T10:58:33Z " + " ".join(f"{k}={json.dumps(v)}" for k, v in fields.items()) + "\n"
    if style == "ansi":
        return "\x1b[32m" + encode(fields, "text") + "\x1b[0m\n"
    if style == "nested":
        return json.dumps({"spans": [fields], "fields": {"message": "SPAN_CLOSED"}}) + "\n"
    if style == "span":
        return json.dumps({"span": fields, "fields": {"message": "SPAN_CLOSED"}}) + "\n"
    return json.dumps(fields) + "\n"


def binding(role: str, server: str = SERVER, process: str | None = None) -> dict[str, str]:
    return {
        "dynamo.request.id": server,
        "dynamo.instance.id": f"{role}-host",
        "dynamo.operation.role": role,
        "dynamo.process.epoch": process or f"epoch-{role}",
    }


def dynamo_run(root: Path, style: str = "flat") -> tuple[Path, Path]:
    logs, sqlites = write_run(root)
    (logs / "front_frontend_0.out").write_text(encode({"x_request_id": CLIENT, "dynamo.request.id": SERVER}, style))
    for role in ("prefill", "decode"):
        (logs / f"{role}-host_{role}_w0.out").write_text(encode(binding(role), style) * 2)
    return logs, sqlites


@pytest.mark.parametrize("style", ["flat", "nested", "span", "text", "ansi"])
def test_dynamo_identities_join_without_engine_local_ids(tmp_path, style):
    logs, sqlites = dynamo_run(tmp_path, style)
    data = Importer(logs, sqlites).run()
    request = TraceDataset(data).query("request", request_id=CLIENT)
    assert request["server_ids"] == [SERVER]
    assert request["engine"] == []
    assert request["workers"] == ["decode-0", "prefill-0"]
    assert len(request["worker_bindings"]) == 2
    assert {b["process"] for b in request["worker_bindings"]} == {"epoch-decode", "epoch-prefill"}
    for b in request["worker_bindings"]:
        assert not b["ambiguous"] and b["evidence"][1] == 1
        assert "client_id" not in b and "disagg_id" not in b
        assert data["sources"][b["evidence"][0]]["path"].endswith(f"{b['host']}_{b['role']}_w0.out")
    assert request["lifecycle"]["available"]
    assert not request["lifecycle"]["issues"]
    assert data["audit"]["joined_spans"] == 14
    assert all(s["worker"] == f"{s['role']}-0" for s in request["spans"] if s["role"] != "frontend")
    assert TraceDataset(data).query("requests", worker="decode-0")["total"] == 1


@pytest.mark.parametrize("style", ["flat", "text"])
def test_legacy_frontend_request_id_field(tmp_path, style):
    logs, _ = dynamo_run(tmp_path)
    (logs / "front_frontend_0.out").write_text(encode({"x_request_id": CLIENT, "request_id": SERVER}, style))
    data = Importer(logs).run()
    assert data["requests"][0]["server_ids"] == [SERVER]
    assert data["audit"]["joined_spans"] == 14


@pytest.mark.parametrize("duplicate", [False, True])
def test_flat_otel_layout_and_duplicate_collector_copy(tmp_path, duplicate):
    logs, _ = dynamo_run(tmp_path)
    nested = logs / "otel/collector/traces.jsonl"
    flat = logs / "otel/traces.jsonl"
    if duplicate:
        flat.write_bytes(nested.read_bytes())
    else:
        nested.rename(flat)
    data = Importer(logs).run()
    assert data["audit"]["joined_spans"] == 14
    assert data["audit"].get("duplicate_spans", 0) == (14 if duplicate else 0)
    assert len(data["requests"][0]["spans"]) == 14


def test_conflicting_otel_identity_in_duplicate_copy_is_rejected(tmp_path):
    logs, _ = dynamo_run(tmp_path)
    doc = json.loads((logs / "otel/collector/traces.jsonl").read_text())
    doc["resourceSpans"][0]["scopeSpans"][0]["spans"][0]["attributes"][2]["value"] = {"stringValue": "other-host"}
    (logs / "otel/traces.jsonl").write_text(json.dumps(doc) + "\n")
    with pytest.raises(ValueError, match="conflicting duplicate OTel"):
        Importer(logs).run()


def test_disabled_otel_keeps_worker_evidence_and_never_parses_flat_file(tmp_path):
    logs, _ = dynamo_run(tmp_path)
    (logs / "otel/traces.jsonl").write_text("malformed but disabled\n")
    data = Importer(logs, otel=False).run()
    request = data["requests"][0]
    assert request["workers"] == ["decode-0", "prefill-0"]
    assert request["engine"] == [] and request["spans"] == []
    assert not request["lifecycle"]["available"]
    assert all(b["process"] for b in request["worker_bindings"])
    assert not any(s["kind"] == "otel" for s in data["sources"])


@pytest.mark.parametrize("engine_maps", [False, True])
def test_conflicting_bindings_stay_in_evidence_but_out_of_request_path(tmp_path, engine_maps):
    logs, _ = write_run(tmp_path) if engine_maps else dynamo_run(tmp_path)
    path = logs / "decode-host_decode_w0.out"
    (logs / "decode-host_decode_w1.out").write_bytes(path.read_bytes())
    data = Importer(logs).run()
    request = data["requests"][0]
    assert request["workers"] == ["prefill-0"]
    assert len(request["worker_bindings"]) == 3
    assert all(b["ambiguous"] for b in request["worker_bindings"] if b["role"] == "decode")
    assert all(s["worker"] is None for s in request["spans"] if s["role"] == "decode")
    assert data["audit"]["ambiguous_worker_bindings"] == 2
    assert TraceDataset(data).query("requests", worker="decode-0")["total"] == 0
    assert bool(request["engine"]) is engine_maps


def test_distinct_processes_disambiguate_colocated_workers(tmp_path):
    logs, _ = dynamo_run(tmp_path)
    (logs / "decode-host_decode_w1.out").write_text(encode(binding("decode", process="other-process")))
    data = Importer(logs).run()
    request = data["requests"][0]
    assert all(s["worker"] == "decode-0" for s in request["spans"] if s["role"] == "decode")
    assert request["workers"] == ["decode-0", "decode-1", "prefill-0"]
    assert not data["audit"]["ambiguous_worker_bindings"]


def test_server_attempts_are_not_deduplicated_or_used_for_another_attempt(tmp_path):
    logs, _ = dynamo_run(tmp_path)
    other = "33333333-3333-4333-8333-333333333333"
    front = logs / "front_frontend_0.out"
    front.write_text(front.read_text() + encode({"x_request_id": CLIENT, "dynamo.request.id": other}))
    path = logs / "decode-host_decode_w0.out"
    path.write_text(path.read_text() + encode(binding("decode", other)))
    (logs / "decode-host_decode_w1.out").write_text(encode(binding("decode", other)))
    data = Importer(logs).run()
    request = data["requests"][0]
    assert request["server_ids"] == [SERVER, other]
    assert len(request["worker_bindings"]) == 4
    decode = [b for b in request["worker_bindings"] if b["role"] == "decode"]
    assert {b["server_id"] for b in decode} == {SERVER, other}
    assert all(b["ambiguous"] == (b["server_id"] == other) for b in decode)
    assert all(s["worker"] == "decode-0" for s in request["spans"] if s["role"] == "decode")
    assert request["workers"] == ["decode-0", "prefill-0"]


@pytest.mark.parametrize("recorded_host", [False, True])
def test_unique_host_fallback_requires_a_recorded_otel_host(tmp_path, recorded_host):
    logs, _ = dynamo_run(tmp_path)
    for p in logs.glob("*_w*.out"):
        p.write_text("")
    path = logs / "otel/collector/traces.jsonl"
    doc = json.loads(path.read_text())
    if not recorded_host:
        for span in doc["resourceSpans"][0]["scopeSpans"][0]["spans"]:
            span["attributes"] = [a for a in span["attributes"] if a["key"] != "dynamo.instance.id"]
    # A directory name is never evidence of the emitting host.
    renamed = path.parent.with_name("decode-host")
    path.parent.rename(renamed)
    (renamed / path.name).write_text(json.dumps(doc) + "\n")
    data = Importer(logs).run()
    request = data["requests"][0]
    assert request["worker_bindings"] == []
    assert request["workers"] == (["decode-0", "prefill-0"] if recorded_host else [])
    if recorded_host:
        assert all("unique recorded host" in s["worker_basis"] for s in request["spans"] if s["role"] != "frontend")


@pytest.mark.parametrize(
    "field,value",
    [
        ("dynamo.instance.id", "wrong-host"),
        ("dynamo.operation.role", "frontend"),
        ("dynamo.process.epoch", ""),
        ("dynamo.process.epoch", 123),
        ("dynamo.request.id", "not-a-uuid"),
        ("dynamo.request.id", "33333333-3333-4333-8333-333333333333"),
    ],
)
def test_unusable_worker_identity_is_not_a_binding(tmp_path, field, value):
    logs, _ = dynamo_run(tmp_path)
    for p in logs.glob("*_w*.out"):
        p.write_text("")
    fields = {**binding("decode"), field: value}
    (logs / "decode-host_decode_w0.out").write_text(json.dumps(fields) + "\n")
    data = Importer(logs, otel=False).run()
    assert data["requests"][0]["worker_bindings"] == []
    assert data["requests"][0]["workers"] == []


def test_malformed_or_unrelated_json_is_ignored_and_span_context_can_supply_ids(tmp_path):
    logs, _ = dynamo_run(tmp_path)
    prefix = (
        "{bad x_request_id json\n"
        "[]\n"
        '{"spans": null, "x_request_id": false}\n'
        '{"x_request_id": ["not-an-id"], "dynamo.request.id": {}}\n'
    )
    path = logs / "front_frontend_0.out"
    path.write_text(prefix + encode({"x_request_id": CLIENT, "dynamo.request.id": SERVER}, "nested"))
    data = Importer(logs).run()
    assert data["requests"][0]["server_ids"] == [SERVER]
    assert data["requests"][0]["bridge_evidence"][0][1] == 5


def test_aggregated_role_is_normalized_across_worker_log_and_otel(tmp_path):
    logs, _ = dynamo_run(tmp_path)
    (logs / "decode-host_decode_w0.out").unlink()
    fields = {**binding("decode"), "dynamo.operation.role": "aggregated"}
    (logs / "decode-host_aggregated_w0_e1.out").write_text(encode(fields))
    path = logs / "otel/collector/traces.jsonl"
    doc = json.loads(path.read_text())
    for span in doc["resourceSpans"][0]["scopeSpans"][0]["spans"]:
        for a in span["attributes"]:
            if a["key"] == "dynamo.operation.role" and a["value"]["stringValue"] == "decode":
                a["value"]["stringValue"] = "aggregated"
    path.write_text(json.dumps(doc) + "\n")
    request = Importer(logs).run()["requests"][0]
    assert request["workers"] == ["agg-0", "prefill-0"]
    assert all(s["worker"] == "agg-0" for s in request["spans"] if s["role"] == "agg")


def test_conflicting_json_frontend_bridge_is_rejected(tmp_path):
    logs, _ = dynamo_run(tmp_path)
    path = logs / "front_frontend_0.out"
    path.write_text(path.read_text() + encode({"x_request_id": "client-only", "dynamo.request.id": SERVER}))
    with pytest.raises(ValueError, match="Ambiguous client/server bridge"):
        Importer(logs).run()


def test_process_binding_does_not_use_collector_directory_as_span_host(tmp_path):
    logs, _ = dynamo_run(tmp_path)
    path = logs / "otel/collector/traces.jsonl"
    doc = json.loads(path.read_text())
    for span in doc["resourceSpans"][0]["scopeSpans"][0]["spans"]:
        span["attributes"] = [a for a in span["attributes"] if a["key"] != "dynamo.instance.id"]
    directory = path.parent.with_name("decode-host")
    path.parent.rename(directory)
    (directory / path.name).write_text(json.dumps(doc) + "\n")
    request = Importer(logs).run()["requests"][0]
    # The logs still prove the request path, but these spans have no recorded host.
    assert request["workers"] == ["decode-0", "prefill-0"]
    assert all(s["worker"] is None for s in request["spans"])
