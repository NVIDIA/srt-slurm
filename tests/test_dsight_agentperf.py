# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from srtctl.dsight.importer import Importer


def test_agentperf_prefers_phase_analysis_and_joins_http_identity(tmp_path):
    logs = tmp_path / "logs"
    native = logs / "agentperf"
    native.mkdir(parents=True)
    row = {
        "phase_idx": 0,
        "request_id": "record-id",
        "x_request_id": "a" * 32,
        "user_id": 4,
        "conversation_id": "conversation",
        "conversation_idx": 2,
        "start_time": 1789642701.1234567,
        "end_time": 1789642703.1234567,
        "success": True,
        "ttft": 1.1,
        "server_input_tokens": 99,
        "server_output_tokens": 30,
    }
    warmup = {
        **row,
        "request_id": "settling",
        "x_request_id": "b" * 32,
        "start_time": 1789642690.0,
        "end_time": 1789642692.0,
    }
    (native / "requests.jsonl").write_text(json.dumps(warmup) + "\n" + json.dumps(row) + "\n")
    analysis = {**row, "ttft": 0.9, "response_text": {"secret": "MUST_NOT_BE_EMBEDDED"}}
    (native / "model__4u__phase0__dur30s__settle10s__traj3.jsonl").write_text(json.dumps(analysis) + "\n")
    (native / "phase_manifest.jsonl").write_text(
        json.dumps({"phase_idx": 0, "settling_end": 1789642700.0, "actual_phase_end": 1789642710.0}) + "\n"
    )
    server = "12345678-1234-4234-8234-123456789012"
    (logs / "front_frontend_0.out").write_text(f'x_request_id="{row["x_request_id"]}" dynamo.request.id={server}\n')
    data = Importer(logs).run()
    assert data["audit"]["excluded_client_rows"] == 1
    assert data["audit"]["client_requests"] == 1
    request = data["requests"][0]
    assert request["id"] == "a" * 32 and request["server_ids"] == [server]
    assert request["ttft_ms"] == pytest.approx(900)
    assert request["end"] - request["start"] == pytest.approx(2)
    assert request["raw_start_ns"] == "1789642701123456700"
    assert request["session"] == "phase-0/user-4/conversation-conversation"
    assert request["timing_quality"] == "phase analysis"
    assert request["evidence"] != request["identity_evidence"]
    assert request["phase_evidence"] is not None
    assert "MUST_NOT_BE_EMBEDDED" not in json.dumps(data)


def test_agentperf_liveness_and_missing_manifest_are_explicit(tmp_path):
    row = {
        "request_id": "req",
        "user_id": 1,
        "conversation_id": "c",
        "conversation_idx": 0,
        "start_time": 1789642700.0,
        "end_time": 1789642701.0,
        "success": False,
        "ttft": None,
    }
    (tmp_path / "requests.jsonl").write_text(json.dumps(row) + "\n")
    data = Importer(tmp_path).run()
    request = data["requests"][0]
    assert request["status"] == "error" and request["first"] is None
    assert request["timing_quality"] == "request log (liveness)"
    assert "liveness" in " ".join(data["meta"]["warnings"])
    assert data["audit"]["clients_without_phase"] == 1


def test_duplicate_agentperf_analysis_identity_is_rejected(tmp_path):
    row = {
        "phase_idx": 0,
        "request_id": "r",
        "user_id": 1,
        "conversation_id": "c",
        "conversation_idx": 0,
        "start_time": 1789642700,
        "end_time": 1789642701,
        "success": True,
    }
    (tmp_path / "requests.jsonl").write_text(json.dumps(row) + "\n")
    (tmp_path / "model__phase0__traj1.jsonl").write_text((json.dumps(row) + "\n") * 2)
    with pytest.raises(ValueError, match="ambiguous AgentPerf"):
        Importer(tmp_path).run()
