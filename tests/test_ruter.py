# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Coverage for the small, offline Dynamo ruter post-processor."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

from srtctl.cli.mixins.postprocess_stage import PostProcessStageMixin
from srtctl.ruter.normalize import normalize_run, parse_router_line


def _write(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents)


def test_normalize_run_writes_separate_router_and_worker_jsonl(tmp_path: Path) -> None:
    _write(
        tmp_path / "logs" / "router.log",
        "2026-08-20T04:00:30.000000Z DEBUG selector: Formula for worker_id=42 dp_rank=0 with 32.00 effective cached blocks: 80.500 = prefill_load_scale * adjusted_prefill_blocks + decode_blocks + active_request_cost_blocks = 1.000 * 24.500 + 56.000 + 0.000 (raw_prefill_blocks: 56.500, overlap_credit_blocks: 32.000, overlap_credit_decay: 1.000) request_id=internal-1\n"
        "2026-08-20T04:00:30.010000Z DEBUG router: [ROUTING] Best: worker_42 dp_rank=0 with 8/12 blocks overlap request_id=internal-1 worker_id=42 overlap_blocks=8 total_blocks=12 x_request_id=client-1\n",
    )
    _write(
        tmp_path / "logs" / "node-a_agg_w3.out",
        "[2026-08-20 04:40:20] Prefill batch, #new-seq: 1, #new-token: 8192, #cached-token: 0\n"
        "[2026-08-20 04:40:21] INFO handle_payload: request received instance_id=42 request_id=internal-1\n",
    )
    _write(tmp_path / "artifacts" / "tachometer" / "final.parquet", "not parsed")

    report = normalize_run(tmp_path)

    assert report.router_events == 2
    assert report.worker_events == 2
    assert report.worker_logs == 1
    assert report.tachometer_parquet == "artifacts/tachometer/final.parquet"
    assert (tmp_path / "artifacts" / "tachometer" / "final.parquet").read_text() == "not parsed"

    bundle = tmp_path / "logs" / ".ruter"
    router_events = [json.loads(line) for line in (bundle / "router-events.jsonl").read_text().splitlines()]
    worker_events = [json.loads(line) for line in (bundle / "worker-events.jsonl").read_text().splitlines()]
    manifest = json.loads((bundle / "manifest.json").read_text())

    assert router_events[0]["kind"] == "routing_formula"
    assert router_events[0]["fields"]["cost_blocks"] == "80.500"
    assert router_events[1]["request_id"] == "client-1"
    assert router_events[1]["fields"]["dynamo_request_id"] == "internal-1"
    assert [event["worker_index"] for event in worker_events] == [3, 3]
    assert manifest["tachometer_parquet"] == "artifacts/tachometer/final.parquet"


def test_router_parser_preserves_overlap_when_only_debug_message_has_it() -> None:
    event = parse_router_line(
        "2026-08-20T04:00:30.0Z DEBUG router: request_id=req-2 [ROUTING] Best: worker_99 dp_rank=0 with 8/12 blocks overlap"
    )

    assert event is not None
    assert event.kind == "routing_decision"
    assert event.fields["worker_id"] == "worker_99"
    assert event.fields["overlap_blocks"] == "8"
    assert event.fields["total_blocks"] == "12"


def test_router_parser_preserves_json_debug_request_and_worker_ids() -> None:
    event = parse_router_line(
        '{"time":"2026-08-20T19:15:32.905245Z","message":"[ROUTING] Best: worker_42 dp_rank=0 with 8/16 blocks overlap","request_id":"internal-1","worker_id":42}'
    )

    assert event is not None
    assert event.kind == "routing_decision"
    assert event.request_id == "internal-1"
    assert event.fields["dynamo_request_id"] == "internal-1"
    assert event.fields["worker_id"] == "42"
    assert event.fields["overlap_blocks"] == "8"


def test_router_parser_records_disaggregation_dispatch_phase() -> None:
    event = parse_router_line(
        '{"time":"2026-08-20T19:15:32.905245Z","message":"TCP client sending request",'
        '"request_id":"internal-1","span_name":"kv_router.route_request","phase":"Decode"}'
    )

    assert event is not None
    assert event.kind == "routing_dispatch"
    assert event.request_id == "internal-1"
    assert event.fields["phase"] == "Decode"


def test_slurm_postprocess_normalizes_before_artifact_upload(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    _write(
        log_dir / "router.log",
        "2026-08-20T04:00:30.0Z DEBUG router: [ROUTING] Best: worker_1 with 1/2 blocks overlap\n",
    )
    _write(log_dir / "node-a_agg_w0.out", "[2026-08-20 04:40:20] Prefill batch, #new-seq: 1\n")

    mixin = PostProcessStageMixin()
    mixin.runtime = MagicMock(log_dir=log_dir)
    mixin.config = MagicMock()
    mixin.config.frontend.type = "dynamo"
    mixin.config.observability.enabled = True

    mixin._normalize_ruter()

    assert (log_dir / ".ruter" / "router-events.jsonl").is_file()
    assert (log_dir / ".ruter" / "worker-events.jsonl").is_file()
