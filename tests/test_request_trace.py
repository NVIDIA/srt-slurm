# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the request-trace L2 processor and its discovery in the ingest pipeline.

Dynamo's request-trace sink rotates: ``dynamo-request-trace.000000.jsonl.gz``,
``.000001.jsonl.gz``, ... A real capture (hecate 565811, 2026-09-09) produced 12 such
shards and no bare ``dynamo-request-trace`` file, and the ingest logged
``WARN no request trace matched 'dynamo-request-trace'`` -- the axis was silently
empty. These tests pin the shard discovery, the gzip reading and the merge.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def _record(xid: str, received_ms: int, session: str = "s1", **request_fields) -> str:
    request = {
        "request_id": f"rid-{xid}",
        "x_request_id": xid,
        "model": "m",
        "input_tokens": 10,
        "output_tokens": 4,
        "request_received_ms": received_ms,
        "prefill_wait_time_ms": 1.0,
        "prefill_time_ms": 20.0,
        "ttft_ms": 21.0,
        "total_time_ms": 100.0,
        "avg_itl_ms": 26.0,
        "kv_transfer_estimated_latency_ms": 5.0,
    }
    request.update(request_fields)
    return json.dumps(
        {
            "timestamp": received_ms,
            "event": {
                "schema": "dynamo.request.trace.v1",
                "event_type": "request_end",
                "event_time_unix_ms": received_ms + 100,
                "event_source": "dynamo",
                "agent_context": {"session_id": session, "input_trigger": "user_message"},
                "request": request,
            },
        }
    )


def _write_gz(path: Path, lines: list[str]) -> Path:
    with gzip.open(path, "wt") as f:
        f.write("\n".join(lines) + "\n")
    return path


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class TestProcess:
    def test_reads_rotated_gzip_shards_as_one_stream(self, tmp_path):
        from src.ingest.request_trace import process

        # Shard order and arrival order disagree on purpose: the output must be
        # sorted by received_ms across shards, not concatenated.
        shard0 = _write_gz(tmp_path / "dynamo-request-trace.000000.jsonl.gz", [_record("a", 1000), _record("c", 3000)])
        shard1 = _write_gz(tmp_path / "dynamo-request-trace.000001.jsonl.gz", [_record("b", 2000), _record("d", 4000)])
        out = tmp_path / "request_trace.jsonl"

        n = process([str(shard0), str(shard1)], str(out))

        assert n == 4
        assert [r["x_request_id"] for r in _rows(out)] == ["a", "b", "c", "d"]

    def test_plain_file_path_still_accepted(self, tmp_path):
        from src.ingest.request_trace import process

        plain = tmp_path / "dynamo-request-trace"
        plain.write_text(_record("a", 1000) + "\n" + _record("b", 2000) + "\n")
        out = tmp_path / "out.jsonl"

        assert process(str(plain), str(out)) == 2
        assert [r["x_request_id"] for r in _rows(out)] == ["a", "b"]

    def test_torn_last_line_in_a_shard_is_skipped_not_fatal(self, tmp_path):
        from src.ingest.request_trace import process

        shard = _write_gz(
            tmp_path / "dynamo-request-trace.000000.jsonl.gz", [_record("a", 1000), '{"timestamp": 12, "ev']
        )
        out = tmp_path / "out.jsonl"

        assert process([str(shard)], str(out)) == 1

    def test_cli_accepts_multiple_inputs(self, tmp_path):
        from src.ingest.request_trace import main

        shard0 = _write_gz(tmp_path / "dynamo-request-trace.000000.jsonl.gz", [_record("a", 1000)])
        shard1 = _write_gz(tmp_path / "dynamo-request-trace.000001.jsonl.gz", [_record("b", 2000)])
        out = tmp_path / "out.jsonl"

        assert main([str(shard0), str(shard1), str(out)]) == 0
        assert len(_rows(out)) == 2


def _args(request_trace_input: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(request_trace="dynamo", request_trace_input=request_trace_input)


class TestDiscovery:
    def test_discovers_rotated_shards_at_the_log_dir_root(self, tmp_path):
        from src.ingest.ingest import discover_request_trace, run_request_trace

        for i, xid in enumerate(["a", "b", "c"]):
            _write_gz(tmp_path / f"dynamo-request-trace.{i:06d}.jsonl.gz", [_record(xid, 1000 * (i + 1))])
        # Unrelated files that share the prefix must not be picked up.
        (tmp_path / "dynamo-request-trace.lock").write_text("")
        bundle = tmp_path / "bundle"
        bundle.mkdir()

        found = discover_request_trace(tmp_path)
        assert [Path(p).name for p in found] == [
            "dynamo-request-trace.000000.jsonl.gz",
            "dynamo-request-trace.000001.jsonl.gz",
            "dynamo-request-trace.000002.jsonl.gz",
        ]
        assert run_request_trace(_args(), tmp_path, bundle) is True
        assert [r["x_request_id"] for r in _rows(bundle / "request_trace.jsonl")] == ["a", "b", "c"]

    def test_bare_file_and_uncompressed_shard_are_included(self, tmp_path):
        from src.ingest.ingest import discover_request_trace

        (tmp_path / "dynamo-request-trace").write_text(_record("a", 1000) + "\n")
        (tmp_path / "dynamo-request-trace.000000.jsonl").write_text(_record("b", 2000) + "\n")
        _write_gz(tmp_path / "dynamo-request-trace.000001.jsonl.gz", [_record("c", 3000)])

        assert [Path(p).name for p in discover_request_trace(tmp_path)] == [
            "dynamo-request-trace",
            "dynamo-request-trace.000000.jsonl",
            "dynamo-request-trace.000001.jsonl.gz",
        ]

    def test_explicit_input_glob_wins_over_discovery(self, tmp_path):
        from src.ingest.ingest import run_request_trace

        _write_gz(tmp_path / "dynamo-request-trace.000000.jsonl.gz", [_record("ignored", 1000)])
        custom = tmp_path / "custom"
        custom.mkdir()
        _write_gz(custom / "trace.a.jsonl.gz", [_record("x", 1000)])
        _write_gz(custom / "trace.b.jsonl.gz", [_record("y", 2000)])
        bundle = tmp_path / "bundle"
        bundle.mkdir()

        assert run_request_trace(_args("custom/trace.*.jsonl.gz"), tmp_path, bundle) is True
        assert [r["x_request_id"] for r in _rows(bundle / "request_trace.jsonl")] == ["x", "y"]

    def test_nothing_found_is_a_skip_not_an_error(self, tmp_path, caplog):
        import logging

        from src.ingest.ingest import run_request_trace

        bundle = tmp_path / "bundle"
        bundle.mkdir()

        with caplog.at_level(logging.INFO, logger="ingest"):
            assert run_request_trace(_args(), tmp_path, bundle) is False
        assert not (bundle / "request_trace.jsonl").exists()
        assert any("no request trace matched" in r.getMessage() for r in caplog.records)

    @pytest.mark.parametrize("mode", ["none"])
    def test_disabled_axis(self, tmp_path, mode):
        from src.ingest.ingest import run_request_trace

        _write_gz(tmp_path / "dynamo-request-trace.000000.jsonl.gz", [_record("a", 1000)])
        bundle = tmp_path / "bundle"
        bundle.mkdir()

        assert (
            run_request_trace(SimpleNamespace(request_trace=mode, request_trace_input=None), tmp_path, bundle) is False
        )
