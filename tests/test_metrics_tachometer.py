# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the tachometer-parquet L2 processor (``src/ingest/metrics_tachometer.py``).

The fixture parquet reproduces what the Rust writer actually emits (tachometer-writer
``writer.rs`` schema; ``parse.rs`` + the backend/frontend filters' naming): base family
names with inline ``{labels}`` (``le`` kept inline on bucket rows), cumulative bucket
counts in ``metric_value``, ``histogram_sum``/``histogram_count`` attached to finite
buckets but MISSING on the ``+Inf`` bucket (the writer's fix-up never groups it on
filtered endpoints), and per-endpoint metadata columns padded with "".
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

# A fixed wall-epoch anchor on a whole second, so the 1 s snapping is easy to reason about.
T0 = 1_786_194_627_000_000_000
SECOND = 1_000_000_000

_META_COLUMNS = ("frontend_index", "hostname", "job_id", "run_name", "worker_index", "worker_process", "worker_role")


def _write_parquet(path: Path, rows: list[dict]) -> Path:
    """Build a parquet file with the exact column layout of tachometer-writer's Row."""
    cols = {
        "scraper_endpoint": pa.array([r["endpoint"] for r in rows], pa.string()),
        "metric_name": pa.array([r["name"] for r in rows], pa.string()),
        "metric_value": pa.array([r["value"] for r in rows], pa.float64()),
        "histogram_bucket_lower": pa.array([r.get("lower") for r in rows], pa.float64()),
        "histogram_bucket_upper": pa.array([r.get("upper") for r in rows], pa.float64()),
        "histogram_sum": pa.array([r.get("sum") for r in rows], pa.float64()),
        "histogram_count": pa.array([r.get("count") for r in rows], pa.float64()),
        "time_since_start": pa.array([(r["ts"] - T0) / 1e9 for r in rows], pa.float64()),
        "timestamp_ns": pa.array([r["ts"] for r in rows], pa.int64()),
    }
    for meta in _META_COLUMNS:
        cols[meta] = pa.array([r.get(meta, "") for r in rows], pa.string())
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(cols), path)
    return path


def _backend(role: str, host: str, **kw) -> dict:
    return {
        "endpoint": f"backend_{role}0_rank0",
        "hostname": host,
        "worker_index": "0",
        "worker_process": "0",
        "worker_role": role,
        "job_id": "12345",
        "run_name": "testrun",
        **kw,
    }


def _fixture_rows() -> list[dict]:
    return [
        # backend prefill worker: a plain gauge (f64-fidelity probe) + one histogram.
        _backend(
            "prefill",
            "node1",
            name='trtllm_kv_cache_used_blocks{model_name="m"}',
            value=16777217.0,
            ts=T0 + 100_000_000,
        ),
        _backend(
            "prefill",
            "node1",
            name='req_latency_seconds{le="0.5",model="m"}',
            value=3.0,
            lower=0.0,
            upper=0.5,
            sum=1.25,
            count=5.0,
            ts=T0 + 120_000_000,
        ),
        _backend(
            "prefill",
            "node1",
            name='req_latency_seconds{le="1",model="m"}',
            value=4.0,
            lower=0.5,
            upper=1.0,
            sum=1.25,
            count=5.0,
            ts=T0 + 120_000_000,
        ),
        # The +Inf bucket: upper/lower/sum/count all null, exactly as the writer's
        # fix-up leaves it on backend/frontend-filtered endpoints.
        _backend("prefill", "node1", name='req_latency_seconds{le="+Inf",model="m"}', value=5.0, ts=T0 + 120_000_000),
        # backend decode worker on another node: decode maps to dynamo_component=backend.
        _backend(
            "decode", "node2", name='trtllm_kv_cache_used_blocks{model_name="m"}', value=200.0, ts=T0 + 130_000_000
        ),
        # frontend endpoint, two scrapes one second apart (per-second grouping probe).
        {
            "endpoint": "frontend0",
            "frontend_index": "0",
            "hostname": "node0",
            "job_id": "12345",
            "run_name": "testrun",
            "name": 'dynamo_frontend_requests_total{model="m"}',
            "value": 42.0,
            "ts": T0 + 200_000_000,
        },
        {
            "endpoint": "frontend0",
            "frontend_index": "0",
            "hostname": "node0",
            "job_id": "12345",
            "run_name": "testrun",
            "name": 'dynamo_frontend_requests_total{model="m"}',
            "value": 43.0,
            "ts": T0 + SECOND + 150_000_000,
        },
    ]


@pytest.fixture
def log_dir(tmp_path: Path) -> Path:
    """A miniature run log dir whose tachometer leg holds a compacted final.parquet."""
    d = tmp_path / "logs"
    _write_parquet(d / "tachometer" / "raw" / "scrape" / "final.parquet", _fixture_rows())
    return d


def _process(log_dir: Path, tmp_path: Path) -> list[dict]:
    from src.ingest.metrics_tachometer import process

    out = tmp_path / "server_metrics_export.jsonl"
    n = process(str(log_dir), str(out))
    lines = [json.loads(x) for x in out.read_text().splitlines() if x.strip()]
    assert n == len(lines)
    return lines


class TestTachometerProcessor:
    def test_per_second_grouping_sorted(self, log_dir: Path, tmp_path: Path):
        lines = _process(log_dir, tmp_path)
        assert [ln["timestamp_ns"] for ln in lines] == [T0, T0 + SECOND]
        # Both scrapes of the frontend family land, one per second.
        assert lines[0]["metrics"]["dynamo_frontend_requests_total"][0]["value"] == 42.0
        assert lines[1]["metrics"]["dynamo_frontend_requests_total"][0]["value"] == 43.0

    def test_inline_labels_are_parsed(self, log_dir: Path, tmp_path: Path):
        lines = _process(log_dir, tmp_path)
        entries = lines[0]["metrics"]["trtllm_kv_cache_used_blocks"]
        assert all(e["labels"]["model_name"] == "m" for e in entries)
        # The label block must not leak into the metric name.
        assert all("{" not in name for name in lines[0]["metrics"])

    def test_worker_labels_are_injected(self, log_dir: Path, tmp_path: Path):
        """Mirrors metrics_prometheus: prefill -> "prefill", decode -> "backend",
        worker_id from the hostname metadata, injected via setdefault."""
        lines = _process(log_dir, tmp_path)
        entries = lines[0]["metrics"]["trtllm_kv_cache_used_blocks"]
        by_worker = {e["labels"]["worker_id"]: e["labels"]["dynamo_component"] for e in entries}
        assert by_worker == {"node1": "prefill", "node2": "backend"}

    def test_frontend_series_get_no_worker_labels(self, log_dir: Path, tmp_path: Path):
        lines = _process(log_dir, tmp_path)
        for line in lines:
            for entry in line["metrics"]["dynamo_frontend_requests_total"]:
                assert "worker_id" not in entry["labels"]
                assert "dynamo_component" not in entry["labels"]
                # Real capture metadata IS carried (it distinguishes replicas).
                assert entry["labels"]["frontend_index"] == "0"
                assert entry["labels"]["hostname"] == "node0"

    def test_histogram_reconstruction(self, log_dir: Path, tmp_path: Path):
        """Bucket rows -> <fam>_bucket with le (cumulative, verbatim); _sum/_count are
        rebuilt from the histogram_sum/histogram_count columns, with _count taken from
        the +Inf bucket's own value (the writer leaves +Inf rows' columns null)."""
        lines = _process(log_dir, tmp_path)
        m = lines[0]["metrics"]
        buckets = {e["labels"]["le"]: e["value"] for e in m["req_latency_seconds_bucket"]}
        assert buckets == {"0.5": 3.0, "1": 4.0, "+Inf": 5.0}
        (sum_entry,) = m["req_latency_seconds_sum"]
        assert sum_entry["value"] == 1.25
        (count_entry,) = m["req_latency_seconds_count"]
        assert count_entry["value"] == 5.0
        for entry in (sum_entry, count_entry):
            assert "le" not in entry["labels"]
            assert entry["labels"]["model"] == "m"
            assert entry["labels"]["dynamo_component"] == "prefill"
        # The base family must not also appear as a plain series.
        assert "req_latency_seconds" not in m

    def test_f64_values_survive(self, log_dir: Path, tmp_path: Path):
        """16777217 is not representable in f32; it must round-trip exactly."""
        lines = _process(log_dir, tmp_path)
        entries = lines[0]["metrics"]["trtllm_kv_cache_used_blocks"]
        assert any(e["value"] == 16777217.0 for e in entries)

    def test_final_parquet_is_preferred_over_leftovers(self, log_dir: Path, tmp_path: Path):
        """local/ leftovers only matter when compaction never produced final.parquet."""
        from src.ingest.metrics_tachometer import find_parquets

        leftover = log_dir / "tachometer" / "local" / "out-1.parquet"
        _write_parquet(leftover, _fixture_rows()[:1])
        assert find_parquets(str(log_dir)) == [str(log_dir / "tachometer" / "raw" / "scrape" / "final.parquet")]
        (log_dir / "tachometer" / "raw" / "scrape" / "final.parquet").unlink()
        assert find_parquets(str(log_dir)) == [str(leftover)]


class TestMetricsAutoSelection:
    def test_auto_prefers_tachometer_over_everything(self, log_dir: Path, tmp_path: Path, caplog):
        """A run dir with the tachometer parquet AND raw_prometheus.jsonl AND both
        AIPerf exports must pick the tachometer leg (whole-window, per-replica)."""
        from src.ingest.ingest import build_parser, run_metrics

        (log_dir / "raw_prometheus.jsonl").write_text(
            json.dumps(
                {
                    "timestamp_ns": T0,
                    "endpoint_url": "http://head:8000/metrics",
                    "role": "frontend",
                    "worker_id": None,
                    "text": "a 1\n",
                }
            )
            + "\n"
        )
        art = log_dir / "artifacts" / "model_workload_20260826_000000"
        art.mkdir(parents=True)
        (art / "server_metrics_export.json").write_text("{}")
        (art / "server_metrics_export.jsonl").write_text(
            json.dumps(
                {"timestamp_ns": T0, "endpoint_url": "http://head:8000/metrics", "metrics": {"foo": [{"value": 1.0}]}}
            )
            + "\n"
        )

        bundle = tmp_path / "bundle"
        bundle.mkdir()
        args = build_parser().parse_args(["--run-dir", str(log_dir)])
        with caplog.at_level(logging.INFO, logger="ingest"):
            assert run_metrics(args, log_dir, bundle) is True
        assert "auto-selected source: tachometer" in caplog.text
        out = bundle / "server_metrics_export.jsonl"
        lines = [json.loads(x) for x in out.read_text().splitlines() if x.strip()]
        # Tachometer content, not the decoy sources.
        assert "trtllm_kv_cache_used_blocks" in lines[0]["metrics"]
