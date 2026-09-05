# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the AIPerf per-scrape jsonl L2 processor (``src/ingest/metrics_aiperf_jsonl.py``).

The fixture lines reproduce the observed schema of aiperf's own
``server_metrics_export.jsonl`` (requested via ``--server-metrics-formats json jsonl``):
one line per (scrape, endpoint), families keyed by aiperf's base names (``_total``
stripped from counters), each family a LIST of entries -- plain ``{labels?, value}`` or
histogram ``{labels?, buckets, sum, count}`` with cumulative bucket counts.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

T0 = 1_786_194_627_000_000_000
SECOND = 1_000_000_000


def _fixture_lines() -> list[dict]:
    return [
        {
            "timestamp_ns": T0 + 10_000_000,
            "endpoint_url": "http://head:8000/metrics",
            "endpoint_latency_ns": 1_000_000,
            "metrics": {
                # A counter family: aiperf strips _total, so the alias must be re-attached.
                "foo_requests": [{"labels": {"model": "m"}, "value": 5.0}],
                # A gauge that legitimately ends in _total: no double alias.
                "lora_total": [{"value": 7.0}],
                # A histogram family: expands to _bucket/_sum/_count.
                "req_seconds": [
                    {"labels": {"model": "m"}, "buckets": {"0.5": 3.0, "+Inf": 5.0}, "sum": 1.25, "count": 5.0}
                ],
            },
        },
        {
            # Second endpoint scraped milliseconds later: merges into the same second.
            "timestamp_ns": T0 + 30_000_000,
            "endpoint_url": "http://node1:8081/metrics",
            "metrics": {"foo_requests": [{"labels": {"model": "m"}, "value": 2.0}]},
        },
        {
            "timestamp_ns": T0 + SECOND + 20_000_000,
            "endpoint_url": "http://head:8000/metrics",
            "metrics": {"foo_requests": [{"labels": {"model": "m"}, "value": 6.0}]},
        },
    ]


def _write_jsonl(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(line) + "\n" for line in _fixture_lines()))
    return path


def _process(tmp_path: Path) -> list[dict]:
    from src.ingest.metrics_aiperf_jsonl import process

    raw = _write_jsonl(tmp_path / "server_metrics_export.jsonl")
    out = tmp_path / "out.jsonl"
    n = process(str(raw), str(out))
    lines = [json.loads(x) for x in out.read_text().splitlines() if x.strip()]
    assert n == len(lines)
    return lines


class TestAiperfJsonlProcessor:
    def test_per_second_grouping_sorted(self, tmp_path: Path):
        lines = _process(tmp_path)
        assert [ln["timestamp_ns"] for ln in lines] == [T0, T0 + SECOND]

    def test_endpoints_sharing_a_second_merge(self, tmp_path: Path):
        lines = _process(tmp_path)
        values = sorted(e["value"] for e in lines[0]["metrics"]["foo_requests"])
        assert values == [2.0, 5.0], "both endpoints' readings land on the one line"
        assert lines[1]["metrics"]["foo_requests"][0]["value"] == 6.0

    def test_total_alias_is_reattached(self, tmp_path: Path):
        """Counters arrive with _total stripped; the alias makes a panel naming
        either form resolve (metrics_aiperf_json's interchange contract)."""
        lines = _process(tmp_path)
        m = lines[0]["metrics"]
        assert m["foo_requests_total"] == m["foo_requests"]
        # A family already ending in _total is not double-aliased.
        assert m["lora_total"][0]["value"] == 7.0
        assert "lora_total_total" not in m

    def test_histogram_expands_to_exposition_components(self, tmp_path: Path):
        lines = _process(tmp_path)
        m = lines[0]["metrics"]
        buckets = {e["labels"]["le"]: e["value"] for e in m["req_seconds_bucket"]}
        assert buckets == {"0.5": 3.0, "+Inf": 5.0}, "cumulative counts pass through verbatim"
        assert m["req_seconds_sum"] == [{"labels": {"model": "m"}, "value": 1.25}]
        assert m["req_seconds_count"] == [{"labels": {"model": "m"}, "value": 5.0}]
        # The base family must not also appear as a plain (or _total-aliased) series.
        assert "req_seconds" not in m
        assert "req_seconds_total" not in m

    def test_no_worker_labels_are_invented(self, tmp_path: Path):
        """Roles are unknowable from the export alone (mirrors metrics_aiperf_json)."""
        lines = _process(tmp_path)
        for line in lines:
            for entries in line["metrics"].values():
                for entry in entries:
                    assert "worker_id" not in entry["labels"]
                    assert "dynamo_component" not in entry["labels"]


class TestMetricsAutoSelection:
    def test_auto_prefers_jsonl_over_aggregate_json(self, tmp_path: Path, caplog):
        """With only AIPerf's own exports present (no tachometer parquet, no
        raw_prometheus.jsonl), auto picks the per-scrape jsonl over the aggregate."""
        from src.ingest.ingest import build_parser, run_metrics

        run_dir = tmp_path / "logs"
        art = run_dir / "artifacts" / "model_workload_20260826_000000"
        _write_jsonl(art / "server_metrics_export.jsonl")
        (art / "server_metrics_export.json").write_text("{}")

        bundle = tmp_path / "bundle"
        bundle.mkdir()
        args = build_parser().parse_args(["--run-dir", str(run_dir)])
        with caplog.at_level(logging.INFO, logger="ingest"):
            assert run_metrics(args, run_dir, bundle) is True
        assert "auto-selected source: aiperf-jsonl" in caplog.text
        out = bundle / "server_metrics_export.jsonl"
        lines = [json.loads(x) for x in out.read_text().splitlines() if x.strip()]
        assert "foo_requests_total" in lines[0]["metrics"], "the jsonl leg was processed"
