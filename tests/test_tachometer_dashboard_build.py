# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Artifact/source-boundary tests for the raw-only dashboard command."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from srtctl.analysis.tachometer_dashboard.__main__ import build_dashboard, decorate_catalog


@pytest.mark.parametrize("filename", ["final.parquet", "out-1.parquet"])
def test_build_uses_raw_capture_and_embeds_offline_assets(tmp_path: Path, filename: str) -> None:
    raw = tmp_path / "capture"
    raw.mkdir()
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "metric_name": 'gpu_util{gpu="0"}',
                    "metric_value": value,
                    "scraper_endpoint": "dcgm_node-a",
                    "hostname": "node-a",
                    "time_since_start": float(t),
                }
                for t, value in [(0, 30.0), (1, 70.0), (2, 0.0)]
            ]
        ),
        raw / filename,
    )
    # These plausible inputs must not be parsed or used as measurements/time anchors.
    for name in ("server_metrics_export.jsonl", "config.yaml", "perf_dashboard.json", "tachometer.out"):
        (raw / name).write_text('"THIS IS NOT A VALID INPUT; external capacity 999999"')
    out = build_dashboard(raw, tmp_path / "dashboard.html", resolution_s=1, title="A <capture>")
    document = out.read_text()
    catalog = json.loads((tmp_path / "dashboard.data" / "catalog.json").read_text())
    assert any("may be incomplete" in warning for warning in catalog["source_warnings"]) == (
        filename != "final.parquet"
    )
    assert catalog["row_count"] == 3
    assert catalog["start_ns"] is None
    assert catalog["duration_s"] == 2
    assert [Path(source["path"]).name for source in catalog["source_files"]] == [filename]
    assert [m["component"] for m in catalog["metrics"]] == ["GPU"]
    assert 'src="http' not in document
    assert 'href="http' not in document
    assert 'id="payload-' in document
    assert "A \\u003ccapture\\u003e" in document
    assert "999999" not in document


def test_histogram_inventory_aliases_do_not_misclassify_go_summary() -> None:
    catalog = {
        "metrics": [
            {"name": "dynamo_frontend_tokenize_seconds", "kind": "histogram"},
            {"name": "go_gc_duration_seconds", "kind": "scalar"},
            {"name": "go_gc_duration_seconds_count", "kind": "scalar"},
        ]
    }
    decorate_catalog(catalog, "Capture")
    assert catalog["metrics"][0]["aliases"] == [
        "dynamo_frontend_tokenize_seconds_bucket",
        "dynamo_frontend_tokenize_seconds_count",
        "dynamo_frontend_tokenize_seconds_sum",
    ]
    assert all(not metric["aliases"] for metric in catalog["metrics"][1:])


def test_processed_jsonl_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "server_metrics_export.jsonl"
    source.write_text("{}\n")
    with pytest.raises(ValueError, match="parquet|arrow"):
        build_dashboard(source, tmp_path / "dashboard.html")
    assert not (tmp_path / "dashboard.html").exists()
