# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the installed raw UI subprocess and exercise truthful publication failures."""

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from srtctl.analysis.tachometer_dashboard import pipeline


def config(*, enabled=True, subdir="tachometer"):
    return SimpleNamespace(
        name="raw run",
        observability=SimpleNamespace(
            enabled=False,
            tachometer_enabled=enabled,
            tachometer=SimpleNamespace(enabled=enabled, storage_subdir=subdir),
        ),
    )


def raw_capture(path: Path, value=70.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "metric_name": 'DCGM_FI_DEV_GPU_UTIL{gpu="0"}',
                    "metric_value": value,
                    "scraper_endpoint": "dcgm_node-a",
                    "hostname": "node-a",
                    "time_since_start": float(t),
                }
                for t in range(3)
            ]
        ),
        path,
    )


def status(path):
    return json.loads((path / pipeline.STATUS_FILENAME).read_text())


def test_actual_subprocess_uses_only_local_raw_leaf_and_custom_storage(tmp_path):
    cfg = config(subdir="custom capture")
    source = tmp_path / "custom capture" / "local" / "final.parquet"
    raw_capture(source)
    # A conflicting upload mirror and unrelated inputs must never supply values.
    raw_capture(tmp_path / "custom capture" / "raw" / "scrape" / "final.parquet", value=999999.0)
    for name in ["perf_dashboard.json", "raw_prometheus.jsonl", "benchmark.out", "config.yaml"]:
        (tmp_path / name).write_text("unparseable unrelated data 999999")
    result = pipeline.try_build(cfg, SimpleNamespace(log_dir=tmp_path, job_id="123"))
    assert result == tmp_path / "dashboard.html"
    assert "999999" not in result.read_text()
    record = status(tmp_path)
    assert record["state"] == "ready"
    assert record["row_count"] == 3
    assert record["metric_families"] == 1
    assert [Path(item["path"]) for item in record["source_files"]] == [source]
    assert "not certify" in record["note"]
    assert (tmp_path / "dashboard.data" / "catalog.json").is_file()
    assert not list(tmp_path.glob(".dashboard-build-*"))


def test_successful_rebuild_replaces_data_and_html_together(tmp_path):
    source = tmp_path / "tachometer" / "local" / "final.parquet"
    raw_capture(source)
    runtime = SimpleNamespace(log_dir=tmp_path, job_id="123")
    assert pipeline.try_build(config(), runtime)
    (tmp_path / "dashboard.data" / "obsolete.json").write_text("old payload")
    raw_capture(source, value=40.0)
    assert pipeline.try_build(config(), runtime)
    assert not (tmp_path / "dashboard.data" / "obsolete.json").exists()
    assert status(tmp_path)["state"] == "ready"


@pytest.mark.parametrize("enabled,expected", [(False, "disabled"), (True, "missing")])
def test_no_capture_is_explicit_and_does_not_use_client_data(tmp_path, monkeypatch, enabled, expected):
    (tmp_path / "server_metrics_export.jsonl").write_text("{}")
    monkeypatch.setattr(
        pipeline.subprocess, "run", lambda *_a, **_k: pytest.fail("No raw capture; must not run a builder")
    )
    assert pipeline.try_build(config(enabled=enabled), SimpleNamespace(log_dir=tmp_path, job_id="123")) is None
    assert status(tmp_path)["state"] == expected
    assert not (tmp_path / "dashboard.html").exists()


def test_invalid_raw_file_is_reported_without_legacy_fallback(tmp_path):
    source = tmp_path / "tachometer" / "local" / "final.parquet"
    source.parent.mkdir(parents=True)
    source.write_text("not a parquet file")
    assert pipeline.try_build(config(), SimpleNamespace(log_dir=tmp_path, job_id="123")) is None
    assert status(tmp_path)["state"] == "failed"
    assert (tmp_path / pipeline.BUILD_LOG_FILENAME).stat().st_size > 0
    assert not (tmp_path / "dashboard.html").exists()


@pytest.mark.parametrize("failure", ["timeout", "launch", "nonzero", "missing", "empty", "bad_catalog"])
def test_failed_build_preserves_previous_artifact_and_never_reports_ready(tmp_path, monkeypatch, failure):
    raw_capture(tmp_path / "tachometer" / "local" / "final.parquet")
    previous = tmp_path / "dashboard.html"
    previous.write_text("previous completed dashboard")
    old_data = tmp_path / "dashboard.data"
    old_data.mkdir()
    (old_data / "catalog.json").write_text("previous catalog")

    def run(command, **kwargs):
        assert command[0] == pipeline.sys.executable
        assert kwargs["timeout"] == pipeline.BUILD_TIMEOUT_SECONDS
        assert status(tmp_path)["state"] == "building"
        output = Path(command[command.index("--out") + 1])
        if failure == "timeout":
            output.write_text("partial HTML")
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        if failure == "launch":
            raise OSError("cannot start builder")
        if failure == "nonzero":
            output.write_text("partial HTML")
            return SimpleNamespace(returncode=1)
        if failure == "empty":
            output.touch()
        if failure == "bad_catalog":
            output.write_text("HTML exists")
            output.with_suffix(".data").mkdir()
            (output.with_suffix(".data") / "catalog.json").write_text("{}")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(pipeline.subprocess, "run", run)
    assert pipeline.try_build(config(), SimpleNamespace(log_dir=tmp_path, job_id="123")) is None
    assert previous.read_text() == "previous completed dashboard"
    assert (old_data / "catalog.json").read_text() == "previous catalog"
    assert status(tmp_path)["state"] == "failed"
    assert status(tmp_path)["previous_artifact"] is True
    assert not list(tmp_path.glob(".dashboard-build-*"))
