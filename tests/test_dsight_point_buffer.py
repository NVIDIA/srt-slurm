# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Disk-backed import must preserve normalized evidence and browser bytes."""

import gzip
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from srtctl.dsight.bundle import _chunks
from srtctl.dsight.importer import Importer
from srtctl.dsight.metrics import read_metrics
from srtctl.dsight.point_buffer import PointBuffer, PointBuffers
from srtctl.dsight.storage import dumps

ORIGIN = 1_780_000_000_000_000_000


@pytest.mark.parametrize("integer_values", [False, True])
def test_disk_import_preserves_points_audit_catalog_and_shards(tmp_path, integer_values):
    root = tmp_path / "tachometer/local"
    root.mkdir(parents=True)
    rows = []
    # Out-of-order batches, duplicate observations, conflicting values, two
    # identities, out-of-window samples, plus an Arrow tail.
    for time, value, name in [(2, 4, "a"), (1, 3, "a"), (1, 3, "a"), (1, 5, "a"), (3, 0, "b"), (11, 8, "a")]:
        rows.append(
            {
                "timestamp_ns": ORIGIN + time * 10**9,
                "metric_name": name,
                "metric_value": value if integer_values else float(value),
                "scraper_endpoint": "worker",
            }
        )
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, root / "final.parquet", row_group_size=2)
    with (root / "tail.arrow").open("wb") as stream, pa.ipc.new_stream(stream, table.schema) as writer:
        writer.write_table(table.slice(1, 3))
    memory = Importer(tmp_path)
    memory.origin, memory.duration = ORIGIN, 10
    expected = read_metrics(memory)
    buffers = PointBuffers(tmp_path / "scratch.sqlite")
    try:
        disk = Importer(tmp_path, point_buffers=buffers)
        disk.origin, disk.duration = ORIGIN, 10
        actual = read_metrics(disk)
        assert all(isinstance(s["points"], PointBuffer) for s in actual)
        assert disk.audit == memory.audit
        assert disk.metric_catalog == memory.metric_catalog
        normalized = [{**s, "points": list(s["points"])} for s in actual]
        assert dumps(normalized) == dumps(expected)
        # Both consumers may independently iterate the retained points.
        for index, (a, b) in enumerate(zip(actual, expected, strict=True)):
            left, right = tmp_path / f"left{index}", tmp_path / f"right{index}"
            left.mkdir()
            right.mkdir()
            assert _chunks(a["points"], left) == _chunks(b["points"], right)
            for shard in left.iterdir():
                assert shard.read_bytes() == (right / shard.name).read_bytes()
                assert json.loads(gzip.decompress(shard.read_bytes())) == b["points"]
        assert not buffers.conn.execute("SELECT name FROM sqlite_master WHERE name='raw'").fetchall()
    finally:
        buffers.close()


@pytest.mark.parametrize("failure_stage", ["import", "publish", None])
def test_progressive_build_removes_metric_scratch_on_success_and_failure(tmp_path, monkeypatch, failure_stage):
    from test_dsight import write_run

    from srtctl.dsight import build
    from srtctl.dsight.query import TraceDataset

    logs, sqlites = write_run(tmp_path)
    out = tmp_path / "report"
    original = Importer(logs, sqlites, iteration_timezone="UTC").run()

    def fail(*args, **kwargs):
        raise OSError("simulated failure")

    if failure_stage == "import":
        monkeypatch.setattr(Importer, "metrics", fail)
    elif failure_stage == "publish":
        monkeypatch.setattr(build, "write_store", fail)
    if failure_stage:
        with pytest.raises(OSError, match="simulated failure"):
            build.build_dashboard(logs, out, sqlites=sqlites, iteration_timezone="UTC")
        assert not out.exists()
    else:
        build.build_dashboard(logs, out, sqlites=sqlites, iteration_timezone="UTC")
        disk = TraceDataset.from_path(out)
        memory = TraceDataset(original)
        for kind in ("summary", "metrics", "requests", "nsys", "cpu"):
            assert disk.query(kind, points=True) == memory.query(kind, points=True)
    assert not list(tmp_path.glob(".trace-metrics-*"))


def test_point_buffer_preserves_numeric_types_and_earliest_evidence(tmp_path):
    buffers = PointBuffers(tmp_path / "scratch.sqlite")
    try:
        points = buffers.buffer(0)
        points.extend([[1.0, 0.0, 2, 9], [1.0, -0.0, 1, 8], [2.0, 2**64 - 1, 0, 7]])
        buffers.prepare()
        assert points.finalize() == (1, [], 0)
        expected = [[1.0, -0.0, 1, 8], [2.0, 2**64 - 1, 0, 7]]
        assert dumps(list(points)) == dumps(expected)
    finally:
        buffers.close()


def test_point_buffer_does_not_retain_a_python_object_per_sample(tmp_path):
    import tracemalloc

    buffers = PointBuffers(tmp_path / "scratch.sqlite")
    try:
        points = buffers.buffer(0)
        tracemalloc.start()
        try:
            points.extend([float(i), float(i), 0, i] for i in range(250_000))
            buffers.prepare()
            assert points.finalize() == (0, [], 0)
            _, peak = tracemalloc.get_traced_memory()
            assert peak < 4 * 1024 * 1024
            assert len(points) == 250_000
        finally:
            tracemalloc.stop()
    finally:
        buffers.close()


def test_final_timestamp_conflicts_keep_all_values_and_count_duplicates(tmp_path):
    buffers = PointBuffers(tmp_path / "scratch.sqlite")
    try:
        points = buffers.buffer(0)
        points.extend([[2.0, 8.0, 1, 2], [1.0, 3.0, 0, 0], [2.0, 7.0, 0, 1], [2.0, 8.0, 2, 3]])
        buffers.prepare()
        assert points.finalize() == (1, [2.0], 2)
        assert list(points) == [[1.0, 3.0, 0, 0], [2.0, 7.0, 0, 1], [2.0, 8.0, 1, 2]]
    finally:
        buffers.close()
