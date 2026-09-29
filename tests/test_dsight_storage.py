# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare indexed and static delivery against the independent in-memory queries."""

import copy
import gzip
import json
import sqlite3

import pytest
from test_dsight import CLIENT, write_run

from srtctl.dsight.build import build_dashboard, write_dashboard
from srtctl.dsight.bundle import MAX_CHUNK_BYTES, write_details
from srtctl.dsight.importer import Importer
from srtctl.dsight.query import KINDS, TraceDataset
from srtctl.dsight.storage import FILENAME, write_store


@pytest.fixture
def dataset(tmp_path):
    logs, sqlites = write_run(tmp_path)
    data = Importer(logs, sqlites, iteration_timezone="UTC").run()
    p = data["profiles"][0]
    p["names"].append("Ünicode nested")
    p["name_definitions"].append(None)
    # Long roots, exact endpoints, instantaneous ranges, source-order ties,
    # negative relative starts and short ranges near a long root's end.
    p["events"] = [
        [-1.0, 10.0, 0, "17", 1],
        [0.0, 10.0, 0, "17", 2],
        [2.0, 3.0, 1, "17", 3],
        [3.0, 3.0, 2, "17", 4],
        [3.0, 4.0, 0, "17", 5],
        [9.1, 9.100000001, 1, "17", 6],
    ]
    p["cpu"] = {
        "pid": 0,
        "names": ["a", "b"],
        "stacks": [[0, 0, 1], [1]],
        "samples": [[1, "17", 0, 1], [3, "17", 1, 2]],
        "attribution": "samples",
    }
    s = data["metrics"][0]
    s["temporal"] = "setting"
    s["points"] = [[1, 2, 0, 1], [1, 3, 0, 2], [3, None, 0, 3], [4, 5, 0, 4]]
    path = tmp_path / FILENAME
    write_store(data, path)
    return data, TraceDataset.from_path(path)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize(
    "options",
    [
        {},
        {"start": 3, "end": 4},
        {"offset": 1, "limit": 1},
        {"limit": 0},
        {"start": 9.100000001, "end": 9.2},
        {"worker": "decode-0", "rank": 0},
    ],
)
def test_indexed_query_matches_existing_contract(dataset, kind, options):
    data, indexed = dataset
    options = {**options, "request_id": CLIENT, "points": True}
    assert indexed.query(kind, **options) == TraceDataset(data).query(kind, **options)


def test_unicode_filter_and_invalid_queries(dataset):
    data, indexed = dataset
    assert indexed.query("nsys", name="ÜNICODE") == TraceDataset(data).query("nsys", name="ÜNICODE")
    for options in ({"start": float("nan")}, {"limit": 1001}, {"offset": -1}, {"start": 4, "end": 3}):
        with pytest.raises(ValueError):
            indexed.query("nsys", **options)
    assert indexed.query("nsys", worker="missing")["total"] == 0


def test_rounded_duration_does_not_exclude_negative_start(dataset, tmp_path):
    data, _ = dataset
    data["meta"]["duration"] = 11
    data["profiles"][0]["events"] = [[-1e-30, 10, 0, "17", 1]]
    path = tmp_path / "rounded.sqlite"
    write_store(data, path)
    assert TraceDataset.from_path(path).query("nsys", start=10, end=11)["total"] == 1


def test_static_chunks_reconstruct_exact_data_and_crossing_windows(dataset, tmp_path):
    data, _ = dataset
    p = data["profiles"][0]
    p["events"] += [[i / 2000, i / 2000 + 0.001, 0, "17", 100 + i] for i in range(16000)]
    original = copy.deepcopy(data)
    folder = tmp_path / "detail"
    core = write_details(data, folder)
    assert data == original
    assert core["profiles"][0]["events"] == []
    assert "points" not in core["metrics"][0]
    restored = []
    window = []
    for c in core["profiles"][0]["event_chunks"]:
        raw = gzip.decompress((tmp_path / c["url"]).read_bytes())
        assert len(raw) <= MAX_CHUNK_BYTES
        rows = json.loads(raw)
        restored.extend(rows)
        if c["bounds"][0] <= 9.2 and c["bounds"][3] >= 9.1:
            window.extend(e for e in rows if e[0] <= 9.2 and e[1] >= 9.1)
    assert restored == p["events"]
    assert window == [e for e in p["events"] if e[0] <= 9.2 and e[1] >= 9.1]
    assert [e[4] for e in window] == [1, 2, 6]


def test_legacy_and_progressive_builds_and_failed_detail_generation(tmp_path, monkeypatch):
    logs, sqlites = write_run(tmp_path)
    out = tmp_path / "report"
    legacy = build_dashboard(logs, out, sqlites=sqlites, single_file=True)
    old = TraceDataset.from_path(legacy["data"]).query("nsys")
    build_dashboard(logs, out, sqlites=sqlites)
    assert TraceDataset.from_path(out).query("nsys") == old
    html = (out / "index.html").read_bytes()
    assert b'metricPayload0"' not in html
    assert (out / FILENAME).is_file()
    assert not (out / "trace-data.json.gz").exists()
    from srtctl.dsight import bundle

    monkeypatch.setattr(bundle, "write_details", lambda *a: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError, match="disk full"):
        build_dashboard(logs, out)
    assert (out / "index.html").read_bytes() == html
    assert TraceDataset.from_path(out).query("nsys") == old
    (out / "detail" / "user-note.txt").write_text("keep")
    with pytest.raises(ValueError, match="user-note"):
        build_dashboard(logs, out)


def test_sqlite_version_and_read_only_queries(dataset, tmp_path):
    _, indexed = dataset
    path = indexed.store.path
    before = path.stat()
    for kind in ("nsys", "metrics", "cpu"):
        indexed.query(kind)
    assert path.stat().st_mtime_ns == before.st_mtime_ns
    with sqlite3.connect(path) as conn:
        conn.execute("PRAGMA user_version=999")
    with pytest.raises(ValueError, match="version"):
        TraceDataset.from_path(path)


def test_legacy_capabilities_survive_moving_detail_out_of_catalog(dataset, tmp_path):
    data, _ = dataset
    data.pop("capabilities", None)
    data.pop("metric_catalog", None)
    report = tmp_path / "legacy-repacked"
    write_dashboard(data, report)
    assert TraceDataset.from_path(report).query("summary") == TraceDataset(data).query("summary")


def test_replaced_generation_requires_reopening(dataset, tmp_path):
    data, indexed = dataset
    replacement = tmp_path / "replacement.sqlite"
    write_store(data, replacement)
    replacement.replace(indexed.store.path)
    with pytest.raises(ValueError, match="rebuilt; reopen"):
        indexed.query("nsys")
    assert TraceDataset.from_path(indexed.store.path).query("nsys") == TraceDataset(data).query("nsys")


def test_details_do_not_inflate_initial_html(dataset, tmp_path):
    data, _ = dataset
    small = write_dashboard(data, tmp_path / "small")
    data["profiles"][0]["events"] *= 10000
    large = write_dashboard(data, tmp_path / "large")
    from pathlib import Path

    # Six input rows become 60,000, but the initial document grows only by the
    # compressed shard catalog and density bins, not the event payload.
    assert Path(large["html"]).stat().st_size < Path(small["html"]).stat().st_size + 20000
