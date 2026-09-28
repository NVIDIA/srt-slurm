# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Find a recorded time envelope when there is no client measurement window."""

from __future__ import annotations

import datetime as dt
import json
import sqlite3
from typing import TYPE_CHECKING

import pyarrow.compute as pc

from .engines import parse_engine_log
from .metrics import batches, capture_files
from .sources import log_fields, otel_files

if TYPE_CHECKING:
    from .importer import Importer


def source_window(run: Importer) -> tuple[int, int]:
    """Read only timestamps; never invent a client request or a run duration."""
    lower: int | None = None
    upper: int | None = None

    def include(start: int | None, end: int | None) -> None:
        nonlocal lower, upper
        if start is None or end is None or start <= 0 or end < start:
            return
        lower = start if lower is None else min(lower, start)
        upper = end if upper is None else max(upper, end)

    root = run.metrics_path or run.logs / "tachometer/local"
    if run.metrics_path and not root.exists():
        raise ValueError(f"Raw metrics path does not exist: {root}")
    for path in capture_files(root):
        run.source(path, "tachometer_raw")
        for batch in batches(path):
            if "timestamp_ns" not in batch.schema.names:
                raise ValueError(f"{path}: raw metrics require timestamp_ns for UTC alignment")
            bounds = pc.call_function("min_max", [batch["timestamp_ns"]]).as_py()
            include(bounds["min"], bounds["max"])
    if run.otel:
        for path in otel_files(run.logs):
            if not path.stat().st_size:
                continue
            run.source(path, "otel")
            for number, line in enumerate(path.open(), 1):
                try:
                    doc = json.loads(line)
                    for resource in doc.get("resourceSpans", []):
                        for scope in resource.get("scopeSpans", []):
                            for span in scope.get("spans", []):
                                include(int(span["startTimeUnixNano"]), int(span["endTimeUnixNano"]))
                except (KeyError, ValueError, TypeError) as exc:
                    raise ValueError(f"{path}:{number}: invalid OTel record: {exc}") from exc
    if run.sqlites:
        if not run.sqlites.exists():
            raise ValueError(f"Nsight SQLite path does not exist: {run.sqlites}")
        paths = [run.sqlites] if run.sqlites.is_file() else sorted(run.sqlites.rglob("*.sqlite"))
        for path in paths:
            run.source(path, "nsight_sqlite")
            with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as conn:
                tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                if {"TARGET_INFO_SESSION_START_TIME", "ANALYSIS_DETAILS"}.issubset(tables):
                    anchor = conn.execute(
                        "SELECT utcEpochNs,systemClockNs FROM TARGET_INFO_SESSION_START_TIME LIMIT 1"
                    ).fetchone()
                    capture = conn.execute("SELECT startTime,stopTime FROM ANALYSIS_DETAILS LIMIT 1").fetchone()
                    if anchor and capture:
                        include(anchor[0] + capture[0] - anchor[1], anchor[0] + capture[1] - anchor[1])
    # Union all sources; a shorter metrics capture must not hide later log evidence.
    for path in sorted(run.logs.glob("*_w*.out")):
        for line in path.open(errors="replace"):
            row = parse_engine_log(line)
            if row and row["kind"] in ("iteration", "batch_snapshot") and run.iteration_zone:
                time = dt.datetime.fromisoformat(row["local_time"]).replace(tzinfo=run.iteration_zone)
                start = int(time.timestamp() * 1e9)
                include(start, start + int(row["time_resolution_s"] * 1e9))
            elif fields := log_fields(line):
                stamp = fields.get("time") or fields.get("timestamp")
                if isinstance(stamp, str):
                    time = dt.datetime.fromisoformat(stamp.replace("Z", "+00:00"))
                    if time.tzinfo:
                        start = int(time.timestamp() * 1e9)
                        include(start, start)
    if lower is None or upper is None or upper <= lower:
        raise ValueError(
            "No positive recorded time range: supply client, OTel, Tachometer, Nsight, or timestamped worker evidence"
        )
    return lower, upper
