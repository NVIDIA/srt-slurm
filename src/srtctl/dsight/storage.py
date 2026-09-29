# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Versioned, read-only query cache of normalized evidence (never raw sources).

Large arrays live in indexed tables. The catalog retains the existing identity,
clock, coverage and lifecycle contract. Relative seconds are stored unchanged;
this cache does not claim to recover raw timestamps absent from an old import.
"""

from __future__ import annotations

import json
import math
import sqlite3
import uuid
from contextlib import closing
from pathlib import Path
from typing import Any

VERSION = 1
FILENAME = "trace-data.sqlite"


def dumps(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def catalog(data: dict[str, Any]) -> dict[str, Any]:
    profiles = []
    for profile in data["profiles"]:
        item = {**profile, "events": [], "event_count": len(profile["events"])}
        if profile.get("cpu"):
            cpu = profile["cpu"]
            item["cpu"] = {**cpu, "samples": [], "sample_count": len(cpu["samples"])}
        profiles.append(item)
    return {
        **data,
        "profiles": profiles,
        "metrics": [{**s, "points": []} for s in data["metrics"]],
    }


def duration_bucket(start: float, end: float) -> int:
    # Isolate long intervals so they don't extend every short-range index scan.
    return math.ceil(math.log2(max(end - start, 1e-9)))


def write_store(data: dict[str, Any], path: Path) -> None:
    """Write a new store inside the caller's unpublished staging directory."""
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.executescript("""
            CREATE TABLE catalog (json TEXT NOT NULL, generation TEXT NOT NULL);
            CREATE TABLE events (
                profile INTEGER, ordinal INTEGER, bucket INTEGER, start REAL, end REAL,
                name INTEGER, tid TEXT, source_row INTEGER,
                PRIMARY KEY (profile, ordinal)
            ) WITHOUT ROWID;
            CREATE TABLE durations (profile INTEGER, bucket INTEGER, maximum REAL,
                PRIMARY KEY (profile, bucket)) WITHOUT ROWID;
            CREATE TABLE samples (profile INTEGER, ordinal INTEGER, time REAL, payload TEXT,
                PRIMARY KEY (profile, ordinal)) WITHOUT ROWID;
            CREATE TABLE points (series INTEGER, ordinal INTEGER, time REAL, payload TEXT,
                PRIMARY KEY (series, ordinal)) WITHOUT ROWID;
        """)
        conn.execute("INSERT INTO catalog VALUES (?,?)", (dumps(catalog(data)), uuid.uuid4().hex))
        for profile in data["profiles"]:
            pid = profile["id"]
            conn.executemany(
                "INSERT INTO events VALUES (?,?,?,?,?,?,?,?)",
                ((pid, i, duration_bucket(e[0], e[1]), *e) for i, e in enumerate(profile["events"])),
            )
            conn.executemany(
                "INSERT INTO samples VALUES (?,?,?,?)",
                ((pid, i, s[0], dumps(s)) for i, s in enumerate((profile.get("cpu") or {}).get("samples", []))),
            )
        for series in data["metrics"]:
            conn.executemany(
                "INSERT INTO points VALUES (?,?,?,?)",
                ((series["id"], i, p[0], dumps(p)) for i, p in enumerate(series["points"])),
            )
        conn.executescript("""
            INSERT INTO durations SELECT profile,bucket,max(end-start) FROM events GROUP BY profile,bucket;
            CREATE INDEX event_window ON events (profile,bucket,start);
            CREATE INDEX sample_window ON samples (profile,time);
            CREATE INDEX point_window ON points (series,time);
        """)
        conn.execute(f"PRAGMA user_version={VERSION}")


class TraceStore:
    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        with closing(self.connect()) as conn:
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            if version != VERSION:
                raise ValueError(f"Unsupported DSight SQLite version: {version}; expected {VERSION}")
            raw, self.generation = conn.execute("SELECT json,generation FROM catalog").fetchone()
            self.data = json.loads(raw)

    def connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True)

    def query(
        self,
        kind: str,
        *,
        lo: float,
        hi: float,
        worker: str | None,
        rank: int | None,
        profile: int | None,
        name: str | None,
        offset: int,
        limit: int,
        points: bool,
    ) -> dict[str, Any]:
        def page(total: int, items: list[dict[str, Any]], **extra: Any) -> dict[str, Any]:
            return dict(total=total, items=items, offset=offset, limit=limit, range=[lo, hi], **extra)

        profiles = [
            p
            for p in self.data["profiles"]
            if (not worker or p["worker"] == worker)
            and (rank is None or p["rank"] == rank)
            and (profile is None or p["id"] == profile)
        ]
        if kind == "profiles":
            rows = [
                {k: v for k, v in p.items() if k not in ("events", "names", "cpu")}
                | {"cpu_samples": (p.get("cpu") or {}).get("sample_count", 0)}
                for p in profiles
            ]
            return page(len(rows), rows[offset : offset + limit])
        with closing(self.connect()) as conn:
            if conn.execute("SELECT generation FROM catalog").fetchone()[0] != self.generation:
                raise ValueError("DSight dataset was rebuilt; reopen it before querying")
            if kind == "metrics":
                series = [
                    s
                    for s in self.data["metrics"]
                    if (not worker or s["worker"] == worker)
                    and (not name or s["name"] == name)
                    and (rank is None or str(s["rank"]) == str(rank))
                ]
                rows = []
                for s in series[offset : offset + limit]:
                    n = total = 0
                    low = high = last = None
                    summed = 0.0
                    selected = []
                    for (payload,) in conn.execute(
                        "SELECT payload FROM points WHERE series=? AND time>=? AND time<=? ORDER BY ordinal",
                        (s["id"], lo, hi),
                    ):
                        point = json.loads(payload)
                        total += 1
                        if points and len(selected) < 1000:
                            selected.append(point)
                        value = point[1]
                        if value is not None:
                            n += 1
                            low = value if low is None else min(low, value)
                            high = value if high is None else max(high, value)
                            summed += value
                            last = value
                    item = {k: v for k, v in s.items() if k != "points"}
                    item.update(samples=n, min=low, max=high, mean=summed / n if n else None, last=last)
                    if s.get("temporal") == "setting":
                        item["carried_setting"] = [
                            json.loads(r[0])
                            for r in conn.execute(
                                "SELECT payload FROM points WHERE series=? AND time=(SELECT max(time) FROM points "
                                "WHERE series=? AND time<?) ORDER BY ordinal",
                                (s["id"], s["id"], lo),
                            )
                        ]
                    if points:
                        item.update(points=selected, points_total=total, points_truncated=total > 1000)
                    rows.append(item)
                return page(len(series), rows)
            if kind == "cpu":
                hotspots: dict[str, int] = {}
                count = 0
                for p in profiles:
                    cpu = p.get("cpu")
                    if not cpu:
                        continue
                    for (payload,) in conn.execute(
                        "SELECT payload FROM samples WHERE profile=? AND time>=? AND time<=?",
                        (p["id"], lo, hi),
                    ):
                        sample = json.loads(payload)
                        count += 1
                        for symbol in {cpu["names"][i] for i in cpu["stacks"][sample[2]]}:
                            hotspots[symbol] = hotspots.get(symbol, 0) + 1
                ordered = sorted(hotspots.items(), key=lambda x: (-x[1], x[0]))
                return page(
                    len(ordered),
                    [{"symbol": k, "samples": n, "fraction": n / count} for k, n in ordered[offset : offset + limit]],
                    total_samples=count,
                    attribution="Inclusive process samples; not per-request CPU time.",
                )
            # Each duration class has a measured maximum. Both start bounds use
            # the index, followed by an exact inclusive-overlap check. nextafter
            # protects a boundary from cancellation in the subtraction.
            conn.execute("CREATE TEMP TABLE bounds (profile INTEGER, bucket INTEGER, lower REAL)")
            conn.execute("CREATE TEMP TABLE names (profile INTEGER, name INTEGER, PRIMARY KEY(profile,name))")
            by_id = {p["id"]: p for p in profiles}
            for p in profiles:
                if name:
                    conn.executemany(
                        "INSERT INTO names VALUES (?,?)",
                        ((p["id"], i) for i, n in enumerate(p["names"]) if name.lower() in n.lower()),
                    )
                for bucket, maximum in conn.execute("SELECT bucket,maximum FROM durations WHERE profile=?", (p["id"],)):
                    conn.execute(
                        "INSERT INTO bounds VALUES (?,?,?)",
                        (p["id"], bucket, math.nextafter(lo - math.nextafter(maximum, math.inf), -math.inf)),
                    )
            rows, total = [], 0
            if profiles:
                selection = (
                    "SELECT e.* FROM bounds b CROSS JOIN events e INDEXED BY event_window "
                    "WHERE e.profile=b.profile AND e.bucket=b.bucket AND e.start>=b.lower "
                    "AND e.start<=? AND e.end>=?"
                )
                if name:
                    selection += " AND EXISTS (SELECT 1 FROM names n WHERE n.profile=e.profile AND n.name=e.name)"
                args = [hi, lo]
                total = conn.execute("SELECT count(*) FROM (" + selection + ")", args).fetchone()[0]
                for pid, _, _, a, b, ni, tid, rowid in conn.execute(
                    selection + " ORDER BY e.start,e.profile,e.source_row LIMIT ? OFFSET ?", [*args, limit, offset]
                ):
                    p = by_id[pid]
                    rows.append(
                        {
                            "profile": pid,
                            "worker": p["worker"],
                            "rank": p["rank"],
                            "start": a,
                            "end": b,
                            "name": p["names"][ni],
                            "global_tid": tid,
                            "pid": ((int(tid) >> 24) & 0xFFFFFF) if tid.isdigit() else None,
                            "tid": (int(tid) & 0xFFFFFF) if tid.isdigit() else None,
                            "definition": p["name_definitions"][ni] if p.get("name_definitions") else None,
                            "rowid": rowid,
                            "evidence_source": p["evidence_source"],
                        }
                    )
            return page(
                total,
                rows,
                attribution="Shared worker/rank activity; overlap does not establish request ownership.",
                partial=any(p["truncated"] for p in profiles),
            )
