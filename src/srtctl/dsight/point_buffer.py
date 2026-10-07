# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build-only, disk-backed metric buffers; published reports retain schema 1."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any


class PointBuffer:
    def __init__(self, store: PointBuffers, series: int) -> None:
        self.store = store
        self.series = series
        self.count = 0
        self.finalized = False

    def __len__(self) -> int:
        return self.count

    def extend(self, points: Iterable[list[Any]]) -> None:
        def rows() -> Iterator[tuple]:
            for point in points:
                self.count += 1
                time, value, source, row = point
                # Arrow uint64 samples can exceed SQLite's signed integer range.
                if isinstance(value, int) and not -(2**63) <= value < 2**63:
                    value = str(value)
                yield (self.series, time, value, source, row)

        self.store.conn.executemany("INSERT INTO raw VALUES (?,?,?,?,?)", rows())

    def __iter__(self) -> Iterator[list[Any]]:
        if self.finalized:
            query = "SELECT time,value,source,row FROM retained WHERE series=? ORDER BY ordinal"
        else:
            query = "SELECT time,value,source,row FROM raw WHERE series=? ORDER BY time,source,row"
        for row in self.store.conn.execute(query, (self.series,)):
            point = list(row)
            if isinstance(point[1], str):
                point[1] = int(point[1])
            yield point

    def finalize(self) -> tuple[int, list[float], int]:
        """Keep the first source-backed occurrence of each time/value identity.

        Sorted input lets the seen set expire at each timestamp, instead of
        retaining a tuple for every sample in a complete series.
        """
        original = self.count
        self.count = 0
        conflicts: list[float] = []
        conflicting_samples = 0
        timestamp = float("nan")
        seen: set[float] = set()

        def rows() -> Iterator[tuple]:
            nonlocal timestamp, conflicting_samples
            for point in self:
                time, value, source, row = point
                if time != timestamp:
                    if len(seen) > 1:
                        conflicts.append(timestamp)
                        conflicting_samples += len(seen)
                    timestamp = time
                    seen.clear()
                if value in seen:
                    continue
                seen.add(value)
                ordinal = self.count
                self.count += 1
                stored = str(value) if isinstance(value, int) and not -(2**63) <= value < 2**63 else value
                yield (self.series, ordinal, time, stored, source, row)
            if len(seen) > 1:
                conflicts.append(timestamp)
                conflicting_samples += len(seen)

        self.store.conn.executemany("INSERT INTO retained VALUES (?,?,?,?,?,?)", rows())
        self.finalized = True
        return original - self.count, conflicts, conflicting_samples


class PointBuffers:
    def __init__(self, path: Path) -> None:
        self.conn = sqlite3.connect(path)
        self.conn.executescript("""
            PRAGMA journal_mode=OFF;
            PRAGMA synchronous=OFF;
            PRAGMA temp_store=FILE;
            PRAGMA cache_size=-8192;
            CREATE TABLE raw (series INTEGER,time REAL,value,source INTEGER,row INTEGER);
            CREATE TABLE retained (series INTEGER,ordinal INTEGER,time REAL,value,source INTEGER,row INTEGER,
                PRIMARY KEY(series,ordinal)) WITHOUT ROWID;
        """)

    def buffer(self, series: int) -> PointBuffer:
        return PointBuffer(self, series)

    def prepare(self) -> None:
        self.conn.execute("CREATE INDEX raw_order ON raw(series,time,source,row)")

    def release_raw(self) -> None:
        self.conn.execute("DROP TABLE raw")
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()
