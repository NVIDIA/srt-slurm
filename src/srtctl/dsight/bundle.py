# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Immutable exact detail shards for ordinary static HTTP hosts.

SQLite serves local queries; these shards project the same normalized rows for
the browser without requiring a database runtime, Range requests or a service.
"""

from __future__ import annotations

import gzip
import hashlib
import itertools
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .storage import catalog, dumps

MAX_CHUNK_BYTES = 512 * 1024
MAX_CHUNK_ROWS = 8192
DENSITY_BINS = 512


def _chunks(rows: Iterable[list[Any]], folder: Path, *, intervals: bool = False) -> list[dict[str, Any]]:
    result = []
    encoded: list[str] = []
    batch: list[list[Any]] = []
    size = 2

    def flush() -> None:
        if not batch:
            return
        raw = ("[" + ",".join(encoded) + "]").encode()
        compressed = gzip.compress(raw, compresslevel=6, mtime=0)
        digest = hashlib.sha256(compressed).hexdigest()
        (folder / (digest + ".json.gz")).write_bytes(compressed)
        starts = [r[0] for r in batch]
        ends = [r[1] if intervals else r[0] for r in batch]
        result.append(
            {
                "url": f"detail/{digest}.json.gz",
                "sha256": digest,
                "decoded_sha256": hashlib.sha256(raw).hexdigest(),
                "bytes": len(compressed),
                "decoded_bytes": len(raw),
                "count": len(batch),
                "bounds": [min(starts), min(ends), max(starts), max(ends)],
            }
        )

    for row in rows:
        value = dumps(row)
        cost = len(value.encode()) + 1
        if batch and (size + cost > MAX_CHUNK_BYTES or len(batch) == MAX_CHUNK_ROWS):
            flush()
            batch, encoded, size = [], [], 2
        batch.append(row)
        encoded.append(value)
        size += cost
    flush()
    return result


def write_details(data: dict[str, Any], folder: Path) -> dict[str, Any]:
    folder.mkdir()
    core = catalog(data)
    core["delivery"] = {"version": 1, "kind": "static", "cache_bytes": 32 * 1024 * 1024}
    for source, target in zip(data["profiles"], core["profiles"], strict=True):
        delta = [0] * (DENSITY_BINS + 1)
        scale = DENSITY_BINS / data["meta"]["duration"]
        for event in source["events"]:
            if event[0] <= data["meta"]["duration"] and event[1] >= 0:
                a = max(0, min(DENSITY_BINS - 1, int(event[0] * scale)))
                b = max(0, min(DENSITY_BINS - 1, int(event[1] * scale)))
                delta[a] += 1
                delta[b + 1] -= 1
        # Preserve source order for pagination. min(start)/max(end) select every
        # crossing interval, including roots that began many chunks earlier.
        target["event_chunks"] = _chunks(source["events"], folder, intervals=True)
        target["event_density"] = list(itertools.accumulate(delta[:-1]))
        if source.get("cpu"):
            target["cpu"]["chunks"] = _chunks(source["cpu"]["samples"], folder)
    for source, target in zip(data["metrics"], core["metrics"], strict=True):
        target.pop("points")
        target["chunks"] = _chunks(source["points"], folder)
    return core
