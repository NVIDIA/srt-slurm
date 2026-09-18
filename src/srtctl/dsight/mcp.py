# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read a CLI-generated trace dataset without invoking Slurm or a profiler."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from srtctl.dsight.query import TraceDataset


@lru_cache(maxsize=1)
def _load(path: str, mtime_ns: int, size: int, inode: int) -> TraceDataset:
    # Stat identity invalidates the cache after an atomic CLI rebuild.
    return TraceDataset.from_path(path)


def _query_trace(dataset: str, kind: str = "summary", **filters: Any) -> dict[str, Any]:
    try:
        path = Path(dataset).expanduser().resolve()
        if path.is_dir():
            path /= "trace-data.json.gz"
        stat = path.stat()
        return _load(str(path), stat.st_mtime_ns, stat.st_size, stat.st_ino).query(kind, **filters)
    except (OSError, ValueError, KeyError) as exc:
        return {"ok": False, "error": str(exc)}


def query_trace(
    dataset: str,
    kind: str = "summary",
    start: float | None = None,
    end: float | None = None,
    request_id: str | None = None,
    session: str | None = None,
    agent: str | None = None,
    worker: str | None = None,
    rank: int | None = None,
    profile: int | None = None,
    name: str | None = None,
    search: str | None = None,
    min_ttft_ms: float = 0,
    offset: int = 0,
    limit: int = 100,
    points: bool = False,
) -> dict[str, Any]:
    """Read a CLI-generated trace dashboard; never start profiling or a benchmark.

    dataset is a local dashboard directory or trace-data.json.gz. kind is summary,
    requests, request, lifecycle, metrics, profiles, nsys, cpu, iterations, or
    sources. start/end are seconds relative to summary.meta.origin_ns.
    request/lifecycle require request_id. All list queries have offset/limit
    (maximum 1000). Rank selects the Nsight/iteration global rank, not router DP
    rank. Runtime spans and shared batch activity are not exclusive request costs.
    Generate the dataset explicitly with srtctl dsight build before querying.
    """

    return _query_trace(
        dataset,
        kind,
        start=start,
        end=end,
        request_id=request_id,
        session=session,
        agent=agent,
        worker=worker,
        rank=rank,
        profile=profile,
        name=name,
        search=search,
        min_ttft_ms=min_ttft_ms,
        offset=offset,
        limit=limit,
        points=points,
    )


def register(server: Any) -> None:
    server.tool(annotations={"readOnlyHint": True, "destructiveHint": False})(query_trace)
