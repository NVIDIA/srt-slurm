# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded read-only queries over the exact dataset embedded in the HTML."""

from __future__ import annotations

import gzip
import json
import math
from pathlib import Path
from typing import Any

from .model import SCHEMA

KINDS = ("summary", "requests", "request", "lifecycle", "metrics", "profiles", "nsys", "cpu", "iterations", "sources")


class TraceDataset:
    def __init__(self, data: dict[str, Any]) -> None:
        if data.get("schema") != SCHEMA:
            raise ValueError(f"Unsupported trace schema: {data.get('schema')!r}; expected {SCHEMA}")
        if not math.isfinite(data["meta"]["duration"]) or data["meta"]["duration"] <= 0:
            raise ValueError("Dataset has no positive finite duration")
        self.data = data
        self.requests = {r["id"]: r for r in data["requests"]}
        if len(self.requests) != len(data["requests"]):
            raise ValueError("Duplicate client request identities in dataset")

    @classmethod
    def from_path(cls, path: Path | str) -> TraceDataset:
        path = Path(path)
        if path.is_dir():
            path /= "trace-data.json.gz"
        raw = path.read_bytes()
        return cls(json.loads(gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw))

    def query(
        self,
        kind: str = "summary",
        *,
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
        data = self.data
        lo = 0 if start is None else start
        hi = data["meta"]["duration"] if end is None else end
        if not all(math.isfinite(v) for v in (lo, hi)) or not 0 <= lo < hi <= data["meta"]["duration"]:
            raise ValueError("Require 0 <= start < end <= run duration (seconds from origin_ns)")
        if not isinstance(offset, int) or not isinstance(limit, int) or offset < 0 or not 0 <= limit <= 1000:
            raise ValueError("Require integer offset >= 0 and 0 <= limit <= 1000")
        if not math.isfinite(min_ttft_ms) or min_ttft_ms < 0:
            raise ValueError("min_ttft_ms must be finite and nonnegative")
        if kind not in KINDS:
            raise ValueError(f"Unknown query kind {kind!r}; choose from {KINDS}")

        def overlaps(a: float, b: float) -> bool:
            return a <= hi and b >= lo

        def page(rows: list[dict[str, Any]], **extra: Any) -> dict[str, Any]:
            return {
                "total": len(rows),
                "offset": offset,
                "limit": limit,
                "range": [lo, hi],
                "items": rows[offset : offset + limit],
                **extra,
            }

        if kind == "summary":
            return {
                "schema": data["schema"],
                "meta": data["meta"],
                "audit": data["audit"],
                "counts": {
                    key: len(data[key])
                    for key in ("requests", "sessions", "workers", "metrics", "profiles", "iterations")
                },
            }
        if kind in ("request", "lifecycle"):
            if request_id not in self.requests:
                raise ValueError(f"Unknown client request: {request_id}")
            request = self.requests[request_id]
            return request if kind == "request" else request["lifecycle"]
        if kind == "requests":
            result = []
            for request in data["requests"]:
                if not overlaps(request["start"], request["end"]):
                    continue
                if (session and request["session"] != session) or (agent and request["agent"] != agent):
                    continue
                if worker and worker not in request["workers"]:
                    continue
                if (request["ttft_ms"] or 0) < min_ttft_ms:
                    continue
                identities = [
                    request["id"],
                    request["session"],
                    request["agent"],
                    request["conversation"],
                    *request["server_ids"],
                    *request["workers"],
                ]
                if search and not any(search.lower() in str(x or "").lower() for x in identities):
                    continue
                result.append(
                    {k: v for k, v in request.items() if k not in ("spans", "lifecycle")}
                    | {"span_count": len(request["spans"])}
                )
            return page(result)
        if kind == "sources":
            return page(data["sources"])
        if kind == "metrics":
            result = []
            for series in data["metrics"]:
                if (worker and series["worker"] != worker) or (name and series["name"] != name):
                    continue
                if rank is not None and str(series["rank"]) != str(rank):
                    continue
                selected = [p for p in series["points"] if lo <= p[0] <= hi]
                values = [p[1] for p in selected]
                item = {k: v for k, v in series.items() if k != "points"}
                item.update(
                    samples=len(values),
                    min=min(values) if values else None,
                    max=max(values) if values else None,
                    mean=sum(values) / len(values) if values else None,
                    last=values[-1] if values else None,
                )
                if points:
                    item.update(
                        points=selected[:1000], points_total=len(selected), points_truncated=len(selected) > 1000
                    )
                result.append(item)
            return page(result)
        profiles = [
            p
            for p in data["profiles"]
            if (not worker or p["worker"] == worker)
            and (rank is None or p["rank"] == rank)
            and (profile is None or p["id"] == profile)
        ]
        if kind == "profiles":
            return page(
                [
                    {k: v for k, v in p.items() if k not in ("events", "names", "cpu")}
                    | {"event_count": len(p["events"]), "cpu_samples": len((p.get("cpu") or {}).get("samples", []))}
                    for p in profiles
                ]
            )
        if kind == "iterations":
            rows = [
                r
                for r in data["iterations"]
                if (not worker or r["worker"] == worker) and (rank is None or r["global_rank"] == rank)
            ]
            return page(
                [r for r in rows if r["start"] is not None and overlaps(r["start"], r["end"])],
                unaligned_rows=sum(r["start"] is None for r in rows),
                attribution="Shared batches, one-second timestamps. Counters and timers are not mapped to requests or NVTX iterations.",
            )
        if kind == "nsys":
            events = []
            for p in profiles:
                for event in p["events"]:
                    if overlaps(event[0], event[1]) and (not name or name.lower() in p["names"][event[2]].lower()):
                        events.append(
                            {
                                "profile": p["id"],
                                "worker": p["worker"],
                                "rank": p["rank"],
                                "start": event[0],
                                "end": event[1],
                                "name": p["names"][event[2]],
                                "global_tid": event[3],
                                "rowid": event[4],
                                "evidence_source": p["evidence_source"],
                            }
                        )
            events.sort(key=lambda e: (e["start"], e["profile"], e["rowid"]))
            return page(
                events,
                attribution="Shared worker/rank activity; overlap does not establish request ownership.",
                partial=any(p["truncated"] for p in profiles),
            )
        # CPU callchains are inclusive samples, not time charged to a request.
        hotspots: dict[str, int] = {}
        count = 0
        for p in profiles:
            cpu = p.get("cpu")
            if not cpu:
                continue
            for sample in cpu["samples"]:
                if lo <= sample[0] <= hi:
                    count += 1
                    for symbol in {cpu["names"][i] for i in cpu["stacks"][sample[2]]}:
                        hotspots[symbol] = hotspots.get(symbol, 0) + 1
        return page(
            [
                {"symbol": key, "samples": value, "fraction": value / count}
                for key, value in sorted(hotspots.items(), key=lambda x: (-x[1], x[0]))
            ],
            total_samples=count,
            attribution="Inclusive process samples; not per-request CPU time.",
        )
