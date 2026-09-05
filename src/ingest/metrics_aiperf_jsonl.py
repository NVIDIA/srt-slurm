# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""L2 processor: AIPerf per-scrape ``server_metrics_export.jsonl`` -> schema 2.

When the benchmark passes ``--server-metrics-formats json jsonl``, AIPerf writes -- in
addition to the aggregate/timesliced JSON that :mod:`.metrics_aiperf_json` reads -- one
JSON line per (scrape, endpoint), each carrying the endpoint's metric families at that
instant (observed schema, from a real 1.9 GB export)::

    {"timestamp_ns": <int>, "endpoint_url": <str>, "endpoint_latency_ns": <int>,
     "request_sent_ns": <int>, "first_byte_ns": <int>,
     "metrics": {"<family>": [{"labels": {...}, "value": <num>}, ...
                              {"labels": {...}, "buckets": {"<le>": <num>, ...},
                               "sum": <num>, "count": <num>}]}}

Values are the raw scraped readings -- counters stay cumulative, histogram buckets stay
cumulative (the "+Inf" bucket equals ``count``) -- so unlike the aggregate export there
is NO cumulative/interval detection to do: everything is written through verbatim, the
same way :mod:`.metrics_prometheus` treats a raw exposition body.

NAME SUFFIXES (same interchange contract as :mod:`.metrics_aiperf_json`): AIPerf keys a
COUNTER family by its base name (``dynamo_component_router_requests_total`` arrives as
``dynamo_component_router_requests``; verified against the same run's
``raw_prometheus.jsonl``) while gauges keep their exposition names verbatim. The jsonl
carries no type information, so every plain-valued family that does not already end in
``_total`` is emitted under BOTH its own name and the ``_total`` alias -- a panel naming
either form resolves, and the alias on a gauge is inert (no panel reads a name that was
never exposed). Histogram entries expand to exposition components:
``<family>_bucket`` (with the ``le`` label, cumulative), ``<family>_sum``,
``<family>_count``.

No ``worker_id``/``dynamo_component`` labels are invented: the endpoint's role is not
knowable from the export alone, mirroring :mod:`.metrics_aiperf_json`.

Timestamps are snapped to whole seconds (``metrics_aiperf_json``'s slice snapping) and
the per-endpoint lines sharing a second merge into one schema-2 line, deduped with the
same idempotent (labels, value) fold. The export can exceed a GB, so lines are streamed
with a small reorder window (endpoints are scraped concurrently, so their lines can
interleave by milliseconds): a buffered second is flushed once every line seen is
``_FLUSH_LAG_S`` seconds past it. A pathologically late line is emitted as its own
(duplicate-timestamp) line rather than dropped, and counted in the log.

Stdlib only (runs under a bare cluster python3).

Usage:
    python3 -m src.ingest.metrics_aiperf_jsonl RAW.jsonl OUT.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

if __package__ in (None, ""):  # pragma: no cover - only on the bare-script path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.ingest.metrics_prometheus import _dedup  # noqa: E402  (shared idempotent fold)

logger = logging.getLogger("metrics_aiperf_jsonl")

# Reorder window, in seconds. Endpoint scrape skew is milliseconds; a line arriving
# this far behind the newest one seen means the writer itself was out of order.
_FLUSH_LAG_S = 5


def _snap(ts_ns: int) -> int:
    """Epoch ns -> whole-second epoch ns (same grid as metrics_aiperf_json)."""
    return int(round(ts_ns / 1e9)) * 1_000_000_000


def _finite(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _emit_line(rec: dict, merged: dict) -> int:
    """Expand one raw line's families into ``merged`` (schema-2 metrics dict)."""
    n = 0
    for family, entries in (rec.get("metrics") or {}).items():
        for entry in entries or []:
            if not isinstance(entry, dict):
                continue
            labels = dict(entry.get("labels") or {})
            if "buckets" in entry:
                for le, cnt in (entry.get("buckets") or {}).items():
                    if not _finite(cnt):
                        continue
                    blabels = dict(labels)
                    blabels["le"] = str(le)
                    merged.setdefault(f"{family}_bucket", []).append(
                        {"labels": blabels, "value": float(cnt)}
                    )
                    n += 1
                for f, suffix in (("sum", "_sum"), ("count", "_count")):
                    v = entry.get(f)
                    if _finite(v):
                        merged.setdefault(f"{family}{suffix}", []).append(
                            {"labels": dict(labels), "value": float(v)}
                        )
                        n += 1
            else:
                v = entry.get("value")
                if not _finite(v):
                    continue
                # Counter families arrive with _total stripped; the alias makes a
                # panel naming either form resolve (see the module docstring).
                out_names = [family] if family.endswith("_total") else [family, f"{family}_total"]
                for out_name in out_names:
                    merged.setdefault(out_name, []).append(
                        {"labels": dict(labels), "value": float(v)}
                    )
                    n += 1
    return n


def process(raw_path: str, out_path: str) -> int:
    """Convert AIPerf's per-scrape jsonl to schema 2. Returns lines written."""
    pending: dict[int, dict] = {}
    lines_written = samples = late = 0
    max_ts = None

    with open(raw_path) as f, open(out_path, "w") as out:

        def flush_older_than(limit_ns):
            nonlocal lines_written
            for ts in sorted(pending):
                if limit_ns is not None and ts >= limit_ns:
                    break
                metrics = pending.pop(ts)
                for name in metrics:
                    metrics[name] = _dedup(metrics[name])
                out.write(json.dumps({"timestamp_ns": ts, "metrics": metrics}) + "\n")
                lines_written += 1

        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            ts = _snap(rec["timestamp_ns"])
            if max_ts is not None and ts < max_ts - _FLUSH_LAG_S * 1_000_000_000:
                # Its second may already be flushed; it becomes its own line below.
                late += 1
            merged = pending.setdefault(ts, {})
            samples += _emit_line(rec, merged)
            if max_ts is None or ts > max_ts:
                max_ts = ts
            flush_older_than(max_ts - _FLUSH_LAG_S * 1_000_000_000)
        flush_older_than(None)

    logger.info(
        "read %d samples -> %d timestamps%s",
        samples, lines_written, f" ({late} late line(s) past the reorder window)" if late else "",
    )
    return lines_written


def main(argv=None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("raw_path", help="input AIPerf per-scrape server_metrics_export.jsonl")
    ap.add_argument("out_path", help="output server_metrics_export.jsonl (schema 2)")
    args = ap.parse_args(argv)
    n = process(args.raw_path, args.out_path)
    logger.info("wrote %s: %d timestamps", args.out_path, n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
